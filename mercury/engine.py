from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable

from .allocator import BlockAllocator
from .config import EngineConfig, ModelConfig
from .hashing import block_hashes, extend_hashes, use_cryptographic_hashes
from .metrics import EngineMetrics, StepRecord
from .request import Request, RequestState, SamplingParams
from .scheduler import Scheduler


@dataclass(slots=True)
class StreamEvent:
    rid: str
    kind: str
    token: int | None = None
    index: int = 0
    finish_reason: str | None = None
    message: str | None = None
    t: float = 0.0


Sink = Callable[[StreamEvent], None]


class Engine:
    """Owns the GPU. One thread, one loop: drain commands, schedule, execute, route output.

    Everything that touches the KV cache happens on this thread. The HTTP layer talks to
    it through a queue and gets tokens back through per-request sink callbacks, which are
    responsible for hopping onto whatever loop they came from.
    """

    def __init__(
        self,
        cfg: EngineConfig,
        runner=None,
        model_config: ModelConfig | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self.cfg = cfg
        self.clock = clock
        if cfg.cryptographic_hashes:
            use_cryptographic_hashes(True)

        self.alloc = BlockAllocator(cfg.num_blocks, cfg.block_size)
        self.sched = Scheduler(
            self.alloc,
            policy=cfg.policy,
            max_running=cfg.max_running,
            prefill_budget=cfg.prefill_budget,
            aging_tokens_per_s=cfg.aging_tokens_per_s,
            max_wait_s=cfg.max_wait_s,
            scan_limit=cfg.scan_limit,
        )
        self.metrics = EngineMetrics(max_steps=cfg.max_step_records)

        if runner is None:
            from .runner import ModelRunner

            runner = ModelRunner(cfg, alloc=self.alloc)
        self.runner = runner
        self.model_config = model_config or getattr(runner, "model_config", None)
        self.eos_ids = set(getattr(self.model_config, "eos_token_ids", ()) or ())

        self._commands: queue.Queue = queue.Queue()
        self._sinks: dict[str, Sink] = {}
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._submitted = 0

    # ---------------------------------------------------------------- lifecycle

    def start(self) -> None:
        if self._thread is not None:
            return
        self.metrics.started_at = self.clock()
        self._thread = threading.Thread(target=self._run, name="mercury-engine", daemon=True)
        self._thread.start()

    def shutdown(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            record = self.step()
            if record is None:
                self._wake.wait(self.cfg.idle_sleep_s)
                self._wake.clear()

    # ---------------------------------------------------------------- client API

    def submit(
        self,
        rid: str,
        prompt_ids: list[int],
        max_tokens: int,
        sink: Sink,
        params: SamplingParams | None = None,
        session_id: str | None = None,
        arrival: float | None = None,
    ) -> None:
        params = params or SamplingParams()
        arrival = self.clock() if arrival is None else arrival
        ceiling = min(self.cfg.max_model_len, self.alloc.num_blocks * self.cfg.block_size)
        if not prompt_ids:
            self._reject(rid, sink, "prompt must contain at least one token")
            return
        if len(prompt_ids) + max_tokens > ceiling:
            self._reject(
                rid,
                sink,
                f"prompt ({len(prompt_ids)}) + max_tokens ({max_tokens}) exceeds the "
                f"{ceiling}-token ceiling of this server",
            )
            return

        r = Request(
            rid=rid,
            prompt=list(prompt_ids),
            max_tokens=max_tokens,
            arrival=arrival,
            hashes=block_hashes(prompt_ids, self.cfg.block_size),
            params=params,
            session_id=session_id,
        )
        with self._lock:
            self._sinks[rid] = sink
            self._submitted += 1
        self._commands.put(("add", r))
        self._wake.set()

    def abort(self, rid: str) -> None:
        self._commands.put(("abort", rid))
        self._wake.set()

    def _reject(self, rid: str, sink: Sink, message: str) -> None:
        self.metrics.counters["rejected_requests"] += 1
        sink(StreamEvent(rid=rid, kind="error", message=message, t=self.clock()))

    # ---------------------------------------------------------------- the loop

    def step(self) -> StepRecord | None:
        t_start = time.perf_counter()
        self._drain_commands()
        plan = self.sched.schedule(self.clock())
        if not plan:
            self.metrics.counters["idle_steps"] += 1
            return None

        for r in plan.preempted:
            self._emit(r, StreamEvent(rid=r.rid, kind="preempted", t=self.clock()))

        result = self.runner.execute(plan)
        now = self.clock()

        for r, token in zip(plan.prefill, result.prefill_tokens):
            self._accept(r, token, now)
        for r, token in zip(plan.decode, result.decode_tokens):
            self._accept(r, token, now)

        if self.cfg.debug_invariants:
            self.alloc.check_invariants()
            self._check_running_blocks()

        record = StepRecord(
            t=now,
            num_prefill=len(plan.prefill),
            num_decode=len(plan.decode),
            prefill_tokens=plan.num_prefill_tokens,
            sched_s=plan.sched_seconds,
            prepare_s=result.prepare_seconds,
            forward_s=result.forward_seconds,
            step_s=time.perf_counter() - t_start,
            waiting=len(self.sched.waiting),
            running=len(self.sched.running),
            free_blocks=self.alloc.num_available(),
            preempted=len(plan.preempted),
        )
        self.metrics.add_step(record)
        return record

    def _drain_commands(self) -> None:
        while True:
            try:
                kind, payload = self._commands.get_nowait()
            except queue.Empty:
                return
            if kind == "add":
                self.sched.add(payload)
            elif kind == "abort":
                r = self.sched.abort(payload)
                if r is not None:
                    self.metrics.counters["aborted_requests"] += 1
                    r.finished_at = self.clock()
                    self.metrics.add_finished(r.record())
                    self._emit(
                        r,
                        StreamEvent(
                            rid=r.rid, kind="done", finish_reason="abort", t=r.finished_at
                        ),
                        close=True,
                    )
                else:
                    with self._lock:
                        self._sinks.pop(payload, None)

    def _accept(self, r: Request, token: int, now: float) -> None:
        self._commit_full_blocks(r)
        index = len(r.out)
        r.out.append(int(token))
        if r.first_token_at is None:
            r.first_token_at = now
        self._emit(r, StreamEvent(rid=r.rid, kind="token", token=int(token), index=index, t=now))

        if len(r.out) >= r.max_tokens:
            self._finish(r, "length", now)
        elif not r.params.ignore_eos and int(token) in self.eos_ids:
            self._finish(r, "stop", now)

    def _commit_full_blocks(self, r: Request) -> None:
        """Called after the runner wrote KV for positions 0..seq_len-1, before the new token."""
        bs = self.cfg.block_size
        full = r.seq_len // bs
        if full <= r.committed:
            return
        extend_hashes(r.hashes, r.tokens, bs)
        for index in range(r.committed, full):
            self.alloc.commit(r.blocks[index], r.hashes[index])
        r.committed = full

    def _finish(self, r: Request, reason: str, now: float) -> None:
        r.state = RequestState.FINISHED
        r.finish_reason = reason
        r.finished_at = now
        self.sched.drop(r)
        self.metrics.counters["finished_requests"] += 1
        self.metrics.add_finished(r.record())
        self._emit(
            r, StreamEvent(rid=r.rid, kind="done", finish_reason=reason, t=now), close=True
        )

    def _emit(self, r: Request, event: StreamEvent, close: bool = False) -> None:
        with self._lock:
            sink = self._sinks.pop(r.rid, None) if close else self._sinks.get(r.rid)
        if sink is None:
            return
        try:
            sink(event)
        except Exception:
            self.abort(r.rid)

    def _check_running_blocks(self) -> None:
        for r in self.sched.running:
            for block in r.blocks:
                if self.alloc.ref[block] < 1:
                    raise AssertionError(
                        f"request {r.rid} holds block {block} with ref {self.alloc.ref[block]}"
                    )

    # ---------------------------------------------------------------- reporting

    def snapshot(self) -> dict:
        return {
            "config": {
                "policy": self.cfg.policy,
                "block_size": self.cfg.block_size,
                "num_blocks": self.cfg.num_blocks,
                "kv_capacity_tokens": self.cfg.kv_capacity_tokens,
                "max_running": self.cfg.max_running,
                "prefill_budget": self.cfg.prefill_budget,
                "aging_tokens_per_s": self.cfg.aging_tokens_per_s,
                "max_wait_s": self.cfg.max_wait_s,
                "scan_limit": self.cfg.scan_limit,
                "fused_rmsnorm": self.cfg.fused_rmsnorm,
                "model": self.cfg.model,
                "revision": self.cfg.revision,
            },
            "allocator": self.alloc.snapshot(),
            "scheduler": self.sched.snapshot(),
            "engine": self.metrics.summary(),
            "submitted": self._submitted,
        }

    def leaked_blocks(self) -> int:
        return self.alloc.num_blocks - self.alloc.num_available()

    def drain_request_records(self) -> list[dict]:
        return self.metrics.drain_finished()
