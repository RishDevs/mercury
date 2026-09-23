from __future__ import annotations

import bisect
import time
from dataclasses import dataclass, field

from .allocator import BlockAllocator
from .hashing import block_hashes
from .request import Request, RequestState


@dataclass
class Plan:
    prefill: list[Request] = field(default_factory=list)
    decode: list[Request] = field(default_factory=list)
    sched_seconds: float = 0.0
    preempted: list[Request] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.prefill or self.decode)

    @property
    def num_prefill_tokens(self) -> int:
        return sum(r.seq_len - r.num_cached for r in self.prefill)


class Scheduler:
    """One pass per engine step: reserve decode room, admit in policy order, emit a plan.

    The three policies differ only in admission order and in what they do when the
    request at the front does not fit.
    """

    def __init__(
        self,
        alloc: BlockAllocator,
        policy: str = "fcfs",
        max_running: int = 64,
        prefill_budget: int = 8192,
        aging_tokens_per_s: float = 2000.0,
        max_wait_s: float = 5.0,
        scan_limit: int = 128,
    ):
        self.alloc = alloc
        self.policy = policy
        self.max_running = max_running
        self.prefill_budget = prefill_budget
        self.aging = aging_tokens_per_s
        self.max_wait_s = max_wait_s
        self.scan_limit = scan_limit
        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self.stats = {
            "steps": 0,
            "sched_seconds": 0.0,
            "preemptions": 0,
            "admissions": 0,
            "max_waiting": 0,
            "scan_limit_exceeded": 0,
            "scored_requests": 0,
        }

    def add(self, r: Request) -> None:
        r.state = RequestState.WAITING
        bisect.insort(self.waiting, r, key=lambda x: x.arrival)

    def abort(self, rid: str) -> Request | None:
        for queue in (self.waiting, self.running):
            for r in queue:
                if r.rid == rid:
                    queue.remove(r)
                    self.alloc.release(r.blocks)
                    r.blocks = []
                    r.state = RequestState.ABORTED
                    r.finish_reason = "abort"
                    return r
        return None

    def drop(self, r: Request) -> None:
        """Finished normally: give the blocks back, keeping committed ones cached."""
        if r in self.running:
            self.running.remove(r)
        elif r in self.waiting:
            self.waiting.remove(r)
        self.alloc.release(r.blocks)
        r.blocks = []

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def _preempt(self, r: Request) -> None:
        """Recompute-style: release the blocks and resume later as a longer prompt."""
        self.running.remove(r)
        self.alloc.release(r.blocks)
        r.blocks = []
        r.num_cached = 0
        r.committed = 0
        r.preemptions += 1
        r.hashes = block_hashes(r.tokens, self.alloc.block_size)
        self.stats["preemptions"] += 1
        self.add(r)

    def _score(self, r: Request, now: float) -> float:
        """Admission priority, high first.

        `aging` is an exchange rate in tokens per second, and it has to be scaled to the
        length of the prefixes being shared: alpha ~ prefix tokens / acceptable wait. Set
        it far above that and one fraction of a second outweighs a complete cache hit, so
        the policy degenerates into FCFS plus enough reordering to break the groups up,
        which is worse than either.
        """
        if self.policy == "fcfs":
            return -r.arrival
        cached = len(self.alloc.cached_prefix_blocks(r.hashes)) * self.alloc.block_size
        if self.policy == "lpm":
            return float(cached)
        waited = now - r.arrival
        if waited >= self.max_wait_s:
            return float("inf")
        return cached + self.aging * waited

    def schedule(self, now: float) -> Plan:
        t0 = time.perf_counter()
        bs = self.alloc.block_size
        preempted: list[Request] = []

        for r in list(self.running):
            while r.state is RequestState.RUNNING and r.seq_len > len(r.blocks) * bs:
                block = self.alloc.append_block()
                if block is not None:
                    r.blocks.append(block)
                    continue
                victim = self.running[-1]
                self._preempt(victim)
                preempted.append(victim)
        decode = list(self.running)

        head = self.waiting[: self.scan_limit]
        tail = self.waiting[self.scan_limit :]
        if tail:
            self.stats["scan_limit_exceeded"] += 1
        self.stats["scored_requests"] += len(head)
        order = sorted(head, key=lambda r: (-self._score(r, now), r.arrival)) + tail

        prefill: list[Request] = []
        budget = self.prefill_budget
        for r in order:
            if len(self.running) + len(prefill) >= self.max_running:
                break
            hit, _ = self.alloc.plan_prompt(r.hashes, r.seq_len)
            new_tokens = r.seq_len - len(hit) * bs
            if prefill and new_tokens > budget:
                break
            got = self.alloc.allocate_prompt(r.hashes, r.seq_len)
            if got is None:
                starving = self.policy == "lpm_aging" and self._score(r, now) == float("inf")
                if self.policy == "fcfs" or starving:
                    break
                continue
            r.blocks, r.num_cached = got
            r.state = RequestState.RUNNING
            r.last_admitted_at = now
            if r.admitted_at is None:
                r.admitted_at = now
            r.recomputed_tokens += r.seq_len - r.num_cached
            budget -= new_tokens
            prefill.append(r)

        if prefill:
            admitted = {id(r) for r in prefill}
            self.waiting = [r for r in self.waiting if id(r) not in admitted]
            self.running.extend(prefill)
            self.stats["admissions"] += len(prefill)

        elapsed = time.perf_counter() - t0
        self.stats["steps"] += 1
        self.stats["sched_seconds"] += elapsed
        self.stats["max_waiting"] = max(self.stats["max_waiting"], len(self.waiting))
        return Plan(prefill=prefill, decode=decode, sched_seconds=elapsed, preempted=preempted)

    def snapshot(self) -> dict:
        steps = self.stats["steps"] or 1
        return {
            "policy": self.policy,
            "waiting": len(self.waiting),
            "running": len(self.running),
            "mean_sched_seconds": self.stats["sched_seconds"] / steps,
            **self.stats,
        }
