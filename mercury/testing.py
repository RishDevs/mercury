"""A runner that produces deterministic tokens without a model.

This is the seam the whole project is built around. `execute(plan) -> ExecResult` is the
entire contract between the scheduler and the GPU, so swapping in this implementation
makes the allocator, every policy, preemption, streaming and cancellation testable on a
laptop, and lets the serving stack and the benchmark client be exercised end to end
without weights. It also checks the one thing the real runner silently depends on: that
the blocks it was handed actually cover the sequence it was asked to run.
"""

from __future__ import annotations

import time
import zlib

from .runner import ExecResult
from .scheduler import Plan


class DeterministicRunner:
    def __init__(self, vocab_size: int = 32000, block_size: int = 16, step_delay: float = 0.0):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.step_delay = step_delay
        self.model_config = None
        self.model_path = None
        self.revision = None
        self.kv_bytes = 0
        self.steps = 0
        self.prefill_tokens_seen = 0
        self.decode_tokens_seen = 0

    def token_for(self, rid: str, index: int) -> int:
        return (zlib.crc32(rid.encode()) + index * 7919) % self.vocab_size

    def _check(self, requests) -> None:
        for r in requests:
            covered = len(r.blocks) * self.block_size
            if covered < r.seq_len:
                raise AssertionError(
                    f"{r.rid}: {len(r.blocks)} blocks cover {covered} tokens "
                    f"but seq_len is {r.seq_len}"
                )
            if len(set(r.blocks)) != len(r.blocks):
                raise AssertionError(f"{r.rid}: the same block appears twice in {r.blocks}")

    def execute(self, plan: Plan) -> ExecResult:
        started = time.perf_counter()
        self._check(plan.prefill)
        self._check(plan.decode)
        self.steps += 1
        self.prefill_tokens_seen += plan.num_prefill_tokens
        self.decode_tokens_seen += len(plan.decode)
        if self.step_delay:
            time.sleep(self.step_delay)
        return ExecResult(
            prefill_tokens=[self.token_for(r.rid, len(r.out)) for r in plan.prefill],
            decode_tokens=[self.token_for(r.rid, len(r.out)) for r in plan.decode],
            prepare_seconds=0.0,
            forward_seconds=time.perf_counter() - started,
        )

    def snapshot(self) -> dict:
        return {"runner": "deterministic", "steps": self.steps, "device": "none"}


class FakeClock:
    """A clock the caller advances, so scheduling tests never depend on wall time."""

    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += seconds
        return self.now
