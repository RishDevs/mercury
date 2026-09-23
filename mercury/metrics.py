from __future__ import annotations

from collections import deque
from dataclasses import astuple, dataclass, fields


@dataclass(slots=True)
class StepRecord:
    t: float
    num_prefill: int
    num_decode: int
    prefill_tokens: int
    sched_s: float
    prepare_s: float
    forward_s: float
    step_s: float
    waiting: int
    running: int
    free_blocks: int
    preempted: int


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


class EngineMetrics:
    """Per-step timings plus a drainable log of finished requests.

    Percentiles are computed on demand; the engine thread only appends.
    """

    def __init__(self, max_steps: int = 500_000, max_requests: int = 200_000):
        self.steps: deque[StepRecord] = deque(maxlen=max_steps)
        self.finished: deque[dict] = deque(maxlen=max_requests)
        self.counters = {
            "steps": 0,
            "idle_steps": 0,
            "prefill_requests": 0,
            "prefill_tokens": 0,
            "decode_tokens": 0,
            "finished_requests": 0,
            "aborted_requests": 0,
            "rejected_requests": 0,
            "preemptions": 0,
        }
        self.started_at: float | None = None

    def add_step(self, record: StepRecord) -> None:
        self.steps.append(record)
        self.counters["steps"] += 1
        self.counters["prefill_requests"] += record.num_prefill
        self.counters["prefill_tokens"] += record.prefill_tokens
        self.counters["decode_tokens"] += record.num_decode
        self.counters["preemptions"] += record.preempted

    def add_finished(self, record: dict) -> None:
        self.finished.append(record)

    def drain_finished(self) -> list[dict]:
        out = list(self.finished)
        self.finished.clear()
        return out

    def step_columns(self) -> dict[str, list]:
        names = [f.name for f in fields(StepRecord)]
        cols: dict[str, list] = {n: [] for n in names}
        for record in self.steps:
            for name, value in zip(names, astuple(record)):
                cols[name].append(value)
        return cols

    def summary(self) -> dict:
        cols = self.step_columns()
        step_s = cols["step_s"]
        sched_s = cols["sched_s"]
        prepare_s = cols["prepare_s"]
        forward_s = cols["forward_s"]
        total_step = sum(step_s) or 1e-9
        shares = [s / t for s, t in zip(sched_s, step_s) if t > 0]
        decode_only = [
            t for t, p in zip(step_s, cols["num_prefill"]) if p == 0
        ]
        return {
            **self.counters,
            "wall_seconds": sum(step_s),
            "mean_step_ms": 1e3 * total_step / max(1, len(step_s)),
            "p50_step_ms": 1e3 * _percentile(step_s, 0.50),
            "p99_step_ms": 1e3 * _percentile(step_s, 0.99),
            "mean_decode_step_ms": 1e3 * (sum(decode_only) / len(decode_only))
            if decode_only
            else 0.0,
            "sched_seconds": sum(sched_s),
            "prepare_seconds": sum(prepare_s),
            "forward_seconds": sum(forward_s),
            "sched_share": sum(sched_s) / total_step,
            "prepare_share": sum(prepare_s) / total_step,
            "forward_share": sum(forward_s) / total_step,
            "p95_sched_share": _percentile(shares, 0.95),
            "mean_sched_us": 1e6 * sum(sched_s) / max(1, len(sched_s)),
            "max_running_seen": max(cols["running"], default=0),
            "max_waiting_seen": max(cols["waiting"], default=0),
        }
