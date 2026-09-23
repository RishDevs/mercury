"""W1-W6 trace generation.

Prefix-caching behaviour depends on the shape of sharing, not on what the tokens say, so
the traces are synthetic token ids with the sharing structure controlled exactly. Every
engine replays the identical file, and every request carries a fixed max_tokens with EOS
ignored so no engine can look faster by stopping early.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

TOKEN_LOW = 1000
TOKEN_HIGH = 100_000


@dataclass
class TraceRequest:
    rid: str
    session_id: str
    turn: int
    prompt_token_ids: list[int]
    max_tokens: int
    arrival_time_s: float | None = None
    depends_on: str | None = None
    think_time_s: float = 0.0
    shared_prefix_tokens: int = 0

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def private_tokens(self) -> int:
        """Tokens this request needs KV room for that nobody else can share: its own
        suffix plus everything it will generate."""
        return self.prompt_len - self.shared_prefix_tokens + self.max_tokens


@dataclass
class TraceMeta:
    workload: str
    seed: int
    request_rate: float
    num_requests: int
    working_set_tokens: int
    mean_private_tokens: float
    total_prompt_tokens: int
    total_output_tokens: int
    duration_s: float
    params: dict = field(default_factory=dict)


class Painter:
    """Hands out token ids; every distinct call makes content that shares nothing."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def tokens(self, count: int) -> list[int]:
        return self.rng.integers(TOKEN_LOW, TOKEN_HIGH, size=int(count)).tolist()

    def poisson_arrivals(self, count: int, rate: float) -> np.ndarray:
        gaps = self.rng.exponential(1.0 / rate, size=count)
        return np.cumsum(gaps)

    def lognormal_lengths(self, count: int, median: int, sigma: float, low: int, high: int):
        draw = self.rng.lognormal(np.log(median), sigma, size=count)
        return np.clip(draw, low, high).astype(int)

    def uniform_int(self, low: int, high: int, count: int = 1):
        return self.rng.integers(low, high + 1, size=count)


def _finish(workload: str, seed: int, rate: float, requests, working_set: int, params: dict):
    arrivals = [r.arrival_time_s for r in requests if r.arrival_time_s is not None]
    meta = TraceMeta(
        workload=workload,
        seed=seed,
        request_rate=rate,
        num_requests=len(requests),
        working_set_tokens=int(working_set),
        mean_private_tokens=float(np.mean([r.private_tokens for r in requests])) if requests else 0.0,
        total_prompt_tokens=int(sum(r.prompt_len for r in requests)),
        total_output_tokens=int(sum(r.max_tokens for r in requests)),
        duration_s=float(max(arrivals)) if arrivals else 0.0,
        params=params,
    )
    return requests, meta


# ---------------------------------------------------------------------------- W1


def w1_chatbot(seed: int, num_requests: int, rate: float, **params):
    """Independent prompts, no sharing. The control: prefix-aware must not lose here."""
    p = Painter(seed)
    lengths = p.lognormal_lengths(num_requests, median=500, sigma=0.55, low=200, high=1500)
    outputs = p.uniform_int(128, 256, num_requests)
    arrivals = p.poisson_arrivals(num_requests, rate)
    requests = [
        TraceRequest(
            rid=f"w1-{i}",
            session_id=f"w1-{i}",
            turn=0,
            prompt_token_ids=p.tokens(lengths[i]),
            max_tokens=int(outputs[i]),
            arrival_time_s=float(arrivals[i]),
        )
        for i in range(num_requests)
    ]
    return _finish("w1", seed, rate, requests, 0, {})


# ---------------------------------------------------------------------------- W2


def w2_shared_system(seed: int, num_requests: int, rate: float, groups: int = 16,
                     system_tokens: int = 2000, **params):
    """G system prompts shared by everyone in the group, arrivals interleaved across
    groups. G sets the pressure: the working set is groups x system_tokens."""
    p = Painter(seed)
    systems = [p.tokens(system_tokens) for _ in range(groups)]
    arrivals = p.poisson_arrivals(num_requests, rate)
    choice = p.rng.integers(0, groups, size=num_requests)
    suffixes = p.uniform_int(50, 300, num_requests)
    outputs = p.uniform_int(128, 256, num_requests)
    requests = [
        TraceRequest(
            rid=f"w2-{i}",
            session_id=f"g{int(choice[i])}",
            turn=0,
            prompt_token_ids=systems[int(choice[i])] + p.tokens(suffixes[i]),
            max_tokens=int(outputs[i]),
            arrival_time_s=float(arrivals[i]),
            shared_prefix_tokens=system_tokens,
        )
        for i in range(num_requests)
    ]
    return _finish(
        "w2", seed, rate, requests, groups * system_tokens,
        {"groups": groups, "system_tokens": system_tokens},
    )


# ---------------------------------------------------------------------------- W3


def w3_multi_turn(seed: int, num_requests: int, rate: float, turns_low: int = 4,
                  turns_high: int = 8, nominal_service_s: float = 1.0, **params):
    """Sessions whose prompt is the whole history so far. The prefix is session-private
    and grows, which is the realistic agent case.

    Each turn's history uses the trace's scripted assistant tokens rather than whatever
    the engine actually generated, so every engine sees byte-identical prompts. That
    understates caching slightly for all engines equally; it is the price of a fair
    comparison and it is stated in the write-up.
    """
    p = Painter(seed)
    sessions = max(1, num_requests // ((turns_low + turns_high) // 2))
    starts = p.poisson_arrivals(sessions, rate)
    requests: list[TraceRequest] = []
    intervals: list[tuple[float, float, int]] = []

    for s in range(sessions):
        turn_count = int(p.uniform_int(turns_low, turns_high)[0])
        history = p.tokens(int(p.uniform_int(200, 600)[0]))
        previous = None
        previous_len = 0
        clock = float(starts[s])
        peak = 0
        for turn in range(turn_count):
            if turn > 0:
                history = history + p.tokens(int(p.uniform_int(100, 400)[0]))
            output = int(p.uniform_int(64, 192)[0])
            think = float(p.rng.uniform(0.5, 3.0)) if turn > 0 else 0.0
            rid = f"w3-{s}-{turn}"
            requests.append(
                TraceRequest(
                    rid=rid,
                    session_id=f"w3-{s}",
                    turn=turn,
                    prompt_token_ids=list(history),
                    max_tokens=output,
                    arrival_time_s=float(starts[s]) if turn == 0 else None,
                    depends_on=previous,
                    think_time_s=think,
                    shared_prefix_tokens=previous_len,
                )
            )
            previous_len = len(history) + output
            peak = max(peak, len(history))
            history = history + p.tokens(output)
            clock += think + nominal_service_s
            previous = rid
        intervals.append((float(starts[s]), clock, peak))

    edges = sorted({t for start, end, _ in intervals for t in (start, end)})
    working_set = 0
    for edge in edges:
        live = sum(peak for start, end, peak in intervals if start <= edge < end)
        working_set = max(working_set, live)

    return _finish(
        "w3", seed, rate, requests, working_set,
        {"sessions": sessions, "turns": [turns_low, turns_high],
         "nominal_service_s": nominal_service_s},
    )


# ---------------------------------------------------------------------------- W4


def w4_tool_agents(seed: int, num_requests: int, rate: float, catalogue_tokens: int = 3000,
                   tasks: int = 8, task_tokens: int = 1000, **params):
    """Two-level sharing: one catalogue everybody sees, then one of `tasks` instruction
    blocks, then a unique suffix."""
    p = Painter(seed)
    catalogue = p.tokens(catalogue_tokens)
    blocks = [p.tokens(task_tokens) for _ in range(tasks)]
    arrivals = p.poisson_arrivals(num_requests, rate)
    choice = p.rng.integers(0, tasks, size=num_requests)
    suffixes = p.uniform_int(50, 300, num_requests)
    outputs = p.uniform_int(128, 256, num_requests)
    requests = [
        TraceRequest(
            rid=f"w4-{i}",
            session_id=f"t{int(choice[i])}",
            turn=0,
            prompt_token_ids=catalogue + blocks[int(choice[i])] + p.tokens(suffixes[i]),
            max_tokens=int(outputs[i]),
            arrival_time_s=float(arrivals[i]),
            shared_prefix_tokens=catalogue_tokens + task_tokens,
        )
        for i in range(num_requests)
    ]
    return _finish(
        "w4", seed, rate, requests, catalogue_tokens + tasks * task_tokens,
        {"catalogue_tokens": catalogue_tokens, "tasks": tasks, "task_tokens": task_tokens},
    )


# ---------------------------------------------------------------------------- W5, W6


def w5_long_context(seed: int, num_requests: int, rate: float, **params):
    p = Painter(seed)
    lengths = p.uniform_int(6000, 8000, num_requests)
    outputs = p.uniform_int(64, 128, num_requests)
    arrivals = p.poisson_arrivals(num_requests, rate)
    requests = [
        TraceRequest(
            rid=f"w5-{i}",
            session_id=f"w5-{i}",
            turn=0,
            prompt_token_ids=p.tokens(lengths[i]),
            max_tokens=int(outputs[i]),
            arrival_time_s=float(arrivals[i]),
        )
        for i in range(num_requests)
    ]
    return _finish("w5", seed, rate, requests, 0, {})


def w6_high_concurrency(seed: int, num_requests: int, rate: float, **params):
    p = Painter(seed)
    lengths = p.uniform_int(100, 300, num_requests)
    outputs = p.uniform_int(32, 64, num_requests)
    arrivals = p.poisson_arrivals(num_requests, rate)
    requests = [
        TraceRequest(
            rid=f"w6-{i}",
            session_id=f"w6-{i}",
            turn=0,
            prompt_token_ids=p.tokens(lengths[i]),
            max_tokens=int(outputs[i]),
            arrival_time_s=float(arrivals[i]),
        )
        for i in range(num_requests)
    ]
    return _finish("w6", seed, rate, requests, 0, {})


WORKLOADS = {
    "w1": w1_chatbot,
    "w2": w2_shared_system,
    "w3": w3_multi_turn,
    "w4": w4_tool_agents,
    "w5": w5_long_context,
    "w6": w6_high_concurrency,
}


def generate(workload: str, seed: int, num_requests: int, rate: float, **params):
    if workload not in WORKLOADS:
        raise ValueError(f"unknown workload {workload!r}; have {sorted(WORKLOADS)}")
    return WORKLOADS[workload](seed=seed, num_requests=num_requests, rate=rate, **params)


# ---------------------------------------------------------------------------- io


def write_trace(path: Path, requests: list[TraceRequest], meta: TraceMeta) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for r in requests:
            handle.write(json.dumps(asdict(r), separators=(",", ":")) + "\n")
    path.with_suffix(".meta.json").write_text(json.dumps(asdict(meta), indent=2))
    return path


def load_trace(path: Path) -> tuple[list[TraceRequest], TraceMeta]:
    path = Path(path)
    requests = [TraceRequest(**json.loads(line)) for line in path.read_text().splitlines() if line]
    meta = TraceMeta(**json.loads(path.with_suffix(".meta.json").read_text()))
    return requests, meta


def capacity_for_pressure(
    meta: TraceMeta, rho: float, max_running: int = 64, block_size: int = 16
) -> int:
    """Blocks to put a workload at pressure rho = working set / KV capacity.

    The headroom term is what the running batch needs for its own private tokens; without
    it a high-rho point starves on private suffixes rather than on shared prefixes, which
    would measure the wrong thing.
    """
    if meta.working_set_tokens <= 0:
        raise ValueError(f"{meta.workload} has no shared working set; set capacity directly")
    shared = meta.working_set_tokens / rho
    private = max_running * meta.mean_private_tokens
    return max(1, int(-(-(shared + private) // block_size)))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser("bench.workloads")
    parser.add_argument("--workload", required=True, choices=sorted(WORKLOADS))
    parser.add_argument("--out", default="traces")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-requests", type=int, default=500)
    parser.add_argument("--rate", type=float, default=8.0)
    parser.add_argument("--groups", type=int, default=16)
    parser.add_argument("--tasks", type=int, default=8)
    parser.add_argument("--max-running", type=int, default=64)
    parser.add_argument("--rho", type=float, nargs="*", default=[0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--name", default=None)
    args = parser.parse_args(argv)

    extra = {}
    if args.workload == "w2":
        extra["groups"] = args.groups
    if args.workload == "w4":
        extra["tasks"] = args.tasks

    requests, meta = generate(
        args.workload, args.seed, args.num_requests, args.rate, **extra
    )
    name = args.name or f"{args.workload}_r{args.rate:g}_s{args.seed}"
    path = write_trace(Path(args.out) / f"{name}.jsonl", requests, meta)
    print(f"wrote {path} ({meta.num_requests} requests, {meta.total_prompt_tokens} prompt tokens)")
    print(
        f"working set {meta.working_set_tokens} tokens, "
        f"mean private {meta.mean_private_tokens:.0f} tokens/request"
    )
    if meta.working_set_tokens > 0:
        for rho in args.rho:
            blocks = capacity_for_pressure(meta, rho, args.max_running)
            print(f"  rho={rho:<5g} -> {blocks} blocks ({blocks * 16} tokens)")


if __name__ == "__main__":
    main()
