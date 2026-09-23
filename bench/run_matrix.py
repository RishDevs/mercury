"""Drives the experiment grid: engine x policy x workload x pressure x rate x repeat.

Rules this encodes, because each of them is a way benchmark numbers get dismissed:
  - a fresh server per point, so no run inherits the previous one's warm cache
  - off-trace warmup requests, then a stats reset
  - run order shuffled inside a session, so thermal drift spreads across conditions
  - the capacity cap verified from the server's own startup log, never assumed
  - every raw per-request record written to disk; percentiles computed later
  - points whose result file already exists are skipped, so a killed session resumes
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import itertools
import json
import os
import random
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .client import run_once, wait_for_health
from .workloads import capacity_for_pressure, generate, load_trace, write_trace

CAPACITY_PATTERNS = {
    "mercury": [r"kv_blocks=(\d+)"],
    "vllm": [
        r"GPU KV cache size:\s*([\d,]+)\s*tokens",
        r"#\s*GPU blocks:\s*(\d+)",
        r"gpu_blocks[\"']?\s*[:=]\s*(\d+)",
    ],
    "sglang": [
        r"max_total_num_tokens\s*[:=]\s*(\d+)",
        r"KV Cache is allocated.*?#tokens:\s*(\d+)",
    ],
}


@dataclass
class Point:
    engine: str
    policy: str
    workload: str
    rho: float
    rate: float
    seed: int
    blocks: int
    extra: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        base = (
            f"{self.engine}_{self.policy}_{self.workload}"
            f"_rho{self.rho:g}_rate{self.rate:g}_seed{self.seed}"
        )
        if self.extra:
            # Without this, two points that differ only in alpha or in a flag would write
            # to the same file and the second would be skipped as already done.
            tag = "-".join(f"{k}{v}" for k, v in sorted(self.extra.items()))
            base += "_" + re.sub(r"[^A-Za-z0-9.-]+", "", tag)
        return base


def gpu_facts() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip().splitlines()
        names = [line.split(",")[0].strip() for line in out]
        driver = out[0].split(",")[1].strip() if out else None
        return {"gpus": names, "driver": driver}
    except Exception:
        return {"gpus": [], "driver": None}


def package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def server_command(point: Point, cfg: dict, port: int) -> list[str]:
    model = cfg["model"]
    revision = cfg.get("revision")
    if point.engine == "mercury":
        cmd = [
            sys.executable, "-m", "mercury.server",
            "--model", model,
            "--port", str(port),
            "--dtype", "float16",
            "--policy", point.policy,
            "--num-blocks", str(point.blocks),
            "--block-size", str(cfg.get("block_size", 16)),
            "--max-running", str(cfg.get("max_running", 64)),
            "--prefill-budget", str(cfg.get("prefill_budget", 8192)),
            "--max-model-len", str(cfg.get("max_model_len", 8192)),
            "--aging-tokens-per-s", str(point.extra.get("alpha", cfg.get("alpha", 2000.0))),
            "--max-wait-s", str(point.extra.get("max_wait_s", cfg.get("max_wait_s", 5.0))),
            "--scan-limit", str(cfg.get("scan_limit", 128)),
        ]
        if revision:
            cmd += ["--revision", revision]
        if not cfg.get("fused_rmsnorm", True):
            cmd += ["--no-fused-rmsnorm"]
        if cfg.get("fake_runner"):
            # Exercises the entire harness with no GPU and no weights. Use it to prove the
            # pipeline works before spending session quota, never to report numbers.
            cmd += [
                "--fake-runner",
                "--device", "cpu",
                "--fake-step-delay", str(cfg.get("fake_step_delay", 0.0)),
            ]
        return cmd

    if point.engine == "vllm":
        cmd = [
            "vllm", "serve", model,
            "--port", str(port),
            "--dtype", "float16",
            "--enable-prefix-caching",
            "--num-gpu-blocks-override", str(point.blocks),
            "--max-num-seqs", str(cfg.get("max_running", 64)),
            "--max-model-len", str(cfg.get("max_model_len", 8192)),
            "--disable-log-requests",
        ]
        if revision:
            cmd += ["--revision", revision]
        if point.extra.get("enforce_eager"):
            cmd += ["--enforce-eager"]
        return cmd

    if point.engine == "sglang":
        cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", model,
            "--port", str(port),
            "--dtype", "float16",
            "--attention-backend", "triton",
            "--sampling-backend", "pytorch",
            "--max-total-tokens", str(point.blocks * cfg.get("block_size", 16)),
            "--max-running-requests", str(cfg.get("max_running", 64)),
            "--schedule-policy", point.policy,
        ]
        if revision:
            cmd += ["--revision", revision]
        return cmd

    raise ValueError(f"unknown engine {point.engine!r}")


class Server:
    def __init__(self, command: list[str], log_path: Path, env: dict | None = None):
        self.command = command
        self.log_path = log_path
        self.env = {**os.environ, **(env or {})}
        self.process: subprocess.Popen | None = None

    def __enter__(self) -> "Server":
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.log_path.open("w")
        self.process = subprocess.Popen(
            self.command,
            stdout=self.handle,
            stderr=subprocess.STDOUT,
            env=self.env,
            start_new_session=True,
        )
        return self

    def __exit__(self, *_exc) -> None:
        if self.process and self.process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(self.process.pid), signal.SIGINT)
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                self.process.wait(timeout=30)
        self.handle.close()

    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def log(self) -> str:
        return self.log_path.read_text(errors="replace")


def reported_capacity(engine: str, log: str) -> int | None:
    for pattern in CAPACITY_PATTERNS[engine]:
        match = re.search(pattern, log, re.IGNORECASE | re.DOTALL)
        if match:
            return int(match.group(1).replace(",", ""))
    return None


def check_capacity(point: Point, log: str, block_size: int, tolerance: float = 0.10) -> dict:
    reported = reported_capacity(point.engine, log)
    if point.engine == "mercury":
        expected, unit = point.blocks, "blocks"
    elif point.engine == "vllm":
        expected, unit = point.blocks, "blocks"
    else:
        expected, unit = point.blocks * block_size, "tokens"
    if reported is None:
        raise RuntimeError(
            f"{point.engine} never reported its KV capacity; the flag name has probably "
            f"changed in this version. Check the log at the path in this run's metadata."
        )
    if abs(reported - expected) > tolerance * expected:
        raise RuntimeError(
            f"{point.engine} ignored the capacity cap: asked for {expected} {unit}, "
            f"it reports {reported}. Running this point would silently measure the wrong rho."
        )
    return {"reported_capacity": reported, "expected_capacity": expected, "unit": unit}


class DeviceMonitor:
    """Samples memory, clocks and temperature at 10 Hz for the whole run. Clocks cannot
    be locked on a shared Kaggle machine, but drift can at least be shown not to have
    happened, and peak memory is a reported metric."""

    FIELDS = "timestamp,memory.used,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu"

    def __init__(self, path: Path, interval_ms: int = 100):
        self.path = path
        self.interval_ms = interval_ms
        self.process = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.handle = self.path.open("w")
            self.process = subprocess.Popen(
                [
                    "nvidia-smi",
                    f"--query-gpu={self.FIELDS}",
                    "--format=csv,noheader,nounits",
                    f"-lms{self.interval_ms}",
                ],
                stdout=self.handle,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            self.process = None
        return self

    def __exit__(self, *_exc):
        if self.process:
            self.process.terminate()
            with contextlib.suppress(Exception):
                self.process.wait(timeout=10)
            self.handle.close()


def trace_for(cfg: dict, workload: str, rate: float, seed: int) -> Path:
    traces = Path(cfg.get("traces_dir", "traces"))
    path = traces / f"{workload}_r{rate:g}_s{seed}.jsonl"
    if not path.exists():
        extra = dict(cfg.get("workload_params", {}).get(workload, {}))
        requests, meta = generate(
            workload, seed=seed, num_requests=cfg.get("num_requests", 500), rate=rate, **extra
        )
        write_trace(path, requests, meta)
    return path


def blocks_for(cfg: dict, trace_path: Path, rho: float) -> int:
    _requests, meta = load_trace(trace_path)
    max_running = cfg.get("max_running", 64)
    block_size = cfg.get("block_size", 16)
    if meta.working_set_tokens <= 0:
        # No sharing: pressure is meaningless, so give the batch twice what it needs and
        # say so. This is the control point, not a pressure point.
        tokens = 2 * max_running * meta.mean_private_tokens
        return max(1, int(-(-tokens // block_size)))
    return capacity_for_pressure(meta, rho, max_running, block_size)


def expand(cfg: dict) -> list[Point]:
    points: list[Point] = []
    r_sat = cfg["r_sat"]
    for group in cfg["points"]:
        engine = group["engine"]
        for policy, workload, rho, factor, seed in itertools.product(
            group.get("policies", ["fcfs"]),
            group["workloads"],
            group.get("rhos", [1.0]),
            group.get("rate_factors", cfg.get("rate_factors", [0.8])),
            group.get("seeds", cfg.get("seeds", [0])),
        ):
            rate = round(factor * r_sat, 4)
            trace = trace_for(cfg, workload, rate, seed)
            points.append(
                Point(
                    engine=engine,
                    policy=policy,
                    workload=workload,
                    rho=float(rho),
                    rate=rate,
                    seed=seed,
                    blocks=blocks_for(cfg, trace, float(rho)),
                    extra=dict(group.get("extra", {})),
                )
            )
    return points


def run_point(point: Point, cfg: dict, out_dir: Path, port: int) -> dict:
    trace = trace_for(cfg, point.workload, point.rate, point.seed)
    log_path = out_dir / "logs" / f"{point.name}.server.log"
    command = server_command(point, cfg, port)
    base_url = f"http://127.0.0.1:{port}"

    with Server(command, log_path) as server:
        if not asyncio.run(wait_for_health(base_url, timeout=cfg.get("startup_timeout", 900))):
            raise RuntimeError(f"{point.engine} never became healthy; see {log_path}")
        capacity = check_capacity(point, server.log(), cfg.get("block_size", 16))
        gpu_log = out_dir / "logs" / f"{point.name}.gpu.csv"
        with DeviceMonitor(gpu_log):
            result = asyncio.run(
                run_once(
                    base_url,
                    trace,
                    cfg["model"],
                    engine=point.engine,
                    warmup_requests=cfg.get("warmup_requests", 50),
                    cancel_fraction=point.extra.get("cancel_fraction", 0.0),
                    seed=point.seed,
                )
            )
        if not server.alive():
            raise RuntimeError(f"{point.engine} died during the run; see {log_path}")

    result["point"] = {
        "engine": point.engine,
        "policy": point.policy,
        "workload": point.workload,
        "rho": point.rho,
        "rate": point.rate,
        "seed": point.seed,
        "blocks": point.blocks,
        "kv_capacity_tokens": point.blocks * cfg.get("block_size", 16),
        **point.extra,
    }
    result["provenance"] = {
        "command": " ".join(shlex.quote(c) for c in command),
        "server_log": str(log_path),
        "gpu_log": str(gpu_log),
        "model": cfg["model"],
        "revision": cfg.get("revision"),
        "engine_version": package_version(
            {"mercury": "mercury-serve", "vllm": "vllm", "sglang": "sglang"}[point.engine]
        ),
        "torch_version": package_version("torch"),
        "timestamp": time.time(),
        **gpu_facts(),
        **capacity,
    }
    return result


def verify(result: dict) -> list[str]:
    problems = []
    records = result["requests"]
    ok = [r for r in records if r["status"] == "ok"]
    if len(ok) < 0.98 * len(records):
        problems.append(f"only {len(ok)}/{len(records)} requests completed")
    wrong = [r for r in ok if r["output_tokens"] != r["expected_tokens"]]
    if wrong:
        problems.append(
            f"{len(wrong)} requests returned a different token count than asked for; "
            f"engines cannot be compared on this run"
        )
    if result["client_cpu_utilisation"] > 0.7:
        problems.append(
            f"client used {result['client_cpu_utilisation']:.0%} of a core; it is part of "
            f"the measurement, not an observer of it"
        )
    return problems


def push_to_hub(path: Path, repo: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=f"raw/{path.name}",
        repo_id=repo,
        repo_type="dataset",
    )


def run_matrix(cfg: dict, out_dir: Path, port: int, hub_repo: str | None, dry_run: bool) -> None:
    points = expand(cfg)
    random.Random(cfg.get("shuffle_seed", 0)).shuffle(points)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "grid.json").write_text(json.dumps([p.name for p in points], indent=2))

    pending = [p for p in points if not (out_dir / f"{p.name}.json").exists()]
    print(f"{len(points)} points, {len(points) - len(pending)} already done, {len(pending)} to run")
    if dry_run:
        for p in pending:
            print(f"  {p.name}  blocks={p.blocks}")
        return

    for index, point in enumerate(pending, 1):
        target = out_dir / f"{point.name}.json"
        print(f"[{index}/{len(pending)}] {point.name} (blocks={point.blocks})", flush=True)
        started = time.time()
        try:
            result = run_point(point, cfg, out_dir, port)
        except Exception as exc:
            print(f"  FAILED: {exc}", flush=True)
            (out_dir / f"{point.name}.failed.txt").write_text(str(exc))
            continue
        problems = verify(result)
        result["verification"] = problems
        target.write_text(json.dumps(result))
        status = "clean" if not problems else "; ".join(problems)
        print(f"  {time.time() - started:.0f}s -> {target.name} [{status}]", flush=True)
        if hub_repo:
            with contextlib.suppress(Exception):
                push_to_hub(target, hub_repo)
        time.sleep(cfg.get("cooldown_s", 5))


def rate_sweep(cfg: dict, out_dir: Path, port: int, rates: list[float]) -> None:
    """Finds R_sat: the rate at which p95 TTFT goes vertical. The matrix then runs at
    0.5 and 0.8 of it, because below ~0.3 the waiting queue is empty and no policy can
    reorder anything, and above it every policy is dominated by an unbounded queue."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for rate in rates:
        trace = trace_for(cfg, cfg.get("sweep_workload", "w2"), rate, cfg.get("seeds", [0])[0])
        point = Point(
            engine="mercury",
            policy="fcfs",
            workload=cfg.get("sweep_workload", "w2"),
            rho=cfg.get("sweep_rho", 2.0),
            rate=rate,
            seed=cfg.get("seeds", [0])[0],
            blocks=blocks_for(cfg, trace, cfg.get("sweep_rho", 2.0)),
        )
        print(f"sweep rate={rate:g}", flush=True)
        result = run_point(point, cfg, out_dir, port)
        (out_dir / f"sweep_rate{rate:g}.json").write_text(json.dumps(result))
        ttfts = [
            r["first_token"] - r["sent"]
            for r in result["requests"]
            if r["status"] == "ok" and r["first_token"] is not None
        ]
        output_tokens = sum(r["output_tokens"] for r in result["requests"])
        rows.append(
            {
                "rate": rate,
                "p50_ttft": float(np.percentile(ttfts, 50)) if ttfts else None,
                "p95_ttft": float(np.percentile(ttfts, 95)) if ttfts else None,
                "output_tok_per_s": output_tokens / result["wall_seconds"],
            }
        )
        print(f"  p95 TTFT {rows[-1]['p95_ttft']:.3f}s, {rows[-1]['output_tok_per_s']:.0f} tok/s")

    (out_dir / "rate_sweep.json").write_text(json.dumps(rows, indent=2))
    print("\nrate   p50 TTFT   p95 TTFT   output tok/s")
    for row in rows:
        print(f"{row['rate']:<6g} {row['p50_ttft']:<10.3f} {row['p95_ttft']:<10.3f} {row['output_tok_per_s']:.0f}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser("bench.run_matrix")
    parser.add_argument("--grid", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--hub-repo", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sweep", type=float, nargs="*", default=None,
                        help="run a rate sweep at these request rates instead of the matrix")
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(Path(args.grid).read_text())
    out_dir = Path(args.out or cfg.get("out_dir", "results"))
    if args.sweep is not None:
        rate_sweep(cfg, out_dir / "sweep", args.port, args.sweep)
    else:
        run_matrix(cfg, out_dir, args.port, args.hub_repo, args.dry_run)


if __name__ == "__main__":
    main()
