"""End to end over a socket: the real server process, the real trace replayer, the real
analysis. No GPU and no weights, so this runs in CI and on a laptop and still covers the
SSE parsing, the open-loop timing, the warmup-then-reset protocol and the stats plumbing
that the reported numbers are computed from."""

from __future__ import annotations

import asyncio
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from bench.analyze import load_runs, main_table, summarize
from bench.client import run_once, wait_for_health
from bench.run_matrix import Point, check_capacity, reported_capacity, verify
from bench.workloads import capacity_for_pressure, generate, load_trace, write_trace

pytest.importorskip("aiohttp")


def free_port() -> int:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


class ServerProcess:
    def __init__(self, tmp_path: Path, blocks: int = 512, policy: str = "lpm_aging", **flags):
        self.port = free_port()
        self.log = tmp_path / "server.log"
        self.command = [
            sys.executable, "-m", "mercury.server",
            "--fake-runner",
            "--device", "cpu",
            "--model", "fake-model",
            "--port", str(self.port),
            "--policy", policy,
            "--num-blocks", str(blocks),
            "--max-running", "8",
            "--max-model-len", "4096",
            *[str(f) for f in flags.get("extra", [])],
        ]

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> "ServerProcess":
        self.handle = self.log.open("w")
        self.process = subprocess.Popen(
            self.command, stdout=self.handle, stderr=subprocess.STDOUT
        )
        if not asyncio.run(wait_for_health(self.url, timeout=90)):
            self.__exit__()
            raise RuntimeError(f"server never came up:\n{self.log.read_text()}")
        return self

    def __exit__(self, *_exc) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=20)
        self.handle.close()


@pytest.fixture(scope="module")
def trace(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("traces")
    requests, meta = generate("w2", seed=0, num_requests=40, rate=25.0, groups=4, system_tokens=256)
    return write_trace(directory / "w2_small.jsonl", requests, meta)


def test_the_server_reports_its_capacity_in_a_form_the_matrix_can_check(tmp_path):
    with ServerProcess(tmp_path, blocks=333) as server:
        time.sleep(0.2)
        log = server.log.read_text()
    assert reported_capacity("mercury", log) == 333
    point = Point("mercury", "fcfs", "w2", 2.0, 8.0, 0, blocks=333)
    assert check_capacity(point, log, 16)["reported_capacity"] == 333


def test_a_capacity_cap_that_did_not_take_effect_is_caught(tmp_path):
    with ServerProcess(tmp_path, blocks=333) as server:
        time.sleep(0.2)
        log = server.log.read_text()
    point = Point("mercury", "fcfs", "w2", 2.0, 8.0, 0, blocks=4096)
    with pytest.raises(RuntimeError, match="ignored the capacity cap"):
        check_capacity(point, log, 16)


def test_full_replay_produces_analysable_records(tmp_path, trace):
    with ServerProcess(tmp_path, blocks=768, policy="lpm_aging") as server:
        result = asyncio.run(
            run_once(server.url, trace, "fake-model", engine="mercury", warmup_requests=5)
        )

    requests, meta = load_trace(trace)
    assert len(result["requests"]) == len(requests)
    ok = [r for r in result["requests"] if r["status"] == "ok"]
    assert len(ok) == len(requests)

    for record in ok:
        assert record["output_tokens"] == record["expected_tokens"]
        assert record["first_token"] is not None
        assert record["last_token"] >= record["first_token"]
        assert len(record["token_times"]) == record["output_tokens"]

    stats = result["server_stats"]
    assert stats["allocator"]["hit_tokens"] > 0, "four shared system prompts should hit"
    assert stats["config"]["policy"] == "lpm_aging"
    assert result["server_requests"], "server-side queue records are the fairness metric"

    # The warmup ran off-trace and the counters were reset, so every queried token here
    # belongs to the measured run.
    assert stats["allocator"]["query_tokens"] == sum(r["prompt_tokens"] for r in ok)


def test_the_summary_row_and_table_come_out_of_a_real_run(tmp_path, trace):
    with ServerProcess(tmp_path, blocks=768) as server:
        result = asyncio.run(
            run_once(server.url, trace, "fake-model", engine="mercury", warmup_requests=5)
        )
    result["point"] = {
        "engine": "mercury", "policy": "fcfs", "workload": "w2",
        "rho": 2.0, "rate": 25.0, "seed": 0, "blocks": 768,
        "kv_capacity_tokens": 768 * 16,
    }
    path = tmp_path / "mercury_fcfs_w2_rho2_rate25_seed0.json"
    path.write_text(json.dumps(result))

    run = summarize(path)
    assert run.row["requests_ok"] == run.row["requests_total"]
    assert run.row["ttft_p95"] is not None and run.row["ttft_p95"] > 0
    assert run.row["queue_max"] is not None
    assert 0.0 <= run.row["hit_rate"] <= 1.0
    assert run.row["output_tok_per_s"] > 0
    assert 0.0 <= run.row["sched_share"] <= 1.0
    assert run.row["verification"] == "clean"

    runs = load_runs(tmp_path)
    table = main_table(runs, "w2", 2.0, None)
    assert "mercury FCFS" in table
    assert "Hit rate" in table


def test_verification_flags_a_run_with_the_wrong_token_counts():
    fake = {
        "requests": [
            {"status": "ok", "output_tokens": 12, "expected_tokens": 16},
            {"status": "ok", "output_tokens": 16, "expected_tokens": 16},
        ],
        "client_cpu_utilisation": 0.1,
    }
    problems = verify(fake)
    assert any("different token count" in p for p in problems)


def test_capacity_scales_the_way_the_pressure_definition_says(trace):
    _requests, meta = load_trace(trace)
    capacities = {
        rho: capacity_for_pressure(meta, rho, max_running=8) for rho in (0.5, 1.0, 2.0, 4.0)
    }
    assert capacities[0.5] > capacities[1.0] > capacities[2.0] > capacities[4.0]

    shared_only = {
        rho: blocks * 16 - 8 * meta.mean_private_tokens for rho, blocks in capacities.items()
    }
    # The shared allowance is the working set divided by rho, to within a block of rounding.
    for rho, tokens in shared_only.items():
        assert abs(tokens - meta.working_set_tokens / rho) < 32
