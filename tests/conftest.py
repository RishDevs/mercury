from __future__ import annotations

import pytest

from mercury.config import EngineConfig
from mercury.engine import Engine

from .fake_runner import FakeClock, FakeRunner


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires a CUDA device")
    config.addinivalue_line("markers", "hf: requires transformers and downloaded weights")
    config.addinivalue_line("markers", "slow: takes more than a few seconds")


def pytest_collection_modifyitems(config, items):
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False
    skip_gpu = pytest.mark.skip(reason="no CUDA device available")
    for item in items:
        if "gpu" in item.keywords and not has_cuda:
            item.add_marker(skip_gpu)


class Harness:
    """An engine wired to a fake runner and a clock the test drives by hand."""

    def __init__(self, engine: Engine, clock: FakeClock, runner: FakeRunner):
        self.engine = engine
        self.clock = clock
        self.runner = runner
        self.events: dict[str, list] = {}

    def sink_for(self, rid: str):
        bucket = self.events.setdefault(rid, [])

        def sink(event):
            bucket.append(event)

        return sink

    def submit(self, rid: str, prompt, max_tokens=4, **kwargs):
        self.engine.submit(rid, prompt, max_tokens, self.sink_for(rid), **kwargs)
        return rid

    def run(self, steps: int, dt: float = 0.01) -> int:
        executed = 0
        for _ in range(steps):
            if self.engine.step() is not None:
                executed += 1
            self.clock.advance(dt)
        return executed

    def drain(self, limit: int = 100_000, dt: float = 0.01) -> int:
        steps = 0
        while steps < limit:
            self.engine._drain_commands()
            if not self.engine.sched.has_work():
                break
            self.engine.step()
            self.clock.advance(dt)
            steps += 1
        return steps

    def tokens(self, rid: str) -> list[int]:
        return [e.token for e in self.events.get(rid, []) if e.kind == "token"]

    def finish_reason(self, rid: str) -> str | None:
        for event in self.events.get(rid, []):
            if event.kind == "done":
                return event.finish_reason
        return None


def build(policy: str = "fcfs", num_blocks: int = 64, **overrides) -> Harness:
    cfg = EngineConfig(
        model="fake",
        device="cpu",
        num_blocks=num_blocks,
        block_size=overrides.pop("block_size", 16),
        max_running=overrides.pop("max_running", 8),
        prefill_budget=overrides.pop("prefill_budget", 8192),
        max_model_len=overrides.pop("max_model_len", 1024),
        policy=policy,
        **overrides,
    )
    clock = FakeClock()
    runner = FakeRunner(block_size=cfg.block_size)
    engine = Engine(cfg, runner=runner, clock=clock)
    return Harness(engine, clock, runner)


@pytest.fixture
def harness():
    return build()
