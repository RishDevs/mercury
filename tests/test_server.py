from __future__ import annotations

import asyncio
import json

import pytest

from mercury.config import EngineConfig
from mercury.engine import Engine
from mercury.server import create_app

from .fake_runner import FakeRunner

httpx = pytest.importorskip("httpx")


def parse_sse(body: str) -> list[dict]:
    out = []
    for line in body.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        out.append({"done": True} if payload == "[DONE]" else json.loads(payload))
    return out


def token_chunks(events: list[dict]) -> list[dict]:
    return [e for e in events if e.get("choices") and e["choices"][0]["token_ids"]]


class App:
    """The real FastAPI app and the real engine thread, wired to a fake runner so the
    whole HTTP path can be exercised without a GPU."""

    def __init__(self, **overrides):
        step_delay = overrides.pop("step_delay", 0.0)
        cfg = EngineConfig(
            model="fake",
            device="cpu",
            num_blocks=overrides.pop("num_blocks", 256),
            max_running=overrides.pop("max_running", 8),
            max_model_len=overrides.pop("max_model_len", 1024),
            idle_sleep_s=1e-4,
            **overrides,
        )
        self.cfg = cfg
        self.engine = Engine(cfg, runner=FakeRunner(step_delay=step_delay))
        self.app = create_app(cfg, tokenizer=None, engine=self.engine)

    async def __aenter__(self) -> "httpx.AsyncClient":
        self.engine.start()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://engine"
        )
        return self.client

    async def __aexit__(self, *_exc) -> None:
        await self.client.aclose()
        self.engine.shutdown()

    async def wait_until_drained(self, timeout: float = 5.0) -> None:
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            if self.engine.leaked_blocks() == 0 and not self.engine.sched.has_work():
                return
            await asyncio.sleep(0.01)


def run(scenario):
    asyncio.run(scenario())


async def completion(client, prompt, max_tokens, stream=True, **extra):
    return await client.post(
        "/v1/completions",
        json={
            "prompt": prompt,
            "max_tokens": max_tokens,
            "stream": stream,
            "ignore_eos": True,
            **extra,
        },
    )


def test_streaming_returns_one_chunk_per_token_then_done():
    async def scenario():
        harness = App()
        async with harness as client:
            async with client.stream(
                "POST",
                "/v1/completions",
                json={"prompt": list(range(64)), "max_tokens": 12, "stream": True, "ignore_eos": True},
            ) as response:
                assert response.status_code == 200
                assert response.headers["content-type"].startswith("text/event-stream")
                body = "".join([chunk async for chunk in response.aiter_text()])
            events = parse_sse(body)
            assert len(token_chunks(events)) == 12
            assert events[-1] == {"done": True}
            assert events[-2]["choices"][0]["finish_reason"] == "length"

    run(scenario)


def test_non_streaming_returns_the_whole_completion():
    async def scenario():
        harness = App()
        async with harness as client:
            response = await completion(client, list(range(40)), 9, stream=False)
            assert response.status_code == 200
            payload = response.json()
            assert len(payload["choices"][0]["token_ids"]) == 9
            assert payload["usage"]["completion_tokens"] == 9
            assert payload["usage"]["prompt_tokens"] == 40
            assert payload["choices"][0]["finish_reason"] == "length"

    run(scenario)


def test_two_requests_with_a_shared_prefix_hit_the_cache():
    async def scenario():
        harness = App()
        shared = list(range(128))
        async with harness as client:
            await completion(client, shared + [1], 4, stream=False)
            await completion(client, shared + [2], 4, stream=False)
            stats = (await client.get("/stats")).json()
            assert stats["allocator"]["hit_tokens"] > 0
            assert stats["config"]["policy"] == "fcfs"

    run(scenario)


def test_reset_stats_clears_counters_without_flushing_the_cache():
    async def scenario():
        harness = App()
        shared = list(range(128))
        async with harness as client:
            await completion(client, shared + [1], 4, stream=False)
            cached_before = (await client.get("/stats")).json()["allocator"]["cached_blocks"]
            assert cached_before > 0
            await client.post("/reset_stats")
            after = (await client.get("/stats")).json()
            assert after["allocator"]["query_tokens"] == 0
            assert after["allocator"]["cached_blocks"] == cached_before
            await completion(client, shared + [2], 4, stream=False)
            assert (await client.get("/stats")).json()["allocator"]["hit_tokens"] > 0

    run(scenario)


def test_a_prompt_larger_than_the_pool_is_refused():
    async def scenario():
        harness = App(num_blocks=8, max_model_len=1024)
        async with harness as client:
            response = await completion(client, list(range(500)), 10, stream=False)
            assert response.status_code == 400
            assert "ceiling" in response.json()["detail"]

    run(scenario)


def test_a_text_prompt_without_a_tokenizer_is_a_clear_error():
    async def scenario():
        harness = App()
        async with harness as client:
            response = await completion(client, "hello there", 4, stream=False)
            assert response.status_code == 400
            assert "token ids" in response.json()["detail"]

    run(scenario)


def test_batched_prompts_are_refused_rather_than_silently_truncated():
    async def scenario():
        harness = App()
        async with harness as client:
            response = await completion(client, [[1, 2, 3], [4, 5, 6]], 4, stream=False)
            assert response.status_code == 400
            assert "one per request" in response.json()["detail"]

    run(scenario)


class LiveServer:
    """A real uvicorn server on a real socket.

    Client disconnects cannot be tested through an in-process ASGI transport: it never
    delivers http.disconnect, so the cancellation path never runs and the test would pass
    for the wrong reason. Anything about disconnects goes through this.
    """

    def __init__(self, **overrides):
        import socket

        import uvicorn

        self.harness = App(**overrides)
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        self.port = probe.getsockname()[1]
        probe.close()
        self.server = uvicorn.Server(
            uvicorn.Config(
                self.harness.app, host="127.0.0.1", port=self.port, log_level="error",
                access_log=False,
            )
        )

    @property
    def engine(self):
        return self.harness.engine

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> "LiveServer":
        import threading
        import time

        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        for _ in range(200):
            if self.server.started:
                return self
            time.sleep(0.05)
        raise RuntimeError("uvicorn never started")

    def __exit__(self, *_exc) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


async def stream_then_abandon(url: str, prompt, max_tokens: int, after: int) -> int:
    seen = 0
    async with httpx.AsyncClient(base_url=url) as client:
        async with client.stream(
            "POST",
            "/v1/completions",
            json={"prompt": prompt, "max_tokens": max_tokens, "stream": True, "ignore_eos": True},
        ) as response:
            async for line in response.aiter_lines():
                if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                    continue
                chunk = json.loads(line[5:])
                if chunk.get("choices") and chunk["choices"][0]["token_ids"]:
                    seen += 1
                if seen >= after:
                    break
    return seen


async def wait_for(predicate, timeout: float = 10.0) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return False


def test_abandoning_a_stream_aborts_it_and_releases_the_blocks():
    with LiveServer(num_blocks=64, max_running=2, step_delay=0.003) as live:
        seen = asyncio.run(stream_then_abandon(live.url, list(range(96)), 400, after=3))
        assert seen == 3
        aborted = asyncio.run(
            wait_for(lambda: live.engine.metrics.counters["aborted_requests"] == 1)
        )
        assert aborted, "the disconnect never reached the engine"
        assert live.engine.metrics.counters["finished_requests"] == 0
        assert live.engine.leaked_blocks() == 0


def test_repeated_cancellations_do_not_bleed_the_pool():
    """The slow leak this guards against only shows up over many cancellations, which is
    why the benchmark has a cancel mode at all."""
    with LiveServer(num_blocks=96, max_running=4, step_delay=0.001) as live:
        for index in range(25):
            asyncio.run(
                stream_then_abandon(live.url, list(range(index * 100, index * 100 + 96)), 300, after=2)
            )
        settled = asyncio.run(wait_for(lambda: not live.engine.sched.has_work()))
        assert settled
        assert live.engine.metrics.counters["aborted_requests"] == 25
        assert live.engine.leaked_blocks() == 0
        live.engine.alloc.check_invariants()


def test_metrics_endpoint_is_scrapeable():
    async def scenario():
        harness = App()
        async with harness as client:
            await completion(client, list(range(32)), 4, stream=False)
            text = (await client.get("/metrics")).text
            assert "mercury_allocator_hit_rate" in text
            assert "mercury_engine_steps" in text

    run(scenario)


def test_request_records_are_drained_once():
    async def scenario():
        harness = App()
        async with harness as client:
            await completion(client, list(range(32)), 4, stream=False)
            first = (await client.get("/stats/requests")).json()["requests"]
            second = (await client.get("/stats/requests")).json()["requests"]
            assert len(first) == 1
            assert first[0]["output_tokens"] == 4
            assert first[0]["queue_delay"] is not None
            assert second == []

    run(scenario)


def test_concurrent_requests_are_all_served():
    async def scenario():
        harness = App(max_running=4)
        async with harness as client:
            responses = await asyncio.gather(
                *(
                    completion(client, list(range(index * 50, index * 50 + 48)), 6, stream=False)
                    for index in range(12)
                )
            )
            assert all(r.status_code == 200 for r in responses)
            assert all(len(r.json()["choices"][0]["token_ids"]) == 6 for r in responses)
            await harness.wait_until_drained()
            assert harness.engine.leaked_blocks() == 0

    run(scenario)


def test_health_and_model_listing():
    async def scenario():
        harness = App(policy="lpm_aging")
        async with harness as client:
            health = (await client.get("/health")).json()
            assert health["status"] == "ok"
            assert health["policy"] == "lpm_aging"
            models = (await client.get("/v1/models")).json()
            assert models["data"][0]["id"] == "fake"

    run(scenario)
