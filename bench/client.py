"""Open-loop trace replayer.

One client for every engine. Mercury, vLLM and SGLang all speak a streaming
OpenAI-compatible /v1/completions, so the same code drives all three and client
differences drop out of the comparison.

Open-loop matters: a closed-loop client with a fixed number of users caps offered load at
whatever the server can do, so queueing never builds and the entire scheduling effect
disappears. Here every request is sent at its scheduled arrival time whether or not
earlier ones have finished. The exception is W3, where turns inside a session genuinely
wait for the previous answer, because that is how agents behave.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import resource
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import aiohttp

from .workloads import TraceRequest, load_trace


@dataclass
class RequestRecord:
    rid: str
    session_id: str
    turn: int
    scheduled_arrival: float
    sent: float = 0.0
    first_token: float | None = None
    last_token: float | None = None
    token_times: list[float] = field(default_factory=list)
    chunk_tokens: list[int] = field(default_factory=list)
    prompt_tokens: int = 0
    output_tokens: int = 0
    expected_tokens: int = 0
    reported_tokens: int | None = None
    finish_reason: str | None = None
    status: str = "ok"
    error: str | None = None

    @property
    def ttft(self) -> float | None:
        return None if self.first_token is None else self.first_token - self.sent

    @property
    def e2e(self) -> float | None:
        return None if self.last_token is None else self.last_token - self.sent

    @property
    def tpot(self) -> float | None:
        if self.first_token is None or self.last_token is None or self.output_tokens < 2:
            return None
        return (self.last_token - self.first_token) / (self.output_tokens - 1)

    def inter_token_gaps(self) -> list[float]:
        return [b - a for a, b in zip(self.token_times, self.token_times[1:])]


class Replayer:
    def __init__(
        self,
        base_url: str,
        model: str,
        engine: str = "mercury",
        cancel_fraction: float = 0.0,
        cancel_at: float = 0.5,
        request_timeout: float = 600.0,
        seed: int = 0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.engine = engine
        self.cancel_fraction = cancel_fraction
        self.cancel_at = cancel_at
        self.request_timeout = request_timeout
        self.records: dict[str, RequestRecord] = {}
        self.t0 = 0.0
        import random

        self.rng = random.Random(seed)

    def _payload(self, request: TraceRequest) -> dict:
        body = {
            "model": self.model,
            "prompt": request.prompt_token_ids,
            "max_tokens": request.max_tokens,
            "temperature": 0.0,
            "stream": True,
            "ignore_eos": True,
        }
        if self.engine == "mercury":
            body["request_id"] = request.rid
            body["detokenize"] = False
            body["session_id"] = request.session_id
        else:
            body["stream_options"] = {"include_usage": True}
        return body

    async def _one(self, session: aiohttp.ClientSession, request: TraceRequest) -> RequestRecord:
        record = RequestRecord(
            rid=request.rid,
            session_id=request.session_id,
            turn=request.turn,
            scheduled_arrival=request.arrival_time_s or 0.0,
            prompt_tokens=len(request.prompt_token_ids),
            expected_tokens=request.max_tokens,
        )
        self.records[request.rid] = record
        cancel_after = None
        if self.cancel_fraction and self.rng.random() < self.cancel_fraction:
            cancel_after = max(1, int(request.max_tokens * self.cancel_at))

        record.sent = time.perf_counter() - self.t0
        try:
            async with session.post(
                f"{self.base_url}/v1/completions",
                json=self._payload(request),
                timeout=aiohttp.ClientTimeout(total=self.request_timeout),
            ) as response:
                if response.status != 200:
                    record.status = "error"
                    record.error = f"HTTP {response.status}: {(await response.text())[:200]}"
                    return record
                async for raw in response.content:
                    line = raw.strip()
                    if not line.startswith(b"data:"):
                        continue
                    body = line[5:].strip()
                    if body == b"[DONE]":
                        break
                    chunk = json.loads(body)
                    if "error" in chunk:
                        record.status = "error"
                        record.error = str(chunk["error"])[:200]
                        break
                    now = time.perf_counter() - self.t0
                    usage = chunk.get("usage")
                    if usage and usage.get("completion_tokens") is not None:
                        record.reported_tokens = usage["completion_tokens"]
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    produced = len(choice.get("token_ids") or []) or (
                        1 if choice.get("text") else 0
                    )
                    if produced:
                        if record.first_token is None:
                            record.first_token = now
                        record.last_token = now
                        record.token_times.append(now)
                        record.chunk_tokens.append(produced)
                        record.output_tokens += produced
                    if choice.get("finish_reason"):
                        record.finish_reason = choice["finish_reason"]
                    if cancel_after is not None and record.output_tokens >= cancel_after:
                        record.status = "cancelled"
                        response.close()
                        return record
        except asyncio.TimeoutError:
            record.status = "timeout"
            record.error = "client timeout"
        except aiohttp.ClientError as exc:
            record.status = "error"
            record.error = f"{type(exc).__name__}: {exc}"[:200]
        return record

    async def _session_task(
        self, session: aiohttp.ClientSession, turns: list[TraceRequest]
    ) -> None:
        for request in turns:
            if request.arrival_time_s is not None:
                delay = request.arrival_time_s - (time.perf_counter() - self.t0)
                if delay > 0:
                    await asyncio.sleep(delay)
            elif request.think_time_s:
                await asyncio.sleep(request.think_time_s)
            await self._one(session, request)

    async def replay(self, requests: list[TraceRequest]) -> list[RequestRecord]:
        by_session: dict[str, list[TraceRequest]] = {}
        for request in requests:
            by_session.setdefault(request.session_id, []).append(request)
        for turns in by_session.values():
            turns.sort(key=lambda r: r.turn)

        connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
        async with aiohttp.ClientSession(connector=connector) as session:
            self.t0 = time.perf_counter()
            await asyncio.gather(
                *(self._session_task(session, turns) for turns in by_session.values())
            )
        return [self.records[r.rid] for r in requests if r.rid in self.records]

    async def warmup(self, count: int = 50, prompt_tokens: int = 512, max_tokens: int = 8) -> None:
        """Deliberately off-trace prompts: the server ends up warm, the prefix cache ends
        up holding nothing the measured run will ask for."""
        connector = aiohttp.TCPConnector(limit=0)
        async with aiohttp.ClientSession(connector=connector) as session:
            self.t0 = time.perf_counter()
            fillers = [
                TraceRequest(
                    rid=f"warmup-{i}",
                    session_id="warmup",
                    turn=0,
                    prompt_token_ids=[900_000_007 % 90_000 + (i * 7919 + j) % 90_000 + 1000
                                      for j in range(prompt_tokens)],
                    max_tokens=max_tokens,
                    arrival_time_s=0.0,
                )
                for i in range(count)
            ]
            await asyncio.gather(*(self._one(session, f) for f in fillers))
        self.records.clear()


async def fetch_json(url: str) -> dict | None:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status != 200:
                    return None
                return await response.json()
    except Exception:
        return None


async def post(url: str) -> bool:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                return response.status < 400
    except Exception:
        return False


async def wait_for_health(base_url: str, timeout: float = 600.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        for path in ("/health", "/v1/models"):
            if await fetch_json(f"{base_url.rstrip('/')}{path}") is not None:
                return True
        await asyncio.sleep(1.0)
    return False


def cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


async def run_once(
    base_url: str,
    trace_path: Path,
    model: str,
    engine: str = "mercury",
    warmup_requests: int = 50,
    cancel_fraction: float = 0.0,
    seed: int = 0,
) -> dict:
    requests, meta = load_trace(trace_path)
    replayer = Replayer(base_url, model, engine=engine, cancel_fraction=cancel_fraction, seed=seed)

    if warmup_requests:
        await replayer.warmup(warmup_requests)
    if engine == "mercury":
        await post(f"{base_url.rstrip('/')}/reset_stats")

    cpu_before = cpu_seconds()
    wall_before = time.perf_counter()
    records = await replayer.replay(requests)
    wall = time.perf_counter() - wall_before
    cpu = cpu_seconds() - cpu_before

    server_stats = await fetch_json(f"{base_url.rstrip('/')}/stats")
    server_requests = None
    if engine == "mercury":
        payload = await fetch_json(f"{base_url.rstrip('/')}/stats/requests")
        server_requests = (payload or {}).get("requests")

    return {
        "trace": str(trace_path),
        "trace_meta": asdict(meta),
        "engine": engine,
        "model": model,
        "base_url": base_url,
        "wall_seconds": wall,
        "client_cpu_seconds": cpu,
        "client_cpu_utilisation": cpu / wall if wall else 0.0,
        "cancel_fraction": cancel_fraction,
        "requests": [asdict(r) for r in records],
        "server_stats": server_stats,
        "server_requests": server_requests,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser("bench.client")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--engine", default="mercury", choices=("mercury", "vllm", "sglang"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--warmup-requests", type=int, default=50)
    parser.add_argument("--cancel-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    result = asyncio.run(
        run_once(
            args.base_url,
            Path(args.trace),
            args.model,
            engine=args.engine,
            warmup_requests=args.warmup_requests,
            cancel_fraction=args.cancel_fraction,
            seed=args.seed,
        )
    )
    ok = [r for r in result["requests"] if r["status"] == "ok"]
    mismatched = [r for r in ok if r["output_tokens"] != r["expected_tokens"]]
    print(f"{len(ok)}/{len(result['requests'])} requests ok, {len(mismatched)} with wrong token counts")
    print(f"client CPU {result['client_cpu_utilisation'] * 100:.0f}% of one core over {result['wall_seconds']:.1f}s")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
