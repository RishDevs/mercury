from __future__ import annotations

import argparse
import asyncio
import gc
import json
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from .config import EngineConfig
from .engine import Engine, StreamEvent
from .request import SamplingParams


class IncrementalDetokenizer:
    """Decodes the whole output each time and emits the delta; only used when a client
    actually wants text. The benchmark client sends and receives token ids."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.ids: list[int] = []
        self.text = ""

    def push(self, token: int) -> str:
        self.ids.append(token)
        full = self.tokenizer.decode(self.ids, skip_special_tokens=False)
        delta = full[len(self.text) :]
        self.text = full
        return delta


def _chunk(rid: str, model: str, created: int, text: str, token: int | None, reason: str | None) -> str:
    payload = {
        "id": rid,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "text": text,
                "token_ids": [] if token is None else [token],
                "logprobs": None,
                "finish_reason": reason,
            }
        ],
    }
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def create_app(cfg: EngineConfig, tokenizer=None, engine: Engine | None = None) -> FastAPI:
    engine = engine or Engine(cfg)
    if tokenizer is None:
        path = getattr(engine.runner, "model_path", None)
        if path:
            from .model.loader import load_tokenizer

            tokenizer = load_tokenizer(path)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        engine.start()
        # Everything alive now is long-lived; keep the collector from walking it every gen0 pass.
        gc.collect()
        gc.freeze()
        yield
        engine.shutdown()

    app = FastAPI(title="mercury", version="0.1.0", lifespan=lifespan)
    app.state.engine = engine
    app.state.tokenizer = tokenizer

    def resolve_prompt(raw) -> list[int]:
        if isinstance(raw, str):
            if tokenizer is None:
                raise HTTPException(400, "this server has no tokenizer; send prompt as token ids")
            return tokenizer.encode(raw)
        if isinstance(raw, list) and raw and all(isinstance(t, int) for t in raw):
            return list(raw)
        if isinstance(raw, list) and raw and isinstance(raw[0], (list, str)):
            raise HTTPException(400, "batched prompts are not supported; send one per request")
        raise HTTPException(400, "prompt must be a string or a list of token ids")

    @app.get("/health")
    async def health():
        return {"status": "ok", "policy": cfg.policy}

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [{"id": cfg.served_model_name, "object": "model", "owned_by": "mercury"}],
        }

    @app.get("/stats")
    async def stats():
        return JSONResponse(engine.snapshot())

    @app.get("/stats/requests")
    async def request_records():
        return JSONResponse({"requests": engine.drain_request_records()})

    @app.post("/reset_stats")
    async def reset_stats():
        """Clears counters without flushing the KV cache, which is what the warmup
        protocol needs: the cache holds non-trace blocks, the numbers start at zero."""
        engine.alloc.reset_stats()
        for key in engine.sched.stats:
            engine.sched.stats[key] = 0 if isinstance(engine.sched.stats[key], int) else 0.0
        engine.metrics.steps.clear()
        engine.metrics.finished.clear()
        for key in engine.metrics.counters:
            engine.metrics.counters[key] = 0
        return {"status": "reset"}

    @app.get("/metrics")
    async def metrics():
        snap = engine.snapshot()
        lines = []
        for group in ("allocator", "engine"):
            for key, value in snap[group].items():
                if isinstance(value, (int, float)):
                    lines.append(f"mercury_{group}_{key} {value}")
        lines.append(f"mercury_scheduler_waiting {snap['scheduler']['waiting']}")
        lines.append(f"mercury_scheduler_running {snap['scheduler']['running']}")
        return PlainTextResponse("\n".join(lines) + "\n")

    @app.post("/v1/completions")
    async def completions(request: Request):
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, "body must be JSON")

        prompt_ids = resolve_prompt(body.get("prompt"))
        max_tokens = int(body.get("max_tokens", 16))
        if max_tokens < 1:
            raise HTTPException(400, "max_tokens must be at least 1")
        stream = bool(body.get("stream", False))
        want_text = bool(body.get("detokenize", isinstance(body.get("prompt"), str)))
        params = SamplingParams(
            temperature=float(body.get("temperature", 0.0)),
            top_p=float(body.get("top_p", 1.0)),
            ignore_eos=bool(body.get("ignore_eos", False)),
        )
        rid = body.get("request_id") or f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        model_name = cfg.served_model_name

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[StreamEvent] = asyncio.Queue()

        def sink(event: StreamEvent) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, event)

        detok = IncrementalDetokenizer(tokenizer) if (want_text and tokenizer) else None
        engine.submit(
            rid,
            prompt_ids,
            max_tokens,
            sink,
            params=params,
            session_id=body.get("session_id"),
        )

        if stream:

            async def event_stream():
                settled = False
                try:
                    while True:
                        event = await queue.get()
                        if event.kind == "token":
                            text = detok.push(event.token) if detok else ""
                            yield _chunk(rid, model_name, created, text, event.token, None)
                            if await request.is_disconnected():
                                return
                        elif event.kind == "done":
                            settled = True
                            yield _chunk(rid, model_name, created, "", None, event.finish_reason)
                            yield "data: [DONE]\n\n"
                            return
                        elif event.kind == "error":
                            settled = True
                            yield f"data: {json.dumps({'error': event.message})}\n\n"
                            return
                finally:
                    if not settled:
                        engine.abort(rid)

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        tokens: list[int] = []
        text_parts: list[str] = []
        reason = "length"
        settled = False
        try:
            while True:
                event = await queue.get()
                if event.kind == "token":
                    tokens.append(event.token)
                    if detok:
                        text_parts.append(detok.push(event.token))
                elif event.kind == "done":
                    settled = True
                    reason = event.finish_reason or "length"
                    break
                elif event.kind == "error":
                    settled = True
                    raise HTTPException(400, event.message or "request rejected")
        finally:
            if not settled:
                engine.abort(rid)

        return JSONResponse(
            {
                "id": rid,
                "object": "text_completion",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "text": "".join(text_parts),
                        "token_ids": tokens,
                        "logprobs": None,
                        "finish_reason": reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": len(tokens),
                    "total_tokens": len(prompt_ids) + len(tokens),
                },
            }
        )

    return app


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("mercury-serve")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--revision", default=None, help="pin the exact weights commit")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float16", help="T4 is fp16 only")
    p.add_argument("--policy", default="fcfs", choices=("fcfs", "lpm", "lpm_aging"))
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--num-blocks", type=int, default=4096, help="-1 sizes to free memory")
    p.add_argument("--max-running", type=int, default=64)
    p.add_argument("--prefill-budget", type=int, default=8192)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--aging-tokens-per-s", type=float, default=2000.0)
    p.add_argument("--max-wait-s", type=float, default=5.0)
    p.add_argument("--scan-limit", type=int, default=128)
    p.add_argument("--no-fused-rmsnorm", action="store_true")
    p.add_argument("--cryptographic-hashes", action="store_true")
    p.add_argument("--debug-invariants", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-level", default="warning")
    p.add_argument(
        "--fake-runner",
        action="store_true",
        help="serve deterministic tokens with no model loaded, for exercising the serving "
             "stack and the benchmark client without a GPU",
    )
    p.add_argument(
        "--fake-step-delay",
        type=float,
        default=0.0,
        help="seconds of simulated GPU time per step for --fake-runner; without it the "
             "queue never builds and no scheduling policy can do anything",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    args = build_parser().parse_args(argv)
    cfg = EngineConfig(
        model=args.model,
        revision=args.revision,
        device=args.device,
        dtype=args.dtype,
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        max_running=args.max_running,
        prefill_budget=args.prefill_budget,
        max_model_len=args.max_model_len,
        policy=args.policy,
        aging_tokens_per_s=args.aging_tokens_per_s,
        max_wait_s=args.max_wait_s,
        scan_limit=args.scan_limit,
        fused_rmsnorm=not args.no_fused_rmsnorm,
        cryptographic_hashes=args.cryptographic_hashes,
        debug_invariants=args.debug_invariants,
        seed=args.seed,
    )

    if cfg.num_blocks < 0:
        # Convenience for interactive use only. The experiment always sets capacity by hand,
        # because the point of the study is to control the pressure regime.
        import torch

        from .config import ModelConfig, autosize_blocks
        from .model.loader import resolve_model

        path, _ = resolve_model(cfg.model, cfg.revision)
        model_cfg = ModelConfig.from_directory(path)
        weights = sum(f.stat().st_size for f in path.glob("*.safetensors"))
        free, _total = torch.cuda.mem_get_info(0)
        cfg.num_blocks = autosize_blocks(model_cfg, cfg, reserve_bytes=weights)
        print(
            f"[mercury] autosized to {cfg.num_blocks} blocks "
            f"({free / 2**30:.1f} GiB free, {weights / 2**30:.1f} GiB of weights reserved)",
            flush=True,
        )

    engine = None
    if args.fake_runner:
        from .testing import DeterministicRunner

        engine = Engine(
            cfg,
            runner=DeterministicRunner(
                block_size=cfg.block_size, step_delay=args.fake_step_delay
            ),
        )

    app = create_app(cfg, engine=engine)
    engine = app.state.engine
    print(
        f"[mercury] policy={cfg.policy} kv_blocks={cfg.num_blocks} block_size={cfg.block_size} "
        f"kv_capacity_tokens={cfg.kv_capacity_tokens} kv_bytes={engine.runner.kv_bytes} "
        f"max_running={cfg.max_running} prefill_budget={cfg.prefill_budget} "
        f"model={cfg.model} revision={engine.runner.revision}",
        flush=True,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level, access_log=False)


if __name__ == "__main__":
    main()
