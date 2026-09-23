"""Three implementations of the same arithmetic, timed against each other.

The honest competitor is torch.compile, not eager: Inductor emits a fused Triton kernel
for this pattern too. For a memory-bound reduction the number that says whether the kernel
is any good is the share of peak bandwidth it reaches, so that is the headline column.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mercury.kernels.rmsnorm_quant import (
    HAS_TRITON,
    add_rmsnorm_quant_torch,
    add_rmsnorm_quant_triton,
    bytes_moved,
    fused_available,
)

T4_PEAK_GB_S = 320.0
ROWS = [1, 16, 64, 256, 1024, 4096]


def timer(fn):
    if HAS_TRITON:
        import triton

        return triton.testing.do_bench(fn, warmup=50, rep=200)
    import time

    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(200):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e3 / 200


def bench_one(rows: int, cols: int, quant: bool, compiled) -> dict:
    x = torch.randn(rows, cols, device="cuda", dtype=torch.float16)
    weight = torch.randn(cols, device="cuda", dtype=torch.float16)
    residual = torch.randn_like(x)
    traffic = bytes_moved(rows, cols, True, quant)

    entries = {
        "eager": lambda: add_rmsnorm_quant_torch(x, weight, 1e-6, residual, quant),
        "compiled": lambda: compiled(x, weight, 1e-6, residual, quant),
        "triton": lambda: add_rmsnorm_quant_triton(x, weight, 1e-6, residual, quant),
    }
    out = {"rows": rows, "cols": cols, "quant": quant, "bytes": traffic}
    for name, fn in entries.items():
        try:
            milliseconds = timer(fn)
        except Exception as exc:
            out[f"{name}_us"] = None
            out[f"{name}_pct_peak"] = None
            out[f"{name}_error"] = f"{type(exc).__name__}: {exc}"[:120]
            continue
        seconds = milliseconds * 1e-3
        out[f"{name}_us"] = milliseconds * 1e3
        out[f"{name}_pct_peak"] = 100.0 * (traffic / seconds) / (T4_PEAK_GB_S * 1e9)
    return out


def table(rows: list[dict]) -> str:
    header = (
        "| M | N | quant | eager (us) | compile (us) | triton (us) | "
        "eager %peak | compile %peak | triton %peak |"
    )
    lines = [header, "| " + " | ".join("---" for _ in range(9)) + " |"]
    for row in rows:
        def cell(key, fmt=".1f"):
            value = row.get(key)
            return "–" if value is None else format(value, fmt)

        lines.append(
            f"| {row['rows']} | {row['cols']} | {'int8' if row['quant'] else 'fp16'} | "
            f"{cell('eager_us')} | {cell('compiled_us')} | {cell('triton_us')} | "
            f"{cell('eager_pct_peak')} | {cell('compiled_pct_peak')} | {cell('triton_pct_peak')} |"
        )
    return "\n".join(lines)


def decode_step_impact(model_path: str, revision: str | None, steps: int = 30) -> dict:
    """End-to-end honesty check: what the fused norm is worth inside a real decode step."""
    from mercury.config import EngineConfig
    from mercury.engine import Engine
    from mercury.request import SamplingParams

    out = {}
    for fused in (True, False):
        cfg = EngineConfig(
            model=model_path, revision=revision, device="cuda", dtype="float16",
            num_blocks=2048, max_running=32, max_model_len=2048, fused_rmsnorm=fused,
        )
        engine = Engine(cfg)
        try:
            for index in range(32):
                engine.submit(
                    f"b{index}",
                    list(range(1000 + index * 400, 1000 + index * 400 + 512)),
                    steps + 2,
                    lambda _event: None,
                    params=SamplingParams(ignore_eos=True),
                )
            engine._drain_commands()
            for _ in range(steps + 8):
                engine.step()
            summary = engine.metrics.summary()
            out["fused" if fused else "unfused"] = {
                "mean_decode_step_ms": summary["mean_decode_step_ms"],
                "forward_share": summary["forward_share"],
            }
        finally:
            engine.shutdown()

    if "fused" in out and "unfused" in out:
        a = out["unfused"]["mean_decode_step_ms"]
        b = out["fused"]["mean_decode_step_ms"]
        out["decode_speedup_pct"] = 100.0 * (a - b) / a if a else None
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser("bench.kernel_bench")
    parser.add_argument("--cols", type=int, nargs="*", default=[896, 1536])
    parser.add_argument("--rows", type=int, nargs="*", default=ROWS)
    parser.add_argument("--out", default="analysis")
    parser.add_argument("--model", default=None, help="also measure the end-to-end decode impact")
    parser.add_argument("--revision", default=None)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("this benchmark needs a CUDA device")
    if not fused_available():
        raise SystemExit("triton is not importable; nothing to compare against")

    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"{name} (sm{capability[0]}{capability[1]}), peak bandwidth assumed {T4_PEAK_GB_S} GB/s")
    if capability[0] < 8:
        print("note: below sm80 Triton's tl.dot falls back to FMA; this kernel has no dot, "
              "so only elementwise and reduction paths are exercised")

    compiled = torch.compile(add_rmsnorm_quant_torch, dynamic=True)
    rows = []
    for cols in args.cols:
        for count in args.rows:
            for quant in (False, True):
                entry = bench_one(count, cols, quant, compiled)
                rows.append(entry)
                print(
                    f"M={count:<5d} N={cols} {'int8' if quant else 'fp16'}  "
                    f"eager {entry.get('eager_us') or float('nan'):7.1f}us  "
                    f"compile {entry.get('compiled_us') or float('nan'):7.1f}us  "
                    f"triton {entry.get('triton_us') or float('nan'):7.1f}us  "
                    f"({entry.get('triton_pct_peak') or float('nan'):.0f}% of peak)",
                    flush=True,
                )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "kernel_table.md").write_text(
        f"# Fused residual-add + RMSNorm ({name})\n\n{table(rows)}\n"
    )
    payload = {"device": name, "capability": list(capability), "rows": rows}

    if args.model:
        print("\nmeasuring the end-to-end decode impact...", flush=True)
        payload["decode"] = decode_step_impact(args.model, args.revision)
        print(json.dumps(payload["decode"], indent=2))

    (out_dir / "kernel_bench.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_dir / 'kernel_table.md'}")


if __name__ == "__main__":
    main()
