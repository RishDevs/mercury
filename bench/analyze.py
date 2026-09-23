"""Turns raw per-request records into the figures, the table and the pre-registered
comparisons. Nothing here runs during a benchmark; percentiles are always computed
afterwards from the saved records, never live."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TTFT_SLO = 2.0
TPOT_SLO = 0.100
POLICY_ORDER = ["fcfs", "lpm", "lpm_aging"]
POLICY_LABEL = {"fcfs": "FCFS", "lpm": "LPM", "lpm_aging": "LPM + aging"}


def percentile(values, q: float) -> float | None:
    clean = [v for v in values if v is not None]
    return float(np.percentile(clean, q)) if clean else None


def bootstrap_ci(values, q: float, iterations: int = 2000, seed: int = 0):
    """95% CI for a percentile, by resampling the per-request values with replacement."""
    clean = np.asarray([v for v in values if v is not None], dtype=float)
    if clean.size < 8:
        return (percentile(clean, q), None, None)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, clean.size, size=(iterations, clean.size))
    stats = np.percentile(clean[draws], q, axis=1)
    return (float(np.percentile(clean, q)), float(np.percentile(stats, 2.5)),
            float(np.percentile(stats, 97.5)))


def gpu_peak_memory(path: str | None) -> float | None:
    if not path or not Path(path).exists():
        return None
    peak = 0.0
    for line in Path(path).read_text(errors="replace").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            peak = max(peak, float(parts[1]))
        except ValueError:
            continue
    return peak or None


@dataclass
class Run:
    meta: dict
    ttft: list[float]
    tpot: list[float]
    e2e: list[float]
    itl: list[float]
    queue_delay: list[float]
    row: dict


def summarize(path: Path) -> Run:
    result = json.loads(Path(path).read_text())
    point = result.get("point", {})
    records = [r for r in result["requests"] if r["status"] == "ok"]

    ttft, tpot, e2e, itl = [], [], [], []
    met_slo = 0
    for r in records:
        if r["first_token"] is None:
            continue
        time_to_first = r["first_token"] - r["sent"]
        ttft.append(time_to_first)
        e2e.append(r["last_token"] - r["sent"])
        per_token = None
        if r["output_tokens"] >= 2:
            per_token = (r["last_token"] - r["first_token"]) / (r["output_tokens"] - 1)
            tpot.append(per_token)
        times = r["token_times"]
        itl.extend(b - a for a, b in zip(times, times[1:]))
        if time_to_first <= TTFT_SLO and (per_token is None or per_token <= TPOT_SLO):
            met_slo += 1

    server = result.get("server_stats") or {}
    allocator = server.get("allocator", {})
    engine_stats = server.get("engine", {})
    scheduler = server.get("scheduler", {})
    server_requests = result.get("server_requests") or []
    queue_delay = [
        r["queue_delay"] for r in server_requests if r.get("queue_delay") is not None
    ]

    wall = result["wall_seconds"] or 1e-9
    output_tokens = sum(r["output_tokens"] for r in records)

    row = {
        "engine": point.get("engine", result.get("engine")),
        "policy": point.get("policy", "fcfs"),
        "workload": point.get("workload"),
        "rho": point.get("rho"),
        "rate": point.get("rate"),
        "seed": point.get("seed"),
        "blocks": point.get("blocks"),
        "kv_capacity_tokens": point.get("kv_capacity_tokens"),
        "requests_ok": len(records),
        "requests_total": len(result["requests"]),
        "hit_rate": allocator.get("hit_rate"),
        "evictions_per_s": (allocator.get("evictions", 0) / wall) if allocator else None,
        "preemptions": scheduler.get("preemptions"),
        "ttft_p50": percentile(ttft, 50),
        "ttft_p95": percentile(ttft, 95),
        "ttft_p99": percentile(ttft, 99),
        "tpot_p50": percentile(tpot, 50),
        "tpot_p95": percentile(tpot, 95),
        "itl_p99": percentile(itl, 99),
        "e2e_p95": percentile(e2e, 95),
        "queue_p99": percentile(queue_delay, 99),
        "queue_max": max(queue_delay) if queue_delay else None,
        "output_tok_per_s": output_tokens / wall,
        "req_per_s": len(records) / wall,
        "goodput_req_per_s": met_slo / wall,
        "sched_share": engine_stats.get("sched_share"),
        "prepare_share": engine_stats.get("prepare_share"),
        "mean_sched_us": engine_stats.get("mean_sched_us"),
        "mean_step_ms": engine_stats.get("mean_step_ms"),
        "peak_gpu_mib": gpu_peak_memory((result.get("provenance") or {}).get("gpu_log")),
        "client_cpu": result.get("client_cpu_utilisation"),
        "verification": "; ".join(result.get("verification") or []) or "clean",
        "path": str(path),
    }
    return Run(meta=result, ttft=ttft, tpot=tpot, e2e=e2e, itl=itl, queue_delay=queue_delay, row=row)


def load_runs(results_dir: Path) -> list[Run]:
    runs = []
    for path in sorted(Path(results_dir).glob("*.json")):
        if path.name in {"grid.json", "rate_sweep.json"}:
            continue
        try:
            runs.append(summarize(path))
        except (KeyError, json.JSONDecodeError):
            continue
    return runs


def group(runs: list[Run], *keys: str) -> dict[tuple, list[Run]]:
    out: dict[tuple, list[Run]] = {}
    for run in runs:
        out.setdefault(tuple(run.row.get(k) for k in keys), []).append(run)
    return out


def aggregate(runs: list[Run], field: str) -> dict:
    """Median across repetitions, which is what the table reports, plus the spread."""
    values = [r.row[field] for r in runs if r.row.get(field) is not None]
    if not values:
        return {"median": None, "low": None, "high": None, "n": 0}
    return {
        "median": float(np.median(values)),
        "low": float(np.min(values)),
        "high": float(np.max(values)),
        "n": len(values),
    }


def pooled(runs: list[Run], attribute: str) -> list[float]:
    out: list[float] = []
    for run in runs:
        out.extend(getattr(run, attribute))
    return out


def figure_pressure_curves(runs: list[Run], out_dir: Path, workloads=("w2", "w4")) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    mercury = [r for r in runs if r.row["engine"] == "mercury"]

    for name, field, ylabel, filename in [
        ("Prefix hit rate", "hit_rate", "prompt tokens served from cache", "fig1_hitrate.png"),
        ("p95 time to first token", "ttft_p95", "seconds", "fig2_ttft.png"),
    ]:
        panels = [w for w in workloads if any(r.row["workload"] == w for r in mercury)]
        if not panels:
            continue
        extra = 1 if field == "ttft_p95" else 0
        fig, axes = plt.subplots(
            1, len(panels) + extra, figsize=(5 * (len(panels) + extra), 4), squeeze=False
        )
        for column, workload in enumerate(panels):
            axis = axes[0][column]
            for policy in POLICY_ORDER:
                selected = [
                    r for r in mercury if r.row["workload"] == workload and r.row["policy"] == policy
                ]
                by_rho = group(selected, "rho")
                xs = sorted(k[0] for k in by_rho if k[0] is not None)
                if not xs:
                    continue
                stats = [aggregate(by_rho[(x,)], field) for x in xs]
                ys = [s["median"] for s in stats]
                lo = [s["low"] for s in stats]
                hi = [s["high"] for s in stats]
                axis.plot(xs, ys, marker="o", label=POLICY_LABEL[policy])
                axis.fill_between(xs, lo, hi, alpha=0.15)
            axis.set_xscale("log")
            axis.set_xlabel("cache pressure rho = shared prefix tokens / KV capacity")
            axis.set_ylabel(ylabel)
            axis.set_title(f"{name}: {workload.upper()}")
            axis.axvline(1.0, color="grey", linestyle=":", linewidth=1)
            axis.legend()
            axis.grid(alpha=0.25)

        if extra:
            axis = axes[0][len(panels)]
            for policy in POLICY_ORDER:
                selected = [r for r in mercury if r.row["policy"] == policy and r.row["workload"] in panels]
                by_rho = group(selected, "rho")
                xs = sorted(k[0] for k in by_rho if k[0] is not None)
                if not xs:
                    continue
                ys = [aggregate(by_rho[(x,)], "queue_max")["median"] for x in xs]
                axis.plot(xs, ys, marker="s", label=POLICY_LABEL[policy])
            axis.set_xscale("log")
            axis.set_xlabel("cache pressure rho")
            axis.set_ylabel("seconds")
            axis.set_title("Worst-case queueing delay")
            axis.legend()
            axis.grid(alpha=0.25)

        fig.tight_layout()
        path = out_dir / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)
    return written


def markdown_table(rows: list[dict], columns: list[tuple[str, str, str]]) -> str:
    header = "| " + " | ".join(title for title, _, _ in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    for row in rows:
        cells = []
        for _title, key, fmt in columns:
            value = row.get(key)
            cells.append("–" if value is None else format(value, fmt) if fmt else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main_table(runs: list[Run], workload: str, rho: float, rate: float | None) -> str:
    selected = [
        r for r in runs
        if r.row["workload"] == workload
        and r.row["rho"] == rho
        and (rate is None or r.row["rate"] == rate)
    ]
    rows = []
    for key, bucket in sorted(group(selected, "engine", "policy").items()):
        engine, policy = key
        label = f"{engine} {POLICY_LABEL.get(policy, policy)}"
        rows.append(
            {
                "label": label,
                "hit_rate": aggregate(bucket, "hit_rate")["median"],
                "ttft_p95": aggregate(bucket, "ttft_p95")["median"],
                "queue_p99": aggregate(bucket, "queue_p99")["median"],
                "queue_max": aggregate(bucket, "queue_max")["median"],
                "output_tok_per_s": aggregate(bucket, "output_tok_per_s")["median"],
                "goodput_req_per_s": aggregate(bucket, "goodput_req_per_s")["median"],
                "sched_share": aggregate(bucket, "sched_share")["median"],
                "peak_gpu_mib": aggregate(bucket, "peak_gpu_mib")["median"],
            }
        )
    return markdown_table(
        rows,
        [
            ("Engine / policy", "label", ""),
            ("Hit rate", "hit_rate", ".3f"),
            ("p95 TTFT (s)", "ttft_p95", ".3f"),
            ("p99 queue (s)", "queue_p99", ".3f"),
            ("Max queue (s)", "queue_max", ".3f"),
            ("Output tok/s", "output_tok_per_s", ".0f"),
            ("Goodput (req/s)", "goodput_req_per_s", ".2f"),
            ("Sched. overhead", "sched_share", ".4f"),
            ("Peak GPU (MiB)", "peak_gpu_mib", ".0f"),
        ],
    )


def preregistered(runs: list[Run], rho: float = 2.0) -> str:
    """The four comparisons stated in the README before E1 was run."""
    lines = ["# Pre-registered comparisons", ""]

    def pick(policy, workload, at_rho):
        return [
            r for r in runs
            if r.row["engine"] == "mercury"
            and r.row["policy"] == policy
            and r.row["workload"] == workload
            and r.row["rho"] == at_rho
        ]

    lines.append(f"## 1. lpm_aging vs fcfs at rho = {rho:g} (the headline)")
    for workload in ("w2", "w4"):
        base, test = pick("fcfs", workload, rho), pick("lpm_aging", workload, rho)
        if not base or not test:
            lines.append(f"- {workload.upper()}: not enough runs")
            continue
        for field, label, better in [
            ("hit_rate", "hit rate", "higher"),
            ("ttft_p95", "p95 TTFT", "lower"),
        ]:
            a, b = aggregate(base, field)["median"], aggregate(test, field)["median"]
            if a is None or b is None:
                continue
            change = (b - a) / a * 100 if a else float("nan")
            attribute = "ttft" if field == "ttft_p95" else None
            if attribute:
                point, lo, hi = bootstrap_ci(pooled(test, attribute), 95)
                _, blo, bhi = bootstrap_ci(pooled(base, attribute), 95)
                lines.append(
                    f"- {workload.upper()} {label}: {a:.3f} [{blo:.3f}, {bhi:.3f}] -> "
                    f"{b:.3f} [{lo:.3f}, {hi:.3f}] ({change:+.1f}%, {better} is better)"
                )
            else:
                lines.append(f"- {workload.upper()} {label}: {a:.3f} -> {b:.3f} ({change:+.1f}%, {better} is better)")

    lines += ["", "## 2. lpm vs lpm_aging on worst-case queueing (the fairness claim)"]
    for workload in ("w2", "w4"):
        plain, aged = pick("lpm", workload, rho), pick("lpm_aging", workload, rho)
        if not plain or not aged:
            lines.append(f"- {workload.upper()}: not enough runs")
            continue
        a = aggregate(plain, "queue_max")["median"]
        b = aggregate(aged, "queue_max")["median"]
        ha, hb = aggregate(plain, "hit_rate")["median"], aggregate(aged, "hit_rate")["median"]
        if a is None or b is None:
            lines.append(f"- {workload.upper()}: no server-side queue records")
            continue
        lines.append(
            f"- {workload.upper()}: max queueing {a:.2f}s -> {b:.2f}s, "
            f"hit rate {ha:.3f} -> {hb:.3f}"
        )

    lines += ["", "## 3. All policies within noise below capacity (the sanity check)"]
    for workload in ("w2", "w4"):
        values = {
            policy: aggregate(pick(policy, workload, 0.5), "hit_rate")["median"]
            for policy in POLICY_ORDER
        }
        present = {k: v for k, v in values.items() if v is not None}
        if len(present) < 2:
            lines.append(f"- {workload.upper()}: not enough runs at rho = 0.5")
            continue
        spread = max(present.values()) - min(present.values())
        verdict = "PASS" if spread < 0.02 else "FAIL - the headline is suspect"
        lines.append(
            f"- {workload.upper()} at rho = 0.5: hit rates "
            + ", ".join(f"{POLICY_LABEL[k]} {v:.3f}" for k, v in present.items())
            + f" (spread {spread:.3f}) {verdict}"
        )

    lines += ["", "## 4. No harm on the no-sharing control (W1)"]
    base, test = pick("fcfs", "w1", 1.0), pick("lpm_aging", "w1", 1.0)
    if base and test:
        a = aggregate(base, "output_tok_per_s")["median"]
        b = aggregate(test, "output_tok_per_s")["median"]
        ratio = b / a if a else float("nan")
        verdict = "PASS" if ratio >= 0.97 else "FAIL"
        lines.append(f"- throughput {a:.0f} -> {b:.0f} tok/s ({ratio:.3f} of FCFS) {verdict}")
    else:
        lines.append("- not enough W1 runs")
    return "\n".join(lines)


def write_csv(runs: list[Run], path: Path) -> None:
    import csv

    rows = [r.row for r in runs]
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser("bench.analyze")
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", default="analysis")
    parser.add_argument("--table-workload", default="w2")
    parser.add_argument("--table-rho", type=float, default=2.0)
    parser.add_argument("--table-rate", type=float, default=None)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    runs = load_runs(Path(args.results))
    if not runs:
        raise SystemExit(f"no result files under {args.results}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dirty = [r for r in runs if r.row["verification"] != "clean"]
    print(f"{len(runs)} runs loaded, {len(dirty)} with verification warnings")
    for run in dirty:
        print(f"  {Path(run.row['path']).name}: {run.row['verification']}")

    write_csv(runs, out_dir / "runs.csv")
    table = main_table(runs, args.table_workload, args.table_rho, args.table_rate)
    (out_dir / "main_table.md").write_text(
        f"# {args.table_workload.upper()}, rho = {args.table_rho:g}, median of repeats\n\n{table}\n"
    )
    (out_dir / "preregistered.md").write_text(preregistered(runs, args.table_rho) + "\n")
    print("\n" + table + "\n")
    print(preregistered(runs, args.table_rho))

    if not args.no_figures:
        for path in figure_pressure_curves(runs, out_dir):
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
