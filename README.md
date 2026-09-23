# Mercury

A from-scratch PyTorch inference server for a 0.5B model on one T4: paged KV cache, prefix
caching, continuous batching and streaming, written to answer one measurement question.

**The question.** When the KV cache cannot hold every shared prefix at once, does a
prefix-aware scheduler with a starvation bound beat first-come-first-served on prefix-cache
hit rate and p95 time-to-first-token — and what does the starvation bound cost?

## What is and is not claimed

Prefix-aware scheduling is not new. SGLang ships a longest-prefix-match policy
(`--schedule-policy lpm`) next to FCFS. An open SGLang issue documents that LPM can starve
cache-cold requests indefinitely and proposes an aging term to bound it. A 2026 paper,
Feather, measures prefix-aware schedulers in vLLM and SGLang and finds their radix-tree
prefix detection can cost CPU time comparable to GPU execution.

So Mercury does not claim to have invented prefix-aware scheduling. It claims four things:

1. The whole stack is built here, so every number is explainable down to the allocator.
2. A clean measurement of *when* prefix-aware scheduling helps, as a function of cache
   pressure, on commodity hardware.
3. An LPM variant with an explicit starvation bound, measured on the fairness cost that
   plain LPM hides.
4. Scheduler CPU time reported as a first-class metric, because the literature says it can
   eat the gain.

**What a T4 can show and what it cannot.** It can show scheduling effects: hit rate, TTFT,
queueing delay and throughput, all relative between policies on the same engine. It cannot
show that Mercury is fast in absolute terms. Mercury's attention runs through PyTorch
gathers rather than fused paged-attention kernels, so vLLM will beat it on raw throughput.
That is expected. The comparison that carries the finding is Mercury-FCFS against
Mercury-prefix-aware; vLLM and SGLang are reference points, and SGLang's own FCFS-vs-LPM
pair is an external check on the direction of the effect.

## The hypothesis, stated before the experiment

Prefix-aware scheduling should matter only when the KV cache cannot hold all live prefixes
at once. Cache pressure is

    rho = distinct shared-prefix tokens live in the workload / KV cache capacity in tokens

- **rho < 1**: all three policies have about the same hit rate and p95 TTFT. A large win
  here would mean a bug, not a result.
- **rho about 1 to 3**: FCFS interleaves prefix groups, evicts each prefix just before the
  next request that needs it arrives, and recomputes it. Its hit rate falls off a cliff
  while prefix-aware holds. p95 TTFT separates.
- **rho much greater than 1**: both degrade, because even perfect grouping cannot keep
  enough prefixes resident, and the gap narrows again.

### Pre-registered comparisons

Stated here before E1 was run, and computed verbatim by `bench/analyze.py`:

1. `lpm_aging` vs `fcfs` on hit rate and p95 TTFT at rho = 2, on W2 and W4, at 0.8 x R_sat.
   **This is the headline.**
2. `lpm` vs `lpm_aging` on maximum queueing delay. **This is the fairness claim.**
3. All policies at rho = 0.5 within noise of each other. **This is the sanity check, and
   the headline is suspect if it fails.**
4. On W1, which has no sharing, `lpm_aging` throughput within 3% of `fcfs`. **This is the
   does-no-harm check.**

### Why capacity is set by hand

Qwen2.5-0.5B's KV cache costs about 12 KB per token (24 layers x 2 KV heads x 64 dims x K
and V x 2 bytes). With roughly 10 GB free on a T4 that is about 850,000 tokens, which no
realistic prefix working set fills. The pool is therefore capped explicitly — Mercury's
`--num-blocks`, vLLM's `--num-gpu-blocks-override`, SGLang's `--max-total-tokens` — to put
the system in each pressure regime. This is standard and it emulates a bigger model or a
busier GPU where KV memory really is scarce. `bench/run_matrix.py` reads each engine's
startup log and refuses to run a point whose capacity cap did not take effect.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[bench,hf,dev]"
```

`torch` brings its own Triton on Linux. On a T4 use the Triton that ships with the pinned
torch; Turing support is best-effort upstream, so pin back rather than debug the compiler
if an upgrade breaks compilation. About 40% of the code — allocator, scheduler, workload
generator, metrics — needs no GPU at all and is covered by tests that run on a laptop.

## Run it

Serve:

```bash
python -m mercury.server --model Qwen/Qwen2.5-0.5B-Instruct --dtype float16 \
  --policy lpm_aging --num-blocks 4096 --max-running 64 --port 8000
```

Ask it something:

```bash
curl -s localhost:8000/v1/completions -H 'content-type: application/json' \
  -d '{"prompt":"Explain paged attention in two sentences.","max_tokens":64,"stream":true}'
```

The endpoint is OpenAI-compatible, accepts a token-id list as `prompt`, and adds two
fields: `token_ids` on each streamed choice, and `detokenize` on the request, so the
benchmark client never puts a tokenizer in the measurement loop.

### Check the harness before spending GPU quota

The entire pipeline runs with a deterministic runner and no weights:

```bash
python -m bench.run_matrix --grid bench/grids/smoke.yaml --out results/smoke
python -m bench.analyze --results results/smoke --out analysis/smoke
```

Those numbers describe the scheduler, not the GPU. Never report them as performance. They
are still useful: the smoke matrix is what caught that `alpha` has to be scaled to the
prefix length rather than hard-coded, and `bench/grids/alpha.yaml` sweeps it without costing
GPU quota.

### The real experiment

```bash
# 1. find R_sat: the rate where p95 TTFT goes vertical
python -m bench.run_matrix --grid bench/grids/e1.yaml --sweep 2 4 8 12 16 24

# 2. put the measured value into e1.yaml as r_sat, then run the matrix
python -m bench.run_matrix --grid bench/grids/e1.yaml --out results/e1

# 3. figures, table, pre-registered comparisons
python -m bench.analyze --results results/e1 --out analysis/e1
```

`run_matrix` skips points whose result file already exists, so a killed Kaggle session
resumes where it stopped. Run order is shuffled within a session so thermal drift spreads
across conditions instead of landing on one policy.

## Benchmark protocol

Latency numbers get dismissed for five reasons. Each is closed here:

1. **Closed-loop load hides queueing.** The client is open-loop: every request goes out at
   its scheduled arrival time whether or not earlier ones have finished. The exception is
   W3, where turns inside a session genuinely wait for the previous answer.
2. **Different token counts across engines.** Every request has a fixed `max_tokens` with
   `ignore_eos`, and any run where the streamed counts disagree is flagged and dropped.
3. **No warmup.** 50 deliberately off-trace requests, then a stats reset, so the server is
   warm but the prefix cache holds nothing the measured run will ask for.
4. **Single runs.** Three repetitions with different trace seeds; the table reports the
   median and the figures show the spread.
5. **A client competing with the server for CPU.** Client CPU is measured and reported, and
   a run where it passes 70% of a core is flagged. Pin it with `taskset -c 3` on Linux.

Every per-request record — sent, first token, every token timestamp, token count — is saved
as raw JSON along with the engine version, the exact command line, the model revision, the
GPU name and the driver. Percentiles are computed afterwards, never live.

### Metrics

| Metric | Definition |
| --- | --- |
| TTFT | first streamed token minus request sent, queueing included, which is the point |
| TPOT | (last token time − first token time) / (output tokens − 1) |
| ITL | every gap between consecutive streamed tokens; p95/p99 of the pooled gaps |
| Queueing delay | server-side `admitted_at − arrival`; Mercury only, the fairness metric |
| Prefix hit rate | cached prompt tokens / prompt tokens, token-level, same as vLLM counts |
| Goodput | requests/s meeting both TTFT <= 2 s and TPOT <= 100 ms |
| Scheduler overhead | `schedule()` time / step time, per step |

## Layout

```
mercury/     config, request, hashing, allocator, scheduler, runner, engine, server
mercury/model/    qwen2, paged attention, rope, safetensors loading
mercury/kernels/  the fused residual-add + RMSNorm + INT8 Triton kernel
bench/       workload generator, open-loop client, run matrix, analysis, kernel bench
tests/       allocator and scheduler on CPU, parity and kernel on GPU
docs/        significance, code walkthrough, defense
```

## Tests

```bash
pytest tests -q                 # laptop: allocator, scheduler, engine, server, harness
pytest tests -q -m gpu          # adds parity against HF transformers and the kernel
```

The GPU-free tests cover the allocator's three-state invariant under randomized churn,
every scheduling policy, preemption, cancellation, the whole HTTP path against a real
uvicorn socket, and the benchmark harness end to end against a real server process.

## Known scope

Mercury runs Qwen2 only, fp16 only, one GPU, no chunked prefill, no CUDA graphs, and no
fused paged-attention kernel. Each of those is a deliberate choice with an alternative that
was considered and rejected; `docs/DEFENSE.md` lists them with the reasoning.

## Sources

- Rathi et al., "Requests of a Feather Must Flock Together", arXiv 2605.06046 (2026) —
  prefix-homogeneity scheduling, LPM overhead, hash-based prefix detection
- SGLang server arguments — `--schedule-policy` options
- SGLang issue #31954 — LPM starvation and the proposed aging term
- SGLang issue #5064 — Triton attention backend as the sm75 default
- vLLM benchmark CLI — `vllm bench serve`, prefix-repetition dataset, latency definitions
- vLLM issue #1157 — the bf16 error on T4
- Triton PR #5066 — FMA fallback for `tl.dot` below compute capability 8.0
