# Defending Mercury

For demos, presentations, and interviews. Everything here is meant to be said out loud.

---

## The 30-second version

> Mercury is an LLM inference server I wrote from scratch in PyTorch — paged KV cache,
> prefix caching, continuous batching, streaming. But the server is the instrument, not the
> product. The experiment is this: when the KV cache can't hold every shared prefix at once,
> serving requests in arrival order makes the cache thrash — you evict a prefix right before
> the next request that needs it. Grouping requests by prefix fixes that, but it can starve
> requests whose prefix isn't cached. So I built three schedulers that differ only in
> admission order, added a starvation bound to the third, and measured what the bound costs
> as a function of how much cache pressure the system is under. The headline isn't a
> number, it's a curve: nothing happens below capacity, the policies separate in the middle,
> and they converge again when nothing fits.

## The 2-minute version

> Serving a language model is a memory problem before it's a compute problem. Every token
> the model has seen leaves a key and a value in every layer — about 12 KB per token for the
> model I used. So a 2,000-token system prompt costs 24 MB, and it costs that again for
> every request that carries it, unless you cache it. That's prefix caching, and every
> serious engine has it.
>
> Here's what isn't settled. The cache is finite. When the prefixes live in your workload
> add up to more than fits, something gets evicted — and the order you serve requests in
> decides what. Strict arrival order interleaves prefix groups: A, then B, then C, then A
> again. By the time the second A arrives you've evicted A's prefix to make room for B and
> C, so you recompute it, and then evict it again. That's thrashing. In my measurements it
> takes the hit rate from 82% down to 45%, and you can see the mechanism directly in the
> eviction count, which more than doubles.
>
> Grouping requests by prefix fixes it — SGLang ships that policy. But it has a known
> problem: if you always prefer the longest cached prefix, a request whose prefix isn't
> cached gets displaced by every cache-hot arrival, potentially forever. There's an open
> issue about exactly this. So my third policy adds an aging term — your priority grows as
> you wait — plus a hard deadline. And the thing I actually measured is what that fairness
> bound costs in hit rate, because nobody publishes that.
>
> The measurement design is the part I'd defend hardest. The x-axis of the main figure is
> cache pressure: shared prefix tokens divided by cache capacity, swept from 0.25 to 4. I
> pre-registered four comparisons in the README before running anything, including one
> designed to fail if my headline is a measurement artifact — below capacity, all three
> policies must be within noise of each other. The load generator is open-loop, so queueing
> actually builds. Output lengths are fixed with EOS ignored, so no engine can look faster
> by stopping early. And I read the cache-capacity cap back out of each engine's own startup
> log, because a renamed flag would silently put the whole sweep at the wrong pressure and
> produce a beautiful, meaningless figure.
>
> What I'm not claiming: that Mercury is fast. My attention gathers KV pages into a padded
> tensor instead of using a fused paged-attention kernel, so vLLM beats it on throughput.
> That's fine — the gather is identical across all three policies, so it can't bias the
> comparison that carries the finding.

---

## The whiteboard version

Draw two boxes and a queue between them. Talk from left to right.

**"There are two halves and they never touch each other's memory."**

Left box: the HTTP front end, asyncio, FastAPI. It owns sockets, SSE streams, and
cancellation. Right box: the engine, one thread, owns the GPU and everything in it. Between
them a command queue going right, and per-request callbacks coming left.

> All GPU work is on one thread. That kills a whole class of concurrency bug — no two threads
> ever touch the KV cache — and it's how vLLM and SGLang are structured too. The engine never
> touches the asyncio loop directly; it calls a callback, and the callback does
> `loop.call_soon_threadsafe`. That one line is the whole interface back.

**"Now zoom into the engine. It's a loop that does the same four things forever."**

Write: **drain → schedule → execute → route**.

> Drain: apply queued submits and aborts. Aborts land here and nowhere else, so a
> cancellation never arrives in the middle of a forward pass.
>
> Schedule: three steps in a fixed order. First, every running request needs room for the
> token it's about to write — hand out a block, and if the pool is empty, preempt the
> newest-admitted request. Second, score the oldest 128 waiting requests by policy, sort, and
> admit as many as fit. Third, hand back a plan: who to prefill, who to decode.
>
> Execute: build tensors, run the forward, sample.
>
> Route: append tokens, register newly-full blocks in the prefix cache, close finished
> streams.
>
> Continuous batching falls out of doing this every step — a finished request's slot and
> blocks go to a waiting request on the very next step. It isn't a feature, it's what you get
> for free by scheduling every step instead of per-batch.

**"Now the memory, because that's where the interesting part lives."**

Draw a row of numbered boxes.

> The KV cache is one big preallocated tensor per layer, chopped into fixed 16-token blocks.
> A request doesn't get contiguous memory, it gets a list of block IDs — a block table. So
> memory can never fragment: any free block serves any sequence. The only waste is internal,
> the unfilled tail of each sequence's last block, which averages about eight tokens per
> sequence. That's the answer if someone asks about fragmentation.
>
> Every block is in exactly one of three states: free, in use, or cached-but-unreferenced.
> That third state is the whole trick. When a request finishes, its blocks don't go back to
> free — if they're full and hashed, they go into an LRU where a future request with the same
> prefix can claim them.

**"How do we know two requests share a prefix? Chain hashes."**

> Each block's hash is computed from its own 16 tokens *and* the previous block's hash. So
> matching on block five is a promise that blocks one through five all match. That means the
> matched prefix is always contiguous from the start, which is why lookup can stop at the
> first miss. And it means scheduling never looks at token IDs — hashes are computed once on
> arrival, and scoring a request is a handful of dictionary lookups on integers. That's the
> design the Feather paper argues for over radix trees, on the grounds that tree traversal
> can cost CPU comparable to the GPU work you're scheduling.

**"And here's the one line I'd point at if you asked what I learned."**

> When a finished request releases its blocks, it releases them in *reverse* order. Its last
> blocks are its private suffix — nothing else will ever want those. Its first blocks are the
> shared prefix the next request in the group needs. The eviction queue is oldest-first, so
> releasing backwards means the private tail is thrown away before the shared head. Release
> forwards and you evict exactly the thing you were trying to cache. It's one keyword and it
> moves the hit rate.

**"The three policies differ in one function."**

> `_score`. FCFS returns negative arrival time. LPM returns cached prefix tokens. LPM+aging
> returns cached tokens plus alpha times seconds waited, and infinity past a hard deadline.
> That's it. Everything else — the allocator, the batching, the attention — is byte-identical
> across the three. That's deliberate: if the only thing that differs is the sort key, then
> any difference in the results is attributable to the sort key.

---

## Design decisions, and what I rejected

### Attention over pages: gather into a padded tensor + SDPA

**Rejected:** writing a Triton paged-decode kernel; using FlashInfer's paged kernels.

A second Triton kernel is scope creep, and FlashInfer on a T4 via JIT adds install risk on a
machine where I can't debug the toolchain. The gather costs real bandwidth — it copies each
sequence's KV every layer, every step — but it's identical across all three policies, so it
can't bias the finding. The attention lives behind a small context interface precisely so a
faster backend can drop in later without touching the model. If someone says "your server is
slow", the answer is: yes, measurably, and here's the line that makes it slow, and here's why
it doesn't threaten the result.

### Prefix detection: hash chain over 16-token blocks

**Rejected:** a radix tree, SGLang-style.

Hash chains give O(1) lookups per block and match vLLM's design. Radix trees give token-level
matching — they'd catch a prefix that diverges mid-block, where I lose up to 15 tokens — but
they cost more CPU per step, and the Feather paper's finding is specifically that this CPU
can rival GPU time. At 2,000-token prefixes, losing up to 15 tokens of reuse is under 1%.
I took the cheap lookup.

### Block size: 16

**Rejected:** 32 or 64.

Bigger blocks mean fewer hash lookups and less gather overhead, but coarser sharing and more
internal fragmentation. 16 is vLLM's default, which keeps the cross-engine comparison clean —
if I used 64 and vLLM used 16, a reviewer would reasonably ask whether the difference was the
policy or the granularity.

### Prefill and decode in separate forward passes

**Rejected:** chunked prefill mixed with decode, Sarathi-style.

Separate phases are far simpler. The cost is real and I report it rather than hide it: a long
prefill stalls every decode in flight, which shows up as inter-token-latency spikes. That's
why ITL p99 is a reported metric and not just TPOT — TPOT averages the stall away. Chunked
prefill is the obvious next version.

### Preemption by recompute

**Rejected:** swapping KV to pinned host memory.

Recompute is simple, and with prefix caching it's often cheap: the victim's committed blocks
may still be in the cache when it resumes, so the "recompute" is largely a cache hit. Swap
needs pinned buffers, async copy streams, and a second memory pool, for a benefit that prefix
caching already partly provides.

### Starvation bound: aging term *plus* a hard deadline

**Rejected:** aging only; deadline only.

This one I can defend with a proof rather than a preference. Two requests both age at the same
rate, so a fixed gap in arrival times gives a *constant* advantage that aging never closes.
Aging only beats arrivals that keep being new — which is the actual starvation scenario, a
continuous hot stream — but it gives no guarantee against any particular competitor. The
deadline converts "usually fair" into a bound. Conversely a deadline alone is a cliff: nothing
happens, nothing happens, then a hard preemption. Together you get a smooth trade with a hard
guarantee. Both properties are tests.

### Scoring only the oldest 128 waiting requests

**Rejected:** scoring everything.

Scoring is O(waiting × prefix blocks) per step. SGLang caps it the same way and falls back to
FCFS past 128. I also count how often the queue was deeper than the window, so I can report
whether the cap actually bound during the experiment rather than assuming it didn't matter.

### The Triton kernel: fused residual-add + RMSNorm + INT8 quant

**Rejected:** an INT4 dequant-matmul kernel.

This is a hardware argument, not a taste argument. Below compute capability 8.0, Triton's
`tl.dot` falls back to plain FMA on CUDA cores instead of tensor cores. A T4 is 7.5. So an
INT4 matmul in Triton on a T4 would race cuBLAS's tensor-core fp16 GEMM *without tensor
cores* and lose, for reasons that have nothing to do with my code. RMSNorm is a memory-bound
reduction with no `dot`, so it runs on the well-supported part of Triton and the benefit of
fusion shows up cleanly. Picking the kernel the hardware can actually show you something
about is the decision.

### Synthetic workloads

**Rejected:** real conversation datasets.

Prefix-caching behaviour depends on the *shape* of sharing — how many prefix groups, how long,
how arrivals interleave — not on what the tokens say. Synthetic traces let me control every
parameter the finding depends on, with no licensing questions, and every engine replays the
identical file. The honest cost, which is stated in the write-up: in the multi-turn workload
each turn's history uses scripted assistant tokens rather than what the engine actually
generated, so every engine sees byte-identical prompts. That understates caching for agents,
but it understates it equally for everyone.

### Capacity set by hand

**Rejected:** using whatever the GPU has free.

A T4 with 10 GB free holds about 850,000 tokens of KV for this model, which no realistic
prefix working set fills. Left alone, every point in the sweep would sit at pressure far below
1 and I'd measure nothing. Capping the pool emulates a bigger model or a busier GPU where KV
memory really is scarce. It's standard — vLLM and SGLang both expose the flag — and the only
requirement is saying so plainly, which the README does.

### One engine thread

**Rejected:** a thread pool, or async GPU work.

Python's GIL makes multi-threaded GPU submission mostly pointless, and the correctness benefit
of a single owner for the KV cache is large. The cost is that per-step tensor building is on
the critical path, which is why `prepare_seconds` is measured separately instead of being
buried in step time.

---

## The hard questions

### On the result itself

**1. Isn't this just SGLang's LPM? What did you actually add?**

SGLang's LPM is the second of my three policies and I say so in the README's second
paragraph. What I added is three things. First, the measurement of *when* it helps, as a
function of cache pressure — SGLang ships the policy but doesn't publish the curve. Second,
the aging variant with a hard bound, which is a proposal on an open SGLang issue that hasn't
been measured. Third, the fairness cost stated as a number: how much of LPM's hit-rate gain
survives the starvation bound. If you take LPM's number and my aged number, the difference is
the price of the guarantee, and that's the contribution.

**2. Your engine is slower than vLLM. Why should I care about your numbers?**

Because the comparison that carries the finding is Mercury against Mercury. The three policies
share the same allocator, the same attention, the same batching, and differ only in a sort
key. Absolute speed cancels out of a within-engine comparison. vLLM and SGLang are in the
study as reference points, and specifically SGLang's own FCFS-versus-LPM pair is there as an
external check: if my direction of effect disagrees with a production engine's on the same
traces, that's a signal I have a bug, and I'd want to know.

**3. How do I know your prefix cache isn't just returning wrong answers quickly?**

Parity testing, and it's the exit gate on every milestone. I compare greedy output against
HuggingFace `transformers` on the same fp16 weights across four modes: single request,
batched with others, with a warm prefix cache, and after a forced preemption. The requirement
is identical first 16 tokens on at least 48 of 50 prompts, plus first-step logits within 1e-2
max absolute difference. Those four modes exist because a prefix-cache bug shows up in exactly
one of them — the warm-cache mode — and would pass the other three. In fp16 I don't require
exact agreement over long outputs because tiny numeric differences can flip a near-tie, and
claiming bit-exactness would be a claim I can't support.

**4. You showed me a fake-runner result. Isn't that made up?**

It's a scheduler measurement, not a performance measurement, and I label it that way
everywhere. The deterministic runner produces tokens with no model, so nothing about it tells
you anything about GPU speed. What it does tell you is what the *scheduler* does, because the
scheduler's inputs are arrival times, prompt hashes and block availability — none of which
involve the model. It reproduces the mechanism end to end: six prefix groups against an
undersized pool gives FCFS 0.455 hit rate with 1,287 evictions and LPM 0.817 with 576. The
eviction count more than doubling is the thrashing hypothesis made visible.

**5. What if you run the real experiment and there's no effect?**

Then that's the result, and I have three specific things to check before believing it, in
order. One: pressure never actually exceeded 1 — check the capacity cap took effect, which is
why the matrix runner reads it back out of the startup log. Two: arrivals were too sparse for
any reordering to be possible, because the waiting queue was usually empty and there was
nothing to reorder — that's why the rate sweep finds R_sat first and the matrix runs at 0.5
and 0.8 of it. Three: scheduler CPU ate the gain, which is a small-scale replication of the
Feather finding and is a publishable result rather than a failure. Each of those is measured,
not guessed at.

**6. Your sanity check says all policies should be equal below capacity. What if it fails?**

Then my headline is not trustworthy and I say so. That's the entire point of pre-registering
it. A failure there means the runs weren't interleaved, or a cache stayed warm from the
previous run, or token counts differed between conditions — all measurement bugs, not policy
effects. The analysis script prints PASS or FAIL on that comparison specifically so I can't
quietly not look.

**7. In your own smoke run, LPM+aging had a *worse* hit rate than FCFS. Doesn't that sink the
whole thing?**

It's the most useful number I got, and it's an alpha calibration result rather than a policy
result. Here is the run, on the deterministic runner at pressure 2, median of two seeds:

| Policy | Hit rate | p95 TTFT | p99 queue | Max queue | Output tok/s |
| --- | --- | --- | --- | --- | --- |
| FCFS | 0.351 | 0.904 s | 1.054 s | 1.231 s | 628 |
| LPM | 0.376 | 0.590 s | 1.227 s | 1.781 s | 652 |
| LPM + aging (alpha 2000) | 0.265 | 0.666 s | 0.827 s | 0.963 s | 642 |

The fairness half worked exactly as designed — maximum queueing fell from 1.78 s to 0.96 s.
The locality half went backwards, and the reason is that alpha is an exchange rate in tokens
per second and I had it miscalibrated for that workload. Those shared prefixes were 512
tokens; at alpha = 2,000, one quarter of a second of waiting outweighs a *complete* cache
hit. Cache affinity was swamped almost immediately and the policy degenerated into FCFS plus
enough reordering to break the groups up — the worst of both.

So I swept it, at the same pressure, same seeds:

| alpha | Hit rate | p95 TTFT | Max queue | Output tok/s |
| --- | --- | --- | --- | --- |
| 125 | 0.344 | 0.815 s | 1.162 s | 649 |
| 500 | 0.307 | 0.872 s | 1.341 s | 645 |
| 2,000 | 0.281 | 0.734 s | 0.993 s | 635 |
| 8,000 | 0.269 | 0.775 s | 1.036 s | 626 |

Monotone in hit rate and monotone in throughput, which is what a locality-versus-fairness
exchange rate should look like. The calibration rule that falls out is
`alpha ~ shared prefix tokens / acceptable wait in seconds`. For 512-token prefixes and a
one-second tolerance that's about 500; for the 2,000-token prefixes in the real experiment
it's about 2,000, which is the default the design started with. So the constant was right
for the workload it was chosen for and wrong for the one I smoke-tested it on, and the fix
is to derive it rather than hard-code it. That's item 4 on the next-version list.

### On the systems engineering

**8. Where's your bottleneck?**

Two, and I measure both separately. First, the gather in paged attention: `k_cache[block_tables]`
is advanced indexing, so it copies each sequence's entire KV into a padded tensor, every layer,
every step. At batch 64 with 2,000-token sequences that's about 32 MB per tensor per layer.
That's the dominant cost and it's the thing a fused kernel would remove. Second, per-step
tensor building in Python — block tables, slot mappings, sequence lengths, rebuilt every step.
At 0.5B a decode step is short enough that this is visible, which is why `prepare_seconds` is
its own metric. Production engines hide it with persistent buffers and CUDA graphs; I don't
capture CUDA graphs at all.

**9. What happens under memory pressure — does it deadlock?**

No, and there are two guards. At the request level, a prompt plus max_tokens that can't fit in
an empty pool is rejected at submit with a 400 rather than queued, because such a request
would be admitted, need a block, find none, preempt the newest-admitted request — itself — and
loop forever. At startup, the config refuses to start if the whole pool is smaller than
max_model_len. Inside the preemption loop, termination is guaranteed because preempting
`running[-1]` either frees blocks from another request, which always holds at least one, or
preempts the request being served, which flips its state and ends the loop.

**10. What's your worst failure mode?**

A stale block table producing fluent, wrong text. If a block gets evicted and reused while a
running request still points at it, the model attends over another request's keys and emits
grammatical nonsense with no error anywhere. The structural defence is the reference counting —
an in-use block has ref ≥ 1 and is therefore never in the evictable pool — and there's a debug
flag that asserts, every step, that every block in every running request's table has ref ≥ 1.
The behavioural defence is the warm-cache parity test.

**11. How do you know you don't leak memory?**

An end-of-run assertion that available blocks equals total blocks, on every offline test, plus
a benchmark mode that cancels 10% of requests at a random point and asserts the pool comes
back whole. Leaked blocks from cancelled streams are the classic slow leak in a serving
system: nothing looks wrong, and three hours later the pool is empty. I also found that the
obvious way to test this doesn't work — an in-process ASGI transport never delivers
`http.disconnect`, so the cancellation path never runs and the test passes for the wrong
reason. The cancellation tests run against a real uvicorn socket.

**12. What breaks under bad data?**

Adversarial input is the interesting case. I use Python's 64-bit `hash` over token tuples. A
user who could construct a colliding prompt would be served another user's cached KV. At this
scale collisions are negligible and I say so, but there's a switch for SHA-256 and the reason
it exists is that in a real multi-tenant deployment I'd turn it on. Malformed requests get
explicit 400s: empty prompt, batched prompts, a text prompt with no tokenizer loaded, a
request past the token ceiling. A request that asks for more tokens than the pool could ever
hold is refused rather than queued.

**13. Where does this stop scaling?**

Several places, in the order you'd hit them. One GPU, no tensor parallelism, so the model has
to fit on one card. The scheduler's scan is capped at 128 waiting requests, so past that it
degrades toward FCFS by design, and I count how often that binds. The gather in decode
attention grows with batch times longest sequence, so long-context plus high concurrency runs
out of memory in the gather before it runs out of KV cache. And the per-step Python work is
serial with the GPU, so as the model gets smaller and steps get shorter, scheduling and
tensor-building become a larger share — which is exactly the Feather finding, and why I report
scheduler time as a fraction of step time.

**14. Why one engine thread? Isn't that leaving performance on the table?**

Under the GIL, multi-threaded GPU submission from Python buys very little, and the correctness
benefit of a single owner for the KV cache is large — no lock discipline to get wrong, no
possibility of two threads writing the same block. The real cost is that per-step CPU work
sits on the critical path, and I measure that separately rather than hiding it. If I wanted it
back, the answer is CUDA graphs and persistent buffers, not more threads.

**15. Your block size is 16. Did you try anything else?**

Not as an experiment, and I'd call that a gap rather than defend it. I picked 16 because it's
vLLM's default and keeping it identical across engines removes a confounder — if I ran 64 and
vLLM ran 16, a reviewer would reasonably ask whether a difference was the policy or the
granularity. The tradeoff is understood: bigger blocks mean fewer hash lookups and less gather
overhead but coarser sharing and more internal waste. Sweeping it is a next-version item.

**16. How much of vLLM's lead is kernels and how much is CUDA graphs?**

That's exactly why the experiment runs vLLM both with and without `--enforce-eager`. I don't
want to attribute a gap to kernel quality when part of it is graph capture that I simply
didn't implement. If I'm going to lose to vLLM — and I will — I'd rather be able to say which
part of the loss is which.

**17. Isn't fragmentation a problem with a block allocator?**

Not external fragmentation — that's the whole point of fixed-size blocks. Any free block
serves any sequence, so there's no such thing as a free region too small to use. What you pay
instead is internal fragmentation: the unfilled tail of each sequence's last block, which
averages about eight tokens per sequence at block size 16. That's the honest answer, and
because it's bounded and predictable it's a much easier thing to live with than the
contiguous-allocation alternative.

### On the measurement

**18. Why open-loop? Everyone uses closed-loop clients.**

Because a closed-loop client with N simulated users caps offered load at whatever the server
can serve. The queue never grows past N. And this entire project is about the *order* of the
waiting queue — if there's no queue, there's nothing to measure, and all three policies would
correctly come out identical. I verified that experimentally, in fact: with a zero-cost fake
runner, every request is served instantly, the queue is always empty, and the harness reports
no difference between policies. That's the harness being right, and it's the second thing I'd
check if the real experiment showed nothing.

**19. Why fixed output lengths? That's not realistic.**

It's not realistic and it is necessary. If engines stop at different points, an engine that
stops early generates fewer tokens and looks faster, and you're no longer comparing like with
like. So every request has a fixed max_tokens with EOS ignored, and the client verifies the
streamed token counts and flags any run where they disagree. Realism in output length would
cost me the ability to compare at all, and the thing I'm measuring — admission order — doesn't
depend on it.

**20. Your warmup sends requests that aren't in the trace. Why not just warm it up with the
trace?**

Because then the prefix cache would start the measured run already holding the exact prefixes
the run is about to ask for, and the first N requests would get free hits they wouldn't get in
production. The server does need warming — CUDA context, allocator, Python caches — so the
warmup is 50 deliberately off-trace requests, followed by a stats reset. Warm server, cold
cache, zeroed counters.

**21. Three repetitions is not a lot. How do you justify any confidence interval?**

The three repetitions are different trace seeds, and I report the median across them plus the
spread. The confidence intervals come from bootstrapping the pooled per-request values — 500
requests per point, resampled 2,000 times — not from three numbers. And below eight samples
the function returns a point estimate with no interval rather than drawing an error bar from
nothing.

**22. How do you know thermal throttling isn't your result?**

Three things. Run order is shuffled within each session, so if the machine drifts, the drift
spreads across conditions rather than landing on whichever policy ran last. GPU clocks,
temperature and memory are sampled at 10 Hz for the whole of every run, so I can show they
didn't drift rather than assert it. And I can't lock clocks on Kaggle without root, which I
state rather than pretend otherwise.

**23. How do you know your client isn't the bottleneck?**

I measure it. Client CPU time is recorded per run and reported, and any run where the client
exceeded 70% of a core is flagged in the verification output. Kaggle has few cores and the
client shares them with the server, so this is a real risk and not a theoretical one. The
client is deliberately light — asyncio plus aiohttp, no per-token JSON pretty-printing — and
the benchmark sends and receives raw token IDs so there's no tokenizer in the measurement
loop.

**24. What stops you from picking the comparison that looks best after seeing the data?**

Four comparisons written into the README before the experiment ran, computed verbatim by the
analysis script, printed with PASS/FAIL. One of them is designed to fail if the headline is an
artifact. That's not a complete defence against motivated reasoning, but it's the standard
defence, and the alternative — deciding what to report after looking — is how most benchmark
projects fool themselves.

**25. Your `max_wait_s` is 5 seconds. Where did that come from?**

It's a choice, not a derivation, and I'd say so. At 0.5B with these output lengths, a
five-second wait is already a bad user experience, so bounding there rather than at one second
keeps the bound from dominating the policy entirely. The real answer is that it's a tunable
fairness knob and the interesting output is the tradeoff curve, not the point. I have
measured that the bound behaves correctly — the starvation test asserts it — and I've measured
that tightening it costs hit rate.

**26. What's the actual guarantee your starvation bound gives?**

I want to be precise here because the obvious claim is slightly too strong. The deadline
bounds *scheduling* delay: past `max_wait_s` a request outranks everything and no other
request will be admitted ahead of it. It does not bound *memory* delay — the request still
can't start until enough blocks free up. So the honest guarantee is "max_wait_s plus the time
for the running batch to release enough memory", and the test asserts that bound rather than
pretending it's one step. If I wanted a hard wall-clock bound I'd have to preempt running
requests on behalf of a starving one, which is a real design option I didn't take because it
trades one unfairness for another.

---

## Known limitations

Stated as facts, because each one was a choice.

- **Qwen2 only, fp16 only, one GPU.** No tensor parallelism, no other architectures, no
  quantised weights. fp16 specifically because a T4 is compute capability 7.5 and has no bf16.
- **No fused paged-attention kernel.** The gather is the throughput ceiling. vLLM beats
  Mercury on raw speed and that gap is mostly this.
- **No CUDA graphs.** Per-step launch overhead is paid in full, every step, and measured.
- **No chunked prefill.** A long prefill stalls every decode in flight. I report ITL p99 so
  the cost is visible rather than averaged away by TPOT.
- **Preemption is recompute-only.** No swap to host memory.
- **Block size is fixed at 16 and was not swept.**
- **64-bit non-cryptographic hashing by default.** Fine at this scale, not fine against an
  adversary. The switch exists; the default doesn't use it.
- **Scheduler scan is capped at 128** and degrades toward FCFS beyond that. Counted, not
  hidden.
- **Synthetic workloads only**, and in the multi-turn case the history uses scripted
  assistant tokens rather than generated ones, which understates caching for agents equally
  across engines.
- **Single-node measurement on a shared free-tier GPU.** Clocks can't be locked. Logged
  instead.

## What the next version adds

In the order I'd actually do them:

1. **A fused paged-attention kernel**, or FlashInfer behind the existing context interface.
   This is the single biggest absolute-performance item and the interface already exists to
   accept it.
2. **Chunked prefill.** Mixing a slice of prefill into each decode step removes the ITL spikes
   that separate prefill phases cause. This is the fix for the metric I currently just report.
3. **CUDA graphs for the decode path,** plus persistent buffers so per-step tensor building
   stops being serial with the GPU. Together with (1) this is most of the vLLM gap.
4. **Alpha as a function of workload rather than a constant.** The smoke run showed that alpha
   has to scale with prefix length; deriving it from the observed mean cached-prefix length
   instead of setting it by hand would make the policy self-tuning and would remove the one
   calibration failure I found.
5. **Preemption on behalf of a starving request,** so the starvation bound becomes a true
   wall-clock guarantee rather than a scheduling-order guarantee. This needs a fairness
   argument about what you're allowed to preempt, which is why it isn't in version one.
6. **A block-size sweep,** to replace "I used vLLM's default" with a measurement.
7. **Cryptographic hashing on by default** with the performance cost measured, so multi-tenant
   use is safe out of the box.
8. **Tensor parallelism**, which is the point where the single-engine-thread design has to be
   revisited seriously rather than defended.
