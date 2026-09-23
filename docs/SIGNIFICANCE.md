# Mercury: what it is and why it matters

## What it does, plainly

Mercury is an LLM inference server I wrote from scratch in PyTorch. It takes HTTP requests,
runs a 0.5B-parameter model on one GPU, and streams tokens back. Internally it does the
things a production serving engine does: it stores attention keys and values in fixed-size
pages instead of one contiguous buffer per request, it recognises when two requests start
with the same text and reuses the cached computation instead of redoing it, it keeps a
batch of requests in flight and swaps finished ones out for waiting ones every step, and it
streams tokens over server-sent events with proper cancellation when a client hangs up.

But the server is not the point. The server is the instrument.

The point is an experiment. Mercury implements three request-scheduling policies that are
identical in every way except the order in which they admit waiting requests, and it
measures what that order is worth. The question it answers is:

> When the KV cache cannot hold every shared prefix at once, does ordering requests by
> cache affinity beat first-come-first-served — and what does it cost the requests that get
> pushed back?

Everything else in the repository exists to make that question answerable with numbers
nobody can wave away.

## The real problem

Serving an LLM is a memory problem before it is a compute problem.

Every token a model has already seen leaves behind a key and a value vector in every layer.
For Qwen2.5-0.5B that is about 12 KB per token. A 2,000-token system prompt costs 24 MB of
GPU memory, and it costs that again for every request carrying it, unless you do something
about it. The "something" is prefix caching: notice that a hundred requests share the same
2,000-token preamble, compute it once, and let them all point at the same physical memory.

This matters because of how people actually use models now. A chatbot deployment has one
system prompt shared by every user. An agent framework sends a tool catalogue thousands of
tokens long on every single call. A multi-turn conversation resends its entire history
every turn, so turn seven is turn six's prompt plus a bit. Shared prefixes are not an edge
case; they are most of the tokens.

Prefix caching is well understood and every serious engine has it. Here is the part that
isn't settled. The cache is finite. When the prefixes that are live in your workload add up
to more than the cache can hold, the cache has to evict something, and *what order you
serve requests in decides what gets evicted*.

Serve requests strictly in arrival order and you interleave prefix groups: a request for
group A, then B, then C, then A again. By the time the second A arrives, A's prefix has
been evicted to make room for B and C, so you recompute it from scratch. Then you evict it
again for the next B. The cache is full the whole time and almost never useful. That is
thrashing, and it is not a subtle effect — in my own measurements it takes the hit rate
from 82% down to 45%.

Group the requests by prefix instead and you stop paying for that. That policy exists;
SGLang ships it as `--schedule-policy lpm`. And it has a problem its users have reported:
if you always prefer the request with the longest cached prefix, a request whose prefix
isn't cached can be displaced by every cache-hot arrival, forever. Someone waits
indefinitely so that everyone else can be slightly faster. The fix is obvious in outline —
let a request's priority grow as it waits, and put a hard ceiling on the wait — but the
interesting question is what that fix costs. Nobody publishes that number.

Mercury measures it.

## Why this is harder than it sounds

**The memory system and the scheduler are one system.** You cannot design the scheduler
without the allocator, because "can this request be admitted?" is a question about free
blocks, cached blocks that would be pinned, and blocks the running batch will need on the
next step. And you cannot design the allocator without the scheduler, because the order
blocks are released in determines which ones get evicted. One line in my allocator — the
loop that releases a finished request's blocks in reverse order, so its private tail enters
the eviction queue before its shared head — measurably changes the hit rate. That line only
makes sense if you are holding both halves in your head at once.

**Being wrong is silent.** This is the part that makes an inference engine different from
most software. If the block table points at a page that was evicted and handed to another
request, the model does not crash. It reads whatever is in that memory, attends over it,
and produces fluent, grammatical, completely wrong text. If rotary position embeddings
restart at zero for a suffix resumed on top of a cached prefix, the output looks plausible
and drifts. There is no exception, no stack trace, no failing assertion. The only defence
is a correctness gate you build deliberately: Mercury compares its greedy output against
HuggingFace `transformers` on the same weights across four modes — single request, batched,
with a warm prefix cache, and after a forced preemption — because a prefix-cache bug shows
up in exactly one of those and nowhere else.

**The effect is conditional, so measuring it means controlling a variable most benchmarks
ignore.** Prefix-aware scheduling should do nothing when the cache is big enough to hold
everything, because then FCFS already hits on almost every request. It should matter in a
band where the working set exceeds capacity, and it should stop mattering again when the
working set exceeds capacity so badly that even perfect grouping can't keep prefixes
resident. A benchmark that reports one number from one configuration cannot tell you which
of those three regimes it was in, which means it cannot tell you whether the number
generalises. So Mercury's main figure has cache pressure on the x-axis — the ratio of
shared prefix tokens live in the workload to KV cache capacity — and sweeps it from 0.25 to
4. The shape of that curve is the finding. A flat left half is not a disappointing result;
it is the control that makes the middle believable.

**The cost is invisible in the metrics people report.** Mean latency, p95 latency, and
throughput all improve or stay flat under a policy that starves a minority of requests,
because the starved requests are a minority. You only see it if you go looking: maximum
queueing delay, measured server-side as the gap between a request arriving and being
admitted. That metric is in Mercury because the thing it measures is the reason the policy
is controversial.

## What makes this different from a tutorial build

I want to be exact about this, because "I built an inference server" is a sentence a lot of
people can say, and most of what they built is a wrapper.

**The prior art is stated up front, not discovered by the reader.** Prefix-aware scheduling
is not mine. SGLang shipped it. The starvation problem is a known open issue. A 2026 paper
called Feather already measured that prefix-aware schedulers can burn CPU comparable to
their GPU work. All of that is in the README's second paragraph. What is mine is the
measurement of *when* it helps as a function of pressure, the aging variant with a hard
bound, and the honest accounting of what the bound costs. Claiming to have invented what
SGLang already ships is how a portfolio project gets laughed out of an interview.

**Every number is explainable down to the allocator.** Nothing here is a call into a
library I can't read. The paged attention, the block allocator, the hash chain that
establishes prefix identity, the scheduler, the continuous batching loop, the Qwen2 forward
pass, the safetensors loading, the sampler, the SSE streaming — all of it is in this
repository, and when a number moves I can say which line moved it. That is the difference
between "vLLM got 2,400 tokens per second" and "the hit rate fell because the release order
put the shared prefix into the eviction queue ahead of the private suffix."

**The architecture was chosen to be testable, and it is tested.** The entire contract
between the scheduler and the GPU is one method: `execute(plan) -> sampled token ids`.
Because that seam exists, the allocator, all three policies, preemption, continuous
batching, cancellation and the whole HTTP path can be exercised on a laptop with a
deterministic fake runner — including randomised property tests that assert reference
counts never go negative, that no block is ever in two states at once, and that after
thousands of operations every block comes home. The GPU is needed only for parity and for
performance. In practice about 40% of the code never needs a GPU, and 84 tests run in under
ten seconds on a laptop. I also found a real scheduling insight from those tests rather than
from theory: with only one prefix group, all three policies are identical, because there is
nothing to thrash between. That is now a test.

**The comparisons were pre-registered.** Four specific comparisons are written into the
README before the experiment runs, including the one designed to fail if the headline is a
measurement artifact ("all policies at pressure 0.5 should be within noise; if they aren't,
the headline is suspect"). The analysis script computes exactly those four and prints
PASS/FAIL. This removes the most common way benchmark projects fool themselves, which is
choosing which comparison to report after seeing the data.

**The benchmark protocol closes the holes that get latency numbers dismissed.** Open-loop
load, so queueing actually builds instead of being capped by a fixed number of simulated
users. Fixed output lengths with EOS ignored, so no engine can look faster by stopping
early, and any run where the token counts disagree is flagged and dropped. Warmup requests
that are deliberately *not* from the trace, so the server is warm but the prefix cache
starts empty for the measured run. Three repetitions with different seeds. Run order
shuffled within a session, so thermal drift spreads across conditions instead of landing on
one policy. Client CPU measured and reported, because a client at 80% of a core is part of
the measurement and not an observer of it. And the capacity cap read back out of every
engine's own startup log and asserted, because a flag that silently changed name between
releases would put the whole sweep at the wrong pressure and produce a beautiful,
meaningless figure.

**Scheduler CPU time is reported, not assumed away.** The Feather paper's finding is that
prefix-aware scheduling can cost CPU comparable to GPU execution. Mercury times
`schedule()` on every single step and reports it as a share of step time. If the policy
wins on hit rate and loses it all to scheduling overhead, that is the result, and I would
rather publish it than not know.

## What this project is not

It is not a fast inference server. Mercury's attention gathers KV blocks into a padded
tensor and calls PyTorch's `scaled_dot_product_attention` instead of using a fused
paged-attention kernel. vLLM will beat it on raw throughput and that is expected and fine —
the gather is identical across all three policies, so it cannot bias the comparison that
carries the finding. Using it as an excuse to skip the comparison would be dishonest;
pretending Mercury competes on absolute speed would be worse.

It is not a new scheduling algorithm. It is a careful measurement of a known one, plus a
starvation bound, on hardware anyone can get for free.

It does not run on anything but Qwen2, fp16, one GPU. Those are scope choices, each with a
rejected alternative I can defend.

## The sentence this is all aimed at

The experiment is designed so that the write-up ends with one sentence of this shape:

> On a T4 serving Qwen2.5-0.5B, when shared prefixes exceeded KV capacity by 2x,
> prefix-aware scheduling with a 5-second starvation bound raised prefix-cache hit rate
> from A% to B% and cut p95 time-to-first-token by C%, at D% of FCFS throughput, while
> capping worst-case queueing at E seconds versus F seconds for unbounded LPM; below
> capacity all policies were within noise.

The last clause is the one that makes the rest believable, and it is the reason the
pressure sweep exists at all.
