# Mercury: code walkthrough

Every file, every class, every function that isn't obvious. For each block there are two
explanations:

- **Plain** — the idea, with a concrete analogy, no jargon.
- **Technical** — the mechanics, the data shapes, and what breaks if it were written
  differently.

If you read only one section, read [the execution trace](#the-execution-trace) and
[the load-bearing blocks](#the-load-bearing-blocks).

---

## The shape of the thing

Two halves joined by a queue.

```
HTTP side (asyncio)                    Engine side (one thread)
-------------------                    ------------------------
FastAPI /v1/completions                schedule -> execute -> route output
  owns sockets, SSE streams,             owns the GPU, the KV cache,
  cancellation                           the block allocator, the scheduler
        |                                        ^
        +------- command queue ------------------+
        <------- per-request sink callbacks -----+
```

All GPU work lives on one thread. That removes an entire class of concurrency bug — no two
threads ever touch the KV cache — and it matches how vLLM and SGLang separate their API
server from their engine core. The HTTP side never blocks on the GPU, and the engine side
never touches the asyncio loop directly; it calls a per-request *sink* callback, and the
sink is the thing that knows how to hop back onto the right event loop.

```
mercury/
  config.py        shapes and knobs, and the checks that stop a nonsense configuration
  request.py       one request's data and its state machine
  hashing.py       prefix identity: a chain hash over full blocks
  allocator.py     the paged KV pool, reference counts, prefix cache, LRU eviction
  scheduler.py     continuous batching, the three policies, preemption
  runner.py        plan -> tensors -> forward -> sampled tokens
  engine.py        the loop, the command queue, output routing, metrics
  metrics.py       per-step timings and finished-request records
  sampling.py      greedy / temperature / top-p
  server.py        FastAPI, SSE, cancellation, /stats
  testing.py       the deterministic runner that makes all of the above testable
  model/
    qwen2.py       RMSNorm, SwiGLU MLP, GQA attention, decoder stack
    attention.py   writing and reading KV pages, the two attention contexts
    rope.py        rotary position embeddings
    loader.py      resolve the model, build on meta device, load safetensors
  kernels/
    rmsnorm_quant.py   the fused residual-add + RMSNorm + INT8 Triton kernel
bench/
  workloads.py     W1-W6 trace generation and the pressure calibration
  client.py        open-loop trace replayer
  run_matrix.py    the experiment grid and every guard around it
  analyze.py       percentiles, CIs, figures, tables, pre-registered comparisons
  kernel_bench.py  eager vs torch.compile vs Triton
```

---

## The execution trace

### Startup

1. `server.main()` parses flags into an `EngineConfig`.
2. `create_app(cfg)` constructs an `Engine`.
3. `Engine.__init__` builds a `BlockAllocator`, a `Scheduler` wrapping it, an
   `EngineMetrics`, and — unless one is injected — a `ModelRunner`.
4. `ModelRunner.__init__` calls `load_qwen2`, which resolves the model path (local
   directory or a pinned Hub snapshot), reads `config.json` into a `ModelConfig`, validates
   the engine config against it, builds the module tree on the `meta` device, calls
   `to_empty` to allocate real storage, copies every tensor in from safetensors, and builds
   the rotary cos/sin table.
5. Still in `ModelRunner.__init__`: allocate the KV cache — for each of the 24 layers, a K
   tensor and a V tensor of shape `[num_blocks, 16, 2, 64]` in fp16.
6. The server prints one line containing `kv_blocks=<N>`. `bench/run_matrix.py` greps for
   exactly that to prove the capacity cap took effect.
7. FastAPI's lifespan starts the engine thread and calls `gc.freeze()`.

### One request

1. `POST /v1/completions` arrives. The handler parses the body, resolves the prompt to
   token ids, and builds a `SamplingParams`.
2. It creates an `asyncio.Queue` and a closure `sink(event)` that does
   `loop.call_soon_threadsafe(queue.put_nowait, event)`. That closure is the only bridge
   from the engine thread back to this request's stream.
3. `engine.submit(...)` checks the request against the server's token ceiling, builds a
   `Request` — computing its block hash chain right here, once, so scheduling never has to
   look at token ids again — and pushes `("add", request)` onto the command queue.
4. The handler returns a `StreamingResponse` wrapping an async generator that awaits the
   queue.

### One engine step

`Engine.step()` runs this, over and over:

1. **Drain commands.** Every `add` goes into the scheduler's waiting list, sorted by
   arrival. Every `abort` releases that request's blocks immediately. Aborts are applied
   here and nowhere else, so a cancellation never lands in the middle of a forward pass.
2. **Schedule.** `Scheduler.schedule(now)` does three things in a fixed order:
   a. Every running request needs KV room for the token it is about to process. Hand out a
      block; if the pool is empty, preempt the newest-admitted request to free some.
   b. Take the oldest `scan_limit` waiting requests, score them by the active policy, sort,
      and admit as many as fit under the running-count budget, the prefill token budget,
      and available memory.
   c. Return a `Plan`: the newly admitted requests to prefill, and the already-running
      requests to decode.
3. **Execute.** `ModelRunner.execute(plan)`:
   - For each prefill request, build its slot mapping and block table, run the model over
     `tokens[num_cached:]` with positions starting at `num_cached`, and keep the last
     hidden state.
   - For the decode batch, build one row per running request — last token id, its position,
     its slot, its padded block table, its sequence length — and run one forward.
   - Run `lm_head` once over the stacked prefill hidden states, once over the decode hidden
     states, and sample.
4. **Route output.** For each request that produced a token:
   - `_commit_full_blocks` extends the hash chain over any block that just became full and
     registers it in the allocator's cache, making it shareable.
   - Append the token, record the first-token timestamp, call the sink.
   - If the request hit `max_tokens` (or EOS, when not ignored), release its blocks, drop it
     from the running list, record it, and send a final `done` event that closes the stream.
5. **Record.** One `StepRecord` with scheduling time, tensor-preparation time, forward time,
   total step time, queue depths and free blocks.

### The stream ending

The generator receives `done`, emits a final chunk carrying `finish_reason`, emits
`data: [DONE]`, and returns. If instead the client hangs up, the generator is cancelled or
`request.is_disconnected()` returns true; either way its `finally` fires `engine.abort(rid)`
and the blocks come back on the next step.

---

## The load-bearing blocks

The eight worth knowing cold. Everything else is plumbing around these.

| # | Block | Why it is load-bearing |
| --- | --- | --- |
| 1 | `hashing.block_hashes` | Defines what "the same prefix" means. Every cache hit and every correctness guarantee rests on this chain. |
| 2 | `allocator.allocate_prompt` | The only place blocks are handed out. Gets the pinning arithmetic right or the pool silently oversubscribes. |
| 3 | `allocator.release` | Reverse order. One line, and it measurably changes the hit rate. |
| 4 | `scheduler.schedule` step 1 | Decode reservation and preemption. The only loop that can deadlock the engine. |
| 5 | `scheduler._score` | The three policies live here and nowhere else. |
| 6 | `engine._commit_full_blocks` | The moment a block becomes shareable. Too early and you corrupt other requests; too late and you never hit. |
| 7 | `attention.paged_prefill_attention` | The explicit mask. `is_causal=True` is wrong here and wrong silently. |
| 8 | `runner.slot_mapping` | The off-by-one that gives you exactly one correct block and then garbage. |

Two more that are load-bearing for the *experiment* rather than the engine:

| # | Block | Why |
| --- | --- | --- |
| 9 | `workloads.capacity_for_pressure` | Defines the x-axis of the main figure. Wrong here and every point is at the wrong pressure. |
| 10 | `run_matrix.check_capacity` | Reads the cap back out of the engine's own log. Without it a renamed flag produces a beautiful, meaningless sweep. |

---

# File by file

## `mercury/config.py`

### `ModelConfig`

A frozen dataclass holding the architecture numbers read out of HuggingFace's
`config.json`: hidden size, layer count, attention heads, KV heads, head dimension,
vocabulary size, RMSNorm epsilon, the rotary base, and whether the output embedding is tied
to the input embedding.

**Plain.** It's the spec sheet for the model, copied off the box. Before you can build the
engine you need to know how many cylinders it has.

**Technical.** `from_dict` refuses anything whose `architectures[0]` doesn't contain
`Qwen2`, because loading a Llama checkpoint into this module tree would succeed at the
tensor-shape level for some layers and then produce nonsense. `head_dim` falls back to
`hidden_size // num_attention_heads` because older configs don't carry it. `eos_token_id`
is normalised to a tuple because some configs give an int and some give a list.

`kv_bytes_per_token` is the number the whole experiment is calibrated against:

```
2 (K and V) x num_hidden_layers x num_key_value_heads x head_dim x bytes_per_element
= 2 x 24 x 2 x 64 x 2 = 12,288 bytes
```

Twelve kilobytes per token. That is why a 2,000-token system prompt costs 24 MB and why
prefix sharing is worth building.

### `EngineConfig`

Every knob: block size, block count, batch and prefill budgets, the policy and its two
parameters, dtype, device, seed, and the debug switches.

**Plain.** The dashboard. Also the bouncer — it refuses settings that would break things in
confusing ways instead of letting you find out an hour into a benchmark.

**Technical.** `__post_init__` rejects an unknown policy, an unsupported dtype, and a block
size that isn't a power of two. `validate_against(model_cfg)` is the interesting one and it
encodes three real failure modes:

- `max_model_len` above the model's `max_position_embeddings` would index past the end of
  the rotary table.
- **The livelock guard.** If the whole KV pool is smaller than one maximum-length request,
  that request preempts itself forever: it is admitted, it needs another block, nothing is
  available, the scheduler preempts the newest-admitted request, which is itself, and round
  it goes. Nothing runs and nothing errors. The check refuses to start.
- bf16 on compute capability below 8.0. A T4 is 7.5. This is a real vLLM bug report and it
  is better as a startup error than a kernel failure.

`autosize_blocks` exists for interactive use: measure free memory, take a fraction, divide
by bytes per block. The experiment never uses it, because the whole design of the study is
that capacity is set by hand to put the system at a chosen pressure.

---

## `mercury/hashing.py`

### `block_hashes(token_ids, block_size, parent=0)`

```python
out, parent = [], 0
for start in range(0, len(token_ids) - block_size + 1, block_size):
    parent = block_hash(parent, token_ids[start : start + block_size])
    out.append(parent)
```

**Plain.** Imagine numbering the pages of a book, except each page's number is computed
from its own words *and* the previous page's number. Two books get the same number for page
five only if they are word-for-word identical through the end of page five. That lets you
compare two 2,000-word documents by comparing a handful of numbers, and it means a match on
page five is a *promise* that pages one through five all match.

**Technical.** A chain hash over full blocks. `h_i = H(h_{i-1}, tokens[i*B : (i+1)*B])`, so
`h_i` transitively depends on every token before position `(i+1)*B`. Two consequences the
rest of the system leans on:

1. A hit on block *i* guarantees blocks *0..i* are shared. That is why
   `cached_prefix_blocks` can stop at the first miss instead of scanning — the matched
   prefix is necessarily contiguous from the start.
2. Scheduling never touches token ids. Hashes are computed once when the request arrives;
   scoring a request for LPM is then a handful of dictionary lookups on integers. This is
   the design the Feather paper argues for over radix-tree traversal, on the grounds that
   tree walks cost CPU comparable to the GPU work they are scheduling.

The loop runs to `len - block_size + 1`, so a partial trailing block gets no hash. That is
deliberate and is enforced downstream: only a full block is shareable, because writing the
next token into a partial block would corrupt anyone sharing it.

If it were written differently: hashing each block independently rather than chaining would
let two requests that share block three but differ in block one collide on block three, and
the model would attend over another request's keys. That produces fluent, wrong output with
no error anywhere.

### `block_hash` and the cryptographic switch

Default is Python's built-in `hash` over a tuple of ints, which is deterministic across
processes (`PYTHONHASHSEED` randomises strings and bytes, not integers) and fast. The
`sha256` mode exists because a 64-bit hash over adversarial input is forgeable: a user who
could construct a colliding prompt would be served another user's KV. At this scale
collisions are negligible; the switch documents that I know why it would matter and what
the fix is. It must be set before any request is hashed, since flipping it mid-run would
split the cache into two incompatible namespaces.

### `extend_hashes(hashes, token_ids, block_size)`

**Plain.** The book is still being written. Every time another full page is finished, give
it a number, chaining from the last one. Don't renumber the pages you already did.

**Technical.** Grows an existing chain in place to cover `len(token_ids) // block_size`
blocks, continuing from `hashes[-1]`. Called once per step per request from
`_commit_full_blocks`. Without it, extending the chain over generated tokens would mean
rehashing the whole prompt every step — O(prompt) work per token, which at 2,000-token
prompts would dominate the step.

---

## `mercury/allocator.py` — `BlockAllocator`

The single most important invariant in the codebase: **every block is in exactly one of
three states.**

| State | ref count | Has a hash? | Lives in |
| --- | --- | --- | --- |
| Free | 0 | No | the `free` deque |
| In use | >= 1 | Once full, yes | some request's block table |
| Cached, evictable | 0 | Yes | the `evictable` `OrderedDict`, oldest first |

Every allocator bug is a violation of that table, which is why `check_invariants()` exists
and why the randomised property test calls it after every single operation.

**Plain.** Think of a library with identical numbered shelves. A shelf is empty, or someone
is using it, or it still holds a book nobody is reading right now but which someone might
ask for again. That third state is the whole trick: you don't throw the book out the moment
the reader leaves, you keep it until you need the shelf.

**Technical.** The state is represented redundantly and the redundancy is checked:
`ref[b]` (an int list), `hash_of[b]` (hash or `None`), `by_hash` (hash -> canonical block),
`free` (a deque), `evictable` (an `OrderedDict` used as an LRU keyed by release time).
`num_available()` is `len(free) + len(evictable)` — cached blocks count as available
because they can be reclaimed.

### `cached_prefix_blocks(hashes)`

Walks the hash chain, stops at the first miss. `O(matched blocks)` dictionary lookups. The
early stop is correct only because of the chain property above.

### `plan_prompt(hashes, num_tokens)`

```python
hit = self.cached_prefix_blocks(hashes)[: (num_tokens - 1) // bs]
need_new = -(-num_tokens // bs) - len(hit)
```

**Plain.** A dry run: "if I admitted this request right now, which shelves could I reuse and
how many new ones would I need?" It changes nothing. The odd part is the deliberate refusal
to reuse *everything* — it always leaves at least one token's worth of work to do.

**Technical.** Read-only, so the scheduler can ask "would this fit?" without side effects.
The `(num_tokens - 1) // bs` cap is not an off-by-one; it is a requirement. If the entire
prompt were served from cache, the prefill forward pass would have zero tokens to run and
would produce no logits, so there would be nothing to sample the first output token from.
Capping the reuse one block short guarantees at least one token is recomputed. vLLM does
the same thing for the same reason.

`-(-a // b)` is ceiling division.

### `allocate_prompt(hashes, num_tokens)`

The only place blocks are handed out.

```python
hit, need_new = self.plan_prompt(hashes, num_tokens)
pinned = sum(1 for b in hit if self.ref[b] == 0)
if need_new > self.num_available() - pinned:
    return None
```

**Plain.** Before taking new shelves, notice that some of the books you want to reuse are
currently on the "nobody's reading it" pile. The moment you claim them, they stop being
available for anyone else to clear. So the shelves you can still take is availability minus
the ones you're about to claim. Forget that subtraction and you promise the same shelf to
two people.

**Technical.** `pinned` counts hit blocks currently sitting in `evictable` with `ref == 0`.
Those are inside `num_available()`, but claiming them removes them from it. Without the
subtraction, a request whose prefix is entirely in the LRU would be admitted when
`need_new` exceeds the truly free count, and `_take()` would then evict blocks — possibly
the very cached blocks this request just pinned, if the pinning hadn't happened first.
Ordering matters: hit blocks are pinned (removed from `evictable`, ref incremented)
*before* any `_take()` runs.

Returns `None` with no side effects when it doesn't fit — the scheduler depends on that,
because for LPM it skips this request and tries the next one, and a half-applied allocation
would corrupt the pool.

Returns `(blocks, cached_token_count)` where `blocks` is the hit blocks followed by the new
ones, in logical order, which is exactly the block table.

### `_take()`

```python
if self.free: return self.free.popleft()
block, _ = self.evictable.popitem(last=False)
del self.by_hash[self.hash_of[block]]
self.hash_of[block] = None
```

**Plain.** Prefer an empty shelf. If there are none, clear the book nobody has read for the
longest, and cross it out of the catalogue.

**Technical.** `popitem(last=False)` takes the front of the `OrderedDict`, which is the
least recently released block. Removing the `by_hash` entry *and* clearing `hash_of` keeps
the two directions of the mapping consistent; `check_invariants` asserts exactly that. A
block returned by `_take` is always unhashed and unreferenced.

### `release(blocks)` — the reverse-order line

```python
for b in reversed(blocks):
    self.ref[b] -= 1
    if self.ref[b] == 0:
        if self.hash_of[b] is not None: self.evictable[b] = None
        else: self.free.append(b)
```

**Plain.** When a reader finishes, put their books back on the "might be wanted again" pile
*starting from the last page*. The pages at the end are the ones only they were reading;
the pages at the front are the shared introduction everybody needs. Since the pile is
cleared oldest-first, putting the private ending in first means the private ending is what
gets thrown away first, and the shared introduction survives.

**Technical.** `reversed` puts the tail blocks into the LRU before the head blocks, so they
sit closer to the eviction end. A finished request's last blocks are its private suffix and
its generated tokens — content nothing else will ever ask for. Its first blocks are the
shared prefix that the next request in the same group needs. Forward-order release inverts
that priority and evicts the shared prefix first, which is precisely the thrashing the
whole project is about. This is one keyword and it is a real contributor to hit rate.

### `commit(block, h)`

```python
if self.hash_of[block] is None and h not in self.by_hash:
    self.hash_of[block] = h
    self.by_hash[h] = block
```

**Plain.** Add the book to the catalogue — but only once it is completely written, and only
if no copy is already catalogued. If two people happened to write identical copies at the
same time, the first one catalogued wins and the second stays a private copy. Don't try to
merge them; that way lies pain.

**Technical.** Two guards. `hash_of[block] is None` makes it idempotent, so calling it
every step on already-committed blocks is free. `h not in by_hash` handles duplicate
content: two requests can be computing the same block concurrently before either commits.
The first becomes canonical; the second stays unhashed, is never shared, and on release
goes to `free` rather than `evictable`. Trying to merge them would mean rewriting a running
request's block table mid-flight, which is a class of bug not worth opening.

### `check_invariants()`

`O(num_blocks)`, off by default, on under `--debug-invariants` and in every property test.
It asserts the three-state table directly: no block in both `free` and `evictable`, no
duplicate in `free`, free blocks unhashed and unreferenced, evictable blocks hashed and
unreferenced, no block with ref 0 that is in neither list, and `by_hash` and `hash_of`
agreeing in both directions.

---

## `mercury/request.py`

### `RequestState`

`WAITING -> RUNNING -> FINISHED | ABORTED`, plus the edge that makes this interesting:
`RUNNING -> WAITING` on preemption.

### `Request`

**Plain.** One customer's order slip: what they asked for, when they walked in, which
shelves they've been given, how much of the answer is written, and whether they've ever
been bumped out of the queue.

**Technical.** The fields that carry meaning beyond the obvious:

- `hashes` — the chain over `tokens`. Recomputed on preemption because the request resumes
  as a *longer* prompt: prompt plus whatever it already generated.
- `num_cached` — tokens served from cache on the current admission. The prefill runs over
  `tokens[num_cached:]` and rotary positions start at `num_cached`.
- `committed` — how many of this request's blocks have been registered in the prefix cache.
  Monotonic within an admission, reset to 0 on preemption because the blocks were released.
- `admitted_at` vs `last_admitted_at` — the first is set once and never overwritten, so
  queueing delay measures the wait from arrival to *first* service. The second tracks
  re-admissions after preemption.
- `tokens` is `prompt + out`, and `seq_len` is their combined length. The whole engine
  treats a request as one flat token sequence, which is why resuming after preemption needs
  no special case: it is just a prefill with a longer prompt.

`record()` produces the per-request dictionary the `/stats/requests` endpoint serves and
the analysis reads for the fairness metric.

---

## `mercury/scheduler.py`

### `Plan`

`prefill` (newly admitted, run a prefill forward), `decode` (already running, one token
each), `sched_seconds`, `preempted`. The runner's entire input.

### `Scheduler.add` / `abort` / `drop`

`add` uses `bisect.insort(..., key=lambda x: x.arrival)` to keep `waiting` sorted by
arrival. That matters twice: FCFS is then a no-op sort, and `waiting[:scan_limit]` is
genuinely the oldest 128 rather than an arbitrary 128.

`abort` scans both queues, releases blocks, marks the state. `drop` is the normal-completion
path: release and remove. Both go through `alloc.release`, so both get the reverse-order
behaviour.

### `_preempt(r)` — recompute-style preemption

```python
self.running.remove(r)
self.alloc.release(r.blocks)
r.blocks, r.num_cached, r.committed = [], 0, 0
r.preemptions += 1
r.hashes = block_hashes(r.tokens, self.alloc.block_size)
self.add(r)
```

**Plain.** When the library runs out of shelves, someone has to give theirs up. The person
who most recently sat down is asked to leave — but their books go back on the "might be
wanted again" pile, not in the bin. When they come back, a lot of what they need is often
still there, so restarting is cheaper than it sounds.

**Technical.** Recompute-style rather than swap-to-CPU. The victim's blocks are released
normally, so any committed ones remain in the prefix cache; when it is re-admitted,
`allocate_prompt` may hit most of them and the "recompute" is nearly free. The hash chain
is rebuilt over `prompt + out` because the request now resumes as a longer prompt.
Re-inserted by arrival time, so it lands near the front of the waiting queue rather than at
the back — a preempted request should not lose its place.

The alternative, swapping KV to pinned host memory and back, needs pinned buffers, async
copies, and a second memory pool. Recompute plus prefix caching gets most of the benefit for
a fraction of the machinery.

### `_score(r, now)` — the three policies, and only here

```python
if self.policy == "fcfs":
    return -r.arrival
cached = len(self.alloc.cached_prefix_blocks(r.hashes)) * self.alloc.block_size
if self.policy == "lpm":
    return float(cached)
waited = now - r.arrival
if waited >= self.max_wait_s:
    return float("inf")
return cached + self.aging * waited
```

**Plain.** Three ways to decide who gets served next. First: whoever has been here longest.
Second: whoever needs the least new work, because their ingredients are already on the
counter. Third: the second one, but everyone waiting earns credit as they wait, and anyone
who has waited past a hard limit jumps straight to the front no matter what.

**Technical.** `alpha` (`aging_tokens_per_s`) is an exchange rate with units of tokens per
second: how many cached tokens one second of waiting is worth. At alpha = 2,000 a request
that has waited just over a second outranks a freshly arrived request carrying a
2,000-token cache hit.

There is a subtlety worth knowing, and it is why the hard deadline is not redundant. Two
requests both age at the same rate, so a fixed gap in arrival times gives a *constant*
advantage that aging never closes. Aging only wins against arrivals that keep being new —
which is the actual starvation scenario, a continuous stream of hot requests. Against any
single competitor with a fixed head start, the aging term alone never flips the order. The
deadline is what converts "usually fair" into a guarantee. Both of these are tests:
`test_a_constant_arrival_gap_is_never_closed_by_aging_alone` and
`test_the_deadline_makes_a_starving_request_outrank_everything`.

Returning `-r.arrival` for FCFS lets a single sort key handle all three policies: the caller
sorts by `(-score, arrival)`, which for FCFS becomes `(arrival, arrival)`.

### `schedule(now)` — step 1: reserve decode room, preempt if needed

```python
for r in list(self.running):
    while r.state is RequestState.RUNNING and r.seq_len > len(r.blocks) * bs:
        block = self.alloc.append_block()
        if block is not None:
            r.blocks.append(block); continue
        victim = self.running[-1]
        self._preempt(victim)
decode = list(self.running)
```

**Plain.** Before anyone writes their next word, make sure they have a page to write it on.
If there's no paper left, take it from whoever sat down most recently — possibly yourself,
in which case you stand up and wait your turn again.

**Technical.** The condition `seq_len > len(blocks) * bs` asks whether the token about to be
written — at logical position `seq_len - 1` — falls past the last block this request owns.
Normally it's true once every 16 decode steps.

Termination is the thing to check. Preempting `running[-1]` either frees blocks from
someone else (so the next `append_block` succeeds, since a running request always holds at
least one block) or preempts `r` itself, which flips its state and ends the `while`. It
cannot spin. The state check rather than an `r in self.running` membership test keeps the
loop O(1) per iteration instead of O(n).

`decode = list(self.running)` is taken *after* the preemption loop, so a request preempted
this step is correctly excluded from this step's decode batch. Getting that order wrong
would run a forward pass over a request whose blocks were just handed to someone else —
fluent, wrong output, no error.

Victim selection is newest-first. The alternative, oldest-first, would throw away the most
accumulated work.

### `schedule(now)` — step 2: admission

```python
head, tail = self.waiting[:self.scan_limit], self.waiting[self.scan_limit:]
order = sorted(head, key=lambda r: (-self._score(r, now), r.arrival)) + tail
```

**Plain.** Look at the first 128 people in line, decide the order among just those, and
leave everyone behind them where they are. Scoring the whole line every time would cost more
than it saves when the line is long.

**Technical.** Scoring is `O(waiting x prefix blocks)` dictionary lookups per step, and the
Feather paper's finding is that this can rival GPU time. SGLang caps it the same way,
falling back to FCFS past 128 waiting requests. `scan_limit_exceeded` counts the steps where
the queue was deeper than the window, so the write-up can report how often the cap bound.

Then the admission loop, with three ways to stop and one to skip:

```python
if len(self.running) + len(prefill) >= self.max_running: break
hit, _ = self.alloc.plan_prompt(r.hashes, r.seq_len)
new_tokens = r.seq_len - len(hit) * bs
if prefill and new_tokens > budget: break
got = self.alloc.allocate_prompt(r.hashes, r.seq_len)
if got is None:
    starving = self.policy == "lpm_aging" and self._score(r, now) == float("inf")
    if self.policy == "fcfs" or starving: break
    continue
```

- **Batch budget.** Bounds decode batch size, which bounds the gather in paged attention.
- **Prefill token budget.** Bounds how much prefill work lands in one step. `if prefill and`
  means the first request is admitted regardless — otherwise a request larger than the whole
  budget could never start.
- **Doesn't fit.** FCFS stops entirely: strict head-of-line order, same as vLLM's default.
  LPM skips and tries the next, because a smaller or more-cached request may fit. `lpm_aging`
  skips too, *unless* this request is starving, in which case it stops and holds memory so
  the request gets in as soon as the running batch frees enough.

That last branch is the honest limit of the starvation bound, and the walkthrough should say
so plainly: the deadline bounds *scheduling* delay, not *memory* delay. Once a starving
request outranks everything, admission stops — but it still cannot be admitted until enough
blocks free up. The real guarantee is "max_wait_s plus the time for the running batch to
release enough memory", and the test asserts that bound rather than pretending it is one
step.

Admitted requests are removed from `waiting` by rebuilding the list against a set of `id()`s
rather than calling `list.remove` per request, which would be O(n) each.

---

## `mercury/engine.py`

### `Engine.__init__` and the runner seam

The engine builds the allocator, the scheduler and the metrics, then takes a runner — and
if none is passed, imports and constructs `ModelRunner` lazily. That lazy import is load
bearing for the test suite: importing `mercury.engine` must not pull in safetensors or the
model loader, or none of the CPU tests could run.

`self.eos_ids` comes from the model config, and is empty for the deterministic runner.

### `submit(...)` — the ceiling check

```python
ceiling = min(self.cfg.max_model_len, self.alloc.num_blocks * self.cfg.block_size)
if len(prompt_ids) + max_tokens > ceiling:
    self._reject(...)
```

**Plain.** If someone orders more than the kitchen could ever cook, tell them now rather
than letting them sit in the queue forever.

**Technical.** This is the request-level half of the livelock guard. A request that cannot
fit in an empty pool would be admitted, need a block, find none, and preempt itself
forever. Rejecting it at submit turns a hang into a 400. Note the check is against the pool
size *and* `max_model_len` — whichever binds first.

The hash chain is computed here, on the calling thread, once. That keeps hashing off the
engine thread and out of the per-step cost.

### `step()`

The loop body. Drains commands, schedules, executes, routes output, records one
`StepRecord`. It returns `None` when there is nothing to do, which is the signal the thread
loop uses to sleep on an event rather than spin — a real consideration on Kaggle, where CPU
cores are few and shared with the load generator.

### `_commit_full_blocks(r)` — when a block becomes shareable

```python
full = r.seq_len // bs
if full <= r.committed: return
extend_hashes(r.hashes, r.tokens, bs)
for index in range(r.committed, full):
    self.alloc.commit(r.blocks[index], r.hashes[index])
r.committed = full
```

**Plain.** A page goes in the catalogue the moment it's completely written, and not a moment
before. Catalogue it while there's still room to write and the next word you add will
change a page someone else is already reading.

**Technical.** The timing is exact and it matters. This runs *after* the runner wrote KV for
positions `0..seq_len-1` and *before* the newly sampled token is appended. At that instant
`seq_len // bs` blocks are completely written, and `tokens[0:seq_len]` are all known, so the
hashes for those blocks can be computed from data that will never change.

Commit one step later and you'd hash a block whose last slot hasn't been written — a sharer
would attend over uninitialised memory. Commit one step earlier, same problem. The
`committed` counter makes it O(new blocks) per step rather than O(all blocks).

This is also where generated tokens enter the cache, which is what lets a multi-turn agent's
next turn reuse the previous turn's answer.

### `_accept`, `_finish`, `_emit`

`_accept` commits, appends the token, stamps `first_token_at` on the first one, emits, and
checks the two stop conditions (`max_tokens`, or EOS when `ignore_eos` is false).

`_finish` releases blocks *after* the commit has already run, so committed blocks land in
`evictable` and stay cacheable rather than going straight to `free`.

`_emit` looks up the sink under a lock and pops it when closing, so a request's stream is
closed exactly once. If a sink raises — a dead client, a full queue — it aborts the request
rather than letting the exception kill the engine thread.

### `_check_running_blocks()`

Under `--debug-invariants`, asserts every block in every running request's table has
`ref >= 1`. This is the direct test for "the block table points at a block that was evicted
and reused", which is the failure that produces fluent wrong text. It is off in production
because it is O(running x blocks) per step.

---

## `mercury/runner.py`

### `slot_mapping(blocks, start, stop, block_size)`

```python
positions = np.arange(start, stop, dtype=np.int64)
table = np.asarray(blocks, dtype=np.int64)
return table[positions // block_size] * block_size + (positions % block_size)
```

**Plain.** Turn "the 37th word of this conversation" into "shelf 9, slot 5". The book's
pages are scattered across the library, and this is the index card that says where each one
physically lives.

**Technical.** Logical position -> physical slot in the flattened KV cache. Vectorised in
numpy because it runs for every token of every prefill and every request of every decode
batch, and a Python loop here is visible in the step time at 0.5B.

The off-by-one that this function's docstring warns about: the decode token's position is
`seq_len - 1`, not `seq_len`. Using `seq_len` writes one slot too far, which stays inside
the same block for fifteen out of sixteen tokens — so the output is correct for exactly one
block and then becomes garbage. `test_slot_mapping_at_block_boundaries` pins positions 15,
16, 17, 31 and 32 for exactly this reason.

### `ModelRunner.__init__`

Loads the model (unless injected), seeds the generators, and allocates the KV cache: per
layer, a K and a V tensor of shape `[num_blocks, block_size, num_kv_heads, head_dim]`.
Zero-filled rather than `empty` so that a bug reading unwritten memory produces zeros rather
than plausible noise.

### `execute(plan)` — prefill

```python
for r in plan.prefill:
    start = r.num_cached
    ids = np.asarray(r.tokens[start:], dtype=np.int64)
    positions = np.arange(start, r.seq_len, dtype=np.int64)
    slots = self._slots_for(r.blocks, start, r.seq_len)
    ctx = PrefillContext(self.kv_caches, block_table, slots, r.seq_len, start)
    hidden = self.model(id_tensor, pos_tensor, ctx)
    prefill_hidden.append(hidden[-1:])
```

**Plain.** For each newly admitted request, feed the model only the part of the prompt that
isn't already cached — but tell it the true position of each word, because a word's meaning
depends on where it sits in the whole conversation, not where it sits in the chunk you
happen to be feeding.

**Technical.** Three things are doing real work here:

- `positions` starts at `num_cached`, not at 0. Rotary embeddings are absolute. Restarting
  at zero is the single most likely prefix-caching bug and its symptom is output that reads
  fine and drifts.
- Only `hidden[-1:]` is kept. The vocabulary is 151,936; running `lm_head` over all `Tq`
  prefill positions would be a `[Tq, 151936]` matrix for no reason, since only the last
  position's logits produce a token.
- Prefill requests are looped one at a time. Prefill is compute-bound, so batching them buys
  little, and doing them separately keeps the attention context per-request and simple.

The `lm_head` for all prefill requests is then run once over the concatenated last hidden
states, so a step with eight new requests does one big matmul rather than eight small ones.

### `execute(plan)` — decode

One row per running request: last token id, position `seq_len - 1`, its slot, its sequence
length, and its block table padded to the widest in the batch.

```python
tables[i, :owned] = r.blocks
tables[i, owned:] = r.blocks[-1]
```

**Plain.** Every request's list of shelves has to be the same length to make one
rectangular table, so short lists get padded. The padding repeats a shelf the request
already owns, so even if the padding is read it's reading the request's own memory — and it
gets masked out anyway.

**Technical.** Padding with a *valid* block id is required: an out-of-range id would index
past the end of the cache tensor. Padding with the request's own last block means the worst
case is reading its own stale slots, which the `arange(L) < seq_len` mask discards.
`test_padding_a_block_table_does_not_leak_into_the_output` proves it by filling a padding
block with 1e4 and checking the result is bit-identical to the unpadded call.

### The prepare/forward timing split

`prepare` accumulates only the time spent building tensors; a single `torch.cuda.synchronize()`
at the end makes `forward` the honest remainder. Building block tables and slot mappings in
Python every step is real overhead at 0.5B — production engines hide it with persistent
buffers and CUDA graphs — so it is measured separately rather than hidden inside "step
time".

### `forward_prompt(token_ids)`

Runs a prompt through the paged path using scratch blocks borrowed from the allocator with
an empty hash list (so it can never hit or pollute the cache), and releases them in a
`finally`. The parity test uses it to compare first-step logits against HuggingFace.

---

## `mercury/model/rope.py`

### `rotate_half` and `apply_rope`

**Plain.** Rotary embeddings encode a word's position by physically rotating its query and
key vectors by an angle proportional to the position. Two words three apart are rotated by
the same relative angle no matter where in the sentence they are, so the model learns
relative distance for free.

**Technical.** The NeoX layout: split the head dimension in half and treat
`(x[:d/2], x[d/2:])` as the real and imaginary parts.
`q' = q * cos + rotate_half(q) * sin` is complex multiplication by `e^{i*theta}`.
`cos`/`sin` are `[T, d]`, unsqueezed to `[T, 1, d]` to broadcast across heads.

`test_rope_preserves_the_dot_product_of_equally_shifted_positions` checks the property that
makes RoPE work at all: `<q_m, k_n>` depends only on `m - n`.

### `RotaryEmbedding`

Precomputes the whole `cos`/`sin` table up to `max_model_len` once, in fp32, then casts to
the working dtype. `lookup(positions)` is an `index_select`. Computing the table per step
would be pure waste; more importantly, indexing by absolute position is what makes the
cached-prefix case work — the suffix looks up positions `num_cached..seq_len-1` and gets
exactly the angles it would have got in a full prefill.

---

## `mercury/model/attention.py`

### `_sdpa(...)` and the GQA fallback

Qwen2.5-0.5B has 14 query heads and 2 KV heads: grouped-query attention, each KV head
serving seven query heads. `enable_gqa=True` lets PyTorch handle that without materialising
the expansion. It arrived in torch 2.5, so the helper tries it once, catches `TypeError`,
flips a module flag, and from then on falls back to `repeat_interleave`. One try, not one
per call.

### `write_kv(k_cache, v_cache, k, v, slots)`

```python
k_cache.view(num_blocks * block_size, *k_cache.shape[2:]).index_copy_(0, slots, k)
```

**Plain.** Take the shelf-and-slot grid and squint at it as one long row of slots, then drop
each new key and value into its numbered slot.

**Technical.** `.view()` on the contiguous `[num_blocks, block_size, n_kv, d]` cache gives a
`[num_blocks * block_size, n_kv, d]` view over the same storage, so `index_copy_` writes
through to the cache. This is why `slot_mapping` produces `block * block_size + offset`:
that flat index is the whole interface between the allocator's block ids and physical
memory.

### `paged_decode_attention(...)`

```python
k = k_cache[block_tables].reshape(batch, -1, num_kv_heads, head_dim)
valid = torch.arange(span, device=q.device)[None, :] < seq_lens[:, None]
out = _sdpa(q.unsqueeze(2), k.transpose(1, 2), v.transpose(1, 2), valid[:, None, None, :], scale)
```

**Plain.** Each request's pages are scattered around the library. To do the maths, gather
each request's pages into one contiguous stack, pad the stacks to equal height, and mark
which rows are real. Then do ordinary attention and ignore the marked-out rows.

**Technical.** `k_cache[block_tables]` is advanced indexing, so it **copies**:
`[B, max_blocks, block_size, n_kv, d]`, reshaped to `[B, L, n_kv, d]` where
`L = max_blocks * block_size`. That copy is the deliberate trade in this project. A fused
paged-attention kernel reads the blocks in place; this gathers them first, costing bandwidth
and memory proportional to batch times longest sequence, every layer, every step.

It is slower and it is *identical across all three scheduling policies*, so it cannot bias
the comparison that carries the finding. The interface (`PrefillContext` / `DecodeContext`)
exists so a faster backend can be dropped in without touching the model.

The mask is `arange(L) < seq_len`, which simultaneously handles two kinds of junk: the
unwritten tail of the last real block, and the padding blocks past the end of the table.

### `paged_prefill_attention(...)` — the explicit mask

```python
if num_cached == 0:
    out = _sdpa(q4, k4, v4, None, scale, is_causal=True)
else:
    query_pos = torch.arange(q.shape[0], device=q.device) + num_cached
    key_pos = torch.arange(seq_len, device=q.device)
    mask = key_pos[None, :] <= query_pos[:, None]
    out = _sdpa(q4, k4, v4, mask[None, None], scale)
```

**Plain.** Normally "you may only look backwards" is a neat triangle, because the first word
you're processing is also the first word of the sentence. But when the first 2,000 words are
already cached and you're only processing words 2,001 onwards, the triangle is in the wrong
place: word 2,001 may look at all 2,000 before it, not at zero of them. So you write the
rule out explicitly instead of using the built-in shortcut.

**Technical.** `is_causal=True` assumes queries and keys are aligned at the same origin. With
a cached prefix, queries start at `num_cached` and keys at 0, so the built-in mask would
forbid the suffix from attending to the cached prefix at all — and produce output that is
grammatical and wrong. The explicit mask is `key_pos <= num_cached + query_index`.

The `num_cached == 0` branch is not just an optimisation: at full prefill lengths the
explicit mask would be `[8000, 8000]` booleans, 64 MB, for no benefit. When there is a
cached prefix, `Tq` is the short suffix so the mask is small.

`test_prefill_on_top_of_a_cached_prefix_matches_the_full_prefill` is the test that pins
this: it computes a full 48-token prefill by an explicit reference, then computes only the
last 16 positions with `num_cached=32`, and asserts the two agree.

### `PrefillContext` and `DecodeContext`

**Plain.** The model layers shouldn't have to know anything about shelves and slots. These
two small objects know, and hand the layer back a plain answer.

**Technical.** Both expose `attend(layer, q, k, v, scale)`. Each writes the new K/V into the
cache and then runs the appropriate attention. This is what keeps `qwen2.py` free of paging
logic and what makes the model's activations a flat `[num_tokens, hidden]` — the batch
structure lives entirely in the context.

---

## `mercury/model/qwen2.py`

Module names match HuggingFace's exactly (`model.layers.0.self_attn.q_proj`), so weight
loading is a direct name lookup with no translation table to get wrong.

### `Qwen2RMSNorm.forward(x, residual)` — returning two things

```python
if residual is not None: x = x + residual
variance = x.float().pow(2).mean(-1, keepdim=True)
normed = (x.float() * torch.rsqrt(variance + self.eps)).to(x.dtype)
return normed * self.weight, x
```

**Plain.** Rescale a vector so its typical size is one, then apply a learned per-channel
gain. It also hands back the un-normalised sum, because the next layer needs that too — and
returning both is what lets the add and the normalise become a single GPU kernel.

**Technical.** Returns `(normalized, residual)`. The residual returned is the *sum*, not the
normalised value — the classic fused-residual pattern. The first layer passes `residual=None`
and gets back `x` itself, so the stack needs no special case.

Reductions happen in fp32 even though the tensors are fp16: `x.float().pow(2).mean(-1)` over
896 channels in fp16 would lose precision badly enough to shift logits. The cast back to
fp16 happens before multiplying by the weight, matching HuggingFace's ordering, which is
what keeps parity tight.

When the fused path is enabled and the input is fp16 on CUDA, the same call goes to the
Triton kernel instead.

### `Qwen2Attention.forward`

Projects to `[T, n_q, d]` and `[T, n_kv, d]`, applies rotary, delegates to
`ctx.attend(...)`, reshapes, and projects out. Qwen2 has biases on q/k/v and none on the
output projection — an architecture detail that produces a silent accuracy loss if you get
it backwards, since the shapes all still work.

`T` is the flat token count: the suffix length for prefill, the batch size for decode. The
same code serves both because the context knows which.

### `Qwen2ForCausalLM` and the tied embedding

```python
self.lm_head = None if cfg.tie_word_embeddings else nn.Linear(...)

@property
def output_embedding(self):
    return self.model.embed_tokens.weight if self.lm_head is None else self.lm_head.weight
```

**Plain.** For this model the table that turns token ids into vectors and the table that
turns vectors back into token scores are literally the same table. So don't make a second
one.

**Technical.** Qwen2.5-0.5B ties them. The obvious implementation — build an `nn.Linear` and
assign `lm_head.weight = embed_tokens.weight` — breaks under `to_empty()`, because
`Module._apply` walks each module's own `_parameters` and would create two separate storages,
silently untying them and then failing to load one of them. Not creating the module at all
sidesteps the whole problem, and `output_embedding` makes the intent explicit.

---

## `mercury/model/loader.py`

### `build_model`

```python
torch.set_default_dtype(engine_cfg.torch_dtype)
with torch.device("meta"):
    model = Qwen2ForCausalLM(model_cfg, fused_norm=engine_cfg.fused_rmsnorm)
torch.set_default_dtype(previous)
model.to_empty(device=device)
```

**Plain.** Build the whole model as a blueprint with no actual material in it, then allocate
the material once and pour the weights straight in. Building it normally would fill a
gigabyte with random numbers and immediately overwrite them.

**Technical.** Meta-device construction allocates no storage. `to_empty` then allocates
uninitialised storage of the right dtype and shape. The `set_default_dtype` dance is what
makes the meta modules fp16 in the first place, since `nn.Linear` takes the default dtype.

The consequence is that **every** parameter must be loaded, because uninitialised memory is
whatever was there. `load_weights` tracks exactly which names it filled and raises if any
parameter went unfilled — a missing-tensor bug that would otherwise surface as a model that
runs and produces garbage.

### `load_weights`

Iterates the safetensors files, matches names against
`named_parameters(remove_duplicate=False)`, checks shapes, copies. Raises on missing
parameters and on unexpected tensors, with one allowed exception: `lm_head.weight` is
expected to be absent when embeddings are tied. Strict in both directions, because both
directions of mismatch mean you are not running the model you think you are.

### `resolve_model`

A local directory is used as-is; otherwise `snapshot_download` with an allow-list and a
pinned revision. It returns the resolved commit hash, which goes into every result file —
"which weights produced this number" is not a question you want to be unable to answer.

---

## `mercury/kernels/rmsnorm_quant.py`

### Why RMSNorm and not an INT4 matmul

On GPUs below compute capability 8.0, Triton's `tl.dot` falls back to plain FMA on CUDA
cores instead of tensor cores. An INT4 matmul written in Triton on a T4 would therefore race
cuBLAS's tensor-core fp16 GEMM without tensor cores, and lose, for reasons that have nothing
to do with the quality of the code. RMSNorm is a memory-bound reduction with no `dot`, so it
lives on the well-supported part of Triton and the benefit of fusion shows up cleanly.

### `_add_rmsnorm_quant` — the kernel

One program per token row. Loads `x`, optionally adds the residual and writes the new
residual back, computes the reciprocal standard deviation, applies the weight, and either
stores fp16 or quantises to INT8 with one fp32 scale per row.

**Plain.** In every layer the model does the same three small things back to back: add two
vectors, measure the result's size, and divide by it. Done separately, each of those reads
the whole vector from memory and writes it back — three round trips for arithmetic that
takes no time at all. This does all three while the data is in registers, so the vector
crosses the memory bus once.

**Technical.** `BLOCK = next_power_of_2(N)` — 896 rounds to 1024, 1536 rounds to 2048 — with
a mask so the padding lanes load 0.0 and contribute nothing to the sum or the max.
Accumulation is fp32 regardless of the fp16 storage.

`(x * rstd).to(tl.float16).to(tl.float32) * w` casts to fp16 *before* applying the weight,
matching HuggingFace's `self.weight * hidden_states.to(input_dtype)`. Skipping that cast
changes the last bits of the output and widens the gap against the parity reference for no
gain.

Quantisation is symmetric per token: `scale = max(|y|) / 127`, then round half away from
zero via `where(q >= 0, q + 0.5, q - 0.5)` followed by Triton's truncating cast, then clamp
to +/-127. The `tl.maximum(..., 1e-6)` floor prevents a division by zero on an all-zero row —
a real case, and a test.

### `add_rmsnorm_quant_torch`

The same arithmetic in the same order in PyTorch. It is both the correctness reference for
the kernel and the eager competitor in the benchmark; `torch.compile` of this same function
is the *honest* competitor, because Inductor fuses this pattern into a Triton kernel too.

The expected and reported result: the hand-written kernel and Inductor roughly tied, both
well ahead of eager at small row counts where launch overhead dominates, and a single-digit
percent end-to-end decode improvement because norms are a small slice of a decode step.
"I matched Inductor and measured that it didn't matter much end to end" is a stronger
engineering claim than a cherry-picked microbenchmark.

The INT8 mode is benchmarked as a component and is not used in production, because Mercury
has no INT8 matmul to consume its output. Saying that out loud is part of the deliverable.

### `bytes_moved`

The traffic a fused implementation is obliged to move. Dividing it by measured time gives
effective bandwidth, and the fraction of the T4's 320 GB/s peak is the number that says
whether a memory-bound kernel is any good — far more informative than raw microseconds.

---

## `mercury/sampling.py`

`sample_tokens(logits, temperatures, top_ps, generator)`. Takes the fast path — a single
`argmax` — when every row is greedy, which is every benchmark run. Otherwise scales by
temperature in fp32, softmaxes, optionally applies nucleus filtering, renormalises, and
draws. `torch.where(greedy, argmax, drawn)` merges the two so a mixed batch works without
splitting it.

`_top_p_mask` sorts descending, takes the cumulative sum, and zeroes everything after the
token that crosses `p` — using `(cumulative - ordered) > p` rather than `cumulative > p` so
that the crossing token itself is kept. Off by one here and a `top_p` near zero would mask
every token and leave nothing to sample.

---

## `mercury/server.py`

### The sink closure

```python
def sink(event: StreamEvent) -> None:
    loop.call_soon_threadsafe(queue.put_nowait, event)
```

**Plain.** The engine runs on its own thread and the web server runs on another. This is the
letterbox: the engine drops a token in, and the web server's thread picks it up when it next
looks. The engine never reaches into the web server's world directly.

**Technical.** `call_soon_threadsafe` is the only safe way to touch an asyncio loop from
another thread. Calling `queue.put_nowait` directly would corrupt the loop's internal state
in ways that surface as sporadic hangs. This closure is the entire engine-to-HTTP interface.

### `event_stream()` and cancellation

```python
finally:
    if not settled:
        engine.abort(rid)
```

**Plain.** However this stream ends — finished normally, client closed the tab, network
died, server error — make sure the request's memory goes back. Leaked pages from cancelled
streams are the classic slow memory leak in a serving system: nothing looks wrong, and three
hours later the pool is empty.

**Technical.** Two mechanisms, deliberately. `await request.is_disconnected()` after each
token catches a client that has gone away while tokens are still flowing; task cancellation
(which raises inside `await queue.get()`) catches everything else. Both funnel through the
same `finally`. The `settled` flag stops a normally-completed request from enqueueing a
pointless abort.

This is tested against a **real uvicorn socket**, not an in-process ASGI transport. The
in-process transport never delivers `http.disconnect`, so the test would pass without the
cancellation path ever running — a test that proves nothing. `LiveServer` in
`tests/test_server.py` exists specifically because I found that.

### `/reset_stats`

Clears counters without touching the KV cache. This is not a convenience; it is what the
benchmark protocol needs. Warmup uses off-trace prompts so the server is warm but the cache
holds nothing the measured run will ask for — then the counters are zeroed so every token
counted belongs to the measured run.

### The startup line

```
[mercury] policy=lpm_aging kv_blocks=4096 block_size=16 kv_capacity_tokens=65536 ...
```

Machine-readable on purpose. `run_matrix.check_capacity` greps `kv_blocks=(\d+)` and refuses
to run a point whose cap did not take effect.

### `--fake-runner` and `--fake-step-delay`

Serves deterministic tokens with no model loaded. It lets the entire harness — server
process, HTTP, SSE, client, analysis — be exercised on a laptop.

`--fake-step-delay` matters more than it looks. With zero simulated GPU time, every request
is served the instant it arrives, the waiting queue is always empty, and **no scheduling
policy can do anything** — the harness correctly reports all three policies as identical.
That is one of the three failure modes to check for if the real experiment shows no effect,
and having it reproducible on a laptop is useful.

---

## `bench/workloads.py`

### `TraceRequest` and `shared_prefix_tokens`

Each request records how many of its prompt tokens belong to the shared working set, so
`private_tokens` is `prompt_len - shared_prefix_tokens + max_tokens`.

**Plain.** To size the cache you need to know which part of each request is "the bit
everyone shares" and which part is "the bit only this request needs". Counting the whole
prompt as private makes the pressure knob stop working.

**Technical.** I got this wrong first time and the symptom was diagnostic: W4's capacity
barely moved between pressure 0.5 and 4, because the headroom term (`max_running x mean
private tokens`) was computed off the full 4,200-token prompt and swamped the 11,000-token
working set. With the split correct, mean private drops to ~370 tokens and the shared
allowance scales cleanly as `W / rho`.

### The six workloads

| | Structure | What it tests |
| --- | --- | --- |
| W1 | Independent prompts, log-normal 200-1500 tokens | Control: prefix-aware must not lose |
| W2 | G groups, 2,000-token system prompt + 50-300 suffix, interleaved | The clean case; G sets the pressure |
| W3 | Sessions of 4-8 turns, each prompt is the whole history | Growing session-private prefixes |
| W4 | 3,000-token catalogue + one of 8 task blocks + suffix | Two-level tree sharing |
| W5 | 6,000-8,000-token prompts, low concurrency | Prefill-bound; stresses the gather |
| W6 | 100-300-token prompts, high rate | Scheduler overhead and batching |

Arrivals are open-loop Poisson except within a W3 session, where a turn genuinely waits for
the previous answer plus think time — that is how agents behave, and modelling it
open-loop would be wrong.

Every request has a fixed `max_tokens` with `ignore_eos`, so every engine generates exactly
the same number of tokens. Without that, an engine that stops early looks faster.

**The W3 honesty note.** Each turn's history uses the trace's *scripted* assistant tokens,
not what the engine actually generated, so every engine sees byte-identical prompts. The
cost is that a real agent would also reuse the KV of its own previous answer, and this setup
only reuses prompt-side history. It understates caching for agents, applies equally to every
engine, and is the price of a fair comparison. It belongs in the write-up, not in a footnote.

### `capacity_for_pressure(meta, rho, max_running, block_size)`

```python
shared = meta.working_set_tokens / rho
private = max_running * meta.mean_private_tokens
return ceil((shared + private) / block_size)
```

**Plain.** Decide how crowded the cache should be. Give it exactly enough room for the
shared material divided by how crowded you want it, plus enough elbow room for the requests
currently in flight to hold their own private stuff.

**Technical.** This defines the x-axis of the main figure. The headroom term is not padding:
without it, a high-pressure point starves on private suffixes rather than on shared
prefixes, which measures a different phenomenon and would still produce a plausible-looking
curve.

The working set itself is per-workload: `G x 2000` for W2; `3000 + 8 x 1000` for W4; for W3,
a sweep over session start and end times taking the maximum, at any instant, of the summed
peak history length of the sessions live at that instant.

---

## `bench/client.py`

### Open loop, and why the whole finding depends on it

```python
delay = request.arrival_time_s - (time.perf_counter() - self.t0)
if delay > 0: await asyncio.sleep(delay)
```

**Plain.** Customers arrive when they arrive. They do not politely wait for the previous
customer to leave before walking in. If they did, the queue would never grow, and a project
about queue ordering would have nothing to measure.

**Technical.** One asyncio task per session; each request goes out at its absolute scheduled
time regardless of whether earlier ones finished. A closed-loop client with N simulated
users caps offered load at what the server can serve, so queueing never builds and the whole
scheduling effect vanishes. This is the single most common way serving benchmarks
accidentally measure nothing.

W3 is the exception by design: a dependent turn waits for the previous request in its
session, then sleeps its think time.

### Token accounting

Every chunk's arrival time is recorded, giving per-token timestamps for TTFT, TPOT and ITL.
`usage.completion_tokens` is captured when an engine reports it (vLLM does, via
`stream_options`), so the streamed count can be cross-checked. `verify()` in `run_matrix`
flags any run where output counts disagree with what was requested.

### `warmup`

Fifty requests built from deliberately off-trace token ids, then `/reset_stats`. The server
ends up warm — CUDA context, allocator, Python caches — with a prefix cache holding nothing
the measured run will ask for.

### `cpu_seconds`

`getrusage` around the run. If the client is using more than 70% of a core it is part of the
measurement rather than an observer of it, and `verify()` says so.

---

## `bench/run_matrix.py`

### `Point.name`

The result filename, and therefore the resume key. It includes the `extra` dict, because
without that the alpha-sweep points and the `--enforce-eager` point would collide with their
plain counterparts and the second would be skipped as "already done" — a bug I caught by
asserting name uniqueness when expanding the grids.

### `Server` as a context manager

Starts the engine in its own process group and, on exit, `SIGINT`s the whole group with a
`SIGKILL` fallback. Process-group signalling matters because vLLM and SGLang spawn worker
processes that survive a plain `terminate()` and would then hold the GPU against the next
point.

### `check_capacity`

**Plain.** Ask the engine to use a small cache, then read its own startup message to confirm
it actually did. Flags get renamed between releases. If the cap silently didn't apply, every
point in the sweep is at the wrong pressure and the resulting figure is beautiful and
meaningless.

**Technical.** Per-engine regex lists against the captured startup log, with a 10% tolerance
for engines that round to their own block granularity. Two distinct failures are raised
separately: "never reported its capacity" (probably a renamed flag — go read the log) and
"ignored the capacity cap" (reported a number far from the request). Both refuse to run the
point rather than producing data.

### `DeviceMonitor`

`nvidia-smi --query-gpu=timestamp,memory.used,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu`
at 10 Hz for the duration of every run. Clocks cannot be locked without root on Kaggle, but
drift can at least be shown not to have happened, and peak memory is a reported metric.

### `run_matrix`

Expands the grid, **shuffles** it, skips points whose result file exists, and runs the rest.
The shuffle is not cosmetic: running all of FCFS then all of LPM would fold any thermal
drift or noisy-neighbour effect directly into the policy comparison. Shuffling spreads it
across conditions. The skip-if-exists behaviour is what lets a killed Kaggle session resume.

### `rate_sweep`

Finds `R_sat`, the rate where p95 TTFT goes vertical. The matrix then runs at 0.5 and 0.8 of
it. Below about 0.3 `R_sat` the waiting queue is nearly always empty and no policy can
reorder anything; above `R_sat` every policy is dominated by an unbounded queue. The
scheduling effect lives in between, and running the matrix without measuring this first is
how you get a null result you can't explain.

---

## `bench/analyze.py`

### `summarize(path)`

One result file to one row: hit rate, evictions per second, TTFT/TPOT/ITL/E2E percentiles,
queueing delay p99 and max, throughput, goodput, scheduler overhead share, peak GPU memory,
client CPU, and the verification verdict. Only requests with `status == "ok"` count.

Goodput is computed in the same pass as the percentiles rather than by zipping filtered
lists afterwards — the lists have different lengths (a request with one output token has no
TPOT) and zipping them silently misaligns. I wrote that bug and then fixed it.

### `bootstrap_ci(values, q)`

**Plain.** You measured a p95 from 500 requests. Would you have got a similar number from a
different 500? Resample the ones you have, with replacement, two thousand times, recompute
the p95 each time, and look at how much it moves. That spread is your error bar.

**Technical.** Vectorised: one `[iterations, n]` index matrix, one `np.percentile` along
axis 1. Returns the point estimate and the 2.5/97.5 percentiles of the bootstrap
distribution. Below eight samples it returns the point estimate and no interval, because a
CI from five numbers is decoration.

### `figure_pressure_curves`

Figure 1 is hit rate against pressure, one line per policy, one panel per workload, log
x-axis, a dotted vertical line at rho = 1. Figure 2 is p95 TTFT in the same layout, with
maximum queueing delay as a third panel. Bands show the spread across repetitions.

The log x-axis and the rho = 1 line are the point: the reader should be able to see at a
glance that the curves are together on the left and apart in the middle.

### `preregistered(runs)`

Computes the four comparisons written into the README before the experiment ran, and prints
PASS/FAIL for the two that have a threshold. Comparison 3 — all policies within noise at
rho = 0.5 — is the one designed to catch a measurement artifact, and if it fails the
headline is not trustworthy. Having the script say so out loud removes the temptation to
quietly not look.

---

## `tests/`

84 tests run on a laptop in about ten seconds; 39 more need a GPU.

- **`test_hashing.py`** — the chain property, that a later difference doesn't change earlier
  blocks, that incremental extension equals a full recompute, and the sha256 mode.
- **`test_allocator.py`** — the three-state table, the one-token-to-recompute rule, the
  reverse-release order (asserted directly: `list(evictable) == list(reversed(blocks))`),
  eviction order, the pinning arithmetic, duplicate content, and two randomised property
  tests that call `check_invariants()` after every operation across thousands of steps and
  then assert every block came home.
- **`test_scheduler.py`** — each policy's admission order, head-of-line blocking under FCFS
  versus skipping under LPM, the aging exchange rate, the constant-arrival-gap property, the
  deadline, scan-limit bounding, newest-first preemption, and two adversarial hot-stream
  tests: plain LPM starves a cold request, and `lpm_aging` on the identical stream bounds it.
- **`test_engine.py`** — exact token counts, ordering, the hand-computed hit rate (four
  requests sharing a 64-token prefix: 192 cached of 320 queried, exactly 0.6), partial blocks
  never shared, abort paths, the 10%-cancel leak test, and the oversize rejection.
- **`test_attention.py`** — slot mapping at every block boundary, `write_kv` placement,
  prefill and decode against an explicit einsum reference, the cached-prefix mask, physical
  placement invariance, and padding not leaking.
- **`test_server.py`** — the SSE contract, non-streaming, the cache across requests,
  `/reset_stats` semantics, error paths, and cancellation against a **real uvicorn socket**.
- **`test_bench_integration.py`** — a real server subprocess, the real replayer over HTTP,
  the real analysis: token counts, capacity verification in both directions, and that the
  pressure formula scales the shared allowance as `W / rho`.
- **`test_parity.py`** (GPU) — greedy output against HuggingFace in four modes: single,
  batched, warm prefix cache, and after forced preemption. A prefix-cache bug appears in
  exactly one of those, which is why all four exist.
- **`test_kernel.py`** (GPU) — fp16 within 2e-3, INT8 within one quantisation step, scales
  within 1e-5 relative, odd row counts for masking bugs, and the all-zero row.

### One result that came out of the tests

Building the multi-group scheduler test produced the whole finding in miniature, on CPU,
with a deterministic runner. Six prefix groups against a pool too small to hold them:

| Policy | Hit rate | Evictions | Requests finished |
| --- | --- | --- | --- |
| FCFS | 0.455 | 1,287 | 267 |
| LPM | 0.817 | 576 | 391 |
| LPM + aging | 0.602 | 1,188 | 337 |

The eviction counts are the mechanism made visible: FCFS evicts more than twice as often,
which is exactly the thrashing the hypothesis predicts. And aging recovers about 40% of the
gap between FCFS and LPM — that is the fairness cost, stated as a number, on a workload
where I control every variable.

A second result from the same work: with only **one** prefix group, all three policies are
identical to within 2%, because there is nothing to thrash between. That is now a test, and
it is the reason the main experiment uses several groups.

### And one from the smoke matrix

Running the full harness — real server process, real HTTP, real analysis — over a pressure
sweep with the deterministic runner turned up a calibration failure worth keeping. At
pressure 2 with 512-token shared prefixes and alpha = 2,000:

| Policy | Hit rate | p95 TTFT | Max queue | Output tok/s |
| --- | --- | --- | --- | --- |
| FCFS | 0.351 | 0.904 s | 1.231 s | 628 |
| LPM | 0.376 | 0.590 s | 1.781 s | 652 |
| LPM + aging | 0.265 | 0.666 s | 0.963 s | 642 |

The fairness half worked — maximum queueing fell from 1.78 s to 0.96 s. The locality half
went *backwards*, below FCFS. The reason is that alpha is an exchange rate and it was
miscalibrated: at 2,000 tokens per second against 512-token prefixes, a quarter second of
waiting outweighs a complete cache hit, so cache affinity is swamped immediately. Sweeping
it confirms the mechanism, monotone in both hit rate and throughput:

| alpha | 125 | 500 | 2,000 | 8,000 |
| --- | --- | --- | --- | --- |
| Hit rate | 0.344 | 0.307 | 0.281 | 0.269 |
| Output tok/s | 649 | 645 | 635 | 626 |

The rule: `alpha ~ shared prefix tokens / acceptable wait in seconds`. It is now in
`Scheduler._score`'s docstring rather than living as a magic default, and deriving it from
the observed mean cached-prefix length is on the next-version list.

Also worth noting: the pressure-0.5 sanity check passed exactly — all three policies at
0.685 hit rate, spread 0.000. That is the control working.
