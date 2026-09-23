from __future__ import annotations

import random

from mercury.allocator import BlockAllocator
from mercury.hashing import block_hashes
from mercury.request import Request, RequestState
from mercury.scheduler import Scheduler

from .conftest import build

BS = 16


def make_request(rid, prompt, arrival, max_tokens=8):
    return Request(
        rid=rid,
        prompt=list(prompt),
        max_tokens=max_tokens,
        arrival=arrival,
        hashes=block_hashes(prompt, BS),
    )


def warm(alloc, tokens):
    hashes = block_hashes(tokens, BS)
    blocks, _ = alloc.allocate_prompt(hashes, len(tokens))
    for index in range(len(tokens) // BS):
        alloc.commit(blocks[index], hashes[index])
    alloc.release(blocks)
    return blocks


def test_fcfs_admits_in_exact_arrival_order():
    alloc = BlockAllocator(64, BS)
    sched = Scheduler(alloc, policy="fcfs", max_running=8)
    for index in [3, 0, 4, 1, 2]:
        sched.add(make_request(f"r{index}", [index] * 32, arrival=float(index)))
    plan = sched.schedule(now=10.0)
    assert [r.rid for r in plan.prefill] == ["r0", "r1", "r2", "r3", "r4"]


def test_lpm_puts_the_cached_request_first():
    alloc = BlockAllocator(64, BS)
    shared = list(range(64))
    warm(alloc, shared)
    sched = Scheduler(alloc, policy="lpm", max_running=8)
    sched.add(make_request("cold", [999] * 80, arrival=0.0))
    sched.add(make_request("hot", shared + [7] * 16, arrival=1.0))
    plan = sched.schedule(now=2.0)
    assert [r.rid for r in plan.prefill] == ["hot", "cold"]
    assert plan.prefill[0].num_cached == 64


def test_fcfs_ignores_the_cache_when_ordering():
    alloc = BlockAllocator(64, BS)
    shared = list(range(64))
    warm(alloc, shared)
    sched = Scheduler(alloc, policy="fcfs", max_running=8)
    sched.add(make_request("cold", [999] * 80, arrival=0.0))
    sched.add(make_request("hot", shared + [7] * 16, arrival=1.0))
    plan = sched.schedule(now=2.0)
    assert [r.rid for r in plan.prefill] == ["cold", "hot"]


def test_fcfs_stops_at_a_request_that_does_not_fit():
    alloc = BlockAllocator(6, BS)
    sched = Scheduler(alloc, policy="fcfs", max_running=8)
    sched.add(make_request("big", list(range(160)), arrival=0.0))
    sched.add(make_request("small", list(range(32)), arrival=1.0))
    plan = sched.schedule(now=2.0)
    assert plan.prefill == []
    assert len(sched.waiting) == 2


def test_lpm_skips_past_a_request_that_does_not_fit():
    alloc = BlockAllocator(6, BS)
    sched = Scheduler(alloc, policy="lpm", max_running=8)
    sched.add(make_request("big", list(range(160)), arrival=0.0))
    sched.add(make_request("small", list(range(32)), arrival=1.0))
    plan = sched.schedule(now=2.0)
    assert [r.rid for r in plan.prefill] == ["small"]


def test_aging_trades_cached_tokens_against_seconds_waited():
    """alpha is an exchange rate: at 2000 tokens/s a request that has waited just over a
    second outranks a freshly arrived one carrying a 2000-token cache hit."""
    alloc = BlockAllocator(512, BS)
    shared = list(range(2048))
    warm(alloc, shared)
    sched = Scheduler(alloc, policy="lpm_aging", max_running=8, aging_tokens_per_s=2000.0, max_wait_s=30.0)
    cold = make_request("cold", [99_999] * 80, arrival=0.0)

    fresh_at_one = make_request("hot1", shared + [7] * 16, arrival=1.0)
    assert sched._score(cold, now=1.0) < sched._score(fresh_at_one, now=1.0)

    fresh_at_two = make_request("hot2", shared + [7] * 16, arrival=2.0)
    assert sched._score(cold, now=2.0) > sched._score(fresh_at_two, now=2.0)


def test_a_constant_arrival_gap_is_never_closed_by_aging_alone():
    """Both requests age at the same rate, so a fixed head start in arrival time gives a
    constant advantage. Aging only wins against arrivals that keep being new, which is
    why the hard deadline is not redundant."""
    alloc = BlockAllocator(512, BS)
    shared = list(range(2048))
    warm(alloc, shared)
    sched = Scheduler(alloc, policy="lpm_aging", max_running=8, aging_tokens_per_s=2000.0, max_wait_s=30.0)
    cold = make_request("cold", [99_999] * 80, arrival=0.0)
    hot = make_request("hot", shared + [7] * 16, arrival=1.0)
    for now in (2.0, 5.0, 20.0):
        assert sched._score(cold, now=now) < sched._score(hot, now=now)


def test_the_deadline_makes_a_starving_request_outrank_everything():
    alloc = BlockAllocator(512, BS)
    shared = list(range(2048))
    warm(alloc, shared)
    sched = Scheduler(alloc, policy="lpm_aging", max_running=8, aging_tokens_per_s=2000.0, max_wait_s=5.0)
    cold = make_request("cold", [99_999] * 80, arrival=0.0)
    hot = make_request("hot", shared + [7] * 16, arrival=1.0)
    assert sched._score(cold, now=5.0) == float("inf")
    assert sched._score(cold, now=5.0) > sched._score(hot, now=5.0)


def test_a_starving_request_scores_infinite_and_blocks_admission():
    alloc = BlockAllocator(6, BS)
    sched = Scheduler(alloc, policy="lpm_aging", max_running=8, max_wait_s=5.0)
    sched.add(make_request("big", list(range(160)), arrival=0.0))
    sched.add(make_request("small", list(range(32)), arrival=1.0))
    assert [r.rid for r in sched.schedule(now=2.0).prefill] == ["small"]
    sched.waiting = [r for r in sched.waiting if r.rid == "big"]
    plan = sched.schedule(now=20.0)
    assert plan.prefill == []


def test_scan_limit_bounds_the_work_per_step():
    alloc = BlockAllocator(8, BS)
    sched = Scheduler(alloc, policy="lpm", max_running=1, scan_limit=128)
    for index in range(200):
        sched.add(make_request(f"r{index}", list(range(32)), arrival=float(index)))
    sched.schedule(now=500.0)
    assert sched.stats["scored_requests"] == 128
    assert sched.stats["scan_limit_exceeded"] == 1


def test_preemption_releases_blocks_and_requeues_by_arrival():
    alloc = BlockAllocator(8, BS)
    sched = Scheduler(alloc, policy="fcfs", max_running=8)
    first = make_request("first", list(range(64)), arrival=0.0, max_tokens=64)
    second = make_request("second", list(range(100, 164)), arrival=1.0, max_tokens=64)
    sched.add(first)
    sched.add(second)
    plan = sched.schedule(now=2.0)
    assert len(plan.prefill) == 2
    assert alloc.num_available() == 0

    first.out.extend([1])
    second.out.extend([1])
    plan = sched.schedule(now=3.0)
    assert plan.preempted
    assert sched.stats["preemptions"] == 1
    victim = plan.preempted[0]
    assert victim.rid == "second"
    assert victim.blocks == []
    assert victim.state is RequestState.WAITING
    assert all(count >= 0 for count in alloc.ref)


def test_preemption_takes_the_newest_admitted_first():
    """Four 2-block requests exactly fill an 8-block pool, then each needs a third block.
    Victims come off the back of the running list, newest first, and the survivors are
    always a prefix of the admission order."""
    alloc = BlockAllocator(8, BS)
    sched = Scheduler(alloc, policy="fcfs", max_running=8)
    for index in range(4):
        sched.add(
            make_request(
                f"r{index}",
                list(range(index * 100, index * 100 + 32)),
                arrival=float(index),
                max_tokens=64,
            )
        )
    sched.schedule(now=5.0)
    for r in sched.running:
        r.out.append(1)
    plan = sched.schedule(now=6.0)
    assert [r.rid for r in plan.preempted] == ["r3", "r2"]
    assert [r.rid for r in sched.running] == ["r0", "r1"]


def test_end_to_end_run_returns_every_block():
    rng = random.Random(4)
    h = build(policy="fcfs", num_blocks=48, max_running=4)
    for index in range(200):
        prompt = [rng.randrange(50)] * 16 + [rng.randrange(10_000) for _ in range(rng.randrange(1, 80))]
        h.submit(f"r{index}", prompt, max_tokens=rng.randrange(1, 12))
    h.drain()
    assert h.engine.metrics.counters["finished_requests"] == 200
    assert h.engine.leaked_blocks() == 0
    h.engine.alloc.check_invariants()


def test_preemption_heavy_run_stays_consistent():
    h = build(policy="fcfs", num_blocks=20, max_running=8, max_model_len=256, debug_invariants=True)
    for index in range(40):
        h.submit(f"r{index}", list(range(index, index + 64)), max_tokens=40)
    h.drain()
    assert h.engine.metrics.counters["finished_requests"] == 40
    assert h.engine.sched.stats["preemptions"] > 0
    assert h.engine.leaked_blocks() == 0
    for rid in (f"r{i}" for i in range(40)):
        assert len(h.tokens(rid)) == 40


STEP_DT = 0.02


def hot_stream(policy: str, steps: int = 600, cold_at: int = 80, seed: int = 11, **cfg):
    """A saturated stream of requests that all share one hot prefix, plus a single cold
    request that arrives once the queue is already deep. This is the arrival pattern the
    SGLang starvation report describes."""
    rng = random.Random(seed)
    hot_prefix = list(range(2000, 2000 + 128))
    h = build(policy=policy, num_blocks=28, max_running=4, max_model_len=512, **cfg)
    for step in range(steps):
        if rng.random() < 0.8:
            suffix = [rng.randrange(9_000)] * rng.randrange(8, 32)
            h.submit(f"hot{step}", hot_prefix + suffix, max_tokens=6)
        if step == cold_at:
            h.submit("cold", [7] * 208, max_tokens=6)
        h.engine.step()
        h.clock.advance(STEP_DT)
    return h


def cold_delay(h) -> float | None:
    for record in h.engine.metrics.finished:
        if record["rid"] == "cold":
            return record["queue_delay"]
    for r in h.engine.sched.waiting + h.engine.sched.running:
        if r.rid == "cold":
            return r.queue_delay()
    return None


def test_plain_lpm_starves_a_cold_request_under_a_hot_stream():
    h = hot_stream("lpm")
    delay = cold_delay(h)
    assert delay is None or delay > 5.0
    assert h.engine.metrics.counters["finished_requests"] > 50


def test_the_starvation_bound_holds_on_the_same_stream():
    """The deadline bounds the scheduling delay. It cannot bound the memory delay: once a
    starving request outranks everything, admission stops until the running batch drains
    enough blocks for it, which is why the slack below is a batch drain, not a step."""
    h = hot_stream("lpm_aging", max_wait_s=2.0, aging_tokens_per_s=2000.0)
    delay = cold_delay(h)
    assert delay is not None, "the cold request never got in at all"
    assert delay <= 2.0 + 0.4
    assert h.engine.metrics.counters["finished_requests"] > 50


def test_a_single_prefix_group_leaves_nothing_for_a_policy_to_do():
    """One shared prefix stays resident under any order, so reordering cannot help. If a
    prefix-aware policy ever looks better here, the measurement is wrong, not the policy."""
    rates = {p: hot_stream(p, max_wait_s=2.0).engine.alloc.hit_rate() for p in ("fcfs", "lpm", "lpm_aging")}
    assert max(rates.values()) - min(rates.values()) < 0.02


def grouped_stream(policy: str, groups: int = 6, steps: int = 600, seed: int = 5, **cfg):
    """Several prefix groups whose combined working set does not fit, arriving interleaved.
    This is the regime the whole project is about: FCFS evicts each group's prefix just
    before the next request that needs it arrives."""
    rng = random.Random(seed)
    prefixes = [list(range(g * 10_000, g * 10_000 + 128)) for g in range(groups)]
    h = build(policy=policy, num_blocks=28, max_running=4, max_model_len=512, **cfg)
    for step in range(steps):
        if rng.random() < 0.9:
            group = rng.randrange(groups)
            suffix = [rng.randrange(9_000)] * rng.randrange(8, 32)
            h.submit(f"r{step}", prefixes[group] + suffix, max_tokens=6)
        h.engine.step()
        h.clock.advance(STEP_DT)
    return h


def test_many_groups_under_pressure_is_where_prefix_aware_wins():
    """The whole hypothesis in miniature, on a fake runner. The ordering is the claim;
    the size of the gap is what the real experiment measures."""
    fcfs = grouped_stream("fcfs").engine.alloc.hit_rate()
    lpm = grouped_stream("lpm").engine.alloc.hit_rate()
    aged = grouped_stream("lpm_aging", max_wait_s=2.0, aging_tokens_per_s=2000.0).engine.alloc.hit_rate()
    assert lpm > fcfs + 0.05
    assert fcfs < aged < lpm
    recovered = (aged - fcfs) / (lpm - fcfs)
    assert recovered > 0.2, f"the starvation bound gave back all of the gain ({recovered:.2f})"


def test_grouping_shows_up_as_fewer_evictions():
    fcfs = grouped_stream("fcfs")
    lpm = grouped_stream("lpm")
    assert lpm.engine.alloc.stats["evictions"] < fcfs.engine.alloc.stats["evictions"]
