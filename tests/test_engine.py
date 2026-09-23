from __future__ import annotations

import random

import pytest

from .conftest import build

BS = 16


def test_every_request_emits_exactly_max_tokens_then_done():
    h = build(num_blocks=64, max_running=4)
    for index in range(12):
        h.submit(f"r{index}", list(range(index * 50, index * 50 + 40)), max_tokens=6)
    h.drain()
    for index in range(12):
        rid = f"r{index}"
        assert len(h.tokens(rid)) == 6
        assert h.finish_reason(rid) == "length"


def test_tokens_arrive_in_index_order():
    h = build(num_blocks=64)
    h.submit("a", list(range(40)), max_tokens=10)
    h.drain()
    indices = [e.index for e in h.events["a"] if e.kind == "token"]
    assert indices == list(range(10))


def test_hit_rate_matches_the_hand_computed_value():
    """Four requests share a 64-token prefix and each adds a private 16-token suffix.
    The first is a cold miss, the other three reuse four blocks each:
    192 cached tokens out of 320 queried."""
    h = build(num_blocks=64)
    shared = list(range(64))
    for index in range(4):
        h.submit(f"r{index}", shared + [1000 + index] * 16, max_tokens=1)
        h.drain()
    stats = h.engine.alloc.snapshot()
    assert stats["query_tokens"] == 320
    assert stats["hit_tokens"] == 192
    assert stats["hit_rate"] == pytest.approx(0.6)


def test_no_sharing_means_no_hits():
    h = build(num_blocks=64)
    for index in range(4):
        h.submit(f"r{index}", [index * 7919 + j for j in range(64)], max_tokens=1)
        h.drain()
    assert h.engine.alloc.snapshot()["hit_tokens"] == 0


def test_a_partial_block_is_never_shared():
    h = build(num_blocks=64)
    h.submit("a", list(range(24)), max_tokens=1)
    h.drain()
    h.submit("b", list(range(24)), max_tokens=1)
    h.drain()
    assert h.engine.alloc.snapshot()["hit_tokens"] == 16


def test_generated_tokens_extend_the_cache():
    h = build(num_blocks=64)
    h.submit("a", list(range(64)), max_tokens=40)
    h.drain()
    assert h.engine.alloc.num_cached_blocks() > 4


def test_aborting_a_running_request_returns_its_blocks():
    h = build(num_blocks=32, max_running=2)
    for index in range(6):
        h.submit(f"r{index}", list(range(index * 100, index * 100 + 48)), max_tokens=50)
    h.run(3)
    h.engine.abort("r0")
    h.engine.abort("r1")
    h.drain()
    assert h.engine.metrics.counters["aborted_requests"] == 2
    assert h.finish_reason("r0") == "abort"
    assert h.engine.leaked_blocks() == 0


def test_aborting_a_waiting_request_never_runs_it():
    h = build(num_blocks=32, max_running=1)
    h.submit("head", list(range(64)), max_tokens=50)
    h.submit("tail", list(range(200, 264)), max_tokens=50)
    h.run(2)
    h.engine.abort("tail")
    h.drain()
    assert h.tokens("tail") == []
    assert h.finish_reason("tail") == "abort"


def test_a_tenth_of_requests_cancelled_leaks_nothing():
    """The benchmark's cancel mode in miniature. Leaked blocks from cancelled streams are
    the classic slow memory leak in a serving system, so the assertion that matters is the
    pool being whole at the end."""
    rng = random.Random(3)
    h = build(num_blocks=64, max_running=6)
    cancelled = 0
    for index in range(300):
        rid = f"r{index}"
        h.submit(
            rid,
            [rng.randrange(40)] * 16 + [rng.randrange(9000) for _ in range(rng.randrange(1, 60))],
            max_tokens=rng.randrange(4, 24),
        )
        h.engine.step()
        h.clock.advance(0.01)
        if rng.random() < 0.1:
            live = [r.rid for r in h.engine.sched.running + h.engine.sched.waiting]
            if live:
                h.engine.abort(rng.choice(live))
                cancelled += 1
    h.drain()
    assert cancelled > 15
    assert h.engine.metrics.counters["aborted_requests"] >= cancelled - 2
    assert h.engine.leaked_blocks() == 0
    h.engine.alloc.check_invariants()


def test_a_request_bigger_than_the_pool_is_rejected_not_queued():
    h = build(num_blocks=4, max_model_len=1024)
    h.submit("huge", list(range(200)), max_tokens=10)
    h.drain()
    errors = [e for e in h.events["huge"] if e.kind == "error"]
    assert len(errors) == 1
    assert "ceiling" in errors[0].message
    assert h.engine.sched.waiting == []


def test_an_empty_prompt_is_rejected():
    h = build()
    h.submit("empty", [], max_tokens=4)
    assert [e.kind for e in h.events["empty"]] == ["error"]


def test_a_freed_slot_is_reused_on_the_very_next_step():
    h = build(num_blocks=64, max_running=2)
    h.submit("a", list(range(32)), max_tokens=2)
    h.submit("b", list(range(100, 132)), max_tokens=20)
    h.submit("c", list(range(200, 232)), max_tokens=4)
    h.engine.step()
    assert len(h.engine.sched.running) == 2
    assert h.engine.sched.waiting[0].rid == "c"
    h.engine.step()
    assert h.finish_reason("a") == "length"
    h.engine.step()
    assert any(r.rid == "c" for r in h.engine.sched.running)


def test_the_step_log_separates_scheduling_from_execution():
    h = build(num_blocks=64)
    for index in range(5):
        h.submit(f"r{index}", list(range(index * 40, index * 40 + 32)), max_tokens=5)
    h.drain()
    summary = h.engine.metrics.summary()
    assert summary["steps"] > 0
    assert summary["sched_seconds"] > 0
    assert 0.0 <= summary["sched_share"] <= 1.0
    assert summary["decode_tokens"] > 0


def test_snapshot_reports_the_policy_and_capacity():
    h = build(policy="lpm_aging", num_blocks=32)
    snap = h.engine.snapshot()
    assert snap["config"]["policy"] == "lpm_aging"
    assert snap["config"]["kv_capacity_tokens"] == 32 * BS
    assert snap["allocator"]["num_blocks"] == 32


def test_request_records_carry_the_queue_delay():
    h = build(num_blocks=64, max_running=1)
    h.submit("a", list(range(32)), max_tokens=3)
    h.clock.advance(0.5)
    h.submit("b", list(range(100, 132)), max_tokens=3)
    h.drain()
    records = {r["rid"]: r for r in h.engine.drain_request_records()}
    assert records["a"]["queue_delay"] >= 0.0
    assert records["b"]["queue_delay"] >= 0.0
    assert records["b"]["output_tokens"] == 3
