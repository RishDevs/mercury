from __future__ import annotations

import random

import pytest

from mercury.allocator import BlockAllocator
from mercury.hashing import block_hashes

BS = 16


def make(num_blocks=16):
    return BlockAllocator(num_blocks=num_blocks, block_size=BS)


def allocate_and_commit(alloc, tokens):
    hashes = block_hashes(tokens, BS)
    got = alloc.allocate_prompt(hashes, len(tokens))
    assert got is not None
    blocks, cached = got
    for index in range(len(tokens) // BS):
        alloc.commit(blocks[index], hashes[index])
    return blocks, cached


def test_starts_empty_and_returns_everything():
    alloc = make(8)
    assert alloc.num_available() == 8
    blocks, cached = allocate_and_commit(alloc, list(range(64)))
    assert cached == 0
    assert len(blocks) == 4
    assert alloc.num_available() == 4
    alloc.release(blocks)
    assert alloc.num_available() == 8
    alloc.check_invariants()


def test_partial_last_block_is_counted_but_not_shareable():
    alloc = make(8)
    tokens = list(range(40))
    blocks, _ = allocate_and_commit(alloc, tokens)
    assert len(blocks) == 3
    assert alloc.hash_of[blocks[2]] is None
    alloc.release(blocks)
    assert len(alloc.evictable) == 2
    assert len(alloc.free) == 6


def test_second_identical_prompt_hits_the_cache():
    alloc = make(16)
    tokens = list(range(80))
    first, _ = allocate_and_commit(alloc, tokens)
    alloc.release(first)
    second, cached = allocate_and_commit(alloc, tokens)
    assert cached == 4 * BS
    assert second[:4] == first[:4]
    assert second[4] != first[4]


def test_one_token_is_always_left_to_recompute():
    alloc = make(16)
    tokens = list(range(64))
    first, _ = allocate_and_commit(alloc, tokens)
    alloc.release(first)
    hashes = block_hashes(tokens, BS)
    assert len(hashes) == 4
    hit, need_new = alloc.plan_prompt(hashes, len(tokens))
    assert len(hit) == 3
    assert need_new == 1


def test_prefix_match_stops_at_the_first_miss():
    alloc = make(32)
    shared = list(range(64))
    left, _ = allocate_and_commit(alloc, shared + [1] * 16)
    alloc.release(left)
    hashes = block_hashes(shared + [2] * 16, BS)
    assert len(alloc.cached_prefix_blocks(hashes)) == 4


def test_release_puts_the_tail_into_the_lru_before_the_head():
    alloc = make(8)
    blocks, _ = allocate_and_commit(alloc, list(range(128)))
    assert len(blocks) == 8
    alloc.release(blocks)
    assert list(alloc.evictable) == list(reversed(blocks))
    # The next eviction takes the private suffix, not the shared prefix head.
    assert alloc._take() == blocks[-1]


def test_allocation_that_does_not_fit_has_no_side_effects():
    alloc = make(4)
    before = alloc.snapshot()
    assert alloc.allocate_prompt([], 128) is None
    after = alloc.snapshot()
    assert after["free_blocks"] == before["free_blocks"]
    assert after["query_tokens"] == before["query_tokens"]
    alloc.check_invariants()


def test_pinning_a_cached_block_counts_against_capacity():
    alloc = make(5)
    tokens = list(range(64))
    first, _ = allocate_and_commit(alloc, tokens)
    alloc.release(first)
    assert alloc.num_available() == 5
    # 4 cached blocks get pinned, leaving one free block; a 5-block prompt cannot fit.
    assert alloc.allocate_prompt(block_hashes(tokens + [9] * 16, BS), 80) is not None
    assert alloc.num_available() == 0


def test_eviction_reclaims_the_least_recently_released():
    alloc = make(4)
    a, _ = allocate_and_commit(alloc, list(range(32)))
    alloc.release(a)
    b, _ = allocate_and_commit(alloc, list(range(100, 132)))
    alloc.release(b)
    assert alloc.num_available() == 4
    fresh, cached = allocate_and_commit(alloc, list(range(200, 264)))
    assert cached == 0
    assert alloc.stats["evictions"] == 4
    assert set(fresh) == set(a) | set(b)


def test_duplicate_content_keeps_the_first_block_canonical():
    alloc = make(16)
    tokens = list(range(32))
    hashes = block_hashes(tokens, BS)
    left, _ = alloc.allocate_prompt(hashes, len(tokens))
    right, _ = alloc.allocate_prompt(hashes, len(tokens))
    alloc.commit(left[0], hashes[0])
    alloc.commit(right[0], hashes[0])
    assert alloc.by_hash[hashes[0]] == left[0]
    assert alloc.hash_of[right[0]] is None
    alloc.release(right)
    assert right[0] in alloc.free


def test_append_block_fails_cleanly_when_full():
    alloc = make(2)
    alloc.allocate_prompt([], 32)
    assert alloc.append_block() is None


def test_hit_rate_is_token_level():
    alloc = make(32)
    tokens = list(range(80))
    first, _ = allocate_and_commit(alloc, tokens)
    alloc.release(first)
    allocate_and_commit(alloc, tokens)
    assert alloc.stats["query_tokens"] == 160
    assert alloc.stats["hit_tokens"] == 64
    assert alloc.hit_rate() == pytest.approx(0.4)


def test_randomized_workload_preserves_every_invariant():
    rng = random.Random(1234)
    alloc = make(64)
    prefixes = [[rng.randrange(5000) for _ in range(rng.choice([32, 64, 96]))] for _ in range(6)]
    live: list[list[int]] = []

    for _ in range(4000):
        roll = rng.random()
        if roll < 0.5 or not live:
            tokens = list(rng.choice(prefixes)) + [rng.randrange(5000) for _ in range(rng.randrange(1, 40))]
            hashes = block_hashes(tokens, BS)
            got = alloc.allocate_prompt(hashes, len(tokens))
            if got is not None:
                blocks, _ = got
                for index in range(min(len(tokens) // BS, len(blocks))):
                    alloc.commit(blocks[index], hashes[index])
                live.append(blocks)
        elif roll < 0.8:
            alloc.release(live.pop(rng.randrange(len(live))))
        else:
            block = alloc.append_block()
            if block is not None:
                live.append([block])
        alloc.check_invariants()
        assert all(count >= 0 for count in alloc.ref)

    for blocks in live:
        alloc.release(blocks)
    alloc.check_invariants()
    assert alloc.num_available() == 64


def test_every_block_is_reachable_after_a_churn():
    rng = random.Random(99)
    alloc = make(24)
    held = []
    for step in range(500):
        tokens = [step % 3] * 16 + [rng.randrange(100) for _ in range(rng.randrange(1, 50))]
        got = alloc.allocate_prompt(block_hashes(tokens, BS), len(tokens))
        if got is None:
            if held:
                alloc.release(held.pop(0))
            continue
        held.append(got[0])
    for blocks in held:
        alloc.release(blocks)
    assert alloc.num_available() == alloc.num_blocks
    assert sum(alloc.ref) == 0
