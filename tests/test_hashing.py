from __future__ import annotations

import random

from mercury.hashing import (
    block_hash,
    block_hashes,
    cryptographic_hashes_enabled,
    extend_hashes,
    use_cryptographic_hashes,
)


def test_only_full_blocks_are_hashed():
    assert block_hashes(list(range(15)), 16) == []
    assert len(block_hashes(list(range(16)), 16)) == 1
    assert len(block_hashes(list(range(31)), 16)) == 1
    assert len(block_hashes(list(range(32)), 16)) == 2


def test_chain_depends_on_every_earlier_token():
    a = list(range(64))
    b = list(range(64))
    b[0] = 999
    ha, hb = block_hashes(a, 16), block_hashes(b, 16)
    assert all(x != y for x, y in zip(ha, hb))


def test_shared_prefix_shares_exactly_the_shared_blocks():
    shared = list(range(48))
    left = shared + [1] * 16
    right = shared + [2] * 16
    hl, hr = block_hashes(left, 16), block_hashes(right, 16)
    assert hl[:3] == hr[:3]
    assert hl[3] != hr[3]


def test_a_later_difference_does_not_change_earlier_blocks():
    base = list(range(64))
    changed = list(base)
    changed[63] = -1
    assert block_hashes(base, 16)[:3] == block_hashes(changed, 16)[:3]


def test_extend_matches_a_full_recompute():
    rng = random.Random(7)
    tokens = [rng.randrange(1000) for _ in range(200)]
    incremental: list[int] = []
    for end in range(0, 201, 8):
        extend_hashes(incremental, tokens[:end], 16)
    assert incremental == block_hashes(tokens, 16)


def test_extend_reports_how_many_it_added():
    tokens = list(range(48))
    out: list[int] = []
    assert extend_hashes(out, tokens[:16], 16) == 1
    assert extend_hashes(out, tokens[:16], 16) == 0
    assert extend_hashes(out, tokens, 16) == 2


def test_cryptographic_mode_round_trip():
    tokens = list(range(32))
    plain = block_hashes(tokens, 16)
    use_cryptographic_hashes(True)
    try:
        assert cryptographic_hashes_enabled()
        crypto = block_hashes(tokens, 16)
        assert crypto != plain
        assert all(h >= 0 for h in crypto)
        assert block_hash(crypto[0], tokens[16:32]) == crypto[1]
    finally:
        use_cryptographic_hashes(False)
