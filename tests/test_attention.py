from __future__ import annotations

import torch

from mercury.model.attention import (
    paged_decode_attention,
    paged_prefill_attention,
    write_kv,
)
from mercury.model.rope import apply_rope, rotate_half
from mercury.runner import slot_mapping

BS = 16
HEADS = 4
KV_HEADS = 2
DIM = 8
DTYPE = torch.float32


def empty_cache(num_blocks=32):
    shape = (num_blocks, BS, KV_HEADS, DIM)
    return torch.zeros(shape, dtype=DTYPE), torch.zeros(shape, dtype=DTYPE)


def reference_attention(q, k, v, scale, query_offset):
    """q: [Tq, n_q, d]; k, v: [S, n_kv, d]. Explicit, slow, obviously correct."""
    groups = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q, k) * scale
    key_pos = torch.arange(k.shape[0])
    query_pos = torch.arange(q.shape[0]) + query_offset
    scores = scores.masked_fill(key_pos[None, None, :] > query_pos[None, :, None], float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return torch.einsum("hqk,khd->qhd", weights, v)


def test_slot_mapping_at_block_boundaries():
    blocks = [5, 9, 2]
    assert slot_mapping(blocks, 15, 16, BS).tolist() == [5 * BS + 15]
    assert slot_mapping(blocks, 16, 17, BS).tolist() == [9 * BS + 0]
    assert slot_mapping(blocks, 17, 18, BS).tolist() == [9 * BS + 1]
    assert slot_mapping(blocks, 31, 32, BS).tolist() == [9 * BS + 15]
    assert slot_mapping(blocks, 32, 33, BS).tolist() == [2 * BS + 0]


def test_slot_mapping_covers_a_whole_prompt():
    blocks = [7, 1]
    slots = slot_mapping(blocks, 0, 20, BS)
    assert slots[:16].tolist() == [7 * BS + i for i in range(16)]
    assert slots[16:].tolist() == [1 * BS + i for i in range(4)]


def test_write_kv_lands_in_the_right_physical_slots():
    k_cache, v_cache = empty_cache(8)
    k = torch.arange(3 * KV_HEADS * DIM, dtype=DTYPE).reshape(3, KV_HEADS, DIM)
    v = k * -1
    slots = torch.tensor([4 * BS + 0, 4 * BS + 15, 6 * BS + 3])
    write_kv(k_cache, v_cache, k, v, slots)
    assert torch.equal(k_cache[4, 0], k[0])
    assert torch.equal(k_cache[4, 15], k[1])
    assert torch.equal(k_cache[6, 3], k[2])
    assert torch.equal(v_cache[6, 3], v[2])
    assert torch.count_nonzero(k_cache[0]) == 0


def test_paged_prefill_matches_the_explicit_reference():
    torch.manual_seed(0)
    seq_len = 40
    k_cache, v_cache = empty_cache()
    blocks = [11, 3, 7]
    q = torch.randn(seq_len, HEADS, DIM)
    k = torch.randn(seq_len, KV_HEADS, DIM)
    v = torch.randn(seq_len, KV_HEADS, DIM)
    slots = torch.from_numpy(slot_mapping(blocks, 0, seq_len, BS))
    write_kv(k_cache, v_cache, k, v, slots)
    scale = DIM**-0.5
    out = paged_prefill_attention(
        q, k_cache, v_cache, torch.tensor(blocks), seq_len, 0, scale
    )
    assert torch.allclose(out, reference_attention(q, k, v, scale, 0), atol=1e-5)


def test_prefill_on_top_of_a_cached_prefix_matches_the_full_prefill():
    """The whole point of the explicit mask: queries start at num_cached while keys
    start at zero, so torch's is_causal would mask the wrong triangle."""
    torch.manual_seed(1)
    seq_len, cached = 48, 32
    k_cache, v_cache = empty_cache()
    blocks = [2, 5, 9]
    q_full = torch.randn(seq_len, HEADS, DIM)
    k = torch.randn(seq_len, KV_HEADS, DIM)
    v = torch.randn(seq_len, KV_HEADS, DIM)
    slots = torch.from_numpy(slot_mapping(blocks, 0, seq_len, BS))
    write_kv(k_cache, v_cache, k, v, slots)
    scale = DIM**-0.5

    full = reference_attention(q_full, k, v, scale, 0)
    suffix = paged_prefill_attention(
        q_full[cached:], k_cache, v_cache, torch.tensor(blocks), seq_len, cached, scale
    )
    assert torch.allclose(suffix, full[cached:], atol=1e-5)


def test_paged_decode_matches_the_reference_for_every_sequence():
    torch.manual_seed(2)
    k_cache, v_cache = empty_cache(64)
    tables = [[12, 1, 30], [7, 20, 4], [15, 9, 22]]
    lengths = [33, 17, 48]
    scale = DIM**-0.5
    keys, values = [], []
    for blocks, length in zip(tables, lengths):
        k = torch.randn(length, KV_HEADS, DIM)
        v = torch.randn(length, KV_HEADS, DIM)
        write_kv(
            k_cache, v_cache, k, v, torch.from_numpy(slot_mapping(blocks, 0, length, BS))
        )
        keys.append(k)
        values.append(v)

    q = torch.randn(len(tables), HEADS, DIM)
    out = paged_decode_attention(
        q,
        k_cache,
        v_cache,
        torch.tensor(tables),
        torch.tensor(lengths),
        scale,
    )
    for index, (k, v, length) in enumerate(zip(keys, values, lengths)):
        expected = reference_attention(q[index : index + 1], k, v, scale, length - 1)
        assert torch.allclose(out[index], expected[0], atol=1e-5)


def test_physical_placement_does_not_change_the_answer():
    torch.manual_seed(3)
    seq_len = 40
    q = torch.randn(seq_len, HEADS, DIM)
    k = torch.randn(seq_len, KV_HEADS, DIM)
    v = torch.randn(seq_len, KV_HEADS, DIM)
    scale = DIM**-0.5
    results = []
    for blocks in ([0, 1, 2], [29, 4, 17]):
        k_cache, v_cache = empty_cache()
        write_kv(
            k_cache, v_cache, k, v, torch.from_numpy(slot_mapping(blocks, 0, seq_len, BS))
        )
        results.append(
            paged_prefill_attention(
                q, k_cache, v_cache, torch.tensor(blocks), seq_len, 0, scale
            )
        )
    assert torch.allclose(results[0], results[1], atol=1e-6)


def test_padding_a_block_table_does_not_leak_into_the_output():
    torch.manual_seed(4)
    k_cache, v_cache = empty_cache(64)
    blocks = [3, 8]
    length = 20
    k = torch.randn(length, KV_HEADS, DIM)
    v = torch.randn(length, KV_HEADS, DIM)
    write_kv(k_cache, v_cache, k, v, torch.from_numpy(slot_mapping(blocks, 0, length, BS)))
    k_cache[50] = 1e4
    v_cache[50] = 1e4
    q = torch.randn(1, HEADS, DIM)
    scale = DIM**-0.5
    padded = paged_decode_attention(
        q, k_cache, v_cache, torch.tensor([blocks + [50]]), torch.tensor([length]), scale
    )
    exact = paged_decode_attention(
        q, k_cache, v_cache, torch.tensor([blocks]), torch.tensor([length]), scale
    )
    assert torch.allclose(padded, exact, atol=1e-6)


def test_rotate_half_is_the_neox_layout():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert torch.equal(rotate_half(x), torch.tensor([[-3.0, -4.0, 1.0, 2.0]]))


def test_rope_preserves_the_dot_product_of_equally_shifted_positions():
    torch.manual_seed(5)
    dim = 8
    inv = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
    def table(pos):
        freqs = torch.outer(torch.tensor(pos, dtype=torch.float32), inv)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    q = torch.randn(1, 1, dim)
    k = torch.randn(1, 1, dim)
    cos_a, sin_a = table([3.0])
    cos_b, sin_b = table([9.0])
    qa, ka = apply_rope(q, k, cos_a, sin_a)
    qb, kb = apply_rope(q, k, cos_b, sin_b)
    assert torch.allclose((qa * ka).sum(), (qb * kb).sum(), atol=1e-5)
