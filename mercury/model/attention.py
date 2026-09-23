from __future__ import annotations

import torch
import torch.nn.functional as F

_GQA_KWARG = True


def _sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
    scale: float,
    is_causal: bool = False,
) -> torch.Tensor:
    """q: [B, n_q, T, d]; k/v: [B, n_kv, S, d] with n_q a multiple of n_kv."""
    global _GQA_KWARG
    if _GQA_KWARG:
        try:
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, scale=scale, is_causal=is_causal, enable_gqa=True
            )
        except TypeError:
            _GQA_KWARG = False
    groups = q.shape[1] // k.shape[1]
    if groups > 1:
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, scale=scale, is_causal=is_causal
    )


def write_kv(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    slots: torch.Tensor,
) -> None:
    """k, v: [T, n_kv, d] for T new tokens; slots: [T] int64 = block * block_size + offset."""
    num_blocks, block_size = k_cache.shape[:2]
    k_cache.view(num_blocks * block_size, *k_cache.shape[2:]).index_copy_(0, slots, k)
    v_cache.view(num_blocks * block_size, *v_cache.shape[2:]).index_copy_(0, slots, v)


def paged_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """q: [B, n_q, d], one new token per sequence whose K/V is already written.

    block_tables: [B, max_blocks] int64, right-padded with any valid block id.
    seq_lens: [B] int64, the number of KV positions each sequence now has.
    """
    batch, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    k = k_cache[block_tables].reshape(batch, -1, num_kv_heads, head_dim)
    v = v_cache[block_tables].reshape(batch, -1, num_kv_heads, head_dim)
    span = k.shape[1]
    valid = torch.arange(span, device=q.device)[None, :] < seq_lens[:, None]
    out = _sdpa(
        q.unsqueeze(2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        valid[:, None, None, :],
        scale,
    )
    return out.squeeze(2)


def paged_prefill_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
    num_cached: int,
    scale: float,
) -> torch.Tensor:
    """q: [Tq, n_q, d] for positions num_cached..seq_len-1, attending over 0..seq_len-1."""
    num_kv_heads, head_dim = k_cache.shape[2], k_cache.shape[3]
    k = k_cache[block_table].reshape(-1, num_kv_heads, head_dim)[:seq_len]
    v = v_cache[block_table].reshape(-1, num_kv_heads, head_dim)[:seq_len]
    q4 = q.transpose(0, 1).unsqueeze(0)
    k4 = k.transpose(0, 1).unsqueeze(0)
    v4 = v.transpose(0, 1).unsqueeze(0)
    if num_cached == 0:
        out = _sdpa(q4, k4, v4, None, scale, is_causal=True)
    else:
        # Queries start at num_cached while keys start at 0, so is_causal is wrong here.
        query_pos = torch.arange(q.shape[0], device=q.device) + num_cached
        key_pos = torch.arange(seq_len, device=q.device)
        mask = key_pos[None, :] <= query_pos[:, None]
        out = _sdpa(q4, k4, v4, mask[None, None], scale)
    return out.squeeze(0).transpose(0, 1)


class PrefillContext:
    """One request's prefill: write the suffix's K/V, then attend over the whole sequence."""

    __slots__ = ("kv_caches", "block_table", "slots", "seq_len", "num_cached")

    def __init__(
        self,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]],
        block_table: torch.Tensor,
        slots: torch.Tensor,
        seq_len: int,
        num_cached: int,
    ):
        self.kv_caches = kv_caches
        self.block_table = block_table
        self.slots = slots
        self.seq_len = seq_len
        self.num_cached = num_cached

    def attend(
        self, layer: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float
    ) -> torch.Tensor:
        k_cache, v_cache = self.kv_caches[layer]
        write_kv(k_cache, v_cache, k, v, self.slots)
        return paged_prefill_attention(
            q, k_cache, v_cache, self.block_table, self.seq_len, self.num_cached, scale
        )


class DecodeContext:
    """A whole decode batch: one new token per running sequence."""

    __slots__ = ("kv_caches", "block_tables", "slots", "seq_lens")

    def __init__(
        self,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]],
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        seq_lens: torch.Tensor,
    ):
        self.kv_caches = kv_caches
        self.block_tables = block_tables
        self.slots = slots
        self.seq_lens = seq_lens

    def attend(
        self, layer: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float
    ) -> torch.Tensor:
        k_cache, v_cache = self.kv_caches[layer]
        write_kv(k_cache, v_cache, k, v, self.slots)
        return paged_decode_attention(
            q, k_cache, v_cache, self.block_tables, self.seq_lens, scale
        )
