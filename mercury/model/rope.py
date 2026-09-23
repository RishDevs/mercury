from __future__ import annotations

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """q: [T, n_q, d], k: [T, n_kv, d], cos/sin: [T, d] already in q's dtype."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class RotaryEmbedding:
    """Precomputed cos/sin table indexed by absolute position.

    Positions are absolute, never relative to the start of the chunk being prefilled:
    a suffix resumed on top of a cached prefix starts at num_cached, not at zero.
    """

    def __init__(
        self,
        head_dim: int,
        max_positions: int,
        base: float,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float16,
    ):
        self.head_dim = head_dim
        self.max_positions = max_positions
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim
            )
        )
        positions = torch.arange(max_positions, dtype=torch.float32, device=device)
        freqs = torch.outer(positions, inv_freq)
        table = torch.cat((freqs, freqs), dim=-1)
        self.cos = table.cos().to(dtype)
        self.sin = table.sin().to(dtype)

    def lookup(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos.index_select(0, positions), self.sin.index_select(0, positions)
