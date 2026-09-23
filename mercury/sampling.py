from __future__ import annotations

import torch


def _top_p_mask(probs: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
    ordered, index = probs.sort(dim=-1, descending=True)
    cumulative = ordered.cumsum(dim=-1)
    # Keep everything up to and including the token that crosses p.
    drop = (cumulative - ordered) > top_p.unsqueeze(1)
    ordered = ordered.masked_fill(drop, 0.0)
    out = torch.zeros_like(probs)
    out.scatter_(1, index, ordered)
    return out


def sample_tokens(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ps: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """logits: [B, V]. Rows with temperature <= 0 are greedy regardless of top_p."""
    greedy = temperatures <= 0.0
    argmax = logits.argmax(dim=-1)
    if bool(greedy.all()):
        return argmax
    scaled = logits.float() / temperatures.clamp_min(1e-5).unsqueeze(1)
    probs = torch.softmax(scaled, dim=-1)
    if bool((top_ps < 1.0).any()):
        probs = _top_p_mask(probs, top_ps)
    total = probs.sum(dim=-1, keepdim=True)
    probs = torch.where(total > 0, probs / total, torch.full_like(probs, 1.0 / probs.shape[-1]))
    drawn = torch.multinomial(probs, 1, generator=generator).squeeze(1)
    return torch.where(greedy, argmax, drawn)
