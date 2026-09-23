from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from ..kernels import add_rmsnorm_quant, fused_available
from .rope import RotaryEmbedding, apply_rope


class Qwen2RMSNorm(nn.Module):
    """Returns (normalized, residual) so the add and the norm can be one kernel."""

    def __init__(self, hidden_size: int, eps: float, fused: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.fused = fused

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.fused and x.is_cuda and x.dtype is torch.float16 and fused_available():
            out, _, res = add_rmsnorm_quant(x, self.weight, self.eps, residual, quant=False)
            return out, (res if res is not None else x)
        if residual is not None:
            x = x + residual
        variance = x.float().pow(2).mean(-1, keepdim=True)
        normed = (x.float() * torch.rsqrt(variance + self.eps)).to(x.dtype)
        return normed * self.weight, x


class Qwen2MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2Attention(nn.Module):
    """Qwen2 puts a bias on q/k/v and none on the output projection."""

    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.scale = self.head_dim**-0.5
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, q_size, bias=True)
        self.k_proj = nn.Linear(cfg.hidden_size, kv_size, bias=True)
        self.v_proj = nn.Linear(cfg.hidden_size, kv_size, bias=True)
        self.o_proj = nn.Linear(q_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, ctx) -> torch.Tensor:
        tokens = x.shape[0]
        q = self.q_proj(x).view(tokens, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(tokens, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(tokens, self.num_kv_heads, self.head_dim)
        q, k = apply_rope(q, k, cos, sin)
        attended = ctx.attend(self.layer_idx, q, k, v, self.scale)
        return self.o_proj(attended.reshape(tokens, self.num_heads * self.head_dim))


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int, fused_norm: bool):
        super().__init__()
        self.self_attn = Qwen2Attention(cfg, layer_idx)
        self.mlp = Qwen2MLP(cfg.hidden_size, cfg.intermediate_size)
        self.input_layernorm = Qwen2RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, fused_norm)
        self.post_attention_layernorm = Qwen2RMSNorm(
            cfg.hidden_size, cfg.rms_norm_eps, fused_norm
        )

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        ctx,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm(x, residual)
        x = self.self_attn(x, cos, sin, ctx)
        x, residual = self.post_attention_layernorm(x, residual)
        x = self.mlp(x)
        return x, residual


class Qwen2Model(nn.Module):
    def __init__(self, cfg: ModelConfig, fused_norm: bool):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            Qwen2DecoderLayer(cfg, i, fused_norm) for i in range(cfg.num_hidden_layers)
        )
        self.norm = Qwen2RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, fused_norm)
        self.rotary: RotaryEmbedding | None = None

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, ctx) -> torch.Tensor:
        cos, sin = self.rotary.lookup(positions)
        x = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            x, residual = layer(x, residual, cos, sin, ctx)
        x, _ = self.norm(x, residual)
        return x


class Qwen2ForCausalLM(nn.Module):
    """Flat token layout everywhere: activations are [num_tokens, hidden], never [B, T, H].

    The batch structure lives in the attention context, which is the only thing that knows
    how the tokens in a step map to sequences and to physical KV blocks.
    """

    def __init__(self, cfg: ModelConfig, fused_norm: bool = False):
        super().__init__()
        self.config = cfg
        self.model = Qwen2Model(cfg, fused_norm)
        self.lm_head = (
            None if cfg.tie_word_embeddings else nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        )

    def init_rotary(self, device: torch.device | str, dtype: torch.dtype, max_positions: int):
        self.model.rotary = RotaryEmbedding(
            self.config.head_dim,
            max_positions,
            self.config.rope_theta,
            device=device,
            dtype=dtype,
        )
        return self.model.rotary

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, ctx) -> torch.Tensor:
        return self.model(input_ids, positions, ctx)

    @property
    def output_embedding(self) -> torch.Tensor:
        return self.model.embed_tokens.weight if self.lm_head is None else self.lm_head.weight

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden, self.output_embedding)
