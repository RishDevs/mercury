from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

POLICIES = ("fcfs", "lpm", "lpm_aging")

_DTYPE_BY_NAME = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool
    eos_token_ids: tuple[int, ...] = ()
    bos_token_id: int | None = None
    architecture: str = "Qwen2ForCausalLM"

    @classmethod
    def from_directory(cls, path: str | Path) -> "ModelConfig":
        raw = json.loads((Path(path) / "config.json").read_text())
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "ModelConfig":
        arch = (raw.get("architectures") or ["Qwen2ForCausalLM"])[0]
        if "Qwen2" not in arch:
            raise ValueError(
                f"mercury implements the Qwen2 architecture only; config says {arch!r}"
            )
        heads = raw["num_attention_heads"]
        hidden = raw["hidden_size"]
        eos = raw.get("eos_token_id", [])
        if isinstance(eos, int):
            eos = [eos]
        return cls(
            hidden_size=hidden,
            intermediate_size=raw["intermediate_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_attention_heads=heads,
            num_key_value_heads=raw.get("num_key_value_heads", heads),
            head_dim=raw.get("head_dim") or hidden // heads,
            vocab_size=raw["vocab_size"],
            rms_norm_eps=raw.get("rms_norm_eps", 1e-6),
            rope_theta=float(raw.get("rope_theta", 10000.0)),
            max_position_embeddings=raw.get("max_position_embeddings", 32768),
            tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
            eos_token_ids=tuple(int(t) for t in eos),
            bos_token_id=raw.get("bos_token_id"),
            architecture=arch,
        )

    def kv_bytes_per_token(self, dtype: torch.dtype = torch.float16) -> int:
        elem = torch.finfo(dtype).bits // 8
        return (
            2
            * self.num_hidden_layers
            * self.num_key_value_heads
            * self.head_dim
            * elem
        )


@dataclass
class EngineConfig:
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    revision: str | None = None
    device: str = "cuda"
    dtype: str = "float16"

    block_size: int = 16
    num_blocks: int = 4096
    gpu_memory_fraction: float = 0.85

    max_running: int = 64
    prefill_budget: int = 8192
    max_model_len: int = 8192

    policy: str = "fcfs"
    aging_tokens_per_s: float = 2000.0
    max_wait_s: float = 5.0
    scan_limit: int = 128

    fused_rmsnorm: bool = True
    cryptographic_hashes: bool = False
    seed: int = 0

    idle_sleep_s: float = 5e-4
    max_step_records: int = 500_000
    debug_invariants: bool = False

    served_model_name: str | None = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}, got {self.policy!r}")
        if self.dtype not in _DTYPE_BY_NAME:
            raise ValueError(f"unsupported dtype {self.dtype!r}")
        if self.block_size <= 0 or self.block_size & (self.block_size - 1):
            raise ValueError("block_size must be a positive power of two")
        if self.served_model_name is None:
            self.served_model_name = self.model

    @property
    def torch_dtype(self) -> torch.dtype:
        return _DTYPE_BY_NAME[self.dtype]

    @property
    def kv_capacity_tokens(self) -> int:
        return self.num_blocks * self.block_size

    def validate_against(self, model_cfg: ModelConfig) -> None:
        if self.max_model_len > model_cfg.max_position_embeddings:
            raise ValueError(
                f"max_model_len {self.max_model_len} exceeds the model's "
                f"{model_cfg.max_position_embeddings} position embeddings"
            )
        if self.kv_capacity_tokens < self.max_model_len:
            raise ValueError(
                f"KV pool holds {self.kv_capacity_tokens} tokens but a single request may "
                f"reach {self.max_model_len}; such a request would preempt itself forever. "
                f"Raise num_blocks above {self.max_model_len // self.block_size} or lower "
                f"max_model_len."
            )
        if self.torch_dtype is torch.bfloat16 and torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            if major < 8:
                raise ValueError(
                    "bfloat16 needs compute capability 8.0+; a T4 is 7.5, use --dtype float16"
                )

    def kv_bytes(self, model_cfg: ModelConfig) -> int:
        return self.kv_capacity_tokens * model_cfg.kv_bytes_per_token(self.torch_dtype)

    def to_dict(self) -> dict:
        return asdict(self)


def blocks_for_capacity(num_tokens: int, block_size: int = 16) -> int:
    return max(1, -(-int(num_tokens) // block_size))


def autosize_blocks(
    model_cfg: ModelConfig,
    engine_cfg: EngineConfig,
    reserve_bytes: int = 0,
    device_index: int = 0,
) -> int:
    """Blocks that fit in the memory left after the weights are resident.

    `reserve_bytes` must cover the weights, which are not loaded yet when this is called;
    sizing against raw free memory would hand the whole card to the KV cache and then fail
    to load the model.
    """
    free, _total = torch.cuda.mem_get_info(device_index)
    budget = int(max(0, free - reserve_bytes) * engine_cfg.gpu_memory_fraction)
    per_block = model_cfg.kv_bytes_per_token(engine_cfg.torch_dtype) * engine_cfg.block_size
    blocks = budget // per_block
    if blocks < 1:
        raise RuntimeError(
            f"only {free / 2**30:.1f} GiB free and the weights need "
            f"{reserve_bytes / 2**30:.1f} GiB; there is no room for a KV cache"
        )
    return blocks
