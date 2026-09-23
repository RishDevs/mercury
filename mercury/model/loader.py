from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from safetensors import safe_open

from ..config import EngineConfig, ModelConfig
from .qwen2 import Qwen2ForCausalLM

_ALLOW = ["config.json", "*.safetensors", "*.safetensors.index.json", "generation_config.json"]
_TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]


def resolve_model(model: str, revision: str | None = None) -> tuple[Path, str | None]:
    """Local directory, or a Hub snapshot pinned to a revision. Returns (path, commit)."""
    if os.path.isdir(model):
        return Path(model), revision
    from huggingface_hub import snapshot_download

    path = Path(
        snapshot_download(
            model,
            revision=revision,
            allow_patterns=_ALLOW + _TOKENIZER_FILES,
        )
    )
    commit = path.name if path.parent.name == "snapshots" else revision
    return path, commit


def _weight_files(path: Path) -> list[Path]:
    index = path / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        return sorted({path / name for name in weight_map.values()})
    single = path / "model.safetensors"
    if single.exists():
        return [single]
    shards = sorted(path.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors weights under {path}")
    return shards


def build_model(
    model_cfg: ModelConfig, engine_cfg: EngineConfig, device: torch.device | str
) -> Qwen2ForCausalLM:
    previous = torch.get_default_dtype()
    torch.set_default_dtype(engine_cfg.torch_dtype)
    try:
        with torch.device("meta"):
            model = Qwen2ForCausalLM(model_cfg, fused_norm=engine_cfg.fused_rmsnorm)
    finally:
        torch.set_default_dtype(previous)
    model.to_empty(device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_weights(model: Qwen2ForCausalLM, path: Path, device: torch.device | str) -> None:
    params = dict(model.named_parameters(remove_duplicate=False))
    filled: set[str] = set()
    unexpected: list[str] = []
    for file in _weight_files(path):
        with safe_open(str(file), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                target = params.get(name)
                if target is None:
                    unexpected.append(name)
                    continue
                tensor = handle.get_tensor(name)
                if tuple(tensor.shape) != tuple(target.shape):
                    raise ValueError(
                        f"{name}: checkpoint has {tuple(tensor.shape)}, "
                        f"model expects {tuple(target.shape)}"
                    )
                target.data.copy_(tensor.to(device=device, dtype=target.dtype))
                filled.add(name)

    missing = sorted(set(params) - filled)
    if missing:
        raise RuntimeError(f"checkpoint is missing {len(missing)} tensors, first: {missing[:4]}")
    tied = model.lm_head is None
    surprises = [n for n in unexpected if not (tied and n == "lm_head.weight")]
    if surprises:
        raise RuntimeError(f"checkpoint has tensors this model has no home for: {surprises[:4]}")


def load_qwen2(engine_cfg: EngineConfig, device: torch.device | str | None = None):
    device = torch.device(device or engine_cfg.device)
    path, commit = resolve_model(engine_cfg.model, engine_cfg.revision)
    model_cfg = ModelConfig.from_directory(path)
    engine_cfg.validate_against(model_cfg)
    model = build_model(model_cfg, engine_cfg, device)
    load_weights(model, path, device)
    model.init_rotary(device, engine_cfg.torch_dtype, engine_cfg.max_model_len)
    return model, model_cfg, path, commit


def load_tokenizer(path: Path):
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None
    try:
        return AutoTokenizer.from_pretrained(str(path))
    except Exception:
        return None
