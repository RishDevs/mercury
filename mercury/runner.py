from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .allocator import BlockAllocator
from .config import EngineConfig, ModelConfig
from .model.attention import DecodeContext, PrefillContext
from .request import Request
from .sampling import sample_tokens
from .scheduler import Plan


@dataclass
class ExecResult:
    prefill_tokens: list[int] = field(default_factory=list)
    decode_tokens: list[int] = field(default_factory=list)
    prepare_seconds: float = 0.0
    forward_seconds: float = 0.0


def slot_mapping(blocks: list[int], start: int, stop: int, block_size: int) -> np.ndarray:
    """Physical KV slot for each logical position in [start, stop).

    The decode token at logical position seq_len-1 lands in block seq_len-1 // block_size;
    computing it from seq_len instead is the classic off-by-one that produces correct
    output for exactly one block and then garbage.
    """
    positions = np.arange(start, stop, dtype=np.int64)
    table = np.asarray(blocks, dtype=np.int64)
    return table[positions // block_size] * block_size + (positions % block_size)


class ModelRunner:
    """Turns a plan into tensors, runs Qwen2, samples one token per request.

    The interface the engine depends on is exactly `execute(plan) -> ExecResult`, which is
    what lets the whole scheduler and allocator be tested on a laptop against a fake.
    """

    def __init__(
        self,
        cfg: EngineConfig,
        alloc: BlockAllocator | None = None,
        model=None,
        model_config: ModelConfig | None = None,
        tokenizer_path=None,
    ):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dtype = cfg.torch_dtype
        self.alloc = alloc
        self.model_path = None
        self.revision = cfg.revision

        if model is None:
            from .model.loader import load_qwen2

            model, model_config, self.model_path, self.revision = load_qwen2(cfg, self.device)
        self.model = model
        self.model_config = model_config or model.config

        torch.manual_seed(cfg.seed)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(cfg.seed)

        mc = self.model_config
        self.kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(mc.num_hidden_layers):
            shape = (cfg.num_blocks, cfg.block_size, mc.num_key_value_heads, mc.head_dim)
            self.kv_caches.append(
                (
                    torch.zeros(shape, dtype=self.dtype, device=self.device),
                    torch.zeros(shape, dtype=self.dtype, device=self.device),
                )
            )
        self.kv_bytes = (
            2
            * mc.num_hidden_layers
            * cfg.num_blocks
            * cfg.block_size
            * mc.num_key_value_heads
            * mc.head_dim
            * torch.finfo(self.dtype).bits
            // 8
        )

    # ------------------------------------------------------------------ helpers

    def _to_device(self, array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(array).to(self.device, non_blocking=True)

    def _slots_for(self, blocks: list[int], start: int, stop: int) -> np.ndarray:
        return slot_mapping(blocks, start, stop, self.cfg.block_size)

    def _sampling_tensors(self, requests: list[Request]) -> tuple[torch.Tensor, torch.Tensor]:
        temps = np.fromiter((r.params.temperature for r in requests), dtype=np.float32, count=len(requests))
        tops = np.fromiter((r.params.top_p for r in requests), dtype=np.float32, count=len(requests))
        return self._to_device(temps), self._to_device(tops)

    # ------------------------------------------------------------------ the step

    @torch.inference_mode()
    def execute(self, plan: Plan) -> ExecResult:
        started = time.perf_counter()
        prepare = 0.0
        bs = self.cfg.block_size

        prefill_hidden = []
        for r in plan.prefill:
            mark = time.perf_counter()
            start = r.num_cached
            ids = np.asarray(r.tokens[start:], dtype=np.int64)
            positions = np.arange(start, r.seq_len, dtype=np.int64)
            slots = self._slots_for(r.blocks, start, r.seq_len)
            ctx = PrefillContext(
                self.kv_caches,
                self._to_device(np.asarray(r.blocks, dtype=np.int64)),
                self._to_device(slots),
                r.seq_len,
                start,
            )
            id_tensor = self._to_device(ids)
            pos_tensor = self._to_device(positions)
            prepare += time.perf_counter() - mark
            hidden = self.model(id_tensor, pos_tensor, ctx)
            prefill_hidden.append(hidden[-1:])

        decode_hidden = None
        if plan.decode:
            mark = time.perf_counter()
            count = len(plan.decode)
            widest = max(len(r.blocks) for r in plan.decode)
            ids = np.empty(count, dtype=np.int64)
            positions = np.empty(count, dtype=np.int64)
            slots = np.empty(count, dtype=np.int64)
            seq_lens = np.empty(count, dtype=np.int64)
            tables = np.empty((count, widest), dtype=np.int64)
            for i, r in enumerate(plan.decode):
                last = r.seq_len - 1
                ids[i] = r.tokens[last]
                positions[i] = last
                slots[i] = r.blocks[last // bs] * bs + (last % bs)
                seq_lens[i] = r.seq_len
                owned = len(r.blocks)
                tables[i, :owned] = r.blocks
                tables[i, owned:] = r.blocks[-1]
            ctx = DecodeContext(
                self.kv_caches,
                self._to_device(tables),
                self._to_device(slots),
                self._to_device(seq_lens),
            )
            id_tensor = self._to_device(ids)
            pos_tensor = self._to_device(positions)
            prepare += time.perf_counter() - mark
            decode_hidden = self.model(id_tensor, pos_tensor, ctx)

        result = ExecResult()
        if prefill_hidden:
            logits = self.model.compute_logits(torch.cat(prefill_hidden, dim=0))
            temps, tops = self._sampling_tensors(plan.prefill)
            result.prefill_tokens = sample_tokens(
                logits, temps, tops, self.generator
            ).tolist()
        if decode_hidden is not None:
            logits = self.model.compute_logits(decode_hidden)
            temps, tops = self._sampling_tensors(plan.decode)
            result.decode_tokens = sample_tokens(logits, temps, tops, self.generator).tolist()

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        result.prepare_seconds = prepare
        result.forward_seconds = max(0.0, time.perf_counter() - started - prepare)
        return result

    # ------------------------------------------------------------------ offline

    @torch.inference_mode()
    def forward_prompt(self, token_ids: list[int], all_positions: bool = False) -> torch.Tensor:
        """Logits for a prompt through the paged path, using scratch blocks. For tests."""
        if self.alloc is None:
            raise RuntimeError("forward_prompt needs an allocator to borrow blocks from")
        got = self.alloc.allocate_prompt([], len(token_ids))
        if got is None:
            raise RuntimeError("not enough free blocks for forward_prompt")
        blocks, _ = got
        try:
            ctx = PrefillContext(
                self.kv_caches,
                self._to_device(np.asarray(blocks, dtype=np.int64)),
                self._to_device(self._slots_for(blocks, 0, len(token_ids))),
                len(token_ids),
                0,
            )
            hidden = self.model(
                self._to_device(np.asarray(token_ids, dtype=np.int64)),
                self._to_device(np.arange(len(token_ids), dtype=np.int64)),
                ctx,
            )
            return self.model.compute_logits(hidden if all_positions else hidden[-1:])
        finally:
            self.alloc.release(blocks)

    def snapshot(self) -> dict:
        return {
            "device": str(self.device),
            "dtype": str(self.dtype).replace("torch.", ""),
            "kv_bytes": self.kv_bytes,
            "layers": self.model_config.num_hidden_layers,
            "kv_heads": self.model_config.num_key_value_heads,
            "head_dim": self.model_config.head_dim,
            "model_path": str(self.model_path) if self.model_path else None,
            "revision": self.revision,
        }
