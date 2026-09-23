"""The real model runner, the real paged attention, a tiny random-weight Qwen2 on CPU.

This is the four parity modes from tests/test_parity.py — single, batched, warm prefix
cache, after preemption — run against a model small enough to build in a millisecond. It
cannot tell you the weights are loaded correctly, which is what the GPU parity test is for.
It can tell you that caching a prefix, batching, and resuming a preempted request all
produce byte-identical output, which is the entire class of bug that makes an inference
server quietly wrong.
"""

from __future__ import annotations

import torch

from mercury.config import EngineConfig, ModelConfig
from mercury.engine import Engine
from mercury.model.qwen2 import Qwen2ForCausalLM
from mercury.request import SamplingParams
from mercury.runner import ModelRunner

from .fake_runner import collect

TINY = ModelConfig(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    vocab_size=512,
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
    max_position_embeddings=2048,
    tie_word_embeddings=True,
    eos_token_ids=(),
)


def build(num_blocks: int = 64, max_running: int = 8, policy: str = "fcfs") -> Engine:
    cfg = EngineConfig(
        model="tiny",
        device="cpu",
        dtype="float32",
        num_blocks=num_blocks,
        block_size=16,
        max_running=max_running,
        max_model_len=512,
        policy=policy,
        fused_rmsnorm=False,
        seed=0,
    )
    torch.manual_seed(7)
    model = Qwen2ForCausalLM(TINY, fused_norm=False)
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.05)
    model.eval()
    model.init_rotary("cpu", torch.float32, cfg.max_model_len)

    runner = ModelRunner(cfg, model=model, model_config=TINY)
    engine = Engine(cfg, runner=runner, model_config=TINY)
    runner.alloc = engine.alloc
    return engine


def run(engine: Engine, prompts: dict[str, list[int]], max_tokens: int = 12) -> dict[str, list[int]]:
    buckets: dict[str, list] = {}
    for rid, ids in prompts.items():
        buckets[rid] = []
        engine.submit(
            rid, ids, max_tokens, collect(buckets[rid]), params=SamplingParams(ignore_eos=True)
        )
    engine._drain_commands()
    guard = 0
    while engine.sched.has_work() and guard < 20_000:
        engine.step()
        guard += 1
    return {rid: [e.token for e in events if e.kind == "token"] for rid, events in buckets.items()}


PROMPT = [(i * 13 + 7) % 500 for i in range(80)]
OTHER = [(i * 29 + 3) % 500 for i in range(56)]


def test_generation_is_deterministic_across_engines():
    first = build()
    second = build()
    assert run(first, {"a": PROMPT}) == run(second, {"a": PROMPT})


def test_a_warm_prefix_cache_does_not_change_the_output():
    engine = build()
    cold = run(engine, {"a": PROMPT})
    assert engine.alloc.stats["hit_tokens"] == 0
    warm = run(engine, {"b": PROMPT})
    assert engine.alloc.stats["hit_tokens"] > 0
    assert warm["b"] == cold["a"]


def test_a_partially_shared_prefix_does_not_change_the_output():
    engine = build()
    extended = PROMPT + [11] * 24
    alone = build()
    expected = run(alone, {"x": extended})["x"]

    run(engine, {"a": PROMPT})
    got = run(engine, {"b": extended})
    assert engine.alloc.stats["hit_tokens"] >= 64
    assert got["b"] == expected


def test_batching_does_not_change_any_request_s_output():
    solo = {rid: run(build(), {rid: ids})[rid] for rid, ids in (("a", PROMPT), ("b", OTHER))}
    together = run(build(), {"a": PROMPT, "b": OTHER})
    assert together["a"] == solo["a"]
    assert together["b"] == solo["b"]


def test_a_larger_batch_still_matches_the_single_request_answer():
    expected = run(build(), {"a": PROMPT})["a"]
    crowd = {f"f{i}": [(i * 31 + j) % 500 for j in range(48)] for i in range(6)}
    crowd["a"] = PROMPT
    assert run(build(max_running=8), crowd)["a"] == expected


def test_forced_preemption_does_not_change_the_output():
    expected = run(build(), {"a": PROMPT}, max_tokens=24)["a"]
    # Each request grows from 5 blocks to 7; two of them fit at admission and then cannot
    # both keep growing inside a pool of 11.
    tight = build(num_blocks=11, max_running=4)
    crowd = {f"f{i}": [(i * 37 + j) % 500 for j in range(80)] for i in range(3)}
    crowd["a"] = PROMPT
    got = run(tight, crowd, max_tokens=24)
    assert tight.sched.stats["preemptions"] > 0
    assert got["a"] == expected
    assert tight.leaked_blocks() == 0


def test_every_policy_produces_the_same_tokens():
    prompts = {"a": PROMPT, "b": OTHER, "c": PROMPT + [5] * 16}
    baseline = run(build(policy="fcfs"), dict(prompts))
    for policy in ("lpm", "lpm_aging"):
        assert run(build(policy=policy), dict(prompts)) == baseline


def test_rope_positions_survive_a_cached_prefix():
    """If the suffix's rotary positions restarted at zero instead of at num_cached, the
    output would still be fluent and would silently differ from the uncached answer."""
    engine = build()
    long_prompt = PROMPT + OTHER
    expected = run(build(), {"x": long_prompt})["x"]
    run(engine, {"warm": PROMPT})
    assert engine.alloc.stats["hit_tokens"] == 0
    got = run(engine, {"x": long_prompt})
    assert engine.alloc.stats["hit_tokens"] > 0
    assert got["x"] == expected


def test_where_the_blocks_physically_land_does_not_change_the_logits():
    """Same prompt, same model, completely different physical blocks. If the block table,
    the slot mapping and the gather were not consistent with each other, these would
    differ."""
    from mercury.model.attention import PrefillContext

    engine = build()
    runner = engine.runner
    first = runner.forward_prompt(PROMPT)[0]

    scattered = [61, 3, 47, 12, 55, 8]
    ctx = PrefillContext(
        runner.kv_caches,
        torch.tensor(scattered, dtype=torch.int64),
        torch.from_numpy(runner._slots_for(scattered, 0, len(PROMPT))),
        len(PROMPT),
        0,
    )
    hidden = runner.model(
        torch.tensor(PROMPT, dtype=torch.int64),
        torch.arange(len(PROMPT), dtype=torch.int64),
        ctx,
    )
    second = runner.model.compute_logits(hidden[-1:])[0]
    assert torch.allclose(first, second, atol=1e-5)


def test_forward_prompt_returns_its_scratch_blocks():
    engine = build()
    before = engine.alloc.num_available()
    engine.runner.forward_prompt(PROMPT)
    assert engine.alloc.num_available() == before
    engine.alloc.check_invariants()
