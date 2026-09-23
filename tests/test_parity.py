from __future__ import annotations

import os

import pytest
import torch

from mercury.config import EngineConfig
from mercury.engine import Engine
from mercury.request import SamplingParams

from .fake_runner import collect

pytestmark = [pytest.mark.gpu, pytest.mark.hf, pytest.mark.slow]

MODEL = os.environ.get("MERCURY_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
REVISION = os.environ.get("MERCURY_TEST_REVISION") or None
COMPARE_TOKENS = 16
NUM_PROMPTS = 50
MIN_AGREEING = 48

TOPICS = [
    "a paged memory allocator", "rotary position embeddings", "the TCP three-way handshake",
    "a red-black tree rebalance", "gradient checkpointing", "speculative decoding",
    "the Raft leader election", "copy-on-write fork", "a bloom filter", "CUDA warp divergence",
    "the LRU-K eviction policy", "consistent hashing", "write-ahead logging", "vectorised I/O",
    "the Karatsuba multiplication", "a skip list", "branch prediction", "the Kalman filter",
    "reservoir sampling", "the Hungarian algorithm", "quorum reads", "a trie index",
    "bloom hierarchies in LSM trees", "the actor model", "lock striping",
]
TEMPLATES = [
    "Explain how {} works, step by step.",
    "What goes wrong when you implement {} naively?",
]


def prompts_for(tokenizer) -> list[list[int]]:
    texts = [
        TEMPLATES[i % len(TEMPLATES)].format(TOPICS[i % len(TOPICS)]) for i in range(NUM_PROMPTS)
    ]
    return [tokenizer.encode(t) for t in texts]


@pytest.fixture(scope="module")
def reference():
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=torch.float16
    ).cuda().eval()
    return tokenizer, model


def build_engine(num_blocks=4096, **overrides) -> Engine:
    cfg = EngineConfig(
        model=MODEL,
        revision=REVISION,
        device="cuda",
        dtype="float16",
        num_blocks=num_blocks,
        max_running=overrides.pop("max_running", 64),
        max_model_len=overrides.pop("max_model_len", 2048),
        **overrides,
    )
    return Engine(cfg)


def run_to_completion(engine: Engine, prompts, max_tokens=COMPARE_TOKENS):
    buckets = {}
    for index, ids in enumerate(prompts):
        rid = f"p{index}"
        buckets[rid] = []
        engine.submit(
            rid, ids, max_tokens, collect(buckets[rid]), params=SamplingParams(ignore_eos=True)
        )
    engine._drain_commands()
    guard = 0
    while engine.sched.has_work() and guard < 100_000:
        engine.step()
        guard += 1
    return {
        rid: [e.token for e in events if e.kind == "token"] for rid, events in buckets.items()
    }


def hf_greedy(model, ids, max_tokens=COMPARE_TOKENS) -> list[int]:
    tensor = torch.tensor([ids], device="cuda")
    with torch.inference_mode():
        out = model.generate(
            tensor,
            max_new_tokens=max_tokens,
            min_new_tokens=max_tokens,
            do_sample=False,
            eos_token_id=None,
            pad_token_id=0,
        )
    return out[0, len(ids) :].tolist()


def agreement(ours: dict, theirs: list[list[int]]) -> int:
    return sum(
        1
        for index, expected in enumerate(theirs)
        if ours[f"p{index}"][:COMPARE_TOKENS] == expected[:COMPARE_TOKENS]
    )


@pytest.fixture(scope="module")
def expected(reference):
    tokenizer, model = reference
    prompts = prompts_for(tokenizer)
    return prompts, [hf_greedy(model, ids) for ids in prompts]


def test_first_step_logits_match(reference, expected):
    _tokenizer, model = reference
    prompts, _ = expected
    engine = build_engine()
    try:
        worst = 0.0
        for ids in prompts[:10]:
            ours = engine.runner.forward_prompt(ids)[0].float()
            with torch.inference_mode():
                theirs = model(torch.tensor([ids], device="cuda")).logits[0, -1].float()
            worst = max(worst, (ours - theirs).abs().max().item())
        assert worst < 1e-2, f"max absolute logit difference {worst}"
    finally:
        engine.shutdown()


def test_greedy_parity_single_request(expected):
    prompts, theirs = expected
    engine = build_engine(max_running=1)
    try:
        ours = {}
        for index, ids in enumerate(prompts):
            ours.update(run_to_completion(engine, [ids]))
            ours[f"p{index}"] = ours.pop("p0")
        assert agreement(ours, theirs) >= MIN_AGREEING
    finally:
        engine.shutdown()


def test_greedy_parity_in_a_batch(expected):
    prompts, theirs = expected
    engine = build_engine(max_running=64)
    try:
        assert agreement(run_to_completion(engine, prompts), theirs) >= MIN_AGREEING
    finally:
        engine.shutdown()


def test_greedy_parity_with_a_warm_prefix_cache(expected):
    """A prefix-cache bug shows up here and nowhere else, which is the reason this mode
    exists: the second pass reuses blocks the first pass committed."""
    prompts, theirs = expected
    engine = build_engine(max_running=16)
    try:
        run_to_completion(engine, prompts)
        assert engine.alloc.stats["hit_tokens"] == 0
        second = run_to_completion(engine, prompts)
        assert engine.alloc.stats["hit_tokens"] > 0
        assert agreement(second, theirs) >= MIN_AGREEING
    finally:
        engine.shutdown()


def test_greedy_parity_after_forced_preemption(expected):
    """A pool this small cannot hold the batch, so requests are evicted mid-generation and
    resumed as longer prompts. Output must not change."""
    prompts, theirs = expected
    longest = max(len(p) for p in prompts) + COMPARE_TOKENS
    engine = build_engine(num_blocks=(longest // 16 + 2) * 2, max_running=16, max_model_len=2048)
    try:
        ours = run_to_completion(engine, prompts)
        assert engine.sched.stats["preemptions"] > 0
        assert agreement(ours, theirs) >= MIN_AGREEING
    finally:
        engine.shutdown()


def test_the_fused_norm_does_not_change_the_output(expected):
    prompts, theirs = expected
    plain = build_engine(fused_rmsnorm=False, max_running=16)
    try:
        assert agreement(run_to_completion(plain, prompts), theirs) >= MIN_AGREEING
    finally:
        plain.shutdown()


def test_no_blocks_leak_across_a_long_run(expected):
    prompts, _ = expected
    engine = build_engine(num_blocks=512, max_running=16)
    try:
        for _ in range(4):
            run_to_completion(engine, prompts, max_tokens=8)
        assert engine.leaked_blocks() == 0
        engine.alloc.check_invariants()
    finally:
        engine.shutdown()
