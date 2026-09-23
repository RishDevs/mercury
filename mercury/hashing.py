from __future__ import annotations

import hashlib
from typing import Sequence

_CRYPTO = False


def use_cryptographic_hashes(enabled: bool) -> None:
    """Must be called before any request is hashed; switching mid-run splits the cache."""
    global _CRYPTO
    _CRYPTO = enabled


def cryptographic_hashes_enabled() -> bool:
    return _CRYPTO


def block_hash(parent: int, tokens: Sequence[int]) -> int:
    if _CRYPTO:
        digest = hashlib.sha256()
        digest.update(parent.to_bytes(16, "big", signed=False))
        digest.update(b"".join(int(t).to_bytes(4, "big", signed=False) for t in tokens))
        return int.from_bytes(digest.digest()[:16], "big")
    return hash((parent, tuple(tokens)))


def block_hashes(token_ids: Sequence[int], block_size: int, parent: int = 0) -> list[int]:
    """Chain hash over FULL blocks only; h_i depends on every token before it."""
    out: list[int] = []
    for start in range(0, len(token_ids) - block_size + 1, block_size):
        parent = block_hash(parent, token_ids[start : start + block_size])
        out.append(parent)
    return out


def extend_hashes(hashes: list[int], token_ids: Sequence[int], block_size: int) -> int:
    """Grow an existing chain to cover every full block of token_ids. Returns blocks added."""
    full = len(token_ids) // block_size
    added = 0
    parent = hashes[-1] if hashes else 0
    for index in range(len(hashes), full):
        parent = block_hash(parent, token_ids[index * block_size : (index + 1) * block_size])
        hashes.append(parent)
        added += 1
    return added
