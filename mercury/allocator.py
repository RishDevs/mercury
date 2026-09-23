from __future__ import annotations

from collections import OrderedDict, deque
from typing import Sequence


class BlockAllocator:
    """Fixed-size physical blocks plus a hash-addressed cache of shareable prefix blocks.

    Every block is in exactly one of three states and every allocator bug is a violation
    of that invariant:

        free       ref == 0, no hash, sitting in `free`
        in use     ref >= 1, hashed once it is full, listed in some request's block table
        cached     ref == 0, hashed, sitting in `evictable` (LRU by release time)
    """

    def __init__(self, num_blocks: int, block_size: int):
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.ref = [0] * num_blocks
        self.hash_of: list[int | None] = [None] * num_blocks
        self.by_hash: dict[int, int] = {}
        self.free = deque(range(num_blocks))
        self.evictable: OrderedDict[int, None] = OrderedDict()
        self.stats = {
            "hit_tokens": 0,
            "query_tokens": 0,
            "evictions": 0,
            "prefills": 0,
            "allocation_failures": 0,
        }

    def num_available(self) -> int:
        return len(self.free) + len(self.evictable)

    def num_cached_blocks(self) -> int:
        return len(self.by_hash)

    def hit_rate(self) -> float:
        queried = self.stats["query_tokens"]
        return self.stats["hit_tokens"] / queried if queried else 0.0

    def cached_prefix_blocks(self, hashes: Sequence[int]) -> list[int]:
        out: list[int] = []
        for h in hashes:
            block = self.by_hash.get(h)
            if block is None:
                break
            out.append(block)
        return out

    def _take(self) -> int:
        if self.free:
            return self.free.popleft()
        block, _ = self.evictable.popitem(last=False)
        del self.by_hash[self.hash_of[block]]
        self.hash_of[block] = None
        self.stats["evictions"] += 1
        return block

    def plan_prompt(self, hashes: Sequence[int], num_tokens: int) -> tuple[list[int], int]:
        """Read-only: which cached blocks a prompt would reuse, and how many new it needs.

        At least one token is always left to recompute, otherwise prefill produces no
        logits for the first output token.
        """
        bs = self.block_size
        hit = self.cached_prefix_blocks(hashes)[: (num_tokens - 1) // bs]
        need_new = -(-num_tokens // bs) - len(hit)
        return hit, need_new

    def allocate_prompt(
        self, hashes: Sequence[int], num_tokens: int
    ) -> tuple[list[int], int] | None:
        """(block_ids, cached_token_count), or None with no side effects if it does not fit."""
        hit, need_new = self.plan_prompt(hashes, num_tokens)
        pinned = sum(1 for b in hit if self.ref[b] == 0)
        if need_new > self.num_available() - pinned:
            self.stats["allocation_failures"] += 1
            return None
        for b in hit:
            if self.ref[b] == 0:
                del self.evictable[b]
            self.ref[b] += 1
        new = [self._take() for _ in range(need_new)]
        for b in new:
            self.ref[b] = 1
        cached_tokens = len(hit) * self.block_size
        self.stats["hit_tokens"] += cached_tokens
        self.stats["query_tokens"] += num_tokens
        self.stats["prefills"] += 1
        return hit + new, cached_tokens

    def append_block(self) -> int | None:
        if self.num_available() == 0:
            return None
        block = self._take()
        self.ref[block] = 1
        return block

    def commit(self, block: int, h: int) -> None:
        """Make a now-full block shareable. First writer of a hash wins; duplicates stay private."""
        if self.hash_of[block] is None and h not in self.by_hash:
            self.hash_of[block] = h
            self.by_hash[h] = block

    def release(self, blocks: Sequence[int]) -> None:
        # Reverse order so private suffixes enter the LRU ahead of the shared prefix head,
        # and are therefore evicted first.
        for b in reversed(blocks):
            self.ref[b] -= 1
            if self.ref[b] < 0:
                raise AssertionError(f"double free of block {b}")
            if self.ref[b] == 0:
                if self.hash_of[b] is not None:
                    self.evictable[b] = None
                else:
                    self.free.append(b)

    def reset_stats(self) -> None:
        for key in self.stats:
            self.stats[key] = 0

    def snapshot(self) -> dict:
        return {
            "num_blocks": self.num_blocks,
            "block_size": self.block_size,
            "free_blocks": len(self.free),
            "evictable_blocks": len(self.evictable),
            "available_blocks": self.num_available(),
            "cached_blocks": self.num_cached_blocks(),
            "hit_rate": self.hit_rate(),
            **self.stats,
        }

    def check_invariants(self) -> None:
        """O(num_blocks); for tests and the --debug-invariants flag, not the hot path."""
        free = set(self.free)
        evictable = set(self.evictable)
        if len(free) != len(self.free):
            raise AssertionError("duplicate block id in the free list")
        if free & evictable:
            raise AssertionError(f"blocks in both free and evictable: {sorted(free & evictable)}")
        for b in range(self.num_blocks):
            if self.ref[b] < 0:
                raise AssertionError(f"block {b} has ref {self.ref[b]}")
            if b in free:
                if self.ref[b] != 0 or self.hash_of[b] is not None:
                    raise AssertionError(f"free block {b} is hashed or referenced")
            elif b in evictable:
                if self.ref[b] != 0 or self.hash_of[b] is None:
                    raise AssertionError(f"evictable block {b} is unhashed or referenced")
            elif self.ref[b] == 0:
                raise AssertionError(f"block {b} has ref 0 but is in neither free nor evictable")
        for h, b in self.by_hash.items():
            if self.hash_of[b] != h:
                raise AssertionError(f"by_hash[{h}] -> {b} but hash_of[{b}] = {self.hash_of[b]}")
