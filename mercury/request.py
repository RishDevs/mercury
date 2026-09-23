from __future__ import annotations

import enum
from dataclasses import dataclass, field


class RequestState(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    ABORTED = "aborted"


@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_p: float = 1.0
    ignore_eos: bool = False
    seed: int | None = None

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0


@dataclass
class Request:
    rid: str
    prompt: list[int]
    max_tokens: int
    arrival: float
    hashes: list[int]
    params: SamplingParams = field(default_factory=SamplingParams)
    session_id: str | None = None

    out: list[int] = field(default_factory=list)
    blocks: list[int] = field(default_factory=list)
    num_cached: int = 0
    committed: int = 0
    state: RequestState = RequestState.WAITING
    admitted_at: float | None = None
    last_admitted_at: float | None = None
    first_token_at: float | None = None
    finished_at: float | None = None
    preemptions: int = 0
    recomputed_tokens: int = 0
    finish_reason: str | None = None

    @property
    def tokens(self) -> list[int]:
        return self.prompt + self.out

    @property
    def seq_len(self) -> int:
        return len(self.prompt) + len(self.out)

    @property
    def prompt_len(self) -> int:
        return len(self.prompt)

    @property
    def max_seq_len(self) -> int:
        return len(self.prompt) + self.max_tokens

    @property
    def waiting(self) -> bool:
        return self.state is RequestState.WAITING

    @property
    def running(self) -> bool:
        return self.state is RequestState.RUNNING

    @property
    def done(self) -> bool:
        return self.state in (RequestState.FINISHED, RequestState.ABORTED)

    def queue_delay(self) -> float | None:
        return None if self.admitted_at is None else self.admitted_at - self.arrival

    def record(self) -> dict:
        return {
            "rid": self.rid,
            "session_id": self.session_id,
            "arrival": self.arrival,
            "admitted_at": self.admitted_at,
            "first_token_at": self.first_token_at,
            "finished_at": self.finished_at,
            "queue_delay": self.queue_delay(),
            "prompt_tokens": len(self.prompt),
            "output_tokens": len(self.out),
            "cached_tokens": self.num_cached,
            "recomputed_tokens": self.recomputed_tokens,
            "preemptions": self.preemptions,
            "finish_reason": self.finish_reason,
            "state": self.state.value,
        }
