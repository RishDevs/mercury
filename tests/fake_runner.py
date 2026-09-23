from __future__ import annotations

from mercury.testing import DeterministicRunner as FakeRunner
from mercury.testing import FakeClock

__all__ = ["FakeRunner", "FakeClock", "collect"]


def collect(events: list):
    """A sink that just records; what the offline tests use instead of an asyncio queue."""

    def sink(event):
        events.append(event)

    return sink
