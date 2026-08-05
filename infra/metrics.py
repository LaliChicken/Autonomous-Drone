"""Per-stage timing and percentile rollups.

Usage::

    metrics = Metrics.from_config(cfg)
    with metrics.stage("depth"):
        result = backend.infer(bundle)
    ...
    for name, stats in metrics.rollup().items():
        print(name, stats.p50_ms, stats.p99_ms)

Timing uses CLOCK_MONOTONIC nanoseconds, like every other timestamp in the
stack. Samples live in a bounded ring per stage so a long flight cannot grow
the process without limit; the window comes from ``metrics.window`` in config.

Percentiles use the nearest-rank definition rather than an interpolating one.
Interpolated percentiles invent a value that was never measured, and for a
latency budget the honest answer is "the 99th-slowest tick actually took this
long". Nearest-rank is also exactly reproducible, which matters because the
replay determinism test compares rollups.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from config import Config

NS_PER_MS = 1_000_000.0


@dataclass(frozen=True)
class StageStats:
    """Rollup for one stage. All durations are integer nanoseconds."""

    name: str
    count: int
    p50_ns: int
    p99_ns: int
    min_ns: int
    max_ns: int
    mean_ns: float

    @property
    def p50_ms(self) -> float:
        return self.p50_ns / NS_PER_MS

    @property
    def p99_ms(self) -> float:
        return self.p99_ns / NS_PER_MS

    @property
    def min_ms(self) -> float:
        return self.min_ns / NS_PER_MS

    @property
    def max_ms(self) -> float:
        return self.max_ns / NS_PER_MS

    @property
    def mean_ms(self) -> float:
        return self.mean_ns / NS_PER_MS

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "p50_ns": self.p50_ns,
            "p99_ns": self.p99_ns,
            "min_ns": self.min_ns,
            "max_ns": self.max_ns,
            "mean_ns": self.mean_ns,
        }


def nearest_rank(sorted_samples: list[int], percentile: float) -> int:
    """Nearest-rank percentile of an already-sorted, non-empty sample list.

    ``percentile`` is 0..100; p0 is the minimum and p100 the maximum. The
    returned value is always one of the input samples.
    """
    if not sorted_samples:
        raise ValueError("nearest_rank requires at least one sample")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"percentile must be within [0, 100], got {percentile}")
    n = len(sorted_samples)
    rank = max(1, min(n, math.ceil(percentile / 100.0 * n)))
    return sorted_samples[rank - 1]


class Metrics:
    """Thread-safe per-stage timing registry.

    Safe to share between the capture, perception, and offboard threads: every
    mutation happens under one lock, and the lock is never held across the
    caller's code.
    """

    def __init__(self, window: int = 2048) -> None:
        if window < 1:
            raise ValueError(f"metrics window must be >= 1, got {window}")
        self._window = window
        self._samples: dict[str, deque[int]] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, cfg: Config) -> Metrics:
        return cls(window=cfg.metrics.window)

    @property
    def window(self) -> int:
        return self._window

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time a block and file the duration under ``name``.

        Records even when the block raises, so a stage that fails slowly still
        appears in the rollup instead of vanishing from it.
        """
        start = time.monotonic_ns()
        try:
            yield
        finally:
            self.record(name, time.monotonic_ns() - start)

    def record(self, name: str, elapsed_ns: int) -> None:
        """File a pre-measured duration. Negative durations are rejected."""
        if elapsed_ns < 0:
            raise ValueError(f"elapsed_ns must be >= 0, got {elapsed_ns}")
        with self._lock:
            bucket = self._samples.get(name)
            if bucket is None:
                bucket = deque(maxlen=self._window)
                self._samples[name] = bucket
            bucket.append(int(elapsed_ns))

    def samples(self, name: str) -> list[int]:
        with self._lock:
            return list(self._samples.get(name, ()))

    def stage_names(self) -> list[str]:
        with self._lock:
            return sorted(self._samples)

    def rollup(self) -> dict[str, StageStats]:
        """Percentile summary per stage, ordered by stage name.

        Stages that were never recorded are absent rather than zeroed: a stage
        with no samples has no latency, and a 0 ms row reads as "instant" on a
        dashboard, which is the opposite of the truth.
        """
        with self._lock:
            snapshot = {name: list(bucket) for name, bucket in self._samples.items()}
        out: dict[str, StageStats] = {}
        for name in sorted(snapshot):
            values = sorted(snapshot[name])
            if not values:
                continue
            out[name] = StageStats(
                name=name,
                count=len(values),
                p50_ns=nearest_rank(values, 50.0),
                p99_ns=nearest_rank(values, 99.0),
                min_ns=values[0],
                max_ns=values[-1],
                mean_ns=sum(values) / len(values),
            )
        return out

    def to_dict(self) -> dict[str, Any]:
        return {name: stats.to_dict() for name, stats in self.rollup().items()}

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()
