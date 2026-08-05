"""Temporal polar occupancy map.

Holds one range estimate per azimuth bin, decays it with age, and reports a
snapshot the planner can act on.

Three rules govern everything here:

1. **A bin that ages out becomes unknown, never clear.** Forgetting an
   obstacle is not the same as observing empty space. The decayed state is
   `unknown=True` with a NaN range, which fails every "is this far enough"
   test rather than passing it.
2. **Merging keeps the nearer of the new and the remembered range.** If two
   recent observations disagree about how close something is, the closer one
   is the one worth flying by. The cost is that a transient false positive
   holds the bin pessimistic for up to `max_age_s`; decay bounds that, and
   pessimistic is the survivable direction.
3. **`danger` and `unknown` are separate flags and neither implies the other.**
   `danger` means "something measured is too close for the current speed";
   `unknown` means "nothing was measured". A planner must refuse both, but for
   different reasons, and collapsing them would hide which one is happening.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from perception.obstacles import ObstacleScan, bin_centres
from sources.types import OccupancySnapshot

if TYPE_CHECKING:
    from config import Config

NS_PER_S = 1_000_000_000.0


class OccupancyMap:
    """Azimuth-binned range memory with age decay.

    Timestamps are CLOCK_MONOTONIC nanoseconds throughout; the map never reads
    a clock itself, so replaying a log reproduces the same state exactly.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.occupancy = cfg.occupancy
        self.n_bins = self.occupancy.n_bins
        self.bearings = bin_centres(self.n_bins, self.occupancy.fov_rad)
        self._distances = np.full(self.n_bins, np.nan, dtype=np.float32)
        self._confidence = np.zeros(self.n_bins, dtype=np.float32)
        self._updated_ns = np.zeros(self.n_bins, dtype=np.int64)
        self._observed = np.zeros(self.n_bins, dtype=bool)
        self._last_t_ns = 0

    def reset(self) -> None:
        self._distances[:] = np.nan
        self._confidence[:] = 0.0
        self._updated_ns[:] = 0
        self._observed[:] = False
        self._last_t_ns = 0

    @property
    def last_t_ns(self) -> int:
        return self._last_t_ns

    def _aged(self, t_ns: int) -> tuple[np.ndarray, np.ndarray]:
        """Confidence and observed-flag decayed forward to ``t_ns``.

        Confidence is multiplied by ``age_decay_per_s ** age``, so the config
        value reads exactly as its comment says: the fraction remaining after
        one second. Past ``max_age_s`` the bin is dropped outright rather than
        left with a small non-zero confidence that would keep its range alive.
        """
        age_s = np.maximum(0.0, (t_ns - self._updated_ns) / NS_PER_S)
        decay = np.power(np.float64(self.occupancy.age_decay_per_s), age_s)
        confidence = (self._confidence * decay).astype(np.float32)
        alive = self._observed & (age_s <= self.occupancy.max_age_s)
        confidence = np.where(alive, confidence, 0.0).astype(np.float32)
        return confidence, alive

    def update(self, scan: ObstacleScan) -> None:
        """Fold one scan into the map."""
        if len(scan) != self.n_bins:
            raise ValueError(
                f"scan has {len(scan)} bins but the map has {self.n_bins}; "
                "obstacles.n_bins and occupancy.n_bins must agree"
            )
        t_ns = int(scan.t_ns)
        aged_confidence, alive = self._aged(t_ns)
        aged_distance = np.where(alive, self._distances, np.nan).astype(np.float32)

        fresh = ~np.asarray(scan.unknown, dtype=bool)
        new_distance = np.asarray(scan.distances, dtype=np.float32)
        new_confidence = np.asarray(scan.confidence, dtype=np.float32)

        # Nearer of the two where both exist; whichever exists otherwise.
        # np.fmin ignores NaN, which is exactly the "whichever exists" case.
        merged_distance = np.where(
            fresh, np.fmin(new_distance, aged_distance), aged_distance
        ).astype(np.float32)
        merged_confidence = np.where(
            fresh, np.maximum(new_confidence, aged_confidence), aged_confidence
        ).astype(np.float32)

        observed = alive | fresh
        # A bin with no range is not observed, whatever the flags say.
        observed &= np.isfinite(merged_distance)

        self._distances = np.where(observed, merged_distance, np.nan).astype(np.float32)
        self._confidence = np.where(observed, merged_confidence, 0.0).astype(np.float32)
        self._updated_ns = np.where(fresh, t_ns, self._updated_ns).astype(np.int64)
        self._observed = observed
        self._last_t_ns = t_ns

    def snapshot(self, t_ns: int | None = None, speed_mps: float = 0.0) -> OccupancySnapshot:
        """Current belief, aged to ``t_ns``, with danger evaluated at ``speed_mps``.

        Aging happens here as well as in ``update`` so that a planner ticking
        faster than the camera still sees stale bins expire on time rather
        than at the next frame.
        """
        t_ns = self._last_t_ns if t_ns is None else int(t_ns)
        confidence, alive = self._aged(t_ns)

        known = alive & np.isfinite(self._distances)
        known &= confidence >= self.occupancy.min_confidence
        unknown = ~known

        distances = np.where(known, self._distances, np.nan).astype(np.float32)
        confidence = np.where(known, confidence, 0.0).astype(np.float32)

        # NaN < x is False, so unknown bins are never flagged dangerous. That
        # is deliberate: they are impassable because they are unknown, and
        # labelling them "danger" too would hide which condition tripped.
        threshold = self.occupancy.danger_distance(speed_mps)
        with np.errstate(invalid="ignore"):
            danger = np.asarray(distances < threshold, dtype=bool)
        danger &= known

        return OccupancySnapshot(
            t_ns=t_ns,
            bearings=self.bearings.copy(),
            distances=distances,
            confidence=confidence,
            unknown=np.asarray(unknown, dtype=bool),
            danger=danger,
        )


def passable(snapshot: OccupancySnapshot, clearance_m: float) -> np.ndarray:
    """Bins that are known, not dangerous, and clear beyond ``clearance_m``.

    The single place the "unknown is impassable" rule is written down, so the
    planner cannot accidentally reimplement it as "unknown is fine".
    """
    with np.errstate(invalid="ignore"):
        clear = np.asarray(snapshot.distances > clearance_m, dtype=bool)
    return clear & ~snapshot.unknown & ~snapshot.danger
