"""Temporal occupancy map: merging, decay, and the unknown/danger split."""

from __future__ import annotations

import numpy as np
import pytest

from config import Config
from perception.obstacles import ObstacleScan, bin_centres
from world.occupancy import NS_PER_S, OccupancyMap, passable

N_BINS = 8


@pytest.fixture
def cfg8(cfg_from) -> Config:
    """Eight bins, so a test can write out an expected array by hand."""
    return cfg_from({"obstacles": {"n_bins": N_BINS}, "occupancy": {"n_bins": N_BINS}})


def make_scan(
    cfg: Config,
    distances: list[float],
    t_ns: int,
    confidence: float = 1.0,
) -> ObstacleScan:
    dist = np.array(distances, dtype=np.float32)
    unknown = ~np.isfinite(dist)
    conf = np.where(unknown, 0.0, confidence).astype(np.float32)
    return ObstacleScan(
        t_ns=t_ns,
        bearings=bin_centres(len(distances), cfg.occupancy.fov_rad),
        distances=dist,
        confidence=conf,
        unknown=unknown,
        counts=np.where(unknown, 0, 500).astype(np.int32),
    )


def all_bins(value: float) -> list[float]:
    return [value] * N_BINS


# --------------------------------------------------------------------------
# Basics
# --------------------------------------------------------------------------


def test_a_fresh_map_is_entirely_unknown(cfg8: Config) -> None:
    snapshot = OccupancyMap(cfg8).snapshot(t_ns=0)
    assert snapshot.unknown.all()
    assert np.all(np.isnan(snapshot.distances))
    assert not snapshot.danger.any()


def test_update_then_snapshot_reports_the_scan(cfg8: Config) -> None:
    occupancy = OccupancyMap(cfg8)
    occupancy.update(make_scan(cfg8, all_bins(5.0), t_ns=1_000_000_000))
    snapshot = occupancy.snapshot(t_ns=1_000_000_000)
    assert not snapshot.unknown.any()
    assert np.allclose(snapshot.distances, 5.0)
    assert snapshot.t_ns == 1_000_000_000


def test_snapshot_bearings_match_the_bins(cfg8: Config) -> None:
    snapshot = OccupancyMap(cfg8).snapshot(t_ns=0)
    assert np.allclose(snapshot.bearings, bin_centres(N_BINS, cfg8.occupancy.fov_rad))


def test_bin_count_mismatch_is_rejected(cfg8: Config) -> None:
    occupancy = OccupancyMap(cfg8)
    wrong = make_scan(cfg8, [1.0, 2.0, 3.0], t_ns=0)
    with pytest.raises(ValueError, match="bins"):
        occupancy.update(wrong)


def test_reset_clears_the_map(cfg8: Config) -> None:
    occupancy = OccupancyMap(cfg8)
    occupancy.update(make_scan(cfg8, all_bins(4.0), t_ns=1_000_000_000))
    occupancy.reset()
    assert occupancy.snapshot(t_ns=1_000_000_000).unknown.all()


# --------------------------------------------------------------------------
# The rule that matters: aged out means unknown, never clear
# --------------------------------------------------------------------------


def test_a_bin_that_ages_out_becomes_unknown_not_clear(cfg8: Config) -> None:
    occupancy = OccupancyMap(cfg8)
    t0 = 1_000_000_000
    occupancy.update(make_scan(cfg8, all_bins(3.0), t_ns=t0))

    expired = t0 + int((cfg8.occupancy.max_age_s + 0.5) * NS_PER_S)
    snapshot = occupancy.snapshot(t_ns=expired)

    assert snapshot.unknown.all()
    assert np.all(np.isnan(snapshot.distances))
    # The critical assertion: forgetting is not observing empty space.
    assert not passable(snapshot, clearance_m=0.1).any()


def test_an_aged_out_bin_is_not_reported_as_a_long_range(cfg8: Config) -> None:
    occupancy = OccupancyMap(cfg8)
    t0 = 0
    occupancy.update(make_scan(cfg8, all_bins(3.0), t_ns=t0))
    expired = int((cfg8.occupancy.max_age_s + 1.0) * NS_PER_S)
    distances = occupancy.snapshot(t_ns=expired).distances
    assert not np.any(distances > 100.0)
    assert not np.any(distances == 0.0)


def test_confidence_decays_with_age(cfg8: Config) -> None:
    occupancy = OccupancyMap(cfg8)
    occupancy.update(make_scan(cfg8, all_bins(3.0), t_ns=0))
    fresh = occupancy.snapshot(t_ns=0).confidence.copy()
    later = occupancy.snapshot(t_ns=int(0.5 * NS_PER_S)).confidence
    assert np.all(later < fresh)
    assert np.all(later > 0.0)


def test_decay_matches_the_configured_rate(cfg8: Config) -> None:
    # age_decay_per_s is the fraction remaining after one second.
    occupancy = OccupancyMap(cfg8)
    occupancy.update(make_scan(cfg8, all_bins(3.0), t_ns=0, confidence=1.0))
    after_one_second = occupancy.snapshot(t_ns=int(NS_PER_S)).confidence
    assert after_one_second[0] == pytest.approx(cfg8.occupancy.age_decay_per_s, rel=1e-4)


def test_low_confidence_reads_as_unknown(cfg_from) -> None:
    cfg = cfg_from(
        {
            "obstacles": {"n_bins": N_BINS},
            "occupancy": {"n_bins": N_BINS, "min_confidence": 0.9, "age_decay_per_s": 0.1},
        }
    )
    occupancy = OccupancyMap(cfg)
    occupancy.update(make_scan(cfg, all_bins(3.0), t_ns=0, confidence=1.0))
    assert not occupancy.snapshot(t_ns=0).unknown.any()
    # After a second, confidence has fallen to 0.1, under the 0.9 floor.
    assert occupancy.snapshot(t_ns=int(NS_PER_S)).unknown.all()


def test_a_scan_that_reports_unknown_does_not_erase_memory(cfg8: Config) -> None:
    # A frame where the matcher found nothing is not evidence of empty space.
    occupancy = OccupancyMap(cfg8)
    occupancy.update(make_scan(cfg8, all_bins(3.0), t_ns=0))
    occupancy.update(make_scan(cfg8, all_bins(float("nan")), t_ns=int(0.1 * NS_PER_S)))
    snapshot = occupancy.snapshot(t_ns=int(0.1 * NS_PER_S))
    assert not snapshot.unknown.any()
    assert np.allclose(snapshot.distances, 3.0)


# --------------------------------------------------------------------------
# Merging
# --------------------------------------------------------------------------


def test_merge_keeps_the_nearer_range(cfg8: Config) -> None:
    occupancy = OccupancyMap(cfg8)
    occupancy.update(make_scan(cfg8, all_bins(6.0), t_ns=0))
    occupancy.update(make_scan(cfg8, all_bins(2.0), t_ns=int(0.1 * NS_PER_S)))
    assert np.allclose(occupancy.snapshot(t_ns=int(0.1 * NS_PER_S)).distances, 2.0)


def test_merge_keeps_the_nearer_range_even_when_the_new_scan_is_further(cfg8: Config) -> None:
    # Pessimistic on purpose: if two recent observations disagree about how
    # close something is, the closer one is the one worth flying by.
    occupancy = OccupancyMap(cfg8)
    occupancy.update(make_scan(cfg8, all_bins(2.0), t_ns=0))
    occupancy.update(make_scan(cfg8, all_bins(6.0), t_ns=int(0.1 * NS_PER_S)))
    assert np.allclose(occupancy.snapshot(t_ns=int(0.1 * NS_PER_S)).distances, 2.0)


def test_a_stale_near_reading_does_expire(cfg8: Config) -> None:
    # The cost of pessimistic merging is bounded by max_age_s, not permanent.
    occupancy = OccupancyMap(cfg8)
    occupancy.update(make_scan(cfg8, all_bins(1.0), t_ns=0))
    later = int((cfg8.occupancy.max_age_s + 0.1) * NS_PER_S)
    occupancy.update(make_scan(cfg8, all_bins(8.0), t_ns=later))
    assert np.allclose(occupancy.snapshot(t_ns=later).distances, 8.0)


def test_a_new_observation_fills_a_previously_unknown_bin(cfg8: Config) -> None:
    occupancy = OccupancyMap(cfg8)
    partial = all_bins(float("nan"))
    partial[3] = 4.0
    occupancy.update(make_scan(cfg8, partial, t_ns=0))
    snapshot = occupancy.snapshot(t_ns=0)
    assert not snapshot.unknown[3]
    assert snapshot.unknown[0] and snapshot.unknown[7]
    assert snapshot.distances[3] == pytest.approx(4.0)


def test_bins_are_independent(cfg8: Config) -> None:
    occupancy = OccupancyMap(cfg8)
    distances = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    occupancy.update(make_scan(cfg8, distances, t_ns=0))
    assert np.allclose(occupancy.snapshot(t_ns=0).distances, distances)


# --------------------------------------------------------------------------
# Danger
# --------------------------------------------------------------------------


def test_danger_follows_speed(cfg8: Config) -> None:
    occupancy = OccupancyMap(cfg8)
    occupancy.update(make_scan(cfg8, all_bins(3.0), t_ns=0))
    assert not occupancy.snapshot(t_ns=0, speed_mps=0.0).danger.any()
    assert occupancy.snapshot(t_ns=0, speed_mps=5.0).danger.all()


def test_danger_threshold_matches_the_config_formula(cfg8: Config) -> None:
    speed = 3.0
    threshold = cfg8.occupancy.danger_distance(speed)
    occupancy = OccupancyMap(cfg8)
    distances = [threshold - 0.1, threshold + 0.1] + [50.0] * (N_BINS - 2)
    occupancy.update(make_scan(cfg8, distances, t_ns=0))
    danger = occupancy.snapshot(t_ns=0, speed_mps=speed).danger
    assert danger[0]
    assert not danger[1]


def test_unknown_bins_are_never_flagged_dangerous(cfg8: Config) -> None:
    # unknown and danger answer different questions. An unknown bin is
    # impassable because nothing was measured, not because something is close;
    # conflating them would hide which condition tripped.
    occupancy = OccupancyMap(cfg8)
    snapshot = occupancy.snapshot(t_ns=0, speed_mps=5.0)
    assert snapshot.unknown.all()
    assert not snapshot.danger.any()


def test_danger_and_unknown_are_both_impassable(cfg8: Config) -> None:
    occupancy = OccupancyMap(cfg8)
    distances = all_bins(float("nan"))
    distances[0] = 0.5  # very close: dangerous
    distances[1] = 50.0  # far: fine
    occupancy.update(make_scan(cfg8, distances, t_ns=0))
    snapshot = occupancy.snapshot(t_ns=0, speed_mps=3.0)

    allowed = passable(snapshot, clearance_m=2.5)
    assert not allowed[0], "a dangerous bin must not be passable"
    assert allowed[1], "a far, known bin must be passable"
    assert not allowed[2:].any(), "unknown bins must not be passable"


def test_passable_requires_clearance(cfg8: Config) -> None:
    occupancy = OccupancyMap(cfg8)
    occupancy.update(make_scan(cfg8, all_bins(3.0), t_ns=0))
    snapshot = occupancy.snapshot(t_ns=0, speed_mps=0.0)
    assert passable(snapshot, clearance_m=2.0).all()
    assert not passable(snapshot, clearance_m=4.0).any()


# --------------------------------------------------------------------------
# Determinism / no hidden clock
# --------------------------------------------------------------------------


def test_the_map_never_reads_a_clock(cfg8: Config) -> None:
    # Every result must come from the timestamps handed in, so replaying a log
    # reproduces the same state exactly.
    def run() -> np.ndarray:
        occupancy = OccupancyMap(cfg8)
        for step in range(5):
            occupancy.update(make_scan(cfg8, all_bins(3.0 + step), t_ns=step * 100_000_000))
        return occupancy.snapshot(t_ns=500_000_000, speed_mps=1.0).distances

    first = run()
    second = run()
    assert np.array_equal(first, second, equal_nan=True)


def test_snapshot_does_not_mutate_the_map(cfg8: Config) -> None:
    occupancy = OccupancyMap(cfg8)
    occupancy.update(make_scan(cfg8, all_bins(3.0), t_ns=0))
    first = occupancy.snapshot(t_ns=0).distances.copy()
    occupancy.snapshot(t_ns=int(NS_PER_S))
    assert np.array_equal(occupancy.snapshot(t_ns=0).distances, first, equal_nan=True)


def test_snapshot_bearings_are_a_copy(cfg8: Config) -> None:
    occupancy = OccupancyMap(cfg8)
    snapshot = occupancy.snapshot(t_ns=0)
    snapshot.bearings[0] = 99.0
    assert occupancy.snapshot(t_ns=0).bearings[0] != 99.0


def test_snapshot_defaults_to_the_last_update_time(cfg8: Config) -> None:
    occupancy = OccupancyMap(cfg8)
    occupancy.update(make_scan(cfg8, all_bins(3.0), t_ns=777))
    assert occupancy.snapshot().t_ns == 777
    assert occupancy.last_t_ns == 777


def test_snapshot_dtypes_match_the_contract(cfg8: Config) -> None:
    occupancy = OccupancyMap(cfg8)
    occupancy.update(make_scan(cfg8, all_bins(3.0), t_ns=0))
    snapshot = occupancy.snapshot(t_ns=0)
    assert snapshot.distances.dtype == np.float32
    assert snapshot.confidence.dtype == np.float32
    assert snapshot.unknown.dtype == bool
    assert snapshot.danger.dtype == bool
