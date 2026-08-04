"""Stage timing and percentile rollups."""

from __future__ import annotations

import threading

import pytest

from config import Config
from infra.metrics import Metrics, StageStats, nearest_rank


def test_nearest_rank_returns_an_actual_sample() -> None:
    samples = list(range(1, 101))  # 1..100
    assert nearest_rank(samples, 0.0) == 1
    assert nearest_rank(samples, 50.0) == 50
    assert nearest_rank(samples, 99.0) == 99
    assert nearest_rank(samples, 100.0) == 100
    for pct in (13.0, 37.5, 82.1):
        assert nearest_rank(samples, pct) in samples


def test_nearest_rank_on_a_single_sample() -> None:
    assert nearest_rank([42], 50.0) == 42
    assert nearest_rank([42], 99.0) == 42
    assert nearest_rank([42], 0.0) == 42


def test_nearest_rank_never_interpolates() -> None:
    # With two samples an interpolating p50 would return 15; nearest-rank must
    # return a value that was actually measured.
    assert nearest_rank([10, 20], 50.0) == 10
    assert nearest_rank([10, 20], 51.0) == 20


def test_nearest_rank_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        nearest_rank([], 50.0)
    with pytest.raises(ValueError, match=r"within \[0, 100\]"):
        nearest_rank([1], 101.0)
    with pytest.raises(ValueError, match=r"within \[0, 100\]"):
        nearest_rank([1], -1.0)


def test_record_and_rollup() -> None:
    metrics = Metrics(window=100)
    for value in (5, 1, 3, 2, 4):
        metrics.record("depth", value)
    stats = metrics.rollup()["depth"]
    assert isinstance(stats, StageStats)
    assert stats.count == 5
    assert stats.min_ns == 1
    assert stats.max_ns == 5
    assert stats.p50_ns == 3
    assert stats.p99_ns == 5
    assert stats.mean_ns == pytest.approx(3.0)


def test_millisecond_properties_convert_from_ns() -> None:
    metrics = Metrics()
    metrics.record("stage", 2_500_000)
    stats = metrics.rollup()["stage"]
    assert stats.p50_ms == pytest.approx(2.5)
    assert stats.max_ms == pytest.approx(2.5)
    assert stats.mean_ms == pytest.approx(2.5)


def test_stage_context_manager_times_the_block() -> None:
    metrics = Metrics()
    with metrics.stage("work"):
        sum(range(10_000))
    samples = metrics.samples("work")
    assert len(samples) == 1
    assert samples[0] > 0


def test_stage_records_even_when_the_block_raises() -> None:
    metrics = Metrics()
    with pytest.raises(RuntimeError, match="boom"):
        with metrics.stage("failing"):
            raise RuntimeError("boom")
    # A stage that fails slowly must still show up, or a latency regression
    # hides behind an exception.
    assert len(metrics.samples("failing")) == 1


def test_nested_stages_are_independent() -> None:
    metrics = Metrics()
    with metrics.stage("outer"):
        with metrics.stage("inner"):
            pass
    assert metrics.stage_names() == ["inner", "outer"]
    assert metrics.rollup()["outer"].p50_ns >= metrics.rollup()["inner"].p50_ns


def test_window_bounds_memory() -> None:
    metrics = Metrics(window=8)
    for value in range(100):
        metrics.record("stage", value)
    samples = metrics.samples("stage")
    assert len(samples) == 8
    # The ring keeps the most recent samples.
    assert samples == list(range(92, 100))
    assert metrics.rollup()["stage"].count == 8


def test_window_must_be_positive() -> None:
    with pytest.raises(ValueError, match="window must be >= 1"):
        Metrics(window=0)


def test_negative_durations_are_rejected() -> None:
    metrics = Metrics()
    with pytest.raises(ValueError, match="elapsed_ns must be >= 0"):
        metrics.record("stage", -1)


def test_unrecorded_stages_are_absent_not_zero() -> None:
    metrics = Metrics()
    metrics.record("seen", 10)
    rollup = metrics.rollup()
    assert "seen" in rollup
    assert "never_run" not in rollup
    assert metrics.samples("never_run") == []


def test_rollup_is_ordered_by_stage_name() -> None:
    metrics = Metrics()
    for name in ("zebra", "alpha", "middle"):
        metrics.record(name, 1)
    assert list(metrics.rollup()) == ["alpha", "middle", "zebra"]


def test_reset_clears_everything() -> None:
    metrics = Metrics()
    metrics.record("stage", 1)
    metrics.reset()
    assert metrics.rollup() == {}
    assert metrics.stage_names() == []


def test_to_dict_is_json_shaped() -> None:
    import json

    metrics = Metrics()
    metrics.record("depth", 1_000_000)
    payload = json.loads(json.dumps(metrics.to_dict()))
    assert payload["depth"]["count"] == 1
    assert payload["depth"]["p50_ns"] == 1_000_000
    assert payload["depth"]["name"] == "depth"


def test_from_config_uses_the_configured_window(cfg: Config) -> None:
    assert Metrics.from_config(cfg).window == cfg.metrics.window


def test_concurrent_recording_loses_nothing() -> None:
    # The capture, perception, and offboard threads all share one Metrics.
    metrics = Metrics(window=10_000)
    per_thread = 500
    threads = [
        threading.Thread(target=lambda: [metrics.record("shared", 1) for _ in range(per_thread)])
        for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert metrics.rollup()["shared"].count == 8 * per_thread
