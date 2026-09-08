"""Replay: re-run determinism and A/B config comparison."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from config import Config
from infra.flightlog import FlightLog, FlightLogReader
from infra.metrics import Metrics
from infra.replay import (
    ReplayError,
    TelemetryTrack,
    ab_compare,
    command_stream_bytes,
    decode_frame,
    diff_command_streams,
    replay_run,
    split_side_by_side,
)
from sources.types import FcState, PlannerCommand
from tests.synthetic import (
    FakePipeline,
    encode_mjpg,
    fake_pipeline_factory,
    make_fc_state,
    make_side_by_side,
    write_synthetic_run,
)


@pytest.fixture
def run_dir(tmp_path: Path, cfg: Config) -> Path:
    return write_synthetic_run(tmp_path, cfg, n_frames=12, n_telemetry=30)


# --------------------------------------------------------------------------
# Determinism -- the headline requirement for Package A.
# --------------------------------------------------------------------------


def test_replaying_twice_gives_a_byte_identical_command_stream(run_dir: Path) -> None:
    first = replay_run(run_dir, fake_pipeline_factory)
    second = replay_run(run_dir, fake_pipeline_factory)
    assert first.stream == second.stream
    assert len(first.stream) > 0
    assert first.commands == second.commands


def test_determinism_holds_across_a_fresh_reader(run_dir: Path, cfg: Config) -> None:
    # Same log, same config, separate reader objects and separate pipelines.
    a = replay_run(run_dir, fake_pipeline_factory, cfg)
    b = replay_run(run_dir, fake_pipeline_factory, cfg)
    assert a.stream == b.stream


def test_replay_resets_pipeline_state(run_dir: Path, cfg: Config) -> None:
    # One pipeline instance reused across two replays must still produce the
    # same stream, which only holds if replay_run calls reset().
    shared = FakePipeline(cfg=cfg)
    first = replay_run(run_dir, lambda _cfg: shared, cfg)
    second = replay_run(run_dir, lambda _cfg: shared, cfg)
    assert first.stream == second.stream


def test_a_stateful_pipeline_that_is_not_reset_would_diverge(run_dir: Path, cfg: Config) -> None:
    # Guards the guard: prove the determinism test above can actually fail.
    class NoResetPipeline(FakePipeline):
        def reset(self) -> None:  # deliberately does nothing
            pass

    shared = NoResetPipeline(cfg=cfg)
    first = replay_run(run_dir, lambda _cfg: shared, cfg)
    second = replay_run(run_dir, lambda _cfg: shared, cfg)
    assert first.stream != second.stream


# --------------------------------------------------------------------------
# Re-run mode
# --------------------------------------------------------------------------


def test_replay_defaults_to_the_logged_config(run_dir: Path, cfg: Config) -> None:
    result = replay_run(run_dir, fake_pipeline_factory)
    assert result.config_label == "logged"
    explicit = replay_run(run_dir, fake_pipeline_factory, cfg)
    assert result.stream == explicit.stream


def test_replay_counts_what_it_consumed(run_dir: Path) -> None:
    result = replay_run(run_dir, fake_pipeline_factory)
    assert result.frames_seen == 12
    assert result.telemetry_seen == 30
    # The fake drops every third command, so the stream is shorter than frames.
    assert len(result.commands) == 8


def test_replay_limit_truncates(run_dir: Path) -> None:
    result = replay_run(run_dir, fake_pipeline_factory, limit=4)
    assert result.frames_seen == 4
    assert len(result.commands) == 3


def test_replay_records_stage_metrics(run_dir: Path, cfg: Config) -> None:
    metrics = Metrics.from_config(cfg)
    result = replay_run(run_dir, fake_pipeline_factory, cfg, metrics=metrics)
    assert result.metrics["replay.tick"]["count"] == 12


def test_replay_feeds_the_pipeline_matching_telemetry(run_dir: Path, cfg: Config) -> None:
    captured: list[FcState | None] = []

    class Recorder(FakePipeline):
        def tick(self, bundle, state):  # type: ignore[no-untyped-def]
            captured.append(state)
            return super().tick(bundle, state)

    replay_run(run_dir, lambda c: Recorder(cfg=c), cfg)
    assert len(captured) == 12
    assert all(state is not None for state in captured)


def test_replay_hands_over_split_stereo_halves(run_dir: Path, cfg: Config) -> None:
    shapes: list[tuple[int, int]] = []

    class ShapeSpy(FakePipeline):
        def tick(self, bundle, state):  # type: ignore[no-untyped-def]
            shapes.append((bundle.left.shape[1], bundle.right.shape[1]))
            assert bundle.left.shape == bundle.right.shape
            return super().tick(bundle, state)

    replay_run(run_dir, lambda c: ShapeSpy(cfg=c), cfg)
    # 64 px side-by-side -> 32 px per eye.
    assert set(shapes) == {(32, 32)}


# --------------------------------------------------------------------------
# A/B mode
# --------------------------------------------------------------------------


def test_ab_with_the_same_config_is_identical(run_dir: Path, cfg: Config) -> None:
    result = ab_compare(run_dir, fake_pipeline_factory, cfg, cfg)
    assert result.identical
    assert result.diffs == ()
    assert result.first_divergence is None
    assert "identical" in result.summary()


def test_ab_with_different_configs_diverges(run_dir: Path, cfg: Config, cfg_from) -> None:
    slower = cfg_from({"behaviours": {"approach_speed_mps": 0.5}})
    assert slower.behaviours.approach_speed_mps != cfg.behaviours.approach_speed_mps

    result = ab_compare(run_dir, fake_pipeline_factory, cfg, slower, label_a="base", label_b="slow")
    assert not result.identical
    assert result.first_divergence == 0
    assert {d.field_name for d in result.diffs} == {"vx"}
    assert "base" in result.summary() and "slow" in result.summary()


def test_ab_isolates_the_two_runs(run_dir: Path, cfg: Config, cfg_from) -> None:
    # Running B must not be affected by having run A first: comparing cfg
    # against itself after a divergent pair still has to come out identical.
    other = cfg_from({"behaviours": {"approach_speed_mps": 0.5}})
    ab_compare(run_dir, fake_pipeline_factory, cfg, other)
    assert ab_compare(run_dir, fake_pipeline_factory, cfg, cfg).identical


def test_ab_reports_a_yaw_rate_change(run_dir: Path, cfg: Config, cfg_from) -> None:
    turnier = cfg_from({"local_planner": {"max_yaw_rate_dps": 30.0}})
    result = ab_compare(run_dir, fake_pipeline_factory, cfg, turnier)
    assert not result.identical
    assert {d.field_name for d in result.diffs} == {"yaw_rate"}


# --------------------------------------------------------------------------
# Stream diffing
# --------------------------------------------------------------------------


def _cmd(vx: float, t_ns: int = 0, reason: str = "r") -> PlannerCommand:
    return PlannerCommand(vx=vx, vy=0.0, vz=0.0, yaw_rate=0.0, reason=reason, t_ns=t_ns)


def test_diff_reports_each_differing_field() -> None:
    a = [_cmd(1.0, reason="left")]
    b = [_cmd(2.0, reason="right")]
    diffs = diff_command_streams(a, b)
    assert {d.field_name for d in diffs} == {"vx", "reason"}
    assert all(d.index == 0 for d in diffs)


def test_diff_reports_a_length_mismatch() -> None:
    diffs = diff_command_streams([_cmd(1.0), _cmd(1.0)], [_cmd(1.0)])
    assert diffs[-1].field_name == "<stream length>"
    assert (diffs[-1].a, diffs[-1].b) == (2, 1)


def test_diff_of_equal_streams_is_empty() -> None:
    assert diff_command_streams([_cmd(1.0)], [_cmd(1.0)]) == ()


def test_command_stream_bytes_is_stable_and_newline_terminated() -> None:
    commands = [_cmd(1.0, t_ns=1), _cmd(2.0, t_ns=2)]
    stream = command_stream_bytes(commands)
    assert stream == command_stream_bytes(commands)
    assert stream.endswith(b"\n")
    assert len(stream.splitlines()) == 2
    assert command_stream_bytes([]) == b""


def test_command_stream_distinguishes_only_the_reason() -> None:
    # reason is part of the decision record, so a changed reason is a changed
    # stream even when the velocities match.
    assert command_stream_bytes([_cmd(1.0, reason="gap")]) != command_stream_bytes(
        [_cmd(1.0, reason="stop")]
    )


# --------------------------------------------------------------------------
# Telemetry pairing and frame decoding
# --------------------------------------------------------------------------


def test_telemetry_track_holds_the_last_value() -> None:
    states = [make_fc_state(i, i * 100) for i in range(5)]  # t = 0,100,...,400
    track = TelemetryTrack(states)
    assert track.at(0) is states[0]
    assert track.at(150) is states[1]
    assert track.at(399) is states[3]
    assert track.at(10_000) is states[4]
    assert len(track) == 5


def test_telemetry_track_before_the_first_sample_is_none() -> None:
    track = TelemetryTrack([make_fc_state(0, 500)])
    assert track.at(499) is None
    assert track.at(500) is not None


def test_telemetry_track_sorts_unordered_input() -> None:
    states = [make_fc_state(1, 200), make_fc_state(0, 100)]
    track = TelemetryTrack(states)
    assert track.at(150).t_ns == 100


def test_empty_telemetry_track() -> None:
    track = TelemetryTrack([])
    assert len(track) == 0
    assert track.at(123) is None


def test_replay_tolerates_a_log_with_no_telemetry(tmp_path: Path, cfg: Config) -> None:
    with FlightLog.create(cfg, root=tmp_path, run_id="noteleme") as log:
        for seq in range(3):
            log.write_frame(seq, seq * 1000, encode_mjpg(make_side_by_side(seq)))
    result = replay_run(tmp_path / "noteleme", fake_pipeline_factory, cfg)
    assert result.telemetry_seen == 0
    assert result.frames_seen == 3


def test_split_side_by_side_halves_the_frame() -> None:
    frame = np.arange(2 * 8 * 3, dtype=np.uint8).reshape(2, 8, 3)
    left, right = split_side_by_side(frame)
    assert left.shape == right.shape == (2, 4, 3)
    assert np.array_equal(left, frame[:, :4])
    assert np.array_equal(right, frame[:, 4:])


def test_split_rejects_an_odd_width() -> None:
    with pytest.raises(ReplayError, match="width must be even"):
        split_side_by_side(np.zeros((2, 7, 3), dtype=np.uint8))


def test_decode_frame_rejects_garbage(tmp_path: Path, cfg: Config) -> None:
    with FlightLog.create(cfg, root=tmp_path, run_id="bad") as log:
        log.write_frame(0, 0, b"not a jpeg at all")
    record = next(iter(FlightLogReader(tmp_path / "bad").frames()))
    with pytest.raises(ReplayError, match="not decodable"):
        decode_frame(record)


def test_decode_frame_preserves_timestamps_and_seq(run_dir: Path) -> None:
    reader = FlightLogReader(run_dir)
    for record in reader.frames():
        bundle = decode_frame(record)
        assert bundle.t_ns == record.t_ns
        assert bundle.seq == record.seq
