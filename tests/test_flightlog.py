"""Run-directory writing and reading."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from config import Config
from infra.flightlog import (
    CONFIG_JSON_NAME,
    CONFIG_YAML_NAME,
    FRAMES_BLOB_NAME,
    META_NAME,
    FlightLog,
    FlightLogError,
    FlightLogReader,
    canonical_json,
    decode_fc_state,
    decode_occupancy,
    decode_planner_command,
    encode_fc_state,
    encode_occupancy,
    encode_planner_command,
    git_revision,
    new_run_id,
)
from infra.metrics import Metrics
from sources.types import OccupancySnapshot, PlannerCommand
from tests.synthetic import encode_mjpg, make_fc_state, make_side_by_side, write_synthetic_run


def test_creates_the_expected_layout(tmp_path: Path, cfg: Config) -> None:
    with FlightLog.create(cfg, root=tmp_path, run_id="run") as log:
        log.write_frame(0, 100, b"\xff\xd8fake\xff\xd9")
    run = tmp_path / "run"
    for name in (META_NAME, CONFIG_YAML_NAME, CONFIG_JSON_NAME, FRAMES_BLOB_NAME):
        assert (run / name).is_file(), name


def test_refuses_to_overwrite_an_existing_run(tmp_path: Path, cfg: Config) -> None:
    FlightLog.create(cfg, root=tmp_path, run_id="run").close()
    with pytest.raises(FileExistsError):
        FlightLog.create(cfg, root=tmp_path, run_id="run")


def test_config_snapshot_reloads_to_an_equal_config(tmp_path: Path, cfg: Config) -> None:
    FlightLog.create(cfg, root=tmp_path, run_id="run").close()
    reloaded = FlightLogReader(tmp_path / "run").config()
    assert reloaded == cfg


def test_resolved_config_json_is_written_too(tmp_path: Path, cfg: Config) -> None:
    FlightLog.create(cfg, root=tmp_path, run_id="run").close()
    payload = json.loads((tmp_path / "run" / CONFIG_JSON_NAME).read_text())
    # The resolved view is radians and carries no source_path/raw bookkeeping.
    assert payload["obstacles"]["fov_rad"] == pytest.approx(cfg.obstacles.fov_rad)
    assert "raw" not in payload
    assert "source_path" not in payload


def test_frames_round_trip_byte_for_byte(tmp_path: Path, cfg: Config) -> None:
    payloads = [encode_mjpg(make_side_by_side(seq)) for seq in range(5)]
    with FlightLog.create(cfg, root=tmp_path, run_id="run") as log:
        for seq, payload in enumerate(payloads):
            log.write_frame(seq, 1000 + seq, payload)

    frames = list(FlightLogReader(tmp_path / "run").frames())
    assert [f.mjpg for f in frames] == payloads
    assert [f.seq for f in frames] == [0, 1, 2, 3, 4]
    assert [f.t_ns for f in frames] == [1000, 1001, 1002, 1003, 1004]


def test_frame_offsets_are_contiguous(tmp_path: Path, cfg: Config) -> None:
    payloads = [b"a" * 10, b"b" * 25, b"c" * 3]
    with FlightLog.create(cfg, root=tmp_path, run_id="run") as log:
        for seq, payload in enumerate(payloads):
            log.write_frame(seq, seq, payload)
    frames = list(FlightLogReader(tmp_path / "run").frames())
    assert [(f.offset, f.length) for f in frames] == [(0, 10), (10, 25), (35, 3)]


def test_logged_frames_are_not_re_encoded(tmp_path: Path, cfg: Config) -> None:
    # The blob must be the concatenation of exactly what was handed in.
    payloads = [encode_mjpg(make_side_by_side(seq)) for seq in range(3)]
    with FlightLog.create(cfg, root=tmp_path, run_id="run") as log:
        for seq, payload in enumerate(payloads):
            log.write_frame(seq, seq, payload)
    blob = (tmp_path / "run" / FRAMES_BLOB_NAME).read_bytes()
    assert blob == b"".join(payloads)


def test_truncated_frame_blob_is_reported(tmp_path: Path, cfg: Config) -> None:
    with FlightLog.create(cfg, root=tmp_path, run_id="run") as log:
        log.write_frame(0, 0, b"x" * 100)
    blob = tmp_path / "run" / FRAMES_BLOB_NAME
    blob.write_bytes(b"x" * 40)
    with pytest.raises(FlightLogError, match="truncated"):
        list(FlightLogReader(tmp_path / "run").frames())


def test_write_frames_false_skips_the_blob(tmp_path: Path, cfg_from) -> None:
    cfg = cfg_from({"flightlog": {"write_frames": False}})
    with FlightLog.create(cfg, root=tmp_path, run_id="run") as log:
        log.write_frame(0, 0, b"ignored")
    assert not (tmp_path / "run" / FRAMES_BLOB_NAME).exists()
    assert list(FlightLogReader(tmp_path / "run").frames()) == []


def test_telemetry_round_trips(tmp_path: Path, cfg: Config) -> None:
    states = [make_fc_state(i, 1000 + i) for i in range(10)]
    with FlightLog.create(cfg, root=tmp_path, run_id="run") as log:
        for state in states:
            log.write_telemetry(state)
    assert list(FlightLogReader(tmp_path / "run").telemetry()) == states


def test_fc_state_encoding_preserves_awkward_fields() -> None:
    state = make_fc_state(7, 42)  # index 7 -> agl_m is None
    assert state.agl_m is None
    decoded = decode_fc_state(json.loads(canonical_json(encode_fc_state(state))))
    assert decoded == state
    # rc keys survive the trip through JSON string keys.
    assert decoded.rc == {5: 1507, 6: 1000}
    assert all(isinstance(k, int) for k in decoded.rc)
    # vel_ned comes back a tuple, not a list.
    assert isinstance(decoded.vel_ned, tuple)


def test_planner_ticks_round_trip_with_their_snapshot(tmp_path: Path, cfg: Config) -> None:
    snapshot = OccupancySnapshot(
        t_ns=500,
        bearings=np.linspace(-0.5, 0.5, 4),
        distances=np.array([1.0, 2.0, np.inf, 4.0], dtype=np.float32),
        confidence=np.array([0.1, 0.2, 0.0, 0.4], dtype=np.float32),
        unknown=np.array([False, False, True, False]),
        danger=np.array([True, False, False, False]),
    )
    command = PlannerCommand(
        vx=1.5, vy=0.0, vz=0.0, yaw_rate=-0.2, reason="gap at +12deg", t_ns=500
    )
    with FlightLog.create(cfg, root=tmp_path, run_id="run") as log:
        log.write_planner_tick(500, command, snapshot, extra={"state": "APPROACH"})

    ticks = list(FlightLogReader(tmp_path / "run").planner_ticks())
    assert len(ticks) == 1
    assert ticks[0]["extra"] == {"state": "APPROACH"}
    assert decode_planner_command(ticks[0]["command"]) == command

    restored = decode_occupancy(ticks[0]["snapshot"])
    assert np.allclose(restored.bearings, snapshot.bearings)
    assert np.array_equal(restored.unknown, snapshot.unknown)
    assert np.array_equal(restored.danger, snapshot.danger)
    # Unknown bins carry inf range; that must survive, not become a number.
    assert np.isinf(restored.distances[2])


def test_planner_commands_shortcut(tmp_path: Path, cfg: Config) -> None:
    commands = [
        PlannerCommand(vx=float(i), vy=0.0, vz=0.0, yaw_rate=0.0, reason=f"r{i}", t_ns=i)
        for i in range(4)
    ]
    with FlightLog.create(cfg, root=tmp_path, run_id="run") as log:
        for command in commands:
            log.write_planner_tick(command.t_ns, command)
    assert list(FlightLogReader(tmp_path / "run").planner_commands()) == commands


def test_meta_records_counts_and_git(tmp_path: Path, cfg: Config) -> None:
    with FlightLog.create(cfg, root=tmp_path, run_id="run") as log:
        log.write_frame(0, 0, b"f")
        log.write_telemetry(make_fc_state(0, 0))
        log.write_telemetry(make_fc_state(1, 1))
        log.write_planner_tick(
            0, PlannerCommand(vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0, reason="x", t_ns=0)
        )
    meta = FlightLogReader(tmp_path / "run").meta
    assert meta["counts"] == {"frames": 1, "telemetry": 2, "planner": 1}
    assert meta["run_id"] == "run"
    assert meta["t_end_ns"] >= meta["t_start_ns"]
    assert "git" in meta


def test_metrics_are_written(tmp_path: Path, cfg: Config) -> None:
    metrics = Metrics()
    metrics.record("depth", 1_000_000)
    with FlightLog.create(cfg, root=tmp_path, run_id="run") as log:
        log.write_metrics(metrics)
    assert FlightLogReader(tmp_path / "run").metrics()["depth"]["count"] == 1


def test_writing_after_close_is_refused(tmp_path: Path, cfg: Config) -> None:
    log = FlightLog.create(cfg, root=tmp_path, run_id="run")
    log.close()
    with pytest.raises(FlightLogError, match="closed"):
        log.write_frame(0, 0, b"x")


def test_close_is_idempotent(tmp_path: Path, cfg: Config) -> None:
    log = FlightLog.create(cfg, root=tmp_path, run_id="run")
    log.close()
    log.close()


def test_git_revision_finds_this_repo() -> None:
    revision = git_revision()
    assert revision["error"] is None
    assert revision["commit"] and len(revision["commit"]) == 40
    assert isinstance(revision["dirty"], bool)


def test_git_revision_outside_a_repo_reports_why(tmp_path: Path) -> None:
    revision = git_revision(tmp_path)
    assert revision["commit"] is None
    assert revision["error"]


def test_reader_rejects_a_non_run_directory(tmp_path: Path) -> None:
    with pytest.raises(FlightLogError, match="not found"):
        FlightLogReader(tmp_path / "missing")
    (tmp_path / "empty").mkdir()
    with pytest.raises(FlightLogError, match="not a run directory"):
        FlightLogReader(tmp_path / "empty")


def test_canonical_json_is_key_order_independent() -> None:
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})
    assert canonical_json({"a": 1}) == '{"a":1}'


def test_encode_planner_command_is_stable() -> None:
    command = PlannerCommand(vx=0.1, vy=-0.2, vz=0.0, yaw_rate=0.3, reason="why", t_ns=9)
    assert canonical_json(encode_planner_command(command)) == canonical_json(
        encode_planner_command(command)
    )


def test_new_run_id_is_sortable_and_unique_per_pid() -> None:
    assert new_run_id(pid=7).endswith("-7")
    assert new_run_id(pid=7) != new_run_id(pid=8)
    assert new_run_id(pid=1)[:4].isdigit()


def test_synthetic_run_is_readable(tmp_path: Path, cfg: Config) -> None:
    run = write_synthetic_run(tmp_path, cfg, n_frames=6, n_telemetry=15)
    reader = FlightLogReader(run)
    assert len(list(reader.frames())) == 6
    assert len(list(reader.telemetry())) == 15
    assert reader.config() == cfg


def test_occupancy_encoding_preserves_bin_semantics() -> None:
    # Non-finite distances are how "no measurement" is carried, so they have to
    # survive the log. Python's JSON extension round-trips NaN and Infinity.
    snapshot = OccupancySnapshot(
        t_ns=1,
        bearings=np.array([0.0, 0.1]),
        distances=np.array([np.nan, np.inf], dtype=np.float32),
        confidence=np.array([0.0, 0.0], dtype=np.float32),
        unknown=np.array([True, True]),
        danger=np.array([False, False]),
    )
    restored = decode_occupancy(json.loads(canonical_json(encode_occupancy(snapshot))))
    assert bool(restored.unknown[0]) is True
    assert restored.unknown.dtype == bool
    assert restored.distances.dtype == np.float32
    assert np.isnan(restored.distances[0])
    assert np.isinf(restored.distances[1])
