"""Full CPU frames -> perception -> planning -> log -> same-pipeline replay."""
from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from pymavlink.dialects.v20 import ardupilotmega as mav

from config import Config, ConfigError, load_config_from_dict
from infra.flightlog import FlightLogReader
from infra.pipeline import RecordPipeline
from infra.replay import command_stream_bytes, replay_run
from sources.mavlink_client import MavlinkClient
from sources.stereo_uvc import CapturedFrame, decode_mjpg
from sources.telemetry import RangeReading, TelemetrySample
from sources.types import FcState
from tools.flight_report import report
from tools.run import main, record_frames
from tools.synthetic_stereo import planar_scene

NS = 1_000_000_000


@pytest.fixture
def small_cfg(cfg_from: Any) -> Config:
    return cfg_from({'camera': {'frame_width': 640, 'frame_height': 160},
                     'depth': {'sgbm': {'num_disparities': 32}}})


def frames(cfg: Config, count: int=6) -> Iterator[CapturedFrame]:
    scene = planar_scene(disparity_px=4, width=cfg.camera.eye_width, height=cfg.camera.eye_height)
    left, right = scene.left.copy(), scene.right.copy()
    left[50:80, 150:177] = (0, 0, 255)
    right[50:80, 146:173] = (0, 0, 255)
    ok, encoded = cv2.imencode('.jpg', np.concatenate([left, right], axis=1))
    assert ok
    packet = encoded.tobytes()
    for seq in range(count):
        yield CapturedFrame(decode_mjpg(packet, seq, NS + seq * NS // 20), packet, 0)


def guided(t_ns: int) -> FcState:
    return FcState(t_ns, 0., 0., 0., (0., 0., 0.), 3., 'GUIDED', True, True, {8: 1800})


def test_full_runner_replays_nontrivial_commands_and_telemetry_loss(
    small_cfg: Config, tmp_path: Path,
) -> None:
    def state_at(t_ns: int) -> FcState | None:
        return guided(t_ns) if t_ns < NS + NS // 4 else None

    path = record_frames(small_cfg, frames(small_cfg), root=tmp_path, run_id='full',
                         state_at=state_at, source='synthetic')
    reader = FlightLogReader(path)
    commands = tuple(reader.planner_commands())
    assert len(commands) == 6
    assert any(c.yaw_rate != 0 for c in commands)
    assert any('APPROACH' in c.reason or 'VFH' in c.reason or c.vx > 0 for c in commands)
    assert commands[-1].reason.startswith('IDLE')
    result = replay_run(path, RecordPipeline)
    assert command_stream_bytes(commands) == result.stream
    assert result.stream == replay_run(path, RecordPipeline).stream
    assert main(['replay', str(path)]) == 0
    summary = report(path)
    assert summary['runtime']['setpoints_transmitted'] == 0
    assert summary['runtime']['processed_frames'] == 6
    assert summary['unknown_bin_fraction'] > 0
    assert 'pipeline.tick' in summary['metrics']


def test_pipeline_reset_and_timestamp_rejection(small_cfg: Config) -> None:
    pipeline = RecordPipeline(small_cfg)
    sequence = list(frames(small_cfg))
    first = [pipeline.tick(f.bundle, guided(f.bundle.t_ns)) for f in sequence]
    with pytest.raises(ValueError, match='timestamps'):
        pipeline.tick(sequence[-1].bundle, None)
    pipeline.reset()
    second = [pipeline.tick(f.bundle, guided(f.bundle.t_ns)) for f in sequence]
    assert command_stream_bytes(first) == command_stream_bytes(second)


def test_no_live_nominal_geometry_and_no_transmit_option(small_cfg: Config, tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match='calibration'):
        record_frames(small_cfg, [], root=tmp_path)
    with pytest.raises(SystemExit):
        main(['record', '--transmit'])


def test_byte_limit_and_worker_fault_are_reported(small_cfg: Config, tmp_path: Path) -> None:
    raw = small_cfg.raw.copy()
    raw['runtime'] = dict(raw['runtime'], max_run_bytes=1)
    limited = load_config_from_dict(raw)
    path = record_frames(limited, frames(limited), root=tmp_path,
                         run_id='limit', source='synthetic')
    assert report(path)['runtime']['status'] == 'frame_byte_limit'
    assert list(FlightLogReader(path).frames()) == []

    def broken_source() -> Iterator[CapturedFrame]:
        yield next(frames(small_cfg))
        raise OSError('camera disappeared')

    with pytest.raises(OSError, match='disappeared'):
        record_frames(small_cfg, broken_source(), root=tmp_path, run_id='fault', source='synthetic')
    summary = report(tmp_path / 'fault')
    assert summary['runtime']['status'] == 'fault'
    assert 'disappeared' in summary['runtime']['error']
    assert FlightLogReader(tmp_path / 'fault').meta['counts']['frames'] == 1


def test_sensor_freshness_does_not_follow_other_traffic(cfg: Config) -> None:
    client = MavlinkClient(cfg)
    client.handle(mav.MAVLink_distance_sensor_message(0, 10, 1200, 250, 0, 1, 25, 0), NS)
    client.handle(mav.MAVLink_attitude_message(0, 0, 0, 0, 0, 0, 0), 2 * NS)
    snapshot = client.telemetry_at(2 * NS)
    assert snapshot.age_s('DISTANCE_SENSOR', 2 * NS) == 1
    assert snapshot.range_at(1, 25, 2 * NS, cfg.runtime.range_timeout_s) is None
    assert snapshot.paired_state(2 * NS, cfg).agl_m is None
    assert not snapshot.paired_state(2 * NS, cfg).armed
    assert client.telemetry_at(NS - 1) is None
    historic = client.telemetry_at(NS)
    assert historic.range_at(1, 25, NS, cfg.runtime.range_timeout_s).distance_m == 2.5
    snapshot.received.clear()
    assert client.telemetry_at(2 * NS).age_s('ATTITUDE', 2 * NS) == 0


def test_same_orientation_sensors_are_independent(cfg: Config) -> None:
    client = MavlinkClient(cfg)
    for sensor_id, t in [(1, NS), (2, 2*NS)]:
        client.handle(mav.MAVLink_distance_sensor_message(0, 10, 1200, 250, 0, sensor_id, 0, 0), t)
    sample = client.telemetry_at(2*NS)
    assert sample.range_at(1, 0, 2*NS, cfg.runtime.range_timeout_s) is None
    assert sample.range_at(2, 0, 2*NS, cfg.runtime.range_timeout_s).distance_m == 2.5


def test_stale_heartbeat_cannot_keep_authority(cfg: Config) -> None:
    received = {name: 8*NS for name in
                ('ATTITUDE', 'LOCAL_POSITION_NED', 'RC_CHANNELS', 'SYS_STATUS')}
    received['HEARTBEAT'] = NS
    sample = TelemetrySample(guided(8*NS), received, {(1, 25): RangeReading(1, 25, 8*NS, 2.)})
    assert sample.age_s('ATTITUDE', 8*NS) == 0
    assert not sample.paired_state(8*NS, cfg).armed


def test_moving_body_does_not_reuse_old_bearings(small_cfg: Config, tmp_path: Path) -> None:
    index = 0

    def state_at(t_ns: int) -> FcState | None:
        nonlocal index
        index += 1
        return replace(guided(t_ns), yaw=index / 10)

    path = record_frames(small_cfg, frames(small_cfg), root=tmp_path,
                         state_at=state_at, source='synthetic')
    assert not any(t['extra']['occupancy_history_retained']
                   for t in FlightLogReader(path).planner_ticks())


def test_processing_failure_closes_log(small_cfg: Config, tmp_path: Path) -> None:
    class BrokenPipeline(RecordPipeline):
        def tick(self, bundle: Any, state: Any) -> None:
            raise RuntimeError('depth failure')

    with pytest.raises(RuntimeError, match='depth failure'):
        record_frames(small_cfg, frames(small_cfg), root=tmp_path, run_id='broken',
                      source='synthetic', pipeline_factory=BrokenPipeline)
    status = json.loads((tmp_path / 'broken' / 'runtime.json').read_text())
    assert status['setpoints_transmitted'] == 0
    assert status['status'] == 'fault'


def test_calibration_is_portable_and_replay_uses_its_snapshot(
    small_cfg: Config, tmp_path: Path,
) -> None:
    width, height = small_cfg.camera.eye_width, small_cfg.camera.eye_height
    q = small_cfg.camera.nominal_q()
    f, cx, cy = q[2, 3], -q[0, 3], -q[1, 3]
    p1 = np.array([[f, 0, cx, 0], [0, f, cy, 0], [0, 0, 1, 0]])
    p2 = p1.copy()
    p2[0, 3] = -f * small_cfg.camera.baseline_m
    yy, xx = np.indices((height, width), dtype=np.float32)
    original = tmp_path / 'original.npz'
    np.savez(original, Q=q, R1=np.eye(3), P1=p1, P2=p2,
             image_size=[width, height], schema_version=1,
             left_x=xx, left_y=yy, right_x=xx, right_y=yy,
             roi1=[0, 0, width, height], roi2=[0, 0, width, height])
    raw = dict(small_cfg.raw, camera=dict(small_cfg.raw['camera'], calibration_npz=str(original)))
    cfg = load_config_from_dict(raw)
    path = record_frames(cfg, frames(cfg), root=tmp_path, run_id='portable', state_at=guided)
    original.unlink()
    reader = FlightLogReader(path)
    assert reader.meta['calibration_sha256']
    assert (path / 'calibration.npz').is_file()
    replayed = replay_run(path, RecordPipeline).stream
    assert replayed == command_stream_bytes(reader.planner_commands())
    from infra.flightlog import FlightLogError

    (path / 'calibration.npz').write_bytes(b'tampered')
    with pytest.raises(FlightLogError, match='hash mismatch'):
        replay_run(path, RecordPipeline)
