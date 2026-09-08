"""Record-only autonomy runner, synthetic smoke run, and deterministic replay.

There is intentionally no actuator-output option. The command gate, watchdog,
calibration validation, GPU selection and measured flight envelope are dependencies.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import time
from collections.abc import Callable, Iterable, Iterator
from itertools import islice
from pathlib import Path

import cv2
import numpy as np

from config import Config, ConfigError, load_config
from infra.flightlog import FlightLog, FlightLogReader
from infra.metrics import Metrics
from infra.pipeline import RecordPipeline
from infra.replay import command_stream_bytes, replay_run
from sources.mavlink_client import MavlinkClient
from sources.stereo_uvc import CapturedFrame, StereoCapture, decode_mjpg, discover_devices
from sources.types import FcState
from tools.synthetic_stereo import planar_scene


def unavailable_state(t_ns: int) -> FcState:
    return FcState(t_ns, 0., 0., 0., (0., 0., 0.), None, 'UNKNOWN', False, False, {})


def record_frames(
    cfg: Config, frames: Iterable[CapturedFrame], *, root: Path | None = None,
    run_id: str | None = None, state_at: Callable[[int], FcState | None] | None = None,
    limit: int | None = None, source: str = 'live',
    pipeline_factory: Callable[..., RecordPipeline] = RecordPipeline,
) -> Path:
    if source not in ('live', 'synthetic'):
        raise ValueError('source must be live or synthetic')
    if source == 'live' and cfg.camera.calibration_npz is None:
        raise ConfigError('live perception requires calibration; use calib.capture for raw pairs')
    if not cfg.flightlog.write_frames:
        raise ConfigError('record-only runs require flightlog.write_frames=true for replay')
    if limit is not None and limit <= 0:
        raise ValueError('frame limit must be positive')
    metrics = Metrics.from_config(cfg)
    started = time.monotonic_ns()
    count = dropped = encoded_bytes = 0
    status, error = 'completed', None
    with FlightLog.create(cfg, root=root, run_id=run_id) as log:
        print(log.run_dir, flush=True)
        try:
            # Use the calibration copy in the log, so replay sees identical geometry.
            pipeline = pipeline_factory(FlightLogReader(log.run_dir).config(), log, metrics)
            stream = frames if limit is None else islice(frames, limit)
            for frame in stream:
                if (time.monotonic_ns() - started) / 1e9 >= cfg.runtime.max_duration_s:
                    status = 'duration_limit'
                    break
                if encoded_bytes + len(frame.mjpg) > cfg.runtime.max_run_bytes:
                    status = 'frame_byte_limit'
                    break
                t_ns = frame.bundle.t_ns
                state = state_at(t_ns) if state_at else None
                state = state if state is not None else unavailable_state(t_ns)
                log.write_frame(frame.bundle.seq, t_ns, frame.mjpg)
                log.write_telemetry(state)
                with metrics.stage('runner.process'):
                    pipeline.tick(frame.bundle, state)
                if source == 'live':
                    metrics.record('host_dequeue_to_proposal', time.monotonic_ns() - t_ns)
                count += 1
                dropped += frame.dropped_before
                encoded_bytes += len(frame.mjpg)
        except BaseException as exc:
            status = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'fault'
            error = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            log.write_metrics(metrics)
            summary = dict(build_id=os.environ.get('DRONE_BUILD_ID'),
                           python=platform.python_version(), opencv=cv2.__version__,
                           numpy=np.__version__, mode='record_only', source=source,
                           status=status, error=error,
                           processed_frames=count, dropped_frames=dropped,
                           encoded_bytes=encoded_bytes, setpoints_transmitted=0,
                           timestamp_source=('synthetic_monotonic' if source == 'synthetic'
                                             else 'host_dequeue_monotonic'),
                           elapsed_s=(time.monotonic_ns() - started) / 1e9)
            (log.run_dir / 'runtime.json').write_text(json.dumps(summary, indent=2) + '\n')
        return log.run_dir


def live_frames(source: StereoCapture) -> Iterator[CapturedFrame]:
    while True:
        yield source.read()


def synthetic_frames(cfg: Config, count: int | None = None) -> Iterator[CapturedFrame]:
    scene = planar_scene(height=cfg.camera.eye_height, width=cfg.camera.eye_width)
    ok, encoded = cv2.imencode('.jpg', np.concatenate([scene.left, scene.right], axis=1))
    if not ok:
        raise RuntimeError('synthetic JPEG encoding failed')
    packet = encoded.tobytes()
    for seq in range(cfg.runtime.synthetic_frames if count is None else count):
        t_ns = time.monotonic_ns()
        yield CapturedFrame(decode_mjpg(packet, seq, t_ns), packet, 0, 'synthetic_monotonic')


def _interrupt(signum: int, frame: object) -> None:
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('devices', help='list local UVC paths without opening them')
    for mode in ('record', 'synthetic'):
        cmd = sub.add_parser(mode)
        cmd.add_argument('--config', type=Path)
        cmd.add_argument('--output-root', type=Path)
        cmd.add_argument('--frames', type=int)
        if mode == 'record':
            cmd.add_argument('--telemetry', action='store_true',
                             help='receive MAVLink telemetry; requests stream rates only')
    replay = sub.add_parser('replay')
    replay.add_argument('run_dir', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'devices':
        print(json.dumps(discover_devices(), indent=2))
        return 0
    if args.command == 'replay':
        result = replay_run(args.run_dir, RecordPipeline)
        logged = tuple(FlightLogReader(args.run_dir).planner_commands())
        matches = command_stream_bytes(logged) == result.stream
        print(json.dumps(dict(frames=result.frames_seen, commands=len(result.commands),
                              matches_recorded_commands=matches), indent=2))
        return 0 if matches else 1
    cfg = load_config(args.config)
    old_handler = signal.signal(signal.SIGTERM, _interrupt)
    try:
        if args.command == 'synthetic':
            if cfg.camera.calibration_npz is not None:
                parser.error('synthetic images require nominal geometry (calibration_npz=null)')
            record_frames(cfg, synthetic_frames(cfg, args.frames), root=args.output_root,
                          limit=args.frames, source='synthetic')
        else:
            if cfg.camera.calibration_npz is None:
                parser.error('live perception requires calibration; first use calib.capture')
            client = MavlinkClient(cfg) if args.telemetry else None
            try:
                if client:
                    client.connect()
                    client.start()

                def state_at(t_ns: int) -> FcState | None:
                    sample = client.telemetry_at(t_ns) if client else None
                    return sample.paired_state(t_ns, cfg) if sample else None

                with StereoCapture(cfg) as capture:
                    record_frames(cfg, live_frames(capture), root=args.output_root,
                                  limit=args.frames, state_at=state_at)
            finally:
                if client:
                    client.stop(timeout_s=cfg.capture.shutdown_timeout_s)
                    if client.connection is not None:
                        client.connection.close()
    except KeyboardInterrupt:
        return 130
    finally:
        signal.signal(signal.SIGTERM, old_handler)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
