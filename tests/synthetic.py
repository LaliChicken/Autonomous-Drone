"""Synthetic run directories and a deterministic stand-in pipeline.

Everything here is real data through real code paths -- actual JPEG bitstreams
written by the actual FlightLog writer, read back by the actual reader. The
only thing faked is the pipeline itself, which is legitimate: the thing under
test in Package A is the logging and replay machinery, and the perception and
planning packages it will eventually drive do not exist yet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from config import Config
from infra.flightlog import FlightLog
from sources.types import FcState, FrameBundle, PlannerCommand

FRAME_WIDTH = 64  # side-by-side, so 32 px per eye
FRAME_HEIGHT = 24
FRAME_PERIOD_NS = 33_333_333  # ~30 Hz
TELEMETRY_PERIOD_NS = 20_000_000  # 50 Hz


def make_side_by_side(seq: int, width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT) -> np.ndarray:
    """A deterministic side-by-side stereo frame that varies with ``seq``.

    The two halves differ by a horizontal shift, so a real stereo matcher sees
    a plausible disparity rather than two identical images.
    """
    rng = np.random.default_rng(seed=1234)
    texture = rng.integers(0, 256, size=(height, width // 2, 3), dtype=np.uint8)
    shift = 1 + (seq % 3)
    left = np.roll(texture, shift, axis=1)
    right = texture
    frame = np.concatenate([left, right], axis=1)
    # A seq-dependent brightness ramp gives the fake pipeline something to
    # respond to that changes frame to frame.
    frame = np.clip(frame.astype(np.int16) + (seq * 7) % 60, 0, 255).astype(np.uint8)
    return frame


def encode_mjpg(frame: np.ndarray, quality: int = 90) -> bytes:
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("failed to encode synthetic frame")
    return bytes(buffer.tobytes())


def make_fc_state(index: int, t_ns: int) -> FcState:
    return FcState(
        t_ns=t_ns,
        roll=0.01 * index,
        pitch=-0.02 * index,
        yaw=math.radians((index * 3) % 360),
        vel_ned=(0.5 * index, 0.0, -0.1),
        agl_m=None if index % 7 == 0 else 3.0 + 0.1 * index,
        mode="GUIDED",
        armed=index > 1,
        ekf_ok=True,
        rc={5: 1500 + index, 6: 1000},
    )


def write_synthetic_run(
    root: Path,
    cfg: Config,
    run_id: str = "synthetic",
    n_frames: int = 12,
    n_telemetry: int = 30,
    t0_ns: int = 1_000_000_000,
) -> Path:
    """Write a complete run directory and return its path."""
    with FlightLog.create(cfg, root=root, run_id=run_id) as log:
        for index in range(n_telemetry):
            log.write_telemetry(make_fc_state(index, t0_ns + index * TELEMETRY_PERIOD_NS))
        for seq in range(n_frames):
            frame = make_side_by_side(seq)
            log.write_frame(seq, t0_ns + seq * FRAME_PERIOD_NS, encode_mjpg(frame))
    return root / run_id


@dataclass
class FakePipeline:
    """Deterministic pipeline stand-in that reads config and carries state.

    Deliberately stateful (``self._tick``) so a determinism test can catch
    state leaking across replays: if ``reset()`` were not called, the second
    replay's tick indices would differ and the command stream would not match.

    Deliberately config-sensitive (speed comes from ``behaviours``) so an A/B
    comparison over two configs produces a real difference rather than a
    trivially identical stream.
    """

    cfg: Config
    _tick: int = 0
    seen_states: int = 0

    def reset(self) -> None:
        self._tick = 0
        self.seen_states = 0

    def tick(self, bundle: FrameBundle, state: FcState | None) -> PlannerCommand | None:
        self._tick += 1
        if state is not None:
            self.seen_states += 1
        # Every third frame produces no command, so the command stream is not
        # trivially one-per-frame.
        if self._tick % 3 == 0:
            return None
        brightness = float(np.mean(bundle.left)) / 255.0
        speed = round(brightness * self.cfg.behaviours.approach_speed_mps, 6)
        yaw_rate = round(
            math.copysign(1.0, state.yaw if state else 1.0)
            * self.cfg.local_planner.max_yaw_rate_rad_s
            * 0.1,
            6,
        )
        return PlannerCommand(
            vx=speed,
            vy=0.0,
            vz=0.0,
            yaw_rate=yaw_rate,
            reason=f"tick={self._tick} seq={bundle.seq}",
            t_ns=bundle.t_ns,
        )


def fake_pipeline_factory(cfg: Config) -> FakePipeline:
    return FakePipeline(cfg=cfg)
