from dataclasses import dataclass
from typing import Protocol
import numpy as np

@dataclass(frozen=True)
class FrameBundle:
    left: np.ndarray          # HxWx3 uint8, raw (not rectified)
    right: np.ndarray
    t_ns: int                 # CLOCK_MONOTONIC at capture
    seq: int

@dataclass
class DepthResult:
    depth_m: np.ndarray       # HxW float32, metres
    valid: np.ndarray         # HxW bool
    conf: np.ndarray | None   # HxW float32 0..1, optional
    t_ns: int

class DepthBackend(Protocol):
    def infer(self, f: FrameBundle) -> DepthResult: ...

@dataclass
class OccupancySnapshot:
    t_ns: int
    bearings: np.ndarray      # N bin centres, rad, body frame
    distances: np.ndarray     # N float32; 5th-percentile range per bin
    confidence: np.ndarray    # N float32 0..1
    unknown: np.ndarray       # N bool
    danger: np.ndarray        # N bool

@dataclass(frozen=True)
class PlannerCommand:
    vx: float; vy: float; vz: float
    yaw_rate: float
    reason: str               # human-readable, always logged
    t_ns: int

@dataclass
class FcState:
    t_ns: int
    roll: float; pitch: float; yaw: float
    vel_ned: tuple[float, float, float]
    agl_m: float | None
    mode: str
    armed: bool
    ekf_ok: bool
    rc: dict[int, int]        # channel -> pwm
