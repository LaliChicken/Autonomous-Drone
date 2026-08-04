"""Frozen-dataclass configuration for the autonomy stack.

The YAML file is the single source of truth for tunables; this module is the
only place allowed to read it. Angles are degrees in YAML (human-editable) and
radians everywhere past ``load_config``.

Every dataclass here is frozen and holds scalars only, so a Config is cheap to
compare, hash, and snapshot into a flight log. Derived matrices (Q, the
camera->body rotation) are returned by methods rather than stored, which keeps
the dataclasses free of numpy arrays.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


class ConfigError(ValueError):
    """Raised when the configuration file is missing keys or self-inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


@dataclass(frozen=True)
class EnvelopeConfig:
    """Hard flight envelope. Nothing in the stack may command outside this."""

    max_speed_mps: float
    max_agl_m: float
    max_yaw_rate_rad_s: float

    def validate(self) -> None:
        _require(self.max_speed_mps > 0.0, "envelope.max_speed_mps must be > 0")
        _require(self.max_agl_m > 0.0, "envelope.max_agl_m must be > 0")
        _require(self.max_yaw_rate_rad_s > 0.0, "envelope.max_yaw_rate_dps must be > 0")


@dataclass(frozen=True)
class CameraConfig:
    """Side-by-side stereo UVC device, split at the vertical midline."""

    device_index: int
    frame_width: int
    frame_height: int
    fps: int
    fourcc: str
    baseline_m: float
    hfov_rad: float
    calibration_npz: str | None

    @property
    def eye_width(self) -> int:
        return self.frame_width // 2

    @property
    def eye_height(self) -> int:
        return self.frame_height

    @property
    def nominal_focal_px(self) -> float:
        """Pinhole focal length implied by the per-eye HFOV."""
        return (self.eye_width / 2.0) / math.tan(self.hfov_rad / 2.0)

    def nominal_q(self) -> np.ndarray:
        """Reprojection matrix for a *nominal* (uncalibrated) rectified pair.

        Valid only for offline development: it assumes perfectly rectified eyes
        with identical intrinsics and principal points at the image centre.
        With real optics the Q matrix must come from stereo calibration.

        # QUESTION(rahul): calib/solve.py is out of scope for this branch and no
        # calibration exists yet, so depth from a real camera will be
        # systematically wrong until camera.calibration_npz is populated. Should
        # sgbm_cpu refuse to run without a calibration file, or keep falling back
        # to this nominal Q with a loud warning?
        """
        f = self.nominal_focal_px
        cx = (self.eye_width - 1) / 2.0
        cy = (self.eye_height - 1) / 2.0
        return np.array(
            [
                [1.0, 0.0, 0.0, -cx],
                [0.0, 1.0, 0.0, -cy],
                [0.0, 0.0, 0.0, f],
                [0.0, 0.0, 1.0 / self.baseline_m, 0.0],
            ],
            dtype=np.float64,
        )

    def load_q(self) -> np.ndarray:
        """Q from calibration when available, else the nominal Q."""
        if self.calibration_npz is None:
            return self.nominal_q()
        path = Path(self.calibration_npz)
        if not path.is_file():
            raise ConfigError(f"camera.calibration_npz does not exist: {path}")
        with np.load(path) as data:
            if "Q" not in data:
                raise ConfigError(f"calibration file {path} has no 'Q' array")
            q = np.asarray(data["Q"], dtype=np.float64)
        _require(q.shape == (4, 4), f"calibration Q must be 4x4, got {q.shape}")
        return q

    def validate(self) -> None:
        _require(self.frame_width > 0 and self.frame_height > 0, "camera frame size must be > 0")
        _require(
            self.frame_width % 2 == 0,
            "camera.frame_width must be even (frame is split at the vertical midline)",
        )
        _require(self.fps > 0, "camera.fps must be > 0")
        _require(self.baseline_m > 0.0, "camera.baseline_m must be > 0")
        _require(0.0 < self.hfov_rad < math.pi, "camera.hfov_deg must be in (0, 180)")
        _require(len(self.fourcc) == 4, "camera.fourcc must be exactly 4 characters")


@dataclass(frozen=True)
class MountConfig:
    """Camera pose in the body frame (FRD)."""

    x_m: float
    y_m: float
    z_m: float
    roll_rad: float
    pitch_rad: float
    yaw_rad: float

    def translation(self) -> np.ndarray:
        """Camera optical centre in body frame, metres."""
        return np.array([self.x_m, self.y_m, self.z_m], dtype=np.float64)

    def rotation_body_from_cam(self) -> np.ndarray:
        """Rotation taking a point in camera (RDF) coords into body (FRD) coords.

        Two parts. First the fixed axis swap between the OpenCV camera frame
        (x right, y down, z forward) and the body frame (x forward, y right,
        z down)::

            body_x (fwd)   = cam_z
            body_y (right) = cam_x
            body_z (down)  = cam_y

        Then the mount rotation, an ordinary FRD roll-pitch-yaw applied to the
        already-swapped axes. Signs follow the same convention as FcState, i.e.
        ArduPilot's: positive pitch is nose-UP, positive yaw is to the right.
        A camera tilted downwards therefore has a NEGATIVE mount.pitch_deg.
        """
        swap = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        cr, sr = math.cos(self.roll_rad), math.sin(self.roll_rad)
        cp, sp = math.cos(self.pitch_rad), math.sin(self.pitch_rad)
        cy, sy = math.cos(self.yaw_rad), math.sin(self.yaw_rad)
        r_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
        r_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
        r_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
        return r_z @ r_y @ r_x @ swap

    def validate(self) -> None:
        for name in ("roll_rad", "pitch_rad", "yaw_rad"):
            value = getattr(self, name)
            _require(
                -math.pi <= value <= math.pi,
                f"mount.{name.replace('_rad', '_deg')} must be within [-180, 180]",
            )


@dataclass(frozen=True)
class SgbmConfig:
    min_disparity: int
    num_disparities: int
    block_size: int
    uniqueness_ratio: int
    speckle_window_size: int
    speckle_range: int
    disp12_max_diff: int
    pre_filter_cap: int
    p1: int | None
    p2: int | None
    mode_sgbm_3way: bool

    def effective_p1(self, channels: int = 1) -> int:
        if self.p1 is not None:
            return self.p1
        return 8 * channels * self.block_size * self.block_size

    def effective_p2(self, channels: int = 1) -> int:
        if self.p2 is not None:
            return self.p2
        return 32 * channels * self.block_size * self.block_size

    def validate(self) -> None:
        _require(
            self.num_disparities > 0 and self.num_disparities % 16 == 0,
            "depth.sgbm.num_disparities must be > 0 and divisible by 16",
        )
        _require(
            self.block_size >= 1 and self.block_size % 2 == 1,
            "depth.sgbm.block_size must be odd and >= 1",
        )
        _require(self.uniqueness_ratio >= 0, "depth.sgbm.uniqueness_ratio must be >= 0")
        _require(self.speckle_window_size >= 0, "depth.sgbm.speckle_window_size must be >= 0")
        _require(self.pre_filter_cap > 0, "depth.sgbm.pre_filter_cap must be > 0")
        if self.p1 is not None and self.p2 is not None:
            _require(self.p2 > self.p1, "depth.sgbm.p2 must be > p1")


@dataclass(frozen=True)
class PostprocessConfig:
    min_depth_m: float
    max_depth_m: float
    speckle_max_area_px: int
    speckle_max_diff_px: float
    lr_consistency: bool
    lr_max_disp_diff_px: float
    disparity_sigma_px: float
    max_depth_sigma_m: float
    min_confidence: float

    def validate(self) -> None:
        _require(self.min_depth_m > 0.0, "depth.postprocess.min_depth_m must be > 0")
        _require(
            self.max_depth_m > self.min_depth_m,
            "depth.postprocess.max_depth_m must be > min_depth_m",
        )
        _require(
            self.speckle_max_area_px >= 0, "depth.postprocess.speckle_max_area_px must be >= 0"
        )
        _require(
            self.speckle_max_diff_px > 0.0, "depth.postprocess.speckle_max_diff_px must be > 0"
        )
        _require(
            self.lr_max_disp_diff_px > 0.0, "depth.postprocess.lr_max_disp_diff_px must be > 0"
        )
        _require(
            self.disparity_sigma_px > 0.0, "depth.postprocess.disparity_sigma_px must be > 0"
        )
        _require(
            self.max_depth_sigma_m > 0.0, "depth.postprocess.max_depth_sigma_m must be > 0"
        )
        _require(
            0.0 <= self.min_confidence <= 1.0,
            "depth.postprocess.min_confidence must be within [0, 1]",
        )


@dataclass(frozen=True)
class DepthConfig:
    sgbm: SgbmConfig
    postprocess: PostprocessConfig

    def validate(self) -> None:
        self.sgbm.validate()
        self.postprocess.validate()


@dataclass(frozen=True)
class ObstaclesConfig:
    n_bins: int
    fov_rad: float
    distance_percentile: float
    min_points_per_bin: int
    points_for_full_confidence: int
    height_floor_m: float
    height_ceiling_m: float

    def validate(self) -> None:
        _require(self.n_bins > 0, "obstacles.n_bins must be > 0")
        _require(0.0 < self.fov_rad <= 2 * math.pi, "obstacles.fov_deg must be in (0, 360]")
        _require(
            0.0 <= self.distance_percentile <= 100.0,
            "obstacles.distance_percentile must be within [0, 100]",
        )
        _require(self.min_points_per_bin >= 1, "obstacles.min_points_per_bin must be >= 1")
        _require(
            self.points_for_full_confidence >= self.min_points_per_bin,
            "obstacles.points_for_full_confidence must be >= min_points_per_bin",
        )
        _require(
            self.height_ceiling_m > self.height_floor_m,
            "obstacles.height_ceiling_m must be > height_floor_m",
        )


@dataclass(frozen=True)
class RedBoxConfig:
    hue_lo_1: int
    hue_hi_1: int
    hue_lo_2: int
    hue_hi_2: int
    sat_min: int
    sat_max: int
    val_min: int
    val_max: int
    morph_kernel_px: int
    morph_open_iterations: int
    morph_close_iterations: int
    min_area_px: float
    max_area_px: float
    min_aspect: float
    max_aspect: float
    min_solidity: float
    box_width_m: float
    min_confidence: float

    def validate(self) -> None:
        for name in ("hue_lo_1", "hue_hi_1", "hue_lo_2", "hue_hi_2"):
            value = getattr(self, name)
            _require(0 <= value <= 179, f"red_box.{name} must be within [0, 179] (OpenCV hue)")
        _require(self.hue_lo_1 <= self.hue_hi_1, "red_box.hue_lo_1 must be <= hue_hi_1")
        _require(self.hue_lo_2 <= self.hue_hi_2, "red_box.hue_lo_2 must be <= hue_hi_2")
        for name in ("sat_min", "sat_max", "val_min", "val_max"):
            value = getattr(self, name)
            _require(0 <= value <= 255, f"red_box.{name} must be within [0, 255]")
        _require(self.sat_min <= self.sat_max, "red_box.sat_min must be <= sat_max")
        _require(self.val_min <= self.val_max, "red_box.val_min must be <= val_max")
        _require(
            self.morph_kernel_px >= 1 and self.morph_kernel_px % 2 == 1,
            "red_box.morph_kernel_px must be odd and >= 1",
        )
        _require(self.min_area_px > 0.0, "red_box.min_area_px must be > 0")
        _require(self.max_area_px > self.min_area_px, "red_box.max_area_px must be > min_area_px")
        _require(self.min_aspect > 0.0, "red_box.min_aspect must be > 0")
        _require(self.max_aspect > self.min_aspect, "red_box.max_aspect must be > min_aspect")
        _require(0.0 <= self.min_solidity <= 1.0, "red_box.min_solidity must be within [0, 1]")
        _require(self.box_width_m > 0.0, "red_box.box_width_m must be > 0")
        _require(0.0 <= self.min_confidence <= 1.0, "red_box.min_confidence must be within [0, 1]")


@dataclass(frozen=True)
class OccupancyConfig:
    n_bins: int
    fov_rad: float
    max_age_s: float
    age_decay_per_s: float
    min_confidence: float
    danger_reaction_s: float
    danger_decel_mps2: float
    danger_margin_m: float

    def danger_distance(self, speed_mps: float) -> float:
        """Distance inside which a bin is dangerous at the given speed.

        Reaction distance plus braking distance plus a fixed margin. This is
        the formula referenced by occupancy.danger_* in config.yaml; it lives
        here so world/occupancy.py holds no constants of its own.
        """
        speed = max(0.0, speed_mps)
        return (
            self.danger_reaction_s * speed
            + (speed * speed) / (2.0 * self.danger_decel_mps2)
            + self.danger_margin_m
        )

    def validate(self) -> None:
        _require(self.n_bins > 0, "occupancy.n_bins must be > 0")
        _require(0.0 < self.fov_rad <= 2 * math.pi, "occupancy.fov_deg must be in (0, 360]")
        _require(self.max_age_s > 0.0, "occupancy.max_age_s must be > 0")
        _require(self.age_decay_per_s >= 0.0, "occupancy.age_decay_per_s must be >= 0")
        _require(
            0.0 <= self.min_confidence <= 1.0, "occupancy.min_confidence must be within [0, 1]"
        )
        _require(self.danger_reaction_s >= 0.0, "occupancy.danger_reaction_s must be >= 0")
        _require(self.danger_decel_mps2 > 0.0, "occupancy.danger_decel_mps2 must be > 0")
        _require(self.danger_margin_m >= 0.0, "occupancy.danger_margin_m must be >= 0")


@dataclass(frozen=True)
class LocalPlannerConfig:
    clearance_threshold_m: float
    w_goal: float
    w_clearance: float
    w_turn: float
    max_yaw_rate_rad_s: float
    min_clearance_for_cost_m: float

    def validate(self) -> None:
        _require(
            self.clearance_threshold_m > 0.0, "local_planner.clearance_threshold_m must be > 0"
        )
        for name in ("w_goal", "w_clearance", "w_turn"):
            _require(getattr(self, name) >= 0.0, f"local_planner.{name} must be >= 0")
        _require(self.max_yaw_rate_rad_s > 0.0, "local_planner.max_yaw_rate_dps must be > 0")
        _require(
            self.min_clearance_for_cost_m > 0.0,
            "local_planner.min_clearance_for_cost_m must be > 0",
        )


@dataclass(frozen=True)
class BehavioursConfig:
    autonomy_speed_max_mps: float
    search_yaw_rate_rad_s: float
    search_timeout_s: float
    approach_speed_mps: float
    standoff_m: float
    arrive_tolerance_m: float
    acquire_consecutive_ticks: int
    lost_target_timeout_s: float

    def validate(self) -> None:
        _require(
            self.autonomy_speed_max_mps > 0.0, "behaviours.autonomy_speed_max_mps must be > 0"
        )
        _require(self.search_yaw_rate_rad_s > 0.0, "behaviours.search_yaw_rate_dps must be > 0")
        _require(self.search_timeout_s > 0.0, "behaviours.search_timeout_s must be > 0")
        _require(self.approach_speed_mps > 0.0, "behaviours.approach_speed_mps must be > 0")
        _require(self.standoff_m > 0.0, "behaviours.standoff_m must be > 0")
        _require(self.arrive_tolerance_m > 0.0, "behaviours.arrive_tolerance_m must be > 0")
        _require(
            self.acquire_consecutive_ticks >= 1,
            "behaviours.acquire_consecutive_ticks must be >= 1",
        )
        _require(self.lost_target_timeout_s > 0.0, "behaviours.lost_target_timeout_s must be > 0")


@dataclass(frozen=True)
class OffboardConfig:
    rate_hz: float
    max_accel_mps2: float
    max_yaw_accel_rad_s2: float
    command_timeout_s: float

    @property
    def period_s(self) -> float:
        return 1.0 / self.rate_hz

    def validate(self) -> None:
        _require(self.rate_hz > 0.0, "offboard.rate_hz must be > 0")
        _require(self.max_accel_mps2 > 0.0, "offboard.max_accel_mps2 must be > 0")
        _require(self.max_yaw_accel_rad_s2 > 0.0, "offboard.max_yaw_accel_dps2 must be > 0")
        _require(self.command_timeout_s > 0.0, "offboard.command_timeout_s must be > 0")


@dataclass(frozen=True)
class MavlinkConfig:
    endpoint: str
    source_system: int
    source_component: int
    heartbeat_timeout_s: float
    stream_rates_hz: tuple[tuple[str, int], ...]
    attitude_buffer_len: int
    attitude_max_extrapolation_s: float

    def stream_rates(self) -> dict[str, int]:
        return dict(self.stream_rates_hz)

    def validate(self) -> None:
        _require(bool(self.endpoint), "mavlink.endpoint must not be empty")
        _require(0 <= self.source_system <= 255, "mavlink.source_system must be within [0, 255]")
        _require(
            0 <= self.source_component <= 255, "mavlink.source_component must be within [0, 255]"
        )
        _require(self.heartbeat_timeout_s > 0.0, "mavlink.heartbeat_timeout_s must be > 0")
        _require(bool(self.stream_rates_hz), "mavlink.stream_rates_hz must not be empty")
        for name, rate in self.stream_rates_hz:
            _require(rate > 0, f"mavlink.stream_rates_hz.{name} must be > 0")
        _require(self.attitude_buffer_len >= 2, "mavlink.attitude_buffer_len must be >= 2")
        _require(
            self.attitude_max_extrapolation_s >= 0.0,
            "mavlink.attitude_max_extrapolation_s must be >= 0",
        )


@dataclass(frozen=True)
class BenchConfig:
    bad_pixel_threshold_px: float
    baseline_path: str
    regression_tolerance: float
    middlebury_dir: str

    def validate(self) -> None:
        _require(self.bad_pixel_threshold_px > 0.0, "bench.bad_pixel_threshold_px must be > 0")
        _require(bool(self.baseline_path), "bench.baseline_path must not be empty")
        _require(
            self.regression_tolerance >= 0.0, "bench.regression_tolerance must be >= 0"
        )
        _require(bool(self.middlebury_dir), "bench.middlebury_dir must not be empty")


@dataclass(frozen=True)
class MetricsConfig:
    window: int

    def validate(self) -> None:
        _require(self.window >= 1, "metrics.window must be >= 1")


@dataclass(frozen=True)
class FlightlogConfig:
    root: str
    write_frames: bool
    fsync_every: int

    def validate(self) -> None:
        _require(bool(self.root), "flightlog.root must not be empty")
        _require(self.fsync_every >= 0, "flightlog.fsync_every must be >= 0")


@dataclass(frozen=True)
class Config:
    envelope: EnvelopeConfig
    camera: CameraConfig
    mount: MountConfig
    depth: DepthConfig
    obstacles: ObstaclesConfig
    red_box: RedBoxConfig
    occupancy: OccupancyConfig
    local_planner: LocalPlannerConfig
    behaviours: BehavioursConfig
    offboard: OffboardConfig
    mavlink: MavlinkConfig
    bench: BenchConfig
    metrics: MetricsConfig
    flightlog: FlightlogConfig
    source_path: str = field(default="", compare=False)
    # The YAML-shaped mapping this Config was built from, kept verbatim so a
    # flight log can snapshot something that loads straight back through
    # load_config_from_dict. to_dict() is the *resolved* view (radians,
    # derived values) and is not re-loadable, which is why both are stored.
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def validate(self) -> None:
        for f in fields(self):
            section = getattr(self, f.name)
            if is_dataclass(section):
                section.validate()
        self._validate_cross_section()

    def _validate_cross_section(self) -> None:
        _require(
            self.obstacles.n_bins == self.occupancy.n_bins,
            "obstacles.n_bins and occupancy.n_bins must match "
            f"({self.obstacles.n_bins} != {self.occupancy.n_bins})",
        )
        _require(
            math.isclose(self.obstacles.fov_rad, self.occupancy.fov_rad, rel_tol=1e-9),
            "obstacles.fov_deg and occupancy.fov_deg must match",
        )
        _require(
            self.behaviours.autonomy_speed_max_mps <= self.envelope.max_speed_mps,
            "behaviours.autonomy_speed_max_mps must not exceed envelope.max_speed_mps",
        )
        _require(
            self.behaviours.approach_speed_mps <= self.behaviours.autonomy_speed_max_mps,
            "behaviours.approach_speed_mps must not exceed autonomy_speed_max_mps",
        )
        _require(
            self.local_planner.max_yaw_rate_rad_s <= self.envelope.max_yaw_rate_rad_s,
            "local_planner.max_yaw_rate_dps must not exceed envelope.max_yaw_rate_dps",
        )
        _require(
            self.behaviours.search_yaw_rate_rad_s <= self.envelope.max_yaw_rate_rad_s,
            "behaviours.search_yaw_rate_dps must not exceed envelope.max_yaw_rate_dps",
        )
        _require(
            self.behaviours.standoff_m >= self.depth.postprocess.min_depth_m,
            "behaviours.standoff_m must be >= depth.postprocess.min_depth_m "
            "(cannot hold a standoff closer than the sensor can measure)",
        )
        _require(
            self.local_planner.clearance_threshold_m <= self.depth.postprocess.max_depth_m,
            "local_planner.clearance_threshold_m must not exceed "
            "depth.postprocess.max_depth_m (no bin could ever clear it)",
        )

    def to_dict(self) -> dict[str, Any]:
        """Resolved plain-dict form (radians, derived values), for inspection.

        Not re-loadable — use ``raw`` for a snapshot you intend to load back.
        """
        out = asdict(self)
        out.pop("raw", None)
        out.pop("source_path", None)
        out["mavlink"]["stream_rates_hz"] = dict(self.mavlink.stream_rates_hz)
        return out


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    if name not in raw:
        raise ConfigError(f"config is missing the '{name}' section")
    value = raw[name]
    if not isinstance(value, dict):
        raise ConfigError(f"config section '{name}' must be a mapping, got {type(value).__name__}")
    return value


def _get(section: dict[str, Any], name: str, path: str) -> Any:
    if name not in section:
        raise ConfigError(f"config is missing '{path}.{name}'")
    return section[name]


def _deg(section: dict[str, Any], name: str, path: str) -> float:
    return math.radians(float(_get(section, name, path)))


def _build(raw: dict[str, Any], source_path: str) -> Config:
    env = _section(raw, "envelope")
    cam = _section(raw, "camera")
    mount = _section(raw, "mount")
    depth = _section(raw, "depth")
    sgbm = _section(depth, "sgbm")
    post = _section(depth, "postprocess")
    obst = _section(raw, "obstacles")
    box = _section(raw, "red_box")
    occ = _section(raw, "occupancy")
    plan = _section(raw, "local_planner")
    beh = _section(raw, "behaviours")
    off = _section(raw, "offboard")
    mav = _section(raw, "mavlink")
    ben = _section(raw, "bench")
    met = _section(raw, "metrics")
    log = _section(raw, "flightlog")

    rates = _get(mav, "stream_rates_hz", "mavlink")
    if not isinstance(rates, dict):
        raise ConfigError("mavlink.stream_rates_hz must be a mapping of message name -> Hz")

    calib = _get(cam, "calibration_npz", "camera")

    return Config(
        envelope=EnvelopeConfig(
            max_speed_mps=float(_get(env, "max_speed_mps", "envelope")),
            max_agl_m=float(_get(env, "max_agl_m", "envelope")),
            max_yaw_rate_rad_s=_deg(env, "max_yaw_rate_dps", "envelope"),
        ),
        camera=CameraConfig(
            device_index=int(_get(cam, "device_index", "camera")),
            frame_width=int(_get(cam, "frame_width", "camera")),
            frame_height=int(_get(cam, "frame_height", "camera")),
            fps=int(_get(cam, "fps", "camera")),
            fourcc=str(_get(cam, "fourcc", "camera")),
            baseline_m=float(_get(cam, "baseline_m", "camera")),
            hfov_rad=_deg(cam, "hfov_deg", "camera"),
            calibration_npz=None if calib is None else str(calib),
        ),
        mount=MountConfig(
            x_m=float(_get(mount, "x_m", "mount")),
            y_m=float(_get(mount, "y_m", "mount")),
            z_m=float(_get(mount, "z_m", "mount")),
            roll_rad=_deg(mount, "roll_deg", "mount"),
            pitch_rad=_deg(mount, "pitch_deg", "mount"),
            yaw_rad=_deg(mount, "yaw_deg", "mount"),
        ),
        depth=DepthConfig(
            sgbm=SgbmConfig(
                min_disparity=int(_get(sgbm, "min_disparity", "depth.sgbm")),
                num_disparities=int(_get(sgbm, "num_disparities", "depth.sgbm")),
                block_size=int(_get(sgbm, "block_size", "depth.sgbm")),
                uniqueness_ratio=int(_get(sgbm, "uniqueness_ratio", "depth.sgbm")),
                speckle_window_size=int(_get(sgbm, "speckle_window_size", "depth.sgbm")),
                speckle_range=int(_get(sgbm, "speckle_range", "depth.sgbm")),
                disp12_max_diff=int(_get(sgbm, "disp12_max_diff", "depth.sgbm")),
                pre_filter_cap=int(_get(sgbm, "pre_filter_cap", "depth.sgbm")),
                p1=None if _get(sgbm, "p1", "depth.sgbm") is None else int(sgbm["p1"]),
                p2=None if _get(sgbm, "p2", "depth.sgbm") is None else int(sgbm["p2"]),
                mode_sgbm_3way=bool(_get(sgbm, "mode_sgbm_3way", "depth.sgbm")),
            ),
            postprocess=PostprocessConfig(
                min_depth_m=float(_get(post, "min_depth_m", "depth.postprocess")),
                max_depth_m=float(_get(post, "max_depth_m", "depth.postprocess")),
                speckle_max_area_px=int(_get(post, "speckle_max_area_px", "depth.postprocess")),
                speckle_max_diff_px=float(_get(post, "speckle_max_diff_px", "depth.postprocess")),
                lr_consistency=bool(_get(post, "lr_consistency", "depth.postprocess")),
                lr_max_disp_diff_px=float(_get(post, "lr_max_disp_diff_px", "depth.postprocess")),
                disparity_sigma_px=float(_get(post, "disparity_sigma_px", "depth.postprocess")),
                max_depth_sigma_m=float(_get(post, "max_depth_sigma_m", "depth.postprocess")),
                min_confidence=float(_get(post, "min_confidence", "depth.postprocess")),
            ),
        ),
        obstacles=ObstaclesConfig(
            n_bins=int(_get(obst, "n_bins", "obstacles")),
            fov_rad=_deg(obst, "fov_deg", "obstacles"),
            distance_percentile=float(_get(obst, "distance_percentile", "obstacles")),
            min_points_per_bin=int(_get(obst, "min_points_per_bin", "obstacles")),
            points_for_full_confidence=int(
                _get(obst, "points_for_full_confidence", "obstacles")
            ),
            height_floor_m=float(_get(obst, "height_floor_m", "obstacles")),
            height_ceiling_m=float(_get(obst, "height_ceiling_m", "obstacles")),
        ),
        red_box=RedBoxConfig(
            hue_lo_1=int(_get(box, "hue_lo_1", "red_box")),
            hue_hi_1=int(_get(box, "hue_hi_1", "red_box")),
            hue_lo_2=int(_get(box, "hue_lo_2", "red_box")),
            hue_hi_2=int(_get(box, "hue_hi_2", "red_box")),
            sat_min=int(_get(box, "sat_min", "red_box")),
            sat_max=int(_get(box, "sat_max", "red_box")),
            val_min=int(_get(box, "val_min", "red_box")),
            val_max=int(_get(box, "val_max", "red_box")),
            morph_kernel_px=int(_get(box, "morph_kernel_px", "red_box")),
            morph_open_iterations=int(_get(box, "morph_open_iterations", "red_box")),
            morph_close_iterations=int(_get(box, "morph_close_iterations", "red_box")),
            min_area_px=float(_get(box, "min_area_px", "red_box")),
            max_area_px=float(_get(box, "max_area_px", "red_box")),
            min_aspect=float(_get(box, "min_aspect", "red_box")),
            max_aspect=float(_get(box, "max_aspect", "red_box")),
            min_solidity=float(_get(box, "min_solidity", "red_box")),
            box_width_m=float(_get(box, "box_width_m", "red_box")),
            min_confidence=float(_get(box, "min_confidence", "red_box")),
        ),
        occupancy=OccupancyConfig(
            n_bins=int(_get(occ, "n_bins", "occupancy")),
            fov_rad=_deg(occ, "fov_deg", "occupancy"),
            max_age_s=float(_get(occ, "max_age_s", "occupancy")),
            age_decay_per_s=float(_get(occ, "age_decay_per_s", "occupancy")),
            min_confidence=float(_get(occ, "min_confidence", "occupancy")),
            danger_reaction_s=float(_get(occ, "danger_reaction_s", "occupancy")),
            danger_decel_mps2=float(_get(occ, "danger_decel_mps2", "occupancy")),
            danger_margin_m=float(_get(occ, "danger_margin_m", "occupancy")),
        ),
        local_planner=LocalPlannerConfig(
            clearance_threshold_m=float(_get(plan, "clearance_threshold_m", "local_planner")),
            w_goal=float(_get(plan, "w_goal", "local_planner")),
            w_clearance=float(_get(plan, "w_clearance", "local_planner")),
            w_turn=float(_get(plan, "w_turn", "local_planner")),
            max_yaw_rate_rad_s=_deg(plan, "max_yaw_rate_dps", "local_planner"),
            min_clearance_for_cost_m=float(
                _get(plan, "min_clearance_for_cost_m", "local_planner")
            ),
        ),
        behaviours=BehavioursConfig(
            autonomy_speed_max_mps=float(_get(beh, "autonomy_speed_max_mps", "behaviours")),
            search_yaw_rate_rad_s=_deg(beh, "search_yaw_rate_dps", "behaviours"),
            search_timeout_s=float(_get(beh, "search_timeout_s", "behaviours")),
            approach_speed_mps=float(_get(beh, "approach_speed_mps", "behaviours")),
            standoff_m=float(_get(beh, "standoff_m", "behaviours")),
            arrive_tolerance_m=float(_get(beh, "arrive_tolerance_m", "behaviours")),
            acquire_consecutive_ticks=int(_get(beh, "acquire_consecutive_ticks", "behaviours")),
            lost_target_timeout_s=float(_get(beh, "lost_target_timeout_s", "behaviours")),
        ),
        offboard=OffboardConfig(
            rate_hz=float(_get(off, "rate_hz", "offboard")),
            max_accel_mps2=float(_get(off, "max_accel_mps2", "offboard")),
            max_yaw_accel_rad_s2=_deg(off, "max_yaw_accel_dps2", "offboard"),
            command_timeout_s=float(_get(off, "command_timeout_s", "offboard")),
        ),
        mavlink=MavlinkConfig(
            endpoint=str(_get(mav, "endpoint", "mavlink")),
            source_system=int(_get(mav, "source_system", "mavlink")),
            source_component=int(_get(mav, "source_component", "mavlink")),
            heartbeat_timeout_s=float(_get(mav, "heartbeat_timeout_s", "mavlink")),
            stream_rates_hz=tuple(sorted((str(k), int(v)) for k, v in rates.items())),
            attitude_buffer_len=int(_get(mav, "attitude_buffer_len", "mavlink")),
            attitude_max_extrapolation_s=float(
                _get(mav, "attitude_max_extrapolation_s", "mavlink")
            ),
        ),
        bench=BenchConfig(
            bad_pixel_threshold_px=float(_get(ben, "bad_pixel_threshold_px", "bench")),
            baseline_path=str(_get(ben, "baseline_path", "bench")),
            regression_tolerance=float(_get(ben, "regression_tolerance", "bench")),
            middlebury_dir=str(_get(ben, "middlebury_dir", "bench")),
        ),
        metrics=MetricsConfig(
            window=int(_get(met, "window", "metrics")),
        ),
        flightlog=FlightlogConfig(
            root=str(_get(log, "root", "flightlog")),
            write_frames=bool(_get(log, "write_frames", "flightlog")),
            fsync_every=int(_get(log, "fsync_every", "flightlog")),
        ),
        source_path=source_path,
        raw=copy.deepcopy(raw),
    )


def load_config(path: str | Path | None = None) -> Config:
    """Load, convert, and validate the configuration.

    Raises ConfigError on a missing key, a wrong-typed section, or any value
    that fails a range or cross-section check.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ConfigError(f"config file {config_path} must contain a top-level mapping")
    config = _build(raw, str(config_path))
    config.validate()
    return config


def load_config_from_dict(raw: dict[str, Any], source_path: str = "<dict>") -> Config:
    """Same as load_config but from an already-parsed mapping. Used by tests
    and by replay, which reads the config snapshot out of a run directory."""
    config = _build(raw, source_path)
    config.validate()
    return config
