"""Synthetic worlds for driving the planner, with or without SITL.

A scenario builds ``OccupancySnapshot``s and target detections directly. It
does *not* render images or run stereo: the thing under test here is the
decision layer, and feeding it a synthetic occupancy map is how a specific
geometry -- "wall at 4 m", "two obstacles with a 20 deg gap" -- gets stated
exactly rather than approximately.

The same scenarios run two ways. Without SITL they exercise the planner and
behaviour machine as pure functions, which is what the non-SITL unit tests
use. With SITL (``@pytest.mark.sitl``) the resulting commands are streamed to
a real ArduPilot instance flying in GUIDED, and the assertions are about what
the aircraft actually did.

Bearing convention throughout: radians, body frame, right-positive, zero
straight ahead -- the same as ``OccupancySnapshot.bearings``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from perception.obstacles import bin_centres
from perception.red_box import RedBoxDetection
from sources.types import FcState, OccupancySnapshot

if TYPE_CHECKING:
    from config import Config

FAR_RANGE_M = 30.0


def open_snapshot(
    cfg: Config, t_ns: int = 0, range_m: float = FAR_RANGE_M, speed_mps: float = 0.0
) -> OccupancySnapshot:
    """Nothing anywhere: every bin known and clear."""
    n_bins = cfg.occupancy.n_bins
    distances = np.full(n_bins, float(range_m), dtype=np.float32)
    return _snapshot(cfg, t_ns, distances, speed_mps)


def _snapshot(
    cfg: Config, t_ns: int, distances: np.ndarray, speed_mps: float
) -> OccupancySnapshot:
    """Build a snapshot with danger derived from the same formula as the map."""
    n_bins = cfg.occupancy.n_bins
    distances = np.asarray(distances, dtype=np.float32)
    unknown = ~np.isfinite(distances)
    threshold = cfg.occupancy.danger_distance(speed_mps)
    with np.errstate(invalid="ignore"):
        danger = np.asarray(distances < threshold, dtype=bool) & ~unknown
    return OccupancySnapshot(
        t_ns=int(t_ns),
        bearings=bin_centres(n_bins, cfg.occupancy.fov_rad),
        distances=distances,
        confidence=np.where(unknown, 0.0, 1.0).astype(np.float32),
        unknown=unknown,
        danger=danger,
    )


def wall_snapshot(
    cfg: Config, distance_m: float, t_ns: int = 0, speed_mps: float = 0.0
) -> OccupancySnapshot:
    """A fronto-parallel wall straight ahead.

    Range grows as ``d / cos(bearing)`` across the bins, because that is what
    a flat wall actually looks like in polar coordinates. A constant-range
    "wall" would be a cylinder centred on the aircraft, and would let a
    planner that only ever looks at the centre bin pass a test it should fail.
    """
    bearings = bin_centres(cfg.occupancy.n_bins, cfg.occupancy.fov_rad)
    distances = (distance_m / np.cos(bearings)).astype(np.float32)
    return _snapshot(cfg, t_ns, distances, speed_mps)


def gap_snapshot(
    cfg: Config,
    gap_bearing_rad: float,
    gap_width_rad: float,
    obstacle_distance_m: float = 2.0,
    open_distance_m: float = FAR_RANGE_M,
    t_ns: int = 0,
    speed_mps: float = 0.0,
) -> OccupancySnapshot:
    """Two obstacles with a navigable gap between them."""
    bearings = bin_centres(cfg.occupancy.n_bins, cfg.occupancy.fov_rad)
    distances = np.full(bearings.shape, float(obstacle_distance_m), dtype=np.float32)
    in_gap = np.abs(bearings - gap_bearing_rad) <= gap_width_rad / 2.0
    distances[in_gap] = float(open_distance_m)
    return _snapshot(cfg, t_ns, distances, speed_mps)


def unknown_snapshot(cfg: Config, t_ns: int = 0) -> OccupancySnapshot:
    """No information at all. Must never be treated as clear."""
    n_bins = cfg.occupancy.n_bins
    return OccupancySnapshot(
        t_ns=int(t_ns),
        bearings=bin_centres(n_bins, cfg.occupancy.fov_rad),
        distances=np.full(n_bins, np.nan, dtype=np.float32),
        confidence=np.zeros(n_bins, dtype=np.float32),
        unknown=np.ones(n_bins, dtype=bool),
        danger=np.zeros(n_bins, dtype=bool),
    )


def make_target(
    bearing_rad: float,
    range_m: float,
    t_ns: int = 0,
    elevation_rad: float = 0.0,
    confidence: float = 0.95,
) -> RedBoxDetection:
    """A red-box detection at a given bearing and range."""
    return RedBoxDetection(
        t_ns=int(t_ns),
        bearing_rad=float(bearing_rad),
        elevation_rad=float(elevation_rad),
        range_stereo_m=float(range_m),
        range_size_m=float(range_m),
        confidence=float(confidence),
        bbox=(0, 0, 40, 40),
        area_px=1600.0,
        solidity=0.98,
        fill_ratio=0.98,
        touches_border=False,
    )


def guided_state(
    t_ns: int = 0, yaw_rad: float = 0.0, agl_m: float = 3.0, speed_mps: float = 0.0
) -> FcState:
    """An FC state that satisfies the flight-authority guard."""
    return FcState(
        t_ns=int(t_ns),
        roll=0.0,
        pitch=0.0,
        yaw=float(yaw_rad),
        vel_ned=(float(speed_mps), 0.0, 0.0),
        agl_m=float(agl_m),
        mode="GUIDED",
        armed=True,
        ekf_ok=True,
        rc={5: 1500},
    )


@dataclass
class Scenario:
    """A named world the planner can be flown through.

    ``snapshot_at`` and ``target_at`` are functions of time, so a scenario can
    describe something that changes -- a target that only becomes visible once
    the aircraft has yawed far enough, for instance.
    """

    name: str
    description: str
    snapshot_at: Callable[[int], OccupancySnapshot]
    target_at: Callable[[int], RedBoxDetection | None] = lambda _t: None
    duration_s: float = 10.0
    more_targets: bool = False
    metadata: dict[str, float] = field(default_factory=dict)


def empty_field(cfg: Config, range_m: float = FAR_RANGE_M) -> Scenario:
    """Nothing in the way. The aircraft should be free to fly a square."""
    return Scenario(
        name="empty_field",
        description="open space in every bin; no target",
        snapshot_at=lambda t: open_snapshot(cfg, t, range_m=range_m),
        duration_s=20.0,
        metadata={"range_m": range_m},
    )


def wall_ahead(cfg: Config, distance_m: float = 3.0) -> Scenario:
    """A wall the aircraft must stop short of."""
    return Scenario(
        name="wall_ahead",
        description=f"flat wall {distance_m} m ahead spanning the whole FOV",
        snapshot_at=lambda t: wall_snapshot(cfg, distance_m, t),
        duration_s=10.0,
        metadata={"distance_m": distance_m},
    )


def two_obstacles_with_gap(
    cfg: Config, gap_bearing_deg: float = 15.0, gap_width_deg: float = 14.0
) -> Scenario:
    """Blocked either side, with one navigable gap off to one side."""
    return Scenario(
        name="two_obstacles_with_gap",
        description=f"obstacles at 2 m with a {gap_width_deg} deg gap at {gap_bearing_deg} deg",
        snapshot_at=lambda t: gap_snapshot(
            cfg,
            math.radians(gap_bearing_deg),
            math.radians(gap_width_deg),
            t_ns=t,
        ),
        duration_s=10.0,
        metadata={"gap_bearing_deg": gap_bearing_deg, "gap_width_deg": gap_width_deg},
    )


def red_box_at_bearing(
    cfg: Config, bearing_deg: float = 40.0, range_m: float = 8.0
) -> Scenario:
    """A target outside the camera FOV, so the aircraft must search for it.

    40 deg is deliberately outside the +/-32.5 deg FOV: the box is invisible
    until SEARCH has yawed far enough to bring it into view. A scenario that
    put the target inside the FOV would never exercise the search behaviour at
    all, which is the interesting part.
    """
    half_fov = cfg.obstacles.fov_rad / 2.0
    bearing = math.radians(bearing_deg)
    yaw_rate = cfg.behaviours.search_yaw_rate_rad_s
    # Time for the yaw scan to swing the target inside the FOV.
    visible_after_s = max(0.0, (abs(bearing) - half_fov) / yaw_rate)

    def target(t_ns: int) -> RedBoxDetection | None:
        if t_ns / 1e9 < visible_after_s:
            return None
        # Once in view, it sits at the FOV edge and closes as we approach.
        apparent = math.copysign(min(abs(bearing), half_fov * 0.9), bearing)
        return make_target(apparent, range_m, t_ns=t_ns)

    return Scenario(
        name="red_box_at_bearing",
        description=f"red box at {bearing_deg} deg, outside the FOV until searched for",
        snapshot_at=lambda t: open_snapshot(cfg, t),
        target_at=target,
        duration_s=30.0,
        metadata={
            "bearing_deg": bearing_deg,
            "range_m": range_m,
            "visible_after_s": visible_after_s,
        },
    )


def all_scenarios(cfg: Config) -> list[Scenario]:
    return [
        empty_field(cfg),
        wall_ahead(cfg),
        two_obstacles_with_gap(cfg),
        red_box_at_bearing(cfg),
    ]


@dataclass
class RunResult:
    """What a scenario run produced."""

    scenario: str
    commands: list
    states: list[str]
    transitions: list

    @property
    def final_state(self) -> str | None:
        return self.states[-1] if self.states else None

    def states_seen(self) -> list[str]:
        """State names in order of first appearance."""
        seen: list[str] = []
        for state in self.states:
            if not seen or seen[-1] != state:
                seen.append(state)
        return seen

    def max_forward_speed(self) -> float:
        return max((c.vx for c in self.commands), default=0.0)


def run_scenario(
    scenario: Scenario,
    cfg: Config,
    machine,
    rate_hz: float | None = None,
    t0_ns: int = 0,
) -> RunResult:
    """Drive a BehaviourMachine through a scenario. No SITL, no clock.

    Everything is a function of the tick timestamp, so a run is reproducible
    and a test can assert on the whole command stream.
    """
    from planner.behaviours import BehaviourInputs

    rate_hz = rate_hz or cfg.offboard.rate_hz
    dt_s = 1.0 / rate_hz
    n_ticks = int(scenario.duration_s * rate_hz)

    commands = []
    states = []
    for tick in range(n_ticks):
        t_ns = t0_ns + int(tick * dt_s * 1e9)
        elapsed_ns = t_ns - t0_ns
        command = machine.tick(
            BehaviourInputs(
                t_ns=t_ns,
                snapshot=scenario.snapshot_at(elapsed_ns),
                target=scenario.target_at(elapsed_ns),
                fc=guided_state(t_ns),
                mission_active=True,
                more_targets=scenario.more_targets,
                dt_s=dt_s,
            )
        )
        commands.append(command)
        states.append(machine.state.value)

    return RunResult(
        scenario=scenario.name,
        commands=commands,
        states=states,
        transitions=list(machine.transitions),
    )
