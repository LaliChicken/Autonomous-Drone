"""VFH-lite heading selection."""

from __future__ import annotations

import math

import numpy as np
import pytest

from config import Config
from planner.local_planner import LocalPlanner, wrap_pi
from sim.scenarios import gap_snapshot, open_snapshot, unknown_snapshot, wall_snapshot

DT = 0.05


@pytest.fixture
def cfg16(cfg_from) -> Config:
    return cfg_from({"obstacles": {"n_bins": 16}, "occupancy": {"n_bins": 16}})


def test_wrap_pi_wraps() -> None:
    assert wrap_pi(0.0) == pytest.approx(0.0)
    assert wrap_pi(math.pi + 0.1) == pytest.approx(-math.pi + 0.1)
    assert wrap_pi(-math.pi - 0.1) == pytest.approx(math.pi - 0.1)


# --------------------------------------------------------------------------
# Choosing
# --------------------------------------------------------------------------


def test_open_space_goes_straight_at_a_straight_goal(cfg16: Config) -> None:
    planner = LocalPlanner(cfg16)
    command, debug = planner.plan(open_snapshot(cfg16), goal_bearing_rad=0.0, speed_mps=2.0,
                                  dt_s=DT)
    assert debug.chosen_bearing_rad is not None
    assert abs(debug.chosen_bearing_rad) < math.radians(5.0)
    assert command.vx > 0.0
    assert abs(command.yaw_rate) < 0.2


def test_it_turns_toward_an_off_axis_goal(cfg16: Config) -> None:
    planner = LocalPlanner(cfg16)
    goal = math.radians(25.0)
    _command, debug = planner.plan(open_snapshot(cfg16), goal, speed_mps=2.0, dt_s=DT)
    assert debug.chosen_bearing_rad is not None
    assert debug.chosen_bearing_rad > math.radians(10.0)


def test_it_picks_the_gap(cfg16: Config) -> None:
    planner = LocalPlanner(cfg16)
    gap = math.radians(18.0)
    snapshot = gap_snapshot(cfg16, gap, math.radians(14.0))
    _command, debug = planner.plan(snapshot, goal_bearing_rad=0.0, speed_mps=2.0, dt_s=DT)
    assert debug.chosen_bearing_rad is not None
    # The goal is dead ahead but ahead is blocked, so it must go via the gap.
    assert abs(debug.chosen_bearing_rad - gap) <= math.radians(8.0)


def test_it_picks_the_gap_on_the_other_side_too(cfg16: Config) -> None:
    planner = LocalPlanner(cfg16)
    gap = math.radians(-18.0)
    snapshot = gap_snapshot(cfg16, gap, math.radians(14.0))
    _command, debug = planner.plan(snapshot, goal_bearing_rad=0.0, speed_mps=2.0, dt_s=DT)
    assert debug.chosen_bearing_rad is not None
    assert debug.chosen_bearing_rad < 0.0


# --------------------------------------------------------------------------
# Stopping
# --------------------------------------------------------------------------


def test_a_wall_ahead_stops(cfg16: Config) -> None:
    planner = LocalPlanner(cfg16)
    # Inside the clearance threshold everywhere.
    snapshot = wall_snapshot(cfg16, distance_m=1.0)
    command, debug = planner.plan(snapshot, goal_bearing_rad=0.0, speed_mps=2.0, dt_s=DT)
    assert debug.n_candidates == 0
    assert (command.vx, command.vy, command.vz, command.yaw_rate) == (0.0, 0.0, 0.0, 0.0)
    assert command.reason.startswith("STOP")


def test_an_all_unknown_map_stops(cfg16: Config) -> None:
    # The rule that matters most: no information is not permission to fly.
    planner = LocalPlanner(cfg16)
    command, debug = planner.plan(unknown_snapshot(cfg16), 0.0, speed_mps=2.0, dt_s=DT)
    assert debug.n_candidates == 0
    assert debug.n_unknown == cfg16.occupancy.n_bins
    assert command.vx == 0.0
    assert "unknown" in command.reason


def test_stop_reason_counts_the_bins(cfg16: Config) -> None:
    planner = LocalPlanner(cfg16)
    command, _ = planner.plan(wall_snapshot(cfg16, 1.0), 0.0, speed_mps=2.0, dt_s=DT)
    assert "too close" in command.reason
    assert str(cfg16.occupancy.n_bins) in command.reason


def test_a_dangerous_bin_is_not_a_candidate(cfg_from) -> None:
    # The brief's two conditions are clearance and unknown. Danger is a third
    # and is not implied: at this speed the danger distance exceeds the
    # clearance threshold, so a bin can be "clear" and still too close to
    # stop in. Excluding it is deliberate.
    cfg = cfg_from(
        {
            "obstacles": {"n_bins": 8},
            "occupancy": {"n_bins": 8},
            "local_planner": {"clearance_threshold_m": 2.5},
        }
    )
    speed = 2.5
    danger_distance = cfg.occupancy.danger_distance(speed)
    assert danger_distance > cfg.local_planner.clearance_threshold_m

    between = (danger_distance + cfg.local_planner.clearance_threshold_m) / 2.0
    snapshot = open_snapshot(cfg, range_m=between, speed_mps=speed)
    assert snapshot.danger.all(), "fixture should place every bin in the danger band"
    assert np.all(snapshot.distances > cfg.local_planner.clearance_threshold_m)

    _command, debug = LocalPlanner(cfg).plan(snapshot, 0.0, speed_mps=speed, dt_s=DT)
    assert debug.n_candidates == 0


# --------------------------------------------------------------------------
# Cost terms
# --------------------------------------------------------------------------


def test_non_candidates_cost_infinity(cfg16: Config) -> None:
    planner = LocalPlanner(cfg16)
    snapshot = gap_snapshot(cfg16, 0.0, math.radians(10.0))
    cost, candidates = planner.costs(snapshot, 0.0, DT)
    assert np.all(np.isinf(cost[~candidates]))
    assert np.all(np.isfinite(cost[candidates]))


def test_goal_weight_pulls_the_choice(cfg_from) -> None:
    base = {"obstacles": {"n_bins": 16}, "occupancy": {"n_bins": 16}}
    goal_driven = cfg_from({**base, "local_planner": {"w_goal": 50.0, "w_clearance": 0.0,
                                                      "w_turn": 0.0}})
    snapshot = open_snapshot(goal_driven)
    goal = math.radians(28.0)
    _c, debug = LocalPlanner(goal_driven).plan(snapshot, goal, speed_mps=1.0, dt_s=DT)
    # With only the goal term, the nearest bin centre to the goal must win.
    nearest = int(np.argmin(np.abs(snapshot.bearings - goal)))
    assert debug.chosen_index == nearest


def test_clearance_weight_prefers_open_space(cfg_from) -> None:
    cfg = cfg_from(
        {
            "obstacles": {"n_bins": 16},
            "occupancy": {"n_bins": 16},
            "local_planner": {"w_goal": 0.0, "w_clearance": 10.0, "w_turn": 0.0},
        }
    )
    snapshot = gap_snapshot(cfg, math.radians(20.0), math.radians(12.0),
                            obstacle_distance_m=4.0, open_distance_m=25.0)
    _c, debug = LocalPlanner(cfg).plan(snapshot, 0.0, speed_mps=1.0, dt_s=DT)
    assert debug.chosen_bearing_rad is not None
    assert debug.chosen_bearing_rad > math.radians(10.0)


def test_turn_weight_resists_switching(cfg_from) -> None:
    # Two symmetric gaps: without hysteresis the planner alternates between
    # them every frame and the aircraft shudders down the middle.
    cfg = cfg_from(
        {
            "obstacles": {"n_bins": 16},
            "occupancy": {"n_bins": 16},
            "local_planner": {"w_goal": 0.0, "w_clearance": 1.0, "w_turn": 5.0},
        }
    )
    bearings = np.asarray(open_snapshot(cfg).bearings)
    distances = np.full(bearings.shape, 1.0, dtype=np.float32)
    left = np.argmin(np.abs(bearings + math.radians(20.0)))
    right = np.argmin(np.abs(bearings - math.radians(20.0)))
    distances[left] = 25.0
    distances[right] = 25.0
    from sim.scenarios import _snapshot

    snapshot = _snapshot(cfg, 0, distances, 0.0)

    planner = LocalPlanner(cfg)
    first = planner.plan(snapshot, 0.0, speed_mps=1.0, dt_s=DT)[1].chosen_index
    for _ in range(5):
        again = planner.plan(snapshot, 0.0, speed_mps=1.0, dt_s=DT)[1].chosen_index
        assert again == first, "planner oscillated between two equal gaps"


def test_previous_bearing_is_forgotten_on_stop(cfg16: Config) -> None:
    planner = LocalPlanner(cfg16)
    planner.plan(open_snapshot(cfg16), math.radians(25.0), speed_mps=1.0, dt_s=DT)
    assert planner.previous_bearing is not None
    planner.plan(unknown_snapshot(cfg16), 0.0, speed_mps=1.0, dt_s=DT)
    assert planner.previous_bearing is None


def test_reset_clears_state(cfg16: Config) -> None:
    planner = LocalPlanner(cfg16)
    planner.plan(open_snapshot(cfg16), math.radians(25.0), speed_mps=1.0, dt_s=DT)
    planner.reset()
    assert planner.previous_bearing is None


# --------------------------------------------------------------------------
# Command shape
# --------------------------------------------------------------------------


def test_it_never_commands_lateral_velocity(cfg16: Config) -> None:
    # A quad could fly sideways, but sideways is where the forward-facing
    # camera is not looking, and unknown is impassable.
    planner = LocalPlanner(cfg16)
    for goal_deg in (-30.0, -10.0, 0.0, 10.0, 30.0):
        command, _ = planner.plan(
            open_snapshot(cfg16), math.radians(goal_deg), speed_mps=2.0, dt_s=DT
        )
        assert command.vy == 0.0
        assert command.vz == 0.0


def test_forward_speed_falls_off_with_turn_angle(cfg16: Config) -> None:
    planner = LocalPlanner(cfg16)
    straight, _ = planner.plan(open_snapshot(cfg16), 0.0, speed_mps=2.0, dt_s=DT)
    planner.reset()
    turning, debug = planner.plan(
        open_snapshot(cfg16), math.radians(30.0), speed_mps=2.0, dt_s=DT
    )
    assert turning.vx < straight.vx
    assert turning.vx == pytest.approx(2.0 * math.cos(debug.chosen_bearing_rad), rel=1e-6)


def test_yaw_rate_is_a_proportional_correction(cfg16: Config) -> None:
    planner = LocalPlanner(cfg16)
    command, debug = planner.plan(open_snapshot(cfg16), math.radians(10.0), 2.0, DT)
    expected = debug.chosen_bearing_rad / cfg16.local_planner.yaw_align_time_s
    assert command.yaw_rate == pytest.approx(expected)


def test_a_small_heading_error_does_not_command_full_yaw(cfg16: Config) -> None:
    # Regression guard: using dt as the gain made a sub-bin-width error
    # command the maximum yaw rate.
    planner = LocalPlanner(cfg16)
    command, debug = planner.plan(open_snapshot(cfg16), 0.0, speed_mps=2.0, dt_s=DT)
    assert abs(debug.chosen_bearing_rad) < math.radians(5.0)
    assert abs(command.yaw_rate) < 0.3 * cfg16.local_planner.max_yaw_rate_rad_s


def test_yaw_rate_is_capped_by_config(cfg_from) -> None:
    cfg = cfg_from(
        {
            "obstacles": {"n_bins": 16},
            "occupancy": {"n_bins": 16},
            "local_planner": {"yaw_align_time_s": 0.05},
        }
    )
    planner = LocalPlanner(cfg)
    command, _ = planner.plan(open_snapshot(cfg), math.radians(30.0), speed_mps=2.0, dt_s=DT)
    assert abs(command.yaw_rate) == pytest.approx(cfg.local_planner.max_yaw_rate_rad_s)


def test_yaw_rate_sign_follows_the_chosen_bearing(cfg16: Config) -> None:
    planner = LocalPlanner(cfg16)
    right, debug_r = planner.plan(open_snapshot(cfg16), math.radians(25.0), 2.0, DT)
    planner.reset()
    left, debug_l = planner.plan(open_snapshot(cfg16), math.radians(-25.0), 2.0, DT)
    assert debug_r.chosen_bearing_rad > 0 and right.yaw_rate > 0
    assert debug_l.chosen_bearing_rad < 0 and left.yaw_rate < 0


def test_command_carries_the_snapshot_timestamp(cfg16: Config) -> None:
    planner = LocalPlanner(cfg16)
    command, _ = planner.plan(open_snapshot(cfg16, t_ns=4242), 0.0, speed_mps=1.0, dt_s=DT)
    assert command.t_ns == 4242


def test_reason_is_human_readable(cfg16: Config) -> None:
    planner = LocalPlanner(cfg16)
    command, _ = planner.plan(open_snapshot(cfg16), math.radians(20.0), 2.0, DT)
    assert "heading" in command.reason
    assert "clearance" in command.reason
    assert "candidates" in command.reason


def test_planning_is_deterministic(cfg16: Config) -> None:
    def run() -> list[tuple[float, float]]:
        planner = LocalPlanner(cfg16)
        out = []
        for tick in range(10):
            command, _ = planner.plan(
                open_snapshot(cfg16, t_ns=tick), math.radians(15.0), 2.0, DT
            )
            out.append((command.vx, command.yaw_rate))
        return out

    assert run() == run()
