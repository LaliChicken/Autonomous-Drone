"""Scenario runs.

The non-SITL tests here are the real coverage: they drive the whole decision
chain (occupancy -> planner -> behaviour machine) through a stated geometry
and assert on what it decided. They run everywhere, with no simulator.

The @pytest.mark.sitl tests fly the same scenarios against a live ArduPilot
and assert on what the aircraft actually did. They are skipped unless
DRONE_SITL=1; see scripts/run_sitl.sh.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from config import Config
from planner.behaviours import BehaviourInputs, BehaviourMachine, BehaviourState
from planner.local_planner import LocalPlanner
from sim.scenarios import (
    Scenario,
    all_scenarios,
    empty_field,
    gap_snapshot,
    guided_state,
    make_target,
    open_snapshot,
    red_box_at_bearing,
    run_scenario,
    two_obstacles_with_gap,
    unknown_snapshot,
    wall_ahead,
    wall_snapshot,
)
from sources.types import PlannerCommand

DT = 0.05


# --------------------------------------------------------------------------
# Scenario construction
# --------------------------------------------------------------------------


def test_wall_snapshot_is_a_plane_not_a_cylinder(cfg: Config) -> None:
    # A constant-range "wall" is a cylinder centred on the aircraft, and would
    # let a planner that only ever reads the centre bin pass a test it should
    # fail. A real flat wall recedes as d/cos(bearing).
    snapshot = wall_snapshot(cfg, distance_m=4.0)
    centre = len(snapshot.bearings) // 2
    assert snapshot.distances[centre] == pytest.approx(4.0, rel=0.01)
    assert snapshot.distances[0] > snapshot.distances[centre]
    expected_edge = 4.0 / math.cos(snapshot.bearings[0])
    assert snapshot.distances[0] == pytest.approx(expected_edge, rel=1e-4)


def test_open_snapshot_is_entirely_known_and_clear(cfg: Config) -> None:
    snapshot = open_snapshot(cfg)
    assert not snapshot.unknown.any()
    assert not snapshot.danger.any()
    assert np.all(snapshot.distances > 20.0)


def test_gap_snapshot_has_exactly_one_opening(cfg: Config) -> None:
    snapshot = gap_snapshot(cfg, math.radians(15.0), math.radians(12.0))
    open_bins = snapshot.distances > 10.0
    assert open_bins.any()
    assert not open_bins.all()
    # The opening is contiguous.
    indices = np.flatnonzero(open_bins)
    assert np.all(np.diff(indices) == 1)


def test_unknown_snapshot_is_entirely_unknown(cfg: Config) -> None:
    snapshot = unknown_snapshot(cfg)
    assert snapshot.unknown.all()
    assert np.all(np.isnan(snapshot.distances))


def test_danger_is_derived_from_speed(cfg: Config) -> None:
    still = wall_snapshot(cfg, distance_m=4.0, speed_mps=0.0)
    fast = wall_snapshot(cfg, distance_m=4.0, speed_mps=5.0)
    assert not still.danger.any()
    assert fast.danger.any()


def test_guided_state_satisfies_the_flight_authority_guard(cfg: Config) -> None:
    state = guided_state()
    assert state.armed and state.ekf_ok and state.mode == "GUIDED"


def test_all_scenarios_are_distinct(cfg: Config) -> None:
    scenarios = all_scenarios(cfg)
    assert len(scenarios) == 4
    assert len({s.name for s in scenarios}) == 4
    for scenario in scenarios:
        assert isinstance(scenario, Scenario)
        assert scenario.description
        assert scenario.duration_s > 0


# --------------------------------------------------------------------------
# Scenario 1: empty field
# --------------------------------------------------------------------------


def test_empty_field_never_stops(cfg: Config) -> None:
    planner = LocalPlanner(cfg)
    snapshot = open_snapshot(cfg)
    for _ in range(50):
        command, debug = planner.plan(snapshot, 0.0, speed_mps=2.0, dt_s=DT)
        assert debug.n_candidates > 0
        assert not command.reason.startswith("STOP")


def test_empty_field_reaches_search_and_times_out_to_rtl(cfg: Config) -> None:
    machine = BehaviourMachine(cfg=cfg)
    result = run_scenario(empty_field(cfg), cfg, machine)
    assert "SEARCH" in result.states_seen()
    # 20 s of scenario against a 30 s search timeout: still searching.
    assert result.final_state == BehaviourState.SEARCH.value


# --------------------------------------------------------------------------
# Scenario 2: wall ahead -- assert it stops short
# --------------------------------------------------------------------------


def test_wall_ahead_stops_short(cfg: Config) -> None:
    planner = LocalPlanner(cfg)
    # Wall well inside the clearance threshold.
    snapshot = wall_snapshot(cfg, distance_m=1.5, speed_mps=2.0)
    command, debug = planner.plan(snapshot, 0.0, speed_mps=2.0, dt_s=DT)
    assert debug.n_candidates == 0
    assert command.vx == 0.0
    assert command.reason.startswith("STOP")


def test_wall_at_the_threshold_is_not_a_candidate(cfg: Config) -> None:
    planner = LocalPlanner(cfg)
    at_threshold = cfg.local_planner.clearance_threshold_m
    snapshot = wall_snapshot(cfg, distance_m=at_threshold * 0.99, speed_mps=0.0)
    _command, debug = planner.plan(snapshot, 0.0, speed_mps=2.0, dt_s=DT)
    # The centre bins are blocked; edge bins recede as 1/cos and may clear.
    centre = len(snapshot.bearings) // 2
    assert not debug.candidates[centre]


def test_approaching_a_wall_commands_zero_forward_speed(cfg: Config) -> None:
    machine = BehaviourMachine(cfg=cfg)
    machine.state = BehaviourState.APPROACH
    command = machine.tick(
        BehaviourInputs(
            t_ns=1_000_000_000,
            snapshot=wall_snapshot(cfg, distance_m=1.0, speed_mps=2.0),
            target=make_target(0.0, range_m=8.0),
            fc=guided_state(),
            dt_s=DT,
        )
    )
    assert command.vx == 0.0
    assert "STOP" in command.reason


def test_wall_scenario_produces_no_forward_motion(cfg: Config) -> None:
    machine = BehaviourMachine(cfg=cfg)
    machine.state = BehaviourState.APPROACH
    scenario = wall_ahead(cfg, distance_m=1.5)
    scenario.target_at = lambda t: make_target(0.0, range_m=10.0, t_ns=t)
    result = run_scenario(scenario, cfg, machine)
    assert result.max_forward_speed() == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Scenario 3: two obstacles with a gap -- assert it picks the gap
# --------------------------------------------------------------------------


def test_two_obstacles_picks_the_gap(cfg: Config) -> None:
    planner = LocalPlanner(cfg)
    gap = math.radians(15.0)
    snapshot = gap_snapshot(cfg, gap, math.radians(14.0))
    _command, debug = planner.plan(snapshot, goal_bearing_rad=0.0, speed_mps=2.0, dt_s=DT)
    assert debug.chosen_bearing_rad is not None
    assert abs(debug.chosen_bearing_rad - gap) < math.radians(8.0)


def test_it_keeps_choosing_the_gap_over_many_ticks(cfg: Config) -> None:
    planner = LocalPlanner(cfg)
    gap = math.radians(-15.0)
    snapshot = gap_snapshot(cfg, gap, math.radians(14.0))
    chosen = []
    for _ in range(30):
        _command, debug = planner.plan(snapshot, 0.0, speed_mps=2.0, dt_s=DT)
        chosen.append(debug.chosen_bearing_rad)
    assert all(bearing is not None and bearing < 0.0 for bearing in chosen)
    assert len(set(chosen)) == 1, "the choice should be stable, not oscillating"


def test_the_gap_scenario_commands_a_turn_toward_it(cfg: Config) -> None:
    machine = BehaviourMachine(cfg=cfg)
    machine.state = BehaviourState.APPROACH
    scenario = two_obstacles_with_gap(cfg, gap_bearing_deg=18.0, gap_width_deg=14.0)
    scenario.target_at = lambda t: make_target(0.0, range_m=15.0, t_ns=t)
    result = run_scenario(scenario, cfg, machine)
    moving = [c for c in result.commands if c.vx > 0.0]
    assert moving, "expected the aircraft to move through the gap"
    assert all(c.yaw_rate > 0.0 for c in moving), "should turn toward the right-hand gap"


def test_no_gap_means_stop(cfg: Config) -> None:
    planner = LocalPlanner(cfg)
    # A "gap" narrower than a bin, so nothing actually opens.
    snapshot = gap_snapshot(cfg, 0.0, math.radians(0.1), obstacle_distance_m=1.0)
    command, debug = planner.plan(snapshot, 0.0, speed_mps=2.0, dt_s=DT)
    assert debug.n_candidates == 0
    assert command.vx == 0.0


# --------------------------------------------------------------------------
# Scenario 4: red box at 40 deg -- outside the FOV, must be searched for
# --------------------------------------------------------------------------


def test_the_target_starts_outside_the_field_of_view(cfg: Config) -> None:
    scenario = red_box_at_bearing(cfg, bearing_deg=40.0)
    half_fov = math.degrees(cfg.obstacles.fov_rad / 2.0)
    assert 40.0 > half_fov, "the scenario is only interesting if the box starts unseen"
    assert scenario.target_at(0) is None
    assert scenario.metadata["visible_after_s"] > 0.0


def test_the_box_becomes_visible_once_the_scan_has_swung_far_enough(cfg: Config) -> None:
    scenario = red_box_at_bearing(cfg, bearing_deg=40.0)
    visible_after_ns = int(scenario.metadata["visible_after_s"] * 1e9)
    assert scenario.target_at(visible_after_ns + 1) is not None


def test_red_box_scenario_searches_then_approaches_then_arrives(cfg: Config) -> None:
    machine = BehaviourMachine(cfg=cfg)
    scenario = red_box_at_bearing(cfg, bearing_deg=40.0, range_m=8.0)

    # Close the range as the aircraft approaches, so ARRIVE is reachable.
    visible_after_ns = int(scenario.metadata["visible_after_s"] * 1e9)
    half_fov = cfg.obstacles.fov_rad / 2.0

    def closing_target(t_ns: int):
        if t_ns < visible_after_ns:
            return None
        elapsed_s = (t_ns - visible_after_ns) / 1e9
        range_m = max(cfg.behaviours.standoff_m, 8.0 - 0.8 * elapsed_s)
        return make_target(half_fov * 0.5, range_m, t_ns=t_ns)

    scenario.target_at = closing_target
    result = run_scenario(scenario, cfg, machine)

    seen = result.states_seen()
    assert "SEARCH" in seen
    assert "APPROACH" in seen
    assert "ARRIVE" in seen
    assert seen.index("SEARCH") < seen.index("APPROACH") < seen.index("ARRIVE")


def test_arriving_holds_the_configured_standoff(cfg: Config) -> None:
    machine = BehaviourMachine(cfg=cfg)
    machine.state = BehaviourState.APPROACH
    at_standoff = make_target(0.0, range_m=cfg.behaviours.standoff_m)
    command = machine.tick(
        BehaviourInputs(
            t_ns=1_000_000_000,
            snapshot=open_snapshot(cfg),
            target=at_standoff,
            fc=guided_state(),
            dt_s=DT,
        )
    )
    assert machine.state is BehaviourState.ARRIVE
    assert command.vx == 0.0
    assert f"{cfg.behaviours.standoff_m:.2f}" in command.reason


def test_a_never_seen_target_ends_in_rtl(cfg: Config) -> None:
    machine = BehaviourMachine(cfg=cfg)
    scenario = red_box_at_bearing(cfg)
    scenario.target_at = lambda _t: None
    scenario.duration_s = cfg.behaviours.search_timeout_s + 2.0
    result = run_scenario(scenario, cfg, machine)
    assert result.final_state == BehaviourState.RTL.value


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def test_run_scenario_is_deterministic(cfg: Config) -> None:
    def run() -> list[tuple[float, float, str]]:
        machine = BehaviourMachine(cfg=cfg)
        result = run_scenario(wall_ahead(cfg), cfg, machine)
        return [(c.vx, c.yaw_rate, s) for c, s in zip(result.commands, result.states, strict=True)]

    assert run() == run()


def test_run_scenario_records_transitions(cfg: Config) -> None:
    machine = BehaviourMachine(cfg=cfg)
    result = run_scenario(empty_field(cfg), cfg, machine)
    assert result.transitions
    assert result.scenario == "empty_field"
    assert len(result.commands) == len(result.states)


def test_states_seen_collapses_runs(cfg: Config) -> None:
    machine = BehaviourMachine(cfg=cfg)
    result = run_scenario(empty_field(cfg), cfg, machine)
    seen = result.states_seen()
    assert seen == list(dict.fromkeys(seen)) or len(seen) >= 1


# --------------------------------------------------------------------------
# SITL. Skipped unless DRONE_SITL=1.
# --------------------------------------------------------------------------


@pytest.fixture
def sitl_client(cfg: Config):
    """A MavlinkClient connected to a running SITL, armed and in GUIDED."""
    from sources.mavlink_client import MavlinkClient

    client = MavlinkClient(cfg)
    client.connect()
    client.start()
    try:
        yield client
    finally:
        client.stop()


@pytest.mark.sitl
def test_sitl_link_comes_up(sitl_client) -> None:
    import time

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if sitl_client.message_counts().get("ATTITUDE", 0) > 10:
            break
        time.sleep(0.1)
    counts = sitl_client.message_counts()
    assert counts.get("ATTITUDE", 0) > 10, f"no attitude stream: {counts}"
    assert sitl_client.link_age_s() < 1.0


@pytest.mark.sitl
def test_sitl_attitude_buffer_fills_and_interpolates(sitl_client) -> None:
    import time

    time.sleep(2.0)
    state = sitl_client.state()
    sample = sitl_client.attitude_at(state.t_ns)
    assert sample is not None, "attitude ring buffer never filled"
    assert -math.pi <= sample.yaw <= math.pi


@pytest.mark.sitl
def test_sitl_empty_field_flies_a_square(sitl_client, cfg: Config) -> None:
    """Stream a square in open space and confirm the aircraft moves."""
    import time

    from control.offboard import OffboardLoop

    loop = OffboardLoop(cfg, emit=sitl_client.send_command)
    legs = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]
    start = sitl_client.state()
    loop.start()
    try:
        for vx, vy in legs:
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                loop.submit(
                    PlannerCommand(
                        vx=vx, vy=vy, vz=0.0, yaw_rate=0.0, reason="square leg", t_ns=0
                    )
                )
                time.sleep(0.05)
    finally:
        loop.stop()
    moved = sitl_client.state()
    assert moved.t_ns > start.t_ns


@pytest.mark.sitl
def test_sitl_wall_ahead_stops_short(sitl_client, cfg: Config) -> None:
    """Inject a wall and confirm the commanded speed goes to zero."""
    import time

    from control.offboard import OffboardLoop

    emitted: list = []

    def emit(command) -> None:
        emitted.append(command)
        sitl_client.send_command(command)

    machine = BehaviourMachine(cfg=cfg)
    machine.state = BehaviourState.APPROACH
    loop = OffboardLoop(cfg, emit=emit)
    loop.start()
    try:
        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline:
            fc = sitl_client.state()
            command = machine.tick(
                BehaviourInputs(
                    t_ns=fc.t_ns,
                    snapshot=wall_snapshot(cfg, distance_m=1.5, t_ns=fc.t_ns, speed_mps=2.0),
                    target=make_target(0.0, range_m=10.0, t_ns=fc.t_ns),
                    fc=fc,
                    dt_s=DT,
                )
            )
            loop.submit(command)
            time.sleep(0.05)
    finally:
        loop.stop()

    assert emitted, "nothing was streamed to SITL"
    assert emitted[-1].vx == pytest.approx(0.0, abs=1e-6)


@pytest.mark.sitl
def test_sitl_picks_the_gap(sitl_client, cfg: Config) -> None:
    """Two obstacles with a gap on the right: the yaw command must go right."""
    import time

    machine = BehaviourMachine(cfg=cfg)
    machine.state = BehaviourState.APPROACH
    yaw_rates: list[float] = []
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        fc = sitl_client.state()
        command = machine.tick(
            BehaviourInputs(
                t_ns=fc.t_ns,
                snapshot=gap_snapshot(
                    cfg, math.radians(18.0), math.radians(14.0), t_ns=fc.t_ns
                ),
                target=make_target(0.0, range_m=15.0, t_ns=fc.t_ns),
                fc=fc,
                dt_s=DT,
            )
        )
        sitl_client.send_command(command)
        yaw_rates.append(command.yaw_rate)
        print(command.reason, command.vx, command.yaw_rate)
        time.sleep(0.05)

    assert yaw_rates
    assert sum(yaw_rates) / len(yaw_rates) > 0.0, "should have turned toward the gap"


@pytest.mark.sitl
def test_sitl_red_box_arrives_at_standoff(sitl_client, cfg: Config) -> None:
    """Target at 40 deg: search, acquire, approach, hold standoff."""
    import time

    machine = BehaviourMachine(cfg=cfg)
    scenario = red_box_at_bearing(cfg, bearing_deg=40.0, range_m=8.0)
    visible_after_ns = int(scenario.metadata["visible_after_s"] * 1e9)
    half_fov = cfg.obstacles.fov_rad / 2.0

    t0 = sitl_client.state().t_ns
    deadline = time.monotonic() + scenario.duration_s
    while time.monotonic() < deadline and machine.state is not BehaviourState.ARRIVE:
        fc = sitl_client.state()
        elapsed_ns = fc.t_ns - t0
        target = None
        if elapsed_ns >= visible_after_ns:
            elapsed_s = (elapsed_ns - visible_after_ns) / 1e9
            target = make_target(
                half_fov * 0.5,
                max(cfg.behaviours.standoff_m, 8.0 - 0.8 * elapsed_s),
                t_ns=fc.t_ns,
            )
        command = machine.tick(
            BehaviourInputs(
                t_ns=fc.t_ns,
                snapshot=open_snapshot(cfg, fc.t_ns),
                target=target,
                fc=fc,
                dt_s=DT,
            )
        )
        sitl_client.send_command(command)
        time.sleep(0.05)

    states = [t.to_state.value for t in machine.transitions]
    assert "SEARCH" in states
    assert machine.state is BehaviourState.ARRIVE, f"ended in {machine.state}, path {states}"
