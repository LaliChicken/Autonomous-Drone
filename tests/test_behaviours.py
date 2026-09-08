"""Behaviour state machine. Runs entirely without SITL.

The guards are tested one at a time as the pure functions they are, then the
machine is tested for the sequencing it puts around them.
"""

from __future__ import annotations

import math

import pytest

from config import BehavioursConfig, Config
from planner.behaviours import (
    AUTONOMOUS_STATES,
    BehaviourInputs,
    BehaviourMachine,
    BehaviourState,
    GuardContext,
    Transition,
    guard_lost_flight_authority,
    guard_mission_cancelled,
    guard_mission_complete,
    guard_more_targets,
    guard_search_exhausted,
    guard_standoff_reached,
    guard_start_mission,
    guard_target_acquired,
    guard_target_lost,
)
from sim.scenarios import guided_state, make_target, open_snapshot, unknown_snapshot

NS = 1_000_000_000

#: Distinguishes "argument not supplied" from an explicit None, which for the
#: FC state is a meaningful value meaning "no telemetry at all".
UNSET = object()


def make_ctx(
    cfg: Config,
    *,
    t_ns: int = 10 * NS,
    state_entered_ns: int = 10 * NS,
    fc=UNSET,
    target=None,
    target_range_m=None,
    consecutive_target_ticks: int = 0,
    last_target_ns=None,
    mission_active: bool = True,
    more_targets: bool = False,
) -> GuardContext:
    return GuardContext(
        cfg=cfg.behaviours,
        t_ns=t_ns,
        state_entered_ns=state_entered_ns,
        fc=guided_state(t_ns) if fc is UNSET else fc,
        target=target,
        target_range_m=target_range_m,
        consecutive_target_ticks=consecutive_target_ticks,
        last_target_ns=last_target_ns,
        mission_active=mission_active,
        more_targets=more_targets,
    )


# ==========================================================================
# Guards, individually
# ==========================================================================


def test_guards_are_pure_functions(cfg: Config) -> None:
    # Calling a guard twice on the same context must give the same answer and
    # change nothing. That is what makes them reviewable in isolation.
    ctx = make_ctx(cfg, consecutive_target_ticks=5)
    assert guard_target_acquired(ctx) == guard_target_acquired(ctx)
    assert ctx.consecutive_target_ticks == 5


def test_guard_lost_flight_authority(cfg: Config) -> None:
    assert not guard_lost_flight_authority(make_ctx(cfg))
    assert guard_lost_flight_authority(make_ctx(cfg, fc=None))

    disarmed = guided_state()
    disarmed.armed = False
    assert guard_lost_flight_authority(make_ctx(cfg, fc=disarmed))

    unhealthy = guided_state()
    unhealthy.ekf_ok = False
    assert guard_lost_flight_authority(make_ctx(cfg, fc=unhealthy))

    manual = guided_state()
    manual.mode = "LOITER"
    assert guard_lost_flight_authority(make_ctx(cfg, fc=manual))


def test_guard_start_mission(cfg: Config) -> None:
    assert guard_start_mission(make_ctx(cfg))
    assert not guard_start_mission(make_ctx(cfg, mission_active=False))
    assert not guard_start_mission(make_ctx(cfg, fc=None))


def test_guard_target_acquired_needs_consecutive_sightings(cfg: Config) -> None:
    needed = cfg.behaviours.acquire_consecutive_ticks
    assert not guard_target_acquired(make_ctx(cfg, consecutive_target_ticks=needed - 1))
    assert guard_target_acquired(make_ctx(cfg, consecutive_target_ticks=needed))


def test_guard_search_exhausted(cfg: Config) -> None:
    timeout = cfg.behaviours.search_timeout_s
    entered = 0
    early = make_ctx(cfg, t_ns=int((timeout - 1) * NS), state_entered_ns=entered)
    late = make_ctx(cfg, t_ns=int((timeout + 1) * NS), state_entered_ns=entered)
    assert not guard_search_exhausted(early)
    assert guard_search_exhausted(late)


def test_guard_target_lost(cfg: Config) -> None:
    timeout = cfg.behaviours.lost_target_timeout_s
    recent = make_ctx(cfg, t_ns=10 * NS, last_target_ns=int(10 * NS - 0.1 * NS))
    stale = make_ctx(cfg, t_ns=10 * NS, last_target_ns=int(10 * NS - (timeout + 1) * NS))
    assert not guard_target_lost(recent)
    assert guard_target_lost(stale)


def test_guard_target_lost_when_never_seen(cfg: Config) -> None:
    # Never seen is infinitely stale, not zero seconds ago.
    ctx = make_ctx(cfg, last_target_ns=None)
    assert ctx.time_since_target_s == math.inf
    assert guard_target_lost(ctx)


def test_guard_standoff_reached(cfg: Config) -> None:
    standoff = cfg.behaviours.standoff_m
    tolerance = cfg.behaviours.arrive_tolerance_m
    assert guard_standoff_reached(make_ctx(cfg, target_range_m=standoff))
    assert guard_standoff_reached(make_ctx(cfg, target_range_m=standoff + tolerance))
    assert not guard_standoff_reached(make_ctx(cfg, target_range_m=standoff + tolerance + 0.5))


def test_guard_standoff_reached_without_a_range(cfg: Config) -> None:
    # No range is not "close enough".
    assert not guard_standoff_reached(make_ctx(cfg, target_range_m=None))


def test_guard_more_targets_and_complete_are_complementary(cfg: Config) -> None:
    with_more = make_ctx(cfg, more_targets=True)
    without = make_ctx(cfg, more_targets=False)
    assert guard_more_targets(with_more) and not guard_mission_complete(with_more)
    assert not guard_more_targets(without) and guard_mission_complete(without)


def test_guard_mission_cancelled(cfg: Config) -> None:
    assert guard_mission_cancelled(make_ctx(cfg, mission_active=False))
    assert not guard_mission_cancelled(make_ctx(cfg, mission_active=True))


def test_guard_context_is_frozen(cfg: Config) -> None:
    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        make_ctx(cfg).t_ns = 0  # type: ignore[misc]


def test_guard_config_is_the_behaviours_section(cfg: Config) -> None:
    assert isinstance(make_ctx(cfg).cfg, BehavioursConfig)


# ==========================================================================
# Machine
# ==========================================================================


def tick(machine: BehaviourMachine, cfg: Config, t_ns: int, **kwargs):
    inputs = BehaviourInputs(
        t_ns=t_ns,
        snapshot=kwargs.pop("snapshot", open_snapshot(cfg, t_ns)),
        fc=kwargs.pop("fc", guided_state(t_ns)),
        **kwargs,
    )
    return machine.tick(inputs)


def test_starts_idle(cfg: Config) -> None:
    machine = BehaviourMachine(cfg=cfg)
    assert machine.state is BehaviourState.IDLE


def test_idle_commands_nothing(cfg: Config) -> None:
    machine = BehaviourMachine(cfg=cfg)
    command = tick(machine, cfg, 0, mission_active=False)
    assert (command.vx, command.vy, command.vz, command.yaw_rate) == (0.0, 0.0, 0.0, 0.0)
    assert "IDLE" in command.reason


def test_idle_to_search_on_mission_start(cfg: Config) -> None:
    machine = BehaviourMachine(cfg=cfg)
    tick(machine, cfg, 0, mission_active=True)
    assert machine.state is BehaviourState.SEARCH


def test_search_yaw_scans(cfg: Config) -> None:
    machine = BehaviourMachine(cfg=cfg)
    command = tick(machine, cfg, 0, mission_active=True)
    assert command.vx == 0.0
    assert abs(command.yaw_rate) == pytest.approx(cfg.behaviours.search_yaw_rate_rad_s)
    assert "SEARCH" in command.reason


def test_search_to_approach_after_enough_sightings(cfg: Config) -> None:
    machine = BehaviourMachine(cfg=cfg)
    tick(machine, cfg, 0, mission_active=True)
    assert machine.state is BehaviourState.SEARCH
    target = make_target(math.radians(10.0), range_m=8.0)
    for index in range(cfg.behaviours.acquire_consecutive_ticks):
        tick(machine, cfg, (index + 1) * 50_000_000, target=target)
    assert machine.state is BehaviourState.APPROACH


def test_a_single_flicker_does_not_trigger_approach(cfg: Config) -> None:
    machine = BehaviourMachine(cfg=cfg)
    tick(machine, cfg, 0, mission_active=True)
    target = make_target(math.radians(10.0), range_m=8.0)
    tick(machine, cfg, 50_000_000, target=target)
    tick(machine, cfg, 100_000_000, target=None)
    tick(machine, cfg, 150_000_000, target=target)
    assert machine.state is BehaviourState.SEARCH


def test_search_times_out_to_rtl(cfg: Config) -> None:
    machine = BehaviourMachine(cfg=cfg)
    tick(machine, cfg, 0, mission_active=True)
    tick(machine, cfg, int((cfg.behaviours.search_timeout_s + 1) * NS))
    assert machine.state is BehaviourState.RTL


def test_approach_to_arrive_at_standoff(cfg: Config) -> None:
    machine = BehaviourMachine(cfg=cfg)
    machine.state = BehaviourState.APPROACH
    close = make_target(0.0, range_m=cfg.behaviours.standoff_m)
    command = tick(machine, cfg, NS, target=close)
    assert machine.state is BehaviourState.ARRIVE
    assert command.vx == 0.0
    assert "ARRIVE" in command.reason


def test_arrive_holds_still(cfg: Config) -> None:
    machine = BehaviourMachine(cfg=cfg)
    machine.state = BehaviourState.ARRIVE
    command = tick(machine, cfg, NS, target=make_target(0.0, cfg.behaviours.standoff_m),
                   more_targets=False)
    # It transitions onward, but the command for the arriving tick is a hold.
    assert command.vx == 0.0 and command.yaw_rate == 0.0


def test_arrive_to_search_when_more_targets(cfg: Config) -> None:
    machine = BehaviourMachine(cfg=cfg)
    machine.state = BehaviourState.ARRIVE
    tick(machine, cfg, NS, more_targets=True)
    assert machine.state is BehaviourState.SEARCH


def test_arrive_to_rtl_when_done(cfg: Config) -> None:
    machine = BehaviourMachine(cfg=cfg)
    machine.state = BehaviourState.ARRIVE
    tick(machine, cfg, NS, more_targets=False)
    assert machine.state is BehaviourState.RTL


def test_approach_back_to_search_when_target_lost(cfg: Config) -> None:
    machine = BehaviourMachine(cfg=cfg)
    tick(machine, cfg, 0, mission_active=True)
    target = make_target(math.radians(5.0), range_m=9.0)
    for index in range(cfg.behaviours.acquire_consecutive_ticks):
        tick(machine, cfg, (index + 1) * 50_000_000, target=target)
    assert machine.state is BehaviourState.APPROACH
    lost_at = int((cfg.behaviours.lost_target_timeout_s + 1) * NS)
    tick(machine, cfg, lost_at, target=None)
    assert machine.state is BehaviourState.SEARCH


# --------------------------------------------------------------------------
# Flight authority outranks everything
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "state",
    [BehaviourState.SEARCH, BehaviourState.APPROACH, BehaviourState.ARRIVE, BehaviourState.RTL],
)
def test_losing_guided_drops_to_idle_from_any_state(cfg: Config, state) -> None:
    machine = BehaviourMachine(cfg=cfg)
    machine.state = state
    manual = guided_state()
    manual.mode = "LOITER"
    command = tick(machine, cfg, NS, fc=manual)
    assert machine.state is BehaviourState.IDLE
    assert command.vx == 0.0


def test_disarming_drops_to_idle(cfg: Config) -> None:
    machine = BehaviourMachine(cfg=cfg)
    machine.state = BehaviourState.APPROACH
    disarmed = guided_state()
    disarmed.armed = False
    tick(machine, cfg, NS, fc=disarmed)
    assert machine.state is BehaviourState.IDLE


def test_cancelling_the_mission_drops_to_idle(cfg: Config) -> None:
    machine = BehaviourMachine(cfg=cfg)
    machine.state = BehaviourState.SEARCH
    tick(machine, cfg, NS, mission_active=False)
    assert machine.state is BehaviourState.IDLE


# --------------------------------------------------------------------------
# The speed clamp
# --------------------------------------------------------------------------


def test_clamp_speed_caps_at_the_autonomy_limit(cfg: Config) -> None:
    machine = BehaviourMachine(cfg=cfg)
    limit = cfg.behaviours.autonomy_speed_max_mps
    assert machine.clamp_speed(limit + 5.0) == pytest.approx(limit)
    assert machine.clamp_speed(0.5) == pytest.approx(0.5)
    assert machine.clamp_speed(-9.0) == pytest.approx(limit)


def test_no_command_ever_exceeds_the_autonomy_speed(cfg: Config) -> None:
    # The clamp is applied to every command, not only on state entry: a clamp
    # that fires only at the entry tick does nothing about the tick after it.
    machine = BehaviourMachine(cfg=cfg)
    limit = cfg.behaviours.autonomy_speed_max_mps
    tick(machine, cfg, 0, mission_active=True)
    target = make_target(math.radians(8.0), range_m=20.0)
    for index in range(60):
        command = tick(
            machine,
            cfg,
            (index + 1) * 50_000_000,
            target=target,
            speed_mps=99.0,  # ask for something absurd every single tick
        )
        assert abs(command.vx) <= limit + 1e-9
        assert abs(command.vy) <= limit + 1e-9


def test_autonomous_states_are_the_ones_that_move(cfg: Config) -> None:
    assert BehaviourState.IDLE not in AUTONOMOUS_STATES
    assert BehaviourState.SEARCH in AUTONOMOUS_STATES
    assert BehaviourState.APPROACH in AUTONOMOUS_STATES


# --------------------------------------------------------------------------
# Safety in APPROACH
# --------------------------------------------------------------------------


def test_approach_without_a_snapshot_holds(cfg: Config) -> None:
    machine = BehaviourMachine(cfg=cfg)
    machine.state = BehaviourState.APPROACH
    command = machine.tick(
        BehaviourInputs(
            t_ns=NS,
            snapshot=None,
            target=make_target(0.0, range_m=9.0),
            fc=guided_state(),
        )
    )
    assert command.vx == 0.0
    assert "no occupancy snapshot" in command.reason


def test_approach_into_unknown_space_stops(cfg: Config) -> None:
    machine = BehaviourMachine(cfg=cfg)
    machine.state = BehaviourState.APPROACH
    command = tick(
        machine,
        cfg,
        NS,
        snapshot=unknown_snapshot(cfg, NS),
        target=make_target(0.0, range_m=9.0),
    )
    assert command.vx == 0.0
    assert "STOP" in command.reason


# --------------------------------------------------------------------------
# Transition logging
# --------------------------------------------------------------------------


def test_every_transition_is_recorded_with_a_reason(cfg: Config) -> None:
    machine = BehaviourMachine(cfg=cfg)
    tick(machine, cfg, 0, mission_active=True)
    tick(machine, cfg, int((cfg.behaviours.search_timeout_s + 1) * NS))
    assert len(machine.transitions) == 2
    for record in machine.transitions:
        assert isinstance(record, Transition)
        assert record.guard.startswith("guard_")
        assert record.reason
        assert record.from_state is not record.to_state


def test_transition_callback_fires(cfg: Config) -> None:
    seen: list[Transition] = []
    machine = BehaviourMachine(cfg=cfg, on_transition=seen.append)
    tick(machine, cfg, 0, mission_active=True)
    assert len(seen) == 1
    assert seen[0].to_state is BehaviourState.SEARCH
    assert seen[0].guard == "guard_start_mission"


def test_transition_serialises_for_the_flight_log(cfg: Config) -> None:
    import json

    machine = BehaviourMachine(cfg=cfg)
    tick(machine, cfg, 0, mission_active=True)
    payload = json.loads(json.dumps(machine.transitions[0].to_dict()))
    assert payload["from"] == "IDLE"
    assert payload["to"] == "SEARCH"


def test_no_transition_is_logged_when_nothing_changes(cfg: Config) -> None:
    machine = BehaviourMachine(cfg=cfg)
    tick(machine, cfg, 0, mission_active=True)
    before = len(machine.transitions)
    tick(machine, cfg, 50_000_000)
    assert len(machine.transitions) == before


def test_reset_returns_to_idle(cfg: Config) -> None:
    machine = BehaviourMachine(cfg=cfg)
    tick(machine, cfg, 0, mission_active=True)
    machine.reset()
    assert machine.state is BehaviourState.IDLE
    assert machine.transitions == []


def test_the_machine_never_reads_a_clock(cfg: Config) -> None:
    def run() -> list[tuple[str, float, float]]:
        machine = BehaviourMachine(cfg=cfg)
        out = []
        target = make_target(math.radians(12.0), range_m=6.0)
        for index in range(40):
            command = tick(
                machine, cfg, index * 50_000_000, target=target if index > 5 else None
            )
            out.append((machine.state.value, command.vx, command.yaw_rate))
        return out

    assert run() == run()
