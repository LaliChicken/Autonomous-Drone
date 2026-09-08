"""Mission state machine: IDLE -> SEARCH -> APPROACH -> ARRIVE -> next / RTL.

(The brief called this file ``behaviors.py``; the repo already had
``behaviours.py`` and the instruction was to fill existing files rather than
restructure, so this is it.)

Every transition guard is a pure function of a frozen ``GuardContext`` --
no clock, no I/O, no mutation, no reading of machine state. They are all
collected in ONE block below, marked for review, so the conditions under
which this aircraft changes what it is doing can be read in one sitting and
tested one at a time. The machine itself only sequences them.

Two invariants the machine enforces regardless of state:

* **Entry into any autonomous state clamps commanded speed** to
  ``behaviours.autonomy_speed_max_mps``. The clamp is applied to every command
  leaving the machine, not just at the entry tick, because "clamped on entry"
  is worthless if a later tick can exceed it.
* **Every transition is logged with its reason**, including the inputs that
  triggered it.

Actually commanding the vehicle is not this module's job. ``control/gate.py``
and ``control/watchdog.py`` are owned; this emits PlannerCommands and someone
else decides whether they are allowed out.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from perception.red_box import RedBoxDetection
from planner.local_planner import LocalPlanner, stop_command
from sources.types import FcState, OccupancySnapshot, PlannerCommand

if TYPE_CHECKING:
    from config import BehavioursConfig, Config

NS_PER_S = 1_000_000_000.0


class BehaviourState(StrEnum):
    IDLE = "IDLE"
    SEARCH = "SEARCH"
    APPROACH = "APPROACH"
    ARRIVE = "ARRIVE"
    RTL = "RTL"


#: States in which the aircraft moves under its own decisions. Entering any of
#: these clamps speed to behaviours.autonomy_speed_max_mps.
AUTONOMOUS_STATES = frozenset(
    {BehaviourState.SEARCH, BehaviourState.APPROACH, BehaviourState.ARRIVE, BehaviourState.RTL}
)


@dataclass(frozen=True)
class GuardContext:
    """Everything a transition guard is allowed to look at.

    Frozen and self-contained on purpose: a guard that could reach into the
    machine could depend on history the tests do not control, and would stop
    being a pure function of the situation.
    """

    cfg: BehavioursConfig
    t_ns: int
    state_entered_ns: int
    fc: FcState | None
    target: RedBoxDetection | None
    target_range_m: float | None
    consecutive_target_ticks: int
    last_target_ns: int | None
    mission_active: bool
    more_targets: bool

    @property
    def time_in_state_s(self) -> float:
        return max(0.0, (self.t_ns - self.state_entered_ns) / NS_PER_S)

    @property
    def time_since_target_s(self) -> float:
        """Seconds since the target was last seen; inf if it never has been."""
        if self.last_target_ns is None:
            return math.inf
        return max(0.0, (self.t_ns - self.last_target_ns) / NS_PER_S)

    @property
    def flyable(self) -> bool:
        """FC is in a state where autonomous commands make sense at all."""
        fc = self.fc
        return fc is not None and fc.armed and fc.ekf_ok and fc.mode.upper() == "GUIDED"


# ==========================================================================
# TRANSITION GUARDS -- pure functions, review block
#
# Each takes a GuardContext and returns a bool. No clock reads, no I/O, no
# mutation, no access to machine internals. Ordering between them is the
# machine's business, not theirs.
# ==========================================================================


def guard_lost_flight_authority(ctx: GuardContext) -> bool:
    """Disarmed, EKF unhappy, or no longer in GUIDED -> stop being autonomous."""
    return not ctx.flyable


def guard_start_mission(ctx: GuardContext) -> bool:
    """IDLE -> SEARCH: told to go, and the FC will accept guidance."""
    return ctx.mission_active and ctx.flyable


def guard_target_acquired(ctx: GuardContext) -> bool:
    """SEARCH -> APPROACH: seen for long enough to not be a flicker."""
    return ctx.consecutive_target_ticks >= ctx.cfg.acquire_consecutive_ticks


def guard_search_exhausted(ctx: GuardContext) -> bool:
    """SEARCH -> RTL: scanned for the whole timeout and found nothing."""
    return ctx.time_in_state_s >= ctx.cfg.search_timeout_s


def guard_target_lost(ctx: GuardContext) -> bool:
    """APPROACH -> SEARCH: target gone for longer than the tolerance."""
    return ctx.time_since_target_s >= ctx.cfg.lost_target_timeout_s


def guard_standoff_reached(ctx: GuardContext) -> bool:
    """APPROACH -> ARRIVE: within tolerance of the configured standoff."""
    if ctx.target_range_m is None:
        return False
    return ctx.target_range_m <= ctx.cfg.standoff_m + ctx.cfg.arrive_tolerance_m


def guard_more_targets(ctx: GuardContext) -> bool:
    """ARRIVE -> SEARCH: another target to go and find."""
    return ctx.more_targets


def guard_mission_complete(ctx: GuardContext) -> bool:
    """ARRIVE -> RTL: nothing left to do."""
    return not ctx.more_targets


def guard_mission_cancelled(ctx: GuardContext) -> bool:
    """Any autonomous state -> IDLE: the mission flag was cleared."""
    return not ctx.mission_active


# ==========================================================================
# END TRANSITION GUARDS
# ==========================================================================


@dataclass(frozen=True)
class Transition:
    """One state change, with the evidence for it."""

    t_ns: int
    from_state: BehaviourState
    to_state: BehaviourState
    guard: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "t_ns": self.t_ns,
            "from": self.from_state.value,
            "to": self.to_state.value,
            "guard": self.guard,
            "reason": self.reason,
        }


@dataclass
class BehaviourInputs:
    """One tick's worth of the world."""

    t_ns: int
    snapshot: OccupancySnapshot | None = None
    target: RedBoxDetection | None = None
    fc: FcState | None = None
    mission_active: bool = True
    more_targets: bool = False
    dt_s: float = 0.05
    speed_mps: float | None = None


@dataclass
class BehaviourMachine:
    """Sequences the guards and turns the current state into a command."""

    cfg: Config
    planner: LocalPlanner | None = None
    on_transition: Callable[[Transition], None] | None = None
    state: BehaviourState = BehaviourState.IDLE
    transitions: list[Transition] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.behaviours: BehavioursConfig = self.cfg.behaviours
        self.planner = self.planner or LocalPlanner(self.cfg)
        self._state_entered_ns = 0
        self._consecutive_target_ticks = 0
        self._last_target_ns: int | None = None
        self._search_direction = 1.0

    # -- lifecycle ---------------------------------------------------------

    def reset(self) -> None:
        self.state = BehaviourState.IDLE
        self.transitions.clear()
        self._state_entered_ns = 0
        self._consecutive_target_ticks = 0
        self._last_target_ns = None
        self._search_direction = 1.0
        if self.planner is not None:
            self.planner.reset()

    def clamp_speed(self, speed_mps: float) -> float:
        """The autonomy speed clamp.

        Applied to every command the machine emits rather than only on state
        entry: a clamp that only fires at the entry tick does nothing about
        the tick after it.
        """
        return float(min(abs(speed_mps), self.behaviours.autonomy_speed_max_mps))

    def context(self, inputs: BehaviourInputs) -> GuardContext:
        target_range = None
        if inputs.target is not None:
            target_range = inputs.target.best_range_m
        return GuardContext(
            cfg=self.behaviours,
            t_ns=inputs.t_ns,
            state_entered_ns=self._state_entered_ns,
            fc=inputs.fc,
            target=inputs.target,
            target_range_m=target_range,
            consecutive_target_ticks=self._consecutive_target_ticks,
            last_target_ns=self._last_target_ns,
            mission_active=inputs.mission_active,
            more_targets=inputs.more_targets,
        )

    def _transition(
        self, to_state: BehaviourState, guard: str, reason: str, t_ns: int
    ) -> None:
        if to_state == self.state:
            return
        record = Transition(
            t_ns=int(t_ns),
            from_state=self.state,
            to_state=to_state,
            guard=guard,
            reason=reason,
        )
        self.transitions.append(record)
        self.state = to_state
        self._state_entered_ns = int(t_ns)
        if to_state in (BehaviourState.SEARCH, BehaviourState.IDLE):
            self._consecutive_target_ticks = 0
        if self.planner is not None:
            self.planner.reset()
        if self.on_transition is not None:
            self.on_transition(record)

    # -- tick --------------------------------------------------------------

    def tick(self, inputs: BehaviourInputs) -> PlannerCommand:
        """Advance one step and return the command for this tick."""
        if inputs.target is not None:
            self._consecutive_target_ticks += 1
            self._last_target_ns = inputs.t_ns
        else:
            self._consecutive_target_ticks = 0

        ctx = self.context(inputs)
        self._step_state(ctx)
        return self._command(inputs, self.context(inputs))

    def _step_state(self, ctx: GuardContext) -> None:
        """Evaluate the guards for the current state, in priority order."""
        # Losing flight authority outranks everything else, in every state.
        if self.state is not BehaviourState.IDLE and guard_lost_flight_authority(ctx):
            mode = ctx.fc.mode if ctx.fc else "no FC state"
            armed = ctx.fc.armed if ctx.fc else False
            self._transition(
                BehaviourState.IDLE,
                "guard_lost_flight_authority",
                f"lost flight authority (mode={mode}, armed={armed})",
                ctx.t_ns,
            )
            return

        if self.state is not BehaviourState.IDLE and guard_mission_cancelled(ctx):
            self._transition(
                BehaviourState.IDLE, "guard_mission_cancelled", "mission cancelled", ctx.t_ns
            )
            return

        if self.state is BehaviourState.IDLE:
            if guard_start_mission(ctx):
                self._transition(
                    BehaviourState.SEARCH,
                    "guard_start_mission",
                    "mission active and FC accepting guidance",
                    ctx.t_ns,
                )
            return

        if self.state is BehaviourState.SEARCH:
            if guard_target_acquired(ctx):
                self._transition(
                    BehaviourState.APPROACH,
                    "guard_target_acquired",
                    f"target seen {ctx.consecutive_target_ticks} consecutive ticks",
                    ctx.t_ns,
                )
            elif guard_search_exhausted(ctx):
                self._transition(
                    BehaviourState.RTL,
                    "guard_search_exhausted",
                    f"no target after {ctx.time_in_state_s:.1f}s of scanning",
                    ctx.t_ns,
                )
            return

        if self.state is BehaviourState.APPROACH:
            if guard_standoff_reached(ctx):
                self._transition(
                    BehaviourState.ARRIVE,
                    "guard_standoff_reached",
                    f"range {ctx.target_range_m:.2f}m within standoff "
                    f"{ctx.cfg.standoff_m:.2f}m +/- {ctx.cfg.arrive_tolerance_m:.2f}m",
                    ctx.t_ns,
                )
            elif guard_target_lost(ctx):
                self._transition(
                    BehaviourState.SEARCH,
                    "guard_target_lost",
                    f"target unseen for {ctx.time_since_target_s:.1f}s",
                    ctx.t_ns,
                )
            return

        if self.state is BehaviourState.ARRIVE:
            if guard_more_targets(ctx):
                self._transition(
                    BehaviourState.SEARCH, "guard_more_targets", "another target queued", ctx.t_ns
                )
            elif guard_mission_complete(ctx):
                self._transition(
                    BehaviourState.RTL, "guard_mission_complete", "no targets remaining", ctx.t_ns
                )
            return

    # -- command generation ------------------------------------------------

    def _command(self, inputs: BehaviourInputs, ctx: GuardContext) -> PlannerCommand:
        t_ns = inputs.t_ns
        if self.state is BehaviourState.IDLE:
            return stop_command(t_ns, "IDLE: holding, no autonomous command")

        if self.state is BehaviourState.RTL:
            # The actual return is the FC's job; this stops commanding so
            # GUIDED brakes, and the owned gate decides what happens next.
            return stop_command(t_ns, "RTL: mission over, releasing guidance")

        if self.state is BehaviourState.SEARCH:
            yaw_rate = self._search_direction * self.behaviours.search_yaw_rate_rad_s
            return PlannerCommand(
                vx=0.0,
                vy=0.0,
                vz=0.0,
                yaw_rate=float(yaw_rate),
                reason=(
                    f"SEARCH: yaw scan at {math.degrees(yaw_rate):+.1f}deg/s, "
                    f"{ctx.time_in_state_s:.1f}s of {self.behaviours.search_timeout_s:.1f}s"
                ),
                t_ns=t_ns,
            )

        if self.state is BehaviourState.ARRIVE:
            range_text = (
                f"{ctx.target_range_m:.2f}m" if ctx.target_range_m is not None else "unknown"
            )
            return stop_command(
                t_ns,
                f"ARRIVE: holding standoff {self.behaviours.standoff_m:.2f}m (range {range_text})",
            )

        # APPROACH
        return self._approach_command(inputs, ctx)

    def _approach_command(self, inputs: BehaviourInputs, ctx: GuardContext) -> PlannerCommand:
        t_ns = inputs.t_ns
        speed = self.clamp_speed(
            self.behaviours.approach_speed_mps if inputs.speed_mps is None else inputs.speed_mps
        )
        goal_bearing = inputs.target.bearing_rad if inputs.target is not None else 0.0

        if inputs.snapshot is None:
            # No occupancy map is not a licence to fly blind.
            return stop_command(t_ns, "APPROACH: no occupancy snapshot, holding")

        assert self.planner is not None
        command, _debug = self.planner.plan(
            snapshot=inputs.snapshot,
            goal_bearing_rad=goal_bearing,
            speed_mps=speed,
            dt_s=inputs.dt_s,
            t_ns=t_ns,
        )
        clamped = self.clamp_speed(command.vx)
        return PlannerCommand(
            vx=clamped,
            vy=command.vy,
            vz=command.vz,
            yaw_rate=command.yaw_rate,
            reason=f"APPROACH: {command.reason}",
            t_ns=command.t_ns,
        )
