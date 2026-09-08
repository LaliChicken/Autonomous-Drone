"""VFH-lite: pick a heading from the occupancy snapshot, or stop.

Candidate headings are bins that are clear beyond ``clearance_threshold_m``
and not unknown. Each is scored::

    cost = w_goal * |angle_to_goal|
         + w_clearance / clearance
         + w_turn * |turn_rate|

and the cheapest wins. No viable candidate means STOP -- not "carry on and
hope", and not "pick the least bad bin". A frame with nothing passable is a
frame with no evidence that any direction is safe.

Two things about this implementation are worth arguing with.

**Candidates also exclude `danger`.** The brief says "clearance > threshold AND
not unknown". Danger is a third condition, and it is not implied by the first
two: at 2.5 m/s the danger distance is ~3.25 m while ``clearance_threshold_m``
is 2.5 m, so a bin at 3.0 m would pass the clearance test while being flagged
too close to stop in. Excluding it is a deliberate addition, made through
``world.occupancy.passable`` so the rule lives in one place.

**The aircraft yaws toward its chosen heading instead of translating along it.**
A quad can fly sideways, and flying sideways would be faster. It would also
mean moving into a direction the only forward-facing camera cannot see, where
the occupancy map is by definition unknown -- and unknown is impassable. So
forward speed is scaled by ``cos(theta)`` and the aircraft turns to face where
it is going. Never translate where you cannot see.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from sources.types import OccupancySnapshot, PlannerCommand
from world.occupancy import passable

if TYPE_CHECKING:
    from config import Config, LocalPlannerConfig


@dataclass(frozen=True)
class PlanDebug:
    """Why the planner chose what it chose. Logged with every tick."""

    chosen_index: int | None
    chosen_bearing_rad: float | None
    costs: np.ndarray
    candidates: np.ndarray
    n_candidates: int
    n_unknown: int
    n_danger: int
    n_too_close: int


def wrap_pi(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap to [-pi, pi). Used everywhere an angular difference is taken."""
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def stop_command(t_ns: int, reason: str) -> PlannerCommand:
    return PlannerCommand(vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0, reason=reason, t_ns=int(t_ns))


class LocalPlanner:
    """Stateful only in that it remembers its previous choice.

    That memory is the whole point of the turn-rate term: without it, two
    equally good gaps make the planner alternate between them every frame and
    the aircraft shudders down the middle. The state is one float, and
    ``reset()`` clears it, so replay is deterministic.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.params: LocalPlannerConfig = cfg.local_planner
        self._previous_bearing: float | None = None

    def reset(self) -> None:
        self._previous_bearing = None

    @property
    def previous_bearing(self) -> float | None:
        return self._previous_bearing

    def costs(
        self, snapshot: OccupancySnapshot, goal_bearing_rad: float, dt_s: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """(cost per bin, candidate mask). Cost is inf for non-candidates."""
        params = self.params
        bearings = np.asarray(snapshot.bearings, dtype=np.float64)
        distances = np.asarray(snapshot.distances, dtype=np.float64)

        candidates = passable(snapshot, params.clearance_threshold_m)

        angle_to_goal = np.abs(wrap_pi(bearings - goal_bearing_rad))
        # Floored so the reciprocal cannot blow up; a bin at the floor is
        # already far too close to be a candidate anyway.
        clearance = np.maximum(distances, params.min_clearance_for_cost_m)
        with np.errstate(invalid="ignore", divide="ignore"):
            clearance_cost = params.w_clearance / clearance

        # The turn term is hysteresis, so it needs something to be hysteretic
        # about: on the first plan after a reset or a STOP there is no previous
        # choice and no turn to penalise.
        #
        # It is normalised by max_yaw_rate rather than used raw. A raw rate
        # scales as 1/dt, which would make w_turn mean something different at
        # every loop rate -- at dt=0.05 the raw term outweighs the goal term
        # roughly tenfold and the planner simply refuses to turn. Dividing by
        # the maximum achievable rate leaves a dimensionless 0..1 penalty that
        # is comparable with the other two terms.
        if self._previous_bearing is None:
            turn_penalty = np.zeros_like(bearings)
        else:
            turn_rate = np.abs(wrap_pi(bearings - self._previous_bearing)) / max(dt_s, 1e-6)
            achievable = np.minimum(turn_rate, params.max_yaw_rate_rad_s)
            turn_penalty = achievable / params.max_yaw_rate_rad_s

        cost = (
            params.w_goal * angle_to_goal
            + np.nan_to_num(clearance_cost, nan=np.inf, posinf=np.inf)
            + params.w_turn * turn_penalty
        )
        cost = np.where(candidates, cost, np.inf)
        return cost, candidates

    def plan(
        self,
        snapshot: OccupancySnapshot,
        goal_bearing_rad: float,
        speed_mps: float,
        dt_s: float,
        t_ns: int | None = None,
    ) -> tuple[PlannerCommand, PlanDebug]:
        """Choose a heading and turn it into a body-frame command."""
        t_ns = int(snapshot.t_ns if t_ns is None else t_ns)
        cost, candidates = self.costs(snapshot, goal_bearing_rad, dt_s)

        distances = np.asarray(snapshot.distances, dtype=np.float64)
        with np.errstate(invalid="ignore"):
            too_close = np.asarray(distances <= self.params.clearance_threshold_m, dtype=bool)
        too_close &= ~snapshot.unknown

        debug_counts = {
            "n_unknown": int(np.count_nonzero(snapshot.unknown)),
            "n_danger": int(np.count_nonzero(snapshot.danger)),
            "n_too_close": int(np.count_nonzero(too_close)),
        }

        if not candidates.any():
            self._previous_bearing = None
            reason = (
                f"STOP: no viable heading ({debug_counts['n_unknown']} unknown, "
                f"{debug_counts['n_too_close']} too close, "
                f"{debug_counts['n_danger']} danger of {len(snapshot.bearings)} bins)"
            )
            return stop_command(t_ns, reason), PlanDebug(
                chosen_index=None,
                chosen_bearing_rad=None,
                costs=cost,
                candidates=candidates,
                n_candidates=0,
                **debug_counts,
            )

        index = int(np.argmin(cost))
        bearing = float(snapshot.bearings[index])
        clearance = float(snapshot.distances[index])
        self._previous_bearing = bearing

        # Proportional correction over a real time constant, capped by
        # max_yaw_rate. Dividing by dt instead would make a 2 deg heading error
        # -- less than one bin width -- command full yaw rate, and would change
        # the gain silently whenever the loop rate changed.
        yaw_rate = float(
            np.clip(
                bearing / self.params.yaw_align_time_s,
                -self.params.max_yaw_rate_rad_s,
                self.params.max_yaw_rate_rad_s,
            )
        )
        # cos(theta) so the aircraft slows as it turns, and never commands
        # forward speed into a heading it is not yet facing.
        forward = float(max(0.0, math.cos(bearing)) * speed_mps)

        reason = (
            f"heading {math.degrees(bearing):+.1f}deg, clearance {clearance:.2f}m, "
            f"goal {math.degrees(goal_bearing_rad):+.1f}deg, "
            f"{int(np.count_nonzero(candidates))}/{len(snapshot.bearings)} candidates"
        )
        command = PlannerCommand(
            vx=forward,
            vy=0.0,
            vz=0.0,
            yaw_rate=yaw_rate,
            reason=reason,
            t_ns=t_ns,
        )
        return command, PlanDebug(
            chosen_index=index,
            chosen_bearing_rad=bearing,
            costs=cost,
            candidates=candidates,
            n_candidates=int(np.count_nonzero(candidates)),
            **debug_counts,
        )
