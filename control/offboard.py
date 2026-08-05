"""20 Hz command loop: latest PlannerCommand -> slew limit -> emit.

``emit`` is injected. ``control/gate.py`` and ``control/watchdog.py`` are
owned, and this module has no opinion about whether a command is *allowed* --
it produces a smooth, non-stale stream at a fixed rate and hands each one to
whatever was passed in. In tests that is a list; in flight it is the gate.

Staleness: if no fresh PlannerCommand has been submitted within
``offboard.command_timeout_s``, the target becomes zero velocity. The ramp to
zero is still slew-limited. That is deliberate for *this* layer -- a planner
that missed a few frames should produce a smooth brake, not a jolt that
upsets the aircraft. A genuine emergency cut is the owned watchdog's job, and
GUIDED brakes by itself after ~3 s of silence as a third backstop. Three
different mechanisms for three different failures.

The loop never reads a clock except in ``run()``. ``tick(t_ns)`` is a pure
step given a timestamp, so the whole slew behaviour is testable and replays
identically.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from sources.types import PlannerCommand

if TYPE_CHECKING:
    from config import Config, OffboardConfig

NS_PER_S = 1_000_000_000.0


class Emitter(Protocol):
    def __call__(self, command: PlannerCommand) -> None: ...


@dataclass(frozen=True)
class Setpoint:
    """The smoothed state the loop is currently commanding."""

    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw_rate: float = 0.0

    def as_command(self, t_ns: int, reason: str) -> PlannerCommand:
        return PlannerCommand(
            vx=self.vx,
            vy=self.vy,
            vz=self.vz,
            yaw_rate=self.yaw_rate,
            reason=reason,
            t_ns=int(t_ns),
        )


def slew(current: float, target: float, max_delta: float) -> float:
    """Move ``current`` toward ``target`` by at most ``max_delta``."""
    if max_delta < 0.0:
        raise ValueError(f"max_delta must be >= 0, got {max_delta}")
    delta = target - current
    if delta > max_delta:
        return current + max_delta
    if delta < -max_delta:
        return current - max_delta
    return target


class OffboardLoop:
    """Fixed-rate setpoint streamer with slew limiting and a stale-command guard."""

    def __init__(
        self,
        cfg: Config,
        emit: Emitter,
        clock: Callable[[], int] | None = None,
    ) -> None:
        self.cfg = cfg
        self.params: OffboardConfig = cfg.offboard
        self.emit = emit
        self.clock = clock or time.monotonic_ns
        self.setpoint = Setpoint()

        self._lock = threading.Lock()
        self._pending: PlannerCommand | None = None
        self._pending_ns: int | None = None
        self._last_tick_ns: int | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.emitted_count = 0

    # -- input side --------------------------------------------------------

    def submit(self, command: PlannerCommand, t_ns: int | None = None) -> None:
        """Hand the loop a new command. Safe to call from another thread.

        Freshness is measured from *submission*, not from ``command.t_ns``.
        The command's own timestamp is when the frame was captured, which is
        already tens of milliseconds old by the time a plan comes out of it;
        using it would count perception latency as staleness and brake for no
        reason.
        """
        with self._lock:
            self._pending = command
            self._pending_ns = int(self.clock() if t_ns is None else t_ns)

    def latest(self) -> PlannerCommand | None:
        with self._lock:
            return self._pending

    # -- step --------------------------------------------------------------

    def tick(self, t_ns: int) -> PlannerCommand:
        """Advance the setpoint one step and emit it. Returns what was sent."""
        with self._lock:
            pending = self._pending
            pending_ns = self._pending_ns

        dt_s = self._elapsed_s(t_ns)
        self._last_tick_ns = int(t_ns)

        stale = (
            pending is None
            or pending_ns is None
            or (t_ns - pending_ns) / NS_PER_S > self.params.command_timeout_s
        )

        if stale:
            target = Setpoint(0.0, 0.0, 0.0, 0.0)
            age_s = float("inf") if pending_ns is None else (t_ns - pending_ns) / NS_PER_S
            reason = (
                "STALE: no command in "
                f"{age_s:.2f}s (timeout {self.params.command_timeout_s:.2f}s), braking to zero"
                if pending is not None
                else "STALE: no command received, holding zero"
            )
        else:
            assert pending is not None
            target = Setpoint(pending.vx, pending.vy, pending.vz, pending.yaw_rate)
            reason = pending.reason

        max_dv = self.params.max_accel_mps2 * dt_s
        max_dyaw = self.params.max_yaw_accel_rad_s2 * dt_s
        self.setpoint = Setpoint(
            vx=slew(self.setpoint.vx, target.vx, max_dv),
            vy=slew(self.setpoint.vy, target.vy, max_dv),
            vz=slew(self.setpoint.vz, target.vz, max_dv),
            yaw_rate=slew(self.setpoint.yaw_rate, target.yaw_rate, max_dyaw),
        )

        command = self.setpoint.as_command(t_ns, reason)
        self.emit(command)
        self.emitted_count += 1
        return command

    def _elapsed_s(self, t_ns: int) -> float:
        """Seconds since the last tick, falling back to the nominal period.

        Clamped to one period on the first tick and to a sane maximum after a
        stall: a long gap must not licence an unbounded slew step, which would
        defeat the point of having an acceleration limit at all.
        """
        if self._last_tick_ns is None:
            return self.params.period_s
        dt_s = (t_ns - self._last_tick_ns) / NS_PER_S
        if dt_s <= 0.0:
            return 0.0
        return min(dt_s, self.params.period_s * 4.0)

    # -- threaded operation ------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("offboard loop already started")
        self._stop.clear()
        self._thread = threading.Thread(target=self.run, name="offboard", daemon=True)
        self._thread.start()

    def run(self) -> None:
        """Tick at ``offboard.rate_hz`` until stopped."""
        period = self.params.period_s
        next_tick = self.clock()
        while not self._stop.is_set():
            now = self.clock()
            self.tick(now)
            next_tick += int(period * NS_PER_S)
            sleep_s = (next_tick - self.clock()) / NS_PER_S
            if sleep_s > 0:
                self._stop.wait(sleep_s)
            else:
                # Fell behind; resynchronise rather than sprinting to catch up.
                next_tick = self.clock()

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            self._thread = None
