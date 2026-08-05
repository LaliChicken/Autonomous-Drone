"""Offboard loop: slew limiting, staleness, and fixed-rate emission."""

from __future__ import annotations

import time

import pytest

from config import Config
from control.offboard import OffboardLoop, Setpoint, slew
from sources.types import PlannerCommand

NS = 1_000_000_000


def command(vx: float = 0.0, yaw_rate: float = 0.0, reason: str = "test") -> PlannerCommand:
    return PlannerCommand(vx=vx, vy=0.0, vz=0.0, yaw_rate=yaw_rate, reason=reason, t_ns=0)


class Sink:
    """Records what the loop emitted. This is the injected gate stand-in."""

    def __init__(self) -> None:
        self.commands: list[PlannerCommand] = []

    def __call__(self, cmd: PlannerCommand) -> None:
        self.commands.append(cmd)

    @property
    def last(self) -> PlannerCommand:
        return self.commands[-1]


@pytest.fixture
def sink() -> Sink:
    return Sink()


@pytest.fixture
def loop(cfg: Config, sink: Sink) -> OffboardLoop:
    return OffboardLoop(cfg, emit=sink)


def tick_seconds(loop: OffboardLoop, seconds: float, start_ns: int = 0) -> None:
    """Tick at the configured rate for a wall-clock duration of sim time."""
    period_ns = int(loop.params.period_s * NS)
    for index in range(int(seconds / loop.params.period_s)):
        loop.tick(start_ns + index * period_ns)


# --------------------------------------------------------------------------
# slew
# --------------------------------------------------------------------------


def test_slew_reaches_the_target_when_within_reach() -> None:
    assert slew(0.0, 0.5, 1.0) == pytest.approx(0.5)
    assert slew(0.0, -0.5, 1.0) == pytest.approx(-0.5)


def test_slew_limits_the_step() -> None:
    assert slew(0.0, 10.0, 1.0) == pytest.approx(1.0)
    assert slew(0.0, -10.0, 1.0) == pytest.approx(-1.0)


def test_slew_with_zero_budget_does_not_move() -> None:
    assert slew(2.0, 5.0, 0.0) == pytest.approx(2.0)


def test_slew_rejects_a_negative_budget() -> None:
    with pytest.raises(ValueError, match="must be >= 0"):
        slew(0.0, 1.0, -1.0)


# --------------------------------------------------------------------------
# Emission
# --------------------------------------------------------------------------


def test_tick_emits_exactly_once(loop: OffboardLoop, sink: Sink) -> None:
    loop.tick(0)
    assert len(sink.commands) == 1
    assert loop.emitted_count == 1


def test_first_tick_with_no_command_is_zero(loop: OffboardLoop, sink: Sink) -> None:
    loop.tick(0)
    assert (sink.last.vx, sink.last.vy, sink.last.vz, sink.last.yaw_rate) == (0, 0, 0, 0)
    assert "STALE" in sink.last.reason


def test_a_submitted_command_is_ramped_toward(loop: OffboardLoop, sink: Sink) -> None:
    loop.submit(command(vx=2.0), t_ns=0)
    loop.tick(0)
    first = sink.last.vx
    assert 0.0 < first < 2.0, "should not jump straight to the target"
    loop.tick(int(0.05 * NS))
    assert sink.last.vx > first


def test_the_ramp_respects_max_accel(cfg: Config, sink: Sink) -> None:
    loop = OffboardLoop(cfg, emit=sink)
    loop.submit(command(vx=5.0), t_ns=0)
    period = cfg.offboard.period_s
    previous = 0.0
    for index in range(10):
        loop.tick(int(index * period * NS))
        step = sink.last.vx - previous
        assert step <= cfg.offboard.max_accel_mps2 * period + 1e-9
        previous = sink.last.vx


def test_it_eventually_reaches_the_commanded_speed(cfg: Config, sink: Sink) -> None:
    loop = OffboardLoop(cfg, emit=sink)
    target = 2.0
    period_ns = int(cfg.offboard.period_s * NS)
    for index in range(200):
        # Re-submit every tick so the command never goes stale.
        loop.submit(command(vx=target), t_ns=index * period_ns)
        loop.tick(index * period_ns)
    assert sink.last.vx == pytest.approx(target)


def test_yaw_rate_is_slew_limited_too(cfg: Config, sink: Sink) -> None:
    loop = OffboardLoop(cfg, emit=sink)
    loop.submit(command(yaw_rate=1.0), t_ns=0)
    loop.tick(0)
    assert 0.0 < sink.last.yaw_rate <= cfg.offboard.max_yaw_accel_rad_s2 * cfg.offboard.period_s


def test_the_reason_is_carried_through(loop: OffboardLoop, sink: Sink) -> None:
    loop.submit(command(vx=1.0, reason="gap at +12deg"), t_ns=0)
    loop.tick(0)
    assert sink.last.reason == "gap at +12deg"


# --------------------------------------------------------------------------
# Staleness
# --------------------------------------------------------------------------


def test_a_stale_command_brakes_to_zero(cfg: Config, sink: Sink) -> None:
    loop = OffboardLoop(cfg, emit=sink)
    period_ns = int(cfg.offboard.period_s * NS)
    for index in range(100):
        loop.submit(command(vx=2.0), t_ns=index * period_ns)
        loop.tick(index * period_ns)
    assert sink.last.vx == pytest.approx(2.0)

    # Stop submitting. The loop keeps ticking.
    start = 100 * period_ns
    for index in range(200):
        loop.tick(start + index * period_ns)
    assert sink.last.vx == pytest.approx(0.0)
    assert "STALE" in sink.last.reason


def test_the_brake_is_slew_limited_not_a_jolt(cfg: Config, sink: Sink) -> None:
    # An emergency cut is the owned watchdog's job; this layer produces a
    # smooth stop so a couple of dropped frames do not upset the aircraft.
    loop = OffboardLoop(cfg, emit=sink)
    period_ns = int(cfg.offboard.period_s * NS)
    for index in range(100):
        loop.submit(command(vx=2.0), t_ns=index * period_ns)
        loop.tick(index * period_ns)

    start = 100 * period_ns
    previous = sink.last.vx
    for index in range(20):
        loop.tick(start + index * period_ns)
        assert previous - sink.last.vx <= cfg.offboard.max_accel_mps2 * cfg.offboard.period_s + 1e-9
        previous = sink.last.vx
    assert previous > 0.0, "should still be braking, not already stopped"


def test_a_command_just_inside_the_timeout_is_still_fresh(cfg: Config, sink: Sink) -> None:
    loop = OffboardLoop(cfg, emit=sink)
    loop.submit(command(vx=2.0), t_ns=0)
    just_inside = int((cfg.offboard.command_timeout_s - 0.01) * NS)
    loop.tick(just_inside)
    assert "STALE" not in sink.last.reason


def test_a_command_past_the_timeout_is_stale(cfg: Config, sink: Sink) -> None:
    loop = OffboardLoop(cfg, emit=sink)
    loop.submit(command(vx=2.0), t_ns=0)
    past = int((cfg.offboard.command_timeout_s + 0.01) * NS)
    loop.tick(past)
    assert "STALE" in sink.last.reason


def test_freshness_is_measured_from_submission_not_frame_time(cfg: Config, sink: Sink) -> None:
    # The command's own t_ns is when the frame was captured, already tens of
    # ms old by the time a plan comes out of it. Counting that as staleness
    # would brake for no reason.
    loop = OffboardLoop(cfg, emit=sink)
    old_frame = PlannerCommand(
        vx=2.0, vy=0.0, vz=0.0, yaw_rate=0.0, reason="from an old frame", t_ns=0
    )
    now = 10 * NS
    loop.submit(old_frame, t_ns=now)
    loop.tick(now)
    assert "STALE" not in sink.last.reason
    assert sink.last.vx > 0.0


def test_a_fresh_command_resumes_after_a_stale_period(cfg: Config, sink: Sink) -> None:
    loop = OffboardLoop(cfg, emit=sink)
    period_ns = int(cfg.offboard.period_s * NS)
    for index in range(50):
        loop.tick(index * period_ns)
    assert sink.last.vx == 0.0

    resume = 50 * period_ns
    for index in range(50):
        loop.submit(command(vx=1.0), t_ns=resume + index * period_ns)
        loop.tick(resume + index * period_ns)
    assert sink.last.vx == pytest.approx(1.0)


# --------------------------------------------------------------------------
# dt handling
# --------------------------------------------------------------------------


def test_a_long_gap_does_not_licence_an_unbounded_step(cfg: Config, sink: Sink) -> None:
    # If a stalled loop could slew by accel * (whole gap), the acceleration
    # limit would stop meaning anything on exactly the tick that matters.
    loop = OffboardLoop(cfg, emit=sink)
    loop.tick(0)
    loop.submit(command(vx=5.0), t_ns=10 * NS)
    loop.tick(10 * NS)  # a ten-second gap
    max_step = cfg.offboard.max_accel_mps2 * cfg.offboard.period_s * 4.0
    assert sink.last.vx <= max_step + 1e-9


def test_a_repeated_timestamp_does_not_move_the_setpoint(cfg: Config, sink: Sink) -> None:
    loop = OffboardLoop(cfg, emit=sink)
    loop.submit(command(vx=2.0), t_ns=0)
    loop.tick(0)
    first = sink.last.vx
    loop.tick(0)
    assert sink.last.vx == pytest.approx(first)


# --------------------------------------------------------------------------
# Plumbing
# --------------------------------------------------------------------------


def test_setpoint_converts_to_a_command() -> None:
    setpoint = Setpoint(vx=1.0, vy=2.0, vz=3.0, yaw_rate=0.4)
    cmd = setpoint.as_command(t_ns=7, reason="why")
    assert (cmd.vx, cmd.vy, cmd.vz, cmd.yaw_rate) == (1.0, 2.0, 3.0, 0.4)
    assert cmd.t_ns == 7 and cmd.reason == "why"


def test_latest_returns_what_was_submitted(loop: OffboardLoop) -> None:
    assert loop.latest() is None
    cmd = command(vx=1.0)
    loop.submit(cmd, t_ns=0)
    assert loop.latest() is cmd


def test_tick_returns_what_it_emitted(loop: OffboardLoop, sink: Sink) -> None:
    returned = loop.tick(0)
    assert returned is sink.last


def test_the_threaded_loop_runs_at_roughly_the_configured_rate(cfg: Config, sink: Sink) -> None:
    loop = OffboardLoop(cfg, emit=sink)
    loop.start()
    try:
        time.sleep(0.6)
    finally:
        loop.stop()
    expected = cfg.offboard.rate_hz * 0.6
    # Generous bounds: this asserts the loop is running near its rate, not
    # that the OS scheduler is precise.
    assert 0.4 * expected <= len(sink.commands) <= 2.0 * expected


def test_starting_twice_is_refused(cfg: Config, sink: Sink) -> None:
    loop = OffboardLoop(cfg, emit=sink)
    loop.start()
    try:
        with pytest.raises(RuntimeError, match="already started"):
            loop.start()
    finally:
        loop.stop()


def test_stop_is_safe_without_start(cfg: Config, sink: Sink) -> None:
    OffboardLoop(cfg, emit=sink).stop()


def test_submission_is_thread_safe(cfg: Config, sink: Sink) -> None:
    import threading

    loop = OffboardLoop(cfg, emit=sink)
    stop = threading.Event()

    def submitter() -> None:
        index = 0
        while not stop.is_set():
            loop.submit(command(vx=1.0), t_ns=index)
            index += 1

    threads = [threading.Thread(target=submitter) for _ in range(4)]
    for thread in threads:
        thread.start()
    for index in range(200):
        loop.tick(index * int(cfg.offboard.period_s * NS))
    stop.set()
    for thread in threads:
        thread.join()
    assert loop.emitted_count == 200
