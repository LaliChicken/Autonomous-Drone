"""MAVLink client, driven with real pymavlink messages over a fake transport.

Only the socket is faked. Every message here is a genuine pymavlink message
object with genuine field semantics, so the field mappings, the type mask, and
the frame constant are all really being checked -- not a test's idea of them.
"""

from __future__ import annotations

import math

import pytest
from pymavlink import mavutil
from pymavlink.dialects.v20 import ardupilotmega as mav2

from config import Config
from sources.mavlink_client import (
    ORIENTATION_DOWN,
    ORIENTATION_FORWARD,
    TYPE_MASK_VELOCITY_YAW_RATE,
    AttitudeSample,
    MavlinkClient,
    MavlinkError,
    lerp_angle,
    wrap_pi,
)

NS = 1_000_000_000


class FakeMav:
    """Records outgoing messages instead of writing them to a socket."""

    def __init__(self) -> None:
        self.commands: list[tuple] = []
        self.setpoints: list[tuple] = []

    def command_long_send(self, *args) -> None:
        self.commands.append(args)

    def set_position_target_local_ned_send(self, *args) -> None:
        self.setpoints.append(args)


class FakeLink:
    """Stands in for mavutil.mavlink_connection. The hardware boundary only."""

    def __init__(self, heartbeat: bool = True) -> None:
        self.mav = FakeMav()
        self.target_system = 1
        self.target_component = 1
        self._heartbeat = heartbeat
        self.inbox: list = []

    def wait_heartbeat(self, timeout: float | None = None):
        return object() if self._heartbeat else None

    def recv_match(self, blocking: bool = False, timeout: float | None = None):
        return self.inbox.pop(0) if self.inbox else None


@pytest.fixture
def link() -> FakeLink:
    return FakeLink()


@pytest.fixture
def client(cfg: Config, link: FakeLink) -> MavlinkClient:
    return MavlinkClient(cfg, connection=link)


# --------------------------------------------------------------------------
# Angles
# --------------------------------------------------------------------------


def test_wrap_pi() -> None:
    # Range is [-pi, pi): the half-open end is at +pi, so an angle landing
    # exactly there comes back as -pi. Same value, different representative.
    assert wrap_pi(0.0) == pytest.approx(0.0)
    assert wrap_pi(math.pi / 2) == pytest.approx(math.pi / 2)
    assert wrap_pi(3 * math.pi) == pytest.approx(-math.pi)
    assert wrap_pi(-3 * math.pi) == pytest.approx(-math.pi)
    assert wrap_pi(math.radians(190.0)) == pytest.approx(math.radians(-170.0))


def test_lerp_angle_takes_the_short_way_round() -> None:
    # The bug this exists to prevent: a naive lerp between 179 and -179 gives
    # 0 -- pointing exactly backwards while the aircraft is doing nothing
    # unusual.
    result = lerp_angle(math.radians(179.0), math.radians(-179.0), 0.5)
    assert abs(result) > math.radians(179.0)
    naive = (math.radians(179.0) + math.radians(-179.0)) / 2.0
    assert naive == pytest.approx(0.0)


def test_lerp_angle_endpoints() -> None:
    assert lerp_angle(0.2, 0.8, 0.0) == pytest.approx(0.2)
    assert lerp_angle(0.2, 0.8, 1.0) == pytest.approx(0.8)
    assert lerp_angle(0.2, 0.8, 0.5) == pytest.approx(0.5)


# --------------------------------------------------------------------------
# Connect and stream rates
# --------------------------------------------------------------------------


def test_connect_requests_every_configured_stream(client: MavlinkClient, link: FakeLink,
                                                  cfg: Config) -> None:
    client.connect()
    assert len(link.mav.commands) == len(cfg.mavlink.stream_rates())
    for args in link.mav.commands:
        assert args[2] == mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL


def test_stream_intervals_match_the_configured_rates(client: MavlinkClient, link: FakeLink,
                                                     cfg: Config) -> None:
    client.connect()
    # args: (target_sys, target_comp, cmd, confirm, msg_id, interval_us, ...)
    sent = {int(args[4]): args[5] for args in link.mav.commands}
    for name, rate_hz in cfg.mavlink.stream_rates().items():
        message_id = getattr(mavutil.mavlink, f"MAVLINK_MSG_ID_{name}")
        assert message_id in sent, f"{name} was never requested"
        assert sent[message_id] == pytest.approx(1_000_000.0 / rate_hz, rel=1e-6)


def test_attitude_is_requested_at_50hz(client: MavlinkClient, link: FakeLink) -> None:
    client.connect()
    sent = {int(args[4]): args[5] for args in link.mav.commands}
    assert sent[mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE] == pytest.approx(20_000.0)


def test_a_missing_heartbeat_is_an_error(cfg: Config) -> None:
    client = MavlinkClient(cfg, connection=FakeLink(heartbeat=False))
    with pytest.raises(MavlinkError, match="no heartbeat"):
        client.connect()


def test_requesting_streams_without_a_connection(cfg: Config) -> None:
    with pytest.raises(MavlinkError, match="not connected"):
        MavlinkClient(cfg).request_streams()


def test_an_unknown_stream_name_is_rejected(cfg_from) -> None:
    cfg = cfg_from({"mavlink": {"stream_rates_hz": {"NOT_A_MESSAGE": 5}}})
    client = MavlinkClient(cfg, connection=FakeLink())
    with pytest.raises(MavlinkError, match="unknown MAVLink message name"):
        client.connect()


# --------------------------------------------------------------------------
# Message handling
# --------------------------------------------------------------------------


def test_attitude_updates_state(client: MavlinkClient) -> None:
    client.handle(mav2.MAVLink_attitude_message(0, 0.1, -0.2, 1.5, 0.0, 0.0, 0.0), t_ns=NS)
    state = client.state(t_ns=NS)
    assert state.roll == pytest.approx(0.1)
    assert state.pitch == pytest.approx(-0.2)
    assert state.yaw == pytest.approx(1.5)


def test_local_position_updates_velocity(client: MavlinkClient) -> None:
    client.handle(
        mav2.MAVLink_local_position_ned_message(0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0), t_ns=NS
    )
    assert client.state(t_ns=NS).vel_ned == pytest.approx((4.0, 5.0, 6.0))


def test_downward_rangefinder_becomes_agl(client: MavlinkClient) -> None:
    message = mav2.MAVLink_distance_sensor_message(
        0, 10, 1200, 250, 0, 1, ORIENTATION_DOWN, 0
    )
    client.handle(message, t_ns=NS)
    assert client.state(t_ns=NS).agl_m == pytest.approx(2.5)


def test_forward_rangefinder_is_kept_separately(client: MavlinkClient) -> None:
    message = mav2.MAVLink_distance_sensor_message(
        0, 10, 1200, 400, 0, 2, ORIENTATION_FORWARD, 0
    )
    client.handle(message, t_ns=NS)
    assert client.forward_range_m == pytest.approx(4.0)
    assert client.state(t_ns=NS).agl_m is None


def test_an_out_of_range_rangefinder_reading_is_discarded(client: MavlinkClient) -> None:
    # A TFmini reporting its maximum means "nothing seen", not "the ground is
    # 12 m away". Storing it would put a phantom floor under the aircraft.
    good = mav2.MAVLink_distance_sensor_message(0, 10, 1200, 250, 0, 1, ORIENTATION_DOWN, 0)
    client.handle(good, t_ns=NS)
    assert client.state(t_ns=NS).agl_m == pytest.approx(2.5)

    beyond = mav2.MAVLink_distance_sensor_message(
        0, 10, 1200, 1500, 0, 1, ORIENTATION_DOWN, 0
    )
    client.handle(beyond, t_ns=2 * NS)
    assert client.state(t_ns=2 * NS).agl_m is None


def test_heartbeat_sets_armed_and_mode(client: MavlinkClient) -> None:
    armed = mav2.MAVLink_heartbeat_message(
        mavutil.mavlink.MAV_TYPE_QUADROTOR,
        mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA,
        mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
        | mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        4,  # ArduCopter GUIDED
        mavutil.mavlink.MAV_STATE_ACTIVE,
        3,
    )
    client.handle(armed, t_ns=NS)
    state = client.state(t_ns=NS)
    assert state.armed is True
    assert state.mode == "GUIDED"


def test_heartbeat_reports_disarmed(client: MavlinkClient) -> None:
    disarmed = mav2.MAVLink_heartbeat_message(
        mavutil.mavlink.MAV_TYPE_QUADROTOR,
        mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        0,
        mavutil.mavlink.MAV_STATE_STANDBY,
        3,
    )
    client.handle(disarmed, t_ns=NS)
    assert client.state(t_ns=NS).armed is False


def _sys_status(present: int, enabled: int, health: int):
    return mav2.MAVLink_sys_status_message(
        present, enabled, health, 500, 12000, -1, 50, 0, 0, 0, 0, 0, 0
    )


def test_sys_status_reports_ekf_health(client: MavlinkClient) -> None:
    bit = mavutil.mavlink.MAV_SYS_STATUS_AHRS
    client.handle(_sys_status(bit, bit, bit), t_ns=NS)
    assert client.state(t_ns=NS).ekf_ok is True

    client.handle(_sys_status(bit, bit, 0), t_ns=2 * NS)
    assert client.state(t_ns=2 * NS).ekf_ok is False


def test_a_sensor_the_fc_does_not_have_is_not_a_failure(client: MavlinkClient) -> None:
    # Absent-and-disabled must not read as unhealthy, or every FC without the
    # sensor would look broken.
    client.handle(_sys_status(0, 0, 0), t_ns=NS)
    assert client.state(t_ns=NS).ekf_ok is True


def test_rc_channels_are_mapped(client: MavlinkClient) -> None:
    message = mav2.MAVLink_rc_channels_message(
        0, 8, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100,
    )
    client.handle(message, t_ns=NS)
    assert client.rc_channel(1) == 1100
    assert client.rc_channel(5) == 1500
    assert client.rc_channel(9) is None  # zero means "not provided"
    assert client.state(t_ns=NS).rc[8] == 1800


def test_state_is_a_snapshot_not_a_view(client: MavlinkClient) -> None:
    message = mav2.MAVLink_rc_channels_message(
        0, 1, 1500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100
    )
    client.handle(message, t_ns=NS)
    state = client.state(t_ns=NS)
    state.rc[1] = 9999
    assert client.rc_channel(1) == 1500


def test_message_counts_and_link_age(client: MavlinkClient) -> None:
    assert client.link_age_s(t_ns=NS) == math.inf
    client.handle(mav2.MAVLink_attitude_message(0, 0, 0, 0, 0, 0, 0), t_ns=NS)
    assert client.message_counts()["ATTITUDE"] == 1
    assert client.link_age_s(t_ns=NS + NS // 2) == pytest.approx(0.5)


def test_an_unknown_message_is_ignored(client: MavlinkClient) -> None:
    client.handle(mav2.MAVLink_vfr_hud_message(1, 2, 3, 4, 5, 6), t_ns=NS)
    assert client.message_counts()["VFR_HUD"] == 1


# --------------------------------------------------------------------------
# attitude_at
# --------------------------------------------------------------------------


def push_attitude(client: MavlinkClient, t_ns: int, yaw: float, roll: float = 0.0) -> None:
    client.handle(mav2.MAVLink_attitude_message(0, roll, 0.0, yaw, 0, 0, 0), t_ns=t_ns)


def test_attitude_at_on_an_empty_buffer(client: MavlinkClient) -> None:
    assert client.attitude_at(NS) is None


def test_attitude_at_interpolates(client: MavlinkClient) -> None:
    push_attitude(client, 1_000_000_000, yaw=0.0)
    push_attitude(client, 1_020_000_000, yaw=0.2)
    sample = client.attitude_at(1_010_000_000)
    assert isinstance(sample, AttitudeSample)
    assert sample.yaw == pytest.approx(0.1)
    assert sample.t_ns == 1_010_000_000


def test_attitude_at_interpolates_across_the_wrap(client: MavlinkClient) -> None:
    push_attitude(client, 1_000_000_000, yaw=math.radians(179.0))
    push_attitude(client, 1_020_000_000, yaw=math.radians(-179.0))
    sample = client.attitude_at(1_010_000_000)
    assert sample is not None
    assert abs(sample.yaw) > math.radians(179.0), "interpolated the long way round"


def test_attitude_at_picks_the_right_bracket(client: MavlinkClient) -> None:
    for index in range(20):
        push_attitude(client, 1_000_000_000 + index * 20_000_000, yaw=index * 0.01)
    sample = client.attitude_at(1_000_000_000 + 10 * 20_000_000)
    assert sample is not None
    assert sample.yaw == pytest.approx(0.10)


def test_attitude_at_clamps_within_the_extrapolation_limit(client: MavlinkClient,
                                                           cfg: Config) -> None:
    push_attitude(client, 1_000_000_000, yaw=0.4)
    limit_ns = int(cfg.mavlink.attitude_max_extrapolation_s * NS)
    just_after = client.attitude_at(1_000_000_000 + limit_ns - 1)
    assert just_after is not None
    assert just_after.yaw == pytest.approx(0.4)


def test_attitude_at_refuses_to_extrapolate_far(client: MavlinkClient, cfg: Config) -> None:
    # A frame timestamped before the link came up has no attitude. Inventing
    # one silently mis-projects every obstacle in that frame.
    push_attitude(client, 1_000_000_000, yaw=0.4)
    limit_ns = int(cfg.mavlink.attitude_max_extrapolation_s * NS)
    assert client.attitude_at(1_000_000_000 + limit_ns + NS) is None
    assert client.attitude_at(1_000_000_000 - limit_ns - NS) is None


def test_attitude_buffer_is_bounded(cfg_from) -> None:
    cfg = cfg_from({"mavlink": {"attitude_buffer_len": 4}})
    client = MavlinkClient(cfg, connection=FakeLink())
    for index in range(50):
        push_attitude(client, 1_000_000_000 + index * 10_000_000, yaw=index * 0.001)
    # Only the last four remain, so anything older is outside the buffer.
    assert client.attitude_at(1_000_000_000) is None
    assert client.attitude_at(1_000_000_000 + 49 * 10_000_000) is not None


# --------------------------------------------------------------------------
# Sending setpoints
# --------------------------------------------------------------------------


def test_send_velocity_body_uses_the_right_frame_and_mask(client: MavlinkClient,
                                                          link: FakeLink) -> None:
    client.send_velocity_body(1.0, 0.0, -0.5, 0.3)
    assert len(link.mav.setpoints) == 1
    args = link.mav.setpoints[0]
    # (time_boot_ms, sys, comp, frame, type_mask, x, y, z, vx, vy, vz, ...)
    assert args[3] == mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED
    assert args[4] == TYPE_MASK_VELOCITY_YAW_RATE
    assert (args[8], args[9], args[10]) == pytest.approx((1.0, 0.0, -0.5))
    assert args[15] == pytest.approx(0.3)


def test_the_type_mask_ignores_position_accel_and_absolute_yaw() -> None:
    mask = TYPE_MASK_VELOCITY_YAW_RATE
    assert mask & 0b111 == 0b111, "position must be ignored"
    assert mask & (0b111 << 6) == (0b111 << 6), "acceleration must be ignored"
    assert mask & (1 << 10), "absolute yaw must be ignored"
    assert not mask & (0b111 << 3), "velocity must be honoured"
    assert not mask & (1 << 11), "yaw rate must be honoured"
    assert mask == 1479


def test_position_fields_are_sent_as_zero(client: MavlinkClient, link: FakeLink) -> None:
    client.send_velocity_body(2.0, 0.0, 0.0, 0.0)
    args = link.mav.setpoints[0]
    assert (args[5], args[6], args[7]) == (0.0, 0.0, 0.0)


def test_send_command_adapts_a_planner_command(client: MavlinkClient, link: FakeLink) -> None:
    from sources.types import PlannerCommand

    client.send_command(
        PlannerCommand(vx=1.5, vy=0.0, vz=0.0, yaw_rate=-0.2, reason="gap", t_ns=0)
    )
    args = link.mav.setpoints[0]
    assert args[8] == pytest.approx(1.5)
    assert args[15] == pytest.approx(-0.2)


def test_sending_without_a_connection(cfg: Config) -> None:
    with pytest.raises(MavlinkError, match="not connected"):
        MavlinkClient(cfg).send_velocity_body(0.0, 0.0, 0.0, 0.0)


# --------------------------------------------------------------------------
# Reader thread
# --------------------------------------------------------------------------


def test_the_reader_thread_consumes_the_inbox(client: MavlinkClient, link: FakeLink) -> None:
    import time

    link.inbox = [
        mav2.MAVLink_attitude_message(0, 0.0, 0.0, float(index) / 100.0, 0, 0, 0)
        for index in range(5)
    ]
    client.start()
    try:
        deadline = time.monotonic() + 2.0
        while link.inbox and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        client.stop()
    assert link.inbox == []
    assert client.message_counts().get("ATTITUDE") == 5


def test_starting_the_reader_twice_is_refused(client: MavlinkClient) -> None:
    client.start()
    try:
        with pytest.raises(RuntimeError, match="already started"):
            client.start()
    finally:
        client.stop()


def test_a_throwing_link_does_not_kill_the_reader(cfg: Config) -> None:
    class ExplodingLink(FakeLink):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def recv_match(self, blocking: bool = False, timeout: float | None = None):
            self.calls += 1
            raise OSError("link down")

    import time

    link = ExplodingLink()
    client = MavlinkClient(cfg, connection=link)
    client.start()
    try:
        time.sleep(0.1)
    finally:
        client.stop()
    assert link.calls > 1, "reader gave up after the first failure"
