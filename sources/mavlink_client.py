"""MAVLink link to the flight controller: threaded reader, state, setpoints.

Requests its own stream rates on connect (``SET_MESSAGE_INTERVAL``) rather
than trusting whatever the FC happens to be sending, keeps a ring buffer of
attitudes so a frame can be matched to the attitude at its *capture* time, and
sends body-frame velocity + yaw-rate setpoints.

Attitude interpolation wraps. Yaw crosses +/-pi routinely and a naive lerp
between 179 deg and -179 deg gives 0 deg -- pointing exactly backwards, at the
moment the aircraft is doing nothing unusual. Every interpolation here goes
through the shortest angular difference.

The link object is injectable so the whole client can be tested against real
pymavlink messages without a flight controller. Only the transport is faked;
message construction, parsing, and every field mapping is the real thing.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pymavlink import mavutil

from sources.telemetry import RangeReading, TelemetrySample
from sources.types import FcState

if TYPE_CHECKING:
    from config import Config, MavlinkConfig

NS_PER_S = 1_000_000_000.0

#: DISTANCE_SENSOR orientations we care about (per the hardware notes).
ORIENTATION_FORWARD = mavutil.mavlink.MAV_SENSOR_ROTATION_NONE  # 0
ORIENTATION_DOWN = mavutil.mavlink.MAV_SENSOR_ROTATION_PITCH_270  # 25

#: Velocity + yaw-rate only: ignore position (bits 0-2), acceleration
#: (bits 6-8) and absolute yaw (bit 10), but honour yaw rate (bit 11 clear).
TYPE_MASK_VELOCITY_YAW_RATE = 0b0000_0101_1100_0111  # 1479


def wrap_pi(angle: float) -> float:
    """Wrap to [-pi, pi)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def lerp_angle(a: float, b: float, fraction: float) -> float:
    """Interpolate between two angles the short way round.

    ``lerp_angle(3.1, -3.1, 0.5)`` is near pi, not near zero. Getting this
    wrong yields an attitude pointing backwards exactly when the aircraft
    happens to be flying near due south.
    """
    return wrap_pi(a + wrap_pi(b - a) * fraction)


@dataclass(frozen=True)
class AttitudeSample:
    t_ns: int
    roll: float
    pitch: float
    yaw: float


class MavlinkError(RuntimeError):
    """Raised when the link cannot be established or used."""


class MavlinkClient:
    """Threaded MAVLink reader plus setpoint sender.

    All shared state is behind one lock. ``state()`` returns an immutable
    ``FcState`` snapshot rather than a live view, so a caller cannot observe a
    half-updated aircraft.
    """

    def __init__(
        self,
        cfg: Config,
        connection: Any | None = None,
        clock: Callable[[], int] | None = None,
    ) -> None:
        self.cfg = cfg
        self.params: MavlinkConfig = cfg.mavlink
        self.clock = clock or time.monotonic_ns
        self.connection = connection

        self._lock = threading.Lock()
        self._attitudes: deque[AttitudeSample] = deque(maxlen=self.params.attitude_buffer_len)
        self._roll = 0.0
        self._pitch = 0.0
        self._yaw = 0.0
        self._vel_ned: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._agl_m: float | None = None
        self._forward_range_m: float | None = None
        self._mode = "UNKNOWN"
        self._armed = False
        self._ekf_ok = False
        self._rc: dict[int, int] = {}
        self._last_message_ns = 0
        self._message_counts: dict[str, int] = {}
        self._received: dict[str, int] = {}
        self._ranges: dict[tuple[int, int], RangeReading] = {}
        self._history: deque[TelemetrySample] = deque(maxlen=cfg.runtime.telemetry_buffer_len)

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # -- connection --------------------------------------------------------

    def connect(self, wait_heartbeat: bool = True) -> None:
        """Open the link (if not injected), wait for a heartbeat, set rates."""
        if self.connection is None:
            self.connection = mavutil.mavlink_connection(
                self.params.endpoint,
                source_system=self.params.source_system,
                source_component=self.params.source_component,
            )
        if wait_heartbeat:
            heartbeat = self.connection.wait_heartbeat(timeout=self.params.heartbeat_timeout_s)
            if heartbeat is None:
                raise MavlinkError(
                    f"no heartbeat from {self.params.endpoint} within "
                    f"{self.params.heartbeat_timeout_s}s"
                )
        self.request_streams()

    def request_streams(self) -> None:
        """Ask for the rates in config rather than trusting the FC's defaults.

        ArduPilot's default stream rates depend on the SR*_ parameters, which
        are per-airframe and per-link. Requesting explicitly means the rate
        this code assumes is the rate it gets.
        """
        if self.connection is None:
            raise MavlinkError("not connected")
        for name, rate_hz in sorted(self.params.stream_rates().items()):
            message_id = getattr(mavutil.mavlink, f"MAVLINK_MSG_ID_{name}", None)
            if message_id is None:
                raise MavlinkError(f"unknown MAVLink message name in config: {name}")
            interval_us = int(round(1_000_000.0 / rate_hz))
            self.connection.mav.command_long_send(
                self.connection.target_system,
                self.connection.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                0,
                float(message_id),
                float(interval_us),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )

    # -- reader thread -----------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("reader already started")
        self._stop.clear()
        self._thread = threading.Thread(target=self._reader_loop, name="mavlink", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            self._thread = None

    def _reader_loop(self) -> None:
        while not self._stop.is_set():
            try:
                message = self.connection.recv_match(blocking=True, timeout=0.5)
            except Exception:  # noqa: BLE001 - a dead link must not kill the thread
                continue
            if message is not None:
                self.handle(message)

    # -- message handling --------------------------------------------------

    def handle(self, message: Any, t_ns: int | None = None) -> None:
        """Fold one MAVLink message into the cached state.

        Public and timestamp-injectable so tests can drive it directly with
        real pymavlink messages, no socket and no thread involved.
        """
        t_ns = int(self.clock() if t_ns is None else t_ns)
        kind = message.get_type()
        with self._lock:
            if self._history and t_ns < self._history[-1].state.t_ns:
                raise MavlinkError("telemetry timestamps must not move backwards")
            self._received[kind] = t_ns
            self._last_message_ns = t_ns
            self._message_counts[kind] = self._message_counts.get(kind, 0) + 1

            if kind == "ATTITUDE":
                self._roll = float(message.roll)
                self._pitch = float(message.pitch)
                self._yaw = float(message.yaw)
                self._attitudes.append(
                    AttitudeSample(t_ns, self._roll, self._pitch, self._yaw)
                )
            elif kind == "LOCAL_POSITION_NED":
                self._vel_ned = (float(message.vx), float(message.vy), float(message.vz))
            elif kind == "DISTANCE_SENSOR":
                distance_m = float(message.current_distance) / 100.0
                in_range = (
                    message.min_distance / 100.0 <= distance_m <= message.max_distance / 100.0
                )
                self._ranges[(int(message.id), int(message.orientation))] = RangeReading(
                    int(message.id), int(message.orientation), t_ns,
                    distance_m if in_range else None,
                )
                # QUESTION(rahul): confirm TFmini/FC max-range sentinel semantics.
                # The existing inclusive MAVLink bounds are preserved until measured.
                # Out-of-range readings are dropped rather than stored: a
                # rangefinder reporting its max value means "nothing seen",
                # not "the ground is 12 m away".
                if message.orientation == ORIENTATION_DOWN:
                    self._agl_m = distance_m if in_range else None
                elif message.orientation == ORIENTATION_FORWARD:
                    self._forward_range_m = distance_m if in_range else None
            elif kind == "SYS_STATUS":
                self._ekf_ok = self._sys_status_ekf_ok(message)
            elif kind == "HEARTBEAT":
                self._armed = bool(
                    message.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                )
                self._mode = self._mode_name(message)
            elif kind == "RC_CHANNELS":
                self._rc = self._rc_map(message)

            self._history.append(TelemetrySample(
                FcState(t_ns, self._roll, self._pitch, self._yaw, self._vel_ned,
                        self._agl_m, self._mode, self._armed, self._ekf_ok, dict(self._rc)),
                dict(self._received), dict(self._ranges),
            ))

    def telemetry_at(self, t_ns: int) -> TelemetrySample | None:
        """Last actual observation at/before the frame; never extrapolate into its past."""
        with self._lock:
            sample = next((s for s in reversed(self._history) if s.state.t_ns <= t_ns), None)
            if sample is None:
                return None
            from dataclasses import replace

            return TelemetrySample(replace(sample.state, rc=dict(sample.state.rc)),
                                   dict(sample.received), dict(sample.ranges))

    @staticmethod
    def _sys_status_ekf_ok(message: Any) -> bool:
        """AHRS healthy per SYS_STATUS.

        Present-and-enabled but unhealthy is the only combination that means
        "broken"; a sensor the FC does not have at all must not read as a
        failure.
        """
        bit = mavutil.mavlink.MAV_SYS_STATUS_AHRS
        present = bool(message.onboard_control_sensors_present & bit)
        enabled = bool(message.onboard_control_sensors_enabled & bit)
        healthy = bool(message.onboard_control_sensors_health & bit)
        if not (present and enabled):
            return True
        return healthy

    @staticmethod
    def _mode_name(message: Any) -> str:
        try:
            return str(mavutil.mode_string_v10(message))
        except Exception:  # noqa: BLE001 - unknown custom modes must not crash the reader
            return f"CUSTOM({getattr(message, 'custom_mode', '?')})"

    @staticmethod
    def _rc_map(message: Any) -> dict[int, int]:
        channels: dict[int, int] = {}
        for index in range(1, 19):
            value = getattr(message, f"chan{index}_raw", 0)
            # 0 means "not provided" in RC_CHANNELS; UINT16_MAX means unknown.
            if value and value != 65535:
                channels[index] = int(value)
        return channels

    # -- state -------------------------------------------------------------

    def state(self, t_ns: int | None = None) -> FcState:
        """Immutable snapshot of everything the FC has told us."""
        with self._lock:
            return FcState(
                t_ns=int(self.clock() if t_ns is None else t_ns),
                roll=self._roll,
                pitch=self._pitch,
                yaw=self._yaw,
                vel_ned=self._vel_ned,
                agl_m=self._agl_m,
                mode=self._mode,
                armed=self._armed,
                ekf_ok=self._ekf_ok,
                rc=dict(self._rc),
            )

    def rc_channel(self, channel: int) -> int | None:
        """Raw PWM for one RC channel, or None if the FC has not reported it."""
        with self._lock:
            return self._rc.get(int(channel))

    @property
    def forward_range_m(self) -> float | None:
        with self._lock:
            return self._forward_range_m

    def link_age_s(self, t_ns: int | None = None) -> float:
        """Seconds since any message arrived; inf if none ever has."""
        with self._lock:
            if self._last_message_ns == 0:
                return math.inf
            now = int(self.clock() if t_ns is None else t_ns)
            return max(0.0, (now - self._last_message_ns) / NS_PER_S)

    def message_counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._message_counts)

    def attitude_at(self, t_ns: int) -> AttitudeSample | None:
        """Attitude at ``t_ns``, interpolated between ring-buffer samples.

        Returns None when the buffer is empty, or when ``t_ns`` is further
        outside it than ``attitude_max_extrapolation_s``. Refusing to
        extrapolate is the point: a frame timestamped before the link came up
        has no attitude, and inventing one silently mis-projects every obstacle
        in that frame.
        """
        with self._lock:
            samples = list(self._attitudes)
        if not samples:
            return None

        limit_ns = self.params.attitude_max_extrapolation_s * NS_PER_S
        if t_ns <= samples[0].t_ns:
            if samples[0].t_ns - t_ns > limit_ns:
                return None
            return AttitudeSample(t_ns, samples[0].roll, samples[0].pitch, samples[0].yaw)
        if t_ns >= samples[-1].t_ns:
            if t_ns - samples[-1].t_ns > limit_ns:
                return None
            return AttitudeSample(t_ns, samples[-1].roll, samples[-1].pitch, samples[-1].yaw)

        low, high = 0, len(samples) - 1
        while high - low > 1:
            middle = (low + high) // 2
            if samples[middle].t_ns <= t_ns:
                low = middle
            else:
                high = middle

        before, after = samples[low], samples[high]
        span = after.t_ns - before.t_ns
        fraction = 0.0 if span <= 0 else (t_ns - before.t_ns) / span
        return AttitudeSample(
            t_ns=int(t_ns),
            roll=lerp_angle(before.roll, after.roll, fraction),
            pitch=lerp_angle(before.pitch, after.pitch, fraction),
            yaw=lerp_angle(before.yaw, after.yaw, fraction),
        )

    # -- output ------------------------------------------------------------

    def send_velocity_body(
        self, vx: float, vy: float, vz: float, yaw_rate: float, time_boot_ms: int = 0
    ) -> None:
        """Stream one body-frame velocity + yaw-rate setpoint.

        MAV_FRAME_BODY_OFFSET_NED so the axes follow the aircraft's nose, and
        a type mask that leaves position, acceleration and absolute yaw
        ignored. Position fields are sent as zero and *are* ignored -- the mask
        is what makes that true, so it is asserted in the tests rather than
        assumed.
        """
        if self.connection is None:
            raise MavlinkError("not connected")
        self.connection.mav.set_position_target_local_ned_send(
            int(time_boot_ms),
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
            TYPE_MASK_VELOCITY_YAW_RATE,
            0.0,
            0.0,
            0.0,
            float(vx),
            float(vy),
            float(vz),
            0.0,
            0.0,
            0.0,
            0.0,
            float(yaw_rate),
        )

    def send_command(self, command) -> None:
        """Emitter adaptor: hand a PlannerCommand straight to the FC."""
        self.send_velocity_body(command.vx, command.vy, command.vz, command.yaw_rate)

    def __enter__(self) -> MavlinkClient:
        self.connect()
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
