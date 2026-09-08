"""Per-stream freshness and bounded historical telemetry for frame-time pairing.

These are observational interfaces, not the owned command gate/supervisor.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

from config import Config
from sources.types import FcState


@dataclass(frozen=True)
class RangeReading:
    sensor_id: int
    orientation: int
    t_ns: int
    distance_m: float | None


@dataclass(frozen=True)
class TelemetrySample:
    state: FcState
    received: dict[str, int]
    ranges: dict[tuple[int, int], RangeReading]

    def age_s(self, kind: str, t_ns: int) -> float:
        received = self.received.get(kind)
        if received is None or received > t_ns:
            return math.inf
        return (t_ns - received) / 1e9

    def range_at(self, sensor_id: int, orientation: int, t_ns: int,
                 timeout_s: float) -> RangeReading | None:
        reading = self.ranges.get((sensor_id, orientation))
        if reading is None or not 0 <= (t_ns - reading.t_ns) / 1e9 <= timeout_s:
            return None
        return reading

    def paired_state(self, t_ns: int, cfg: Config) -> FcState:
        """Stamp the exact effective input so replay reproduces absence/staleness."""
        essential = ('ATTITUDE', 'LOCAL_POSITION_NED', 'SYS_STATUS', 'RC_CHANNELS')
        fresh = (self.age_s('HEARTBEAT', t_ns) <= cfg.mavlink.heartbeat_timeout_s
                 and all(self.age_s(k, t_ns) <= cfg.runtime.telemetry_timeout_s
                         for k in essential))
        downward = [r for r in self.ranges.values() if r.orientation == 25
                    and 0 <= (t_ns - r.t_ns) / 1e9 <= cfg.runtime.range_timeout_s]
        newest = max(downward, key=lambda r: r.t_ns) if downward else None
        return replace(self.state, t_ns=t_ns, rc=dict(self.state.rc),
                       mode=self.state.mode if fresh else 'UNKNOWN',
                       armed=self.state.armed if fresh else False,
                       ekf_ok=self.state.ekf_ok if fresh else False,
                       agl_m=None if newest is None else newest.distance_m)
