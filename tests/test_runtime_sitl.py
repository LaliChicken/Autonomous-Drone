"""Optional real SITL telemetry through the record-only pipeline and replay."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from config import Config
from infra.flightlog import FlightLogReader
from infra.pipeline import RecordPipeline
from infra.replay import command_stream_bytes, replay_run
from sources.mavlink_client import MavlinkClient
from sources.types import FcState
from tools.run import record_frames, synthetic_frames


@pytest.mark.sitl
def test_sitl_record_only_pipeline_replays(cfg: Config, tmp_path: Path) -> None:
    client = MavlinkClient(cfg)
    client.connect()
    client.start()
    required = ('ATTITUDE', 'LOCAL_POSITION_NED', 'SYS_STATUS', 'RC_CHANNELS', 'HEARTBEAT')
    try:
        deadline = time.monotonic() + 2 * cfg.mavlink.heartbeat_timeout_s
        while time.monotonic() < deadline:
            sample = client.telemetry_at(time.monotonic_ns())
            if sample and all(k in sample.received for k in required):
                break
            time.sleep(cfg.offboard.period_s)
        else:
            pytest.fail(f'SITL missing essential telemetry streams: {client.message_counts()}')

        def state_at(t_ns: int) -> FcState | None:
            sample = client.telemetry_at(t_ns)
            return sample.paired_state(t_ns, cfg) if sample else None

        path = record_frames(cfg, synthetic_frames(cfg, 5), root=tmp_path,
                             state_at=state_at, source='synthetic')
        reader = FlightLogReader(path)
        recorded = command_stream_bytes(reader.planner_commands())
        assert len(tuple(reader.telemetry())) == 5
        assert replay_run(path, RecordPipeline).stream == recorded
    finally:
        client.stop()
        client.connection.close()
