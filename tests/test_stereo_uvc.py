"""Exercise the source with a real child process and synthetic MJPG transport."""
from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np
import pytest

from config import Config, ConfigError, load_config_from_dict
from infra.flightlog import FrameRecord
from infra.replay import decode_frame
from sources.stereo_uvc import CaptureError, StereoCapture, decode_mjpg, split_side_by_side


def jpeg() -> bytes:
    frame = np.zeros((24, 64, 3), dtype=np.uint8)
    frame[:, :32, 2] = 255
    frame[:, 32:, 0] = 255
    ok, data = cv2.imencode('.jpg', frame)
    assert ok
    return data.tobytes()


class SyntheticTransport:
    def __init__(self, cfg: Config) -> None:
        self.packet = jpeg()

    def read(self) -> tuple[bytes, int]:
        time.sleep(0.01)
        return self.packet, time.monotonic_ns()

    def close(self) -> None:
        pass


class StalledTransport(SyntheticTransport):
    def read(self) -> tuple[bytes, int]:
        time.sleep(60)
        return super().read()


class BadTransport(SyntheticTransport):
    def read(self) -> tuple[bytes, int]:
        time.sleep(0.01)
        return b'not jpeg', time.monotonic_ns()


class DisconnectTransport(SyntheticTransport):
    def read(self) -> tuple[bytes, int]:
        raise OSError('disconnected')


def test_decode_replay_and_capture_share_pixels() -> None:
    packet = jpeg()
    live = decode_mjpg(packet, 4, 123)
    replay = decode_frame(FrameRecord(4, 123, 0, len(packet), packet))
    assert np.array_equal(live.left, replay.left)
    assert np.array_equal(live.right, replay.right)
    assert live.left[..., 2].mean() > 250
    assert live.right[..., 0].mean() > 250
    assert (live.seq, live.t_ns) == (4, 123)


@pytest.mark.parametrize('data', [b'', b'bad'])
def test_bad_jpeg(data: bytes) -> None:
    with pytest.raises(CaptureError, match='not decodable'):
        decode_mjpg(data, 0, 1)


def test_odd_and_empty_images() -> None:
    for shape in [(0, 4), (2, 0), (2, 3)]:
        with pytest.raises(CaptureError):
            split_side_by_side(np.zeros(shape, dtype=np.uint8))


def test_latest_frame_and_original_bytes(cfg_from: Any) -> None:
    cfg = cfg_from({'camera': {'frame_width': 64, 'frame_height': 24}})
    with StereoCapture(cfg, SyntheticTransport) as source:
        first = source.read()
        time.sleep(0.1)
        second = source.read()
        assert second.bundle.seq > first.bundle.seq + 1
        assert second.dropped_before == second.bundle.seq - first.bundle.seq - 1
        assert first.mjpg == second.mjpg == jpeg()
        assert second.bundle.t_ns > first.bundle.t_ns
    assert not source._process.is_alive()


@pytest.mark.parametrize('factory', [BadTransport, DisconnectTransport])
def test_fault_latches(cfg_from: Any, factory: Any) -> None:
    cfg = cfg_from({'camera': {'frame_width': 64, 'frame_height': 24}})
    with StereoCapture(cfg, factory) as source:
        with pytest.raises(CaptureError):
            source.read()
        with pytest.raises(CaptureError):
            source.read()


def test_stall_has_bounded_shutdown(cfg_from: Any) -> None:
    cfg = cfg_from({'capture': {'timeout_s': 0.5, 'shutdown_timeout_s': 0.1}})
    start = time.monotonic()
    with StereoCapture(cfg, StalledTransport) as source:
        with pytest.raises(CaptureError, match='timed out'):
            source.read()
    assert not source._process.is_alive()
    assert time.monotonic() - start < 5


def test_actual_resolution_is_checked(cfg: Config) -> None:
    with StereoCapture(cfg, SyntheticTransport) as source:
        with pytest.raises(CaptureError, match='shape'):
            source.read()


def test_capture_config_validation_and_old_logs(mutable_config: Any) -> None:
    del mutable_config['capture']
    assert load_config_from_dict(mutable_config).capture.timeout_s > 0
    mutable_config['capture'] = {
        'device_path': None, 'timeout_s': float('nan'),
        'shutdown_timeout_s': 1, 'max_packet_bytes': 1024,
    }
    with pytest.raises(ConfigError, match='timeout_s'):
        load_config_from_dict(mutable_config)


@pytest.mark.hw
def test_real_uvc_capture() -> None:
    import os

    from config import load_config

    cfg = load_config(os.environ.get('DRONE_CONFIG'))
    with StereoCapture(cfg) as source:
        a, b = source.read(), source.read()
        assert b.bundle.seq > a.bundle.seq
        assert b.bundle.t_ns > a.bundle.t_ns
        assert a.bundle.left.shape == (cfg.camera.eye_height, cfg.camera.eye_width, 3)
        assert a.mjpg.startswith(b'\xff\xd8')
