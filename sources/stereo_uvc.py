"""Bounded, record-only UVC capture and the shared MJPG decode/split rule.

OpenCV's V4L2 raw transport returns encoded MJPG; decoded BGR is rejected rather
than re-encoded. See https://docs.opencv.org/4.13.0/d4/d15/group__videoio__flags__base.html.
A separate process contains blocking driver calls. One shared-memory slot holds
only the latest packet, so slow perception cannot accumulate stale frames.

QUESTION(rahul): verify the actual UVC driver's exposure timestamp and buffering
on the Jetson. This transport timestamps host dequeue, not sensor exposure. Do
not use its latency measurements as exposure-to-command latency or enable flight.
"""
from __future__ import annotations

import ctypes
import multiprocessing as mp
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any, Protocol

import cv2
import numpy as np

from config import Config
from sources.types import FrameBundle


class CaptureError(RuntimeError):
    """Terminal capture fault; restart explicitly after correcting its cause."""


def discover_devices() -> list[str]:
    """List stable device names when available; do not open devices to probe."""
    stable = sorted(Path('/dev/v4l/by-id').glob('*'))
    return [str(p) for p in (stable or sorted(Path('/dev').glob('video*')))]


def split_side_by_side(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if frame.ndim < 2 or frame.shape[0] == 0 or frame.shape[1] == 0:
        raise CaptureError(f"expected a nonempty image, got shape {frame.shape}")
    if frame.shape[1] % 2:
        raise CaptureError(f"side-by-side frame width must be even, got {frame.shape[1]}")
    half = frame.shape[1] // 2
    return frame[:, :half], frame[:, half:]


def decode_mjpg(mjpg: bytes, seq: int, t_ns: int) -> FrameBundle:
    if not mjpg:
        raise CaptureError(f"frame seq={seq} is not decodable MJPG")
    frame = cv2.imdecode(np.frombuffer(mjpg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise CaptureError(f"frame seq={seq} is not decodable MJPG")
    left, right = split_side_by_side(frame)
    return FrameBundle(left, right, t_ns, seq)


@dataclass(frozen=True)
class CapturedFrame:
    bundle: FrameBundle
    mjpg: bytes
    dropped_before: int
    timestamp_source: str = "host_dequeue_monotonic"


class PacketTransport(Protocol):
    def read(self) -> tuple[bytes, int]: ...
    def close(self) -> None: ...


class OpenCvTransport:
    """One V4L2 device; the worker owns its complete lifetime."""

    def __init__(self, cfg: Config) -> None:
        cam = cfg.camera
        device = cfg.capture.device_path or cam.device_index
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        try:
            if not self.cap.isOpened():
                raise CaptureError(f"cannot open camera {device}")
            if cam.fourcc != "MJPG":
                raise CaptureError("raw logging requires camera.fourcc=MJPG")
            settings = (
                (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cam.fourcc)),
                (cv2.CAP_PROP_FRAME_WIDTH, cam.frame_width),
                (cv2.CAP_PROP_FRAME_HEIGHT, cam.frame_height),
                (cv2.CAP_PROP_FPS, cam.fps),
                (cv2.CAP_PROP_CONVERT_RGB, 0),
            )
            for key, value in settings:
                if not self.cap.set(key, value):
                    raise CaptureError(f"camera rejected property {key}={value}")
            for key, value in settings[:4]:
                if self.cap.get(key) != value:
                    raise CaptureError(f"camera negotiated {key}={self.cap.get(key)}, need {value}")
        except BaseException:
            self.cap.release()
            raise

    def read(self) -> tuple[bytes, int]:
        ok, raw = self.cap.read()
        t_ns = time.monotonic_ns()
        if not ok or raw is None:
            raise CaptureError("camera read failed or disconnected")
        if raw.dtype != np.uint8 or raw.ndim != 2:
            raise CaptureError("driver did not return encoded MJPG; refusing re-encoding")
        packet = raw.tobytes()
        if not packet.startswith(b'\xff\xd8'):
            raise CaptureError("driver returned non-JPEG bytes")
        return packet, t_ns

    def close(self) -> None:
        self.cap.release()


def _capture_worker(
    cfg: Config, factory: Callable[[Config], PacketTransport], slot: Any,
    header: Any, lock: Any, ready: Any, stop: Any, errors: Any,
) -> None:
    transport = None
    try:
        transport = factory(cfg)
        seq = 0
        last_t_ns = -1
        while not stop.is_set():
            packet, t_ns = transport.read()
            if not packet or len(packet) > cfg.capture.max_packet_bytes:
                raise CaptureError("empty or oversized MJPG packet")
            if t_ns <= last_t_ns:
                raise CaptureError("capture timestamps must increase monotonically")
            last_t_ns = t_ns
            with lock:
                slot[:len(packet)] = packet
                header[:] = (seq, t_ns, len(packet))
                ready.set()
            seq += 1
    except Exception as exc:
        errors.put(f"{type(exc).__name__}: {exc}")
        ready.set()
    finally:
        if transport is not None:
            transport.close()


class StereoCapture:
    """Single consumer. Faults latch; construct a new source to restart."""

    def __init__(
        self, cfg: Config, factory: Callable[[Config], PacketTransport] = OpenCvTransport,
    ) -> None:
        self.cfg = cfg
        self.factory = factory
        self._process: Any = None
        self._last_seq = -1
        self._fault: str | None = None
        ctx = mp.get_context("spawn")
        self._slot = ctx.RawArray(ctypes.c_ubyte, cfg.capture.max_packet_bytes)
        self._header = ctx.RawArray(ctypes.c_longlong, (-1, 0, 0))
        self._lock = ctx.Lock()
        self._ready = ctx.Event()
        self._stop = ctx.Event()
        self._errors = ctx.Queue(maxsize=1)
        self._ctx = ctx

    def start(self) -> StereoCapture:
        if self._process is not None:
            raise CaptureError("capture has already been started")
        self._process = self._ctx.Process(
            target=_capture_worker,
            args=(self.cfg, self.factory, self._slot, self._header, self._lock,
                  self._ready, self._stop, self._errors),
            daemon=True,
        )
        self._process.start()
        return self

    def read(self) -> CapturedFrame:
        if self._process is None or self._stop.is_set():
            raise CaptureError("capture is not running")
        if self._fault:
            raise CaptureError(self._fault)
        try:
            if not self._ready.wait(self.cfg.capture.timeout_s):
                raise CaptureError("capture timed out")
            try:
                raise CaptureError(self._errors.get_nowait())
            except Empty:
                pass
            if not self._process.is_alive():
                raise CaptureError(f"capture worker exited: {self._process.exitcode}")
            with self._lock:
                seq, t_ns, size = self._header[:]
                packet = bytes(self._slot[:size])
                self._ready.clear()
            if seq <= self._last_seq:
                raise CaptureError("capture worker failed without a fresh frame")
            bundle = decode_mjpg(packet, seq, t_ns)
            expected = (self.cfg.camera.eye_height, self.cfg.camera.eye_width, 3)
            if bundle.left.shape != expected or bundle.right.shape != expected:
                raise CaptureError(f"decoded camera shape {bundle.left.shape}, expected {expected}")
            dropped = seq - self._last_seq - 1
            self._last_seq = seq
            return CapturedFrame(bundle, packet, dropped)
        except CaptureError as exc:
            self._fault = str(exc)
            raise

    def close(self) -> None:
        self._stop.set()
        if self._process is not None:
            self._process.join(self.cfg.capture.shutdown_timeout_s)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(self.cfg.capture.shutdown_timeout_s)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(self.cfg.capture.shutdown_timeout_s)
        self._errors.close()

    def __enter__(self) -> StereoCapture:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.close()
