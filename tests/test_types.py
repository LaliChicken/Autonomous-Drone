"""The frozen contract in sources/types.py.

These tests exist to catch accidental edits to a file that must not change:
they pin the field names, the mutability of each dataclass, and the
DepthBackend protocol shape that depth backends are written against.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from sources.types import (
    DepthBackend,
    DepthResult,
    FcState,
    FrameBundle,
    OccupancySnapshot,
    PlannerCommand,
)


def _bundle() -> FrameBundle:
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    return FrameBundle(left=img, right=img.copy(), t_ns=123, seq=0)


def test_frame_bundle_is_frozen() -> None:
    bundle = _bundle()
    with pytest.raises(dataclasses.FrozenInstanceError):
        bundle.t_ns = 456  # type: ignore[misc]


def test_planner_command_is_frozen() -> None:
    cmd = PlannerCommand(vx=1.0, vy=0.0, vz=0.0, yaw_rate=0.0, reason="test", t_ns=1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cmd.vx = 2.0  # type: ignore[misc]


def test_mutable_dataclasses_stay_mutable() -> None:
    result = DepthResult(
        depth_m=np.zeros((2, 2), np.float32),
        valid=np.zeros((2, 2), bool),
        conf=None,
        t_ns=0,
    )
    result.t_ns = 5
    assert result.t_ns == 5

    snapshot = OccupancySnapshot(
        t_ns=0,
        bearings=np.zeros(3),
        distances=np.zeros(3, np.float32),
        confidence=np.zeros(3, np.float32),
        unknown=np.ones(3, bool),
        danger=np.zeros(3, bool),
    )
    snapshot.t_ns = 7
    assert snapshot.t_ns == 7


def test_contract_field_names_are_unchanged() -> None:
    assert [f.name for f in dataclasses.fields(FrameBundle)] == ["left", "right", "t_ns", "seq"]
    assert [f.name for f in dataclasses.fields(DepthResult)] == [
        "depth_m",
        "valid",
        "conf",
        "t_ns",
    ]
    assert [f.name for f in dataclasses.fields(OccupancySnapshot)] == [
        "t_ns",
        "bearings",
        "distances",
        "confidence",
        "unknown",
        "danger",
    ]
    assert [f.name for f in dataclasses.fields(PlannerCommand)] == [
        "vx",
        "vy",
        "vz",
        "yaw_rate",
        "reason",
        "t_ns",
    ]
    assert [f.name for f in dataclasses.fields(FcState)] == [
        "t_ns",
        "roll",
        "pitch",
        "yaw",
        "vel_ned",
        "agl_m",
        "mode",
        "armed",
        "ekf_ok",
        "rc",
    ]


def test_depth_result_conf_is_optional() -> None:
    conf_field = {f.name: f for f in dataclasses.fields(DepthResult)}["conf"]
    assert "None" in str(conf_field.type)


def test_depth_backend_protocol_is_structural() -> None:
    class Fake:
        def infer(self, f: FrameBundle) -> DepthResult:
            h, w = f.left.shape[:2]
            return DepthResult(
                depth_m=np.full((h, w), 3.0, np.float32),
                valid=np.ones((h, w), bool),
                conf=None,
                t_ns=f.t_ns,
            )

    backend: DepthBackend = Fake()
    result = backend.infer(_bundle())
    assert result.depth_m.shape == (4, 4)
    assert result.t_ns == 123


def test_planner_command_reason_is_required() -> None:
    with pytest.raises(TypeError):
        PlannerCommand(vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0, t_ns=0)  # type: ignore[call-arg]
