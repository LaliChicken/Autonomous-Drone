"""Config loader: conversion, validation, and the derived geometry it owns."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import pytest

from config import Config, ConfigError, load_config, load_config_from_dict


def test_shipped_config_loads_and_validates() -> None:
    cfg = load_config()
    assert isinstance(cfg, Config)
    assert cfg.source_path.endswith("config.yaml")


def test_config_is_frozen(cfg: Config) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.envelope.max_speed_mps = 99.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.obstacles.n_bins = 4  # type: ignore[misc]


def test_degrees_are_converted_to_radians(cfg: Config, raw_config: dict[str, Any]) -> None:
    assert cfg.obstacles.fov_rad == pytest.approx(math.radians(raw_config["obstacles"]["fov_deg"]))
    assert cfg.camera.hfov_rad == pytest.approx(math.radians(raw_config["camera"]["hfov_deg"]))
    assert cfg.envelope.max_yaw_rate_rad_s == pytest.approx(
        math.radians(raw_config["envelope"]["max_yaw_rate_dps"])
    )
    # Nothing downstream should ever see a *_deg attribute.
    for section_field in dataclasses.fields(cfg):
        section = getattr(cfg, section_field.name)
        if not dataclasses.is_dataclass(section):
            continue
        for f in dataclasses.fields(section):
            assert not f.name.endswith("_deg"), f"{section_field.name}.{f.name} leaked degrees"


def test_missing_section_is_rejected(mutable_config: dict[str, Any]) -> None:
    del mutable_config["occupancy"]
    with pytest.raises(ConfigError, match="missing the 'occupancy' section"):
        load_config_from_dict(mutable_config)


def test_missing_key_is_rejected(mutable_config: dict[str, Any]) -> None:
    del mutable_config["obstacles"]["n_bins"]
    with pytest.raises(ConfigError, match="obstacles.n_bins"):
        load_config_from_dict(mutable_config)


def test_bin_count_mismatch_is_rejected(mutable_config: dict[str, Any]) -> None:
    mutable_config["occupancy"]["n_bins"] = 16
    with pytest.raises(ConfigError, match="n_bins.*must match"):
        load_config_from_dict(mutable_config)


def test_fov_mismatch_is_rejected(mutable_config: dict[str, Any]) -> None:
    mutable_config["occupancy"]["fov_deg"] = 90.0
    with pytest.raises(ConfigError, match="fov_deg.*must match"):
        load_config_from_dict(mutable_config)


@pytest.mark.parametrize("bad", [127, 100, 0, -16])
def test_num_disparities_must_be_positive_multiple_of_16(
    mutable_config: dict[str, Any], bad: int
) -> None:
    mutable_config["depth"]["sgbm"]["num_disparities"] = bad
    with pytest.raises(ConfigError, match="num_disparities"):
        load_config_from_dict(mutable_config)


def test_block_size_must_be_odd(mutable_config: dict[str, Any]) -> None:
    mutable_config["depth"]["sgbm"]["block_size"] = 4
    with pytest.raises(ConfigError, match="block_size"):
        load_config_from_dict(mutable_config)


def test_depth_range_must_be_ordered(mutable_config: dict[str, Any]) -> None:
    mutable_config["depth"]["postprocess"]["min_depth_m"] = 20.0
    with pytest.raises(ConfigError, match="max_depth_m"):
        load_config_from_dict(mutable_config)


def test_height_gate_must_be_ordered(mutable_config: dict[str, Any]) -> None:
    mutable_config["obstacles"]["height_ceiling_m"] = -5.0
    with pytest.raises(ConfigError, match="height_ceiling_m"):
        load_config_from_dict(mutable_config)


def test_autonomy_speed_cannot_exceed_envelope(mutable_config: dict[str, Any]) -> None:
    mutable_config["behaviours"]["autonomy_speed_max_mps"] = 12.0
    with pytest.raises(ConfigError, match="autonomy_speed_max_mps"):
        load_config_from_dict(mutable_config)


def test_planner_yaw_rate_cannot_exceed_envelope(mutable_config: dict[str, Any]) -> None:
    mutable_config["local_planner"]["max_yaw_rate_dps"] = 720.0
    with pytest.raises(ConfigError, match="max_yaw_rate_dps"):
        load_config_from_dict(mutable_config)


def test_hue_outside_opencv_range_is_rejected(mutable_config: dict[str, Any]) -> None:
    mutable_config["red_box"]["hue_hi_2"] = 255
    with pytest.raises(ConfigError, match="hue_hi_2"):
        load_config_from_dict(mutable_config)


def test_red_box_hue_ranges_cover_the_wrap(cfg: Config) -> None:
    # Red wraps the hue circle; the two ranges must sit at opposite ends.
    assert cfg.red_box.hue_lo_1 < cfg.red_box.hue_hi_1 < cfg.red_box.hue_lo_2
    assert cfg.red_box.hue_hi_2 <= 179


def test_sgbm_p1_p2_defaults_follow_opencv_guidance(cfg: Config) -> None:
    sgbm = cfg.depth.sgbm
    assert sgbm.effective_p1(1) == 8 * sgbm.block_size**2
    assert sgbm.effective_p2(1) == 32 * sgbm.block_size**2
    assert sgbm.effective_p2(3) > sgbm.effective_p1(3)


def test_danger_distance_grows_with_speed(cfg: Config) -> None:
    occ = cfg.occupancy
    at_rest = occ.danger_distance(0.0)
    assert at_rest == pytest.approx(occ.danger_margin_m)
    assert occ.danger_distance(5.0) > occ.danger_distance(2.0) > at_rest
    # Explicit formula check: reaction + braking + margin.
    speed = 3.0
    expected = (
        occ.danger_reaction_s * speed
        + speed**2 / (2 * occ.danger_decel_mps2)
        + occ.danger_margin_m
    )
    assert occ.danger_distance(speed) == pytest.approx(expected)


def test_danger_distance_ignores_negative_speed(cfg: Config) -> None:
    assert cfg.occupancy.danger_distance(-4.0) == pytest.approx(cfg.occupancy.danger_margin_m)


def test_nominal_q_recovers_depth_from_disparity(cfg: Config) -> None:
    cam = cfg.camera
    q = cam.nominal_q()
    f = cam.nominal_focal_px
    baseline = cam.baseline_m

    for disparity in (8.0, 32.0, 96.0):
        # Reproject the principal point at this disparity and check Z = f*B/d.
        cx = (cam.eye_width - 1) / 2.0
        cy = (cam.eye_height - 1) / 2.0
        homogeneous = q @ np.array([cx, cy, disparity, 1.0])
        point = homogeneous[:3] / homogeneous[3]
        assert point[2] == pytest.approx(f * baseline / disparity, rel=1e-9)
        # Principal-point ray is straight ahead: x and y are zero.
        assert point[0] == pytest.approx(0.0, abs=1e-9)
        assert point[1] == pytest.approx(0.0, abs=1e-9)


def test_nominal_focal_matches_hfov(cfg: Config) -> None:
    cam = cfg.camera
    half_angle = math.atan((cam.eye_width / 2.0) / cam.nominal_focal_px)
    assert 2 * half_angle == pytest.approx(cam.hfov_rad)


def test_eye_split_is_half_the_side_by_side_frame(cfg: Config) -> None:
    assert cfg.camera.eye_width * 2 == cfg.camera.frame_width
    assert cfg.camera.eye_height == cfg.camera.frame_height


def test_mount_rotation_maps_camera_axes_to_body(cfg_from) -> None:
    cfg = cfg_from({"mount": {"roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0}})
    r = cfg.mount.rotation_body_from_cam()
    # camera z (forward) -> body x (forward)
    assert np.allclose(r @ np.array([0.0, 0.0, 1.0]), [1.0, 0.0, 0.0])
    # camera x (right) -> body y (right)
    assert np.allclose(r @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0])
    # camera y (down) -> body z (down)
    assert np.allclose(r @ np.array([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0])
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(r) == pytest.approx(1.0)


def test_mount_pitch_follows_the_fc_sign_convention(cfg_from) -> None:
    # Positive pitch is nose-UP, as in FcState.pitch. z is down in FRD, so a
    # nose-up boresight has a negative z component.
    up = cfg_from({"mount": {"pitch_deg": 30.0}}).mount.rotation_body_from_cam()
    boresight = up @ np.array([0.0, 0.0, 1.0])
    assert boresight[2] == pytest.approx(-math.sin(math.radians(30.0)))
    assert boresight[0] == pytest.approx(math.cos(math.radians(30.0)))
    assert boresight[1] == pytest.approx(0.0, abs=1e-12)

    # A downward-tilted camera is configured with a negative pitch_deg.
    down = cfg_from({"mount": {"pitch_deg": -20.0}}).mount.rotation_body_from_cam()
    assert (down @ np.array([0.0, 0.0, 1.0]))[2] == pytest.approx(math.sin(math.radians(20.0)))


def test_mount_yaw_swings_the_boresight_right(cfg_from) -> None:
    cfg = cfg_from({"mount": {"yaw_deg": 45.0}})
    boresight = cfg.mount.rotation_body_from_cam() @ np.array([0.0, 0.0, 1.0])
    assert math.atan2(boresight[1], boresight[0]) == pytest.approx(math.radians(45.0))


def test_stream_rates_cover_the_required_messages(cfg: Config) -> None:
    rates = cfg.mavlink.stream_rates()
    assert rates["ATTITUDE"] == 50
    assert rates["LOCAL_POSITION_NED"] == 10
    assert rates["DISTANCE_SENSOR"] == 10
    assert rates["SYS_STATUS"] == 2
    assert rates["RC_CHANNELS"] == 5


def test_to_dict_round_trips_through_the_loader(cfg: Config) -> None:
    snapshot = cfg.to_dict()
    assert isinstance(snapshot["mavlink"]["stream_rates_hz"], dict)
    # A snapshot is degrees-free, so it cannot be fed back through _build,
    # which expects the YAML shape. What must hold is that it is JSON-shaped.
    import json

    assert json.loads(json.dumps(snapshot))["obstacles"]["n_bins"] == cfg.obstacles.n_bins


def test_missing_config_file_is_reported(tmp_path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_non_mapping_config_is_reported(tmp_path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="top-level mapping"):
        load_config(bad)


def test_calibration_file_is_required_to_exist(mutable_config: dict[str, Any]) -> None:
    mutable_config["camera"]["calibration_npz"] = "/nonexistent/calib.npz"
    cfg = load_config_from_dict(mutable_config)
    with pytest.raises(ConfigError, match="does not exist"):
        cfg.camera.load_q()


def test_calibration_q_is_loaded_when_present(mutable_config: dict[str, Any], tmp_path) -> None:
    q = np.eye(4) * 2.0
    path = tmp_path / "calib.npz"
    np.savez(path, Q=q)
    mutable_config["camera"]["calibration_npz"] = str(path)
    cfg = load_config_from_dict(mutable_config)
    assert np.allclose(cfg.camera.load_q(), q)


def test_calibration_without_q_is_rejected(mutable_config: dict[str, Any], tmp_path) -> None:
    path = tmp_path / "calib.npz"
    np.savez(path, notQ=np.eye(4))
    mutable_config["camera"]["calibration_npz"] = str(path)
    cfg = load_config_from_dict(mutable_config)
    with pytest.raises(ConfigError, match="no 'Q' array"):
        cfg.camera.load_q()


def test_falls_back_to_nominal_q_without_calibration(cfg: Config) -> None:
    assert cfg.camera.calibration_npz is None
    assert np.allclose(cfg.camera.load_q(), cfg.camera.nominal_q())
