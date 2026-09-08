"""Use projected physical geometry to check the real OpenCV solver and rectifier."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from calib.rectify import Rectifier
from calib.solve import board_points, solve_points
from config import ConfigError


def observations(cfg):
    points = board_points(cfg.calibration)
    k = np.array([[500., 0., 319.5], [0., 500., 239.5], [0., 0., 1.]])
    left, right = [], []
    for i in range(10):
        rotation = np.array([0.08 * (i % 3 - 1), 0.12 * (i % 4 - 1), 0.03 * i])
        translation = np.array([-0.08 + 0.008 * i, -0.06 + 0.005 * i, 0.65 + 0.04 * i])
        a, _ = cv2.projectPoints(points, rotation, translation, k, np.zeros(5))
        b, _ = cv2.projectPoints(points, rotation, translation + [-0.052, 0., 0.],
                                 k, np.zeros(5))
        left.append(a)
        right.append(b)
    return left, right


@pytest.fixture
def solved(cfg_from):
    cfg = cfg_from({'camera': {'frame_width': 1280, 'frame_height': 480},
                    'calibration': {'columns': 7, 'rows': 5, 'square_m': 0.025, 'min_pairs': 8}})
    a, b = observations(cfg)
    return cfg, solve_points(a, b, (640, 480), cfg)


def test_unmeasured_board_is_rejected(cfg) -> None:
    with pytest.raises(ConfigError, match='measured'):
        board_points(cfg.calibration)


def test_real_solver_recovers_geometry_and_does_not_validate(solved, tmp_path) -> None:
    cfg, data = solved
    assert np.linalg.norm(data['T']) == pytest.approx(0.052, abs=1e-4)
    assert data['fit_rms_px'].max() < 0.01
    assert not data['validated']
    path = tmp_path / 'candidate.npz'
    np.savez(path, **data)
    rectifier = Rectifier(path, cfg)
    image = np.full((480, 640, 3), 128, dtype=np.uint8)
    a, b = rectifier.apply(image, image)
    assert a.shape == b.shape == image.shape
    assert np.array_equal(image, np.full_like(image, 128))
    assert not rectifier.valid[:, :cfg.depth.sgbm.num_disparities].any()
    assert rectifier.valid.any()
    assert np.allclose(rectifier.rotation_body_from_rectified,
                       cfg.mount.rotation_body_from_cam(), atol=1e-3)


def test_missing_or_wrong_resolution_calibration_rejected(solved, tmp_path) -> None:
    cfg, data = solved
    path = tmp_path / 'bad.npz'
    np.savez(path, Q=data['Q'])
    with pytest.raises(ConfigError, match='missing'):
        Rectifier(path, cfg)
    data['image_size'] = np.array([1280, 720])
    np.savez(path, **data)
    with pytest.raises(ConfigError, match='resolution'):
        Rectifier(path, cfg)


def test_corrupt_rotation_and_maps_rejected(solved, tmp_path) -> None:
    cfg, data = solved
    path = tmp_path / 'bad.npz'
    for field in ['R1', 'left_x', 'Q']:
        copy = dict(data)
        copy[field] = np.full_like(data[field], np.nan)
        np.savez(path, **copy)
        with pytest.raises(ConfigError):
            Rectifier(path, cfg)


def test_solver_requires_matching_diverse_sample_count(cfg_from) -> None:
    cfg = cfg_from({'calibration': {'columns': 7, 'rows': 5, 'square_m': 0.025}})
    with pytest.raises(ConfigError, match='matched'):
        solve_points([], [], (1280, 720), cfg)
