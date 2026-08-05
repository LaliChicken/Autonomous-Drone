"""Depth -> body-frame azimuth scan."""

from __future__ import annotations

import math

import numpy as np
import pytest

from config import Config
from depth.sgbm_cpu import SgbmCpuBackend
from perception.obstacles import (
    ObstacleScan,
    bin_centres,
    bin_edges,
    camera_intrinsics_from_q,
    depth_to_points_camera,
    empty_scan,
    points_camera_to_body,
    scan_from_depth,
)
from sources.types import DepthResult, FrameBundle
from tools.synthetic_stereo import planar_scene

WIDTH, HEIGHT = 160, 120
# Chosen so the test camera's horizontal FOV (~65.2 deg) is a shade wider than
# obstacles.fov_deg. With a narrower lens the outer bins would legitimately
# see nothing and every edge-bin assertion would be testing the fixture.
FOCAL = 125.0


def make_q(width: int = WIDTH, height: int = HEIGHT, focal: float = FOCAL) -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0, -(width - 1) / 2.0],
            [0.0, 1.0, 0.0, -(height - 1) / 2.0],
            [0.0, 0.0, 0.0, focal],
            [0.0, 0.0, 1.0 / 0.05, 0.0],
        ],
        dtype=np.float64,
    )


def flat_depth_result(
    depth_m: float, width: int = WIDTH, height: int = HEIGHT, t_ns: int = 5
) -> DepthResult:
    return DepthResult(
        depth_m=np.full((height, width), depth_m, dtype=np.float32),
        valid=np.ones((height, width), dtype=bool),
        conf=None,
        t_ns=t_ns,
    )


@pytest.fixture
def flat_cfg(cfg_from):
    """No mount offset or rotation, so geometry tests read straight through."""
    return cfg_from(
        {
            "mount": {
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "roll_deg": 0.0,
                "pitch_deg": 0.0,
                "yaw_deg": 0.0,
            },
            "obstacles": {"min_points_per_bin": 1, "height_floor_m": -50.0,
                          "height_ceiling_m": 50.0},
        }
    )


# --------------------------------------------------------------------------
# Binning
# --------------------------------------------------------------------------


def test_bin_edges_span_the_fov() -> None:
    edges = bin_edges(4, math.radians(60.0))
    assert edges[0] == pytest.approx(math.radians(-30.0))
    assert edges[-1] == pytest.approx(math.radians(30.0))
    assert len(edges) == 5


def test_bin_centres_are_between_the_edges() -> None:
    centres = bin_centres(4, math.radians(60.0))
    assert len(centres) == 4
    assert centres[0] == pytest.approx(math.radians(-22.5))
    assert centres[-1] == pytest.approx(math.radians(22.5))
    # Symmetric about straight ahead.
    assert centres.sum() == pytest.approx(0.0)


def test_bin_centres_are_monotonic_left_to_right() -> None:
    centres = bin_centres(32, math.radians(65.0))
    assert np.all(np.diff(centres) > 0)


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def test_intrinsics_are_recovered_from_q() -> None:
    focal, cx, cy = camera_intrinsics_from_q(make_q())
    assert focal == pytest.approx(FOCAL)
    assert cx == pytest.approx((WIDTH - 1) / 2.0)
    assert cy == pytest.approx((HEIGHT - 1) / 2.0)


def test_intrinsics_reject_a_bad_q() -> None:
    with pytest.raises(ValueError, match="4x4"):
        camera_intrinsics_from_q(np.eye(3))


def test_principal_ray_maps_to_straight_ahead() -> None:
    depth = np.full((HEIGHT, WIDTH), 3.0, dtype=np.float32)
    x, y, z = depth_to_points_camera(depth, make_q())
    cu, cv = (WIDTH - 1) // 2, (HEIGHT - 1) // 2
    assert x[cv, cu] == pytest.approx(0.0, abs=0.02)
    assert y[cv, cu] == pytest.approx(0.0, abs=0.02)
    assert z[cv, cu] == pytest.approx(3.0)


def test_pixel_offset_becomes_lateral_offset() -> None:
    depth = np.full((HEIGHT, WIDTH), 2.0, dtype=np.float32)
    x, _, _ = depth_to_points_camera(depth, make_q())
    cx = (WIDTH - 1) / 2.0
    column = 120
    offset_px = column - cx
    assert x[0, column] == pytest.approx(offset_px * 2.0 / FOCAL, rel=1e-4)


def test_camera_to_body_applies_the_axis_swap(cfg_from) -> None:
    cfg = cfg_from({"mount": {"x_m": 0.0, "y_m": 0.0, "z_m": 0.0}})
    rotation = cfg.mount.rotation_body_from_cam()
    translation = cfg.mount.translation()
    # A point 5 m straight ahead of the camera (camera z) is 5 m forward (body x).
    bx, by, bz = points_camera_to_body(
        np.array([0.0], np.float32),
        np.array([0.0], np.float32),
        np.array([5.0], np.float32),
        rotation,
        translation,
    )
    assert (bx[0], by[0], bz[0]) == pytest.approx((5.0, 0.0, 0.0), abs=1e-5)


def test_camera_to_body_applies_the_translation(cfg_from) -> None:
    cfg = cfg_from({"mount": {"x_m": 0.1, "y_m": 0.0, "z_m": 0.0}})
    bx, _, _ = points_camera_to_body(
        np.array([0.0], np.float32),
        np.array([0.0], np.float32),
        np.array([5.0], np.float32),
        cfg.mount.rotation_body_from_cam(),
        cfg.mount.translation(),
    )
    assert bx[0] == pytest.approx(5.1)


# --------------------------------------------------------------------------
# Scanning
# --------------------------------------------------------------------------


def test_flat_wall_gives_the_expected_range(flat_cfg: Config) -> None:
    scan = scan_from_depth(flat_depth_result(4.0), flat_cfg, q=make_q())
    assert len(scan) == flat_cfg.obstacles.n_bins
    centre = scan.distances[flat_cfg.obstacles.n_bins // 2]
    assert centre == pytest.approx(4.0, rel=0.05)
    assert not scan.unknown.any()


def test_a_wall_is_further_away_off_axis(flat_cfg: Config) -> None:
    # A fronto-parallel wall at range R is at R/cos(theta) along a bearing of
    # theta, so the edge bins must read further than the centre.
    scan = scan_from_depth(flat_depth_result(4.0), flat_cfg, q=make_q())
    n = flat_cfg.obstacles.n_bins
    assert scan.distances[0] > scan.distances[n // 2]
    assert scan.distances[-1] > scan.distances[n // 2]


def test_scan_carries_the_depth_timestamp(flat_cfg: Config) -> None:
    assert scan_from_depth(flat_depth_result(4.0, t_ns=12345), flat_cfg, q=make_q()).t_ns == 12345


def test_bins_with_no_points_are_unknown_with_nan_range(cfg_from) -> None:
    cfg = cfg_from({"obstacles": {"min_points_per_bin": 25}})
    depth = flat_depth_result(4.0)
    depth.valid[:] = False  # nothing measured anywhere
    scan = scan_from_depth(depth, cfg, q=make_q())
    assert scan.unknown.all()
    assert np.all(np.isnan(scan.distances))
    assert np.all(scan.confidence == 0.0)
    assert np.all(scan.counts == 0)


def test_unknown_range_is_nan_not_zero_and_not_infinity(cfg_from) -> None:
    # NaN is the only value that fails both "is it close" and "is it far".
    cfg = cfg_from(
        {
            "obstacles": {
                "min_points_per_bin": 10_000_000,
                "points_for_full_confidence": 10_000_000,
            }
        }
    )
    scan = scan_from_depth(flat_depth_result(4.0), cfg, q=make_q())
    assert scan.unknown.all()
    unknown_range = scan.distances[0]
    assert np.isnan(unknown_range)
    assert not (unknown_range > 2.5)  # does not read as clear
    assert not (unknown_range < 2.5)  # does not read as an obstacle either


def test_sparse_bins_fall_below_the_point_threshold(cfg_from) -> None:
    cfg = cfg_from({"obstacles": {"min_points_per_bin": 50, "height_floor_m": -50.0,
                                  "height_ceiling_m": 50.0}})
    depth = flat_depth_result(4.0)
    depth.valid[:] = False
    depth.valid[HEIGHT // 2, :10] = True  # ten pixels, all in a couple of bins
    scan = scan_from_depth(depth, cfg, q=make_q())
    assert scan.unknown.all()
    assert scan.counts.sum() == 10


def test_confidence_scales_with_point_count(flat_cfg: Config) -> None:
    scan = scan_from_depth(flat_depth_result(4.0), flat_cfg, q=make_q())
    assert np.all((scan.confidence >= 0.0) & (scan.confidence <= 1.0))
    # A full frame comfortably saturates the configured point count.
    assert scan.confidence.max() == pytest.approx(1.0)


def test_height_gate_excludes_points_above_the_ceiling(cfg_from) -> None:
    cfg = cfg_from(
        {
            "mount": {"x_m": 0.0, "y_m": 0.0, "z_m": 0.0, "pitch_deg": 0.0},
            "obstacles": {"height_floor_m": -0.1, "height_ceiling_m": 0.1,
                          "min_points_per_bin": 1},
        }
    )
    depth = flat_depth_result(4.0)
    scan = scan_from_depth(depth, cfg, q=make_q())
    # Only the rows near the optical axis survive a 20 cm slab at 4 m.
    assert scan.counts.sum() < depth.valid.size
    assert scan.counts.sum() > 0

    wide = cfg_from(
        {
            "mount": {"x_m": 0.0, "y_m": 0.0, "z_m": 0.0, "pitch_deg": 0.0},
            "obstacles": {"height_floor_m": -50.0, "height_ceiling_m": 50.0,
                          "min_points_per_bin": 1},
        }
    )
    assert scan_from_depth(depth, wide, q=make_q()).counts.sum() > scan.counts.sum()


def test_exclusion_mask_removes_points(flat_cfg: Config) -> None:
    depth = flat_depth_result(4.0)
    without = scan_from_depth(depth, flat_cfg, q=make_q())
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[HEIGHT // 2 :, :] = True  # pretend the bottom half is ground
    with_mask = scan_from_depth(depth, flat_cfg, q=make_q(), exclude_mask=mask)
    assert with_mask.counts.sum() < without.counts.sum()
    assert with_mask.counts.sum() == pytest.approx(without.counts.sum() / 2, rel=0.05)


def test_exclusion_mask_shape_is_checked(flat_cfg: Config) -> None:
    with pytest.raises(ValueError, match="does not match"):
        scan_from_depth(
            flat_depth_result(4.0), flat_cfg, q=make_q(), exclude_mask=np.zeros((4, 4), bool)
        )


def test_nan_depths_never_reach_the_bins(flat_cfg: Config) -> None:
    # Invalid pixels are NaN by contract; they must be dropped, not binned.
    depth = flat_depth_result(4.0)
    depth.depth_m[:, : WIDTH // 2] = np.nan
    depth.valid[:, : WIDTH // 2] = False
    scan = scan_from_depth(depth, flat_cfg, q=make_q())
    assert np.all(np.isfinite(scan.distances[~scan.unknown]))
    assert scan.counts.sum() == pytest.approx(depth.valid.size / 2, rel=0.05)


def test_a_nearer_object_pulls_its_bin_in(flat_cfg: Config) -> None:
    depth = flat_depth_result(6.0)
    # A patch of wall much closer, on the right of the image.
    depth.depth_m[:, int(WIDTH * 0.75) :] = 1.5
    scan = scan_from_depth(depth, flat_cfg, q=make_q())
    n = flat_cfg.obstacles.n_bins
    assert scan.distances[-1] < 2.5
    assert scan.distances[n // 2] > 5.0


def test_percentile_resists_a_single_outlier(cfg_from) -> None:
    # One surviving speckle must not park a phantom obstacle in the bin: that
    # is the entire reason for the 5th percentile over the minimum.
    cfg = cfg_from(
        {
            "mount": {"x_m": 0.0, "y_m": 0.0, "z_m": 0.0},
            "obstacles": {"n_bins": 4, "min_points_per_bin": 1, "height_floor_m": -50.0,
                          "height_ceiling_m": 50.0},
            "occupancy": {"n_bins": 4},
        }
    )
    depth = flat_depth_result(8.0)
    depth.depth_m[HEIGHT // 2, WIDTH // 2] = 0.5  # a single rogue pixel
    scan = scan_from_depth(depth, cfg, q=make_q())
    assert np.nanmin(scan.distances) > 5.0


def test_empty_scan_is_entirely_unknown(cfg: Config) -> None:
    scan = empty_scan(cfg, t_ns=99)
    assert isinstance(scan, ObstacleScan)
    assert scan.t_ns == 99
    assert scan.unknown.all()
    assert np.all(np.isnan(scan.distances))
    assert len(scan) == cfg.obstacles.n_bins


def test_scan_arrays_are_the_expected_dtypes(flat_cfg: Config) -> None:
    scan = scan_from_depth(flat_depth_result(4.0), flat_cfg, q=make_q())
    assert scan.distances.dtype == np.float32
    assert scan.confidence.dtype == np.float32
    assert scan.unknown.dtype == bool
    assert scan.counts.dtype == np.int32


def test_end_to_end_from_a_real_stereo_pair(cfg_from) -> None:
    """Real SGBM output through the real scan path, against a known range."""
    cfg = cfg_from(
        {
            "mount": {"x_m": 0.0, "y_m": 0.0, "z_m": 0.0},
            "depth": {"sgbm": {"num_disparities": 64}},
            "obstacles": {"height_floor_m": -50.0, "height_ceiling_m": 50.0},
        }
    )
    scene = planar_scene(disparity_px=30.0, width=384, height=200)
    depth = SgbmCpuBackend(cfg, q=scene.q_matrix()).infer(
        FrameBundle(left=scene.left, right=scene.right, t_ns=7, seq=0)
    )
    scan = scan_from_depth(depth, cfg, q=scene.q_matrix())
    centre = scan.distances[cfg.obstacles.n_bins // 2]
    assert centre == pytest.approx(scene.depth_for(30.0), rel=0.05)
    # The left border cannot be matched by SGBM, so those bins report unknown
    # rather than inventing a range.
    assert scan.unknown[:2].any()
