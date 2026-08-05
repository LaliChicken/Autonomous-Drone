"""The synthetic scenes themselves.

These are the ground truth the depth baseline is measured against, so their
correctness is checked without going anywhere near StereoSGBM: the pair is
verified by direct pixel comparison against the disparity that generated it.
A generator that quietly disagreed with its own ground truth would make every
downstream number meaningless.
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.synthetic_stereo import (
    DEFAULT_BASELINE_M,
    DEFAULT_FOCAL_PX,
    default_scenes,
    occlusion_scene,
    planar_scene,
    random_texture,
    slanted_scene,
)


def _gray(image: np.ndarray) -> np.ndarray:
    return image[:, :, 0].astype(np.int32)


def test_planar_pair_matches_its_own_ground_truth() -> None:
    disparity = 17.0
    scene = planar_scene(disparity_px=disparity, width=200, height=40)
    left, right = _gray(scene.left), _gray(scene.right)
    d = int(disparity)
    # left[y, x] must be the same surface point as right[y, x - d].
    assert np.array_equal(left[:, d:], right[:, : -d or None])


def test_planar_ground_truth_is_constant_outside_the_border() -> None:
    scene = planar_scene(disparity_px=20.0, width=200, height=40)
    gt = scene.disparity_gt
    assert np.all(np.isnan(gt[:, :20]))
    assert np.all(gt[:, 20:] == 20.0)


def test_depth_ground_truth_follows_the_stereo_equation() -> None:
    scene = planar_scene(disparity_px=25.0, width=200, height=40)
    expected = DEFAULT_FOCAL_PX * DEFAULT_BASELINE_M / 25.0
    depth = scene.depth_gt
    assert np.nanmin(depth) == pytest.approx(expected)
    assert np.nanmax(depth) == pytest.approx(expected)
    assert scene.depth_for(25.0) == pytest.approx(expected)


def test_depth_ground_truth_is_nan_where_disparity_is() -> None:
    scene = planar_scene(disparity_px=20.0, width=200, height=40)
    assert np.array_equal(np.isnan(scene.depth_gt), ~scene.gt_mask)


def test_slanted_scene_ramps_from_far_to_near() -> None:
    scene = slanted_scene(near_disparity_px=40.0, far_disparity_px=10.0, width=200, height=60)
    gt = scene.disparity_gt
    top = np.nanmean(gt[0])
    bottom = np.nanmean(gt[-1])
    assert top == pytest.approx(10.0)
    assert bottom == pytest.approx(40.0)
    # Nearer means larger disparity, so smaller depth at the bottom.
    depth = scene.depth_gt
    assert np.nanmean(depth[0]) > np.nanmean(depth[-1])


def test_slanted_pair_matches_its_ground_truth_row_by_row() -> None:
    scene = slanted_scene(near_disparity_px=30.0, far_disparity_px=10.0, width=200, height=40)
    left, right = _gray(scene.left), _gray(scene.right)
    for row in range(scene.disparity_gt.shape[0]):
        d = int(round(float(np.nanmax(scene.disparity_gt[row]))))
        assert np.array_equal(left[row, d:], right[row, : -d or None]), f"row {row}"


def test_occlusion_scene_marks_a_band_of_the_right_width() -> None:
    scene = occlusion_scene(
        background_disparity_px=10.0, foreground_disparity_px=34.0, width=300, height=100
    )
    occluded = scene.occlusion_mask
    assert occluded.any()
    # The band is d_fg - d_bg wide.
    widths = {int(row.sum()) for row in occluded if row.any()}
    assert widths == {24}
    # And everything marked occluded has no ground truth.
    assert np.all(np.isnan(scene.disparity_gt[occluded]))


def test_occlusion_scene_uses_a_distinct_foreground_texture() -> None:
    # The whole point: displacing a region of one continuous texture occludes
    # nothing, because the right view stays consistent with both depths.
    scene = occlusion_scene(width=300, height=100)
    right = _gray(scene.right)
    x0, x1 = 300 // 3, (2 * 300) // 3
    y0, y1 = 100 // 5, (4 * 100) // 5
    slab = right[y0:y1, x0:x1]
    background = right[y0:y1, :x0]
    # Two independent textures should not be near-identical in distribution
    # *and* pixel alignment; compare the overlap directly.
    overlap = min(slab.shape[1], background.shape[1])
    assert not np.array_equal(slab[:, :overlap], background[:, -overlap:])


def test_occlusion_scene_has_two_distinct_depths() -> None:
    scene = occlusion_scene(
        background_disparity_px=10.0, foreground_disparity_px=34.0, width=300, height=100
    )
    values = np.unique(scene.disparity_gt[np.isfinite(scene.disparity_gt)])
    assert set(values.tolist()) == {10.0, 34.0}


def test_occlusion_scene_rejects_a_foreground_behind_the_background() -> None:
    with pytest.raises(ValueError, match="nearer"):
        occlusion_scene(background_disparity_px=30.0, foreground_disparity_px=10.0)


def test_scenes_without_occlusion_report_an_empty_mask() -> None:
    scene = planar_scene(disparity_px=20.0, width=100, height=40)
    assert scene.occluded is None
    assert scene.occlusion_mask.shape == scene.disparity_gt.shape
    assert not scene.occlusion_mask.any()


def test_q_matrix_round_trips_disparity_to_depth() -> None:
    scene = planar_scene(disparity_px=20.0, width=200, height=40)
    q = scene.q_matrix()
    homogeneous = q @ np.array([100.0, 20.0, 20.0, 1.0])
    z = (homogeneous[:3] / homogeneous[3])[2]
    assert z == pytest.approx(scene.depth_for(20.0), rel=1e-6)


def test_generation_is_deterministic() -> None:
    a = planar_scene(disparity_px=20.0, width=120, height=40, seed=99)
    b = planar_scene(disparity_px=20.0, width=120, height=40, seed=99)
    assert np.array_equal(a.left, b.left)
    assert np.array_equal(a.right, b.right)
    assert np.array_equal(a.disparity_gt, b.disparity_gt, equal_nan=True)


def test_different_seeds_give_different_texture() -> None:
    a = planar_scene(disparity_px=20.0, width=120, height=40, seed=1)
    b = planar_scene(disparity_px=20.0, width=120, height=40, seed=2)
    assert not np.array_equal(a.right, b.right)


def test_texture_has_real_spatial_structure() -> None:
    # Pure per-pixel noise flatters a block matcher; the generator blurs and
    # adds a coarse layer so neighbouring pixels are correlated.
    texture = random_texture(64, 64, seed=3).astype(np.float64)
    neighbour_diff = np.abs(np.diff(texture, axis=1)).mean()
    shuffled = texture.copy()
    np.random.default_rng(0).shuffle(shuffled.reshape(-1))
    assert neighbour_diff < np.abs(np.diff(shuffled, axis=1)).mean()


def test_images_are_the_shape_a_frame_bundle_expects() -> None:
    scene = planar_scene(disparity_px=20.0, width=120, height=40)
    assert scene.left.shape == (40, 120, 3)
    assert scene.left.dtype == np.uint8
    assert scene.right.shape == scene.left.shape


def test_default_scene_set_is_varied() -> None:
    scenes = default_scenes()
    assert len(scenes) == 4
    assert len({s.name for s in scenes}) == 4
    assert any(s.occlusion_mask.any() for s in scenes)
    # A near set and a far set, so the baseline covers both ends of the range.
    medians = [float(np.nanmedian(s.depth_gt)) for s in scenes]
    assert max(medians) / min(medians) > 2.0
