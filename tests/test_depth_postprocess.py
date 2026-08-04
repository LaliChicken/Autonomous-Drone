"""Disparity/depth cleanup, with the NaN discipline as the headline."""

from __future__ import annotations

import numpy as np
import pytest

from config import Config
from depth.postprocess import (
    DISP_SCALE,
    NO_DEPTH,
    apply_confidence_floor,
    blank_invalid,
    clamp_range,
    depth_confidence,
    disparity_valid_mask,
    invalid_marker,
    left_right_consistency_mask,
    postprocess_depth,
    remove_speckles,
    to_float_disparity,
)


def test_invalid_marker_matches_opencv() -> None:
    # OpenCV writes (minDisparity - 1) * 16 for an unmatched pixel.
    assert invalid_marker(0) == -16
    assert invalid_marker(5) == 64


def test_valid_mask_rejects_unmatched_pixels() -> None:
    disparity = np.array([[-16, 16, 32, 160]], dtype=np.int16)
    assert disparity_valid_mask(disparity, 0).tolist() == [[False, True, True, True]]


def test_valid_mask_rejects_zero_disparity() -> None:
    # Zero disparity is the point at infinity: matched, but not a range.
    # Letting it through produces an enormous Z that sails past a range clamp.
    disparity = np.array([[0, 8, -16]], dtype=np.int16)
    assert disparity_valid_mask(disparity, 0).tolist() == [[False, True, False]]


def test_to_float_disparity_undoes_the_fixed_point() -> None:
    disparity = np.array([[16, 24, 160]], dtype=np.int16)
    assert to_float_disparity(disparity).tolist() == [[1.0, 1.5, 10.0]]
    assert DISP_SCALE == 16


# --------------------------------------------------------------------------
# The rule: invalid means NaN, never zero.
# --------------------------------------------------------------------------


def test_blank_invalid_writes_nan_not_zero() -> None:
    depth = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    valid = np.array([[True, False, True]])
    out = blank_invalid(depth, valid)
    assert np.isnan(out[0, 1])
    assert out[0, 1] != 0.0
    assert out[0, 0] == 1.0 and out[0, 2] == 3.0


def test_nan_fails_every_proximity_test() -> None:
    # Why NaN and not 0.0: a pixel that leaks past a valid check must still
    # fail "is this closer than X", not pass it.
    assert not (NO_DEPTH < 1.0)
    assert not (NO_DEPTH > 1.0)
    assert not (np.float32(0.0) > 1.0)
    assert np.float32(0.0) < 1.0  # zero would read as an obstacle at the lens


def test_postprocess_blanks_every_invalid_pixel(cfg: Config) -> None:
    depth = np.array([[0.1, 1.0, 50.0, 3.0]], dtype=np.float32)
    valid = np.ones((1, 4), dtype=bool)
    result = postprocess_depth(depth, valid, 7, 400.0, 0.05, cfg.depth.postprocess)
    assert np.all(np.isnan(result.depth_m[~result.valid]))
    assert not np.any(result.depth_m[~result.valid] == 0.0)
    assert result.t_ns == 7


# --------------------------------------------------------------------------
# Range clamp
# --------------------------------------------------------------------------


def test_clamp_range_invalidates_rather_than_clipping(cfg: Config) -> None:
    post = cfg.depth.postprocess
    depth = np.array([[post.min_depth_m - 0.1, 5.0, post.max_depth_m + 10.0]], dtype=np.float32)
    valid = np.ones((1, 3), dtype=bool)
    out = clamp_range(depth, valid, post)
    assert out.tolist() == [[False, True, False]]
    # A 40 m reading must not become a confident 12 m obstacle.
    assert depth[0, 2] > post.max_depth_m


def test_clamp_range_keeps_the_boundaries(cfg: Config) -> None:
    post = cfg.depth.postprocess
    depth = np.array([[post.min_depth_m, post.max_depth_m]], dtype=np.float32)
    assert clamp_range(depth, np.ones((1, 2), bool), post).all()


def test_clamp_range_kills_non_finite(cfg: Config) -> None:
    depth = np.array([[np.inf, np.nan, -np.inf, 2.0]], dtype=np.float32)
    out = clamp_range(depth, np.ones((1, 4), bool), cfg.depth.postprocess)
    assert out.tolist() == [[False, False, False, True]]


def test_clamp_range_never_revives_an_invalid_pixel(cfg: Config) -> None:
    depth = np.array([[2.0, 2.0]], dtype=np.float32)
    valid = np.array([[True, False]])
    assert clamp_range(depth, valid, cfg.depth.postprocess).tolist() == [[True, False]]


# --------------------------------------------------------------------------
# Speckles
# --------------------------------------------------------------------------


def test_remove_speckles_drops_a_small_blob(cfg: Config) -> None:
    disparity = np.full((40, 40), 32, dtype=np.int16)
    disparity[5:8, 5:8] = 200  # 9-pixel blob, far from its neighbours
    cleaned = remove_speckles(disparity, cfg.depth.postprocess, 0)
    assert (cleaned[5:8, 5:8] == invalid_marker(0)).all()
    # The surrounding plane survives.
    assert (cleaned[20:30, 20:30] == 32).all()


def test_remove_speckles_keeps_a_large_region(cfg: Config) -> None:
    disparity = np.full((60, 60), 32, dtype=np.int16)
    disparity[10:50, 10:50] = 200  # 1600 px, well over speckle_max_area_px
    cleaned = remove_speckles(disparity, cfg.depth.postprocess, 0)
    assert (cleaned[20:40, 20:40] == 200).all()


def test_remove_speckles_does_not_mutate_its_input(cfg: Config) -> None:
    disparity = np.full((30, 30), 32, dtype=np.int16)
    disparity[5:8, 5:8] = 200
    original = disparity.copy()
    remove_speckles(disparity, cfg.depth.postprocess, 0)
    assert np.array_equal(disparity, original)


def test_remove_speckles_disabled_by_zero_area(cfg_from) -> None:
    cfg = cfg_from({"depth": {"postprocess": {"speckle_max_area_px": 0}}})
    disparity = np.full((20, 20), 32, dtype=np.int16)
    disparity[2:4, 2:4] = 200
    assert np.array_equal(remove_speckles(disparity, cfg.depth.postprocess, 0), disparity)


# --------------------------------------------------------------------------
# Left/right consistency
# --------------------------------------------------------------------------


def test_lr_check_accepts_an_agreeing_match() -> None:
    left = np.full((1, 10), 3.0, dtype=np.float32)
    right = np.full((1, 10), 3.0, dtype=np.float32)
    mask = left_right_consistency_mask(left, right, 1.0)
    # Columns 0..2 have no in-bounds partner at disparity 3.
    assert mask[0, 3:].all()
    assert not mask[0, :3].any()


def test_lr_check_rejects_a_disagreeing_match() -> None:
    left = np.full((1, 10), 3.0, dtype=np.float32)
    right = np.full((1, 10), 9.0, dtype=np.float32)
    assert not left_right_consistency_mask(left, right, 1.0)[0, 3:].any()


def test_lr_check_honours_the_tolerance() -> None:
    left = np.full((1, 8), 4.0, dtype=np.float32)
    right = np.full((1, 8), 5.0, dtype=np.float32)
    assert left_right_consistency_mask(left, right, 1.5)[0, 4:].all()
    assert not left_right_consistency_mask(left, right, 0.5)[0, 4:].any()


def test_lr_check_abstains_where_the_right_map_has_no_data() -> None:
    # The right disparity map is structurally invalid at its right edge. If
    # "cannot check" counted as "failed", the right of the FOV would blank on
    # every frame and downstream would be permanently unknown, so the planner
    # could never turn that way.
    left = np.full((1, 10), 3.0, dtype=np.float32)
    right = np.full((1, 10), -1.0, dtype=np.float32)  # invalid everywhere
    right_valid = np.zeros((1, 10), dtype=bool)

    without = left_right_consistency_mask(left, right, 1.0)
    with_abstention = left_right_consistency_mask(left, right, 1.0, right_valid=right_valid)
    assert not without[0, 3:].any()
    assert with_abstention[0, 3:].all()


def test_lr_check_still_rejects_where_it_can_see() -> None:
    left = np.full((1, 10), 3.0, dtype=np.float32)
    right = np.full((1, 10), 9.0, dtype=np.float32)
    right_valid = np.ones((1, 10), dtype=bool)
    right_valid[0, 5:] = False  # only the left half is checkable
    mask = left_right_consistency_mask(left, right, 1.0, right_valid=right_valid)
    # Column x looks up x-3; columns 3..7 look up 0..4 (checkable -> rejected),
    # columns 8..9 look up 5..6 (not checkable -> abstain).
    assert not mask[0, 3:8].any()
    assert mask[0, 8:].all()


def test_lr_check_rejects_out_of_bounds_lookups() -> None:
    left = np.full((1, 6), 20.0, dtype=np.float32)  # disparity exceeds the width
    right = np.full((1, 6), 20.0, dtype=np.float32)
    assert not left_right_consistency_mask(left, right, 1.0).any()


# --------------------------------------------------------------------------
# Confidence
# --------------------------------------------------------------------------


def test_confidence_falls_with_range(cfg: Config) -> None:
    depth = np.array([[1.0, 3.0, 6.0]], dtype=np.float32)
    conf = depth_confidence(depth, np.ones((1, 3), bool), 400.0, 0.05, cfg.depth.postprocess)
    assert conf[0, 0] > conf[0, 1] > conf[0, 2]
    assert np.all((conf >= 0.0) & (conf <= 1.0))


def test_confidence_follows_the_inverse_square_law(cfg: Config) -> None:
    post = cfg.depth.postprocess
    focal, baseline = 400.0, 0.05
    depth = np.array([[2.0]], dtype=np.float32)
    conf = depth_confidence(depth, np.ones((1, 1), bool), focal, baseline, post)
    sigma_z = (2.0**2) * post.disparity_sigma_px / (focal * baseline)
    assert conf[0, 0] == pytest.approx(
        max(0.0, min(1.0, 1.0 - sigma_z / post.max_depth_sigma_m)), rel=1e-5
    )


def test_confidence_is_zero_where_invalid(cfg: Config) -> None:
    depth = np.array([[1.0, 1.0]], dtype=np.float32)
    valid = np.array([[True, False]])
    conf = depth_confidence(depth, valid, 400.0, 0.05, cfg.depth.postprocess)
    assert conf[0, 1] == 0.0
    assert conf[0, 0] > 0.0


def test_confidence_rejects_impossible_geometry(cfg: Config) -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        depth_confidence(
            np.ones((1, 1), np.float32), np.ones((1, 1), bool), 0.0, 0.05, cfg.depth.postprocess
        )


def test_confidence_floor_invalidates_low_confidence(cfg_from) -> None:
    cfg = cfg_from({"depth": {"postprocess": {"min_confidence": 0.5}}})
    valid = np.ones((1, 3), dtype=bool)
    conf = np.array([[0.9, 0.5, 0.1]], dtype=np.float32)
    assert apply_confidence_floor(valid, conf, cfg.depth.postprocess).tolist() == [
        [True, True, False]
    ]


def test_confidence_floor_of_zero_is_a_no_op(cfg: Config) -> None:
    assert cfg.depth.postprocess.min_confidence == 0.0
    valid = np.array([[True, False]])
    conf = np.zeros((1, 2), dtype=np.float32)
    assert apply_confidence_floor(valid, conf, cfg.depth.postprocess) is valid


def test_postprocess_zeroes_confidence_for_invalid_pixels(cfg: Config) -> None:
    depth = np.array([[2.0, 100.0]], dtype=np.float32)
    result = postprocess_depth(
        depth, np.ones((1, 2), bool), 0, 400.0, 0.05, cfg.depth.postprocess
    )
    assert result.conf is not None
    assert result.conf[0, 1] == 0.0
    assert result.valid.tolist() == [[True, False]]


def test_postprocess_output_dtypes(cfg: Config) -> None:
    result = postprocess_depth(
        np.full((4, 4), 2.0, np.float32),
        np.ones((4, 4), bool),
        0,
        400.0,
        0.05,
        cfg.depth.postprocess,
    )
    assert result.depth_m.dtype == np.float32
    assert result.valid.dtype == bool
    assert result.conf is not None and result.conf.dtype == np.float32
