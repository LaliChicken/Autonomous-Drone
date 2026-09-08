"""Disparity and depth cleanup: speckles, L/R consistency, range, confidence.

The one rule everything here serves: **an unmeasurable pixel is `valid=False`
with `depth_m = NaN`, never `depth_m = 0.0`**. Zero metres reads downstream as
an obstacle pressed against the lens, and NaN cannot be mistaken for a
measurement -- every comparison against it is False, so a pixel that leaks past
a `valid` check still fails "is this closer than X" rather than passing it.

Disparity-domain work (speckles, L/R check) happens on OpenCV's int16
fixed-point disparity, which carries 4 fractional bits. That is where
``cv2.filterSpeckles`` is defined -- it rejects float32 outright -- and it is
also where the thresholds are meaningful: a fixed metre tolerance corresponds
to a different disparity step at every range, so a speckle threshold cannot
honestly be expressed in metres.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from sources.types import DepthResult

if TYPE_CHECKING:
    from config import PostprocessConfig

# OpenCV's fixed-point disparity carries 4 fractional bits.
DISP_SCALE = 16
NO_DEPTH = np.float32(np.nan)


def invalid_marker(min_disparity: int) -> int:
    """The fixed-point value OpenCV writes for an unmatched pixel."""
    return (min_disparity - 1) * DISP_SCALE


def disparity_valid_mask(disparity_fixed: np.ndarray, min_disparity: int) -> np.ndarray:
    """Pixels StereoSGBM actually matched, and that carry a usable range.

    Two conditions, not one. OpenCV marks unmatched pixels with
    ``(minDisparity - 1) * 16``, but a *matched* disparity of zero is equally
    unusable: zero disparity is the point at infinity, and reprojecting it
    yields an enormous or infinite Z that would sail through a range clamp as
    a real measurement.
    """
    matched = disparity_fixed > invalid_marker(min_disparity)
    positive = disparity_fixed > 0
    return np.asarray(matched & positive)


def to_float_disparity(disparity_fixed: np.ndarray) -> np.ndarray:
    """int16 fixed-point -> float32 disparity in pixels."""
    return disparity_fixed.astype(np.float32) / float(DISP_SCALE)


def remove_speckles(
    disparity_fixed: np.ndarray, cfg: PostprocessConfig, min_disparity: int
) -> np.ndarray:
    """Drop small isolated disparity blobs.

    Returns a new array; the input is not modified. Removed pixels are set to
    the invalid marker so they fail ``disparity_valid_mask`` afterwards rather
    than becoming a plausible-looking disparity.
    """
    if cfg.speckle_max_area_px <= 0:
        return disparity_fixed.copy()
    # An explicit copy, not ascontiguousarray: that returns the input itself
    # when it is already contiguous int16, and filterSpeckles works in place,
    # so the caller's disparity map would be modified behind its back.
    work = np.array(disparity_fixed, dtype=np.int16, copy=True, order="C")
    cv2.filterSpeckles(
        work,
        invalid_marker(min_disparity),
        int(cfg.speckle_max_area_px),
        int(round(cfg.speckle_max_diff_px * DISP_SCALE)),
    )
    return work


def left_right_consistency_mask(
    disparity_left: np.ndarray,
    disparity_right: np.ndarray,
    max_diff_px: float,
    right_valid: np.ndarray | None = None,
) -> np.ndarray:
    """Mask of left-image pixels whose match survives the reverse lookup.

    For a left pixel at column x with disparity d, the matching right-image
    pixel is at column ``x - d``. The match is rejected if the right image's
    own disparity there disagrees by more than ``max_diff_px``.

    This is what catches occlusions: a surface visible only to the left camera
    still gets *some* disparity from the matcher, and only the reverse lookup
    reveals that the right camera never saw it.

    **This is a refutation test, not an evidence source.** Where the right
    disparity map has no data of its own, the check abstains and the pixel
    keeps whatever validity the matcher gave it. That distinction is not
    cosmetic. The right-view disparity map is structurally invalid over its
    rightmost ``numDisparities`` columns -- a geometric consequence of the
    search direction, not a data problem -- so treating "could not check" as
    "failed the check" would blank roughly a tenth of the image width at the
    right edge of every single frame. Downstream those bins would be
    permanently ``unknown``, and since unknown is impassable, the planner
    would never be able to turn right. Abstaining is not the same as assuming
    clear: the pixel is still carrying real evidence from the forward match,
    it simply has no second opinion available.

    Pass ``right_valid`` to enable abstention; omit it and every pixel is
    checked, including against right-image values that are meaningless.
    """
    height, width = disparity_left.shape[:2]
    columns = np.arange(width, dtype=np.float32)[None, :]
    target = columns - disparity_left
    in_bounds = (target >= 0.0) & (target <= width - 1)

    lookup = np.clip(np.rint(target), 0, width - 1).astype(np.intp)
    rows = np.arange(height, dtype=np.intp)[:, None]
    right_at_match = disparity_right[rows, lookup]

    agrees = np.abs(disparity_left - right_at_match) <= max_diff_px
    if right_valid is not None:
        checkable = right_valid[rows, lookup]
        agrees = agrees | ~checkable
    return np.asarray(in_bounds & agrees)


def clamp_range(
    depth_m: np.ndarray, valid: np.ndarray, cfg: PostprocessConfig
) -> np.ndarray:
    """Invalidate depths outside the sensor's honest working range.

    Values are *not* clipped to the limits -- clipping would turn a 40 m
    reading into a confident 12 m obstacle. Out-of-range means "not measured".
    Non-finite depths are invalidated here too, which is what removes the
    infinities produced by near-zero disparity.
    """
    finite = np.isfinite(depth_m)
    in_range = (depth_m >= cfg.min_depth_m) & (depth_m <= cfg.max_depth_m)
    return np.asarray(valid & finite & in_range)


def depth_confidence(
    depth_m: np.ndarray,
    valid: np.ndarray,
    focal_px: float,
    baseline_m: float,
    cfg: PostprocessConfig,
) -> np.ndarray:
    """Per-pixel confidence from stereo range uncertainty.

    Differentiating Z = f*B/d gives ``sigma_Z = Z^2 * sigma_d / (f*B)``: range
    error grows with the *square* of range, which is the dominant fact about
    stereo depth. Confidence falls linearly from 1.0 at zero error to 0.0 at
    ``max_depth_sigma_m``.

    This is a geometric bound, not a match-quality score -- it says how precise
    a correct match can be at that range, not how likely the match is right.
    Invalid pixels get 0.0.
    """
    scale = focal_px * baseline_m
    if scale <= 0.0:
        raise ValueError(f"focal_px * baseline_m must be > 0, got {scale}")
    depth = np.where(valid, depth_m, 0.0).astype(np.float32)
    sigma_z = (depth * depth) * (cfg.disparity_sigma_px / scale)
    conf = 1.0 - (sigma_z / cfg.max_depth_sigma_m)
    return np.where(valid, np.clip(conf, 0.0, 1.0), 0.0).astype(np.float32)


def apply_confidence_floor(
    valid: np.ndarray, conf: np.ndarray, cfg: PostprocessConfig
) -> np.ndarray:
    if cfg.min_confidence <= 0.0:
        return valid
    return np.asarray(valid & (conf >= cfg.min_confidence))


def blank_invalid(depth_m: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Force every invalid pixel to NaN.

    The last line of defence for the rule at the top of this module: whatever
    garbage the matcher or the reprojection left behind, an invalid pixel
    leaves here as NaN and never as a number a consumer could act on.
    """
    out = np.asarray(depth_m, dtype=np.float32).copy()
    out[~valid] = NO_DEPTH
    return out


def postprocess_depth(
    depth_m: np.ndarray,
    valid: np.ndarray,
    t_ns: int,
    focal_px: float,
    baseline_m: float,
    cfg: PostprocessConfig,
) -> DepthResult:
    """Range clamp, confidence, confidence floor, and NaN blanking.

    The disparity-domain steps (speckles, L/R) happen before this, in the
    backend, because they need the fixed-point disparity this function has
    already lost.
    """
    valid = clamp_range(depth_m, valid, cfg)
    conf = depth_confidence(depth_m, valid, focal_px, baseline_m, cfg)
    valid = apply_confidence_floor(valid, conf, cfg)
    conf = np.where(valid, conf, 0.0).astype(np.float32)
    return DepthResult(
        depth_m=blank_invalid(depth_m, valid),
        valid=np.asarray(valid, dtype=bool),
        conf=conf,
        t_ns=int(t_ns),
    )
