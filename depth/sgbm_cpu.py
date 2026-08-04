"""CPU stereo depth via OpenCV StereoSGBM. Implements the DepthBackend protocol.

Pipeline per frame::

    grayscale -> SGBM disparity (int16 fixed-point)
              -> speckle removal            (disparity domain)
              -> left/right consistency     (disparity domain)
              -> reprojectImageTo3D(Q)      -> Z in metres
              -> range clamp + confidence   (depth domain)

``depth/backends.py`` is an owned stub, so this class satisfies the
``DepthBackend`` protocol from ``sources/types.py`` structurally rather than by
inheriting anything.

The left/right check runs a second SGBM pass on the horizontally flipped pair.
``cv2.ximgproc.createRightMatcher`` would be the tidy way to get a right-view
disparity, but ximgproc is opencv-contrib and the dependency list is closed;
flipping is the standard equivalent and costs one extra match.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from depth.postprocess import (
    disparity_valid_mask,
    left_right_consistency_mask,
    postprocess_depth,
    remove_speckles,
    to_float_disparity,
)
from sources.types import DepthResult, FrameBundle

if TYPE_CHECKING:
    from config import Config
    from infra.metrics import Metrics


def to_gray(image: np.ndarray) -> np.ndarray:
    """Single-channel uint8 view of a frame, whatever it arrived as."""
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 1:
        gray = image[:, :, 0]
    else:
        raise ValueError(f"expected HxW, HxWx1, or HxWx3 image, got shape {image.shape}")
    if gray.dtype != np.uint8:
        raise ValueError(f"expected uint8 image, got {gray.dtype}")
    return np.ascontiguousarray(gray)


class SgbmCpuBackend:
    """StereoSGBM depth backend.

    Holds two matchers: the forward one, and -- when the left/right consistency
    check is enabled -- a second identical matcher for the flipped pair. Both
    are built once, because StereoSGBM construction is not free and infer() is
    on the 30 Hz path.
    """

    def __init__(
        self,
        cfg: Config,
        q: np.ndarray | None = None,
        metrics: Metrics | None = None,
    ) -> None:
        self.cfg = cfg
        self.post = cfg.depth.postprocess
        self.metrics = metrics
        self.q = cfg.camera.load_q() if q is None else np.asarray(q, dtype=np.float64)
        if self.q.shape != (4, 4):
            raise ValueError(f"Q must be 4x4, got {self.q.shape}")

        self._matcher = self._build_matcher()
        self._right_matcher = self._build_matcher() if self.post.lr_consistency else None

        # Q encodes f and B; the confidence model needs them separately.
        # Q[2][3] is f and Q[3][2] is 1/baseline for the standard layout.
        self.focal_px = float(self.q[2][3])
        q32 = float(self.q[3][2])
        self.baseline_m = 1.0 / q32 if q32 != 0.0 else float("nan")

    def _build_matcher(self) -> cv2.StereoSGBM:
        sgbm = self.cfg.depth.sgbm
        mode = (
            cv2.StereoSGBM_MODE_SGBM_3WAY if sgbm.mode_sgbm_3way else cv2.StereoSGBM_MODE_SGBM
        )
        return cv2.StereoSGBM_create(
            minDisparity=sgbm.min_disparity,
            numDisparities=sgbm.num_disparities,
            blockSize=sgbm.block_size,
            P1=sgbm.effective_p1(channels=1),
            P2=sgbm.effective_p2(channels=1),
            disp12MaxDiff=sgbm.disp12_max_diff,
            preFilterCap=sgbm.pre_filter_cap,
            uniquenessRatio=sgbm.uniqueness_ratio,
            speckleWindowSize=sgbm.speckle_window_size,
            speckleRange=sgbm.speckle_range,
            mode=mode,
        )

    def _stage(self, name: str):
        if self.metrics is None:
            from contextlib import nullcontext

            return nullcontext()
        return self.metrics.stage(name)

    def compute_disparity(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Fixed-point int16 disparity for the left image."""
        with self._stage("depth.match"):
            return self._matcher.compute(to_gray(left), to_gray(right))

    def compute_right_disparity(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Fixed-point int16 disparity for the *right* image.

        Matching the flipped pair turns the right view into a left view, so the
        same matcher applies; flipping the result back puts it in right-image
        coordinates with positive disparities.
        """
        if self._right_matcher is None:
            raise RuntimeError("left/right consistency is disabled")
        flipped_left = cv2.flip(to_gray(right), 1)
        flipped_right = cv2.flip(to_gray(left), 1)
        with self._stage("depth.match_right"):
            disparity = self._right_matcher.compute(flipped_left, flipped_right)
        return cv2.flip(disparity, 1)

    def infer(self, f: FrameBundle) -> DepthResult:
        """FrameBundle -> DepthResult. Satisfies sources.types.DepthBackend."""
        if f.left.shape != f.right.shape:
            raise ValueError(
                f"stereo halves must match: left {f.left.shape} vs right {f.right.shape}"
            )
        min_disparity = self.cfg.depth.sgbm.min_disparity

        disparity_fixed = self.compute_disparity(f.left, f.right)

        with self._stage("depth.speckle"):
            disparity_fixed = remove_speckles(disparity_fixed, self.post, min_disparity)

        valid = disparity_valid_mask(disparity_fixed, min_disparity)
        disparity = to_float_disparity(disparity_fixed)

        if self.post.lr_consistency:
            right_fixed = self.compute_right_disparity(f.left, f.right)
            with self._stage("depth.lr_check"):
                consistent = left_right_consistency_mask(
                    disparity,
                    to_float_disparity(right_fixed),
                    self.post.lr_max_disp_diff_px,
                    right_valid=disparity_valid_mask(right_fixed, min_disparity),
                )
            valid = valid & consistent

        with self._stage("depth.reproject"):
            points = cv2.reprojectImageTo3D(disparity, self.q)
        depth_m = np.ascontiguousarray(points[:, :, 2], dtype=np.float32)

        with self._stage("depth.postprocess"):
            return postprocess_depth(
                depth_m=depth_m,
                valid=valid,
                t_ns=f.t_ns,
                focal_px=self.focal_px,
                baseline_m=self.baseline_m,
                cfg=self.post,
            )
