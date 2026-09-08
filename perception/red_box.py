"""Find the red target box: HSV threshold, morphology, contour filters.

Red is the awkward hue. It sits at both ends of the OpenCV 0..179 hue circle,
so a single ``inRange`` cannot express it and the mask is the union of a low
band (0..10) and a high band (170..179). A detector written with one band
silently loses half the reds -- typically the *saturated* half, which is
exactly the target.

Two independent range estimates come out of a detection:

``range_stereo``  median valid depth inside the box. Needs texture and needs
                  the box to be inside the stereo overlap, but does not care
                  how big the box actually is.
``range_size``    ``f * box_width_m / box_width_px``. Always available, but
                  wrong the moment the real box is not the configured width,
                  or the box is clipped by the frame edge.

Both are reported rather than blended. They fail in unrelated ways, so their
*disagreement* is diagnostic -- ``range_agreement`` carries it -- and folding
them into one number would throw that away.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from sources.types import DepthResult

if TYPE_CHECKING:
    from config import Config, RedBoxConfig


@dataclass(frozen=True)
class RedBoxDetection:
    """One candidate, in body-frame angles.

    ``bearing_rad`` is right-positive and ``elevation_rad`` is up-positive,
    both in the body frame, so the mount extrinsic has already been applied
    and a consumer never needs to know where the camera is bolted.
    """

    t_ns: int
    bearing_rad: float
    elevation_rad: float
    range_stereo_m: float | None
    range_size_m: float
    confidence: float
    bbox: tuple[int, int, int, int]  # x, y, w, h
    area_px: float
    solidity: float
    fill_ratio: float
    touches_border: bool

    @property
    def range_agreement(self) -> float | None:
        """|stereo - size| / stereo, or None when there is no stereo range.

        Near zero means the two independent estimates agree. Large means one
        of them is wrong and the detection deserves suspicion.
        """
        if self.range_stereo_m is None or self.range_stereo_m <= 0.0:
            return None
        return abs(self.range_stereo_m - self.range_size_m) / self.range_stereo_m

    @property
    def best_range_m(self) -> float:
        """Range to act on: stereo when available, else the size estimate."""
        return self.range_size_m if self.range_stereo_m is None else self.range_stereo_m


def red_mask(image_bgr: np.ndarray, cfg: RedBoxConfig) -> np.ndarray:
    """Binary mask of red pixels, as the union of the two hue bands."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    low = cv2.inRange(
        hsv,
        np.array([cfg.hue_lo_1, cfg.sat_min, cfg.val_min], dtype=np.uint8),
        np.array([cfg.hue_hi_1, cfg.sat_max, cfg.val_max], dtype=np.uint8),
    )
    high = cv2.inRange(
        hsv,
        np.array([cfg.hue_lo_2, cfg.sat_min, cfg.val_min], dtype=np.uint8),
        np.array([cfg.hue_hi_2, cfg.sat_max, cfg.val_max], dtype=np.uint8),
    )
    return cv2.bitwise_or(low, high)


def clean_mask(mask: np.ndarray, cfg: RedBoxConfig) -> np.ndarray:
    """Open then close.

    Open first: it removes isolated red speckle, and doing it after the close
    would first have glued that speckle onto the target. Close afterwards
    fills the holes left by specular highlights on the box face.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (cfg.morph_kernel_px, cfg.morph_kernel_px)
    )
    out = mask
    if cfg.morph_open_iterations:
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel, iterations=cfg.morph_open_iterations)
    if cfg.morph_close_iterations:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=cfg.morph_close_iterations)
    return out


def _solidity(contour: np.ndarray, area: float) -> float:
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    if hull_area <= 0.0:
        return 0.0
    return float(area / hull_area)


def _pixel_ray_body(
    u: float, v: float, focal: float, cx: float, cy: float, rotation: np.ndarray
) -> np.ndarray:
    """Unit ray through a pixel, expressed in the body frame."""
    ray_camera = np.array([(u - cx) / focal, (v - cy) / focal, 1.0], dtype=np.float64)
    ray_body = rotation @ ray_camera
    norm = float(np.linalg.norm(ray_body))
    return ray_body / norm if norm > 0.0 else ray_body


def detect(
    image_bgr: np.ndarray,
    cfg: Config,
    t_ns: int,
    depth: DepthResult | None = None,
    q: np.ndarray | None = None,
    rotation_body_from_cam: np.ndarray | None = None,
) -> list[RedBoxDetection]:
    """All candidates passing the filters, best confidence first.

    ``depth`` is optional: the detector is useful on colour alone, and the
    stereo range is an enrichment rather than a requirement.
    """
    box: RedBoxConfig = cfg.red_box
    q = cfg.camera.load_q() if q is None else np.asarray(q, dtype=np.float64)
    focal, cx, cy = float(q[2][3]), float(-q[0][3]), float(-q[1][3])
    rotation = (cfg.mount.rotation_body_from_cam() if rotation_body_from_cam is None
                else rotation_body_from_cam)

    height, width = image_bgr.shape[:2]
    mask = clean_mask(red_mask(image_bgr, box), box)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections: list[RedBoxDetection] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < box.min_area_px or area > box.max_area_px:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue
        aspect = w / h
        if not (box.min_aspect <= aspect <= box.max_aspect):
            continue

        solidity = _solidity(contour, area)
        if solidity < box.min_solidity:
            continue

        fill_ratio = float(area / (w * h))
        # Both cues are already in 0..1 and a genuine square scores high on
        # each, so the product needs no weights -- and weights would be two
        # more tunables with no principled value.
        confidence = float(np.clip(solidity * fill_ratio, 0.0, 1.0))
        if confidence < box.min_confidence:
            continue

        centre_u = x + w / 2.0
        centre_v = y + h / 2.0
        ray = _pixel_ray_body(centre_u, centre_v, focal, cx, cy, rotation)
        bearing = float(np.arctan2(ray[1], ray[0]))
        elevation = float(np.arctan2(-ray[2], float(np.hypot(ray[0], ray[1]))))

        range_size = float(focal * box.box_width_m / w)
        range_stereo = _median_depth_in_box(depth, x, y, w, h)
        touches_border = x <= 0 or y <= 0 or (x + w) >= width or (y + h) >= height

        detections.append(
            RedBoxDetection(
                t_ns=int(t_ns),
                bearing_rad=bearing,
                elevation_rad=elevation,
                range_stereo_m=range_stereo,
                range_size_m=range_size,
                confidence=confidence,
                bbox=(int(x), int(y), int(w), int(h)),
                area_px=area,
                solidity=solidity,
                fill_ratio=fill_ratio,
                touches_border=touches_border,
            )
        )

    detections.sort(key=lambda d: (d.confidence, d.area_px), reverse=True)
    return detections


def _median_depth_in_box(
    depth: DepthResult | None, x: int, y: int, w: int, h: int
) -> float | None:
    """Median of the valid depths inside the box, or None if there are none.

    Median, not mean: a handful of background pixels around the box edges
    would drag a mean toward the wall behind the target.
    """
    if depth is None:
        return None
    patch_valid = depth.valid[y : y + h, x : x + w]
    patch_depth = depth.depth_m[y : y + h, x : x + w]
    usable = patch_valid & np.isfinite(patch_depth)
    if not usable.any():
        return None
    return float(np.median(patch_depth[usable]))


def best_detection(
    image_bgr: np.ndarray,
    cfg: Config,
    t_ns: int,
    depth: DepthResult | None = None,
    q: np.ndarray | None = None,
    rotation_body_from_cam: np.ndarray | None = None,
) -> RedBoxDetection | None:
    """Highest-confidence candidate, or None."""
    found = detect(image_bgr, cfg, t_ns, depth=depth, q=q,
                   rotation_body_from_cam=rotation_body_from_cam)
    return found[0] if found else None
