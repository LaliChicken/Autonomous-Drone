"""Depth image -> azimuth-binned obstacle scan in the body frame.

    valid depth pixels
      -> 3D points (camera frame, from Q)
      -> body frame (mount extrinsic)
      -> height gate
      -> azimuth bins across the FOV
      -> per bin: 5th-percentile range, point count, unknown flag

A bin with too few points is ``unknown=True`` with a NaN range. It is not
"clear", and it is not "far away": both of those are claims, and the whole
point is that nothing was measured there. NaN carries that through arithmetic
-- every comparison against it is False, so an unknown bin fails
"clearance > threshold" without anyone having to remember to check the flag
first.

The 5th percentile rather than the minimum: the minimum of half a million
noisy points is whatever the single worst outlier happened to be, and one
speckle that survived filtering would park a phantom obstacle in the bin. The
5th percentile still answers "how close is the near surface here" while
needing a few hundred pixels to agree before it moves.

Ground masking is owned elsewhere (``perception/ground_mask.py``), so this
module takes an optional exclusion mask and applies it. It never tries to
work out what the ground is.

# QUESTION(rahul): the FOV is not symmetric in practice and the config does
# not admit it. StereoSGBM cannot match the leftmost ``numDisparities``
# columns of the left image, so with 128 disparities on a 1280 px eye the
# left ~10% of the frame -- about 6.5 deg of the 65 deg FOV, roughly the
# leftmost 3 of 32 bins -- is unknown on every single frame, forever. Since
# unknown is impassable, the planner permanently sees a wall in its left
# periphery and will be biased against left turns. Options: narrow
# obstacles.fov_deg to the wedge that is actually observable so the bins are
# honest; keep the bins and accept the bias; or run a second matcher with the
# right image as reference to recover that strip. Which do you want? This
# needs a call on flight behaviour, so I have not picked one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from sources.types import DepthResult

if TYPE_CHECKING:
    from config import Config, ObstaclesConfig


@dataclass(frozen=True)
class ObstacleScan:
    """One frame's worth of binned obstacle evidence.

    Deliberately *not* an ``OccupancySnapshot``: there is no ``danger`` field,
    because danger depends on how fast the aircraft is going and perception
    has no business knowing that. ``world.occupancy`` adds it.
    """

    t_ns: int
    bearings: np.ndarray  # N bin centres, rad, body frame, right-positive
    distances: np.ndarray  # N float32 horizontal range, NaN where unknown
    confidence: np.ndarray  # N float32 0..1
    unknown: np.ndarray  # N bool
    counts: np.ndarray  # N int32 points that landed in the bin

    def __len__(self) -> int:
        return int(self.bearings.size)


def bin_edges(n_bins: int, fov_rad: float) -> np.ndarray:
    """``n_bins + 1`` edges spanning the FOV, centred on straight ahead."""
    half = fov_rad / 2.0
    return np.linspace(-half, half, n_bins + 1, dtype=np.float64)


def bin_centres(n_bins: int, fov_rad: float) -> np.ndarray:
    edges = bin_edges(n_bins, fov_rad)
    return (edges[:-1] + edges[1:]) / 2.0


def camera_intrinsics_from_q(q: np.ndarray) -> tuple[float, float, float]:
    """Recover (focal_px, cx, cy) from a standard reprojection matrix."""
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (4, 4):
        raise ValueError(f"Q must be 4x4, got {q.shape}")
    return float(q[2][3]), float(-q[0][3]), float(-q[1][3])


def depth_to_points_camera(
    depth_m: np.ndarray, q: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-pixel camera-frame coordinates (x right, y down, z forward).

    Returns three HxW float32 arrays rather than an HxWx3 stack, because every
    consumer here immediately flattens and masks them and the stack would just
    be copied apart again.
    """
    focal, cx, cy = camera_intrinsics_from_q(q)
    if focal == 0.0:
        raise ValueError("Q has zero focal length")
    height, width = depth_m.shape[:2]
    u = np.arange(width, dtype=np.float32)[None, :]
    v = np.arange(height, dtype=np.float32)[:, None]
    z = np.asarray(depth_m, dtype=np.float32)
    x = (u - np.float32(cx)) * z / np.float32(focal)
    y = (v - np.float32(cy)) * z / np.float32(focal)
    return x, y, z


def points_camera_to_body(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, rotation: np.ndarray, translation: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate and translate flattened camera-frame points into the body frame."""
    rotation = np.asarray(rotation, dtype=np.float32)
    translation = np.asarray(translation, dtype=np.float32)
    bx = rotation[0, 0] * x + rotation[0, 1] * y + rotation[0, 2] * z + translation[0]
    by = rotation[1, 0] * x + rotation[1, 1] * y + rotation[1, 2] * z + translation[1]
    bz = rotation[2, 0] * x + rotation[2, 1] * y + rotation[2, 2] * z + translation[2]
    return bx, by, bz


def _percentile_per_bin(
    bin_index: np.ndarray, values: np.ndarray, n_bins: int, percentile: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin percentile and count, via one sort rather than N masks.

    Masking per bin costs N passes over the full point cloud; at 1280x720 that
    is tens of millions of comparisons per frame. One argsort plus searchsorted
    gets the same answer in a single pass.
    """
    counts = np.bincount(bin_index, minlength=n_bins).astype(np.int32)
    result = np.full(n_bins, np.nan, dtype=np.float32)
    if bin_index.size == 0:
        return result, counts

    order = np.argsort(bin_index, kind="stable")
    sorted_bins = bin_index[order]
    sorted_values = values[order]
    starts = np.searchsorted(sorted_bins, np.arange(n_bins), side="left")
    ends = np.searchsorted(sorted_bins, np.arange(n_bins), side="right")

    for index in range(n_bins):
        segment = sorted_values[starts[index] : ends[index]]
        if segment.size:
            result[index] = np.percentile(segment, percentile)
    return result, counts


def scan_from_depth(
    depth: DepthResult,
    cfg: Config,
    q: np.ndarray | None = None,
    exclude_mask: np.ndarray | None = None,
) -> ObstacleScan:
    """Turn a DepthResult into an azimuth scan in the body frame.

    ``exclude_mask`` is an HxW boolean where True means "ignore this pixel".
    It is how ground masking arrives: ``perception/ground_mask.py`` is owned,
    so this module accepts its output and never infers the ground itself.
    """
    obstacles: ObstaclesConfig = cfg.obstacles
    n_bins = obstacles.n_bins
    q = cfg.camera.load_q() if q is None else q

    valid = np.asarray(depth.valid, dtype=bool)
    if exclude_mask is not None:
        exclude_mask = np.asarray(exclude_mask, dtype=bool)
        if exclude_mask.shape != valid.shape:
            raise ValueError(
                f"exclude_mask shape {exclude_mask.shape} does not match depth {valid.shape}"
            )
        valid = valid & ~exclude_mask

    x, y, z = depth_to_points_camera(depth.depth_m, q)
    # Flatten and drop invalid pixels before any of the expensive work; NaN
    # depths are excluded here, so nothing downstream has to tolerate them.
    keep = valid.reshape(-1) & np.isfinite(depth.depth_m.reshape(-1))
    xf, yf, zf = x.reshape(-1)[keep], y.reshape(-1)[keep], z.reshape(-1)[keep]

    rotation = cfg.mount.rotation_body_from_cam()
    translation = cfg.mount.translation()
    bx, by, bz = points_camera_to_body(xf, yf, zf, rotation, translation)

    # Height gate. Body z is DOWN, so height above the aircraft is -z.
    height = -bz
    in_slab = (height >= obstacles.height_floor_m) & (height <= obstacles.height_ceiling_m)

    bearing = np.arctan2(by, bx)
    horizontal_range = np.hypot(bx, by)

    edges = bin_edges(n_bins, obstacles.fov_rad)
    in_fov = (bearing >= edges[0]) & (bearing <= edges[-1])
    selected = in_slab & in_fov

    bearing = bearing[selected]
    horizontal_range = horizontal_range[selected].astype(np.float32)

    # searchsorted gives 1..n_bins for in-range bearings; shift to 0-based and
    # clamp the exact right edge back into the last bin.
    index = np.searchsorted(edges, bearing, side="right") - 1
    index = np.clip(index, 0, n_bins - 1).astype(np.intp)

    distances, counts = _percentile_per_bin(
        index, horizontal_range, n_bins, obstacles.distance_percentile
    )

    unknown = counts < obstacles.min_points_per_bin
    confidence = np.clip(
        counts.astype(np.float32) / float(obstacles.points_for_full_confidence), 0.0, 1.0
    )
    # An unknown bin asserts nothing: no range, no confidence.
    distances[unknown] = np.nan
    confidence[unknown] = 0.0

    return ObstacleScan(
        t_ns=int(depth.t_ns),
        bearings=bin_centres(n_bins, obstacles.fov_rad),
        distances=distances.astype(np.float32),
        confidence=confidence.astype(np.float32),
        unknown=np.asarray(unknown, dtype=bool),
        counts=counts,
    )


def empty_scan(cfg: Config, t_ns: int) -> ObstacleScan:
    """An all-unknown scan. What you get when there is no depth at all."""
    n_bins = cfg.obstacles.n_bins
    return ObstacleScan(
        t_ns=int(t_ns),
        bearings=bin_centres(n_bins, cfg.obstacles.fov_rad),
        distances=np.full(n_bins, np.nan, dtype=np.float32),
        confidence=np.zeros(n_bins, dtype=np.float32),
        unknown=np.ones(n_bins, dtype=bool),
        counts=np.zeros(n_bins, dtype=np.int32),
    )
