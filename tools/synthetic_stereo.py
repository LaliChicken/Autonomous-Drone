"""Synthetic stereo scenes with exactly known disparity.

Ground truth from a real dataset is the honest benchmark, but it needs a
download. These scenes need nothing: they are generated from a seeded RNG, so
they are identical on every machine, which makes them the right basis for a
*committed* baseline that CI can check without network access.

Construction is deliberately backwards from the usual "shift the left image".
A texture is sampled once, the RIGHT image is a window into it, and the LEFT
image is the same window displaced by the disparity. That way the disparity is
the input rather than something recovered, and both views are exact crops of
one continuous texture with no interpolation blur on either side.

Sign convention matches OpenCV: for a left pixel at column x with disparity d,
the matching right pixel is at ``x - d``. Larger disparity means nearer.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# 640x480 rather than something smaller: StereoSGBM cannot match the leftmost
# numDisparities columns, so at 320 px wide a 128-disparity search writes off
# 40% of the frame and the benchmark measures the border more than the matcher.
# At 640 it is 20%, against 10% for the real 1280 px eye.
DEFAULT_HEIGHT = 480
DEFAULT_WIDTH = 640
DEFAULT_FOCAL_PX = 400.0
DEFAULT_BASELINE_M = 0.052


@dataclass(frozen=True)
class StereoScene:
    """A stereo pair with per-pixel ground truth.

    ``disparity_gt`` is NaN where no correct answer exists -- the left border
    strip that has no right-image counterpart, and genuinely occluded pixels.
    Benchmarks must skip those rather than score against a made-up value.
    """

    name: str
    left: np.ndarray  # HxWx3 uint8
    right: np.ndarray  # HxWx3 uint8
    disparity_gt: np.ndarray  # HxW float32, NaN where undefined
    focal_px: float
    baseline_m: float
    # True where the surface is visible to the left camera only. A subset of
    # the NaN region, kept separate from the left border strip so a benchmark
    # can score "did the matcher hallucinate through an occlusion" on its own.
    occluded: np.ndarray | None = None

    @property
    def occlusion_mask(self) -> np.ndarray:
        if self.occluded is None:
            return np.zeros(self.disparity_gt.shape, dtype=bool)
        return self.occluded

    @property
    def depth_gt(self) -> np.ndarray:
        """Z = f*B/d, NaN wherever disparity is undefined or non-positive."""
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = (self.focal_px * self.baseline_m) / self.disparity_gt
        depth = np.asarray(depth, dtype=np.float32)
        depth[~np.isfinite(depth)] = np.nan
        depth[self.disparity_gt <= 0.0] = np.nan
        return depth

    @property
    def gt_mask(self) -> np.ndarray:
        return np.isfinite(self.disparity_gt)

    def q_matrix(self) -> np.ndarray:
        """Reprojection matrix consistent with this scene's geometry."""
        height, width = self.disparity_gt.shape
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        return np.array(
            [
                [1.0, 0.0, 0.0, -cx],
                [0.0, 1.0, 0.0, -cy],
                [0.0, 0.0, 0.0, self.focal_px],
                [0.0, 0.0, 1.0 / self.baseline_m, 0.0],
            ],
            dtype=np.float64,
        )

    def depth_for(self, disparity_px: float) -> float:
        return (self.focal_px * self.baseline_m) / disparity_px


def random_texture(height: int, width: int, seed: int) -> np.ndarray:
    """Deterministic broadband texture.

    Blurred noise plus a coarse low-frequency layer. Pure per-pixel noise
    matches almost too well and flatters the matcher; blurring gives it
    something with real spatial scale to fail on.
    """
    rng = np.random.default_rng(seed)
    fine = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    fine = cv2.GaussianBlur(fine, (3, 3), 0.8)
    coarse = rng.integers(0, 256, size=(max(1, height // 8), max(1, width // 8)), dtype=np.uint8)
    coarse = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    blended = (0.65 * fine.astype(np.float32) + 0.35 * coarse.astype(np.float32))
    return np.clip(blended, 0, 255).astype(np.uint8)


def _to_bgr(gray: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))


def _sample_pair(
    texture: np.ndarray, disparity: np.ndarray, pad: int
) -> tuple[np.ndarray, np.ndarray]:
    """Crop a left/right pair out of a padded texture given a disparity field.

    ``texture`` is (H, W + pad) wide. The right image is the window starting at
    ``pad``; the left image at column x samples ``pad + x - d(y, x)``, i.e. the
    same surface point seen from a camera displaced to the left.
    Disparities are rounded to integers so both views are exact texture
    samples -- no resampling blur to bias the matcher.
    """
    height, width = disparity.shape
    right = texture[:, pad : pad + width]

    columns = np.arange(width)[None, :]
    source = pad + columns - np.rint(disparity).astype(np.intp)
    source = np.clip(source, 0, texture.shape[1] - 1)
    rows = np.arange(height)[:, None]
    left = texture[rows, source]
    return np.ascontiguousarray(left), np.ascontiguousarray(right)


def planar_scene(
    disparity_px: float = 24.0,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    seed: int = 11,
    name: str | None = None,
) -> StereoScene:
    """Fronto-parallel textured plane: one constant disparity everywhere."""
    pad = int(np.ceil(disparity_px)) + 2
    texture = random_texture(height, width + pad, seed)
    disparity = np.full((height, width), float(disparity_px), dtype=np.float32)
    left, right = _sample_pair(texture, disparity, pad)

    gt = disparity.copy()
    # The leftmost columns have no right-image counterpart at this disparity.
    gt[:, : int(np.ceil(disparity_px))] = np.nan
    return StereoScene(
        name=name or f"planar_d{disparity_px:g}",
        left=_to_bgr(left),
        right=_to_bgr(right),
        disparity_gt=gt,
        focal_px=DEFAULT_FOCAL_PX,
        baseline_m=DEFAULT_BASELINE_M,
    )


def slanted_scene(
    near_disparity_px: float = 40.0,
    far_disparity_px: float = 12.0,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    seed: int = 23,
    name: str = "slanted",
) -> StereoScene:
    """Ground-like plane receding with row: disparity ramps top to bottom."""
    pad = int(np.ceil(max(near_disparity_px, far_disparity_px))) + 2
    texture = random_texture(height, width + pad, seed)
    ramp = np.linspace(far_disparity_px, near_disparity_px, height, dtype=np.float32)
    disparity = np.repeat(ramp[:, None], width, axis=1)
    left, right = _sample_pair(texture, disparity, pad)

    gt = disparity.copy()
    for row in range(height):
        gt[row, : int(np.ceil(ramp[row]))] = np.nan
    return StereoScene(
        name=name,
        left=_to_bgr(left),
        right=_to_bgr(right),
        disparity_gt=gt,
        focal_px=DEFAULT_FOCAL_PX,
        baseline_m=DEFAULT_BASELINE_M,
    )


def occlusion_scene(
    background_disparity_px: float = 10.0,
    foreground_disparity_px: float = 34.0,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    seed: int = 37,
    name: str = "occlusion",
) -> StereoScene:
    """A near slab in front of a far plane, with a genuine occlusion band.

    The slab carries its **own texture**, which is what makes the occlusion
    real. Displacing a region of one continuous texture does not occlude
    anything: the right image is then consistent with both the near and the
    far interpretation, the matcher finds a background match that genuinely
    agrees in both views, and there is nothing for a consistency check to
    catch. Two independent textures mean the slab actually hides the
    background pixels behind it.

    A left pixel showing background at column x looks for its match at
    ``x - d_bg``. If the slab covers that right-image column, that surface is
    visible to the left camera only, and no correct disparity exists. The band
    is ``d_fg - d_bg`` wide, sits immediately left of the slab, and is marked
    NaN so a benchmark never scores it -- it is exactly what the left/right
    consistency check exists to reject.
    """
    if foreground_disparity_px <= background_disparity_px:
        raise ValueError("foreground must be nearer (larger disparity) than background")

    d_bg = float(background_disparity_px)
    d_fg = float(foreground_disparity_px)
    pad = int(np.ceil(d_fg)) + 2
    background = random_texture(height, width + pad, seed)
    foreground = random_texture(height, width + pad, seed + 991)

    # Slab extent in the RIGHT image.
    x0, x1 = width // 3, (2 * width) // 3
    y0, y1 = height // 5, (4 * height) // 5

    rows = np.arange(height)[:, None]
    columns = np.arange(width)[None, :]

    # Right view: background window with the slab composited in place.
    right = background[:, pad : pad + width].copy()
    right[y0:y1, x0:x1] = foreground[:, pad : pad + width][y0:y1, x0:x1]

    # Left view: background displaced by its disparity, slab displaced by its
    # own. Both sample their texture at the same surface point as the right
    # view does, which is what makes the pair geometrically consistent.
    left = background[rows, pad + columns - int(round(d_bg))].copy()
    lx0 = int(round(x0 + d_fg))
    lx1 = min(width, int(round(x1 + d_fg)))
    if lx1 > lx0:
        slab_columns = np.arange(lx0, lx1)[None, :]
        left[y0:y1, lx0:lx1] = foreground[
            np.arange(y0, y1)[:, None], pad + slab_columns - int(round(d_fg))
        ]

    disparity = np.full((height, width), d_bg, dtype=np.float32)
    disparity[y0:y1, lx0:lx1] = d_fg

    gt = disparity.copy()
    gt[:, : int(np.ceil(d_bg))] = np.nan
    # Occluded: background in the left view, hidden by the slab in the right.
    occ0 = int(round(x0 + d_bg))
    occ1 = min(width, lx0)
    occluded = np.zeros((height, width), dtype=bool)
    if occ1 > occ0:
        gt[y0:y1, occ0:occ1] = np.nan
        occluded[y0:y1, occ0:occ1] = True
    return StereoScene(
        name=name,
        left=_to_bgr(left),
        right=_to_bgr(right),
        disparity_gt=gt,
        focal_px=DEFAULT_FOCAL_PX,
        baseline_m=DEFAULT_BASELINE_M,
        occluded=occluded,
    )


def default_scenes() -> list[StereoScene]:
    """The scene set the committed baseline is measured over."""
    return [
        planar_scene(disparity_px=16.0, name="planar_far"),
        planar_scene(disparity_px=48.0, seed=13, name="planar_near"),
        slanted_scene(),
        occlusion_scene(),
    ]
