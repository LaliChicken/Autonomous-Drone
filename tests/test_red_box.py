"""Red-box detection on synthetic renders."""

from __future__ import annotations

import math

import numpy as np
import pytest

from config import Config
from perception.red_box import (
    RedBoxDetection,
    best_detection,
    clean_mask,
    detect,
    red_mask,
)
from tests.synthetic import DISTRACTOR_COLOURS, RED_BGR, flat_depth, render_scene

WIDTH, HEIGHT = 320, 240
FOCAL = 250.0


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


@pytest.fixture
def straight_cfg(cfg_from) -> Config:
    """No mount rotation, so image angles map straight to body angles."""
    return cfg_from({"mount": {"roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0}})


# --------------------------------------------------------------------------
# The hue wrap -- the thing most likely to be got wrong
# --------------------------------------------------------------------------


def test_mask_catches_both_ends_of_the_hue_circle(cfg: Config) -> None:
    import cv2

    # Two reds that land on opposite sides of the 0/179 wrap.
    low_hue = cv2.cvtColor(np.array([[[0, 0, 200]]], np.uint8), cv2.COLOR_BGR2HSV)[0, 0, 0]
    high = np.zeros((10, 10, 3), np.uint8)
    high[:] = (20, 0, 200)  # slightly blue-shifted red -> high hue
    high_hue = cv2.cvtColor(high, cv2.COLOR_BGR2HSV)[0, 0, 0]
    assert low_hue <= cfg.red_box.hue_hi_1 or low_hue >= cfg.red_box.hue_lo_2
    assert high_hue <= cfg.red_box.hue_hi_1 or high_hue >= cfg.red_box.hue_lo_2

    for colour in ((0, 0, 200), (20, 0, 200)):
        patch = np.full((10, 10, 3), colour, np.uint8)
        assert red_mask(patch, cfg.red_box).any(), f"{colour} not detected as red"


def test_a_single_hue_band_would_miss_half_the_reds(cfg: Config) -> None:
    # Guards the reason for the two-band mask: if the high band were dropped,
    # this colour would vanish.
    import cv2

    patch = np.full((10, 10, 3), (20, 0, 200), np.uint8)
    hue = int(cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[0, 0, 0])
    assert hue >= cfg.red_box.hue_lo_2, "expected this red to sit in the high band"
    only_low = cv2.inRange(
        cv2.cvtColor(patch, cv2.COLOR_BGR2HSV),
        np.array([cfg.red_box.hue_lo_1, cfg.red_box.sat_min, cfg.red_box.val_min], np.uint8),
        np.array([cfg.red_box.hue_hi_1, cfg.red_box.sat_max, cfg.red_box.val_max], np.uint8),
    )
    assert not only_low.any()
    assert red_mask(patch, cfg.red_box).any()


def test_mask_rejects_a_desaturated_red_wash(cfg: Config) -> None:
    # A pinkish wall is red in hue but not in saturation. Without the
    # saturation floor a hue-only threshold would light up the whole frame.
    patch = np.full((10, 10, 3), (170, 170, 200), np.uint8)
    assert not red_mask(patch, cfg.red_box).any()


def test_mask_rejects_a_dark_red(cfg: Config) -> None:
    patch = np.full((10, 10, 3), (0, 0, 30), np.uint8)
    assert not red_mask(patch, cfg.red_box).any()


def test_clean_mask_removes_speckle(cfg: Config) -> None:
    mask = np.zeros((60, 60), np.uint8)
    mask[30:50, 30:50] = 255  # a real blob
    mask[5, 5] = 255  # one stray pixel
    cleaned = clean_mask(mask, cfg.red_box)
    assert cleaned[5, 5] == 0
    assert cleaned[40, 40] == 255


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


def test_finds_a_red_box_on_a_plain_background(straight_cfg: Config) -> None:
    image = render_scene(WIDTH, HEIGHT, box=(130, 100, 60, 60))
    found = detect(image, straight_cfg, t_ns=11, q=make_q())
    assert len(found) == 1
    assert isinstance(found[0], RedBoxDetection)
    assert found[0].t_ns == 11
    x, y, w, h = found[0].bbox
    assert (x, y) == pytest.approx((130, 100), abs=2)
    assert (w, h) == pytest.approx((60, 60), abs=3)


@pytest.mark.parametrize("background", ["grey", "noise", "gradient", "red_ish"])
def test_finds_the_box_over_varied_backgrounds(straight_cfg: Config, background: str) -> None:
    image = render_scene(WIDTH, HEIGHT, box=(130, 100, 60, 60), background=background)
    best = best_detection(image, straight_cfg, t_ns=0, q=make_q())
    assert best is not None, f"missed the box on a {background} background"
    assert best.bbox[2] == pytest.approx(60, abs=6)


def test_finds_nothing_when_there_is_no_box(straight_cfg: Config) -> None:
    image = render_scene(WIDTH, HEIGHT, box=None)
    assert detect(image, straight_cfg, t_ns=0, q=make_q()) == []
    assert best_detection(image, straight_cfg, t_ns=0, q=make_q()) is None


@pytest.mark.parametrize("colour", sorted(DISTRACTOR_COLOURS))
def test_distractor_colours_are_rejected(straight_cfg: Config, colour: str) -> None:
    image = render_scene(WIDTH, HEIGHT, box=None, distractors=[(colour, (100, 80, 70, 70))])
    assert detect(image, straight_cfg, t_ns=0, q=make_q()) == [], f"{colour} was taken for red"


def test_the_box_wins_against_distractors(straight_cfg: Config) -> None:
    image = render_scene(
        WIDTH,
        HEIGHT,
        box=(200, 100, 60, 60),
        distractors=[("orange", (20, 30, 70, 70)), ("magenta", (20, 140, 60, 60))],
    )
    found = detect(image, straight_cfg, t_ns=0, q=make_q())
    assert len(found) == 1
    assert found[0].bbox[0] == pytest.approx(200, abs=3)


def test_a_partially_occluded_box_is_still_found(straight_cfg: Config) -> None:
    image = render_scene(
        WIDTH, HEIGHT, box=(130, 100, 70, 70), occluder=(130, 100, 20, 70)
    )
    best = best_detection(image, straight_cfg, t_ns=0, q=make_q())
    assert best is not None
    # The visible part is narrower than the real box.
    assert best.bbox[2] < 70


def test_a_heavily_occluded_box_fails_the_shape_filters(straight_cfg: Config) -> None:
    # Occluding the middle splits it into two thin slivers; neither should
    # pass the aspect filter, so the detector reports nothing rather than
    # guessing.
    image = render_scene(
        WIDTH, HEIGHT, box=(130, 100, 70, 70), occluder=(140, 100, 50, 70)
    )
    for found in detect(image, straight_cfg, t_ns=0, q=make_q()):
        assert found.bbox[2] >= 0.5 * found.bbox[3]


def test_a_too_small_box_is_ignored(straight_cfg: Config) -> None:
    image = render_scene(WIDTH, HEIGHT, box=(150, 120, 8, 8))
    assert detect(image, straight_cfg, t_ns=0, q=make_q()) == []


def test_a_wrong_aspect_ratio_is_ignored(straight_cfg: Config) -> None:
    image = render_scene(WIDTH, HEIGHT, box=(40, 110, 240, 20))  # 12:1 strip
    assert detect(image, straight_cfg, t_ns=0, q=make_q()) == []


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def test_a_centred_box_is_dead_ahead(straight_cfg: Config) -> None:
    box_size = 60
    x = (WIDTH - box_size) // 2
    y = (HEIGHT - box_size) // 2
    best = best_detection(
        render_scene(WIDTH, HEIGHT, box=(x, y, box_size, box_size)),
        straight_cfg,
        t_ns=0,
        q=make_q(),
    )
    assert best is not None
    assert best.bearing_rad == pytest.approx(0.0, abs=0.02)
    assert best.elevation_rad == pytest.approx(0.0, abs=0.02)


def test_a_box_to_the_right_has_a_positive_bearing(straight_cfg: Config) -> None:
    best = best_detection(
        render_scene(WIDTH, HEIGHT, box=(240, 100, 50, 50)), straight_cfg, t_ns=0, q=make_q()
    )
    assert best is not None
    assert best.bearing_rad > 0.1
    # atan of the pixel offset over the focal length.
    centre_u = 240 + 25
    expected = math.atan((centre_u - (WIDTH - 1) / 2.0) / FOCAL)
    assert best.bearing_rad == pytest.approx(expected, rel=0.05)


def test_a_box_to_the_left_has_a_negative_bearing(straight_cfg: Config) -> None:
    best = best_detection(
        render_scene(WIDTH, HEIGHT, box=(30, 100, 50, 50)), straight_cfg, t_ns=0, q=make_q()
    )
    assert best is not None
    assert best.bearing_rad < -0.1


def test_a_box_high_in_the_frame_has_a_positive_elevation(straight_cfg: Config) -> None:
    # Image y grows downwards; elevation is up-positive.
    best = best_detection(
        render_scene(WIDTH, HEIGHT, box=(130, 20, 50, 50)), straight_cfg, t_ns=0, q=make_q()
    )
    assert best is not None
    assert best.elevation_rad > 0.1


def test_mount_yaw_shifts_the_reported_bearing(cfg_from) -> None:
    # The detection is in the body frame, so bolting the camera at an angle
    # must move the bearing, not leave it in camera coordinates.
    box = (130, 100, 60, 60)
    image = render_scene(WIDTH, HEIGHT, box=box)
    straight = cfg_from({"mount": {"yaw_deg": 0.0}})
    turned = cfg_from({"mount": {"yaw_deg": 20.0}})
    a = best_detection(image, straight, t_ns=0, q=make_q())
    b = best_detection(image, turned, t_ns=0, q=make_q())
    assert a is not None and b is not None
    assert b.bearing_rad - a.bearing_rad == pytest.approx(math.radians(20.0), abs=0.02)


# --------------------------------------------------------------------------
# Ranging
# --------------------------------------------------------------------------


def test_range_from_size_uses_the_configured_width(straight_cfg: Config) -> None:
    box_px = 60
    best = best_detection(
        render_scene(WIDTH, HEIGHT, box=(130, 100, box_px, box_px)),
        straight_cfg,
        t_ns=0,
        q=make_q(),
    )
    assert best is not None
    expected = FOCAL * straight_cfg.red_box.box_width_m / best.bbox[2]
    assert best.range_size_m == pytest.approx(expected, rel=1e-6)


def test_a_smaller_box_reads_as_further_away(straight_cfg: Config) -> None:
    near = best_detection(
        render_scene(WIDTH, HEIGHT, box=(130, 100, 80, 80)), straight_cfg, t_ns=0, q=make_q()
    )
    far = best_detection(
        render_scene(WIDTH, HEIGHT, box=(140, 110, 30, 30)), straight_cfg, t_ns=0, q=make_q()
    )
    assert near is not None and far is not None
    assert far.range_size_m > near.range_size_m


def test_stereo_range_is_the_median_depth_in_the_box(straight_cfg: Config) -> None:
    image = render_scene(WIDTH, HEIGHT, box=(130, 100, 60, 60))
    depth = flat_depth(WIDTH, HEIGHT, depth_m=3.5)
    best = best_detection(image, straight_cfg, t_ns=0, depth=depth, q=make_q())
    assert best is not None
    assert best.range_stereo_m == pytest.approx(3.5)


def test_stereo_range_is_none_without_depth(straight_cfg: Config) -> None:
    best = best_detection(
        render_scene(WIDTH, HEIGHT, box=(130, 100, 60, 60)), straight_cfg, t_ns=0, q=make_q()
    )
    assert best is not None
    assert best.range_stereo_m is None
    assert best.range_agreement is None
    assert best.best_range_m == best.range_size_m


def test_stereo_range_is_none_when_nothing_in_the_box_is_valid(straight_cfg: Config) -> None:
    depth = flat_depth(WIDTH, HEIGHT, valid=False)
    best = best_detection(
        render_scene(WIDTH, HEIGHT, box=(130, 100, 60, 60)),
        straight_cfg,
        t_ns=0,
        depth=depth,
        q=make_q(),
    )
    assert best is not None
    assert best.range_stereo_m is None


def test_stereo_range_ignores_invalid_pixels(straight_cfg: Config) -> None:
    # Invalid depth is NaN by contract; a median that included it would be NaN.
    depth = flat_depth(WIDTH, HEIGHT, depth_m=3.5)
    depth.depth_m[100:120, 130:160] = np.nan
    depth.valid[100:120, 130:160] = False
    best = best_detection(
        render_scene(WIDTH, HEIGHT, box=(130, 100, 60, 60)),
        straight_cfg,
        t_ns=0,
        depth=depth,
        q=make_q(),
    )
    assert best is not None
    assert best.range_stereo_m == pytest.approx(3.5)


def test_range_agreement_is_small_when_both_estimates_agree(straight_cfg: Config) -> None:
    box_px = 60
    image = render_scene(WIDTH, HEIGHT, box=(130, 100, box_px, box_px))
    truthful = FOCAL * straight_cfg.red_box.box_width_m / box_px
    best = best_detection(
        image, straight_cfg, t_ns=0, depth=flat_depth(WIDTH, HEIGHT, truthful), q=make_q()
    )
    assert best is not None
    assert best.range_agreement == pytest.approx(0.0, abs=0.01)


def test_range_agreement_exposes_a_disagreement(straight_cfg: Config) -> None:
    # The two estimates fail for unrelated reasons, so their disagreement is
    # the useful signal. Blending them into one number would hide this.
    image = render_scene(WIDTH, HEIGHT, box=(130, 100, 60, 60))
    best = best_detection(
        image, straight_cfg, t_ns=0, depth=flat_depth(WIDTH, HEIGHT, 10.0), q=make_q()
    )
    assert best is not None
    assert best.range_agreement is not None and best.range_agreement > 0.5
    assert best.best_range_m == pytest.approx(10.0)  # stereo wins when present


# --------------------------------------------------------------------------
# Confidence and reporting
# --------------------------------------------------------------------------


def test_confidence_is_bounded_and_high_for_a_clean_square(straight_cfg: Config) -> None:
    best = best_detection(
        render_scene(WIDTH, HEIGHT, box=(130, 100, 60, 60)), straight_cfg, t_ns=0, q=make_q()
    )
    assert best is not None
    assert 0.0 <= best.confidence <= 1.0
    assert best.confidence > 0.9
    assert best.solidity > 0.9
    assert best.fill_ratio > 0.9


def test_detections_are_sorted_best_first(straight_cfg: Config) -> None:
    image = render_scene(
        WIDTH,
        HEIGHT,
        box=(200, 100, 60, 60),
        distractors=[],
    )
    # Add a second, ragged red blob.
    import cv2

    cv2.circle(image, (60, 60), 22, RED_BGR, -1)
    cv2.circle(image, (60, 60), 8, (128, 128, 128), -1)  # punch a hole in it
    found = detect(image, straight_cfg, t_ns=0, q=make_q())
    assert len(found) >= 2
    assert found[0].confidence >= found[-1].confidence


def test_border_contact_is_reported(straight_cfg: Config) -> None:
    # A clipped box makes range_size wrong, so callers need to know.
    flush = best_detection(
        render_scene(WIDTH, HEIGHT, box=(0, 100, 60, 60)), straight_cfg, t_ns=0, q=make_q()
    )
    inside = best_detection(
        render_scene(WIDTH, HEIGHT, box=(130, 100, 60, 60)), straight_cfg, t_ns=0, q=make_q()
    )
    assert flush is not None and flush.touches_border
    assert inside is not None and not inside.touches_border


def test_min_confidence_filters_candidates(cfg_from) -> None:
    image = render_scene(WIDTH, HEIGHT, box=(130, 100, 60, 60))
    permissive = cfg_from({"red_box": {"min_confidence": 0.1}})
    strict = cfg_from({"red_box": {"min_confidence": 0.999}})
    assert detect(image, permissive, t_ns=0, q=make_q())
    assert detect(image, strict, t_ns=0, q=make_q()) == []
