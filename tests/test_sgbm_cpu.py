"""SGBM backend against synthetic scenes with exactly known ground truth."""

from __future__ import annotations

import numpy as np
import pytest

from config import Config
from depth.sgbm_cpu import SgbmCpuBackend, to_gray
from infra.metrics import Metrics
from sources.types import DepthBackend, DepthResult, FrameBundle
from tools.synthetic_stereo import occlusion_scene, planar_scene, slanted_scene

# Small enough to keep the suite quick, wide enough that the numDisparities
# border does not swallow the image.
TEST_WIDTH = 384
TEST_HEIGHT = 160


def _bundle(scene, t_ns: int = 42, seq: int = 3) -> FrameBundle:
    return FrameBundle(left=scene.left, right=scene.right, t_ns=t_ns, seq=seq)


@pytest.fixture(scope="module")
def small_cfg_raw() -> dict:
    import copy

    import yaml

    from config.schema import DEFAULT_CONFIG_PATH

    return copy.deepcopy(yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")))


@pytest.fixture
def narrow_cfg(cfg_from):
    """Fewer disparities, so a 384 px test image still has usable width."""
    return cfg_from({"depth": {"sgbm": {"num_disparities": 64}}})


def test_satisfies_the_depth_backend_protocol(cfg: Config) -> None:
    backend: DepthBackend = SgbmCpuBackend(cfg)
    assert hasattr(backend, "infer")


def test_recovers_a_known_plane(narrow_cfg: Config) -> None:
    scene = planar_scene(disparity_px=24.0, width=TEST_WIDTH, height=TEST_HEIGHT)
    backend = SgbmCpuBackend(narrow_cfg, q=scene.q_matrix())
    result = backend.infer(_bundle(scene))

    expected = scene.depth_for(24.0)
    measured = result.depth_m[result.valid]
    assert measured.size > 0
    assert np.median(measured) == pytest.approx(expected, rel=1e-3)
    assert np.abs(measured - expected).max() < 0.05


def test_recovers_a_second_plane_at_a_different_range(narrow_cfg: Config) -> None:
    scene = planar_scene(disparity_px=48.0, width=TEST_WIDTH, height=TEST_HEIGHT, seed=5)
    result = SgbmCpuBackend(narrow_cfg, q=scene.q_matrix()).infer(_bundle(scene))
    assert np.median(result.depth_m[result.valid]) == pytest.approx(
        scene.depth_for(48.0), rel=1e-3
    )


def test_tracks_a_depth_gradient(narrow_cfg: Config) -> None:
    scene = slanted_scene(
        near_disparity_px=40.0, far_disparity_px=14.0, width=TEST_WIDTH, height=TEST_HEIGHT
    )
    result = SgbmCpuBackend(narrow_cfg, q=scene.q_matrix()).infer(_bundle(scene))
    gt = scene.depth_gt
    scored = result.valid & np.isfinite(gt)
    assert scored.sum() > 1000
    assert np.sqrt(np.mean((result.depth_m[scored] - gt[scored]) ** 2)) < 0.1

    # Top of the frame is far, bottom is near.
    top = result.depth_m[: TEST_HEIGHT // 4][result.valid[: TEST_HEIGHT // 4]]
    bottom = result.depth_m[-TEST_HEIGHT // 4 :][result.valid[-TEST_HEIGHT // 4 :]]
    assert np.median(top) > np.median(bottom)


def test_result_carries_the_frame_timestamp(narrow_cfg: Config) -> None:
    scene = planar_scene(disparity_px=24.0, width=TEST_WIDTH, height=TEST_HEIGHT)
    result = SgbmCpuBackend(narrow_cfg, q=scene.q_matrix()).infer(_bundle(scene, t_ns=987_654_321))
    assert result.t_ns == 987_654_321


def test_invalid_pixels_are_nan_never_zero(narrow_cfg: Config) -> None:
    scene = planar_scene(disparity_px=24.0, width=TEST_WIDTH, height=TEST_HEIGHT)
    result = SgbmCpuBackend(narrow_cfg, q=scene.q_matrix()).infer(_bundle(scene))
    invalid = ~result.valid
    assert invalid.any(), "the numDisparities border should always be invalid"
    assert np.all(np.isnan(result.depth_m[invalid]))
    assert not np.any(result.depth_m[invalid] == 0.0)


def test_the_left_border_is_never_claimed(narrow_cfg: Config) -> None:
    # SGBM cannot search left of column numDisparities; those pixels must come
    # back invalid rather than as some plausible-looking depth.
    scene = planar_scene(disparity_px=24.0, width=TEST_WIDTH, height=TEST_HEIGHT)
    result = SgbmCpuBackend(narrow_cfg, q=scene.q_matrix()).infer(_bundle(scene))
    border = narrow_cfg.depth.sgbm.num_disparities
    assert not result.valid[:, : border - 1].any()


def test_confidence_is_present_and_bounded(narrow_cfg: Config) -> None:
    scene = planar_scene(disparity_px=24.0, width=TEST_WIDTH, height=TEST_HEIGHT)
    result = SgbmCpuBackend(narrow_cfg, q=scene.q_matrix()).infer(_bundle(scene))
    assert result.conf is not None
    assert result.conf.shape == result.depth_m.shape
    assert np.all((result.conf >= 0.0) & (result.conf <= 1.0))
    assert result.conf[result.valid].min() > 0.0
    assert result.conf[~result.valid].max(initial=0.0) == 0.0


def test_nearer_surfaces_get_higher_confidence(narrow_cfg: Config) -> None:
    near = planar_scene(disparity_px=48.0, width=TEST_WIDTH, height=TEST_HEIGHT, seed=5)
    far = planar_scene(disparity_px=12.0, width=TEST_WIDTH, height=TEST_HEIGHT, seed=5)
    backend_near = SgbmCpuBackend(narrow_cfg, q=near.q_matrix())
    backend_far = SgbmCpuBackend(narrow_cfg, q=far.q_matrix())
    conf_near = backend_near.infer(_bundle(near)).conf
    conf_far = backend_far.infer(_bundle(far)).conf
    assert conf_near is not None and conf_far is not None
    assert np.median(conf_near[conf_near > 0]) > np.median(conf_far[conf_far > 0])


# --------------------------------------------------------------------------
# Left/right consistency, the thing that keeps occlusions out of the map
# --------------------------------------------------------------------------


def test_lr_check_reduces_occlusion_leak(cfg_from) -> None:
    scene = occlusion_scene(width=TEST_WIDTH, height=TEST_HEIGHT)
    occluded = scene.occlusion_mask
    assert occluded.sum() > 0

    with_lr = cfg_from(
        {"depth": {"sgbm": {"num_disparities": 64}, "postprocess": {"lr_consistency": True}}}
    )
    without_lr = cfg_from(
        {"depth": {"sgbm": {"num_disparities": 64}, "postprocess": {"lr_consistency": False}}}
    )
    leak_on = (
        SgbmCpuBackend(with_lr, q=scene.q_matrix()).infer(_bundle(scene)).valid & occluded
    ).sum()
    leak_off = (
        SgbmCpuBackend(without_lr, q=scene.q_matrix()).infer(_bundle(scene)).valid & occluded
    ).sum()
    assert leak_on < leak_off, "the L/R check must remove invented depth in occlusions"


def test_lr_check_does_not_blank_the_right_of_the_frame(cfg_from) -> None:
    # Regression guard. Treating "the right disparity map has no data here" as
    # a failed check used to blank the rightmost numDisparities columns on
    # every frame, which would make the right of the FOV permanently unknown,
    # and therefore permanently impassable.
    scene = planar_scene(disparity_px=24.0, width=TEST_WIDTH, height=TEST_HEIGHT)
    with_lr = cfg_from(
        {"depth": {"sgbm": {"num_disparities": 64}, "postprocess": {"lr_consistency": True}}}
    )
    without_lr = cfg_from(
        {"depth": {"sgbm": {"num_disparities": 64}, "postprocess": {"lr_consistency": False}}}
    )
    valid_on = SgbmCpuBackend(with_lr, q=scene.q_matrix()).infer(_bundle(scene)).valid
    valid_off = SgbmCpuBackend(without_lr, q=scene.q_matrix()).infer(_bundle(scene)).valid

    right_quarter = slice(3 * TEST_WIDTH // 4, TEST_WIDTH)
    assert valid_on[:, right_quarter].mean() > 0.9
    assert valid_on.mean() > valid_off.mean() - 0.05


def test_right_disparity_is_refused_when_disabled(cfg_from) -> None:
    cfg = cfg_from({"depth": {"postprocess": {"lr_consistency": False}}})
    scene = planar_scene(disparity_px=24.0, width=TEST_WIDTH, height=TEST_HEIGHT)
    backend = SgbmCpuBackend(cfg, q=scene.q_matrix())
    with pytest.raises(RuntimeError, match="disabled"):
        backend.compute_right_disparity(scene.left, scene.right)


# --------------------------------------------------------------------------
# Plumbing
# --------------------------------------------------------------------------


def test_focal_and_baseline_are_read_back_out_of_q(cfg: Config) -> None:
    scene = planar_scene(disparity_px=24.0, width=64, height=32)
    backend = SgbmCpuBackend(cfg, q=scene.q_matrix())
    assert backend.focal_px == pytest.approx(scene.focal_px)
    assert backend.baseline_m == pytest.approx(scene.baseline_m)


def test_q_defaults_to_the_configured_camera(cfg: Config) -> None:
    backend = SgbmCpuBackend(cfg)
    assert np.allclose(backend.q, cfg.camera.load_q())
    assert backend.focal_px == pytest.approx(cfg.camera.nominal_focal_px)
    assert backend.baseline_m == pytest.approx(cfg.camera.baseline_m)


def test_rejects_a_bad_q(cfg: Config) -> None:
    with pytest.raises(ValueError, match="4x4"):
        SgbmCpuBackend(cfg, q=np.eye(3))


def test_rejects_mismatched_halves(narrow_cfg: Config) -> None:
    backend = SgbmCpuBackend(narrow_cfg)
    bundle = FrameBundle(
        left=np.zeros((10, 20, 3), np.uint8),
        right=np.zeros((10, 18, 3), np.uint8),
        t_ns=0,
        seq=0,
    )
    with pytest.raises(ValueError, match="must match"):
        backend.infer(bundle)


def test_records_stage_metrics(narrow_cfg: Config) -> None:
    scene = planar_scene(disparity_px=24.0, width=TEST_WIDTH, height=TEST_HEIGHT)
    metrics = Metrics()
    SgbmCpuBackend(narrow_cfg, q=scene.q_matrix(), metrics=metrics).infer(_bundle(scene))
    names = metrics.stage_names()
    assert "depth.match" in names
    assert "depth.reproject" in names
    assert "depth.postprocess" in names
    assert "depth.match_right" in names  # LR is on by default


def test_works_without_metrics(narrow_cfg: Config) -> None:
    scene = planar_scene(disparity_px=24.0, width=TEST_WIDTH, height=TEST_HEIGHT)
    assert isinstance(
        SgbmCpuBackend(narrow_cfg, q=scene.q_matrix()).infer(_bundle(scene)), DepthResult
    )


def test_backend_is_reusable_across_frames(narrow_cfg: Config) -> None:
    scene = planar_scene(disparity_px=24.0, width=TEST_WIDTH, height=TEST_HEIGHT)
    backend = SgbmCpuBackend(narrow_cfg, q=scene.q_matrix())
    first = backend.infer(_bundle(scene, t_ns=1))
    second = backend.infer(_bundle(scene, t_ns=2))
    assert np.array_equal(first.valid, second.valid)
    assert np.allclose(first.depth_m, second.depth_m, equal_nan=True)


def test_to_gray_accepts_the_shapes_a_source_can_produce() -> None:
    colour = np.zeros((4, 6, 3), np.uint8)
    assert to_gray(colour).shape == (4, 6)
    assert to_gray(np.zeros((4, 6), np.uint8)).shape == (4, 6)
    assert to_gray(np.zeros((4, 6, 1), np.uint8)).shape == (4, 6)


def test_to_gray_rejects_wrong_types() -> None:
    with pytest.raises(ValueError, match="uint8"):
        to_gray(np.zeros((4, 6), np.float32))
    with pytest.raises(ValueError, match="shape"):
        to_gray(np.zeros((4, 6, 4), np.uint8))
