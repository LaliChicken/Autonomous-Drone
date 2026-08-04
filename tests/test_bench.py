"""Benchmark scoring, and a regression guard on the committed baseline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from config import Config
from sources.types import DepthResult, FrameBundle
from tools.bench import (
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    SceneMetrics,
    aggregate,
    baseline_path,
    compare_to_baseline,
    depth_to_disparity,
    format_table,
    load_baseline,
    main,
    run_bench,
    score_scene,
    write_baseline,
)
from tools.synthetic_stereo import StereoScene, occlusion_scene, planar_scene


class OracleBackend:
    """Returns the scene's own ground truth. The best score achievable."""

    def __init__(self, scene: StereoScene) -> None:
        self.scene = scene

    def infer(self, f: FrameBundle) -> DepthResult:
        depth = self.scene.depth_gt.copy()
        valid = np.isfinite(depth)
        depth[~valid] = np.nan
        return DepthResult(depth_m=depth, valid=valid, conf=None, t_ns=f.t_ns)


class BiasedBackend(OracleBackend):
    """Ground truth plus a fixed depth offset."""

    def __init__(self, scene: StereoScene, offset_m: float) -> None:
        super().__init__(scene)
        self.offset_m = offset_m

    def infer(self, f: FrameBundle) -> DepthResult:
        result = super().infer(f)
        result.depth_m = result.depth_m + np.float32(self.offset_m)
        return result


def test_depth_to_disparity_inverts_the_stereo_equation() -> None:
    depth = np.array([[1.0, 2.0]], dtype=np.float32)
    disparity = depth_to_disparity(depth, 400.0, 0.05)
    assert disparity[0, 0] == pytest.approx(20.0)
    assert disparity[0, 1] == pytest.approx(10.0)


def test_depth_to_disparity_keeps_nan_as_nan() -> None:
    # A NaN depth must not become a huge disparity that then scores as a
    # gigantic error; it must stay out of the scoring set entirely.
    disparity = depth_to_disparity(np.array([[np.nan, 0.0]], np.float32), 400.0, 0.05)
    assert np.isnan(disparity).all()


def test_a_perfect_backend_scores_perfectly(cfg: Config) -> None:
    scene = planar_scene(disparity_px=20.0, width=200, height=60)
    metrics = score_scene(scene, cfg, backend=OracleBackend(scene))
    assert metrics.bad_pixel_pct == 0.0
    assert metrics.disparity_rmse_px == pytest.approx(0.0, abs=1e-4)
    assert metrics.depth_rmse_m == pytest.approx(0.0, abs=1e-6)
    assert metrics.completeness_pct == pytest.approx(100.0)
    assert metrics.occlusion_leak_pct == 0.0


def test_a_biased_backend_is_penalised(cfg: Config) -> None:
    scene = planar_scene(disparity_px=20.0, width=200, height=60)
    metrics = score_scene(scene, cfg, backend=BiasedBackend(scene, offset_m=0.5))
    assert metrics.depth_rmse_m == pytest.approx(0.5, rel=1e-3)
    assert metrics.bad_pixel_pct > 0.0


def test_scoring_ignores_pixels_without_ground_truth(cfg: Config) -> None:
    # A backend must not be punished for correctly declining to guess, nor
    # credited for guessing where there is no answer.
    scene = planar_scene(disparity_px=20.0, width=200, height=60)
    metrics = score_scene(scene, cfg, backend=OracleBackend(scene))
    assert metrics.scored_px == metrics.gt_px
    assert metrics.gt_px < scene.disparity_gt.size  # the border has no GT


def test_occlusion_leak_counts_invented_depth(cfg: Config) -> None:
    scene = occlusion_scene(width=300, height=100)

    class LeakyBackend:
        def infer(self, f: FrameBundle) -> DepthResult:
            depth = np.nan_to_num(scene.depth_gt, nan=1.0).astype(np.float32)
            return DepthResult(
                depth_m=depth, valid=np.ones(depth.shape, bool), conf=None, t_ns=f.t_ns
            )

    metrics = score_scene(scene, cfg, backend=LeakyBackend())
    assert metrics.occlusion_leak_pct == pytest.approx(100.0)


def test_aggregate_is_an_unweighted_mean() -> None:
    def make(name: str, bad: float) -> SceneMetrics:
        return SceneMetrics(
            scene=name,
            scored_px=1,
            gt_px=1,
            bad_pixel_pct=bad,
            disparity_rmse_px=bad,
            depth_rmse_m=bad,
            depth_mae_m=bad,
            coverage_pct=bad,
            completeness_pct=bad,
            occlusion_leak_pct=bad,
        )

    assert aggregate([make("a", 1.0), make("b", 3.0)])["bad_pixel_pct"] == pytest.approx(2.0)
    assert aggregate([]) == {}


def test_aggregate_skips_non_finite_scenes() -> None:
    def make(bad: float) -> SceneMetrics:
        return SceneMetrics("s", 0, 0, bad, bad, bad, bad, 0.0, 0.0, 0.0)

    assert aggregate([make(float("nan")), make(4.0)])["bad_pixel_pct"] == pytest.approx(4.0)


# --------------------------------------------------------------------------
# Regression detection
# --------------------------------------------------------------------------


def _report(**scene_values: float) -> dict:
    base = {
        "bad_pixel_pct": 1.0,
        "disparity_rmse_px": 1.0,
        "depth_rmse_m": 1.0,
        "depth_mae_m": 1.0,
        "coverage_pct": 80.0,
        "completeness_pct": 90.0,
        "occlusion_leak_pct": 5.0,
    }
    base.update(scene_values)
    return {"scenes": {"s": base}}


def test_no_regression_when_nothing_changes() -> None:
    assert compare_to_baseline(_report(), _report(), 0.1) == []


def test_error_getting_worse_is_a_regression() -> None:
    problems = compare_to_baseline(_report(depth_rmse_m=2.0), _report(), 0.1)
    assert any("depth_rmse_m" in p for p in problems)


def test_error_getting_better_is_not_a_regression() -> None:
    assert compare_to_baseline(_report(depth_rmse_m=0.1), _report(), 0.1) == []


def test_coverage_dropping_is_a_regression() -> None:
    problems = compare_to_baseline(_report(coverage_pct=50.0), _report(), 0.1)
    assert any("coverage_pct" in p for p in problems)


def test_coverage_rising_is_not_a_regression() -> None:
    assert compare_to_baseline(_report(coverage_pct=95.0), _report(), 0.1) == []


def test_occlusion_leak_is_scored_as_lower_is_better() -> None:
    assert "occlusion_leak_pct" in LOWER_IS_BETTER
    assert "coverage_pct" in HIGHER_IS_BETTER
    problems = compare_to_baseline(_report(occlusion_leak_pct=20.0), _report(), 0.1)
    assert any("occlusion_leak_pct" in p for p in problems)


def test_change_within_tolerance_is_accepted() -> None:
    assert compare_to_baseline(_report(depth_rmse_m=1.05), _report(), 0.1) == []


def test_a_missing_scene_is_reported() -> None:
    problems = compare_to_baseline({"scenes": {}}, _report(), 0.1)
    assert any("missing" in p for p in problems)


def test_near_zero_metrics_do_not_trip_on_float_noise() -> None:
    baseline = _report(bad_pixel_pct=0.0)
    current = _report(bad_pixel_pct=1e-9)
    assert compare_to_baseline(current, baseline, 0.1) == []


# --------------------------------------------------------------------------
# The committed baseline
# --------------------------------------------------------------------------


def test_committed_baseline_exists(cfg: Config) -> None:
    assert baseline_path(cfg).is_file(), "tools/depth_baseline.json must be committed"


def test_current_backend_matches_the_committed_baseline(cfg: Config) -> None:
    """The point of committing a baseline: this fails when depth regresses."""
    report = run_bench(cfg)
    problems = compare_to_baseline(report, load_baseline(cfg), cfg.bench.regression_tolerance)
    assert problems == [], "\n".join(problems)


def test_baseline_covers_every_default_scene(cfg: Config) -> None:
    baseline = load_baseline(cfg)
    assert set(baseline["scenes"]) == {"planar_far", "planar_near", "slanted", "occlusion"}
    assert baseline["opencv_version"]
    assert baseline["depth_config"]["sgbm"]["num_disparities"] == cfg.depth.sgbm.num_disparities


def test_bench_is_deterministic(cfg: Config) -> None:
    first = run_bench(cfg)
    second = run_bench(cfg)
    assert first["scenes"] == second["scenes"]


def test_write_baseline_round_trips(cfg_from, tmp_path: Path) -> None:
    cfg = cfg_from({"bench": {"baseline_path": str(tmp_path / "b.json")}})
    scene_report = run_bench(cfg, scenes=[planar_scene(disparity_px=20.0, width=200, height=60)])
    path = write_baseline(cfg, scene_report)
    assert json.loads(path.read_text())["scenes"] == scene_report["scenes"]


def test_load_baseline_explains_a_missing_file(cfg_from, tmp_path: Path) -> None:
    cfg = cfg_from({"bench": {"baseline_path": str(tmp_path / "absent.json")}})
    with pytest.raises(FileNotFoundError, match="--write-baseline"):
        load_baseline(cfg)


def test_format_table_lists_every_scene(cfg: Config) -> None:
    report = run_bench(cfg, scenes=[planar_scene(disparity_px=20.0, width=200, height=60)])
    table = format_table(report)
    assert "planar_d20" in table
    assert "MEAN" in table


def test_cli_check_passes_against_the_committed_baseline() -> None:
    assert main(["--check"]) == 0


def test_cli_check_fails_on_a_regressed_baseline(cfg_from, tmp_path: Path, monkeypatch) -> None:
    # Write a baseline that claims a much better score than reality, then
    # confirm --check actually fails rather than reporting success.
    config_path = tmp_path / "config.yaml"
    import yaml

    from config.schema import DEFAULT_CONFIG_PATH

    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["bench"]["baseline_path"] = str(tmp_path / "strict.json")
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    cfg = cfg_from({"bench": {"baseline_path": str(tmp_path / "strict.json")}})
    report = run_bench(cfg)
    for scene in report["scenes"].values():
        scene["depth_rmse_m"] = 0.0
        scene["coverage_pct"] = 100.0
    write_baseline(cfg, report)

    assert main(["--check", "--config", str(config_path)]) == 1
