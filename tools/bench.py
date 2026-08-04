"""Score the depth backend against ground truth and guard the result.

    python -m tools.bench                    # print the table
    python -m tools.bench --write-baseline   # regenerate tools/depth_baseline.json
    python -m tools.bench --check            # fail if anything regressed
    python -m tools.bench --middlebury       # include downloaded real scenes

Metrics per scene:

``bad_pixel_pct``      fraction of scored pixels off by more than
                       ``bench.bad_pixel_threshold_px`` disparity pixels. The
                       headline stereo number, and the Middlebury convention.
``disparity_rmse_px``  RMSE in disparity, which is what the matcher controls.
``depth_rmse_m``       RMSE in metres, which is what the planner consumes.
                       Reported separately because the map between them is
                       Z^2/(fB) -- the same disparity error is millimetres up
                       close and metres far away.
``coverage_pct``       valid pixels over all pixels.
``completeness_pct``   valid pixels over pixels that *have* ground truth. The
                       honest denominator: no backend can fill the left
                       ``numDisparities`` border, so coverage alone
                       understates a matcher on a narrow image.
``occlusion_leak_pct`` valid pixels inside a known occlusion. These are
                       confident depths for surfaces the right camera never
                       saw, i.e. invented obstacles, and are the one metric
                       here where a low score is a safety property rather than
                       an accuracy one.

Only pixels that have ground truth *and* a valid estimate are scored for
accuracy. Scoring an invalid pixel against ground truth would punish the
backend for correctly declining to guess, which is the behaviour we most want
to keep.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config import Config, load_config
from depth.sgbm_cpu import SgbmCpuBackend
from sources.types import FrameBundle
from tools.synthetic_stereo import StereoScene, default_scenes

REPO_ROOT = Path(__file__).resolve().parent.parent

# Metrics where a larger number is worse.
LOWER_IS_BETTER = frozenset(
    {"bad_pixel_pct", "disparity_rmse_px", "depth_rmse_m", "depth_mae_m", "occlusion_leak_pct"}
)
# Metrics where a smaller number is worse.
HIGHER_IS_BETTER = frozenset({"coverage_pct", "completeness_pct"})


@dataclass(frozen=True)
class SceneMetrics:
    scene: str
    scored_px: int
    gt_px: int
    bad_pixel_pct: float
    disparity_rmse_px: float
    depth_rmse_m: float
    depth_mae_m: float
    coverage_pct: float
    completeness_pct: float
    occlusion_leak_pct: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def depth_to_disparity(depth_m: np.ndarray, focal_px: float, baseline_m: float) -> np.ndarray:
    """d = f*B/Z. NaN depth stays NaN rather than becoming a huge disparity."""
    with np.errstate(divide="ignore", invalid="ignore"):
        disparity = (focal_px * baseline_m) / depth_m
    disparity = np.asarray(disparity, dtype=np.float32)
    disparity[~np.isfinite(disparity)] = np.nan
    return disparity


def score_scene(
    scene: StereoScene, cfg: Config, backend: SgbmCpuBackend | None = None
) -> SceneMetrics:
    """Run the backend over one scene and score it."""
    backend = backend or SgbmCpuBackend(cfg, q=scene.q_matrix())
    result = backend.infer(FrameBundle(left=scene.left, right=scene.right, t_ns=0, seq=0))

    gt_disparity = scene.disparity_gt
    gt_depth = scene.depth_gt
    gt_mask = scene.gt_mask
    valid = result.valid

    scored = valid & gt_mask
    n_scored = int(scored.sum())
    n_gt = int(gt_mask.sum())

    if n_scored:
        estimated = depth_to_disparity(result.depth_m, scene.focal_px, scene.baseline_m)
        disparity_error = np.abs(estimated[scored] - gt_disparity[scored])
        depth_error = result.depth_m[scored] - gt_depth[scored]
        bad = float(np.mean(disparity_error > cfg.bench.bad_pixel_threshold_px) * 100.0)
        disparity_rmse = float(np.sqrt(np.mean(disparity_error**2)))
        depth_rmse = float(np.sqrt(np.mean(depth_error**2)))
        depth_mae = float(np.mean(np.abs(depth_error)))
    else:
        bad = disparity_rmse = depth_rmse = depth_mae = float("nan")

    occlusion = scene.occlusion_mask
    leak = float((valid & occlusion).sum() / occlusion.sum() * 100.0) if occlusion.any() else 0.0

    return SceneMetrics(
        scene=scene.name,
        scored_px=n_scored,
        gt_px=n_gt,
        bad_pixel_pct=bad,
        disparity_rmse_px=disparity_rmse,
        depth_rmse_m=depth_rmse,
        depth_mae_m=depth_mae,
        coverage_pct=float(valid.mean() * 100.0),
        completeness_pct=float(n_scored / n_gt * 100.0) if n_gt else 0.0,
        occlusion_leak_pct=leak,
    )


def aggregate(metrics: list[SceneMetrics]) -> dict[str, float]:
    """Unweighted mean over scenes, so one big scene cannot dominate."""
    if not metrics:
        return {}
    keys = LOWER_IS_BETTER | HIGHER_IS_BETTER
    out: dict[str, float] = {}
    for key in sorted(keys):
        values = [getattr(m, key) for m in metrics]
        finite = [v for v in values if np.isfinite(v)]
        out[key] = float(np.mean(finite)) if finite else float("nan")
    return out


def run_bench(cfg: Config, scenes: list[StereoScene] | None = None) -> dict[str, Any]:
    scenes = scenes if scenes is not None else default_scenes()
    metrics = [score_scene(scene, cfg) for scene in scenes]
    return {
        "generated_by": "tools/bench.py",
        "scene_source": "tools.synthetic_stereo.default_scenes",
        "opencv_version": cv2.__version__,
        "bad_pixel_threshold_px": cfg.bench.bad_pixel_threshold_px,
        "depth_config": {
            "sgbm": asdict(cfg.depth.sgbm),
            "postprocess": asdict(cfg.depth.postprocess),
        },
        "scenes": {m.scene: m.to_dict() for m in metrics},
        "aggregate": aggregate(metrics),
    }


def format_table(report: dict[str, Any]) -> str:
    header = (
        f"{'scene':<14}{'bad%':>8}{'d_rmse':>9}{'z_rmse':>9}"
        f"{'cover%':>9}{'complete%':>11}{'occ_leak%':>11}"
    )
    lines = [header, "-" * len(header)]
    for name, m in report["scenes"].items():
        lines.append(
            f"{name:<14}{m['bad_pixel_pct']:>8.2f}{m['disparity_rmse_px']:>9.3f}"
            f"{m['depth_rmse_m']:>9.4f}{m['coverage_pct']:>9.2f}"
            f"{m['completeness_pct']:>11.2f}{m['occlusion_leak_pct']:>11.2f}"
        )
    agg = report["aggregate"]
    lines.append("-" * len(header))
    lines.append(
        f"{'MEAN':<14}{agg['bad_pixel_pct']:>8.2f}{agg['disparity_rmse_px']:>9.3f}"
        f"{agg['depth_rmse_m']:>9.4f}{agg['coverage_pct']:>9.2f}"
        f"{agg['completeness_pct']:>11.2f}{agg['occlusion_leak_pct']:>11.2f}"
    )
    return "\n".join(lines)


def baseline_path(cfg: Config) -> Path:
    path = Path(cfg.bench.baseline_path)
    return path if path.is_absolute() else REPO_ROOT / path


def write_baseline(cfg: Config, report: dict[str, Any]) -> Path:
    path = baseline_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def compare_to_baseline(
    report: dict[str, Any], baseline: dict[str, Any], tolerance: float
) -> list[str]:
    """Regressions, as human-readable lines. Empty means nothing got worse.

    Tolerance is a fraction of the baseline value, with a small absolute floor
    so a metric that is already near zero cannot produce a regression from
    floating-point noise.
    """
    problems: list[str] = []
    for name, base in baseline.get("scenes", {}).items():
        current = report.get("scenes", {}).get(name)
        if current is None:
            problems.append(f"{name}: missing from this run")
            continue
        for key in sorted(LOWER_IS_BETTER | HIGHER_IS_BETTER):
            base_value = base.get(key)
            new_value = current.get(key)
            if base_value is None or new_value is None:
                continue
            if not (np.isfinite(base_value) and np.isfinite(new_value)):
                continue
            slack = abs(base_value) * tolerance + 1e-6
            if key in LOWER_IS_BETTER and new_value > base_value + slack:
                problems.append(f"{name}.{key}: {base_value:.4f} -> {new_value:.4f} (worse)")
            elif key in HIGHER_IS_BETTER and new_value < base_value - slack:
                problems.append(f"{name}.{key}: {base_value:.4f} -> {new_value:.4f} (worse)")
    return problems


def load_baseline(cfg: Config) -> dict[str, Any]:
    path = baseline_path(cfg)
    if not path.is_file():
        raise FileNotFoundError(
            f"no baseline at {path}; run: python -m tools.bench --write-baseline"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _middlebury_scenes(cfg: Config) -> list[StereoScene]:
    from tools.get_middlebury import load_local_scenes

    root = Path(cfg.bench.middlebury_dir)
    root = root if root.is_absolute() else REPO_ROOT / root
    return load_local_scenes(root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write-baseline", action="store_true", help="regenerate the baseline")
    parser.add_argument("--check", action="store_true", help="fail on regression vs the baseline")
    parser.add_argument(
        "--middlebury",
        action="store_true",
        help="also score downloaded Middlebury scenes (excluded from the baseline)",
    )
    parser.add_argument("--config", default=None, help="path to config.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    report = run_bench(cfg)
    print(format_table(report))

    if args.middlebury:
        extra = _middlebury_scenes(cfg)
        if not extra:
            print(
                f"\nno Middlebury scenes under {cfg.bench.middlebury_dir}; "
                "run: python -m tools.get_middlebury --all",
                file=sys.stderr,
            )
        else:
            # Kept out of the baseline on purpose: the baseline must be
            # reproducible without a download.
            real = {"scenes": {m.scene: m.to_dict() for m in map(
                lambda s: score_scene(s, cfg), extra)}}
            real["aggregate"] = aggregate(
                [SceneMetrics(**m) for m in real["scenes"].values()]
            )
            print("\nMiddlebury (not part of the committed baseline):")
            print(format_table(real))

    if args.write_baseline:
        path = write_baseline(cfg, report)
        print(f"\nwrote baseline: {path}")
        return 0

    if args.check:
        baseline = load_baseline(cfg)
        if baseline.get("opencv_version") != cv2.__version__:
            print(
                f"\nnote: baseline was generated with OpenCV "
                f"{baseline.get('opencv_version')}, running {cv2.__version__}",
                file=sys.stderr,
            )
        problems = compare_to_baseline(report, baseline, cfg.bench.regression_tolerance)
        if problems:
            print("\nREGRESSIONS:", file=sys.stderr)
            for line in problems:
                print(f"  {line}", file=sys.stderr)
            return 1
        print("\nno regressions against the baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
