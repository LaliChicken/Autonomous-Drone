"""Fetch a few Middlebury 2014 stereo pairs with ground truth.

    python -m tools.get_middlebury --list
    python -m tools.get_middlebury --all
    python -m tools.get_middlebury --scene Motorcycle

Downloads into ``bench.middlebury_dir`` (``data/middlebury``), which is
gitignored: the script is committed, the data is not. Nothing in the test
suite requires this data -- the committed baseline is measured on the
synthetic scenes in ``tools/synthetic_stereo.py`` precisely so that it can be
reproduced with no network at all. Real scenes are a richer check to run by
hand, not a dependency.

Each scene directory ends up as::

    data/middlebury/<Scene>/im0.png      left
                            im1.png      right
                            disp0.pfm    left disparity ground truth
                            calib.txt    intrinsics, baseline, ndisp

# QUESTION(rahul): these URLs follow the documented layout of the Middlebury
# 2014 "perfect" set, but this environment cannot reach vision.middlebury.edu
# (its TLS chain does not verify here), so they are UNVERIFIED -- the download
# path has never been executed end to end. Please run
# `python -m tools.get_middlebury --all` once from a machine with plain
# internet access and tell me what breaks. Everything downstream of the
# download (PFM parsing, calib parsing, scene construction) is covered by
# tests against locally written fixtures.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from config import load_config
from tools.synthetic_stereo import StereoScene

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_URL = "https://vision.middlebury.edu/stereo/data/scenes2014/datasets"
FILES = ("im0.png", "im1.png", "disp0.pfm", "calib.txt")

# Four scenes with a range of difficulty: textureless regions, thin structure,
# and large disparity ranges. Names are the 2014 dataset's own.
SCENES = ("Motorcycle", "Piano", "Playtable", "Vintage")


class MiddleburyError(RuntimeError):
    """Raised when a scene cannot be fetched or parsed."""


@dataclass(frozen=True)
class Calibration:
    focal_px: float
    baseline_m: float
    doffs: float
    width: int
    height: int
    ndisp: int


def read_pfm(path: Path) -> np.ndarray:
    """Read a PFM disparity map as float32 with NaN for 'no ground truth'.

    Middlebury stores unknown disparities as ``inf``; those become NaN here so
    that a single ``isfinite`` test covers every no-data case in the codebase.
    """
    with path.open("rb") as handle:
        header = handle.readline().rstrip()
        if header not in (b"Pf", b"PF"):
            raise MiddleburyError(f"{path}: not a PFM file (header {header!r})")
        channels = 3 if header == b"PF" else 1

        line = handle.readline()
        while line.startswith(b"#"):
            line = handle.readline()
        match = re.match(rb"^\s*(\d+)\s+(\d+)\s*$", line)
        if not match:
            raise MiddleburyError(f"{path}: bad PFM dimensions line {line!r}")
        width, height = int(match.group(1)), int(match.group(2))

        scale = float(handle.readline().rstrip())
        dtype = "<f4" if scale < 0 else ">f4"
        data = np.frombuffer(handle.read(width * height * channels * 4), dtype=dtype)

    if data.size != width * height * channels:
        raise MiddleburyError(f"{path}: truncated PFM payload")
    image = data.reshape(height, width, channels) if channels > 1 else data.reshape(height, width)
    # PFM rows run bottom-to-top.
    image = np.flipud(image).astype(np.float32)
    image[~np.isfinite(image)] = np.nan
    return np.ascontiguousarray(image)


def parse_calib(path: Path) -> Calibration:
    """Parse Middlebury's calib.txt.

    ``cam0=[f 0 cx; 0 f cy; 0 0 1]``, ``baseline`` in millimetres, and
    ``doffs`` the x-difference between the two principal points, which has to
    be added back to the disparity before it means anything metric.
    """
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()

    try:
        cam0 = values["cam0"]
        focal = float(re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cam0)[0])
        baseline_mm = float(values["baseline"])
        return Calibration(
            focal_px=focal,
            baseline_m=baseline_mm / 1000.0,
            doffs=float(values.get("doffs", 0.0)),
            width=int(values["width"]),
            height=int(values["height"]),
            ndisp=int(values["ndisp"]),
        )
    except (KeyError, IndexError, ValueError) as exc:
        raise MiddleburyError(f"{path}: cannot parse calibration ({exc})") from exc


def scene_url(scene: str, filename: str) -> str:
    return f"{BASE_URL}/{scene}-perfect/{filename}"


def download_file(url: str, destination: Path, timeout: float = 120.0) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            with partial.open("wb") as handle:
                shutil.copyfileobj(response, handle)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        partial.unlink(missing_ok=True)
        raise MiddleburyError(f"failed to download {url}: {exc}") from exc
    # Rename only on success, so an interrupted run never leaves a file that
    # looks complete and silently poisons the benchmark.
    partial.replace(destination)


def fetch_scene(scene: str, root: Path, force: bool = False) -> Path:
    target = root / scene
    for filename in FILES:
        destination = target / filename
        if destination.is_file() and not force:
            continue
        download_file(scene_url(scene, filename), destination)
    return target


def load_scene(directory: Path) -> StereoScene:
    """Build a StereoScene from a downloaded Middlebury directory."""
    for filename in FILES:
        if not (directory / filename).is_file():
            raise MiddleburyError(f"{directory} is missing {filename}")

    left = cv2.imread(str(directory / "im0.png"), cv2.IMREAD_COLOR)
    right = cv2.imread(str(directory / "im1.png"), cv2.IMREAD_COLOR)
    if left is None or right is None:
        raise MiddleburyError(f"{directory}: could not decode im0.png / im1.png")

    calibration = parse_calib(directory / "calib.txt")
    disparity = read_pfm(directory / "disp0.pfm")
    # doffs is a constant principal-point offset baked into the stored
    # disparity; without adding it back, Z = f*B/d is wrong by a fixed factor.
    disparity = disparity + calibration.doffs
    disparity[disparity <= 0.0] = np.nan

    return StereoScene(
        name=directory.name,
        left=left,
        right=right,
        disparity_gt=disparity,
        focal_px=calibration.focal_px,
        baseline_m=calibration.baseline_m,
    )


def load_local_scenes(root: Path) -> list[StereoScene]:
    """Every scene already present under ``root``. Missing data is not an error."""
    if not root.is_dir():
        return []
    scenes = []
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        if all((directory / f).is_file() for f in FILES):
            scenes.append(load_scene(directory))
    return scenes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true", help="list scenes and their URLs")
    parser.add_argument("--all", action="store_true", help="fetch every scene")
    parser.add_argument("--scene", action="append", default=[], help="fetch one scene by name")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    root = Path(cfg.bench.middlebury_dir)
    root = root if root.is_absolute() else REPO_ROOT / root

    if args.list:
        for scene in SCENES:
            print(scene)
            for filename in FILES:
                print(f"    {scene_url(scene, filename)}")
        return 0

    wanted = list(SCENES) if args.all else list(args.scene)
    if not wanted:
        parser.error("nothing to do: pass --all, --scene NAME, or --list")

    unknown = [s for s in wanted if s not in SCENES]
    if unknown:
        parser.error(f"unknown scene(s): {', '.join(unknown)}; known: {', '.join(SCENES)}")

    failures = 0
    for scene in wanted:
        try:
            target = fetch_scene(scene, root, force=args.force)
            loaded = load_scene(target)
            finite = int(np.isfinite(loaded.disparity_gt).sum())
            print(
                f"{scene}: {loaded.left.shape[1]}x{loaded.left.shape[0]}, "
                f"{finite} ground-truth pixels -> {target}"
            )
        except MiddleburyError as exc:
            failures += 1
            print(f"{scene}: {exc}", file=sys.stderr)

    if failures:
        print(
            f"\n{failures} scene(s) failed. The benchmark does not need them: "
            "the committed baseline is measured on tools/synthetic_stereo.py.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
