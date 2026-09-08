"""Candidate stereo calibration from paired checkerboard observations.

Produces an unvalidated NPZ; the owned calib/validate.py must supply real-camera
acceptance. OpenCV calibration/rectification APIs:
https://docs.opencv.org/4.11.0/d9/d0c/group__calib3d.html
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from config import Config, ConfigError, load_config
from config.schema import CalibrationConfig
from infra.flightlog import FlightLogReader
from infra.replay import decode_frame


def board_points(cfg: CalibrationConfig) -> np.ndarray:
    if cfg.columns is None or cfg.rows is None or cfg.square_m is None:
        raise ConfigError("set measured calibration.columns, rows and square_m first")
    points = np.zeros((cfg.columns * cfg.rows, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[:cfg.columns, :cfg.rows].T.reshape(-1, 2) * cfg.square_m
    return points


def find_corners(image: np.ndarray, cfg: CalibrationConfig) -> np.ndarray | None:
    board_points(cfg)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCornersSB(gray, (cfg.columns, cfg.rows))
    return np.asarray(corners, dtype=np.float32) if ok else None


def solve_points(
    left: list[np.ndarray], right: list[np.ndarray], size: tuple[int, int], cfg: Config,
) -> dict[str, np.ndarray]:
    """Solve matching corner observations. Does not assert flight accuracy."""
    cal = cfg.calibration
    objects = board_points(cal)
    if len(left) != len(right) or len(left) < cal.min_pairs:
        raise ConfigError(f"need at least {cal.min_pairs} matched calibration pairs")
    expected = (len(objects), 1, 2)
    if any(p.shape != expected or not np.isfinite(p).all() for p in [*left, *right]):
        raise ConfigError(f"corner arrays must be finite with shape {expected}")
    if size != (cfg.camera.eye_width, cfg.camera.eye_height):
        raise ConfigError("calibration image size does not match camera config")
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                cal.max_iterations, cal.epsilon)
    obj = [objects] * len(left)
    rms1, k1, d1, _, _ = cv2.calibrateCamera(obj, left, size, None, None, criteria=criteria)
    rms2, k2, d2, _, _ = cv2.calibrateCamera(obj, right, size, None, None, criteria=criteria)
    rms, k1, d1, k2, d2, r, t, _, _ = cv2.stereoCalibrate(
        obj, left, right, k1, d1, k2, d2, size,
        criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC,
    )
    r1, r2, p1, p2, q, roi1, roi2 = cv2.stereoRectify(
        k1, d1, k2, d2, size, r, t, flags=cv2.CALIB_ZERO_DISPARITY, alpha=cal.alpha,
    )
    if q[3, 2] <= 0 or abs(p2[1, 3]) > abs(p2[0, 3]):
        raise ConfigError("expected a horizontal stereo pair with positive left disparity")
    maps1 = cv2.initUndistortRectifyMap(k1, d1, r1, p1, size, cv2.CV_32FC1)
    maps2 = cv2.initUndistortRectifyMap(k2, d2, r2, p2, size, cv2.CV_32FC1)
    result = dict(K1=k1, D1=d1, K2=k2, D2=d2, R=r, T=t, R1=r1, R2=r2,
                  P1=p1, P2=p2, Q=q, left_x=maps1[0], left_y=maps1[1],
                  right_x=maps2[0], right_y=maps2[1], image_size=np.array(size),
                  roi1=np.array(roi1), roi2=np.array(roi2),
                  fit_rms_px=np.array([rms1, rms2, rms]),
                  schema_version=np.array(1), validated=np.array(False))
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ConfigError("calibration produced non-finite parameters")
    return result


def solve_run(run_dir: Path, output: Path, cfg: Config) -> dict[str, object]:
    reader = FlightLogReader(run_dir)
    left, right, sequences = [], [], []
    digest = hashlib.sha256()
    size = (cfg.camera.eye_width, cfg.camera.eye_height)
    for record in reader.frames():
        frame = decode_frame(record)
        if frame.left.shape[:2] != size[::-1]:
            raise ConfigError("recorded image dimensions differ from camera config")
        a = find_corners(frame.left, cfg.calibration)
        b = find_corners(frame.right, cfg.calibration)
        if a is not None and b is not None:
            left.append(a)
            right.append(b)
            sequences.append(frame.seq)
            digest.update(record.mjpg)
    result = solve_points(left, right, size, cfg)
    metadata = dict(source_run=str(run_dir), sequences=sequences,
                    input_sha256=digest.hexdigest(), opencv=cv2.__version__,
                    config=cfg.to_dict(), validation="pending_owner_validation")
    result['metadata_json'] = np.array(json.dumps(metadata, sort_keys=True))
    # Exclusive creation prevents accidental replacement of a measured calibration.
    with output.open('xb') as handle:
        np.savez_compressed(handle, **result)
    return dict(output=str(output), pairs=len(left), fit_rms_px=result['fit_rms_px'].tolist(),
                validation="pending_owner_validation")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--config', type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(solve_run(args.run_dir, args.output, load_config(args.config)), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
