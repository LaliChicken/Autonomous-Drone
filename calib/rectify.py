"""Load a candidate calibration and rectify raw eyes without changing frame contracts."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from config import Config, ConfigError


class Rectifier:
    def __init__(self, path: str | Path, cfg: Config) -> None:
        self.size = (cfg.camera.eye_width, cfg.camera.eye_height)
        with np.load(path, allow_pickle=False) as data:
            required = {'image_size', 'schema_version', 'Q', 'R1', 'left_x', 'left_y',
                        'right_x', 'right_y', 'roi1', 'roi2', 'P1', 'P2'}
            if missing := required - set(data.files):
                raise ConfigError(f"calibration missing rectification arrays: {sorted(missing)}")
            if int(data['schema_version']) != 1:
                raise ConfigError("unsupported calibration schema")
            if tuple(data['image_size']) != self.size:
                raise ConfigError("calibration resolution does not match camera config")
            self.q = np.array(data['Q'], dtype=np.float64)
            self.r1 = np.array(data['R1'], dtype=np.float64)
            self.maps = [np.array(data[name], dtype=np.float32) for name in
                         ('left_x', 'left_y', 'right_x', 'right_y')]
            if self.q.shape != (4, 4) or not np.isfinite(self.q).all() or self.q[3, 2] <= 0:
                raise ConfigError("invalid positive-disparity Q")
            if (self.r1.shape != (3, 3) or not np.isfinite(self.r1).all()
                    or not np.allclose(self.r1.T @ self.r1, np.eye(3))
                    or not np.isclose(np.linalg.det(self.r1), 1)):
                raise ConfigError("invalid left rectification rotation")
            if any(m.shape != self.size[::-1] or not np.isfinite(m).all() for m in self.maps):
                raise ConfigError("invalid rectification maps")
            # Only supported horizontal, zero-disparity geometry is accepted.
            p1, p2 = np.asarray(data['P1']), np.asarray(data['P2'])
            if (p1.shape != (3, 4) or p2.shape != (3, 4)
                    or not np.isfinite(p1).all() or not np.isfinite(p2).all()
                    or p2[0, 3] >= 0 or not np.isclose(p2[1, 3], 0)
                    or not np.isclose(p1[0, 2], p2[0, 2])):
                raise ConfigError("unsupported rectified projection geometry")
            self.valid_roi = cv2.getValidDisparityROI(
                tuple(map(int, data['roi1'])), tuple(map(int, data['roi2'])),
                cfg.depth.sgbm.min_disparity, cfg.depth.sgbm.num_disparities,
                cfg.depth.sgbm.block_size,
            )
        self.rotation_body_from_rectified = cfg.mount.rotation_body_from_cam() @ self.r1.T
        self.valid = np.zeros(self.size[::-1], dtype=bool)
        x, y, w, h = self.valid_roi
        self.valid[y:y+h, x:x+w] = True

    def apply(self, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if left.shape != (*self.size[::-1], 3) or right.shape != left.shape:
            raise ConfigError("raw stereo image shape does not match calibration")
        a = cv2.remap(left, *self.maps[:2], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        b = cv2.remap(right, *self.maps[2:], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return a, b
