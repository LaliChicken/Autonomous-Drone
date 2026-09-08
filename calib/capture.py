"""Record original paired MJPG samples for offline calibration solving."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from config import load_config
from infra.flightlog import FlightLog
from sources.stereo_uvc import StereoCapture


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--pairs', type=int, help='defaults to calibration.min_pairs')
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    count = args.pairs if args.pairs is not None else cfg.calibration.min_pairs
    if count <= 0:
        parser.error('--pairs must be positive')
    if not cfg.flightlog.write_frames:
        parser.error('calibration capture requires flightlog.write_frames=true')
    with FlightLog.create(cfg, root=args.output_root) as log, StereoCapture(cfg) as source:
        print(log.run_dir, flush=True)
        last = None
        captured = 0
        while captured < count:
            frame = source.read()
            elapsed = None if last is None else (frame.bundle.t_ns - last) / 1e9
            if elapsed is None or elapsed >= cfg.calibration.sample_interval_s:
                log.write_frame(frame.bundle.seq, frame.bundle.t_ns, frame.mjpg)
                captured += 1
                last = frame.bundle.t_ns
                print(f'pair {captured}/{count}; move target to a different pose', flush=True)
            # Capture worker continues replacing its slot during this sampling pause.
            time.sleep(cfg.calibration.sample_interval_s)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
