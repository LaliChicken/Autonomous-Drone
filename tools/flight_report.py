"""Summarize a recorded run without implying physical flight validation."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from infra.flightlog import FlightLogReader


def report(run_dir: Path) -> dict[str, Any]:
    reader = FlightLogReader(run_dir)
    reasons: Counter[str] = Counter()
    unknown = bins = ticks = 0
    for tick in reader.planner_ticks():
        ticks += 1
        reasons[tick['command']['reason'].split(':', 1)[0]] += 1
        snapshot = tick.get('snapshot')
        if snapshot is not None:
            unknown += sum(snapshot['unknown'])
            bins += len(snapshot['unknown'])
    runtime_path = run_dir / 'runtime.json'
    return dict(run_id=reader.run_id, git=reader.meta.get('git'),
                runtime=json.loads(runtime_path.read_text()) if runtime_path.exists() else None,
                planner_ticks=ticks, decision_counts=dict(reasons),
                unknown_bin_fraction=None if bins == 0 else unknown / bins,
                metrics=reader.metrics(), calibration_sha256=reader.meta.get('calibration_sha256'))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(report(args.run_dir), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
