# Autonomous drone — companion-computer autonomy stack

Perception → planning → offboard-control pipeline for a 7" quadcopter. ArduPilot
(MicoAir743) flies the aircraft; this runs on a Jetson Orin Nano and talks to it
over MAVLink.

**There is no hardware attached.** Everything here is developed against datasets,
synthetic data, and ArduPilot SITL. Nothing assumes a camera or flight controller
is present.

## Layout

| Path | What it does |
| --- | --- |
| `config/` | Every tunable, as YAML + validated frozen dataclasses |
| `sources/types.py` | **Frozen contract.** Do not modify |
| `sources/mavlink_client.py` | Threaded MAVLink reader, state, setpoint sender |
| `depth/sgbm_cpu.py` | StereoSGBM `DepthBackend` |
| `depth/postprocess.py` | Speckles, L/R consistency, range clamp, confidence |
| `perception/obstacles.py` | Depth → body-frame azimuth scan |
| `perception/red_box.py` | HSV target detection with two range estimates |
| `world/occupancy.py` | Temporal polar map with age decay |
| `planner/local_planner.py` | VFH-lite heading selection |
| `planner/behaviours.py` | IDLE → SEARCH → APPROACH → ARRIVE → RTL |
| `control/offboard.py` | 20 Hz slew-limited setpoint stream |
| `infra/` | Metrics, flight logging, replay |
| `sim/scenarios.py` | Synthetic worlds for driving the planner |
| `tools/` | Dataset fetch, synthetic stereo, depth benchmark |

Stubs owned elsewhere and deliberately left empty: `depth/backends.py`,
`depth/holepunch.py`, `perception/ground_mask.py`, `planner/governor.py`,
`planner/supervisor.py`, `control/gate.py`, `control/watchdog.py`,
`calib/validate.py`.

## Setup

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python numpy opencv-python pymavlink pyyaml pytest ruff
```

## Checks

```bash
.venv/bin/ruff check .
.venv/bin/python -m pytest -q
```

Nothing is skipped except `@pytest.mark.sitl` and `@pytest.mark.hw`, both opt-in:

```bash
DRONE_SITL=1 .venv/bin/python -m pytest -m sitl
DRONE_HW=1   .venv/bin/python -m pytest -m hw
```

## Depth benchmark

Scored against ground truth, with a committed baseline so a regression fails
loudly:

```bash
.venv/bin/python -m tools.bench                  # print the table
.venv/bin/python -m tools.bench --check          # fail on regression
.venv/bin/python -m tools.bench --write-baseline # after an intended change
```

The baseline is measured on the synthetic scenes in `tools/synthetic_stereo.py`,
so it reproduces with no network access. Real Middlebury scenes are optional:

```bash
.venv/bin/python -m tools.get_middlebury --all   # into data/ (gitignored)
.venv/bin/python -m tools.bench --middlebury
```

## Running ArduPilot SITL

The SITL scenario tests need a Copter SITL instance on `udp:127.0.0.1:14550`.
SITL is not vendored here — you need an ArduPilot checkout once:

```bash
git clone --recurse-submodules https://github.com/ArduPilot/ardupilot
cd ardupilot
./Tools/environment_install/install-prereqs-ubuntu.sh -y   # one time, follow its prompts
. ~/.profile
export ARDUPILOT_HOME=$PWD
```

Then, from this repo:

```bash
./scripts/run_sitl.sh              # starts Copter SITL, streams to udp:127.0.0.1:14550
./scripts/run_sitl.sh --speedup 5  # faster than real time
```

`scripts/run_sitl.sh` finds `sim_vehicle.py` on `PATH` or under `$ARDUPILOT_HOME`,
and tells you what to do if it finds neither. Override the endpoint with
`DRONE_SITL_ENDPOINT`; it must match `mavlink.endpoint` in `config/config.yaml`.

With SITL up, in a second terminal:

```bash
DRONE_SITL=1 .venv/bin/python -m pytest -m sitl -v
```

In the SITL console (MAVProxy), get the aircraft airborne before running the
flight scenarios:

```
mode guided
arm throttle
takeoff 5
```

The scenarios inject synthetic occupancy snapshots and target detections into the
planner while the real ArduPilot flies, so what is under test is the decision
chain and the MAVLink plumbing — not a simulated camera.

## Conventions

These are enforced by tests, not just documented:

- **Timestamps** are integer nanoseconds from `CLOCK_MONOTONIC`. Wall clock appears
  only in flight-log directory names, never in the pipeline.
- **Frames**: camera is OpenCV (x right, y down, z forward); body is FRD (x forward,
  y right, z down). Bearings are right-positive, elevation up-positive.
- **Angles** are radians everywhere past `config/`. YAML is degrees.
- **Invalid depth** is `valid=False` with `depth_m = NaN` — never `0.0`, which would
  read as an obstacle at the lens.
- **Unknown ≠ clear.** Missing information stays unknown and is impassable. A bin
  that ages out becomes unknown, not clear.

## Open questions

Marked `# QUESTION(rahul):` in the source:

- `config/schema.py` — no stereo calibration exists, so depth falls back to a nominal
  Q. Should `sgbm_cpu` refuse to run without one?
- `perception/obstacles.py` — SGBM cannot match the leftmost `numDisparities`
  columns, so ~6.5° of the 65° FOV is permanently unknown, and therefore permanently
  impassable. This biases the planner against left turns.
- `infra/replay.py` — replay owns the MJPG decode/split rule because
  `sources/stereo_uvc.py` is still empty.
- `tools/get_middlebury.py` — the download path has never been executed; this
  environment cannot reach `vision.middlebury.edu`.
