# Autonomous drone — companion-computer autonomy stack

Perception → planning → offboard-control pipeline for a 7" quadcopter. ArduPilot
(MicoAir743) flies the aircraft; this runs on a Jetson Orin Nano and talks to it
over MAVLink.

**There is no hardware attached.** Everything here is developed against datasets,
synthetic data, and ArduPilot SITL. Nothing assumes a camera or flight controller
is present.

A CPU **record-only application** now connects capture, candidate rectification,
perception, planning and logs. It never transmits flight commands. Calibration
validation, GPU evaluation, safety-owner deliveries and hardware acceptance remain
open; see [completion status](docs/COMPLETION_STATUS.md).

## Layout

| Path | What it does |
| --- | --- |
| `config/` | Every tunable, as YAML + validated frozen dataclasses |
| `sources/types.py` | **Frozen contract.** Do not modify |
| `sources/stereo_uvc.py` | Bounded original-MJPG capture and shared decoding |
| `calib/` | Calibration recording, candidate solving and rectification |
| `infra/pipeline.py` | Shared record-only CPU pipeline for live and replay |
| `tools/run.py` | Device listing, recording, synthetic runs and replay CLI |
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
.venv/bin/ruff check calib config control depth infra perception planner sources sim tests tools world
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

## Record-only workflow

**Workstation-side:** exercise the complete CPU pipeline without hardware:

```bash
.venv/bin/python -m tools.run synthetic --frames 5
# Substitute the printed run directory:
.venv/bin/python -m tools.run replay runs/RUN_ID
.venv/bin/python -m tools.flight_report runs/RUN_ID
```

The default synthetic run has no FC telemetry, so its proposed commands stay IDLE.
The integration tests also exercise target acquisition, movement proposals and
telemetry loss. Replay compares proposed commands byte-for-byte with those logged.

**Jetson-side, in the proven Python/CUDA shell:** select the camera and record
calibration samples, without flight-controller access:

```bash
python3 -m tools.run devices
# Configure capture.device_path and the measured calibration target in deployment.yaml.
DRONE_HW=1 DRONE_CONFIG=deployment.yaml python3 -m pytest -q tests/test_stereo_uvc.py
python3 -m calib.capture --config deployment.yaml --pairs 30
python3 -m calib.solve runs/RUN_ID candidate.npz --config deployment.yaml
```

`calibration.columns` and `rows` are **inner checkerboard corners**; `square_m` must
be measured. Defaults are unset. Move the board through varied positions and tilts;
record extra pairs because the solver uses only pairs detected in both eyes.
The solver writes an **unvalidated candidate**, not flight approval.

Set `camera.calibration_npz` to that candidate's absolute path in deployment.yaml,
then run CPU perception with logging:

```bash
python3 -m tools.run record --config deployment.yaml --frames 200
# Optionally add --telemetry to receive the configured MAVLink streams.
```

Live perception refuses nominal geometry. Raw calibration capture does not require
calibration. Each run keeps the original processed-frame MJPG bytes, effective
telemetry inputs, proposed commands, metrics and a hashed calibration copy. Frame
queue drops and terminal faults appear in `runtime.json`; timestamps from the
OpenCV transport are **host-dequeue monotonic time**, not verified exposure time.

Recording stops at the configured duration or encoded-frame-byte limit. Restart
explicitly after correcting capture faults. Neither the CLI nor the optional
[NixOS deployment module](docs/DEPLOYMENT.md) offers command transmission.

## Remaining decisions

- Verify actual camera timestamps, raw MJPG format and eye ordering on the Jetson.
- Measure the calibration target; obtain held-out validation from the calibration owner.
- Recover the missing CUDA shell/comparison patches and review measured GPU results
  before selecting a runtime backend or changing resolution.
- Stereo's left search margin and rectification borders remain unknown and impassable.
  The runner clears body-relative occupancy history while moving because no
  ego-motion transform exists yet.
- Confirm actual rangefinder sentinel semantics and implement forward-range fusion.
- Integrate the owned safety modules and establish stopping/latency acceptance before
  any flight-command path. The complete remaining list is in the completion status.
- `tools/get_middlebury.py` still needs a verified real dataset download; its local
  fixture tests do not establish remote download availability.
