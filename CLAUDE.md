# CLAUDE.md — drone autonomy stack

Companion-computer stack for an autonomous 7" quadcopter. ArduPilot (MicoAir743 FC)
flies the aircraft; this repo is the perception → planning → offboard-control pipeline
that will run on a Jetson Orin Nano over MAVLink.

**There is no hardware attached.** Everything is developed against datasets, synthetic
data, and ArduPilot SITL. Never write code that assumes a camera or FC is present.

## Hard rules
- Python 3.11+, full type hints, ruff-clean. Deps: numpy, opencv-python, pymavlink,
  pyyaml, pytest only. Anything else needs a `# QUESTION(rahul):` justifying it.
- Every threshold/size/tunable lives in config/config.yaml, loaded into frozen
  dataclasses in config/. No magic numbers in module code.
- Timestamps are int nanoseconds from CLOCK_MONOTONIC. Never time.time() in-pipeline.
- Frames: camera = OpenCV convention (x right, y down, z forward); body = FRD
  (x fwd, y right, z down). Radians internally, everywhere.
- Invalid depth is valid=False, NEVER depth_m=0.0 (zero reads as "obstacle at the
  lens" downstream).
- unknown ≠ clear. Missing information propagates as unknown and is impassable.
  An occupancy bin that ages out becomes unknown, not clear.
- Every decision-making module logs inputs → decision with timestamps via infra/.
- Tests assert real behavior on real or synthetic data. Never mock the thing under
  test. Nothing skipped except @pytest.mark.sitl and @pytest.mark.hw.
- Ambiguity or a hardware-dependent question → STOP, leave `# QUESTION(rahul):`,
  move on. Do not invent hardware behavior.

## Contracts
sources/types.py is frozen. Implement against it; never modify it. Where your code
touches an OWNED file, depend on its contract and use a fake in tests.

## File ownership — do not implement or modify these (stubs stay stubs)
depth/backends.py, depth/holepunch.py, perception/ground_mask.py,
planner/governor.py, planner/supervisor.py, control/gate.py, control/watchdog.py,
calib/validate.py

## Hardware facts (context only)
- Camera: Waveshare AR0144 stereo — ONE UVC device, side-by-side frame, split at the
  vertical midline. 2560x720@30 MJPG (1280x720/eye). Global shutter, hardware-synced,
  baseline 52 mm, ~65° HFOV per eye.
- Rangefinders: 2× TFmini-S arrive via FC as MAVLink DISTANCE_SENSOR
  (orientation 0 = forward, 25 = down), range 0.1–12 m.
- FC: ArduPilot Copter, GUIDED. Commands: SET_POSITION_TARGET_LOCAL_NED,
  MAV_FRAME_BODY_OFFSET_NED, velocity + yaw-rate only, streamed at 20 Hz.
  GUIDED brakes on its own if setpoints stop (~3 s) — that's a feature.
- Envelope: ≤5 m/s autonomous, ≤10 m AGL, low-speed 7" quad.

## Workflow
- Branch: agent/foundation. One package at a time, in the order given.
- pytest green + ruff clean before the next package. Commit per package:
  message says what AND why, noting design decisions.
- Append a short "## Package X decisions" section to NOTES.md each time.
