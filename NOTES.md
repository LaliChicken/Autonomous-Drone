# NOTES

Design decisions, one section per package. Newest last.

## Step 0 decisions — contract + config

**sources/types.py is byte-for-byte the spec.** It declares `vx: float; vy: float;
vz: float` on one line, which trips ruff's E702, and puts `import numpy as np`
adjacent to the stdlib imports, which trips I001. Rather than reformat a frozen
file, both rules are waived for that one path in `pyproject.toml`
(`[tool.ruff.lint.per-file-ignores]`). `tests/test_types.py` pins the field names
and the frozen/mutable split so an accidental edit fails loudly.

**Degrees in YAML, radians everywhere else.** `config.yaml` is written in degrees
because it is the human-edited surface; the loader converts on the way in and the
dataclass fields are named `*_rad` / `*_rad_s`. A test walks every config dataclass
and asserts no field name ends in `_deg`, so degrees cannot leak downstream.

**Config dataclasses hold scalars only.** Derived matrices (`nominal_q`,
`rotation_body_from_cam`) are methods returning fresh arrays rather than stored
fields. Keeping numpy out of the dataclasses means a `Config` stays comparable,
hashable, and trivially serialisable into the flight-log config snapshot — a
stored array would break `__eq__` and `asdict`.

**Mount angles follow the FC sign convention, not "camera tilt".** Positive
`mount.pitch_deg` is nose-UP and positive `yaw_deg` is to the right, matching
`FcState.roll/pitch/yaw` as ArduPilot reports them. A downward-tilted camera is
configured with a *negative* pitch. The alternative (positive = tilt down, which
reads more naturally for a mount) was rejected: having two pitch signs in one
codebase is exactly the kind of thing that produces an inverted obstacle map.

**Camera -> body is a fixed axis swap composed with the mount rotation.**
`rotation_body_from_cam` = `Rz(yaw) @ Ry(pitch) @ Rx(roll) @ swap`, where swap maps
camera RDF (x right, y down, z forward) onto body FRD (x forward, y right, z down).
Tested against orthonormality, `det == +1`, and each axis individually.

**Cross-section validation is where the real bugs get caught.** Single-field range
checks are cheap; the checks that matter compare sections — `obstacles.n_bins` must
equal `occupancy.n_bins`, `behaviours.autonomy_speed_max_mps` must fit inside
`envelope.max_speed_mps`, `local_planner.clearance_threshold_m` must be reachable
given `depth.postprocess.max_depth_m` (otherwise no bin could ever be a candidate
and the planner would silently always STOP).

**The danger formula lives in `OccupancyConfig.danger_distance`,** not in
`world/occupancy.py`. CLAUDE.md forbids magic numbers in module code, and the
formula is three tunables glued together — putting the arithmetic next to the
tunables keeps `occupancy.py` free of constants entirely.

**Nominal Q is a development-only fallback.** With no calibration file, the loader
synthesises Q from baseline + HFOV + resolution assuming perfect rectification and
centred principal points. That is good enough for synthetic tests and wrong for a
real camera. Left a `# QUESTION(rahul):` in `config/schema.py` asking whether
`sgbm_cpu` should hard-refuse to run without `camera.calibration_npz`.

**Packages stay namespace packages.** No `__init__.py` was added to the existing
directories, since the brief was to fill files rather than restructure; `config/`
is new so it gets one for a clean public API. `pytest.ini_options.pythonpath = ["."]`
makes imports work.

**Python 3.13 in `.venv`.** System Python is 3.14 and had none of the deps;
`opencv-python` wheel coverage is safer on 3.13. Installed: numpy 2.5.1,
opencv-python 5.0.0.93, pymavlink 2.4.49, pyyaml 6.0.3, pytest 9.1.1, ruff 0.16.1.
Note OpenCV is v5 — `StereoSGBM_create`, `reprojectImageTo3D`, and `filterSpeckles`
were all verified present before Package B was designed around them.
