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

## Package A decisions — infra

**Nearest-rank percentiles, not interpolated ones.** An interpolated p99 reports
a latency that no tick ever took. For a budget the useful claim is "the
99th-slowest tick actually took this long", so `nearest_rank` always returns a
value from the sample set. It is also exactly reproducible, which the replay
determinism test depends on.

**Stage samples live in a bounded ring** (`metrics.window`, default 2048) so a long
flight cannot grow the process. `Metrics.stage()` records in a `finally`, so a
stage that fails slowly still appears in the rollup instead of disappearing along
with the exception. Stages never recorded are *absent* from the rollup rather than
zeroed — a 0 ms row reads as "instant", which is the opposite of the truth.

**Frames are logged as the received MJPG bitstream, never re-encoded.** A re-encode
would change the pixels every downstream result was computed from, so a "replay"
would silently be a different run. Storage is one concatenated blob plus a JSONL
index of `(seq, t_ns, offset, length)`; the reader seeks by offset rather than
assuming contiguity, so a truncated log still yields the frames it can address and
raises on the one it cannot.

**The run directory stores the config twice, on purpose.** `config.yaml` is the
verbatim YAML mapping (degrees) and loads straight back through
`load_config_from_dict`; `config.json` is the resolved view (radians, derived) for
reading. `Config.to_dict()` deliberately is *not* re-loadable, so `Config.raw` was
added to carry the original mapping — without it replay cannot reconstruct the
config a run was flown with, which makes A/B meaningless.

**Wall clock appears exactly twice** — the run directory name and `created_utc` —
and never in a record the pipeline consumes. CLOCK_MONOTONIC cannot name a
directory a human will look for tomorrow, but everything read back into the
pipeline is monotonic ns.

**`git_revision()` never raises.** A run must stay loggable outside a checkout or
with git absent; the failure reason is recorded in the meta so the log explains
why it has no revision rather than silently claiming none.

**Replay takes the pipeline as an injected factory, not an import.** Partly because
perception and planning do not exist yet, but mainly because replay must drive the
same object the live stack builds — a replay that constructs its own pipeline is
testing a lookalike. `ReplayPipeline` (`reset()` + `tick()`) is the whole contract.

**Determinism is tested with a guard on the guard.** `FakePipeline` is deliberately
stateful and config-sensitive, and there is a second test using a subclass whose
`reset()` does nothing, asserting that the streams *do* diverge. Without it, the
determinism test could pass because the fake was too simple to fail.

**Telemetry pairing is a zero-order hold, not interpolation.** `FcState` carries
`mode`, `armed`, and `rc`, none of which can be averaged, so the honest answer for
a frame is the last state actually received at or before it.

**Non-finite distances round-trip through the log.** Python's JSON extension
encodes `NaN`/`Infinity` and reads them back, and Package C needs that: a bin with
no measurement must not serialise to a number that reads as a range. Cost is that
`planner.jsonl` is Python-JSON, not strict RFC 8259 — `jq` will reject those lines.
Acceptable while the only consumer is this stack's replay.

**`decode_frame` currently owns the split rule.** `sources/stereo_uvc.py` is empty
and was not part of packages A–D, so replay decodes MJPG and splits at the midline
itself. It splits at the midline of the *decoded* frame rather than at
`config.camera.frame_width / 2`, so an older log at a different resolution fails
loudly instead of handing over two misaligned halves. Left a `# QUESTION(rahul):`
that this should call into `stereo_uvc` once it exists — two places deciding where
the midline is, is one too many.

**`infra/netview.py` was left empty**: not part of the Package A brief.

## Package B decisions — depth

**Invalid depth is NaN, not 0.0 and not clipped.** CLAUDE.md forbids `depth_m=0.0`;
NaN is the positive choice rather than merely a non-zero one, because every
comparison against NaN is False. A pixel that leaks past a `valid` check still
fails "is this closer than X" instead of passing it, which is the safe direction.
`blank_invalid()` is the last line of defence and runs unconditionally.
Out-of-range depths are *invalidated, not clipped* — clipping would turn a 40 m
reading into a confident 12 m obstacle.

**Zero disparity is treated as unmeasurable, not as a match.** OpenCV marks
unmatched pixels `(minDisparity-1)*16`, but a matched disparity of exactly zero is
the point at infinity and reprojects to an enormous Z that would sail through the
range clamp looking like a real measurement. `disparity_valid_mask` requires both
conditions.

**Speckle removal lives in the disparity domain and its threshold is in disparity
pixels.** Two reasons. `cv2.filterSpeckles` rejects float32 outright in OpenCV 5
(int16/uint8 only), and more fundamentally a fixed metre tolerance maps to a
different disparity step at every range, so `speckle_max_diff_m` was not an
honestly expressible tunable. Renamed to `speckle_max_diff_px`.

**`remove_speckles` copies explicitly.** `np.ascontiguousarray` returns the *input
object* when it is already contiguous int16, and `filterSpeckles` works in place,
so the first version silently modified its caller's disparity map. Caught by a test
that asserts the input is unchanged, which is worth keeping.

**The L/R consistency check abstains where it cannot see — this was the important
decision in this package.** The right-view disparity map is structurally invalid
over its rightmost `numDisparities` columns; that is a consequence of the search
direction, not a data problem, and no configuration avoids it. Treating "could not
check" as "failed the check" blanked ~10% of the image width at the right edge of
*every* frame. Measured on a planar scene, coverage in the right third fell to
essentially nothing. Downstream those azimuth bins would be permanently `unknown`,
and since unknown is impassable, **the planner could never turn right**. Abstaining
is not "unknown = clear": the pixel still carries real evidence from the forward
match, it simply has no second opinion available. With abstention the check earns
its cost cleanly — on the occlusion scene, leak 5.0% → 3.2% and depth RMSE
0.058 → 0.051 m, for 0.09 points of coverage and no edge asymmetry.

**Confidence is a geometric bound, not a match score.** Differentiating Z = f·B/d
gives σ_Z = Z²·σ_d/(f·B), so range error grows with the *square* of range — the
dominant fact about stereo. Confidence falls linearly to zero at
`max_depth_sigma_m`. It says how precise a correct match can be at that range, not
how likely the match is to be right; those are different questions and conflating
them would overstate what the number means.

**The synthetic generator builds the pair *from* the disparity, not by shifting the
left image.** The right image is a window into a padded texture and the left image
is the same window displaced, so disparity is an input rather than something
recovered, and both views are exact texture crops with no resampling blur to
flatter the matcher. `tests/test_synthetic_stereo.py` verifies the pairs against
their own ground truth by direct pixel comparison, never through StereoSGBM — a
generator that disagreed with its own GT would make every downstream number
meaningless.

**The first occlusion scene was not one.** Displacing a region of a single
continuous texture occludes nothing: the right view stays consistent with both the
near and the far interpretation, the matcher finds a background match that
genuinely agrees in both views, and there is nothing for a consistency check to
catch. The measured "leak" was an artefact of that. The slab now carries its own
independent texture, which makes the occlusion real and gives the L/R check
something true to find.

**The committed baseline is measured on synthetic scenes, deliberately.**
`vision.middlebury.edu` is unreachable from this environment (its TLS chain does
not verify here; general network is fine), but even with access the baseline
belongs on data that needs no download — CI can then check it offline and
reproducibly. `tools/get_middlebury.py` is written and its parsing is fully tested
against local fixtures, but **the download path has never executed end to end**;
there is a `# QUESTION(rahul):` on it asking for one manual run.

**Honest limitation: the synthetic scenes are too easy.** `bad_pixel_pct` is 0.00
on three of four, so that metric currently has no headroom to detect a regression.
`disparity_rmse_px`, `coverage_pct`, and `occlusion_leak_pct` do carry real signal
and are what the guard actually rests on. Real Middlebury scenes would fix this.

**Timing, unprompted but relevant: full resolution does not hit 30 Hz on this CPU.**
At 1280×720 with 128 disparities, `SgbmCpuBackend.infer` is ~86 ms with the L/R
check and ~53 ms without — 11.6 Hz and 18.9 Hz, on a desktop x86 part. A Jetson
Orin Nano CPU will be slower. `depth/sgm_jetson.py` (the presumable CUDA path) is
an empty stub and was not in packages A–D. Flagged rather than acted on.

**`depth/backends.py` and `depth/holepunch.py` untouched** — owned stubs.
`SgbmCpuBackend` satisfies the `DepthBackend` protocol structurally.

## Package C decisions — perception + world

**`ObstacleScan` is not an `OccupancySnapshot`.** The scan deliberately has no
`danger` field, because danger depends on airspeed and perception has no business
knowing how fast the aircraft is going. `world.occupancy` adds it at snapshot time,
where the speed actually is.

**An unknown bin carries NaN range, not zero and not infinity.** Zero reads as an
obstacle at the lens; infinity reads as clear. NaN is the only value that fails
*both* "is this closer than X" and "is this further than X", so a bin whose
`unknown` flag someone forgot to check still cannot be mistaken for either. The
same reasoning as the depth NaN rule, one level up.

**5th percentile, not minimum.** The minimum of half a million noisy points is
whatever the single worst outlier happened to be — one surviving speckle would park
a phantom obstacle in the bin. The 5th percentile still answers "how close is the
near surface" but needs a few hundred pixels to agree before it moves. There is a
test that drops one rogue pixel at 0.5 m into an 8 m wall and asserts the bin does
not move.

**Per-bin percentile via one argsort, not N masks.** Masking per bin costs N passes
over the full point cloud; at 1280×720 with 32 bins that is tens of millions of
comparisons per frame. One `argsort` plus `searchsorted` gets the same answer in a
single pass.

**Merging keeps the nearer of new and remembered.** If two recent observations
disagree about how close something is, the closer one is the one worth flying by.
The cost — a transient false positive holds the bin pessimistic — is bounded by
`max_age_s` rather than permanent, and there is a test for exactly that expiry.
Pessimistic is the survivable direction.

**A scan that reports `unknown` does not erase memory.** A frame where the matcher
found nothing is not evidence of empty space, so an unknown scan bin leaves the
remembered range alone and lets normal decay handle it.

**`danger` and `unknown` are separate and neither implies the other.** `danger`
means "something measured is too close for the current speed"; `unknown` means
"nothing was measured". `NaN < threshold` is False, so unknown bins are never
flagged dangerous — that is deliberate, not an accident of NaN. A planner must
refuse both, but for different reasons, and collapsing them would hide which one is
happening. `passable()` is the single place the "unknown is impassable" rule is
written down, so the planner cannot accidentally reimplement it as "unknown is
fine".

**The occupancy map never reads a clock.** Every timestamp is passed in, so
replaying a log reproduces the same state exactly. Tested by running the same
sequence twice and comparing.

**Snapshot ages as well as update.** A planner ticking faster than the camera has
to see stale bins expire on time, not at the next frame.

**Red uses two hue bands because it wraps 0/179.** A single `inRange` cannot express
it, and the half a one-band detector loses is typically the saturated half — i.e.
the target. There is a test that takes a red which lands in the high band, asserts
the low band alone misses it, and asserts the two-band mask catches it, so the
reason for the complexity is pinned down.

**Two range estimates are reported, never blended.** `range_stereo` (median valid
depth in the box) needs texture and stereo overlap but does not care about the
box's true size; `range_size` (`f·w_m/w_px`) is always available but wrong if the
box is not the configured width or is clipped by the frame edge. They fail in
unrelated ways, so their *disagreement* is diagnostic — `range_agreement` exposes
it, and `touches_border` flags the case that breaks `range_size`. Averaging them
would destroy exactly the signal worth having.

**Box confidence is `solidity × fill_ratio`, with no weights.** Both cues are
already 0..1 and a genuine square scores high on each, so the product needs no
tuning constants — and weights would be two more tunables with no principled value.
Range agreement is deliberately *not* folded in, since it is unavailable without
depth and would make confidence mean different things frame to frame.

**Left QUESTION(rahul) in `perception/obstacles.py`: the FOV is not symmetric and
the config does not admit it.** StereoSGBM cannot match the leftmost
`numDisparities` columns, so with 128 disparities on a 1280 px eye about 6.5° of
the 65° FOV — roughly the leftmost 3 of 32 bins — is unknown on every frame,
permanently. Since unknown is impassable, the planner sees a standing wall in its
left periphery and will be biased against left turns. This is the mirror image of
the right-edge problem fixed in Package B, except that one was mine to fix and this
one is inherent to the geometry. Three options offered (narrow `fov_deg` to the
observable wedge, accept the bias, or run a right-referenced matcher); it needs a
flight-behaviour call, so I did not pick.

**`perception/ground_mask.py` untouched** — owned. `scan_from_depth` takes an
optional `exclude_mask` and applies it, and never tries to infer the ground itself.

## Package D decisions — SITL control chain

**`behaviours.py`, not `behaviors.py`.** The brief used the American spelling; the
repo already had the British one and the instruction was to fill existing files
rather than restructure. Flagging it because a future `from planner.behaviors
import ...` will fail.

**Two additions to the VFH-lite spec, both deliberate.**

*Candidates also exclude `danger`.* The brief says "clearance > threshold AND not
unknown". Danger is a third condition and is *not* implied by the other two: at
2.5 m/s the danger distance is ~3.25 m while `clearance_threshold_m` is 2.5 m, so a
bin at 3.0 m passes the clearance test while being flagged too close to stop in.
Excluded via `world.occupancy.passable` so the rule stays in one place. There is a
test that constructs exactly that band and asserts zero candidates.

*The aircraft yaws toward its heading instead of translating along it.* A quad can
fly sideways and it would be faster, but sideways is where the only forward-facing
camera is not looking, so the occupancy map there is unknown — and unknown is
impassable. Forward speed is scaled by `cos(theta)` and the aircraft turns to face
where it is going. Never translate where you cannot see.

**The turn term as literally specified was unusable, and this was the real bug of
the package.** `w_turn * |turn_rate|` with `turn_rate = Δbearing/dt` scales as
`1/dt`, so the weight means something different at every loop rate. At the actual
20 Hz it outweighed the goal term roughly tenfold and the planner simply refused to
turn — with the goal at +30° it picked +2°. Two fixes: the rate is normalised by
`max_yaw_rate` into a dimensionless 0..1 penalty, and there is no penalty at all on
the first plan after a reset or STOP, since hysteresis needs something to be
hysteretic about.

**Yaw rate needed a real gain, so `local_planner.yaw_align_time_s` was added.**
Using `dt` as the gain made a 2° heading error — less than one bin width — command
full yaw rate, and silently changed the gain with the loop rate. Now
`yaw_rate = bearing / yaw_align_time_s`, capped by `max_yaw_rate`. There is a
regression test asserting a small error does not command full yaw.

**Guards are pure functions of a frozen `GuardContext`, in one marked block.** The
context is frozen and self-contained so a guard cannot reach into the machine and
depend on history the tests do not control. Each is tested individually; the
machine is tested separately for the sequencing it puts around them.

**The speed clamp is applied to every emitted command, not on state entry.** A clamp
that only fires at the entry tick does nothing about the tick after it. Tested by
asking for 99 m/s on every tick for 60 ticks.

**Losing flight authority outranks every other transition, from every state.** Not
GUIDED, disarmed, EKF unhappy, or no FC state at all → IDLE immediately.

**Offboard staleness brakes smoothly rather than cutting.** Three mechanisms cover
three failures: this layer ramps to zero over `command_timeout_s` so a few dropped
frames do not jolt the aircraft, the owned watchdog does the emergency cut, and
GUIDED brakes by itself after ~3 s of silence. Freshness is measured from
*submission*, not `command.t_ns` — the command's own timestamp is when the frame was
captured, so using it would count perception latency as staleness and brake for no
reason. A long tick gap is clamped to 4 periods, or a stalled loop could slew by
`accel × gap` and the acceleration limit would stop meaning anything on exactly the
tick that matters.

**Attitude interpolation goes the short way round.** Yaw crosses ±π routinely and a
naive lerp between 179° and −179° gives 0° — pointing exactly backwards while the
aircraft is doing nothing unusual. `attitude_at` also refuses to extrapolate beyond
`attitude_max_extrapolation_s`: a frame timestamped before the link came up has no
attitude, and inventing one silently mis-projects every obstacle in it.

**Out-of-range rangefinder readings are dropped, not stored.** A TFmini reporting
its maximum means "nothing seen", not "the ground is 12 m away"; storing it would
put a phantom floor under the aircraft.

**SYS_STATUS: absent-and-disabled is not a failure.** Only present-and-enabled-but-
unhealthy means the EKF is broken, or every FC without the sensor would read as
faulty.

**The MAVLink tests use real pymavlink messages over a fake transport.** Only the
socket is faked — the hardware boundary. Every field mapping, the type mask (1479),
and `MAV_FRAME_BODY_OFFSET_NED` are checked against genuine message objects rather
than a test's idea of them.

**The red-box scenario puts the target at 40°, outside the ±32.5° FOV, on purpose.**
The box is invisible until SEARCH has yawed far enough to bring it into view. A
scenario with the target already in frame would never exercise the search behaviour
at all, which is the interesting part.

**The wall scenario is a plane, not a cylinder.** Range grows as `d/cos(bearing)`
across the bins because that is what a flat wall looks like in polar coordinates. A
constant-range "wall" is a cylinder centred on the aircraft and would let a planner
that only reads the centre bin pass a test it should fail.

**Non-SITL tests are the real coverage.** They drive occupancy → planner → behaviour
machine through a stated geometry and assert on the decision. The SITL tests fly the
same scenarios against real ArduPilot and assert on what the aircraft did; they are
skipped unless `DRONE_SITL=1`. **The SITL tests have never been executed** — there is
no ArduPilot checkout in this environment. They are written against the documented
MAVLink behaviour and are unverified end to end.

**`control/gate.py`, `control/watchdog.py`, `planner/governor.py`,
`planner/supervisor.py` untouched** — owned. `OffboardLoop` takes `emit` as an
injected callable; in tests that is a list, in flight it is the gate.

## Package E0 decisions — restore validation

Removed stray backticks from the behaviour guard without changing its logic.
Preserved pre-existing executable-bit changes and the SITL diagnostic print.
Validation: 467 passed, 6 opt-in tests skipped; project-scoped Ruff clean;
`tools.bench --check` reports no regressions. The CUDA shell/comparison patches
are absent from local branches and the searched workspace. Their recovery and
Jetson measurements remain external dependencies; historical counts are not
results from this checkout.

## Package E decisions — live stereo source

Capture owns one bounded shared-memory MJPG slot in a spawned process. Blocking
OpenCV calls can be terminated on shutdown; the consumer never waits through a
backlog. Sequence gaps expose discarded frames. Invalid formats, shape changes,
non-monotonic timestamps, disconnects and timeouts latch a fault until restart.
Raw bytes are preserved and replay now delegates to the source's decoder/splitter.
Existing log configs acquire additive capture defaults from the shipped YAML.

The OpenCV V4L2 transport is unverified on the actual camera. Its timestamps are
explicitly host-dequeue CLOCK_MONOTONIC, not exposure timestamps. The frozen frame
contract is unchanged; latency from these timestamps excludes earlier device
buffering. Flight use remains blocked on timestamp verification. Raw logging
requires MJPG; a decoded BGR fallback is deliberately not re-encoded.

Validation: 477 passed, 6 opt-in tests skipped. New tests exercise real spawned
workers with synthetic transport, latest-frame delivery, original byte retention,
decode/replay equivalence, disconnects, corrupt packets, shape mismatch and bounded
shutdown of a stalled read. No physical camera was opened.

## Package F decisions — calibration candidates

Added original-MJPG calibration recording, measured-checkerboard corner detection,
OpenCV intrinsic/stereo solving and a versioned NPZ with maps, geometry, fit RMS,
input hashes, sequences and configuration metadata. Board dimensions default to
null and solving refuses to guess them. Output creation is exclusive.

Rectification verifies dimensions, finite arrays and supported horizontal geometry;
its valid disparity ROI excludes unsupported edges. It exposes the inverse left
rectification rotation for downstream body-frame projection. Calibration fit RMS
is not held-out accuracy; every solver output is explicitly unvalidated. The
owned calib/validate.py remains empty. Physical board diversity, held-out epipolar
error and measured-distance acceptance still require hardware/owner delivery.

Validation: 482 passed, 6 opt-in tests skipped; Ruff clean. Projected checkerboard
observations exercise the real solver and recover the 52 mm baseline within
0.1 mm; corrupt artifacts and resolution mismatch are rejected. No real camera
calibration or physical accuracy claim is made.
