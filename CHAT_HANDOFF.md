
# Autonomous Drone: chat handoff

Snapshot: 2026-09-08. This is project history, not proof of the current checkout.
Read the repository's instructions and inspect its actual files before editing.

## First actions for the receiving agent

1. Read applicable AGENTS.md files, CLAUDE.md, README.md and NOTES.md.
2. Inspect `git status --short` and `git rev-parse HEAD`. Preserve local changes.
3. Check which patches and tools below actually exist. The public repository may
   lag behind Rahul's Jetson checkout; chat-created commits were not pushed.
4. Summarize verified state and the outstanding next step. Do not redo OS setup.

## Goal and working style

Rahul is developing a companion-computer autonomy stack for a 7-inch quadcopter.
Give small, numbered steps and label commands as Arch-side or Jetson-side.
He wants to log milestones, tests and benchmark results. Explain what a result
proves and what remains untested. Preserve timestamps, coordinate conventions,
unknown-space handling and the existing safety boundaries.

Physical assembly and printed mounts are unfinished. Power/motor soldering is
done; a replacement barrel connector passed continuity checks and measured 12.1 V.
Those checks did not establish loaded power stability. Do not assume a camera,
LiDAR or flight controller is connected to the running software.

## Machines and environment

- Main workstation: Arch Linux. Rahul works over SSH to the Jetson.
- Last working SSH destination: `rahul@192.168.68.51`; the IP can change.
- Jetson: Orin Nano Super developer kit, 8 GB class, hostname `drone`, user `rahul`.
- NixOS is installed and booting from NVMe: root ext4 on nvme0n1p2, EFI vfat on p1.
- JetPack 7 / Jetson Linux 39.2.1; latest reported power mode was 25 W, mode 1.
- `/etc/nixos/flake.nix` uses `nixosConfigurations.drone` and pins jetpack to:
  `github:anduril/jetpack-nixos/babe3e5558ca130985c9461d0e59bbbebbc43564`.
- nixpkgs follows JetPack's pinned nixpkgs:
  `714a5f8c4ead6b31148d829288440ed033ccc041`.
- Use the installed system's JetPack-aware package set for CUDA, not an unrelated
  generic nixpkgs CUDA environment. CUDA 13.2 and Orin SM 8.7 were selected.
- Repo on Jetson: `~/Autonomous-Drone`; logs: `~/drone-logs`.
- Public repository: https://github.com/LaliChicken/Autonomous-Drone
- Initial user-reported clean revision: `672466b9cc237798dec9f8987dc1a0c4a68185bb`.
  Patches were applied afterward; do not assume that is the present tree state.

## CUDA OpenCV setup: completed on the Jetson

The ordinary Python Nix shell had CPU-only OpenCV. A direct CUDA context and
32-element GPU addition test had already passed, so GPU hardware/driver access
was established independently of OpenCV.

The CUDA shell is `infra/jetson-cuda.nix`. Enter it with:

```bash
cd ~/Autonomous-Drone
nix-shell infra/jetson-cuda.nix
```

It reads the installed drone flake, builds selected OpenCV 4.13.0 modules, exposes
the driver libraries and includes Python, NumPy, pymavlink, YAML, pytest and Ruff.
Python is provided by the shell; it was not globally installed initially.

Two build corrections were necessary and are already applied by Rahul:
- Explicitly include `cudev` in enabledModules.
- Set `runAccuracyTests = false; runPerformanceTests = false;` because the module
  selection excluded upstream test executables but packaging tried to move them.
  Python import checks and the project's tests are retained.

Rahul confirmed:
- CUDA devices: 1; `cv2.cuda.createStereoSGM` exists.
- Capture the returned object: `result = matcher.compute(gpu_left, gpu_right)`.
  Ignoring the return and downloading an initially empty output argument yielded
  None. Capturing the return yielded a nonempty GpuMat.
- `raw = result.download()` is int16 fixed-point disparity; divide by 16 for pixels.
- Synthetic 16-pixel shift test: median 16.0 pixels; 99.64% of interior pixels
  within one pixel. This is a smoke test, not real-world accuracy or performance.
- In the CUDA shell: 475 tests passed, 6 deselected in 15.42 s, and
  `python3 -m tools.bench --check` reported no regressions.
- User environment reported OpenCV 4.13.0 and NumPy 2.4.4 before the CUDA rebuild;
  check current NumPy/build metadata instead of assuming it remained identical.

## Existing software

Implemented: CPU SGBM, depth cleanup, obstacle scanning, HSV target detection,
temporal polar occupancy map, VFH-lite planning, mission state machine,
MAVLink reader/sender, 20 Hz slew-limited command streaming with stale-command
handling, metrics, flight logs, replay, synthetic fixtures and SITL scenarios.

Runtime is not a complete live-camera autonomy application yet. The standalone
monocular Depth Anything/TensorRT resume project is separate from this stereo repo.

Read CLAUDE.md for ownership and frozen contracts. In the reviewed tree,
`sources/types.py` is frozen, and several empty modules are explicitly owned
elsewhere. Listing missing work does not override those restrictions.

## Benchmarks already measured by Rahul

Native CPU: 1280x720 per eye, 128 disparities, extra L/R check enabled.
Report: `~/drone-logs/latency-20260906T202318Z.json`.

| Scene | Mean processing ms | p99 ms | Compute-only pairs/s |
| --- | ---: | ---: | ---: |
| planar_d24 | 524.267 | 547.584 | 1.91 |
| slanted | 543.377 | 563.500 | 1.84 |
| occlusion | 519.300 | 546.251 | 1.93 |

CPU matching takes about 152 ms per direction; total depth about 380 ms;
obstacle scan about 138-164 ms. These exclude capture and the full control loop.

Half-resolution experiment: same native pairs downsampled to 640x360 per eye,
64 disparities, scaled geometry, same baseline and other postprocessing settings.
Report: `~/drone-logs/half-resolution-20260906T211344Z.json`.
Mean totals: 122.089, 128.519, 121.928 ms respectively (about 4.2-4.3x faster).
Do not adopt this configuration based on speed alone:
- Occlusion depth RMSE: 0.07495 -> 0.12514 m.
- Occlusion leakage: 1.61073% -> 5.43981%.
- Slanted scene maximum bin range increase: +0.21118 m versus CPU native output;
  this is not a ground-truth error or proof of a systematic bias.
The committed accuracy baseline and flight configuration were not changed.

## Latest delivered patch: GPU comparison, results still pending

`jetson-cuda-benchmark.patch` was delivered, but Rahul has not confirmed applying
or running it. Check for `tools/bench_cuda.py` and the other files before proceeding.
The authoring checkout has benchmark commit `ceb1bae` after shell commit `fe0161a`;
these are local authoring commits, not confirmed public GitHub/user commits.

It adds:
- `tools/cuda_stereo_experiment.py`: GPU matcher adapter using the returned GpuMat.
- `config/cuda_benchmark.yaml`: independent candidate P1=10, P2=120,
  uniqueness=5, mode HH4; no flight configuration changes.
- Explicit matcher injection into the existing CPU backend so Q, speckles,
  extra flipped L/R pass, confidence and invalid-depth handling stay shared.
- `tools/bench_cuda.py`: identical native scenes, fresh CPU and GPU timings,
  accuracy against GT, scan differences, raw timings, hashes and thermal readings.
- `tests/test_bench_cuda.py`: five CPU-side cases and two hardware-marked cases.

Authoring-environment validation: 480 passed, 8 deselected; Ruff clean; patch
application checked. Local OpenCV 4.13.0 / NumPy 2.5.3, no GPU execution there.
Do not report these as Jetson benchmark results.

Next commands, after verifying the patch is applied, in the Jetson CUDA shell:

```bash
DRONE_HW=1 python3 -m pytest -q tests/test_bench_cuda.py
# Expect 7 passed before proceeding.
run_stamp=$(date -u +%Y%m%dT%H%M%SZ)
set -o pipefail
sudo nvpmodel -q 2>&1 | tee ~/drone-logs/cuda-power-$run_stamp.txt
python3 -m tools.bench_cuda --warmup 10 --frames 200 \
  --output ~/drone-logs/cuda-comparison-$run_stamp.json \
  2>&1 | tee ~/drone-logs/cuda-comparison-$run_stamp.txt
```

Review accuracy, coverage, occlusion leakage, unknown bins and timing together.
GPU SGM uses Census and internal median/consistency filtering; it is not the
same algorithm as CPU SGBM. compute_download includes blocking GPU completion and
download. No GPU speedup, live camera rate or flight readiness is established yet.

## Remaining development, in proposed order

1. Live stereo capture: sources/stereo_uvc.py is empty. Camera discovery, split,
   monotonic timestamps, bounded latest-frame delivery, capture fault detection.
2. Calibration capture, solve, validation and rectification; calib files are empty.
3. Runtime GPU backend after experiment review; depth/sgm_jetson.py is empty.
4. Main application connecting all stages, initially in record-only mode.
5. Command gate/watchdog: authority, stale perception, worker failures, takeover.
6. Speed governor/health supervisor tied to measured latency and stopping behaviour.
7. Extend existing LiDAR/MAVLink parsing with per-sensor freshness and integration.
8. Ground exclusion as needed, and explicit treatment of stereo's observable region.
9. Reproducible Nix application deployment, device permissions, service lifecycle,
   health reporting and log retention.
10. Full-runner replay/SITL/bench fault tests and end-to-end latency measurements.

The suggested next coding task was live stereo capture while the GPU experiment
is pending. Rahul asked for this handoff; he has not yet chosen a new coding task.
Do not automatically fill every owned stub or enable live command transmission.
## Local implementation update — 2026-09-08 UTC

The earlier snapshot above is retained as history. The current workstation branch
`agent/foundation` now has bounded original-MJPG capture, candidate calibration and
rectification, a record-only CPU application, portable deterministic replay,
per-sensor telemetry freshness, reports, and a Nix deployment package/module.
Read `docs/COMPLETION_STATUS.md`, `docs/DEPLOYMENT.md` and the appended NOTES.md
packages E0–H for delivered work, evidence and remaining gates. The CUDA shell and
comparison patch remain absent. Runtime GPU selection, owned safety/validation
modules and physical acceptance are not completed. There is no transmit option.
Default checks: 492 passed / 8 opt-in skipped. Existing SITL scenarios: 6 passed;
new runner-with-SITL-telemetry test: 1 passed separately. These are local simulator
results, not new Jetson hardware benchmarks. The next action is to recover the
CUDA patch and verify capture/calibration on the actual Jetson/camera.
