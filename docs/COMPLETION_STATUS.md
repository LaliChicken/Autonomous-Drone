# Completion status

Implementation update: 2026-09-08 UTC. This records verified checkout work, not
Jetson hardware or flight acceptance. Historical context remains in CHAT_HANDOFF.md.

| Milestone | Delivered | Still required |
| --- | --- | --- |
| Baseline | Syntax repair; passing unit tests, Ruff and depth regression check | Recover missing Jetson CUDA shell/comparison patch from its authoring source |
| Capture | Latest-frame process, original MJPG bytes, shared replay decoder, fault detection, bounded shutdown, opt-in hardware test | Verify actual negotiated mode, eye ordering, device buffering and exposure timestamps on the Jetson |
| Calibration | Capture command, measured-checkerboard solver, candidate NPZ, rectification and geometry checks | Measured board configuration, varied physical samples, owner validator, held-out alignment/range acceptance |
| GPU | No runtime GPU implementation or configuration promotion | Recover comparison patch; run hardware tests and native-resolution CPU/GPU accuracy/latency comparison; select backend only from results |
| Record-only runtime | CPU pipeline, telemetry pairing, logging, portable calibration snapshot, deterministic replay, reports | Sustained calibrated live-camera run on the Jetson and real thermal/performance measurements |
| Safety and sensors | Per-stream/per-sensor ages, bounded history, stale effective telemetry, fresh downward range | Owner gate/watchdog/governor/supervisor, forward-range fusion, verified range sentinels, takeover configuration, measured stopping limits |
| Deployment | Nix package requiring the proven Python environment; disabled-by-default NixOS module; per-run size/duration limits | Build and activate on the pinned JetPack system; verify device groups, full environment, lifecycle, total storage budget and retention policy |
| Validation | CPU end-to-end fault/replay tests; existing SITL flight scenarios; record-only runner with SITL telemetry | Real capture/rectification accuracy, complete safety-boundary fault tests, assembled-platform measurements and controlled flight acceptance |

The following owned files remain unchanged and empty: depth/backends.py,
depth/holepunch.py, perception/ground_mask.py, planner/governor.py,
planner/supervisor.py, control/gate.py, control/watchdog.py and calib/validate.py.
The frozen sources/types.py content is unchanged.

## Next steps

1. **Workstation:** recover the CUDA shell/comparison changes mentioned in the
   handoff; they were not found in local branches or the searched workspace.
2. **Jetson, proven CUDA shell:** run `python3 -m tools.run devices`; select a stable
   UVC path in `capture.device_path` and run the opt-in capture test. Confirm original
   MJPG retrieval works on this driver. Host-dequeue timestamps do not establish
   hardware synchronization or exposure latency.
3. **Jetson:** set the measured inner-corner counts and square size in a deployment
   YAML, record diverse calibration poses, then solve a candidate. The owned
   validator must establish acceptance from held-out data and measured ranges.
4. **Jetson:** run calibrated record-only perception and save its report; review
   unknown coverage, stage timing, host-dequeue age and drops. Separately run the
   recovered GPU experiment; do not change the committed baseline to hide regressions.
5. **Owners:** deliver the safety interfaces and validation criteria. Keep command
   transmission unavailable until these dependencies and hardware gates are met.
6. **Jetson:** build the deployment package with the proven interpreter environment,
   validate the service on the assembled platform, establish log retention and
   loaded-power stability, then perform the remaining controlled acceptance steps.

## Evidence and limits

- Default suite: 492 passed, eight opt-in tests skipped (seven SITL, one hardware).
- Existing SITL scenarios: six passed against the local built ArduPilot simulator,
  normally armed in GUIDED and taken off to 5 m. No physical FC was contacted.
- Record-only SITL integration: one passed separately with real telemetry through
  the CPU runner and replay, after reaching the documented airborne precondition.
- Project-scoped Ruff is clean; committed synthetic depth baseline has no regressions.
- Five-frame CLI synthetic run and command comparison completed successfully.
- Both Nix files parse. No nixpkgs evaluation environment is configured on this
  workstation, so no package build or NixOS module activation is claimed.

Synthetic solver accuracy is a software geometry check, not camera accuracy.
The current runner uses CPU SGBM; there is no GPU speedup or live frame-rate claim.
SITL scenario tests inject geometry and test selected flight behaviors; they do not
validate the missing command gate or simulate an actual stereo camera.
