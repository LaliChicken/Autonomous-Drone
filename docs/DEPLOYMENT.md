# Record-only Nix deployment

The OS installation is already complete according to the handoff. These files do
not reinstall it or choose a different CUDA package set.

`infra/package.nix` accepts the existing JetPack-aware `pkgs` and an explicit
`pythonEnvironment` providing `bin/python3`, NumPy, OpenCV, pymavlink and PyYAML.
Use the tested Python environment underlying the recovered Jetson CUDA shell;
a `mkShell` derivation is not itself that interpreter environment. The package
copies only application directories and wraps the record/replay/report entrypoints.
Its store path is recorded as `build_id` because a deployed package has no .git.

In the installed drone flake, where `provenPythonEnvironment` refers to that
existing environment, construct the package with:

```nix
import /path/to/Autonomous-Drone/infra/package.nix {
  inherit pkgs;
  pythonEnvironment = provenPythonEnvironment;
}
```

Import `infra/nixos-module.nix` into `nixosConfigurations.drone`. Assign the package
above to `services.drone-record.package`, set `configFile` to an absolute readable
YAML path, then explicitly set `enable = true` when ready for recording. Keep the
camera calibration path absolute in that YAML. The module is disabled by default.
`deviceGroups` defaults to `[ "video" ]`; add the actual serial-device group only
if using the optional `telemetry = true` setting. Verify ownership/group access on
the actual device rather than guessing a UART name.

The service runs as the dedicated `drone-record` user and writes under
`/var/lib/drone-record/runs`. It uses the configured frame-byte and duration caps,
and does not automatically restart when it reaches a limit or encounters a fault.
This prevents a fault/restart cycle from continually creating logs. Caps apply per
run, not to the aggregate retained history; establish a storage quota and an archive
or deletion policy before unattended use. The implementation does not delete logs.

**Jetson-side**, after building and activating the module through the existing
flake's normal deployment workflow:

```bash
systemctl status drone-record
journalctl -u drone-record --no-pager -n 100
sudo systemctl stop drone-record
# After resolving any reported fault, explicitly start a new recording:
sudo systemctl start drone-record
```

Use the package's `drone-record replay RUN_DIRECTORY` and `drone-report
RUN_DIRECTORY` to verify the result. There is no command-transmission flag.
`--telemetry` requests MAVLink stream rates but does not arm, take off, change flight
modes or send velocity commands.

Before considering this deployment validated, check startup with a missing camera,
wrong resolution and missing calibration; camera loss while running; SIGTERM
shutdown; configured storage/duration limits; device permissions; and a sustained
run under real power and thermal conditions. Recover the historical CUDA shell and
benchmark first if they are not already in the Jetson checkout.

Only syntax parsing has been performed locally. Package building, complete module
evaluation and activation on the pinned JetPack system remain unverified. The module
uses the standard NixOS option/service mechanisms described in the
[NixOS manual](https://nixos.org/manual/nixos/stable/); the caller supplies the
[Python environment](https://wiki.nixos.org/wiki/Python) explicitly.
