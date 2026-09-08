# Disabled by default. Import into the existing drone NixOS configuration only
# after packaging with its pinned JetPack Python/OpenCV environment.
{ config, lib, pkgs, ... }:
let
  cfg = config.services.drone-record;
  launcher = pkgs.writeShellScript "drone-record-service" ''
    exec ${cfg.package}/bin/drone-record record \
      --config ${lib.escapeShellArg cfg.configFile} \
      --output-root /var/lib/drone-record/runs \
      ${lib.optionalString cfg.telemetry "--telemetry"}
  '';
in {
  options.services.drone-record = {
    enable = lib.mkEnableOption "record-only drone perception and planning";
    package = lib.mkOption {
      type = lib.types.package;
      description = "Package from infra/package.nix using the proven JetPack Python environment.";
    };
    configFile = lib.mkOption {
      type = lib.types.str;
      description = "Absolute path to deployment YAML with an absolute calibration_npz path.";
    };
    telemetry = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Receive MAVLink telemetry and request stream rates; never transmit setpoints.";
    };
    deviceGroups = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ "video" ];
      description = "Existing groups granting access to the selected camera and optional serial link.";
    };
  };
  config = lib.mkIf cfg.enable {
    assertions = [{
      assertion = lib.hasPrefix "/" cfg.configFile;
      message = "services.drone-record.configFile must be absolute";
    }];
    users.groups.drone-record = {};
    users.users.drone-record = {
      isSystemUser = true;
      group = "drone-record";
    };
    systemd.services.drone-record = {
      description = "Drone record-only autonomy";
      wantedBy = [ "multi-user.target" ];
      after = [ "local-fs.target" ];
      serviceConfig = {
        Type = "simple";
        User = "drone-record";
        Group = "drone-record";
        SupplementaryGroups = cfg.deviceGroups;
        StateDirectory = "drone-record";
        WorkingDirectory = "/var/lib/drone-record";
        ExecStart = launcher;
        # A fault or configured recording limit requires an explicit restart.
        # Automatic retries would create unbounded new recordings.
        Restart = "no";
        KillMode = "control-group";
        UMask = "0027";
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = "read-only";
        ReadWritePaths = [ "/var/lib/drone-record" ];
      };
    };
  };
}
