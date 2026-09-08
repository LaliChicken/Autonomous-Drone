# Use the installed JetPack-aware package set and its tested Python environment.
# pythonEnvironment must provide bin/python3 plus numpy, cv2, pymavlink and yaml;
# pass the environment itself, not a mkShell derivation. No generic CUDA fallback.
{ pkgs, pythonEnvironment, src ? ../. }:
let
  source = pkgs.lib.cleanSourceWith {
    inherit src;
    filter = path: type:
      let
        relative = pkgs.lib.removePrefix "${toString src}/" (toString path);
        top = builtins.head (pkgs.lib.splitString "/" relative);
      in
        toString path == toString src || (
          builtins.elem top [
            "calib" "config" "control" "depth" "infra" "perception"
            "planner" "sources" "tools" "world"
          ] && baseNameOf path != "__pycache__"
          && !(pkgs.lib.hasSuffix ".pyc" (baseNameOf path))
        );
  };
in pkgs.runCommand "drone-record" {
  nativeBuildInputs = [ pkgs.makeWrapper ];
} ''
  mkdir -p "$out/lib/drone" "$out/bin"
  cp -r ${source}/. "$out/lib/drone/"
  makeWrapper ${pythonEnvironment}/bin/python3 "$out/bin/drone-record" \
    --set PYTHONPATH "$out/lib/drone" \
    --set PYTHONDONTWRITEBYTECODE 1 \
    --set DRONE_BUILD_ID "$out" \
    --add-flags "-m tools.run"
  makeWrapper ${pythonEnvironment}/bin/python3 "$out/bin/drone-report" \
    --set PYTHONPATH "$out/lib/drone" \
    --set PYTHONDONTWRITEBYTECODE 1 \
    --add-flags "-m tools.flight_report"
''
