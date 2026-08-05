#!/usr/bin/env bash
# Launch ArduPilot Copter SITL for the scenario tests.
#
#   ./scripts/run_sitl.sh              # start SITL on udp:127.0.0.1:14550
#   ./scripts/run_sitl.sh --speedup 5  # run faster than real time
#
# Then, in another terminal:
#
#   DRONE_SITL=1 .venv/bin/python -m pytest -m sitl
#
# SITL is not vendored here. Point ARDUPILOT_HOME at a checkout of
# https://github.com/ArduPilot/ardupilot (see its BUILD.md for the one-time
# setup), or put sim_vehicle.py on PATH.

set -euo pipefail

ENDPOINT="${DRONE_SITL_ENDPOINT:-udp:127.0.0.1:14550}"
FRAME="${DRONE_SITL_FRAME:-quad}"
SPEEDUP=1
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --speedup) SPEEDUP="$2"; shift 2 ;;
        --frame)   FRAME="$2";   shift 2 ;;
        -h|--help) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)         EXTRA+=("$1"); shift ;;
    esac
done

if command -v sim_vehicle.py >/dev/null 2>&1; then
    SIM_VEHICLE=sim_vehicle.py
elif [[ -n "${ARDUPILOT_HOME:-}" && -x "${ARDUPILOT_HOME}/Tools/autotest/sim_vehicle.py" ]]; then
    SIM_VEHICLE="${ARDUPILOT_HOME}/Tools/autotest/sim_vehicle.py"
else
    cat >&2 <<'MSG'
error: sim_vehicle.py not found.

Either put it on PATH, or set ARDUPILOT_HOME to an ArduPilot checkout:

    git clone --recurse-submodules https://github.com/ArduPilot/ardupilot
    cd ardupilot && ./Tools/environment_install/install-prereqs-ubuntu.sh -y
    export ARDUPILOT_HOME=$PWD

The SITL tests are marked @pytest.mark.sitl and are skipped unless
DRONE_SITL=1, so the rest of the suite does not need any of this.
MSG
    exit 1
fi

echo "starting Copter SITL: frame=${FRAME} speedup=${SPEEDUP} out=${ENDPOINT}"
exec "${SIM_VEHICLE}" \
    -v ArduCopter \
    -f "${FRAME}" \
    --speedup "${SPEEDUP}" \
    --out "${ENDPOINT}" \
    --no-rebuild \
    "${EXTRA[@]}"
