#!/usr/bin/env bash

set -euo pipefail

print_help() {
    cat <<'EOF'
Usage: bash scripts/deploy/reorient_z.sh [options]

Options:
  --checkpoint PATH         Policy checkpoint. Default: pretrained/leap_hand_reorient.pth
  --device DEVICE           Torch device. Default: cuda:0
  --port PORT               Serial port for the LEAP hand. Default: auto
  --baudrate N              Dynamixel baudrate. Default: 4000000
  --hz N                    Control frequency. Default: 30
  --kp N                    Motor P gain. Default: 800
  --kd N                    Motor D gain. Default: 200
  --curr-lim N              Motor current limit in mA. Default: 500
  --max-steps N             Optional deployment step limit. Default: 0
  --dry-run                 Run policy inference without connecting to motors
  --disable-torque-on-exit  Disable torque when the process exits
  --conda-env NAME          Conda env name. Default: env_isaaclab
  --python PYTHON_BIN       Override python executable inside the activated env
  -h, --help                Show this help message
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SOURCE_ROOT="${PROJECT_ROOT}/source/LEAP_Isaaclab"

CHECKPOINT="${PROJECT_ROOT}/pretrained/leap_hand_reorient.pth"
DEVICE="cuda:0"
PORT="auto"
BAUDRATE="4000000"
HZ="30"
KP="800"
KD="200"
CURR_LIM="500"
MAX_STEPS="0"
DRY_RUN=0
DISABLE_TORQUE_ON_EXIT=0
CONDA_ENV_NAME="env_isaaclab"
PYTHON_BIN=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --baudrate) BAUDRATE="$2"; shift 2 ;;
        --hz) HZ="$2"; shift 2 ;;
        --kp) KP="$2"; shift 2 ;;
        --kd) KD="$2"; shift 2 ;;
        --curr-lim) CURR_LIM="$2"; shift 2 ;;
        --max-steps) MAX_STEPS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --disable-torque-on-exit) DISABLE_TORQUE_ON_EXIT=1; shift ;;
        --conda-env) CONDA_ENV_NAME="$2"; shift 2 ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        -h|--help) print_help; exit 0 ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; print_help; exit 1 ;;
    esac
done

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "[ERROR] Checkpoint not found: ${CHECKPOINT}" >&2
    exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
    if [[ -f "/home/tools/anaconda3/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "/home/tools/anaconda3/etc/profile.d/conda.sh"
    else
        echo "[ERROR] Conda is not available in the current shell." >&2
        exit 1
    fi
else
    CONDA_BASE="$(conda info --base)"
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
fi

echo "[INFO] Activating conda env: ${CONDA_ENV_NAME}"
conda activate "${CONDA_ENV_NAME}"

if [[ -z "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="$(command -v python)"
fi

export PYTHONPATH="${SOURCE_ROOT}/LEAP_Isaaclab:${SOURCE_ROOT}:${PYTHONPATH:-}"

CMD=(
    "${PYTHON_BIN}"
    "${SOURCE_ROOT}/LEAP_Isaaclab/deployment_scripts/reorient_z.py"
    --checkpoint "${CHECKPOINT}"
    --device "${DEVICE}"
    --port "${PORT}"
    --baudrate "${BAUDRATE}"
    --hz "${HZ}"
    --kp "${KP}"
    --kd "${KD}"
    --curr-lim "${CURR_LIM}"
    --max-steps "${MAX_STEPS}"
)

if [[ "${DRY_RUN}" -eq 1 ]]; then
    CMD+=(--dry-run)
fi
if [[ "${DISABLE_TORQUE_ON_EXIT}" -eq 1 ]]; then
    CMD+=(--disable-torque-on-exit)
fi

echo "[INFO] Launch command:"
printf '  %q' "${CMD[@]}"
printf '\n'

cd "${PROJECT_ROOT}"
exec "${CMD[@]}"
