#!/usr/bin/env bash

set -euo pipefail

print_help() {
    cat <<'EOF'
Usage: bash scripts/eval/play_stage1.sh [options] [-- extra Isaac Lab args]

Options:
  --task TASK             Gym task name. Default: Isaac-Reorient-Cube-Leap
  --num-envs N            Number of play environments. Default: 1
  --device DEVICE         Device passed to Isaac Lab. Default: cuda:0
  --conda-env NAME        Conda env name. Default: env_isaaclab
  --checkpoint PATH       Explicit checkpoint path
  --use-last-checkpoint   Load the latest checkpoint instead of the best checkpoint
  --real-time             Run in real-time if possible
  --stochastic            Sample actions instead of deterministic mean actions
  --max-steps N           Stop after N policy steps. Default: 0, run until closed
  --headless              Launch Isaac Sim headless
  --physx-smoke-buffers   Use smaller PhysX GPU buffers for smoke tests on 8GB GPUs
  --video                 Record a play video
  --video-length N        Video length. Default: 300
  -h, --help              Show this help message
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SOURCE_ROOT="${PROJECT_ROOT}/source/LEAP_Isaaclab"
PRETRAINED_CHECKPOINT="${PROJECT_ROOT}/pretrained/leap_hand_reorient.pth"

TASK="Isaac-Reorient-Cube-Leap"
NUM_ENVS="1"
DEVICE="cuda:0"
CONDA_ENV_NAME="env_isaaclab"
CHECKPOINT=""
USE_LAST_CHECKPOINT=0
REAL_TIME=0
STOCHASTIC=0
MAX_STEPS="0"
HEADLESS=0
PHYSX_SMOKE_BUFFERS=0
VIDEO=0
VIDEO_LENGTH="300"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task) TASK="$2"; shift 2 ;;
        --num-envs) NUM_ENVS="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --conda-env) CONDA_ENV_NAME="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --use-last-checkpoint) USE_LAST_CHECKPOINT=1; shift ;;
        --real-time) REAL_TIME=1; shift ;;
        --stochastic) STOCHASTIC=1; shift ;;
        --max-steps) MAX_STEPS="$2"; shift 2 ;;
        --headless) HEADLESS=1; shift ;;
        --physx-smoke-buffers) PHYSX_SMOKE_BUFFERS=1; shift ;;
        --video) VIDEO=1; shift ;;
        --video-length) VIDEO_LENGTH="$2"; shift 2 ;;
        --) shift; EXTRA_ARGS+=("$@"); break ;;
        -h|--help) print_help; exit 0 ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; print_help; exit 1 ;;
    esac
done

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

export PYTHONPATH="${SOURCE_ROOT}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${PROJECT_ROOT}/.mplconfig"
mkdir -p "${MPLCONFIGDIR}"

CMD=(
    python
    scripts/internal/rl_games_play.py
    --task "${TASK}"
    --num_envs "${NUM_ENVS}"
    --device "${DEVICE}"
    --max_steps "${MAX_STEPS}"
)

if [[ -z "${CHECKPOINT}" && "${USE_LAST_CHECKPOINT}" -eq 0 && -f "${PRETRAINED_CHECKPOINT}" ]]; then
    CHECKPOINT="${PRETRAINED_CHECKPOINT}"
fi

if [[ -n "${CHECKPOINT}" ]]; then
    CMD+=(--checkpoint "${CHECKPOINT}")
elif [[ "${USE_LAST_CHECKPOINT}" -eq 1 ]]; then
    CMD+=(--use_last_checkpoint)
fi

if [[ "${REAL_TIME}" -eq 1 ]]; then
    CMD+=(--real-time)
fi
if [[ "${STOCHASTIC}" -eq 1 ]]; then
    CMD+=(--stochastic)
fi
if [[ "${VIDEO}" -eq 1 ]]; then
    CMD+=(--video --video_length "${VIDEO_LENGTH}")
fi
if [[ "${HEADLESS}" -eq 1 ]]; then
    CMD+=(--headless)
fi
if [[ "${PHYSX_SMOKE_BUFFERS}" -eq 1 ]]; then
    CMD+=(--physx_gpu_contact_count 1048576 --physx_gpu_patch_count 1048576)
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[INFO] Launch command:"
if [[ -n "${CHECKPOINT}" ]]; then
    echo "[INFO] Checkpoint: ${CHECKPOINT}"
fi
printf '  %q' "${CMD[@]}"
printf '\n'

cd "${PROJECT_ROOT}"
exec "${CMD[@]}"
