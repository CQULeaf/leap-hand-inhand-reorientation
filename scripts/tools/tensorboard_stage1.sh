#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs/rl_games/leap_hand_reorient"
PORT="${1:-6006}"

cd "${PROJECT_ROOT}"
exec tensorboard --logdir "${LOG_DIR}" --port "${PORT}"
