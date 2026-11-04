#!/usr/bin/env bash
# Start the PACE policy server for LIBERO evaluation (PACE / model environment).
#
# Usage: bash examples/LIBERO/run_server.sh [CKPT] [POLICY_PORT] [--dry-run]
#   CKPT  .pt file, checkpoint directory or Hugging Face repo id (default: $CKPT or checkpoints/PACE-LIBERO)
#   POLICY_PORT  WebSocket port (default: $POLICY_PORT or 10093)
# Environment: PYTHON (default python), BIND_HOST bind address (default 0.0.0.0), USE_BF16 (default 1),
#   BASE_VLM, DINOV2_WEIGHTS, PACE_CACHE, PACE_ATTN_IMPLEMENTATION (see pace/model/checkpoint_io.py and
#   pace/model/progress/dinov2_assets.py).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CKPT="${CKPT:-checkpoints/PACE-LIBERO}"
POLICY_PORT="${POLICY_PORT:-10093}"
BIND_HOST="${BIND_HOST:-0.0.0.0}"
PYTHON="${PYTHON:-python}"
USE_BF16="${USE_BF16:-1}"
DRY_RUN="${DRY_RUN:-0}"

POSITIONAL=()
for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help) sed -n '2,10p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) POSITIONAL+=("${arg}") ;;
  esac
done
if [[ ${#POSITIONAL[@]} -ge 1 ]]; then CKPT="${POSITIONAL[0]}"; fi
if [[ ${#POSITIONAL[@]} -ge 2 ]]; then POLICY_PORT="${POSITIONAL[1]}"; fi

BF16_ARGS=()
if [[ "${USE_BF16}" == "1" ]]; then BF16_ARGS=(--use_bf16); fi

echo "EFFECTIVE: role=server benchmark=LIBERO ckpt=${CKPT} host=${BIND_HOST} port=${POLICY_PORT} bf16=${USE_BF16}" \
  "base_vlm=${BASE_VLM:-<config/HF default>} dinov2_weights=${DINOV2_WEIGHTS:-<cache>}" \
  "attn=${PACE_ATTN_IMPLEMENTATION:-<config>}"
if [[ "${DRY_RUN}" == "1" ]]; then exit 0; fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON}" -m deployment.model_server.server_policy \
  --ckpt_path "${CKPT}" --host "${BIND_HOST}" --port "${POLICY_PORT}" ${BF16_ARGS[@]+"${BF16_ARGS[@]}"}
