#!/usr/bin/env bash
# Download the BridgeData V2 LeRobot conversion (IPEC-COMMUNITY/bridge_orig_lerobot on Hugging Face), write the
# modality.json used by the PACE data loader and validate a sample of episodes.
#
# Usage: bash examples/SimplerEnv/prepare_oxe_lerobot_data.sh [--dry-run]
# Environment: DATA_ROOT (default datasets/OXE_LEROBOT_DATASET), HF_CLI (default huggingface-cli),
#   PYTHON (default python), MAX_WORKERS (default 1), DOWNLOAD_RETRIES (default 20), RETRY_SLEEP_SECONDS (default 60),
#   SAMPLE_EPISODES (default 8)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATA_ROOT="${DATA_ROOT:-datasets/OXE_LEROBOT_DATASET}"
HF_CLI="${HF_CLI:-huggingface-cli}"
PYTHON="${PYTHON:-python}"
MAX_WORKERS="${MAX_WORKERS:-1}"
DOWNLOAD_RETRIES="${DOWNLOAD_RETRIES:-20}"
RETRY_SLEEP_SECONDS="${RETRY_SLEEP_SECONDS:-60}"
SAMPLE_EPISODES="${SAMPLE_EPISODES:-8}"
DRY_RUN="${DRY_RUN:-0}"
for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help) sed -n '2,8p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "Unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

DATASET_NAME="bridge_orig_1.0.0_lerobot"
REPO_ID="IPEC-COMMUNITY/bridge_orig_lerobot"

echo "EFFECTIVE: role=data benchmark=Bridge data_root=${DATA_ROOT} dataset=${DATASET_NAME} repo=${REPO_ID}"
if [[ "${DRY_RUN}" == "1" ]]; then exit 0; fi

cd "${REPO_ROOT}"
command -v "${HF_CLI}" >/dev/null 2>&1 || { echo "Missing Hugging Face CLI: ${HF_CLI}" >&2; exit 1; }
mkdir -p "${DATA_ROOT}"
export HF_HUB_DISABLE_TELEMETRY=1

local_dir="${DATA_ROOT}/${DATASET_NAME}"
echo "==== Downloading ${REPO_ID} -> ${local_dir} ===="
attempt=1
until "${HF_CLI}" download --repo-type dataset --max-workers "${MAX_WORKERS}" "${REPO_ID}" --local-dir "${local_dir}"; do
  if [[ "${attempt}" -ge "${DOWNLOAD_RETRIES}" ]]; then
    echo "Failed to download ${REPO_ID} after ${attempt} attempts." >&2
    exit 1
  fi
  echo "Download attempt ${attempt} failed; retrying in ${RETRY_SLEEP_SECONDS}s..." >&2
  attempt="$((attempt + 1))"
  sleep "${RETRY_SLEEP_SECONDS}"
done

echo "==== Writing modality.json ===="
"${PYTHON}" examples/SimplerEnv/generate_oxe_modality_json.py --data-root "${DATA_ROOT}" --datasets "${DATASET_NAME}"

echo "==== Validating ${SAMPLE_EPISODES} sample episodes ===="
"${PYTHON}" examples/SimplerEnv/validate_oxe_lerobot_data.py \
  --data-root "${DATA_ROOT}" --sample-episodes "${SAMPLE_EPISODES}" --datasets "${DATASET_NAME}"
