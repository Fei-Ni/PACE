#!/usr/bin/env bash
# Download the LIBERO LeRobot datasets (IPEC-COMMUNITY on Hugging Face), write the modality.json files used by
# the PACE data loader and validate the result.
#
# Usage: bash examples/LIBERO/prepare_libero_lerobot_data.sh [--dry-run]
# Environment: DATA_ROOT (default datasets/LEROBOT_LIBERO_DATA), HF_CLI (default huggingface-cli),
#   PYTHON (default python), MAX_WORKERS (default 1), DOWNLOAD_RETRIES (default 20), RETRY_SLEEP_SECONDS (default 60)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATA_ROOT="${DATA_ROOT:-datasets/LEROBOT_LIBERO_DATA}"
HF_CLI="${HF_CLI:-huggingface-cli}"
PYTHON="${PYTHON:-python}"
MAX_WORKERS="${MAX_WORKERS:-1}"
DOWNLOAD_RETRIES="${DOWNLOAD_RETRIES:-20}"
RETRY_SLEEP_SECONDS="${RETRY_SLEEP_SECONDS:-60}"
DRY_RUN="${DRY_RUN:-0}"
for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help) sed -n '2,7p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "Unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

DATASETS=(
  "libero_object_no_noops_1.0.0_lerobot"
  "libero_goal_no_noops_1.0.0_lerobot"
  "libero_spatial_no_noops_1.0.0_lerobot"
  "libero_10_no_noops_1.0.0_lerobot"
)

echo "EFFECTIVE: role=data benchmark=LIBERO data_root=${DATA_ROOT} datasets=[${DATASETS[*]}] hf_cli=${HF_CLI}"
if [[ "${DRY_RUN}" == "1" ]]; then exit 0; fi

cd "${REPO_ROOT}"
command -v "${HF_CLI}" >/dev/null 2>&1 || { echo "Missing Hugging Face CLI: ${HF_CLI}" >&2; exit 1; }
mkdir -p "${DATA_ROOT}"
export HF_HUB_DISABLE_TELEMETRY=1

for dataset_name in "${DATASETS[@]}"; do
  repo_id="IPEC-COMMUNITY/${dataset_name}"
  local_dir="${DATA_ROOT}/${dataset_name}"
  echo
  echo "==== Downloading ${repo_id} -> ${local_dir} ===="
  attempt=1
  until "${HF_CLI}" download --repo-type dataset --max-workers "${MAX_WORKERS}" "${repo_id}" --local-dir "${local_dir}"; do
    if [[ "${attempt}" -ge "${DOWNLOAD_RETRIES}" ]]; then
      echo "Failed to download ${repo_id} after ${attempt} attempts." >&2
      exit 1
    fi
    echo "Download attempt ${attempt} for ${repo_id} failed; retrying in ${RETRY_SLEEP_SECONDS}s..." >&2
    attempt="$((attempt + 1))"
    sleep "${RETRY_SLEEP_SECONDS}"
  done
done

echo
echo "==== Writing modality.json ===="
"${PYTHON}" examples/LIBERO/generate_libero_modality_json.py --data-root "${DATA_ROOT}"

echo
echo "==== Validating datasets ===="
"${PYTHON}" examples/LIBERO/validate_libero_lerobot_data.py --data-root "${DATA_ROOT}"
