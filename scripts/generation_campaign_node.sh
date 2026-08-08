#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 )); then
  printf 'Usage: %s CAMPAIGN_CONFIG [run-campaign-worker options]\n' "$0" >&2
  exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  printf 'Campaign production workers must run inside a Slurm allocation.\n' >&2
  exit 2
fi
if [[ ! "${GENERATION_GIT_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]]; then
  printf 'GENERATION_GIT_COMMIT must contain the exact launch commit.\n' >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
CAMPAIGN_CONFIG="$1"
shift

GENERATION_CPU_VENV="${GENERATION_CPU_VENV:-${HOME}/.venvs/grainlegumes-generation-cpu}"
STORAGE_ROOT="${STORAGE_ROOT:-${HOME}/grainlegumes-generation/storage}"
if [[ "${GENERATION_CPU_VENV}" != /* || "${STORAGE_ROOT}" != /* ]]; then
  printf 'GENERATION_CPU_VENV and STORAGE_ROOT must be absolute paths.\n' >&2
  exit 2
fi
if [[ ! -x "${GENERATION_CPU_VENV}/bin/python" ]]; then
  printf 'Prepared CPU generation venv is missing: %s\n' "${GENERATION_CPU_VENV}" >&2
  exit 1
fi

WORKER_INDEX="${SLURM_ARRAY_TASK_ID:-}"
WORKER_COUNT="${CAMPAIGN_WORKER_COUNT:-${SLURM_ARRAY_TASK_COUNT:-}}"
if [[ ! "${WORKER_INDEX}" =~ ^(0|[1-9][0-9]*)$ ]]; then
  printf 'SLURM_ARRAY_TASK_ID must be a non-negative integer.\n' >&2
  exit 2
fi
if [[ ! "${WORKER_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'CAMPAIGN_WORKER_COUNT must be a positive integer.\n' >&2
  exit 2
fi

module load Python/3.10
module load Comsol/v6.4
source "${GENERATION_CPU_VENV}/bin/activate"
export STORAGE_ROOT
WORK_ROOT="${TMPDIR:-/tmp}/vp2-generation-${SLURM_JOB_ID}-${WORKER_INDEX}"
mkdir -p "${WORK_ROOT}"

cd "${REPOSITORY_ROOT}"
exec python -m src.generation.cli.cli_generation run-campaign-worker \
  "${CAMPAIGN_CONFIG}" \
  "$@" \
  --worker-index "${WORKER_INDEX}" \
  --worker-count "${WORKER_COUNT}" \
  --scheduler-kind slurm \
  --storage-root "${STORAGE_ROOT}" \
  --work-root "${WORK_ROOT}"
