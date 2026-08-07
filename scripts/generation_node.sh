#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 )); then
  printf 'Usage: %s GENERATION_CONFIG [run-node options]\n' "$0" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
CONFIG_PATH="$1"
shift

if [[ -z "${STORAGE_ROOT:-}" ]]; then
  STORAGE_ROOT="$(cd "${REPOSITORY_ROOT}/.." && pwd -P)/storage"
  export STORAGE_ROOT
fi

WORKER_INDEX="${SLURM_ARRAY_TASK_ID:-${SLURM_PROCID:-${NODE_WORKER_INDEX:-}}}"
WORKER_COUNT="${NODE_WORKER_COUNT:-${SLURM_ARRAY_TASK_COUNT:-${SLURM_NTASKS:-}}}"
if [[ ! "${WORKER_INDEX}" =~ ^(0|[1-9][0-9]*)$ ]]; then
  printf 'A scheduler array/task index or NODE_WORKER_INDEX must be non-negative.\n' >&2
  exit 2
fi
if [[ ! "${WORKER_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'A scheduler array/task count or NODE_WORKER_COUNT must be positive.\n' >&2
  exit 2
fi

cd "${REPOSITORY_ROOT}"
exec python -m src.generation.cli.cli_generation run-node \
  "${CONFIG_PATH}" \
  "$@" \
  --worker-index "${WORKER_INDEX}" \
  --worker-count "${WORKER_COUNT}" \
  --scheduler-kind slurm
