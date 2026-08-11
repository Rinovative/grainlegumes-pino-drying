#!/bin/bash -l
set -euo pipefail

if (( $# != 4 )); then
  printf 'Usage: %s CAMPAIGN_RUN_ID BATCH_NAME CASE_INDEX CORES_PER_CASE\n' "$0" >&2
  exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  printf 'Campaign cases must run inside a Slurm allocation.\n' >&2
  exit 2
fi
if [[ ! "${GENERATION_GIT_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]]; then
  printf 'GENERATION_GIT_COMMIT must contain the exact launch commit.\n' >&2
  exit 2
fi

CAMPAIGN_RUN_ID="$1"
BATCH_NAME="$2"
CASE_INDEX="$3"
CORES_PER_CASE="$4"
if [[ ! "${CAMPAIGN_RUN_ID}" =~ ^[A-Za-z0-9._-]+__[0-9a-f]{16}$ || "${GENERATION_CAMPAIGN_RUN_ID:-}" != "${CAMPAIGN_RUN_ID}" ]]; then
  printf 'Campaign run identity is missing, malformed, or inconsistent.\n' >&2
  exit 2
fi
if [[ ! "${BATCH_NAME}" =~ ^[A-Za-z0-9._-]+$ || ! "${CASE_INDEX}" =~ ^[1-9][0-9]*$ || ! "${CORES_PER_CASE}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'Campaign batch, case, or core arguments are malformed.\n' >&2
  exit 2
fi
if [[ ! "${SLURM_CPUS_PER_TASK:-}" =~ ^[1-9][0-9]*$ || "${SLURM_CPUS_PER_TASK}" -ne "${CORES_PER_CASE}" ]]; then
  printf 'Slurm cpus-per-task (%s) must equal cores_per_case (%s).\n' \
    "${SLURM_CPUS_PER_TASK:-unset}" "${CORES_PER_CASE}" >&2
  exit 2
fi
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  printf 'Campaign cases must not run as Slurm array tasks.\n' >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
GENERATION_CPU_VENV="${GENERATION_CPU_VENV:-}"
STORAGE_ROOT="${STORAGE_ROOT:-}"
if [[ "${GENERATION_CPU_VENV}" != /* || "${STORAGE_ROOT}" != /* ]]; then
  printf 'GENERATION_CPU_VENV and STORAGE_ROOT must be explicit absolute paths.\n' >&2
  exit 2
fi
if [[ ! -x "${GENERATION_CPU_VENV}/bin/python" ]]; then
  printf 'Prepared CPU generation venv is missing: %s\n' "${GENERATION_CPU_VENV}" >&2
  exit 1
fi
for variable_name in GENERATION_PYTHON_MODULE GENERATION_COMSOL_MODULE \
  GENERATION_PYTHON_EXECUTABLE GENERATION_COMSOL_EXECUTABLE; do
  value="${!variable_name:-}"
  if [[ -z "${value}" || "${value}" == *$'\n'* || "${value}" == *$'\r'* ]]; then
    printf '%s must be supplied by the resolved campaign execution plan.\n' "${variable_name}" >&2
    exit 2
  fi
done

module load "${GENERATION_PYTHON_MODULE}"
module load "${GENERATION_COMSOL_MODULE}"
command -v "${GENERATION_PYTHON_EXECUTABLE}"
"${GENERATION_PYTHON_EXECUTABLE}" --version
command -v "${GENERATION_COMSOL_EXECUTABLE}"
"${GENERATION_COMSOL_EXECUTABLE}" -version
command -v rsync
rsync --version
source "${GENERATION_CPU_VENV}/bin/activate"
"${GENERATION_CPU_VENV}/bin/python" -c 'import h5py, numpy, scipy, yaml; import src.generation.cli.cli_generation'

export STORAGE_ROOT
SCRATCH_PARENT="${TMPDIR:-/tmp}"
if [[ "${SCRATCH_PARENT}" != /* || ! -d "${SCRATCH_PARENT}" || ! -w "${SCRATCH_PARENT}" ]]; then
  printf 'Slurm scratch parent is unavailable: %s\n' "${SCRATCH_PARENT}" >&2
  exit 1
fi
WORK_ROOT="$(mktemp -d "${SCRATCH_PARENT%/}/vp2-generation-${SLURM_JOB_ID}.XXXXXX")"
MARKER_READY=false
CHILD_PID=""
INTERRUPTION_SIGNAL=""

record_interruption() {
  local status="$1"
  if [[ -z "${INTERRUPTION_SIGNAL}" ]]; then
    return 0
  fi
  "${GENERATION_CPU_VENV}/bin/python" -m src.generation.cli.cli_generation \
    record-worker-interruption "${CAMPAIGN_RUN_ID}" \
    --signal "${INTERRUPTION_SIGNAL}" \
    --exit-code "${status}" \
    --storage-root "${STORAGE_ROOT}" \
    || printf 'Could not persist worker interruption receipt.\n' >&2
}

cleanup_worker() {
  if [[ "${MARKER_READY}" != true ]]; then
    return 0
  fi
  "${GENERATION_CPU_VENV}/bin/python" -m src.generation.cli.cli_generation \
    cleanup-worker-workspace "${WORK_ROOT}" \
    --campaign-run-id "${CAMPAIGN_RUN_ID}" \
    --storage-root "${STORAGE_ROOT}"
}

handle_signal() {
  INTERRUPTION_SIGNAL="$1"
  if [[ -n "${CHILD_PID}" ]]; then
    kill -TERM "${CHILD_PID}" 2>/dev/null || true
  fi
}

on_exit() {
  local status="$?"
  local cleanup_status=0
  trap - EXIT INT TERM
  set +e
  record_interruption "${status}"
  cleanup_worker
  cleanup_status="$?"
  set -e
  if (( cleanup_status != 0 )); then
    printf 'Owned worker scratch cleanup failed: %s\n' "${WORK_ROOT}" >&2
    if (( status == 0 )); then
      status="${cleanup_status}"
    fi
  fi
  exit "${status}"
}

trap 'handle_signal INT' INT
trap 'handle_signal TERM' TERM
trap on_exit EXIT

cd "${REPOSITORY_ROOT}"
"${GENERATION_CPU_VENV}/bin/python" -m src.generation.cli.cli_generation \
  initialize-worker-workspace "${WORK_ROOT}" \
  --campaign-run-id "${CAMPAIGN_RUN_ID}" \
  --storage-root "${STORAGE_ROOT}" >/dev/null
MARKER_READY=true

"${GENERATION_CPU_VENV}/bin/python" -m src.generation.cli.cli_generation \
  run-campaign-case "${CAMPAIGN_RUN_ID}" "${BATCH_NAME}" "${CASE_INDEX}" \
  --storage-root "${STORAGE_ROOT}" \
  --work-root "${WORK_ROOT}" &
CHILD_PID="$!"
set +e
wait "${CHILD_PID}"
WORKER_STATUS="$?"
if [[ -n "${INTERRUPTION_SIGNAL}" ]]; then
  wait "${CHILD_PID}" 2>/dev/null
  case "${INTERRUPTION_SIGNAL}" in
    INT) WORKER_STATUS=130 ;;
    TERM) WORKER_STATUS=143 ;;
  esac
fi
set -e
CHILD_PID=""
exit "${WORKER_STATUS}"
