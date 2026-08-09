#!/bin/bash -l
set -euo pipefail

if (( $# < 1 )); then
  printf 'Usage: %s CAMPAIGN_CONFIG [run-campaign-worker options]\n' "$0" >&2
  exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  printf 'Campaign workers must run inside a Slurm allocation.\n' >&2
  exit 2
fi
if [[ ! "${GENERATION_GIT_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]]; then
  printf 'GENERATION_GIT_COMMIT must contain the exact launch commit.\n' >&2
  exit 2
fi
if [[ ! "${GENERATION_CAMPAIGN_RUN_ID:-}" =~ ^[A-Za-z0-9._-]+__[0-9a-f]{16}$ ]]; then
  printf 'GENERATION_CAMPAIGN_RUN_ID is missing or malformed.\n' >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
CAMPAIGN_CONFIG="$1"
shift
WORKER_ARGUMENTS=("$@")

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

CASES_PER_NODE=""
CORES_PER_CASE=""
for (( index=0; index<${#WORKER_ARGUMENTS[@]}; index++ )); do
  case "${WORKER_ARGUMENTS[index]}" in
    --cases-per-node)
      CASES_PER_NODE="${WORKER_ARGUMENTS[index+1]:-}"
      ;;
    --cores-per-case)
      CORES_PER_CASE="${WORKER_ARGUMENTS[index+1]:-}"
      ;;
  esac
done
if [[ ! "${CASES_PER_NODE}" =~ ^[1-9][0-9]*$ || ! "${CORES_PER_CASE}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'Worker resource arguments are incomplete or malformed.\n' >&2
  exit 2
fi
EXPECTED_CPUS=$((CASES_PER_NODE * CORES_PER_CASE))
if [[ ! "${SLURM_CPUS_PER_TASK:-}" =~ ^[1-9][0-9]*$ || "${SLURM_CPUS_PER_TASK}" -ne "${EXPECTED_CPUS}" ]]; then
  printf 'Slurm cpus-per-task (%s) does not match cases_per_node*cores_per_case (%s).\n'     "${SLURM_CPUS_PER_TASK:-unset}" "${EXPECTED_CPUS}" >&2
  exit 2
fi

module load Python/3.10
module load Comsol/v6.4
command -v python3
python3 --version
command -v comsol
COMSOL_VERSION_OUTPUT="$(comsol -version 2>&1)"
printf '%s\n' "${COMSOL_VERSION_OUTPUT}"
[[ "${COMSOL_VERSION_OUTPUT}" == *"6.4"* ]] || { printf 'COMSOL 6.4 required.\n' >&2; exit 1; }
command -v rsync
rsync --version
command -v srun
source "${GENERATION_CPU_VENV}/bin/activate"
python -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 10) else f"Python 3.10 required, got {sys.version}")'
python -c 'import h5py, numpy, scipy, yaml; import src.generation.cli.cli_generation'

export STORAGE_ROOT
SCRATCH_PARENT="${TMPDIR:-/tmp}"
if [[ "${SCRATCH_PARENT}" != /* || ! -d "${SCRATCH_PARENT}" || ! -w "${SCRATCH_PARENT}" ]]; then
  printf 'Slurm scratch parent is unavailable: %s\n' "${SCRATCH_PARENT}" >&2
  exit 1
fi
WORK_ROOT="$(mktemp -d "${SCRATCH_PARENT%/}/vp2-generation-${SLURM_JOB_ID}-${WORKER_INDEX}.XXXXXX")"
MARKER_READY=false
CHILD_PID=""
INTERRUPTION_SIGNAL=""

record_interruption() {
  local status="$1"
  if [[ -z "${INTERRUPTION_SIGNAL}" ]]; then
    return 0
  fi
  "${GENERATION_CPU_VENV}/bin/python" -m src.generation.cli.cli_generation     record-worker-interruption "${GENERATION_CAMPAIGN_RUN_ID}"     --signal "${INTERRUPTION_SIGNAL}"     --exit-code "${status}"     --storage-root "${STORAGE_ROOT}"     || printf 'Could not persist worker interruption receipt.\n' >&2
}

cleanup_worker() {
  if [[ "${MARKER_READY}" != true ]]; then
    return 0
  fi
  "${GENERATION_CPU_VENV}/bin/python" -m src.generation.cli.cli_generation     cleanup-worker-workspace "${WORK_ROOT}"     --campaign-run-id "${GENERATION_CAMPAIGN_RUN_ID}"     --storage-root "${STORAGE_ROOT}"
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
"${GENERATION_CPU_VENV}/bin/python" -m src.generation.cli.cli_generation   initialize-worker-workspace "${WORK_ROOT}"   --campaign-run-id "${GENERATION_CAMPAIGN_RUN_ID}"   --storage-root "${STORAGE_ROOT}" >/dev/null
MARKER_READY=true

"${GENERATION_CPU_VENV}/bin/python" -m src.generation.cli.cli_generation   run-campaign-worker "${CAMPAIGN_CONFIG}"   "${WORKER_ARGUMENTS[@]}"   --worker-index "${WORKER_INDEX}"   --worker-count "${WORKER_COUNT}"   --scheduler-kind slurm   --storage-root "${STORAGE_ROOT}"   --work-root "${WORK_ROOT}" &
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
