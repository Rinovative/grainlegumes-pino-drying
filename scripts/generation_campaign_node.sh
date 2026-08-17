#!/bin/bash -l
set -euo pipefail

if (( $# != 5 )); then
  printf 'Usage: %s REPOSITORY CAMPAIGN_RUN_ID BATCH_NAME CASE_INDEX CORES_PER_CASE\n' "$0" >&2
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

CAMPAIGN_RUN_ID="$2"
BATCH_NAME="$3"
CASE_INDEX="$4"
CORES_PER_CASE="$5"
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

printf -v CASE_ID 'case_%04d' "${CASE_INDEX}"
CASE_NODE="${SLURMD_NODENAME:-${HOSTNAME:-unavailable}}"
CASE_STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || printf unavailable)"
printf 'CASE START campaign_run_id=%s batch=%s case=%s case_index=%s job=%s node=%s cores=%s started_at=%s\n' \
  "${CAMPAIGN_RUN_ID}" "${BATCH_NAME}" "${CASE_ID}" "${CASE_INDEX}" \
  "${SLURM_JOB_ID}" "${CASE_NODE}" "${CORES_PER_CASE}" "${CASE_STARTED_AT}"

REPOSITORY_ROOT="$1"
PREREQUISITE_HELPER="${REPOSITORY_ROOT}/scripts/generation_prerequisites.sh"
if [[ "${REPOSITORY_ROOT}" != /* || "${REPOSITORY_ROOT}" == / ]]; then
  printf 'CPU compute-node prerequisite failed: explicit canonical CPU repository required: %s (Slurm script: %s).\n' \
    "${REPOSITORY_ROOT}" "${BASH_SOURCE[0]}" >&2
  exit 1
fi
if [[ ! -f "${PREREQUISITE_HELPER}" || -L "${PREREQUISITE_HELPER}" || ! -r "${PREREQUISITE_HELPER}" ]]; then
  printf 'CPU compute-node prerequisite failed: repository helper missing or unreadable: scripts/generation_prerequisites.sh (canonical CPU checkout: %s; Slurm script: %s).\n' \
    "${REPOSITORY_ROOT}" "${BASH_SOURCE[0]}" >&2
  exit 1
fi
/bin/bash "${PREREQUISITE_HELPER}" validate-worker-repository \
  "${REPOSITORY_ROOT}" "${GENERATION_GIT_COMMIT:-}" "${BASH_SOURCE[0]}"
# shellcheck source=generation_prerequisites.sh
source "${PREREQUISITE_HELPER}"
GENERATION_CPU_VENV="${GENERATION_CPU_VENV:-}"
STORAGE_ROOT="${STORAGE_ROOT:-}"
if [[ "${GENERATION_CPU_VENV}" != /* || "${STORAGE_ROOT}" != /* ]]; then
  printf 'GENERATION_CPU_VENV and STORAGE_ROOT must be explicit absolute paths.\n' >&2
  exit 2
fi
for variable_name in GENERATION_PYTHON_MODULE GENERATION_COMSOL_MODULE \
  GENERATION_PYTHON_EXECUTABLE GENERATION_COMSOL_EXECUTABLE; do
  value="${!variable_name:-}"
  if [[ -z "${value}" || "${value}" == *$'\n'* || "${value}" == *$'\r'* ]]; then
    printf '%s must be supplied by the resolved campaign execution plan.\n' "${variable_name}" >&2
    exit 2
  fi
done

COMPUTE_DOMAIN="CPU compute-node"
generation_require_command "${COMPUTE_DOMAIN}" module "compute bootstrap"
generation_require_command "${COMPUTE_DOMAIN}" mktemp "case scratch creation"
if ! module load "${GENERATION_PYTHON_MODULE}"; then
  generation_prerequisite_failed \
    "${COMPUTE_DOMAIN}" "Python module ${GENERATION_PYTHON_MODULE}" "compute"
fi
if ! module load "${GENERATION_COMSOL_MODULE}"; then
  generation_prerequisite_failed \
    "${COMPUTE_DOMAIN}" "COMSOL module ${GENERATION_COMSOL_MODULE}" "compute"
fi
generation_require_command \
  "${COMPUTE_DOMAIN}" "${GENERATION_PYTHON_EXECUTABLE}" "case materialization and HDF5 admission"
generation_run_check \
  "${COMPUTE_DOMAIN}" "python-version:${GENERATION_PYTHON_EXECUTABLE}" "compute" \
  "${GENERATION_PYTHON_EXECUTABLE}" --version
generation_require_command \
  "${COMPUTE_DOMAIN}" "${GENERATION_COMSOL_EXECUTABLE}" "compute"
generation_run_check \
  "${COMPUTE_DOMAIN}" "comsol-version:${GENERATION_COMSOL_EXECUTABLE}" "compute" \
  "${GENERATION_COMSOL_EXECUTABLE}" -version
cd "${REPOSITORY_ROOT}"
generation_validate_cpu_venv \
  "${COMPUTE_DOMAIN}" "${GENERATION_CPU_VENV}" \
  "case materialization and HDF5 conversion/admission"

export STORAGE_ROOT
SCRATCH_PARENT="${TMPDIR:-/tmp}"
if [[ "${SCRATCH_PARENT}" != /* || ! -d "${SCRATCH_PARENT}" || ! -w "${SCRATCH_PARENT}" ]]; then
  generation_prerequisite_missing \
    "${COMPUTE_DOMAIN}" "writable scratch parent: ${SCRATCH_PARENT}" "compute"
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
