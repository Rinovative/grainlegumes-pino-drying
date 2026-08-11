#!/bin/bash -l
set -euo pipefail

if (( $# != 2 && $# != 4 )); then
  printf 'Usage: %s BENCHMARK_RUN_ID prepare | BENCHMARK_RUN_ID measure VARIANT_ID REPETITION\n' "$0" >&2
  exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  printf 'Benchmark workers must run inside a Slurm allocation.\n' >&2
  exit 2
fi
if [[ ! "${GENERATION_GIT_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]]; then
  printf 'GENERATION_GIT_COMMIT must contain the exact launch commit.\n' >&2
  exit 2
fi
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  printf 'Core benchmark jobs must not use Slurm arrays.\n' >&2
  exit 2
fi

RUN_ID="$1"
MODE="$2"
VARIANT_ID="${3:-}"
REPETITION="${4:-}"
if [[ ! "${RUN_ID}" =~ ^core_scaling_transient__[0-9a-f]{16}$ ]]; then
  printf 'Benchmark run ID is malformed: %s\n' "${RUN_ID}" >&2
  exit 2
fi
if [[ "${GENERATION_BENCHMARK_RUN_ID:-}" != "${RUN_ID}" ]]; then
  printf 'GENERATION_BENCHMARK_RUN_ID does not match the requested run.\n' >&2
  exit 2
fi
case "${MODE}" in
  prepare)
    [[ -z "${VARIANT_ID}" && -z "${REPETITION}" ]] || {
      printf 'Preparation does not accept a variant or repetition.\n' >&2
      exit 2
    }
    if [[ "${SLURM_CPUS_PER_TASK:-}" != 1 ]]; then
      printf 'Benchmark preparation requires exactly one allocated CPU.\n' >&2
      exit 2
    fi
    TASK_ID="prepare"
    ;;
  measure)
    [[ "${VARIANT_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || {
      printf 'Measured benchmark requires one safe variant ID.\n' >&2
      exit 2
    }
    [[ "${REPETITION}" =~ ^[1-9][0-9]*$ ]] || {
      printf 'Measured benchmark requires one positive repetition argument.\n' >&2
      exit 2
    }
    [[ "${SLURM_CPUS_PER_TASK:-}" =~ ^[1-9][0-9]*$ ]] || {
      printf 'Measured benchmark requires a positive cpus-per-task allocation.\n' >&2
      exit 2
    }
    TASK_ID="${REPETITION}"
    ;;
  *)
    printf 'Unsupported benchmark worker mode: %s\n' "${MODE}" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
# shellcheck source=generation_prerequisites.sh
source "${SCRIPT_DIR}/generation_prerequisites.sh"
COMPUTE_DOMAIN="CPU compute-node"
generation_require_command "${COMPUTE_DOMAIN}" git "benchmark checkout validation"
[[ -d "${REPOSITORY_ROOT}/.git" ]] || {
  printf 'Benchmark repository checkout is missing: %s\n' "${REPOSITORY_ROOT}" >&2
  exit 1
}
ACTUAL_COMMIT="$(git -C "${REPOSITORY_ROOT}" rev-parse HEAD)"
if [[ "${ACTUAL_COMMIT}" != "${GENERATION_GIT_COMMIT}" ]]; then
  printf 'Benchmark checkout %s does not match launch commit %s.\n' \
    "${ACTUAL_COMMIT}" "${GENERATION_GIT_COMMIT}" >&2
  exit 1
fi
if [[ -n "$(git -C "${REPOSITORY_ROOT}" status --porcelain)" ]]; then
  printf 'Benchmark repository checkout must be clean.\n' >&2
  exit 1
fi
GENERATION_CPU_VENV="${GENERATION_CPU_VENV:-}"
STORAGE_ROOT="${STORAGE_ROOT:-}"
if [[ "${GENERATION_CPU_VENV}" != /* || "${STORAGE_ROOT}" != /* ]]; then
  printf 'GENERATION_CPU_VENV and STORAGE_ROOT must be explicit absolute paths.\n' >&2
  exit 2
fi
if [[ ! -x "${GENERATION_CPU_VENV}/bin/python" ]]; then
  generation_prerequisite_missing \
    "${COMPUTE_DOMAIN}" "Generation CPU venv: ${GENERATION_CPU_VENV}" "benchmark compute"
fi
for variable_name in GENERATION_PYTHON_MODULE GENERATION_COMSOL_MODULE \
  GENERATION_PYTHON_EXECUTABLE GENERATION_COMSOL_EXECUTABLE; do
  value="${!variable_name:-}"
  if [[ -z "${value}" || "${value}" == *$'\n'* || "${value}" == *$'\r'* ]]; then
    printf '%s must be supplied by the resolved benchmark plan.\n' "${variable_name}" >&2
    exit 2
  fi
done

generation_require_command "${COMPUTE_DOMAIN}" module "benchmark compute bootstrap"
generation_require_command "${COMPUTE_DOMAIN}" mktemp "benchmark scratch creation"
if ! module load "${GENERATION_PYTHON_MODULE}"; then
  generation_prerequisite_failed \
    "${COMPUTE_DOMAIN}" "Python module ${GENERATION_PYTHON_MODULE}" "benchmark compute"
fi
if ! module load "${GENERATION_COMSOL_MODULE}"; then
  generation_prerequisite_failed \
    "${COMPUTE_DOMAIN}" "COMSOL module ${GENERATION_COMSOL_MODULE}" "benchmark compute"
fi
generation_require_command \
  "${COMPUTE_DOMAIN}" "${GENERATION_PYTHON_EXECUTABLE}" "benchmark preparation and admission"
generation_run_check \
  "${COMPUTE_DOMAIN}" "python-version:${GENERATION_PYTHON_EXECUTABLE}" "benchmark compute" \
  "${GENERATION_PYTHON_EXECUTABLE}" --version
generation_require_command \
  "${COMPUTE_DOMAIN}" "${GENERATION_COMSOL_EXECUTABLE}" "benchmark compute"
generation_run_check \
  "${COMPUTE_DOMAIN}" "comsol-version:${GENERATION_COMSOL_EXECUTABLE}" "benchmark compute" \
  "${GENERATION_COMSOL_EXECUTABLE}" -version
source "${GENERATION_CPU_VENV}/bin/activate"
if ! "${GENERATION_CPU_VENV}/bin/python" -c \
  'import h5py, numpy, scipy, yaml; import src.generation.cli.cli_generation'; then
  generation_prerequisite_failed \
    "${COMPUTE_DOMAIN}" "Generation CPU venv package/imports" \
    "benchmark preparation and HDF5 conversion/admission"
fi

export STORAGE_ROOT
SCRATCH_PARENT="${TMPDIR:-/tmp}"
if [[ "${SCRATCH_PARENT}" != /* || ! -d "${SCRATCH_PARENT}" || ! -w "${SCRATCH_PARENT}" ]]; then
  generation_prerequisite_missing \
    "${COMPUTE_DOMAIN}" "writable scratch parent: ${SCRATCH_PARENT}" "benchmark compute"
fi
WORK_ROOT="$(mktemp -d "${SCRATCH_PARENT%/}/vp2-benchmark-${SLURM_JOB_ID}-${TASK_ID}.XXXXXX")"
MARKER_READY=false
CHILD_PID=""
INTERRUPTION_SIGNAL=""

cleanup_worker() {
  if [[ "${MARKER_READY}" != true ]]; then
    return 0
  fi
  "${GENERATION_CPU_VENV}/bin/python" -m src.generation.cli.cli_generation \
    cleanup-worker-workspace "${WORK_ROOT}" \
    --campaign-run-id "${RUN_ID}" \
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
  cleanup_worker
  cleanup_status="$?"
  set -e
  if (( cleanup_status != 0 )); then
    printf 'Owned benchmark scratch cleanup failed: %s\n' "${WORK_ROOT}" >&2
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
  --campaign-run-id "${RUN_ID}" \
  --storage-root "${STORAGE_ROOT}" >/dev/null
MARKER_READY=true

if [[ "${MODE}" == prepare ]]; then
  COMMAND=(
    "${GENERATION_CPU_VENV}/bin/python" -m src.generation.cli.cli_generation
    prepare-core-benchmark-case "${RUN_ID}"
    --storage-root "${STORAGE_ROOT}"
    --work-root "${WORK_ROOT}"
  )
else
  COMMAND=(
    "${GENERATION_CPU_VENV}/bin/python" -m src.generation.cli.cli_generation
    run-core-benchmark-repetition "${RUN_ID}" "${VARIANT_ID}"
    "${REPETITION}"
    --storage-root "${STORAGE_ROOT}"
    --work-root "${WORK_ROOT}"
  )
fi

"${COMMAND[@]}" &
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
