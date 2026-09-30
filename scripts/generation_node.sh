#!/bin/bash -l
set -Eeuo pipefail

if (( $# < 2 )); then
  printf 'Usage: %s REPOSITORY cli|benchmark-preflight|campaign-case|benchmark-case|smoke [ARGS...]\n' "$0" >&2
  exit 2
fi

REPOSITORY_ROOT="$1"
MODE="$2"
shift 2
PREREQUISITE_HELPER="${REPOSITORY_ROOT}/scripts/generation_prerequisites.sh"
PYTHON_MODULE=src.generation.cli.cli_generation

if [[ ! -f "${PREREQUISITE_HELPER}" || -L "${PREREQUISITE_HELPER}" || ! -r "${PREREQUISITE_HELPER}" ]]; then
  printf 'Generation node prerequisite helper is missing or unsafe.\n' >&2
  exit 1
fi

# Admit repository-owned source before importing any project Python.
if [[ "${MODE}" == cli || "${MODE}" == benchmark-preflight ]]; then
  /bin/bash "${PREREQUISITE_HELPER}" validate-worker-repository \
    "${REPOSITORY_ROOT}" "${GENERATION_GIT_COMMIT:-}" "${BASH_SOURCE[0]}" >&2
else
  /bin/bash "${PREREQUISITE_HELPER}" validate-worker-repository \
    "${REPOSITORY_ROOT}" "${GENERATION_GIT_COMMIT:-}" "${BASH_SOURCE[0]}"
fi
# shellcheck source=generation_prerequisites.sh
source "${PREREQUISITE_HELPER}"

EXPECTED_STORAGE="$(realpath -m -- "${REPOSITORY_ROOT}/../storage")"
EXPECTED_VENV="$(realpath -m -- "${REPOSITORY_ROOT}/../runtime/venvs/native")"
if [[ "${MODE}" != smoke && "${STORAGE_ROOT:-}" != "${EXPECTED_STORAGE}" \
  || "${GENERATION_NATIVE_VENV:-}" != "${EXPECTED_VENV}" \
  || ! -x "${EXPECTED_VENV}/bin/python" ]]; then
  printf 'Generation node requires sibling storage and native Generation venv.\n' >&2
  exit 2
fi
if [[ ! "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ \
  || ! "${SLURM_CPUS_PER_TASK:-}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'Generation node requires a Slurm CPU allocation.\n' >&2
  exit 2
fi

cd "${REPOSITORY_ROOT}"

run_python() {
  "${GENERATION_NATIVE_VENV}/bin/python" -m "${PYTHON_MODULE}" "$@"
}

cleanup_benchmark_preflight() {
  local status="$?"
  trap - EXIT
  if [[ "${SCRATCH}" != "${SCRATCH_PARENT%/}/generation-benchmark-preflight."* ]]; then
    printf 'Refusing to remove an unexpected benchmark scratch directory.\n' >&2
    exit 1
  fi
  rm -rf -- "${SCRATCH}" || exit 1
  exit "${status}"
}

validate_source_after_python() {
  local status="$1"
  if ! /bin/bash "${PREREQUISITE_HELPER}" validate-worker-repository \
    "${REPOSITORY_ROOT}" "${GENERATION_GIT_COMMIT}" "${BASH_SOURCE[0]}" >&2; then
    if (( status == 0 )); then
      status=1
    fi
  fi
  return "${status}"
}

run_python_mode() {
  local status=0
  if (( $# < 1 )); then
    printf 'Generation Python mode requires a CLI command.\n' >&2
    return 2
  fi
  if [[ "${MODE}" == benchmark-preflight ]]; then
    case "$1" in
      materialize-core-benchmark-inputs|submit-core-benchmark|resume-core-benchmark) ;;
      *) printf 'Unsupported native benchmark preparation operation: %s\n' "$1" >&2; return 2 ;;
    esac
    if ! module load Comsol/v6.4 >&2; then
      printf 'Generation node could not load Comsol/v6.4.\n' >&2
      return 1
    fi
    local comsol_executable comsol_version
    comsol_executable="$(command -v comsol)" || {
      printf 'Native COMSOL executable is unavailable.\n' >&2
      return 1
    }
    comsol_executable="$(readlink -f -- "${comsol_executable}")"
    comsol_version="$(generation_comsol_version "${comsol_executable}")" || {
      printf 'Native COMSOL version query failed.\n' >&2
      return 1
    }
    SCRATCH_PARENT="${TMPDIR:-/tmp}"
    if [[ "${SCRATCH_PARENT}" != /* || ! -d "${SCRATCH_PARENT}" || ! -w "${SCRATCH_PARENT}" ]]; then
      printf 'Native benchmark scratch parent is unavailable: %s\n' "${SCRATCH_PARENT}" >&2
      return 1
    fi
    SCRATCH="$(mktemp -d "${SCRATCH_PARENT%/}/generation-benchmark-preflight.XXXXXXXX")"
    trap cleanup_benchmark_preflight EXIT
    if [[ "$1" != resume-core-benchmark ]]; then
      run_python "$@" --scratch-root "${SCRATCH}" \
        --comsol-version-output "${comsol_version}" \
        --comsol-executable-path "${comsol_executable}" || status=$?
    else
      run_python "$@" --scratch-root "${SCRATCH}" || status=$?
    fi
  else
    run_python "$@" || status=$?
  fi
  validate_source_after_python "${status}"
}

validate_case_environment() {
  local variable_name value
  for variable_name in GENERATION_COMSOL_MODULE \
    GENERATION_PYTHON_EXECUTABLE GENERATION_COMSOL_EXECUTABLE; do
    value="${!variable_name:-}"
    if [[ -z "${value}" || "${value}" == *$'\n'* || "${value}" == *$'\r'* ]]; then
      printf '%s must be supplied by the resolved execution plan.\n' "${variable_name}" >&2
      return 2
    fi
  done
  generation_require_command "CPU compute-node" module "compute bootstrap"
  generation_require_command "CPU compute-node" mktemp "case scratch creation"
  if ! module load "${GENERATION_COMSOL_MODULE}"; then
    generation_prerequisite_failed \
      "CPU compute-node" "COMSOL module ${GENERATION_COMSOL_MODULE}" "compute"
  fi
  generation_require_command \
    "CPU compute-node" "${GENERATION_PYTHON_EXECUTABLE}" "case materialization and HDF5 admission"
  generation_run_check \
    "CPU compute-node" "python-version:${GENERATION_PYTHON_EXECUTABLE}" "compute" \
    "${GENERATION_PYTHON_EXECUTABLE}" --version
  generation_require_command "CPU compute-node" "${GENERATION_COMSOL_EXECUTABLE}" "compute"
  generation_run_check \
    "CPU compute-node" "comsol-version:${GENERATION_COMSOL_EXECUTABLE}" "compute" \
    generation_comsol_version "${GENERATION_COMSOL_EXECUTABLE}"
  generation_validate_native_venv \
    "CPU compute-node" "${GENERATION_NATIVE_VENV}" \
    "case materialization and HDF5 conversion/admission"
}

CASE_RUN_ID=""
WORK_ROOT=""
MARKER_READY=false
CHILD_PID=""
INTERRUPTION_SIGNAL=""

cleanup_case_worker() {
  if [[ "${MARKER_READY}" != true ]]; then
    rmdir -- "${WORK_ROOT}"
    return
  fi
  run_python cleanup-worker-workspace "${WORK_ROOT}" \
    --campaign-run-id "${CASE_RUN_ID}" --storage-root "${STORAGE_ROOT}"
}

handle_case_signal() {
  INTERRUPTION_SIGNAL="$1"
  if [[ -n "${CHILD_PID}" ]]; then
    kill -TERM "${CHILD_PID}" 2>/dev/null || true
  fi
}

on_case_exit() {
  local status="$?" cleanup_status=0
  trap - EXIT INT TERM
  set +e
  if [[ "${MODE}" == campaign-case && -n "${INTERRUPTION_SIGNAL}" ]]; then
    run_python record-worker-interruption "${CASE_RUN_ID}" \
      --signal "${INTERRUPTION_SIGNAL}" --exit-code "${status}" \
      --storage-root "${STORAGE_ROOT}" \
      || printf 'Could not persist worker interruption receipt.\n' >&2
  fi
  cleanup_case_worker
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

run_case_worker() {
  local scratch_parent prefix worker_status
  validate_case_environment
  scratch_parent="${TMPDIR:-/tmp}"
  if [[ "${scratch_parent}" != /* || ! -d "${scratch_parent}" || ! -w "${scratch_parent}" ]]; then
    generation_prerequisite_missing \
      "CPU compute-node" "writable scratch parent: ${scratch_parent}" "compute"
  fi
  if [[ "${MODE}" == campaign-case ]]; then
    prefix="vp2-generation-${SLURM_JOB_ID}"
  else
    prefix="vp2-benchmark-${SLURM_JOB_ID}-${CASE_ROLE}"
  fi
  WORK_ROOT="$(mktemp -d "${scratch_parent%/}/${prefix}.XXXXXX")"
  trap 'handle_case_signal INT' INT
  trap 'handle_case_signal TERM' TERM
  trap on_case_exit EXIT

  run_python initialize-worker-workspace "${WORK_ROOT}" \
    --campaign-run-id "${CASE_RUN_ID}" --storage-root "${STORAGE_ROOT}" >/dev/null
  MARKER_READY=true

  if [[ "${MODE}" == campaign-case ]]; then
    run_python run-campaign-case "${CASE_RUN_ID}" "${BATCH_NAME}" "${CASE_INDEX}" \
      --storage-root "${STORAGE_ROOT}" --work-root "${WORK_ROOT}" &
  else
    run_python run-core-benchmark-case "${CASE_RUN_ID}" "${VARIANT_ID}" "${CASE_ROLE}" \
      --storage-root "${STORAGE_ROOT}" --work-root "${WORK_ROOT}" &
  fi
  CHILD_PID="$!"
  set +e
  wait "${CHILD_PID}"
  worker_status="$?"
  if [[ -n "${INTERRUPTION_SIGNAL}" ]]; then
    wait "${CHILD_PID}" 2>/dev/null
    case "${INTERRUPTION_SIGNAL}" in
      INT) worker_status=130 ;;
      TERM) worker_status=143 ;;
    esac
  fi
  set -e
  CHILD_PID=""
  exit "${worker_status}"
}

cleanup_smoke() {
  local status="$?"
  trap - EXIT
  if (( status != 0 )) && [[ -f "${WORK_ROOT}/smoke.log" ]]; then
    tail -n 80 "${WORK_ROOT}/smoke.log" >&2
  fi
  if [[ "${WORK_ROOT}" != "${SCRATCH_PARENT%/}/generation-native-smoke-${SLURM_JOB_ID}."* ]]; then
    printf 'Refusing to remove an unexpected smoke workspace.\n' >&2
    exit 1
  fi
  rm -rf -- "${WORK_ROOT}"
  exit "${status}"
}

run_smoke() {
  if (( $# != 0 )) || [[ "${SLURM_CPUS_PER_TASK}" != 1 ]]; then
    printf 'Native Generation smoke requires a one-CPU Slurm allocation.\n' >&2
    return 2
  fi
  module load Comsol/v6.4
  "${GENERATION_NATIVE_VENV}/bin/python" -c \
    'import sys, h5py, numpy, scipy, torch, yaml; import src.generation.cli.cli_generation; print("Generation Python", sys.version.split()[0])'

  SCRATCH_PARENT="${TMPDIR:-/tmp}"
  if [[ "${SCRATCH_PARENT}" != /* || ! -d "${SCRATCH_PARENT}" || ! -w "${SCRATCH_PARENT}" ]]; then
    printf 'Smoke scratch parent is unavailable.\n' >&2
    return 1
  fi
  WORK_ROOT="$(mktemp -d "${SCRATCH_PARENT%/}/generation-native-smoke-${SLURM_JOB_ID}.XXXXXXXX")"
  trap cleanup_smoke EXIT
  mkdir -p -- "${WORK_ROOT}/configuration" "${WORK_ROOT}/tmp"
  cat > "${WORK_ROOT}/GenerationNativeSmoke.java" <<'JAVA'
import com.comsol.model.Model;
import com.comsol.model.util.ModelUtil;

public class GenerationNativeSmoke {
    public static void main(String[] args) throws java.io.IOException {
        Model model = run();
        model.save("smoke.mph");
    }

    public static Model run() {
        return ModelUtil.create("GenerationNativeSmoke");
    }
}
JAVA
  cd "${WORK_ROOT}"
  generation_comsol_version comsol
  comsol compile -configuration "${WORK_ROOT}/configuration" GenerationNativeSmoke.java
  [[ -s GenerationNativeSmoke.class ]] ||
    { printf 'COMSOL Java smoke model was not compiled.\n' >&2; return 1; }
  comsol batch -configuration "${WORK_ROOT}/configuration" -tmpdir "${WORK_ROOT}/tmp" \
    -inputfile GenerationNativeSmoke.class -outputfile smoke.mph \
    -job generation-native-smoke -np 1 -batchlog smoke.log -batchlogout
  [[ -s smoke.mph ]] ||
    { printf 'COMSOL native batch did not publish its disposable model.\n' >&2; return 1; }
  printf 'GENERATION NATIVE SMOKE PASS job=%s comsol=6.4 model_bytes=%s\n' \
    "${SLURM_JOB_ID}" "$(stat -c %s smoke.mph)"
}

case "${MODE}" in
  cli|benchmark-preflight)
    run_python_mode "$@"
    ;;
  campaign-case)
    if (( $# != 4 )); then
      printf 'Usage: %s REPOSITORY campaign-case CAMPAIGN_RUN_ID BATCH_NAME CASE_INDEX CORES_PER_CASE\n' "$0" >&2
      exit 2
    fi
    CASE_RUN_ID="$1"
    BATCH_NAME="$2"
    CASE_INDEX="$3"
    CORES_PER_CASE="$4"
    if [[ ! "${CASE_RUN_ID}" =~ ^[A-Za-z0-9._-]+__[0-9a-f]{16}$ \
      || "${GENERATION_CAMPAIGN_RUN_ID:-}" != "${CASE_RUN_ID}" \
      || ! "${BATCH_NAME}" =~ ^[A-Za-z0-9._-]+$ \
      || ! "${CASE_INDEX}" =~ ^[1-9][0-9]*$ \
      || ! "${CORES_PER_CASE}" =~ ^[1-9][0-9]*$ \
      || ! "${GENERATION_ATTEMPT_INDEX:-}" =~ ^[1-9][0-9]*$ \
      || "${SLURM_CPUS_PER_TASK}" -ne "${CORES_PER_CASE}" \
      || -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
      printf 'Campaign case identity or Slurm allocation is inconsistent.\n' >&2
      exit 2
    fi
    printf -v CASE_ID 'case_%04d' "${CASE_INDEX}"
    CASE_NODE="${SLURMD_NODENAME:-${HOSTNAME:-unavailable}}"
    CASE_STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || printf unavailable)"
    printf 'CASE START campaign_run_id=%s batch=%s case=%s case_index=%s job=%s node=%s cores=%s started_at=%s\n' \
      "${CASE_RUN_ID}" "${BATCH_NAME}" "${CASE_ID}" "${CASE_INDEX}" \
      "${SLURM_JOB_ID}" "${CASE_NODE}" "${CORES_PER_CASE}" "${CASE_STARTED_AT}"
    run_case_worker
    ;;
  benchmark-case)
    if (( $# != 3 )); then
      printf 'Usage: %s REPOSITORY benchmark-case BENCHMARK_RUN_ID VARIANT_ID CASE_ROLE\n' "$0" >&2
      exit 2
    fi
    CASE_RUN_ID="$1"
    VARIANT_ID="$2"
    CASE_ROLE="$3"
    if [[ ! "${CASE_RUN_ID}" =~ ^core_scaling_transient__[0-9a-f]{16}$ \
      || "${GENERATION_BENCHMARK_RUN_ID:-}" != "${CASE_RUN_ID}" \
      || ! "${VARIANT_ID}" =~ ^[A-Za-z0-9._-]+$ \
      || ! "${CASE_ROLE}" =~ ^(nominal|natural)$ \
      || -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
      printf 'Benchmark case identity or Slurm allocation is inconsistent.\n' >&2
      exit 2
    fi
    run_case_worker
    ;;
  smoke)
    run_smoke "$@"
    ;;
  *)
    printf 'Unsupported Generation node mode: %s\n' "${MODE}" >&2
    exit 2
    ;;
esac
