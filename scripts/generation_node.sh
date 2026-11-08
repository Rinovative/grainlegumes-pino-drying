#!/bin/bash -l
set -Eeuo pipefail

if (( $# < 2 )); then
  printf 'Usage: %s SOURCE_TREE cli|campaign-case|node-worker [ARGS...]\n' "$0" >&2
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

# A case step repeats the checks that the node worker reported once at node start: its passing checks stay silent,
# a failing check is still reported on stderr and ends the step.
if [[ "${MODE}" == campaign-case ]]; then
  export GENERATION_PREFLIGHT_QUIET=1
fi

# Admit the frozen source tree, its roots, and the native venv before importing any project Python.
if [[ "${MODE}" == cli ]]; then
  /bin/bash "${PREREQUISITE_HELPER}" validate-worker-repository \
    "${REPOSITORY_ROOT}" "${GENERATION_GIT_COMMIT:-}" "${BASH_SOURCE[0]}" >&2
else
  /bin/bash "${PREREQUISITE_HELPER}" validate-worker-repository \
    "${REPOSITORY_ROOT}" "${GENERATION_GIT_COMMIT:-}" "${BASH_SOURCE[0]}"
fi
# shellcheck source=generation_prerequisites.sh
source "${PREREQUISITE_HELPER}"

if [[ ! "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ \
  || ! "${SLURM_CPUS_PER_TASK:-}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'Generation node requires a Slurm CPU allocation.\n' >&2
  exit 2
fi

cd "${REPOSITORY_ROOT}"

run_python() {
  "${GENERATION_NATIVE_VENV}/bin/python" -m "${PYTHON_MODULE}" "$@"
}

validate_source_after_python() {
  local status="$1"
  if ! GENERATION_PREFLIGHT_QUIET=1 /bin/bash "${PREREQUISITE_HELPER}" validate-worker-repository \
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
  run_python "$@" || status=$?
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

WORKER_WORKSPACE_MARKER=".generation-worker-workspace.json"

# ICE does not reclaim node-local scratch after a job ends, and a killed worker (time limit, OOM, KILL) never runs its trap.
# Remove only this user's marked worker roots whose job no longer exists in the queue; skip the sweep when the queue is unknown.
sweep_stale_worker_scratch() {
  local scratch_parent="$1" active_jobs entry name job_id lock
  if ! active_jobs="$(squeue -h -u "$(id -un)" -o %A 2>/dev/null)"; then
    printf 'Stale worker scratch sweep skipped: the Slurm queue is unavailable.\n' >&2
    return 0
  fi
  active_jobs=$'\n'"${active_jobs// /}"$'\n'
  for entry in "${scratch_parent%/}"/vp2-generation-*.*; do
    if [[ ! -d "${entry}" || -L "${entry}" || ! -O "${entry}" \
      || ! -f "${entry}/${WORKER_WORKSPACE_MARKER}" || -L "${entry}/${WORKER_WORKSPACE_MARKER}" ]]; then
      continue
    fi
    name="${entry##*/}"
    if [[ ! "${name}" =~ ^vp2-generation-([0-9]+)[.-] ]]; then
      continue
    fi
    job_id="${BASH_REMATCH[1]}"
    if [[ "${active_jobs}" == *$'\n'"${job_id}"$'\n'* ]]; then
      continue
    fi
    # The worker holds this lock for its whole lifetime; only a free lock proves the owner is gone.
    lock="${entry}.lock"
    if [[ ! -f "${lock}" || -L "${lock}" || ! -O "${lock}" ]]; then
      continue
    fi
    (
      flock -n -x 9 || exit 0
      printf 'Removing stale worker scratch of finished job %s: %s\n' "${job_id}" "${entry}" >&2
      rm -rf -- "${entry}" && rm -f -- "${lock}"
    ) 9<>"${lock}" || printf 'Could not remove stale worker scratch: %s\n' "${entry}" >&2
  done
}

CASE_RUN_ID=""
WORK_ROOT=""
WORKER_LOCK=""
MARKER_READY=false
CHILD_PID=""
INTERRUPTION_SIGNAL=""

cleanup_case_worker() {
  local cleanup_status=0
  if [[ "${MARKER_READY}" != true ]]; then
    rmdir -- "${WORK_ROOT}" || cleanup_status=$?
  else
    run_python cleanup-worker-workspace "${WORK_ROOT}" \
      --campaign-run-id "${CASE_RUN_ID}" --storage-root "${STORAGE_ROOT}" || cleanup_status=$?
  fi
  if [[ -n "${WORKER_LOCK}" ]]; then
    rm -f -- "${WORKER_LOCK}"
  fi
  return "${cleanup_status}"
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
  if [[ -n "${INTERRUPTION_SIGNAL}" ]]; then
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
  sweep_stale_worker_scratch "${scratch_parent}"
  # A case step inside a node job shares its job ID with every other step; the step ID keeps the scratch root attributable.
  prefix="vp2-generation-${SLURM_JOB_ID}"
  if [[ "${SLURM_STEP_ID:-}" =~ ^[0-9]+$ ]]; then
    prefix+="-${SLURM_STEP_ID}"
  fi
  WORK_ROOT="$(mktemp -d "${scratch_parent%/}/${prefix}.XXXXXX")"
  # Held until this worker and every child inheriting the descriptor exit; the sweep never removes a locked scratch.
  WORKER_LOCK="${WORK_ROOT}.lock"
  exec {WORKER_LOCK_FD}>"${WORKER_LOCK}"
  flock -x "${WORKER_LOCK_FD}"
  trap 'handle_case_signal INT' INT
  trap 'handle_case_signal TERM' TERM
  trap on_case_exit EXIT

  run_python initialize-worker-workspace "${WORK_ROOT}" \
    --campaign-run-id "${CASE_RUN_ID}" --storage-root "${STORAGE_ROOT}" >/dev/null
  MARKER_READY=true

  # Python starts directly in the background (not through run_python): a backgrounded shell function is a subshell,
  # and the trap's TERM would stop that subshell while the case worker kept running without its orderly stop.
  "${GENERATION_NATIVE_VENV}/bin/python" -m "${PYTHON_MODULE}" run-campaign-case "${CASE_RUN_ID}" "${BATCH_NAME}" "${CASE_INDEX}" \
    --storage-root "${STORAGE_ROOT}" --work-root "${WORK_ROOT}" &
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

NODE_DIRECTORY=""
NODE_CORES=""
NODE_CHILD_PID=""

handle_node_signal() {
  if [[ -n "${NODE_CHILD_PID}" ]]; then
    kill -"$1" "${NODE_CHILD_PID}" 2>/dev/null || true
  fi
}

# The worker runs in the foreground of this batch script; Slurm's B: signal and scancel reach only this shell, so the
# trap hands TERM and INT on to the Python worker, which stops launching and forwards TERM to its running steps.
run_node_worker_mode() {
  local worker_status=0
  # The one preflight report of this node job. A subshell keeps the loaded COMSOL module out of the worker, and a failing
  # check does not stop the worker: each case step repeats the checks and fails on its own.
  ( validate_case_environment ) || printf 'Node job preflight: a check failed (see above); each case step repeats it and fails on its own.\n' >&2
  trap 'handle_node_signal TERM' TERM
  trap 'handle_node_signal INT' INT
  # Not run_python: a backgrounded shell function runs in a subshell, and a TERM aimed at that subshell never reaches Python.
  "${GENERATION_NATIVE_VENV}/bin/python" -m "${PYTHON_MODULE}" run-node-worker "${NODE_DIRECTORY}" --cores "${NODE_CORES}" &
  NODE_CHILD_PID="$!"
  set +e
  while :; do
    wait "${NODE_CHILD_PID}"
    worker_status="$?"
    # A trapped signal ends wait early with a status above 128 while the worker still runs; wait again for its own status.
    if ! kill -0 "${NODE_CHILD_PID}" 2>/dev/null; then
      break
    fi
  done
  set -e
  NODE_CHILD_PID=""
  trap - TERM INT
  validate_source_after_python "${worker_status}" || exit "$?"
}

case "${MODE}" in
  cli)
    run_python_mode "$@"
    ;;
  campaign-case)
    if (( $# != 4 )); then
      printf 'Usage: %s SOURCE_TREE campaign-case CAMPAIGN_RUN_ID BATCH_NAME CASE_INDEX CORES_PER_CASE\n' "$0" >&2
      exit 2
    fi
    CASE_RUN_ID="$1"
    BATCH_NAME="$2"
    CASE_INDEX="$3"
    CORES_PER_CASE="$4"
    if [[ ! "${CASE_RUN_ID}" =~ ^source~[a-z0-9]+(_[a-z0-9]+)*~[a-z0-9]+([-_+][a-z0-9]+)*(__[a-z0-9]+([-_+][a-z0-9]+)*)*$ ]]; then
      printf 'Campaign case identity or Slurm allocation is inconsistent.\n' >&2
      exit 2
    fi
    if [[ "${GENERATION_CAMPAIGN_RUN_ID:-}" != "${CASE_RUN_ID}" \
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
    CASE_STARTED_AT="$(date +%Y-%m-%dT%H:%M:%S%:z 2>/dev/null || printf unavailable)"
    printf 'CASE START campaign_run_id=%s batch=%s case=%s case_index=%s job=%s step=%s dispatch=%s node=%s cores=%s started_at=%s\n' \
      "${CASE_RUN_ID}" "${BATCH_NAME}" "${CASE_ID}" "${CASE_INDEX}" \
      "${SLURM_JOB_ID}" "${SLURM_STEP_ID:-unavailable}" "${GENERATION_DISPATCH_KEY:-unavailable}" "${CASE_NODE}" "${CORES_PER_CASE}" "${CASE_STARTED_AT}"
    run_case_worker
    ;;
  node-worker)
    if (( $# != 2 )); then
      printf 'Usage: %s SOURCE_TREE node-worker NODE_DIRECTORY CORES\n' "$0" >&2
      exit 2
    fi
    NODE_DIRECTORY="$1"
    NODE_CORES="$2"
    if [[ "${NODE_DIRECTORY}" != /* \
      || "${NODE_DIRECTORY}" == *$'\n'* || "${NODE_DIRECTORY}" == *$'\r'* \
      || -L "${NODE_DIRECTORY}" || ! -d "${NODE_DIRECTORY}/tickets" \
      || ! "${NODE_CORES}" =~ ^[1-9][0-9]*$ \
      || "${SLURM_CPUS_PER_TASK}" -ne "${NODE_CORES}" \
      || -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
      printf 'Node worker directory or Slurm allocation is inconsistent.\n' >&2
      exit 2
    fi
    run_node_worker_mode
    ;;
  *)
    printf 'Unsupported Generation node mode: %s\n' "${MODE}" >&2
    exit 2
    ;;
esac
