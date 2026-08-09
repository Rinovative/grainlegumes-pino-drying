#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
HOST_STORAGE_ROOT="${STORAGE_ROOT:-${PROJECT_DIR}/../storage}"
LOCAL_PYTHON="${GENERATION_LOCAL_PYTHON:-python3}"
REPOSITORY_URL="https://github.com/Rinovative/grainlegumes-pino-drying.git"
DEFAULT_CPU_HOST="sricehpc01"
GENERATION_MODULE="src.generation.cli.cli_generation"
PYTHON_MODULE="Python/3.10"
COMSOL_MODULE="Comsol/v6.4"
CORES_PER_NODE=32
STATUS_POLL_SECONDS="${GENERATION_STATUS_POLL_SECONDS:-30}"
ALL_WORKFLOW_ACTIVE=false
ALL_STAGE="not_started"
RUN_ID=""
CPU_BYTES_RETAINED=0

usage() {
  cat >&2 <<EOF
Usage:
  $0 setup-cpu [--cpu-host HOST] [--remote-root PATH] [--git-commit COMMIT] [--execute]
  $0 preflight CAMPAIGN --max-nodes N --cases-per-node N --cores-per-case N --max-parallel-cases N [options]
  $0 plan CAMPAIGN --max-nodes N --cases-per-node N --cores-per-case N --max-parallel-cases N [options]
  $0 launch CAMPAIGN --max-nodes N --cases-per-node N --cores-per-case N --max-parallel-cases N [options]
  $0 all CAMPAIGN --max-nodes N --cases-per-node N --cores-per-case N --max-parallel-cases N [--detach] [--keep-cpu-source] [options]
  $0 status [CAMPAIGN_RUN_ID] [remote options]
  $0 collect|build-datasets|resume CAMPAIGN_RUN_ID [options]
  $0 cleanup CAMPAIGN_RUN_ID [--confirm] [remote options]
  $0 accounting|cancel|validate CAMPAIGN_RUN_ID [remote options]

Remote options:
  --cpu-host HOST       default: ${DEFAULT_CPU_HOST}
  --remote-root PATH    default: remote HOME/grainlegumes-generation
  --git-commit COMMIT   exact lowercase 40-character commit
  --only-batch NAME     one predeclared batch
  --wall-time TIME      Slurm [days-]hours:minutes:seconds

setup-cpu and cleanup are dry runs unless --execute or --confirm is supplied.
collect validates and publishes GPU generation data without deleting CPU sources.
all waits synchronously, builds every package, smokes every loader, and cleans the
verified CPU source by default. Use only --keep-cpu-source to retain it.
EOF
}
fail() {
  local status="$1"
  shift
  printf '%s\n' "$*" >&2
  exit "${status}"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail 1 "Required command was not found: $1"
}

validate_host() {
  [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail 2 "Unsafe CPU host: $1"
}

validate_path() {
  local label="$1"
  local value="$2"
  [[ "${value}" == /* && "${value}" != "/" ]] || fail 2 "${label} must be an absolute non-root path."
  [[ "${value}" != *$'\n'* && "${value}" != *$'\r'* && "${value}" != *$'\t'* ]]     || fail 2 "${label} contains a control character."
  local component
  IFS='/' read -r -a components <<< "${value#/}"
  for component in "${components[@]}"; do
    [[ -n "${component}" && "${component}" != . && "${component}" != .. ]]       || fail 2 "${label} contains an unsafe component."
  done
}

validate_commit() {
  [[ "$1" =~ ^[0-9a-f]{40}$ ]] || fail 2 "Git commit must be one lowercase 40-character identifier."
}

validate_run_id() {
  [[ "$1" =~ ^[A-Za-z0-9._-]+__[0-9a-f]{16}$ ]] || fail 2 "Malformed campaign-run ID: $1"
}

validate_positive() {
  [[ "$2" =~ ^[1-9][0-9]*$ ]] || fail 2 "$1 must be an integer >= 1."
}

validate_nonnegative() {
  [[ "$2" =~ ^[0-9]+$ ]] || fail 2 "$1 must be an integer >= 0."
}

validate_digest() {
  [[ "$1" =~ ^[0-9a-f]{64}$ ]] || fail 2 "Malformed SHA-256 digest."
}

validate_batch() {
  [[ "$1" =~ ^[a-z0-9_]+__[a-z0-9_]+__(natural|parameter_ood)$ ]]     || fail 2 "Invalid --only-batch value."
}

validate_wall_time() {
  [[ "$1" =~ ^([0-9]+-)?[0-9]{1,2}:[0-9]{2}:[0-9]{2}$ ]]     || fail 2 "Invalid Slurm wall time."
}

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

remote_bash() {
  local host="$1"
  shift
  local command="bash -l -s --"
  local argument quoted
  for argument in "$@"; do
    printf -v quoted '%q' "${argument}"
    command+=" ${quoted}"
  done
  ssh -o BatchMode=yes -- "${host}" "${command}"
}

read_remote_home() {
  remote_bash "$1" <<'REMOTE'
set -euo pipefail
printf '%s\n' "${HOME}"
REMOTE
}

resolve_remote_layout() {
  validate_host "${CPU_HOST}"
  REMOTE_HOME="$(read_remote_home "${CPU_HOST}")" || fail 1 "Could not resolve remote HOME."
  validate_path "remote HOME" "${REMOTE_HOME}"
  [[ -n "${REMOTE_ROOT}" ]] || REMOTE_ROOT="${REMOTE_HOME}/grainlegumes-generation"
  validate_path "remote root" "${REMOTE_ROOT}"
  [[ "${REMOTE_ROOT}" != "${REMOTE_HOME}" ]] || fail 2 "Remote root must not equal HOME."
  REMOTE_REPOSITORY="${REMOTE_ROOT}/repo"
  REMOTE_STORAGE_ROOT="${REMOTE_ROOT}/storage"
  REMOTE_VENV="${REMOTE_ROOT}/venv"
  validate_path "remote repository" "${REMOTE_REPOSITORY}"
  validate_path "remote storage" "${REMOTE_STORAGE_ROOT}"
  validate_path "remote venv" "${REMOTE_VENV}"
}

resolve_local_commit() {
  local clean="$1"
  require_command git
  local head status
  head="$(git -C "${PROJECT_DIR}" rev-parse HEAD)" || fail 1 "Could not resolve Git HEAD."
  validate_commit "${head}"
  [[ -z "${REQUESTED_COMMIT}" || "${REQUESTED_COMMIT}" == "${head}" ]]     || fail 1 "Requested commit differs from local HEAD."
  REQUESTED_COMMIT="${head}"
  if [[ "${clean}" == true ]]; then
    status="$(git -C "${PROJECT_DIR}" status --porcelain)"
    [[ -z "${status}" ]] || fail 1 "This operation requires a clean local worktree."
  fi
}

resolve_campaign() {
  require_command realpath
  CAMPAIGN_CONFIG_PATH="$(realpath -e -- "$1")" || fail 2 "Campaign config does not exist."
  [[ -f "${CAMPAIGN_CONFIG_PATH}" && ! -L "${CAMPAIGN_CONFIG_PATH}" ]]     || fail 2 "Campaign config is not a safe regular file."
  [[ "${CAMPAIGN_CONFIG_PATH}" == "${PROJECT_DIR}/"* ]]     || fail 2 "Campaign config must be inside the repository."
  CAMPAIGN_RELATIVE_PATH="${CAMPAIGN_CONFIG_PATH#"${PROJECT_DIR}/"}"
  [[ "${CAMPAIGN_RELATIVE_PATH}" != *$'\n'* && "${CAMPAIGN_RELATIVE_PATH}" != *$'\r'* && "${CAMPAIGN_RELATIVE_PATH}" != *$'\t'* ]] ||
    fail 2 "Campaign path contains a control character."
  local component
  IFS='/' read -r -a components <<< "${CAMPAIGN_RELATIVE_PATH}"
  for component in "${components[@]}"; do
    [[ -n "${component}" && "${component}" != . && "${component}" != .. ]]       || fail 2 "Campaign path contains traversal."
  done
}

validate_resources() {
  validate_positive --max-nodes "${MAX_NODES}"
  validate_positive --cases-per-node "${CASES_PER_NODE}"
  validate_positive --cores-per-case "${CORES_PER_CASE}"
  validate_positive --max-parallel-cases "${MAX_PARALLEL_CASES}"
  (( CASES_PER_NODE * CORES_PER_CASE <= CORES_PER_NODE ))     || fail 2 "cases_per_node * cores_per_case exceeds ${CORES_PER_NODE}."
  (( MAX_PARALLEL_CASES <= MAX_NODES * CASES_PER_NODE ))     || fail 2 "max_parallel_cases exceeds max_nodes * cases_per_node."
  [[ -z "${ONLY_BATCH}" ]] || validate_batch "${ONLY_BATCH}"
  [[ -z "${WALL_TIME}" ]] || validate_wall_time "${WALL_TIME}"
}

print_layout() {
  printf 'CPU host: %s\nRemote HOME: %s\nRepository: %s\n'     "${CPU_HOST}" "${REMOTE_HOME}" "${REMOTE_REPOSITORY}"
  printf 'Persistent storage: %s\nVenv: %s\nExact commit: %s\n'     "${REMOTE_STORAGE_ROOT}" "${REMOTE_VENV}" "${REQUESTED_COMMIT}"
  printf 'Modules: %s, %s\n' "${PYTHON_MODULE}" "${COMSOL_MODULE}"
}

verify_remote_setup() {
  resolve_remote_layout
  remote_bash "${CPU_HOST}" \
    "${REMOTE_REPOSITORY}" "${REMOTE_STORAGE_ROOT}" "${REMOTE_VENV}" \
    "${REQUESTED_COMMIT}" "${REPOSITORY_URL}" <<'REMOTE'
set -euo pipefail
repository="$1"; storage="$2"; venv="$3"; commit="$4"; repository_url="$5"
[[ -d "${repository}/.git" && -d "${storage}" && -x "${venv}/bin/python" ]]
[[ ! -L "${repository}" && ! -L "${storage}" && ! -L "${venv}" ]]
[[ "$(stat -c %u "${repository}")" -eq "${UID}" ]]
[[ "$(stat -c %u "${storage}")" -eq "${UID}" ]]
[[ "$(stat -c %u "${venv}")" -eq "${UID}" ]]
[[ -z "$(git -C "${repository}" status --porcelain)" ]]
[[ "$(git -C "${repository}" rev-parse HEAD)" == "${commit}" ]]
[[ "$(git -C "${repository}" remote get-url origin)" == "${repository_url}" ]]
module load Python/3.10
module load Comsol/v6.4
COMSOL_VERSION_OUTPUT="$(comsol -version 2>&1)"
printf '%s\n' "${COMSOL_VERSION_OUTPUT}"
[[ "${COMSOL_VERSION_OUTPUT}" == *"6.4"* ]] || { printf 'COMSOL 6.4 required.\n' >&2; exit 1; }
for name in python3 comsol sbatch squeue sacct scancel rsync; do command -v "${name}" >/dev/null; done
source "${venv}/bin/activate"
cd "${repository}"
python -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 10) else f"Python 3.10 required, got {sys.version}")'
python -c 'import h5py, numpy, scipy, yaml; import src.generation.cli.cli_generation'
REMOTE
}

setup_cpu() {
  resolve_local_commit false
  resolve_remote_layout
  print_layout
  printf 'Mode: %s\n' "$([[ "${EXECUTE_SETUP}" == true ]] && printf execute || printf dry-run)"
  print_command mkdir -p "${REMOTE_ROOT}" "${REMOTE_STORAGE_ROOT}"
  print_command git clone --no-checkout "${REPOSITORY_URL}" "${REMOTE_REPOSITORY}"
  print_command git -C "${REMOTE_REPOSITORY}" fetch origin "${REQUESTED_COMMIT}"
  print_command git -C "${REMOTE_REPOSITORY}" checkout --detach "${REQUESTED_COMMIT}"
  print_command module load "${PYTHON_MODULE}"
  print_command python3 -m venv "${REMOTE_VENV}"
  print_command "${REMOTE_VENV}/bin/python" -m pip install -e "${REMOTE_REPOSITORY}[generation-cpu]"
  print_command module load "${COMSOL_MODULE}"
  if [[ "${EXECUTE_SETUP}" != true ]]; then
    printf 'Dry run: no remote files or jobs were created.\n'
    return
  fi
  remote_bash "${CPU_HOST}" \
    "${REMOTE_ROOT}" "${REMOTE_REPOSITORY}" "${REMOTE_STORAGE_ROOT}" \
    "${REMOTE_VENV}" "${REQUESTED_COMMIT}" "${REPOSITORY_URL}" <<'REMOTE'
set -euo pipefail
root="$1"; repository="$2"; storage="$3"; venv="$4"; commit="$5"; repository_url="$6"
[[ "${root}" != / && "${root}" != "${HOME}" ]]
parent="${root}"
while [[ ! -e "${parent}" ]]; do parent="$(dirname "${parent}")"; done
[[ -d "${parent}" && ! -L "${parent}" && "$(stat -c %u "${parent}")" -eq "${UID}" && -w "${parent}" ]]
mkdir -p "${root}" "${storage}"
[[ ! -L "${root}" && ! -L "${storage}" ]]
if [[ ! -e "${repository}" ]]; then
  git clone --no-checkout "${repository_url}" "${repository}"
else
  [[ -d "${repository}/.git" && ! -L "${repository}" ]]
  [[ -z "$(git -C "${repository}" status --porcelain)" ]]
  [[ "$(git -C "${repository}" remote get-url origin)" == "${repository_url}" ]]
fi
git -C "${repository}" fetch origin "${commit}"
git -C "${repository}" cat-file -e "${commit}^{commit}"
git -C "${repository}" checkout --detach "${commit}"
module load Python/3.10
[[ -x "${venv}/bin/python" ]] || python3 -m venv "${venv}"
source "${venv}/bin/activate"
python -m pip install -e "${repository}[generation-cpu]"
module load Comsol/v6.4
python3 --version
COMSOL_VERSION_OUTPUT="$(comsol -version 2>&1)"
printf '%s\n' "${COMSOL_VERSION_OUTPUT}"
[[ "${COMSOL_VERSION_OUTPUT}" == *"6.4"* ]] || { printf 'COMSOL 6.4 required.\n' >&2; exit 1; }
sbatch --version
rsync --version
for name in squeue sacct scancel; do command -v "${name}" >/dev/null; done
cd "${repository}"
python -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 10) else f"Python 3.10 required, got {sys.version}")'
python -c 'import h5py, numpy, scipy, yaml; import src.generation.cli.cli_generation'
printf 'CPU setup complete: %s\n' "${root}"
REMOTE
}

remote_plan_submit() {
  local operation="$1"
  verify_remote_setup
  remote_bash "${CPU_HOST}" \
    "${REMOTE_REPOSITORY}" "${REMOTE_STORAGE_ROOT}" "${REMOTE_VENV}" \
    "${REQUESTED_COMMIT}" "${CAMPAIGN_RELATIVE_PATH}" \
    "${MAX_NODES}" "${CASES_PER_NODE}" "${CORES_PER_CASE}" \
    "${MAX_PARALLEL_CASES}" "${ONLY_BATCH}" "${WALL_TIME}" "${operation}" <<'REMOTE'
set -euo pipefail
repository="$1"; storage="$2"; venv="$3"; commit="$4"; campaign="$5"
max_nodes="$6"; cases_per_node="$7"; cores_per_case="$8"; max_parallel="$9"
only_batch="${10}"; wall_time="${11}"; operation="${12}"
module load Python/3.10
module load Comsol/v6.4
source "${venv}/bin/activate"
export GENERATION_CPU_VENV="${venv}"
export STORAGE_ROOT="${storage}"
cd "${repository}"
command=(python -m src.generation.cli.cli_generation "${operation}" "${repository}/${campaign}"
  --git-commit "${commit}" --max-nodes "${max_nodes}"
  --cases-per-node "${cases_per_node}" --cores-per-case "${cores_per_case}"
  --max-parallel-cases "${max_parallel}" --cores-per-node 32
  --storage-root "${storage}")
[[ -z "${only_batch}" ]] || command+=(--only-batch "${only_batch}")
[[ -z "${wall_time}" ]] || command+=(--wall-time "${wall_time}")
"${command[@]}"
REMOTE
}

preflight_cpu() {
  resolve_local_commit true
  resolve_campaign "${CAMPAIGN_ARGUMENT}"
  validate_resources
  verify_remote_setup
  print_layout
  remote_bash "${CPU_HOST}" \
    "${REMOTE_REPOSITORY}" "${REMOTE_STORAGE_ROOT}" "${REMOTE_VENV}" \
    "${CAMPAIGN_RELATIVE_PATH}" "${ONLY_BATCH}" \
    "${MAX_NODES}" "${CASES_PER_NODE}" "${CORES_PER_CASE}" "${MAX_PARALLEL_CASES}" <<'REMOTE'
set -euo pipefail
repository="$1"; storage="$2"; venv="$3"; campaign="$4"; only_batch="$5"
max_nodes="$6"; cases_per_node="$7"; cores_per_case="$8"; max_parallel="$9"
preflight_id="$(date -u +%Y%m%dT%H%M%SZ)"
logs="${storage}/01_generation/meta/preflight/${preflight_id}"
mkdir -p "${logs}"
[[ -n "${only_batch}" ]] || only_batch=-
printf 'Preflight log root: %s\n' "${logs}"
sbatch --wait --parsable --partition=standard --nodes=1 --ntasks=1 \
  --cpus-per-task=1 --time=00:05:00 --job-name=vp2-generation-preflight \
  --output="${logs}/slurm-%j.out" --error="${logs}/slurm-%j.err" \
  --chdir="${repository}" "${repository}/scripts/generation_cpu_smoke.sh" \
  "${venv}" "${repository}/${campaign}" "${storage}" "${only_batch}" \
  "${max_nodes}" "${cases_per_node}" "${cores_per_case}" "${max_parallel}" 32 environment-only
REMOTE
}

plan_campaign() {
  resolve_local_commit true
  resolve_campaign "${CAMPAIGN_ARGUMENT}"
  validate_resources
  resolve_remote_layout
  print_layout
  remote_plan_submit plan-campaign
}

launch_campaign() {
  resolve_local_commit true
  resolve_campaign "${CAMPAIGN_ARGUMENT}"
  validate_resources
  resolve_remote_layout
  print_layout
  local output
  output="$(remote_plan_submit submit-campaign)" || fail 1 "Remote launch failed."
  printf '%s\n' "${output}"
  if [[ ${output} =~ \"campaign_run_id\"[[:space:]]*:[[:space:]]*\"([A-Za-z0-9._-]+__[0-9a-f]{16})\" ]]; then
    RUN_ID="${BASH_REMATCH[1]}"
    printf 'Campaign run ID: %s\n' "${RUN_ID}"
  else
    fail 1 "Launch returned no campaign-run ID."
  fi
}

remote_cli() {
  verify_remote_setup
  remote_bash "${CPU_HOST}" "${REMOTE_REPOSITORY}" "${REMOTE_STORAGE_ROOT}" \
    "${REMOTE_VENV}" "$@" <<'REMOTE'
set -euo pipefail
repository="$1"; storage="$2"; venv="$3"
shift 3
module load Python/3.10
source "${venv}/bin/activate"
cd "${repository}"
python -m src.generation.cli.cli_generation "$@"
REMOTE
}

remote_transfer_plan() {
  remote_cli campaign-transfer-plan "${RUN_ID}" --format tsv --storage-root "${REMOTE_STORAGE_ROOT}"
}

resolve_local_storage() {
  require_command realpath
  LOCAL_STORAGE_ROOT="$(realpath -m -- "${HOST_STORAGE_ROOT}")"
  validate_path "local storage" "${LOCAL_STORAGE_ROOT}"
}

resolve_local_python() {
  if [[ "${LOCAL_PYTHON}" == */* ]]; then
    [[ -x "${LOCAL_PYTHON}" ]] || fail 1 "Local generation Python is not executable."
  else
    require_command "${LOCAL_PYTHON}"
  fi
  (
    cd "${PROJECT_DIR}"
    PROJECT_ROOT="${PROJECT_DIR}" "${LOCAL_PYTHON}" -c \
      'import h5py, numpy, scipy, yaml; import src.generation.cli.cli_generation'
  ) || fail 1 "Local generation Python lacks required dependencies."
}

local_cli() {
  (
    cd "${PROJECT_DIR}"
    PROJECT_ROOT="${PROJECT_DIR}" "${LOCAL_PYTHON}" -m "${GENERATION_MODULE}" "$@"
  )
}

validate_transfer_path() {
  local value="$1"
  [[ -n "${value}" && "${value}" != /* \
    && "${value}" != *$'\n'* && "${value}" != *$'\r'* && "${value}" != *$'\t'* ]] ||
    fail 1 "Unsafe transfer path."
  [[ "${value}" != .state* && "${value}" != *"/.state/"* && "${value}" != *"/work/"* ]] ||
    fail 1 "Transfer plan contains private state."
  local component
  IFS='/' read -r -a components <<< "${value}"
  for component in "${components[@]}"; do
    [[ -n "${component}" && "${component}" != . && "${component}" != .. ]] ||
      fail 1 "Transfer path contains traversal."
  done
}

gpu_publication_is_valid() {
  local_cli validate-published-campaign "${RUN_ID}" --storage-root "${LOCAL_STORAGE_ROOT}" >/dev/null 2>&1
}

collect_campaign() {
  resolve_local_python
  resolve_remote_layout
  resolve_local_storage
  if gpu_publication_is_valid; then
    printf 'GPU generation publication validated and reused for %s.\n' "${RUN_ID}"
    return
  fi
  local plan
  plan="$(remote_transfer_plan)" || fail 1 "Remote campaign is not terminally valid."
  local -a directories=()
  local kind field2 field3 field4 field5 field6 field7 extra
  while IFS=$'\t' read -r kind field2 field3 field4 field5 field6 field7 extra; do
    [[ -z "${extra:-}" ]] || fail 1 "Malformed transfer plan."
    case "${kind}" in
      campaign) directories+=("${field4}") ;;
      batch) directories+=("${field5}" "${field6}" "${field7}") ;;
      *) fail 1 "Unknown transfer-plan row." ;;
    esac
  done <<< "${plan}"
  (( ${#directories[@]} >= 4 )) || fail 1 "Transfer plan is empty."
  local directory
  for directory in "${directories[@]}"; do validate_transfer_path "${directory}"; done
  require_command rsync
  local staging receipt
  staging="$(local_cli create-transfer-staging "${RUN_ID}" \
    --storage-root "${LOCAL_STORAGE_ROOT}")" ||
    fail 1 "Could not create marked transfer staging."
  printf 'Transfer staging: %s\n' "${staging}"
  for directory in "${directories[@]}"; do
    rsync -a --protect-args --relative --exclude='.state/' --exclude='work/' \
      "${CPU_HOST}:${REMOTE_STORAGE_ROOT}/./${directory}" "${staging}/" ||
      fail 1 "Transfer failed; staging retained at ${staging}."
  done
  receipt="$(local_cli publish-transferred-campaign "${RUN_ID}" \
    --staging-root "${staging}" --destination-root "${LOCAL_STORAGE_ROOT}" \
    --source-host "${CPU_HOST}" --source-storage-root "${REMOTE_STORAGE_ROOT}")" ||
    fail 1 "GPU publication validation failed; staging retained at ${staging}."
  local_cli cleanup-transfer-staging --campaign-run-id "${RUN_ID}" \
    --directory "${staging}" --storage-root "${LOCAL_STORAGE_ROOT}" --confirm >/dev/null
  printf '%s\nCPU source retained: %s:%s\n' "${receipt}" "${CPU_HOST}" "${REMOTE_STORAGE_ROOT}"
}

build_datasets() {
  resolve_local_python
  resolve_local_storage
  local_cli build-campaign-datasets "${RUN_ID}" --storage-root "${LOCAL_STORAGE_ROOT}"
}

remote_campaign_state() {
  remote_cli campaign-status "${RUN_ID}" --format state --storage-root "${REMOTE_STORAGE_ROOT}"
}

remote_source_status_tsv() {
  remote_cli campaign-source-status "${RUN_ID}" --query-scheduler --format tsv \
    --storage-root "${REMOTE_STORAGE_ROOT}"
}

read_remote_source_status() {
  local line kind status_run state bytes eligibility active extra
  line="$(remote_source_status_tsv)"
  IFS=$'\t' read -r kind status_run state bytes eligibility active extra <<< "${line}"
  [[ "${kind}" == source-status && "${status_run}" == "${RUN_ID}" && -z "${extra:-}" ]] ||
    fail 1 "Malformed CPU source status."
  validate_nonnegative "CPU retained bytes" "${bytes}"
  REMOTE_SOURCE_STATE="${state}"
  CPU_BYTES_RETAINED="${bytes}"
  REMOTE_CLEANUP_ELIGIBILITY="${eligibility}"
  REMOTE_SOURCE_ACTIVE="${active}"
}

wait_for_terminal_publication() {
  validate_nonnegative GENERATION_STATUS_POLL_SECONDS "${STATUS_POLL_SECONDS}"
  while true; do
    read_remote_source_status
    if [[ "${REMOTE_SOURCE_STATE}" == source_cleanup_complete ]]; then
      printf 'CPU source cleanup receipt already exists for %s.\n' "${RUN_ID}"
      return
    fi
    local state
    state="$(remote_campaign_state)"
    printf 'Campaign %s state: %s\n' "${RUN_ID}" "${state}"
    case "${state}" in
      publication_complete)
        remote_cli validate-campaign-terminal "${RUN_ID}" --storage-root "${REMOTE_STORAGE_ROOT}" >/dev/null
        return
        ;;
      running|submitted|submission_pending_or_unknown)
        (( STATUS_POLL_SECONDS == 0 )) || sleep "${STATUS_POLL_SECONDS}"
        ;;
      completed)
        remote_cli validate-campaign-terminal "${RUN_ID}" --storage-root "${REMOTE_STORAGE_ROOT}" >/dev/null
        return
        ;;
      failed|partially_failed|cancelled)
        if [[ "${ALLOW_REMOTE_RESUME}" == true ]]; then
          remote_cli resume-campaign "${RUN_ID}" --storage-root "${REMOTE_STORAGE_ROOT}" >/dev/null
          ALLOW_REMOTE_RESUME=false
        else
          fail 1 "Campaign requires resume from state ${state}."
        fi
        ;;
      *)
        fail 1 "Campaign entered unsupported state: ${state}"
        ;;
    esac
  done
}

prepare_all_receipt() {
  local -a arguments=(prepare-all-workflow "${RUN_ID}" --storage-root "${LOCAL_STORAGE_ROOT}")
  [[ "${KEEP_CPU_SOURCE}" != true ]] || arguments+=(--keep-cpu-source)
  local_cli "${arguments[@]}"
}

read_cleanup_authorization() {
  local line kind extra
  line="$(local_cli cpu-cleanup-authorization "${RUN_ID}" --format tsv \
    --storage-root "${LOCAL_STORAGE_ROOT}")"
  IFS=$'\t' read -r kind AUTHORIZATION_SHA AUTH_SOURCE_HOST AUTH_SOURCE_ROOT \
    AUTH_DESTINATION_ROOT AUTH_TRANSFER_SHA AUTH_DATASET_SHA AUTH_WORKFLOW_SHA \
    AUTH_SOURCE_INVENTORY_SHA AUTH_SOURCE_FILE_COUNT AUTH_SOURCE_BYTES extra <<< "${line}"
  [[ "${kind}" == authorization && -z "${extra:-}" ]] || fail 1 "Malformed CPU cleanup authorization."
  [[ "${AUTH_SOURCE_HOST}" == "${CPU_HOST}" ]] ||
    fail 1 "Cleanup authorization source host differs from the selected CPU host."
  [[ "${AUTH_SOURCE_ROOT}" == "${REMOTE_STORAGE_ROOT}" ]] ||
    fail 1 "Cleanup authorization source root differs from the selected CPU storage."
  [[ "${AUTH_DESTINATION_ROOT}" == "${LOCAL_STORAGE_ROOT}" ]] ||
    fail 1 "Cleanup authorization destination differs from GPU storage."
  validate_digest "${AUTHORIZATION_SHA}"
  validate_digest "${AUTH_TRANSFER_SHA}"
  validate_digest "${AUTH_DATASET_SHA}"
  validate_digest "${AUTH_WORKFLOW_SHA}"
  validate_digest "${AUTH_SOURCE_INVENTORY_SHA}"
  validate_nonnegative "authorized source file count" "${AUTH_SOURCE_FILE_COUNT}"
  validate_nonnegative "authorized source bytes" "${AUTH_SOURCE_BYTES}"
  CPU_BYTES_RETAINED="${AUTH_SOURCE_BYTES}"
}

remote_cleanup_arguments() {
  CLEANUP_ARGUMENTS=(
    cleanup-campaign-source "${RUN_ID}"
    --storage-root "${REMOTE_STORAGE_ROOT}"
    --source-host "${AUTH_SOURCE_HOST}"
    --destination-storage-root "${AUTH_DESTINATION_ROOT}"
    --transfer-receipt-sha256 "${AUTH_TRANSFER_SHA}"
    --dataset-receipt-sha256 "${AUTH_DATASET_SHA}"
    --workflow-gate-sha256 "${AUTH_WORKFLOW_SHA}"
    --source-inventory-sha256 "${AUTH_SOURCE_INVENTORY_SHA}"
    --source-file-count "${AUTH_SOURCE_FILE_COUNT}"
    --source-bytes "${AUTH_SOURCE_BYTES}"
    --authorization-sha256 "${AUTHORIZATION_SHA}"
  )
}

confirm_cpu_cleanup() {
  read_cleanup_authorization
  remote_cleanup_arguments
  local line kind cleanup_status cleanup_mode cleanup_auth reclaimed receipt_sha extra
  line="$(remote_cli "${CLEANUP_ARGUMENTS[@]}" --confirm --format tsv)"
  IFS=$'\t' read -r kind cleanup_status cleanup_mode cleanup_auth reclaimed receipt_sha extra <<< "${line}"
  [[ "${kind}" == cleanup && "${cleanup_status}" == complete \
    && "${cleanup_auth}" == "${AUTHORIZATION_SHA}" && -z "${extra:-}" ]] ||
    fail 1 "Malformed or incomplete CPU cleanup result."
  validate_nonnegative "CPU reclaimed bytes" "${reclaimed}"
  validate_digest "${receipt_sha}"
  [[ "${reclaimed}" == "${AUTH_SOURCE_BYTES}" ]] || fail 1 "CPU reclaimed bytes differ from authorization."
  local_cli record-cpu-cleanup "${RUN_ID}" --storage-root "${LOCAL_STORAGE_ROOT}" \
    --authorization-sha256 "${AUTHORIZATION_SHA}" \
    --cleanup-receipt-sha256 "${receipt_sha}" --reclaimed-bytes "${reclaimed}"
  CPU_BYTES_RETAINED=0
}

cleanup_cpu_source() {
  resolve_local_python
  resolve_remote_layout
  resolve_local_storage
  if [[ "${CONFIRM_CLEANUP}" == true ]]; then
    confirm_cpu_cleanup
    return
  fi
  read_cleanup_authorization
  remote_cleanup_arguments
  remote_cli "${CLEANUP_ARGUMENTS[@]}"
}

storage_status_report() {
  resolve_local_python
  resolve_remote_layout
  resolve_local_storage
  local -a local_arguments=(storage-status --role gpu --storage-root "${LOCAL_STORAGE_ROOT}")
  local -a remote_arguments=(storage-status --role cpu --query-scheduler --storage-root "${REMOTE_STORAGE_ROOT}")
  if [[ -n "${RUN_ID}" ]]; then
    local_arguments+=(--campaign-run-id "${RUN_ID}")
    remote_arguments+=(--campaign-run-id "${RUN_ID}")
  fi
  printf 'GPU storage status:\n'
  local_cli "${local_arguments[@]}"
  printf 'CPU storage status:\n'
  remote_cli "${remote_arguments[@]}"
}

resume_command_text() {
  local -a command=(
    ./scripts/generation_workflow.sh resume "${RUN_ID}"
    --cpu-host "${CPU_HOST}" --remote-root "${REMOTE_ROOT}"
    --git-commit "${REQUESTED_COMMIT}"
  )
  [[ "${KEEP_CPU_SOURCE}" != true ]] || command+=(--keep-cpu-source)
  local rendered="" argument quoted
  for argument in "${command[@]}"; do
    printf -v quoted '%q' "${argument}"
    rendered+="${quoted} "
  done
  printf '%s' "${rendered% }"
}

all_command_text() {
  local -a command=(
    ./scripts/generation_workflow.sh all "${CAMPAIGN_ARGUMENT}"
    --max-nodes "${MAX_NODES}" --cases-per-node "${CASES_PER_NODE}"
    --cores-per-case "${CORES_PER_CASE}" --max-parallel-cases "${MAX_PARALLEL_CASES}"
    --cpu-host "${CPU_HOST}" --remote-root "${REMOTE_ROOT}"
    --git-commit "${REQUESTED_COMMIT}"
  )
  [[ -z "${ONLY_BATCH}" ]] || command+=(--only-batch "${ONLY_BATCH}")
  [[ -z "${WALL_TIME}" ]] || command+=(--wall-time "${WALL_TIME}")
  [[ "${KEEP_CPU_SOURCE}" != true ]] || command+=(--keep-cpu-source)
  local rendered="" argument quoted
  for argument in "${command[@]}"; do
    printf -v quoted '%q' "${argument}"
    rendered+="${quoted} "
  done
  printf '%s' "${rendered% }"
}

workflow_failure_report() {
  local status="$1"
  trap - EXIT
  ALL_WORKFLOW_ACTIVE=false
  local resume
  if [[ -n "${RUN_ID}" ]]; then
    resume="$(resume_command_text)"
    if [[ -n "${LOCAL_STORAGE_ROOT:-}" ]]; then
      local_cli record-workflow-failure "${RUN_ID}" \
        --storage-root "${LOCAL_STORAGE_ROOT}" --stage "${ALL_STAGE}" \
        --resume-command "${resume}" --cpu-bytes-retained "${CPU_BYTES_RETAINED}" \
        >/dev/null 2>&1 || true
    fi
  else
    resume="$(all_command_text)"
  fi
  printf 'All workflow failed.\nStage: %s\nCPU bytes retained: %s\nResume command: %s\n' \
    "${ALL_STAGE}" "${CPU_BYTES_RETAINED}" "${resume}" >&2
  return "${status}"
}

workflow_exit_handler() {
  local status="$1"
  if [[ "${ALL_WORKFLOW_ACTIVE}" == true && "${status}" -ne 0 ]]; then
    workflow_failure_report "${status}" || true
  fi
}

continue_all_workflow() {
  ALL_WORKFLOW_ACTIVE=true
  trap 'workflow_exit_handler $?' EXIT
  resolve_local_python
  resolve_local_storage
  resolve_remote_layout
  ALL_STAGE="existing terminal receipt validation"
  if local_cli validate-all-workflow "${RUN_ID}" --storage-root "${LOCAL_STORAGE_ROOT}" >/dev/null 2>&1; then
    printf 'All workflow already complete and validated for %s.\n' "${RUN_ID}"
    ALL_WORKFLOW_ACTIVE=false
    trap - EXIT
    return
  fi
  ALL_STAGE="remote generation completion"
  wait_for_terminal_publication
  ALL_STAGE="GPU collection and atomic publication"
  if ! gpu_publication_is_valid; then
    [[ "${REMOTE_SOURCE_STATE}" != source_cleanup_complete ]] ||
      fail 1 "CPU source is cleaned but no valid GPU publication exists."
    collect_campaign
  else
    printf 'GPU generation publication validated and reused for %s.\n' "${RUN_ID}"
  fi
  ALL_STAGE="dataset build, inspection, and loader smokes"
  build_datasets
  ALL_STAGE="terminal pre-cleanup receipt"
  prepare_all_receipt
  if [[ "${KEEP_CPU_SOURCE}" == true ]]; then
    printf 'CPU source retained by --keep-cpu-source.\n'
  else
    ALL_STAGE="verified CPU source cleanup"
    confirm_cpu_cleanup
  fi
  ALL_STAGE="terminal receipt validation"
  local_cli validate-all-workflow "${RUN_ID}" --storage-root "${LOCAL_STORAGE_ROOT}"
  ALL_WORKFLOW_ACTIVE=false
  trap - EXIT
}

run_all() {
  ALL_WORKFLOW_ACTIVE=true
  trap 'workflow_exit_handler $?' EXIT
  ALL_STAGE="local repository, campaign, and resource validation"
  resolve_local_commit true
  resolve_campaign "${CAMPAIGN_ARGUMENT}"
  validate_resources
  resolve_remote_layout
  resolve_local_storage
  resolve_local_python
  print_layout
  if [[ "${KEEP_CPU_SOURCE}" == true ]]; then
    printf 'CPU source cleanup after complete success: disabled\n'
  else
    printf 'CPU source cleanup after complete success: enabled\n'
  fi
  printf 'Resolved campaign plan:\n'
  ALL_STAGE="CPU setup and resolved campaign plan"
  remote_plan_submit plan-campaign
  ALL_STAGE="campaign launch"
  launch_campaign
  if [[ "${DETACH}" == true ]]; then
    printf 'Detached after launch; collection, dataset building, and cleanup were not claimed.\n'
    printf 'Resume-all command: %s\n' "$(resume_command_text)"
    ALL_WORKFLOW_ACTIVE=false
    trap - EXIT
    return
  fi
  ALLOW_REMOTE_RESUME=false
  continue_all_workflow
}

resume_all() {
  resolve_local_commit false
  resolve_remote_layout
  resolve_local_storage
  ALLOW_REMOTE_RESUME=true
  continue_all_workflow
}

(( $# > 0 )) || { usage; exit 2; }
[[ "$1" != -h && "$1" != --help ]] || { usage; exit 0; }
SUBCOMMAND="$1"
shift
CPU_HOST="${DEFAULT_CPU_HOST}"
REMOTE_ROOT=""
REQUESTED_COMMIT=""
EXECUTE_SETUP=false
CONFIRM_CLEANUP=false
DETACH=false
KEEP_CPU_SOURCE=false
ONLY_BATCH=""
WALL_TIME=""
MAX_NODES=""
CASES_PER_NODE=""
CORES_PER_CASE=""
MAX_PARALLEL_CASES=""
POSITIONAL=()

while (( $# > 0 )); do
  case "$1" in
    --cpu-host) (( $# >= 2 )) || fail 2 "--cpu-host requires a value."; CPU_HOST="$2"; shift 2 ;;
    --remote-root) (( $# >= 2 )) || fail 2 "--remote-root requires a value."; REMOTE_ROOT="$2"; shift 2 ;;
    --git-commit) (( $# >= 2 )) || fail 2 "--git-commit requires a value."; REQUESTED_COMMIT="$2"; shift 2 ;;
    --execute) EXECUTE_SETUP=true; shift ;;
    --confirm) CONFIRM_CLEANUP=true; shift ;;
    --detach) DETACH=true; shift ;;
    --keep-cpu-source) KEEP_CPU_SOURCE=true; shift ;;
    --only-batch) (( $# >= 2 )) || fail 2 "--only-batch requires a value."; ONLY_BATCH="$2"; shift 2 ;;
    --wall-time) (( $# >= 2 )) || fail 2 "--wall-time requires a value."; WALL_TIME="$2"; shift 2 ;;
    --max-nodes) (( $# >= 2 )) || fail 2 "--max-nodes requires a value."; MAX_NODES="$2"; shift 2 ;;
    --cases-per-node) (( $# >= 2 )) || fail 2 "--cases-per-node requires a value."; CASES_PER_NODE="$2"; shift 2 ;;
    --cores-per-case) (( $# >= 2 )) || fail 2 "--cores-per-case requires a value."; CORES_PER_CASE="$2"; shift 2 ;;
    --max-parallel-cases) (( $# >= 2 )) || fail 2 "--max-parallel-cases requires a value."; MAX_PARALLEL_CASES="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --*) fail 2 "Unsupported option: $1" ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done

[[ -z "${REQUESTED_COMMIT}" ]] || validate_commit "${REQUESTED_COMMIT}"

case "${SUBCOMMAND}" in
  setup-cpu)
    (( ${#POSITIONAL[@]} == 0 )) || fail 2 "setup-cpu accepts no positional arguments."
    [[ "${DETACH}" == false && "${KEEP_CPU_SOURCE}" == false && "${CONFIRM_CLEANUP}" == false ]]       || fail 2 "Unsupported setup-cpu option."
    setup_cpu
    ;;
  preflight|plan|launch|all)
    (( ${#POSITIONAL[@]} == 1 )) || fail 2 "${SUBCOMMAND} requires one campaign config."
    CAMPAIGN_ARGUMENT="${POSITIONAL[0]}"
    case "${SUBCOMMAND}" in
      preflight)
        [[ "${DETACH}" == false && "${KEEP_CPU_SOURCE}" == false ]] || fail 2 "Unsupported preflight option."
        preflight_cpu
        ;;
      plan)
        [[ "${DETACH}" == false && "${KEEP_CPU_SOURCE}" == false ]] || fail 2 "Unsupported plan option."
        plan_campaign
        ;;
      launch)
        [[ "${DETACH}" == false && "${KEEP_CPU_SOURCE}" == false ]] || fail 2 "launch already submits and returns."
        launch_campaign
        ;;
      all) run_all ;;
    esac
    ;;
  status)
    (( ${#POSITIONAL[@]} <= 1 )) || fail 2 "status accepts at most one campaign-run ID."
    RUN_ID="${POSITIONAL[0]:-}"
    [[ -z "${RUN_ID}" ]] || validate_run_id "${RUN_ID}"
    resolve_local_commit false
    storage_status_report
    ;;
  collect|build-datasets|resume|cleanup|accounting|cancel|validate)
    (( ${#POSITIONAL[@]} == 1 )) || fail 2 "${SUBCOMMAND} requires one campaign-run ID."
    RUN_ID="${POSITIONAL[0]}"
    validate_run_id "${RUN_ID}"
    resolve_local_commit false
    case "${SUBCOMMAND}" in
      collect)
        [[ "${CONFIRM_CLEANUP}" == false && "${KEEP_CPU_SOURCE}" == false ]]           || fail 2 "collect is always non-destructive."
        collect_campaign
        ;;
      build-datasets)
        [[ "${CONFIRM_CLEANUP}" == false && "${KEEP_CPU_SOURCE}" == false ]]           || fail 2 "Unsupported build-datasets option."
        build_datasets
        ;;
      resume)
        [[ "${CONFIRM_CLEANUP}" == false && "${DETACH}" == false ]]           || fail 2 "Unsupported resume option."
        resume_all
        ;;
      cleanup)
        [[ "${KEEP_CPU_SOURCE}" == false && "${DETACH}" == false ]]           || fail 2 "Unsupported cleanup option."
        cleanup_cpu_source
        ;;
      accounting)
        resolve_remote_layout
        remote_cli campaign-accounting "${RUN_ID}" --storage-root "${REMOTE_STORAGE_ROOT}"
        ;;
      cancel)
        resolve_remote_layout
        remote_cli cancel-campaign "${RUN_ID}" --storage-root "${REMOTE_STORAGE_ROOT}"
        ;;
      validate)
        resolve_remote_layout
        remote_cli validate-campaign-terminal "${RUN_ID}" --storage-root "${REMOTE_STORAGE_ROOT}"
        ;;
    esac
    ;;
  *)
    usage
    fail 2 "Unsupported subcommand: ${SUBCOMMAND}"
    ;;
esac
