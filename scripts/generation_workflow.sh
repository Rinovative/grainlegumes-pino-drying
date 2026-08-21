#!/usr/bin/env bash
set -Eeuo pipefail

ORIGINAL_ARGUMENTS=("$@")
SCRIPT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DEVELOPMENT_REPO_ROOT=""
HOST_REPO_ROOT=""
HOST_STORAGE_ROOT=""
DOCKER_PYTHON=""
PINNED_SOURCE_CONTAINER=""
PINNED_SOURCE_PARENT=""
PINNED_SOURCE_COMMIT=""
CPU_BOOTSTRAP_REPOSITORY_URL="https://github.com/Rinovative/grainlegumes-pino-drying.git"
GENERATION_MODULE="src.generation.cli.cli_generation"
BENCHMARK_SUITE_RELATIVE_PATH="configs/generation/benchmarks/transient_core_scaling/suite.yaml"
STATIONARY_SMOKE_CAMPAIGN_PATH=""
TRANSIENT_SMOKE_CAMPAIGN_PATH=""
STATIONARY_PRIMARY_CAMPAIGN_PATH=""
TRANSIENT_PRIMARY_CAMPAIGN_PATH=""
STATIONARY_SMOKE_CAMPAIGN_HOST_PATH=""
TRANSIENT_SMOKE_CAMPAIGN_HOST_PATH=""
STATIONARY_PRIMARY_CAMPAIGN_HOST_PATH=""
TRANSIENT_PRIMARY_CAMPAIGN_HOST_PATH=""
CAMPAIGN_PURPOSE=""
SCHEDULER_KIND=""
PARTITION=""
CORES_PER_NODE=""
PYTHON_MODULE=""
COMSOL_MODULE=""
PYTHON_EXECUTABLE=""
COMSOL_EXECUTABLE=""
STATUS_POLL_SECONDS=""
ALL_WORKFLOW_ACTIVE=false
ALL_STAGE="not_started"
RUN_ID=""
CPU_BYTES_RETAINED=0
CPU_BYTES_RETAINED_EXACT=false
CPU_BYTES_RECLAIMED=0
CPU_CLEANUP_RECEIPT_SHA=""
PILOT_MODE=false
PILOT_STAGING_RECLAIMED=0
REMOTE_SETUP_IDENTITY=""
HUMAN_WORKFLOW_MODE=false
CONSOLE_PROGRESS_KEY=""
CONSOLE_PROGRESS_SIGNATURE=""
CONSOLE_PROGRESS_DETAIL_SIGNATURE=""
CONSOLE_PROGRESS_RENDERED_AT=0
REMOTE_CAMPAIGN_STATE=""
REMOTE_CAMPAIGN_STATE_SIGNATURE=""
REMOTE_CAMPAIGN_PROGRESS_SIGNATURE=""
REMOTE_CAMPAIGN_SUMMARY=""
TRANSFER_SUMMARY=""
DATASET_SUMMARY=""
WORKFLOW_FAILURE_EVIDENCE=""
CAMPAIGN_INTERRUPT_ACTIVE=false
CAMPAIGN_INTERRUPT_COUNT=0
DEFER_COLLECTION=false
CONSOLE_CHANGED_PROGRESS_SECONDS="${GENERATION_CONSOLE_CHANGED_PROGRESS_SECONDS:-60}"
CONSOLE_HEARTBEAT_SECONDS="${GENERATION_CONSOLE_HEARTBEAT_SECONDS:-120}"
COMPOSITE_CHILD_MODE=false
CAMPAIGN_PARTIAL=false
PAIRED_SMOKE_RECEIPT=""
LOCAL_STORAGE_ROOT=""
LOCAL_PYTHON_READY=false
CONFIGURED_UNIFORM_CASE_COUNT=""
CONFIGURED_TOTAL_CASE_COUNT=""

usage() {
  cat >&2 <<EOF
Usage:
  $0 run CONFIG [--background] [--keep-cpu-source|--defer-collection] [remote options]
  $0 run CONFIG --dry-run
  $0 run CONFIG --preflight-only [remote options]
  $0 setup-cpu [--execute] [remote options]
  $0 status CONFIG_OR_RUN_ID [remote options]
  $0 cancel RUN_ID [--force] [remote options]
  $0 cleanup RUN_ID --confirm [remote options]
  $0 background-status WORKFLOW_SESSION_ID
  $0 background-list

Remote options:
  --cpu-host HOST       explicit override for the configured CPU site
  --remote-root PATH    bootstrap layout default: remote HOME/grainlegumes-generation
  --git-commit COMMIT   exact lowercase 40-character commit

Every maintained Generation workflow starts and resumes with run CONFIG.
Foreground execution is the default. --background changes only controller ownership.
--defer-collection leaves the validated CPU source as the exclusive copy; rerun the
same CONFIG without it to collect and finish. --keep-cpu-source collects and retains
an additional CPU copy. The two collection options are mutually exclusive.
EOF
}
fail() {
  local status="$1"
  shift
  printf '%s\n' "$*" >&2
  exit "${status}"
}

fail_preserving_interrupt() {
  local observed_status="$1" failure_status="$2"
  shift 2
  (( observed_status != 130 )) || exit 130
  fail "${failure_status}" "$@"
}

SSH_TRANSPORT_HELPER="${SCRIPT_DIRECTORY}/generation_ssh_transport.sh"
[[ -f "${SSH_TRANSPORT_HELPER}" && ! -L "${SSH_TRANSPORT_HELPER}" ]] ||
  fail 1 "Generation SSH transport helper is missing or unsafe: ${SSH_TRANSPORT_HELPER}"
# shellcheck source=scripts/generation_ssh_transport.sh
source "${SSH_TRANSPORT_HELPER}"

generation_console_stage() {
  local index="$1" total="$2" label="$3" status="$4" detail="${5:-}"
  local decorated="${label} "
  while (( ${#decorated} < 30 )); do decorated+=.; done
  printf '[%s/%s] %s %s\n' "${index}" "${total}" "${decorated}" "${status}"
  if [[ -n "${detail}" ]]; then
    local detail_line
    while IFS= read -r detail_line; do
      printf '      %s\n' "${detail_line}"
    done <<< "${detail}"
  fi
}

generation_console_progress() {
  local key="$1" index="$2" total="$3" label="$4" status="$5"
  local signature="$6" detail="${7:-}" detail_signature="${8:-$6}" now
  validate_positive "console changed-progress interval" "${CONSOLE_CHANGED_PROGRESS_SECONDS}"
  validate_positive "console heartbeat interval" "${CONSOLE_HEARTBEAT_SECONDS}"
  now="$(date +%s)"
  if [[ "${CONSOLE_PROGRESS_KEY}" == "${key}" && "${CONSOLE_PROGRESS_SIGNATURE}" == "${signature}" ]]; then
    if [[ "${CONSOLE_PROGRESS_DETAIL_SIGNATURE}" == "${detail_signature}" ]]; then
      if (( now - CONSOLE_PROGRESS_RENDERED_AT < CONSOLE_HEARTBEAT_SECONDS )); then
        return
      fi
      detail="${detail}${detail:+$'\n'}heartbeat=unchanged"
    elif (( now - CONSOLE_PROGRESS_RENDERED_AT < CONSOLE_CHANGED_PROGRESS_SECONDS )); then
      return
    fi
  fi
  generation_console_stage "${index}" "${total}" "${label}" "${status}" "${detail}"
  CONSOLE_PROGRESS_KEY="${key}"
  CONSOLE_PROGRESS_SIGNATURE="${signature}"
  CONSOLE_PROGRESS_DETAIL_SIGNATURE="${detail_signature}"
  CONSOLE_PROGRESS_RENDERED_AT="${now}"
}

generation_console_elapsed() {
  local elapsed="$1"
  validate_nonnegative "heartbeat elapsed seconds" "${elapsed}"
  printf "%02d:%02d:%02d" \
    "$((elapsed / 3600))" "$(((elapsed % 3600) / 60))" "$((elapsed % 60))"
}

generation_run_with_heartbeat() {
  local key="$1" index="$2" total="$3" label="$4" operation="$5"
  local progress_detail="${6:-}"
  shift 6
  validate_positive "console heartbeat interval" "${CONSOLE_HEARTBEAT_SECONDS}"
  (( $# > 0 )) || fail 2 "Heartbeat execution requires one command."
  local started_seconds="${SECONDS}" command_pid heartbeat_pid status
  "$@" &
  command_pid=$!
  (
    local elapsed last_progress_at detail current_run child_run sleep_pid=""
    heartbeat_stop() {
      [[ -z "${sleep_pid}" ]] || kill "${sleep_pid}" 2>/dev/null || true
      exit 0
    }
    trap heartbeat_stop TERM INT
    while true; do
      sleep "${CONSOLE_HEARTBEAT_SECONDS}" &
      sleep_pid=$!
      wait "${sleep_pid}" || exit 0
      sleep_pid=""
      kill -0 "${command_pid}" 2>/dev/null || exit 0
      elapsed="$((SECONDS - started_seconds))"
      last_progress_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      current_run="${RUN_PLAN_ID:-${RUN_ID:-unknown}}"
      child_run="${RUN_ID:-}"
      detail="stage=${label}"
      printf -v detail "%s\nrun_id=%s" "${detail}" "${current_run}"
      if [[ -n "${child_run}" && "${child_run}" != "${current_run}" ]]; then
        printf -v detail "%s\nchild_run=%s" "${detail}" "${child_run}"
      fi
      printf -v detail "%s\nelapsed=%s" \
        "${detail}" "$(generation_console_elapsed "${elapsed}")"
      printf -v detail "%s\noperation=%s\nlast_progress_at=%s" \
        "${detail}" "${operation}" "${last_progress_at}"
      [[ -z "${progress_detail}" ]] || \
        printf -v detail "%s\n%s" "${detail}" "${progress_detail}"
      printf -v detail "%s\nheartbeat=active\neta=unavailable" "${detail}"
      generation_console_progress \
        "${key}" "${index}" "${total}" "${label}" RUNNING \
        "${operation}" "${detail}" "${operation}:${elapsed}" >&2
    done
  ) &
  heartbeat_pid=$!
  if wait "${command_pid}"; then
    status=0
  else
    status=$?
  fi
  kill "${heartbeat_pid}" 2>/dev/null || true
  wait "${heartbeat_pid}" 2>/dev/null || true
  return "${status}"
}

generation_console_warning() {
  printf 'WARNING: %s\n' "$*" >&2
}

generation_console_failure() {
  local stage="$1" run_id="$2" reason="$3" evidence="$4"
  local retained="$5" resume="$6"
  printf 'FAILED: %s\n' "${stage}" >&2
  [[ -z "${run_id}" ]] || printf 'campaign_run_id: %s\n' "${run_id}" >&2
  printf 'reason: %s\n' "${reason}" >&2
  [[ -z "${evidence}" ]] || printf 'workflow evidence: %s\n' "${evidence}" >&2
  printf 'CPU bytes retained: %s\nResume:\n  %s\n' "${retained}" "${resume}" >&2
}

generation_console_final() {
  printf 'DONE: %s\n' "$*"
}

disarm_campaign_interrupt() {
  CAMPAIGN_INTERRUPT_ACTIVE=false
  trap - INT
}

campaign_interrupt_handler() {
  [[ "${CAMPAIGN_INTERRUPT_ACTIVE}" == true && -n "${RUN_ID}" ]] || return 130
  CAMPAIGN_INTERRUPT_COUNT=$(( CAMPAIGN_INTERRUPT_COUNT + 1 ))
  local -a cancellation=(cancel-campaign "${RUN_ID}")
  local run_label=campaign
  if [[ "${RUN_KIND:-}" == benchmark ]]; then
    cancellation=(cancel-core-benchmark "${RUN_ID}")
    run_label=benchmark
  fi
  cancellation+=(--storage-root "${REMOTE_STORAGE_ROOT}")
  if (( CAMPAIGN_INTERRUPT_COUNT == 1 )); then
    printf '%s
'       "Graceful ${run_label} cancellation requested."       'Press Ctrl+C again to force cancellation.' >&2
    remote_cli "${cancellation[@]}" >/dev/null &
    local cancellation_pid=$!
    if ! wait "${cancellation_pid}"; then
      generation_console_warning         "graceful cancellation request failed; ${run_label} state remains authoritative"
    fi
    return 0
  fi
  printf 'Force %s cancellation requested.
' "${run_label}" >&2
  cancellation+=(--force)
  if ! remote_cli "${cancellation[@]}" >/dev/null; then
    generation_console_warning       "force cancellation request failed; inspect scheduler and run evidence"
  fi
  disarm_campaign_interrupt
  exit 130
}
arm_campaign_interrupt() {
  CAMPAIGN_INTERRUPT_COUNT=0
  CAMPAIGN_INTERRUPT_ACTIVE=true
  trap campaign_interrupt_handler INT
}

require_command() {
  local command_name="$1"
  local blocked_operation="${2:-host control}"
  command -v "${command_name}" >/dev/null 2>&1 ||
    fail 1 "Bare hpc115 prerequisite missing: ${command_name} (blocks ${blocked_operation})."
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

validate_logical_path() {
  local label="$1"
  local value="$2"
  [[ -n "${value}" && "${value}" != /* ]] ||
    fail 2 "${label} must be a non-empty repository-relative path."
  [[ "${value}" != *$'\n'* && "${value}" != *$'\r'* && "${value}" != *$'\t'* ]] ||
    fail 2 "${label} contains a control character."
  local component
  IFS='/' read -r -a components <<< "${value}"
  for component in "${components[@]}"; do
    [[ -n "${component}" && "${component}" != . && "${component}" != .. ]] ||
      fail 2 "${label} contains an unsafe path component."
  done
}

resolve_host_layout() {
  require_command git
  require_command realpath
  local discovered_root script_relative
  discovered_root="$(git -C "${SCRIPT_DIRECTORY}" rev-parse --show-toplevel)" ||
    fail 1 "Could not resolve the bare-host repository root."
  HOST_REPO_ROOT="$(realpath -e -- "${discovered_root}")" ||
    fail 1 "Could not canonicalize the bare-host repository root."
  [[ -d "${HOST_REPO_ROOT}" && ! -L "${HOST_REPO_ROOT}" ]] ||
    fail 1 "Bare-host repository root is not a safe directory."
  script_relative="$(realpath --relative-to="${HOST_REPO_ROOT}" -- "${SCRIPT_DIRECTORY}")" ||
    fail 1 "Could not locate the workflow script inside the repository."
  [[ "${script_relative}" != .. && "${script_relative}" != ../* ]] ||
    fail 1 "Generation workflow script is outside the resolved repository."
  if [[ "${GENERATION_WORKFLOW_PINNED_HANDOFF:-}" == 1 ]]; then
    local development_root storage_root
    development_root="$(realpath -e -- "${GENERATION_WORKFLOW_DEVELOPMENT_REPO_ROOT:-}")" ||
      fail 1 "Could not recover the development checkout from the pinned-source handoff."
    [[ -d "${development_root}" && ! -L "${development_root}" ]] ||
      fail 1 "Pinned-source handoff development checkout is not a safe directory."
    validate_path "development repository" "${development_root}"
    DEVELOPMENT_REPO_ROOT="${development_root}"
    storage_root="$(realpath -m -- "${GENERATION_WORKFLOW_STORAGE_ROOT:-}")" ||
      fail 1 "Could not recover local storage from the pinned-source handoff."
    validate_path "pinned-source local storage" "${storage_root}"
    HOST_STORAGE_ROOT="${storage_root}"
  else
    DEVELOPMENT_REPO_ROOT="${HOST_REPO_ROOT}"
    HOST_STORAGE_ROOT="$(realpath -m -- "${STORAGE_ROOT:-${DEVELOPMENT_REPO_ROOT}/../storage}")" ||
      fail 1 "Could not resolve canonical local storage."
    validate_path "local storage" "${HOST_STORAGE_ROOT}"
  fi
  DOCKER_PYTHON="${HOST_REPO_ROOT}/scripts/docker_python.sh"
}

admit_repository_file() {
  local value="$1"
  local label="$2"
  local candidate lexical resolved relative
  if [[ "${value}" == /* ]]; then
    lexical="$(realpath -ms -- "${value}")" ||
      fail 2 "Could not normalize ${label}."
    if [[ "${lexical}" == "${HOST_REPO_ROOT}/"* ]]; then
      relative="${lexical#"${HOST_REPO_ROOT}/"}"
    elif [[ "${lexical}" == "${DEVELOPMENT_REPO_ROOT}/"* ]]; then
      relative="${lexical#"${DEVELOPMENT_REPO_ROOT}/"}"
    else
      fail 2 "${label} must remain inside the repository."
    fi
    validate_logical_path "${label}" "${relative}"
    candidate="${HOST_REPO_ROOT}/${relative}"
  else
    validate_logical_path "${label}" "${value}"
    candidate="${HOST_REPO_ROOT}/${value}"
  fi
  lexical="$(realpath -ms -- "${candidate}")" ||
    fail 2 "Could not normalize ${label}."
  resolved="$(realpath -e -- "${candidate}")" ||
    fail 2 "${label} does not exist in pinned commit ${REQUESTED_COMMIT}."
  [[ "${lexical}" == "${resolved}" ]] ||
    fail 2 "${label} must not traverse a symbolic link."
  [[ -f "${resolved}" && ! -L "${resolved}" ]] ||
    fail 2 "${label} is not a safe regular file."
  relative="$(realpath --relative-to="${HOST_REPO_ROOT}" -- "${resolved}")" ||
    fail 2 "Could not reduce ${label} to a repository-relative path."
  validate_logical_path "${label}" "${relative}"
  ADMITTED_HOST_PATH="${resolved}"
  ADMITTED_REPOSITORY_PATH="${relative}"
}

remote_repository_path() {
  validate_logical_path "remote repository input" "$1"
  printf '%s/%s' "${REMOTE_REPOSITORY}" "$1"
}

validate_commit() {
  [[ "$1" =~ ^[0-9a-f]{40}$ ]] || fail 2 "Git commit must be one lowercase 40-character identifier."
}

validate_run_id() {
  [[ "$1" =~ ^[A-Za-z0-9._-]+__[0-9a-f]{16}$ ]] || fail 2 "Malformed campaign-run ID: $1"
}

validate_benchmark_run_id() {
  [[ "$1" =~ ^core_scaling_transient__[0-9a-f]{16}$ ]] ||
    fail 2 "Malformed core benchmark run ID: $1"
}

validate_batch_name() {
  [[ "$1" =~ ^[A-Za-z0-9._-]+$ ]] ||
    fail 2 "Malformed campaign batch name: $1"
}

validate_case_id() {
  [[ "$1" =~ ^case_[0-9]+$ ]] || fail 2 "Malformed Generation case ID: $1"
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

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

collection_mode_argument() {
  if [[ "${KEEP_CPU_SOURCE:-false}" == true ]]; then
    printf '%s' --keep-cpu-source
  elif [[ "${DEFER_COLLECTION:-false}" == true ]]; then
    printf '%s' --defer-collection
  fi
}

remote_bash() {
  remote_bash_once "$@"
}

read_remote_home() {
  remote_bash_retryable "remote HOME resolution" "$1" <<'REMOTE'
set -euo pipefail
printf '%s\n' "${HOME}"
REMOTE
}

resolve_remote_layout() {
  ensure_execution_bootstrap
  require_command ssh "CPU login control"
  validate_host "${CPU_HOST}"
  REMOTE_HOME="$(read_remote_home "${CPU_HOST}")" ||
    fail_preserving_interrupt "$?" 1 "Could not resolve remote HOME."
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

cleanup_pinned_source() {
  if [[ "${GENERATION_WORKFLOW_PINNED_HANDOFF:-}" == 1
    && "${GENERATION_WORKFLOW_PINNED_CLEANUP_OWNER:-}" == bootstrap ]]; then
    return 0
  fi
  local container="${PINNED_SOURCE_CONTAINER}"
  [[ -n "${container}" ]] || return 0
  PINNED_SOURCE_CONTAINER=""
  local marker="${container}/.generation-workflow-source"
  if [[ ! -d "${container}" || -L "${container}" || ! -f "${marker}" || -L "${marker}" \
    || -z "${PINNED_SOURCE_PARENT}" \
    || "${container}" != "${PINNED_SOURCE_PARENT}/generation-workflow-source."* ]]; then
    generation_console_warning "refusing to remove an unverified pinned-source directory: ${container}"
    return 1
  fi
  rm -rf -- "${container}"
  if [[ -e "${container}" ]]; then
    generation_console_warning "could not remove pinned-source directory: ${container}"
    return 1
  fi
}

adopt_pinned_source() {
  local source container commit marker marker_kind marker_commit marker_development extra
  source="$(realpath -e -- "${GENERATION_WORKFLOW_PINNED_SOURCE_ROOT:-}")" ||
    fail 1 "Could not recover the exact pinned source checkout."
  container="$(realpath -e -- "${GENERATION_WORKFLOW_PINNED_SOURCE_CONTAINER:-}")" ||
    fail 1 "Could not recover the pinned source container."
  commit="${GENERATION_WORKFLOW_PINNED_COMMIT:-}"
  validate_commit "${commit}"
  [[ "${source}" == "${HOST_REPO_ROOT}" && "${source}" == "${container}/repo" \
    && -d "${source}/.git" && ! -L "${source}" && ! -L "${container}" ]] ||
    fail 1 "Pinned-source handoff does not identify the executing clean checkout."
  marker="${container}/.generation-workflow-source"
  [[ -f "${marker}" && ! -L "${marker}" ]] ||
    fail 1 "Pinned-source handoff marker is missing or unsafe."
  IFS=$'\t' read -r marker_kind marker_commit marker_development extra < "${marker}"
  [[ "${marker_kind}" == generation-workflow-source && "${marker_commit}" == "${commit}" \
    && "${marker_development}" == "${DEVELOPMENT_REPO_ROOT}" && -z "${extra:-}" ]] ||
    fail 1 "Pinned-source handoff marker is malformed or inconsistent."
  PINNED_SOURCE_CONTAINER="${container}"
  PINNED_SOURCE_PARENT="${container%/*}"
  local head status
  head="$(git -C "${source}" rev-parse HEAD)" ||
    fail 1 "Could not verify the pinned source commit."
  status="$(git --no-optional-locks -C "${source}" status --porcelain=v1 --untracked-files=all)" ||
    fail 1 "Could not verify the pinned source worktree."
  [[ "${head}" == "${commit}" && -z "${status}" ]] ||
    fail 1 "Pinned source is not the exact clean committed checkout."
  [[ -z "${REQUESTED_COMMIT}" || "${REQUESTED_COMMIT}" == "${commit}" ]] ||
    fail 1 "Requested commit differs from the pinned workflow source."
  PINNED_SOURCE_COMMIT="${commit}"
  REQUESTED_COMMIT="${commit}"
  DOCKER_PYTHON="${HOST_REPO_ROOT}/scripts/docker_python.sh"
}

materialize_pinned_source() {
  require_command git
  require_command mktemp
  require_command rm
  local head status temporary_parent container source snapshot_head snapshot_status
  head="$(git -C "${DEVELOPMENT_REPO_ROOT}" rev-parse HEAD)" ||
    fail 1 "Could not resolve Git HEAD."
  validate_commit "${head}"
  [[ -z "${REQUESTED_COMMIT}" || "${REQUESTED_COMMIT}" == "${head}" ]] ||
    fail 1 "Requested commit differs from local HEAD."
  status="$(git --no-optional-locks -C "${DEVELOPMENT_REPO_ROOT}" status --porcelain=v1 --untracked-files=all)" ||
    fail 1 "Could not inspect the local worktree for committed HEAD ${head}."
  REQUESTED_COMMIT="${head}"
  printf 'Source: committed HEAD %s\n' "${head}" >&2
  if [[ -n "${status}" ]]; then
    printf 'Local worktree: dirty; uncommitted changes ignored\n' >&2
  else
    printf 'Local worktree: clean\n' >&2
  fi
  temporary_parent="$(realpath -e -- "${TMPDIR:-/tmp}")" ||
    fail 1 "Could not resolve the temporary source parent."
  case "${temporary_parent}/" in
    "${DEVELOPMENT_REPO_ROOT}/"*|"${HOST_STORAGE_ROOT}/"*)
      temporary_parent="$(realpath -e -- /tmp)" ||
        fail 1 "Could not resolve the fallback temporary source parent."
      ;;
  esac
  validate_path "temporary source parent" "${temporary_parent}"
  [[ "${temporary_parent}/" != "${DEVELOPMENT_REPO_ROOT}/"* \
    && "${temporary_parent}/" != "${HOST_STORAGE_ROOT}/"* ]] ||
    fail 1 "Temporary source infrastructure must remain outside the repository and canonical storage."
  [[ -d "${temporary_parent}" && ! -L "${temporary_parent}" && -w "${temporary_parent}" ]] ||
    fail 1 "Temporary source parent is not a safe writable directory: ${temporary_parent}"
  container="$(mktemp -d "${temporary_parent%/}/generation-workflow-source.XXXXXXXX")" ||
    fail 1 "Could not create the pinned source container."
  PINNED_SOURCE_CONTAINER="${container}"
  PINNED_SOURCE_PARENT="${temporary_parent}"
  printf 'generation-workflow-source\t%s\t%s\n' \
    "${head}" "${DEVELOPMENT_REPO_ROOT}" > "${container}/.generation-workflow-source"
  source="${container}/repo"
  if ! git init --quiet "${source}" \
    || ! git -C "${source}" fetch --quiet --depth=1 --no-tags \
      "${DEVELOPMENT_REPO_ROOT}" "${head}" \
    || ! git -C "${source}" -c advice.detachedHead=false checkout --quiet --detach "${head}"; then
    cleanup_pinned_source || true
    fail 1 "Could not materialize the exact committed Generation source."
  fi
  snapshot_head="$(git -C "${source}" rev-parse HEAD)" || {
    cleanup_pinned_source || true
    fail 1 "Could not verify the materialized source commit."
  }
  snapshot_status="$(git --no-optional-locks -C "${source}" status --porcelain=v1 --untracked-files=all)" || {
    cleanup_pinned_source || true
    fail 1 "Could not verify the materialized source worktree."
  }
  [[ "${snapshot_head}" == "${head}" && -z "${snapshot_status}" ]] || {
    cleanup_pinned_source || true
    fail 1 "Materialized Generation source is not the exact clean pinned commit."
  }
  HOST_REPO_ROOT="$(realpath -e -- "${source}")"
  DOCKER_PYTHON="${HOST_REPO_ROOT}/scripts/docker_python.sh"
  PINNED_SOURCE_COMMIT="${head}"
}

resolve_local_commit() {
  if [[ -n "${PINNED_SOURCE_COMMIT}" ]]; then
    [[ -z "${REQUESTED_COMMIT}" || "${REQUESTED_COMMIT}" == "${PINNED_SOURCE_COMMIT}" ]] ||
      fail 1 "Requested commit differs from the pinned workflow source."
    REQUESTED_COMMIT="${PINNED_SOURCE_COMMIT}"
    return
  fi
  if [[ "${GENERATION_WORKFLOW_PINNED_HANDOFF:-}" == 1 ]]; then
    adopt_pinned_source
  else
    materialize_pinned_source
  fi
}

resolve_bootstrap_requested_commit() {
  REQUESTED_COMMIT=""
  local index
  for ((index=0; index<${#ORIGINAL_ARGUMENTS[@]}; index++)); do
    if [[ "${ORIGINAL_ARGUMENTS[index]}" == --git-commit ]]; then
      (( index + 1 < ${#ORIGINAL_ARGUMENTS[@]} )) ||
        fail 2 "--git-commit requires a value."
      REQUESTED_COMMIT="${ORIGINAL_ARGUMENTS[index+1]}"
      ((index += 1))
    fi
  done
  [[ -z "${REQUESTED_COMMIT}" ]] || validate_commit "${REQUESTED_COMMIT}"
}

handoff_to_pinned_workflow() {
  [[ "${GENERATION_WORKFLOW_PINNED_HANDOFF:-}" != 1 ]] || return 0
  local workflow="${HOST_REPO_ROOT}/scripts/generation_workflow.sh"
  [[ -x "${workflow}" && ! -L "${workflow}" ]] ||
    fail 1 "Pinned Generation workflow is missing or unsafe: ${workflow}"
  local workflow_status
  if env \
    GENERATION_WORKFLOW_PINNED_HANDOFF=1 \
    GENERATION_WORKFLOW_PINNED_CLEANUP_OWNER=bootstrap \
    GENERATION_WORKFLOW_PINNED_SOURCE_ROOT="${HOST_REPO_ROOT}" \
    GENERATION_WORKFLOW_PINNED_SOURCE_CONTAINER="${PINNED_SOURCE_CONTAINER}" \
    GENERATION_WORKFLOW_PINNED_COMMIT="${PINNED_SOURCE_COMMIT}" \
    GENERATION_WORKFLOW_DEVELOPMENT_REPO_ROOT="${DEVELOPMENT_REPO_ROOT}" \
    GENERATION_WORKFLOW_STORAGE_ROOT="${HOST_STORAGE_ROOT}" \
    "${workflow}" "${ORIGINAL_ARGUMENTS[@]}"; then
    workflow_status=0
  else
    workflow_status=$?
  fi
  cleanup_pinned_source || true
  trap - EXIT
  exit "${workflow_status}"
}

background_active_arguments() {
  BACKGROUND_ACTIVE_ARGUMENTS=()
  command -v tmux >/dev/null 2>&1 || return 0
  local sessions session
  sessions="$(tmux list-sessions -F '#S' 2>/dev/null || true)"
  while IFS= read -r session; do
    [[ -n "${session}" ]] || continue
    BACKGROUND_ACTIVE_ARGUMENTS+=(--active-tmux-session "${session}")
  done <<< "${sessions}"
}

resolve_background_host_runtime() {
  local require_clean="${1:-false}"
  [[ "${require_clean}" == true || "${require_clean}" == false ]] ||
    fail 2 "Internal background clean-check selector is invalid."
  resolve_bootstrap_requested_commit
  resolve_host_layout
  local head status
  head="$(git -C "${DEVELOPMENT_REPO_ROOT}" rev-parse HEAD)" ||
    fail 1 "Could not resolve the background workflow source commit."
  validate_commit "${head}"
  [[ -z "${REQUESTED_COMMIT}" || "${REQUESTED_COMMIT}" == "${head}" ]] ||
    fail 1 "Requested commit differs from local HEAD."
  if [[ "${require_clean}" == true ]]; then
    status="$(git --no-optional-locks -C "${DEVELOPMENT_REPO_ROOT}" status \
      --porcelain=v1 --untracked-files=all)" ||
      fail 1 "Could not verify the stable background workflow checkout."
    [[ -z "${status}" ]] ||
      fail 1 "--background requires the stable host checkout to be clean and committed."
  fi
  REQUESTED_COMMIT="${head}"
  PINNED_SOURCE_COMMIT="${head}"
  HOST_REPO_ROOT="${DEVELOPMENT_REPO_ROOT}"
  DOCKER_PYTHON="${DEVELOPMENT_REPO_ROOT}/scripts/docker_python.sh"
  [[ -x "${DEVELOPMENT_REPO_ROOT}/scripts/generation_workflow.sh" \
    && ! -L "${DEVELOPMENT_REPO_ROOT}/scripts/generation_workflow.sh" \
    && -x "${DOCKER_PYTHON}" && ! -L "${DOCKER_PYTHON}" ]] ||
    fail 1 "Stable host workflow or canonical Docker Python runner is missing or unsafe."
  resolve_local_storage
  resolve_local_python
}

background_host_paths_json() {
  local host_name
  host_name="$(hostname -f 2>/dev/null || hostname)"
  printf '%s\n%s\n%s\n%s\n' \
    "${DEVELOPMENT_REPO_ROOT}/scripts/generation_workflow.sh" \
    "${DEVELOPMENT_REPO_ROOT}/scripts/docker_python.sh" \
    "${LOCAL_STORAGE_ROOT}" "${host_name}" |
    local_python -c 'import json, sys
values = [line.rstrip("\n") for line in sys.stdin]
if len(values) != 4 or any(not value for value in values):
    raise SystemExit("background host paths are incomplete")
print(json.dumps(dict(zip(("stable_script", "docker_python", "storage_root", "host"), values, strict=True)), separators=(",", ":"), sort_keys=True))'
}

launch_background_workflow() {
  [[ "${GENERATION_WORKFLOW_BACKGROUND_CHILD:-}" != 1 ]] ||
    fail 2 "A background workflow child cannot create another tmux session."
  require_command tmux "background workflow execution"
  local subcommand="${ORIGINAL_ARGUMENTS[0]}" background_count=0 argument
  [[ "${subcommand}" == run ]] ||
    fail 2 "--background is supported only by run CONFIG."
  local -a child_arguments=()
  local has_commit=false has_cpu_host=false has_remote_root=false
  for argument in "${ORIGINAL_ARGUMENTS[@]}"; do
    if [[ "${argument}" == --background ]]; then
      background_count=$((background_count + 1))
      continue
    fi
    child_arguments+=("${argument}")
    [[ "${argument}" != --git-commit ]] || has_commit=true
    [[ "${argument}" != --cpu-host ]] || has_cpu_host=true
    [[ "${argument}" != --remote-root ]] || has_remote_root=true
  done
  (( background_count == 1 )) || fail 2 "Specify --background exactly once."
  if [[ " ${child_arguments[*]} " == *' --defer-collection '* \
    && " ${child_arguments[*]} " == *' --keep-cpu-source '* ]]; then
    fail 2 "--defer-collection cannot be combined with --keep-cpu-source."
  fi
  resolve_background_host_runtime true
  if [[ "${has_commit}" != true ]]; then
    child_arguments+=(--git-commit "${REQUESTED_COMMIT}")
  fi
  if [[ "${subcommand}" == run ]]; then
    CPU_HOST="${GENERATION_CPU_HOST:-}"
    REMOTE_ROOT=""
    local index
    for ((index=0; index<${#child_arguments[@]}; index++)); do
      case "${child_arguments[index]}" in
        --cpu-host) CPU_HOST="${child_arguments[index+1]:-}" ;;
        --remote-root) REMOTE_ROOT="${child_arguments[index+1]:-}" ;;
      esac
    done
    ensure_execution_bootstrap
    resolve_remote_layout
    [[ "${has_cpu_host}" == true ]] || child_arguments+=(--cpu-host "${CPU_HOST}")
    [[ "${has_remote_root}" == true ]] || child_arguments+=(--remote-root "${REMOTE_ROOT}")
  fi
  background_active_arguments
  local host_paths session_json record status session_id tmux_name source_commit log_path command_path
  host_paths="$(background_host_paths_json)" || fail 1 "Could not encode background host paths."
  session_json="$(local_cli create-background-session \
    --source-commit "${REQUESTED_COMMIT}" --storage-root "${LOCAL_STORAGE_ROOT}" \
    --host-paths-json "${host_paths}" "${BACKGROUND_ACTIVE_ARGUMENTS[@]}" \
    -- "${child_arguments[@]}")" || fail 1 "Could not create durable background session metadata."
  record="$(printf '%s' "${session_json}" | local_python -c 'import json, sys
value = json.load(sys.stdin)
keys = ("status", "workflow_session_id", "tmux_session_name", "source_commit", "log_path", "command_path", "host")
print("\t".join(str(value[key]) for key in keys))')" ||
    fail 1 "Could not decode background session metadata."
  local session_host
  IFS=$'\t' read -r status session_id tmux_name source_commit log_path command_path session_host <<< "${record}"
  if [[ "${status}" == reused ]]; then
    printf 'BACKGROUND REUSED\nworkflow_session_id=%s\ntmux_session=%s\nhost=%s\nsource_commit=%s\nlog=%s\n\nAttach:\n  tmux attach-session -t %q\n\nStatus:\n  %q background-status %q\n' \
      "${session_id}" "${tmux_name}" "${session_host}" "${source_commit}" \
      "${log_path}" "${tmux_name}" \
      "${DEVELOPMENT_REPO_ROOT}/scripts/generation_workflow.sh" "${session_id}"
    exit 3
  fi
  [[ "${status}" == created && -x "${command_path}" ]] ||
    fail 1 "Created background command is missing or unsafe: ${command_path}"
  local quoted_command
  printf -v quoted_command '%q' "${command_path}"
  if ! tmux new-session -d -s "${tmux_name}" "${quoted_command}"; then
    local_cli complete-background-session "${session_id}" --exit-code 1 \
      --storage-root "${LOCAL_STORAGE_ROOT}" >/dev/null 2>&1 || true
    fail 1 "tmux could not start the durable background workflow session."
  fi
  if ! tmux has-session -t "=${tmux_name}" 2>/dev/null; then
    background_active_arguments
    local completion_json completion_record completion_state completion_exit completion_stage
    completion_json="$(local_cli inspect-background-session "${session_id}" \
      --storage-root "${LOCAL_STORAGE_ROOT}" "${BACKGROUND_ACTIVE_ARGUMENTS[@]}")" ||
      fail 1 "tmux exited before its durable workflow result could be inspected."
    completion_record="$(printf '%s' "${completion_json}" | local_python -c 'import json, sys
value = json.load(sys.stdin)
fields = (value["workflow_state"], value["exit_code"], value["final_stage"])
print("\t".join("-" if item is None else str(item).replace("\t", " ").replace("\n", " ") for item in fields))')" ||
      fail 1 "Could not decode the immediate background workflow result."
    IFS=$'\t' read -r completion_state completion_exit completion_stage <<< "${completion_record}"
    case "${completion_state}" in
      completed)
        printf 'BACKGROUND COMPLETED
workflow_session_id=%s
tmux_session=%s
host=%s
source_commit=%s
exit_code=0
final_stage=%s
log=%s
' \
          "${session_id}" "${tmux_name}" "${session_host}" "${source_commit}" \
          "${completion_stage}" "${log_path}"
        return 0
        ;;
      failed)
        validate_nonnegative "background exit code" "${completion_exit}"
        printf 'BACKGROUND FAILED
workflow_session_id=%s
tmux_session=%s
host=%s
source_commit=%s
exit_code=%s
final_stage=%s
log=%s
' \
          "${session_id}" "${tmux_name}" "${session_host}" "${source_commit}" \
          "${completion_exit}" "${completion_stage}" "${log_path}" >&2
        return "${completion_exit}"
        ;;
      *) fail 1 "tmux returned without an active session or a durable terminal workflow result." ;;
    esac
  fi
  local pane_pid
  pane_pid="$(tmux display-message -p -t "=${tmux_name}" '#{pane_pid}' 2>/dev/null || true)"
  [[ -n "${pane_pid}" ]] || pane_pid=unavailable
  printf 'BACKGROUND STARTED\nworkflow_session_id=%s\ntmux_session=%s\nhost=%s\nsource_commit=%s\npid=%s\nlog=%s\n\nAttach:\n  tmux attach-session -t %q\n\nDetach without stopping:\n  press Ctrl+B, then D\n\nStatus:\n  %q background-status %q\n\nFollow log:\n  tail -n 100 -F %q\n\nThe workflow survives terminal/SSH disconnection.\nIt does not survive a reboot of %s; rerun the same config afterwards.\n' \
    "${session_id}" "${tmux_name}" "${session_host}" "${source_commit}" \
    "${pane_pid}" "${log_path}" "${tmux_name}" \
    "${DEVELOPMENT_REPO_ROOT}/scripts/generation_workflow.sh" "${session_id}" \
    "${log_path}" "${session_host}"
}

background_status_command() {
  (( ${#ORIGINAL_ARGUMENTS[@]} == 2 )) || fail 2 "background-status requires one workflow-session ID."
  resolve_background_host_runtime
  background_active_arguments
  local session_json record
  session_json="$(local_cli inspect-background-session "${ORIGINAL_ARGUMENTS[1]}" \
    --storage-root "${LOCAL_STORAGE_ROOT}" "${BACKGROUND_ACTIVE_ARGUMENTS[@]}")" ||
    fail 1 "Could not inspect background workflow session."
  record="$(printf '%s' "${session_json}" | local_python -c 'import json, sys
value = json.load(sys.stdin)
def field(name):
    item = value[name]
    if isinstance(item, list):
        item = ",".join(str(entry) for entry in item) or "-"
    if item is None:
        item = "-"
    return str(item).replace("\t", " ").replace("\n", " ")
keys = ("workflow_session_id", "source_commit", "subcommand", "tmux_session_name", "tmux_active", "workflow_state", "exit_code", "started_at", "ended_at", "campaign_run_ids", "benchmark_run_ids", "final_stage", "log_path")
print("\t".join(field(key) for key in keys))')" || fail 1 "Could not decode background status."
  local session_id source_commit subcommand tmux_name tmux_active state exit_code started ended campaigns benchmarks stage log_path
  IFS=$'\t' read -r session_id source_commit subcommand tmux_name tmux_active state exit_code started ended campaigns benchmarks stage log_path <<< "${record}"
  printf 'workflow_session_id=%s\nsource_commit=%s\nsubcommand=%s\ntmux_session=%s\ntmux_active=%s\nworkflow_state=%s\nexit_code=%s\nstarted_at=%s\nended_at=%s\ncampaign_run_ids=%s\nbenchmark_run_ids=%s\ncurrent_or_final_stage=%s\nlog=%s\n' \
    "${session_id}" "${source_commit}" "${subcommand}" "${tmux_name}" \
    "${tmux_active}" "${state}" "${exit_code}" "${started}" "${ended}" \
    "${campaigns}" "${benchmarks}" "${stage}" "${log_path}"
  if [[ "${tmux_active}" == True || "${tmux_active}" == true ]]; then
    printf 'Attach:\n  tmux attach-session -t %q\n' "${tmux_name}"
  else
    printf 'Follow log:\n  tail -n 100 -F %q\n' "${log_path}"
  fi
}

background_list_command() {
  (( ${#ORIGINAL_ARGUMENTS[@]} == 1 )) || fail 2 "background-list accepts no arguments."
  resolve_background_host_runtime
  background_active_arguments
  local sessions_json
  sessions_json="$(local_cli list-background-sessions --storage-root "${LOCAL_STORAGE_ROOT}" \
    "${BACKGROUND_ACTIVE_ARGUMENTS[@]}")" || fail 1 "Could not list background workflow sessions."
  printf '%s' "${sessions_json}" | local_python -c 'import json, sys
sessions = json.load(sys.stdin)["sessions"]
if not sessions:
    print("No background workflow sessions.")
else:
    print("workflow_session_id\tstate\tsubcommand\tstarted_at\ttmux_session")
    for item in sessions:
        print("\t".join((item["workflow_session_id"], item["workflow_state"], item["subcommand"], item["started_at"], item["tmux_session_name"])))'
}

resolve_workflow_campaigns() {
  resolve_local_python
  local record kind extra configured_cpu_host configured_scheduler configured_partition
  local configured_cores_per_node configured_python_module configured_comsol_module
  local configured_python_executable configured_comsol_executable
  record="$(local_cli list-campaigns --workflow |
    local_python -c 'import json, sys
value = json.load(sys.stdin)
workflow = value["workflow"]
site = value["shared_execution_site"]
fields = (
    workflow["technical_runtime_smoke"]["stationary"]["repository_path"],
    workflow["technical_runtime_smoke"]["transient"]["repository_path"],
    workflow["primary"]["stationary"]["repository_path"],
    workflow["primary"]["transient"]["repository_path"],
    site["cpu_host"], site["scheduler"], site["partition"], str(site["cores_per_node"]),
    site["python_module"], site["comsol_module"],
    site["python_executable"], site["comsol_executable"],
)
if any("\t" in str(item) or "\n" in str(item) or "\r" in str(item) for item in fields):
    raise SystemExit("workflow catalog contains unsafe shell transport text")
print("\t".join(("workflow", *(str(item) for item in fields))))')" ||
    fail 1 "Could not resolve the unique configured workflow campaigns."
  IFS=$'\t' read -r kind STATIONARY_SMOKE_CAMPAIGN_PATH \
    TRANSIENT_SMOKE_CAMPAIGN_PATH STATIONARY_PRIMARY_CAMPAIGN_PATH \
    TRANSIENT_PRIMARY_CAMPAIGN_PATH configured_cpu_host configured_scheduler \
    configured_partition configured_cores_per_node configured_python_module \
    configured_comsol_module configured_python_executable \
    configured_comsol_executable extra <<< "${record}"
  [[ "${kind}" == workflow && -z "${extra:-}" ]] ||
    fail 1 "Malformed workflow campaign catalog record."
  admit_repository_file "${STATIONARY_SMOKE_CAMPAIGN_PATH}" "stationary technical-smoke campaign"
  STATIONARY_SMOKE_CAMPAIGN_HOST_PATH="${ADMITTED_HOST_PATH}"
  STATIONARY_SMOKE_CAMPAIGN_PATH="${ADMITTED_REPOSITORY_PATH}"
  admit_repository_file "${TRANSIENT_SMOKE_CAMPAIGN_PATH}" "transient technical-smoke campaign"
  TRANSIENT_SMOKE_CAMPAIGN_HOST_PATH="${ADMITTED_HOST_PATH}"
  TRANSIENT_SMOKE_CAMPAIGN_PATH="${ADMITTED_REPOSITORY_PATH}"
  admit_repository_file "${STATIONARY_PRIMARY_CAMPAIGN_PATH}" "stationary production campaign"
  STATIONARY_PRIMARY_CAMPAIGN_HOST_PATH="${ADMITTED_HOST_PATH}"
  STATIONARY_PRIMARY_CAMPAIGN_PATH="${ADMITTED_REPOSITORY_PATH}"
  admit_repository_file "${TRANSIENT_PRIMARY_CAMPAIGN_PATH}" "transient production campaign"
  TRANSIENT_PRIMARY_CAMPAIGN_HOST_PATH="${ADMITTED_HOST_PATH}"
  TRANSIENT_PRIMARY_CAMPAIGN_PATH="${ADMITTED_REPOSITORY_PATH}"
  [[ -n "${CPU_HOST}" ]] || CPU_HOST="${configured_cpu_host}"
  [[ -n "${SCHEDULER_KIND}" ]] || SCHEDULER_KIND="${configured_scheduler}"
  [[ -n "${PARTITION}" ]] || PARTITION="${configured_partition}"
  [[ -n "${CORES_PER_NODE}" ]] || CORES_PER_NODE="${configured_cores_per_node}"
  [[ -n "${PYTHON_MODULE}" ]] || PYTHON_MODULE="${configured_python_module}"
  [[ -n "${COMSOL_MODULE}" ]] || COMSOL_MODULE="${configured_comsol_module}"
  [[ -n "${PYTHON_EXECUTABLE}" ]] || PYTHON_EXECUTABLE="${configured_python_executable}"
  [[ -n "${COMSOL_EXECUTABLE}" ]] || COMSOL_EXECUTABLE="${configured_comsol_executable}"
}

resolve_configured_resources() {
  resolve_local_python
  local validation_mode="${1:-inspect}"
  local -a validation_arguments=(validate-config "${CAMPAIGN_CONFIG_PATH}")
  case "${validation_mode}" in
    executable) ;;
    inspect) validation_arguments+=(--allow-incomplete) ;;
    *) fail 2 "Unsupported campaign configuration resolution mode: ${validation_mode}" ;;
  esac
  local record kind configured_cores_per_case configured_wall_time
  local configured_cores_per_node configured_max_admission_cases configured_poll_interval
  local configured_max_running configured_cpu_host configured_scheduler
  local configured_partition configured_python_module configured_comsol_module
  local configured_python_executable configured_comsol_executable extra
  record="$(local_cli "${validation_arguments[@]}" |
    local_python -c 'import json, sys
value = json.load(sys.stdin)
resources = value["execution_resources"]
cluster = resources["cluster"]
submission = resources["submission"]
site = resources["site"]
wall = cluster.get("wall_time")
max_running = submission.get("max_running_cases")
counts = tuple(value["counts"].values())
uniform_count = counts[0] if counts and len(set(counts)) == 1 else "-"
fields = (
    value["campaign_purpose"], cluster["cores_per_case"],
    "-" if wall is None else wall, cluster["cores_per_node"],
    submission["max_admission_cases"], submission["poll_interval_seconds"],
    "-" if max_running is None else max_running,
    site["cpu_host"], site["scheduler"], site["partition"],
    site["python_module"], site["comsol_module"],
    site["python_executable"], site["comsol_executable"],
    uniform_count, sum(counts),
)
if any("\t" in str(item) or "\n" in str(item) or "\r" in str(item) for item in fields):
    raise SystemExit("execution configuration contains unsafe shell transport text")
print("\t".join(("execution", *(str(item) for item in fields))))')" ||
    fail 1 "Could not resolve configured campaign execution."
  IFS=$'\t' read -r kind CAMPAIGN_PURPOSE configured_cores_per_case \
    configured_wall_time configured_cores_per_node configured_max_admission_cases \
    configured_poll_interval configured_max_running configured_cpu_host \
    configured_scheduler configured_partition configured_python_module \
    configured_comsol_module configured_python_executable \
    configured_comsol_executable CONFIGURED_UNIFORM_CASE_COUNT \
    CONFIGURED_TOTAL_CASE_COUNT extra <<< "${record}"
  [[ "${kind}" == execution && -z "${extra:-}" ]] ||
    fail 1 "Malformed configured execution record."
  validate_positive "configured cores_per_case" "${configured_cores_per_case}"
  validate_positive "configured cores_per_node" "${configured_cores_per_node}"
  validate_positive "configured max_admission_cases" "${configured_max_admission_cases}"
  validate_positive "configured poll_interval_seconds" "${configured_poll_interval}"
  validate_nonnegative "configured total case count" "${CONFIGURED_TOTAL_CASE_COUNT}"
  [[ "${configured_max_running}" == - ]] ||
    validate_positive "configured max_running_cases" "${configured_max_running}"
  [[ "${configured_wall_time}" != - ]] || configured_wall_time=""
  [[ -n "${CPU_HOST}" ]] || CPU_HOST="${configured_cpu_host}"
  SCHEDULER_KIND="${configured_scheduler}"
  PARTITION="${configured_partition}"
  CORES_PER_NODE="${configured_cores_per_node}"
  CORES_PER_CASE="${configured_cores_per_case}"
  MAX_ADMISSION_CASES="${configured_max_admission_cases}"
  MAX_RUNNING_CASES="${configured_max_running}"
  STATUS_POLL_SECONDS="${configured_poll_interval}"
  WALL_TIME="${configured_wall_time}"
  PYTHON_MODULE="${configured_python_module}"
  COMSOL_MODULE="${configured_comsol_module}"
  PYTHON_EXECUTABLE="${configured_python_executable}"
  COMSOL_EXECUTABLE="${configured_comsol_executable}"
}

ensure_execution_bootstrap() {
  if [[ -z "${PYTHON_MODULE}" || -z "${COMSOL_MODULE}" || -z "${PYTHON_EXECUTABLE}" \
    || -z "${COMSOL_EXECUTABLE}" || -z "${CPU_HOST}" || -z "${SCHEDULER_KIND}" \
    || -z "${PARTITION}" || -z "${CORES_PER_NODE}" ]]; then
    resolve_workflow_campaigns
  fi
  [[ "${SCHEDULER_KIND}" == slurm ]] ||
    fail 2 "The maintained remote workflow requires configured scheduler=slurm."
  validate_positive "configured cores_per_node" "${CORES_PER_NODE}"
}


resolve_campaign() {
  admit_repository_file "$1" "campaign config"
  CAMPAIGN_CONFIG_PATH="${ADMITTED_HOST_PATH}"
  CAMPAIGN_RELATIVE_PATH="${ADMITTED_REPOSITORY_PATH}"
}

resolve_pilot_contract() {
  [[ "${CAMPAIGN_PURPOSE}" == pilot_check \
    && "${CONFIGURED_UNIFORM_CASE_COUNT}" != - \
    && -n "${CONFIGURED_UNIFORM_CASE_COUNT}" ]] ||
    fail 2 "pilot-check requires a dedicated campaign with uniform cases per material."
  validate_positive "configured pilot cases per material" "${CONFIGURED_UNIFORM_CASE_COUNT}"
  validate_positive "configured pilot total" "${CONFIGURED_TOTAL_CASE_COUNT}"
  (( CONFIGURED_TOTAL_CASE_COUNT % CONFIGURED_UNIFORM_CASE_COUNT == 0 )) ||
    fail 2 "Pilot total must be divisible by its uniform cases-per-material count."
  PILOT_MODE=true
}

validate_resources() {
  validate_positive "configured cores_per_case" "${CORES_PER_CASE}"
  validate_positive "configured cores_per_node" "${CORES_PER_NODE}"
  validate_positive "configured max_admission_cases" "${MAX_ADMISSION_CASES}"
  validate_positive "configured poll_interval_seconds" "${STATUS_POLL_SECONDS}"
  [[ "${MAX_RUNNING_CASES}" == - ]] ||
    validate_positive "configured max_running_cases" "${MAX_RUNNING_CASES}"
  (( CORES_PER_CASE <= CORES_PER_NODE )) ||
    fail 2 "cores_per_case exceeds configured site cores_per_node."
}

print_layout() {
  printf 'CPU host: %s\nRemote HOME: %s\nRepository: %s\n'     "${CPU_HOST}" "${REMOTE_HOME}" "${REMOTE_REPOSITORY}"
  printf 'Persistent storage: %s\nVenv: %s\nRepository source: %s\nExact commit: %s\n' \
    "${REMOTE_STORAGE_ROOT}" "${REMOTE_VENV}" "${CPU_BOOTSTRAP_REPOSITORY_URL}" \
    "${REQUESTED_COMMIT}"
  printf 'Modules: %s, %s\n' "${PYTHON_MODULE}" "${COMSOL_MODULE}"
}

verify_remote_setup() {
  resolve_remote_layout
  local setup_identity
  printf -v setup_identity '%s\t%s\t%s\t%s\t%s\t%s\t%s' \
    "${CPU_HOST}" "${REMOTE_REPOSITORY}" "${REMOTE_STORAGE_ROOT}" \
    "${REMOTE_VENV}" "${REQUESTED_COMMIT}" "${PYTHON_MODULE}" "${PYTHON_EXECUTABLE}"
  [[ "${REMOTE_SETUP_IDENTITY}" != "${setup_identity}" ]] || return 0
  if remote_bash_retryable "remote setup verification" "${CPU_HOST}" \
    "${REMOTE_REPOSITORY}" "${REMOTE_STORAGE_ROOT}" "${REMOTE_VENV}" \
    "${REQUESTED_COMMIT}" "${CPU_BOOTSTRAP_REPOSITORY_URL}" "${PYTHON_MODULE}" \
    "${PYTHON_EXECUTABLE}" <<'REMOTE'
set -euo pipefail
repository="$1"; storage="$2"; venv="$3"; commit="$4"; repository_url="$5"
python_module="$6"; python_executable="$7"
"${repository}/scripts/generation_cpu_login_preflight.sh" \
  "${repository}" "${storage}" "${venv}" "${commit}" "${repository_url}" \
  "${python_module}" "${python_executable}"
REMOTE
  then
    REMOTE_SETUP_IDENTITY="${setup_identity}"
  else
    return $?
  fi
}

verify_remote_setup_for_output() {
  if [[ "${HUMAN_WORKFLOW_MODE}" == true ]]; then
    verify_remote_setup >/dev/null
  else
    verify_remote_setup >&2
  fi
}


setup_cpu() {
  resolve_local_commit
  resolve_remote_layout
  print_layout
  printf 'Mode: %s\n' "$([[ "${EXECUTE_SETUP}" == true ]] && printf execute || printf dry-run)"
  print_command mkdir -p "${REMOTE_ROOT}" "${REMOTE_STORAGE_ROOT}"
  print_command git clone --no-checkout "${CPU_BOOTSTRAP_REPOSITORY_URL}" "${REMOTE_REPOSITORY}"
  print_command "${REMOTE_VENV}/bin/python" -m src.generation.cli.cli_generation \
    assert-shared-setup-idle --storage-root "${REMOTE_STORAGE_ROOT}"
  print_command git -C "${REMOTE_REPOSITORY}" fetch origin "${REQUESTED_COMMIT}"
  print_command git -C "${REMOTE_REPOSITORY}" checkout --detach "${REQUESTED_COMMIT}"
  print_command module load "${PYTHON_MODULE}"
  print_command "${PYTHON_EXECUTABLE}" -m venv "${REMOTE_VENV}"
  print_command "${REMOTE_VENV}/bin/python" -m pip install -e "${REMOTE_REPOSITORY}[generation-cpu]"
  print_command module load "${COMSOL_MODULE}"
  if [[ "${EXECUTE_SETUP}" != true ]]; then
    printf 'Dry run: no remote files or jobs were created.\n'
    return
  fi
  remote_bash "${CPU_HOST}" \
    "${REMOTE_ROOT}" "${REMOTE_REPOSITORY}" "${REMOTE_STORAGE_ROOT}" \
    "${REMOTE_VENV}" "${REQUESTED_COMMIT}" "${CPU_BOOTSTRAP_REPOSITORY_URL}" \
    "${PYTHON_MODULE}" "${COMSOL_MODULE}" "${PYTHON_EXECUTABLE}" \
    "${COMSOL_EXECUTABLE}" <<'REMOTE'
set -euo pipefail
root="$1"; repository="$2"; storage="$3"; venv="$4"; commit="$5"; repository_url="$6"
python_module="$7"; comsol_module="$8"; python_executable="$9"; comsol_executable="${10}"
setup_require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    printf 'CPU login prerequisite missing: %s (blocks setup).\n' "$1" >&2
    exit 1
  }
}
setup_fail() {
  printf 'CPU setup refused: %s\n' "$1" >&2
  exit 1
}
for name in git stat module; do setup_require_command "${name}"; done
[[ "${root}" != / && "${root}" != "${HOME}" ]] || setup_fail "remote root is unsafe"
parent="${root}"
while [[ ! -e "${parent}" ]]; do parent="$(dirname "${parent}")"; done
[[ -d "${parent}" && ! -L "${parent}" && "$(stat -c %u "${parent}")" -eq "${UID}" && -w "${parent}" ]] ||
  setup_fail "remote root parent is not a writable owned directory"
if [[ ! -e "${root}" ]]; then
  mkdir -p -- "${root}" "${storage}"
elif [[ -d "${root}" && ! -L "${root}" ]]; then
  mkdir -p -- "${storage}"
else
  setup_fail "remote root is not a directory"
fi
[[ -d "${root}" && ! -L "${root}" && -d "${storage}" && ! -L "${storage}" ]] ||
  setup_fail "remote layout contains an unsafe root or storage path"
shopt -s nullglob dotglob
root_entries=("${root}"/*)
shopt -u nullglob dotglob
for entry in "${root_entries[@]}"; do
  case "${entry}" in
    "${repository}"|"${storage}"|"${venv}") ;;
    *) setup_fail "remote root contains an unsupported top-level entry: ${entry}" ;;
  esac
done
repository_ready=false
if [[ -e "${repository}" ]]; then
  [[ -d "${repository}/.git" && ! -L "${repository}" ]] ||
    setup_fail "existing repository is unsafe or incomplete"
  [[ -z "$(git -C "${repository}" status --porcelain)" ]] ||
    setup_fail "existing repository has uncommitted changes"
  [[ "$(git -C "${repository}" remote get-url origin)" == "${repository_url}" ]] ||
    setup_fail "existing repository origin differs from the configured source"
  repository_ready=true
fi
venv_ready=false
if [[ -e "${venv}" ]]; then
  [[ -d "${venv}" && ! -L "${venv}" && -x "${venv}/bin/python" ]] ||
    setup_fail "existing virtual environment is unsafe or incomplete"
  venv_ready=true
fi
if [[ "${venv_ready}" == true && "${repository_ready}" != true ]]; then
  setup_fail "existing virtual environment has no matching repository"
fi
if [[ "${repository_ready}" == true && "${venv_ready}" == true ]]; then
  fresh_installation=false
  if ! module load "${python_module}"; then
    printf 'CPU login prerequisite failed: Python module %s (blocks setup).\n' \
      "${python_module}" >&2
    exit 1
  fi
  if ! "${venv}/bin/python" -m src.generation.cli.cli_generation \
    assert-shared-setup-idle --storage-root "${storage}"; then
    exit 1
  fi
else
  fresh_installation=true
  shopt -s nullglob dotglob
  storage_entries=("${storage}"/*)
  shopt -u nullglob dotglob
  (( ${#storage_entries[@]} == 0 )) ||
    setup_fail "incomplete installation cannot be repaired while persistent storage is non-empty"
fi
if [[ "${repository_ready}" != true ]]; then
  git clone --no-checkout "${repository_url}" "${repository}"
fi
if [[ "${fresh_installation}" == false \
  && "$(git -C "${repository}" rev-parse HEAD)" == "${commit}" ]]; then
  setup_changed=false
else
  git -C "${repository}" fetch origin "${commit}"
  git -C "${repository}" cat-file -e "${commit}^{commit}"
  git -C "${repository}" checkout --detach "${commit}"
  setup_changed=true
fi
if [[ "${fresh_installation}" == true ]]; then
  if ! module load "${python_module}"; then
    printf 'CPU login prerequisite failed: Python module %s (blocks setup).\n' \
      "${python_module}" >&2
    exit 1
  fi
fi
setup_require_command "${python_executable}"
if [[ "${fresh_installation}" == true ]]; then
  "${python_executable}" -m venv "${venv}"
fi
if [[ "${fresh_installation}" == true || "${setup_changed}" == true ]]; then
  "${venv}/bin/python" -m pip install -e "${repository}[generation-cpu]"
fi
if ! module load "${comsol_module}"; then
  printf 'CPU login prerequisite failed: COMSOL module %s (blocks setup capability check).\n' \
    "${comsol_module}" >&2
  exit 1
fi
setup_require_command "${comsol_executable}"
"${comsol_executable}" -version 2>&1
printf 'CPU setup complete: %s\n' "${root}"
REMOTE
  verify_remote_setup
}

remote_plan_submit() {
  local operation="$1"
  verify_remote_setup_for_output || return $?
  local remote_campaign
  remote_campaign="$(remote_repository_path "${CAMPAIGN_RELATIVE_PATH}")"
  remote_bash "${CPU_HOST}" \
    "${REMOTE_REPOSITORY}" "${REMOTE_STORAGE_ROOT}" "${REMOTE_VENV}" \
    "${REQUESTED_COMMIT}" "${remote_campaign}" "${operation}" \
    "${PYTHON_MODULE}" <<'REMOTE'
set -euo pipefail
repository="$1"; storage="$2"; venv="$3"; commit="$4"; campaign="$5"
operation="$6"; python_module="$7"
module load "${python_module}"
export GENERATION_CPU_VENV="${venv}"
export STORAGE_ROOT="${storage}"
export GENERATION_GIT_COMMIT="${commit}"
cd "${repository}"
command=("${venv}/bin/python" -m src.generation.cli.cli_generation
  "${operation}" "${campaign}"
  --git-commit "${commit}" --storage-root "${storage}")
if [[ "${operation}" == submit-campaign ]]; then
  command+=(--inputs-prepared)
fi
"${command[@]}"
REMOTE
}


remote_prepare_campaign_inputs() {
  remote_plan_submit prepare-campaign-inputs
}


technical_smoke_evidence_status_cpu() {
  local campaign_argument="$1"
  local comsol_version_output="$2"
  resolve_campaign "${campaign_argument}"
  resolve_configured_resources
  resolve_remote_layout
  local remote_campaign
  remote_campaign="$(remote_repository_path "${CAMPAIGN_RELATIVE_PATH}")"
  remote_bash_retryable "technical-smoke evidence status" "${CPU_HOST}" \
    "${REMOTE_REPOSITORY}" "${REMOTE_STORAGE_ROOT}" "${REMOTE_VENV}" \
    "${remote_campaign}" "${comsol_version_output}" "${PYTHON_MODULE}" <<'REMOTE'
set -euo pipefail
repository="$1"; storage="$2"; venv="$3"; campaign="$4"
comsol_version_output="$5"; python_module="$6"
module load "${python_module}"
cd "${repository}"
"${venv}/bin/python" -m src.generation.cli.cli_generation \
  technical-smoke-evidence-status "${campaign}" --storage-root "${storage}" \
  --comsol-version-output "${comsol_version_output}"
REMOTE
}


remote_comsol_version() {
  remote_bash_retryable "remote COMSOL version query" \
    "${CPU_HOST}" "${COMSOL_MODULE}" "${COMSOL_EXECUTABLE}" <<'REMOTE'
set -euo pipefail
comsol_module="$1"; comsol_executable="$2"
if ! module load "${comsol_module}"; then
  printf 'CPU login prerequisite failed: COMSOL module %s (blocks native smoke finalization).\n' \
    "${comsol_module}" >&2
  exit 1
fi
if ! command -v "${comsol_executable}" >/dev/null 2>&1; then
  printf 'CPU login prerequisite missing: %s (blocks native smoke finalization).\n' \
    "${comsol_executable}" >&2
  exit 1
fi
if ! "${comsol_executable}" -version 2>&1; then
  printf 'CPU login prerequisite failed: COMSOL version query (blocks native smoke finalization).\n' >&2
  exit 1
fi
REMOTE
}


sync_technical_smoke_evidence() {
  local evidence="$1" campaign_argument="$2" comsol_version_output="$3"
  require_command rsync "technical-smoke evidence transfer"
  [[ "${evidence}" == "${LOCAL_STORAGE_ROOT}/"* && -f "${evidence}" && ! -L "${evidence}" ]] ||
    fail 1 "Technical-smoke evidence is outside canonical local storage."
  local relative="${evidence#"${LOCAL_STORAGE_ROOT}/"}"
  validate_transfer_path "${relative}"
  local destination="${REMOTE_STORAGE_ROOT}/${relative}"
  local temporary="${destination}.incoming.$$"
  remote_bash "${CPU_HOST}" "$(dirname "${destination}")" <<'REMOTE'
set -euo pipefail
directory="$1"
mkdir -p "${directory}"
REMOTE
  rsync -a --protect-args "${evidence}" "${CPU_HOST}:${temporary}" ||
    fail 1 "Could not transfer compact technical-smoke evidence to the CPU host."
  remote_bash "${CPU_HOST}" "${destination}" "${temporary}" <<'REMOTE'
set -euo pipefail
destination="$1"; temporary="$2"
[[ -f "${temporary}" && ! -L "${temporary}" ]]
if [[ -e "${destination}" ]]; then
  [[ -f "${destination}" && ! -L "${destination}" ]]
  if ! cmp -s "${temporary}" "${destination}"; then
    rm -f -- "${temporary}"
    printf 'Existing CPU technical-smoke evidence conflicts: %s\n' "${destination}" >&2
    exit 1
  fi
  rm -f -- "${temporary}"
else
  mv -- "${temporary}" "${destination}"
fi
REMOTE
  technical_smoke_evidence_status_cpu \
    "${campaign_argument}" "${comsol_version_output}" >/dev/null ||
    fail_preserving_interrupt "$?" 2 \
      "CPU-side technical-smoke evidence is missing, stale, or incomplete after transfer."
}

finalize_smoke_runs() {
  local stationary_run_id="$1" transient_run_id="$2"
  validate_run_id "${stationary_run_id}"
  validate_run_id "${transient_run_id}"
  resolve_local_commit
  resolve_workflow_campaigns
  resolve_local_storage
  resolve_local_python
  resolve_remote_layout
  verify_remote_setup_for_output >/dev/null
  local comsol_version steady_evidence transient_evidence smoke_children
  comsol_version="$(remote_comsol_version)"
  printf -v smoke_children "children=%s,%s" \
    "${stationary_run_id}" "${transient_run_id}"
  RUN_ID="${stationary_run_id}"
  steady_evidence="$(generation_run_with_heartbeat \
    "profile-smoke-${stationary_run_id}" 8 9 "Paired finalizer" \
    "validating steady-flow Technical Smoke evidence" \
    "current_child=${stationary_run_id}" \
    local_cli finalize-technical-smoke-evidence "${stationary_run_id}" \
      --comsol-version-output "${comsol_version}" \
      --storage-root "${LOCAL_STORAGE_ROOT}")" ||
    fail 1 "Could not finalize steady-flow Technical Smoke evidence for ${stationary_run_id}."
  steady_evidence="$(container_path_to_host "${steady_evidence}")"
  sync_technical_smoke_evidence \
    "${steady_evidence}" "${STATIONARY_SMOKE_CAMPAIGN_PATH}" "${comsol_version}"
  RUN_ID="${transient_run_id}"
  transient_evidence="$(generation_run_with_heartbeat \
    "profile-smoke-${transient_run_id}" 8 9 "Paired finalizer" \
    "validating transient-drying Technical Smoke evidence" \
    "current_child=${transient_run_id}" \
    local_cli finalize-technical-smoke-evidence "${transient_run_id}" \
      --comsol-version-output "${comsol_version}" \
      --storage-root "${LOCAL_STORAGE_ROOT}")" ||
    fail 1 "Could not finalize transient-drying Technical Smoke evidence for ${transient_run_id}."
  transient_evidence="$(container_path_to_host "${transient_evidence}")"
  sync_technical_smoke_evidence \
    "${transient_evidence}" "${TRANSIENT_SMOKE_CAMPAIGN_PATH}" "${comsol_version}"
  PAIRED_SMOKE_RECEIPT="$(generation_run_with_heartbeat \
    "paired-smoke-${RUN_PLAN_ID}" 8 9 "Paired finalizer" \
    "building and validating paired Smoke payload" "${smoke_children}" \
    local_cli finalize-real-smoke "${stationary_run_id}" "${transient_run_id}" \
      --comsol-version-output "${comsol_version}" \
      --storage-root "${LOCAL_STORAGE_ROOT}")" ||
    fail 1 "Could not atomically finalize paired Technical Smoke evidence."
  PAIRED_SMOKE_RECEIPT="$(container_path_to_host "${PAIRED_SMOKE_RECEIPT}")"
  generation_run_with_heartbeat \
    "paired-smoke-validation-${RUN_PLAN_ID}" 8 9 "Paired finalizer" \
    "validating the current paired Smoke receipt" "${smoke_children}" \
    local_cli validate-real-smoke "${PAIRED_SMOKE_RECEIPT}" \
      --storage-root "${LOCAL_STORAGE_ROOT}" >/dev/null ||
    fail 1 "Paired Technical Smoke receipt did not validate after finalization: ${PAIRED_SMOKE_RECEIPT}"
  printf 'Profile technical-smoke evidence: %s and %s\n' \
    "${steady_evidence}" "${transient_evidence}"
  printf 'Paired technical runtime diagnostic receipt: %s\n' "${PAIRED_SMOKE_RECEIPT}"
  printf 'Paired Technical Smoke children validated: %s and %s.\n' \
    "${stationary_run_id}" "${transient_run_id}"
}

launch_campaign() {
  local output observed_run_id
  output="$(remote_plan_submit submit-campaign)" ||
    fail 1 "Remote campaign submission failed."
  if [[ ${output} =~ \"campaign_run_id\"[[:space:]]*:[[:space:]]*\"([A-Za-z0-9._-]+__[0-9a-f]{16})\" ]]; then
    observed_run_id="${BASH_REMATCH[1]}"
  else
    fail 1 "Campaign submission returned no campaign-run ID."
  fi
  if [[ -n "${EXPECTED_RUN_ID:-}" && "${observed_run_id}" != "${EXPECTED_RUN_ID}" ]]; then
    fail 1 "Campaign submission identity disagrees with the common run plan."
  fi
  RUN_ID="${observed_run_id}"
  printf 'campaign_run_id=%s\n' "${RUN_ID}"
}

_remote_cli_with_transport() {
  local transport_kind="$1" operation="$2"
  shift 2
  verify_remote_setup_for_output || return $?
  validate_commit "${REQUESTED_COMMIT}"
  local -a transport=(remote_bash)
  case "${transport_kind}" in
    once) ;;
    retryable) transport=(remote_bash_retryable "${operation}") ;;
    *) fail 2 "Unsupported remote CLI transport kind: ${transport_kind}" ;;
  esac
  "${transport[@]}" "${CPU_HOST}" "${REMOTE_REPOSITORY}" \
    "${REMOTE_STORAGE_ROOT}" "${REMOTE_VENV}" "${REQUESTED_COMMIT}" \
    "${PYTHON_MODULE}" "$@" <<'REMOTE'
set -euo pipefail
repository="$1"; storage="$2"; venv="$3"; commit="$4"; python_module="$5"
shift 5
module load "${python_module}"
export GENERATION_CPU_VENV="${venv}"
export STORAGE_ROOT="${storage}"
export GENERATION_GIT_COMMIT="${commit}"
cd "${repository}"
"${venv}/bin/python" -m src.generation.cli.cli_generation "$@"
REMOTE
}

remote_cli() {
  _remote_cli_with_transport once "" "$@"
}

remote_cli_retryable() {
  local operation="$1"
  shift
  _remote_cli_with_transport retryable "${operation}" "$@"
}

remote_transfer_plan() {
  local -a arguments=(campaign-transfer-plan "${RUN_ID}" --format tsv --storage-root "${REMOTE_STORAGE_ROOT}")
  [[ "${CAMPAIGN_PARTIAL:-false}" != true ]] || arguments+=(--partial)
  remote_cli_retryable "campaign transfer-plan read" "${arguments[@]}"
}

resolve_local_storage() {
  [[ -z "${LOCAL_STORAGE_ROOT}" ]] || return 0
  require_command realpath
  LOCAL_STORAGE_ROOT="$(realpath -m -- "${HOST_STORAGE_ROOT}")"
  validate_path "local storage" "${LOCAL_STORAGE_ROOT}"
}

resolve_local_python() {
  [[ "${LOCAL_PYTHON_READY}" != true ]] || return 0
  [[ -x "${DOCKER_PYTHON}" ]] || fail 1 "Canonical Docker Python runner is not executable: ${DOCKER_PYTHON}"
  local_python -c 'import h5py, numpy, scipy, yaml; import src.generation.cli.cli_generation' ||
    fail 1 "Canonical Docker Python environment lacks required dependencies."
  LOCAL_PYTHON_READY=true
}

local_python() {
  env GENERATION_GIT_COMMIT="${PINNED_SOURCE_COMMIT}" \
    STORAGE_ROOT="${LOCAL_STORAGE_ROOT:-${HOST_STORAGE_ROOT}}" \
    "${DOCKER_PYTHON}" "$@"
}

local_cli() {
  local_python -m "${GENERATION_MODULE}" "$@"
}

local_cli_quiet() {
  local_cli "$@" >/dev/null 2>&1
}

container_path_to_host() {
  local value="$1"
  local relative
  if [[ "${value}" == /workspace/storage ]]; then
    printf '%s' "${LOCAL_STORAGE_ROOT}"
  elif [[ "${value}" == /workspace/storage/* ]]; then
    relative="${value#/workspace/storage/}"
    validate_logical_path "container storage output" "${relative}"
    printf '%s/%s' "${LOCAL_STORAGE_ROOT}" "${relative}"
  elif [[ "${value}" == /workspace/repo ]]; then
    printf '%s' "${HOST_REPO_ROOT}"
  elif [[ "${value}" == /workspace/repo/* ]]; then
    relative="${value#/workspace/repo/}"
    validate_logical_path "container repository output" "${relative}"
    printf '%s/%s' "${HOST_REPO_ROOT}" "${relative}"
  else
    printf '%s' "${value}"
  fi
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
  [[ "${CAMPAIGN_PARTIAL:-false}" != true ]] || return 1
  local -a arguments=(
    validate-published-campaign "${RUN_ID}"
    --storage-root "${LOCAL_STORAGE_ROOT}"
  )
  [[ "${CAMPAIGN_PARTIAL:-false}" != true ]] || arguments+=(--partial)
  generation_run_with_heartbeat \
    "host-publication-existing-${RUN_ID}" 6 9 "Host publication" \
    "validating existing host inventory" "" \
    local_cli_quiet "${arguments[@]}"
}

repair_existing_campaign_publication() {
  local authority
  authority="$(remote_cli_retryable "campaign transfer-authority read" \
    campaign-transfer-authority "${RUN_ID}" \
    --storage-root "${REMOTE_STORAGE_ROOT}")" || {
    local status=$?
    (( status != 130 )) || exit 130
    return 1
  }
  local_cli repair-transferred-campaign "${RUN_ID}" \
    --source-host "${CPU_HOST}" --source-storage-root "${REMOTE_STORAGE_ROOT}" \
    --authority-json "${authority}" --storage-root "${LOCAL_STORAGE_ROOT}" >/dev/null 2>&1
}

collect_campaign() {
  resolve_local_python
  resolve_remote_layout
  resolve_local_storage
  verify_remote_setup_for_output
  if gpu_publication_is_valid; then
    printf 'GPU generation publication validated and reused for %s.\n' "${RUN_ID}"
    return
  fi
  if [[ "${CAMPAIGN_PARTIAL:-false}" != true ]] && repair_existing_campaign_publication; then
    printf 'GPU generation publication receipt reconstructed from canonical CPU identity for %s.\n' "${RUN_ID}"
    return
  fi
  if [[ "${PILOT_MODE}" == true ]]; then
    remote_cli record-pilot-source-inventory "${RUN_ID}" \
      --storage-root "${REMOTE_STORAGE_ROOT}" >/dev/null ||
      fail 1 "Could not record exact pre-cleanup CPU pilot storage."
  fi
  local plan
  plan="$(remote_transfer_plan)" ||
    fail_preserving_interrupt "$?" 1 "Remote campaign is not terminally valid."
  local -a directories=()
  local kind field2 field3 field4 field5 field6 field7 extra
  while IFS=$'\t' read -r kind field2 field3 field4 field5 field6 field7 extra; do
    [[ -z "${extra:-}" ]] || fail 1 "Malformed transfer plan."
    case "${kind}" in
      campaign)
        [[ -n "${field2}" && -n "${field3}" && -n "${field4}" \
          && -n "${field5}" && -z "${field6}" && -z "${field7}" ]] ||
          fail 1 "Malformed campaign transfer-plan row."
        directories+=("${field4}")
        ;;
      batch)
        [[ -n "${field2}" && -n "${field3}" && -n "${field4}" \
          && -n "${field5}" && -n "${field6}" && -n "${field7}" ]] ||
          fail 1 "Malformed batch transfer-plan row."
        directories+=("${field5}" "${field6}" "${field7}")
        ;;
      attempt)
        [[ -n "${field2}" && -n "${field3}" && -z "${field4}" \
          && -z "${field5}" && -z "${field6}" && -z "${field7}" ]] ||
          fail 1 "Malformed attempt transfer-plan row."
        directories+=("${field3}")
        ;;
      *) fail 1 "Unknown transfer-plan row." ;;
    esac
  done <<< "${plan}"
  (( ${#directories[@]} >= 4 )) || fail 1 "Transfer plan is empty."
  local directory
  for directory in "${directories[@]}"; do validate_transfer_path "${directory}"; done
  require_command rsync "campaign transfer"
  local staging receipt directory_index=0 directory_total transfer_progress
  staging="$(local_cli create-transfer-staging "${RUN_ID}" \
    --storage-root "${LOCAL_STORAGE_ROOT}")" ||
    fail 1 "Could not create marked transfer staging."
  staging="$(container_path_to_host "${staging}")"
  printf 'Transfer staging: %s\n' "${staging}"
  directory_total="${#directories[@]}"
  for directory in "${directories[@]}"; do
    directory_index="$((directory_index + 1))"
    printf -v transfer_progress \
      "directories_completed=%s/%s\ncurrent_artifact=%s" \
      "$((directory_index - 1))" "${directory_total}" "${directory}"
    local -a rsync_arguments=(
      -a --protect-args --relative --exclude='.state/' --exclude='work/'
    )
    if [[ -d "${LOCAL_STORAGE_ROOT}/${directory}" ]]; then
      rsync_arguments+=(--link-dest="${LOCAL_STORAGE_ROOT}")
    fi
    generation_run_with_heartbeat \
      "host-transfer-${RUN_ID}-${directory_index}" 6 9 "Host publication" \
      "transferring ${directory}" "${transfer_progress}" \
      rsync "${rsync_arguments[@]}" \
        "${CPU_HOST}:${REMOTE_STORAGE_ROOT}/./${directory}" "${staging}/" ||
      fail 1 "Transfer failed; staging retained at ${staging}."
  done
  if [[ "${PILOT_MODE}" == true ]]; then
    local_cli record-pilot-staging-inventory "${RUN_ID}" --staging-root "${staging}" >/dev/null ||
      fail 1 "Could not record exact pilot transfer-staging storage."
  fi
  printf -v transfer_progress \
    "directories_completed=%s/%s" "${directory_total}" "${directory_total}"
  local -a publication_arguments=(
    publish-transferred-campaign "${RUN_ID}"
    --staging-root "${staging}" --destination-root "${LOCAL_STORAGE_ROOT}"
    --source-host "${CPU_HOST}" --source-storage-root "${REMOTE_STORAGE_ROOT}"
  )
  [[ "${CAMPAIGN_PARTIAL:-false}" != true ]] || publication_arguments+=(--partial)
  receipt="$(generation_run_with_heartbeat \
    "host-publication-validate-${RUN_ID}" 6 9 "Host publication" \
    "validating destination inventory and hashes" "${transfer_progress}" \
    local_cli "${publication_arguments[@]}")" ||
    fail 1 "GPU publication validation failed; staging retained at ${staging}."
  if [[ "${PILOT_MODE}" != true ]]; then
    local_cli cleanup-transfer-staging --campaign-run-id "${RUN_ID}" \
      --directory "${staging}" --storage-root "${LOCAL_STORAGE_ROOT}" --confirm >/dev/null
  else
    printf 'Pilot transfer staging marker retained through analysis: %s\n' "${staging}"
  fi
  printf '%s\nCPU source retained: %s:%s\n' "${receipt}" "${CPU_HOST}" "${REMOTE_STORAGE_ROOT}"
}

build_datasets() {
  resolve_local_python
  resolve_local_storage
  local -a arguments=(build-campaign-datasets "${RUN_ID}" --storage-root "${LOCAL_STORAGE_ROOT}")
  [[ "${CAMPAIGN_PARTIAL:-false}" != true ]] || arguments+=(--partial)
  generation_run_with_heartbeat \
    "dataset-packages-${RUN_ID}" 7 9 "Packages/finalizer" \
    "building Dataset packages and loader smoke evidence" "" \
    local_cli "${arguments[@]}"
}

remote_campaign_monitor() {
  remote_cli_retryable "campaign status read" campaign-status \
    "${RUN_ID}" --format monitor --max-active-cases 8 \
    --storage-root "${REMOTE_STORAGE_ROOT}"
}

read_remote_campaign_monitor() {
  local output header kind state state_signature progress_signature extra
  output="$(remote_campaign_monitor)" ||
    fail_preserving_interrupt "$?" 1 "Could not reconstruct campaign case status."
  [[ "${output}" == *$'\n'* ]] || fail 1 "Malformed campaign monitor output."
  header="${output%%$'\n'*}"
  IFS=$'\t' read -r kind state state_signature progress_signature extra <<< "${header}"
  [[ "${kind}" == campaign-monitor && "${state_signature}" =~ ^[0-9a-f]{64}$ \
    && "${progress_signature}" =~ ^[0-9a-f]{64}$ && -z "${extra:-}" ]] ||
    fail 1 "Malformed campaign monitor header."
  REMOTE_CAMPAIGN_STATE="${state}"
  REMOTE_CAMPAIGN_STATE_SIGNATURE="${state_signature}"
  REMOTE_CAMPAIGN_PROGRESS_SIGNATURE="${progress_signature}"
  REMOTE_CAMPAIGN_SUMMARY="${output#*$'\n'}"
}

remote_source_status_tsv() {
  remote_cli_retryable "campaign source-status read" campaign-source-status \
    "${RUN_ID}" --query-scheduler --include-sizes --format tsv \
    --storage-root "${REMOTE_STORAGE_ROOT}"
}

read_remote_source_status() {
  local line kind status_run campaign_state source_state bytes eligibility active extra
  line="$(remote_source_status_tsv)"
  IFS=$'\t' read -r kind status_run campaign_state source_state bytes \
    eligibility active extra <<< "${line}"
  [[ "${kind}" == source-status && "${status_run}" == "${RUN_ID}" \
    && -z "${extra:-}" ]] || fail 1 "Malformed CPU source status."
  validate_nonnegative "CPU retained bytes" "${bytes}"
  REMOTE_RUN_STATE="${campaign_state}"
  REMOTE_SOURCE_STATE="${source_state}"
  CPU_BYTES_RETAINED="${bytes}"
  CPU_BYTES_RETAINED_EXACT=true
  REMOTE_CLEANUP_ELIGIBILITY="${eligibility}"
  REMOTE_SOURCE_ACTIVE="${active}"
}

remote_workflow_monitor() {
  remote_cli_retryable "campaign resume and status snapshot" resume-campaign \
    "${RUN_ID}" --format workflow-monitor --max-active-cases 8 \
    --storage-root "${REMOTE_STORAGE_ROOT}"
}

read_remote_workflow_monitor() {
  local output campaign_header source_header tab
  local kind state state_signature progress_signature extra
  local source_kind status_run campaign_state source_state bytes eligibility active source_extra
  local -a monitor_lines=()
  output="$(remote_workflow_monitor)" ||
    fail_preserving_interrupt "$?" 1 "Could not resume and reconstruct campaign status."
  mapfile -t monitor_lines <<< "${output}"
  (( ${#monitor_lines[@]} >= 3 )) || fail 1 "Malformed combined campaign monitor output."
  campaign_header="${monitor_lines[0]}"
  source_header="${monitor_lines[1]}"
  REMOTE_CAMPAIGN_SUMMARY="$(printf "%s\n" "${monitor_lines[@]:2}")"
  tab="$(printf "\t")"
  IFS="${tab}" read -r kind state state_signature progress_signature extra <<< "${campaign_header}"
  [[ "${kind}" == campaign-monitor && "${state_signature}" =~ ^[0-9a-f]{64}$ \
    && "${progress_signature}" =~ ^[0-9a-f]{64}$ && -z "${extra:-}" ]] ||
    fail 1 "Malformed combined campaign monitor header."
  IFS="${tab}" read -r source_kind status_run campaign_state source_state bytes \
    eligibility active source_extra <<< "${source_header}"
  [[ "${source_kind}" == source-monitor && "${status_run}" == "${RUN_ID}" \
    && "${campaign_state}" == "${state}" && -z "${source_extra:-}" ]] ||
    fail 1 "Malformed combined CPU source status."
  if [[ "${bytes}" != unavailable ]]; then
    validate_nonnegative "CPU retained bytes" "${bytes}"
    CPU_BYTES_RETAINED_EXACT=true
  else
    CPU_BYTES_RETAINED_EXACT=false
  fi
  REMOTE_CAMPAIGN_STATE="${state}"
  REMOTE_CAMPAIGN_STATE_SIGNATURE="${state_signature}"
  REMOTE_CAMPAIGN_PROGRESS_SIGNATURE="${progress_signature}"
  REMOTE_RUN_STATE="${campaign_state}"
  REMOTE_SOURCE_STATE="${source_state}"
  CPU_BYTES_RETAINED="${bytes}"
  REMOTE_CLEANUP_ELIGIBILITY="${eligibility}"
  REMOTE_SOURCE_ACTIVE="${active}"
}

refresh_failure_cpu_bytes() {
  [[ "${CPU_BYTES_RETAINED_EXACT}" == true ]] && return 0
  [[ "${RUN_KIND:-}" == campaign && -n "${RUN_ID:-}" \
    && -n "${REMOTE_STORAGE_ROOT:-}" ]] || return 1
  local line kind status_run campaign_state source_state bytes eligibility active extra
  line="$(remote_cli campaign-source-status "${RUN_ID}" --include-sizes --format tsv \
    --storage-root "${REMOTE_STORAGE_ROOT}" 2>/dev/null)" || return 1
  IFS=$'\t' read -r kind status_run campaign_state source_state bytes \
    eligibility active extra <<< "${line}"
  [[ "${kind}" == source-status && "${status_run}" == "${RUN_ID}" \
    && "${bytes}" =~ ^[0-9]+$ && -z "${extra:-}" ]] || return 1
  CPU_BYTES_RETAINED="${bytes}"
  CPU_BYTES_RETAINED_EXACT=true
}

deferred_campaign_report() {
  read_remote_source_status
  printf 'campaign_run_id=%s\n' "${RUN_ID}"
  printf 'state=awaiting_collection\nsource_state=%s\nretained_cpu_bytes=%s\n' \
    "${REMOTE_SOURCE_STATE}" "${CPU_BYTES_RETAINED}"
  printf 'Resume collection with the same config:\n'
  local -a continuation_arguments=(
    "${HOST_REPO_ROOT}/scripts/generation_workflow.sh" run
    "${RUN_CONFIG_ARGUMENT}" --cpu-host "${CPU_HOST}"
    --remote-root "${REMOTE_ROOT}" --git-commit "${REQUESTED_COMMIT}"
  )
  local collection_mode
  collection_mode="$(collection_mode_argument)"
  [[ -z "${collection_mode}" ]] || continuation_arguments+=("${collection_mode}")
  print_command "${continuation_arguments[@]}"
}

prepare_all_receipt() {
  local -a arguments=(prepare-all-workflow "${RUN_ID}" --storage-root "${LOCAL_STORAGE_ROOT}")
  if [[ "${CAMPAIGN_PARTIAL:-false}" == true ]]; then
    arguments+=(--partial --keep-cpu-source)
  elif [[ "${KEEP_CPU_SOURCE}" == true ]]; then
    arguments+=(--keep-cpu-source)
  fi
  generation_run_with_heartbeat \
    "workflow-gates-${RUN_ID}" 8 9 "Retention policy" \
    "validating immutable host and Dataset workflow gates" "" \
    local_cli "${arguments[@]}"
}

read_cleanup_authorization() {
  local line kind authorization_destination_host extra
  line="$(generation_run_with_heartbeat \
    "cleanup-authorization-${RUN_ID}" 8 9 "Retention policy" \
    "validating cleanup authorization inventory" "" \
    local_cli cpu-cleanup-authorization "${RUN_ID}" --format tsv \
      --storage-root "${LOCAL_STORAGE_ROOT}")"
  IFS=$'\t' read -r kind AUTHORIZATION_SHA AUTH_SOURCE_HOST AUTH_SOURCE_ROOT \
    AUTH_DESTINATION_ROOT AUTH_TRANSFER_SHA AUTH_DATASET_SHA AUTH_WORKFLOW_SHA \
    AUTH_SOURCE_INVENTORY_SHA AUTH_SOURCE_FILE_COUNT AUTH_SOURCE_BYTES extra <<< "${line}"
  [[ "${kind}" == authorization && -z "${extra:-}" ]] || fail 1 "Malformed CPU cleanup authorization."
  [[ "${AUTH_SOURCE_HOST}" == "${CPU_HOST}" ]] ||
    fail 1 "Cleanup authorization source host differs from the selected CPU host."
  [[ "${AUTH_SOURCE_ROOT}" == "${REMOTE_STORAGE_ROOT}" ]] ||
    fail 1 "Cleanup authorization source root differs from the selected CPU storage."
  authorization_destination_host="$(container_path_to_host "${AUTH_DESTINATION_ROOT}")"
  [[ "${authorization_destination_host}" == "${LOCAL_STORAGE_ROOT}" ]] ||
    fail 1 "Cleanup authorization destination differs from GPU storage."
  validate_digest "${AUTHORIZATION_SHA}"
  validate_digest "${AUTH_TRANSFER_SHA}"
  validate_digest "${AUTH_DATASET_SHA}"
  validate_digest "${AUTH_WORKFLOW_SHA}"
  validate_digest "${AUTH_SOURCE_INVENTORY_SHA}"
  validate_nonnegative "authorized source file count" "${AUTH_SOURCE_FILE_COUNT}"
  validate_nonnegative "authorized source bytes" "${AUTH_SOURCE_BYTES}"
  CPU_BYTES_RETAINED="${AUTH_SOURCE_BYTES}"
  CPU_BYTES_RETAINED_EXACT=true
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
  local cleanup_progress
  printf -v cleanup_progress "bytes_completed=0/%s" "${AUTH_SOURCE_BYTES}"
  line="$(generation_run_with_heartbeat \
    "cpu-cleanup-${RUN_ID}" 8 9 "Retention policy" \
    "removing the exact authorized CPU source" "${cleanup_progress}" \
    remote_cli "${CLEANUP_ARGUMENTS[@]}" --confirm --format tsv)"
  IFS=$'\t' read -r kind cleanup_status cleanup_mode cleanup_auth reclaimed receipt_sha extra <<< "${line}"
  [[ "${kind}" == cleanup && "${cleanup_status}" == complete \
    && "${cleanup_auth}" == "${AUTHORIZATION_SHA}" && -z "${extra:-}" ]] ||
    fail 1 "Malformed or incomplete CPU cleanup result."
  validate_nonnegative "CPU reclaimed bytes" "${reclaimed}"
  validate_digest "${receipt_sha}"
  [[ "${reclaimed}" == "${AUTH_SOURCE_BYTES}" ]] || fail 1 "CPU reclaimed bytes differ from authorization."
  generation_run_with_heartbeat \
    "cleanup-record-${RUN_ID}" 8 9 "Retention policy" \
    "recording and revalidating cleanup completion" \
    "bytes_completed=${reclaimed}/${AUTH_SOURCE_BYTES}" \
    local_cli record-cpu-cleanup "${RUN_ID}" --storage-root "${LOCAL_STORAGE_ROOT}" \
      --authorization-sha256 "${AUTHORIZATION_SHA}" \
      --cleanup-receipt-sha256 "${receipt_sha}" --reclaimed-bytes "${reclaimed}"
  CPU_BYTES_RETAINED=0
  CPU_BYTES_RETAINED_EXACT=true
  CPU_BYTES_RECLAIMED="${reclaimed}"
  CPU_CLEANUP_RECEIPT_SHA="${receipt_sha}"
}

cleanup_cpu_source() {
  resolve_local_python
  resolve_remote_layout
  resolve_local_storage
  verify_remote_setup_for_output
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
  local -a local_arguments=(
    storage-status --role gpu --metadata-only --omit-run-status
    --storage-root "${LOCAL_STORAGE_ROOT}"
  )
  local -a remote_arguments=(
    storage-status --role cpu --metadata-only --omit-run-status
    --storage-root "${REMOTE_STORAGE_ROOT}"
  )
  if [[ -n "${RUN_ID}" ]]; then
    local_arguments+=(--campaign-run-id "${RUN_ID}")
    remote_arguments+=(--campaign-run-id "${RUN_ID}")
  fi
  if [[ -n "${RUN_ID}" ]]; then
    printf 'Campaign status:\n'
    remote_cli_retryable "campaign status read" campaign-status \
      "${RUN_ID}" --format workflow-monitor --max-active-cases 8 \
      --storage-root "${REMOTE_STORAGE_ROOT}"
  fi
  printf 'GPU storage status:\n'
  local_cli "${local_arguments[@]}"
  printf 'CPU storage status:\n'
  remote_cli_retryable "CPU storage-status read" "${remote_arguments[@]}"
  if [[ -n "${RUN_ID}" ]]; then
    local_cli validate-pilot-check "${RUN_ID}" --if-present --format summary \
      --storage-root "${LOCAL_STORAGE_ROOT}"
  fi
}

workflow_failure_report() {
  local status="$1"
  trap - EXIT
  ALL_WORKFLOW_ACTIVE=false
  local -a continuation_arguments=(
    "${HOST_REPO_ROOT}/scripts/generation_workflow.sh" run
    "${RUN_CONFIG_ARGUMENT:-CONFIG}"
    --cpu-host "${CPU_HOST:-configured}"
    --remote-root "${REMOTE_ROOT:-configured}"
    --git-commit "${REQUESTED_COMMIT:-unknown}"
  )
  local collection_mode
  collection_mode="$(collection_mode_argument)"
  [[ -z "${collection_mode}" ]] || continuation_arguments+=("${collection_mode}")
  local continuation="" argument quoted
  for argument in "${continuation_arguments[@]}"; do
    printf -v quoted '%q' "${argument}"
    continuation+="${quoted} "
  done
  continuation="${continuation% }"
  WORKFLOW_FAILURE_EVIDENCE=""
  if [[ "${RUN_KIND:-}" == campaign && -n "${RUN_ID:-}" ]]; then
    [[ "${CPU_BYTES_RETAINED_EXACT}" == true ]] || refresh_failure_cpu_bytes || true
    local record kind canonical visible extra
    if [[ -n "${LOCAL_STORAGE_ROOT:-}" \
      && "${CPU_BYTES_RETAINED_EXACT}" == true \
      && "${CPU_BYTES_RETAINED}" =~ ^[0-9]+$ ]] &&
      record="$(local_cli record-workflow-failure "${RUN_ID}" \
        --storage-root "${LOCAL_STORAGE_ROOT}" --stage "${ALL_STAGE}" \
        --continuation-command "${continuation}" --cpu-bytes-retained "${CPU_BYTES_RETAINED}" \
        --format tsv 2>/dev/null)"; then
      IFS=$'\t' read -r kind canonical visible extra <<< "${record}"
      if [[ "${kind}" == workflow-failure && -n "${canonical}" \
        && -n "${visible}" && -z "${extra:-}" ]]; then
        WORKFLOW_FAILURE_EVIDENCE="local canonical=${canonical} container=${visible}"
      fi
    fi
  fi
  generation_console_failure "${ALL_STAGE}" "${RUN_ID:-}" \
    "inspect the preceding exact error and preserved evidence" \
    "${WORKFLOW_FAILURE_EVIDENCE}" "${CPU_BYTES_RETAINED}" "${continuation}"
  return "${status}"
}

workflow_exit_handler() {
  local status="$1"
  if [[ "${ALL_WORKFLOW_ACTIVE}" == true && "${status}" -ne 0 ]]; then
    workflow_failure_report "${status}" || true
  fi
  cleanup_pinned_source || true
}

prepare_pilot_check_receipt() {
  resolve_workflow_campaigns
  local -a arguments=(
    prepare-pilot-check "${RUN_ID}"
    --production-campaign "${TRANSIENT_PRIMARY_CAMPAIGN_HOST_PATH}"
    --storage-root "${LOCAL_STORAGE_ROOT}"
  )
  [[ "${KEEP_CPU_SOURCE}" != true ]] || arguments+=(--keep-cpu-source)
  local_cli "${arguments[@]}" >/dev/null
}

cleanup_pilot_staging() {
  local line kind status removed reclaimed receipt_sha extra
  line="$(local_cli cleanup-pilot-staging "${RUN_ID}" --confirm --format tsv \
    --storage-root "${LOCAL_STORAGE_ROOT}")" ||
    fail 1 "Authorised pilot transfer-staging cleanup failed."
  IFS=$'\t' read -r kind status removed reclaimed receipt_sha extra <<< "${line}"
  [[ "${kind}" == pilot-staging-cleanup && "${status}" == complete \
    && "${removed}" == True && -z "${extra:-}" ]] ||
    fail 1 "Malformed or incomplete pilot transfer-staging cleanup result."
  validate_nonnegative "staging reclaimed bytes" "${reclaimed}"
  validate_digest "${receipt_sha}"
  PILOT_STAGING_RECLAIMED="${reclaimed}"
  PILOT_STAGING_CLEANUP_SHA="${receipt_sha}"
}

record_pilot_cleanup_result() {
  local -a arguments=(
    record-pilot-cleanup "${RUN_ID}"
    --storage-root "${LOCAL_STORAGE_ROOT}"
    --cpu-bytes-reclaimed "${CPU_BYTES_RECLAIMED}"
    --transfer-staging-removed
    --staging-bytes-reclaimed "${PILOT_STAGING_RECLAIMED}"
    --staging-cleanup-receipt-sha256 "${PILOT_STAGING_CLEANUP_SHA}"
  )
  if [[ "${KEEP_CPU_SOURCE}" != true ]]; then
    validate_digest "${CPU_CLEANUP_RECEIPT_SHA}"
    arguments+=(
      --cpu-source-removed
      --cpu-cleanup-receipt-sha256 "${CPU_CLEANUP_RECEIPT_SHA}"
    )
  fi
  local_cli "${arguments[@]}" >/dev/null
}

resolve_benchmark_contract() {
  admit_repository_file "${RUN_LEAF_CONFIG}" "core benchmark suite"
  BENCHMARK_SUITE_PATH="${ADMITTED_HOST_PATH}"
  BENCHMARK_SUITE_RELATIVE_PATH="${ADMITTED_REPOSITORY_PATH}"
  local -a inspect_arguments=(inspect-core-benchmark "${BENCHMARK_SUITE_PATH}")
  local inspection record kind extra configured_cpu_host configured_scheduler
  local configured_partition configured_cores_per_node configured_python_module
  local configured_comsol_module configured_python_executable configured_comsol_executable
  inspection="$(local_cli "${inspect_arguments[@]}")" ||
    fail 2 "Could not resolve the maintained core benchmark suite."
  record="$(printf '%s\n' "${inspection}" | local_python -c 'import json, sys
value = json.load(sys.stdin)
resource = value["resource_contract"]
waves = value["variant_waves"]
fields = (
    value["suite_name"], value["suite_digest"],
    str(value["required_successful_measurements"]),
    str(value["parallel_cases_per_variant"]),
    ",".join(str(item["cores_per_case"]) for item in waves),
    resource["cpu_host"], resource["scheduler"], resource["partition"],
    str(resource["cores_per_node"]), resource["python_module"],
    resource["comsol_module"], resource["python_executable"],
    resource["comsol_executable"], str(resource["poll_interval_seconds"]),
)
if any("\t" in str(item) or "\n" in str(item) or "\r" in str(item) for item in fields):
    raise SystemExit("benchmark inspection contains unsafe shell transport text")
print("\t".join(("benchmark", *(str(item) for item in fields))))')" ||
    fail 2 "Could not parse the maintained core benchmark suite."
  IFS=$'\t' read -r kind BENCHMARK_SUITE_NAME BENCHMARK_SUITE_DIGEST \
    BENCHMARK_MEASUREMENTS BENCHMARK_CASES_PER_VARIANT BENCHMARK_CORE_COUNTS \
    configured_cpu_host configured_scheduler configured_partition configured_cores_per_node \
    configured_python_module configured_comsol_module configured_python_executable \
    configured_comsol_executable STATUS_POLL_SECONDS extra <<< "${record}"
  [[ "${kind}" == benchmark && -z "${extra:-}" ]] ||
    fail 1 "Malformed benchmark inspection record."
  validate_positive "configured benchmark poll_interval_seconds" "${STATUS_POLL_SECONDS}"
  [[ -n "${CPU_HOST}" ]] || CPU_HOST="${configured_cpu_host}"
  SCHEDULER_KIND="${configured_scheduler}"
  PARTITION="${configured_partition}"
  CORES_PER_NODE="${configured_cores_per_node}"
  PYTHON_MODULE="${configured_python_module}"
  COMSOL_MODULE="${configured_comsol_module}"
  PYTHON_EXECUTABLE="${configured_python_executable}"
  COMSOL_EXECUTABLE="${configured_comsol_executable}"
  printf '%s\n' "${inspection}"
}

remote_benchmark_plan_submit() {
  local operation="$1"
  local remote_suite
  remote_suite="$(remote_repository_path "${BENCHMARK_SUITE_RELATIVE_PATH}")"
  remote_bash "${CPU_HOST}" \
    "${REMOTE_REPOSITORY}" "${REMOTE_STORAGE_ROOT}" "${REMOTE_VENV}" \
    "${REQUESTED_COMMIT}" "${remote_suite}" "${operation}" \
    "${PYTHON_MODULE}" "${COMSOL_MODULE}" "${COMSOL_EXECUTABLE}" <<'REMOTE'
set -euo pipefail
repository="$1"; storage="$2"; venv="$3"; commit="$4"; suite="$5"
operation="$6"; python_module="$7"; comsol_module="$8"; comsol_executable="$9"
for name in realpath stat mktemp rm; do
  command -v "${name}" >/dev/null 2>&1 || {
    printf 'CPU benchmark preflight failed: required command %s is unavailable.\n' "${name}" >&2
    exit 1
  }
done
benchmark_scratch_parent="$(realpath -e -- "${TMPDIR:-/tmp}")" || {
  printf 'CPU benchmark preflight failed: temporary parent cannot be resolved.\n' >&2
  exit 1
}
[[ -d "${benchmark_scratch_parent}" && ! -L "${benchmark_scratch_parent}" \
  && -w "${benchmark_scratch_parent}" && -x "${benchmark_scratch_parent}" ]] || {
  printf 'CPU benchmark preflight failed: temporary parent is unsafe.\n' >&2
  exit 1
}
scratch="$(mktemp -d "${benchmark_scratch_parent%/}/generation-benchmark-preflight.XXXXXXXX")" || {
  printf 'CPU benchmark preflight failed: temporary scratch creation failed.\n' >&2
  exit 1
}
[[ -d "${scratch}" && ! -L "${scratch}" \
  && "$(stat -c %u "${scratch}")" -eq "${UID}" ]] || {
  printf 'CPU benchmark preflight failed: temporary scratch ownership is unsafe.\n' >&2
  exit 1
}
scratch_marker="${scratch}/.generation-benchmark-preflight"
printf 'generation-benchmark-preflight\n' > "${scratch_marker}"
cleanup_benchmark_scratch() {
  local status="$1"
  if [[ -d "${scratch}" && ! -L "${scratch}" && -f "${scratch_marker}" \
    && ! -L "${scratch_marker}" \
    && "${scratch}" == "${benchmark_scratch_parent}/generation-benchmark-preflight."* ]]; then
    rm -rf -- "${scratch}"
  else
    printf 'CPU benchmark preflight refused to remove an unverified scratch directory.\n' >&2
    status=1
  fi
  trap - EXIT
  exit "${status}"
}
trap 'cleanup_benchmark_scratch "$?"' EXIT
module load "${python_module}"
if ! module load "${comsol_module}"; then
  printf 'CPU benchmark preflight failed: COMSOL module %s is unavailable.\n' \
    "${comsol_module}" >&2
  exit 1
fi
resolved_comsol="$(command -v "${comsol_executable}")" || {
  printf 'CPU benchmark preflight failed: COMSOL executable %s is unavailable.\n' \
    "${comsol_executable}" >&2
  exit 1
}
resolved_comsol="$(readlink -f -- "${resolved_comsol}")"
[[ "${resolved_comsol}" == /* && -x "${resolved_comsol}" ]] || {
  printf 'CPU benchmark preflight resolved an unsafe executable: %s\n' \
    "${resolved_comsol}" >&2
  exit 1
}
comsol_version="$("${resolved_comsol}" -version 2>&1)" || {
  printf 'CPU benchmark preflight failed: COMSOL version query failed.\n' >&2
  exit 1
}
export GENERATION_CPU_VENV="${venv}"
export STORAGE_ROOT="${storage}"
export GENERATION_GIT_COMMIT="${commit}"
cd "${repository}"
command=("${venv}/bin/python" -m src.generation.cli.cli_generation
  "${operation}" "${suite}"
  --git-commit "${commit}"
  --storage-root "${storage}"
  --scratch-root "${scratch}"
  --comsol-version-output "${comsol_version}"
  --comsol-executable-path "${resolved_comsol}")
"${command[@]}"
REMOTE
}

collect_core_benchmark() {
  require_command rsync "core benchmark transfer"
  if local_cli validate-core-benchmark "${RUN_ID}" \
    --storage-root "${LOCAL_STORAGE_ROOT}" >/dev/null 2>&1; then
    printf 'GPU benchmark publication validated and reused for %s.\n' "${RUN_ID}"
    local_cli core-benchmark-summary "${RUN_ID}" --format markdown \
      --storage-root "${LOCAL_STORAGE_ROOT}"
    return
  fi
  local plan kind plan_run plan_commit relative inventory_sha file_count size_bytes extra staging receipt
  plan="$(remote_cli_retryable "benchmark transfer-plan read" \
    core-benchmark-transfer-plan "${RUN_ID}" --format tsv \
    --storage-root "${REMOTE_STORAGE_ROOT}")" ||
    fail_preserving_interrupt "$?" 1 "Remote core benchmark is not terminally valid."
  IFS=$'\t' read -r kind plan_run plan_commit relative inventory_sha file_count size_bytes extra <<< "${plan}"
  [[ "${kind}" == benchmark && "${plan_run}" == "${RUN_ID}" \
    && "${plan_commit}" == "${REQUESTED_COMMIT}" \
    && "${inventory_sha}" =~ ^[0-9a-f]{64}$ \
    && "${file_count}" =~ ^[0-9]+$ && "${size_bytes}" =~ ^[0-9]+$ \
    && -z "${extra:-}" ]] ||
    fail 1 "Malformed core benchmark transfer plan."
  validate_transfer_path "${relative}"
  staging="$(local_cli create-transfer-staging "${RUN_ID}" \
    --storage-root "${LOCAL_STORAGE_ROOT}")" ||
    fail 1 "Could not create marked benchmark transfer staging."
  staging="$(container_path_to_host "${staging}")"
  rsync -a --protect-args --relative --exclude='.state/' --exclude='work/' \
    "${CPU_HOST}:${REMOTE_STORAGE_ROOT}/./${relative}" "${staging}/" ||
    fail 1 "Benchmark transfer failed; staging retained at ${staging}."
  receipt="$(local_cli publish-transferred-core-benchmark "${RUN_ID}" \
    --staging-root "${staging}" --destination-root "${LOCAL_STORAGE_ROOT}" \
    --source-host "${CPU_HOST}" --source-storage-root "${REMOTE_STORAGE_ROOT}" \
    --expected-inventory-sha256 "${inventory_sha}" \
    --expected-file-count "${file_count}" --expected-size-bytes "${size_bytes}")" ||
    fail 1 "GPU benchmark publication failed; staging retained at ${staging}."
  local_cli cleanup-transfer-staging --campaign-run-id "${RUN_ID}" \
    --directory "${staging}" --storage-root "${LOCAL_STORAGE_ROOT}" \
    --confirm >/dev/null
  printf '%s\n' "${receipt}"
  local_cli core-benchmark-summary "${RUN_ID}" --format markdown \
    --storage-root "${LOCAL_STORAGE_ROOT}"
  printf 'CPU benchmark evidence retained at %s:%s.\n' \
    "${CPU_HOST}" "${REMOTE_STORAGE_ROOT}"
}



resolve_generation_run_plan() {
  resolve_local_storage
  resolve_local_python
  admit_repository_file "$RUN_CONFIG_ARGUMENT" "Generation run config"
  RUN_CONFIG_PATH="$ADMITTED_HOST_PATH"
  RUN_CONFIG_RELATIVE="$ADMITTED_REPOSITORY_PATH"
  local -a arguments=(
    resolve-generation-run "$RUN_CONFIG_PATH"
    --git-commit "$REQUESTED_COMMIT"
  )
  [[ "$ALLOW_INCOMPLETE_PLAN" != true ]] || arguments+=(--allow-incomplete)
  RUN_PLAN_JSON="$(local_cli "${arguments[@]}")" ||
    fail 2 "Could not resolve the Generation run plan."
  local records line kind field2 field3 field4 field5 field6 field7 extra
  records="$(printf '%s' "$RUN_PLAN_JSON" | local_python -c 'import json, sys
plan = json.load(sys.stdin)
def clean(value):
    text = str(value)
    if any(character in text for character in "\t\r\n"):
        raise SystemExit("run plan contains unsafe shell transport text")
    return text
purpose = "-"
profile = "-"
if plan["units"]:
    purpose = plan["units"][0]["metadata"].get("campaign_purpose", "-")
    profile = plan["units"][0]["metadata"].get("simulation_profile", "-")
print("\t".join((
    "plan", clean(plan["run_kind"]), clean(plan["identity"]),
    clean(plan["config_path"]), clean(purpose), clean(profile),
    clean(len(plan["children"])),
)))
for child in plan["children"]:
    metadata = child["units"][0]["metadata"]
    print("\t".join((
        "child", clean(child["config_path"]), clean(child["identity"]),
        clean(metadata["campaign_purpose"]),
        clean(metadata["simulation_profile"]),
        clean(child["input_identity"]),
    )))')" || fail 1 "Could not decode the common Generation run plan."
  RUN_CHILD_CONFIGS=()
  RUN_CHILD_IDENTITIES=()
  RUN_CHILD_PURPOSES=()
  RUN_CHILD_PROFILES=()
  RUN_CHILD_INPUT_IDENTITIES=()
  while IFS=$'\t' read -r kind field2 field3 field4 field5 field6 field7 extra; do
    [[ -z "${extra:-}" ]] || fail 1 "Malformed common Generation run plan record."
    case "$kind" in
      plan)
        RUN_KIND="$field2"
        RUN_PLAN_ID="$field3"
        RUN_PLAN_CONFIG="$field4"
        RUN_PLAN_PURPOSE="$field5"
        RUN_PLAN_PROFILE="$field6"
        RUN_CHILD_COUNT="$field7"
        ;;
      child)
        RUN_CHILD_CONFIGS+=("$field2")
        RUN_CHILD_IDENTITIES+=("$field3")
        RUN_CHILD_PURPOSES+=("$field4")
        RUN_CHILD_PROFILES+=("$field5")
        RUN_CHILD_INPUT_IDENTITIES+=("$field6")
        ;;
      *) fail 1 "Unknown common Generation run plan record." ;;
    esac
  done <<< "$records"
  validate_nonnegative "Generation child count" "$RUN_CHILD_COUNT"
  (( ${#RUN_CHILD_CONFIGS[@]} == RUN_CHILD_COUNT )) ||
    fail 1 "Common Generation run plan child count is inconsistent."
}

campaign_local_completion_is_valid() {
  local_cli validate-all-workflow "$RUN_ID"     --storage-root "$LOCAL_STORAGE_ROOT" >/dev/null 2>&1 || return 1
  local_cli validate-campaign-package-state "$RUN_ID"     --storage-root "$LOCAL_STORAGE_ROOT" >/dev/null 2>&1 || return 1
  if [[ "$CAMPAIGN_PURPOSE" == pilot_check ]]; then
    local_cli validate-pilot-check "$RUN_ID" --require-cleanup-complete       --storage-root "$LOCAL_STORAGE_ROOT" >/dev/null 2>&1 || return 1
  fi
}

select_compatible_campaign_source() {
  [[ "$CAMPAIGN_PURPOSE" != pilot_check     && "$CAMPAIGN_PURPOSE" != technical_runtime_smoke ]] || return 1
  local output record status compatible_run package_state artifact_identity extra
  output="$(local_cli find-compatible-campaign-source     "$CAMPAIGN_CONFIG_PATH" --storage-root "$LOCAL_STORAGE_ROOT")" ||
    fail 1 "Could not inspect compatible completed campaign sources."
  record="$(printf '%s' "$output" | local_python -c 'import json, sys
value = json.load(sys.stdin)
print("\t".join((
    str(value["status"]),
    "-" if value["campaign_run_id"] is None else str(value["campaign_run_id"]),
    str(value.get("package_state", "-")),
    str(value.get("artifact_set_sha256", "-")),
)))')" || fail 1 "Could not decode compatible campaign-source discovery."
  IFS=$'\t' read -r status compatible_run package_state artifact_identity extra <<< "$record"
  [[ -z "${extra:-}" ]] || fail 1 "Malformed compatible campaign-source result."
  [[ "$status" == compatible_complete ]] || return 1
  validate_run_id "$compatible_run"
  validate_digest "$artifact_identity"
  RUN_ID="$compatible_run"
  case "$package_state" in
    complete)
      campaign_local_completion_is_valid ||
        fail 1 "Compatible campaign source reported an invalid current package state."
      LEAF_STATE=complete
      LEAF_RESULT=REUSED
      LEAF_EXISTING_DETAIL="campaign_run_id=$RUN_ID artifact_set=$artifact_identity compatible host workflow"
      ;;
    extension_required)
      LEAF_STATE=package_only
      LEAF_EXISTING_DETAIL="campaign_run_id=$RUN_ID artifact_set=$artifact_identity package-only continuation"
      ;;
    *) fail 1 "Compatible campaign source returned unsupported package state: $package_state" ;;
  esac
  return 0
}

select_compatible_smoke_child() {
  [[ "$CAMPAIGN_PURPOSE" == technical_runtime_smoke ]] || return 1
  local output record status compatible_run extra
  output="$(local_cli find-compatible-technical-smoke-run     "$CAMPAIGN_CONFIG_PATH" --storage-root "$LOCAL_STORAGE_ROOT")" ||
    fail 1 "Could not inspect dependency-compatible completed Technical Smoke runs."
  record="$(printf '%s' "$output" | local_python -c 'import json, sys
value = json.load(sys.stdin)
print("\t".join((
    str(value["status"]),
    "-" if value["campaign_run_id"] is None else str(value["campaign_run_id"]),
)))')" || fail 1 "Could not decode compatible Technical Smoke discovery."
  IFS=$'\t' read -r status compatible_run extra <<< "$record"
  [[ -z "${extra:-}" ]] || fail 1 "Malformed compatible Technical Smoke result."
  case "$status" in
    compatible_complete)
      validate_run_id "$compatible_run"
      RUN_ID="$compatible_run"
      campaign_local_completion_is_valid ||
        fail 1 "Selected Technical Smoke compatibility candidate is not terminally valid."
      LEAF_RESULT=REUSED
      LEAF_STATE=complete
      LEAF_EXISTING_DETAIL="campaign_run_id=$RUN_ID dependency-compatible completed Smoke child"
      ;;
    compatible_repairable)
      validate_run_id "$compatible_run"
      RUN_ID="$compatible_run"
      resolve_remote_layout
      verify_remote_setup >/dev/null
      LEAF_STATE=transfer_repair
      LEAF_EXISTING_DETAIL="campaign_run_id=$RUN_ID compatible scientific source requires transfer-evidence recovery"
      ;;
    missing) return 1 ;;
    *) fail 1 "Compatible Technical Smoke discovery returned unsupported status: $status" ;;
  esac
  return 0
}

monitor_generation_units() {
  local monitor_kind="$1"
  validate_positive "configured poll_interval_seconds" "$STATUS_POLL_SECONDS"
  arm_campaign_interrupt
  while true; do
    case "$monitor_kind" in
      campaign)
        read_remote_workflow_monitor
        local campaign_detail
        campaign_detail="$REMOTE_CAMPAIGN_SUMMARY"$'\n'"Source storage: state=$REMOTE_SOURCE_STATE retained_bytes=$CPU_BYTES_RETAINED"
        generation_console_progress units 5 9 "Work units" RUNNING           "$REMOTE_CAMPAIGN_STATE_SIGNATURE|$REMOTE_SOURCE_STATE|$REMOTE_SOURCE_ACTIVE"           "$campaign_detail"           "$REMOTE_CAMPAIGN_PROGRESS_SIGNATURE|$CPU_BYTES_RETAINED"
        case "$REMOTE_CAMPAIGN_STATE" in
          successful|transfer_complete)
            remote_cli_retryable "campaign terminal validation" \
              validate-campaign-terminal "$RUN_ID" \
              --storage-root "$REMOTE_STORAGE_ROOT" >/dev/null
            disarm_campaign_interrupt
            return
            ;;
          running|feeding|license_blocked|submission_pending_or_unknown)
            sleep "$STATUS_POLL_SECONDS" || true
            ;;
          completed_with_failures)
            CAMPAIGN_PARTIAL=true
            disarm_campaign_interrupt
            return
            ;;
          cancelled)
            disarm_campaign_interrupt
            fail 1 "Campaign is cancelled; rerun the same config to resume eligible work."
            ;;
          *)
            disarm_campaign_interrupt
            fail 1 "Campaign entered unsupported state: $REMOTE_CAMPAIGN_STATE"
            ;;
        esac
        ;;
      benchmark)
        remote_cli resume-core-benchmark "$RUN_ID" \
          --storage-root "$REMOTE_STORAGE_ROOT" >/dev/null
        local output header state state_signature progress_signature detail extra
        output="$(remote_cli_retryable "benchmark status read" \
          core-benchmark-status "$RUN_ID" \
          --storage-root "$REMOTE_STORAGE_ROOT" --format monitor)" ||
          fail_preserving_interrupt "$?" 1 \
            "Could not reconstruct benchmark work-unit status."
        header="${output%%$'\n'*}"
        detail="${output#*$'\n'}"
        IFS=$'\t' read -r _monitor_record state state_signature progress_signature extra <<< "$header"
        [[ "$_monitor_record" == "campaign-monitor" && -z "${extra:-}" ]] ||
          fail 1 "Malformed benchmark monitor record."
        generation_console_progress units 5 9 "Work units" RUNNING \
          "$state_signature" "$detail" "$progress_signature"
        case "$state" in
          complete)
            disarm_campaign_interrupt
            return
            ;;
          inputs_ready|running|license_blocked)
            sleep "$STATUS_POLL_SECONDS" || true
            ;;
          canary_failed|work_unit_failed)
            disarm_campaign_interrupt
            fail 1 "Benchmark reached a terminal canary or work-unit failure; inspect compact retained evidence."
            ;;
          cancelled)
            disarm_campaign_interrupt
            fail 1 "Benchmark is cancelled; rerun the same suite to resume eligible work."
            ;;
          *)
            disarm_campaign_interrupt
            fail 1 "Benchmark entered unsupported state: $state"
            ;;
        esac
        ;;
      *)
        disarm_campaign_interrupt
        fail 2 "Unknown common monitor run kind: $monitor_kind"
        ;;
    esac
  done
}

resolve_leaf_plan() {
  LEAF_STATE=running
  LEAF_RESULT=OK
  CAMPAIGN_PARTIAL=false
  LEAF_EXISTING_DETAIL=""
  case "$RUN_KIND" in
    campaign)
      RUN_ID="$EXPECTED_RUN_ID"
      PILOT_MODE=false
      resolve_campaign "$RUN_LEAF_CONFIG"
      resolve_configured_resources executable
      validate_resources
      if [[ "$CAMPAIGN_PURPOSE" == pilot_check ]]; then
        resolve_pilot_contract
      fi
      if select_compatible_smoke_child; then
        return
      fi
      RUN_ID="$EXPECTED_RUN_ID"
      if campaign_local_completion_is_valid; then
        LEAF_RESULT=REUSED
        LEAF_STATE=complete
        LEAF_EXISTING_DETAIL="campaign_run_id=$RUN_ID complete host workflow"
        return
      fi
      if select_compatible_campaign_source; then
        return
      fi
      resolve_remote_layout
      verify_remote_setup >/dev/null
      LEAF_EXISTING_DETAIL="campaign_run_id=$RUN_ID continuation inspected"
      ;;
    benchmark)
      resolve_benchmark_contract >/dev/null
      [[ "$SCHEDULER_KIND" == slurm ]] ||
        fail 2 "Core benchmarking requires configured scheduler=slurm."
      resolve_remote_layout
      verify_remote_setup >/dev/null
      local comsol_version identity_json
      comsol_version="$(remote_comsol_version)"
      identity_json="$(local_cli resolve-core-benchmark-run "$BENCHMARK_SUITE_PATH" \
        --git-commit "$REQUESTED_COMMIT" \
        --comsol-version-output "$comsol_version")" ||
        fail 1 "Could not resolve deterministic benchmark runtime identity."
      RUN_ID="$(printf '%s' "$identity_json" | local_python -c 'import json, sys
print(json.load(sys.stdin)["benchmark_run_id"])')" ||
        fail 1 "Could not decode benchmark runtime identity."
      validate_benchmark_run_id "$RUN_ID"
      LEAF_EXISTING_DETAIL="benchmark_run_id=$RUN_ID continuation inspected"
      if local_cli validate-core-benchmark "$RUN_ID" \
        --storage-root "$LOCAL_STORAGE_ROOT" >/dev/null 2>&1; then
        LEAF_RESULT=REUSED
        if [[ "$KEEP_CPU_SOURCE" != true ]]; then
          cleanup_core_benchmark_cpu
        fi
        LEAF_STATE=complete
        LEAF_EXISTING_DETAIL="benchmark_run_id=$RUN_ID complete host workflow"
      fi
      ;;
    *) fail 2 "Unsupported common leaf run kind: $RUN_KIND" ;;
  esac
}

materialize_leaf_inputs() {
  case "$RUN_KIND" in
    campaign)
      local input_record input_kind generated reused extra
      input_record="$(remote_prepare_campaign_inputs)" ||
        fail 1 "Canonical campaign input preparation failed before submission."
      IFS=$'\t' read -r input_kind generated reused extra <<< "$input_record"
      [[ "$input_kind" == canonical-inputs && -z "${extra:-}" ]] ||
        fail 1 "Malformed canonical campaign input readiness result."
      validate_nonnegative "generated canonical input count" "$generated"
      validate_nonnegative "reused canonical input count" "$reused"
      LEAF_INPUT_DETAIL="reused=$reused generated=$generated"
      ;;
    benchmark)
      local output observed_run
      output="$(remote_benchmark_plan_submit materialize-core-benchmark-inputs)" ||
        fail 1 "Benchmark canonical input preparation failed before submission."
      observed_run="$(printf '%s' "$output" | local_python -c 'import json, sys
print(json.load(sys.stdin)["benchmark_run_id"])')" ||
        fail 1 "Benchmark input preparation returned no run identity."
      [[ "$observed_run" == "$RUN_ID" ]] ||
        fail 1 "Benchmark input preparation identity disagrees with the common run plan."
      LEAF_INPUT_DETAIL="benchmark_run_id=$RUN_ID canonical input ready on CPU login node"
      ;;
    *) fail 2 "Unsupported canonical-input adapter: $RUN_KIND" ;;
  esac
}

submit_leaf_units() {
  case "$RUN_KIND" in
    campaign)
      launch_campaign >/dev/null
      LEAF_PLAN_DETAIL="campaign_run_id=$RUN_ID purpose=$CAMPAIGN_PURPOSE"
      ;;
    benchmark)
      local output observed_run
      output="$(remote_benchmark_plan_submit submit-core-benchmark)" ||
        fail 1 "Benchmark first work-unit submission failed after input readiness."
      observed_run="$(printf '%s' "$output" | local_python -c 'import json, sys
print(json.load(sys.stdin)["benchmark_run_id"])')" ||
        fail 1 "Benchmark submission returned no run identity."
      [[ "$observed_run" == "$RUN_ID" ]] ||
        fail 1 "Benchmark submission identity disagrees with the common run plan."
      LEAF_PLAN_DETAIL="suite=$BENCHMARK_SUITE_NAME measurements=$BENCHMARK_MEASUREMENTS cases_per_wave=$BENCHMARK_CASES_PER_VARIANT cores=$BENCHMARK_CORE_COUNTS"
      ;;
    *) fail 2 "Unsupported work-unit submission adapter: $RUN_KIND" ;;
  esac
}

cleanup_core_benchmark_cpu() {
  local line kind auth_run source_host source_root destination_root destination_host inventory_sha
  local file_count size_bytes authorization_sha extra
  line="$(local_cli core-benchmark-cleanup-authorization "$RUN_ID"     --format tsv --storage-root "$LOCAL_STORAGE_ROOT")" ||
    fail 1 "Could not authorize benchmark CPU cleanup."
  IFS=$'\t' read -r kind auth_run source_host source_root destination_root     inventory_sha file_count size_bytes authorization_sha extra <<< "$line"
  destination_host="$(container_path_to_host "$destination_root")"
  [[ "$kind" == benchmark-cleanup-authorization && "$auth_run" == "$RUN_ID"     && "$source_host" == "$CPU_HOST" && "$source_root" == "$REMOTE_STORAGE_ROOT"     && "$destination_host" == "$LOCAL_STORAGE_ROOT" && -z "${extra:-}" ]] ||
    fail 1 "Malformed benchmark cleanup authorization."
  validate_digest "$inventory_sha"
  validate_nonnegative "benchmark source file count" "$file_count"
  validate_nonnegative "benchmark source bytes" "$size_bytes"
  validate_digest "$authorization_sha"
  local output record cleanup_status receipt_sha reclaimed
  output="$(remote_cli cleanup-core-benchmark-source "$RUN_ID"     --storage-root "$REMOTE_STORAGE_ROOT" --source-host "$source_host"     --destination-storage-root "$destination_root"     --expected-inventory-sha256 "$inventory_sha"     --expected-file-count "$file_count" --expected-size-bytes "$size_bytes"     --authorization-sha256 "$authorization_sha" --confirm)" ||
    fail 1 "Authorized benchmark CPU cleanup failed."
  record="$(printf '%s' "$output" | local_python -c 'import json, sys
value = json.load(sys.stdin)
print("\t".join((
    str(value["status"]), str(value["receipt_sha256"]),
    str(value["reclaimed_bytes"]),
)))')" || fail 1 "Could not decode benchmark cleanup result."
  IFS=$'\t' read -r cleanup_status receipt_sha reclaimed extra <<< "$record"
  [[ "$cleanup_status" == complete && -z "${extra:-}" ]] ||
    fail 1 "Benchmark cleanup did not return complete evidence."
  validate_digest "$receipt_sha"
  validate_nonnegative "benchmark reclaimed bytes" "$reclaimed"
  [[ "$reclaimed" == "$size_bytes" ]] ||
    fail 1 "Benchmark cleanup reclaimed-byte count differs from authorization."
  local_cli record-core-benchmark-cleanup "$RUN_ID"     --storage-root "$LOCAL_STORAGE_ROOT"     --authorization-sha256 "$authorization_sha"     --cleanup-receipt-sha256 "$receipt_sha"     --reclaimed-bytes "$reclaimed" >/dev/null
  CPU_BYTES_RECLAIMED="$reclaimed"
}

benchmark_deferred_report() {
  local output record source_state bytes extra
  output="$(remote_cli_retryable "benchmark source-status read" \
    core-benchmark-source-status "$RUN_ID" --format tsv \
    --storage-root "$REMOTE_STORAGE_ROOT")" ||
    fail_preserving_interrupt "$?" 1 \
      "Could not reconstruct deferred benchmark source state."
  IFS=$'\t' read -r _kind _run _run_state source_state bytes _eligibility _active extra <<< "$output"
  [[ -z "${extra:-}" ]] || fail 1 "Malformed deferred benchmark source state."
  printf 'benchmark_run_id=%s\nstate=awaiting_collection\nsource_state=%s\nretained_cpu_bytes=%s\n'     "$RUN_ID" "$source_state" "$bytes"
  printf 'Resume collection with the same config:\n'
  local -a continuation_arguments=(
    "$HOST_REPO_ROOT/scripts/generation_workflow.sh" run
    "$RUN_CONFIG_ARGUMENT" --cpu-host "$CPU_HOST"
    --remote-root "$REMOTE_ROOT" --git-commit "$REQUESTED_COMMIT"
  )
  local collection_mode
  collection_mode="$(collection_mode_argument)"
  [[ -z "${collection_mode}" ]] || continuation_arguments+=("${collection_mode}")
  print_command "${continuation_arguments[@]}"
}

finalize_leaf_cpu_evidence() {
  case "$RUN_KIND" in
    campaign) ;;
    benchmark)
      remote_cli finalize-core-benchmark "$RUN_ID" \
        --storage-root "$REMOTE_STORAGE_ROOT" >/dev/null
      ;;
    *) fail 2 "Unsupported CPU-finalization adapter: $RUN_KIND" ;;
  esac
}

deferred_leaf_report() {
  case "$RUN_KIND" in
    campaign) deferred_campaign_report ;;
    benchmark) benchmark_deferred_report ;;
    *) fail 2 "Unsupported deferred-collection adapter: $RUN_KIND" ;;
  esac
}

collect_leaf_results() {
  case "$RUN_KIND" in
    campaign) collect_campaign >/dev/null ;;
    benchmark) collect_core_benchmark >/dev/null ;;
    *) fail 2 "Unsupported collection adapter: $RUN_KIND" ;;
  esac
}

build_leaf_packages_and_finalizers() {
  case "$RUN_KIND" in
    campaign)
      if [[ "$CAMPAIGN_PURPOSE" == pilot_check ]]; then
        prepare_pilot_check_receipt
      fi
      local dataset_output dataset_record dataset_status dataset_reason declared_package_count extra
      dataset_output="$(build_datasets)"
      dataset_record="$(printf '%s' "$dataset_output" | local_python -c 'import json, sys
value = json.load(sys.stdin)
reason = str(value.get("reason", "-")).replace("\t", " ").replace("\r", " ").replace("\n", " ")
count = value.get("declared_package_count")
if isinstance(count, bool) or not isinstance(count, int) or count < 0:
    raise SystemExit("Dataset package stage has no valid declared_package_count")
print("\t".join((str(value["status"]), reason, str(count))))')" ||
        fail 1 "Could not decode Dataset package stage."
      IFS=$'\t' read -r dataset_status dataset_reason declared_package_count extra <<< "$dataset_record"
      [[ -z "${extra:-}" && "${declared_package_count}" =~ ^[0-9]+$ ]] ||
        fail 1 "Malformed Dataset package stage."
      if [[ "${CAMPAIGN_PARTIAL:-false}" == true ]]; then
        [[ "$dataset_status" == incomplete ]] ||
          fail 1 "Partial Dataset package stage returned unsupported status: $dataset_status"
        LEAF_PACKAGE_DETAIL="campaign_run_id=$RUN_ID Dataset packages incomplete; successful cases retained for resume"
      else
        [[ "$dataset_status" == complete ]] ||
          fail 1 "Dataset package stage returned unsupported status: $dataset_status"
        if [[ "${declared_package_count}" == 0 ]]; then
          LEAF_PACKAGE_DETAIL="campaign_run_id=$RUN_ID no Dataset packages declared; package finalizer gates validated"
        else
          LEAF_PACKAGE_DETAIL="campaign_run_id=$RUN_ID declared packages and finalizers validated"
        fi
      fi
      ;;
    benchmark)
      LEAF_PACKAGE_DETAIL="benchmark_run_id=$RUN_ID summary validated; Dataset packages=none"
      ;;
    *) fail 2 "Unsupported package/finalizer adapter: $RUN_KIND" ;;
  esac
}

apply_leaf_retention() {
  CPU_BYTES_RECLAIMED=0
  case "$RUN_KIND" in
    campaign)
      prepare_all_receipt >/dev/null
      if [[ "${CAMPAIGN_PARTIAL:-false}" == true || "$KEEP_CPU_SOURCE" == true ]]; then
        read_remote_source_status
      else
        confirm_cpu_cleanup >/dev/null
      fi
      if [[ "$CAMPAIGN_PURPOSE" == pilot_check \
        && "${CAMPAIGN_PARTIAL:-false}" != true ]]; then
        cleanup_pilot_staging >/dev/null
        record_pilot_cleanup_result
      fi
      ;;
    benchmark)
      if [[ "$KEEP_CPU_SOURCE" != true ]]; then
        cleanup_core_benchmark_cpu
      fi
      ;;
    *) fail 2 "Unsupported retention adapter: $RUN_KIND" ;;
  esac
}

prepare_leaf_for_parent() {
  [[ "$RUN_KIND" == campaign ]] ||
    fail 2 "Only campaign children support parent-owned retention."
  ALL_STAGE="validated child gates awaiting parent finalization"
  generation_console_stage 8 9 "Retention policy" RUNNING
  prepare_all_receipt >/dev/null
  generation_console_stage 8 9 "Retention policy" DEFERRED \
    "parent-owned cleanup waits for paired finalization and parent validation"
  generation_console_stage 9 9 "Child validation" RUNNING
  generation_run_with_heartbeat \
    "parent-gate-packages-$RUN_ID" 9 9 "Child validation" \
    "validating required Dataset package state" "" \
    local_cli validate-campaign-package-state "$RUN_ID" \
      --storage-root "$LOCAL_STORAGE_ROOT" >/dev/null
  LEAF_STATE=ready_for_parent
  generation_console_stage 9 9 "Child validation" OK \
    "run_id=$RUN_ID host and required package gates complete"
}

validate_leaf_result() {
  case "$RUN_KIND" in
    campaign)
      if [[ "${CAMPAIGN_PARTIAL:-false}" == true ]]; then
        generation_run_with_heartbeat \
          "partial-workflow-$RUN_ID" 9 9 "Final validation" \
          "validating partial publication, retained CPU source, and resume metadata" "" \
          local_cli validate-all-workflow "$RUN_ID" --partial \
            --storage-root "$LOCAL_STORAGE_ROOT" >/dev/null
      else
        generation_run_with_heartbeat \
          "final-workflow-$RUN_ID" 9 9 "Final validation" \
          "validating terminal workflow and cleanup evidence" "" \
          local_cli validate-all-workflow "$RUN_ID" \
            --storage-root "$LOCAL_STORAGE_ROOT" >/dev/null
        generation_run_with_heartbeat \
          "final-packages-$RUN_ID" 9 9 "Final validation" \
          "validating current Dataset package state" "" \
          local_cli validate-campaign-package-state "$RUN_ID" \
            --storage-root "$LOCAL_STORAGE_ROOT" >/dev/null
      fi
      if [[ "$CAMPAIGN_PURPOSE" == pilot_check \
        && "${CAMPAIGN_PARTIAL:-false}" != true ]]; then
        generation_run_with_heartbeat \
          "final-pilot-$RUN_ID" 9 9 "Final validation" \
          "validating pilot cleanup evidence" "" \
          local_cli validate-pilot-check "$RUN_ID" --require-cleanup-complete \
            --storage-root "$LOCAL_STORAGE_ROOT" >/dev/null
      fi
      ;;
    benchmark)
      generation_run_with_heartbeat \
        "final-benchmark-$RUN_ID" 9 9 "Final validation" \
        "validating core benchmark publication" "" \
        local_cli validate-core-benchmark "$RUN_ID" \
          --storage-root "$LOCAL_STORAGE_ROOT" >/dev/null
      ;;
    *) fail 2 "Unsupported terminal-validation adapter: $RUN_KIND" ;;
  esac
}

print_validated_leaf_result() {
  case "$RUN_KIND" in
    benchmark)
      local_cli core-benchmark-summary "$RUN_ID" --format markdown \
        --storage-root "$LOCAL_STORAGE_ROOT"
      ;;
    campaign) ;;
    *) fail 2 "Unsupported validated-result presentation adapter: $RUN_KIND" ;;
  esac
}

run_leaf_plan() {
  RUN_KIND="$1"
  RUN_LEAF_CONFIG="$2"
  EXPECTED_RUN_ID="$3"
  CAMPAIGN_PURPOSE="$4"
  RUN_LEAF_PROFILE="$5"
  resolve_leaf_plan
  if [[ "$LEAF_STATE" == complete ]]; then
    generation_console_stage 2 9 "Existing state" REUSED "$LEAF_EXISTING_DETAIL"
    print_validated_leaf_result
    return
  fi
  generation_console_stage 2 9 "Existing state" OK "$LEAF_EXISTING_DETAIL"

  if [[ "$LEAF_STATE" == package_only ]]; then
    generation_console_stage 3 9 "Canonical inputs" REUSED \
      "validated host case.h5 publications; CPU input preparation skipped"
    generation_console_stage 4 9 "Work-unit plan" REUSED \
      "package request adds zero Generation work units and zero COMSOL submissions"
    generation_console_stage 5 9 "Work units" REUSED \
      "completed source run=$RUN_ID retained as immutable base"
    generation_console_stage 6 9 "Host publication" REUSED \
      "validated transferred artifact inventory"
    ALL_STAGE="missing declared package extension"
    generation_console_stage 7 9 "Packages/finalizer" RUNNING
    build_leaf_packages_and_finalizers
    generation_console_stage 7 9 "Packages/finalizer" OK "$LEAF_PACKAGE_DETAIL"
    if [[ "$COMPOSITE_CHILD_MODE" == true ]]; then
      prepare_leaf_for_parent
      return
    fi
    generation_console_stage 8 9 "Retention policy" REUSED \
      "historical CPU cleanup policy and receipts remain unchanged"
    ALL_STAGE="terminal package-extension validation"
    validate_leaf_result
    LEAF_STATE=complete
    generation_console_stage 9 9 "Final validation" OK \
      "run_id=$RUN_ID kind=$RUN_KIND package_only=true"
    return
  fi

  if [[ "$LEAF_STATE" == transfer_repair ]]; then
    generation_console_stage 3 9 "Canonical inputs" REUSED \
      "canonical CPU scientific publication already complete"
    generation_console_stage 4 9 "Work-unit plan" REUSED \
      "repair continuation adds zero Generation work units and zero COMSOL submissions"
    generation_console_stage 5 9 "Work units" REUSED \
      "completed source run=$RUN_ID remains authoritative"
    if [[ "$DEFER_COLLECTION" == true ]]; then
      LEAF_STATE=awaiting_collection
      deferred_leaf_report
      return
    fi
    ALL_STAGE="repairable host transfer publication"
    generation_console_stage 6 9 "Host publication" RUNNING
    collect_leaf_results
    generation_console_stage 6 9 "Host publication" OK \
      "run_id=$RUN_ID destination=$LOCAL_STORAGE_ROOT repaired_or_recollected=true"
    ALL_STAGE="revalidated declared packages and scientific finalizers"
    generation_console_stage 7 9 "Packages/finalizer" RUNNING
    build_leaf_packages_and_finalizers
    generation_console_stage 7 9 "Packages/finalizer" OK "$LEAF_PACKAGE_DETAIL"
    if [[ "$COMPOSITE_CHILD_MODE" == true ]]; then
      prepare_leaf_for_parent
      return
    fi
    ALL_STAGE="reconstructed workflow receipt and guarded CPU retention policy"
    generation_console_stage 8 9 "Retention policy" RUNNING
    apply_leaf_retention
    generation_console_stage 8 9 "Retention policy" OK \
      "keep_cpu_source=$KEEP_CPU_SOURCE reclaimed_bytes=$CPU_BYTES_RECLAIMED"
    ALL_STAGE="terminal repaired workflow validation"
    validate_leaf_result
    print_validated_leaf_result
    LEAF_STATE=complete
    generation_console_stage 9 9 "Final validation" OK \
      "run_id=$RUN_ID kind=$RUN_KIND transfer_repair=true"
    return
  fi

  ALL_STAGE="canonical input readiness"
  generation_console_stage 3 9 "Canonical inputs" RUNNING
  materialize_leaf_inputs
  generation_console_stage 3 9 "Canonical inputs" OK "$LEAF_INPUT_DETAIL"

  ALL_STAGE="common work-unit plan admission"
  generation_console_stage 4 9 "Work-unit plan" RUNNING
  submit_leaf_units
  generation_console_stage 4 9 "Work-unit plan" OK "$LEAF_PLAN_DETAIL"

  ALL_STAGE="common work-unit monitoring and CPU finalization"
  monitor_generation_units "$RUN_KIND"
  if [[ "${CAMPAIGN_PARTIAL:-false}" == true ]]; then
    LEAF_RESULT=PARTIAL
    generation_console_stage 5 9 "Work units" OK \
      "run_id=$RUN_ID kind=$RUN_KIND state=completed_with_failures partial=true"
  else
    finalize_leaf_cpu_evidence
    generation_console_stage 5 9 "Work units" OK \
      "run_id=$RUN_ID kind=$RUN_KIND state=cpu_complete"
  fi

  if [[ "$DEFER_COLLECTION" == true ]]; then
    LEAF_STATE=awaiting_collection
    deferred_leaf_report
    return
  fi

  ALL_STAGE="atomic host publication"
  generation_console_stage 6 9 "Host publication" RUNNING
  collect_leaf_results
  generation_console_stage 6 9 "Host publication" OK \
    "run_id=$RUN_ID destination=$LOCAL_STORAGE_ROOT"

  ALL_STAGE="declared packages and scientific finalizers"
  generation_console_stage 7 9 "Packages/finalizer" RUNNING
  build_leaf_packages_and_finalizers
  generation_console_stage 7 9 "Packages/finalizer" OK "$LEAF_PACKAGE_DETAIL"
  if [[ "$COMPOSITE_CHILD_MODE" == true \
    && "${CAMPAIGN_PARTIAL:-false}" != true ]]; then
    prepare_leaf_for_parent
    return
  fi

  ALL_STAGE="workflow receipt and guarded CPU retention policy"
  generation_console_stage 8 9 "Retention policy" RUNNING
  apply_leaf_retention
  generation_console_stage 8 9 "Retention policy" OK \
    "keep_cpu_source=$KEEP_CPU_SOURCE reclaimed_bytes=$CPU_BYTES_RECLAIMED"

  ALL_STAGE="terminal common workflow validation"
  validate_leaf_result
  print_validated_leaf_result
  LEAF_STATE=complete
  generation_console_stage 9 9 "Final validation" OK \
    "run_id=$RUN_ID kind=$RUN_KIND"
}

run_workflow_plan() {
  local index
  WORKFLOW_CHILD_RUN_IDS=()
  PAIRED_SMOKE_RECEIPT=""
  local workflow_result=REUSED workflow_children workflow_partial=false
  COMPOSITE_CHILD_MODE=true
  for ((index=0; index<RUN_CHILD_COUNT; index++)); do
    run_leaf_plan campaign \
      "${RUN_CHILD_CONFIGS[index]}" "${RUN_CHILD_IDENTITIES[index]}" \
      "${RUN_CHILD_PURPOSES[index]}" "${RUN_CHILD_PROFILES[index]}"
    WORKFLOW_CHILD_RUN_IDS+=("$RUN_ID")
    if [[ "$LEAF_RESULT" == PARTIAL ]]; then
      workflow_partial=true
    elif [[ "$LEAF_RESULT" != REUSED ]]; then
      workflow_result=OK
    fi
    case "$LEAF_STATE" in
      complete|ready_for_parent) ;;
      awaiting_collection)
        WORKFLOW_STATE="$LEAF_STATE"
        ;;
      *) fail 1 "Workflow child returned unsupported state: $LEAF_STATE" ;;
    esac
  done
  COMPOSITE_CHILD_MODE=false
  if [[ "${WORKFLOW_STATE:-}" == awaiting_collection ]]; then
    printf 'AWAITING: workflow=%s state=awaiting_collection\n' "$RUN_PLAN_ID"
    return
  fi
  if [[ "$workflow_partial" == true ]]; then
    LEAF_RESULT=PARTIAL
    LEAF_STATE=complete
    WORKFLOW_STATE=complete
    return
  fi
  (( ${#WORKFLOW_CHILD_RUN_IDS[@]} == 2 )) ||
    fail 1 "Paired Technical Smoke requires exactly two host-complete children."
  printf -v workflow_children "children=%s,%s" \
    "${WORKFLOW_CHILD_RUN_IDS[0]}" "${WORKFLOW_CHILD_RUN_IDS[1]}"

  ALL_STAGE="paired Technical Smoke finalizer"
  generation_console_stage 8 9 "Paired finalizer" RUNNING
  finalize_smoke_runs \
    "${WORKFLOW_CHILD_RUN_IDS[0]}" "${WORKFLOW_CHILD_RUN_IDS[1]}"
  ALL_STAGE="complete parent workflow validation before cleanup"
  generation_run_with_heartbeat \
    "parent-validation-$RUN_PLAN_ID" 8 9 "Paired finalizer" \
    "validating complete paired workflow before cleanup" "${workflow_children}" \
    local_cli validate-real-smoke "$PAIRED_SMOKE_RECEIPT" \
      --storage-root "$LOCAL_STORAGE_ROOT" >/dev/null
  generation_console_stage 8 9 "Paired finalizer" OK \
    "workflow=$RUN_PLAN_ID ${workflow_children} parent_success=true"

  for ((index=0; index<RUN_CHILD_COUNT; index++)); do
    RUN_KIND=campaign
    RUN_ID="${WORKFLOW_CHILD_RUN_IDS[index]}"
    RUN_LEAF_CONFIG="${RUN_CHILD_CONFIGS[index]}"
    CAMPAIGN_PURPOSE="${RUN_CHILD_PURPOSES[index]}"
    RUN_LEAF_PROFILE="${RUN_CHILD_PROFILES[index]}"
    ALL_STAGE="parent-authorized child CPU retention policy"
    generation_console_stage 8 9 "Retention policy" RUNNING \
      "parent_success=true child_run=$RUN_ID"
    apply_leaf_retention
    generation_console_stage 8 9 "Retention policy" OK \
      "child_run=$RUN_ID keep_cpu_source=$KEEP_CPU_SOURCE reclaimed_bytes=$CPU_BYTES_RECLAIMED"
  done

  for ((index=0; index<RUN_CHILD_COUNT; index++)); do
    RUN_KIND=campaign
    RUN_ID="${WORKFLOW_CHILD_RUN_IDS[index]}"
    RUN_LEAF_CONFIG="${RUN_CHILD_CONFIGS[index]}"
    CAMPAIGN_PURPOSE="${RUN_CHILD_PURPOSES[index]}"
    RUN_LEAF_PROFILE="${RUN_CHILD_PROFILES[index]}"
    ALL_STAGE="terminal child validation after parent-owned retention"
    generation_console_stage 9 9 "Final validation" RUNNING \
      "child_run=$RUN_ID"
    validate_leaf_result
    generation_console_stage 9 9 "Final validation" OK \
      "child_run=$RUN_ID parent_success=true"
  done

  ALL_STAGE="paired receipt stability after parent-owned retention"
  generation_console_stage 9 9 "Final validation" RUNNING \
    "workflow=$RUN_PLAN_ID post_cleanup=true"
  generation_run_with_heartbeat \
    "post-cleanup-parent-validation-$RUN_PLAN_ID" 9 9 "Final validation" \
    "revalidating paired receipt after child retention" "${workflow_children}" \
    local_cli validate-real-smoke "$PAIRED_SMOKE_RECEIPT" \
      --storage-root "$LOCAL_STORAGE_ROOT" >/dev/null
  generation_console_stage 9 9 "Final validation" OK \
    "workflow=$RUN_PLAN_ID ${workflow_children} post_cleanup=true"
  LEAF_RESULT="$workflow_result"
  LEAF_STATE=complete
  WORKFLOW_STATE=complete
}

preflight_generation_plan() {
  case "$RUN_KIND" in
    campaign)
      RUN_LEAF_CONFIG="$RUN_PLAN_CONFIG"
      resolve_campaign "$RUN_LEAF_CONFIG"
      resolve_configured_resources executable
      validate_resources
      ;;
    benchmark)
      RUN_LEAF_CONFIG="$RUN_PLAN_CONFIG"
      resolve_benchmark_contract >/dev/null
      ;;
    workflow)
      RUN_LEAF_CONFIG="${RUN_CHILD_CONFIGS[0]}"
      resolve_campaign "$RUN_LEAF_CONFIG"
      resolve_configured_resources executable
      validate_resources
      ;;
    *) fail 2 "Unsupported common preflight run kind: $RUN_KIND" ;;
  esac
  resolve_remote_layout
  verify_remote_setup >/dev/null
  local version
  version="$(remote_comsol_version)"
  printf 'PREFLIGHT COMPLETE: plan=%s kind=%s host=%s COMSOL=%s\n'     "$RUN_PLAN_ID" "$RUN_KIND" "$CPU_HOST" "$version"
}

execute_generation_run() {
  HUMAN_WORKFLOW_MODE=true
  ALL_WORKFLOW_ACTIVE=true
  ALL_STAGE="common plan resolution"
  generation_console_stage 1 9 "Run plan" OK     "kind=$RUN_KIND identity=$RUN_PLAN_ID config=$RUN_CONFIG_RELATIVE"
  case "$RUN_KIND" in
    campaign)
      run_leaf_plan campaign "$RUN_PLAN_CONFIG" "$RUN_PLAN_ID" \
        "$RUN_PLAN_PURPOSE" "$RUN_PLAN_PROFILE"
      case "$LEAF_STATE" in
        complete)
          generation_console_final             "run_identity=$RUN_PLAN_ID campaign_run_id=$RUN_ID state=${REMOTE_CAMPAIGN_STATE:-complete} result=$LEAF_RESULT"
          ;;
        awaiting_collection)
          printf 'AWAITING: run_identity=%s state=%s\n'             "$RUN_PLAN_ID" "$LEAF_STATE"
          ;;
      esac
      ;;
    benchmark)
      run_leaf_plan benchmark "$RUN_PLAN_CONFIG" "$RUN_PLAN_ID" "-" "-"
      case "$LEAF_STATE" in
        complete)
          generation_console_final             "run_identity=$RUN_PLAN_ID benchmark_run_id=$RUN_ID state=complete result=$LEAF_RESULT"
          ;;
        awaiting_collection)
          printf 'AWAITING: run_identity=%s state=awaiting_collection\n'             "$RUN_PLAN_ID"
          ;;
      esac
      ;;
    workflow)
      WORKFLOW_STATE=""
      run_workflow_plan
      if [[ "$WORKFLOW_STATE" == complete ]]; then
        generation_console_final           "run_identity=$RUN_PLAN_ID state=complete result=$LEAF_RESULT children=${WORKFLOW_CHILD_RUN_IDS[*]}"
      fi
      ;;
    *) fail 2 "Unsupported Generation run kind: $RUN_KIND" ;;
  esac
  ALL_WORKFLOW_ACTIVE=false
}

benchmark_status_report() {
  resolve_local_storage
  resolve_local_python
  resolve_remote_layout
  printf 'Benchmark status:\n'
  remote_cli_retryable "benchmark status read" core-benchmark-status \
    "${RUN_ID}" --storage-root "${REMOTE_STORAGE_ROOT}" --format summary
  printf 'CPU source status:\n'
  remote_cli_retryable "benchmark source-status read" \
    core-benchmark-source-status "${RUN_ID}" \
    --storage-root "${REMOTE_STORAGE_ROOT}"
  if local_cli validate-core-benchmark "${RUN_ID}" \
    --storage-root "${LOCAL_STORAGE_ROOT}" >/dev/null 2>&1; then
    printf 'Host publication state: complete\n'
  else
    printf 'Host publication state: absent_or_incomplete\n'
  fi
}

status_generation_target() {
  local target="$1"
  local target_is_config=false
  if [[ "${target}" == /* ]]; then
    [[ -f "${target}" ]] && target_is_config=true
  elif [[ -f "${HOST_REPO_ROOT}/${target}" ]]; then
    target_is_config=true
  fi
  if [[ "${target_is_config}" == true ]]; then
    RUN_CONFIG_ARGUMENT="${target}"
    ALLOW_INCOMPLETE_PLAN=true
    resolve_generation_run_plan
    printf 'run_identity=%s\nrun_kind=%s\nconfig=%s\n' \
      "${RUN_PLAN_ID}" "${RUN_KIND}" "${RUN_CONFIG_RELATIVE}"
    case "${RUN_KIND}" in
      campaign)
        RUN_ID="${RUN_PLAN_ID}"
        RUN_LEAF_CONFIG="${RUN_PLAN_CONFIG}"
        resolve_campaign "${RUN_LEAF_CONFIG}"
        resolve_configured_resources
        storage_status_report
        ;;
      benchmark)
        RUN_LEAF_CONFIG="${RUN_PLAN_CONFIG}"
        resolve_benchmark_contract >/dev/null
        resolve_remote_layout
        local version identity_json
        version="$(remote_comsol_version)"
        identity_json="$(local_cli resolve-core-benchmark-run \
          "${BENCHMARK_SUITE_PATH}" --git-commit "${REQUESTED_COMMIT}" \
          --comsol-version-output "${version}")" ||
          fail 1 "Could not resolve benchmark runtime identity for status."
        RUN_ID="$(printf '%s' "${identity_json}" | local_python -c \
          'import json, sys; print(json.load(sys.stdin)["benchmark_run_id"])')"
        validate_benchmark_run_id "${RUN_ID}"
        printf 'benchmark_run_id=%s\n' "${RUN_ID}"
        benchmark_status_report
        ;;
      workflow)
        local index
        for ((index=0; index<RUN_CHILD_COUNT; index++)); do
          RUN_ID="${RUN_CHILD_IDENTITIES[index]}"
          RUN_LEAF_CONFIG="${RUN_CHILD_CONFIGS[index]}"
          resolve_campaign "${RUN_LEAF_CONFIG}"
          resolve_configured_resources
          printf 'Child %s/%s: %s\n' \
            "$((index + 1))" "${RUN_CHILD_COUNT}" "${RUN_ID}"
          storage_status_report
        done
        if local_cli validate-real-smoke \
          --storage-root "${LOCAL_STORAGE_ROOT}" >/dev/null 2>&1; then
          printf 'Package/finalizer state: complete\n'
        else
          printf 'Package/finalizer state: absent_or_incomplete\n'
        fi
        ;;
      *) fail 2 "Unsupported Generation status plan kind: ${RUN_KIND}" ;;
    esac
    return
  fi
  if [[ "${target}" == core_scaling_transient__* ]]; then
    validate_benchmark_run_id "${target}"
    RUN_ID="${target}"
    RUN_LEAF_CONFIG="${BENCHMARK_SUITE_RELATIVE_PATH}"
    resolve_benchmark_contract >/dev/null
    benchmark_status_report
    return
  fi
  validate_run_id "${target}"
  RUN_ID="${target}"
  resolve_workflow_campaigns
  storage_status_report
}

cancel_generation_run() {
  if [[ "${RUN_ID}" == core_scaling_transient__* ]]; then
    validate_benchmark_run_id "${RUN_ID}"
    RUN_LEAF_CONFIG="${BENCHMARK_SUITE_RELATIVE_PATH}"
    resolve_benchmark_contract >/dev/null
    resolve_remote_layout
    local -a benchmark_arguments=(
      cancel-core-benchmark "${RUN_ID}"
      --storage-root "${REMOTE_STORAGE_ROOT}"
    )
    [[ "${FORCE_CANCEL}" != true ]] || benchmark_arguments+=(--force)
    remote_cli "${benchmark_arguments[@]}"
    return
  fi
  validate_run_id "${RUN_ID}"
  resolve_workflow_campaigns
  resolve_remote_layout
  local -a campaign_arguments=(
    cancel-campaign "${RUN_ID}" --storage-root "${REMOTE_STORAGE_ROOT}"
  )
  [[ "${FORCE_CANCEL}" != true ]] || campaign_arguments+=(--force)
  remote_cli "${campaign_arguments[@]}"
}

cleanup_generation_run() {
  if [[ "${RUN_ID}" == core_scaling_transient__* ]]; then
    validate_benchmark_run_id "${RUN_ID}"
    RUN_LEAF_CONFIG="${BENCHMARK_SUITE_RELATIVE_PATH}"
    resolve_benchmark_contract >/dev/null
    resolve_remote_layout
    resolve_local_storage
    resolve_local_python
    cleanup_core_benchmark_cpu
    return
  fi
  validate_run_id "${RUN_ID}"
  resolve_workflow_campaigns
  cleanup_cpu_source
}

(( $# > 0 )) || { usage; exit 2; }
[[ "$1" != -h && "$1" != --help ]] || { usage; exit 0; }

case "${ORIGINAL_ARGUMENTS[0]}" in
  background-status)
    background_status_command
    exit 0
    ;;
  background-list)
    background_list_command
    exit 0
    ;;
esac

for bootstrap_argument in "${ORIGINAL_ARGUMENTS[@]}"; do
  if [[ "${bootstrap_argument}" == --background ]]; then
    launch_background_workflow
    exit 0
  fi
done

if [[ "${GENERATION_WORKFLOW_PINNED_HANDOFF:-}" != 1 ]]; then
  resolve_bootstrap_requested_commit
  resolve_host_layout
  trap 'workflow_exit_handler $?' EXIT
  resolve_local_commit
  handoff_to_pinned_workflow
fi

SUBCOMMAND="$1"
shift
CPU_HOST="${GENERATION_CPU_HOST:-}"
REMOTE_ROOT=""
REQUESTED_COMMIT=""
EXECUTE_SETUP=false
CONFIRM_CLEANUP=false
KEEP_CPU_SOURCE=false
DEFER_COLLECTION=false
FORCE_CANCEL=false
DRY_RUN=false
PREFLIGHT_ONLY=false
ALLOW_INCOMPLETE_PLAN=false
POSITIONAL=()

while (( $# > 0 )); do
  case "$1" in
    --cpu-host)
      (( $# >= 2 )) || fail 2 "--cpu-host requires a value."
      CPU_HOST="$2"
      shift 2
      ;;
    --remote-root)
      (( $# >= 2 )) || fail 2 "--remote-root requires a value."
      REMOTE_ROOT="$2"
      shift 2
      ;;
    --git-commit)
      (( $# >= 2 )) || fail 2 "--git-commit requires a value."
      REQUESTED_COMMIT="$2"
      shift 2
      ;;
    --execute) EXECUTE_SETUP=true; shift ;;
    --confirm) CONFIRM_CLEANUP=true; shift ;;
    --force) FORCE_CANCEL=true; shift ;;
    --keep-cpu-source) KEEP_CPU_SOURCE=true; shift ;;
    --defer-collection) DEFER_COLLECTION=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    --preflight-only) PREFLIGHT_ONLY=true; shift ;;
    -h|--help) usage; exit 0 ;;
    --*) fail 2 "Unsupported option: $1" ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done

[[ -z "${REQUESTED_COMMIT}" ]] || validate_commit "${REQUESTED_COMMIT}"
if [[ "${DEFER_COLLECTION}" == true && "${KEEP_CPU_SOURCE}" == true ]]; then
  fail 2 "--defer-collection cannot be combined with --keep-cpu-source."
fi
if [[ "${DRY_RUN}" == true && "${PREFLIGHT_ONLY}" == true ]]; then
  fail 2 "--dry-run cannot be combined with --preflight-only."
fi

resolve_host_layout
trap 'workflow_exit_handler $?' EXIT
resolve_local_commit
handoff_to_pinned_workflow

case "${SUBCOMMAND}" in
  run)
    (( ${#POSITIONAL[@]} == 1 )) ||
      fail 2 "run requires exactly one Generation config."
    [[ "${EXECUTE_SETUP}" == false && "${CONFIRM_CLEANUP}" == false \
      && "${FORCE_CANCEL}" == false ]] ||
      fail 2 "run received an administrative-only option."
    RUN_CONFIG_ARGUMENT="${POSITIONAL[0]}"
    [[ "${DRY_RUN}" != true ]] || ALLOW_INCOMPLETE_PLAN=true
    resolve_generation_run_plan
    if [[ "${DRY_RUN}" == true ]]; then
      printf '%s\n' "${RUN_PLAN_JSON}"
      exit 0
    fi
    if [[ "${PREFLIGHT_ONLY}" == true ]]; then
      preflight_generation_plan
      exit 0
    fi
    execute_generation_run
    ;;
  setup-cpu)
    (( ${#POSITIONAL[@]} == 0 )) ||
      fail 2 "setup-cpu accepts no positional arguments."
    [[ "${CONFIRM_CLEANUP}" == false && "${FORCE_CANCEL}" == false \
      && "${KEEP_CPU_SOURCE}" == false && "${DEFER_COLLECTION}" == false \
      && "${DRY_RUN}" == false && "${PREFLIGHT_ONLY}" == false ]] ||
      fail 2 "setup-cpu received an unsupported option."
    setup_cpu
    ;;
  status)
    (( ${#POSITIONAL[@]} == 1 )) ||
      fail 2 "status requires exactly one config or run ID."
    [[ "${EXECUTE_SETUP}" == false && "${CONFIRM_CLEANUP}" == false \
      && "${FORCE_CANCEL}" == false && "${KEEP_CPU_SOURCE}" == false \
      && "${DEFER_COLLECTION}" == false && "${DRY_RUN}" == false \
      && "${PREFLIGHT_ONLY}" == false ]] ||
      fail 2 "status received an unsupported option."
    status_generation_target "${POSITIONAL[0]}"
    ;;
  cancel)
    (( ${#POSITIONAL[@]} == 1 )) ||
      fail 2 "cancel requires exactly one run ID."
    [[ "${EXECUTE_SETUP}" == false && "${CONFIRM_CLEANUP}" == false \
      && "${KEEP_CPU_SOURCE}" == false && "${DEFER_COLLECTION}" == false \
      && "${DRY_RUN}" == false && "${PREFLIGHT_ONLY}" == false ]] ||
      fail 2 "cancel received an unsupported option."
    RUN_ID="${POSITIONAL[0]}"
    cancel_generation_run
    ;;
  cleanup)
    (( ${#POSITIONAL[@]} == 1 )) ||
      fail 2 "cleanup requires exactly one run ID."
    [[ "${CONFIRM_CLEANUP}" == true ]] ||
      fail 2 "cleanup requires --confirm."
    [[ "${EXECUTE_SETUP}" == false && "${FORCE_CANCEL}" == false \
      && "${KEEP_CPU_SOURCE}" == false && "${DEFER_COLLECTION}" == false \
      && "${DRY_RUN}" == false && "${PREFLIGHT_ONLY}" == false ]] ||
      fail 2 "cleanup received an unsupported option."
    RUN_ID="${POSITIONAL[0]}"
    cleanup_generation_run
    ;;
  *)
    usage
    fail 2 "Unsupported subcommand: ${SUBCOMMAND}. Start or resume Generation work with: $0 run CONFIG"
    ;;
esac
