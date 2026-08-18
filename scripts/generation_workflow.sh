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
CPU_BYTES_RECLAIMED=0
CPU_CLEANUP_RECEIPT_SHA=""
PILOT_CASES_PER_MATERIAL=""
PILOT_MATERIAL_COUNT=""
PILOT_TOTAL_CASES=""
SKIP_EXTREME_FAMILY_OOD=false
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
SMOKE_PROFILE_MODE=false
SMOKE_PROFILE_NAME=""

usage() {
  cat >&2 <<EOF
Usage:
  $0 setup-cpu [--cpu-host HOST] [--remote-root PATH] [--git-commit COMMIT] [--execute]
  $0 preflight CAMPAIGN [options]
  $0 plan CAMPAIGN [options]
  $0 launch CAMPAIGN [options]
  $0 all CAMPAIGN [--background] [--defer-collection|--keep-cpu-source] [options]
  $0 smoke [--background] [--defer-collection] [options]
  $0 finalize-smoke STEADY_CAMPAIGN_RUN_ID TRANSIENT_CAMPAIGN_RUN_ID [remote options]
  $0 benchmark-cores [--variant VARIANT_ID] [--background] [--defer-collection] [remote options]
  $0 collect-benchmark BENCHMARK_RUN_ID [remote options]
  $0 pilot-check CAMPAIGN [--cases-per-material N] [--background] [--defer-collection|--keep-cpu-source] [options]
  $0 status [CAMPAIGN_RUN_ID] [remote options]
  $0 collect|build-datasets|resume CAMPAIGN_RUN_ID [options]
  $0 background-status WORKFLOW_SESSION_ID
  $0 background-list
  $0 retry-case CAMPAIGN_RUN_ID BATCH_NAME CASE_ID [remote options]
  $0 cleanup CAMPAIGN_RUN_ID [--confirm] [remote options]
  $0 accounting|validate CAMPAIGN_RUN_ID [remote options]
  $0 cancel CAMPAIGN_RUN_ID [--force] [remote options]

Remote options:
  --cpu-host HOST       explicit override for the configured CPU site
  --remote-root PATH    bootstrap layout default: remote HOME/grainlegumes-generation
  --git-commit COMMIT   exact lowercase 40-character commit
  --only-batch NAME     one predeclared batch
  --skip-extreme-family-ood
                         skip only the extreme-family batch for this execution

Execution resources and feeder cadence are owned only by the selected campaign
configuration. setup-cpu and cleanup are dry runs unless --execute or --confirm
is supplied.
collect validates and publishes GPU generation data without deleting CPU sources.
all waits synchronously, builds every package, smokes every loader, and cleans the
verified CPU source by default. --keep-cpu-source still collects but retains CPU data.
--defer-collection validates CPU results without host transfer, package build, or cleanup.
smoke owns the canonical paired two-profile technical run and retains CPU source.
benchmark-cores runs the four isolated same-case transient core variants; --variant retries one.
pilot-check runs configured-material transient diagnostics and safely cleans CPU source by default.
EOF
}
fail() {
  local status="$1"
  shift
  printf '%s\n' "$*" >&2
  exit "${status}"
}

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
  local changed_progress_seconds=60 heartbeat_seconds=300
  now="$(date +%s)"
  if [[ "${CONSOLE_PROGRESS_KEY}" == "${key}" && "${CONSOLE_PROGRESS_SIGNATURE}" == "${signature}" ]]; then
    if [[ "${CONSOLE_PROGRESS_DETAIL_SIGNATURE}" == "${detail_signature}" ]]; then
      if (( now - CONSOLE_PROGRESS_RENDERED_AT < heartbeat_seconds )); then
        return
      fi
      detail="${detail}${detail:+$'\n'}heartbeat=unchanged"
    elif (( now - CONSOLE_PROGRESS_RENDERED_AT < changed_progress_seconds )); then
      return
    fi
  fi
  generation_console_stage "${index}" "${total}" "${label}" "${status}" "${detail}"
  CONSOLE_PROGRESS_KEY="${key}"
  CONSOLE_PROGRESS_SIGNATURE="${signature}"
  CONSOLE_PROGRESS_DETAIL_SIGNATURE="${detail_signature}"
  CONSOLE_PROGRESS_RENDERED_AT="${now}"
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

campaign_workflow_complete() {
  if [[ "${SMOKE_PROFILE_MODE}" == true ]]; then
    printf 'PROFILE COMPLETE:\nprofile=%s\ncampaign_run_id=%s\nworkflow_receipt=validated\n' \
      "${SMOKE_PROFILE_NAME}" "${RUN_ID}"
    return
  fi
  generation_console_final "campaign_run_id=${RUN_ID} workflow receipt validated"
}

disarm_campaign_interrupt() {
  CAMPAIGN_INTERRUPT_ACTIVE=false
  trap - INT
}

campaign_interrupt_handler() {
  [[ "${CAMPAIGN_INTERRUPT_ACTIVE}" == true && -n "${RUN_ID}" ]] || return 130
  CAMPAIGN_INTERRUPT_COUNT=$(( CAMPAIGN_INTERRUPT_COUNT + 1 ))
  if (( CAMPAIGN_INTERRUPT_COUNT == 1 )); then
    printf '%s\n' \
      'Graceful campaign cancellation requested.' \
      'Press Ctrl+C again to force cancellation.' >&2
    remote_cli cancel-campaign "${RUN_ID}" \
      --storage-root "${REMOTE_STORAGE_ROOT}" >/dev/null &
    local cancellation_pid=$!
    if ! wait "${cancellation_pid}"; then
      generation_console_warning \
        "graceful cancellation request failed; campaign state remains authoritative"
    fi
    return 0
  fi
  printf '%s\n' 'Force campaign cancellation requested.' >&2
  if ! remote_cli cancel-campaign "${RUN_ID}" --force \
    --storage-root "${REMOTE_STORAGE_ROOT}" >/dev/null; then
    generation_console_warning \
      "force cancellation request failed; inspect scheduler and campaign evidence"
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
  ensure_execution_bootstrap
  require_command ssh "CPU login control"
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
  case "${subcommand}" in
    all|resume|smoke|benchmark-cores|pilot-check|collect|build-datasets|finalize-smoke|collect-benchmark) ;;
    *) fail 2 "--background is not supported for ${subcommand}." ;;
  esac
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
  if [[ "${subcommand}" != build-datasets ]]; then
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
  printf 'BACKGROUND STARTED\nworkflow_session_id=%s\ntmux_session=%s\nhost=%s\nsource_commit=%s\npid=%s\nlog=%s\n\nAttach:\n  tmux attach-session -t %q\n\nDetach without stopping:\n  press Ctrl+B, then D\n\nStatus:\n  %q background-status %q\n\nFollow log:\n  tail -n 100 -F %q\n\nThe workflow survives terminal/SSH disconnection.\nIt does not survive a reboot of %s; use resume afterwards.\n' \
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
    workflow["family_generalization"]["stationary"]["repository_path"],
    workflow["family_generalization"]["transient"]["repository_path"],
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
  local record kind configured_cores_per_case configured_wall_time
  local configured_cores_per_node configured_pending_buffer configured_poll_interval
  local configured_max_running configured_cpu_host configured_scheduler
  local configured_partition configured_python_module configured_comsol_module
  local configured_python_executable configured_comsol_executable extra
  record="$(local_cli validate-config "${CAMPAIGN_CONFIG_PATH}" --allow-incomplete |
    local_python -c 'import json, sys
value = json.load(sys.stdin)
resources = value["execution_resources"]
cluster = resources["cluster"]
submission = resources["submission"]
site = resources["site"]
wall = cluster.get("wall_time")
max_running = submission.get("max_running_cases")
fields = (
    value["campaign_purpose"], cluster["cores_per_case"],
    "-" if wall is None else wall, cluster["cores_per_node"],
    submission["pending_buffer"], submission["poll_interval_seconds"],
    "-" if max_running is None else max_running,
    site["cpu_host"], site["scheduler"], site["partition"],
    site["python_module"], site["comsol_module"],
    site["python_executable"], site["comsol_executable"],
)
if any("\t" in str(item) or "\n" in str(item) or "\r" in str(item) for item in fields):
    raise SystemExit("execution configuration contains unsafe shell transport text")
print("\t".join(("execution", *(str(item) for item in fields))))')" ||
    fail 1 "Could not resolve configured campaign execution."
  IFS=$'\t' read -r kind CAMPAIGN_PURPOSE configured_cores_per_case \
    configured_wall_time configured_cores_per_node configured_pending_buffer \
    configured_poll_interval configured_max_running configured_cpu_host \
    configured_scheduler configured_partition configured_python_module \
    configured_comsol_module configured_python_executable \
    configured_comsol_executable extra <<< "${record}"
  [[ "${kind}" == execution && -z "${extra:-}" ]] ||
    fail 1 "Malformed configured execution record."
  validate_positive "configured cores_per_case" "${configured_cores_per_case}"
  validate_positive "configured cores_per_node" "${configured_cores_per_node}"
  validate_positive "configured pending_buffer" "${configured_pending_buffer}"
  validate_positive "configured poll_interval_seconds" "${configured_poll_interval}"
  [[ "${configured_max_running}" == - ]] ||
    validate_positive "configured max_running_cases" "${configured_max_running}"
  [[ "${configured_wall_time}" != - ]] || configured_wall_time=""
  [[ -n "${CPU_HOST}" ]] || CPU_HOST="${configured_cpu_host}"
  SCHEDULER_KIND="${configured_scheduler}"
  PARTITION="${configured_partition}"
  CORES_PER_NODE="${configured_cores_per_node}"
  CORES_PER_CASE="${configured_cores_per_case}"
  PENDING_BUFFER="${configured_pending_buffer}"
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
  resolve_local_python
  local record kind purpose configured_count configured_total extra
  record="$(local_cli validate-config "${CAMPAIGN_CONFIG_PATH}" --allow-incomplete |
    local_python -c 'import json, sys
value = json.load(sys.stdin)
counts = tuple(value["counts"].values())
if not counts or len(set(counts)) != 1:
    raise SystemExit("pilot campaign must have one uniform cases-per-material count")
print("\t".join(("pilot", str(value["campaign_purpose"]), str(counts[0]), str(sum(counts)))))')" ||
    fail 2 "Could not resolve the dedicated pilot-check campaign contract."
  IFS=$'\t' read -r kind purpose configured_count configured_total extra <<< "${record}"
  [[ "${kind}" == pilot && "${purpose}" == pilot_check && -z "${extra:-}" ]] ||
    fail 2 "pilot-check requires a dedicated campaign with campaign_purpose: pilot_check."
  validate_positive "configured pilot cases per material" "${configured_count}"
  validate_positive "configured pilot total" "${configured_total}"
  (( configured_total % configured_count == 0 )) ||
    fail 2 "Pilot total must be divisible by its uniform cases-per-material count."
  PILOT_MATERIAL_COUNT="$((configured_total / configured_count))"
  if [[ -z "${PILOT_CASES_PER_MATERIAL}" ]]; then
    PILOT_CASES_PER_MATERIAL="${configured_count}"
  else
    validate_positive --cases-per-material "${PILOT_CASES_PER_MATERIAL}"
  fi
  PILOT_TOTAL_CASES="$((PILOT_MATERIAL_COUNT * PILOT_CASES_PER_MATERIAL))"
}

validate_resources() {
  validate_positive "configured cores_per_case" "${CORES_PER_CASE}"
  validate_positive "configured cores_per_node" "${CORES_PER_NODE}"
  validate_positive "configured pending_buffer" "${PENDING_BUFFER}"
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
  if remote_bash "${CPU_HOST}" \
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
for name in git stat module; do setup_require_command "${name}"; done
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
if ! module load "${python_module}"; then
  printf 'CPU login prerequisite failed: Python module %s (blocks setup).\n' \
    "${python_module}" >&2
  exit 1
fi
setup_require_command "${python_executable}"
[[ -x "${venv}/bin/python" ]] || "${python_executable}" -m venv "${venv}"
"${venv}/bin/python" -m pip install -e "${repository}[generation-cpu]"
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
    "${REQUESTED_COMMIT}" "${remote_campaign}" "${ONLY_BATCH}" \
    "${operation}" "${PILOT_CASES_PER_MATERIAL}" \
    "${SKIP_EXTREME_FAMILY_OOD}" "${PYTHON_MODULE}" <<'REMOTE'
set -euo pipefail
repository="$1"; storage="$2"; venv="$3"; commit="$4"; campaign="$5"
only_batch="$6"; operation="$7"; pilot_cases="$8"; skip_extreme="$9"
python_module="${10}"
module load "${python_module}"
export GENERATION_CPU_VENV="${venv}"
export STORAGE_ROOT="${storage}"
export GENERATION_GIT_COMMIT="${commit}"
cd "${repository}"
command=("${venv}/bin/python" -m src.generation.cli.cli_generation
  "${operation}" "${campaign}"
  --git-commit "${commit}" --storage-root "${storage}")
[[ -z "${only_batch}" ]] || command+=(--only-batch "${only_batch}")
[[ -z "${pilot_cases}" ]] || command+=(--pilot-cases-per-material "${pilot_cases}")
[[ "${skip_extreme}" != true ]] || command+=(--skip-extreme-family-ood)
"${command[@]}"
REMOTE
}


remote_prepare_campaign_inputs() {
  remote_plan_submit prepare-campaign-inputs
}


preflight_cpu() {
  HUMAN_WORKFLOW_MODE=true
  generation_console_stage 1 3 "Local preflight" RUNNING
  resolve_local_commit
  resolve_campaign "${CAMPAIGN_ARGUMENT}"
  resolve_configured_resources
  validate_resources
  generation_console_stage 1 3 "Local preflight" OK     "commit=${REQUESTED_COMMIT:0:12} campaign=${CAMPAIGN_RELATIVE_PATH}"

  generation_console_stage 2 3 "CPU login preflight" RUNNING
  verify_remote_setup >/dev/null
  generation_console_stage 2 3 "CPU login preflight" OK     "host=${CPU_HOST} Python=${PYTHON_MODULE}"

  generation_console_stage 3 3 "CPU compute preflight" RUNNING
  local remote_campaign record status kind job_id logs extra
  remote_campaign="$(remote_repository_path "${CAMPAIGN_RELATIVE_PATH}")"
  set +e
  record="$(remote_bash "${CPU_HOST}"     "${REMOTE_REPOSITORY}" "${REMOTE_STORAGE_ROOT}" "${REMOTE_VENV}"     "${remote_campaign}" "${ONLY_BATCH}"     "${PARTITION}" "${WALL_TIME}" "${PYTHON_MODULE}" "${COMSOL_MODULE}"     "${PYTHON_EXECUTABLE}" "${COMSOL_EXECUTABLE}" "${SCHEDULER_KIND}"     "${REQUESTED_COMMIT}" <<'REMOTE'
set -euo pipefail
repository="$1"; storage="$2"; venv="$3"; campaign="$4"; only_batch="$5"
partition="$6"; wall_time="$7"; python_module="$8"; comsol_module="$9"
python_executable="${10}"; comsol_executable="${11}"; scheduler="${12}"
commit="${13}"
preflight_id="$(date -u +%Y%m%dT%H%M%SZ)"
cd "${repository}"
meta_root="$("${venv}/bin/python" -c 'import sys; from src import common; print(common.paths.get_generation_meta_root(storage_root=sys.argv[1]))' "${storage}")"
logs="${meta_root}/preflight/${preflight_id}"
mkdir -p "${logs}"
[[ -n "${only_batch}" ]] || only_batch=-
submission=(sbatch --wait --parsable --partition="${partition}" --nodes=1 --ntasks=1
  --cpus-per-task=1 --job-name=vp2-generation-preflight
  --export="ALL,GENERATION_GIT_COMMIT=${commit}"
  --output="${logs}/slurm-%j.out" --error="${logs}/slurm-%j.err"
  --chdir="${repository}")
[[ -z "${wall_time}" ]] || submission+=(--time="${wall_time}")
set +e
job_id="$("${submission[@]}" "${repository}/scripts/generation_cpu_smoke.sh"   "${repository}" "${venv}" "${campaign}" "${storage}" "${only_batch}"   environment-only "${python_module}" "${comsol_module}"   "${python_executable}" "${comsol_executable}" "${scheduler}")"
status="$?"
set -e
job_id="${job_id%%;*}"
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
  printf 'CPU login Slurm submission failed: malformed preflight job ID %s.\n'     "${job_id}" >&2
  exit 1
fi
if (( status != 0 )); then
  [[ ! -f "${logs}/slurm-${job_id}.err" ]] || tail -n 8 "${logs}/slurm-${job_id}.err" >&2
  printf 'CPU compute-node preflight failed in Slurm job %s; logs: %s\n'     "${job_id}" "${logs}" >&2
  exit "${status}"
fi
printf 'preflight\t%s\t%s\n' "${job_id}" "${logs}"
REMOTE
)"
  status="$?"
  set -e
  if (( status != 0 )); then
    generation_console_stage 3 3 "CPU compute preflight" FAILED       "execution_started=true; inspect the reported Slurm evidence"
    return "${status}"
  fi
  IFS=$'\t' read -r kind job_id logs extra <<< "${record}"
  [[ "${kind}" == preflight && "${job_id}" =~ ^[0-9]+$ && -n "${logs}" && -z "${extra:-}" ]] ||
    fail 1 "Malformed CPU compute preflight summary."
  generation_console_stage 3 3 "CPU compute preflight" OK     "job=${job_id} evidence=${logs}"
}


validate_local_launch_gates() {
  local_cli validate-config "${CAMPAIGN_CONFIG_PATH}" >/dev/null
  [[ "${CAMPAIGN_PURPOSE}" == technical_runtime_smoke ]] && return
  local selected_campaign="${CAMPAIGN_CONFIG_PATH}"
  resolve_workflow_campaigns
  local_cli static-sentinels "${STATIONARY_PRIMARY_CAMPAIGN_HOST_PATH}" \
    "${TRANSIENT_PRIMARY_CAMPAIGN_HOST_PATH}" >/dev/null ||
    fail 2 "Static scientific sentinels block production planning or launch."
  local comsol_version
  comsol_version="$(remote_comsol_version)"
  technical_smoke_evidence_status_cpu "${selected_campaign}" "${comsol_version}" >/dev/null ||
    fail 2 "Current successful technical-smoke evidence for the selected campaign profile is required; run the technical smoke first."
  resolve_campaign "${selected_campaign}"
  resolve_configured_resources
}

technical_smoke_evidence_status_cpu() {
  local campaign_argument="$1"
  local comsol_version_output="$2"
  resolve_campaign "${campaign_argument}"
  resolve_configured_resources
  resolve_remote_layout
  local remote_campaign
  remote_campaign="$(remote_repository_path "${CAMPAIGN_RELATIVE_PATH}")"
  remote_bash "${CPU_HOST}" \
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
  remote_bash "${CPU_HOST}" "${COMSOL_MODULE}" "${COMSOL_EXECUTABLE}" <<'REMOTE'
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
    fail 2 "CPU-side technical-smoke evidence is missing, stale, or incomplete after transfer."
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
  local comsol_version steady_evidence transient_evidence receipt
  comsol_version="$(remote_comsol_version)"
  steady_evidence="$(local_cli finalize-technical-smoke-evidence "${stationary_run_id}" \
    --comsol-version-output "${comsol_version}" --storage-root "${LOCAL_STORAGE_ROOT}")" ||
    fail 1 "Could not finalize steady-flow Technical Smoke evidence for ${stationary_run_id}."
  steady_evidence="$(container_path_to_host "${steady_evidence}")"
  sync_technical_smoke_evidence \
    "${steady_evidence}" "${STATIONARY_SMOKE_CAMPAIGN_PATH}" "${comsol_version}"
  transient_evidence="$(local_cli finalize-technical-smoke-evidence "${transient_run_id}" \
    --comsol-version-output "${comsol_version}" --storage-root "${LOCAL_STORAGE_ROOT}")" ||
    fail 1 "Could not finalize transient-drying Technical Smoke evidence for ${transient_run_id}."
  transient_evidence="$(container_path_to_host "${transient_evidence}")"
  sync_technical_smoke_evidence \
    "${transient_evidence}" "${TRANSIENT_SMOKE_CAMPAIGN_PATH}" "${comsol_version}"
  receipt="$(local_cli finalize-real-smoke "${stationary_run_id}" "${transient_run_id}" \
    --comsol-version-output "${comsol_version}" --storage-root "${LOCAL_STORAGE_ROOT}")" ||
    fail 1 "Could not atomically finalize paired Technical Smoke evidence."
  receipt="$(container_path_to_host "${receipt}")"
  local_cli validate-real-smoke "${receipt}" --storage-root "${LOCAL_STORAGE_ROOT}" >/dev/null ||
    fail 1 "Paired Technical Smoke receipt did not validate after finalization: ${receipt}"
  printf 'Profile technical-smoke evidence: %s and %s\n' \
    "${steady_evidence}" "${transient_evidence}"
  printf 'Paired technical runtime diagnostic receipt: %s\n' "${receipt}"
  printf 'CPU sources retained for review for runs %s and %s.\n' \
    "${stationary_run_id}" "${transient_run_id}"
  generation_console_final "paired Technical Smoke and all workflow receipts validated"
}

run_smoke() {
  [[ "${CONFIRM_CLEANUP}" == false && -z "${ONLY_BATCH}" ]] ||
    fail 2 "smoke does not support --confirm or --only-batch."
  resolve_local_commit
  resolve_workflow_campaigns
  KEEP_CPU_SOURCE=true
  SMOKE_PROFILE_MODE=true
  SMOKE_PROFILE_NAME=steady_flow
  CAMPAIGN_ARGUMENT="${STATIONARY_SMOKE_CAMPAIGN_PATH}"
  preflight_cpu
  run_all
  local stationary_run_id="${RUN_ID}"
  SMOKE_PROFILE_NAME=transient_drying
  CAMPAIGN_ARGUMENT="${TRANSIENT_SMOKE_CAMPAIGN_PATH}"
  run_all
  local transient_run_id="${RUN_ID}"
  SMOKE_PROFILE_MODE=false
  SMOKE_PROFILE_NAME=""
  if [[ "${DEFER_COLLECTION}" == true ]]; then
    printf 'Later paired Smoke finalization (after both collect and build-datasets commands):\n'
    print_command "${HOST_REPO_ROOT}/scripts/generation_workflow.sh" finalize-smoke \
      "${stationary_run_id}" "${transient_run_id}" --cpu-host "${CPU_HOST}" \
      --remote-root "${REMOTE_ROOT}" --git-commit "${REQUESTED_COMMIT}"
    printf 'DEFERRED: CPU Technical Smoke results validated and awaiting host collection\n'
    return
  fi
  finalize_smoke_runs "${stationary_run_id}" "${transient_run_id}"
}

plan_campaign() {
  resolve_local_commit
  resolve_campaign "${CAMPAIGN_ARGUMENT}"
  resolve_configured_resources
  validate_resources
  resolve_remote_layout
  resolve_local_storage
  resolve_local_python
  validate_local_launch_gates
  print_layout
  remote_plan_submit plan-campaign
}

launch_requested_campaign() {
  resolve_local_commit
  resolve_campaign "${CAMPAIGN_ARGUMENT}"
  resolve_configured_resources
  validate_resources
  resolve_remote_layout
  resolve_local_storage
  resolve_local_python
  validate_local_launch_gates
  launch_campaign
}

launch_campaign() {
  resolve_local_commit
  resolve_campaign "${CAMPAIGN_ARGUMENT}"
  resolve_configured_resources
  validate_resources
  resolve_remote_layout
  print_layout
  verify_remote_setup_for_output
  local output
  output="$(remote_plan_submit submit-campaign)" || fail 1 "Remote launch failed."
  if [[ "${HUMAN_WORKFLOW_MODE}" != true ]]; then
    printf '%s\n' "${output}"
  fi
  if [[ ${output} =~ \"campaign_run_id\"[[:space:]]*:[[:space:]]*\"([A-Za-z0-9._-]+__[0-9a-f]{16})\" ]]; then
    RUN_ID="${BASH_REMATCH[1]}"
    printf 'Campaign run ID: %s\n' "${RUN_ID}"
    ALL_STAGE="initial campaign status reconstruction"
    if ! remote_cli campaign-status "${RUN_ID}" --format summary --max-active-cases 8 \
      --storage-root "${REMOTE_STORAGE_ROOT}"; then
      printf 'Campaign was submitted, but initial status reconstruction failed.\n' >&2
      printf 'campaign_run_id=%s\n' "${RUN_ID}" >&2
      return 1
    fi
  else
    fail 1 "Launch returned no campaign-run ID."
  fi
}

remote_cli() {
  verify_remote_setup_for_output || return $?
  validate_commit "${REQUESTED_COMMIT}"
  remote_bash "${CPU_HOST}" "${REMOTE_REPOSITORY}" "${REMOTE_STORAGE_ROOT}" \
    "${REMOTE_VENV}" "${REQUESTED_COMMIT}" "${PYTHON_MODULE}" "$@" <<'REMOTE'
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

remote_transfer_plan() {
  remote_cli campaign-transfer-plan "${RUN_ID}" --format tsv --storage-root "${REMOTE_STORAGE_ROOT}"
}

resolve_local_storage() {
  require_command realpath
  LOCAL_STORAGE_ROOT="$(realpath -m -- "${HOST_STORAGE_ROOT}")"
  validate_path "local storage" "${LOCAL_STORAGE_ROOT}"
}

resolve_local_python() {
  [[ -x "${DOCKER_PYTHON}" ]] || fail 1 "Canonical Docker Python runner is not executable: ${DOCKER_PYTHON}"
  local_python -c 'import h5py, numpy, scipy, yaml; import src.generation.cli.cli_generation' ||
    fail 1 "Canonical Docker Python environment lacks required dependencies."
}

local_python() {
  env GENERATION_GIT_COMMIT="${PINNED_SOURCE_COMMIT}" \
    STORAGE_ROOT="${LOCAL_STORAGE_ROOT:-${HOST_STORAGE_ROOT}}" \
    "${DOCKER_PYTHON}" "$@"
}

local_cli() {
  local_python -m "${GENERATION_MODULE}" "$@"
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
  local_cli validate-published-campaign "${RUN_ID}" --storage-root "${LOCAL_STORAGE_ROOT}" >/dev/null 2>&1
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
  if [[ -n "${PILOT_CASES_PER_MATERIAL}" ]]; then
    remote_cli record-pilot-source-inventory "${RUN_ID}" \
      --storage-root "${REMOTE_STORAGE_ROOT}" >/dev/null ||
      fail 1 "Could not record exact pre-cleanup CPU pilot storage."
  fi
  local plan
  plan="$(remote_transfer_plan)" || fail 1 "Remote campaign is not terminally valid."
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
  local staging receipt
  staging="$(local_cli create-transfer-staging "${RUN_ID}" \
    --storage-root "${LOCAL_STORAGE_ROOT}")" ||
    fail 1 "Could not create marked transfer staging."
  staging="$(container_path_to_host "${staging}")"
  printf 'Transfer staging: %s\n' "${staging}"
  for directory in "${directories[@]}"; do
    rsync -a --protect-args --relative --exclude='.state/' --exclude='work/' \
      "${CPU_HOST}:${REMOTE_STORAGE_ROOT}/./${directory}" "${staging}/" ||
      fail 1 "Transfer failed; staging retained at ${staging}."
  done
  if [[ -n "${PILOT_CASES_PER_MATERIAL}" ]]; then
    local_cli record-pilot-staging-inventory "${RUN_ID}" --staging-root "${staging}" >/dev/null ||
      fail 1 "Could not record exact pilot transfer-staging storage."
  fi
  receipt="$(local_cli publish-transferred-campaign "${RUN_ID}" \
    --staging-root "${staging}" --destination-root "${LOCAL_STORAGE_ROOT}" \
    --source-host "${CPU_HOST}" --source-storage-root "${REMOTE_STORAGE_ROOT}")" ||
    fail 1 "GPU publication validation failed; staging retained at ${staging}."
  if [[ -z "${PILOT_CASES_PER_MATERIAL}" ]]; then
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
  local_cli build-campaign-datasets "${RUN_ID}" --storage-root "${LOCAL_STORAGE_ROOT}"
}

remote_campaign_monitor() {
  remote_cli campaign-status "${RUN_ID}" --format monitor --max-active-cases 8 \
    --storage-root "${REMOTE_STORAGE_ROOT}"
}

read_remote_campaign_monitor() {
  local output header kind state state_signature progress_signature extra
  output="$(remote_campaign_monitor)" || fail 1 "Could not reconstruct campaign case status."
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
  local console_index="${1:-5}" console_total="${2:-8}" console_label="${3:-Generation}"
  validate_positive "configured poll_interval_seconds" "${STATUS_POLL_SECONDS}"
  verify_remote_setup >/dev/null
  arm_campaign_interrupt
  while true; do
    read_remote_source_status
    if [[ "${REMOTE_SOURCE_STATE}" == source_cleanup_complete ]]; then
      generation_console_progress campaign "${console_index}" "${console_total}" "${console_label}" REUSED \
        "source_cleanup_complete|0|False" \
        "campaign_run_id=${RUN_ID} source_cleanup_complete"
      disarm_campaign_interrupt
      return
    fi
    if [[ "${ALLOW_REMOTE_RESUME}" == true ]]; then
      remote_cli resume-campaign "${RUN_ID}" \
        --storage-root "${REMOTE_STORAGE_ROOT}" >/dev/null
    else
      remote_cli feed-campaign "${RUN_ID}" \
        --storage-root "${REMOTE_STORAGE_ROOT}" >/dev/null
    fi
    read_remote_campaign_monitor
    local state detail
    state="${REMOTE_CAMPAIGN_STATE}"
    detail="${REMOTE_CAMPAIGN_SUMMARY}"$'\n'"Source storage: state=${REMOTE_SOURCE_STATE} retained_bytes=${CPU_BYTES_RETAINED}"
    generation_console_progress campaign "${console_index}" "${console_total}" "${console_label}" RUNNING \
      "${REMOTE_CAMPAIGN_STATE_SIGNATURE}|${REMOTE_SOURCE_STATE}|${REMOTE_SOURCE_ACTIVE}" \
      "${detail}" \
      "${REMOTE_CAMPAIGN_PROGRESS_SIGNATURE}|${CPU_BYTES_RETAINED}"
    case "${state}" in
      successful|transfer_complete)
        remote_cli validate-campaign-terminal "${RUN_ID}" \
          --storage-root "${REMOTE_STORAGE_ROOT}" >/dev/null
        disarm_campaign_interrupt
        return
        ;;
      running|feeding|license_blocked|submission_pending_or_unknown)
        sleep "${STATUS_POLL_SECONDS}" || true
        ;;
      completed_with_failures)
        disarm_campaign_interrupt
        fail 1 $'No resumable cases remain.\nUse retry-case for an explicit failed or timed-out case.'
        ;;
      cancelled)
        disarm_campaign_interrupt
        fail 1 "Campaign is cancelled; use resume to restart interrupted work from time zero."
        ;;
      *)
        disarm_campaign_interrupt
        fail 1 "Campaign entered unsupported state: ${state}"
        ;;
    esac
  done
}


deferred_campaign_report() {
  local report kind report_run report_commit successful blocked failed extra
  report="$(remote_cli campaign-status "${RUN_ID}" --no-scheduler \
    --storage-root "${REMOTE_STORAGE_ROOT}" | local_python -c 'import json, sys
value = json.load(sys.stdin)
fields = (
    value["campaign_run_id"], value["git_commit"],
    value["completed_cases"], value["license_blocked_cases"],
    value["failed_cases"],
)
if any("\t" in str(item) or "\n" in str(item) or "\r" in str(item) for item in fields):
    raise SystemExit("campaign status contains unsafe shell transport text")
print("\t".join(("deferred", *(str(item) for item in fields))))')" ||
    fail 1 "Could not render the deferred CPU campaign summary."
  IFS=$'\t' read -r kind report_run report_commit successful blocked failed extra <<< "${report}"
  [[ "${kind}" == deferred && "${report_run}" == "${RUN_ID}" \
    && "${report_commit}" == "${REQUESTED_COMMIT}" && -z "${extra:-}" ]] ||
    fail 1 "Malformed deferred CPU campaign summary."
  validate_nonnegative "successful case count" "${successful}"
  validate_nonnegative "blocked case count" "${blocked}"
  validate_nonnegative "failed case count" "${failed}"
  validate_nonnegative "retained CPU bytes" "${CPU_BYTES_RETAINED}"
  printf '%s\n' \
    "campaign_run_id=${RUN_ID}" \
    "git_commit=${report_commit}" \
    "cpu_host=${CPU_HOST}" \
    "remote_storage_root=${REMOTE_STORAGE_ROOT}" \
    "successful_count=${successful}" \
    "blocked_count=${blocked}" \
    "failed_count=${failed}" \
    "retained_cpu_bytes=${CPU_BYTES_RETAINED}" \
    'state=cpu_terminal_awaiting_collection' \
    'Later collection:'
  print_command "${HOST_REPO_ROOT}/scripts/generation_workflow.sh" collect "${RUN_ID}" \
    --cpu-host "${CPU_HOST}" --remote-root "${REMOTE_ROOT}" \
    --git-commit "${REQUESTED_COMMIT}"
  printf 'Later dataset build:\n'
  print_command "${HOST_REPO_ROOT}/scripts/generation_workflow.sh" build-datasets "${RUN_ID}" \
    --git-commit "${REQUESTED_COMMIT}"
  if [[ "${SMOKE_PROFILE_MODE}" == true ]]; then
    printf 'PROFILE COMPLETE:\nprofile=%s\ncampaign_run_id=%s\ncpu_terminal_publication=validated\ncollection=deferred\n' \
      "${SMOKE_PROFILE_NAME}" "${RUN_ID}"
  else
    printf 'DEFERRED: CPU campaign validated and awaiting collection\n'
  fi
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
  local -a local_arguments=(storage-status --role gpu --storage-root "${LOCAL_STORAGE_ROOT}")
  local -a remote_arguments=(storage-status --role cpu --query-scheduler --storage-root "${REMOTE_STORAGE_ROOT}")
  if [[ -n "${RUN_ID}" ]]; then
    local_arguments+=(--campaign-run-id "${RUN_ID}")
    remote_arguments+=(--campaign-run-id "${RUN_ID}")
  fi
  if [[ -n "${RUN_ID}" ]]; then
    printf 'Campaign status:\n'
    remote_cli campaign-status "${RUN_ID}" --format summary \
      --storage-root "${REMOTE_STORAGE_ROOT}"
  fi
  printf 'GPU storage status:\n'
  local_cli "${local_arguments[@]}"
  printf 'CPU storage status:\n'
  remote_cli "${remote_arguments[@]}"
  if [[ -n "${RUN_ID}" ]]; then
    local_cli validate-pilot-check "${RUN_ID}" --if-present --format summary \
      --storage-root "${LOCAL_STORAGE_ROOT}"
  fi
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
  local -a command
  if [[ -n "${PILOT_CASES_PER_MATERIAL}" ]]; then
    command=(
      ./scripts/generation_workflow.sh pilot-check "${CAMPAIGN_ARGUMENT}"
      --cases-per-material "${PILOT_CASES_PER_MATERIAL}"
      --cpu-host "${CPU_HOST}" --remote-root "${REMOTE_ROOT}"
      --git-commit "${REQUESTED_COMMIT}"
    )
  else
    command=(
      ./scripts/generation_workflow.sh all "${CAMPAIGN_ARGUMENT}"
      --cpu-host "${CPU_HOST}" --remote-root "${REMOTE_ROOT}"
      --git-commit "${REQUESTED_COMMIT}"
    )
  fi
  [[ -z "${ONLY_BATCH}" ]] || command+=(--only-batch "${ONLY_BATCH}")
  [[ "${SKIP_EXTREME_FAMILY_OOD}" != true ]] || command+=(--skip-extreme-family-ood)
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
  WORKFLOW_FAILURE_EVIDENCE=""
  if [[ -n "${RUN_ID}" ]]; then
    resume="$(resume_command_text)"
    local record kind canonical visible extra
    if [[ -n "${LOCAL_STORAGE_ROOT:-}" ]] &&
      record="$(local_cli record-workflow-failure "${RUN_ID}" \
        --storage-root "${LOCAL_STORAGE_ROOT}" --stage "${ALL_STAGE}" \
        --resume-command "${resume}" --cpu-bytes-retained "${CPU_BYTES_RETAINED}" \
        --format tsv 2>/dev/null)"; then
      IFS=$'\t' read -r kind canonical visible extra <<< "${record}"
      if [[ "${kind}" == workflow-failure && -n "${canonical}" \
        && -n "${visible}" && -z "${extra:-}" ]]; then
        WORKFLOW_FAILURE_EVIDENCE="local canonical=${canonical} container=${visible}"
      fi
    fi
    if [[ -z "${WORKFLOW_FAILURE_EVIDENCE}" ]] &&
      record="$(remote_cli record-workflow-failure "${RUN_ID}" \
        --storage-root "${REMOTE_STORAGE_ROOT}" --stage "${ALL_STAGE}" \
        --resume-command "${resume}" --cpu-bytes-retained "${CPU_BYTES_RETAINED}" \
        --format tsv 2>/dev/null)"; then
      IFS=$'\t' read -r kind canonical visible extra <<< "${record}"
      if [[ "${kind}" == workflow-failure && -n "${canonical}" \
        && -n "${visible}" && -z "${extra:-}" ]]; then
        WORKFLOW_FAILURE_EVIDENCE="CPU host=${CPU_HOST} canonical=${canonical} remote=${visible}"
      fi
    fi
  else
    resume="$(all_command_text)"
  fi
  if [[ -n "${PILOT_CASES_PER_MATERIAL}" ]]; then
    local cleanup_state=cleanup_not_authorized
    if (( CPU_BYTES_RECLAIMED > 0 || PILOT_STAGING_RECLAIMED > 0 )); then
      cleanup_state=cleanup_incomplete
    fi
    generation_console_warning \
      "pilot cleanup state=${cleanup_state}; preserved bytes and staging remain authoritative"
  fi
  generation_console_failure "${ALL_STAGE}" "${RUN_ID}" \
    "cause unavailable from consolidated evidence; inspect the preceding error and preserved logs" \
    "${WORKFLOW_FAILURE_EVIDENCE}" "${CPU_BYTES_RETAINED}" "${resume}"
  return "${status}"
}
workflow_exit_handler() {
  local status="$1"
  if [[ "${ALL_WORKFLOW_ACTIVE}" == true && "${status}" -ne 0 ]]; then
    workflow_failure_report "${status}" || true
  fi
  cleanup_pinned_source || true
}

read_remote_campaign_identity() {
  resolve_local_python
  verify_remote_setup_for_output
  local identity kind purpose cases extra
  identity="$(remote_cli campaign-status "${RUN_ID}" --no-scheduler \
    --storage-root "${REMOTE_STORAGE_ROOT}" |
    local_python -c 'import json, sys
value = json.load(sys.stdin)
print("\t".join((
    "campaign",
    str(value["campaign_purpose"]),
    "-" if value["cases_per_material"] is None else str(value["cases_per_material"]),
    str(value["submission_config"]["poll_interval_seconds"]),
)))')" ||
    fail 1 "Could not reconstruct campaign identity for resume."
  local poll_interval
  IFS=$'\t' read -r kind purpose cases poll_interval extra <<< "${identity}"
  [[ "${kind}" == campaign && -z "${extra:-}" ]] || fail 1 "Malformed campaign identity record."
  validate_positive "configured poll_interval_seconds" "${poll_interval}"
  STATUS_POLL_SECONDS="${poll_interval}"
  if [[ "${purpose}" == pilot_check ]]; then
    validate_positive "pilot cases per material" "${cases}"
    PILOT_CASES_PER_MATERIAL="${cases}"
  else
    PILOT_CASES_PER_MATERIAL=""
  fi
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

continue_pilot_workflow() {
  ALL_WORKFLOW_ACTIVE=true
  trap 'workflow_exit_handler $?' EXIT
  resolve_local_python
  resolve_local_storage
  resolve_remote_layout
  ALL_STAGE="existing pilot receipt validation"
  if [[ "${DEFER_COLLECTION}" != true ]] \
    && local_cli validate-pilot-check "${RUN_ID}" --require-cleanup-complete \
      --storage-root "${LOCAL_STORAGE_ROOT}" >/dev/null 2>&1; then
    local_cli validate-pilot-check "${RUN_ID}" --format summary \
      --storage-root "${LOCAL_STORAGE_ROOT}" >/dev/null
    generation_console_stage 8 12 "Generation" REUSED "campaign_run_id=${RUN_ID}"
    generation_console_stage 9 12 "GPU publication" REUSED "validated destination=${LOCAL_STORAGE_ROOT}"
    generation_console_stage 10 12 "Pilot analysis" REUSED "Dataset and physical checks validated"
    generation_console_stage 11 12 "Cleanup" REUSED "completed cleanup receipts validated"
    generation_console_stage 12 12 "Final validation" REUSED "pilot receipt already complete"
    generation_console_final "campaign_run_id=${RUN_ID} pilot receipt validated"
    ALL_WORKFLOW_ACTIVE=false
    return
  fi

  ALL_STAGE="remote pilot terminal monitoring"
  wait_for_terminal_publication 8 12 "Generation"
  generation_console_stage 8 12 "Generation" OK \
    "campaign_run_id=${RUN_ID} state=successful"
  if [[ "${DEFER_COLLECTION}" == true ]]; then
    ALL_STAGE="deferred CPU pilot summary"
    deferred_campaign_report
    ALL_WORKFLOW_ACTIVE=false
    return
  fi

  ALL_STAGE="pilot source inventory, transfer, and hash validation"
  generation_console_stage 9 12 "GPU publication" RUNNING
  local transfer_status=REUSED
  if ! gpu_publication_is_valid; then
    [[ "${REMOTE_SOURCE_STATE}" != source_cleanup_complete ]] ||
      fail 1 "CPU source is cleaned but no valid GPU pilot publication exists."
    collect_campaign >/dev/null
    transfer_status=OK
  fi
  generation_console_stage 9 12 "GPU publication" "${transfer_status}" \
    "campaign_run_id=${RUN_ID} destination=${LOCAL_STORAGE_ROOT} retained_bytes=${CPU_BYTES_RETAINED}"

  ALL_STAGE="canonical HDF5 and physical/runtime pilot analysis"
  generation_console_stage 10 12 "Pilot analysis" RUNNING
  prepare_pilot_check_receipt >/dev/null
  ALL_STAGE="empty production-dataset gate bound to pilot evidence"
  local dataset_output dataset_id=unavailable
  local dataset_pattern='"dataset_id"[[:space:]]*:[[:space:]]*"([A-Za-z0-9._:-]+)"'
  dataset_output="$(build_datasets)"
  if [[ ${dataset_output} =~ ${dataset_pattern} ]]; then
    dataset_id="${BASH_REMATCH[1]}"
  fi
  generation_console_stage 10 12 "Pilot analysis" OK \
    "campaign_run_id=${RUN_ID} dataset_id=${dataset_id}"

  ALL_STAGE="terminal pre-cleanup workflow receipt"
  generation_console_stage 11 12 "Cleanup" RUNNING
  prepare_all_receipt >/dev/null
  read_remote_source_status
  [[ "${REMOTE_SOURCE_ACTIVE}" == False ]] ||
    fail 1 "cleanup_not_authorized: an active Slurm attempt still owns the CPU pilot source."
  local cleanup_detail
  if [[ "${KEEP_CPU_SOURCE}" == true ]]; then
    CPU_BYTES_RECLAIMED=0
    CPU_CLEANUP_RECEIPT_SHA=""
    cleanup_detail="CPU source retained by request; retained_bytes=${CPU_BYTES_RETAINED}"
  else
    ALL_STAGE="verified CPU pilot source cleanup"
    confirm_cpu_cleanup >/dev/null
    cleanup_detail="CPU source cleanup complete; reclaimed_bytes=${CPU_BYTES_RECLAIMED}"
  fi
  ALL_STAGE="verified pilot transfer-staging cleanup"
  cleanup_pilot_staging >/dev/null
  ALL_STAGE="canonical final pilot cleanup receipt"
  record_pilot_cleanup_result
  generation_console_stage 11 12 "Cleanup" OK "${cleanup_detail}; pilot staging reclaimed"

  ALL_STAGE="terminal pilot and workflow receipt validation"
  generation_console_stage 12 12 "Final validation" RUNNING
  local_cli validate-all-workflow "${RUN_ID}" --storage-root "${LOCAL_STORAGE_ROOT}" >/dev/null
  local_cli validate-pilot-check "${RUN_ID}" --require-cleanup-complete --format summary \
    --storage-root "${LOCAL_STORAGE_ROOT}" >/dev/null
  generation_console_stage 12 12 "Final validation" OK \
    "campaign_run_id=${RUN_ID} pilot and workflow receipts validated"
  generation_console_final "campaign_run_id=${RUN_ID} pilot receipt validated"
  ALL_WORKFLOW_ACTIVE=false
}
run_pilot_check() {
  [[ "${CONFIRM_CLEANUP}" == false && -z "${ONLY_BATCH}" ]] ||
    fail 2 "pilot-check does not support --confirm or --only-batch."
  HUMAN_WORKFLOW_MODE=true
  ALL_WORKFLOW_ACTIVE=true
  trap 'workflow_exit_handler $?' EXIT

  ALL_STAGE="local exact-commit and campaign validation"
  generation_console_stage 1 12 "Local preflight" RUNNING
  resolve_local_commit
  resolve_campaign "${CAMPAIGN_ARGUMENT}"
  resolve_pilot_contract
  resolve_local_storage
  resolve_configured_resources
  validate_resources
  generation_console_stage 1 12 "Local preflight" OK \
    "commit=${REQUESTED_COMMIT:0:12} campaign=${CAMPAIGN_RELATIVE_PATH} cases=${PILOT_TOTAL_CASES}"

  ALL_STAGE="CPU setup and readiness"
  generation_console_stage 2 12 "CPU readiness" RUNNING
  EXECUTE_SETUP=true
  setup_cpu >/dev/null
  preflight_cpu >/dev/null
  generation_console_stage 2 12 "CPU readiness" OK \
    "host=${CPU_HOST} scheduler=${SCHEDULER_KIND} partition=${PARTITION}"

  ALL_STAGE="profile technical-smoke evidence"
  generation_console_stage 3 12 "Runtime-smoke evidence" RUNNING
  local comsol_version pilot_campaign_path
  comsol_version="$(remote_comsol_version)"
  pilot_campaign_path="${CAMPAIGN_CONFIG_PATH}"
  technical_smoke_evidence_status_cpu "${pilot_campaign_path}" "${comsol_version}" >/dev/null ||
    fail 2 "Current successful transient technical-smoke evidence is required before pilot launch; run the technical smoke first."
  generation_console_stage 3 12 "Runtime-smoke evidence" OK "COMSOL=${comsol_version}"

  resolve_campaign "${pilot_campaign_path}"
  resolve_configured_resources
  ALL_STAGE="configured-material static scientific sentinels"
  generation_console_stage 4 12 "Scientific sentinels" RUNNING
  local_cli static-sentinels "${STATIONARY_PRIMARY_CAMPAIGN_HOST_PATH}" \
    "${TRANSIENT_PRIMARY_CAMPAIGN_HOST_PATH}" >/dev/null ||
    fail 2 "Static scientific sentinels block pilot launch; inspect the sentinel report before rerunning."
  generation_console_stage 4 12 "Scientific sentinels" OK "configured-material checks passed"
  print_layout >/dev/null

  ALL_STAGE="canonical pilot input preparation and admission"
  generation_console_stage 5 12 "Canonical inputs" RUNNING
  local input_record input_kind input_generated input_reused input_extra
  input_record="$(remote_prepare_campaign_inputs)" ||
    fail 1 "Canonical pilot input preparation failed before campaign planning or launch."
  IFS=$'\t' read -r input_kind input_generated input_reused input_extra <<< "${input_record}"
  [[ "${input_kind}" == canonical-inputs && -z "${input_extra:-}" ]] ||
    fail 1 "Malformed canonical pilot input readiness result."
  validate_nonnegative "generated canonical input count" "${input_generated}"
  validate_nonnegative "reused canonical input count" "${input_reused}"
  generation_console_stage 5 12 "Canonical inputs" OK \
    "reused=${input_reused} generated=${input_generated}"

  ALL_STAGE="resolved pilot campaign plan"
  generation_console_stage 6 12 "Campaign plan" RUNNING
  remote_plan_submit plan-campaign >/dev/null
  generation_console_stage 6 12 "Campaign plan" OK \
    "materials=${PILOT_MATERIAL_COUNT} cases_per_material=${PILOT_CASES_PER_MATERIAL} total=${PILOT_TOTAL_CASES}"

  ALL_STAGE="pilot campaign launch"
  generation_console_stage 7 12 "Campaign launch" RUNNING
  launch_campaign >/dev/null
  generation_console_stage 7 12 "Campaign launch" OK "campaign_run_id=${RUN_ID}"
  ALLOW_REMOTE_RESUME=false
  continue_pilot_workflow
}
continue_all_workflow() {
  ALL_WORKFLOW_ACTIVE=true
  trap 'workflow_exit_handler $?' EXIT
  resolve_local_python
  resolve_local_storage
  resolve_remote_layout
  ALL_STAGE="existing terminal receipt validation"
  if [[ "${DEFER_COLLECTION}" != true ]] \
    && local_cli validate-all-workflow "${RUN_ID}" \
      --storage-root "${LOCAL_STORAGE_ROOT}" >/dev/null 2>&1; then
    generation_console_stage 6 9 "Generation" REUSED "campaign_run_id=${RUN_ID}"
    generation_console_stage 7 9 "GPU publication" REUSED "validated destination=${LOCAL_STORAGE_ROOT}"
    generation_console_stage 8 9 "Dataset packages" REUSED "validated campaign_run_id=${RUN_ID}"
    generation_console_stage 9 9 "Workflow receipt" REUSED \
      "campaign_run_id=${RUN_ID} already complete and validated"
    campaign_workflow_complete
    ALL_WORKFLOW_ACTIVE=false
    return
  fi

  ALL_STAGE="remote generation completion"
  wait_for_terminal_publication 6 9 "Generation"
  generation_console_stage 6 9 "Generation" OK \
    "campaign_run_id=${RUN_ID} state=successful"
  if [[ "${DEFER_COLLECTION}" == true ]]; then
    ALL_STAGE="deferred CPU terminal summary"
    deferred_campaign_report
    ALL_WORKFLOW_ACTIVE=false
    return
  fi

  ALL_STAGE="GPU collection and atomic publication"
  generation_console_stage 7 9 "GPU publication" RUNNING
  local transfer_status=REUSED
  if ! gpu_publication_is_valid; then
    [[ "${REMOTE_SOURCE_STATE}" != source_cleanup_complete ]] ||
      fail 1 "CPU source is cleaned but no valid GPU publication exists."
    collect_campaign >/dev/null
    transfer_status=OK
  fi
  generation_console_stage 7 9 "GPU publication" "${transfer_status}" \
    "campaign_run_id=${RUN_ID} destination=${LOCAL_STORAGE_ROOT} retained_bytes=${CPU_BYTES_RETAINED}"

  ALL_STAGE="dataset build, inspection, and loader smokes"
  generation_console_stage 8 9 "Dataset packages" RUNNING
  local dataset_output dataset_id=unavailable
  local dataset_pattern='"dataset_id"[[:space:]]*:[[:space:]]*"([A-Za-z0-9._:-]+)"'
  dataset_output="$(build_datasets)"
  if [[ ${dataset_output} =~ ${dataset_pattern} ]]; then
    dataset_id="${BASH_REMATCH[1]}"
  fi
  generation_console_stage 8 9 "Dataset packages" OK \
    "campaign_run_id=${RUN_ID} dataset_id=${dataset_id}"

  ALL_STAGE="terminal pre-cleanup receipt"
  generation_console_stage 9 9 "Workflow receipt" RUNNING
  prepare_all_receipt >/dev/null
  local cleanup_detail
  if [[ "${KEEP_CPU_SOURCE}" == true ]]; then
    cleanup_detail="CPU source retained by request; retained_bytes=${CPU_BYTES_RETAINED}"
  else
    ALL_STAGE="verified CPU source cleanup"
    confirm_cpu_cleanup >/dev/null
    cleanup_detail="CPU source cleanup complete; reclaimed_bytes=${CPU_BYTES_RECLAIMED}"
  fi
  ALL_STAGE="terminal receipt validation"
  local_cli validate-all-workflow "${RUN_ID}" --storage-root "${LOCAL_STORAGE_ROOT}" >/dev/null
  generation_console_stage 9 9 "Workflow receipt" OK \
    "campaign_run_id=${RUN_ID} ${cleanup_detail}"
  campaign_workflow_complete
  ALL_WORKFLOW_ACTIVE=false
}
run_all() {
  HUMAN_WORKFLOW_MODE=true
  ALL_WORKFLOW_ACTIVE=true
  trap 'workflow_exit_handler $?' EXIT
  ALL_STAGE="local repository, campaign, and resource validation"
  generation_console_stage 1 9 "Local preflight" RUNNING
  resolve_local_commit
  resolve_campaign "${CAMPAIGN_ARGUMENT}"
  resolve_configured_resources
  validate_resources
  resolve_remote_layout
  resolve_local_storage
  resolve_local_python
  validate_local_launch_gates
  generation_console_stage 1 9 "Local preflight" OK \
    "commit=${REQUESTED_COMMIT:0:12} campaign=${CAMPAIGN_RELATIVE_PATH}"
  ALL_STAGE="CPU setup and canonical input readiness"
  generation_console_stage 2 9 "CPU login preflight" RUNNING
  verify_remote_setup >/dev/null
  generation_console_stage 2 9 "CPU login preflight" OK \
    "host=${CPU_HOST} scheduler=${SCHEDULER_KIND} partition=${PARTITION}"
  ALL_STAGE="canonical input preparation and admission"
  generation_console_stage 3 9 "Canonical inputs" RUNNING
  local input_record input_kind input_generated input_reused input_extra
  input_record="$(remote_prepare_campaign_inputs)" ||
    fail 1 "Canonical input preparation failed before campaign planning or launch."
  IFS=$'\t' read -r input_kind input_generated input_reused input_extra <<< "${input_record}"
  [[ "${input_kind}" == canonical-inputs && -z "${input_extra:-}" ]] ||
    fail 1 "Malformed canonical input readiness result."
  validate_nonnegative "generated canonical input count" "${input_generated}"
  validate_nonnegative "reused canonical input count" "${input_reused}"
  generation_console_stage 3 9 "Canonical inputs" OK \
    "reused=${input_reused} generated=${input_generated}"
  ALL_STAGE="resolved campaign plan"
  generation_console_stage 4 9 "Campaign plan" RUNNING
  remote_plan_submit plan-campaign >/dev/null
  generation_console_stage 4 9 "Campaign plan" OK \
    "purpose=${CAMPAIGN_PURPOSE} cleanup=$([[ "${DEFER_COLLECTION}" == true ]] && printf deferred || { [[ "${KEEP_CPU_SOURCE}" == true ]] && printf retain || printf verified-delete; })"
  ALL_STAGE="campaign launch"
  generation_console_stage 5 9 "Campaign launch" RUNNING
  launch_campaign >/dev/null
  generation_console_stage 5 9 "Campaign launch" OK "campaign_run_id=${RUN_ID}"
  ALLOW_REMOTE_RESUME=false
  continue_all_workflow
}

resume_all() {
  HUMAN_WORKFLOW_MODE=true
  resolve_local_commit
  resolve_remote_layout
  resolve_local_storage
  verify_remote_setup >/dev/null
  read_remote_campaign_identity
  if [[ -n "${PILOT_CASES_PER_MATERIAL}" ]]; then
    generation_console_stage 1 12 "Local preflight" OK \
      "commit=${REQUESTED_COMMIT:0:12} campaign_run_id=${RUN_ID}"
    generation_console_stage 2 12 "CPU readiness" OK "host=${CPU_HOST}"
    generation_console_stage 3 12 "Runtime-smoke evidence" REUSED "persisted campaign evidence"
    generation_console_stage 4 12 "Scientific sentinels" REUSED "persisted campaign contract"
    generation_console_stage 5 12 "Canonical inputs" REUSED "persisted campaign identity"
    generation_console_stage 6 12 "Campaign plan" REUSED "persisted campaign identity"
    generation_console_stage 7 12 "Campaign launch" REUSED "campaign_run_id=${RUN_ID}"
  else
    generation_console_stage 1 9 "Local preflight" OK \
      "commit=${REQUESTED_COMMIT:0:12} campaign_run_id=${RUN_ID}"
    generation_console_stage 2 9 "CPU login preflight" OK "host=${CPU_HOST}"
    generation_console_stage 3 9 "Canonical inputs" REUSED "persisted campaign identity"
    generation_console_stage 4 9 "Campaign plan" REUSED "persisted campaign identity"
    generation_console_stage 5 9 "Campaign launch" REUSED "campaign_run_id=${RUN_ID}"
  fi
  ALLOW_REMOTE_RESUME=true
  if [[ -n "${PILOT_CASES_PER_MATERIAL}" ]]; then
    continue_pilot_workflow
  else
    continue_all_workflow
  fi
}
resolve_benchmark_contract() {
  admit_repository_file "${BENCHMARK_SUITE_RELATIVE_PATH}" "maintained core benchmark suite"
  BENCHMARK_SUITE_PATH="${ADMITTED_HOST_PATH}"
  local -a inspect_arguments=(inspect-core-benchmark "${BENCHMARK_SUITE_PATH}")
  [[ -z "${BENCHMARK_VARIANT}" ]] || inspect_arguments+=(--variant "${BENCHMARK_VARIANT}")
  local inspection record kind extra configured_cpu_host configured_scheduler
  local configured_partition configured_cores_per_node configured_python_module
  local configured_comsol_module configured_python_executable configured_comsol_executable
  inspection="$(local_cli "${inspect_arguments[@]}")" ||
    fail 2 "Could not resolve the maintained core benchmark suite."
  record="$(printf '%s\n' "${inspection}" | local_python -c 'import json, sys
value = json.load(sys.stdin)
resource = value["resource_contract"]
variants = value["variants"]
fields = (
    value["suite_name"], value["suite_digest"], str(value["repetitions"]),
    ",".join(str(item["cores_per_case"]) for item in variants),
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
    BENCHMARK_REPETITIONS BENCHMARK_CORE_COUNTS configured_cpu_host \
    configured_scheduler configured_partition configured_cores_per_node \
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
    "${REQUESTED_COMMIT}" "${remote_suite}" "${BENCHMARK_VARIANT}" \
    "${operation}" "${REMOTE_ROOT}/benchmark-preflight-scratch" \
    "${PYTHON_MODULE}" "${COMSOL_MODULE}" "${COMSOL_EXECUTABLE}" <<'REMOTE'
set -euo pipefail
repository="$1"; storage="$2"; venv="$3"; commit="$4"; suite="$5"
variant="$6"; operation="$7"; scratch="$8"; python_module="$9"
comsol_module="${10}"; comsol_executable="${11}"
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
mkdir -p -- "${scratch}"
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
if [[ "${operation}" != preflight-core-benchmark && -n "${variant}" ]]; then
  command+=(--variant "${variant}")
fi
"${command[@]}"
REMOTE
}

wait_for_core_benchmark() {
  validate_positive "configured benchmark poll_interval_seconds" "${STATUS_POLL_SECONDS}"
  while true; do
    local state
    state="$(remote_cli core-benchmark-status "${RUN_ID}" --format state \
      --storage-root "${REMOTE_STORAGE_ROOT}")"
    generation_console_progress benchmark 4 7 "Canary and measurements" RUNNING "${state}" \
      "benchmark_run_id=${RUN_ID} state=${state}"
    case "${state}" in
      complete|retry_required|canary_failed)
        BENCHMARK_TERMINAL_STATE="${state}"
        return
        ;;
      incomplete)
        local -a arguments=(resume-core-benchmark "${RUN_ID}" --storage-root "${REMOTE_STORAGE_ROOT}")
        [[ -z "${BENCHMARK_VARIANT}" ]] || arguments+=(--variant "${BENCHMARK_VARIANT}")
        local output
        output="$(remote_cli "${arguments[@]}")" ||
          fail 1 "Could not submit the next serial benchmark job."
        if [[ "${output}" == *'"state": "incomplete"'* || "${output}" == *'"state":"incomplete"'* ]]; then
          BENCHMARK_TERMINAL_STATE=incomplete
          return
        fi
        sleep "${STATUS_POLL_SECONDS}"
        ;;
      running|scheduler_unknown|license_blocked)
        sleep "${STATUS_POLL_SECONDS}"
        ;;
      *)
        fail 1 "Core benchmark entered unsupported state: ${state}"
        ;;
    esac
  done
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
  plan="$(remote_cli core-benchmark-transfer-plan "${RUN_ID}" --format tsv \
    --storage-root "${REMOTE_STORAGE_ROOT}")" ||
    fail 1 "Remote core benchmark is not terminally valid."
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

run_core_benchmark() {
  [[ "${CONFIRM_CLEANUP}" == false \
    && "${KEEP_CPU_SOURCE}" == false && -z "${ONLY_BATCH}" ]] ||
    fail 2 "benchmark-cores accepts only --variant, --defer-collection, and remote options."
  HUMAN_WORKFLOW_MODE=true

  generation_console_stage 1 7 "Local preflight" RUNNING
  resolve_local_commit
  resolve_local_storage
  resolve_local_python
  resolve_benchmark_contract >/dev/null
  resolve_remote_layout
  generation_console_stage 1 7 "Local preflight" OK \
    "commit=${REQUESTED_COMMIT:0:12} suite=${BENCHMARK_SUITE_NAME} smoke_dependency=none"

  generation_console_stage 2 7 "CPU runtime preflight" RUNNING
  EXECUTE_SETUP=true
  setup_cpu >/dev/null
  remote_benchmark_plan_submit preflight-core-benchmark >/dev/null ||
    fail 1 "Standalone benchmark CPU preflight failed before any measured submission."
  generation_console_stage 2 7 "CPU runtime preflight" OK \
    "host=${CPU_HOST} COMSOL=${COMSOL_MODULE} scratch=${REMOTE_ROOT}/benchmark-preflight-scratch"

  generation_console_stage 3 7 "Benchmark plan" RUNNING
  remote_benchmark_plan_submit plan-core-benchmark >/dev/null
  generation_console_stage 3 7 "Benchmark plan" OK \
    "suite=${BENCHMARK_SUITE_NAME} repetitions=${BENCHMARK_REPETITIONS} cores=${BENCHMARK_CORE_COUNTS} variant=${BENCHMARK_VARIANT:-all}"

  generation_console_stage 4 7 "Canary and measurements" RUNNING
  local output
  local benchmark_run_pattern='"benchmark_run_id"[[:space:]]*:[[:space:]]*"(core_scaling_transient__[0-9a-f]{16})"'
  output="$(remote_benchmark_plan_submit submit-core-benchmark)" ||
    fail 1 "Remote core benchmark submission failed."
  if [[ ${output} =~ ${benchmark_run_pattern} ]]; then
    RUN_ID="${BASH_REMATCH[1]}"
    validate_benchmark_run_id "${RUN_ID}"
  else
    fail 1 "Core benchmark submission returned no run ID."
  fi
  wait_for_core_benchmark
  case "${BENCHMARK_TERMINAL_STATE}" in
    complete)
      generation_console_stage 4 7 "Canary and measurements" OK \
        "benchmark_run_id=${RUN_ID} canary=validated state=complete"
      generation_console_stage 5 7 "CPU finalization" RUNNING
      remote_cli finalize-core-benchmark "${RUN_ID}" \
        --storage-root "${REMOTE_STORAGE_ROOT}" >/dev/null
      generation_console_stage 5 7 "CPU finalization" OK \
        "benchmark_run_id=${RUN_ID} terminal evidence validated"
      if [[ "${DEFER_COLLECTION}" == true ]]; then
        generation_console_stage 6 7 "GPU publication" DEFERRED \
          "benchmark_run_id=${RUN_ID} CPU evidence retained exclusively"
        printf '%s\n' \
          "benchmark_run_id=${RUN_ID}" \
          "git_commit=${REQUESTED_COMMIT}" \
          "cpu_host=${CPU_HOST}" \
          "remote_storage_root=${REMOTE_STORAGE_ROOT}" \
          'state=cpu_terminal_awaiting_collection' \
          'Later benchmark collection:'
        print_command "${HOST_REPO_ROOT}/scripts/generation_workflow.sh" collect-benchmark \
          "${RUN_ID}" --cpu-host "${CPU_HOST}" --remote-root "${REMOTE_ROOT}" \
          --git-commit "${REQUESTED_COMMIT}"
        generation_console_stage 7 7 "Final validation" DEFERRED \
          "benchmark_run_id=${RUN_ID} CPU evidence validated"
        printf 'DEFERRED: CPU benchmark validated and awaiting collection\n'
        return
      fi
      generation_console_stage 6 7 "GPU publication" RUNNING
      local benchmark_output publication_status=OK
      benchmark_output="$(collect_core_benchmark)"
      if [[ "${benchmark_output}" == *"validated and reused"* ]]; then
        publication_status=REUSED
      fi
      printf '%s\n' "${benchmark_output}"
      generation_console_stage 6 7 "GPU publication" "${publication_status}" \
        "benchmark_run_id=${RUN_ID} destination=${LOCAL_STORAGE_ROOT}"
      generation_console_stage 7 7 "Final validation" OK \
        "benchmark_run_id=${RUN_ID} evidence published; CPU source retained"
      generation_console_final "benchmark_run_id=${RUN_ID} benchmark evidence validated"
      ;;
    canary_failed)
      generation_console_stage 4 7 "Canary and measurements" FAILED \
        "benchmark_run_id=${RUN_ID} production-core repetition-1 canary failed; later repetitions were not submitted"
      fail 1 "Benchmark canary failed validation. Inspect the preserved repetition and scheduler evidence before an explicit retry."
      ;;
    retry_required)
      local retry_output variant_id=unavailable
      local variant_pattern='"variant_id"[[:space:]]*:[[:space:]]*"([A-Za-z0-9._:-]+)"'
      retry_output="$(remote_cli core-benchmark-status "${RUN_ID}" \
        --storage-root "${REMOTE_STORAGE_ROOT}")"
      if [[ ${retry_output} =~ ${variant_pattern} ]]; then
        variant_id="${BASH_REMATCH[1]}"
      fi
      generation_console_stage 4 7 "Canary and measurements" FAILED \
        "benchmark_run_id=${RUN_ID} variant_id=${variant_id}; CPU evidence retained"
      fail 1 "One benchmark repetition requires retry. Use its reported variant_id with: ./scripts/generation_workflow.sh benchmark-cores --variant VARIANT_ID"
      ;;
    incomplete)
      generation_console_stage 4 7 "Canary and measurements" FAILED \
        "benchmark_run_id=${RUN_ID} selected subset complete; suite incomplete"
      fail 1 "The selected benchmark subset finished but the four-variant suite is incomplete. Run benchmark-cores or retry another --variant."
      ;;
  esac
}

(( $# > 0 )) || { usage; exit 2; }
[[ "$1" != -h && "$1" != --help ]] || { usage; exit 0; }
for bootstrap_argument in "${ORIGINAL_ARGUMENTS[@]}"; do
  if [[ "${bootstrap_argument}" == --detach ]]; then
    fail 2 $'--detach is no longer supported because its former launch-only meaning was ambiguous.\nUse:
  generation_workflow.sh launch CAMPAIGN\nto submit and return, or:
  generation_workflow.sh all CAMPAIGN --background\nto keep the complete workflow running in tmux.'
  fi
done
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
SKIP_EXTREME_FAMILY_OOD=false
ONLY_BATCH=""
WALL_TIME=""
CORES_PER_CASE=""
PENDING_BUFFER=""
MAX_RUNNING_CASES="-"
PILOT_CASES_PER_MATERIAL=""
PILOT_MATERIAL_COUNT=""
PILOT_TOTAL_CASES=""
BENCHMARK_VARIANT=""
POSITIONAL=()

while (( $# > 0 )); do
  case "$1" in
    --cpu-host) (( $# >= 2 )) || fail 2 "--cpu-host requires a value."; CPU_HOST="$2"; shift 2 ;;
    --remote-root) (( $# >= 2 )) || fail 2 "--remote-root requires a value."; REMOTE_ROOT="$2"; shift 2 ;;
    --git-commit) (( $# >= 2 )) || fail 2 "--git-commit requires a value."; REQUESTED_COMMIT="$2"; shift 2 ;;
    --execute) EXECUTE_SETUP=true; shift ;;
    --confirm) CONFIRM_CLEANUP=true; shift ;;
    --force) FORCE_CANCEL=true; shift ;;
    --keep-cpu-source) KEEP_CPU_SOURCE=true; shift ;;
    --defer-collection) DEFER_COLLECTION=true; shift ;;
    --skip-extreme-family-ood) SKIP_EXTREME_FAMILY_OOD=true; shift ;;
    --only-batch) (( $# >= 2 )) || fail 2 "--only-batch requires a value."; ONLY_BATCH="$2"; shift 2 ;;
    --cases-per-material) (( $# >= 2 )) || fail 2 "--cases-per-material requires a value."; PILOT_CASES_PER_MATERIAL="$2"; shift 2 ;;
    --variant) (( $# >= 2 )) || fail 2 "--variant requires a value."; BENCHMARK_VARIANT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --*) fail 2 "Unsupported option: $1" ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done

[[ -z "${REQUESTED_COMMIT}" ]] || validate_commit "${REQUESTED_COMMIT}"
if [[ "${SUBCOMMAND}" != pilot-check && -n "${PILOT_CASES_PER_MATERIAL}" ]]; then
  fail 2 "--cases-per-material is supported only by pilot-check."
fi
if [[ "${SUBCOMMAND}" != benchmark-cores && -n "${BENCHMARK_VARIANT}" ]]; then
  fail 2 "--variant is supported only by benchmark-cores."
fi
if [[ "${SUBCOMMAND}" != cancel && "${FORCE_CANCEL}" == true ]]; then
  fail 2 "--force is supported only by cancel."
fi
if [[ "${DEFER_COLLECTION}" == true && "${KEEP_CPU_SOURCE}" == true ]]; then
  fail 2 "--defer-collection cannot be combined with --keep-cpu-source."
fi
if [[ "${DEFER_COLLECTION}" == true \
  && ! "${SUBCOMMAND}" =~ ^(all|resume|smoke|benchmark-cores|pilot-check)$ ]]; then
  fail 2 "--defer-collection is supported only by all, resume, smoke, benchmark-cores, and pilot-check."
fi
if [[ "${SKIP_EXTREME_FAMILY_OOD}" == true ]]; then
  [[ "${SUBCOMMAND}" =~ ^(plan|launch|all)$ ]] ||
    fail 2 "--skip-extreme-family-ood is supported only by plan, launch, and all."
  [[ -z "${ONLY_BATCH}" ]] ||
    fail 2 "--skip-extreme-family-ood cannot be combined with --only-batch."
fi

resolve_host_layout
trap 'workflow_exit_handler $?' EXIT
resolve_local_commit
handoff_to_pinned_workflow

case "${SUBCOMMAND}" in
  setup-cpu)
    (( ${#POSITIONAL[@]} == 0 )) || fail 2 "setup-cpu accepts no positional arguments."
    [[ "${KEEP_CPU_SOURCE}" == false && "${CONFIRM_CLEANUP}" == false ]] ||
      fail 2 "Unsupported setup-cpu option."
    setup_cpu
    ;;
  pilot-check)
    (( ${#POSITIONAL[@]} == 1 )) || fail 2 "pilot-check requires the dedicated transient pilot-check campaign config."
    CAMPAIGN_ARGUMENT="${POSITIONAL[0]}"
    run_pilot_check
    ;;
  benchmark-cores)
    (( ${#POSITIONAL[@]} == 0 )) || fail 2 "benchmark-cores accepts no positional arguments."
    run_core_benchmark
    ;;
  preflight|plan|launch|all)
    (( ${#POSITIONAL[@]} == 1 )) || fail 2 "${SUBCOMMAND} requires one campaign config."
    CAMPAIGN_ARGUMENT="${POSITIONAL[0]}"
    case "${SUBCOMMAND}" in
      preflight)
        [[ "${KEEP_CPU_SOURCE}" == false ]] || fail 2 "Unsupported preflight option."
        preflight_cpu
        ;;
      plan)
        [[ "${KEEP_CPU_SOURCE}" == false ]] || fail 2 "Unsupported plan option."
        plan_campaign
        ;;
      launch)
        [[ "${KEEP_CPU_SOURCE}" == false ]] || fail 2 "launch already submits and returns."
        launch_requested_campaign
        ;;
      all) run_all ;;
    esac
    ;;
  smoke)
    (( ${#POSITIONAL[@]} == 0 )) || fail 2 "smoke accepts no campaign positional argument."
    run_smoke
    ;;
  finalize-smoke)
    (( ${#POSITIONAL[@]} == 2 )) ||
      fail 2 "finalize-smoke requires explicit steady and transient campaign run IDs."
    [[ "${CONFIRM_CLEANUP}" == false && "${KEEP_CPU_SOURCE}" == false \
      && -z "${ONLY_BATCH}" ]] || fail 2 "Unsupported finalize-smoke option."
    [[ "${POSITIONAL[0]}" != "${POSITIONAL[1]}" ]] ||
      fail 2 "finalize-smoke requires two distinct explicit campaign run IDs."
    finalize_smoke_runs "${POSITIONAL[0]}" "${POSITIONAL[1]}"
    ;;
  collect-benchmark)
    (( ${#POSITIONAL[@]} == 1 )) ||
      fail 2 "collect-benchmark requires one benchmark run ID."
    [[ "${CONFIRM_CLEANUP}" == false && "${KEEP_CPU_SOURCE}" == false \
      && -z "${ONLY_BATCH}" ]] || fail 2 "collect-benchmark is always non-destructive."
    RUN_ID="${POSITIONAL[0]}"
    validate_benchmark_run_id "${RUN_ID}"
    resolve_local_commit
    resolve_local_storage
    resolve_local_python
    resolve_benchmark_contract >/dev/null
    resolve_remote_layout
    collect_core_benchmark
    generation_console_final "benchmark_run_id=${RUN_ID} benchmark evidence validated"
    ;;
  status)
    (( ${#POSITIONAL[@]} <= 1 )) || fail 2 "status accepts at most one campaign-run ID."
    RUN_ID="${POSITIONAL[0]:-}"
    [[ -z "${RUN_ID}" ]] || validate_run_id "${RUN_ID}"
    resolve_local_commit
    storage_status_report
    ;;
  retry-case)
    (( ${#POSITIONAL[@]} == 3 )) ||
      fail 2 "retry-case requires CAMPAIGN_RUN_ID BATCH_NAME CASE_ID."
    RUN_ID="${POSITIONAL[0]}"
    RETRY_BATCH_NAME="${POSITIONAL[1]}"
    RETRY_CASE_ID="${POSITIONAL[2]}"
    validate_run_id "${RUN_ID}"
    validate_batch_name "${RETRY_BATCH_NAME}"
    validate_case_id "${RETRY_CASE_ID}"
    resolve_local_commit
    resolve_remote_layout
    remote_cli retry-case "${RUN_ID}" "${RETRY_BATCH_NAME}" \
      "${RETRY_CASE_ID}" --storage-root "${REMOTE_STORAGE_ROOT}"
    ;;
  collect|build-datasets|resume|cleanup|accounting|cancel|validate)
    (( ${#POSITIONAL[@]} == 1 )) || fail 2 "${SUBCOMMAND} requires one campaign-run ID."
    RUN_ID="${POSITIONAL[0]}"
    validate_run_id "${RUN_ID}"
    resolve_local_commit
    case "${SUBCOMMAND}" in
      collect)
        [[ "${CONFIRM_CLEANUP}" == false && "${KEEP_CPU_SOURCE}" == false ]] ||
          fail 2 "collect is always non-destructive."
        collect_campaign
        ;;
      build-datasets)
        [[ "${CONFIRM_CLEANUP}" == false && "${KEEP_CPU_SOURCE}" == false ]] ||
          fail 2 "Unsupported build-datasets option."
        build_datasets
        ;;
      resume)
        [[ "${CONFIRM_CLEANUP}" == false ]] || fail 2 "Unsupported resume option."
        resume_all
        ;;
      cleanup)
        [[ "${KEEP_CPU_SOURCE}" == false ]] || fail 2 "Unsupported cleanup option."
        cleanup_cpu_source
        ;;
      accounting)
        resolve_remote_layout
        remote_cli campaign-accounting "${RUN_ID}" --storage-root "${REMOTE_STORAGE_ROOT}"
        ;;
      cancel)
        [[ "${CONFIRM_CLEANUP}" == false && "${KEEP_CPU_SOURCE}" == false ]] ||
          fail 2 "Unsupported cancel option."
        resolve_remote_layout
        cancel_arguments=(
          cancel-campaign "${RUN_ID}" --storage-root "${REMOTE_STORAGE_ROOT}"
        )
        [[ "${FORCE_CANCEL}" != true ]] || cancel_arguments+=(--force)
        remote_cli "${cancel_arguments[@]}"
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
