#!/usr/bin/env bash
set -Eeuo pipefail

ORIGINAL_ARGUMENTS=("$@")
SCRIPT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
HOST_REPO_ROOT=""
HOST_STORAGE_ROOT=""
SOURCE_COMMIT=""
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
HUMAN_WORKFLOW_MODE=false
CONSOLE_PROGRESS_KEY=""
CONSOLE_PROGRESS_SIGNATURE=""
CONSOLE_PROGRESS_DETAIL_SIGNATURE=""
CONSOLE_PROGRESS_RENDERED_AT=0
SHARED_CAMPAIGN_STATE=""
SHARED_CAMPAIGN_STATE_SIGNATURE=""
SHARED_CAMPAIGN_PROGRESS_SIGNATURE=""
SHARED_CAMPAIGN_SUMMARY=""
TRANSFER_SUMMARY=""
DATASET_SUMMARY=""
WORKFLOW_FAILURE_EVIDENCE=""
CAMPAIGN_INTERRUPT_ACTIVE=false
CAMPAIGN_INTERRUPT_COUNT=0
CONSOLE_CHANGED_PROGRESS_SECONDS="${GENERATION_CONSOLE_CHANGED_PROGRESS_SECONDS:-60}"
CONSOLE_HEARTBEAT_SECONDS="${GENERATION_CONSOLE_HEARTBEAT_SECONDS:-120}"
COMPOSITE_CHILD_MODE=false
CAMPAIGN_PARTIAL=false
PAIRED_SMOKE_RECEIPT=""
LOCAL_STORAGE_ROOT=""
LOCAL_PYTHON_READY=false
CONFIGURED_UNIFORM_CASE_COUNT=""
CONFIGURED_TOTAL_CASE_COUNT=""
REPLACEMENT_POOL_SIZE=""
PARENT_RUN_ID=""
COMPLETION_PARENT_STATUS=""
COMPLETION_PARENT_RUN_ID=""
COMPLETION_ID=""
COMPLETION_OWNER_PERSISTED=false
COMPLETION_PARENT_PARTIAL_PATH=""
COMPLETION_PARENT_PARTIAL_SHA256=""
COMPLETION_PARENT_JSON=""
COMPLETION_TRANSFER_JSON=""
COMPLETION_INITIAL_STATUS_JSON=""
COMPLETION_REPLACEMENT_RUN_IDS=()
COMPLETION_REPLACEMENT_RUN_PARTIAL=()
COMPLETION_REPLACEMENT_TERMINAL_BATCH_IDS=()

usage() {
  cat >&2 <<EOF
Usage:
  $0 run CONFIG [--replacement-pool-size N [--parent-run-id RUN_ID]] [--background]
  $0 run CONFIG --dry-run [--replacement-pool-size N [--parent-run-id RUN_ID]]
  $0 run CONFIG --preflight-only [--replacement-pool-size N [--parent-run-id RUN_ID]]
  $0 inputs CAMPAIGN_CONFIG [generate-input-cases selection options]
  $0 status CONFIG_OR_RUN_ID
  $0 cancel RUN_ID [--force]
  $0 smoke [--partition gpu|standard]
  $0 background-status WORKFLOW_SESSION_ID
  $0 background-list

Source option:
  --git-commit COMMIT   require current shared HEAD to equal this commit

Every maintained Generation workflow starts and resumes with run CONFIG.
--replacement-pool-size N enables cumulative deterministic completion of a partial
campaign; increasing N extends the stable candidate prefix. --parent-run-id is an
expert disambiguation override and is valid only with the replacement pool.
Foreground execution is the default. --background changes only controller ownership.
Durable inputs, case results, receipts, and Dataset packages remain in sibling storage.
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
  cancellation+=(--storage-root "${SHARED_STORAGE_ROOT}")
  if (( CAMPAIGN_INTERRUPT_COUNT == 1 )); then
    printf '%s
'       "Graceful ${run_label} cancellation requested."       'Press Ctrl+C again to force cancellation.' >&2
    shared_cli "${cancellation[@]}" >/dev/null &
    local cancellation_pid=$!
    if ! wait "${cancellation_pid}"; then
      generation_console_warning         "graceful cancellation request failed; ${run_label} state remains authoritative"
    fi
    return 0
  fi
  printf 'Force %s cancellation requested.
' "${run_label}" >&2
  cancellation+=(--force)
  if ! shared_cli "${cancellation[@]}" >/dev/null; then
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
    fail 1 "ICE login prerequisite missing: ${command_name} (blocks ${blocked_operation})."
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
  HOST_REPO_ROOT="$(realpath -e -- "${SCRIPT_DIRECTORY}/..")" ||
    fail 1 "Could not resolve the shared repository."
  [[ -d "${HOST_REPO_ROOT}/.git" && ! -L "${HOST_REPO_ROOT}" ]] ||
    fail 1 "Generation requires the current shared Git repository."
  HOST_STORAGE_ROOT="$(realpath -m -- "${STORAGE_ROOT:-${HOST_REPO_ROOT}/../storage}")"
  RUNTIME_ROOT="$(realpath -m -- "${RUNTIME_ROOT:-${HOST_REPO_ROOT}/../runtime}")"
  [[ "${HOST_STORAGE_ROOT}" == "$(realpath -m -- "${HOST_REPO_ROOT}/../storage")" \
    && "${RUNTIME_ROOT}" == "$(realpath -m -- "${HOST_REPO_ROOT}/../runtime")" ]] ||
    fail 2 "Generation requires the sibling storage and runtime roots."
  [[ -d "${HOST_STORAGE_ROOT}" && ! -L "${HOST_STORAGE_ROOT}" ]] ||
    fail 1 "Shared durable storage is missing or unsafe."
  [[ -d "${RUNTIME_ROOT}" && ! -L "${RUNTIME_ROOT}" ]] ||
    fail 1 "Replaceable runtime root is missing or unsafe."
  GENERATION_NATIVE_VENV="${GENERATION_NATIVE_VENV:-${RUNTIME_ROOT}/venvs/generation}"
  [[ "${GENERATION_NATIVE_VENV}" == "${RUNTIME_ROOT}/venvs/generation" \
    && -x "${GENERATION_NATIVE_VENV}/bin/python" ]] ||
    fail 1 "Native Python 3.12 environment is missing: ${GENERATION_NATIVE_VENV}."
  export GENERATION_NATIVE_VENV
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
    fail 2 "${label} does not exist in the current shared repository."
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

validate_commit() {
  [[ "$1" =~ ^[0-9a-f]{40}$ ]] || fail 2 "Git commit must be one lowercase 40-character identifier."
}

validate_run_id() {
  [[ "$1" =~ ^[A-Za-z0-9._-]+__[0-9a-f]{16}$ ]] || fail 2 "Malformed campaign-run ID: $1"
}

validate_completion_id() {
  [[ "$1" =~ ^completion__[0-9a-f]{24}$ ]] ||
    fail 2 "Malformed campaign-completion ID: $1"
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

resolve_shared_layout() {
  resolve_local_storage
  SHARED_STORAGE_ROOT="${LOCAL_STORAGE_ROOT}"
  CPU_HOST="shared-filesystem"
}

admit_shared_source() {
  local head requested status requires_clean=false
  head="$(git -C "${HOST_REPO_ROOT}" rev-parse HEAD)" ||
    fail 1 "Could not resolve the shared source revision."
  validate_commit "${head}"
  requested="${REQUESTED_COMMIT:-${head}}"
  [[ "${requested}" == "${head}" ]] ||
    fail 2 "Generation runs only the current shared repository HEAD; requested commit differs."
  REQUESTED_COMMIT="${head}"
  SOURCE_COMMIT="${head}"
  GENERATION_SOURCE_SHA256="$(python3 "${HOST_REPO_ROOT}/scripts/source_fingerprint.py" "${HOST_REPO_ROOT}")" ||
    fail 1 "Could not fingerprint the current shared source."
  validate_digest "${GENERATION_SOURCE_SHA256}"
  export GENERATION_SOURCE_SHA256
  status="$(git --no-optional-locks -C "${HOST_REPO_ROOT}" status --porcelain=v1 --untracked-files=all)" ||
    fail 1 "Could not inspect the shared source worktree."
  if [[ "${ORIGINAL_ARGUMENTS[0]}" == inputs && "${INPUT_DRY_RUN:-false}" != true ]]; then
    requires_clean=true
  elif [[ "${SUBCOMMAND:-}" == run && "${DRY_RUN:-false}" != true \
    && "${PREFLIGHT_ONLY:-false}" != true ]]; then
    requires_clean=true
  fi
  if [[ "${requires_clean}" == true && -n "${status}" ]]; then
    fail 2 "Generation publication requires a clean committed shared source; current worktree has changes."
  fi
  printf 'Source: shared HEAD %s fingerprint %s\n' "${head}" "${GENERATION_SOURCE_SHA256}" >&2
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
  resolve_bootstrap_requested_commit
  resolve_host_layout
  admit_shared_source
  if [[ "${require_clean}" == true ]]; then
    local status
    status="$(git --no-optional-locks -C "${HOST_REPO_ROOT}" status --porcelain=v1 --untracked-files=all)" ||
      fail 1 "Could not inspect the shared source."
    [[ -z "${status}" ]] ||
      fail 1 "Background Generation requires a clean committed shared source."
  fi
  resolve_local_storage
  resolve_local_python
}

background_host_paths_json() {
  local host_name
  host_name="$(hostname -f 2>/dev/null || hostname)"
  printf '%s\n%s\n%s\n%s\n' \
    "${HOST_REPO_ROOT}/scripts/generation_workflow.sh" \
    "${GENERATION_NATIVE_VENV}/bin/python" \
    "${LOCAL_STORAGE_ROOT}" "${host_name}" |
    local_python -c 'import json, sys
values = [line.rstrip("\n") for line in sys.stdin]
if len(values) != 4 or any(not value for value in values):
    raise SystemExit("background host paths are incomplete")
print(json.dumps(dict(zip(("stable_script", "python_executable", "storage_root", "host"), values, strict=True)), separators=(",", ":"), sort_keys=True))'
}

launch_background_workflow() {
  [[ "${GENERATION_WORKFLOW_BACKGROUND_CHILD:-}" != 1 ]] ||
    fail 2 "A background workflow child cannot create another tmux session."
  require_command tmux "background workflow execution"
  local subcommand="${ORIGINAL_ARGUMENTS[0]}" background_count=0 argument
  [[ "${subcommand}" == run ]] ||
    fail 2 "--background is supported only by run CONFIG."
  local -a child_arguments=()
  local has_commit=false
  for argument in "${ORIGINAL_ARGUMENTS[@]}"; do
    if [[ "${argument}" == --background ]]; then
      background_count=$((background_count + 1))
      continue
    fi
    child_arguments+=("${argument}")
    [[ "${argument}" != --git-commit ]] || has_commit=true
  done
  (( background_count == 1 )) || fail 2 "Specify --background exactly once."
  resolve_background_host_runtime true
  if [[ "${has_commit}" != true ]]; then
    child_arguments+=(--git-commit "${REQUESTED_COMMIT}")
  fi
  ensure_execution_bootstrap
  resolve_shared_layout
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
      "${HOST_REPO_ROOT}/scripts/generation_workflow.sh" "${session_id}"
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
    "${HOST_REPO_ROOT}/scripts/generation_workflow.sh" "${session_id}" \
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
    fail 2 "The maintained Generation workflow requires configured scheduler=slurm."
  validate_positive "configured cores_per_node" "${CORES_PER_NODE}"
}


resolve_campaign() {
  admit_repository_file "$1" "campaign config"
  CAMPAIGN_CONFIG_PATH="${ADMITTED_HOST_PATH}"
  CAMPAIGN_RELATIVE_PATH="${ADMITTED_REPOSITORY_PATH}"
}

resolve_completion_parent_local() {
  resolve_local_storage
  resolve_local_python
  local requested_parent="${1:-${PARENT_RUN_ID}}"
  local -a arguments=(
    find-completion-parent "${CAMPAIGN_CONFIG_PATH}"
    --storage-root "${LOCAL_STORAGE_ROOT}"
  )
  [[ -z "${requested_parent}" ]] || arguments+=(--parent-run-id "${requested_parent}")
  COMPLETION_PARENT_JSON="$(local_cli "${arguments[@]}")" ||
    fail 1 "Could not resolve one structurally compatible completion parent."
  local record partial_path partial_sha completion_status extra
  record="$(printf '%s' "${COMPLETION_PARENT_JSON}" | local_python -c 'import json, sys
value = json.load(sys.stdin)
def clean(item):
    if item is None:
        return "-"
    text = str(item)
    if any(character in text for character in "\t\r\n"):
        raise SystemExit("completion parent result contains unsafe shell transport text")
    return text
print("\t".join(clean(value.get(key)) for key in (
    "status", "parent_run_id", "completion_id", "parent_partial_path",
    "parent_partial_sha256", "completion_status", "replacement_pool_size",
)))')" || fail 1 "Could not decode completion-parent resolution."
  IFS=$'\t' read -r COMPLETION_PARENT_STATUS COMPLETION_PARENT_RUN_ID \
    COMPLETION_ID partial_path partial_sha completion_status persisted_pool extra <<< "${record}"
  [[ -z "${extra:-}" ]] || fail 1 "Malformed completion-parent result."
  COMPLETION_OWNER_PERSISTED=false
  case "${COMPLETION_PARENT_STATUS}" in
    fresh)
      [[ "${COMPLETION_PARENT_RUN_ID}" == - && "${COMPLETION_ID}" == - \
        && "${partial_path}" == - && "${partial_sha}" == - ]] ||
        fail 1 "Fresh completion-parent result contains contradictory identity."
      COMPLETION_PARENT_RUN_ID=""
      COMPLETION_ID=""
      COMPLETION_PARENT_PARTIAL_PATH=""
      COMPLETION_PARENT_PARTIAL_SHA256=""
      ;;
    compatible_active|compatible_complete)
      validate_run_id "${COMPLETION_PARENT_RUN_ID}"
      [[ "${partial_path}" == - && "${partial_sha}" == - ]] ||
        fail 1 "Non-partial completion parent returned partial evidence."
      [[ "${COMPLETION_ID}" == - ]] || validate_completion_id "${COMPLETION_ID}"
      [[ "${COMPLETION_ID}" != - ]] || COMPLETION_ID=""
      COMPLETION_PARENT_PARTIAL_PATH=""
      COMPLETION_PARENT_PARTIAL_SHA256=""
      ;;
    compatible_partial)
      validate_run_id "${COMPLETION_PARENT_RUN_ID}"
      validate_completion_id "${COMPLETION_ID}"
      validate_digest "${partial_sha}"
      COMPLETION_PARENT_PARTIAL_PATH="$(admit_shared_cli_path "${partial_path}")"
      COMPLETION_PARENT_PARTIAL_PATH="$(realpath -e -- "${COMPLETION_PARENT_PARTIAL_PATH}")" ||
        fail 1 "Could not resolve compatible parent partial evidence on the host."
      [[ "${COMPLETION_PARENT_PARTIAL_PATH}" == "${LOCAL_STORAGE_ROOT}/"* \
        && -f "${COMPLETION_PARENT_PARTIAL_PATH}" \
        && ! -L "${COMPLETION_PARENT_PARTIAL_PATH}" ]] ||
        fail 1 "Compatible parent partial evidence escaped canonical local storage."
      COMPLETION_PARENT_PARTIAL_SHA256="${partial_sha}"
      if [[ "${completion_status}" != - ]]; then
        COMPLETION_OWNER_PERSISTED=true
        [[ "${persisted_pool}" != - ]] ||
          fail 1 "Persisted completion status lacks its replacement pool high-water."
        validate_positive "persisted replacement pool high-water" "${persisted_pool}"
        [[ -n "${REPLACEMENT_POOL_SIZE}" ]] || REPLACEMENT_POOL_SIZE="${persisted_pool}"
      fi
      ;;
    *) fail 1 "Unsupported completion-parent status: ${COMPLETION_PARENT_STATUS}" ;;
  esac
}

attach_completion_plan_metadata() {
  RUN_PLAN_JSON="$(local_python -c 'import json, sys
plan = json.loads(sys.argv[1])
resolution = json.loads(sys.argv[2])
pool = int(sys.argv[3])
override = None if sys.argv[4] == "-" else sys.argv[4]
plan["replacement_completion"] = {
    "enabled": True,
    "replacement_pool_size": pool,
    "requested_high_water_mark": pool,
    "parent_run_id_override": override,
    "compatible_parent_candidates": resolution["compatible_parent_candidates"],
    "selected_parent": resolution["selected_parent"],
    "target_counts": resolution["target_counts"],
    "current_successes": resolution["current_successes"],
    "deficits": resolution["success_deficits"],
    "expected_completion_id": resolution["expected_completion_id"],
    "package_declarations": resolution["package_declarations"],
    "pt_shard_requirements": resolution["pt_shard_requirements"],
    "parent_resolution": resolution,
}
print(json.dumps(plan, sort_keys=True))' \
    "${RUN_PLAN_JSON}" "${COMPLETION_PARENT_JSON}" "${REPLACEMENT_POOL_SIZE}" \
    "${PARENT_RUN_ID:--}")" || fail 1 "Could not bind replacement metadata to the dry-run plan."
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
  printf 'Repository: %s\nPersistent storage: %s\nReplaceable runtime: %s\nNative Python: %s\n'     "${HOST_REPO_ROOT}" "${SHARED_STORAGE_ROOT}" "${RUNTIME_ROOT}" "${GENERATION_NATIVE_VENV}"
  printf 'Source commit: %s\nSource fingerprint: %s\nModules: %s, %s\n'     "${REQUESTED_COMMIT}" "${GENERATION_SOURCE_SHA256}" "${PYTHON_MODULE}" "${COMSOL_MODULE}"
}

verify_shared_setup() {
  resolve_shared_layout
  resolve_local_python
  [[ -x "${HOST_REPO_ROOT}/scripts/generation_campaign_node.sh"     && -x "${HOST_REPO_ROOT}/scripts/generation_benchmark_node.sh" ]] ||
    fail 1 "Native Generation Slurm workers are missing."
  [[ "${SCHEDULER_KIND}" == slurm ]] ||
    fail 2 "Generation requires native Slurm."
  [[ "${PARTITION}" == standard || "${PARTITION}" == long ]] ||
    fail 2 "Generation requires the standard or long CPU partition."
  mkdir -p -- "${RUNTIME_ROOT}/logs/generation"
  printf 'Shared Generation setup verified: %s\n' "${HOST_REPO_ROOT}"
}

verify_shared_setup_for_output() {
  if [[ "${HUMAN_WORKFLOW_MODE}" == true ]]; then
    verify_shared_setup >/dev/null
  else
    verify_shared_setup >&2
  fi
}


shared_plan_submit() {
  local operation="$1"
  verify_shared_setup_for_output || return $?
  local -a arguments=(
    "${operation}" "${CAMPAIGN_CONFIG_PATH}"
    --git-commit "${REQUESTED_COMMIT}" --storage-root "${SHARED_STORAGE_ROOT}"
  )
  [[ "${operation}" != submit-campaign ]] || arguments+=(--inputs-prepared)
  local_cli "${arguments[@]}"
}

prepare_shared_campaign_inputs() {
  shared_plan_submit prepare-campaign-inputs
}


technical_smoke_evidence_status_cpu() {
  local campaign_argument="$1" comsol_version_output="$2"
  resolve_campaign "${campaign_argument}"
  local_cli technical-smoke-evidence-status "${CAMPAIGN_CONFIG_PATH}"     --storage-root "${LOCAL_STORAGE_ROOT}"     --comsol-version-output "${comsol_version_output}"
}

native_comsol_version() (
  module load "${COMSOL_MODULE}" ||
    fail 1 "COMSOL module ${COMSOL_MODULE} is unavailable."
  command -v "${COMSOL_EXECUTABLE}" >/dev/null ||
    fail 1 "Native COMSOL executable is unavailable."
  # shellcheck source=generation_prerequisites.sh
  source "${HOST_REPO_ROOT}/scripts/generation_prerequisites.sh"
  generation_comsol_version "${COMSOL_EXECUTABLE}"
)

sync_technical_smoke_evidence() {
  local evidence="$1" campaign_argument="$2" comsol_version_output="$3"
  [[ "${evidence}" == "${LOCAL_STORAGE_ROOT}/"* && -f "${evidence}" && ! -L "${evidence}" ]] ||
    fail 1 "Technical-smoke evidence is outside shared durable storage."
  technical_smoke_evidence_status_cpu     "${campaign_argument}" "${comsol_version_output}" >/dev/null ||
    fail_preserving_interrupt "$?" 2 "Shared technical-smoke evidence is missing or invalid."
}

finalize_smoke_runs() {
  local stationary_run_id="$1" transient_run_id="$2"
  validate_run_id "${stationary_run_id}"
  validate_run_id "${transient_run_id}"
  admit_shared_source
  resolve_workflow_campaigns
  resolve_local_storage
  resolve_local_python
  resolve_shared_layout
  verify_shared_setup_for_output >/dev/null
  local comsol_version steady_evidence transient_evidence smoke_children
  comsol_version="$(native_comsol_version)"
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
  steady_evidence="$(admit_shared_cli_path "${steady_evidence}")"
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
  transient_evidence="$(admit_shared_cli_path "${transient_evidence}")"
  sync_technical_smoke_evidence \
    "${transient_evidence}" "${TRANSIENT_SMOKE_CAMPAIGN_PATH}" "${comsol_version}"
  PAIRED_SMOKE_RECEIPT="$(generation_run_with_heartbeat \
    "paired-smoke-${RUN_PLAN_ID}" 8 9 "Paired finalizer" \
    "building and validating paired Smoke payload" "${smoke_children}" \
    local_cli finalize-real-smoke "${stationary_run_id}" "${transient_run_id}" \
      --comsol-version-output "${comsol_version}" \
      --storage-root "${LOCAL_STORAGE_ROOT}")" ||
    fail 1 "Could not atomically finalize paired Technical Smoke evidence."
  PAIRED_SMOKE_RECEIPT="$(admit_shared_cli_path "${PAIRED_SMOKE_RECEIPT}")"
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
  output="$(shared_plan_submit submit-campaign)" ||
    fail 1 "Shared campaign submission failed."
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

shared_cli() {
  verify_shared_setup_for_output || return $?
  local_cli "$@"
}

resolve_local_storage() {
  [[ -z "${LOCAL_STORAGE_ROOT}" ]] || return 0
  require_command realpath
  LOCAL_STORAGE_ROOT="$(realpath -m -- "${HOST_STORAGE_ROOT}")"
  validate_path "local storage" "${LOCAL_STORAGE_ROOT}"
}

resolve_local_python() {
  [[ "${LOCAL_PYTHON_READY}" != true ]] || return 0
  [[ -x "${GENERATION_NATIVE_VENV}/bin/python" ]] ||
    fail 1 "Native Generation Python environment is missing."
  local_python -c 'import h5py, numpy, scipy, yaml, torch; import src.generation.cli.cli_generation' ||
    fail 1 "Locked native Python environment lacks required Generation dependencies."
  LOCAL_PYTHON_READY=true
}

local_python() {
  env GENERATION_GIT_COMMIT="${SOURCE_COMMIT}"     GENERATION_SOURCE_SHA256="${GENERATION_SOURCE_SHA256}"     GENERATION_NATIVE_VENV="${GENERATION_NATIVE_VENV}"     STORAGE_ROOT="${LOCAL_STORAGE_ROOT:-${HOST_STORAGE_ROOT}}"     "${GENERATION_NATIVE_VENV}/bin/python" "$@"
}

slurm_python_cli() {
  local mode="$1" operation="$2"
  shift
  [[ "${PARTITION:-}" == standard || "${PARTITION:-}" == long ]] ||
    fail 2 "Native Generation Python work requires the configured standard or long partition."
  [[ -x "${HOST_REPO_ROOT}/scripts/generation_python_node.sh" ]] ||
    fail 1 "Native Generation Python worker is missing."
  require_command srun "native Generation Python work"
  resolve_local_storage
  local logs="${RUNTIME_ROOT}/logs/generation" output_log error_log status
  mkdir -p -- "${logs}" || fail 1 "Could not prepare Generation runtime logs."
  output_log="$(mktemp "${logs}/python-${operation}.XXXXXXXX.out")" ||
    fail 1 "Could not prepare Generation Python output log."
  error_log="${output_log%.out}.err"
  export GENERATION_GIT_COMMIT="${SOURCE_COMMIT}"
  export STORAGE_ROOT="${HOST_STORAGE_ROOT}"
  if srun --partition="${PARTITION}" --nodes=1 --ntasks=1 \
    --cpus-per-task=4 --mem=16G --time="${WALL_TIME:-02:00:00}" \
    --job-name="generation-python-${operation}" \
    --chdir="${HOST_REPO_ROOT}" --export=ALL \
    --error="${error_log}" \
    "${HOST_REPO_ROOT}/scripts/generation_python_node.sh" \
    "${HOST_REPO_ROOT}" "${mode}" "$@" | tee "${output_log}"; then
    return 0
  else
    status=$?
    [[ ! -f "${error_log}" ]] || tail -n 80 -- "${error_log}" >&2
    return "${status}"
  fi
}

local_cli() {
  if [[ "${SUBCOMMAND:-}" == run && "${DRY_RUN:-false}" != true \
    && "${PREFLIGHT_ONLY:-false}" != true ]]; then
    case "$1" in
      resume-core-benchmark)
        slurm_python_cli benchmark "$@"
        return $?
        ;;
      prepare-campaign-inputs|resume-campaign|\
        build-campaign-datasets|prepare-gpu-datasets|prepare-all-workflow|\
        advance-campaign-completion|build-campaign-completion-composite|\
        build-campaign-completion-lifecycle|finalize-core-benchmark|\
        finalize-technical-smoke-evidence|finalize-real-smoke|\
        repair-transferred-campaign|repair-partial-campaign-publication|\
        record-pilot-source-inventory|record-shared-pilot-staging|\
        prepare-pilot-check|cleanup-pilot-staging|campaign-transfer-authority|\
        validate-published-campaign|validate-campaign-terminal|\
        validate-campaign-package-state|validate-all-workflow|\
        validate-pilot-check|validate-core-benchmark|validate-real-smoke|\
        validate-campaign-completion-lifecycle)
        slurm_python_cli cli "$@"
        return $?
        ;;
    esac
  fi
  local_python -m "${GENERATION_MODULE}" "$@"
}

local_cli_quiet() {
  local_cli "$@" >/dev/null 2>&1
}

admit_shared_cli_path() {
  local resolved
  resolved="$(realpath -e -- "$1")" || fail 1 "Generation CLI path does not exist: $1"
  [[ "${resolved}" == "${LOCAL_STORAGE_ROOT}/"* \
    || "${resolved}" == "${HOST_REPO_ROOT}/"* ]] ||
    fail 1 "Generation CLI path is outside shared repository and storage: ${resolved}"
  printf '%s' "${resolved}"
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
  authority="$(shared_cli campaign-transfer-authority "${RUN_ID}" \
    --storage-root "${SHARED_STORAGE_ROOT}")" || {
    local status=$?
    (( status != 130 )) || exit 130
    return 1
  }
  local_cli repair-transferred-campaign "${RUN_ID}" \
    --source-host "${CPU_HOST}" --source-storage-root "${SHARED_STORAGE_ROOT}" \
    --authority-json "${authority}" --storage-root "${LOCAL_STORAGE_ROOT}" >/dev/null 2>&1
}

collect_campaign() {
  resolve_local_python
  resolve_shared_layout
  resolve_local_storage
  verify_shared_setup_for_output
  if [[ "${CAMPAIGN_PARTIAL:-false}" == true ]]; then
    local_cli repair-partial-campaign-publication "${RUN_ID}" \
      --source-host shared-filesystem --source-storage-root "${LOCAL_STORAGE_ROOT}" \
      --storage-root "${LOCAL_STORAGE_ROOT}" >/dev/null ||
      fail 1 "Partial campaign shared publication failed validation."
  elif ! gpu_publication_is_valid; then
    repair_existing_campaign_publication ||
      fail 1 "Complete campaign shared publication failed exact source/inventory validation."
  fi
  if [[ "${PILOT_MODE}" == true ]]; then
    local_cli record-pilot-source-inventory "${RUN_ID}" \
      --storage-root "${LOCAL_STORAGE_ROOT}" >/dev/null ||
      fail 1 "Could not bind pilot source inventory."
    local_cli record-shared-pilot-staging "${RUN_ID}" --storage-root "${LOCAL_STORAGE_ROOT}" >/dev/null ||
      fail 1 "Could not bind pilot accounting staging inventory."
  fi
  printf 'Shared campaign publication validated: %s
' "${RUN_ID}"
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

shared_campaign_monitor() {
  shared_cli campaign-status \
    "${RUN_ID}" --format monitor --max-active-cases 8 \
    --storage-root "${SHARED_STORAGE_ROOT}"
}

read_shared_campaign_monitor() {
  local output header kind state state_signature progress_signature extra
  output="$(shared_campaign_monitor)" ||
    fail_preserving_interrupt "$?" 1 "Could not reconstruct campaign case status."
  [[ "${output}" == *$'\n'* ]] || fail 1 "Malformed campaign monitor output."
  header="${output%%$'\n'*}"
  IFS=$'\t' read -r kind state state_signature progress_signature extra <<< "${header}"
  [[ "${kind}" == campaign-monitor && "${state_signature}" =~ ^[0-9a-f]{64}$ \
    && "${progress_signature}" =~ ^[0-9a-f]{64}$ && -z "${extra:-}" ]] ||
    fail 1 "Malformed campaign monitor header."
  SHARED_CAMPAIGN_STATE="${state}"
  SHARED_CAMPAIGN_STATE_SIGNATURE="${state_signature}"
  SHARED_CAMPAIGN_PROGRESS_SIGNATURE="${progress_signature}"
  SHARED_CAMPAIGN_SUMMARY="${output#*$'\n'}"
}

shared_source_status_tsv() {
  shared_cli campaign-source-status \
    "${RUN_ID}" --query-scheduler --include-sizes --format tsv \
    --storage-root "${SHARED_STORAGE_ROOT}"
}

read_shared_source_status() {
  local line kind status_run campaign_state source_state bytes eligibility active extra
  line="$(shared_source_status_tsv)"
  IFS=$'\t' read -r kind status_run campaign_state source_state bytes \
    eligibility active extra <<< "${line}"
  [[ "${kind}" == source-status && "${status_run}" == "${RUN_ID}" \
    && -z "${extra:-}" ]] || fail 1 "Malformed CPU source status."
  validate_nonnegative "CPU retained bytes" "${bytes}"
  SHARED_RUN_STATE="${campaign_state}"
  SHARED_SOURCE_STATE="${source_state}"
  CPU_BYTES_RETAINED="${bytes}"
  CPU_BYTES_RETAINED_EXACT=true
  SHARED_CLEANUP_ELIGIBILITY="${eligibility}"
  SHARED_SOURCE_ACTIVE="${active}"
}

shared_workflow_monitor() {
  shared_cli resume-campaign \
    "${RUN_ID}" --format workflow-monitor --max-active-cases 8 \
    --storage-root "${SHARED_STORAGE_ROOT}"
}

read_shared_workflow_monitor() {
  local output campaign_header source_header tab
  local kind state state_signature progress_signature extra
  local source_kind status_run campaign_state source_state bytes eligibility active source_extra
  local -a monitor_lines=()
  output="$(shared_workflow_monitor)" ||
    fail_preserving_interrupt "$?" 1 "Could not resume and reconstruct campaign status."
  mapfile -t monitor_lines <<< "${output}"
  (( ${#monitor_lines[@]} >= 3 )) || fail 1 "Malformed combined campaign monitor output."
  campaign_header="${monitor_lines[0]}"
  source_header="${monitor_lines[1]}"
  SHARED_CAMPAIGN_SUMMARY="$(printf "%s\n" "${monitor_lines[@]:2}")"
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
  SHARED_CAMPAIGN_STATE="${state}"
  SHARED_CAMPAIGN_STATE_SIGNATURE="${state_signature}"
  SHARED_CAMPAIGN_PROGRESS_SIGNATURE="${progress_signature}"
  SHARED_RUN_STATE="${campaign_state}"
  SHARED_SOURCE_STATE="${source_state}"
  CPU_BYTES_RETAINED="${bytes}"
  SHARED_CLEANUP_ELIGIBILITY="${eligibility}"
  SHARED_SOURCE_ACTIVE="${active}"
}

refresh_failure_cpu_bytes() {
  [[ "${CPU_BYTES_RETAINED_EXACT}" == true ]] && return 0
  [[ "${RUN_KIND:-}" == campaign && -n "${RUN_ID:-}" \
    && -n "${SHARED_STORAGE_ROOT:-}" ]] || return 1
  local line kind status_run campaign_state source_state bytes eligibility active extra
  line="$(shared_cli campaign-source-status "${RUN_ID}" --include-sizes --format tsv \
    --storage-root "${SHARED_STORAGE_ROOT}" 2>/dev/null)" || return 1
  IFS=$'\t' read -r kind status_run campaign_state source_state bytes \
    eligibility active extra <<< "${line}"
  [[ "${kind}" == source-status && "${status_run}" == "${RUN_ID}" \
    && "${bytes}" =~ ^[0-9]+$ && -z "${extra:-}" ]] || return 1
  CPU_BYTES_RETAINED="${bytes}"
  CPU_BYTES_RETAINED_EXACT=true
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

storage_status_report() {
  resolve_local_python
  resolve_shared_layout
  resolve_local_storage
  local -a local_arguments=(
    storage-status --role gpu --metadata-only --omit-run-status
    --storage-root "${LOCAL_STORAGE_ROOT}"
  )
  local -a shared_arguments=(
    storage-status --role cpu --metadata-only --omit-run-status
    --storage-root "${SHARED_STORAGE_ROOT}"
  )
  if [[ -n "${RUN_ID}" ]]; then
    local_arguments+=(--campaign-run-id "${RUN_ID}")
    shared_arguments+=(--campaign-run-id "${RUN_ID}")
  fi
  if [[ -n "${RUN_ID}" ]]; then
    printf 'Campaign status:\n'
    shared_cli campaign-status \
      "${RUN_ID}" --format workflow-monitor --max-active-cases 8 \
      --storage-root "${SHARED_STORAGE_ROOT}"
  fi
  printf 'Durable storage status:\n'
  local_cli "${local_arguments[@]}"
  printf 'Shared source status:\n'
  shared_cli "${shared_arguments[@]}"
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
  )
  [[ -z "${REQUESTED_COMMIT:-}" ]] ||
    continuation_arguments+=(--git-commit "${REQUESTED_COMMIT}")
  if (( REPLACEMENT_POOL_OPTION_COUNT == 1 )); then
    continuation_arguments+=(--replacement-pool-size "${REPLACEMENT_POOL_SIZE}")
    if [[ "${COMPLETION_OWNER_PERSISTED:-false}" != true \
      && -n "${COMPLETION_PARENT_RUN_ID:-}" ]]; then
      continuation_arguments+=(--parent-run-id "${COMPLETION_PARENT_RUN_ID}")
    fi
  fi
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
        WORKFLOW_FAILURE_EVIDENCE="canonical=${canonical} visible=${visible}"
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

shared_benchmark_plan_submit() (
  local operation="$1"
  verify_shared_setup_for_output || return $?
  slurm_python_cli benchmark "${operation}" "${BENCHMARK_SUITE_PATH}" \
    --git-commit "${REQUESTED_COMMIT}" --storage-root "${SHARED_STORAGE_ROOT}"
)


collect_core_benchmark() {
  local_cli validate-core-benchmark "${RUN_ID}"     --storage-root "${LOCAL_STORAGE_ROOT}" >/dev/null ||
    fail 1 "Shared core benchmark did not validate."
  local_cli core-benchmark-summary "${RUN_ID}" --format markdown     --storage-root "${LOCAL_STORAGE_ROOT}"
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
      LEAF_EXISTING_DETAIL="campaign_run_id=$RUN_ID artifact_set=$artifact_identity compatible shared workflow"
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
      resolve_shared_layout
      verify_shared_setup >/dev/null
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
        read_shared_workflow_monitor
        local campaign_detail
        campaign_detail="$SHARED_CAMPAIGN_SUMMARY"$'\n'"Source storage: state=$SHARED_SOURCE_STATE retained_bytes=$CPU_BYTES_RETAINED"
        generation_console_progress units 5 9 "Work units" RUNNING           "$SHARED_CAMPAIGN_STATE_SIGNATURE|$SHARED_SOURCE_STATE|$SHARED_SOURCE_ACTIVE"           "$campaign_detail"           "$SHARED_CAMPAIGN_PROGRESS_SIGNATURE|$CPU_BYTES_RETAINED"
        case "$SHARED_CAMPAIGN_STATE" in
          successful|transfer_complete)
            shared_cli validate-campaign-terminal "$RUN_ID" \
              --storage-root "$SHARED_STORAGE_ROOT" >/dev/null
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
            fail 1 "Campaign entered unsupported state: $SHARED_CAMPAIGN_STATE"
            ;;
        esac
        ;;
      benchmark)
        shared_cli resume-core-benchmark "$RUN_ID" \
          --storage-root "$SHARED_STORAGE_ROOT" >/dev/null
        local output header state state_signature progress_signature detail extra
        output="$(shared_cli core-benchmark-status "$RUN_ID" \
          --storage-root "$SHARED_STORAGE_ROOT" --format monitor)" ||
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
        LEAF_EXISTING_DETAIL="campaign_run_id=$RUN_ID complete shared workflow"
        return
      fi
      if select_compatible_campaign_source; then
        return
      fi
      resolve_shared_layout
      verify_shared_setup >/dev/null
      LEAF_EXISTING_DETAIL="campaign_run_id=$RUN_ID continuation inspected"
      ;;
    benchmark)
      resolve_benchmark_contract >/dev/null
      [[ "$SCHEDULER_KIND" == slurm ]] ||
        fail 2 "Core benchmarking requires configured scheduler=slurm."
      resolve_shared_layout
      verify_shared_setup >/dev/null
      local comsol_version identity_json
      comsol_version="$(native_comsol_version)"
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
        LEAF_STATE=complete
        LEAF_EXISTING_DETAIL="benchmark_run_id=$RUN_ID complete shared workflow"
      fi
      ;;
    *) fail 2 "Unsupported common leaf run kind: $RUN_KIND" ;;
  esac
}

materialize_leaf_inputs() {
  case "$RUN_KIND" in
    campaign)
      local input_record input_kind generated reused extra
      input_record="$(prepare_shared_campaign_inputs)" ||
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
      output="$(shared_benchmark_plan_submit materialize-core-benchmark-inputs)" ||
        fail 1 "Benchmark canonical input preparation failed before submission."
      observed_run="$(printf '%s' "$output" | local_python -c 'import json, sys
print(json.load(sys.stdin)["benchmark_run_id"])')" ||
        fail 1 "Benchmark input preparation returned no run identity."
      [[ "$observed_run" == "$RUN_ID" ]] ||
        fail 1 "Benchmark input preparation identity disagrees with the common run plan."
      LEAF_INPUT_DETAIL="benchmark_run_id=$RUN_ID canonical input ready in Slurm allocation"
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
      output="$(shared_benchmark_plan_submit submit-core-benchmark)" ||
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

finalize_leaf_cpu_evidence() {
  case "$RUN_KIND" in
    campaign) ;;
    benchmark)
      shared_cli finalize-core-benchmark "$RUN_ID" \
        --storage-root "$SHARED_STORAGE_ROOT" >/dev/null
      ;;
    *) fail 2 "Unsupported CPU-finalization adapter: $RUN_KIND" ;;
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
  KEEP_CPU_SOURCE=true
  case "$RUN_KIND" in
    campaign)
      prepare_all_receipt >/dev/null
      read_shared_source_status
      if [[ "$CAMPAIGN_PURPOSE" == pilot_check         && "${CAMPAIGN_PARTIAL:-false}" != true ]]; then
        cleanup_pilot_staging >/dev/null
        record_pilot_cleanup_result
      fi
      ;;
    benchmark)
      local_cli validate-core-benchmark "${RUN_ID}"         --storage-root "${LOCAL_STORAGE_ROOT}" >/dev/null
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

sync_completion_parent_evidence() {
  validate_completion_id "${COMPLETION_ID}"
  validate_run_id "${COMPLETION_PARENT_RUN_ID}"
  validate_digest "${COMPLETION_PARENT_PARTIAL_SHA256}"
  [[ -f "${COMPLETION_PARENT_PARTIAL_PATH}" && ! -L "${COMPLETION_PARENT_PARTIAL_PATH}" ]] ||
    fail 1 "Completion parent partial evidence is missing or unsafe."
  local relative="01_generation/meta/completion-inputs/${COMPLETION_ID}/campaign_partial.json"
  validate_transfer_path "${relative}"
  COMPLETION_SHARED_PARENT_PARTIAL="${LOCAL_STORAGE_ROOT}/${relative}"
  local directory temporary observed
  directory="$(dirname -- "${COMPLETION_SHARED_PARENT_PARTIAL}")"
  mkdir -p -- "${directory}"
  [[ -d "${directory}" && ! -L "${directory}" ]] ||
    fail 1 "Completion input directory is unsafe."
  temporary="$(mktemp "${directory}/.campaign_partial.XXXXXXXX")" ||
    fail 1 "Could not create atomic completion evidence copy."
  if ! cp -- "${COMPLETION_PARENT_PARTIAL_PATH}" "${temporary}"; then
    rm -f -- "${temporary}"
    fail 1 "Could not stage exact parent partial evidence."
  fi
  observed="$(sha256sum -- "${temporary}")"; observed="${observed%% *}"
  [[ "${observed}" == "${COMPLETION_PARENT_PARTIAL_SHA256}" ]] || {
    rm -f -- "${temporary}"
    fail 1 "Staged parent partial evidence hash changed."
  }
  if [[ -e "${COMPLETION_SHARED_PARENT_PARTIAL}" ]]; then
    [[ -f "${COMPLETION_SHARED_PARENT_PARTIAL}" && ! -L "${COMPLETION_SHARED_PARENT_PARTIAL}" ]] ||
      fail 1 "Existing completion input is unsafe."
    cmp -s -- "${temporary}" "${COMPLETION_SHARED_PARENT_PARTIAL}" || {
      rm -f -- "${temporary}"
      fail 1 "Existing completion input conflicts with parent evidence."
    }
    rm -f -- "${temporary}"
  else
    mv -- "${temporary}" "${COMPLETION_SHARED_PARENT_PARTIAL}"
  fi
}

render_completion_plan() {
  local report="$1"
  printf '%s' "${report}" | local_python -c 'import json, sys
value = json.load(sys.stdin)
def label(item):
    return str(item.get("material_family") or item["batch_id"]).replace("_", " ").title()
def rows(field):
    for item in value["target_batches"]:
        if field == "inventory":
            count = item["current_successes"]
            target = item["target_successes"]
            print("  {:18} {} / {}".format(label(item), count, target))
        elif field == "deficit":
            print("  {:18} {}".format(label(item), item["uncovered_deficit"]))
        else:
            print("  {:18} {}".format(label(item), item["new_replacements_required"]))
execution = value["execution"]
max_running = execution["max_running_cases"]
if max_running is None:
    max_running = "unlimited"
print("Completion reconciliation ........ OK")
print("Completion mode: resume partial campaign")
print("Parent campaign: {}".format(value["parent_run_id"]))
print("Parent state: {}".format(value["parent_state"]))
print("Completion ID: {}".format(value["completion_id"]))
print("\n{} successful inventory:".format("Final" if value["status"] == "complete" else "Existing"))
rows("inventory")
remaining = sum(value["success_deficits"].values())
print("\nRemaining deficits:")
rows("deficit")
print("  {:18} {}".format("Total", remaining))
print("\nRemaining successful deficit: {}".format(remaining))
print("New COMSOL work required: {}".format(value.get("new_comsol_work_required", remaining)))
print("Composite source: state={}".format(value.get("composite_source_state", "pending")))
print("\nReplacement pool:")
print("  high water         {}".format(value["pool_high_water"]))
print("  already consumed   {}".format(value["pool_consumed"]))
print("  active/reserving   {}".format(value["reserved_candidates"]))
print("  reserve remaining  {}".format(value["pool_remaining"]))
print("\nReplacement plan:\n  newly required:")
rows("required")
print("  total newly required = {}".format(sum(value["new_replacements_required"].values())))
print("\nExecution:")
print("  max admission cases = {}".format(execution["max_admission_cases"]))
print("  max running cases   = {}".format(max_running))
print("  cores per case      = {}".format(execution["cores_per_case"]))
print("  partition           = {}".format(execution["partition"]))
print("  wall time           = {}".format(execution["wall_time"]))
print("  poll interval       = {} s".format(execution["poll_interval_seconds"]))
print("\nNext:\n  {}".format(value["next_operation"]))' ||
    fail 1 "Could not render exact completion startup plan."
}

initialize_shared_completion() {
  local output record status observed_id extra
  local -a arguments=(
    initialize-campaign-completion "${CAMPAIGN_RELATIVE_PATH}"
    --parent-run-id "${COMPLETION_PARENT_RUN_ID}"
    --parent-partial "${COMPLETION_SHARED_PARENT_PARTIAL}"
    --parent-partial-sha256 "${COMPLETION_PARENT_PARTIAL_SHA256}"
    --storage-root "${SHARED_STORAGE_ROOT}"
  )
  if (( REPLACEMENT_POOL_OPTION_COUNT == 1 )); then
    arguments+=(--replacement-pool-size "${REPLACEMENT_POOL_SIZE}")
  fi
  output="$(shared_cli "${arguments[@]}")" ||
    fail 1 "Could not initialize or extend the shared campaign completion owner."
  COMPLETION_INITIAL_STATUS_JSON="${output}"
  record="$(printf '%s' "${output}" | local_python -c 'import json, sys
value = json.load(sys.stdin)
print("\t".join((str(value["status"]), str(value["completion_id"]))))')" ||
    fail 1 "Could not decode shared completion initialization."
  IFS=$'\t' read -r status observed_id extra <<< "${record}"
  [[ -z "${extra:-}" ]] || fail 1 "Malformed shared completion initialization result."
  validate_completion_id "${observed_id}"
  [[ "${observed_id}" == "${COMPLETION_ID}" ]] ||
    fail 1 "Shared completion identity differs from the compatible parent resolution."
  COMPLETION_OWNER_PERSISTED=true
  case "${status}" in
    active|complete|pool_exhausted) ;;
    failure_circuit_open)
      printf 'COMPLETION FAILURE CIRCUIT OPEN: completion_id=%s; configured replacement failure allowance is exhausted. Evidence is retained and no new candidates will be admitted.\n' \
        "${COMPLETION_ID}" >&2
      return 4
      ;;
    *) fail 1 "Shared completion initialization returned unsupported status: ${status}" ;;
  esac
}

advance_shared_completion() {
  validate_positive "configured poll_interval_seconds" "${STATUS_POLL_SECONDS}"
  local previous_run_id="${RUN_ID}"
  while true; do
    local output record status observed_id pool_size monitor_run active_count
    local required_total allocated_candidates reserved_candidates next_operation extra
    local startup_detail
    printf -v startup_detail \
      "completion_id=%s\noperation=%s" \
      "${COMPLETION_ID}" \
      "reconciling evidence and preparing exact replacement membership"
    generation_console_progress completion-startup 3 9 "Canonical inputs" RUNNING \
      "${COMPLETION_ID}" "${startup_detail}" "${COMPLETION_ID}"
    output="$(generation_run_with_heartbeat \
      "completion-advance-${COMPLETION_ID}" 3 9 "Canonical inputs" \
      "reconciling evidence, allocating exact deficits, and preparing replacement inputs" \
      "completion_id=${COMPLETION_ID}" \
      shared_cli advance-campaign-completion "${CAMPAIGN_RELATIVE_PATH}" \
      "${COMPLETION_ID}" --git-commit "${REQUESTED_COMMIT}" \
      --storage-root "${SHARED_STORAGE_ROOT}")" ||
      fail_preserving_interrupt "$?" 1 "Could not advance replacement campaign completion."
    record="$(printf '%s' "${output}" | local_python -c 'import json, sys
value = json.load(sys.stdin)
active = list(value.get("active_run_ids", []))
rounds = list(value.get("active_round_run_ids", []))
monitor = (rounds or active or ["-"])[0]
next_operation = str(value["next_operation"])
if any(character in next_operation for character in "\t\r\n"):
    raise SystemExit("unsafe completion next-operation text")
print("\t".join((
    str(value["status"]), str(value["completion_id"]),
    str(value["replacement_pool_size"]), str(monitor), str(len(active)),
    str(sum(value["new_replacements_required"].values())),
    str(value["allocated_candidates"]), str(value["reserved_candidates"]),
    next_operation,
)))')" || fail 1 "Could not decode replacement completion status."
    IFS=$'\t' read -r status observed_id pool_size monitor_run active_count \
      required_total allocated_candidates reserved_candidates next_operation extra <<< "${record}"
    [[ -z "${extra:-}" ]] || fail 1 "Malformed replacement completion status."
    validate_completion_id "${observed_id}"
    [[ "${observed_id}" == "${COMPLETION_ID}" ]] ||
      fail 1 "Replacement completion status changed owner identity."
    validate_positive "replacement completion pool high-water" "${pool_size}"
    validate_nonnegative "active replacement campaign count" "${active_count}"
    validate_nonnegative "newly required replacement count" "${required_total}"
    validate_nonnegative "allocated replacement count" "${allocated_candidates}"
    validate_nonnegative "reserved replacement count" "${reserved_candidates}"
    case "${status}" in
      complete)
        generation_console_stage 4 9 "Work-unit plan" OK \
          "exact deficits=0 allocated=${allocated_candidates} reserved=0"
        printf 'COMPLETION READY: completion_id=%s parent_run_id=%s pool_high_water=%s\n' \
          "${COMPLETION_ID}" "${COMPLETION_PARENT_RUN_ID}" "${pool_size}"
        RUN_ID="${previous_run_id}"
        return 0
        ;;
      active)
        generation_console_stage 3 9 "Canonical inputs" OK \
          "replacement inputs admitted through normal campaign preparation"
        generation_console_stage 4 9 "Work-unit plan" OK \
          "allocated=${allocated_candidates} active_or_reserved=${reserved_candidates} newly_uncovered=${required_total} active_runs=${active_count}"
        if [[ "${monitor_run}" != - ]]; then
          validate_run_id "${monitor_run}"
          RUN_ID="${monitor_run}"
          CAMPAIGN_PARTIAL=false
          monitor_generation_units campaign
          RUN_ID="${previous_run_id}"
          continue
        fi
        generation_console_progress completion-reconcile 5 9 "Work units" RUNNING \
          "${allocated_candidates}|${reserved_candidates}|${required_total}" \
          "operation=${next_operation}" \
          "${allocated_candidates}|${reserved_candidates}|${required_total}"
        sleep "${STATUS_POLL_SECONDS}" || true
        ;;
      failure_circuit_open)
        printf 'COMPLETION FAILURE CIRCUIT OPEN: completion_id=%s; configured replacement failure allowance is exhausted. Evidence is retained and no new candidates will be admitted.\n' \
          "${COMPLETION_ID}" >&2
        RUN_ID="${previous_run_id}"
        return 4
        ;;
      pool_exhausted)
        (( required_total > 0 )) ||
          fail 1 "Pool exhaustion lacks an uncovered replacement deficit."
        local next_pool=$((pool_size + required_total))
        printf 'COMPLETION POOL EXHAUSTED: completion_id=%s\n' "${COMPLETION_ID}" >&2
        printf 'remaining successful deficit = %s\n' "${required_total}" >&2
        printf 'persisted replacement pool high-water = %s\n' "${pool_size}" >&2
        printf 'unused authorized capacity = 0\n\nNext:\n' >&2
        print_command ./scripts/generation_workflow.sh run \
          "${RUN_CONFIG_ARGUMENT}" --replacement-pool-size "${next_pool}" >&2
        RUN_ID="${previous_run_id}"
        return 3
        ;;
      *) fail 1 "Replacement completion entered unsupported state: ${status}" ;;
    esac
  done
}

sync_completion_state_and_plan() {
  COMPLETION_TRANSFER_JSON="$(shared_cli campaign-completion-transfer-plan \
    "${COMPLETION_ID}" --storage-root "${SHARED_STORAGE_ROOT}")" ||
    fail 1 "Could not resolve successful replacement transfer membership."
  local records kind field2 field3 field4 field5 field6 extra
  records="$(printf '%s' "${COMPLETION_TRANSFER_JSON}" | local_python -c 'import json, sys
value = json.load(sys.stdin)
def clean(item):
    text = str(item)
    if any(character in text for character in "\t\r\n"):
        raise SystemExit("completion transfer plan contains unsafe shell transport text")
    return text
print("\t".join((
    "completion", clean(value["completion_id"]), clean(value["parent_run_id"]),
    clean(value["parent_partial_sha256"]), clean(value["completion_state_path"]),
    clean(value["completion_state_sha256"]),
)))
for item in value["replacement_runs"]:
    print("\t".join((
        "replacement-run", clean(item["campaign_run_id"]),
        "true" if item["partial"] else "false", clean(item["campaign_state"]),
    )))
for item in value["replacement_campaigns"]:
    print("\t".join((
        "replacement", clean(item["candidate_id"]), clean(item["target_batch_id"]),
        clean(item["campaign_run_id"]), clean(item["terminal_batch_id"]),
    )))')" || fail 1 "Could not decode completion transfer plan."
  COMPLETION_REPLACEMENT_RUN_IDS=()
  COMPLETION_REPLACEMENT_RUN_PARTIAL=()
  COMPLETION_REPLACEMENT_TERMINAL_BATCH_IDS=()
  local state_path="" state_sha=""
  while IFS=$'\t' read -r kind field2 field3 field4 field5 field6 extra; do
    [[ -z "${extra:-}" ]] || fail 1 "Malformed completion transfer-plan row."
    case "${kind}" in
      completion)
        [[ "${field2}" == "${COMPLETION_ID}" \
          && "${field3}" == "${COMPLETION_PARENT_RUN_ID}" \
          && "${field4}" == "${COMPLETION_PARENT_PARTIAL_SHA256}" ]] ||
          fail 1 "Completion transfer plan changed immutable parent identity."
        state_path="${field5}"
        state_sha="${field6}"
        validate_digest "${state_sha}"
        ;;
      replacement-run)
        validate_run_id "${field2}"
        [[ "${field3}" == true || "${field3}" == false ]] ||
          fail 1 "Malformed replacement run collection mode."
        [[ "${field4}" == complete || "${field4}" == completed_with_failures ]] ||
          fail 1 "Malformed replacement run terminal state."
        COMPLETION_REPLACEMENT_RUN_IDS+=("${field2}")
        COMPLETION_REPLACEMENT_RUN_PARTIAL+=("${field3}")
        ;;
      replacement)
        [[ "${field2}" =~ ^replacement__[0-9a-f]{24}$ ]] ||
          fail 1 "Malformed replacement candidate identity."
        validate_run_id "${field4}"
        validate_batch_name "${field3}"
        validate_batch_name "${field5}"
        COMPLETION_REPLACEMENT_TERMINAL_BATCH_IDS+=("${field5}")
        ;;
      *) fail 1 "Unknown completion transfer-plan row." ;;
    esac
  done <<< "${records}"
  (( ${#COMPLETION_REPLACEMENT_RUN_IDS[@]} > 0 \
    && ${#COMPLETION_REPLACEMENT_RUN_IDS[@]} == ${#COMPLETION_REPLACEMENT_RUN_PARTIAL[@]} \
    && ${#COMPLETION_REPLACEMENT_TERMINAL_BATCH_IDS[@]} > 0 )) ||
    fail 1 "Complete campaign completion has malformed successful replacement transfer membership."
  local expected_state="${SHARED_STORAGE_ROOT}/01_generation/meta/completions/${COMPLETION_ID}/completion.json"
  [[ "${state_path}" == "${expected_state}" ]] ||
    fail 1 "Completion state path differs from its dedicated owner."
  [[ -f "${state_path}" && ! -L "${state_path}" ]] ||
    fail 1 "Shared completion state is missing or unsafe."
  local observed
  observed="$(sha256sum -- "${state_path}")"
  observed="${observed%% *}"
  [[ "${observed}" == "${state_sha}" ]] ||
    fail 1 "Shared completion state failed exact digest validation."
  local_cli campaign-completion-status "${COMPLETION_ID}"     --config "${CAMPAIGN_CONFIG_PATH}" --storage-root "${LOCAL_STORAGE_ROOT}" >/dev/null ||
    fail 1 "Shared completion state did not validate."
}

collect_completion_replacements() {
  local parent_run="${COMPLETION_PARENT_RUN_ID}" replacement_run partial index total successful_total
  total="${#COMPLETION_REPLACEMENT_RUN_IDS[@]}"
  successful_total="${#COMPLETION_REPLACEMENT_TERMINAL_BATCH_IDS[@]}"
  for ((index=0; index<total; index++)); do
    replacement_run="${COMPLETION_REPLACEMENT_RUN_IDS[index]}"
    partial="${COMPLETION_REPLACEMENT_RUN_PARTIAL[index]}"
    RUN_ID="${replacement_run}"
    CAMPAIGN_PARTIAL="${partial}"
    PILOT_MODE=false
    generation_console_stage 6 9 "Host publication" RUNNING \
      "collecting successful replacements validated=${index}/${total}
replacement=${RUN_ID} partial=${CAMPAIGN_PARTIAL} state=collecting"
    collect_campaign >/dev/null
    generation_console_stage 6 9 "Host publication" RUNNING \
      "collecting successful replacements validated=$((index + 1))/${total}
replacement=${RUN_ID} partial=${CAMPAIGN_PARTIAL} state=complete"
  done
  generation_console_stage 6 9 "Host publication" OK \
    "successful replacement publications=${successful_total}/${successful_total} replacement_runs=${total}/${total}"
  CAMPAIGN_PARTIAL=false
  RUN_ID="${parent_run}"
}

finalize_completion_composite() {
  local parent_run="${COMPLETION_PARENT_RUN_ID}" cleanup_json
  printf 'Composite source:\n  state=building_or_reusing expected_completion=%s\n' "${COMPLETION_ID}"
  local_cli build-campaign-completion-composite "${COMPLETION_ID}" \
    --storage-root "${LOCAL_STORAGE_ROOT}" >/dev/null ||
    fail 1 "Could not construct exact parent-success plus replacement-success membership."
  generation_console_stage 7 9 "Packages/finalizer" RUNNING \
    "source=completion composite completion_id=${COMPLETION_ID}"
  printf 'Dataset packages:\n  source=composite completion\n'
  printf 'PT shards:\n  source=final Dataset identity\n'
  local_cli build-campaign-completion-lifecycle "${parent_run}" "${COMPLETION_ID}" \
    --storage-root "${LOCAL_STORAGE_ROOT}" >/dev/null ||
    fail 1 "Could not build completion Dataset packages, PT shards, smoke, and readiness evidence."
  generation_console_stage 7 9 "Packages/finalizer" OK \
    "source=completion composite completion_id=${COMPLETION_ID}"
  local_cli validate-campaign-completion-lifecycle "${parent_run}" "${COMPLETION_ID}" \
    --storage-root "${LOCAL_STORAGE_ROOT}" >/dev/null ||
    fail 1 "Completion lifecycle failed its pre-cleanup validation."
  cleanup_json="$(local_cli campaign-completion-cleanup-plan "${parent_run}" \
    "${COMPLETION_ID}" --storage-root "${LOCAL_STORAGE_ROOT}")" ||
    fail 1 "Could not admit successful replacement-only cleanup eligibility."
  local_python -c 'import json, sys
transfer = json.loads(sys.argv[1])
cleanup = json.loads(sys.argv[2])
expected = sorted(item["terminal_batch_id"] for item in transfer["replacement_campaigns"])
observed = sorted(item["terminal_batch_id"] for item in cleanup["sources"])
if not cleanup.get("eligible") or observed != expected:
    raise SystemExit("completion cleanup plan differs from successful replacement transfer membership")' \
    "${COMPLETION_TRANSFER_JSON}" "${cleanup_json}" ||
    fail 1 "Completion cleanup plan includes missing, failed, or non-replacement sources."
  generation_console_stage 8 9 "Retention policy" RETAINED \
    "completion-owned replacement sources remain outside standalone workflow finalization"
  local_cli validate-campaign-completion-lifecycle "${parent_run}" "${COMPLETION_ID}" \
    --storage-root "${LOCAL_STORAGE_ROOT}" >/dev/null ||
    fail 1 "Completion lifecycle changed after replacement-only retention."
  RUN_ID="${parent_run}"
  CAMPAIGN_PARTIAL=false
  LEAF_RESULT=OK
  LEAF_STATE=complete
  SHARED_CAMPAIGN_STATE=complete_composite
}

run_campaign_with_completion() {
  [[ "${RUN_KIND}" == campaign && -n "${REPLACEMENT_POOL_SIZE}" ]] ||
    fail 2 "Internal completion workflow requires one campaign and cumulative pool."
  generation_console_progress completion-inspection 1 9 "Run plan" RUNNING \
    "${RUN_PLAN_ID}|parent-resolution" \
    "operation=resolving a structurally compatible partial parent" \
    "${RUN_PLAN_ID}|parent-resolution"
  RUN_LEAF_CONFIG="${RUN_PLAN_CONFIG}"
  resolve_campaign "${RUN_LEAF_CONFIG}"
  resolve_completion_parent_local
  local parent_run=""
  case "${COMPLETION_PARENT_STATUS}" in
    compatible_partial)
      parent_run="${COMPLETION_PARENT_RUN_ID}"
      ;;
    compatible_active)
      [[ "${COMPLETION_PARENT_RUN_ID}" == "${RUN_PLAN_ID}" ]] ||
        fail 1 "Compatible active parent belongs to a different source commit; wait for transfer or use its exact commit."
      run_leaf_plan campaign "${RUN_PLAN_CONFIG}" "${RUN_PLAN_ID}" \
        "${RUN_PLAN_PURPOSE}" "${RUN_PLAN_PROFILE}"
      ;;
    fresh|compatible_complete)
      run_leaf_plan campaign "${RUN_PLAN_CONFIG}" "${RUN_PLAN_ID}" \
        "${RUN_PLAN_PURPOSE}" "${RUN_PLAN_PROFILE}"
      ;;
    *) fail 1 "Unsupported completion parent state before execution: ${COMPLETION_PARENT_STATUS}" ;;
  esac
  if [[ -z "${parent_run}" ]]; then
    if [[ "${LEAF_RESULT}" != PARTIAL ]]; then
      return 0
    fi
    parent_run="${RUN_ID}"
    resolve_completion_parent_local "${parent_run}"
    [[ "${COMPLETION_PARENT_STATUS}" == compatible_partial \
      && "${COMPLETION_PARENT_RUN_ID}" == "${parent_run}" ]] ||
      fail 1 "Newly partial campaign did not resolve to its exact transferred parent evidence."
  fi
  generation_console_progress completion-inspection 1 9 "Run plan" RUNNING \
    "${COMPLETION_ID}|execution" \
    "operation=resolving normal campaign admission, resources, retries, and polling" \
    "${COMPLETION_ID}|execution"
  resolve_configured_resources executable
  validate_resources
  resolve_shared_layout
  generation_console_progress completion-inspection 1 9 "Run plan" RUNNING \
    "${COMPLETION_ID}|cpu-setup" \
    "operation=validating the shared native execution environment" \
    "${COMPLETION_ID}|cpu-setup"
  verify_shared_setup >/dev/null
  generation_console_progress completion-inspection 1 9 "Run plan" RUNNING \
    "${COMPLETION_ID}|parent" \
    "operation=validating retained parent successes and persisted completion evidence" \
    "${COMPLETION_ID}|parent"
  sync_completion_parent_evidence
  generation_console_progress completion-inspection 1 9 "Run plan" RUNNING \
    "${COMPLETION_ID}|state" \
    "operation=reconciling existing replacement state and exact per-material reservations" \
    "${COMPLETION_ID}|state"
  initialize_shared_completion
  render_completion_plan "${COMPLETION_INITIAL_STATUS_JSON}"
  generation_console_stage 2 9 "Existing state" OK \
    "parent_run=${COMPLETION_PARENT_RUN_ID} completion_id=${COMPLETION_ID}"
  advance_shared_completion || return $?
  sync_completion_state_and_plan
  collect_completion_replacements
  finalize_completion_composite
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


  ALL_STAGE="atomic shared publication"
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
      *) fail 1 "Workflow child returned unsupported state: $LEAF_STATE" ;;
    esac
  done
  COMPOSITE_CHILD_MODE=false
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
  resolve_shared_layout
  verify_shared_setup >/dev/null
  local version
  version="$(native_comsol_version)"
  printf 'PREFLIGHT COMPLETE: plan=%s kind=%s host=%s COMSOL=%s\n'     "$RUN_PLAN_ID" "$RUN_KIND" "$CPU_HOST" "$version"
}

execute_generation_run() {
  HUMAN_WORKFLOW_MODE=true
  ALL_WORKFLOW_ACTIVE=true
  ALL_STAGE="common plan resolution"
  generation_console_stage 1 9 "Run plan" OK     "kind=$RUN_KIND identity=$RUN_PLAN_ID config=$RUN_CONFIG_RELATIVE"
  case "$RUN_KIND" in
    campaign)
      if [[ -z "${REPLACEMENT_POOL_SIZE}" ]]; then
        RUN_LEAF_CONFIG="${RUN_PLAN_CONFIG}"
        resolve_campaign "${RUN_LEAF_CONFIG}"
        resolve_completion_parent_local
      fi
      if [[ -n "${REPLACEMENT_POOL_SIZE}" ]]; then
        run_campaign_with_completion
      else
        run_leaf_plan campaign "$RUN_PLAN_CONFIG" "$RUN_PLAN_ID" \
          "$RUN_PLAN_PURPOSE" "$RUN_PLAN_PROFILE"
      fi
      case "$LEAF_STATE" in
        complete)
          generation_console_final             "run_identity=$RUN_PLAN_ID campaign_run_id=$RUN_ID state=${SHARED_CAMPAIGN_STATE:-complete} result=$LEAF_RESULT"
          ;;
        *) fail 1 "Campaign returned unsupported final state: $LEAF_STATE" ;;
      esac
      ;;
    benchmark)
      run_leaf_plan benchmark "$RUN_PLAN_CONFIG" "$RUN_PLAN_ID" "-" "-"
      case "$LEAF_STATE" in
        complete)
          generation_console_final             "run_identity=$RUN_PLAN_ID benchmark_run_id=$RUN_ID state=complete result=$LEAF_RESULT"
          ;;
        *) fail 1 "Benchmark returned unsupported final state: $LEAF_STATE" ;;
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
  resolve_shared_layout
  printf 'Benchmark status:\n'
  shared_cli core-benchmark-status \
    "${RUN_ID}" --storage-root "${SHARED_STORAGE_ROOT}" --format summary
  printf 'CPU source status:\n'
  shared_cli core-benchmark-source-status "${RUN_ID}" \
    --storage-root "${SHARED_STORAGE_ROOT}"
  if local_cli validate-core-benchmark "${RUN_ID}" \
    --storage-root "${LOCAL_STORAGE_ROOT}" >/dev/null 2>&1; then
    printf 'Host publication state: complete\n'
  else
    printf 'Host publication state: absent_or_incomplete\n'
  fi
}

completion_status_report_for_config() {
  resolve_local_storage
  resolve_local_python
  local report record resolution_status completion_id extra
  report="$(local_cli find-completion-parent "${CAMPAIGN_CONFIG_PATH}" \
    --storage-root "${LOCAL_STORAGE_ROOT}" --allow-untransferred)" ||
    fail 1 "Could not resolve read-only completion status for the campaign config."
  printf 'Completion status:\n%s\n' "${report}"
  record="$(printf '%s' "${report}" | local_python -c 'import json, sys
value = json.load(sys.stdin)
identifier = value.get("completion_id") or value.get("expected_completion_id") or "-"
print("\t".join((str(value["status"]), str(identifier))))')" ||
    fail 1 "Could not decode read-only completion status."
  IFS=$'\t' read -r resolution_status completion_id extra <<< "${record}"
  [[ -z "${extra:-}" ]] || fail 1 "Malformed read-only completion status."
  if [[ "${resolution_status}" == compatible_partial ]]; then
    validate_completion_id "${completion_id}"
    printf 'GPU completion and finalization status:\n'
    local_cli campaign-completion-status "${completion_id}" --if-present \
      --config "${CAMPAIGN_CONFIG_PATH}" --storage-root "${LOCAL_STORAGE_ROOT}"
    resolve_shared_layout
    printf 'CPU completion execution status:\n'
    shared_cli campaign-completion-status "${completion_id}" --if-present \
      --config "${CAMPAIGN_RELATIVE_PATH}" --storage-root "${SHARED_STORAGE_ROOT}"
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
        completion_status_report_for_config
        storage_status_report
        ;;
      benchmark)
        RUN_LEAF_CONFIG="${RUN_PLAN_CONFIG}"
        resolve_benchmark_contract >/dev/null
        resolve_shared_layout
        local version identity_json
        version="$(native_comsol_version)"
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
          completion_status_report_for_config
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
    resolve_shared_layout
    local -a benchmark_arguments=(
      cancel-core-benchmark "${RUN_ID}"
      --storage-root "${SHARED_STORAGE_ROOT}"
    )
    [[ "${FORCE_CANCEL}" != true ]] || benchmark_arguments+=(--force)
    shared_cli "${benchmark_arguments[@]}"
    return
  fi
  validate_run_id "${RUN_ID}"
  resolve_workflow_campaigns
  resolve_shared_layout
  local -a campaign_arguments=(
    cancel-campaign "${RUN_ID}" --storage-root "${SHARED_STORAGE_ROOT}"
  )
  [[ "${FORCE_CANCEL}" != true ]] || campaign_arguments+=(--force)
  shared_cli "${campaign_arguments[@]}"
}

submit_generation_smoke() {
  local worker="${HOST_REPO_ROOT}/scripts/generation_smoke_node.sh"
  local logs="${RUNTIME_ROOT}/logs/generation"
  require_command sbatch "native Generation smoke submission"
  [[ -x "${GENERATION_NATIVE_VENV}/bin/python" ]] ||
    fail 1 "Native Generation Python environment is missing: ${GENERATION_NATIVE_VENV}"
  [[ -f "${worker}" && -x "${worker}" && ! -L "${worker}" ]] ||
    fail 1 "Native Generation smoke worker is missing or unsafe: ${worker}"
  mkdir -p -- "${logs}" || fail 1 "Could not prepare Generation runtime logs: ${logs}"
  export GENERATION_GIT_COMMIT="${SOURCE_COMMIT}"
  local job_id
  job_id="$(sbatch --parsable \
    --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G --time=00:10:00 \
    "--partition=${SMOKE_PARTITION}" "--chdir=${HOST_REPO_ROOT}" \
    --job-name=generation-native-smoke \
    "--output=${logs}/slurm-%j.out" "--error=${logs}/slurm-%j.err" \
    --export=ALL \
    "${worker}" "${HOST_REPO_ROOT}")" || fail 1 "Native Generation smoke submission failed."
  [[ "${job_id}" =~ ^[0-9]+(;[A-Za-z0-9._-]+)?$ ]] ||
    fail 1 "Native Generation smoke returned an invalid Slurm job ID: ${job_id}"
  printf 'GENERATION SMOKE SUBMITTED job=%s partition=%s logs=%s\n' \
    "${job_id}" "${SMOKE_PARTITION}" "${logs}"
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

if [[ "${ORIGINAL_ARGUMENTS[0]}" == inputs ]]; then
  shift
  (( $# >= 1 )) || fail 2 "inputs requires one campaign config and case selection."
  input_config="$1"
  shift
  input_arguments=("$@")
  REQUESTED_COMMIT=""
  input_commit_seen=false
  INPUT_DRY_RUN=false
  while (( $# > 0 )); do
    case "$1" in
      --dry-run)
        INPUT_DRY_RUN=true
        shift
        ;;
      --git-commit)
        [[ "${input_commit_seen}" == false && $# -ge 2 ]] ||
          fail 2 "inputs accepts exactly one valued --git-commit."
        REQUESTED_COMMIT="$2"
        input_commit_seen=true
        shift 2
        ;;
      --storage-root|--storage-root=*|--git-commit=*)
        fail 2 "inputs owns the sibling storage root and requires separate --git-commit syntax."
        ;;
      *) shift ;;
    esac
  done
  [[ -z "${REQUESTED_COMMIT}" ]] || validate_commit "${REQUESTED_COMMIT}"
  resolve_host_layout
  admit_shared_source
  admit_repository_file "${input_config}" "input-generation campaign config"
  PARTITION=standard
  slurm_python_cli cli generate-input-cases "${ADMITTED_HOST_PATH}" "${input_arguments[@]}" \
    --git-commit "${SOURCE_COMMIT}" --storage-root "${HOST_STORAGE_ROOT}"
  exit $?
fi

SUBCOMMAND="$1"
shift
CPU_HOST="shared-filesystem"
REQUESTED_COMMIT=""
KEEP_CPU_SOURCE=true
FORCE_CANCEL=false
DRY_RUN=false
PREFLIGHT_ONLY=false
SMOKE_PARTITION=gpu
SMOKE_PARTITION_GIVEN=false
ALLOW_INCOMPLETE_PLAN=false
REPLACEMENT_POOL_SIZE=""
PARENT_RUN_ID=""
REPLACEMENT_POOL_OPTION_COUNT=0
PARENT_RUN_OPTION_COUNT=0
POSITIONAL=()

while (( $# > 0 )); do
  case "$1" in
    --git-commit)
      (( $# >= 2 )) || fail 2 "--git-commit requires a value."
      REQUESTED_COMMIT="$2"
      shift 2
      ;;
    --replacement-pool-size)
      (( REPLACEMENT_POOL_OPTION_COUNT == 0 )) ||
        fail 2 "Specify --replacement-pool-size at most once."
      (( $# >= 2 )) || fail 2 "--replacement-pool-size requires a value."
      REPLACEMENT_POOL_OPTION_COUNT=1
      REPLACEMENT_POOL_SIZE="$2"
      shift 2
      ;;
    --parent-run-id)
      (( PARENT_RUN_OPTION_COUNT == 0 )) ||
        fail 2 "Specify --parent-run-id at most once."
      (( $# >= 2 )) || fail 2 "--parent-run-id requires a value."
      PARENT_RUN_OPTION_COUNT=1
      PARENT_RUN_ID="$2"
      shift 2
      ;;
    --force) FORCE_CANCEL=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    --preflight-only) PREFLIGHT_ONLY=true; shift ;;
    --partition)
      (( $# >= 2 )) || fail 2 "--partition requires gpu or standard."
      [[ "${SMOKE_PARTITION_GIVEN}" == false ]] || fail 2 "Specify --partition at most once."
      SMOKE_PARTITION="$2"
      SMOKE_PARTITION_GIVEN=true
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    --*) fail 2 "Unsupported option: $1" ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done

[[ -z "${REQUESTED_COMMIT}" ]] || validate_commit "${REQUESTED_COMMIT}"
[[ -z "${REPLACEMENT_POOL_SIZE}" ]] || validate_positive "replacement_pool_size" "${REPLACEMENT_POOL_SIZE}"
[[ -z "${PARENT_RUN_ID}" ]] || validate_run_id "${PARENT_RUN_ID}"
if [[ -n "${PARENT_RUN_ID}" && -z "${REPLACEMENT_POOL_SIZE}" ]]; then
  fail 2 "--parent-run-id requires --replacement-pool-size."
fi
if [[ -n "${REPLACEMENT_POOL_SIZE}" && "${SUBCOMMAND}" != run ]]; then
  fail 2 "Replacement completion options are supported only by run CONFIG."
fi
if [[ "${DRY_RUN}" == true && "${PREFLIGHT_ONLY}" == true ]]; then
  fail 2 "--dry-run cannot be combined with --preflight-only."
fi
if [[ "${SMOKE_PARTITION_GIVEN}" == true && "${SUBCOMMAND}" != smoke ]]; then
  fail 2 "--partition is supported only by smoke."
fi
if [[ "${SUBCOMMAND}" == smoke \
  && "${SMOKE_PARTITION}" != gpu && "${SMOKE_PARTITION}" != standard ]]; then
  fail 2 "Smoke partition must be gpu or standard."
fi

resolve_host_layout
trap 'workflow_exit_handler $?' EXIT
admit_shared_source

case "${SUBCOMMAND}" in
  smoke)
    (( ${#POSITIONAL[@]} == 0 )) || fail 2 "smoke accepts no positional arguments."
    [[ "${FORCE_CANCEL}" == false && "${DRY_RUN}" == false \
      && "${PREFLIGHT_ONLY}" == false ]] ||
      fail 2 "smoke received an unsupported option."
    submit_generation_smoke
    ;;
  run)
    (( ${#POSITIONAL[@]} == 1 )) ||
      fail 2 "run requires exactly one Generation config."
    [[ "${FORCE_CANCEL}" == false ]] ||
      fail 2 "run received an administrative-only option."
    RUN_CONFIG_ARGUMENT="${POSITIONAL[0]}"
    [[ "${DRY_RUN}" != true ]] || ALLOW_INCOMPLETE_PLAN=true
    resolve_generation_run_plan
    if [[ -n "${REPLACEMENT_POOL_SIZE}" ]]; then
      [[ "${RUN_KIND}" == campaign ]] ||
        fail 2 "--replacement-pool-size is supported only for one campaign run plan."
      RUN_LEAF_CONFIG="${RUN_PLAN_CONFIG}"
      resolve_campaign "${RUN_LEAF_CONFIG}"
      if [[ "${DRY_RUN}" == true || "${PREFLIGHT_ONLY}" == true ]]; then
        resolve_completion_parent_local
        attach_completion_plan_metadata
      fi
    fi
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
  status)
    (( ${#POSITIONAL[@]} == 1 )) ||
      fail 2 "status requires exactly one config or run ID."
    [[ "${FORCE_CANCEL}" == false && "${DRY_RUN}" == false \
      && "${PREFLIGHT_ONLY}" == false ]] ||
      fail 2 "status received an unsupported option."
    status_generation_target "${POSITIONAL[0]}"
    ;;
  cancel)
    (( ${#POSITIONAL[@]} == 1 )) ||
      fail 2 "cancel requires exactly one run ID."
    [[ "${DRY_RUN}" == false && "${PREFLIGHT_ONLY}" == false ]] ||
      fail 2 "cancel received an unsupported option."
    RUN_ID="${POSITIONAL[0]}"
    cancel_generation_run
    ;;
  *)
    usage
    fail 2 "Unsupported subcommand: ${SUBCOMMAND}. Start or resume Generation work with: $0 run CONFIG"
    ;;
esac
