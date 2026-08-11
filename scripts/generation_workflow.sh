#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
HOST_REPO_ROOT=""
HOST_STORAGE_ROOT=""
DOCKER_PYTHON=""
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
PILOT_STAGING=""
PILOT_STAGING_BYTES=0
PILOT_STAGING_RECLAIMED=0

usage() {
  cat >&2 <<EOF
Usage:
  $0 setup-cpu [--cpu-host HOST] [--remote-root PATH] [--git-commit COMMIT] [--execute]
  $0 preflight CAMPAIGN [options]
  $0 plan CAMPAIGN [options]
  $0 launch CAMPAIGN [options]
  $0 all CAMPAIGN [--detach] [--keep-cpu-source] [options]
  $0 smoke [options]
  $0 benchmark-cores [--variant VARIANT_ID] [remote options]
  $0 pilot-check CAMPAIGN [--cases-per-material N] [--keep-cpu-source] [options]
  $0 status [CAMPAIGN_RUN_ID] [remote options]
  $0 collect|build-datasets|resume CAMPAIGN_RUN_ID [options]
  $0 cleanup CAMPAIGN_RUN_ID [--confirm] [remote options]
  $0 accounting|cancel|validate CAMPAIGN_RUN_ID [remote options]

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
verified CPU source by default. Use only --keep-cpu-source to retain it.
smoke owns the canonical paired two-profile technical run and always retains CPU source.
benchmark-cores runs the four isolated same-case transient core variants; --variant retries one.
pilot-check runs configured-material transient diagnostics and safely cleans CPU/staging by default.
EOF
}
fail() {
  local status="$1"
  shift
  printf '%s\n' "$*" >&2
  exit "${status}"
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
  HOST_STORAGE_ROOT="${STORAGE_ROOT:-${HOST_REPO_ROOT}/../storage}"
  DOCKER_PYTHON="${HOST_REPO_ROOT}/scripts/docker_python.sh"
}

admit_repository_file() {
  local value="$1"
  local label="$2"
  local candidate lexical resolved relative
  if [[ "${value}" == /* ]]; then
    candidate="${value}"
  else
    validate_logical_path "${label}" "${value}"
    candidate="${HOST_REPO_ROOT}/${value}"
  fi
  lexical="$(realpath -ms -- "${candidate}")" ||
    fail 2 "Could not normalize ${label}."
  resolved="$(realpath -e -- "${candidate}")" ||
    fail 2 "${label} does not exist."
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

resolve_local_commit() {
  local clean="$1"
  require_command git
  local head status
  head="$(git -C "${HOST_REPO_ROOT}" rev-parse HEAD)" || fail 1 "Could not resolve Git HEAD."
  validate_commit "${head}"
  [[ -z "${REQUESTED_COMMIT}" || "${REQUESTED_COMMIT}" == "${head}" ]]     || fail 1 "Requested commit differs from local HEAD."
  REQUESTED_COMMIT="${head}"
  if [[ "${clean}" == true ]]; then
    status="$(git -C "${HOST_REPO_ROOT}" status --porcelain)"
    [[ -z "${status}" ]] || fail 1 "This operation requires a clean local worktree."
  fi
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
  remote_bash "${CPU_HOST}" \
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
}


setup_cpu() {
  resolve_local_commit false
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
source "${venv}/bin/activate"
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
  verify_remote_setup
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
source "${venv}/bin/activate"
export GENERATION_CPU_VENV="${venv}"
export STORAGE_ROOT="${storage}"
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


preflight_cpu() {
  resolve_local_commit true
  resolve_campaign "${CAMPAIGN_ARGUMENT}"
  resolve_configured_resources
  validate_resources
  verify_remote_setup
  print_layout
  local remote_campaign
  remote_campaign="$(remote_repository_path "${CAMPAIGN_RELATIVE_PATH}")"
  remote_bash "${CPU_HOST}" \
    "${REMOTE_REPOSITORY}" "${REMOTE_STORAGE_ROOT}" "${REMOTE_VENV}" \
    "${remote_campaign}" "${ONLY_BATCH}" "${CORES_PER_CASE}" \
    "${PARTITION}" "${WALL_TIME}" "${PYTHON_MODULE}" "${COMSOL_MODULE}" \
    "${PYTHON_EXECUTABLE}" "${COMSOL_EXECUTABLE}" "${SCHEDULER_KIND}" \
    "${REQUESTED_COMMIT}" <<'REMOTE'
set -euo pipefail
repository="$1"; storage="$2"; venv="$3"; campaign="$4"; only_batch="$5"
cores_per_case="$6"; partition="$7"; wall_time="$8"; python_module="$9"
comsol_module="${10}"; python_executable="${11}"; comsol_executable="${12}"
scheduler="${13}"; commit="${14}"
preflight_id="$(date -u +%Y%m%dT%H%M%SZ)"
cd "${repository}"
meta_root="$("${venv}/bin/python" -c 'import sys; from src import common; print(common.paths.get_generation_meta_root(storage_root=sys.argv[1]))' "${storage}")"
logs="${meta_root}/preflight/${preflight_id}"
mkdir -p "${logs}"
[[ -n "${only_batch}" ]] || only_batch=-
printf 'Preflight log root: %s\n' "${logs}"
submission=(sbatch --wait --parsable --partition="${partition}" --nodes=1 --ntasks=1
  --cpus-per-task=1 --job-name=vp2-generation-preflight
  --export="ALL,GENERATION_GIT_COMMIT=${commit}"
  --output="${logs}/slurm-%j.out" --error="${logs}/slurm-%j.err"
  --chdir="${repository}")
[[ -z "${wall_time}" ]] || submission+=(--time="${wall_time}")
set +e
job_id="$("${submission[@]}" "${repository}/scripts/generation_cpu_smoke.sh" \
  "${repository}" "${venv}" "${campaign}" "${storage}" "${only_batch}" \
  "${cores_per_case}" environment-only "${python_module}" "${comsol_module}" \
  "${python_executable}" "${comsol_executable}" "${scheduler}")"
status="$?"
set -e
job_id="${job_id%%;*}"
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
  printf 'CPU login Slurm submission failed: malformed preflight job ID %s.\n' \
    "${job_id}" >&2
  exit 1
fi
printf 'CPU compute-node preflight Slurm job: %s\n' "${job_id}"
[[ ! -f "${logs}/slurm-${job_id}.out" ]] || cat "${logs}/slurm-${job_id}.out"
[[ ! -f "${logs}/slurm-${job_id}.err" ]] || cat "${logs}/slurm-${job_id}.err" >&2
if (( status != 0 )); then
  printf 'CPU compute-node preflight failed in Slurm job %s; logs: %s\n' \
    "${job_id}" "${logs}" >&2
  exit "${status}"
fi
REMOTE
}


mapping_probe_cpu() {
  local campaign_argument="$1"
  resolve_campaign "${campaign_argument}"
  resolve_configured_resources
  validate_resources
  resolve_remote_layout
  local remote_campaign
  remote_campaign="$(remote_repository_path "${CAMPAIGN_RELATIVE_PATH}")"
  remote_bash "${CPU_HOST}" \
    "${REMOTE_REPOSITORY}" "${REMOTE_STORAGE_ROOT}" "${REMOTE_VENV}" \
    "${REQUESTED_COMMIT}" "${remote_campaign}" "${CORES_PER_CASE}" \
    "${WALL_TIME}" "${PARTITION}" "${PYTHON_MODULE}" "${COMSOL_MODULE}" \
    "${PYTHON_EXECUTABLE}" "${COMSOL_EXECUTABLE}" "${SCHEDULER_KIND}" <<'REMOTE'
set -euo pipefail
repository="$1"; storage="$2"; venv="$3"; commit="$4"; campaign="$5"
cores_per_case="$6"; wall_time="$7"; partition="$8"; python_module="$9"
comsol_module="${10}"; python_executable="${11}"; comsol_executable="${12}"
scheduler="${13}"
probe_id="$(date -u +%Y%m%dT%H%M%SZ)-${RANDOM}"
cd "${repository}"
meta_root="$("${venv}/bin/python" -c 'import sys; from src import common; print(common.paths.get_generation_meta_root(storage_root=sys.argv[1]))' "${storage}")"
logs="${meta_root}/mapping_probe_jobs/${probe_id}"
mkdir -p "${logs}"
submission=(sbatch --wait --parsable --partition="${partition}" --nodes=1 --ntasks=1
  --cpus-per-task="${cores_per_case}" --job-name=vp2-mapping-probe
  --export="ALL,GENERATION_GIT_COMMIT=${commit}"
  --output="${logs}/slurm-%j.out" --error="${logs}/slurm-%j.err"
  --chdir="${repository}")
[[ -z "${wall_time}" ]] || submission+=(--time="${wall_time}")
set +e
job_id="$("${submission[@]}" "${repository}/scripts/generation_cpu_smoke.sh" \
  "${repository}" "${venv}" "${campaign}" "${storage}" - \
  "${cores_per_case}" mapping-probe "${python_module}" "${comsol_module}" \
  "${python_executable}" "${comsol_executable}" "${scheduler}")"
status="$?"
set -e
job_id="${job_id%%;*}"
[[ "${job_id}" =~ ^[0-9]+$ ]] || {
  printf 'Mapping-probe submission returned malformed job ID: %s\n' "${job_id}" >&2
  exit 1
}
printf 'Mapping-probe Slurm job: %s\n' "${job_id}"
[[ ! -f "${logs}/slurm-${job_id}.out" ]] || cat "${logs}/slurm-${job_id}.out"
[[ ! -f "${logs}/slurm-${job_id}.err" ]] || cat "${logs}/slurm-${job_id}.err" >&2
exit "${status}"
REMOTE
}


validate_local_launch_gates() {
  local_cli validate-config "${CAMPAIGN_CONFIG_PATH}" >/dev/null
  [[ "${CAMPAIGN_PURPOSE}" == technical_runtime_smoke ]] && return
  resolve_workflow_campaigns
  local_cli static-sentinels "${STATIONARY_PRIMARY_CAMPAIGN_HOST_PATH}" \
    "${TRANSIENT_PRIMARY_CAMPAIGN_HOST_PATH}" >/dev/null ||
    fail 2 "Static scientific sentinels block production planning or launch."
  local_cli validate-real-smoke --storage-root "${LOCAL_STORAGE_ROOT}" >/dev/null ||
    fail 2 "No immutable real runtime-smoke receipt is valid for the current source."
}

technical_profiles_ready() {
  resolve_local_python
  resolve_workflow_campaigns
  local_cli validate-config "${STATIONARY_SMOKE_CAMPAIGN_HOST_PATH}" >/dev/null 2>&1 &&
    local_cli validate-config "${TRANSIENT_SMOKE_CAMPAIGN_HOST_PATH}" >/dev/null 2>&1
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


run_smoke() {
  [[ "${DETACH}" == false && "${CONFIRM_CLEANUP}" == false && -z "${ONLY_BATCH}" ]] ||
    fail 2 "smoke does not support --detach, --confirm, or --only-batch."
  resolve_local_commit true
  resolve_workflow_campaigns
  KEEP_CPU_SOURCE=true
  CAMPAIGN_ARGUMENT="${STATIONARY_SMOKE_CAMPAIGN_PATH}"
  preflight_cpu
  if ! technical_profiles_ready; then
    printf 'Profile mappings remain unconfirmed; running isolated retained probes.\n' >&2
    local probe_failed=false
    mapping_probe_cpu "${STATIONARY_SMOKE_CAMPAIGN_PATH}" || probe_failed=true
    mapping_probe_cpu "${TRANSIENT_SMOKE_CAMPAIGN_PATH}" || probe_failed=true
    if [[ "${probe_failed}" == true ]]; then
      printf 'One or more mapping probes reported an execution or mapping failure.\n' >&2
    fi
    fail 2 "Mapping confirmation is required. Review mapping_probe.json artifacts, update only explicit profile mappings, commit, and rerun this smoke command."
  fi
  CAMPAIGN_ARGUMENT="${STATIONARY_SMOKE_CAMPAIGN_PATH}"
  run_all
  local stationary_run_id="${RUN_ID}"
  CAMPAIGN_ARGUMENT="${TRANSIENT_SMOKE_CAMPAIGN_PATH}"
  run_all
  local transient_run_id="${RUN_ID}"
  local comsol_version receipt
  comsol_version="$(remote_comsol_version)"
  receipt="$(local_cli finalize-real-smoke "${stationary_run_id}" "${transient_run_id}" \
    --comsol-version-output "${comsol_version}" --storage-root "${LOCAL_STORAGE_ROOT}")"
  receipt="$(container_path_to_host "${receipt}")"
  printf 'Real technical runtime-smoke receipt: %s\n' "${receipt}"
  printf 'CPU sources retained for review for runs %s and %s.\n' \
    "${stationary_run_id}" "${transient_run_id}"
}


plan_campaign() {
  resolve_local_commit true
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

launch_campaign() {
  resolve_local_commit true
  resolve_campaign "${CAMPAIGN_ARGUMENT}"
  resolve_configured_resources
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
    "${REMOTE_VENV}" "${PYTHON_MODULE}" "$@" <<'REMOTE'
set -euo pipefail
repository="$1"; storage="$2"; venv="$3"; python_module="$4"
shift 4
module load "${python_module}"
source "${venv}/bin/activate"
export GENERATION_CPU_VENV="${venv}"
export STORAGE_ROOT="${storage}"
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
  env STORAGE_ROOT="${LOCAL_STORAGE_ROOT:-${HOST_STORAGE_ROOT}}" "${DOCKER_PYTHON}" "$@"
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
      campaign) directories+=("${field4}") ;;
      batch) directories+=("${field5}" "${field6}" "${field7}") ;;
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
    PILOT_STAGING="${staging}"
  fi
  receipt="$(local_cli publish-transferred-campaign "${RUN_ID}" \
    --staging-root "${staging}" --destination-root "${LOCAL_STORAGE_ROOT}" \
    --source-host "${CPU_HOST}" --source-storage-root "${REMOTE_STORAGE_ROOT}")" ||
    fail 1 "GPU publication validation failed; staging retained at ${staging}."
  if [[ -z "${PILOT_CASES_PER_MATERIAL}" ]]; then
    local_cli cleanup-transfer-staging --campaign-run-id "${RUN_ID}" \
      --directory "${staging}" --storage-root "${LOCAL_STORAGE_ROOT}" --confirm >/dev/null
  else
    printf 'Pilot transfer staging retained through analysis: %s\n' "${staging}"
  fi
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
  validate_positive "configured poll_interval_seconds" "${STATUS_POLL_SECONDS}"
  while true; do
    read_remote_source_status
    if [[ "${REMOTE_SOURCE_STATE}" == source_cleanup_complete ]]; then
      printf 'CPU source cleanup receipt already exists for %s.\n' "${RUN_ID}"
      return
    fi
    if [[ "${ALLOW_REMOTE_RESUME}" == true ]]; then
      remote_cli resume-campaign "${RUN_ID}" \
        --storage-root "${REMOTE_STORAGE_ROOT}" >/dev/null
      ALLOW_REMOTE_RESUME=false
    else
      remote_cli feed-campaign "${RUN_ID}" \
        --storage-root "${REMOTE_STORAGE_ROOT}" >/dev/null
    fi
    local state
    state="$(remote_campaign_state)"
    printf 'Campaign %s state: %s\n' "${RUN_ID}" "${state}"
    case "${state}" in
      publication_complete)
        remote_cli validate-campaign-terminal "${RUN_ID}" \
          --storage-root "${REMOTE_STORAGE_ROOT}" >/dev/null
        return
        ;;
      running|submitted|feeding|submission_pending_or_unknown)
        sleep "${STATUS_POLL_SECONDS}"
        ;;
      completed)
        remote_cli validate-campaign-terminal "${RUN_ID}" \
          --storage-root "${REMOTE_STORAGE_ROOT}" >/dev/null
        return
        ;;
      failed|partially_failed|cancelled)
        fail 1 "Campaign requires explicit resume from state ${state}."
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
  CPU_BYTES_RECLAIMED="${reclaimed}"
  CPU_CLEANUP_RECEIPT_SHA="${receipt_sha}"
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
  if [[ -n "${PILOT_CASES_PER_MATERIAL}" ]]; then
    local cleanup_state=cleanup_not_authorized
    if (( CPU_BYTES_RECLAIMED > 0 || PILOT_STAGING_RECLAIMED > 0 )); then
      cleanup_state=cleanup_incomplete
    fi
    printf 'Pilot check failed.\nStage: %s\nCleanup: %s\nCPU bytes retained: %s\nResume command: %s\n' \
      "${ALL_STAGE}" "${cleanup_state}" "${CPU_BYTES_RETAINED}" "${resume}" >&2
  else
    printf 'All workflow failed.\nStage: %s\nCPU bytes retained: %s\nResume command: %s\n' \
      "${ALL_STAGE}" "${CPU_BYTES_RETAINED}" "${resume}" >&2
  fi
  return "${status}"
}

workflow_exit_handler() {
  local status="$1"
  if [[ "${ALL_WORKFLOW_ACTIVE}" == true && "${status}" -ne 0 ]]; then
    workflow_failure_report "${status}" || true
  fi
}

read_remote_campaign_identity() {
  resolve_local_python
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
  if local_cli validate-pilot-check "${RUN_ID}" --require-cleanup-complete \
    --storage-root "${LOCAL_STORAGE_ROOT}" >/dev/null 2>&1; then
    local_cli validate-pilot-check "${RUN_ID}" --format summary \
      --storage-root "${LOCAL_STORAGE_ROOT}"
    ALL_WORKFLOW_ACTIVE=false
    trap - EXIT
    return
  fi
  ALL_STAGE="remote pilot terminal monitoring"
  wait_for_terminal_publication
  ALL_STAGE="pilot source inventory, transfer, and hash validation"
  if ! gpu_publication_is_valid; then
    [[ "${REMOTE_SOURCE_STATE}" != source_cleanup_complete ]] ||
      fail 1 "CPU source is cleaned but no valid GPU pilot publication exists."
    collect_campaign
  else
    printf 'GPU pilot publication validated and reused for %s.\n' "${RUN_ID}"
  fi
  ALL_STAGE="canonical HDF5 and physical/runtime pilot analysis"
  prepare_pilot_check_receipt
  ALL_STAGE="empty production-dataset gate bound to pilot evidence"
  build_datasets
  ALL_STAGE="terminal pre-cleanup workflow receipt"
  prepare_all_receipt
  read_remote_source_status
  [[ "${REMOTE_SOURCE_ACTIVE}" == False ]] ||
    fail 1 "cleanup_not_authorized: an active Slurm attempt still owns the CPU pilot source."
  if [[ "${KEEP_CPU_SOURCE}" == true ]]; then
    printf 'CPU pilot source retained by --keep-cpu-source.\n'
    CPU_BYTES_RECLAIMED=0
    CPU_CLEANUP_RECEIPT_SHA=""
  else
    ALL_STAGE="verified CPU pilot source cleanup"
    confirm_cpu_cleanup
  fi
  ALL_STAGE="verified pilot transfer-staging cleanup"
  cleanup_pilot_staging
  ALL_STAGE="canonical final pilot cleanup receipt"
  record_pilot_cleanup_result
  ALL_STAGE="terminal pilot and workflow receipt validation"
  local_cli validate-all-workflow "${RUN_ID}" --storage-root "${LOCAL_STORAGE_ROOT}" >/dev/null
  local_cli validate-pilot-check "${RUN_ID}" --require-cleanup-complete --format summary \
    --storage-root "${LOCAL_STORAGE_ROOT}"
  ALL_WORKFLOW_ACTIVE=false
  trap - EXIT
}

run_pilot_check() {
  [[ "${DETACH}" == false && "${CONFIRM_CLEANUP}" == false && -z "${ONLY_BATCH}" ]] ||
    fail 2 "pilot-check does not support --detach, --confirm, or --only-batch."
  ALL_WORKFLOW_ACTIVE=true
  trap 'workflow_exit_handler $?' EXIT
  ALL_STAGE="local exact-commit and campaign validation"
  resolve_local_commit true
  resolve_campaign "${CAMPAIGN_ARGUMENT}"
  resolve_pilot_contract
  resolve_local_storage
  resolve_configured_resources
  validate_resources
  ALL_STAGE="CPU setup and readiness"
  EXECUTE_SETUP=true
  setup_cpu
  preflight_cpu
  ALL_STAGE="profile mapping validation"
  if ! local_cli validate-config "${CAMPAIGN_CONFIG_PATH}" >/dev/null 2>&1; then
    printf 'Transient profile mappings remain unconfirmed; running one retained probe.\n' >&2
    mapping_probe_cpu "${CAMPAIGN_CONFIG_PATH}" || true
    fail 2 "Mapping confirmation is required. Review mapping_probe.json, update only explicit profile mappings, commit, and rerun pilot-check."
  fi
  ALL_STAGE="configured-material static scientific sentinels"
  resolve_workflow_campaigns
  local_cli static-sentinels "${STATIONARY_PRIMARY_CAMPAIGN_HOST_PATH}" \
    "${TRANSIENT_PRIMARY_CAMPAIGN_HOST_PATH}" ||
    fail 2 "Static scientific sentinels block pilot launch; inspect the sentinel report before rerunning."
  print_layout
  printf 'Pilot cases: %s materials x %s = %s total.\n' \
    "${PILOT_MATERIAL_COUNT}" "${PILOT_CASES_PER_MATERIAL}" "${PILOT_TOTAL_CASES}"
  ALL_STAGE="resolved pilot campaign plan"
  remote_plan_submit plan-campaign
  ALL_STAGE="pilot campaign launch"
  launch_campaign
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
  resolve_configured_resources
  validate_resources
  resolve_remote_layout
  resolve_local_storage
  resolve_local_python
  validate_local_launch_gates
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
  read_remote_campaign_identity
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

resolve_runtime_smoke_receipt() {
  local report record kind container_path extra
  report="$(local_cli validate-real-smoke --storage-root "${LOCAL_STORAGE_ROOT}")" ||
    fail 2 "No immutable real runtime-smoke receipt is valid for the current source."
  record="$(printf '%s\n' "${report}" | local_python -c 'import json, sys
value = json.load(sys.stdin)
valid = value["valid_receipts"]
if not valid:
    raise SystemExit("no current runtime-smoke receipt")
record = valid[0]
fields = (record["path"], record["receipt_digest"])
if any("\t" in item or "\n" in item or "\r" in item for item in fields):
    raise SystemExit("runtime-smoke receipt contains unsafe shell transport text")
print("\t".join(("smoke", *fields)))')" ||
    fail 2 "Could not resolve the native runtime-smoke receipt."
  IFS=$'\t' read -r kind container_path RUNTIME_SMOKE_DIGEST extra <<< "${record}"
  [[ "${kind}" == smoke && -z "${extra:-}" ]] ||
    fail 1 "Malformed runtime-smoke receipt record."
  validate_digest "${RUNTIME_SMOKE_DIGEST}"
  RUNTIME_SMOKE_RECEIPT="$(container_path_to_host "${container_path}")"
  [[ "${RUNTIME_SMOKE_RECEIPT}" == "${LOCAL_STORAGE_ROOT}/"* \
    && -f "${RUNTIME_SMOKE_RECEIPT}" && ! -L "${RUNTIME_SMOKE_RECEIPT}" ]] ||
    fail 1 "Runtime-smoke receipt is outside canonical local storage."
  RUNTIME_SMOKE_RELATIVE="${RUNTIME_SMOKE_RECEIPT#"${LOCAL_STORAGE_ROOT}/"}"
  validate_transfer_path "${RUNTIME_SMOKE_RELATIVE}"
}

sync_runtime_smoke_receipt() {
  require_command rsync "runtime-smoke receipt transfer"
  local destination="${REMOTE_STORAGE_ROOT}/${RUNTIME_SMOKE_RELATIVE}"
  local temporary="${destination}.incoming.$$"
  remote_bash "${CPU_HOST}" "$(dirname "${destination}")" <<'REMOTE'
set -euo pipefail
directory="$1"
mkdir -p "${directory}"
REMOTE
  rsync -a --protect-args "${RUNTIME_SMOKE_RECEIPT}" "${CPU_HOST}:${temporary}" ||
    fail 1 "Could not transfer the compact runtime-smoke receipt to the CPU host."
  remote_bash "${CPU_HOST}" "${destination}" "${temporary}" <<'REMOTE'
set -euo pipefail
destination="$1"; temporary="$2"
[[ -f "${temporary}" && ! -L "${temporary}" ]]
if [[ -e "${destination}" ]]; then
  [[ -f "${destination}" && ! -L "${destination}" ]]
  if ! cmp -s "${temporary}" "${destination}"; then
    rm -f -- "${temporary}"
    printf 'Existing CPU runtime-smoke receipt conflicts: %s\n' "${destination}" >&2
    exit 1
  fi
  rm -f -- "${temporary}"
else
  mv -- "${temporary}" "${destination}"
fi
REMOTE
  remote_cli validate-real-smoke --storage-root "${REMOTE_STORAGE_ROOT}" >/dev/null ||
    fail 2 "CPU-side runtime-smoke evidence is missing, stale, or incomplete."
}

remote_benchmark_plan_submit() {
  local operation="$1"
  local remote_suite
  remote_suite="$(remote_repository_path "${BENCHMARK_SUITE_RELATIVE_PATH}")"
  local -a arguments=(
    "${operation}" "${remote_suite}"
    --git-commit "${REQUESTED_COMMIT}"
    --storage-root "${REMOTE_STORAGE_ROOT}"
  )
  [[ -z "${BENCHMARK_VARIANT}" ]] || arguments+=(--variant "${BENCHMARK_VARIANT}")
  remote_cli "${arguments[@]}"
}

wait_for_core_benchmark() {
  validate_positive "configured benchmark poll_interval_seconds" "${STATUS_POLL_SECONDS}"
  while true; do
    local state
    state="$(remote_cli core-benchmark-status "${RUN_ID}" --format state \
      --storage-root "${REMOTE_STORAGE_ROOT}")"
    printf 'Core benchmark %s state: %s\n' "${RUN_ID}" "${state}"
    case "${state}" in
      complete|retry_required)
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
      running|scheduler_unknown)
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
  [[ "${DETACH}" == false && "${CONFIRM_CLEANUP}" == false \
    && "${KEEP_CPU_SOURCE}" == false && -z "${ONLY_BATCH}" ]] ||
    fail 2 "benchmark-cores accepts only --variant and remote options."
  resolve_local_commit true
  resolve_local_storage
  resolve_local_python
  resolve_benchmark_contract
  resolve_runtime_smoke_receipt
  resolve_remote_layout
  EXECUTE_SETUP=true
  setup_cpu
  sync_runtime_smoke_receipt
  printf 'Resolved core benchmark plan:\n'
  remote_benchmark_plan_submit plan-core-benchmark
  local output
  output="$(remote_benchmark_plan_submit submit-core-benchmark)" ||
    fail 1 "Remote core benchmark submission failed."
  printf '%s\n' "${output}"
  if [[ ${output} =~ \"benchmark_run_id\"[[:space:]]*:[[:space:]]*\"(core_scaling_transient__[0-9a-f]{16})\" ]]; then
    RUN_ID="${BASH_REMATCH[1]}"
    validate_benchmark_run_id "${RUN_ID}"
    printf 'Core benchmark run ID: %s\n' "${RUN_ID}"
  else
    fail 1 "Core benchmark submission returned no run ID."
  fi
  wait_for_core_benchmark
  case "${BENCHMARK_TERMINAL_STATE}" in
    complete)
      remote_cli finalize-core-benchmark "${RUN_ID}" \
        --storage-root "${REMOTE_STORAGE_ROOT}" >/dev/null
      collect_core_benchmark
      ;;
    retry_required)
      remote_cli core-benchmark-status "${RUN_ID}" \
        --storage-root "${REMOTE_STORAGE_ROOT}"
      fail 1 "One benchmark repetition requires retry. Use its reported variant_id with: ./scripts/generation_workflow.sh benchmark-cores --variant VARIANT_ID"
      ;;
    incomplete)
      fail 1 "The selected benchmark subset finished but the four-variant suite is incomplete. Run benchmark-cores or retry another --variant."
      ;;
  esac
}

(( $# > 0 )) || { usage; exit 2; }
[[ "$1" != -h && "$1" != --help ]] || { usage; exit 0; }
SUBCOMMAND="$1"
shift
CPU_HOST="${GENERATION_CPU_HOST:-}"
REMOTE_ROOT=""
REQUESTED_COMMIT=""
EXECUTE_SETUP=false
CONFIRM_CLEANUP=false
DETACH=false
KEEP_CPU_SOURCE=false
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
    --detach) DETACH=true; shift ;;
    --keep-cpu-source) KEEP_CPU_SOURCE=true; shift ;;
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
if [[ "${SKIP_EXTREME_FAMILY_OOD}" == true ]]; then
  [[ "${SUBCOMMAND}" =~ ^(plan|launch|all)$ ]] ||
    fail 2 "--skip-extreme-family-ood is supported only by plan, launch, and all."
  [[ -z "${ONLY_BATCH}" ]] ||
    fail 2 "--skip-extreme-family-ood cannot be combined with --only-batch."
fi

resolve_host_layout

case "${SUBCOMMAND}" in
  setup-cpu)
    (( ${#POSITIONAL[@]} == 0 )) || fail 2 "setup-cpu accepts no positional arguments."
    [[ "${DETACH}" == false && "${KEEP_CPU_SOURCE}" == false && "${CONFIRM_CLEANUP}" == false ]]       || fail 2 "Unsupported setup-cpu option."
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
  smoke)
    (( ${#POSITIONAL[@]} == 0 )) || fail 2 "smoke accepts no campaign positional argument."
    run_smoke
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
