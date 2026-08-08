#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
HOST_STORAGE_ROOT="${STORAGE_ROOT:-${PROJECT_DIR}/../storage}"
IMAGE_NAME="grainlegumes-pino-drying"
REPOSITORY_URL="https://github.com/Rinovative/grainlegumes-pino-drying.git"
DEFAULT_CPU_HOST="sricehpc01"
PYTHON_MODULE="Python/3.10"
COMSOL_MODULE="Comsol/v6.4"
GENERATION_MODULE="src.generation.cli.cli_generation"

usage() {
  cat >&2 <<EOF
Usage:
  $0 setup-cpu [--cpu-host HOST] [--remote-root PATH] [--git-commit COMMIT] [--execute]
  $0 launch CAMPAIGN --max-nodes N --cases-per-node N --cores-per-case N --max-parallel-cases N [options]
  $0 status CAMPAIGN_RUN_ID [--cpu-host HOST] [--remote-root PATH]
  $0 collect CAMPAIGN_RUN_ID [--cpu-host HOST] [--remote-root PATH] [--build-datasets]
  $0 build-datasets CAMPAIGN_RUN_ID
  $0 all CAMPAIGN --max-nodes N --cases-per-node N --cores-per-case N --max-parallel-cases N [--wait] [--build-datasets] [options]

Shared remote options:
  --cpu-host HOST       CPU login host (default: ${DEFAULT_CPU_HOST})
  --remote-root PATH    Exact remote workflow root (default: remote HOME/grainlegumes-generation)

Launch execution-only options:
  --only-batch NAME
  --wall-time SLURM_TIME
  --max-nodes N
  --cases-per-node N
  --cores-per-case N
  --max-parallel-cases N

setup-cpu is a dry run unless --execute is supplied. launch returns immediately.
all returns after launch unless --wait is supplied; Slurm work remains recoverable.
EOF
}

fail() {
  local status="$1"
  shift
  printf '%s\n' "$*" >&2
  exit "${status}"
}

require_command() {
  local name="$1"
  command -v "${name}" >/dev/null 2>&1 || fail 1 "Required command was not found: ${name}"
}

validate_host() {
  local value="$1"
  [[ "${value}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
    || fail 2 "CPU host is unsafe: ${value@Q}"
}

validate_remote_path() {
  local value="$1"
  [[ "${value}" =~ ^/[A-Za-z0-9._/+@-]+$ ]] \
    || fail 2 "Remote path must be one absolute shell-safe path: ${value@Q}"
  [[ "${value}" != "/" && "${value}" != *"//"* && "${value}" != *"/../"* \
      && "${value}" != */.. && "${value}" != *"/./"* && "${value}" != */. ]] \
    || fail 2 "Remote path contains an unsafe component: ${value@Q}"
}

validate_commit() {
  local value="$1"
  [[ "${value}" =~ ^[0-9a-f]{40}$ ]] \
    || fail 2 "Git commit must be one exact 40-character lowercase object identifier."
}

validate_run_id() {
  local value="$1"
  [[ "${value}" =~ ^[A-Za-z0-9._-]+__[0-9a-f]{16}$ ]] \
    || fail 2 "Campaign-run ID is malformed: ${value@Q}"
}

validate_positive_integer() {
  local label="$1"
  local value="$2"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || fail 2 "${label} must be an integer >= 1."
}

validate_batch_name() {
  local value="$1"
  [[ "${value}" =~ ^[a-z0-9_]+__[a-z0-9_]+__(natural|parameter_ood)$ ]] \
    || fail 2 "--only-batch must be one canonical predeclared batch name."
}

validate_wall_time() {
  local value="$1"
  [[ "${value}" =~ ^([0-9]+-)?[0-9]{1,2}:[0-9]{2}:[0-9]{2}$ ]] \
    || fail 2 "--wall-time must use Slurm [days-]hours:minutes:seconds syntax."
}

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

remote_bash() {
  local host="$1"
  shift
  ssh -o BatchMode=yes -- "${host}" bash -l -s -- "$@"
}

read_remote_home() {
  local host="$1"
  remote_bash "${host}" <<'REMOTE'
set -euo pipefail
printf '%s\n' "${HOME}"
REMOTE
}

resolve_remote_layout() {
  validate_host "${CPU_HOST}"
  if [[ -z "${REMOTE_ROOT}" ]]; then
    REMOTE_HOME="$(read_remote_home "${CPU_HOST}")" \
      || fail 1 "Could not derive remote HOME from ${CPU_HOST}."
    validate_remote_path "${REMOTE_HOME}"
    REMOTE_ROOT="${REMOTE_HOME}/grainlegumes-generation"
  fi
  validate_remote_path "${REMOTE_ROOT}"
  REMOTE_REPOSITORY="${REMOTE_ROOT}/repo"
  REMOTE_STORAGE_ROOT="${REMOTE_ROOT}/storage"
  REMOTE_VENV="${REMOTE_ROOT}/venv"
  validate_remote_path "${REMOTE_REPOSITORY}"
  validate_remote_path "${REMOTE_STORAGE_ROOT}"
  validate_remote_path "${REMOTE_VENV}"
}

resolve_local_commit() {
  local require_clean="$1"
  require_command git
  local head
  head="$(git -C "${PROJECT_DIR}" rev-parse HEAD)" \
    || fail 1 "Could not resolve the local Git commit."
  validate_commit "${head}"
  if [[ -n "${REQUESTED_COMMIT}" && "${REQUESTED_COMMIT}" != "${head}" ]]; then
    fail 1 "Requested commit ${REQUESTED_COMMIT} is not the exact local HEAD ${head}."
  fi
  REQUESTED_COMMIT="${head}"
  if [[ "${require_clean}" == true ]]; then
    local status
    status="$(git -C "${PROJECT_DIR}" status --porcelain)"
    [[ -z "${status}" ]] \
      || fail 1 "Launch requires a clean local worktree; commit or otherwise resolve every local change first."
  fi
}

verify_commit_on_remote() {
  git -C "${PROJECT_DIR}" remote get-url origin >/dev/null \
    || fail 1 "The local repository has no configured origin remote."
  git -C "${PROJECT_DIR}" fetch --quiet origin \
    || fail 1 "Could not fetch the configured origin remote."
  local containing_refs
  containing_refs="$(git -C "${PROJECT_DIR}" for-each-ref \
    --contains "${REQUESTED_COMMIT}" --format='%(refname)' \
    refs/remotes/origin refs/tags)"
  [[ -n "${containing_refs}" ]] \
    || fail 1 "Exact local commit ${REQUESTED_COMMIT} is not reachable from a fetched origin ref."
}

resolve_campaign_config() {
  local requested="$1"
  local resolved
  require_command realpath
  resolved="$(realpath -e -- "${requested}")" \
    || fail 2 "Campaign config does not exist: ${requested}"
  [[ -f "${resolved}" && ! -L "${resolved}" ]] \
    || fail 2 "Campaign config must be one non-symlink regular file: ${requested}"
  if [[ "${resolved}" != "${PROJECT_DIR}/"* ]]; then
    fail 2 "Campaign config must remain inside ${PROJECT_DIR}."
  fi
  CAMPAIGN_RELATIVE_PATH="${resolved#"${PROJECT_DIR}/"}"
  [[ "${CAMPAIGN_RELATIVE_PATH}" =~ ^[A-Za-z0-9._/-]+$ \
      && "${CAMPAIGN_RELATIVE_PATH}" != *"/../"* \
      && "${CAMPAIGN_RELATIVE_PATH}" != ../* ]] \
    || fail 2 "Campaign config has an unsafe repository-relative path."
  CAMPAIGN_CONFIG_PATH="${resolved}"
  CONTAINER_CAMPAIGN_CONFIG="/workspace/repo/${CAMPAIGN_RELATIVE_PATH}"
}

docker_ready() {
  require_command docker
  docker info >/dev/null 2>&1 || fail 1 "The Docker daemon is unavailable on the GPU host."
  docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1 \
    || fail 1 "Maintained image '${IMAGE_NAME}' is missing; build it with ./scripts/docker_build.sh."
}

docker_validate_campaign() {
  local arguments=(validate-config "${CONTAINER_CAMPAIGN_CONFIG}")
  if [[ -n "${ONLY_BATCH}" ]]; then
    arguments+=(--only-batch "${ONLY_BATCH}")
  fi
  docker_ready
  docker run --rm \
    --network none \
    --workdir /workspace/repo \
    --tmpfs /tmp:rw,nosuid,nodev,size=128m \
    --tmpfs /workspace/storage:rw,nosuid,nodev,size=64m \
    -e HOME=/tmp \
    -e PROJECT_ROOT=/workspace/repo \
    -e STORAGE_ROOT=/workspace/storage \
    -v "${PROJECT_DIR}:/workspace/repo:ro" \
    "${IMAGE_NAME}" \
    python -m "${GENERATION_MODULE}" "${arguments[@]}"
}

docker_validate_transfer() {
  local storage="$1"
  local batch_name="$2"
  docker_ready
  docker run --rm \
    --network none \
    --workdir /workspace/repo \
    --tmpfs /tmp:rw,nosuid,nodev,size=128m \
    -e HOME=/tmp \
    -e PROJECT_ROOT=/workspace/repo \
    -e STORAGE_ROOT=/workspace/storage \
    -v "${PROJECT_DIR}:/workspace/repo:ro" \
    -v "${storage}:/workspace/storage:ro" \
    "${IMAGE_NAME}" \
    python -m "${GENERATION_MODULE}" validate-transfer \
      "${CONTAINER_CAMPAIGN_CONFIG}" --only-batch "${batch_name}" \
      --storage-root /workspace/storage >/dev/null
}

print_layout() {
  printf 'Remote host: %s\n' "${CPU_HOST}"
  printf 'Remote project root: %s\n' "${REMOTE_ROOT}"
  printf 'Remote repository path: %s\n' "${REMOTE_REPOSITORY}"
  printf 'Remote storage root: %s\n' "${REMOTE_STORAGE_ROOT}"
  printf 'Remote venv path: %s\n' "${REMOTE_VENV}"
  printf 'Requested Git commit: %s\n' "${REQUESTED_COMMIT}"
  printf 'Python module: %s\n' "${PYTHON_MODULE}"
  printf 'COMSOL module: %s\n' "${COMSOL_MODULE}"
}

setup_cpu() {
  resolve_local_commit false
  resolve_remote_layout
  print_layout
  printf 'Mode: %s\n' "$([[ "${EXECUTE_SETUP}" == true ]] && printf execute || printf dry-run)"
  printf 'Remote commands:\n'
  print_command mkdir -p "${REMOTE_ROOT}" "${REMOTE_STORAGE_ROOT}"
  printf '  if repository is absent:\n'
  print_command git clone --no-checkout "${REPOSITORY_URL}" "${REMOTE_REPOSITORY}"
  printf '  if repository exists, require a clean checkout and exact HTTPS origin.\n'
  print_command git -C "${REMOTE_REPOSITORY}" fetch origin "${REQUESTED_COMMIT}"
  print_command git -C "${REMOTE_REPOSITORY}" checkout --detach "${REQUESTED_COMMIT}"
  print_command module load "${PYTHON_MODULE}"
  print_command python3 -m venv "${REMOTE_VENV}"
  print_command "${REMOTE_VENV}/bin/python" -m pip install -e "${REMOTE_REPOSITORY}[generation-cpu]"
  print_command module load "${COMSOL_MODULE}"
  print_command command -v comsol sbatch squeue sacct git rsync
  print_command "${REMOTE_VENV}/bin/python" -m "${GENERATION_MODULE}" --help
  print_command sbatch --wait --parsable --partition=standard --nodes=1 --ntasks=1 \
    --cpus-per-task=1 --time=00:05:00 --job-name=vp2-generation-smoke \
    --output=/dev/null --error=/dev/null \
    "${REMOTE_REPOSITORY}/scripts/generation_cpu_smoke.sh" "${REMOTE_VENV}"
  if [[ "${EXECUTE_SETUP}" != true ]]; then
    printf 'Dry run only: no remote directory, checkout, venv, package, or job was created.\n'
    return 0
  fi

  remote_bash "${CPU_HOST}" \
    "${REMOTE_ROOT}" "${REMOTE_REPOSITORY}" "${REMOTE_STORAGE_ROOT}" \
    "${REMOTE_VENV}" "${REQUESTED_COMMIT}" "${REPOSITORY_URL}" \
    "${PYTHON_MODULE}" "${COMSOL_MODULE}" <<'REMOTE'
set -euo pipefail
root="$1"
repository="$2"
storage="$3"
venv="$4"
commit="$5"
repository_url="$6"
python_module="$7"
comsol_module="$8"
mkdir -p "${root}" "${storage}"
if [[ ! -e "${repository}" ]]; then
  git clone --no-checkout "${repository_url}" "${repository}"
elif [[ ! -d "${repository}/.git" ]]; then
  printf 'Existing repository path is not the approved Git checkout: %s\n' "${repository}" >&2
  exit 1
else
  [[ -z "$(git -C "${repository}" status --porcelain)" ]] || {
    printf 'Existing CPU checkout is dirty: %s\n' "${repository}" >&2
    exit 1
  }
  [[ "$(git -C "${repository}" remote get-url origin)" == "${repository_url}" ]] || {
    printf 'Existing CPU checkout origin is not the approved HTTPS repository.\n' >&2
    exit 1
  }
fi
git -C "${repository}" fetch origin "${commit}"
git -C "${repository}" cat-file -e "${commit}^{commit}"
git -C "${repository}" checkout --detach "${commit}"
[[ "$(git -C "${repository}" rev-parse HEAD)" == "${commit}" ]]
module load "${python_module}"
if [[ ! -x "${venv}/bin/python" ]]; then
  python3 -m venv "${venv}"
fi
source "${venv}/bin/activate"
python -m pip install -e "${repository}[generation-cpu]"
module load "${comsol_module}"
command -v comsol >/dev/null
command -v sbatch >/dev/null
command -v squeue >/dev/null
command -v sacct >/dev/null
command -v git >/dev/null
command -v rsync >/dev/null
cd "${repository}"
python -m src.generation.cli.cli_generation --help >/dev/null
smoke_job_id="$(sbatch --wait --parsable --partition=standard --nodes=1 --ntasks=1 \
  --cpus-per-task=1 --time=00:05:00 --job-name=vp2-generation-smoke \
  --output=/dev/null --error=/dev/null \
  "${repository}/scripts/generation_cpu_smoke.sh" "${venv}")"
printf 'Compute-node smoke job: %s\n' "${smoke_job_id}"
printf 'CPU setup verified.\nRepository: %s\nStorage: %s\nVenv: %s\nCommit: %s\n' \
  "${repository}" "${storage}" "${venv}" "${commit}"
REMOTE
}

verify_remote_setup() {
  resolve_remote_layout
  remote_bash "${CPU_HOST}" \
    "${REMOTE_REPOSITORY}" "${REMOTE_STORAGE_ROOT}" "${REMOTE_VENV}" \
    "${REQUESTED_COMMIT}" "${REPOSITORY_URL}" "${PYTHON_MODULE}" "${COMSOL_MODULE}" <<'REMOTE'
set -euo pipefail
repository="$1"
storage="$2"
venv="$3"
commit="$4"
repository_url="$5"
python_module="$6"
comsol_module="$7"
[[ -d "${repository}/.git" && -d "${storage}" && -x "${venv}/bin/python" ]]
[[ -z "$(git -C "${repository}" status --porcelain)" ]]
[[ "$(git -C "${repository}" rev-parse HEAD)" == "${commit}" ]]
[[ "$(git -C "${repository}" remote get-url origin)" == "${repository_url}" ]]
module load "${python_module}"
module load "${comsol_module}"
command -v comsol >/dev/null
command -v sbatch >/dev/null
command -v squeue >/dev/null
command -v sacct >/dev/null
command -v rsync >/dev/null
cd "${repository}"
"${venv}/bin/python" -m src.generation.cli.cli_generation --help >/dev/null
printf 'CPU setup matches exact commit %s.\n' "${commit}"
REMOTE
}

launch_campaign() {
  resolve_local_commit true
  verify_commit_on_remote
  resolve_campaign_config "${CAMPAIGN_ARGUMENT}"
  validate_positive_integer --max-nodes "${MAX_NODES}"
  validate_positive_integer --cases-per-node "${CASES_PER_NODE}"
  validate_positive_integer --cores-per-case "${CORES_PER_CASE}"
  validate_positive_integer --max-parallel-cases "${MAX_PARALLEL_CASES}"
  (( CASES_PER_NODE * CORES_PER_CASE <= 32 )) \
    || fail 2 "cases_per_node * cores_per_case must not exceed 32."
  (( MAX_PARALLEL_CASES <= MAX_NODES * CASES_PER_NODE )) \
    || fail 2 "max_parallel_cases must not exceed max_nodes * cases_per_node."
  [[ -z "${ONLY_BATCH}" ]] || validate_batch_name "${ONLY_BATCH}"
  [[ -z "${WALL_TIME}" ]] || validate_wall_time "${WALL_TIME}"

  printf 'Validating campaign and exact predeclared batch plan in the maintained container.\n'
  local plan_output
  plan_output="$(docker_validate_campaign)" \
    || fail 1 "Campaign validation failed; unresolved science remains non-executable."
  printf 'Planned generation batches:\n%s\n' "${plan_output}"
  verify_remote_setup

  local launch_output
  launch_output="$(remote_bash "${CPU_HOST}" \
    "${REMOTE_REPOSITORY}" "${REMOTE_STORAGE_ROOT}" "${REMOTE_VENV}" \
    "${REQUESTED_COMMIT}" "${CAMPAIGN_RELATIVE_PATH}" \
    "${MAX_NODES}" "${CASES_PER_NODE}" "${CORES_PER_CASE}" \
    "${MAX_PARALLEL_CASES}" "${ONLY_BATCH}" "${WALL_TIME}" \
    "${PYTHON_MODULE}" "${COMSOL_MODULE}" <<'REMOTE'
set -euo pipefail
repository="$1"
storage="$2"
venv="$3"
commit="$4"
campaign_relative="$5"
max_nodes="$6"
cases_per_node="$7"
cores_per_case="$8"
max_parallel_cases="$9"
only_batch="${10}"
wall_time="${11}"
python_module="${12}"
comsol_module="${13}"
[[ "$(git -C "${repository}" rev-parse HEAD)" == "${commit}" ]]
[[ -z "$(git -C "${repository}" status --porcelain)" ]]
module load "${python_module}"
module load "${comsol_module}"
source "${venv}/bin/activate"
export GENERATION_CPU_VENV="${venv}"
export STORAGE_ROOT="${storage}"
cd "${repository}"
command=(
  python -m src.generation.cli.cli_generation submit-campaign
  "${repository}/${campaign_relative}"
  --git-commit "${commit}"
  --max-nodes "${max_nodes}"
  --cases-per-node "${cases_per_node}"
  --cores-per-case "${cores_per_case}"
  --max-parallel-cases "${max_parallel_cases}"
  --cores-per-node 32
  --storage-root "${storage}"
)
[[ -z "${only_batch}" ]] || command+=(--only-batch "${only_batch}")
[[ -z "${wall_time}" ]] || command+=(--wall-time "${wall_time}")
"${command[@]}"
REMOTE
)" || fail 1 "Remote campaign submission failed."

  if [[ "${launch_output}" =~ \"campaign_run_id\"[[:space:]]*:[[:space:]]*\"([A-Za-z0-9._-]+__[0-9a-f]{16})\" ]]; then
    LAST_RUN_ID="${BASH_REMATCH[1]}"
  else
    fail 1 "Remote launch returned no valid campaign-run ID: ${launch_output}"
  fi
  printf 'Campaign run ID: %s\n' "${LAST_RUN_ID}"
  printf 'Exact Git commit: %s\n' "${REQUESTED_COMMIT}"
  printf 'Remote storage root: %s\n' "${REMOTE_STORAGE_ROOT}"
  printf 'Campaign-run manifest:\n%s\n' "${launch_output}"
  printf 'Launch returned without waiting. Next: %q status %q\n' "$0" "${LAST_RUN_ID}"
}

remote_status_json() {
  remote_bash "${CPU_HOST}" "${REMOTE_REPOSITORY}" "${REMOTE_STORAGE_ROOT}" \
    "${REMOTE_VENV}" "${RUN_ID}" "${PYTHON_MODULE}" <<'REMOTE'
set -euo pipefail
repository="$1"
storage="$2"
venv="$3"
run_id="$4"
python_module="$5"
module load "${python_module}"
source "${venv}/bin/activate"
cd "${repository}"
python -m src.generation.cli.cli_generation campaign-status \
  "${run_id}" --storage-root "${storage}"
REMOTE
}

status_campaign() {
  validate_run_id "${RUN_ID}"
  resolve_remote_layout
  local output
  output="$(remote_status_json)" || fail 1 "Could not reconstruct remote campaign status."
  printf 'Campaign status for %s:\n%s\n' "${RUN_ID}" "${output}"
}

remote_transfer_plan() {
  remote_bash "${CPU_HOST}" "${REMOTE_REPOSITORY}" "${REMOTE_STORAGE_ROOT}" \
    "${REMOTE_VENV}" "${RUN_ID}" "${PYTHON_MODULE}" <<'REMOTE'
set -euo pipefail
repository="$1"
storage="$2"
venv="$3"
run_id="$4"
python_module="$5"
module load "${python_module}"
source "${venv}/bin/activate"
cd "${repository}"
python -m src.generation.cli.cli_generation campaign-transfer-plan \
  "${run_id}" --format tsv --storage-root "${storage}"
REMOTE
}

collect_campaign() {
  validate_run_id "${RUN_ID}"
  resolve_remote_layout
  require_command rsync
  require_command mktemp
  local transfer_plan
  transfer_plan="$(remote_transfer_plan)" \
    || fail 1 "Remote campaign is not terminally valid; nothing was transferred."

  local campaign_name=""
  local campaign_commit=""
  local campaign_directory=""
  local campaign_config=""
  local -a batch_names=()
  local -a batch_ids=()
  local -a case_counts=()
  local -a meta_directories=()
  local -a raw_directories=()
  local -a processed_directories=()
  while IFS=$'\t' read -r kind field2 field3 field4 field5 field6 field7 extra; do
    [[ -z "${extra:-}" ]] || fail 1 "Malformed transfer-plan row."
    case "${kind}" in
      campaign)
        [[ -z "${campaign_name}" ]] || fail 1 "Transfer plan declared multiple campaigns."
        campaign_name="${field2}"
        campaign_commit="${field3}"
        campaign_directory="${field4}"
        campaign_config="${field5}"
        ;;
      batch)
        batch_names+=("${field2}")
        batch_ids+=("${field3}")
        case_counts+=("${field4}")
        meta_directories+=("${field5}")
        raw_directories+=("${field6}")
        processed_directories+=("${field7}")
        ;;
      *)
        fail 1 "Unknown transfer-plan row kind: ${kind@Q}"
        ;;
    esac
  done <<< "${transfer_plan}"
  validate_commit "${campaign_commit}"
  [[ -n "${campaign_name}" && ${#batch_names[@]} -gt 0 ]] \
    || fail 1 "Remote transfer plan is empty or malformed."
  resolve_campaign_config "${PROJECT_DIR}/${campaign_config}"

  mkdir -p "${HOST_STORAGE_ROOT}"
  local storage_root
  storage_root="$(cd "${HOST_STORAGE_ROOT}" && pwd -P)"
  local staging
  staging="$(mktemp -d "${storage_root}/.generation-transfer.${RUN_ID}.XXXXXX")"
  printf 'Transfer staging root: %s\n' "${staging}"

  local -a transfer_directories=("${campaign_directory}")
  local index directory
  for index in "${!batch_names[@]}"; do
    validate_batch_name "${batch_names[index]}"
    validate_positive_integer case_count "${case_counts[index]}"
    transfer_directories+=(
      "${meta_directories[index]}"
      "${raw_directories[index]}"
      "${processed_directories[index]}"
    )
  done
  for directory in "${transfer_directories[@]}"; do
    [[ "${directory}" =~ ^[A-Za-z0-9._/-]+$ \
        && "${directory}" != /* && "${directory}" != *"/../"* \
        && "${directory}" != .state* && "${directory}" != *"/.state/"* \
        && "${directory}" != *"/work/"* ]] \
      || fail 1 "Transfer plan contains an unsafe or private path: ${directory@Q}"
    rsync -a --protect-args --relative \
      --exclude='.state/' --exclude='work/' \
      "${CPU_HOST}:${REMOTE_STORAGE_ROOT}/./${directory}" "${staging}/"
  done

  for index in "${!batch_names[@]}"; do
    docker_validate_transfer "${staging}" "${batch_names[index]}" \
      || fail 1 "Transferred batch failed local validation: ${batch_names[index]}. Staging retained at ${staging}."
  done

  local -a states=()
  local rejected_cases=0
  for index in "${!batch_names[@]}"; do
    local present=0
    for directory in "${meta_directories[index]}" "${raw_directories[index]}" "${processed_directories[index]}"; do
      [[ -e "${storage_root}/${directory}" ]] && present=$((present + 1))
    done
    if (( present == 0 )); then
      states+=(new)
    elif (( present == 3 )); then
      if docker_validate_transfer "${storage_root}" "${batch_names[index]}" \
          && cmp -s \
            "${staging}/${meta_directories[index]}/batch_manifest.json" \
            "${storage_root}/${meta_directories[index]}/batch_manifest.json"; then
        states+=(reused)
      else
        states+=(rejected)
        rejected_cases=$((rejected_cases + case_counts[index]))
      fi
    else
      states+=(rejected)
      rejected_cases=$((rejected_cases + case_counts[index]))
    fi
  done
  if (( rejected_cases > 0 )); then
    printf 'Transferred cases: 0\nReused cases: 0\nMissing cases: 0\nRejected cases: %s\n' "${rejected_cases}" >&2
    fail 1 "Collection found a partial or conflicting local identity. Staging retained at ${staging}."
  fi
  if [[ -e "${storage_root}/${campaign_directory}" ]] \
      && ! diff -qr "${staging}/${campaign_directory}" "${storage_root}/${campaign_directory}" >/dev/null; then
    fail 1 "Existing local campaign-run evidence conflicts. Staging retained at ${staging}."
  fi

  local transferred_cases=0
  local reused_cases=0
  for index in "${!batch_names[@]}"; do
    if [[ "${states[index]}" == new ]]; then
      for directory in "${meta_directories[index]}" "${raw_directories[index]}" "${processed_directories[index]}"; do
        mkdir -p "$(dirname "${storage_root}/${directory}")"
        mv -- "${staging}/${directory}" "${storage_root}/${directory}"
      done
      transferred_cases=$((transferred_cases + case_counts[index]))
    else
      reused_cases=$((reused_cases + case_counts[index]))
    fi
  done
  if [[ ! -e "${storage_root}/${campaign_directory}" ]]; then
    mkdir -p "$(dirname "${storage_root}/${campaign_directory}")"
    mv -- "${staging}/${campaign_directory}" "${storage_root}/${campaign_directory}"
  fi
  rm -rf -- "${staging}"
  printf 'Campaign: %s\nExact Git commit: %s\n' "${campaign_name}" "${campaign_commit}"
  printf 'Transferred cases: %s\nReused cases: %s\nMissing cases: 0\nRejected cases: 0\n' \
    "${transferred_cases}" "${reused_cases}"
  printf 'Local storage root: %s\n' "${storage_root}"
  if [[ "${BUILD_DATASETS}" == true ]]; then
    build_datasets
  fi
}

campaign_config_from_terminal() {
  local terminal_path="$1"
  local line=""
  local value=""
  local matches=0
  while IFS= read -r line; do
    if [[ "${line}" =~ ^[[:space:]]*\"campaign_config\"[[:space:]]*:[[:space:]]*\"([A-Za-z0-9._/-]+)\",?[[:space:]]*$ ]]; then
      value="${BASH_REMATCH[1]}"
      matches=$((matches + 1))
    fi
  done < "${terminal_path}"
  (( matches == 1 )) \
    || fail 1 "Terminal campaign evidence must contain exactly one controlled campaign_config field."
  [[ "${value}" == configs/generation/campaigns/*.yaml \
      && "${value}" != /* && "${value}" != *"//"* \
      && "${value}" != *"/../"* && "${value}" != ../* ]] \
    || fail 1 "Terminal campaign_config is not one safe canonical campaign path."
  printf '%s' "${value}"
}

build_datasets() {
  validate_run_id "${RUN_ID}"
  local storage_root="${STORAGE_ROOT:-${PROJECT_DIR}/../storage}"
  local terminal_path="${storage_root}/01_generation/meta/campaigns/${RUN_ID}/campaign_terminal.json"
  [[ -f "${terminal_path}" ]] \
    || fail 1 "Collect and validate campaign ${RUN_ID} before building datasets."
  if [[ -z "${CAMPAIGN_CONFIG_PATH:-}" ]]; then
    local campaign_relative_path
    campaign_relative_path="$(campaign_config_from_terminal "${terminal_path}")"
    resolve_campaign_config "${PROJECT_DIR}/${campaign_relative_path}"
  fi
  printf 'Building every declared package through the maintained Docker wrapper.\n'
  STORAGE_ROOT="${storage_root}" "${PROJECT_DIR}/scripts/docker_job.sh" \
    build-datasets "${CAMPAIGN_CONFIG_PATH}"
}

wait_for_campaign() {
  while true; do
    local output
    output="$(remote_status_json)" || fail 1 "Remote status query failed while waiting."
    printf '%s\n' "${output}"
    if [[ "${output}" =~ \"campaign_state\"[[:space:]]*:[[:space:]]*\"complete\" ]]; then
      return 0
    fi
    if [[ "${output}" =~ \"campaign_state\"[[:space:]]*:[[:space:]]*\"failed\" ]]; then
      fail 1 "Campaign entered a terminal failure state; inspect status and retained work."
    fi
    sleep "${POLL_SECONDS}"
  done
}

if (( $# == 0 )); then
  usage
  exit 2
fi
if [[ "$1" == -h || "$1" == --help ]]; then
  usage
  exit 0
fi
SUBCOMMAND="$1"
shift

CPU_HOST="${DEFAULT_CPU_HOST}"
REMOTE_ROOT=""
REQUESTED_COMMIT=""
EXECUTE_SETUP=false
ONLY_BATCH=""
WALL_TIME=""
MAX_NODES=""
CASES_PER_NODE=""
CORES_PER_CASE=""
MAX_PARALLEL_CASES=""
WAIT_FOR_COMPLETION=false
BUILD_DATASETS=false
POLL_SECONDS=30
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
    --execute)
      EXECUTE_SETUP=true
      shift
      ;;
    --only-batch)
      (( $# >= 2 )) || fail 2 "--only-batch requires a value."
      ONLY_BATCH="$2"
      shift 2
      ;;
    --wall-time)
      (( $# >= 2 )) || fail 2 "--wall-time requires a value."
      WALL_TIME="$2"
      shift 2
      ;;
    --max-nodes)
      (( $# >= 2 )) || fail 2 "--max-nodes requires a value."
      MAX_NODES="$2"
      shift 2
      ;;
    --cases-per-node)
      (( $# >= 2 )) || fail 2 "--cases-per-node requires a value."
      CASES_PER_NODE="$2"
      shift 2
      ;;
    --cores-per-case)
      (( $# >= 2 )) || fail 2 "--cores-per-case requires a value."
      CORES_PER_CASE="$2"
      shift 2
      ;;
    --max-parallel-cases)
      (( $# >= 2 )) || fail 2 "--max-parallel-cases requires a value."
      MAX_PARALLEL_CASES="$2"
      shift 2
      ;;
    --wait)
      WAIT_FOR_COMPLETION=true
      shift
      ;;
    --build-datasets)
      BUILD_DATASETS=true
      shift
      ;;
    --poll-seconds)
      (( $# >= 2 )) || fail 2 "--poll-seconds requires a value."
      POLL_SECONDS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      fail 2 "Unsupported option: $1"
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

[[ -z "${REQUESTED_COMMIT}" ]] || validate_commit "${REQUESTED_COMMIT}"
validate_positive_integer --poll-seconds "${POLL_SECONDS}"

case "${SUBCOMMAND}" in
  setup-cpu)
    (( ${#POSITIONAL[@]} == 0 )) || fail 2 "setup-cpu accepts no positional arguments."
    setup_cpu
    ;;
  launch)
    (( ${#POSITIONAL[@]} == 1 )) || fail 2 "launch requires exactly one campaign config."
    [[ "${EXECUTE_SETUP}" == false && "${WAIT_FOR_COMPLETION}" == false && "${BUILD_DATASETS}" == false ]] \
      || fail 2 "launch does not accept --execute, --wait, or --build-datasets."
    CAMPAIGN_ARGUMENT="${POSITIONAL[0]}"
    launch_campaign
    ;;
  status)
    (( ${#POSITIONAL[@]} == 1 )) || fail 2 "status requires exactly one campaign-run ID."
    RUN_ID="${POSITIONAL[0]}"
    status_campaign
    ;;
  collect)
    (( ${#POSITIONAL[@]} == 1 )) || fail 2 "collect requires exactly one campaign-run ID."
    RUN_ID="${POSITIONAL[0]}"
    collect_campaign
    ;;
  build-datasets)
    (( ${#POSITIONAL[@]} == 1 )) || fail 2 "build-datasets requires exactly one campaign-run ID."
    RUN_ID="${POSITIONAL[0]}"
    build_datasets
    ;;
  all)
    (( ${#POSITIONAL[@]} == 1 )) || fail 2 "all requires exactly one campaign config."
    [[ "${EXECUTE_SETUP}" == false ]] || fail 2 "Use setup-cpu --execute separately before all."
    if [[ "${BUILD_DATASETS}" == true && "${WAIT_FOR_COMPLETION}" != true ]]; then
      fail 2 "all --build-datasets also requires --wait."
    fi
    CAMPAIGN_ARGUMENT="${POSITIONAL[0]}"
    launch_campaign
    RUN_ID="${LAST_RUN_ID}"
    if [[ "${WAIT_FOR_COMPLETION}" != true ]]; then
      printf 'Slurm jobs continue independently. Resume with: %q status %q\n' "$0" "${RUN_ID}"
      exit 0
    fi
    wait_for_campaign
    collect_campaign
    ;;
  *)
    usage
    fail 2 "Unsupported subcommand: ${SUBCOMMAND}"
    ;;
esac
