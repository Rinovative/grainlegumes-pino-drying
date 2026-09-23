#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  scripts/slurm_ml.sh --mode cpu --partition PARTITION --cpus-per-task N --mem SIZE --time HH:MM:SS train CONFIG [--resume DIR] [--output-root DIR] [--no-build-artifacts]
  scripts/slurm_ml.sh --mode gpu --partition PARTITION --cpus-per-task N --mem SIZE --time HH:MM:SS --gres gpu:TYPE:1 train CONFIG [--resume DIR] [--output-root DIR] [--no-build-artifacts]
  scripts/slurm_ml.sh [same resource options] optuna CONFIG [Optuna CLI options]
  scripts/slurm_ml.sh [same resource options] artifacts [artifact CLI options]
  scripts/slurm_ml.sh [same resource options] probe

GPU TYPE is v100, rtxa6000, or rtx6000ada. The probe checks the allocated
container and ML CLIs without opening datasets or starting workloads.
EOF
}

fail() {
  printf '%s\n' "$2" >&2
  exit "$1"
}

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIRECTORY}/.." && pwd -P)"
STORAGE_ROOT="$(realpath -m -- "${STORAGE_ROOT:-${PROJECT_ROOT}/../storage}")"
RUNTIME_ROOT="$(realpath -m -- "${RUNTIME_ROOT:-${PROJECT_ROOT}/../runtime}")"
WORKER="${SCRIPT_DIRECTORY}/slurm_ml_worker.sh"
EXECUTOR="${SCRIPT_DIRECTORY}/apptainer_exec.sh"

mode=""
partition=""
cpus=""
memory=""
wall_time=""
gres=""
while (( $# > 0 )); do
  case "$1" in
    --mode|--partition|--cpus-per-task|--mem|--time|--gres)
      (( $# >= 2 )) || fail 2 "$1 requires a value."
      option="$1"
      value="$2"
      shift 2
      case "${option}" in
        --mode) mode="${value}" ;;
        --partition) partition="${value}" ;;
        --cpus-per-task) cpus="${value}" ;;
        --mem) memory="${value}" ;;
        --time) wall_time="${value}" ;;
        --gres) gres="${value}" ;;
      esac
      ;;
    train|optuna|artifacts|probe)
      operation="$1"
      shift
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      fail 2 "Unsupported resource option or operation: $1"
      ;;
  esac
done

[[ "${mode}" == cpu || "${mode}" == gpu ]] || fail 2 "--mode must be cpu or gpu."
[[ "${partition}" =~ ^[A-Za-z][A-Za-z0-9_-]*$ ]] || fail 2 "--partition requires one Slurm partition name."
[[ "${cpus}" =~ ^[1-9][0-9]*$ ]] || fail 2 "--cpus-per-task must be a positive integer."
[[ "${memory}" =~ ^[1-9][0-9]*[MGT]$ ]] || fail 2 "--mem must be a positive Slurm M, G, or T quantity."
[[ "${wall_time}" =~ ^([0-9]+-)?[0-9]{1,2}:[0-5][0-9]:[0-5][0-9]$ ]] || fail 2 "--time must be HH:MM:SS or D-HH:MM:SS."
if [[ "${mode}" == cpu ]]; then
  [[ -z "${gres}" ]] || fail 2 "CPU mode must not request a GPU GRES."
else
  [[ "${gres}" =~ ^gpu:(v100|rtxa6000|rtx6000ada):1$ ]] || fail 2 "GPU mode requires --gres gpu:v100:1, gpu:rtxa6000:1, or gpu:rtx6000ada:1."
fi
[[ "${operation:-}" == train || "${operation:-}" == optuna || "${operation:-}" == artifacts || "${operation:-}" == probe ]] || fail 2 "Choose train, optuna, artifacts, or probe."
[[ -x "${WORKER}" && -x "${EXECUTOR}" ]] || fail 1 "The Slurm worker and Apptainer executor must be executable."
[[ -d "${STORAGE_ROOT}" && -r "${STORAGE_ROOT}" && -w "${STORAGE_ROOT}" ]] || fail 1 "Storage root must exist and be readable and writable: ${STORAGE_ROOT}"
paths_overlap() {
  [[ "$1/" == "$2/"* || "$2/" == "$1/"* ]]
}
if paths_overlap "${PROJECT_ROOT}" "${STORAGE_ROOT}" \
  || paths_overlap "${PROJECT_ROOT}" "${RUNTIME_ROOT}" \
  || paths_overlap "${STORAGE_ROOT}" "${RUNTIME_ROOT}"; then
  fail 2 "Repository, storage, and runtime roots must not overlap."
fi
[[ -f "${RUNTIME_ROOT}/containers/grainlegumes-pino-drying.sif" ]] || fail 1 "Maintained SIF is missing from ${RUNTIME_ROOT}/containers."

translate_path() {
  local requested="$1"
  local resolved=""
  if [[ "${requested}" == /workspace/repo/* ]]; then
    resolved="$(realpath -m -- "${PROJECT_ROOT}/${requested#/workspace/repo/}")"
  elif [[ "${requested}" == /workspace/storage/* ]]; then
    resolved="$(realpath -m -- "${STORAGE_ROOT}/${requested#/workspace/storage/}")"
  elif [[ "${requested}" == /* ]]; then
    resolved="$(realpath -m -- "${requested}")"
  elif [[ -e "${PWD}/${requested}" ]]; then
    resolved="$(realpath -m -- "${PWD}/${requested}")"
  else
    resolved="$(realpath -m -- "${PROJECT_ROOT}/${requested}")"
  fi
  if [[ "${resolved}" == "${PROJECT_ROOT}/"* ]]; then
    printf '/workspace/repo/%s' "${resolved#"${PROJECT_ROOT}/"}"
  elif [[ "${resolved}" == "${STORAGE_ROOT}/"* ]]; then
    printf '/workspace/storage/%s' "${resolved#"${STORAGE_ROOT}/"}"
  else
    fail 2 "Path must remain below the repository or storage root: ${requested}"
  fi
}

semantic_args=()
config_host=""
config_sha="none"
task="probe"
if [[ "${operation}" == probe ]]; then
  (( $# == 0 )) || fail 2 "probe accepts no arguments."
elif [[ "${operation}" == train || "${operation}" == optuna ]]; then
  (( $# > 0 )) || fail 2 "${operation} requires one config."
  config_requested="$1"
  shift
  config_container="$(translate_path "${config_requested}")"
  if [[ "${config_container}" == /workspace/repo/* ]]; then
    config_host="${PROJECT_ROOT}/${config_container#/workspace/repo/}"
  else
  config_host="${STORAGE_ROOT}/${config_container#/workspace/storage/}"
  fi
  [[ -f "${config_host}" && -r "${config_host}" ]] || fail 2 "${operation} config must be a readable file: ${config_requested}"
  config_host="$(realpath -e -- "${config_host}")"
  [[ "${config_host}" == "${PROJECT_ROOT}/"* || "${config_host}" == "${STORAGE_ROOT}/"* ]] || fail 2 "Config must remain below the repository or storage root."
  semantic_args+=("${config_container}")
  resume_seen=false
  output_seen=false
  device_seen=false
  artifacts_seen=false
  while (( $# > 0 )); do
    if [[ "${operation}" == optuna ]]; then
      case "$1" in
        --n-trials)
          (( $# >= 2 )) || fail 2 "--n-trials requires a value."
          [[ "$2" =~ ^[1-9][0-9]*$ ]] || fail 2 "--n-trials must be a positive integer."
          semantic_args+=("$1" "$2")
          shift 2
          continue
          ;;
        --dry-run|--show-progress-bar)
          semantic_args+=("$1")
          shift
          continue
          ;;
      esac
    fi
    case "$1" in
      --resume|--output-root)
        [[ "${operation}" == train || "$1" == --output-root ]] || fail 2 "--resume is training-only."
        (( $# >= 2 )) || fail 2 "$1 requires a path."
        option="$1"
        value="$(translate_path "$2")"
        [[ "${value}" == /workspace/storage/* ]] || fail 2 "$1 must be below durable storage."
        if [[ "${option}" == --resume ]]; then
          [[ "${resume_seen}" == false ]] || fail 2 "Duplicate --resume."
          resume_seen=true
        else
          [[ "${output_seen}" == false ]] || fail 2 "Duplicate --output-root."
          output_seen=true
        fi
        semantic_args+=("${option}" "${value}")
        shift 2
        ;;
      --device)
        (( $# >= 2 )) || fail 2 "--device requires cpu or cuda."
        [[ "${device_seen}" == false ]] || fail 2 "Duplicate --device."
        expected_device="cuda"
        [[ "${mode}" == cpu ]] && expected_device="cpu"
        [[ "$2" == "${expected_device}" ]] || fail 2 "--device must match the requested Slurm execution mode."
        semantic_args+=("--device" "$2")
        device_seen=true
        shift 2
        ;;
      --no-build-artifacts)
        [[ "${operation}" == train ]] || fail 2 "--no-build-artifacts is training-only."
        [[ "${artifacts_seen}" == false ]] || fail 2 "Duplicate --no-build-artifacts."
        semantic_args+=("$1")
        artifacts_seen=true
        shift
        ;;
      *) fail 2 "Unsupported ${operation} argument: $1" ;;
    esac
  done
  if [[ "${device_seen}" == false ]]; then
    [[ "${mode}" == cpu ]] && semantic_args+=("--device" "cpu") || semantic_args+=("--device" "cuda")
  fi
  preflight="$("${EXECUTOR}" python -m src.experiments.cli.cli_config_preflight "${operation}" "${config_container}")" || fail 2 "${operation} config preflight failed."
  IFS=$'\t' read -r family task canonical_path run_label extra <<< "${preflight}"
  expected_family=experiment
  [[ "${operation}" == optuna ]] && expected_family=optuna
  [[ "${family}" == "${expected_family}" && -n "${task}" && -n "${canonical_path}" && -n "${run_label}" && -z "${extra:-}" ]] || fail 1 "${operation} config preflight returned an invalid summary."
  [[ "${task}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || fail 1 "${operation} config preflight returned an unsafe task name."
  config_sha="$(sha256sum -- "${config_host}")"
  config_sha="${config_sha%% *}"
else
  (( $# > 0 )) || fail 2 "artifacts requires artifact CLI options."
  task="artifacts"
  device_seen=false
  selection_seen=false
  while (( $# > 0 )); do
    case "$1" in
      --task|--run-name|--case-id|--split|--evaluation-spatial-stride)
        (( $# >= 2 )) || fail 2 "$1 requires a value."
        [[ -n "$2" ]] || fail 2 "$1 requires a value."
        [[ "$1" != --task ]] || selection_seen=true
        semantic_args+=("$1" "$2")
        shift 2
        ;;
      --runs-root|--run-dir|--dataset-root|--metadata-root|--output-root)
        (( $# >= 2 )) || fail 2 "$1 requires a path."
        option="$1"
        value="$(translate_path "$2")"
        [[ "${value}" == /workspace/storage/* ]] || fail 2 "$1 must be below durable storage."
        [[ "${option}" != --runs-root && "${option}" != --run-dir ]] || selection_seen=true
        semantic_args+=("${option}" "${value}")
        shift 2
        ;;
      --device)
        (( $# >= 2 )) || fail 2 "--device requires cpu or cuda."
        [[ "${device_seen}" == false ]] || fail 2 "Duplicate --device."
        expected_device=cuda
        [[ "${mode}" == cpu ]] && expected_device=cpu
        [[ "$2" == "${expected_device}" ]] || fail 2 "--device must match the requested Slurm execution mode."
        semantic_args+=("$1" "$2")
        device_seen=true
        shift 2
        ;;
      --rebuild|--one-case)
        semantic_args+=("$1")
        shift
        ;;
      *) fail 2 "Unsupported artifact argument: $1" ;;
    esac
  done
  [[ "${selection_seen}" == true ]] || fail 2 "artifacts requires --task, --runs-root, or --run-dir."
  if [[ "${device_seen}" == false ]]; then
    [[ "${mode}" == cpu ]] && semantic_args+=("--device" "cpu") || semantic_args+=("--device" "cuda")
  fi
fi

source_commit="$(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
[[ "${source_commit}" =~ ^[0-9a-f]{40}$ ]] || fail 1 "Cannot identify repository commit."
source_sha="$("${WORKER}" --source-fingerprint)"
[[ "${source_sha}" =~ ^[0-9a-f]{64}$ ]] || fail 1 "Cannot fingerprint repository source state."
sif_identity="$(stat -c '%s:%Y' -- "${RUNTIME_ROOT}/containers/grainlegumes-pino-drying.sif")"
sif_sha="$(sha256sum -- "${RUNTIME_ROOT}/containers/grainlegumes-pino-drying.sif")"
sif_sha="${sif_sha%% *}"
log_directory="${RUNTIME_ROOT}/logs/ml"
mkdir -p -- "${log_directory}"
job_name="ml-${operation}-${task}"
job_name="${job_name:0:80}"

submission=(
  sbatch --parsable --nodes=1 --ntasks=1
  "--cpus-per-task=${cpus}" "--mem=${memory}" "--time=${wall_time}"
  "--partition=${partition}" "--job-name=${job_name}"
  "--chdir=${PROJECT_ROOT}" --export=ALL
  "--output=${log_directory}/slurm-%j.out"
  "--error=${log_directory}/slurm-%j.err"
)
if [[ "${mode}" == gpu ]]; then
  submission+=("--gres=${gres}")
fi
export PROJECT_ROOT STORAGE_ROOT RUNTIME_ROOT
submission+=("${WORKER}" "${mode}" "${gres:-none}" "${source_commit}" "${source_sha}" "${config_sha}" "${sif_identity}" "${sif_sha}" \
  "${cpus}" "${memory}" "${wall_time}" "${partition}" "${operation}" "${semantic_args[@]}")
job_reference="$("${submission[@]}")" || fail 1 "Slurm ML submission failed."
[[ "${job_reference}" =~ ^[0-9]+(;[A-Za-z0-9._-]+)?$ ]] || fail 1 "Slurm returned an invalid job reference: ${job_reference}"
job_id="${job_reference%%;*}"
printf 'Slurm job ID: %s\nMode: %s\nOperation: %s\nStdout: %s/slurm-%s.out\nStderr: %s/slurm-%s.err\n' \
  "${job_id}" "${mode}" "${operation}" "${log_directory}" "${job_id}" "${log_directory}" "${job_id}"
