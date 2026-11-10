#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  scripts/slurm_ml.sh [RESOURCES] train CONFIG [--epochs N] [--train-dataset ID] [--seed N] [--train-seconds N] [--rollout-length N] [--lr-scale X] [--revision LABEL] [--train-stages a0,b,a1] [--artifacts selected|none|all|a0,b,a1] [--max-epochs N] [--continue-with-current-code]
  scripts/slurm_ml.sh [RESOURCES] train --resume STAGE [--max-epochs N] [--continue-with-current-code]
  scripts/slurm_ml.sh [RESOURCES] optuna CONFIG [--epochs N] [--train-dataset ID] [--revision LABEL] [Optuna CLI options] [--continue-with-current-code]
  scripts/slurm_ml.sh [RESOURCES] artifacts [artifact CLI options]
  scripts/slurm_ml.sh [RESOURCES] probe

RESOURCES are all optional: [--mode cpu|gpu] [--partition PARTITION] [--cpus-per-task N] [--mem SIZE] [--time HH:MM:SS] [--gres gpu:TYPE:1].
Defaults: mode gpu (probe: cpu); 4 CPUs; cpu: partition standard, 16G; gpu: partition gpu,
32G, gres gpu:rtx6000ada:1; time 04:00:00 (train, optuna), 02:00:00 (artifacts), 00:10:00 (probe).
GPU TYPE is v100, rtxa6000, or rtx6000ada. The probe checks the allocated
container and ML CLIs without opening datasets or starting workloads.
Rerunning train CONFIG continues an existing run automatically (more --epochs
extends a steady run; another transient A0 --epochs trains beside it);
train --resume STAGE continues one saved run from its own config.
LABEL (--revision) names a deliberately separate run after its science changed
(e.g. code or config content): [A-Za-z0-9][A-Za-z0-9._-]{0,63}.
Every job runs from a frozen, read-only copy of the code under runtime/sources/.
A job that continues an existing run or study uses the copy the unit last ran in;
--continue-with-current-code (train and optuna only) switches it to the current
code and is recorded as an explicit source transition. New units, artifacts and
probe always use the current code.
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
LIVE_WORKER="${SCRIPT_DIRECTORY}/slurm_ml_worker.sh"
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

[[ "${operation:-}" == train || "${operation:-}" == optuna || "${operation:-}" == artifacts || "${operation:-}" == probe ]] || fail 2 "Choose train, optuna, artifacts, or probe."
# Defaults for every omitted resource option; an explicit option always wins.
case "${operation}" in
  probe) mode="${mode:-cpu}"; wall_time="${wall_time:-00:10:00}" ;;
  artifacts) mode="${mode:-gpu}"; wall_time="${wall_time:-02:00:00}" ;;
  *) mode="${mode:-gpu}"; wall_time="${wall_time:-04:00:00}" ;;
esac
cpus="${cpus:-4}"
case "${mode}" in
  cpu) partition="${partition:-standard}"; memory="${memory:-16G}" ;;
  gpu) partition="${partition:-gpu}"; memory="${memory:-32G}"; gres="${gres:-gpu:rtx6000ada:1}" ;;
esac
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
[[ -x "${LIVE_WORKER}" && -x "${EXECUTOR}" ]] || fail 1 "The Slurm worker and Apptainer executor must be executable."
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
preflight_override_args=()
config_host=""
config_container=""
config_sha="none"
bound_tree="-"
continue_with_current_code=false
task="probe"
reject_source_choice() {
  fail 2 "--continue-with-current-code applies only to train and optuna; ${operation} always runs the current code."
}
if [[ "${operation}" == probe ]]; then
  [[ "${1:-}" != --continue-with-current-code ]] || reject_source_choice
  (( $# == 0 )) || fail 2 "probe accepts no arguments."
elif [[ "${operation}" == train || "${operation}" == optuna ]]; then
  (( $# > 0 )) || fail 2 "${operation} requires one config."
  resume_only=false
  resume_only_container=""
  if [[ "${operation}" == train && "$1" == --resume ]]; then
    (( $# >= 2 )) || fail 2 "--resume requires a stage run directory."
    resume_only_container="$(translate_path "$2")"
    [[ "${resume_only_container}" == /workspace/storage/* ]] || fail 2 "--resume must name a stage below durable storage."
    config_requested="${2%/}/config.yaml"
    config_container="${resume_only_container%/}/config.yaml"
    resume_only=true
    shift 2
  else
    config_requested="$1"
    shift
    config_container="$(translate_path "${config_requested}")"
  fi
  if [[ "${config_container}" == /workspace/repo/* ]]; then
    config_host="${PROJECT_ROOT}/${config_container#/workspace/repo/}"
  else
  config_host="${STORAGE_ROOT}/${config_container#/workspace/storage/}"
  fi
  [[ -f "${config_host}" && -r "${config_host}" ]] || fail 2 "${operation} config must be a readable file: ${config_requested}"
  config_host="$(realpath -e -- "${config_host}")"
  [[ "${config_host}" == "${PROJECT_ROOT}/"* || "${config_host}" == "${STORAGE_ROOT}/"* ]] || fail 2 "Config must remain below the repository or storage root."
  semantic_args+=("${config_container}")
  if [[ "${resume_only}" == true ]]; then
    semantic_args+=("--resume" "${resume_only_container}")
    preflight_override_args+=("--resume" "${resume_only_container}")
  fi
  revision_seen=false
  artifacts_seen=false
  train_stages_seen=false
  max_epochs_seen=false
  epochs_seen=false
  train_dataset_seen=false
  train_seconds_seen=false
  lr_scale_seen=false
  rollout_length_seen=false
  seed_seen=false
  while (( $# > 0 )); do
    if [[ "${operation}" == optuna ]]; then
      case "$1" in
        --epochs)
          (( $# >= 2 )) || fail 2 "--epochs requires a positive integer."
          [[ "$2" =~ ^[1-9][0-9]*$ ]] || fail 2 "--epochs requires a positive integer."
          [[ "${epochs_seen}" == false ]] || fail 2 "Duplicate --epochs."
          semantic_args+=("$1" "$2")
          preflight_override_args+=("--epochs" "$2")
          epochs_seen=true
          shift 2
          continue
          ;;
        --train-dataset)
          (( $# >= 2 )) || fail 2 "--train-dataset requires an exact Dataset ID."
          [[ -n "$2" ]] || fail 2 "--train-dataset requires an exact Dataset ID."
          [[ "${train_dataset_seen}" == false ]] || fail 2 "Duplicate --train-dataset."
          semantic_args+=("$1" "$2")
          preflight_override_args+=("--train-dataset" "$2")
          train_dataset_seen=true
          shift 2
          continue
          ;;
        --n-trials)
          (( $# >= 2 )) || fail 2 "--n-trials requires a value."
          [[ "$2" =~ ^[1-9][0-9]*$ ]] || fail 2 "--n-trials must be a positive integer."
          semantic_args+=("$1" "$2")
          shift 2
          continue
          ;;
        --dry-run)
          semantic_args+=("$1")
          shift
          continue
          ;;
      esac
    fi
    case "$1" in
      --resume)
        fail 2 "--resume takes a stage directory instead of CONFIG (train --resume STAGE); rerun CONFIG without --resume to continue a run."
        ;;
      --revision)
        (( $# >= 2 )) || fail 2 "--revision requires a label."
        [[ "$2" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] || fail 2 "--revision requires a label of the form [A-Za-z0-9][A-Za-z0-9._-]{0,63}."
        [[ "${revision_seen}" == false ]] || fail 2 "Duplicate --revision."
        semantic_args+=("$1" "$2")
        preflight_override_args+=("--revision" "$2")
        revision_seen=true
        shift 2
        ;;
      --continue-with-current-code)
        [[ "${continue_with_current_code}" == false ]] || fail 2 "Duplicate --continue-with-current-code."
        continue_with_current_code=true
        shift
        ;;
      --train-stages|--artifacts)
        [[ "${operation}" == train ]] || fail 2 "$1 is training-only."
        (( $# >= 2 )) || fail 2 "$1 requires a selection."
        if [[ "$1" == --train-stages ]]; then
          [[ "${train_stages_seen}" == false ]] || fail 2 "Duplicate --train-stages."
          train_stages_seen=true
        else
          [[ "${artifacts_seen}" == false ]] || fail 2 "Duplicate --artifacts."
          artifacts_seen=true
        fi
        [[ "$2" =~ ^(all|selected|none|a0|b|a1)(,(a0|b|a1))*$ ]] || fail 2 "Invalid stage selection."
        semantic_args+=("$1" "$2")
        # Only the trained stages' leaves bind the source tree; unselected stages are read-only prerequisites.
        [[ "$1" != --train-stages ]] || preflight_override_args+=("$1" "$2")
        shift 2
        ;;
      --max-epochs)
        [[ "${operation}" == train ]] || fail 2 "--max-epochs is training-only."
        [[ "${max_epochs_seen}" == false ]] || fail 2 "Duplicate --max-epochs."
        (( $# >= 2 )) || fail 2 "--max-epochs requires a positive integer."
        [[ "$2" =~ ^[1-9][0-9]*$ ]] || fail 2 "--max-epochs requires a positive integer."
        semantic_args+=("$1" "$2")
        max_epochs_seen=true
        shift 2
        ;;
      --epochs)
        [[ "${operation}" == train ]] || fail 2 "--epochs is training-only for this parser branch."
        (( $# >= 2 )) || fail 2 "--epochs requires a positive integer."
        [[ "$2" =~ ^[1-9][0-9]*$ ]] || fail 2 "--epochs requires a positive integer."
        [[ "${epochs_seen}" == false ]] || fail 2 "Duplicate --epochs."
        semantic_args+=("$1" "$2")
        preflight_override_args+=("--epochs" "$2")
        epochs_seen=true
        shift 2
        ;;
      --train-dataset)
        (( $# >= 2 )) || fail 2 "--train-dataset requires an exact Dataset ID."
        [[ -n "$2" ]] || fail 2 "--train-dataset requires an exact Dataset ID."
        [[ "${train_dataset_seen}" == false ]] || fail 2 "Duplicate --train-dataset."
        semantic_args+=("$1" "$2")
        preflight_override_args+=("--train-dataset" "$2")
        train_dataset_seen=true
        shift 2
        ;;
      --train-seconds)
        [[ "${operation}" == train ]] || fail 2 "--train-seconds is training-only."
        (( $# >= 2 )) || fail 2 "--train-seconds requires a positive number."
        [[ "$2" =~ ^[0-9]+([.][0-9]+)?$ ]] || fail 2 "--train-seconds requires a positive number."
        [[ ! "$2" =~ ^0+([.]0+)?$ ]] || fail 2 "--train-seconds requires a positive number."
        [[ "${train_seconds_seen}" == false ]] || fail 2 "Duplicate --train-seconds."
        semantic_args+=("$1" "$2")
        preflight_override_args+=("--train-seconds" "$2")
        train_seconds_seen=true
        shift 2
        ;;
      --rollout-length)
        [[ "${operation}" == train ]] || fail 2 "--rollout-length is training-only."
        (( $# >= 2 )) || fail 2 "--rollout-length requires a positive integer."
        [[ "$2" =~ ^[1-9][0-9]*$ ]] || fail 2 "--rollout-length requires a positive integer."
        [[ "${rollout_length_seen}" == false ]] || fail 2 "Duplicate --rollout-length."
        semantic_args+=("$1" "$2")
        preflight_override_args+=("--rollout-length" "$2")
        rollout_length_seen=true
        shift 2
        ;;
      --lr-scale)
        [[ "${operation}" == train ]] || fail 2 "--lr-scale is training-only."
        (( $# >= 2 )) || fail 2 "--lr-scale requires a positive number."
        [[ "$2" =~ ^[0-9]+([.][0-9]+)?$ ]] || fail 2 "--lr-scale requires a positive number."
        [[ ! "$2" =~ ^0+([.]0+)?$ ]] || fail 2 "--lr-scale requires a positive number."
        [[ "${lr_scale_seen}" == false ]] || fail 2 "Duplicate --lr-scale."
        semantic_args+=("$1" "$2")
        preflight_override_args+=("--lr-scale" "$2")
        lr_scale_seen=true
        shift 2
        ;;
      --seed)
        [[ "${operation}" == train ]] || fail 2 "--seed is training-only."
        (( $# >= 2 )) || fail 2 "--seed requires a non-negative integer."
        [[ "$2" =~ ^(0|[1-9][0-9]*)$ ]] || fail 2 "--seed requires a non-negative integer."
        [[ "${seed_seen}" == false ]] || fail 2 "Duplicate --seed."
        semantic_args+=("$1" "$2")
        preflight_override_args+=("--seed" "$2")
        seed_seen=true
        shift 2
        ;;
      *) fail 2 "Unsupported ${operation} argument: $1" ;;
    esac
  done
  [[ "${mode}" == cpu ]] && semantic_args+=("--device" "cpu") || semantic_args+=("--device" "cuda")
  preflight="$("${EXECUTOR}" python -m src.experiments.cli.cli_config_preflight "${operation}" "${config_container}" "${preflight_override_args[@]}")" || fail 2 "${operation} config preflight failed."
  IFS=$'\t' read -r family task canonical_path run_label bound_tree extra <<< "${preflight}"
  expected_family=experiment
  [[ "${operation}" == optuna ]] && expected_family=optuna
  [[ "${family}" == "${expected_family}" && -n "${task}" && -n "${canonical_path}" && -n "${run_label}" && -n "${bound_tree}" && -z "${extra:-}" ]] || fail 1 "${operation} config preflight returned an invalid summary."
  [[ "${task}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || fail 1 "${operation} config preflight returned an unsafe task name."
  [[ "${bound_tree}" == - || "${bound_tree}" =~ ^[0-9a-f]{12}-[0-9a-f]{12}$ ]] || fail 1 "${operation} config preflight returned an invalid source tree."
else
  (( $# > 0 )) || fail 2 "artifacts requires artifact CLI options."
  task="artifacts"
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
      --runs-root|--run-dir|--dataset-root|--output-root)
        (( $# >= 2 )) || fail 2 "$1 requires a path."
        option="$1"
        value="$(translate_path "$2")"
        [[ "${value}" == /workspace/storage/* ]] || fail 2 "$1 must be below durable storage."
        [[ "${option}" != --runs-root && "${option}" != --run-dir ]] || selection_seen=true
        semantic_args+=("${option}" "${value}")
        shift 2
        ;;
      --rebuild|--one-case)
        semantic_args+=("$1")
        shift
        ;;
      --continue-with-current-code) reject_source_choice ;;
      *) fail 2 "Unsupported artifact argument: $1" ;;
    esac
  done
  [[ "${selection_seen}" == true ]] || fail 2 "artifacts requires --task, --runs-root, or --run-dir."
  [[ "${mode}" == cpu ]] && semantic_args+=("--device" "cpu") || semantic_args+=("--device" "cuda")
fi

# Every job runs from a frozen read-only source tree instead of this checkout; the launcher is the only step that needs Git.
source_script="${PROJECT_ROOT}/scripts/source_fingerprint.py"
snapshots_root="${STORAGE_ROOT}/source_snapshots"
trees_root="${RUNTIME_ROOT}/sources"
[[ -f "${source_script}" ]] || fail 1 "Source freezing helper is missing: ${source_script}"
python3 "${source_script}" --prune "${trees_root}" > /dev/null || fail 1 "Pruning unused source trees failed."

# A job that continues an existing unit runs in the tree that unit last ran in; new units, artifacts and probe use the current code.
if [[ ( "${operation}" == train || "${operation}" == optuna ) && "${bound_tree}" != - && "${continue_with_current_code}" == false ]]; then
  receipt="$(python3 "${source_script}" "${PROJECT_ROOT}" --materialize-snapshot "${snapshots_root}/${bound_tree}/snapshot.json" "${trees_root}")" \
    || fail 1 "Cannot restore the source tree ${bound_tree} this ${operation} unit last ran in (its snapshot or base commit is unavailable or differs). Start a new unit with --revision, or use --continue-with-current-code if the science is unchanged."
  tree_origin="bound to the unit"
else
  receipt="$(python3 "${source_script}" "${PROJECT_ROOT}" --materialize "${snapshots_root}" "${trees_root}")" \
    || fail 1 "Cannot freeze the current source state."
  tree_origin="current code"
fi
read -r tree_id tree_path source_commit source_sha < <(
  python3 -c 'import json, sys
receipt = json.load(sys.stdin)
print(*(receipt[key] for key in ("tree_id", "tree_path", "git_commit", "source_fingerprint_sha256")))' <<<"${receipt}"
) || fail 1 "Cannot read the source tree receipt."
[[ "${tree_id}" =~ ^[0-9a-f]{12}-[0-9a-f]{12}$ && "${tree_path}" == "${trees_root}/${tree_id}" ]] || fail 1 "The source tree receipt is invalid."
[[ "${source_commit}" =~ ^[0-9a-f]{40}$ ]] || fail 1 "Cannot identify the source tree commit."
[[ "${source_sha}" =~ ^[0-9a-f]{64}$ ]] || fail 1 "Cannot fingerprint the source tree."
[[ "${bound_tree}" == - || "${tree_origin}" != "bound to the unit" || "${tree_id}" == "${bound_tree}" ]] || fail 1 "Restored source tree ${tree_id} differs from the bound tree ${bound_tree}."

# Switching a bound unit to different code is a deliberate, recorded transition.
source_transition=none
if [[ "${continue_with_current_code}" == true && "${bound_tree}" != - && "${bound_tree}" != "${tree_id}" ]]; then
  source_transition=explicit
  tree_origin="current code, explicit transition from ${bound_tree}"
fi

# The authored config is read from the tree; a saved run config below storage stays where it is.
if [[ "${operation}" == train || "${operation}" == optuna ]]; then
  if [[ "${config_container}" == /workspace/repo/* ]]; then
    [[ "${canonical_path}" != /* && "${canonical_path}" != .. && "${canonical_path}" != ../* && "${canonical_path}" != */../* ]] \
      || fail 1 "The ${operation} config does not resolve to a repository-relative path: ${canonical_path}"
    config_host="$(realpath -e -- "${tree_path}/${canonical_path}")" || fail 1 "The ${operation} config is missing from source tree ${tree_id}: ${canonical_path}"
    [[ "${config_host}" == "${tree_path}/"* ]] || fail 1 "The ${operation} config leaves source tree ${tree_id}: ${canonical_path}"
  fi
  config_sha="$(sha256sum -- "${config_host}")"
  config_sha="${config_sha%% *}"
fi

worker="${tree_path}/scripts/slurm_ml_worker.sh"
[[ -x "${worker}" ]] || fail 1 "Source tree ${tree_id} lacks an executable Slurm worker."
sif_identity="$(stat -c '%s:%Y' -- "${RUNTIME_ROOT}/containers/grainlegumes-pino-drying.sif")"
sif_sha="$(sha256sum -- "${RUNTIME_ROOT}/containers/grainlegumes-pino-drying.sif")"
sif_sha="${sif_sha%% *}"
log_directory="${RUNTIME_ROOT}/logs/ml"
mkdir -p -- "${log_directory}"
# The one definition of the log names: sbatch expands %j, and the job links the files into its run and study folders.
stdout_template="${log_directory}/${operation}-%j.out"
stderr_template="${log_directory}/${operation}-%j.err"
job_name="ml-${operation}-${task}"
job_name="${job_name:0:80}"

# --open-mode=append: the site default truncates, and a requeue would also truncate the hard links of the logs.
submission=(
  sbatch --parsable --nodes=1 --ntasks=1
  "--cpus-per-task=${cpus}" "--mem=${memory}" "--time=${wall_time}"
  "--partition=${partition}" "--job-name=${job_name}"
  "--chdir=${tree_path}" --export=ALL
  "--output=${stdout_template}" "--error=${stderr_template}" --open-mode=append
)
if [[ "${mode}" == gpu ]]; then
  submission+=("--gres=${gres}")
fi
submission+=("${worker}" "${mode}" "${gres:-none}" "${source_commit}" "${source_sha}" "${config_sha}" "${sif_identity}" "${sif_sha}" \
  "${cpus}" "${memory}" "${wall_time}" "${partition}" "${operation}" "${semantic_args[@]}")
# The job inherits this environment: explicit roots for the tree, and the transition marker only when the switch is explicit.
if [[ "${source_transition}" == explicit ]]; then
  job_environment=("ML_SOURCE_TRANSITION=explicit")
else
  job_environment=(-u ML_SOURCE_TRANSITION)
fi
job_environment+=("PROJECT_ROOT=${tree_path}" "SOURCE_TREE_ID=${tree_id}" "STORAGE_ROOT=${STORAGE_ROOT}" "RUNTIME_ROOT=${RUNTIME_ROOT}")
job_environment+=("ML_JOB_LOG_TEMPLATES=${stdout_template}"$'\n'"${stderr_template}")
job_reference="$(env "${job_environment[@]}" "${submission[@]}")" || fail 1 "Slurm ML submission failed."
[[ "${job_reference}" =~ ^[0-9]+(;[A-Za-z0-9._-]+)?$ ]] || fail 1 "Slurm returned an invalid job reference: ${job_reference}"
job_id="${job_reference%%;*}"
python3 "${tree_path}/scripts/source_fingerprint.py" --lease "${trees_root}" "${tree_id}" "${job_id}" > /dev/null \
  || fail 1 "Slurm job ${job_id} was submitted, but leasing its source tree ${tree_id} failed; the tree stays protected from pruning for 24 hours."
printf 'Slurm job ID: %s\nMode: %s\nOperation: %s\nSource tree: %s (%s)\nStdout: %s\nStderr: %s\n' \
  "${job_id}" "${mode}" "${operation}" "${tree_id}" "${tree_origin}" "${stdout_template//%j/${job_id}}" "${stderr_template//%j/${job_id}}"
