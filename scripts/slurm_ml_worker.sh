#!/usr/bin/env bash
set -euo pipefail

# The worker runs from the frozen source tree RUNTIME_ROOT/sources/SOURCE_TREE_ID submitted by scripts/slurm_ml.sh.
# Its roots are explicit: a tree never infers storage or runtime from its own location, and no Git is needed here.
for variable_name in PROJECT_ROOT STORAGE_ROOT RUNTIME_ROOT SOURCE_TREE_ID; do
  [[ -n "${!variable_name:-}" ]] || { printf '%s must be set by the submission for a frozen source tree.\n' "${variable_name}" >&2; exit 1; }
done
PROJECT_ROOT="$(realpath -m -- "${PROJECT_ROOT}")"
STORAGE_ROOT="$(realpath -m -- "${STORAGE_ROOT}")"
RUNTIME_ROOT="$(realpath -m -- "${RUNTIME_ROOT}")"
[[ "${SOURCE_TREE_ID}" =~ ^[0-9a-f]{12}-[0-9a-f]{12}$ \
  && "${PROJECT_ROOT##*/}" == "${SOURCE_TREE_ID}" \
  && "${PROJECT_ROOT}" == "${RUNTIME_ROOT}/sources/${SOURCE_TREE_ID}" ]] || {
  printf 'PROJECT_ROOT must be the source tree RUNTIME_ROOT/sources/SOURCE_TREE_ID (%s), got %s.\n' "${SOURCE_TREE_ID}" "${PROJECT_ROOT}" >&2
  exit 1
}
source_transition="${ML_SOURCE_TRANSITION:-none}"
[[ "${source_transition}" == none || "${source_transition}" == explicit ]] || { printf 'ML_SOURCE_TRANSITION must be explicit when set.\n' >&2; exit 2; }

if (( $# < 12 )); then
  printf 'Usage: %s MODE GRES COMMIT SOURCE_SHA CONFIG_SHA SIF_IDENTITY SIF_SHA CPUS MEMORY TIME PARTITION train|optuna|artifacts|probe [CLI arguments...]\n' "$0" >&2
  exit 2
fi
mode="$1"
gres="$2"
expected_commit="$3"
expected_source_sha="$4"
expected_config_sha="$5"
expected_sif_identity="$6"
expected_sif_sha="$7"
expected_cpus="$8"
requested_memory="$9"
requested_time="${10}"
requested_partition="${11}"
operation="${12}"
shift 12

[[ -n "${SLURM_JOB_ID:-}" ]] || { printf 'ML worker requires a Slurm allocation.\n' >&2; exit 1; }
[[ "${mode}" == cpu || "${mode}" == gpu ]] || exit 2
[[ "${operation}" == train || "${operation}" == optuna || "${operation}" == artifacts || "${operation}" == probe ]] || exit 2
[[ "${SLURM_CPUS_PER_TASK:-}" =~ ^[1-9][0-9]*$ ]] || { printf 'Slurm CPU allocation is missing.\n' >&2; exit 1; }
[[ "${SLURM_CPUS_PER_TASK}" == "${expected_cpus}" ]] || { printf 'Slurm CPU allocation differs from the requested count.\n' >&2; exit 1; }
[[ "${SLURM_JOB_PARTITION:-}" == "${requested_partition}" ]] || { printf 'Slurm partition differs from the requested partition.\n' >&2; exit 1; }
if [[ "${mode}" == gpu ]]; then
  [[ "${gres}" =~ ^gpu:(v100|rtxa6000|rtx6000ada):1$ ]] || exit 2
  [[ -n "${CUDA_VISIBLE_DEVICES:-}" && ( -n "${SLURM_JOB_GPUS:-}" || -n "${SLURM_STEP_GPUS:-}" ) ]] || {
    printf 'GPU worker requires Slurm GPU allocation and CUDA_VISIBLE_DEVICES.\n' >&2
    exit 1
  }
else
  [[ "${gres}" == none ]] || exit 2
  [[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_STEP_GPUS:-}" ]] || {
    printf 'CPU worker received an unexpected GPU allocation.\n' >&2
    exit 1
  }
  unset CUDA_VISIBLE_DEVICES
fi
# The tree has no Git metadata: its bytes, paths and modes are checked against the durable snapshot named like it.
verification="$(python3 "${PROJECT_ROOT}/scripts/source_fingerprint.py" --verify-tree "${PROJECT_ROOT}" "${STORAGE_ROOT}/source_snapshots")" || {
  printf 'Source tree failed verification against its snapshot: %s\n' "${PROJECT_ROOT}" >&2
  exit 1
}
read -r tree_id tree_commit tree_sha < <(
  python3 -c 'import json, sys
verified = json.load(sys.stdin)
print(verified["tree_id"], verified["git_commit"], verified["source_fingerprint_sha256"])' <<<"${verification}"
) || { printf 'Source tree verification result is unreadable.\n' >&2; exit 1; }
[[ "${tree_id}" == "${SOURCE_TREE_ID}" ]] || { printf 'Verified source tree differs from SOURCE_TREE_ID.\n' >&2; exit 1; }
[[ "${tree_commit}" == "${expected_commit}" ]] || {
  printf 'Source tree commit differs from the submitted commit.\n' >&2
  exit 1
}
[[ "${tree_sha}" == "${expected_source_sha}" ]] || {
  printf 'Source tree fingerprint differs from the submitted fingerprint.\n' >&2
  exit 1
}
sif_path="${RUNTIME_ROOT}/containers/grainlegumes-pino-drying.sif"
[[ -f "${sif_path}" && -r "${sif_path}" && "$(stat -c '%s:%Y' -- "${sif_path}")" == "${expected_sif_identity}" ]] || {
  printf 'Maintained SIF changed or is unavailable after submission.\n' >&2
  exit 1
}
sif_sha="$(sha256sum -- "${sif_path}")"
sif_sha="${sif_sha%% *}"
[[ "${sif_sha}" == "${expected_sif_sha}" ]] || {
  printf 'Maintained SIF content changed after submission.\n' >&2
  exit 1
}

if [[ "${operation}" == train || "${operation}" == optuna ]]; then
  (( $# > 0 )) || exit 2
  config="$1"
  case "${config}" in
    /workspace/repo/*) config_host="${PROJECT_ROOT}/${config#/workspace/repo/}" ;;
    /workspace/storage/*) config_host="${STORAGE_ROOT}/${config#/workspace/storage/}" ;;
    *) printf 'ML config is outside mounted roots.\n' >&2; exit 2 ;;
  esac
  config_real="$(realpath -e -- "${config_host}")"
  [[ "${config_real}" == "${PROJECT_ROOT}/"* || "${config_real}" == "${STORAGE_ROOT}/"* ]] || exit 2
  config_sha="$(sha256sum -- "${config_real}")"
  [[ "${config_sha%% *}" == "${expected_config_sha}" ]] || {
    printf 'ML config changed after submission.\n' >&2
    exit 1
  }
elif [[ "${operation}" == probe ]]; then
  (( $# == 0 )) || exit 2
  [[ "${expected_config_sha}" == none ]] || exit 2
else
  (( $# > 0 )) || exit 2
  [[ "${expected_config_sha}" == none ]] || exit 2
fi

# Three header lines; the full digests are persisted with the run in the provenance JSON below.
transition_note=""
[[ "${source_transition}" == explicit ]] && transition_note=" | continued with current code"
printf 'job %s | partition %s | gres %s (CUDA %s) | %s CPUs | %s | %s\nsource %s | commit %s%s\nimage %s | sha256 %s\n' \
  "${SLURM_JOB_ID}" "${SLURM_JOB_PARTITION}" "${gres}" "${CUDA_VISIBLE_DEVICES:-none}" "${SLURM_CPUS_PER_TASK}" "${requested_memory}" "${requested_time}" \
  "${SOURCE_TREE_ID}" "${expected_commit:0:12}" "${transition_note}" "${sif_path##*/}" "${sif_sha:0:12}"

provenance="$(python3 - "${mode}" "${gres}" "${expected_commit}" "${expected_source_sha}" "${source_transition}" "${expected_config_sha}" "${sif_sha}" "${requested_memory}" "${requested_time}" <<'PY'
import json
import os
import sys

mode, gres, commit, source_sha, source_transition, config_sha, sif_sha, memory, wall_time = sys.argv[1:]
print(json.dumps({
    "job_id": os.environ["SLURM_JOB_ID"],
    "partition": os.environ.get("SLURM_JOB_PARTITION", ""),
    "cpus_per_task": int(os.environ["SLURM_CPUS_PER_TASK"]),
    "requested_memory": memory,
    "requested_wall_time": wall_time,
    "allocated_mem_per_node": os.environ.get("SLURM_MEM_PER_NODE", ""),
    "mode": mode,
    "gres": gres,
    "source_commit": commit,
    "source_worktree_sha256": source_sha,
    "source_transition": source_transition,
    "config_sha256": config_sha,
    "sif_path": "/workspace/runtime/containers/grainlegumes-pino-drying.sif",
    "sif_sha256": sif_sha,
}, separators=(",", ":"), sort_keys=True))
PY
)"
export APPTAINERENV_ML_SLURM_PROVENANCE="${provenance}"
if [[ -z "${WANDB_API_KEY:-}" && -r "${HOME}/wandb_key.txt" ]]; then
  WANDB_API_KEY="$(tr -d '\r\n' < "${HOME}/wandb_key.txt")"
fi
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  export APPTAINERENV_WANDB_API_KEY="${WANDB_API_KEY}"
fi

# Slurm expands %j only in its own file names: the job's log files (host paths, one per line) are linked into the run and study folders by the executor.
JOB_LOG_FILES="${ML_JOB_LOG_TEMPLATES:-}"
export JOB_LOG_FILES="${JOB_LOG_FILES//%j/${SLURM_JOB_ID}}"

executor="${PROJECT_ROOT}/scripts/apptainer_exec.sh"
[[ -x "${executor}" ]] || { printf 'Maintained Apptainer executor is unavailable.\n' >&2; exit 1; }
container_args=()
[[ "${mode}" == gpu ]] && container_args+=(--gpu)
if [[ "${operation}" == train ]]; then
  exec "${executor}" "${container_args[@]}" python -m src.experiments.cli.cli_train "$@"
fi
if [[ "${operation}" == optuna ]]; then
  exec "${executor}" "${container_args[@]}" python -m src.experiments.cli.cli_optuna "$@"
fi
if [[ "${operation}" == artifacts ]]; then
  exec "${executor}" "${container_args[@]}" python -m src.experiments.cli.cli_build_artifacts "$@"
fi
exec "${executor}" "${container_args[@]}" python -m src.experiments.cli.cli_probe "${mode}"
