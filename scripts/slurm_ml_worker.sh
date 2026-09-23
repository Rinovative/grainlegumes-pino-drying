#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_PROJECT_ROOT="$(cd -- "${SCRIPT_DIRECTORY}/.." && pwd -P)"
PROJECT_ROOT="$(realpath -m -- "${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}")"
STORAGE_ROOT="$(realpath -m -- "${STORAGE_ROOT:-${PROJECT_ROOT}/../storage}")"
RUNTIME_ROOT="$(realpath -m -- "${RUNTIME_ROOT:-${PROJECT_ROOT}/../runtime}")"

source_fingerprint() {
  python3 "${PROJECT_ROOT}/scripts/source_fingerprint.py" "${PROJECT_ROOT}"
}

if [[ "${1:-}" == --source-fingerprint ]]; then
  (( $# == 1 )) || exit 2
  source_fingerprint
  exit
fi

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
[[ "$(git -C "${PROJECT_ROOT}" rev-parse HEAD)" == "${expected_commit}" ]] || {
  printf 'Repository commit changed after submission.\n' >&2
  exit 1
}
[[ "$(source_fingerprint)" == "${expected_source_sha}" ]] || {
  printf 'Repository source state changed after submission.\n' >&2
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

printf 'Slurm job: %s\nPartition: %s\nCPUs per task: %s\nRequested memory: %s\nRequested time: %s\nGPU GRES request: %s\nCUDA_VISIBLE_DEVICES: %s\nSource commit: %s\nSource worktree SHA256: %s\nSIF: %s\nSIF SHA256: %s\n' \
  "${SLURM_JOB_ID}" "${SLURM_JOB_PARTITION}" "${SLURM_CPUS_PER_TASK}" "${requested_memory}" "${requested_time}" "${gres}" "${CUDA_VISIBLE_DEVICES:-none}" \
  "${expected_commit}" "${expected_source_sha}" "${sif_path}" "${sif_sha}"

provenance="$(python3 - "${mode}" "${gres}" "${expected_commit}" "${expected_source_sha}" "${expected_config_sha}" "${sif_sha}" "${requested_memory}" "${requested_time}" <<'PY'
import json
import os
import sys

mode, gres, commit, source_sha, config_sha, sif_sha, memory, wall_time = sys.argv[1:]
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
exec "${executor}" "${container_args[@]}" python -c '
import importlib
import json
import os
import sys
import torch

mode = sys.argv[1]
importlib.import_module("src.experiments.cli.cli_train")
importlib.import_module("src.experiments.cli.cli_optuna")
importlib.import_module("src.experiments.cli.cli_build_artifacts")
provenance = json.loads(os.environ["ML_SLURM_PROVENANCE"])
assert provenance["job_id"] and provenance["sif_sha256"]
job_id = provenance["job_id"]
if mode == "gpu":
    assert torch.cuda.device_count() == 1, "expected exactly one framework-visible GPU"
    device_name = torch.cuda.get_device_name(0)
    if provenance["gres"] == "gpu:v100:1":
        assert device_name == "Tesla V100-PCIE-32GB", f"unexpected V100 model: {device_name}"
    torch.empty(1, device="cuda").sum().item()
else:
    assert mode == "cpu"
    device_name = "none"
visible_gpus = torch.cuda.device_count()
if mode == "cpu":
    assert visible_gpus == 0, "expected no framework-visible GPUs in CPU mode"
print(f"ML probe passed: mode={mode} job={job_id} visible_gpus={visible_gpus} device={device_name}")
' "${mode}"
