#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'Usage: %s [--gpu] [--] COMMAND [ARGUMENT ...]\n' "$0" >&2
}

GPU_MODE=false
if [[ "${1:-}" == "--gpu" ]]; then
  GPU_MODE=true
  shift
fi
if [[ "${1:-}" == "--" ]]; then
  shift
fi
if (( $# == 0 )); then
  usage
  exit 2
fi
if [[ "${GPU_MODE}" == true \
  && ( -z "${SLURM_JOB_ID:-}" \
    || -z "${CUDA_VISIBLE_DEVICES:-}" \
    || ( -z "${SLURM_STEP_GPUS:-}" && -z "${SLURM_JOB_GPUS:-}" ) ) ]]; then
  printf 'GPU mode requires a Slurm GPU allocation with CUDA_VISIBLE_DEVICES set.\n' >&2
  exit 1
fi

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_PROJECT_ROOT="$(cd -- "${SCRIPT_DIRECTORY}/.." && pwd -P)"
PROJECT_ROOT="$(realpath -m -- "${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}")"
STORAGE_ROOT="$(realpath -m -- "${STORAGE_ROOT:-${PROJECT_ROOT}/../storage}")"
RUNTIME_ROOT="$(realpath -m -- "${RUNTIME_ROOT:-${PROJECT_ROOT}/../runtime}")"

if [[ ! -d "${PROJECT_ROOT}" || ! -r "${PROJECT_ROOT}" ]]; then
  printf 'Repository root is not a readable directory: %s\n' "${PROJECT_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${STORAGE_ROOT}" || ! -r "${STORAGE_ROOT}" || ! -w "${STORAGE_ROOT}" ]]; then
  printf 'Storage root must be an existing readable and writable directory: %s\n' "${STORAGE_ROOT}" >&2
  exit 1
fi

paths_overlap() {
  local first="$1"
  local second="$2"
  [[ "${first}/" == "${second}/"* || "${second}/" == "${first}/"* ]]
}

if paths_overlap "${PROJECT_ROOT}" "${STORAGE_ROOT}"; then
  printf 'Repository and storage roots must not overlap: %s ; %s\n' "${PROJECT_ROOT}" "${STORAGE_ROOT}" >&2
  exit 1
fi
if paths_overlap "${PROJECT_ROOT}" "${RUNTIME_ROOT}"; then
  printf 'Repository and runtime roots must not overlap: %s ; %s\n' "${PROJECT_ROOT}" "${RUNTIME_ROOT}" >&2
  exit 1
fi
if paths_overlap "${STORAGE_ROOT}" "${RUNTIME_ROOT}"; then
  printf 'Storage and runtime roots must not overlap: %s ; %s\n' "${STORAGE_ROOT}" "${RUNTIME_ROOT}" >&2
  exit 1
fi

CONTAINER_DIRECTORY="${RUNTIME_ROOT}/containers"
CACHE_DIRECTORY="${RUNTIME_ROOT}/apptainer/cache"
HOME_DIRECTORY="${RUNTIME_ROOT}/apptainer/home"
TEMP_DIRECTORY="${RUNTIME_ROOT}/apptainer/tmp"
XDG_CACHE_DIRECTORY="${RUNTIME_ROOT}/apptainer/xdg-cache"
XDG_CONFIG_DIRECTORY="${RUNTIME_ROOT}/apptainer/xdg-config"
LOG_DIRECTORY="${RUNTIME_ROOT}/logs"
IMAGE_PATH="${CONTAINER_DIRECTORY}/grainlegumes-pino-drying.sif"
mkdir -p -- \
  "${CONTAINER_DIRECTORY}" \
  "${CACHE_DIRECTORY}" \
  "${HOME_DIRECTORY}" \
  "${TEMP_DIRECTORY}" \
  "${XDG_CACHE_DIRECTORY}" \
  "${XDG_CONFIG_DIRECTORY}" \
  "${LOG_DIRECTORY}"

if [[ ! -f "${IMAGE_PATH}" || ! -r "${IMAGE_PATH}" ]]; then
  printf 'Maintained Apptainer image is missing or unreadable: %s\n' "${IMAGE_PATH}" >&2
  exit 1
fi
if ! command -v apptainer >/dev/null 2>&1; then
  printf 'Apptainer is required but was not found on PATH.\n' >&2
  exit 1
fi

export APPTAINER_CACHEDIR="${CACHE_DIRECTORY}"
export APPTAINER_TMPDIR="${TEMP_DIRECTORY}"
export HOME="${HOME_DIRECTORY}"
export XDG_CACHE_HOME="${XDG_CACHE_DIRECTORY}"
export XDG_CONFIG_HOME="${XDG_CONFIG_DIRECTORY}"

APPTAINER_ARGUMENTS=(
  --cleanenv
  --home "${HOME_DIRECTORY}:/home/vp2"
  --bind "${PROJECT_ROOT}:/workspace/repo:ro"
  --bind "${STORAGE_ROOT}:/workspace/storage:rw"
  --bind "${RUNTIME_ROOT}:/workspace/runtime:rw"
  --pwd /workspace/repo
  --env HOME=/home/vp2
  --env PROJECT_ROOT=/workspace/repo
  --env STORAGE_ROOT=/workspace/storage
  --env RUNTIME_ROOT=/workspace/runtime
  --env TMPDIR=/workspace/runtime/apptainer/tmp
  --env XDG_CACHE_HOME=/workspace/runtime/apptainer/xdg-cache
  --env XDG_CONFIG_HOME=/workspace/runtime/apptainer/xdg-config
)

if [[ "${GPU_MODE}" == true ]]; then
  APPTAINER_ARGUMENTS+=(--nv --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}")
fi

exec apptainer exec "${APPTAINER_ARGUMENTS[@]}" "${IMAGE_PATH}" "$@"
