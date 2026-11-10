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
# The one admitted overlap: a frozen source tree RUNTIME_ROOT/sources/<commit12>-<fingerprint12> as the repository.
# It stays bound read-only at /workspace/repo; inside the image every root is an explicit /workspace path.
frozen_source_tree() {
  local tree_name="${PROJECT_ROOT#"${RUNTIME_ROOT}/sources/"}"
  [[ "${PROJECT_ROOT}" == "${RUNTIME_ROOT}/sources/"* && "${tree_name}" =~ ^[0-9a-f]{12}-[0-9a-f]{12}$ ]]
}
if paths_overlap "${PROJECT_ROOT}" "${RUNTIME_ROOT}" && ! frozen_source_tree; then
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
XDG_CACHE_DIRECTORY="${RUNTIME_ROOT}/apptainer/xdg-cache"
XDG_CONFIG_DIRECTORY="${RUNTIME_ROOT}/apptainer/xdg-config"
LOG_DIRECTORY="${RUNTIME_ROOT}/logs"
IMAGE_PATH="${CONTAINER_DIRECTORY}/grainlegumes-pino-drying.sif"
mkdir -p -- \
  "${CONTAINER_DIRECTORY}" \
  "${CACHE_DIRECTORY}" \
  "${HOME_DIRECTORY}" \
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

# Temporary files live in a private directory on node-local storage that is removed when the container exits:
# on the shared runtime volume, Python multiprocessing and W&B could not delete theirs (EBUSY) and left them behind.
SCRATCH_PARENT="${TMPDIR:-/tmp}"
# A job ended by SIGKILL or the OOM killer cannot run the EXIT trap, and ICE does not clear node scratch after jobs:
# remove such leftovers of this user's ended jobs. Only directories older than an hour whose job is no longer queued
# go, and nothing goes when the queue cannot be read.
if [[ -n "${SLURM_JOB_ID:-}" ]] && ACTIVE_JOBS="$(squeue -h -u "$(id -un)" -o %i 2>/dev/null)"; then
  while IFS= read -r -d '' leftover; do
    leftover_job="${leftover##*/vp2-apptainer.}"
    leftover_job="${leftover_job%%.*}"
    if [[ "${leftover_job}" =~ ^[0-9]+$ && "${leftover_job}" != "${SLURM_JOB_ID}" ]] && ! grep -qx -- "${leftover_job}" <<<"${ACTIVE_JOBS}"; then
      rm -rf -- "${leftover}"
    fi
  done < <(find "${SCRATCH_PARENT}" -mindepth 1 -maxdepth 1 -type d -name 'vp2-apptainer.*.*' -user "$(id -u)" -mmin +60 -print0 2>/dev/null)
fi
TEMP_DIRECTORY="$(mktemp -d "${SCRATCH_PARENT}/vp2-apptainer.${SLURM_JOB_ID:-local}.XXXXXXXX")"

# A container cannot hard-link across the separate storage and runtime binds (EXDEV), so the host links the Slurm logs after the run:
# JOB_LOG_FILES (host paths, one per line; empty files are skipped) go into logs/ of every folder the job recorded in JOB_LOG_OWNERS_FILE. This runs in the
# EXIT trap, so it never changes the exit status or aborts the cleanup: errexit is off, every problem is one warning, nothing exits.
link_job_logs() (
  set +e
  local record="${TEMP_DIRECTORY}/job-log-owners" storage line owner log target identity error
  local -a logs=()
  [[ -n "${JOB_LOG_FILES:-}" && -f "${record}" ]] || return 0
  storage="$(realpath -e -- "${STORAGE_ROOT}")" || return 0
  # The files to link are decided once, before any owner: a warning appended to the .err during the loop cannot change the set.
  while IFS= read -r log; do
    [[ -f "${log}" && ! -L "${log}" && -s "${log}" ]] && logs+=("${log}")
  done <<<"${JOB_LOG_FILES}"
  (( ${#logs[@]} > 0 )) || return 0
  while IFS= read -r line; do
    # The container records resolved paths, so an owner must be its own canonical path: `..` or a symbolic link cannot lead out.
    owner="${storage}/${line#/workspace/storage/}"
    if [[ "${line}" != /workspace/storage/?* || "$(realpath -e -- "${owner}" 2>/dev/null)" != "${owner}" || ! -d "${owner}" || -L "${owner}/logs" ]]; then
      printf 'Warning: job log owner is not a directory below storage reached without .. or symbolic links (logs/ included): %s\n' "${line}" >&2
      continue
    fi
    if ! error="$(mkdir -p -- "${owner}/logs" 2>&1)"; then
      printf 'Warning: could not create %s/logs: %s\n' "${owner}" "${error}" >&2
      continue
    fi
    for log in "${logs[@]}"; do
      target="${owner}/logs/${log##*/}"
      if [[ -e "${target}" || -L "${target}" ]]; then
        [[ "${target}" -ef "${log}" ]] || printf 'Warning: %s is a different file; %s was not linked.\n' "${target}" "${log}" >&2
        continue
      fi
      if ! identity="$(stat -c '%d:%i' -- "${log}" 2>&1)"; then
        printf 'Warning: could not read the identity of %s; it was not linked: %s\n' "${log}" "${identity}" >&2
        continue
      fi
      if ! error="$(ln -- "${log}" "${target}" 2>&1)"; then
        printf 'Warning: could not link %s: %s\n' "${log}" "${error}" >&2
      elif [[ "$(stat -c '%d:%i' -- "${target}")" != "${identity}" ]]; then
        rm -f -- "${target}"
        printf 'Warning: %s changed while it was linked; the link was removed.\n' "${log}" >&2
      fi
    done
  done < <(sort -u -- "${record}")
  return 0
)
# The EXIT trap keeps the container's exit status: errexit is off inside it, a further TERM or INT cannot cut it short, and
# it ends with that status explicitly, even when a warning cannot be written.
finish_job() {
  local status=$?
  trap '' TERM INT
  set +e
  link_job_logs
  rm -rf -- "${TEMP_DIRECTORY}" || printf 'Warning: could not remove %s\n' "${TEMP_DIRECTORY}" >&2
  exit "${status}"
}
trap finish_job EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

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
  --bind "${TEMP_DIRECTORY}:/workspace/tmp:rw"
  --pwd /workspace/repo
  --env PROJECT_ROOT=/workspace/repo
  --env STORAGE_ROOT=/workspace/storage
  --env RUNTIME_ROOT=/workspace/runtime
  --env TMPDIR=/workspace/tmp
  --env JOB_LOG_OWNERS_FILE=/workspace/tmp/job-log-owners
  --env XDG_CACHE_HOME=/workspace/runtime/apptainer/xdg-cache
  --env XDG_CONFIG_HOME=/workspace/runtime/apptainer/xdg-config
)

if [[ "${GPU_MODE}" == true ]]; then
  APPTAINER_ARGUMENTS+=(--nv --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}")
fi

# No exec: the shell stays to link the job logs and remove the temporary directory; set -e passes the container's exit status on.
apptainer exec "${APPTAINER_ARGUMENTS[@]}" "${IMAGE_PATH}" "$@"
