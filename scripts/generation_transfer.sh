#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 || $# > 4 )); then
  printf 'Usage: %s CPU_HOST CPU_STORAGE_ROOT GPU_STORAGE_ROOT [--execute]\n' "$0" >&2
  exit 2
fi

CPU_HOST="$1"
CPU_STORAGE_ROOT="$2"
GPU_STORAGE_ROOT="$3"
MODE="${4:---dry-run}"

if [[ ! "${CPU_HOST}" =~ ^[A-Za-z0-9._@-]+$ || -z "${CPU_STORAGE_ROOT}" || "${CPU_STORAGE_ROOT}" != /* ]]; then
  printf 'A safe CPU host and an absolute CPU storage root are required.\n' >&2
  exit 2
fi
if [[ "${CPU_STORAGE_ROOT}" == *$'\n'* || "${CPU_STORAGE_ROOT}" == *$'\r'* ]]; then
  printf 'CPU storage root cannot contain control characters.\n' >&2
  exit 2
fi
if [[ -z "${GPU_STORAGE_ROOT}" || "${GPU_STORAGE_ROOT}" != /* ]]; then
  printf 'GPU storage root must be absolute.\n' >&2
  exit 2
fi
if [[ "${MODE}" != "--dry-run" && "${MODE}" != "--execute" ]]; then
  printf 'Final option must be --execute; omission performs a dry run.\n' >&2
  exit 2
fi
if ! command -v rsync >/dev/null 2>&1; then
  printf 'rsync is required for generation transfer.\n' >&2
  exit 1
fi

RSYNC_ARGUMENTS=(
  -a
  --protect-args
  --prune-empty-dirs
  --exclude=/.state/
  --exclude='*_work_*'
)
if [[ "${MODE}" == "--dry-run" ]]; then
  RSYNC_ARGUMENTS+=(--dry-run)
else
  mkdir -p "${GPU_STORAGE_ROOT%/}/01_generation"
fi

exec rsync \
  "${RSYNC_ARGUMENTS[@]}" \
  -- \
  "${CPU_HOST}:${CPU_STORAGE_ROOT%/}/01_generation/" \
  "${GPU_STORAGE_ROOT%/}/01_generation/"
