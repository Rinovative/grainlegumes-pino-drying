#!/bin/bash -l
set -Eeuo pipefail

if (( $# < 3 )); then
  printf 'Usage: %s REPOSITORY cli|benchmark COMMAND [ARGS...]\n' "$0" >&2
  exit 2
fi

REPOSITORY_ROOT="$1"
MODE="$2"
shift 2
PREREQUISITE_HELPER="${REPOSITORY_ROOT}/scripts/generation_prerequisites.sh"
if [[ ! -f "${PREREQUISITE_HELPER}" || -L "${PREREQUISITE_HELPER}" ]]; then
  printf 'Native Generation Python worker prerequisite helper is missing or unsafe.\n' >&2
  exit 1
fi

# Repository and fingerprint checks must finish before importing project Python.
/bin/bash "${PREREQUISITE_HELPER}" validate-worker-repository \
  "${REPOSITORY_ROOT}" "${GENERATION_GIT_COMMIT:-}" "$0" >&2

EXPECTED_STORAGE="$(realpath -m -- "${REPOSITORY_ROOT}/../storage")"
EXPECTED_VENV="$(realpath -m -- "${REPOSITORY_ROOT}/../runtime/venvs/generation")"
if [[ "${STORAGE_ROOT:-}" != "${EXPECTED_STORAGE}" \
  || "${GENERATION_NATIVE_VENV:-}" != "${EXPECTED_VENV}" \
  || ! -x "${GENERATION_NATIVE_VENV}/bin/python" ]]; then
  printf 'Native Generation Python worker requires sibling storage and Generation venv.\n' >&2
  exit 2
fi
if [[ ! "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ \
  || ! "${SLURM_CPUS_PER_TASK:-}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'Native Generation Python worker requires a Slurm CPU allocation.\n' >&2
  exit 2
fi

if ! command -v module >/dev/null 2>&1 || ! module load Python/3.12 >&2; then
  printf 'Native Generation Python worker could not load Python/3.12.\n' >&2
  exit 1
fi
if [[ "${MODE}" == benchmark ]]; then
  if [[ "$1" != materialize-core-benchmark-inputs \
    && "$1" != submit-core-benchmark \
    && "$1" != resume-core-benchmark ]]; then
    printf 'Unsupported native benchmark preparation operation: %s\n' "$1" >&2
    exit 2
  fi
  if ! module load Comsol/v6.4 >&2; then
    printf 'Native Generation Python worker could not load Comsol/v6.4.\n' >&2
    exit 1
  fi
  COMSOL_EXECUTABLE="$(command -v comsol)" || {
    printf 'Native COMSOL executable is unavailable.\n' >&2
    exit 1
  }
COMSOL_EXECUTABLE="$(readlink -f -- "${COMSOL_EXECUTABLE}")"
  # shellcheck source=generation_prerequisites.sh
  source "${PREREQUISITE_HELPER}"
  COMSOL_VERSION="$(generation_comsol_version "${COMSOL_EXECUTABLE}")" || {
    printf 'Native COMSOL version query failed.\n' >&2
    exit 1
  }
  SCRATCH_PARENT="${TMPDIR:-/tmp}"
  if [[ "${SCRATCH_PARENT}" != /* || ! -d "${SCRATCH_PARENT}" \
    || ! -w "${SCRATCH_PARENT}" ]]; then
    printf 'Native benchmark scratch parent is unavailable: %s\n' "${SCRATCH_PARENT}" >&2
    exit 1
  fi
  SCRATCH="$(mktemp -d "${SCRATCH_PARENT%/}/generation-benchmark-preflight.XXXXXXXX")"
  cleanup_benchmark_scratch() {
    [[ "${SCRATCH}" == "${SCRATCH_PARENT%/}/generation-benchmark-preflight."* ]] ||
      { printf 'Refusing to remove an unexpected benchmark scratch directory.\n' >&2; return 1; }
    rm -rf -- "${SCRATCH}"
  }
  trap cleanup_benchmark_scratch EXIT
  set -- "$@" --scratch-root "${SCRATCH}"
  if [[ "$1" != resume-core-benchmark ]]; then
    set -- "$@" --comsol-version-output "${COMSOL_VERSION}" \
      --comsol-executable-path "${COMSOL_EXECUTABLE}"
  fi
elif [[ "${MODE}" != cli ]]; then
  printf 'Unsupported native Generation Python worker mode: %s\n' "${MODE}" >&2
  exit 2
fi

cd "${REPOSITORY_ROOT}"
status=0
"${GENERATION_NATIVE_VENV}/bin/python" -m src.generation.cli.cli_generation "$@" || status=$?
if ! /bin/bash "${PREREQUISITE_HELPER}" validate-worker-repository \
  "${REPOSITORY_ROOT}" "${GENERATION_GIT_COMMIT}" "$0" >&2; then
  if (( status == 0 )); then
    status=1
  fi
fi
exit "${status}"
