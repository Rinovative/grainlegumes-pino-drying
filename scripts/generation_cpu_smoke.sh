#!/bin/bash -l
set -euo pipefail

if (( $# != 10 )); then
  printf 'Usage: %s VENV CAMPAIGN STORAGE ONLY_BATCH MAX_NODES CASES_PER_NODE CORES_PER_CASE MAX_PARALLEL CORES_PER_NODE MODE\n' "$0" >&2
  exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  printf 'CPU preflight must run inside a Slurm allocation.\n' >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
GENERATION_CPU_VENV="$1"
CAMPAIGN_CONFIG="$2"
STORAGE_ROOT="$3"
ONLY_BATCH="$4"
MAX_NODES="$5"
CASES_PER_NODE="$6"
CORES_PER_CASE="$7"
MAX_PARALLEL_CASES="$8"
CORES_PER_NODE="$9"
PREFLIGHT_MODE="${10}"

if [[ "${GENERATION_CPU_VENV}" != /* || "${STORAGE_ROOT}" != /* || "${CAMPAIGN_CONFIG}" != /* ]]; then
  printf 'Venv, campaign, and storage paths must be absolute.\n' >&2
  exit 2
fi
if [[ ! -x "${GENERATION_CPU_VENV}/bin/python" ]]; then
  printf 'Generation CPU venv is missing: %s\n' "${GENERATION_CPU_VENV}" >&2
  exit 2
fi
if [[ "${PREFLIGHT_MODE}" != environment-only && "${PREFLIGHT_MODE}" != production-ready && "${PREFLIGHT_MODE}" != mapping-probe ]]; then
  printf 'Mode must be environment-only, production-ready, or mapping-probe.\n' >&2
  exit 2
fi

module load Python/3.10
module load Comsol/v6.4
command -v python3
python3 --version
command -v comsol
COMSOL_VERSION_OUTPUT="$(comsol -version 2>&1)"
printf '%s\n' "${COMSOL_VERSION_OUTPUT}"
[[ "${COMSOL_VERSION_OUTPUT}" == *"6.4"* ]] || { printf 'COMSOL 6.4 required.\n' >&2; exit 1; }
command -v sbatch
sbatch --version
command -v squeue
command -v sacct
command -v scancel
command -v rsync
rsync --version
source "${GENERATION_CPU_VENV}/bin/activate"
python -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 10) else f"Python 3.10 required, got {sys.version}")'
python -c 'import h5py, numpy, scipy, yaml; import src.generation.cli.cli_generation'

SCRATCH_PARENT="${TMPDIR:-/tmp}"
if [[ "${SCRATCH_PARENT}" != /* || ! -d "${SCRATCH_PARENT}" || ! -w "${SCRATCH_PARENT}" ]]; then
  printf 'Preflight scratch parent is unavailable: %s\n' "${SCRATCH_PARENT}" >&2
  exit 1
fi
PROBE_ROOT="$(mktemp -d "${SCRATCH_PARENT%/}/vp2-preflight-${SLURM_JOB_ID}.XXXXXX")"
cleanup_probe_root() {
  local status="$?"
  trap - EXIT
  if ! rmdir -- "${PROBE_ROOT}"; then
    printf 'Preflight left unexpected content under %s\n' "${PROBE_ROOT}" >&2
    if (( status == 0 )); then
      status=1
    fi
  fi
  exit "${status}"
}
trap cleanup_probe_root EXIT

cd "${REPOSITORY_ROOT}"
if [[ "${PREFLIGHT_MODE}" == mapping-probe ]]; then
  COMMAND=(
    "${GENERATION_CPU_VENV}/bin/python"
    -m src.generation.cli.cli_generation
    mapping-probe "${CAMPAIGN_CONFIG}"
    --storage-root "${STORAGE_ROOT}"
    --work-root "${PROBE_ROOT}"
    --cores-per-case "${CORES_PER_CASE}"
  )
  if [[ "${ONLY_BATCH}" != - ]]; then
    COMMAND+=(--only-batch "${ONLY_BATCH}")
  fi
  "${COMMAND[@]}"
  printf 'Native COMSOL mapping probe completed on %s.\n' "$(hostname)"
else
  COMMAND=(
    "${GENERATION_CPU_VENV}/bin/python"
    -m src.generation.cli.cli_generation
    preflight "${CAMPAIGN_CONFIG}"
    --storage-root "${STORAGE_ROOT}"
    --work-root "${PROBE_ROOT}"
    --venv-path "${GENERATION_CPU_VENV}"
    --max-nodes "${MAX_NODES}"
    --cases-per-node "${CASES_PER_NODE}"
    --cores-per-case "${CORES_PER_CASE}"
    --max-parallel-cases "${MAX_PARALLEL_CASES}"
    --cores-per-node "${CORES_PER_NODE}"
  )
  if [[ "${ONLY_BATCH}" != - ]]; then
    COMMAND+=(--only-batch "${ONLY_BATCH}")
  fi
  if [[ "${PREFLIGHT_MODE}" == environment-only ]]; then
    COMMAND+=(--environment-only)
  fi
  "${COMMAND[@]}"
  printf 'Native CPU preflight completed on %s.\n' "$(hostname)"
fi
