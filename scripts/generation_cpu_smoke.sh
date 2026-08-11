#!/bin/bash -l
set -euo pipefail

if (( $# != 11 )); then
  printf '%s\n' \
    "Usage: $0 VENV CAMPAIGN STORAGE ONLY_BATCH CORES_PER_CASE MODE PYTHON_MODULE COMSOL_MODULE PYTHON_EXECUTABLE COMSOL_EXECUTABLE SCHEDULER" >&2
  exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  printf 'CPU preflight must run inside a Slurm allocation.\n' >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
# shellcheck source=generation_prerequisites.sh
source "${SCRIPT_DIR}/generation_prerequisites.sh"
GENERATION_CPU_VENV="$1"
CAMPAIGN_CONFIG="$2"
STORAGE_ROOT="$3"
ONLY_BATCH="$4"
CORES_PER_CASE="$5"
PREFLIGHT_MODE="$6"
PYTHON_MODULE="$7"
COMSOL_MODULE="$8"
PYTHON_EXECUTABLE="$9"
COMSOL_EXECUTABLE="${10}"
SCHEDULER_KIND="${11}"

if [[ "${GENERATION_CPU_VENV}" != /* || "${STORAGE_ROOT}" != /* || "${CAMPAIGN_CONFIG}" != /* ]]; then
  printf 'Venv, campaign, and storage paths must be absolute.\n' >&2
  exit 2
fi
if [[ ! -x "${GENERATION_CPU_VENV}/bin/python" ]]; then
  printf 'CPU compute-node prerequisite missing: Generation CPU venv %s (blocks compute).\n' \
    "${GENERATION_CPU_VENV}" >&2
  exit 1
fi
if [[ "${PREFLIGHT_MODE}" != environment-only && "${PREFLIGHT_MODE}" != production-ready && "${PREFLIGHT_MODE}" != mapping-probe ]]; then
  printf 'Mode must be environment-only, production-ready, or mapping-probe.\n' >&2
  exit 2
fi
for value in "${PYTHON_MODULE}" "${COMSOL_MODULE}" "${PYTHON_EXECUTABLE}" \
  "${COMSOL_EXECUTABLE}" "${SCHEDULER_KIND}"; do
  [[ -n "${value}" && "${value}" != *$'\n'* && "${value}" != *$'\r'* ]] || {
    printf 'Resolved execution bootstrap values must be safe non-empty text.\n' >&2
    exit 2
  }
done
[[ "${SCHEDULER_KIND}" == slurm ]] || {
  printf 'CPU preflight requires configured scheduler=slurm.\n' >&2
  exit 2
}

COMPUTE_DOMAIN="CPU compute-node"
generation_require_command "${COMPUTE_DOMAIN}" module "compute bootstrap"
generation_require_command "${COMPUTE_DOMAIN}" mktemp "scratch probe"
generation_require_command "${COMPUTE_DOMAIN}" rmdir "scratch probe cleanup"
generation_require_command "${COMPUTE_DOMAIN}" hostname "compute-node identity"
if ! module load "${PYTHON_MODULE}"; then
  generation_prerequisite_failed \
    "${COMPUTE_DOMAIN}" "Python module ${PYTHON_MODULE}" "compute"
fi
generation_report_pass "${COMPUTE_DOMAIN}" "module:${PYTHON_MODULE}"
if ! module load "${COMSOL_MODULE}"; then
  generation_prerequisite_failed \
    "${COMPUTE_DOMAIN}" "COMSOL module ${COMSOL_MODULE}" "native smoke and compute"
fi
generation_report_pass "${COMPUTE_DOMAIN}" "module:${COMSOL_MODULE}"
generation_require_command \
  "${COMPUTE_DOMAIN}" "${PYTHON_EXECUTABLE}" "case materialization and HDF5 admission"
generation_run_check \
  "${COMPUTE_DOMAIN}" "python-version:${PYTHON_EXECUTABLE}" "compute" \
  "${PYTHON_EXECUTABLE}" --version
generation_require_command \
  "${COMPUTE_DOMAIN}" "${COMSOL_EXECUTABLE}" "native smoke and compute"
generation_run_check \
  "${COMPUTE_DOMAIN}" "comsol-version:${COMSOL_EXECUTABLE}" "native smoke and compute" \
  "${COMSOL_EXECUTABLE}" -version
source "${GENERATION_CPU_VENV}/bin/activate"
if ! "${GENERATION_CPU_VENV}/bin/python" -c \
  'import h5py, numpy, scipy, yaml; import src.generation.cli.cli_generation'; then
  generation_prerequisite_failed \
    "${COMPUTE_DOMAIN}" "Generation CPU venv package/imports" \
    "case materialization and HDF5 conversion/admission"
fi
generation_report_pass "${COMPUTE_DOMAIN}" "Generation-venv-imports"

SCRATCH_PARENT="${TMPDIR:-/tmp}"
if [[ "${SCRATCH_PARENT}" != /* || ! -d "${SCRATCH_PARENT}" || ! -w "${SCRATCH_PARENT}" ]]; then
  generation_prerequisite_missing \
    "${COMPUTE_DOMAIN}" "writable scratch parent: ${SCRATCH_PARENT}" "compute"
fi
PROBE_ROOT="$(mktemp -d "${SCRATCH_PARENT%/}/vp2-preflight-${SLURM_JOB_ID}.XXXXXX")"
generation_report_pass "${COMPUTE_DOMAIN}" "owned-scratch-probe" "${PROBE_ROOT}"
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
else
  COMMAND=(
    "${GENERATION_CPU_VENV}/bin/python"
    -m src.generation.cli.cli_generation
    preflight "${CAMPAIGN_CONFIG}"
    --storage-root "${STORAGE_ROOT}"
    --work-root "${PROBE_ROOT}"
    --venv-path "${GENERATION_CPU_VENV}"
  )
  if [[ "${PREFLIGHT_MODE}" == environment-only ]]; then
    COMMAND+=(--environment-only)
  fi
fi
if [[ "${ONLY_BATCH}" != - ]]; then
  COMMAND+=(--only-batch "${ONLY_BATCH}")
fi
if ! "${COMMAND[@]}"; then
  generation_prerequisite_failed \
    "${COMPUTE_DOMAIN}" "Generation environment validation" "native smoke and compute"
fi
printf 'Native CPU %s completed on %s.\n' "${PREFLIGHT_MODE}" "$(hostname)"
