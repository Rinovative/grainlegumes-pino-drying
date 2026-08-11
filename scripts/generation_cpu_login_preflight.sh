#!/bin/bash -l
set -euo pipefail

if (( $# != 7 )); then
  printf '%s\n' \
    "Usage: $0 REPOSITORY STORAGE VENV COMMIT REPOSITORY_URL PYTHON_MODULE PYTHON_EXECUTABLE" >&2
  exit 2
fi

REPOSITORY="$1"
STORAGE_ROOT="$2"
GENERATION_CPU_VENV="$3"
EXPECTED_COMMIT="$4"
EXPECTED_REPOSITORY_URL="$5"
PYTHON_MODULE="$6"
PYTHON_EXECUTABLE="$7"
PREREQUISITE_HELPER="${REPOSITORY}/scripts/generation_prerequisites.sh"

if [[ ! -r "${PREREQUISITE_HELPER}" ]]; then
  printf 'CPU login prerequisite missing: readable repository checkout (%s) (blocks setup and Slurm control).\n' \
    "${REPOSITORY}" >&2
  exit 1
fi
# shellcheck source=generation_prerequisites.sh
source "${PREREQUISITE_HELPER}"

for command_name in git rsync sbatch squeue sacct scancel stat module; do
  case "${command_name}" in
    rsync) blocked_operation="transfer" ;;
    sbatch|squeue|sacct|scancel) blocked_operation="Slurm control" ;;
    git) blocked_operation="setup and checkout validation" ;;
    module) blocked_operation="CPU venv bootstrap and Slurm orchestration" ;;
    *) blocked_operation="login-host setup validation" ;;
  esac
  generation_require_command "CPU login" "${command_name}" "${blocked_operation}"
done

if [[ ! -d "${REPOSITORY}/.git" || -L "${REPOSITORY}" || ! -r "${REPOSITORY}" || ! -x "${REPOSITORY}" ]]; then
  generation_prerequisite_missing \
    "CPU login" "readable safe repository checkout: ${REPOSITORY}" "setup and Slurm control"
fi
if [[ ! -d "${STORAGE_ROOT}" || -L "${STORAGE_ROOT}" || ! -w "${STORAGE_ROOT}" || ! -x "${STORAGE_ROOT}" ]]; then
  generation_prerequisite_missing \
    "CPU login" "writable durable storage: ${STORAGE_ROOT}" "compute publication and transfer"
fi
if [[ ! -d "${GENERATION_CPU_VENV}" || -L "${GENERATION_CPU_VENV}" \
  || ! -x "${GENERATION_CPU_VENV}/bin/python" ]]; then
  generation_prerequisite_missing \
    "CPU login" "Generation CPU venv: ${GENERATION_CPU_VENV}" "Slurm orchestration"
fi
for path in "${REPOSITORY}" "${STORAGE_ROOT}" "${GENERATION_CPU_VENV}"; do
  if [[ "$(stat -c %u "${path}")" -ne "${UID}" ]]; then
    generation_prerequisite_failed \
      "CPU login" "current-user ownership: ${path}" "safe setup and orchestration"
  fi
done
generation_report_pass "CPU login" "repository-storage-venv-access"

if [[ -n "$(git -C "${REPOSITORY}" status --porcelain)" ]]; then
  generation_prerequisite_failed \
    "CPU login" "clean repository checkout" "exact-commit Slurm orchestration"
fi
if [[ "$(git -C "${REPOSITORY}" rev-parse HEAD)" != "${EXPECTED_COMMIT}" ]]; then
  generation_prerequisite_failed \
    "CPU login" "checkout commit ${EXPECTED_COMMIT}" "exact-commit Slurm orchestration"
fi
if [[ "$(git -C "${REPOSITORY}" remote get-url origin)" != "${EXPECTED_REPOSITORY_URL}" ]]; then
  generation_prerequisite_failed \
    "CPU login" "HTTPS checkout origin ${EXPECTED_REPOSITORY_URL}" "setup and update"
fi
generation_report_pass "CPU login" "exact-checkout" "${EXPECTED_COMMIT}"

if ! module load "${PYTHON_MODULE}"; then
  generation_prerequisite_failed \
    "CPU login" "Python module ${PYTHON_MODULE}" "CPU venv and Slurm orchestration"
fi
generation_report_pass "CPU login" "module:${PYTHON_MODULE}"
generation_require_command \
  "CPU login" "${PYTHON_EXECUTABLE}" "CPU venv bootstrap and Slurm orchestration"
generation_run_check \
  "CPU login" "python-version:${PYTHON_EXECUTABLE}" "CPU venv bootstrap" \
  "${PYTHON_EXECUTABLE}" --version
generation_run_check "CPU login" "scheduler-version:sbatch" "Slurm control" \
  sbatch --version
generation_run_check "CPU login" "transfer-version:rsync" "transfer" \
  rsync --version

source "${GENERATION_CPU_VENV}/bin/activate"
cd "${REPOSITORY}"
if ! "${GENERATION_CPU_VENV}/bin/python" -c \
  'import h5py, numpy, scipy, yaml; import src.generation.cli.cli_generation'; then
  generation_prerequisite_failed \
    "CPU login" "Generation CPU venv package/imports" "Slurm orchestration"
fi
generation_report_pass "CPU login" "Generation-venv-imports"
