#!/usr/bin/env bash
# Create one fresh locked native environment inside a Slurm CPU allocation.
set -euo pipefail

environment_name=native
case "${1:-}" in
  '') ;;
  --candidate) environment_name=native-candidate ;;
  *) printf 'Usage: bash %s [--candidate]\n' "$0" >&2; exit 2 ;;
esac
[[ $# -le 1 && "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || {
  printf 'Provisioning requires a Slurm allocation and at most one argument.\n' >&2
  exit 2
}
repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
runtime_root="$(realpath -e -- "${repository_root}/../runtime")"
export UV_PROJECT_ENVIRONMENT="${runtime_root}/venvs/${environment_name}"
[[ ! -e "${UV_PROJECT_ENVIRONMENT}" && ! -L "${UV_PROJECT_ENVIRONMENT}" ]] || {
  printf 'Fresh provisioning requires an absent target: %s\n' "${UV_PROJECT_ENVIRONMENT}" >&2
  exit 1
}
/usr/bin/python3.12 -I -c 'import sys, venv, ssl; assert sys.version_info[:2] == (3, 12)'
uv_executable="${runtime_root}/bin/uv"
if [[ ! -e "${uv_executable}" ]]; then
  [[ "$(/zfspool/software/uv/uv --version | cut -d ' ' -f 2)" == 0.11.7 ]]
  install -D /zfspool/software/uv/uv "${uv_executable}"
fi
[[ "$("${uv_executable}" --version | cut -d ' ' -f 2)" == 0.11.7 ]]
export UV_CACHE_DIR="${runtime_root}/uv/cache"
export UV_PYTHON_DOWNLOADS=never
export UV_LINK_MODE=copy
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV
cd -- "${repository_root}"
"${uv_executable}" sync --locked --group dev --python /usr/bin/python3.12
"${uv_executable}" pip check --python "${UV_PROJECT_ENVIRONMENT}/bin/python"
