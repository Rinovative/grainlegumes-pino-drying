#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
  printf 'Usage: %s GENERATION_CPU_VENV\n' "$0" >&2
  exit 2
fi

GENERATION_CPU_VENV="$1"
if [[ "${GENERATION_CPU_VENV}" != /* || ! -x "${GENERATION_CPU_VENV}/bin/python" ]]; then
  printf 'Generation CPU venv is missing or unsafe: %q\n' "${GENERATION_CPU_VENV}" >&2
  exit 2
fi

module load Python/3.10
source "${GENERATION_CPU_VENV}/bin/activate"
python -c 'import h5py, numpy, scipy, yaml; import src.generation.cli.cli_generation'
python -m src.generation.cli.cli_generation --help >/dev/null
printf 'Generation CPU compute-node smoke passed on %s.\n' "$(hostname)"
