#!/bin/bash -l
set -euo pipefail

if (( $# != 1 )) || [[ -z "${SLURM_JOB_ID:-}" || "${SLURM_CPUS_PER_TASK:-}" != 1 ]]; then
  printf 'Native Generation smoke requires one repository and a one-CPU Slurm allocation.\n' >&2
  exit 2
fi

REPOSITORY_ROOT="$(realpath -e -- "$1")"
[[ -f "${REPOSITORY_ROOT}/scripts/generation_prerequisites.sh" ]] ||
  { printf 'Smoke source must be the shared repository.\n' >&2; exit 2; }
[[ "${GENERATION_NATIVE_VENV:-}" == "$(realpath -m -- "${REPOSITORY_ROOT}/../runtime/venvs/generation")" ]] ||
  { printf 'Smoke requires the sibling native Generation environment.\n' >&2; exit 2; }

/bin/bash "${REPOSITORY_ROOT}/scripts/generation_prerequisites.sh" validate-worker-repository \
  "${REPOSITORY_ROOT}" "${GENERATION_GIT_COMMIT:-}" "${BASH_SOURCE[0]}"
# shellcheck source=generation_prerequisites.sh
source "${REPOSITORY_ROOT}/scripts/generation_prerequisites.sh"
module load Python/3.12
module load Comsol/v6.4
cd "${REPOSITORY_ROOT}"
"${GENERATION_NATIVE_VENV}/bin/python" -c \
  'import sys, h5py, numpy, scipy, torch, yaml; import src.generation.cli.cli_generation; print("Generation Python", sys.version.split()[0])'

SCRATCH_PARENT="${TMPDIR:-/tmp}"
[[ "${SCRATCH_PARENT}" == /* && -d "${SCRATCH_PARENT}" && -w "${SCRATCH_PARENT}" ]] ||
  { printf 'Smoke scratch parent is unavailable.\n' >&2; exit 1; }
WORK_ROOT="$(mktemp -d "${SCRATCH_PARENT%/}/generation-native-smoke-${SLURM_JOB_ID}.XXXXXXXX")"
cleanup() {
  local status="$?"
  trap - EXIT
  if (( status != 0 )) && [[ -f "${WORK_ROOT}/smoke.log" ]]; then
    tail -n 80 "${WORK_ROOT}/smoke.log" >&2
  fi
  [[ "${WORK_ROOT}" == "${SCRATCH_PARENT%/}/generation-native-smoke-${SLURM_JOB_ID}."* ]] ||
    { printf 'Refusing to remove an unexpected smoke workspace.\n' >&2; exit 1; }
  rm -rf -- "${WORK_ROOT}"
  exit "${status}"
}
trap cleanup EXIT
mkdir -p -- "${WORK_ROOT}/configuration" "${WORK_ROOT}/tmp"
cat > "${WORK_ROOT}/GenerationNativeSmoke.java" <<'JAVA'
import com.comsol.model.Model;
import com.comsol.model.util.ModelUtil;

public class GenerationNativeSmoke {
    public static void main(String[] args) throws java.io.IOException {
        Model model = run();
        model.save("smoke.mph");
    }

    public static Model run() {
        return ModelUtil.create("GenerationNativeSmoke");
    }
}
JAVA
cd "${WORK_ROOT}"
generation_comsol_version comsol
comsol compile -configuration "${WORK_ROOT}/configuration" GenerationNativeSmoke.java
[[ -s GenerationNativeSmoke.class ]] ||
  { printf 'COMSOL Java smoke model was not compiled.\n' >&2; exit 1; }
comsol batch -configuration "${WORK_ROOT}/configuration" -tmpdir "${WORK_ROOT}/tmp" \
  -inputfile GenerationNativeSmoke.class -outputfile smoke.mph \
  -job generation-native-smoke -np 1 -batchlog smoke.log -batchlogout
[[ -s smoke.mph ]] ||
  { printf 'COMSOL native batch did not publish its disposable model.\n' >&2; exit 1; }
printf 'GENERATION NATIVE SMOKE PASS job=%s comsol=6.4 model_bytes=%s\n' \
  "${SLURM_JOB_ID}" "$(stat -c %s smoke.mph)"
