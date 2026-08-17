# ruff: noqa: S101, S603, PLR2004
"""Host workflow lifecycle tests using fake Docker, remote, Slurm, and COMSOL."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from src import generation
from src.generation.cli import cli_generation

pytestmark = pytest.mark.integration

_COMMIT = "a" * 40
_RUN_ID = "steady_flow_family_generalization__0123456789abcdef"
_BENCHMARK_RUN_ID = "core_scaling_transient__fedcba9876543210"
_SMOKE_SHA = "7" * 64
_BATCH_NAME = generation.cases.config.build_batch_name(
    "steady_flow",
    "synthetic_material",
    "natural",
)
_AUTHORIZATION_SHA = "1" * 64
_TRANSFER_SHA = "2" * 64
_DATASET_SHA = "3" * 64
_WORKFLOW_SHA = "4" * 64
_INVENTORY_SHA = "5" * 64
_CLEANUP_RECEIPT_SHA = "6" * 64
_BENCHMARK_INVENTORY_SHA = "8" * 64
_BENCHMARK_FILE_COUNT = 1
_BENCHMARK_SIZE_BYTES = 3
_AUTHORIZED_BYTES = 24


def _executable(path: Path, content: str) -> Path:
    """Write one test-owned executable."""
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)
    return path


def _harness(tmp_path: Path) -> tuple[Path, Path, dict[str, str], Path, Path]:
    """Return copied workflow, command log, environment, storage, and CPU mirror."""
    repository = Path(__file__).resolve().parents[2]
    project = tmp_path / "project with spaces"
    scripts = project / "scripts"
    campaigns = project / "configs/generation/campaigns/steady_flow"
    pilot_campaigns = project / "configs/generation/campaigns/transient_drying"
    scripts.mkdir(parents=True)
    campaigns.mkdir(parents=True)
    pilot_campaigns.mkdir(parents=True)
    workflow = scripts / "generation_workflow.sh"
    shutil.copyfile(repository / "scripts/generation_workflow.sh", workflow)
    workflow.chmod(workflow.stat().st_mode | 0o111)
    docker_python = scripts / "docker_python.sh"
    shutil.copyfile(repository / "scripts/docker_python.sh", docker_python)
    docker_python.chmod(docker_python.stat().st_mode | 0o111)
    for relative in (
        "configs/generation/campaigns/steady_flow/family_generalization.yaml",
        "configs/generation/campaigns/steady_flow/technical_smoke.yaml",
        "configs/generation/campaigns/transient_drying/family_generalization.yaml",
        "configs/generation/campaigns/transient_drying/technical_smoke.yaml",
        "configs/generation/campaigns/transient_drying/pilot_check.yaml",
    ):
        source = repository / relative
        destination = project / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    shutil.copytree(
        repository / "configs/generation/benchmarks/transient_core_scaling",
        project / "configs/generation/benchmarks/transient_core_scaling",
    )
    committed_project = tmp_path / "committed project"
    shutil.copytree(project, committed_project)
    log = tmp_path / "commands.log"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _executable(
        fake_bin / "docker",
        r"""#!/usr/bin/env bash
set -euo pipefail
printf 'docker-start\n' >> "${FAKE_COMMAND_LOG}"
for argument in "$@"; do printf '<%s>\n' "${argument}" >> "${FAKE_COMMAND_LOG}"; done
if [[ "${FAKE_DOCKER_FAIL:-false}" == true ]]; then
  printf 'Synthetic Docker daemon failure.\n' >&2
  exit 88
fi
case "${1:-}" in
  info) exit 0 ;;
  image) [[ "${2:-}" == inspect && "${3:-}" == grainlegumes-pino-drying ]]; exit ;;
  run)
    arguments=("$@")
    python_index=-1
    repository_source=''
    for ((index=0; index<${#arguments[@]}; index++)); do
      if [[ "${arguments[index]}" == python ]]; then
        python_index="${index}"
      elif [[ "${arguments[index]}" == type=bind,source=*,target=/workspace/repo,readonly ]]; then
        repository_source="${arguments[index]#type=bind,source=}"
        repository_source="${repository_source%,target=/workspace/repo,readonly}"
      fi
    done
    (( python_index >= 0 )) || { printf 'Docker invocation omitted Python.\n' >&2; exit 91; }
    [[ -n "${repository_source}" && -d "${repository_source}" ]] ||
      { printf 'Docker invocation omitted the repository source mount.\n' >&2; exit 91; }
    if [[ -n "${FAKE_DOCKER_READY_FILE:-}" ]]; then
      printf '%s\n' "${repository_source}" > "${FAKE_DOCKER_READY_FILE}"
      while [[ ! -e "${FAKE_DOCKER_CONTINUE_FILE}" ]]; do sleep 0.01; done
    fi
    translated=()
    for argument in "${arguments[@]:python_index+1}"; do
      case "${argument}" in
        /workspace/repo) argument="${repository_source}" ;;
        /workspace/repo/*) argument="${repository_source}/${argument#/workspace/repo/}" ;;
        /workspace/storage) argument="${STORAGE_ROOT}" ;;
        /workspace/storage/*) argument="${STORAGE_ROOT}/${argument#/workspace/storage/}" ;;
      esac
      translated+=("${argument}")
    done
    export FAKE_MOUNTED_REPO_ROOT="${repository_source}"
    cd "${repository_source}"
    exec "${FAKE_LOCAL_PYTHON}" "${translated[@]}"
    ;;
  *) printf 'Unsupported fake Docker invocation.\n' >&2; exit 92 ;;
esac
""",
    )
    for forbidden_name in ("python", "python3"):
        _executable(
            fake_bin / forbidden_name,
            r"""#!/usr/bin/env bash
set -euo pipefail
printf 'forbidden-host-python <%s>\n' "$*" >> "${FAKE_COMMAND_LOG}"
printf 'Bare-host Python is forbidden.\n' >&2
exit 93
""",
        )
    _executable(
        fake_bin / "realpath",
        r"""#!/usr/bin/env bash
set -euo pipefail
printf 'realpath-start\n' >> "${FAKE_COMMAND_LOG}"
for argument in "$@"; do
  printf '<%s>\n' "${argument}" >> "${FAKE_COMMAND_LOG}"
  case "${argument}" in
    /workspace/repo|/workspace/repo/*|/workspace/storage|/workspace/storage/*)
      printf 'Bare-host realpath received a Docker-only path: %s\n' "${argument}" >&2
      exit 94
      ;;
  esac
done
exec "${FAKE_REALPATH}" "$@"
""",
    )
    _executable(
        fake_bin / "date",
        r"""#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == +%s && -n "${FAKE_CONSOLE_TIMES:-}" ]]; then
  IFS=',' read -r -a values <<< "${FAKE_CONSOLE_TIMES}"
  index=0
  [[ ! -f "${FAKE_CONSOLE_TIME_INDEX_FILE}" ]] ||
    read -r index < "${FAKE_CONSOLE_TIME_INDEX_FILE}"
  (( ${#values[@]} > 0 )) || exit 95
  if (( index >= ${#values[@]} )); then
    index=$(( ${#values[@]} - 1 ))
  fi
  printf '%s\n' "${values[index]}"
  printf '%s\n' "$(( index + 1 ))" > "${FAKE_CONSOLE_TIME_INDEX_FILE}"
else
  exec "${FAKE_REAL_DATE}" "$@"
fi
""".replace("$", "$"),
    )
    _executable(
        fake_bin / "git",
        r"""#!/usr/bin/env bash
set -euo pipefail
printf 'git <%s>\n' "$*" >> "${FAKE_COMMAND_LOG}"
arguments=("$@")
git_directory=''
for ((index=0; index<${#arguments[@]}; index++)); do
  if [[ "${arguments[index]}" == -C ]]; then
    git_directory="${arguments[index+1]}"
    break
  fi
done
git_root() {
  local candidate="${git_directory:-${PWD}}"
  if [[ "${candidate}" == "${FAKE_PROJECT_ROOT}" || "${candidate}" == "${FAKE_PROJECT_ROOT}/"* ]]; then
    printf '%s\n' "${FAKE_PROJECT_ROOT}"
    return
  fi
  while [[ "${candidate}" != / && ! -d "${candidate}/.git" ]]; do
    candidate="$(dirname "${candidate}")"
  done
  [[ -d "${candidate}/.git" ]] || exit 97
  printf '%s\n' "${candidate}"
}
case " $* " in
  *" init --quiet "*)
    target="${arguments[${#arguments[@]}-1]}"
    mkdir -p "${target}/.git"
    ;;
  *" checkout --quiet --detach "*)
    root="$(git_root)"
    cp -a -- "${FAKE_COMMITTED_ROOT}/." "${root}/"
    mkdir -p "${root}/.git"
    ;;
  *" fetch --quiet --depth=1 --no-tags "*) ;;
  *" rev-parse --show-toplevel "*) git_root ;;
  *" rev-parse HEAD "*) printf '%s\n' "${FAKE_GIT_COMMIT}" ;;
  *" status --porcelain"*)
    root="$(git_root)"
    if [[ "${root}" == "${FAKE_PROJECT_ROOT}" ]]; then
      printf '%s' "${FAKE_GIT_STATUS:-}"
    fi
    ;;
  *" remote get-url origin "*) printf '%s\n' 'git@github.com:Rinovative/grainlegumes-pino-drying.git' ;;
  *) ;;
esac
""",
    )
    _executable(
        fake_bin / "ssh",
        r"""#!/usr/bin/env bash
set -euo pipefail
payload="$(cat)"
printf 'ssh-start\n' >> "${FAKE_COMMAND_LOG}"
for argument in "$@"; do printf '<%s>\n' "${argument}" >> "${FAKE_COMMAND_LOG}"; done
printf 'ssh-stdin-start\n%s\nssh-stdin-end\n' "${payload}" >> "${FAKE_COMMAND_LOG}"
next_campaign_state() {
  local state="${FAKE_CAMPAIGN_STATE}" index=0
  if [[ -n "${FAKE_CAMPAIGN_STATES:-}" ]]; then
    IFS=',' read -r -a states <<< "${FAKE_CAMPAIGN_STATES}"
    [[ ! -f "${FAKE_CAMPAIGN_STATE_INDEX_FILE}" ]] ||
      read -r index < "${FAKE_CAMPAIGN_STATE_INDEX_FILE}"
    (( ${#states[@]} > 0 )) || exit 96
    if (( index >= ${#states[@]} )); then
      index=$(( ${#states[@]} - 1 ))
    fi
    state="${states[index]}"
    printf '%s\n' "$(( index + 1 ))" > "${FAKE_CAMPAIGN_STATE_INDEX_FILE}"
  fi
  printf '%s\n' "${state}"
}
if [[ "${FAKE_CPU_LOGIN_RSYNC_MISSING:-false}" == true \
  && "${payload}" == *'generation_cpu_login_preflight.sh'* ]]; then
  printf 'CPU login prerequisite missing: rsync (blocks transfer).\n' >&2
  exit 71
fi
if [[ "${FAKE_LOGIN_PREFLIGHT_STDOUT:-false}" == true \
  && "${payload}" == *'generation_cpu_login_preflight.sh'* ]]; then
  printf '%s\n' \
    'Preflight domain=CPU login check=command:git status=pass detail=resolved' \
    'Generation-venv-runtime status=pass'
fi
if [[ "${payload}" == *'${HOME}'* && "${payload}" != *'root="$1"'* ]]; then
  printf '%s\n' '/remote/home'
elif [[ " $* " == *' core-benchmark-status '* && " $* " == *' --format state '* ]]; then
  printf '%s\n' "${FAKE_BENCHMARK_STATE}"
elif [[ " $* " == *' core-benchmark-status '* ]]; then
  printf '%s\n' '{"state":"retry_required","retry_repetitions":[{"variant_id":"cores_16","repetition":2,"evidence_status":"failed"}]}'
elif [[ " $* " == *' core-benchmark-transfer-plan '* ]]; then
  printf 'benchmark\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${FAKE_BENCHMARK_RUN_ID}" "${FAKE_GIT_COMMIT}" "${FAKE_BENCHMARK_RELATIVE}" \
    "${FAKE_BENCHMARK_INVENTORY_SHA}" "${FAKE_BENCHMARK_FILE_COUNT}" \
    "${FAKE_BENCHMARK_SIZE_BYTES}"
elif [[ " $* " == *' submit-core-benchmark '* ]]; then
  printf '{"benchmark_run_id":"%s","state":"submitted"}\n' "${FAKE_BENCHMARK_RUN_ID}"
elif [[ " $* " == *' plan-core-benchmark '* ]]; then
  printf '%s\n' '{"filesystem_mutated":false,"state":"planned"}'
elif [[ " $* " == *' finalize-core-benchmark '* ]]; then
  printf '%s\n' '{"state":"complete"}'
elif [[ "${payload}" == *'technical-smoke-evidence-status'* ]]; then
  requested_profile=steady_flow
  [[ " $* " != *'/transient_drying/'* ]] || requested_profile=transient_drying
  printf 'technical-smoke-evidence-profile <%s>\n' "${requested_profile}" >> "${FAKE_COMMAND_LOG}"
  if [[ "${FAKE_SMOKE_EVIDENCE_REJECT_PROFILE:-}" == "${requested_profile}" ]]; then
    printf '%s\n' '{"status":"technical_smoke_evidence_missing"}'
    exit 2
  fi
  printf '%s\n' '{"status":"technical_smoke_evidence_valid"}'
elif [[ "${payload}" == *'COMSOL version query'* ]]; then
  printf '%s\n' 'COMSOL Multiphysics 6.4.0.293'
elif [[ " $* " == *' core-benchmark-summary '* ]]; then
  printf '%s\n' '# Synthetic remote core benchmark summary'
elif [[ " $* " == *' validate-real-smoke '* ]]; then
  printf '%s\n' '{"status":"valid"}'
elif [[ " $* " == *' campaign-transfer-plan '* ]]; then
  printf '%b' "${FAKE_TRANSFER_PLAN}"
elif [[ " $* " == *' campaign-source-status '* ]]; then
  if [[ -f "${FAKE_REMOTE_CLEANED_FILE}" ]]; then
    printf 'source-status\t%s\tsource_cleanup_complete\t0\talready_complete\tFalse\n' "${FAKE_RUN_ID}"
  else
    printf 'source-status\t%s\t%s\t%s\tineligible\tFalse\n'       "${FAKE_RUN_ID}" "${FAKE_SOURCE_STATE}" "${FAKE_AUTHORIZED_BYTES}"
  fi
elif [[ " $* " == *' campaign-status '* && " $* " == *' --format monitor '* ]]; then
  state="$(next_campaign_state)"
  case "${state}" in
    submitted) state_signature="$(printf '1%.0s' {1..64})" ;;
    running) state_signature="$(printf '2%.0s' {1..64})" ;;
    publication_complete) state_signature="$(printf '3%.0s' {1..64})" ;;
    failed|partially_failed|cancelled) state_signature="$(printf '4%.0s' {1..64})" ;;
    *) state_signature="$(printf '5%.0s' {1..64})" ;;
  esac
  progress_signature="${state_signature}"
  progress_value="0.100"
  state_index=1
  [[ ! -f "${FAKE_CAMPAIGN_STATE_INDEX_FILE}" ]] ||
    read -r state_index < "${FAKE_CAMPAIGN_STATE_INDEX_FILE}"
  sequence_index=$(( state_index > 0 ? state_index - 1 : 0 ))
  if [[ -n "${FAKE_CAMPAIGN_PROGRESS_SIGNATURES:-}" ]]; then
    IFS=',' read -r -a signatures <<< "${FAKE_CAMPAIGN_PROGRESS_SIGNATURES}"
    if (( sequence_index >= ${#signatures[@]} )); then sequence_index=$(( ${#signatures[@]} - 1 )); fi
    progress_signature="${signatures[sequence_index]}"
  fi
  if [[ -n "${FAKE_PROGRESS_VALUES:-}" ]]; then
    IFS=',' read -r -a values <<< "${FAKE_PROGRESS_VALUES}"
    value_index="${sequence_index}"
    if (( value_index >= ${#values[@]} )); then value_index=$(( ${#values[@]} - 1 )); fi
    progress_value="${values[value_index]}"
  fi
  printf 'campaign-monitor\t%s\t%s\t%s\n' "${state}" "${state_signature}" "${progress_signature}"
  printf 'Campaign: %s\nState: %s\nCases: 0/1 completed, 1 active, 0 pending, 0 failed\n\n' "${FAKE_RUN_ID}" "${state}"
  printf 'Active cases:\ncase_0001  job=591776  node=node-a  elapsed=00:01:00\n'
  printf '  phase=transient_drying  sim_time=%s h  step=0.075 s\n' "${progress_value}"
  printf '  order=2  Tfail=1  NLfail=3  updated=4 s ago\n'
elif [[ " $* " == *' campaign-status '* && " $* " == *' --format summary '* ]]; then
  printf 'Campaign: %s\nState: %s\nCases: 0/1 completed, 0 active, 1 pending, 0 failed\n\n' \
    "${FAKE_RUN_ID}" "${FAKE_CAMPAIGN_STATE}"
  printf 'Pending cases:\ncase_0001  job=591776  node=unavailable  elapsed=unavailable  state=active  reason=PENDING\n'
elif [[ " $* " == *' campaign-status '* && " $* " == *' --format state '* ]]; then
  next_campaign_state
elif [[ " $* " == *' campaign-status '* ]]; then
  printf '%s\n' '{"campaign_purpose":"family_generalization","cases_per_material":null,"submission_config":{"poll_interval_seconds":1}}'
elif [[ " $* " == *' storage-status '* && " $* " == *' --role cpu '* ]]; then
  printf '%s\n' '{"role":"cpu","generation_total_bytes":24,"datasets_total_bytes":0,"runs":[]}'
elif [[ " $* " == *' cleanup-campaign-source '* ]]; then
  if [[ " $* " != *' --confirm '* ]]; then
    printf '%s\n' '{"mode":"dry-run","status":"eligible","reclaimable_bytes":24,"source_directories":["exact"]}'
  else
    while IFS= read -r relative; do
      [[ -z "${relative}" || ! -d "${FAKE_REMOTE_MIRROR}/${relative}" ]]         || rm -r -- "${FAKE_REMOTE_MIRROR}/${relative}"
    done < "${FAKE_SOURCE_DIRECTORIES_FILE}"
    : > "${FAKE_REMOTE_CLEANED_FILE}"
    printf 'cleanup\tcomplete\tcomplete\t%s\t%s\t%s\n'       "${FAKE_AUTHORIZATION_SHA}" "${FAKE_AUTHORIZED_BYTES}" "${FAKE_CLEANUP_RECEIPT_SHA}"
  fi
elif [[ " $* " == *' validate-campaign-terminal '* ]]; then
  printf '%s\n' '{"status":"terminal"}'
elif [[ " $* " == *' resume-campaign '* ]]; then
  printf '%s\n' '{"state":"submitted"}'
elif [[ " $* " == *' campaign-accounting '* ]]; then
  printf '%s\n' '{"squeue":{"output":"12345_0|RUNNING|node-a"}}'
elif [[ " $* " == *' cancel-campaign '* ]]; then
  printf '%s\n' '{"status":"cancel_requested"}'
elif [[ " $* " == *' submit-campaign'* ]]; then
  if [[ "${FAKE_TRACK_SINGLE_SUBMISSION:-false}" == true ]]; then
    [[ ! -e "${FAKE_SUBMISSION_FILE}" ]] || {
      printf '%s\n' 'synthetic duplicate campaign submission' >&2
      exit 72
    }
    printf '%s\n' '591776' > "${FAKE_SUBMISSION_FILE}"
  fi
  printf '{"campaign_run_id":"%s","state":"active","slurm_job_ids":["591776"],'\
'"submissions":[{"case_id":"case_0001","case_index":1,"status":"submitted","error":null}]}\n' \
    "${FAKE_RUN_ID}"
elif [[ " $* " == *' plan-campaign'* ]]; then
  printf '%s\n' '{"filesystem_mutated":false,"state":"planned"}'
elif [[ "${payload}" == *'sbatch --wait --parsable'* ]]; then
  printf 'preflight\t12345\t/remote/preflight/fixture\n'
fi
""".replace("$", "$"),
    )
    _executable(
        fake_bin / "rsync",
        r"""#!/usr/bin/env bash
set -euo pipefail
printf 'rsync-start\n' >> "${FAKE_COMMAND_LOG}"
for argument in "$@"; do printf '<%s>\n' "${argument}" >> "${FAKE_COMMAND_LOG}"; done
source_argument="${@: -2:1}"
destination="${@: -1}"
if [[ "${destination}" == *:* ]]; then
  exit 0
fi
remote_path="${source_argument#*:}"
relative="${remote_path#*/./}"
source="${FAKE_REMOTE_MIRROR}/${relative}"
mkdir -p "${destination}/$(dirname "${relative}")"
cp -a -- "${source}" "${destination}/${relative}"
""".replace("$", "$"),
    )
    local_python = _executable(
        fake_bin / "local-python",
        r"""#!/usr/bin/env bash
set -euo pipefail
printf 'local-python-start\n' >> "${FAKE_COMMAND_LOG}"
for argument in "$@"; do printf '<%s>\n' "${argument}" >> "${FAKE_COMMAND_LOG}"; done
printf 'local-python-repository <%s>\n' "${FAKE_MOUNTED_REPO_ROOT}" >> "${FAKE_COMMAND_LOG}"
printf 'local-python-commit <%s>\n' "${GENERATION_GIT_COMMIT:-}" >> "${FAKE_COMMAND_LOG}"
if [[ -n "${FAKE_UNTRACKED_SOURCE_PATH:-}" \
  && -e "${FAKE_MOUNTED_REPO_ROOT}/${FAKE_UNTRACKED_SOURCE_PATH}" ]]; then
  printf 'Dirty untracked source reached local Python.\n' >&2
  exit 98
fi
if [[ -n "${FAKE_EXPECT_SOURCE_FILE:-}" ]]; then
  source_file="${FAKE_MOUNTED_REPO_ROOT}/${FAKE_EXPECT_SOURCE_FILE}"
  if [[ ! -f "${source_file}" || "$(< "${source_file}")" != "${FAKE_EXPECT_SOURCE_TEXT}" ]]; then
    printf 'Local Python did not receive the committed Generation source.\n' >&2
    exit 100
  fi
fi
if [[ " $* " == *' list-campaigns '* ]]; then
  printf '%s\n' '{}'
  exit 0
fi
if [[ " $* " == *' validate-config '* ]]; then
  arguments=("$@")
  config=''
  for ((index=0; index<${#arguments[@]}; index++)); do
    if [[ "${arguments[index]}" == validate-config ]]; then
      config="${arguments[index+1]}"
      break
    fi
  done
  if [[ -n "${FAKE_REJECT_CONFIG_TEXT:-}" \
    && -n "${config}" && -f "${config}" \
    && "$(grep -F -c -- "${FAKE_REJECT_CONFIG_TEXT}" "${config}")" -ne 0 ]]; then
    printf 'Dirty tracked configuration reached local Python.\n' >&2
    exit 99
  fi
  purpose=family_generalization
  [[ " $* " != *'/technical_smoke.yaml'* ]] || purpose=technical_runtime_smoke
  [[ " $* " != *'/pilot_check.yaml'* ]] || purpose=pilot_check
  printf '%s\n' "${purpose}" > "${FAKE_CAMPAIGN_PURPOSE_FILE}"
fi
if [[ " $* " == *' inspect-core-benchmark '* ]]; then
  printf '%s\n' \
    '{"suite_name":"transient_core_scaling",'\
'"suite_digest":"8888888888888888888888888888888888888888888888888888888888888888",'\
'"repetitions":3,"resource_contract":{"cpu_host":"fixture.cluster","scheduler":"slurm",'\
'"partition":"standard","cores_per_node":32,"python_module":"Python/3.10",'\
'"comsol_module":"Comsol/v6.4","python_executable":"python","comsol_executable":"comsol","poll_interval_seconds":1},'\
'"variants":[{"variant_id":"cores_04","cores_per_case":4},'\
'{"variant_id":"cores_08","cores_per_case":8},{"variant_id":"cores_16","cores_per_case":16},'\
'{"variant_id":"cores_32","cores_per_case":32}]}'
  exit 0
fi
if [[ " $* " == *' -c '* ]]; then
  cat >/dev/null
  if [[ " $* " == *'resource = value'* ]]; then
    printf 'benchmark\ttransient_core_scaling\t%s\t3\t4,8,16,32\tfixture.cluster\t'\
'slurm\tstandard\t32\tPython/3.10\tComsol/v6.4\tpython\tcomsol\t1\n' \
      '8888888888888888888888888888888888888888888888888888888888888888'
  elif [[ " $* " == *'valid = value'* ]]; then
    printf 'smoke\t/workspace/storage/01_generation/meta/smoke_receipts/current.json\t%s\n' "${FAKE_SMOKE_SHA}"
  elif [[ " $* " == *'workflow = value'* ]]; then
    if [[ " $* " == *'repository_path'* ]]; then
      catalog_prefix=''
    else
      catalog_prefix='/workspace/repo/'
    fi
    printf 'workflow\t%s\t%s\t%s\t%s\tfixture.cluster\tslurm\tfixture\t48\t'\
'Python/fixture-3.12\tComsol/fixture-9.9\tfixture-python\tfixture-comsol\n' \
      "${catalog_prefix}configs/generation/campaigns/steady_flow/technical_smoke.yaml" \
      "${catalog_prefix}configs/generation/campaigns/transient_drying/technical_smoke.yaml" \
      "${catalog_prefix}configs/generation/campaigns/steady_flow/family_generalization.yaml" \
      "${catalog_prefix}configs/generation/campaigns/transient_drying/family_generalization.yaml"
  elif [[ " $* " == *'counts = tuple'* ]]; then
    printf 'pilot\tpilot_check\t4\t20\n'
  elif [[ " $* " == *'execution_resources'* ]]; then
    purpose="$(cat "${FAKE_CAMPAIGN_PURPOSE_FILE}")"
    printf 'execution\t%s\t8\t01:00:00\t48\t1\t1\t-\tfixture.cluster\tslurm\tfixture\t'\
'Python/fixture-3.12\tComsol/fixture-9.9\tfixture-python\tfixture-comsol\n' "${purpose}"
  elif [[ " $* " == *'campaign_purpose'* ]]; then
    printf 'campaign\tfamily_generalization\t-\t1\n'
  fi
  exit 0
fi
storage=''
directory=''
arguments=("$@")
for ((index=0; index<${#arguments[@]}; index++)); do
  case "${arguments[index]}" in
    --storage-root) storage="${arguments[index+1]}" ;;
    --directory) directory="${arguments[index+1]}" ;;
  esac
done
if [[ " $* " == *' validate-real-smoke '* ]]; then
  receipt="${storage}/01_generation/meta/smoke_receipts/current.json"
  mkdir -p "$(dirname "${receipt}")"
  printf '%s\n' '{}' > "${receipt}"
  printf '{"status":"valid","valid_receipts":[{'\
'"path":"/workspace/storage/01_generation/meta/smoke_receipts/current.json",'\
'"receipt_digest":"%s"}]}\n' "${FAKE_SMOKE_SHA}"
elif [[ " $* " == *' validate-core-benchmark '* ]]; then
  [[ -f "${FAKE_BENCHMARK_PUBLISHED_FILE}" ]]
elif [[ " $* " == *' validate-published-campaign '* ]]; then
  [[ "${FAKE_GPU_ALWAYS_VALID:-false}" == true || -f "${FAKE_GPU_PUBLISHED_FILE}" ]]
elif [[ " $* " == *' create-transfer-staging '* ]]; then
  staging="${storage}/01_generation/.state/transfer-staging/${FAKE_RUN_ID}.synthetic"
  mkdir -p "${staging}"
  printf '%s\n' "${staging}"
elif [[ " $* " == *' publish-transferred-core-benchmark '* ]]; then
  : > "${FAKE_BENCHMARK_PUBLISHED_FILE}"
  printf '%s\n' '{"status":"transfer_complete","dataset_membership":"none"}'
elif [[ " $* " == *' core-benchmark-summary '* ]]; then
  printf '%s\n' '# Synthetic local core benchmark summary'
elif [[ " $* " == *' publish-transferred-campaign '* ]]; then
  if [[ "${FAKE_PUBLISH_FAIL:-false}" == true ]]; then
    printf '%s\n' 'synthetic destination hash validation failed' >&2
    exit 4
  fi
  : > "${FAKE_GPU_PUBLISHED_FILE}"
  printf '%s\n' '{"source_removed":false,"status":"transfer_complete"}'
elif [[ " $* " == *' cleanup-transfer-staging '* ]]; then
  [[ -z "${directory}" || ! -d "${directory}" ]] || rm -r -- "${directory}"
  printf '%s\n' '{"mode":"delete"}'
elif [[ " $* " == *' build-campaign-datasets '* ]]; then
  if [[ "${FAKE_BUILD_FAIL:-false}" == true ]]; then
    printf '%s\n' 'synthetic dataset build failed' >&2
    exit 5
  fi
  : > "${FAKE_DATASETS_COMPLETE_FILE}"
  printf '%s\n' '{"status":"complete","packages":[{"dataset_id":"synthetic"}]}'
elif [[ " $* " == *' prepare-all-workflow '* ]]; then
  : > "${FAKE_WORKFLOW_READY_FILE}"
  if [[ " $* " == *' --keep-cpu-source '* ]]; then
    : > "${FAKE_WORKFLOW_COMPLETE_FILE}"
    printf '%s\n' '{"workflow_result":"success","cpu_cleanup_complete":{"status":"skipped_by_request"}}'
  else
    printf '%s\n' '{"workflow_result":"ready_for_cpu_cleanup","cpu_cleanup_complete":{"status":"pending"}}'
  fi
elif [[ " $* " == *' validate-all-workflow '* ]]; then
  [[ -f "${FAKE_WORKFLOW_COMPLETE_FILE}" ]]
  printf '%s\n' '{"workflow_result":"success"}'
elif [[ " $* " == *' cpu-cleanup-authorization '* ]]; then
  printf 'authorization\t%s\tcpu.example\t/remote/generation root/storage\t%s\t%s\t%s\t%s\t%s\t4\t%s\n' \
    "${FAKE_AUTHORIZATION_SHA}" "${storage}" "${FAKE_TRANSFER_SHA}" \
    "${FAKE_DATASET_SHA}" "${FAKE_WORKFLOW_SHA}" "${FAKE_INVENTORY_SHA}" \
    "${FAKE_AUTHORIZED_BYTES}"
elif [[ " $* " == *' record-cpu-cleanup '* ]]; then
  : > "${FAKE_WORKFLOW_COMPLETE_FILE}"
  printf '%s\n' '{"workflow_result":"success","cpu_cleanup_complete":{"status":"complete"}}'
elif [[ " $* " == *' storage-status '* ]]; then
  printf '%s\n' '{"role":"gpu","generation_total_bytes":48,"datasets_total_bytes":12,"runs":[]}'
elif [[ " $* " == *' finalize-technical-smoke-evidence '* ]]; then
  run_id="${arguments[3]}"
  evidence="${storage}/01_generation/meta/campaigns/${run_id}/technical_smoke_evidence.json"
  mkdir -p "$(dirname "${evidence}")"
  printf '%s\n' '{}' > "${evidence}"
  printf '/workspace/storage/01_generation/meta/campaigns/%s/technical_smoke_evidence.json\n' "${run_id}"
elif [[ " $* " == *' finalize-real-smoke '* ]]; then
  printf '%s\n' '/workspace/storage/01_generation/meta/smoke_receipts/current.json'
elif [[ " $* " == *' record-workflow-failure '* ]]; then
  printf '%s\n' "${storage}/failure.json"
fi
""".replace("$", "$"),
    )
    storage = tmp_path / "host storage"
    mirror = tmp_path / "remote mirror"
    state_root = tmp_path / "fake state"
    state_root.mkdir()
    source_directories_file = state_root / "source-directories.txt"
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{environment['PATH']}",
            "FAKE_COMMAND_LOG": str(log),
            "FAKE_CAMPAIGN_PURPOSE_FILE": str(state_root / "campaign-purpose"),
            "FAKE_GIT_COMMIT": _COMMIT,
            "FAKE_GIT_STATUS": "",
            "FAKE_COMMITTED_ROOT": str(committed_project),
            "FAKE_UNTRACKED_SOURCE_PATH": "",
            "FAKE_REJECT_CONFIG_TEXT": "",
            "FAKE_EXPECT_SOURCE_FILE": "",
            "FAKE_EXPECT_SOURCE_TEXT": "",
            "FAKE_DOCKER_READY_FILE": "",
            "FAKE_DOCKER_CONTINUE_FILE": "",
            "FAKE_REALPATH": shutil.which("realpath", path=os.defpath) or "/usr/bin/realpath",
            "FAKE_REAL_DATE": shutil.which("date", path=os.defpath) or "/usr/bin/date",
            "GENERATION_REPOSITORY_URL": "git@github.com:Rinovative/grainlegumes-pino-drying.git",
            "FAKE_PROJECT_ROOT": str(project),
            "FAKE_REMOTE_MIRROR": str(mirror),
            "FAKE_RUN_ID": _RUN_ID,
            "FAKE_BENCHMARK_RUN_ID": _BENCHMARK_RUN_ID,
            "FAKE_BENCHMARK_STATE": "complete",
            "FAKE_BENCHMARK_RELATIVE": (f"01_generation/meta/performance_benchmarks/core_scaling/{_BENCHMARK_RUN_ID}"),
            "FAKE_BENCHMARK_INVENTORY_SHA": _BENCHMARK_INVENTORY_SHA,
            "FAKE_BENCHMARK_FILE_COUNT": str(_BENCHMARK_FILE_COUNT),
            "FAKE_BENCHMARK_SIZE_BYTES": str(_BENCHMARK_SIZE_BYTES),
            "FAKE_SMOKE_SHA": _SMOKE_SHA,
            "FAKE_TRANSFER_PLAN": "",
            "FAKE_CAMPAIGN_STATE": "publication_complete",
            "FAKE_CAMPAIGN_STATES": "",
            "FAKE_CAMPAIGN_PROGRESS_SIGNATURES": "",
            "FAKE_PROGRESS_VALUES": "",
            "FAKE_CAMPAIGN_STATE_INDEX_FILE": str(state_root / "campaign-state-index"),
            "FAKE_CONSOLE_TIMES": "",
            "FAKE_CONSOLE_TIME_INDEX_FILE": str(state_root / "console-time-index"),
            "FAKE_SOURCE_STATE": "publication_complete",
            "FAKE_AUTHORIZED_BYTES": str(_AUTHORIZED_BYTES),
            "FAKE_AUTHORIZATION_SHA": _AUTHORIZATION_SHA,
            "FAKE_TRANSFER_SHA": _TRANSFER_SHA,
            "FAKE_DATASET_SHA": _DATASET_SHA,
            "FAKE_WORKFLOW_SHA": _WORKFLOW_SHA,
            "FAKE_INVENTORY_SHA": _INVENTORY_SHA,
            "FAKE_CLEANUP_RECEIPT_SHA": _CLEANUP_RECEIPT_SHA,
            "FAKE_LOGIN_PREFLIGHT_STDOUT": "false",
            "FAKE_TRACK_SINGLE_SUBMISSION": "false",
            "FAKE_SUBMISSION_FILE": str(state_root / "submission"),
            "FAKE_GPU_PUBLISHED_FILE": str(state_root / "gpu-published"),
            "FAKE_BENCHMARK_PUBLISHED_FILE": str(state_root / "benchmark-published"),
            "FAKE_DATASETS_COMPLETE_FILE": str(state_root / "datasets-complete"),
            "FAKE_WORKFLOW_READY_FILE": str(state_root / "workflow-ready"),
            "FAKE_WORKFLOW_COMPLETE_FILE": str(state_root / "workflow-complete"),
            "FAKE_REMOTE_CLEANED_FILE": str(state_root / "remote-cleaned"),
            "FAKE_SOURCE_DIRECTORIES_FILE": str(source_directories_file),
            "FAKE_LOCAL_PYTHON": str(local_python),
            "GENERATION_STATUS_POLL_SECONDS": "0",
            "STORAGE_ROOT": str(storage),
        }
    )
    return workflow, log, environment, storage, mirror


def _run(
    workflow: Path,
    arguments: list[str],
    environment: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    """Run one host workflow command through the fake harness."""
    return subprocess.run(
        [str(workflow), *arguments],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def _real_git(
    repository: Path,
    *arguments: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run the host Git executable against one test-owned repository."""
    executable = shutil.which("git", path=os.defpath)
    if executable is None:
        message = "The workflow source tests require Git."
        raise RuntimeError(message)
    return subprocess.run(
        [executable, "-C", str(repository), *arguments],
        check=check,
        capture_output=True,
        text=True,
    )


def _initialize_real_repository(
    workflow: Path,
    environment: dict[str, str],
) -> tuple[Path, Path, str]:
    """Commit the compact harness and route workflow Git calls to real Git."""
    project = workflow.parent.parent
    source_probe = project / "src/generation/source_probe.py"
    source_probe.parent.mkdir(parents=True)
    source_probe.write_text("committed generation behavior", encoding="utf-8")
    _real_git(project, "init", "--quiet")
    _real_git(project, "config", "user.name", "Generation Test")
    _real_git(project, "config", "user.email", "generation-test@example.invalid")
    _real_git(project, "add", ".")
    _real_git(project, "commit", "--quiet", "-m", "test: committed generation source")
    commit = _real_git(project, "rev-parse", "HEAD").stdout.strip()
    assert len(commit) == 40

    executable = shutil.which("git", path=os.defpath)
    assert executable is not None
    fake_bin = Path(environment["PATH"].split(os.pathsep, maxsplit=1)[0])
    _executable(
        fake_bin / "git",
        r"""#!/usr/bin/env bash
set -euo pipefail
printf 'git <%s>\n' "$*" >> "${FAKE_COMMAND_LOG}"
exec "${FAKE_REAL_GIT}" "$@"
""",
    )
    environment["FAKE_REAL_GIT"] = executable
    environment["FAKE_GIT_COMMIT"] = commit
    return project, source_probe, commit


def _wait_for_file(
    path: Path,
    process: subprocess.Popen[str],
    *,
    timeout_seconds: float = 10.0,
) -> str:
    """Return one synchronization file or fail with completed process output."""
    deadline = time.monotonic() + timeout_seconds
    while not path.exists():
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            pytest.fail(f"workflow exited before synchronization: code={process.returncode}\nstdout={stdout}\nstderr={stderr}")
        if time.monotonic() >= deadline:
            process.terminate()
            stdout, stderr = process.communicate(timeout=5)
            pytest.fail(f"workflow synchronization timed out\nstdout={stdout}\nstderr={stderr}")
        time.sleep(0.01)
    return path.read_text(encoding="utf-8").strip()


def _logged_source_roots(log_text: str) -> set[Path]:
    """Return the exact repository roots observed by fake local Python."""
    prefix = "local-python-repository <"
    return {Path(line[len(prefix) : -1]) for line in log_text.splitlines() if line.startswith(prefix) and line.endswith(">")}


def _remote_options() -> list[str]:
    """Return safe remote options that exercise embedded spaces."""
    return [
        "--cpu-host",
        "cpu.example",
        "--remote-root",
        "/remote/generation root",
        "--git-commit",
        _COMMIT,
    ]


def _selection_options() -> list[str]:
    """Return one valid configured-campaign batch selection."""
    return ["--only-batch", _BATCH_NAME]


def _campaign(workflow: Path) -> Path:
    """Return the copied campaign configuration."""
    return workflow.parent.parent / "configs/generation/campaigns/steady_flow/family_generalization.yaml"


def _seed_transfer(mirror: Path, environment: dict[str, str]) -> tuple[str, ...]:
    """Create one complete fake terminal transfer tree and TSV plan."""
    campaign_directory = f"01_generation/meta/campaigns/{_RUN_ID}"
    batch_id = generation.cases.config.build_batch_id(
        _BATCH_NAME,
        "fedcba9876543210" + "0" * 48,
    )
    meta_directory = f"01_generation/meta/batches/{batch_id}"
    raw_directory = f"01_generation/raw/{batch_id}"
    processed_directory = f"01_generation/processed/{batch_id}"
    source_directories = (meta_directory, raw_directory, processed_directory)
    for relative in (campaign_directory, *source_directories):
        (mirror / relative).mkdir(parents=True)
    (mirror / campaign_directory / "campaign_terminal.json").write_text(
        json.dumps({"campaign_run_id": _RUN_ID}) + "\n",
        encoding="utf-8",
    )
    (mirror / meta_directory / "batch_manifest.json").write_text("{}\n", encoding="utf-8")
    (mirror / raw_directory / "case_0001.txt").write_text("raw\n", encoding="utf-8")
    (mirror / processed_directory / "case_0001.txt").write_text("processed\n", encoding="utf-8")
    environment["FAKE_TRANSFER_PLAN"] = (
        f"campaign\tsteady_flow_family_generalization\t{_COMMIT}\t{campaign_directory}"
        "\tconfigs/generation/campaigns/steady_flow/family_generalization.yaml\n"
        f"batch\t{_BATCH_NAME}\t{batch_id}\t1\t{meta_directory}\t{raw_directory}\t{processed_directory}\n"
    )
    Path(environment["FAKE_SOURCE_DIRECTORIES_FILE"]).write_text(
        "\n".join(source_directories) + "\n",
        encoding="utf-8",
    )
    return source_directories


def test_campaign_source_status_cli_emits_positional_tsv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Keep the source-status producer to one parseable six-field TSV record."""
    status = {
        "campaign_run_id": _RUN_ID,
        "campaign_state": "active",
        "reclaimable_bytes": 1234,
        "cleanup_eligibility": "ineligible",
        "active_slurm": True,
    }
    monkeypatch.setattr(
        cli_generation.workflow_service,
        "campaign_source_status",
        lambda *_args, **_kwargs: status,
    )

    result = cli_generation.main(
        [
            "campaign-source-status",
            _RUN_ID,
            "--query-scheduler",
            "--format",
            "tsv",
            "--storage-root",
            str(tmp_path),
        ]
    )

    assert result == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    fields = captured.out.rstrip("\n").split("\t")
    assert len(fields) == 6
    assert fields[0] == "source-status"
    assert fields[1] == _RUN_ID
    assert fields[2] == "active"
    assert int(fields[3]) == status["reclaimable_bytes"]
    assert fields[4] == status["cleanup_eligibility"]
    assert fields[5] == str(status["active_slurm"])


def test_fresh_campaign_monitoring_reports_concise_success(tmp_path: Path) -> None:
    """Report one successful campaign without repeating state or dumping machine JSON."""
    workflow, log, environment, storage, _mirror = _harness(tmp_path)
    environment["FAKE_LOGIN_PREFLIGHT_STDOUT"] = "true"
    environment["FAKE_TRACK_SINGLE_SUBMISSION"] = "true"
    environment["FAKE_SOURCE_STATE"] = "active"
    environment["FAKE_CAMPAIGN_STATES"] = "submitted,submitted,running,publication_complete"
    environment["FAKE_GPU_ALWAYS_VALID"] = "true"
    campaign = workflow.parent.parent / "configs/generation/campaigns/steady_flow/technical_smoke.yaml"
    assert not storage.exists()

    result = _run(
        workflow,
        ["all", str(campaign), *_remote_options(), "--keep-cpu-source"],
        environment,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("State: submitted") == 1
    assert result.stdout.count("State: running") == 1
    assert "State: publication_complete" in result.stdout
    assert f"Campaign: {_RUN_ID}" in result.stdout
    assert any(line.startswith("      case_0001") for line in result.stdout.splitlines())
    assert any(line.startswith("        phase=transient_drying") for line in result.stdout.splitlines())
    assert "dataset_id=synthetic" in result.stdout
    assert not any(line.lstrip().startswith("{") and line.rstrip().endswith("}") for line in result.stdout.splitlines())
    assert Path(environment["FAKE_SUBMISSION_FILE"]).read_text(encoding="utf-8") == "591776\n"
    log_text = log.read_text(encoding="utf-8")
    assert sum("submit-campaign" in line for line in log_text.splitlines()) == 1


def test_unchanged_campaign_states_are_coalesced(tmp_path: Path) -> None:
    """Suppress repeated unchanged states while reporting a later state change."""
    workflow, _log, environment, _storage, _mirror = _harness(tmp_path)
    environment["FAKE_SOURCE_STATE"] = "active"
    environment["FAKE_CAMPAIGN_STATES"] = "submitted,submitted,submitted,publication_complete"
    environment["FAKE_CONSOLE_TIMES"] = "1000,1299,1300,1301"
    environment["FAKE_GPU_ALWAYS_VALID"] = "true"

    result = _run(
        workflow,
        ["all", str(_campaign(workflow)), *_remote_options(), "--keep-cpu-source"],
        environment,
    )

    assert result.returncode == 0, result.stderr
    submitted_lines = [line for line in result.stdout.splitlines() if "State: submitted" in line]
    assert 0 < len(submitted_lines) < 3
    assert "State: publication_complete" in result.stdout


def test_explicit_status_prints_campaign_summary_before_storage(tmp_path: Path) -> None:
    """Present canonical per-case campaign status before storage diagnostics."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)

    result = _run(workflow, ["status", _RUN_ID, *_remote_options()], environment)

    assert result.returncode == 0, result.stderr
    assert result.stdout.index("Campaign status:") < result.stdout.index("GPU storage status:")
    assert result.stdout.index("GPU storage status:") < result.stdout.index("CPU storage status:")
    assert f"Campaign: {_RUN_ID}" in result.stdout
    assert "case_0001" in result.stdout
    assert "campaign-status" in log.read_text(encoding="utf-8")
    assert "--format summary" in log.read_text(encoding="utf-8")


def test_changed_solver_progress_is_rendered_only_after_the_minimum_interval(tmp_path: Path) -> None:
    """Show advancing solver evidence after 60 seconds without printing every poll."""
    workflow, _log, environment, _storage, _mirror = _harness(tmp_path)
    environment["FAKE_SOURCE_STATE"] = "active"
    environment["FAKE_CAMPAIGN_STATES"] = "running,running,running,publication_complete"
    environment["FAKE_CAMPAIGN_PROGRESS_SIGNATURES"] = ",".join(("a" * 64, "b" * 64, "c" * 64, "d" * 64))
    environment["FAKE_PROGRESS_VALUES"] = "0.100,0.200,0.300,0.400"
    environment["FAKE_CONSOLE_TIMES"] = "1000,1030,1060,1061"
    environment["FAKE_GPU_ALWAYS_VALID"] = "true"

    result = _run(
        workflow,
        ["all", str(_campaign(workflow)), *_remote_options(), "--keep-cpu-source"],
        environment,
    )

    assert result.returncode == 0, result.stderr
    assert "sim_time=0.100 h" in result.stdout
    assert "sim_time=0.200 h" not in result.stdout
    assert "sim_time=0.300 h" in result.stdout
    assert "sim_time=0.400 h" in result.stdout


def test_smoke_translates_logical_campaigns_across_all_path_domains(tmp_path: Path) -> None:
    """Reproduce the native smoke path boundary from an arbitrary host checkout."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    environment["FAKE_GPU_ALWAYS_VALID"] = "true"

    result = _run(workflow, ["smoke", "--keep-cpu-source"], environment)

    assert result.returncode == 0, result.stderr
    development_root = workflow.parent.parent
    remote_root = Path("/remote/home/grainlegumes-generation/repo")
    relative_campaigns = (
        Path("configs/generation/campaigns/steady_flow/technical_smoke.yaml"),
        Path("configs/generation/campaigns/transient_drying/technical_smoke.yaml"),
    )
    log_text = log.read_text(encoding="utf-8")
    prefix = "local-python-repository <"
    source_roots = {Path(line[len(prefix) : -1]) for line in log_text.splitlines() if line.startswith(prefix) and line.endswith(">")}
    assert len(source_roots) == 1
    pinned_root = next(iter(source_roots))
    assert pinned_root != development_root
    assert not pinned_root.exists()
    for relative in relative_campaigns:
        assert f"<{pinned_root / relative}>" in log_text
        assert f"</workspace/repo/{relative.as_posix()}>" in log_text
        assert str(remote_root / relative) in log_text
    assert "Bare-host realpath received a Docker-only path" not in result.stderr
    assert "forbidden-host-python" not in log_text
    assert "submit-campaign" in log_text


def test_production_plan_requires_only_selected_profile_evidence(tmp_path: Path) -> None:
    """Scope the readiness gate to the selected campaign profile."""
    cases = (
        ("transient_drying", True),
        ("steady_flow", False),
    )
    for index, (rejected_profile, expected_success) in enumerate(cases):
        root = tmp_path / f"case-{index}"
        root.mkdir()
        workflow, log, environment, _storage, _mirror = _harness(root)
        environment["FAKE_SMOKE_EVIDENCE_REJECT_PROFILE"] = rejected_profile

        result = _run(
            workflow,
            ["plan", str(_campaign(workflow)), *_remote_options()],
            environment,
        )

        assert (result.returncode == 0) is expected_success, result.stderr
        evidence_lines = [line for line in log.read_text(encoding="utf-8").splitlines() if line.startswith("technical-smoke-evidence-profile")]
        assert evidence_lines == ["technical-smoke-evidence-profile <steady_flow>"]


def test_combined_smoke_records_and_checks_both_profile_evidence(tmp_path: Path) -> None:
    """Keep the combined smoke responsible for both profile campaigns."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    environment["FAKE_GPU_ALWAYS_VALID"] = "true"

    result = _run(workflow, ["smoke", "--keep-cpu-source"], environment)

    assert result.returncode == 0, result.stderr
    evidence_lines = [line for line in log.read_text(encoding="utf-8").splitlines() if line.startswith("technical-smoke-evidence-profile")]
    assert evidence_lines == [
        "technical-smoke-evidence-profile <steady_flow>",
        "technical-smoke-evidence-profile <transient_drying>",
    ]


def test_failed_technical_smoke_publishes_no_profile_evidence(tmp_path: Path) -> None:
    """Keep a failed technical workflow from producing readiness evidence."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    environment["FAKE_GPU_ALWAYS_VALID"] = "true"
    environment["FAKE_BUILD_FAIL"] = "true"

    result = _run(workflow, ["smoke", "--keep-cpu-source"], environment)

    assert result.returncode != 0
    log_text = log.read_text(encoding="utf-8")
    assert "submit-campaign" in log_text
    assert "<finalize-technical-smoke-evidence>" not in log_text


def test_dirty_worktree_uses_clean_pinned_source_without_modifying_checkout(tmp_path: Path) -> None:
    """Ignore real staged, tracked, workflow, config, and untracked edits exactly."""
    workflow, log, environment, storage, _mirror = _harness(tmp_path)
    project, source_probe, commit = _initialize_real_repository(workflow, environment)
    campaign = _campaign(workflow)
    dirty_marker = "# DIRTY_GENERATION_CONFIG"
    campaign.write_text(
        campaign.read_text(encoding="utf-8") + f"\n{dirty_marker}\n",
        encoding="utf-8",
    )
    source_probe.write_text("dirty staged generation behavior", encoding="utf-8")
    _real_git(project, "add", source_probe.relative_to(project).as_posix())

    workflow_text = workflow.read_text(encoding="utf-8")
    workflow_anchor = "local_python() {\n  env GENERATION_GIT_COMMIT="
    parser_anchor = 'SUBCOMMAND="$1"\nshift\n'
    assert workflow_text.count(workflow_anchor) == 1
    assert workflow_text.count(parser_anchor) == 1
    dirty_workflow_text = workflow_text.replace(
        parser_anchor,
        "printf 'DIRTY PARSER EXECUTED\\n' >&2\nexit 87\nSUBCOMMAND=\"$1\"\nshift\n",
    ).replace(
        workflow_anchor,
        "local_python() {\n  printf 'DIRTY WORKFLOW EXECUTED\\n' >&2\n  return 86\n  env GENERATION_GIT_COMMIT=",
    )
    workflow.write_text(
        dirty_workflow_text,
        encoding="utf-8",
    )
    untracked_relative = Path("src/uncommitted_generation_behavior.py")
    untracked = project / untracked_relative
    untracked.write_text("raise RuntimeError('dirty source executed')\n", encoding="utf-8")

    status_before = _real_git(
        project,
        "--no-optional-locks",
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).stdout
    index_before = (project / ".git/index").read_bytes()
    file_bytes_before = {path: path.read_bytes() for path in (campaign, source_probe, workflow, untracked)}
    environment["FAKE_REJECT_CONFIG_TEXT"] = dirty_marker
    environment["FAKE_UNTRACKED_SOURCE_PATH"] = untracked_relative.as_posix()
    environment["FAKE_EXPECT_SOURCE_FILE"] = source_probe.relative_to(project).as_posix()
    environment["FAKE_EXPECT_SOURCE_TEXT"] = "committed generation behavior"
    forbidden_source_parent = storage / "01_generation/source-infrastructure"
    forbidden_source_parent.mkdir(parents=True)
    environment["TMPDIR"] = str(forbidden_source_parent)

    result = _run(
        workflow,
        [
            "plan",
            str(campaign),
            "--cpu-host",
            "cpu.example",
            "--remote-root",
            "/remote/generation root",
        ],
        environment,
    )

    assert result.returncode == 0, result.stderr
    assert "This operation requires a clean local worktree." not in result.stderr
    assert result.stderr.splitlines()[:2] == [
        f"Source: committed HEAD {commit}",
        "Local worktree: dirty; uncommitted changes ignored",
    ]
    assert (project / ".git/index").read_bytes() == index_before
    status_after = _real_git(
        project,
        "--no-optional-locks",
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).stdout
    assert status_after == status_before
    assert _real_git(project, "rev-parse", "HEAD").stdout.strip() == commit
    for path, expected in file_bytes_before.items():
        assert path.read_bytes() == expected

    log_text = log.read_text(encoding="utf-8")
    source_roots = _logged_source_roots(log_text)
    assert len(source_roots) == 1
    pinned_root = next(iter(source_roots))
    assert pinned_root != project
    assert not pinned_root.is_relative_to(storage)
    assert not pinned_root.exists()
    assert not any(forbidden_source_parent.iterdir())
    assert f"local-python-commit <{commit}>" in log_text
    assert f"<GENERATION_GIT_COMMIT={commit}>" in log_text
    assert f"<type=bind,source={storage},target=/workspace/storage>" in log_text
    assert log_text.count(f"git <-C {project} rev-parse HEAD>") == 1
    assert f"git <-C {project} checkout" not in log_text
    assert f"git <-C {project} reset" not in log_text
    assert f"git <-C {project} clean" not in log_text
    assert f"git <-C {project} stash" not in log_text
    assert "DIRTY PARSER EXECUTED" not in result.stderr
    assert "DIRTY WORKFLOW EXECUTED" not in result.stderr
    assert "Dirty tracked configuration reached local Python." not in result.stderr
    assert "Dirty untracked source reached local Python." not in result.stderr
    assert "Local Python did not receive the committed Generation source." not in result.stderr


def test_dirty_real_worktree_runs_the_motivating_smoke_command(tmp_path: Path) -> None:
    """Run paired Technical Smoke orchestration from committed source while dirty."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    project, source_probe, commit = _initialize_real_repository(workflow, environment)
    source_probe.write_text("dirty generation behavior", encoding="utf-8")
    untracked = project / "notes-in-progress.txt"
    untracked.write_text("continue local development\n", encoding="utf-8")
    environment["FAKE_EXPECT_SOURCE_FILE"] = source_probe.relative_to(project).as_posix()
    environment["FAKE_EXPECT_SOURCE_TEXT"] = "committed generation behavior"
    environment["FAKE_UNTRACKED_SOURCE_PATH"] = untracked.relative_to(project).as_posix()
    environment["FAKE_GPU_ALWAYS_VALID"] = "true"

    result = _run(workflow, ["smoke", "--keep-cpu-source"], environment)

    assert result.returncode == 0, result.stderr
    assert result.stderr.count(f"Source: committed HEAD {commit}") == 1
    assert result.stderr.count("Local worktree: dirty; uncommitted changes ignored") == 1
    assert source_probe.read_text(encoding="utf-8") == "dirty generation behavior"
    assert untracked.read_text(encoding="utf-8") == "continue local development\n"
    log_text = log.read_text(encoding="utf-8")
    assert log_text.count("submit-campaign") == 2
    assert "<finalize-technical-smoke-evidence>" in log_text
    assert "<finalize-real-smoke>" in log_text
    assert f"local-python-commit <{commit}>" in log_text
    assert "Dirty untracked source reached local Python." not in result.stderr
    assert "Local Python did not receive the committed Generation source." not in result.stderr


def test_source_remains_pinned_when_development_head_advances(tmp_path: Path) -> None:
    """Keep one clean invocation on commit A while the development checkout reaches B."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    project, source_probe, commit_a = _initialize_real_repository(workflow, environment)
    environment["FAKE_EXPECT_SOURCE_FILE"] = source_probe.relative_to(project).as_posix()
    environment["FAKE_EXPECT_SOURCE_TEXT"] = "committed generation behavior"
    ready = tmp_path / "source-ready"
    continuation = tmp_path / "source-continue"
    environment["FAKE_DOCKER_READY_FILE"] = str(ready)
    environment["FAKE_DOCKER_CONTINUE_FILE"] = str(continuation)
    command = [
        str(workflow),
        "plan",
        str(_campaign(workflow)),
        "--cpu-host",
        "cpu.example",
        "--remote-root",
        "/remote/generation root",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    try:
        pinned_root = Path(_wait_for_file(ready, process))
        assert _real_git(pinned_root, "rev-parse", "HEAD").stdout.strip() == commit_a
        assert (
            _real_git(
                pinned_root,
                "--no-optional-locks",
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ).stdout
            == ""
        )

        (project / "after_launch.txt").write_text("commit B\n", encoding="utf-8")
        _real_git(project, "add", "after_launch.txt")
        _real_git(project, "commit", "--quiet", "-m", "test: advance development head")
        commit_b = _real_git(project, "rev-parse", "HEAD").stdout.strip()
        assert commit_b != commit_a

        continuation.touch()
        stdout, stderr = process.communicate(timeout=15)
    finally:
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=5)

    assert process.returncode == 0, stderr
    assert json.loads(stdout.splitlines()[-1])["state"] == "planned"
    assert stderr.splitlines()[:2] == [
        f"Source: committed HEAD {commit_a}",
        "Local worktree: clean",
    ]
    assert not pinned_root.exists()
    log_text = log.read_text(encoding="utf-8")
    assert f"local-python-commit <{commit_a}>" in log_text
    assert f"local-python-commit <{commit_b}>" not in log_text
    assert log_text.count(f"git <-C {project} rev-parse HEAD>") == 1


def test_concurrent_workflows_own_independent_clean_sources(tmp_path: Path) -> None:
    """Let one invocation clean up while another exact source remains active."""
    workflow, _log, environment, _storage, _mirror = _harness(tmp_path)
    project, source_probe, commit = _initialize_real_repository(workflow, environment)
    command = [
        str(workflow),
        "plan",
        str(_campaign(workflow)),
        "--cpu-host",
        "cpu.example",
        "--remote-root",
        "/remote/generation root",
    ]

    first_log = tmp_path / "first-commands.log"
    second_log = tmp_path / "second-commands.log"
    first_ready = tmp_path / "first-ready"
    first_continue = tmp_path / "first-continue"
    first_environment = environment.copy()
    first_environment.update(
        {
            "FAKE_COMMAND_LOG": str(first_log),
            "FAKE_DOCKER_READY_FILE": str(first_ready),
            "FAKE_DOCKER_CONTINUE_FILE": str(first_continue),
            "FAKE_EXPECT_SOURCE_FILE": source_probe.relative_to(project).as_posix(),
            "FAKE_EXPECT_SOURCE_TEXT": "committed generation behavior",
        }
    )
    second_environment = environment.copy()
    second_environment.update(
        {
            "FAKE_COMMAND_LOG": str(second_log),
            "FAKE_EXPECT_SOURCE_FILE": source_probe.relative_to(project).as_posix(),
            "FAKE_EXPECT_SOURCE_TEXT": "committed generation behavior",
        }
    )

    first = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=first_environment,
    )
    try:
        first_root = Path(_wait_for_file(first_ready, first))
        assert first_root.is_dir()

        second = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=second_environment,
            timeout=15,
        )
        assert second.returncode == 0, second.stderr
        second_roots = _logged_source_roots(second_log.read_text(encoding="utf-8"))
        assert len(second_roots) == 1
        second_root = next(iter(second_roots))
        assert second_root != first_root
        assert not second_root.exists()
        assert first_root.is_dir()

        first_continue.touch()
        _first_stdout, first_stderr = first.communicate(timeout=15)
    finally:
        if first.poll() is None:
            first.terminate()
            first.communicate(timeout=5)

    assert first.returncode == 0, first_stderr
    assert not first_root.exists()
    assert f"local-python-commit <{commit}>" in first_log.read_text(encoding="utf-8")
    assert f"local-python-commit <{commit}>" in second_log.read_text(encoding="utf-8")


def test_source_setup_failure_reports_frozen_commit_before_remote_work(tmp_path: Path) -> None:
    """Expose the frozen commit and stop before remote work when setup fails."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    environment["TMPDIR"] = str(tmp_path / "missing-source-parent")

    result = _run(
        workflow,
        [
            "plan",
            str(_campaign(workflow)),
            "--cpu-host",
            "cpu.example",
            "--remote-root",
            "/remote/generation root",
        ],
        environment,
    )

    assert result.returncode == 1
    assert result.stderr.splitlines()[:2] == [
        f"Source: committed HEAD {_COMMIT}",
        "Local worktree: clean",
    ]
    assert "Could not resolve the temporary source parent." in result.stderr
    assert "ssh-start" not in log.read_text(encoding="utf-8")


def test_committed_parser_failure_cleans_its_pinned_source(tmp_path: Path) -> None:
    """Clean the bootstrap-owned source when committed argument parsing fails."""
    workflow, _log, environment, _storage, _mirror = _harness(tmp_path)
    _initialize_real_repository(workflow, environment)
    source_parent = tmp_path / "source temp"
    source_parent.mkdir()
    environment["TMPDIR"] = str(source_parent)

    result = _run(
        workflow,
        [
            "plan",
            str(_campaign(workflow)),
            "--unsupported-after-handoff",
        ],
        environment,
    )

    assert result.returncode == 2
    assert "Unsupported option: --unsupported-after-handoff" in result.stderr
    assert not any(source_parent.iterdir())


def test_source_cleanup_failure_warns_without_rewriting_workflow_success(tmp_path: Path) -> None:
    """Retain an owned source with its path when cleanup alone fails."""
    workflow, _log, environment, _storage, _mirror = _harness(tmp_path)
    fake_bin = Path(environment["PATH"].split(os.pathsep, maxsplit=1)[0])
    real_rm = shutil.which("rm", path=os.defpath)
    assert real_rm is not None
    _executable(
        fake_bin / "rm",
        r"""#!/usr/bin/env bash
set -euo pipefail
target="${@: -1}"
if [[ "${target}" == */generation-workflow-source.* ]]; then
  exit 74
fi
exec "${FAKE_REAL_RM}" "$@"
""",
    )
    source_parent = tmp_path / "source temp"
    source_parent.mkdir()
    environment["TMPDIR"] = str(source_parent)
    environment["FAKE_REAL_RM"] = real_rm

    result = _run(
        workflow,
        [
            "plan",
            str(_campaign(workflow)),
            "--cpu-host",
            "cpu.example",
            "--remote-root",
            "/remote/generation root",
        ],
        environment,
    )

    assert result.returncode == 0, result.stderr
    warning_prefix = "WARNING: could not remove pinned-source directory: "
    warnings = [line.removeprefix(warning_prefix) for line in result.stderr.splitlines() if line.startswith(warning_prefix)]
    assert len(warnings) == 1
    retained = Path(warnings[0])
    try:
        assert retained.is_dir()
        assert retained.parent == source_parent
    finally:
        shutil.rmtree(retained)


def test_exact_requested_commit_validation_survives_dirty_safe_source_resolution(tmp_path: Path) -> None:
    """Reject malformed and non-HEAD commits before remote workflow operations."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    _project, _source_probe, commit = _initialize_real_repository(workflow, environment)
    mismatch = ("f" * 40) if commit != ("f" * 40) else ("e" * 40)
    common = [
        "plan",
        str(_campaign(workflow)),
        "--cpu-host",
        "cpu.example",
        "--remote-root",
        "/remote/generation root",
    ]

    malformed = _run(workflow, [*common, "--git-commit", "short"], environment)
    mismatched = _run(workflow, [*common, "--git-commit", mismatch], environment)

    assert malformed.returncode == 2
    assert "Git commit must be one lowercase 40-character identifier." in malformed.stderr
    assert mismatched.returncode == 1
    assert "Requested commit differs from local HEAD." in mismatched.stderr
    assert "ssh-start" not in log.read_text(encoding="utf-8")


def test_repository_admission_rejects_escape_ambiguous_and_container_paths(tmp_path: Path) -> None:
    """Reject traversal, outside absolutes, symlinks, and foreign-domain paths."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    project = workflow.parent.parent
    outside = tmp_path / "outside.yaml"
    outside.write_text("schema_kind: outside\n", encoding="utf-8")
    symlink = project / "configs/generation/campaigns/steady_flow/escape.yaml"
    symlink.symlink_to(outside)
    arguments = (
        "../outside.yaml",
        "../../outside.yaml",
        str(outside),
        "./configs/generation/campaigns/steady_flow/family_generalization.yaml",
        str(symlink),
        "/workspace/repo/configs/generation/campaigns/steady_flow/family_generalization.yaml",
    )

    for campaign in arguments:
        result = _run(workflow, ["plan", campaign, *_remote_options()], environment)
        assert result.returncode == 2, (campaign, result.stderr)

    assert "ssh-start" not in log.read_text(encoding="utf-8")


def test_captured_launch_stops_when_login_preflight_fails(tmp_path: Path) -> None:
    """Keep captured launch output fail-closed on a prerequisite error."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    environment["FAKE_CPU_LOGIN_RSYNC_MISSING"] = "true"

    result = _run(
        workflow,
        ["launch", str(_campaign(workflow)), *_remote_options()],
        environment,
    )

    assert result.returncode != 0
    assert "CPU login prerequisite missing: rsync (blocks transfer)." in result.stderr
    log_text = log.read_text(encoding="utf-8")
    assert "generation_cpu_login_preflight.sh" in log_text
    assert "submit-campaign" not in log_text


def test_preflight_passes_admitted_checkout_to_relocated_worker(tmp_path: Path) -> None:
    """Bind the compute worker to the admitted repository and launch commit."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)

    result = _run(
        workflow,
        ["preflight", str(_campaign(workflow)), *_remote_options()],
        environment,
    )

    assert result.returncode == 0, result.stderr
    assert "job=12345 evidence=/remote/preflight/fixture" in result.stdout
    log_text = log.read_text(encoding="utf-8")
    assert '--export="ALL,GENERATION_GIT_COMMIT=${commit}"' in log_text
    assert '"${repository}/scripts/generation_cpu_smoke.sh"' in log_text
    assert '"${repository}" "${venv}"' in log_text


def test_setup_is_read_only_by_default_and_execute_is_explicit(tmp_path: Path) -> None:
    """Keep setup non-mutating by default and secure when explicitly executed."""
    workflow, log, environment, storage, _mirror = _harness(tmp_path)
    dry_run = _run(workflow, ["setup-cpu", *_remote_options()], environment)

    assert dry_run.returncode == 0, dry_run.stderr
    assert storage.is_dir()
    assert not any(storage.iterdir())
    dry_log = log.read_text(encoding="utf-8")
    assert "<--rm>" in dry_log
    assert "<--network>\n<none>" in dry_log
    assert ",target=/workspace/repo,readonly>" in dry_log
    assert ",target=/workspace/storage>" in dry_log

    execute = _run(workflow, ["setup-cpu", *_remote_options(), "--execute"], environment)
    assert execute.returncode == 0, execute.stderr
    log_text = log.read_text(encoding="utf-8")
    assert "<BatchMode=yes>" in log_text
    assert "checkout --detach" in log_text

    unsafe = _run(workflow, ["setup-cpu", "--cpu-host", "bad;host"], environment)
    assert unsafe.returncode == 2


def test_local_docker_failure_stops_before_remote_mutation(tmp_path: Path) -> None:
    """Stop setup before remote mutation when the required local container fails."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    environment["FAKE_DOCKER_FAIL"] = "true"

    result = _run(workflow, ["setup-cpu", *_remote_options()], environment)

    assert result.returncode == 1
    assert "ssh-start" not in log.read_text(encoding="utf-8")


def test_plan_launch_and_current_option_validation(tmp_path: Path) -> None:
    """Keep planning machine-readable, launchable, and strict for current options."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    plan = _run(
        workflow,
        ["plan", str(_campaign(workflow)), *_remote_options(), *_selection_options()],
        environment,
    )
    assert plan.returncode == 0, plan.stderr
    plan_record = json.loads(plan.stdout.splitlines()[-1])
    assert plan_record["filesystem_mutated"] is False

    logical_campaign = _campaign(workflow).relative_to(workflow.parent.parent).as_posix()
    logical_plan = _run(workflow, ["plan", logical_campaign, *_remote_options()], environment)
    assert logical_plan.returncode == 0, logical_plan.stderr

    configured = _run(
        workflow,
        [
            "plan",
            str(_campaign(workflow)),
            *_remote_options(),
            "--only-batch",
            "future.profile::batch",
        ],
        environment,
    )
    assert configured.returncode == 0, configured.stderr
    assert " future.profile::batch plan-campaign " in log.read_text(encoding="utf-8")

    incompatible = _run(
        workflow,
        [
            "plan",
            str(_campaign(workflow)),
            *_remote_options(),
            *_selection_options(),
            "--skip-extreme-family-ood",
        ],
        environment,
    )
    assert incompatible.returncode == 2

    launch = _run(
        workflow,
        ["launch", str(_campaign(workflow)), *_remote_options(), *_selection_options()],
        environment,
    )
    assert launch.returncode == 0, launch.stderr
    assert f'{{"campaign_run_id":"{_RUN_ID}"' in launch.stdout
    assert f"Campaign: {_RUN_ID}" in launch.stdout
    assert "submit-campaign" in log.read_text(encoding="utf-8")

    rejected = _run(
        workflow,
        ["plan", str(_campaign(workflow)), *_remote_options(), "--unsupported-option"],
        environment,
    )
    assert rejected.returncode == 2


def test_high_level_core_benchmark_preserves_transfer_contract(tmp_path: Path) -> None:
    """Exercise benchmark submission, publication, and one-variant recovery."""
    workflow, log, environment, _storage, mirror = _harness(tmp_path)
    environment["FAKE_LOGIN_PREFLIGHT_STDOUT"] = "true"
    relative = environment["FAKE_BENCHMARK_RELATIVE"]
    remote_directory = mirror / relative
    remote_directory.mkdir(parents=True)
    (remote_directory / "summary.json").write_text("{}\n", encoding="utf-8")

    result = _run(workflow, ["benchmark-cores", *_remote_options()], environment)

    assert result.returncode == 0, result.stderr
    assert f"benchmark_run_id={_BENCHMARK_RUN_ID}" in result.stdout
    assert "state=complete" in result.stdout
    assert remote_directory.is_dir()
    assert Path(environment["FAKE_BENCHMARK_PUBLISHED_FILE"]).is_file()
    log_text = log.read_text(encoding="utf-8")
    assert "submit-core-benchmark" in log_text
    assert "publish-transferred-core-benchmark" in log_text
    assert "<build-campaign-datasets>" not in log_text
    assert f"<--expected-inventory-sha256>\n<{_BENCHMARK_INVENTORY_SHA}>" in log_text
    assert f"<--expected-file-count>\n<{_BENCHMARK_FILE_COUNT}>" in log_text
    assert f"<--expected-size-bytes>\n<{_BENCHMARK_SIZE_BYTES}>" in log_text

    recovered = _run(
        workflow,
        ["benchmark-cores", "--variant", "cores_08", *_remote_options()],
        environment,
    )
    assert recovered.returncode == 0, recovered.stderr
    assert "<--variant>\n<cores_08>" in log.read_text(encoding="utf-8")


def test_core_benchmark_failure_does_not_finalize(tmp_path: Path) -> None:
    """Stop a failed benchmark before terminal publication."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    environment["FAKE_BENCHMARK_STATE"] = "retry_required"

    result = _run(workflow, ["benchmark-cores", *_remote_options()], environment)

    assert result.returncode != 0
    assert "<finalize-core-benchmark>" not in log.read_text(encoding="utf-8")


def test_collect_is_non_destructive_and_publication_failure_retains_staging(tmp_path: Path) -> None:
    """Protect safe transfer, source retention, and retryable failed publication."""
    workflow, log, environment, storage, mirror = _harness(tmp_path)
    environment["FAKE_LOGIN_PREFLIGHT_STDOUT"] = "true"
    source_directories = _seed_transfer(mirror, environment)
    collected = _run(
        workflow,
        ["collect", _RUN_ID, "--cpu-host", "cpu.example", "--remote-root", "/remote/generation root"],
        environment,
    )

    assert collected.returncode == 0, collected.stderr
    assert all((mirror / relative).is_dir() for relative in source_directories)
    log_text = log.read_text(encoding="utf-8")
    assert log_text.count("rsync-start") == 4
    assert f"<{storage}/01_generation/.state/transfer-staging/{_RUN_ID}.synthetic/>" in log_text
    assert "<cpu.example:/remote/generation root/storage/./" in log_text
    assert not any((storage / "01_generation/.state/transfer-staging").glob("*"))

    failed_root = tmp_path / "failed"
    failed_root.mkdir()
    failed_workflow, failed_log, failed_environment, failed_storage, failed_mirror = _harness(failed_root)
    _seed_transfer(failed_mirror, failed_environment)
    failed_environment["FAKE_PUBLISH_FAIL"] = "true"
    failed = _run(
        failed_workflow,
        ["collect", _RUN_ID, "--cpu-host", "cpu.example", "--remote-root", "/remote/generation root"],
        failed_environment,
    )

    assert failed.returncode == 1
    failed_text = failed_log.read_text(encoding="utf-8")
    assert "<publish-transferred-campaign>" in failed_text
    assert any((failed_storage / "01_generation/.state/transfer-staging").glob("*"))


def test_all_default_cleanup_orders_every_gate_and_keep_opt_out_retains_source(tmp_path: Path) -> None:
    """Protect default cleanup ordering and the explicit retention opt-out."""
    workflow, log, environment, _storage, mirror = _harness(tmp_path)
    source_directories = _seed_transfer(mirror, environment)
    complete = _run(
        workflow,
        ["all", str(_campaign(workflow)), *_remote_options(), *_selection_options()],
        environment,
    )

    assert complete.returncode == 0, complete.stderr
    assert all(not (mirror / relative).exists() for relative in source_directories)
    log_text = log.read_text(encoding="utf-8")
    positions = [
        log_text.index("<publish-transferred-campaign>"),
        log_text.index("<build-campaign-datasets>"),
        log_text.index("<prepare-all-workflow>"),
        log_text.index("<cpu-cleanup-authorization>"),
        log_text.index("cleanup-campaign-source"),
        log_text.index("<record-cpu-cleanup>"),
        log_text.rindex("<validate-all-workflow>"),
    ]
    assert positions == sorted(positions)

    retained_root = tmp_path / "retained"
    retained_root.mkdir()
    retained_workflow, _retained_log, retained_environment, _storage, retained_mirror = _harness(retained_root)
    retained_directories = _seed_transfer(retained_mirror, retained_environment)
    retained = _run(
        retained_workflow,
        [
            "all",
            str(_campaign(retained_workflow)),
            *_remote_options(),
            *_selection_options(),
            "--keep-cpu-source",
        ],
        retained_environment,
    )

    assert retained.returncode == 0, retained.stderr
    assert all((retained_mirror / relative).is_dir() for relative in retained_directories)


def test_failure_preserves_evidence_and_resume_is_idempotent(tmp_path: Path) -> None:
    """Preserve source/evidence on failure and reuse publication during resume."""
    workflow, log, environment, _storage, mirror = _harness(tmp_path)
    source_directories = _seed_transfer(mirror, environment)
    environment["FAKE_BUILD_FAIL"] = "true"
    failed = _run(
        workflow,
        ["all", str(_campaign(workflow)), *_remote_options(), *_selection_options()],
        environment,
    )

    assert failed.returncode != 0
    assert "dataset build" in failed.stderr.lower()
    assert "/failure.json" in failed.stderr
    assert "./scripts/generation_workflow.sh resume" in failed.stderr
    assert all((mirror / relative).is_dir() for relative in source_directories)
    assert Path(environment["FAKE_GPU_PUBLISHED_FILE"]).is_file()
    first_log = log.read_text(encoding="utf-8")
    assert not any(Path(environment["STORAGE_ROOT"]).joinpath("01_generation/.state/transfer-staging").glob("*"))

    environment["FAKE_BUILD_FAIL"] = "false"
    resumed = _run(
        workflow,
        ["resume", _RUN_ID, "--cpu-host", "cpu.example", "--remote-root", "/remote/generation root"],
        environment,
    )
    assert resumed.returncode == 0, resumed.stderr
    after_resume = log.read_text(encoding="utf-8")
    assert after_resume.count("rsync-start") == first_log.count("rsync-start")
    assert all(not (mirror / relative).exists() for relative in source_directories)

    build_count = after_resume.count("<build-campaign-datasets>")
    cleanup_count = after_resume.count("cleanup-campaign-source")
    repeated = _run(
        workflow,
        ["resume", _RUN_ID, "--cpu-host", "cpu.example", "--remote-root", "/remote/generation root"],
        environment,
    )
    assert repeated.returncode == 0, repeated.stderr
    final_log = log.read_text(encoding="utf-8")
    assert final_log.count("<build-campaign-datasets>") == build_count
    assert final_log.count("cleanup-campaign-source") == cleanup_count


def test_partial_remote_and_detached_modes_never_cleanup(tmp_path: Path) -> None:
    """Preserve source for partial failure and detached submit-only execution."""
    workflow, _log, environment, _storage, mirror = _harness(tmp_path)
    source_directories = _seed_transfer(mirror, environment)
    environment["FAKE_CAMPAIGN_STATE"] = "partially_failed"
    environment["FAKE_SOURCE_STATE"] = "partially_failed"
    partial = _run(
        workflow,
        ["all", str(_campaign(workflow)), *_remote_options(), *_selection_options()],
        environment,
    )

    assert partial.returncode != 0
    assert all((mirror / relative).is_dir() for relative in source_directories)

    detached_root = tmp_path / "detached"
    detached_root.mkdir()
    detached_workflow, detached_log, detached_environment, _storage, detached_mirror = _harness(detached_root)
    detached_directories = _seed_transfer(detached_mirror, detached_environment)
    detached = _run(
        detached_workflow,
        ["all", str(_campaign(detached_workflow)), *_remote_options(), *_selection_options(), "--detach"],
        detached_environment,
    )

    assert detached.returncode == 0, detached.stderr
    assert all((detached_mirror / relative).is_dir() for relative in detached_directories)
    detached_text = detached_log.read_text(encoding="utf-8")
    assert "submit-campaign" in detached_text
