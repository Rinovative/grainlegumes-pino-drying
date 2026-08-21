# ruff: noqa: E501, S101, S603, PLR2004
"""Host workflow lifecycle tests using fake Docker, remote, Slurm, and COMSOL."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

from src import generation
from src.generation.cli import cli_generation

pytestmark = pytest.mark.integration

_COMMIT = "a" * 40
_RUN_ID = "steady_flow_steady_flow_id_dataset_v1__0123456789abcdef"
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
    ssh_transport = scripts / "generation_ssh_transport.sh"
    shutil.copyfile(repository / "scripts/generation_ssh_transport.sh", ssh_transport)
    ssh_transport.chmod(ssh_transport.stat().st_mode | 0o111)
    docker_python = scripts / "docker_python.sh"
    shutil.copyfile(repository / "scripts/docker_python.sh", docker_python)
    docker_python.chmod(docker_python.stat().st_mode | 0o111)
    for relative in (
        "configs/generation/campaigns/steady_flow/id_dataset.yaml",
        "configs/generation/campaigns/steady_flow/technical_smoke.yaml",
        "configs/generation/campaigns/transient_drying/family_generalization.yaml",
        "configs/generation/campaigns/transient_drying/technical_smoke.yaml",
        "configs/generation/campaigns/transient_drying/material_pilot.yaml",
        "configs/generation/workflows/technical_smoke.yaml",
        "configs/generation/execution/cluster_cpu.yaml",
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
        fake_bin / "sleep",
        r"""#!/usr/bin/env bash
set -euo pipefail
if [[ "${FAKE_BYPASS_SSH_RETRY_SLEEP:-false}" == true ]]; then
  case "${1:-}" in
    5|15|30|60)
      printf '%s\n' "$1" >> "${FAKE_SSH_RETRY_SLEEP_LOG}"
      exit 0
      ;;
  esac
fi
exec "${FAKE_REAL_SLEEP}" "$@"
""",
    )
    _executable(
        fake_bin / "tmux",
        r"""#!/usr/bin/env bash
set -euo pipefail
printf 'tmux <%s>\n' "$*" >> "${FAKE_COMMAND_LOG}"
case "${1:-}" in
  list-sessions)
    [[ ! -f "${FAKE_TMUX_SESSION_FILE}" ]] || cat "${FAKE_TMUX_SESSION_FILE}"
    ;;
  new-session)
    session=''
    arguments=("$@")
    for ((index=0; index<${#arguments[@]}; index++)); do
      [[ "${arguments[index]}" != -s ]] || session="${arguments[index+1]}"
    done
    [[ -n "${session}" ]] || exit 2
    if [[ "${FAKE_TMUX_IMMEDIATE_EXIT:-false}" != true ]]; then
      printf '%s\n' "${session}" > "${FAKE_TMUX_SESSION_FILE}"
    fi
    count=0
    [[ ! -f "${FAKE_TMUX_START_COUNT_FILE}" ]] || read -r count < "${FAKE_TMUX_START_COUNT_FILE}"
    printf '%s\n' "$((count + 1))" > "${FAKE_TMUX_START_COUNT_FILE}"
    ;;
  has-session)
    [[ -f "${FAKE_TMUX_SESSION_FILE}" ]]
    ;;
  display-message)
    printf '%s\n' '4242'
    ;;
  *) exit 2 ;;
esac
""",
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
    commit="${arguments[${#arguments[@]}-1]}"
    cp -a -- "${FAKE_COMMITTED_ROOT}/." "${root}/"
    mkdir -p "${root}/.git"
    printf '%s\n' "${commit}" > "${root}/.git/fake-head"
    ;;
  *" fetch --quiet --depth=1 --no-tags "*) ;;
  *" rev-parse --show-toplevel "*) git_root ;;
  *" rev-parse --verify "*)
    requested="${arguments[${#arguments[@]}-1]}"
    requested="${requested%\^\{commit\}}"
    [[ "${requested}" != "${FAKE_UNAVAILABLE_GIT_COMMIT:-}" ]] || exit 1
    printf '%s\n' "${requested}"
    ;;
  *" rev-parse HEAD "*)
    root="$(git_root)"
    if [[ -f "${root}/.git/fake-head" ]]; then
      cat "${root}/.git/fake-head"
    else
      printf '%s\n' "${FAKE_GIT_COMMIT}"
    fi
    ;;
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
if [[ "${FAKE_SSH_FAILURE_PHASE:-before}" != after \
  && -n "${FAKE_SSH_FAILURE_MATCH:-}" \
  && " $*"$'\n'"${payload}" == *"${FAKE_SSH_FAILURE_MATCH}"* ]]; then
  failure_count=0
  [[ ! -f "${FAKE_SSH_FAILURE_COUNT_FILE}" ]] ||
    read -r failure_count < "${FAKE_SSH_FAILURE_COUNT_FILE}"
  if (( failure_count < FAKE_SSH_FAILURE_LIMIT )); then
    printf '%s\n' "$((failure_count + 1))" > "${FAKE_SSH_FAILURE_COUNT_FILE}"
    printf 'ssh-injected-failure <%s>\n' "${FAKE_SSH_FAILURE_MATCH}" >> "${FAKE_COMMAND_LOG}"
    printf '%s\n' "${FAKE_SSH_FAILURE_MESSAGE}" >&2
    exit "${FAKE_SSH_FAILURE_STATUS}"
  fi
fi
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
if [[ "${FAKE_SETUP_IDLE_REJECT:-false}" == true \
  && -f "${FAKE_SETUP_INSTALLED_FILE}" \
  && "${payload}" == *'assert-shared-setup-idle'* ]]; then
  printf 'setup-idle-check <established>\n' >> "${FAKE_COMMAND_LOG}"
  printf '%s\n' '{"status":"active_dependent_jobs"}' >&2
  exit 65
fi
if [[ "${payload}" == *'CPU setup complete: %s'* ]]; then
  : > "${FAKE_SETUP_INSTALLED_FILE}"
fi
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
if [[ " $* " == *' campaign-status '* || " $* " == *' resume-campaign '* \
  || " $* " == *' feed-campaign '* ]]; then
  if [[ " $* " != *" ${FAKE_GIT_COMMIT} "* ]]; then
    printf '%s\n' 'Remote Generation operation omitted the pinned commit argument.' >&2
    exit 76
  fi
  if [[ "${payload}" != *'commit="$4"'* ]]; then
    printf '%s\n' 'Remote Generation operation did not bind its commit argument.' >&2
    exit 77
  fi
  if [[ "${payload}" != *'export GENERATION_GIT_COMMIT="${commit}"'* ]]; then
    printf '%s\n' 'Remote Generation operation omitted launcher provenance.' >&2
    exit 78
  fi
fi
if [[ "${payload}" == *'${HOME}'* && "${payload}" != *'root="$1"'* ]]; then
  printf '%s\n' '/remote/home'
elif [[ " $* " == *' core-benchmark-status '* && " $* " == *' --format state '* ]]; then
  printf '%s\n' "${FAKE_BENCHMARK_STATE}"
elif [[ " $* " == *' core-benchmark-status '* && " $* " == *' --format monitor '* ]]; then
  failed=0
  [[ "${FAKE_BENCHMARK_STATE}" == complete ]] || failed=1
  signature=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  printf 'campaign-monitor\t%s\t%s\t%s\n' "${FAKE_BENCHMARK_STATE}" "${signature}" "${signature}"
  printf 'Benchmark: transient_core_scaling\nRun: %s\nState: %s\n' \
    "${FAKE_BENCHMARK_RUN_ID}" "${FAKE_BENCHMARK_STATE}"
  printf 'Work units: successful=%s  running=0  scheduler_pending=0  license_blocked=0  not_admitted=0  failed=%s  total=8\n' \
    "$((8 - failed))" "${failed}"
elif [[ " $* " == *' core-benchmark-status '* && " $* " == *' --format summary '* ]]; then
  printf 'Benchmark: transient_core_scaling\nRun: %s\nState: %s\n' \
    "${FAKE_BENCHMARK_RUN_ID}" "${FAKE_BENCHMARK_STATE}"
elif [[ " $* " == *' core-benchmark-status '* ]]; then
  printf '{"state":"%s"}\n' "${FAKE_BENCHMARK_STATE}"
elif [[ " $* " == *' core-benchmark-transfer-plan '* ]]; then
  printf 'benchmark\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${FAKE_BENCHMARK_RUN_ID}" "${FAKE_GIT_COMMIT}" "${FAKE_BENCHMARK_RELATIVE}" \
    "${FAKE_BENCHMARK_INVENTORY_SHA}" "${FAKE_BENCHMARK_FILE_COUNT}" \
    "${FAKE_BENCHMARK_SIZE_BYTES}"
elif [[ " $* " == *' materialize-core-benchmark-inputs '* ]]; then
  printf '{"benchmark_run_id":"%s","state":"inputs_ready"}\n' "${FAKE_BENCHMARK_RUN_ID}"
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
elif [[ " $* " == *' campaign-transfer-authority '* ]]; then
  [[ "${FAKE_COMPATIBLE_SMOKE_STATUS:-missing}" == compatible_repairable ]] || exit 4
  printf '{"schema_kind":"generation_campaign_transfer_authority","schema_version":1,'
  printf '"campaign_run_id":"%s","campaign_id":"synthetic-campaign",' "${FAKE_RUN_ID}"
  printf '"git_commit":"%s","file_count":5,"size_bytes":5,' "${FAKE_GIT_COMMIT}"
  printf '"inventory_sha256":"%s"}\n' "${FAKE_INVENTORY_SHA}"
elif [[ " $* " == *' campaign-transfer-plan '* ]]; then
  printf '%b' "${FAKE_TRANSFER_PLAN}"
elif [[ " $* " == *' campaign-source-status '* ]]; then
  if [[ -f "${FAKE_REMOTE_CLEANED_FILE}" ]]; then
    printf 'source-status\t%s\tcomplete\tcleaned\t0\talready_complete\tFalse\n' "${FAKE_RUN_ID}"
  else
    printf 'source-status\t%s\trunning\t%s\t%s\tineligible\tFalse\n' \
      "${FAKE_RUN_ID}" "${FAKE_SOURCE_STATE}" "${FAKE_AUTHORIZED_BYTES}"
  fi
elif [[ ( " $* " == *' campaign-status '* || " $* " == *' resume-campaign '* ) \
  && "${FAKE_CAMPAIGN_STATUS_FAIL:-false}" == true ]]; then
  printf '%s\n' 'Synthetic initial campaign status failure.' >&2
  exit 79
elif [[ " $* " == *' campaign-status '* && " $* " == *' --format monitor '* ]]; then
  state="$(next_campaign_state)"
  case "${state}" in
    feeding) state_signature="$(printf '1%.0s' {1..64})" ;;
    running) state_signature="$(printf '2%.0s' {1..64})" ;;
    successful|transfer_complete) state_signature="$(printf '3%.0s' {1..64})" ;;
    completed_with_failures|cancelled) state_signature="$(printf '4%.0s' {1..64})" ;;
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
  printf 'Campaign: %s\nState: %s\nExecution: commit=%s  config_digest=%s\n' "${FAKE_RUN_ID}" "${state}" "${FAKE_GIT_COMMIT}" "${FAKE_INVENTORY_SHA}"
  printf 'Resources: cores_per_case=16  max_admission_cases=2  max_running_cases=null\n'
  printf 'Cases: 0/1 completed, 1 active, 0 pending, 0 failed\n\n'
  printf 'Active cases:\ncase_0001  job=591776  node=node-a  elapsed=00:01:00\n'
  printf '  phase=transient_drying  sim_time=%s h  step=0.075 s\n' "${progress_value}"
  printf '  order=2  Tfail=1  NLfail=3  updated=4 s ago\n'
elif [[ ( " $* " == *' campaign-status '* || " $* " == *' resume-campaign '* ) \
  && " $* " == *' --format workflow-monitor '* ]]; then
  state="$(next_campaign_state)"
  case "${state}" in
    feeding) state_signature="$(printf '1%.0s' {1..64})" ;;
    running) state_signature="$(printf '2%.0s' {1..64})" ;;
    successful|transfer_complete) state_signature="$(printf '3%.0s' {1..64})" ;;
    completed_with_failures|cancelled) state_signature="$(printf '4%.0s' {1..64})" ;;
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
  printf 'source-monitor\t%s\t%s\t%s\tunavailable\tineligible\tFalse\n' \
    "${FAKE_RUN_ID}" "${state}" "${FAKE_SOURCE_STATE}"
  printf 'Campaign: %s\nState: %s\nExecution: commit=%s  config_digest=%s\n' \
    "${FAKE_RUN_ID}" "${state}" "${FAKE_GIT_COMMIT}" "${FAKE_INVENTORY_SHA}"
  printf 'Resources: cores_per_case=16  max_admission_cases=2  max_running_cases=null\n'
  printf 'Cases: 0/1 completed, 1 active, 0 pending, 0 failed\n\n'
  printf 'Active cases:\ncase_0001  job=591776  node=node-a  elapsed=00:01:00\n'
  printf '  phase=transient_drying  sim_time=%s h  step=0.075 s\n' "${progress_value}"
  printf '  order=2  Tfail=1  NLfail=3  updated=4 s ago\n'
elif [[ " $* " == *' campaign-status '* && " $* " == *' --format summary '* ]]; then
  printf 'Campaign: %s\nState: %s\nExecution: commit=%s  config_digest=%s\n' \
    "${FAKE_RUN_ID}" "${FAKE_CAMPAIGN_STATE}" "${FAKE_GIT_COMMIT}" "${FAKE_INVENTORY_SHA}"
  printf 'Resources: cores_per_case=16  max_admission_cases=2  max_running_cases=null\n'
  printf 'Cases: 0/1 completed, 0 active, 1 pending, 0 failed\n\n'
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
  printf '%s\n' '{"state":"active"}'
elif [[ " $* " == *' campaign-accounting '* ]]; then
  printf '%s\n' '{"squeue":{"output":"12345_0|RUNNING|node-a"}}'
elif [[ " $* " == *' cancel-campaign '* ]]; then
  if [[ " $* " != *' --force'* && "${FAKE_GRACEFUL_CANCEL_DELAY_SECONDS:-0}" != 0 ]]; then
    sleep "${FAKE_GRACEFUL_CANCEL_DELAY_SECONDS}"
  fi
  printf '%s\n' '{"status":"cancel_requested"}'
elif [[ " $* " == *' prepare-campaign-inputs'* ]]; then
  if [[ "${FAKE_INPUT_PREPARATION_DELAY_SECONDS:-0}" != 0 ]]; then
    sleep "${FAKE_INPUT_PREPARATION_DELAY_SECONDS}"
  fi
  if [[ "${FAKE_INPUT_PREPARATION_FAIL:-false}" == true ]]; then
    printf '%s\n' 'Canonical inputs invalid for synthetic batch.' >&2
    exit 73
  fi
  printf 'canonical-inputs\t%s\t%s\n' \
    "${FAKE_INPUT_GENERATED_COUNT:-1}" "${FAKE_INPUT_REUSED_COUNT:-0}"
elif [[ " $* " == *' submit-campaign'* ]]; then
  if [[ "${payload}" != *'export GENERATION_GIT_COMMIT="${commit}"'* ]]; then
    printf '%s\n' 'Remote submit-campaign omitted GENERATION_GIT_COMMIT launcher provenance.' >&2
    exit 74
  fi
  if [[ "${payload}" != *'--git-commit "${commit}"'* ]]; then
    printf '%s\n' 'Remote submit-campaign omitted the matching explicit Git commit.' >&2
    exit 75
  fi
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
if [[ "${FAKE_SSH_FAILURE_PHASE:-before}" == after \
  && -n "${FAKE_SSH_FAILURE_MATCH:-}" \
  && " $*"$'\n'"${payload}" == *"${FAKE_SSH_FAILURE_MATCH}"* ]]; then
  failure_count=0
  [[ ! -f "${FAKE_SSH_FAILURE_COUNT_FILE}" ]] ||
    read -r failure_count < "${FAKE_SSH_FAILURE_COUNT_FILE}"
  if (( failure_count < FAKE_SSH_FAILURE_LIMIT )); then
    printf '%s\n' "$((failure_count + 1))" > "${FAKE_SSH_FAILURE_COUNT_FILE}"
    printf 'ssh-injected-failure-after <%s>\n' "${FAKE_SSH_FAILURE_MATCH}" >> "${FAKE_COMMAND_LOG}"
    printf '%s\n' "${FAKE_SSH_FAILURE_MESSAGE}" >&2
    exit "${FAKE_SSH_FAILURE_STATUS}"
  fi
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
if [[ " $* " == *' resolve-generation-run '* ]]; then
  arguments=("$@")
  config=''
  for ((index=0; index<${#arguments[@]}; index++)); do
    if [[ "${arguments[index]}" == resolve-generation-run ]]; then
      config="${arguments[index+1]}"
      break
    fi
  done
  case "${config}" in
    *'/configs/generation/workflows/technical_smoke.yaml')
      : > "${FAKE_SMOKE_WORKFLOW_FILE}"
      printf '%s\n' workflow configs/generation/workflows/technical_smoke.yaml - - > "${FAKE_RUN_PLAN_FILE}"
      printf '%s\n' '{"run_kind":"workflow","identity":"workflow__0123456789abcdef","config_path":"configs/generation/workflows/technical_smoke.yaml","children":[],"units":[]}'
      ;;
    *'/configs/generation/benchmarks/transient_core_scaling/suite.yaml')
      printf '%s\n' benchmark configs/generation/benchmarks/transient_core_scaling/suite.yaml - - > "${FAKE_RUN_PLAN_FILE}"
      printf '%s\n' '{"run_kind":"benchmark","identity":"benchmark_plan__0123456789abcdef","config_path":"configs/generation/benchmarks/transient_core_scaling/suite.yaml","children":[],"units":[]}'
      ;;
    *'/configs/generation/campaigns/transient_drying/material_pilot.yaml')
      printf '%s\n' campaign configs/generation/campaigns/transient_drying/material_pilot.yaml pilot_check transient_drying > "${FAKE_RUN_PLAN_FILE}"
      printf '%s\n' '{"run_kind":"campaign","identity":"steady_flow_steady_flow_id_dataset_v1__0123456789abcdef","config_path":"configs/generation/campaigns/transient_drying/material_pilot.yaml","children":[],"units":[{"metadata":{"campaign_purpose":"pilot_check","simulation_profile":"transient_drying"}}]}'
      ;;
    *'/configs/generation/campaigns/'*'/technical_smoke.yaml')
      profile=steady_flow
      [[ "${config}" != *'/transient_drying/'* ]] || profile=transient_drying
      relative="configs/generation/campaigns/${profile}/technical_smoke.yaml"
      printf '%s\n' campaign "${relative}" technical_runtime_smoke "${profile}" > "${FAKE_RUN_PLAN_FILE}"
      printf '{"run_kind":"campaign","identity":"%s","config_path":"%s","children":[],"units":[{"metadata":{"campaign_purpose":"technical_runtime_smoke","simulation_profile":"%s"}}]}\n' "${FAKE_RUN_ID}" "${relative}" "${profile}"
      ;;
    *'/configs/generation/campaigns/transient_drying/family_generalization.yaml')
      printf '%s\n' campaign configs/generation/campaigns/transient_drying/family_generalization.yaml family_generalization transient_drying > "${FAKE_RUN_PLAN_FILE}"
      printf '%s\n' '{"run_kind":"campaign","identity":"steady_flow_steady_flow_id_dataset_v1__0123456789abcdef","config_path":"configs/generation/campaigns/transient_drying/family_generalization.yaml","children":[],"units":[{"metadata":{"campaign_purpose":"family_generalization","simulation_profile":"transient_drying"}}]}'
      ;;
    *)
      printf '%s\n' campaign configs/generation/campaigns/steady_flow/id_dataset.yaml steady_flow_id_dataset steady_flow > "${FAKE_RUN_PLAN_FILE}"
      printf '%s\n' '{"run_kind":"campaign","identity":"steady_flow_steady_flow_id_dataset_v1__0123456789abcdef","config_path":"configs/generation/campaigns/steady_flow/id_dataset.yaml","children":[],"units":[{"metadata":{"campaign_purpose":"steady_flow_id_dataset","simulation_profile":"steady_flow"}}]}'
      ;;
  esac
  exit 0
fi
if [[ " $* " == *' find-compatible-technical-smoke-run '* ]]; then
  compatible_smoke_status="${FAKE_COMPATIBLE_SMOKE_STATUS:-missing}"
  if [[ "${compatible_smoke_status}" == missing ]]; then
    printf '%s\n' '{"status":"missing","campaign_run_id":null}'
  else
    printf '{"status":"%s","campaign_run_id":"%s"}\n' \
      "${compatible_smoke_status}" "${FAKE_RUN_ID}"
  fi
  exit 0
fi
if [[ " $* " == *' find-compatible-campaign-source '* ]]; then
  compatible_state="${FAKE_COMPATIBLE_CAMPAIGN_PACKAGE_STATE:-missing}"
  if [[ "${compatible_state}" == missing     && -f "${FAKE_DATASETS_COMPLETE_FILE}"     && -f "${FAKE_WORKFLOW_COMPLETE_FILE}"     && ! -f "${FAKE_PACKAGE_STATE_READY_FILE}" ]]; then
    compatible_state=extension_required
  fi
  if [[ "${compatible_state}" == missing ]]; then
    printf '%s\n' '{"status":"missing","campaign_run_id":null}'
  else
    printf '{"status":"compatible_complete","campaign_run_id":"%s",' "${FAKE_RUN_ID}"
    printf '"package_state":"%s","artifact_set_sha256":"%s"}\n' \
      "${compatible_state}" "${FAKE_INVENTORY_SHA}"
  fi
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
  [[ " $* " != *'/material_pilot.yaml'* ]] || purpose=pilot_check
  printf '%s\n' "${purpose}" > "${FAKE_CAMPAIGN_PURPOSE_FILE}"
fi
if [[ " $* " == *' resolve-core-benchmark-run '* ]]; then
  printf '{"benchmark_run_id":"%s"}\n' "${FAKE_BENCHMARK_RUN_ID}"
  exit 0
fi
if [[ " $* " == *' inspect-core-benchmark '* ]]; then
  printf '%s\n' \
    '{"suite_name":"transient_core_scaling",'\
'"suite_digest":"8888888888888888888888888888888888888888888888888888888888888888",'\
'"benchmark_mode":"core_selection","parallel_cases_per_variant":2,'\
'"required_successful_measurements":8,'\
'"representative_cases":[{"case_role":"nominal"},{"case_role":"natural"}],'\
'"resource_contract":{"cpu_host":"fixture.cluster","scheduler":"slurm",'\
'"partition":"standard","cores_per_node":32,"python_module":"Python/3.10",'\
'"comsol_module":"Comsol/v6.4","python_executable":"python","comsol_executable":"comsol","poll_interval_seconds":1},'\
'"variant_waves":[{"wave_position":1,"variant_id":"cores_16","cores_per_case":16},'\
'{"wave_position":2,"variant_id":"cores_04","cores_per_case":4},'\
'{"wave_position":3,"variant_id":"cores_08","cores_per_case":8},'\
'{"wave_position":4,"variant_id":"cores_32","cores_per_case":32}]}'
  exit 0
fi
if [[ " $* " == *' create-background-session '* ]]; then
  storage=''
  child=false
  : > "${FAKE_BACKGROUND_CHILD_ARGUMENTS_FILE}"
  arguments=("$@")
  for ((index=0; index<${#arguments[@]}; index++)); do
    argument="${arguments[index]}"
    if [[ "${argument}" == --storage-root ]]; then
      storage="${arguments[index+1]}"
    elif [[ "${argument}" == -- ]]; then
      child=true
    elif [[ "${child}" == true ]]; then
      printf '%s\n' "${argument}" >> "${FAKE_BACKGROUND_CHILD_ARGUMENTS_FILE}"
    fi
  done
  session_id='gw-20260818T154501Z-run-01234567'
  tmux_name='gw-run-154501-01234567'
  directory="${storage}/01_generation/meta/workflow_sessions/${session_id}"
  mkdir -p "${directory}"
  command_path="${directory}/command.sh"
  log_path="${directory}/workflow.log"
  printf '%s\n' '#!/usr/bin/env bash' 'exit 0' > "${command_path}"
  chmod 700 "${command_path}"
  : > "${log_path}"
  printf '{"status":"created","workflow_session_id":"%s",' "${session_id}"
  printf '"tmux_session_name":"%s","source_commit":"%s",' \
    "${tmux_name}" "${FAKE_GIT_COMMIT}"
  printf '"log_path":"%s","command_path":"%s",' \
    "${log_path}" "${command_path}"
  printf '%s\n' '"host":"synthetic-host"}'
  exit 0
fi
if [[ " $* " == *' inspect-background-session '* ]]; then
  printf '{"workflow_session_id":"gw-20260818T154501Z-run-01234567",'
  printf '"source_commit":"%s","subcommand":"run",' "${FAKE_GIT_COMMIT}"
  if [[ "${FAKE_TMUX_IMMEDIATE_EXIT:-false}" == true ]]; then
    printf '"tmux_session_name":"gw-run-154501-01234567","tmux_active":false,'
    printf '"workflow_state":"completed","exit_code":0,'
    printf '"started_at":"2026-08-18T15:45:01+00:00","ended_at":"2026-08-18T15:45:02+00:00",'
    printf '"campaign_run_ids":[],"benchmark_run_ids":[],"final_stage":"DONE: synthetic",'
  else
    printf '"tmux_session_name":"gw-run-154501-01234567","tmux_active":true,'
    printf '"workflow_state":"running","exit_code":null,'
    printf '"started_at":"2026-08-18T15:45:01+00:00","ended_at":null,'
    printf '"campaign_run_ids":[],"benchmark_run_ids":[],"final_stage":"running",'
  fi
  printf '"log_path":"%s/01_generation/meta/workflow_sessions/' "${STORAGE_ROOT}"
  printf '%s\n' 'gw-20260818T154501Z-run-01234567/workflow.log"}'
  exit 0
fi
if [[ " $* " == *' list-background-sessions '* ]]; then
  printf '%s\n' '{"sessions":[]}'
  exit 0
fi
if [[ " $* " == *' -c '* ]]; then
  payload="$(cat)"
  if [[ " $* " == *'plan = json.load'* ]]; then
    mapfile -t plan < "${FAKE_RUN_PLAN_FILE}"
    kind="${plan[0]}"; config="${plan[1]}"; purpose="${plan[2]}"; profile="${plan[3]}"
    if [[ "${kind}" == workflow ]]; then
      printf 'plan\tworkflow\tworkflow__0123456789abcdef\t%s\t-\t-\t2\n' "${config}"
      printf 'child\tconfigs/generation/campaigns/steady_flow/technical_smoke.yaml\t%s\ttechnical_runtime_smoke\tsteady_flow\t%s\n' "${FAKE_RUN_ID}" "${FAKE_TRANSFER_SHA}"
      printf 'child\tconfigs/generation/campaigns/transient_drying/technical_smoke.yaml\t%s\ttechnical_runtime_smoke\ttransient_drying\t%s\n' "${FAKE_RUN_ID}" "${FAKE_DATASET_SHA}"
    elif [[ "${kind}" == benchmark ]]; then
      printf 'plan\tbenchmark\tbenchmark_plan__0123456789abcdef\t%s\t-\t-\t0\n' "${config}"
    else
      printf 'plan\tcampaign\t%s\t%s\t%s\t%s\t0\n' "${FAKE_RUN_ID}" "${config}" "${purpose}" "${profile}"
    fi
  elif [[ " $* " == *'value["campaign_run_id"] is None'*     && " $* " != *'value.get("package_state"'* ]]; then
    compatible_smoke_status="${FAKE_COMPATIBLE_SMOKE_STATUS:-missing}"
    compatible_smoke_run='-'
    [[ "${compatible_smoke_status}" == missing ]] || compatible_smoke_run="${FAKE_RUN_ID}"
    printf '%s\t%s\n' "${compatible_smoke_status}" "${compatible_smoke_run}"
  elif [[ " $* " == *'value.get("package_state"'* ]]; then
    compatible_state="${FAKE_COMPATIBLE_CAMPAIGN_PACKAGE_STATE:-missing}"
    if [[ "${compatible_state}" == missing \
      && -f "${FAKE_DATASETS_COMPLETE_FILE}" \
      && -f "${FAKE_WORKFLOW_COMPLETE_FILE}" \
      && ! -f "${FAKE_PACKAGE_STATE_READY_FILE}" ]]; then
      compatible_state=extension_required
    fi
    if [[ "${compatible_state}" == missing ]]; then
      printf 'missing\t-\t-\t-\n'
    else
      printf 'compatible_complete\t%s\t%s\t%s\n' \
        "${FAKE_RUN_ID}" "${compatible_state}" "${FAKE_INVENTORY_SHA}"
    fi
  elif [[ " $* " == *'reason = str(value.get'* ]]; then
    if [[ "${FAKE_CAMPAIGN_STATE}" == completed_with_failures ]]; then
      printf 'incomplete\tpartial campaign completion\t%s\n' \
        "${FAKE_DECLARED_PACKAGE_COUNT:-1}"
    else
      printf 'complete\t-\t%s\n' "${FAKE_DECLARED_PACKAGE_COUNT:-1}"
    fi
  elif [[ " $* " == *'keys = ("status"'* ]]; then
    printf 'created\tgw-20260818T154501Z-run-01234567\tgw-run-154501-01234567\t%s\t' \
      "${FAKE_GIT_COMMIT}"
    printf '%s/01_generation/meta/workflow_sessions/' "${STORAGE_ROOT}"
    printf 'gw-20260818T154501Z-run-01234567/workflow.log\t'
    printf '%s/01_generation/meta/workflow_sessions/' "${STORAGE_ROOT}"
    printf 'gw-20260818T154501Z-run-01234567/command.sh\tsynthetic-host\n'
  elif [[ " $* " == *'fields = (value["workflow_state"]'* ]]; then
    if [[ "${FAKE_TMUX_IMMEDIATE_EXIT:-false}" == true ]]; then
      printf 'completed\t0\tDONE: synthetic\n'
    else
      printf 'running\t-\trunning\n'
    fi
  elif [[ " $* " == *'keys = ("workflow_session_id"'* ]]; then
    printf 'gw-20260818T154501Z-run-01234567\t%s\trun\t' "${FAKE_GIT_COMMIT}"
    printf 'gw-run-154501-01234567\ttrue\trunning\t-\t'
    printf '2026-08-18T15:45:01+00:00\t-\t-\t-\trunning\t'
    printf '%s/01_generation/meta/workflow_sessions/' "${STORAGE_ROOT}"
    printf 'gw-20260818T154501Z-run-01234567/workflow.log\n'
  elif [[ " $* " == *'sessions = json.load'* ]]; then
    printf '%s\n' 'No background workflow sessions.'
  elif [[ " $* " == *'waits = value.get("license_waits"'* ]]; then
    failed=0
    [[ "${FAKE_BENCHMARK_STATE}" == complete ]] || failed=1
    printf '%s\t%s\t0\t0\t0\t%s\t8\t-\n' \
      "${FAKE_BENCHMARK_STATE}" "$((8 - failed))" "${failed}"
  elif [[ " $* " == *'["benchmark_run_id"]'* ]]; then
    printf '%s\n' "${FAKE_BENCHMARK_RUN_ID}"
  elif [[ " $* " == *'resource = value'* ]]; then
    printf 'benchmark\ttransient_core_scaling\t%s\t8\t2\t16,4,8,32\tfixture.cluster\t'\
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
      "${catalog_prefix}configs/generation/campaigns/steady_flow/id_dataset.yaml" \
      "${catalog_prefix}configs/generation/campaigns/transient_drying/family_generalization.yaml"
  elif [[ " $* " == *'execution_resources'* ]]; then
    purpose="$(cat "${FAKE_CAMPAIGN_PURPOSE_FILE}")"
    printf 'execution\t%s\t16\t01:00:00\t48\t2\t1\t-\tfixture.cluster\tslurm\tfixture\t'\
'Python/fixture-3.12\tComsol/fixture-9.9\tfixture-python\tfixture-comsol\t1\t1\n' "${purpose}"
  elif [[ " $* " == *'value["completed_cases"]'* ]]; then
    printf 'deferred\t%s\t%s\t1\t0\t0\n' "${FAKE_RUN_ID}" "${FAKE_GIT_COMMIT}"
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
  if [[ "${FAKE_TARGET_VALIDATION_FAIL_AFTER_FIRST:-false}" == true \
    && " $* " == *'/smoke_receipts/current.json '* ]]; then
    validation_count=0
    [[ ! -f "${FAKE_TARGET_VALIDATION_COUNT_FILE}" ]] || \
      read -r validation_count < "${FAKE_TARGET_VALIDATION_COUNT_FILE}"
    printf '%s\n' "$((validation_count + 1))" > "${FAKE_TARGET_VALIDATION_COUNT_FILE}"
    (( validation_count == 0 )) || exit 4
  fi
  receipt="${storage}/01_generation/meta/smoke_receipts/current.json"
  mkdir -p "$(dirname "${receipt}")"
  printf '%s\n' '{}' > "${receipt}"
  printf '{"status":"valid","valid_receipts":[{'\
'"path":"/workspace/storage/01_generation/meta/smoke_receipts/current.json",'\
'"receipt_digest":"%s"}]}\n' "${FAKE_SMOKE_SHA}"
elif [[ " $* " == *' validate-core-benchmark '* ]]; then
  [[ -f "${FAKE_BENCHMARK_PUBLISHED_FILE}" ]]
elif [[ " $* " == *' validate-published-campaign '* ]]; then
  if [[ " $* " == *' --partial '* ]]; then
    [[ -f "${FAKE_GPU_PARTIAL_FILE}" ]]
  else
    [[ "${FAKE_GPU_ALWAYS_VALID:-false}" == true || -f "${FAKE_GPU_PUBLISHED_FILE}" ]]
  fi
elif [[ " $* " == *' repair-transferred-campaign '* ]]; then
  [[ "${FAKE_COMPATIBLE_SMOKE_STATUS:-missing}" == compatible_repairable ]] || exit 4
  : > "${FAKE_GPU_PUBLISHED_FILE}"
  printf '%s\n' '{"source_removed":false,"status":"transfer_complete"}'
elif [[ " $* " == *' create-transfer-staging '* ]]; then
  staging="${storage}/.incoming/${FAKE_RUN_ID}.synthetic"
  mkdir -p "${staging}"
  printf '%s\n' "${staging}"
elif [[ " $* " == *' publish-transferred-core-benchmark '* ]]; then
  : > "${FAKE_BENCHMARK_PUBLISHED_FILE}"
  printf '%s\n' '{"status":"transfer_complete","dataset_membership":"none"}'
elif [[ " $* " == *' core-benchmark-summary '* ]]; then
  printf '%s\n' '# Synthetic local core benchmark summary'
elif [[ " $* " == *' publish-transferred-campaign '* ]]; then
  if [[ "${FAKE_PUBLISH_DELAY_SECONDS:-0}" != 0 ]]; then
    sleep "${FAKE_PUBLISH_DELAY_SECONDS}"
  fi
  if [[ "${FAKE_PUBLISH_FAIL:-false}" == true ]]; then
    printf '%s\n' 'synthetic destination hash validation failed' >&2
    exit 4
  fi
  if [[ " $* " == *' --partial '* ]]; then
    : > "${FAKE_GPU_PARTIAL_FILE}"
    printf '%s\n' '{"source_removed":false,"status":"partial"}'
  else
    : > "${FAKE_GPU_PUBLISHED_FILE}"
    printf '%s\n' '{"source_removed":false,"status":"transfer_complete"}'
  fi
elif [[ " $* " == *' cleanup-transfer-staging '* ]]; then
  [[ -z "${directory}" || ! -d "${directory}" ]] || rm -r -- "${directory}"
  printf '%s\n' '{"mode":"delete"}'
elif [[ " $* " == *' cleanup-pilot-staging '* ]]; then
  printf 'pilot-staging-cleanup\tcomplete\tTrue\t0\t%s\n' "${FAKE_CLEANUP_RECEIPT_SHA}"
elif [[ " $* " == *' build-campaign-datasets '* ]]; then
  if [[ "${FAKE_BUILD_FAIL:-false}" == true ]]; then
    printf '%s\n' 'synthetic dataset build failed' >&2
    exit 5
  fi
  if [[ " $* " == *' --partial '* || "${FAKE_CAMPAIGN_STATE}" == completed_with_failures ]]; then
    printf '%s\n' '{"status":"incomplete","declared_package_count":1,"packages":[]}'
  else
    if [[ "${FAKE_COMPATIBLE_CAMPAIGN_PACKAGE_STATE:-missing}" != extension_required ]]; then
      : > "${FAKE_DATASETS_COMPLETE_FILE}"
    fi
    : > "${FAKE_PACKAGE_STATE_READY_FILE}"
    printf '{"status":"complete","declared_package_count":%s,"packages":[{"dataset_id":"synthetic"}]}\n' "${FAKE_DECLARED_PACKAGE_COUNT:-1}"
  fi
elif [[ " $* " == *' prepare-all-workflow '* ]]; then
  : > "${FAKE_WORKFLOW_READY_FILE}"
  if [[ " $* " == *' --keep-cpu-source '* \
    && ( ! -f "${FAKE_SMOKE_WORKFLOW_FILE}" || -f "${FAKE_PARENT_FINALIZED_FILE}" ) ]]; then
    : > "${FAKE_WORKFLOW_COMPLETE_FILE}"
    printf '%s\n' '{"workflow_result":"success","cpu_cleanup_complete":{"status":"skipped_by_request"}}'
  else
    printf '%s\n' '{"workflow_result":"ready_for_cpu_cleanup","cpu_cleanup_complete":{"status":"pending"}}'
  fi
elif [[ " $* " == *' validate-campaign-package-state '* ]]; then
  [[ -f "${FAKE_PACKAGE_STATE_READY_FILE}" ]]
  printf '%s\n' '{"status":"complete","packages":[{"dataset_id":"synthetic"}]}'
elif [[ " $* " == *' validate-all-workflow '* ]]; then
  [[ -f "${FAKE_WORKFLOW_COMPLETE_FILE}" ]]
  if [[ -f "${FAKE_SMOKE_WORKFLOW_FILE}" \
    && ! -f "${FAKE_PARENT_FINALIZED_FILE}" \
    && "${FAKE_PRESERVE_WORKFLOW_COMPLETE:-false}" != true ]]; then
    rm -f -- "${FAKE_WORKFLOW_COMPLETE_FILE}"
  fi
  printf '%s\n' '{"workflow_result":"success"}'
elif [[ " $* " == *' cpu-cleanup-authorization '* ]]; then
  authorization_destination="${FAKE_AUTH_DESTINATION_ROOT:-${storage}}"
  printf 'authorization\t%s\tcpu.example\t/remote/generation root/storage\t%s\t%s\t%s\t%s\t%s\t4\t%s\n' \
    "${FAKE_AUTHORIZATION_SHA}" "${authorization_destination}" "${FAKE_TRANSFER_SHA}" \
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
  if [[ "${FAKE_FINALIZE_SMOKE_DELAY_SECONDS:-0}" != 0 ]]; then
    sleep "${FAKE_FINALIZE_SMOKE_DELAY_SECONDS}"
  fi
  if [[ "${FAKE_FINALIZE_SMOKE_FAIL:-false}" == true ]]; then
    exit 4
  fi
  : > "${FAKE_PARENT_FINALIZED_FILE}"
  printf '%s\n' '/workspace/storage/01_generation/meta/smoke_receipts/current.json'
elif [[ " $* " == *' record-workflow-failure '* ]]; then
  run_id="${arguments[3]}"
  canonical="01_generation/meta/campaigns/${run_id}/workflow_failures/failure-0001.json"
  printf 'workflow-failure\t%s\t%s/%s\n' \
    "${canonical}" "${storage}" "${canonical}"
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
            "FAKE_RUN_PLAN_FILE": str(state_root / "run-plan"),
            "FAKE_SMOKE_WORKFLOW_FILE": str(state_root / "smoke-workflow"),
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
            "FAKE_REAL_SLEEP": shutil.which("sleep", path=os.defpath) or "/usr/bin/sleep",
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
            "FAKE_CAMPAIGN_STATE": "successful",
            "FAKE_COMPATIBLE_CAMPAIGN_PACKAGE_STATE": "missing",
            "FAKE_CAMPAIGN_STATUS_FAIL": "false",
            "FAKE_CAMPAIGN_STATES": "",
            "FAKE_CAMPAIGN_PROGRESS_SIGNATURES": "",
            "FAKE_PROGRESS_VALUES": "",
            "FAKE_CAMPAIGN_STATE_INDEX_FILE": str(state_root / "campaign-state-index"),
            "FAKE_CONSOLE_TIMES": "",
            "FAKE_CONSOLE_TIME_INDEX_FILE": str(state_root / "console-time-index"),
            "FAKE_SOURCE_STATE": "successful",
            "FAKE_AUTHORIZED_BYTES": str(_AUTHORIZED_BYTES),
            "FAKE_AUTHORIZATION_SHA": _AUTHORIZATION_SHA,
            "FAKE_TRANSFER_SHA": _TRANSFER_SHA,
            "FAKE_DATASET_SHA": _DATASET_SHA,
            "FAKE_WORKFLOW_SHA": _WORKFLOW_SHA,
            "FAKE_INVENTORY_SHA": _INVENTORY_SHA,
            "FAKE_CLEANUP_RECEIPT_SHA": _CLEANUP_RECEIPT_SHA,
            "FAKE_LOGIN_PREFLIGHT_STDOUT": "false",
            "FAKE_SETUP_IDLE_REJECT": "false",
            "FAKE_SETUP_INSTALLED_FILE": str(state_root / "setup-installed"),
            "FAKE_SSH_FAILURE_MATCH": "",
            "FAKE_SSH_FAILURE_PHASE": "before",
            "FAKE_SSH_FAILURE_LIMIT": "0",
            "FAKE_SSH_FAILURE_STATUS": "255",
            "FAKE_SSH_FAILURE_MESSAGE": "Connection timed out",
            "FAKE_SSH_FAILURE_COUNT_FILE": str(state_root / "ssh-failure-count"),
            "FAKE_BYPASS_SSH_RETRY_SLEEP": "false",
            "FAKE_SSH_RETRY_SLEEP_LOG": str(state_root / "ssh-retry-sleep"),
            "FAKE_TRACK_SINGLE_SUBMISSION": "false",
            "FAKE_SUBMISSION_FILE": str(state_root / "submission"),
            "FAKE_GPU_PUBLISHED_FILE": str(state_root / "gpu-published"),
            "FAKE_GPU_PARTIAL_FILE": str(state_root / "gpu-partial"),
            "FAKE_BENCHMARK_PUBLISHED_FILE": str(state_root / "benchmark-published"),
            "FAKE_DATASETS_COMPLETE_FILE": str(state_root / "datasets-complete"),
            "FAKE_PACKAGE_STATE_READY_FILE": str(state_root / "package-state-ready"),
            "FAKE_WORKFLOW_READY_FILE": str(state_root / "workflow-ready"),
            "FAKE_WORKFLOW_COMPLETE_FILE": str(state_root / "workflow-complete"),
            "FAKE_PARENT_FINALIZED_FILE": str(state_root / "parent-finalized"),
            "FAKE_TARGET_VALIDATION_COUNT_FILE": str(state_root / "target-validation-count"),
            "FAKE_REMOTE_CLEANED_FILE": str(state_root / "remote-cleaned"),
            "FAKE_SOURCE_DIRECTORIES_FILE": str(source_directories_file),
            "FAKE_LOCAL_PYTHON": str(local_python),
            "FAKE_TMUX_SESSION_FILE": str(state_root / "tmux-session"),
            "FAKE_TMUX_START_COUNT_FILE": str(state_root / "tmux-start-count"),
            "FAKE_BACKGROUND_CHILD_ARGUMENTS_FILE": str(state_root / "background-child-arguments"),
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


def _wait_for_log(
    path: Path,
    needle: str,
    *,
    minimum_count: int = 1,
    timeout_seconds: float = 10.0,
) -> None:
    """Wait until one fake-command log contains the requested evidence."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        content = path.read_text(encoding="utf-8") if path.exists() else ""
        if content.count(needle) >= minimum_count:
            return
        time.sleep(0.05)
    message = f"Timed out waiting for {minimum_count} occurrence(s) of {needle!r}."
    raise AssertionError(message)


def _run_interruptible(
    workflow: Path,
    arguments: list[str],
    environment: dict[str, str],
) -> subprocess.Popen[str]:
    """Start one foreground workflow in its own test-owned process group."""
    return subprocess.Popen(
        [str(workflow), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
        start_new_session=True,
    )


def _terminate_test_process(process: subprocess.Popen[str]) -> None:
    """Force-stop one still-live test-owned workflow process group."""
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate(timeout=5)


def _execution_config_with_cores(content: str, cores_per_case: int) -> str:
    """Return test-owned execution YAML with one replaced cluster core value."""
    lines = content.splitlines(keepends=True)
    matches = [index for index, line in enumerate(lines) if line.startswith("  cores_per_case:")]
    assert len(matches) == 1
    index = matches[0]
    newline = "\n" if lines[index].endswith("\n") else ""
    lines[index] = f"  cores_per_case: {cores_per_case}{newline}"
    return "".join(lines)


def _initialize_snapshot_repository(
    workflow: Path,
    environment: dict[str, str],
) -> tuple[Path, Path, str]:
    """Freeze one committed fixture and keep workflow Git at a fake boundary."""
    project = workflow.parent.parent
    source_probe = project / "src/generation/source_probe.py"
    source_probe.parent.mkdir(parents=True)
    source_probe.write_text("committed generation behavior", encoding="utf-8")
    committed_root = Path(environment["FAKE_COMMITTED_ROOT"])
    shutil.copytree(project, committed_root, dirs_exist_ok=True)
    git_directory = project / ".git"
    git_directory.mkdir()
    (git_directory / "index").write_bytes(b"synthetic immutable index\n")
    commit = _COMMIT
    environment["FAKE_GIT_COMMIT"] = commit
    environment["FAKE_GIT_STATUS"] = ""
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


def _campaign(workflow: Path) -> Path:
    """Return the copied campaign configuration."""
    return workflow.parent.parent / "configs/generation/campaigns/steady_flow/id_dataset.yaml"


def _smoke_workflow(workflow: Path) -> Path:
    """Return the copied paired Technical Smoke workflow configuration."""
    return workflow.parent.parent / "configs/generation/workflows/technical_smoke.yaml"


def _benchmark_suite(workflow: Path) -> Path:
    """Return the copied core-scaling benchmark suite configuration."""
    return workflow.parent.parent / "configs/generation/benchmarks/transient_core_scaling/suite.yaml"


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
    attempt_directory = f"01_generation/attempts/{batch_id}/case_00001/{_RUN_ID}"
    source_directories = (
        meta_directory,
        raw_directory,
        processed_directory,
        attempt_directory,
    )
    for relative in (campaign_directory, *source_directories):
        (mirror / relative).mkdir(parents=True)
    (mirror / campaign_directory / "campaign_terminal.json").write_text(
        json.dumps({"campaign_run_id": _RUN_ID}) + "\n",
        encoding="utf-8",
    )
    (mirror / meta_directory / "batch_manifest.json").write_text("{}\n", encoding="utf-8")
    (mirror / raw_directory / "case_0001.txt").write_text("raw\n", encoding="utf-8")
    (mirror / processed_directory / "case_0001.txt").write_text("processed\n", encoding="utf-8")
    (mirror / attempt_directory / "attempt.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    environment["FAKE_TRANSFER_PLAN"] = (
        f"campaign\tsteady_flow_steady_flow_id_dataset_v1\t{_COMMIT}\t{campaign_directory}"
        "\tconfigs/generation/campaigns/steady_flow/id_dataset.yaml\n"
        f"batch\t{_BATCH_NAME}\t{batch_id}\t1\t{meta_directory}\t{raw_directory}\t{processed_directory}\n"
        f"attempt\t{_BATCH_NAME}\t{attempt_directory}\n"
    )
    Path(environment["FAKE_SOURCE_DIRECTORIES_FILE"]).write_text(
        "\n".join(source_directories[:3]) + "\n",
        encoding="utf-8",
    )
    return source_directories


def test_campaign_source_status_cli_emits_positional_tsv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Keep the source-status TSV stable while making sizing explicit opt-in."""
    status = {
        "campaign_run_id": _RUN_ID,
        "campaign_state": "active",
        "source_state": "retained",
        "reclaimable_bytes": 1234,
        "cleanup_eligibility": "ineligible",
        "active_slurm": True,
    }
    include_sizes: list[bool] = []

    def source_status(*_args: object, **kwargs: object) -> dict[str, object]:
        include_sizes.append(bool(kwargs["include_sizes"]))
        return status

    monkeypatch.setattr(
        cli_generation.workflow_service,
        "campaign_source_status",
        source_status,
    )
    common_arguments = [
        "campaign-source-status",
        _RUN_ID,
        "--format",
        "tsv",
        "--storage-root",
        str(tmp_path),
    ]
    result = cli_generation.main([*common_arguments, "--query-scheduler"])

    assert result == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    fields = captured.out.rstrip("\n").split("\t")
    assert len(fields) == 7
    assert fields[0] == "source-status"
    assert fields[1] == _RUN_ID
    assert fields[2] == "active"
    assert fields[3] == "retained"
    assert int(fields[4]) == status["reclaimable_bytes"]
    assert fields[5] == status["cleanup_eligibility"]
    assert fields[6] == str(status["active_slurm"])
    assert include_sizes == [False]

    assert cli_generation.main([*common_arguments, "--include-sizes"]) == 0
    capsys.readouterr()
    assert include_sizes == [False, True]


def test_storage_status_cli_sizes_are_explicit_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Keep recursive storage sizing behind an explicit administrative flag."""
    include_sizes: list[bool] = []

    def storage_status(*_args: object, **kwargs: object) -> dict[str, object]:
        include_sizes.append(bool(kwargs["include_sizes"]))
        return {"role": kwargs["role"]}

    monkeypatch.setattr(
        cli_generation.workflow_service,
        "storage_status",
        storage_status,
    )
    common_arguments = [
        "storage-status",
        "--role",
        "gpu",
        "--storage-root",
        str(tmp_path),
    ]
    assert cli_generation.main(common_arguments) == 0
    capsys.readouterr()
    assert cli_generation.main([*common_arguments, "--include-sizes"]) == 0
    capsys.readouterr()
    assert include_sizes == [False, True]


def test_fresh_campaign_monitoring_reports_concise_success(tmp_path: Path) -> None:
    """Report one successful campaign without repeating state or dumping machine JSON."""
    workflow, log, environment, storage, _mirror = _harness(tmp_path)
    environment["FAKE_LOGIN_PREFLIGHT_STDOUT"] = "true"
    environment["FAKE_TRACK_SINGLE_SUBMISSION"] = "true"
    environment["FAKE_SOURCE_STATE"] = "active"
    environment["FAKE_CAMPAIGN_STATES"] = "feeding,feeding,running,successful"
    environment["FAKE_GPU_ALWAYS_VALID"] = "true"
    campaign = workflow.parent.parent / "configs/generation/campaigns/steady_flow/technical_smoke.yaml"
    assert not storage.exists()

    result = _run(
        workflow,
        ["run", str(campaign), *_remote_options(), "--keep-cpu-source"],
        environment,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("State: feeding") == 1
    assert result.stdout.count("State: running") == 1
    assert "State: successful" in result.stdout
    assert f"Campaign: {_RUN_ID}" in result.stdout
    assert any(line.startswith("      case_0001") for line in result.stdout.splitlines())
    assert any(line.startswith("        phase=transient_drying") for line in result.stdout.splitlines())
    assert "declared packages and finalizers validated" in result.stdout
    assert not any(line.lstrip().startswith("{") and line.rstrip().endswith("}") for line in result.stdout.splitlines())
    assert Path(environment["FAKE_SUBMISSION_FILE"]).read_text(encoding="utf-8") == "591776\n"
    assert "reused=0 generated=1" in result.stdout
    log_text = log.read_text(encoding="utf-8")
    remote_commands = [line for line in log_text.splitlines() if line.startswith("<bash -l -s --")]
    prepare_index = next(index for index, line in enumerate(remote_commands) if " prepare-campaign-inputs " in line)
    submit_index = next(index for index, line in enumerate(remote_commands) if " submit-campaign " in line)
    assert prepare_index < submit_index
    assert not any(" plan-campaign " in line for line in remote_commands)
    assert sum(" submit-campaign " in line for line in remote_commands) == 1


def test_zero_declared_packages_reports_finalizer_only_success(tmp_path: Path) -> None:
    """Describe zero-package campaigns without claiming package construction."""
    workflow, _log, environment, storage, _mirror = _harness(tmp_path)
    environment["FAKE_GPU_ALWAYS_VALID"] = "true"
    environment["FAKE_DECLARED_PACKAGE_COUNT"] = "0"
    campaign = workflow.parent.parent / "configs/generation/campaigns/transient_drying/material_pilot.yaml"

    result = _run(
        workflow,
        ["run", str(campaign), *_remote_options(), "--keep-cpu-source"],
        environment,
    )

    assert result.returncode == 0, result.stderr
    assert "no Dataset packages declared; package finalizer gates validated" in result.stdout
    assert storage.exists()


def test_transient_permission_denial_during_remote_home_resolution_recovers(
    tmp_path: Path,
) -> None:
    """Continue the same workflow after a short-lived login authentication failure."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    environment["FAKE_TRACK_SINGLE_SUBMISSION"] = "true"
    environment["FAKE_CAMPAIGN_STATES"] = "successful"
    environment["FAKE_SSH_FAILURE_MATCH"] = "${HOME}"
    environment["FAKE_SSH_FAILURE_LIMIT"] = "1"
    environment["FAKE_SSH_FAILURE_MESSAGE"] = "Permission denied (publickey,gssapi-keyex,gssapi-with-mic,password)."
    environment["FAKE_BYPASS_SSH_RETRY_SLEEP"] = "true"

    result = _run(
        workflow,
        ["run", str(_campaign(workflow)), *_remote_options(), "--defer-collection"],
        environment,
    )

    assert result.returncode == 0, result.stderr
    assert f"campaign_run_id={_RUN_ID}" in result.stdout
    assert "WARNING: transient SSH failure during remote HOME resolution" in result.stderr
    assert "SSH connection recovered:" in result.stderr
    assert "FAILED:" not in result.stderr
    log_text = log.read_text(encoding="utf-8")
    assert log_text.count(" submit-campaign Python/") == 1
    assert "cancel-campaign" not in log_text
    assert "record-workflow-failure" not in log_text
    assert Path(environment["FAKE_SSH_RETRY_SLEEP_LOG"]).read_text(encoding="utf-8").splitlines() == ["5"]


def test_transient_combined_monitor_failure_retries_same_iteration(
    tmp_path: Path,
) -> None:
    """Retry one combined snapshot without another submission or cancellation."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    environment["FAKE_TRACK_SINGLE_SUBMISSION"] = "true"
    environment["FAKE_CAMPAIGN_STATES"] = "successful"
    environment["FAKE_SSH_FAILURE_MATCH"] = " resume-campaign "
    environment["FAKE_SSH_FAILURE_LIMIT"] = "1"
    environment["FAKE_SSH_FAILURE_MESSAGE"] = "Connection reset by peer"
    environment["FAKE_BYPASS_SSH_RETRY_SLEEP"] = "true"

    result = _run(
        workflow,
        ["run", str(_campaign(workflow)), *_remote_options(), "--defer-collection"],
        environment,
    )

    assert result.returncode == 0, result.stderr
    assert f"Campaign: {_RUN_ID}" in result.stdout
    assert "WARNING: transient SSH failure during campaign resume and status snapshot" in result.stderr
    assert "FAILED:" not in result.stderr
    command_lines = log.read_text(encoding="utf-8").splitlines()
    remote_commands = [line for line in command_lines if line.startswith("<bash -l -s --")]
    assert sum(" resume-campaign " in line for line in remote_commands) == 2
    assert not any(" campaign-status " in line for line in remote_commands)
    assert sum("submit-campaign" in line for line in remote_commands) == 1
    assert not any("cancel-campaign" in line for line in command_lines)
    assert not any("record-workflow-failure" in line for line in command_lines)


def test_persistent_combined_monitor_failure_preserves_resume_evidence(
    tmp_path: Path,
) -> None:
    """Fail closed after five attempts without cancelling or duplicating the run."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    environment["FAKE_TRACK_SINGLE_SUBMISSION"] = "true"
    environment["FAKE_CAMPAIGN_STATES"] = "running"
    environment["FAKE_SSH_FAILURE_MATCH"] = " resume-campaign "
    environment["FAKE_SSH_FAILURE_LIMIT"] = "5"
    environment["FAKE_SSH_FAILURE_MESSAGE"] = "Permission denied (publickey,gssapi-keyex,gssapi-with-mic,password)."
    environment["FAKE_BYPASS_SSH_RETRY_SLEEP"] = "true"
    campaign = _campaign(workflow)

    result = _run(
        workflow,
        ["run", str(campaign), *_remote_options(), "--defer-collection"],
        environment,
    )

    assert result.returncode != 0
    assert "Persistent SSH transport/authentication failure after 5 attempts" in result.stderr
    assert f"campaign_run_id: {_RUN_ID}" in result.stderr
    assert "CPU bytes retained:" in result.stderr
    assert "generation_workflow.sh run" in result.stderr
    assert campaign.name in result.stderr
    assert "--defer-collection" in result.stderr
    command_lines = log.read_text(encoding="utf-8").splitlines()
    remote_commands = [line for line in command_lines if line.startswith("<bash -l -s --")]
    assert sum(" resume-campaign " in line for line in remote_commands) == 5
    assert not any(" campaign-status " in line for line in remote_commands)
    assert sum(" campaign-source-status " in line for line in remote_commands) == 1
    assert sum("submit-campaign" in line for line in remote_commands) == 1
    assert not any("cancel-campaign" in line for line in command_lines)
    assert any("record-workflow-failure" in line for line in command_lines)


def test_lost_campaign_resume_acknowledgement_replays_deduplicated_resume(
    tmp_path: Path,
) -> None:
    """Replay only durable campaign reconciliation after its SSH response is lost."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    environment["FAKE_TRACK_SINGLE_SUBMISSION"] = "true"
    environment["FAKE_CAMPAIGN_STATES"] = "successful"
    environment["FAKE_SSH_FAILURE_MATCH"] = " resume-campaign "
    environment["FAKE_SSH_FAILURE_PHASE"] = "after"
    environment["FAKE_SSH_FAILURE_LIMIT"] = "1"
    environment["FAKE_SSH_FAILURE_MESSAGE"] = "Broken pipe"
    environment["FAKE_BYPASS_SSH_RETRY_SLEEP"] = "true"

    result = _run(
        workflow,
        ["run", str(_campaign(workflow)), *_remote_options(), "--defer-collection"],
        environment,
    )

    assert result.returncode == 0, result.stderr
    assert f"campaign_run_id={_RUN_ID}" in result.stdout
    assert "WARNING: transient SSH failure during campaign resume and status snapshot" in result.stderr
    command_lines = log.read_text(encoding="utf-8").splitlines()
    remote_commands = [line for line in command_lines if line.startswith("<bash -l -s --")]
    assert sum(" resume-campaign " in line for line in remote_commands) == 2
    assert sum("submit-campaign" in line for line in remote_commands) == 1
    assert not any("cancel-campaign" in line for line in command_lines)
    assert not any("record-workflow-failure" in line for line in command_lines)


def test_transient_benchmark_status_failure_retries_without_replaying_resume(
    tmp_path: Path,
) -> None:
    """Keep ambiguous benchmark resume one-shot while retrying its status read."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    environment["FAKE_BENCHMARK_STATE"] = "complete"
    environment["FAKE_SSH_FAILURE_MATCH"] = " --format monitor"
    environment["FAKE_SSH_FAILURE_LIMIT"] = "1"
    environment["FAKE_SSH_FAILURE_MESSAGE"] = "kex_exchange_identification: read: Connection reset by peer"
    environment["FAKE_BYPASS_SSH_RETRY_SLEEP"] = "true"

    result = _run(
        workflow,
        ["run", str(_benchmark_suite(workflow)), *_remote_options(), "--defer-collection"],
        environment,
    )

    assert result.returncode == 0, result.stderr
    assert f"Run: {_BENCHMARK_RUN_ID}" in result.stdout
    assert "WARNING: transient SSH failure during benchmark status read" in result.stderr
    command_lines = log.read_text(encoding="utf-8").splitlines()
    remote_commands = [line for line in command_lines if line.startswith("<bash -l -s --")]
    assert sum(" resume-core-benchmark " in line for line in remote_commands) == 1
    assert sum(" core-benchmark-status " in line and "--format monitor" in line for line in remote_commands) == 2
    assert sum("submit-core-benchmark" in line for line in remote_commands) == 1
    assert not any("cancel-core-benchmark" in line for line in command_lines)


def test_interrupt_during_ssh_retry_wait_preserves_owned_cancellation_contract(
    tmp_path: Path,
) -> None:
    """Keep graceful/force ownership and status 130 while retry sleep is active."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    environment["FAKE_CAMPAIGN_STATES"] = "running"
    environment["FAKE_SSH_FAILURE_MATCH"] = " resume-campaign "
    environment["FAKE_SSH_FAILURE_LIMIT"] = "5"
    environment["FAKE_SSH_FAILURE_MESSAGE"] = "Connection timed out"
    environment["FAKE_GRACEFUL_CANCEL_DELAY_SECONDS"] = "10"
    process = _run_interruptible(
        workflow,
        ["run", str(_campaign(workflow)), *_remote_options(), "--defer-collection"],
        environment,
    )
    try:
        _wait_for_log(log, "ssh-injected-failure < resume-campaign >")
        os.killpg(process.pid, signal.SIGINT)
        _wait_for_log(log, " cancel-campaign ")
        os.killpg(process.pid, signal.SIGINT)
        stdout, stderr = process.communicate(timeout=10)
    finally:
        _terminate_test_process(process)

    assert process.returncode == 130
    assert stdout
    assert stderr.count("Graceful campaign cancellation requested.") == 1
    assert stderr.count("Force campaign cancellation requested.") == 1
    remote_commands = [line for line in log.read_text(encoding="utf-8").splitlines() if line.startswith("<bash -l -s --")]
    cancellation_commands = [line for line in remote_commands if " cancel-campaign " in line]
    assert len(cancellation_commands) == 2
    assert all(_RUN_ID in line for line in cancellation_commands)
    assert sum(" --force" in line for line in cancellation_commands) == 1


def test_foreground_interrupts_request_graceful_then_force_cancellation(
    tmp_path: Path,
) -> None:
    """Route first and second Ctrl+C through the shared campaign owner."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    environment["FAKE_SOURCE_STATE"] = "active"
    environment["FAKE_CAMPAIGN_STATES"] = "running"
    environment["FAKE_GRACEFUL_CANCEL_DELAY_SECONDS"] = "10"
    process = _run_interruptible(
        workflow,
        [
            "run",
            str(_campaign(workflow)),
            *_remote_options(),
            "--keep-cpu-source",
        ],
        environment,
    )
    try:
        _wait_for_log(log, " --format workflow-monitor ")
        os.killpg(process.pid, signal.SIGINT)
        _wait_for_log(log, " cancel-campaign ")
        os.killpg(process.pid, signal.SIGINT)
        stdout, stderr = process.communicate(timeout=10)
    finally:
        _terminate_test_process(process)

    assert process.returncode != 0
    assert stdout
    assert stderr.count("Graceful campaign cancellation requested.") == 1
    assert stderr.count("Press Ctrl+C again to force cancellation.") == 1
    assert stderr.count("Force campaign cancellation requested.") == 1
    log_text = log.read_text(encoding="utf-8")
    assert log_text.count(" cancel-campaign ") == 2
    assert log_text.count(" --force") == 1


def test_interrupt_before_campaign_launch_writes_no_cancellation_request(
    tmp_path: Path,
) -> None:
    """Leave cancellation unarmed until a persisted campaign run ID exists."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    environment["FAKE_INPUT_PREPARATION_DELAY_SECONDS"] = "10"
    process = _run_interruptible(
        workflow,
        [
            "run",
            str(_campaign(workflow)),
            *_remote_options(),
            "--keep-cpu-source",
        ],
        environment,
    )
    try:
        _wait_for_log(log, " prepare-campaign-inputs ")
        os.killpg(process.pid, signal.SIGINT)
        _stdout, _stderr = process.communicate(timeout=10)
    finally:
        _terminate_test_process(process)

    assert process.returncode != 0
    remote_commands = [line for line in log.read_text(encoding="utf-8").splitlines() if line.startswith("<bash -l -s --")]
    assert not any(" submit-campaign " in line for line in remote_commands)
    assert not any(" cancel-campaign " in line for line in remote_commands)


def test_same_config_run_reapplies_the_matrix_until_campaign_success(
    tmp_path: Path,
) -> None:
    """Continue multiple replay or restart decisions through the same config."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    environment["FAKE_SOURCE_STATE"] = "active"
    environment["FAKE_CAMPAIGN_STATES"] = "feeding,feeding,successful"
    environment["FAKE_GPU_ALWAYS_VALID"] = "true"

    result = _run(
        workflow,
        ["run", str(_campaign(workflow)), *_remote_options(), "--keep-cpu-source"],
        environment,
    )

    assert result.returncode == 0, result.stderr
    log_text = log.read_text(encoding="utf-8")
    assert log_text.count(" resume-campaign ") == 3
    assert " feed-campaign " not in log_text


def test_public_cancel_force_cancels_only_the_owned_campaign(
    tmp_path: Path,
) -> None:
    """Expose force cancellation as a distinct administrative action."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)

    cancelled = _run(
        workflow,
        ["cancel", _RUN_ID, "--force", *_remote_options()],
        environment,
    )

    assert cancelled.returncode == 0, cancelled.stderr
    log_text = log.read_text(encoding="utf-8")
    assert log_text.count(" cancel-campaign ") == 1
    assert log_text.count(" --force") == 1


def test_valid_canonical_inputs_are_reused_before_campaign_submission(tmp_path: Path) -> None:
    """Admit valid exact inputs without regenerating them before launch."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    environment["FAKE_INPUT_GENERATED_COUNT"] = "0"
    environment["FAKE_INPUT_REUSED_COUNT"] = "1"
    environment["FAKE_TRACK_SINGLE_SUBMISSION"] = "true"

    result = _run(
        workflow,
        ["run", str(_campaign(workflow)), *_remote_options(), "--defer-collection"],
        environment,
    )

    assert result.returncode == 0, result.stderr
    assert "reused=1 generated=0" in result.stdout
    log_text = log.read_text(encoding="utf-8")
    remote_commands = [line for line in log_text.splitlines() if line.startswith("<bash -l -s --")]
    prepare_index = next(index for index, line in enumerate(remote_commands) if " prepare-campaign-inputs " in line)
    submit_index = next(index for index, line in enumerate(remote_commands) if " submit-campaign " in line)
    assert prepare_index < submit_index
    assert not any(" plan-campaign " in line for line in remote_commands)
    assert "state=awaiting_collection" in result.stdout
    assert "rsync-start" not in log_text
    assert "<build-campaign-datasets>" not in log_text
    assert "cleanup-campaign-source" not in log_text


def test_invalid_canonical_inputs_abort_before_plan_or_submission(tmp_path: Path) -> None:
    """Stop an invalid exact input generation before campaign side effects."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    environment["FAKE_INPUT_PREPARATION_FAIL"] = "true"

    result = _run(
        workflow,
        ["run", str(_campaign(workflow)), *_remote_options(), "--keep-cpu-source"],
        environment,
    )

    assert result.returncode != 0
    assert "Canonical campaign input preparation failed before submission." in result.stderr
    remote_commands = [line for line in log.read_text(encoding="utf-8").splitlines() if line.startswith("<bash -l -s --")]
    assert any(" prepare-campaign-inputs " in line for line in remote_commands)
    assert not any(" plan-campaign " in line for line in remote_commands)
    assert not any(" submit-campaign " in line for line in remote_commands)


def test_pilot_prepares_canonical_inputs_before_plan_and_submission(tmp_path: Path) -> None:
    """Keep pilot input readiness ahead of every planning or launch side effect."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    environment["FAKE_GPU_ALWAYS_VALID"] = "true"
    campaign = workflow.parent.parent / "configs/generation/campaigns/transient_drying/material_pilot.yaml"

    result = _run(
        workflow,
        ["run", str(campaign), *_remote_options(), "--keep-cpu-source"],
        environment,
    )

    assert result.returncode == 0, result.stderr
    assert "reused=0 generated=1" in result.stdout
    remote_commands = [line for line in log.read_text(encoding="utf-8").splitlines() if line.startswith("<bash -l -s --")]
    prepare_index = next(index for index, line in enumerate(remote_commands) if " prepare-campaign-inputs " in line)
    submit_index = next(index for index, line in enumerate(remote_commands) if " submit-campaign " in line)
    assert prepare_index < submit_index
    assert not any(" plan-campaign " in line for line in remote_commands)


def test_license_blocked_campaign_polling_reaches_publication_without_failure_evidence(tmp_path: Path) -> None:
    """Poll a retry-delay campaign until its later publication completes."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    environment["FAKE_SOURCE_STATE"] = "active"
    environment["FAKE_CAMPAIGN_STATES"] = "license_blocked,successful"
    environment["FAKE_GPU_ALWAYS_VALID"] = "true"
    environment["FAKE_TRACK_SINGLE_SUBMISSION"] = "true"

    result = _run(
        workflow,
        ["run", str(_campaign(workflow)), *_remote_options(), "--keep-cpu-source"],
        environment,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.index("State: license_blocked") < result.stdout.index("State: successful")
    log_text = log.read_text(encoding="utf-8")
    remote_commands = [line for line in log_text.splitlines() if line.startswith("<bash -l -s --")]
    assert sum(" --format workflow-monitor " in line for line in remote_commands) == 2
    assert sum(" validate-campaign-terminal " in line for line in remote_commands) == 1
    assert sum(" submit-campaign " in line for line in remote_commands) == 1
    assert "record-workflow-failure" not in log_text


def test_unchanged_campaign_states_are_coalesced(tmp_path: Path) -> None:
    """Suppress repeated unchanged states while reporting a later state change."""
    workflow, _log, environment, _storage, _mirror = _harness(tmp_path)
    environment["FAKE_SOURCE_STATE"] = "active"
    environment["FAKE_CAMPAIGN_STATES"] = "feeding,feeding,feeding,successful"
    environment["FAKE_CONSOLE_TIMES"] = "1000,1299,1300,1301"
    environment["FAKE_GPU_ALWAYS_VALID"] = "true"

    result = _run(
        workflow,
        ["run", str(_campaign(workflow)), *_remote_options(), "--keep-cpu-source"],
        environment,
    )

    assert result.returncode == 0, result.stderr
    feeding_lines = [line for line in result.stdout.splitlines() if "State: feeding" in line]
    assert 0 < len(feeding_lines) < 3
    assert "State: successful" in result.stdout


def test_explicit_status_prints_campaign_summary_before_storage(tmp_path: Path) -> None:
    """Present canonical per-case campaign status before storage diagnostics."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)

    result = _run(workflow, ["status", _RUN_ID, *_remote_options()], environment)

    assert result.returncode == 0, result.stderr
    assert result.stdout.index("Campaign status:") < result.stdout.index("GPU storage status:")
    assert result.stdout.index("GPU storage status:") < result.stdout.index("CPU storage status:")
    assert f"Campaign: {_RUN_ID}" in result.stdout
    assert "case_0001" in result.stdout
    commands = log.read_text(encoding="utf-8")
    assert commands.count("campaign-status") == 1
    assert "--format workflow-monitor" in commands
    assert commands.count("<--metadata-only>") == 2
    assert commands.count("<--omit-run-status>") == 2


def test_changed_solver_progress_is_rendered_only_after_the_minimum_interval(tmp_path: Path) -> None:
    """Show advancing solver evidence after 60 seconds without printing every poll."""
    workflow, _log, environment, _storage, _mirror = _harness(tmp_path)
    environment["FAKE_SOURCE_STATE"] = "active"
    environment["FAKE_CAMPAIGN_STATES"] = "running,running,running,successful"
    environment["FAKE_CAMPAIGN_PROGRESS_SIGNATURES"] = ",".join(("a" * 64, "b" * 64, "c" * 64, "d" * 64))
    environment["FAKE_PROGRESS_VALUES"] = "0.100,0.200,0.300,0.400"
    environment["FAKE_CONSOLE_TIMES"] = "1000,1030,1060,1061"
    environment["FAKE_GPU_ALWAYS_VALID"] = "true"

    result = _run(
        workflow,
        ["run", str(_campaign(workflow)), *_remote_options(), "--keep-cpu-source"],
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

    result = _run(workflow, ["run", str(_smoke_workflow(workflow)), "--keep-cpu-source"], environment)

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


def test_dry_run_resolves_the_config_without_runtime_smoke_gates(
    tmp_path: Path,
) -> None:
    """Keep immutable plan inspection independent of remote runtime evidence."""
    for index, rejected_profile in enumerate(("transient_drying", "steady_flow")):
        root = tmp_path / f"case-{index}"
        root.mkdir()
        workflow, log, environment, _storage, _mirror = _harness(root)
        environment["FAKE_SMOKE_EVIDENCE_REJECT_PROFILE"] = rejected_profile

        result = _run(
            workflow,
            ["run", str(_campaign(workflow)), "--dry-run", *_remote_options()],
            environment,
        )

        assert result.returncode == 0, result.stderr
        assert '"run_kind":"campaign"' in result.stdout
        assert "technical-smoke-evidence-profile" not in log.read_text(encoding="utf-8")


def test_combined_smoke_records_and_checks_both_profile_evidence(tmp_path: Path) -> None:
    """Keep the combined smoke responsible for both profile campaigns."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    environment["FAKE_GPU_ALWAYS_VALID"] = "true"

    result = _run(workflow, ["run", str(_smoke_workflow(workflow)), "--keep-cpu-source"], environment)

    assert result.returncode == 0, result.stderr
    log_lines = log.read_text(encoding="utf-8").splitlines()
    evidence_lines = [line for line in log_lines if line.startswith("technical-smoke-evidence-profile")]
    assert evidence_lines == [
        "technical-smoke-evidence-profile <steady_flow>",
        "technical-smoke-evidence-profile <transient_drying>",
    ]
    remote_commands = [line for line in log_lines if line.startswith("<bash -l -s --")]
    lifecycle = [command for line in remote_commands for command in ("prepare-campaign-inputs", "submit-campaign") if f" {command} " in line]
    assert lifecycle == [
        "prepare-campaign-inputs",
        "submit-campaign",
        "prepare-campaign-inputs",
        "submit-campaign",
    ]
    assert not any(" plan-campaign " in line for line in remote_commands)


def test_failed_technical_smoke_publishes_no_profile_evidence(tmp_path: Path) -> None:
    """Keep a failed technical workflow from producing readiness evidence."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    environment["FAKE_GPU_ALWAYS_VALID"] = "true"
    environment["FAKE_BUILD_FAIL"] = "true"

    result = _run(workflow, ["run", str(_smoke_workflow(workflow)), "--keep-cpu-source"], environment)

    assert result.returncode != 0
    log_text = log.read_text(encoding="utf-8")
    assert "submit-campaign" in log_text
    assert "<finalize-technical-smoke-evidence>" not in log_text


def test_dirty_worktree_uses_clean_pinned_source_without_modifying_checkout(tmp_path: Path) -> None:
    """Ignore real staged, tracked, workflow, config, and untracked edits exactly."""
    workflow, log, environment, storage, _mirror = _harness(tmp_path)
    project, source_probe, commit = _initialize_snapshot_repository(workflow, environment)
    campaign = _campaign(workflow)
    dirty_marker = "# DIRTY_GENERATION_CONFIG"
    campaign.write_text(
        campaign.read_text(encoding="utf-8") + f"\n{dirty_marker}\n",
        encoding="utf-8",
    )
    source_probe.write_text("dirty staged generation behavior", encoding="utf-8")

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

    environment["FAKE_GIT_STATUS"] = (
        "M  src/generation/source_probe.py\n M scripts/generation_workflow.sh\n?? src/uncommitted_generation_behavior.py\n"
    )
    status_before = environment["FAKE_GIT_STATUS"]
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
            "run",
            str(campaign),
            "--dry-run",
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
    status_after = environment["FAKE_GIT_STATUS"]
    assert status_after == status_before
    assert environment["FAKE_GIT_COMMIT"] == commit
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


def test_dirty_snapshot_worktree_runs_the_motivating_smoke_command(tmp_path: Path) -> None:
    """Run paired Technical Smoke orchestration from committed source while dirty."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    project = workflow.parent.parent
    execution_config = project / "configs/generation/execution/cluster_cpu.yaml"
    committed_cores = 16
    dirty_cores = 8
    committed_execution = _execution_config_with_cores(
        execution_config.read_text(encoding="utf-8"),
        committed_cores,
    )
    execution_config.write_text(committed_execution, encoding="utf-8")
    project, source_probe, commit = _initialize_snapshot_repository(workflow, environment)
    source_probe.write_text("dirty generation behavior", encoding="utf-8")
    execution_config.write_text(
        _execution_config_with_cores(committed_execution, dirty_cores),
        encoding="utf-8",
    )
    untracked = project / "notes-in-progress.txt"
    untracked.write_text("continue local development\n", encoding="utf-8")
    environment["FAKE_EXPECT_SOURCE_FILE"] = source_probe.relative_to(project).as_posix()
    environment["FAKE_EXPECT_SOURCE_TEXT"] = "committed generation behavior"
    environment["FAKE_UNTRACKED_SOURCE_PATH"] = untracked.relative_to(project).as_posix()
    environment["FAKE_GPU_ALWAYS_VALID"] = "true"
    environment["FAKE_GIT_STATUS"] = " M configs/generation/execution/cluster_cpu.yaml\n M src/generation/source_probe.py\n?? notes-in-progress.txt\n"

    result = _run(workflow, ["run", str(_smoke_workflow(workflow)), "--keep-cpu-source"], environment)

    assert result.returncode == 0, result.stderr
    assert result.stderr.count(f"Source: committed HEAD {commit}") == 1
    assert result.stderr.count("Local worktree: dirty; uncommitted changes ignored") == 1
    assert source_probe.read_text(encoding="utf-8") == "dirty generation behavior"
    dirty_execution = execution_config.read_text(encoding="utf-8")
    assert f"cores_per_case: {dirty_cores}" in dirty_execution
    assert untracked.read_text(encoding="utf-8") == "continue local development\n"
    assert f"Resources: cores_per_case={committed_cores}  max_admission_cases=2  max_running_cases=null" in result.stdout
    log_text = log.read_text(encoding="utf-8")
    assert log_text.count(" submit-campaign Python/") == 2
    assert "<finalize-technical-smoke-evidence>" in log_text
    assert "<finalize-real-smoke>" in log_text
    assert "<cpu-cleanup-authorization>" not in log_text
    assert result.stdout.count("reused=0 generated=1") == 2
    assert result.stdout.count("DONE:") == 1
    assert "run_identity=workflow__0123456789abcdef state=complete" in result.stdout.splitlines()[-1]
    assert f"local-python-commit <{commit}>" in log_text
    assert "Dirty untracked source reached local Python." not in result.stderr
    assert "Local Python did not receive the committed Generation source." not in result.stderr


def test_smoke_finalization_failure_prints_no_top_level_done(tmp_path: Path) -> None:
    """Retain both CPU children when paired finalization does not succeed."""
    workflow, log, environment, _storage, mirror = _harness(tmp_path)
    source_directories = _seed_transfer(mirror, environment)
    environment["FAKE_FINALIZE_SMOKE_FAIL"] = "true"

    result = _run(
        workflow,
        ["run", str(_smoke_workflow(workflow)), *_remote_options()],
        environment,
    )

    assert result.returncode != 0
    assert result.stdout.count("reused=0 generated=1") >= 1
    assert "DONE:" not in result.stdout
    assert "Could not atomically finalize paired Technical Smoke evidence" in result.stderr
    continuation = next(line for line in result.stderr.splitlines() if "generation_workflow.sh run" in line)
    assert "technical_smoke.yaml" in continuation
    assert all((mirror / relative).is_dir() for relative in source_directories)
    log_text = log.read_text(encoding="utf-8")
    assert "<cpu-cleanup-authorization>" not in log_text
    assert "cleanup-campaign-source" not in log_text
    assert "<record-cpu-cleanup>" not in log_text


def test_parent_validation_targets_just_finalized_receipt_before_cleanup(
    tmp_path: Path,
) -> None:
    """Do not let an unrelated valid receipt authorize child cleanup."""
    workflow, log, environment, storage, mirror = _harness(tmp_path)
    source_directories = _seed_transfer(mirror, environment)
    unrelated = storage / "01_generation/meta/smoke_receipts/unrelated.json"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("{}\n", encoding="utf-8")
    environment["FAKE_TARGET_VALIDATION_FAIL_AFTER_FIRST"] = "true"

    result = _run(
        workflow,
        ["run", str(_smoke_workflow(workflow)), *_remote_options()],
        environment,
    )

    assert result.returncode != 0
    assert "DONE:" not in result.stdout
    assert Path(environment["FAKE_TARGET_VALIDATION_COUNT_FILE"]).read_text(encoding="utf-8").strip() == "2"
    assert all((mirror / relative).is_dir() for relative in source_directories)
    log_text = log.read_text(encoding="utf-8")
    assert "<cpu-cleanup-authorization>" not in log_text
    assert "cleanup-campaign-source" not in log_text


def test_smoke_cleanup_follows_parent_receipt_and_survives_retention(
    tmp_path: Path,
) -> None:
    """Authorize child cleanup only after the paired receipt validates."""
    workflow, log, environment, _storage, mirror = _harness(tmp_path)
    source_directories = _seed_transfer(mirror, environment)

    result = _run(
        workflow,
        ["run", str(_smoke_workflow(workflow)), *_remote_options()],
        environment,
    )

    assert result.returncode == 0, result.stderr
    assert all(not (mirror / relative).exists() for relative in source_directories[:3])
    assert (mirror / source_directories[3]).is_dir()
    log_text = log.read_text(encoding="utf-8")
    finalizer = log_text.index("<finalize-real-smoke>")
    cleanup = log_text.index("<cpu-cleanup-authorization>")
    pre_cleanup_validation = log_text.rfind("<validate-real-smoke>", 0, cleanup)
    cleanup_record = log_text.rfind("<record-cpu-cleanup>")
    post_cleanup_validation = log_text.find("<validate-real-smoke>", cleanup_record)
    assert finalizer < pre_cleanup_validation < cleanup < cleanup_record
    assert post_cleanup_validation > cleanup_record
    assert result.stdout.count("DONE:") == 1


def test_smoke_defer_collection_stops_before_host_finalization_and_cleanup(
    tmp_path: Path,
) -> None:
    """Retain both composite children when collection is explicitly deferred."""
    workflow, log, environment, _storage, mirror = _harness(tmp_path)
    source_directories = _seed_transfer(mirror, environment)

    result = _run(
        workflow,
        [
            "run",
            str(_smoke_workflow(workflow)),
            *_remote_options(),
            "--defer-collection",
        ],
        environment,
    )

    assert result.returncode == 0, result.stderr
    assert "state=awaiting_collection" in result.stdout
    assert "DONE:" not in result.stdout
    assert all((mirror / relative).is_dir() for relative in source_directories)
    log_text = log.read_text(encoding="utf-8")
    assert log_text.count(" submit-campaign Python/") == 2
    for forbidden in (
        "rsync-start",
        "<build-campaign-datasets>",
        "<finalize-real-smoke>",
        "<validate-real-smoke>",
        "<cpu-cleanup-authorization>",
        "cleanup-campaign-source",
    ):
        assert forbidden not in log_text


def test_completed_cpu_cleaned_smoke_children_reach_parent_without_new_work(
    tmp_path: Path,
) -> None:
    """Reuse completed children and run only paired downstream finalization."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    Path(environment["FAKE_DATASETS_COMPLETE_FILE"]).touch()
    Path(environment["FAKE_PACKAGE_STATE_READY_FILE"]).touch()
    Path(environment["FAKE_WORKFLOW_COMPLETE_FILE"]).touch()
    Path(environment["FAKE_REMOTE_CLEANED_FILE"]).touch()
    Path(environment["FAKE_SOURCE_DIRECTORIES_FILE"]).write_text("", encoding="utf-8")
    environment["FAKE_COMPATIBLE_SMOKE_STATUS"] = "compatible_complete"
    environment["FAKE_PRESERVE_WORKFLOW_COMPLETE"] = "true"

    result = _run(
        workflow,
        ["run", str(_smoke_workflow(workflow)), *_remote_options()],
        environment,
    )

    assert result.returncode == 0, result.stderr
    log_text = log.read_text(encoding="utf-8")
    assert "<finalize-real-smoke>" in log_text
    assert "<validate-real-smoke>" in log_text
    for forbidden in (
        "<prepare-campaign-inputs>",
        "<plan-campaign>",
        "<submit-campaign>",
        "<resume-campaign>",
        "<publish-transferred-campaign>",
        "<build-campaign-datasets>",
    ):
        assert forbidden not in log_text
    assert result.stdout.count("DONE:") == 1


def test_source_remains_pinned_when_development_head_advances(tmp_path: Path) -> None:
    """Keep one clean invocation on commit A while the development checkout reaches B."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    project, source_probe, commit_a = _initialize_snapshot_repository(workflow, environment)
    environment["FAKE_EXPECT_SOURCE_FILE"] = source_probe.relative_to(project).as_posix()
    environment["FAKE_EXPECT_SOURCE_TEXT"] = "committed generation behavior"
    ready = tmp_path / "source-ready"
    continuation = tmp_path / "source-continue"
    environment["FAKE_DOCKER_READY_FILE"] = str(ready)
    environment["FAKE_DOCKER_CONTINUE_FILE"] = str(continuation)
    command = [
        str(workflow),
        "run",
        str(_campaign(workflow)),
        "--dry-run",
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
        assert (pinned_root / source_probe.relative_to(project)).read_text(encoding="utf-8") == "committed generation behavior"

        (project / "after_launch.txt").write_text("commit B\n", encoding="utf-8")
        commit_b = "b" * 40
        environment["FAKE_GIT_COMMIT"] = commit_b
        environment["FAKE_GIT_STATUS"] = ""
        assert commit_b != commit_a

        continuation.touch()
        stdout, stderr = process.communicate(timeout=15)
    finally:
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=5)

    assert process.returncode == 0, stderr
    assert json.loads(stdout.splitlines()[-1])["run_kind"] == "campaign"
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
    project, source_probe, commit = _initialize_snapshot_repository(workflow, environment)
    command = [
        str(workflow),
        "run",
        str(_campaign(workflow)),
        "--dry-run",
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
            "run",
            str(_campaign(workflow)),
            "--dry-run",
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
    _initialize_snapshot_repository(workflow, environment)
    source_parent = tmp_path / "source temp"
    source_parent.mkdir()
    environment["TMPDIR"] = str(source_parent)

    result = _run(
        workflow,
        [
            "run",
            str(_campaign(workflow)),
            "--dry-run",
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
            "run",
            str(_campaign(workflow)),
            "--dry-run",
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
    """Reject malformed and unavailable commits before remote workflow operations."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    _project, _source_probe, commit = _initialize_snapshot_repository(workflow, environment)
    mismatch = ("f" * 40) if commit != ("f" * 40) else ("e" * 40)
    common = [
        "run",
        str(_campaign(workflow)),
        "--dry-run",
        "--cpu-host",
        "cpu.example",
        "--remote-root",
        "/remote/generation root",
    ]

    malformed = _run(workflow, [*common, "--git-commit", "short"], environment)
    environment["FAKE_UNAVAILABLE_GIT_COMMIT"] = mismatch
    unavailable = _run(workflow, [*common, "--git-commit", mismatch], environment)

    assert malformed.returncode == 2
    assert "Git commit must be one lowercase 40-character identifier." in malformed.stderr
    assert unavailable.returncode == 1
    assert "Requested commit is unavailable in the local repository" in unavailable.stderr
    assert "ssh-start" not in log.read_text(encoding="utf-8")


def test_explicit_historical_commit_runs_its_pinned_config_after_head_advances(tmp_path: Path) -> None:
    """Run an available historical commit without reading newer worktree content."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    project, source_probe, commit_a = _initialize_snapshot_repository(workflow, environment)
    campaign = _campaign(workflow)
    historical_campaign = campaign.read_text(encoding="utf-8")
    commit_b = "b" * 40
    environment["FAKE_GIT_COMMIT"] = commit_b
    environment["FAKE_GIT_STATUS"] = " M configs/generation/campaigns/steady_flow/id_dataset.yaml\n"
    campaign.write_text(historical_campaign + "\n# NEWER_HEAD_ONLY\n", encoding="utf-8")
    source_probe.write_text("newer dirty generation behavior", encoding="utf-8")
    environment["FAKE_REJECT_CONFIG_TEXT"] = "NEWER_HEAD_ONLY"
    environment["FAKE_EXPECT_SOURCE_FILE"] = source_probe.relative_to(project).as_posix()
    environment["FAKE_EXPECT_SOURCE_TEXT"] = "committed generation behavior"

    result = _run(
        workflow,
        [
            "run",
            str(campaign),
            "--dry-run",
            "--cpu-host",
            "cpu.example",
            "--remote-root",
            "/remote/generation root",
            "--git-commit",
            commit_a,
        ],
        environment,
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr.splitlines()[:2] == [
        f"Source: explicit commit {commit_a} (local HEAD {commit_b})",
        "Local worktree: dirty; uncommitted changes ignored",
    ]
    assert f"local-python-commit <{commit_a}>" in log.read_text(encoding="utf-8")
    assert "Dirty tracked configuration reached local Python." not in result.stderr
    assert "Local Python did not receive the committed Generation source." not in result.stderr
    assert campaign.read_text(encoding="utf-8").endswith("# NEWER_HEAD_ONLY\n")
    assert source_probe.read_text(encoding="utf-8") == "newer dirty generation behavior"


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
        "./configs/generation/campaigns/steady_flow/id_dataset.yaml",
        str(symlink),
        "/workspace/repo/configs/generation/campaigns/steady_flow/id_dataset.yaml",
    )

    for campaign in arguments:
        result = _run(workflow, ["run", campaign, "--dry-run", *_remote_options()], environment)
        assert result.returncode == 2, (campaign, result.stderr)

    assert "ssh-start" not in log.read_text(encoding="utf-8")


def test_captured_launch_stops_when_login_preflight_fails(tmp_path: Path) -> None:
    """Keep captured launch output fail-closed on a prerequisite error."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    environment["FAKE_CPU_LOGIN_RSYNC_MISSING"] = "true"

    result = _run(
        workflow,
        ["run", str(_campaign(workflow)), *_remote_options()],
        environment,
    )

    assert result.returncode != 0
    assert "CPU login prerequisite missing: rsync (blocks transfer)." in result.stderr
    log_text = log.read_text(encoding="utf-8")
    assert "generation_cpu_login_preflight.sh" in log_text
    assert "submit-campaign" not in log_text


def test_preflight_validates_admitted_cpu_runtime_without_submission(tmp_path: Path) -> None:
    """Validate the admitted CPU runtime without creating scientific work."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)

    result = _run(
        workflow,
        ["run", str(_campaign(workflow)), "--preflight-only", *_remote_options()],
        environment,
    )

    assert result.returncode == 0, result.stderr
    assert "PREFLIGHT COMPLETE:" in result.stdout
    assert "COMSOL=COMSOL Multiphysics 6.4.0.293" in result.stdout
    log_text = log.read_text(encoding="utf-8")
    assert "generation_cpu_login_preflight.sh" in log_text
    assert "submit-campaign" not in log_text
    assert "sbatch" not in log_text


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
    assert "fresh_installation=true" in log_text
    assert 'repository="${root}/${commit}"' not in log_text

    repeat = _run(workflow, ["setup-cpu", *_remote_options(), "--execute"], environment)
    assert repeat.returncode == 0, repeat.stderr
    log_text = log.read_text(encoding="utf-8")
    assert "fresh_installation=false" in log_text
    assert log_text.index("assert-shared-setup-idle") < log_text.index('git -C "${repository}" fetch origin "${commit}"')

    unsafe = _run(workflow, ["setup-cpu", "--cpu-host", "bad;host"], environment)
    assert unsafe.returncode == 2


def test_setup_refuses_active_shared_jobs_before_remote_mutation(tmp_path: Path) -> None:
    """Reject an established setup before Git or virtual-environment mutation."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    established = _run(workflow, ["setup-cpu", *_remote_options(), "--execute"], environment)
    assert established.returncode == 0, established.stderr
    assert Path(environment["FAKE_SETUP_INSTALLED_FILE"]).is_file()
    log.write_text("", encoding="utf-8")
    environment["FAKE_SETUP_IDLE_REJECT"] = "true"

    result = _run(workflow, ["setup-cpu", *_remote_options(), "--execute"], environment)

    assert result.returncode != 0
    assert "active_dependent_jobs" in result.stderr
    assert "active dependent scheduler jobs block shared setup" not in result.stderr
    payload = log.read_text(encoding="utf-8")
    assert "setup-idle-check <established>" in payload
    assert "assert-shared-setup-idle" in payload
    assert payload.index("assert-shared-setup-idle") < payload.index('git -C "${repository}" fetch origin "${commit}"')


def test_benchmark_preflight_scratch_is_ephemeral_and_verified(tmp_path: Path) -> None:
    """Keep benchmark preflight scratch outside the persistent CPU layout."""
    workflow, _log, _environment, _storage, _mirror = _harness(tmp_path)

    source = workflow.read_text(encoding="utf-8")

    assert "benchmark-preflight-scratch" not in source
    assert 'mktemp -d "${benchmark_scratch_parent%/}/generation-benchmark-preflight.XXXXXXXX"' in source
    assert ".generation-benchmark-preflight" in source
    assert "cleanup_benchmark_scratch" in source


def test_local_docker_failure_stops_before_remote_mutation(tmp_path: Path) -> None:
    """Stop setup before remote mutation when the required local container fails."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    environment["FAKE_DOCKER_FAIL"] = "true"

    result = _run(workflow, ["setup-cpu", *_remote_options()], environment)

    assert result.returncode == 1
    assert "ssh-start" not in log.read_text(encoding="utf-8")


def test_unified_run_modes_and_option_validation(tmp_path: Path) -> None:
    """Keep dry-run machine-readable, execution resumable, and options strict."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    plan = _run(
        workflow,
        ["run", str(_campaign(workflow)), "--dry-run", *_remote_options()],
        environment,
    )
    assert plan.returncode == 0, plan.stderr
    plan_record = json.loads(plan.stdout.splitlines()[-1])
    assert plan_record["run_kind"] == "campaign"

    logical_campaign = _campaign(workflow).relative_to(workflow.parent.parent).as_posix()
    logical_plan = _run(workflow, ["run", logical_campaign, "--dry-run", *_remote_options()], environment)
    assert logical_plan.returncode == 0, logical_plan.stderr

    configured = _run(
        workflow,
        [
            "run",
            str(_campaign(workflow)),
            "--dry-run",
            *_remote_options(),
            "--only-batch",
            "future.profile::batch",
        ],
        environment,
    )
    assert configured.returncode == 2
    assert "Unsupported option: --only-batch" in configured.stderr

    incompatible = _run(
        workflow,
        [
            "run",
            str(_campaign(workflow)),
            "--dry-run",
            *_remote_options(),
            "--skip-extreme-family-ood",
        ],
        environment,
    )
    assert incompatible.returncode == 2

    environment["FAKE_GPU_ALWAYS_VALID"] = "true"
    launch = _run(
        workflow,
        ["run", str(_campaign(workflow)), *_remote_options(), "--keep-cpu-source"],
        environment,
    )
    assert launch.returncode == 0, launch.stderr
    assert launch.stdout.count("DONE:") == 1
    assert f"Campaign: {_RUN_ID}" in launch.stdout
    assert "submit-campaign" in log.read_text(encoding="utf-8")

    rejected = _run(
        workflow,
        ["run", str(_campaign(workflow)), "--dry-run", *_remote_options(), "--unsupported-option"],
        environment,
    )
    assert rejected.returncode == 2


def test_removed_specialized_starts_fail_before_remote_mutation(
    tmp_path: Path,
) -> None:
    """Keep one public start boundary with concise run-CONFIG guidance."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    removed = (
        "all",
        "launch",
        "resume",
        "smoke",
        "finalize-smoke",
        "benchmark-cores",
        "collect-benchmark",
        "pilot-check",
        "collect",
        "build-datasets",
        "retry-case",
    )

    for command in removed:
        log.write_text("", encoding="utf-8")
        result = _run(workflow, [command, *_remote_options()], environment)
        assert result.returncode == 2
        assert f"Unsupported subcommand: {command}" in result.stderr
        assert "run CONFIG" in result.stderr
        log_text = log.read_text(encoding="utf-8")
        assert "ssh-start" not in log_text
        assert "docker-start" not in log_text


def test_remote_campaign_launch_binds_pinned_commit_to_environment_and_cli(
    tmp_path: Path,
) -> None:
    """Bind one pinned commit through both remote Generation source contracts."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)

    result = _run(
        workflow,
        ["run", str(_campaign(workflow)), *_remote_options(), "--defer-collection"],
        environment,
    )

    assert result.returncode == 0, result.stderr
    assert len(_COMMIT) == 40
    log_text = log.read_text(encoding="utf-8")
    assert f"<{_COMMIT}>" in log_text
    assert 'export GENERATION_GIT_COMMIT="${commit}"' in log_text
    assert '--git-commit "${commit}"' in log_text
    remote_commands = [line for line in log_text.splitlines() if line.startswith("<bash -l -s --")]
    submit_index = next(index for index, line in enumerate(remote_commands) if " submit-campaign " in line)
    monitor_index = next(index for index, line in enumerate(remote_commands) if " resume-campaign " in line)
    assert submit_index < monitor_index
    assert sum(" submit-campaign " in line for line in remote_commands) == 1
    assert sum(" resume-campaign " in line for line in remote_commands) == 1
    assert not any(" campaign-status " in line for line in remote_commands)


def test_remote_campaign_monitoring_binds_pinned_commit_to_resume_feed(
    tmp_path: Path,
) -> None:
    """Bind status and automatic resume feeding to one pinned remote source."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    environment["FAKE_SOURCE_STATE"] = "active"
    environment["FAKE_CAMPAIGN_STATES"] = "successful"
    environment["FAKE_GPU_ALWAYS_VALID"] = "true"

    result = _run(
        workflow,
        ["run", str(_campaign(workflow)), *_remote_options(), "--keep-cpu-source"],
        environment,
    )

    assert result.returncode == 0, result.stderr
    log_text = log.read_text(encoding="utf-8")
    remote_commands = [line for line in log_text.splitlines() if line.startswith("<bash -l -s --")]
    assert sum(" resume-campaign " in line for line in remote_commands) == 1
    assert not any(" campaign-status " in line for line in remote_commands)
    assert not any(" feed-campaign " in line for line in remote_commands)
    assert f" {_COMMIT} " in log_text
    assert 'commit="$4"' in log_text
    assert 'export GENERATION_GIT_COMMIT="${commit}"' in log_text


def test_initial_status_failure_retains_submitted_run_for_safe_inspection(
    tmp_path: Path,
) -> None:
    """Keep one submitted run authoritative when initial status rendering fails."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    environment["FAKE_TRACK_SINGLE_SUBMISSION"] = "true"
    environment["FAKE_CAMPAIGN_STATUS_FAIL"] = "true"

    failed = _run(
        workflow,
        ["run", str(_campaign(workflow)), *_remote_options(), "--keep-cpu-source"],
        environment,
    )

    assert failed.returncode != 0
    assert "Synthetic initial campaign status failure." in failed.stderr
    assert f"campaign_run_id: {_RUN_ID}" in failed.stderr
    assert "FAILED: common work-unit monitoring" in failed.stderr
    assert "Remote launch failed." not in failed.stderr
    assert "generation_workflow.sh run" in failed.stderr
    first_log = log.read_text(encoding="utf-8")
    first_remote_commands = [line for line in first_log.splitlines() if line.startswith("<bash -l -s --")]
    assert sum(" submit-campaign " in line for line in first_remote_commands) == 1
    submission_file = Path(environment["FAKE_SUBMISSION_FILE"])
    assert submission_file.read_text(encoding="utf-8") == "591776\n"

    environment["FAKE_CAMPAIGN_STATUS_FAIL"] = "false"
    inspected = _run(workflow, ["status", _RUN_ID, *_remote_options()], environment)

    assert inspected.returncode == 0, inspected.stderr
    assert f"Campaign: {_RUN_ID}" in inspected.stdout
    final_log = log.read_text(encoding="utf-8")
    final_remote_commands = [line for line in final_log.splitlines() if line.startswith("<bash -l -s --")]
    assert sum(" submit-campaign " in line for line in final_remote_commands) == 1
    assert submission_file.read_text(encoding="utf-8") == "591776\n"


def test_high_level_core_benchmark_preserves_transfer_contract(tmp_path: Path) -> None:
    """Exercise benchmark input readiness, submission, and publication."""
    workflow, log, environment, _storage, mirror = _harness(tmp_path)
    environment["FAKE_LOGIN_PREFLIGHT_STDOUT"] = "true"
    relative = environment["FAKE_BENCHMARK_RELATIVE"]
    remote_directory = mirror / relative
    remote_directory.mkdir(parents=True)
    (remote_directory / "summary.json").write_text("{}\n", encoding="utf-8")

    result = _run(
        workflow,
        ["run", str(_benchmark_suite(workflow)), *_remote_options(), "--keep-cpu-source"],
        environment,
    )

    assert result.returncode == 0, result.stderr
    assert f"Run: {_BENCHMARK_RUN_ID}" in result.stdout
    assert "state=complete" in result.stdout
    assert result.stdout.count("# Synthetic local core benchmark summary") == 1
    assert result.stdout.index("# Synthetic local core benchmark summary") < result.stdout.index("DONE:")
    assert remote_directory.is_dir()
    assert Path(environment["FAKE_BENCHMARK_PUBLISHED_FILE"]).is_file()
    log_text = log.read_text(encoding="utf-8")
    assert "materialize-core-benchmark-inputs" in log_text
    assert "submit-core-benchmark" in log_text
    assert log_text.index("materialize-core-benchmark-inputs") < log_text.index("submit-core-benchmark")
    assert "publish-transferred-core-benchmark" in log_text
    assert "<build-campaign-datasets>" not in log_text
    assert f"<--expected-inventory-sha256>\n<{_BENCHMARK_INVENTORY_SHA}>" in log_text
    assert f"<--expected-file-count>\n<{_BENCHMARK_FILE_COUNT}>" in log_text
    assert f"<--expected-size-bytes>\n<{_BENCHMARK_SIZE_BYTES}>" in log_text


def test_core_benchmark_failure_does_not_finalize(tmp_path: Path) -> None:
    """Stop a failed benchmark before terminal publication."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    environment["FAKE_BENCHMARK_STATE"] = "work_unit_failed"

    result = _run(
        workflow,
        ["run", str(_benchmark_suite(workflow)), *_remote_options(), "--keep-cpu-source"],
        environment,
    )

    assert result.returncode != 0
    assert "<finalize-core-benchmark>" not in log.read_text(encoding="utf-8")


def test_slow_host_publication_emits_heartbeat_and_fast_path_is_quiet(
    tmp_path: Path,
) -> None:
    """Emit bounded semantic progress only while host publication is slow."""
    workflow, _log, environment, _storage, mirror = _harness(tmp_path)
    _seed_transfer(mirror, environment)
    environment["FAKE_PUBLISH_DELAY_SECONDS"] = "2"
    environment["GENERATION_CONSOLE_HEARTBEAT_SECONDS"] = "1"

    slow = _run(
        workflow,
        ["run", str(_campaign(workflow)), *_remote_options(), "--keep-cpu-source"],
        environment,
    )

    assert slow.returncode == 0, slow.stderr
    assert "stage=Host publication" in slow.stderr
    assert "operation=validating destination inventory and hashes" in slow.stderr
    assert "directories_completed=5/5" in slow.stderr
    assert "elapsed=" in slow.stderr
    assert "last_progress_at=" in slow.stderr
    assert "heartbeat=active" in slow.stderr
    assert "eta=unavailable" in slow.stderr

    fast_root = tmp_path / "fast"
    fast_root.mkdir()
    fast_workflow, _log, fast_environment, _storage, fast_mirror = _harness(fast_root)
    _seed_transfer(fast_mirror, fast_environment)
    fast_environment["GENERATION_CONSOLE_HEARTBEAT_SECONDS"] = "1"

    fast = _run(
        fast_workflow,
        [
            "run",
            str(_campaign(fast_workflow)),
            *_remote_options(),
            "--keep-cpu-source",
        ],
        fast_environment,
    )

    assert fast.returncode == 0, fast.stderr
    assert "heartbeat=active" not in fast.stderr


def test_slow_paired_finalization_emits_parent_and_child_heartbeat(
    tmp_path: Path,
) -> None:
    """Report paired receipt progress with stable parent and child identity."""
    workflow, _log, environment, _storage, _mirror = _harness(tmp_path)
    environment["FAKE_GPU_ALWAYS_VALID"] = "true"
    environment["FAKE_FINALIZE_SMOKE_DELAY_SECONDS"] = "2"
    environment["GENERATION_CONSOLE_HEARTBEAT_SECONDS"] = "1"

    result = _run(
        workflow,
        ["run", str(_smoke_workflow(workflow)), *_remote_options(), "--keep-cpu-source"],
        environment,
    )

    assert result.returncode == 0, result.stderr
    assert "stage=Paired finalizer" in result.stderr
    assert "run_id=workflow__0123456789abcdef" in result.stderr
    assert f"child_run={_RUN_ID}" in result.stderr
    assert "operation=building and validating paired Smoke payload" in result.stderr
    assert "elapsed=" in result.stderr
    assert "last_progress_at=" in result.stderr
    assert "heartbeat=active" in result.stderr
    assert "eta=unavailable" in result.stderr


def test_automatic_collection_is_non_destructive_and_failure_retains_staging(tmp_path: Path) -> None:
    """Protect safe transfer, source retention, and retryable failed publication."""
    workflow, log, environment, storage, mirror = _harness(tmp_path)
    environment["FAKE_LOGIN_PREFLIGHT_STDOUT"] = "true"
    source_directories = _seed_transfer(mirror, environment)
    collected = _run(
        workflow,
        ["run", str(_campaign(workflow)), "--keep-cpu-source", "--cpu-host", "cpu.example", "--remote-root", "/remote/generation root"],
        environment,
    )

    assert collected.returncode == 0, collected.stderr
    assert all((mirror / relative).is_dir() for relative in source_directories)
    log_text = log.read_text(encoding="utf-8")
    assert log_text.count("rsync-start") == 5
    assert f"<{storage}/.incoming/{_RUN_ID}.synthetic/>" in log_text
    assert "<cpu.example:/remote/generation root/storage/./" in log_text
    assert not any((storage / ".incoming").glob("*"))

    failed_root = tmp_path / "failed"
    failed_root.mkdir()
    failed_workflow, failed_log, failed_environment, failed_storage, failed_mirror = _harness(failed_root)
    _seed_transfer(failed_mirror, failed_environment)
    failed_environment["FAKE_PUBLISH_FAIL"] = "true"
    failed = _run(
        failed_workflow,
        ["run", str(_campaign(failed_workflow)), "--keep-cpu-source", "--cpu-host", "cpu.example", "--remote-root", "/remote/generation root"],
        failed_environment,
    )

    assert failed.returncode == 1
    failed_text = failed_log.read_text(encoding="utf-8")
    assert "<publish-transferred-campaign>" in failed_text
    assert any((failed_storage / ".incoming").glob("*"))


def test_all_default_cleanup_orders_every_gate_and_keep_opt_out_retains_source(tmp_path: Path) -> None:
    """Protect default cleanup ordering and the explicit retention opt-out."""
    workflow, log, environment, _storage, mirror = _harness(tmp_path)
    source_directories = _seed_transfer(mirror, environment)
    complete = _run(
        workflow,
        ["run", str(_campaign(workflow)), *_remote_options()],
        environment,
    )

    assert complete.returncode == 0, complete.stderr
    assert all(not (mirror / relative).exists() for relative in source_directories[:3])
    assert (mirror / source_directories[3]).is_dir()
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
    retained_workflow, retained_log, retained_environment, _storage, retained_mirror = _harness(retained_root)
    retained_directories = _seed_transfer(retained_mirror, retained_environment)
    retained = _run(
        retained_workflow,
        [
            "run",
            str(_campaign(retained_workflow)),
            *_remote_options(),
            "--keep-cpu-source",
        ],
        retained_environment,
    )

    assert retained.returncode == 0, retained.stderr
    assert all((retained_mirror / relative).is_dir() for relative in retained_directories)
    assert "cpu-cleanup-authorization" not in retained_log.read_text(encoding="utf-8")


def test_cleanup_destination_accepts_verified_container_alias_and_rejects_unrelated_path(
    tmp_path: Path,
) -> None:
    """Compare cleanup destinations only after verified namespace mapping."""
    workflow, _log, environment, _storage, mirror = _harness(tmp_path)
    _seed_transfer(mirror, environment)
    environment["FAKE_AUTH_DESTINATION_ROOT"] = "/workspace/storage"

    accepted = _run(
        workflow,
        ["run", str(_campaign(workflow)), *_remote_options()],
        environment,
    )

    assert accepted.returncode == 0, accepted.stderr

    rejected_root = tmp_path / "rejected"
    rejected_root.mkdir()
    rejected_workflow, _log, rejected_environment, _storage, rejected_mirror = _harness(rejected_root)
    _seed_transfer(rejected_mirror, rejected_environment)
    rejected_environment["FAKE_AUTH_DESTINATION_ROOT"] = "/unrelated/storage"

    rejected = _run(
        rejected_workflow,
        ["run", str(_campaign(rejected_workflow)), *_remote_options()],
        rejected_environment,
    )

    assert rejected.returncode != 0
    assert "destination differs from GPU storage" in rejected.stderr


@pytest.mark.parametrize(
    "collection_mode",
    [None, "--keep-cpu-source", "--defer-collection"],
)
def test_failure_continuation_preserves_collection_mode(
    tmp_path: Path,
    collection_mode: str | None,
) -> None:
    """Keep operator-selected collection semantics in failure guidance."""
    workflow, _log, environment, _storage, _mirror = _harness(tmp_path)
    environment["FAKE_CAMPAIGN_STATUS_FAIL"] = "true"
    arguments = ["run", str(_campaign(workflow)), *_remote_options()]
    if collection_mode is not None:
        arguments.append(collection_mode)

    failed = _run(workflow, arguments, environment)

    assert failed.returncode != 0
    continuation = next(line for line in failed.stderr.splitlines() if "generation_workflow.sh run" in line)
    if collection_mode is None:
        assert "--keep-cpu-source" not in continuation
        assert "--defer-collection" not in continuation
    else:
        assert collection_mode in continuation


def test_repairable_technical_smoke_continues_without_new_work_units(
    tmp_path: Path,
) -> None:
    """Repair transfer evidence and downstream gates without COMSOL resubmission."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    environment["FAKE_COMPATIBLE_SMOKE_STATUS"] = "compatible_repairable"
    campaign = workflow.parent.parent / "configs/generation/campaigns/steady_flow/technical_smoke.yaml"

    result = _run(
        workflow,
        ["run", str(campaign), *_remote_options(), "--keep-cpu-source"],
        environment,
    )

    assert result.returncode == 0, result.stderr
    log_text = log.read_text(encoding="utf-8")
    assert "campaign-transfer-authority" in log_text
    assert "<repair-transferred-campaign>" in log_text
    assert "<build-campaign-datasets>" in log_text
    assert "<prepare-all-workflow>" in log_text
    assert "<submit-campaign>" not in log_text
    assert "<resume-campaign>" not in log_text
    assert "rsync-start" not in log_text
    assert "<cpu-cleanup-authorization>" not in log_text


def test_failure_preserves_evidence_and_resume_is_idempotent(tmp_path: Path) -> None:
    """Preserve source/evidence on failure and reuse publication during resume."""
    workflow, log, environment, _storage, mirror = _harness(tmp_path)
    source_directories = _seed_transfer(mirror, environment)
    environment["FAKE_BUILD_FAIL"] = "true"
    failed = _run(
        workflow,
        ["run", str(_campaign(workflow)), *_remote_options()],
        environment,
    )

    assert failed.returncode != 0
    assert "dataset build" in failed.stderr.lower()
    assert "workflow_failures/failure-0001.json" in failed.stderr
    assert "local canonical=" in failed.stderr
    assert "CPU bytes retained: 24" in failed.stderr
    failed_commands = [line for line in log.read_text(encoding="utf-8").splitlines() if line.startswith("<bash -l -s --")]
    assert sum(" campaign-source-status " in line for line in failed_commands) == 1
    assert all((mirror / relative).is_dir() for relative in source_directories)
    assert Path(environment["FAKE_GPU_PUBLISHED_FILE"]).is_file()
    first_log = log.read_text(encoding="utf-8")
    assert not any(Path(environment["STORAGE_ROOT"]).joinpath(".incoming").glob("*"))

    environment["FAKE_BUILD_FAIL"] = "false"
    resumed = _run(
        workflow,
        ["run", str(_campaign(workflow)), *_remote_options()],
        environment,
    )
    assert resumed.returncode == 0, resumed.stderr
    after_resume = log.read_text(encoding="utf-8")
    assert after_resume.count("rsync-start") == first_log.count("rsync-start")
    assert all(not (mirror / relative).exists() for relative in source_directories[:3])
    assert (mirror / source_directories[3]).is_dir()

    build_count = after_resume.count("<build-campaign-datasets>")
    cleanup_count = after_resume.count("cleanup-campaign-source")
    repeated = _run(
        workflow,
        ["run", str(_campaign(workflow)), *_remote_options()],
        environment,
    )
    assert repeated.returncode == 0, repeated.stderr
    final_log = log.read_text(encoding="utf-8")
    assert final_log.count("<build-campaign-datasets>") == build_count
    assert final_log.count("cleanup-campaign-source") == cleanup_count


def test_package_only_resume_builds_no_generation_work_units(tmp_path: Path) -> None:
    """Reuse one completed host source without CPU or COMSOL lifecycle work."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    base_receipt = Path(environment["FAKE_DATASETS_COMPLETE_FILE"])
    base_receipt.write_text("immutable-base\n", encoding="utf-8")
    Path(environment["FAKE_WORKFLOW_COMPLETE_FILE"]).touch()
    environment["FAKE_COMPATIBLE_CAMPAIGN_PACKAGE_STATE"] = "extension_required"

    result = _run(
        workflow,
        ["run", str(_campaign(workflow)), *_remote_options()],
        environment,
    )

    assert result.returncode == 0, result.stderr
    assert base_receipt.read_text(encoding="utf-8") == "immutable-base\n"
    assert Path(environment["FAKE_PACKAGE_STATE_READY_FILE"]).is_file()
    log_text = log.read_text(encoding="utf-8")
    assert "<find-compatible-campaign-source>" in log_text
    assert "<build-campaign-datasets>" in log_text
    for forbidden in (
        "<prepare-campaign-inputs>",
        "<plan-campaign>",
        "<submit-campaign>",
        "<resume-campaign>",
        "<publish-transferred-campaign>",
        "<prepare-all-workflow>",
        "cleanup-campaign-source",
        "<record-cpu-cleanup>",
        "rsync-start",
    ):
        assert forbidden not in log_text


def test_partial_remote_failure_publishes_and_remains_resumable(tmp_path: Path) -> None:
    """Publish valid successes, retain failures, and reuse them on partial resume."""
    workflow, log, environment, storage, mirror = _harness(tmp_path)
    source_directories = _seed_transfer(mirror, environment)
    environment["FAKE_CAMPAIGN_STATE"] = "completed_with_failures"
    environment["FAKE_SOURCE_STATE"] = "completed_with_failures"
    partial = _run(
        workflow,
        ["run", str(_campaign(workflow)), *_remote_options()],
        environment,
    )

    assert partial.returncode == 0, partial.stderr
    assert "DONE:" in partial.stdout
    assert "state=completed_with_failures result=PARTIAL" in partial.stdout
    assert all((mirror / relative).is_dir() for relative in source_directories)
    assert not Path(environment["FAKE_DATASETS_COMPLETE_FILE"]).exists()
    assert not Path(environment["FAKE_PACKAGE_STATE_READY_FILE"]).exists()
    first_log = log.read_text(encoding="utf-8")
    assert "<publish-transferred-campaign" in first_log
    assert "--partial" in first_log
    assert "<build-campaign-datasets" in first_log
    assert "cleanup-campaign-source" not in first_log
    assert "<validate-all-workflow" in first_log
    assert " resume-campaign " in first_log
    first_rsync_count = first_log.count("rsync-start")
    first_resume_count = first_log.count(" resume-campaign ")
    first_publish_count = first_log.count("<publish-transferred-campaign")

    for relative in source_directories:
        (storage / relative).mkdir(parents=True, exist_ok=True)

    repeated = _run(
        workflow,
        ["run", str(_campaign(workflow)), *_remote_options()],
        environment,
    )

    assert repeated.returncode == 0, repeated.stderr
    assert "result=PARTIAL" in repeated.stdout
    repeated_log = log.read_text(encoding="utf-8")
    assert repeated_log.count("rsync-start") == first_rsync_count * 2
    assert repeated_log.count(" resume-campaign ") == first_resume_count + 1
    assert repeated_log.count("<publish-transferred-campaign") == first_publish_count * 2
    assert repeated_log.count(f"<--link-dest={storage}>") == len(source_directories)
    assert "cleanup-campaign-source" not in repeated_log


def test_partial_publication_integrity_failure_remains_failed(tmp_path: Path) -> None:
    """Keep destination validation corruption on the global FAILED path."""
    workflow, _log, environment, _storage, mirror = _harness(tmp_path)
    source_directories = _seed_transfer(mirror, environment)
    environment["FAKE_CAMPAIGN_STATE"] = "completed_with_failures"
    environment["FAKE_SOURCE_STATE"] = "completed_with_failures"
    environment["FAKE_PUBLISH_FAIL"] = "true"

    result = _run(
        workflow,
        ["run", str(_campaign(workflow)), *_remote_options()],
        environment,
    )

    assert result.returncode != 0
    assert "FAILED: atomic host publication" in result.stderr
    assert "result=PARTIAL" not in result.stdout
    assert all((mirror / relative).is_dir() for relative in source_directories)


def test_partial_rerun_completes_missing_publication_and_packages(
    tmp_path: Path,
) -> None:
    """Promote only after remaining cases succeed on the existing resume path."""
    workflow, log, environment, _storage, mirror = _harness(tmp_path)
    source_directories = _seed_transfer(mirror, environment)
    environment["FAKE_CAMPAIGN_STATE"] = "completed_with_failures"
    environment["FAKE_SOURCE_STATE"] = "completed_with_failures"
    first = _run(
        workflow,
        ["run", str(_campaign(workflow)), *_remote_options()],
        environment,
    )
    assert first.returncode == 0, first.stderr
    assert "result=PARTIAL" in first.stdout
    assert Path(environment["FAKE_GPU_PARTIAL_FILE"]).is_file()
    assert not Path(environment["FAKE_GPU_PUBLISHED_FILE"]).exists()
    first_resume_count = log.read_text(encoding="utf-8").count(" resume-campaign ")

    environment["FAKE_CAMPAIGN_STATE"] = "successful"
    environment["FAKE_SOURCE_STATE"] = "successful"
    completed = _run(
        workflow,
        ["run", str(_campaign(workflow)), *_remote_options()],
        environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert "state=successful result=OK" in completed.stdout
    assert Path(environment["FAKE_GPU_PUBLISHED_FILE"]).is_file()
    assert Path(environment["FAKE_DATASETS_COMPLETE_FILE"]).is_file()
    assert Path(environment["FAKE_PACKAGE_STATE_READY_FILE"]).is_file()
    assert all(not (mirror / relative).exists() for relative in source_directories[:3])
    assert (mirror / source_directories[3]).is_dir()
    final_log = log.read_text(encoding="utf-8")
    assert final_log.count(" resume-campaign ") == first_resume_count + 1
    assert " validate-campaign-terminal " in final_log
    assert "<build-campaign-datasets>" in final_log
    assert "cleanup-campaign-source" in final_log


@pytest.mark.parametrize("collection_mode", ["--defer-collection", "--keep-cpu-source"])
def test_background_launch_starts_one_exact_tmux_child(
    tmp_path: Path,
    collection_mode: str,
) -> None:
    """Preserve exact child argv and actionable detached-session output."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    campaign = _campaign(workflow)
    result = _run(
        workflow,
        [
            "run",
            str(campaign),
            *_remote_options(),
            collection_mode,
            "--background",
        ],
        environment,
    )

    assert result.returncode == 0, result.stderr
    assert "BACKGROUND STARTED" in result.stdout
    assert "workflow_session_id=gw-20260818T154501Z-run-01234567" in result.stdout
    assert "tmux attach-session -t gw-run-154501-01234567" in result.stdout
    assert "press Ctrl+B, then D" in result.stdout
    assert "background-status gw-20260818T154501Z-run-01234567" in result.stdout
    assert "tail -n 100 -F" in result.stdout
    state_root = Path(environment["FAKE_TMUX_START_COUNT_FILE"]).parent
    assert (state_root / "tmux-start-count").read_text(encoding="utf-8").strip() == "1"
    child_arguments = (state_root / "background-child-arguments").read_text(encoding="utf-8").splitlines()
    assert child_arguments[0] == "run"
    assert str(campaign) in child_arguments
    assert collection_mode in child_arguments
    assert "--background" not in child_arguments
    assert child_arguments[child_arguments.index("--git-commit") + 1] == _COMMIT
    assert child_arguments[child_arguments.index("--cpu-host") + 1] == "cpu.example"
    assert "cancel-campaign" not in log.read_text(encoding="utf-8")

    environment["FAKE_GIT_STATUS"] = " M unrelated-development-file\n"
    status = _run(
        workflow,
        ["background-status", "gw-20260818T154501Z-run-01234567"],
        environment,
    )
    assert status.returncode == 0, status.stderr
    assert "workflow_state=running" in status.stdout
    assert "Attach:" in status.stdout
    listing = _run(workflow, ["background-list"], environment)
    assert listing.returncode == 0, listing.stderr
    assert listing.stdout.strip() == "No background workflow sessions."


def test_background_launch_requires_a_clean_stable_host_checkout(tmp_path: Path) -> None:
    """Reject dirty stable bootstrap code before session metadata or tmux mutation."""
    workflow, _log, environment, storage, _mirror = _harness(tmp_path)
    environment["FAKE_GIT_STATUS"] = " M scripts/generation_workflow.sh\n"

    result = _run(
        workflow,
        [
            "run",
            str(_campaign(workflow)),
            *_remote_options(),
            "--defer-collection",
            "--background",
        ],
        environment,
    )

    assert result.returncode == 1
    assert "stable host checkout to be clean and committed" in result.stderr
    assert not Path(environment["FAKE_TMUX_START_COUNT_FILE"]).exists()
    assert not storage.exists()


def test_background_launch_accepts_an_immediately_completed_child(tmp_path: Path) -> None:
    """Treat a durable terminal result as success when tmux exits immediately."""
    workflow, _log, environment, _storage, _mirror = _harness(tmp_path)
    environment["FAKE_TMUX_IMMEDIATE_EXIT"] = "true"

    result = _run(
        workflow,
        [
            "run",
            str(_campaign(workflow)),
            *_remote_options(),
            "--defer-collection",
            "--background",
        ],
        environment,
    )

    assert result.returncode == 0, result.stderr
    assert "BACKGROUND COMPLETED" in result.stdout
    assert "workflow_session_id=gw-20260818T154501Z-run-01234567" in result.stdout
    assert "exit_code=0" in result.stdout
    assert "final_stage=DONE: synthetic" in result.stdout
    assert "tmux attach-session" not in result.stdout


def test_background_launch_rejects_administration_and_recursion(tmp_path: Path) -> None:
    """Reject administrative or recursively detached controller invocations."""
    workflow, _log, environment, _storage, _mirror = _harness(tmp_path)
    unsupported = _run(
        workflow,
        ["status", _RUN_ID, *_remote_options(), "--background"],
        environment,
    )
    assert unsupported.returncode == 2
    assert "--background is supported only by run CONFIG" in unsupported.stderr
    environment["GENERATION_WORKFLOW_BACKGROUND_CHILD"] = "1"
    recursive = _run(
        workflow,
        ["run", str(_campaign(workflow)), *_remote_options(), "--background"],
        environment,
    )
    assert recursive.returncode == 2
    assert "cannot create another tmux session" in recursive.stderr
    assert not Path(environment["FAKE_TMUX_START_COUNT_FILE"]).exists()
