# ruff: noqa: S101, S603, PLR2004
"""Host workflow lifecycle tests using fake Docker, remote, Slurm, and COMSOL."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from src import generation

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
_CPU_BOOTSTRAP_URL = "https://github.com/Rinovative/grainlegumes-pino-drying.git"


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
    for ((index=0; index<${#arguments[@]}; index++)); do
      if [[ "${arguments[index]}" == python ]]; then
        python_index="${index}"
      fi
    done
    (( python_index >= 0 )) || { printf 'Docker invocation omitted Python.\n' >&2; exit 91; }
    translated=()
    for argument in "${arguments[@]:python_index+1}"; do
      case "${argument}" in
        /workspace/repo) argument="${FAKE_PROJECT_ROOT}" ;;
        /workspace/repo/*) argument="${FAKE_PROJECT_ROOT}/${argument#/workspace/repo/}" ;;
        /workspace/storage) argument="${STORAGE_ROOT}" ;;
        /workspace/storage/*) argument="${STORAGE_ROOT}/${argument#/workspace/storage/}" ;;
      esac
      translated+=("${argument}")
    done
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
        fake_bin / "git",
        r"""#!/usr/bin/env bash
set -euo pipefail
printf 'git <%s>\n' "$*" >> "${FAKE_COMMAND_LOG}"
case " $* " in
  *" rev-parse --show-toplevel "*) printf '%s\n' "${FAKE_PROJECT_ROOT}" ;;
  *" rev-parse HEAD "*) printf '%s\n' "${FAKE_GIT_COMMIT}" ;;
  *" status --porcelain "*) ;;
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
if [[ "${FAKE_CPU_LOGIN_RSYNC_MISSING:-false}" == true \
  && "${payload}" == *'generation_cpu_login_preflight.sh'* ]]; then
  printf 'CPU login prerequisite missing: rsync (blocks transfer).\n' >&2
  exit 71
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
elif [[ " $* " == *' campaign-status '* && " $* " == *' --format state '* ]]; then
  printf '%s\n' "${FAKE_CAMPAIGN_STATE}"
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
  printf '%s\n' '{"campaign_run_id":"steady_flow_family_generalization__0123456789abcdef","state":"submitted"}'
elif [[ " $* " == *' plan-campaign'* ]]; then
  printf '%s\n' '{"filesystem_mutated":false,"state":"planned"}'
elif [[ "${payload}" == *'sbatch --wait --parsable'* ]]; then
  printf '%s\n' '12345'
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
if [[ " $* " == *' list-campaigns '* ]]; then
  printf '%s\n' '{}'
  exit 0
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
    printf 'execution\tfamily_generalization\t8\t01:00:00\t48\t1\t1\t-\tfixture.cluster\tslurm\tfixture\t'\
'Python/fixture-3.12\tComsol/fixture-9.9\tfixture-python\tfixture-comsol\n'
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
            "FAKE_GIT_COMMIT": _COMMIT,
            "FAKE_REALPATH": shutil.which("realpath", path=os.defpath) or "/usr/bin/realpath",
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
            "FAKE_SOURCE_STATE": "publication_complete",
            "FAKE_AUTHORIZED_BYTES": str(_AUTHORIZED_BYTES),
            "FAKE_AUTHORIZATION_SHA": _AUTHORIZATION_SHA,
            "FAKE_TRANSFER_SHA": _TRANSFER_SHA,
            "FAKE_DATASET_SHA": _DATASET_SHA,
            "FAKE_WORKFLOW_SHA": _WORKFLOW_SHA,
            "FAKE_INVENTORY_SHA": _INVENTORY_SHA,
            "FAKE_CLEANUP_RECEIPT_SHA": _CLEANUP_RECEIPT_SHA,
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


def _pilot_campaign(workflow: Path) -> Path:
    """Return the copied dedicated pilot-check configuration."""
    return workflow.parent.parent / "configs/generation/campaigns/transient_drying/pilot_check.yaml"


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


def test_smoke_translates_logical_campaigns_across_all_path_domains(tmp_path: Path) -> None:
    """Reproduce the native smoke path boundary from an arbitrary host checkout."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    environment["FAKE_GPU_ALWAYS_VALID"] = "true"

    result = _run(workflow, ["smoke", "--keep-cpu-source"], environment)

    assert result.returncode == 0, result.stderr
    host_root = workflow.parent.parent
    remote_root = Path("/remote/home/grainlegumes-generation/repo")
    relative_campaigns = (
        Path("configs/generation/campaigns/steady_flow/technical_smoke.yaml"),
        Path("configs/generation/campaigns/transient_drying/technical_smoke.yaml"),
    )
    log_text = log.read_text(encoding="utf-8")
    for relative in relative_campaigns:
        assert f"<{host_root / relative}>" in log_text
        assert f"</workspace/repo/{relative.as_posix()}>" in log_text
        assert str(remote_root / relative) in log_text
    assert "Bare-host realpath received a Docker-only path" not in result.stderr
    assert "forbidden-host-python" not in log_text
    assert "submit-campaign" in log_text


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


def test_preflight_stops_on_missing_login_rsync_before_slurm_submission(tmp_path: Path) -> None:
    """Route transfer requirements through the login gate before allocation."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    environment["FAKE_CPU_LOGIN_RSYNC_MISSING"] = "true"

    result = _run(
        workflow,
        ["preflight", str(_campaign(workflow)), *_remote_options()],
        environment,
    )

    assert result.returncode != 0
    assert "CPU login prerequisite missing: rsync (blocks transfer)." in result.stderr
    log_text = log.read_text(encoding="utf-8")
    assert "generation_cpu_login_preflight.sh" in log_text
    assert "sbatch --wait --parsable" not in log_text


def test_preflight_passes_exact_cpu_checkout_to_relocated_worker(tmp_path: Path) -> None:
    """Bind the direct sbatch worker to the explicit repository and launch commit."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)

    result = _run(
        workflow,
        ["preflight", str(_campaign(workflow)), *_remote_options()],
        environment,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.rstrip().endswith("12345")
    log_text = log.read_text(encoding="utf-8")
    assert '--export="ALL,GENERATION_GIT_COMMIT=${commit}"' in log_text
    worker_index = log_text.index('"${repository}/scripts/generation_cpu_smoke.sh"')
    repository_index = log_text.index('"${repository}" "${venv}"', worker_index)
    assert repository_index - worker_index < 100


def test_setup_is_read_only_by_default_and_execute_is_explicit(tmp_path: Path) -> None:
    """Protect setup dry-run, exact modules, and noninteractive SSH execution."""
    workflow, log, environment, storage, _mirror = _harness(tmp_path)
    dry_run = _run(workflow, ["setup-cpu", *_remote_options()], environment)
    assert dry_run.returncode == 0, dry_run.stderr
    assert "Mode: dry-run" in dry_run.stdout
    assert "Dry run: no remote files or jobs were created." in dry_run.stdout
    assert "Python/fixture-3.12" in dry_run.stdout
    assert "Comsol/fixture-9.9" in dry_run.stdout
    assert f"Repository source: {_CPU_BOOTSTRAP_URL}" in dry_run.stdout
    assert f"git clone --no-checkout {_CPU_BOOTSTRAP_URL}" in dry_run.stdout
    assert f"fetch origin {_COMMIT}" in dry_run.stdout
    assert f"checkout --detach {_COMMIT}" in dry_run.stdout
    assert "git@github.com" not in dry_run.stdout
    assert "ssh-agent" not in dry_run.stdout
    assert "ssh-add" not in dry_run.stdout
    assert "known_hosts" not in dry_run.stdout
    assert storage.is_dir()
    assert not any(storage.iterdir())
    dry_log = log.read_text(encoding="utf-8")
    assert "forbidden-host-python" not in dry_log
    assert "<--rm>" in dry_log
    assert "<--network>\n<none>" in dry_log
    assert "<--name>" not in dry_log
    assert "<type=bind,source=" in dry_log
    assert ",target=/workspace/repo,readonly>" in dry_log
    assert ",target=/workspace/storage>" in dry_log
    assert "remote set-url" not in dry_log
    assert "remote get-url origin" not in dry_log

    execute = _run(workflow, ["setup-cpu", *_remote_options(), "--execute"], environment)
    assert execute.returncode == 0, execute.stderr
    log_text = log.read_text(encoding="utf-8")
    assert "<BatchMode=yes>" in log_text
    assert "checkout --detach" in log_text
    assert " Python/fixture-3.12 Comsol/fixture-9.9 " in log_text
    assert 'module load "${python_module}"' in log_text
    assert 'module load "${comsol_module}"' in log_text
    assert "[generation-cpu]" in log_text
    assert "forbidden-host-python" not in log_text

    home_based = _run(
        workflow,
        ["setup-cpu", "--cpu-host", "cpu.example", "--git-commit", _COMMIT],
        environment,
    )
    assert home_based.returncode == 0, home_based.stderr
    assert "Repository: /remote/home/grainlegumes-generation/repo" in home_based.stdout
    assert "Persistent storage: /remote/home/grainlegumes-generation/storage" in home_based.stdout
    assert "Venv: /remote/home/grainlegumes-generation/venv" in home_based.stdout
    assert f"Repository source: {_CPU_BOOTSTRAP_URL}" in home_based.stdout
    assert (f"git clone --no-checkout {_CPU_BOOTSTRAP_URL} /remote/home/grainlegumes-generation/repo") in home_based.stdout
    assert (f"git -C /remote/home/grainlegumes-generation/repo fetch origin {_COMMIT}") in home_based.stdout
    assert (f"git -C /remote/home/grainlegumes-generation/repo checkout --detach {_COMMIT}") in home_based.stdout

    unsafe = _run(workflow, ["setup-cpu", "--cpu-host", "bad;host"], environment)
    assert unsafe.returncode == 2
    assert "Unsafe CPU host" in unsafe.stderr


def test_local_docker_failure_is_clear_and_stops_before_remote_mutation(tmp_path: Path) -> None:
    """Surface canonical Docker failures without falling back to native Python."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    environment["FAKE_DOCKER_FAIL"] = "true"
    result = _run(workflow, ["setup-cpu", *_remote_options()], environment)
    assert result.returncode == 1
    assert "Local project Python requires the Docker daemon" in result.stderr
    log_text = log.read_text(encoding="utf-8")
    assert "forbidden-host-python" not in log_text
    assert "ssh-start" not in log_text


def test_plan_launch_status_and_resource_rejection_are_canonical(tmp_path: Path) -> None:
    """Protect planning, launch, unified storage status, and resource caps."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    plan = _run(workflow, ["plan", str(_campaign(workflow)), *_remote_options(), *_selection_options()], environment)
    assert plan.returncode == 0, plan.stderr
    assert '"filesystem_mutated":false' in plan.stdout
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
    configured_log = log.read_text(encoding="utf-8")
    assert " future.profile::batch plan-campaign " in configured_log
    assert "--max-nodes" not in configured_log
    assert "--cases-per-node" not in configured_log
    assert "--max-parallel-cases" not in configured_log

    skipped = _run(
        workflow,
        [
            "plan",
            str(_campaign(workflow)),
            *_remote_options(),
            "--skip-extreme-family-ood",
        ],
        environment,
    )
    assert skipped.returncode == 0, skipped.stderr
    assert " plan-campaign '' true Python/fixture-3.12" in log.read_text(encoding="utf-8")
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
    assert "cannot be combined with --only-batch" in incompatible.stderr

    launch = _run(workflow, ["launch", str(_campaign(workflow)), *_remote_options(), *_selection_options()], environment)
    assert launch.returncode == 0, launch.stderr
    assert f"Campaign run ID: {_RUN_ID}" in launch.stdout
    assert "submit-campaign" in log.read_text(encoding="utf-8")

    status = _run(
        workflow,
        ["status", _RUN_ID, "--cpu-host", "cpu.example", "--remote-root", "/remote/generation root"],
        environment,
    )
    assert status.returncode == 0, status.stderr
    assert "GPU storage status:" in status.stdout
    assert '"role":"gpu"' in status.stdout
    assert "CPU storage status:" in status.stdout
    assert '"role":"cpu"' in status.stdout

    for obsolete in (
        "--max-nodes",
        "--cases-per-node",
        "--cores-per-case",
        "--max-parallel-cases",
        "--wall-time",
    ):
        rejected = _run(
            workflow,
            ["plan", str(_campaign(workflow)), *_remote_options(), obsolete, "1"],
            environment,
        )
        assert rejected.returncode == 2
        assert f"Unsupported option: {obsolete}" in rejected.stderr


def test_pilot_command_uses_config_default_and_explicit_fast_override(tmp_path: Path) -> None:
    """Protect the copyable pilot command without a hidden mandatory count."""
    workflow, _log, environment, _storage, _mirror = _harness(tmp_path)
    normal = _run(
        workflow,
        ["pilot-check", str(_pilot_campaign(workflow)), *_remote_options()],
        environment,
    )
    assert normal.returncode != 2
    assert "Pilot cases: 5 materials x 4 = 20 total." in normal.stdout

    fast_root = tmp_path / "fast"
    fast_root.mkdir()
    fast_workflow, _fast_log, fast_environment, _fast_storage, _fast_mirror = _harness(fast_root)
    fast = _run(
        fast_workflow,
        [
            "pilot-check",
            str(_pilot_campaign(fast_workflow)),
            "--cases-per-material",
            "1",
            *_remote_options(),
        ],
        fast_environment,
    )
    assert fast.returncode != 2
    assert "Pilot cases: 5 materials x 1 = 5 total." in fast.stdout
    help_result = _run(workflow, ["--help"], environment)
    assert "pilot-check CAMPAIGN [--cases-per-material N]" in help_result.stderr
    assert "duration-check" not in help_result.stderr


def test_high_level_core_benchmark_routes_through_docker_cpu_and_dedicated_transfer(tmp_path: Path) -> None:
    """Exercise all-four submission, serial collection, and one-variant recovery routing."""
    workflow, log, environment, _storage, mirror = _harness(tmp_path)
    relative = environment["FAKE_BENCHMARK_RELATIVE"]
    remote_directory = mirror / relative
    remote_directory.mkdir(parents=True)
    (remote_directory / "summary.json").write_text("{}\n", encoding="utf-8")

    result = _run(workflow, ["benchmark-cores", *_remote_options()], environment)
    assert result.returncode == 0, result.stderr
    assert f"Core benchmark run ID: {_BENCHMARK_RUN_ID}" in result.stdout
    assert "Synthetic local core benchmark summary" in result.stdout
    assert "CPU benchmark evidence retained" in result.stdout
    assert remote_directory.is_dir()
    assert Path(environment["FAKE_BENCHMARK_PUBLISHED_FILE"]).is_file()
    log_text = log.read_text(encoding="utf-8")
    for command in (
        "inspect-core-benchmark",
        "validate-real-smoke",
        "plan-core-benchmark",
        "submit-core-benchmark",
        "core-benchmark-status",
        "finalize-core-benchmark",
        "core-benchmark-transfer-plan",
        "publish-transferred-core-benchmark",
    ):
        assert command in log_text
    assert "<build-campaign-datasets>" not in log_text
    assert f"<--expected-inventory-sha256>\n<{_BENCHMARK_INVENTORY_SHA}>" in log_text
    assert f"<--expected-file-count>\n<{_BENCHMARK_FILE_COUNT}>" in log_text
    assert f"<--expected-size-bytes>\n<{_BENCHMARK_SIZE_BYTES}>" in log_text
    assert "forbidden-host-python" not in log_text

    recovered = _run(
        workflow,
        ["benchmark-cores", "--variant", "cores_08", *_remote_options()],
        environment,
    )
    assert recovered.returncode == 0, recovered.stderr
    recovered_log = log.read_text(encoding="utf-8")
    assert "<--variant>\n<cores_08>" in recovered_log
    assert "GPU benchmark publication validated and reused" in recovered.stdout


def test_core_benchmark_failure_reports_retry_without_premature_finalization(tmp_path: Path) -> None:
    """Stop on one failed repetition and preserve the remaining serial sequence."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    environment["FAKE_BENCHMARK_STATE"] = "retry_required"

    result = _run(workflow, ["benchmark-cores", *_remote_options()], environment)

    assert result.returncode != 0
    assert '"variant_id":"cores_16"' in result.stdout
    assert "requires retry" in result.stderr
    assert "<finalize-core-benchmark>" not in log.read_text(encoding="utf-8")


def test_collect_is_non_destructive_and_publication_failure_retains_staging(tmp_path: Path) -> None:
    """Protect safe rsync, source retention, staging cleanup, and retry evidence."""
    workflow, log, environment, storage, mirror = _harness(tmp_path)
    source_directories = _seed_transfer(mirror, environment)
    collected = _run(
        workflow,
        ["collect", _RUN_ID, "--cpu-host", "cpu.example", "--remote-root", "/remote/generation root"],
        environment,
    )
    assert collected.returncode == 0, collected.stderr
    assert "CPU source retained" in collected.stdout
    assert all((mirror / relative).is_dir() for relative in source_directories)
    log_text = log.read_text(encoding="utf-8")
    assert log_text.count("rsync-start") == 4
    assert f"<{storage}/01_generation/.state/transfer-staging/{_RUN_ID}.synthetic/>" in log_text
    assert "<cpu.example:/remote/generation root/storage/./" in log_text
    assert "cpu.example:/workspace/storage" not in log_text
    assert "<--delete>" not in log_text
    assert "<--remove-source-files>" not in log_text
    assert "<cleanup-campaign-source>" not in log_text
    assert not any((storage / "01_generation/.state/transfer-staging").glob("*"))

    failed_root = tmp_path / "failed"
    failed_root.mkdir()
    failed_workflow, failed_log, failed_environment, _failed_storage, failed_mirror = _harness(failed_root)
    _seed_transfer(failed_mirror, failed_environment)
    failed_environment["FAKE_PUBLISH_FAIL"] = "true"
    failed = _run(
        failed_workflow,
        ["collect", _RUN_ID, "--cpu-host", "cpu.example", "--remote-root", "/remote/generation root"],
        failed_environment,
    )
    assert failed.returncode == 1
    assert "staging retained" in failed.stderr
    failed_text = failed_log.read_text(encoding="utf-8")
    assert "<publish-transferred-campaign>" in failed_text
    assert "<cleanup-transfer-staging>" not in failed_text


def test_all_default_cleanup_orders_every_gate_and_keep_opt_out_retains_source(tmp_path: Path) -> None:
    """Protect default cleanup ordering and the sole retention opt-out."""
    workflow, log, environment, _storage, mirror = _harness(tmp_path)
    source_directories = _seed_transfer(mirror, environment)
    complete = _run(
        workflow,
        ["all", str(_campaign(workflow)), *_remote_options(), *_selection_options()],
        environment,
    )
    assert complete.returncode == 0, complete.stderr
    assert "CPU source cleanup after complete success: enabled" in complete.stdout
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
    retained_workflow, retained_log, retained_environment, _storage, retained_mirror = _harness(retained_root)
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
    assert "CPU source cleanup after complete success: disabled" in retained.stdout
    assert all((retained_mirror / relative).is_dir() for relative in retained_directories)
    retained_text = retained_log.read_text(encoding="utf-8")
    assert "<--keep-cpu-source>" in retained_text
    assert "cleanup-campaign-source" not in retained_text


def test_failure_preserves_cpu_and_gpu_then_resume_reuses_publication_idempotently(tmp_path: Path) -> None:
    """Protect no-cleanup failure semantics, reusable GPU data, resume, and repeats."""
    workflow, log, environment, _storage, mirror = _harness(tmp_path)
    source_directories = _seed_transfer(mirror, environment)
    environment["FAKE_BUILD_FAIL"] = "true"
    failed = _run(
        workflow,
        ["all", str(_campaign(workflow)), *_remote_options(), *_selection_options()],
        environment,
    )
    assert failed.returncode != 0
    assert "Stage: dataset build, inspection, and loader smokes" in failed.stderr
    assert f"CPU bytes retained: {_AUTHORIZED_BYTES}" in failed.stderr
    assert "./scripts/generation_workflow.sh resume" in failed.stderr
    assert all((mirror / relative).is_dir() for relative in source_directories)
    assert Path(environment["FAKE_GPU_PUBLISHED_FILE"]).is_file()
    first_log = log.read_text(encoding="utf-8")
    assert "cleanup-campaign-source" not in first_log
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
    assert "already complete and validated" in repeated.stdout
    final_log = log.read_text(encoding="utf-8")
    assert final_log.count("<build-campaign-datasets>") == build_count
    assert final_log.count("cleanup-campaign-source") == cleanup_count


def test_partial_remote_and_detached_modes_never_cleanup(tmp_path: Path) -> None:
    """Protect no partial cleanup and detached submit-only behavior."""
    workflow, log, environment, _storage, mirror = _harness(tmp_path)
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
    partial_text = log.read_text(encoding="utf-8")
    assert "rsync-start" not in partial_text
    assert "cleanup-campaign-source" not in partial_text

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
    assert "Resume-all command:" in detached.stdout
    assert "./scripts/generation_workflow.sh resume" in detached.stdout
    assert all((detached_mirror / relative).is_dir() for relative in detached_directories)
    detached_text = detached_log.read_text(encoding="utf-8")
    assert "submit-campaign" in detached_text
    assert "rsync-start" not in detached_text
    assert "<build-campaign-datasets>" not in detached_text
    assert "cleanup-campaign-source" not in detached_text
