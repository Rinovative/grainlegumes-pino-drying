# ruff: noqa: S101, S603, PLR2004
"""Host workflow lifecycle tests using fake remote, Slurm, COMSOL, and Python."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

_COMMIT = "a" * 40
_RUN_ID = "steady_flow_family_generalization__0123456789abcdef"
_BATCH_NAME = "steady_flow__lentil__natural"
_AUTHORIZATION_SHA = "1" * 64
_TRANSFER_SHA = "2" * 64
_DATASET_SHA = "3" * 64
_WORKFLOW_SHA = "4" * 64
_INVENTORY_SHA = "5" * 64
_CLEANUP_RECEIPT_SHA = "6" * 64
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
    scripts.mkdir(parents=True)
    campaigns.mkdir(parents=True)
    workflow = scripts / "generation_workflow.sh"
    shutil.copyfile(repository / "scripts/generation_workflow.sh", workflow)
    workflow.chmod(workflow.stat().st_mode | 0o111)
    shutil.copyfile(
        repository / "configs/generation/campaigns/steady_flow/family_generalization.yaml",
        campaigns / "family_generalization.yaml",
    )
    log = tmp_path / "commands.log"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _executable(
        fake_bin / "git",
        r"""#!/usr/bin/env bash
set -euo pipefail
printf 'git <%s>\n' "$*" >> "${FAKE_COMMAND_LOG}"
case " $* " in
  *" rev-parse HEAD "*) printf '%s\n' "${FAKE_GIT_COMMIT}" ;;
  *" status --porcelain "*) ;;
  *" remote get-url origin "*) printf '%s\n' 'https://github.com/Rinovative/grainlegumes-pino-drying.git' ;;
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
if [[ "${payload}" == *'${HOME}'* && "${payload}" != *'root="$1"'* ]]; then
  printf '%s\n' '/remote/home'
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
[[ " $* " != *' -c '* ]] || exit 0
storage=''
directory=''
arguments=("$@")
for ((index=0; index<${#arguments[@]}; index++)); do
  case "${arguments[index]}" in
    --storage-root) storage="${arguments[index+1]}" ;;
    --directory) directory="${arguments[index+1]}" ;;
  esac
done
if [[ " $* " == *' validate-published-campaign '* ]]; then
  [[ -f "${FAKE_GPU_PUBLISHED_FILE}" ]]
elif [[ " $* " == *' create-transfer-staging '* ]]; then
  staging="${storage}/01_generation/.state/transfer-staging/${FAKE_RUN_ID}.synthetic"
  mkdir -p "${staging}"
  printf '%s\n' "${staging}"
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
            "FAKE_REMOTE_MIRROR": str(mirror),
            "FAKE_RUN_ID": _RUN_ID,
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
            "FAKE_DATASETS_COMPLETE_FILE": str(state_root / "datasets-complete"),
            "FAKE_WORKFLOW_READY_FILE": str(state_root / "workflow-ready"),
            "FAKE_WORKFLOW_COMPLETE_FILE": str(state_root / "workflow-complete"),
            "FAKE_REMOTE_CLEANED_FILE": str(state_root / "remote-cleaned"),
            "FAKE_SOURCE_DIRECTORIES_FILE": str(source_directories_file),
            "GENERATION_LOCAL_PYTHON": str(local_python),
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


def _resource_options() -> list[str]:
    """Return one valid explicit Slurm resource request."""
    return [
        "--max-nodes",
        "2",
        "--cases-per-node",
        "2",
        "--cores-per-case",
        "8",
        "--max-parallel-cases",
        "3",
        "--only-batch",
        _BATCH_NAME,
        "--wall-time",
        "01:00:00",
    ]


def _campaign(workflow: Path) -> Path:
    """Return the copied campaign configuration."""
    return workflow.parent.parent / "configs/generation/campaigns/steady_flow/family_generalization.yaml"


def _seed_transfer(mirror: Path, environment: dict[str, str]) -> tuple[str, ...]:
    """Create one complete fake terminal transfer tree and TSV plan."""
    campaign_directory = f"01_generation/meta/campaigns/{_RUN_ID}"
    batch_id = f"{_BATCH_NAME}__fedcba9876543210"
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


def test_setup_is_read_only_by_default_and_execute_is_explicit(tmp_path: Path) -> None:
    """Protect setup dry-run, exact modules, and noninteractive SSH execution."""
    workflow, log, environment, storage, _mirror = _harness(tmp_path)
    dry_run = _run(workflow, ["setup-cpu", *_remote_options()], environment)
    assert dry_run.returncode == 0, dry_run.stderr
    assert "Mode: dry-run" in dry_run.stdout
    assert "Dry run: no remote files or jobs were created." in dry_run.stdout
    assert "Python/3.10" in dry_run.stdout
    assert "Comsol/v6.4" in dry_run.stdout
    assert not storage.exists()

    execute = _run(workflow, ["setup-cpu", *_remote_options(), "--execute"], environment)
    assert execute.returncode == 0, execute.stderr
    log_text = log.read_text(encoding="utf-8")
    assert "<BatchMode=yes>" in log_text
    assert "checkout --detach" in log_text
    assert "module load Python/3.10" in log_text
    assert "module load Comsol/v6.4" in log_text

    unsafe = _run(workflow, ["setup-cpu", "--cpu-host", "bad;host"], environment)
    assert unsafe.returncode == 2
    assert "Unsafe CPU host" in unsafe.stderr


def test_plan_launch_status_and_resource_rejection_are_canonical(tmp_path: Path) -> None:
    """Protect planning, launch, unified storage status, and resource caps."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    plan = _run(workflow, ["plan", str(_campaign(workflow)), *_remote_options(), *_resource_options()], environment)
    assert plan.returncode == 0, plan.stderr
    assert '"filesystem_mutated":false' in plan.stdout

    launch = _run(workflow, ["launch", str(_campaign(workflow)), *_remote_options(), *_resource_options()], environment)
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

    rejected = _run(
        workflow,
        [
            "plan",
            str(_campaign(workflow)),
            *_remote_options(),
            "--max-nodes",
            "1",
            "--cases-per-node",
            "5",
            "--cores-per-case",
            "8",
            "--max-parallel-cases",
            "1",
        ],
        environment,
    )
    assert rejected.returncode == 2
    assert "exceeds 32" in rejected.stderr


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
        ["all", str(_campaign(workflow)), *_remote_options(), *_resource_options()],
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
            *_resource_options(),
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
        ["all", str(_campaign(workflow)), *_remote_options(), *_resource_options()],
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
        ["all", str(_campaign(workflow)), *_remote_options(), *_resource_options()],
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
        ["all", str(_campaign(detached_workflow)), *_remote_options(), *_resource_options(), "--detach"],
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
