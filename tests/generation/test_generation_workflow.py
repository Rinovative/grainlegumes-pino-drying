# ruff: noqa: S101, S603, PLR2004
"""Host CPU workflow tests using only fake local executables and directories."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

_COMMIT = "a" * 40
_RUN_ID = "steady_flow_family_generalization__0123456789abcdef"
_BATCH_NAME = "steady_flow__lentil__natural"


def _executable(path: Path, content: str) -> Path:
    """Write one test-owned executable."""
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)
    return path


def _harness(tmp_path: Path) -> tuple[Path, Path, dict[str, str], Path, Path]:
    """Return copied workflow, fake command log, environment, storage, and mirror."""
    repository = Path(__file__).resolve().parents[2]
    project = tmp_path / "project"
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
printf 'git %s\n' "$*" >> "${FAKE_COMMAND_LOG}"
case " $* " in
  *" rev-parse HEAD "*) printf '%s\n' "${FAKE_GIT_COMMIT}" ;;
  *" status --porcelain "*) ;;
  *" remote get-url origin "*) printf '%s\n' 'https://github.com/Rinovative/grainlegumes-pino-drying.git' ;;
  *" for-each-ref "*) printf '%s\n' 'refs/remotes/origin/main' ;;
  *) ;;
esac
""",
    )
    _executable(
        fake_bin / "docker",
        r"""#!/usr/bin/env bash
set -euo pipefail
printf 'docker %s\n' "$*" >> "${FAKE_COMMAND_LOG}"
if [[ "${1:-}" == run ]]; then
  printf '%s\n' '{"campaign_name":"steady_flow_family_generalization","batches":[{"batch_name":"steady_flow__lentil__natural"}]}'
fi
""",
    )
    _executable(
        fake_bin / "ssh",
        r"""#!/usr/bin/env bash
set -euo pipefail
payload="$(cat)"
printf 'ssh %s\n' "$*" >> "${FAKE_COMMAND_LOG}"
printf 'ssh-stdin-start\n%s\nssh-stdin-end\n' "${payload}" >> "${FAKE_COMMAND_LOG}"
if [[ "${payload}" == *campaign-transfer-plan* ]]; then
  printf '%b' "${FAKE_TRANSFER_PLAN}"
elif [[ "${payload}" == *campaign-status* ]]; then
  printf '%s\n' '{"campaign_state":"active","suggested_next_command":"status"}'
elif [[ "${payload}" == *submit-campaign* ]]; then
  printf '%s\n' '{"campaign_run_id":"steady_flow_family_generalization__0123456789abcdef","state":"submitted"}'
elif [[ "${payload}" == *'printf '\''%s\\n'\'' "${HOME}"'* ]]; then
  printf '%s\n' '/remote/home'
else
  printf '%s\n' 'fake remote verification complete'
fi
""",
    )
    _executable(
        fake_bin / "rsync",
        r"""#!/usr/bin/env bash
set -euo pipefail
printf 'rsync %s\n' "$*" >> "${FAKE_COMMAND_LOG}"
source_argument="${@: -2:1}"
destination="${@: -1}"
remote_path="${source_argument#*:}"
relative="${remote_path#*/./}"
source="${FAKE_REMOTE_MIRROR}/${relative}"
mkdir -p "${destination}/$(dirname "${relative}")"
cp -a -- "${source}" "${destination}/${relative}"
""",
    )
    _executable(
        scripts / "docker_job.sh",
        r"""#!/usr/bin/env bash
set -euo pipefail
printf 'docker-job %s\n' "$*" >> "${FAKE_COMMAND_LOG}"
""",
    )
    storage = tmp_path / "host-storage"
    mirror = tmp_path / "remote-mirror"
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{environment['PATH']}",
            "FAKE_COMMAND_LOG": str(log),
            "FAKE_GIT_COMMIT": _COMMIT,
            "FAKE_REMOTE_MIRROR": str(mirror),
            "FAKE_TRANSFER_PLAN": "",
            "STORAGE_ROOT": str(storage),
        }
    )
    return workflow, log, environment, storage, mirror


def _run(workflow: Path, arguments: list[str], environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Run one host workflow command through the fake harness."""
    return subprocess.run(
        [str(workflow), *arguments],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_setup_is_dry_run_safe_and_execute_uses_noninteractive_ssh(tmp_path: Path) -> None:
    """Protect printed setup, no-op default, input safety, and fake execute path."""
    workflow, log, environment, storage, _mirror = _harness(tmp_path)
    common = ["--cpu-host", "cpu.example", "--remote-root", "/remote/generation", "--git-commit", _COMMIT]
    dry_run = _run(workflow, ["setup-cpu", *common], environment)
    assert dry_run.returncode == 0, dry_run.stderr
    assert "Mode: dry-run" in dry_run.stdout
    assert "Dry run only" in dry_run.stdout
    assert "/remote/generation/repo" in dry_run.stdout
    assert "Python/3.10" in dry_run.stdout
    assert "Comsol/v6.4" in dry_run.stdout
    assert "generation-cpu" in dry_run.stdout
    assert "--output=/dev/null" in dry_run.stdout
    assert "--error=/dev/null" in dry_run.stdout
    assert not storage.exists()
    assert "ssh " not in log.read_text(encoding="utf-8")

    execute = _run(workflow, ["setup-cpu", *common, "--execute"], environment)
    assert execute.returncode == 0, execute.stderr
    assert "Mode: execute" in execute.stdout
    log_text = log.read_text(encoding="utf-8")
    assert "ssh -o BatchMode=yes" in log_text
    assert _COMMIT in log_text
    assert "checkout --detach" in log_text
    assert "pip install -e" in log_text

    unsafe = _run(
        workflow,
        ["setup-cpu", "--cpu-host", "bad;host", "--remote-root", "/remote/generation"],
        environment,
    )
    assert unsafe.returncode == 2
    assert "CPU host is unsafe" in unsafe.stderr


def test_launch_status_and_unwaited_all_are_resumable(tmp_path: Path) -> None:
    """Protect exact commit launch, execution flags, status, and nonwaiting all."""
    workflow, log, environment, _storage, _mirror = _harness(tmp_path)
    campaign = workflow.parent.parent / "configs/generation/campaigns/steady_flow/family_generalization.yaml"
    remote = ["--cpu-host", "cpu.example", "--remote-root", "/remote/generation", "--git-commit", _COMMIT]
    resources = [
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
    launch = _run(workflow, ["launch", str(campaign), *remote, *resources], environment)
    assert launch.returncode == 0, launch.stderr
    assert f"Campaign run ID: {_RUN_ID}" in launch.stdout
    assert f"Exact Git commit: {_COMMIT}" in launch.stdout
    assert "Launch returned without waiting" in launch.stdout
    log_text = log.read_text(encoding="utf-8")
    assert "ssh -o BatchMode=yes" in log_text
    assert "submit-campaign" in log_text
    assert "--max-nodes" in log_text
    assert "--max-parallel-cases" in log_text
    assert _BATCH_NAME in log_text
    assert _COMMIT in log_text

    status = _run(
        workflow,
        ["status", _RUN_ID, "--cpu-host", "cpu.example", "--remote-root", "/remote/generation"],
        environment,
    )
    assert status.returncode == 0, status.stderr
    assert '"campaign_state":"active"' in status.stdout

    before_all = log.read_text(encoding="utf-8")
    all_result = _run(workflow, ["all", str(campaign), *remote, *resources], environment)
    assert all_result.returncode == 0, all_result.stderr
    assert "Resume with:" in all_result.stdout
    appended = log.read_text(encoding="utf-8")[len(before_all) :]
    assert "submit-campaign" in appended
    assert "campaign-transfer-plan" not in appended
    assert not any(line.startswith("rsync ") for line in appended.splitlines())


def test_collect_uses_safe_rsync_and_delegates_dataset_build(tmp_path: Path) -> None:
    """Protect terminal-plan collection, nonoverwriting transfer, and Docker delegation."""
    workflow, log, environment, storage, mirror = _harness(tmp_path)
    campaign_directory = f"01_generation/meta/campaigns/{_RUN_ID}"
    batch_id = f"{_BATCH_NAME}__fedcba9876543210"
    meta_directory = f"01_generation/meta/batches/{batch_id}"
    raw_directory = f"01_generation/raw/{batch_id}"
    processed_directory = f"01_generation/processed/{batch_id}"
    for relative in (campaign_directory, meta_directory, raw_directory, processed_directory):
        (mirror / relative).mkdir(parents=True)
    campaign_relative_path = "configs/generation/campaigns/steady_flow/family_generalization.yaml"
    (mirror / campaign_directory / "campaign_terminal.json").write_text(
        json.dumps({"campaign_config": campaign_relative_path}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (mirror / meta_directory / "batch_manifest.json").write_text("{}\n", encoding="utf-8")
    (mirror / raw_directory / "case_0001.txt").write_text("raw\n", encoding="utf-8")
    (mirror / processed_directory / "case_0001.txt").write_text("processed\n", encoding="utf-8")
    environment["FAKE_TRANSFER_PLAN"] = (
        f"campaign\tsteady_flow_family_generalization\t{_COMMIT}\t{campaign_directory}"
        f"\t{campaign_relative_path}\n"
        f"batch\t{_BATCH_NAME}\t{batch_id}\t1\t{meta_directory}\t{raw_directory}\t{processed_directory}\n"
    )
    collect = _run(
        workflow,
        [
            "collect",
            _RUN_ID,
            "--cpu-host",
            "cpu.example",
            "--remote-root",
            "/remote/generation",
            "--build-datasets",
        ],
        environment,
    )
    assert collect.returncode == 0, collect.stderr
    assert "Transferred cases: 1" in collect.stdout
    assert "Reused cases: 0" in collect.stdout
    assert "Rejected cases: 0" in collect.stdout
    assert (storage / campaign_directory / "campaign_terminal.json").is_file()
    assert (storage / meta_directory / "batch_manifest.json").is_file()
    assert (storage / raw_directory / "case_0001.txt").is_file()
    assert (storage / processed_directory / "case_0001.txt").is_file()
    log_text = log.read_text(encoding="utf-8")
    rsync_lines = [line for line in log_text.splitlines() if line.startswith("rsync ")]
    assert len(rsync_lines) == 4
    assert all("--relative" in line and "--exclude=.state/" in line and "--exclude=work/" in line for line in rsync_lines)
    assert all("--delete" not in line for line in rsync_lines)
    assert f"docker-job build-datasets {workflow.parent.parent}/{campaign_relative_path}" in log_text

    rebuilt = _run(workflow, ["build-datasets", _RUN_ID], environment)
    assert rebuilt.returncode == 0, rebuilt.stderr
    assert "Building every declared package" in rebuilt.stdout
