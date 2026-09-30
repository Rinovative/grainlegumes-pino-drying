# ruff: noqa: D103, S101, S603, S607, PLR2004
"""Focused native Generation controller contracts without Slurm submission."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


_RUN_ID = "steady_flow_dataset_v1__0123456789abcdef"
_CAMPAIGN = "configs/generation/campaigns/steady_flow/id_dataset.yaml"
_CAMPAIGNS = (
    _CAMPAIGN,
    "configs/generation/campaigns/steady_flow/technical_smoke.yaml",
    "configs/generation/campaigns/transient_drying/family_generalization.yaml",
    "configs/generation/campaigns/transient_drying/technical_smoke.yaml",
)


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def native_controller(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, str]]:
    """Build one small shared-layout checkout and replace only external commands."""
    source = Path(__file__).resolve().parents[2]
    project = tmp_path / "project with spaces"
    repository = project / "repo"
    storage = project / "storage"
    runtime = project / "runtime"
    scripts = repository / "scripts"
    scripts.mkdir(parents=True)
    storage.mkdir()
    (runtime / "venvs/native/bin").mkdir(parents=True)
    for name in (
        "generation_workflow.sh",
        "source_fingerprint.py",
        "generation_smoke_node.sh",
        "generation_campaign_node.sh",
        "generation_benchmark_node.sh",
        "generation_python_node.sh",
        "generation_prerequisites.sh",
    ):
        shutil.copy2(source / "scripts" / name, scripts / name)
    for relative in _CAMPAIGNS:
        campaign = repository / relative
        campaign.parent.mkdir(parents=True, exist_ok=True)
        campaign.write_text("test-owned campaign\n", encoding="utf-8")

    fake_cli = tmp_path / "fake_generation_cli.py"
    fake_cli.write_text(
        """import json
import os
import pathlib
import sys

args = sys.argv[1:]
with pathlib.Path(os.environ["FAKE_CLI_LOG"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(args) + "\\n")
if os.environ.get("FAKE_MUTATE_SOURCE"):
    pathlib.Path(os.environ["FAKE_MUTATE_SOURCE"]).write_text("changed during worker\\n", encoding="utf-8")
if os.environ.get("FAKE_CLI_FAIL"):
    raise SystemExit(int(os.environ["FAKE_CLI_FAIL"]))
operation = args[0]
campaigns = (
    "configs/generation/campaigns/steady_flow/id_dataset.yaml",
    "configs/generation/campaigns/steady_flow/technical_smoke.yaml",
    "configs/generation/campaigns/transient_drying/family_generalization.yaml",
    "configs/generation/campaigns/transient_drying/technical_smoke.yaml",
)
if operation == "list-campaigns":
    print(json.dumps({
        "workflow": {
            "technical_runtime_smoke": {
                "stationary": {"repository_path": campaigns[1]},
                "transient": {"repository_path": campaigns[3]},
            },
            "primary": {
                "stationary": {"repository_path": campaigns[0]},
                "transient": {"repository_path": campaigns[2]},
            },
        },
        "shared_execution_site": {
            "cpu_host": "shared-filesystem", "scheduler": "slurm",
            "partition": "standard", "cores_per_node": 32,
            "python_module": "Python/3.12", "comsol_module": "Comsol/v6.4",
            "python_executable": "python3", "comsol_executable": "comsol",
        },
    }))
elif operation == "resolve-generation-run":
    print(json.dumps({
        "run_kind": "campaign", "identity": os.environ["FAKE_RUN_ID"],
        "config_path": campaigns[0], "units": [{"metadata": {
            "campaign_purpose": "family_generalization",
            "simulation_profile": "steady_flow",
        }}], "children": [],
    }))
elif operation == "find-completion-parent":
    print(json.dumps({"status": "fresh"}))
elif operation == "validate-config":
    print(json.dumps({
        "campaign_purpose": "family_generalization",
        "counts": {"one": 1},
        "execution_resources": {
            "cluster": {"cores_per_case": 8, "wall_time": "01:05:00", "cores_per_node": 32},
            "submission": {"max_admission_cases": 4, "poll_interval_seconds": 15,
                           "max_running_cases": None},
            "site": {"cpu_host": "shared-filesystem", "scheduler": "slurm",
                     "partition": "standard", "python_module": "Python/3.12",
                     "comsol_module": "Comsol/v6.4", "python_executable": "python3",
                     "comsol_executable": "comsol"},
        },
    }))
elif operation == "find-compatible-campaign-source":
    print(json.dumps({"status": "missing", "campaign_run_id": None}))
elif operation in ("validate-all-workflow", "submit-campaign"):
    raise SystemExit(6)
else:
    print(f"native-cli:{operation}")
""",
        encoding="utf-8",
    )
    python_wrapper = runtime / "venvs/native/bin/python"
    _write_executable(
        python_wrapper,
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'if [[ "${1:-}" == -m && "${2:-}" == src.generation.cli.cli_generation ]]; then\n'
        "  shift 2\n"
        '  exec "${FAKE_REAL_PYTHON}" "${FAKE_CLI}" "$@"\n'
        "fi\n"
        'exec "${FAKE_REAL_PYTHON}" "$@"\n',
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "sbatch",
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "pathlib.Path(os.environ['FAKE_SBATCH_LOG']).write_text(\n"
        "    json.dumps({'args': sys.argv[1:], 'env': {\n"
        "        key: os.environ.get(key) for key in (\n"
        "            'GENERATION_GIT_COMMIT', 'GENERATION_SOURCE_SHA256',\n"
        "            'GENERATION_NATIVE_VENV')\n"
        "    }}), encoding='utf-8')\n"
        "print('4242')\n",
    )
    _write_executable(
        fake_bin / "srun",
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "pathlib.Path(os.environ['FAKE_SRUN_LOG']).write_text(\n"
        "    json.dumps({'args': sys.argv[1:], 'env': {\n"
        "        key: os.environ.get(key) for key in (\n"
        "            'GENERATION_GIT_COMMIT', 'GENERATION_SOURCE_SHA256',\n"
        "            'GENERATION_NATIVE_VENV', 'STORAGE_ROOT')\n"
        "    }}), encoding='utf-8')\n"
        "if os.environ.get('FAKE_SRUN_FAIL'):\n"
        "    raise SystemExit(int(os.environ['FAKE_SRUN_FAIL']))\n"
        "if 'validate-all-workflow' in sys.argv:\n"
        "    raise SystemExit(6)\n"
        "print('canonical-inputs\\t1\\t0')\n",
    )
    _write_executable(fake_bin / "module", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(
        fake_bin / "comsol",
        '#!/usr/bin/env bash\n[[ "${1:-}" == -configuration && "${3:-}" == -version ]] || exit 2\nprintf \'COMSOL Multiphysics 6.4.0.293\\n\'\n',
    )
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "Test")
    _git(repository, "config", "user.email", "test@example.invalid")
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", "test source")
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{environment['PATH']}",
            "STORAGE_ROOT": str(storage),
            "RUNTIME_ROOT": str(runtime),
            "FAKE_REAL_PYTHON": sys.executable,
            "FAKE_CLI": str(fake_cli),
            "FAKE_CLI_LOG": str(tmp_path / "cli.jsonl"),
            "FAKE_SBATCH_LOG": str(tmp_path / "sbatch.json"),
            "FAKE_SRUN_LOG": str(tmp_path / "srun.json"),
            "FAKE_RUN_ID": _RUN_ID,
        }
    )
    return repository, storage, runtime, environment


def _run(controller: tuple[Path, Path, Path, dict[str, str]], *arguments: str) -> subprocess.CompletedProcess[str]:
    repository, _storage, _runtime, environment = controller
    return subprocess.run(
        ["bash", str(repository / "scripts/generation_workflow.sh"), *arguments],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def _cli_calls(controller: tuple[Path, Path, Path, dict[str, str]]) -> list[list[str]]:
    log = Path(controller[3]["FAKE_CLI_LOG"])
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


def test_smoke_submits_native_worker_with_runtime_logs_and_source_evidence(
    native_controller: tuple[Path, Path, Path, dict[str, str]],
) -> None:
    repository, _storage, runtime, environment = native_controller
    result = _run(native_controller, "smoke")

    assert result.returncode == 0, result.stderr
    assert "job=4242 partition=gpu" in result.stdout
    submission = json.loads(Path(environment["FAKE_SBATCH_LOG"]).read_text(encoding="utf-8"))
    arguments = submission["args"]
    assert "--partition=gpu" in arguments
    assert f"--chdir={repository}" in arguments
    assert f"--output={runtime}/logs/generation/slurm-%j.out" in arguments
    assert f"--error={runtime}/logs/generation/slurm-%j.err" in arguments
    assert arguments[-2:] == [str(repository / "scripts/generation_smoke_node.sh"), str(repository)]
    assert not any(argument.startswith(("--gres", "--nodelist", "--exclude")) for argument in arguments)
    assert submission["env"]["GENERATION_GIT_COMMIT"] == _git(repository, "rev-parse", "HEAD")
    assert len(submission["env"]["GENERATION_SOURCE_SHA256"]) == 64
    assert submission["env"]["GENERATION_NATIVE_VENV"] == str(runtime / "venvs/native")


def test_smoke_accepts_dirty_fingerprinted_source_and_explicit_standard(
    native_controller: tuple[Path, Path, Path, dict[str, str]],
) -> None:
    repository, _storage, _runtime, environment = native_controller
    (repository / _CAMPAIGN).write_text("dirty source\n", encoding="utf-8")

    result = _run(native_controller, "smoke", "--partition", "standard")

    assert result.returncode == 0, result.stderr
    submission = json.loads(Path(environment["FAKE_SBATCH_LOG"]).read_text(encoding="utf-8"))
    assert "--partition=standard" in submission["args"]
    assert len(submission["env"]["GENERATION_SOURCE_SHA256"]) == 64


@pytest.mark.parametrize(
    "arguments",
    [
        ("smoke", "--partition", "rsm"),
        ("smoke", "--partition", "gpu", "--partition", "standard"),
        ("smoke", "--dry-run"),
        ("smoke", "unexpected"),
        ("run", _CAMPAIGN, "--partition", "gpu"),
    ],
)
def test_invalid_launch_arguments_fail_before_submission(
    native_controller: tuple[Path, Path, Path, dict[str, str]],
    arguments: tuple[str, ...],
) -> None:
    result = _run(native_controller, *arguments)

    assert result.returncode == 2
    assert not Path(native_controller[3]["FAKE_SBATCH_LOG"]).exists()


def test_run_publication_requires_clean_current_source(
    native_controller: tuple[Path, Path, Path, dict[str, str]],
) -> None:
    repository, _storage, _runtime, environment = native_controller
    (repository / _CAMPAIGN).write_text("dirty source\n", encoding="utf-8")

    result = _run(native_controller, "run", _CAMPAIGN)

    assert result.returncode == 2
    assert "clean committed shared source" in result.stderr
    assert not Path(environment["FAKE_CLI_LOG"]).exists()
    assert not Path(environment["FAKE_SBATCH_LOG"]).exists()


@pytest.mark.parametrize(
    ("operation", "embedded_option"),
    [("run", "--dry-run"), ("run", "--preflight-only"), ("inputs", "--dry-run")],
)
def test_quoted_config_path_cannot_bypass_clean_source_admission(
    native_controller: tuple[Path, Path, Path, dict[str, str]],
    operation: str,
    embedded_option: str,
) -> None:
    """Treat option-like text inside a filename as data during source admission."""
    repository, _storage, _runtime, environment = native_controller
    quoted_path = repository / f"configs/generation/campaigns/steady_flow/source {embedded_option} config.yaml"
    quoted_path.write_text("uncommitted campaign\n", encoding="utf-8")
    selection = ("--all-batches", "--all-cases") if operation == "inputs" else ()

    result = _run(native_controller, operation, str(quoted_path.relative_to(repository)), *selection)

    assert result.returncode == 2
    assert "clean committed shared source" in result.stderr
    assert not Path(environment["FAKE_CLI_LOG"]).exists()
    assert not Path(environment["FAKE_SBATCH_LOG"]).exists()
    assert not Path(environment["FAKE_SRUN_LOG"]).exists()


def test_input_dry_run_keeps_dirty_source_read_only(
    native_controller: tuple[Path, Path, Path, dict[str, str]],
) -> None:
    """Admit an explicit input dry run without allowing scientific publication."""
    repository, _storage, _runtime, environment = native_controller
    (repository / _CAMPAIGN).write_text("dirty source\n", encoding="utf-8")

    result = _run(native_controller, "inputs", _CAMPAIGN, "--all-batches", "--all-cases", "--dry-run")

    assert result.returncode == 0, result.stderr
    assert Path(environment["FAKE_SRUN_LOG"]).exists()
    assert not Path(environment["FAKE_SBATCH_LOG"]).exists()


def test_dry_run_resolves_plan_from_current_shared_checkout(
    native_controller: tuple[Path, Path, Path, dict[str, str]],
) -> None:
    repository, _storage, _runtime, environment = native_controller
    result = _run(native_controller, "run", _CAMPAIGN, "--dry-run")

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["identity"] == _RUN_ID
    assert [call[0] for call in _cli_calls(native_controller)] == ["resolve-generation-run"]
    assert _cli_calls(native_controller)[0][1] == str(repository / _CAMPAIGN)
    assert "--allow-incomplete" in _cli_calls(native_controller)[0]
    assert not Path(environment["FAKE_SBATCH_LOG"]).exists()


def test_status_and_cancel_use_native_cli_and_sibling_storage(
    native_controller: tuple[Path, Path, Path, dict[str, str]],
) -> None:
    _repository, storage, _runtime, environment = native_controller
    status = _run(native_controller, "status", _RUN_ID)
    cancel = _run(native_controller, "cancel", _RUN_ID, "--force")

    assert status.returncode == 0, status.stderr
    assert cancel.returncode == 0, cancel.stderr
    calls = _cli_calls(native_controller)
    assert any(call[:2] == ["campaign-status", _RUN_ID] for call in calls)
    assert any(call[:2] == ["cancel-campaign", _RUN_ID] and "--force" in call for call in calls)
    assert all(call[call.index("--storage-root") + 1] == str(storage) for call in calls if "--storage-root" in call)
    assert not Path(environment["FAKE_SBATCH_LOG"]).exists()


def test_requested_historical_commit_is_rejected(
    native_controller: tuple[Path, Path, Path, dict[str, str]],
) -> None:
    _repository, _storage, _runtime, environment = native_controller
    result = _run(native_controller, "smoke", "--git-commit", "a" * 40)

    assert result.returncode == 2
    assert "current shared repository HEAD" in result.stderr
    assert not Path(environment["FAKE_SBATCH_LOG"]).exists()


def test_campaign_run_materializes_inputs_through_native_srun(
    native_controller: tuple[Path, Path, Path, dict[str, str]],
) -> None:
    repository, storage, runtime, environment = native_controller

    result = _run(native_controller, "run", _CAMPAIGN)

    assert result.returncode == 1
    assert "Shared campaign submission failed" in result.stderr
    submission = json.loads(Path(environment["FAKE_SRUN_LOG"]).read_text(encoding="utf-8"))
    arguments = submission["args"]
    assert "--partition=standard" in arguments
    assert "--cpus-per-task=4" in arguments
    assert "--mem=16G" in arguments
    assert "--time=01:05:00" in arguments
    assert f"--chdir={repository}" in arguments
    assert any(argument.startswith(f"--error={runtime}/logs/generation/") for argument in arguments)
    assert arguments[-9:] == [
        str(repository / "scripts/generation_python_node.sh"),
        str(repository),
        "cli",
        "prepare-campaign-inputs",
        str(repository / _CAMPAIGN),
        "--git-commit",
        _git(repository, "rev-parse", "HEAD"),
        "--storage-root",
        str(storage),
    ]
    assert "--storage-root" in arguments
    assert arguments[arguments.index("--storage-root") + 1] == str(storage)
    assert not any(argument.startswith(("--gres", "--nodelist")) for argument in arguments)
    assert submission["env"]["GENERATION_GIT_COMMIT"] == _git(repository, "rev-parse", "HEAD")
    assert len(submission["env"]["GENERATION_SOURCE_SHA256"]) == 64
    assert "prepare-campaign-inputs" not in [call[0] for call in _cli_calls(native_controller)]
    assert "submit-campaign" in [call[0] for call in _cli_calls(native_controller)]


def test_input_only_generation_uses_standard_cpu_allocation_and_preserves_selection(
    native_controller: tuple[Path, Path, Path, dict[str, str]],
) -> None:
    repository, storage, runtime, environment = native_controller
    result = _run(
        native_controller,
        "inputs",
        _CAMPAIGN,
        "--only-batch",
        "lentil",
        "--case-start",
        "3",
        "--case-count",
        "2",
    )

    assert result.returncode == 0, result.stderr
    submission = json.loads(Path(environment["FAKE_SRUN_LOG"]).read_text(encoding="utf-8"))
    arguments = submission["args"]
    assert "--partition=standard" in arguments
    assert "--cpus-per-task=4" in arguments
    assert "--mem=16G" in arguments
    assert not any(argument.startswith(("--gres", "--nodelist")) for argument in arguments)
    assert any(argument.startswith(f"--error={runtime}/logs/generation/") for argument in arguments)
    assert arguments[-15:] == [
        str(repository / "scripts/generation_python_node.sh"),
        str(repository),
        "cli",
        "generate-input-cases",
        str(repository / _CAMPAIGN),
        "--only-batch",
        "lentil",
        "--case-start",
        "3",
        "--case-count",
        "2",
        "--git-commit",
        _git(repository, "rev-parse", "HEAD"),
        "--storage-root",
        str(storage),
    ]


def test_input_only_generation_rejects_dirty_source_and_overridden_ownership(
    native_controller: tuple[Path, Path, Path, dict[str, str]],
) -> None:
    repository, _storage, _runtime, environment = native_controller
    override = _run(native_controller, "inputs", _CAMPAIGN, "--storage-root", "/tmp/other")  # noqa: S108 -- rejected test path
    assert override.returncode == 2
    assert not Path(environment["FAKE_SRUN_LOG"]).exists()

    (repository / _CAMPAIGN).write_text("dirty source\n", encoding="utf-8")
    dirty = _run(native_controller, "inputs", _CAMPAIGN, "--all-batches", "--all-cases")
    assert dirty.returncode == 2
    assert "clean committed shared source" in dirty.stderr
    assert not Path(environment["FAKE_SRUN_LOG"]).exists()


def test_srun_failure_stops_campaign_before_submission(
    native_controller: tuple[Path, Path, Path, dict[str, str]],
) -> None:
    environment = native_controller[3]
    environment["FAKE_SRUN_FAIL"] = "7"

    result = _run(native_controller, "run", _CAMPAIGN)

    assert result.returncode == 1
    assert "Canonical campaign input preparation failed" in result.stderr
    assert Path(environment["FAKE_SRUN_LOG"]).exists()
    assert "submit-campaign" not in [call[0] for call in _cli_calls(native_controller)]


@pytest.mark.parametrize(
    ("cli_failure", "mutate_source", "expected_status"),
    [("", False, 0), ("9", False, 9), ("", True, 1)],
)
def test_python_worker_preserves_cli_status_and_rechecks_source(
    native_controller: tuple[Path, Path, Path, dict[str, str]],
    cli_failure: str,
    mutate_source: bool,
    expected_status: int,
) -> None:
    repository, storage, runtime, original_environment = native_controller
    environment = original_environment.copy()
    environment.update(
        {
            "SLURM_JOB_ID": "4242",
            "SLURM_CPUS_PER_TASK": "4",
            "GENERATION_GIT_COMMIT": _git(repository, "rev-parse", "HEAD"),
            "GENERATION_NATIVE_VENV": str(runtime / "venvs/native"),
            "GENERATION_SOURCE_SHA256": subprocess.run(
                [sys.executable, str(repository / "scripts/source_fingerprint.py"), str(repository)],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
        }
    )
    if cli_failure:
        environment["FAKE_CLI_FAIL"] = cli_failure
    if mutate_source:
        environment["FAKE_MUTATE_SOURCE"] = str(repository / _CAMPAIGN)

    result = subprocess.run(
        [
            "bash",
            str(repository / "scripts/generation_python_node.sh"),
            str(repository),
            "cli",
            "storage-status",
            "--storage-root",
            str(storage),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == expected_status, result.stderr
    if mutate_source:
        assert "unchanged source worktree since submission" in result.stderr
    elif not cli_failure:
        assert result.stdout == "native-cli:storage-status\n"


@pytest.mark.parametrize(
    "operation",
    ["materialize-core-benchmark-inputs", "resume-core-benchmark"],
)
def test_benchmark_worker_uses_node_scratch_and_preserves_identity_arguments(
    native_controller: tuple[Path, Path, Path, dict[str, str]],
    tmp_path: Path,
    operation: str,
) -> None:
    repository, storage, runtime, original_environment = native_controller
    scratch_parent = tmp_path / "node scratch"
    scratch_parent.mkdir()
    environment = original_environment.copy()
    environment.update(
        {
            "SLURM_JOB_ID": "4242",
            "SLURM_CPUS_PER_TASK": "4",
            "GENERATION_GIT_COMMIT": _git(repository, "rev-parse", "HEAD"),
            "GENERATION_NATIVE_VENV": str(runtime / "venvs/native"),
            "GENERATION_SOURCE_SHA256": subprocess.run(
                [sys.executable, str(repository / "scripts/source_fingerprint.py"), str(repository)],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "TMPDIR": str(scratch_parent),
        }
    )
    suite = repository / "configs/generation/benchmarks/suite.yaml"
    target = str(suite) if operation == "materialize-core-benchmark-inputs" else "core_scaling_transient__0123456789abcdef"
    identity_arguments = ["--git-commit", environment["GENERATION_GIT_COMMIT"]] if operation == "materialize-core-benchmark-inputs" else []
    result = subprocess.run(
        [
            "bash",
            str(repository / "scripts/generation_python_node.sh"),
            str(repository),
            "benchmark",
            operation,
            target,
            *identity_arguments,
            "--storage-root",
            str(storage),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    call = _cli_calls(native_controller)[-1]
    assert call[: 2 + len(identity_arguments) + 2] == [
        operation,
        target,
        *identity_arguments,
        "--storage-root",
        str(storage),
    ]
    assert Path(call[call.index("--scratch-root") + 1]).is_relative_to(scratch_parent)
    if operation == "materialize-core-benchmark-inputs":
        assert call[call.index("--comsol-version-output") + 1] == "COMSOL Multiphysics 6.4.0.293"
        comsol_executable = Path(call[call.index("--comsol-executable-path") + 1])
        assert comsol_executable.is_absolute()
        assert comsol_executable.name == "comsol"
        assert comsol_executable.is_file()
    else:
        assert "--comsol-version-output" not in call
        assert "--comsol-executable-path" not in call
    assert not list(scratch_parent.iterdir())
