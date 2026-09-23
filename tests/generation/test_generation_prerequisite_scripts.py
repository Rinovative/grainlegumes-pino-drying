# ruff: noqa: S101, S603, S607, PLR2004
"""Native Generation worker source, environment, and scratch contracts."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

_CAMPAIGN_RUN_ID = "synthetic__0123456789abcdef"
_BENCHMARK_RUN_ID = "core_scaling_transient__0123456789abcdef"
_SCRIPTS = (
    "generation_campaign_node.sh",
    "generation_benchmark_node.sh",
    "generation_smoke_node.sh",
    "generation_prerequisites.sh",
    "source_fingerprint.py",
)


def _write_executable(path: Path, contents: str) -> None:
    path.write_text("#!/bin/bash\nset -euo pipefail\n" + contents, encoding="utf-8")
    path.chmod(0o755)


@pytest.fixture
def native_worker(request: pytest.FixtureRequest) -> tuple[Path, dict[str, str], Path]:
    """Build one shared checkout with inert site commands and a disposable venv."""
    temporary = TemporaryDirectory(prefix="generation-worker-test-", dir="/tmp")
    request.addfinalizer(temporary.cleanup)  # noqa: PT021 -- clean the checkout if fixture setup fails
    tmp_path = Path(temporary.name)
    source = Path(__file__).resolve().parents[2] / "scripts"
    repository = tmp_path / "project" / "repo"
    scripts = repository / "scripts"
    scripts.mkdir(parents=True)
    storage = repository.parent / "storage"
    runtime = repository.parent / "runtime"
    storage.mkdir()
    runtime.mkdir()
    for name in _SCRIPTS:
        shutil.copy2(source / name, scripts / name)
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(repository), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repository), "commit", "-qm", "fixture"], check=True)
    commit = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    source_sha = subprocess.check_output(["python3", str(scripts / "source_fingerprint.py"), str(repository)], text=True).strip()
    binary = tmp_path / "bin"
    binary.mkdir()
    command_log = tmp_path / "commands.log"
    _write_executable(binary / "module", 'printf "module <%s>\\n" "$*" >> "$COMMAND_LOG"\n')
    _write_executable(binary / "comsol", 'printf "COMSOL Multiphysics 6.4.0.293\\n"\n')
    venv = runtime / "venvs" / "generation"
    (venv / "bin").mkdir(parents=True)
    _write_executable(
        venv / "bin" / "python",
        'printf "python <%s>\\n" "$*" >> "$COMMAND_LOG"\nif [[ "${3:-}" == cleanup-worker-workspace ]]; then rmdir -- "$4"; fi\n',
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    environment = {
        **{key: value for key, value in os.environ.items() if not key.startswith("BASH_FUNC_module")},
        "PATH": f"{binary}:{os.environ['PATH']}",
        "COMMAND_LOG": str(command_log),
        "SLURM_JOB_ID": "12345",
        "SLURM_CPUS_PER_TASK": "2",
        "TMPDIR": str(scratch),
        "GENERATION_GIT_COMMIT": commit,
        "GENERATION_SOURCE_SHA256": source_sha,
        "GENERATION_NATIVE_VENV": str(venv),
        "GENERATION_PYTHON_MODULE": "Python/3.12",
        "GENERATION_COMSOL_MODULE": "Comsol/v6.4",
        "GENERATION_PYTHON_EXECUTABLE": "python3",
        "GENERATION_COMSOL_EXECUTABLE": "comsol",
        "GENERATION_ATTEMPT_INDEX": "1",
        "GENERATION_CAMPAIGN_RUN_ID": _CAMPAIGN_RUN_ID,
        "GENERATION_BENCHMARK_RUN_ID": _BENCHMARK_RUN_ID,
        "STORAGE_ROOT": str(storage),
    }
    return repository, environment, command_log


@pytest.mark.parametrize(
    ("script", "arguments", "expected_cli"),
    [
        ("generation_campaign_node.sh", (_CAMPAIGN_RUN_ID, "batch", "1", "2"), "run-campaign-case"),
        ("generation_benchmark_node.sh", (_BENCHMARK_RUN_ID, "cores_02", "nominal"), "run-core-benchmark-case"),
    ],
)
def test_spooled_worker_uses_shared_source_native_modules_and_cleans_scratch(
    native_worker: tuple[Path, dict[str, str], Path],
    tmp_path: Path,
    script: str,
    arguments: tuple[str, ...],
    expected_cli: str,
) -> None:
    """Run the worker from Slurm's spool path with only shared source beside it."""
    repository, environment, command_log = native_worker
    spooled = tmp_path / "slurm_script"
    shutil.copy2(repository / "scripts" / script, spooled)
    result = subprocess.run(
        ["/bin/bash", str(spooled), str(repository), *arguments],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "check=source-worktree status=pass" in result.stdout
    commands = command_log.read_text(encoding="utf-8")
    assert "module <load Python/3.12>" in commands
    assert "module <load Comsol/v6.4>" in commands
    assert expected_cli in commands
    assert "cleanup-worker-workspace" in commands
    assert not tuple(Path(environment["TMPDIR"]).iterdir())


def test_worker_rejects_source_change_before_python_or_comsol(
    native_worker: tuple[Path, dict[str, str], Path],
) -> None:
    """Fail closed when shared source changes after the launch fingerprint."""
    repository, environment, command_log = native_worker
    helper = repository / "scripts" / "source_fingerprint.py"
    helper.write_text(helper.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")
    result = subprocess.run(
        ["/bin/bash", str(repository / "scripts" / "generation_campaign_node.sh"), str(repository), _CAMPAIGN_RUN_ID, "batch", "1", "2"],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "unchanged source worktree since submission" in result.stderr
    assert not command_log.exists()


def test_comsol_version_rejects_zero_exit_with_invalid_launcher_output(
    native_worker: tuple[Path, dict[str, str], Path],
) -> None:
    """Do not report a ready solver when its launcher prints an error and exits zero."""
    repository, environment, _command_log = native_worker
    fake_comsol = Path(environment["PATH"].split(os.pathsep)[0]) / "comsol"
    _write_executable(fake_comsol, 'printf "Invalid Configuration Location\\n"\n')
    result = subprocess.run(
        [
            "/bin/bash",
            "-c",
            'source "$1"; generation_comsol_version comsol',
            "bash",
            str(repository / "scripts" / "generation_prerequisites.sh"),
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "invalid v6.4 evidence" in result.stderr
    assert not tuple(Path(environment["TMPDIR"]).iterdir())


@pytest.mark.parametrize("batch_exit", [0, 37])
def test_native_smoke_runs_compiled_java_and_propagates_batch_failure(
    native_worker: tuple[Path, dict[str, str], Path],
    batch_exit: int,
) -> None:
    """Exercise the disposable COMSOL batch contract and scratch cleanup."""
    repository, environment, command_log = native_worker
    fake_comsol = Path(environment["PATH"].split(os.pathsep)[0]) / "comsol"
    _write_executable(
        fake_comsol,
        'printf "comsol <%s>\\n" "$*" >> "$COMMAND_LOG"\n'
        'if [[ "$1" == -configuration ]]; then printf "COMSOL Multiphysics 6.4.0.293\\n"; exit 0; fi\n'
        'if [[ "$1" == compile ]]; then\n'
        '  grep -Fq "public static void main(String[] args) throws java.io.IOException" GenerationNativeSmoke.java\n'
        '  grep -Fq "model.save(\\"smoke.mph\\");" GenerationNativeSmoke.java\n'
        '  printf "compiled" > GenerationNativeSmoke.class\n'
        "  exit 0\n"
        "fi\n"
        'printf "batch log\\n" > smoke.log\n'
        'if (( SMOKE_BATCH_EXIT != 0 )); then exit "$SMOKE_BATCH_EXIT"; fi\n'
        'printf "disposable model" > smoke.mph\n',
    )
    environment["SLURM_CPUS_PER_TASK"] = "1"
    environment["SMOKE_BATCH_EXIT"] = str(batch_exit)
    result = subprocess.run(
        ["/bin/bash", str(repository / "scripts" / "generation_smoke_node.sh"), str(repository)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == batch_exit, result.stderr
    assert "check=source-worktree status=pass" in result.stdout
    assert "comsol <compile" in command_log.read_text(encoding="utf-8")
    assert "comsol <batch" in command_log.read_text(encoding="utf-8")
    assert ("GENERATION NATIVE SMOKE PASS" in result.stdout) == (batch_exit == 0)
    assert not tuple(Path(environment["TMPDIR"]).iterdir())


def test_worker_rejects_non_sibling_environment(
    native_worker: tuple[Path, dict[str, str], Path],
    tmp_path: Path,
) -> None:
    """Keep durable storage and the canonical venv in their sibling roots."""
    repository, environment, command_log = native_worker
    environment["STORAGE_ROOT"] = str(tmp_path / "other-storage")
    result = subprocess.run(
        ["/bin/bash", str(repository / "scripts" / "generation_campaign_node.sh"), str(repository), _CAMPAIGN_RUN_ID, "batch", "1", "2"],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "sibling storage" in result.stderr
    assert not command_log.exists()
