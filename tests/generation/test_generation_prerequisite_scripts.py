# ruff: noqa: S101, S603
"""Execution-domain routing for Generation shell prerequisites."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

_COMMIT = "a" * 40
_REPOSITORY_URL = "https://github.com/Rinovative/grainlegumes-pino-drying.git"
_CAMPAIGN_RUN_ID = "synthetic__0123456789abcdef"
_BENCHMARK_RUN_ID = "core_scaling_transient__0123456789abcdef"
_WORKER_SCRIPTS = (
    "generation_cpu_smoke.sh",
    "generation_campaign_node.sh",
    "generation_benchmark_node.sh",
)


def _executable(path: Path, body: str) -> Path:
    """Create one test-owned executable."""
    path.write_text(f"#!/bin/bash\nset -euo pipefail\n{body}\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _link_command(binary: Path, name: str) -> None:
    """Expose one host utility inside the otherwise isolated fake PATH."""
    source = shutil.which(name)
    if source is None:
        message = f"Test requires host utility {name!r}."
        raise RuntimeError(message)
    (binary / name).symlink_to(source)


def _fake_environment(tmp_path: Path, *, include_rsync: bool) -> tuple[dict[str, str], Path, Path]:
    """Build isolated login/compute commands without exposing host rsync."""
    binary = tmp_path / "bin"
    binary.mkdir(parents=True)
    log = tmp_path / "commands.log"
    for name in ("dirname", "hostname", "mktemp", "rmdir", "stat"):
        _link_command(binary, name)
    _executable(binary / "module", 'printf \'module <%s>\\n\' "$*" >> "${FAKE_COMMAND_LOG}"')
    _executable(
        binary / "git",
        """printf 'git <%s>\\n' "$*" >> "${FAKE_COMMAND_LOG}"
case " $* " in
  *" status --porcelain "*) ;;
  *" rev-parse HEAD "*) printf '%s\\n' "${FAKE_GIT_COMMIT}" ;;
  *" remote get-url origin "*) printf '%s\\n' "${FAKE_REPOSITORY_URL}" ;;
esac""",
    )
    _executable(binary / "python3", "printf 'Python 3.10.13\\n'")
    _executable(binary / "comsol", "printf 'COMSOL Multiphysics 6.4.0.293\\n'")
    _executable(binary / "sbatch", "printf 'slurm 22.05.9\\n'")
    for name in ("squeue", "sacct", "scancel"):
        _executable(binary / name, ":")
    if include_rsync:
        _executable(binary / "rsync", "printf 'rsync version 3.2.7\\n'")

    venv = tmp_path / "venv"
    base_python = tmp_path / "software/Python/3.10/bin/python3.10"
    (venv / "bin").mkdir(parents=True)
    base_python.parent.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text(
        f"home = {base_python.parent}\nversion = 3.10.13\n",
        encoding="utf-8",
    )
    _executable(
        base_python,
        """printf 'venv-python <%s>\n' "$*" >> "${FAKE_COMMAND_LOG}"
if [[ "${FAKE_VENV_IMPORT_FAIL:-false}" == true && " $* " == *" -c "* ]]; then
  printf 'synthetic package import failure\n' >&2
  exit 17
fi""",
    )
    (venv / "bin/python").symlink_to(base_python)
    environment = {
        "PATH": str(binary),
        "FAKE_COMMAND_LOG": str(log),
        "FAKE_GIT_COMMIT": _COMMIT,
        "FAKE_REPOSITORY_URL": _REPOSITORY_URL,
    }
    return environment, venv, log


def _fake_checkout(tmp_path: Path) -> Path:
    """Create one detached exact checkout containing only worker-owned scripts."""
    source_repository = Path(__file__).resolve().parents[2]
    repository = tmp_path / "fake/home/grainlegumes-generation/repo"
    scripts = repository / "scripts"
    scripts.mkdir(parents=True)
    git_directory = repository / ".git"
    git_directory.mkdir()
    (git_directory / "HEAD").write_text(f"{_COMMIT}\n", encoding="utf-8")
    for name in (*_WORKER_SCRIPTS, "generation_prerequisites.sh"):
        shutil.copy2(source_repository / "scripts" / name, scripts / name)
    return repository


def _spooled_script(repository: Path, name: str, tmp_path: Path) -> Path:
    """Copy one submitted worker to the scheduler-managed runtime location."""
    spool = tmp_path / "var/spool/slurmd/job123"
    spool.mkdir(parents=True)
    runtime_script = spool / "slurm_script"
    shutil.copy2(repository / "scripts" / name, runtime_script)
    assert not (spool / "generation_prerequisites.sh").exists()
    return runtime_script


def _compute_command(
    runtime_script: Path,
    repository: Path,
    venv: Path,
    tmp_path: Path,
    storage: Path,
    *,
    mode: str = "environment-only",
) -> list[str]:
    """Return the exact relocated fake compute preflight command."""
    return [
        "/bin/bash",
        str(runtime_script),
        str(repository),
        str(venv),
        str(tmp_path / "campaign.yaml"),
        str(storage),
        "-",
        "16",
        mode,
        "Python/3.10",
        "Comsol/v6.4",
        "python3",
        "comsol",
        "slurm",
    ]


def _campaign_command(runtime_script: Path, repository: Path) -> list[str]:
    """Return one relocated production-worker startup command."""
    return [
        "/bin/bash",
        str(runtime_script),
        str(repository),
        _CAMPAIGN_RUN_ID,
        "synthetic.batch",
        "1",
        "16",
    ]


def _benchmark_command(runtime_script: Path, repository: Path) -> list[str]:
    """Return one relocated benchmark preparation-worker startup command."""
    return [
        "/bin/bash",
        str(runtime_script),
        str(repository),
        _BENCHMARK_RUN_ID,
        "prepare",
    ]


def _login_command(repository: Path, venv: Path, storage: Path) -> list[str]:
    """Return the exact fake login preflight command."""
    return [
        "/bin/bash",
        str(repository / "scripts/generation_cpu_login_preflight.sh"),
        str(repository),
        str(storage),
        str(venv),
        _COMMIT,
        _REPOSITORY_URL,
        "Python/3.10",
        "python3",
    ]


def _assert_canonical_venv_validation(log: Path) -> None:
    """Prove one shell entry point invoked the sole semantic validator."""
    command_log = log.read_text(encoding="utf-8")
    assert "preflight.validate_generation_venv" in command_log


def test_compute_preflight_is_independent_of_slurm_spool_location(tmp_path: Path) -> None:
    """Pass beyond helper loading when Slurm relocates the submitted preflight."""
    repository = _fake_checkout(tmp_path)
    runtime_script = _spooled_script(repository, "generation_cpu_smoke.sh", tmp_path)
    environment, venv, log = _fake_environment(tmp_path, include_rsync=False)
    scratch = tmp_path / "scratch"
    storage = tmp_path / "storage"
    scratch.mkdir()
    storage.mkdir()
    environment.update(
        {
            "SLURM_JOB_ID": "123",
            "TMPDIR": str(scratch),
            "GENERATION_GIT_COMMIT": _COMMIT,
        }
    )

    result = subprocess.run(
        _compute_command(runtime_script, repository, venv, tmp_path, storage),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert "check=exact-worker-checkout status=pass" in result.stdout
    assert "Native CPU environment-only completed" in result.stdout
    assert str(runtime_script.parent / "generation_prerequisites.sh") not in result.stderr
    _assert_canonical_venv_validation(log)
    evidence = result.stdout + result.stderr + log.read_text(encoding="utf-8")
    for login_only in ("rsync", "sbatch", "squeue", "sacct", "scancel"):
        assert login_only not in evidence
    assert not tuple(scratch.iterdir())


def test_mapping_probe_uses_canonical_venv_validation(tmp_path: Path) -> None:
    """Validate the venv before the relocated mapping-probe path starts."""
    repository = _fake_checkout(tmp_path)
    runtime_script = _spooled_script(repository, "generation_cpu_smoke.sh", tmp_path)
    environment, venv, log = _fake_environment(tmp_path, include_rsync=False)
    scratch = tmp_path / "scratch"
    storage = tmp_path / "storage"
    scratch.mkdir()
    storage.mkdir()
    environment.update(
        {
            "SLURM_JOB_ID": "123",
            "TMPDIR": str(scratch),
            "GENERATION_GIT_COMMIT": _COMMIT,
        }
    )

    result = subprocess.run(
        _compute_command(
            runtime_script,
            repository,
            venv,
            tmp_path,
            storage,
            mode="mapping-probe",
        ),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert "Native CPU mapping-probe completed" in result.stdout
    _assert_canonical_venv_validation(log)


def test_login_preflight_requires_rsync_with_explicit_diagnostic(tmp_path: Path) -> None:
    """Fail the CPU login gate clearly when transfer-side rsync is absent."""
    repository = Path(__file__).resolve().parents[2]
    environment, venv, _log = _fake_environment(tmp_path, include_rsync=False)
    storage = tmp_path / "storage"
    storage.mkdir()

    result = subprocess.run(
        _login_command(repository, venv, storage),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 1
    assert "CPU login prerequisite missing: rsync (blocks transfer)." in result.stderr


def test_login_preflight_accepts_complete_control_plane(tmp_path: Path) -> None:
    """Pass the authoritative login gate when all used capabilities exist."""
    repository = Path(__file__).resolve().parents[2]
    environment, venv, log = _fake_environment(tmp_path, include_rsync=True)
    storage = tmp_path / "storage"
    storage.mkdir()

    result = subprocess.run(
        _login_command(repository, venv, storage),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert "domain=CPU login check=transfer-version:rsync status=pass" in result.stdout
    assert "domain=CPU login check=Generation-venv-runtime status=pass" in result.stdout
    _assert_canonical_venv_validation(log)


def test_compute_preflight_reports_missing_venv_launcher(tmp_path: Path) -> None:
    """Report a missing launcher before attempting canonical Python validation."""
    repository = _fake_checkout(tmp_path)
    runtime_script = _spooled_script(repository, "generation_cpu_smoke.sh", tmp_path)
    environment, venv, _log = _fake_environment(tmp_path, include_rsync=False)
    (venv / "bin/python").unlink()
    storage = tmp_path / "storage"
    storage.mkdir()
    environment.update({"SLURM_JOB_ID": "123", "GENERATION_GIT_COMMIT": _COMMIT})

    result = subprocess.run(
        _compute_command(runtime_script, repository, venv, tmp_path, storage),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 1
    assert "CPU compute-node prerequisite missing: executable Generation venv launcher" in result.stderr
    assert "No such file or directory" not in result.stderr


def test_compute_preflight_reports_package_import_failure(tmp_path: Path) -> None:
    """Surface a project/dependency import failure as a compute prerequisite."""
    repository = _fake_checkout(tmp_path)
    runtime_script = _spooled_script(repository, "generation_cpu_smoke.sh", tmp_path)
    environment, venv, _log = _fake_environment(tmp_path, include_rsync=False)
    scratch = tmp_path / "scratch"
    storage = tmp_path / "storage"
    scratch.mkdir()
    storage.mkdir()
    environment.update(
        {
            "SLURM_JOB_ID": "123",
            "TMPDIR": str(scratch),
            "FAKE_VENV_IMPORT_FAIL": "true",
            "GENERATION_GIT_COMMIT": _COMMIT,
        }
    )

    result = subprocess.run(
        _compute_command(runtime_script, repository, venv, tmp_path, storage),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 1
    assert "synthetic package import failure" in result.stderr
    assert "CPU compute-node prerequisite failed: Generation-venv-runtime" in result.stderr


def _worker_environment(
    environment: dict[str, str],
    *,
    venv: Path,
    storage: Path,
    scratch: Path,
) -> dict[str, str]:
    """Bind the shared compute values supplied by the control plane."""
    return {
        **environment,
        "SLURM_JOB_ID": "123",
        "TMPDIR": str(scratch),
        "GENERATION_GIT_COMMIT": _COMMIT,
        "GENERATION_CPU_VENV": str(venv),
        "STORAGE_ROOT": str(storage),
        "GENERATION_PYTHON_MODULE": "Python/3.10",
        "GENERATION_COMSOL_MODULE": "Comsol/v6.4",
        "GENERATION_PYTHON_EXECUTABLE": "python3",
        "GENERATION_COMSOL_EXECUTABLE": "comsol",
    }


def test_campaign_worker_is_independent_of_slurm_spool_location(tmp_path: Path) -> None:
    """Start the real campaign worker from a spool copy with no sibling helper."""
    repository = _fake_checkout(tmp_path)
    runtime_script = _spooled_script(repository, "generation_campaign_node.sh", tmp_path)
    environment, venv, log = _fake_environment(tmp_path, include_rsync=False)
    scratch = tmp_path / "scratch"
    storage = tmp_path / "storage"
    scratch.mkdir()
    storage.mkdir()
    worker_environment = _worker_environment(
        environment,
        venv=venv,
        storage=storage,
        scratch=scratch,
    )
    worker_environment.update(
        {
            "SLURM_CPUS_PER_TASK": "16",
            "GENERATION_CAMPAIGN_RUN_ID": _CAMPAIGN_RUN_ID,
        }
    )

    result = subprocess.run(
        _campaign_command(runtime_script, repository),
        check=False,
        capture_output=True,
        text=True,
        env=worker_environment,
    )

    assert result.returncode == 0, result.stderr
    assert "check=exact-worker-checkout status=pass" in result.stdout
    assert "initialize-worker-workspace" in log.read_text(encoding="utf-8")
    assert "run-campaign-case" in log.read_text(encoding="utf-8")
    _assert_canonical_venv_validation(log)
    assert str(runtime_script.parent / "generation_prerequisites.sh") not in result.stderr


def test_benchmark_worker_is_independent_of_slurm_spool_location(tmp_path: Path) -> None:
    """Start benchmark preparation from a spool copy with no sibling helper."""
    repository = _fake_checkout(tmp_path)
    runtime_script = _spooled_script(repository, "generation_benchmark_node.sh", tmp_path)
    environment, venv, log = _fake_environment(tmp_path, include_rsync=False)
    scratch = tmp_path / "scratch"
    storage = tmp_path / "storage"
    scratch.mkdir()
    storage.mkdir()
    worker_environment = _worker_environment(
        environment,
        venv=venv,
        storage=storage,
        scratch=scratch,
    )
    worker_environment.update(
        {
            "SLURM_CPUS_PER_TASK": "1",
            "GENERATION_BENCHMARK_RUN_ID": _BENCHMARK_RUN_ID,
        }
    )

    result = subprocess.run(
        _benchmark_command(runtime_script, repository),
        check=False,
        capture_output=True,
        text=True,
        env=worker_environment,
    )

    assert result.returncode == 0, result.stderr
    assert "check=exact-worker-checkout status=pass" in result.stdout
    assert "prepare-core-benchmark-case" in log.read_text(encoding="utf-8")
    _assert_canonical_venv_validation(log)
    assert str(runtime_script.parent / "generation_prerequisites.sh") not in result.stderr


def test_relocated_worker_reports_missing_repository_helper(tmp_path: Path) -> None:
    """Fail explicitly instead of letting source report a spool sibling error."""
    repository = _fake_checkout(tmp_path)
    runtime_script = _spooled_script(repository, "generation_cpu_smoke.sh", tmp_path)
    (repository / "scripts/generation_prerequisites.sh").unlink()
    environment, venv, _log = _fake_environment(tmp_path, include_rsync=False)
    storage = tmp_path / "storage"
    storage.mkdir()
    environment.update({"SLURM_JOB_ID": "123", "GENERATION_GIT_COMMIT": _COMMIT})

    result = subprocess.run(
        _compute_command(runtime_script, repository, venv, tmp_path, storage),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 1
    assert "CPU compute-node prerequisite failed: repository helper missing or unreadable" in result.stderr
    assert f"canonical CPU checkout: {repository}" in result.stderr
    assert f"Slurm script: {runtime_script}" in result.stderr
    assert str(runtime_script.parent / "generation_prerequisites.sh") not in result.stderr


def test_worker_repository_rejects_unsafe_paths_and_wrong_commit(tmp_path: Path) -> None:
    """Reject relative, symlinked, helper-symlinked, and wrong-commit checkouts."""
    repository = _fake_checkout(tmp_path)
    runtime_script = _spooled_script(repository, "generation_cpu_smoke.sh", tmp_path)
    environment, venv, _log = _fake_environment(tmp_path, include_rsync=False)
    storage = tmp_path / "storage"
    storage.mkdir()
    environment.update({"SLURM_JOB_ID": "123", "GENERATION_GIT_COMMIT": _COMMIT})

    relative = subprocess.run(
        _compute_command(runtime_script, Path("relative/repo"), venv, tmp_path, storage),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert relative.returncode == 1
    assert "explicit canonical CPU repository required" in relative.stderr

    alias = tmp_path / "checkout-alias"
    alias.symlink_to(repository, target_is_directory=True)
    linked_repository = subprocess.run(
        _compute_command(runtime_script, alias, venv, tmp_path, storage),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert linked_repository.returncode == 1
    assert "safe canonical CPU checkout" in linked_repository.stderr

    helper = repository / "scripts/generation_prerequisites.sh"
    external_helper = tmp_path / "external-prerequisites.sh"
    shutil.copy2(helper, external_helper)
    helper.unlink()
    helper.symlink_to(external_helper)
    linked_helper = subprocess.run(
        _compute_command(runtime_script, repository, venv, tmp_path, storage),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert linked_helper.returncode == 1
    assert "repository helper missing or unreadable" in linked_helper.stderr

    helper.unlink()
    shutil.copy2(external_helper, helper)
    (repository / ".git/HEAD").write_text(f"{'b' * 40}\n", encoding="utf-8")
    wrong_commit = subprocess.run(
        _compute_command(runtime_script, repository, venv, tmp_path, storage),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert wrong_commit.returncode == 1
    assert f"checkout commit {_COMMIT}" in wrong_commit.stderr


def test_submitted_workers_source_only_the_explicit_checkout_helper() -> None:
    """Protect every maintained worker from script-directory sibling loading."""
    repository = Path(__file__).resolve().parents[2]
    for name in _WORKER_SCRIPTS:
        source = (repository / "scripts" / name).read_text(encoding="utf-8")
        assert 'source "${PREREQUISITE_HELPER}"' in source
        assert "generation_validate_cpu_venv" in source
        assert 'source "${GENERATION_CPU_VENV}/bin/activate"' not in source
        assert 'source "${SCRIPT_DIR}/generation_prerequisites.sh"' not in source
        assert 'dirname "${BASH_SOURCE[0]}"' not in source
