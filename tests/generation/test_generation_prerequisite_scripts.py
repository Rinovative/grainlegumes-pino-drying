# ruff: noqa: S101, S603
"""Execution-domain routing for Generation shell prerequisites."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

_COMMIT = "a" * 40
_REPOSITORY_URL = "https://github.com/Rinovative/grainlegumes-pino-drying.git"


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
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "activate").write_text("", encoding="utf-8")
    _executable(
        venv / "bin" / "python",
        """printf 'venv-python <%s>\\n' "$*" >> "${FAKE_COMMAND_LOG}"
if [[ "${FAKE_VENV_IMPORT_FAIL:-false}" == true && " $* " == *" -c "* ]]; then
  printf 'synthetic package import failure\\n' >&2
  exit 17
fi""",
    )
    environment = {
        "PATH": str(binary),
        "FAKE_COMMAND_LOG": str(log),
        "FAKE_GIT_COMMIT": _COMMIT,
        "FAKE_REPOSITORY_URL": _REPOSITORY_URL,
    }
    return environment, venv, log


def _compute_command(repository: Path, venv: Path, tmp_path: Path, storage: Path) -> list[str]:
    """Return the exact fake compute preflight command."""
    return [
        "/bin/bash",
        str(repository / "scripts/generation_cpu_smoke.sh"),
        str(venv),
        str(tmp_path / "campaign.yaml"),
        str(storage),
        "-",
        "16",
        "environment-only",
        "Python/3.10",
        "Comsol/v6.4",
        "python3",
        "comsol",
        "slurm",
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


def test_compute_preflight_requires_only_compute_capabilities(tmp_path: Path) -> None:
    """Pass with login rsync and scheduler commands absent from the allocation."""
    repository = Path(__file__).resolve().parents[2]
    environment, venv, log = _fake_environment(tmp_path, include_rsync=False)
    scratch = tmp_path / "scratch"
    storage = tmp_path / "storage"
    scratch.mkdir()
    storage.mkdir()
    environment.update({"SLURM_JOB_ID": "123", "TMPDIR": str(scratch)})

    result = subprocess.run(
        _compute_command(repository, venv, tmp_path, storage),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert "Native CPU environment-only completed" in result.stdout
    evidence = result.stdout + result.stderr + log.read_text(encoding="utf-8")
    for login_only in ("rsync", "sbatch", "squeue", "sacct", "scancel"):
        assert login_only not in evidence
    assert not tuple(scratch.iterdir())


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
    environment, venv, _log = _fake_environment(tmp_path, include_rsync=True)
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
    assert "domain=CPU login check=Generation-venv-imports status=pass" in result.stdout


def test_compute_preflight_reports_package_import_failure(tmp_path: Path) -> None:
    """Surface a project/dependency import failure as a compute prerequisite."""
    repository = Path(__file__).resolve().parents[2]
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
        }
    )

    result = subprocess.run(
        _compute_command(repository, venv, tmp_path, storage),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 1
    assert "CPU compute-node prerequisite failed: Generation CPU venv package/imports" in result.stderr
