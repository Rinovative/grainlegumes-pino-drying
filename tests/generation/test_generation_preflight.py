# ruff: noqa: S101
"""Native CPU preflight environment, resource, and self-cleaning contracts."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from src import generation
from src.generation.runtime import generation_runtime_preflight as preflight

_CAMPAIGN = Path("configs/generation/campaigns/steady_flow/family_generalization.yaml")


def _paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path]:
    """Create test-owned persistent roots and bind one synthetic active venv."""
    storage = tmp_path / "persistent storage"
    work = tmp_path / "node work"
    venv = tmp_path / "native venv"
    base_prefix = tmp_path / "software/Python/3.10"
    base_python = base_prefix / "bin/python3.10"
    storage.mkdir(parents=True)
    work.mkdir()
    (venv / "bin").mkdir(parents=True)
    base_python.parent.mkdir(parents=True)
    base_python.write_text("synthetic module Python\n", encoding="utf-8")
    base_python.chmod(0o755)
    (venv / "bin/python").symlink_to(base_python)
    (venv / "pyvenv.cfg").write_text(
        f"home = {base_python.parent}\nversion = 3.10.14\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(preflight.sys, "executable", str(venv / "bin/python"))
    monkeypatch.setattr(preflight.sys, "prefix", str(venv))
    monkeypatch.setattr(preflight.sys, "base_prefix", str(base_prefix))
    monkeypatch.setattr(preflight.sys, "exec_prefix", str(venv))
    monkeypatch.setattr(preflight.sys, "version_info", (3, 10, 14, "final", 0))
    monkeypatch.setattr(preflight.sys, "version", "3.10.14 (synthetic test runtime)")
    return storage, work, venv


def _fake_capabilities(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide non-solving native command and version evidence."""

    def fake_which(name: str) -> str | None:
        login_only = {"quota", "rsync", "sbatch", "squeue", "sacct", "scancel"}
        return None if name in login_only else f"/synthetic/bin/{name}"

    def fake_version(command: list[str], *, timeout_seconds: float) -> dict[str, Any]:
        assert timeout_seconds > 0.0
        return {
            "arguments": command,
            "exit_code": 0,
            "output": "Python 3.10.14; COMSOL Multiphysics 6.4; synthetic tool",
        }

    monkeypatch.setattr(preflight.shutil, "which", fake_which)
    monkeypatch.setattr(preflight, "_version_output", fake_version)


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Run one production-blocked but environment-complete preflight."""
    storage, work, venv = _paths(tmp_path, monkeypatch)
    _fake_capabilities(monkeypatch)
    return preflight.run_cpu_preflight(
        _CAMPAIGN,
        only_batch=None,
        storage_root=storage,
        work_root=work,
        venv_path=venv,
    )


def test_generation_venv_accepts_external_module_python_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accept a venv launcher whose physical target is the module Python."""
    _storage, _work, venv = _paths(tmp_path, monkeypatch)

    evidence = preflight.validate_generation_venv(
        venv,
        domain="CPU compute-node",
    )

    launcher = venv / "bin/python"
    assert launcher.is_symlink()
    assert launcher.resolve().parent.parent == tmp_path / "software/Python/3.10"
    assert launcher.resolve().is_relative_to(venv) is False
    assert evidence["launcher"] == str(launcher)
    assert evidence["resolved_launcher_target"] == str(launcher.resolve())
    assert evidence["sys_prefix"] == str(venv)
    assert evidence["sys_base_prefix"] == str(tmp_path / "software/Python/3.10")


def test_generation_venv_rejects_system_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject an interpreter whose runtime prefix is its base prefix."""
    _storage, _work, venv = _paths(tmp_path, monkeypatch)
    base_prefix = tmp_path / "software/Python/3.10"
    monkeypatch.setattr(preflight.sys, "executable", str(base_prefix / "bin/python3.10"))
    monkeypatch.setattr(preflight.sys, "prefix", str(base_prefix))
    monkeypatch.setattr(preflight.sys, "base_prefix", str(base_prefix))
    monkeypatch.setattr(preflight.sys, "exec_prefix", str(base_prefix))

    with pytest.raises(RuntimeError, match=r"sys[.]prefix == sys[.]base_prefix.*not active"):
        preflight.validate_generation_venv(venv, domain="CPU compute-node")


def test_generation_venv_rejects_wrong_runtime_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject an active interpreter belonging to a different venv."""
    _storage, _work, venv = _paths(tmp_path, monkeypatch)
    other = tmp_path / "another venv"
    monkeypatch.setattr(preflight.sys, "prefix", str(other))
    monkeypatch.setattr(preflight.sys, "exec_prefix", str(other))

    with pytest.raises(RuntimeError, match=rf"configured Generation venv is {venv}.*sys[.]prefix={other}"):
        preflight.validate_generation_venv(venv, domain="CPU compute-node")


@pytest.mark.parametrize("broken", ["launcher", "non_executable_launcher", "metadata"])
def test_generation_venv_rejects_broken_runtime_files(
    broken: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require the configured launcher and standard Python venv metadata."""
    _storage, _work, venv = _paths(tmp_path, monkeypatch)
    if broken == "launcher":
        (venv / "bin/python").unlink()
    elif broken == "non_executable_launcher":
        (venv / "bin/python").resolve().chmod(0o644)
    else:
        (venv / "pyvenv.cfg").unlink()

    expected = "metadata" if broken == "metadata" else "launcher"
    with pytest.raises(FileNotFoundError, match=expected):
        preflight.validate_generation_venv(venv, domain="CPU compute-node")


def test_generation_venv_rejects_missing_configured_root(tmp_path: Path) -> None:
    """Report a configured venv directory that does not exist."""
    missing = tmp_path / "missing venv"

    with pytest.raises(FileNotFoundError, match="configured Generation venv root"):
        preflight.validate_generation_venv(missing, domain="CPU compute-node")


def test_generation_venv_rejects_unsafe_configured_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a configured venv root that is itself a symlink."""
    _storage, _work, venv = _paths(tmp_path, monkeypatch)
    alias = tmp_path / "venv alias"
    alias.symlink_to(venv, target_is_directory=True)

    with pytest.raises(ValueError, match="root must not be a symlink"):
        preflight.validate_generation_venv(alias, domain="CPU compute-node")


def test_preflight_separates_environment_from_runtime_and_removes_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protect non-solving readiness evidence and the self-cleaning probe."""
    report = _run(tmp_path, monkeypatch)
    work = tmp_path / "node work"
    assert report["status"] == "environment_ready_production_blocked"
    assert report["production_configuration_ready"] is False
    assert "unconfirmed required export mappings" in report["production_configuration_blocker"]
    assert report["production_solve_started"] is False
    assert report["domain"] == "CPU compute-node"
    assert set(report["commands"]) == {"python", "comsol"}
    assert set(report["versions"]) == {"python", "comsol"}
    assert report["checks"]["Generation-venv-imports"]["status"] == "pass"
    assert report["python"]["executable"].endswith("native venv/bin/python")
    assert report["python"]["resolved_executable"].endswith("software/Python/3.10/bin/python3.10")
    assert report["python"]["venv_runtime"]["sys_prefix"].endswith("native venv")
    assert report["submission_plan"] == {
        "cases_per_job": 1,
        "cores_per_case": 16,
        "cores_per_node": 32,
        "pending_buffer": 1,
        "poll_interval_seconds": 15,
        "max_running_cases": None,
    }
    assert report["path_cleanup_probe"]["probe_removed"] is True
    assert not Path(report["path_cleanup_probe"]["probe_path"]).exists()
    assert not tuple(work.iterdir())
    assert set(report["templates"]) == {"steady_flow", "transient_drying"}
    assert all(
        template["sidecar_validation"] == "pass" and template["comsol_internal_contract"] == "runtime_unverified"
        for template in report["templates"].values()
    )


@pytest.mark.parametrize("missing", ["comsol", "python3"])
def test_preflight_fails_clearly_when_native_command_is_missing(
    missing: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protect explicit failure for missing native Python or COMSOL commands."""
    storage, work, venv = _paths(tmp_path, monkeypatch)

    def fake_which(name: str) -> str | None:
        if name in {missing, "quota"}:
            return None
        return f"/synthetic/bin/{name}"

    monkeypatch.setattr(preflight.shutil, "which", fake_which)
    with pytest.raises(FileNotFoundError, match=missing):
        preflight.run_cpu_preflight(
            _CAMPAIGN,
            only_batch=None,
            storage_root=storage,
            work_root=work,
            venv_path=venv,
        )
    assert not tuple(work.iterdir())


def test_preflight_fails_clearly_for_missing_import_and_wrong_modules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protect venv dependency and exact module-stack failure messages."""
    storage, work, venv = _paths(tmp_path, monkeypatch)
    monkeypatch.setattr(
        preflight.importlib.util,
        "find_spec",
        lambda name: None if name == "h5py" else object(),
    )
    with pytest.raises(ModuleNotFoundError, match="h5py"):
        preflight.run_cpu_preflight(
            _CAMPAIGN,
            only_batch=None,
            storage_root=storage,
            work_root=work,
            venv_path=venv,
        )
    monkeypatch.undo()
    storage, work, venv = _paths(tmp_path / "wrong modules", monkeypatch)
    _fake_capabilities(monkeypatch)
    campaign = generation.cases.config.load_campaign_config(
        _CAMPAIGN,
        require_executable=False,
    )
    execution = copy.deepcopy(campaign.execution_values)
    execution["site"]["python_module"] = "Python/3.11"
    wrong = replace(campaign, execution_values=execution)
    monkeypatch.setattr(
        preflight.config_service,
        "load_campaign_config",
        lambda *_args, **_kwargs: wrong,
    )
    with pytest.raises(RuntimeError, match=r"Configured Python module expects version 3[.]11"):
        preflight.run_cpu_preflight(
            _CAMPAIGN,
            only_batch=None,
            storage_root=storage,
            work_root=work,
            venv_path=venv,
        )
    assert not tuple(work.iterdir())


def test_preflight_fails_clearly_when_project_package_import_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require the project Generation package in the compute venv."""
    storage, work, venv = _paths(tmp_path, monkeypatch)
    monkeypatch.setattr(
        preflight.importlib.util,
        "find_spec",
        lambda name: None if name == "src.generation.cli.cli_generation" else object(),
    )
    with pytest.raises(ModuleNotFoundError, match=r"CPU compute-node prerequisite missing.*src[.]generation"):
        preflight.run_cpu_preflight(
            _CAMPAIGN,
            only_batch=None,
            storage_root=storage,
            work_root=work,
            venv_path=venv,
        )
    assert not tuple(work.iterdir())


def test_preflight_fails_clearly_when_scratch_is_not_writable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require writable compute scratch before any runtime command checks."""
    storage, work, venv = _paths(tmp_path, monkeypatch)
    real_access = preflight.os.access

    def fake_access(path: Path | str, mode: int) -> bool:
        return False if Path(path).resolve() == work.resolve() else real_access(path, mode)

    monkeypatch.setattr(preflight.os, "access", fake_access)
    with pytest.raises(PermissionError, match="work_root is not writable"):
        preflight.run_cpu_preflight(
            _CAMPAIGN,
            only_batch=None,
            storage_root=storage,
            work_root=work,
            venv_path=venv,
        )


@pytest.mark.parametrize("wrong_runtime", ["python", "comsol"])
def test_preflight_rejects_wrong_binding_runtime_version(
    wrong_runtime: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protect configured Python and COMSOL module/runtime versions."""
    storage, work, venv = _paths(tmp_path, monkeypatch)
    _fake_capabilities(monkeypatch)
    expected = r"Configured Python module expects version 3[.]10"
    if wrong_runtime == "python":
        monkeypatch.setattr(preflight.sys, "version_info", (3, 11, 0, "final", 0))
        monkeypatch.setattr(preflight.sys, "version", "3.11.0 (synthetic wrong runtime)")
    else:
        expected = r"Configured comsol module expects version 6[.]4"

        def wrong_comsol_version(command: list[str], *, timeout_seconds: float) -> dict[str, Any]:
            assert timeout_seconds > 0.0
            output = "COMSOL Multiphysics 6.3" if command[0].endswith("/comsol") else "Python 3.10.14; synthetic tool"
            return {"arguments": command, "exit_code": 0, "output": output}

        monkeypatch.setattr(preflight, "_version_output", wrong_comsol_version)
    with pytest.raises(RuntimeError, match=expected):
        preflight.run_cpu_preflight(
            _CAMPAIGN,
            only_batch=None,
            storage_root=storage,
            work_root=work,
            venv_path=venv,
        )
    assert not tuple(work.iterdir())
