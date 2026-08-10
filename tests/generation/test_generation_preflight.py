# ruff: noqa: S101
"""Native CPU preflight environment, resource, and self-cleaning contracts."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from src import generation

_CAMPAIGN = Path("configs/generation/campaigns/steady_flow/family_generalization.yaml")


def _paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path]:
    """Create test-owned persistent roots and bind one synthetic active venv."""
    storage = tmp_path / "persistent storage"
    work = tmp_path / "node work"
    venv = tmp_path / "native venv"
    storage.mkdir(parents=True)
    work.mkdir()
    (venv / "bin").mkdir(parents=True)
    monkeypatch.setattr(
        generation.preflight.sys,
        "executable",
        str(venv / "bin/python3"),
    )
    monkeypatch.setattr(generation.preflight.sys, "version_info", (3, 10, 14, "final", 0))
    monkeypatch.setattr(generation.preflight.sys, "version", "3.10.14 (synthetic test runtime)")
    return storage, work, venv


def _fake_capabilities(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide non-solving native command and version evidence."""

    def fake_which(name: str) -> str | None:
        return None if name == "quota" else f"/synthetic/bin/{name}"

    def fake_version(command: list[str]) -> dict[str, Any]:
        return {
            "arguments": command,
            "exit_code": 0,
            "output": "Python 3.10.14; COMSOL Multiphysics 6.4; synthetic tool",
        }

    monkeypatch.setattr(generation.preflight.shutil, "which", fake_which)
    monkeypatch.setattr(generation.preflight, "_version_output", fake_version)


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Run one production-blocked but environment-complete preflight."""
    storage, work, venv = _paths(tmp_path, monkeypatch)
    _fake_capabilities(monkeypatch)
    return generation.preflight.run_cpu_preflight(
        _CAMPAIGN,
        only_batch=None,
        storage_root=storage,
        work_root=work,
        venv_path=venv,
        max_nodes=1,
        cases_per_node=2,
        cores_per_case=16,
        max_parallel_cases=2,
        cores_per_node=32,
    )


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
    assert report["resource_plan"] == {
        "max_nodes": 1,
        "cases_per_node": 2,
        "cores_per_case": 16,
        "max_parallel_cases": 2,
        "cores_per_node": 32,
        "effective_parallel_cases": 2,
        "effective_nodes": 1,
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

    monkeypatch.setattr(generation.preflight.shutil, "which", fake_which)
    with pytest.raises(FileNotFoundError, match=missing):
        generation.preflight.run_cpu_preflight(
            _CAMPAIGN,
            only_batch=None,
            storage_root=storage,
            work_root=work,
            venv_path=venv,
            max_nodes=1,
            cases_per_node=1,
            cores_per_case=1,
            max_parallel_cases=1,
            cores_per_node=32,
        )
    assert not tuple(work.iterdir())


def test_preflight_fails_clearly_for_missing_import_and_wrong_modules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protect venv dependency and exact module-stack failure messages."""
    storage, work, venv = _paths(tmp_path, monkeypatch)
    monkeypatch.setattr(
        generation.preflight.importlib.util,
        "find_spec",
        lambda name: None if name == "h5py" else object(),
    )
    with pytest.raises(ModuleNotFoundError, match="h5py"):
        generation.preflight.run_cpu_preflight(
            _CAMPAIGN,
            only_batch=None,
            storage_root=storage,
            work_root=work,
            venv_path=venv,
            max_nodes=1,
            cases_per_node=1,
            cores_per_case=1,
            max_parallel_cases=1,
            cores_per_node=32,
        )
    monkeypatch.undo()
    storage, work, venv = _paths(tmp_path / "wrong modules", monkeypatch)
    _fake_capabilities(monkeypatch)
    campaign = generation.config.load_campaign_config(
        _CAMPAIGN,
        require_executable=False,
    )
    execution = copy.deepcopy(campaign.execution_values)
    execution["site"]["python_module"] = "Python/3.11"
    wrong = replace(campaign, execution_values=execution)
    monkeypatch.setattr(
        generation.preflight.config_service,
        "load_campaign_config",
        lambda *_args, **_kwargs: wrong,
    )
    with pytest.raises(ValueError, match="native ICE contract"):
        generation.preflight.run_cpu_preflight(
            _CAMPAIGN,
            only_batch=None,
            storage_root=storage,
            work_root=work,
            venv_path=venv,
            max_nodes=1,
            cases_per_node=1,
            cores_per_case=1,
            max_parallel_cases=1,
            cores_per_node=32,
        )
    assert not tuple(work.iterdir())


@pytest.mark.parametrize("wrong_runtime", ["python", "comsol"])
def test_preflight_rejects_wrong_binding_runtime_version(
    wrong_runtime: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protect the exact native Python 3.10 and COMSOL 6.4 runtime."""
    storage, work, venv = _paths(tmp_path, monkeypatch)
    _fake_capabilities(monkeypatch)
    expected = "Python 3.10 exactly"
    if wrong_runtime == "python":
        monkeypatch.setattr(generation.preflight.sys, "version_info", (3, 11, 0, "final", 0))
        monkeypatch.setattr(generation.preflight.sys, "version", "3.11.0 (synthetic wrong runtime)")
    else:
        expected = "COMSOL must report version 6.4"
        monkeypatch.setattr(
            generation.preflight,
            "_version_output",
            lambda command: {"arguments": command, "exit_code": 0, "output": "COMSOL Multiphysics 6.3"},
        )
    with pytest.raises(RuntimeError, match=expected):
        generation.preflight.run_cpu_preflight(
            _CAMPAIGN,
            only_batch=None,
            storage_root=storage,
            work_root=work,
            venv_path=venv,
            max_nodes=1,
            cases_per_node=1,
            cores_per_case=1,
            max_parallel_cases=1,
            cores_per_node=32,
        )
    assert not tuple(work.iterdir())


def test_preflight_rejects_resource_oversubscription_before_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protect the physical cores-per-node cap before scratch mutation."""
    storage, work, venv = _paths(tmp_path, monkeypatch)
    _fake_capabilities(monkeypatch)
    with pytest.raises(ValueError, match="cores_per_node"):
        generation.preflight.run_cpu_preflight(
            _CAMPAIGN,
            only_batch=None,
            storage_root=storage,
            work_root=work,
            venv_path=venv,
            max_nodes=1,
            cases_per_node=5,
            cores_per_case=8,
            max_parallel_cases=1,
            cores_per_node=32,
        )
    assert not tuple(work.iterdir())
