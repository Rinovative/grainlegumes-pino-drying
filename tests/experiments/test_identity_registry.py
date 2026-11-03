"""Protect experiment identity: the epoch-free digest, the registry and fresh-start admission.

Responsibilities:
  - Check which settings the allocation digest covers and which it ignores
  - Check that the registry adapter binds the allocation digest to the run directory
  - Check that an identical experiment is refused under another config name or revision
  - Check that the train command and service refuse an authored config combined with --resume

Design principles:
  - Identity is decided by resolved science, never by names, paths, epochs or revision labels
  - Admission rejects before any run state is allocated or written; the shared registry core has its own suite

This module does NOT:
  - Train models or continue existing runs; ``test_run_allocation`` and the lifecycle tests own that
"""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest
from support import configs

from src import common, experiments
from src.experiments.cli import cli_train
from src.experiments.config import experiments_config_loader as loader

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_IdenticalExperimentError = experiments.registry.IdenticalExperimentError


def _resolved(raw: dict[str, Any], output_root: Path, *, name_suffix: str | None = None) -> dict[str, Any]:
    """Resolve one request under the test family, optionally as another config file name."""
    source = configs.source_for(raw)
    if name_suffix is not None:
        source = replace(source, name=f"{source.name}__{name_suffix}")
    return loader.resolve_config(copy.deepcopy(raw), source=source, output_root=output_root)


def _allocate(config: dict[str, Any], run_dir: Path) -> None:
    """Bind ``run_dir`` to the config's identity and allocate it, as a fresh start does."""
    with experiments.registry.reserve(config, run_dir):
        run_dir.mkdir(parents=True)


@pytest.mark.parametrize(
    ("mutate", "same_experiment"),
    [
        (lambda raw: raw["training"].update(epochs=raw["training"]["epochs"] + 5), True),
        (lambda raw: raw.update(run={**raw["run"], "revision": "rerun"}), True),
        (lambda raw: raw["run"].update(seed=raw["run"]["seed"] + 1), False),
        (lambda raw: raw["optimizer"].update(lr=raw["optimizer"]["lr"] * 2), False),
        (lambda raw: raw["data"].update(train_dataset="other_dataset"), False),
    ],
    ids=["epochs", "revision", "seed", "optimizer", "dataset"],
)
def test_allocation_digest_ignores_epochs_and_labels_but_not_science(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], object],
    same_experiment: bool,
) -> None:
    """Only a change of the training science makes an experiment new."""
    raw = configs.direct_config()
    changed = copy.deepcopy(raw)
    mutate(changed)

    baseline = loader.allocation_digest(_resolved(raw, tmp_path))

    assert (loader.allocation_digest(_resolved(changed, tmp_path)) == baseline) is same_experiment


def test_transient_allocation_digest_binds_the_a0_budget_and_the_a0_of_each_continuation(tmp_path: Path) -> None:
    """A0's epoch budget is identity; B and A1 belong to the A0 they start from, while their epoch caps stay progress."""
    plan = configs.resolve_plan(configs.transient_matched_config(), output_root=tmp_path)
    longer_a0 = copy.deepcopy(dict(plan.a0))
    longer_a0["training"]["epochs"] = int(plan.a0["training"]["epochs"]) + 1
    other_a0_b = copy.deepcopy(dict(plan.b))
    other_a0_b["run"]["layout"]["a0_id"] = "f" * 16
    capped_b = copy.deepcopy(dict(plan.b))
    capped_b["training"]["epochs"] = int(plan.b["training"]["epochs"]) + 1

    assert loader.allocation_digest(longer_a0) != loader.allocation_digest(plan.a0)
    assert loader.allocation_digest(other_a0_b) != loader.allocation_digest(plan.b)
    assert loader.allocation_digest(capped_b) == loader.allocation_digest(plan.b)


def test_resume_admission_refuses_another_a0_epoch_budget(tmp_path: Path) -> None:
    """A saved A0 refuses another epoch budget before any run state is touched; the hint points at --epochs, which trains it beside it."""
    plan = configs.resolve_plan(configs.transient_matched_config(), output_root=tmp_path)
    requested = copy.deepcopy(dict(plan.a0))
    requested["training"]["epochs"] = int(plan.a0["training"]["epochs"]) + 1
    requested["training"]["stage_schedule"]["a0_max_epochs"] = requested["training"]["epochs"]

    with pytest.raises(ValueError, match=r"cannot change its epoch budget.*--epochs"):
        experiments.run.validate_resume_config(requested, plan.a0)


def test_allocation_digest_ignores_the_config_name_and_output_root(tmp_path: Path) -> None:
    """The same science is the same experiment under another config file name or output root."""
    raw = configs.direct_config()

    baseline = loader.allocation_digest(_resolved(raw, tmp_path / "first"))

    assert loader.allocation_digest(_resolved(raw, tmp_path / "second", name_suffix="copy")) == baseline


def test_registry_binds_the_allocation_digest_and_reports_an_identical_experiment(tmp_path: Path) -> None:
    """The adapter scopes the shared registry to the output root and reports a duplicate with both directories."""
    config = _resolved(configs.direct_config(), tmp_path)
    run_dir = common.paths.resolve_config_run_dir(config)
    second = tmp_path / "second"
    assert experiments.registry.lookup(config) is None

    _allocate(config, run_dir)

    entry = experiments.registry.lookup(config)
    assert entry is not None
    assert (entry.target, entry.kind, entry.digest) == (run_dir, "run", loader.allocation_digest(config))
    with pytest.raises(_IdenticalExperimentError) as rejected:
        _allocate(config, second)
    assert (rejected.value.existing_run_dir, rejected.value.requested_run_dir) == (run_dir, second)
    assert not second.exists()
    with pytest.raises(ValueError, match="outside its namespace root"), experiments.registry.reserve(config, tmp_path.parent / "elsewhere"):
        pytest.fail("a directory outside the output root must never be admitted")


def _write_request(tmp_path: Path, raw: dict[str, Any], *, name_suffix: str | None = None) -> Path:
    """Write the request under its canonical test-family path, optionally as another config file name."""
    source = configs.source_for(raw)
    name = source.name if name_suffix is None else f"{source.name}__{name_suffix}"
    return configs.write_yaml(
        tmp_path / "configs" / "03_learning" / source.task / "experiments" / source.family / f"{name}.yaml",
        raw,
    )


def _tree(root: Path) -> dict[str, bytes]:
    """Return every file below ``root`` for byte-for-byte comparison."""
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


@pytest.mark.parametrize("variant", ["config_name", "revision"])
def test_identical_experiment_is_rejected_under_another_config_name_or_revision(tmp_path: Path, variant: str) -> None:
    """A copy of an existing experiment is refused before any allocation; --revision does not bypass it."""
    raw = configs.direct_config()
    output_root = tmp_path / "outputs"
    first = experiments.run.prepare_fresh_run(_resolved(raw, output_root))
    before = _tree(output_root)
    request = _write_request(tmp_path, raw, name_suffix="copy" if variant == "config_name" else None)

    with pytest.raises(_IdenticalExperimentError, match="identical experiment already exists") as rejected:
        experiments.run.run_experiment(
            request,
            output_root=output_root,
            revision="rerun" if variant == "revision" else None,
        )

    assert rejected.value.existing_run_dir == first
    assert _tree(output_root) == before


def test_train_cli_reports_an_identical_experiment_as_a_rejected_admission(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The command fails with a clear admission message instead of a training traceback."""
    raw = configs.direct_config()
    experiments.run.prepare_fresh_run(configs.resolve(raw))
    request = _write_request(tmp_path, raw, name_suffix="copy")

    assert cli_train.main([str(request), "--artifacts", "none"]) == 1

    stderr = capsys.readouterr().err
    assert "Run admission rejected: The identical experiment already exists at" in stderr
    assert "Traceback" not in stderr


def test_train_cli_rejects_a_config_combined_with_resume(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--resume is the saved-config mode only; an authored config continues by being rerun."""
    request = _write_request(tmp_path, configs.direct_config())

    with pytest.raises(SystemExit) as usage_error:
        cli_train.main([str(request), "--resume", str(tmp_path / "run")])

    assert usage_error.value.code == 2
    assert "--resume continues a saved run from its own config" in capsys.readouterr().err


def test_authored_config_with_resume_is_rejected_by_the_run_service(tmp_path: Path) -> None:
    """The service refuses the removed config-plus-resume combination before reading either path."""
    request = _write_request(tmp_path, configs.direct_config())

    with pytest.raises(ValueError, match="continues a saved run from its own saved config only"):
        experiments.run.run_experiment(request, resume=tmp_path / "run")
