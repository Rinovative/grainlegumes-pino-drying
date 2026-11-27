"""Protect the steady yardstick's batch wiring from validation Sources to the steady airflow service.

Responsibilities:
  - Check that one steady run admits its validation Sources in one catalog call and predicts them in one airflow
    service call, in validation order, without the one-case paths
  - Check that each served reference is still compared with its Source ``case.h5`` fields

Design principles:
  - Test-owned fakes stand in for the run, split, manifest, catalog, reader and service; the contract is the wiring
    of ``steady_validation_pairs``, not admission or inference, which their own tests cover

This module does NOT:
  - Admit real Generation cases, run a model or build a tolerance table
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import torch

from src.analysis import field_tests

if TYPE_CHECKING:
    from pathlib import Path

_TOLERANCES = field_tests.tolerances
_CASE_IDS = ("lentil__case_0001__a", "lentil__case_0002__b")
_SHAPE = (2, 3)


def _source(index: int) -> dict[str, str]:
    """Return the Source identity of one validation member."""
    return {
        "task": "steady_flow",
        "campaign_name": "source",
        "campaign_digest": "a" * 64,
        "material_family": "lentil",
        "case_id": f"case_000{index + 1}",
        "case_input_id": f"input-{index}",
        "simulation_case_id": f"simulation-{index}",
        "case_hdf5_sha256": str(index) * 64,
    }


def _fields(index: int) -> dict[str, np.ndarray]:
    """Return the u, v, p reference fields of one validation member."""
    return {name: np.full(_SHAPE, float(index + offset)) for offset, name in enumerate(("u", "v", "p"))}


def _patch_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, served_offset: float = 0.0) -> dict[str, list[Any]]:
    """Replace the run, split, manifest, catalog, reader and airflow service with recording fakes."""
    calls: dict[str, list[Any]] = {"resolve": [], "predict": []}
    paths = {tmp_path / f"{case_id}.h5": index for index, case_id in enumerate(_CASE_IDS)}
    part = SimpleNamespace(dataset_id="validation-dataset", case_ids=_CASE_IDS)
    manifest = {
        "source_task": "steady_flow",
        "source_case_identities": [{"package_case_id": case_id, "source": _source(index)} for index, case_id in enumerate(_CASE_IDS)],
    }
    completed = {"run_dir": str(tmp_path / "run"), "effective_config_digest": "c" * 64, "selected_checkpoint_sha256": "d" * 64}

    def resolve(task: str, sources: list[dict[str, str]], *, storage_root: Any) -> tuple[Any, ...]:
        calls["resolve"].append((task, sources, storage_root))
        return tuple(SimpleNamespace(hdf5_path=path) for path in paths)

    def predict(_steady: Any, *, task: str, picks: list[dict[str, str]], storage_root: Any) -> Any:
        calls["predict"].append((task, picks, storage_root))
        for index in range(len(picks)):
            reference = torch.from_numpy(np.stack(list(_fields(index).values())) + served_offset * index).float()
            yield SimpleNamespace(fields=("u", "v", "p"), reference=reference, prediction=2.0 * reference)

    def one_case(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("the steady yardstick used a one-case path")

    def read_case_fields(path: Path) -> Any:
        return field_tests.endpoints.CaseFields(task="steady_flow", fields=_fields(paths[path]))

    split = SimpleNamespace(task="steady_flow", role=lambda _name: part)
    monkeypatch.setattr(_TOLERANCES.experiments.run, "validate_evaluable_run", lambda _run_dir: completed)
    monkeypatch.setattr(_TOLERANCES.learning.inference.context, "load_steady_model", lambda **_kwargs: object())
    monkeypatch.setattr(_TOLERANCES.datasets.preprocessing.splits, "read_saved_split", lambda _run_dir: split)
    monkeypatch.setattr(_TOLERANCES.datasets.packages, "load_package_manifest_evidence", lambda *_args, **_kwargs: manifest)
    monkeypatch.setattr(_TOLERANCES.generation.publication.catalog, "resolve_raw_case_references", resolve)
    monkeypatch.setattr(_TOLERANCES.generation.publication.catalog, "resolve_raw_case_reference", one_case)
    monkeypatch.setattr(_TOLERANCES.endpoints, "read_case_fields", read_case_fields)
    monkeypatch.setattr(_TOLERANCES.learning.inference.airflow, "predict_cases_airflow", predict)
    monkeypatch.setattr(_TOLERANCES.learning.inference.airflow, "predict_case_airflow", one_case)
    return calls


def test_steady_run_admits_and_predicts_its_validation_cases_in_one_call_each(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One run makes one catalog admission and one airflow service call, and pairs each case with its own prediction."""
    calls = _patch_run(monkeypatch, tmp_path)

    pairs = list(_TOLERANCES.steady_validation_pairs(tmp_path / "run", storage_root=tmp_path))

    assert [len(calls["resolve"]), len(calls["predict"])] == [1, 1]
    assert calls["resolve"][0][1] == [{key: _source(index)[key] for key in _TOLERANCES._SOURCE_SELECTOR_KEYS} for index in range(len(_CASE_IDS))]
    task, picks, _storage = calls["predict"][0]
    assert task == "steady_flow"
    assert picks == [{"source_name": "source", "material_family": "lentil", "case_id": f"case_000{index + 1}"} for index in range(len(_CASE_IDS))]
    assert [pair.case_id for pair in pairs] == list(_CASE_IDS)
    for index, pair in enumerate(pairs):
        expected = {name: 2.0 * value for name, value in _fields(index).items()}
        assert pair.run["path"] == "run"
        for name in ("u", "v", "p"):
            np.testing.assert_allclose(pair.prediction.fields[name], expected[name])
        np.testing.assert_allclose(pair.prediction.fields["speed"], np.hypot(expected["u"], expected["v"]))


def test_steady_run_still_compares_each_served_reference_with_its_case_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A served reference of the second case that differs from its Source case.h5 fields is refused."""
    _patch_run(monkeypatch, tmp_path, served_offset=1.0)

    pairs = _TOLERANCES.steady_validation_pairs(tmp_path / "run", storage_root=tmp_path)

    assert next(pairs).case_id == _CASE_IDS[0]
    with pytest.raises(ValueError, match=r"differs from its Source case\.h5"):
        next(pairs)
