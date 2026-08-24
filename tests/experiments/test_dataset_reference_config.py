# ruff: noqa: S101
"""Protect logical Dataset-reference resolution and saved exact evidence."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from support import configs

from src import datasets, experiments


def _record(name: str, dataset_id: str, *, revision: int = 0) -> dict[str, Any]:
    """Return one complete test-owned immutable Dataset-reference record."""
    return {
        "schema_kind": "vp2_dataset_reference",
        "schema_version": 1,
        "task": "steady_flow",
        "name": name,
        "revision": revision,
        "dataset_id": dataset_id,
        "dataset_digest": "d" * 64,
        "manifest_sha256": "a" * 64,
        "payload_sha256": "b" * 64,
        "dataset_view": "steady_flow",
        "evaluation_regime": "id" if name.endswith("id") else "near_family_ood",
        "materials": ["synthetic"],
        "source_package": {
            "dataset_name": f"steady_flow__{name}",
            "campaign_id": "campaign",
            "campaign_digest": "c" * 64,
            "source_case_count": 1,
            "source_batch_ids": ["batch"],
            "source_simulation_profiles": ["steady_flow"],
            "source_git_commits": ["commit"],
            "channel_contract_digest": "d" * 64,
        },
        "creation": {"created_at": "2026-01-01T00:00:00+00:00", "publisher": "dataset_reference"},
    }


def _logical_request() -> dict[str, Any]:
    """Return one direct request using explicit logical names and revisions."""
    raw = configs.direct_config(suffix=None)
    raw["data"]["train_dataset"] = {"name": "synthetic_id", "revision": 0}
    raw["data"]["ood_datasets"] = [{"name": "synthetic_ood", "revision": 1}]
    return raw


def test_logical_references_resolve_once_and_persist_exact_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve authored refs to exact IDs while retaining their complete records."""
    records = {
        ("synthetic_id", 0): _record("synthetic_id", "steady_exact_id"),
        ("synthetic_ood", 1): _record("synthetic_ood", "steady_exact_ood", revision=1),
    }
    calls: list[tuple[str, int]] = []

    def resolve(_task: str, name: str, revision: int, **_kwargs: Any) -> dict[str, Any]:
        calls.append((name, revision))
        return copy.deepcopy(records[(name, revision)])

    monkeypatch.setattr(datasets.packages.references, "resolve_dataset_reference_record", resolve)
    resolved = experiments.config.loader.resolve_config(_logical_request())

    assert calls == [("synthetic_id", 0), ("synthetic_ood", 1)]
    assert resolved["data"]["train_dataset"] == "steady_exact_id"
    assert resolved["data"]["ood_datasets"] == ["steady_exact_ood"]
    assert resolved["data"]["dataset_references"] == {
        "train": records[("synthetic_id", 0)],
        "ood": [records[("synthetic_ood", 1)]],
    }
    assert "synthetic_id" in resolved["run"]["name"]
    assert "_d0" not in resolved["run"]["name"]
    assert "_d1" not in resolved["run"]["name"]
    assert experiments.config.loader.validate_resolved_config(resolved) == resolved


def test_train_dataset_revision_one_is_visible_in_the_run_label(monkeypatch: pytest.MonkeyPatch) -> None:
    """Display a nonzero Train Dataset revision without exposing exact identity."""
    raw = _logical_request()
    raw["data"]["train_dataset"]["revision"] = 1
    records = {
        ("synthetic_id", 1): _record("synthetic_id", "steady_exact_id", revision=1),
        ("synthetic_ood", 1): _record("synthetic_ood", "steady_exact_ood", revision=1),
    }

    def resolve(_task: str, name: str, revision: int, **_kwargs: Any) -> dict[str, Any]:
        return copy.deepcopy(records[(name, revision)])

    monkeypatch.setattr(datasets.packages.references, "resolve_dataset_reference_record", resolve)
    resolved = experiments.config.loader.resolve_config(raw)

    assert "_d1_" in resolved["run"]["name"]
    assert "steady_exact_id" not in resolved["run"]["name"]


def test_pinned_resume_evidence_never_chases_current_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-resolve a request from saved evidence even when the external ref is absent."""
    evidence = {
        "train": _record("synthetic_id", "steady_exact_id"),
        "ood": [_record("synthetic_ood", "steady_exact_ood", revision=1)],
    }

    def unexpected(*_args: object, **_kwargs: object) -> dict[str, Any]:
        pytest.fail("Pinned resume attempted to resolve current external reference state")

    monkeypatch.setattr(datasets.packages.references, "resolve_dataset_reference_record", unexpected)
    resolved = experiments.config.loader.resolve_config(
        _logical_request(),
        pinned_dataset_references=evidence,
    )

    assert resolved["data"]["dataset_references"] == evidence
    assert resolved["data"]["train_dataset"] == "steady_exact_id"


def test_reference_audit_allows_missing_but_rejects_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    """Allow deleted aliases on resume while rejecting a present changed binding."""
    raw = _logical_request()
    evidence = {
        "train": _record("synthetic_id", "steady_exact_id"),
        "ood": [_record("synthetic_ood", "steady_exact_ood", revision=1)],
    }
    resolved = experiments.config.loader.resolve_config(raw, pinned_dataset_references=evidence)

    def missing(*_args: object, **_kwargs: object) -> dict[str, Any]:
        raise FileNotFoundError

    monkeypatch.setattr(datasets.packages.references, "resolve_dataset_reference_record", missing)
    assert experiments.config.loader.audit_resolved_dataset_references(resolved) == (
        "steady_flow/synthetic_id/r0",
        "steady_flow/synthetic_ood/r1",
    )

    def drift(_task: str, name: str, _revision: int, **_kwargs: Any) -> dict[str, Any]:
        record = copy.deepcopy(evidence["train"] if name == "synthetic_id" else evidence["ood"][0])
        record["manifest_sha256"] = "f" * 64
        return record

    monkeypatch.setattr(datasets.packages.references, "resolve_dataset_reference_record", drift)
    with pytest.raises(experiments.config.loader.ConfigError, match="identity drift"):
        experiments.config.loader.audit_resolved_dataset_references(resolved)


def test_legacy_exact_resolution_preserves_pre_schema_shape() -> None:
    """Keep old exact-ID configs resumable without injecting current metadata."""
    raw = configs.direct_config(suffix=None)
    resolved = experiments.config.loader.resolve_config(raw, naming_schema_version=1)

    assert "revision" not in resolved["run"]
    assert "naming_schema_version" not in resolved["run"]
    assert "dataset_references" not in resolved["data"]
    assert "metric_schema_version" not in resolved["tracking"]["wandb"]
    assert "__" in resolved["run"]["name"]
    assert experiments.config.loader.validate_resolved_config(resolved) == resolved


def test_logical_reference_requires_explicit_non_boolean_revision() -> None:
    """Reject missing and boolean logical revisions before any storage read."""
    raw = configs.direct_config()
    raw["data"]["train_dataset"] = {"name": "synthetic_id"}
    with pytest.raises(experiments.config.loader.ConfigError, match="explicit name and revision"):
        experiments.config.loader.resolve_config(raw)

    raw["data"]["train_dataset"] = {"name": "synthetic_id", "revision": True}
    with pytest.raises(experiments.config.loader.ConfigError, match="non-negative integer"):
        experiments.config.loader.resolve_config(raw)
