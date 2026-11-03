"""Canonical input-generation storage, admission, and identity contracts.

Responsibilities:
  - Check canonical input storage, source identity, dry runs, and complete campaign membership

Design principles:
  - Persisted case membership and admitted source bytes govern execution

This module does NOT:
  - Launch solvers or silently generate worker-side missing inputs
"""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

import pytest

from src import common, generation
from src.generation import generation_campaign_lifecycle as lifecycle
from src.generation.cli import cli_generation
from src.generation.runtime import generation_runtime_preparation as runtime_preparation

if TYPE_CHECKING:
    from pathlib import Path

_FAKE_GIT_COMMIT = "a" * 40
_OTHER_GIT_COMMIT = "b" * 40
_THIRD_GIT_COMMIT = "c" * 40
_FOURTH_GIT_COMMIT = "d" * 40


def _load_config(
    generation_config_factory: Any,
    profile_id: str,
    *,
    campaign_purpose: str = "source_generation",
    natural_count: int = 4,
) -> Any:
    """Load one compact synthetic maintained-profile batch."""
    config_path, _template = generation_config_factory(
        simulation_profile=profile_id,
        natural_count=natural_count,
        campaign_purpose=campaign_purpose,
    )
    return generation.cases.config.load_generation_config(
        config_path,
        only_batch=generation.cases.config.build_batch_name(
            profile_id,
            "lentil",
            "natural",
        ),
    )


def test_case_range_validation_rejects_overflow_and_non_integer_start(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """Reject bounded requests outside configured canonical membership."""
    config = _load_config(generation_config_factory, "steady_flow")
    with pytest.raises(ValueError, match="exceeds configured batch membership"):
        generation.cases.input_generation.generate_input_cases(
            config,
            2,
            case_start=config.case_indices[-1],
            storage_root=tmp_path / "overflow",
        )
    with pytest.raises(ValueError, match="must be an integer"):
        generation.cases.input_generation.generate_input_cases(
            config,
            1,
            case_start=True,
            storage_root=tmp_path / "invalid-start",
        )


def test_discovery_ignores_unmanifested_raw_directories(
    tmp_path: Path,
) -> None:
    """Use canonical metadata as the sole discovery boundary."""
    storage = tmp_path / "storage"
    source = common.paths.resolve_generation_campaign_directory("steady_flow", "unmanifested-source", storage_root=storage)
    (source / "lentil" / "case_0001").mkdir(parents=True)
    (common.paths.get_generation_root(storage_root=storage) / "arbitrary" / "case_0001").mkdir(parents=True)

    discovery = generation.cases.admission.discover_input_batches(storage)

    assert discovery.sources == ()
    assert discovery.issues == ()


@pytest.mark.parametrize("profile_id", ["steady_flow", "transient_drying"])
def test_input_generation_is_exactly_equivalent_to_runtime_preparation(
    generation_config_factory: Any,
    tmp_path: Path,
    profile_id: str,
) -> None:
    """Prove exact case identity and adapter bytes across independent owners."""
    config = _load_config(
        generation_config_factory,
        profile_id,
        campaign_purpose="source_generation",
    )
    storage = tmp_path / "storage"
    campaign = generation.cases.config.load_campaign_config(config.source_path)
    lifecycle.create_campaign(campaign, storage_root=storage)
    generation.cases.input_generation.generate_input_cases(
        config,
        len(config.case_indices),
        storage_root=storage,
    )
    blocked = AssertionError("Runtime preparation regenerated an already persisted case.")
    with patch(
        "src.generation.cases.generation_cases_input.case_service.generate_case_input_bundle",
        side_effect=blocked,
    ):
        prepared = runtime_preparation.prepare_case_work_directory(
            config,
            1,
            storage_root=storage,
            work_root=tmp_path / "work",
        )

    reference = generation.cases.input_generation.admit_configured_input_references(config, storage_root=storage)[1]
    payload = json.loads((reference.case_directory / "case.json").read_text())
    assert payload["input_files"]
    for filename in ("case.json", *sorted(payload["input_files"])):
        assert (reference.case_directory / filename).read_bytes() == (prepared.work_directory / filename).read_bytes()


def test_git_provenance_changes_case_json_but_not_scientific_case_ids(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bind the generating Git commit into case.json while the scientific case IDs stay stable."""
    config = _load_config(generation_config_factory, "steady_flow")
    first = generation.cases.case.generate_case_input_bundle(
        config,
        1,
        tmp_path / "first",
    )
    monkeypatch.setenv("GENERATION_GIT_COMMIT", _OTHER_GIT_COMMIT)
    second = generation.cases.case.generate_case_input_bundle(
        config,
        1,
        tmp_path / "second",
    )
    first_payload = json.loads((first.directory / "case.json").read_text())
    second_payload = json.loads((second.directory / "case.json").read_text())
    assert first.case_input_id == second.case_input_id
    assert first.simulation_case_id == second.simulation_case_id
    assert first_payload["git_commit"] != second_payload["git_commit"]


def test_clean_repository_source_identity_fails_closed_on_dirty_evidence() -> None:
    """Resolve exact clean HEAD and reject any porcelain worktree evidence."""
    source_service = generation.contracts.source
    clean_status = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    clean_commit = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=f"{_FAKE_GIT_COMMIT}\n",
        stderr="",
    )
    with patch.object(
        source_service.subprocess,
        "run",
        side_effect=(clean_status, clean_commit),
    ):
        assert source_service.clean_repository_git_commit() == _FAKE_GIT_COMMIT

    dirty_status = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=" M tracked.py\n?? untracked.py\n",
        stderr="",
    )
    with (
        patch.object(source_service.subprocess, "run", return_value=dirty_status),
        pytest.raises(RuntimeError, match="clean repository worktree"),
    ):
        source_service.clean_repository_git_commit()


def test_all_batch_request_preserves_canonical_campaign_order(
    tmp_path: Path,
) -> None:
    """Resolve all batches in authored campaign order without label sorting."""
    service = generation.cases.input_generation
    first = SimpleNamespace(case_indices=(1, 3))
    second = SimpleNamespace(case_indices=(2, 5))
    campaign = SimpleNamespace(batches=(first, second))
    request = service.CampaignInputGenerationRequest(
        campaign_config=tmp_path / "campaign.yaml",
        storage_root=tmp_path / "storage",
        action="dry_run",
        all_batches=True,
        all_cases=True,
    )
    with patch.object(
        service.config_service,
        "load_campaign_config",
        return_value=campaign,
    ):
        resolved_campaign, selections = service._resolve_campaign_request(request)

    assert resolved_campaign is campaign
    assert tuple(batch for batch, _indices in selections) == (first, second)
    assert tuple(indices for _batch, indices in selections) == ((1, 3), (2, 5))


def test_input_generation_command_matches_supported_shared_launch(tmp_path: Path) -> None:
    """Offer a copyable command only for the launcher's source and storage."""
    service = generation.cases.input_generation
    repository = tmp_path / "repo"
    campaign = SimpleNamespace(source_path=repository / "configs" / "campaign.yaml")
    request = service.CampaignInputGenerationRequest(
        campaign_config=campaign.source_path,
        storage_root=tmp_path / "storage",
        only_batch="synthetic-batch",
        case_count=1,
        git_commit=_FAKE_GIT_COMMIT,
    )
    with patch.object(common.paths, "get_project_root", return_value=repository):
        command, blocker = service._equivalent_cli_command(request, cast("Any", campaign))
    assert blocker is None
    assert command == "./scripts/generation inputs configs/campaign.yaml --only-batch synthetic-batch --case-count 1 --dry-run"


def test_campaign_input_generation_rejects_actions_and_reports_source_blocker(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """Reject unsupported modes and report dirty interactive provenance read-only."""
    service = generation.cases.input_generation
    with pytest.raises(ValueError, match="Unsupported input-generation action"):
        service.run_campaign_input_generation(
            service.CampaignInputGenerationRequest(
                campaign_config=tmp_path / "unused.yaml",
                storage_root=tmp_path / "unused",
                action="invalid",
            )
        )

    config_path, _template = generation_config_factory(
        simulation_profile="steady_flow",
        natural_count=1,
        campaign_purpose="source_generation",
    )
    batch_name = generation.cases.config.build_batch_name(
        "steady_flow",
        "lentil",
        "natural",
    )
    storage = tmp_path / "blocked"
    request = service.CampaignInputGenerationRequest(
        campaign_config=config_path,
        storage_root=storage,
        action="dry_run",
        only_batch=batch_name,
        case_count=1,
    )
    with (
        patch.object(
            service.source_service,
            "clean_repository_git_commit",
            side_effect=RuntimeError("worktree is dirty"),
        ),
        patch.object(
            service,
            "plan_input_cases",
            side_effect=AssertionError("blocked dry_run planned generated bytes"),
        ),
    ):
        blocked = service.run_campaign_input_generation(request)
    assert blocked["source_identity_status"] == "blocked"
    assert blocked["source_identity_blocker"] == "worktree is dirty"
    assert blocked["requested_case_range"] == [1, 1]
    assert blocked["generated_case_count"] == 0
    assert not storage.exists()


def test_generate_input_cases_cli_dry_run_selects_campaign_without_writes(
    generation_config_factory: Any,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Plan all natural campaign inputs and estimate bytes without persistence."""
    config_path, _template = generation_config_factory(
        simulation_profile="transient_drying",
        natural_count=2,
        campaign_purpose="source_generation",
    )
    storage = tmp_path / "storage"
    arguments = [
        "generate-input-cases",
        str(config_path),
        "--all-batches",
        "--only-regime",
        "natural",
        "--all-cases",
        "--dry-run",
        "--git-commit",
        _FAKE_GIT_COMMIT,
        "--storage-root",
        str(storage),
    ]

    assert cli_generation.main(arguments) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["dry_run"] is True
    assert result["selected_batch_count"] == 1
    assert result["requested_case_count"] == 2
    assert result["would_generate_case_count"] == 2
    assert result["generated_case_count"] == 0
    assert result["estimated_storage_bytes"] > 0
    assert result["input_generation_id"] is None
    assert result["input_only"] is True
    assert not storage.exists()

    conflicting = [*arguments, "--only-batch", "unused"]
    with pytest.raises(SystemExit):
        cli_generation.main(conflicting)


def test_generation_cli_renders_context_for_unexpected_key_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Never reduce a workflow failure to one bare missing-key string."""

    def fail_dispatch(_arguments: Any) -> int:
        missing_key = "T_flow_ref"
        raise KeyError(missing_key)

    monkeypatch.setattr(cli_generation, "_dispatch", fail_dispatch)

    assert cli_generation.main(["list-campaigns"]) == 2
    error = json.loads(capsys.readouterr().err)
    assert error == {
        "command": "list-campaigns",
        "context": {},
        "error_type": "KeyError",
        "message": "'T_flow_ref'",
        "stage": "list-campaigns",
        "status": "error",
    }


def test_latin_hypercube_membership_is_not_a_prefix_contract() -> None:
    """Protect case identity assignment without assuming SciPy prefix stability."""
    sampler = generation.cases.sampling
    first = sampler._lhs_design(count=160, dimensions=3, seed=17)
    second = sampler._lhs_design(count=50, dimensions=3, seed=17)
    assert first.shape == (160, first.shape[1])
    assert second.shape == (50, second.shape[1])
    assert not (first[:50] == second).all()


def test_campaign_requires_complete_persisted_input_sources(
    generation_config_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require complete source provenance before campaign input admission."""
    config_path, _template = generation_config_factory(natural_count=1)
    campaign = generation.cases.config.load_campaign_config(config_path)
    batch = campaign.batches[0]
    source_id = "input-" + "1" * 24
    manifest = {
        "batches": [
            {
                "batch_name": batch.batch_name,
                "input_generation_id": source_id,
                "input_source_git_commit": _FAKE_GIT_COMMIT,
                "input_case_indices": list(batch.case_indices),
            }
        ]
    }
    selected = generation.campaign._persisted_input_sources(manifest)
    assert selected == {
        batch.batch_name: {
            "input_generation_id": source_id,
            "source_git_commit": _FAKE_GIT_COMMIT,
            "case_indices": list(batch.case_indices),
        }
    }
    with pytest.raises(RuntimeError, match="incomplete or corrupt"):
        generation.campaign._persisted_input_sources({"batches": [{"batch_name": batch.batch_name}]})
    with pytest.raises(RuntimeError, match="incomplete or corrupt"):
        generation.campaign._persisted_input_sources(
            {
                "batches": [
                    {
                        "batch_name": batch.batch_name,
                        "input_generation_id": source_id,
                    }
                ]
            }
        )

    calls: list[dict[str, Any]] = []

    def admit(_batch: Any, **kwargs: Any) -> dict[int, Any]:
        calls.append(dict(kwargs))
        return {}

    monkeypatch.setattr(generation.campaign.input_service, "admit_configured_input_references", admit)
    generation.campaign._campaign_input_references(
        campaign,
        git_commit=_OTHER_GIT_COMMIT,
        storage_root=None,
        input_sources=selected,
    )
    assert calls == [
        {
            "storage_root": None,
            "validation_depth": "evidence",
            "git_commit": _OTHER_GIT_COMMIT,
            "input_generation_id": source_id,
            "input_source_git_commit": _FAKE_GIT_COMMIT,
        }
    ]


def test_campaign_rejects_persisted_input_case_membership_drift(
    generation_config_factory: Any,
) -> None:
    """Bind persisted input membership to the exact active campaign batch."""
    config_path, _template = generation_config_factory(natural_count=2)
    campaign = generation.cases.config.load_campaign_config(config_path)
    batch = campaign.batches[0]
    selected = {
        batch.batch_name: {
            "input_generation_id": "input-" + "1" * 24,
            "source_git_commit": _FAKE_GIT_COMMIT,
            "case_indices": list(reversed(batch.case_indices)),
        }
    }

    with pytest.raises(RuntimeError, match="case membership disagrees"):
        generation.campaign._campaign_input_references(
            campaign,
            git_commit=_OTHER_GIT_COMMIT,
            storage_root=None,
            input_sources=selected,
        )
