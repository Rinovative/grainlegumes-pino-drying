"""Canonical input-generation storage, admission, and identity contracts."""

# ruff: noqa: PLR2004, S101, SLF001

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest

from src import common, generation
from src.generation.cli import cli_generation
from src.generation.runtime import generation_runtime_preparation as runtime_preparation

_FAKE_GIT_COMMIT = "a" * 40
_OTHER_GIT_COMMIT = "b" * 40


def _load_config(
    generation_config_factory: Any,
    profile_id: str,
    *,
    campaign_purpose: str = "technical_runtime_smoke",
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


def test_batch_storage_name_is_flat_explicit_and_separate_from_identity() -> None:
    """Build one safe ordered locator without changing scientific identity."""
    digest = "a" * 64
    batch_name = generation.cases.config.build_batch_name(
        "transient_drying",
        "kidney_bean",
        "natural",
    )
    batch_id = generation.cases.config.build_batch_id(batch_name, digest)
    family = generation.cases.config.build_batch_storage_name(
        "transient_drying",
        "kidney_bean",
        "natural",
        "family_generalization",
        digest,
    )
    technical = generation.cases.config.build_batch_storage_name(
        "transient_drying",
        "kidney_bean",
        "natural",
        "technical_runtime_smoke",
        digest,
    )
    assert batch_id == f"transient_drying__kidney_bean__natural__{'a' * 16}"
    assert family == (f"transient_drying__kidney_bean__natural__family_generalization__{'a' * 16}")
    assert technical == (f"transient_drying__kidney_bean__natural__technical_runtime_smoke__{'a' * 16}")
    assert family == generation.cases.config.build_batch_storage_name(
        "transient_drying",
        "kidney_bean",
        "natural",
        "family_generalization",
        digest,
    )
    assert family.count("family_generalization") == 1
    for invalid in (
        "Family_Generalization",
        "family-generalization",
        "family__generalization",
        "../family_generalization",
    ):
        with pytest.raises(ValueError, match="campaign_purpose"):
            generation.cases.config.build_batch_storage_name(
                "transient_drying",
                "kidney_bean",
                "natural",
                invalid,
                digest,
            )


def test_generation_stages_share_one_flat_batch_storage_leaf(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """Resolve raw, meta, and processed directly under their stage roots."""
    config = _load_config(generation_config_factory, "transient_drying")
    storage = tmp_path / "storage"
    paths = (
        common.paths.resolve_generation_input_raw_batch_directory(
            config.batch_storage_name,
            storage_root=storage,
        ),
        common.paths.resolve_generation_input_metadata_directory(
            config.batch_storage_name,
            storage_root=storage,
        ),
        common.paths.resolve_generated_batch_dir(
            config.batch_storage_name,
            stage="processed",
            storage_root=storage,
        ),
    )

    assert tuple(path.name for path in paths) == (config.batch_storage_name,) * 3
    assert tuple(path.parent.name for path in paths) == ("raw", "meta", "processed")


@pytest.mark.parametrize("profile_id", ["steady_flow", "transient_drying"])
def test_input_cases_use_canonical_batch_storage_and_never_execute(
    generation_config_factory: Any,
    tmp_path: Path,
    profile_id: str,
) -> None:
    """Generate genuine production inputs without external processes or success state."""
    config = _load_config(generation_config_factory, profile_id)
    storage = tmp_path / "storage"
    blocked = AssertionError("Input generation attempted to start an external process.")
    with (
        patch("subprocess.Popen", side_effect=blocked),
        patch("subprocess.run", side_effect=blocked),
    ):
        generated = generation.cases.input_generation.generate_input_cases(
            config,
            1,
            storage_root=storage,
        )

    manifest_path = generated.metadata_directory / "input_generation_manifest.json"
    resolved_path = generated.metadata_directory / "resolved_generation_config.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    assert generated.metadata_directory == common.paths.resolve_generation_input_metadata_directory(
        config.batch_storage_name,
        storage_root=storage,
    )
    assert generated.raw_directory == common.paths.resolve_generation_input_raw_batch_directory(
        config.batch_storage_name,
        storage_root=storage,
    )
    assert manifest["schema_kind"] == "generation_input_batch"
    assert manifest["schema_version"] == 1
    assert manifest["status"] == "ready"
    assert "execution_status" not in manifest
    assert "executed" not in manifest
    assert manifest["simulation_profile"] == profile_id
    assert manifest["batch_id"] == config.batch_id
    assert manifest["batch_storage_name"] == config.batch_storage_name
    assert manifest["campaign_purpose"] == "technical_runtime_smoke"
    assert generated.raw_directory.name == config.batch_storage_name
    assert config.batch_storage_name == (f"{profile_id}__lentil__natural__technical_runtime_smoke__{config.scientific_config_digest[:16]}")
    assert manifest["case_indices"] == [1]
    assert manifest["resolved_config_sha256"] == (common.serialization.canonical_json_sha256(resolved))
    assert generated.generated_case_count == 1
    assert generated.reused_case_count == 0
    assert {entry.name for entry in generated.metadata_directory.iterdir()} == {
        "execution_configs",
        manifest_path.name,
        resolved_path.name,
    }
    case_directory = generated.raw_directory / config.case_id(1)
    assert {entry.name for entry in generated.raw_directory.iterdir()} == {config.case_id(1)}
    assert {entry.name for entry in case_directory.iterdir()} == {"case.json", "inputs"}
    prohibited = {
        "_SUCCESS",
        "case.h5",
        "execution_provenance.json",
        "model.mph",
        "solved.mph",
        "solver.log",
        "status.json",
        "timing.json",
    }
    assert not any(path.name in prohibited for path in generated.raw_directory.rglob("*"))

    discovery = generation.cases.admission.discover_input_batches(storage)
    assert not discovery.issues
    assert len(discovery.sources) == 1
    source = discovery.sources[0]
    assert source.source_id == generated.input_generation_id
    assert source.source_kind == "input_generated"
    admitted = generation.cases.admission.admit_input_case_reference(source.cases[0])
    assert admitted.profile_id == profile_id
    assert (admitted.schedule is None) is (profile_id == "steady_flow")
    assert (not admitted.scalars) is (profile_id == "steady_flow")
    metadata_name = next(iter(resolved["registry_metadata"]))
    assert admitted.parameter_metadata[metadata_name] == resolved["registry_metadata"][metadata_name]
    with pytest.raises(TypeError):
        admitted.payload["stationary_fixed_values"][0]["value"] = 0.0
    with pytest.raises(TypeError):
        cast("dict[str, str]", admitted.parameter_metadata[metadata_name])["description"] = "changed"


def test_bounded_requests_merge_membership_and_reuse_exact_cases(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """Maintain one source per batch while overlapping bounded requests merge."""
    config = _load_config(generation_config_factory, "transient_drying")
    storage = tmp_path / "storage"
    first = generation.cases.input_generation.generate_input_cases(
        config,
        2,
        case_start=1,
        storage_root=storage,
    )
    overlap = generation.cases.input_generation.generate_input_cases(
        config,
        3,
        case_start=2,
        storage_root=storage,
    )
    blocked = AssertionError("Exact persisted input reuse regenerated a case.")
    with patch(
        "src.generation.cases.generation_cases_input.case_service.generate_case_input_bundle",
        side_effect=blocked,
    ):
        repeated = generation.cases.input_generation.generate_input_cases(
            config,
            2,
            case_start=1,
            storage_root=storage,
        )

    assert first.input_generation_id == overlap.input_generation_id
    assert overlap.input_generation_id == repeated.input_generation_id
    assert first.generated_case_count == 2
    assert overlap.generated_case_count == 2
    assert overlap.reused_case_count == 1
    assert overlap.case_indices == (1, 2, 3, 4)
    assert repeated.generated_case_count == 0
    assert repeated.reused_case_count == 2
    assert repeated.case_indices == (1, 2, 3, 4)
    discovery = generation.cases.admission.discover_input_batches(storage)
    assert len(discovery.sources) == 1
    assert [case.case_index for case in discovery.sources[0].cases] == [1, 2, 3, 4]


def test_journaled_input_publication_recovers_after_case_move(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """Complete an interrupted raw-case move from its precommitted transaction."""
    service = generation.cases.input_generation
    config = _load_config(generation_config_factory, "steady_flow")
    storage = tmp_path / "storage"
    first = service.generate_input_cases(config, 1, storage_root=storage)
    resolved = service._resolved_config(config)
    base = service._manifest_base(config, resolved)
    manifest = json.loads((first.metadata_directory / "input_generation_manifest.json").read_text(encoding="utf-8"))
    records = {int(record["case_index"]): record for record in manifest["cases"]}
    transaction = common.paths.resolve_generation_input_transaction_directory(
        str(base["input_generation_id"]),
        storage_root=storage,
    )
    staged_case = transaction / "raw" / config.batch_storage_name / config.case_id(2)
    service.case_service.generate_case_input_bundle(config, 2, staged_case)
    records[2] = service._case_record(staged_case)
    candidate = service._complete_manifest(base, records)
    staged_metadata = transaction / "meta" / config.batch_storage_name
    service._write_staged_metadata(
        staged_metadata,
        resolved_config=resolved,
        manifest=candidate,
    )
    service._write_input_transaction_journal(
        transaction,
        batch_id=config.batch_id,
        batch_storage_name=config.batch_storage_name,
        input_generation_id=str(base["input_generation_id"]),
        manifest_path=staged_metadata / "input_generation_manifest.json",
        new_case_ids=(config.case_id(2),),
    )
    target = first.raw_directory / config.case_id(2)
    staged_case.replace(target)

    blocked = AssertionError("Crash recovery regenerated a journaled case.")
    with patch.object(
        service.case_service,
        "generate_case_input_bundle",
        side_effect=blocked,
    ):
        recovered = service.generate_input_cases(
            config,
            1,
            case_start=2,
            storage_root=storage,
        )

    assert recovered.generated_case_count == 0
    assert recovered.reused_case_count == 1
    assert recovered.case_indices == (1, 2)
    assert not transaction.exists()
    source = generation.cases.admission.admit_input_batch_source(first.metadata_directory)
    assert [reference.case_index for reference in source.cases] == [1, 2]


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
    arbitrary = common.paths.get_generation_raw_root(storage_root=storage)
    (arbitrary / "unmanifested_source" / "case_0001").mkdir(parents=True)

    discovery = generation.cases.admission.discover_input_batches(storage)

    assert discovery.sources == ()
    assert discovery.issues == ()


@pytest.mark.parametrize(
    ("corruption", "error_match"),
    [
        ("missing_purpose", "manifest schema"),
        ("contradictory_purpose", "configuration does not bind"),
        ("duplicated_purpose_leaf", "identities do not bind"),
    ],
)
def test_input_admission_rejects_incomplete_or_false_storage_provenance(
    generation_config_factory: Any,
    tmp_path: Path,
    corruption: str,
    error_match: str,
) -> None:
    """Reject missing, contradictory, or duplicated purpose provenance."""
    config = _load_config(generation_config_factory, "transient_drying")
    storage = tmp_path / "storage"
    generated = generation.cases.input_generation.generate_input_cases(
        config,
        1,
        storage_root=storage,
    )
    metadata_directory = generated.metadata_directory
    raw_directory = generated.raw_directory
    manifest_path = metadata_directory / "input_generation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if corruption == "missing_purpose":
        manifest.pop("campaign_purpose")
    elif corruption == "contradictory_purpose":
        manifest["campaign_purpose"] = "family_generalization"
        manifest["input_generation_id"] = generation.cases.admission.compute_input_generation_id(manifest)
    else:
        false_name = config.batch_storage_name.replace(
            "__technical_runtime_smoke__",
            "__technical_runtime_smoke__technical_runtime_smoke__",
        )
        metadata_directory = metadata_directory.rename(metadata_directory.with_name(false_name))
        raw_directory.rename(raw_directory.with_name(false_name))
        manifest_path = metadata_directory / manifest_path.name
        manifest["batch_storage_name"] = false_name
    common.serialization.atomic_write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match=error_match):
        generation.cases.admission.admit_input_batch(metadata_directory)


def test_adapter_corruption_and_execution_artifacts_fail_closed(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """Reject changed adapter bytes and any evidence of execution."""
    config = _load_config(generation_config_factory, "transient_drying")
    storage = tmp_path / "storage"
    generated = generation.cases.input_generation.generate_input_cases(
        config,
        1,
        storage_root=storage,
    )
    schedule_path = generated.raw_directory / config.case_id(1) / "inputs" / "schedule.csv"
    original = schedule_path.read_bytes()
    schedule_path.write_bytes(original + b"\n")
    corrupted = generation.cases.admission.discover_input_batches(storage)
    assert not corrupted.sources
    assert len(corrupted.issues) == 1
    assert "adapter hash or size" in corrupted.issues[0].message
    with pytest.raises(FileExistsError, match="incomplete or invalid"):
        generation.cases.input_generation.generate_input_cases(
            config,
            1,
            storage_root=storage,
        )

    schedule_path.write_bytes(original)
    artifact = generated.raw_directory / config.case_id(1) / "_SUCCESS"
    artifact.write_text("invalid execution evidence\n", encoding="utf-8")
    with pytest.raises(ValueError, match="execution or success artifact"):
        generation.cases.admission.admit_input_batch(generated.metadata_directory)


def test_orphaned_raw_batch_fails_generation_but_remains_undiscoverable(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """Never reuse raw input cases without their final metadata boundary."""
    config = _load_config(generation_config_factory, "steady_flow")
    storage = tmp_path / "storage"
    generated = generation.cases.input_generation.generate_input_cases(
        config,
        1,
        storage_root=storage,
    )
    shutil.rmtree(generated.metadata_directory)

    with pytest.raises(FileExistsError, match="must exist together"):
        generation.cases.input_generation.generate_input_cases(
            config,
            1,
            storage_root=storage,
        )
    assert not generation.cases.admission.discover_input_batches(storage).sources


def test_metadata_publication_failure_preserves_journal_and_recovers(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain post-move evidence and complete it on the next invocation."""
    config = _load_config(generation_config_factory, "steady_flow")
    storage = tmp_path / "storage"
    metadata_directory = common.paths.resolve_generation_input_metadata_directory(
        config.batch_storage_name,
        storage_root=storage,
    )
    raw_directory = common.paths.resolve_generation_input_raw_batch_directory(
        config.batch_storage_name,
        storage_root=storage,
    )
    manifest_path = metadata_directory / "input_generation_manifest.json"
    original_atomic_write_json = common.serialization.atomic_write_json

    class ForcedMetadataPublicationError(OSError):
        """Identify the intentionally injected final publication failure."""

    def write_with_manifest_failure(path: Any, payload: Any) -> Any:
        """Fail only the final canonical input-manifest publication."""
        if Path(path) == manifest_path:
            raise ForcedMetadataPublicationError
        return original_atomic_write_json(path, payload)

    with monkeypatch.context() as scoped:
        scoped.setattr(
            common.serialization,
            "atomic_write_json",
            write_with_manifest_failure,
        )
        with pytest.raises(ForcedMetadataPublicationError):
            generation.cases.input_generation.generate_input_cases(
                config,
                1,
                storage_root=storage,
            )

    transactions = tuple(common.paths.get_generation_input_transactions_root(storage_root=storage).iterdir())
    assert not manifest_path.exists()
    assert (raw_directory / config.case_id(1)).is_dir()
    assert len(transactions) == 1
    assert (transactions[0] / "transaction.json").is_file()
    assert (metadata_directory / "resolved_generation_config.json").is_file()

    blocked = AssertionError("Journal recovery regenerated a published case.")
    with patch.object(
        generation.cases.input_generation.case_service,
        "generate_case_input_bundle",
        side_effect=blocked,
    ):
        recovered = generation.cases.input_generation.generate_input_cases(
            config,
            1,
            storage_root=storage,
        )
    assert recovered.generated_case_count == 0
    assert recovered.reused_case_count == 1
    assert manifest_path.is_file()
    assert not transactions[0].exists()


@pytest.mark.parametrize(
    ("profile_id", "campaign_purpose"),
    [
        ("steady_flow", "family_generalization"),
        ("transient_drying", "family_generalization"),
        ("transient_drying", "technical_runtime_smoke"),
    ],
)
def test_input_generation_is_exactly_equivalent_to_runtime_preparation(
    generation_config_factory: Any,
    tmp_path: Path,
    profile_id: str,
    campaign_purpose: str,
) -> None:
    """Prove exact case identity and adapter bytes across independent owners."""
    config = _load_config(
        generation_config_factory,
        profile_id,
        campaign_purpose=campaign_purpose,
    )
    storage = tmp_path / "storage"
    generated = generation.cases.input_generation.generate_input_cases(
        config,
        1,
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

    equivalence = generation.cases.admission.assert_case_bundle_equivalent(
        generated.raw_directory / config.case_id(1),
        prepared.work_directory,
    )

    assert equivalence.case_id == config.case_id(1)
    assert equivalence.case_input_id == prepared.bundle.case_input_id
    assert equivalence.simulation_case_id == prepared.bundle.simulation_case_id
    assert equivalence.git_commit == _FAKE_GIT_COMMIT
    assert equivalence.input_files


def test_bundle_equivalence_rejects_changed_git_source_identity(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require exact source provenance even when scientific IDs stay stable."""
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
    with pytest.raises(ValueError, match=r"differing case\.json fields"):
        generation.cases.admission.assert_case_bundle_equivalent(
            first.directory,
            second.directory,
        )


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


def test_campaign_input_generation_actions_remain_thin_and_non_mutating(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """Default to dry-run and delegate execute through one canonical service."""
    service = generation.cases.input_generation
    assert (
        service.CampaignInputGenerationRequest(
            campaign_config=tmp_path / "unused.yaml",
            storage_root=tmp_path / "unused",
        ).action
        == "dry_run"
    )
    with pytest.raises(ValueError, match="Unsupported input-generation action"):
        service.run_campaign_input_generation(
            service.CampaignInputGenerationRequest(
                campaign_config=tmp_path / "unused.yaml",
                storage_root=tmp_path / "unused",
                action="skip",
            )
        )
    config_path, _template = generation_config_factory(
        simulation_profile="transient_drying",
        natural_count=3,
        campaign_purpose="family_generalization",
    )
    batch_name = generation.cases.config.build_batch_name(
        "transient_drying",
        "lentil",
        "natural",
    )
    storage = tmp_path / "storage"
    base = {
        "campaign_config": config_path,
        "storage_root": storage,
        "only_batch": batch_name,
        "case_start": 1,
        "case_count": 1,
        "git_commit": _FAKE_GIT_COMMIT,
    }
    with patch.object(
        service,
        "generate_input_cases",
        side_effect=AssertionError("dry_run generated canonical input"),
    ):
        planned = service.run_campaign_input_generation(service.CampaignInputGenerationRequest(action="dry_run", **base))
    assert planned["dry_run"] is True
    assert planned["would_generate_case_count"] == 1
    assert planned["generated_case_count"] == 0
    assert planned["batch_identity"]
    assert planned["simulation_profile"] == "transient_drying"
    assert planned["campaign_purpose"] == "family_generalization"
    assert planned["equivalent_cli_command"].endswith(f"--storage-root {storage}")
    assert not storage.exists()

    with patch.object(
        service,
        "generate_input_cases",
        wraps=service.generate_input_cases,
    ) as generate:
        executed = service.run_campaign_input_generation(service.CampaignInputGenerationRequest(action="execute", **base))
    assert generate.call_count == 1
    assert executed["generated_case_count"] == 1
    assert executed["reused_case_count"] == 0
    assert Path(executed["raw_case_paths"][0]).is_dir()


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
        campaign_purpose="technical_runtime_smoke",
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
        campaign_purpose="technical_runtime_smoke",
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


def test_generate_input_cases_cli_reports_bounded_technical_smoke(
    generation_config_factory: Any,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Expose one truthful final command with bounded technical-smoke output."""
    config_path, _template = generation_config_factory(
        simulation_profile="steady_flow",
        natural_count=2,
        campaign_purpose="technical_runtime_smoke",
    )
    batch_name = generation.cases.config.build_batch_name(
        "steady_flow",
        "lentil",
        "natural",
    )
    storage = tmp_path / "storage"
    arguments = [
        "generate-input-cases",
        str(config_path),
        "--only-batch",
        batch_name,
        "--case-start",
        "2",
        "--case-count",
        "1",
        "--git-commit",
        _FAKE_GIT_COMMIT,
        "--storage-root",
        str(storage),
    ]

    assert cli_generation.main(arguments) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["campaign_purpose"] == "technical_runtime_smoke"
    assert result["requested_case_range"] == [2, 2]
    assert result["generated_case_count"] == 1
    assert result["reused_case_count"] == 0
    assert result["input_only"] is True
    assert result["execution_status"] == "not_executed"

    overflow = [
        *arguments[:],
    ]
    overflow[overflow.index("--case-count") + 1] = "2"
    assert cli_generation.main(overflow) == 2
    assert "exceeds configured batch membership" in capsys.readouterr().err
