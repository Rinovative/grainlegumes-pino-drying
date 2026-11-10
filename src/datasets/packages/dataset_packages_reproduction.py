"""Capture and inspect the recoverable dependencies of an immutable Dataset.

Responsibilities:
  - Bind new Dataset publications to retained producing source and environment evidence
  - Reuse the admitted, tree-verified snapshot as producing source when running from a frozen source tree
  - Present Source parameters, seeds, model and regeneration boundaries in one inspection
  - Declare the conversion code projection of each Dataset view and bind the converter implementation digest to it;
    pure runtime readers belong to the model projection, not to conversion

Design principles:
  - Existing source snapshots retain code bytes; the Dataset manifest owns provenance
  - A frozen source tree never captures its own snapshot and never calls Git
  - Availability and hash verification are separate from scientific reproduction

This module does NOT:
  - Run COMSOL, regenerate Sources or mutate evidence during inspection
  - Claim environment or solver recovery from version strings alone
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import platform
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

from src import common
from src.generation.contracts import generation_contracts_source as source_contract

_PRODUCER_KEYS = frozenset({"source_snapshot", "environment"})
_SNAPSHOT_KEYS = frozenset({"relative_path", "sha256", "git_commit", "source_fingerprint_sha256"})
_CONVERSION_FILES = (
    "common/common_code_identity.py",
    "common/common_serialization.py",
    "datasets/contracts/dataset_contracts_identity.py",
    "datasets/contracts/dataset_contracts_partition.py",
    "datasets/contracts/dataset_contracts_views.py",
    "datasets/packages/dataset_packages_builder.py",
    "datasets/packages/dataset_packages_generated_batch.py",
    "datasets/packages/dataset_packages_manifest.py",
    "datasets/packages/dataset_packages_planning.py",
    "domain/domain_field_sets.py",
    "domain/domain_fields.py",
    "domain/tasks/domain_task_registry.py",
    "domain/tasks/domain_task_spec.py",
    "generation/contracts/generation_contracts_descriptors.py",
    "generation/contracts/generation_contracts_materials.py",
    "generation/contracts/generation_contracts_porosity.py",
    "generation/contracts/generation_contracts_profiles.py",
    "generation/contracts/generation_contracts_registry.py",
    "generation/contracts/generation_contracts_source.py",
    "generation/contracts/generation_contracts_vocabulary.py",
)
_CONVERSION_EXEMPTIONS = {
    "datasets/packages/dataset_packages_selection.py": "Dataset membership is bound by the Dataset content identity",
    "datasets/packages/dataset_packages_reproduction.py": "owner and provenance tooling around the projection, not conversion",
    "generation/cases/generation_cases_config.py": "Generation config reaches Datasets only through Source content",
}
DATASET_CONVERSION_PROJECTIONS = {
    "steady_flow": common.code_identity.CodeProjection(
        name="dataset_conversion_steady_flow",
        files=(
            *_CONVERSION_FILES,
            "datasets/contracts/dataset_contracts_transient.py",
            "domain/physics/domain_physics_contracts.py",
            "domain/tasks/domain_task_steady_flow.py",
        ),
        excluded_markers=("transient",),
        exemptions={
            **_CONVERSION_EXEMPTIONS,
            "datasets/packages/dataset_packages_trajectory.py": "transient trajectory conversion, reached through the package view dispatch",
        },
    ),
    "transient_drying": common.code_identity.CodeProjection(
        name="dataset_conversion_transient_drying",
        files=(
            *_CONVERSION_FILES,
            "datasets/contracts/dataset_contracts_transient.py",
            "datasets/packages/dataset_packages_trajectory.py",
            "datasets/packages/dataset_packages_transient_shards.py",
            "domain/tasks/domain_task_transient_drying.py",
        ),
        excluded_markers=("steady",),
        exemptions=_CONVERSION_EXEMPTIONS,
    ),
}


def validate_producer_evidence(value: Any) -> None:
    """Validate manifest-owned producing source and environment evidence."""
    if not isinstance(value, dict) or set(value) != _PRODUCER_KEYS:
        message = "Dataset producer evidence has invalid keys."
        raise ValueError(message)
    snapshot = value["source_snapshot"]
    if not isinstance(snapshot, dict) or set(snapshot) != _SNAPSHOT_KEYS:
        message = "Dataset producer source snapshot is invalid."
        raise ValueError(message)
    for name in ("sha256", "source_fingerprint_sha256"):
        if not isinstance(snapshot[name], str) or re.fullmatch(r"[0-9a-f]{64}", snapshot[name]) is None:
            message = f"Dataset source snapshot {name} is invalid."
            raise ValueError(message)
    if not isinstance(snapshot["git_commit"], str) or re.fullmatch(r"[0-9a-f]{40}", snapshot["git_commit"]) is None:
        message = "Dataset producing Git commit is invalid."
        raise ValueError(message)
    if not isinstance(snapshot["relative_path"], str):
        message = "Dataset source snapshot locator must be a relative string."
        raise TypeError(message)
    path = Path(snapshot["relative_path"])
    if path.is_absolute() or ".." in path.parts or path.parts[:1] != (common.paths.SOURCE_SNAPSHOTS_DIRNAME,) or path.name != "snapshot.json":
        message = "Dataset producing source snapshot locator is unsafe."
        raise ValueError(message)
    environment = value["environment"]
    if not isinstance(environment, dict) or set(environment) != {"python", "platform", "packages"}:
        message = "Dataset environment evidence is invalid."
        raise ValueError(message)
    if not all(isinstance(environment[key], str) and environment[key] for key in ("python", "platform")):
        message = "Dataset environment version evidence is invalid."
        raise ValueError(message)
    if not isinstance(environment["packages"], dict) or any(
        not isinstance(key, str) or not isinstance(version, str) for key, version in environment["packages"].items()
    ):
        message = "Dataset dependency version evidence is invalid."
        raise ValueError(message)


def conversion_implementation_sha256(dataset_view: str) -> str:
    """Identify the selected conversion and byte writer of the executing source.

    The shared package implementations own Source admission, ordered payloads,
    identity and file serialization. View-specific implementations own channel
    and tensor interpretation. Exact producing bytes remain in source snapshots;
    this syntax digest excludes comments/docstrings and does not prove equivalence.
    """
    projection = DATASET_CONVERSION_PROJECTIONS.get(dataset_view)
    if projection is None:
        message = f"Unsupported Dataset implementation view: {dataset_view!r}."
        raise ValueError(message)
    return projection.sha256()


def _admitted_tree_receipt(storage: Path) -> dict[str, str]:
    """Return the snapshot receipt of the executing source tree after verifying the tree against it."""
    admitted = source_contract.verify_admitted_source_tree()
    if not admitted.snapshot_path.is_relative_to(storage):
        message = f"The admitted source snapshot lies outside the Dataset storage root {storage}: {admitted.snapshot_path}"
        raise ValueError(message)
    return {
        "snapshot_path": str(admitted.snapshot_path),
        "snapshot_sha256": admitted.snapshot_sha256,
        "fingerprint": admitted.source_fingerprint_sha256,
    }


def _captured_receipt(storage: Path) -> dict[str, str]:
    """Capture the live producing source using the maintained scheduler snapshot mechanism."""
    repository = common.paths.get_project_root()
    script = repository / "scripts/source_fingerprint.py"
    spec = importlib.util.spec_from_file_location("dataset_source_snapshot", script)
    if spec is None or spec.loader is None:
        message = "The maintained source snapshot implementation is unavailable."
        raise RuntimeError(message)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.capture_source_snapshot(repository, common.paths.get_source_snapshots_root(storage_root=storage))


def capture_producer_evidence(*, storage_root: Path | str | None = None) -> dict[str, Any]:
    """Bind the producing source: the admitted snapshot inside a source tree, otherwise a fresh live capture."""
    storage = common.paths.get_storage_root(storage_root=storage_root).resolve()
    receipt = _admitted_tree_receipt(storage) if common.paths.source_tree_id() is not None else _captured_receipt(storage)
    snapshot_path = Path(receipt["snapshot_path"])
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    value = {
        "source_snapshot": {
            "relative_path": snapshot_path.relative_to(storage).as_posix(),
            "sha256": receipt["snapshot_sha256"],
            "git_commit": snapshot["git_commit"],
            "source_fingerprint_sha256": receipt["fingerprint"],
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "h5py", "torch", "PyYAML")},
        },
    }
    validate_producer_evidence(value)
    return value


def _contained_file(storage: Path, relative: str) -> Path:
    path = Path(relative)
    target = storage / path
    if path.is_absolute() or ".." in path.parts or not target.resolve().is_relative_to(storage) or target.is_symlink():
        message = f"Dataset reproduction evidence escapes storage: {relative!r}."
        raise ValueError(message)
    return target


def _read_record(path: Path) -> dict[str, Any] | None:
    if not path.is_file() or path.is_symlink():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        message = f"Dataset reproduction record is not a mapping: {path}."
        raise TypeError(message)
    return value


def _snapshot_report(storage: Path, evidence: dict[str, Any]) -> dict[str, Any]:
    path = _contained_file(storage, evidence["relative_path"])
    record = _read_record(path)
    if record is None:
        return {"manifest": str(path), "available": False, "verified": False}
    if record.get("git_commit") != evidence["git_commit"] or record.get("source_fingerprint_sha256") != evidence["source_fingerprint_sha256"]:
        message = "Producing source snapshot disagrees with Dataset provenance."
        raise ValueError(message)
    verified = common.serialization.file_sha256(path) == evidence["sha256"]
    if not verified:
        message = f"Producing source snapshot manifest changed: {path}."
        raise ValueError(message)
    diff = path.parent / "tracked.diff"
    diff_verified = diff.is_file() and not diff.is_symlink() and common.serialization.file_sha256(diff) == record.get("tracked_diff_sha256")
    missing = []
    for entry in record.get("untracked", []):
        saved = (
            _contained_file(storage, (path.parent.relative_to(storage) / "untracked" / entry["path"]).as_posix())
            if entry.get("kind") == "file"
            else path.parent / "untracked" / entry["path"]
        )
        if entry.get("kind") == "file":
            if not saved.is_file() or common.serialization.file_sha256(saved) != entry.get("sha256"):
                missing.append(entry["path"])
        elif entry.get("kind") == "symlink":
            if not saved.is_symlink() or str(saved.readlink()) != entry.get("target"):
                missing.append(entry["path"])
        else:
            missing.append(entry.get("path", "invalid inventory entry"))
    base = subprocess.run(  # noqa: S603 -- fixed read-only Git operation with validated object ID
        ["git", "-C", str(common.paths.get_project_root()), "cat-file", "-t", record["git_commit"]],  # noqa: S607 -- maintained site Git executable
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "manifest": str(path),
        "available": True,
        "manifest_verified": verified,
        "tracked_diff_verified": diff_verified,
        "missing_or_changed_untracked": missing,
        "base_commit_available": base.returncode == 0 and base.stdout.strip() == "commit",
        "full_modes_recoverable": record.get("full_modes_recoverable") is True,
        "restore_command": f"python scripts/source_fingerprint.py . --restore {path} EMPTY_ISOLATED_DIRECTORY",
    }


def _generation_invocation(run: dict[str, Any] | None, plan: dict[str, Any] | None) -> dict[str, Any]:
    if not run or not plan or not isinstance(run.get("campaign_config"), str):
        return {"reconstructible": False, "reason": "Retained campaign configuration invocation is unavailable."}
    overrides = plan.get("scientific_overrides")
    if not isinstance(overrides, dict):
        return {"reconstructible": False, "reason": "Effective scientific overrides are not retained."}
    if plan.get("authored_value") is not None:
        return {
            "reconstructible": False,
            "reason": (
                "Source originated from a Dataset plan; use this Dataset manifest's plan_invocation, not the Generation CLI against Dataset YAML."
            ),
        }
    command = ["./scripts/generation", "run", run["campaign_config"]]
    for key, flag in (("duration_hours", "--duration-hours"), ("revision", "--revision")):
        if overrides.get(key) is not None:
            command.extend((flag, str(overrides[key])))
    if overrides.get("materials"):
        command.extend(("--materials", *overrides["materials"]))
    execution = plan.get("execution_overrides")
    execution_flags = (
        ("node_jobs", "--node-jobs"),
        ("max_admission_cases", "--max-admission-cases"),
        ("cores_per_case", "--cores-per-case"),
        ("case_timeout_seconds", "--case-timeout-seconds"),
    )
    execution_command = list(command)
    if isinstance(execution, dict):
        for key, flag in execution_flags:
            if execution.get(key) is not None:
                execution_command.extend((flag, str(execution[key])))
        if execution.get("no_retry") is True:
            execution_command.append("--no-retry")
    return {
        "reconstructible": True,
        "scientific_command": shlex.join(command),
        "recorded_execution_command": shlex.join(execution_command) if isinstance(execution, dict) else None,
        "execution_overrides": execution if isinstance(execution, dict) else None,
        "execution_choices_not_recorded": [key for key, _flag in execution_flags if not isinstance(execution, dict) or key not in execution]
        + ([] if isinstance(execution, dict) and "no_retry" in execution else ["no_retry"]),
        "execution_boundary": (
            "Run only from an admitted Git/source environment through Slurm with COMSOL authorization. "
            "Only explicitly recorded resource overrides are emitted; absent choices retain configuration defaults, "
            "whose invocation overrides are unknown/not recorded. This is not automatic solver replay."
        ),
    }


def _template_report(value: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(value.get("relative_path"), str):
        return {"recorded": False, "current_bytes_available": False}
    relative = Path(value["relative_path"])
    root = common.paths.get_project_root().resolve()
    path = root / relative
    safe = not relative.is_absolute() and ".." not in relative.parts and path.resolve().is_relative_to(root) and not path.is_symlink()
    available = safe and path.is_file()
    return {
        **value,
        "current_path": str(path),
        "current_bytes_available": available,
        "current_bytes_match_recorded_hash": common.serialization.file_sha256(path) == value.get("sha256") if available else None,
        "recovery_boundary": (
            "Exact model bytes are part of the producing source snapshot inventory; base Git objects and dirty snapshot bytes must remain available."
        ),
    }


def inspect_dataset_reproduction(task: str, dataset_id: str, *, storage_root: Path | str | None = None) -> dict[str, Any]:
    """Inspect Dataset dependencies without tensor loading or scientific execution."""
    from . import dataset_packages_manifest as manifests  # noqa: PLC0415 -- avoid producer-validation import cycle

    storage = common.paths.get_storage_root(storage_root=storage_root).resolve()
    manifest = manifests.load_package_manifest_evidence(task, dataset_id, storage_root=storage)
    package = common.paths.resolve_dataset_dir(task, dataset_id, dataset_root=common.paths.get_datasets_root(storage_root=storage))
    cases = []
    campaigns: dict[str, Any] = {}
    for entry in manifest["source_case_identities"]:
        source = entry["source"]
        hdf5 = _contained_file(storage, source["case_hdf5_relative"])
        location = common.paths.locate_generation_case_result_file(source["case_hdf5_relative"], common.paths.GENERATION_RESULT_HDF5_FILENAME)
        case_root = common.paths.resolve_generation_campaign_case_directory(
            location.task, location.campaign_name, location.material, location.case_id, storage_root=storage
        )
        raw_record = common.paths.resolve_generation_case_inputs_directory(case_root) / common.paths.GENERATION_CASE_PAYLOAD_FILENAME
        raw = _read_record(raw_record)
        execution = _read_record(common.paths.resolve_generation_result_evidence_directory(hdf5.parent) / "execution_provenance.json")
        campaign_root = common.paths.resolve_generation_campaign_directory(location.task, location.campaign_name, storage_root=storage)
        common.paths.admit_generation_campaign_record_directory(campaign_root)  # plan and ledger come only from the campaign's own _record/
        campaign = _contained_file(storage, common.paths.resolve_generation_campaign_plan_path(campaign_root).relative_to(storage).as_posix())
        plan = _read_record(campaign)
        run = _read_record(common.paths.resolve_generation_run_manifest_path(common.paths.resolve_generation_execution_directory(campaign_root)))
        source_name = source["source_name"]
        if source_name not in campaigns:
            campaigns[source_name] = {
                "campaign_name": source["campaign_name"],
                "source_name": source_name,
                "plan": str(campaign),
                "plan_available": plan is not None,
                "template": _template_report(plan.get("template") if plan else None),
                "effective_configuration_files": [
                    {"path": file["relative_path"], "retained_bytes": isinstance(file.get("content"), str)} for file in plan.get("source_files", [])
                ]
                if plan
                else [],
                "generation_source_snapshot": run.get("source_snapshot") if run else None,
                "effective_scientific_overrides": plan.get("scientific_overrides") if plan else None,
                "effective_generation_values": {
                    "execution": plan.get("execution_values"),
                    "batches": [
                        {
                            "batch_id": batch.get("batch_id"),
                            "scientific_parameters": {
                                key: batch.get("scientific_values", {}).get(key)
                                for key in ("grid", "time", "schedule_time", "physical_formulas", "scientific_fixed_values", "sampling", "material")
                            },
                        }
                        for batch in plan.get("batches", [])
                    ],
                }
                if plan
                else None,
                "generation_invocation": _generation_invocation(run, plan),
            }
        cases.append(
            {
                "package_case_id": entry["package_case_id"],
                "source": source,
                "seeds": entry["design"],
                "hdf5": {
                    "path": str(hdf5),
                    "available": hdf5.is_file(),
                    "size_bytes": hdf5.stat().st_size if hdf5.is_file() else None,
                    "hash_verified_in_this_inspection": False,
                },
                "effective_parameters": {
                    "sampled_values": raw.get("sampled_values"),
                    "units": raw.get("sampled_units"),
                    "scalars": raw.get("scalars"),
                    "fixed_values": raw.get("stationary_fixed_values"),
                    "input_files": raw.get("input_files"),
                }
                if raw
                else None,
                "raw_case_record": str(raw_record),
                "solver_environment": execution.get("invocation") if execution else None,
            }
        )
    recovered_source = "EMPTY_ISOLATED_SOURCE_DIRECTORY"
    verification_output = "NEW_VERIFICATION_OUTPUT_DIRECTORY"
    snapshot_path = storage / manifest["producer_evidence"]["source_snapshot"]["relative_path"]
    source_root = common.paths.get_project_root()
    restore_command = (
        f"python {shlex.quote(str(source_root / 'scripts/source_fingerprint.py'))} {shlex.quote(str(source_root))} "
        f"--restore {shlex.quote(str(snapshot_path))} {recovered_source}"
    )
    rebuild_command = (
        f"cd {recovered_source} && env -u PROJECT_ROOT python -m src.datasets.dataset_packages rebuild "
        f"{shlex.quote(str(package / 'dataset_manifest.json'))} --storage-root {shlex.quote(str(storage))} "
        f"--output-directory {verification_output}"
    )
    return {
        "dataset_id": dataset_id,
        "dataset_name": manifest["dataset_name"],
        "dataset_view": manifest["dataset_view"],
        "manifest": str(package / "dataset_manifest.json"),
        "payload": str(package / manifest["payload_filename"]),
        "grid": manifest["grid"],
        "selection": manifest["selection"],
        "case_partition": manifest["case_partition"],
        "plan_invocation": manifest["plan_invocation"],
        "source_outcomes": manifest["source_outcomes"],
        "effective_request": manifest_selection_request(manifest),
        "channel_contract": manifest["channel_contract"],
        "source_case_count": manifest["source_case_count"],
        "sample_count": manifest["sample_count"],
        "relevant_compatibility": {
            "recorded_implementation_sha256": manifest["builder_identity"]["implementation_sha256"],
            "current_interpreter_admission": "passed manifest validation",
        },
        "transition_count": manifest["transition_count"],
        "producer": {**manifest["producer_evidence"], "source_recovery": _snapshot_report(storage, manifest["producer_evidence"]["source_snapshot"])},
        "campaigns": campaigns,
        "cases": cases,
        "regeneration": {
            "restore_command": restore_command,
            "dataset_rebuild_command": rebuild_command,
            "path_mapping": (
                "Replace the two uppercase paths with new directories. The original Source/snapshot dependency root is explicit; "
                "rebuild reads it and writes only the separately selected verification output, without copying Source trees."
            ),
            "required_storage_subtrees": [f"01_generation/{manifest['source_task']}/{name}" for name in campaigns]
            + [str(Path(manifest["producer_evidence"]["source_snapshot"]["relative_path"]).parent)],
            "dataset_from_retained_sources": (
                "Build the manifest's exact ordered source selection with the restored producing source "
                "and matching dependencies; no COMSOL is needed."
            ),
            "source_regeneration": (
                "The retained campaign configuration, exact inputs and COMSOL model are required; "
                "execute the ordinary Generation workflow through Slurm only after resource authorization."
            ),
            "environment_boundary": (
                "Recorded versions and dependency lock files support reconstruction; installed Python/solver "
                "environments and numerical equivalence have not been verified by inspection."
            ),
        },
        "payload_hash_verified_in_this_inspection": False,
        "dependency_preparation": (
            "Use the recorded Python/package versions and recovered dependency lock; rebuild verifies them. "
            "Original Source/snapshot files remain read-only inputs. COMSOL is unnecessary when selected HDF5 Sources remain available."
        ),
    }


def verify_current_producer(value: dict[str, Any], *, storage_root: Path | str | None = None) -> None:
    """Require current producing source bytes and full modes to match the saved snapshot."""
    validate_producer_evidence(value)
    storage = common.paths.get_storage_root(storage_root=storage_root).resolve()
    path = _contained_file(storage, value["source_snapshot"]["relative_path"])
    if not path.is_file() or common.serialization.file_sha256(path) != value["source_snapshot"]["sha256"]:
        message = "The exact producing source snapshot is missing or changed."
        raise ValueError(message)
    script = common.paths.get_project_root() / "scripts/source_fingerprint.py"
    spec = importlib.util.spec_from_file_location("dataset_reconstruction_source", script)
    if spec is None or spec.loader is None:
        message = "The maintained source verification implementation is unavailable."
        raise RuntimeError(message)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.verify_source_snapshot(path, common.paths.get_project_root(), frozen=common.paths.source_tree_id() is not None)
    environment = value["environment"]
    current_versions = {name: importlib.metadata.version(name) for name in environment["packages"]}
    if environment["python"] != platform.python_version() or current_versions != environment["packages"]:
        message = "Dataset reconstruction requires the recorded Python and dependency versions."
        raise ValueError(message)


def rebuild_dataset(manifest_path: Path, *, storage_root: Path | str, output_directory: Path) -> dict[str, Any]:
    """Rebuild an exact manifest selection with restored source and retained Source inputs."""
    from . import dataset_packages_builder as builder  # noqa: PLC0415 -- reusable CLI service
    from . import dataset_packages_manifest as manifests  # noqa: PLC0415 -- exact input schema

    raw = _read_record(manifest_path)
    if raw is None:
        message = f"Dataset reconstruction manifest is unavailable: {manifest_path}."
        raise FileNotFoundError(message)
    task = raw["dataset_view"]
    payload = manifest_path.parent / common.paths.get_dataset_payload_filename(task)
    manifest = manifests.validate_manifest_content(raw, task=task, dataset_id=raw["dataset_id"], payload_path=payload, validate_payload_hash=False)
    request = manifest_selection_request(manifest)
    return builder.build_dataset_selection(request, storage_root=storage_root, reproduction_manifest=manifest, output_directory=output_directory)


def manifest_selection_request(manifest: dict[str, Any]) -> dict[str, Any]:
    """Project a manifest's recorded selection and ordered sources onto the selection request schema."""
    from . import dataset_packages_selection as selection  # noqa: PLC0415 -- single selection schema projection

    return {
        "schema_kind": selection.SELECTION_SCHEMA_KIND,
        "source_task": manifest["source_task"],
        "dataset_name": manifest["dataset_name"],
        "dataset_view": manifest["dataset_view"],
        "selection": manifest["selection"],
        "source_cases": [{key: entry["source"][key] for key in selection.SOURCE_REFERENCE_ORDER} for entry in manifest["source_case_identities"]],
    }
