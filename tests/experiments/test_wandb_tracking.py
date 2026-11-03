"""Exercise W&B lifecycle, provenance, failure, and upload behavior.

Responsibilities:
  - Check disabled/offline/online lifecycle, resume identity, redaction, history, and upload gates
  - Check the silent SDK and where its local state lives, including the failed-finish preservation

Design principles:
  - Requested tracking fails closed; disabled mode has no SDK or filesystem effects

This module does NOT:
  - Contact live W&B accounts or bypass artifact root boundaries
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import torch
from support import configs

from src import datasets, experiments

tracking = experiments.tracking

_CURRENT_METRIC_SCHEMA_VERSION = 3
_GIB_BYTES = 1024**3
_PROJECTED_GIB = 2.0


class _FakeArtifact:
    def __init__(self, **metadata: Any) -> None:
        self.metadata = metadata
        self.files: list[tuple[str, str]] = []
        self.tables: list[tuple[Any, str]] = []

    def add_file(self, path: str, *, name: str) -> None:
        self.files.append((path, name))

    def add(self, value: Any, name: str) -> None:
        self.tables.append((value, name))


class _FakeRun:
    def __init__(self, *, tags: tuple[str, ...] = ()) -> None:
        self.summary: dict[str, Any] = {}
        self.tags = tags
        self.metric_definitions: list[tuple[str, dict[str, Any]]] = []
        self.logs: list[tuple[dict[str, Any], int]] = []
        self.exit_codes: list[int] = []
        self.saved: list[tuple[str, str, str]] = []
        self.artifacts: list[tuple[_FakeArtifact, list[str]]] = []
        self.fail_log = False
        self.fail_finish = False

    def define_metric(self, name: str, **kwargs: Any) -> None:
        self.metric_definitions.append((name, kwargs))

    def log(self, data: dict[str, Any], *, step: int) -> None:
        if self.fail_log:
            msg = "synthetic transport failure"
            raise OSError(msg)
        self.logs.append((dict(data), step))

    def finish(self, exit_code: int = 0) -> None:
        if self.fail_finish:
            msg = "synthetic finish failure"
            raise OSError(msg)
        self.exit_codes.append(exit_code)

    def save(self, path: str, *, base_path: str, policy: str) -> None:
        self.saved.append((path, base_path, policy))

    def log_artifact(self, artifact: _FakeArtifact, *, aliases: list[str]) -> None:
        self.artifacts.append((artifact, aliases))


class _FakeWandb:
    def __init__(
        self,
        *,
        init_error: BaseException | None = None,
        resumed_tags: tuple[str, ...] | None = None,
    ) -> None:
        self.initializations: list[dict[str, Any]] = []
        self.runs: list[_FakeRun] = []
        self.init_error = init_error
        self.resumed_tags = resumed_tags
        self.created_artifacts: list[_FakeArtifact] = []
        self.created_tables: list[dict[str, Any]] = []
        self.setups: list[Any] = []

    def Settings(self, **settings: Any) -> dict[str, Any]:  # noqa: N802
        return settings

    def setup(self, *, settings: Any) -> None:
        self.setups.append(settings)

    def init(self, **settings: Any) -> _FakeRun:
        self.initializations.append(settings)
        if self.init_error is not None:
            raise self.init_error
        tags = tuple(settings.get("tags") or ())
        if settings.get("resume") == "must" and self.resumed_tags is not None:
            tags = self.resumed_tags
        run = _FakeRun(tags=tags)
        self.runs.append(run)
        return run

    def Artifact(self, **metadata: Any) -> _FakeArtifact:  # noqa: N802
        artifact = _FakeArtifact(**metadata)
        self.created_artifacts.append(artifact)
        return artifact

    def Table(  # noqa: N802
        self,
        *,
        columns: list[str],
        data: list[list[object]],
    ) -> dict[str, Any]:
        table = {"columns": columns, "data": data}
        self.created_tables.append(table)
        return table


@pytest.fixture(autouse=True)
def _fake_online_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "test-only-wandb-key")


def _resolved_config(
    *,
    mode: str = "online",
    workflow: str = "train",
    study: str | None = None,
    epochs: int = 3,
    upload: bool = False,
    physics_enabled: bool = False,
) -> dict[str, Any]:
    raw = configs.direct_config(
        physics_enabled=physics_enabled,
        wandb_mode=mode,
        workflow=workflow,
    )
    settings = raw["tracking"]["wandb"]
    settings["upload"]["evaluation_artifacts"] = upload
    if study is not None:
        settings["study"] = study
    raw["training"]["epochs"] = epochs
    return configs.resolve(raw)


def _current_transient_config(*, mode: str = "online") -> dict[str, Any]:
    """Return one current transient child with exact Dataset ID selections."""
    train_id = "synthetic_transient_train"
    ood_id = "synthetic_transient_evaluation"
    raw = configs.transient_matched_config()
    raw["data"]["train_dataset"] = train_id
    raw["data"]["additional_evaluation_datasets"] = [ood_id]
    raw["tracking"]["wandb"]["mode"] = mode
    plan = configs.resolve_plan(raw)
    config = copy.deepcopy(dict(plan.a0))
    config["run"]["device"] = "cuda"
    return config


def _transient_split_evidence() -> dict[str, Any]:
    """Return compact transient split evidence for semantic W&B projection."""
    return {
        "schema_kind": "experiment_transient_case_split",
        "schema_version": 1,
        "tensorizer": {"input_profile": "complete", "temporal_conditioning": "none"},
        "sampling": {"window": 2},
        "additional_evaluation_fraction": 0.2,
        "split_seed": 7,
        "dataset_identity": {
            "train": {"dataset_id": "train", "index_digest": "b" * 64},
            "additional_evaluation": [{"dataset_id": "ood", "index_digest": "c" * 64}],
        },
        "roles": {
            "train": {"case_ids": ["a"]},
            "scaling_train_one_step": {"case_ids": ["a"]},
            "validation": {"case_ids": ["b"]},
            "id_test": {"case_ids": ["c"]},
            "additional_evaluation": {"parts": [{"case_ids": ["d"]}]},
        },
    }


def _patch_wandb(monkeypatch: pytest.MonkeyPatch, fake: _FakeWandb) -> None:
    original = tracking.importlib.import_module
    monkeypatch.setattr(
        tracking.importlib,
        "import_module",
        lambda name: fake if name == "wandb" else original(name),
    )


def _state_recorder() -> tuple[dict[str, Any], Any]:
    state: dict[str, Any] = {}

    def update(values: dict[str, Any]) -> None:
        state.update(values)

    return state, update


def _split_evidence(config: dict[str, Any]) -> dict[str, Any]:
    splits = datasets.preprocessing.splits
    identities = {
        role: datasets.contracts.identity.DatasetIdentity(
            dataset_id=f"artificial-{role}",
            task=config["task"],
            data_contract_digest=config["task_contract"]["data_contract_digest"],
            fingerprint=marker * 64,
            sample_ids=("sample-a", "sample-b", "sample-c"),
            sample_count=3,
            spatial_shape=(8, 8),
        )
        for role, marker in (("train", "a"), ("ood", "b"))
    }
    case_ids = list(identities["train"].sample_ids)
    partition = datasets.contracts.partition.build_fixed_partition(
        case_ids,
        material_by_case=dict.fromkeys(case_ids, "lentil"),
        material_roles={"lentil": "id"},
        recipe={"kind": "exact", "train": ["sample-b"], "validation": ["sample-c"], "id_test": ["sample-a"]},
        seed_input=9,
        seed=9,
    )
    evidence = {
        role: splits.role_evidence(role, identities["train"], splits.role_positions(identities["train"], partition[role], label=role))
        for role in splits.ID_SPLIT_ROLES
    }
    return splits.SplitContract(
        schema_version=splits.SPLIT_SCHEMA_VERSION,
        task=config["task"],
        task_contract_digest=config["task_contract"]["digest"],
        split_seed=partition["seed"],
        case_partition=partition,
        case_partition_digest=datasets.contracts.partition.partition_digest(partition),
        train=evidence["train"],
        validation=evidence["validation"],
        id_test=evidence["id_test"],
        additional_evaluation=splits.role_evidence("additional_evaluation", identities["ood"], (2,)),
    ).as_payload()


def test_disabled_tracking_has_no_sdk_or_filesystem_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leave SDK and disk untouched when tracking is disabled."""
    config = _resolved_config(mode="disabled")
    monkeypatch.setattr(
        tracking.importlib,
        "import_module",
        lambda _name: pytest.fail("disabled tracking imported the SDK"),
    )
    state, update = _state_recorder()
    session = tracking.initialize_wandb(
        config,
        run_dir=tmp_path,
        state_updater=update,
    )
    session.log_epoch(1, {"train/loss_total": 2.0})
    session.log_transient_evaluation_summary(
        {
            "evaluation/0/id/autonomous_full/full/cumulative/normalized_drying_group_macro_rmse": 0.25,
        }
    )
    session.finish(status="completed")
    assert not session.enabled
    assert state == {}
    assert list(tmp_path.iterdir()) == []


def test_offline_evaluation_summary_is_bounded_flat_and_persisted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mirror only finite bounded Evaluation scalars into offline run summary state."""
    monkeypatch.delenv("WANDB_API_KEY")
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    state, update = _state_recorder()
    session = tracking.initialize_wandb(
        _resolved_config(mode="offline"),
        run_dir=tmp_path,
        state_updater=update,
    )
    metric_prefix = "evaluation/0/id/autonomous_full/full/cumulative"
    additional_prefix = "evaluation/0/additional_evaluation/autonomous_full/full/cumulative"
    evidence = {
        f"{metric_prefix}/normalized_drying_group_macro_rmse": 0.25,
        f"{metric_prefix}/contributing_case_count": 3,
        f"{additional_prefix}/normalized_drying_group_macro_rmse": 0.4,
        "evaluation/pipeline/0/c_available": 0,
        "evaluation/identity/checkpoint_sha256": "a" * 64,
        "evaluation/identity/dataset/additional_evaluation/name": "independent_diagnostic",
        "evaluation/identity/input_profile": "canonical_physics_complete_v1",
        "evaluation/identity/backend": "canonical_hdf5",
    }
    session.log_transient_evaluation_summary(evidence)
    assert fake.runs[0].summary == evidence
    assert state["evaluation_summary_keys"] == sorted(evidence)
    assert state["evaluation_summary_entry_count"] == len(evidence)
    with pytest.raises(tracking.TrackingError, match="finite"):
        session.log_transient_evaluation_summary({f"{metric_prefix}/normalized_drying_group_macro_rmse": float("nan")})
    with pytest.raises(tracking.TrackingError, match="Unsupported"):
        session.log_transient_evaluation_summary({"unscoped": 1.0})
    with pytest.raises(tracking.TrackingError, match="Unsupported"):
        session.log_transient_evaluation_summary({"evaluation/raw_dataset_value": 1.0})
    with pytest.raises(tracking.TrackingError, match="Unsupported"):
        session.log_transient_evaluation_summary({"evaluation/identity/run_name": "arbitrary"})
    with pytest.raises(tracking.TrackingError, match="Unsupported"):
        session.log_transient_evaluation_summary({"evaluation/0/ood/autonomous_full/full/cumulative/contributing_case_count": 1})
    for unauthorized_key in (
        "evaluation/identity/scaling_semantic_digest",
        "evaluation/identity/comparison_arm",
        "evaluation/identity/training_strategy",
        "evaluation/identity/matched_compute_manifest_sha256",
    ):
        with pytest.raises(tracking.TrackingError, match="Unsupported"):
            session.log_transient_evaluation_summary({unauthorized_key: "a" * 64})
    with pytest.raises(tracking.TrackingError, match="logical name"):
        session.log_transient_evaluation_summary({"evaluation/identity/dataset/id/name": '{"raw_dataset_value":1}'})
    with pytest.raises(tracking.TrackingError, match="finite numeric"):
        session.log_transient_evaluation_summary({f"{metric_prefix}/normalized_drying_group_macro_rmse": [0.25]})
    with pytest.raises(tracking.TrackingError, match="bounded mapping"):
        session.log_transient_evaluation_summary(
            {f"evaluation/{index}/id/autonomous_full/full/cumulative/normalized_drying_group_macro_rmse": 0.25 for index in range(513)}
        )


def test_online_authentication_is_checked_before_sdk_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject absent online credentials before importing the SDK."""
    monkeypatch.delenv("WANDB_API_KEY")
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    state, update = _state_recorder()
    with pytest.raises(tracking.TrackingInitializationError, match="authentication"):
        tracking.initialize_wandb(
            _resolved_config(),
            run_dir=tmp_path,
            state_updater=update,
        )
    assert fake.initializations == []
    assert state["status"] == "failed_before_start"
    assert state["failed_operation"] == "authentication"


def test_fresh_runs_are_isolated_and_resume_reuses_persisted_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Isolate fresh identities and resume only the persisted run."""
    manual_tags = ("reviewed-manually",)
    fake = _FakeWandb(resumed_tags=manual_tags)
    _patch_wandb(monkeypatch, fake)
    config = _resolved_config()
    first_state, first_update = _state_recorder()
    first = tracking.initialize_wandb(
        config,
        run_dir=tmp_path / "first",
        state_updater=first_update,
    )
    second = tracking.initialize_wandb(config, run_dir=tmp_path / "second")
    resumed = tracking.initialize_wandb(
        config,
        run_dir=tmp_path / "first",
        resume=True,
        persisted_run_id=first.run_id,
        previous_last_logged_epoch=2,
    )
    assert first.run_id != second.run_id
    assert resumed.run_id == first.run_id == first_state["wandb_run_id"]
    assert [item["resume"] for item in fake.initializations] == ["never", "never", "must"]
    assert set(config["tracking"]["wandb"]["tags"]) <= set(fake.runs[2].tags)
    assert set(manual_tags) <= set(fake.runs[2].tags)
    resumed.log_epoch(3, {"train/loss_total": 1.0})
    with pytest.raises(tracking.TrackingError, match="cannot rewrite"):
        resumed.log_epoch(3, {"train/loss_total": 0.5})


def test_offline_mode_needs_no_key_and_preserves_resume_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep offline resume local and bound to its persisted identity."""
    monkeypatch.delenv("WANDB_API_KEY")
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    config = _resolved_config(mode="offline")
    fresh = tracking.initialize_wandb(config, run_dir=tmp_path)
    state, update = _state_recorder()
    resumed = tracking.initialize_wandb(
        config,
        run_dir=tmp_path,
        resume=True,
        persisted_run_id=fresh.run_id,
        state_updater=update,
    )
    assert resumed.run_id == fresh.run_id
    assert all(item["mode"] == "offline" for item in fake.initializations)
    assert all(item["resume"] is None for item in fake.initializations)
    assert state["offline_resume_fallback"]
    assert "api_key" not in str(fake.initializations).lower()


def _write_sdk_state(local_root: Path) -> Path:
    """Create the file that the SDK would leave in its local run directory."""
    state_file = local_root / "wandb" / "run-test" / "files" / "output.log"
    state_file.parent.mkdir(parents=True)
    state_file.write_text("unsynced\n", encoding="utf-8")
    return state_file


@pytest.mark.parametrize("mode", ["online", "offline"])
def test_sdk_is_silent_and_online_state_is_node_local_while_offline_state_stays_in_the_run_folder(
    mode: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Silence the SDK; keep online state on TMPDIR and offline state in the run folder for `wandb sync`."""
    node_tmp = tmp_path / "node-tmp"
    node_tmp.mkdir()
    monkeypatch.setenv("TMPDIR", str(node_tmp))
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    run_dir = tmp_path / "run"
    session = tracking.initialize_wandb(_resolved_config(mode=mode), run_dir=run_dir)

    assert fake.setups == [{"silent": True}]
    sdk_dir = Path(fake.initializations[0]["dir"])
    if mode == "online":
        assert sdk_dir.parent == node_tmp
        _write_sdk_state(sdk_dir)
    else:
        assert sdk_dir == run_dir
    session.finish(status="completed")
    assert not (run_dir / "wandb").exists()  # a finished session leaves nothing in storage; offline state is the SDK's own


def test_failed_online_finish_preserves_the_node_local_sdk_state_in_the_run_folder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep unsynced SDK state durable when finish fails, while the tracking error stays visible."""
    node_tmp = tmp_path / "node-tmp"
    node_tmp.mkdir()
    monkeypatch.setenv("TMPDIR", str(node_tmp))
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    session = tracking.initialize_wandb(_resolved_config(), run_dir=run_dir)
    state_file = _write_sdk_state(Path(fake.initializations[0]["dir"]))
    fake.runs[0].fail_finish = True

    with pytest.raises(tracking.TrackingIOError, match="finish"):
        session.finish(status="completed")

    assert (run_dir / "wandb" / "run-test" / "files" / "output.log").read_text(encoding="utf-8") == state_file.read_text(encoding="utf-8")


@pytest.mark.parametrize("finish_fails", [False, True])
def test_failed_sdk_state_copy_is_a_warning_and_never_masks_the_finish_outcome(
    finish_fails: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A copy error after an observer failure warns on stderr; finish still ends as it would without the copy."""
    node_tmp = tmp_path / "node-tmp"
    node_tmp.mkdir()
    monkeypatch.setenv("TMPDIR", str(node_tmp))
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    session = tracking.initialize_wandb(_resolved_config(), run_dir=run_dir)
    _write_sdk_state(Path(fake.initializations[0]["dir"]))
    session._observer_failed = True  # an earlier observer operation failed
    fake.runs[0].fail_finish = finish_fails

    def refuse_copy(*_args: object, **_kwargs: object) -> None:
        raise PermissionError

    monkeypatch.setattr(tracking.shutil, "copytree", refuse_copy)

    if finish_fails:
        with pytest.raises(tracking.TrackingIOError, match="finish"):
            session.finish(status="completed")
    else:
        session.finish(status="completed")
    assert "Local W&B state was not preserved in the run folder: PermissionError" in capsys.readouterr().err


@pytest.mark.parametrize("physics_enabled", [False, True])
def test_semantic_config_is_path_free_and_preserves_active_science(
    physics_enabled: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publish scientific provenance without runtime paths or secrets."""
    secret = "super-secret-test-key"  # noqa: S105 -- test-owned artificial credential verifies redaction without using real secrets
    monkeypatch.setenv("WANDB_API_KEY", secret)
    config = _resolved_config(physics_enabled=physics_enabled)
    payload = tracking.build_semantic_config(
        config,
        split_indices=_split_evidence(config),
        split_indices_sha256="2" * 64,
        normalizer_sha256="f" * 64,
        checkpoint_identity={"effective_config_digest": "1" * 64},
        model=torch.nn.Linear(2, 3),
        device_metadata={
            "requested_policy": "cpu",
            "resolved_device": "cpu",
            "device_type": "cpu",
            "pytorch_version": torch.__version__,
        },
        duration_contract=experiments.run.RUN_DURATION_CONTRACT,
    )
    assert payload["model"]["parameter_counts"] == {"total": 9, "trainable": 9}
    assert payload["loss"]["physics"]["enabled"] is physics_enabled
    if physics_enabled:
        assert payload["loss"]["physics"]["momentum_weight"] == config["loss"]["physics"]["momentum_weight"]
        assert payload["loss"]["physics"]["continuity_weight"] == config["loss"]["physics"]["continuity_weight"]
    else:
        assert payload["loss"]["physics"] == {"enabled": False}
    assert payload["data"]["datasets"]["id"]["dataset_id"] == "artificial-train"
    assert payload["data"]["split"]["artifact_sha256"] == "2" * 64
    assert payload["provenance"]["config_digest"] == "1" * 64
    assert {"effective_config", "paths", "tracking"}.isdisjoint(payload)
    serialized = str(payload)
    assert secret not in serialized
    assert "WANDB_API_KEY" not in serialized
    assert str(Path.home()) not in serialized


def test_transient_tracking_definitions_and_semantic_payload_use_existing_keys() -> None:
    """Keep transient observer keys fixed while preserving local evidence ownership."""
    config = _current_transient_config(mode="disabled")
    config["evaluation"]["metrics"] = [
        {"id": "normalized_drying_group_macro_rmse", "kind": "drying_group_macro_rmse", "fields": ["T", "phi", "w_surf", "w_int"]},
        {"id": "physical_mae_T", "kind": "mae", "fields": ["T"]},
    ]
    config["evaluation"]["objective"] = {"id": "normalized_drying_group_macro_rmse", "direction": "minimize"}
    config = experiments.config.loader.finalize_run_layout(config)
    split = {
        "schema_kind": "experiment_transient_case_split",
        "schema_version": 1,
        "tensorizer": {"input_profile": "complete", "temporal_conditioning": "none"},
        "sampling": {"window": 2},
        "additional_evaluation_fraction": 0.2,
        "split_seed": 7,
        "dataset_identity": {
            "train": {"dataset_id": "train", "index_digest": "b" * 64},
            "additional_evaluation": [{"dataset_id": "added", "index_digest": "c" * 64}],
        },
        "roles": {
            "train": {"case_ids": ["a"]},
            "scaling_train_one_step": {"case_ids": ["a"]},
            "validation": {"case_ids": ["b"]},
            "id_test": {"case_ids": ["c"]},
            "additional_evaluation": {"parts": [{"case_ids": ["d"]}]},
        },
    }
    payload = tracking.build_semantic_config(
        config,
        split_indices=split,
        split_indices_sha256="d" * 64,
        normalizer_sha256="e" * 64,
        checkpoint_identity={"effective_config_digest": "f" * 64},
        model=torch.nn.Linear(2, 3),
        device_metadata={"resolved_device": "cpu", "pytorch_version": torch.__version__},
        duration_contract=experiments.run.RUN_DURATION_CONTRACT,
        runtime_provenance={"train": "pt_shards"},
        transient_scaling={"semantic_digest": "1" * 64, "scale_mode": "delta_rms"},
        transient_handoff={"source_run_name": "a0", "compatibility_digest": "2" * 64},
    )
    assert payload["data"]["configured_identity"]["train_dataset"] == config["data"]["train_dataset"]
    assert payload["data"]["split"]["roles"]["validation"]["case_ids"] == ["b"]
    assert payload["data"]["normalization"]["semantic_digest"] == "1" * 64
    assert payload["provenance"]["runtime_backend"] == {"train": "pt_shards"}
    assert payload["provenance"]["checkpoint_config_digest"] == "f" * 64
    assert payload["provenance"]["resolved_config_sha256"] != payload["provenance"]["checkpoint_config_digest"]
    assert payload["provenance"]["schema_versions"]["wandb_metrics"] == _CURRENT_METRIC_SCHEMA_VERSION
    definitions = tracking.automatic_history_metric_definitions(
        config["evaluation"]["metrics"],
        objective_id="normalized_drying_group_macro_rmse",
        physics_training_enabled=False,
        physics_monitor_enabled=False,
        cuda_enabled=False,
        task_id="transient_drying",
    )
    destinations = {definition.wandb_key for definition in definitions}
    assert "Loss/w_int" in destinations
    assert "Overview/id_objective" in destinations
    assert "Accuracy/ID/T" in destinations
    assert all(not destination.startswith("Transient/") for destination in destinations)
    assert len(destinations) == len(definitions)


def test_current_transient_session_projects_curated_history_and_parent_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Project only current curated metrics while retaining opaque SDK run identity."""
    config = _current_transient_config()
    semantic = tracking.build_semantic_config(
        config,
        split_indices=_transient_split_evidence(),
        split_indices_sha256="d" * 64,
        normalizer_sha256="e" * 64,
        checkpoint_identity={"effective_config_digest": "f" * 64},
        model=torch.nn.Linear(2, 3),
        device_metadata={"resolved_device": "cuda:0", "pytorch_version": torch.__version__},
        duration_contract=experiments.run.RUN_DURATION_CONTRACT,
        runtime_provenance={"train": "pt_shards"},
        transient_scaling={"semantic_digest": "1" * 64, "scale_mode": "state_std"},
    )
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)

    session = tracking.initialize_wandb(config, run_dir=tmp_path, semantic_config=semantic)

    initialization = fake.initializations[0]
    configured_identity = initialization["config"]["data"]["configured_identity"]
    definitions = session.history_metric_definitions
    destinations = {definition.wandb_key for definition in definitions}
    assert all(not destination.startswith("Transient/") for destination in destinations)
    assert initialization["project"] == "grainlegumes-pino-drying-transient"
    assert initialization["group"] == experiments.config.naming.parent_label(config["run"]["layout"])
    assert initialization["job_type"] == "stage_a0"
    assert initialization["id"] == session.run_id
    assert initialization["id"] != initialization["name"]
    assert initialization["resume"] == "never"
    assert config["data"]["train_dataset"] not in initialization["group"]
    assert "0123456789abcdef" not in initialization["group"]
    assert configured_identity["train_dataset"] == config["data"]["train_dataset"]
    assert configured_identity["additional_evaluation_datasets"] == config["data"]["additional_evaluation_datasets"]
    metrics = {definition.source_key: 1.0 for definition in definitions}
    metrics["system/cuda_peak_memory_allocated_bytes"] = _PROJECTED_GIB * _GIB_BYTES

    session.log_epoch(1, metrics)

    payload, step = fake.runs[0].logs[-1]
    assert step == 1
    assert len(payload) == len(definitions) + 1
    assert payload["Performance/cuda_peak_memory_gib"] == _PROJECTED_GIB


def test_model_parameter_counts_ignore_device_and_dtype() -> None:
    """Count parameter state independently of placement and precision."""
    model = torch.nn.Linear(2, 3)
    model.bias.requires_grad_(False)
    expected = {"total": 9, "trainable": 6}
    assert tracking.model_parameter_counts(model) == expected
    model.to(device=torch.device("meta"), dtype=torch.float16)
    assert tracking.model_parameter_counts(model) == expected


@pytest.mark.parametrize("mode", ["online", "offline"])
def test_requested_history_failures_are_fail_closed(
    mode: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Treat requested history loss as a tracking failure."""
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    state, update = _state_recorder()
    session = tracking.initialize_wandb(
        _resolved_config(mode=mode),
        run_dir=tmp_path,
        state_updater=update,
    )
    fake.runs[0].fail_log = True
    with pytest.raises(tracking.TrackingIOError, match=rf"{mode} W&B history"):
        session.log_epoch(1, {"train/loss_total": 1.0})
    assert state["status"] == "failed"
    assert state["failed_operation"] == "history"


def test_initialization_and_finish_errors_are_redacted_and_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Redact credentials while recording lifecycle failures."""
    secret = "top-secret-wandb-key"  # noqa: S105 -- test-owned artificial credential verifies redaction without using real secrets
    monkeypatch.setenv("WANDB_API_KEY", secret)
    fake = _FakeWandb(init_error=RuntimeError(f"api_key={secret}"))
    _patch_wandb(monkeypatch, fake)
    state, update = _state_recorder()
    with pytest.raises(tracking.TrackingInitializationError) as captured:
        tracking.initialize_wandb(
            _resolved_config(),
            run_dir=tmp_path,
            state_updater=update,
        )
    assert secret not in str(captured.value)
    assert secret not in str(state)

    finish_fake = _FakeWandb()
    _patch_wandb(monkeypatch, finish_fake)
    finish_state, finish_update = _state_recorder()
    session = tracking.initialize_wandb(
        _resolved_config(),
        run_dir=tmp_path,
        state_updater=finish_update,
    )
    finish_fake.runs[0].fail_finish = True
    with pytest.raises(tracking.TrackingIOError, match="finish"):
        session.finish(status="completed")
    assert finish_state["status"] == "failed"
    assert finish_state["failed_operation"] == "finish"


def test_artifact_upload_enforces_enablement_allowlist_and_root_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restrict uploads to enabled, bounded run artifacts."""
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    analysis_root = tmp_path / "analysis" / "id"
    analysis_root.mkdir(parents=True)
    provenance = analysis_root / "artifact_provenance.json"
    provenance.write_text("{}\n", encoding="utf-8")

    disabled = tracking.initialize_wandb(_resolved_config(), run_dir=tmp_path)
    with pytest.raises(tracking.TrackingUploadError, match="disabled"):
        disabled.upload_files({"artifact_provenance": provenance})

    session = tracking.initialize_wandb(_resolved_config(upload=True), run_dir=tmp_path)
    with pytest.raises(tracking.TrackingUploadError, match="Unsupported"):
        session.upload_files({"arbitrary": provenance})
    outside = tmp_path / "outside" / "artifact_provenance.json"
    outside.parent.mkdir()
    outside.write_text("{}\n", encoding="utf-8")
    with pytest.raises(tracking.TrackingUploadError, match="analysis root"):
        session.upload_files({"artifact_provenance": outside})
    session.upload_files({"artifact_provenance": provenance})
    assert Path(fake.runs[1].saved[0][0]) == provenance

    media: dict[str, Path] = {}
    for name in tracking.POST_ARTIFACT_MEDIA_KEYS - {"run_summary_table"}:
        rendered = tmp_path / "rendered" / f"{name}.png"
        rendered.parent.mkdir(exist_ok=True)
        rendered.write_bytes(b"rendered")
        media[name] = rendered
    session.upload_post_artifact(
        artifact_root=analysis_root,
        media_files=media,
        tables={"run_summary_table": {"columns": ["score"], "data": [[0.25]]}},
    )
    assert len(fake.runs[1].artifacts) == 1


def test_persisted_identity_requires_one_run_id_and_uses_latest_epoch() -> None:
    """Recover one run identity and its latest logged epoch."""
    summary = {
        "runtime_sessions": [
            {"tracking": {"wandb_run_id": "same", "last_logged_epoch": 2}},
            {"tracking": {"wandb_run_id": "same", "last_logged_epoch": 5}},
        ]
    }
    assert tracking.persisted_wandb_identity(summary) == ("same", 5)
    summary["runtime_sessions"].append({"tracking": {"wandb_run_id": "different"}})
    with pytest.raises(tracking.TrackingInitializationError, match="found 2"):
        tracking.persisted_wandb_identity(summary)


@pytest.mark.parametrize(
    ("settings", "match"),
    [
        ({"mode": "invalid"}, r"tracking\.wandb\.mode"),
        ({"monitor": {"interval": 0}}, r"tracking\.wandb\.monitor\.interval"),
        (
            {"upload": {"evaluation_artifacts": 1}},
            r"tracking\.wandb\.upload\.evaluation_artifacts",
        ),
        ({"workflow": "optuna_trial"}, r"tracking\.wandb\.study is required"),
        (
            {"workflow": "train", "study": "not-allowed"},
            r"tracking\.wandb\.study is valid only",
        ),
    ],
)
def test_wandb_config_rejects_invalid_values(
    settings: dict[str, Any],
    match: str,
) -> None:
    """Reject malformed tracking settings through the public resolver."""
    raw = configs.direct_config()
    raw["tracking"] = {"wandb": settings}
    with pytest.raises(ValueError, match=match):
        configs.resolve(raw)
