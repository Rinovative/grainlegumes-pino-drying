"""Exercise W&B lifecycle, provenance, failure, and upload behavior.

Responsibilities:
  - Check disabled/offline/online lifecycle, resume identity, redaction, history, and upload gates
  - Check project, group, job type, name, tags, summary datasets and logged key set of steady, transient and Optuna sessions
  - Check that the terminal summary reports the run's committed training work
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
from support import configs, optuna_trials
from support.synthetic_task import synthetic_split_payload

from src import experiments

tracking = experiments.tracking

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
    session.log_evaluation_summary(
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
        "evaluation/id/normalized_group_macro_rmse": 0.3,
        "evaluation/additional_evaluation/nonfinite_value_count": 2,
        "evaluation/timing/component/batch_amortized_neural_operator_forward_seconds/median_seconds": 0.01,
    }
    session.log_evaluation_summary(evidence)
    assert {key: value for key, value in fake.runs[0].summary.items() if key.startswith("evaluation/")} == evidence
    assert state["evaluation_summary_keys"] == sorted(evidence)
    assert state["evaluation_summary_entry_count"] == len(evidence)
    with pytest.raises(tracking.TrackingError, match="finite"):
        session.log_evaluation_summary({f"{metric_prefix}/normalized_drying_group_macro_rmse": float("nan")})
    with pytest.raises(tracking.TrackingError, match="Unsupported"):
        session.log_evaluation_summary({"unscoped": 1.0})
    with pytest.raises(tracking.TrackingError, match="Unsupported"):
        session.log_evaluation_summary({"evaluation/raw_dataset_value": 1.0})
    with pytest.raises(tracking.TrackingError, match="Unsupported"):
        session.log_evaluation_summary({"evaluation/identity/run_name": "arbitrary"})
    with pytest.raises(tracking.TrackingError, match="Unsupported"):
        session.log_evaluation_summary({"evaluation/0/ood/autonomous_full/full/cumulative/contributing_case_count": 1})
    with pytest.raises(tracking.TrackingError, match="Unsupported"):
        session.log_evaluation_summary({"evaluation/id/physical_rmse_T": 1.0})
    with pytest.raises(tracking.TrackingError, match="non-negative integer"):
        session.log_evaluation_summary({"evaluation/id/nonfinite_value_count": 0.5})
    for unauthorized_key in (
        "evaluation/identity/scaling_semantic_digest",
        "evaluation/identity/comparison_arm",
        "evaluation/identity/training_strategy",
        "evaluation/identity/matched_compute_manifest_sha256",
    ):
        with pytest.raises(tracking.TrackingError, match="Unsupported"):
            session.log_evaluation_summary({unauthorized_key: "a" * 64})
    with pytest.raises(tracking.TrackingError, match="logical name"):
        session.log_evaluation_summary({"evaluation/identity/dataset/id/name": '{"raw_dataset_value":1}'})
    with pytest.raises(tracking.TrackingError, match="finite numeric"):
        session.log_evaluation_summary({f"{metric_prefix}/normalized_drying_group_macro_rmse": [0.25]})
    with pytest.raises(tracking.TrackingError, match="bounded mapping"):
        session.log_evaluation_summary(
            {f"evaluation/{index}/id/autonomous_full/full/cumulative/normalized_drying_group_macro_rmse": 0.25 for index in range(513)}
        )


_RMSE_KEY = "evaluation/0/id/autonomous_full/full/cumulative/normalized_drying_group_macro_rmse"
_NONFINITE_KEY = "evaluation/id/nonfinite_value_count"


class _SummaryWithoutRemoval:
    def __init__(self) -> None:
        self.values: dict[str, Any] = {}

    def update(self, payload: dict[str, Any]) -> None:
        self.values.update(payload)

    def keys(self) -> list[str]:
        return list(self.values)


def test_evaluation_summary_replacement_removes_only_stale_tracker_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drop earlier tracker keys the replacement omits, in-session and after a resume, and keep every other summary key."""
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    state, update = _state_recorder()
    config = _resolved_config()
    first = tracking.initialize_wandb(config, run_dir=tmp_path, state_updater=update)
    first.log_evaluation_summary({_RMSE_KEY: 0.25, _NONFINITE_KEY: 0})
    first.log_evaluation_summary({_NONFINITE_KEY: 3})
    summary = fake.runs[0].summary
    assert _RMSE_KEY not in summary
    assert summary[_NONFINITE_KEY] == 3
    assert "data/train_dataset" in summary
    assert state["evaluation_summary_keys"] == [_NONFINITE_KEY]
    assert state["evaluation_summary_entry_count"] == 1

    first.log_evaluation_summary({_RMSE_KEY: 0.5, _NONFINITE_KEY: 0})
    persisted = {"runtime_sessions": [{"tracking": state}, {"tracking": {"status": "active"}}, "not a session"]}
    previous_keys = tracking.persisted_evaluation_summary_keys(persisted)
    assert previous_keys == {_RMSE_KEY, _NONFINITE_KEY}
    for keys, expected_rmse in ((previous_keys, False), (frozenset(), True)):
        resumed = tracking.initialize_wandb(
            config,
            run_dir=tmp_path,
            resume=True,
            persisted_run_id=first.run_id,
            previous_evaluation_summary_keys=keys,
        )
        fake.runs[-1].summary.update(summary)
        resumed.log_evaluation_summary({_NONFINITE_KEY: 5})
        assert (_RMSE_KEY in fake.runs[-1].summary) is expected_rmse
        assert fake.runs[-1].summary[_NONFINITE_KEY] == 5


def test_evaluation_summary_replacement_needs_summary_key_removal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed when stale tracker keys exist but the run summary cannot remove keys."""
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    session = tracking.initialize_wandb(_resolved_config(), run_dir=tmp_path)
    monkeypatch.setattr(fake.runs[0], "summary", _SummaryWithoutRemoval())
    session.log_evaluation_summary({_RMSE_KEY: 0.25, _NONFINITE_KEY: 0})
    session.log_evaluation_summary({_RMSE_KEY: 0.5, _NONFINITE_KEY: 1})
    with pytest.raises(tracking.TrackingIOError, match="key removal"):
        session.log_evaluation_summary({_NONFINITE_KEY: 2})


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


def test_resume_from_an_older_last_keeps_the_first_attempt_history_of_retrained_epochs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Epochs a resume trains again were logged by an earlier session: W&B keeps those values and logs only newer epochs."""
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    config = _resolved_config()
    first = tracking.initialize_wandb(config, run_dir=tmp_path)
    state, update = _state_recorder()
    resumed = tracking.initialize_wandb(
        config,
        run_dir=tmp_path,
        resume=True,
        persisted_run_id=first.run_id,
        previous_last_logged_epoch=4,
        state_updater=update,
    )

    resumed.log_epoch(3, {"train/loss_total": 1.0})
    resumed.log_epoch(4, {"train/loss_total": 0.9})
    assert fake.runs[1].logs == []
    assert "last_logged_epoch" not in state
    resumed.log_epoch(5, {"train/loss_total": 0.8})
    assert [step for _data, step in fake.runs[1].logs] == [5]
    assert state["last_logged_epoch"] == 5
    with pytest.raises(tracking.TrackingError, match="cannot rewrite"):
        resumed.log_epoch(5, {"train/loss_total": 0.7})


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
        split_indices=synthetic_split_payload("steady_flow", train_dataset="artificial-train"),
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
    assert payload["data"]["datasets"]["train"]["dataset_id"] == "artificial-train"
    assert [part["dataset_id"] for part in payload["data"]["datasets"]["additional_evaluation"]] == ["ood_data"]
    assert payload["data"]["split"]["roles"]["additional_evaluation"]["parts"][0]["case_ids"] == ["chickpea__case_0001__00000000000a"]
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
    split = synthetic_split_payload("transient_drying")
    payload = tracking.build_semantic_config(
        config,
        split_indices=split,
        split_indices_sha256="d" * 64,
        normalizer_sha256="e" * 64,
        checkpoint_identity={"effective_config_digest": "f" * 64},
        model=torch.nn.Linear(2, 3),
        device_metadata={"resolved_device": "cpu", "pytorch_version": torch.__version__},
        duration_contract=experiments.run.RUN_DURATION_CONTRACT,
        transient_scaling={"semantic_digest": "1" * 64, "scale_mode": "delta_rms"},
        transient_handoff={"source_run_name": "a0", "compatibility_digest": "2" * 64},
    )
    assert payload["data"]["configured_identity"]["train_dataset"] == config["data"]["train_dataset"]
    assert payload["data"]["split"]["roles"]["validation"] == split["roles"]["validation"]
    assert payload["data"]["split"]["extension"] == split["extension"]
    assert payload["data"]["normalization"]["semantic_digest"] == "1" * 64
    assert payload["provenance"]["checkpoint_config_digest"] == "f" * 64
    assert payload["provenance"]["resolved_config_sha256"] != payload["provenance"]["checkpoint_config_digest"]
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
    assert "Overview/objective_validation" in destinations
    assert "Accuracy/Validation/T" in destinations
    assert all(not destination.startswith("Transient/") for destination in destinations)
    assert len(destinations) == len(definitions)


def test_current_transient_session_projects_curated_history_and_config_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Project only current curated metrics under the config's group while retaining opaque SDK run identity."""
    config = _current_transient_config()
    semantic = tracking.build_semantic_config(
        config,
        split_indices=synthetic_split_payload("transient_drying"),
        split_indices_sha256="d" * 64,
        normalizer_sha256="e" * 64,
        checkpoint_identity={"effective_config_digest": "f" * 64},
        model=torch.nn.Linear(2, 3),
        device_metadata={"resolved_device": "cuda:0", "pytorch_version": torch.__version__},
        duration_contract=experiments.run.RUN_DURATION_CONTRACT,
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
    layout = config["run"]["layout"]
    assert initialization["project"] == experiments.config.defaults.WANDB_TASK_PROJECTS["transient_drying"]
    assert initialization["group"] == f"{layout['family']}/{layout['config_name']}"
    assert initialization["job_type"] == "stage_a0"
    assert initialization["name"] == config["run"]["name"]
    assert initialization["tags"] == ["fno"]
    assert initialization["id"] == session.run_id
    assert initialization["id"] != initialization["name"]
    assert initialization["resume"] == "never"
    assert config["data"]["train_dataset"] not in initialization["group"]
    assert fake.runs[0].summary == {
        "data/train_dataset": config["data"]["train_dataset"],
        "data/ood_datasets": config["data"]["additional_evaluation_datasets"],
    }
    assert configured_identity["train_dataset"] == config["data"]["train_dataset"]
    assert configured_identity["additional_evaluation_datasets"] == config["data"]["additional_evaluation_datasets"]
    metrics = {definition.source_key: 1.0 for definition in definitions}
    metrics["system/cuda_peak_memory_allocated_bytes"] = _PROJECTED_GIB * _GIB_BYTES
    metrics["system/session_elapsed_seconds"] = 5.0  # telemetry outside the curated list is not logged

    session.log_epoch(1, metrics)

    payload, step = fake.runs[0].logs[-1]
    assert step == 1
    assert set(payload) == {"epoch", *destinations}
    assert payload["Performance/cuda_peak_memory_gib"] == _PROJECTED_GIB


def _steady_loop_telemetry(config: dict[str, Any]) -> dict[str, float]:
    """Return the completed-epoch names a physics-informed steady training loop emits, with telemetry W&B must not log."""
    names = [
        "train/loss_total",
        "train/loss_data",
        "optimization/learning_rate",
        "additional_evaluation/objective_difference",
        "system/epoch_duration_seconds",
        "system/train_duration_seconds",
        "system/train_samples_per_second",
        "system/session_elapsed_seconds",
        "checkpoint/new_best",
        "global_step",
    ]
    names += [f"{role}/{metric['id']}" for role in ("id", "additional_evaluation") for metric in config["evaluation"]["metrics"]]
    names += [f"physics/id/{name}" for name in ("momentum_residual_mse", "flux_balance_mse", "pressure_boundary_mse")]
    for term in ("momentum", "continuity", "boundary"):
        names += [f"physics/train/loss_{term}", f"physics/train/{term}_weight"]
    return dict.fromkeys(names, 1.0)


def test_steady_session_is_grouped_by_config_and_logs_the_curated_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Group a steady run by family and config, log every design section, and fill the OOD accuracy series."""
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    config = _resolved_config(physics_enabled=True)
    layout = config["run"]["layout"]

    session = tracking.initialize_wandb(config, run_dir=tmp_path)

    initialization = fake.initializations[0]
    assert initialization["project"] == experiments.config.defaults.WANDB_TASK_PROJECTS["steady_flow"]
    assert initialization["group"] == f"{layout['family']}/{layout['config_name']}"
    assert (initialization["job_type"], initialization["name"], initialization["tags"]) == ("train", config["run"]["name"], ["pi-fno"])
    assert fake.runs[0].summary == {"data/train_dataset": "synthetic_train", "data/ood_datasets": ["synthetic_ood"]}

    session.log_epoch(1, _steady_loop_telemetry(config))

    payload, _step = fake.runs[0].logs[-1]
    keys = set(payload) - {"epoch"}
    validation = {key.removeprefix("Accuracy/Validation/") for key in keys if key.startswith("Accuracy/Validation/")}
    ood = {key.removeprefix("Accuracy/OOD/") for key in keys if key.startswith("Accuracy/OOD/")}
    assert validation == ood
    assert {"normalized_rmse_p", "physical_velocity_vector_rmse"} <= ood
    assert {key for key in keys if not key.startswith("Accuracy/")} == {
        *(f"Overview/{name}" for name in ("objective_validation", "objective_ood", "ood_gap", "train_loss", "learning_rate")),
        *(f"Loss/{name}" for name in ("data", "momentum", "continuity", "boundary")),
        *(f"Physics/Validation/{name}" for name in ("momentum", "flux_balance", "pressure_boundary")),
        *(f"Physics/Weight/{name}" for name in ("momentum", "continuity", "boundary")),
        *(f"Performance/{name}" for name in ("epoch_seconds", "train_seconds", "samples_per_second")),
    }
    # The local summary carries the counts in every terminal state, a pruned or failed run included.
    work = {"optimizer_steps": 3, "gradient_predictions": 6}
    session.finish(status="pruned", result=None, local_summary={"training_work": work})
    assert fake.runs[0].summary["training_work"] == work


class _TrialStub:
    """Provide the sampling surface that preparing one Optuna trial config needs."""

    def __init__(self, number: int) -> None:
        self.number = number

    def suggest_categorical(self, name: str, choices: Any) -> Any:
        del name
        return choices[0]

    def suggest_float(self, name: str, low: float, high: float, *, log: bool = False, step: float | None = None) -> float:
        del name, high, log, step
        return low

    def suggest_int(self, name: str, low: int, high: int, *, log: bool = False, step: int = 1) -> int:
        del name, high, log, step
        return low

    def set_user_attr(self, key: str, value: Any) -> None:
        del key, value

    def report(self, value: float, step: int) -> None:
        del value, step

    def should_prune(self) -> bool:
        return False


def _trial_config(tmp_path: Path, *, trial_number: int, epochs: int, revision: str | None = None) -> dict[str, Any]:
    """Return one prepared steady Optuna trial config of an artificial study with online tracking."""
    payload = configs.optuna_config()
    payload["experiment"]["data"]["train_dataset"] = "synthetic_train"
    payload["experiment"]["tracking"]["wandb"]["mode"] = "online"
    path = configs.write_yaml(tmp_path / "synthetic-study.yaml", payload)
    study = optuna_trials.with_configured_identities(experiments.tuning.optuna.load_optuna_study_config(path, epochs=epochs, revision=revision))
    config, _context = experiments.tuning.optuna._prepare_trial_config(study, _TrialStub(trial_number))
    return config


def test_optuna_trials_group_by_study_directory_and_log_the_report_and_best_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep studies of different budgets and revisions in separate groups with unique trial names and exact Optuna series."""
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    studies = ((2, None, "synthetic-study__ep-2"), (5, None, "synthetic-study__ep-5"), (6, "fix", "synthetic-study__ep-6__rev-fix"))
    sessions = [
        tracking.initialize_wandb(
            _trial_config(tmp_path, trial_number=3, epochs=epochs, revision=revision),
            run_dir=tmp_path / f"ep{epochs}",
            semantic_config={"tuning": {"trial_number": 3}},
        )
        for epochs, revision, _study_dir in studies
    ]

    for (_epochs, _revision, study_dir), initialization in zip(studies, fake.initializations, strict=True):
        assert initialization["group"] == f"optuna/{study_dir}"
        assert initialization["job_type"] == "optuna_trial"
        assert initialization["name"] == f"{study_dir}__trial-0003"
        assert initialization["tags"] == ["fno", "optuna"]
    assert len({initialization["name"] for initialization in fake.initializations}) == len(studies)

    sessions[0].log_epoch(1, {"id/normalized_group_macro_rmse": 0.5, "optuna/objective": 0.5, "optuna/best_objective": 0.4})
    payload, _step = fake.runs[0].logs[-1]
    assert payload["Optuna/objective"] == 0.5
    assert payload["Optuna/best_objective"] == 0.4
    summaries = {name: options["summary"] for name, options in fake.runs[0].metric_definitions if name.startswith(("Optuna/", "Overview/"))}
    assert summaries["Optuna/objective"] is summaries["Optuna/best_objective"] is None  # SDK default: the summary keeps the history's last value
    assert summaries["Overview/train_loss"] == "none"

    sessions[0].finish(status="completed", result={"best_metric": 0.4})
    terminal = fake.runs[0].summary
    assert terminal["Optuna/trial_number"] == 3
    assert {"Optuna/objective", "Optuna/best_objective"}.isdisjoint(terminal)  # only the history aggregates define them


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
        ({"workflow": "train", "study": "not-allowed"}, r"tracking\.wandb.*study"),
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
