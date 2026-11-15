"""Exercise W&B lifecycle, provenance, and failure behavior.

Responsibilities:
  - Check disabled/offline/online lifecycle, resume identity, redaction, and history
  - Check project, group, job type, name, tags, summary datasets and logged key set of steady, transient and Optuna sessions
  - Check that the terminal summary reports the run's committed training work
  - Check the silent SDK and where its local state lives, including the failed-finish preservation

Design principles:
  - Requested tracking fails closed; disabled mode has no SDK or filesystem effects

This module does NOT:
  - Contact live W&B accounts
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import torch
from support import configs, optuna_trials
from support.synthetic_task import synthetic_split_payload

from src import experiments
from src.experiments import experiments_wandb_schema as schema

if TYPE_CHECKING:
    from collections.abc import Mapping

tracking = experiments.tracking

_GIB_BYTES = 1024**3
_PROJECTED_GIB = 2.0


class _FakeRun:
    def __init__(self, *, tags: tuple[str, ...] = ()) -> None:
        self.summary: dict[str, Any] = {}
        self.tags = tags
        self.metric_definitions: list[tuple[str, dict[str, Any]]] = []
        self.logs: list[tuple[dict[str, Any], int]] = []
        self.exit_codes: list[int] = []
        self.fail_log = False
        self.events: list[str] = []
        self.fail_finish = False

    def define_metric(self, name: str, **kwargs: Any) -> None:
        self.metric_definitions.append((name, kwargs))

    def log(self, data: dict[str, Any], **options: Any) -> None:
        assert not options  # W&B's own step only counts calls; train_seconds in the payload is the x-axis
        if self.fail_log:
            msg = "synthetic transport failure"
            raise OSError(msg)
        self.logs.append((dict(data), data["global_step"]))
        self.events.append("log")

    def finish(self, exit_code: int = 0) -> None:
        if self.fail_finish:
            msg = "synthetic finish failure"
            raise OSError(msg)
        self.exit_codes.append(exit_code)


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


@pytest.fixture(autouse=True)
def _fake_online_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "test-only-wandb-key")


def _resolved_config(
    *,
    mode: str = "online",
    workflow: str = "train",
    epochs: int = 3,
    physics_enabled: bool = False,
) -> dict[str, Any]:
    raw = configs.direct_config(
        physics_enabled=physics_enabled,
        wandb_mode=mode,
        workflow=workflow,
    )
    raw["training"]["budget_epochs"] = epochs
    return configs.resolve(raw)


def _current_transient_config(*, mode: str = "online") -> dict[str, Any]:
    """Return one current transient child with exact Dataset ID selections."""
    train_id = "synthetic_transient_train"
    ood_id = "synthetic_transient_evaluation"
    raw = configs.transient_matched_config()
    raw["data"]["train_dataset"] = train_id
    raw["data"]["material_ood_datasets"] = [ood_id]
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
        ood_roles=(),
    )
    session.log_event(1, {"train/loss_total": 2.0, "budget/train_seconds": 2.0})
    session.finish(status="completed")
    assert not session.enabled
    assert state == {}
    assert list(tmp_path.iterdir()) == []


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
            ood_roles=(),
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
        ood_roles=(),
    )
    second = tracking.initialize_wandb(config, run_dir=tmp_path / "second", ood_roles=())
    resumed = tracking.initialize_wandb(
        config,
        run_dir=tmp_path / "first",
        resume=True,
        persisted_run_id=first.run_id,
        previous_last_logged_step=2,
        ood_roles=(),
    )
    assert first.run_id != second.run_id
    assert resumed.run_id == first.run_id == first_state["wandb_run_id"]
    assert [item["resume"] for item in fake.initializations] == ["never", "never", "must"]
    assert set(config["tracking"]["wandb"]["tags"]) <= set(fake.runs[2].tags)
    assert set(manual_tags) <= set(fake.runs[2].tags)
    resumed.log_event(3, {"train/loss_total": 1.0, "budget/train_seconds": 2.0})
    with pytest.raises(tracking.TrackingError, match="cannot rewrite"):
        resumed.log_event(3, {"train/loss_total": 0.5, "budget/train_seconds": 2.0})


def test_resume_from_an_older_last_keeps_the_first_attempt_history_of_retrained_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Steps a resume trains again were logged by an earlier session: W&B keeps those values and logs only newer steps."""
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    config = _resolved_config()
    first = tracking.initialize_wandb(config, run_dir=tmp_path, ood_roles=())
    state, update = _state_recorder()
    resumed = tracking.initialize_wandb(
        config,
        run_dir=tmp_path,
        resume=True,
        persisted_run_id=first.run_id,
        previous_last_logged_step=4,
        state_updater=update,
        ood_roles=(),
    )

    resumed.log_event(3, {"train/loss_total": 1.0, "budget/train_seconds": 2.0})
    resumed.log_event(4, {"train/loss_total": 0.9, "budget/train_seconds": 2.0})
    assert fake.runs[1].logs == []
    assert "last_logged_step" not in state
    resumed.log_event(5, {"train/loss_total": 0.8, "budget/train_seconds": 2.0})
    assert [step for _data, step in fake.runs[1].logs] == [5]
    assert fake.runs[1].logs[0][0]["global_step"] == 5
    assert state["last_logged_step"] == 5
    with pytest.raises(tracking.TrackingError, match="cannot rewrite"):
        resumed.log_event(5, {"train/loss_total": 0.7, "budget/train_seconds": 2.0})


def test_resume_dedupes_training_by_step_and_validation_by_point(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Training projections follow the step rule and validation projections the point rule, each independent of the other."""
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    config = _resolved_config()
    objective = f"id/{config['evaluation']['objective']['id']}"
    event = {"train/loss_total": 1.0, "budget/train_seconds": 2.0, objective: 0.5}
    first = tracking.initialize_wandb(config, run_dir=tmp_path, ood_roles=())
    state, update = _state_recorder()

    def recording_update(values: Mapping[str, Any]) -> None:
        update(values)
        if "last_logged_point" in values:
            fake.runs[1].events.append("persist-point")

    resumed = tracking.initialize_wandb(
        config,
        run_dir=tmp_path,
        resume=True,
        persisted_run_id=first.run_id,
        previous_last_logged_step=4,
        previous_last_logged_point=2,
        state_updater=recording_update,
        ood_roles=(),
    )
    run = fake.runs[1]

    resumed.log_event(5, {**event, "validation/point": 2.0})  # a replayed point at a new step keeps only its training keys
    assert run.logs[-1][0] == {"global_step": 5, "train_seconds": 2.0, "Overview/train-loss_total": 1.0}
    assert "last_logged_point" not in state
    resumed.log_event(6, {**event, "validation/point": 3.0})  # a new point sends everything, persisted before the log
    assert set(run.logs[-1][0]) == {"global_step", "train_seconds", "Overview/train-loss_total", "Overview/objective-id_validation"}
    assert run.events == ["log", "persist-point", "log"]
    assert (state["last_logged_step"], state["last_logged_point"]) == (6, 3)
    with pytest.raises(tracking.TrackingError, match="cannot rewrite"):
        resumed.log_event(6, event)


def test_resume_sends_a_new_point_below_the_restored_step_without_training_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a seconds budget the replay can reach a new point at a lower global step: only its validation keys are sent."""
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    config = _resolved_config()
    objective = f"id/{config['evaluation']['objective']['id']}"
    first = tracking.initialize_wandb(config, run_dir=tmp_path, ood_roles=())
    state, update = _state_recorder()
    resumed = tracking.initialize_wandb(
        config,
        run_dir=tmp_path,
        resume=True,
        persisted_run_id=first.run_id,
        previous_last_logged_step=8,
        previous_last_logged_point=2,
        state_updater=update,
        ood_roles=(),
    )

    resumed.log_event(5, {"train/loss_total": 1.0, "budget/train_seconds": 2.0})  # at or below the restored step: nothing
    resumed.log_event(6, {"train/loss_total": 1.0, "budget/train_seconds": 2.0, objective: 0.5, "validation/point": 3.0})
    assert fake.runs[1].logs == [({"global_step": 6, "train_seconds": 2.0, "Overview/objective-id_validation": 0.5}, 6)]
    assert state["last_logged_point"] == 3
    assert "last_logged_step" not in state


def test_every_curated_history_definition_uses_the_optimisation_clock_as_its_axis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hidden axes are declared once and every curated series of a steady run plots over train_seconds (one loop serves both tasks)."""
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    session = tracking.initialize_wandb(_resolved_config(), run_dir=tmp_path, ood_roles=())
    definitions = dict(fake.runs[0].metric_definitions)
    assert definitions["train_seconds"] == {"hidden": True, "summary": "none"}
    curated = {definition.wandb_key for definition in session.history_metric_definitions}
    assert curated <= definitions.keys()
    assert {definitions[key]["step_metric"] for key in curated} == {"train_seconds"}
    assert {definitions[key]["step_sync"] for key in curated} == {False}


def test_a_training_event_without_the_optimisation_clock_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The x-axis is mandatory: an event without budget/train_seconds fails closed instead of falling back to steps."""
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    session = tracking.initialize_wandb(_resolved_config(), run_dir=tmp_path, ood_roles=())
    with pytest.raises(tracking.TrackingError, match=r"budget/train_seconds"):
        session.log_event(1, {"train/loss_total": 1.0})
    assert fake.runs[0].logs == []
    session.log_event(1, {"train/loss_total": 1.0, "budget/train_seconds": 2.0})  # the rejected event consumed no step
    assert fake.runs[0].logs[0][0]["train_seconds"] == 2.0


def test_offline_mode_needs_no_key_and_preserves_resume_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep offline resume local and bound to its persisted identity."""
    monkeypatch.delenv("WANDB_API_KEY")
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    config = _resolved_config(mode="offline")
    fresh = tracking.initialize_wandb(config, run_dir=tmp_path, ood_roles=())
    state, update = _state_recorder()
    resumed = tracking.initialize_wandb(
        config,
        run_dir=tmp_path,
        resume=True,
        persisted_run_id=fresh.run_id,
        state_updater=update,
        ood_roles=(),
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
    session = tracking.initialize_wandb(_resolved_config(mode=mode), run_dir=run_dir, ood_roles=())

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
    session = tracking.initialize_wandb(_resolved_config(), run_dir=run_dir, ood_roles=())
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
    session = tracking.initialize_wandb(_resolved_config(), run_dir=run_dir, ood_roles=())
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
    assert [part["dataset_id"] for part in payload["data"]["datasets"]["material_ood"]] == ["ood_data"]
    assert payload["data"]["split"]["roles"]["material_ood"]["parts"][0]["case_ids"] == ["chickpea__case_0001__00000000000a"]
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
        transient_scaling={"semantic_digest": "1" * 64, "train_case_count": 2},
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
        ood_roles=("material_ood",),
    )
    destinations = {definition.wandb_key for definition in definitions}
    expected = {"Loss/data-w_int", "Overview/objective-id_validation", "Fields/material-id_validation-temperature"}
    assert expected | {"Moisture/material-id_validation-w_gr_physical"} <= destinations
    assert len(destinations) == len(definitions)


@pytest.mark.parametrize(
    ("device", "visible", "expected"),
    [("cuda", "2", [2]), ("cuda", "0,3", [0, 3]), ("cuda", None, None), ("cpu", "2", None)],
)
def test_system_monitor_watches_only_the_allocated_gpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    visible: str | None,
    expected: list[int] | None,
) -> None:
    """Bind the W&B GPU monitor to Slurm's node indices of a CUDA run; without them, or on CPU, leave it unrestricted."""
    config = _current_transient_config()
    config["run"]["device"] = device
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)

    tracking.initialize_wandb(config, run_dir=tmp_path, ood_roles=("material_ood",))

    assert fake.initializations[0]["settings"].get("x_stats_gpu_device_ids") == expected


def test_system_monitor_refuses_device_names_it_cannot_bind() -> None:
    """Refuse a device list that names GPUs by anything but node indices instead of monitoring foreign GPUs."""
    with pytest.raises(ValueError, match="integer GPU indices"):
        tracking.allocated_gpu_ids("GPU-1a2b,1")


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
        transient_scaling={"semantic_digest": "1" * 64},
    )
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)

    session = tracking.initialize_wandb(config, run_dir=tmp_path, semantic_config=semantic, ood_roles=("material_ood", "parameter_ood"))

    initialization = fake.initializations[0]
    configured_identity = initialization["config"]["data"]["configured_identity"]
    definitions = session.history_metric_definitions
    destinations = {definition.wandb_key for definition in definitions}
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
        "data/ood_datasets": config["data"]["material_ood_datasets"],
    }
    assert configured_identity["train_dataset"] == config["data"]["train_dataset"]
    assert configured_identity["material_ood_datasets"] == config["data"]["material_ood_datasets"]
    metrics = {definition.source_key: 1.0 for definition in definitions}
    metrics["system/cuda_peak_memory_allocated_bytes"] = _PROJECTED_GIB * _GIB_BYTES
    metrics["system/session_elapsed_seconds"] = 5.0  # telemetry outside the curated list is not logged
    metrics["validation/point"] = 1.0  # validation series are sent with their point
    metrics["budget/train_seconds"] = 1.0  # the optimisation clock is the hidden x-axis, not a curated series

    session.log_event(1, metrics)

    payload, step = fake.runs[0].logs[-1]
    assert step == 1
    assert set(payload) == {"global_step", "train_seconds", *destinations}
    assert payload["Performance/memory-cuda_peak_gib"] == _PROJECTED_GIB
    # One ID source feeds both pages.
    assert payload["Fields/material-id_validation-temperature"] == payload["Fields/parameter-id_validation-temperature"] == 1.0
    assert list(payload) == [
        "global_step",
        "train_seconds",
        *(definition.wandb_key for definition in definitions),
    ]  # projection order, not producer order


def _steady_loop_telemetry(config: dict[str, Any]) -> dict[str, float]:
    """Return the event names a physics-informed steady training loop emits, with telemetry W&B must not log."""
    names = [
        "train/loss_total",
        "train/loss_data",
        *(f"train/loss_data_{part}" for part in ("p", "velocity", "u", "v")),
        "optimization/learning_rate",
        "budget/progress",
        "budget/train_seconds",
        "budget/units",
        "validation/point",
        "material_ood/objective_difference",
        "system/event_duration_seconds",
        "system/train_duration_seconds",
        "system/train_samples_per_second",
        "system/session_elapsed_seconds",
        "checkpoint/new_best",
        "global_step",
    ]
    names += [f"{role}/{metric['id']}" for role in ("id", "material_ood") for metric in config["evaluation"]["metrics"]]
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

    # Only the material part is configured here, so no parameter-OOD row is defined or logged.
    session = tracking.initialize_wandb(config, run_dir=tmp_path, ood_roles=("material_ood",))

    initialization = fake.initializations[0]
    assert initialization["project"] == experiments.config.defaults.WANDB_TASK_PROJECTS["steady_flow"]
    assert initialization["group"] == f"{layout['family']}/{layout['config_name']}"
    assert (initialization["job_type"], initialization["name"], initialization["tags"]) == ("train", config["run"]["name"], ["pi-fno"])
    assert fake.runs[0].summary == {"data/train_dataset": "synthetic_train", "data/ood_datasets": ["synthetic_ood"]}

    session.log_event(1, _steady_loop_telemetry(config))

    payload, _step = fake.runs[0].logs[-1]
    keys = set(payload) - {"global_step", "train_seconds"}
    assert {key.partition("/")[0] for key in keys} == {"Overview", "Training", "Loss", "Residuals", "Fields", "Physical", "Norms", "Performance"}
    assert {key for key in keys if key.startswith(("Fields/", "Physical/", "Norms/"))} == {
        f"{section}/material-{label}-{name}"
        for section, names in (("Fields", ("p", "u", "v", "velocity")), ("Physical", "puv"), ("Norms", ("relative_l2", "relative_h1")))
        for label in ("id_validation", "ood")
        for name in names
    }
    terms = ("momentum", "continuity", "boundary")
    assert {key for key in keys if key.startswith(("Overview/", "Training/", "Loss/", "Residuals/", "Performance/"))} == {
        *(f"Overview/{name}" for name in ("objective-id_validation", "objective-material", "transfer_gap-material", "train-loss_total")),
        *(f"Training/{name}" for name in ("budget-progress", "learning_rate", "optimizer_steps")),
        *(f"Training/weight-{term}" for term in terms),
        *(f"Loss/data-{part}" for part in ("p", "total", "u", "v", "velocity")),
        *(f"Loss/physics-{term}" for term in terms),
        *(f"Residuals/val-{term}" for term in terms),
        *(f"Performance/{name}" for name in ("duration-event", "duration-train", "rate-units_per_second")),
    }
    # The local summary carries the counts in every terminal state, a pruned or failed run included.
    work = {"optimizer_steps": 3, "gradient_predictions": 6}
    session.finish(status="pruned", result=None, local_summary={"training_work": work})
    assert fake.runs[0].summary["training_work"] == work


def test_terminal_summary_copies_only_the_selected_metrics_the_history_projects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the objective and the projected series in the W&B summary; summary.json keeps the complete selected metrics locally."""
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    config = _resolved_config(physics_enabled=True)
    session = tracking.initialize_wandb(config, run_dir=tmp_path, ood_roles=("material_ood",))
    objective = f"selected/id/{config['evaluation']['objective']['id']}"
    selected = {
        objective: 0.1,
        "selected/id/normalized_rmse_p": 0.2,
        "selected/id/physical_velocity_vector_rmse": 0.3,
        "selected/physics/flux_balance_mse": 0.4,
    }

    session.finish(status="completed", result={"selected_metrics": selected})

    copied = {key: value for key, value in fake.runs[0].summary.items() if key.startswith("selected/")}
    assert copied == {key: value for key, value in selected.items() if key != "selected/id/physical_velocity_vector_rmse"}


def test_every_summary_only_key_is_declared_hidden_before_it_is_written(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End values reach W&B only through hidden metrics, so they make no panel and no section; history series stay visible."""
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    config = _trial_config(tmp_path, trial_number=1, epochs=2)
    semantic = {
        "task": {"id": "steady_flow", "contract_digest": "a"},
        "provenance": {"config_digest": "b"},
        "model": {"variant": "fno", "parameter_counts": {"total": 3, "trainable": 2}},
        "data": {"split": {"artifact_sha256": "c"}, "normalization": {"artifact_sha256": "d"}},
        "tuning": {"study_name": "s", "trial_number": 1, "sampled_parameters": {"lr": 0.1}},
    }
    session = tracking.initialize_wandb(config, run_dir=tmp_path, semantic_config=semantic, ood_roles=())
    run = fake.runs[0]
    hidden = [name for name, options in run.metric_definitions if options.get("hidden") and name not in {"global_step", "train_seconds"}]
    assert hidden == list(schema.SUMMARY_ONLY_KEYS)

    objective = config["evaluation"]["objective"]["id"]
    result = {
        "completed_epoch": 2,
        "global_step": 8,
        "selected_epoch": 1,
        "terminal_epoch": 2,
        "selected_metrics": {f"selected/id/{objective}": 0.1},
        "terminal_metrics": {"train/loss_total": 0.2},
        "training_outcome": {"status": "completed"},
    }
    local = {"training_work": {"optimizer_steps": 3}, "elapsed_seconds": 4.0, "best_checkpoint_sha256": "e" * 64}
    session.finish(status="completed", result=result, local_summary={**local, "runtime_sessions": [{"tracking": {"session_kind": "resume"}}]})

    def covered(key: str) -> bool:
        return any(key.startswith(name[:-1]) if name.endswith("*") else key == name for name in hidden)

    assert run.summary["run/outcome"] == "completed"
    assert {key for key in run.summary if not covered(key)} == set()
    assert not covered("Optuna/best_objective")


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

    def set_constraint(self, key: str, value: float) -> None:
        del key, value

    def should_prune(self) -> bool:
        return False


def _trial_config(tmp_path: Path, *, trial_number: int, epochs: int, revision: str | None = None) -> dict[str, Any]:
    """Return one prepared steady Optuna trial config of an artificial study with online tracking."""
    payload = configs.optuna_config()
    payload["experiment"]["data"]["train_dataset"] = "synthetic_train"
    payload["experiment"]["tracking"]["wandb"]["mode"] = "online"
    path = configs.write_yaml(tmp_path / "synthetic-study.yaml", payload)
    study = optuna_trials.with_configured_identities(experiments.tuning.optuna.load_optuna_study_config(path, epochs=epochs, revision=revision))
    config, _context = experiments.tuning.optuna._prepare_trial_config(study, _TrialStub(trial_number), sampler_seed=0)
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
            ood_roles=(),
        )
        for epochs, revision, _study_dir in studies
    ]

    for (_epochs, _revision, study_dir), initialization in zip(studies, fake.initializations, strict=True):
        assert initialization["group"] == f"optuna/{study_dir}"
        assert initialization["job_type"] == "optuna_trial"
        assert initialization["name"] == f"{study_dir}__trial-0003"
        assert initialization["tags"] == ["fno", "optuna"]
    assert len({initialization["name"] for initialization in fake.initializations}) == len(studies)

    sessions[0].log_event(
        1, {"id/normalized_group_macro_rmse": 0.5, "optuna/best_objective": 0.4, "validation/point": 1.0, "budget/train_seconds": 2.0}
    )
    payload, _step = fake.runs[0].logs[-1]
    assert payload["Overview/objective-id_validation"] == 0.5
    assert payload["Optuna/best_objective"] == 0.4
    summaries = {name: options["summary"] for name, options in fake.runs[0].metric_definitions if "summary" in options}
    assert summaries["Optuna/best_objective"] is None  # SDK default: the summary keeps the history's last value
    assert summaries["Overview/objective-id_validation"] == "none"

    sessions[0].finish(status="completed", result={"best_metric": 0.4, "training_outcome": {"status": "completed"}})
    terminal = fake.runs[0].summary
    assert terminal["Optuna/trial_number"] == 3
    assert "Optuna/best_objective" not in terminal  # only the history aggregate defines it
    # A run that completed with a checkpoint selected before its divergence is a pruned trial of its study.
    sessions[1].finish(status="completed", result={"best_metric": 0.4, "training_outcome": {"status": "diverged"}})
    assert [(run.summary["Optuna/state"], run.summary["Optuna/pruned"]) for run in fake.runs[:2]] == [("completed", False), ("diverged", True)]


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
        ood_roles=(),
    )
    fake.runs[0].fail_log = True
    with pytest.raises(tracking.TrackingIOError, match=rf"{mode} W&B history"):
        session.log_event(1, {"train/loss_total": 1.0, "budget/train_seconds": 2.0})
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
            ood_roles=(),
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
        ood_roles=(),
    )
    finish_fake.runs[0].fail_finish = True
    with pytest.raises(tracking.TrackingIOError, match="finish"):
        session.finish(status="completed")
    assert finish_state["status"] == "failed"
    assert finish_state["failed_operation"] == "finish"


def test_persisted_identity_requires_one_run_id_and_uses_latest_step() -> None:
    """Recover one run identity and its latest logged optimizer step and validation point."""
    summary = {
        "runtime_sessions": [
            {"tracking": {"wandb_run_id": "same", "last_logged_step": 2, "last_logged_point": 1}},
            {"tracking": {"wandb_run_id": "same", "last_logged_step": 5}},
        ]
    }
    assert tracking.persisted_wandb_identity(summary) == ("same", 5, 1)
    summary["runtime_sessions"].append({"tracking": {"wandb_run_id": "different"}})
    with pytest.raises(tracking.TrackingInitializationError, match="found 2"):
        tracking.persisted_wandb_identity(summary)


@pytest.mark.parametrize(
    ("settings", "match"),
    [
        ({"mode": "invalid"}, r"tracking\.wandb\.mode"),
        ({"monitor": {"max_cases": 0}}, r"tracking\.wandb\.monitor\.max_cases"),
        ({"workflow": "train", "study": "not-allowed"}, r"tracking\.wandb.*study"),
    ],
)
def test_wandb_config_rejects_invalid_values(
    settings: dict[str, Any],
    match: str,
) -> None:
    """Reject malformed tracking settings through the public resolver."""
    raw = configs.direct_config()
    raw["tracking"]["wandb"].update(settings)
    with pytest.raises(ValueError, match=match):
        configs.resolve(raw)
