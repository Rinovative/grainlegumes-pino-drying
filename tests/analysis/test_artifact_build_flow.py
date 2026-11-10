"""Verify the artifact build flow both tasks share: staging, timing, admission and publication.

Responsibilities:
  - Check that a timing failure publishes nothing and keeps the deterministic stage
  - Check that a complete stage is reused without generation and admitted before publication
  - Check that a complete target or stage of other analysis code is refused before any change, and rebuilt on request
  - Check that one role validator refuses a corrupted steady payload on write, cache hit and read
  - Check that a missing target is generated beside another published role only when both come from the running analysis code
  - Check that the sibling check and the publication of a target hold the run-wide artifact lock
  - Check that the prepared artifacts of one run share one analysis producer, read as evidence when they do

Design principles:
  - Task writers are explicit fakes; the shared flow, stage handling and role validator run unchanged
  - The running analysis projection is a stub with a fixed digest, so recorded and running code differ explicitly

This module does NOT:
  - Run inference, COMSOL or real Dataset packages
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
import pytest
import torch

from src import common, experiments
from src.analysis.artifacts import analysis_artifact_contracts as contracts
from src.analysis.artifacts import analysis_artifact_plan as artifact_plan
from src.analysis.artifacts import analysis_artifact_service as service
from src.analysis.evaluation import evaluation_artifact_loader as loader

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

_TASKS = ("steady_flow", "transient_drying")
_CASE_ID = "lentil__case_0001__000000000001"
_ID_ROLE = artifact_plan.ArtifactRolePlan(
    split="eval",
    dataset_role="id",
    dataset_name="dataset",
    source_dataset_ids=("dataset",),
    source_identities=({"case_ids": [_CASE_ID]},),
    case_ids_by_source=((_CASE_ID,),),
    membership_digests=("m" * 64,),
)
_RUNNING_ANALYSIS = "c" * 64
_OTHER_ANALYSIS = "0" * 64


def _write_artifact(root: Path, _reporter: Any = None, *, analysis_sha256: str = _RUNNING_ANALYSIS) -> None:
    """Write one tiny artifact whose completion marker binds its producer and payload digests."""
    (root / "npz").mkdir(parents=True)
    (root / "artifacts.parquet").write_bytes(b"table")
    (root / "npz" / "case.npz").write_bytes(b"payload")
    marker = {"producer": {"analysis_sha256": analysis_sha256}, "outputs": contracts.artifact_output_manifest(root)}
    contracts.artifact_provenance_path(root).write_text(json.dumps(marker), encoding="utf-8")


def _tree_bytes(root: Path) -> dict[str, bytes]:
    """Return every file below ``root`` with its bytes, to prove that a refusal changed nothing."""
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def _corrupted_artifact(root: Path, _reporter: Any = None) -> None:
    """Write the tiny artifact, then change its table behind its marker (case payloads are verified when a case opens)."""
    _write_artifact(root)
    (root / "artifacts.parquet").write_bytes(b"corrupted")


def _fake_task(
    monkeypatch: pytest.MonkeyPatch,
    task: str,
    generate: Callable[[Path, Any], None],
) -> None:
    """Replace run admission, the task writer and the running analysis digest; the shared flow stays real."""
    monkeypatch.setitem(contracts.ANALYSIS_PROJECTIONS, task, SimpleNamespace(name=f"analysis_{task}", sha256=lambda: _RUNNING_ANALYSIS))
    completed = {"config": {}, "summary": {}, "scientific_run_name": "run"}
    monkeypatch.setattr(service.experiments.run, "validate_completed_run", lambda _run_dir: completed)
    monkeypatch.setattr(experiments.config.loader, "validate_resolved_task_contract", lambda _config: SimpleNamespace(id=task))
    plan = SimpleNamespace(role=lambda _split: _ID_ROLE, ood_roles={}, roles=(_ID_ROLE,))
    monkeypatch.setattr(service.artifact_plan, "artifact_plan_from_completed", lambda *_args, **_kwargs: plan)
    build = service._TargetBuild(case_ids=(_CASE_ID,), generate=generate, finish_stage=lambda _stage: None)
    monkeypatch.setattr(service, "_transient_target_build" if task == "transient_drying" else "_steady_target_build", lambda **_kwargs: build)


def _recording_admission(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Admit every root with its recorded provenance, as the role validator does, and record the order in which the flow asked."""
    admitted: list[Path] = []

    def admit(_run_dir: Path, _completed: Any, *, root: Path, **_kwargs: Any) -> Any:
        admitted.append(root)
        frame = pd.DataFrame()
        frame.attrs["artifact_provenance"] = json.loads(contracts.artifact_provenance_path(root).read_text(encoding="utf-8"))
        return SimpleNamespace(frame=frame)

    monkeypatch.setattr(loader, "admit_role_artifact", admit)
    monkeypatch.setattr(service, "_report_admitted_target", lambda _frame: None)
    return admitted


def _run(run_dir: Path, *, rebuild: bool = False) -> pd.DataFrame:
    """Run the shared locked flow for the ID role."""
    return service._run_or_load_artifacts_locked(
        run_dir=run_dir,
        dataset_name="dataset",
        split="eval",
        device_resolution=cast("Any", SimpleNamespace(device=torch.device("cpu"))),
        dataset_root=run_dir / "datasets",
        rebuild=rebuild,
    )


@pytest.mark.parametrize("task", _TASKS)
def test_timing_failure_publishes_nothing_and_keeps_the_stage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, task: str) -> None:
    """Timing is written into the stage before validation, so its failure leaves no published target."""
    run_dir = tmp_path.resolve() / "run"
    target = common.paths.resolve_id_analysis_dir(run_dir)
    _fake_task(monkeypatch, task, _write_artifact)
    admitted = _recording_admission(monkeypatch)

    def fail(*_args: Any, **_kwargs: Any) -> None:
        msg = "timing evidence is incomplete"
        raise ValueError(msg)

    monkeypatch.setattr(service.timing, "write_stage_timing", fail)
    with pytest.raises(ValueError, match="timing evidence"):
        _run(run_dir)
    assert not target.exists()
    assert contracts.artifact_provenance_path(target.parent / ".id.stage").is_file()
    assert admitted == []


@pytest.mark.parametrize("task", _TASKS)
def test_complete_stage_is_reused_without_generation_and_admitted_before_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    task: str,
) -> None:
    """A stage kept from a failed attempt is admitted, timed and published without a new forward pass."""
    run_dir = tmp_path.resolve() / "run"
    target = common.paths.resolve_id_analysis_dir(run_dir)
    stage = target.parent / ".id.stage"
    _write_artifact(stage)
    _fake_task(monkeypatch, task, lambda _stage, _reporter: pytest.fail("a complete stage was generated again"))
    admitted = _recording_admission(monkeypatch)

    _run(run_dir)

    assert admitted == [stage, target]
    assert not stage.exists()
    counts = json.loads(contracts.artifact_provenance_path(target).read_text(encoding="utf-8"))["timing"]["operational_performance"]["counts"]
    assert counts["reused_case_count"] == 1
    assert counts["forward_call_count"] == 0


@pytest.mark.parametrize("task", _TASKS)
def test_complete_target_of_the_running_analysis_code_is_reused_without_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    task: str,
) -> None:
    """A requested target already in place with the running analysis digest is admitted and returned unchanged."""
    run_dir = tmp_path.resolve() / "run"
    target = common.paths.resolve_id_analysis_dir(run_dir)
    _write_artifact(target)
    before = _tree_bytes(run_dir)
    _fake_task(monkeypatch, task, lambda _stage, _reporter: pytest.fail("a fresh complete target was generated again"))
    admitted = _recording_admission(monkeypatch)

    frame = _run(run_dir)

    assert frame.attrs["artifact_provenance"]["producer"] == {"analysis_sha256": _RUNNING_ANALYSIS}
    assert admitted == [target]
    assert _tree_bytes(run_dir) == before


@pytest.mark.parametrize("task", _TASKS)
@pytest.mark.parametrize("location", ["target", "stage"])
def test_complete_target_or_stage_of_other_analysis_code_is_refused_before_any_change(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    task: str,
    location: str,
) -> None:
    """Requested-result freshness: a recorded analysis digest that differs from the running code refuses reuse with the rebuild remedy."""
    run_dir = tmp_path.resolve() / "run"
    target = common.paths.resolve_id_analysis_dir(run_dir)
    _write_artifact(target if location == "target" else target.parent / ".id.stage", analysis_sha256=_OTHER_ANALYSIS)
    before = _tree_bytes(run_dir)
    _fake_task(monkeypatch, task, lambda _stage, _reporter: pytest.fail("a stale complete result was regenerated without --rebuild"))
    admitted = _recording_admission(monkeypatch)

    changed = rf"analysis_{task} implementation changed \(recorded 000000000000, current cccccccccccc\); rerun only this exact target with --rebuild"
    with pytest.raises(ValueError, match=changed):
        _run(run_dir)

    assert _tree_bytes(run_dir) == before
    assert admitted == ([target] if location == "target" else [])


@pytest.mark.parametrize("task", _TASKS)
def test_rebuild_replaces_a_target_of_other_analysis_code(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, task: str) -> None:
    """With --rebuild the stale target and stage are regenerated and the published target records the running digest."""
    run_dir = tmp_path.resolve() / "run"
    target = common.paths.resolve_id_analysis_dir(run_dir)
    _write_artifact(target, analysis_sha256=_OTHER_ANALYSIS)
    _write_artifact(target.parent / ".id.stage", analysis_sha256=_OTHER_ANALYSIS)
    generated: list[Path] = []

    def generate(stage: Path, reporter: Any) -> None:
        generated.append(stage)
        _write_artifact(stage, reporter)

    _fake_task(monkeypatch, task, generate)
    _recording_admission(monkeypatch)

    frame = _run(run_dir, rebuild=True)

    assert generated == [target.parent / ".id.stage"]
    assert frame.attrs["artifact_provenance"]["producer"] == {"analysis_sha256": _RUNNING_ANALYSIS}
    stored = json.loads(contracts.artifact_provenance_path(target).read_text(encoding="utf-8"))
    assert stored["producer"] == {"analysis_sha256": _RUNNING_ANALYSIS}


@pytest.mark.parametrize("path", ["write", "cache_hit", "read"])
def test_one_validator_refuses_a_corrupted_steady_payload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, path: str) -> None:
    """Stage admission, cache hits and reads share the role validator, so each refuses a payload changed behind its marker."""
    run_dir = tmp_path.resolve() / "run"
    target = common.paths.resolve_id_analysis_dir(run_dir)
    role = loader._SavedRole(plan=_ID_ROLE, root=target)
    context = {
        "task": "steady_flow",
        "run_name": "run",
        "run_dir": run_dir,
        "config": {},
        "summary": {},
        "normalizer_state": {},
        "task_spec": SimpleNamespace(id="steady_flow"),
        "evaluation": {},
    }
    monkeypatch.setattr(loader, "_steady_role_context", lambda *_args: (context, {"id": role}))
    _fake_task(monkeypatch, "steady_flow", _corrupted_artifact)
    if path != "write":
        _corrupted_artifact(target)
    if path == "read":

        @contextmanager
        def lease(_path: Path) -> Iterator[dict[str, Any]]:
            yield {"config": {}, "summary": {}}

        monkeypatch.setattr(loader.experiments.run, "saved_artifact_run_lease", lease)
        with pytest.raises(loader.IncompatibleEvaluationArtifactsError, match="output manifest"):
            loader.load_run_artifacts(run_dir)
        return
    with pytest.raises(service.ArtifactCacheError, match=r"output manifest[\s\S]*--rebuild"):
        _run(run_dir)
    if path == "write":
        assert not target.exists()
        assert (target.parent / ".id.stage").is_dir()


@pytest.mark.parametrize("task", _TASKS)
@pytest.mark.parametrize("rebuild", [False, True])
def test_a_missing_target_beside_a_role_of_other_analysis_code_is_generated_only_on_rebuild(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, task: str, *, rebuild: bool
) -> None:
    """Generating beside a published role of other analysis code would give the run two producers: refused before inference unless rebuilt."""
    run_dir = tmp_path.resolve() / "run"
    generated: list[Path] = []

    def generate(stage: Path, reporter: Any) -> None:
        generated.append(stage)
        _write_artifact(stage, reporter)

    _fake_task(monkeypatch, task, generate)
    other_role = SimpleNamespace(dataset_name="other", split="material_ood")
    plan = SimpleNamespace(
        ood_roles={"material_ood": other_role}, roles=(_ID_ROLE, other_role), role=lambda split: _ID_ROLE if split == "eval" else other_role
    )
    monkeypatch.setattr(service.artifact_plan, "artifact_plan_from_completed", lambda *_args, **_kwargs: plan)
    sibling_root = service._artifact_save_root(run_dir=run_dir, dataset_name="other", split="material_ood")
    _write_artifact(sibling_root, analysis_sha256=_OTHER_ANALYSIS)
    _recording_admission(monkeypatch)
    before = _tree_bytes(run_dir)

    if not rebuild:
        with pytest.raises(service.ArtifactCacheError, match=r"two analysis producers.*--rebuild"):
            _run(run_dir)
        assert generated == []
        assert _tree_bytes(run_dir) == before
        return
    frame = _run(run_dir, rebuild=True)
    assert frame.attrs["artifact_provenance"]["producer"] == {"analysis_sha256": _RUNNING_ANALYSIS}


@pytest.mark.parametrize("task", _TASKS)
def test_the_sibling_check_and_publication_hold_the_run_wide_artifact_lock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, task: str) -> None:
    """Builders of both roles of one run share one lock, so a concurrent builder of other analysis code cannot pass the sibling check unseen."""
    run_dir = tmp_path.resolve() / "run"
    run_dir.mkdir()
    _fake_task(monkeypatch, task, _write_artifact)
    _recording_admission(monkeypatch)
    run_lock = common.paths.resolve_artifact_lock_path(common.paths.resolve_analysis_root(run_dir))
    held: dict[str, bool] = {}
    sibling_check = service._sibling_from_other_analysis
    publish = service._publish_staged_artifact

    def check(**kwargs: Any) -> Any:
        held["sibling_check"] = common.locking.file_lock_is_active(run_lock)
        return sibling_check(**kwargs)

    def publish_held(**kwargs: Any) -> None:
        held["publication"] = common.locking.file_lock_is_active(run_lock)
        publish(**kwargs)

    monkeypatch.setattr(service, "_sibling_from_other_analysis", check)
    monkeypatch.setattr(service, "_publish_staged_artifact", publish_held)
    service.run_or_load_artifacts(
        run_dir=run_dir,
        dataset_name="dataset",
        split="eval",
        device_resolution=cast("Any", SimpleNamespace(device=torch.device("cpu"))),
        dataset_root=run_dir / "datasets",
    )

    assert held == {"sibling_check": True, "publication": True}
    assert not common.locking.file_lock_is_active(run_lock)


_ROLE_SPLITS = {"id": "eval", "material_ood": "material_ood"}


@pytest.mark.parametrize(
    ("recorded", "rebuild_incompatible", "expected_builds", "expected_actions"),
    [
        pytest.param(
            {"id": _OTHER_ANALYSIS, "material_ood": _OTHER_ANALYSIS},
            False,
            [],
            {"id": "reused", "material_ood": "reused"},
            id="one-older-producer-is-read",
        ),
        pytest.param({"id": _RUNNING_ANALYSIS, "material_ood": _OTHER_ANALYSIS}, False, [], None, id="mixed-refused"),
        pytest.param(
            {"id": _RUNNING_ANALYSIS, "material_ood": _OTHER_ANALYSIS},
            True,
            [("material_ood", True)],
            {"id": "reused", "material_ood": "rebuilt"},
            id="mixed-rebuilt",
        ),
        pytest.param({"id": _OTHER_ANALYSIS, "material_ood": None}, False, [], None, id="missing-beside-older-refused"),
        pytest.param(
            {"id": _OTHER_ANALYSIS, "material_ood": None},
            True,
            [("id", True), ("material_ood", False)],
            {"id": "rebuilt", "material_ood": "generated"},
            id="missing-beside-older-rebuilds-it-first",
        ),
        pytest.param(
            {"id": None, "material_ood": _RUNNING_ANALYSIS},
            False,
            [("id", False)],
            {"id": "generated", "material_ood": "reused"},
            id="missing-beside-running-is-generated",
        ),
    ],
)
def test_prepared_run_artifacts_share_one_analysis_producer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recorded: dict[str, str | None],
    *,
    rebuild_incompatible: bool,
    expected_builds: list[tuple[str, bool]],
    expected_actions: dict[str, str] | None,
) -> None:
    """Roles of one producer are read as evidence; a preparation that would leave two producers rebuilds on request, else refuses before a build."""
    run_dir = tmp_path.resolve()
    monkeypatch.setitem(contracts.ANALYSIS_PROJECTIONS, "steady_flow", SimpleNamespace(name="analysis_steady_flow", sha256=lambda: _RUNNING_ANALYSIS))
    plan = SimpleNamespace(
        task="steady_flow",
        dataset_roles=("id", "material_ood"),
        roles=tuple(SimpleNamespace(split=split, dataset_name=f"{split}-dataset") for split in ("eval", "material_ood")),
        role=lambda split: SimpleNamespace(dataset_name=f"{split}-dataset"),
    )
    monkeypatch.setattr(service, "load_run_artifact_plan", lambda _path: plan)

    def publish(role: str, digest: str) -> None:
        recorded[role] = digest
        split = cast("Any", _ROLE_SPLITS[role])
        marker = contracts.artifact_provenance_path(service._artifact_save_root(run_dir=run_dir, dataset_name=f"{split}-dataset", split=split))
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"producer": {"analysis_sha256": digest}}), encoding="utf-8")

    for role, digest in tuple(recorded.items()):
        if digest is not None:
            publish(role, digest)

    def artifact(role: str) -> SimpleNamespace:
        frame = pd.DataFrame()
        frame.attrs["artifact_provenance"] = {"producer": {"analysis_sha256": recorded[role]}}
        return SimpleNamespace(frame=frame)

    def load(_path: Path, **_kwargs: Any) -> loader.LoadedRunArtifacts:
        for role, digest in recorded.items():
            if digest is None:
                msg = f"{role} is missing"
                raise loader.MissingEvaluationArtifactsError(msg, role=cast("Any", _ROLE_SPLITS[role]))
        return loader.LoadedRunArtifacts(
            task="steady_flow",
            run_name="run",
            run_dir=run_dir,
            id_artifact=cast("Any", artifact("id")),
            material_ood_artifact=cast("Any", artifact("material_ood")),
        )

    monkeypatch.setattr(loader, "load_run_artifacts", load)
    builds: list[tuple[str, bool]] = []

    def build(**kwargs: Any) -> None:
        role = "id" if kwargs["split"] == "eval" else "material_ood"
        builds.append((role, kwargs["rebuild"]))
        publish(role, _RUNNING_ANALYSIS)

    monkeypatch.setattr(service, "run_or_load_artifacts", build)

    def prepare() -> service.PreparedRunArtifacts:
        return service.load_or_build_run_artifacts(
            run_dir, rebuild_incompatible=rebuild_incompatible, device_resolution=experiments.run.learning.device.resolve_device("cpu")
        )

    if expected_actions is None:
        with pytest.raises(service.ArtifactCacheError, match=r"two analysis producers.*--rebuild"):
            prepare()
        assert builds == []
        return
    prepared = prepare()
    assert prepared.role_actions == expected_actions
    assert builds == expected_builds
