"""Verify the artifact build flow both tasks share: staging, timing, admission and publication.

Responsibilities:
  - Check that a timing failure publishes nothing and keeps the deterministic stage
  - Check that a complete stage is reused without generation and admitted before publication
  - Check that a complete target or stage of other analysis code is refused before any change, and rebuilt on request
  - Check that one role validator refuses a corrupted steady payload on write, cache hit and read
  - Check that a missing target is generated beside another published role only when both come from the running analysis code
  - Check that a rebuild names every role of other analysis code and publishes its roles all or nothing, so a run keeps one producer
  - Check that a read refuses one run of two producers, that a rollback attempts every target, and that a build refuses
    the backup of an interrupted publication
  - Check that the sibling check and the publication of a target hold the run-wide artifact lock
  - Check that preparation only reads: published roles are evidence, and a missing held-out role is refused with the
    artifact command that releases exactly that role

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


def _run(run_dir: Path) -> pd.DataFrame:
    """Run the shared locked flow for the ID role."""
    return service._run_or_load_artifacts_locked(
        run_dir=run_dir,
        dataset_name="dataset",
        split="eval",
        device_resolution=cast("Any", SimpleNamespace(device=torch.device("cpu"))),
        dataset_root=run_dir / "datasets",
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

    plan = SimpleNamespace(run_dir=run_dir, task=task, roles=(_ID_ROLE,))
    frames = service._rebuild_run_roles(
        plan=cast("Any", plan),
        roles=("id",),
        device_resolution=cast("Any", SimpleNamespace(device=torch.device("cpu"))),
        dataset_root=run_dir / "datasets",
    )

    assert generated == [target.parent / ".id.stage"]
    assert frames["eval"].attrs["artifact_provenance"]["producer"] == {"analysis_sha256": _RUNNING_ANALYSIS}
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


_OTHER_ROLE = SimpleNamespace(dataset_name="other", split="material_ood", dataset_role="material_ood")


def _two_role_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, task: str, generate: Callable[[Path, Any], None]) -> Path:
    """Fake one run with an ID and a material-OOD role and return its directory."""
    _fake_task(monkeypatch, task, generate)
    plan = SimpleNamespace(
        ood_roles={"material_ood": _OTHER_ROLE},
        roles=(_ID_ROLE, _OTHER_ROLE),
        role=lambda split: _ID_ROLE if split == "eval" else _OTHER_ROLE,
    )
    monkeypatch.setattr(service.artifact_plan, "artifact_plan_from_completed", lambda *_args, **_kwargs: plan)
    _recording_admission(monkeypatch)
    return tmp_path.resolve() / "run"


def _targets(run_dir: Path) -> tuple[Path, Path]:
    """Return the ID and material-OOD targets of the two-role run."""
    return (
        common.paths.resolve_id_analysis_dir(run_dir),
        service._artifact_save_root(run_dir=run_dir, dataset_name="other", split="material_ood"),
    )


def _producers(run_dir: Path) -> list[str]:
    """Return the recorded analysis producer of each published target of the two-role run."""
    markers = (contracts.artifact_provenance_path(target) for target in _targets(run_dir))
    return [json.loads(marker.read_text(encoding="utf-8"))["producer"]["analysis_sha256"] for marker in markers]


def _rebuild(run_dir: Path, task: str, roles: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """Rebuild the named roles of the two-role run through the run-level rebuild."""
    plan = SimpleNamespace(run_dir=run_dir, task=task, roles=(_ID_ROLE, _OTHER_ROLE))
    return service._rebuild_run_roles(
        plan=cast("Any", plan),
        roles=cast("Any", roles),
        device_resolution=cast("Any", SimpleNamespace(device=torch.device("cpu"))),
        dataset_root=run_dir / "datasets",
    )


@pytest.mark.parametrize("task", _TASKS)
def test_a_missing_target_beside_a_role_of_other_analysis_code_is_refused_before_inference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, task: str
) -> None:
    """Generating beside a published role of other analysis code would give the run two producers: refused before inference."""
    run_dir = _two_role_run(monkeypatch, tmp_path, task, lambda _stage, _reporter: pytest.fail("generated beside a stale role"))
    _write_artifact(_targets(run_dir)[1], analysis_sha256=_OTHER_ANALYSIS)
    before = _tree_bytes(run_dir)

    with pytest.raises(service.ArtifactCacheError, match=r"two analysis producers.*--split material_ood --rebuild"):
        _run(run_dir)

    assert _tree_bytes(run_dir) == before


@pytest.mark.parametrize("task", _TASKS)
def test_a_partial_rebuild_beside_a_stale_role_is_refused_with_the_command_naming_every_affected_role(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, task: str
) -> None:
    """Rebuilding only ID while material OOD records other analysis code would leave two producers: refused before inference."""
    run_dir = _two_role_run(monkeypatch, tmp_path, task, lambda _stage, _reporter: pytest.fail("a partial rebuild reached inference"))
    for target in _targets(run_dir):
        _write_artifact(target, analysis_sha256=_OTHER_ANALYSIS)
    before = _tree_bytes(run_dir)

    command = rf"--run-dir {run_dir} --split id --split material_ood --rebuild"
    with pytest.raises(service.ArtifactCacheError, match=rf"two analysis producers[\s\S]*{command}"):
        _rebuild(run_dir, task, ("id",))

    assert _tree_bytes(run_dir) == before


@pytest.mark.parametrize("task", _TASKS)
def test_a_rebuild_naming_every_stale_role_leaves_one_producer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, task: str) -> None:
    """Every stale role named: both are regenerated and the run records only the running analysis code."""
    generated: list[Path] = []

    def generate(stage: Path, reporter: Any) -> None:
        generated.append(stage)
        _write_artifact(stage, reporter)

    run_dir = _two_role_run(monkeypatch, tmp_path, task, generate)
    for target in _targets(run_dir):
        _write_artifact(target, analysis_sha256=_OTHER_ANALYSIS)

    frames = _rebuild(run_dir, task, ("id", "material_ood"))

    assert set(frames) == {"eval", "material_ood"}
    assert len(generated) == len(_targets(run_dir))
    assert _producers(run_dir) == [_RUNNING_ANALYSIS, _RUNNING_ANALYSIS]


@pytest.mark.parametrize("task", _TASKS)
@pytest.mark.parametrize("failure", ["generation", "publication"])
def test_a_failed_rebuild_publishes_no_role_beside_its_stale_siblings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, task: str, failure: str
) -> None:
    """All or nothing per run: when one named role fails to stage or publish, no new artifact replaces any target."""

    def generate(stage: Path, reporter: Any) -> None:
        if failure == "generation" and stage.name.startswith(".other"):
            msg = "material-OOD inference failed"
            raise RuntimeError(msg)
        _write_artifact(stage, reporter)

    run_dir = _two_role_run(monkeypatch, tmp_path, task, generate)
    id_target, ood_target = _targets(run_dir)
    for target in (id_target, ood_target):
        _write_artifact(target, analysis_sha256=_OTHER_ANALYSIS)
    before = {target: _tree_bytes(target) for target in (id_target, ood_target)}
    if failure == "publication":
        replace = type(ood_target).replace

        def refuse_second_publication(self: Path, target: Path) -> Path:
            if self.name.startswith(".other") and self.name.endswith(".stage"):
                msg = "rename refused"
                raise OSError(msg)
            return replace(self, target)

        monkeypatch.setattr(type(ood_target), "replace", refuse_second_publication)

    with pytest.raises((RuntimeError, OSError)):
        _rebuild(run_dir, task, ("id", "material_ood"))

    assert {target: _tree_bytes(target) for target in (id_target, ood_target)} == before
    assert _producers(run_dir) == [_OTHER_ANALYSIS, _OTHER_ANALYSIS]


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
    publish = service._publish_staged_artifacts

    def check(**kwargs: Any) -> Any:
        held["sibling_check"] = common.locking.file_lock_is_active(run_lock)
        return sibling_check(**kwargs)

    def publish_held(**kwargs: Any) -> None:
        held["publication"] = common.locking.file_lock_is_active(run_lock)
        publish(**kwargs)

    monkeypatch.setattr(service, "_sibling_from_other_analysis", check)
    monkeypatch.setattr(service, "_publish_staged_artifacts", publish_held)
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


@pytest.mark.parametrize("role", ["id", "material_ood"])
def test_preparation_never_builds_and_names_the_artifact_command_of_exactly_the_missing_role(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
) -> None:
    """Preparation only reads: a missing held-out role is refused with the explicit artifact command that releases that role alone."""
    run_dir = tmp_path.resolve()
    split = cast("Any", _ROLE_SPLITS[role])

    def missing(_path: Path, **_kwargs: Any) -> loader.LoadedRunArtifacts:
        raise loader._missing(task="steady_flow", run_name="run", run_dir=run_dir, role=split, root=run_dir / "analysis")

    monkeypatch.setattr(loader, "load_run_artifacts", missing)
    monkeypatch.setattr(service, "run_or_load_artifacts", lambda **_kwargs: pytest.fail("a preparation built an artifact"))

    with pytest.raises(loader.MissingEvaluationArtifactsError) as refused:
        service.load_prepared_run_artifacts(run_dir)

    assert str(refused.value).endswith(f"artifacts --run-dir {run_dir} --split {role}")


def test_preparation_reads_roles_of_other_analysis_code_as_evidence_and_never_rebuilds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Published roles are historical evidence: preparation reads them whatever code produced them and replaces nothing."""
    run_dir = tmp_path.resolve()
    monkeypatch.setitem(contracts.ANALYSIS_PROJECTIONS, "steady_flow", SimpleNamespace(name="analysis_steady_flow", sha256=lambda: _RUNNING_ANALYSIS))

    def artifact() -> Any:
        frame = pd.DataFrame()
        frame.attrs["artifact_provenance"] = {"producer": {"analysis_sha256": _OTHER_ANALYSIS}}
        return SimpleNamespace(frame=frame)

    loaded = loader.LoadedRunArtifacts(task="steady_flow", run_name="run", run_dir=run_dir, id_artifact=artifact(), material_ood_artifact=artifact())
    monkeypatch.setattr(loader, "load_run_artifacts", lambda _path, **_kwargs: loaded)
    monkeypatch.setattr(service, "run_or_load_artifacts", lambda **_kwargs: pytest.fail("a preparation rebuilt an artifact"))

    prepared = service.load_prepared_run_artifacts(run_dir)

    assert prepared.loaded_run is loaded
    assert prepared.role_actions == {"id": "reused", "material_ood": "reused"}


def test_a_read_of_one_run_whose_roles_carry_two_producers_is_refused_with_the_rebuild_of_every_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash after one publication, or a read across a publication, mixes producers within one run: refused on read."""
    run_dir = tmp_path.resolve() / "run"
    producers = {"id": _RUNNING_ANALYSIS, "material_ood": _OTHER_ANALYSIS}

    def load_role(*, role: str, **_kwargs: Any) -> Any:
        frame = pd.DataFrame()
        frame.attrs["artifact_provenance"] = {"producer": {"analysis_sha256": producers[role]}}
        return SimpleNamespace(frame=frame)

    monkeypatch.setattr(loader.experiments.config.loader, "validate_resolved_task_contract", lambda _config: SimpleNamespace(id="steady_flow"))
    monkeypatch.setattr(loader, "_steady_role_context", lambda *_args: ({}, {role: role for role in producers}))
    monkeypatch.setattr(loader, "_load_role", load_role)
    command = rf"--run-dir {run_dir} --split id --split material_ood --rebuild"

    with pytest.raises(loader.IncompatibleEvaluationArtifactsError, match=rf"two analysis producers[\s\S]*{command}"):
        loader._load_admitted_run_artifacts(run_dir, {"summary": {}, "config": {}})


def test_a_failed_rollback_still_restores_every_other_target_and_names_the_unrestored_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Rollback attempts every target: one failed restore does not stop the others, and the error names what stayed unrestored."""
    run_dir = tmp_path.resolve() / "run"
    targets = (
        common.paths.resolve_id_analysis_dir(run_dir),
        service._artifact_save_root(run_dir=run_dir, dataset_name="first", split="material_ood"),
        service._artifact_save_root(run_dir=run_dir, dataset_name="second", split="parameter_ood"),
    )
    stages = tuple(contracts.artifact_stage_path(target) for target in targets)
    for target, stage in zip(targets, stages, strict=True):
        _write_artifact(target, analysis_sha256=_OTHER_ANALYSIS)
        _write_artifact(stage)
    before = {target: _tree_bytes(target) for target in targets}
    replace = type(run_dir).replace

    def failing_replace(self: Path, destination: Path) -> Path:
        if self == stages[2] or (self == targets[1] and destination == stages[1]):
            msg = f"rename refused: {self}"
            raise OSError(msg)
        return replace(self, destination)

    monkeypatch.setattr(type(run_dir), "replace", failing_replace)

    with pytest.raises(service.ArtifactCacheError, match=rf"left unrestored[\s\S]*{targets[1]}") as refused:
        service._publish_staged_artifacts(run_dir=run_dir, stages=tuple(zip(targets, stages, strict=True)))

    assert str(targets[0]) not in str(refused.value)
    assert _tree_bytes(targets[0]) == before[targets[0]]
    assert _tree_bytes(targets[2]) == before[targets[2]]


def test_a_build_beside_the_backup_of_an_interrupted_publication_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A backup left by a crash between two renames may hold the only old copy: the next build refuses and names it."""
    run_dir = tmp_path.resolve() / "run"
    leftover = common.paths.resolve_analysis_root(run_dir) / ".id.backup.0123abcd"
    _write_artifact(leftover, analysis_sha256=_OTHER_ANALYSIS)
    _fake_task(monkeypatch, "steady_flow", lambda _stage, _reporter: pytest.fail("a build ran beside an interrupted publication"))
    before = _tree_bytes(run_dir)

    with pytest.raises(service.ArtifactCacheError, match=rf"interrupted publication[\s\S]*{leftover}"):
        service.run_or_load_artifacts(
            run_dir=run_dir,
            dataset_name="dataset",
            split="eval",
            device_resolution=cast("Any", SimpleNamespace(device=torch.device("cpu"))),
            dataset_root=run_dir / "datasets",
        )

    assert _tree_bytes(run_dir) == before
