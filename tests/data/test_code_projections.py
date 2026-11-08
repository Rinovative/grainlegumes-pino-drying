"""Protect the declared code projections that decide scientific implementation identity.

Responsibilities:
  - Check that every declared projection resolves, and that every static import of every member is classified
  - Check that an unclassified import is reported by file and line, on synthetic trees and on a copy of the real tree
  - Check that package-alias resolution lets a runtime binding win over a TYPE_CHECKING alias and reports contradictory runtime bindings
  - Check member resolution including package initializers, the syntax-only digest, composition, and the task separation of the training projections
  - Check that the model projection follows fit and application code while loss, metric and lifecycle edits spare it
  - Check that the analysis projection includes the model projection of its task and follows metric and artifact code
    while training-only edits and the other task spare it
  - Check that the owners compute their identities from their declared projections

Design principles:
  - Classification is verified from the source syntax and never by importing or executing members
  - Projection fixtures are tiny test-owned trees; the real tree is only read, or copied for the fail-closed check
  - A scientific dependency can no longer be omitted silently: a new import must be added to a projection or exempted with a reason

This module does NOT:
  - Prove that a classification is scientifically right; the projection owners declare it
  - Test the Dataset, checkpoint, Generation or Optuna lifecycles that consume the digests
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from support import import_closure

from src import common, datasets, generation, learning
from src.analysis.artifacts import analysis_artifact_contracts as artifact_contracts
from src.experiments.tuning import experiments_tuning_optuna as optuna_runtime

if TYPE_CHECKING:
    from collections.abc import Mapping

CodeProjection = common.code_identity.CodeProjection

_SOURCE_ROOT = Path(common.code_identity.__file__).resolve().parents[1]
_PROJECTIONS: dict[str, CodeProjection] = {
    projection.name: projection
    for projection in (
        generation.contracts.source.GENERATION_INPUT_PROJECTION,
        generation.contracts.source.GENERATION_SOURCE_PROJECTION,
        *datasets.packages.reproduction.DATASET_CONVERSION_PROJECTIONS.values(),
        *learning.training.checkpoint.MODEL_PROJECTIONS.values(),
        *learning.training.checkpoint.TRAINING_PROJECTIONS.values(),
        *optuna_runtime.HPO_PROJECTIONS.values(),
        *artifact_contracts.ANALYSIS_PROJECTIONS.values(),
    )
}
_EXPECTED_PROJECTION_COUNT = 12
_ONE = "value = 1\n"


def _write_tree(root: Path, files: Mapping[str, str]) -> None:
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def test_all_declared_projections_are_covered() -> None:
    """Cover generation input and source, both Dataset views, and the model, training, study and analysis projections of both tasks."""
    assert len(_PROJECTIONS) == _EXPECTED_PROJECTION_COUNT


@pytest.mark.parametrize("name", sorted(_PROJECTIONS))
def test_projection_resolves_to_sorted_unique_existing_members(name: str) -> None:
    """Resolve every declared path below the executing source and fail on none missing."""
    members = _PROJECTIONS[name].members()
    assert members
    assert list(members) == sorted(set(members))
    assert all((_SOURCE_ROOT / member).is_file() for member in members)


@pytest.mark.parametrize("name", sorted(_PROJECTIONS))
def test_every_static_import_of_every_member_is_classified(name: str) -> None:
    """Require each intra-src import of a member to be a member, infrastructure, an exemption, or cross-task."""
    unclassified = import_closure.unclassified_imports(_PROJECTIONS[name], _SOURCE_ROOT)
    assert not unclassified, "\n".join(item.message() for item in unclassified)


@pytest.mark.parametrize("name", sorted(_PROJECTIONS))
def test_no_reached_package_initializer_binds_a_name_contradictorily(name: str) -> None:
    """Require one runtime module per package name, so alias resolution never has to pick between two."""
    contradictions = import_closure.contradictory_aliases(_PROJECTIONS[name], _SOURCE_ROOT)
    assert not contradictions, "\n".join(item.message() for item in contradictions)


def test_exemptions_and_infrastructure_name_existing_modules_with_reasons() -> None:
    """Keep exemptions and infrastructure entries from going stale after a rename or removal."""
    entries = {
        **common.code_identity.INFRASTRUCTURE_MODULES,
        **{
            name: reason
            for projection in _PROJECTIONS.values()
            for part in import_closure.flatten(projection)
            for name, reason in part.exemptions.items()
        },
    }
    for name, reason in entries.items():
        assert (_SOURCE_ROOT / name).is_file(), name
        assert reason.strip(), name
    for directory, reason in common.code_identity.INFRASTRUCTURE_DIRECTORIES.items():
        assert (_SOURCE_ROOT / directory).is_dir(), directory
        assert reason.strip(), directory


def test_exemptions_never_name_members_of_the_same_projection() -> None:
    """Keep a module either hashed or exempted, so an exemption is never a hidden membership claim."""
    for projection in _PROJECTIONS.values():
        members = set(projection.members())
        assert members.isdisjoint(projection.exemptions), projection.name


def test_unclassified_import_of_a_real_member_is_reported_with_file_and_line(tmp_path: Path) -> None:
    """Fail closed on a copy of the real tree when a member gains an import no projection classifies."""
    copied = tmp_path / "src"
    shutil.copytree(_SOURCE_ROOT, copied, ignore=shutil.ignore_patterns("__pycache__"))
    projection = generation.contracts.source.GENERATION_INPUT_PROJECTION
    assert import_closure.unclassified_imports(projection, copied) == []
    member = copied / "generation/cases/generation_cases_seeding.py"
    lines = member.read_text(encoding="utf-8").splitlines()
    member.write_text("\n".join([*lines, "from src.generation import generation_run as _unclassified  # noqa: F401", ""]), encoding="utf-8")
    reported = import_closure.unclassified_imports(projection, copied)
    assert [(item.importer, item.lineno, item.target) for item in reported] == [
        ("generation/cases/generation_cases_seeding.py", len(lines) + 1, "generation/generation_run.py"),
    ]
    assert "classify it: add it to the projection or exempt it with a reason" in reported[0].message()


def test_unclassified_imports_are_resolved_through_direct_alias_and_lazy_references(tmp_path: Path) -> None:
    """Resolve relative imports, package-alias attribute use, function-local imports and lazy tables, not TYPE_CHECKING."""
    _write_tree(
        tmp_path,
        {
            "__init__.py": "",
            "pkg/__init__.py": (
                "from __future__ import annotations\n"
                "from typing import TYPE_CHECKING\n"
                "if TYPE_CHECKING:\n"
                "    from . import pkg_aliased as aliased\n"
                "_MODULES = {'aliased': 'pkg_aliased', 'lazy': 'pkg_lazy'}\n"
            ),
            "pkg/pkg_member.py": (
                "from __future__ import annotations\n"
                "from typing import TYPE_CHECKING\n"
                "from src import pkg\n"
                "from . import pkg_direct as direct\n"
                "if TYPE_CHECKING:\n"
                "    from . import pkg_typing_only as typing_only\n"
                "\n"
                "def run() -> object:\n"
                "    from . import pkg_local as local\n"
                "    return pkg.aliased, direct, local\n"
            ),
            "pkg/pkg_direct.py": _ONE,
            "pkg/pkg_aliased.py": _ONE,
            "pkg/pkg_local.py": _ONE,
            "pkg/pkg_lazy.py": _ONE,
            "pkg/pkg_typing_only.py": _ONE,
        },
    )
    projection = CodeProjection(name="synthetic", files=("pkg/pkg_member.py", "pkg/__init__.py"))
    reported = {(item.importer, item.target) for item in import_closure.unclassified_imports(projection, tmp_path)}
    assert reported == {
        ("pkg/pkg_member.py", "pkg/pkg_direct.py"),
        ("pkg/pkg_member.py", "pkg/pkg_aliased.py"),
        ("pkg/pkg_member.py", "pkg/pkg_local.py"),
        ("pkg/__init__.py", "pkg/pkg_aliased.py"),
        ("pkg/__init__.py", "pkg/pkg_lazy.py"),
    }


def test_runtime_lazy_binding_wins_over_a_stale_type_checking_alias(tmp_path: Path) -> None:
    """Follow the lazy table that executes, not the TYPE_CHECKING alias that only the type checker reads."""
    _write_tree(
        tmp_path,
        {
            "__init__.py": "",
            "pkg/__init__.py": (
                "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from . import pkg_old as chosen\n_MODULES = {'chosen': 'pkg_new'}\n"
            ),
            "pkg/pkg_member.py": "from src import pkg\n\nvalue = pkg.chosen\n",
            "pkg/pkg_old.py": _ONE,
            "pkg/pkg_new.py": _ONE,
        },
    )
    projection = CodeProjection(name="synthetic", files=("pkg/pkg_member.py", "pkg/pkg_old.py"))
    reported = import_closure.unclassified_imports(projection, tmp_path)
    assert [(item.importer, item.target) for item in reported] == [("pkg/pkg_member.py", "pkg/pkg_new.py")]
    assert import_closure.contradictory_aliases(projection, tmp_path) == []


def test_contradictory_runtime_bindings_of_one_package_name_are_reported_not_resolved(tmp_path: Path) -> None:
    """Report a name that an initializer binds to different modules at runtime; agreeing bindings are fine."""
    _write_tree(
        tmp_path,
        {
            "__init__.py": "",
            "pkg/__init__.py": (
                "from . import pkg_old as chosen\nfrom . import pkg_old as agreed\n_MODULES = {'chosen': 'pkg_new', 'agreed': 'pkg_old'}\n"
            ),
            "pkg/pkg_member.py": "from src import pkg\n\nvalue = pkg.chosen, pkg.agreed\n",
            "pkg/pkg_old.py": _ONE,
            "pkg/pkg_new.py": _ONE,
        },
    )
    projection = CodeProjection(name="synthetic", files=("pkg/pkg_member.py", "pkg/pkg_old.py", "pkg/pkg_new.py"))
    reported = import_closure.contradictory_aliases(projection, tmp_path)
    assert [(item.initializer, item.name, item.targets) for item in reported] == [
        ("pkg/__init__.py", "chosen", ("pkg/pkg_new.py", "pkg/pkg_old.py")),
    ]
    assert "contradictory alias" in reported[0].message()


def test_imports_are_accepted_as_member_exemption_infrastructure_cross_task_or_package_init(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accept each classification the projection rules define, and no other import."""
    _write_tree(
        tmp_path,
        {
            "__init__.py": "",
            "pkg/__init__.py": "",
            "pkg/pkg_member.py": (
                "from . import pkg_other_member, pkg_exempt, pkg_infra, pkg_transient_dispatch, pkg_unclassified\n"
                "from .transient import tr_module\n"
                "from src import pkg\n"
                "from src.infra import infra_module\n"
            ),
            "pkg/pkg_other_member.py": _ONE,
            "pkg/pkg_exempt.py": _ONE,
            "pkg/pkg_infra.py": _ONE,
            "pkg/pkg_transient_dispatch.py": _ONE,
            "pkg/pkg_unclassified.py": _ONE,
            "pkg/transient/__init__.py": "",
            "pkg/transient/tr_module.py": _ONE,
            "infra/__init__.py": "",
            "infra/infra_module.py": _ONE,
        },
    )
    monkeypatch.setattr(common.code_identity, "INFRASTRUCTURE_MODULES", {"pkg/pkg_infra.py": "test infrastructure module"})
    monkeypatch.setattr(common.code_identity, "INFRASTRUCTURE_DIRECTORIES", {"infra": "test infrastructure directory"})
    projection = CodeProjection(
        name="synthetic",
        files=("pkg/pkg_member.py", "pkg/pkg_other_member.py"),
        excluded_markers=("transient",),
        exemptions={"pkg/pkg_exempt.py": "test exemption"},
    )
    reported = import_closure.unclassified_imports(projection, tmp_path)
    assert [(item.importer, item.target) for item in reported] == [("pkg/pkg_member.py", "pkg/pkg_unclassified.py")]
    assert reported[0].lineno == 1


def test_members_resolve_directories_recursively_with_markers_and_never_filter_explicit_files(tmp_path: Path) -> None:
    """Remove other-task directory members by filename marker while keeping explicitly declared files."""
    _write_tree(
        tmp_path,
        {
            "area/area_core.py": _ONE,
            "area/sub/area_deep.py": _ONE,
            "area/area_transient_only.py": _ONE,
            "area/not_python.txt": "text\n",
            "single.py": _ONE,
            "other/other_transient_declared.py": _ONE,
        },
    )
    projection = CodeProjection(
        name="synthetic",
        directories=("area",),
        files=("single.py", "other/other_transient_declared.py"),
        excluded_markers=("transient",),
    )
    assert projection.members(tmp_path) == (
        "area/area_core.py",
        "area/sub/area_deep.py",
        "other/other_transient_declared.py",
        "single.py",
    )


def test_members_add_the_package_initializers_up_to_the_source_root(tmp_path: Path) -> None:
    """Hash the initializers a member runs through, existing ones only, without adding unrelated packages."""
    _write_tree(
        tmp_path,
        {
            "__init__.py": "",
            "area/__init__.py": "",
            "area/sub/area_deep.py": _ONE,
            "namespace/inner/__init__.py": "",
            "namespace/inner/inner_core.py": _ONE,
            "other/__init__.py": "",
            "other/other_core.py": _ONE,
        },
    )
    projection = CodeProjection(name="synthetic", files=("area/sub/area_deep.py", "namespace/inner/inner_core.py"))
    assert projection.declared_members(tmp_path) == ("area/sub/area_deep.py", "namespace/inner/inner_core.py")
    assert projection.members(tmp_path) == (
        "__init__.py",
        "area/__init__.py",
        "area/sub/area_deep.py",
        "namespace/inner/__init__.py",
        "namespace/inner/inner_core.py",
    )


@pytest.mark.parametrize(
    ("initializer", "changes_identity"),
    [("__init__.py", True), ("pkg/__init__.py", True), ("pkg/sub/__init__.py", True), ("other/__init__.py", False)],
)
def test_digest_follows_package_initializer_dispatch_but_not_its_docstring_or_comments(
    tmp_path: Path,
    initializer: str,
    changes_identity: bool,
) -> None:
    """Bind which module an alias dispatches to in every initializer a member runs through."""
    alias = "from . import pkg_old as chosen\n"
    _write_tree(
        tmp_path,
        {
            "__init__.py": f'"""Source doc."""\n{alias}',
            "pkg/__init__.py": f'"""Package doc."""\n{alias}',
            "pkg/sub/__init__.py": f'"""Subpackage doc."""\n{alias}',
            "pkg/sub/pkg_member.py": _ONE,
            "other/__init__.py": f'"""Other doc."""\n{alias}',
            "other/other_core.py": _ONE,
        },
    )
    projection = CodeProjection(name="synthetic", files=("pkg/sub/pkg_member.py",))
    baseline = projection.sha256(tmp_path)
    path = tmp_path / initializer
    path.write_text(f'"""Changed doc."""\n# new comment\n{alias}', encoding="utf-8")
    assert projection.sha256(tmp_path) == baseline
    path.write_text(f'"""Changed doc."""\n{alias.replace("pkg_old", "pkg_new")}', encoding="utf-8")
    assert (projection.sha256(tmp_path) != baseline) is changes_identity


@pytest.mark.parametrize(
    "projection",
    [
        CodeProjection(name="missing-directory", directories=("absent",)),
        CodeProjection(name="missing-file", files=("absent.py",)),
        CodeProjection(name="escaping-file", files=("../present.py",)),
        CodeProjection(name="not-python", files=("present.txt",)),
        CodeProjection(name="absolute-directory", directories=("/present",)),
    ],
    ids=lambda projection: projection.name,
)
def test_missing_and_uncontained_declarations_raise_value_error(tmp_path: Path, projection: CodeProjection) -> None:
    """Refuse a projection that names a missing path or a path outside the source root."""
    _write_tree(tmp_path, {"present.py": _ONE})
    with pytest.raises(ValueError, match=r"declares a missing|contained relative"):
        projection.members(tmp_path)


def test_digest_ignores_comments_and_docstrings_and_follows_code_and_membership(tmp_path: Path) -> None:
    """Bind code semantics: presentation edits keep the digest, real code or membership edits change it."""
    _write_tree(
        tmp_path,
        {
            "area/area_core.py": '"""Module doc."""\n\n\ndef compute(x):\n    """Doc."""\n    return x + 1  # comment\n',
            "area/area_extra.py": _ONE,
        },
    )
    projection = CodeProjection(name="synthetic", files=("area/area_core.py",))
    baseline = projection.sha256(tmp_path)
    core = tmp_path / "area/area_core.py"
    core.write_text('"""Changed module doc."""\n# new comment\n\n\ndef compute(x):\n    """Changed doc."""\n    return x + 1\n', encoding="utf-8")
    assert projection.sha256(tmp_path) == baseline
    core.write_text('"""Module doc."""\n\n\ndef compute(x):\n    """Doc."""\n    return x + 2\n', encoding="utf-8")
    changed = projection.sha256(tmp_path)
    assert changed != baseline
    widened = CodeProjection(name="synthetic", files=("area/area_core.py", "area/area_extra.py"))
    assert widened.sha256(tmp_path) != changed


def test_current_implementation_guard_accepts_the_running_digest_and_refuses_drift_or_bad_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pass a matching digest; refuse a changed, malformed or missing one, naming unit, projection and remedy."""
    _write_tree(tmp_path, {"core.py": _ONE})
    unrooted_sha256 = CodeProjection.sha256
    monkeypatch.setattr(CodeProjection, "sha256", lambda self, root=None: unrooted_sha256(self, tmp_path if root is None else root))
    projection = CodeProjection(name="synthetic", files=("core.py",))
    recorded = projection.sha256()
    guard = common.code_identity.require_current_implementation

    guard(projection, recorded, unit="Run x", remedy="start a new run")
    (tmp_path / "core.py").write_text("value = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"^Run x: synthetic implementation changed \(recorded \w{12}, current \w{12}\); start a new run$"):
        guard(projection, recorded, unit="Run x", remedy="start a new run")
    for bad in (None, recorded.upper(), recorded[:-1]):
        with pytest.raises(ValueError, match=r"^Run x: recorded synthetic implementation digest is missing or malformed; start a new run$"):
            guard(projection, bad, unit="Run x", remedy="start a new run")


def test_composition_includes_the_included_members_in_members_and_digest(tmp_path: Path) -> None:
    """Resolve an including projection as its own files plus the included projection, hash included."""
    _write_tree(tmp_path, {"own.py": _ONE, "shared/shared.py": _ONE})
    shared = CodeProjection(name="shared", files=("shared/shared.py",))
    including = CodeProjection(name="including", files=("own.py",), includes=(shared,))
    assert including.members(tmp_path) == ("own.py", "shared/shared.py")
    before = including.sha256(tmp_path)
    (tmp_path / "shared/shared.py").write_text("value = 2\n", encoding="utf-8")
    assert including.sha256(tmp_path) != before


@pytest.mark.parametrize("task_id", sorted(learning.training.checkpoint.TRAINING_PROJECTIONS))
def test_training_projection_includes_the_model_projection_of_its_task(task_id: str) -> None:
    """Make a model change invalidate the training identity of the same task, never the other way round."""
    checkpoint = learning.training.checkpoint
    model = checkpoint.MODEL_PROJECTIONS[task_id]
    training = checkpoint.TRAINING_PROJECTIONS[task_id]
    assert training.includes == (model,)
    assert set(model.members()) < set(training.members())


@pytest.mark.parametrize("task_id", sorted(learning.training.checkpoint.TRAINING_PROJECTIONS))
def test_study_projection_is_its_files_plus_the_training_projection_of_its_task(task_id: str) -> None:
    """Make a training change invalidate the study identity of the same task."""
    training = learning.training.checkpoint.TRAINING_PROJECTIONS[task_id]
    study = optuna_runtime.HPO_PROJECTIONS[task_id]
    assert study.includes == (training,)
    assert set(study.members()) == {
        *training.members(),
        "experiments/tuning/__init__.py",
        "experiments/tuning/experiments_tuning_optuna.py",
        "experiments/tuning/experiments_tuning_search_space.py",
    }


def test_owners_compute_identities_from_their_declared_projections() -> None:
    """Bind the digest functions to the declared projections and reject unsupported keys."""
    checkpoint = learning.training.checkpoint
    for task_id, projection in checkpoint.TRAINING_PROJECTIONS.items():
        assert checkpoint.scientific_implementation_digest(task_id) == projection.sha256()
        assert checkpoint.model_implementation_digest(task_id) == checkpoint.MODEL_PROJECTIONS[task_id].sha256()
        assert optuna_runtime._hpo_implementation_sha256(task_id) == optuna_runtime.HPO_PROJECTIONS[task_id].sha256()
        assert artifact_contracts.analysis_producer(task_id) == {"analysis_sha256": artifact_contracts.ANALYSIS_PROJECTIONS[task_id].sha256()}
    for view, projection in datasets.packages.reproduction.DATASET_CONVERSION_PROJECTIONS.items():
        assert datasets.packages.reproduction.conversion_implementation_sha256(view) == projection.sha256()
    source = generation.contracts.source
    identity = source.source_producer_identity()
    assert identity["implementation_sha256"] == source.GENERATION_INPUT_PROJECTION.sha256()
    assert identity["source_implementation_sha256"] == source.GENERATION_SOURCE_PROJECTION.sha256()
    with pytest.raises(ValueError, match="No training implementation projection"):
        checkpoint.scientific_implementation_digest("unregistered_task")
    with pytest.raises(ValueError, match="No model implementation projection"):
        checkpoint.model_implementation_digest("unregistered_task")
    with pytest.raises(ValueError, match="No HPO implementation projection"):
        optuna_runtime._hpo_implementation_sha256("unregistered_task")
    with pytest.raises(ValueError, match="Unsupported Dataset implementation view"):
        datasets.packages.reproduction.conversion_implementation_sha256("unregistered_view")


def test_generation_source_identity_includes_input_code_but_input_identity_excludes_solver_code() -> None:
    """Keep solver edits out of input reuse while any input-producing edit moves the Source identity."""
    source = generation.contracts.source
    input_members = set(source.GENERATION_INPUT_PROJECTION.members())
    source_members = set(source.GENERATION_SOURCE_PROJECTION.members())
    assert input_members < source_members
    assert "generation/runtime/generation_runtime_comsol.py" in source_members - input_members


@pytest.mark.parametrize(
    ("task_id", "relative_path", "changes_identity"),
    [
        ("steady_flow", "learning/models/learning_models_added.py", True),
        ("steady_flow", "domain/physics/domain_physics_brinkman.py", True),
        ("steady_flow", "common/common_random.py", True),
        ("steady_flow", "domain/tasks/domain_task_steady_flow.py", True),
        ("steady_flow", "experiments/config/experiments_config_loader.py", True),
        ("steady_flow", "learning/transient/learning_transient_rollout.py", False),
        ("steady_flow", "domain/tasks/domain_task_transient_drying.py", False),
        ("steady_flow", "experiments/experiments_matched_budget.py", False),
        ("steady_flow", "experiments/cli/cli_train.py", False),
        ("transient_drying", "learning/transient/learning_transient_rollout.py", True),
        ("transient_drying", "datasets/packages/dataset_packages_trajectory.py", True),
        ("transient_drying", "experiments/experiments_matched_budget.py", True),
        ("transient_drying", "domain/tasks/domain_task_transient_drying.py", True),
        ("transient_drying", "learning/losses/learning_losses_pino.py", False),
        ("transient_drying", "domain/tasks/domain_task_steady_flow.py", False),
        ("transient_drying", "analysis/presentation/added.py", False),
    ],
)
def test_training_projection_covers_task_science_without_unrelated_coupling(
    tmp_path: Path,
    task_id: str,
    relative_path: str,
    changes_identity: bool,
) -> None:
    """Detect scientific implementation changes without binding presentation or the other task."""
    projection = learning.training.checkpoint.TRAINING_PROJECTIONS[task_id]
    root = tmp_path / "src"
    _write_tree(root, dict.fromkeys(projection.members(_SOURCE_ROOT), _ONE))
    _write_tree(root, {relative_path: _ONE})
    before = projection.sha256(root)
    (root / relative_path).write_text("value = 2\n", encoding="utf-8")
    assert (projection.sha256(root) != before) is changes_identity


@pytest.mark.parametrize(
    ("task_id", "relative_path", "changes_identity"),
    [
        ("steady_flow", "datasets/runtime/dataset_runtime_training.py", True),
        ("steady_flow", "learning/models/learning_models_factory.py", True),
        ("steady_flow", "learning/losses/learning_losses_pino.py", False),
        ("steady_flow", "learning/metrics/learning_metrics.py", False),
        ("transient_drying", "learning/transient/learning_transient_scaling.py", True),
        ("transient_drying", "learning/losses/learning_losses_transient.py", False),
        ("transient_drying", "learning/transient/learning_transient_curriculum.py", False),
    ],
)
def test_model_projection_covers_fit_and_application_code_but_not_losses_metrics_or_lifecycle(
    tmp_path: Path,
    task_id: str,
    relative_path: str,
    changes_identity: bool,
) -> None:
    """Refit scaling and refuse stale weights on prediction-code edits while a loss-only edit spares the model."""
    projection = learning.training.checkpoint.MODEL_PROJECTIONS[task_id]
    root = tmp_path / "src"
    _write_tree(root, dict.fromkeys(projection.members(_SOURCE_ROOT), _ONE))
    _write_tree(root, {relative_path: _ONE})
    before = projection.sha256(root)
    (root / relative_path).write_text("value = 2\n", encoding="utf-8")
    assert (projection.sha256(root) != before) is changes_identity


@pytest.mark.parametrize("task_id", sorted(artifact_contracts.ANALYSIS_PROJECTIONS))
def test_analysis_projection_includes_the_model_projection_of_its_task_only(task_id: str) -> None:
    """Refuse stale artifacts after prediction-code edits of the same task, never after the other task's artifact code."""
    model = learning.training.checkpoint.MODEL_PROJECTIONS[task_id]
    analysis = artifact_contracts.ANALYSIS_PROJECTIONS[task_id]
    assert analysis.includes == (model,)
    assert set(model.members()) < set(analysis.members())
    other = next(other for other in artifact_contracts.ANALYSIS_PROJECTIONS if other != task_id)
    own_writer = {"steady_flow": "analysis_artifact_generation.py", "transient_drying": "analysis_artifact_transient.py"}
    members = {Path(member).name for member in analysis.members()}
    assert own_writer[task_id] in members
    assert own_writer[other] not in members


@pytest.mark.parametrize(
    ("task_id", "relative_path", "changed"),
    [
        ("steady_flow", "learning/models/learning_models_factory.py", {"model", "training", "analysis"}),
        ("steady_flow", "learning/metrics/learning_metrics.py", {"training", "analysis"}),
        ("steady_flow", "analysis/artifacts/analysis_artifact_service.py", {"analysis"}),
        ("steady_flow", "analysis/artifacts/analysis_artifact_transient.py", set()),
        ("transient_drying", "learning/transient/learning_transient_rollout.py", {"model", "training", "analysis"}),
        ("transient_drying", "analysis/evaluation/evaluation_transient_metrics.py", {"analysis"}),
        ("transient_drying", "learning/transient/learning_transient_curriculum.py", {"training"}),
        ("transient_drying", "analysis/artifacts/analysis_artifact_generation.py", set()),
    ],
)
def test_analysis_projection_composes_with_model_and_training_identities(
    tmp_path: Path,
    task_id: str,
    relative_path: str,
    changed: set[str],
) -> None:
    """A model edit moves model, training and analysis; a metric edit moves training and analysis; artifact edits move analysis only."""
    projections = {
        "model": learning.training.checkpoint.MODEL_PROJECTIONS[task_id],
        "training": learning.training.checkpoint.TRAINING_PROJECTIONS[task_id],
        "analysis": artifact_contracts.ANALYSIS_PROJECTIONS[task_id],
    }
    root = tmp_path / "src"
    _write_tree(root, dict.fromkeys({member for projection in projections.values() for member in projection.members(_SOURCE_ROOT)}, _ONE))
    _write_tree(root, {relative_path: _ONE})
    before = {name: projection.sha256(root) for name, projection in projections.items()}
    (root / relative_path).write_text("value = 2\n", encoding="utf-8")
    assert {name for name, projection in projections.items() if projection.sha256(root) != before[name]} == changed
