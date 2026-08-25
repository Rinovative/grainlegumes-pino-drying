"""
evaluation_workspace.py

Prepare automatic run-discovery workspaces for Evaluation notebooks.

Responsibilities:
  - Discover persisted experiments once and retain the normalized catalog
  - Build one concise exact-run selector or compatible multi-run selector
  - Load validated artifacts read-only only for artifact-ready selections
  - Present lifecycle, provenance, compatibility, and recovery status without tracebacks
  - Delegate task-aware scientific panels to maintained Evaluation sessions

Design principles:
  - Notebook startup never reconstructs models or contacts tracking services
  - Human labels remain separate from exact run, Dataset, and checkpoint evidence
  - EDA's Aggregate and Single case values select task-appropriate section sets
  - Widget changes reuse the catalog and close superseded artifact sessions

This module does NOT:
  - Generate, rebuild, upload, or delete Evaluation artifacts
  - Infer task, stage, Dataset, or protocol identity from directory names
  - Reimplement scientific metrics, plotting, or artifact admission
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import ipywidgets as widgets

from src import analysis, common

from . import evaluation_run_discovery as run_discovery
from . import evaluation_selection as selection
from . import evaluation_workflow as workflow

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

WorkspaceKind = Literal["single", "comparison"]

_MINIMUM_COMPARISON_RUNS = 2
_TRAINING_COMPARISON_ARM_COUNT = 3


@dataclass(frozen=True, slots=True)
class PreparedEvaluationWorkspace:
    """Hold one read-only catalog, selection state, live panel, and audit summary."""

    catalog: run_discovery.EvaluationRunCatalog
    selection_state: selection.EvaluationSelectionState | None
    panel: widgets.Widget | None
    summary_text: str
    kind: WorkspaceKind
    _controller: _EvaluationWorkspaceController | None

    def close(self) -> None:
        """Release the currently loaded Evaluation session."""
        if self._controller is not None:
            self._controller.close()


def _dropdown(
    options: Sequence[str] | Sequence[tuple[str, Any]],
    *,
    description: str,
) -> widgets.Dropdown:
    """Build one compact Evaluation selector in the maintained widget style."""
    resolved = tuple(options)
    if not resolved:
        message = f"{description.rstrip(':')} options must not be empty."
        raise ValueError(message)
    first = resolved[0][1] if isinstance(resolved[0], tuple) else resolved[0]
    return widgets.Dropdown(
        options=resolved,
        value=first,
        description=description,
        style={"description_width": "initial"},
        layout=widgets.Layout(width="auto"),
    )


def _artifact_for_role(
    run: run_discovery.EvaluationRunDiscovery,
    role: Literal["id", "ood"],
) -> run_discovery.EvaluationArtifactInspection:
    """Return one selected artifact inspection."""
    return run.id_artifact if role == "id" else run.ood_artifact


def _role_label(role: str) -> str:
    """Return the exact shared ID/OOD presentation term."""
    labels = {
        "id": "ID",
        "ood": "Near-family OOD",
        "both": "ID + Near-family OOD",
    }
    try:
        return labels[role]
    except KeyError as error:
        message = f"Unsupported Evaluation regime {role!r}."
        raise ValueError(message) from error


def _run_option_label(run: run_discovery.EvaluationRunDiscovery) -> str:
    """Return one concise architecture, child-stage, and material label."""
    parts = [run.model_label]
    if run.stage is not None and run.stage.lower() not in run.model_label.lower():
        parts.append(run.stage)
    if run.dataset_label is not None:
        material = run.dataset_label.removesuffix("_id").replace("_", " ")
        if material.lower() not in " ".join(parts).lower():
            parts.append(material)
    return " · ".join(parts)


def _artifact_state_label(
    inspection: run_discovery.EvaluationArtifactInspection,
) -> str:
    """Return one concise human lifecycle state with scoped coverage."""
    labels = {
        "ready": "available",
        "generating": "generating",
        "scoped_partial": "scoped partial",
        "missing": "missing",
        "invalid": "invalid",
    }
    label = labels[inspection.state]
    if inspection.state == "scoped_partial" and inspection.case_count is not None:
        return f"{label} ({inspection.case_count} case{'s' if inspection.case_count != 1 else ''})"
    return label


def _compatibility_issues(
    runs: Sequence[run_discovery.EvaluationRunDiscovery],
) -> tuple[str, ...]:
    """Return explicit persisted-metadata incompatibilities before artifact loading."""
    if len(runs) < _MINIMUM_COMPARISON_RUNS:
        return ("Select at least two runs.",)
    fields = {
        "task": "task",
        "evaluation objective": "evaluation_protocol",
        "evaluation protocol": "evaluation_config_identity",
        "spatial stride": "spatial_stride",
    }
    issues: list[str] = []
    for label, field in fields.items():
        values = {getattr(run, field) for run in runs}
        if len(values) > 1:
            issues.append(f"Selected runs have incompatible {label}.")
    stages = {run.stage for run in runs}
    exact_three_arm = (
        {run.task for run in runs} == {"transient_drying"} and len(runs) == _TRAINING_COMPARISON_ARM_COUNT and stages == {"A0", "A+", "B"}
    )
    if len(stages) > 1 and not exact_three_arm:
        issues.append("Selected runs have incompatible stage semantics.")
    return tuple(issues)


class _EvaluationWorkspaceController:
    """Bind one cached run catalog to task-inferred load-only workflows."""

    def __init__(
        self,
        catalog: run_discovery.EvaluationRunCatalog,
        *,
        kind: WorkspaceKind,
        title: str,
    ) -> None:
        """Create one run explorer without loading numerical payloads."""
        self.catalog = catalog
        self.kind = kind
        self.title = title
        self.state = selection.EvaluationSelectionState(
            catalog,
            comparison=kind == "comparison",
        )
        self._workflow: workflow.PreparedEvaluationWorkflow | None = None
        self._updating = False
        self.status = widgets.HTML(layout=widgets.Layout(width="100%"))
        self.analysis = widgets.VBox()
        if kind == "single":
            self.run = _dropdown(
                tuple((_run_option_label(run), str(run.run_dir)) for run in catalog.runs),
                description="Run:",
            )
            controls: tuple[widgets.Widget, ...] = (self.run,)
            self.run.observe(self._single_run_changed, names="value")
            self._render_single()
        else:
            self.run_filter = widgets.SelectMultiple(
                options=tuple((_run_option_label(run), str(run.run_dir)) for run in catalog.runs),
                description="Runs:",
                style={"description_width": "initial"},
                layout=widgets.Layout(width="420px", height="100px"),
            )
            self.dataset_filter = _dropdown(
                (("Select compatible runs", None),),
                description="Material/Dataset:",
            )
            controls = (self.run_filter, self.dataset_filter)
            self.run_filter.observe(self._comparison_runs_changed, names="value")
            self.dataset_filter.observe(self._comparison_dataset_changed, names="value")
            self._select_initial_comparison()
        self.panel = widgets.VBox(
            (
                widgets.HTML(f"<h2>{escape(title)}</h2>"),
                widgets.VBox(controls),
                self.status,
                self.analysis,
            ),
            layout=widgets.Layout(width="100%"),
        )

    def close(self) -> None:
        """Close the current task session and clear its panel reference."""
        if self._workflow is not None:
            self._workflow.close()
            self._workflow = None
        self.analysis.children = ()

    def _run_by_path(self, path: str | Path) -> run_discovery.EvaluationRunDiscovery:
        """Return one exact catalog run by resolved current path."""
        resolved = Path(path).expanduser().resolve()
        matches = tuple(run for run in self.catalog.runs if run.run_dir == resolved)
        if len(matches) != 1:
            message = f"Selected run path is absent from the Evaluation catalog: {resolved}"
            raise RuntimeError(message)
        return matches[0]

    def _group_for_run(
        self,
        run: run_discovery.EvaluationRunDiscovery,
    ) -> run_discovery.EvaluationRunGroup:
        """Return the exact logical experiment that owns one selected child."""
        matches = tuple(group for group in self.catalog.groups if run in group.children)
        if len(matches) != 1:
            message = "Selected Evaluation run has no unique logical experiment owner."
            raise RuntimeError(message)
        return matches[0]

    @staticmethod
    def _available_roles(
        run: run_discovery.EvaluationRunDiscovery,
    ) -> tuple[Literal["id", "ood"], ...]:
        """Return every complete or explicitly scoped artifact role for one run."""
        return tuple(
            role
            for role in ("id", "ood")
            if _artifact_for_role(
                run,
                cast("Literal['id', 'ood']", role),
            ).state
            in {"ready", "scoped_partial"}
        )

    @staticmethod
    def _artifact_role(
        roles: Sequence[Literal["id", "ood"]],
    ) -> selection.ArtifactRole:
        """Collapse an available role inventory into shared selection state."""
        admitted = tuple(roles)
        if admitted == ("id", "ood"):
            return "both"
        if len(admitted) == 1:
            return admitted[0]
        return "id"

    @staticmethod
    def _dataset_binding_label(
        *,
        task: str,
        dataset_name: str,
        role: Literal["id", "ood"],
    ) -> str:
        """Project one exact Dataset identity to compact material-first text."""
        projection = run_discovery.dataset_display_projection(
            task,
            dataset_name,
        )
        suffix = "_id" if role == "id" else "_near_family_ood"
        material_projection = projection.removesuffix(suffix)
        material_ids = tuple(value for value in material_projection.split("+") if value)
        try:
            materials = tuple(
                analysis.presentation.display_labels.material_display_label(
                    material,
                )
                for material in material_ids
            )
        except (TypeError, ValueError):
            materials = ()
        material_text = " + ".join(materials) if materials else material_projection.replace("_", " ").replace("-", " ").capitalize()
        return f"{material_text} · {_role_label(role)}"

    def _dataset_bindings(
        self,
        run: run_discovery.EvaluationRunDiscovery,
    ) -> dict[str, tuple[str, Literal["id", "ood"]]]:
        """Return exact available Dataset choices and their artifact roles."""
        bindings: dict[str, tuple[str, Literal["id", "ood"]]] = {}
        if run.dataset_id is not None and run.id_artifact.state == "ready":
            bindings[f"id:{run.dataset_id}"] = (
                self._dataset_binding_label(
                    task=run.task,
                    dataset_name=run.dataset_id,
                    role="id",
                ),
                "id",
            )
        if run.ood_dataset_ids and run.ood_artifact.state == "ready":
            identity = common.serialization.canonical_json_sha256({"source_dataset_ids": list(run.ood_dataset_ids)})
            labels = tuple(
                self._dataset_binding_label(
                    task=run.task,
                    dataset_name=dataset_name,
                    role="ood",
                )
                for dataset_name in run.ood_dataset_ids
            )
            bindings[f"ood:{identity}"] = (
                " + ".join(dict.fromkeys(labels)),
                "ood",
            )
        return bindings

    def _common_dataset_bindings(
        self,
        runs: Sequence[run_discovery.EvaluationRunDiscovery],
    ) -> dict[str, tuple[str, Literal["id", "ood"]]]:
        """Return the exact available Material/Dataset intersection."""
        admitted = tuple(runs)
        if not admitted:
            return {}
        inventories = tuple(self._dataset_bindings(run) for run in admitted)
        keys = set(inventories[0])
        for inventory in inventories[1:]:
            keys.intersection_update(inventory)
        result: dict[str, tuple[str, Literal["id", "ood"]]] = {}
        for key in sorted(keys):
            values = {inventory[key] for inventory in inventories}
            if len(values) == 1:
                result[key] = next(iter(values))
        return result

    def _status_html(
        self,
        run: run_discovery.EvaluationRunDiscovery,
        *,
        selected_roles: Sequence[Literal["id", "ood"]],
        detail: str | None = None,
        material_coverage: Sequence[tuple[str, Literal["id", "ood"]]] = (),
    ) -> str:
        """Return one compact EDA-aligned overview and lifecycle action."""
        del selected_roles
        coverage_roles: tuple[Literal["id", "ood"], ...] = ("id", "ood")
        coverage = (
            "; ".join(
                f"{analysis.presentation.display_labels.material_role_display_label(material, role)} · "
                f"{_artifact_state_label(_artifact_for_role(run, role))}"
                for material, role in material_coverage
            )
            if material_coverage
            else "; ".join(f"{_role_label(role)} · {_artifact_state_label(_artifact_for_role(run, role))}" for role in coverage_roles)
        )
        rows = (
            (
                "Task",
                analysis.presentation.display_labels.task_display_label(
                    run.task,
                ),
            ),
            ("Model", run.model_label),
            ("Stage", run.stage or "—"),
            ("Seed", "—" if run.seed is None else str(run.seed)),
            ("Dataset", run.dataset_label or run.dataset_id or "unavailable"),
            (
                "Protocol",
                "unavailable"
                if run.evaluation_protocol is None
                else analysis.presentation.display_labels.evaluation_protocol_display_label(run.evaluation_protocol),
            ),
            ("Spatial stride", "—" if run.spatial_stride is None else str(run.spatial_stride)),
            ("Run state", run.status),
            ("Artifact coverage", coverage),
        )
        table = "".join(
            f"<tr><th style='text-align:left;padding-right:1.25em'>{escape(label)}</th><td>{escape(value)}</td></tr>" for label, value in rows
        )
        states = {artifact.state for artifact in (run.id_artifact, run.ood_artifact)}
        command = ""
        if states.intersection({"missing", "scoped_partial"}):
            command = f"<p><b>Generate full artifact coverage</b><br><code>{escape(run.artifact_command)}</code></p>"
        extra = "" if detail is None else f"<p>{escape(detail)}</p>"
        return f"<h3>Run overview</h3><table>{table}</table>{extra}{command}"

    def _replace_workflow(
        self,
        prepared: workflow.PreparedEvaluationWorkflow,
    ) -> None:
        """Install one newly prepared panel after closing its predecessor."""
        previous = self._workflow
        self._workflow = prepared
        self.analysis.children = (prepared.panel,)
        if previous is not None:
            previous.close()

    def _clear_workflow(self) -> None:
        """Release loaded artifact state while retaining the cached run catalog."""
        if self._workflow is not None:
            self._workflow.close()
            self._workflow = None
        self.analysis.children = ()

    def _single_run_changed(self, _change: Mapping[str, object]) -> None:
        """Dispatch the newly selected run through its persisted task metadata."""
        if not self._updating:
            self._render_single()

    def _render_single(self) -> None:
        """Render every available Dataset role for one exact selected run."""
        run = self._run_by_path(str(self.run.value))
        group = self._group_for_run(run)
        self.state.select_experiment(group.identity_sha256, run_dirs=(run.run_dir,))
        selected_roles = self._available_roles(run)
        self.state.select_artifact_role(self._artifact_role(selected_roles))
        if not run.evaluable:
            self._clear_workflow()
            self.status.value = self._status_html(
                run,
                selected_roles=selected_roles,
                detail="This run is not terminal and checkpoint-admitted for Evaluation.",
            )
            return
        if not selected_roles:
            self._clear_workflow()
            states = {artifact.state for artifact in (run.id_artifact, run.ood_artifact)}
            if "generating" in states:
                detail = "Artifact generation is active; completed staging payloads remain hidden until atomic publication."
            elif "invalid" in states:
                detail = "Artifact evidence is invalid or stale and was not admitted."
            else:
                detail = "No completed artifact is available; generation remains an explicit host workflow."
            self.status.value = self._status_html(
                run,
                selected_roles=(),
                detail=detail,
            )
            return
        scoped_roles: tuple[Literal["id", "ood"], ...] = tuple(
            cast("Literal['id', 'ood']", role)
            for role in selected_roles
            if _artifact_for_role(
                run,
                cast("Literal['id', 'ood']", role),
            ).state
            == "scoped_partial"
        )
        artifact_roots: dict[workflow.ArtifactSelectionRole, Path] = {role: _artifact_for_role(run, role).root for role in scoped_roles}
        sections: Sequence[str] | str = ("sample_viewer",) if scoped_roles and run.task == "transient_drying" else "all"
        try:
            prepared = workflow.prepare_single_model_evaluation_workspace(
                run.run_dir,
                label=run.label,
                expected_task=run.task,
                artifact_roles=selected_roles,
                artifact_roots=artifact_roots or None,
                auto_build_missing=False,
                rebuild_incompatible=False,
                sections=sections,
                selection_state=self.state,
            )
        except Exception as error:  # noqa: BLE001 -- bounded notebook status
            self._clear_workflow()
            self.status.value = self._status_html(
                run,
                selected_roles=selected_roles,
                detail=f"Artifact admission failed ({type(error).__name__}): {error}",
            )
            return
        unavailable = tuple(role for role in ("id", "ood") if role not in selected_roles)
        availability_detail: str | None = None
        if scoped_roles:
            availability_detail = "Selected-case evidence is admitted for Single-case Evaluation only; aggregate coverage remains unavailable."
        elif unavailable:
            availability_detail = (
                "Available Dataset roles were loaded automatically; unavailable roles remain capability-gated: "
                + ", ".join(_role_label(role) for role in unavailable)
                + "."
            )
        material_coverage: tuple[tuple[str, Literal["id", "ood"]], ...] = ()
        session_inventory = getattr(getattr(prepared, "session", None), "partitioned_case_inventory", None)
        if callable(session_inventory):
            entries = tuple(cast("Any", session_inventory)())
            unique: dict[tuple[str, Literal["id", "ood"]], tuple[int, int]] = {}
            artifact_positions: dict[Path, int] = {}
            for entry in entries:
                role = cast("Literal['id', 'ood']", entry.dataset_role)
                artifact_position = artifact_positions.setdefault(
                    entry.artifact_root,
                    len(artifact_positions),
                )
                material_position = entry.dataset_name.find(entry.material_family)
                unique.setdefault(
                    (entry.material_family, role),
                    (
                        artifact_position,
                        material_position if material_position >= 0 else len(entry.dataset_name),
                    ),
                )
            material_coverage = tuple(
                key
                for key, _position in sorted(
                    unique.items(),
                    key=lambda item: (*item[1], item[0][0]),
                )
            )
        self.status.value = self._status_html(
            run,
            selected_roles=selected_roles,
            detail=availability_detail,
            material_coverage=material_coverage,
        )
        self._replace_workflow(prepared)

    def _select_initial_comparison(self) -> None:
        """Select the first compatible ready pair without task or seed filters."""
        runs = self.catalog.runs
        selected: tuple[run_discovery.EvaluationRunDiscovery, ...] = ()
        for left_index, left in enumerate(runs):
            for right in runs[left_index + 1 :]:
                candidate = (left, right)
                if not _compatibility_issues(candidate) and self._common_dataset_bindings(candidate):
                    selected = candidate
                    break
            if selected:
                break
        self._updating = True
        try:
            self.run_filter.value = tuple(str(run.run_dir) for run in selected)
            self._sync_dataset_filter(selected)
        finally:
            self._updating = False
        self._render_comparison()

    def _sync_dataset_filter(
        self,
        runs: Sequence[run_discovery.EvaluationRunDiscovery],
    ) -> None:
        """Rebind concrete Material/Dataset choices from selected-run intersection."""
        previous = self.dataset_filter.value
        bindings = self._common_dataset_bindings(runs)
        if not bindings:
            self.dataset_filter.options = (("No compatible available Dataset", None),)
            self.dataset_filter.value = None
            return
        options = tuple((label, key) for key, (label, _role) in bindings.items())
        self.dataset_filter.options = options
        values = {value for _label, value in options}
        self.dataset_filter.value = previous if previous in values else options[0][1]

    def _comparison_runs_changed(self, _change: Mapping[str, object]) -> None:
        """Recompute Dataset compatibility after an exact run selection."""
        if self._updating:
            return
        selected = tuple(self._run_by_path(value) for value in self.run_filter.value)
        self._updating = True
        try:
            self._sync_dataset_filter(selected)
        finally:
            self._updating = False
        self._render_comparison()

    def _comparison_dataset_changed(self, _change: Mapping[str, object]) -> None:
        """Render the selected concrete Dataset without changing run discovery."""
        if not self._updating:
            self._render_comparison()

    @staticmethod
    def _comparison_label(run: run_discovery.EvaluationRunDiscovery) -> str:
        """Return one concise architecture label with provenance as secondary text."""
        parts = [run.model_label]
        if run.stage is not None and run.stage.lower() not in run.model_label.lower():
            parts.append(run.stage)
        return " · ".join(parts)

    def _render_comparison(self) -> None:
        """Render one task-compatible comparison on a concrete shared Dataset."""
        selected = tuple(self._run_by_path(value) for value in self.run_filter.value)
        issues = list(_compatibility_issues(selected))
        bindings = self._common_dataset_bindings(selected)
        dataset_key = self.dataset_filter.value
        if selected and not bindings:
            issues.append("Selected runs share no available Material/Dataset artifact.")
        if bindings and dataset_key not in bindings:
            issues.append("Select one compatible Material/Dataset.")
        if issues:
            self._clear_workflow()
            items = "".join(f"<li>{escape(issue)}</li>" for issue in dict.fromkeys(issues))
            self.status.value = (
                f"<h3>Compatible selected runs</h3><ul>{items}</ul>"
                "<p>All discovered runs remain visible; incompatible artifacts are capability-gated.</p>"
            )
            return
        dataset_label, role = bindings[cast("str", dataset_key)]
        self.state.select_catalog_runs(tuple(run.run_dir for run in selected))
        self.state.select_artifact_role(role)
        try:
            prepared = workflow.prepare_model_comparison_evaluation_workspace(
                tuple(run.run_dir for run in selected),
                labels=tuple(self._comparison_label(run) for run in selected),
                expected_task=selected[0].task,
                artifact_roles=(role,),
                auto_build_missing=False,
                rebuild_incompatible=False,
                sections="all",
                selection_state=self.state,
            )
        except Exception as error:  # noqa: BLE001 -- bounded notebook status
            self._clear_workflow()
            self.status.value = f"<h3>Compatibility warning</h3><p>{escape(type(error).__name__)}: {escape(str(error))}</p>"
            return
        self.status.value = (
            f"<h3>Compatible selected runs</h3><p>{len(selected)} artifacts admitted for {escape(dataset_label)} using exact persisted metadata.</p>"
        )
        self._replace_workflow(prepared)


def _summary(
    catalog: run_discovery.EvaluationRunCatalog,
    *,
    kind: WorkspaceKind,
) -> str:
    """Build one concise plain-text discovery and artifact audit."""
    task_counts = ", ".join(f"{task}={count}" for task, count in sorted(catalog.counts_by_task.items())) or "none"
    states = ", ".join(f"{state}={count}" for state, count in sorted(catalog.counts_by_artifact_state.items())) or "none"
    return (
        f"Evaluation workspace summary\n\n"
        f"Mode: {kind}\n"
        f"Discovered runs: {len(catalog.runs)}\n"
        f"Logical experiments: {len(catalog.groups)}\n"
        f"Tasks: {task_counts}\n"
        f"Artifact roles: {states}\n"
        f"Discovery issues: {len(catalog.issues)}"
    )


def prepare_evaluation_workspace(
    *,
    experiments_root: Path | str | None = None,
    storage_root: Path | str | None = None,
    kind: WorkspaceKind,
    title: str,
) -> PreparedEvaluationWorkspace:
    """Discover runs once and prepare one automatic Evaluation notebook workspace."""
    if kind not in {"single", "comparison"}:
        message = f"Unsupported Evaluation workspace kind {kind!r}."
        raise ValueError(message)
    if experiments_root is not None and storage_root is not None:
        message = "Provide experiments_root or storage_root, not both."
        raise ValueError(message)
    root = Path(experiments_root) if experiments_root is not None else common.paths.get_experiments_root(storage_root=storage_root)
    catalog = run_discovery.discover_evaluation_runs(root)
    if not catalog.groups:
        message = "No persisted Evaluation runs were discovered. Complete or inspect Training runs below the canonical experiments root."
        empty_catalog = run_discovery.EvaluationRunCatalog(
            catalog.runs,
            catalog.groups,
            (
                *catalog.issues,
                run_discovery.EvaluationDiscoveryIssue(Path(root), message),
            ),
            catalog.counts_by_task,
            catalog.counts_by_status,
            catalog.counts_by_identity,
            catalog.counts_by_artifact_state,
        )
        placeholder = widgets.HTML(f"<p>{escape(message)}</p>")
        return PreparedEvaluationWorkspace(
            catalog=empty_catalog,
            selection_state=None,
            panel=placeholder,
            summary_text=_summary(empty_catalog, kind=kind),
            kind=kind,
            _controller=None,
        )
    controller = _EvaluationWorkspaceController(
        catalog,
        kind=kind,
        title=title,
    )
    return PreparedEvaluationWorkspace(
        catalog=catalog,
        selection_state=controller.state,
        panel=controller.panel,
        summary_text=_summary(catalog, kind=kind),
        kind=kind,
        _controller=controller,
    )


def prepare_single_model_evaluation_workspace(
    *,
    experiments_root: Path | str | None = None,
    storage_root: Path | str | None = None,
    title: str = "Single-model Evaluation",
) -> PreparedEvaluationWorkspace:
    """Prepare automatic single-model Evaluation without run-name cells."""
    return prepare_evaluation_workspace(
        experiments_root=experiments_root,
        storage_root=storage_root,
        kind="single",
        title=title,
    )


def prepare_model_comparison_evaluation_workspace(
    *,
    experiments_root: Path | str | None = None,
    storage_root: Path | str | None = None,
    title: str = "Model comparison",
) -> PreparedEvaluationWorkspace:
    """Prepare automatic compatible-run comparison without a Python run list."""
    return prepare_evaluation_workspace(
        experiments_root=experiments_root,
        storage_root=storage_root,
        kind="comparison",
        title=title,
    )
