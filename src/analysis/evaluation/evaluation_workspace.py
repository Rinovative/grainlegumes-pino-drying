"""Prepare automatic run-discovery workspaces for Evaluation notebooks.

Responsibilities:
  - Discover persisted experiments once and retain the normalized catalog
  - Offer the runs as one collapsed selector grouped by architecture, one run (single) or several compatible runs (comparison)
  - Load validated artifacts read-only only for artifact-ready selections
  - Present lifecycle, provenance, compatibility, and recovery status without tracebacks, as one summary line with details on demand
  - Delegate task-aware scientific panels to maintained Evaluation sessions

Design principles:
  - Notebook startup never reconstructs models or contacts tracking services
  - Human labels remain separate from exact run, Dataset, and checkpoint evidence
  - EDA's Aggregate and Single case values select task-appropriate section sets
  - Widget changes reuse the catalog and close superseded artifact sessions
  - Preparation and its outcome run inside the publication into the workspace panel's plot place, so a newer selection supersedes an older one
    and an older failure cannot clear a newer panel

This module does NOT:
  - Generate, rebuild, or delete Evaluation artifacts
  - Infer task, stage, Dataset, or protocol identity from directory names
  - Reimplement scientific metrics, plotting, or artifact admission
  - Choose among the datasets of a run; the views of the panel do

Notes:
  Workspace modes:
    - Single-run mode admits every available role for one selected run and keeps missing, partial, or invalid artifact states visible without building
      artifacts.
    - Comparison mode requests every role whose exact Dataset binding all selected runs share; the workflow admits each role on its own, and a role it
      or the metadata refused is listed as unavailable with its reason instead of failing the others.
  Selection lifecycle:
    - Run discovery is retained across widget changes; replacing a prepared workflow closes its previous session.

"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from IPython.display import HTML

from src import analysis, common
from src.analysis.presentation import analysis_compute as compute

from . import evaluation_run_discovery as run_discovery
from . import evaluation_selection as selection
from . import evaluation_transient_artifact as transient_artifact
from . import evaluation_workflow as workflow

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import ipywidgets as widgets

WorkspaceKind = Literal["single", "comparison"]

_MINIMUM_COMPARISON_RUNS = 2
ArtifactRole = transient_artifact.DatasetRole
_ARTIFACT_ROLES: tuple[ArtifactRole, ...] = transient_artifact.DATASET_ROLES
_GPU_NOTE = "GPU work is limited to transient aggregate error reductions under a Slurm GPU allocation; other views use CPU."


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


@dataclass(frozen=True, slots=True)
class _Outcome:
    """What loading one selection came to: the prepared workflow or none, the summary line, its level, and the details."""

    prepared: workflow.PreparedEvaluationWorkflow | None
    summary: str
    level: analysis.ui.controls.Level
    details: str


def _artifact_for_role(
    run: run_discovery.EvaluationRunDiscovery,
    role: ArtifactRole,
) -> run_discovery.EvaluationArtifactInspection | None:
    """Return one selected artifact inspection."""
    return run.id_artifact if role == "id" else run.ood_artifacts.get(role)


def _role_label(role: str) -> str:
    """Return the exact task-owned artifact presentation term."""
    if role == "all":
        return "All"
    if role not in _ARTIFACT_ROLES:
        message = f"Unsupported Evaluation regime {role!r}."
        raise ValueError(message)
    return analysis.presentation.display_labels.evaluation_role_display_label(role)


def _model_labels(runs: Sequence[run_discovery.EvaluationRunDiscovery]) -> tuple[str, ...]:
    """Return short unique model labels for the panel's frames; the selector shows the full run names."""
    bases = []
    for run in runs:
        parts = [run.model_label, f"seed {run.seed}" if run.seed is not None else "seed unavailable"]
        if run.stage is not None and run.stage.lower() not in run.model_label.lower():
            parts.append(run.stage)
        bases.append(" · ".join(parts))
    facets = tuple((run.pair_label, run.config_label, run.dataset_id, run.run_name, str(run.run_dir)) for run in runs)
    return analysis.presentation.display_labels.unique_display_labels(bases, facets)


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
    issues = [f"{run.model_label}: {reason}" for run in runs for reason in run.metadata_issues]
    for label, field in fields.items():
        values = {getattr(run, field) for run in runs}
        if len(values) > 1:
            issues.append(f"Selected runs have incompatible {label}.")
    return tuple(issues)


def _compute_text(requested: str) -> str:
    """Return the HTML paragraph that states the analysis compute route."""
    return f"<p><b>Analysis compute:</b> {escape(compute.selection_status(requested))}. {_GPU_NOTE}</p>"


def _dispose_workflow(prepared: workflow.PreparedEvaluationWorkflow) -> None:
    """Close the panel and its rendered views before the session they read from."""
    analysis.evaluation.panel.close_panel(prepared.panel)
    prepared.close()


class _EvaluationWorkspaceController:
    """Bind one cached run catalog to task-inferred load-only workflows."""

    def __init__(
        self,
        catalog: run_discovery.EvaluationRunCatalog,
        *,
        kind: WorkspaceKind,
        title: str,
        evaluation_compute_device: str,
    ) -> None:
        """Create one run explorer without loading numerical payloads."""
        ui = analysis.ui
        self.catalog = catalog
        self.kind = kind
        self.title = title
        self.widths = ui.notebook.WidthScope()
        self.compute_preference = compute.AnalysisComputePreference(evaluation_compute_device)
        self._compute_text = _compute_text(self.compute_preference.requested)
        self._details = ""
        self._updating_compute = False
        compute_options = tuple(dict.fromkeys((evaluation_compute_device, "auto", "cpu", "cuda", *compute.visible_cuda_options())))
        self.compute_selector = ui.controls.dropdown(compute_options, description="Evaluation compute", value=evaluation_compute_device)
        self.compute_selector.observe(self._compute_device_changed, names="value")
        self.state = selection.EvaluationSelectionState(
            catalog,
            comparison=kind == "comparison",
        )
        self._workflow: workflow.PreparedEvaluationWorkflow | None = None
        self.detail_slot = ui.panel.PlotSlot()
        options = [
            ui.selectors.RunOption(
                name=run.run_name,
                family=run.model_family,
                key=str(run.run_dir),
                issue="; ".join(run.metadata_issues) or None,
            )
            for run in catalog.runs
        ]
        self.runs = ui.selectors.RunSelector(options, multiple=kind == "comparison", selected=self._initial_runs())
        self.runs.observe(self._render)
        details_header, details_body = ui.panel.disclosure("Details", self.detail_slot.widget)
        self.view = ui.panel.ViewPanel(
            [ui.panel.row(self.runs.header, self.compute_selector, details_header)],
            slot=ui.panel.PlotSlot(),
            popovers=[(self.runs.header, self.runs.body), (details_header, details_body)],
        )
        self.view.widget.children = (ui.style.stylesheet_widget(), *self.view.widget.children)
        self.panel = self.view.widget
        self._render()

    def close(self) -> None:
        """Release the loaded artifact state and clear the panel, retaining the cached run catalog."""
        previous, self._workflow = self._workflow, None
        analysis.ui.viewers.render_view(self.view, lambda: None)
        if previous is not None:
            _dispose_workflow(previous)

    def _initial_runs(self) -> tuple[str, ...]:
        """Return the run (single) or the first compatible ready pair (comparison) the selector starts on."""
        known = {spec.family for spec in analysis.presentation.visual_semantics.ARCHITECTURES}
        runs = tuple(run for run in self.catalog.runs if run.model_family in known)
        if self.kind == "single":
            return tuple(str(run.run_dir) for run in runs[:1])
        for left_index, left in enumerate(runs):
            for right in runs[left_index + 1 :]:
                candidate = (left, right)
                if not _compatibility_issues(candidate) and self._common_dataset_bindings(candidate):
                    return tuple(str(run.run_dir) for run in candidate)
        return ()

    def _say(self, summary: str, level: analysis.ui.controls.Level, details: str) -> None:
        """Show the one-line summary under the selectors and the details behind the «Details» disclosure."""
        self.view.status.set(summary, level)
        self._details = details
        self._show_details()

    def _show_details(self) -> None:
        self.detail_slot.show((HTML(f"{self._compute_text}{self._details}"),))

    def _compute_device_changed(self, change: Mapping[str, object]) -> None:
        """Update the one shared route while retaining mounted field choices."""
        if self._updating_compute:
            return
        requested = change.get("new")
        if not isinstance(requested, str):
            return
        try:
            self.compute_preference.set_requested(requested)
        except (ValueError, RuntimeError) as error:
            self._updating_compute = True
            try:
                self.compute_selector.value = self.compute_preference.requested
            finally:
                self._updating_compute = False
            self.view.status.set(f"Analysis compute unavailable: {error}", "warning")
            return
        self._compute_text = _compute_text(requested)
        self._show_details()

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
    ) -> tuple[ArtifactRole, ...]:
        """Return every complete or explicitly scoped artifact role for one run."""
        return tuple(
            role
            for role in _ARTIFACT_ROLES
            if (artifact := _artifact_for_role(run, role)) is not None and artifact.state in {"ready", "scoped_partial"}
        )

    @staticmethod
    def _artifact_role(
        roles: Sequence[ArtifactRole],
    ) -> selection.ArtifactRole:
        """Collapse an available role inventory into shared selection state."""
        admitted = tuple(roles)
        if len(admitted) > 1:
            return "all"
        if len(admitted) == 1:
            return admitted[0]
        return "id"

    @staticmethod
    def _dataset_binding_label(
        *,
        dataset_name: str,
        role: ArtifactRole,
        declared_role: str | None = None,
    ) -> str:
        """Display the authored Dataset name and its exact evaluation role."""
        role_text = analysis.presentation.display_labels.evaluation_role_display_label(declared_role) if declared_role else _role_label(role)
        return f"{dataset_name} · {role_text}"

    def _dataset_bindings(
        self,
        run: run_discovery.EvaluationRunDiscovery,
    ) -> dict[str, tuple[str, ArtifactRole]]:
        """Return exact available Dataset choices and their artifact roles."""
        bindings: dict[str, tuple[str, ArtifactRole]] = {}
        if run.dataset_id is not None and run.id_artifact.state == "ready":
            bindings[f"id:{run.dataset_id}"] = (
                self._dataset_binding_label(dataset_name=run.dataset_id, role="id"),
                "id",
            )
        for role, artifact in run.ood_artifacts.items():
            if artifact.state != "ready":
                continue
            dataset_ids = run.ood_dataset_ids[role]
            identity = common.serialization.canonical_json_sha256({"source_dataset_ids": list(dataset_ids)})
            labels = tuple(
                self._dataset_binding_label(dataset_name=dataset_name, role=artifact.role, declared_role=str(assignment["role"]))
                for dataset_name, assignment in zip(dataset_ids, run.evaluation_assignments[role], strict=True)
            )
            bindings[f"{role}:{identity}"] = (" + ".join(dict.fromkeys(labels)), artifact.role)
        return bindings

    def _common_dataset_bindings(
        self,
        runs: Sequence[run_discovery.EvaluationRunDiscovery],
    ) -> dict[str, tuple[str, ArtifactRole]]:
        """Return the exact available Material/Dataset intersection."""
        admitted = tuple(runs)
        if not admitted:
            return {}
        inventories = tuple(self._dataset_bindings(run) for run in admitted)
        keys = set(inventories[0])
        for inventory in inventories[1:]:
            keys.intersection_update(inventory)
        result: dict[str, tuple[str, ArtifactRole]] = {}
        for key in sorted(keys):
            values = {inventory[key] for inventory in inventories}
            if len(values) == 1:
                result[key] = next(iter(values))
        return result

    def _unshared_roles(
        self,
        runs: Sequence[run_discovery.EvaluationRunDiscovery],
        shared: Sequence[ArtifactRole],
    ) -> dict[str, str]:
        """Return why each role some selected run holds is not offered: a run without it ready, or another Dataset binding."""
        refused: dict[str, str] = {}
        for role in _ARTIFACT_ROLES:
            if role in shared:
                continue
            lacking = [run.label for run in runs if (artifact := _artifact_for_role(run, role)) is None or artifact.state != "ready"]
            if len(lacking) < len(runs):
                refused[role] = f"nicht bereit in {', '.join(lacking)}" if lacking else "andere Datensatzbindung je Lauf"
        return refused

    def _status_html(
        self,
        run: run_discovery.EvaluationRunDiscovery,
        *,
        selected_roles: Sequence[ArtifactRole],
        detail: str | None = None,
        material_coverage: Sequence[tuple[str, str, ArtifactRole]] = (),
    ) -> str:
        """Return one compact EDA-aligned overview and lifecycle action."""
        del selected_roles
        coverage_roles = _ARTIFACT_ROLES
        coverage = (
            "; ".join(
                f"{analysis.presentation.display_labels.material_role_display_label(material, declared_role)} · "
                f"{_artifact_state_label(cast('run_discovery.EvaluationArtifactInspection', _artifact_for_role(run, slot)))}"
                for material, declared_role, slot in material_coverage
            )
            if material_coverage
            else "; ".join(
                f"{_role_label(role)} · {_artifact_state_label(artifact)}"
                for role in coverage_roles
                if (artifact := _artifact_for_role(run, role)) is not None
            )
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
            ("Metadata", "; ".join(run.metadata_issues) if run.metadata_issues else "Available"),
            ("Run name", run.run_name),
            ("Dataset", run.dataset_id or "unavailable"),
            ("Run directory", str(run.run_dir)),
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
        states = {artifact.state for artifact in run.artifacts()}
        command = ""
        if states.intersection({"missing", "scoped_partial"}):
            command = f"<p><b>Generate full artifact coverage</b><br><code>{escape(run.artifact_command)}</code></p>"
        extra = "" if detail is None else f"<p>{escape(detail)}</p>"
        return f"<h3>Run overview</h3><table style='overflow-wrap:anywhere'>{table}</table>{extra}{command}"

    def _publish(self, load: Callable[[], _Outcome]) -> None:
        """
        Load one selection inside the publication of the panel, so only the newest selection shows or reports its outcome.

        A selection made while ``load`` runs, or while its panel is displayed, supersedes it: its prepared panel is closed
        unshown and its summary, details and failure are dropped. The panel it replaces is closed after the new one is shown.
        """
        outcomes: list[_Outcome] = []
        requested: list[int] = []

        def plot() -> widgets.Widget | None:
            requested.append(self.view.generation)  # ``render_view`` superseded before calling this: the generation of this request
            outcomes.append(load())
            return None if outcomes[0].prepared is None else outcomes[0].prepared.panel

        published = analysis.ui.viewers.render_view(self.view, plot)
        outcome = outcomes[0] if outcomes else None
        if not published or requested[0] != self.view.generation:
            if outcome is not None and outcome.prepared is not None:
                _dispose_workflow(outcome.prepared)
            return
        previous, self._workflow = self._workflow, None if outcome is None else outcome.prepared
        if outcome is None:  # the load itself failed; the panel shows the failure in the plot's place
            self._say("Loading the Evaluation failed; the failure is shown below.", "error", "")
        else:
            self._say(outcome.summary, outcome.level, outcome.details)
        if previous is not None:
            _dispose_workflow(previous)

    def _render(self) -> None:
        """Dispatch the selected runs through their persisted task metadata."""
        if self.kind == "single":
            self._render_single()
        else:
            self._render_comparison()

    def _render_single(self) -> None:
        """Render every available Dataset role for one exact selected run."""
        keys = self.runs.selected()
        if not keys:
            self._publish(lambda: _Outcome(None, "Select a run.", "warning", "<p>No run with architecture metadata is selected.</p>"))
            return
        run = self._run_by_path(str(keys[0]))
        group = self._group_for_run(run)
        self.state.select_experiment(group.identity_sha256, run_dirs=(run.run_dir,))
        selected_roles = self._available_roles(run)
        self.state.select_artifact_role(self._artifact_role(selected_roles))
        self._publish(lambda: self._load_single(run, selected_roles))

    def _load_single(self, run: run_discovery.EvaluationRunDiscovery, selected_roles: tuple[ArtifactRole, ...]) -> _Outcome:
        """Prepare the workflow of one run, or say why none is prepared; the caller publishes the outcome."""
        if not run.evaluable:
            detail = "This run is not terminal and checkpoint-admitted for Evaluation."
            return _Outcome(None, detail, "warning", self._status_html(run, selected_roles=selected_roles, detail=detail))
        if not selected_roles:
            states = {artifact.state for artifact in run.artifacts()}
            if "generating" in states:
                detail = "Artifact generation is active; completed staging payloads remain hidden until atomic publication."
            elif "invalid" in states:
                detail = "Artifact evidence is invalid or stale and was not admitted."
            else:
                detail = "No completed artifact is available; generation remains an explicit host workflow."
            return _Outcome(None, detail, "warning", self._status_html(run, selected_roles=(), detail=detail))
        scoped_roles: tuple[ArtifactRole, ...] = tuple(
            role for role in selected_roles if (artifact := _artifact_for_role(run, role)) is not None and artifact.state == "scoped_partial"
        )
        artifact_roots: dict[workflow.ArtifactSelectionRole, Path] = {
            role: artifact.root for role in scoped_roles if (artifact := _artifact_for_role(run, role)) is not None
        }
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
                evaluation_compute_device=self.compute_preference,
                widths=self.widths,
            )
        except Exception as error:  # noqa: BLE001 -- bounded notebook status
            detail = f"Artifact admission failed ({type(error).__name__}): {error}"
            return _Outcome(None, detail, "error", self._status_html(run, selected_roles=selected_roles, detail=detail))
        try:
            return self._single_outcome(run, selected_roles, scoped_roles, prepared)
        except BaseException:
            _dispose_workflow(prepared)  # nothing publishes it: its session must not stay open
            raise

    def _single_outcome(
        self,
        run: run_discovery.EvaluationRunDiscovery,
        selected_roles: tuple[ArtifactRole, ...],
        scoped_roles: tuple[ArtifactRole, ...],
        prepared: workflow.PreparedEvaluationWorkflow,
    ) -> _Outcome:
        """Return the outcome of one prepared single-run workflow: its summary and the run overview with the material coverage."""
        unavailable = tuple(role for role in _ARTIFACT_ROLES if role not in selected_roles)
        availability_detail: str | None = None
        if scoped_roles:
            availability_detail = "Selected-case evidence is admitted for Single-case Evaluation only; aggregate coverage remains unavailable."
        elif unavailable:
            availability_detail = (
                "Available Dataset roles were loaded automatically; unavailable roles remain capability-gated: "
                + ", ".join(_role_label(role) for role in unavailable)
                + "."
            )
        material_coverage: tuple[tuple[str, str, ArtifactRole], ...] = ()
        session_inventory = getattr(getattr(prepared, "session", None), "partitioned_case_inventory", None)
        if callable(session_inventory):
            entries = tuple(cast("Any", session_inventory)())
            unique: dict[tuple[str, str, ArtifactRole], tuple[int, int]] = {}
            artifact_positions: dict[Path, int] = {}
            for entry in entries:
                role = cast("ArtifactRole", entry.dataset_role)
                assignment = getattr(entry, "evaluation_assignment", None)
                declared_role = str(assignment["role"]) if isinstance(assignment, Mapping) else role
                artifact_position = artifact_positions.setdefault(
                    entry.artifact_root,
                    len(artifact_positions),
                )
                material_position = entry.dataset_name.find(entry.material_family)
                unique.setdefault(
                    (entry.material_family, declared_role, role),
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
        return _Outcome(
            prepared,
            f"{run.run_name}: {len(selected_roles)} of {len(_ARTIFACT_ROLES)} dataset roles loaded.",
            "normal",
            self._status_html(run, selected_roles=selected_roles, detail=availability_detail, material_coverage=material_coverage),
        )

    def _render_comparison(self) -> None:
        """Render one task-compatible comparison over every dataset role the selected runs share."""
        selected = tuple(self._run_by_path(str(key)) for key in self.runs.selected())
        issues = list(_compatibility_issues(selected))
        bindings = self._common_dataset_bindings(selected)
        if selected and not bindings:
            issues.append("Selected runs share no available Material/Dataset artifact.")
        if issues:
            items = "".join(f"<li>{escape(issue)}</li>" for issue in dict.fromkeys(issues))
            note = "All discovered runs remain visible; incompatible artifacts are capability-gated."
            html = f"<h3>Compatible selected runs</h3><ul>{items}</ul><p>{note}</p>"
            self._publish(lambda: _Outcome(None, issues[0], "warning", html))
            return
        shared = {role for _label, role in bindings.values()}
        roles: tuple[ArtifactRole, ...] = tuple(role for role in _ARTIFACT_ROLES if role in shared)
        self.state.select_catalog_runs(tuple(run.run_dir for run in selected))
        self.state.select_artifact_role(self._artifact_role(roles))
        self._publish(lambda: self._load_comparison(selected, roles))

    def _load_comparison(self, selected: tuple[run_discovery.EvaluationRunDiscovery, ...], roles: tuple[ArtifactRole, ...]) -> _Outcome:
        """Prepare the comparison of the selected runs over ``roles``, or say why none is prepared; the caller publishes the outcome."""
        try:
            prepared = workflow.prepare_model_comparison_evaluation_workspace(
                tuple(run.run_dir for run in selected),
                labels=_model_labels(selected),
                expected_task=selected[0].task,
                artifact_roles=roles,
                auto_build_missing=False,
                rebuild_incompatible=False,
                sections="all",
                selection_state=self.state,
                evaluation_compute_device=self.compute_preference,
                widths=self.widths,
                refused_roles=self._unshared_roles(selected, roles),
            )
        except Exception as error:  # noqa: BLE001 -- bounded notebook status
            detail = f"{type(error).__name__}: {error}"
            return _Outcome(None, f"Compatibility warning ({detail})", "error", f"<h3>Compatibility warning</h3><p>{escape(detail)}</p>")
        try:
            return self._comparison_outcome(selected, roles, prepared)
        except BaseException:
            _dispose_workflow(prepared)  # nothing publishes it: its session must not stay open
            raise

    @staticmethod
    def _comparison_outcome(
        selected: tuple[run_discovery.EvaluationRunDiscovery, ...],
        roles: tuple[ArtifactRole, ...],
        prepared: workflow.PreparedEvaluationWorkflow,
    ) -> _Outcome:
        """Return the outcome of one prepared comparison: its summary and the roles it admitted and refused."""
        admitted = [role for role in roles if role not in prepared.refused_roles]
        refused = "".join(f"<li>{escape(_role_label(role))}: {escape(reason)}</li>" for role, reason in prepared.refused_roles.items())
        names = escape(", ".join(_role_label(role) for role in admitted))
        return _Outcome(
            prepared,
            f"{len(selected)} runs compared on {len(admitted)} of {len(_ARTIFACT_ROLES)} dataset roles.",
            "normal",
            f"<h3>Compatible selected runs</h3><p>{len(selected)} artifacts admitted for {names} using exact persisted metadata.</p>"
            + (f"<p>Not available:</p><ul>{refused}</ul>" if refused else ""),
        )


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
    evaluation_compute_device: str = "auto",
) -> PreparedEvaluationWorkspace:
    """Discover runs once and prepare one automatic Evaluation notebook workspace."""
    if kind not in {"single", "comparison"}:
        message = f"Unsupported Evaluation workspace kind {kind!r}."
        raise ValueError(message)
    if experiments_root is not None and storage_root is not None:
        message = "Provide experiments_root or storage_root, not both."
        raise ValueError(message)
    compute.admit_analysis_request(evaluation_compute_device)
    compute_status = compute.selection_status(evaluation_compute_device)
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
        placeholder = analysis.ui.panel.ViewPanel((), status=analysis.ui.controls.StatusLine(message, "warning"))
        return PreparedEvaluationWorkspace(
            catalog=empty_catalog,
            selection_state=None,
            panel=placeholder.widget,
            summary_text=_summary(empty_catalog, kind=kind) + f"\nAnalysis compute: {compute_status}",
            kind=kind,
            _controller=None,
        )
    controller = _EvaluationWorkspaceController(
        catalog,
        kind=kind,
        title=title,
        evaluation_compute_device=evaluation_compute_device,
    )
    return PreparedEvaluationWorkspace(
        catalog=catalog,
        selection_state=controller.state,
        panel=controller.panel,
        summary_text=_summary(catalog, kind=kind) + f"\nAnalysis compute: {compute_status}",
        kind=kind,
        _controller=controller,
    )


def prepare_single_model_evaluation_workspace(
    *,
    experiments_root: Path | str | None = None,
    storage_root: Path | str | None = None,
    title: str = "Single-model Evaluation",
    evaluation_compute_device: str = "auto",
) -> PreparedEvaluationWorkspace:
    """Prepare automatic single-model Evaluation without run-name cells."""
    return prepare_evaluation_workspace(
        experiments_root=experiments_root,
        storage_root=storage_root,
        kind="single",
        title=title,
        evaluation_compute_device=evaluation_compute_device,
    )


def prepare_model_comparison_evaluation_workspace(
    *,
    experiments_root: Path | str | None = None,
    storage_root: Path | str | None = None,
    title: str = "Model comparison",
    evaluation_compute_device: str = "auto",
) -> PreparedEvaluationWorkspace:
    """Prepare automatic compatible-run comparison without a Python run list."""
    return prepare_evaluation_workspace(
        experiments_root=experiments_root,
        storage_root=storage_root,
        kind="comparison",
        title=title,
        evaluation_compute_device=evaluation_compute_device,
    )
