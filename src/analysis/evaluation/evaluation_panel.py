"""Compose curated evaluation panels over admitted artifacts.

Responsibilities:
  - Keep the approved view order explicit per task and per single-model or comparison mode
  - Number the displayed tabs and views from their final position
  - Bind every view to its defined local controls and defaults
  - Accept caller-provided context labels and session-owned canonical frames
  - Offer the datasets (material and role) of one panel in each view's own selector, kept in step through the selection state;
    a multi-dataset view opens on every selected dataset, also in a comparison and for maps
  - Open a transient snapshot view (error maps, parity) at the latest stored time every shown case holds
  - List a comparison role the workflow refused, or a material without shared cases, as unavailable with its reason
  - Preserve lazy first rendering and panel-local figure export
  - Close a replaced panel together with every view it rendered
  - Export the selected transient sample case as a GIF over all stored times
  - Keep the transient sample viewer's state-map norms over all stored times for its last case and setting while any
    field is fixed over time
  - Offer the break-even against COMSOL from each run's timing and training evidence, naming a run without it with each
    missing piece (a continued stage's A0 apart from its own seconds), the transient training curves from a lazy history
    loader, and the pooled time views over the drying phase or in hours
  - Show the tables with display headings and names, the run-level ones (configuration, work, break-even) one entry per run

Design principles:
  - Strict artifact loading and EvaluationSession binding precede construction
  - Dataset, model, channel, case, and parameter controls remain view-specific
  - Presentation labels never alter canonical identity or numerical cache keys; a transient series joins its scaler and
    records by its frame, and runs of equal labels are told apart by their frame, never merged
  - A run-level view names a run by its run label and colours it as every other view of the panel does, also a single run;
    its clocks and hardware stand as plain text below the figure
  - Panel construction opens no case payloads and computes no plot mathematics
  - Only a finite error ranks: the scoreboard numbers only runs with a finite selection objective, an outlier view without
    any finite error titles its cases unranked, and the outlier table without one stays empty
  - Every render, clear, and failure of a view goes through the view panel's publication; a nested viewer is built with its view as host

This module does NOT:
  - Discover, generate, repair, rebuild, or admit artifact payloads
  - Create an alternate dataframe or generic interaction registry
  - Infer dataset identity from display names, paths, or row positions
  - Cache figures, case arrays, or any numerical reduction beyond those last state-map norms
  - Pool transient datasets or steady datasets in a view whose reduction is not material-balanced
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial, wraps
from html import escape
from itertools import pairwise
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast
from weakref import WeakKeyDictionary

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import PillowWriter
from PIL import Image

from src import analysis
from src.analysis.evaluation import evaluation_datasets as dataset_cuts
from src.analysis.evaluation import evaluation_prediction_validity as validity_contract
from src.analysis.evaluation import evaluation_transient_artifact as transient_artifact
from src.analysis.presentation import analysis_compute as compute
from src.analysis.presentation import analysis_error_diagnostics as error_diagnostics
from src.learning.models import naming

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Hashable, Iterable
    from typing import IO

    from matplotlib.figure import Figure

    from src.analysis.evaluation.evaluation_plot.evaluation_plot_series import SeriesStyles
    from src.analysis.evaluation.evaluation_run_discovery import StageHistory
    from src.analysis.evaluation.evaluation_selection import EvaluationSelectionState
    from src.analysis.evaluation.evaluation_session import EvaluationSession
    from src.analysis.evaluation.evaluation_transient_comparison import AirflowDegradationMetrics, LineageComparison
    from src.analysis.evaluation.evaluation_transient_session import TransientEvaluationSession
    from src.analysis.presentation.analysis_dataset_selection import DatasetChoice, DatasetInventory
    from src.analysis.ui.analysis_ui_navigation import TimeNavigator, TimeStepSelection
    from src.analysis.ui.analysis_ui_notebook import LazyPanel, ViewSection, WidthScope
    from src.analysis.ui.analysis_ui_panel import ViewPanel
    from src.analysis.ui.analysis_ui_viewers import ViewControl

_MINIMUM_COMPARISON_RUNS = 2
_MINIMUM_VELOCITY_COMPONENTS = 2
_FRAME_MATERIAL_CONTEXT_SIZE = 2
_OUTLIER_COUNT = 5
_CASE_COUNT_STEP = 50
_RANKING_CRITERION: Final = (  # the one criterion of every row of the comparison summary, named once in its title
    "the selection objective of the cumulative full rollouts: case errors equal over the cases of a material and over the materials (lower is better)"
)
_DEFAULT_TRANSIENT_CHANNELS: Final = ("T", "phi", "w_gr")  # owner 14.11.: checked before any view chose, as in EDA
_GIF_DPI = 100  # owner 14.11.: tick labels sharp; three states over 169 hourly frames stay under the memory limit
_GIF_FPS = 3  # the default of the viewer's fps field, which is bounded to 1 to 10
_GIF_BYTES_PER_PIXEL = 4  # Pillow keeps RGB and RGBA frames at four bytes per pixel
# Beside the stored frames: renderer buffer 4, first or previous frame's colour keys 4, one RGB conversion with its
# raw bytes and colour keys 12, and the unchanged- or changed-pixel mask 1.
_GIF_WORKING_BYTES_PER_PIXEL = 21
_GIF_COLOUR_KEYS = 1 << 24  # every 24-bit RGB colour
_GIF_COUNT_CHUNK = 1 << 18  # colour keys counted per sort
_GIF_PALETTE_SIZE = 256
_GIF_STATIC_ENTRIES = 160  # frame 0's entries for pixels equal in every frame; its maps get the other 96
_GIF_TRANSPARENT_INDEX = 255  # unchanged pixels of a later frame; its changed pixels use the 255 entries below
_GIF_KEEP_DISPOSAL = 1  # "do not dispose": a later frame's transparent pixels show the frames before
_GIF_RESCUE_SHARE = 8  # one in eight entries goes to the colour farthest from all entries
_GIF_CANDIDATE_CAP = 1 << 16  # more distinct colours are merged into coarser RGB bins
_GIF_KMEANS_ITERATIONS = 20
_GIF_DISTANCE_BLOCK = 1 << 12  # colours per block of palette distances
_GIF_PALETTE_STEP_BYTES = (
    _GIF_COLOUR_KEYS * (4 + 1)  # pixel-count table and colour-to-index table
    + _GIF_COUNT_CHUNK * 64  # one chunk's sorted keys, mask, indices and counts
    + (1 << 18) * 6 * 8  # coarse-bin sums, their occupancy and one bin count at four values per channel
    + _GIF_CANDIDATE_CAP * 16 * 8  # candidate colours, counts, labels, distances and their temporaries
    + _GIF_DISTANCE_BLOCK * _GIF_PALETTE_SIZE * 8 * 3  # one block of distances, its channel difference and square
    + (1 << 16) * 8 * 8  # one blue plane: its counts, keys, colours and per-plane bin weights
)
_GIF_MEMORY_LIMIT_BYTES = 1_500_000_000
_GIF_PROGRESS_STEP = 10

STEADY_SINGLE_SECTION_KEYS = (
    "overview",
    "global_error",
    "error_decomposition",
    "physical_consistency",
    "spectral_analysis",
    "parameter_sensitivity",
    "sample_viewer",
    "outlier_analysis",
    "runtime",
)
STEADY_COMPARISON_SECTION_KEYS = (
    "overview",
    "global_error",
    "architecture",
    "error_decomposition",
    "physical_consistency",
    "spectral_analysis",
    "parameter_sensitivity",
    "sample_viewer",
    "outlier_analysis",
    "runtime",
)
TRANSIENT_SINGLE_SECTION_KEYS = (
    "overview",
    "global_error",
    "error_decomposition",
    "temporal_rollout",
    "sample_viewer",
    "outlier_analysis",
    "runtime",
    "matched_stages",
)
TRANSIENT_COMPARISON_SECTION_KEYS = (
    "overview",
    "global_error",
    "architecture",
    "error_decomposition",
    "temporal_rollout",
    "sample_viewer",
    "outlier_analysis",
    "runtime",
    "matched_stages",
)

EVALUATION_SECTION_TITLES = {
    "overview": "Overview",
    "global_error": "Global Error Analysis",
    "architecture": "Architecture Sensitivity",
    "error_decomposition": "Error Decomposition",
    "physical_consistency": "Physical Consistency",
    "spectral_analysis": "Spectral & Representation Analysis",
    "parameter_sensitivity": "Error Sensitivity",
    "sample_viewer": "Sample Viewer",
    "outlier_analysis": "Outlier & Extreme Case Analysis",
    "temporal_rollout": "Temporal & Rollout Analysis",
    "runtime": "Measured Runtime & Accuracy",
    "matched_stages": "Controlled Training Stages",
}


@dataclass(frozen=True, slots=True)
class EvaluationContext:
    """Define one notebook-provided dataset context and its ordered model frames."""

    key: str
    label: str
    datasets: Mapping[str, pd.DataFrame]


_NO_DATASET: Final = "No dataset is available for this view."
_POOL_NOTE: Final = "Pooling does not apply in this view: the Datasets appear separately."
_SELECTION_CONTROLS: Final = frozenset({"channel", "parameter"})  # sit beside the case navigator, the rest in the display row


def _entry(title: str, build: Callable[[dict[str, Any]], Any]) -> tuple[str, Callable[..., Any], str]:
    """Return one view-section entry whose factory hands the export identity of its view to ``build``."""

    def factory(*, export_state: dict[str, Any] | None = None, export_plot_name: str | None = None, export_title: str | None = None) -> Any:
        return build({"export_state": export_state, "export_plot_name": export_plot_name, "export_title": export_title})

    return title, factory, title


def _guard(view: ViewPanel, export: Mapping[str, Any], action: Callable[[], None]) -> Callable[[], None]:
    """Return ``action`` inside the failure boundary of ``view``: a failure replaces the plot and releases the export target."""

    def run() -> None:
        try:
            action()
        except Exception as error:  # noqa: BLE001 -- UI boundary shows the failure in place
            analysis.ui.viewers.fail_view(view, error, **export)

    return run


class _SharedChoice:
    """One dataset selector kept in step with the selection state: it starts from the shared choice and publishes the user's."""

    def __init__(
        self,
        inventory: DatasetInventory,
        *,
        multiple: bool,
        pool_reason: str | None,
        selection_state: EvaluationSelectionState | None,
    ) -> None:
        """Create the selector on the shared choice, else on every dataset (several) or the first one (single)."""
        self._state = selection_state
        self._multiple = multiple
        self._keys = {item.key for item in inventory.items}
        wanted = self._shared()
        default = [item.key for item in inventory.items] if multiple else []
        self.selector = analysis.ui.selectors.DatasetSelector(
            inventory,
            multiple=multiple,
            selected=default if wanted is None else wanted[0],
            pooled=wanted is not None and wanted[1],
            pool_reason=pool_reason,
        )
        self.selector.observe(self._publish)

    @property
    def pooled_elsewhere(self) -> bool:
        """Return whether the shared choice pools materials but this view cannot, so it draws them separately."""
        return self._state is not None and self._state.selection.pooled and self.selector.inventory.poolable and not self.selector.pool_enabled

    def _shared(self) -> tuple[list[Hashable], bool] | None:
        """Return the shared choice cut to the keys this selector offers, or ``None`` when it holds none."""
        if self._state is None:
            return None
        shared = self._state.selection
        if self._multiple:
            if shared.datasets is None:
                return None
            if not shared.datasets:
                return [], shared.pooled  # the user cleared every dataset: that is a choice, not an absent one
            keys = [key for key in shared.datasets if key in self._keys]
        else:
            keys = [shared.focus_dataset] if shared.focus_dataset in self._keys else []
        return (keys, shared.pooled) if keys else None

    def pull(self) -> bool:
        """Apply the shared choice to the selector without notifying its observers; return whether the selector changed."""
        wanted = self._shared()
        if wanted is None:
            return False
        keys, pooled = wanted
        pooled = pooled and self.selector.pool_enabled
        if set(keys) == set(self.selector.selected()) and pooled == self.selector.pooled:
            return False
        self.selector.set_state(keys, pooled)
        return True

    def _publish(self) -> None:
        """Retain this selector's choice in the selection state."""
        if self._state is None:
            return
        selected = self.selector.selected()
        if self._multiple:
            pooled = self.selector.pooled if self.selector.pool_enabled else self._state.selection.pooled
            self._state.select_datasets(selected, pooled=pooled)
        elif selected:
            self._state.select_focus_dataset(selected[0])


class _SteadyScope:
    """
    The datasets of one steady panel: their inventory, the frames of every role, and the cuts the views draw.

    ``frames_by_role`` maps a role to its models' frames, each keyed by its run label; ``run_frames`` are the frames of the
    per-artifact views (a single model's roles keyed by role), while the run-level views take the first role's frame of every
    run under its run label. A single-model panel draws one cut per selected dataset; a comparison draws every selected
    dataset, cut for every model, keyed by model label for one dataset and ``model · dataset`` for several; a pooled dataset
    is one cut per material, whose key names the material. Every cut is a ``DisplayFrames`` whose ``labels`` give each key's
    display dataset and model, and whose ``unnamed`` model labels (storage aliases) no title names. A cut is made once per
    choice and shared by the views.
    """

    def __init__(
        self, contexts: Sequence[EvaluationContext], *, comparison: bool, refused_roles: Mapping[str, str], unnamed: Collection[str] = ()
    ) -> None:
        """Build the scope from the role contexts of a steady panel."""
        self.unnamed = frozenset(unnamed)
        self.frames_by_role = {context.key: dict(context.datasets) for context in contexts}
        if comparison:
            self.inventory = dataset_cuts.comparison_inventory(self.frames_by_role, refused=refused_roles)
            self.run_frames = dict(contexts[0].datasets)
        else:
            self.inventory = dataset_cuts.steady_inventory({role: next(iter(models.values())) for role, models in self.frames_by_role.items()})
            self.run_frames = {context.label: next(iter(context.datasets.values())) for context in contexts}
        self.comparison = comparison
        self.models = tuple(self.run_frames) if comparison else ()  # the compared models; a single-model panel has none
        self._cuts: dict[tuple[Any, ...], dataset_cuts.DisplayFrames] = {}

    def frames(self, choices: Sequence[DatasetChoice]) -> dataset_cuts.DisplayFrames:
        """Return the frames a view draws for ``choices``: one cut per dataset label, or in a comparison one per model."""
        available = [choice for choice in choices if choice.available]
        key = tuple((choice.group, choice.members) for choice in available)
        if key not in self._cuts:
            cut = self._cut(available)
            self._cuts[key] = dataset_cuts.DisplayFrames(cut, cut.labels, unnamed=self.unnamed)
        return self._cuts[key]

    def _cut(self, available: Sequence[DatasetChoice]) -> dataset_cuts.DisplayFrames:
        if not available:
            return dataset_cuts.DisplayFrames({}, {})
        if self.comparison:
            cuts = {choice.label: dataset_cuts.comparison_frames(self.frames_by_role[choice.group], choice) for choice in available}
            if len(cuts) == 1:
                return next(iter(cuts.values()))
            keyed: dict[str, pd.DataFrame] = {}
            labels: dict[str, tuple[str, str]] = {}
            for label, by_model in cuts.items():
                for inner, frame in by_model.items():
                    model = by_model.labels[inner][1]
                    key = f"{inner} · {label}"
                    if key in keyed:
                        message = f"Comparison key {key!r} names two frames (model {model!r} of dataset {label!r} and another pair)."
                        raise ValueError(message)
                    keyed[key], labels[key] = frame, (label, model)
            return dataset_cuts.DisplayFrames(keyed, labels)
        first = next(iter(self.frames_by_role.values()))
        return dataset_cuts.steady_frames(
            {role: next(iter(models.values())) for role, models in self.frames_by_role.items()}, available, model=next(iter(first))
        )


def _shown_count(nested: ViewPanel) -> int:
    """Return the case count a nested case-count viewer shows: the slider of the first element of its first row."""
    count_row = cast("Any", nested.rows[0])
    return int(count_row.children[0].children[0].value)


def _dataset_view(
    scope: _SteadyScope,
    *,
    plot_func: Callable[..., Any],
    selection_state: EvaluationSelectionState | None,
    export: Mapping[str, Any],
    pooled: bool = False,
    single_model: bool = False,
    plot_kwargs: Mapping[str, Any] | None = None,
    count: tuple[int, int] | None = None,
    controls: Mapping[str, ViewControl] | None = None,
) -> ViewPanel:
    """
    Build one steady view over the datasets its selector offers.

    A view offers several datasets and opens on every selected one, maps included, or one for a ``single_model`` view; in a
    comparison a ``single_model`` view adds a model choice, and the other views draw the compared models as series of every selected
    dataset. ``pooled`` allows the pooling switch where the view weights every material equally, one model or several. The plot
    is drawn directly, or with ``count`` (start, step) as a case-count viewer built inside this view for the selected datasets, at
    the count the former one showed.
    """
    ui = analysis.ui
    choice = _SharedChoice(
        scope.inventory,
        multiple=not single_model,
        pool_reason=None if pooled else dataset_cuts.POOL_DISABLED_REASON,
        selection_state=selection_state,
    )
    model = ui.controls.dropdown(scope.models, description="Model") if single_model and scope.comparison else None
    view: ViewPanel | None = None
    nested: ViewPanel | None = None
    stale = False  # the last draw was superseded before it published: the view draws again when it is shown

    def update_now() -> None:
        nonlocal nested, stale
        current = cast("ViewPanel", view)
        frames = scope.frames(choice.selector.choices())
        if not frames:
            ui.viewers.clear_view(current, ui.viewers.EMPTY_SELECTION_MESSAGE if scope.inventory.items else _NO_DATASET, **export)
            return
        if model is not None:
            frames = frames.only(str(model.value))
        if choice.pooled_elsewhere:
            current.status.set(_POOL_NOTE)
        else:
            current.status.clear()
        if count is None:
            stale = not ui.viewers.render_view(current, plot_func, kwargs={"datasets": frames, **(plot_kwargs or {})}, **export)
            return
        carried: dict[str, Any] = {} if nested is None else {"resume_cases": _shown_count(nested)}
        built: list[ViewPanel] = []
        start_cases, step_size = count

        def build() -> widgets.Widget:
            built.append(
                ui.viewers.case_count_view(
                    plot_func,
                    datasets=cast("Mapping[Hashable, pd.DataFrame]", frames),
                    start_cases=start_cases,
                    step_size=step_size,
                    controls=controls,
                    host=current,
                    **carried,
                    **export,
                    **(plot_kwargs or {}),
                )
            )
            return built[0].widget

        # A nested viewer is kept only when its render of this view published; the one it replaces, or the unpublished one, stops
        # following the controls they share.
        stale = not ui.viewers.render_view(current, build)
        if not stale:
            if nested is not None:
                nested.deactivate()
            nested = built[0] if built else nested
        elif built:
            built[0].deactivate()

    def activate() -> None:
        current = cast("ViewPanel", view)
        if nested is not None:
            # Hiding superseded this view's render, which the nested viewer is bound to: bind it to the render on show again and
            # let it draw what its controls asked for meanwhile.
            nested.host = (current, current.generation)
            nested.activate()
        if choice.pull() or stale:
            current.request(update)

    def deactivate() -> None:
        if nested is not None:
            nested.deactivate()

    elements = [choice.selector.header] if model is None else [choice.selector.header, model]
    view = ui.panel.ViewPanel(
        [ui.panel.row(*elements)],
        slot=ui.panel.PlotSlot(),
        popovers=[(choice.selector.header, choice.selector.body)],
        on_activate=activate,
        on_deactivate=deactivate,
    )
    update = _guard(view, export, update_now)
    choice.selector.observe(lambda: cast("ViewPanel", view).request(update))
    if model is not None:
        model.observe(lambda _change: cast("ViewPanel", view).request(update), names="value")
    view.request(update)
    return view


def _model_run_selector(scope: _SteadyScope) -> analysis.ui.selectors.RunSelector:
    """
    Return the multiple run choice over the models of every role, the first two with an offered architecture selected.

    A model's family comes from its frame's provenance; one without it, or of an unknown architecture, stays visible and disabled.
    """
    known = {spec.family for spec in analysis.presentation.visual_semantics.ARCHITECTURES}
    options: list[analysis.ui.selectors.RunOption] = []
    for models in scope.frames_by_role.values():
        for name, frame in models.items():
            if any(option.key == name for option in options):
                continue
            try:
                family, _trainable, physics_enabled = analysis.evaluation.plots.series.model_metadata(frame)
                options.append(analysis.ui.selectors.RunOption(name, f"pi-{family}" if physics_enabled else family, name))
            except TypeError as error:
                options.append(analysis.ui.selectors.RunOption(name, None, name, issue=str(error)))
    selectable = [option.key for option in options if option.family in known]
    return analysis.ui.selectors.RunSelector(options, multiple=True, selected=selectable[:2])


def _run_parts(
    runs: analysis.ui.selectors.RunSelector | None, on_change: Callable[[], None]
) -> tuple[list[widgets.Widget], list[tuple[widgets.ToggleButton, widgets.DOMWidget]]]:
    """Return the run choice's header for the dataset row and its (header, body) popover, both empty without it; it calls ``on_change``."""
    if runs is None:
        return [], []
    runs.observe(on_change)
    return [runs.header], [(runs.header, runs.body)]


def _drawn_models(
    runs: analysis.ui.selectors.RunSelector | None, shown: Iterable[str], status: analysis.ui.controls.StatusLine, role: str
) -> dict[str, tuple[str, ...]]:
    """
    Return the plot keyword ``models``, or nothing without a run choice: the selected models the drawn role holds.

    A selected model the role lacks is named on ``status``. With none left the first two it holds are selected, and the selector follows.
    """
    if runs is None:
        return {}
    held = tuple(shown)
    dropped = [str(key) for key in runs.selected() if key not in held]
    if dropped:
        status.set(f"{', '.join(dropped)}: no artifacts for {role}.", "warning")
    chosen = tuple(str(key) for key in runs.selected() if key in held)
    if not chosen:
        chosen = tuple(name for name in held if name in runs.boxes and not runs.boxes[name].disabled)[:2]
    if chosen != tuple(str(key) for key in runs.selected()):
        runs.select(chosen)
    return {"models": chosen}


def _generation_case_labels(frames: Mapping[str, pd.DataFrame], positions: Sequence[int], *, rows: bool) -> dict[int, str]:
    """Return the generation case of each one-based row position of the first frame (the models are case-matched); none off ``rows``."""
    first = next(iter(frames.values()), None)
    if first is None or not rows:
        return {}
    return {position: f"generation case {int(first.iloc[position - 1]['case_index'])}" for position in positions if position <= len(first)}


def _indexed_view(
    scope: _SteadyScope,
    *,
    plot_func: Callable[..., Any],
    controls: Mapping[str, widgets.ValueWidget],
    selection_state: EvaluationSelectionState | None,
    export: Mapping[str, Any],
    single_model: bool,
    sync: bool = False,
    choose_models: bool = False,
    channel_fields: Sequence[Any] = (),
    offered: Callable[[int], int] | None = None,
    selection_name: Callable[[int, int], str] | None = None,
    index_to_kwargs: Callable[[int], Mapping[str, Any]] | None = None,
) -> ViewPanel:
    """
    Build one steady single-case view: dataset choice, case navigator, and the view's controls.

    A comparison adds a model choice. The navigator steps the one-based positions the plot offers: the rows of the drawn
    frames, or, for a plot that resolves its own selection (ranked outliers and a reference, minimum and maximum),
    ``offered(rows)`` positions that ``selection_name(index, offered)`` names in the export stem; on rows, the navigator's status
    line names the generation case of the shown position (of the first drawn frame, the models being case-matched). With ``sync`` the case,
    and the channel of a ``channel`` control, are shared with the other views through the selection state: a change is
    published, and the view takes the shared case and channel when it becomes active. ``choose_models`` offers the compared models as
    one multiple run choice (the first two selected) that the plot receives as ``models``, limited to the drawn role's models.
    """
    ui = analysis.ui
    choice = _SharedChoice(scope.inventory, multiple=False, pool_reason=dataset_cuts.POOL_DISABLED_REASON, selection_state=selection_state)
    runs = _model_run_selector(scope) if choose_models and scope.comparison else None
    model = ui.controls.dropdown(scope.models, description="Model") if single_model and scope.comparison else None
    state = selection_state if sync else None
    key_by_value = {name: str(field.key) for field in channel_fields for name in (str(field.key), str(field.label))}
    value_by_key = {str(field.key): str(field.label) for field in channel_fields}
    view: ViewPanel | None = None
    taking = False  # while the view takes the shared case and channel, its own controls must not publish theirs over them

    def drawn() -> dataset_cuts.DisplayFrames:
        frames = scope.frames(choice.selector.choices())
        return frames.only(str(model.value)) if frames and model is not None else frames

    def positions() -> tuple[int, ...]:
        count = max(min((len(frame) for frame in drawn().values()), default=1), 1)
        return tuple(range(1, (count if offered is None else offered(count)) + 1))

    def case_id_at(frame: pd.DataFrame, position: int) -> str:
        return str(int(frame.iloc[position - 1]["case_index"]))

    def publish() -> None:
        """Publish the shown case and the channel to the other views."""
        frames = drawn()
        if state is None or taking or not frames:
            return
        state.select_case(case_id_at(next(iter(frames.values())), navigator.value))
        channel = controls.get("channel")
        if channel is not None:
            value = str(channel.value)
            state.select_channels((key_by_value.get(value, value),))

    def changed() -> None:
        publish()
        cast("ViewPanel", view).request(render)

    status = ui.controls.StatusLine()
    # The status line names each row's generation case; a plot that picks its cases itself (``offered``) gets none.
    labels = _generation_case_labels(drawn(), positions(), rows=offered is None)
    navigator = ui.navigation.CaseNavigator(positions(), labels=labels, status=status, on_change=lambda _position: changed())
    selection = [navigator.widget, *(control for name, control in controls.items() if name in _SELECTION_CONTROLS)]
    display = [control for name, control in controls.items() if name not in _SELECTION_CONTROLS]
    dataset_row = [choice.selector.header] if model is None else [choice.selector.header, model]
    run_header, run_popover = _run_parts(runs, changed)
    rows = [
        ui.panel.row(*dataset_row, *run_header),
        ui.panel.row(*selection),
        *([ui.panel.row(*display)] if display else []),
    ]

    def render_now() -> None:
        current = cast("ViewPanel", view)
        frames = drawn()
        if not frames:
            ui.viewers.clear_view(current, ui.viewers.EMPTY_SELECTION_MESSAGE if scope.inventory.items else _NO_DATASET, **export)
            return
        zero_based = navigator.value - 1
        index_kwargs = {"row_position": zero_based} if index_to_kwargs is None else dict(index_to_kwargs(zero_based))
        role = ", ".join(item.label for item in choice.selector.choices())
        values = {name: control.value for name, control in controls.items()} | _drawn_models(runs, frames, status, role)
        published = ui.viewers.render_view(current, plot_func, kwargs={"datasets": frames, **index_kwargs, **values}, **export)
        state_for_export = export["export_state"]
        if published and state_for_export is not None:
            channel = controls.get("channel")
            parts = (
                "steady_flow",
                "-".join(values.get("models", (next(iter(frames)),))),
                str(export["export_plot_name"] or "sample"),
                f"case_{case_id_at(next(iter(frames.values())), navigator.value)}"
                if selection_name is None
                else selection_name(zero_based, len(positions())),
                None if channel is None else f"channel_{channel.value}",
            )
            state_for_export["filename_stem"] = "__".join(part for part in parts if part is not None)

    def rebind_now() -> None:
        navigator.rebind(positions(), preserve=True, notify=False, labels=_generation_case_labels(drawn(), positions(), rows=offered is None))
        publish()
        cast("ViewPanel", view).request(render)

    def activate_now() -> None:
        """Take the dataset, case and channel the other views chose, then draw once."""
        nonlocal taking
        current = cast("ViewPanel", view)
        taking = True
        try:
            with current.batch():
                if choice.pull():
                    navigator.rebind(
                        positions(), preserve=True, notify=False, labels=_generation_case_labels(drawn(), positions(), rows=offered is None)
                    )
                shared = None if state is None else state.selection
                if shared is not None:
                    frames = drawn()
                    cases = [str(int(case)) for case in next(iter(frames.values()))["case_index"]][: len(positions())] if frames else []
                    if shared.case_id in cases:
                        navigator.field.value = cases.index(shared.case_id) + 1
                    channel = controls.get("channel")
                    if isinstance(channel, widgets.Dropdown) and shared.channels:
                        wanted = value_by_key.get(shared.channels[0], shared.channels[0])
                        if wanted in {value for _label, value in channel.options}:
                            channel.value = wanted
                current.request(render)
        finally:
            taking = False

    view = ui.panel.ViewPanel(
        rows,
        status=status,
        slot=ui.panel.PlotSlot(),
        popovers=[(choice.selector.header, choice.selector.body), *run_popover],
        on_activate=activate_now,
    )
    render = _guard(view, export, render_now)
    rebind = _guard(view, export, rebind_now)
    choice.selector.observe(lambda: cast("ViewPanel", view).request(rebind))
    if model is not None:
        model.observe(lambda _change: cast("ViewPanel", view).request(rebind), names="value")
    for control in controls.values():
        control.observe(lambda _change: changed(), names="value")
    activate_now()
    return view


def _physics_available(datasets: Mapping[str, pd.DataFrame]) -> bool:
    """Return whether every frame supplies compatible maintained physics evidence."""
    try:
        analysis.evaluation.dataframe.validate_comparison(datasets, require_physics=True, verify_grid=False)
    except analysis.evaluation.dataframe.ComparisonCompatibilityError:
        return False
    return True


def _permeability_available(datasets: Mapping[str, pd.DataFrame]) -> bool:
    """Return whether every frame declares the required permeability tensor inputs."""
    required = {"Kxx", "Kxy", "Kyy"}
    return all(required.issubset(frame.attrs.get("input_fields", ())) for frame in datasets.values())


def _pressure_velocity_available(datasets: Mapping[str, pd.DataFrame]) -> bool:
    """Return whether the synchronized two-model pressure/velocity view is valid."""
    if len(datasets) < _MINIMUM_COMPARISON_RUNS:
        return False
    roles = {analysis.evaluation.dataframe.dataset_role(frame) for frame in datasets.values()}
    if len(roles) != 1:
        return False
    try:
        for frame in datasets.values():
            analysis.evaluation.dataframe.single_output_group_field(frame, group_id="pressure")
            velocity = analysis.evaluation.dataframe.output_group_fields(frame, group_id="velocity")
            if len(velocity) < _MINIMUM_VELOCITY_COMPONENTS:
                return False
    except analysis.evaluation.dataframe.ComparisonCompatibilityError:
        return False
    return True


# Physics flag, COMSOL and surrogate seconds per case, training seconds before the run's own (or why they are missing), clocks, role.
type _BreakEvenEvidence = tuple[bool, Mapping[str, float], Mapping[str, float], float | str, str, str]


def _break_even_view(
    frames: Mapping[str, pd.DataFrame],
    evidence: Callable[[str, pd.DataFrame, Mapping[str, Any], Mapping[str, Any]], _BreakEvenEvidence],
    *,
    styles: SeriesStyles,
    labels: Mapping[str, str] | None = None,
    models: Sequence[str] = (),
) -> tuple[Figure, str] | widgets.HTML:
    """
    Draw the break-even against COMSOL of every run of ``frames`` that has its evidence, and name every run that lacks it.

    A run is its first frame, as in the work table, shown by its run label (``labels`` by frame, else the frame's key) in the
    colour ``styles`` gives that frame in the other views. ``evidence`` (given the key, frame, summary and work row) gives the
    physics flag, the per-case COMSOL and surrogate seconds (the clocks of the speed-up view), the training seconds spent before
    the run's own (``0.0``, or the text naming why they cannot be told), the text naming clocks and hardware and the frame's
    role. A run lacking model kind, its own training seconds, those before them, train cases or a case with both clocks is named
    with each missing piece below the figure, or in a message when no run has its evidence. ``models`` end the title. The
    figure comes with its runtime details, one plain line of clocks and hardware per drawn run, shown below it.
    """
    plots = analysis.evaluation.plots
    costs: list[Any] = []
    unavailable: list[str] = []
    shown = plots.series.SeriesStyles()
    shown.run_colours = styles.run_colours
    for key, summary in analysis.evaluation.run_discovery.distinct_runs(frames).items():
        label = (labels or {}).get(key, key)
        work = analysis.evaluation.run_discovery.stage_work_row(summary)
        physics, comsol, surrogate, prior_seconds, clocks, role = evidence(key, frames[key], summary, work)
        present = {
            "model kind": work["model_kind"] is not None,
            "training seconds": work["total_seconds"] is not None,
            "train cases": work["train_cases"] is not None,
            "case with both clocks": bool(comsol.keys() & surrogate.keys()),
        }
        missing = [f"no {name}" for name, available in present.items() if not available]
        if isinstance(prior_seconds, str):  # the seconds a continued stage needs from its A0, apart from its own
            missing.append(prior_seconds)
        if missing:
            unavailable.append(f"{label} ({'; '.join(missing)})")
            continue
        shown[label] = styles[key]
        entity = plots.series.SeriesEntity(label, work["model_kind"], physics, role, (), key)
        costs.append(
            plots.training.break_even_cost(
                entity,
                comsol_case_seconds=comsol,
                surrogate_case_seconds=surrogate,
                training_seconds=cast("float", work["total_seconds"]) + cast("float", prior_seconds),
                train_cases=work["train_cases"],
                clocks=clocks,
            )
        )
    if not costs:
        return widgets.HTML(f"<b>Break-even unavailable.</b> No run has its training and timing evidence: {escape('; '.join(unavailable))}.")
    figure = plots.training.plot_break_even(costs, styles=shown, unavailable=unavailable, models=models)
    return figure, "\n".join(f"{cost.entity.label}: {cost.clocks}" for cost in costs)


def _device_text(device: object, name: object = None) -> str:
    """Return a recorded compute device as its readable kind with its model name: ``cuda:0`` reads «GPU 0», ``cpu`` «CPU»."""
    kind, _separator, index = str(device).partition(":")
    shown = {"cuda": "GPU", "cpu": "CPU"}.get(kind, kind)
    shown = f"{shown} {index}" if index else shown
    return f"{shown} ({name})" if name else shown


def _steady_break_even(*, datasets: Mapping[str, pd.DataFrame], models: Sequence[str] = ()) -> tuple[Figure, str] | widgets.HTML:
    """
    Show the break-even of the steady runs with the clocks of the solver speed-up: COMSOL solver time against the batch-amortized forward.

    Each run takes its architecture colour (``series.series_styles`` by run), a single run too; its clocks read as names and
    readable devices, never as storage keys.
    """
    component = analysis.artifacts.timing.STEADY_SPEEDUPS["solver_speedup"][0]

    def evidence(_label: str, frame: pd.DataFrame, _summary: Mapping[str, Any], _work: Mapping[str, Any]) -> _BreakEvenEvidence:
        physics = analysis.evaluation.plots.series.model_metadata(frame)[2]
        report = analysis.evaluation.dataframe.require_complete_provenance(frame)["timing"]["report"]
        comsol = {case["case_id"]: case[component] for case in report["cases"] if case[component] is not None}
        surrogate = {case["case_id"]: case["batch_amortized_neural_operator_forward_s"] for case in report["cases"]}
        _surrogate, solver_hardware = analysis.evaluation.plots.run_summary.runtime_hardware(report)
        runtime = report["neural_runtime"]
        device = _device_text(runtime["resolved_device"], runtime.get("cuda_device_name") or runtime["processor"])
        clocks = f"COMSOL solver time on {solver_hardware}; surrogate batch-amortized forward on {device}"
        role = str(frame.attrs.get("dataset_role"))
        return physics, comsol, surrogate, 0.0, clocks, "id" if role == "eval" else role

    return _break_even_view(datasets, evidence, styles=analysis.evaluation.plots.series.series_styles(datasets, by_run=True), models=models)


def _build_sections(
    scope: _SteadyScope,
    *,
    comparison: bool,
    selection_state: EvaluationSelectionState | None = None,
) -> dict[str, tuple[list[tuple[str, Callable[..., Any], str]], str]]:
    """Bind the approved views directly to their view-specific controls."""
    plots = analysis.evaluation.plots
    controls = analysis.ui.controls
    dataset_map = dict(scope.run_frames)
    analysis.evaluation.dataframe.validate_comparison(dataset_map, verify_grid=False)
    frames = tuple(dataset_map.values())
    display_fields = analysis.evaluation.presentation.shared_display_fields(frames)
    channels = tuple(field.label for field in display_fields)
    parameters = analysis.evaluation.presentation.metadata_parameters(frames)
    parameter_options = tuple((analysis.evaluation.presentation.metadata_label(parameter), parameter) for parameter in parameters)
    toggle = analysis.ui.notebook.make_toggle_shortcut(dataset_map)
    # The run-level views take each run's first role frame under its run label, so a single model is never named by its role.
    runs = dict(next(iter(scope.frames_by_role.values())))
    run_toggle = analysis.ui.notebook.make_toggle_shortcut(runs)

    def dataset(
        title: str,
        plot_func: Callable[..., Any],
        *,
        view: str | None = None,
        single_model: bool = False,
        start: int | None = None,
        extra: Mapping[str, ViewControl] | None = None,
        **plot_kwargs: Any,
    ) -> tuple[str, Callable[..., Any], str]:
        """Return a plot of the selected datasets, as a case-count viewer from ``start`` with its own controls ``extra``.

        ``view`` names one of ``POOLABLE_STEADY_VIEWS``, whose views pool the materials of a role with equal weight.
        """
        if view is not None and view not in dataset_cuts.POOLABLE_STEADY_VIEWS:
            msg = f"Steady view {view!r} is not in POOLABLE_STEADY_VIEWS."
            raise ValueError(msg)
        return _entry(
            title,
            lambda export: _dataset_view(
                scope,
                plot_func=plot_func,
                selection_state=selection_state,
                export=export,
                pooled=view in dataset_cuts.POOLABLE_STEADY_VIEWS,
                single_model=single_model,
                plot_kwargs=plot_kwargs,
                count=None if start is None else (start, _CASE_COUNT_STEP),
                controls=extra,
            ),
        )

    def indexed(
        title: str, plot_func: Callable[..., Any], extra: Mapping[str, widgets.ValueWidget], **options: Any
    ) -> tuple[str, Callable[..., Any], str]:
        """Return a single-case viewer; its case, and a channel control, follow the shared selection."""
        return _entry(
            title,
            lambda export: _indexed_view(
                scope, plot_func=plot_func, controls=extra, selection_state=selection_state, export=export, channel_fields=display_fields, **options
            ),
        )

    configuration_entry = run_toggle(
        "Model-family configuration",
        plots.run_summary.plot_architecture_table,
        plot_name="Model-family configuration",
    )
    overview_entries = [dataset("Summary table", plots.run_summary.plot_run_summary_table)]
    if comparison:
        overview_entries.extend(
            (
                dataset(
                    "Global comparison summary",
                    plots.run_summary.plot_relative_comparison_scoreboard,
                    view="overview_global_comparison_summary",
                ),
                dataset("Pareto (Error vs Physics)", plots.run_summary.plot_accuracy_physics_pareto),
            )
        )
    else:
        overview_entries.append(configuration_entry)
    global_error_entries = [
        dataset("Global error metrics", plots.error_behavior.plot_global_error_metrics),
        dataset("Global error distribution", plots.error_behavior.plot_predictive_error_distributions, start=50),
        dataset("GT vs Prediction (mean)", plots.error_behavior.plot_mean_field_bias, start=50),
        dataset(
            "Mean error maps",
            plots.error_behavior.plot_mean_error_maps,
            start=100,
            extra={"error_mode": controls.error_mode_group("steady"), "display_spatial_stride": controls.display_stride_dropdown()},
        ),
        dataset(
            "Std error maps",
            plots.error_behavior.plot_std_error_maps,
            start=100,
            extra={"display_spatial_stride": controls.display_stride_dropdown()},
        ),
    ]

    architecture_entries = [
        configuration_entry,
        dataset(
            "Exact trainable parameters vs performance",
            plots.sensitivity_capacity.plot_capacity_accuracy,
            view="architecture_capacity_accuracy",
            start=50,
        ),
    ]

    error_decomposition_entries = [
        dataset("Error vs |GT| magnitude", plots.error_behavior.plot_error_vs_target_magnitude, start=50),
        dataset(
            "Boundary vs interior error",
            plots.error_behavior.plot_boundary_error_decomposition,
            start=100,
            extra={"channels": controls.ChannelChecks(channels)},
        ),
    ]

    physical_entries: list[tuple[str, Callable[..., Any], str]] = []
    if _physics_available(dataset_map):
        physical = plots.physical_consistency
        physical_entries = [
            dataset("Physical consistency summary table", physical.plot_physical_consistency_summary_table),
            dataset("Physical consistency CDFs", physical.plot_physical_consistency_cdf_grid, start=100),
            dataset("Cross-section flux balance", physical.plot_flux_balance, start=100),
            dataset("Darcy–Brinkman operator residual", physical.plot_brinkman_residual, start=100),  # noqa: RUF001
            dataset("Pressure and boundary-condition consistency", physical.plot_pressure_consistency, start=100),
            dataset("Pressure-boundary summary", physical.plot_pressure_boundary_summary, start=100),
            dataset("Spatial residuals", physical.plot_spatial_residuals, start=100),
        ]

    spectral_entries = [
        dataset("Demand vs prediction + error", plots.spectral_fidelity.plot_spectral_demand_prediction_error, start=50),
        dataset("Log spectral transfer with support", plots.spectral_fidelity.plot_spectral_transfer_ratio, start=50),
    ]

    sensitivity_entries: list[tuple[str, Callable[..., Any], str]] = []
    if parameters:
        sensitivity_entries = [
            dataset("Parameter-error correlation (heatmap)", plots.sensitivity_capacity.plot_metadata_error_heatmap, start=100),
            dataset(
                "Error vs input parameter (binned trend)",
                plots.sensitivity_capacity.plot_metadata_error_trends,
                start=100,
                extra={"channels": controls.ChannelChecks(channels)},
            ),
        ]

    def display_controls(*names: str) -> dict[str, widgets.ValueWidget]:
        """Return fresh display controls, each by the keyword its plot takes."""
        available: dict[str, Callable[[], widgets.ValueWidget]] = {
            "scale_mode": controls.scale_mode_group,
            "error_mode": lambda: controls.error_mode_group("steady"),
            "display_spatial_stride": controls.display_stride_dropdown,
            "kappa_scale": controls.kappa_scale_group,
        }
        return {name: available[name]() for name in names}

    sample_entries = [
        indexed(
            "Sample GT vs Prediction",
            plots.samples_outliers.plot_sample_prediction_overview,
            display_controls("error_mode", "display_spatial_stride"),
            single_model=True,
            sync=True,
        ),
    ]
    if _permeability_available(dataset_map):
        sample_entries.append(
            indexed(
                "Kappa tensor with error overlay",
                plots.samples_outliers.plot_permeability_error_overlay,
                {
                    "channel": controls.channel_dropdown(channels),
                    **display_controls("kappa_scale", "error_mode", "display_spatial_stride"),
                },
                single_model=True,
                sync=True,
            )
        )
    if comparison and _pressure_velocity_available(dataset_map):
        sample_entries.append(
            indexed(
                "Pressure & velocity field comparison",
                plots.samples_outliers.plot_pressure_velocity_comparison,
                display_controls("scale_mode", "display_spatial_stride"),
                single_model=False,
                sync=True,
                choose_models=True,
            )
        )

    outlier_entries = [
        dataset("Worst per-channel cases (tables)", plots.samples_outliers.plot_outlier_table, single_model=True, top_k=_OUTLIER_COUNT),
        indexed(
            "Worst per-channel cases (field plots)",
            plots.samples_outliers.plot_linked_outlier_cases,
            {"channel": controls.channel_dropdown(channels), **display_controls("error_mode", "display_spatial_stride")},
            single_model=True,
            offered=lambda rows: min(rows, _OUTLIER_COUNT) + 1,
            selection_name=lambda index, total: "reference" if index == total - 1 else f"worst_{index + 1}",
            index_to_kwargs=lambda index: {"selection_index": index},
        ),
    ]
    if parameters:
        outlier_entries.extend(
            (
                dataset("Extreme input parameters (table view)", plots.samples_outliers.plot_input_extremes_table),
                indexed(
                    "Extreme input parameter cases (field plots)",
                    plots.samples_outliers.plot_linked_input_extreme_cases,
                    {
                        "parameter": controls.dropdown(parameter_options, description="Parameter"),
                        **display_controls("error_mode", "display_spatial_stride"),
                    },
                    single_model=True,
                    offered=lambda _rows: 2,
                    selection_name=lambda index, _total: ("minimum", "maximum")[index],
                    index_to_kwargs=lambda index: {"selection_index": index},
                ),
            )
        )

    registry = {
        "overview": (overview_entries, EVALUATION_SECTION_TITLES["overview"]),
        "global_error": (global_error_entries, EVALUATION_SECTION_TITLES["global_error"]),
        "architecture": (architecture_entries if comparison else [], EVALUATION_SECTION_TITLES["architecture"]),
        "error_decomposition": (error_decomposition_entries, EVALUATION_SECTION_TITLES["error_decomposition"]),
        "physical_consistency": (physical_entries, EVALUATION_SECTION_TITLES["physical_consistency"]),
        "spectral_analysis": (spectral_entries, EVALUATION_SECTION_TITLES["spectral_analysis"]),
        "parameter_sensitivity": (sensitivity_entries, EVALUATION_SECTION_TITLES["parameter_sensitivity"]),
        "sample_viewer": (sample_entries, EVALUATION_SECTION_TITLES["sample_viewer"]),
        "outlier_analysis": (outlier_entries, EVALUATION_SECTION_TITLES["outlier_analysis"]),
        "runtime": (
            [
                toggle("Runtime and speed-up", plots.run_summary.plot_runtime_table, plot_name="Runtime and speed-up"),
                run_toggle("Work per stage", plots.run_summary.plot_work_table, plot_name="Work per stage"),
                run_toggle(
                    "Break-even", _steady_break_even, plot_name="Break-even", models=tuple("" if run in scope.unnamed else run for run in runs)
                ),
            ],
            EVALUATION_SECTION_TITLES["runtime"],
        ),
    }
    return {key: value for key, value in registry.items() if value[0]}


def _selected_section_keys(
    sections: Sequence[str] | str,
    *,
    registry: Mapping[str, object],
    order: Sequence[str],
) -> tuple[str, ...]:
    """Validate section selection against the approved order of one task and mode."""
    if sections == "all":
        return tuple(key for key in order if key in registry)
    if isinstance(sections, str) or not isinstance(sections, Sequence) or not sections:
        msg = "sections must be 'all' or a non-empty sequence of section keys."
        raise TypeError(msg)
    selected = tuple(sections)
    if not all(isinstance(key, str) for key in selected):
        msg = "sections must contain only string keys."
        raise TypeError(msg)
    if len(selected) != len(set(selected)):
        msg = "sections must not contain duplicate keys."
        raise ValueError(msg)
    unknown = sorted(set(selected).difference(order))
    if unknown:
        msg = f"Unknown evaluation sections: {unknown}."
        raise ValueError(msg)
    unavailable = [key for key in selected if key not in registry]
    if unavailable:
        msg = f"Evaluation sections are unavailable for these artifacts: {unavailable}."
        raise ValueError(msg)
    return selected


_RENDERED_VIEWS: WeakKeyDictionary[widgets.Widget, list[Any]] = WeakKeyDictionary()
_LAZY_PANELS: WeakKeyDictionary[widgets.Widget, LazyPanel] = WeakKeyDictionary()


def _record_views(factory: Callable[..., Any], views: list[Any]) -> Callable[..., Any]:
    """Wrap one lazy entry so that its panel knows every view panel and widget the entry renders."""

    @wraps(factory)
    def build(**context: Any) -> Any:
        result = factory(**context)
        views.extend(
            item for item in (result if isinstance(result, tuple) else (result,)) if isinstance(item, (widgets.Widget, analysis.ui.panel.ViewPanel))
        )
        return result

    return build


def close_panel(panel: widgets.Widget) -> None:
    """Close a replaced panel, its nested widgets and every view it rendered.

    Rendered views that are no longer shown are in no ``children``, and keep their loaded records and compute-preference
    callbacks alive until they are closed. Each is deactivated, not only the current view of its section, because a view
    that a re-entrant selection displaced is no longer any section's current view; deactivation stops a later preference
    change from reaching a view whose session is already closed.
    """
    lazy = _LAZY_PANELS.pop(panel, None)
    rendered = _RENDERED_VIEWS.pop(panel, ())
    if lazy is not None:
        for section in lazy.sections:
            section.detach()
    nested = (*getattr(panel, "children", ()), *(() if lazy is None else (lazy.tabs,)))
    for item in (*nested, *rendered):
        if isinstance(item, analysis.ui.panel.ViewPanel):
            item.deactivate()
        widget = item.widget if isinstance(item, analysis.ui.panel.ViewPanel) else item
        if isinstance(widget, widgets.Widget):
            close_panel(widget)
    panel.close()


def _assemble_panel(
    registry: Mapping[str, tuple[list[tuple[str, Callable[..., Any], str]], str]],
    keys: Sequence[str],
    *,
    export_state: dict[str, Any],
    comparison: bool,
    widths: WidthScope | None = None,
) -> widgets.Widget:
    """Build the lazy panel and number its tabs and views by displayed position.

    The tab title is ``"{n}. {title}"`` and a view title ``"{n}-{k}. {title}"``; both its dropdown label and its export
    plot name carry the displayed number, so capability filtering and section selection never leave a gap.
    """
    views: list[Any] = []
    sections: list[ViewSection] = []
    titles: list[str] = []
    for tab, key in enumerate(keys, start=1):
        entries, title = registry[key]
        numbered: list[tuple[str, Callable[..., Any], str]] = []
        for position, (entry_title, factory, _entry_key) in enumerate(entries, start=1):
            name = f"{tab}-{position}. {entry_title}"
            numbered.append((name, _record_views(factory, views), name))
        sections.append(analysis.ui.notebook.view_section(numbered, export_state=export_state))
        titles.append(f"{tab}. {title}")
    lazy = analysis.ui.notebook.LazyPanel(
        sections,
        titles,
        open_label="Model comparison - Open Evaluation" if comparison else "Single model - Open Evaluation",
        export_state=export_state,
        widths=widths,
    )
    _RENDERED_VIEWS[lazy.widget] = views
    _LAZY_PANELS[lazy.widget] = lazy
    return lazy.widget


def _build_panel(
    *,
    contexts: Sequence[EvaluationContext],
    comparison: bool,
    sections: Sequence[str] | str,
    selection_state: EvaluationSelectionState | None = None,
    widths: WidthScope | None = None,
    refused_roles: Mapping[str, str] | None = None,
    unnamed_models: Collection[str] = (),
) -> widgets.Widget:
    """Build one lazy panel whose views offer the datasets of every context in their own selectors; ``unnamed_models`` titles name none."""
    registry = _build_sections(
        _SteadyScope(contexts, comparison=comparison, refused_roles=refused_roles or {}, unnamed=unnamed_models),
        comparison=comparison,
        selection_state=selection_state,
    )
    section_keys = _selected_section_keys(
        sections,
        registry=registry,
        order=STEADY_COMPARISON_SECTION_KEYS if comparison else STEADY_SINGLE_SECTION_KEYS,
    )
    if not section_keys:
        msg = "The selected artifacts provide no approved evaluation sections."
        raise ValueError(msg)
    export_state = {
        "fig": None,
        "plot_name": None,
        "title": None,
        "filename_stem": None,
        "filename_prefix": "steady_flow",
    }
    return _assemble_panel(registry, section_keys, export_state=export_state, comparison=comparison, widths=widths)


def _normalize_contexts(
    session: EvaluationSession,
    contexts: Sequence[EvaluationContext],
    *,
    comparison: bool,
) -> tuple[EvaluationContext, ...]:
    """Validate ordered contexts and exact live-session frame ownership."""
    if session.closed:
        msg = "Cannot build an evaluation panel from a closed session."
        raise analysis.evaluation.session.EvaluationSessionClosedError(msg)
    if isinstance(contexts, (str, bytes)) or not isinstance(contexts, Sequence) or not contexts:
        msg = "contexts must be a non-empty ordered sequence of EvaluationContext values."
        raise TypeError(msg)
    canonical_frames = tuple(session.canonical_frames.values())
    normalized = []
    for position, context in enumerate(contexts):
        if not isinstance(context, EvaluationContext):
            msg = f"contexts[{position}] must be an EvaluationContext."
            raise TypeError(msg)
        key = context.key.strip() if isinstance(context.key, str) else ""
        label = context.label.strip() if isinstance(context.label, str) else ""
        if not key or not label:
            msg = "Every evaluation context requires non-blank key and label text."
            raise ValueError(msg)
        datasets = dict(context.datasets)
        if comparison and len(datasets) < _MINIMUM_COMPARISON_RUNS:
            msg = f"Evaluation context {key!r} must contain at least two models."
            raise ValueError(msg)
        if not comparison and len(datasets) != 1:
            msg = f"Evaluation context {key!r} must contain exactly one model."
            raise ValueError(msg)
        if any(not isinstance(name, str) or not name.strip() for name in datasets):
            msg = f"Evaluation context {key!r} contains a blank or non-string model label."
            raise TypeError(msg)
        for model_label, frame in datasets.items():
            if not any(frame is canonical for canonical in canonical_frames):
                msg = f"Evaluation context {key!r} frame {model_label!r} is not an exact canonical frame owned by the supplied session."
                raise ValueError(msg)
        analysis.evaluation.dataframe.validate_comparison(datasets, verify_grid=False)
        normalized.append(EvaluationContext(key=key, label=label, datasets=datasets))
    keys = tuple(context.key for context in normalized)
    labels = tuple(context.label for context in normalized)
    if len(keys) != len(set(keys)):
        msg = "Evaluation context keys must be unique."
        raise ValueError(msg)
    if len(labels) != len(set(labels)):
        msg = "Evaluation context labels must be unique."
        raise ValueError(msg)
    return tuple(normalized)


def build_single_model_panel(
    *,
    session: EvaluationSession,
    contexts: Sequence[EvaluationContext],
    sections: Sequence[str] | str = "all",
    selection_state: EvaluationSelectionState | None = None,
    widths: WidthScope | None = None,
    unnamed_models: Collection[str] = (),
) -> widgets.Widget:
    """Build one single-model panel containing every supplied dataset; a model label in ``unnamed_models`` (an alias) no title names."""
    normalized = _normalize_contexts(session, contexts, comparison=False)
    model_labels = tuple(next(iter(context.datasets)) for context in normalized)
    if len(set(model_labels)) != 1:
        msg = "Single-model contexts must all refer to the same presentation model label."
        raise ValueError(msg)
    return _build_panel(
        contexts=normalized, comparison=False, sections=sections, selection_state=selection_state, widths=widths, unnamed_models=unnamed_models
    )


def build_comparison_panel(
    *,
    session: EvaluationSession,
    contexts: Sequence[EvaluationContext],
    sections: Sequence[str] | str = "all",
    selection_state: EvaluationSelectionState | None = None,
    widths: WidthScope | None = None,
    refused_roles: Mapping[str, str] | None = None,
    unnamed_models: Collection[str] = (),
) -> widgets.Widget:
    """
    Build one comparison panel; its views choose among the datasets of every role, the refused roles listed as unavailable.

    A model label in ``unnamed_models`` is no explicit short label (a storage alias): the legend shows it, no title names a model then.
    """
    normalized = _normalize_contexts(session, contexts, comparison=True)
    return _build_panel(
        contexts=normalized,
        comparison=True,
        sections=sections,
        selection_state=selection_state,
        widths=widths,
        refused_roles=refused_roles,
        unnamed_models=unnamed_models,
    )


def _shared_transient_physical_times(records: Sequence[Any]) -> tuple[float, ...]:
    """Return exact physical times supported by at least one selected case."""
    admitted = tuple(records)
    if not admitted:
        msg = "Aggregate transient spatial views require complete-rollout records."
        raise ValueError(msg)
    support: set[float] = set()
    for record in admitted:
        values = tuple(float(value) for value in record.physical_times)
        if not values or not all(isfinite(value) for value in values) or any(right <= left for left, right in pairwise(values)):
            msg = "Aggregate transient physical times must be finite and strictly increasing."
            raise ValueError(msg)
        support.update(values)
    times = tuple(sorted(support))
    if not times:
        msg = "Aggregate transient cases contain no exact stored physical time."
        raise ValueError(msg)
    return times


def _transient_channel_fields(
    session: TransientEvaluationSession,
) -> tuple[Any, ...]:
    """Return comparable fields from exact session-owned frame provenance."""
    return analysis.evaluation.presentation.transient_channel_resolution(tuple(session.canonical_frames.values())).fields


def _supported_transient_fields(fields: Sequence[Any]) -> tuple[Any, ...]:
    """Retain channels representable by the current admitted sequence arrays."""
    supported = {
        *analysis.evaluation.transient_artifact.STATE_ORDER,
        "w_gr",
    }
    return tuple(field for field in fields if field.key in supported)


def _stored_transient_fields(fields: Sequence[Any]) -> tuple[Any, ...]:
    """Retain stored channels with persisted normalized per-channel metrics."""
    return tuple(field for field in fields if field.stored and field.key in analysis.evaluation.transient_artifact.STATE_ORDER)


def _full_autonomous_summary(
    session: TransientEvaluationSession,
    *,
    frame_name: str,
    case_id: str,
) -> Any:
    """Return the sole complete-rollout summary without opening its NPZ payload."""
    matches = tuple(
        summary
        for summary in session.record_summaries_for_case(frame_name, case_id)
        if summary.mode == "autonomous_full" and summary.requested_horizon == "full"
    )
    if len(matches) != 1:
        msg = "Transient field views require exactly one complete autonomous rollout per case."
        raise ValueError(msg)
    return matches[0]


def _full_autonomous_record(
    session: TransientEvaluationSession,
    *,
    frame_name: str,
    case_id: str,
) -> Any:
    """Load one exact complete-rollout record through the bounded session owner."""
    summary = _full_autonomous_summary(
        session,
        frame_name=frame_name,
        case_id=case_id,
    )
    return session.record_for_coordinates(
        frame_name,
        case_id,
        mode="autonomous_full",
        origin_index=int(summary.origin_index),
        requested_horizon="full",
    )


def _transient_inventory(
    session: TransientEvaluationSession,
    *,
    comparison: bool,
    selection_state: EvaluationSelectionState | None,
) -> tuple[Any, ...]:
    """Return disjoint single-model partitions or comparison-owned case entries."""
    inventory = session.case_inventory() if comparison else session.partitioned_case_inventory()
    if comparison and selection_state is not None and selection_state.capabilities is not None:
        allowed = set(selection_state.capabilities.case_ids)
        inventory = tuple(entry for entry in inventory if entry.case_id in allowed)
    if not inventory:
        msg = "Transient field views require at least one exact case."
        raise ValueError(msg)
    return inventory


def _transient_case_label(entry: Any, *, comparison: bool) -> str:
    """Return the compact EDA material/case label while ownership stays internal."""
    material = analysis.presentation.display_labels.material_display_label(
        entry.material_family,
    )
    parts = (material, entry.frame_name if comparison else None, entry.case_label)
    return " · ".join(part for part in parts if part is not None)


def _focus_entries(session: TransientEvaluationSession, entries: Sequence[Any], choices: Sequence[DatasetChoice]) -> tuple[Any, ...]:
    """Return the inventory entries (cases) that belong to the chosen datasets, in inventory order."""
    contexts = dataset_cuts.transient_contexts(session, choices)
    return tuple(
        entry for entry in entries if any(entry.frame_name in context.frame_names and entry.case_id in context.case_ids for context in contexts)
    )


def _case_labels(entries: Sequence[Any], *, comparison: bool) -> dict[int, str]:
    """Return the label of every one-based case position."""
    return {position: _transient_case_label(entry, comparison=comparison) for position, entry in enumerate(entries, start=1)}


class _Channels:
    """Direct channel checkboxes of a transient view that start from, and follow, the channels the other views chose."""

    def __init__(self, title: str, fields: Sequence[Any], selection_state: EvaluationSelectionState | None) -> None:
        """Create the checkboxes, checked as the shared selection has them, else T, phi and w_gr where offered, else all."""
        self._keys = tuple(field.key for field in fields)
        self._labels = {field.key: analysis.presentation.moisture_display.channel_label(field.key, field.label, field.unit) for field in fields}
        self._state = selection_state
        self.checks = analysis.ui.controls.ChannelChecks(self._keys, labels=self._labels, title=title)
        default = tuple(key for key in _DEFAULT_TRANSIENT_CHANNELS if key in self._keys)
        if default:
            self.checks.rebind(self._keys, labels=self._labels, retained=default)
        self.pull()

    def pull(self) -> bool:
        """Check the shared channels this view offers without notifying; return whether the checkboxes changed."""
        shared = tuple(key for key in (() if self._state is None else self._state.selection.channels) if key in self._keys)
        if not shared or set(shared) == set(self.checks.selected):
            return False
        self.checks.rebind(self._keys, labels=self._labels, retained=shared)
        return True

    def companion(self, title: str, checked: Sequence[str]) -> analysis.ui.controls.ChannelChecks:
        """Return a second row of checkboxes over the same channels and labels with only ``checked`` ticked, not tied to the shared selection."""
        checks = analysis.ui.controls.ChannelChecks(self._keys, labels=self._labels, title=title)
        checks.rebind(self._keys, labels=self._labels, retained=checked)
        return checks


class _PooledClock:
    """The clock of a pooled time view: hours while every drawn curve holds one case, else the drying phase, until the user picks one."""

    def __init__(self, on_pick: Callable[[], None]) -> None:
        """Create the choice on the drying phase; ``on_pick`` redraws after each clock the user picks."""
        self.widget = analysis.ui.controls.clock_group("phase")
        self._default, self._picked, self._on_pick = "phase", False, on_pick
        self.widget.observe(self._changed, names="value")

    def resolve(self, series: Mapping[str, Sequence[Any]]) -> str:
        """Return the clock to draw ``series`` with: the user's pick, else the default its curves call for, which the choice then shows."""
        if not self._picked:
            self._default = "hours" if all(len(records) <= 1 for records in series.values()) else "phase"
            self.widget.value = self._default  # taken for no pick, so nothing redraws
        return str(self.widget.value)

    def _changed(self, change: Mapping[str, Any]) -> None:
        if self._picked or change["new"] != self._default:
            self._picked = True
            self._on_pick()


class _CaseFocus:
    """The datasets a transient case view shows, the cases they hold, and a case navigator over those cases' one-based positions."""

    def __init__(
        self,
        session: TransientEvaluationSession,
        entries: Sequence[Any],
        *,
        inventory: DatasetInventory,
        multiple: bool,
        comparison: bool,
        selection_state: EvaluationSelectionState | None,
        status: analysis.ui.controls.StatusLine,
        on_change: Callable[[], None],
        order: Callable[[Sequence[Any]], tuple[Any, ...]] | None = None,
    ) -> None:
        """
        Create the dataset choice and the navigator on the shared case, else the first one; ``on_change`` follows the navigator.

        ``order`` ranks the cases of the chosen datasets again at every change of the choice; without it they keep inventory order.
        A ranked view opens on rank one and never jumps to the case another view chose.
        """
        self._session = session
        self._entries = entries
        self._comparison = comparison
        self._state = selection_state
        self._order = order
        self.choice = _SharedChoice(inventory, multiple=multiple, pool_reason=None, selection_state=selection_state)
        self.shown = self._focused()
        shared = None if selection_state is None or order is not None else selection_state.selection.case_id
        self.navigator = analysis.ui.navigation.CaseNavigator(
            self._positions(),
            value=self._position_of(shared) or 1,
            labels=_case_labels(self.shown, comparison=comparison),
            status=status,
            on_change=lambda _position: on_change(),
        )

    @property
    def entry(self) -> Any:
        """Return the inventory entry of the shown case."""
        return self.shown[self.navigator.value - 1]

    def _positions(self) -> tuple[int, ...]:
        return tuple(range(1, max(len(self.shown), 1) + 1))

    def _position_of(self, case_id: str | None) -> int | None:
        return next((position for position, entry in enumerate(self.shown, start=1) if entry.case_id == case_id), None)

    def _focused(self) -> tuple[Any, ...]:
        """Return the cases of the chosen datasets, in the view's order when it ranks them."""
        shown = _focus_entries(self._session, self._entries, self.choice.selector.choices())
        return shown if self._order is None else self._order(shown)

    def rebind(self) -> bool:
        """List the cases of the chosen datasets again, keeping the shown position (ranked: rank one); return whether there are any."""
        self.shown = self._focused()
        if self.shown:
            labels = _case_labels(self.shown, comparison=self._comparison)
            self.navigator.rebind(self._positions(), preserve=self._order is None, notify=False, labels=labels)
        return bool(self.shown)

    def pull(self) -> None:
        """Take the datasets and (unranked) the case the other views chose; only the navigator's own change callback reports a new case."""
        if self.choice.pull():
            self.rebind()
        wanted = self._position_of(None if self._state is None or self._order is not None else self._state.selection.case_id)
        if wanted is not None:
            self.navigator.field.value = wanted


def _transient_outlier_order(
    session: TransientEvaluationSession, *, fields: Sequence[Any], ranks: dict[tuple[str, str], tuple[str, int]] | None = None
) -> Callable[[Sequence[Any]], tuple[Any, ...]]:
    """
    Return the ranking of given inventory entries: the stable union of their per-channel worst complete-rollout cases, else the entries.

    The view ranks only the cases of the chosen datasets; a ranking over every dataset of the session can leave the chosen one empty.
    Only a finite error ranks, so the first field with one is the ranking field of rank one; without any, the entries stay in
    inventory order, unranked. ``ranks`` receives, at every ranking, each ranked case's (frame, case) with the field key and rank
    under which it first entered.
    """
    frame = session.case_dataframe(modes=("autonomous_full",))
    selected = frame.loc[(frame["requested_horizon"] == "full") & (frame["scope"] == "cumulative")]

    def order(entries: Sequence[Any]) -> tuple[Any, ...]:
        by_key = {(entry.frame_name, entry.case_id): entry for entry in entries}
        within = selected.loc[[key in by_key for key in zip(selected["frame"].astype(str), selected["case_id"].astype(str), strict=True)]]
        ordered: dict[tuple[str, str], Any] = {}
        found: dict[tuple[str, str], tuple[str, int]] = {}
        for field in fields:
            column = f"normalized_rmse_{field.key}" if field.stored else "physical_w_gr_rmse"
            if column not in within:
                continue
            finite = within.loc[np.isfinite(within[column].to_numpy(dtype=np.float64))]  # ``nlargest`` would rank NaN when short of values
            for rank, row in enumerate(finite.nlargest(_OUTLIER_COUNT, column).itertuples(index=False), start=1):
                key = (str(row.frame), str(row.case_id))
                ordered.setdefault(key, by_key[key])
                found.setdefault(key, (str(field.key), rank))
        if ranks is not None:
            ranks.clear()
            ranks.update(found)
        return tuple(ordered.values()) or tuple(entries)

    return order


def _frame_reference_scales(
    session: TransientEvaluationSession, frame_name: str, fields: Sequence[str]
) -> dict[str, error_diagnostics.ReferenceScale | str]:
    """Read one frame's persisted train-only state scales for only the selected stored fields."""
    scaling_state = session.scaling_state(frame_name)
    result: dict[str, error_diagnostics.ReferenceScale | str] = {}
    for field in fields:
        index = (
            transient_artifact.STATE_ORDER.index(field) if field in transient_artifact.STATE_ORDER else transient_artifact.STATE_ORDER.index("w_surf")
        )
        result[field] = error_diagnostics.transient_reference_scale(
            scaling_state,
            field=field,
            unit=transient_artifact.STATE_UNITS[index],
            dataset_identity=str(scaling_state.get("train_membership_digest", "")),
        )
    return result


def _record_scales(scales: Mapping[str, error_diagnostics.ReferenceScale | str], record: Any) -> dict[str, error_diagnostics.ReferenceScale | str]:
    """Return the frame's scales that carry the scaler identity the record names; the others become the reason."""
    identity = record.identity if isinstance(record.identity, Mapping) else {}
    scaling_identity = identity.get("scaling_identity")
    expected = str(scaling_identity.get("semantic_digest", "")) if isinstance(scaling_identity, Mapping) else ""
    reason = "Saved transient state scale contradicts the evaluated artifact's scaler identity."
    return {field: scale if isinstance(scale, str) or scale.scaler_identity == expected else reason for field, scale in scales.items()}


def _transient_reference_scales(
    session: TransientEvaluationSession, frame_name: str, fields: Sequence[str], record: Any
) -> dict[str, error_diagnostics.ReferenceScale | str]:
    """Read the persisted train-only state scales of the selected stored fields, checked against the record's scaler identity."""
    return _record_scales(_frame_reference_scales(session, frame_name, fields), record)


def _series_reference_scales(
    session: TransientEvaluationSession,
    owners: Mapping[int, str],
    series: Mapping[str, Sequence[Any]],
    fields: Sequence[str],
    error_mode: widgets.ValueWidget | None,
) -> dict[str, Any]:
    """
    Return the mean maps' ``error_mode`` and, in the normalized mode, ``reference_scales`` per series and field (none without the control).

    A series' scale is the train-only one every record carries, each read from the record's own frame; records that disagree or
    lack a scale give that field's reason (a string) instead of the first record's scale. A water field refuses the mode in the builder.
    """
    if error_mode is None:
        return {}
    if error_mode.value != error_diagnostics.NORMALIZED_ABSOLUTE_ERROR:
        return {"error_mode": str(error_mode.value)}
    result: dict[str, dict[str, error_diagnostics.ReferenceScale | str]] = {}
    frame_scales = {
        frame: _frame_reference_scales(session, frame, fields) for frame in {owners[id(record)] for records in series.values() for record in records}
    }
    for label, records in series.items():
        if not records:
            continue
        held = [_record_scales(frame_scales[owners[id(record)]], record) for record in records]
        result[label] = {}
        for field in fields:
            scales = [scales_of[field] for scales_of in held]
            reason = next((scale for scale in scales if isinstance(scale, str)), f"The records of one series carry different scales of {field}.")
            result[label][field] = scales[0] if all(scale == scales[0] for scale in scales) else reason
    return {"error_mode": str(error_mode.value), "reference_scales": result}


def _gif_peak_bytes(frame_count: int, frame_pixels: int) -> int:
    """Return an upper bound of one GIF export's memory: stored and working frames plus the bounded palette step."""
    return (frame_count * _GIF_BYTES_PER_PIXEL + _GIF_WORKING_BYTES_PER_PIXEL) * frame_pixels + _GIF_PALETTE_STEP_BYTES


def _rgb_keys(frame: Image.Image) -> np.ndarray:
    """Return the 24-bit RGB key (red + 256 green + 65536 blue) of every pixel of one frame in row-major order."""
    rgb = frame if frame.mode == "RGB" else frame.convert("RGB")
    return np.frombuffer(rgb.tobytes("raw", "RGBX"), dtype="<u4") & 0xFFFFFF


def _key_colours(keys: np.ndarray) -> np.ndarray:
    """Return the RGB rows of 24-bit colour keys as floats."""
    return np.stack([(keys >> shift) & 0xFF for shift in (0, 8, 16)], axis=1).astype(np.float64)


def _squared_distances(colours: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """Return the squared RGB distance of every colour to one centre."""
    return ((colours - centre) ** 2).sum(axis=1)


def _nearest_entries(colours: np.ndarray, entries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the nearest entry and its squared distance for every colour, in blocks of bounded size (exact for integers)."""
    labels = np.empty(len(colours), dtype=np.intp)
    distances = np.empty(len(colours))
    for start in range(0, len(colours), _GIF_DISTANCE_BLOCK):
        block = colours[start : start + _GIF_DISTANCE_BLOCK]
        squared = np.zeros((len(block), len(entries)))
        for channel in range(3):
            squared += np.subtract.outer(block[:, channel], entries[:, channel]) ** 2
        labels[start : start + len(block)] = squared.argmin(axis=1)
        distances[start : start + len(block)] = squared.min(axis=1)
    return labels, distances


def _count_colours(counts: np.ndarray, keys: np.ndarray) -> None:
    """Add the exact pixel count of every 24-bit colour key to ``counts``; chunks bound the sorting memory whatever the colours."""
    for start in range(0, keys.size, _GIF_COUNT_CHUNK):
        colours, chunk_counts = np.unique(keys[start : start + _GIF_COUNT_CHUNK], return_counts=True)
        counts[colours] += chunk_counts.astype(np.uint32)


def _gif_candidates(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return candidate colours and their pixel counts: the distinct colours, or above the cap coarser RGB bins.

    A bin (4 or, if still too many, 8 values per channel) carries the summed counts and the count-weighted mean
    colour of its members, so runtime and memory stay bounded whatever the frames contain.
    """
    if np.count_nonzero(counts) <= _GIF_CANDIDATE_CAP:
        keys = np.flatnonzero(counts)
        return _key_colours(keys), counts[keys].astype(np.float64)
    shift = 2 if np.count_nonzero(counts.reshape(64, 4, 64, 4, 64, 4).any(axis=(1, 3, 5))) <= _GIF_CANDIDATE_CAP else 3
    side = 256 >> shift
    green, red = np.divmod(np.arange(1 << 16), 256)
    plane_bins = (green >> shift) * side + (red >> shift)
    sums = np.zeros((side**3, 4))  # pixels and red, green, blue sums per bin
    for blue, plane in enumerate(counts.reshape(256, 1 << 16)):
        pixels = plane.astype(np.float64)
        start = (blue >> shift) * side * side
        for column, weights in enumerate((pixels, pixels * red, pixels * green, pixels * blue)):
            sums[start : start + side * side, column] += np.bincount(plane_bins, weights=weights, minlength=side * side)
    occupied = sums[:, 0] > 0
    return sums[occupied, 1:] / sums[occupied, :1], sums[occupied, 0]


def _gif_palette_entries(colours: np.ndarray, weights: np.ndarray, fixed: np.ndarray, size: int) -> np.ndarray:
    """
    Return the ``fixed`` entries followed by at most ``size - len(fixed)`` entries for weighted candidate colours.

    Without fixed entries the heaviest candidate (the background) becomes one, exact. When the free entries hold every
    candidate no fixed entry matches, all are kept exactly. Otherwise weighted Lloyd k-means places all free entries
    but the rescue share where the weight is, while the fixed entries stay; it starts from a deterministic
    farthest-point traversal that weights each squared distance with the candidate's weight and stops after a fixed
    number of iterations. Each rescue entry then goes to the candidate farthest from all entries, so a rare colour
    never maps to another hue.
    """
    if len(colours) == 0:
        return fixed
    if len(fixed) == 0:
        fixed = colours[[int(np.argmax(weights))]].round()
    free = size - len(fixed)
    distance = _nearest_entries(colours, fixed)[1]
    if np.count_nonzero(distance) <= free:
        return np.vstack((fixed, colours[distance > 0].round()))
    rescue = free // _GIF_RESCUE_SHARE
    chosen: list[int] = []
    while len(chosen) < free - rescue:
        chosen.append(int(np.argmax(weights * distance)))
        distance = np.minimum(distance, _squared_distances(colours, colours[chosen[-1]]))
    centres = np.vstack((fixed, colours[chosen]))
    movable = np.arange(len(centres)) >= len(fixed)
    for _iteration in range(_GIF_KMEANS_ITERATIONS):
        labels = _nearest_entries(colours, centres)[0]
        mass = np.bincount(labels, weights=weights, minlength=len(centres))
        moved = movable & (mass > 0)
        updated = centres.copy()
        for channel in range(3):
            sums = np.bincount(labels, weights=weights * colours[:, channel], minlength=len(centres))
            updated[moved, channel] = sums[moved] / mass[moved]
        if np.array_equal(updated, centres):
            break
        centres = updated
    entries = centres.round()
    residual = _nearest_entries(colours, entries)[1]
    for _rescue in range(rescue):
        farthest = int(np.argmax(residual))
        if residual[farthest] == 0.0:
            break
        entries = np.vstack((entries, colours[farthest].round()))
        residual = np.minimum(residual, _squared_distances(colours, entries[-1]))
    return entries


def _colour_candidates(keys: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the candidate colours of the colour keys and their pixel counts; ``counts`` is a reused scratch table."""
    counts.fill(0)
    _count_colours(counts, keys)
    return _gif_candidates(counts)


def _nearest_indices(keys: np.ndarray, entries: np.ndarray, counts: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Return each colour key's nearest-entry index; ``counts`` and ``table`` are reused scratch tables over all colours."""
    counts.fill(0)
    _count_colours(counts, keys)
    for blue, plane in enumerate(counts.reshape(256, 1 << 16)):  # one blue plane at a time bounds the memory
        present = np.flatnonzero(plane) + (blue << 16)
        if present.size:
            table[present] = _nearest_entries(_key_colours(present), entries)[0]
    return table[keys]


def _index_gif_frames(frames: list[Image.Image]) -> None:
    """
    Replace every frame in place by palette indices for an animation of delta frames, without dithering.

    Frame 0 is complete. Its first entries are fitted to the pixels equal in every frame (colour bars, text,
    background), weighted with the square root of their pixel counts so that thin bars get entries beside the
    large background; the remaining entries are fitted, pixel-weighted, to its other pixels (the maps at t = 0),
    and each pixel takes its nearest entry. Each later frame encodes only the pixels whose RGB differs from the
    previous frame, with up to 255 pixel-weighted entries fitted to those pixels; every other pixel takes the
    transparent index 255. Written with disposal 1 ("do not dispose"), an unchanged pixel keeps its earlier decode:
    colour bars, text and background decode alike in every frame, and changing maps get a nearly full palette in
    every later frame. Frame 0 redraws every pixel when the animation loops.
    """
    counts = np.zeros(_GIF_COLOUR_KEYS, dtype=np.uint32)
    table = np.zeros(_GIF_COLOUR_KEYS, dtype=np.uint8)
    previous = _rgb_keys(frames[0])
    static = np.ones(previous.size, dtype=bool)
    for frame in frames[1:]:
        static &= _rgb_keys(frame) == previous
    colours, pixels = _colour_candidates(previous[static], counts)
    shared = _GIF_PALETTE_SIZE if static.all() else _GIF_STATIC_ENTRIES
    entries = _gif_palette_entries(colours, np.sqrt(pixels), np.empty((0, 3)), shared)
    colours, pixels = _colour_candidates(previous[~static], counts)
    entries = _gif_palette_entries(colours, pixels, entries, _GIF_PALETTE_SIZE)
    del static, colours, pixels
    first = Image.fromarray(_nearest_indices(previous, entries, counts, table).reshape(frames[0].height, frames[0].width))
    first.putpalette(entries.astype(np.uint8).tobytes())
    frames[0] = first  # the original is released once indexed
    for index in range(1, len(frames)):
        frame = frames[index]
        keys = _rgb_keys(frame)
        changed = keys != previous
        selected = keys[changed]
        colours, pixels = _colour_candidates(selected, counts)
        entries = _gif_palette_entries(colours, pixels, np.empty((0, 3)), _GIF_TRANSPARENT_INDEX)
        indices = np.full(keys.size, _GIF_TRANSPARENT_INDEX, dtype=np.uint8)
        indices[changed] = _nearest_indices(selected, entries, counts, table)
        del changed, selected
        indexed = Image.fromarray(indices.reshape(frame.height, frame.width))
        indexed.putpalette(np.vstack((entries, np.zeros((_GIF_PALETTE_SIZE - len(entries), 3)))).astype(np.uint8).tobytes())
        indexed.info["transparency"] = _GIF_TRANSPARENT_INDEX
        frames[index] = indexed
        previous = keys


class _HandleGifWriter(PillowWriter):
    """Collect frames like PillowWriter, then write a GIF of delta frames through one already open handle."""

    _frames: list[Image.Image]

    def __init__(self, handle: IO[bytes], fps: int) -> None:
        super().__init__(fps=fps)
        self._handle = handle

    def finish(self) -> None:
        """Write the collected frames as one looping GIF without reopening the reserved path."""
        # One palette over all frames leaves colours shown in a few frames too few entries, and a palette shared by
        # unchanged and changing pixels steps the colour bars; delta frames give each frame's changes their own palette.
        _index_gif_frames(self._frames)
        # Another process could replace the reserved path after its exclusive creation; the open handle stays the reserved file.
        # optimize=False keeps each palette and transparent index as they are; frames after the first carry their palette
        # as a local colour table, and Pillow crops each to where it differs from the frame before.
        self._frames[0].save(
            self._handle,
            format="GIF",
            save_all=True,
            append_images=self._frames[1:],
            duration=int(1000 / self.fps),
            loop=0,
            disposal=_GIF_KEEP_DISPOSAL,
            optimize=False,
        )


def _state_norms_over_all_times(record: Any, plot_kwargs: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the state-map norms over all stored times for the viewer's settings: the scales of every GIF frame."""
    return analysis.evaluation.plots.transient.state_map_norms(
        record,
        state_fields=plot_kwargs["state_fields"],
        time_indices=range(len(record.physical_times)),
        error_mode=plot_kwargs["error_mode"],
        reference_scales=plot_kwargs["reference_scales"],
        lock_scale=plot_kwargs["lock_scale"],
        fixed_time_fields=plot_kwargs["fixed_time_fields"],
    )


def _viewer_state_norms(cache: dict[str, Any], entry: Any, record: Any, plot_kwargs: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return the norms over all stored times while any drawn field is fixed over time, kept in ``cache`` for the next time step, else ``None``."""
    if not plot_kwargs["fixed_time_fields"]:
        return None
    # The display stride only samples the drawn cells, so it never enters the norms.
    key = (
        entry.frame_name,
        entry.case_id,
        *(plot_kwargs[name] for name in ("state_fields", "lock_scale", "error_mode", "fixed_time_fields", "reference_scales")),
    )
    if cache.get("key") != key:
        cache.update(key=key, norms=_state_norms_over_all_times(record, plot_kwargs))
    return cache["norms"]


def _write_state_map_gif(
    handle: IO[bytes],
    *,
    record: Any,
    times: tuple[float, ...],
    plot_kwargs: dict[str, Any],
    fps: int,
    progress: Callable[[int], None],
) -> None:
    """Render one frame per stored time with the viewer's settings on one reused figure; write them as a looping GIF at ``fps``."""
    transient = analysis.evaluation.plots.transient
    norms = _state_norms_over_all_times(record, plot_kwargs)
    geometry = transient.state_map_geometry(record, state_fields=plot_kwargs["state_fields"])
    figure, _grid = analysis.ui.plot_layout.map_figure(geometry)
    try:
        # The writer keeps its frames in memory; the setup path only names the reserved file's folder.
        with _HandleGifWriter(handle, fps).saving(figure, str(Path(handle.name)), _GIF_DPI) as writer:
            for done, time in enumerate(times, start=1):
                transient.plot_state_maps(record, physical_time=time, norms=norms, figure=figure, **plot_kwargs)
                writer.grab_frame()
                progress(done)
    finally:
        plt.close(figure)


def _transient_material_role(entry: Any) -> str:
    """Return the material and declared evaluation role of one sample-viewer case."""
    assignment = entry.evaluation_assignment
    declared_role = str(assignment["role"]) if assignment is not None else entry.dataset_role
    return analysis.presentation.display_labels.material_role_display_label(entry.material_family, declared_role)


@dataclass(frozen=True, slots=True)
class FrameRun:
    """The run a session frame belongs to: its display label, and whether that is an explicit short label (a storage alias is never named)."""

    label: str
    named: bool


def _run_models(frame_runs: Mapping[str, FrameRun] | None, frame_names: Iterable[str]) -> tuple[str, ...]:
    """Return the short label of the run of each of ``frame_names`` for ``layout.models_suffix``; an unnamed or unmapped run is ``""``."""
    runs = frame_runs or {}
    return tuple(runs[name].label if name in runs and runs[name].named else "" for name in frame_names)


def _run_labels(frame_runs: Mapping[str, FrameRun] | None, frame_names: Sequence[str]) -> dict[str, str]:
    """Return the shown label of each of ``frame_names``, frames of distinct runs: its run label (else itself), equal ones told apart by frame."""
    runs = frame_runs or {}
    bases = [runs[name].label if name in runs else name for name in frame_names]
    shown = analysis.presentation.display_labels.unique_display_labels(bases, [(name,) for name in frame_names])
    return dict(zip(frame_names, shown, strict=True))


def _transient_sample_title(entry: Any, *, outlier: bool, rank: tuple[str, int] | None = None, ranked: bool = True) -> str:
    """
    Return the state-map title context: the picked case, its material, role and number.

    An outlier is named by its ranking field and rank (``rank``: field key, rank) where known: rank one is the «Largest» error.
    An outlier view without any finite error to rank (not ``ranked``) names its case «Unranked».
    """
    if not outlier:
        lead = "Selected sample:"
    elif not ranked:
        lead = "Unranked case in"
    elif rank is None:
        lead = "Largest error in"
    else:
        field = analysis.presentation.moisture_display.display_name(rank[0], mathtext=True)
        lead = f"Largest {field} error in" if rank[1] == 1 else f"Rank {rank[1]} {field} error in"
    return f"{lead} {_transient_material_role(entry)} · {entry.case_label}"


def _transient_sample_caption(entry: Any, suffix: str) -> str:
    """Return the sample-viewer case caption: material and role, exact case and relation."""
    assignment = entry.evaluation_assignment
    relation = f" · relation: {assignment['relation']}" if assignment is not None else ""
    return f"{_transient_material_role(entry)} · Exact case: {entry.case_id}{relation}{suffix}"


def _transient_sample_stem(entry: Any, plot_name: str | None, channels: tuple[str, ...], time_part: str) -> str:
    """Return the export file stem of one sample-viewer case, channel selection and time part."""
    return "__".join(
        (
            "transient_drying",
            entry.frame_name,
            str(plot_name or "sample"),
            f"case_{entry.case_id}",
            f"channel_{'-'.join(channels)}",
            time_part,
        )
    )


def _export_state_map_gif(status: analysis.ui.controls.StatusLine, *, stem: str, record: Any, plot_kwargs: dict[str, Any], fps: int) -> None:
    """Refuse a GIF whose frames would exceed the memory limit, otherwise export it through the shared allocator."""
    channels = plot_kwargs["state_fields"]
    if not channels:
        status.set("[GIF] Select at least one compatible channel.", "warning")
        return
    times = tuple(float(value) for value in record.physical_times)
    geometry = analysis.evaluation.plots.transient.state_map_geometry(record, state_fields=channels)
    frame_bytes = _gif_peak_bytes(len(times), round(geometry.figure_width * _GIF_DPI) * round(geometry.figure_height * _GIF_DPI))
    if frame_bytes > _GIF_MEMORY_LIMIT_BYTES:
        status.set(
            f"[GIF] Refused: {len(times)} frames need about {frame_bytes / 1e9:.1f} GB (limit {_GIF_MEMORY_LIMIT_BYTES / 1e9:.1f} GB). "
            "Deselect channels or choose a case with fewer stored times.",
            "warning",
        )
        return

    def progress(done: int) -> None:
        if done % _GIF_PROGRESS_STEP == 0 and done < len(times):
            status.progress("[GIF] Error scales and the fixed state scales over all times, frame", done, len(times))

    analysis.ui.notebook.write_export(
        status,
        stem=stem,
        suffix=".gif",
        write=partial(_write_state_map_gif, record=record, times=times, plot_kwargs=plot_kwargs, fps=fps, progress=progress),
    )


def _publish_transient_capabilities(
    selection_state: EvaluationSelectionState,
    fields: Sequence[Any],
    case_ids: Sequence[str],
    *,
    physical_times: Sequence[float] = (),
    scope: str | None = None,
    channels: tuple[str, ...] = (),
    case_id: str | None = None,
    physical_time: float | None = None,
) -> None:
    """Bind the full-rollout coordinates a transient view offers and publish what it shows to the other views."""
    selection_state.bind_capabilities(
        analysis.evaluation.selection.EvaluationViewCapabilities(
            task="transient_drying",
            channels=tuple(field.key for field in fields),
            case_ids=tuple(dict.fromkeys(case_ids)),
            physical_times=tuple(float(value) for value in physical_times),
            protocols=("autonomous_full",),
            horizons=("full",),
        )
    )
    if scope is not None:
        selection_state.select_scope(cast("Any", scope))
    selection_state.select_channels(channels)
    if case_id is not None:
        selection_state.select_case(case_id)
    selection_state.select_protocol("autonomous_full")
    selection_state.select_horizon("full")
    if physical_time is not None:
        selection_state.select_physical_time(physical_time)


# The time a user last stepped to on a time navigator of one of a session's views, per session. A shared selection's own time
# cannot tell such a choice from a view's default, because every view publishes the time it draws. Every run or artifact-role
# change opens a new session, so a choice stays across the views and cases of one session and never reaches another run's.
_CHOSEN_TIMES: WeakKeyDictionary[TransientEvaluationSession, float] = WeakKeyDictionary()


def _on_time_chosen(session: TransientEvaluationSession, redraw: Callable[[], None]) -> Callable[[TimeStepSelection], None]:
    """Return the change callback of a time navigator: remember the user's time for the session's views, then redraw."""

    def chosen(selection: TimeStepSelection) -> None:
        _CHOSEN_TIMES[session] = float(selection.physical_time)
        redraw()

    return chosen


def _chosen_or_default_time(session: TransientEvaluationSession, times: Sequence[float], default: float) -> float:
    """Return the time the user chose on a view of the session where ``times`` stores it, else ``default``."""
    chosen = _CHOSEN_TIMES.get(session)
    return chosen if chosen is not None and chosen in times else default


def _coverage_time(shown: Mapping[str, Sequence[Any]]) -> float:
    """
    Return the latest stored time that every shown record holds, so every selected case has a state there: the snapshot default.

    Full rollouts all store their origin, so a common time exists; records without one are refused rather than given another default.
    """
    stored = [{float(value) for value in record.physical_times} for records in shown.values() for record in records]
    common = set.intersection(*stored) if stored else set()
    if not common:
        msg = "The shown records share no stored time, so the snapshot has no time at which every selected case has a state."
        raise ValueError(msg)
    return max(common)


def _bind_snapshot_clock(
    session: TransientEvaluationSession,
    clock: TimeNavigator,
    shown: Mapping[str, Sequence[Any]],
    *,
    at_phase: bool,
) -> None:
    """Bind a snapshot navigator to the times the shown records store, at the chosen or covered time; only a shown navigator keeps its time."""
    if shown:
        times = _shared_transient_physical_times(tuple(record for records in shown.values() for record in records))
        # On the drying phase the navigator is hidden: it reports no time, so another control's message on the status line stays.
        clock.rebind(times, preserve_time=_chosen_or_default_time(session, times, _coverage_time(shown)), notify=False, report=not at_phase)
    if not shown or at_phase:
        clock.clear_status()  # the navigator is hidden: its time is no selection of an empty view or of the drying phase


def _make_transient_sample_viewer(
    *,
    session: TransientEvaluationSession,
    fields: Sequence[Any],
    comparison: bool,
    selection_state: EvaluationSelectionState | None = None,
    export_state: dict[str, Any] | None = None,
    export_plot_name: str | None = None,
    export_title: str | None = None,
    outlier: bool = False,
    refused_roles: Mapping[str, str] | None = None,
    frame_runs: Mapping[str, FrameRun] | None = None,
) -> ViewPanel:
    """Build the EDA-native selected-case Reference/Prediction/Error view of one dataset (the focus dataset), titled with its frame's run label."""
    ui = analysis.ui
    admitted_fields = _supported_transient_fields(fields)
    if not admitted_fields:
        msg = "Transient sample views require representable channels."
        raise ValueError(msg)
    entries = _transient_inventory(session, comparison=comparison, selection_state=selection_state)
    export: dict[str, Any] = {"export_state": export_state, "export_plot_name": export_plot_name, "export_title": export_title}
    status = ui.controls.StatusLine()
    view: ViewPanel | None = None
    ranks: dict[tuple[str, str], tuple[str, int]] = {}  # each ranked case's ranking field and rank, for its title
    focus = _CaseFocus(
        session,
        entries,
        inventory=dataset_cuts.transient_inventory(session, refused=refused_roles or {}),
        multiple=False,
        comparison=comparison,
        selection_state=selection_state,
        status=status,
        on_change=lambda: cast("ViewPanel", view).request(refresh),
        order=_transient_outlier_order(session, fields=admitted_fields, ranks=ranks) if outlier else None,
    )
    if not focus.shown:
        msg = "Transient field views require at least one exact case in the chosen dataset."
        raise ValueError(msg)
    channels = _Channels("Channels", admitted_fields, selection_state)
    scale_lock = ui.controls.scale_lock_checkbox(value=True)
    transient = analysis.evaluation.plots.transient
    fixed_time = channels.companion("Fixed over time", transient.default_fixed_time_fields(channels.checks.options))
    error_mode = ui.controls.error_mode_group("transient")
    display_stride = ui.controls.display_stride_dropdown()
    gif_fps = ui.controls.int_field("fps", value=_GIF_FPS, minimum=1, maximum=10)  # frames per second of the GIF
    gif_button = ui.controls.action_button("Export GIF", kind="export")
    gif_button.tooltip = "Export a GIF over all stored times; error and fixed-over-time state scales stay fixed, other state scales follow each frame"
    gif_guard = ui.controls.BusyGuard([gif_button], status)

    def selected_record() -> Any:
        return _full_autonomous_record(session, frame_name=focus.entry.frame_name, case_id=focus.entry.case_id)

    times = tuple(float(value) for value in selected_record().physical_times)
    clock = (
        None
        if outlier
        else ui.navigation.TimeNavigator(
            times,
            value=_chosen_or_default_time(session, times, times[-1]),
            status=status,
            on_change=_on_time_chosen(session, lambda: cast("ViewPanel", view).request(render)),
        )
    )
    rows = [
        ui.panel.row(focus.choice.selector.header, focus.navigator.widget),
        ui.panel.row(channels.checks.widget),
        ui.panel.row(error_mode, scale_lock, fixed_time.widget, display_stride),
    ]
    if clock is not None:
        rows.append(ui.panel.row(clock.widget, gif_fps, gif_button))

    def plot_kwargs(entry: Any, selected: tuple[str, ...], record: Any) -> dict[str, Any]:
        return {
            "state_fields": selected,
            "lock_scale": bool(scale_lock.value),
            "fixed_time_fields": tuple(key for key in fixed_time.selected if key in selected),
            "error_mode": str(error_mode.value),
            "reference_scales": _transient_reference_scales(session, entry.frame_name, selected, record),
            "display_spatial_stride": display_stride.value,
            # An empty ranking is the inventory fallback: no case of the chosen datasets has a finite error.
            "title_context": _transient_sample_title(entry, outlier=outlier, rank=ranks.get((entry.frame_name, entry.case_id)), ranked=bool(ranks)),
            "models": _run_models(frame_runs, (entry.frame_name,)),  # the run's label ends the title, after the time
        }

    norms_cache: dict[str, Any] = {}  # the norms over all stored times of the last setting, replaced when the case or a norm setting changes

    def render_now() -> None:
        current = cast("ViewPanel", view)
        selected = channels.checks.selected
        if not selected:
            ui.viewers.clear_view(current, "Select at least one compatible channel.", **export)
            return
        entry, record = focus.entry, selected_record()
        physical_time = (
            analysis.evaluation.plots.transient.maximum_error_physical_time(record, state_fields=selected)
            if clock is None
            else clock.selection.physical_time
        )
        if selection_state is not None:
            _publish_transient_capabilities(
                selection_state,
                admitted_fields,
                [item.case_id for item in entries],
                physical_times=record.physical_times,
                channels=selected,
                case_id=entry.case_id,
                physical_time=physical_time,
            )
        caption = _transient_sample_caption(entry, f" · selected-channel error-max time: {physical_time:g} h" if outlier else "")
        kwargs = {"physical_time": physical_time, **plot_kwargs(entry, selected, record)}

        def draw() -> tuple[str, Figure]:
            return caption, transient.plot_state_maps(record, norms=_viewer_state_norms(norms_cache, entry, record, kwargs), **kwargs)

        if ui.viewers.render_view(current, draw, **export) and export_state is not None:
            export_state["filename_stem"] = _transient_sample_stem(entry, export_plot_name, selected, f"t_{physical_time:g}h")

    def refresh_now() -> None:
        """Load the shown case's record, keep the shared time where the case stores it, and draw once."""
        times = tuple(float(value) for value in selected_record().physical_times)
        if clock is not None:
            clock.rebind(times, preserve_time=_chosen_or_default_time(session, times, times[-1]), notify=False)
        render_now()

    def rebind_now() -> None:
        """List the cases of the newly chosen dataset again and refresh."""
        if focus.rebind():
            refresh_now()
        else:
            ui.viewers.clear_view(cast("ViewPanel", view), "The chosen dataset has no exact case.", **export)

    def activate_now() -> None:
        """Take the dataset, case, channels and time the other views chose, then draw once."""
        current = cast("ViewPanel", view)
        with current.batch():
            focus.pull()
            channels.pull()
            current.request(refresh)

    def gif_export() -> None:
        entry, record = focus.entry, selected_record()
        selected = channels.checks.selected
        kwargs = plot_kwargs(entry, selected, record)
        fixed = kwargs["fixed_time_fields"]
        _export_state_map_gif(
            status,
            stem=_transient_sample_stem(entry, export_plot_name, selected, f"t-all__fixed-{'-'.join(fixed)}" if fixed else "t-all"),
            record=record,
            plot_kwargs=kwargs,
            fps=int(gif_fps.value),
        )

    view = ui.panel.ViewPanel(
        rows,
        status=status,
        slot=ui.panel.PlotSlot(),
        popovers=[(focus.choice.selector.header, focus.choice.selector.body)],
        on_activate=activate_now,
    )
    render = _guard(view, export, render_now)
    refresh = _guard(view, export, refresh_now)
    rebind = _guard(view, export, rebind_now)
    focus.choice.selector.observe(lambda: cast("ViewPanel", view).request(rebind))
    for checks in (channels.checks, fixed_time):
        checks.observe(lambda: cast("ViewPanel", view).request(render))
    for control in (scale_lock, error_mode, display_stride):
        control.observe(lambda _change: cast("ViewPanel", view).request(render), names="value")
    gif_button.on_click(lambda _button: gif_guard.run(gif_export, busy_text="[GIF] Exporting …"))
    view.request(refresh)
    return view


def _largest_dataset(session: TransientEvaluationSession, choice: _SharedChoice) -> int:
    """Return the case count of the largest dataset the selector offers: the bound of the view's case count."""
    inventory = choice.selector.inventory
    everything = analysis.presentation.dataset_selection.resolve_choices(inventory, [item.key for item in inventory.items], pooled=False)
    return max((len(context.case_ids) for context in dataset_cuts.transient_contexts(session, everything)), default=1)


def _transient_series(
    session: TransientEvaluationSession,
    contexts: Sequence[dataset_cuts.TransientDatasetContext],
    *,
    limit: int,
    comparison: bool,
    frame_runs: Mapping[str, FrameRun] | None = None,
) -> dict[str, tuple[str, tuple[Any, ...], str]]:
    """
    Return the frame, the full-rollout records of the first ``limit`` cases and the dataset label of every dataset and frame.

    The frame is the series' identity, which every join uses; the key is only its shown name. A comparison names each series
    by its run (the run label of ``frame_runs``, else the frame; equal ones told apart) before the dataset label, which names
    the role; a single model has one frame per dataset.
    """
    series: dict[str, tuple[str, tuple[Any, ...], str]] = {}
    for context in contexts:
        runs = _run_labels(frame_runs, context.frame_names)  # the frames of one dataset are distinct runs
        for frame_name in context.frame_names:
            held = set(session.case_ids(frame_name))
            label = f"{runs[frame_name]} · {context.label}" if comparison else context.label
            records = tuple(
                _full_autonomous_record(session, frame_name=frame_name, case_id=case_id)
                for case_id in [case_id for case_id in context.case_ids if case_id in held][:limit]
            )
            series[label] = frame_name, records, context.label
    return series


def _make_transient_aggregate_viewer(
    *,
    session: TransientEvaluationSession,
    fields: Sequence[Any],
    plot_kind: str,
    comparison: bool,
    evaluation_compute_device: str | compute.AnalysisComputePreference = "cpu",
    selection_state: EvaluationSelectionState | None = None,
    export_state: dict[str, Any] | None = None,
    export_plot_name: str | None = None,
    export_title: str | None = None,
    refused_roles: Mapping[str, str] | None = None,
    frame_runs: Mapping[str, FrameRun] | None = None,
) -> ViewPanel:
    """Build one exact-time multi-dataset aggregate with EDA-native controls; the title names the runs of the chosen frames."""
    if plot_kind not in {"predicted_reference", "mean_error", "std_error"}:
        msg = "Unknown transient aggregate field-view kind."
        raise ValueError(msg)
    ui = analysis.ui
    admitted_fields = _supported_transient_fields(fields)
    export: dict[str, Any] = {"export_state": export_state, "export_plot_name": export_plot_name, "export_title": export_title}
    choice = _SharedChoice(
        dataset_cuts.transient_inventory(session, refused=refused_roles or {}), multiple=True, pool_reason=None, selection_state=selection_state
    )
    maximum = _largest_dataset(session, choice)
    all_case_ids = [entry.case_id for entry in session.case_inventory()]
    status = ui.controls.StatusLine()
    channels = _Channels("Channels", admitted_fields, selection_state)
    # owners: the frame each shown record comes from; datasets: each series' dataset label, which a series without a case keeps
    active: dict[str, Any] = {"key": None, "series": None, "owners": {}, "frames": (), "datasets": {}}
    view: ViewPanel | None = None

    def series() -> dict[str, tuple[Any, ...]]:
        contexts = dataset_cuts.transient_contexts(session, choice.selector.choices())
        key = (tuple((context.label, context.frame_names) for context in contexts), count.value)
        if active["key"] != key:
            active["key"] = key
            active["frames"] = tuple(name for context in contexts for name in context.frame_names)
            pairs = _transient_series(session, contexts, limit=count.value, comparison=comparison, frame_runs=frame_runs)
            active["series"] = {label: records for label, (_frame, records, _dataset) in pairs.items()}
            active["owners"] = {id(record): frame for frame, records, _dataset in pairs.values() for record in records}
            active["datasets"] = {label: dataset for label, (_frame, _records, dataset) in pairs.items()}
        return cast("dict[str, tuple[Any, ...]]", active["series"])

    count = ui.navigation.CountControl(
        value=min(_CASE_COUNT_STEP, maximum),
        minimum=1,
        maximum=maximum,
        step=_CASE_COUNT_STEP,
        on_change=lambda _count: cast("ViewPanel", view).request(refresh),
    )
    clock = ui.navigation.TimeNavigator(  # hidden on a placeholder time until records exist: every refresh binds the shown times
        (0.0,), status=status, on_change=_on_time_chosen(session, lambda: cast("ViewPanel", view).request(render))
    )
    ui.controls.set_visible(clock.widget, False)
    status.clear()  # the placeholder time is not a selection
    clock_mode = ui.controls.clock_group("hours")
    percent = [round(100 * float(point)) for point in ui.time.PHASE_GRID]  # the slider steps through the phase grid in percent
    phase = ui.navigation.CountControl(
        value=percent[len(percent) // 2 - 1],
        minimum=percent[0],
        maximum=percent[-1],
        step=percent[1] - percent[0],
        description="Phase [%]",
        on_change=lambda _phase: cast("ViewPanel", view).request(render),
    )
    error_mode = ui.controls.error_mode_group("transient") if plot_kind == "mean_error" else None  # the std maps keep one meaning
    rows = [
        ui.panel.row(choice.selector.header, count.widget),
        ui.panel.row(channels.checks.widget),
        ui.panel.row(clock_mode, clock.widget, phase.widget),
        *([ui.panel.row(error_mode)] if error_mode is not None else []),
    ]

    def render_now() -> None:
        current = cast("ViewPanel", view)
        at_phase = clock_mode.value == "phase"
        shown = series()
        ui.controls.set_visible(clock.widget, bool(shown) and not at_phase)  # only the chosen clock's navigator shows, once records exist
        ui.controls.set_visible(phase.widget, at_phase)
        selected = channels.checks.selected
        if not shown:
            ui.viewers.clear_view(current, ui.viewers.EMPTY_SELECTION_MESSAGE, **export)
            return
        if not selected:
            ui.viewers.clear_view(current, "Select at least one compatible channel.", **export)
            return
        physical_time, tau = clock.selection.physical_time, phase.value / 100
        if at_phase:  # every record passes: the builder pools the reached ones and names its cohort
            when, supported, coverage, time_part = {"phase": tau}, shown, f"Drying phase τ={tau:.2f}", f"tau_{tau:.2f}"
        else:
            when, time_part = {"physical_time": physical_time}, f"t_{physical_time:g}h"
            supported = {label: tuple(record for record in records if physical_time in record.physical_times) for label, records in shown.items()}
            coverage = f"Exact t={physical_time:g} h support · " + "; ".join(f"{label}: n(t)={len(records)}" for label, records in supported.items())
        if selection_state is not None:
            _publish_transient_capabilities(
                selection_state, admitted_fields, all_case_ids, physical_times=clock.times_h, channels=selected, physical_time=physical_time
            )
        plot_func: Callable[..., Figure]
        if plot_kind == "predicted_reference":
            plot_func = analysis.evaluation.plots.transient.plot_predicted_vs_reference_channels
            kwargs: dict[str, Any] = {"state_fields": selected, **when, "styles": session.series_styles()}
        else:
            plot_func = analysis.evaluation.plots.transient.plot_aggregate_error_maps
            preference = evaluation_compute_device
            requested_device = preference.requested if isinstance(preference, compute.AnalysisComputePreference) else preference
            kwargs = {
                "state_fields": selected,
                **when,
                "statistic": "mean" if plot_kind == "mean_error" else "std",
                "compute_device": requested_device,
            }
            kwargs |= _series_reference_scales(session, active["owners"], supported, selected, error_mode)
        kwargs["models"], kwargs["context_labels"] = _run_models(frame_runs, active["frames"]), active["datasets"]

        def draw() -> tuple[str, Figure]:
            return coverage, plot_func(supported, **kwargs)

        if ui.viewers.render_view(current, draw, **export) and export_state is not None:
            export_state["filename_stem"] = "__".join(
                (
                    "transient_drying",
                    str(export_plot_name or plot_kind),
                    f"cases_{count.value}",
                    f"channel_{'-'.join(selected)}",
                    time_part,
                    "aggregate",
                )
            )

    def refresh_now() -> None:
        """Keep the shared time where the changed cases store it, and draw once."""
        _bind_snapshot_clock(session, clock, series(), at_phase=clock_mode.value == "phase")
        render_now()

    def device_changed() -> None:
        cast("ViewPanel", view).request(render)

    def subscribe_device() -> None:
        # The preference holds its listeners weakly; the activation callback the view keeps keeps this one alive with the view.
        if isinstance(evaluation_compute_device, compute.AnalysisComputePreference):
            evaluation_compute_device.subscribe(device_changed)

    def activate_now() -> None:
        current = cast("ViewPanel", view)
        subscribe_device()
        with current.batch():
            choice.pull()
            channels.pull()
            current.request(refresh)

    view = ui.panel.ViewPanel(
        rows,
        status=status,
        slot=ui.panel.PlotSlot(),
        popovers=[(choice.selector.header, choice.selector.body)],
        on_activate=activate_now,
    )
    render = _guard(view, export, render_now)
    refresh = _guard(view, export, refresh_now)
    choice.selector.observe(lambda: cast("ViewPanel", view).request(refresh))
    channels.checks.observe(lambda: cast("ViewPanel", view).request(render))
    if error_mode is not None:
        error_mode.observe(lambda _change: cast("ViewPanel", view).request(render), names="value")
    # A refresh, so the navigator reports its time again when the hours come back and its note goes with the phase.
    clock_mode.observe(lambda _change: cast("ViewPanel", view).request(refresh), names="value")
    ui.controls.set_visible(phase.widget, False)  # hours first
    subscribe_device()
    view.request(refresh)
    return view


def _make_transient_scope_viewer(
    *,
    session: TransientEvaluationSession,
    fields: Sequence[Any],
    plot_kind: str,
    comparison: bool,
    selection_state: EvaluationSelectionState | None = None,
    export_state: dict[str, Any] | None = None,
    export_plot_name: str | None = None,
    export_title: str | None = None,
    refused_roles: Mapping[str, str] | None = None,
    frame_runs: Mapping[str, FrameRun] | None = None,
) -> ViewPanel:
    """Build one local Aggregate/Single case trajectory-style view of the selected datasets; the title names the runs of the shown frames."""
    if plot_kind not in {"trajectory", "error_time", "target_magnitude"}:
        msg = "Unknown transient scoped view kind."
        raise ValueError(msg)
    ui = analysis.ui
    admitted_fields = _stored_transient_fields(fields) if plot_kind == "error_time" else _supported_transient_fields(fields)
    entries = _transient_inventory(session, comparison=comparison, selection_state=selection_state)
    export: dict[str, Any] = {"export_state": export_state, "export_plot_name": export_plot_name, "export_title": export_title}
    status = ui.controls.StatusLine()
    view: ViewPanel | None = None
    focus = _CaseFocus(
        session,
        entries,
        inventory=dataset_cuts.transient_inventory(session, refused=refused_roles or {}),
        multiple=True,
        comparison=comparison,
        selection_state=selection_state,
        status=status,
        on_change=lambda: cast("ViewPanel", view).request(render),
    )
    choice, navigator = focus.choice, focus.navigator
    maximum = _largest_dataset(session, choice)
    channels = _Channels("Stored channels" if plot_kind == "error_time" else "Channels", admitted_fields, selection_state)
    scope = ui.controls.scope_group(selection_state.selection.scope if selection_state is not None else "aggregate")
    active: dict[str, Any] = {"key": None, "series": None, "scaling_states": None, "frames": ()}

    def selected_series() -> tuple[dict[str, tuple[Any, ...]], dict[str, Mapping[str, Any]]]:
        contexts = dataset_cuts.transient_contexts(session, choice.selector.choices())
        entry = focus.entry if scope.value == "single" else None
        if entry is not None:
            key: tuple[Any, ...] = ("single", entry.frame_name, entry.case_id)
        else:
            key = ("aggregate", tuple((context.label, context.frame_names) for context in contexts), count.value)
        if active["key"] != key:
            active["frames"] = (entry.frame_name,) if entry is not None else tuple(name for context in contexts for name in context.frame_names)
            if entry is not None:
                label = _transient_case_label(entry, comparison=comparison)
                series = {label: (_full_autonomous_record(session, frame_name=entry.frame_name, case_id=entry.case_id),)}
                scaling_states = {label: session.scaling_state(entry.frame_name)}
            else:  # both mappings take their keys from the same series, each joined to the scaler of its own frame
                pairs = _transient_series(session, contexts, limit=count.value, comparison=comparison, frame_runs=frame_runs)
                series = {label: records for label, (_frame, records, _dataset) in pairs.items()}
                scaling_states = {label: session.scaling_state(frame) for label, (frame, _records, _dataset) in pairs.items()}
            active["key"], active["series"], active["scaling_states"] = key, series, scaling_states
        return cast("dict[str, tuple[Any, ...]]", active["series"]), cast("dict[str, Mapping[str, Any]]", active["scaling_states"])

    count = ui.navigation.CountControl(
        value=min(100, maximum),
        minimum=1,
        maximum=maximum,
        step=_CASE_COUNT_STEP,
        on_change=lambda _count: cast("ViewPanel", view).request(render),
    )
    pooled_time = plot_kind in {"trajectory", "error_time"}
    time_axis = _PooledClock(lambda: cast("ViewPanel", view).request(render))  # placed in the row of the pooled time views only
    rows = [
        ui.panel.row(choice.selector.header, scope, count.widget, navigator.widget),
        ui.panel.row(channels.checks.widget, *([time_axis.widget] if pooled_time else [])),
    ]

    def update_scope_row() -> None:
        """Expose only the navigator of the active scope: the case count (aggregate) or the case (single), and the clock only in aggregate."""
        ui.controls.set_visible(count.widget, scope.value == "aggregate")
        ui.controls.set_visible(navigator.widget, scope.value == "single")
        ui.controls.set_visible(time_axis.widget, scope.value == "aggregate")

    def render_now() -> None:
        current = cast("ViewPanel", view)
        selected = channels.checks.selected
        if not selected:
            ui.viewers.clear_view(current, "Select at least one compatible channel.", **export)
            return
        if not choice.selector.selected() or (scope.value == "single" and not focus.shown):
            ui.viewers.clear_view(current, ui.viewers.EMPTY_SELECTION_MESSAGE, **export)
            return
        series, scaling_states = selected_series()
        entry = focus.entry if scope.value == "single" else None
        if selection_state is not None:
            _publish_transient_capabilities(
                selection_state,
                admitted_fields,
                [item.case_id for item in entries],
                scope=str(scope.value),
                channels=selected,
                case_id=None if entry is None else entry.case_id,
            )
        clock = time_axis.resolve(series) if pooled_time and scope.value == "aggregate" else "hours"  # a single case keeps its hours
        plot_func: Callable[..., Figure]
        if plot_kind == "trajectory":
            plot_func = analysis.evaluation.plots.transient.plot_state_trajectory_summary
            kwargs: dict[str, Any] = {"state_fields": selected, "styles": session.series_styles(), "time_axis": clock}
        elif plot_kind == "error_time":
            plot_func = analysis.evaluation.plots.transient.plot_error_over_physical_time
            kwargs = {"scaling_states": scaling_states, "state_fields": selected, "styles": session.series_styles(), "time_axis": clock}
        else:
            plot_func = analysis.evaluation.plots.transient.plot_error_vs_target_magnitude
            kwargs = {"state_fields": selected, "styles": session.series_styles()}
        kwargs["models"] = _run_models(frame_runs, active["frames"])

        def draw() -> Figure:
            return plot_func(series, **kwargs)

        if ui.viewers.render_view(current, draw, **export) and export_state is not None:
            case_part = f"case_{entry.case_id}" if entry is not None else f"cases_{count.value}"
            export_state["filename_stem"] = "__".join(
                ("transient_drying", str(export_plot_name or plot_kind), str(scope.value), case_part, f"channel_{'-'.join(selected)}")
                + ((clock,) if pooled_time else ())
            )

    def rebind_now() -> None:
        """List the cases of the newly selected datasets again and draw once."""
        focus.rebind()
        render_now()

    def activate_now() -> None:
        """Take the datasets, case, scope and channels the other views chose, then draw once."""
        current = cast("ViewPanel", view)
        with current.batch():
            focus.pull()
            if selection_state is not None and selection_state.selection.scope != scope.value:
                scope.value = selection_state.selection.scope
            channels.pull()
            current.request(render)

    def scope_changed() -> None:
        update_scope_row()
        cast("ViewPanel", view).request(render)

    view = ui.panel.ViewPanel(
        rows,
        status=status,
        slot=ui.panel.PlotSlot(),
        popovers=[(choice.selector.header, choice.selector.body)],
        on_activate=activate_now,
    )
    render = _guard(view, export, render_now)
    rebind = _guard(view, export, rebind_now)
    choice.selector.observe(lambda: cast("ViewPanel", view).request(rebind))
    channels.checks.observe(lambda: cast("ViewPanel", view).request(render))
    scope.observe(lambda _change: scope_changed(), names="value")
    update_scope_row()
    view.request(render)
    return view


def _transient_summary_table(session: TransientEvaluationSession) -> pd.DataFrame:
    """
    Return a compact aggregate table without exposing identity hashes; the artifact names its run and role.

    A missing value shows its counters, as the steady overview does, once per row in ``availability`` where the whole row
    lacks values (no case at its horizon, or the support counts of a group with non-finite predictions), its metric cells
    empty; a relative norm missing alone names its cases below the reference floor in its own cell. The rows stand in the display
    order of every view: role group, then material display label, then the frames in order, mode, horizon ascending with ``full``
    last, scope; so the runs of one dataset stand together.
    """
    frame = session.dataset_dataframe()
    transient = analysis.evaluation.plots.transient
    first = {column: list(dict.fromkeys(frame[column])) for column in ("frame", "mode", "scope")}
    roles = frame["dataset_role"].astype(str).tolist() if "dataset_role" in frame else [""] * len(frame)  # one per frame
    keys = [
        (
            transient.dataset_order(str(row["material_family"]), role),
            first["frame"].index(row["frame"]),
            first["mode"].index(row["mode"]),
            transient.horizon_order(row["requested_horizon"]),
            first["scope"].index(row["scope"]),
        )
        for row, role in zip(frame.to_dict("records"), roles, strict=True)
    ]
    frame = frame.iloc[sorted(range(len(frame)), key=keys.__getitem__)]
    identity = ("frame", "material_family", "mode", "requested_horizon", "scope", "contributing_record_count")
    relative = ("physical_relative_l2", "physical_relative_h1")
    errors = ("normalized_drying_group_macro_rmse", *(f"normalized_rmse_{field}" for field in transient_artifact.STATE_ORDER), *relative)

    def availability(row: Mapping[Any, Any]) -> str:
        if row["contributing_case_count"] == 0:
            missing = row["unavailable_case_count"]
            return f"unavailable (no cases at this horizon; {missing} missing)" if missing else "unavailable (no cases at this horizon)"
        return validity_contract.unavailable_text(row) if row["finite_value_count"] < row["required_value_count"] else ""

    def shown(row: Mapping[Any, Any], name: str, gap: str) -> object:
        value = row[name]
        if gap or (value is not None and not pd.isna(value)):
            return value
        return f"unavailable ({row[f'{name}_unavailable_cases']} case-field values below the reference floor)"

    rows = []
    for row in frame.to_dict("records"):
        gap = availability(row)
        rows.append(
            {**{name: row[name] for name in identity if name in row}, **{name: shown(row, name, gap) for name in errors}, "availability": gap}
        )
    return pd.DataFrame.from_records(rows)


def _transient_comparison_summary(session: TransientEvaluationSession) -> pd.DataFrame:
    """
    Return one ranked full-rollout model comparison scoreboard whose error columns are the session's selection errors.

    Every row ranks by ``_RANKING_CRITERION``, which the table's title names once; a physical error column carries its unit, the
    humidity errors in percentage points like every shown difference of a fraction, converted on the ranked copy only.
    """
    field_labels = analysis.presentation.field_labels
    frame = session.dataset_dataframe(modes=("autonomous_full",))
    selected = frame.loc[(frame["requested_horizon"] == "full") & (frame["scope"] == "cumulative")]
    ranked = pd.DataFrame.from_records([{"frame": name, **session.selection_errors(name)} for name in dict.fromkeys(selected["frame"])])
    ranked = ranked.sort_values("normalized_drying_group_macro_rmse", na_position="last").reset_index(drop=True)
    shown: dict[str, str] = {}
    for field, unit in zip(transient_artifact.STATE_ORDER, transient_artifact.STATE_UNITS, strict=True):
        percent = field_labels.shown_in_percent(field)  # phi: a fraction shown in percent, its errors in pp
        for column in (f"physical_rmse_{field}", f"physical_mae_{field}"):
            ranked[column] = field_labels.display_values(ranked[column].to_numpy(dtype=np.float64), unit, field=field, quantity_kind="difference")
            shown[column] = f"{column} [{field_labels.display_unit(unit, field=field, quantity_kind='difference') if percent else unit}]"
    ranked = ranked.rename(columns=shown)
    ranked["Material count"] = ranked["frame"].map(selected.groupby("frame")["material_family"].nunique())
    ranked["Metric available"] = ranked["normalized_drying_group_macro_rmse"].map(lambda value: pd.notna(value) and isfinite(float(value)))
    return ranked


def _comparison_summary_view(session: TransientEvaluationSession, frame_runs: Mapping[str, FrameRun] | None = None) -> widgets.VBox:
    """
    Show the ranked scoreboard compactly: one column per frame (rank, run label and role), the criterion first, then a block per field.

    Only a frame with a finite criterion has a rank; the others follow under their run label and role alone. With several frames
    each row is shaded across the runs, so every metric compares against its own values; the counts and the availability are text.
    """
    ranked = _transient_comparison_summary(session)
    frames = [str(name) for name in ranked["frame"]]
    runs = _frame_role_labels(session, frame_runs)  # one run's frames of two roles are told apart by role, never by the frame name
    raw = ranked.drop(columns="frame").assign(**{"Material count": ranked["Material count"].map(str)})  # a count is no error to shade
    criterion = "normalized_drying_group_macro_rmse"
    blocks = [column for field in transient_artifact.STATE_ORDER for column in raw.columns if str(column).split(" [")[0].endswith(f"_{field}")]
    order = [criterion, *blocks, *(column for column in raw.columns if column not in {*blocks, criterion})]
    shown = analysis.evaluation.plots.run_summary.display_table(raw[order])
    # The available frames stand first (``_transient_comparison_summary``), so their positions are their ranks.
    ranks = zip(range(1, len(frames) + 1), frames, ranked["Metric available"], strict=True)
    shown.index = pd.Index([f"{rank}. {runs[frame]}" if available else runs[frame] for rank, frame, available in ranks])
    title = f"Global comparison summary, ranked by {_RANKING_CRITERION}"
    return analysis.ui.tables.styled_dataframe(shown.transpose(), title=title, row_local=len(frames) > 1, row_heading="Metric")


def _transient_work_table(session: TransientEvaluationSession, frame_runs: Mapping[str, FrameRun] | None = None) -> pd.DataFrame:
    """Return one work row per run, named by its run label (else its first frame), with that artifact's drying-rollout model seconds per case."""
    run_summary = analysis.evaluation.plots.run_summary
    table = run_summary.build_frame_work_table(
        session.canonical_frames, lambda frame_name, _frame: run_summary.transient_inference_seconds(session.timing_report(frame_name))
    )
    return table.rename(index=_run_labels(frame_runs, list(table.index)))


def _transient_architecture_table(session: TransientEvaluationSession, frame_runs: Mapping[str, FrameRun] | None = None) -> pd.DataFrame:
    """Return exact model-family configuration from compact record identity, one row per run by its run label (else its first frame)."""
    rows = []
    first = list(analysis.evaluation.run_discovery.distinct_runs(session.canonical_frames))  # a run's first frame stands for it
    for frame_name, label in _run_labels(frame_runs, first).items():
        case_id = session.case_ids(frame_name)[0]
        identity = _full_autonomous_summary(
            session,
            frame_name=frame_name,
            case_id=case_id,
        ).identity
        if not isinstance(identity, Mapping):
            msg = "Transient architecture presentation requires record identity."
            raise TypeError(msg)
        parameters = identity.get("model_parameters")
        saved_config = session.canonical_frames[frame_name].attrs.get(analysis.evaluation.dataframe.COMPLETED_RUN_CONFIG_ATTR)
        model_config = saved_config if isinstance(saved_config, Mapping) else {"model": {"kind": identity.get("model_kind"), "params": parameters}}
        try:
            architecture_label = naming.architecture_label(model_config)
        except (KeyError, TypeError, ValueError):
            architecture_label = "Unavailable: incomplete architecture metadata"
        run = saved_config.get("run", {}) if isinstance(saved_config, Mapping) else {}
        lineage = identity.get("stage_identity")  # the artifact's stage mapping, where no run config was saved
        stage = run.get("layout", {}).get("stage", lineage.get("stage", "Unavailable") if isinstance(lineage, Mapping) else "Unavailable")
        configuration = analysis.evaluation.plots.run_summary.architecture_items(parameters) if isinstance(parameters, Mapping) else ()
        rows.append(
            {
                "Run": label,
                "model_kind": identity.get("model_kind"),
                "Architecture label": architecture_label,
                "Stage": stage.upper() if stage in {"a0", "b", "a1"} else stage,
                "Seed": run.get("seed", "Unavailable"),
                **(dict(configuration) or {"Configuration": "Unavailable"}),
                **(analysis.evaluation.plots.run_summary.channel_items(parameters) if isinstance(parameters, Mapping) else {}),
            }
        )
    return pd.DataFrame.from_records(rows).set_index("Run")


def _transient_outlier_table(
    session: TransientEvaluationSession,
    *,
    fields: Sequence[Any],
    frame_runs: Mapping[str, FrameRun] | None = None,
) -> pd.DataFrame:
    """
    Return per-channel worst complete-rollout cases from persisted metrics, ranked over every loaded dataset.

    A stored channel ranks by its normalized RMSE, the grain water by its physical RMSE, each in a column of its own unit. A
    case reads as its generation case number beside its exact id; its run is named by its label (else by its frame). Only a
    finite error ranks, as in the outlier navigator; without any the table is empty, and without any metric column it raises.
    """
    frame = session.case_dataframe(modes=("autonomous_full",))
    selected = frame.loc[(frame["requested_horizon"] == "full") & (frame["scope"] == "cumulative")]
    cases = {(entry.frame_name, entry.case_id): entry.case_label for entry in session.case_inventory()}
    runs = {name: run.label for name, run in (frame_runs or {}).items()}
    rows: list[dict[str, Any]] = []
    metrics = 0  # the channels with a metric column, finite or not
    for field in fields:
        column = f"normalized_rmse_{field.key}" if field.stored else "physical_w_gr_rmse"
        if column not in selected:
            continue
        metrics += 1
        error = "Normalized RMSE [1]" if field.stored else f"Physical RMSE [{field.unit}]"
        finite = selected.loc[np.isfinite(selected[column].to_numpy(dtype=np.float64))]  # as the navigator: ``nlargest`` ranks NaN and infinity
        for rank, row in enumerate(finite.nlargest(_OUTLIER_COUNT, column).itertuples(index=False), start=1):
            rows.append(
                {
                    "Channel": field.label,
                    "Rank": rank,
                    "Material · Role": analysis.presentation.display_labels.material_role_display_label(
                        str(row.material_family),
                        str(row.declared_evaluation_role),
                    ),
                    "Run": runs.get(str(row.frame), str(row.frame)),
                    "Case": cases.get((str(row.frame), str(row.case_id)), str(row.case_id)),
                    error: float(getattr(row, column)),
                    "Exact case": str(row.case_id),
                }
            )
    if not metrics:
        msg = "Transient outlier tables require at least one per-channel metric."
        raise ValueError(msg)
    if not rows:  # no finite error to rank: an explicitly empty table under its headings
        return pd.DataFrame(columns=["Channel", "Rank", "Material · Role", "Run", "Case", "Exact case"])
    return pd.DataFrame.from_records(rows)


def _transient_temporal_capabilities(
    session: TransientEvaluationSession,
) -> tuple[bool, bool, bool]:
    """Return full-trajectory, rollout-horizon, and final-target availability."""
    summaries = tuple(
        summary
        for frame_name in session.frame_names
        for case_id in session.case_ids(frame_name)
        for summary in session.record_summaries_for_case(frame_name, case_id)
    )
    full = any(summary.mode == "autonomous_full" and summary.requested_horizon == "full" for summary in summaries)
    horizon = any(
        summary.mode in {"rolling_origin", "autonomous_full"}
        and isinstance(summary.requested_horizon, int)
        and not isinstance(summary.requested_horizon, bool)
        for summary in summaries
    )
    target = any(
        isinstance(summary.target, Mapping)
        and (summary.target.get("reference_available") is True or summary.target.get("predicted_available") is True)
        for summary in summaries
        if summary.mode == "autonomous_full" and summary.requested_horizon == "full"
    )
    return full, horizon, target


def _pipeline_evidence(session: TransientEvaluationSession) -> dict[str, AirflowDegradationMetrics]:
    """
    Return the airflow substitution evidence of every model frame that has one, keyed by frame identity.

    The session returns evidence only for the frames whose B/A drying-surrogate error is finite, in frame order, so a
    frame's position in ``frame_names`` is not its position in that result.
    """
    available = [name for name in session.frame_names if isfinite(session.selection_objective(name))]
    return dict(zip(available, session.pipeline_degradation(), strict=True))


def _frame_role_labels(session: TransientEvaluationSession, frame_runs: Mapping[str, FrameRun] | None) -> dict[str, str]:
    """Return every frame named by its run's label and its role (``FNO · B · ID``), never by the frame; the frames of one role are distinct runs."""
    roles = {name: str(session.dataset_role(name)) for name in session.frame_names}
    labels: dict[str, str] = {}
    for role in dict.fromkeys(roles.values()):
        runs = _run_labels(frame_runs, [name for name in roles if roles[name] == role])
        labels |= {name: f"{run} · {analysis.presentation.display_labels.evaluation_role_display_label(role)}" for name, run in runs.items()}
    return labels


def _make_transient_runtime_viewer(
    *,
    session: TransientEvaluationSession,
    kind: str,
    export_state: dict[str, Any] | None = None,
    export_plot_name: str | None = None,
    export_title: str | None = None,
    frame_runs: Mapping[str, FrameRun] | None = None,
) -> ViewPanel:
    """
    Display saved timing evidence with explicit model and measurement boundaries; titles name the compared runs by ``frame_runs``.

    The duration ratios name the chosen frame's Dataset role and run; their figure note and status line also list every other
    selected frame, by its run label and role (``_frame_role_labels``), with the number of its ratios that have paired cases, and
    the status names each ratio's measured boundaries. The airflow substitution draws every selected frame with evidence in its
    run colour, names each one without it, and states each run's pipeline C availability in display text.
    """
    ui = analysis.ui
    plots = analysis.evaluation.plots.transient
    model = ui.controls.dropdown(session.frame_names, description="Saved model")
    measure = ui.controls.dropdown(
        (
            ("Prepared rollout request wall [s]", "prepared_wall"),
            ("Model calls only [s]", "model"),
            ("COMSOL drying solution / model-call time [ratio]", "ratio"),
        ),
        description="Measured boundary",
    )
    boundaries = (
        "Saved measurements only. Model calls exclude request preparation and scaling; "
        "prepared-request wall excludes Dataset/checkpoint loading and request construction. "
        "COMSOL reported solution and process times are separate boundaries. Duration ratios do not establish like-for-like end-to-end acceleration."
    )
    status = ui.controls.StatusLine(boundaries)
    controls = {"tradeoff": [ui.panel.row(measure)], "pipeline": []}.get(kind, [ui.panel.row(model)])  # the pipeline figure shows every run
    view = ui.panel.ViewPanel(controls, status=status, slot=ui.panel.PlotSlot())
    export: dict[str, Any] = {"export_state": export_state, "export_plot_name": export_plot_name, "export_title": export_title}

    def render_now() -> None:
        kwargs: dict[str, Any] = {}
        args: tuple[Any, ...]
        function: Callable[..., Any]
        frame_name = str(model.value)
        if kind == "tradeoff":
            frame = session.case_dataframe(modes=("autonomous_full",))
            frame = frame.loc[(frame["scope"] == "cumulative") & (frame["requested_horizon"] == "full")]
            function = plots.plot_accuracy_runtime_comparison
            args = (frame, {name: session.timing_report(name) for name in session.frame_names})
            kwargs["measure"] = measure.value
            kwargs["styles"] = session.series_styles()
            kwargs["models"] = _run_models(frame_runs, session.frame_names)
            kwargs["run_labels"] = {name: run.label for name, run in (frame_runs or {}).items()}
        elif kind == "pipeline":  # every selected run: its own panel in its colour, or its gap named
            evidence = _pipeline_evidence(session)
            labels = _frame_role_labels(session, frame_runs)
            roles = {str(session.dataset_role(name)) for name in session.frame_names}
            styles = session.series_styles()
            reason = plots.unavailable_reason_text
            gaps = [f"{labels[name]}: no finite B/A drying-surrogate error, so no evidence" for name in session.frame_names if name not in evidence]
            c_states = "; ".join(
                f"{labels[name]}: C {'available' if item.c_available else f'unavailable ({reason(item.c_unavailable_reason)})'}"
                for name, item in evidence.items()
            )
            status.set(
                "Airflow conditions A/B/C are reference, drying-surrogate and full airflow-surrogate pipelines; "
                f"they are separate from training stages A0/B/A1. Full pipeline C: {c_states or 'no run has evidence'}."
            )
            if not evidence:  # no run to draw: the empty view names every run's gap instead of a failure
                ui.viewers.clear_view(view, "No selected run has airflow substitution evidence. " + "; ".join(gaps), **export)
                return
            function = plots.plot_pipeline_degradation
            args = (tuple(evidence.values()),)
            kwargs = {
                "role": next(iter(roles)) if len(roles) == 1 else None,
                "models": _run_models(frame_runs, session.frame_names),
                "labels": [labels[name] for name in evidence],
                "styles": [styles[name] for name in evidence],  # colour and role hatch, as the global error bars
                "unavailable": gaps,
            }
        else:
            function = plots.plot_timing_speedups
            report = session.timing_report(frame_name)
            args = (report,)
            labels = _frame_role_labels(session, frame_runs)
            reports = {name: session.timing_report(name) for name in session.frame_names if name != frame_name}
            others = "; ".join(
                f"{labels[name]} ({sum(summary.available_count > 0 for summary in other.speedups.values())} of {len(other.speedups)} ratios paired)"
                for name, other in reports.items()
            )
            line = f"Other selected runs and roles: {others}" if others else None  # the figure and the status name them alike
            kwargs = {"role": str(session.dataset_role(frame_name)), "models": _run_models(frame_runs, (frame_name,)), "others": line}
            ratios = "; ".join(
                f"{analysis.presentation.display_labels.token_display_label(name.removesuffix('_speedup'))}: "
                f"{summary.reference_boundary} / {summary.surrogate_boundary}"
                for name, summary in report.speedups.items()
            )
            status.set(f"{boundaries} Ratios: {ratios}." + (f" {line}." if line else ""))
        published = ui.viewers.render_view(
            view,
            function,
            args=args,
            kwargs=kwargs,
            export_state=export_state,
            export_plot_name=export_plot_name,
            export_title=f"{export_title or kind} · {'all selected models' if kind != 'ratios' else model.value}",
        )
        if published and export_state is not None:
            context = "all_selected_models" if kind != "ratios" else str(model.value)
            export_state["filename_stem"] = f"transient_{context}_{export_plot_name or kind}_{measure.value if kind == 'tradeoff' else kind}"

    render = _guard(view, export, render_now)
    model.observe(lambda _change: view.request(render), names="value")
    measure.observe(lambda _change: view.request(render), names="value")
    view.request(render)
    return view


def _transient_matched_stage_status(reason: str | None) -> widgets.HTML:
    """Explain missing controlled comparison evidence without blocking general plots."""
    text = reason or "Select exactly one admitted A0/B/A1 group with shared A0 and immutable B budget evidence."
    sentence = text[:1].upper() + text[1:] + ("" if text.endswith(".") else ".")  # the reason is a clause of the lineage check
    return widgets.HTML(f"<b>Controlled stage comparison unavailable.</b> {escape(sentence)} General prediction comparisons remain available.")


def _transient_training_curves(
    session: TransientEvaluationSession, loader: Callable[[], Mapping[str, StageHistory]], frame_runs: Mapping[str, FrameRun] | None
) -> Figure:
    """
    Plot the training curves of the loaded stages; each run takes the colour its run has in the session's other views.

    The history label is the display label of the run, which ``frame_runs`` maps every frame to. A run takes the colour of its
    first frame, as its break-even does, also when it is the only run. The line style is always the plot's own: a history has
    no role. The title names the loaded histories by their labels where these are explicit short labels.
    """
    histories = loader()
    styles = None
    if frame_runs:
        session_styles = session.series_styles()
        styles = analysis.evaluation.plots.series.SeriesStyles()
        for frame, run in frame_runs.items():
            styles.setdefault(run.label, session_styles[frame])
        styles.run_colours = session_styles.run_colours
    named = {run.label for run in (frame_runs or {}).values() if run.named}
    models = tuple(label if label in named else "" for label in histories)
    return analysis.evaluation.plots.training.plot_training_curves(histories, styles=styles, models=models)


def _transient_training_compute(
    session: TransientEvaluationSession,
    loader: Callable[[], pd.DataFrame | None],
    lineage: LineageComparison | None,
    frame_runs: Mapping[str, FrameRun] | None = None,
) -> Figure:
    """
    Plot only the performance evidence admitted by the workflow lineage validator, with every arm's end from it and its run's colour.

    The legend names each arm's run by its label of ``frame_runs`` and the title the compared runs.
    """
    performance = loader()
    if performance is None or lineage is None:
        msg = "Admitted controlled stage comparison lacks performance evidence."
        raise RuntimeError(msg)
    return analysis.evaluation.plots.transient.plot_training_performance_vs_compute(
        performance,
        lineage=lineage,
        styles=session.series_styles(),
        run_labels={name: run.label for name, run in (frame_runs or {}).items()},
        models=_run_models(frame_runs, session.frame_names),
    )


def _transient_break_even(session: TransientEvaluationSession, frame_runs: Mapping[str, FrameRun] | None = None) -> tuple[Figure, str] | widgets.HTML:
    """
    Show the break-even of the loaded stages with the clocks of the drying-only speed-up (COMSOL drying study, drying model calls).

    The training seconds of B and A1 are those of the A0 their teacher handoff records plus their own. That A0 is the one loaded
    stage whose selected checkpoint has the handoff's checkpoint digest (a run name is not an identity); a B or A1 without
    exactly one such A0 is named with that reason (no A0 selected, none selected of its handoff, or several), apart from
    missing seconds of its own.
    Each clock is passed as recorded: the cost pairs the cases and counts the rest. A run is shown by its label of ``frame_runs``
    in its session colour; the title names the compared runs.
    """
    discovery = analysis.evaluation.run_discovery
    timing = analysis.evaluation.transient_timing
    # The distinct A0 runs with their selected checkpoint digest and training seconds.
    a0_runs = [
        (session.canonical_frames[label].attrs["artifact_provenance"].get("run", {}).get("best_checkpoint_sha256"), summary.get("elapsed_seconds"))
        for label, summary in discovery.distinct_runs(session.canonical_frames).items()
        if discovery.stage_work_row(summary)["stage"] == "A0"
    ]

    def evidence(label: str, _frame: pd.DataFrame, summary: Mapping[str, Any], work: Mapping[str, Any]) -> _BreakEvenEvidence:
        report = session.timing_report(label)
        comsol = timing.component_case_medians(report, "comsol_transient_drying_seconds")
        surrogate = timing.component_case_medians(report, "drying_no_rollout_model_seconds")
        solver = ", ".join(sorted({f"{case.solver_hostname} ({case.solver_requested_cores} cores)" for case in report.cases}))
        device = ", ".join(sorted({_device_text(case.device, case.gpu or case.cpu) for case in report.cases}))
        continued = work["stage"] in {"B", "A1"}
        prior: float | str = 0.0
        if continued:
            handoff = ((summary.get("teacher_handoff") or {}).get("files") or {}).get("checkpoint", {}).get("sha256")
            a0 = [elapsed for checkpoint, elapsed in a0_runs if handoff is not None and checkpoint == handoff]
            if not a0_runs:
                prior = "no A0 is selected: select the A0 of its handoff to cost the stage"
            elif len(a0) != 1:
                prior = f"{len(a0)} selected A0 runs have its handoff checkpoint, not one" if a0 else "no selected A0 has its handoff checkpoint"
            else:
                prior = "no training seconds of its A0" if a0[0] is None else float(a0[0])
        clocks = (
            f"COMSOL transient drying study time on {solver}; surrogate drying rollout model calls on {device}; "
            f"training seconds {'A0 + stage' if continued else 'of the stage'}"
        )
        return False, comsol, surrogate, prior, clocks, str(session.dataset_role(label))

    labels = _run_labels(frame_runs, list(discovery.distinct_runs(session.canonical_frames)))
    models = _run_models(frame_runs, session.frame_names)
    return _break_even_view(session.canonical_frames, evidence, styles=session.series_styles(), labels=labels, models=models)


def _transient_runtime_entry(
    session: TransientEvaluationSession, kind: str, frame_runs: Mapping[str, FrameRun] | None = None
) -> Callable[..., ViewPanel]:
    """Bind one lazy runtime viewer to the existing figure export context; ``frame_runs`` name the compared runs in its titles."""

    def build(
        *,
        export_state: dict[str, Any] | None = None,
        export_plot_name: str | None = None,
        export_title: str | None = None,
    ) -> ViewPanel:
        return _make_transient_runtime_viewer(
            session=session,
            kind=kind,
            export_state=export_state,
            export_plot_name=export_plot_name,
            export_title=export_title,
            frame_runs=frame_runs,
        )

    return build


def _transient_relative_norm_entry(
    session: TransientEvaluationSession,
    *,
    comparison: bool,
    selection_state: EvaluationSelectionState | None,
    refused_roles: Mapping[str, str] | None = None,
    frame_runs: Mapping[str, FrameRun] | None = None,
) -> Callable[..., ViewPanel]:
    """
    Bind the relative norms of the chosen datasets to one lazy view with the pooled views' clock: hours while each curve holds one case.

    Else the clock is the drying phase. The dataset choice starts from the shared one, else every dataset, as in the other
    aggregate views: one curve per chosen dataset (a material, or a role whose materials are pooled with equal weight) and frame,
    named like ``_transient_series``.
    """

    def build(
        *,
        export_state: dict[str, Any] | None = None,
        export_plot_name: str | None = None,
        export_title: str | None = None,
    ) -> ViewPanel:
        ui = analysis.ui
        export: dict[str, Any] = {"export_state": export_state, "export_plot_name": export_plot_name, "export_title": export_title}
        clock = _PooledClock(lambda: view.request(render))
        choice = _SharedChoice(
            dataset_cuts.transient_inventory(session, refused=refused_roles or {}), multiple=True, pool_reason=None, selection_state=selection_state
        )

        def render_now() -> None:
            contexts = dataset_cuts.transient_contexts(session, choice.selector.choices())
            if not contexts:
                ui.viewers.clear_view(view, ui.viewers.EMPTY_SELECTION_MESSAGE, **export)
                return
            shown: dict[str, tuple[str, tuple[str, ...]]] = {}  # each curve's frame and cases
            for context in contexts:
                runs = _run_labels(frame_runs, context.frame_names)  # the frames of one dataset are distinct runs
                for name in context.frame_names:
                    shown[f"{runs[name]} · {context.label}" if comparison else context.label] = (name, context.case_ids)
            time_axis = clock.resolve({label: cases for label, (_name, cases) in shown.items()})  # the cases each curve pools
            curves = {label: session.relative_norm_curve(name, time_axis=time_axis, case_ids=cases) for label, (name, cases) in shown.items()}
            plot = analysis.evaluation.plots.transient.plot_relative_norms_over_time
            kwargs = {"styles": session.series_styles(), "models": _run_models(frame_runs, dict.fromkeys(name for name, _cases in shown.values()))}
            if ui.viewers.render_view(view, plot, args=(curves,), kwargs=kwargs, **export) and export_state is not None:
                export_state["filename_stem"] = "__".join(("transient_drying", str(export_plot_name or "relative_norms"), time_axis))

        def activate_now() -> None:
            if choice.pull():
                view.request(render)

        view = ui.panel.ViewPanel(
            [ui.panel.row(choice.selector.header, clock.widget)],
            slot=ui.panel.PlotSlot(),
            popovers=[(choice.selector.header, choice.selector.body)],
            on_activate=activate_now,
        )
        render = _guard(view, export, render_now)
        choice.selector.observe(lambda: view.request(render))
        view.request(render)
        return view

    return build


def _build_transient_sections(
    *,
    session: TransientEvaluationSession,
    comparison: bool,
    selection_state: EvaluationSelectionState | None,
    evaluation_compute_device: str | compute.AnalysisComputePreference,
    training_performance: Callable[[], pd.DataFrame | None] | None = None,
    training_lineage: LineageComparison | None = None,
    training_histories: Callable[[], Mapping[str, StageHistory]],
    matched_stage_unavailable_reason: str | None = None,
    refused_roles: Mapping[str, str] | None = None,
    frame_runs: Mapping[str, FrameRun] | None = None,
) -> dict[str, tuple[list[tuple[str, Callable[..., Any], str]], str]]:
    """Bind compatible transient evidence under the established Evaluation sections."""
    models = _run_models(frame_runs, session.frame_names)  # the compared runs, which every figure of all frames names
    run_labels = {frame: run.label for frame, run in (frame_runs or {}).items()}  # the summary plots name a series' run by it
    fields = _transient_channel_fields(session)
    supported_fields = _supported_transient_fields(fields)
    stored_fields = _stored_transient_fields(fields)
    plots = analysis.evaluation.plots.transient
    run_summary = analysis.evaluation.plots.run_summary

    def table(title: str, build: Callable[[], pd.DataFrame]) -> Callable[[], widgets.VBox]:
        def show() -> widgets.VBox:
            raw = build()
            shown = run_summary.display_table(raw)  # renames each column, keeping their order
            # Shading means something only on error columns (lower is better), each against its own values, as in the steady tables.
            errors = [shown.columns[index] for index, column in enumerate(raw.columns) if str(column).startswith(("normalized_", "physical_"))]
            return analysis.ui.tables.styled_dataframe(shown, title=title, shade_columns=errors)

        return show

    summary_table = table("Summary table", partial(_transient_summary_table, session))
    comparison_summary = partial(_comparison_summary_view, session, frame_runs)

    def global_metrics() -> Figure:
        return plots.plot_channel_error(
            session.dataset_dataframe(),
            state_fields=tuple(field.key for field in stored_fields),
            styles=session.series_styles(),
            models=models,
            run_labels=run_labels,
        )

    def distributions() -> Figure:
        return plots.plot_error_distributions(
            session.case_dataframe(modes=("autonomous_full",)),
            state_fields=tuple(field.key for field in stored_fields),
            styles=session.series_styles(),
            models=models,
            run_labels=run_labels,
        )

    def architecture_table() -> widgets.VBox:
        # The configuration renderer of both tasks: one row per setting, a dash where a run lacks it, no row that every run lacks.
        shown = run_summary.display_configuration_table(_transient_architecture_table(session, frame_runs))
        return analysis.ui.tables.styled_dataframe(shown, title="Model-family configuration")

    outlier_table = table(
        "Worst per-channel cases over all loaded datasets (the field plots rank within their chosen dataset)",
        partial(_transient_outlier_table, session, fields=supported_fields, frame_runs=frame_runs),
    )

    def scoped_view(plot_kind: str) -> Callable[..., ViewPanel]:
        def build(
            *,
            export_state: dict[str, Any] | None = None,
            export_plot_name: str | None = None,
            export_title: str | None = None,
        ) -> ViewPanel:
            return _make_transient_scope_viewer(
                session=session,
                fields=supported_fields,
                plot_kind=plot_kind,
                comparison=comparison,
                selection_state=selection_state,
                export_state=export_state,
                export_plot_name=export_plot_name,
                export_title=export_title,
                refused_roles=refused_roles,
                frame_runs=frame_runs,
            )

        return build

    def aggregate_view(plot_kind: str) -> Callable[..., ViewPanel]:
        def build(
            *,
            export_state: dict[str, Any] | None = None,
            export_plot_name: str | None = None,
            export_title: str | None = None,
        ) -> ViewPanel:
            return _make_transient_aggregate_viewer(
                session=session,
                fields=supported_fields,
                plot_kind=plot_kind,
                comparison=comparison,
                evaluation_compute_device=evaluation_compute_device,
                selection_state=selection_state,
                export_state=export_state,
                export_plot_name=export_plot_name,
                export_title=export_title,
                refused_roles=refused_roles,
                frame_runs=frame_runs,
            )

        return build

    def sample_view(outlier: bool) -> Callable[..., ViewPanel]:
        def build(
            *,
            export_state: dict[str, Any] | None = None,
            export_plot_name: str | None = None,
            export_title: str | None = None,
        ) -> ViewPanel:
            return _make_transient_sample_viewer(
                session=session,
                fields=supported_fields,
                comparison=comparison,
                selection_state=selection_state,
                export_state=export_state,
                export_plot_name=export_plot_name,
                export_title=export_title,
                outlier=outlier,
                refused_roles=refused_roles,
                frame_runs=frame_runs,
            )

        return build

    configuration_entry = ("Model-family configuration", architecture_table, "model_family_configuration")
    overview_entries = [("Summary table", summary_table, "overview_summary_table")]
    if comparison:
        overview_entries.append(("Global comparison summary", comparison_summary, "overview_global_comparison_summary"))
    else:
        overview_entries.append(configuration_entry)
    registry: dict[
        str,
        tuple[list[tuple[str, Callable[..., Any], str]], str],
    ] = {
        "overview": (
            overview_entries,
            EVALUATION_SECTION_TITLES["overview"],
        ),
        "global_error": (
            [
                ("Global error metrics", global_metrics, "global_error_metrics"),
                ("Global error distribution", distributions, "global_error_distribution"),
                ("GT vs Prediction (mean)", aggregate_view("predicted_reference"), "gt_vs_prediction_mean"),
                ("Mean error maps", aggregate_view("mean_error"), "mean_error_maps"),
                ("Std error maps", aggregate_view("std_error"), "std_error_maps"),
                (
                    "Endpoint vs cumulative error",
                    lambda: plots.plot_endpoint_vs_cumulative(
                        session.dataset_dataframe(), styles=session.series_styles(), models=models, run_labels=run_labels
                    ),
                    "endpoint_cumulative_error",
                ),
            ],
            EVALUATION_SECTION_TITLES["global_error"],
        ),
        "error_decomposition": (
            [("Error vs |GT| magnitude", scoped_view("target_magnitude"), "error_vs_gt_magnitude")],
            EVALUATION_SECTION_TITLES["error_decomposition"],
        ),
        "sample_viewer": (
            [("Sample GT vs Prediction", sample_view(False), "sample_gt_vs_prediction")],
            EVALUATION_SECTION_TITLES["sample_viewer"],
        ),
        "outlier_analysis": (
            [
                ("Worst per-channel cases (tables)", outlier_table, "worst_per_channel_tables"),
                ("Worst per-channel cases (field plots)", sample_view(True), "worst_per_channel_fields"),
            ],
            EVALUATION_SECTION_TITLES["outlier_analysis"],
        ),
    }
    if comparison:
        registry["architecture"] = ([configuration_entry], EVALUATION_SECTION_TITLES["architecture"])
    registry["runtime"] = (
        [
            ("Boundary-specific duration ratios", _transient_runtime_entry(session, "ratios", frame_runs), "runtime_boundary_ratios"),
            ("Accuracy vs measured runtime", _transient_runtime_entry(session, "tradeoff", frame_runs), "accuracy_runtime_comparison"),
            ("Airflow substitution evidence", _transient_runtime_entry(session, "pipeline", frame_runs), "airflow_substitution"),
            ("Work per stage", lambda: run_summary.show_work_table(_transient_work_table(session, frame_runs)), "work_per_stage"),
            ("Break-even", partial(_transient_break_even, session, frame_runs), "break_even"),
        ],
        EVALUATION_SECTION_TITLES["runtime"],
    )
    registry["matched_stages"] = (
        [
            ("Training curves", partial(_transient_training_curves, session, training_histories, frame_runs), "training_curves"),
            (
                "Shared-A0 performance vs committed budget",
                partial(_transient_training_compute, session, training_performance, training_lineage, frame_runs),
                "matched_stage_compute",
            )
            if training_performance is not None
            else (
                "Controlled stage availability",
                partial(_transient_matched_stage_status, matched_stage_unavailable_reason),
                "matched_stage_availability",
            ),
        ],
        EVALUATION_SECTION_TITLES["matched_stages"],
    )
    has_full, has_horizon, has_target = _transient_temporal_capabilities(session)
    temporal_entries: list[tuple[str, Callable[..., Any], str]] = []
    if has_full:
        temporal_entries.extend(
            (
                ("Reference vs prediction trajectories", scoped_view("trajectory"), "reference_prediction_trajectories"),
                ("Error over drying phase and hours", scoped_view("error_time"), "error_over_physical_time"),
                (
                    "Relative norms over drying phase and hours",
                    _transient_relative_norm_entry(
                        session, comparison=comparison, selection_state=selection_state, refused_roles=refused_roles, frame_runs=frame_runs
                    ),
                    "relative_norms_over_physical_time",
                ),
            )
        )
    if has_horizon:
        temporal_entries.append(
            (
                "Error vs rollout horizon",
                lambda: plots.plot_horizon_error(session.dataset_dataframe(), styles=session.series_styles(), models=models, run_labels=run_labels),
                "error_vs_rollout_horizon",
            )
        )
    if has_target:
        temporal_entries.append(
            (
                "Final-state drying summary",
                lambda: plots.plot_target_time(
                    {name: session.full_autonomous_summaries(name) for name in session.frame_names},
                    styles=session.series_styles(),
                    models=models,
                    run_labels=run_labels,
                ),
                "final_state_drying_summary",
            )
        )
    if temporal_entries:
        registry["temporal_rollout"] = (
            temporal_entries,
            EVALUATION_SECTION_TITLES["temporal_rollout"],
        )
    return registry


def build_transient_panel(
    *,
    session: TransientEvaluationSession,
    sections: Sequence[str] | str = "all",
    comparison: bool = False,
    training_performance: Callable[[], pd.DataFrame | None] | None = None,
    training_lineage: LineageComparison | None = None,
    training_histories: Callable[[], Mapping[str, StageHistory]],
    matched_stage_unavailable_reason: str | None = None,
    selection_state: EvaluationSelectionState | None = None,
    evaluation_compute_device: str | compute.AnalysisComputePreference = "cpu",
    widths: WidthScope | None = None,
    refused_roles: Mapping[str, str] | None = None,
    frame_runs: Mapping[str, FrameRun] | None = None,
) -> widgets.Widget:
    """
    Build the baseline Evaluation architecture with transient capabilities; a refused role is listed as unavailable with its reason.

    ``frame_runs`` maps every frame of the session to its run: the display label that ``training_histories`` is keyed by, and whether
    that label is an explicit short label (``FNO · B · seed 9``). Only explicit labels name the runs in titles, as a suffix of one run or
    «A vs B» of two; a storage alias names none. With several runs the training curves take each run's colour from the session.
    Only the workflow, which names the frames, knows how a frame name is built, so it supplies this mapping.
    """
    if not session.frame_names:
        msg = "Transient panel requires one open session with frames."
        raise ValueError(msg)
    if frame_runs is not None and (set(frame_runs) != set(session.frame_names) or any(not run.label.strip() for run in frame_runs.values())):
        msg = "Frame runs must name every frame of the session once, each with a non-blank label."
        raise ValueError(msg)
    if comparison:
        session.require_common_comparison_grids()
    registry = _build_transient_sections(
        session=session,
        comparison=comparison,
        selection_state=selection_state,
        evaluation_compute_device=evaluation_compute_device,
        training_performance=training_performance,
        training_lineage=training_lineage,
        training_histories=training_histories,
        matched_stage_unavailable_reason=matched_stage_unavailable_reason,
        refused_roles=refused_roles,
        frame_runs=frame_runs,
    )
    selected = _selected_section_keys(
        sections,
        registry=registry,
        order=TRANSIENT_COMPARISON_SECTION_KEYS if comparison else TRANSIENT_SINGLE_SECTION_KEYS,
    )
    if not selected:
        msg = "The selected transient artifacts provide no approved Evaluation sections."
        raise ValueError(msg)
    export_state = {
        "fig": None,
        "figures": (),
        "plot_name": None,
        "title": None,
        "filename_stem": None,
        "filename_prefix": "transient_drying",
    }
    return _assemble_panel(registry, selected, export_state=export_state, comparison=comparison, widths=widths)
