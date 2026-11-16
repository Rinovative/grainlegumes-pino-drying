"""Compose curated evaluation panels over admitted artifacts.

Responsibilities:
  - Keep the approved view order explicit per task and per single-model or comparison mode
  - Number the displayed tabs and views from their final position
  - Bind every view to its defined local controls and defaults
  - Accept caller-provided context labels and session-owned canonical frames
  - Offer the datasets (material and role) of one panel in each view's own selector, kept in step through the selection state
  - List a comparison role the workflow refused, or a material without shared cases, as unavailable with its reason
  - Preserve lazy first rendering and panel-local figure export
  - Close a replaced panel together with every view it rendered
  - Export the selected transient sample case as a GIF over all stored times
  - Keep the transient sample viewer's state-map norms over all stored times for its last case and setting while any
    field is fixed over time
  - Offer the break-even against COMSOL from each run's timing and training evidence, naming a run without it, the
    transient training curves from a lazy history loader, and the error over time in hours or as τ = t / t_stop

Design principles:
  - Strict artifact loading and EvaluationSession binding precede construction
  - Dataset, model, channel, case, and parameter controls remain view-specific
  - Presentation labels never alter canonical identity or numerical cache keys
  - Panel construction opens no case payloads and computes no plot mathematics
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
    from collections.abc import Callable, Hashable, Iterable
    from typing import IO

    from matplotlib.figure import Figure

    from src.analysis.evaluation.evaluation_run_discovery import StageHistory, UnavailableHistory
    from src.analysis.evaluation.evaluation_selection import EvaluationSelectionState
    from src.analysis.evaluation.evaluation_session import EvaluationSession
    from src.analysis.evaluation.evaluation_transient_comparison import LineageComparison
    from src.analysis.evaluation.evaluation_transient_session import TransientEvaluationSession
    from src.analysis.presentation.analysis_dataset_selection import DatasetChoice, DatasetInventory
    from src.analysis.ui.analysis_ui_notebook import LazyPanel, ViewSection, WidthScope
    from src.analysis.ui.analysis_ui_panel import ViewPanel
    from src.analysis.ui.analysis_ui_viewers import ViewControl

_MINIMUM_COMPARISON_RUNS = 2
_MINIMUM_VELOCITY_COMPONENTS = 2
_FRAME_MATERIAL_CONTEXT_SIZE = 2
_OUTLIER_COUNT = 5
_CASE_COUNT_STEP = 50
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
_POOL_NOTE: Final = "Zusammenfassen gilt in dieser Ansicht nicht: die Datensätze erscheinen getrennt."
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

    ``frames_by_role`` maps a role to its models' frames; ``run_frames`` are the frames of the run-level views. A single-model
    panel draws one cut per selected dataset; a comparison draws every selected dataset, cut for every model, keyed by model
    label for one dataset and ``model · dataset`` for several. Every cut is a ``DisplayFrames`` whose ``labels`` give each key's
    display dataset and model. A cut is made once per choice and shared by the views.
    """

    def __init__(self, contexts: Sequence[EvaluationContext], *, comparison: bool, refused_roles: Mapping[str, str]) -> None:
        """Build the scope from the role contexts of a steady panel."""
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
            self._cuts[key] = self._cut(available)
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
                for model, frame in by_model.items():
                    key = f"{model} · {label}"
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
    single_dataset: bool = False,
    plot_kwargs: Mapping[str, Any] | None = None,
    count: tuple[int, int] | None = None,
    controls: Mapping[str, ViewControl] | None = None,
) -> ViewPanel:
    """
    Build one steady view over the datasets its selector offers.

    A view offers several datasets, or one for a ``single_dataset`` view (maps) or a ``single_model`` view; in a comparison a
    ``single_model`` view adds a model choice, and the other views draw the compared models as series of every selected
    dataset. ``pooled`` allows the pooling switch where the view's reduction is
    material-balanced. The plot is drawn directly, or with ``count`` (start, step) as a case-count viewer built inside
    this view for the selected datasets, at the count the former one showed.
    """
    ui = analysis.ui
    choice = _SharedChoice(
        scope.inventory,
        multiple=not (single_model or single_dataset),
        pool_reason=None if pooled and not scope.comparison else dataset_cuts.POOL_DISABLED_REASON,
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
    ``offered(rows)`` positions that ``selection_name(index, offered)`` names in the export stem. With ``sync`` the case,
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
    navigator = ui.navigation.CaseNavigator(positions(), status=status, on_change=lambda _position: changed())
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
        navigator.rebind(positions(), preserve=True, notify=False)
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
                    navigator.rebind(positions(), preserve=True, notify=False)
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


type _BreakEvenEvidence = tuple[bool, Mapping[str, float], Mapping[str, float], float | None, str]


def _break_even_view(
    frames: Mapping[str, pd.DataFrame],
    evidence: Callable[[str, pd.DataFrame, Mapping[str, Any], Mapping[str, Any]], _BreakEvenEvidence],
) -> Figure | widgets.HTML:
    """
    Draw the break-even against COMSOL of every run of ``frames`` that has its evidence, and name every run that lacks it.

    A run is its first frame, as in the work table. ``evidence`` (given the label, frame, summary and work row) gives the
    physics flag, the per-case COMSOL and surrogate seconds (the clocks of the speed-up view), the training seconds spent
    before the run's own (``0.0``, or ``None`` when they cannot be told) and the text naming clocks and hardware. A run without
    model kind, training seconds, train cases or a case with both clocks is named below the figure, or in a message when no
    run has its evidence.
    """
    plots = analysis.evaluation.plots
    costs: list[Any] = []
    unavailable: list[str] = []
    for label, summary in analysis.evaluation.run_discovery.distinct_runs(frames).items():
        work = analysis.evaluation.run_discovery.stage_work_row(summary)
        physics, comsol, surrogate, prior_seconds, clocks = evidence(label, frames[label], summary, work)
        present = {
            "model kind": work["model_kind"] is not None,
            "training seconds": work["total_seconds"] is not None and prior_seconds is not None,
            "train cases": work["train_cases"] is not None,
            "a case with both clocks": bool(comsol.keys() & surrogate.keys()),
        }
        if missing := [name for name, available in present.items() if not available]:
            unavailable.append(f"{label} (no {', '.join(missing)})")
            continue
        entity = plots.series.SeriesEntity(label, work["model_kind"], physics, "id", (), label)
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
    figure = plots.training.plot_break_even(costs)
    if unavailable:
        figure.supxlabel("Without break-even evidence: " + "; ".join(unavailable), fontsize=analysis.ui.figure_layout.NOTE_PT)
    return figure


def _steady_break_even(*, datasets: Mapping[str, pd.DataFrame]) -> Figure | widgets.HTML:
    """Show the break-even of the steady runs with the clocks of the solver speed-up: COMSOL solver time against the batch-amortized forward."""
    component = analysis.artifacts.timing.STEADY_SPEEDUPS["solver_speedup"][0]

    def evidence(_label: str, frame: pd.DataFrame, _summary: Mapping[str, Any], _work: Mapping[str, Any]) -> _BreakEvenEvidence:
        physics = analysis.evaluation.plots.series.model_metadata(frame)[2]
        report = analysis.evaluation.dataframe.require_complete_provenance(frame)["timing"]["report"]
        comsol = {case["case_id"]: case[component] for case in report["cases"] if case[component] is not None}
        surrogate = {case["case_id"]: case["batch_amortized_neural_operator_forward_s"] for case in report["cases"]}
        surrogate_hardware, solver_hardware = analysis.evaluation.plots.run_summary.runtime_hardware(report)
        clocks = f"COMSOL solver time ({component}) on {solver_hardware}; surrogate batch-amortized forward on {surrogate_hardware}"
        return physics, comsol, surrogate, 0.0, clocks

    return _break_even_view(datasets, evidence)


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

    def dataset(
        title: str,
        plot_func: Callable[..., Any],
        *,
        view: str | None = None,
        single_model: bool = False,
        single_dataset: bool = False,
        start: int | None = None,
        extra: Mapping[str, ViewControl] | None = None,
        **plot_kwargs: Any,
    ) -> tuple[str, Callable[..., Any], str]:
        """Return a plot of the selected datasets, as a case-count viewer from ``start`` with its own controls ``extra``.

        ``view`` names one of ``POOLABLE_STEADY_VIEWS``, whose views stay material-balanced when pooled.
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
                single_dataset=single_dataset,
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

    configuration_entry = toggle(
        "Model-family configuration",
        plots.run_summary.plot_architecture_table,
        plot_name="Model-family configuration",
    )
    overview_entries = [dataset("Summary table", plots.run_summary.plot_run_summary_table, view="overview_summary_table")]
    if comparison:
        overview_entries.extend(
            (
                dataset(
                    "Global comparison summary",
                    plots.run_summary.plot_relative_comparison_scoreboard,
                    view="overview_global_comparison_summary",
                ),
                dataset("Pareto (Error vs Physics)", plots.run_summary.plot_accuracy_physics_pareto, view="overview_accuracy_physics_pareto"),
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
            single_dataset=True,
            start=100,
            extra={"error_mode": controls.error_mode_group("steady"), "display_spatial_stride": controls.display_stride_dropdown()},
        ),
        dataset(
            "Std error maps",
            plots.error_behavior.plot_std_error_maps,
            single_dataset=True,
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
            dataset("Darcy–Brinkman operator residual", physical.plot_brinkman_residual, single_dataset=True, start=100),  # noqa: RUF001
            dataset("Pressure and boundary-condition consistency", physical.plot_pressure_consistency, start=100),
            dataset("Pressure-boundary summary", physical.plot_pressure_boundary_summary, start=100),
            dataset("Spatial residuals", physical.plot_spatial_residuals, single_dataset=True, start=100),
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
                toggle("Work per stage", plots.run_summary.plot_work_table, plot_name="Work per stage"),
                toggle("Break-even", _steady_break_even, plot_name="Break-even"),
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
) -> widgets.Widget:
    """Build one lazy panel whose views offer the datasets of every context in their own selectors."""
    registry = _build_sections(
        _SteadyScope(contexts, comparison=comparison, refused_roles=refused_roles or {}),
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
) -> widgets.Widget:
    """Build one single-model panel containing every supplied dataset."""
    normalized = _normalize_contexts(session, contexts, comparison=False)
    model_labels = tuple(next(iter(context.datasets)) for context in normalized)
    if len(set(model_labels)) != 1:
        msg = "Single-model contexts must all refer to the same presentation model label."
        raise ValueError(msg)
    return _build_panel(contexts=normalized, comparison=False, sections=sections, selection_state=selection_state, widths=widths)


def build_comparison_panel(
    *,
    session: EvaluationSession,
    contexts: Sequence[EvaluationContext],
    sections: Sequence[str] | str = "all",
    selection_state: EvaluationSelectionState | None = None,
    widths: WidthScope | None = None,
    refused_roles: Mapping[str, str] | None = None,
) -> widgets.Widget:
    """Build one comparison panel; its views choose among the datasets of every role, the refused roles listed as unavailable."""
    normalized = _normalize_contexts(session, contexts, comparison=True)
    return _build_panel(
        contexts=normalized,
        comparison=True,
        sections=sections,
        selection_state=selection_state,
        widths=widths,
        refused_roles=refused_roles,
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
    ) -> None:
        """Create the dataset choice and the navigator on the shared case, else the first one; ``on_change`` follows the navigator."""
        self._session = session
        self._entries = entries
        self._comparison = comparison
        self._state = selection_state
        self.choice = _SharedChoice(inventory, multiple=multiple, pool_reason=None, selection_state=selection_state)
        self.shown = _focus_entries(session, entries, self.choice.selector.choices())
        self.navigator = analysis.ui.navigation.CaseNavigator(
            self._positions(),
            value=self._position_of(None if selection_state is None else selection_state.selection.case_id) or 1,
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

    def rebind(self) -> bool:
        """List the cases of the chosen datasets again, keeping the shown position where it exists; return whether there are any."""
        self.shown = _focus_entries(self._session, self._entries, self.choice.selector.choices())
        if self.shown:
            self.navigator.rebind(self._positions(), preserve=True, notify=False, labels=_case_labels(self.shown, comparison=self._comparison))
        return bool(self.shown)

    def pull(self) -> None:
        """Take the datasets and the case the other views chose; only the navigator's own change callback reports a new case."""
        if self.choice.pull():
            self.rebind()
        wanted = self._position_of(None if self._state is None else self._state.selection.case_id)
        if wanted is not None:
            self.navigator.field.value = wanted


def _transient_outlier_inventory(
    session: TransientEvaluationSession,
    *,
    fields: Sequence[Any],
    comparison: bool,
    selection_state: EvaluationSelectionState | None,
) -> tuple[Any, ...]:
    """Return the stable union of per-channel worst complete-rollout cases."""
    inventory = _transient_inventory(
        session,
        comparison=comparison,
        selection_state=selection_state,
    )
    by_key = {(entry.frame_name, entry.case_id): entry for entry in inventory}
    frame = session.case_dataframe(modes=("autonomous_full",))
    selected = frame.loc[(frame["requested_horizon"] == "full") & (frame["scope"] == "cumulative")]
    ordered: list[Any] = []
    seen: set[tuple[str, str]] = set()
    for field in fields:
        column = f"normalized_rmse_{field.key}" if field.stored else "physical_w_gr_rmse"
        if column not in selected:
            continue
        for row in selected.nlargest(_OUTLIER_COUNT, column).itertuples(index=False):
            key = (str(row.frame), str(row.case_id))
            if key in by_key and key not in seen:
                ordered.append(by_key[key])
                seen.add(key)
    return tuple(ordered) or inventory


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


def _transient_model_label(session: TransientEvaluationSession, frame_name: str) -> str:
    """Return the short model label «architecture · stage» from the saved run config, the frame name where it is incomplete."""
    config = session.canonical_frames[frame_name].attrs.get(analysis.evaluation.dataframe.COMPLETED_RUN_CONFIG_ATTR)
    if not isinstance(config, Mapping):
        return frame_name
    try:
        label = naming.architecture_label(config)
        stage = config.get("run", {}).get("layout", {}).get("stage")
    except (AttributeError, KeyError, TypeError, ValueError):
        return frame_name
    return label if stage is None else f"{label} · {stage}"


def _transient_sample_title(entry: Any, *, model_label: str | None) -> str:
    """Return the state-map title context like the case selector: material and role, the short model label when comparing, the compact case."""
    parts = (_transient_material_role(entry), model_label, entry.case_label)
    return " · ".join(part for part in parts if part is not None)


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


def _shared_or_last_time(selection_state: EvaluationSelectionState | None, times: Sequence[float]) -> float:
    """Return the physical time the other views chose where ``times`` stores it, else the last stored time."""
    preferred = None if selection_state is None else selection_state.selection.physical_time
    return times[-1] if preferred is None or preferred not in times else preferred


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
) -> ViewPanel:
    """Build the EDA-native selected-case Reference/Prediction/Error view of one dataset (the focus dataset)."""
    ui = analysis.ui
    admitted_fields = _supported_transient_fields(fields)
    if not admitted_fields:
        msg = "Transient sample views require representable channels."
        raise ValueError(msg)
    entries = (
        _transient_outlier_inventory(session, fields=admitted_fields, comparison=comparison, selection_state=selection_state)
        if outlier
        else _transient_inventory(session, comparison=comparison, selection_state=selection_state)
    )
    export: dict[str, Any] = {"export_state": export_state, "export_plot_name": export_plot_name, "export_title": export_title}
    status = ui.controls.StatusLine()
    view: ViewPanel | None = None
    focus = _CaseFocus(
        session,
        entries,
        inventory=dataset_cuts.transient_inventory(session, refused=refused_roles or {}),
        multiple=False,
        comparison=comparison,
        selection_state=selection_state,
        status=status,
        on_change=lambda: cast("ViewPanel", view).request(refresh),
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
            value=_shared_or_last_time(selection_state, times),
            status=status,
            on_change=lambda _selection: cast("ViewPanel", view).request(render),
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
            "title_context": _transient_sample_title(entry, model_label=_transient_model_label(session, entry.frame_name) if comparison else None),
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
            clock.rebind(times, preserve_time=_shared_or_last_time(selection_state, times), notify=False)
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
    owners: dict[int, str] | None = None,
) -> dict[str, tuple[Any, ...]]:
    """
    Return the full-rollout records of the first ``limit`` cases of every dataset, one series per dataset and model frame.

    A comparison names each series by its model frame after the dataset label; a single model has one frame per dataset.
    ``owners``, when given, receives the frame name of every returned record under ``id(record)``.
    """
    series: dict[str, tuple[Any, ...]] = {}
    for context in contexts:
        for frame_name in context.frame_names:
            held = set(session.case_ids(frame_name))
            label = f"{context.label} · {frame_name}" if comparison else context.label
            series[label] = tuple(
                _full_autonomous_record(session, frame_name=frame_name, case_id=case_id)
                for case_id in [case_id for case_id in context.case_ids if case_id in held][:limit]
            )
            if owners is not None:
                owners.update((id(record), frame_name) for record in series[label])
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
) -> ViewPanel:
    """Build one exact-time multi-dataset aggregate with EDA-native controls."""
    if plot_kind not in {"predicted_reference", "mean_error", "std_error"}:
        msg = "Unknown transient aggregate field-view kind."
        raise ValueError(msg)
    ui = analysis.ui
    admitted_fields = _supported_transient_fields(fields)
    export: dict[str, Any] = {"export_state": export_state, "export_plot_name": export_plot_name, "export_title": export_title}
    choice = _SharedChoice(
        dataset_cuts.transient_inventory(session, refused=refused_roles or {}),
        multiple=not comparison,  # a comparison draws one dataset (O1)
        pool_reason=None,
        selection_state=selection_state,
    )
    maximum = _largest_dataset(session, choice)
    all_case_ids = [entry.case_id for entry in session.case_inventory()]
    status = ui.controls.StatusLine()
    channels = _Channels("Channels", admitted_fields, selection_state)
    active: dict[str, Any] = {"key": None, "series": None, "owners": {}}  # owners: the frame each shown record comes from
    view: ViewPanel | None = None

    def series() -> dict[str, tuple[Any, ...]]:
        contexts = dataset_cuts.transient_contexts(session, choice.selector.choices())
        key = (tuple((context.label, context.frame_names) for context in contexts), count.value)
        if active["key"] != key:
            active["key"], active["owners"] = key, {}
            active["series"] = _transient_series(session, contexts, limit=count.value, comparison=comparison, owners=active["owners"])
        return cast("dict[str, tuple[Any, ...]]", active["series"])

    def times_of(shown: dict[str, tuple[Any, ...]]) -> tuple[float, ...]:
        return _shared_transient_physical_times(tuple(record for records in shown.values() for record in records))

    count = ui.navigation.CountControl(
        value=min(_CASE_COUNT_STEP, maximum),
        minimum=1,
        maximum=maximum,
        step=_CASE_COUNT_STEP,
        on_change=lambda _count: cast("ViewPanel", view).request(refresh),
    )
    first = times_of(series())
    clock = ui.navigation.TimeNavigator(
        first,
        value=_shared_or_last_time(selection_state, first),
        status=status,
        on_change=lambda _selection: cast("ViewPanel", view).request(render),
    )
    error_mode = ui.controls.error_mode_group("transient") if plot_kind == "mean_error" else None  # the std maps keep one meaning
    rows = [
        ui.panel.row(choice.selector.header, count.widget),
        ui.panel.row(channels.checks.widget),
        ui.panel.row(clock.widget),
        *([ui.panel.row(error_mode)] if error_mode is not None else []),
    ]

    def render_now() -> None:
        current = cast("ViewPanel", view)
        selected = channels.checks.selected
        shown = series()
        if not shown:
            ui.viewers.clear_view(current, ui.viewers.EMPTY_SELECTION_MESSAGE, **export)
            return
        if not selected:
            ui.viewers.clear_view(current, "Select at least one compatible channel.", **export)
            return
        physical_time = clock.selection.physical_time
        supported = {label: tuple(record for record in records if physical_time in record.physical_times) for label, records in shown.items()}
        coverage = f"Exact t={physical_time:g} h support · " + "; ".join(f"{label}: n(t)={len(records)}" for label, records in supported.items())
        if selection_state is not None:
            _publish_transient_capabilities(
                selection_state, admitted_fields, all_case_ids, physical_times=clock.times_h, channels=selected, physical_time=physical_time
            )
        plot_func: Callable[..., Figure]
        if plot_kind == "predicted_reference":
            plot_func = analysis.evaluation.plots.transient.plot_predicted_vs_reference_channels
            kwargs: dict[str, Any] = {"state_fields": selected, "physical_time": physical_time, "styles": session.series_styles()}
        else:
            plot_func = analysis.evaluation.plots.transient.plot_aggregate_error_maps
            preference = evaluation_compute_device
            requested_device = preference.requested if isinstance(preference, compute.AnalysisComputePreference) else preference
            kwargs = {
                "state_fields": selected,
                "physical_time": physical_time,
                "statistic": "mean" if plot_kind == "mean_error" else "std",
                "compute_device": requested_device,
            }
            kwargs |= _series_reference_scales(session, active["owners"], supported, selected, error_mode)

        def draw() -> tuple[str, Figure]:
            return coverage, plot_func(supported, **kwargs)

        if ui.viewers.render_view(current, draw, **export) and export_state is not None:
            export_state["filename_stem"] = "__".join(
                (
                    "transient_drying",
                    str(export_plot_name or plot_kind),
                    f"cases_{count.value}",
                    f"channel_{'-'.join(selected)}",
                    f"t_{physical_time:g}h",
                    "aggregate",
                )
            )

    def refresh_now() -> None:
        """Keep the shared time where the changed cases store it, and draw once."""
        shown = series()
        if shown:
            times = times_of(shown)
            clock.rebind(times, preserve_time=_shared_or_last_time(selection_state, times), notify=False)
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
) -> ViewPanel:
    """Build one local Aggregate/Single case trajectory-style view of the selected datasets."""
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
        multiple=not comparison,  # a comparison draws one dataset (O1)
        comparison=comparison,
        selection_state=selection_state,
        status=status,
        on_change=lambda: cast("ViewPanel", view).request(render),
    )
    choice, navigator = focus.choice, focus.navigator
    maximum = _largest_dataset(session, choice)
    channels = _Channels("Stored channels" if plot_kind == "error_time" else "Channels", admitted_fields, selection_state)
    scope = ui.controls.scope_group(selection_state.selection.scope if selection_state is not None else "aggregate")
    active: dict[str, Any] = {"key": None, "series": None, "scaling_states": None}

    def selected_series() -> tuple[dict[str, tuple[Any, ...]], dict[str, Mapping[str, Any]]]:
        contexts = dataset_cuts.transient_contexts(session, choice.selector.choices())
        entry = focus.entry if scope.value == "single" else None
        if entry is not None:
            key: tuple[Any, ...] = ("single", entry.frame_name, entry.case_id)
        else:
            key = ("aggregate", tuple((context.label, context.frame_names) for context in contexts), count.value)
        if active["key"] != key:
            if entry is not None:
                label = _transient_case_label(entry, comparison=comparison)
                series = {label: (_full_autonomous_record(session, frame_name=entry.frame_name, case_id=entry.case_id),)}
                scaling_states = {label: session.scaling_state(entry.frame_name)}
            else:
                series = _transient_series(session, contexts, limit=count.value, comparison=comparison)
                scaling_states = {
                    f"{context.label} · {name}" if comparison else context.label: session.scaling_state(name)
                    for context in contexts
                    for name in context.frame_names
                }
            active["key"], active["series"], active["scaling_states"] = key, series, scaling_states
        return cast("dict[str, tuple[Any, ...]]", active["series"]), cast("dict[str, Mapping[str, Any]]", active["scaling_states"])

    count = ui.navigation.CountControl(
        value=min(100, maximum),
        minimum=1,
        maximum=maximum,
        step=_CASE_COUNT_STEP,
        on_change=lambda _count: cast("ViewPanel", view).request(render),
    )
    time_axis = ui.controls.choice_group((("Hours", "hours"), ("τ = t / t_stop", "tau"))) if plot_kind == "error_time" else None
    rows = [
        ui.panel.row(choice.selector.header, scope, count.widget, navigator.widget),
        ui.panel.row(channels.checks.widget, *([time_axis] if time_axis else [])),
    ]

    def update_scope_row() -> None:
        """Expose only the navigator of the active scope: the case count (aggregate) or the case (single)."""
        ui.controls.set_visible(count.widget, scope.value == "aggregate")
        ui.controls.set_visible(navigator.widget, scope.value == "single")

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
        plot_func: Callable[..., Figure]
        if plot_kind == "trajectory":
            plot_func = analysis.evaluation.plots.transient.plot_state_trajectory_summary
            kwargs: dict[str, Any] = {"state_fields": selected, "styles": session.series_styles()}
        elif plot_kind == "error_time":
            plot_func = analysis.evaluation.plots.transient.plot_error_over_physical_time
            kwargs = {
                "scaling_states": scaling_states,
                "state_fields": selected,
                "styles": session.series_styles(),
                "time_axis": cast("widgets.ToggleButtons", time_axis).value,
            }
        else:
            plot_func = analysis.evaluation.plots.transient.plot_error_vs_target_magnitude
            kwargs = {"state_fields": selected, "styles": session.series_styles()}

        def draw() -> Figure:
            return plot_func(series, **kwargs)

        if ui.viewers.render_view(current, draw, **export) and export_state is not None:
            case_part = f"case_{entry.case_id}" if entry is not None else f"cases_{count.value}"
            export_state["filename_stem"] = "__".join(
                ("transient_drying", str(export_plot_name or plot_kind), str(scope.value), case_part, f"channel_{'-'.join(selected)}")
                + ((str(time_axis.value),) if time_axis else ())
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
    if time_axis is not None:
        time_axis.observe(lambda _change: cast("ViewPanel", view).request(render), names="value")
    update_scope_row()
    view.request(render)
    return view


def _transient_summary_table(session: TransientEvaluationSession) -> pd.DataFrame:
    """
    Return a compact aggregate table without exposing identity hashes.

    A missing value shows its counters, as the steady overview does: the support counts of a
    group with non-finite predictions, or the cases below the reference floor of a relative norm.
    """
    frame = session.dataset_dataframe()
    identity = ("frame", "material_family", "dataset_role", "mode", "requested_horizon", "scope", "contributing_record_count")
    relative = ("physical_relative_l2", "physical_relative_h1")
    errors = ("normalized_drying_group_macro_rmse", *(f"normalized_rmse_{field}" for field in transient_artifact.STATE_ORDER), *relative)

    def shown(row: Mapping[Any, Any], name: str) -> object:
        value = row[name]
        if value is not None and not pd.isna(value):
            return value
        if row["contributing_case_count"] == 0:
            missing = row["unavailable_case_count"]
            return f"unavailable (no cases at this horizon; {missing} missing)" if missing else "unavailable (no cases at this horizon)"
        if row["finite_value_count"] < row["required_value_count"]:
            return validity_contract.unavailable_text(row)
        return f"unavailable ({row[f'{name}_unavailable_cases']} case-field values below the reference floor)"

    rows = [
        {**{name: row[name] for name in identity if name in row}, **{name: shown(row, name) for name in errors}} for row in frame.to_dict("records")
    ]
    return pd.DataFrame.from_records(rows)


def _transient_comparison_summary(session: TransientEvaluationSession) -> pd.DataFrame:
    """Return one ranked full-rollout model comparison scoreboard whose error columns are the session's selection errors."""
    frame = session.dataset_dataframe(modes=("autonomous_full",))
    selected = frame.loc[(frame["requested_horizon"] == "full") & (frame["scope"] == "cumulative")]
    ranked = pd.DataFrame.from_records([{"frame": name, **session.selection_errors(name)} for name in dict.fromkeys(selected["frame"])])
    ranked = ranked.sort_values("normalized_drying_group_macro_rmse", na_position="last").reset_index(drop=True)
    ranked["Ranking criterion"] = (
        "Selection objective of the cumulative full rollouts: case errors equal over the cases of a material and over the materials (lower is better)"
    )
    ranked["Material count"] = ranked["frame"].map(selected.groupby("frame")["material_family"].nunique())
    ranked["Metric available"] = ranked["normalized_drying_group_macro_rmse"].map(lambda value: pd.notna(value) and isfinite(float(value)))
    return ranked


def _transient_work_table(session: TransientEvaluationSession) -> pd.DataFrame:
    """Return one work row per selected run, named by its first frame, with that artifact's drying-rollout model seconds per case."""
    run_summary = analysis.evaluation.plots.run_summary
    return run_summary.build_frame_work_table(
        session.canonical_frames, lambda frame_name, _frame: run_summary.transient_inference_seconds(session.timing_report(frame_name))
    )


def _transient_architecture_table(session: TransientEvaluationSession) -> pd.DataFrame:
    """Return exact model-family configuration from compact record identity."""
    rows = []
    for frame_name in session.frame_names:
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
        rows.append(
            {
                "Artifact": frame_name,
                "Model family": identity.get("model_kind"),
                "Architecture label": architecture_label,
                "Stage": run.get("layout", {}).get("stage", identity.get("stage_identity", "Unavailable")),
                "Seed": run.get("seed", "Unavailable"),
                "Configuration": (dict(parameters) if isinstance(parameters, Mapping) else "Unavailable"),
            }
        )
    return pd.DataFrame.from_records(rows)


def _transient_outlier_table(
    session: TransientEvaluationSession,
    *,
    fields: Sequence[Any],
) -> pd.DataFrame:
    """Return per-channel worst complete-rollout cases from persisted metrics."""
    frame = session.case_dataframe(modes=("autonomous_full",))
    selected = frame.loc[(frame["requested_horizon"] == "full") & (frame["scope"] == "cumulative")]
    rows: list[dict[str, Any]] = []
    for field in fields:
        column = f"normalized_rmse_{field.key}" if field.stored else "physical_w_gr_rmse"
        if column not in selected:
            continue
        for rank, row in enumerate(
            selected.nlargest(_OUTLIER_COUNT, column).itertuples(index=False),
            start=1,
        ):
            rows.append(
                {
                    "Channel": field.label,
                    "Rank": rank,
                    "Material · Role": analysis.presentation.display_labels.material_role_display_label(
                        str(row.material_family),
                        str(row.declared_evaluation_role),
                    ),
                    "Artifact": str(row.frame),
                    "Exact case": str(row.case_id),
                    "Error": float(getattr(row, column)),
                    "Error unit": "1" if field.stored else field.unit,
                }
            )
    if not rows:
        msg = "Transient outlier tables require at least one per-channel metric."
        raise ValueError(msg)
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


def _make_transient_runtime_viewer(
    *,
    session: TransientEvaluationSession,
    kind: str,
    export_state: dict[str, Any] | None = None,
    export_plot_name: str | None = None,
    export_title: str | None = None,
) -> ViewPanel:
    """Display saved timing evidence with explicit model and measurement boundaries."""
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
    status = ui.controls.StatusLine(
        "Saved measurements only. Model calls exclude request preparation and scaling; "
        "prepared-request wall excludes Dataset/checkpoint loading and request construction. "
        "COMSOL reported solution and process times are separate boundaries. Duration ratios do not establish like-for-like end-to-end acceleration."
    )
    view = ui.panel.ViewPanel([ui.panel.row(measure if kind == "tradeoff" else model)], status=status, slot=ui.panel.PlotSlot())

    def render_now() -> None:
        kwargs: dict[str, Any] = {}
        args: tuple[Any, ...]
        function: Callable[..., Any]
        if kind == "tradeoff":
            frame = session.case_dataframe(modes=("autonomous_full",))
            frame = frame.loc[(frame["scope"] == "cumulative") & (frame["requested_horizon"] == "full")]
            function = plots.plot_accuracy_runtime_comparison
            args = (frame, {name: session.timing_report(name) for name in session.frame_names})
            kwargs["measure"] = measure.value
            kwargs["styles"] = session.series_styles()
        elif kind == "pipeline":
            values = session.pipeline_degradation()
            selected = values[session.frame_names.index(model.value)]
            function = plots.plot_pipeline_degradation
            args = ((selected,),)
            status.set(
                "Airflow conditions A/B/C are reference, drying-surrogate and full airflow-surrogate pipelines; "
                "they are separate from training stages A0/B/A1. "
                + (
                    "Full pipeline C prediction evidence is available."
                    if selected.c_available
                    else f"Full pipeline C unavailable: {selected.c_unavailable_reason}"
                )
            )
        else:
            function = plots.plot_timing_speedups
            args = (session.timing_report(str(model.value)),)
        published = ui.viewers.render_view(
            view,
            function,
            args=args,
            kwargs=kwargs,
            export_state=export_state,
            export_plot_name=export_plot_name,
            export_title=f"{export_title or kind} · {'all selected models' if kind == 'tradeoff' else model.value}",
        )
        if published and export_state is not None:
            context = "all_selected_models" if kind == "tradeoff" else str(model.value)
            export_state["filename_stem"] = f"transient_{context}_{export_plot_name or kind}_{measure.value if kind == 'tradeoff' else kind}"

    render = _guard(view, {"export_state": export_state, "export_plot_name": export_plot_name, "export_title": export_title}, render_now)
    model.observe(lambda _change: view.request(render), names="value")
    measure.observe(lambda _change: view.request(render), names="value")
    view.request(render)
    return view


def _transient_matched_stage_status(reason: str | None) -> widgets.HTML:
    """Explain missing controlled comparison evidence without blocking general plots."""
    text = reason or "Select exactly one admitted A0/B/A1 group with shared A0 and immutable B budget evidence."
    return widgets.HTML(f"<b>Controlled stage comparison unavailable.</b> {escape(text)} General prediction comparisons remain available.")


def _transient_training_compute(loader: Callable[[], pd.DataFrame | None], lineage: LineageComparison | None) -> Figure:
    """Plot only the performance evidence admitted by the workflow lineage validator, with every arm's end from it."""
    performance = loader()
    if performance is None or lineage is None:
        msg = "Admitted controlled stage comparison lacks performance evidence."
        raise RuntimeError(msg)
    return analysis.evaluation.plots.transient.plot_training_performance_vs_compute(performance, lineage=lineage)


def _transient_break_even(session: TransientEvaluationSession) -> Figure | widgets.HTML:
    """
    Show the break-even of the loaded stages with the clocks of the drying-only speed-up (COMSOL drying study, drying model calls).

    The training seconds of B and A1 are those of the A0 their teacher handoff records plus their own. That A0 is the one loaded
    stage whose selected checkpoint has the handoff's checkpoint digest (a run name is not an identity); a B or A1 without
    exactly one such A0 has no training seconds. Each clock is passed as recorded: the cost pairs the cases and counts the rest.
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
        device = ", ".join(sorted({f"{case.device} ({case.gpu or case.cpu})" for case in report.cases}))
        continued = work["stage"] in {"B", "A1"}
        prior: float | None = 0.0
        if continued:
            handoff = ((summary.get("teacher_handoff") or {}).get("files") or {}).get("checkpoint", {}).get("sha256")
            a0 = [elapsed for checkpoint, elapsed in a0_runs if handoff is not None and checkpoint == handoff]
            prior = a0[0] if len(a0) == 1 else None
        clocks = (
            f"COMSOL transient drying study time on {solver}; surrogate drying rollout model calls on {device}; "
            f"training seconds {'A0 + stage' if continued else 'of the stage'}"
        )
        return False, comsol, surrogate, prior, clocks

    return _break_even_view(session.canonical_frames, evidence)


def _transient_runtime_entry(session: TransientEvaluationSession, kind: str) -> Callable[..., ViewPanel]:
    """Bind one lazy runtime viewer to the existing figure export context."""

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
        )

    return build


def _build_transient_sections(
    *,
    session: TransientEvaluationSession,
    comparison: bool,
    selection_state: EvaluationSelectionState | None,
    evaluation_compute_device: str | compute.AnalysisComputePreference,
    training_performance: Callable[[], pd.DataFrame | None] | None = None,
    training_lineage: LineageComparison | None = None,
    training_histories: Callable[[], Mapping[str, StageHistory | UnavailableHistory]],
    matched_stage_unavailable_reason: str | None = None,
    refused_roles: Mapping[str, str] | None = None,
) -> dict[str, tuple[list[tuple[str, Callable[..., Any], str]], str]]:
    """Bind compatible transient evidence under the established Evaluation sections."""
    fields = _transient_channel_fields(session)
    supported_fields = _supported_transient_fields(fields)
    stored_fields = _stored_transient_fields(fields)
    plots = analysis.evaluation.plots.transient

    def table(title: str, build: Callable[[], pd.DataFrame]) -> Callable[[], widgets.VBox]:
        return lambda: analysis.ui.tables.styled_dataframe(build(), title=title)

    summary_table = table("Summary table", partial(_transient_summary_table, session))
    comparison_summary = table("Global comparison summary", partial(_transient_comparison_summary, session))

    def global_metrics() -> Figure:
        return plots.plot_channel_error(
            session.dataset_dataframe(),
            state_fields=tuple(field.key for field in stored_fields),
            styles=session.series_styles(),
        )

    def distributions() -> Figure:
        return plots.plot_error_distributions(
            session.case_dataframe(modes=("autonomous_full",)),
            state_fields=tuple(field.key for field in stored_fields),
            styles=session.series_styles(),
        )

    architecture_table = table("Model-family configuration", partial(_transient_architecture_table, session))
    outlier_table = table("Worst per-channel cases (tables)", partial(_transient_outlier_table, session, fields=supported_fields))

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
                    lambda: plots.plot_endpoint_vs_cumulative(session.dataset_dataframe(), styles=session.series_styles()),
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
            ("Boundary-specific duration ratios", _transient_runtime_entry(session, "ratios"), "runtime_boundary_ratios"),
            ("Accuracy vs measured runtime", _transient_runtime_entry(session, "tradeoff"), "accuracy_runtime_comparison"),
            ("Airflow substitution evidence", _transient_runtime_entry(session, "pipeline"), "airflow_substitution"),
            ("Work per stage", table("Work per stage", partial(_transient_work_table, session)), "work_per_stage"),
            ("Break-even", partial(_transient_break_even, session), "break_even"),
        ],
        EVALUATION_SECTION_TITLES["runtime"],
    )
    registry["matched_stages"] = (
        [
            ("Training curves", lambda: analysis.evaluation.plots.training.plot_training_curves(training_histories()), "training_curves"),
            (
                "Shared-A0 performance vs committed budget",
                partial(_transient_training_compute, training_performance, training_lineage),
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
                ("Error over physical time", scoped_view("error_time"), "error_over_physical_time"),
                (
                    "Relative norms over physical time",
                    lambda: plots.plot_relative_norms_over_time(
                        {name: session.relative_norm_curve(name) for name in session.frame_names}, styles=session.series_styles()
                    ),
                    "relative_norms_over_physical_time",
                ),
            )
        )
    if has_horizon:
        temporal_entries.append(
            (
                "Error vs rollout horizon",
                lambda: plots.plot_horizon_error(session.dataset_dataframe(), styles=session.series_styles()),
                "error_vs_rollout_horizon",
            )
        )
    if has_target:
        temporal_entries.append(
            (
                "Final-state drying summary",
                lambda: plots.plot_target_time(
                    {name: session.full_autonomous_summaries(name) for name in session.frame_names}, styles=session.series_styles()
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
    training_histories: Callable[[], Mapping[str, StageHistory | UnavailableHistory]],
    matched_stage_unavailable_reason: str | None = None,
    selection_state: EvaluationSelectionState | None = None,
    evaluation_compute_device: str | compute.AnalysisComputePreference = "cpu",
    widths: WidthScope | None = None,
    refused_roles: Mapping[str, str] | None = None,
) -> widgets.Widget:
    """Build the baseline Evaluation architecture with transient capabilities; a refused role is listed as unavailable with its reason."""
    if not session.frame_names:
        msg = "Transient panel requires one open session with frames."
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
