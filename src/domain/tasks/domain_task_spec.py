"""Define immutable semantic contracts for learned operator tasks.

Responsibilities:
  - Describe fields, datasets, preprocessing, metrics, reference floors, and physics selection
  - Declare the per-field metric family both tasks report
  - Validate task structure, exact field membership, and tensor axes
  - Produce deterministic resolved, data, and complete contract digests

Design principles:
  - Task-fixed semantics have one immutable source of truth
  - Physical units remain distinct from stored/model representations
  - Channel counts, metric identities, and contract digests are deterministic

This module does NOT:
  - Register tasks or declare a concrete task's fields and defaults
  - Implement metric, loss, normalization, or physics equations
  - Load datasets or mutate persisted task contracts after construction
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Final, Literal, get_args

# The one Train-only scaling of inputs and states in every task; the fitted normalizer and artifacts record it.
TRAIN_SCALING_KIND: Final = "case_equal_train_channel_scaling"
FieldRole = Literal[
    "airflow",
    "boundary",
    "coordinate",
    "material",
    "permeability",
    "porosity",
    "state",
]
MetricSpace = Literal["normalized", "physical"]
MetricReduction = Literal["material_case_mean"]
OptimizationDirection = Literal["minimize", "maximize"]
_SUPPORTED_TENSOR_LAYOUT = ("batch", "channel", "y", "x")
_SUPPORTED_OPERATOR_AXES = (2, 3)
_SUPPORTED_NORMALIZATION_AXES = (0, 2, 3)
_FIELD_ROLES = frozenset(get_args(FieldRole))
_METRIC_SPACES = frozenset(get_args(MetricSpace))
_METRIC_REDUCTIONS = frozenset(get_args(MetricReduction))
_OPTIMIZATION_DIRECTIONS = frozenset(get_args(OptimizationDirection))


def _canonical_sha256(payload: dict[str, object]) -> str:
    """Return the SHA-256 hex digest of the compact, key-sorted ASCII JSON encoding of ``payload``."""
    canonical = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """
    Describe one named tensor field.

    Attributes
    ----------
    name : str
        Canonical machine-readable field name.
    role : FieldRole
        Semantic role within the task.
    unit : str
        Physical unit before train-fitted normalization.
    representation : str
        Stored/model representation distinct from the physical unit.

    Raises
    ------
    ValueError
        If names, units, or representations are empty or the role is unsupported.

    """

    name: str
    role: FieldRole
    unit: str
    representation: str

    def __post_init__(self) -> None:
        """Reject empty names/units/representations and unsupported runtime roles."""
        for label, value in (("name", self.name), ("unit", self.unit), ("representation", self.representation)):
            if not isinstance(value, str) or not value:
                msg = f"Field {label} must be a non-empty string."
                raise ValueError(msg)
        if self.role not in _FIELD_ROLES:
            msg = f"Field {self.name!r} has unsupported role {self.role!r}."
            raise ValueError(msg)

    def as_dict(self) -> dict[str, str]:
        """
        Return a JSON-serializable field contract.

        Returns
        -------
        dict[str, str]
            Field name, role, unit, and representation.

        """
        return {
            "name": self.name,
            "role": self.role,
            "unit": self.unit,
            "representation": self.representation,
        }


@dataclass(frozen=True, slots=True)
class InputProfileSpec:
    """
    Describe one task-owned model input profile.

    Attributes
    ----------
    id : str
        Stable profile identifier persisted with runs and checkpoints.
    fields : tuple[str, ...]
        Exact ordered subset of the task input fields.
    coordinate_policy : str
        Exclusive coordinate representation used by every supported model.

    """

    id: str
    fields: tuple[str, ...]
    coordinate_policy: str

    def __post_init__(self) -> None:
        """Reject empty identifiers, duplicate fields, and implicit coordinates."""
        if not isinstance(self.id, str) or not self.id:
            msg = "Input profile id must be a non-empty string."
            raise ValueError(msg)
        if (
            not isinstance(self.fields, tuple)
            or not self.fields
            or any(not isinstance(field, str) or not field for field in self.fields)
            or len(set(self.fields)) != len(self.fields)
        ):
            msg = f"Input profile {self.id!r} fields must be a unique non-empty tuple of strings."
            raise ValueError(msg)
        if not isinstance(self.coordinate_policy, str) or not self.coordinate_policy:
            msg = f"Input profile {self.id!r} coordinate_policy must be a non-empty string."
            raise ValueError(msg)

    def as_dict(self) -> dict[str, object]:
        """Return the JSON-serializable profile identity."""
        return {
            "id": self.id,
            "fields": list(self.fields),
            "coordinate_policy": self.coordinate_policy,
        }


@dataclass(frozen=True, slots=True)
class PreprocessingSpec:
    """
    Describe task-owned preprocessing assumptions.

    Attributes
    ----------
    normalization : str
        The one scaling of inputs and states, shared by every task:
        ``TRAIN_SCALING_KIND``.
    fit_split : str
        Dataset split used to fit preprocessing statistics. Only ``"train"`` is
        supported so evaluation membership cannot influence fitted state.

    Raises
    ------
    ValueError
        If ``normalization`` is not ``TRAIN_SCALING_KIND`` or ``fit_split`` is
        not exactly ``"train"``.

    """

    normalization: str
    fit_split: str

    def __post_init__(self) -> None:
        """Require the shared Train-only scaling."""
        if self.normalization != TRAIN_SCALING_KIND:
            msg = f"Task normalization must be the shared {TRAIN_SCALING_KIND!r}, got {self.normalization!r}."
            raise ValueError(msg)
        if self.fit_split != "train":
            msg = f"Normalizer fit_split must be 'train', got {self.fit_split!r}."
            raise ValueError(msg)

    def as_dict(self) -> dict[str, str]:
        """
        Return a JSON-serializable preprocessing contract.

        Returns
        -------
        dict[str, str]
            Input/output normalization assumptions and fit split.

        """
        return {"normalization": self.normalization, "fit_split": self.fit_split}


@dataclass(frozen=True, slots=True)
class OutputGroupSpec:
    """
    Describe one physical output quantity represented by one or more fields.

    Attributes
    ----------
    id : str
        Canonical group identifier.
    fields : tuple[str, ...]
        Ordered output fields that jointly represent the physical quantity.
    vector : bool
        Whether the fields are the components of one vector, whose error is one
        Euclidean quantity; otherwise every field is a separate compartment.

    Raises
    ------
    ValueError
        If the identifier is empty or fields are empty or duplicated.

    """

    id: str
    fields: tuple[str, ...]
    vector: bool = False

    def __post_init__(self) -> None:
        """Reject empty identifiers and ambiguous field membership."""
        if not isinstance(self.id, str) or not self.id:
            msg = "Output group id must be a non-empty string."
            raise ValueError(msg)
        if (
            not isinstance(self.fields, tuple)
            or not self.fields
            or any(not isinstance(field, str) or not field for field in self.fields)
            or len(set(self.fields)) != len(self.fields)
        ):
            msg = f"Output group {self.id!r} fields must be a unique non-empty tuple of strings."
            raise ValueError(msg)
        if not isinstance(self.vector, bool):
            msg = f"Output group {self.id!r} vector must be a boolean."
            raise TypeError(msg)

    def as_dict(self) -> dict[str, object]:
        """
        Return a JSON-serializable output-group declaration.

        Returns
        -------
        dict[str, object]
            Group identifier, ordered member fields, and whether they form one vector.

        """
        return {"id": self.id, "fields": list(self.fields), "vector": self.vector}


@dataclass(frozen=True, slots=True)
class ErrorTermSpec:
    """
    Describe one weighted term of the training loss and of the selection objective.

    Attributes
    ----------
    id : str
        The field name of a one-field term, else the vector group's id.
    fields : tuple[str, ...]
        Fields whose squared errors the term sums before dividing by the sum of their squared Train scales.
    weight : float
        Positive term weight; the terms of a task sum to one.

    """

    id: str
    fields: tuple[str, ...]
    weight: float


def error_terms(groups: tuple[OutputGroupSpec, ...]) -> tuple[ErrorTermSpec, ...]:
    """
    Return the weighted error terms of the given physical output groups.

    Every group receives the equal weight ``1 / len(groups)``. A vector group is one
    term over all its components; any other group splits its weight equally over
    one term per field (steady: pressure 1/2, velocity vector 1/2; transient: T 1/3,
    phi 1/3, w_surf 1/6, w_int 1/6).

    Parameters
    ----------
    groups : tuple[OutputGroupSpec, ...]
        Non-empty ordered physical output groups.

    Returns
    -------
    tuple[ErrorTermSpec, ...]
        Terms in group and field order.

    Raises
    ------
    ValueError
        If no group is given.

    """
    if not groups:
        msg = "Error terms require at least one physical output group."
        raise ValueError(msg)
    group_weight = 1.0 / len(groups)
    terms: list[ErrorTermSpec] = []
    for group in groups:
        if group.vector or len(group.fields) == 1:
            terms.append(ErrorTermSpec(group.fields[0] if len(group.fields) == 1 else group.id, group.fields, group_weight))
        else:
            terms.extend(ErrorTermSpec(field, (field,), group_weight / len(group.fields)) for field in group.fields)
    return tuple(terms)


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """
    Describe one semantic evaluation metric selected by a task.

    Attributes
    ----------
    id : str
        Unique metric/reporting identifier within the task config.
    kind : str
        Semantic metric implementation identifier.
    space : MetricSpace
        Normalized or physical tensor space.
    fields : tuple[str, ...]
        Ordered task output fields evaluated by the metric.
    reduction : MetricReduction
        Aggregation contract. ``material_case_mean`` reduces every case over its
        grid and time first (squared or absolute errors, or one relative norm per
        case), then averages equally over the cases of a material and over the
        materials; RMSE values take their root only after that mean.
    direction : OptimizationDirection
        Objective optimization direction.

    Raises
    ------
    ValueError
        If identifiers or fields are empty/duplicated, or a runtime literal is
        outside the supported metric space, reduction, or direction sets.

    """

    id: str
    kind: str
    space: MetricSpace
    fields: tuple[str, ...]
    reduction: MetricReduction
    direction: OptimizationDirection

    def __post_init__(self) -> None:
        """Reject invalid runtime literals and ambiguous metric declarations."""
        if not isinstance(self.id, str) or not self.id or not isinstance(self.kind, str) or not self.kind:
            msg = "Metric id and kind must be non-empty strings."
            raise ValueError(msg)
        if self.space not in _METRIC_SPACES:
            msg = f"Metric {self.id!r} has unsupported space {self.space!r}."
            raise ValueError(msg)
        if not isinstance(self.fields, tuple) or not self.fields or any(not isinstance(field, str) or not field for field in self.fields):
            msg = f"Metric {self.id!r} fields must be a non-empty tuple of strings."
            raise ValueError(msg)
        if len(set(self.fields)) != len(self.fields):
            msg = f"Metric {self.id!r} contains duplicate fields."
            raise ValueError(msg)
        if self.reduction not in _METRIC_REDUCTIONS:
            msg = f"Metric {self.id!r} has unsupported reduction {self.reduction!r}."
            raise ValueError(msg)
        if self.direction not in _OPTIMIZATION_DIRECTIONS:
            msg = f"Metric {self.id!r} has unsupported direction {self.direction!r}."
            raise ValueError(msg)

    def as_dict(self, *, all_fields: tuple[str, ...]) -> dict[str, object]:
        """
        Return a JSON-serializable semantic metric declaration.

        Parameters
        ----------
        all_fields : tuple[str, ...]
            Complete ordered task outputs used to encode an all-fields selection.

        Returns
        -------
        dict[str, object]
            Semantic metric identifier, kind, space, fields, reduction, and direction.

        """
        fields: object = "all" if self.fields == all_fields else list(self.fields)
        return {
            "id": self.id,
            "kind": self.kind,
            "space": self.space,
            "fields": fields,
            "reduction": self.reduction,
            "direction": self.direction,
        }


def field_metric_specs(fields: tuple[str, ...]) -> tuple[MetricSpec, ...]:
    """
    Return the per-field metric family that both tasks declare over their metric fields.

    Per field: scaled and physical RMSE and MAE, and the physical relative L2 and H1
    norms; for each relative norm also one summary over all fields, which averages the
    fields within their output group and then the groups with the group weights.

    Parameters
    ----------
    fields : tuple[str, ...]
        Ordered task metric fields.

    Returns
    -------
    tuple[MetricSpec, ...]
        ``{normalized,physical}_{rmse,mae}_<field>``, ``physical_relative_{l2,h1}`` and
        ``physical_relative_{l2,h1}_<field>``, all reduced by ``material_case_mean``.

    """
    specs = [
        MetricSpec(f"{space}_{kind}_{field}", kind, space, (field,), "material_case_mean", "minimize")
        for kind in ("rmse", "mae")
        for space in ("normalized", "physical")
        for field in fields
    ]
    for kind in ("relative_l2", "relative_h1"):
        specs.append(MetricSpec(f"physical_{kind}", kind, "physical", fields, "material_case_mean", "minimize"))
        specs.extend(MetricSpec(f"physical_{kind}_{field}", kind, "physical", (field,), "material_case_mean", "minimize") for field in fields)
    return tuple(specs)


@dataclass(frozen=True, slots=True)
class ReferenceFloorSpec:
    """
    Declare the smallest reference magnitude of one metric field that supports a relative norm.

    Attributes
    ----------
    field : str
        Metric field name.
    rms : float
        Positive finite reference root mean square in the field's physical unit. A case
        whose reference RMS over a norm's domain (the L2 or H1 integrand, per grid point)
        falls below it has no relative value for that field: it is counted as unavailable,
        never set to zero or stabilized.

    Raises
    ------
    ValueError
        If the field is empty or the floor is not positive and finite.

    """

    field: str
    rms: float

    def __post_init__(self) -> None:
        """Reject empty fields and non-positive or non-finite floors."""
        if not isinstance(self.field, str) or not self.field:
            msg = "Reference floor field must be a non-empty string."
            raise ValueError(msg)
        if isinstance(self.rms, bool) or not isinstance(self.rms, float) or not math.isfinite(self.rms) or self.rms <= 0.0:
            msg = f"Reference floor of {self.field!r} must be a positive finite float."
            raise ValueError(msg)

    def as_dict(self) -> dict[str, object]:
        """Return the JSON-serializable floor declaration."""
        return {"field": self.field, "rms": self.rms}


@dataclass(frozen=True, slots=True)
class PhysicsSpec:
    """
    Describe the semantic selector for a task-owned equation set.

    Attributes
    ----------
    kind : str
        Physics registry identifier.
    equation_set : str
        Descriptive equation-set identifier.
    continuity : str
        Task-owned continuity-formulation identifier.
    boundary : str
        Task-owned boundary-formulation identifier.

    Raises
    ------
    ValueError
        If any selector is empty.

    """

    kind: str
    equation_set: str
    continuity: str
    boundary: str

    def __post_init__(self) -> None:
        """Require explicit semantic physics selectors."""
        values = (self.kind, self.equation_set, self.continuity, self.boundary)
        if any(not isinstance(value, str) or not value for value in values):
            msg = "Physics kind, equation_set, continuity, and boundary must be non-empty strings."
            raise ValueError(msg)

    def as_dict(self) -> dict[str, object]:
        """
        Return a JSON-serializable physics selection.

        Returns
        -------
        dict[str, object]
            Semantic physics selector and task-owned formulation identifiers.

        """
        return {
            "kind": self.kind,
            "equation_set": self.equation_set,
            "continuity": self.continuity,
            "boundary": self.boundary,
        }


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """
    Describe the complete immutable contract for one learned operator task.

    Attributes
    ----------
    id : str
        Canonical task identifier.
    inputs : tuple[FieldSpec, ...]
        Exact ordered learned input fields.
    outputs : tuple[FieldSpec, ...]
        Exact ordered learned output fields.
    tensor_layout : tuple[str, ...]
        Named model tensor axes.
    operator_axes : tuple[int, ...]
        Spatial axes operated on by the neural operator.
    normalization_axes : tuple[int, ...]
        Axes used to fit per-channel normalization.
    preprocessing : PreprocessingSpec
        Task-owned preprocessing assumptions.
    data_losses : tuple[str, ...]
        Allowed semantic data-loss kinds.
    default_metrics : tuple[MetricSpec, ...]
        Default semantic evaluation metrics. Declaration order has no model-selection meaning.
    physics : PhysicsSpec
        Task-owned physics selector.
    output_groups : tuple[OutputGroupSpec, ...]
        Ordered physical output quantities. When declared, the groups form an
        exact ordered partition of the task metric fields.
    reference_floors : tuple[ReferenceFloorSpec, ...], optional
        Smallest reference RMS per metric field that supports a relative norm; when
        declared, one floor per metric field in declared order.
    metric_fields : tuple[FieldSpec, ...], optional
        Reconstructed physical fields used for metric reporting when model
        outputs represent increments or another derived quantity.
    input_profiles : tuple[InputProfileSpec, ...], optional
        Exact named model input subsets retained in the persisted contract.
    primary_input_profile : str | None, optional
        Default input-profile identifier, or ``None`` when all inputs apply.
    temporal_conditioning_kinds : tuple[str, ...], optional
        Explicit task-supported temporal conditioning identifiers.
    training_airflow_source : str | None, optional
        Semantic provenance identifier for the airflow fields used in training.

    Raises
    ------
    TypeError
        If nested dataset, preprocessing, or physics declarations have the
        wrong runtime type.
    ValueError
        If identifiers, fields, axes, losses, or metrics violate the current
        immutable task schema, including duplicate or unknown metric fields.

    Notes
    -----
    Contract payloads and SHA-256 digests preserve metric declaration order and
    contain no mutable runtime state. Executable requests select objectives by
    exact metric ID. Tuple position never selects a model.

    """

    id: str
    inputs: tuple[FieldSpec, ...]
    outputs: tuple[FieldSpec, ...]
    tensor_layout: tuple[str, ...]
    operator_axes: tuple[int, ...]
    normalization_axes: tuple[int, ...]
    preprocessing: PreprocessingSpec
    data_losses: tuple[str, ...]
    default_metrics: tuple[MetricSpec, ...]
    physics: PhysicsSpec
    output_groups: tuple[OutputGroupSpec, ...] = ()
    reference_floors: tuple[ReferenceFloorSpec, ...] = ()
    metric_fields: tuple[FieldSpec, ...] = ()
    input_profiles: tuple[InputProfileSpec, ...] = ()
    primary_input_profile: str | None = None
    temporal_conditioning_kinds: tuple[str, ...] = ()
    training_airflow_source: str | None = None

    def __post_init__(self) -> None:
        """Reject structurally ambiguous task declarations."""
        self._validate_identity_and_axes()
        self._validate_field_and_nested_contracts()
        self._validate_metrics_and_losses()
        self._validate_input_profiles_and_temporal_contract()
        self._validate_output_groups_and_metric_references()

    def _validate_identity_and_axes(self) -> None:
        """Validate the task identifier and fixed 2D axes."""
        if not isinstance(self.id, str) or not self.id:
            msg = "Task id must be a non-empty string."
            raise ValueError(msg)
        if self.tensor_layout != _SUPPORTED_TENSOR_LAYOUT:
            msg = f"Task {self.id!r} tensor_layout must be the current 2D layout {_SUPPORTED_TENSOR_LAYOUT!r}."
            raise ValueError(msg)
        if self.operator_axes != _SUPPORTED_OPERATOR_AXES or self.normalization_axes != _SUPPORTED_NORMALIZATION_AXES:
            msg = (
                f"Task {self.id!r} axes must match current 2D operator/normalizer support: "
                f"operator={_SUPPORTED_OPERATOR_AXES!r}, normalization={_SUPPORTED_NORMALIZATION_AXES!r}."
            )
            raise ValueError(msg)

    def _validate_field_and_nested_contracts(self) -> None:
        """Validate field tuples and required nested declarations."""
        valid_fields = (
            isinstance(self.inputs, tuple)
            and self.inputs
            and all(isinstance(field, FieldSpec) for field in self.inputs)
            and isinstance(self.outputs, tuple)
            and self.outputs
            and all(isinstance(field, FieldSpec) for field in self.outputs)
        )
        if not valid_fields:
            msg = f"Task {self.id!r} must declare non-empty FieldSpec input/output tuples."
            raise ValueError(msg)
        for value, expected, label in (
            (self.preprocessing, PreprocessingSpec, "preprocessing"),
            (self.physics, PhysicsSpec, "physics"),
        ):
            if not isinstance(value, expected):
                msg = f"Task {self.id!r} {label} must be a {expected.__name__} declaration."
                raise TypeError(msg)
        if not isinstance(self.output_groups, tuple) or not all(isinstance(group, OutputGroupSpec) for group in self.output_groups):
            msg = f"Task {self.id!r} output_groups must be an OutputGroupSpec tuple."
            raise TypeError(msg)
        if not isinstance(self.metric_fields, tuple) or not all(isinstance(field, FieldSpec) for field in self.metric_fields):
            msg = f"Task {self.id!r} metric_fields must be a FieldSpec tuple."
            raise TypeError(msg)
        names = (*self.input_names, *self.output_names)
        if len(names) != len(set(names)):
            msg = f"Task {self.id!r} contains duplicate field names: {names!r}."
            raise ValueError(msg)
        if len(self.metric_names) != len(set(self.metric_names)):
            msg = f"Task {self.id!r} contains duplicate metric field names: {self.metric_names!r}."
            raise ValueError(msg)

    def _validate_metrics_and_losses(self) -> None:
        """Validate task-owned data losses and metric declarations."""
        if not isinstance(self.data_losses, tuple) or not self.data_losses:
            msg = f"Task {self.id!r} must declare at least one semantic data loss."
            raise ValueError(msg)
        if not all(isinstance(loss, str) and loss for loss in self.data_losses) or len(set(self.data_losses)) != len(self.data_losses):
            msg = f"Task {self.id!r} data_losses must be unique non-empty strings."
            raise ValueError(msg)
        valid_metrics = (
            isinstance(self.default_metrics, tuple)
            and self.default_metrics
            and all(isinstance(metric, MetricSpec) for metric in self.default_metrics)
        )
        if not valid_metrics:
            msg = f"Task {self.id!r} must declare a non-empty MetricSpec tuple."
            raise ValueError(msg)
        metric_ids = tuple(metric.id for metric in self.default_metrics)
        if len(metric_ids) != len(set(metric_ids)):
            msg = f"Task {self.id!r} contains duplicate default metric ids: {metric_ids!r}."
            raise ValueError(msg)
        unknown = sorted({field for metric in self.default_metrics for field in metric.fields if field not in self.metric_names})
        if unknown:
            msg = f"Task {self.id!r} metrics reference unknown output fields: {unknown!r}."
            raise ValueError(msg)

    def _validate_input_profiles_and_temporal_contract(self) -> None:
        """Validate optional input profiles and temporal provenance declarations."""
        if not isinstance(self.input_profiles, tuple) or not all(isinstance(profile, InputProfileSpec) for profile in self.input_profiles):
            msg = f"Task {self.id!r} input_profiles must be an InputProfileSpec tuple."
            raise TypeError(msg)
        if self.input_profiles:
            profile_ids = tuple(profile.id for profile in self.input_profiles)
            if len(profile_ids) != len(set(profile_ids)) or self.primary_input_profile not in profile_ids:
                msg = f"Task {self.id!r} primary_input_profile must select one uniquely declared profile."
                raise ValueError(msg)
            unknown = sorted({field for profile in self.input_profiles for field in profile.fields if field not in self.input_names})
            if unknown:
                msg = f"Task {self.id!r} input profiles reference unknown fields: {unknown!r}."
                raise ValueError(msg)
        elif self.primary_input_profile is not None:
            msg = f"Task {self.id!r} cannot declare primary_input_profile without input_profiles."
            raise ValueError(msg)
        valid_temporal = (
            isinstance(self.temporal_conditioning_kinds, tuple)
            and all(isinstance(kind, str) and kind for kind in self.temporal_conditioning_kinds)
            and len(set(self.temporal_conditioning_kinds)) == len(self.temporal_conditioning_kinds)
        )
        if not valid_temporal:
            msg = f"Task {self.id!r} temporal_conditioning_kinds must be a unique tuple of non-empty strings."
            raise ValueError(msg)
        if self.training_airflow_source is not None and (not isinstance(self.training_airflow_source, str) or not self.training_airflow_source):
            msg = f"Task {self.id!r} training_airflow_source must be None or a non-empty string."
            raise ValueError(msg)

    def _validate_output_groups_and_metric_references(self) -> None:
        """Validate that optional output groups partition and reference floors cover the metric fields exactly."""
        if not isinstance(self.reference_floors, tuple) or not all(isinstance(floor, ReferenceFloorSpec) for floor in self.reference_floors):
            msg = f"Task {self.id!r} reference_floors must be a ReferenceFloorSpec tuple."
            raise TypeError(msg)
        if self.reference_floors and tuple(floor.field for floor in self.reference_floors) != self.metric_names:
            msg = f"Task {self.id!r} reference floors must name every metric field once in declared order: {self.metric_names!r}."
            raise ValueError(msg)
        if not self.output_groups:
            return
        group_ids = tuple(group.id for group in self.output_groups)
        if len(group_ids) != len(set(group_ids)):
            msg = f"Task {self.id!r} contains duplicate output group ids: {group_ids!r}."
            raise ValueError(msg)
        grouped_fields = tuple(field for group in self.output_groups for field in group.fields)
        if grouped_fields != self.metric_names:
            msg = (
                f"Task {self.id!r} output groups must partition metric fields in declared order: "
                f"expected {self.metric_names!r}, got {grouped_fields!r}."
            )
            raise ValueError(msg)

    @property
    def input_names(self) -> tuple[str, ...]:
        """
        Return exact ordered input field names.

        Returns
        -------
        tuple[str, ...]
            Canonical task input order.

        """
        return tuple(field.name for field in self.inputs)

    @property
    def output_names(self) -> tuple[str, ...]:
        """
        Return exact ordered output field names.

        Returns
        -------
        tuple[str, ...]
            Canonical task output order.

        """
        return tuple(field.name for field in self.outputs)

    @property
    def metric_names(self) -> tuple[str, ...]:
        """Return ordered reconstructed-state metric field names."""
        selected = self.metric_fields or self.outputs
        return tuple(field.name for field in selected)

    @property
    def error_terms(self) -> tuple[ErrorTermSpec, ...]:
        """Return the weighted error terms of the task's output groups, shared by loss and selection."""
        return error_terms(self.output_groups)

    def reference_floor(self, name: str) -> float:
        """Return the declared reference-RMS floor of one metric field's relative norms."""
        for floor in self.reference_floors:
            if floor.field == name:
                return floor.rms
        msg = f"Task {self.id!r} declares no reference floor for metric field {name!r}."
        raise ValueError(msg)

    def metric_field(self, name: str) -> FieldSpec:
        """Resolve one exact metric field without conflating model outputs."""
        selected = self.metric_fields or self.outputs
        for field in selected:
            if field.name == name:
                return field
        available = ", ".join(self.metric_names)
        msg = f"Unknown metric field {name!r} for task {self.id!r}. Available fields: {available}."
        raise ValueError(msg)

    def input_profile(self, profile_id: str | None = None) -> InputProfileSpec:
        """Resolve one declared input profile, defaulting only to the task primary."""
        selected_id = self.primary_input_profile if profile_id is None else profile_id
        if selected_id is None:
            return InputProfileSpec(
                id="task_inputs",
                fields=self.input_names,
                coordinate_policy="task_declared",
            )
        for profile in self.input_profiles:
            if profile.id == selected_id:
                return profile
        available = ", ".join(profile.id for profile in self.input_profiles)
        msg = f"Unknown input profile {selected_id!r} for task {self.id!r}. Available profiles: {available}."
        raise ValueError(msg)

    @property
    def in_channels(self) -> int:
        """
        Return the input-channel count derived from ordered fields.

        Returns
        -------
        int
            Number of declared task inputs.

        """
        return len(self.inputs)

    @property
    def out_channels(self) -> int:
        """
        Return the output-channel count derived from ordered fields.

        Returns
        -------
        int
            Number of declared task outputs.

        """
        return len(self.outputs)

    @property
    def operator_dimensionality(self) -> int:
        """
        Return the number of task-owned operator axes.

        Returns
        -------
        int
            Spatial operator dimensionality.

        """
        return len(self.operator_axes)

    def field(self, name: str) -> FieldSpec:
        """
        Resolve one field by its exact canonical name.

        Parameters
        ----------
        name : str
            Canonical task input or output field name.

        Returns
        -------
        FieldSpec
            Immutable matching field descriptor.

        Raises
        ------
        ValueError
            If `name` is not declared by the task.

        """
        for field in (*self.inputs, *self.outputs):
            if field.name == name:
                return field
        available = ", ".join((*self.input_names, *self.output_names))
        msg = f"Unknown field {name!r} for task {self.id!r}. Available fields: {available}."
        raise ValueError(msg)

    def data_contract_payload(self) -> dict[str, object]:
        """
        Return the canonical learned-data compatibility contract.

        Returns
        -------
        dict[str, object]
            Task identity, fields, tensor/normalization axes, and preprocessing
            semantics that determine whether stored tensors remain compatible.

        Notes
        -----
        Metrics, output groups, losses, physics choices, and dataset defaults do
        not alter learned tensor content and therefore remain outside this digest.

        """
        payload: dict[str, object] = {
            "task": self.id,
            "inputs": [field.as_dict() for field in self.inputs],
            "outputs": [field.as_dict() for field in self.outputs],
            "tensor_layout": list(self.tensor_layout),
            "operator_axes": list(self.operator_axes),
            "normalization_axes": list(self.normalization_axes),
            "preprocessing": self.preprocessing.as_dict(),
        }
        if self.metric_fields:
            payload["metric_fields"] = [field.as_dict() for field in self.metric_fields]
        if self.input_profiles:
            payload["input_profiles"] = [profile.as_dict() for profile in self.input_profiles]
            payload["primary_input_profile"] = self.primary_input_profile
        if self.temporal_conditioning_kinds:
            payload["temporal_conditioning_kinds"] = list(self.temporal_conditioning_kinds)
        if self.training_airflow_source is not None:
            payload["training_airflow_source"] = self.training_airflow_source
        return payload

    @property
    def data_contract_digest(self) -> str:
        """
        Return a stable SHA-256 digest of learned-data compatibility semantics.

        Returns
        -------
        str
            Lowercase hexadecimal SHA-256 digest independent of evaluation policy.

        """
        return _canonical_sha256(self.data_contract_payload())

    def contract_payload(self) -> dict[str, object]:
        """
        Return the canonical complete task payload used for persistence and hashing.

        Returns
        -------
        dict[str, object]
            Complete JSON-serializable task contract without derived digest metadata.

        """
        payload: dict[str, object] = {
            "task": self.id,
            "inputs": [field.as_dict() for field in self.inputs],
            "outputs": [field.as_dict() for field in self.outputs],
            "output_groups": [group.as_dict() for group in self.output_groups],
            "tensor_layout": list(self.tensor_layout),
            "operator_axes": list(self.operator_axes),
            "normalization_axes": list(self.normalization_axes),
            "operator_dimensionality": self.operator_dimensionality,
            "preprocessing": self.preprocessing.as_dict(),
            "data_losses": list(self.data_losses),
            "default_metrics": [metric.as_dict(all_fields=self.metric_names) for metric in self.default_metrics],
            "physics": self.physics.as_dict(),
        }
        if self.reference_floors:
            payload["reference_floors"] = [floor.as_dict() for floor in self.reference_floors]
        if self.metric_fields:
            payload["metric_fields"] = [field.as_dict() for field in self.metric_fields]
        if self.input_profiles:
            payload["input_profiles"] = [profile.as_dict() for profile in self.input_profiles]
            payload["primary_input_profile"] = self.primary_input_profile
        if self.temporal_conditioning_kinds:
            payload["temporal_conditioning_kinds"] = list(self.temporal_conditioning_kinds)
        if self.training_airflow_source is not None:
            payload["training_airflow_source"] = self.training_airflow_source
        return payload

    @property
    def contract_digest(self) -> str:
        """
        Return a stable SHA-256 digest of the complete task contract.

        Returns
        -------
        str
            Lowercase hexadecimal SHA-256 digest.

        """
        return _canonical_sha256(self.contract_payload())

    def resolved_contract(self) -> dict[str, object]:
        """
        Return the persisted resolved task contract with derived metadata.

        Returns
        -------
        dict[str, object]
            Canonical contract plus digest and derived channel counts.

        """
        payload = self.contract_payload()
        payload["digest"] = self.contract_digest
        payload["data_contract_digest"] = self.data_contract_digest
        payload["in_channels"] = self.in_channels
        payload["out_channels"] = self.out_channels
        return payload
