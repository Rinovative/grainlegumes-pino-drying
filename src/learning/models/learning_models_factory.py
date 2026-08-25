"""
learning_models_factory.py

Construct FNO, UNO, and capability-gated RNO models from resolved experiment configs.

Responsibilities:
  - Build FNO models from channel, mode and layer settings
  - Build UNO models with configured mode schedules
  - Build official neuraloperator RNO models when the installed dependency exposes it
  - Declare the maintained neuraloperator UNO resampling semantics
  - Resolve semantic model identifiers from strict registries

Design principles:
  - Neuraloperator provides the architecture implementations
  - Parameter passing stays explicit and traceable
  - Semantic identifiers remain independent of implementation class names
  - Device placement happens only when requested by the caller

This module does NOT:
  - Implement FNO, UNO, or RNO architectures. ``neuraloperator`` supplies them
  - Orchestrate training. ``learning.training.loop`` owns execution
  - Derive task channels or axes. Task contracts and config resolution own them
"""

from __future__ import annotations

import copy
import importlib
import io
import math
import sys
from collections.abc import Mapping
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    import torch
    from neuralop.models import FNO, UNO

_SkipConnection = Literal["linear", "identity", "soft-gating"]
_UNO_LAYERS_5 = 5
_UNO_LAYERS_7 = 7
_MIN_UNO_MODE = 8
_FNO_MODE_DIMENSIONS = 2
_UNO_SPATIAL_DIMENSIONS = 2
UNO_RESAMPLING_MODE = "bicubic"


def resolve_uno_scalings(
    n_layers: int,
    uno_scalings: list[list[float]] | None,
) -> list[list[float]]:
    """Return validated explicit or layer-derived UNO spatial scalings."""
    if n_layers == _UNO_LAYERS_5:
        defaults = [
            [1.0, 1.0],
            [0.5, 0.5],
            [1.0, 1.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ]
    elif n_layers == _UNO_LAYERS_7:
        defaults = [
            [1.0, 1.0],
            [0.5, 0.5],
            [0.5, 0.5],
            [1.0, 1.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [2.0, 2.0],
        ]
    else:
        msg = f"UNO supports exactly {_UNO_LAYERS_5} or {_UNO_LAYERS_7} layers, got {n_layers}."
        raise ValueError(msg)

    selected = defaults if uno_scalings is None else uno_scalings
    if len(selected) != n_layers:
        msg = f"uno_scalings length must match n_layers={n_layers}, got {len(selected)}."
        raise ValueError(msg)

    resolved: list[list[float]] = []
    for index, pair in enumerate(selected):
        if not isinstance(pair, (list, tuple)) or len(pair) != _UNO_SPATIAL_DIMENSIONS:
            msg = f"uno_scalings[{index}] must contain exactly two spatial scaling values, got {pair!r}."
            raise ValueError(msg)
        values: list[float] = []
        for axis, value in zip(("x", "y"), pair, strict=True):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                msg = f"uno_scalings[{index}][{axis}] must be a finite positive number, got {value!r}."
                raise TypeError(msg)
            numeric = float(value)
            if not math.isfinite(numeric) or numeric <= 0:
                msg = f"uno_scalings[{index}][{axis}] must be a finite positive number, got {value!r}."
                raise ValueError(msg)
            values.append(numeric)
        resolved.append(values)
    return resolved


def _validate_skip(name: str, value: str) -> _SkipConnection:
    """Validate a neuralop skip-connection option."""
    if value not in ("linear", "identity", "soft-gating"):
        msg = f"{name} must be one of 'linear', 'identity', 'soft-gating', got: {value!r}"
        raise ValueError(msg)
    return cast("_SkipConnection", value)


def build_fno(
    in_channels: int,
    out_channels: int,
    n_modes: list[int] | tuple[int, int],
    hidden_channels: int,
    n_layers: int,
    lifting_channel_ratio: float = 2,
    projection_channel_ratio: float = 2,
    fno_skip: str = "linear",
    channel_mlp_skip: str = "soft-gating",
    implementation: str = "factorized",
    positional_embedding: str | None = "grid",
    device: torch.device | str | None = None,
) -> FNO:
    """
    Build a Fourier Neural Operator (FNO) model.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    n_modes : list[int] | tuple[int, ...]
        Number of Fourier modes for each spatial dimension
    hidden_channels : int
        Number of hidden channels
    n_layers : int
        Number of Fourier layers
    lifting_channel_ratio : float, optional
        Channel ratio for lifting layer (default: 2)
    projection_channel_ratio : float, optional
        Channel ratio for projection layer (default: 2)
    fno_skip : str, optional
        Skip connection type for FNO blocks (default: "linear")
    channel_mlp_skip : str, optional
        Skip connection type for channel MLP (default: "soft-gating")
    implementation : str, optional
        Implementation type (default: "factorized")
    positional_embedding : str | None, optional
        Neuraloperator positional embedding (default: "grid")
    device : torch.device | str | None, optional
        Device to place model on (default: None - caller handles)

    Returns
    -------
    FNO
        Initialized FNO model

    """
    from neuralop.models import FNO  # noqa: PLC0415

    n_modes_tuple = tuple(int(mode) for mode in n_modes)
    if len(n_modes_tuple) != _FNO_MODE_DIMENSIONS:
        msg = f"FNO requires exactly two n_modes entries, got: {n_modes!r}"
        raise ValueError(msg)

    kwargs: dict[str, Any] = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "n_modes": n_modes_tuple,
        "hidden_channels": hidden_channels,
        "n_layers": n_layers,
        "lifting_channel_ratio": lifting_channel_ratio,
        "projection_channel_ratio": projection_channel_ratio,
        "fno_skip": _validate_skip("fno_skip", fno_skip),
        "channel_mlp_skip": _validate_skip("channel_mlp_skip", channel_mlp_skip),
        "implementation": implementation,
    }
    if positional_embedding != "grid":
        kwargs["positional_embedding"] = positional_embedding
    model = FNO(**kwargs)

    if device is not None:
        model.to(device)

    return model


@contextmanager
def _filter_uno_constructor_stdout() -> Iterator[None]:
    """Suppress only neuraloperator's known UNO skip-option debug lines."""
    captured = io.StringIO()
    known_noise = {f"{name}={value!r}" for name in ("fno_skip", "channel_mlp_skip") for value in ("linear", "identity", "soft-gating", None)}
    try:
        with redirect_stdout(captured):
            yield
    finally:
        for line in captured.getvalue().splitlines(keepends=True):
            if line.rstrip("\r\n") not in known_noise:
                sys.stdout.write(line)


def _validate_transient_axis_modes(
    *,
    label: str,
    n_modes: tuple[int, int],
    spatial_shape: tuple[int, int],
) -> None:
    """Require [Y, X] spectral modes to fit one concrete transient grid."""
    y_size, x_size = spatial_shape
    y_modes, x_modes = n_modes
    if y_modes > y_size:
        message = f"{label} modes_y={y_modes} exceeds admitted Y axis {y_size}."
        raise ValueError(message)
    if x_modes > x_size:
        message = f"{label} modes_x={x_modes} exceeds admitted X axis {x_size}."
        raise ValueError(message)


def _uno_mode_schedule(
    *,
    n_layers: int,
    modes_y: int,
    modes_x: int,
    mode_ratio: float,
) -> list[list[int]]:
    """Build the authoritative UNO [Y, X] spectral-mode schedule."""
    mid_y = max(_MIN_UNO_MODE, int(modes_y * mode_ratio))
    mid_x = max(_MIN_UNO_MODE, int(modes_x * mode_ratio))
    middle = [[mid_y, mid_x]] * (n_layers - 2)
    return [[modes_y, modes_x], *middle, [modes_y, modes_x]]


def _scaled_axis_size(size: int, scale: float) -> int:
    """Return neuraloperator's positive integer resampling target size."""
    return max(1, round(size * scale))


def validate_transient_model_spatial_shape(
    config: Mapping[str, Any],
    spatial_shape: tuple[int, int],
) -> None:
    """Validate transient model spectral modes against Train-admitted [Y, X]."""
    if config.get("task") != "transient_drying":
        return
    if (
        not isinstance(spatial_shape, tuple)
        or len(spatial_shape) != _FNO_MODE_DIMENSIONS
        or any(isinstance(axis, bool) or not isinstance(axis, int) or axis < 1 for axis in spatial_shape)
    ):
        message = "Transient spatial_shape must contain positive exact [Y, X] integers."
        raise ValueError(message)
    model = config.get("model")
    if not isinstance(model, Mapping) or not isinstance(model.get("params"), Mapping):
        message = "Transient model validation requires resolved model.params."
        raise TypeError(message)
    kind = model.get("kind")
    params = model["params"]
    if kind in {"fno", "rno"}:
        raw_modes = params.get("n_modes")
        if not isinstance(raw_modes, (list, tuple)) or len(raw_modes) != _FNO_MODE_DIMENSIONS:
            message = "Transient FNO/RNO requires exact n_modes [Y, X]."
            raise ValueError(message)
        axis_modes = tuple(raw_modes)
        if any(isinstance(mode, bool) or not isinstance(mode, int) or mode < 1 for mode in axis_modes):
            message = "Transient FNO/RNO n_modes must contain positive exact integers."
            raise ValueError(message)
        _validate_transient_axis_modes(
            label=str(kind).upper(),
            n_modes=(axis_modes[0], axis_modes[1]),
            spatial_shape=spatial_shape,
        )
    elif kind == "uno":
        modes_y = params.get("modes_y")
        modes_x = params.get("modes_x")
        n_layers = params.get("n_layers")
        ratio = params.get("mode_ratio", 0.5)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in (modes_y, modes_x, n_layers)):
            message = "Transient UNO modes_y, modes_x, and n_layers must be positive exact integers."
            raise ValueError(message)
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(float(ratio)) or float(ratio) <= 0:
            message = "Transient UNO mode_ratio must be one positive finite number."
            raise ValueError(message)
        scalings = resolve_uno_scalings(n_layers, params.get("uno_scalings"))
        shape = spatial_shape
        for index, block_modes in enumerate(
            _uno_mode_schedule(
                n_layers=n_layers,
                modes_y=modes_y,
                modes_x=modes_x,
                mode_ratio=float(ratio),
            )
        ):
            _validate_transient_axis_modes(
                label=f"UNO block {index}",
                n_modes=(block_modes[0], block_modes[1]),
                spatial_shape=shape,
            )
            scale_y, scale_x = scalings[index]
            shape = (_scaled_axis_size(shape[0], scale_y), _scaled_axis_size(shape[1], scale_x))


def assess_transient_model_spatial_compatibility(
    config: Mapping[str, Any],
    *,
    training_shape: tuple[int, int],
    evaluation_shape: tuple[int, int],
) -> dict[str, Any]:
    """Return one explicit architecture decision for a requested grid."""
    model = config.get("model")
    if config.get("task") != "transient_drying" or not isinstance(model, Mapping):
        message = "Transient spatial compatibility requires resolved transient_drying model configuration."
        raise TypeError(message)
    kind = model.get("kind")
    if kind not in {"fno", "uno", "rno"}:
        return {
            "decision": "UNKNOWN",
            "model_kind": kind,
            "training_shape": list(training_shape),
            "evaluation_shape": list(evaluation_shape),
            "reason": "The model kind has no maintained transient spatial capability contract.",
        }
    try:
        validate_transient_model_spatial_shape(config, training_shape)
        if evaluation_shape != training_shape:
            validate_transient_model_spatial_shape(config, evaluation_shape)
    except (TypeError, ValueError) as error:
        return {
            "decision": "UNSUPPORTED",
            "model_kind": kind,
            "training_shape": list(training_shape),
            "evaluation_shape": list(evaluation_shape),
            "reason": str(error),
        }
    decision = "SUPPORTED_EXACTLY" if evaluation_shape == training_shape else "SUPPORTED_WITH_CONTRACT"
    constraints = [
        "retained_fourier_modes_fit_requested_axes",
        "explicit_coordinate_channels_follow_requested_grid",
        "no_persisted_fixed_spatial_shape_parameter",
        "model_output_must_match_requested_shape_exactly",
    ]
    if kind == "uno":
        constraints.append("every_uno_resampling_level_retains_valid_axis_modes")
    if kind == "rno":
        constraints.append("recurrent_hidden_state_is_request_local_and_shape_checked")
    params = model.get("params")
    positional_embedding = params.get("positional_embedding") if isinstance(params, Mapping) else None
    return {
        "decision": decision,
        "model_kind": kind,
        "training_shape": list(training_shape),
        "evaluation_shape": list(evaluation_shape),
        "validated_constraints": constraints,
        "coordinate_handling": {
            "tensorizer": "explicit_x_y_channels",
            "model_positional_embedding": positional_embedding,
        },
        "forward_output_shape_proof": (
            "checkpoint_training_shape_contract" if decision == "SUPPORTED_EXACTLY" else "pending_synthetic_single_step_probe"
        ),
        "reason": (
            "Requested Evaluation shape equals the admitted Training shape."
            if decision == "SUPPORTED_EXACTLY"
            else "The maintained architecture has shape-independent parameters and passes all requested-grid mode and resampling constraints."
        ),
    }


def build_uno(
    in_channels: int,
    out_channels: int,
    n_layers: int,
    hidden_channels: int,
    modes_x: int,
    modes_y: int,
    mode_ratio: float = 0.5,
    uno_scalings: list[list[float]] | None = None,
    channel_mlp_skip: str = "linear",
    positional_embedding: str | None = "grid",
    device: torch.device | str | None = None,
) -> UNO:
    """
    Build a U-shaped Neural Operator (UNO) model.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    n_layers : int
        Number of UNO layers
    hidden_channels : int
        Number of hidden channels
    modes_x : int
        Base number of Fourier modes in x-direction
    modes_y : int
        Base number of Fourier modes in y-direction
    mode_ratio : float, optional
        Ratio for computing intermediate layer modes (default: 0.5)
    uno_scalings : list[list[float]] | None, optional
        Layer-specific spatial scalings (default: None - auto-computed)
    channel_mlp_skip : str, optional
        Skip connection type for channel MLP (default: "linear")
    positional_embedding : str | None, optional
        Neuraloperator positional embedding (default: "grid")
    device : torch.device | str | None, optional
        Device to place model on (default: None - caller handles)

    Returns
    -------
    UNO
        Initialized UNO model

    """
    from neuralop.models import UNO  # noqa: PLC0415

    if n_layers not in {_UNO_LAYERS_5, _UNO_LAYERS_7}:
        msg = f"UNO supports exactly {_UNO_LAYERS_5} or {_UNO_LAYERS_7} layers, got {n_layers}."
        raise ValueError(msg)

    uno_scalings = resolve_uno_scalings(n_layers, uno_scalings)

    uno_n_modes = _uno_mode_schedule(
        n_layers=n_layers,
        modes_y=modes_y,
        modes_x=modes_x,
        mode_ratio=mode_ratio,
    )

    uno_out_channels = [hidden_channels] * n_layers

    constructor_options: dict[str, Any] = {}
    if positional_embedding != "grid":
        constructor_options["positional_embedding"] = positional_embedding
    with _filter_uno_constructor_stdout():
        model = UNO(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            uno_out_channels=uno_out_channels,
            uno_n_modes=uno_n_modes,
            uno_scalings=uno_scalings,
            channel_mlp_skip=_validate_skip("channel_mlp_skip", channel_mlp_skip),
            **constructor_options,
        )

    if device is not None:
        model.to(device)

    return model


def build_rno(
    in_channels: int,
    out_channels: int,
    n_modes: list[int] | tuple[int, int],
    hidden_channels: int,
    n_layers: int = 4,
    rno_skip: bool = False,
    lifting_channel_ratio: float = 2,
    projection_channel_ratio: float = 2,
    positional_embedding: str | None = "grid",
    channel_mlp_skip: str = "soft-gating",
    fno_skip: str = "linear",
    return_sequences: bool = False,
    device: torch.device | str | None = None,
) -> torch.nn.Module:
    """Build the official neuraloperator recurrent neural operator."""
    if return_sequences is not False:
        message = "RNO requires return_sequences=False; transient rollouts carry hidden state explicitly."
        raise ValueError(message)
    rno_builder = _resolve_rno_builder()
    if rno_builder is None:
        message = "The installed neuraloperator package does not expose official RNO support."
        raise RuntimeError(message)
    n_modes_tuple = tuple(int(mode) for mode in n_modes)
    if len(n_modes_tuple) != _FNO_MODE_DIMENSIONS:
        message = f"RNO requires exactly two n_modes entries, got: {n_modes!r}"
        raise ValueError(message)
    model = rno_builder(
        n_modes=n_modes_tuple,
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        rno_skip=rno_skip,
        lifting_channel_ratio=lifting_channel_ratio,
        projection_channel_ratio=projection_channel_ratio,
        positional_embedding=positional_embedding,
        channel_mlp_skip=_validate_skip("channel_mlp_skip", channel_mlp_skip),
        fno_skip=_validate_skip("fno_skip", fno_skip),
        return_sequences=False,
    )
    if device is not None:
        model.to(device)
    return model


def _resolve_rno_builder() -> Callable[..., torch.nn.Module] | None:
    """Return the official neuraloperator RNO constructor when exposed."""
    try:
        models = importlib.import_module("neuralop.models")
    except ImportError:
        return None
    builder = getattr(models, "RNO", None)
    if not callable(builder):
        return None
    return cast("Callable[..., torch.nn.Module]", builder)


def _rno_is_available() -> bool:
    """Return whether the installed neuraloperator exposes official RNO."""
    return _resolve_rno_builder() is not None


@dataclass(frozen=True, slots=True)
class ModelKindSpec:
    """
    Describe one semantic model kind.

    Attributes
    ----------
    kind : str
        Canonical saved configuration identifier.
    builder : Callable[..., torch.nn.Module]
        Internal model-construction callable.
    defaults : Mapping[str, Any]
        Immutable implementation defaults.
    allowed_params : frozenset[str]
        Accepted resolved model parameter names.
    required_params : frozenset[str]
        Required resolved model parameter names.

    """

    kind: str
    builder: Callable[..., torch.nn.Module]
    defaults: Mapping[str, Any]
    allowed_params: frozenset[str]
    required_params: frozenset[str]


_MODEL_KINDS = MappingProxyType(
    {
        "fno": ModelKindSpec(
            kind="fno",
            builder=build_fno,
            defaults=MappingProxyType(
                {
                    "lifting_channel_ratio": 2,
                    "projection_channel_ratio": 2,
                    "fno_skip": "linear",
                    "channel_mlp_skip": "soft-gating",
                    "implementation": "factorized",
                }
            ),
            allowed_params=frozenset(
                {
                    "in_channels",
                    "out_channels",
                    "n_modes",
                    "hidden_channels",
                    "n_layers",
                    "lifting_channel_ratio",
                    "projection_channel_ratio",
                    "fno_skip",
                    "channel_mlp_skip",
                    "implementation",
                    "positional_embedding",
                }
            ),
            required_params=frozenset(
                {
                    "in_channels",
                    "out_channels",
                    "n_modes",
                    "hidden_channels",
                    "n_layers",
                }
            ),
        ),
        "uno": ModelKindSpec(
            kind="uno",
            builder=build_uno,
            defaults=MappingProxyType({"channel_mlp_skip": "linear"}),
            allowed_params=frozenset(
                {
                    "in_channels",
                    "out_channels",
                    "n_layers",
                    "hidden_channels",
                    "modes_x",
                    "modes_y",
                    "mode_ratio",
                    "uno_scalings",
                    "channel_mlp_skip",
                    "positional_embedding",
                }
            ),
            required_params=frozenset(
                {
                    "in_channels",
                    "out_channels",
                    "n_layers",
                    "hidden_channels",
                    "modes_x",
                    "modes_y",
                }
            ),
        ),
        "rno": ModelKindSpec(
            kind="rno",
            builder=build_rno,
            defaults=MappingProxyType(
                {
                    "n_layers": 4,
                    "rno_skip": False,
                    "lifting_channel_ratio": 2,
                    "projection_channel_ratio": 2,
                    "positional_embedding": "grid",
                    "channel_mlp_skip": "soft-gating",
                    "fno_skip": "linear",
                    "return_sequences": False,
                }
            ),
            allowed_params=frozenset(
                {
                    "in_channels",
                    "out_channels",
                    "n_modes",
                    "hidden_channels",
                    "n_layers",
                    "rno_skip",
                    "lifting_channel_ratio",
                    "projection_channel_ratio",
                    "positional_embedding",
                    "channel_mlp_skip",
                    "fno_skip",
                    "return_sequences",
                }
            ),
            required_params=frozenset({"in_channels", "out_channels", "n_modes", "hidden_channels"}),
        ),
    }
)


def available_model_kinds() -> tuple[str, ...]:
    """
    Return registered semantic model identifiers.

    Returns
    -------
    tuple[str, ...]
        Exact model kinds accepted by the factory.

    """
    kinds = set(_MODEL_KINDS)
    if not _rno_is_available():
        kinds.discard("rno")
    return tuple(sorted(kinds))


def resolve_model_kind(kind: str) -> ModelKindSpec:
    """
    Resolve an exact semantic model identifier.

    Parameters
    ----------
    kind : str
        Canonical model kind.

    Returns
    -------
    ModelKindSpec
        Immutable model schema and implementation descriptor.

    Raises
    ------
    ValueError
        If `kind` is not registered.

    """
    try:
        return _MODEL_KINDS[kind]
    except KeyError as error:
        available = ", ".join(available_model_kinds())
        msg = f"Unknown model identifier {kind!r}. Available models: {available}."
        raise ValueError(msg) from error


def model_defaults(kind: str) -> dict[str, Any]:
    """
    Return an isolated copy of defaults owned by a model kind.

    Parameters
    ----------
    kind : str
        Canonical model kind.

    Returns
    -------
    dict[str, Any]
        Mutable copy of the registered model defaults.

    Raises
    ------
    ValueError
        If `kind` is not registered.

    """
    return copy.deepcopy(dict(resolve_model_kind(kind).defaults))


def validate_model_params(
    kind: str,
    params: Mapping[str, Any],
    *,
    require_channels: bool,
    operator_dimensionality: int,
) -> None:
    """
    Validate model parameters against a semantic model schema.

    Parameters
    ----------
    kind : str
        Canonical model kind.
    params : Mapping[str, Any]
        Candidate model parameter mapping.
    require_channels : bool
        Whether derived input/output channel parameters must already be present.
    operator_dimensionality : int
        Number of task-owned spatial operator axes.

    Raises
    ------
    ValueError
        If the model kind, parameter names, required values, or mode shape are invalid.

    """
    spec = resolve_model_kind(kind)
    if operator_dimensionality != _FNO_MODE_DIMENSIONS:
        msg = f"Model kind {kind!r} currently supports exactly two operator axes, got {operator_dimensionality}."
        raise ValueError(msg)
    unknown = sorted(set(params).difference(spec.allowed_params))
    if unknown:
        msg = f"model.params contains unknown key(s) for {kind!r}: {unknown}."
        raise ValueError(msg)

    required = set(spec.required_params)
    if not require_channels:
        required.difference_update({"in_channels", "out_channels"})
    missing = sorted(required.difference(params))
    if missing:
        msg = f"model.params is missing required key(s) for {kind!r}: {missing}."
        raise ValueError(msg)

    if kind in {"fno", "rno"} and "n_modes" in params:
        n_modes = params["n_modes"]
        if not isinstance(n_modes, (list, tuple)) or len(n_modes) != operator_dimensionality:
            msg = f"model.params.n_modes must contain exactly {operator_dimensionality} entries for this task, got: {n_modes!r}."
            raise ValueError(msg)

    if kind == "rno" and params.get("return_sequences", False) is not False:
        message = "model.params.return_sequences must be false for RNO."
        raise ValueError(message)

    if kind == "uno":
        n_layers = params["n_layers"]
        if isinstance(n_layers, bool) or not isinstance(n_layers, int) or n_layers not in (_UNO_LAYERS_5, _UNO_LAYERS_7):
            msg = f"UNO supports exactly {_UNO_LAYERS_5} or {_UNO_LAYERS_7} layers, got {n_layers!r}."
            raise ValueError(msg)
        resolve_uno_scalings(n_layers, params.get("uno_scalings"))


def build_model(config: dict[str, Any], *, device: torch.device) -> torch.nn.Module:
    """
    Build a registered semantic model from a resolved configuration.

    Parameters
    ----------
    config : dict[str, Any]
        Resolved configuration containing model kind, parameters, and task contract.
    device : torch.device
        Concrete device resolved by the top-level runtime service.

    Returns
    -------
    torch.nn.Module
        Constructed FNO or UNO implementation.

    Raises
    ------
    TypeError
        If required resolved config sections have invalid types.
    ValueError
        If the semantic model identifier or parameters are invalid.

    """
    import torch  # noqa: PLC0415

    model_config = config.get("model")
    if not isinstance(model_config, dict):
        msg = "Resolved config must contain a model mapping."
        raise TypeError(msg)
    kind = model_config.get("kind")
    if not isinstance(kind, str):
        msg = "Resolved config must contain model.kind as a string."
        raise TypeError(msg)
    params = model_config.get("params")
    if not isinstance(params, dict):
        msg = "Resolved config must contain model.params as a mapping."
        raise TypeError(msg)

    task_contract = config.get("task_contract", {})
    operator_dimensionality = int(task_contract.get("operator_dimensionality", _FNO_MODE_DIMENSIONS))
    validate_model_params(
        kind,
        params,
        require_channels=True,
        operator_dimensionality=operator_dimensionality,
    )
    if not isinstance(device, torch.device) or device.type not in {"cpu", "cuda"}:
        msg = f"Model construction requires one concrete CPU or CUDA torch.device, got {device!r}."
        raise TypeError(msg)
    if device.type == "cuda" and device.index is None:
        msg = "Model construction requires an indexed CUDA device resolved by the runtime boundary."
        raise ValueError(msg)
    return resolve_model_kind(kind).builder(**params, device=device)
