"""
===============================================================================
generation_sampling.py
===============================================================================
Generate deterministic configured design-of-experiments values.
Responsibilities:
  - Produce uniform, Latin-hypercube, and scrambled Sobol unit designs
  - Apply the maintained log, logit, linear, phase, integer, and softmax maps
  - Resolve sampled and explicit case-level generator or scalar overrides
Design principles:
  - Parameter inventory and ranges remain configuration-owned
  - Python stream determinism is bound to the explicit generator identity
  - Explicit case overrides take precedence over batch sampling
This module does NOT:
  - Define drying parameters, grain families, or scientific ranges
  - Generate spatial fields or persist case files
===============================================================================
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial import cKDTree  # pyright: ignore[reportAttributeAccessIssue] -- SciPy exports it without complete typing
from scipy.stats import qmc

if TYPE_CHECKING:
    from . import generation_config as config_contract


LHS_MAXIMIN_CANDIDATES = 50
SOBOL_SKIP = 1000
SOBOL_LEAP = 200
NEAREST_NEIGHBOR_COUNT = 2


def _base_value(
    config: config_contract.GenerationConfig,
    *,
    target: str,
    name: str,
    component: int | None,
) -> float:
    """Return one configured finite sampling base value or list component."""
    if target == "generator":
        value = config.values["generator"]["parameters"][name]
        if component is not None:
            value = value[component]
    else:
        entries = config.values["inputs"]["scalar_file"]["entries"]
        value = next(entry["value"] for entry in entries if entry["name"] == name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        message = f"Sampling base {target}.{name} must be one finite scalar, got {value!r}."
        raise ValueError(message)
    return float(value)


def _lhs_design(*, count: int, dimensions: int, seed: int) -> np.ndarray:
    """Select the most separated of the maintained 50 LHS candidates."""
    best: np.ndarray | None = None
    best_separation = -math.inf
    for iteration in range(LHS_MAXIMIN_CANDIDATES):
        candidate_rng = np.random.default_rng(np.random.SeedSequence([seed, iteration]))
        candidate = qmc.LatinHypercube(d=dimensions, scramble=True, rng=candidate_rng).random(count)
        if count < NEAREST_NEIGHBOR_COUNT:
            separation = math.inf
        else:
            distances, _indices = cKDTree(candidate).query(candidate, k=NEAREST_NEIGHBOR_COUNT)
            separation = float(np.min(distances[:, 1]))
        if separation > best_separation:
            best = candidate
            best_separation = separation
    if best is None:
        message = "Latin-hypercube generation produced no candidate design."
        raise RuntimeError(message)
    return best


def _sobol_design(*, count: int, dimensions: int, seed: int) -> np.ndarray:
    """Return a scrambled Sobol design with maintained skip and leap values."""
    sampler = qmc.Sobol(d=dimensions, scramble=True, rng=seed)
    sampler.fast_forward(SOBOL_SKIP)
    design = np.empty((count, dimensions), dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for row in range(count):
            design[row] = sampler.random(1)[0]
            if row + 1 < count:
                sampler.fast_forward(SOBOL_LEAP)
    return design


def _unit_design(method: str, *, count: int, dimensions: int, seed: int) -> np.ndarray:
    """Return one deterministic Python-owned unit design."""
    if dimensions == 0:
        return np.empty((count, 0), dtype=np.float64)
    if method == "uniform":
        return np.random.default_rng(seed).random((count, dimensions))
    if method == "lhs":
        return _lhs_design(count=count, dimensions=dimensions, seed=seed)
    return _sobol_design(count=count, dimensions=dimensions, seed=seed)


def _map_parameter(base: float, z: np.ndarray, *, variation: float, spec: dict[str, Any]) -> np.ndarray:
    """Apply one maintained sampling transform to a centered unit coordinate."""
    transform = spec["transform"]
    span = math.log1p(variation)
    if transform == "log":
        if base <= 0:
            message = f"Log-space sampling requires a positive base, got {base}."
            raise ValueError(message)
        return base * np.exp(span * z)
    if transform == "logit":
        if not 0 < base < 1:
            message = f"Logit-space sampling requires a base strictly inside (0, 1), got {base}."
            raise ValueError(message)
        logit = math.log(base / (1.0 - base))
        return 1.0 / (1.0 + np.exp(-(logit + span * z)))
    if transform == "linear":
        return base * (1.0 + variation * z)
    if transform == "phase":
        return np.mod(base + variation * math.pi * z, 2.0 * math.pi)
    if transform == "integer":
        scale = float(spec.get("scale", 3.0))
        values = np.rint(base + np.rint(variation * scale * z))
        if "minimum" in spec:
            values = np.maximum(values, float(spec["minimum"]))
        if "maximum" in spec:
            values = np.minimum(values, float(spec["maximum"]))
        return values
    if transform == "softmax":
        if base <= 0:
            message = f"Softmax sampling requires positive base weights, got {base}."
            raise ValueError(message)
        return np.exp(math.log(base) + span * z)
    message = f"Unsupported sampling transform: {transform!r}."
    raise ValueError(message)


def sample_case_overrides(config: config_contract.GenerationConfig) -> dict[int, dict[str, dict[str, Any]]]:
    """
    Return deterministic sampled values for every configured case.

    NumPy/SciPy streams and transforms are owned by the configured Python
    generator identity.
    """
    sampling = config.values.get("sampling")
    empty: dict[int, dict[str, dict[str, Any]]] = {case_index: {"generator": {}, "scalar": {}} for case_index in config.case_indices}
    if sampling is None or not sampling["parameters"]:
        return empty
    parameters = sampling["parameters"]
    design = _unit_design(
        sampling["method"],
        count=len(config.case_indices),
        dimensions=len(parameters),
        seed=config.seed_base,
    )
    centered = 2.0 * design - 1.0
    mapped: dict[tuple[str, str, int | None], np.ndarray] = {}
    softmax_groups: dict[str, list[tuple[str, str, int | None]]] = {}
    for column, spec in enumerate(parameters):
        key = (spec["target"], spec["name"], spec.get("index"))
        base = _base_value(config, target=key[0], name=key[1], component=key[2])
        mapped[key] = _map_parameter(base, centered[:, column], variation=sampling["variation"], spec=spec)
        if spec["transform"] == "softmax":
            softmax_groups.setdefault(spec["group"], []).append(key)
    for group, keys in softmax_groups.items():
        if len(keys) < NEAREST_NEIGHBOR_COUNT:
            message = f"Softmax sampling group {group!r} must contain at least two parameters."
            raise ValueError(message)
        denominator = np.sum(np.stack([mapped[key] for key in keys], axis=1), axis=1)
        for key in keys:
            mapped[key] = mapped[key] / denominator

    result: dict[int, dict[str, dict[str, Any]]] = {case_index: {"generator": {}, "scalar": {}} for case_index in config.case_indices}
    for row, case_index in enumerate(config.case_indices):
        for (target, name, component), values in mapped.items():
            sampled_value = float(values[row])
            if component is None:
                result[case_index][target][name] = sampled_value
                continue
            components = result[case_index][target].setdefault(
                name,
                list(config.values["generator"]["parameters"][name]),
            )
            components[component] = sampled_value
    return result


def resolve_case_values(
    config: config_contract.GenerationConfig,
    case_index: int,
    *,
    sampled_values: dict[int, dict[str, dict[str, Any]]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[list[float]] | None]:
    """Resolve generator parameters, scalar entries, and optional schedule rows."""
    config.case_id(case_index)
    resolved_samples = sample_case_overrides(config) if sampled_values is None else sampled_values
    sampled = resolved_samples[case_index]
    parameters = dict(config.values["generator"]["parameters"])
    parameters.update(sampled["generator"])

    scalar_spec = config.values["inputs"].get("scalar_file")
    entries = [] if scalar_spec is None else [dict(entry) for entry in scalar_spec["entries"]]
    scalar_by_name = {entry["name"]: entry for entry in entries}
    for name, value in sampled["scalar"].items():
        scalar_by_name[name]["value"] = value

    schedule_spec = config.values["inputs"].get("schedule_file")
    schedule_rows = None if schedule_spec is None else [list(row) for row in schedule_spec["rows"]]
    explicit = config.overrides.get(case_index, {})
    raw_generator = explicit.get("generator", {})
    if not isinstance(raw_generator, dict) or any(name not in parameters for name in raw_generator):
        message = f"Case {case_index} generator override references an unknown parameter."
        raise ValueError(message)
    parameters.update(raw_generator)
    raw_scalars = explicit.get("scalars", {})
    if not isinstance(raw_scalars, dict) or any(name not in scalar_by_name for name in raw_scalars):
        message = f"Case {case_index} scalar override references an unknown scalar."
        raise ValueError(message)
    for name, value in raw_scalars.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            message = f"Case {case_index} scalar override {name!r} must be finite."
            raise ValueError(message)
        scalar_by_name[name]["value"] = float(value)
    if "schedule_rows" in explicit:
        rows = explicit["schedule_rows"]
        if schedule_spec is None:
            message = f"Case {case_index} cannot override a schedule when inputs.schedule_file is absent."
            raise ValueError(message)
        if not isinstance(rows, list):
            message = f"Case {case_index} schedule_rows override must be a list."
            raise TypeError(message)
        schedule_rows = [list(row) if isinstance(row, list) else [] for row in rows]
    return parameters, entries, schedule_rows
