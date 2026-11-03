"""Derive human architecture names from model configuration metadata.

Responsibilities:
  - Describe Fourier modes, hidden width, depth and UNO spatial schedules
  - Reuse model-owned defaults without importing or constructing neural networks
  - Provide the core label for configuration filenames and run navigation and the full label for model views

Design principles:
  - Labels describe architecture; exact saved identities detect scientific collisions
  - Dataset membership, training seeds, losses and stages remain separate attributes
  - Formatting is deterministic and never changes caller-owned configuration

This module does NOT:
  - Allocate paths, choose training settings or execute models
  - Replace model validation or identify exact scientific compatibility
"""

from collections.abc import Mapping
from typing import Any

from src import common

from . import learning_models_factory as factory


def _architecture_parts(config: Mapping[str, Any]) -> tuple[str, str]:
    """Return the core descriptor and the non-default switch suffix (empty when none)."""
    model = config["model"]
    kind = model["kind"]
    params = {**factory.model_defaults(kind), **model["params"]}
    modes = (params["modes_x"], params["modes_y"]) if kind == "uno" else params["n_modes"]
    number, text = common.names.number_token, common.names.text_token
    label = f"{kind}_m{number(modes[0])}x{number(modes[1])}_h{number(params['hidden_channels'])}_l{number(params['n_layers'])}"
    core = {"n_modes", "modes_x", "modes_y", "hidden_channels", "n_layers", "in_channels", "out_channels"}
    if kind == "uno":
        scales = factory.resolve_uno_scalings(params["n_layers"], params.get("uno_scalings"))
        tokens = []
        for x, y in scales:
            # Preserve the familiar isotropic 0.5 -> 05 schedule notation.
            a, b = number(x).replace("p", ""), number(y).replace("p", "")
            tokens.append(a if x == y else f"{a}x{b}")
        label += "_s" + "-".join(tokens) + "_r" + number(params.get("mode_ratio", 0.5), precision=3)
        core.update({"uno_scalings", "mode_ratio"})
    defaults = factory.model_defaults(kind)
    switches = []
    for key, value in sorted(params.items()):
        if key in core or value is None or (key in defaults and value == defaults[key]):
            continue
        switches.append(key.replace("_", "-") + "-" + text(value))
    return label, "_".join(switches)


def core_architecture_label(config: Mapping[str, Any]) -> str:
    """Return the architecture descriptor without model-switch suffix.

    Parameters
    ----------
    config : Mapping[str, Any]
        Authored or resolved experiment containing ``model.kind`` and ``params``.

    Returns
    -------
    str
        Filename-safe kind, Fourier-mode, width and depth descriptor (plus the UNO
        scaling schedule and mode ratio). Non-default model switches are omitted;
        :func:`architecture_label` appends them for display.

    """
    return _architecture_parts(config)[0]


def architecture_label(config: Mapping[str, Any]) -> str:
    """Return the shared readable architecture descriptor without model execution.

    Parameters
    ----------
    config : Mapping[str, Any]
        Authored or resolved experiment containing ``model.kind`` and ``params``.

    Returns
    -------
    str
        Filename-safe architecture descriptor. UNO ratios are displayed to three
        decimals; full parameters and exact identities remain in the saved config.

    """
    label, switches = _architecture_parts(config)
    return label + ("__" + switches if switches else "")
