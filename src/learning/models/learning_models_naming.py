"""Derive human architecture names from model configuration metadata.

Responsibilities:
  - Describe Fourier modes, hidden width, depth, domain padding and UNO spatial schedules
  - Name every model parameter that deviates from its model-owned default
  - Reuse model-owned defaults without importing or constructing neural networks
  - Provide the one label for configuration filenames, run navigation and model views

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

# Parameters the label names positionally; every other deviating parameter becomes a ``__<key>-<value>`` token.
_POSITIONAL_PARAMS = frozenset({"n_modes", "modes_x", "modes_y", "hidden_channels", "n_layers", "in_channels", "out_channels", "domain_padding"})


def architecture_label(config: Mapping[str, Any]) -> str:
    """Return the shared readable architecture descriptor without model execution.

    Parameters
    ----------
    config : Mapping[str, Any]
        Authored or resolved experiment containing ``model.kind`` and ``params``.

    Returns
    -------
    str
        Filename-safe kind, Fourier-mode, width, depth and domain-padding descriptor
        (``_p0p1``; no padding is ``_p0``), plus the UNO scaling schedule and mode ratio
        and one ``__<key>-<value>`` token per other parameter that differs from the
        model default. UNO ratios are displayed to three decimals; full parameters
        and exact identities remain in the saved config.

    """
    model = config["model"]
    kind = model["kind"]
    defaults = factory.model_defaults(kind)
    params = {**defaults, **model["params"]}
    # Y before X for every kind, as the FNO and RNO n_modes order the grid axes.
    modes = (params["modes_y"], params["modes_x"]) if kind == "uno" else params["n_modes"]
    number, text = common.names.number_token, common.names.text_token
    label = f"{kind}_m{number(modes[0])}x{number(modes[1])}_h{number(params['hidden_channels'])}_l{number(params['n_layers'])}"
    padding = params["domain_padding"]
    label += f"_p{number(0 if padding is None else padding)}"  # a boolean or non-scalar padding raises ValueError
    named = set(_POSITIONAL_PARAMS)
    if kind == "uno":
        scales = factory.resolve_uno_scalings(params["n_layers"], params.get("uno_scalings"))
        tokens = []
        for y, x in scales:  # one scaling per grid axis, y before x like the modes
            # A value below 1 keeps the familiar 0.5 -> 05 notation (its leading zero marks the decimal); 1.5 stays 1p5, never 15.
            a, b = (number(v).replace("p", "") if v < 1 else number(v) for v in (y, x))
            tokens.append(a if y == x else f"{a}x{b}")
        label += "_s" + "-".join(tokens) + "_r" + number(params.get("mode_ratio", 0.5), precision=3)
        named.update({"uno_scalings", "mode_ratio"})
    if kind == "rno":
        # The RNO accepts only the tanh candidate, so the value names no alternative.
        named.add("candidate_activation")
    switches = [
        f"{key.replace('_', '-')}-{text(value)}"
        for key, value in sorted(params.items())
        if key not in named and value is not None and not (key in defaults and value == defaults[key])
    ]
    return "__".join([label, *switches])
