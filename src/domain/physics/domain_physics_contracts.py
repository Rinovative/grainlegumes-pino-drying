"""Define dependency-light semantic identifiers and the air viscosity for steady-flow physics.

Responsibilities:
  - Own canonical Brinkman equation, boundary, continuity, and derivative identifiers
  - Own the one air viscosity shared by the Brinkman residuals, their diagnostics, and Generation's inlet-pressure derivation
  - Validate derivative selectors without tensor implementations

Design principles:
  - Task declarations and runtime equations share one identifier source
  - Configuration admission and Generation remain independent of Torch-backed computation
  - Exact identifiers fail closed without aliases or compatibility fallbacks

This module does NOT:
  - Implement residual equations, derivatives, masks, scales, or scalar reductions
  - Import Torch or allocate tensors
  - Select experiment loss weights or diagnostic schedules
"""

from __future__ import annotations

from typing import Final, Literal, cast

DerivativeKind = Literal["physical", "spectral"]
# COMSOL "Air" eta(T) evaluated at the stationary template's T_flow_ref = 300.65 K
# (27.5 degC); the template takes viscosity from the material at that temperature.
AIR_DYNAMIC_VISCOSITY: Final = 1.85075e-5
STEADY_BRINKMAN_KIND: Final = "steady_2d_brinkman"
STEADY_BRINKMAN_EQUATION_SET: Final = "steady_two_dimensional_brinkman"
CONTINUITY_KIND: Final = "cross_section_flux_balance"
PRESSURE_BOUNDARY_KIND: Final = "pressure_inlet_zero_pressure_outlet"
_DERIVATIVE_KINDS: Final = ("physical", "spectral")

__all__ = [
    "AIR_DYNAMIC_VISCOSITY",
    "CONTINUITY_KIND",
    "PRESSURE_BOUNDARY_KIND",
    "STEADY_BRINKMAN_EQUATION_SET",
    "STEADY_BRINKMAN_KIND",
    "DerivativeKind",
    "validate_derivative_kind",
]


def validate_derivative_kind(kind: str) -> DerivativeKind:
    """Validate and return one exact derivative kind."""
    if kind not in _DERIVATIVE_KINDS:
        available = ", ".join(_DERIVATIVE_KINDS)
        msg = f"Unknown derivative identifier {kind!r}. Available derivatives: {available}."
        raise ValueError(msg)
    return cast("DerivativeKind", kind)
