"""
generation_input_table_schema.py

Define compact grouped labels for generation-input EDA tables.
Responsibilities:
  - Group persisted parameters into airflow and drying presentation families
  - Shorten repeated parameter, field-summary, and boundary labels
  - Expand schedule component weights into independently comparable rows
  - Preserve deterministic category and row ordering across table views
Design principles:
  - Presentation metadata never changes persisted names or scientific values
  - Unknown future rows remain visible in an explicit uncategorized group
  - Units remain owned by diagnostics and appear once in displayed row labels
This module does NOT:
  - Read generation inputs, calculate summaries, or render notebook widgets
  - Define parameter validity, schedule science, or persistence contracts
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final


@dataclass(frozen=True, slots=True)
class TableRowSpec:
    """Describe one stable table section, category, and compact row label."""

    section: str
    category: str
    label: str


AIRFLOW_SECTION: Final = "Airflow"
DRYING_SECTION: Final = "Drying"
UNCATEGORIZED_SECTION: Final = "Other"
COMPONENT_WEIGHT_NAMES: Final = ("smooth", "event", "trend")


def _rows(
    section: str,
    category: str,
    entries: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, TableRowSpec], ...]:
    """Return ordered canonical names paired with one shared category."""
    return tuple((name, TableRowSpec(section, category, label)) for name, label in entries)


_PARAMETER_ROWS: Final = (
    *_rows(
        AIRFLOW_SECTION,
        "Bed structure",
        (
            ("bed.structure.coarse_len_rel", "Coarse corr. length"),
            ("bed.structure.fine_len_rel", "Fine corr. length"),
            ("bed.structure.coarse_weight", "Coarse weight"),
            ("bed.structure.fine_weight", "Fine weight"),
            ("bed.structure.cross_scale_corr", "Cross-scale corr."),
            ("bed.structure.fine_ani_x", "Fine anisotropy x"),
            ("bed.structure.fine_ani_y", "Fine anisotropy y"),
        ),
    ),
    *_rows(
        AIRFLOW_SECTION,
        "Bed perturb.",
        (
            ("bed.perturbations.amplitude", "Amplitude"),
            ("bed.perturbations.granularity", "Granularity"),
            ("bed.perturbations.sign_bias", "Sign bias"),
        ),
    ),
    *_rows(
        AIRFLOW_SECTION,
        "Permeability",
        (
            ("kappa_mean", "Mean"),
            ("kappa_cv", "Coefficient of variation"),
            ("permeability.anisotropy.max_ratio", "Max. anisotropy ratio"),
            ("permeability.anisotropy.exponent", "Anisotropy exponent"),
            ("permeability.anisotropy.strength", "Anisotropy strength"),
            ("permeability.orientation.jitter", "Orientation jitter"),
            ("permeability.orientation.smooth_len_rel", "Orientation smooth length"),
        ),
    ),
    *_rows(
        AIRFLOW_SECTION,
        "Porosity",
        (
            ("porosity.smooth_len_rel", "Smooth length"),
            ("porosity.texture_amp", "Texture amplitude"),
            ("eps_bed_cal_ref", "Calibration reference"),
            ("eps_min_global", "Global minimum"),
            ("eps_max_global", "Global maximum"),
        ),
    ),
    *_rows(
        AIRFLOW_SECTION,
        "Inlet pressure",
        (
            ("pressure_bc.mean", "Mean"),
            ("pressure_bc.sin_amp", "Sine amplitude"),
            ("pressure_bc.sin_freq", "Sine frequency"),
            ("pressure_bc.sin_phase", "Sine phase"),
            ("pressure_bc.gauss_count", "Gaussian count"),
            ("pressure_bc.gauss_amp", "Gaussian amplitude"),
            ("pressure_bc.gauss_width", "Gaussian width"),
            ("pressure_bc.gauss_jitter", "Gaussian jitter"),
            ("pressure_bc.linear_amp", "Linear amplitude"),
        ),
    ),
    *_rows(
        DRYING_SECTION,
        "Initial moisture",
        (
            ("initial_moisture.mean_db", "Mean, dry basis"),
            ("initial_moisture.amplitude_db", "Amplitude, dry basis"),
            ("initial_moisture.structure.coarse_len_rel", "Coarse corr. length"),
            ("initial_moisture.structure.fine_len_rel", "Fine corr. length"),
            ("initial_moisture.structure.coarse_weight", "Coarse weight"),
            ("initial_moisture.structure.fine_weight", "Fine weight"),
            ("initial_moisture.structure.cross_scale_corr", "Cross-scale corr."),
            ("initial_moisture.structure.fine_ani_x", "Fine anisotropy x"),
            ("initial_moisture.structure.fine_ani_y", "Fine anisotropy y"),
        ),
    ),
    *_rows(
        DRYING_SECTION,
        "Initial state / sorption",
        (
            ("T_init", "Initial temperature"),
            ("X_target_wb", "Target moisture, wet basis"),
            ("A_osw", "Oswin A"),
            ("B_osw", "Oswin B"),
            ("C_osw", "Oswin C"),
        ),
    ),
    *_rows(
        DRYING_SECTION,
        "Material / transfer",
        (
            ("rho_bu_dry_ref", "Dry bulk-density reference"),
            ("k_gr", "Granular conductivity"),
            ("cp_gr_dry", "Dry granular heat capacity"),
            ("r_surf_0", "Base surface rate"),
            ("r_int_surf", "Internal/surface ratio"),
            ("f_surf", "Surface fraction"),
            ("r_surf", "Surface rate"),
            ("r_int", "Internal rate"),
        ),
    ),
    *_rows(
        DRYING_SECTION,
        "Inlet schedule",
        (
            ("T_in_base", "Temperature baseline"),
            ("T_in_amp", "Temperature amplitude"),
            ("omega_in_base", "Humidity-ratio baseline"),
            ("omega_in_amp", "Humidity-ratio amplitude"),
            ("schedule.corr", "Temperature/humidity corr."),
            ("schedule.timescale_rel", "Smooth timescale"),
            ("schedule.component_weights.smooth", "Smooth component weight"),
            ("schedule.component_weights.event", "Event component weight"),
            ("schedule.component_weights.trend", "Trend component weight"),
            ("schedule.event_count", "Event count"),
            ("schedule.event_duration_rel", "Event duration"),
            ("schedule.event_width_rel", "Event width"),
            ("T_amb", "Ambient temperature"),
        ),
    ),
)
_PARAMETER_SPECS: Final = MappingProxyType(dict(_PARAMETER_ROWS))
PARAMETER_ORDER: Final = tuple(name for name, _spec in _PARAMETER_ROWS)

_FIELD_ROWS: Final = (
    *_rows(
        AIRFLOW_SECTION,
        "Bed and pressure",
        (
            ("eps_bed min", "Bed porosity — min"),
            ("eps_bed mean", "Bed porosity — mean"),
            ("eps_bed max", "Bed porosity — max"),
            ("p_in_bc min", "Inlet pressure — min"),
            ("p_in_bc mean", "Inlet pressure — mean"),
            ("p_in_bc max", "Inlet pressure — max"),
        ),
    ),
    *_rows(
        AIRFLOW_SECTION,
        "Permeability tensor",
        (
            ("Kxx median", "Kxx — median"),
            ("Kxy median", "Kxy — median"),
            ("Kyy median", "Kyy — median"),
        ),
    ),
    *_rows(
        AIRFLOW_SECTION,
        "Principal permeability",
        (
            ("K_min median", "Minimum — median"),
            ("K_max median", "Maximum — median"),
            ("K_anisotropy q95", "Anisotropy — 95th pct."),
        ),
    ),
    *_rows(
        DRYING_SECTION,
        "Moisture / sorption",
        (
            ("X_0_db_field min", "Dry-basis moisture — min"),
            ("X_0_db_field mean", "Dry-basis moisture — mean"),
            ("X_0_db_field max", "Dry-basis moisture — max"),
            ("phi_eq mean", "Equilibrium RH — mean"),
            ("phi_eq q95", "Equilibrium RH — 95th pct."),
        ),
    ),
    *_rows(
        DRYING_SECTION,
        "Initial bulk state",
        (
            ("rho_bu_dry mean", "Dry bulk density — mean"),
            ("w_gr0 mean", "Granular water — mean"),
        ),
    ),
)
_FIELD_SPECS: Final = MappingProxyType(dict(_FIELD_ROWS))
FIELD_SUMMARY_ORDER: Final = tuple(name for name, _spec in _FIELD_ROWS)

_BOUNDARY_ROWS: Final = (
    *_rows(
        AIRFLOW_SECTION,
        "Reference state",
        (
            ("T_flow_ref", "Flow temperature"),
            ("p_ref", "Reference pressure"),
            ("p_out", "Outlet pressure"),
        ),
    ),
    *_rows(
        AIRFLOW_SECTION,
        "Inlet pressure",
        tuple(
            (f"p_in_bc {statistic}", f"{label}")
            for statistic, label in (
                ("min", "Minimum"),
                ("q05", "5th pct."),
                ("median", "Median"),
                ("mean", "Mean"),
                ("q95", "95th pct."),
                ("max", "Maximum"),
                ("std", "Std. deviation"),
            )
        ),
    ),
    *_rows(
        DRYING_SECTION,
        "Inlet temperature",
        (
            ("T_in_bc start", "Start"),
            ("T_in_bc startup end", "Startup end"),
            ("T_in_bc startup delta", "Startup change"),
        ),
    ),
    *_rows(
        DRYING_SECTION,
        "Humidity ratio",
        (
            ("omega_in_bc start", "Start"),
            ("omega_in_bc startup end", "Startup end"),
            ("omega_in_bc startup delta", "Startup change"),
        ),
    ),
    *_rows(
        DRYING_SECTION,
        "Relative humidity",
        (
            ("phi_in_bc start", "Start"),
            ("phi_in_bc startup end", "Startup end"),
            ("phi_in_bc startup delta", "Startup change"),
        ),
    ),
)
_BOUNDARY_SPECS: Final = MappingProxyType(dict(_BOUNDARY_ROWS))
BOUNDARY_ORDER: Final = tuple(name for name, _spec in _BOUNDARY_ROWS)


def _fallback(name: str) -> TableRowSpec:
    """Keep one unknown future row visible without claiming a category."""
    return TableRowSpec(
        UNCATEGORIZED_SECTION,
        "Uncategorized",
        name.replace("_", " "),
    )


def parameter_row_spec(name: str) -> TableRowSpec:
    """Return compact presentation metadata for one parameter row."""
    return _PARAMETER_SPECS.get(name, _fallback(name))


def field_summary_row_spec(quantity: str, statistic: str) -> TableRowSpec:
    """Return compact presentation metadata for one field-summary row."""
    name = f"{quantity} {statistic}"
    return _FIELD_SPECS.get(name, _fallback(name))


def boundary_row_spec(name: str) -> TableRowSpec:
    """Return compact presentation metadata for one boundary-summary row."""
    return _BOUNDARY_SPECS.get(name, _fallback(name))
