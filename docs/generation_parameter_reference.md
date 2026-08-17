# Generation Scientific Parameter Reference

This document explains the scientific meaning, sampling intent, equations,
evidence classes, and modelling assumptions of Generation. Current values and
supports remain authoritative only in validated YAML under
`configs/generation`; operational commands and recovery procedures belong in
the [Generation workflow](simulation_generation.md).

## Evidence and interpretation

Configured values are executable modelling and sampling decisions, not universal
material constants. A value may be directly reported, refitted, converted,
transferred, estimated, calibrated, derived, or selected as a synthetic design
assumption. A citation therefore does not imply that the final configured value
appears verbatim in the cited source.

Each authored value or record carries:

- an `evidence` classification describing the basis of the value;
- `source_refs` linking it to `configs/generation/sources.yaml`;
- a method or applicability limit where needed to interpret a conversion,
  transfer, fit, or restricted evidence domain.

An empty source list is valid for an explicit project or synthetic decision.
Material OOD supports have separate OOD provenance, so literature support for a
natural value is never presented as evidence for a synthetic stress interval.
Technical Smoke and Pilot validate software and data flow; they do not upgrade
scientific evidence. Campaign purpose is explicit scientific provenance in every
resolved batch and contributes to its scientific identity.

## Authoritative owners

| Scientific concern | Owner |
| --- | --- |
| Grid, time, fixed values, formulas, numerical validation | `configs/generation/common.yaml` |
| Parameter definitions, units, transforms, profiles, OOD groups | `configs/generation/registry.yaml` |
| Material-specific values, supports, records, provenance | `configs/generation/materials/*.yaml` |
| Operation and inlet-schedule supports and constraints | `configs/generation/operations/fixed_bed.yaml` |
| Family roles, sampling counts, membership, seeds | Selected campaign YAML |
| Bibliographic metadata and locators | `configs/generation/sources.yaml` |

Inspect the validated effective configuration rather than treating mutable prose
as the owner of resolved values. The operator workflow explains how to validate
and inspect a campaign.

## Parameter families

The registry is the parameter catalogue; the table below explains stable
families rather than freezing its current entry count or profile dimensions.

| Family | Representative quantities and units | Scientific role |
| --- | --- | --- |
| Permeability | `kappa_mean` (`m^2`), coefficient of variation, anisotropy | Sets the mean and spatial structure of the heterogeneous airflow resistance |
| Bed morphology | relative correlation lengths, multiscale weights, orientation, porosity texture | Defines synthetic spatial heterogeneity around material-calibrated bed properties |
| Pressure boundary | mean pressure (`Pa`), spatial sinusoidal/Gaussian/trend terms | Generates the inlet pressure field for Darcy–Brinkman airflow |
| Initial moisture | dry-basis mean/amplitude (`kg water/kg dry solid`) and spatial structure | Generates the transient initial grain-water field |
| Inlet schedule | temperature (`K`), humidity ratio (`kg/kg`), correlation, timescales, events, trend | Generates time-dependent drying-air forcing |
| Density and thermal properties | dry bulk density (`kg/m^3`), porosity, conductivity (`W/(m K)`), heat capacity (`J/(kg K)`) | Couples material and packing state to transient heat and storage terms |
| Sorption and kinetics | Oswin coefficients, surface rate (`1/s`), internal/surface rate ratio, surface-water fraction | Defines equilibrium moisture and two-compartment transfer/evaporation behavior |
| Fixed and derived quantities | pressure/temperature references, vapour diffusivity, latent heat, wall transfer, target moisture | Closes the model without adding independent DOE coordinates |

Material identity is metadata, not an implicit one-hot model input. Derived
quantities, coupled records, and latent packing scatter are not independent DOE
coordinates.

## Sampling and OOD semantics

Natural support is the intended in-distribution region for a material and
profile. Detailed supports and nominals are authored in the operation and
material YAML; transforms and OOD eligibility are registry-owned.

Parameter OOD activates one eligible physical unit outside its natural support.
A unit may be one scalar tail or one complete coupled record. Scalar tails must
be separated from natural support in the declared transform. Coupled records
such as density calibration, sorption, kinetics, and schedule weights are
selected atomically; components from different records are never mixed.

Family OOD withholds material families while keeping their parameters on natural
support. It does not combine family shift with parameter-OOD stress. Campaign
YAML owns which families are seen or held out and how cases are allocated; these
current memberships are experiment design, not permanent scientific definitions.

Natural and parameter-OOD schedules use the same stochastic process family with
different supports. Natural and permeability-OOD packing states use the same
material-calibrated relation, with OOD porosity supports mapped so the realized
packing state remains outside the natural interval.

## Scientific model

### Airflow and spatial fields

The preserved steady profile solves Darcy–Brinkman airflow from heterogeneous
permeability `kappa(x)`, porosity `epsilon(x)`, and inlet pressure
`p_in_bc(y)`. Generation controls their means, variation, multiscale spatial
structure, anisotropy, and physical bounds. These field-shape choices are
project or synthetic generator assumptions unless their resolved provenance
states otherwise.

The transient profile reuses the airflow state and adds temperature, water-vapour
transport, heat transfer, and local grain-moisture dynamics. The initial
moisture field is expressed on a dry basis; wet-basis and dry-basis quantities
remain explicit:

```text
X_wb = X_db / (1 + X_db)
X_db = X_wb / (1 - X_wb)
```

### Material-calibrated Kozeny--Carman trend and packing scatter

Mean permeability and packing porosity are coupled through a
material-calibrated Kozeny–Carman reference trend. Define

```text
g(epsilon) = epsilon^3 / (1 - epsilon)^2
A_KC_reference = kappa_nominal / g(eps_bed_cal_ref)
eps_kc_trend = g^-1(kappa_mean / A_KC_reference)
```

`kappa_mean` is the DOE coordinate. The canonical material reference fixes
`A_KC_reference`, and the monotonic trend supplies the corresponding packing
porosity. The effective natural permeability support is the intersection of the
authored support with the range compatible with the material's natural porosity
support; `validate-config` shows both authored and resolved intervals.
Permeability-OOD tails are mapped through the same relation to consistent
porosity-OOD regions.

A small bounded case-level packing scatter represents unresolved morphology.
It is a synthetic modelling assumption, not an experimentally calibrated
parameter or an independent model input. Local porosity heterogeneity remains a
spatial field around that case reference. No pointwise Kozeny–Carman equality is
imposed between `kappa(x)` and `epsilon(x)`; the relation is a global
material trend only.

### Inlet schedule

Generation creates temporal signals `T_in_bc(t)` and
`omega_in_bc(t)`. Relative humidity `phi_in_bc(t)` is derived
thermodynamically from temperature, humidity ratio, and reference pressure; it
is not sampled independently.

Schedule variability combines:

- smooth correlated variation;
- a finite configured number of smooth-edged events;
- a horizon-scale trend;
- deliberate correlation between temperature and humidity-ratio signals.

The configured bases are exact temporal means and the amplitudes are exact
maximum absolute deviations. The symmetric temperature-amplitude contract is
retained: it does not prescribe equal positive and negative extrema, and it does
not insert a deterministic low-temperature phase. The current natural
`T_in_amp` support is `0-8 K`; its hard-boundary parameter-OOD tail is
`9.25-10 K`. Together with the natural `T_in_base` support
`303.15-309.15 K`, the authored natural support can reach about `295.15 K`
before whole-schedule feasibility is applied. The canonical inlet-temperature
envelope is `290.15-313.15 K`. Ambient temperature, the stochastic shape,
humidity, and the static heater and psychrometric envelopes determine which
complete candidates are accepted, so not every case realizes a low interval.

A generated schedule must satisfy temperature, humidity-ratio, inlet-relative-
humidity, source-air saturation, and heater-only constraints. An infeasible
candidate is rejected and deterministically resampled as a whole rather than
repaired by pointwise clipping. Schedule supports and composition are synthetic
design assumptions, not literature measurements or active control.

The accepted stochastic schedule remains canonical on
`common.time.regular_times`. The fixed-bed operation applies a separate COMSOL
handoff policy from `boundary_schedule.startup_ramp`: `enabled` selects the
transformation, and `duration_h` is the physical boundary-startup duration in
hours. It must lie strictly between zero and the first regular interval. The
policy uses no bed-state or solver feedback and does not change regular COMSOL
output, HDF5, or Dataset state times.

At zero, `omega_in_bc` remains exactly the canonical incoming-air humidity
ratio. The schedule psychrometric owner solves the maintained Magnus conversion
for the minimum temperature satisfying `phi_in_bc <= phi_operational_max`, then
uses `T_start = max(T_init, T_in_min, T_required)`. This is a static configuration
constraint: it uses no bed state, solver value, or runtime feedback. Generation
fails closed when the humidity ratio is invalid, the configured maximum
temperature cannot satisfy the RH limit, or the safe start is above the required
heating-ramp rejoin state. Persisted startup evidence records the quantities
needed to reproduce and verify this boundary condition.

The rejoin temperature and humidity ratio are the original canonical linear
interpolation at the configured startup duration; rejoin relative humidity is
recomputed thermodynamically from those two values. Every physical handoff node
must lie inside the persisted operational temperature, humidity-ratio, and RH
envelopes, and every retained regular node remains unchanged.

The extra row is boundary interpolation support, not a regular output state.
COMSOL output, HDF5 state time, and Dataset transitions continue to use only
`common.time.regular_times`. The final schedule bytes and handoff provenance
participate in exact input identity. Disabling the ramp retains the canonical
table values and times exactly. Absolute temperatures are persisted in kelvin;
Celsius conversion is a
presentation concern, while amplitudes, changes, differences, and rates remain
in K or K/h.

### Moisture, heat, and equilibrium coupling

Granular water is divided between surface and internal compartments:

```text
f_surf d(w_surf)/dt       = j_int - m_evap
(1 - f_surf) d(w_int)/dt  = -j_int
w_gr = f_surf w_surf + (1 - f_surf) w_int
j_int = (1 - f_surf) r_int (w_int - w_surf)
m_evap = f_surf r_surf max(w_surf - w_eq, 0)
```

The equilibrium water content uses the configured Oswin record:

```text
X_eq_db = 0.01 [A_osw + B_osw (T - 273.15 K)]
          [phi_eff / (1 - phi_eff)]^C_osw
w_eq = rho_bu_dry X_eq_db
```

The local initial equilibrium relative humidity follows from the exact
algebraic inverse of that same relation:

```text
R_0 = [100 X_0_db_field /
       (A_osw + B_osw (T_init - 273.15 K))]^(1/C_osw)
phi_eq_0 = R_0 / (1 + R_0)
```

This derived quantity interprets the initial-moisture field at
`T_init = T_amb`; it is neither a generated field nor an inlet boundary, and it
does not apply the forward solver's numerical humidity clip.

Dry bulk density and initial granular water follow the realized packing state:

```text
rho_bu_dry =
  rho_bu_dry_ref (1 - eps_bed) / (1 - eps_bed_cal_ref)
w_gr_0 = rho_bu_dry X_0_db_field
```

Material heat capacity and conductivity provide effective storage and transport,
while evaporation contributes the latent heat sink `Q_evap = -h_fg m_evap`.
The bulk water-balance diagnostic compares change in stored grain and gas water
with inlet and outlet vapour flow.

These equations are modelling bindings. The resolved provenance of each
coefficient determines whether it is sourced, transferred, estimated, calibrated,
derived, or synthetic.

## Material records and assumptions

Every material uses the same role-neutral schema. Campaigns assign learning and
evaluation roles separately. The material files retain the complete natural and
OOD records, source references, methods, and applicability limits.

Important assumptions are:

- natural values and stress supports are material-specific, but the schema and
  validation rules are shared;
- density OOD selects a complete density/porosity calibration record;
- sorption and kinetics select complete records with their equation convention
  or applicability domain;
- schedule component weights are one constrained record;
- a runtime pass cannot convert a synthetic or transferred value into
  experimental evidence;
- exact current supports, mapped porosity intervals, and campaign memberships
  must be read from resolved configuration.

## Numerical validation contracts

Transient initial-state consistency, transient bulk-moisture consistency, and
float32 storage fidelity use separate tolerance classes because they protect
different numerical contracts. Initial-state validation compares the exported
state with its prescribed moisture definition; bulk-moisture validation compares
the exported diagnostic with integrated dry and water mass; storage fidelity
bounds float64-to-float32 publication error.

`configs/generation/common.yaml` is the authoritative owner of the current
values. Keeping the classes separate prevents a storage-format tolerance from
silently weakening a physical consistency check, or vice versa. Schedule
correlation and temporal-resolution checks are generator invariants and remain
implementation-owned.

## Provenance and persistence

Resolved configuration and generated artifacts preserve material family and
role, natural or OOD selection, selected values or complete records, scientific
configuration, relevant diagnostics, and content identities. Exact admission
rejects stale or structurally incompatible artifacts, and ordinary reads never
rewrite them. Transient input identity includes resolved schedule and startup
semantics; steady-flow identity excludes those transient dependencies.

## Source attribution

`configs/generation/sources.yaml` is the sole machine-readable bibliography and
locator owner. Material YAML and resolved `source_refs` provide the exact
value-to-source chain; a material-level citation must not be interpreted as
support for every parameter of that material.

## Operational cross-reference

See the [Generation workflow](simulation_generation.md) for all operator
commands, execution stages, monitoring, recovery, and cleanup. The project-level
entry point is the [README](../README.md).
