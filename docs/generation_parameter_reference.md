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
resolved batch. It contributes to the scientific digest and therefore to
`batch_id`; the human-readable `batch_storage_name` additionally spells out that
purpose without replacing or weakening the scientific identity.

## Authoritative owners

| Scientific concern | Owner |
| --- | --- |
| Grid, time, fixed values, formulas, numerical validation | `configs/generation/common.yaml` |
| Parameter definitions, units, transforms, profiles, OOD groups | `configs/generation/registry.yaml` |
| Material-specific values, supports, records, provenance | `configs/generation/materials/*.yaml` |
| Operation and inlet-schedule supports and constraints | `configs/generation/operations/fixed_bed.yaml` |
| Family roles, sampling counts, membership, seeds | Selected campaign YAML |
| Bibliographic metadata and locators | `configs/generation/sources.yaml` |
| Resolved supports, identities, and generated diagnostics | `validate-config`, `readiness-report`, `plan`, and generated receipts |
| Exact algorithms and invariants | `src/generation` and focused tests |

Inspect the effective scientific contract instead of copying resolved values into
prose:

```bash
./scripts/docker_python.sh -m src.generation.cli.cli_generation validate-config \
  configs/generation/campaigns/transient_drying/family_generalization.yaml \
  --allow-incomplete
```

The optional `--inspect-parameter <name>` argument follows authored, inherited,
selected, and derived provenance for a parameter or complete atomic record.

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

The input EDA reports exact principal permeabilities and their anisotropy ratio.
It uses Generation's positive-definiteness check and does not alter the tensor.

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

The accepted stochastic schedule remains canonical on `common.time.regular_times`
(`0 h, 1 h, ..., 168 h` for the current configuration). The fixed-bed operation
then applies a separate COMSOL handoff policy from
`boundary_schedule.startup_ramp`: `enabled` selects the transformation and
`duration_h` must lie strictly between zero and the regular interval. With the
default `duration_h: 0.16666666666666666`, the final interpolation table begins
at `0 h, 1/6 h, 1 h, ...`.

At zero, `omega_in_bc` remains exactly the canonical incoming-air humidity
ratio. The schedule psychrometric owner solves the maintained Magnus conversion
for the minimum temperature satisfying `phi_in_bc <= phi_operational_max`, then
uses `T_start = max(T_init, T_in_min, T_required)`. This is a static configuration
constraint: it uses no bed state, solver value, or runtime feedback. Generation
fails closed when the humidity ratio is invalid, the configured maximum
temperature cannot satisfy the RH limit, or the safe start is above the required
heating-ramp rejoin state. Startup metadata records `T_init`, canonical
`omega_in_bc(0)`, the RH limit, `T_required`, final `T_start`, whether preheating
was required, startup RH, and the rejoin row.

The rejoin temperature and humidity ratio are the original canonical linear
interpolation at the configured startup duration; rejoin relative humidity is
recomputed thermodynamically from those two values. Every physical handoff node
must lie inside the persisted operational temperature, humidity-ratio, and RH
envelopes, and every retained regular node remains unchanged.

The extra row is boundary interpolation support, not a regular output state.
COMSOL output, HDF5 state time, and Dataset transitions continue to use only
`common.time.regular_times`. The final `schedule.csv` bytes are hashed before
execution and are persisted with handoff version, policy, source, humidity, and
rejoin provenance. Disabling the ramp retains the canonical table values and
times exactly.

Generation-input EDA converts absolute temperatures to degrees Celsius only at
the presentation boundary. `T_amb`, `T_in_base`, `T_init`, and plotted or
tabulated `T_in_bc` use `T_C = T_K - 273.15`; amplitudes, changes, differences,
and rates remain in K or K/h. Persisted configuration, CSV, HDF5, COMSOL, hash,
and identity values remain in kelvin.

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

The input EDA derives the local initial equilibrium relative humidity by the
exact algebraic inverse of that same relation:

```text
R_0 = [100 X_0_db_field /
       (A_osw + B_osw (T_init - 273.15 K))]^(1/C_osw)
phi_eq_0 = R_0 / (1 + R_0)
```

This diagnostic interprets the generated initial-moisture field at
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

## Provenance, persistence, and versions

Resolved configuration and generated artifacts preserve material family and
role, natural or OOD selection, selected values or complete records, scientific
configuration, relevant diagnostics, and content identities. Exact admission
rejects stale or structurally incompatible artifacts. The maintained
[identity and provenance policy](simulation_generation.md#identity-and-provenance-policy)
defines semantic, implementation, execution, and provenance dependencies without
duplicating them here.

The active Generation YAML, case, canonical HDF5, transient-transition,
schedule-generator, and COMSOL boundary-handoff version values are all `1`.
These remain distinct contract domains even though their numeric values match:
schema versions describe persisted structure, while algorithm values describe
how scientific values are constructed. The resolved temperature supports,
startup metadata structure, and exact input hashes also participate in current
transient identity. Transient input cases created under the previous schedule
or cold-start handoff contract must therefore be regenerated before execution;
ordinary reads never rewrite them. Steady-flow case identity excludes transient
schedule and handoff dependencies.

## Source attribution

`configs/generation/sources.yaml` is the sole machine-readable bibliography and
locator owner. The following keys preserve the reference grouping used by the
current material and model records; inspect resolved `source_refs` for the
exact value-to-source chain.

| Context | Source keys |
| --- | --- |
| Project and model baselines | `ba`, `pino_airflow_project`, `transient_comsol_model_report`, `vm2_project` |
| Shared thermal evidence | `matouk_thermal` |
| Chickpea | `chickpea_physical`, `chickpea_oswin_source`, `chickpea_drying` |
| Field pea | `pea_physical`, `pea_sorption`, `pea_drying`, `manitoba_field_beans`, `swiss_crops` |
| Kidney bean | `kidney_physical`, `kidney_oswin_source`, `kidney_drying`, `manitoba_field_beans` |
| Lentil | `lentil_oswin_source`, `lentil_sorption_menkov`, `lentil_drying` |
| Rapeseed/canola | `canola_thermal`, `canola_airflow`, `canola_oswin`, `canola_drying`, `canola_drying_gazor`, `swiss_oil`, `swiss_oil_2026` |
| Sunflower seed | `sunflower_sorption_drying_munder_2019`, `sunflower_density_isik_izli_2007`, `sunflower_physical_gupta_das_1997`, `sunflower_thermal_ince_2008`, `sunflower_airflow_canada_1990`, `sunflower_class_munder_2017` |

This grouping is not a claim that every source supports every parameter for the
listed material. The material YAML and resolved provenance provide that
specific attribution.

## Operational cross-reference

See the [Generation workflow](simulation_generation.md) for configuration
inspection, CPU setup, preflight, Technical Smoke, benchmark, Pilot, Production,
publication, resume, cancellation, retention, cleanup, and troubleshooting. The
project-level entry point is the [README](../README.md).
