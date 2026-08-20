# Generation Scientific Parameter Reference

This document explains the stable scientific meaning of Generation inputs and
model equations. Validated YAML under `configs/generation` is authoritative for
current values, supports, units, campaign membership, and seeds. Commands,
monitoring, recovery, and cleanup belong in the
[Generation operations guide](simulation_generation.md).

## Interpreting configured values

Generation values are modelling and sampling decisions, not automatically
universal material constants. Each authored record identifies its evidence
class, source references, and any conversion, fit, transfer, applicability
limit, or synthetic-design rationale. A successful simulation validates the
software and data contracts; it does not upgrade scientific evidence.

An empty source list is valid for an explicit project decision. Natural values
and OOD supports have separate provenance. Campaign purpose and the fully
resolved configuration participate in scientific identity.

## Authoritative configuration

| Concern | Owner |
| --- | --- |
| Grid, time, fixed values, equations, tolerances | `configs/generation/common.yaml` |
| Parameter definitions, units, transforms, OOD groups | `configs/generation/registry.yaml` |
| Material records and provenance | `configs/generation/materials/*.yaml` |
| Operation and inlet-schedule constraints | `configs/generation/operations/fixed_bed.yaml` |
| Family roles, counts, membership, and seeds | Selected campaign YAML |
| Bibliographic metadata | `configs/generation/sources.yaml` |

Use `validate-config` or `run CONFIG --dry-run` to inspect effective values.
Do not copy mutable supports or campaign inventories from this document.

## Sampling semantics

- Natural support defines the intended in-distribution region for a material and
  profile.
- Parameter OOD moves one eligible scalar or one complete coupled record outside
  natural support. Coupled density, sorption, kinetics, and schedule records are
  selected atomically.
- Family OOD withholds material families while retaining natural parameter
  support; it is distinct from parameter stress.
- Material identity is provenance, not an implicit one-hot model input.
- Derived quantities, record components, and latent packing scatter are not
  independent design coordinates.

Campaign YAML owns current learning and evaluation roles. Material files remain
role-neutral.

## Scientific model

### Airflow, fields, and moisture basis

The steady profile solves Darcy–Brinkman airflow using heterogeneous
permeability `kappa(x)`, porosity `epsilon(x)`, and inlet pressure
`p_in_bc(y)`. The transient profile reuses that airflow state and adds vapour
transport, heat transfer, and grain-moisture dynamics.

Initial moisture is stored on a dry basis:

```text
X_wb = X_db / (1 + X_db)
X_db = X_wb / (1 - X_wb)
```

Spatial-field statistics and bounds are configuration-owned synthetic generator
contracts unless their resolved provenance states otherwise.

Each material owns exactly one operational drying target:

```yaml
target_moisture:
  target_moisture_wb: 0.12
```

`target_moisture_wb` is the material-specific practical safe-storage drying
target expressed as a wet-basis mass fraction; `0.12` means `12.0 % wb`.
Exact literature or isotherm-derived safe-storage values remain in provenance
and are normally rounded deterministically to a practical whole wet-basis
percentage point. A stricter conventional source-backed value may be retained
when upward rounding would contradict safe storage. Market-acceptance limits
may be cited for context, but they are not a competing runtime target.

The resolved target supplies the existing `X_target_wb` COMSOL scalar and
therefore participates in the existing scientific and case-input identities.
The stopping criterion remains spatial: `f_wet_dm` measures the dry-mass
fraction above `X_target_wb` and is compared with the configured tolerance.
This target cleanup does not replace that criterion with bulk-only termination.

### Packing and permeability

Mean permeability is coupled to packing porosity through a material-calibrated
Kozeny–Carman reference trend:

```text
g(epsilon) = epsilon^3 / (1 - epsilon)^2
A_KC_reference = kappa_nominal / g(eps_bed_cal_ref)
eps_kc_trend = g^-1(kappa_mean / A_KC_reference)
```

`kappa_mean` is the sampled coordinate. A small bounded case-level packing
scatter represents unresolved morphology. The relation is a global material
trend; it does not impose pointwise equality between `kappa(x)` and
`epsilon(x)`.

### Inlet schedule and startup

The persisted schedule has exactly three columns:

```text
t;T_in_bc;omega_in_bc
h;K;kg/kg
```

Relative humidity is derived after interpolating temperature and humidity ratio:

```text
phi_in_bc = [p_ref omega_in_bc / (0.621945 + omega_in_bc)]
            / p_sat(T_in_bc)
```

Generation validates derived relative humidity continuously, including interior
extrema. Infeasible stochastic schedules are deterministically resampled as a
whole rather than clipped.

The transient startup handoff begins at `T_init`. Its inlet relative humidity is
the minimum initial bed-equilibrium RH minus the configured absolute drying
margin:

```text
phi_in_start = min(phi_eq_init) - initial_equilibrium_rh_dry_margin
```

Temperature and humidity ratio then ramp to the unchanged canonical schedule at
the configured rejoin time. Startup-only RH bounds, canonical operating bounds,
the COMSOL numerical clip, and the physical saturation limit are distinct
contracts. The startup humidity ratio is derived psychrometrically and may lie
outside the stochastic schedule support; it is never independently clipped.
The handoff is deterministic, does not change case membership or seeds, and is
not an additional learned channel.

### Moisture, equilibrium, and heat

Grain water uses surface and internal compartments:

```text
f_surf d(w_surf)/dt       = j_int - m_evap
(1 - f_surf) d(w_int)/dt  = -j_int
w_gr = f_surf w_surf + (1 - f_surf) w_int
j_int = (1 - f_surf) r_int (w_int - w_surf)
m_evap = f_surf r_surf max(w_surf - w_eq, 0)
```

The configured Oswin record supplies equilibrium dry-basis moisture:

```text
X_eq_db = 0.01 [A_osw + B_osw (T - 273.15 K)]
          [phi_eff / (1 - phi_eff)]^C_osw
w_eq = rho_bu_dry X_eq_db
```

Initial equilibrium RH is the exact algebraic inverse evaluated from
`X_0_db_field` and `T_init`. Realized packing adjusts dry bulk density, and
evaporation contributes the latent heat sink `Q_evap = -h_fg m_evap`. The
resolved evidence attached to each coefficient determines whether it is
measured, fitted, transferred, estimated, derived, or synthetic.

## Validation and provenance

Initial-state consistency, bulk-moisture balance, and float32 storage fidelity
use separate tolerance classes because they protect different contracts.
Schedule feasibility and temporal resolution are generator invariants.

Resolved configuration, seeds, case and input identities, source commit,
templates, schedules, diagnostics, and content hashes bind durable artifacts.
Ordinary reads never rewrite admitted evidence. Dataset identity remains
separate from filenames, storage paths, display labels, and campaign roles.

`configs/generation/sources.yaml` is the machine-readable bibliography. Follow a
value's resolved `source_refs`; a material-level citation is not evidence for
every parameter of that material.
