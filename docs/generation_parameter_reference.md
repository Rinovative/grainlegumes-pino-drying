# Generation Scientific and Technical Parameter Reference

This is the single maintained human-readable reference for Generation
parameters, ranges, equations, classifications, and scientific provenance. The
validated YAML and Python resolver remain executable authority; this document
explains their scientific meaning without becoming a second configuration. See
`simulation_generation.md` for configuration ownership and execution workflows.

## Scientific parameter note

> The configured values and ranges are executable modelling and sampling
> decisions, not universal experimentally validated material constants. A value
> may be directly reported, refitted, convention- or unit-converted, transferred,
> engineering-calculated or inverted, selected as a calibration prior, or
> selected as synthetic generator design. A citation therefore does not imply
> that the final configured number appears directly in that source. The
> machine-readable `evidence` classification, `source_refs`, and any explicit
> method or applicability limit define the authoritative interpretation. Software,
> COMSOL, smoke, and pilot passes validate runtime and data flow only; they do not
> experimentally validate the configured science.

## Interpreting the compact provenance record

Every authored provenance record has two required concepts: `evidence` states
what kind of support underlies the value, and `source_refs` points to the
canonical bibliography below. An empty source list is valid only for an explicit
project modelling or synthetic design decision.

The controlled evidence vocabulary preserves the distinctions that matter:
direct literature or official targets; literature fits, transfers, and
convention conversions; project baselines; engineering estimates, conversions,
or inversions; calibration or hierarchical priors; synthetic designs; derived
values; and coupled records. It is the sole controlled classification for the
scientific basis of a value.

Four optional concepts appear only when they add information:

- `method` records an actual derivation, conversion, inversion, refit, or atomic
  selection method;
- `verification: mathematically_reproduced` records a positive reproduction
  performed by the resolver; absence makes no independent-verification claim;
- `applicability` retains genuine evidence domains or limitations, while
  material identity and product form remain owned once by `material_scope`;
- `note` holds one concise caveat that is not already represented elsewhere.

A separate confidence field is not used because uncalibrated compound judgements
would overlap evidence class, method, source, and explicit applicability limits.
Those canonical concepts communicate directness and uncertainty without a second
vocabulary.

## Owners at a glance

| Owner | Scientific decision |
| --- | --- |
| `configs/generation/sources.yaml` | One bibliographic record per source key |
| `configs/generation/registry.yaml` | Parameter name, symbol, unit, kind, transform, block, sampling order, OOD group, components, and derivation |
| `configs/generation/common.yaml` | Grid, time, shared fixed physics, formulas, adapter and HDF5 contracts |
| `configs/generation/operations/<operation>.yaml` | Apparatus and operation supports and constraints |
| `configs/generation/materials/<family>.yaml` | Role-neutral family values, natural supports, coupled records, targets, evidence |
| `configs/generation/profiles/<profile>.yaml` | Template, export schema, explicit native mappings, profile conditioning |
| `configs/generation/campaigns/<profile>/<campaign>.yaml` | Roles, counts, seeds, memberships, package requests |
| `configs/generation/execution/<site>.yaml` | Execution and retention only; excluded from scientific identity |

Python resolution is the only layer that combines these owners. It validates
cross-layer compatibility, projects the registry into the selected profile,
derives identities and allocations, and persists the effective scientific
configuration. Documentation is not a second parameter registry.

## Inspect the resolved scientific contract

Show the complete fail-closed campaign view without changing unresolved launch
state:

```bash
# from the repository checkout
python -m src.generation.cli.cli_generation validate-config \
  configs/generation/campaigns/transient_drying/family_generalization.yaml \
  --allow-incomplete
```

The output exposes the effective material inventory and roles, source-case
counts and derived total, memberships, campaign and derived seed plan, package
requests and expanded package inventory, profile-projected OOD units and exact
allocation, purpose-specific pilot or smoke scope, bounded static-sentinel
workload, template identity, execution resources, and readiness gates.

Inspect individual values or complete atomic records through their inherited
provenance chain:

```bash
python -m src.generation.cli.cli_generation validate-config \
  configs/generation/campaigns/transient_drying/family_generalization.yaml \
  --allow-incomplete \
  --inspect-parameter A_osw \
  --inspect-parameter density_calibration \
  --inspect-parameter cp_w \
  --inspect-parameter grid.dx \
  --inspect-parameter time.stop \
  --inspect-parameter physical_formulas.rho_bu_dry \
  --inspect-parameter packing_porosity_mean_support
```

Inspection distinguishes authored, inherited, selected, and derived evidence.
It also distinguishes generator consumption, admitted COMSOL CLI overrides, and
package-fixed values bound to the hashed native template without a Python
runtime override.
Use the same command after every valid configuration edit; derived totals,
dimensions, memberships, packages, and identities must change from the edited
owners without synchronized Python or documentation edits.

## Registry, profile, and channel ownership

The registry owns the active sampling blocks, effective dimensions, names,
units, kinds, transforms, physical OOD groups, and atomic-record components.
The profile projection determines which registry entries participate in one
simulation profile. Do not maintain those inventories in prose.

The schema-only public profile contract exposes adapter fields and available
learning views without loading campaign YAML or native templates:

```bash
python - <<'PY'
import json
from src import generation

def fields(items):
    return [field.as_dict() for field in items]

for profile_id in generation.contracts.available_profile_ids():
    contract = generation.contracts.get_profile_contract(profile_id)
    print(json.dumps({
        "profile": contract.id,
        "available_learning_views": contract.available_learning_views,
        "airflow_source": contract.airflow_source,
        "coordinate_fields": fields(contract.coordinate_fields),
        "stationary_fixed_fields": fields(contract.stationary_fixed_fields),
        "static_fields": fields(contract.static_fields),
        "transient_fields": fields(contract.transient_fields),
        "schedule_fields": fields(contract.schedule_fields),
        "scalar_inputs": fields(contract.scalar_inputs),
    }, indent=2))
PY
```

Dataset task and step contracts remain the authoritative tensor-channel owners.
Generation profile fields are source and provenance contracts; they must not be
copied into a separate learning-channel table. A material family is metadata,
not an implicit one-hot channel, and an evaluation category does not create a
new model, equation, sampling coordinate, or native profile.

## Canonical parameter catalogue

The registry declares 63 canonical entries. Sampling coordinates contribute 28 dimensions to `steady_flow` and 54 to `transient_drying`; coupled records and derived or fixed entries add no independent coordinates. OOD is admitted only when the resolved record supplies a valid tail or alternate complete record.

| Name | Symbol | Unit | Class | Profiles | Transform | Owner/block | OOD group | Meaning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `kappa_mean` | `\bar{\kappa}` | `m^2` | sampled | `steady_flow`, `transient_drying` | `log` | `airflow` | `bed` | Mean scalar bed permeability used by the lognormal permeability field. |
| `kappa_cv` | `c_{\kappa}` | `1` | sampled | `steady_flow`, `transient_drying` | `linear` | `airflow` | `bed` | Coefficient of variation used to derive the lognormal permeability spread. |
| `bed.structure.coarse_len_rel` | `\ell_{b,c}/L_x` | `1` | sampled | `steady_flow`, `transient_drying` | `log` | `airflow` | `bed` | Coarse bed correlation length divided by bed length. |
| `bed.structure.fine_len_rel` | `\ell_{b,f}/L_x` | `1` | sampled | `steady_flow`, `transient_drying` | `log` | `airflow` | `bed` | Fine bed correlation length divided by bed length. |
| `bed.structure.coarse_weight` | `\alpha_{b,c}` | `1` | sampled | `steady_flow`, `transient_drying` | `logit` | `airflow` | `bed` | Independent coarse contribution to the bed multiscale field. |
| `bed.structure.fine_ani_x` | `a_{b,x}` | `1` | sampled | `steady_flow`, `transient_drying` | `log` | `airflow` | `bed` | Fine bed correlation-length multiplier along x. |
| `bed.structure.fine_ani_y` | `a_{b,y}` | `1` | sampled | `steady_flow`, `transient_drying` | `log` | `airflow` | `bed` | Fine bed correlation-length multiplier along y. |
| `bed.structure.cross_scale_corr` | `\rho_b` | `1` | sampled | `steady_flow`, `transient_drying` | `linear` | `airflow` | `bed` | Correlation between coarse and fine bed latent seeds. |
| `bed.perturbations.amplitude` | `\eta_b` | `1` | sampled | `steady_flow`, `transient_drying` | `linear` | `airflow` | `bed` | Root-mean-square amplitude of bed-only local perturbations. |
| `bed.perturbations.granularity` | `g_b` | `1` | sampled | `steady_flow`, `transient_drying` | `linear` | `airflow` | `bed` | Scale selector for bed-only local perturbations. |
| `bed.perturbations.sign_bias` | `q_b` | `1` | sampled | `steady_flow`, `transient_drying` | `linear` | `airflow` | `bed` | Positive-sign probability for bed-only local perturbations. |
| `permeability.anisotropy.max_ratio` | `a_{\max}` | `1` | sampled | `steady_flow`, `transient_drying` | `linear` | `airflow` | `bed` | Maximum permeability anisotropy ratio. |
| `permeability.anisotropy.exponent` | `\gamma_a` | `1` | sampled | `steady_flow`, `transient_drying` | `linear` | `airflow` | `bed` | Exponent mapping bed structure magnitude to anisotropy. |
| `permeability.anisotropy.strength` | `s_K` | `1` | sampled | `steady_flow`, `transient_drying` | `linear` | `airflow` | `bed` | Multiplier controlling permeability-tensor anisotropy. |
| `permeability.orientation.jitter` | `j_{\theta}` | `1` | sampled | `steady_flow`, `transient_drying` | `linear` | `airflow` | `bed` | Random perturbation amplitude for permeability orientation. |
| `permeability.orientation.smooth_len_rel` | `\ell_{\theta}/L_x` | `1` | sampled | `steady_flow`, `transient_drying` | `log` | `airflow` | `bed` | Relative smoothing length for permeability orientation. |
| `porosity.kc_anchor_factor` | `a_KC` | `1` | sampled | `steady_flow`, `transient_drying` | `log` | `airflow` | `bed` | Dimensionless factor scaling the material-calibrated Kozeny-Carman reference coefficient. |
| `porosity.smooth_len_rel` | `\ell_{\varepsilon}/L_x` | `1` | sampled | `steady_flow`, `transient_drying` | `linear` | `airflow` | `bed` | Relative smoothing length for the porosity latent field. |
| `porosity.texture_amp` | `\Delta\varepsilon` | `1` | sampled | `steady_flow`, `transient_drying` | `linear` | `airflow` | `bed` | Porosity texture amplitude around the calibrated reference. |
| `pressure_bc.mean` | `\bar{p}_{\mathrm{in}}` | `Pa` | sampled | `steady_flow`, `transient_drying` | `linear` | `airflow` | `operation` | Mean inlet pressure boundary magnitude. |
| `pressure_bc.sin_amp` | `a_{p,\sin}` | `1` | sampled | `steady_flow`, `transient_drying` | `linear` | `airflow` | `operation` | Relative sinusoidal inlet-pressure amplitude. |
| `pressure_bc.sin_freq` | `f_{p,\sin}` | `1` | sampled | `steady_flow`, `transient_drying` | `linear` | `airflow` | `operation` | Sinusoidal inlet-pressure spatial frequency. |
| `pressure_bc.sin_phase` | `\varphi_{p,\sin}` | `rad` | sampled | `steady_flow`, `transient_drying` | `phase` | `airflow` | `operation` | Sinusoidal inlet-pressure phase. |
| `pressure_bc.gauss_count` | `n_{p,G}` | `1` | sampled | `steady_flow`, `transient_drying` | `none` | `airflow` | `operation` | Number of Gaussian inlet-pressure components. |
| `pressure_bc.gauss_amp` | `a_{p,G}` | `1` | sampled | `steady_flow`, `transient_drying` | `linear` | `airflow` | `operation` | Combined relative amplitude of Gaussian pressure components. |
| `pressure_bc.gauss_width` | `\sigma_{p,G}` | `1` | sampled | `steady_flow`, `transient_drying` | `log` | `airflow` | `operation` | Reference width of Gaussian pressure components. |
| `pressure_bc.gauss_jitter` | `j_{p,G}` | `1` | sampled | `steady_flow`, `transient_drying` | `linear` | `airflow` | `operation` | Relative Gaussian pressure-width jitter. |
| `pressure_bc.linear_amp` | `a_{p,\mathrm{lin}}` | `1` | sampled | `steady_flow`, `transient_drying` | `linear` | `airflow` | `operation` | Relative linear inlet-pressure trend. |
| `initial_moisture.mean_db` | `\bar{X}_{0,db}` | `kg/kg` | sampled | `transient_drying` | `linear` | `initial_moisture` | `initial_moisture` | Mean initial dry-basis moisture of the generated field. |
| `initial_moisture.amplitude_db` | `\Delta X_{0,db}` | `kg/kg` | sampled | `transient_drying` | `linear` | `initial_moisture` | `initial_moisture` | Maximum dry-basis deviation from the configured initial mean. |
| `initial_moisture.structure.coarse_len_rel` | `\ell_{X,c}/L_x` | `1` | sampled | `transient_drying` | `log` | `initial_moisture` | `initial_moisture` | Coarse initial-moisture correlation length divided by bed length. |
| `initial_moisture.structure.fine_len_rel` | `\ell_{X,f}/L_x` | `1` | sampled | `transient_drying` | `log` | `initial_moisture` | `initial_moisture` | Fine initial-moisture correlation length divided by bed length. |
| `initial_moisture.structure.coarse_weight` | `\alpha_{X,c}` | `1` | sampled | `transient_drying` | `logit` | `initial_moisture` | `initial_moisture` | Independent coarse contribution to the initial-moisture multiscale field. |
| `initial_moisture.structure.fine_ani_x` | `a_{X,x}` | `1` | sampled | `transient_drying` | `log` | `initial_moisture` | `initial_moisture` | Fine initial-moisture correlation-length multiplier along x. |
| `initial_moisture.structure.fine_ani_y` | `a_{X,y}` | `1` | sampled | `transient_drying` | `log` | `initial_moisture` | `initial_moisture` | Fine initial-moisture correlation-length multiplier along y. |
| `initial_moisture.structure.cross_scale_corr` | `\rho_X` | `1` | sampled | `transient_drying` | `linear` | `initial_moisture` | `initial_moisture` | Correlation between coarse and fine initial-moisture latent seeds. |
| `T_in_base` | `T_{\mathrm{in},0}` | `K` | sampled | `transient_drying` | `linear` | `operation` | `operation` | Baseline inlet-air temperature. |
| `T_in_amp` | `\Delta T_{\mathrm{in}}` | `K` | sampled | `transient_drying` | `linear` | `operation` | `operation` | Inlet-temperature schedule amplitude. |
| `omega_in_base` | `\omega_{\mathrm{in},0}` | `kg/kg` | sampled | `transient_drying` | `linear` | `operation` | `operation` | Baseline inlet humidity ratio. |
| `omega_in_amp` | `\Delta\omega_{\mathrm{in}}` | `kg/kg` | sampled | `transient_drying` | `linear` | `operation` | `operation` | Inlet humidity-ratio schedule amplitude. |
| `schedule.corr` | `\rho_{T,\omega}` | `1` | sampled | `transient_drying` | `linear` | `operation` | `operation` | Cross-correlation of temperature and humidity schedule latent processes. |
| `schedule.timescale_rel` | `\tau_{\mathrm{sched}}/t_{\max}` | `1` | sampled | `transient_drying` | `log` | `operation` | `operation` | Schedule correlation timescale divided by total duration. |
| `schedule.component_weights` | `\boldsymbol{\lambda}_{\mathrm{sched}}` | `1` | sampled | `transient_drying` | `none` | `operation` | `operation` | Smooth, event, and trend schedule simplex. |
| `schedule.event_count` | `n_{\mathrm{event}}` | `1` | sampled | `transient_drying` | `none` | `operation` | `operation` | Number of generated schedule events. |
| `schedule.event_duration_rel` | `d_{\mathrm{event}}/t_{\max}` | `1` | sampled | `transient_drying` | `log` | `operation` | `operation` | Event duration divided by total schedule duration. |
| `schedule.event_width_rel` | `w_{\mathrm{event}}/t_{\max}` | `1` | sampled | `transient_drying` | `log` | `operation` | `operation` | Event-edge width divided by total schedule duration. |
| `rho_bu_dry_ref` | `\rho_{\mathrm{bu,dry,ref}}` | `kg/m^3` | sampled | `transient_drying` | `log` | `material_properties` | `material_properties` | Reference dry bulk density at calibration porosity. |
| `k_gr` | `k_{\mathrm{gr}}` | `W/(m*K)` | sampled | `transient_drying` | `log` | `material_properties` | `material_properties` | Dry granular-phase thermal conductivity. |
| `cp_gr_dry` | `c_{p,\mathrm{gr,dry}}` | `J/(kg*K)` | sampled | `transient_drying` | `log` | `material_properties` | `material_properties` | Dry granular-phase specific heat capacity. |
| `r_surf_0` | `r_{\mathrm{surf},0}` | `1/s` | sampled | `transient_drying` | `log` | `material_properties` | `material_properties` | Reference surface-moisture transfer rate. |
| `r_int_surf` | `r_{\mathrm{int/surf}}` | `1` | sampled | `transient_drying` | `log` | `material_properties` | `material_properties` | Internal-to-surface transfer-rate ratio. |
| `f_surf` | `f_{\mathrm{surf}}` | `1` | sampled | `transient_drying` | `logit` | `material_properties` | `material_properties` | Initial surface-water fraction. |
| `T_amb` | `T_{\mathrm{amb}}` | `K` | sampled | `transient_drying` | `linear` | `operation` | `operation` | Ambient temperature and initial granular-phase temperature source. |
| `bed.structure.fine_weight` | `\alpha_{b,f}` | `1` | derived | `steady_flow`, `transient_drying` | `none` | `derived/fixed` | `not eligible` | Derived complementary fine contribution to the bed multiscale field. |
| `initial_moisture.structure.fine_weight` | `\alpha_{X,f}` | `1` | derived | `transient_drying` | `none` | `derived/fixed` | `not eligible` | Derived complementary fine contribution to initial-moisture structure. |
| `eps_min_global` | `\varepsilon_{\min}` | `1` | fixed | `steady_flow`, `transient_drying` | `none` | `derived/fixed` | `not eligible` | Universal lower porosity bound. |
| `eps_max_global` | `\varepsilon_{\max}` | `1` | fixed | `steady_flow`, `transient_drying` | `none` | `derived/fixed` | `not eligible` | Universal upper porosity bound. |
| `eps_bed_cal_ref` | `\varepsilon_{\mathrm{bed,cal,ref}}` | `1` | fixed | `steady_flow`, `transient_drying` | `none` | `derived/fixed` | `not eligible` | Material calibration porosity for Kozeny-Carman reference coupling and dry bulk density. |
| `X_target_wb` | `X_{\mathrm{target,wb}}` | `1` | fixed | `transient_drying` | `none` | `derived/fixed` | `not eligible` | Material wet-basis target moisture. |
| `oswin` | `\boldsymbol{\theta}_{\mathrm{Oswin}}` | `component-specific` | coupled_record | `transient_drying` | `none` | `derived/fixed` | `material_properties` | Coupled Oswin A, B, and C equilibrium-isotherm record. |
| `T_init` | `T_{\mathrm{init}}` | `K` | derived | `transient_drying` | `none` | `derived/fixed` | `not eligible` | Initial temperature derived from ambient temperature. |
| `r_surf` | `r_{\mathrm{surf}}` | `1/s` | derived | `transient_drying` | `none` | `derived/fixed` | `not eligible` | Surface rate copied from its reference value. |
| `r_int` | `r_{\mathrm{int}}` | `1/s` | derived | `transient_drying` | `none` | `derived/fixed` | `not eligible` | Internal rate derived from r_int_surf and r_surf. |

## Shared grid, time, and fixed values

The domain is `Lx=1.2 m`, `Ly=0.75 m`, and `Lz=0.8 m` on a boundary-inclusive `nx=401` by `ny=251` grid. The resolver derives `dx=Lx/(nx-1)` and `dy=Ly/(ny-1)`. Regular transient output is `0:1:168 h`; internal steps are adaptive and an irregular exact-stop state is diagnostic only.

| Name | Value | Unit | Evidence |
| --- | ---: | --- | --- |
| `T_flow_ref` | 300.65 | `K` | `synthetic_design` |
| `p_ref` | 101325 | `Pa` | `project_baseline` |
| `p_out` | 0 | `Pa` | `project_baseline` |
| `T_in_min` | 298.15 | `K` | `synthetic_design` |
| `T_in_max` | 313.15 | `K` | `synthetic_design` |
| `omega_min` | 0.0025 | `kg/kg` | `engineering_estimate` |
| `omega_max` | 0.0145 | `kg/kg` | `engineering_estimate` |
| `phi_operational_min` | 0.05 | `1` | `synthetic_design` |
| `phi_operational_max` | 0.85 | `1` | `synthetic_design` |
| `phi_clip_min` | 1e-06 | `1` | `synthetic_design` |
| `phi_clip_max` | 0.999 | `1` | `synthetic_design` |
| `cp_w` | 4180 | `J/(kg*K)` | `project_baseline` |
| `h_fg` | 2.4182e+06 | `J/kg` | `project_baseline` |
| `D_v_air` | 2.811e-05 | `m^2/s` | `project_baseline` |
| `M_v` | 0.0180153 | `kg/mol` | `literature_direct` |
| `d_wall` | 0.019 | `m` | `project_baseline` |
| `k_wall` | 0.13 | `W/(m*K)` | `project_baseline` |
| `h_ext` | 8 | `W/(m^2*K)` | `engineering_estimate` |
| `U_wall` | derived by configured formula | `W/(m^2*K)` | `derived` |
| `f_wet_dm_max` | 0.05 | `1` | `synthetic_design` |
| `schedule_interpolation` | linear | `method` | `synthetic_design` |

## Operation and generator natural supports

These are natural or ID supports from `fixed_bed.yaml`; transforms and OOD groups are registry-owned. The schedule simplex is one atomic smooth, event, and trend record. Inspect the resolved record for every disjoint OOD tail.

| Parameter | Natural support or record | Nominal | Unit | Evidence |
| --- | --- | ---: | --- | --- |
| `pressure_bc.mean` | 250-750 | 500 | `Pa` | `project_baseline` |
| `pressure_bc.sin_amp` | -0.02-0.02 | 0 | `1` | `project_baseline` |
| `pressure_bc.sin_freq` | 0.5-1.5 | 1 | `1` | `project_baseline` |
| `pressure_bc.sin_phase` | 0-6.28319 | 3.14159 | `rad` | `synthetic_design` |
| `pressure_bc.gauss_count` | 1-5 | 3 | `1` | `project_baseline` |
| `pressure_bc.gauss_amp` | -0.06-0.06 | 0 | `1` | `project_baseline` |
| `pressure_bc.gauss_width` | 0.02-0.1 | 0.05 | `1` | `project_baseline` |
| `pressure_bc.gauss_jitter` | 0.1-0.5 | 0.3 | `1` | `project_baseline` |
| `pressure_bc.linear_amp` | -0.03-0.03 | 0 | `1` | `project_baseline` |
| `T_in_base` | 303.15-309.15 | 308.15 | `K` | `project_baseline` |
| `T_in_amp` | 0-4 | 2.5 | `K` | `synthetic_design` |
| `omega_in_base` | 0.0045-0.0115 | 0.0075 | `kg/kg` | `engineering_estimate` |
| `omega_in_amp` | 0-0.003 | 0.0015 | `kg/kg` | `synthetic_design` |
| `schedule.corr` | -0.75-0.25 | -0.35 | `1` | `synthetic_design` |
| `schedule.timescale_rel` | 0.03-0.18 | 0.08 | `1` | `synthetic_design` |
| `schedule.component_weights` | {'smooth': 0.55, 'event': 0.3, 'trend': 0.15} | {'smooth': 0.55, 'event': 0.3, 'trend': 0.15} | `1` | `synthetic_design` |
| `schedule.event_count` | 0-4 | 2 | `1` | `synthetic_design` |
| `schedule.event_duration_rel` | 0.015-0.08 | 0.04 | `1` | `synthetic_design` |
| `schedule.event_width_rel` | 0.003-0.02 | 0.008 | `1` | `synthetic_design` |
| `T_amb` | 288.15-298.15 | 293.15 | `K` | `project_baseline` |
| `kappa_cv` | 0.25-0.65 | 0.45 | `1` | `synthetic_design` |
| `bed.structure.coarse_len_rel` | 0.06-0.16 | 0.1 | `1` | `project_baseline` |
| `bed.structure.fine_len_rel` | 0.02-0.07 | 0.04 | `1` | `project_baseline` |
| `bed.structure.coarse_weight` | 0.35-0.7 | 0.55 | `1` | `synthetic_design` |
| `bed.structure.fine_ani_x` | 0.65-1.55 | 1 | `1` | `synthetic_design` |
| `bed.structure.fine_ani_y` | 0.65-1.55 | 1 | `1` | `synthetic_design` |
| `bed.structure.cross_scale_corr` | 0.25-0.7 | 0.5 | `1` | `project_baseline` |
| `bed.perturbations.amplitude` | 0.08-0.28 | 0.18 | `1` | `project_baseline` |
| `bed.perturbations.granularity` | 0.3-0.7 | 0.5 | `1` | `project_baseline` |
| `bed.perturbations.sign_bias` | 0.35-0.65 | 0.5 | `1` | `project_baseline` |
| `permeability.anisotropy.max_ratio` | 1.2-3 | 2 | `1` | `project_baseline` |
| `permeability.anisotropy.exponent` | 0.8-2.5 | 1.5 | `1` | `project_baseline` |
| `permeability.anisotropy.strength` | 0.4-1 | 0.7 | `1` | `synthetic_design` |
| `permeability.orientation.jitter` | 0.005-0.02 | 0.012 | `1` | `project_baseline` |
| `permeability.orientation.smooth_len_rel` | 0.04-0.16 | 0.08 | `1` | `project_baseline` |
| `porosity.kc_anchor_factor` | 1.0 | 1 | `1` | `synthetic_design` |
| `porosity.smooth_len_rel` | 0.03-0.09 | 0.055 | `1` | `project_baseline` |
| `porosity.texture_amp` | 0.003-0.015 | 0.008 | `1` | `project_baseline` |
| `initial_moisture.structure.coarse_len_rel` | 0.08-0.24 | 0.14 | `1` | `synthetic_design` |
| `initial_moisture.structure.fine_len_rel` | 0.025-0.08 | 0.045 | `1` | `synthetic_design` |
| `initial_moisture.structure.coarse_weight` | 0.4-0.75 | 0.58 | `1` | `synthetic_design` |
| `initial_moisture.structure.fine_ani_x` | 0.7-1.4 | 1 | `1` | `synthetic_design` |
| `initial_moisture.structure.fine_ani_y` | 0.7-1.4 | 1 | `1` | `synthetic_design` |
| `initial_moisture.structure.cross_scale_corr` | 0.25-0.7 | 0.5 | `1` | `synthetic_design` |

## Material-specific natural supports

Moisture supports are dry basis (`kg water/kg dry solid`) except `X_target_wb`, which is wet basis. Oswin values are the complete `A_osw, B_osw, C_osw` record; kinetics are supports for `r_surf_0, r_int_surf, f_surf`. Exact OOD records, evidence classes, source references, methods, and genuine applicability limits remain beside each value in material YAML.

| Material | kappa_mean m^2 | packing eps | rho_bu_dry_ref kg/m^3; eps ref | k_gr W/(m*K) | cp_gr_dry J/(kg*K) | initial mean db | X_target_wb | Oswin A,B,C | kinetics supports |
| --- | --- | --- | --- | --- | --- | --- | ---: | --- | --- |
| `chickpea` | 8e-09-3e-08 | 0.37-0.45 | 620-730; 0.403 | 0.24-0.38 | 1600-2200 | 0.190476-0.315789 | 0.12 | 11.2175, -0.0615631, 0.49142 | 3e-06-1.4e-05; 0.06-0.3; 0.28-0.62 |
| `field_pea` | 8e-09-4e-08 | 0.37-0.43 | 530-620; 0.3938 | 0.25-0.38 | 1650-2250 | 0.190476-0.315789 | 0.12 | 12.062, -0.0573838, 0.343383 | 3.5e-06-1.6e-05; 0.06-0.3; 0.28-0.62 |
| `kidney_bean` | 1.5e-08-8e-08 | 0.42-0.5 | 510-620; 0.456 | 0.25-0.4 | 1600-2200 | 0.219512-0.388889 | 0.135 | 14.0761, -0.0763, 0.462107 | 2e-06-1e-05; 0.04-0.22; 0.22-0.55 |
| `lentil` | 5e-09-1.3e-08 | 0.29-0.36 | 590-680; 0.3125 | 0.3-0.44 | 1800-2300 | 0.190476-0.315789 | 0.12 | 12.062, -0.0573838, 0.343383 | 5e-06-2e-05; 0.08-0.35; 0.3-0.65 |
| `rapeseed` | 2e-09-9e-09 | 0.365-0.422 | 590-640; 0.4 | 0.15-0.25 | 1600-2100 | 0.0989011-0.190476 | 0.06 | 6.95472, -0.032779, 0.470207 | 5e-06-2.5e-05; 0.08-0.4; 0.3-0.68 |
| `sunflower_seed` | 4e-08-1.5e-07 | 0.5-0.57 | 330-410; 0.5306 | 0.16-0.3 | 500-1000 | 0.0989011-0.219512 | 0.06 | 4.8, -0.0158, 0.622278 | 3e-06-1.6e-05; 0.05-0.3; 0.25-0.65 |

## Canonical coupling equations

The exact executable formula strings are owned by `common.yaml`. Wet-basis `X_wb` and dry-basis `X_db` remain explicit.

| Quantity | Configured relationship |
| --- | --- |
| `w_surf_balance` | `f_surf*d(w_surf)/dt = j_int - m_evap` |
| `w_int_balance` | `(1-f_surf)*d(w_int)/dt = -j_int` |
| `w_gr` | `f_surf*w_surf + (1-f_surf)*w_int` |
| `w_gr_0` | `rho_bu_dry*X_0_db_field; w_surf(0)=w_int(0)=w_gr_0` |
| `r_surf` | `r_surf_0` |
| `r_int` | `r_int_surf*r_surf` |
| `j_int` | `(1-f_surf)*r_int*(w_int-w_surf)` |
| `m_evap` | `f_surf*r_surf*max(w_surf-w_eq,0)` |
| `X_db` | `w_gr/rho_bu_dry` |
| `X_wb` | `w_gr/(rho_bu_dry+w_gr)` |
| `X_wb_from_X_db` | `X_db/(1+X_db)` |
| `X_db_from_X_wb` | `X_wb/(1-X_wb)` |
| `X_wb_bulk` | `integral(w_gr)/(integral(rho_bu_dry)+integral(w_gr))` |
| `rho_bu_dry` | `rho_bu_dry_ref*(1-eps_bed)/(1-eps_bed_cal_ref)` |
| `solid_phase_density` | `rho_bu_dry/(1-eps_bed)` |
| `cp_gr_eff` | `cp_gr_dry + X_db*cp_w` |
| `volumetric_heat_capacity` | `rho_bu_dry*cp_gr_eff` |
| `k_eff` | `k_gr*(2*k_gr+k_air-2*eps_bed*(k_gr-k_air))/(2*k_gr+k_air+eps_bed*(k_gr-k_air))` |
| `phi_eff` | `min(max(phi,1e-6),0.999)` |
| `X_eq_db` | `0.01*(A_osw+B_osw*(T-273.15[K]))*(phi_eff/(1-phi_eff))^C_osw` |
| `w_eq` | `rho_bu_dry*X_eq_db` |
| `osw_ratio_0` | `(100*X_0_db_field/(A_osw+B_osw*(T_init-273.15[K])))^(1/C_osw)` |
| `phi_init` | `osw_ratio_0/(1+osw_ratio_0)` |
| `Q_evap` | `-h_fg*m_evap` |
| `f_wet_dm` | `integral(rho_bu_dry*indicator(X_wb>X_target_wb))/integral(rho_bu_dry)` |
| `total_water_balance` | `d/dt(m_w_gr+m_v_gas)=m_dot_v_in-m_dot_v_out` |

## Scientific bindings

Exact formula strings, numerical fixed values, grids, time nodes, storage
settings, and provenance records are owned by `common.yaml` and displayed by
parameter inspection. The stable interpretation is:

- granular water is split between surface and internal compartments;
- internal transfer exchanges those compartments, while evaporation removes
  surface water toward the configured equilibrium relation;
- dry-basis and wet-basis moisture remain explicitly distinguished;
- dry bulk density couples its selected reference record to the realized
  porosity field;
- effective heat capacity and conductivity couple material and bed state;
- latent heat closes the thermal source term;
- total-water diagnostics compare stored water change with inlet and outlet
  vapour flow.

These relationships are executable bindings, not a claim that one cited source
reported every final configured coefficient or range.

### Material-calibrated porosity

There is one physical porosity field. Its scalar calibration reference comes
from the active complete density-calibration record and is not another field.
The material YAML owns the natural representative-porosity support; common
configuration owns pointwise numerical guards.

The Kozeny-Carman anchor deterministically couples the selected permeability,
calibration reference, and sampled anchor factor. Natural sampling resolves a
case- and material-specific conditional support. Parameter OOD resolves disjoint
transformed tails using the registry-owned separation policy. Tail separation
and width are not restated here; inspect `porosity.kc_anchor_factor` and the
resolved `parameter_ood` allocation for the exact active contract.

Local morphology is generated from the shared background field and the resolved
porosity texture settings. Permeability and porosity share spatial structure,
but realized local permeability components do not directly become porosity
texture inputs. The same realized porosity is consumed by airflow, transient
transport and heat transfer, material density, water inventory, native solves,
HDF5, and maintained dataset views.

### Inlet schedule

The schedule contract and constraints come from the common and operation owners.
The generator validates the complete schedule, including temperature, humidity,
source-air, and interpolation rules, and retries it deterministically as one
unit. It does not clip individual time samples. Schedule diagnostics, names, and
units come from the schedule/profile contracts rather than a documentation
column list.

### Transient scalar handoff

The Generation profile contract owns one ordered 12-field case-dependent
runtime handoff. Each transient case writes exactly these values to
`scalars.csv`, admits the file before process creation, and supplies the same
ordered values to COMSOL through an argument vector without a shell.

| Position | Field | Unit |
| ---: | --- | --- |
| 1 | `T_amb` | `K` |
| 2 | `eps_bed_cal_ref` | `1` |
| 3 | `rho_bu_dry_ref` | `kg/m^3` |
| 4 | `k_gr` | `W/(m*K)` |
| 5 | `cp_gr_dry` | `J/(kg*K)` |
| 6 | `X_target_wb` | `1` |
| 7 | `r_surf_0` | `1/s` |
| 8 | `r_int_surf` | `1` |
| 9 | `f_surf` | `1` |
| 10 | `A_osw` | `1` |
| 11 | `B_osw` | `1/K` |
| 12 | `C_osw` | `1` |

`T_init` is not an independent runtime value. Python supplies `T_amb`, and the
canonical COMSOL template derives `T_init = T_amb`. The template/package-fixed
values `T_flow_ref`, `p_ref`, `p_out`, and `f_wet_dm_max` remain with their
existing canonical scientific-configuration/template owners; they are not
duplicated into `scalars.csv`. Other fixed template parameters such as `cp_w`,
`h_fg`, `d_wall`, `k_wall`, `h_ext`, and `U_wall` likewise do not belong in the
case handoff.

`generation.contracts.generation_contracts_scalar_handoff` is the sole reader
and admission owner. It binds the source filename, hash, size, exact header and
row order, canonical units, finite float64 values, case JSON representations,
ownership, and workspace containment before solver evidence or process
creation. Canonical HDF5 publication calls the same owner and writes an exact
`(12,)` float64 runtime-scalar dataset with the same names, units, ownership,
and values. Older scalar layouts fail exact admission and must be regenerated.
The learned Dataset view remains a separately name-selected eight-field
projection.

The schedule is a separate four-column time-dependent adapter:
`t,T_in_bc,omega_in_bc,phi_in_bc`. The corrected native interpolation feature
uses column 1 as its `h` argument and columns 2--4 as
`T_in_bc_file`, `omega_in_bc_file`, and `phi_in_bc_file`, with units `K`,
`kg/kg`, and `1`. It uses linear interpolation and constant extrapolation. Each
hourly interval is therefore determined by its adjacent endpoint values; no
interval mean, integral, or additional scalar is needed.

After a successful solver exit, runtime accepts exactly one new or replaced
nonempty regular `solved*.mph` candidate. A sole suffixed candidate is
atomically renamed to `solved.mph`; stale, empty, symbolic-link, missing, or
ambiguous candidates fail closed. Execution provenance records both the
produced and canonical names.

The test-owned fake COMSOL executable proves the Python handoff and output
contract. Static archive inspection proves the saved template descriptor and
the SHA-256 sidecar binds its exact bytes. Neither is native execution evidence:
a real transient technical smoke remains required to prove native parameter
application, schedule-file reload, retained output, HDF5, packages, and both
DataLoader worker modes.

## Material records and atomic selection

Material files are role-neutral. Their schema and evidence validation are owned
by the material resolver; campaign files assign roles and memberships. Each
independent value or complete record retains its evidence class and source
reference, plus a method or applicability limit only where meaningful.
Runtime smoke or pilot success never upgrades that scientific evidence.

Atomic selection rules are stable:

- density OOD selects one complete configured density-calibration record;
- sorption selection keeps one complete record with its equation convention and
  applicability domain;
- kinetics selection keeps one complete kinetic record;
- simplex-valued schedule weights remain one complete constrained vector;
- components from different records are never mixed;
- parameter OOD selects a scalar tail or a complete atomic record, never an
  accidental component-wise hybrid.

## Family and parameter OOD

Campaign roles determine natural-support learning and evaluation eligibility.
Inspect `material_roles`, `material_memberships`, and
`dataset_package_inventory` for the current families and regimes. Family OOD
uses natural material support only and never invokes parameter OOD.

Each parameter-OOD case activates one profile-eligible unit. Eligibility comes
from registry classification, profile projection, parameter kind, and actual
configured tails or alternate records. Deterministic round-robin allocation
covers every eligible unit when capacity permits and balances remaining cases.
The resolved summary and persisted provenance provide, per batch:

- eligible unit identifiers, physical groups, and unit kinds;
- per-unit allocation counts and exact per-case assignment;
- selected record or tail, realized value, natural and OOD support, transform,
  and transform-space distance at case generation time.

Scalar tails are nonempty, separated from natural support in the declared
transform, and fail closed when invalid. Natural cases have no active OOD unit.

## Adapters and persistence

Grid shape, geometry, regular times, exact-stop handling, adapter fields, export
roles, units, HDF5 layout, chunks, compression, and tolerances are resolved from
common and profile configuration. Use `validate-config`, parameter inspection,
and the public profile contract instead of copying their current values into
prose.

Canonical HDF5 binds scientific and case identities, material role, evaluation
and sampling regimes, natural-support state, sampled values and units, seed and
block evidence, complete OOD and coupled-record provenance, inputs, outputs,
diagnostics, hashes, template identity, source-export identity, and the embedded
resolved scientific configuration. Dataset package admission validates this
evidence against the requested source role and regime.

## Source provenance

`configs/generation/sources.yaml` is the machine-readable bibliography and sole owner of locators and source metadata. An empty `source_refs` list truthfully denotes an explicit modelling or synthetic decision without an asserted external source. Unknown keys fail closed.

- `ba` - Albertin, R. M. (2025). Trocknung von Koernerleguminosen. Bachelorarbeit, OST. (SHA-256 0d59f098 (full digest recorded in input coverage))
- `matouk_thermal` - Matouk, A. M. et al. (2018). Thermal Properties of Some Legume Seeds. Journal of Soil Sciences and Agricultural Engineering, 9, 261-267. (doi:10.21608/jssae.2018.35758)
- `lentil_oswin_source` - Cenkowski, S., Jayas, D. S., Pabis, S. & Muir, W. E. (2015). Sorption characteristics of red lentils during storage. Canadian Biosystems Engineering, 57, 3.1-3.8. (doi:10.7451/CBE.2015.57.3.1)
- `lentil_sorption_menkov` - Menkov, N. D. (2000). Moisture sorption isotherms of lentil seeds at several temperatures. Journal of Food Engineering, 44, 205-211. (doi:10.1016/S0260-8774(00)00028-5)
- `lentil_drying` - Mangueira, E. R. et al. (2021). Analysis of the thin layer drying kinetic of brown lentil grains. Research, Society and Development, 10. (RSD article 19258)
- `swiss_crops` - swiss granum (2026). Empfehlungen fuer Uebernahmebedingungen und gesetzliche Bestimmungen von Ackerkulturen zur menschlichen Ernaehrung, Ernte 2026. (2026-03-12 edition)
- `chickpea_physical` - Guerhan, R., Oezarslan, C., Topuz, N., Akbas, T. & Simsek, E. (2009). Effects of Moisture Content on Physical Properties of Black Kabuli Chickpea (Cicer arietinum L.) Seed. Asian Journal of Chemistry, 21(4), 3270-3278. (Asian Journal of Chemistry 21(4):3270-3278)
- `chickpea_oswin_source` - Armstrong, P. R., Maghirang, E. B., Bhadriraju, S., & McNeill, S. G. (2017). Equilibrium Moisture Content of Kabuli Chickpea, Black Sesame, and White Sesame Seeds. Applied Engineering in Agriculture, 33(5), 737-742. (doi:10.13031/aea.12460)
- `chickpea_drying` - Cavalcanti-Mata, M. E. R. M. et al. (2020). A new approach to traditional drying models for thin-layer drying kinetics of chickpeas. Journal of Food Process Engineering, 43. (doi:10.1111/jfpe.13569)
- `kidney_physical` - Isik, E. & Unal, H. (2011). Some physical properties of white kidney beans (Phaseolus vulgaris L.). African Journal of Biotechnology, 10. (stable journal PDF)
- `kidney_oswin_source` - Campos, R. C. et al. (2016). Bean grain hysteresis with induced mechanical damage. Revista Brasileira de Engenharia Agricola e Ambiental, 20(10), 930-935. (stable PDF; journal article 20(10):930-935)
- `kidney_drying` - Doymaz, I. (2016). Hot-Air Drying and Rehydration Characteristics of Red Kidney Bean Seeds. Chemical Engineering Communications, 203. (doi:10.1080/00986445.2015.1056299)
- `manitoba_field_beans` - Province of Manitoba, Agriculture. Field Beans (official crop-management guidance; accessed 2026-08-09). (Government of Manitoba field-beans guidance)
- `pea_physical` - Yalcin, I., Ozarslan, C. & Akbas, T. (2007). Physical properties of pea (Pisum sativum) seed. Journal of Food Engineering, 79, 731-735. (Journal of Food Engineering 79:731-735)
- `pea_sorption` - Garg, M. K. & Chandra, P. (2003). Sorption Characteristics of Pea Seeds. Journal of Agricultural Engineering, 40(4). (ICAR journal record)
- `pea_drying` - Ganesh, C. V. & Sokhansanj, S. (1997). High temperature mechanical drying of field peas (Pisum sativum L.). University of Saskatchewan proceedings/thesis record. (University of Saskatchewan repository)
- `canola_thermal` - Yu, D., Shrestha, B. L. & Baik, O.-D. (2015). Thermal conductivity, specific heat, thermal diffusivity, and bulk density of canola seeds. Journal of Food Engineering, 165, 156-165. (doi:10.1016/j.jfoodeng.2015.05.012)
- `canola_airflow` - Jayas, D. S., Sokhansanj, S., Moysey, E. B. & Barber, E. M. (1987). Airflow Resistance of Canola (Rapeseed). Transactions of the ASAE, 30(5), 1484-1488. (doi:10.13031/2013.30590)
- `canola_oswin` - Gazor, H. R. (2010). Moisture Isotherms and Heat of Desorption of Canola. Agricultural Engineering International: CIGR Journal, manuscript 1440. (CIGR manuscript 1440)
- `canola_drying` - Costa, L. M. et al. (2020). Drying kinetics of Hyola 430 hybrid canola (Brassica napus L.) seeds. Australian Journal of Crop Science, 14(10), 1623-1629. (AJCS 14(10):1623-1629)
- `canola_drying_gazor` - Gazor, H. R. et al. (2010). Modelling the drying kinetics of canola in a fluidised bed dryer. Czech Journal of Food Sciences, 28(6), 531-537. (Czech J. Food Sci. 28(6):531-537)
- `swiss_oil` - swiss granum (2026). Uebernahmebedingungen Oelsaaten, Ernte 2026. (2026-03-19 edition)
- `sunflower_sorption_drying_munder_2019` - Munder, S., Argyropoulos, D. & Müller, J. (2019). Acquisition of Sorption and Drying Data with Embedded Devices: Improving Standard Models for High Oleic Sunflower Seeds by Continuous Measurements in Dynamic Systems. Agriculture, 9(1), 1. (doi:10.3390/agriculture9010001)
- `sunflower_density_isik_izli_2007` - Isik, E. & Izli, N. (2007). Physical Properties of Sunflower Seeds (Helianthus annuus L.). International Journal of Agricultural Research, 2, 677-686. (doi:10.3923/ijar.2007.677.686)
- `sunflower_physical_gupta_das_1997` - Gupta, R. K. & Das, S. K. (1997). Physical Properties of Sunflower Seeds. Journal of Agricultural Engineering Research, 66(1), 1-8. (doi:10.1006/jaer.1996.0111)
- `sunflower_thermal_ince_2008` - Ince, R., Güzel, E. & Ince, A. (2008). Thermal Properties of Some Oily Seeds. Journal of Agricultural Machinery Science, 4(4), 399-405. (Journal of Agricultural Machinery Science 4(4):399-405)
- `sunflower_airflow_canada_1990` - Agriculture Canada (1990). Handling Agricultural Materials: Storage and Conditioning of Grain and Forage. Agriculture Canada Publication 1855/E. (Agriculture Canada Publication 1855/E)
- `sunflower_class_munder_2017` - Munder, S., Argyropoulos, D. & Müller, J. (2017). Class-based physical properties of air-classified sunflower seeds and kernels. Biosystems Engineering, 164, 124-134. (doi:10.1016/j.biosystemseng.2017.10.005)
- `swiss_oil_2026` - swiss granum (2026). Übernahmebedingungen Ölsaaten Ernte 2026, Ausgabe 19. März 2026. (2026-03-19 edition)
- `pino_airflow_project` - Albertin, R. M. (2026). PINO Airflow Porous Media. Vertiefungsprojekt, OST. (SHA-256 5fcae206aff935195eea0f8b149747c1a3e83fef16899d8533c0f89b3fbd954e)
- `transient_comsol_model_report` - COMSOL Multiphysics 6.4 model report: transient_drying_template, generated 2026-08-09. (SHA-256 463ecea45bad7c221f3c85ef53c92dbf77a603dc4b2fda2d54bf8e852d3e96a5)
- `vm2_project` - Albertin, R. M. (2026). VM2 Vertiefungsprojekt. OST. (local project report)

A citation does not imply that the final configured number appears directly in it. Follow `evidence`, `source_refs`, and any `method`, positive `verification`, `applicability`, or `note` to distinguish direct values, conversions, refits, transfers, inversions, estimates, priors, synthetic supports, and modelling decisions.

## Operational cross-reference

Technical checks validate software, native mappings, conversion, persistence, and data flow; they do not experimentally validate configured science. See the [Generation operational guide](simulation_generation.md) for host setup, configuration inspection, readiness, smoke, pilot, production, transfer, Dataset packaging, resume, retention, and cleanup.
