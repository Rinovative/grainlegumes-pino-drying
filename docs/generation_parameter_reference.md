# Generation Scientific and Configuration Parameter Reference

This is the single maintained human-readable reference for Generation
parameters, units, ranges, equations, support roles, classifications, and
scientific provenance. Validated configuration and its resolved form remain the
executable authority; this document explains their scientific meaning without
becoming a second configuration. The [operations guide](simulation_generation.md)
owns setup, execution, publication, resume, and troubleshooting.

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
canonical bibliography in `configs/generation/sources.yaml`. An empty source
list is valid only for an explicit project modelling or synthetic design
decision. Material OOD supports additionally carry a separate
`ood_provenance`, so a literature-backed ID value is never presented as evidence
for a synthetic stress interval.

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
Those concepts communicate directness and uncertainty without a second
vocabulary. Configuration ownership and edit locations are summarized once in
the [operations guide](simulation_generation.md#where-do-i-change-what).

## Inspect the resolved scientific contract

Show the complete fail-closed campaign view without changing unresolved launch
state:

```bash
# from the repository checkout
python -m src.generation.cli.cli_generation validate-config \
  configs/generation/campaigns/transient_drying/family_generalization.yaml \
  --allow-incomplete
```

The output exposes the effective material inventory, resolved parameters,
dimensions, support roles, OOD allocation, provenance, and identities.

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
Use it after a configuration edit to verify the effective values, dimensions,
supports, and identities.

## Registry and profile semantics

The registry owns the active sampling blocks, effective dimensions, names,
units, kinds, transforms, physical OOD groups, and atomic-record components.
The profile projection determines which registry entries participate in one
simulation profile. Do not maintain those inventories in prose.

Dataset task and step contracts remain the tensor-channel owners. A material
family is metadata, not an implicit one-hot channel, and an evaluation category
does not create a new model, equation, sampling coordinate, or native profile.

## Canonical parameter catalogue

The registry declares 62 canonical entries. Sampling coordinates contribute 27 dimensions to `steady_flow` and 53 to `transient_drying`; coupled records, latent packing scatter, and derived or fixed entries add no independent coordinates. OOD is admitted only when the resolved record supplies a valid tail or alternate complete record.

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
| `T_in_base` | `T_{\mathrm{in},0}` | `K` | sampled | `transient_drying` | `linear` | `operation` | `operation` | Exact temporal mean of the planned inlet-air temperature schedule. |
| `T_in_amp` | `\Delta T_{\mathrm{in}}` | `K` | sampled | `transient_drying` | `linear` | `operation` | `operation` | Maximum absolute inlet-temperature deviation from its exact temporal mean. |
| `omega_in_base` | `\omega_{\mathrm{in},0}` | `kg/kg` | sampled | `transient_drying` | `linear` | `operation` | `operation` | Exact temporal mean of the planned inlet humidity-ratio schedule. |
| `omega_in_amp` | `\Delta\omega_{\mathrm{in}}` | `kg/kg` | sampled | `transient_drying` | `linear` | `operation` | `operation` | Maximum absolute inlet humidity-ratio deviation from its exact temporal mean. |
| `schedule.corr` | `\rho_{T,\omega}` | `1` | sampled | `transient_drying` | `linear` | `operation` | `operation` | Exact discrete-node Pearson correlation when both inlet schedules vary. |
| `schedule.timescale_rel` | `\tau_{\mathrm{sched}}/t_{\max}` | `1` | sampled | `transient_drying` | `log` | `operation` | `operation` | Gaussian low-pass e-folding correlation time divided by planned duration. |
| `schedule.component_weights` | `\boldsymbol{\lambda}_{\mathrm{sched}}` | `1` | sampled | `transient_drying` | `none` | `operation` | `operation` | Relative smooth, event, and trend contributions, each applied once. |
| `schedule.event_count` | `n_{\mathrm{event}}` | `1` | sampled | `transient_drying` | `none` | `operation` | `operation` | Deterministic count of finite-duration events; zero disables only the event contribution. |
| `schedule.event_duration_rel` | `d_{\mathrm{event}}/t_{\max}` | `1` | sampled | `transient_drying` | `log` | `operation` | `operation` | Finite event duration divided by planned schedule duration. |
| `schedule.event_width_rel` | `w_{\mathrm{event}}/t_{\max}` | `1` | sampled | `transient_drying` | `log` | `operation` | `operation` | Smooth event-edge transition width divided by planned schedule duration. |
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
| `schedule.timescale_rel` | 0.05-0.18 | 0.08 | `1` | `synthetic_design` |
| `schedule.component_weights` | {'smooth': 0.55, 'event': 0.3, 'trend': 0.15} | {'smooth': 0.55, 'event': 0.3, 'trend': 0.15} | `1` | `synthetic_design` |
| `schedule.event_count` | 0-4 | 2 | `1` | `synthetic_design` |
| `schedule.event_duration_rel` | 0.08-0.16 | 0.1 | `1` | `synthetic_design` |
| `schedule.event_width_rel` | 0.02-0.04 | 0.03 | `1` | `synthetic_design` |
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
| `porosity.smooth_len_rel` | 0.03-0.09 | 0.055 | `1` | `project_baseline` |
| `porosity.texture_amp` | 0.003-0.015 | 0.008 | `1` | `project_baseline` |
| `initial_moisture.structure.coarse_len_rel` | 0.08-0.24 | 0.14 | `1` | `synthetic_design` |
| `initial_moisture.structure.fine_len_rel` | 0.025-0.08 | 0.045 | `1` | `synthetic_design` |
| `initial_moisture.structure.coarse_weight` | 0.4-0.75 | 0.58 | `1` | `synthetic_design` |
| `initial_moisture.structure.fine_ani_x` | 0.7-1.4 | 1 | `1` | `synthetic_design` |
| `initial_moisture.structure.fine_ani_y` | 0.7-1.4 | 1 | `1` | `synthetic_design` |
| `initial_moisture.structure.cross_scale_corr` | 0.25-0.7 | 0.5 | `1` | `synthetic_design` |

## Uniform material contract

Every maintained family uses the same role-neutral material schema and can
participate in every scientifically applicable material-owned OOD role. Evidence
quality and numerical values may differ by family; required fields, OOD role
inventory, provenance shape, validation, resolution, persistence, and static
sentinel coverage do not. Campaign membership remains a separate experiment
design decision.

| Contract item | Classification | Authoritative owner |
| --- | --- | --- |
| Material identity, scope, natural values, supports, targets, and source evidence | `AUTHORED` | Material YAML |
| Lower/upper scalar stress intervals and complete coupled stress records | `AUTHORED` synthetic design | Material YAML `ood_supports` or `ood_records` with separate `ood_provenance` |
| Fixed KC coefficient, KC-compatible support, mapped porosity tails, and inferred dry particle density | `DERIVED` and deterministic | Material resolver |
| Natural or OOD scalar values and selected complete records | `SAMPLED` or atomically selected | Case generator from the resolved contract |
| Bounded latent packing deviation | `SAMPLED` from its semantic seed, but not a DOE coordinate | Case generator |
| Seen/family-OOD role and production case count | `AUTHORED` experiment design | Campaign YAML |

### Natural values and supports

Moisture supports are dry basis (`kg water/kg dry solid`) except `X_target_wb`,
which is wet basis. Oswin values are the complete `A_osw, B_osw, C_osw` record;
kinetics are supports for `r_surf_0, r_int_surf, f_surf`. Exact evidence, source
references, methods, and applicability limits remain beside each value in the
material YAML.

| Material | authored kappa_mean m^2 | packing eps | rho_bu_dry_ref kg/m^3; eps ref | k_gr W/(m*K) | cp_gr_dry J/(kg*K) | initial mean db | X_target_wb | Oswin A,B,C | kinetics supports |
| --- | --- | --- | --- | --- | --- | --- | ---: | --- | --- |
| `chickpea` | 8e-09-3e-08 | 0.37-0.45 | 620-730; 0.403 | 0.24-0.38 | 1600-2200 | 0.190476-0.315789 | 0.12 | 11.2175, -0.0615631, 0.49142 | 3e-06-1.4e-05; 0.06-0.3; 0.28-0.62 |
| `field_pea` | 8e-09-4e-08 | 0.37-0.43 | 530-620; 0.3938 | 0.25-0.38 | 1650-2250 | 0.190476-0.315789 | 0.12 | 12.062, -0.0573838, 0.343383 | 3.5e-06-1.6e-05; 0.06-0.3; 0.28-0.62 |
| `kidney_bean` | 1.5e-08-8e-08 | 0.42-0.5 | 510-620; 0.456 | 0.25-0.4 | 1600-2200 | 0.219512-0.388889 | 0.135 | 14.0761, -0.0763, 0.462107 | 2e-06-1e-05; 0.04-0.22; 0.22-0.55 |
| `lentil` | 5e-09-1.3e-08 | 0.29-0.36 | 590-680; 0.3125 | 0.3-0.44 | 1800-2300 | 0.190476-0.315789 | 0.12 | 12.062, -0.0573838, 0.343383 | 5e-06-2e-05; 0.08-0.35; 0.3-0.65 |
| `rapeseed` | 2e-09-9e-09 | 0.365-0.422 | 590-640; 0.4 | 0.15-0.25 | 1600-2100 | 0.0989011-0.190476 | 0.06 | 6.95472, -0.032779, 0.470207 | 5e-06-2.5e-05; 0.08-0.4; 0.3-0.68 |
| `sunflower_seed` | 4e-08-1.5e-07 | 0.5-0.57 | 330-410; 0.5306 | 0.16-0.3 | 500-1000 | 0.0989011-0.219512 | 0.06 | 4.8, -0.0158, 0.622278 | 3e-06-1.6e-05; 0.05-0.3; 0.25-0.65 |

### Required material OOD inventory

Every family authors exactly two ordered tails for `kappa_mean`, `k_gr`, and
`cp_gr_dry` (lower, then upper); one upper tail for initial-moisture mean and
amplitude; complete `loose_low_density` and `dense_high_density` records at the
reference inferred dry particle density; and complete `slow_internal_limited`
and `fast_surface_exposed` kinetics records. Missing, extra, misdirected, or
mis-provenanced roles fail material resolution.

| Material | kappa lower / upper m^2 | k_gr lower / upper | cp_gr_dry lower / upper | initial mean upper | amplitude upper |
| --- | --- | --- | --- | --- | --- |
| `chickpea` | 2.5e-09-5e-09 / 4.8e-08-8e-08 | 0.16-0.20 / 0.46-0.58 | 1200-1450 / 2500-2900 | 0.36-0.45 | 0.055-0.080 |
| `field_pea` | 2.4e-09-5.2e-09 / 6.4e-08-1.1e-07 | 0.16-0.21 / 0.46-0.58 | 1200-1500 / 2550-2950 | 0.36-0.45 | 0.055-0.080 |
| `kidney_bean` | 4e-09-9e-09 / 1.3e-07-2.2e-07 | 0.16-0.21 / 0.49-0.62 | 1200-1450 / 2500-2900 | 0.44-0.55 | 0.065-0.095 |
| `lentil` | 1.8e-09-3.5e-09 / 2e-08-3.8e-08 | 0.20-0.26 / 0.52-0.65 | 1300-1600 / 2600-3000 | 0.36-0.45 | 0.055-0.080 |
| `rapeseed` | 6e-10-1.3e-09 / 1.44e-08-2.475e-08 | 0.09-0.12 / 0.31-0.40 | 1200-1450 / 2400-2800 | 0.23-0.30 | 0.035-0.050 |
| `sunflower_seed` | 1.2e-08-2.6e-08 / 2.4e-07-4.125e-07 | 0.10-0.13 / 0.38-0.50 | 300-400 / 1200-1500 | 0.27-0.36 | 0.042-0.060 |

| Material | density records: rho_bu_dry_ref, eps_bed_cal_ref | kinetics records: r_surf_0, r_int_surf, f_surf |
| --- | --- | --- |
| `chickpea` | loose 505, 0.55401627; dense 855, 0.24491864 | slow 1.75e-06, 0.075, 0.30; fast 2.45e-05, 0.225, 0.60 |
| `field_pea` | loose 430, 0.54116177; dense 710, 0.24238338 | slow 2e-06, 0.08, 0.31; fast 2.8e-05, 0.24, 0.61 |
| `kidney_bean` | loose 415, 0.59685714; dense 740, 0.28114286 | slow 1.25e-06, 0.05, 0.23; fast 1.75e-05, 0.15, 0.53 |
| `lentil` | loose 475, 0.48164683; dense 720, 0.21428571 | slow 2.5e-06, 0.10, 0.35; fast 3.5e-05, 0.30, 0.65 |
| `rapeseed` | loose 500, 0.51612903; dense 800, 0.22580645 | slow 3e-06, 0.11, 0.35; fast 4.2e-05, 0.33, 0.65 |
| `sunflower_seed` | loose 270, 0.66418124; dense 610, 0.24129836 | slow 2e-06, 0.075, 0.30; fast 2.8e-05, 0.225, 0.60 |

The current family-generalization campaigns intentionally allocate production
parameter-OOD cases only to the three `seen` families; the other three families
remain natural-support family-OOD evaluations. That campaign policy is not a
material-schema exception: static sentinels resolve and exercise the complete
material-owned OOD inventory for all six families.

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

### Material-calibrated Kozeny--Carman trend and packing scatter

The Kozeny--Carman relation defines a material-specific reference trend between
mean permeability and packing porosity. For
`g(epsilon) = epsilon^3/(1-epsilon)^2`, the generator derives one fixed
`A_KC_reference = kappa_nominal/g(eps_bed_cal_ref)` from the canonical material
reference record. A case-specific density-calibration OOD record does not change
this coefficient. The deterministic trend is
`eps_kc_trend = g^-1(kappa_mean/A_KC_reference)`, so `kappa_mean` is the only DOE
coordinate governing the global permeability/packing coupling and the trend is
strictly increasing.

The authored permeability support is intersected before sampling with the
permeability interval obtained by mapping the natural packing-porosity support
through the fixed KC relation. The resolved configuration preserves both
intervals and labels their intersection as the effective joint ID support; it
never presents the narrowed interval as the authored source range.

| Material | Authored permeability | KC-compatible permeability | Effective ID permeability | Natural porosity | Lower/upper permeability-OOD mapped porosity |
| --- | --- | --- | --- | --- | --- |
| `chickpea` | 8e-09--3e-08 | 9.93788e-09--2.34575e-08 | 9.93788e-09--2.34575e-08 | 0.37--0.45 | 0.260018--0.312058 / 0.521007--0.572562 |
| `field_pea` | 8e-09--4e-08 | 1.26711e-08--2.42965e-08 | 1.26711e-08--2.42965e-08 | 0.37--0.43 | 0.240657--0.296068 / 0.525511--0.580167 |
| `kidney_bean` | 1.5e-08--8e-08 | 2.49518e-08--5.66474e-08 | 2.49518e-08--5.66474e-08 | 0.42--0.50 | 0.266757--0.329360 / 0.583695--0.636010 |
| `lentil` | 5e-09--1.3e-08 | 5.90471e-09--1.39017e-08 | 5.90471e-09--1.3e-08 | 0.29--0.36 | 0.209637--0.252181 / 0.392517--0.453211 |
| `rapeseed` | 2e-09--9e-09 | 2.71340e-09--5.06132e-09 | 2.71340e-09--5.06132e-09 | 0.365--0.422 | 0.247228--0.303703 / 0.524789--0.579445 |
| `sunflower_seed` | 4e-08--1.5e-07 | 5.89990e-08--1.18185e-07 | 5.89990e-08--1.18185e-07 | 0.50--0.57 | 0.350184--0.420045 / 0.640540--0.692071 |

A small bounded case-level deviation represents unresolved packing morphology.
For the active natural or mapped `kappa_mean`-OOD porosity support, the generator
uses `margin = min(eps_kc_trend-lower, upper-eps_kc_trend)`,
`sigma = margin/3`, draws a true standard normal truncated to `(-3, 3)` from its
own semantic `packing_scatter` seed, and sets
`eps_reference = eps_kc_trend + sigma*packing_scatter_z`. The symmetric scatter
is a synthetic modelling assumption, not an experimentally calibrated
parameter. It is not a DOE coordinate, OOD unit, configurable scientific
parameter, model input, or dataset channel. The draw and scalar reference are
fixed across retries of local spatial fields.

Local morphology remains `eps(x) = eps_reference + Delta-epsilon*chi(x)` using
the existing shared background, texture, pointwise physical guards, and clipping
diagnostics. No pointwise Kozeny--Carman relation is imposed between
`kappa(x)` and `eps(x)`. Lower and upper `kappa_mean` OOD intervals use their
complete KC-mapped porosity intervals for scatter, so scatter cannot return an
OOD case to the natural packing interval. Case provenance records the active
permeability interval as `active_kappa_mean_support` beside the authored,
KC-compatible, and effective ID intervals.

The active generation and persisted contract version is exactly `1`. Changed
registry, resolved-support, seed, metadata, case, batch, HDF5, and dataset
content changes the existing digests and identities. Dataset package identity
and admission include the exact current case/porosity contract digest, so old
version-1 anchor-factor artifacts are not reinterpreted, migrated, or loadable
as current packages.

### Inlet schedule

`generation_cases_schedule.py` is the one canonical temporal generator for
natural and parameter-OOD views. It is deliberately separate from the
spatial pressure and field generator. The smooth component filters seeded
one-dimensional white excitation with a reflected Gaussian kernel;
`schedule.timescale_rel` is the ideal filtered process's e-folding correlation
time divided by the planned horizon. Finite-duration step-like and pulse events
use `schedule.event_count`, duration, and smooth edge width, while the trend is a
horizon-scale drift without high-frequency structure.

The smooth, event, and trend simplex values are relative contributions applied
once. Event count, rather than a hidden Bernoulli draw, determines event
presence. After complete composition, each latent is centered and normalized
once. `T_in_base` and `omega_in_base` are therefore exact temporal means, and
their amplitudes are exact maximum absolute deviations. The independent
humidity latent is orthogonalized against the shared temperature latent on the
actual schedule nodes, so `schedule.corr` is the realized discrete-time Pearson
correlation whenever both schedules vary; correlation is not applicable for an
intentional constant signal.

All characteristic scales are validated against `common.time.interval`.
Natural supports resolve the low-pass correlation time to at least 8.4 intervals,
event edges to at least 3.36 intervals, and event durations to at least 13.44
intervals on the maintained 168-hour, one-hour grid. Faster parameter-OOD tails
remain resolved at minima of 4.032, 2.016, and 6.72 intervals respectively.
Event duration is also at least twice its edge width. These schedule supports
remain `synthetic_design`; they are not presented as literature measurements.

The generator creates `T_in_bc(t)` and `omega_in_bc(t)` and derives
`phi_in_bc(t)` thermodynamically from those values and `p_ref`; relative humidity
is never sampled or smoothed independently. The complete schedule is checked
against temperature, humidity-ratio, inlet-RH, source-air saturation, and
heater-only constraints and is deterministically retried as one unit without
clipping individual nodes. Natural and parameter-OOD behavior differ only by
supports within this same process family. Diagnostics remain provenance and
evaluation metadata, not learned fields or scalar inputs.

### Transient scalar handoff

The Generation profile contract owns one ordered 12-field case-dependent
runtime handoff. Each transient case writes exactly these values to
`scalars.csv`; runtime admission and canonical HDF5 publication require the same
names, order, units, finite float64 values, and case representations.

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

`T_init` is derived from `T_amb`. Template-fixed values are not duplicated into
the case handoff. The learned transient Dataset view is a separate eight-field
projection: the three transfer terms, three Oswin coefficients, `k_gr`, and
`cp_gr_dry`. `T_amb` is step-boundary conditioning; the realized `eps_bed` and
`rho_bu_dry` static fields represent the two density-calibration values;
`X_target_wb` controls termination diagnostics.

The independent time-dependent adapter contains
`t,T_in_bc,omega_in_bc,phi_in_bc`. Exact adapter, runtime, native mapping,
solver-output, and publication procedures belong to the operations guide and
executable contracts.

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
from registry classification, profile projection, parameter kind, and the
required material-owned tails or alternate records. Complete material
capability is validated independently of campaign membership. Deterministic
round-robin allocation covers every eligible unit when capacity permits and
balances remaining cases.
The resolved summary and persisted provenance provide, per batch:

- eligible unit identifiers, physical groups, and unit kinds;
- per-unit allocation counts and exact per-case assignment;
- selected record or tail, realized value, natural and OOD support, transform,
  and transform-space distance at case generation time.

Scalar tails are nonempty, separated from natural support in the declared
transform, and fail closed when invalid. Natural cases have no active OOD unit.

## Persisted scientific evidence

Resolved configuration and generated artifacts preserve exact scientific and
case identities, material family and role, natural and OOD supports, selected
values or atomic records, material OOD provenance, semantic seeds, coupled KC
and packing-scatter evidence, diagnostics, hashes, and the embedded resolved
scientific configuration. Exact admission rejects stale or structurally
incompatible version-1 artifacts. File layouts, adapters, publication, and
Dataset workflow are specified in the operations guide and executable
contracts.

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

A citation does not imply that the final configured number appears directly in
it. Follow the complete provenance chain to distinguish direct values,
conversions, refits, transfers, inversions, estimates, priors, synthetic
supports, and modelling decisions.

## Operational cross-reference

Technical checks validate software, native mappings, conversion, persistence, and data flow; they do not experimentally validate configured science. See the [Generation operational guide](simulation_generation.md) for host setup, configuration inspection, readiness, smoke, pilot, production, transfer, Dataset packaging, resume, retention, and cleanup.
