# VP2 Generation Parameter and Config Reference

YAML and Python validation are the executable authority. This reference gives
the compact scientific contract, canonical names and units, ownership rules,
and inspection commands. Numeric material records and their source evidence
remain in `configs/generation/materials/<family>.yaml`; they are not duplicated
here.

## Scientific parameter note

> The configured values and ranges are executable modelling and sampling
> decisions, not universal experimentally validated material constants. A value
> may be directly reported, refitted, convention- or unit-converted, transferred,
> engineering-calculated or inverted, selected as a calibration prior, or
> selected as synthetic generator design. A citation therefore does not imply
> that the final configured number appears directly in that source. The
> machine-readable `status`, `derivation`, `confidence`, and `validity` fields
> define the authoritative interpretation. Software, COMSOL, smoke, and pilot
> passes validate runtime and data flow only; they do not experimentally validate
> the configured science.

## Owners at a glance

| Owner | Scientific decision |
| --- | --- |
| `configs/generation/sources.yaml` | One central bibliographic record per supplied source key |
| `configs/generation/registry.yaml` | Parameter name, unit, kind, transform, block, OOD group, components, derivation |
| `configs/generation/common.yaml` | Grid/time, shared fixed physics, formulas, adapters, HDF5 contract |
| `configs/generation/operations/fixed_bed.yaml` | Apparatus/operation supports and constraints |
| `configs/generation/materials/<family>.yaml` | Role-neutral family values, natural supports, coupled records, targets, evidence |
| `configs/generation/profiles/<profile>.yaml` | Template, export schema, explicit mappings, stationary conditioning |
| `configs/generation/campaigns/<profile>/<campaign>.yaml` | Roles, counts, seeds, package declarations |
| `configs/generation/execution/cluster_cpu.yaml` | Execution only; excluded from scientific identity |

The decision-package digest bound into scientific provenance is:

```text
774ce0e39bf989ad77b5fe80e37c364f46ff83b3c6be1bd7410ea4c72d7269f5
```

## Sampling blocks and dimensions

The only active contracts are 28 dimensions for `steady_flow` (the `airflow`
block) and 54 for `transient_drying` (all four blocks below). There is no 53D,
pair-mode, alternate-density mode, 55th coordinate, or `alpha_duration`
coordinate. An alternate density-calibration record remains one atomic OOD
selection, not another dimension.

| Block | Effective dimensions |
| --- | ---: |
| `airflow` | 28 |
| `initial_moisture` | 8 |
| `operation` | 12 |
| `material_properties` | 6 |
| Total | 54 |

The operation block has eleven semantic entries because
`schedule.component_weights` is a three-component simplex with two independent
coordinates. Derived complements such as the fine multiscale weights add no
coordinate.

## Canonical sampled names and units

`d` is the effective numerical dimension contributed by the semantic entry.

### `airflow` - 28 dimensions

| Name | Unit | Kind/transform | d |
| --- | --- | --- | ---: |
| `kappa_mean` | `m^2` | interval/log | 1 |
| `kappa_cv` | `1` | interval/linear | 1 |
| `bed.structure.coarse_len_rel` | `1` | interval/log | 1 |
| `bed.structure.fine_len_rel` | `1` | interval/log | 1 |
| `bed.structure.coarse_weight` | `1` | interval/logit | 1 |
| `bed.structure.cross_scale_corr` | `1` | interval/linear | 1 |
| `bed.structure.fine_ani_x` | `1` | interval/log | 1 |
| `bed.structure.fine_ani_y` | `1` | interval/log | 1 |
| `bed.perturbations.amplitude` | `1` | interval/linear | 1 |
| `bed.perturbations.granularity` | `1` | interval/linear | 1 |
| `bed.perturbations.sign_bias` | `1` | interval/linear | 1 |
| `permeability.anisotropy.max_ratio` | `1` | interval/linear | 1 |
| `permeability.anisotropy.exponent` | `1` | interval/linear | 1 |
| `permeability.anisotropy.strength` | `1` | interval/linear | 1 |
| `permeability.orientation.jitter` | `1` | interval/linear | 1 |
| `permeability.orientation.smooth_len_rel` | `1` | interval/log | 1 |
| `porosity.kc_anchor_factor` | `1` | conditional interval/conditional log | 1 |
| `porosity.smooth_len_rel` | `1` | interval/linear | 1 |
| `porosity.texture_amp` | `1` | interval/linear | 1 |
| `pressure_bc.mean` | `Pa` | interval/linear | 1 |
| `pressure_bc.sin_amp` | `1` | interval/linear | 1 |
| `pressure_bc.sin_freq` | `1` | interval/linear | 1 |
| `pressure_bc.sin_phase` | `rad` | interval/phase | 1 |
| `pressure_bc.gauss_count` | `1` | integer | 1 |
| `pressure_bc.gauss_amp` | `1` | interval/linear | 1 |
| `pressure_bc.gauss_width` | `1` | interval/log | 1 |
| `pressure_bc.gauss_jitter` | `1` | interval/linear | 1 |
| `pressure_bc.linear_amp` | `1` | interval/linear | 1 |

Pressure-profile entries belong to the numerical `airflow` block because they
generate `p_in_bc`; their physical OOD group is `operation`.

### `initial_moisture` - 8 dimensions

| Name | Unit | Kind/transform | d |
| --- | --- | --- | ---: |
| `initial_moisture.mean_db` | `kg/kg` | interval/linear | 1 |
| `initial_moisture.amplitude_db` | `kg/kg` | interval/linear | 1 |
| `initial_moisture.structure.coarse_len_rel` | `1` | interval/log | 1 |
| `initial_moisture.structure.fine_len_rel` | `1` | interval/log | 1 |
| `initial_moisture.structure.coarse_weight` | `1` | interval/logit | 1 |
| `initial_moisture.structure.cross_scale_corr` | `1` | interval/linear | 1 |
| `initial_moisture.structure.fine_ani_x` | `1` | interval/log | 1 |
| `initial_moisture.structure.fine_ani_y` | `1` | interval/log | 1 |

### `operation` - 12 dimensions

| Name | Unit | Kind/transform | d |
| --- | --- | --- | ---: |
| `T_in_base` | `K` | interval/linear | 1 |
| `T_in_amp` | `K` | interval/linear | 1 |
| `omega_in_base` | `kg/kg` | interval/log | 1 |
| `omega_in_amp` | `kg/kg` | interval/linear | 1 |
| `schedule.corr` | `1` | interval/linear | 1 |
| `schedule.timescale_rel` | `1` | interval/log | 1 |
| `schedule.component_weights` | `1` | simplex: `smooth,event,trend` | 2 |
| `schedule.event_count` | `1` | integer | 1 |
| `schedule.event_duration_rel` | `1` | interval/linear | 1 |
| `schedule.event_width_rel` | `1` | interval/linear | 1 |
| `T_amb` | `K` | interval/linear | 1 |

### `material_properties` - 6 dimensions

| Name | Unit | Kind/transform | d |
| --- | --- | --- | ---: |
| `rho_bu_dry_ref` | `kg/m^3` | interval/log | 1 |
| `k_gr` | `W/(m*K)` | interval/log | 1 |
| `cp_gr_dry` | `J/(kg*K)` | interval/log | 1 |
| `r_surf_0` | `1/s` | interval/log | 1 |
| `r_int_surf` | `1` | interval/log | 1 |
| `f_surf` | `1` | interval/logit | 1 |

Material-specific supports, nominals, target moisture, Oswin records, density
records, and kinetics records are owned by the six material YAML files.

## Binding formulas

The exact formula strings are owned by `common.yaml`; the table below is the
short scientific cross-check.

| Quantity | Binding |
| --- | --- |
| Granular water | `w_gr = f_surf*w_surf + (1-f_surf)*w_int` |
| Surface balance | `f_surf*d(w_surf)/dt = j_int - m_evap` |
| Internal balance | `(1-f_surf)*d(w_int)/dt = -j_int` |
| Initial water | `w_gr_0 = rho_bu_dry*X_0_db_field; w_surf(0)=w_int(0)=w_gr_0` |
| Internal transfer | `j_int = (1-f_surf)*r_int*(w_int-w_surf)` |
| Evaporation | `m_evap = f_surf*r_surf*max(w_surf-w_eq,0)` |
| Rates | `r_surf=r_surf_0; r_int=r_int_surf*r_surf` |
| Moisture bases | `X_db=w_gr/rho_bu_dry; X_wb=X_db/(1+X_db)` |
| Bulk moisture | `X_wb_bulk=integral(w_gr)/(integral(rho_bu_dry)+integral(w_gr))` |
| Dry bulk density | `rho_bu_dry=rho_bu_dry_ref*(1-eps_bed)/(1-eps_bed_cal_ref)` |
| Heat capacity | `cp_gr_eff=cp_gr_dry+X_db*cp_w` |
| Conductivity | `k_eff=k_gr*(2*k_gr+k_air-2*eps_bed*(k_gr-k_air))/(2*k_gr+k_air+eps_bed*(k_gr-k_air))` |
| Oswin equilibrium | `X_eq_db=0.01*(A_osw+B_osw*(T-273.15[K]))*(phi_eff/(1-phi_eff))^C_osw` |
| Equilibrium water | `w_eq=rho_bu_dry*X_eq_db` |
| Initial humidity | `phi_init=osw_ratio_0/(1+osw_ratio_0)` with `osw_ratio_0=(100*X_0_db_field/(A_osw+B_osw*(T_init-273.15[K])))^(1/C_osw)` |
| Latent heat | `Q_evap=-h_fg*m_evap` |
| Total-water balance | `d/dt(m_w_gr+m_v_gas)=m_dot_v_in-m_dot_v_out` |

`phi_eff=min(max(phi,1e-6),0.999)`. Moisture quantities carrying `_db` are dry
basis; `_wb` are wet basis.

### Material-calibrated porosity coupling

There is one physical porosity field, `eps_bed(x,y)`. The scalar
`eps_bed_cal_ref` is the calibration reference inherited from the active
complete density-calibration record; it is not another field.
`packing_porosity_mean_support=[eps_nat_lo,eps_nat_hi]` is the material-specific
natural interval for the representative and realised mean packing, not a
sampling coordinate or COMSOL scalar. The universal `eps_min_global=0.2` and
`eps_max_global=0.8` remain pointwise guards rather than material-natural
supports.

Define

```text
g(eps) = eps^3/(1-eps)^2
A_KC_ref = kappa_nom/g(eps_bed_cal_ref)
A_KC_case = kc_anchor_factor*A_KC_ref
kappa_mean = A_KC_case*g(eps_reference)
```

`A_KC_ref` is derived from the selected material nominal permeability and the
active density-calibration reference; it is never authored or sampled. The
last equation is inverted deterministically inside the global guards. Therefore
`kappa_mean=kappa_nom` and `kc_anchor_factor=1` recover
`eps_reference=eps_bed_cal_ref` to root-solver precision.

For an already sampled `kappa_mean`, the material-natural conditional factor
support is

```text
a_ID_lower(kappa_mean) = kappa_mean/(A_KC_ref*g(eps_nat_hi))
a_ID_upper(kappa_mean) = kappa_mean/(A_KC_ref*g(eps_nat_lo))
```

The ordinary LHS unit coordinate is mapped logarithmically into this positive
case- and material-specific interval. Thus `porosity.kc_anchor_factor` is one
sampled airflow coordinate with report symbol `a_KC`, but its physical support
is conditional; no six material-specific factor ranges are authored.

For Seen-material parameter OOD, let
`L=log(a_ID_lower)`, `U=log(a_ID_upper)`, and `W=U-L`. The exact synthetic tails
are

```text
conditional ID: log(a_KC) in [L,U]
lower tail:    log(a_KC) in [L-0.40*W, L-0.15*W]
upper tail:    log(a_KC) in [U+0.15*W, U+0.40*W]
transformed gap = 0.15*W
tail width      = 0.25*W
```

A lower factor produces looser, higher-porosity global packing; an upper factor
produces denser, lower-porosity packing. These tails have `synthetic_design`
status and represent conditional extrapolation beyond material-natural packing
support, not literature packing ranges. Field pea, Rapeseed, and Sunflower use
natural supports only.

Local morphology remains strictly background-field based:

```text
porosity_latent = smooth(z_background, porosity.smooth_len_rel)
chi = zero-mean, unit-RMS normalization(porosity_latent)
eps_unbounded = eps_reference + porosity.texture_amp*chi
eps_bed = pointwise application of [eps_min_global,eps_max_global]
```

No realised local permeability scalar, tensor component, eigenvalue,
determinant, or perturbation directly enters that texture. Permeability and
porosity share `z_background`. Within porosity generation, `kappa_mean` enters
only the global Kozeny-Carman anchor. The same realised `eps_bed` is consumed by steady
airflow, transient moist-air volume, heat transfer, effective conductivity,
dry bulk density, water inventory, COMSOL, HDF5, and the maintained learning
views.

The inlet schedule is heater-only: source air supplies humidity ratio,
heating conserves `omega_in_bc`, and `phi_in_bc` is derived at the inlet
temperature. Every accepted complete schedule satisfies `T_in_bc >= T_amb`,
positive humidity ratio, source RH in `(0,1]`, inlet RH in `[0.05,0.85]`, and
the configured temperature/humidity envelopes. Generation retries a whole
schedule deterministically; it never clips individual samples.
`phi_source_air` is validation/provenance only and is not a new adapter column.

## Material schema and coupled records

Every material YAML has these role-neutral top-level keys:

```text
schema_kind
schema_version
material_family
decision_source
material_scope
permeability
packing_porosity_mean_support
density_calibration
thermal_properties
initial_moisture
target_moisture
oswin
two_compartment_kinetics
```

`material_scope` fixes the common name, species, market class, product form,
coat or hull state, and concise description for each family. `decision_source`
binds the authoritative YAML digest. Each independent value or complete record
retains its supplied source reference, status, derivation, confidence, and
validity. A technical smoke never upgrades evidence status.

Atomic selection rules:

- Natural sampling varies `rho_bu_dry_ref` while its material-specific
  `eps_bed_cal_ref` remains fixed; density parameter OOD replaces both with one
  complete configured density-calibration record.
- Oswin selects one complete `(A_osw, B_osw, C_osw)` record with its equation
  convention and validity domain.
- Kinetics selects one complete `(r_surf_0, r_int_surf, f_surf)` record.
- `schedule.component_weights` selects the complete
  `(smooth, event, trend)` simplex vector.
- Components from different records are never mixed.
- Parameter OOD selects a complete coupled record; it never creates
  component-wise tails.

## Family roles and OOD

| Role/evaluation regime | Materials | Support and eligibility |
| --- | --- | --- |
| `id` | Lentil, Chickpea, Kidney bean | Natural support; eligible for train, validation, and ID test |
| `parameter_ood` | Lentil, Chickpea, Kidney bean | Disjoint parameter support; evaluation only |
| `near_family_ood` | Field pea | Natural support only; evaluation only |
| `far_family_ood` | Rapeseed | Natural support only; evaluation only |
| `extreme_family_ood` | Sunflower seed | Natural support only; evaluation only |

The evaluation-regime set is exactly the five rows above. Technical smoke is a
campaign purpose and operational membership, not an evaluation regime.

Each parameter-OOD case activates exactly one unit in exactly one physical OOD
group:

| OOD group | Examples of eligible units |
| --- | --- |
| `bed` | Permeability and porosity structure |
| `operation` | Pressure profile, inlet schedule, ambient temperature |
| `initial_moisture` | Initial-moisture mean and structure |
| `material_properties` | Density calibration, thermal properties, complete kinetics record |

Eligible units are derived from registry classification, profile projection,
parameter kind, and actual OOD tails or alternate atomic records. The
allocation covers every eligible unit where possible and distributes remaining
cases deterministically with counts differing by at most one; it does not use
fixed group quotas. The selected unit persists its group, unit ID, record/tail
ID, realized value, natural support, OOD support, transform, and
transform-space distance. Scalar tails are nonempty, hard-gap separated, and
disjoint from ID in the declared transform. The humidity hard-boundary
exception is local; it does not weaken the width rule globally. Natural cases
have no active OOD unit.

Family OOD never invokes parameter OOD. In particular, Sunflower has no
parameter-OOD, training, validation, or ID-test membership and requires no
special generator, package builder, loader, evaluator, equation, coordinate,
or profile.

## Fields, adapters, and persistence

The fixed grid is boundary-inclusive:

```text
Lx=1.2 m, Ly=0.75 m, Lz=0.8 m
nx=401, ny=251, dx=dy=0.003 m
array order=[ny,nx]=[251,401]
regular output time=0..168 h in 1 h steps
```

`dx`, `dy`, and `U_wall` are resolved from their supplied formulas and are not
independently authored or sampled.

The steady case adapter contains only:

```text
fields.csv: x, y, Kxx, Kxy, Kyy, eps_bed, p_in_bc
```

The transient adapters add:

```text
fields.csv:   X_0_db_field
scalars.csv:  T_flow_ref, p_ref, p_out, T_init, T_amb, T_in_ref,
              eps_bed_cal_ref, rho_bu_dry_ref, k_gr, cp_gr_dry,
              X_target_wb, r_surf_0, r_int_surf, f_surf,
              A_osw, B_osw, C_osw, f_wet_dm_max
schedule.csv: t, T_in_bc, omega_in_bc, phi_in_bc
```

The canonical HDF5 persists scientific identities, `material_role`,
`evaluation_regime`, `sampling_regime`, `natural_support_state`, realized
sampled values and units, block and seed evidence, complete OOD and coupled
selection provenance, inputs, outputs, diagnostics, hashes, template identity,
and source-export identity. Dataset packages validate those values against
their declared source role and regime before admission.

## Learning views

The programmatic task/step contracts remain the sole authoritative channel
owners; these tables are validated against them.

### Steady learning view

| Tensor | Channel order |
| --- | --- |
| Inputs | `x`, `y`, `Kxx`, `Kxy`, `Kyy`, `eps_bed`, `p_in_bc` |
| Targets | `p`, `u`, `v` |

### Transient Neural-Operator learning view

| Tensor | dtype | Shape | Channel order / definition |
| --- | --- | --- | --- |
| State at `t_n` | `float32` | `[4,251,401]` | `T`, `phi`, `w_surf`, `w_int` |
| Static fields | `float32` | `[7,251,401]` | `x`, `y`, `u`, `v`, `p`, `eps_bed`, `rho_bu_dry` |
| Boundary conditioning | `float32` | `[5]` | `T_in_bc(t_n)`, `T_in_bc(t_n+1)`, `phi_in_bc(t_n)`, `phi_in_bc(t_n+1)`, `T_amb` |
| Material/scientific scalars | `float32` | `[8]` | `r_surf_0`, `r_int_surf`, `f_surf`, `A_osw`, `B_osw`, `C_osw`, `k_gr`, `cp_gr_dry` |
| Target | `float32` | `[4,251,401]` | `delta_T`, `delta_phi`, `delta_w_surf`, `delta_w_int`; `q(t_n+1) - q(t_n)` |
| Time step | — | scalar | `dt = 1 h` |

`omega_in_bc` remains schedule/provenance and is not another baseline learning
channel. Material family is metadata, not a one-hot channel. `Kxx`, `Kxy`,
`Kyy`, `p_in_bc`, and `X_0_db_field` are source/provenance or explicit ablation
fields, not baseline transient channels. Exact irregular stop states are
runtime diagnostics and never regular training transitions.

This evaluation/category expansion creates no model, training regime, operator
channel, equation, sampling coordinate, or COMSOL profile.

## Inspect the resolved contract

Show a complete fail-closed campaign view and selected resolved provenance
chains without changing unresolved launch state:

```bash
cd /workspace/repo
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

Atomic components expose their complete inherited record provenance. Grid and
time components, derived physical formulas, and the family-specific porosity
support guard are inspectable through the same view. Fixed-value inspection
distinguishes generator consumption, scalar-adapter handoff, and constants that
Python only binds to the canonical template without a runtime setter.

Run every static scientific sentinel:

```bash
python -m src.generation.cli.cli_generation static-sentinels \
  configs/generation/campaigns/steady_flow/family_generalization.yaml \
  configs/generation/campaigns/transient_drying/family_generalization.yaml
```

Show exact production, scientific, mapping, and runtime blockers:

```bash
python -m src.generation.cli.cli_generation readiness-report \
  configs/generation/campaigns/steady_flow/family_generalization.yaml \
  configs/generation/campaigns/transient_drying/family_generalization.yaml \
  --run-static-sentinels
```

For execution, transfer, smoke, inspection, resume, and cleanup, use the
[generation workflow](simulation_generation.md).

## Source catalogue

`configs/generation/sources.yaml` is the sole bibliographic owner. This concise
view lists every maintained source once. The role/use column is derived from
current executable `source_refs`; a supporting source with no direct reference
is stated as such rather than assigned speculatively.

<!-- source-catalogue:start -->
| Source key | Full citation | Identifier | Canonical locator | Supported role/use |
| --- | --- | --- | --- | --- |
| `ba` | Albertin, R. M. (2025). Trocknung von Koernerleguminosen. Bachelorarbeit, OST. | SHA-256 0d59f098 (full digest recorded in input coverage) | `local:upload/Albertin_2025_Trocknung_von_Koernerleguminosen_Bachelorarbeit(3).pdf` | Executable source_refs — shared: T_in_min, T_in_max, d_wall, k_wall, h_ext; chickpea: permeability, initial_moisture; field_pea: permeability; kidney_bean: permeability, initial_moisture; lentil: permeability, packing_porosity_mean_support, density_calibration, initial_moisture; operation: T_in_base, T_in_amp, T_amb. |
| `matouk_thermal` | Matouk, A. M. et al. (2018). Thermal Properties of Some Legume Seeds. Journal of Soil Sciences and Agricultural Engineering, 9, 261-267. | doi:10.21608/jssae.2018.35758 | `https://doi.org/10.21608/jssae.2018.35758` | Executable source_refs — chickpea: thermal_properties; field_pea: thermal_properties; kidney_bean: thermal_properties; lentil: thermal_properties. |
| `lentil_oswin_source` | Cenkowski, S., Jayas, D. S., Pabis, S. & Muir, W. E. (2015). Sorption characteristics of red lentils during storage. Canadian Biosystems Engineering, 57, 3.1-3.8. | doi:10.7451/CBE.2015.57.3.1 | `https://library.csbe-scgab.ca/docs/journal/57/C15220.pdf` | Executable source_refs — field_pea: oswin; lentil: oswin. |
| `lentil_sorption_menkov` | Menkov, N. D. (2000). Moisture sorption isotherms of lentil seeds at several temperatures. Journal of Food Engineering, 44, 205-211. | doi:10.1016/S0260-8774(00)00028-5 | `https://doi.org/10.1016/S0260-8774(00)00028-5` | Supplied supporting source; no current executable record directly cites this key. |
| `lentil_drying` | Mangueira, E. R. et al. (2021). Analysis of the thin layer drying kinetic of brown lentil grains. Research, Society and Development, 10. | RSD article 19258 | `https://rsdjournal.org/rsd/article/view/19258` | Executable source_refs — lentil: two_compartment_kinetics. |
| `swiss_crops` | swiss granum (2026). Empfehlungen fuer Uebernahmebedingungen und gesetzliche Bestimmungen von Ackerkulturen zur menschlichen Ernaehrung, Ernte 2026. | 2026-03-12 edition | `local:upload/2026-03-12_Empfehlung_Uebernahmebedingungen_Ackerkulturen_zur_menschl_Ernaehrung_2026_D(3).pdf` | Executable source_refs — chickpea: target_moisture; field_pea: target_moisture; lentil: target_moisture. |
| `chickpea_physical` | Guerhan, R., Oezarslan, C., Topuz, N., Akbas, T. & Simsek, E. (2009). Effects of Moisture Content on Physical Properties of Black Kabuli Chickpea (Cicer arietinum L.) Seed. Asian Journal of Chemistry, 21(4), 3270-3278. | Asian Journal of Chemistry 21(4):3270-3278 | `https://asianpubs.org/index.php/ajchem/article/download/12450/12431` | Executable source_refs — chickpea: packing_porosity_mean_support, density_calibration. |
| `chickpea_oswin_source` | Armstrong, P. R., Maghirang, E. B., Bhadriraju, S., & McNeill, S. G. (2017). Equilibrium Moisture Content of Kabuli Chickpea, Black Sesame, and White Sesame Seeds. Applied Engineering in Agriculture, 33(5), 737-742. | doi:10.13031/aea.12460 | `https://doi.org/10.13031/aea.12460` | Executable source_refs — chickpea: oswin. |
| `chickpea_drying` | Cavalcanti-Mata, M. E. R. M. et al. (2020). A new approach to traditional drying models for thin-layer drying kinetics of chickpeas. Journal of Food Process Engineering, 43. | doi:10.1111/jfpe.13569 | `https://doi.org/10.1111/jfpe.13569` | Executable source_refs — chickpea: two_compartment_kinetics. |
| `kidney_physical` | Isik, E. & Unal, H. (2011). Some physical properties of white kidney beans (Phaseolus vulgaris L.). African Journal of Biotechnology, 10. | stable journal PDF | `https://academicjournals.org/journal/AJB/article-full-text-pdf/A932BBF38233` | Executable source_refs — kidney_bean: packing_porosity_mean_support, density_calibration. |
| `kidney_oswin_source` | Campos, R. C. et al. (2016). Bean grain hysteresis with induced mechanical damage. Revista Brasileira de Engenharia Agricola e Ambiental, 20(10), 930-935. | stable PDF; journal article 20(10):930-935 | `https://pdfs.semanticscholar.org/7400/4771a004cf1530362e41e9969bc9ff3551cb.pdf` | Executable source_refs — kidney_bean: oswin. |
| `kidney_drying` | Doymaz, I. (2016). Hot-Air Drying and Rehydration Characteristics of Red Kidney Bean Seeds. Chemical Engineering Communications, 203. | doi:10.1080/00986445.2015.1056299 | `https://doi.org/10.1080/00986445.2015.1056299` | Executable source_refs — kidney_bean: two_compartment_kinetics. |
| `manitoba_field_beans` | Province of Manitoba, Agriculture. Field Beans (official crop-management guidance; accessed 2026-08-09). | Government of Manitoba field-beans guidance | `https://www.gov.mb.ca/agriculture/crops/crop-management/print%2Cfield-beans.html` | Executable source_refs — kidney_bean: target_moisture. |
| `pea_physical` | Yalcin, I., Ozarslan, C. & Akbas, T. (2007). Physical properties of pea (Pisum sativum) seed. Journal of Food Engineering, 79, 731-735. | Journal of Food Engineering 79:731-735 | `https://doi.org/10.1016/j.jfoodeng.2006.02.039` | Executable source_refs — field_pea: packing_porosity_mean_support, density_calibration. |
| `pea_sorption` | Garg, M. K. & Chandra, P. (2003). Sorption Characteristics of Pea Seeds. Journal of Agricultural Engineering, 40(4). | ICAR journal record | `https://epubs.icar.org.in/index.php/JAE/article/view/14171` | Supplied supporting source; no current executable record directly cites this key. |
| `pea_drying` | Ganesh, C. V. & Sokhansanj, S. (1997). High temperature mechanical drying of field peas (Pisum sativum L.). University of Saskatchewan proceedings/thesis record. | University of Saskatchewan repository | `https://harvest.usask.ca/bitstreams/b28e4b4a-4bbf-4e6b-8f49-d3890bd8a3a8/download` | Executable source_refs — field_pea: initial_moisture, two_compartment_kinetics. |
| `canola_thermal` | Yu, D., Shrestha, B. L. & Baik, O.-D. (2015). Thermal conductivity, specific heat, thermal diffusivity, and bulk density of canola seeds. Journal of Food Engineering, 165, 156-165. | doi:10.1016/j.jfoodeng.2015.05.012 | `https://doi.org/10.1016/j.jfoodeng.2015.05.012` | Executable source_refs — rapeseed: packing_porosity_mean_support, density_calibration, thermal_properties. |
| `canola_airflow` | Jayas, D. S., Sokhansanj, S., Moysey, E. B. & Barber, E. M. (1987). Airflow Resistance of Canola (Rapeseed). Transactions of the ASAE, 30(5), 1484-1488. | doi:10.13031/2013.30590 | `https://elibrary.asabe.org/azdez.asp?AID=30590&CID=t1987&JID=3&T=2&i=5&redirType=&v=30` | Executable source_refs — rapeseed: permeability. |
| `canola_oswin` | Gazor, H. R. (2010). Moisture Isotherms and Heat of Desorption of Canola. Agricultural Engineering International: CIGR Journal, manuscript 1440. | CIGR manuscript 1440 | `https://cigrjournal.org/index.php/Ejounral/article/download/1440/1296/0` | Executable source_refs — rapeseed: oswin. |
| `canola_drying` | Costa, L. M. et al. (2020). Drying kinetics of Hyola 430 hybrid canola (Brassica napus L.) seeds. Australian Journal of Crop Science, 14(10), 1623-1629. | AJCS 14(10):1623-1629 | `https://www.cropj.com/costa_14_10_2020_1623_1629.pdf` | Executable source_refs — rapeseed: initial_moisture, two_compartment_kinetics. |
| `canola_drying_gazor` | Gazor, H. R. et al. (2010). Modelling the drying kinetics of canola in a fluidised bed dryer. Czech Journal of Food Sciences, 28(6), 531-537. | Czech J. Food Sci. 28(6):531-537 | `https://www.agriculturejournals.cz/artkey/cjf-201006-0009_modelling-the-drying-kinetics-of-canola-in-fluidised-bed-dryer.php` | Supplied supporting source; no current executable record directly cites this key. |
| `swiss_oil` | swiss granum (2026). Uebernahmebedingungen Oelsaaten, Ernte 2026. | 2026-03-19 edition | `local:upload/2026-03-19_Uebernahmebedingungen_Oelsaaten_2026_D(3).pdf` | Executable source_refs — rapeseed: target_moisture. |
| `sunflower_sorption_drying_munder_2019` | Munder, S., Argyropoulos, D. & Müller, J. (2019). Acquisition of Sorption and Drying Data with Embedded Devices: Improving Standard Models for High Oleic Sunflower Seeds by Continuous Measurements in Dynamic Systems. Agriculture, 9(1), 1. | doi:10.3390/agriculture9010001 | `https://doi.org/10.3390/agriculture9010001` | Executable source_refs — sunflower_seed: initial_moisture, oswin, two_compartment_kinetics. |
| `sunflower_density_isik_izli_2007` | Isik, E. & Izli, N. (2007). Physical Properties of Sunflower Seeds (Helianthus annuus L.). International Journal of Agricultural Research, 2, 677-686. | doi:10.3923/ijar.2007.677.686 | `https://scialert.net/fulltext/?doi=ijar.2007.677.686` | Executable source_refs — sunflower_seed: packing_porosity_mean_support, density_calibration. |
| `sunflower_physical_gupta_das_1997` | Gupta, R. K. & Das, S. K. (1997). Physical Properties of Sunflower Seeds. Journal of Agricultural Engineering Research, 66(1), 1-8. | doi:10.1006/jaer.1996.0111 | `https://doi.org/10.1006/jaer.1996.0111` | Supplied supporting source; no current executable record directly cites this key. |
| `sunflower_thermal_ince_2008` | Ince, R., Güzel, E. & Ince, A. (2008). Thermal Properties of Some Oily Seeds. Journal of Agricultural Machinery Science, 4(4), 399-405. | Journal of Agricultural Machinery Science 4(4):399-405 | `https://dergipark.org.tr/tr/download/article-file/119120` | Executable source_refs — sunflower_seed: thermal_properties. |
| `sunflower_airflow_canada_1990` | Agriculture Canada (1990). Handling Agricultural Materials: Storage and Conditioning of Grain and Forage. Agriculture Canada Publication 1855/E. | Agriculture Canada Publication 1855/E | `https://publications.gc.ca/collections/collection_2014/aac-aafc/agrhist/A15-1855-1990-eng.pdf` | Executable source_refs — sunflower_seed: permeability. |
| `sunflower_class_munder_2017` | Munder, S., Argyropoulos, D. & Müller, J. (2017). Class-based physical properties of air-classified sunflower seeds and kernels. Biosystems Engineering, 164, 124-134. | doi:10.1016/j.biosystemseng.2017.10.005 | `https://doi.org/10.1016/j.biosystemseng.2017.10.005` | Supplied supporting source; no current executable record directly cites this key. |
| `swiss_oil_2026` | swiss granum (2026). Übernahmebedingungen Ölsaaten Ernte 2026, Ausgabe 19. März 2026. | 2026-03-19 edition | `local:upload/2026-03-19_Uebernahmebedingungen_Oelsaaten_2026_D(3).pdf` | Executable source_refs — sunflower_seed: target_moisture. |
| `pino_airflow_project` | Albertin, R. M. (2026). PINO Airflow Porous Media. Vertiefungsprojekt, OST. | SHA-256 5fcae206aff935195eea0f8b149747c1a3e83fef16899d8533c0f89b3fbd954e | `local:upload/Albertin_2026_PINO_Airflow_PorousMedia(20260809-103733).pdf` | Executable source_refs — shared: grid_provenance, eps_min_global, eps_max_global; chickpea: initial_moisture; field_pea: initial_moisture; kidney_bean: initial_moisture; lentil: initial_moisture; rapeseed: initial_moisture; operation: pressure_bc, kappa_cv, bed, permeability, porosity, initial_moisture. |
| `transient_comsol_model_report` | COMSOL Multiphysics 6.4 model report: transient_drying_template, generated 2026-08-09. | SHA-256 463ecea45bad7c221f3c85ef53c92dbf77a603dc4b2fda2d54bf8e852d3e96a5 | `local:upload/transient_drying_template(3).docx` | Executable source_refs — shared: T_flow_ref, p_ref, p_out, omega_min, omega_max, phi_operational_min, phi_operational_max, phi_clip_min, phi_clip_max, cp_w, h_fg, D_v_air, M_v, U_wall, f_wet_dm_max, grid_provenance, time_provenance, physical_formulas_provenance; operation: omega_in_base, omega_in_amp. |
| `vm2_project` | Albertin, R. M. (2026). VM2 Vertiefungsprojekt. OST. | local project report | `local:project_sources/02-VM2_Vertiefungsprojekt_Rino_Moreno_Albertin-3-.pdf` | Executable source_refs — operation: schedule. |
| `vp2_decision_contract` | VP2 Parameter Decisions, schema 1.1.0 (2026-08-09). | sha256:774ce0e39bf989ad77b5fe80e37c364f46ff83b3c6be1bd7410ea4c72d7269f5 | `artifact:VP2_Parameter_Decisions.yaml` | Executable source_refs — shared: T_in_min, phi_operational_min, phi_operational_max, schedule_interpolation, grid_provenance, time_provenance, physical_formulas_provenance. |
<!-- source-catalogue:end -->
