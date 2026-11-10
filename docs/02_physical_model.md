# Physical model and parameters

What each Generation case simulates, where its values come from and how cases are sampled. The YAML under `configs/01_generation/physical_model/` owns the values, supports and units, and the loader code owns the formula contract (see [Parameters and evidence](#parameters-and-evidence)); this guide explains their meaning.

[Documentation index](README.md)

## Domain and resolution

Each case is a two-dimensional porous bed of 1.2 m × 0.75 m without resolved grains. Fields are exported on a boundary-inclusive 401 × 251 (X × Y) grid. The transient horizon is 168 physical hours with hourly output; the solver uses adaptive internal steps.

<details>
<summary>Where grid and horizon come from</summary>

Grid and horizon are physical model inputs from `configs/01_generation/physical_model/common.yaml`. The transient COMSOL call receives grid, domain size and horizon as run parameters; the steady call passes none, so the steady template must produce the configured grid itself, and conversion admits only an export that covers the configured axes exactly. A Dataset plan or Source campaign can select a shorter physical horizon (`--duration-hours H`; omitted: the `time` block of `common.yaml`, `time.stop`); the choice becomes part of the case identity. Solver timeouts and Slurm walltimes are execution limits and leave the physical horizon unchanged.

</details>

## Model

The two profiles share one physical description and are solved in sequence: the steady profile gives the airflow, the transient profile adds vapour transport, heat transfer and grain moisture on that fixed airflow.

### Airflow

The steady profile solves Darcy–Brinkman airflow with heterogeneous permeability `kappa(x)` (tensor $\mathbf K$), porosity $\varepsilon(x)$ and an inlet pressure that varies along x on the inlet edge y = 0 (`p_in_bc`; outlet gauge zero at y = $L_y$). The result $(p, u, v)$ is the steady-flow learning target and the fixed airflow of the transient profile. With viscosity $\mu$ and velocity $\mathbf u$, the momentum residual (as in COMSOL's Brinkman interface for incompressible flow, `src/domain/physics/domain_physics_brinkman.py:25-26`) and the mass balance are

$$\mathbf R=-\nabla p+\nabla\!\cdot\boldsymbol\tau-\mu\,\mathbf K^{-1}\mathbf u=0,\qquad \boldsymbol\tau=\tfrac{\mu}{\varepsilon}\Big(\nabla\mathbf u+\nabla\mathbf u^{\mathsf T}-\tfrac23(\nabla\!\cdot\mathbf u)\,\mathbf I\Big),\qquad \nabla\!\cdot\mathbf u=0 .$$

The sampled permeability scale $\bar\kappa$ (`kappa_mean`) is coupled to packing porosity through a material-calibrated Kozeny–Carman reference trend. With the material reference porosity $\varepsilon_\mathrm{bed,ref}$ (`eps_bed_ref`) and nominal permeability $\kappa_\mathrm{nom}$:

$$g(\varepsilon)=\frac{\varepsilon^3}{(1-\varepsilon)^2},\quad A_\mathrm{KC}=\frac{\kappa_\mathrm{nom}}{g(\varepsilon_\mathrm{bed,ref})},\quad \varepsilon_\mathrm{KC}=g^{-1}\!\Big(\frac{\bar\kappa}{A_\mathrm{KC}}\Big),\quad \varepsilon_\mathrm{case}=\varepsilon_\mathrm{KC}+\sigma z .$$

Here $z\sim\mathcal N(0,1)$ truncated to $(-3,3)$ and $\sigma=\tfrac13\,\mathrm{dist}(\varepsilon_\mathrm{KC},\partial S)$, with $S$ the active packing support and $\partial S$ its nearer bound, so the case level stays inside that support.

<details>
<summary>Porosity field, clipping and redraw rules</summary>

The scatter $\sigma z$ stands for unresolved morphology. A zero-mean texture with RMS amplitude `porosity.texture_amp`, built from the background of the bed structure field (optionally smoothed by `porosity.smooth_len_rel`), is added and clipped pointwise to `eps_min_global` and `eps_max_global` (0.2 and 0.8 in `common.yaml`); the scalar permeability uses that background plus local perturbations, the anisotropy magnitude the background alone. The case records the clipped fraction (`eps_bed_clipped_fraction`). A realization is redrawn when its mean porosity leaves the natural support (natural cases) or crosses back over the natural bound it lies beyond (`kappa_mean`-OOD cases; [Generation](03_generation.md)). The Kozeny–Carman relation is a global material trend: `kappa(x)` and `epsilon(x)` are spatially correlated through the shared background but not tied pointwise.

</details>

### Anisotropy

The local principal-permeability ratio $r$ follows the normalized background magnitude $\lvert b\rvert\le1$, the anisotropy strength $s$ (`permeability.anisotropy.strength`), the maximal ratio $r_\mathrm{max}$ (`permeability.anisotropy.max_ratio`) and the exponent $e$ (`permeability.anisotropy.exponent`); the principal permeabilities share the geometric mean $\kappa$, so $k_1/k_2=r$:

$$r=1+s\,(r_\mathrm{max}-1)\,\lvert b\rvert^{e},\qquad \kappa_1=\kappa\sqrt r,\qquad \kappa_2=\kappa/\sqrt r .$$

Source: `registry.yaml` (`permeability.anisotropy.*`, L150–157). The range of `max_ratio` (ID 1.2–2.2, nominal 1.6; OOD 2.6–3.2) and the evidence are in [Scientific decisions](08_scientific_decisions.md).

### Grain moisture

The transient state is temperature $T$ (K), relative gas humidity $\varphi$ (–), surface moisture $w_\mathrm{s}$ and internal moisture $w_\mathrm{i}$ (both kg/m³). Grain water uses two exchanging compartments with surface fraction $f_\mathrm{s}$ (`f_surf`), interior exchange rate $r_\mathrm{int}$ and surface rate $r_\mathrm{surf}$ (`generation_cases_config.py:78-101`):

$$f_\mathrm{s}\,\dot w_\mathrm{s}=j_\mathrm{int}-\dot m_\mathrm{evap},\qquad (1-f_\mathrm{s})\,\dot w_\mathrm{i}=-j_\mathrm{int},\qquad w_\mathrm{gr}=f_\mathrm{s}w_\mathrm{s}+(1-f_\mathrm{s})w_\mathrm{i},$$

$$j_\mathrm{int}=(1-f_\mathrm{s})\,r_\mathrm{int}\,(w_\mathrm{i}-w_\mathrm{s}),\qquad \dot m_\mathrm{evap}=f_\mathrm{s}\,r_\mathrm{surf}\max(w_\mathrm{s}-w_\mathrm{eq},0),\qquad Q_\mathrm{evap}=-h_\mathrm{fg}\,\dot m_\mathrm{evap}.$$

The Oswin record of each material (`A_osw`, `B_osw`, `C_osw` as $A$, $B$, $C$) supplies the equilibrium dry-basis moisture, with $\varphi_\mathrm{eff}$ the humidity clipped to $[10^{-6}, 0.999]$ and $\rho_\mathrm{bu,dry}$ the dry bulk density:

$$X_\mathrm{eq,db}=0.01\,[A+B\,(T-273.15\,\mathrm K)]\Big(\frac{\varphi_\mathrm{eff}}{1-\varphi_\mathrm{eff}}\Big)^{C},\qquad w_\mathrm{eq}=\rho_\mathrm{bu,dry}\,X_\mathrm{eq,db}.$$

The initial equilibrium RH is the algebraic inverse, evaluated from `X_0_db_field` ($X_{0,\mathrm{db}}$) and `T_init` (`osw_ratio_0`, `phi_init` in the loader formulas):

$$\varphi_0=\frac{\rho}{1+\rho},\qquad \rho=\Big(\frac{100\,X_{0,\mathrm{db}}}{A+B\,(T_\mathrm{init}-273.15)}\Big)^{1/C}.$$

Initial moisture is stored on a dry basis:

$$X_\mathrm{wb}=\frac{X_\mathrm{db}}{1+X_\mathrm{db}},\qquad X_\mathrm{db}=\frac{X_\mathrm{wb}}{1-X_\mathrm{wb}} .$$

Realized packing adjusts the dry bulk density.

### Inlet

The inlet schedule is a semicolon-separated CSV with two header lines:

```text
t;T_in_bc;omega_in_bc
h;K;kg/kg
```

Relative humidity is derived after interpolating temperature and humidity ratio $\omega_\mathrm{in}$ (`generation_cases_schedule.py:43`, `:1940`), with reference pressure $p_\mathrm{ref}$ and saturation pressure $p_\mathrm{sat}$:

$$\varphi_\mathrm{in}=\frac{p_\mathrm{ref}\,\omega_\mathrm{in}}{(0.621945+\omega_\mathrm{in})\,p_\mathrm{sat}(T_\mathrm{in})} .$$

The startup phase begins at `T_init`. Its inlet humidity target is the strictest of the drying margin $\Delta\varphi_\mathrm{dry}$ (`initial_equilibrium_rh_dry_margin`, 0.05) below the initial bed equilibrium and the startup ceiling $\varphi_\mathrm{max,start}$ (`max_relative_humidity`, 0.90) (`fixed_bed.yaml:168-172`):

$$\varphi_\mathrm{in,start}=\min\Big(\min_x\varphi_\mathrm{eq,init}-\Delta\varphi_\mathrm{dry},\ \varphi_\mathrm{max,start}\Big).$$

<details>
<summary>Schedule checks, startup contracts, drying target and stop criterion</summary>

Generation checks the derived relative humidity continuously, including interior extrema. An infeasible stochastic schedule is resampled as a whole. Temperature and humidity ratio then ramp to the canonical schedule at the configured rejoin time. The startup humidity ratio follows psychrometrically. Startup bounds, canonical operating bounds, the COMSOL numerical clip and the physical saturation limit are separate contracts.

Each material has one operational drying target on a wet basis:

```yaml
target_moisture:
  target_moisture_wb: 0.12   # 12.0 % wb
```

The target is the practical safe-storage moisture, normally rounded to a whole wet-basis percentage point; the exact literature or isotherm-derived value remains in provenance. It supplies the COMSOL scalar `X_target_wb`. The stop criterion is spatial: the dry-mass fraction above the target, compared with the configured tolerance,

$$f_\mathrm{wet,dm}=\frac{\int\rho_\mathrm{bu,dry}\,\mathbb 1[X_\mathrm{wb}>X_\mathrm{target,wb}]}{\int\rho_\mathrm{bu,dry}},$$

named `f_wet_dm` in the loader formulas (`generation_cases_config.py`).

A run that reaches the end of its horizon has therefore not necessarily reached the target.

</details>

## Parameters and evidence

The YAML under `configs/01_generation/physical_model/` owns every value; each record names its evidence. Use `./scripts/generation run CONFIG --dry-run` to see the effective values of a campaign (`--dry-run` is off by default: without it the campaign starts).

<details>
<summary>Owner of each value and evidence classes</summary>

| Concern | Owner |
| --- | --- |
| Grid, output times, fixed physical values, tolerances | `configs/01_generation/physical_model/common.yaml` |
| Parameter definitions, units, transforms, OOD groups | `configs/01_generation/physical_model/registry.yaml` |
| Material records and their provenance | `configs/01_generation/physical_model/materials/*.yaml` |
| Operation and inlet-schedule distributions (registry block `operation`) | `configs/01_generation/physical_model/fixed_bed.yaml` |
| Bed, permeability, porosity, inlet-pressure and initial-moisture field statistics | `configs/01_generation/physical_model/bed_fields.yaml` |
| COMSOL template and exported fields per task | `configs/01_generation/physical_model/profiles/*.yaml` |
| Bibliography | `configs/01_generation/physical_model/sources.yaml` |
| Formula contract, input-file adapters, schedule rules, steady-flow conditioning, fixed time and storage fields, kinetics OOD records | Loader code (`src/generation/cases/generation_cases_config.py`, `src/generation/contracts/generation_contracts_materials.py`); copied into every scientific record |
| Design counts per material and design kind, Generation seed base | Dataset plan or Source campaign |

Configured values are modelling and sampling decisions. Each record names its evidence class (measured, fitted, transferred, estimated, derived or synthetic), its `source_refs` and any conversion, fit or applicability limit. A material-level citation is evidence only for the values that reference it. Shared spatial-field statistics are synthetic generator assumptions unless their provenance states otherwise. Material records carry no learning roles; the Dataset plan assigns those.

</details>

## Sampling

- **Natural support** is the in-distribution region of a material and profile.
- **Parameter OOD** moves one eligible scalar or one complete coupled record (density, sorption, kinetics, schedule) outside natural support.
- **Material OOD** holds out whole material families with natural parameter support; a Dataset assigns it. Material identity is provenance, not an implicit one-hot model input.
- **Designs:** every material has one fixed scrambled Sobol design per design kind (train, validation, test, parameter OOD); a request for N points takes points 0 to N − 1, so smaller sets are subsets of larger ones. Counts are powers of two.

<details>
<summary>Domain profile and OOD groups</summary>

The domain profile is derived from the designs a Source requests: `reference_range` (readable spelling `ref-range`) when every point has natural support, `reference_and_parameter_extension` (`ref-param-ext`) when any material also has a parameter-OOD design. Both tasks admit parameter OOD over their profile-active OOD groups (`ood_group` in the registry).

</details>

<details>
<summary>Design coordinates</summary>

For transient drying the resolved registry has four sampling blocks:

| Block | Normalized design dimensions |
| --- | ---: |
| Airflow | 27 |
| Initial moisture | 8 |
| Operation and schedule | 12 |
| Material properties | 6 |
| Total | 53 |

Each interval or integer entry contributes one coordinate; a simplex with k components contributes k−1. Fixed, derived and coupled-record entries contribute none. The resolved profile is authoritative when registry or profile change. Steady flow samples the airflow block alone (27 coordinates).

</details>

<details>
<summary>Sobol mechanics, prefixes and case identity</summary>

Every design is a scrambled Sobol sequence over all coordinates of all blocks of the task profile (recipe `scrambled_sobol_single_sequence`): one sequence without skip, leap or block permutation, constructed as `scipy.stats.qmc.Sobol(d=<dimensions>, scramble=True, rng=<design seed>)` and read with `.random(N)` ([SciPy Sobol](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html); `rng` is SciPy's current seeding argument). A test pins the first points of a fixed design, so a change of SciPy's scrambling fails loudly.

- **Seed:** the design seed derives from the task's Generation seed base (`seed_base`), the material and the design kind only; count, Source name, campaign, Dataset plan and Dataset roles are excluded. Point i is the i-th point, and every per-case random stream (permeability and porosity fields, inlet schedules) derives from the design seed and the point index.
- **Nesting:** the first ten points of a design are the same cases in every Dataset that requests at least ten. Train prefixes form a nested learning curve; the smoke plan holds the first points of the production designs, and the reduced set is a subset of the production set. The test design of kidney_bean is material OOD in the reduced set and ID-test in the production set.
- **Power-of-two counts:** the balance properties of scrambled Sobol points hold for power-of-two prefixes (1, 2, 4, …), so the Dataset plan admits only such counts. A failed point is replaced by the next unused point of its design; a design with failures then holds points 0 to N − 1 without the failed ones plus later points and keeps the balance only approximately ([Datasets](04_datasets.md#fixed-designs-and-case-roles), [Replacements](03_generation.md#replacements)).
- **Records:** each batch records its design seed, the dimension count, its block columns and a digest of its points. The campaign's `seed_key` frozen in `campaign_plan.json` (a digest of the task, the design sizes per material and design kind, the domain profile, the design recipe with its seed base and the campaign purpose) keys candidate reservations and the campaign identity. The name and the storage layout are not inputs, so renaming a Source keeps its random fields.
- **Case IDs** are material-local (`case_0001`, …) and follow the batches of a campaign; the design point of a case is recorded in its design provenance (`design_index`).

</details>
