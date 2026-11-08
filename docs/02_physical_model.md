# Physical model and parameters

This guide describes what each Generation case simulates, where its values come from and how cases are sampled. The YAML under `configs/01_generation/` owns all values, supports, units and equations; this guide explains their meaning.

[Documentation index](README.md)

## Domain and resolution

Each case is a two-dimensional porous bed of 1.2 m × 0.75 m without resolved grains. Fields are exported on a boundary-inclusive 401 × 251 (X × Y) grid. The transient horizon is 168 physical hours with hourly output; the solver uses adaptive internal steps.

Grid and horizon are physical model inputs from `configs/01_generation/physical_model/common.yaml`. The transient COMSOL call receives grid, domain size and horizon as run parameters; the steady call passes none, so the steady template must produce the configured grid itself, and conversion admits only an export that covers the configured axes exactly. A Dataset plan or Source campaign can select a shorter physical horizon (`--duration-hours`); the choice becomes part of the case identity. Solver timeouts and Slurm walltimes are execution limits and leave the physical horizon unchanged.

## Model

The two profiles share one physical description and are solved in sequence.

<details>
<summary><strong>Airflow</strong></summary>

The steady profile solves Darcy–Brinkman airflow with heterogeneous permeability `kappa(x)`, porosity `epsilon(x)` and an inlet pressure that varies along x on the inlet edge y = 0 (`p_in_bc`; outlet gauge zero at y = L_y). The result `(p, u, v)` is the steady-flow learning target and the fixed airflow of the transient profile.

The sampled permeability scale `kappa_mean` is coupled to packing porosity through a material-calibrated Kozeny–Carman reference trend:

```text
g(epsilon) = epsilon^3 / (1 - epsilon)^2
A_KC_reference = kappa_nominal / g(eps_bed_ref)
eps_kc_trend = g^-1(kappa_mean / A_KC_reference)
```

`kappa_mean` is the sampled coordinate and `eps_bed_ref` the material reference porosity. Each case shifts the trend by one packing scatter `sigma z` for unresolved morphology: `z` is standard normal truncated to (−3, 3) and `sigma` is one third of the distance from `eps_kc_trend` to the nearer bound of the active packing support, so the case level stays inside that support. A zero-mean texture with RMS amplitude `porosity.texture_amp`, built from the background of the bed structure field (optionally smoothed by `porosity.smooth_len_rel`), is added and clipped pointwise to `eps_min_global` and `eps_max_global` (0.2 and 0.8 in `common.yaml`); the scalar permeability uses that background plus local perturbations, the anisotropy magnitude the background alone. The case records the clipped fraction (`eps_bed_clipped_fraction`), and a realization is redrawn when its mean porosity leaves the natural support (natural cases) or crosses back over the natural bound it lies beyond (`kappa_mean`-OOD cases; [Generation reference](06_generation_reference.md)). The Kozeny–Carman relation is a global material trend: `kappa(x)` and `epsilon(x)` are spatially correlated through the shared background but not tied pointwise.

</details>

<details>
<summary><strong>Transient state, moisture and heat</strong></summary>

The transient profile adds vapour transport, heat transfer and grain-moisture dynamics on the fixed airflow. Its state is temperature `T` (K), relative gas humidity `phi` (–), surface moisture `w_surf` and internal moisture `w_int` (both kg/m³).

Grain water uses two exchanging compartments:

```text
f_surf d(w_surf)/dt       = j_int - m_evap
(1 - f_surf) d(w_int)/dt  = -j_int
w_gr = f_surf w_surf + (1 - f_surf) w_int
j_int = (1 - f_surf) r_int (w_int - w_surf)
m_evap = f_surf r_surf max(w_surf - w_eq, 0)
```

The Oswin record of each material supplies equilibrium dry-basis moisture:

```text
X_eq_db = 0.01 [A_osw + B_osw (T - 273.15 K)] [phi_eff / (1 - phi_eff)]^C_osw
w_eq = rho_bu_dry X_eq_db
```

Evaporation contributes the latent heat sink `Q_evap = -h_fg m_evap`. Realized packing adjusts dry bulk density. Initial equilibrium RH is the algebraic inverse evaluated from `X_0_db_field` and `T_init`.

Initial moisture is stored on a dry basis:

```text
X_wb = X_db / (1 + X_db)
X_db = X_wb / (1 - X_wb)
```

</details>

<details>
<summary><strong>Inlet schedule and startup</strong></summary>

The inlet schedule has three columns:

```text
t;T_in_bc;omega_in_bc
h;K;kg/kg
```

Relative humidity is derived after interpolating temperature and humidity ratio:

```text
phi_in_bc = [p_ref omega_in_bc / (0.621945 + omega_in_bc)] / p_sat(T_in_bc)
```

Generation checks the derived relative humidity continuously, including interior extrema. An infeasible stochastic schedule is resampled as a whole.

The startup phase begins at `T_init`. Its inlet humidity target is the strictest of the drying margin below the initial bed equilibrium and the startup ceiling:

```text
phi_in_start = min(
    min(phi_eq_init) - initial_equilibrium_rh_dry_margin,
    max_relative_humidity
)
```

Temperature and humidity ratio then ramp to the canonical schedule at the configured rejoin time. The startup humidity ratio follows psychrometrically. Startup bounds, canonical operating bounds, the COMSOL numerical clip and the physical saturation limit are separate contracts.

</details>

<details>
<summary><strong>Drying target and stop criterion</strong></summary>

Each material has one operational drying target on a wet basis:

```yaml
target_moisture:
  target_moisture_wb: 0.12   # 12.0 % wb
```

The target is the practical safe-storage moisture, normally rounded to a whole wet-basis percentage point; the exact literature or isotherm-derived value remains in provenance. It supplies the COMSOL scalar `X_target_wb`. The stop criterion is spatial: `f_wet_dm`, the dry-mass fraction above `X_target_wb`, is compared with the configured tolerance. A run that reaches the end of its horizon has therefore not necessarily reached the target.

</details>

## Parameters and evidence

| Concern | Owner |
| --- | --- |
| Grid, time, fixed values, equations, tolerances | `configs/01_generation/physical_model/common.yaml` |
| Parameter definitions, units, transforms, OOD groups | `configs/01_generation/physical_model/registry.yaml` |
| Material records and their provenance | `configs/01_generation/physical_model/materials/*.yaml` |
| Operation and inlet-schedule constraints | `configs/01_generation/physical_model/fixed_bed.yaml` |
| Active blocks, COMSOL template and exported fields per task | `configs/01_generation/physical_model/profiles/*.yaml` |
| Bibliography | `configs/01_generation/physical_model/sources.yaml` |
| Material counts and Generation seed | Dataset plan or Source campaign |

Configured values are modelling and sampling decisions. Each record names its evidence class (measured, fitted, transferred, estimated, derived or synthetic), its `source_refs` and any conversion, fit or applicability limit. A material-level citation is evidence only for the values that reference it. Shared spatial-field statistics are synthetic generator assumptions unless their provenance states otherwise. Material records carry no learning roles; the Dataset plan assigns those.

Use `./scripts/generation run CONFIG --dry-run` to see the effective values of a campaign.

## Sampling

<details>
<summary><strong>Natural support and out-of-distribution branches</strong></summary>

- **Natural support** is the in-distribution region of a material and profile.
- **Parameter OOD** moves one eligible scalar or one complete coupled record (density, sorption, kinetics, schedule) outside natural support.
- **Family OOD** holds out whole material families with natural parameter support; a Dataset assigns it.
- Material identity is provenance, not an implicit one-hot model input.

`domain_profile: reference_range` labels a Source with natural support for the selected materials (readable spelling `ref-range`). `reference_and_parameter_extension` labels a Source that also declares the parameter-OOD branch.

</details>

<details>
<summary><strong>Design coordinates</strong></summary>

For transient drying the resolved registry has four independent sampling blocks:

| Block | Normalized design dimensions |
| --- | ---: |
| Airflow | 27 |
| Initial moisture | 8 |
| Operation and schedule | 12 |
| Material properties | 6 |
| Total | 53 |

Each interval or integer entry contributes one coordinate; a simplex with k components contributes k−1. Fixed, derived and coupled-record entries contribute none. The resolved profile is authoritative when registry or profile change.

</details>

<details>
<summary><strong>Latin hypercube designs and subsets</strong></summary>

Each block uses a Latin hypercube of the requested size N: every normalized marginal has exactly one point in each of N equal strata ([SciPy LatinHypercube](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.LatinHypercube.html)). The sampler draws 50 scrambled candidate designs and keeps the one with the largest minimum nearest-neighbour distance.

The nominal Generation seed expands into separate material, profile, regime and block streams; each block has its own design and permutation seed. Count, Source name and Dataset roles are excluded from the seed derivation. Changing N changes the whole design: an N = 10 design is not the first ten points of an N = 50 design.

Case IDs are material-local (`case_0001`, …). Each block maps a case position to a permuted design row, so the first ten cases are a subset of the full design, not a ten-point Latin hypercube. A check with seed 9200 (NumPy 2.4.6, SciPy 1.17.1) on the normalized coordinates:

| Material / N | Full design | First 10 cases at 10 strata | Independent N = 10 design |
| --- | ---: | ---: | ---: |
| Lentil / 50 | 53/53 marginals | 0/53 | 53/53 |
| Chickpea / 50 | 53/53 | 0/53 | 53/53 |
| Kidney bean / 20 | 53/53 | 0/53 | 53/53 |

Whether a subset is adequate depends on its actual coverage and the analysis; random or ranked subsets, failure filtering, splits and replacement points keep their own coverage and are not Latin hypercubes.

</details>
