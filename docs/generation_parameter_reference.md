# VP2 Generation Parameter and Symbol Reference

Audience: implementers, evidence researchers, COMSOL adapters, report authors, and sensitivity/error analysts. This is the single catalogue for generation parameters, quantities, numerical dimensions, machine-name/report-symbol mappings, and evidence requirements. YAML and source remain executable authority; no unresolved value or literature source is supplied here.

## Naming and catalogue conventions

Canonical scientific names do not encode units. Physical names are shared across Python, YAML, adapters, HDF5, datasets, evaluation, and documentation. Internal controls use the compact `bed`, `permeability`, `porosity`, `pressure_bc`, `initial_moisture`, or `schedule` namespace. Report notation remains concise and independent of machine-name length; each registry symbol maps to exactly one canonical machine name.

In the catalogue, `d` is the effective numerical coordinate contribution. Natural support is ID support; parameter OOD uses a scientifically justified disjoint support in registry-transform coordinates, while family OOD uses the held-out family's natural support. All values/selections persist in `case.json`; the table names their principal additional effect. Exact solver equations are defined once in the [manual COMSOL checklist](simulation_generation.md#manual-comsol-64-adaptation-checklist).

## Complete registry catalogue

| Canonical name | Report symbol | Unit | Category | Block | Status | d | Transform/selection | Owner | Purpose/effect | Generation/derivation | ID/OOD | Consumer | COMSOL relevance | Persistence | Neural-operator relevance | Evidence | Report |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `kappa_mean` | $\bar{\kappa}$ | `m^2` | physical sample | `airflow` | unresolved | 1 | log coordinate | `materials/<family>.yaml` | Mean scalar bed permeability used by the lognormal permeability field. | lognormal/SPD permeability map | natural; disjoint `bed` OOD | `generation_fields` | indirect: tensor fields | `case.json`; permeability fields/static HDF5 | indirectly generates steady inputs; transient ablation provenance | material unresolved | main text |
| `kappa_cv` | $c_{\kappa}$ | `1` | physical sample | `airflow` | unresolved | 1 | linear coordinate | `materials/<family>.yaml` | Coefficient of variation used to derive the lognormal permeability spread. | lognormal/SPD permeability map | natural; disjoint `bed` OOD | `generation_fields` | indirect: tensor fields | `case.json`; permeability fields/static HDF5 | indirectly generates steady inputs; transient ablation provenance | material unresolved | main text |
| `bed.structure.coarse_len_rel` | $\ell_{b,c}/L_x$ | `1` | generator control | `airflow` | unresolved | 1 | log coordinate | `materials/<family>.yaml` | Coarse bed correlation length divided by bed length. | multiscale bed field | natural; disjoint `bed` OOD | `generation_fields` | indirect: `fields.csv` | `case.json`; generated field/static HDF5 | indirectly generates steady inputs; transient ablation provenance | material unresolved | appendix / implementation |
| `bed.structure.fine_len_rel` | $\ell_{b,f}/L_x$ | `1` | generator control | `airflow` | unresolved | 1 | log coordinate | `materials/<family>.yaml` | Fine bed correlation length divided by bed length. | multiscale bed field | natural; disjoint `bed` OOD | `generation_fields` | indirect: `fields.csv` | `case.json`; generated field/static HDF5 | indirectly generates steady inputs; transient ablation provenance | material unresolved | appendix / implementation |
| `bed.structure.coarse_weight` | $\alpha_{b,c}$ | `1` | generator control | `airflow` | unresolved | 1 | logit coordinate | `materials/<family>.yaml` | Independent coarse contribution to the bed multiscale field. | multiscale bed field | natural; disjoint `bed` OOD | `generation_fields` | indirect: `fields.csv` | `case.json`; generated field/static HDF5 | indirectly generates steady inputs; transient ablation provenance | material unresolved | appendix / implementation |
| `bed.structure.fine_ani_x` | $a_{b,x}$ | `1` | generator control | `airflow` | unresolved | 1 | log coordinate | `materials/<family>.yaml` | Fine bed correlation-length multiplier along x. | multiscale bed field | natural; disjoint `bed` OOD | `generation_fields` | indirect: `fields.csv` | `case.json`; generated field/static HDF5 | indirectly generates steady inputs; transient ablation provenance | material unresolved | appendix / implementation |
| `bed.structure.fine_ani_y` | $a_{b,y}$ | `1` | generator control | `airflow` | unresolved | 1 | log coordinate | `materials/<family>.yaml` | Fine bed correlation-length multiplier along y. | multiscale bed field | natural; disjoint `bed` OOD | `generation_fields` | indirect: `fields.csv` | `case.json`; generated field/static HDF5 | indirectly generates steady inputs; transient ablation provenance | material unresolved | appendix / implementation |
| `bed.structure.cross_scale_corr` | $\rho_b$ | `1` | generator control | `airflow` | unresolved | 1 | linear coordinate | `materials/<family>.yaml` | Correlation between coarse and fine bed latent seeds. | multiscale bed field | natural; disjoint `bed` OOD | `generation_fields` | indirect: `fields.csv` | `case.json`; generated field/static HDF5 | indirectly generates steady inputs; transient ablation provenance | material unresolved | appendix / implementation |
| `bed.perturbations.amplitude` | $\eta_b$ | `1` | generator control | `airflow` | unresolved | 1 | linear coordinate | `materials/<family>.yaml` | Root-mean-square amplitude of bed-only local perturbations. | seeded local bed perturbations | natural; disjoint `bed` OOD | `generation_fields` | indirect: `fields.csv` | `case.json`; generated field/static HDF5 | indirectly generates steady inputs; transient ablation provenance | material unresolved | appendix / implementation |
| `bed.perturbations.granularity` | $g_b$ | `1` | generator control | `airflow` | unresolved | 1 | linear coordinate | `materials/<family>.yaml` | Scale selector for bed-only local perturbations. | seeded local bed perturbations | natural; disjoint `bed` OOD | `generation_fields` | indirect: `fields.csv` | `case.json`; generated field/static HDF5 | indirectly generates steady inputs; transient ablation provenance | material unresolved | appendix / implementation |
| `bed.perturbations.sign_bias` | $q_b$ | `1` | generator control | `airflow` | unresolved | 1 | linear coordinate | `materials/<family>.yaml` | Positive-sign probability for bed-only local perturbations. | seeded local bed perturbations | natural; disjoint `bed` OOD | `generation_fields` | indirect: `fields.csv` | `case.json`; generated field/static HDF5 | indirectly generates steady inputs; transient ablation provenance | material unresolved | appendix / implementation |
| `permeability.anisotropy.max_ratio` | $a_{\max}$ | `1` | generator control | `airflow` | unresolved | 1 | linear coordinate | `materials/<family>.yaml` | Base maximum ratio in the anisotropy map. | `1 + strength * (max_ratio - 1) * |z_b|^exponent` | natural; disjoint `bed` OOD | `generation_fields` | indirect: tensor fields | `case.json`; permeability fields/static HDF5 | indirectly generates steady inputs; transient ablation provenance | material unresolved | appendix / implementation |
| `permeability.anisotropy.exponent` | $\gamma_a$ | `1` | generator control | `airflow` | unresolved | 1 | linear coordinate | `materials/<family>.yaml` | Exponent mapping bed structure magnitude to anisotropy. | lognormal/SPD permeability map | natural; disjoint `bed` OOD | `generation_fields` | indirect: tensor fields | `case.json`; permeability fields/static HDF5 | indirectly generates steady inputs; transient ablation provenance | material unresolved | appendix / implementation |
| `permeability.anisotropy.strength` | $s_K$ | `1` | generator control | `airflow` | unresolved | 1 | linear coordinate | `materials/<family>.yaml` | Multiplier controlling permeability-tensor anisotropy. | lognormal/SPD permeability map | natural; disjoint `bed` OOD | `generation_fields` | indirect: tensor fields | `case.json`; permeability fields/static HDF5 | indirectly generates steady inputs; transient ablation provenance | material unresolved | appendix / implementation |
| `permeability.orientation.jitter` | $j_{\theta}$ | `1` | generator control | `airflow` | unresolved | 1 | linear coordinate | `materials/<family>.yaml` | Random perturbation amplitude for permeability orientation. | lognormal/SPD permeability map | natural; disjoint `bed` OOD | `generation_fields` | indirect: tensor fields | `case.json`; permeability fields/static HDF5 | indirectly generates steady inputs; transient ablation provenance | material unresolved | appendix / implementation |
| `permeability.orientation.smooth_len_rel` | $\ell_{\theta}/L_x$ | `1` | generator control | `airflow` | unresolved | 1 | log coordinate | `materials/<family>.yaml` | Relative smoothing length for permeability orientation. | lognormal/SPD permeability map | natural; disjoint `bed` OOD | `generation_fields` | indirect: tensor fields | `case.json`; permeability fields/static HDF5 | indirectly generates steady inputs; transient ablation provenance | material unresolved | appendix / implementation |
| `porosity.anchor_rel` | $A_K/\bar{\kappa}$ | `1` | generator control | `airflow` | unresolved | 1 | log coordinate | `materials/<family>.yaml` | Relative anchor converting mean permeability to the Kozeny–Carman factor. | `kozeny_carman_factor = anchor_rel * kappa_mean` | natural; disjoint `bed` OOD | `generation_fields` | indirect: `eps_bed` | `case.json`; porosity/static HDF5 | indirectly generates `eps_bed`; no direct scalar channel | material unresolved | appendix / implementation |
| `porosity.smooth_len_rel` | $\ell_{\varepsilon}/L_x$ | `1` | generator control | `airflow` | unresolved | 1 | linear coordinate | `materials/<family>.yaml` | Relative smoothing length for the porosity latent field. | Kozeny–Carman porosity map | natural; disjoint `bed` OOD | `generation_fields` | indirect: `eps_bed` | `case.json`; porosity/static HDF5 | indirectly generates `eps_bed`; no direct scalar channel | material unresolved | appendix / implementation |
| `porosity.texture_amp` | $\Delta\varepsilon$ | `1` | generator control | `airflow` | unresolved | 1 | linear coordinate | `materials/<family>.yaml` | Porosity texture amplitude around the calibrated reference. | Kozeny–Carman porosity map | natural; disjoint `bed` OOD | `generation_fields` | indirect: `eps_bed` | `case.json`; porosity/static HDF5 | indirectly generates `eps_bed`; no direct scalar channel | material unresolved | appendix / implementation |
| `pressure_bc.mean` | $\bar{p}_{\mathrm{in}}$ | `Pa` | physical sample | `airflow` | unresolved | 1 | linear coordinate | `operations/fixed_bed.yaml` | Mean inlet pressure boundary magnitude. | inlet-pressure profile | natural; disjoint `operation` OOD | `generation_fields` | indirect: `p_bc` | `case.json`; pressure field/static HDF5 | indirectly generates steady `p_bc`; transient ablation provenance | operation unresolved | main text |
| `pressure_bc.sin_amp` | $a_{p,\sin}$ | `1` | generator control | `airflow` | unresolved | 1 | linear coordinate | `operations/fixed_bed.yaml` | Relative sinusoidal inlet-pressure amplitude. | inlet-pressure profile | natural; disjoint `operation` OOD | `generation_fields` | indirect: `p_bc` | `case.json`; pressure field/static HDF5 | indirectly generates steady `p_bc`; transient ablation provenance | operation unresolved | appendix / implementation |
| `pressure_bc.sin_freq` | $f_{p,\sin}$ | `1` | generator control | `airflow` | unresolved | 1 | linear coordinate | `operations/fixed_bed.yaml` | Sinusoidal inlet-pressure spatial frequency. | inlet-pressure profile | natural; disjoint `operation` OOD | `generation_fields` | indirect: `p_bc` | `case.json`; pressure field/static HDF5 | indirectly generates steady `p_bc`; transient ablation provenance | operation unresolved | appendix / implementation |
| `pressure_bc.sin_phase` | $\varphi_{p,\sin}$ | `rad` | generator control | `airflow` | unresolved | 1 | phase coordinate | `operations/fixed_bed.yaml` | Sinusoidal inlet-pressure phase. | inlet-pressure profile | natural; disjoint `operation` OOD | `generation_fields` | indirect: `p_bc` | `case.json`; pressure field/static HDF5 | indirectly generates steady `p_bc`; transient ablation provenance | operation unresolved | appendix / implementation |
| `pressure_bc.gauss_count` | $n_{p,G}$ | `1` | generator control | `airflow` | unresolved | 1 | integer selection | `operations/fixed_bed.yaml` | Number of Gaussian inlet-pressure components. | inlet-pressure profile | natural; disjoint `operation` OOD | `generation_fields` | indirect: `p_bc` | `case.json`; pressure field/static HDF5 | indirectly generates steady `p_bc`; transient ablation provenance | operation unresolved | appendix / implementation |
| `pressure_bc.gauss_amp` | $a_{p,G}$ | `1` | generator control | `airflow` | unresolved | 1 | linear coordinate | `operations/fixed_bed.yaml` | Combined relative amplitude of Gaussian pressure components. | inlet-pressure profile | natural; disjoint `operation` OOD | `generation_fields` | indirect: `p_bc` | `case.json`; pressure field/static HDF5 | indirectly generates steady `p_bc`; transient ablation provenance | operation unresolved | appendix / implementation |
| `pressure_bc.gauss_width` | $\sigma_{p,G}$ | `1` | generator control | `airflow` | unresolved | 1 | log coordinate | `operations/fixed_bed.yaml` | Reference width of Gaussian pressure components. | inlet-pressure profile | natural; disjoint `operation` OOD | `generation_fields` | indirect: `p_bc` | `case.json`; pressure field/static HDF5 | indirectly generates steady `p_bc`; transient ablation provenance | operation unresolved | appendix / implementation |
| `pressure_bc.gauss_jitter` | $j_{p,G}$ | `1` | generator control | `airflow` | unresolved | 1 | linear coordinate | `operations/fixed_bed.yaml` | Relative Gaussian pressure-width jitter. | inlet-pressure profile | natural; disjoint `operation` OOD | `generation_fields` | indirect: `p_bc` | `case.json`; pressure field/static HDF5 | indirectly generates steady `p_bc`; transient ablation provenance | operation unresolved | appendix / implementation |
| `pressure_bc.linear_amp` | $a_{p,\mathrm{lin}}$ | `1` | generator control | `airflow` | unresolved | 1 | linear coordinate | `operations/fixed_bed.yaml` | Relative linear inlet-pressure trend. | inlet-pressure profile | natural; disjoint `operation` OOD | `generation_fields` | indirect: `p_bc` | `case.json`; pressure field/static HDF5 | indirectly generates steady `p_bc`; transient ablation provenance | operation unresolved | appendix / implementation |
| `initial_moisture.mean_db` | $\bar{X}_{0,db}$ | `kg/kg` | generator control | `initial_moisture` | unresolved | 1 | linear coordinate | `materials/<family>.yaml` | Mean initial dry-basis moisture of the generated field. | bounded multiscale dry-basis field | natural; disjoint `initial_moisture` OOD | `generation_fields` | indirect: `X_0_db_field` | `case.json`; field diagnostics/static HDF5 | transient initial-state provenance/optional ablation; not baseline input | material unresolved | appendix / implementation |
| `initial_moisture.amplitude_db` | $\Delta X_{0,db}$ | `kg/kg` | generator control | `initial_moisture` | unresolved | 1 | linear coordinate | `materials/<family>.yaml` | Maximum dry-basis deviation from the configured initial mean. | bounded multiscale dry-basis field | natural; disjoint `initial_moisture` OOD | `generation_fields` | indirect: `X_0_db_field` | `case.json`; field diagnostics/static HDF5 | transient initial-state provenance/optional ablation; not baseline input | material unresolved | appendix / implementation |
| `initial_moisture.structure.coarse_len_rel` | $\ell_{X,c}/L_x$ | `1` | generator control | `initial_moisture` | unresolved | 1 | log coordinate | `materials/<family>.yaml` | Coarse initial-moisture correlation length divided by bed length. | bounded multiscale dry-basis field | natural; disjoint `initial_moisture` OOD | `generation_fields` | indirect: `X_0_db_field` | `case.json`; field diagnostics/static HDF5 | transient initial-state provenance/optional ablation; not baseline input | material unresolved | appendix / implementation |
| `initial_moisture.structure.fine_len_rel` | $\ell_{X,f}/L_x$ | `1` | generator control | `initial_moisture` | unresolved | 1 | log coordinate | `materials/<family>.yaml` | Fine initial-moisture correlation length divided by bed length. | bounded multiscale dry-basis field | natural; disjoint `initial_moisture` OOD | `generation_fields` | indirect: `X_0_db_field` | `case.json`; field diagnostics/static HDF5 | transient initial-state provenance/optional ablation; not baseline input | material unresolved | appendix / implementation |
| `initial_moisture.structure.coarse_weight` | $\alpha_{X,c}$ | `1` | generator control | `initial_moisture` | unresolved | 1 | logit coordinate | `materials/<family>.yaml` | Independent coarse contribution to the initial-moisture multiscale field. | bounded multiscale dry-basis field | natural; disjoint `initial_moisture` OOD | `generation_fields` | indirect: `X_0_db_field` | `case.json`; field diagnostics/static HDF5 | transient initial-state provenance/optional ablation; not baseline input | material unresolved | appendix / implementation |
| `initial_moisture.structure.fine_ani_x` | $a_{X,x}$ | `1` | generator control | `initial_moisture` | unresolved | 1 | log coordinate | `materials/<family>.yaml` | Fine initial-moisture correlation-length multiplier along x. | bounded multiscale dry-basis field | natural; disjoint `initial_moisture` OOD | `generation_fields` | indirect: `X_0_db_field` | `case.json`; field diagnostics/static HDF5 | transient initial-state provenance/optional ablation; not baseline input | material unresolved | appendix / implementation |
| `initial_moisture.structure.fine_ani_y` | $a_{X,y}$ | `1` | generator control | `initial_moisture` | unresolved | 1 | log coordinate | `materials/<family>.yaml` | Fine initial-moisture correlation-length multiplier along y. | bounded multiscale dry-basis field | natural; disjoint `initial_moisture` OOD | `generation_fields` | indirect: `X_0_db_field` | `case.json`; field diagnostics/static HDF5 | transient initial-state provenance/optional ablation; not baseline input | material unresolved | appendix / implementation |
| `initial_moisture.structure.cross_scale_corr` | $\rho_X$ | `1` | generator control | `initial_moisture` | unresolved | 1 | linear coordinate | `materials/<family>.yaml` | Correlation between coarse and fine initial-moisture latent seeds. | bounded multiscale dry-basis field | natural; disjoint `initial_moisture` OOD | `generation_fields` | indirect: `X_0_db_field` | `case.json`; field diagnostics/static HDF5 | transient initial-state provenance/optional ablation; not baseline input | material unresolved | appendix / implementation |
| `T_in_base` | $T_{\mathrm{in},0}$ | `K` | physical sample | `operation` | unresolved | 1 | linear coordinate | `operations/fixed_bed.yaml` | Baseline inlet-air temperature. | compositional inlet schedule | natural; disjoint `operation` OOD | `generation_schedule` | schedule/scalar adapter | `case.json`; schedule/scalar HDF5 | indirectly generates transient boundary conditioning; not state | operation unresolved | main text |
| `T_in_amp` | $\Delta T_{\mathrm{in}}$ | `K` | physical sample | `operation` | unresolved | 1 | linear coordinate | `operations/fixed_bed.yaml` | Inlet-temperature schedule amplitude. | compositional inlet schedule | natural; disjoint `operation` OOD | `generation_schedule` | schedule/scalar adapter | `case.json`; schedule/scalar HDF5 | indirectly generates transient boundary conditioning; not state | operation unresolved | main text |
| `omega_in_base` | $\omega_{\mathrm{in},0}$ | `kg/kg` | physical sample | `operation` | unresolved | 1 | linear coordinate | `operations/fixed_bed.yaml` | Baseline inlet humidity ratio. | compositional inlet schedule | natural; disjoint `operation` OOD | `generation_schedule` | schedule/scalar adapter | `case.json`; schedule/scalar HDF5 | indirectly generates transient boundary conditioning; not state | operation unresolved | main text |
| `omega_in_amp` | $\Delta\omega_{\mathrm{in}}$ | `kg/kg` | physical sample | `operation` | unresolved | 1 | linear coordinate | `operations/fixed_bed.yaml` | Inlet humidity-ratio schedule amplitude. | compositional inlet schedule | natural; disjoint `operation` OOD | `generation_schedule` | schedule/scalar adapter | `case.json`; schedule/scalar HDF5 | indirectly generates transient boundary conditioning; not state | operation unresolved | main text |
| `schedule.corr` | $\rho_{T,\omega}$ | `1` | generator control | `operation` | unresolved | 1 | linear coordinate | `operations/fixed_bed.yaml` | Cross-correlation of temperature and humidity schedule latent processes. | compositional inlet schedule | natural; disjoint `operation` OOD | `generation_schedule` | schedule/scalar adapter | `case.json`; schedule/scalar HDF5 | indirectly generates transient boundary conditioning; not state | operation unresolved | appendix / implementation |
| `schedule.timescale_rel` | $\tau_{\mathrm{sched}}/t_{\max}$ | `1` | generator control | `operation` | unresolved | 1 | log coordinate | `operations/fixed_bed.yaml` | Schedule correlation timescale divided by total duration. | compositional inlet schedule | natural; disjoint `operation` OOD | `generation_schedule` | schedule/scalar adapter | `case.json`; schedule/scalar HDF5 | indirectly generates transient boundary conditioning; not state | operation unresolved | appendix / implementation |
| `schedule.component_weights` | $\boldsymbol{\lambda}_{\mathrm{sched}}$ | `1` | generator control | `operation` | unresolved | 2 | 3-part simplex | `operations/fixed_bed.yaml` | Smooth, event, and trend schedule simplex. | compositional inlet schedule | natural; disjoint `operation` OOD | `generation_schedule` | schedule/scalar adapter | `case.json`; schedule/scalar HDF5 | indirectly generates transient boundary conditioning; not state | operation unresolved | appendix / implementation |
| `schedule.event_count` | $n_{\mathrm{event}}$ | `1` | generator control | `operation` | unresolved | 1 | integer selection | `operations/fixed_bed.yaml` | Number of generated schedule events. | compositional inlet schedule | natural; disjoint `operation` OOD | `generation_schedule` | schedule/scalar adapter | `case.json`; schedule/scalar HDF5 | indirectly generates transient boundary conditioning; not state | operation unresolved | appendix / implementation |
| `schedule.event_duration_rel` | $d_{\mathrm{event}}/t_{\max}$ | `1` | generator control | `operation` | unresolved | 1 | log coordinate | `operations/fixed_bed.yaml` | Event duration divided by total schedule duration. | compositional inlet schedule | natural; disjoint `operation` OOD | `generation_schedule` | schedule/scalar adapter | `case.json`; schedule/scalar HDF5 | indirectly generates transient boundary conditioning; not state | operation unresolved | appendix / implementation |
| `schedule.event_width_rel` | $w_{\mathrm{event}}/t_{\max}$ | `1` | generator control | `operation` | unresolved | 1 | log coordinate | `operations/fixed_bed.yaml` | Event-edge width divided by total schedule duration. | compositional inlet schedule | natural; disjoint `operation` OOD | `generation_schedule` | schedule/scalar adapter | `case.json`; schedule/scalar HDF5 | indirectly generates transient boundary conditioning; not state | operation unresolved | appendix / implementation |
| `rho_bu_dry_ref` | $\rho_{\mathrm{bu,dry,ref}}$ | `kg/m^3` | physical sample | `material_properties` | unresolved | 1 | log coordinate | `materials/<family>.yaml` | Reference dry bulk density at calibration porosity. | material scalar/field contract | natural; disjoint `material_properties` OOD | case/scalar adapter | scalar/reviewed expression | `case.json`; scalar/static HDF5 | solver/calibration provenance; not a direct baseline input | material unresolved | main text |
| `k_gr` | $k_{\mathrm{gr}}$ | `W/(m*K)` | physical sample | `material_properties` | unresolved | 1 | log coordinate | `materials/<family>.yaml` | Dry granular-phase thermal conductivity. | material scalar/field contract | natural; disjoint `material_properties` OOD | case/scalar adapter | scalar/reviewed expression | `case.json`; scalar/static HDF5 | transient baseline scalar conditioning | material unresolved | main text |
| `cp_gr_dry` | $c_{p,\mathrm{gr,dry}}$ | `J/(kg*K)` | physical sample | `material_properties` | unresolved | 1 | log coordinate | `materials/<family>.yaml` | Dry granular-phase specific heat capacity. | material scalar/field contract | natural; disjoint `material_properties` OOD | case/scalar adapter | scalar/reviewed expression | `case.json`; scalar/static HDF5 | transient baseline scalar conditioning | material unresolved | main text |
| `r_surf_0` | $r_{\mathrm{surf},0}$ | `1/s` | physical sample | `material_properties` | unresolved | 1 | log coordinate | `materials/<family>.yaml` | Reference surface-moisture transfer rate. | material scalar/field contract | natural; disjoint `material_properties` OOD | case/scalar adapter | scalar/reviewed expression | `case.json`; scalar/static HDF5 | transient baseline scalar conditioning | material unresolved | main text |
| `r_int_surf` | $r_{\mathrm{int/surf}}$ | `1` | physical sample | `material_properties` | unresolved | 1 | log coordinate | `materials/<family>.yaml` | Internal-to-surface transfer-rate ratio. | material scalar/field contract | natural; disjoint `material_properties` OOD | case/scalar adapter | scalar/reviewed expression | `case.json`; scalar/static HDF5 | transient baseline scalar conditioning | material unresolved | main text |
| `f_surf` | $f_{\mathrm{surf}}$ | `1` | physical sample | `material_properties` | unresolved | 1 | logit coordinate | `materials/<family>.yaml` | Initial surface-water fraction. | material scalar/field contract | natural; disjoint `material_properties` OOD | case/scalar adapter | scalar/reviewed expression | `case.json`; scalar/static HDF5 | transient baseline scalar conditioning | material unresolved | main text |
| `T_amb` | $T_{\mathrm{amb}}$ | `K` | physical sample | `operation` | unresolved | 1 | linear coordinate | `operations/fixed_bed.yaml` | Ambient temperature and initial granular-phase temperature source. | compositional inlet schedule | natural; disjoint `operation` OOD | `generation_schedule` | schedule/scalar adapter | `case.json`; schedule/scalar HDF5 | transient baseline boundary conditioning | operation unresolved | main text |
| `density_calibration` | $(\rho_{\mathrm{bu,dry,ref}},\varepsilon_{\mathrm{bed,cal,ref}})$ | `rho_bu_dry_ref: kg/m^3, eps_bed_cal_ref: 1` | coupled selection | — | unresolved | 0 | optional complete-pair mode | `materials/<family>.yaml` | Optional evidence-preserving paired dry-bulk-density and calibration-porosity records. | material scalar/field contract | alternative complete natural/OOD pairs | case/scalar adapter | scalar/reviewed expression | `case.json`; scalar/static HDF5 | solver/calibration provenance; not a direct baseline input | material unresolved | appendix / implementation |
| `bed.structure.fine_weight` | $\alpha_{b,f}$ | `1` | generator control | — | derived | 0 | complement_of_one(bed.structure.coarse_weight) | registry + generator derivation | Derived complementary fine contribution to the bed multiscale field. | multiscale bed field | inherits sources | `generation_fields` | indirect: `fields.csv` | `case.json`; generated field/static HDF5 | indirectly generates steady inputs; transient ablation provenance | derived; formula validation | appendix / implementation |
| `initial_moisture.structure.fine_weight` | $\alpha_{X,f}$ | `1` | generator control | — | derived | 0 | complement_of_one(initial_moisture.structure.coarse_weight) | registry + generator derivation | Derived complementary fine contribution to initial-moisture structure. | bounded multiscale dry-basis field | inherits sources | `generation_fields` | indirect: `X_0_db_field` | `case.json`; field diagnostics/static HDF5 | transient initial-state provenance/optional ablation; not baseline input | derived; formula validation | appendix / implementation |
| `eps_min_global` | $\varepsilon_{\min}$ | `1` | fixed support | — | unresolved | 0 | fixed value | `common.yaml` | Universal lower porosity bound. | Kozeny–Carman porosity map | fixed by owner; family OOD only | `generation_fields` | indirect: `eps_bed` | `case.json`; porosity/static HDF5 | solver-derived or archived provenance; not a direct baseline input | global value unresolved | appendix / implementation |
| `eps_max_global` | $\varepsilon_{\max}$ | `1` | fixed support | — | unresolved | 0 | fixed value | `common.yaml` | Universal upper porosity bound. | Kozeny–Carman porosity map | fixed by owner; family OOD only | `generation_fields` | indirect: `eps_bed` | `case.json`; porosity/static HDF5 | solver-derived or archived provenance; not a direct baseline input | global value unresolved | appendix / implementation |
| `eps_bed_cal_ref` | $\varepsilon_{\mathrm{bed,cal,ref}}$ | `1` | fixed support | — | unresolved | 0 | fixed value | `materials/<family>.yaml` | Material calibration porosity for dry bulk density. | Kozeny–Carman porosity map | fixed by owner; family OOD only | `generation_fields` | indirect: `eps_bed` | `case.json`; porosity/static HDF5 | solver/calibration provenance; not a direct baseline input | material unresolved | appendix / implementation |
| `X_target_wb` | $X_{\mathrm{target,wb}}$ | `1` | fixed physical | — | unresolved | 0 | fixed value | `materials/<family>.yaml` | Material wet-basis target moisture. | material scalar/field contract | fixed by owner; family OOD only | case/scalar adapter | scalar/reviewed expression | `case.json`; scalar/static HDF5 | stop/evaluation provenance; not a direct baseline input | material unresolved | main text |
| `oswin` | $\boldsymbol{\theta}_{\mathrm{Oswin}}$ | `A_osw: 1, B_osw: 1, C_osw: 1` | coupled selection | — | unresolved | 0 | complete-set selection | `materials/<family>.yaml` | Coupled Oswin A, B, and C equilibrium-isotherm record. | material scalar/field contract | complete natural/OOD sets | case/scalar adapter | scalar/reviewed expression | `case.json`; scalar/static HDF5 | selected components are transient baseline scalar conditioning | material unresolved | appendix / implementation |
| `T_init` | $T_{\mathrm{init}}$ | `K` | derived quantity | — | derived | 0 | copy(T_amb) | registry + generator derivation | Initial temperature derived from ambient temperature. | case/physics derivation | inherits sources | case derivation / COMSOL | reviewed expression/provenance | `case.json`; `case.json` plus derived effect | solver-derived or archived provenance; not a direct baseline input | derived; formula validation | main text |
| `r_surf` | $r_{\mathrm{surf}}$ | `1/s` | derived quantity | — | derived | 0 | copy(r_surf_0) | registry + generator derivation | Surface rate copied from its reference value. | case/physics derivation | inherits sources | case derivation / COMSOL | reviewed expression/provenance | `case.json`; `case.json` plus derived effect | solver-derived or archived provenance; not a direct baseline input | derived; formula validation | main text |
| `r_int` | $r_{\mathrm{int}}$ | `1/s` | derived quantity | — | derived | 0 | product(r_int_surf, r_surf) | registry + generator derivation | Internal rate derived from r_int_surf and r_surf. | case/physics derivation | inherits sources | case derivation / COMSOL | reviewed expression/provenance | `case.json`; `case.json` plus derived effect | solver-derived or archived provenance; not a direct baseline input | derived; formula validation | main text |
| `T_in_ref` | $\bar{T}_{\mathrm{in}}$ | `K` | derived quantity | — | derived | 0 | schedule_time_average(schedule) | registry + generator derivation | Time-average inlet temperature derived from the schedule. | case/physics derivation | inherits sources | case derivation / COMSOL | reviewed expression/provenance | `case.json`; `case.json` plus derived effect | solver-derived or archived provenance; not a direct baseline input | derived; formula validation | main text |
| `T_flow_ref` | $T_{\mathrm{flow,ref}}$ | `K` | derived quantity | — | derived | 0 | mean(T_in_ref, T_init) | registry + generator derivation | Mean of inlet-reference and initial temperatures. | case/physics derivation | inherits sources | case derivation / COMSOL | reviewed expression/provenance | `case.json`; `case.json` plus derived effect | solver-derived or archived provenance; not a direct baseline input | derived; formula validation | appendix / implementation |

## Exact numerical coordinates

Coordinates follow the source-owned ordered tuples. The three-component schedule simplex contributes exactly two coordinates; `T_amb` is operation coordinate 12.

### `airflow` — 28 dimensions

| Coordinate | Parameter | d | Rule | OOD group |
| --- | --- | --- | --- | --- |
| 1 | `kappa_mean` | 1 | log coordinate | `bed` |
| 2 | `kappa_cv` | 1 | linear coordinate | `bed` |
| 3 | `bed.structure.coarse_len_rel` | 1 | log coordinate | `bed` |
| 4 | `bed.structure.fine_len_rel` | 1 | log coordinate | `bed` |
| 5 | `bed.structure.coarse_weight` | 1 | logit coordinate | `bed` |
| 6 | `bed.structure.cross_scale_corr` | 1 | linear coordinate | `bed` |
| 7 | `bed.structure.fine_ani_x` | 1 | log coordinate | `bed` |
| 8 | `bed.structure.fine_ani_y` | 1 | log coordinate | `bed` |
| 9 | `bed.perturbations.amplitude` | 1 | linear coordinate | `bed` |
| 10 | `bed.perturbations.granularity` | 1 | linear coordinate | `bed` |
| 11 | `bed.perturbations.sign_bias` | 1 | linear coordinate | `bed` |
| 12 | `permeability.anisotropy.max_ratio` | 1 | linear coordinate | `bed` |
| 13 | `permeability.anisotropy.exponent` | 1 | linear coordinate | `bed` |
| 14 | `permeability.anisotropy.strength` | 1 | linear coordinate | `bed` |
| 15 | `permeability.orientation.jitter` | 1 | linear coordinate | `bed` |
| 16 | `permeability.orientation.smooth_len_rel` | 1 | log coordinate | `bed` |
| 17 | `porosity.anchor_rel` | 1 | log coordinate | `bed` |
| 18 | `porosity.smooth_len_rel` | 1 | linear coordinate | `bed` |
| 19 | `porosity.texture_amp` | 1 | linear coordinate | `bed` |
| 20 | `pressure_bc.mean` | 1 | linear coordinate | `operation` |
| 21 | `pressure_bc.sin_amp` | 1 | linear coordinate | `operation` |
| 22 | `pressure_bc.sin_freq` | 1 | linear coordinate | `operation` |
| 23 | `pressure_bc.sin_phase` | 1 | phase coordinate | `operation` |
| 24 | `pressure_bc.gauss_count` | 1 | integer selection | `operation` |
| 25 | `pressure_bc.gauss_amp` | 1 | linear coordinate | `operation` |
| 26 | `pressure_bc.gauss_width` | 1 | log coordinate | `operation` |
| 27 | `pressure_bc.gauss_jitter` | 1 | linear coordinate | `operation` |
| 28 | `pressure_bc.linear_amp` | 1 | linear coordinate | `operation` |

### `initial_moisture` — 8 dimensions

| Coordinate | Parameter | d | Rule | OOD group |
| --- | --- | --- | --- | --- |
| 1 | `initial_moisture.mean_db` | 1 | linear coordinate | `initial_moisture` |
| 2 | `initial_moisture.amplitude_db` | 1 | linear coordinate | `initial_moisture` |
| 3 | `initial_moisture.structure.coarse_len_rel` | 1 | log coordinate | `initial_moisture` |
| 4 | `initial_moisture.structure.fine_len_rel` | 1 | log coordinate | `initial_moisture` |
| 5 | `initial_moisture.structure.coarse_weight` | 1 | logit coordinate | `initial_moisture` |
| 6 | `initial_moisture.structure.cross_scale_corr` | 1 | linear coordinate | `initial_moisture` |
| 7 | `initial_moisture.structure.fine_ani_x` | 1 | log coordinate | `initial_moisture` |
| 8 | `initial_moisture.structure.fine_ani_y` | 1 | log coordinate | `initial_moisture` |

### `operation` — 12 dimensions

| Coordinate | Parameter | d | Rule | OOD group |
| --- | --- | --- | --- | --- |
| 1 | `T_in_base` | 1 | linear coordinate | `operation` |
| 2 | `T_in_amp` | 1 | linear coordinate | `operation` |
| 3 | `omega_in_base` | 1 | linear coordinate | `operation` |
| 4 | `omega_in_amp` | 1 | linear coordinate | `operation` |
| 5 | `schedule.corr` | 1 | linear coordinate | `operation` |
| 6 | `schedule.timescale_rel` | 1 | log coordinate | `operation` |
| 7–8 | `schedule.component_weights` | 2 | 3-part simplex | `operation` |
| 9 | `schedule.event_count` | 1 | integer selection | `operation` |
| 10 | `schedule.event_duration_rel` | 1 | log coordinate | `operation` |
| 11 | `schedule.event_width_rel` | 1 | log coordinate | `operation` |
| 12 | `T_amb` | 1 | linear coordinate | `operation` |

### `material_properties` — 6 dimensions

| Coordinate | Parameter | d | Rule | OOD group |
| --- | --- | --- | --- | --- |
| 1 | `rho_bu_dry_ref` | 1 | log coordinate | `material_properties` |
| 2 | `k_gr` | 1 | log coordinate | `material_properties` |
| 3 | `cp_gr_dry` | 1 | log coordinate | `material_properties` |
| 4 | `r_surf_0` | 1 | log coordinate | `material_properties` |
| 5 | `r_int_surf` | 1 | log coordinate | `material_properties` |
| 6 | `f_surf` | 1 | logit coordinate | `material_properties` |

Independent density therefore gives `28 + 8 + 12 + 6 = 54`. In the optional evidence-matched density-pair mode, independent density/calibration-porosity coordinates are replaced by a complete pair selection: the material block is 5 and the total 53. Pair/set selection adds no numerical coordinate. Both fine weights are derived as `fine_weight = 1 - coarse_weight` and add zero.

## Fixed, support, and derived quantities outside registry coordinates

Every row contributes `d = 0` and has no independent block/transform. Fixed constraints apply in ID and parameter OOD unless their owner changes; derived quantities inherit source assignment. “Formula fixed” points to the single equations in the operational guide.

| Canonical name | Report symbol | Unit | Owner/category | Status | Purpose/effect | Consumer / COMSOL | Persistence | Neural-operator relevance | Evidence | Report |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `p_ref` | $p_{\mathrm{ref}}$ | `Pa` | common fixed | unresolved | humidity-conversion pressure | schedule generator | config/case identity | schedule-generation constraint; not a direct model input | operation evidence | appendix |
| `T_in_max` | $T_{\mathrm{in,max}}$ | `K` | common fixed | configured | inlet-temperature ceiling | schedule constraint | config/case identity | boundary-generation constraint; not a direct model input | design constraint | main text |
| `omega_min`, `omega_max` | $\omega_{\min},\omega_{\max}$ | `kg/kg` | common fixed | unresolved | humidity-ratio bounds | schedule constraint | config/case identity | boundary-generation constraint; not a direct model input | operation evidence | appendix |
| `phi_clip_min`, `phi_clip_max` | $\phi_{\min},\phi_{\max}$ | `1` | common fixed | unresolved | relative-humidity bounds | schedule constraint | config/case identity | boundary-generation constraint; not a direct model input | operation evidence | appendix |
| `f_wet_dm_max` | $f_{\mathrm{wet,dm,max}}$ | `1` | common fixed | configured | dry-mass stop threshold | scalar adapter / stop | scalar HDF5 + status | solver stop/evaluation provenance; not a direct model input | design criterion | main text |
| `Lx`, `Ly` | $L_x,L_y$ | `m` | grid fixed | configured | bed extents | fields / COMSOL geometry | coordinates + metadata | coordinate design provenance; `x`,`y` carry the learned coordinates | design geometry | main text |
| `dx`, `dy` | $\Delta x,\Delta y$ | `m` | grid fixed | configured | boundary-inclusive spacing | fields/export validator | coordinate attributes | discretization provenance; not separate model channels | design geometry | appendix |
| `nx`, `ny` | $n_x,n_y$ | `1` | grid fixed | configured | point counts | fields/storage | shape metadata | tensor-shape provenance; not model channels | design geometry | implementation |
| `time.start`, `time.stop` | $t_0,t_{\max}$ | `h` | time fixed | configured | planned interval | schedule/study | schedule + time HDF5 | sequence-scope provenance; not model channels | design horizon | main text |
| `time.interval` | $\Delta t$ | `h` | time fixed | configured | regular output step | schedule/index | time HDF5 | transient step definition; not a model channel | design horizon | main text |
| `initial_moisture_bounds` | $[X_{0,db}^{\min},X_{0,db}^{\max}]$ | `kg/kg` | material support | unresolved | no-clipping envelope | moisture generator | case diagnostics | initial-state generation provenance; not baseline input | material evidence | appendix |
| `w_gr` | $w_{\mathrm{gr}}$ | `kg/m^3` | derived physical | formula fixed | total granular water density | COMSOL/domain | on-demand | solver-derived physical quantity; not a baseline input | model validation | main text |
| `X_db`, `X_wb` | $X_{db},X_{wb}$ | `1` | derived physical | formula fixed | local dry-/wet-basis moisture | COMSOL/domain | on-demand | solver/evaluation derivations; not baseline inputs | model validation | main text |
| `X_wb_bulk` | $X_{\mathrm{wb,bulk}}$ | `1` | derived global | formula fixed | mass-integrated wet-basis moisture | COMSOL global | global HDF5/status | evaluation and stop provenance; not a baseline input | COMSOL validation | main text |
| `rho_bu_dry` | $\rho_{\mathrm{bu,dry}}$ | `kg/m^3` | derived field | formula fixed | local dry bulk density | fields/COMSOL | static HDF5 | transient baseline static conditioning; not steady input | model validation | main text |
| `w_gr_0` | $w_{\mathrm{gr},0}$ | `kg/m^3` | derived field | formula fixed | initial granular water | fields/COMSOL initial state | case diagnostics | transient initial-state derivation; not per-step baseline input | model validation | main text |
| `cp_gr_eff` | $c_{p,\mathrm{gr,eff}}$ | `J/(kg*K)` | derived physical | moisture term unresolved | effective granular heat capacity | COMSOL expression | not exported | solver-derived physics; not direct baseline conditioning | evidence unresolved | main text |
| `m_evap` | $\dot{m}_{\mathrm{evap},V}$ | `kg/(m^3*s)` | derived physical | expression mapping unresolved | local volumetric evaporation source | COMSOL expression | not exported; only its integral is stored | solver-derived physics; not a model input | COMSOL validation | main text |
| `f_wet_dm` | $f_{\mathrm{wet,dm}}$ | `1` | derived global | formula fixed | dry-mass fraction above target | COMSOL output/stop | global HDF5/status | solver stop/evaluation diagnostic; not a model input | COMSOL validation | main text |

## Adapter, state, output, and diagnostic quantities

These physical/solver-facing rows have `d = 0`, no independent block/transform, inherit case ID/OOD assignment, and are owned by `generation_profiles.py` plus the confirmed profile mapping. COMSOL headers/tags remain unresolved. HDF5 persists ordered names and units; coordinates/time also carry unit attributes.

| Canonical name | Report symbol | Unit | Roles | Purpose | Status/COMSOL | Persistence | Neural-operator relevance | Report |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `x` | $x$ | `m` | spatial input | Cartesian x coordinate | mapping unresolved | HDF5 role table/status | steady input; transient baseline static conditioning | appendix / implementation |
| `y` | $y$ | `m` | spatial input | Cartesian y coordinate | mapping unresolved | HDF5 role table/status | steady input; transient baseline static conditioning | appendix / implementation |
| `Kxx` | $K_{xx}$ | `m^2` | spatial input, static | permeability tensor xx | mapping unresolved | HDF5 role table/status | steady input; archived transient ablation, not transient baseline | main text |
| `Kxy` | $K_{xy}$ | `m^2` | spatial input, static | symmetric permeability tensor xy | mapping unresolved | HDF5 role table/status | steady input; archived transient ablation, not transient baseline | main text |
| `Kyy` | $K_{yy}$ | `m^2` | spatial input, static | permeability tensor yy | mapping unresolved | HDF5 role table/status | steady input; archived transient ablation, not transient baseline | main text |
| `eps_bed` | $\varepsilon_{\mathrm{bed}}$ | `1` | spatial input, static | bed porosity | mapping unresolved | HDF5 role table/status | steady input; transient baseline static conditioning | main text |
| `p_bc` | $p_{bc}$ | `Pa` | spatial input, static | inlet-pressure boundary field | mapping unresolved | HDF5 role table/status | steady input; archived transient ablation, not transient baseline | main text |
| `X_0_db_field` | $X_{0,db}$ | `kg/kg` | spatial input, static | initial dry-basis moisture field | mapping unresolved | HDF5 role table/status | transient initial-state provenance/optional ablation; no baseline input | main text |
| `t` | $t$ | `h` | schedule, global | physical time | mapping unresolved | HDF5 role table/status | sequence/index provenance; not a learned state or conditioning channel | appendix / implementation |
| `T_in` | $T_{\mathrm{in}}$ | `K` | schedule | inlet temperature | mapping unresolved | HDF5 role table/status | transient boundary conditioning; not a state variable | main text |
| `omega_in` | $\omega_{\mathrm{in}}$ | `kg/kg` | schedule | inlet humidity ratio | mapping unresolved | HDF5 role table/status | archived boundary provenance; not baseline transient conditioning | appendix / implementation |
| `phi_in` | $\phi_{\mathrm{in}}$ | `1` | schedule | derived inlet relative humidity | mapping unresolved | HDF5 role table/status | transient boundary conditioning; not a state variable | main text |
| `A_osw` | $A_{\mathrm{Oswin}}$ | `1` | scalar | first coefficient of the selected Oswin equilibrium-isotherm record | equation convention unresolved | case/scalar HDF5 provenance | transient baseline scalar conditioning | main text |
| `B_osw` | $B_{\mathrm{Oswin}}$ | `1` | scalar | second coefficient of the selected Oswin equilibrium-isotherm record | equation convention unresolved | case/scalar HDF5 provenance | transient baseline scalar conditioning | main text |
| `C_osw` | $C_{\mathrm{Oswin}}$ | `1` | scalar | third coefficient of the selected Oswin equilibrium-isotherm record | equation convention unresolved | case/scalar HDF5 provenance | transient baseline scalar conditioning | main text |
| `u` | $u$ | `m/s` | static | x velocity | mapping unresolved | HDF5 role table/status | steady output; transient baseline static conditioning | main text |
| `v` | $v$ | `m/s` | static | y velocity | mapping unresolved | HDF5 role table/status | steady output; transient baseline static conditioning | main text |
| `p` | $p$ | `Pa` | static | pressure | mapping unresolved | HDF5 role table/status | steady output; transient baseline static conditioning | main text |
| `T` | $T$ | `K` | transient | temperature state | mapping unresolved | HDF5 role table/status | transient dynamic state | main text |
| `phi` | $\phi$ | `1` | transient | relative-humidity state | mapping unresolved | HDF5 role table/status | transient dynamic state | main text |
| `w_surf` | $w_{\mathrm{surf}}$ | `kg/m^3` | transient | surface-water density | mapping unresolved | HDF5 role table/status | transient dynamic state | main text |
| `w_int` | $w_{\mathrm{int}}$ | `kg/m^3` | transient | internal-water density | mapping unresolved | HDF5 role table/status | transient dynamic state | main text |
| `X_wb_max` | $X_{\mathrm{wb,max}}$ | `1` | global, final status | maximum local wet-basis moisture | mapping unresolved | HDF5 role table/status | evaluation diagnostic; not a model input | appendix / implementation |
| `X_wb_q95_mass` | $X_{\mathrm{wb},q95,dm}$ | `1` | global, final status | dry-mass-weighted moisture q95 | mapping unresolved | HDF5 role table/status | evaluation diagnostic; not a model input | appendix / implementation |
| `T_out_mean` | $\bar{T}_{\mathrm{out}}$ | `K` | global | outlet mean temperature | mapping unresolved | HDF5 role table/status | evaluation diagnostic; not a model input | appendix / implementation |
| `phi_out_mean` | $\bar{\phi}_{\mathrm{out}}$ | `1` | global | outlet mean relative humidity | mapping unresolved | HDF5 role table/status | evaluation diagnostic; not a model input | appendix / implementation |
| `m_w_gr` | $m_{w,\mathrm{gr}}$ | `kg` | global | granular water mass | mapping unresolved | HDF5 role table/status | mass-balance diagnostic; not a model input | appendix / implementation |
| `m_v_gas` | $m_{v,\mathrm{gas}}$ | `kg` | global | gas vapor mass | mapping unresolved | HDF5 role table/status | mass-balance diagnostic; not a model input | appendix / implementation |
| `m_dot_evap` | $\dot{m}_{\mathrm{evap}}$ | `kg/s` | global | integrated evaporation rate | mapping unresolved | HDF5 role table/status | mass-balance diagnostic; not a model input | appendix / implementation |
| `m_dot_v_in` | $\dot{m}_{v,\mathrm{in}}$ | `kg/s` | global | inlet vapor mass-flow rate | mapping unresolved | HDF5 role table/status | mass-balance diagnostic; not a model input | appendix / implementation |
| `m_dot_v_out` | $\dot{m}_{v,\mathrm{out}}$ | `kg/s` | global | outlet vapor mass-flow rate | mapping unresolved | HDF5 role table/status | mass-balance diagnostic; not a model input | appendix / implementation |
| `t_final` | $t_{\mathrm{final}}$ | `h` | final status | exact final time | mapping unresolved | HDF5 role table/status | terminal status provenance; not a model input | appendix / implementation |
| `f_wet_dm_final` | $f_{\mathrm{wet,dm,final}}$ | `1` | final status | terminal wet dry-mass fraction | mapping unresolved | HDF5 role table/status | terminal stop/evaluation provenance; not a model input | appendix / implementation |
| `T_min_final` | $T_{\min,\mathrm{final}}$ | `K` | final status | terminal minimum temperature | mapping unresolved | HDF5 role table/status | terminal diagnostic; not a model input | appendix / implementation |
| `T_max_final` | $T_{\max,\mathrm{final}}$ | `K` | final status | terminal maximum temperature | mapping unresolved | HDF5 role table/status | terminal diagnostic; not a model input | appendix / implementation |
| `phi_min_final` | $\phi_{\min,\mathrm{final}}$ | `1` | final status | terminal minimum relative humidity | mapping unresolved | HDF5 role table/status | terminal diagnostic; not a model input | appendix / implementation |
| `phi_max_final` | $\phi_{\max,\mathrm{final}}$ | `1` | final status | terminal maximum relative humidity | mapping unresolved | HDF5 role table/status | terminal diagnostic; not a model input | appendix / implementation |

Quantities already defined in earlier tables and also carried by adapters are `rho_bu_dry` (transient baseline static conditioning), `X_wb_bulk`, `f_wet_dm`, and `X_target_wb` (evaluation/stop provenance only). Their adapter roles do not create second definitions.

### Learning-contract role cross-check

This table adds roles only; authoritative definitions remain in the catalogues above.

| Contract role | Canonical fields |
| --- | --- |
| Registered steady inputs | `x`, `y`, `Kxx`, `Kxy`, `Kyy`, `eps_bed`, `p_bc` |
| Registered steady outputs | `p`, `u`, `v` |
| Transient dynamic state | `T`, `phi`, `w_surf`, `w_int` |
| Transient baseline static conditioning | `x`, `y`, `u`, `v`, `p`, `eps_bed`, `rho_bu_dry` |
| Transient boundary conditioning | `T_in(t_n)`, `T_in(t_{n+1})`, `phi_in(t_n)`, `phi_in(t_{n+1})`, `T_amb` |
| Transient scalar conditioning | `r_surf_0`, `r_int_surf`, `f_surf`, `A_osw`, `B_osw`, `C_osw`, `k_gr`, `cp_gr_dry` |
| Archived transient ablation | `Kxx`, `Kxy`, `Kyy`, `p_bc`, `X_0_db_field` |
| Transient increment target | `delta_T`, `delta_phi`, `delta_w_surf`, `delta_w_int`, derived as the next regular one-hour state minus the current state |

### Package OOD relevance and stationary conditioning

Dataset packages preserve the registry's physical OOD group and the numerical block that generated each selected unit. These are different ownership axes: for example, pressure-profile parameters belong to the `operation` OOD group but to the `airflow` numerical block because they generate the steady `p_bc` field.

| Dataset view | Parameter-OOD eligibility |
| --- | --- |
| `transient_drying` | Valid selected units from `airflow`, `initial_moisture`, `operation`, and `material_properties`; all four physical group selectors are retained |
| `steady_flow` | Selected units in the `airflow` block only, because their effects are represented by `Kxx`, `Kxy`, `Kyy`, `eps_bed`, or `p_bc` |

One immutable parameter-OOD package stores the combined membership plus group and parameter indexes. Every included case records selected units, group, registry kind/block/transform/unit, natural support, OOD support, sampled or coupled value, and block row/design evidence. A steady-ineligible case remains in excluded-source provenance with its reason; it is not relabeled or silently returned by another group selector. Family OOD always uses the held-out family's natural support.

The source profile's exhaustive `steady_flow_conditioning` record owns stationary-solution dependencies:

| Owner | Meaning |
| --- | --- |
| `model_input` | Case-varying solution dependency represented by the registered steady input contract |
| `package_fixed` | Solution dependency fixed across the package, with exact value and unit bound into package identity |
| `not_used` | Quantity explicitly verified not to affect the stationary solve for that profile |

The required audit inventory is `Kxx`, `Kxy`, `Kyy`, `eps_bed`, `p_bc`, air dynamic viscosity, air density, `T_flow_ref`, and profile reference temperature, plus an explicit list of any additional case-varying solver scalars. `T_flow_ref` is derived for case provenance but is not assumed to be a model input. If it or another solution dependency varies and is neither an existing model input nor package-fixed, package preflight reports hidden conditioning and stops. A stationary-solution contract ID and canonical digest make cross-profile compatibility explicit; template or profile similarity is never inferred.

The four top-level package regimes are `id`, `parameter_ood`, `near_family_ood`, and `far_family_ood`. ID membership is assigned to physical cases before temporal expansion and persists as `train`, `validation`, or `id_test` in both matched views. OOD regimes are evaluation-only by default and never fit the steady normalizer.

## Semantic conflict decisions

| Potential conflict | Final decision |
| --- | --- |
| Permeability / conductivity | `kappa_mean` is mean scalar permeability; `Kxx`, `Kxy`, `Kyy` are tensor components; `k_gr` is granular thermal conductivity. |
| Humidity / porosity / phase | `phi` and `phi_in` are relative humidity; porosity is `eps_bed`; `pressure_bc.sin_phase` is pressure-sinusoid phase with symbol $\varphi_{p,\sin}$. |
| Density / correlation | `rho_bu_dry_ref` and `rho_bu_dry` are densities; `schedule.corr` is a dimensionless latent correlation. |
| Gaussian count | `pressure_bc.gauss_count` is retained as the validated pressure-profile integer count; report symbol $n_{p,G}$ removes mathematical ambiguity. |
| Evaporation | `m_evap` is local volumetric source; `m_dot_evap` is the integrated rate. |
| Mass / rate / fraction | `m_*` denotes mass, `m_dot_*` rate/flow, and `f_*` a dimensionless fraction. |
| Moisture basis | `_db`/`_wb` stay explicit; `X_wb_bulk` is mass integrated, never an unweighted spatial mean. |
| State qualifier | `_0` is initial, `_ref` reference/calibration, and `_final` terminal. |
| Scientific / execution units | Scientific names are unit-free. `runtime_s` and `timeout_seconds` are deliberately operational timing evidence. |

## Execution-only configuration

All rows have `d = 0`, no scientific block/report symbol/neural input, and are owned by `execution/cluster_cpu.yaml`. They persist only in launch, execution, timing, and status evidence and never enter scientific identities or COMSOL physics.

| Execution path | Unit/type | Purpose | Status |
| --- | --- | --- | --- |
| `runtime.executable` | command | COMSOL invocation | configured |
| `runtime.module_initialization` | commands | native modules | configured |
| `runtime.timeout_seconds` | `s` | per-process timeout | configured |
| `runtime.maximum_failures` | `1` | failure cap | configured |
| `runtime.extra_arguments` | arguments | COMSOL CLI additions | configured empty |
| `retention.retain_raw_csv`, `retention.retain_solved_model` | bool | storage policy | configured false |
| `cluster.max_nodes`, `cluster.cases_per_node`, `cluster.cores_per_case`, `cluster.max_parallel_cases`, `cluster.cores_per_node` | `1` | campaign-wide resource plan/caps | configured |
| `cluster.scheduler_kind`, `cluster.partition` | names | Slurm routing | configured |
| `cluster.wall_time` | Slurm duration | job limit | unresolved |
| `cluster.scheduler_options` | arguments | site-approved additions | configured empty |
| `site.cpu_host` | hostname | CPU host | configured |
| `site.scheduler`, `site.partition`, `site.cores_per_node` | names/count | site cross-check | configured |
| `site.python_module`, `site.comsol_module` | module names | native runtime versions | configured |
| `site.python_executable`, `site.comsol_executable` | commands | native executables | configured |

## Identities and visible names

Identity fields have no unit, sampling block, report symbol, or dimension.

| Canonical field | Meaning | Persistence |
| --- | --- | --- |
| `material_family` | role-neutral material identity | material filename, manifests, HDF5 |
| `simulation_profile` | reference-simulation contract | campaign, case, HDF5 |
| `dataset_view` | `steady_flow` or `transient_drying` package contract | campaign package, manifest, runtime request |
| `evaluation_regime` | `id`, `parameter_ood`, `near_family_ood`, or `far_family_ood` | campaign package and dataset manifest |
| `dataset_membership` | case-level `train`, `validation`, `id_test`, or owning OOD regime | dataset manifest and runtime metadata |
| `sampling_regime` | `natural` or `parameter_ood` | batch, case, HDF5 |
| `case_input_id` | digest of profile-pairable scientific inputs | case/HDF5/dataset source |
| `simulation_case_id` | digest including simulation profile | case/HDF5/dataset source |
| `batch_name`, `batch_id` | readable grammar, digest-bound identity | directories/manifests |
| `dataset_name`, `dataset_id` | readable grammar, content identity | payload/manifest |
| `campaign_id`, `campaign_run_id` | scientific campaign, then science+commit+resources | campaign manifests |
| `scientific_config_digest` | resolved scientific contract digest | case/HDF5 |
| `steady_flow_conditioning_digest` | exhaustive stationary-solution dependency contract | case and dataset provenance |
| `git_commit` | exact source commit | campaign through dataset provenance |

```text
batch_name   = <simulation_profile>__<material_family>__<sampling_regime>
dataset_name = <dataset_view>__<ordered-material-list>__<evaluation-regime>
```

Visible names omit versions, timestamps, seeds, counts, redundant profile words, and technical digests. Immutable IDs append digest prefixes and retain full provenance.

## Material schema and evidence research

Every material file has exactly these role-neutral top-level keys:

```text
schema_kind, schema_version, material_family, executable,
taxonomy, product_form, parameter_values, evidence
```

`schema_kind` is `generation_material`; filenames equal `material_family`. `taxonomy` uniformly contains `common_name`, `species`, `market_class`, `cultivar`, `specificity_status`. `product_form` uniformly contains `whole_or_split`, `shell_state`, `skin_or_seed_coat_state`, `description`, `specificity_status`. Campaigns—not materials—own family roles.

| Material family | Common name | Species | Market class | Cultivar | Product scope | Shell state | Skin/coat |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `lentil` | Lentil | unresolved | unresolved | unresolved | Whole lentil; market class, cultivar, and seed-coat condition unresolved. | unresolved | unresolved |
| `chickpea` | Chickpea | unresolved | Kabuli | unresolved | Whole chickpea of the intended Kabuli market class; cultivar and seed-coat condition unresolved. | unresolved | unresolved |
| `kidney_bean` | Kidney bean | unresolved | red kidney bean | unresolved | Whole kidney bean of the intended red kidney bean market class; cultivar and seed-coat condition unresolved. | unresolved | unresolved |
| `field_pea` | Field pea | unresolved | yellow field pea | unresolved | Whole field pea of the intended yellow field pea market class; cultivar and seed-coat condition unresolved. | unresolved | unresolved |
| `almond` | Almond | unresolved | unresolved | unresolved | Whole almond kernel without its hard shell; cultivar and skin condition unresolved. | `hard_shell_removed` | unresolved |

No species, cultivar, or skin/coat condition is inferred. Kabuli, red kidney bean, and yellow field pea are the user-selected market classes, not literature claims. `almond` means a whole kernel after hard-shell removal, not an in-shell almond, split kernel, flour, or meal.

Each evidence entry contains `source`, `evidence_type`, `confidence`, `temperature_range`, `humidity_range`, `cultivar_or_market_class`, `product_form`, and `status`. Evidence types distinguish measured, fitted, inferred, and explicitly justified assumed records. A material remains non-executable while any required record or specificity is unresolved.

Accepted evidence must record a stable citation/identifier; method, unit/equation convention, basis, confidence, and validity conditions; cultivar/market class and exact product state; conditioning, packing, geometry, scale, temperature, humidity, airflow, and moisture basis where relevant. Natural ranges and parameter-OOD ranges require defensible support and a nonzero transformed-coordinate gap. Oswin coefficients stay source-consistent complete sets; density/calibration-porosity uses either independent evidence or complete matched pairs, never mixed components. A defensible unresolved result is preferable to an invented value or source.

Bed/permeability evidence must match packing, geometry, orientation, scale, moisture, and flow regime. Initial-moisture evidence uses dry basis and must support the analytical no-clipping envelope. Thermal/drying evidence must match method, apparatus/model, temperature, humidity, airflow, particle form, and fit equation. Oswin evidence identifies the exact convention, branch, water-activity/humidity domain, basis, and fit domain.

## Reporting use

Main text introduces physical inputs, governing quantities, block totals, material scopes, and major outputs. The appendix carries the full registry, morphology/realization controls, transforms, adapter schema, OOD construction, and evidence records. Execution and identity fields belong in reproducibility metadata, not the scientific dimension count.
