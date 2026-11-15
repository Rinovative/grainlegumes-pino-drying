# Evaluation

Evaluation works on saved evidence. An explicit artifact step writes reference and predicted fields for a completed run; the notebooks discover runs, load validated artifacts and show compatible views. Opening a notebook never trains, runs inference or builds missing artifacts.

[Documentation index](README.md)

## Notebooks

| Notebook | Use |
| --- | --- |
| `notebooks/01_eda_generation_inputs.ipynb` | Prepared Generation inputs before COMSOL: parameters, fields, schedules, design coverage, failures by parameter |
| `notebooks/02_eda_datasets.ipynb` | Completed cases and Datasets of both tasks: states, static fields, schedules, target attainment, solver timing |
| `notebooks/03_component_checks.ipynb` | Training-free component priors and physics checks on the COMSOL reference fields of one Train or validation membership ([Component checks](#component-checks)) |
| `notebooks/04_training_pipeline.ipynb` | Training configurations, data roles, objectives, stage dependencies and saved runs (inspection only) |
| `notebooks/05_eval_single_model.ipynb` | One saved run |
| `notebooks/06_eval_comparison_models.ipynb` | Two or more compatible saved runs |

Scientific cells need a compute-node kernel ([Setup and execution](01_setup_and_execution.md#notebooks-on-compute-nodes)). The notebooks resolve `STORAGE_ROOT`, scan `03_experiments/` once and build their selectors from persisted run records. Runs without usable artifacts stay visible with the exact command that creates them.

## Build artifacts

A completed training run builds its artifacts by default. To build or validate them explicitly:

```bash
./scripts/slurm_ml.sh --mode gpu --partition gpu --cpus-per-task 4 \
  --mem 32G --time 02:00:00 --gres gpu:rtx6000ada:1 artifacts \
  --run-dir RUN_DIRECTORY
```

Instead of `--run-dir` (repeatable), `--task TASK` (the task's folder below `storage/03_experiments/`) or `--runs-root DIR` selects all evaluable runs below a discovery root, skipping Optuna trial runs; `--run-name` narrows the selection. Only completed runs are evaluated: discovery skips a run that is `interrupted` or ended `unsuccessful` (no eligible checkpoint) with one `[SKIP]` line and continues (exit 0 when nothing else fails), while a run named by `--run-dir` or `--run-name`, or an Optuna trial directory given directly, is refused. Before any case is predicted, the service admits checkpoint, scaling artifact, Dataset identity and the requested grid for the architecture. Incomplete or older artifacts are rejected; `--rebuild` recreates the exact target. A non-finite prediction does not stop the build in either task: every case is classified and published with its validity document (see Prediction validity).

In both tasks every per-case NPZ is named by the Dataset package case ID, for example `lentil__case_0003__fe83e052fbf2.npz`: material, the campaign-local case number as in `01_generation/…/materials/lentil/case_0003`, and a short digest that keeps cases of different campaigns apart. The `case_index` stored inside steady files stays the one-based Dataset position.

### Field metrics of both tasks

Both tasks report the same per-field metrics through one implementation (`learning.metrics.metrics.field_metric_values`), used by training validation, the artifacts and the notebooks. Every case contributes float64 sums per physical field: element count, squared and absolute error, reference square, and the H1 count, error and reference energies. Every metric is formed from these sums.

- **Primary, per field:** physical and scaled RMSE and MAE (`physical_rmse_<f>`, `physical_mae_<f>`, `normalized_rmse_<f>`, `normalized_mae_<f>`; scaled means divided by the Train-only output scale). Each case enters with its mean over its grid points and, in transient drying, its time points; cases weigh equally within a material and materials equally within the role. RMSE takes its root after this mean.
- **Secondary, per field:** relative L2 and H1 on physical fields (`physical_relative_l2_<f>`, `physical_relative_h1_<f>`). Each case and field gets √(E/R) from its error energy E and reference energy R; these ratios then take the same equal case-in-material and equal material mean, each role separately. L2 sums f² over the full grid. H1 sums f² + (1 m)²·|∇f|² over the interior without the two outermost cells at every edge; the gradient is taken in metres with central differences on the full grid, so no one-sided stencil enters the interior, and there is no time derivative.
- **Summary values** weigh the output groups equally. The objective combines the squared errors of each group before its root ([Training](05_training.md#data-loss-and-selection-objective)); `physical_relative_l2` and `physical_relative_h1` average the field ratios within their group and then the groups. A plain field mean would count the velocity group and the two moisture compartments twice.

Transient drying forms the horizon value from each case's energies summed over its post-origin time points before dividing. Its time curve (`relative_norms_over_physical_time`) forms √(E/R) per case at each exact physical time of the full autonomous rollouts. It reports the cases and the supported cases at that time; a non-finite predicted state lowers only the supported cases of its own time, the rollout's other times still count.

<details>
<summary><strong>Near-zero references and ranking</strong></summary>

A relative norm needs a reference. When a case's reference RMS over the norm's domain (√(R/n)) falls below the field's floor, its value is unavailable and counted (`<metric>_unavailable_cases`). It is never set to zero and never stabilized with an epsilon. A material without any supported case makes the role value unavailable, and so does an unavailable field in a summary value. In the transient horizon a temporarily unavailable state still adds its energies to the case's sums; the horizon value's availability is checked on those sums.

| Field | Floor (reference RMS) | Reason |
| --- | --- | --- |
| `p` | 1 Pa | The base inlet pressure is 250–750 Pa (OOD up to 1200 Pa), so a reference below 1 Pa has practically no driving pressure |
| `u`, `v` | 10⁻⁴ m/s | 0.1 mm/s; mainly guards the transverse component of a nearly uniform flow |
| `T` | 1 K | References in kelvin lie near 300 K, so the floor is never reached; relative T errors stay secondary (3 K at 300 K are 1 %) |
| `phi` | 10⁻³ | 0.1 % relative humidity, below any humidity the drying air carries |
| `w_surf`, `w_int` | 0.1 kg/m³ | Grain water at the drying target is about 80 kg/m³ (lentil: 0.12 wet basis at 630 kg/m³ dry bulk density); the floor marks a practically empty compartment |

The floors are declared in the task definitions (`reference_floors`) and recorded in each artifact's `metric_evidence`.

Relative error maps and distributions divide by the reference RMS of the field over the case (or the shown map), never per pixel; below the field's floor (a vector magnitude takes its largest component floor) that case or map has no relative value, is counted and shown as unavailable, never stabilised.

RMSE takes its root after the case and material mean, whereas a relative norm divides within each case first, so the two can rank runs differently; both are reported.

> *Paper sentence.* "For each physical field and evaluation role, we report secondary relative L2 errors on the full grid and relative spatial H1 errors on the two-cell-cropped interior, using derivatives in metres and length scale 1 m; ratios are formed per case at each reported time or from summed post-origin energies over the declared horizon, then averaged equally within materials and equally across the role's declared materials, with reference norms below predefined field-specific thresholds reported as unavailable with counts."

</details>

### Steady case metrics

Each steady Parquet row stores the per-field sums (`physical_<sum>_<f>`) and, derived from them, `physical_rmse_<f>`, `normalized_rmse_<f>` and the case's group-weighted `physical_relative_l2` and `physical_relative_h1` (NaN when a field of the case is below its floor). On load the role values are recomputed from the sums and the stored per-case values are checked against them. The table also holds:

- `physical_velocity_vector_rmse`, `normalized_velocity_vector_rmse` and `physical_rmse_speed_magnitude`: RMSE of the velocity vector (in m/s and scaled) and of |u| (in m/s) on the full grid.
- `momentum_residual_mse`, `flux_balance_mse`, `pressure_inlet_mse`, `pressure_outlet_mean_square` and `pressure_boundary_mse`: the dimensionless training terms ([Training](05_training.md#steady-flow)) with physical finite differences for every run; the momentum term on the `EVAL_PAD` interior, the flux balance over every row, the boundary terms on the full inlet and outlet edges.

### What a transient artifact stores

Per ID case one NPZ bundle (named `<package case id>.npz`) holds the reference states over every stored hour and, per distinct prediction chain (one-step, rolling and autonomous origins; about one per stored hour, about 170 chains per case at 168 hours), the predicted states over every hour after the chain's origin plus the computed-prefix flag. The first state of a chain equals the reference state at its origin and is taken from there on load; chains that are bit-identical, such as the full autonomous rollout and the rolling origin at hour 0, are stored once and shared by their records. The writer checks before publishing that every record rebuilds bit for bit from the stored arrays. Metrics, validity documents and summaries are computed from in-memory predictions before writing and stored in the Parquet index. Predicted and reference increments (state differences) and the non-finite and physical-invalid masks (recomputed with the current validity rule) are derived on load. The raw scaled network outputs are not stored; regenerate them from the checkpoint if needed. This takes about 0.95 GB per ID case at 168 hours instead of 4.1 GB (smoke of 2026-11-09: 159 chains and 0.89 GB for a case of 157 hours, 146 chains and 0.82 GB for 144 hours), and every hour remains available for browsing and animations.
### Evaluation grid

Artifacts of both tasks are always built and read on the full canonical 251 × 401 grid, whatever the training stride; there is no option to choose another grid. Both tasks train on the full grid by default (`data.spatial_stride: 1`); stride 2 is only for the stride comparison:

| Grid | Shape |
| --- | --- |
| Canonical source grid, artifact grid and default training grid | 251 × 401; transient model input `[B, 28, 251, 401]`, output `[B, 4, 251, 401]` |
| Training and scaling grid of the stride-2 comparison | 126 × 201 |

Channel-wise scaling broadcasts to the artifact grid. Each architecture must support the full grid: FNO and RNO need their retained modes to fit the axes, U-NO every resampling level; a transient model trained on a different resolution additionally needs a passing exact-output probe of the loaded checkpoint. Reference and prediction share one grid digest over indices, coordinates, field order and extent; every metric uses the full evaluation grid. An artifact whose spatial identity records a grid stride is refused by the schema check.

Both tasks publish under `analysis/id`, `analysis/material_ood/<dataset>` and `analysis/parameter_ood/<dataset>`. `analysis/id` holds the ID-test membership; every role folder holds its case table as `artifacts.parquet` plus `npz/` and the provenance marker, so the Dataset name appears only as the folder. Both tasks build one artifact per non-empty OOD role over all of its parts, in split order: `analysis/material_ood/<dataset>` covers the material-OOD parts and `analysis/parameter_ood/<dataset>` the training Dataset's parameter-OOD cases; an empty role gets no folder. One material-OOD source gives `analysis/material_ood/<dataset>/artifacts.parquet` under its Dataset ID; several give one combined table `analysis/material_ood/combined_ood__<digest>/artifacts.parquet`, the digest taken over the ordered Dataset IDs, not one folder per Dataset. Every row names its `source_dataset_id`, and the provenance `dataset` block binds every part (`source_dataset_ids`, `source_identities`, `case_ids_by_source`, `membership_digests`). Before any case is predicted, each part's current Dataset identity is checked against the identity saved in the split; a changed part is refused. Optuna trial runs get no artifacts in either task.

### Single-case debug artifacts

```bash
./scripts/slurm_ml.sh --mode gpu --partition gpu --cpus-per-task 4 \
  --mem 32G --time 02:00:00 --gres gpu:rtx6000ada:1 artifacts \
  --run-dir RUN_DIRECTORY --one-case --split id \
  --output-root /workspace/storage/evaluation_debug/DEBUG_NAME
```

Repeated `--case-id CASE_ID` selects an exact case set instead of `--one-case`. Bounded output needs one transient run, one `--split id|material_ood|parameter_ood` and a new output root outside the run's `analysis/`; it is never canonical and never published to W&B.

## Transient protocols and metrics

| Protocol | Meaning |
| --- | --- |
| Teacher-forced one-step | Each step starts from the reference state |
| Full autonomous rollout | The model feeds its own prediction from the first state to the end |
| Rolling origins | Autonomous rollouts from early, middle and late reference states |

Fixed horizons are 1, 2, 4, 8, 16, 32, 64 and 128 transitions; `full` uses every available transition. A horizon longer than a case is recorded as unsupported. Every result keeps elapsed physical time and both cumulative and endpoint reductions. The RNO starts every case from its own initial hidden state. For one-step records and rolling origins it carries the state a teacher-forced pass builds over the case's reference transitions up to that step, as in training; the full autonomous rollout carries only its own state from the first transition (`recurrent_state_policy: teacher_forced_reference_prefix`).

The central metric is `normalized_drying_group_macro_rmse` with the checkpoint's Train-only scaling and weights `T = 1/3`, `phi = 1/3`, `w_surf = 1/6`, `w_int = 1/6`. This objective and the normalized channel errors reduce like checkpoint selection in every table, ranking and pipeline comparison: each case enters with its mean squared error over its records, cases weigh equally within a material and materials equally, and each channel takes its root after the mean; a summary group (material, mode, horizon, scope) applies the same rule to its own cases ([Data loss and selection objective](05_training.md#data-loss-and-selection-objective)). The per-field metrics of every summary group, physical and scaled, and its relative L2 and H1 follow [Field metrics of both tasks](#field-metrics-of-both-tasks). Total grain moisture and bulk moisture errors follow the same rule: each case enters with its mean over its states (bulk moisture over its states with a valid bulk value), and a material without such a state makes the bulk value unavailable. Reports add total grain moisture, bulk dry- and wet-basis moisture over the full grid (trapezoidal weighting as in Generation), target status, plausibility and stability diagnostics, and the number of contributing cases. The overview shows a missing value with its counters, as steady flow does.

Only the full autonomous record carries the reference drying-target evidence from Generation, including the irregular exact-stop time. Final-gap errors pair reference and prediction only at the same physical time.

<details>
<summary><strong>Prediction validity</strong></summary>

Both tasks follow one rule. Values are never clamped, filtered or imputed; validity documents keep per-channel counts (finite, non-finite, NaN, +inf, −inf) and the first invalid coordinate. Metrics that need complete finite support report required, computed, finite and non-finite counts instead of a value; the role's `normalized_group_macro_rmse` is such a metric and is never averaged over the valid cases alone. Whenever the artifact step generates or loads a role with invalid predictions it prints `[WARNING] artifact completed with invalid predictions | valid=… | finite_but_physically_invalid=… | nonfinite=…` and one `[WARNING] invalid prediction case` line per case; the training end line is described in [Training](05_training.md).

Both tasks present an unavailable metric the same way and never raise for it while showing it. Tables show `unavailable (F finite of R required values)` in place of a number; the comparison scoreboard draws no bar and the Pareto and capacity views no point for it, and they name the counts at the tick, in a "Not plotted" note or in the legend. Per-case value plots and quantiles leave out non-finite cases, which the warning above names. A transient summary group (frame, material, mode, horizon, scope) with any non-finite predicted value reports these counts instead of its error metrics and is never reduced over its finite records alone; its frame then has no pipeline degradation entry.

Steady flow classifies each case as `VALID` or `NONFINITE` from its physical `p`, `u`, `v` prediction. It defines no physical range, so it has no `FINITE_BUT_PHYSICALLY_INVALID` class. A `NONFINITE` case keeps its NPZ arrays unchanged (prediction, error and residual fields may hold NaN or ±inf); its Parquet row keeps the element counts and stores NaN for every other metric, so a missing value never reads as zero. The document names the first non-finite channel and grid index `[y, x]`. The writer recomputes the document from the stored prediction, and cache hits and reads bind the checked bytes: admission checks the payload set and the Parquet digest, and each case payload is verified against its recorded digest when the case is first opened (before any byte is parsed); any disagreement is refused.

In transient drying a prediction is `VALID`, `FINITE_BUT_PHYSICALLY_INVALID` or `NONFINITE`. A finite state is physically invalid where `T` leaves [0, 2000] K, `phi` leaves [0, 1.10] or `w_surf` or `w_int` is negative; the 2 × 2 node blocks at the four corners of the evaluation grid are excluded from the physical-invalid count of every channel (non-finite values there still count). The COMSOL reference itself exceeds `phi` = 1 slightly: the evaporation source `max(w_surf − w_eq, 0)` allows no re-wetting and the vapour equation has no condensation sink, so humid air cooling over colder grain is not bounded by saturation. In the transient smoke of 2026-10-30, 2 of 6 cases exceeded 1 in the interior, at most 1.048, only in the first 3 h and never above 1.10; in the smoke of 2026-11-09 none of 10 cases did (maximum 0.936 outside the corner blocks). This is an accepted model limitation and is re-checked on the production data. The reference also has an inlet-corner singularity (`phi` 5.6 at the corner node, −0.64 at its wall neighbour). The persisted validity contract carries these bounds and `excluded_corner_block`. The plausibility diagnostics are separate: they count `phi` outside the strict unit interval [0, 1] without tolerance or corner exclusion, as an informative supersaturation count. Physical bounds are diagnostics only: a finite state outside them is fed back, counted and reported, never stopped or clamped. An autonomous rollout, in training validation and in artifacts alike (`autonomous_rollout` in `src/learning/transient/learning_transient_rollout.py`), stops only the affected case at its first non-finite increment or state, keeps that step as evidence and leaves the case's later steps uncomputed; other modes and cases continue. Training validation logs per state field `bounds/<field>/violating_cases` and, when at least one case violates, `bounds/<field>/first_violation_hours` (aggregates derived from the per-case evidence: the count of violating validation cases and the earliest first violation over them, with the same bounds and corner exclusion), and keeps in the point's `objective_history` entry `bound_violations` (per state field and violating ID case, the physical hours of its first violation), beside the eligibility of the point ([Training](05_training.md#budget-schedule-and-validation-points)). Artifacts keep per record and per case summary each channel's own first violation (`first_bound_violation`: rollout step, transition and physical time, `null` without one) beside the overall `first_invalid`.

</details>

<details>
<summary><strong>Staged finalization and resume</strong></summary>

Both tasks build a target in one deterministic sibling stage, `.<target>.stage` (for example `analysis/.id.stage`). A failed attempt keeps it; `--rebuild` clears it. A stage that already carries its completion marker is not generated again: the next attempt writes the timing block, admits the stage with the same role validator as a cache hit and publishes it. Artifacts record the analysis digest of their producer (`producer.analysis_sha256`); a complete target or complete stage with another analysis digest than the running code is refused with the `--rebuild` hint, while reading an artifact never compares it with the running code ([Code identity](07_reproducibility_and_validation.md#code-identity)). An incomplete stage resumes per case in the transient task, where each completed case manifest binds role, checkpoint, normalizer, protocol, grid, summary, NPZ digest, timing and validity, so a restart reconstructs reused rows from their NPZ and computes only missing cases. The resume descriptor binds the analysis digest and the timing hardware, so a partial stage started with another analysis digest or on other hardware is refused with the `--rebuild` hint instead of mixing cases computed by different analysis code or timed on different devices. A complete bounded (scoped) stage is reused only when its persisted `selected_case_ids` equal the request's cases and its analysis digest equals the running one, otherwise it is refused and its directory must be deleted (bounded generation takes no `--rebuild`). The roles of one run always share one analysis producer: no build path generates a missing role beside a published role of other analysis code without `--rebuild`, and the refusal comes before any inference. One run-wide lock serializes the artifact builds of a run, so two concurrent builders cannot both pass this check. The artifact preparation after training reads existing roles as evidence; where it may replace incompatible artifacts, it rebuilds such an older role first and then generates the missing one. An incomplete steady stage is generated again as a whole target, because its batched inference and runtime comparison are measured over the whole target in one warmed session. Finalization reconciles membership, writes provenance, timing and summaries, publishes the marker and removes the private resume metadata.

Publication renames the complete stage to the target. A new target appears in that one atomic rename. Replacing an existing target, after `--rebuild`, takes two renames: the old target moves to `.<target>.backup.<random>`, the stage takes the target name, and the backup is then deleted; if the second rename fails, the old target is moved back. Between the two renames the target is briefly absent. A reader at that moment fails with a missing-artifact error and never sees a mix of old and new content. After a hard crash in exactly that window the target is missing, the completed stage is still in place and the previous artifact lies under its backup name: running the artifact command again without `--rebuild` publishes the stage, and the backup directory is deleted by hand. An atomic exchange (`renameat2` with `RENAME_EXCHANGE`) is not used, because its support on the filesystem through which the compute nodes mount `storage/` is unverified.

</details>

## Comparisons

The comparison notebook selects an ordered set of saved runs with a shared material set and Dataset. Runs must agree on task, Dataset membership, objective, evaluation protocol and artifact grid, and their artifacts must carry the same `producer.analysis_sha256`; frames of different or missing analysis producers are refused (steady flow and transient sessions alike). In steady views exact case membership is checked per dataset, a role together with the materials of its choice, so two materials of one run can be shown separately; each cut of a loaded run is bound once in that run's session, so redraws reuse its caches. Architecture, seed and stage stay separate visible attributes. The summary ranks transient runs by the selection objective of their cumulative full autonomous rollouts and steady runs by the role's `normalized_group_macro_rmse` aggregate; both reduce cases equally within a material and materials equally before the root, and a run with an unavailable case is not ranked.

<details>
<summary><strong>Controlled training-stage comparison</strong></summary>

A matched set (A0/B/A1) is admitted when it shares the exact A0, architecture, Dataset, split, scaling, initial checkpoint and B receipt. B versus A1 is the protocol comparison at matched compute; B versus A0 includes the additional post-handoff training. The x-axis keeps the admitted clock. A diverged B or A1 is admitted with its best eligible checkpoint and the signed difference actual - planned, negative only after a divergence. Every arm reports its end (`training_status`, budget progress, eligible points; B's from its receipt, A0's and A1's from their own training outcome) in the performance column `training_status`; the comparison plot shows each arm without a result as one status line (arm, status, budget progress, eligible points) read from the admitted comparison. A B or A1 without an eligible checkpoint has no artifact: its selected run (lifecycle `unsuccessful`) enters only the comparison, from its validated terminal summary and the receipt in its leaf, with an unavailable result; a B's receipt must be the one its admitted end builds, and every B and A1 arm must name the selected A0's handoff manifest; a B left out of the selection enters from A1's receipt when the receipt records no eligible checkpoint of B. A0 alone may then carry the artifacts, for example after a B that diverged in its first group and its 0-second A1.

</details>

<details>
<summary><strong>Airflow-to-drying pipeline</strong></summary>

The pipeline view compares three conditions: A uses COMSOL airflow and COMSOL drying (reference), B the drying surrogate on COMSOL airflow, C the drying surrogate on airflow predicted by a steady-flow operator. It reports B/A (drying surrogate error), C/A (complete pipeline error) and C/B (airflow substitution). C needs a compatible steady-flow checkpoint, normalizer, profile and case mapping; without them the view states the reason.

</details>

## Views

Steady-flow views cover overview and scoreboards, global and spatial error, architecture and capacity, error decomposition, physical consistency, spectral fidelity, metadata sensitivity, sample inspection, outliers and runtime. Physical-consistency residuals of every steady run use the same physical finite-difference derivatives, independent of the derivative a run was trained with, so runs share one yardstick. They are the dimensionless training terms (see [Training](05_training.md#steady-flow)): momentum residual maps divided by Δp/L_y, the cross-section flux balance as a per-case value and a mean profile over the height, and the pressure-boundary mismatch divided by Δp².

Transient views reuse the same workspace with case, physical-time, channel and scale controls:

| Figure | View |
| --- | --- |
| `transient_state_maps` | Sample viewer with case, channel, physical time and scale lock |
| `transient_central_error_time` | Error over physical time |
| `transient_horizon_error` | Temporal rollout |
| `relative_norms_over_physical_time` | Temporal rollout: relative L2 and H1 per field with the supported cases at each time |
| `transient_endpoint_cumulative` | Global error |
| `transient_target_time` | Temporal rollout with censored and unavailable counts |
| `transient_pipeline_degradation` | Runtime, with the airflow pipeline evidence |
| `transient_timing_distributions`, `transient_timing_speedups` | Runtime with the declared measurement boundary |
| `transient_accuracy_inference_time`, `transient_accuracy_speedup` | Runtime trade-off, paired by case |
| `transient_training_performance_compute` | Controlled training-stage comparison |

Aggregate maps and time summaries use only cases that have the selected exact physical time and show the count `n(t)`; bands need at least two contributors. Labels, units, Celsius conversion, colours and orientation come from the presentation layer shared with EDA.

<details>
<summary><strong>Error modes and display stride</strong></summary>

The case overviews of both tasks offer the **signed error** (prediction − reference, in the quantity's display unit: °C, % RH, percentage points for moisture, Pa, m/s), the default and recommended kind, plus absolute and normalized absolute error. The steady views call the absolute error **MAE** and add **Relative [%]** (absolute error over the reference RMS of the shown field, times 100; unavailable below the field's floor, as above). The steady mean error maps show the signed kind as the mean of prediction − reference over the selected cases, on the same symmetric blue-white-red percentile scale with arrows.

Normalized absolute error is `abs(prediction - reference) / s` with the saved Train scale `s` = max(σ, floor) of the field, the scale that training and checkpoint selection use (`output_scales` of the saved normalizer for steady flow, the `state` group of the saved scaling for transient, the `grain_moisture` group for `w_gr`). Its weighted mean is `sum(weight * error) / sum(weight)`. It is a diagnostic, independent of Celsius/Kelvin offsets, and requires the same field, unit, Dataset and scaler across compared models.

**Display stride** (`Full`, `2`, `4`, `Auto`) only thins the drawn map after all errors, statistics and colour bounds are computed on the full grid. `Auto` uses stride 2 above a longest axis of 400 points and 4 above 800.

</details>

<details>
<summary><strong>Field-map colour scales</strong></summary>

`src.analysis.presentation.field_maps` defines all scientific field maps: a data-derived continuous field uses 11 colours and 12 strictly increasing boundaries through one `BoundaryNorm`, shared by contours, images and colourbar ticks; a locked round scale (all state and error scales of the Evaluation maps) has 9 to 13 bands with one colour per band. Scale kinds are linear positive, linear signed (neutral interval around zero), log positive (geometric boundaries), absolute, signed and relative error, categorical and constant. Locked comparisons share all their boundaries; values outside them set the colourbar extension.

The transient sample viewer (`transient_state_maps`) and the steady case overview ("Sample GT vs Prediction") follow one display rule, with the same title over every panel of a column and one metric only: each error map carries a small line directly above it with its mean |e| over all finite cells of the field (independent of the display sampling), three significant digits, in the bar's unit and power of ten ("mean |e| = 0.23 Pa", "mean |e| = 15 × 10⁻⁵ m/s" under a bar in [10⁻⁵ m/s]; the normalized error "mean |e|/σ_train = 0.31", a map without finite errors "mean |e| n/a"); no other panel carries a metric:

- **State scale:** by default (the lock toggle on) reference and prediction share the reference's colour scale in every frame, the 5th to the 95th percentile of the reference's finite displayed values (after the display conversion); the prediction never sets the colours, and arrows on the colour bar mark values outside (in the transient GIF a state bar draws an arrow at an end when any frame has values beyond that end of its own scale, and then in every frame, so the bar never moves; the viewer's single frame draws only its own, those of all times once any field is fixed over time). With the toggle off each panel gets its own 5th to 95th percentile scale. The viewer and the GIF export follow the toggle. A degenerate percentile range falls back to the reference's own full range, a constant reference to a padded reference-only scale. The transient state scale of a field either follows each frame or is fixed over all stored times; the transient viewer's "Fixed over time" checkboxes (one per channel, shared by the GIF export) choose it per field, by default fixed for every field but the moisture fields (T and φ fixed, the wet-basis moisture X_wb per frame). A fixed field's scale is the 5th to 95th percentile of the pooled displayed values of the reference (of the prediction itself with the toggle off); with any fixed field the viewer builds its norms over all stored times (once per case and setting), so the viewer frame equals the GIF frame at that time: the error scale, its arrows, the state arrows and the decimals and the mean line's power of ten are those of all times; the steady and the transient (viewer and GIF) state maps are drawn discrete, on 9 to 13 round bands covering the percentile range tightly (one colour per band of the field's colormap, a `BoundaryNorm`; with the toggle on the prediction uses the reference's bands, off its own). The bands are the candidates of step m × 10^k with m in 1, 2, 2.5 or 5 that reach the upper end from the largest step multiple at or below the lower end; the one with the smallest covered span wins, a tie goes to the count nearest 11, then to the larger step (case 42 pressure, 16.25 to 304.9 Pa: 13 bands of 25 Pa from 0 to 325 Pa). In the steady overview the velocity components u and v use the `turbo` colormap, p keeps its own; the overview can show the velocity magnitude |u| in place of u and v (`fields=("p", "U")`), on its own colormap with the same streamlines.
- **Error scale:** the signed error is symmetric, ±max(|5th percentile|, |95th percentile|) of the signed error, on a blue-white-red colormap with a neutral centre band; the absolute and the normalized error (`|e|/σ_train`, σ_train = the saved Train scale max(σ, floor)) run from 0 to the 95th percentile. The levels are round (the same step candidates and choice as the state bands, 9 to 13 bands; the magnitude scales start at 0, the signed one has an odd count of 9, 11 or 13 bands) and errors beyond them get arrows; in the transient viewer they are black. The GIF fixes the error scale per quantity over the whole rollout.
- **Note line:** one small line directly under the figure title reads "Colour scales: 5th–95th percentile; prediction uses the reference scale; arrows mark values outside"; it names the reference scale only where the scale is shared (lock toggle on), and says nothing about times or which fields are fixed over time.
- **Tick labels:** every map colour bar writes exactly the decimals its round step needs (step 25 or 50: none, 2.5: one, 0.25 or 0.01: two; for a signed bar those of the band width; at most 3, constant for that bar over all GIF frames: the decimals the finest frame needs, a frame with a degenerate percentile range counted with its full range). An error bar (centred on 0) whose step would need more than 3 decimals writes its ticks as multiples of a power of ten and names the factor in its unit (`[10⁻⁴ m/s]`), also fixed per bar over all frames. A state bar never carries a factor: it shows at most 3 decimals (the most its frames need, over all GIF frames; for a field fixed over time those of its pooled scale), and a frame or map whose round step is finer than that is near-uniform and is widened symmetrically about its centre to 11 steps of the last decimal, so it shows few colours; a constant frame keeps one tick at its value. State bars label their band boundaries; the absolute and normalized error runs from 0 to its round 95th percentile with its boundaries labelled; the signed error has an odd count n of bands (9, 11 or 13) of width w (n/2 × w at least the percentile reach), the white band centred on 0, and labels the band centres 0, ±w, ±2w and so on.
- **Titles:** no figure or column title carries a metric number.
- **Label style:** every quantity label of the EDA, evaluation and generation-input plots (axis, colour-bar and panel-title text) follows one style from one field table, chosen with `analysis.presentation.field_labels.set_label_style(style)` (`label_style()` reads it). The default `"symbol"` writes formula symbols (`T [°C]`, `φ [%]`, `X_wb,surf [%]`, `nRMSE(T) [-]`), `"name"` writes names with the percent basis in the unit (`Temperature [°C]`, `Relative humidity [% RH]`, `Surface moisture [% w.b.]`, `Temperature nRMSE [-]`), and `"name_symbol"` writes both (`Temperature T [°C]`, `Relative humidity φ [%]`). Water densities read as wet-basis moisture where the dry bulk density is known and as water content in kg/m³ elsewhere; differences are `pp` in every style and dimensionless units `[-]`. Channel controls, table headings and running text keep their names and the basis in the unit whatever the style.
- **Names:** a label starts with a capital letter ("Grain moisture", "Surface moisture", "Internal moisture"); running text keeps lower case. The steady overview names a model in its title by its display name, such as PI-U-NO for a U-NO trained with physics loss terms, not by the run's configuration name.
- **GIF speed:** the `fps` field next to the Export GIF button (integer, 1 to 10, default 3) sets the frames per second.

</details>

## Component checks

`notebooks/03_component_checks.ipynb` runs before training and gives cheap, training-free priors for small model components, next to the physics checks of the reference fields; the service is `src.analysis.component_checks`. It reads only published Datasets and their admitted source `case.h5` files, trains nothing and writes nothing. Every check reads one membership of the Dataset partition, `train` (the notebook default) or `validation`, filtered before any case limit; ID-test and OOD roles are refused, and every table names its Dataset, membership and cases. A check yields a prior for an ablation or a candidate range, never a decision.

| Task | Checks |
| --- | --- |
| Steady flow | Derivative backends (`physical`, `spectral` with Fourier continuation) against exact derivatives of manufactured non-periodic fields per edge band, crop and precision; the physics-loss cost per backend and device (forward plus backward at the training batch, node-specific, not persisted); the dimensionless training terms of the COMSOL solution per backend, interior crop and precision; the momentum split into pressure, viscous and drag parts, with the ⅔(∇·u) trace part shown separately; the pointwise divergence variants removed from the loss; residuals by distance to the edges; finite-difference against spectral derivatives; the floor and the 1 % smooth-error sensitivity per crop; the ratio of each term under controlled smooth, noisy, scaling and offset errors; the truncation error of each output field under the Fourier modes a spectral layer keeps, with domain padding 0 and 0.1 |
| Transient drying | The same truncation floors for states and increments at the training stride; the bilinear full-grid floor of the stride views 1, 2 and 4 under the runtime's endpoint rule; scaled-increment statistics of the transient scaling fitted on the membership (quantiles, exact zeros, loss shares under the group weights of the state loss) and the stored time steps; grain water, pore-gas vapour, evaporation and advective boundary fluxes from the stored fields against the COMSOL global series; closure of the grain, gas and total water balance; pointwise sorption residuals; inlet rows against the schedule; reference values outside the evaluation validity rule; which balance terms the Dataset and artifacts do not persist |

A summary table applies the proposed criteria (named constants in `component_checks_summary.py`, open while the component-check decision criteria are open) to the results computed so far. Fourier floors are priors only: the networks truncate lifted features, not the raw fields. Analytic operator checks stay in `tests/domain/`. The notebook needs a compute-node kernel; the physics checks alone took about 35 s and 1.7 GB on the smoke Datasets, the new sections are not yet measured.

## Timing and speedup

| Measurement | Boundary |
| --- | --- |
| `drying_no_prepared_rollout_wall_seconds` | Wall interval of a prepared rollout request, cold or warm as recorded; excludes Dataset, checkpoint, context and request construction |
| Model-only time | Model calls only; excludes validation, transfer, scaling and reconstruction |
| COMSOL `Solution time` | Steady and transient study phases parsed from the solver log |
| COMSOL process and case execution | Broader wall-clock boundaries of the case job |

Both tasks report a speed-up as Σ COMSOL time / Σ surrogate time over the paired cases, those that have both clocks; the solver time (`Solution time`) and the total COMSOL process time are separate speed-ups. A case without a COMSOL clock is not paired and is counted as unavailable, never as zero; a speed-up without any paired case is unavailable. Benchmarks record warm-up, precision, batch size, surrogate hardware (device, GPU or CPU) and, per case from the Generation timing, the solver host and its requested cores.

> *Paper sentence.* "The speed-up of both surrogates was the sum of COMSOL solver times divided by the sum of surrogate times over the paired cases, reported separately for solver time and total process time together with the batch size and the hardware of surrogate and solver."

- **Steady flow:** the artifact step reads each case's COMSOL times through its Dataset manifest from the Generation evidence. It pairs them with the synchronized, warmed forward of the case's loader batch, divided equally among the batch's cases (`solver_speedup`, `process_speedup`). The steady Runtime section shows both speed-ups with their paired and unavailable counts, the sums, the batch size and both hardware records.
- **Transient drying:** cold time, raw warmed repetitions and payload identity are recorded per component. Speed-ups that combine different boundaries are labelled component-composed, and missing components stay unavailable with their reason.
- **Work per stage:** beside the runtime table of both tasks, one row per completed stage (steady, A0, B, A1) read from its `summary.json` by `analysis.evaluation.run_discovery.stage_work_row`: its end status (`training_outcome.status`: `completed`, `diverged` or `no_eligible_checkpoint`); architecture and trainable parameters; optimisation, total and attempted seconds with their coverage; device, driver, CUDA build and precision, joined over the attempts that computed optimisation seconds (an attempt without its value as `unavailable`); optimizer steps, supervised and forward transitions; unit kind, seen, Train and full-set units, seen and Train cases, k, memory batch; budget, its unit and the overrun ([Training](05_training.md#units-per-update)); and the inference seconds per case (steady: the mean batch-amortized forward; transient: the mean over cases of each case's median model-call seconds of the drying rollout). A value the run or artifact does not record reads `unavailable`, never 0.

Every artifact of both tasks carries a required `timing` block in `artifact_provenance.json`, outside scientific identity: the runtime comparison (`report`, whose speed-ups are recomputed from its cases on admission), the transient component availability and the operational snapshot of the build (phases, forward calls, reused cases). It is written into the stage before validation, so a timing failure publishes nothing; an artifact without it is refused with the `--rebuild` hint. The `producer` block records `analysis_sha256`, the digest of the analysis projection that built the artifact. Runtime provenance records device (on CUDA with GPU name and `cuda_build_version`, the CUDA release PyTorch was built with), PyTorch version, precision, inference batch size and the executing frozen source tree (`source_tree_id`, provenance only) for both tasks. Transient role admission binds every timing report case to its component evidence (the generator's complete field set with a finite non-negative materialization duration) and to its sequence record identity (PT payload identity).

## Export and tracking

Selecting a view creates no file. **Export PDF** saves the selected figure into the repository's `notebooks/` folder, named by plot and UTC timestamp; an existing file is never overwritten. The export is refused, with a message in the panel status, in a frozen source tree or when `notebooks/` is missing or a symlink.

Artifacts, their provenance and their summaries stay local: the artifact step sends nothing to W&B. The W&B run holds what training sends (run configuration, tags, history curves of the training events with validation at the 20 budget points, terminal summary) plus the system metrics and run metadata the W&B SDK records itself ([Training](05_training.md)).

## Inference

Loading a run's weights for inference or artifacts requires its recorded model digest to equal the running model projection ([Code identity](07_reproducibility_and_validation.md#code-identity)). A completed transient run loads without opening its Dataset:

```python
from src import learning

transient = learning.inference.transient
context = transient.load_transient_inference_context(run_dir=run_dir, device="cuda")
```

`predict_transient_step` takes one physical interval; `rollout_transient_autonomous` takes a validated interval sequence and feeds its own reconstructed states back. Both support FNO, U-NO and RNO. Inputs are both boundary endpoints, startup support, exact times and `dt`, and the COMSOL reference airflow or a compatible external airflow. Without an explicit evaluation shape the context uses the checkpoint's training grid.

<details>
<summary><strong>Compute device and memory</strong></summary>

`evaluation_compute_device` (`auto`, `cpu`, `cuda`, `cuda:N`) applies to the aggregate error reductions; artifact building has its own `device_policy`. Discovery, tables, plotting and single-case maps run on CPU. `auto` uses the GPU whenever a Slurm GPU allocation is visible, otherwise the CPU; an explicit CUDA request checks the visible device and never falls back. The probe of 2026-11-09 on one RTX 6000 Ada measured the GPU faster from one case on (4 CPU threads, four float64 channels on 251 × 401, transfer included): 1 case 4.7 ms CPU against 2.8 ms GPU, 16 cases 123.8 ms against 56.3 ms.

Notebook admission verifies artifact, manifest, frame, provenance and output structure with the Parquet digest, for both tasks; each case payload is verified against its recorded digest when its case is first opened, before any byte is parsed, so an unopened payload is never hashed. Each file (payloads and run checkpoints) is hashed at most once per process; the digest is reused while its path, size, modification time and inode are unchanged. Sequence arrays are decompressed only when their case is requested and kept in a bounded LRU cache; aggregate views load all admitted records. Closing a workspace clears its caches.

</details>
