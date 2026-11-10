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
  --run-dir RUN_DIRECTORY --evaluation-spatial-stride 1
```

Instead of `--run-dir` (repeatable), `--task TASK` (the task's folder below `storage/03_experiments/`) or `--runs-root DIR` selects all evaluable runs below a discovery root, skipping Optuna trial runs; `--run-name` narrows the selection. Only completed runs are evaluated: discovery skips a run that ended `budget_unmet` or `interrupted` with one `[SKIP]` line and continues (exit 0 when nothing else fails), while a run named by `--run-dir` or `--run-name`, or an Optuna trial directory given directly, is refused. Before any case is predicted, the service admits checkpoint, scaling artifact, Dataset identity and the requested grid for the architecture. Incomplete or older artifacts are rejected; `--rebuild` recreates the exact target. A non-finite prediction does not stop the build in either task: every case is classified and published with its validity document (see Prediction validity).

In both tasks every per-case NPZ is named by the Dataset package case ID, for example `lentil__case_0003__fe83e052fbf2.npz`: material, the campaign-local case number as in `01_generation/…/materials/lentil/case_0003`, and a short digest that keeps cases of different campaigns apart. The `case_index` stored inside steady files stays the one-based Dataset position.

### Steady case metrics

The metrics below are computed from the physical output fields p, u and v:

- `rel_l2`: ‖e‖/(‖y‖ + 10⁻¹²) per field on the full grid, averaged over the fields, so each field counts equally whatever its magnitude.
- `rel_h1`: per field, NeuralOp `H1Loss` (the same implementation as the training/validation H1) with physical spacing in metres on the interior without the two outermost samples (`EVAL_PAD = 2`); derivatives are taken on that interior (central inside, third-order one-sided at its edge), averaged over the fields.
- `physical_rmse_<field>`, `physical_velocity_vector_rmse` and `physical_rmse_speed_magnitude`: RMSE per field, of the velocity vector and of |u| on the full grid, in the field units.
- `momentum_residual_mse`, `flux_balance_mse`, `pressure_inlet_mse`, `pressure_outlet_mean_square` and `pressure_boundary_mse`: the dimensionless training terms ([Training](05_training.md#steady-flow)) with physical finite differences for every run; the momentum term on the `EVAL_PAD` interior, the flux balance over every row, the boundary terms on the full inlet and outlet edges.

### What a transient artifact stores

Per ID case one NPZ bundle (named `<package case id>.npz`) holds the reference states over every stored hour and, per distinct prediction chain (one-step, rolling and autonomous origins; about one per stored hour, about 170 chains per case at 168 hours), the predicted states over every hour after the chain's origin plus the computed-prefix flag. The first state of a chain equals the reference state at its origin and is taken from there on load; chains that are bit-identical, such as the full autonomous rollout and the rolling origin at hour 0, are stored once and shared by their records. The writer checks before publishing that every record rebuilds bit for bit from the stored arrays. Metrics, validity documents and summaries are computed from in-memory predictions before writing and stored in the Parquet index. Predicted and reference increments (state differences) and the non-finite and physical-invalid masks (recomputed with the current validity rule) are derived on load. The raw scaled network outputs are not stored; regenerate them from the checkpoint if needed. This takes about 0.95 GB per ID case at 168 hours instead of 4.1 GB (smoke of 2026-11-09: 159 chains and 0.89 GB for a case of 157 hours, 146 chains and 0.82 GB for 144 hours), and every hour remains available for browsing and animations.
### Evaluation grid

`--evaluation-spatial-stride N` (default `1`) selects an endpoint-preserving grid from the canonical 251 × 401 source grid, independently of the training stride; a stride other than 1 is supported for transient drying only. With the maintained training stride 2:

| Grid | Shape |
| --- | --- |
| Canonical source grid | 251 × 401 |
| Training and scaling grid (stride 2) | 126 × 201 |
| Artifact grid, stride 1 | 251 × 401; model input `[B, 28, 251, 401]`, output `[B, 4, 251, 401]` |
| Artifact grid, stride 2 | 126 × 201 for every tensor |

Channel-wise scaling broadcasts to the requested grid. Each architecture must support the grid: FNO and RNO need their retained modes to fit the axes, U-NO every resampling level; a different resolution additionally needs a passing exact-output probe of the loaded checkpoint. Reference and prediction share one grid digest over indices, coordinates, mask, field order and extent.

| Task | Stride 1 | Stride N > 1 |
| --- | --- | --- |
| Both tasks | `analysis/id`, `analysis/material_ood/<dataset>`, `analysis/parameter_ood/<dataset>` | Transient drying only: `analysis/grid_sN/id`, `analysis/grid_sN/<material_ood\|parameter_ood>/<dataset>` |

`analysis/id` holds the ID-test membership; every role folder holds its case table as `artifacts.parquet` plus `npz/` and the provenance marker, so the Dataset name appears only as the folder. Both tasks build one artifact per non-empty OOD role over all of its parts, in split order: `analysis/material_ood/<dataset>` covers the material-OOD parts and `analysis/parameter_ood/<dataset>` the training Dataset's parameter-OOD cases; an empty role gets no folder. One material-OOD source gives `analysis/material_ood/<dataset>/artifacts.parquet` under its Dataset ID; several give one combined table `analysis/material_ood/combined_ood__<digest>/artifacts.parquet`, the digest taken over the ordered Dataset IDs, not one folder per Dataset. Every row names its `source_dataset_id`, and the provenance `dataset` block binds every part (`source_dataset_ids`, `source_identities`, `case_ids_by_source`, `membership_digests`). Before any case is predicted, each part's current Dataset identity is checked against the identity saved in the split; a changed part is refused. Optuna trial runs get no artifacts in either task. Steady-flow artifacts use stride 1.

### Single-case debug artifacts

```bash
./scripts/slurm_ml.sh --mode gpu --partition gpu --cpus-per-task 4 \
  --mem 32G --time 02:00:00 --gres gpu:rtx6000ada:1 artifacts \
  --run-dir RUN_DIRECTORY --one-case --split id \
  --output-root /workspace/storage/evaluation_debug/DEBUG_NAME \
  --evaluation-spatial-stride 1
```

Repeated `--case-id CASE_ID` selects an exact case set instead of `--one-case`. Bounded output needs one transient run, one `--split id|material_ood|parameter_ood` and a new output root outside the run's `analysis/`; it is never canonical and never published to W&B.

## Transient protocols and metrics

| Protocol | Meaning |
| --- | --- |
| Teacher-forced one-step | Each step starts from the reference state |
| Full autonomous rollout | The model feeds its own prediction from the first state to the end |
| Rolling origins | Autonomous rollouts from early, middle and late reference states |

Fixed horizons are 1, 2, 4, 8, 16, 32, 64 and 128 transitions; `full` uses every available transition. A horizon longer than a case is recorded as unsupported. Every result keeps elapsed physical time and both cumulative and endpoint reductions. The RNO starts every case from its own initial hidden state. For one-step records and rolling origins it carries the state a teacher-forced pass builds over the case's reference transitions up to that step, as in training; the full autonomous rollout carries only its own state from the first transition (`recurrent_state_policy: teacher_forced_reference_prefix`).

The central metric is `normalized_drying_group_macro_rmse` with the checkpoint's Train-only scaling and weights `T = 1/3`, `phi = 1/3`, `w_surf = 1/6`, `w_int = 1/6`, accumulated over the whole Dataset before the root. Reports add normalized and physical channel errors, total grain moisture, masked bulk dry- and wet-basis moisture (trapezoidal weighting as in Generation), target status, plausibility and stability diagnostics, and the number of contributing cases.

Only the full autonomous record carries the reference drying-target evidence from Generation, including the irregular exact-stop time. Final-gap errors pair reference and prediction only at the same physical time.

<details>
<summary><strong>Prediction validity</strong></summary>

Both tasks follow one rule. Values are never clamped, filtered or imputed; validity documents keep per-channel counts (finite, non-finite, NaN, +inf, −inf) and the first invalid coordinate. Metrics that need complete finite support report required, computed, finite and non-finite counts instead of a value; the role's `normalized_group_macro_rmse` is such a metric and is never averaged over the valid cases alone. Whenever the artifact step generates or loads a role with invalid predictions it prints `[WARNING] artifact completed with invalid predictions | valid=… | finite_but_physically_invalid=… | nonfinite=…` and one `[WARNING] invalid prediction case` line per case; the training end line is described in [Training](05_training.md).

Both tasks present an unavailable metric the same way and never raise for it while showing or uploading. Tables show `unavailable (F finite of R required values)` in place of a number; the comparison scoreboard draws no bar and the Pareto and capacity views no point for it, and they name the counts at the tick, in a "Not plotted" note or in the legend. Per-case value plots and quantiles leave out non-finite cases, which the warning above names. W&B receives the support counts and no value. A transient summary group (frame, material, mode, horizon, scope) with any non-finite predicted value reports these counts instead of its error metrics and is never reduced over its finite records alone; its frame then has no pipeline degradation entry.

Steady flow classifies each case as `VALID` or `NONFINITE` from its physical `p`, `u`, `v` prediction. It defines no physical range, so it has no `FINITE_BUT_PHYSICALLY_INVALID` class. A `NONFINITE` case keeps its NPZ arrays unchanged (prediction, error and residual fields may hold NaN or ±inf); its Parquet row keeps the element counts and stores NaN for every other metric, so a missing value never reads as zero. The document names the first non-finite channel and grid index `[y, x]`. The writer recomputes the document from the stored prediction, and cache hits and reads bind the checked bytes: admission checks the payload set and the Parquet digest, and each case payload is verified against its recorded digest when the case is first opened (before any byte is parsed); any disagreement is refused.

In transient drying a prediction is `VALID`, `FINITE_BUT_PHYSICALLY_INVALID` or `NONFINITE`. A finite state is physically invalid where `T` leaves [0, 2000] K, `phi` leaves [0, 1.10] or `w_surf` or `w_int` is negative; the 2 × 2 node blocks at the four corners of the evaluation grid are excluded from the physical-invalid count of every channel (non-finite values there still count). The COMSOL reference itself exceeds `phi` = 1 slightly: the evaporation source `max(w_surf − w_eq, 0)` allows no re-wetting and the vapour equation has no condensation sink, so humid air cooling over colder grain is not bounded by saturation. In the transient smoke of 2026-10-30, 2 of 6 cases exceeded 1 in the interior, at most 1.048, only in the first 3 h and never above 1.10; in the smoke of 2026-11-09 none of 10 cases did (maximum 0.936 outside the corner blocks). This is an accepted model limitation and is re-checked on the production data. The reference also has an inlet-corner singularity (`phi` 5.6 at the corner node, −0.64 at its wall neighbour). The persisted validity contract carries these bounds and `excluded_corner_block`. The plausibility diagnostics are separate: they count `phi` outside the strict unit interval [0, 1] without tolerance or corner exclusion, as an informative supersaturation count. An autonomous rollout stops before feeding an invalid state back; other modes and cases continue.

</details>

<details>
<summary><strong>Staged finalization and resume</strong></summary>

Both tasks build a target in one deterministic sibling stage, `.<target>.stage` (for example `analysis/.id.stage`). A failed attempt keeps it; `--rebuild` clears it. A stage that already carries its completion marker is not generated again: the next attempt writes the timing block, admits the stage with the same role validator as a cache hit and publishes it. Artifacts record the analysis digest of their producer (`producer.analysis_sha256`); a complete target or complete stage with another analysis digest than the running code is refused with the `--rebuild` hint, while reading an artifact never compares it with the running code ([Code identity](07_reproducibility_and_validation.md#code-identity)). An incomplete stage resumes per case in the transient task, where each completed case manifest binds role, checkpoint, normalizer, protocol, grid, summary, NPZ digest, timing and validity, so a restart reconstructs reused rows from their NPZ and computes only missing cases. The resume descriptor binds the analysis digest and the timing hardware, so a partial stage started with another analysis digest or on other hardware is refused with the `--rebuild` hint instead of mixing cases computed by different analysis code or timed on different devices. A complete bounded (scoped) stage is reused only when its persisted `selected_case_ids` equal the request's cases and its analysis digest equals the running one, otherwise it is refused and its directory must be deleted (bounded generation takes no `--rebuild`). The roles of one run always share one analysis producer: no build path generates a missing role beside a published role of other analysis code without `--rebuild`, and the refusal comes before any inference. One run-wide lock serializes the artifact builds of a run, so two concurrent builders cannot both pass this check. The artifact preparation after training reads existing roles as evidence; where it may replace incompatible artifacts, it rebuilds such an older role first and then generates the missing one. An incomplete steady stage is generated again as a whole target, because its batched inference and runtime comparison are measured over the whole target in one warmed session. Finalization reconciles membership, writes provenance, timing and summaries, publishes the marker and removes the private resume metadata.

Publication renames the complete stage to the target. A new target appears in that one atomic rename. Replacing an existing target, after `--rebuild`, takes two renames: the old target moves to `.<target>.backup.<random>`, the stage takes the target name, and the backup is then deleted; if the second rename fails, the old target is moved back. Between the two renames the target is briefly absent. A reader at that moment fails with a missing-artifact error and never sees a mix of old and new content. After a hard crash in exactly that window the target is missing, the completed stage is still in place and the previous artifact lies under its backup name: running the artifact command again without `--rebuild` publishes the stage, and the backup directory is deleted by hand. An atomic exchange (`renameat2` with `RENAME_EXCHANGE`) is not used, because its support on the filesystem through which the compute nodes mount `storage/` is unverified.

</details>

## Comparisons

The comparison notebook selects an ordered set of saved runs with a shared material set and Dataset. Runs must agree on task, Dataset membership, objective, evaluation protocol and artifact grid, and their artifacts must carry the same `producer.analysis_sha256`; frames of different or missing analysis producers are refused (steady flow and transient sessions alike). Architecture, seed and stage stay separate visible attributes. The summary ranks by equal-material mean cumulative full-autonomous `normalized_drying_group_macro_rmse`.

<details>
<summary><strong>Controlled training-stage comparison</strong></summary>

A matched set (A0/B/A1) is admitted when it shares the exact A0, architecture, Dataset, split, scaling, initial checkpoint and B receipt. B versus A1 is the protocol comparison at matched compute; B versus A0 includes the additional post-handoff training. The x-axis keeps the admitted clock.

</details>

<details>
<summary><strong>Airflow-to-drying pipeline</strong></summary>

The pipeline view compares three conditions: A uses COMSOL airflow and COMSOL drying (reference), B the drying surrogate on COMSOL airflow, C the drying surrogate on airflow predicted by a steady-flow operator. It reports B/A (drying surrogate error), C/A (complete pipeline error) and C/B (airflow substitution). C needs a compatible steady-flow checkpoint, normalizer, profile and case mapping; without them the view states the reason.

</details>

## Views

Steady-flow views cover overview and scoreboards, global and spatial error, architecture and capacity, error decomposition, physical consistency, spectral fidelity, metadata sensitivity, sample inspection and outliers. Physical-consistency residuals of every steady run use the same physical finite-difference derivatives, independent of the derivative a run was trained with, so runs share one yardstick. They are the dimensionless training terms (see [Training](05_training.md#steady-flow)): momentum residual maps divided by Δp/L_y, the cross-section flux balance as a per-case value and a mean profile over the height, and the pressure-boundary mismatch divided by Δp².

Transient views reuse the same workspace with case, physical-time, channel and scale controls:

| Figure | View |
| --- | --- |
| `transient_state_maps` | Sample viewer with case, channel, physical time and scale lock |
| `transient_central_error_time` | Error over physical time |
| `transient_horizon_error` | Temporal rollout |
| `transient_endpoint_cumulative` | Global error |
| `transient_target_time` | Temporal rollout with censored and unavailable counts |
| `transient_pipeline_degradation` | Runtime, with the airflow pipeline evidence |
| `transient_timing_distributions`, `transient_timing_speedups` | Runtime with the declared measurement boundary |
| `transient_accuracy_inference_time`, `transient_accuracy_speedup` | Runtime trade-off, paired by case |
| `transient_training_performance_compute` | Controlled training-stage comparison |

Aggregate maps and time summaries use only cases that have the selected exact physical time and show the count `n(t)`; bands need at least two contributors. Labels, units, Celsius conversion, colours and orientation come from the presentation layer shared with EDA.

<details>
<summary><strong>Error modes and display stride</strong></summary>

Steady field-error maps offer **MAE**, **Relative [%]** (absolute error over the case's reference RMS plus `1e-12`, times 100) and **Normalized absolute error**. Transient maps offer signed, absolute and normalized absolute error; normalized absolute error is the default.

Normalized absolute error is `abs(prediction - reference) / s` with the saved Train-only physical-state standard deviation `s` (`output_standard_deviations` for steady flow, `state_std` for transient). Its weighted mean is `sum(weight * error) / sum(weight)`. It is a diagnostic, independent of Celsius/Kelvin offsets, and requires the same field, unit, Dataset and scaler across compared models.

**Display stride** (`Full`, `2`, `4`, `Auto`) only thins the drawn map after all errors, statistics and colour bounds are computed on the full grid. `Auto` uses stride 2 above a longest axis of 400 points and 4 above 800.

</details>

<details>
<summary><strong>Field-map colour scales</strong></summary>

`src.analysis.presentation.field_maps` defines all scientific field maps: a continuous field uses 11 colours and 12 strictly increasing boundaries through one `BoundaryNorm`, shared by contours, images and colourbar ticks. Scale kinds are linear positive, linear signed (neutral interval around zero), log positive (geometric boundaries), absolute, signed and relative error, categorical and constant. Locked comparisons share all 12 boundaries; values outside them set the colourbar extension. Normalized error locks to 0–1 with an extension above 1.

</details>

## Component checks

`notebooks/03_component_checks.ipynb` runs before training and gives cheap, training-free priors for small model components, next to the physics checks of the reference fields; the service is `src.analysis.component_checks`. It reads only published Datasets and their admitted source `case.h5` files, trains nothing and writes nothing. Every check reads one membership of the Dataset partition, `train` (the notebook default) or `validation`, filtered before any case limit; ID-test and OOD roles are refused, and every table names its Dataset, membership and cases. A check yields a prior for an ablation or a candidate range, never a decision.

| Task | Checks |
| --- | --- |
| Steady flow | Derivative backends (`physical`, `spectral` with Fourier continuation) against exact derivatives of manufactured non-periodic fields per edge band, crop and precision; the physics-loss cost per backend and device (forward plus backward at the training batch, node-specific, not persisted); the dimensionless training terms of the COMSOL solution per backend, interior crop and precision; the momentum split into pressure, viscous and drag parts, with the ⅔(∇·u) trace part shown separately; the pointwise divergence variants removed from the loss; residuals by distance to the edges; finite-difference against spectral derivatives; the floor and the 1 % smooth-error sensitivity per crop; the ratio of each term under controlled smooth, noisy, scaling and offset errors; the truncation error of each output field under the Fourier modes a spectral layer keeps, with domain padding 0 and 0.1 |
| Transient drying | The same truncation floors for states and increments at the training stride; the bilinear full-grid floor of the stride views 1, 2 and 4 under the runtime's endpoint rule; scaled-increment statistics of the transient scaling fitted on the membership (quantiles, exact zeros, fraction above the Huber β, equal-weight loss shares) and the stored time steps; grain water, pore-gas vapour, evaporation and advective boundary fluxes from the stored fields against the COMSOL global series; closure of the grain, gas and total water balance; pointwise sorption residuals; inlet rows against the schedule; reference values outside the evaluation validity rule; which balance terms the Dataset and artifacts do not persist |

A summary table applies the proposed criteria (named constants in `component_checks_summary.py`, open while the component-check decision criteria are open) to the results computed so far. Fourier floors are priors only: the networks truncate lifted features, not the raw fields. Analytic operator checks stay in `tests/domain/`. The notebook needs a compute-node kernel; the physics checks alone took about 35 s and 1.7 GB on the smoke Datasets, the new sections are not yet measured.

## Timing and speedup

| Measurement | Boundary |
| --- | --- |
| `drying_no_prepared_rollout_wall_seconds` | Wall interval of a prepared rollout request, cold or warm as recorded; excludes Dataset, checkpoint, context and request construction |
| Model-only time | Model calls only; excludes validation, transfer, scaling and reconstruction |
| COMSOL `Solution time` | Steady and transient study phases parsed from the solver log |
| COMSOL process and case execution | Broader wall-clock boundaries of the case job |

Benchmarks record cold time, warm-up, raw warmed repetitions, payload identity, precision, batch size, hardware and software versions. Speedups that combine different boundaries are labelled component-composed and aggregate paired cases as the ratio of summed reference time to summed surrogate time. Steady-flow inference time is synchronized, warmed and batch-amortized. Missing components stay unavailable with their reason.

Every artifact of both tasks carries a required `timing` block in `artifact_provenance.json`, outside scientific identity: the runtime comparison (`report`), the transient component availability and the operational snapshot of the build (phases, forward calls, reused cases). It is written into the stage before validation, so a timing failure publishes nothing; an artifact without it is refused with the `--rebuild` hint. The `producer` block records `analysis_sha256`, the digest of the analysis projection that built the artifact. Runtime provenance records device, PyTorch version, precision, inference batch size and the executing frozen source tree (`source_tree_id`, provenance only) for both tasks. Transient role admission binds every timing report case to its component evidence (the generator's complete field set with a finite non-negative materialization duration) and to its sequence record identity (PT payload identity), so the W&B upload cannot fail on timing evidence that admission accepted.

## Export and tracking

Selecting a view creates no file. **Export PDF** saves the selected figure into the repository's `notebooks/` folder, named by plot and UTC timestamp; an existing file is never overwritten. The export is refused, with a message in the panel status, in a frozen source tree or when `notebooks/` is missing or a symlink. `PreparedEvaluationWorkflow.render_report(output_dir)` renders the curated PNG report into a caller-chosen directory.

With `tracking.wandb` enabled and `upload.evaluation_artifacts: true`, the artifact step validates every published role, resumes the training run in W&B (same project, group and job type) and uploads the `artifact_provenance.json` of the ID, material-OOD and parameter-OOD roles for both tasks. Both tasks then add one bounded summary into the run summary, next to the training entries `data/train_dataset` and `data/ood_datasets`: aggregate metrics, availability and counts, timing medians and the identities of Dataset, checkpoint, model, hardware and precision. Transient runs add ratio-of-sums speedups and the input profile, and a summary group whose error metrics are unavailable carries its support counts instead; steady-flow runs carry per role the valid and non-finite case counts, the finite-support counts of `normalized_group_macro_rmse` and, only when it is available, its value and the per-field normalized and physical RMSE. Steady-flow runs finally add the curated media. Sequence arrays, raw data, per-case timings and checkpoints stay local.

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
