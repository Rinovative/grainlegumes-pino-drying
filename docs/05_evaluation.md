# Evaluation

Evaluation works on saved evidence. An explicit artifact step writes reference and predicted fields for a completed run; the notebooks discover runs, load validated artifacts and show compatible views. Opening a notebook never trains, runs inference or builds missing artifacts.

[Documentation index](README.md)

## Notebooks

| Notebook | Use |
| --- | --- |
| `notebooks/01_eda_generation_inputs.ipynb` | Prepared Generation inputs before COMSOL: parameters, fields, schedules, design coverage |
| `notebooks/02_eda_datasets.ipynb` | Completed cases and Datasets of both tasks: states, static fields, schedules, target attainment, solver timing |
| `notebooks/03_training_pipeline.ipynb` | Training configurations, data roles, objectives, stage dependencies and saved runs (inspection only) |
| `notebooks/04_eval_single_model.ipynb` | One saved run |
| `notebooks/05_eval_comparison_models.ipynb` | Two or more compatible saved runs |
| `notebooks/06_physics_diagnostics.ipynb` | Physics residuals, derivative backends and water balances of the COMSOL reference fields ([Physics diagnostics](#physics-diagnostics)) |

Scientific cells need a compute-node kernel ([Setup and execution](01_setup_and_execution.md#notebooks-on-compute-nodes)). The notebooks resolve `STORAGE_ROOT`, scan `03_experiments/` once and build their selectors from persisted run records. Runs without usable artifacts stay visible with the exact command that creates them.

## Build artifacts

A completed training run builds its artifacts by default. To build or validate them explicitly:

```bash
./scripts/slurm_ml.sh --mode gpu --partition gpu --cpus-per-task 4 \
  --mem 32G --time 02:00:00 --gres gpu:rtx6000ada:1 artifacts \
  --run-dir RUN_DIRECTORY --evaluation-spatial-stride 1
```

Instead of `--run-dir` (repeatable), `--task TASK` (the task's folder below `storage/03_experiments/`) or `--runs-root DIR` selects all evaluable runs below a discovery root, skipping Optuna trial runs; `--run-name` narrows the selection. Only completed runs are evaluated: a run that ended `budget_unmet` or `interrupted` is refused until it is resumed to completion. Before any case is predicted, the service admits checkpoint, scaling artifact, Dataset identity and the requested grid for the architecture. Incomplete or older artifacts are rejected; `--rebuild` recreates the exact target.

In both tasks every per-case NPZ is named by the Dataset package case ID, for example `lentil__case_0003__fe83e052fbf2.npz`: material, the campaign-local case number as in `01_generation/…/lentil/case_0003`, and a short digest that keeps cases of different campaigns apart. The `case_index` stored inside steady files stays the one-based Dataset position.

### What a transient artifact stores

Per ID case one NPZ bundle (named `<package case id>.npz`) holds the reference states over every stored hour and, per distinct prediction chain (one-step, rolling and autonomous origins; about 170 chains per case), the predicted states over every hour after the chain's origin plus the computed-prefix flag. The first state of a chain equals the reference state at its origin and is taken from there on load; chains that are bit-identical, such as the full autonomous rollout and the rolling origin at hour 0, are stored once and shared by their records. The writer checks before publishing that every record rebuilds bit for bit from the stored arrays. Metrics, validity documents and summaries are computed from in-memory predictions before writing and stored in the Parquet index. Predicted and reference increments (state differences) and the non-finite and physical-invalid masks (recomputed with the current validity rule) are derived on load. The raw scaled network outputs are not stored; regenerate them from the checkpoint if needed. This takes about 0.95 GB per ID case at 168 hours instead of 4.1 GB, and every hour remains available for browsing and animations.
### Evaluation grid

`--evaluation-spatial-stride N` (default `1`) selects an endpoint-preserving grid from the canonical 251 × 401 source grid, independently of the training stride. With the maintained training stride 2:

| Grid | Shape |
| --- | --- |
| Canonical source grid | 251 × 401 |
| Training and scaling grid (stride 2) | 126 × 201 |
| Artifact grid, stride 1 | 251 × 401; model input `[B, 28, 251, 401]`, output `[B, 4, 251, 401]` |
| Artifact grid, stride 2 | 126 × 201 for every tensor |

Channel-wise scaling broadcasts to the requested grid. Each architecture must support the grid: FNO and RNO need their retained modes to fit the axes, U-NO every resampling level; a different resolution additionally needs a passing exact-output probe of the loaded checkpoint. Reference and prediction share one grid digest over indices, coordinates, mask, field order and extent.

| Task | Stride 1 | Stride N > 1 |
| --- | --- | --- |
| Both tasks | `analysis/id`, `analysis/additional_evaluation/<dataset>` | `analysis/grid_sN/id`, `analysis/grid_sN/additional_evaluation/<dataset>` |

`analysis/id` holds the ID-test membership; `analysis/additional_evaluation/<dataset>` the combined declared holdouts. Steady-flow artifacts use stride 1.

### Single-case debug artifacts

```bash
./scripts/slurm_ml.sh --mode gpu --partition gpu --cpus-per-task 4 \
  --mem 32G --time 02:00:00 --gres gpu:rtx6000ada:1 artifacts \
  --run-dir RUN_DIRECTORY --one-case --split id \
  --output-root /workspace/storage/evaluation_debug/DEBUG_NAME \
  --evaluation-spatial-stride 1
```

Repeated `--case-id CASE_ID` selects an exact case set instead of `--one-case`. Bounded output needs one transient run, one `--split id|additional_evaluation` and a new output root outside the run's `analysis/`; it is never canonical and never published to W&B.

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

A prediction is `VALID`, `FINITE_BUT_PHYSICALLY_INVALID` or `NONFINITE`. `phi` is physically invalid only outside [0, 1.10]; the 2 × 2 node blocks at the four corners of the evaluation grid are excluded from the physical-invalid count (non-finite values there still count). The COMSOL reference itself exceeds `phi` = 1 slightly: the evaporation source `max(w_surf − w_eq, 0)` allows no re-wetting and the vapour equation has no condensation sink, so humid air cooling over colder grain is not bounded by saturation. In the transient smoke of 2026-10-30, 2 of 6 cases exceeded 1 in the interior, at most 1.048, only in the first 3 h and never above 1.10; this is an accepted model limitation and is re-checked on the production data. The reference also has an inlet-corner singularity (`phi` 5.6 at the corner node, −0.64 at its wall neighbour). The persisted validity contract carries the range and `excluded_corner_block`. The plausibility diagnostics are separate: they count `phi` outside the strict unit interval [0, 1] without tolerance or corner exclusion, as an informative supersaturation count. Validity documents keep per-channel counts and the first invalid coordinate; values are never clamped, filtered or imputed. An autonomous rollout stops before feeding an invalid state back; other modes and cases continue. Metrics that need complete finite support report required, computed, finite and non-finite counts instead of a value.

</details>

<details>
<summary><strong>Staged finalization and resume</strong></summary>

Transient case work is staged in a target-specific `.transient-resume` sibling. Each completed case manifest binds role, checkpoint, normalizer, protocol, grid, summary, NPZ digest, timing and validity. A restart reconstructs reused rows from their NPZ and computes only missing cases; a complete stage runs finalization alone. Finalization reconciles membership, writes provenance, timing and summaries, publishes the marker and removes the private resume metadata.

</details>

## Comparisons

The comparison notebook selects an ordered set of saved runs with a shared material set and Dataset. Runs must agree on task, Dataset membership, objective, evaluation protocol and artifact grid. Architecture, seed and stage stay separate visible attributes. The summary ranks by equal-material mean cumulative full-autonomous `normalized_drying_group_macro_rmse`.

<details>
<summary><strong>Controlled training-stage comparison</strong></summary>

A matched set (A0/B/A1 or A0/A+/B) is admitted when it shares the exact A0, architecture, Dataset, split, scaling, initial checkpoint and B receipt. B versus A1 (or A+) is the protocol comparison at matched compute; B versus A0 includes the additional post-handoff training. The x-axis keeps the admitted clock.

</details>

<details>
<summary><strong>Airflow-to-drying pipeline</strong></summary>

The pipeline view compares three conditions: A uses COMSOL airflow and COMSOL drying (reference), B the drying surrogate on COMSOL airflow, C the drying surrogate on airflow predicted by a steady-flow operator. It reports B/A (drying surrogate error), C/A (complete pipeline error) and C/B (airflow substitution). C needs a compatible steady-flow checkpoint, normalizer, profile and case mapping; without them the view states the reason.

</details>

## Views

Steady-flow views cover overview and scoreboards, global and spatial error, architecture and capacity, error decomposition, physical consistency, spectral fidelity, metadata sensitivity, sample inspection and outliers. Physical-consistency residuals of every steady run use the same physical finite-difference derivatives, independent of the derivative a run was trained with, so runs share one yardstick. They are the dimensionless training terms (see [Training](04_training.md#steady-flow)): momentum residual maps divided by Δp/L_y, the cross-section flux balance as a per-case value and a mean profile over the height, and the pressure-boundary mismatch divided by Δp².

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

## Physics diagnostics

`notebooks/06_physics_diagnostics.ipynb` checks the physics implementations on the published reference fields themselves, without models; the service is `src.analysis.physics_diagnostics`. It reads only published Datasets and their admitted source `case.h5` files and writes nothing.

| Task | Checks |
| --- | --- |
| Steady flow | The dimensionless training terms of the COMSOL solution per derivative backend (`physical`, `spectral` with `fourier_continuation`, `spectral` without extension), interior crop and precision; the momentum split into pressure, viscous and drag parts plus the compressible ⅔(∇·u) trace variant; the pointwise divergence variants removed from the loss; residuals by distance to the edges; finite-difference against spectral derivatives; and the ratio of each term under controlled smooth, noisy, scaling and offset errors to its reference value |
| Transient drying | Grain water, pore-gas vapour, evaporation and advective boundary fluxes from the stored fields against the COMSOL global series; closure of the grain, gas and total water balance between stored times; pointwise residuals of the two sorption compartments; inlet rows against the schedule; reference values outside the evaluation validity rule; which balance terms the Dataset and artifacts do not persist |

Analytic operator checks stay in `tests/domain/`. The notebook needs a compute-node kernel; the smoke Datasets take about 35 s and 1.7 GB.

## Timing and speedup

| Measurement | Boundary |
| --- | --- |
| `drying_no_prepared_rollout_wall_seconds` | Wall interval of a prepared rollout request, cold or warm as recorded; excludes Dataset, checkpoint, context and request construction |
| Model-only time | Model calls only; excludes validation, transfer, scaling and reconstruction |
| COMSOL `Solution time` | Steady and transient study phases parsed from the solver log |
| COMSOL process and case execution | Broader wall-clock boundaries of the case job |

Benchmarks record cold time, warm-up, raw warmed repetitions, backend, payload identity, precision, batch size, hardware and software versions. Speedups that combine different boundaries are labelled component-composed and aggregate paired cases as the ratio of summed reference time to summed surrogate time. Steady-flow inference time is synchronized, warmed and batch-amortized. Missing components stay unavailable with their reason.

## Export and tracking

Selecting a view creates no file. **Export PDF** saves the selected figure; the destination defaults to the experiment's `comparison/` folder and must be chosen explicitly for cross-experiment selections (`comparison_output_dir` sets the initial value). File names carry the plot name and a timestamp. `PreparedEvaluationWorkflow.render_report(output_dir)` renders the curated PNG report into a caller-chosen directory.

With `tracking.wandb` enabled and `upload.evaluation_artifacts: true`, the artifact step publishes one bounded summary per run to W&B: aggregate metrics, availability and counts, timing medians, ratio-of-sums speedups and the identities of Dataset, checkpoint, profile, model, backend, hardware and precision. Sequence arrays, raw data, per-case timings and checkpoints stay local.

## Inference

A completed transient run loads without opening its Dataset:

```python
from src import learning

transient = learning.inference.transient
context = transient.load_transient_inference_context(run_dir=run_dir, device="cuda")
```

`predict_transient_step` takes one physical interval; `rollout_transient_autonomous` takes a validated interval sequence and feeds its own reconstructed states back. Both support FNO, U-NO and RNO. Inputs are both boundary endpoints, startup support, exact times and `dt`, and the COMSOL reference airflow or a compatible external airflow. Without an explicit evaluation shape the context uses the checkpoint's training grid.

<details>
<summary><strong>Compute device and memory</strong></summary>

`evaluation_compute_device` (`auto`, `cpu`, `cuda`, `cuda:N`) applies to the aggregate error reductions; artifact building has its own `device_policy`. Discovery, tables, plotting and single-case maps run on CPU. `auto` selects CUDA for batches of at least 3.2 million aligned values; an explicit CUDA request checks the visible device and never falls back. The threshold comes from a probe on one RTX 6000 Ada with the 251 × 401 grid and four float64 channels (16 cases: 58.9 ms CPU, 20.0 ms GPU).

Notebook admission verifies artifact, manifest, frame, provenance, output and payload digests. Each file (payloads and run checkpoints) is hashed at most once per process; the digest is reused while its path, size, modification time and inode are unchanged. Sequence arrays are decompressed only when their case is requested and kept in a bounded LRU cache; aggregate views load all admitted records. Closing a workspace clears its caches.

</details>
