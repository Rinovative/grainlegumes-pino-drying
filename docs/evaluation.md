# Evaluation

Evaluation is a post-training, artifact-backed workflow for admitted
`steady_flow` and `transient_drying` runs. The maintained notebooks discover
persisted runs from current metadata, expose only compatible controls, and load
validated Evaluation artifacts. They do not run inference, generate artifacts,
or contact W&B.

## Notebook entry points

Use:

- `notebooks/eval_single_model.ipynb` for one exact persisted run
- `notebooks/eval_comparison_models.ipynb` for two or more compatible runs

Both notebooks resolve `STORAGE_ROOT`, scan `03_experiments` once, and build
their selectors from persisted run records rather than directory names or
hardcoded inventories. The overview shows scientific run identity, task, model,
stage, seed, Dataset identity, selected checkpoint, lifecycle state, protocol,
artifact Evaluation spatial stride, and the ID and near-family OOD artifact states.

Runs without a usable artifact remain visible. Their status includes the exact
full artifact command and, for transient runs, disposable one-case commands.
Selecting such a run never falls back to model inference. Incomplete,
incompatible, or corrupt artifacts are rejected by their owning validators.

The shared top-level controls follow the EDA vocabulary:

- Experiment and exact run or stage
- Evaluation data: ID, Near-family OOD, or ID + Near-family OOD
- Analysis scope: Aggregate or Single case

Comparison mode additionally selects an ordered set of exact run identities.
Task mixing and incompatible Dataset membership, canonical objective, complete
Evaluation protocol, and exact artifact Evaluation-grid identity fail closed.

## Task-aware analysis

Steady-flow artifacts keep the existing steady Evaluation sections and plots:
summary and comparison scoreboards, global and spatial error behavior,
model-family capacity, error sensitivity, physical consistency, spectral
fidelity, metadata sensitivity, sample inspection, and outlier analysis.
Authoritative steady runtime and speed-up are added only when a validated timing
sidecar is bound to the exact run, checkpoint, Dataset membership, split, and
cases. A missing or invalid optional sidecar removes that timing view without
invalidating the scientific artifact; notebook wall-clock measurements are never
substituted.

Transient Aggregate scope provides:

- Overview: aggregate metrics, per-state error, worst-case error, and paired ID
  versus near-family OOD generalization when both roles are compatible
- Rollout: error by physical horizon and endpoint versus cumulative reductions
- Outcomes: target-time and censoring evidence plus Airflow-to-Drying pipeline
  degradation
- Timing: speedup definitions, component distributions, accuracy versus
  inference time, and accuracy versus speedup

Transient Single case scope provides process views with EDA-style linked
controls:

- a numeric `Case:` selector with previous and next buttons
- protocol and requested-horizon selectors
- the shared `Time [h]:` navigator
- state-channel checkboxes
- trajectory fields including `T`, `phi`, `w_surf`, `w_int`, and derived
  granular moisture `w_gr`
- selected-time reference, prediction, signed-error, and absolute-error maps
- physical-time trajectories with the 10--90% spatial envelope
- central error versus physical time

Plot labels, units, Celsius conversion, channel colors, field orientation, and
histogram conventions come from the same presentation owners used by EDA.
Transient comparisons preserve teacher-forced, full-autonomous, and rolling
origin semantics; unsupported horizons and absent pipeline evidence remain
explicitly unavailable.

## Artifact generation

Generate or validate canonical artifacts outside the notebooks. For one exact
transient run on the canonical original grid, use:

```bash
./scripts/docker_job.sh --queue-gpu auto artifacts \
  --run-dir <run-dir> \
  --evaluation-spatial-stride 1
```

`--evaluation-spatial-stride` accepts one canonical positive integer and
defaults to `1` when omitted. It selects the artifact Evaluation grid from the
canonical source grid; it never inherits the checkpoint's Training stride.
A lower-resolution artifact must be requested explicitly, for example:

```bash
./scripts/docker_job.sh --queue-gpu auto artifacts \
  --run-dir <run-dir> \
  --evaluation-spatial-stride 2
```

The stride must retain both physical endpoints on both axes. The same exact
source indices materialize dynamic fields, static fields, coordinates, masks,
conditioning, targets, model inputs, references, and predictions. No
interpolation, upsampling, or fallback to the Training grid is allowed. The
operator argument is retained verbatim by the queue descriptor and worker
command, then resolved once by the artifact service. Preflight logs one concise
summary:

```text
[ARTIFACT] evaluation_spatial_stride=<stride>
[ARTIFACT] source_shape=<Y, X>
[ARTIFACT] evaluation_shape=<Y, X>
```

The run must be terminal and evaluable. Before case inference, the task-aware
service admits the checkpoint, scaling artifact, Dataset/package identity, and
requested architecture/grid capability. FNO, U-NO, and RNO requests proceed
only after their exact mode/resampling constraints pass and a cross-resolution
request completes an exact-output-shape model probe. Channel-wise scaling may
broadcast to a compatible grid, but shape-bound scaling would be rejected;
artifact generation never refits or mutates scaling.

Canonical stride-one paths remain backward-compatible. Noncanonical grids use
separate sibling variants:

| Evaluation grid | ID artifact | OOD artifact |
| --- | --- | --- |
| stride 1 | `analysis/id` | `analysis/ood/<dataset>` |
| stride N, N > 1 | `analysis/grid_sN/id` | `analysis/grid_sN/ood/<dataset>` |

Locks, rebuilds, cache admission, and publication are target-specific, so one
grid variant cannot overwrite or satisfy another. Transient sequence schema 4
binds every record and role to source shape, Evaluation stride and shape, exact
axis indices, coordinate bytes, physical extent, mask identity, field order,
checkpoint, architecture/scaling decisions, and one common reference/prediction
grid digest. Older transient caches lack the complete current evidence and fail closed; rebuild
the exact desired target with `--rebuild`.

### Transient finalization and diagnostic predictions

Transient case work is staged in one deterministic target-specific
`.transient-resume` sibling. Each completed case manifest binds exact role,
checkpoint, normalizer, protocol, grid, summary, NPZ digest, timing, and validity
evidence. Restart semantically reconstructs every reused row from its NPZ once;
it does not merely trust the manifest digest. A compatible complete stage runs
finalization without model, Dataset, or Generation-source setup, while a partial
stage computes only missing cases. Conflicts fail closed and identify the exact
target-only `--rebuild` action.

Finalization has an explicit reported phase and owns membership reconciliation,
parent provenance, final timing and prediction summaries, marker publication,
and safe resume-metadata cleanup. A retry interrupted after marker publication
cleans only the exact private resume metadata before publishing the canonical
artifact.

A structurally successful model call may yield `VALID`,
`FINITE_BUT_PHYSICALLY_INVALID`, or `NONFINITE` prediction evidence. Schema 4
preserves raw scaled output, decoded increment, reconstructed state, computed
prefix, NaN uncomputed-tail sentinels, invalid masks, per-channel IEEE and
physical-invalid counts, and first-invalid coordinates. It never clamps,
filters, imputes, or retries a value. Autonomous inference stops before feeding
an invalid state back to the model; independent reference-started modes and
later cases continue.

References, source conditioning, scaler/checkpoint/grid identity, shapes, and
serialization remain strict. Finite but physically invalid predictions retain
raw error metrics plus separate physical diagnostics. Metrics requiring complete
finite support become explicitly unavailable with required, computed, finite,
and non-finite counts; no invalid cell is omitted. Non-finite target evidence is
also explicitly unavailable. A diagnostic case counts as processed, and final
warnings report status counts, affected cases and channels, and first-invalid
step/time. Early-stopped diagnostic clocks are not published as complete-rollout
speedups.

For a disposable transient UI/debug artifact, generate one saved case into a
new noncanonical directory:

```bash
./scripts/docker_job.sh --queue-gpu auto artifacts \
  --run-dir <run-dir> \
  --one-case \
  --split id \
  --output-root /workspace/storage/evaluation_debug/<debug-name> \
  --evaluation-spatial-stride 1
```

Use repeated `--case-id <saved-case-id>` arguments instead of `--one-case`
for an exact bounded case set. Bounded generation requires exactly one
`--run-dir`, one explicit `--split id|ood`, and a new `--output-root`.
The output cannot be placed inside the run's canonical analysis directory,
cannot replace an existing path, and is marked ineligible for canonical
publication. It never publishes to W&B. Steady-flow artifacts accept only the
canonical value `--evaluation-spatial-stride 1`.

## Shared scientific field maps

`src.analysis.presentation.field_maps` owns the common scientific field-level
and colorbar contract. A nonconstant continuous field uses exactly 11 visible
resampled colors and exactly 12 strictly increasing boundaries through one
`BoundaryNorm`. The same definition supplies explicit `contourf` levels,
`pcolormesh`/`imshow` normalization, exact colorbar ticks, and an explicit
`extend` policy. Scientific arrays remain continuous, caller-owned, and
unmodified.

The owner distinguishes linear-positive, linear-signed, log-positive, absolute
error, signed error, relative error, categorical, and constant semantics.
Signed scales have one neutral center interval around zero; log-positive scales
use geometric boundaries; categorical and constant fields retain their actual
cardinality instead of fabricating 11 intervals. Comparison locking preserves
all 12 exact boundaries rather than only extrema. Values outside locked bounds
set the truthful colorbar extension while retaining the same 11 interval
colors. Identical immutable scale state uses a bounded in-process cache keyed
by exact field, unit, display transformation, comparison scope, lock state,
bound policy, boundaries, extension, and colormap state. Each outward definition
receives isolated Matplotlib objects and bytes-backed read-only boundaries/ticks,
so caller mutation cannot contaminate another plot. There is no persistent
visualization cache and no color choice enters Dataset scientific identity.

Evaluation and EDA plot-owner migrations are intentionally deferred to their
separate follow-up tasks. New migrations must consume this shared definition
rather than create another discrete-level helper.

## Loading and memory behavior

Transient artifacts retain immutable per-sequence payloads, a validated
manifest, frame metadata, and provenance. Notebook admission verifies artifact,
manifest, frame, provenance, output, and individual payload digests before
analysis. Sequence arrays are then decompressed only when their selected case is
requested. The session keeps a bounded least-recently-used payload cache; case
identity and selector construction do not load every array.

Aggregate analysis intentionally requests all admitted records because its
metrics span the Dataset. Single-case navigation loads only that case's
protocol and horizon records. Closing or replacing a workspace clears its
session caches and closes artifact indexes.

## Provenance and limitations

Evaluation reports retain exact run, checkpoint, Dataset, split membership,
input profile, scaling, model, protocol, horizon, origin, physical-time, timing,
exact spatial representation, and availability evidence. Missing values are not inferred from filenames,
scheduler logs, solver logs, or W&B.

Repository documentation does not freeze the current run inventory or current
research hyperparameters. A real report requires at least one completed,
compatible persisted run with canonical Evaluation artifacts. When storage has
only running runs or missing artifacts, the notebooks remain usable for
discovery and show the commands needed to create the missing evidence after the
run becomes evaluable.
