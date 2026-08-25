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
spatial stride, and the ID and near-family OOD artifact states.

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
Evaluation protocol, and spatial stride fail closed.

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
run:

```bash
./scripts/docker_job.sh --queue-gpu auto artifacts --run-dir <run-dir>
```

The run must be terminal and evaluable. The task-aware service selects the
persisted checkpoint and Dataset identities, validates any reusable artifact,
and atomically publishes new ID and saved OOD artifacts below the run's
canonical `analysis` directory. Rebuilds require the explicit `--rebuild`
option. W&B publication remains controlled by the resolved run configuration;
disabled mode performs no observer work.

For a disposable transient UI/debug artifact, generate one saved case into a
new noncanonical directory:

```bash
./scripts/docker_job.sh --queue-gpu auto artifacts \
  --run-dir <run-dir> \
  --one-case \
  --split id \
  --output-root /workspace/storage/evaluation_debug/<debug-name>
```

Use repeated `--case-id <saved-case-id>` arguments instead of `--one-case`
for an exact bounded case set. Bounded generation requires exactly one
`--run-dir`, one explicit `--split id|ood`, and a new `--output-root`.
The output cannot be placed inside the run's canonical analysis directory,
cannot replace an existing path, and is marked ineligible for canonical
publication. It never publishes to W&B.

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
and availability evidence. Missing values are not inferred from filenames,
scheduler logs, solver logs, or W&B.

Repository documentation does not freeze the current run inventory or current
research hyperparameters. A real report requires at least one completed,
compatible persisted run with canonical Evaluation artifacts. When storage has
only running runs or missing artifacts, the notebooks remain usable for
discovery and show the commands needed to create the missing evidence after the
run becomes evaluable.
