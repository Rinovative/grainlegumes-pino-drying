# Transient drying training

## Scope

The `transient_drying` task is trainable with FNO, U-NO, and the official
`neuralop.models.RNO`. Training consumes immutable Dataset packages, writes
independent run bundles under `03_experiments/transient_drying`, and remains
usable when the original CPU Generation workspace has been deleted.

This workflow owns Training, Optuna, checkpoint/resume, W&B telemetry, matched
compute, inference loading, and task-aware post-training artifacts.
Completed-output transient EDA is implemented against admitted Generation
evidence and validated Dataset runtime items. Transient Evaluation uses a
sequence artifact and the current inference, scaling, Dataset, report, and
artifact-service owners; it does not route transient runs through steady-only
schemas.

## Maintained experiment plans

Normal experiments use one architecture-first YAML per model:

- `configs/learning/transient_drying/experiments/fno_m64x80_h64_l3__lentil_chickpea__s9.yaml`
- `configs/learning/transient_drying/experiments/uno_m64x64_h32_l7_s1-05-05-1-1-2-2_r0p495__lentil_chickpea__s9.yaml`
- `configs/learning/transient_drying/experiments/rno_m24x24_h16_l3__lentil_chickpea__s9.yaml`

The filename makes the architecture recognizable during scheduling and review.
The resolved YAML remains authoritative; filenames do not replace persisted
configuration, Dataset identity, seeds, or checkpoint hashes.

The maintained selections bind exact Dataset IDs and use the endpoint-preserving
`spatial_stride: 2` view. They set `run.deterministic: false`, so seeds, resolved
configuration, and checkpoint provenance remain exact but executions do not
promise bitwise replay. Their W&B mode is `online`; a nonblank `WANDB_API_KEY`
must be available before epoch 1. Optuna and technical-smoke plans retain their
separate offline and disabled tracking policies.

Each file is an authored two-stage plan. Shared sections define the task, data,
model, loss, optimizer, scaling, and tracking policy. One
`training.stage_schedule` owns the fixed total epoch budget and the explicit
`stage_a_fraction`; the loader deterministically gives Stage B the exact integer
remainder. `training.stage_a` and `training.stage_b` own only stage-specific
sampling, evaluation, accumulation, and curriculum settings. There are no
separate Stage A and Stage B normal config directories.

```text
authored architecture plan
    -> Stage A0 run leaf (name contains stage_a0)
    -> selected best checkpoint and immutable Stage A handoff
    -> fresh Stage B run leaf (name contains stage_b)
```

Stage A and Stage B remain independent persisted experiments. They have separate
resolved configs, histories, checkpoints, lifecycle summaries, and W&B runs.
Stage B loads the complete selected Stage A checkpoint state through the
immutable handoff. The shared model, optimizer, scheduler, AMP scaler, loss,
loader/sampler, and process RNG state continue exactly so A+ and B begin from the
same optimization state. Stage B does not overwrite Stage A and owns fresh
history, global-step accounting, controller progress, and curriculum state.

For FNO and U-NO, Stage A trains one-step increments with reference states.
Official RNO Stage A uses contiguous teacher-forced reference sequences, carries
hidden state inside each sequence, and resets it at independent boundaries.
Stage B is autonomous and self-fed for all three architectures. Its configured
curriculum advances continuously through eligible rollout horizons, normally
`2 -> 4 -> 8 -> 16 -> 32`, without restarting Stage A. Maintained joint plans
are epoch-budgeted: neither stage carries a competing matched-compute stop, and
each must complete its exact derived epoch allocation.

## Preflight and training

Run preflight before allocating any experiment directory:

```bash
python -m src.experiments.cli.cli_config_preflight train \
  configs/learning/transient_drying/experiments/fno_m64x80_h64_l3__lentil_chickpea__s9.yaml
```

Start the complete A0-to-B workflow with the same file:

```bash
python -m src.experiments.cli.cli_train \
  configs/learning/transient_drying/experiments/fno_m64x80_h64_l3__lentil_chickpea__s9.yaml
```

The CLI prints both derived run directories. If the exact Stage A leaf already
contains a valid completed run and immutable handoff while Stage B is absent,
rerunning the plan reuses Stage A read-only and starts Stage B. An incomplete or
incompatible existing leaf is never reopened implicitly.

Resume names one exact derived leaf:

```bash
python -m src.experiments.cli.cli_train CONFIG.yaml --resume /exact/stage_a0/run/leaf
python -m src.experiments.cli.cli_train CONFIG.yaml --resume /exact/stage_b/run/leaf
```

Resuming Stage A completes and validates A before a fresh B is allocated.
Resuming Stage B first validates its exact completed Stage A source and handoff.
Normal resume rules still apply: the saved semantics, split, scaling, checkpoint
identity, and output root must match, and completed runs require a deliberate
epoch extension.

Use `--device` only as an explicit runtime override. `cuda` is strict and never
falls back silently. `--output-root` applies to both derived leaves and does not
change Dataset roots.

## Tensorization and scaling

The only registered input profile is
`canonical_physics_complete_v1`. It contains 28 ordered channels:

- four current state channels: `T`, `phi`, `w_surf`, and `w_int`;
- coordinates, COMSOL airflow, porosity, and spatial material fields;
- both boundary endpoints, ambient and startup-support evidence;
- scalar material parameters.

Optional normalized current time is an additional tensorizer channel when that
policy is selected. Maintained plans currently select no temporal channel.
Coordinates are explicit channels; model positional embedding is therefore
`null`.

The former fixed “reduced” profile and reduced experiment variants do not exist.
There is currently no config key for channel exclusion. If exclusions are needed
later, they must be introduced as a general list-based, identity-bound contract
across TaskSpec, tensorization, scaling, checkpoints, handoffs, and inference,
not as another hard-coded reduced profile.

Scaling statistics are fitted from Train membership only. Absolute states are
deduplicated by state identity, and increment scaling preserves exact zero by
scaling without centering. The scaling artifact binds the task and data contract
digests, Dataset identity, Train membership, tensorizer selection, spatial
shape, channel names, horizon, and fitted statistics. The same admitted artifact
is used by both stages, resume, and inference.

Production plans require PT shards bound to the exact immutable family package;
missing or incompatible shard publication fails preflight. The technical smoke
config alone explicitly permits canonical HDF5 and the material-pilot package.
Backend provenance and split membership are persisted, so admitted production
Training does not depend on the deleted CPU Generation workspace.

## Losses, metrics, and model implementations

Transient models predict scaled increments for the four dynamic state channels.
The maintained plans use Huber loss in scaled-increment space; MSE is also
supported. An optional reconstructed-state auxiliary term is explicit. No
transient physics residual is implemented, so physics remains disabled and
cannot be presented as PINO training.

The central selection objective is
`normalized_drying_group_macro_rmse`, a macro average over temperature,
humidity, and grain-moisture groups. Per-channel normalized and physical RMSE,
physical MAE, and grain moisture `w_gr` are reported separately.

FNO and U-NO use the repository's neuraloperator-backed factory. RNO is imported
from the official neuraloperator package; no local recurrent imitation or
UNO-RNO alias exists. Both Conda environments pin neuraloperator commit
`86a8bc7812a31b42c4f7895693cf4ac11521c066` so construction, sequence behavior,
and checkpoint loading use one reproducible API.

## Matched-compute A+ comparison

The normal plan produces A0 and B. To compare rollout continuation B with a
teacher-forced continuation A+ at matched post-handoff compute, generate A+
from the completed A0 and B evidence:

```bash
python -m src.experiments.cli.cli_transient_matched_config \
  /path/to/completed_stage_a0 \
  /path/to/new_a_plus.yaml \
  --arm a_plus \
  --b-run-dir /path/to/completed_stage_b

python -m src.experiments.cli.cli_train /path/to/new_a_plus.yaml
```

The generator refuses to overwrite its destination. It derives the A+ budget
from completed B terminal evidence and requires both arms to share the exact A0
handoff, model/profile, and clock kind. An optimizer group is the atomic matching
unit. CPU arms stop at the group that reaches the exact successful-step target.
CUDA arms stop immediately after the first group whose measured optimizer-device
time crosses the target; the unavoidable overshoot is therefore bounded to one
group and its crossing evidence is persisted rather than presented as exact
second equality. Evaluation at that discrete boundary is eligible for selection,
and no later optimizer group can affect the retained best checkpoint. The
lower-level `--arm b --budget VALUE` generation path remains available for
deliberate standalone B experiments, but normal model training does not require
it.

## Optuna and W&B

Exactly two transient scientific study recipes are maintained:

```text
configs/learning/transient_drying/optuna/
├── transient_drying_lentil_chickpea_stage_a_only.yaml
└── transient_drying_lentil_chickpea_joint_ab.yaml
```

The Stage-A-only study trains the teacher-forced Stage A for the full configured
200-epoch budget and has no allocation parameter. The joint study runs one
model through Stage A, the immutable selected-checkpoint handoff, and autonomous
Stage B. It samples exactly one configuration-owned
`training.stage_schedule.stage_a_fraction` from 0.25 through 0.75 in 0.05
increments. Half-up rounding assigns Stage A at least one epoch, and Stage B
receives the exact remainder, so every completed joint trial consumes exactly
200 epochs. The maintained fixed FNO, U-NO, and RNO plans use the same schedule
resolver with a fixed fraction of 0.5.

Run either study with preflight first:

```bash
python -m src.experiments.cli.cli_config_preflight optuna \
  configs/learning/transient_drying/optuna/transient_drying_lentil_chickpea_stage_a_only.yaml
python -m src.experiments.cli.cli_optuna \
  configs/learning/transient_drying/optuna/transient_drying_lentil_chickpea_stage_a_only.yaml

python -m src.experiments.cli.cli_config_preflight optuna \
  configs/learning/transient_drying/optuna/transient_drying_lentil_chickpea_joint_ab.yaml
python -m src.experiments.cli.cli_optuna \
  configs/learning/transient_drying/optuna/transient_drying_lentil_chickpea_joint_ab.yaml
```

Both studies consume only the 80-case Train and 10-case Validation memberships
of `transient_drying__lentil+chickpea__id`. The 10-case ID-test membership and
the 20 kidney-bean held-out OOD package never participate in fitting, pruning,
checkpoint selection, tie-breaking, or the objective. The study signature binds
the validated package manifest, compact index, exact split membership, study
mode, total and unit, allocation search space, model/training semantics,
objective, and seeds before a database is created or reopened.

Optuna reports actual completed epochs on one monotonic axis across both stages.
A pruned joint trial retains its sampled fraction, derived A/B epoch counts,
current stage, run leaves, and consumed global epoch count. Stage B uses the
normal immutable handoff and returns the final joint objective. Reopening a
study preserves completed history but follows the existing explicit
`new_trials_only` policy: interrupted trials are restarted as fresh numbered
trials, including Stage A, rather than partially resumed or resampled in place.

Transient tracking uses W&B project `grainlegumes-pino-drying-transient` and
supports `online`, `offline`, and `disabled` modes. Stage, curriculum, rollout,
central metrics, budget evidence, throughput, and memory are mapped into
telemetry. Local configs, summaries, histories, checkpoints, handoffs, and study
storage remain authoritative; W&B is an observer, not the persistence owner.

## Completed-output EDA

`notebooks/eda.ipynb` is the maintained generated-output entry point. One
workspace exposes `steady_flow` and `transient_drying` through the same
capability-adaptive panel with no task selector. Discovery includes compatible
terminal batches and independently valid completed cases from partial, failed,
or active campaigns. The notebook remains bounded by `MAX_CASES`, performs no
Training or inference, and stores no generated output.

The completed-output loader preserves strict terminal-batch admission. For
non-terminal campaigns it uses Generation-owned full case validation, including
source and output hashes, before exposing:

- all four absolute physical states in canonical order: `T`, `phi`, `w_surf`,
  and `w_int`;
- retained Training static fields and clearly classified archive-only fields;
- complete inlet schedules plus both endpoints and optional startup support for
  every learned interval;
- realized completed-case material-conditioning parameters;
- regular physical state times and a separately labelled diagnostic exact-stop
  state;
- canonical target attainment, right censoring, physical drying duration,
  reached-only time to target, and signed final gap
  `f_wet_dm_final - f_wet_dm_max`, where positive means still too wet;
- COMSOL process and available operational timing as metadata, kept separate
  from physical duration and scientific Dataset identity.

Spatial state views offer only exact available physical times and never select a
nearest output. Dynamic channel selectors default to all compatible states in
canonical order, with separate axes and units. Full trajectories use physical
time on the horizontal axis and static views have no time control. The target
summary states that reached and unreached percentages use valid completed cases
as their denominator; unreached cases receive no fabricated target time.

Canonical Generation HDF5 is the completed-case scientific and completion
authority. The semantic runtime-item adapter produces the same state, static,
boundary, scalar, and time views from canonical HDF5 and validated PT shards.
Missing PT shards never hide valid HDF5 science, and PT-backed semantic EDA does
not reopen a deleted source HDF5. Full Generation schedules, global series,
target status, and runtime remain explicitly unavailable when a runtime item did
not persist them.

The reusable API is under `src.analysis.eda.transient`; the notebook delegates
field discovery, exact-time selection, trajectory reductions, schedule and
parameter tables, target diagnostics, and runtime tables to that owner. The
Generation timing loader owns `timing.json` and `status.json` admission. EDA does
not parse those files, solver logs, scheduler text, or `sacct` output.

### Prioritized future plot recommendations

| Priority | Scientific question | Required fields | Required reduction | Current availability | Owner | Reason not implemented |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | How does target-attainment probability evolve under right censoring? | Canonical target state, physical duration, reached-only target time, material metadata | Kaplan-Meier or another explicitly selected censoring-aware estimator with uncertainty | Case evidence is available; estimator policy is not selected | Completed-output EDA | The mandatory diagnostic reports truthful counts and reached-only quantiles; choosing a survival estimator and confidence method requires a separate scientific decision. |
| 2 | Which spatial modes dominate each transient state over physical time? | One dynamic state at a time, coordinates, exact state times, valid masks | Per-channel time-aligned POD/SVD with an explicit weighting and centering contract | State evidence is available from HDF5/PT | Completed-output EDA | Cross-channel raw-unit decomposition would be invalid, and a maintained weighting/centering choice has not been established. |
| 3 | How do schedule changes precede moisture and thermal response? | Complete `T_in_bc`/`omega_in_bc` schedules, state/global trajectories, startup support | Physically aligned lag or impulse-response summary with censoring disclosure | Complete evidence is available only in canonical completed-case HDF5 | Completed-output EDA | A defensible lag window and causal interpretation are not yet defined; a generic correlation plot would overstate the evidence. |
| 4 | Which Generation phase drives computational cost? | Stationary-airflow, transient-drying, scientific-solver, queue, licence, and end-to-end timing | Per-component distributions and correlations against case metadata | Component fields are explicitly unavailable in current persisted timing | Generation for persistence; EDA for later presentation | EDA must not infer component timing from logs or scheduler text. |


## Evaluation

A successful transient Training run now invokes the shared task-aware artifact
service unless `--no-build-artifacts` is selected. Existing completed runs can
be processed explicitly with:

```bash
python -m src.experiments.cli.cli_build_artifacts --task transient_drying
```

The immutable artifact stores reference and predicted absolute sequences,
physical times, transition and origin identities, masks, conditioning,
checkpoint-bound scaling and profile identity, Dataset membership, Training
lineage, target and censoring evidence, and admitted timing provenance. It
supports FNO, U-NO, and official RNO checkpoints through the public transient
inference API. Independent cases and rolling origins reset RNO state; one
rollout forwards state only within that request.

Evaluation distinguishes teacher-forced one-step prediction, full autonomous
rollout, and early/middle/late rolling origins. Fixed horizons are measured in
transitions at `1, 2, 4, 8, 16, 32, 64, and 128`, while `full` means every
available transition. Unsupported fixed horizons remain unavailable rather than
being shortened. Artifact admission requires the exact deterministic
mode/origin/horizon inventory for every saved complete case, including explicit
unsupported-horizon evidence; an incomplete cache is rejected before reporting.
Every result retains elapsed physical time and separate cumulative and endpoint
reductions.

Only the full-autonomous/full-horizon record carries canonical completed-case
reference target evidence from Generation. Its reached-only time preserves an
irregular diagnostic exact-stop time even when that time is not on the learned
regular rollout grid. Prediction target evidence remains explicitly
regular-grid-local. Teacher-forced and rolling-origin intervals mark the
completed-case reference target unavailable instead of relabelling a partial
interval as right-censored. Final-gap errors are paired only when reference and
prediction endpoints are the same physical time.

The central metric is
`normalized_drying_group_macro_rmse` with checkpoint-bound Train-only scaling
and channel weights `T=1/3`, `phi=1/3`, `w_surf=1/6`, and `w_int=1/6`.
Complete-Dataset sufficient statistics are accumulated before RMSE reduction.
Reports also include normalized and physical channel errors, granular moisture,
canonical masked bulk dry- and wet-basis moisture, target censoring, plausibility
and stability diagnostics, and contributing and unavailable case counts. Bulk
moisture uses the same structured-grid trapezoidal boundary weighting as the
Generation consistency owner and never imputes physically invalid predictions.

Exact three-run A0/A+/B selections admit persisted common-parent and
matched-compute evidence. B versus A+ is the primary technique comparison; B
versus A0 remains separate because it includes additional post-handoff compute.
The Airflow-to-Drying analysis preserves conditions A, B, and C. If a compatible
Airflow-NO checkpoint, normalizer, profile, Dataset, or case mapping is absent,
C is reported unavailable with its reason and no prediction or zero error is
fabricated.

Timing uses the stable Generation timing interpreter plus bounded public
inference benchmarks. Cold timing, warm-up, raw warmed repetitions, backend,
PT-payload identity, precision, batch size, hardware, and software versions are
retained. Solver and surrogate timing components remain separate; algebraically
combined comparisons are labelled component-composed and aggregate paired cases
with a ratio of summed reference time to summed surrogate time. Missing
components retain exact unavailable reasons.

Use `notebooks/eval_single_model.ipynb` for one completed run and
`notebooks/eval_comparison_models.ipynb` for compatible comparisons. Both
notebooks dispatch by task, treat absent transient OOD evidence honestly, and
can render reports outside immutable artifact caches. Transient panels and local
reports include sequence maps, physical-time and horizon error, endpoint versus
cumulative error, target status, pipeline degradation, matched-compute
performance, runtime distributions, accuracy versus inference time, accuracy
versus speedup, and all five speedup definitions. The Evaluation session exposes
only bounded aggregate tracking fields; large sequence arrays remain in local
immutable artifacts.

The resolved `tracking.wandb` configuration remains the sole publication gate.
When its mode is enabled and `upload.evaluation_artifacts` is true, the explicit
artifact-build workflow validates transient ID and any saved OOD role before it
resumes the existing W&B observer. It publishes one bounded flat summary of
aggregate metrics, availability and sample counts, timing medians,
ratio-of-sums speedups, and Dataset, checkpoint, input-profile, model, backend,
hardware, precision, and timing identities. It never publishes
sequence arrays, raw Dataset values, per-case timings, provenance files, media,
checkpoints, or arbitrary files. Disabled mode performs no observer work;
steady-flow provenance and curated-media behavior is unchanged.

## Inference

A completed transient run can be reconstructed without reopening its Dataset:

```python
from src.learning.inference import (
    load_transient_inference_context,
    predict_transient_step,
    rollout_transient_autonomous,
)

context = load_transient_inference_context(run_dir=run_dir, device="cuda")
```

`predict_transient_step` accepts one explicit physical interval.
`rollout_transient_autonomous` accepts a validated interval sequence and
self-feeds reconstructed states. Both support FNO, U-NO, and RNO, including RNO
hidden-state forwarding within one request. Inputs include both boundary
endpoints, startup support, exact times and `dt`, and either the default COMSOL
reference airflow or explicitly compatible external airflow. Inference uses
float32-equivalent model state and reports measured model-call time without
claiming a speedup.

## Current limitations

- No compatible Airflow-NO checkpoint, normalizer, profile, and complete case
  mapping is selected by the current transient artifact plan, so pipeline
  condition C remains explicitly unavailable until that evidence exists.
- A real Evaluation report still requires completed compatible transient run and
  Dataset artifacts outside the repository; tests use bounded synthetic
  fixtures and do not claim measured model quality or speedup.
- Timing components absent from admitted Generation or inference evidence remain
  unavailable and are never inferred from logs or substituted with zero.
- Surrogate mass-balance closure remains unavailable because the sequence artifact
  does not preserve the gas-water storage and inlet/outlet mass-flux series needed
  for the canonical differential and integral balances; no closure value is
  inferred from moisture states alone.
- Production scientific hyperparameters and Dataset names remain mutable config
  choices; inspect the resolved run config rather than treating this guide as a
  frozen experiment inventory.
- The canonical 28-channel profile is the only supported channel selection until
  a general exclusion contract is designed and identity-bound end to end.
