# Transient drying training

## Scope

The `transient_drying` task is trainable with FNO, U-NO, and the official
`neuralop.models.RNO`. Training consumes immutable Dataset packages, writes
independent run bundles under `03_experiments/transient_drying`, and remains
usable when the original CPU Generation workspace has been deleted.

This workflow owns Training, Optuna, checkpoint/resume, W&B telemetry, matched
compute, and inference loading. Transient EDA, Evaluation, and post-training
artifact generation are not implemented yet. The training CLI reports that
limitation after a successful transient run instead of invoking the steady-flow
artifact builder.

## Maintained experiment plans

Normal experiments use one architecture-first YAML per model:

- `configs/learning/transient_drying/experiments/fno_m128x160_h64_l3__material_pilot__s9.yaml`
- `configs/learning/transient_drying/experiments/uno_m64x64_h32_l7_s1-05-05-1-1-2-2_r0p495__material_pilot__s9.yaml`
- `configs/learning/transient_drying/experiments/rno_m24x24_h16_l3__material_pilot__s9.yaml`

The filename makes the architecture recognizable during scheduling and review.
The resolved YAML remains authoritative; filenames do not replace persisted
configuration, Dataset identity, seeds, or checkpoint hashes.

Each file is an authored two-stage plan. Shared sections define the task, data,
model, loss, optimizer, scaling, and tracking policy. `training.stage_a` and
`training.stage_b` define only the stage-owned duration, sampling, curriculum,
and matched-compute decisions. There are no separate Stage A and Stage B normal
config directories.

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
`2 -> 4 -> 8 -> 16 -> 32`, without restarting Stage A.

## Preflight and training

Run preflight before allocating any experiment directory:

```bash
python -m src.experiments.cli.cli_config_preflight train \
  configs/learning/transient_drying/experiments/fno_m128x160_h64_l3__material_pilot__s9.yaml
```

Start the complete A0-to-B workflow with the same file:

```bash
python -m src.experiments.cli.cli_train \
  configs/learning/transient_drying/experiments/fno_m128x160_h64_l3__material_pilot__s9.yaml
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

Production plans prefer PT shards but can use the admitted canonical HDF5
backend when shards are unavailable. Backend provenance and split membership are
persisted. Once valid PT shards and their publication evidence exist, Training
does not depend on the deleted CPU Generation source.

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

Optuna studies remain stage-specific because Stage A teacher forcing and Stage B
autonomous rollout have different objectives, failure modes, and handoff
requirements. Maintained templates are:

```text
configs/learning/transient_drying/optuna/
├── fno_stage_a.yaml
├── fno_stage_b.yaml
├── rno_stage_a.yaml
├── rno_stage_b.yaml
├── uno_stage_a.yaml
└── uno_stage_b.yaml
```

Run a study with:

```bash
python -m src.experiments.cli.cli_config_preflight optuna OPTUNA_CONFIG.yaml
python -m src.experiments.cli.cli_optuna OPTUNA_CONFIG.yaml
```

Before a Stage B study, replace its explicit `teacher_handoff.source_run_name`
with the admitted completed Stage A source. Studies persist locally and classify
non-finite scientific failures, OOM pruning, and infrastructure failures
separately.

Transient tracking uses W&B project `grainlegumes-pino-drying-transient` and
supports `online`, `offline`, and `disabled` modes. Stage, curriculum, rollout,
central metrics, matched-compute state, throughput, and memory are mapped into
telemetry. Local configs, summaries, histories, checkpoints, handoffs, and study
storage remain authoritative; W&B is an observer, not the persistence owner.

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

- Transient EDA and Evaluation are intentionally not implemented in this change.
- The steady-flow artifact builder does not consume transient runs.
- Production scientific hyperparameters and Dataset names remain mutable config
  choices; inspect the resolved run config rather than treating this guide as a
  frozen experiment inventory.
- The canonical 28-channel profile is the only supported channel selection until
  a general exclusion contract is designed and identity-bound end to end.
