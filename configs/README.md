# Configurations

Every YAML file starts with a `# Role:` line that says what it controls and when you edit it. This page lists the files by how often you change them.

## What you edit

| Edit | Files | Controls | Guide |
| --- | --- | --- | --- |
| For every Dataset | `02_datasets/<task>/*.yaml` | Materials with their role (`id` or `ood`) and power-of-two counts per design kind (Train, validation, test, parameter OOD), the Generation `seed_base` and minimum success rate of one Dataset; case roles follow from the design kinds, the domain profile and the Dataset name are computed; `smoke.yaml` is the acceptance smoke and never edited | [Datasets](../docs/04_datasets.md) |
| For every experiment | `03_learning/<task>/experiments/<family>/<config name>.yaml` ([naming](#reading-a-config-name)) | Model, loss and the training values that differ from the task base | [Training](../docs/05_training.md) |
| For every study | `03_learning/<task>/optuna/<study>.yaml` | Trial count, sampler and pruner, the fixed experiment values and the search space of one Optuna study | [Training](../docs/05_training.md) |
| Occasionally | `03_learning/<task>/base.yaml` | Values every experiment and study of the task shares: seed, data and batch size, loss, optimizer, scheduler, training budget, tracking | [Training](../docs/05_training.md) |
| When cluster use changes | `01_generation/execution.yaml` | Slurm resources, timeouts, retries and retention of the Generation node jobs; no physics | [Generation](../docs/03_generation.md) |
| Only with new physical evidence | `01_generation/physical_model/` | The simulated physics: grid and time (`common.yaml`), parameter definitions (`registry.yaml`), bibliography (`sources.yaml`), operation and inlet schedule (`fixed_bed.yaml`), bed and field statistics (`bed_fields.yaml`), one record per material (`materials/`), COMSOL template and exported fields per task (`profiles/`) | [Physical model](../docs/02_physical_model.md) |

## How the files combine

- A Dataset plan names the Generation files it uses (six `physical_model/` files plus `execution.yaml`) and adds only what defines the Dataset; material records are found by family name in `physical_model/materials/`. A change in `physical_model/` affects every Source generated afterwards.
- An experiment or study names its task base with `base:` and states only what differs from it, such as the model, `optimizer.lr` and the physics settings; a test rejects a restated base value or model-factory default. The study's `experiment` block holds the search base values and the trial-only keys (`training.epochs`, `tracking.wandb.workflow: optuna_trial`). Only the transient A0 studies have no base: they cannot inherit its matched A0 -> {B, A1} plan, so they restate its shared values, and a test keeps those equal to `base.yaml`.
- Each file lists first what you change (model sizes, `optimizer.lr`, physics weights; in a study `n_trials` and `search_space` before the experiment block), then the rest.
- Values set in no file of the chain come from the code defaults of the keys the base does not state (`run.revision`, `evaluation.metrics`, `tracking.wandb.upload`; steady flow also the whole `loss.physics` block and `tracking.wandb.monitor.enabled` and `max_cases`; transient drying also `data.pin_memory`, `temporal.sampling`, `optimizer.betas`, `optimizer.second_moment_floor` and the single-stage `training` constants) and from the model factory (the channel ratios and skip types of the FNO, U-NO and RNO, the FNO `implementation`, the RNO flags); the resolved run configuration records all of them.
- An experiment's folder below `experiments/` is its family and its file stem is its config name, which the loader computes and enforces (any other name is rejected). The maintained family is `architecture_comparison`: steady flow has a supervised and a physics-informed config per architecture (FNO, U-NO), transient drying one matched-plan config per architecture (FNO, U-NO, RNO).
- The loader checks every experiment YAML, also one outside `configs/03_learning/` (passed by any path); Optuna studies and in-memory sources without a file are exempt, and the `optuna/` directory holds only studies (a file with an `experiment` block) and refuses an experiment. Family and name appear in the run path, and the stem of a study file is its study name.

<details>
<summary><strong>Naming enforcement and name length</strong></summary>

- The name is the model label, then one `__<token>-<value>` for every shared, single-stage or A0 leaf the file itself sets (the file without `task` and `base`; the task base supplies no token, so changing a value in `base.yaml` renames nothing).
- The model label reads the merged model parameters and the physics switch, so a base change of a model parameter renames every config; that is correct, because the model changes. [Reading a config name](#reading-a-config-name) lists every part of the model label and every token with the config key it shows; a physics-informed config states `loss.physics.derivatives.kind` itself.
- The B/A1 leaves a matched plan file sets are not in the config name, because the config folder also holds A0 and a B edit must not rename it: they name the B/A1 pair folder instead.
- A leaf that is neither shown by a token nor on the short exclusion list of that module (seed, revision, device, Dataset selection, loader and tracking settings, the single admitted optimizer, scheduler and warm-up kinds, stage markers) is refused with the key named.
- The run name adds seed, A0 epoch budget and the B/A1 pair; the stored identity digest decides a run's identity, never a name.
- A name is at most 255 bytes (the limit of one path component, tested for every maintained config, all its stage run names and a worst-case config); one whose run name exceeds it is rejected at load with the byte count. A pair folder name is at most 255 bytes like every name, with a load error naming the byte count.

</details>

## Command-line options

Options override a file for one run without changing it; each is documented with its default in the guide of its workflow. An option that is omitted leaves the config value in force: `--train-dataset` falls back to `data.train_dataset` and `--train-seconds` to `training.b.train_seconds`, and a config that leaves either `null` needs the option (`--train-seconds` only for a transient matched plan). The other overrides of `train` and `optuna` fall back to their config keys in the same way ([Training](../docs/05_training.md)). The resolved run configuration records the values used.

- Training, Optuna, artifacts and probe (`scripts/slurm_ml.sh <resources> train|optuna|artifacts|probe`, with `--mode`, `--partition`, `--cpus-per-task`, `--mem`, `--time`, `--gres`, all optional): [Training](../docs/05_training.md) and [Setup and execution](../docs/01_setup_and_execution.md#ml-jobs).
- Generation (`scripts/generation run`): [Generation](../docs/03_generation.md).
- Dataset runs (`python -m src.datasets.dataset_packages run PLAN`): [Datasets](../docs/04_datasets.md).

## Reading a config name

A config name reads `<model label>__<token>-<value>__...`. For example `pi-fno_m128x160_h64_l3_p0p1__fd__w-1-1000-100__wu-100__lr-0p000388` is a physics-informed FNO with 128 Fourier modes along y and 160 along x, 64 hidden channels, 3 layers and 10 % domain padding, finite-difference physics derivatives, physics target weights 1, 1000 and 100 with 100 warm-up epochs, and learning rate 0.000388. Numbers write `p` for the decimal point and `minus` for a sign, a list joins its values with `-`, and a boolean reads `yes` or `no`. Setting, stage and pair values keep three significant digits and an integral value stays exact; two configs that differ only beyond the third digit share a name, and the second is refused at allocation. Model-label numbers are written in full, except the U-NO mode ratio (three decimals). The tables follow the name order. `_TOKENS`, `_PAIR_TOKENS` and `_STAGE_LEAVES` in `src/experiments/config/experiments_config_naming.py` are authoritative, and a test keeps the token tables equal to them.

### Model label

| Part | Meaning | Config key |
| --- | --- | --- |
| `pi-` | physics loss on | `loss.physics.enabled` |
| `fno`, `uno`, `rno` | model kind | `model.kind` |
| `m<Y>x<X>` | Fourier modes along y and x | `model.params.n_modes` (FNO, RNO); `model.params.modes_y`, `modes_x` (U-NO) |
| `h<N>` | hidden channels | `model.params.hidden_channels` |
| `l<N>` | layers | `model.params.n_layers` |
| `p<D>` | domain padding per side (`p0p1` is 10 %, `p0` none) | `model.params.domain_padding` |
| `s<...>` | U-NO only: the spatial scaling of each layer, joined with `-`; a value below 1 drops its leading `0p` (`05` is 0.5), from 1 on it keeps `p` (`1p5`); `<Y>x<X>` when the two axes differ | `model.params.uno_scalings` |
| `r<R>` | U-NO only: mode ratio, three decimals | `model.params.mode_ratio` |
| `__<param>-<value>` | any other model parameter that differs from the model factory default, `_` written as `-` | `model.params.<param>` |
| `__fd`, `__spec` | physics derivatives: central finite differences on the grid, or spectral with Fourier continuation | `loss.physics.derivatives.kind` (`physical`, `spectral`) |

<details>
<summary><strong>Setting tokens</strong></summary>

| Token | Meaning | Config key |
| --- | --- | --- |
| `crop` | physics interior crop in grid cells | `loss.physics.interior_crop` |
| `w` | physics target weights of momentum, continuity and boundary; one number when all three are equal | `loss.physics.<term>.target` |
| `wu` | physics warm-up epochs, same rule | `loss.physics.<term>.warmup.epochs` |
| `loss` | data loss kind | `loss.data.kind` |
| `space` | space the data loss is computed in | `loss.data.space` |
| `lw` | data loss weight | `loss.data.weight` |
| `huber` | Huber beta | `loss.data.beta` |
| `cw` | channel weights | `loss.data.channel_weights` |
| `aux` | state auxiliary weight | `loss.data.state_aux_weight` |
| `stride` | spatial stride of the training grid | `data.spatial_stride` |
| `bs` | batch size | `data.batch_size` |
| `sub` | Train subset per ID material | `data.train_subset` |
| `lr` | learning rate | `optimizer.lr` |
| `wd` | weight decay | `optimizer.weight_decay` |
| `betas` | AdamW betas | `optimizer.betas` |
| `eps` | second-moment floor | `optimizer.second_moment_floor` |
| `sch` | `sch-none`: no scheduler | `scheduler: null` |
| `schf` | plateau factor | `scheduler.factor` |
| `schp` | plateau patience in validation checks | `scheduler.patience` |
| `schmin` | minimum learning rate | `scheduler.min_lr` |
| `det` | deterministic algorithms | `run.deterministic` |
| `profile` | transient input profile | `input_profile` |
| `scale` | transient scaling mode | `scaling.mode` |
| `tcond` | transient temporal conditioning | `temporal.temporal_conditioning.kind` |
| `obj` | selection objective | `evaluation.objective.id` |
| `ep` | epochs | `training.epochs` |
| `ev` | evaluation interval | `training.evaluation_interval` |
| `oev` | OOD evaluation interval | `training.ood_evaluation_interval` |
| `acc` | gradient accumulation steps | `training.gradient_accumulation_steps` |

</details>

<details>
<summary><strong>Stage tokens of a transient matched plan</strong></summary>

A stage token is the stage prefix plus a suffix, for example `a0smp` or `bws`. A0 tokens follow the setting tokens in the config name; B and A1 tokens name the pair folder.

| Token | Meaning | Config key | Stages |
| --- | --- | --- | --- |
| `<stage>ep` | maximum epochs | `training.<stage>.max_epochs` | `a0`, `b`, `a1` |
| `<stage>ev` | evaluation interval | `training.<stage>.evaluation_interval` | `a0`, `b`, `a1` |
| `<stage>oev` | OOD evaluation interval | `training.<stage>.ood_evaluation_interval` | `a0`, `b`, `a1` |
| `<stage>acc` | gradient accumulation steps | `training.<stage>.gradient_accumulation_steps` | `a0`, `b`, `a1` |
| `<stage>cur` | curriculum lengths, then `-at-` and the milestone fractions when these are not equally spaced | `training.<stage>.curriculum.lengths`, `milestone_fractions` | `a0`, `b`, `a1` |
| `<stage>smp` | sampling mode | `training.<stage>.sampling.mode` | `a0`, `b`, `a1` |
| `<stage>rl` | rollout length | `training.<stage>.sampling.rollout_length` | `a0`, `a1` (B: `rollout` below) |
| `<stage>ws` | window stride | `training.<stage>.sampling.window_stride` | `a0`, `b`, `a1` |
| `<stage>wo` | window offset | `training.<stage>.sampling.window_offset` | `a0`, `b`, `a1` |
| `<stage>ts` | training seconds | `training.<stage>.train_seconds` | `a0`, `a1` (B: `budget` below) |
| `<stage>lrs` | learning-rate scale | `training.<stage>.lr_scale` | `a0` (B and A1: `lr` below) |

</details>

<details>
<summary><strong>Pair folder</strong></summary>

The B/A1 pair folder below A0 reads `b-a1__budget-<S>s__rollout-<L>__lr-<X>`, for example `b-a1__budget-3655s__rollout-32__lr-1__bsmp-rollout-window__bws-32__bwo-0` (72 bytes). `budget` is B's measured training time `training.b.train_seconds` (or `--train-seconds`), `rollout` B's rollout length `training.b.sampling.rollout_length` (or `--rollout-length`), and `lr` the shared `lr_scale` of B and A1, written `lr-b-<X>__lr-a1-<Y>` when they differ. Then follow the B/A1 tokens the plan file sets:

| Token | Meaning | Config key |
| --- | --- | --- |
| `cev` | continuation evaluations of B and A1 | `training.continuation_evaluations` |
| `b<suffix>`, `a1<suffix>` | the stage tokens above | `training.b.<leaf>`, `training.a1.<leaf>` |

</details>

The run name adds `__seed-<N>`, for a transient run `__a0ep-<E>` (A0's epoch budget), an optional `__rev-<label>`, the pair folder and the stage ([Results layout](../docs/05_training.md#results-layout)); Optuna trial folders are explained under [Optuna and tracking](../docs/05_training.md#optuna-and-tracking).

## Keys a config chain must state

The code holds no default for what a task's `base.yaml` owns: a chain (the experiment merged over its base, or a transient A0 study alone) that omits one of these keys is rejected with its path. Everything else in a base is optional and falls back as described above.

<details>
<summary><strong>Required keys per task</strong></summary>

| Keys | Steady flow | Transient drying |
| --- | --- | --- |
| Task and model | `task`, `model.kind`, the family's sizes (FNO `n_modes`, `hidden_channels`, `n_layers`; U-NO `modes_x`, `modes_y`, `hidden_channels`, `n_layers`), `model.params.positional_embedding`, `model.params.domain_padding` | the same, and the RNO's `n_modes`, `hidden_channels` and `candidate_activation` |
| Run | `run.seed`, `run.device`, `run.deterministic` | the same |
| Data | `data.train_dataset` (`null` needs `--train-dataset`), `data.material_ood_datasets` (`[]` for none), `data.train_subset`, `data.batch_size`, `data.num_workers`, `data.pin_memory`, `data.persistent_workers` | the same without `data.pin_memory`, plus `data.spatial_stride` |
| Transient inputs | none | `input_profile`, `temporal.temporal_conditioning`, `scaling.mode` |
| Loss | `loss.data.kind`, `loss.data.space`, `loss.data.weight` | `loss.data.kind`, `loss.data.space`, `loss.data.weight`, `loss.data.beta`, `loss.data.channel_weights`, `loss.data.state_aux_weight`, `loss.physics.enabled` |
| Objective | `evaluation.objective.id` | the same |
| Optimizer | `optimizer.kind`, `optimizer.lr`, `optimizer.weight_decay`, `optimizer.betas`, `optimizer.second_moment_floor` | `optimizer.kind`, `optimizer.lr`, `optimizer.weight_decay` |
| Scheduler | `scheduler.kind`, `scheduler.factor`, `scheduler.patience`, `scheduler.min_lr` | the same |
| Training | `training.epochs`, `training.evaluation_interval`, `training.ood_evaluation_interval` | matched plan: `training.mode`, `training.continuation_evaluations`, and per stage `training.a0` and `training.b` with `max_epochs`, `evaluation_interval`, `ood_evaluation_interval`, `gradient_accumulation_steps`, `sampling` and `curriculum`, plus `train_seconds` and `lr_scale` for B (`null` needs `--train-seconds`), `training.a1` with `max_epochs`, `match_to` and `lr_scale`; A0 study: `training.epochs`, `training.stage_schedule`, `training.evaluation_interval`, `training.ood_evaluation_interval` |
| Tracking | `tracking.wandb.mode`, `tracking.wandb.workflow`, `tracking.wandb.monitor.interval` | `tracking.wandb.mode`, `tracking.wandb.workflow`, `tracking.wandb.monitor.enabled`, `tracking.wandb.monitor.interval`, `tracking.wandb.monitor.max_cases` |

An Optuna study also states `study.seed`, `tracking.wandb.workflow: optuna_trial` in its experiment and, for transient drying, `study.mode: stage_a_only`; each `search_space` entry states `name` and `kind`.

</details>

## Seeds

| Seed | Where | Value |
| --- | --- | --- |
| Generation `seed_base` | `02_datasets/<task>/*.yaml` | 9200 (transient), 9300 (steady); every design seed derives from it, the material and the design kind, so there is no split seed |
| Learning `run.seed` | `03_learning/<task>/base.yaml` | 9; the replicate runs use 9, 19 and 29 through `train --seed N`, which keeps the config name (the W&B group) and the layout parent and lands in `seed-N` (transient `seed-N__a0ep-E`) |
| Optuna `study.seed` | `03_learning/<task>/optuna/*.yaml` | 9 |
