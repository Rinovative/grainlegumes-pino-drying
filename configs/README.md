# Configurations

Every YAML file starts with a `# Role:` line that says what it controls and when you edit it. This page lists the files by how often you change them.

## What you edit

| Edit | Files | Controls | Guide |
| --- | --- | --- | --- |
| For every Dataset | `02_datasets/<task>/*.yaml` | Materials with their role (`id` or `ood`) and power-of-two counts per design kind (Train, validation, test, parameter OOD), the Generation `seed_base` and minimum success rate of one Dataset; case roles follow from the design kinds, the domain profile and the Dataset name are computed; `smoke.yaml` is the acceptance smoke and never edited | [Datasets](../docs/04_datasets.md) |
| For every experiment | `03_learning/<task>/experiments/<family>/<config name>.yaml` ([naming](#reading-a-config-name)) | Model, loss and the training values that differ from the task base | [Training](../docs/05_training.md) |
| For every study | `03_learning/<task>/optuna/<study>.yaml` | Trial count, trial budget as a fraction of K, sampler and pruner with its start point, the fixed experiment values and the search space of one Optuna study | [Training](../docs/05_training.md) |
| Occasionally | `03_learning/<task>/base.yaml` | Values every experiment and study of the task shares: seed, data, units per update, loss, optimizer, scheduler, training budget, tracking | [Training](../docs/05_training.md) |
| When cluster use changes | `01_generation/execution.yaml` | Slurm resources, timeouts, retries and retention of the Generation node jobs; no physics | [Generation](../docs/03_generation.md) |
| Only with new physical evidence | `01_generation/physical_model/` | The simulated physics: grid and time (`common.yaml`), parameter definitions (`registry.yaml`), bibliography (`sources.yaml`), operation and inlet schedule (`fixed_bed.yaml`), bed and field statistics (`bed_fields.yaml`), one record per material (`materials/`), COMSOL template and exported fields per task (`profiles/`) | [Physical model](../docs/02_physical_model.md) |

## How the files combine

- A Dataset plan names the Generation files it uses (six `physical_model/` files plus `execution.yaml`) and adds only what defines the Dataset; material records are found by family name in `physical_model/materials/`. A change in `physical_model/` affects every Source generated afterwards.
- An experiment or study names its task base with `base:` and states only what differs from it, such as the model, `optimizer.lr` and the physics settings; a test rejects a restated base value or model-factory default. The study's `experiment` block holds the search base values and the trial-only `tracking.wandb.workflow: optuna_trial`; its trials train `study.budget_fraction` of the budget K the chain states. A transient A0 study's experiment is the architecture's matched A0 -> {B, A1} plan on `base.yaml`, of which the study trains A0 alone.
- Each file lists first what you change (model sizes, `optimizer.lr`, physics weights; in a study `n_trials` and `search_space` before the experiment block), then the rest.
- Values set in no file of the chain come from the code defaults of the keys the base does not state (`run.revision` and `evaluation.metrics`; steady flow also the whole `loss.physics` block and `tracking.wandb.monitor.enabled` and `max_cases`; transient drying also `data.pin_memory`, `temporal.sampling`, `optimizer.betas`, `optimizer.second_moment_floor` and the single-stage `training` constants) and from the model factory (the channel ratios and skip types of the FNO, U-NO and RNO, the FNO `implementation`, the RNO flags); the resolved run configuration records all of them.
- An experiment's folder below `experiments/` is its family and its file stem is its config name, which the loader computes and enforces (any other name is rejected). The maintained family is `architecture_comparison`: steady flow has a supervised and a physics-informed config per architecture (FNO, U-NO), transient drying one matched-plan config per architecture (FNO, U-NO, RNO).
- The loader checks every experiment YAML, also one outside `configs/03_learning/` (passed by any path); Optuna studies and in-memory sources without a file are exempt, and the `optuna/` directory holds only studies (a file with an `experiment` block) and refuses an experiment. Family and name appear in the run path, and the stem of a study file is its study name.

<details>
<summary><strong>Naming enforcement and name length</strong></summary>

- The name is the model label, then one `__<token>-<value>` for every shared, single-stage or A0 leaf the file itself sets (the file without `task` and `base`; the task base supplies no token, so changing a value in `base.yaml` renames nothing).
- The model label reads the merged model parameters and the physics switch, so a base change of a model parameter renames every config; that is correct, because the model changes. [Reading a config name](#reading-a-config-name) lists every part of the model label and every token with the config key it shows; a physics-informed config states `loss.physics.derivatives.kind` itself.
- The B/A1 leaves a matched plan file sets are not in the config name, because the config folder also holds A0 and a B edit must not rename it: they name the B/A1 pair folder instead.
- A leaf that is neither shown by a token nor on the short exclusion list of that module (seed, revision, device, Dataset selection, loader and tracking settings, the single admitted optimizer and warm-up kinds, stage markers) is refused with the key named.
- The run name adds seed, A0 epoch budget and the B/A1 pair; the stored identity digest decides a run's identity, never a name.
- A name is at most 255 bytes (the limit of one path component, tested for every maintained config, all its stage run names and a worst-case config); one whose run name exceeds it is rejected at load with the byte count. A pair folder name is at most 255 bytes like every name, with a load error naming the byte count.

</details>

## Command-line options

Options override a file for one run without changing it; each is documented with its default in the guide of its workflow. An option that is omitted leaves the config value in force: `--train-dataset` falls back to `data.train_dataset`, and a config that leaves it `null` needs the option. The other overrides of `train` and `optuna` fall back to their config keys in the same way ([Training](../docs/05_training.md)). The resolved run configuration records the values used.

- Training, Optuna, artifacts and probe (`scripts/slurm_ml.sh <resources> train|optuna|artifacts|probe`, with `--mode`, `--partition`, `--cpus-per-task`, `--mem`, `--time`, `--gres`, all optional): [Training](../docs/05_training.md) and [Setup and execution](../docs/01_setup_and_execution.md#ml-jobs).
- Generation (`scripts/generation run`): [Generation](../docs/03_generation.md).
- Dataset runs (`python -m src.datasets.dataset_packages run PLAN`): [Datasets](../docs/04_datasets.md).

## Reading a config name

A config name reads `<model label>__<token>-<value>__...`. For example `pi-fno_m128x160_h64_l3_p0p1__fd__w-0p1-10-10__wu-100__lr-0p000388` is a physics-informed FNO with 128 Fourier modes along y and 160 along x, 64 hidden channels, 3 layers and 10 % domain padding, finite-difference physics derivatives, physics target weights 0.1, 10 and 10 with 100 warm-up epochs, and learning rate 0.000388. Numbers write `p` for the decimal point and `minus` for a sign, a list joins its values with `-`, and a boolean reads `yes` or `no`. Setting, stage and pair values keep three significant digits and an integral value stays exact; two configs that differ only beyond the third digit share a name, and the second is refused at allocation. Model-label numbers are written in full, except the U-NO mode ratio (three decimals). The tables follow the name order. `_TOKENS`, `_PAIR_TOKENS` and `_STAGE_LEAVES` in `src/experiments/config/experiments_config_naming.py` are authoritative, and a test keeps the token tables equal to them.

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
| `lw` | data loss weight | `loss.data.weight` |
| `stride` | grid stride of training, scaling fit and checkpoint selection | `data.spatial_stride` |
| `sub` | Train subset per ID material | `data.train_subset` |
| `lr` | learning rate | `optimizer.lr` |
| `wd` | weight decay | `optimizer.weight_decay` |
| `betas` | AdamW betas | `optimizer.betas` |
| `eps` | second-moment floor | `optimizer.second_moment_floor` |
| `schf` | plateau factor | `scheduler.factor` |
| `schp` | plateau patience in validation points | `scheduler.patience` |
| `det` | deterministic algorithms | `run.deterministic` |
| `profile` | transient input profile | `input_profile` |
| `tcond` | transient temporal conditioning | `temporal.temporal_conditioning.kind` |
| `obj` | selection objective | `evaluation.objective.id` |

The budget epochs (`training.budget_epochs`, `training.a0.budget_epochs`), the units per update (`training.units_per_update`) and the scheduler kind (`scheduler.kind`, constant for `scheduler: null`) have no config-name token: the seed group of every run names them, authored or inherited (see below).

</details>

<details>
<summary><strong>Stage tokens of a transient matched plan</strong></summary>

A stage token is the stage prefix plus a suffix, for example `a0smp` or `bws`. A0 tokens follow the setting tokens in the config name; B and A1 tokens name the pair folder.

| Token | Meaning | Config key | Stages |
| --- | --- | --- | --- |
| `<stage>cur` | curriculum lengths, then `-at-` and the milestone fractions when these are not equally spaced (only B draws self-fed rollouts) | `training.<stage>.curriculum.lengths`, `milestone_fractions` | `b` |
| `<stage>smp` | sampling mode | `training.<stage>.sampling.mode` | `a0`, `b`, `a1` |
| `<stage>rl` | rollout length | `training.<stage>.sampling.rollout_length` | `a0`, `a1` (B: `rollout` below) |
| `<stage>ws` | window stride | `training.<stage>.sampling.window_stride` | `a0`, `b`, `a1` |
| `<stage>wo` | window offset | `training.<stage>.sampling.window_offset` | `a0`, `b`, `a1` |
| `<stage>lrs` | learning-rate scale | `training.<stage>.lr_scale` | `a0` (B and A1: `lr` below) |

</details>

<details>
<summary><strong>Pair folder</strong></summary>

The B/A1 pair folder below A0 reads `b-a1__rollout-<L>__lr-<X>`, for example `b-a1__rollout-32__lr-1__bsmp-rollout-window__bws-32__bwo-0` (58 bytes). It names no budget: B's K x N supervised transitions follow from A0's `budget_epochs` K, which the seed group names, and A1's from B's receipt. `rollout` is B's rollout length `training.b.sampling.rollout_length` (or `--rollout-length`), and `lr` the shared `lr_scale` of B and A1, written `lr-b-<X>__lr-a1-<Y>` when they differ. Then follow the B/A1 tokens the plan file sets:

| Token | Meaning | Config key |
| --- | --- | --- |
| `b<suffix>`, `a1<suffix>` | the stage tokens above | `training.b.<leaf>`, `training.a1.<leaf>` |

</details>

The run name adds the seed group `__seed-<N>__ep-<E>__k-<k>__sch-<schedule>` (budget epochs, A0's in a transient run; units per update; `cosine`, `reduce-on-plateau` or `constant`), an optional `__rev-<label>`, the pair folder and the stage ([Results layout](../docs/05_training.md#results-layout)); Optuna trial folders are explained under [Optuna and tracking](../docs/05_training.md#optuna-and-tracking).

## Keys a config chain must state

The code holds no default for what a task's `base.yaml` owns: a chain (the experiment or study merged over its base) that omits one of these keys is rejected with its path. Everything else in a base is optional and falls back as described above.

<details>
<summary><strong>Required keys per task</strong></summary>

| Keys | Steady flow | Transient drying |
| --- | --- | --- |
| Task and model | `task`, `model.kind`, the family's sizes (FNO `n_modes`, `hidden_channels`, `n_layers`; U-NO `modes_x`, `modes_y`, `hidden_channels`, `n_layers`), `model.params.positional_embedding`, `model.params.domain_padding` | the same, and the RNO's `n_modes`, `hidden_channels` and `candidate_activation` |
| Run | `run.seed`, `run.device`, `run.deterministic` | the same |
| Data | `data.train_dataset` (`null` needs `--train-dataset`), `data.material_ood_datasets` (`[]` for none), `data.train_subset`, `data.evaluation_batch_size`, `data.num_workers`, `data.pin_memory`, `data.persistent_workers`, `data.spatial_stride` (1: full grid) | the same without `data.pin_memory` |
| Transient inputs | none | `input_profile`, `temporal.temporal_conditioning` |
| Loss | `loss.data.kind`, `loss.data.weight` | `loss.data.kind`, `loss.data.weight`, `loss.physics.enabled` |
| Objective | `evaluation.objective.id` | the same |
| Optimizer | `optimizer.kind`, `optimizer.lr`, `optimizer.weight_decay`, `optimizer.betas`, `optimizer.second_moment_floor` | `optimizer.kind`, `optimizer.lr`, `optimizer.weight_decay` |
| Scheduler | `scheduler.kind` (`cosine`; `reduce_on_plateau` adds `factor` and `patience` and serves only the scheduler comparison; `scheduler: null` is a constant rate) | the same |
| Training | `training.budget_epochs`, `training.units_per_update`, `training.memory_batch` | matched plan: `training.mode`, `training.units_per_update`, `training.a0` with `budget_epochs` (K, one for every architecture), `training.b` with `sampling`, `curriculum` and `lr_scale`, `training.a1` with `lr_scale`; each experiment adds `training.a0.sampling`, which A1 takes, and `memory_batch` under `a0`, `b` and `a1` (an A0 study only under `a0`, the experiment's value), and only B has a curriculum |
| Tracking | `tracking.wandb.mode`, `tracking.wandb.workflow` | `tracking.wandb.mode`, `tracking.wandb.workflow`, `tracking.wandb.monitor.enabled`, `tracking.wandb.monitor.max_cases` |

An Optuna study also states `study.seed`, `study.budget_fraction`, a median pruner's start point `study.pruner.n_warmup_steps`, `tracking.wandb.workflow: optuna_trial` in its experiment and, for transient drying, `study.mode: stage_a_only`; each `search_space` entry states `name` and `kind`.

</details>

## Seeds

| Seed | Where | Value |
| --- | --- | --- |
| Generation `seed_base` | `02_datasets/<task>/*.yaml` | 9200 (transient), 9300 (steady); every design seed derives from it, the material and the design kind, so there is no split seed; the transient pilot plan `pilot.yaml` uses 9250, so no pilot case enters a production Dataset |
| Learning `run.seed` | `03_learning/<task>/base.yaml` | 9; the replicate runs use 9, 19 and 29 through `train --seed N`, which keeps the config name (the W&B group) and the layout parent and lands in `seed-N__ep-E__k-K__sch-S` |
| Optuna `study.seed` | `03_learning/<task>/optuna/*.yaml` | 9 |
