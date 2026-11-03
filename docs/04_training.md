# Training

Training reads a published Dataset and one architecture configuration, fits scaling on the Train cases and writes checkpoints, histories and predictions to `storage/03_experiments/`. All training jobs run in the Apptainer image through `scripts/slurm_ml.sh` ([Setup and execution](01_setup_and_execution.md#ml-jobs)).

[Documentation index](README.md)

## Configurations

| Task | Configurations | Purpose |
| --- | --- | --- |
| Transient drying | `configs/03_learning/transient_drying/base.yaml` | The matched A0 → {B, A1} plan and the values every transient experiment shares |
| Transient drying | `configs/03_learning/transient_drying/experiments/architecture_comparison/` | One matched plan per architecture: `fno_m64x80_h64_l3`, `uno_m64x64_h32_l7_s1-05-05-1-1-2-2_r0p495`, `rno_m24x24_h16_l3` |
| Transient drying | `configs/03_learning/transient_drying/optuna/` | One A0 study per architecture: `fno_a0_search`, `uno_a0_search`, `rno_a0_search` |
| Steady flow | `configs/03_learning/steady_flow/base.yaml` | Values every steady experiment and study shares |
| Steady flow | `configs/03_learning/steady_flow/experiments/` | `model_selection/`, `capacity_and_physics/`, `architecture_comparison/` |
| Steady flow | `configs/03_learning/steady_flow/optuna/` | FNO, U-NO, PI-FNO and PI-U-NO searches |

A config names its task's `base.yaml` with `base:` and sets only what differs from it, such as the model, the physics loss and the learning rate; every value it sets overrides the base, and a base cannot name another base. The A0 studies keep their own single-stage training section and use no base; their shared semantic values stay equal to `base.yaml`. The filename starts with the model label the loader computes from the config: the architecture (modes, hidden channels, layers, U-NO scalings), prefixed `pi-` with the suffix `__fd` (central finite differences on the grid, config value `physical`) or `__spec` (spectral with Fourier continuation) when the physics loss is on, for example `pi-fno_m48x48_h64_l4__spec`. A variant may follow after a double underscore, such as `__lr-low`; a name that does not start with its model label is rejected. The folder below `experiments/` is the family that groups comparable configs. The resolved configuration saved with each run is authoritative. Templates leave `data.train_dataset: null`; select the Dataset there or with `--train-dataset`. The transient plan also leaves `training.b.train_seconds: null`; set it per experiment or pass `--train-seconds`. The rule for its value: A0's measured optimisation time (`terminal_controller.wall_seconds` in the A0 `summary.json`) of the sweep A0 on the same GPU type, fixed per architecture for all seeds.

## Transient drying

### Matched comparison: A0 → {B, A1}

Every transient experiment is one matched plan, authored in `base.yaml` as `training.mode: matched_a1_b` with the maps `a0`, `b` and `a1`; an experiment file overrides `training.a0` and `training.b`.

```bash
./scripts/slurm_ml.sh --mode gpu --partition gpu --cpus-per-task 4 \
  --mem 32G --time 04:00:00 --gres gpu:rtx6000ada:1 train \
  configs/03_learning/transient_drying/experiments/architecture_comparison/fno_m64x80_h64_l3.yaml \
  --train-dataset DATASET_ID --train-seconds 600
```

The plan trains A0 once, then B and A1, which both continue from A0's selected checkpoint. `--train-dataset` and `--train-seconds` are required to run unless the configuration sets them.

| Stage | Protocol | Budget |
| --- | --- | --- |
| A0 | Teacher forcing. FNO and U-NO learn one-step increments from shuffled reference transitions; the RNO trains teacher-forced on whole trajectories (see below). | `training.a0.max_epochs` (100) |
| B | Autonomous, self-fed rollouts for all architectures. The curriculum advances through rollout lengths 2 → 4 → 8 → 16 → 32 at budget fractions 0, 0.2, 0.4, 0.6, 0.8. | `training.b.train_seconds`, measured; `max_epochs` (100) is an operational cap |
| A1 | Teacher forcing like A0, with A0's sampling, curriculum, evaluation cadence and gradient accumulation. It authors only `max_epochs` (100, an operational cap), `match_to: b.train_seconds` and `lr_scale`. | B's measured seconds |

Rollout windows start every 32 transitions; when that stops short of the trajectory end, one more window ends exactly at the end, so the last transitions are trained too. Stage-B origins rotate over the three thirds of the window with the draw counter, so small batches reach every third. A Train case shorter than the window (32 transitions) aborts the start with a clear message; whether to allow partial windows, shorten the window or exclude such cases is decided once the production Dataset shows whether such cases exist. With 168 transitions the windows start at 0, 32, 64, 96, 128 and 136; the last two overlap by 24 transitions.

The RNO carries a hidden state, so it uses `sampling.mode: trajectory`: each item is a whole trajectory, cut by the trainer into the same windows of `rollout_length` in temporal order (truncated backpropagation through time). Every window starts from the hidden state the preceding window of the same trajectory reached at this window's start time (its end, or earlier for the overlapping last window); that state is detached, so gradients stop at the window boundary and memory per optimizer step stays that of one window. Each window is one optimizer microbatch. In B a teacher-forced pass over the reference window, without gradient, provides the state at each curriculum origin and at the next window, so the carried state never contains self-fed errors. Each case ends at its own drying stop, so trajectories may differ in length; a batch only holds trajectories of one length, shuffled anew every epoch. Stateless models cannot use trajectory sampling and the RNO cannot use anything else.

Every stage and architecture selects its best checkpoint by the same validation error: each validation trajectory is rolled out autonomously from its reference start state to its end, whatever its length, and `normalized_drying_group_macro_rmse` is taken over all predicted states. The RNO carries only its own hidden state in this rollout. Batches only hold trajectories of one length from one evaluation Dataset, and the metric pools all predicted states, so every state counts equally. The choice is independent of the training sample shape. A0 validates every `evaluation_interval` epochs. B and A1 validate only at epoch ends that crossed a new one of `training.continuation_evaluations` (20) equally spaced points of their measured budget, plus at budget completion, so both arms get the same number of checkpoint candidates and plateau-scheduler steps at the same budget fractions even though their epochs take different times; an epoch that crosses several points validates once. A0 and A1 still train teacher-forced; the teacher-forced one-step error over all validation transitions (the RNO with the state built over the reference transitions before each one) is logged as `guardrail/one_step`.

B and A1 start from A0's selected best checkpoint through an immutable handoff that restores model, optimizer, scheduler, AMP scaler, loss, sampler and RNG state with a fresh budget controller. Only this best checkpoint of a matched A0 carries that full training state; every other `best_checkpoint.pt` holds what evaluation needs (model and adapter weights, loss state, identity, best epoch and metric, objective history), and `last_checkpoint.pt` is the only source for continuing a run. The handoff's `best_checkpoint.pt` is a hard link to A0's file (one copy on disk; checkpoints are only replaced by rename, so the snapshot never changes, and a filesystem that cannot link fails the handoff instead of copying); copy an A0 with `cp -a` or `rsync -aH` to keep the link, because a copy without it becomes a second file. `training.b.lr_scale` and `training.a1.lr_scale` (default 1) multiply the learning rate restored from A0 together with the plateau floor `scheduler.min_lr`, so the continuation follows A0's schedule shape at another level; the optimizer moments and the plateau history are inherited unchanged. The factor applies only when a fresh B or A1 starts, never on resume. The handoff admits B and A1 only on the GPU model their A0 ran on, and `train_seconds` depends on that GPU; keep one GPU type per architecture for its A0 study, sweep and final runs. Transient work runs on the RTX 6000 Ada (48 GB, three on ICE, one per architecture): it runs B's 32-step rollouts two to three times faster than a V100 and also holds the largest search-space corners (FNO and RNO with batch 4 need 34–39 GB, more than a 32 GB V100). Steady-flow runs use the V100s. Each stage has its own resolved config, history, checkpoints, lifecycle summary and W&B run. Predictions for the selected completed stages are built after training unless `--artifacts` excludes them.

The order is A0, then B, then A1. B trains self-fed until it reaches `train_seconds`. A1 takes its target from B's actual seconds in `b_train_budget_receipt.json`. `train_seconds` counts the optimization time of committed optimizer groups and excludes loader waits, validation, checkpointing and artifacts; the crossing group may overshoot the target by at most its own duration, so A1 can exceed B's seconds by at most one optimizer group.

A stage whose epoch, step or time cap ends before its target finishes as `budget_unmet` (exit code 2) and is not a usable prerequisite; raise the cap and resume that run. A B stage that ends `budget_unmet` stops the pair before A1. Checkpoints store the shuffled epoch order and sample cursor, so B and A1 resume mid-epoch; this requires `data.num_workers: 0` and `drop_last: false`, and `base.yaml` authors `num_workers: 0` with `persistent_workers: false` for that reason. `data.num_workers` is an upper bound for the other runs: the loaders use at most as many workers as the job has CPUs (`--cpus-per-task`), and when the cap applies the job log says so once (`data workers 4 of 8 configured (4 CPUs)`); the configuration keeps the configured value.

| Option | Effect |
| --- | --- |
| `--epochs N` | A0's epoch budget only; it names the seed group (`seed-N__a0ep-E`), so another budget trains beside the first |
| `--train-seconds S` | B's measured target; A1 matches B's receipt |
| `--rollout-length L` | B window length and stride `L`; B's curriculum keeps lengths up to `L` |
| `--lr-scale X` | Continuation factor for both B and A1 (paired sweep) |
| `--revision LABEL` | Deliberately separate run `seed-N__a0ep-E__rev-LABEL` (steady: `seed-N__rev-LABEL`) after its science changed, for example code or config content; an identical experiment is never started twice, also not with a revision. The rule for every unit: *Same name, changed science: add `--revision LABEL` to the command you run. It never changes seeds, never makes identical science new, and never re-simulates a Source unless you also pass `--new-source` to the Dataset run.* |
| `--train-stages all\|a0,b,a1` | Stages to train; unselected prerequisites must be complete |
| `--artifacts selected\|none\|all\|a0,b,a1` | Post-training predictions: every selected completed stage (default), none, every stage, or the listed stages; a single-stage run accepts `selected` and `none` |
| `--max-epochs` | Operational epoch cap for one selected B or A1 stage or a steady run; A0's epochs are its budget and change only with `--epochs` |

With `--rollout-length 4` the curriculum becomes lengths `[2, 4]` at milestones `[0, 0.5]`. Every effective value is saved.

### Continuation sweep for B and A1

The continuation learning rate of B and A1 is tuned with normal `train` runs, not an Optuna study: Optuna tunes A0 only (see [Optuna and tracking](#optuna-and-tracking)). The sweep gives B and A1 the same scales from the same A0 and budget.

1. Copy the selected A0 values into the experiment configuration.
2. Train the shared A0 once:

   ```bash
   ./scripts/slurm_ml.sh --mode gpu --partition gpu --cpus-per-task 4 \
     --mem 32G --time 04:00:00 --gres gpu:rtx6000ada:1 train \
     configs/03_learning/transient_drying/experiments/architecture_comparison/fno_m64x80_h64_l3.yaml \
     --train-dataset DATASET_ID --train-seconds 600 --train-stages a0
   ```

3. For each scale `X` of a log grid declared in advance, such as 0.1, 0.3, 1 and 3, run B and A1 one after another, never concurrently, because every branch under one A0 is placed and published under that A0:

   ```bash
   ./scripts/slurm_ml.sh --mode gpu --partition gpu --cpus-per-task 4 \
     --mem 32G --time 04:00:00 --gres gpu:rtx6000ada:1 train \
     configs/03_learning/transient_drying/experiments/architecture_comparison/fno_m64x80_h64_l3.yaml \
     --train-dataset DATASET_ID --train-seconds 600 --lr-scale X \
     --train-stages b,a1 --artifacts none
   ```

   Every scale gives one B and one A1 pair with the same budget, in its own pair directory beside the shared A0 (see [Results layout](#results-layout)). `--artifacts none` keeps test predictions out of the sweep.
4. Per arm, select the scale with the lowest best validation objective (`best_metric` of `normalized_drying_group_macro_rmse` in each stage's summary), with a tie rule declared in advance, before any test evaluation.
5. Author the winners as `training.b.lr_scale` and `training.a1.lr_scale` in the experiment configuration and run the final matched comparison over several seeds as normal `train` runs.

B and A1 then differ in protocol and in their tuned continuation rate. Their validation sits at the same budget points, so both arms see the same number of checkpoint candidates; the selected minimum on five validation cases per material stays optimistic. A pair whose B ends `budget_unmet` has no A1; report its missing evaluations separately instead of treating them as bad scores.

### Continue, extend and resume

Rerun the same command to continue a run. An interrupted or unfinished run continues from its last checkpoint, a steady-flow run that already finished is extended in place by a larger `--epochs`, and a run that already reached its epochs ends at once (the line `run already_complete: completed_epoch N  run_dir <path>`). A steady run's epochs are not part of its name; a transient A0's budget is (`a0ep-E`). If the existing run holds different science (changed code, config content or Dataset), the start is refused with the list of differences; pass `--revision <label>` for a deliberately separate run. An identical experiment that already exists under another config name, family or revision is refused as well: `storage/03_experiments/.state/identities/` binds every experiment identity to one run directory. A run trained by the current command rebuilds its own artifacts, while a reused run only validates them. A resume that only finalizes writes empty `terminal_metrics` into the summary, and a run interrupted before the last observer callback lacks that epoch in W&B; `history.json` is authoritative and W&B only observes.

A run continues in the code state it last ran with. The launcher reads the source tree of the newest Slurm session among the leaves the job may write and runs the job from that frozen tree, restored from its snapshot when it was pruned; edits in `repo/` since then do not reach it. Those leaves are an existing steady run, the unfinished stages that `--train-stages` selects of a matched comparison, the one stage of `train --resume STAGE`, and for an Optuna study its own session record plus its existing trials. A completed A0, B or A1 is a read-only prerequisite and never binds: a new B/A1 pair on a completed A0 uses the current code. A saved-config resume is only structurally validated by the launcher; the job validates it fully inside the bound tree. A new run, pair or study, a unit that never ran under Slurm, and every `artifacts` job use the current code.

The job enforces the binding as well: continuing a leaf whose newest Slurm session ran in another tree is refused unless the job was submitted with `--continue-with-current-code`, and so is continuing it interactively, outside `scripts/slurm_ml.sh`. A study records its tree in `study_sessions.json` at creation and at every reopen, before its first trial, so it keeps its tree even if it never ran a trial, and a reopen in another tree is refused the same way. `--continue-with-current-code` (`train` and `optuna`) switches an existing unit to the current code. It is admitted only when the scientific identity is unchanged, because the start applies the same check as any rerun (changed implementation, config content or Dataset is refused with the list of differences; use `--revision`). The implementation part is the task's declared training projection (`TRAINING_PROJECTIONS`); a study adds its Optuna modules to the projection of its task. The switch is recorded per session as `source_transition: explicit` beside the source commit and worktree digest in the run's `runtime_sessions`.

A transient A0 is the exception to extension: its epoch budget sets its rollout curriculum and budget controller, so another budget is a different A0, not a longer one. The budget therefore names the seed group (`seed-N__a0ep-E`): another A0 `--epochs` trains in its own seed group beside the first, with its own B/A1 pairs, and needs no `--revision`. The same holds for the stage-A-only Optuna trials. A completed A0 is reused read-only when B is still missing; every pair starts from A0's handoff in `a0/handoff/`.

To continue one saved stage exactly from its own saved configuration, independent of the authored YAML:

```bash
./scripts/slurm_ml.sh --mode gpu --partition gpu --cpus-per-task 4 \
  --mem 32G --time 01:00:00 --gres gpu:rtx6000ada:1 train --resume STAGE_DIRECTORY
```

This reads the stage's `config.yaml` and last checkpoint and validates split, scaler, implementation, numerical environment, parent and handoff. Pass no scientific overrides on this path. A completed B or A1 stage continues only with a larger operational cap, for example `--max-epochs 24`; matched targets and `lr_scale` stay fixed. A0's epoch count is its scientific budget (its curriculum and budget controller depend on it), so this path keeps it.

### Reading the job log

`runtime/logs/ml/slurm-<job>.out` carries the progress, `.err` only warnings and errors; both are appended, and hard-linked into `logs/` of each run stage or study folder the job wrote ([Setup and execution](01_setup_and_execution.md#ml-jobs)). The header has three lines: job and resources, source tree with commit and transition, image with its digest prefix (the full digests are in the run summary). A run then prints a four-line start block (run, data, model with optimizer, loss and selection), one `startup` line for transient runs, one line per epoch such as `epoch   3/50  loss 3.215e-02  val 1.234e-02 *best  lr 3.88e-04  12.8 s  45.0 samples/s` and one end line with the best epoch, the selection metric, the duration and the run folder relative to `storage/`. `val` appears on the epochs that evaluate; loss components share the line when more than one remains besides the total. A failure prints its message and how to continue; a traceback follows only for an unexpected exception. An Optuna study prints a start line, one `trial` line per finished trial and an end line, with the Optuna library logger at WARNING. Artifact generation prints one line per finished phase and, while a phase runs, a progress line at most once per minute. W&B prints nothing. Its online SDK directory lives on the node-local `TMPDIR` and is copied to `<run>/wandb/` only when the tracking session fails; `offline` mode keeps it in `<run>/wandb/` for `wandb sync`.

### Short runs

`--epochs N` sets A0's epoch budget and `--train-seconds S` B's measured target; `--rollout-length L` shortens B's windows and curriculum. Sampling, architecture and loss stay as authored:

```bash
./scripts/slurm_ml.sh --mode gpu --partition gpu --cpus-per-task 4 \
  --mem 32G --time 00:20:00 --gres gpu:rtx6000ada:1 train \
  configs/03_learning/transient_drying/experiments/architecture_comparison/fno_m64x80_h64_l3.yaml \
  --train-dataset DATASET_ID --epochs 2 --train-seconds 30 --rollout-length 4 \
  --artifacts none
```

The effective budget is saved with the run. Short runs check the workflow; their accuracy carries no scientific meaning.

### Inputs, scaling and grid

The input profile `canonical_physics_complete_v1` has 28 ordered channels:

- current state `T`, `phi`, `w_surf`, `w_int`
- coordinates, COMSOL airflow, porosity and spatial material fields
- both inlet boundary endpoints of the interval, ambient and startup support
- scalar material parameters

Coordinates are explicit channels, so `positional_embedding` is `null`. Normalized current time can be added as a channel (`temporal.temporal_conditioning.kind: normalized_current_time`); the maintained plans use `none`.

Scaling (`scaling.mode: state_std`) is fitted on Train membership only. Absolute states are deduplicated by state identity; increments are scaled without centering, so zero stays zero. The scaling artifact binds task, Dataset, Train membership, profile, grid, channels and horizon, and is reused unchanged by every stage, resume and inference.

Training uses the endpoint-preserving view `data.spatial_stride: 2`, which turns the 251 × 401 source grid into 126 × 201. Evaluation artifacts choose their own grid ([Evaluation](05_evaluation.md#evaluation-grid)).

### Losses and objective

Transient models predict scaled increments of the four state channels. The maintained loss is Huber (`beta: 1`) in scaled-increment space with equal channel weights; MSE and an auxiliary reconstructed-state term are available. The transient physics loss is disabled.

Checkpoint selection and Optuna use `normalized_drying_group_macro_rmse`:

```text
(RMSE_T + RMSE_phi) / 3 + (RMSE_w_surf + RMSE_w_int) / 6
```

Squared errors and valid counts are pooled over the complete Dataset before each field root. Per-channel normalized and physical RMSE, physical MAE and total grain moisture `w_gr` are reported separately.

### Results layout

Run and study paths are a pure function of the saved layout (`run.layout`): the task, the training Dataset, the family and name of the config file, the seed and an optional revision. Nothing depends on which runs already exist.

```text
storage/03_experiments/<task>/<dataset>/<family>/<config_name>/seed-N[__a0ep-<E>][__rev-<label>]/
  a0/                                              # teacher-forced origin, shared by every pair
    handoff/                                       # selected A0 state that every pair starts from
  b-a1__budget-<S>s__rollout-<L>__lr-<X>/          # one B/A1 pair
    experiment.json                                # identities and relative child links
    b/                                             # autonomous continuation
    a1/                                            # matched teacher-forced continuation
```

`<family>` and `<config_name>` are the config's folder below `experiments/` and its file stem; `<dataset>` is `data.train_dataset`. The pair directory names B's measured budget, the rollout window `L` and the shared learning-rate factor `X` (`lr-b-<X>__lr-a1-<Y>` when B and A1 differ), always, also for the first pair. Further pairs on the same A0 get their own pair directory and reference the existing A0 (`../a0`) instead of copying it. A different architecture, seed, optimization, input or split gives a different A0, and so does a different A0 epoch budget `E` (`training.a0.max_epochs` after `--epochs`), which names the seed group of a transient experiment as `seed-N__a0ep-E`, shared by A0 and every pair. Steady runs keep `seed-N`: their epochs are not part of a name, rerunning a name continues the run ([Continue, extend and resume](#continue-extend-and-resume)) and a larger `--epochs` extends it. `--revision <label>` starts a deliberately separate run after its science changed and always comes last (`seed-N__a0ep-E__rev-<label>`, steady `seed-N__rev-<label>`; the same flag exists for `optuna`). Run names follow the paths: `<config_name>__seed-N[__a0ep-E][__rev-<label>][__<pair>]__<stage>`; trials are `<study>__trial-<NNNN>`.

## Steady flow

Steady-flow models map `(kappa, eps_bed, p_in_bc)` to `(p, u, v)`:

```bash
./scripts/slurm_ml.sh --mode gpu --partition gpu --cpus-per-task 4 \
  --mem 32G --time 04:00:00 --gres gpu:v100:1 train \
  configs/03_learning/steady_flow/experiments/model_selection/pi-fno_m48x48_h64_l4__spec.yaml \
  --train-dataset DATASET_ID
```

| Section | Meaning |
| --- | --- |
| `data` | `train_dataset` with its fixed Dataset partition, `additional_evaluation_datasets` whose declared OOD cases are evaluated in addition |
| `loss.data` | `relative_h1` (one-sided differences at the non-periodic boundaries) or `relative_l2` in normalized space |
| `loss.physics` | Three dimensionless terms, each with its own weight and linear warm-up: `momentum_weight` (Darcy–Brinkman momentum residual), `continuity_weight` (cross-section flux balance) and `boundary_weight` (inlet pressure and zero outlet gauge); `derivatives.kind` (`physical` with `extension: none`, or `spectral` with `extension: fourier_continuation`) and `interior_crop` apply to the momentum term; all maintained physics configs, finite-difference and spectral, use `interior_crop: 2`: with crop 0 the finite-difference momentum term of the exact COMSOL solution is dominated by the two no-slip wall columns (5.4e-3, against 2.7e-6 with crop 1 and 2.1e-6 with crop 2), because the Brinkman boundary layer (about 0.17 mm) is far below the grid spacing (3 mm), so the loss would mainly push the network to violate the wall values |
| `evaluation.objective` | `normalized_group_macro_rmse` over pressure and velocity groups |
| `training` | `epochs` (600 in the maintained plans), `evaluation_interval` and `additional_evaluation_interval` |

The physics terms are made dimensionless per case with scales taken from the inputs only: the mean inlet pressure Δp (the outlet gauge is zero), G = Δp/L_y and the Darcy velocity U = K_g·G/μ, where K_g is the geometric mean of √(Kxx·Kyy). The momentum term is the mean of (Rx² + Ry²)/G² over the cropped interior. The continuity term is the mean of ((q(y) − q̄)/(U·L_x))², where q(y) = ∫ v dx is the volume flux through each row: with impermeable side walls, ∇·u = 0 makes it constant over the height. Pointwise ∇·u is not used, because the COMSOL solution interpolated to the regular grid is not pointwise divergence-free (26–40 % of U/L for the reference), while its flux-balance term stays between 10⁻⁷ and 3·10⁻⁶ (six smoke cases, see [Physics diagnostics](05_evaluation.md#physics-diagnostics)). The boundary term is the inlet mismatch plus the squared outlet mean, both divided by Δp².

The starting weights (momentum 1, continuity 10³, boundary 10²) make each term comparable to the data loss at about 1 % field error. Measured on two reference cases with a smooth 1 % perturbation (16-cell correlation), the terms are about 2·10⁻², 9·10⁻⁶ and 8·10⁻⁵; for the unperturbed reference they stay below 10⁻⁵, 10⁻⁶ and 10⁻⁹.

Train, validation and ID-test come from the Dataset partition exactly as for transient drying; validation drives checkpoint selection and the scheduler, ID-test stays untouched until evaluation. Inputs and outputs are standardized per channel with statistics fitted on Train. Steady-flow runs have a single stage and live directly in their `seed-N[__rev-<label>]/` directory (see [Results layout](#results-layout)). `scheduler.patience: 20` counts validation checks: with `evaluation_interval: 5` the learning rate drops only after 100 epochs without improvement (at most about five reductions in 600 epochs). The steady Optuna studies use the same plateau floor (`min_lr` 1e-5 from `base.yaml`) and `deterministic: false` as the experiments, so a selected trial retrains under the conditions it was found in.

## Optuna and tracking

```bash
./scripts/slurm_ml.sh --mode gpu --partition gpu --cpus-per-task 4 --mem 32G \
  --time 04:00:00 --gres gpu:rtx6000ada:1 optuna \
  configs/03_learning/transient_drying/optuna/fno_a0_search.yaml \
  --train-dataset DATASET_ID
```

| Study | Trials | Search space |
| --- | ---: | --- |
| `fno_a0_search` (transient) | 30 | A0 alone (`stage_a_only`) for 100 epochs, the matched A0 budget: modes, hidden channels, layers, learning rate, batch size |
| `uno_a0_search` (transient) | 30 | As above, with modes along both axes, 5 or 7 layers and the mode ratio |
| `rno_a0_search` (transient) | 30 | As the FNO study, over the RNO's smaller ranges |
| `steady_flow/optuna/fno_search`, `uno_search` | 30 | Modes, hidden channels, layers, learning rate, batch size; U-NO also its mode ratio |
| `steady_flow/optuna/pi-fno_search`, `pi-uno_search` | 100 | As above plus the three physics weights (two decades around the starting values) and their warm-up epochs |

All studies use TPE sampling and median pruning. The transient studies tune A0 only, one study per architecture; copy the selected values into the experiment configuration. Matched plans are not an Optuna study mode, and the continuation rate of B and A1 is tuned by the [continuation sweep](#continuation-sweep-for-b-and-a1). The transient studies split each material 0.8 / 0.1 / 0.1 (40 / 5 / 5 on 50 cases); ID-test cases never enter fitting, pruning, checkpoint selection or the objective. The study signature binds Dataset, split membership, budget, search space, objective, seeds and numerical environment before trials are added; an incompatible study cannot be reopened. Interrupted trials restart as fresh trials. A trial's `best_checkpoint.pt` and `last_checkpoint.pt` are deleted when its final state is published (a trial killed before that keeps them), and their digests stay in `summary.json`; a selected trial is retrained as a normal run. `train --resume` refuses a trial directory. A trial whose exact experiment already ran (same sampled values, seed and budget) is not trained again. A trial's identity includes its epoch budget, and an ordinary run and a trial never share an identity: only a completed trial of the same study supplies its stored value, and a trial with identical values in a study of another budget trains itself. `--epochs` sets the trial epoch budget and names the study (`<study>__ep-E`), so another budget is another study beside the first; `--n-trials` only limits the trials of one invocation. Give a study with changed science its own `--revision <label>`.

A study is named after its config file and lives in `storage/03_experiments/<task>/<dataset>/optuna/<study>__ep-<E>[__rev-<label>]/` (`E` is the effective trial `training.epochs`; the SQLite study name carries the directory name) with the database `<study>.db`, `study_summary.json`, `study_sessions.json` (the source tree of each Slurm job that created or reopened it) and one directory per trial in `trials/<NNNN>__<model_label>__<name>-<value>.../`. The trial directory shows the zero-padded trial number, the sampled architecture and every sampled value that is not a model parameter, sorted by name with three significant digits.

W&B tracking uses entity `Rinovative-Hub` with project `grainlegumes-pino-drying` (steady flow) or `grainlegumes-pino-drying-transient`, in mode `online`, `offline` or `disabled`; every maintained config and study, including the transient A0 studies, uses `online`. Local configs, histories, checkpoints and study storage are authoritative; W&B observes.

<details>
<summary><strong>Measured clocks</strong></summary>

| Clock | Boundary |
| --- | --- |
| Matched `train_seconds` | Wall time of committed optimizer groups, including batch preparation and device completion; excludes loader waits, validation and publication |
| CUDA optimizer-device seconds | Sum of device events; neither process wall time nor utilization |
| `operational_timing` | Checkpoint capture and publication, selected-checkpoint evaluation |
| Attempt wall time | Monotonic clock from prepared-run entry through terminal work; accumulates across resume |

Imports, CLI resolution, scheduler wait and observer finalization lie outside these clocks.

</details>
