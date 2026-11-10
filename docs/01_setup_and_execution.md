# Setup and execution

The project runs on the OST ICE cluster. The login node is for editing, configuration, inspection and Slurm submission; Generation, Dataset assembly, training and evaluation run in Slurm allocations.

[Documentation index](README.md)

## Project layout

| Folder | Content |
| --- | --- |
| `repo/` | Git checkout: code, configuration, notebooks and docs. Commands in these guides run from here. |
| `storage/` | Research data: `01_generation/` (physical Sources), `02_datasets/<task>/<dataset>/` (published Datasets), `03_experiments/` (runs, checkpoints, predictions), `source_snapshots/` (exact source bytes shared by all three). |
| `runtime/` | Environments (`venvs/native`), the ML container (`containers/grainlegumes-pino-drying.sif`), frozen source trees (`sources/`), caches, temporary files and job logs (`logs/`). |

`STORAGE_ROOT` and `RUNTIME_ROOT` override the default sibling locations; `PROJECT_ROOT` overrides the repository root. Open the project root in VS Code Remote SSH.

### Storage map

Every level of `storage/` has one role, and every name is computed from content or mirrors a config file, never from the order in which runs started. The same name never holds different content: different content under an existing name is refused.

<details>
<summary>Storage tree, naming levels and identity</summary>

Hidden `.state/` folders hold locks, staging and identity registries only, one per stage (`01_generation/.state/`, `02_datasets/.state/`, `03_experiments/.state/`); a Generation campaign keeps none of its own. Lock files are empty anchors, are never deleted during operation and go with their folder. A Generation campaign holds exactly three entries, the human files first: `cases.csv`, `materials/` and `_record/` (machine records and `logs/`); the `_record/` folders of a material and of a case stay with them.

```text
storage/
  01_generation/
    <task>/<campaign>/                                 lentil-32-8-8+chickpea-32-8-8+kidney-bean-16__ref-range__sobol-seed-9200: design sizes per material, domain, sampling, seed; `__h-<hours>` only for a horizon that deviates from the configured one
      cases.csv                                        derived one-row-per-candidate view, written when the campaign completes
      materials/<material>/case_NNNN/                  inputs/, result/ (+ evidence/); _record/ with reservation.json, decision.json, attempts/
      materials/<material>/_record/<regime>/           design manifests and resolved generation config
      materials/<material>/_record/amendments/         the material's minimum success rate amendments (amendment_NNNN.json)
      _record/                                         campaign_plan.json, selected_pool.json, logs/ (hard links of the campaign's driver, node-job and helper logs), execution/ (scheduler ledger, epochs, Slurm progress, licence waits, reconciliation)
    .state/                                            locks/ (empty anchors named TASK~CAMPAIGN~...) and staging/TASK~CAMPAIGN/ of every campaign
  02_datasets/
    <task>/<dataset>/                                  lentil-train32-val8-test8+chickpea-train32-val8-test8+kidney-bean-ood16__ref-range__sobol-seed-9200
  03_experiments/
    <task>/<dataset>[__train-<N>]/<family>/<config_name>/seed-N[__a0ep-E][__rev-<label>]/   one experiment (a0ep only transient, train-N only with data.train_subset), mirroring configs/03_learning/<task>/experiments/<family>/<config_name>.yaml
      a0/, b-a1__budget-<S>s__rollout-<L>__lr-<X>[__<token>-<value>...]/{b,a1}/            transient stages (steady runs keep their files directly here)
    <task>/<dataset>[__train-<N>]/optuna/<study>__ep-E[__rev-<label>]/trials/<NNNN>__<model label>__<values>/
  source_snapshots/<commit12>-<fp12>/                  exact source bytes (commit, diff, untracked files) shared by Generation, Datasets and ML
runtime/
  sources/<commit12>-<fp12>/                           frozen read-only execution copies; reconstructible cache, removed when no queued or running job uses them
```

| Level | Name from | Details |
| --- | --- | --- |
| Generation campaign | Plan content: materials and design sizes, domain, sampling seed | [Source storage](03_generation.md#source-storage) |
| Dataset | Realized membership: per-material Train/validation/test counts (or `ood<n>`), domain, sampling seed | [What a Dataset contains](04_datasets.md#what-a-dataset-contains) |
| Experiment run | Config path, Dataset, seed, optional revision; B/A1 pairs by budget, rollout, learning-rate factor and every other B/A1 setting the config sets | [Results layout](05_training.md#results-layout) |
| Optuna study and trial | Study config name; trial number, model label and sampled values | [Optuna and tracking](05_training.md#optuna-and-tracking) |

Same name, changed science: add `--revision LABEL` to the command you run ([Revision rule](07_reproducibility_and_validation.md#revision-rule)). Identical content never exists twice: Datasets and runs are bound to their identity in `02_datasets/.state/identities/` and `03_experiments/.state/identities/`.

</details>

## Environments

| Environment | Used for | Defined by |
| --- | --- | --- |
| Native Python 3.12 (`runtime/venvs/native`) | Generation, Dataset assembly, light checks | `pyproject.toml`, `uv.lock` |
| Native COMSOL 6.4 (`module load Comsol/v6.4`) | Steady-flow and transient solves as case steps inside node jobs | `simulation/<task>/<task>_template.mph` |
| Apptainer image (`runtime/containers/grainlegumes-pino-drying.sif`) | Training, Optuna, evaluation artifacts | `container/Apptainer.def`, `uv.lock` |

Provision the native environment inside a CPU allocation, then probe the environment and the ML container:

```bash
srun --partition standard --cpus-per-task 2 --mem 8G --time 01:00:00 \
  bash scripts/provision_native.sh
../runtime/venvs/native/bin/python --version
./scripts/generation --help
./scripts/slurm_ml.sh --mode cpu --partition standard --cpus-per-task 2 \
  --mem 4G --time 00:10:00 probe
```

<details>
<summary>Provisioner rules and image build</summary>

The provisioner takes no argument and installs the locked dependencies with `uv` 0.11.7 only into an absent `runtime/venvs/native`; it refuses an existing environment (remove the old one to reprovision). The ML image is built from `container/Apptainer.def` and placed at `runtime/containers/grainlegumes-pino-drying.sif`.

</details>

## Start a workflow

| Workflow | Start command (from `repo/`) | Notebooks | Guide |
| --- | --- | --- | --- |
| Dataset (normal entry) | `../runtime/venvs/native/bin/python -m src.datasets.dataset_packages run PLAN [--dry-run]` | `01`, `02` | [Datasets](04_datasets.md#normal-workflow) |
| Source-only campaign | `./scripts/generation run CONFIG` | `01` | [Generation](03_generation.md) |
| Training | `./scripts/slurm_ml.sh ... train CONFIG --train-dataset DATASET_ID [--train-seconds S]` | `03`, `04` | [Training](05_training.md#matched-comparison-a0--b-a1) |
| Evaluation | notebooks on a compute-node kernel | `05`, `06` | [Evaluation](06_evaluation.md) |

`--dry-run` is off by default: without it `run` submits the driver job. Resource defaults of `slurm_ml.sh` are in [ML jobs](#ml-jobs).

## Generation commands

For learning data use the Dataset plan ([Datasets](04_datasets.md)). `scripts/generation run` freezes the source on the login node, submits the controller as its own CPU driver job and rerunning the same command continues the campaign ([Generation](03_generation.md#campaign-driver)).

<details>
<summary>Generation CLI commands and node-job limits</summary>

`scripts/generation` runs the Generation CLI with the native interpreter. Options of `run` are listed with their defaults in [Generation](03_generation.md#campaign-driver); the ones named here default as follows.

| Command | Purpose | Options and defaults when omitted |
| --- | --- | --- |
| `run CONFIG` | Start or continue a Source campaign by submitting its controller as a Slurm driver job | `CONFIG` required. `--dry-run`: off (the run starts); with it the command prints the plan and checks readiness. `--driver-partition`: `standard`. `--driver-time`: `06:00:00` |
| `status CONFIG_OR_RUN_ID` | Show campaign state | target required |
| `cancel RUN_ID [--force]` | Cancel a campaign | `RUN_ID` required. `--force`: off (graceful cancellation); with it the cancellation is forced |
| `inputs CONFIG` | Prepare selected physical inputs without COMSOL | `CONFIG` required; selection options and defaults in [Generation](03_generation.md#input-only-generation-and-status) |
| `fail-case RUN_ID MATERIAL CASE_ID --reason-code CODE --detail TEXT` | Record an inspected candidate as failed | all required; no default |
| `amend-replacements RUN_ID --material M --minimum-success-rate S --reason TEXT` | Lower a material's minimum success rate, with its reason | all required; no default |

`run` freezes the source, including uncommitted changes, on the login node and submits the controller as its own CPU job (`--driver-partition`, `--driver-time`); the controller runs the cases with native COMSOL as job steps of `run.node_jobs` node jobs (default two, at most four; a per-user cap on the shared 32-core `standard` nodes; the COMSOL batch licence counts once per user across all hosts), and rerunning the same command continues the campaign. [Generation](03_generation.md) covers Source-only campaigns.

</details>

## ML jobs

`scripts/slurm_ml.sh` submits training, Optuna, artifact generation and container probes to Slurm and runs them in the Apptainer image:

```bash
./scripts/slurm_ml.sh --mode gpu --partition gpu --cpus-per-task 4 --mem 32G \
  --time 04:00:00 --gres gpu:rtx6000ada:1 train CONFIG --train-dataset DATASET_ID
```

All resource options are optional; an explicit option always wins. The command above is the same as `./scripts/slurm_ml.sh train CONFIG --train-dataset DATASET_ID`.

| Option | Meaning | Default when omitted |
| --- | --- | --- |
| `--mode cpu\|gpu` | CPU or GPU job; sets `--device cpu` or strict `--device cuda` | `gpu`; `probe`: `cpu` |
| `--partition` | Slurm partition | `gpu` in GPU mode, `standard` in CPU mode |
| `--cpus-per-task` | CPUs | 4 |
| `--mem` | Memory (`<int>M`, `G` or `T`) | `32G` in GPU mode, `16G` in CPU mode |
| `--time` | Wall limit `[D-]HH:MM:SS` | `04:00:00` for `train` and `optuna`, `02:00:00` for `artifacts`, `00:10:00` for `probe` |
| `--gres gpu:TYPE:1` | GPU type `v100`, `rtxa6000` or `rtx6000ada` (GPU mode only; CPU mode refuses it) | `gpu:rtx6000ada:1` in GPU mode, none in CPU mode |
| `train`, `optuna`, `artifacts`, `probe` | Operation and its CLI arguments | required, no default |
| `--continue-with-current-code` | `train` and `optuna` only: continue an existing run or study with the current code instead of the code it last ran in; the code-identity checks still apply ([Training](05_training.md#continue-extend-and-resume)) | off: the run or study continues in the code it last ran in |

The wrapper runs the configuration preflight before submission. Every job runs from a frozen, read-only copy of the code, `runtime/sources/COMMIT12-FINGERPRINT12/`, restored from the snapshot in `storage/source_snapshots/`; later edits in `repo/` do not reach submitted jobs ([Source snapshots and frozen trees](07_reproducibility_and_validation.md#source-snapshots-and-frozen-trees)). `probe` checks the container and the ML CLIs without opening Datasets. Logs go to `runtime/logs/ml/<operation>-<jobid>.out` and `.err`.

<details>
<summary>Single command in the image, W&B key, job logs</summary>

To run a single command inside the image, for example a preflight, use `scripts/apptainer_exec.sh [--gpu] [--] COMMAND [ARGUMENT ...]`; `COMMAND` is required. `--gpu` is off by default (CPU run); with it the call needs a Slurm GPU allocation with `CUDA_VISIBLE_DEVICES`. `PROJECT_ROOT`, `STORAGE_ROOT` and `RUNTIME_ROOT` default to the repository root and its sibling `storage/` and `runtime/`.

```bash
./scripts/apptainer_exec.sh python -m src.experiments.cli.cli_config_preflight train \
  configs/03_learning/transient_drying/experiments/architecture_comparison/fno_m64x80_h64_l3_p0p1__lr-0p000388__a0smp-one-step-transition.yaml --train-dataset DATASET_ID
```

`cli_config_preflight WORKFLOW CONFIG` (`WORKFLOW` is `train` or `optuna`, both required) takes the overrides `--epochs`, `--train-dataset`, `--train-seconds`, `--rollout-length`, `--lr-scale`, `--revision`, `--seed`, `--resume` and `--train-stages`; none has a default, an omitted one leaves the configuration value in force.

Online W&B tracking needs `WANDB_API_KEY` in the environment or a readable `~/wandb_key.txt`; the worker passes it into the container.

Job logs and what the worker records:

- Only the launcher uses `repo/` and Git. The worker verifies the tree against its snapshot without Git and records source commit, worktree digest, source tree and image digest with the job.
- Logs: `runtime/logs/ml/<operation>-<jobid>.out` and `.err` (for example `train-<jobid>.out`, `artifacts-<jobid>.out`), written in append mode so a requeue never truncates them.
- When the container exits, the job also hard-links both files into `logs/` of every run stage (A0, B, A1), Optuna study root and artifact-processed run folder it wrote, so they outlive the cleanup of `runtime/logs/` and go away with their data.
- A file that is empty at that moment is not linked (typically the `.err` of a clean job). Slurm messages appended later, for example an out-of-memory kill notice, stay only in `runtime/logs/ml/<operation>-<jobid>.err`.
- A link that cannot be made is a warning at the end of the `.err`. A job killed outright (SIGKILL, node loss) cannot link and leaves its logs in `runtime/logs/` only.

</details>

## Notebooks

One notebook per step, in workflow order. Each guide names its notebooks where they are used.

| Notebook | Step | Use |
| --- | --- | --- |
| [`01_eda_generation_inputs`](../notebooks/01_eda_generation_inputs.ipynb) | [Generation](03_generation.md) | Prepared Generation inputs before COMSOL: parameters, fields, schedules, design coverage, failures by parameter |
| [`02_eda_datasets`](../notebooks/02_eda_datasets.ipynb) | [Datasets](04_datasets.md) | Completed cases and Datasets of both tasks: states, static fields, schedules, target attainment, solver timing |
| [`03_component_checks`](../notebooks/03_component_checks.ipynb) | [Training](05_training.md) | Training-free component priors and physics checks on the COMSOL reference fields of one Train or validation membership |
| [`04_training_pipeline`](../notebooks/04_training_pipeline.ipynb) | [Training](05_training.md) | Training configurations, data roles, objectives, stage dependencies and saved runs (inspection only) |
| [`05_eval_single_model`](../notebooks/05_eval_single_model.ipynb) | [Evaluation](06_evaluation.md) | One saved run |
| [`06_eval_comparison_models`](../notebooks/06_eval_comparison_models.ipynb) | [Evaluation](06_evaluation.md) | Two or more compatible saved runs |

## Notebooks on compute nodes

Notebooks are viewers and light orchestration; scientific cells need a kernel on an allocated compute node. Selecting an interpreter or opening a notebook on the login node does not move a running kernel. Check the kernel before running scientific cells; if its host cannot be verified, leave scientific cells unexecuted and use the CLI path inside an allocation.

<details>
<summary>Kernel check cell</summary>

```python
import os
import socket
import sys

print({
    "kernel_host": socket.getfqdn(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "python": sys.executable,
})
```

Compare host and job ID with the active allocation (`squeue --me`) and confirm the interpreter.

</details>
