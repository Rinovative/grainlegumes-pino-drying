# Setup and execution

The project runs on the OST ICE cluster. The login node is for editing, configuration, inspection and Slurm submission; Generation, Dataset assembly, training and evaluation run in Slurm allocations.

[Documentation index](README.md)

## Project layout

The project root holds three sibling folders:

| Folder | Content |
| --- | --- |
| `repo/` | Git checkout: code, configuration, notebooks and docs. Commands in these guides run from here. |
| `storage/` | Research data: `01_generation/` (physical Sources), `02_datasets/<task>/<dataset>/` (published Datasets), `03_experiments/` (runs, checkpoints, predictions), `source_snapshots/` (exact source bytes shared by all three). |
| `runtime/` | Environments (`venvs/native`), the ML container (`containers/grainlegumes-pino-drying.sif`), frozen source trees (`sources/`), caches, temporary files and job logs (`logs/`). |

`STORAGE_ROOT` and `RUNTIME_ROOT` override the default sibling locations; `PROJECT_ROOT` overrides the repository root. Open the project root in VS Code Remote SSH.

### Storage map

Every level of `storage/` has one role, and every name is computed from content or mirrors a config file, never from the order in which runs started. Hidden `.state/` folders hold locks, staging and identity registries only. In a Generation campaign the human files come first; machine records sit in the reserved `_record/` folders and job logs in `logs/`.

```text
storage/
  01_generation/
    <task>/<campaign>/                                 lentil-3+chickpea-3__ref-range__lhs-seed-9220: design size per material, domain, sampling, seed; `__h-<hours>` only for a horizon that deviates from the configured one
      <material>/case_NNNN/                            inputs/, result/ (+ evidence/); _record/ with reservation.json, decision.json, attempts/
      <material>/_record/<regime>/                     design manifests and resolved generation config
      logs/                                            hard links of the campaign's driver, node-job and helper logs
      _record/                                         campaign_plan.json, selected_pool.json, execution/ (scheduler ledger, epochs, Slurm progress, licence waits, reconciliation)
  02_datasets/
    <task>/<dataset>/                                  lentil-train40-val5-test5+chickpea-train40-val5-test5__ref-range__lhs-seed-9200
      logs/                                            hard link of the Dataset driver log, added after publication
  03_experiments/
    <task>/<dataset>/<family>/<config_name>/seed-N[__a0ep-E][__rev-<label>]/   one experiment (a0ep only transient), mirroring configs/03_learning/<task>/experiments/<family>/<config_name>.yaml
      a0/, b-a1__budget-<S>s__rollout-<L>__lr-<X>/{b,a1}/            transient stages (steady runs keep their files directly here)
    <task>/<dataset>/optuna/<study>__ep-E[__rev-<label>]/trials/<NNNN>__<model label>__<values>/
  source_snapshots/<commit12>-<fp12>/                  exact source bytes (commit, diff, untracked files) shared by Generation, Datasets and ML
runtime/
  sources/<commit12>-<fp12>/                           frozen read-only execution copies; reconstructible cache, removed when no queued or running job uses them
```

| Level | Name from | Details |
| --- | --- | --- |
| Generation campaign | Plan content: materials and design sizes, domain, sampling seed | [Source storage](06_generation_reference.md#source-storage) |
| Dataset | Realized membership: per-material Train/validation/test counts (or `ood<n>`), domain, sampling seed | [What a Dataset contains](03_datasets.md#what-a-dataset-contains) |
| Experiment run | Config path, Dataset, seed, optional revision; B/A1 pairs by budget, rollout and learning-rate factor | [Results layout](04_training.md#results-layout) |
| Optuna study and trial | Study config name; trial number, model label and sampled values | [Optuna and tracking](04_training.md#optuna-and-tracking) |

The same name never holds different content: different content under an existing name is refused. Same name, changed science: add `--revision LABEL` to the command you run. It never changes seeds, never makes identical science new, and never re-simulates a Source unless you also pass `--new-source` to the Dataset run. Identical content never exists twice: Datasets and runs are bound to their identity in `02_datasets/.state/identities/` and `03_experiments/.state/identities/`.

## Environments

| Environment | Used for | Defined by |
| --- | --- | --- |
| Native Python 3.12 (`runtime/venvs/native`) | Generation, Dataset assembly, light checks | `pyproject.toml`, `uv.lock` |
| Native COMSOL 6.4 (`module load Comsol/v6.4`) | Steady-flow and transient solves inside case jobs | `simulation/<task>/<task>_template.mph` |
| Apptainer image (`runtime/containers/grainlegumes-pino-drying.sif`) | Training, Optuna, evaluation artifacts | `container/Apptainer.def`, `uv.lock` |

Provision the native environment inside a CPU allocation:

```bash
srun --partition standard --cpus-per-task 2 --mem 8G --time 01:00:00 \
  bash scripts/provision_native.sh
```

The provisioner installs the locked dependencies with `uv` 0.11.7 into an absent `runtime/venvs/native`; an existing environment is used as it is. The ML image is built from `container/Apptainer.def` and placed at `runtime/containers/grainlegumes-pino-drying.sif`.

## Generation commands

`scripts/generation` runs the Generation CLI with the native interpreter:

| Command | Purpose |
| --- | --- |
| `run CONFIG` | Start or continue a Source campaign or benchmark by submitting its controller as a Slurm driver job (`--dry-run` prints the plan and checks readiness) |
| `status CONFIG_OR_RUN_ID` | Show campaign state |
| `cancel RUN_ID` | Cancel a campaign |
| `inputs CONFIG` | Prepare selected physical inputs without COMSOL |
| `fail-case RUN_ID MATERIAL CASE_ID --reason-code CODE --detail TEXT` | Record an inspected candidate as failed |
| `amend-replacements RUN_ID --material M --regime R --capacity K --reason TEXT` | Raise a batch's total replacement capacity, with its reason |

`run` freezes the source, including uncommitted changes, on the login node and submits the controller as its own CPU job (`--driver-partition`, `--driver-time`); the controller runs the cases with native COMSOL as job steps of `cluster.node_jobs` node jobs (default two, at most four; a per-user cap on the shared 32-core `standard` nodes; the COMSOL batch licence counts once per user across all hosts), and rerunning the same command continues the campaign. For learning data, use the Dataset plan in [Datasets](03_datasets.md); [Generation reference](06_generation_reference.md) covers Source-only campaigns.

## ML jobs

`scripts/slurm_ml.sh` submits training, Optuna, artifact generation and container probes to Slurm and runs them in the Apptainer image:

```bash
./scripts/slurm_ml.sh --mode gpu --partition gpu --cpus-per-task 4 --mem 32G \
  --time 04:00:00 --gres gpu:rtx6000ada:1 train CONFIG --train-dataset DATASET_ID
```

| Option | Meaning |
| --- | --- |
| `--mode cpu\|gpu` | CPU or GPU job; sets `--device cpu` or strict `--device cuda` |
| `--partition`, `--cpus-per-task`, `--mem`, `--time` | Slurm resources |
| `--gres gpu:TYPE:1` | GPU type `v100`, `rtxa6000` or `rtx6000ada` (GPU mode only) |
| `train`, `optuna`, `artifacts`, `probe` | Operation and its CLI arguments |
| `--continue-with-current-code` | `train` and `optuna` only: continue an existing run or study with the current code instead of the code it last ran in ([Training](04_training.md#continue-extend-and-resume)) |

The wrapper runs the configuration preflight before submission. Every job runs from a frozen, read-only copy of the code, `runtime/sources/COMMIT12-FINGERPRINT12/`, restored from the snapshot in `storage/source_snapshots/` (see [Generation reference](06_generation_reference.md)); only the launcher uses `repo/` and Git, and later edits in `repo/` do not reach submitted jobs. The worker verifies the tree against its snapshot without Git and records source commit, worktree digest, source tree and image digest with the job. Logs go to `runtime/logs/ml/slurm-<jobid>.out` and `.err`, written in append mode so a requeue never truncates them. When the container exits, the job also hard-links both files into `logs/` of every run stage (A0, B, A1), Optuna study root and artifact-processed run folder it wrote, so they outlive the cleanup of `runtime/logs/` and go away with their data; a link that cannot be made is a warning at the end of the `.err`. A job killed outright (SIGKILL, node loss) cannot link and leaves its logs in `runtime/logs/` only. `probe` checks the container and the ML CLIs without opening Datasets.

To run a single command inside the image, for example a preflight, use `scripts/apptainer_exec.sh`:

```bash
./scripts/apptainer_exec.sh python -m src.experiments.cli.cli_config_preflight train \
  configs/03_learning/transient_drying/experiments/architecture_comparison/fno_m64x80_h64_l3.yaml --train-dataset DATASET_ID
```

Online W&B tracking needs `WANDB_API_KEY` in the environment or a readable `~/wandb_key.txt`; the worker passes it into the container.

## Notebooks on compute nodes

Notebooks are viewers and light orchestration; scientific cells need a kernel on an allocated compute node. Selecting an interpreter or opening a notebook on the login node does not move a running kernel. Check the kernel before running scientific cells:

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

Compare host and job ID with the active allocation (`squeue --me`) and confirm the interpreter. If the kernel's host cannot be verified, leave scientific cells unexecuted and use the CLI path inside an allocation.
