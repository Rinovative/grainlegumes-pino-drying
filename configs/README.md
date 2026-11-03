# Configurations

Every YAML file starts with a `# Role:` line that says what it controls and when you edit it. This page lists the files by how often you change them.

## What you edit

| Edit | Files | Controls | Guide |
| --- | --- | --- | --- |
| For every Dataset | `02_datasets/<task>/*.yaml` | Materials and their roles, case counts, Generation sampling seed, selection seed and split of one Dataset; the plan has no name, the Dataset name is computed from it | [Datasets](../docs/03_datasets.md) |
| For every experiment | `03_learning/<task>/experiments/<family>/<model_label>[__variant].yaml` | Model, loss and the training values that differ from the task base | [Training](../docs/04_training.md) |
| For every study | `03_learning/<task>/optuna/<study>.yaml` | Trial count, sampler and pruner, the fixed experiment values and the search space of one Optuna study | [Training](../docs/04_training.md) |
| Occasionally | `03_learning/<task>/base.yaml` | Values every experiment and study of the task shares: seed, data and batch size, loss, optimizer, scheduler, training budget, tracking | [Training](../docs/04_training.md) |
| When cluster use changes | `01_generation/execution.yaml` | Slurm resources, timeouts, retries and retention of the Generation case jobs; no physics | [Generation reference](../docs/06_generation_reference.md) |
| Only with new physical evidence | `01_generation/physical_model/` | The simulated physics: grid and time (`common.yaml`), parameter definitions (`registry.yaml`), bibliography (`sources.yaml`), operation and inlet schedule (`fixed_bed.yaml`), one record per material (`materials/`), COMSOL template and exported fields per task (`profiles/`) | [Physical model](../docs/02_physical_model.md) |
| Rarely | `01_generation/standalone/` | Source-only campaigns for direct `./scripts/generation` runs, and the CPU core-scaling benchmark that informs `cores_per_case` | [Generation reference](../docs/06_generation_reference.md) |

## How the files combine

- A Dataset plan names the Generation files it uses (five `physical_model/` files plus `execution.yaml`) and adds only what defines the Dataset; material records are found by family name in `physical_model/materials/`. A change in `physical_model/` affects every Source generated afterwards.
- An experiment or study names its task base with `base:` and overrides any value by setting it itself; only the transient A0 studies have no base. Values set in neither file come from the code defaults; the resolved run configuration records all of them.
- An experiment's folder below `experiments/` is its family and its file stem must equal or extend its model label (the architecture, prefixed `pi-` and suffixed `__fd` or `__spec` for physics runs); the loader rejects other names. Family and name appear in the run path, and the stem of a study file is its study name.
- Command-line options such as `--train-dataset`, `--revision`, `--duration-hours` or `--cores-per-case` override a file for one run without changing it.
