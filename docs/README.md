# Documentation

The guides follow the order in which the project is used: set up, understand the physical model, build a Dataset, train, evaluate. Guides 06 and 07 are references for Generation details, reproduction and checks. Each guide puts the normal path first and details in collapsible sections.

| Step | Guide | Use it for |
| --- | --- | --- |
| 1 | [Setup and execution](01_setup_and_execution.md) | Project layout, environments, Generation and ML job commands, notebook kernels |
| 2 | [Physical model and parameters](02_physical_model.md) | Equations, inlet schedule, drying target, parameter evidence, sampling |
| 3 | [Datasets](03_datasets.md) | One plan from Generation to published Dataset and PT shards, split, small runs |
| 4 | [Training](04_training.md) | Transient A0/B and matched A1 plans, resume, steady flow, Optuna, tracking |
| 5 | [Evaluation](05_evaluation.md) | Artifacts, EDA and evaluation notebooks, metrics, comparisons, timing, inference |
| 6 | [Generation reference](06_generation_reference.md) | Source-only campaigns, input preparation, failures and continuation, storage layout |
| 7 | [Reproducibility and validation](07_reproducibility_and_validation.md) | Identities, Dataset rebuild, choice of checks |

Maintained YAML under `configs/` owns the scientific values; [the config map](../configs/README.md) says which file you edit for what, and the guides explain their meaning. Computation runs in Slurm allocations.

The [VP1 report](Albertin_2026_PINO_Airflow_PorousMedia.pdf) and the figures describe the steady airflow foundation; the [project overview](../README.md) summarizes the project.
