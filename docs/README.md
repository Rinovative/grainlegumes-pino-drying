# Documentation

The guides follow the order in which the project is used: set up, understand the physical model, generate cases, build a Dataset, train, evaluate. Each guide puts the normal path first and details in collapsible sections. For what the project decided and why, start with the [register of scientific decisions](08_scientific_decisions.md).

| Step | Guide | Purpose |
| --- | --- | --- |
| 1 | [Setup and execution](01_setup_and_execution.md) | Project layout, environments, start commands, ML jobs, notebook kernels |
| 2 | [Physical model and parameters](02_physical_model.md) | Equations, inlet schedule, drying target, parameter evidence, sampling |
| 3 | [Generation](03_generation.md) | How cases arise, fail and are replaced; Source-only campaigns, storage layout |
| 4 | [Datasets](04_datasets.md) | One plan from Generation to published Dataset and PT shards, case roles, small runs |
| 5 | [Training](05_training.md) | Transient A0/B and matched A1 plans, resume, steady flow, Optuna, tracking |
| 6 | [Evaluation](06_evaluation.md) | Artifacts, EDA, component-check and evaluation notebooks, metrics, comparisons, timing, inference |
| 7 | [Reproducibility and validation](07_reproducibility_and_validation.md) | Identities, revision rule, frozen source trees, Dataset rebuild, choice of checks |
| 8 | [Scientific decisions](08_scientific_decisions.md) | Every scientific choice with its reason, owning guide and config, and the open choices |

- [Config map](../configs/README.md): which file you edit for what; maintained YAML under `configs/` owns the scientific values, the guides explain their meaning.
- [VP1 report](Albertin_2026_PINO_Airflow_PorousMedia.pdf) and figures: the steady airflow foundation; its [corrigendum](Albertin_2026_PINO_Airflow_PorousMedia_corrigendum.md) lists what is wrong in it and what VP2 does instead.
- [Project overview](../README.md): summary of the project. Computation runs in Slurm allocations.
