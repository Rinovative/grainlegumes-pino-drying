# GrainLegumes-PINO-Drying

Canonical checkout: `repo/` below the outer ICE workspace. Read `.agents/project/README.md`; resolve canonical records through the global project-workflow skill, including from isolated copies. Portable project identity, task contracts, decisions and reconciled outcomes live in `.agents/project/`. Common lifecycle procedures belong to the global skill.

Preserve governing equations, train/validation/test/OOD separation, Dataset/run/split/grid identity, fitted preprocessing, provenance and fail-closed publication/resume. Curated scientific authority is README and the relevant Generation, transient-training and Evaluation guides under `docs/`; maintained YAML owns current scientific values.

Source is in repo, durable research artifacts in sibling storage, and environments/logs in sibling runtime. Generation uses `../runtime/venvs/native` (Python 3.12); ML uses Slurm/Apptainer. `pyproject.toml`/`uv.lock` own dependencies. Full Generation tests require a CPU Slurm allocation. A source copy must verify actual imports and maintained environment/storage/output resolution before execution. Workflow helpers use system Python and tiny fixtures. Runtime `codex/` may contain durable pending work; preserve it.
