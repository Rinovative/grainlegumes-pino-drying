# Reproducibility and validation

Every Source, Dataset and run is bound to its exact inputs, code and environment, and can be inspected and rebuilt from retained evidence. This guide explains those identities, the rebuild path and how to choose software checks for a change.

[Documentation index](README.md)

## Identity

| Object | Identity binds | Provenance only |
| --- | --- | --- |
| Source | Resolved physical configuration, seeds, case and input identities, COMSOL template, producer implementation, numerical environment | Readable Source name, storage path |
| Dataset | Ordered Source cases, grid, channel and conversion contracts, roles and split membership, converter implementation | Dataset name, directory, file names, Git HEAD, selection method, recorded environment |
| Run | Resolved scientific and training configuration, exact Dataset membership, seed, implementation | Output paths, tracking settings, Git provenance |
| Optuna study | Dataset, split membership, budget, search space, objective, seeds, numerical environment | Study storage location |

Schema versions describe file formats; content hashes identify bytes and detect drift. Neither replaces the explicit compatibility contracts. Ordinary reads never rewrite admitted evidence.

Implementation identity uses conservative syntax-tree hashes of a declared code projection: the Generation input and Source producers, the conversion code of each Dataset view, the training code of each task, and for an Optuna study its own modules plus the training projection of the study's task. Each owning module declares its projection with `common.code_identity.CodeProjection` (directories, files, a filename filter for the other task, exemptions with reasons, composition); nothing is discovered at run time. Each projection also hashes the `__init__.py` of every member's package and of every parent package up to `src/__init__.py`, because an initializer decides which module an alias or lazy name dispatches to. Comments, docstrings and formatting are ignored. The shared run and configuration loaders and the Dataset reproduction module are hashed in full, so an unrelated change inside them can reject current-code reuse. A rejection leaves historical results intact: their exact producing source stays the reproduction authority. CLI, analysis and documentation modules are excluded.

`tests/data/test_code_projections.py` keeps every projection complete: each static import of each member must be a member, shared infrastructure (`INFRASTRUCTURE_MODULES` and `INFRASTRUCTURE_DIRECTORIES`), an exemption with a stated reason, or the other task's dispatch path. Package aliases are resolved through the runtime bindings of the package initializer (a TYPE_CHECKING-only alias never overrides them), and a name that an initializer binds to different modules at runtime fails the suite as a contradictory alias. A new scientific dependency therefore fails the suite until it is added to the projection or exempted. A projection digest decides whether a unit's science changed (identity, cache reuse, `--continue-with-current-code`); the exact executed bytes are bound separately by the source snapshot and the frozen tree.

## Inspect and rebuild a Dataset

1. Inspect the publication and confirm that snapshot, base Git object, ordered Source cases, model and dependency evidence are available:

```bash
../runtime/venvs/native/bin/python -m src.datasets.dataset_packages inspect TASK DATASET_ID
```

2. Restore the producing source into a new, empty directory; Git index and history stay untouched:

```bash
../runtime/venvs/native/bin/python scripts/source_fingerprint.py . \
  --restore /absolute/storage/source_snapshots/COMMIT12-FINGERPRINT12/snapshot.json \
  /absolute/empty-recovered-source
```

The result is a verified source tree without Git metadata, with recorded permissions and internal links.

3. From the recovered source, rebuild the Dataset into new verification storage, using the original storage as a read-only Source root:

```bash
cd /absolute/empty-recovered-source
env -u PROJECT_ROOT /absolute/project/runtime/venvs/native/bin/python -m src.datasets.dataset_packages rebuild \
  /absolute/original-storage/02_datasets/TASK/DATASET_ID/dataset_manifest.json \
  --storage-root /absolute/original-storage \
  --output-directory /absolute/new-verification-output
```

`inspect` prints this sequence with the actual manifest, snapshot and dependency paths. `rebuild` freezes the manifest's ordered Sources, verifies source bytes, permissions and the recorded Python and dependency versions, and checks Dataset identity and payload bytes before publishing. It runs no COMSOL and leaves the original publication unchanged; mismatching bytes stay unpublished with a failure-evidence path. The output references the original Source root.

Missing Source bytes require the normal Generation workflow with the retained configuration, raw inputs and exact model, inside Slurm.

## Runs and publication

Runs persist their full effective configuration, seeds, case split (`split_indices.pt`), Train-only fitted scaling, selected checkpoints and runtime evidence. Rerunning a run continues it after validating the saved scientific and continuation contract; a duration can only stay equal or increase where the workflow allows it, and an identical experiment never runs twice. A0, B and A1 keep their handoff and matched-time contracts ([Training](04_training.md)).

A published result keeps the exact source snapshot or release, resolved configuration, Source and Dataset evidence, experiment bundles and required software and model dependencies.

Generation and Dataset drivers are submitted from the login node and run as Slurm jobs from the read-only source tree `runtime/sources/COMMIT12-FINGERPRINT12/`, restored once from the admitted snapshot (uncommitted changes included); an existing campaign continues in its bound tree unless `--continue-with-current-code` is given. Workers verify its bytes, permissions and membership against `storage/source_snapshots/COMMIT12-FINGERPRINT12/` before and after the command and before publication, and compare the snapshot's commit and fingerprint with the values admitted at submission; compute nodes need no Git ([Generation reference](06_generation_reference.md)). A Source keeps its machine evidence in reserved `_record/` folders (plan, selected pool and execution ledger at campaign level, reservation, decision and attempts per case), and Dataset inspection and rebuild read each case through its storage-relative locator `01_generation/TASK/SOURCE_NAME/MATERIAL/case_NNNN/result/case.h5` and the plan at `_record/campaign_plan.json`. Driver, node-job and helper logs stay in `runtime/logs/` and are hard-linked into the campaign's `logs/` (the Dataset driver's also into the published Dataset's `logs/`); the links are evidence for readers, not part of any identity or hash ([Logs](06_generation_reference.md#logs)). ML jobs (`scripts/slurm_ml.sh`) run the same way: the worker verifies the tree before the container starts, each runtime session records the source commit, worktree digest and `source_transition`, and a run, pair or study continues in the tree it last ran in unless `--continue-with-current-code` switches it ([Training](04_training.md#continue-extend-and-resume)).

## Choosing checks

Format, lint, type and notebook checks run on the login node:

```bash
../runtime/venvs/native/bin/python -m ruff check src tests scripts/check_notebooks.py scripts/check_package_install.py
../runtime/venvs/native/bin/python -m ruff format --check src tests scripts/check_notebooks.py scripts/check_package_install.py --exclude '*.ipynb'
../runtime/venvs/native/bin/python scripts/check_notebooks.py
../runtime/venvs/native/bin/python -m mypy src
../runtime/venvs/native/bin/python -m basedpyright --pythonpath ../runtime/venvs/native/bin/python
```

A few focused unit-test files without the `integration` marker can run on the login node with capped threads. Integration and lifecycle tests, broader selections and the full suite run inside a CPU allocation; GPU lifecycle checks need a typed GPU allocation:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  ../runtime/venvs/native/bin/python -m pytest -q -p no:cacheprovider TEST_FILES
srun --partition=standard --cpus-per-task=4 --mem=16G --time=00:30:00 \
  ../runtime/venvs/native/bin/python -m pytest -q -p no:cacheprovider TEST_NODES
```

Start with the checks for the changed contract, then cross the nearest persisted boundary:

| Change | First checks | Protected boundary |
| --- | --- | --- |
| Values in maintained `configs/` | [Maintained configs](../tests/experiments/test_maintained_configs.py) | Every learning config, Dataset plan and the documented matched command still pass their loaders; current values are not frozen |
| Source planning, candidate decisions, raw admission | [Generation fake pipeline](../tests/generation/test_generation_fake_pipeline.py) | Full design, successful and failed decisions, replacement, HDF5 and timing admission with only the solver substituted |
| Dataset identity, provenance, manifest | [Dataset reproduction](../tests/datasets/test_dataset_reproduction.py), then [transient training lifecycle](../tests/integration/test_transient_training_lifecycle.py) | Conflict, traversal and corruption checks; immutable Source identity and persisted reopen |
| Experiment membership, scaling, loader | [Transient runtime split](../tests/datasets/test_runtime_transient_training.py), [data pipeline](../tests/experiments/test_data_pipeline_validation.py), [configuration semantics](../tests/experiments/test_config_semantics.py), then the CPU lifecycle node | Resolved roles, leakage rejection, Train-only scaler identity, PT and physical view parity |
| A0/B/A1, checkpoint, budget | [Matched budget and resume](../tests/experiments/test_transient_matched_a1_b.py), [training lifecycle](../tests/integration/test_transient_training_lifecycle.py) | Recovery arithmetic and `budget_unmet` states; on a GPU also training, measured B time, matched A1, shared A0, checkpoint, inference and offline tracking |
| EDA or evaluation | [Transient artifacts](../tests/analysis/test_transient_evaluation_artifact.py), [matched artifacts](../tests/analysis/test_transient_matched_artifacts.py), affected EDA tests | Saved membership, evaluation relation, grid, time, units, error support, artifact admission |

Source-specific failures add [transient timing and units](../tests/generation/test_generation_transient.py), [runtime retention and cancellation](../tests/generation/test_generation_runtime.py), [source selection and input bytes](../tests/generation/test_generation_campaign_selection.py).

The smallest cross-boundary CPU check publishes Sources through a fake solver, publishes a Dataset, reopens PT and physical views and checks partition and scaling:

```bash
../runtime/venvs/native/bin/python -m pytest -q \
  tests/integration/test_transient_training_lifecycle.py::test_transient_package_pt_runtime_and_scaling_lifecycle
```

The full CPU suite (`-m "not real_data" tests`, about 9 minutes with 4 CPUs) is for an unresolved cross-subsystem risk, inside an allocation. `real_data` tests need `RUN_REAL_DATA_TESTS=1` and an explicit `STORAGE_ROOT`. Fake-solver and numerical checks establish wiring; real COMSOL execution, physical validity and scientific acceptance need their own evidence. The [quality workflow](../.github/workflows/quality.yml) runs the same static checks and the CPU suite in CI.
