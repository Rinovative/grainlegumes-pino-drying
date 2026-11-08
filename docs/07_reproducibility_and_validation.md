# Reproducibility and validation

Every Source, Dataset and run records its exact inputs, code and environment, and can be inspected and rebuilt from retained evidence. This guide explains those identities, the rebuild path and how to choose software checks for a change.

[Documentation index](README.md)

## Identity

| Object | Identity binds | Provenance only |
| --- | --- | --- |
| Source | Resolved physical configuration, seeds, case and input identities, COMSOL template, recorded producer digest, numerical environment | Readable Source name, storage path |
| Dataset | Ordered Source cases, grid, channel and conversion contracts, roles and split membership, recorded conversion digest | Dataset name, directory, file names, Git HEAD, selection method, recorded environment |
| Run | Resolved scientific and training configuration, exact Dataset membership, seed, recorded producer digests (`training_sha256`, `model_sha256`) | Output paths, tracking settings, Git provenance |
| Optuna study | Dataset, split membership, budget, search space, objective, seeds, numerical environment | Study storage location |

Schema versions describe file formats; content hashes identify bytes and detect drift. Neither replaces the explicit compatibility contracts. Ordinary reads never rewrite admitted evidence.

### Code identity

A persisted result is evidence of the code that produced it. Code identity uses conservative syntax-tree hashes of declared code projections, and three operations never mix:

1. Reading a persisted result (historical admission) compares it only with its own recorded evidence and the recorded identities of its inputs, never with the running code. Compatible code changes never waive the schema, task, Dataset, grid, numerical-environment or integrity contracts.
2. Asking a command for a result that is already in place, complete or as a complete stage (requested-result freshness), requires the recorded producer digest to equal the running projection; otherwise the command refuses with its rebuild remedy: `--rebuild` for artifacts, `--revision <label>` for runs and Sources.
3. Extending persisted computation or using persisted computational state (execution continuation: resume, extension, the A0 handoff to B and A1, loading weights for inference and artifacts, a partial transient artifact stage, a frozen Generation campaign) requires the recorded digest of the continued computation to equal the running projection.

Composite signatures keep their complete comparison (Optuna study signature, Generation campaign digest, checkpoint identity). The third-party numerical environment (neuraloperator digest, container image) keeps its own comparison at execution boundaries. The frozen source tree is execution binding and provenance only; `--continue-with-current-code` switches it and waives none of these checks.

Each owning module declares its projection with `common.code_identity.CodeProjection` (directories, files, a filename filter for the other task, exemptions with reasons, composition); nothing is discovered at run time:

- Generation input and Source producers, and the conversion code of each Dataset view.
- Model projection per task (`learning.training.checkpoint.MODEL_PROJECTIONS`): what prediction from saved weights runs (model construction, tensorization, scaling application and their contracts).
- Training projection per task (`TRAINING_PROJECTIONS`): includes the model projection and adds loss, loop, curriculum and handoff.
- Analysis projection per task (`analysis.artifacts.contracts.ANALYSIS_PROJECTIONS`): includes the model projection and adds metrics, validity and artifact production.
- Optuna study projection: its own modules plus the training projection of the study's task.

Each projection also hashes the `__init__.py` of every member's package and of every parent package up to `src/__init__.py`, because an initializer decides which module an alias or lazy name dispatches to. Comments, docstrings and formatting are ignored. The shared run and configuration loaders and the Dataset reproduction module are hashed in full, so an unrelated change inside them can refuse a continuation. CLI and documentation modules are excluded.

The digests are recorded where the result is written: the `producer` section of a run's `config.yaml` (`training_sha256`, `model_sha256`; stamped when the configuration is resolved, refused when authored), `producer.analysis_sha256` in the artifact provenance, `producer.model_sha256` in the scaling envelope (also bound by its cache key), the conversion digest in the Dataset manifest and `producer_identity.source_implementation_sha256` in the Generation plan.

Consequence: after an edit to the loss, completed runs stay readable and evaluable and their artifacts stay valid, but resuming or extending such a run is refused ("finish it in its bound source tree or start a new run with --revision <label>"). After an edit to the model code, loading the run's weights is refused as well.

`tests/data/test_code_projections.py` keeps every projection complete: each static import of each member must be a member, shared infrastructure (`INFRASTRUCTURE_MODULES` and `INFRASTRUCTURE_DIRECTORIES`), an exemption with a stated reason, or the other task's dispatch path. Package aliases are resolved through the runtime bindings of the package initializer (a TYPE_CHECKING-only alias never overrides them), and a name that an initializer binds to different modules at runtime fails the suite as a contradictory alias. A new scientific dependency therefore fails the suite until it is added to the projection or exempted. A projection digest says whether the code behind a recorded result is the running code; the exact executed bytes are bound separately by the source snapshot and the frozen tree, and a refusal leaves historical results intact because their exact producing source stays the reproduction authority.

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

Runs persist their full effective configuration, seeds, case split (`split_indices.pt`), Train-only fitted scaling (`normalizer.pt`), selected checkpoints and runtime evidence. Rerunning a run continues it after validating the saved scientific and continuation contract and its recorded training digest against the running code ([Code identity](#code-identity)); a completed run stays readable and evaluable after code changes. A duration can only stay equal or increase where the workflow allows it, and an identical experiment never runs twice. A0, B and A1 keep their handoff and matched-time contracts ([Training](04_training.md)).

Both tasks persist one split schema (`experiment_case_split`). Its shared core binds the task contract, the training Dataset's fixed case partition and its digest, and for the training Dataset and every additional Evaluation Dataset the manifest identity (Dataset ID, manifest digest, channel contract digest, ordered cases) plus the identity its payload states (steady flow: content fingerprint and grid; transient drying: index digest and configured horizon). Train, validation and ID-test carry ordered case IDs and a membership digest; each additional Evaluation Dataset keeps its own identity, declared OOD cases, membership digest and evaluation assignment. The only task extension is the transient item expansion (tensorizer selection, Train and evaluation sampling, spatial view). One admission reads every saved split: resume, artifacts, inference and completed-run validation. A run replays a saved split only when it equals the split rebuilt from the current Dataset manifests, so a split of another manifest digest or partition is refused.

`normalizer.pt` is one envelope for both tasks (`train_scaling_artifact`): the task's fitted state and its identity, which binds the task contract, the training Dataset's manifest and payload identity, the Train membership digest and the preprocessing specification. Scaling is fitted on Train membership only. Identical identities share one cache entry under `03_experiments/.state/scaling_cache/`, keyed by the identity digest and the run's recorded model digest (`producer.model_sha256`, the model projection of its task) and published atomically under a lock, so changed model code refits once per Dataset instead of reusing an entry, while a loss-only change reuses it. The envelope records the model digest and a cache read validates key against envelope; `normalizer.pt` keeps a data-only identity and a completed run stays admissible after code changes. An entry whose identity differs from the expected one is refused, never refitted silently. Resume, the B/A1 handoff and completed-run validation admit the saved artifact under the identity derived from the saved split, the same way for both tasks. Split files, normalizers, manifests and studies of earlier contracts are not read; rebuild the Dataset or start a new run instead.

A published result keeps the exact source snapshot or release, resolved configuration, Source and Dataset evidence, experiment bundles and required software and model dependencies.

Generation and Dataset drivers are submitted from the login node and run as Slurm jobs from the read-only source tree `runtime/sources/COMMIT12-FINGERPRINT12/`, restored once from the admitted snapshot (uncommitted changes included); an existing campaign continues in its bound tree unless `--continue-with-current-code` is given. Workers verify its bytes, permissions and membership against `storage/source_snapshots/COMMIT12-FINGERPRINT12/` before and after the command and before publication, and compare the snapshot's commit and fingerprint with the values admitted at submission; compute nodes need no Git ([Generation reference](06_generation_reference.md)). A Source keeps its machine evidence in reserved `_record/` folders (plan, selected pool and execution ledger at campaign level, reservation, decision and attempts per case), and Dataset inspection and rebuild read each case through its storage-relative locator `01_generation/TASK/SOURCE_NAME/MATERIAL/case_NNNN/result/case.h5` and the plan at `_record/campaign_plan.json`. Driver, node-job and helper logs stay in `runtime/logs/` and are hard-linked into the campaign's `logs/` (never into a published Dataset folder, which is complete and immutable once it is renamed into place, while the driver keeps appending to its log); the links are evidence for readers, not part of any identity or hash ([Logs](06_generation_reference.md#logs)). ML jobs (`scripts/slurm_ml.sh`) run the same way: the worker verifies the tree before the container starts, each runtime session records the source commit, worktree digest and `source_transition`, and a run, pair or study continues in the tree it last ran in unless `--continue-with-current-code` switches it (the digest checks of [Code identity](#code-identity) still apply; [Training](04_training.md#continue-extend-and-resume)).

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
| Experiment membership, scaling, loader | [Split admission and leakage](../tests/datasets/test_split_admission.py), [transient runtime split](../tests/datasets/test_runtime_transient_training.py), [data pipeline](../tests/experiments/test_data_pipeline_validation.py) (steady flow; it refuses a transient request, whose group scaling the transient loaders admit), [configuration semantics](../tests/experiments/test_config_semantics.py), then the CPU lifecycle node | Resolved roles, leakage rejection, Train-only scaler identity, PT and physical view parity |
| Optuna trial lifecycle | [Optuna lifecycle](../tests/experiments/test_optuna_lifecycle.py), then the one-trial studies in the [steady-flow](../tests/integration/test_steady_flow_lifecycle.py) and [transient](../tests/integration/test_transient_training_lifecycle.py) lifecycle tests | One terminal record per outcome for both tasks, best evidence bound before the best is deleted, dry-run signature equal to the recorded one, completed trial admitted without patches |
| A0/B/A1, checkpoint, budget | [Matched budget and resume](../tests/experiments/test_transient_matched_a1_b.py), [training lifecycle](../tests/integration/test_transient_training_lifecycle.py) | Recovery arithmetic and `budget_unmet` states; on a GPU also training, measured B time, matched A1, shared A0, checkpoint, inference and offline tracking |
| EDA or evaluation | [Transient artifacts](../tests/analysis/test_transient_evaluation_artifact.py), [matched artifacts](../tests/analysis/test_transient_matched_artifacts.py), affected EDA tests | Saved membership, evaluation relation, grid, time, units, error support, artifact admission |

Source-specific failures add [transient timing and units](../tests/generation/test_generation_transient.py), [runtime retention and cancellation](../tests/generation/test_generation_runtime.py), [source selection and input bytes](../tests/generation/test_generation_campaign_selection.py).

The smallest cross-boundary CPU check publishes Sources through a fake solver, publishes a Dataset, reopens PT and physical views and checks partition and scaling:

```bash
../runtime/venvs/native/bin/python -m pytest -q \
  tests/integration/test_transient_training_lifecycle.py::test_transient_package_pt_runtime_and_scaling_lifecycle
```

The full CPU suite (`-m "not real_data" tests`, about 9 minutes with 4 CPUs) is for an unresolved cross-subsystem risk, inside an allocation. `real_data` tests need `RUN_REAL_DATA_TESTS=1` and an explicit `STORAGE_ROOT`. Fake-solver and numerical checks establish wiring; real COMSOL execution, physical validity and scientific acceptance need their own evidence. The [quality workflow](../.github/workflows/quality.yml) runs the same static checks and the CPU suite in CI.
