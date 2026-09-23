# Generation Operations

Generation uses the shared ICE repository, sibling durable storage, and native
Slurm/COMSOL execution. Scientific parameter meanings remain in the
[scientific parameter reference](generation_parameter_reference.md), and
current values remain in validated YAML under `configs/generation`.

## Setup and entry points

Work from `repo/` on ICE. The sibling roots are `../storage` for
scientific data, case results, receipts, manifests, and Dataset packages, and
`../runtime` for the replaceable Python environment, caches, and Slurm
logs. No second checkout, self-SSH, rsync, Docker, or Conda runtime is used.

Provision `../runtime/venvs/generation` once in a Slurm CPU allocation
from the repository's Python 3.12 `uv.lock`. Full project dependencies
are required because Generation package publication imports PyTorch.

```bash
module load Python/3.12 uv/0.11
export UV_PROJECT_ENVIRONMENT="$(realpath -m ../runtime/venvs/generation)"
export UV_CACHE_DIR="$(realpath -m ../runtime/uv/cache)"
uv sync --locked --no-dev --python "$(command -v python3)"
```

Use `standard` for this environment setup and substantial CPU work.
Do not install the full dependency set on the login node. COMSOL is loaded
natively on compute nodes with `module load Comsol/v6.4`; it is not in
the ML Apptainer image.

```bash
./scripts/generation_workflow.sh run CONFIG --dry-run
./scripts/generation_workflow.sh run CONFIG --preflight-only
./scripts/generation_workflow.sh run CONFIG
./scripts/generation_workflow.sh run CONFIG --background
./scripts/generation_workflow.sh status CONFIG_OR_RUN_ID
./scripts/generation_workflow.sh cancel RUN_ID
./scripts/generation_workflow.sh smoke
```

The optional `smoke` command submits one disposable COMSOL batch
acceptance job with one CPU, no GPU GRES, and no scientific publication.
It defaults to the `gpu` partition for a lightweight probe; the
`standard` partition can be requested explicitly. Slurm logs go to
`../runtime/logs/generation`. Normal Generation cases use
`standard` or `long` according to their configured runtime.
No physical node is selected by the workflow.

| Workflow | Configuration |
| --- | --- |
| Paired technical smoke | `configs/generation/workflows/technical_smoke.yaml` |
| Transient core benchmark | `configs/generation/benchmarks/transient_core_scaling/suite.yaml` |
| All-material pilot | `configs/generation/campaigns/transient_drying/material_pilot.yaml` |
| Transient production | `configs/generation/campaigns/transient_drying/family_generalization.yaml` |
| Airflow ID Dataset | `configs/generation/campaigns/steady_flow/id_dataset.yaml` |

## Source admission and lifecycle

`run CONFIG` is the maintained start and resume entry point. It requires
the current shared HEAD to be clean and committed; `--git-commit` may
assert that same revision but cannot select a different checkout. The
controller fingerprints the source. Workers check the exact commit and
fingerprint before execution; the Python orchestration worker checks again
afterward. Publication checks the fingerprint immediately before its atomic
write. A changed
source fails closed. Keep the shared checkout unchanged while jobs are active;
to resume an older run, check out its admitted revision after dependent jobs
have finished. A dirty worktree cannot be labeled with a clean commit for
scientific publication.

Each config resolves to a deterministic Generation run plan. The controller
prepares missing canonical inputs, submits eligible Slurm cases through the
existing Python admission owner, monitors jobs and case evidence, validates
native COMSOL results, binds the in-place campaign or benchmark publication,
and runs the existing Dataset and finalizer services. Repeating the config
reconciles active ownership and submits only eligible missing work. Slurm
`COMPLETED` alone is never scientific success.

Immutable inputs keep their `input_generation_id` and source commit.
Cross-commit input reuse requires exact scientific, template, schema, batch,
and ordered case-membership compatibility. The execution run identity still
includes its execution commit. Solver outputs are not reused merely because
inputs are compatible. Source, template, config, input, case, and publication
hashes remain owned by the Python services.

`--background` changes only the controller's ownership to a `tmux`
session. Slurm still owns the case jobs. A login-host reboot can end the
controller; rerun the same config to reconcile durable state.

```bash
./scripts/generation_workflow.sh run CONFIG --background
./scripts/generation_workflow.sh background-status "$WORKFLOW_SESSION_ID"
./scripts/generation_workflow.sh background-list
```

## Partial campaigns and completion

A genuine terminal partial campaign retains successful cases, failed-case
evidence, and original membership. Complete and partial publication receipts
bind exact in-place inventories under the sole shared storage root. There is
no transfer copy or remote-source cleanup. Package and finalizer validation
still fails closed on missing or conflicting evidence.

Deterministic completion uses the same interface:

```bash
./scripts/generation_workflow.sh run CONFIG --replacement-pool-size N
./scripts/generation_workflow.sh run CONFIG \
  --replacement-pool-size N --parent-run-id PARENT_RUN_ID
```

`N` is a cumulative high-water mark, not an additional count per
invocation. The replacement pool extends a stable deterministic candidate
prefix. A failed candidate releases only its own material's deficit slot;
successful original cases are never rerun. Compatible parents are selected by
validated structure, not filename or modification time. Ambiguous matches
require `--parent-run-id`. Parent partial evidence is bound within shared
storage. After all deficits are covered, the existing Python services build
the exact composite, Dataset packages, transient PT shards, loader smoke,
readiness evidence, and final receipts. Failed sources remain visible and
the sole durable source is retained.

## COMSOL and failure behavior

Each case loads `Python/3.12` and `Comsol/v6.4` in its Slurm
allocation. The native worker invokes the maintained Generation Python CLI;
the Python runtime constructs and executes `comsol batch` with the
existing inputs, `-batchlog`, and `-batchlogout`. The worker
uses unique node-local scratch and removes it after case completion. COMSOL
exit status and license errors propagate into the existing attempt/failure
evidence; no shell retry can publish duplicate science.

Temporary license capacity is recognized only by the existing strong
feature-bearing classifier. The first checkout after allocation is immediate;
strong pre-solve capacity failure retries in the same allocation according to
the configured `in_allocation_retry` window and pause. Unknown, expired,
missing, or misconfigured licenses remain hard failures. License-only blocks
are operational evidence, not scientific failed attempts. Solver, conversion,
publication, and replay failures retain their stage-specific evidence. One
case failure does not cancel unrelated runnable cases. Duplicate job ownership,
incompatible identities, unsafe paths, and conflicting hashes fail closed.

`maximum_failed_cases` remains inclusive: the circuit opens at the
next genuine solver failure. Slurm pending time does not enter the license
window. Do not cancel or move a valid pending job based only on estimated
start time.

A finalized `comsol_batch.log` is parsed once into compact timing
evidence. `comsol_process_seconds` is whole-process wall time; the
stationary and transient scientific solver times come only from their matched
top-level COMSOL solver blocks. Missing or ambiguous timing remains
unavailable without reclassifying a valid scientific result.

## Input-only generation and status

Normal `run CONFIG` prepares missing canonical inputs. For a bounded
input-only operation, the maintained wrapper admits the current clean source
and runs the Python CLI inside a CPU Slurm allocation:

```bash
./scripts/generation_workflow.sh inputs "$CAMPAIGN_CONFIG" \
  --only-batch "$BATCH_NAME" --case-start 1 --case-count "$CASE_COUNT"
```

The wrapper uses the configured native Python 3.12 environment, the current
source commit, and sibling storage. Input EDA reads only admitted canonical
manifests.

```bash
./scripts/generation_workflow.sh status CONFIG_OR_RUN_ID
./scripts/generation_workflow.sh cancel "$GENERATION_RUN_ID"
./scripts/generation_workflow.sh cancel "$GENERATION_RUN_ID" --force
```

The first Ctrl+C after campaign launch requests graceful cancellation of
owned work; the second requests force cancellation. Before a run identity
exists, cancellation is not armed. Status reports bounded successful and
failed case views, admission occupancy, license blocks, and incomplete work
without reading large HDF5/CSV payloads.

## Persistence

`../storage/01_generation` owns canonical inputs, cases, attempts, run
and pilot evidence; `../storage/02_datasets` owns immutable Dataset
packages. Packages reference validated `case.h5` sources rather than
copying them. The same-root campaign receipt binds an exact inventory and
preserves the existing atomic publication and collision checks. The pilot
uses a marked accounting workspace containing no copied scientific cases,
then removes that workspace after its finalizer; the canonical source stays
in storage. Benchmark validation uses its exact same-root validator. Slurm
stdout/stderr and replaceable cache belong under `../runtime`.

The core benchmark still uses its configured cases and variants. Its primary
measurement is successful COMSOL process time; queue, license wait, failed
checkout, conversion, publication, and controller time remain separate. It
reports a recommendation without editing production configuration.
