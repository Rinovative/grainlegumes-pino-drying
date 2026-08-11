# Generation Operations and Campaign Workflow

## Quick start

Run public workflow commands from the bare `hpc115` GPU/development host in
the repository checkout. The bare host is a shell control plane: it needs Bash,
Git, SSH, rsync, Docker, and ordinary core utilities, but no Generation Python
venv or scientific Python packages. `scripts/generation_workflow.sh` sends every
local Python operation through `scripts/docker_python.sh`, which starts the
canonical `grainlegumes-pino-drying` image with `/workspace/repo` mounted
read-only and `STORAGE_ROOT` mounted at `/workspace/storage`.

The wrapper resolves the bare-host checkout with Git; the checkout may live at
any host-native absolute path. Campaign and benchmark arguments remain
repository-relative logical paths such as
`configs/generation/campaigns/steady_flow/technical_smoke.yaml`. The wrapper
admits each file against the host checkout, translates it to
`/workspace/repo/<logical-path>` only at the Docker boundary, and translates it
to the CPU checkout only at the SSH boundary. `/workspace/repo` and
`/workspace/storage` are Docker-only mount destinations, never bare-host path
requirements. Users do not manually translate configuration paths.

The wrapper owns non-interactive SSH and rsync to `sricehpc01`. Its native
`.[generation-cpu]` venv plans Slurm work, while CPU compute nodes materialize
case inputs, run `Comsol/v6.4`, convert and validate results, and publish durable
CPU evidence. Validated publications return to canonical storage on `hpc115`,
where Docker validates Generation evidence and builds Dataset packages. Docker
is not required on the CPU cluster.

> Configured scientific values are modelling and sampling decisions. A citation
> does not imply that every final number appears verbatim in its source. The
> authoritative interpretation is the resolved `evidence` classification,
> source references, and any explicit method or applicability limit; technical
> runtime evidence does not constitute experimental validation.

## Where do I change what?

> User decisions live in config. Rules and invariants live in code. Derived
> outputs are not separately authored.

| Intent | Canonical owner | What the user changes | Derived automatically |
| --- | --- | --- | --- |
| Domain dimensions and grid resolution | `configs/generation/common.yaml` (`grid`) | `Lx`, `Ly`, `Lz`, `nx`, `ny`, boundary inclusion | `dx`, `dy`, coordinates, shapes |
| Regular output interval and maximum horizon | `configs/generation/common.yaml` (`time`) | `start`, `interval`, `stop`, step and exact-stop policy | Regular time axis and diagnostic-stop handling |
| Shared fixed scientific values | `configs/generation/common.yaml` (`scientific_fixed_values`) | Validated value, unit, and provenance | Formula-derived fixed values such as `U_wall` |
| Parameter definitions, transforms, sampling order, and OOD policy | `configs/generation/registry.yaml` | Typed parameter and OOD declarations | Profile projections, active blocks, dimensions, eligible OOD units |
| Material values, supports, evidence, targets, complete records | `configs/generation/materials/<family>.yaml` | One role-neutral family record | Effective registry, material digest, atomic OOD admission |
| Operation and boundary-condition ranges | `configs/generation/operations/fixed_bed.yaml` | Natural/OOD supports, nominals, schedules, constraints | Case schedules and operation digest |
| Profile I/O, mappings, exports, template identity | `configs/generation/profiles/steady_flow.yaml` or `transient_drying.yaml` | Explicit mapping state and profile contract | Runtime adapter and export admission plan |
| Material roles and family OOD | Selected `configs/generation/campaigns/<profile>/<campaign>.yaml` | `material_roles` | Evaluation regimes and package materials |
| Source-case counts | Production campaign `sampling.counts` | Counts by regime and family | Total, batches, indices, OOD allocation |
| Train, validation, ID membership | Production campaign `membership.per_seen_material` | Split counts | Split membership and package eligibility |
| Campaign and membership seeds | Selected campaign YAML | `sampling.seed_base`, `membership.seed`, paired seed | Batch, block, permutation, and case seeds |
| Technical-smoke cases | Both profile `technical_smoke.yaml` files | Natural counts and paired seed | Paired two-profile plan and retained packages |
| Pilot cases | `configs/generation/campaigns/transient_drying/pilot_check.yaml` | `sampling.cases_per_material` and seed | Default 18; fast override is 6 |
| Transient core benchmark | `configs/generation/benchmarks/transient_core_scaling/` | Shared case, repetitions, four resource variants | Same-case execution IDs and isolated scaling evidence |
| Dataset package requests | Campaign `dataset_packages` | Source role and evaluation regime | Package names, materials, views, counts |
| Per-case Slurm allocation | `configs/generation/execution/cluster_cpu.yaml` (`cluster`) | `cores_per_case`, wall time, scheduler options | One ordinary non-exclusive case-job request |
| Submission feeder | `configs/generation/execution/cluster_cpu.yaml` (`submission`) | Pending buffer, poll interval, optional running safety cap | Durable one-at-a-time queue feeding |
| CPU site, modules, partition | `configs/generation/execution/cluster_cpu.yaml` (`site`) | Site values only when infrastructure changes | Remote setup, module loads, executables |
| Timeout and failure policy | `configs/generation/execution/cluster_cpu.yaml` (`runtime`) | Timeout, maximum failures, extra arguments | Runtime and campaign stop behavior |
| Solved-model and raw-CSV retention | `configs/generation/execution/cluster_cpu.yaml` (`retention`) | Purpose-specific booleans | Per-case retained evidence |
| CPU source retention | Workflow invocation | Only `--keep-cpu-source` | `all` and pilot clean by default; smoke retains |
| Transfer and storage | `STORAGE_ROOT` and wrapper-managed remote layout | Optional `STORAGE_ROOT` and `--remote-root` | Staging, rsync, atomic publication, receipts |
| Source bibliography | `configs/generation/sources.yaml` | Source key and bibliographic metadata | Reference resolution and inspection evidence |

Scientific definitions, ranges, equations, provenance classifications, and
source references are documented only in
`docs/generation_parameter_reference.md`.

## Host responsibilities

| Host | Responsibility | User interaction |
| --- | --- | --- |
| `hpc115` bare host | Shell control plane, exact Git state, Docker invocation, SSH, rsync, and canonical storage path | Type every `generation_workflow.sh` command here; do not maintain a native Generation venv |
| `hpc115` Docker | Config resolution, local readiness/reporting, transferred HDF5/evidence admission, Dataset package building, and loader smokes | Wrapper-owned ephemeral `docker run`; no container ID or running-container name is required |
| `sricehpc01` login | Exact checkout, native `.[generation-cpu]` venv, compact plans/manifests, Slurm submission/status, and durable CPU storage | Wrapper-owned batch SSH and rsync; manual SSH only for requested evidence inspection |
| CPU Slurm compute node | Per-case `$TMPDIR` materialization, native COMSOL, export collection, HDF5 conversion/admission, and durable result publication | Scheduler-owned; no Docker and no manual login in the normal workflow |

### Prerequisites by execution domain

| Domain | Required capabilities | Not required there |
| --- | --- | --- |
| Bare `hpc115` | Bash and ordinary core utilities; Git and `realpath` for checkout/path admission; SSH for CPU control; rsync for transfer; Docker and the canonical image for local project Python | Native Generation venv, scientific Python packages, COMSOL, Slurm commands |
| `hpc115` Docker | Project Python plus the configured local NumPy, SciPy, h5py, PyYAML, Generation, Dataset, and validation dependencies | SSH, rsync, Slurm control, COMSOL |
| `sricehpc01` login | Module command; Git/HTTPS checkout access; exact clean checkout; native `.[generation-cpu]` venv and imports; writable durable CPU storage; `sbatch`, `squeue`, `sacct`, and `scancel`; rsync for the remote side of transfers | Per-case scratch or a compute-node COMSOL executable for ordinary orchestration |
| CPU Slurm compute node | Module command; configured native Python and COMSOL executables/versions; CPU Generation venv and project imports; NumPy, SciPy, h5py, and PyYAML; readable checkout/templates; writable durable storage; writable `$TMPDIR` (or `/tmp` fallback); owned temporary-directory creation/removal | rsync and scheduler-control commands (`sbatch`, `squeue`, `sacct`, `scancel`) |

rsync is intentionally present at both transfer endpoints: bare `hpc115` runs
the client command, and the CPU login environment serves the remote side. A
scientific allocation neither transfers campaign data nor invokes rsync.
Scheduler submission, polling, accounting, and cancellation likewise remain on
the CPU login/control plane. Native-smoke finalization also queries the COMSOL
version on the login host explicitly; ordinary login orchestration does not
otherwise require the compute executable. Compute jobs use Python filesystem operations for
case materialization, HDF5 conversion/admission, and durable publication; they
do not shell out to rsync, `tar`, `find`, `stat`, or `sha256sum`. The compute
launchers invoke `mktemp` for owned scratch, while Python owns hashing and tree
inspection. `srun` is not part of the maintained one-task job workflow.

The CPU checkout is read-only with respect to GitHub and uses the public HTTPS
repository URL, so GitHub SSH credentials are not required on `sricehpc01`.
CPU paths are rooted under the `$HOME` resolved by the remote environment:

- `$HOME/grainlegumes-generation/repo`
- `$HOME/grainlegumes-generation/storage`
- `$HOME/grainlegumes-generation/venv`

The resolved absolute home is environment-specific and is not a maintained
path. The CPU venv launcher may be a symlink to the module-provided base Python.
Validation therefore requires the configured venv root and Python's runtime
prefix evidence (`sys.prefix` at that root and distinct from `sys.base_prefix`),
not physical containment of the resolved base interpreter. `pyvenv.cfg`, the
exact launcher, and the required Generation/scientific imports remain mandatory.
Override the default root only with `--remote-root`. The local host
storage root defaults to the `storage` sibling of the dynamically resolved
checkout and may be overridden with `STORAGE_ROOT`; Docker sees that same
content at `/workspace/storage`, while CPU storage remains under the remote
root. Host-side admission, `realpath`, rsync, and staging operations always use
the host-native storage path.

Set the shared local values first:

```bash
# from the repository checkout on hpc115
export STORAGE_ROOT="$(realpath ../storage)"
STEADY_CAMPAIGN=configs/generation/campaigns/steady_flow/family_generalization.yaml
TRANSIENT_CAMPAIGN=configs/generation/campaigns/transient_drying/family_generalization.yaml
PILOT_CAMPAIGN=configs/generation/campaigns/transient_drying/pilot_check.yaml
CPU_HOST=sricehpc01
```

The execution YAML remains authoritative for `CPU_HOST`; the explicit shell
value makes the initial bootstrap target visible before the remote checkout
exists. Local inspection can be run explicitly through
`scripts/docker_python.sh`; never replace it with bare-host `python`.

1. Preview CPU setup, then perform it explicitly:

```bash
./scripts/generation_workflow.sh setup-cpu --cpu-host sricehpc01
./scripts/generation_workflow.sh setup-cpu --cpu-host sricehpc01 --execute
```

The first command is read-only. The second creates the remote checkout,
storage root, and Python environment for the exact local commit.

2. Validate both primary configs without promoting unresolved values:

```bash
./scripts/docker_python.sh -m src.generation.cli.cli_generation validate-config \
  "$STEADY_CAMPAIGN" --allow-incomplete
./scripts/docker_python.sh -m src.generation.cli.cli_generation validate-config \
  "$TRANSIENT_CAMPAIGN" --allow-incomplete
```

The resolved output includes purpose and profile, material inventory and roles,
counts and their derived total, memberships, authored and derived seed plans,
package requests and resolved package inventory, eligible parameter-OOD units and
allocations, purpose-specific pilot or smoke scope, bounded static-sentinel work,
execution resources, and exact readiness gates.

Run the non-solving native preflight for each campaign after local validation:

```bash
./scripts/generation_workflow.sh preflight "$STEADY_CAMPAIGN"
./scripts/generation_workflow.sh preflight "$TRANSIENT_CAMPAIGN"
```

Preflight first validates CPU login/control capabilities on `sricehpc01`,
including the exact checkout, venv, scheduler commands, and rsync. Only after
that gate passes does it submit one environment-only Slurm allocation to audit
the configured Python and COMSOL modules/executables, project imports, readable
source/templates, writable scratch and durable storage, and template/config
binding. Submitted Slurm workers may execute from scheduler-managed script
locations, so every repository dependency is resolved through the explicit exact
CPU checkout rather than the worker script directory or current working directory.
The preflight starts no COMSOL solve and submits no scientific case.

3. Preview the resolved Slurm plan after every primary gate is filled:

```bash
./scripts/generation_workflow.sh plan "$STEADY_CAMPAIGN"
```

`plan` is read-only but intentionally fails while production values or mappings
remain unresolved. Re-run `validate-config` after any configuration edit; its
resolved JSON is the review surface for the exact work that `plan` will use.

4. Run the canonical paired technical runtime smoke:

```bash
./scripts/generation_workflow.sh smoke --keep-cpu-source
```

If mappings are unconfirmed, this command first runs retained mapping probes
for both profiles and stops. Review the probe artifacts, update only the
explicit profile mapping keys, commit those reviewed changes, and rerun the
same command. A complete smoke always retains its CPU source for review.

5. Inspect the immutable real-smoke receipt printed by the smoke command:

```bash
SMOKE_RECEIPT=/absolute/path/printed/by/the/smoke/command.json
./scripts/docker_python.sh -m src.generation.cli.cli_generation validate-real-smoke \
  "$SMOKE_RECEIPT" --storage-root "$STORAGE_ROOT"
```

The all-in-one workflow already validates every HDF5 file, inspects every
dataset package, and runs DataLoader smokes with `num_workers=0` and
`num_workers=2`.

6. Run the globally serial transient core benchmark:

~~~bash
./scripts/generation_workflow.sh benchmark-cores --cpu-host sricehpc01
~~~

Review recommended_production_cores_per_case in the transferred summary. Compare
it with fastest_single_case_cores_per_case and
best_parallel_efficiency_cores_per_case, then manually edit only
cluster.cores_per_case in
configs/generation/execution/cluster_cpu.yaml. The benchmark never changes that
file. Commit the reviewed setting before continuing.

7. Run the configured transient pilot after the core decision:

~~~bash
./scripts/generation_workflow.sh pilot-check "$PILOT_CAMPAIGN"
~~~

8. After readiness and pilot acceptance, run each resolved production campaign:

~~~bash
./scripts/generation_workflow.sh all "$STEADY_CAMPAIGN"
./scripts/generation_workflow.sh all "$TRANSIENT_CAMPAIGN"
~~~

Execution config owns per-case allocation, pending-buffer, polling, optional
running cap, timeout, and failure policy. The generic
--only-batch <resolved-batch-name> selector can run a declared subset. When the
campaign declares an extreme-family group, --skip-extreme-family-ood may omit it
for one execution and cannot be combined with --only-batch.

9. Inspect or safely resume one persisted campaign:

~~~bash
GENERATION_RUN_ID="<campaign_run_id>"
./scripts/generation_workflow.sh status "$GENERATION_RUN_ID"
./scripts/generation_workflow.sh resume "$GENERATION_RUN_ID"
~~~

The foreground all or resume command owns polling and feeding. For a long
campaign, keep it in a site-approved persistent terminal session. Disconnecting
does not terminate submitted scientific jobs; rerunning resume reconciles exact
job IDs and durable case evidence before feeding again.

10. Cancel every persisted campaign attempt only when cancellation is intended:

~~~bash
./scripts/generation_workflow.sh cancel "$GENERATION_RUN_ID"
~~~

11. Preview and then explicitly confirm CPU-source cleanup for a reviewed run:

~~~bash
./scripts/generation_workflow.sh cleanup "$GENERATION_RUN_ID"
./scripts/generation_workflow.sh cleanup "$GENERATION_RUN_ID" --confirm
~~~

All setup, plan, launch, smoke, resume, and cleanup operations bind an exact
Git commit. Launching operations require a clean worktree.

## Configuration ownership and inspection

Configuration files author decisions; Python resolution validates and combines
them. Do not copy resolved values into scripts, notebooks, or documentation.
The owners are:

| Owner | Controls | Excludes |
| --- | --- | --- |
| `configs/generation/sources.yaml` | Bibliographic records keyed once | Parameter values, inferred assignments, roles, execution |
| `configs/generation/registry.yaml` | Parameter names, units, kinds, transforms, sampling order, OOD groups, components, derivations | Material supports, campaign counts, mappings, cluster resources |
| `configs/generation/common.yaml` | Grid, time, shared fixed physics, formulas, adapter and storage contracts | Material values, roles, counts, learning choices |
| `configs/generation/operations/<operation>.yaml` | Operation supports and constraints | Material values, template mappings, execution resources |
| `configs/generation/materials/<family>.yaml` | Role-neutral material scope, natural supports, coupled records, targets, evidence | Campaign role, membership, count, profile, execution |
| `configs/generation/profiles/<profile>.yaml` | Template identity, adapters, exports, explicit native mappings, profile conditioning | Material values, counts, roles, cluster plans |
| `configs/generation/campaigns/<profile>/<campaign>.yaml` | Purpose, layer references, material roles, sampling counts and seeds, memberships, package requests | Parameter ranges, derived package materials, execution defaults |
| `configs/generation/benchmarks/transient_core_scaling/` | One canonical pilot-case reference, repetitions, resource constraints, and four editable core counts | New scientific samples, package membership, production resource mutation |
| `configs/generation/execution/<site>.yaml` | Site, modules, executables, runtime limits, scheduler resources, purpose-specific retention | Scientific ranges, material roles, learning parameters |
| `configs/learning/<task>/<kind>/<config>.yaml` | Dataset IDs, model, optimization, training, evaluation, artifacts | Generation paths, material ranges, campaign membership |

Inspect the effective campaign rather than maintaining a parallel summary:

```bash
./scripts/docker_python.sh -m src.generation.cli.cli_generation validate-config \
  "$TRANSIENT_CAMPAIGN" --allow-incomplete
```

The JSON has four useful layers:

- `material_inventory`, `material_roles`, `case_counts`, `membership`, and
  `material_memberships` show source-case scope and split eligibility.
- `seed_plan` shows the campaign, membership, paired-equivalence, batch, and
  sampling-block seeds; case seeds remain deterministically derived from batch
  seed and case identity.
- `dataset_package_requests` preserves the campaign intent, while
  `dataset_package_inventory` shows every profile-expanded immutable package.
- `parameter_ood`, `pilot_plan`, `technical_smoke_plan`,
  `static_sentinel_workload`, and `execution_resources` show the applicable
  derived workload and runtime plan.

Validation errors identify the exact file, key, rule, actual value, and owner to
edit. Resolved identities, allocation evidence, and the effective scientific
configuration persist with generated artifacts.

## Supported progression and evidence

| Stage | Public action on `hpc115` | Work performed | Required evidence before continuing |
| ---: | --- | --- | --- |
| 1 | Edit the authoritative YAML | User decisions only | Reviewed diff and clean commit before remote execution |
| 2 | `validate-config ... --allow-incomplete` | Resolves all owners without solving | Counts, dimensions, roles, seeds, packages, OOD plan, template and exact gates |
| 3 | `setup-cpu` then `preflight` | Bootstraps exact commit/venv; audits modules, executables, storage and Slurm | Successful setup and preflight reports |
| 4 | `smoke` when mappings need confirmation | Runs retained one-case native mapping probes and inventories real output names/headers | Human-reviewed profile mapping YAML; rerun from a clean committed state |
| 5 | `smoke` | Paired steady/transient native technical campaign, transfer, publication, packages and loader smokes | Source-current immutable real-smoke receipt; CPU source retained |
| 6 | `benchmark-cores`, review, then edit `cluster.cores_per_case` | Four globally serial core settings with three round-robin solves each | Reviewed core-hour recommendation and a separate committed production-core decision |
| 7 | `pilot-check .../pilot_check.yaml` | Six-material transient diagnostics and storage measurement | Accepted pilot receipt and reviewed diagnostics |
| 8 | `all <production campaign>` | Preflight, plan, Slurm run, monitor, terminal validation, collection, package build, loader smokes and gated cleanup | Terminal all-workflow, transfer, Dataset, and cleanup receipts |
| 9 | `validate`, `status`, or direct receipt validators | Revalidates publication and lifecycle evidence | Exact successful receipt/digest state |
| 10 | `cleanup RUN_ID` then `--confirm` when needed | Preview then execute digest-authorized CPU cleanup | No active job; all transfer, Dataset, workflow and inventory digests agree |

No native COMSOL, Slurm, pilot, or production step is implied by a static test.

## Current native production gate

Static contracts agree on the corrected transient interface:

- `schedule.csv` has four total columns: argument `t` and values `T_in_bc`,
  `omega_in_bc`, `phi_in_bc`;
- native interpolation is linear;
- the runtime scalar handoff contains exactly 12 ordered case-dependent COMSOL
  parameter overrides;
- the template derives `T_init = T_amb`.

Production remains blocked until real native COMSOL evidence proves that the
saved template reloads the four-column schedule, consumes all three functions,
accepts all 12 overrides, preserves `T_init = T_amb`, solves, produces the
expected exports, and passes canonical HDF5/publication/package admission. That
native gate has not been executed merely because the template checksum, static
archive inspection, fake runtime, or unit tests pass. Scientific details and
the exact scalar order are owned by the parameter reference.

## Canonical command surface

All wrapper commands below are invoked on `hpc115`. Commands that would solve or
submit are shown for syntax and must run only after their gates are accepted.

| Command | Effect and evidence | Mutation/retention |
| --- | --- | --- |
| `./scripts/generation_workflow.sh setup-cpu` | Prints exact remote bootstrap plan | Read-only until `--execute` |
| `... setup-cpu --execute` | Creates/updates exact remote checkout, storage and venv | Mutates configured remote layout, no solve |
| `... preflight CAMPAIGN` | Audits resolved config, native environment and resources | No solve or Slurm submission |
| `... plan CAMPAIGN` | Prints exact paths and Slurm arguments | Read-only, fails closed on unresolved gates |
| `... smoke` | Owns mapping probes when required, then paired native technical gate | Native/Slurm; always retains CPU source |
| `... benchmark-cores [--variant ID]` | Runs isolated repeated same-case core scaling or retries one variant | Native/Slurm; benchmark metadata only, CPU evidence retained |
| `... pilot-check configs/generation/campaigns/transient_drying/pilot_check.yaml` | Runs transient diagnostic lifecycle | Native/Slurm; cleans CPU/staging after validated analysis by default |
| `... launch CAMPAIGN` | Submits one campaign and prints run ID | Native/Slurm primitive; does not perform full local lifecycle |
| `... all CAMPAIGN` | Full synchronous production through Dataset receipts and cleanup | Native/Slurm; cleans verified CPU source unless `--keep-cpu-source` |
| `... collect RUN_ID` | Validates terminal CPU source, rsyncs to marked staging, atomically publishes | Nondestructive to CPU source; cleans successful non-pilot staging |
| `... build-datasets RUN_ID` | Builds/reuses every requested package and smokes loaders | Writes immutable `02_datasets` packages; preserves `01_generation` |
| `... status RUN_ID` / `... accounting RUN_ID` | Reconstructs local/remote state or prints scheduler evidence | Read-only |
| `... resume RUN_ID` | Resumes the persisted incomplete workflow/stage | Submits only validated incomplete membership when required |
| `... validate RUN_ID` | Revalidates remote terminal campaign | Read-only |
| `... cancel RUN_ID` | Cancels every persisted active Slurm attempt | Mutates scheduler state, not data |
| `... cleanup RUN_ID` | Prints exact authorized CPU deletion plan | Dry-run |
| `... cleanup RUN_ID --confirm` | Deletes only digest-authorized inactive CPU source and records receipt | Destructive and gated; GPU publication remains |

Production allocation and feeder settings have no command-line overrides; edit the
execution YAML and review the resolved plan. `--only-batch` selects one declared
batch, while `--skip-extreme-family-ood` is a one-run omission and cannot be
combined with it. `--detach` applies to `all`; `--keep-cpu-source` is the sole
source-retention override.

The low-level `run-batch --max-parallel-cases` option is retained only for
bounded local development execution. It limits local worker threads, is never
forwarded to Slurm, and does not control production running or pending jobs.

## CPU/GPU path and lifecycle

The maintained execution path is:

```text
hpc115 bare shell
  -> hpc115 Docker: local config/evidence/Dataset Python
  -> sricehpc01 login: exact checkout + native Generation venv + Slurm control
  -> CPU compute node $TMPDIR: case inputs + COMSOL + conversion
  -> sricehpc01 durable Generation source
  -> hpc115 marked transfer staging
  -> hpc115 canonical Generation publication + Dataset packages
```

Data sent from `hpc115` to the CPU side is compact: the exact source commit and
its configuration/template content, orchestration identity, and small readiness
or smoke receipts. The CPU checkout normally obtains source through Git; the
wrapper does not pre-generate campaign-wide `fields.csv`, `schedule.csv`, or
`scalars.csv` collections on `hpc115`. Completed validated results, HDF5,
provenance, manifests, logs, and receipts travel from CPU storage back to
`hpc115`. Bulk case-input CSV transfer is not part of normal production.

A representative transient case has the following ownership:

| Artifact | Created on and by | Filesystem/lifetime | Transfer |
| --- | --- | --- | --- |
| Compact campaign/case plan | `sricehpc01` login, native Generation venv | Durable CPU metadata bound to commit, config, indices, and seeds | Only compact control identity is needed before execution |
| `case.json` | CPU compute node, deterministic case service | Per-case `$TMPDIR` workspace; required identity is persisted with CPU evidence | Not generated or bulk-transferred by bare `hpc115` |
| `fields.csv` | CPU compute node | Per-case `$TMPDIR`; removed after safe publication | No hpc115-to-CPU bulk transfer |
| `schedule.csv` | CPU compute node for transient cases | Per-case `$TMPDIR`; removed after safe publication | No hpc115-to-CPU bulk transfer |
| `scalars.csv` | CPU compute node for transient cases | Per-case `$TMPDIR`; admitted before COMSOL and removed after safe publication | No hpc115-to-CPU bulk transfer |
| `model.mph` work copy | CPU compute node from the exact checkout template | Per-case `$TMPDIR`; source template remains immutable | Template bytes arrive with the exact checkout |
| COMSOL exports and optional solved model | CPU compute node, native `Comsol/v6.4` | Per-case `$TMPDIR`; retention follows campaign purpose | Required retained evidence publishes to CPU storage |
| Canonical HDF5 | CPU compute node conversion/admission | Published atomically into durable CPU `01_generation` source | Validated CPU-to-hpc115 transfer |
| CPU evidence | Compute worker and campaign services | Durable remote `01_generation/{meta,raw,processed}` | Source for validated collection; retained until cleanup gates pass |
| Transfer staging | `hpc115` wrapper and Docker admission | Marked `01_generation/.state/transfer-staging`; temporary and retryable | Receives CPU-to-hpc115 rsync only |
| GPU-side publication | `hpc115` Docker publication service | Canonical immutable `01_generation` | Atomic promotion from validated staging |
| Dataset package | `hpc115` Docker Dataset service | Immutable `02_datasets` package referencing canonical Generation files | No return transfer to CPU |

`01_generation` is the canonical simulation archive. `02_datasets` contains
immutable package views addressed by `dataset_id`; `03_experiments` contains
training, tuning, and evaluation artifacts. The `all` command performs terminal
validation, transfer, publication, package inspection, both DataLoader smokes,
and digest-authorized CPU cleanup. Failed or incomplete transfers retain CPU
source and marked staging; successful cases and publications are reused rather
than overwritten.

The remote layout is `$HOME/grainlegumes-generation/{repo,storage,venv}`. The
login preflight checks the exact commit before submission. The compute launcher
loads the configured Python and COMSOL modules, activates the native venv,
checks the Slurm allocation, creates a guarded worker root under `$TMPDIR`, and
removes only that owned temporary tree after success or recorded failure.
`$TMPDIR` is never durable evidence. CPU source is
deleted only through the existing inactive-job, source-inventory, transfer,
HDF5, Dataset, workflow-receipt, and digest authorization gates. `collect` and
the core benchmark always retain their CPU source; `--keep-cpu-source` remains
the production/pilot retention override.

## Smoke versus production

Static sentinels start no COMSOL process and mutate no production state. They
exercise every material, active sampling block, and eligible parameter-OOD unit
reported by `static_sentinel_workload`, plus coupled records, fields, schedules,
and deterministic replay.

The fake runtime uses a test-owned executable. It proves Python case isolation,
HDF5 conversion/publication, package construction, factory guards, and
DataLoader behavior; it is not COMSOL evidence.

The real technical smoke is the native COMSOL gate. Inspect each technical
campaign with `validate-config` to review its exact cases, paired seed, package
inventory, and retained-evidence policy. The gate requires:

- every distinct case declared by both resolved technical campaigns;
- exact case-local scalar admission and twelve COMSOL CLI overrides;
- validated raw exports and canonical HDF5;
- paired shared inputs across the two profiles;
- observed `p`, `u`, and `v` difference metrics without an invented tolerance;
- observed differential/integral water-balance metrics;
- immutable technical packages and both DataLoader worker modes;
- retained CPU inputs, exports, solved evidence, logs, Slurm IDs, and version;
- one source-bound real-smoke receipt.

Mapping probes inventory actual output files and headers. They never infer or
write a mapping automatically. Fixed values reported as template-owned have no
Python runtime override; their configured record is bound to the canonical
hashed template and model-report evidence. Case-dependent values use the
admitted CLI vector and still require native runtime evidence.

## Shared-cluster transient core benchmark

The transient_core_scaling suite determines the production cores_per_case for
one ordinary independent COMSOL case. It does not determine campaign
concurrency, pending_buffer, scheduler priority, or node packing. Its owners are
configs/generation/benchmarks/transient_core_scaling/suite.yaml and the four
variant files:

| Variant | Current configured cores | Measured repetitions |
| --- | ---: | ---: |
| cores_04.yaml | 4 | Suite-owned, currently 3 |
| cores_08.yaml | 8 | Suite-owned, currently 3 |
| cores_16.yaml | 16 | Suite-owned, currently 3 |
| cores_32.yaml | 32 | Suite-owned, currently 3 |

The values and repetition count are editable YAML decisions. Every variant uses
the same nominal transient pilot case and proves the same case_input_id,
simulation_case_id, fields, schedule, scalar handoff, scientific configuration,
and template hash. Core count changes execution evidence only.

After CPU setup and a source-current native technical-smoke receipt, run:

~~~bash
./scripts/generation_workflow.sh benchmark-cores \
  --cpu-host sricehpc01
~~~

The measured sequence is round-robin:

~~~text
4, 8, 16, 32, 4, 8, 16, 32, 4, 8, 16, 32
~~~

Each entry is one ordinary non-exclusive Slurm job for one case. The workflow
submits no array, dependency chain, full-node reservation, or later measured
job. It submits the next entry only after the previous job is no longer active
and its immutable success or failure evidence has been reconciled. Thus at most
one benchmark solve is submitted, running, or pending at a time. A failed
setting can be retried explicitly without rerunning successful repetitions:

~~~bash
./scripts/generation_workflow.sh benchmark-cores \
  --variant cores_16 \
  --cpu-host sricehpc01
~~~

Every attempt records submit, scheduler start, and completion timestamps;
derived queue wait and turnaround; COMSOL and complete-case wall times; node,
partition, requested and allocated CPUs, COMSOL process count, Slurm job ID, and
accounting evidence. Queue wait describes observed scheduler conditions only
and never selects the production core count.

For each core setting, summary.json, summary.md, and runs.csv report median,
mean, minimum, maximum, and sample standard deviation of COMSOL time; median
complete-case, queue-wait, and turnaround time; speedup; parallel efficiency;
median COMSOL core-hours per case; cases per 100 core-hours; and the estimated
compute core-hours for the currently configured transient production case
count. The estimate does not claim an exact campaign wall time because Slurm
decides concurrent availability. Finalization re-resolves that current count;
when the measured evidence remains applicable, finalizing the same terminal run
again archives and regenerates its derived summaries without another COMSOL
solve.

The deterministic recommendation is the successful variant with the lowest
median COMSOL core-hours per case, with fewer cores breaking a tie. The summary
also reports fastest_single_case_cores_per_case and
best_parallel_efficiency_cores_per_case separately, plus the recommended
runtime, complete-case time, speedup, efficiency, current-setting difference,
and comparison with the fastest setting. No result is hard-coded and queue wait
is excluded.

The workflow never edits production resources. After reviewing the benchmark
summary, manually edit:

~~~text
configs/generation/execution/cluster_cpu.yaml
cluster.cores_per_case
~~~

Then commit that reviewed configuration decision before the transient pilot and
production workflow. pending_buffer remains a separate submission-policy
setting and is never changed by the benchmark.

## Configured transient pilot check

Inspect the pilot owner before launch:

```bash
./scripts/docker_python.sh -m src.generation.cli.cli_generation validate-config \
  configs/generation/campaigns/transient_drying/pilot_check.yaml \
  --allow-incomplete
```

`pilot_plan` reports the resolved material inventory, cases per material, total,
campaign seed, case semantics, and the absence of learning membership and normal
package publication. Those values come from the pilot campaign, so changing a
valid pilot configuration changes inspection, planning, and execution together.

Run the all-in-one diagnostic from the GPU/development host:

```bash
./scripts/generation_workflow.sh pilot-check \
  configs/generation/campaigns/transient_drying/pilot_check.yaml
```

The configured first-case and remaining-case semantics are resolved into every
batch assignment. The pilot uses natural support only and performs no parameter
OOD, family-OOD sampling, training membership, or automatic calibration. Its
generic analysis checks runtime and conversion contracts, drying duration,
natural-support robustness, physical bounds, water balances without an invented
tolerance, run-wide extrema, heterogeneity, schedules, configured applicability
metadata, and measured storage for every resolved material.

A smaller diagnostic can override the resolved cases-per-material value for that
execution without editing campaign YAML:

```bash
./scripts/generation_workflow.sh pilot-check \
  configs/generation/campaigns/transient_drying/pilot_check.yaml \
  --cases-per-material 1
```

Retained-source debugging form:

```bash
./scripts/generation_workflow.sh pilot-check \
  configs/generation/campaigns/transient_drying/pilot_check.yaml \
  --keep-cpu-source
```

The lifecycle performs host preflight, exact commit/config/template binding, CPU
readiness, a mapping probe when required, deterministic planning, scheduler
execution and monitoring, terminal collection, hash validation, HDF5 conversion,
runtime/physical/mass-balance/extrema/trend analysis, pre-cleanup CPU and staging
measurement, permanent GPU measurement, and a storage projection against the
separately resolved production campaign. It labels that projection
`observed_real_pilot_based_estimate`; no production case total is embedded in
the analysis. One canonical
`01_generation/meta/pilot_checks/<pilot_check_id>/pilot_check.json` owns the
results; `summary.csv` and `summary.md` are derived from it.

After retained evidence validates, the normal command performs authorized
CPU-source and transfer-staging cleanup and verifies deletion. It never deletes
active, incomplete, hash-invalid, or insufficiently retained evidence.
`--keep-cpu-source` is the explicit CPU-source opt-out; staging is still cleaned
after validation. There is no storage-budget pass/fail guard. The command stops
before native execution whenever the current readiness report has an unresolved
gate.

## Scheduler-fed production generation

Production prerequisites are accepted source-current native COMSOL smoke and
transient pilot gates. Static tests and template checksums do not imply that
production has run.

The execution owner is configs/generation/execution/cluster_cpu.yaml. Its
current relevant structure is:

~~~yaml
runtime:
  timeout_seconds: 3600
  maximum_failures: 1
submission:
  pending_buffer: 1
  poll_interval_seconds: 15
  max_running_cases: null
cluster:
  cores_per_case: 16
  wall_time: null
  scheduler_options: []
~~~

cores_per_case is the independently requested CPU count for one COMSOL case.
pending_buffer is the number of jobs from this exact campaign that may wait in
Slurm; changing it from 1 to 2 is a configuration-only operational decision.
poll_interval_seconds owns the foreground reconciliation cadence.
max_running_cases is an optional safety cap, where null means that the workflow
imposes no running target or cap. timeout_seconds and maximum_failures retain
their existing runtime and stop-policy meanings. These settings affect
execution provenance, not scientific case_input_id or Dataset identity.

Fixed campaign-node packing and array-concurrency controls have no production
owner. Each scientific case is one
ordinary non-exclusive Slurm job requesting cores_per_case. There is no
campaign node reservation, manual node packing, or whole-campaign array. Slurm
chooses placement, may colocate compatible jobs, and decides how many run.
The project makes no claim about scheduler priority for work submitted by other users.

With the maintained pending_buffer of 1, the reconciliation loop is:

~~~text
lock the exact campaign run
load atomic manifest and recover any unresolved submission intent
query squeue and sacct for persisted job IDs only
for every declared case:
    trust validated immutable case evidence as success
    otherwise classify exact submitted attempts as active, pending,
    terminal failure, accounting-unknown, or absent
count pending and running jobs for this campaign only
stop on unknown accounting or the configured failure threshold
if optional running cap is set and reached:
    submit nothing
elif pending count is at least pending_buffer:
    submit nothing
elif an eligible unsent case remains:
    atomically persist its exact case/job submission intent
    submit exactly that one ordinary job
    atomically persist the returned job ID
reconcile again before any later submission
~~~

This ramp-up allows every newly submitted job that Slurm admits to become
running, then restores one waiting job. Once one campaign job remains pending,
all remaining cases stay unsubmitted. When the pending slot clears, the next
poll submits exactly one new case. Healthy running or pending jobs are neither
cancelled nor suspended to make room.

Resume uses the same manifest, lock, exact persisted job IDs, squeue, sacct,
case-failure evidence, and validated HDF5 publication. It never reruns a valid
successful case or duplicates an active or pending case. A disappeared job is
not considered successful without authoritative case evidence. Accounting lag
fails closed, and failed cases require the existing explicit resume/retry path
once the failure threshold stops normal feeding. Unique persisted submission
intent and campaign-scoped job identity recover an accepted sbatch after an SSH
or wrapper interruption.

Run:

~~~bash
./scripts/generation_workflow.sh all "$STEADY_CAMPAIGN"
./scripts/generation_workflow.sh all "$TRANSIENT_CAMPAIGN"
~~~

The foreground workflow polls and feeds, then continues through terminal
validation, transfer, immutable Generation publication, requested Dataset
packages, loader smokes, receipts, and gated CPU cleanup. If the controlling
shell exits, already submitted Slurm jobs continue. Rerun
generation_workflow.sh resume with the printed run ID to reconstruct state and
continue safely; do not submit cases manually.

## Readiness gates

Run:

```bash
./scripts/docker_python.sh -m src.generation.cli.cli_generation readiness-report \
  "$STEADY_CAMPAIGN" "$TRANSIENT_CAMPAIGN" \
  --run-static-sentinels
```

The report owns its status vocabulary and includes one structured record for
each configuration, static-science, mapping, native-runtime, and launch gate.
Treat its current JSON as authoritative; do not infer readiness from a template
hash, a cached receipt, or documentation text. Launch is ready only when
the resolved production configuration, static scientific checks, reviewed
mappings, native profile reloads, scalar handoff, paired-equivalence
observations, HDF5/package/loader validation, and a source-current real-smoke
receipt all pass. A template hash proves byte identity only; it does not prove
model-tree behavior.

## Troubleshooting

- Exit status 2 from `validate-config` without `--allow-incomplete`, `plan`, or
  readiness means a fail-closed gate remains. Read the reported file and key;
  do not fill scientific values with CLI defaults.
- If `smoke` stops after mapping probes, review
  `01_generation/meta/mapping_probes/<probe_id>/mapping_probe.json` and its
  retained files on the CPU storage. Update profile YAML only after COMSOL
  inspection.
- A dirty worktree or commit mismatch blocks remote planning and launch. Commit
  reviewed work outside this workflow, then rerun the same command.
- Failed collection retains marked staging and CPU source. Use `status` and
  `resume`; do not manually move partial data into `01_generation`.
- Smoke packages cannot be selected by normal training configuration.
- A benchmark refusal normally means the source-current real-smoke gate is
  absent, one repetition failed, or a successful result conflicts. Use the
  printed `--variant` recovery command; do not delete successful evidence.
- Never delete CPU source manually. Use cleanup dry-run and confirmation so the
  digest authorization is preserved.

Scientific parameter ownership and formulas are documented in the
[parameter reference](generation_parameter_reference.md). The project gateway
is the [README](../README.md).
