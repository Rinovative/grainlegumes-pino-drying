# Generation Operations

This is the operator guide for Generation. Scientific equations and parameter
meaning live in the
[scientific parameter reference](generation_parameter_reference.md); current
values and inventories live in validated YAML under `configs/generation`.

## Quick start

Run maintained workflows from the bare `hpc115` checkout. Generation executes a
clean, commit-pinned source copy; uncommitted local changes are not run.

```bash
export STORAGE_ROOT="$(realpath ../storage)"
CPU_HOST=sricehpc01

./scripts/generation_workflow.sh setup-cpu --cpu-host "$CPU_HOST"
./scripts/generation_workflow.sh setup-cpu --cpu-host "$CPU_HOST" --execute
./scripts/generation_workflow.sh run CONFIG --dry-run
./scripts/generation_workflow.sh run CONFIG --preflight-only --cpu-host "$CPU_HOST"
./scripts/generation_workflow.sh run CONFIG --cpu-host "$CPU_HOST"
```

Maintained entry configurations are:

| Workflow | Configuration |
| --- | --- |
| Paired Technical Smoke | `configs/generation/workflows/technical_smoke.yaml` |
| Transient core benchmark | `configs/generation/benchmarks/transient_core_scaling/suite.yaml` |
| All-material pilot | `configs/generation/campaigns/transient_drying/material_pilot.yaml` |
| Transient production | `configs/generation/campaigns/transient_drying/family_generalization.yaml` |
| Airflow ID Dataset | `configs/generation/campaigns/steady_flow/id_dataset.yaml` |

All maintained Generation schemas and durable records remain version `1`.

## CPU installation layout

The maintained CPU Generation root is
`/zfspool/storage/home/rino.albertin/grainlegumes-generation`. A fresh
`setup-cpu --execute` creates one detached execution checkout, one matching
environment, and the persistent scientific storage root:

```text
grainlegumes-generation/
├── repo/
├── venv/
└── storage/
```

Generation creates scientific domains below `storage/` only when a workflow
needs them. Setup does not create empty Dataset or experiment domains for visual
symmetry. Benchmark preflight and case scratch use marker-owned directories
below the approved system temporary root and are removed deterministically;
they are not persistent children of the CPU root.

Before an existing `repo/` or `venv/` can change, setup queries the exact
persisted campaign and benchmark Slurm identities. Active dependent jobs or
unprovable scheduler evidence block mutation. Repeating setup for the same
installed commit leaves the shared checkout and environment unchanged.
Commit-keyed checkouts and per-commit environments are not supported.

## Common plan and lifecycle

Every config resolves to one deterministic `GenerationRunPlan` containing its
source commit, config and scientific identities, ordered work units, retention
and collection policy, Dataset declarations, and finalizers. `run CONFIG` is the
only normal start/resume entry point.

The controller:

1. resolves and validates source, configuration, identity, and durable state;
2. prepares missing canonical inputs;
3. submits eligible work through the shared admission owner;
4. monitors scheduler, worker, retry, replay, and publication evidence;
5. validates CPU results;
6. optionally collects and publishes them on the host;
7. builds declared Dataset packages and finalizers;
8. performs cleanup only after explicit validated authorization.

Re-running the same config reuses valid work, reconciles active ownership, and
submits only eligible missing work. It never treats Slurm `COMPLETED` alone as
scientific success.

### Exact source pinning and input reuse

Without `--git-commit`, `run CONFIG` resolves the current committed local HEAD.
To resume one historical campaign after local HEAD advances, pass its exact
40-character commit:

```bash
./scripts/generation_workflow.sh run CONFIG --git-commit HISTORICAL_COMMIT --cpu-host "$CPU_HOST"
```

The launcher verifies that object locally, materializes a clean detached source,
and runs both the workflow and config from that source. A dirty development
worktree is ignored; it is never executed under the historical provenance label.
The CPU checkout must equal the requested commit before campaign work proceeds.

Immutable Generation inputs retain their original `input_generation_id` and
source commit. A newer execution may reference one older input generation only
when every scientific, template, schema, batch, and ordered case-membership field
matches exactly; only the execution commit may differ. The current exact source
is preferred, while ambiguous or corrupt compatible history fails closed. Each
campaign manifest persists its selected input source. Execution run identity
continues to include the execution commit, and completed solver outputs are not
reused merely because their inputs are compatible.

For the all-material pilot, `current_pilot_gpu_permanent_bytes` is the
validated destination transfer inventory plus all regular durable files owned
by the pilot-check directory: the receipt, pre-cleanup snapshot, finalized
cleanup receipt, and summaries. A recognized in-progress cleanup receipt remains
outside the pre-cleanup total until finalization records it. The metric excludes
incoming transfer staging, retained CPU source, unexpected pilot-check entries,
and unrelated storage. The finalizer
produces and validates the complete
campaign-, destination-, and inventory-bound accounting mapping before
rendering operator views. Missing or conflicting accounting fails the
finalizer, not any already successful scientific case. Re-running the normal
workflow reuses a valid pre-cleanup pilot receipt and resumes only missing
finalizer or retention evidence.

## Controller and collection modes

Foreground execution is the default. Use `--background` for a commit-pinned
`tmux` controller that survives terminal and SSH-session loss:

```bash
./scripts/generation_workflow.sh run CONFIG --background --cpu-host "$CPU_HOST"
./scripts/generation_workflow.sh background-status "$WORKFLOW_SESSION_ID"
./scripts/generation_workflow.sh background-list
```

A host reboot ends the controller, not the CPU jobs or durable campaign state;
run the same config again.

| Mode | Host collection/finalizers | CPU source |
| --- | --- | --- |
| Default | Automatic | Removed only after authorized cleanup |
| `--keep-cpu-source` | Automatic | Retained as an additional copy |
| `--defer-collection` | Deferred | Retained as the exclusive copy |

`--keep-cpu-source` and `--defer-collection` are mutually exclusive. Resume
deferred collection by running the same config without `--defer-collection`.
That flag controls collection only; it is not an SSH-retry option.

## SSH transport

The controller uses non-interactive SSH with `BatchMode=yes`. It requires no
password prompt, user SSH-config change, ControlMaster, or persistent socket.

Safe reads and idempotent campaign reconciliation use one shared bounded policy:
initial attempt, then retries after 5, 15, 30, and 60 seconds. Retry requires SSH
status `255` and an allowlisted transport/authentication message. Ordinary remote
exit statuses, host-key failures, unsafe configuration or permissions, and
scientific command errors are never retried.

The helper buffers the complete remote script and arguments and replays the same
bytes. Diagnostics go to stderr; machine-readable stdout remains clean. A
recovered interruption continues the same polling iteration. Exhaustion returns
the final SSH failure while retaining CPU evidence, jobs, run identity, and the
same-config resume command.

Initial submission, benchmark resume/finalization, cancellation, transfer
publication, and cleanup remain one-shot because a lost acknowledgement can make
blind repetition ambiguous. Transport failure never invokes cancellation.

## License acquisition

Temporary COMSOL capacity is recognized only from a feature-bearing checkout
message plus strong capacity evidence such as user-limit or FlexNet `-4` output.
Unknown, expired, missing, or misconfigured licensing remains a hard failure.

`max_admission_cases` bounds logical pre-solver cases; it is not a running-solver
limit. Each admitted Slurm worker searches independently. After allocation, the
first COMSOL checkout is immediate. Strong pre-solve capacity failures retry in
the same allocation every five seconds until the 120-second launch deadline.
The process that proves solver startup continues unchanged through solve and
publication. Queue time and workspace preparation are outside the window.

COMSOL may write an exact 13-digit epoch-millisecond timestamp followed by
`Error` after a failed startup checkout. `Error` alone never proves temporary
capacity. A capacity-only `solved.mph.status` file is removed only when the
independent strong capacity classifier succeeds, the checkout process has
terminated, the path was absent before launch inside the isolated owned worker
workspace, the file is a non-symlink owned regular file, and no solver progress,
solved model, export, canonical success, or scientific result exists. The status
bytes must match one exact bounded timestamp-and-state grammar. Device, inode,
owner, size, modification time, and digest are revalidated immediately before
cleanup. Pending and complete schema-version-1 receipts bracket the exact
unlink. Pre-existing, changed, unknown, or otherwise unsafe artifacts remain
preserved and fail closed.

A safely recovered capacity artifact permits the next checkout in the same
allocation. Window exhaustion produces operational evidence and
`license_blocked`; it is not a scientific attempt or failure. The controller
later retries the same logical case through normal admission. Termination may
add at most five seconds after TERM and five after KILL beyond the launch
deadline.

## Failure isolation and completion

One terminal case failure does not stop unrelated runnable work. Solver,
conversion, publication, replay, or explicitly classified case-reconciliation
failures remain visible while pending, active, license-blocked, and never-started
cases continue.

The solver-failure circuit counts genuine solver failures and technical runtime
timeouts. `maximum_failed_cases` is inclusive: admission stops only at
`maximum_failed_cases + 1`. Temporary license capacity, postprocessing replay,
and presentation metadata do not consume that budget. Existing jobs are not
cancelled merely because another case failed or the circuit opened.

Typed case-local reconciliation errors are persisted against exact source bytes
and dependency identity so unchanged defects do not starve later polls. Global
manifest/identity conflicts, duplicate or ambiguous active-job ownership,
unsafe paths or symlinks, incompatible scientific identity, conflicting
immutable hashes, scheduler/admission corruption, and mutation-lock failures
remain immediately fatal.

Final campaign failure is reported only after permitted runnable work finishes,
the configured circuit opens, or a global integrity failure occurs.

## Terminal timestamps and status

Scientific success is owned by validated scientific and publication evidence,
not by a presentation timestamp. Repository-written operational timestamps are
timezone-aware UTC. External naïve timestamps are normalized only when their
source timezone is authoritative; otherwise their raw value is retained and
they are omitted from timestamp ordering without demoting the case.

Status shows at most three recent successful cases and three recent failures.
Older failures are grouped by bounded classification, and cases whose persisted
state is `never_started` are presented as `not_admitted` and grouped by
material or work-unit family. The recurring Cases line always renders, in order,
`successful`, `running`, `scheduler_pending`, `license_blocked`,
`not_admitted`, `failed`, and `total`, including zero counts. The separate
Admission line always renders `pending`, `starting`, `acquiring_license`,
and `license_waiting`; these counts describe admission-slot occupancy rather
than a second case-state inventory.

For a successful transient case, compact terminal science uses the validated
bulk wet-basis moisture and configured material target:

```text
case_0001  batch=lentil
  state=successful  reason=validated_case_evidence
  simulated_end=60.32 h  bulk_moisture=11.6% wb  target=12.0% wb
```

The spatial `f_wet_dm` result remains part of stopping, validation, canonical
results, and detailed diagnostics, but is not repeated on the normal successful
status line. Temporary-capacity cases stay outside the scientific failed
section. Recurring terminal output reports only compact checkout,
recovered-artifact, window, and retry state; content excerpts, digests, raw
dictionaries, and machine-oriented JSON remain in canonical evidence.
Formatting does not read HDF5/CSV, query Slurm again, hash payloads, or expose
raw license logs.

```bash
./scripts/generation_workflow.sh status CONFIG_OR_RUN_ID --cpu-host "$CPU_HOST"
./scripts/generation_workflow.sh cancel "$GENERATION_RUN_ID" --cpu-host "$CPU_HOST"
./scripts/generation_workflow.sh cancel "$GENERATION_RUN_ID" --force --cpu-host "$CPU_HOST"
./scripts/generation_workflow.sh cleanup "$GENERATION_RUN_ID" --confirm --cpu-host "$CPU_HOST"
```

The first Ctrl+C after campaign launch requests graceful cancellation of only
owned work; the second requests force cancellation. Before a run identity exists,
cancellation is not armed.

## Replay and retention

Failure retention is stage-specific:

- license-only blocks retain compact operational evidence;
- solver failures retain bounded logs, status, timing, and identity evidence;
- conversion failures retain exact exports needed for replay;
- publication failures retain validated HDF5 and required provenance;
- successful compact cases retain canonical HDF5 and bounded evidence;
- full-retention Smoke/Pilot successes retain their declared diagnostics.

Postprocessing replay is independent of Slurm admission and never launches
COMSOL. It may reuse identity-compatible conversion/publication evidence across
source-pinned campaigns while preserving the solver commit and recording the new
processing commit. Solver failures are never reclassified as replayable.

## Canonical input case generation and EDA

Normal `run CONFIG` prepares missing canonical inputs automatically. For a
bounded input-only operation, use the Generation CLI with an explicit campaign,
batch, range, commit, and storage root:

```bash
./scripts/docker_python.sh -m src.generation.cli.cli_generation \
  generate-input-cases "$CAMPAIGN_CONFIG" \
  --only-batch "$BATCH_NAME" --case-start 1 --case-count "$CASE_COUNT" \
  --git-commit "$(git rev-parse HEAD)" --storage-root "$STORAGE_ROOT"
```

Input EDA discovers only admitted canonical manifests and reads them without
mutating or regenerating cases.

## Identity and provenance policy

Scientific identity is separate from filenames, paths, labels, controller
sessions, and presentation order. Resolved config, source commit, seeds, input
and simulation identities, templates, schedules, hashes, runtime evidence, and
publication receipts bind each durable stage. Ordinary reads do not rewrite
immutable evidence.

`STORAGE_ROOT` owns three top-level domains:

```text
01_generation/   canonical inputs, processed cases, attempts, run evidence
02_datasets/     immutable Dataset packages and publication metadata
03_experiments/  training and evaluation runs
```

Dataset packages reference validated canonical `case.h5` sources rather than
copying them. Cleanup is restricted to cryptographically authorized run-owned CPU
paths and never removes host publication, shared canonical source, or another
user's files.

The core benchmark uses its dedicated configured cases and variants. Its primary
measurement is successful COMSOL process time; queue, license wait, failed
checkout, conversion, publication, collection, and controller time remain
separate. The workflow reports a recommendation but never edits production
configuration.

Run `./scripts/generation_workflow.sh --help` for current option syntax.
