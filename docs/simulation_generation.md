# Generation Workflow

This guide describes the maintained Generation operator workflow. Scientific
parameter definitions, equations, and evidence classifications live in the
[scientific parameter reference](generation_parameter_reference.md).

## Operator model

Run every maintained workflow from the bare `hpc115` checkout with one command:

```bash
./scripts/generation_workflow.sh run CONFIG [options]
```

The command resolves `schema_kind`, creates one immutable common run plan, and
continues the matching deterministic run if evidence already exists. Foreground
execution is the default. Generation pins the selected committed source; local
uncommitted changes are ignored by the pinned execution copy.

The maintained entry configurations are:

| Workflow | Configuration | Planned work units |
| --- | --- | ---: |
| Paired Technical Smoke | `configs/generation/workflows/technical_smoke.yaml` | two ordinary two-case child campaigns |
| Transient core benchmark | `configs/generation/benchmarks/transient_core_scaling/suite.yaml` | 8 successful measurements |
| All-material pilot | `configs/generation/campaigns/transient_drying/material_pilot.yaml` | 18 cases |
| Transient production | `configs/generation/campaigns/transient_drying/family_generalization.yaml` | 600 cases |
| Airflow ID Dataset | `configs/generation/campaigns/steady_flow/id_dataset.yaml` | 1,050 cases |

All authored Generation schemas and durable records use schema version `1`.

Set the common paths once:

```bash
export STORAGE_ROOT="$(realpath ../storage)"
SMOKE_CONFIG=configs/generation/workflows/technical_smoke.yaml
BENCHMARK_CONFIG=configs/generation/benchmarks/transient_core_scaling/suite.yaml
PILOT_CONFIG=configs/generation/campaigns/transient_drying/material_pilot.yaml
TRANSIENT_CONFIG=configs/generation/campaigns/transient_drying/family_generalization.yaml
AIRFLOW_ID_CONFIG=configs/generation/campaigns/steady_flow/id_dataset.yaml
```

Preview setup, inspect a plan, or perform the non-solving runtime preflight:

```bash
./scripts/generation_workflow.sh setup-cpu --cpu-host sricehpc01
./scripts/generation_workflow.sh setup-cpu --cpu-host sricehpc01 --execute
./scripts/generation_workflow.sh run "$AIRFLOW_ID_CONFIG" --dry-run
./scripts/generation_workflow.sh run "$AIRFLOW_ID_CONFIG"   --preflight-only --cpu-host sricehpc01
```

Run any maintained workflow in the foreground:

```bash
./scripts/generation_workflow.sh run "$SMOKE_CONFIG" --cpu-host sricehpc01
./scripts/generation_workflow.sh run "$BENCHMARK_CONFIG" --cpu-host sricehpc01
./scripts/generation_workflow.sh run "$PILOT_CONFIG" --cpu-host sricehpc01
./scripts/generation_workflow.sh run "$TRANSIENT_CONFIG" --cpu-host sricehpc01
./scripts/generation_workflow.sh run "$AIRFLOW_ID_CONFIG" --cpu-host sricehpc01
```

Technical Smoke is optional and is not a prerequisite for another run. The core
benchmark is standalone and does not read Smoke evidence.

## Common plan and lifecycle

A validated campaign, benchmark suite, or ordered workflow resolves to one
`GenerationRunPlan`. It contains the source commit, authored-config identity,
scientific input identity, ordered child plans, work units, retention and
collection policies, Dataset declarations, and scientific finalizers. Campaign
cases and benchmark measurements share the same operational work-unit lifecycle.
Configuration-specific code only resolves units and finalizers.

The common controller performs these stages in order when declared:

1. Resolve exact source, config, identity, and existing durable state.
2. Validate local and CPU/runtime prerequisites.
3. Materialize missing canonical inputs.
4. Submit eligible work units and monitor the shared state model.
5. Wait and retry genuine temporary license-capacity blocks.
6. Validate and publish successful CPU results.
7. Finalize CPU-owned evidence.
8. Stop at `awaiting_collection` when collection is deferred.
9. Otherwise transfer, validate, and atomically publish on the host.
10. Build declared packages, run declared scientific finalizers, validate the
   complete result, and apply guarded CPU retention.

The same config handles every continuation:

| Existing state | `run CONFIG` behavior |
| --- | --- |
| No matching evidence | Create the deterministic run and submit eligible work |
| Active work | Attach to durable state and monitor without duplicate submission |
| Pending or `license_blocked` | Continue the common retry/feeder policy |
| CPU complete with deferred collection | Without `--defer-collection`, collect and continue |
| Host publication complete | Build only missing declared packages/finalizers |
| Smoke children complete | Reuse compatible children and perform the paired finalizer |
| Benchmark input absent | Materialize it on the CPU login node before canary submission |
| Complete | Revalidate and report `REUSED`/`COMPLETE` without new work or transfer |
| Permanent scientific/configuration failure | Stop, report the blocker, and retain evidence |

## Collection and controller ownership

The collection modes are identical for every run kind:

| Mode | Host transfer and finalizers | CPU source |
| --- | --- | --- |
| Default | Run automatically | Removed only after validated cleanup authorization |
| `--keep-cpu-source` | Run automatically | Retained as an additional copy |
| `--defer-collection` | Do not run | Retained as the exclusive copy |

`--keep-cpu-source` and `--defer-collection` are mutually exclusive. Complete a
deferred run by invoking the same config without `--defer-collection`:

```bash
./scripts/generation_workflow.sh run "$TRANSIENT_CONFIG"   --defer-collection --cpu-host sricehpc01
./scripts/generation_workflow.sh run "$TRANSIENT_CONFIG"   --cpu-host sricehpc01
```

Use `--background` when the host controller must survive terminal or SSH
disconnection:

```bash
./scripts/generation_workflow.sh run "$AIRFLOW_ID_CONFIG"   --background --cpu-host sricehpc01
```

The clean host checkout creates one commit-pinned `tmux` child with durable
metadata and logs. Equivalent active controllers are reused. The printed
workflow-session ID supports:

```bash
./scripts/generation_workflow.sh background-status "$WORKFLOW_SESSION_ID"
./scripts/generation_workflow.sh background-list
TMUX_SESSION="$(./scripts/generation_workflow.sh   background-status "$WORKFLOW_SESSION_ID" | sed -n 's/^tmux_session=//p')"
tmux attach-session -t "$TMUX_SESSION"
```

Inside `tmux`, press Ctrl+B and then D to detach. A host reboot ends the
controller but not durable run evidence; invoke the same config again.

## Airflow ID Dataset campaign

`configs/generation/campaigns/steady_flow/id_dataset.yaml` is the only primary
Airflow training campaign. It has purpose `steady_flow_id_dataset` and contains
one independent 350-row maximin Latin-hypercube design per material. Designs are
not formed by merging smaller LHS runs.

| Material | Total | Train | Validation | ID test |
| --- | ---: | ---: | ---: | ---: |
| Lentil | 350 | 280 | 35 | 35 |
| Chickpea | 350 | 280 | 35 | 35 |
| Kidney Bean | 350 | 280 | 35 | 35 |
| **Total** | **1,050** | **840** | **105** | **105** |

Every material therefore has the same case count and an exact 80/10/10 split.
The campaign contains only seen materials and publishes one `steady_flow` ID
package. Field Pea, Rapeseed, Sunflower Seed, and parameter-OOD cases are not
eligible for Airflow training, validation, or ID-test model selection.

The primary Airflow Dataset uses only this campaign. It does not compose samples
from the transient production campaign or wait for another campaign source.

## Transient production campaign

`configs/generation/campaigns/transient_drying/family_generalization.yaml`
contains the following independent physical cases:

| Material and role | Cases | Train | Validation | ID test |
| --- | ---: | ---: | ---: | ---: |
| Lentil, seen ID | 160 | 128 | 16 | 16 |
| Chickpea, seen ID | 160 | 128 | 16 | 16 |
| Kidney Bean, seen ID | 160 | 128 | 16 | 16 |
| Field Pea, near-family OOD | 40 | - | - | - |
| Rapeseed, far-family OOD | 40 | - | - | - |
| Sunflower Seed, engineering stress | 40 | - | - | - |
| Parameter OOD | 0 | - | - | - |
| **Total** | **600** | **384** | **48** | **48** |

Parameter-OOD infrastructure remains available for future configs. Sunflower is
an engineering stress set, not a literature-secure far-OOD set.

Every transient `case.h5` continues to expose both `steady_flow` and
`transient_drying` learning views. Its stationary view contains the authoritative
coordinates, permeability tensor, porosity, inlet pressure, pressure, and
velocity fields. One transient physical case supplies one stationary view; time
states are not counted as independent Airflow samples.

The transient campaign currently declares only `transient_drying` packages. Its
stationary views are currently selected only for paired Airflow evaluation,
support analysis, comparison against COMSOL, and later coupled
Airflow-to-Drying evaluation. They do not enter the explicitly selected primary
Airflow training, validation, or ID-test Dataset, so every transient physical
case remains out of sample for the current Airflow operator.

A Dataset package is a neutral scientific artifact. Source simulation profile
and requested learning view do not impose a universal training prohibition. A
future transient-derived `steady_flow` package remains valid and may be used only
when an experiment names its exact Dataset ID. Package existence, a train split,
or a cross-profile source/view combination never selects a package
automatically. Adding a package declaration to an already completed compatible
campaign reuses the validated canonical `case.h5` publications and builds only
the missing package; it creates no additional simulation work units.

## Material pilot and Technical Smoke

The all-material pilot contains Lentil, Chickpea, Kidney Bean, Field Pea,
Rapeseed, and Sunflower Seed. It creates three cases per material: one nominal
reference and two natural-pilot cases, for 18 cases total. The count is owned by
YAML, and the declared material diagnostic runs in the common finalization
stage.

The Technical Smoke workflow runs ordinary steady and transient child campaigns
and then the paired finalizer. Completed children may be reused across source
commits only when their scientific config, mapping contract, templates, runtime,
and validated artifacts are compatible. If only a completed `case.h5` is invalid,
Full-Retention source exports may reconstruct it in isolated scratch without
running COMSOL. Recovery succeeds only when every retained input is hash-bound,
the reconstructed bytes restore the immutable published HDF5 identity, and the
whole campaign passes deep validation; a version-1 recovery receipt records the
operation. Missing, changed, or insufficient evidence fails closed with a precise
error. The paired finalizer reads source-export `logical_role`; it never reads a
`role` fallback. Fixed `T_flow_ref`, `p_ref`, and `p_out` values remain owned by
scientific fixed-value provenance.

Exactly one top-level completion is printed after both children and the paired
receipt validate. If children are already complete and only paired evidence is
missing, invoking the workflow config performs no new COMSOL solves.

## Core benchmark

The benchmark is one fast production-oriented core-selection phase. It uses the
same two deterministic scientific cases for every variant: one nominal/reference
case and one nontrivial natural-support case. The variants are 4, 8, 16, and 32
cores per case. Each wave runs its two cases concurrently, waits for both valid
successful measurements, and only then enables the next wave. The configured
production-core variant runs first; with the current 16-core setting the order is
16, 4, 8, 32. The first two-case wave is both the canary and part of the final
measurements, so there is no extra canary solve. The complete benchmark requires
exactly eight successful COMSOL measurements. It has no separate serial phase,
packed-node phase, or three-repetition design.

Canonical inputs are materialized and admitted on the CPU login node through the
normal input-preparation owner. The common controller owns submission,
monitoring, automatic temporary-license retry, resume, collection, compact
retention, and aggregation. Re-running the suite reuses both inputs and every
valid successful work unit, then submits only missing or invalid units; no public
variant-selection or manual retry workflow is required.

For every work unit the report separates scheduler queue, license wait, license
probe, canonical-input preparation, successful COMSOL process, export conversion,
publication, and total controller elapsed time. The primary comparison is only
`comsol_process_seconds`, beginning after allocation and successful license
checkout when the successful COMSOL process starts, and ending when that process
exits. Queue time, compute-slot waiting, license backoff, failed license probes,
earlier attempts, host collection, Dataset publication, controller polling,
conversion, and publication do not enter the primary runtime. License-only
attempts produce no successful runtime, core-hour, throughput, or ranking
observation.

The summary reports the fastest individual solve, the lowest median core-hours
per case, and the production recommendation separately. Compute-only estimated
node throughput is
`floor(cores_per_node / cores_per_case) * 3600 / median_comsol_process_seconds`.
The recommendation maximizes that estimate among variants that pass available
node-memory and scratch constraints. If no authoritative limit exists, the
estimate requires operator review. Variants within five percent of the best
throughput prefer lower median core-hours and then fewer cores. Scheduler and
license waits are reported separately and never alter the compute ranking.
Requested solver overlap and observed peak concurrency qualify the license
evidence; lack of overlap does not imply poor CPU scaling.

This replaces the former separate serial and packed-node benchmark concept. It
estimates production packing from measured per-case solver time and does not
fully measure an all-case packed node for every variant. The controller prints
proposed `cores_per_case` and cases-per-node values for manual review and never
edits production configuration. A real production recommendation exists only
after this suite runs successfully on the CPU cluster.

## License capacity and waiting

A generic message containing only “could not obtain license” is not enough to
trigger a retry. Temporary capacity requires the exact feature-bearing checkout
message plus strong evidence such as `Licensed number of users already reached`,
`License error -4`, or `FlexNet Licensing error: -4`. Missing features, invalid
license configuration, server problems, expired features, and unknown license
errors remain contextual hard failures.

A valid terminal result wins over an earlier warning: COMSOL exit 0, complete
validated exports and HDF5, and successful publication remain successful.
A genuine pre-solve capacity event becomes `license_blocked`, does not count as a
scientific failure, and does not consume `maximum_failed_cases`.

The controller waits outside compute allocations with bounded exponential
backoff. `maximum_wait_seconds: null` means it waits until capacity becomes
available or the operator cancels. Oldest blocked work runs first, at most one
license probe is active per top-level run, and fresh work cannot repeatedly
leapfrog an eligible block. Campaigns and benchmarks use the same policy.
Current OST Slurm exposes no verified license resources, so submissions use
neither `sbatch --licenses` nor COMSOL `-usebatchlic`.

Each blocked work unit owns one mutable `license_wait.json` with schema version
1. It records work-unit and scientific identities, feature, error code, exact
matched signatures, COMSOL exit code, solver-progress and export flags, first
and latest blocked timestamps, retry count, latest and bounded recent job IDs,
next retry time, cumulative wait, and a bounded raw excerpt. A new license-only
event creates no scientific attempt directory and copies no canonical input.

## Attempt retention

Retention is based on failure stage, not campaign label:

| Outcome | Retained evidence |
| --- | --- |
| License block before solver progress | One compact `license_wait.json`; no input CSV, model, exports, HDF5, or workspace copy |
| Solver failure after progress | Bounded stdout/stderr, solver log, status, timing, command, resource and identity evidence |
| Conversion failure with complete exports | Exact exports needed for conversion replay plus small runtime evidence |
| Publication failure after valid HDF5 | `case.h5`, status, timing, and small provenance needed for replay |
| Successful compact case | Canonical `case.h5`, bounded logs/status/timing/provenance, and `_SUCCESS`; no CSV or `solved.mph` |
| Successful full Smoke/Pilot case | Validated successful diagnostics required by the full-retention policy |

Smoke and Pilot do not implicitly give failed attempts full-workspace retention.
Guarded reconciliation can compact an existing license-only attempt only after
strong classification, proof that solver progress and unique exports are absent,
and independent admission of its canonical input. It retains an immutable
compaction audit and reports reclaimed bytes.

## Storage, identity, and Dataset integrity

`STORAGE_ROOT` is the sole storage-root override:

```text
STORAGE_ROOT/
├── 01_generation/
│   ├── raw/          # canonical generated inputs
│   ├── processed/    # canonical successful case.h5 publications
│   ├── attempts/     # bounded failed-attempt/replay evidence
│   └── meta/         # run, transfer, benchmark, Smoke, and session evidence
├── 02_datasets/
│   ├── packages/     # immutable derived learning payloads/indexes
│   ├── meta/         # package manifests and receipts
│   └── .state/       # publication coordination
└── 03_experiments/
```

Canonical HDF5 stays below `01_generation/processed`; Dataset packages never
copy `case.h5`. Package publication verifies path containment, SHA-256,
simulation profile, learning view, case identity, channel contract, units,
shape, grid, and finite values before atomic publication. Loader smoke runs
before CPU cleanup authorization.

Scientific identity is separate from filenames, storage paths, display labels,
and background-session identity. Exact resolved config, source commit, seeds,
input and simulation identities, hashes, runtime evidence, and receipts bind each
stage. Routine validation checks manifests, membership, sizes, and digest chains;
deep validation rehashes and scientifically reopens retained payloads.

## Status, cancellation, and cleanup

The public administrative commands are:

```bash
./scripts/generation_workflow.sh status "$AIRFLOW_ID_CONFIG"
./scripts/generation_workflow.sh status "$GENERATION_RUN_ID"
./scripts/generation_workflow.sh cancel "$GENERATION_RUN_ID"
./scripts/generation_workflow.sh cancel "$GENERATION_RUN_ID" --force
./scripts/generation_workflow.sh cleanup "$GENERATION_RUN_ID" --confirm
./scripts/generation_workflow.sh background-status "$WORKFLOW_SESSION_ID"
./scripts/generation_workflow.sh background-list
```

Status reports successful, running, scheduler-pending, license-blocked,
never-started, and failed units;
current stage; CPU source; host publication; package/finalizer state; and the
same-config continuation action. The first Ctrl+C after campaign launch requests
graceful cancellation and keeps monitoring durable terminal evidence. A second
Ctrl+C requests force cancellation. Before a run identity exists, no campaign
cancellation is armed.

Cleanup is destructive only for the exact run-owned CPU directories admitted by
its cryptographic authorization. It does not remove the canonical host
publication, another run’s shared source, or another user’s files.

Run `./scripts/generation_workflow.sh --help` for the current option syntax.
