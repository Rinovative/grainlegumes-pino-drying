# Generation Workflow

This is the practical guide for configuring, running, inspecting, resuming, and
cleaning up Generation campaigns. Scientific definitions, equations, evidence
classification, and modelling assumptions live in the
[scientific parameter reference](generation_parameter_reference.md).

## Quick start

Run every public workflow command from the bare `hpc115` repository checkout.
Native operations require the exact committed Git revision and a clean worktree.
The commands below invoke real Slurm/COMSOL work where noted; run them only after
reviewing their gates.

Set the common paths:

```bash
export STORAGE_ROOT="$(realpath ../storage)"
STEADY_CAMPAIGN=configs/generation/campaigns/steady_flow/family_generalization.yaml
TRANSIENT_CAMPAIGN=configs/generation/campaigns/transient_drying/family_generalization.yaml
PILOT_CAMPAIGN=configs/generation/campaigns/transient_drying/pilot_check.yaml
```

1. Edit the authoritative YAML under `configs/generation`, then review and
   commit the intended configuration.

2. Preview and perform CPU setup for the exact revision:

```bash
./scripts/generation_workflow.sh setup-cpu --cpu-host sricehpc01
./scripts/generation_workflow.sh setup-cpu --cpu-host sricehpc01 --execute
```

3. Resolve configuration and run the non-solving preflight:

```bash
./scripts/docker_python.sh -m src.generation.cli.cli_generation validate-config \
  "$STEADY_CAMPAIGN" --allow-incomplete
./scripts/docker_python.sh -m src.generation.cli.cli_generation validate-config \
  "$TRANSIENT_CAMPAIGN" --allow-incomplete
./scripts/generation_workflow.sh preflight "$STEADY_CAMPAIGN"
./scripts/generation_workflow.sh preflight "$TRANSIENT_CAMPAIGN"
```

Preflight submits one environment-only Slurm allocation to check the compute
environment, but it starts no COMSOL or scientific solve.

4. Run the paired Technical Smoke and inspect the printed receipt:

```bash
./scripts/generation_workflow.sh smoke --keep-cpu-source
```

5. Benchmark transient core counts and review the generated recommendation:

```bash
./scripts/generation_workflow.sh benchmark-cores --cpu-host sricehpc01
```

The benchmark never edits Production configuration. Record the reviewed
`cluster.cores_per_case` decision in
`configs/generation/execution/cluster_cpu.yaml` and commit it before continuing.

6. Run the transient pilot:

```bash
./scripts/generation_workflow.sh pilot-check "$PILOT_CAMPAIGN"
```

Use `--cases-per-material 1` for the supported small diagnostic, or
`--keep-cpu-source` when retained source is needed for debugging.

7. Inspect static/configuration readiness, then launch Production:

```bash
./scripts/docker_python.sh -m src.generation.cli.cli_generation readiness-report \
  "$STEADY_CAMPAIGN" "$TRANSIENT_CAMPAIGN" --run-static-sentinels
./scripts/generation_workflow.sh all "$STEADY_CAMPAIGN"
./scripts/generation_workflow.sh all "$TRANSIENT_CAMPAIGN"
```

Without an exact COMSOL-version record and paired runtime receipt, the shown
report intentionally leaves real-evidence gates unresolved. The `all` workflow
enforces the complete current gate for its selected profile before launch.
Static tests, template hashes, or an old receipt are not substitutes.

## Where do I change what?

User decisions belong in configuration; derived values belong in resolved
configuration and generated evidence.

| Decision | Authoritative owner |
| --- | --- |
| Grid, time axis, fixed science, validation tolerances, storage contract | `configs/generation/common.yaml` |
| Parameter meanings, units, transforms, sampling blocks, OOD eligibility | `configs/generation/registry.yaml` |
| Material values, natural supports, OOD supports/records, source references | `configs/generation/materials/*.yaml` |
| Boundary conditions, schedule supports, and physical constraints | `configs/generation/operations/fixed_bed.yaml` |
| Profile inputs, exports, mappings, and template identity | `configs/generation/profiles/*.yaml` |
| Campaign purpose, material roles, counts, membership, seeds, Dataset requests | `configs/generation/campaigns/<profile>/*.yaml` |
| Per-case resources, feeder, retry, timeout, retention, CPU site | `configs/generation/execution/cluster_cpu.yaml` |
| Core-benchmark variants and repetitions | `configs/generation/benchmarks/transient_core_scaling/*.yaml` |
| Bibliographic metadata and locators | `configs/generation/sources.yaml` |
| Learning, optimization, and evaluation choices | `configs/learning/<task>` |
| Scientific interpretation | [Scientific parameter reference](generation_parameter_reference.md) |
| Exact invariants and implementation | `src/generation`, `src/datasets`, and focused tests |

Use `validate-config --allow-incomplete`, `readiness-report`, and `plan` to
inspect current resolved supports, counts, seeds, OOD allocation, identities,
resources, and gates. Do not maintain those derived snapshots in documentation.

## Execution model and safety

| Domain | Responsibility |
| --- | --- |
| Bare `hpc115` | Public shell control plane, exact Git state, Docker, SSH, rsync, and canonical `STORAGE_ROOT` |
| `hpc115` Docker | Configuration resolution, local evidence admission, Dataset publication, and loader checks |
| `sricehpc01` login | Exact checkout, native Generation environment, durable CPU storage, and Slurm control |
| CPU compute node | Per-case scratch materialization, COMSOL, conversion, validation, and durable result publication |

The wrapper translates repository-relative configuration paths across these
domains. Users should not substitute Docker paths such as `/workspace/repo` for
bare-host or CPU paths. Docker is not required on the CPU cluster.

CPU jobs use owned temporary scratch and publish evidence before scratch cleanup.
Scientific storage is durable; node-local `$TMPDIR` is not. The normal flow is:

```text
hpc115 control -> sricehpc01/Slurm -> CPU Generation source
               -> hpc115 transfer staging -> 01_generation -> 02_datasets
```

Never delete CPU source or transfer evidence manually. Use the workflow cleanup
command so job state, inventories, hashes, transfer receipts, Dataset receipts,
and publication identity are checked together.

## Operational stages

### Configuration, setup, and preflight

`validate-config` is the main review surface. It reports the effective campaign
without solving and identifies unresolved values by owner. `plan` additionally
prints the resolved Slurm work and intentionally fails while Production gates
remain unresolved.

`setup-cpu` is a dry-run until `--execute`. It creates or updates the remote
checkout, storage root, and native CPU environment for the exact local commit.
Preflight checks the login environment and then submits one environment-only
Slurm job to verify modules, executables, project imports, templates, scratch,
and durable storage. It performs no COMSOL solve.

### Technical Smoke

`smoke` runs both maintained profile campaigns with real COMSOL. It validates
case inputs, expected exports, canonical HDF5, immutable packages, and loader
access before publishing profile-scoped Technical-Smoke evidence. The evidence
is bound to the current profile mapping, template, COMSOL runtime, verifier, and
smoke campaign. It proves the technical path, not experimental validity.

Technical-Smoke inputs, raw exports, solved models, logs, scheduler evidence,
and CPU source are retained. A failed smoke never publishes Production
evidence; inspect its retained case and export diagnostics rather than deleting
or recreating evidence manually.

Transient admission uses distinct numerical contracts for initial-state
consistency, bulk-moisture consistency, and float32 storage fidelity. Their
scientific purpose and single documentation owner are described in the
[scientific reference](generation_parameter_reference.md#numerical-validation-contracts);
the authored values remain in `configs/generation/common.yaml`.

### Core benchmark

The transient core benchmark measures the same nominal case across the
configuration-owned core variants and repetitions. It reports single-case time,
parallel efficiency, and COMSOL core-hour comparisons so a user can choose
`cluster.cores_per_case`. Variants run serially to isolate each measurement.

Review the transferred summary and its recommended, fastest, and
efficiency-oriented settings. The recommendation excludes queue wait and never
changes the Production YAML automatically. A failed variant can be retried with
`benchmark-cores --variant <variant-id>` without discarding successful
evidence.

### Pilot

The transient pilot exercises natural-support cases for every configured
material before Production. Its receipt summarizes runtime and conversion
success, drying duration, physical bounds, balances, extrema, schedule behavior,
and observed storage. The campaign YAML owns the normal cases per material; the
small diagnostic override is explicit on the command line.

The normal pilot cleans validated CPU source and transfer staging after the
receipt is accepted. `--keep-cpu-source` retains CPU source for investigation.
Active, incomplete, hash-invalid, or insufficiently retained evidence is never
eligible for cleanup.

### Production and license retry

Production uses one ordinary non-exclusive Slurm job per scientific case. The
execution YAML owns per-case resources, the pending buffer, optional running
cap, polling, timeout, failure budget, and retention. Slurm owns placement and
concurrency; the workflow only feeds and reconciles this campaign.

Temporary floating-license capacity exhaustion is retried automatically with
bounded backoff from the execution configuration. The same deterministic case
is retried. The controller waits after the failed allocation has ended, so
backoff does not hold a compute node, and retryable capacity events do not
immediately consume the scientific failure budget. Exhausted retry capacity and
terminal license or configuration errors fail normally.

The foreground `all` or `resume` process performs polling and queue feeding.
If it disconnects, already submitted Slurm jobs continue; disconnecting does not
cancel them. Resume reconciles persisted job IDs, scheduler accounting, and
validated case evidence before submitting anything else.

## Evidence, readiness, and retention

`readiness-report` is authoritative for the current launch state. Production is
ready only when the resolved campaign, static scientific checks, and current
profile-specific Technical-Smoke evidence pass. The active Generation YAML
schemas are version 1; content changes still change identities and invalidate
stale evidence where the contract requires it.

| Purpose | Successful evidence and normal cleanup |
| --- | --- |
| Technical Smoke | Retains raw exports, solved models, logs, packages, receipts, and CPU source |
| Pilot | Retains the accepted pilot receipt and GPU publication; cleans verified CPU source/staging by default |
| Production | Retains canonical `01_generation`, requested `02_datasets`, and receipts; cleans verified CPU source by default |
| Failed/incomplete work | Retention follows the purpose-specific execution policy; cleanup remains blocked until recovery or explicit authorized cleanup |

`01_generation` is the canonical simulation archive.
`02_datasets` contains immutable learning packages addressed by `dataset_id`.
`03_experiments` contains training, tuning, and evaluation outputs. Dataset
publication never removes canonical Generation source.

## Resume, cancel, and cleanup

Inspect and resume a persisted run with:

```bash
GENERATION_RUN_ID="<campaign_run_id>"
./scripts/generation_workflow.sh status "$GENERATION_RUN_ID"
./scripts/generation_workflow.sh resume "$GENERATION_RUN_ID"
```

Resume reuses valid successful cases and never duplicates active or pending
attempts. Accounting uncertainty fails closed.

Cancellation is explicit and affects every persisted active attempt:

```bash
./scripts/generation_workflow.sh cancel "$GENERATION_RUN_ID"
```

Preview CPU-source cleanup, review the resolved deletion plan, then confirm it:

```bash
./scripts/generation_workflow.sh cleanup "$GENERATION_RUN_ID"
./scripts/generation_workflow.sh cleanup "$GENERATION_RUN_ID" --confirm
```

Cleanup is destructive only on the authorized CPU source. Canonical GPU
Generation and Dataset publications remain intact.

## Command reference

All wrapper commands run on bare `hpc115`.

| Command | Purpose |
| --- | --- |
| `setup-cpu [--execute]` | Preview or perform exact remote checkout/environment setup |
| `preflight CAMPAIGN` | Check login and compute environment; submits no scientific solve |
| `plan CAMPAIGN` | Print resolved Slurm work without launching |
| `smoke` | Run paired real Technical-Smoke campaigns and publish technical evidence |
| `benchmark-cores [--variant ID]` | Run or resume isolated transient core scaling |
| `pilot-check CAMPAIGN` | Run the transient pilot and its analysis/cleanup lifecycle |
| `launch CAMPAIGN` | Submit the low-level campaign primitive |
| `all CAMPAIGN` | Run the full Production lifecycle through Dataset receipts and gated cleanup |
| `status RUN_ID` / `accounting RUN_ID` | Inspect persisted workflow or scheduler evidence |
| `resume RUN_ID` | Reconcile and continue incomplete work |
| `validate RUN_ID` | Revalidate terminal CPU campaign evidence |
| `collect RUN_ID` | Transfer and atomically publish terminal Generation evidence |
| `build-datasets RUN_ID` | Build/reuse requested immutable packages and check loaders |
| `cancel RUN_ID` | Cancel persisted active Slurm attempts |
| `cleanup RUN_ID [--confirm]` | Preview or execute digest-authorized CPU-source cleanup |

Run `./scripts/generation_workflow.sh --help` for current options. Resource and
feeder decisions have no Production CLI override; edit the execution YAML.
`--only-batch` selects one declared batch, while
`--skip-extreme-family-ood` omits a declared extreme-family group for one
execution and cannot be combined with it.

## Troubleshooting

- A dirty worktree or commit mismatch blocks remote planning and launch. Review
  and commit the intended changes, then rerun the same command.
- Exit status 2 from configuration, plan, or readiness commands means a
  fail-closed gate remains. Follow the reported file/key owner; do not invent a
  CLI default.
- If Technical Smoke fails, inspect retained case, raw-export, and mapping
  diagnostics. A failed or partial smoke is not Production evidence.
- Failed collection retains marked staging and CPU source. Use `status` and
  `resume`; never move partial data into `01_generation` manually.
- A benchmark refusal usually indicates missing current smoke evidence, a failed
  repetition, or conflicting successful evidence. Use the printed variant
  recovery command.
- Temporary license-capacity events retry within the configured bound. Terminal
  license/configuration failures require correction; deleting attempt evidence
  is not a recovery method.
- Smoke packages are technical evidence and cannot be selected by normal
  training configuration.

The project entry point is the [README](../README.md); scientific interpretation
is in the [parameter reference](generation_parameter_reference.md).
