# Generation Workflow

This is the operator guide for configuring, launching, monitoring, resuming,
collecting, and cleaning up Generation campaigns. Scientific definitions,
equations, parameter interpretation, and numerical-validation rationale live in
the [scientific parameter reference](generation_parameter_reference.md).

## Quick start

Run public workflow commands from the bare <code>hpc115</code> repository
checkout.

> Generation pins committed HEAD. Uncommitted development changes are ignored.
> Commit a change before launch when Generation must use it.

The pinned source is used by the wrapper, local Docker commands, and remote CPU
work. The configured <code>STORAGE_ROOT</code> remains the writable evidence
root.

Set the common paths:

~~~bash
export STORAGE_ROOT="$(realpath ../storage)"
STEADY_CAMPAIGN=configs/generation/campaigns/steady_flow/family_generalization.yaml
TRANSIENT_CAMPAIGN=configs/generation/campaigns/transient_drying/family_generalization.yaml
PILOT_CAMPAIGN=configs/generation/campaigns/transient_drying/pilot_check.yaml
~~~

1. Edit the authoritative YAML under <code>configs/generation</code>, review the
   change, and commit it.

2. Preview and perform setup for the exact committed source:

~~~bash
./scripts/generation_workflow.sh setup-cpu --cpu-host sricehpc01
./scripts/generation_workflow.sh setup-cpu --cpu-host sricehpc01 --execute
~~~

3. Validate configuration and run the non-solving preflight:

~~~bash
./scripts/docker_python.sh -m src.generation.cli.cli_generation validate-config   "$STEADY_CAMPAIGN" --allow-incomplete
./scripts/docker_python.sh -m src.generation.cli.cli_generation validate-config   "$TRANSIENT_CAMPAIGN" --allow-incomplete
./scripts/generation_workflow.sh preflight "$STEADY_CAMPAIGN"
./scripts/generation_workflow.sh preflight "$TRANSIENT_CAMPAIGN"
~~~

Preflight submits one environment-only Slurm allocation. It starts no COMSOL
solve.

4. Run the paired Technical Smoke and inspect its receipt:

~~~bash
./scripts/generation_workflow.sh smoke --keep-cpu-source
~~~

5. Benchmark transient core counts:

~~~bash
./scripts/generation_workflow.sh benchmark-cores --cpu-host sricehpc01
~~~

Review the recommendation, record the selected
<code>cluster.cores_per_case</code> in
<code>configs/generation/execution/cluster_cpu.yaml</code>, and commit it. The
benchmark does not edit Production configuration.

6. Run the transient pilot:

~~~bash
./scripts/generation_workflow.sh pilot-check "$PILOT_CAMPAIGN"
~~~

Use <code>--cases-per-material 1</code> for the supported small diagnostic or
<code>--keep-cpu-source</code> when retained CPU source is needed for diagnosis.

7. Review readiness, then launch Production:

~~~bash
./scripts/docker_python.sh -m src.generation.cli.cli_generation readiness-report   "$STEADY_CAMPAIGN" "$TRANSIENT_CAMPAIGN" --run-static-sentinels
./scripts/generation_workflow.sh all "$STEADY_CAMPAIGN"
./scripts/generation_workflow.sh all "$TRANSIENT_CAMPAIGN"
~~~

The <code>all</code> workflow enforces configuration, current Technical-Smoke,
runtime, publication, Dataset, and cleanup gates for the selected profile.

## Where do I change what?

User decisions belong in configuration. Resolved values and identities belong
in generated evidence.

| Decision | Authoritative owner |
| --- | --- |
| Grid, time axis, fixed science, validation tolerances, storage contract | <code>configs/generation/common.yaml</code> |
| Parameter meanings, units, transforms, sampling blocks, OOD eligibility | <code>configs/generation/registry.yaml</code> |
| Material values, natural supports, OOD supports, source references | <code>configs/generation/materials/*.yaml</code> |
| Boundary schedules and physical constraints | <code>configs/generation/operations/fixed_bed.yaml</code> |
| Profile exports, mappings, and COMSOL template locator | <code>configs/generation/profiles/*.yaml</code> |
| Expected template-byte digest | Adjacent <code>.sha256</code> sidecar |
| Campaign purpose, roles, counts, membership, seeds, Dataset requests | <code>configs/generation/campaigns/&lt;profile&gt;/*.yaml</code> |
| Resources, feeder, retries, timeout, retention, CPU site | <code>configs/generation/execution/cluster_cpu.yaml</code> |
| Core-benchmark variants and repetitions | <code>configs/generation/benchmarks/transient_core_scaling/*.yaml</code> |
| Scientific interpretation | [Scientific parameter reference](generation_parameter_reference.md) |

Use <code>validate-config --allow-incomplete</code>,
<code>readiness-report</code>, and <code>plan</code> to inspect current resolved
values. Do not copy derived inventories into this guide.

### Updating a COMSOL template

The selected profile YAML owns the repository-relative <code>template:</code>
value. Its expected SHA-256 is the adjacent <code>.sha256</code> file. The
transient template must read only <code>t</code>, <code>T_in_bc</code>, and
<code>omega_in_bc</code> from the schedule table, derive inlet relative humidity
after primitive interpolation, and continue exporting the solved
<code>mt.phi</code> field.

1. Save the intended <code>.mph</code> file and update the profile locator if it
   was renamed.
2. Regenerate the sidecar deliberately:

~~~bash
TEMPLATE="<value copied from the selected profile YAML>"
SIDECAR="$(dirname "$TEMPLATE")/$(basename "$TEMPLATE" .mph).sha256"
sha256sum -- "$TEMPLATE" | cut -d ' ' -f 1 > "$SIDECAR"
~~~

3. Run <code>validate-config --allow-incomplete</code>, the relevant static
   checks, and Technical Smoke.
4. Commit the profile, template, and sidecar together.

Validation and execution never accept or rewrite a new digest automatically.

## Identity and execution safety

Generation distinguishes case inputs, simulation batches, campaign science,
campaign runs, Dataset packages, and operational provenance. Scientific identity
is separate from storage paths, directory names, display labels, and
configuration filenames.

Planning, launch, feeding, and resume remain bound to the pinned commit.
Completed readers verify persisted contracts and artifact bytes. Node-local
scratch is temporary; do not manually delete CPU source or transfer evidence;
use the gated cleanup command.

## Generate and inspect canonical inputs

<code>generate-input-cases</code> publishes or exactly reuses canonical inputs
without calling COMSOL or Slurm. A bounded exact-batch request is:

~~~bash
./scripts/docker_python.sh -m src.generation.cli.cli_generation generate-input-cases   "$TRANSIENT_CAMPAIGN"   --only-batch transient_drying__lentil__natural   --case-start 1 --case-count 10   --git-commit "$(git rev-parse HEAD)" --storage-root "$STORAGE_ROOT"
~~~

Change only the selection arguments for common alternatives:

| Selection | Arguments |
| --- | --- |
| Technical Runtime Smoke | <code>configs/generation/campaigns/transient_drying/technical_smoke.yaml --all-batches --all-cases</code> |
| Every natural family batch | <code>"$TRANSIENT_CAMPAIGN" --all-batches --only-regime natural --all-cases</code> |
| Validate without publication | Add <code>--dry-run</code> |

[<code>generation_input_eda.ipynb</code>](../notebooks/generation_input_eda.ipynb)
is read-only. It admits persisted canonical input manifests and never plans,
generates, locks, publishes, or inspects completed solver output.

Generate cases first, then rerun the notebook. Completed solver output belongs
to completed-output EDA. Current ramp-disabled transient inputs persist the exact
hourly primitive schedule from time zero; the notebook may evaluate denser
display-only curves but never writes extra support. A deliberately ramp-enabled
campaign adds its documented primitive rejoin row without changing regular
output times. Four-column schedules containing <code>phi_in_bc</code> are stale
and fail admission; regenerate those input cases under the current contract.
Scientific startup and schedule semantics are owned by the
[scientific parameter reference](generation_parameter_reference.md#inlet-schedule).

## Operational stages

Each attempt records five separate stages: solver, exports, conversion,
diagnostics, and publication. Solver success does not imply that later stages
succeeded. Structural validity, identities, required files, array shapes, finite
required values, ordering, hashes, path containment, and atomic publication are
blocking and fail closed. Finite scientific plausibility observations are
advisory: they remain visible as complete quality-flag records but do not turn
an otherwise valid processed publication into a failure.

A configured target event may end a transient solve before the next regular
output time. The accepted event state is a valid irregular final state when the
time axis remains strictly increasing and the final status is consistent with
the stop condition. This does not change the current ramp-disabled scientific
configuration.

### Technical Smoke

<code>smoke</code> runs maintained steady and transient technical cases with real
COMSOL. Before either profile is planned or launched, it prepares or reuses
the exact commit-bound canonical inputs and requires their complete admission.
It proves the configured technical path through validated inputs, exports,
canonical HDF5, immutable packages, and loader access; it does not establish
experimental or scientific validity. Its retained receipt and case evidence
support diagnosis and are required by broader workflow gates.

### Core benchmark

<code>benchmark-cores</code> compares configured core variants and reports the
evidence needed to select <code>cluster.cores_per_case</code>; it does not edit
configuration. Retry one failed variant with
<code>benchmark-cores --variant &lt;variant-id&gt;</code>.

### Pilot

<code>pilot-check</code> exercises natural-support cases across configured
materials and retains a diagnostic receipt. Accepted work follows the configured
gated cleanup policy; incomplete or invalid evidence is not cleanup-eligible.

### Production, attempt budget, and feeder breaker

Production uses one ordinary non-exclusive Slurm job per case. The total
controlled attempt budget is 3600 seconds: ordinary compute ends at 3300 seconds,
leaving a 300-second graceful-stop reserve inside that same hour. Slurm requests
<code>01:05:00</code> only to allow worker publication and cleanup around the
one-hour attempt. The COMSOL stop owner writes exactly <code>Stop 2\n</code>
once to the status file, requests <code>Cancel</code>, and force-escalates only
when the solver does not leave within the reserve.

<code>maximum_failed_cases</code> is the sole feeder circuit breaker. A value
<code>N</code> permits up to <code>N</code> distinct unresolved
<code>failed</code> or <code>timed_out</code> cases and stops feeding new,
never-started work at <code>N + 1</code>. Already active or pending jobs remain
active and monitored. Export, conversion, publication, cancellation,
interruption, license retry, warning, and quality-flag states do not count. The
maintained value is <code>0</code>.

Temporary license-capacity exhaustion uses bounded retry/backoff for the same
deterministic case without holding a compute allocation during backoff. A
campaign in <code>waiting_retry</code> remains nonterminal. There are no automatic
reserve cases and no automatic rerun of scientific solver failures. The
foreground <code>all</code> or <code>resume</code> process reconciles the campaign;
already submitted Slurm jobs continue if that process or terminal disconnects.

## Monitor campaigns

Inspect the full persisted case summary from <code>hpc115</code>:

~~~bash
GENERATION_RUN_ID="<campaign_run_id>"
./scripts/generation_workflow.sh status "$GENERATION_RUN_ID"
~~~

The summary may report the active case, Slurm job, node, phase, simulated time,
adaptive step, <code>Tfail</code>, and <code>NLfail</code>. These are observational
progress indicators, not a convergence, completion, retry, or cleanup verdict.

Useful durable locations are:

- scheduler stdout/stderr: the campaign manifest's
  <code>scheduler_log_directory</code>;
- successful combined solver log:
  <code>01_generation/processed/&lt;batch_storage_name&gt;/&lt;case_id&gt;/solver.log</code>;
- immutable unsuccessful attempt evidence:
  <code>01_generation/attempts/&lt;batch_storage_name&gt;/&lt;case_id&gt;/&lt;campaign_run_id&gt;/attempt_0001/</code>;
- authoritative attempt identity, stage states, retention inventory, hashes,
  metrics, and quality flags: that directory's <code>attempt.json</code>;
- successful postprocessing replay and recovery-payload cleanup audit:
  <code>replay.json</code>.

Later attempts use <code>attempt_0002</code>, <code>attempt_0003</code>, and so
on. Earlier attempts are immutable, and attempt directories never contain
<code>_SUCCESS</code>.

Use <code>campaign-status --format json</code> for the complete machine-readable
case inventory.

## Evidence, readiness, and retention

<code>readiness-report</code> is the launch-state authority. Production is ready
only when resolved campaigns, static checks, and current profile-specific
Technical-Smoke evidence pass. All campaign-lifecycle receipts introduced by
this workflow use schema version <code>1</code> and are admitted against their
exact schema and identities.

Technical Smoke and Pilot retain full unsuccessful-attempt bundles. Production
retains compact logs, timings, provenance, metrics, and hashes while deliberately
omitting unrelated large model and export files. Conversion failures retain the
required exports temporarily; publication failures also retain the converted
payload and publication evidence. A successful replay verifies processed
publication, removes only the declared large temporary recovery payload, and
keeps the small attempt and replay audit.

Collection transfers the exact campaign-scoped attempt directories and their
hashes along with raw, processed, batch metadata, and campaign metadata. It does
not copy another campaign's attempt tree. The ordinary authorized CPU cleanup
removes only its enumerated raw, processed, and batch-metadata source directories;
canonical attempt directories remain protected.

<code>01_generation</code> is the canonical simulation archive, while
<code>02_datasets</code> contains immutable learning packages. Attempts are
excluded from processed membership, batch success, Dataset identity, training
readiness, and completed-output EDA. Processed status and attempt evidence retain
stable batch, case, input-generation, case-input, simulation, campaign, source
commit, and attempt keys for later outcome EDA; this workflow does not perform
outcome-driven parameter analysis or automatic retuning.

## Resume, retry, cancel, collect, and cleanup

Resume reuses successful processed publications, submits permitted
never-started work, and restarts cancelled or interrupted cases from time zero.
It replays conversion or publication without COMSOL when the exact retained
payload is valid. It does not silently rerun <code>failed</code>,
<code>timed_out</code>, or <code>exports_failed</code> solver work. If no automatic
action remains, it prints the explicit <code>retry-case</code> recovery guidance.

~~~bash
./scripts/generation_workflow.sh status "$GENERATION_RUN_ID"
./scripts/generation_workflow.sh resume "$GENERATION_RUN_ID"
./scripts/generation_workflow.sh retry-case \
  "$GENERATION_RUN_ID" "<batch_name>" "<case_id>"
~~~

<code>retry-case</code> accepts only one eligible unresolved case, requires the
original solver commit, template, science, canonical input generation, and case
identities, and allocates a new immutable attempt. There is no
<code>retry-all</code>. A terminal <code>completed_with_failures</code> campaign
blocks terminal validation, collection, and Dataset publication until every case
has a valid processed publication.

The first Ctrl+C after campaign launch prints exactly:

~~~text
Graceful campaign cancellation requested.
Press Ctrl+C again to force cancellation.
~~~

It persists <code>cancel_requested</code>, cancels pending jobs, signals running
workers through the controlled-stop path, keeps rendering status, and waits for
their durable terminal evidence. A second Ctrl+C invokes force cancellation and
stops waiting. Before a run ID exists, no cancellation trap or receipt exists.
The public commands use those same Python-owned paths:

~~~bash
./scripts/generation_workflow.sh cancel "$GENERATION_RUN_ID"
./scripts/generation_workflow.sh cancel "$GENERATION_RUN_ID" --force
~~~

Transfer and publish a successful terminal campaign, then build requested
Datasets:

~~~bash
./scripts/generation_workflow.sh collect "$GENERATION_RUN_ID"
./scripts/generation_workflow.sh build-datasets "$GENERATION_RUN_ID"
~~~

Preview cleanup, review the exact cryptographic authorization, then confirm:

~~~bash
./scripts/generation_workflow.sh cleanup "$GENERATION_RUN_ID"
./scripts/generation_workflow.sh cleanup "$GENERATION_RUN_ID" --confirm
~~~

Cleanup is destructive only for explicitly authorized CPU directories. Canonical
GPU Generation, attempts, and Dataset publications remain intact.

## Command reference

All wrapper commands run on bare <code>hpc115</code>.

| Command | Purpose |
| --- | --- |
| <code>setup-cpu [--execute]</code> | Preview or perform remote checkout/environment setup |
| <code>preflight CAMPAIGN</code> | Check login and compute environment without a scientific solve |
| <code>plan CAMPAIGN</code> | Print resolved Slurm work without launching |
| <code>smoke</code> | Run paired Technical-Smoke campaigns |
| <code>benchmark-cores [--variant ID]</code> | Run or resume transient core scaling |
| <code>pilot-check CAMPAIGN</code> | Run the transient pilot lifecycle |
| <code>launch CAMPAIGN</code> | Submit the low-level campaign primitive |
| <code>all CAMPAIGN</code> | Run Production through Dataset receipts and gated cleanup |
| <code>status [RUN_ID]</code> / <code>accounting RUN_ID</code> | Inspect workflow or scheduler evidence |
| <code>resume RUN_ID</code> | Apply the reuse, restart, and postprocessing-replay matrix |
| <code>retry-case RUN_ID BATCH CASE</code> | Explicitly rerun one eligible unresolved case |
| <code>validate RUN_ID</code> | Revalidate terminal CPU evidence |
| <code>collect RUN_ID</code> | Transfer and publish terminal Generation evidence |
| <code>build-datasets RUN_ID</code> | Build/reuse requested immutable packages |
| <code>cancel RUN_ID [--force]</code> | Request graceful or force cancellation through the shared owner |
| <code>cleanup RUN_ID [--confirm]</code> | Preview or execute authorized CPU-source cleanup |

Run <code>./scripts/generation_workflow.sh --help</code> for complete current
options. Resource and feeder decisions remain configuration-owned.

## Troubleshooting

- Exit status 2 from configuration, plan, or readiness means a fail-closed gate
  remains. Follow the reported file and key.
- If Technical Smoke fails, inspect retained case, export, mapping, and solver
  evidence. Partial smoke evidence is not Production evidence.
- Failed collection retains marked staging and CPU source. Use
  <code>status</code> and <code>resume</code>; never move partial data into
  <code>01_generation</code> manually.
- <code>completed_with_failures</code> means no automatic resumable case remains.
  Inspect the failed-case stage and use <code>retry-case</code> only for one
  deliberately selected eligible case.
- A benchmark refusal usually indicates missing current smoke evidence, a failed
  repetition, or conflicting successful evidence. Use the printed recovery
  command.
- Temporary license-capacity events retry within the configured bound. Terminal
  license or configuration failures require correction, not evidence deletion.
- If a configuration or source change is absent from a run, verify that it was
  committed before the workflow invocation.

The project entry point is the [README](../README.md).
