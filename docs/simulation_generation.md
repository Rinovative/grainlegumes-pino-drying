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
value. Its expected SHA-256 is the adjacent <code>.sha256</code> file.

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
to completed-output EDA. Transient startup and schedule semantics are owned by
the [scientific parameter reference](generation_parameter_reference.md#inlet-schedule).

## Operational stages

### Technical Smoke

<code>smoke</code> runs maintained steady and transient technical cases with real
COMSOL. It proves the configured technical path through validated inputs,
exports, canonical HDF5, immutable packages, and loader access; it does not
establish experimental or scientific validity. Its retained receipt and case
evidence support diagnosis and are required by broader workflow gates.

### Core benchmark

<code>benchmark-cores</code> compares configured core variants and reports the
evidence needed to select <code>cluster.cores_per_case</code>; it does not edit
configuration. Retry one failed variant with
<code>benchmark-cores --variant &lt;variant-id&gt;</code>.

### Pilot

<code>pilot-check</code> exercises natural-support cases across configured
materials and retains a diagnostic receipt. Accepted work follows the configured
gated cleanup policy; incomplete or invalid evidence is not cleanup-eligible.

### Production and license retry

Production uses one ordinary non-exclusive Slurm job per case. Configuration
owns resources, queue feeding, timeout, failure budget, and retention. Temporary
license-capacity exhaustion uses bounded retry/backoff for the same deterministic
case without holding a compute allocation during backoff.

The foreground <code>all</code> or <code>resume</code> process feeds and
reconciles the campaign. Already submitted Slurm jobs continue if that
foreground process disconnects.

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
- retained failed-case log:
  <code>01_generation/meta/&lt;batch_storage_name&gt;/failures/&lt;case_id&gt;/artifacts/runtime/solver.log</code>;
- failure inventory and bounded tail: the failed case's
  <code>failure.json</code>.

Use <code>campaign-status --format json</code> for the complete machine-readable
case inventory.

## Evidence, readiness, and retention

<code>readiness-report</code> is the launch-state authority. Production is ready
only when resolved campaigns, static checks, and current profile-specific
Technical-Smoke evidence pass.

Technical Smoke retains diagnostic evidence. Accepted Pilot and Production
work follows the configured gated cleanup policy; incomplete or invalid evidence
remains ineligible for cleanup. <code>01_generation</code> is the canonical
simulation archive, while <code>02_datasets</code> contains immutable learning
packages. Dataset publication never removes canonical Generation source.

## Resume, cancel, collect, and cleanup

Resume reuses valid successful cases and never duplicates active or pending
attempts. Accounting uncertainty fails closed.

~~~bash
./scripts/generation_workflow.sh status "$GENERATION_RUN_ID"
./scripts/generation_workflow.sh resume "$GENERATION_RUN_ID"
~~~

Cancel every persisted active attempt explicitly:

~~~bash
./scripts/generation_workflow.sh cancel "$GENERATION_RUN_ID"
~~~

Transfer and publish terminal Generation evidence:

~~~bash
./scripts/generation_workflow.sh collect "$GENERATION_RUN_ID"
./scripts/generation_workflow.sh build-datasets "$GENERATION_RUN_ID"
~~~

Preview cleanup, review the deletion plan, then confirm:

~~~bash
./scripts/generation_workflow.sh cleanup "$GENERATION_RUN_ID"
./scripts/generation_workflow.sh cleanup "$GENERATION_RUN_ID" --confirm
~~~

Cleanup is destructive only on the authorized CPU source. Canonical GPU
Generation and Dataset publications remain intact.

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
| <code>resume RUN_ID</code> | Reconcile and continue incomplete work |
| <code>validate RUN_ID</code> | Revalidate terminal CPU evidence |
| <code>collect RUN_ID</code> | Transfer and publish terminal Generation evidence |
| <code>build-datasets RUN_ID</code> | Build/reuse requested immutable packages |
| <code>cancel RUN_ID</code> | Cancel persisted active Slurm attempts |
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
- A benchmark refusal usually indicates missing current smoke evidence, a failed
  repetition, or conflicting successful evidence. Use the printed recovery
  command.
- Temporary license-capacity events retry within the configured bound. Terminal
  license or configuration failures require correction, not evidence deletion.
- If a configuration or source change is absent from a run, verify that it was
  committed before the workflow invocation.

The project entry point is the [README](../README.md).
