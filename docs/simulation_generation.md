# Generation Operations and Campaign Workflow

## Quick start

Run public workflow commands from the `hpc115` GPU/development host, outside
Docker, in the repository checkout. The wrapper uses the local native Python
environment for orchestration and Dataset packaging, then owns non-interactive
SSH and rsync to the configured native CPU/COMSOL host `sricehpc01`. COMSOL 6.4
and Slurm run only on `sricehpc01`; users do not manually SSH for normal
workflow stages. Validated Generation publications return to the canonical
`STORAGE_ROOT` on `hpc115`. Docker remains the development and learning
environment, not a CPU COMSOL requirement.

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
| Dataset package requests | Campaign `dataset_packages` | Source role and evaluation regime | Package names, materials, views, counts |
| Slurm resources | `configs/generation/execution/cluster_cpu.yaml` (`cluster`) | Nodes, packing, cores, parallelism, wall time | Validated `sbatch` arguments and worker pool |
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
| `hpc115` | Repository/config review, wrapper invocation, canonical `01_generation`, Dataset packages in `02_datasets`, Docker-backed development and training | Type every `generation_workflow.sh` command here; provide native Generation Python dependencies |
| `sricehpc01` | Native Python 3.10, `Comsol/v6.4`, COMSOL, Slurm, isolated workspaces, CPU source | Wrapper-owned batch SSH and rsync; manual SSH only for requested inspection of retained evidence |

The default remote layout is `$HOME/grainlegumes-generation/{repo,storage,venv}`
on `sricehpc01`. Override it only with `--remote-root`. The local storage root
defaults to the `storage` sibling of the checkout and may be overridden with
`STORAGE_ROOT`.

Set the shared local values first:

```bash
# from the repository checkout on hpc115
export STORAGE_ROOT="$(realpath ../storage)"
STEADY_CAMPAIGN=configs/generation/campaigns/steady_flow/family_generalization.yaml
TRANSIENT_CAMPAIGN=configs/generation/campaigns/transient_drying/family_generalization.yaml
PILOT_CAMPAIGN=configs/generation/campaigns/transient_drying/pilot_check.yaml
CPU_HOST="$(
  python -m src.generation.cli.cli_generation validate-config \
    "$STEADY_CAMPAIGN" --allow-incomplete | \
    python -c 'import json,sys; print(json.load(sys.stdin)["execution_resources"]["site"]["cpu_host"])'
)"
```

1. Preview CPU setup, then perform it explicitly:

```bash
./scripts/generation_workflow.sh setup-cpu --cpu-host "$CPU_HOST"
./scripts/generation_workflow.sh setup-cpu --cpu-host "$CPU_HOST" --execute
```

The first command is read-only. The second creates the remote checkout,
storage root, and Python environment for the exact local commit.

2. Validate both primary configs without promoting unresolved values:

```bash
python -m src.generation.cli.cli_generation validate-config \
  "$STEADY_CAMPAIGN" --allow-incomplete
python -m src.generation.cli.cli_generation validate-config \
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

Preflight is remotely executed on `sricehpc01` and audits the configured modules,
executables, Slurm capacity, venv, paths, and template/config binding without a
COMSOL solve or job submission.

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
python -m src.generation.cli.cli_generation validate-real-smoke \
  "$SMOKE_RECEIPT" --storage-root "$STORAGE_ROOT"
```

The all-in-one workflow already validates every HDF5 file, inspects every
dataset package, and runs DataLoader smokes with `num_workers=0` and
`num_workers=2`.

6. Preview CPU-source cleanup for a reviewed run:

```bash
GENERATION_RUN_ID='<campaign_run_id>'
./scripts/generation_workflow.sh cleanup "$GENERATION_RUN_ID"
```

7. Execute only the authorized cleanup:

```bash
./scripts/generation_workflow.sh cleanup "$GENERATION_RUN_ID" --confirm
```

8. After readiness is complete, run one complete resolved production campaign:

```bash
./scripts/generation_workflow.sh all "$STEADY_CAMPAIGN"

./scripts/generation_workflow.sh all "$TRANSIENT_CAMPAIGN"
```

Execution config owns the default resources. Optional resource flags are explicit
one-run overrides; review any override in the resolved plan. The generic
`--only-batch <resolved-batch-name>` selector can run a predeclared subset.
`all` otherwise uses exactly the batches displayed by
`validate-config`. When the resolved campaign declares an extreme-family group,
`--skip-extreme-family-ood` can omit that group for one execution without
changing the campaign configuration; it cannot be combined with
`--only-batch`.

9. Inspect local and remote status:

```bash
./scripts/generation_workflow.sh status "$GENERATION_RUN_ID"
```

10. Resume an interrupted all-in-one workflow:

```bash
./scripts/generation_workflow.sh resume "$GENERATION_RUN_ID"
```

11. Cancel every persisted active Slurm attempt for a campaign:

```bash
./scripts/generation_workflow.sh cancel "$GENERATION_RUN_ID"
```

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
| `configs/generation/execution/<site>.yaml` | Site, modules, executables, runtime limits, scheduler resources, purpose-specific retention | Scientific ranges, material roles, learning parameters |
| `configs/learning/<task>/<kind>/<config>.yaml` | Dataset IDs, model, optimization, training, evaluation, artifacts | Generation paths, material ranges, campaign membership |

Inspect the effective campaign rather than maintaining a parallel summary:

```bash
python -m src.generation.cli.cli_generation validate-config \
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
| 6 | `pilot-check .../pilot_check.yaml` | Six-material transient diagnostics and storage measurement | Accepted pilot receipt and reviewed diagnostics |
| 7 | `all <production campaign>` | Preflight, plan, Slurm run, monitor, terminal validation, collection, package build, loader smokes and gated cleanup | Terminal all-workflow, transfer, Dataset, and cleanup receipts |
| 8 | `validate`, `status`, or direct receipt validators | Revalidates publication and lifecycle evidence | Exact successful receipt/digest state |
| 9 | `cleanup RUN_ID` then `--confirm` when needed | Preview then execute digest-authorized CPU cleanup | No active job; all transfer, Dataset, workflow and inventory digests agree |

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

Resource overrides are `--wall-time`, `--only-batch`, and the validated node,
case, core, and parallelism flags shown by `--help`. `--skip-extreme-family-ood`
is a one-run omission and cannot be combined with `--only-batch`. `--detach`
applies to `all`; `--keep-cpu-source` is the sole source-retention override.

## CPU/GPU path and lifecycle

```text
configured native CPU / COMSOL host
  exact commit + configured COMSOL module + scheduler
  -> isolated case workspaces
  -> validated terminal campaign
  -> retained source evidence
                 |
                 | validated rsync staging and atomic publication
                 v
hpc115 canonical storage
  01_generation
  -> immutable packages in 02_datasets
  -> learning runs and artifacts in 03_experiments
```

`01_generation` is the canonical simulation archive. `02_datasets` contains
immutable package views addressed by `dataset_id`; it does not duplicate
campaign configuration. `03_experiments` contains training, tuning, and
evaluation artifacts.

The `all` command performs generation, terminal validation, transfer,
publication, package building, package inspection, both DataLoader smokes, and
authorized CPU cleanup. On failure it records the exact stage and a copyable
resume command. Transfer staging is marked and removable; an incomplete
transfer never becomes a published campaign.

Cleanup is dry-run by default. Confirmation is accepted only after source
inventory, transfer, dataset, and workflow digests agree and no active job owns
the source. The technical smoke overrides cleanup to retain source evidence for
manual review.

On `sricehpc01`, the wrapper owns campaign source below the remote
`$HOME/grainlegumes-generation/storage/01_generation` root and creates isolated
worker directories with guarded markers. On `hpc115`, collection creates marked
`01_generation/.state/transfer-staging` content, validates every declared byte,
and atomically publishes the immutable campaign under canonical `01_generation`
paths. Existing successful cases and valid publications are reused rather than
overwritten. `collect` never deletes CPU source. Dataset packages are built on
`hpc115` under `02_datasets`; transient indexes continue to reference canonical
`01_generation` case files. `resume` reconstructs persisted stage evidence and
submits only incomplete validated membership.

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

## Configured transient pilot check

Inspect the pilot owner before launch:

```bash
python -m src.generation.cli.cli_generation validate-config \
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

## Production generation

Production prerequisites are both accepted gates: the source-current paired
native COMSOL technical smoke and the transient pilot receipt. Do not infer
acceptance from static tests or checksums, and do not claim that production has
already run.

```bash
./scripts/generation_workflow.sh all "$STEADY_CAMPAIGN"
./scripts/generation_workflow.sh all "$TRANSIENT_CAMPAIGN"
```

The wrapper uses campaign-owned counts, roles, seeds, package requests and
execution defaults. It fails closed on a dirty checkout, commit/config/template
mismatch, unresolved mapping, native or HDF5 admission failure, incomplete
Slurm membership, transfer mismatch, package failure, or loader-smoke failure.
The resulting canonical evidence is an immutable `01_generation` publication,
transfer receipt, requested `02_datasets` packages, loader-smoke receipt, and
all-workflow receipt. Use `--keep-cpu-source` only when native source must remain
for review; otherwise CPU cleanup occurs only after every local gate passes.

## Readiness gates

Run:

```bash
python -m src.generation.cli.cli_generation readiness-report \
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
- Never delete CPU source manually. Use cleanup dry-run and confirmation so the
  digest authorization is preserved.

Scientific parameter ownership and formulas are documented in the
[parameter reference](generation_parameter_reference.md). The project gateway
is the [README](../README.md).
