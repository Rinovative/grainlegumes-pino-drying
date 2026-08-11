# VP2 Simulation Generation and Campaign Workflow

## Quick start

Run local commands from `/workspace/repo` in the development container. The
workflow connects to the native CPU/COMSOL host declared by the resolved
execution configuration; it publishes validated results into the local
`STORAGE_ROOT`.

> Configured scientific values are modelling and sampling decisions. A citation
> does not imply that every final number appears verbatim in its source. The
> authoritative interpretation is the resolved `status`, `derivation`,
> `confidence`, and `validity`; technical runtime evidence does not constitute
> experimental validation.

Set the shared local values first:

```bash
cd /workspace/repo
export STORAGE_ROOT=/workspace/storage
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

7. Execute only the previously authorized cleanup:

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
| `configs/generation/sources.yaml` | Supplied bibliographic records keyed once | Parameter values, inferred assignments, roles, execution |
| `configs/generation/registry.yaml` | Canonical decision identity, parameter names, units, kinds, transforms, sampling order, OOD groups, components, derivations | Material supports, campaign counts, mappings, cluster resources |
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

## Python responsibility boundaries

The Generation source layout follows the lifecycle: `generation_contracts_*`
own scientific contracts; `generation_cases_*` own config resolution and case
inputs; `generation_runtime_*` own native execution;
`generation_publication_*` own HDF5 and campaign evidence; and
`generation_validation_*` own sentinels and pilot checks. These visible prefixes
keep ownership clear when a module is imported away from its directory.

`src.generation` is the stable facade, and
`python -m src.generation.cli.cli_generation` remains the supported command.
Dependencies flow from contracts to cases to runtime, then into publication,
validation, and top-level orchestration; lower packages never import workflow or
CLI layers.

After terminal publication, the workflow invokes `src.datasets.packages`. Dataset
code mirrors the split through `dataset_contracts_*`, `dataset_packages_*`,
`dataset_preprocessing_*`, and `dataset_runtime_*`. The canonical package facade
and CLI remain `src.datasets.dataset_packages`, preserving the persisted
`src.datasets.dataset_packages.build_campaign_packages` identity while delegating
to one implementation. Training, EDA, and Evaluation import the public
`datasets` alias with `from src import datasets`, not package directories or HDF5
internals. The exact consumer aliases and Dataset responsibility table are in the
[README](../README.md#-python-ownership-and-public-apis).

## Transient Dataset time and sampling views

The compact transient index stores only authoritative HDF5 and schedule indices.
The shared `datasets.runtime.transient.TransientPhysicalDataset` reads physical
`t_n`, `t_n_plus_1`, and `dt` directly from the regular HDF5 time axis as
float32 tensors in hours. The configured normalization horizon comes from the
embedded resolved `scientific_config.time.stop`; it is never inferred from an
early case stop, exact-stop diagnostic, trajectory length, or rollout length.

`datasets.contracts.transient.TransientSamplingSpec` requires an explicit mode:

- `one_step_transition` returns ground-truth `state`, shared `static` and eight
  scientific `scalars`, endpoint `boundary`, scalar time tensors, and the target
  increment `q_(n+1) - q_n`;
- `rollout_window` returns one initial ground-truth state plus consecutive
  boundary, time, and target-increment sequences for an explicit length, stride,
  and offset.

Both modes share one package interpreter, HDF5 reader, bounded process-local
handle cache, source mutation checks, channel contract, and case membership.
They never cross a case boundary or include an irregular exact-stop state. The
hourly boundary channels remain adjacent `T_in_bc` and `phi_in_bc` endpoints: a
linear interval is uniquely determined by those endpoints, so no mean, integral,
or sub-hour feature is added.

Learning exposes an explicit `normalized_current_time` or `none` policy through
`learning.temporal`; `experiments.config.temporal` validates it together with
the sampling mode without a hidden default. These APIs prepare the stable
Dataset/config boundary only. No transient TaskSpec, full transient trainer,
autoregressive optimization loop, EDA extension, or Evaluation extension is
claimed here.

## Resolved campaign and package semantics

Campaign YAML owns role assignment, sampling counts and seed namespaces,
learning membership, and package requests. Resolution derives material and
evaluation inventories, batch names and identities, source-case totals, package
materials, split eligibility, and profile-expanded package names. A pilot
campaign resolves no normal dataset packages; a technical runtime smoke keeps
its operational membership distinct from learning membership.

Parameter-OOD planning derives profile-applicable units from the projected
registry and the material's actual tails or alternate atomic records. Each case
activates one unit. The deterministic allocation covers every eligible unit
when capacity permits and distributes remaining cases evenly. The exact unit
inventory, group, per-unit counts, and per-case allocation appear under
`parameter_ood` and persist in resolved scientific provenance. A profile cannot
select a unit that its registry projection excludes.

Static sentinels are deliberately independent of production counts. The
`static_sentinel_workload` view reports their bounded natural-material and
eligible-OOD coverage before the sentinels run. This keeps scientific checks
complete when campaign counts or material inventories change without turning
production configuration into a golden fixture.

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
GPU/container storage
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
- case-local file reload and scalar handoff;
- validated raw exports and canonical HDF5;
- paired shared inputs across the two profiles;
- observed `p`, `u`, and `v` difference metrics without an invented tolerance;
- observed differential/integral water-balance metrics;
- immutable technical packages and both DataLoader worker modes;
- retained CPU inputs, exports, solved evidence, logs, Slurm IDs, and version;
- one source-bound real-smoke receipt.

Mapping probes inventory actual output files and headers. They never infer or
write a mapping automatically. Fixed values reported as template-owned have no
Python runtime setter; their configured record is bound to the canonical
hashed template and model-report evidence, while case-adapter reload still
requires native runtime evidence.

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
tolerance, run-wide extrema, heterogeneity, schedules, supplied validity
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
hash, a historical receipt, or documentation text. Launch is ready only when
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
