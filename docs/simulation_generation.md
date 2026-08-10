# VP2 Simulation Generation and Campaign Workflow

## Quick start

Run local commands from `/workspace/repo` in the development container. The
workflow connects to the native CPU/COMSOL host `sricehpc01`; it publishes
validated results into the local `STORAGE_ROOT`.

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
```

1. Preview CPU setup, then perform it explicitly:

```bash
./scripts/generation_workflow.sh setup-cpu --cpu-host sricehpc01
./scripts/generation_workflow.sh setup-cpu --cpu-host sricehpc01 --execute
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

The resolved output includes campaign purpose, roles, regimes, counts, seeds,
batches, packages, template, execution resources, and exact readiness gates.

3. Preview the resolved Slurm plan after every primary gate is filled:

```bash
./scripts/generation_workflow.sh plan "$STEADY_CAMPAIGN" \
  --max-nodes 1 --cases-per-node 2 --cores-per-case 16 \
  --max-parallel-cases 2
```

`plan` is read-only but intentionally fails while production values or mappings
remain unresolved.

4. Run the canonical paired technical runtime smoke:

```bash
./scripts/generation_workflow.sh smoke \
  --max-nodes 1 --cases-per-node 2 --cores-per-case 16 \
  --max-parallel-cases 2 --keep-cpu-source
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

8. After readiness is complete, run one whole six-family production campaign:

```bash
./scripts/generation_workflow.sh all "$STEADY_CAMPAIGN" \
  --max-nodes 1 --cases-per-node 2 --cores-per-case 16 \
  --max-parallel-cases 2

./scripts/generation_workflow.sh all "$TRANSIENT_CAMPAIGN" \
  --max-nodes 1 --cases-per-node 2 --cores-per-case 16 \
  --max-parallel-cases 2
```

Resource values shown here are explicit examples, not hidden production
defaults. Review them and the resolved plan for the intended run. The generic
`--only-batch <profile>__<family>__<natural|parameter_ood>` selector can run a
predeclared subset; there is no material-specific Sunflower path. The standard
`all` command includes the extreme-family group. An explicit
`--skip-extreme-family-ood` flag may omit only that group for one execution;
it does not modify the canonical six-family campaign or material inventory and
cannot be combined with `--only-batch`.

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

## Configuration map

```text
configs/
|-- generation/
|   |-- common.yaml
|   |-- sources.yaml
|   |-- registry.yaml
|   |-- operations/fixed_bed.yaml
|   |-- materials/
|   |   |-- lentil.yaml
|   |   |-- chickpea.yaml
|   |   |-- kidney_bean.yaml
|   |   |-- field_pea.yaml
|   |   |-- rapeseed.yaml
|   |   `-- sunflower_seed.yaml
|   |-- profiles/{steady_flow,transient_drying}.yaml
|   |-- campaigns/steady_flow/
|   |   |-- family_generalization.yaml
|   |   `-- technical_smoke.yaml
|   |-- campaigns/transient_drying/
|   |   |-- family_generalization.yaml
|   |   |-- technical_smoke.yaml
|   |   `-- pilot_check.yaml
|   `-- execution/cluster_cpu.yaml
`-- learning/steady_flow/
```

| Config | Controls | User normally edits | Must not contain |
| --- | --- | --- | --- |
| `generation/sources.yaml` | Central supplied bibliographic records keyed once | Only when supplied source records change | Parameter values, inferred source assignments, roles, execution |
| `generation/registry.yaml` | Canonical names, units, kinds, transforms, blocks, OOD groups, components, derivations | Only when the parameter contract changes | Material ranges, campaign counts, mappings, cluster resources |
| `generation/common.yaml` | Grid, time, shared fixed physics, formulas, adapter/storage contracts | Shared scientific design changes | Material values, roles, counts, learning choices |
| `generation/operations/fixed_bed.yaml` | Pressure-field and inlet/ambient schedule ranges, operation OOD supports and constraints | Reviewed apparatus/operation evidence | Material values, template mappings, Slurm resources |
| `generation/materials/<family>.yaml` | Role-neutral material scope, natural values/supports, coupled records, targets, evidence | Reviewed family evidence | Campaign role, membership, count, profile, execution |
| `generation/profiles/<profile>.yaml` | Template identity, adapters, exports, explicit COMSOL mappings, profile conditioning | Reviewed COMSOL mapping/audit results | Material values, counts, roles, cluster plans |
| `generation/campaigns/<profile>/<campaign>.yaml` | Purpose, profile references, material roles, sampling method/counts/seeds, package declarations | Campaign counts, seeds, memberships | Parameter ranges, package material lists, execution defaults |
| `generation/execution/cluster_cpu.yaml` | Site, module/executable names, runtime limits, Slurm allocation values, purpose-specific retention | Explicit execution resources | Scientific ranges, material roles, learning parameters |
| `learning/<task>/<kind>/<config>.yaml` | Dataset IDs, model, optimization, training, evaluation and artifacts | Learning experiments | Generation paths, material ranges, campaign membership |

Every semantic decision has one authored owner. Campaign material lists,
evaluation-regime lists, package materials, eligibility, names, IDs, module
commands, and purpose-specific retention are derived and persisted in resolved
provenance. Validation errors identify the exact file, key, rule, actual value,
and owner to edit.

## Campaign and package semantics

The primary `steady_flow` and `transient_drying` campaigns both declare:

| Material role | Families | Sampling |
| --- | --- | --- |
| `seen` | `lentil`, `chickpea`, `kidney_bean` | Natural plus parameter OOD |
| `near_family_ood` | `field_pea` | Natural only |
| `far_family_ood` | `rapeseed` | Natural only |
| `extreme_family_ood` | `sunflower_seed` | Natural only |

Their canonical evaluation regimes are exactly:

```text
id
parameter_ood
near_family_ood
far_family_ood
extreme_family_ood
```

`extreme_family_ood` is only an evaluation category. It creates no model,
training regime, operator channel, equation, sampling coordinate, or COMSOL
profile. Sunflower has no training, validation, ID-test, or parameter-OOD
membership.

A concise campaign declaration maps package regimes to source roles:

```yaml
campaign_purpose: family_generalization
material_roles:
  seen: [lentil, chickpea, kidney_bean]
  near_family_ood: [field_pea]
  far_family_ood: [rapeseed]
  extreme_family_ood: [sunflower_seed]
dataset_packages:
  - {evaluation_regime: id, source_role: seen}
  - {evaluation_regime: parameter_ood, source_role: seen}
  - {evaluation_regime: near_family_ood, source_role: near_family_ood}
  - {evaluation_regime: far_family_ood, source_role: far_family_ood}
  - {evaluation_regime: extreme_family_ood, source_role: extreme_family_ood}
```

The steady primary campaign resolves five immutable packages. The transient
primary campaign resolves ten: five `steady_flow` views and five
`transient_drying` views. Each near, far, and extreme package is separate. ID
membership is assigned once per physical case before transient temporal
expansion.

The technical campaigns use `campaign_purpose: technical_runtime_smoke`,
profile seeds `9910` (steady) and `9920` (transient), paired-equivalence seed
`9930`, two Lentil natural cases, and ID packages marked non-training.
Technical membership is operational metadata, not an evaluation regime, and
the dataset factory rejects it unless a caller explicitly enables technical
smoke inspection.

The final primary plans are:

| Profile | Seen natural | Parameter OOD | Near | Far | Extreme | Total | Campaign seed | Membership seed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `steady_flow` | 720 (240/family) | 240 (80/family) | 80 | 80 | 80 | 1,200 | 9100 | 9150 |
| `transient_drying` | 360 (120/family) | 180 (60/family) | 40 | 40 | 40 | 660 | 9200 | 9250 |

Steady Seen membership is 192 train, 24 validation, and 24 ID-test cases per
family (576/72/72 total). Transient Seen membership is 96/12/12 per family
(288/36/36 total). Together the campaigns contain 1,860 profile-specific
source cases.

Parameter-OOD planning derives profile-applicable units from the resolved
registry and actual tails or alternate atomic records. Each case activates one
unit; the deterministic round-robin covers every eligible unit where possible
and keeps allocation counts within one. Exact eligible units, case allocation,
and counts persist in resolved scientific provenance. No top-level group quota
is hard-coded, and steady planning cannot select transient-only units.

## CPU/GPU path and lifecycle

```text
native CPU / sricehpc01
  exact commit + COMSOL 6.4 + Slurm
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
exercise all six families, all four sampling blocks, all four parameter-OOD
groups, coupled records, fields, schedules, and deterministic replay.

The fake runtime uses a test-owned executable. It proves Python case isolation,
HDF5 conversion/publication, package construction, factory guards, and
DataLoader behavior; it is not COMSOL evidence.

The real technical smoke is the native COMSOL gate. It requires:

- two distinct steady cases and two distinct transient cases;
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

## Canonical transient pilot check

Run the normal all-in-one diagnostic pilot from the GPU/development host:

```bash
./scripts/generation_workflow.sh pilot-check \
  configs/generation/campaigns/transient_drying/pilot_check.yaml
```

The dedicated config is the single human-authored pilot owner. It uses
`campaign_purpose: pilot_check`, seed namespace `9940`, and plans 18 transient
cases: one `nominal_reference` followed by two deterministic
`natural_pilot` cases for each of the six families.
It uses no parameter OOD, family-OOD sampling, corner construction, training
membership, or automatic calibration. The same generic analysis checks runtime
and conversion contracts, nominal drying duration, natural-support robustness,
physical bounds, water balances without an invented tolerance, run-wide
extrema, heterogeneity, schedules, supplied validity metadata, and measured
storage for every family.

Fast nominal-only form:

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

The normal lifecycle performs host preflight, exact commit/config/template
binding, CPU readiness, a mapping probe when required, deterministic planning,
Slurm execution and monitoring, terminal collection, hash validation, HDF5
conversion, runtime/physical/mass-balance/extrema/trend analysis, pre-cleanup
CPU and staging measurement, permanent GPU measurement, and a 660-case
projection labelled `observed_real_pilot_based_estimate`. One canonical
`01_generation/meta/pilot_checks/<pilot_check_id>/pilot_check.json` owns the
results; `summary.csv` and `summary.md` are derived from it.

After retained evidence validates, the normal command performs authorised
CPU-source and transfer-staging cleanup and verifies deletion. It never deletes
active, incomplete, hash-invalid, or insufficiently retained evidence.
`--keep-cpu-source` is the explicit CPU-source opt-out; staging is still cleaned
after validation. There is no storage-budget pass/fail guard. Until the current
static scientific guard and native mappings pass, the command stops before
launching COMSOL.

## Readiness gates

Run:

```bash
python -m src.generation.cli.cli_generation readiness-report \
  "$STEADY_CAMPAIGN" "$TRANSIENT_CAMPAIGN" \
  --run-static-sentinels
```

The exact status vocabulary is:

```text
STATIC_SCIENTIFIC_INTEGRATION_COMPLETE
CONFIG_OWNERSHIP_CONSOLIDATION_COMPLETE
DOCUMENTATION_CONSOLIDATION_COMPLETE
STATIC_GENERATOR_SENTINELS_COMPLETE | STATIC_GENERATOR_SENTINELS_PENDING | STATIC_GENERATOR_SENTINELS_BLOCKED
REAL_RUNTIME_VALIDATION_COMPLETE | REAL_RUNTIME_VALIDATION_PENDING
PRIMARY_PRODUCTION_CONFIG_COMPLETE | PRIMARY_PRODUCTION_CONFIG_INCOMPLETE
PRODUCTION_READY_FOR_USER_LAUNCH | PRODUCTION_READY_FOR_USER_LAUNCH_BLOCKED
```

Launch is ready only when explicit production counts/seeds/memberships,
the static scientific guards, reviewed mappings, both native profile reloads,
scalar handoff, paired equivalence observations, HDF5/package/loader
validation, and a current real-smoke receipt all pass. At present, the
material packing-porosity sentinel failures and unconfirmed native mappings
block launch; no real-smoke or pilot receipt can be claimed. A template hash
proves byte identity only; it does not prove model-tree behavior.

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
