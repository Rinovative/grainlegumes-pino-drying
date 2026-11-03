# Datasets

A Dataset is the fixed set of physical cases a model is trained and evaluated on. For both tasks, one Dataset plan requests the cases, runs Generation, selects successful cases, fixes the Train/validation/ID-test split and publishes the tensor payload. Training configurations only reference the published Dataset; run seeds never repartition it.

[Documentation index](README.md)

## Normal workflow

The plans are `configs/02_datasets/transient_drying/lentil_chickpea_ref_range.yaml` and `configs/02_datasets/steady_flow/lentil_chickpea_ref_range.yaml`. The examples use the transient plan; the steady plan runs the same way.

1. Preview the effective plan. The preview writes nothing and starts no solver work; it prints the computed Dataset name (`dataset`) and its `directory` below `storage/`:

```bash
../runtime/venvs/native/bin/python -m src.datasets.dataset_packages run \
  configs/02_datasets/transient_drying/lentil_chickpea_ref_range.yaml --preview
```

2. Run the same command without `--preview` on the login node. It freezes the source (see [Generation reference](06_generation_reference.md)), submits its own driver job and prints the job as JSON (`status`, `job_id`, `tree_id`, `log_path`):

```bash
../runtime/venvs/native/bin/python -m src.datasets.dataset_packages run \
  configs/02_datasets/transient_drying/lentil_chickpea_ref_range.yaml
```

Do not start the command inside an allocation; freezing is refused there. The freeze always includes uncommitted changes of the checkout: they are bound as an exact snapshot, so no option is needed to admit them. The driver job runs from the frozen source tree in a fixed modest CPU allocation (2 CPUs, `8G`; `--driver-partition standard` or `long`, `--driver-time 06:00:00`) and writes `runtime/logs/datasets/dataset-TASK-JOB_ID.log`. It submits each COMSOL case as its own Slurm job and assembles the Dataset afterwards; its two CPUs stay free for coordination. Its wall limit is independent of the solver jobs and of the physical horizon.

3. Rerun the identical command until it publishes; each run submits a new driver that continues the campaign. The driver log shows numbered stage lines (`[n/N] <stage>`, then `... done in <duration>`) and, while cases run, a live board every 5 minutes or on a state change: one line per case with its state, simulated time against the horizon, solver step and failure counts, time since the last update and an estimate, and for a failed case its reason and replacement. It ends with a summary: `DATASET COMPLETE` with the published folder, contents, split and failures, or `STAGED` when cases are still pending, or `FAILED: <stage>: <message>` with the command to run `Next:`. The Dataset and its tensor payload are published once enough cases have succeeded.

4. Inspect the publication:

```bash
../runtime/venvs/native/bin/python -m src.datasets.dataset_packages inspect transient_drying DATASET_ID
```

`inspect` takes the task directory and the Dataset name. It lists the ordered Sources, effective settings, the original plan command with its overrides and the commands for an exact rebuild. Use the plain Dataset name in the training configuration (`data.train_dataset`) or with `--train-dataset`; the task comes from the configuration.

## The maintained transient plan

| Setting | Value |
| --- | --- |
| Task and profile | `transient_drying`, `domain_profile: reference_range` |
| Materials and counts | lentil 50, chickpea 50, both role `id` |
| Generation | Latin hypercube, seed base `9200` |
| Case selection | `seed_ranked`, selection seed `9201` |
| Replacements | `replacement_capacity: 5` per material (10 % of 50) |
| Split | seed `9` (effective `1028622135437308793`), fractions 0.8 / 0.1 / 0.1 per material |
| Resulting membership | 40 Train, 5 validation, 5 ID-test per material |
| Dataset name | `lentil-train40-val5-test5+chickpea-train40-val5-test5__ref-range__lhs-seed-9200` (computed; see *Names and identity* below) |

Each count is both the original design size and the target number of successful cases (see *Failed cases and replacements*). The split applies per ID material before the cases are expanded into temporal transitions. Fractions use floors for Train and validation; the remainder goes to ID-test. A material with role `ood` is held out completely as material-family OOD.

### Failed cases and replacements

Both maintained plans pre-register `replacement_capacity: 5` per material (10 % of 50). With `replacement_capacity: 0` a terminally failed original candidate stays a failure: the Source cannot reach its success target, and only an explicit capacity amendment or a new Source can fill the gap.

`generation.replacement_capacity` pre-registers how many replacement candidates each material may use: one integer for every material, or a mapping `lentil: 3` (unnamed materials get 0). It is part of the authored plan, not a run-time override:

- It changes neither the computed Dataset name nor the Generation seeds, and the Source keeps its name. The first run records it as the Source's capacity amendment; rerunning with another value is refused, so capacity added later is an explicit `amend-replacements` ([Generation reference](06_generation_reference.md)) that the Dataset records as not pre-registered.
- The admission limit defaults to the sum of the counts plus the total capacity.
- A replacement is the next prefix-stable Sobol point of a separate seed stream, not a Latin hypercube point. It takes the target slot of the failed candidate and records `parent_failed_case_id`.
- A solve the runtime stopped itself (`stalled`, `timed_out`, `solver_error`) is decided automatically with that typed reason once its step has ended; its replacement is planned, while capacity remains, as soon as every original candidate of the same material is decided, so replacements run after the material's originals. Any other failure, a COMSOL exit by a signal such as a memory-limit kill included, is declared by hand: inspect it, record it with `fail-case` (original or replacement candidate; [Generation reference](06_generation_reference.md#failures-and-continuation)) and rerun the same plan command.
- The driver stops on a failed case only when no decision is recorded for it for two minutes of reconciliation (it names the case and the `fail-case` route), or when a recorded failure leaves a slot that the pre-registered capacity cannot fill (it names `amend-replacements` and a new Source as the ways out). A recorded failure whose slot a replacement covers never stops the run.

What the Dataset then represents: the cases whose solver run succeeded, selected from the sampled design plus its replacements. It describes that solver-success population, not the full original design domain. Replacing failures does not undo the selection by success, so the failures are reported rather than hidden. The manifest binds this evidence (`source_outcomes`): per contributing Source its replacement policy (pre-registered and total capacity, every amendment with registrant and reason), the counts of original-candidate failures per material, and a table of every candidate (origin, target slot, parent failed case, outcome, failure stage, class and reason code). `inspect` prints the original-candidate failure rate per material. Rates and counts belong in any report of results obtained on the Dataset. To see where in the design the failures lie, each completed Source writes `cases.csv` (one row per candidate with its outcome, typed reason, runtime and every sampled parameter), and the section «Failures by parameter» of [Generation-input EDA](../notebooks/01_eda_generation_inputs.ipynb) plots every terminal candidate at its sampled values per material, coloured by reason; both show recorded association, not cause.

The approved steady plan uses the steady-flow profile with lentil 50 and chickpea 50 (both role `id`, 40 / 5 / 5), replacement capacity 5 per material, Generation seed base `9300` and selection seed `9301`. Neither maintained plan has an OOD material; a material-family OOD evaluation needs its own plan.

## Small runs

Explicit overrides change one run without editing the plan. Three cases per material with equal thirds and a four-hour horizon give one Train, one validation and one ID-test case per material:

```bash
../runtime/venvs/native/bin/python -m src.datasets.dataset_packages run \
  configs/02_datasets/transient_drying/lentil_chickpea_ref_range.yaml \
  --count lentil=3 --count chickpea=3 \
  --split-fractions 1/3,1/3,1/3 --duration-hours 4 \
  --node-jobs 1 --max-admission-cases 6 --cores-per-case 8 --preview
```

| Option | Effect |
| --- | --- |
| `--count MATERIAL=N` | Original design size and success target for one material; changes the design |
| `--duration-hours H` | Physical horizon; changes the physical output domain |
| `--split-fractions T,V,I` | Recorded split fractions (`1/3` accepted) |
| `--revision LABEL` | Appends `__rev-LABEL` to the computed Dataset name, for a deliberately separate Dataset of the same plan; the default Source is reused and nothing is simulated again. The label uses lowercase letters and digits joined by single `-`, `_` or `+` |
| `--new-source` | Only together with `--revision LABEL` (else exit 2): the Source is `<default Source name>__rev-LABEL` and its cases are simulated afresh with licensed COMSOL. Use it when the producer science changed; the preview then says that the Source is simulated afresh |
| `--max-admission-cases N` | Distinct fresh solver candidates; defaults to the sum of the counts plus the planned replacement capacity |
| `--node-jobs N`, `--cores-per-case N` | Concurrent node jobs (1 to 4, default 2; a per-user cluster-share cap; the COMSOL batch licence counts once per user) and CPUs per case; together they cap the concurrent cases ([Generation reference](06_generation_reference.md)) |
| `--case-timeout-seconds S`, `--no-retry` | Solver runtime limit and licence-retry policy |
| `--driver-partition`, `--driver-time` | Driver job partition and wall limit (`standard` or `long`, default `standard`; default `06:00:00`; the 2 CPUs and `8G` are fixed); not part of the Dataset identity |
| `--continue-with-current-code` | Continue an existing campaign with the current code instead of the source tree it is bound to (recorded as a new execution epoch) |

Seeds, grid, solver settings and sampling method stay as authored. Repeat the same complete command, overrides included, to continue a run. The small run above is named `lentil-train1-val1-test1+chickpea-train1-val1-test1__ref-range__lhs-seed-9200__h-4`, so it never collides with the full Dataset.

## What a Dataset contains

```text
storage/02_datasets/<task>/<dataset>/
  dataset_manifest.json       # ordered Sources, representation, roles, split membership
  dataset.json                # transient: temporal transition index
  transient_pt_shards/        # transient: training payload
    receipt.json              # admission and payload identity
    shard_*.pt
  dataset.pt                  # steady flow: tensor payload
storage/02_datasets/.state/   # build locks and the identity registry (identities/)
```

The payload file names are fixed; the Dataset name only names the directory.

The manifest owns the selection, exact Source evidence, the Source outcome evidence (replacement policy and all candidate outcomes), material roles and the split. For transient drying, the transition index enumerates temporal samples of those cases and the PT shards are the training representation with their own receipt; each stage is validated and published separately. The receipt records each file's size, hash and file metadata: unchanged files are admitted directly, copied, restored or moved files are re-verified against the recorded hash, and only changed bytes are rejected. A steady-flow Dataset stores all its cases in one tensor payload.

<details>
<summary><strong>Names and identity</strong></summary>

The name is computed from the resolved plan, never authored: `<material>-train<a>-val<b>-test<c>` for each ID material (the Train, validation and ID-test case counts of the split) or `<material>-ood<n>` for a held-out material, joined by `+` in plan order, then `__<domain>__lhs-seed-<seed>` (`ref-range` for `reference_range`, `ref-param-ext` for `reference_and_parameter_extension`), for transient drying a `--duration-hours` horizon that differs from the one configured in the common config as `__h-<hours>` (`4.5` as `4p5`; the configured horizon belongs to the simulation definition and adds no token), and optionally `__rev-<label>`. Overriding counts, split fractions, the horizon or the plan therefore changes the name.

The physical Source name is computed from the plan the same way Generation names an unrevised campaign: `<material>-<count>` per material joined by `+`, then `__<domain>__lhs-seed-<seed>` and for transient drying `__h-<hours>` only when the horizon differs from the configured one (for example `lentil-50+chickpea-50__ref-range__lhs-seed-9200` at the configured horizon and `...__h-4` for `--duration-hours 4`), never authored. A deviating horizon is therefore another Source with the same seeds and needs no revision. `--revision LABEL` names only the Dataset and reuses this Source. With `--new-source` as well the Source becomes that name plus `__rev-LABEL` and owns its own solver runs instead of continuing the default Source. The label changes only the Source name and storage locator: Generation seeds, batch identities and the scientific campaign digest stay those of the plan without it.

The one revision rule, the same for Dataset, Source, run and study: *Same name, changed science: add `--revision LABEL` to the command you run. It never changes seeds, never makes identical science new, and never re-simulates a Source unless you also pass `--new-source` to the Dataset run.*

- The same name with identical content reuses the publication.
- Different content under an occupied name is refused; pass `--revision LABEL` to keep a deliberately separate Dataset (for example after a converter fix). Changed producer science additionally needs `--new-source`, because only that simulates a Source again.
- Identical content under another name is refused (`Identical Dataset content already exists as <task>/<name>; use that name`), so a Dataset exists once. A `--revision` label alone does not make identical content new.

The internal scientific digest binds the ordered Source cases, grid, channel and conversion contracts, roles and split membership, and the relevant converter implementation. Names, file paths and Git HEAD are provenance, not identity. A registry below `02_datasets/.state/identities/` maps each digest to its Dataset directory; an entry whose directory no longer exists is replaced. Physical Sources carry no Dataset roles, and the Source name never contains the split; a revision label appears in the Source name only with `--new-source`. Details are in [Reproducibility and validation](07_reproducibility_and_validation.md).

</details>

<details>
<summary><strong>Case selection: seed_ranked</strong></summary>

`seed_ranked` chooses from cases that already succeeded. For each material it requires enough successes, orders them by a SHA-256 rank of selection seed, task, material and `case_input_id`, and takes the requested number. The selection seed is independent of the Generation seed. Duplicate physical inputs across Sources fail. `exact` selection instead lists the source cases explicitly. The split follows selection with its own seed.

</details>

<details>
<summary><strong>Evaluation Datasets</strong></summary>

A Dataset whose plan gives a material the role `ood` holds that material out as family OOD. Training configurations list such Datasets in `data.additional_evaluation_datasets`; their declared OOD cases, plus any OOD cases of the training Dataset itself, form the additional evaluation membership. A declared holdout that reuses a training material is rejected.

</details>
