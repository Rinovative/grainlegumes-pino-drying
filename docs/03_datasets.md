# Datasets

A Dataset is the fixed set of physical cases a model is trained and evaluated on. For both tasks, one Dataset plan requests the cases, runs Generation, selects successful cases, fixes the Train/validation/ID-test split and publishes the tensor payload. Training configurations only reference the published Dataset; run seeds never repartition it.

[Documentation index](README.md)

## Normal workflow

The plans are `configs/02_datasets/transient_drying/lentil_chickpea__ood_kidney_bean.yaml` and `configs/02_datasets/steady_flow/lentil_chickpea__ood_kidney_bean.yaml`. The examples use the transient plan; the steady plan runs the same way.

1. Preview the effective plan. The preview writes nothing and starts no solver work; it prints the computed Dataset name (`dataset`) and its `directory` below `storage/`:

```bash
../runtime/venvs/native/bin/python -m src.datasets.dataset_packages run \
  configs/02_datasets/transient_drying/lentil_chickpea__ood_kidney_bean.yaml --preview
```

2. Run the same command without `--preview` on the login node. It freezes the source (see [Generation reference](06_generation_reference.md)), submits its own driver job and prints the job as JSON (`status`, `job_id`, `tree_id`, `log_path`):

```bash
../runtime/venvs/native/bin/python -m src.datasets.dataset_packages run \
  configs/02_datasets/transient_drying/lentil_chickpea__ood_kidney_bean.yaml
```

Do not start the command inside an allocation; freezing is refused there. The freeze always includes uncommitted changes of the checkout: they are bound as an exact snapshot, so no option is needed to admit them. The driver job runs from the frozen source tree in a fixed modest CPU allocation (2 CPUs, `8G`; `--driver-partition standard` or `long`, `--driver-time 06:00:00`) and writes `runtime/logs/datasets/dataset-TASK-JOB_ID.log`. It submits each COMSOL case as its own Slurm job and assembles the Dataset afterwards; its two CPUs stay free for coordination. Its wall limit is independent of the solver jobs and of the physical horizon.

3. Rerun the identical command until it publishes; each run submits a new driver that continues the campaign. The driver log shows numbered stage lines (`[n/N] <stage>`, then `... done in <duration>`) and, while cases run, a live board every 5 minutes or on a state change: one line per case with its state, simulated time against the horizon, solver step and failure counts, time since the last update and an estimate (only once a case ended, otherwise `ETA: not yet estimable`), and for a failed case its reason and replacement; every case that changes its group first gets one timestamped line `case <material>/<case>: <old> -> <new> (<reason>)`. It ends with a summary: `DATASET COMPLETE` with the published folder, contents, split and failures, or `STAGED` when cases are still pending, or `FAILED: <stage>: <message>` with the command to run `Next:`. The Dataset and its tensor payload are published once enough cases have succeeded.

4. Inspect the publication:

```bash
../runtime/venvs/native/bin/python -m src.datasets.dataset_packages inspect transient_drying DATASET_ID
```

`inspect` takes the task directory and the Dataset name. It lists the ordered Sources, effective settings, the original plan command with its overrides and the commands for an exact rebuild. Use the plain Dataset name in the training configuration (`data.train_dataset`) or with `--train-dataset`; the task comes from the configuration.

## The maintained plans

Both plans hold lentil and chickpea in distribution and kidney_bean out as material-family OOD.

| Setting | Transient | Steady |
| --- | --- | --- |
| Task and profile | `transient_drying`, `domain_profile: reference_range` | `steady_flow`, `domain_profile: reference_range` |
| Materials and counts | lentil 100, chickpea 100 (role `id`); kidney_bean 30 (role `ood`) | lentil 500, chickpea 500 (role `id`); kidney_bean 150 (role `ood`) |
| Generation | Latin hypercube, seed base `9200` | Latin hypercube, seed base `9300` |
| Case selection | `design_order` (Generation's order of originals, then replacements) | same |
| Minimum success rate | `0.6` (the 02./03.11. smokes solved 12 of 17 transient solves, every failure a stall) | `0.9` (the 03.11. smoke solved 8 of 8) |
| Split | seed `9` (effective `1028622135437308793`), fractions 0.7 / 0.15 / 0.15 per ID material | seed `9`, fractions 0.8 / 0.1 / 0.1 per ID material |
| Resulting membership | 70 Train, 15 validation, 15 ID-test per ID material; 30 OOD | 400 Train, 50 validation, 50 ID-test per ID material; 150 OOD |
| Dataset name | `lentil-train70-val15-test15+chickpea-train70-val15-test15+kidney-bean-ood30__ref-range__lhs-seed-9200` | `lentil-train400-val50-test50+chickpea-train400-val50-test50+kidney-bean-ood150__ref-range__lhs-seed-9300` |

The names are computed (see *Names and identity* below). Steady Train 800 matches the VP1 training size. Transient validation and ID-test are larger than a 0.1 fraction would give, because selection rolls out whole validation trajectories and the test reports paired per-case differences.

Each count is both the original design size and the target number of successful cases (see *Failed cases and replacements*). The split applies per ID material before the cases are expanded into temporal transitions. Fractions use floors for Train and validation; the remainder goes to ID-test. A material with role `ood` is held out completely as material-family OOD.

### Failed cases and replacements

Each terminally failed candidate is replaced until the material reaches its count, as long as the material's success rate can still reach the plan's `generation.minimum_success_rate` s (table above). The rule holds per original batch (material and sampling regime):

- k counts the successful and n the decided candidates (successful or failed, originals and replacements); undecided candidates, licence waits and cases without a decision do not count. The bound is the one-sided 95 % Clopper–Pearson upper bound of the success rate, the Beta quantile `beta.ppf(0.95, k + 1, n - k)`, and 1 while every decided candidate succeeded.
- Once n ≥ 1 and the bound falls below s, the material is stopped: it admits no fresh candidate of its own, neither original nor replacement, while its running cases finish and the other materials continue. A material that keeps failing stops early: at s = 0.9 two failures without a success already stop it (bound 0.78).
- s is one number with 0 < s ≤ 1 for every material of the plan and part of the authored plan, not a run-time override. It changes neither the computed Dataset name nor the Generation seeds, and the Source keeps its name. The first run records it per original batch as amendment 1 (`registered_by: plan`); rerunning with another value is refused, so a later lowering is an explicit `amend-replacements … --minimum-success-rate S --reason TEXT` ([Generation reference](06_generation_reference.md#failures-and-continuation)) that the Dataset records with its reason.
- Without `--max-admission-cases` the admission limit is derived: the sum over the materials of the count plus the most failures f with which the count can still be reached at s (its last success comes after count − 1 successes and f failures with the bound still at or above s): 430 for the transient plan (85 + 85 + 30 tolerated failures) and 1312 for the steady plan (69 + 69 + 24).
- A replacement is the next prefix-stable Sobol point of a separate seed stream, not a Latin hypercube point. It takes the target slot of the failed candidate and records `parent_failed_case_id`.
- A solve the runtime stopped itself (`stalled`, `timed_out`, `solver_error`) is decided automatically with that typed reason once its step has ended; unless the material is stopped, its replacement is planned as soon as every original candidate of the same material is decided, so replacements run after the material's originals. Any other failure, a COMSOL exit by a signal such as a memory-limit kill included, is declared by hand: inspect it, record it with `fail-case` (original or replacement candidate; [Generation reference](06_generation_reference.md#failures-and-continuation)) and rerun the same plan command.
- The driver stops on a failed case only when no decision is recorded for it for two minutes of reconciliation (it names the case and the `fail-case` route), or when every admitted candidate is decided and a stopped material is below its count, or when an explicit `--max-admission-cases` holds back a replacement that a recorded failure needs (see *Replacements* in `06_generation_reference.md`). The campaign then ends `blocked`, and the message names the material and regime, k/n, the bound and s, and the two ways out: fix the cause and start a new Source (`--new-source --revision LABEL`), or lower s with `amend-replacements` and rerun the same plan command. Combining the cases of the old and the new Source in one Dataset needs the explicit selection `exact` (see *Case selection* below), because `design_order` rejects more than one Source. A recorded failure whose slot a replacement covers never stops the run.

What the Dataset then represents: the cases whose solver run succeeded, selected from the sampled design plus its replacements. It describes that solver-success population, not the full original design domain. Replacing failures does not undo the selection by success, so the failures are reported rather than hidden. The manifest binds this evidence (`source_outcomes`): per contributing Source and original batch its minimum success rate with every amendment (registrant and reason) and the final k, n and whether the batch stopped, the counts of original-candidate failures per material, and a table of every candidate (origin, target slot, parent failed case, outcome, failure stage, class and reason code). `inspect` prints the original-candidate failure rate per material and k/n against s. Rates and counts belong in any report of results obtained on the Dataset. To see where in the design the failures lie, each completed Source writes `cases.csv` (one row per candidate with its outcome, typed reason, runtime and every sampled parameter), and the section «Failures by parameter» of [Generation-input EDA](../notebooks/01_eda_generation_inputs.ipynb) plots every terminal candidate at its sampled values per material, coloured by reason; both show recorded association, not cause.

## Small runs

Explicit overrides change one run without editing the plan. Three cases per ID material with equal thirds, two OOD cases and a four-hour horizon give one Train, one validation and one ID-test case per ID material; the admission cap leaves room for four replacements:

```bash
../runtime/venvs/native/bin/python -m src.datasets.dataset_packages run \
  configs/02_datasets/transient_drying/lentil_chickpea__ood_kidney_bean.yaml \
  --count lentil=3 --count chickpea=3 --count kidney_bean=2 \
  --split-fractions 1/3,1/3,1/3 --duration-hours 4 \
  --node-jobs 1 --max-admission-cases 12 --cores-per-case 8 --preview
```

| Option | Effect |
| --- | --- |
| `--count MATERIAL=N` | Original design size and success target for one material; changes the design |
| `--duration-hours H` | Physical horizon of a transient-drying plan; changes the physical output domain. A steady-flow plan has no horizon and refuses it at plan admission |
| `--split-fractions T,V,I` | Recorded split fractions (`1/3` accepted) |
| `--revision LABEL` | Appends `__rev-LABEL` to the computed Dataset name, for a deliberately separate Dataset of the same plan; the default Source is reused and nothing is simulated again. The label uses lowercase letters and digits joined by single `-`, `_` or `+` |
| `--new-source` | Only together with `--revision LABEL` (else exit 2): the Source is `<default Source name>__rev-LABEL` and its cases are simulated afresh with licensed COMSOL. Use it when the producer science changed; the preview then says that the Source is simulated afresh |
| `--max-admission-cases N` | Distinct fresh solver candidates, recorded and fixed as given; without it the campaign records no cap and derives it from the sum over the materials of the count plus the failures it tolerates at each batch's current minimum success rate, so it grows when a rate is lowered ([Generation reference](06_generation_reference.md#replacements)) |
| `--node-jobs N`, `--cores-per-case N` | Concurrent node jobs (1 to 4, default 2; a per-user cluster-share cap; the COMSOL batch licence counts once per user) and CPUs per case; together they cap the concurrent cases ([Generation reference](06_generation_reference.md)) |
| `--case-timeout-seconds S`, `--no-retry` | Solver runtime limit and licence-retry policy |
| `--driver-partition`, `--driver-time` | Driver job partition and wall limit (`standard` or `long`, default `standard`; default `06:00:00`; the 2 CPUs and `8G` are fixed); not part of the Dataset identity |
| `--continue-with-current-code` | Continue an existing campaign with the current code instead of the source tree it is bound to (recorded as a new execution epoch) |

Seeds, grid, solver settings and sampling method stay as authored. Repeat the same complete command, overrides included, to continue a run. The small run above is named `lentil-train1-val1-test1+chickpea-train1-val1-test1+kidney-bean-ood2__ref-range__lhs-seed-9200__h-4`, so it never collides with the full Dataset.

## What a Dataset contains

```text
storage/02_datasets/<task>/<dataset>/
  dataset_manifest.json       # ordered Sources, representation, roles, split membership, payload digests
  dataset.json                # transient: temporal transition index
  transient_pt_shards/        # transient: training payload
    receipt.json              # admission and payload identity, bound by the manifest
    shard_*.pt
  dataset.pt                  # steady flow: tensor payload
storage/02_datasets/.state/   # build locks and the identity registry (identities/)
```

The payload file names are fixed; the Dataset name only names the directory.

The manifest owns the selection, exact Source evidence, the Source outcome evidence (replacement policy and all candidate outcomes), material roles and the split. For transient drying, the transition index enumerates temporal samples of those cases and the PT shards are the training representation with their own receipt. For both tasks the manifest and the complete payload (index and shards, or the tensor payload) are built and validated in a staging folder under `.state/` and published together in one atomic rename: a Dataset folder is complete or absent, and an interrupted build leaves nothing published. The shards are built before the manifest, which binds the payload digest (`payload_sha256`) and, for transient drying, the digest of the shard receipt (`derived_payload_sha256`, `null` for steady flow); the one manifest validator refuses changed payload or receipt bytes for both tasks. Nothing is added to a published folder afterwards: the Dataset driver's log stays in `runtime/logs/` and the campaign's `logs/`. The receipt records each file's size, hash and file metadata: unchanged files are admitted directly, copied, restored or moved files are re-verified against the recorded hash, and only changed bytes are rejected. A steady-flow Dataset stores all its cases in one tensor payload.

<details>
<summary><strong>Names and identity</strong></summary>

The name is computed from the resolved plan, never authored: `<material>-train<a>-val<b>-test<c>` for each ID material (the Train, validation and ID-test case counts of the split) or `<material>-ood<n>` for a held-out material, joined by `+` in plan order, then `__<domain>__lhs-seed-<seed>` (`ref-range` for `reference_range`, `ref-param-ext` for `reference_and_parameter_extension`), for transient drying a `--duration-hours` horizon that differs from the one configured in the common config as `__h-<hours>` (`4.5` as `4p5`; the configured horizon belongs to the simulation definition and adds no token), and optionally `__rev-<label>`. Overriding counts, split fractions, the horizon or the plan therefore changes the name.

The physical Source name is computed from the plan the same way Generation names an unrevised campaign: `<material>-<count>` per material joined by `+`, then `__<domain>__lhs-seed-<seed>` and for transient drying `__h-<hours>` only when the horizon differs from the configured one (for example `lentil-50+chickpea-50__ref-range__lhs-seed-9200` at the configured horizon and `...__h-4` for `--duration-hours 4`), never authored. A deviating horizon is therefore another Source with the same seeds and needs no revision. `--revision LABEL` names only the Dataset and reuses this Source. With `--new-source` as well the Source becomes that name plus `__rev-LABEL` and owns its own solver runs instead of continuing the default Source. The label changes only the Source name and storage locator: Generation seeds, batch identities and the scientific campaign digest stay those of the plan without it.

The one revision rule, the same for Dataset, Source, run and study: *Same name, changed science: add `--revision LABEL` to the command you run. It never changes seeds, never makes identical science new, and never re-simulates a Source unless you also pass `--new-source` to the Dataset run.*

- The same name with identical content reuses the publication.
- Different content under an occupied name is refused; pass `--revision LABEL` to keep a deliberately separate Dataset (for example after a converter fix). Changed producer science additionally needs `--new-source`, because only that simulates a Source again.
- Identical content under another name is refused (`Identical Dataset content already exists as <task>/<name>; use that name`), so a Dataset exists once. A `--revision` label alone does not make identical content new.

The internal scientific digest binds the ordered Source cases, grid, channel and conversion contracts, roles and split membership, and the recorded conversion digest. The converter implementation is recorded evidence: reading a Dataset never compares it with the running converter, and a converter fix that changes the conversion projection is a new Dataset (`--revision`; see [Code identity](07_reproducibility_and_validation.md#code-identity)). Names, file paths and Git HEAD are provenance, not identity. A registry below `02_datasets/.state/identities/` maps each digest to its Dataset directory; an entry whose directory no longer exists is replaced. Physical Sources carry no Dataset roles, and the Source name never contains the split; a revision label appears in the Source name only with `--new-source`. Details are in [Reproducibility and validation](07_reproducibility_and_validation.md).

</details>

<details>
<summary><strong>Case selection: design_order</strong></summary>

`design_order` chooses from cases that already succeeded. For each material it requires enough successes, orders them as Generation does (original cases by case index, then Sobol replacements by candidate ordinal) and takes the requested number. A Source holds at most as many successes per material as its success target, which a plan sets to the requested count, so the selection takes every success of the Source. No seed takes part: the membership and its order follow from the Source alone. `exact` selection instead lists the source cases explicitly and may combine Sources; a Dataset never holds the same physical input twice. The split follows selection with its own seed, which shuffles each ID material's cases in this order.

</details>

<details>
<summary><strong>Evaluation Datasets</strong></summary>

A Dataset whose plan gives a material the role `ood` holds that material out as family OOD. Training configurations list such Datasets in `data.additional_evaluation_datasets`; their declared OOD cases, plus any OOD cases of the training Dataset itself, form the additional evaluation membership. The run's split records each of these Datasets as its own part with its identity, cases, membership digest and evaluation assignment ([Reproducibility and validation](07_reproducibility_and_validation.md#runs-and-publication)). For both tasks, a declared holdout that reuses a training material is rejected, and no two roles or parts may share a physical source case (the same case input or simulation). When a run is reopened for inference or artifacts, each split part must match the current manifest's identity and fixed partition (an additional part holds exactly its Dataset's fixed OOD cases).

</details>
