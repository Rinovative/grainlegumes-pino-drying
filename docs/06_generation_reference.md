# Generation reference

This guide covers Generation beyond the normal Dataset plan: the campaign driver, input preparation, failure handling and continuation, the storage layout of Sources and the meaning of retained timing. For learning data, start with [Datasets](03_datasets.md); the physics and sampling are in [Physical model and parameters](02_physical_model.md).

[Documentation index](README.md)

## Campaign driver

A Dataset plan run is the campaign entry: it resolves the plan's `generation` block into one physical Source campaign and runs it through the driver and node jobs described here ([Datasets](03_datasets.md)). Sources carry no Train, validation, test or OOD roles. `./scripts/generation run CONFIG` starts the same driver for a campaign YAML (`schema_kind: generation_source_campaign`) without a Dataset; the repository ships no such file. The resolved plan records the overrides and the effective values. `--dry-run` prints the resolved plan and checks execution readiness (node launcher, native imports and the COMSOL module) without starting work or freezing the source.

`run` is a driver, like the Dataset plan run. On the login node it freezes the source and submits the controller as its own CPU Slurm job (2 CPUs, 8 GB, `--driver-partition` `standard` or `long`, `--driver-time` default 06:00:00) that runs from the frozen tree, then prints the submission:

```json
{
  "job_id": "4711",
  "log_path": "RUNTIME/logs/generation/generation-transient_drying-4711.log",
  "status": "SUBMITTED",
  "tree_id": "COMMIT12-FINGERPRINT12"
}
```

The driver prepares the inputs and runs the cases in node jobs: one Slurm job per node-job slot (`gen-seat-1` to `gen-seat-4`, Slurm singleton, at most `cluster.node_jobs` across all Generation runs of the user, `site.cores_per_node` cores on a `standard` node with at least that many CPUs chosen by Slurm; node jobs are not exclusive, so the remaining CPUs of a larger node stay available to other jobs). A generic node worker inside each node job starts the cases as job steps of `cores_per_case` cores each, floor(32/k) at a time, and refills a free slot as soon as a step ends. The driver dispatches each case as a ticket into a node job and records the dispatch in `_record/execution/run.json` before the ticket exists; it reconciles every dispatch from the step's own Slurm accounting (state, exit code, times, node), so an out-of-memory or failed step is attributed to its case and a step that never started is dispatched again without consuming an attempt. When no case remains, the driver ends each node job with a stop ticket; an orphaned node job ends after 15 idle minutes. The driver runs until the campaign is complete, staged at its admission cap, or stopped by an error; its log is the `log_path`. Because the node jobs and `run.json` survive the driver, rerunning the same command adopts the open node jobs and continues the campaign after an error or after the driver's time limit; `status` (with one line per node job) and `cancel` run on the login node at any time. A cancellation is a persisted desired state that every later driver call repeats until all node jobs have ended.

| Option | Effect |
| --- | --- |
| `--materials` | Material families |
| `--cases-per-material` | Original design size per material |
| `--duration-hours` | Physical horizon; regular output intervals must stay exact |
| `--revision LABEL` | Source campaigns only: names the Source `SOURCE_NAME__rev-LABEL` beside the existing one when its science changed (for example the producer code); the label uses lowercase letters and digits joined by single `-`, `_` or `+`; it changes no seed, batch identity or campaign digest, and the plan records it with the other overrides |
| `--max-admission-cases` | Distinct fresh solver candidates, recorded and fixed; input preparation still covers the full design ([Replacements](#replacements)) |
| `--node-jobs` | Node-job slots the run may use (1 to 4, default 2 from `cluster.node_jobs`; a per-user cap on the five shared 32-core nodes, so use 3 or 4 only when they are quiet); with `--cores-per-case` k it caps the concurrent cases at slots × floor(32/k) |
| `--cores-per-case` | CPUs per case (k) |
| `--case-timeout-seconds` | Solver runtime limit |
| `--no-retry` | No licence retries (neither the node wait nor the retry within an allocation) |
| `--driver-partition`, `--driver-time` | Partition (`standard`, `long`) and time limit `[D-]HH:MM:SS` of the driver job |
| `--continue-with-current-code` | Continue an existing campaign with the current code instead of its bound source tree (see [Source snapshots](#source-storage)) |
| `--dry-run` | Print the plan and check readiness; start nothing |

The execution file (`configs/01_generation/execution.yaml`, schema version 1) sets a 3600 s case timeout (the orderly stop starts 300 s earlier, after 55 minutes; in the core-scaling measurement of 28 October 2026 the longest success at k = 4 took 29.8 minutes, the same case took 20.0 and 29.8 minutes in its two replicates, and no case timed out, so the generous limit costs time only for cases that progress but never finish and keeps slowly converging cases in the data), a stall rule, and the node-job resources: `cluster.cores_per_case` k (4, so eight cases run at a time per node), `cluster.node_jobs` (2, at most 4, a per-user cap on the five shared 32-core `standard` nodes that leaves one node free; the batch licence counts once per user and does not limit it, see the licence paragraph under [Source storage](#source-storage)), `cluster.memory_per_case` (20G, provisional; the largest observed transient case peak was 15.8 GB) and `cluster.node_job_wall_time`. The loader derives `concurrent_cases_per_node` = floor(32/k) and `node_job_memory` = floor(32/k) × memory_per_case + `node_memory_overhead`, and rejects a configuration whose node job would exceed `node_memory_limit` instead of lowering the concurrency (k = 2 needs 328G and therefore runs only on the 514 GB nodes). Each case has an outer time limit of the timeout plus 600 s for publication. `--mem`, `--dependency`, `--comment`, `--exclude`, GPU and node-selection options are owned by the pipeline and rejected in `scheduler_options`. The stall rule `runtime.stall_stop` applies to transient cases only, the profile with a time horizon: a stationary steady solve never reports a simulated time, so a steady case builds no stall rule even though the shared execution file sets one, records `stall_progress` as null, and only the timeout bounds it. For a transient case the rule stops a case whose COMSOL simulated time is still below `minimum_simulated_seconds` (3600 s) after `check_after_seconds` (900 s) of solving; the stop is orderly, escalates after the same reserve and is recorded as a solver failure with the typed reason `stalled` (detail for example `stalled: 0.98 h of 168 h simulated after 900 s solving (threshold: 1 h simulated after 900 s)`), not as a timeout. An unknown simulated time never triggers it, and the timeout stays the backstop for slow cases that still progress. The 900 s stays deliberately short so that cases that never converge stop early: in that measurement the stuck cases stood at 0 to 63 s simulated after 15.6 minutes at k = 2 to 8; at k = 4 it stopped one case. A case stuck at its start (2026-10-26: 54 s simulated in 45 minutes) is stopped after 15 instead of 55 minutes and frees its COMSOL licence; successful cases had passed 1 h simulated well before 10 minutes, one of them only after a slow first 5 minutes. A stall is not always reproducible (one case stalled in 1 of its 10 measured runs); a stalled case is replaced, not retried. To calibrate the rule, every case records in `runtime/timing.json` (and its evidence copy) `stall_progress`: the rule's `minimum_simulated_seconds` and `reached_after_seconds`, the solving time, on the clock of `check_after_seconds`, at which COMSOL's simulated time was first seen at or above that minimum (null when it was not seen, for example because the case stalled or its solver ended before stop control started; `stall_progress` is null without a stall rule and for every steady case). A separate uncached reader of the solver output records it after each poll, so the value is accurate to the poll interval and never changes a stop decision. The timing reader exposes both values with the component availability `stall_progress`. After the first production campaign, `check_after_seconds` is set from the largest `reached_after_seconds` of the successful cases. Without `stall_stop` only the timeout applies. The admission cap counts candidates, not allocations: continuations can request further allocations.

`cluster.cores_per_case` is 4. It was chosen on 2026-10-28 from the measured node throughput per core count (k = 2 to 32): about 15 successful cases per node-hour at k = 4 against about 11 at k = 8.

## Input-only generation and status

`inputs` prepares selected physical inputs of a campaign YAML without COMSOL through the Slurm input path (a Dataset plan run prepares its own inputs first). It needs a batch and a case selection; actual execution freezes the source like `run`, with or without uncommitted changes.

```bash
./scripts/generation inputs CAMPAIGN_YAML \
  --only-batch transient_drying__lentil__natural \
  --case-start 1 --case-count 10 --dry-run
```

`--all-batches` replaces `--only-batch`, `--all-cases` replaces `--case-start`/`--case-count`, `--only-regime natural` filters batches. Ten cases from a fifty-point design keep that design's seeds and positions. Inspect prepared inputs in `notebooks/01_eda_generation_inputs.ipynb`; `status` reports solver campaign state.

## Counts and case identity

| Count | Meaning |
| --- | --- |
| Original design size | Points constructed together per material and regime |
| Prepared inputs | Cases with generated and admitted physical inputs |
| Fresh admitted candidates | Distinct candidates charged to the admission limit |
| Attempts | Recorded execution and recovery attempts; can exceed the candidates |
| Successful candidates | Validated physical output with content hashes and a successful terminal decision |
| Source selected pool | Sealed successful cases in original-design order, replacements after primaries |
| Dataset selected count | Successful cases chosen by the Dataset |
| Training count | Dataset members assigned to Train |

Case IDs are material-local (`case_0001`, …) and follow candidate positions, not completion order. A retry keeps its ID and appends attempt evidence; an explicit replacement receives a new ID and keeps its parent design and seeds. A Source seals its selected pool when its success target is met; a short Source stays incomplete.

## Failures and continuation

1. **Spatial inputs** retry a complete stochastic realization with up to 32 deterministic seed plans when the texture violates the mean-porosity support; the packing scatter seed stays fixed. Structural and coupling failures stop at once.
2. **Schedules** retry with up to 32 seed plans for degenerate or infeasible schedules; values are never clipped. Exhaustion raises an error. Both retries keep the candidate and its design coordinates.
3. **Licence capacity** retries follow the configured policy unless `--no-retry` is set.
4. **Solves the runtime stopped itself** are decided automatically. The runtime classifies each stop with a typed reason: `stalled` (the stall rule), `timed_out` (the case timeout) or `solver_error` (COMSOL exited non-zero, except by a signal, for example a memory-limit kill, which stays unclassified and is declared by hand). The attempt stores it as its `reason_code` with a human `detail` that names the solver progress of the task and the solving time: the simulated hours of the horizon for a transient case (`solver_error: COMSOL exited with status 1 at 0.98 h of 168 h simulated after 120 s solving`), the last stationary nonlinear iteration for a steady case (`solver_error: COMSOL exited with status 1 at nonlinear iteration 37 after 120 s solving`), and `no progress reported` only when the solver output holds none (a transient case that ends during its initial stationary airflow solve also names that iteration), and keeps the exception type in its evidence; an attempt that the runtime did not classify reads `unclassified`. Once the step of the case's latest dispatch has terminal Slurm accounting and no dispatch of the case is active, the next feed records the failed decision with that reason and detail. That feed returns without draining a node job; once every original candidate of the batch is terminal, a following feed plans the replacement, unless the batch's minimum success rate has stopped it (see [Replacements](#replacements)).
5. **Export, conversion and validation failures**, and unclassified solver failures, end the candidate attempt without automatic retry: the driver stops with the `fail-case` instruction. Inspect the evidence and declare the case failed; its replacement is solved again with COMSOL.
6. **Controller or publication errors** stop automatic feeding; rerun the same campaign after inspection. The rerun solves the case again with COMSOL; nothing is reused from the retained output.

Record an inspected failure:

```bash
./scripts/generation fail-case "$CAMPAIGN_RUN_ID" MATERIAL case_0001 \
  --reason-code nonconvergence --detail "inspected terminal solver evidence"
```

`fail-case` accepts original and replacement candidates; a replacement is checked against its own one-case batch, not its parent batch. It and the automatic decision share one decision path with the same checks. Terminal decisions are immutable; repeating the same decision is idempotent, a conflicting one fails. A validly published case whose step ended without writing its success decision, for example after a scratch-cleanup error, receives it from the next scheduled feed; a metadata-only feed asks for that feed instead of hashing the case HDF5. Resume uses the frozen campaign, validated artifacts and exact scheduler evidence; missing scheduler evidence blocks duplicate admission.

`status` reports each case's recorded `terminal_decision` (`successful`, `failed` or `null`). While no job is pending or running, the driver (`run`, and the Dataset command that wraps it) keeps reconciling with `resume-campaign` (which prints one JSON summary line with the state, a compute request with its `continuation_reason`, the submission-record counts and the dispatches this resume reclassified as not started; the full manifest stays in the campaign's execution record `_record/execution/run.json`) before it acts on a failure, because a status read right after the jobs ended can predate their evidence on the driver's node (shared-filesystem caching) and their reconciliation. It stops only if the condition has held for two minutes: a failed case with `terminal_decision` `null` (the message names the cases and the `fail-case` route; rerun the same command after recording them), or a success deficit that no planned replacement covers because a stopped batch admits none (the message names the batch with k/n, its bound and s, and the ways out below). A recorded failure whose slot is covered by a replacement never stops the run.

### Replacements

Once every original candidate of an original batch (material and regime) is terminal, each terminal failure takes the next replacement, and a failed replacement frees its target slot for the next one. Replacements are prefix-stable Sobol points from a separate seed namespace and do not restore a Latin hypercube; each keeps its target slot and `parent_failed_case_id`. Originals come before replacements in the selected pool, and every failure stays in the Source as a candidate.

The batch's minimum success rate s bounds this. k counts its successful and n its decided candidates, originals and replacements alike; undecided candidates, licence waits and cases without a decision do not count. Once n ≥ 1 and the one-sided 95 % Clopper–Pearson upper bound of the success rate (`beta.ppf(0.95, k + 1, n - k)`, 1 for k = n) falls below s, the batch is stopped: it admits no fresh candidate of its own, neither original nor replacement, while its running cases finish and other batches continue. Every feed evaluates the stop from the recorded decisions, so a success recorded meanwhile lifts a stop at once and never blocks; a decision recorded while a replacement is planned that stops its batch or closes its deficit ends planning for that batch only, and the feed goes on. When every admitted candidate is decided and a stopped batch is below its count, the campaign ends `blocked`; `status` and the driver name the batch with k/n, the bound and s, and the two ways out: fix the cause and start a new Source (`--revision LABEL`), or lower s and rerun the same command.

- **Dataset plans** set s as `generation.minimum_success_rate` ([Datasets](03_datasets.md)). Campaign creation records it once per original batch as amendment 1 (`registered_by: plan`); a rerun must repeat it exactly and a different value is refused. s changes neither the Source name nor the seeds.
- **Campaigns without a plan** use s = 1, so their first recorded failure stops the batch until s is lowered.

```bash
./scripts/generation amend-replacements "$CAMPAIGN_RUN_ID" --material MATERIAL --regime natural \
  --minimum-success-rate 0.8 --reason "first failure inspected; accept a lower success rate"
```

`--minimum-success-rate` must lie below the batch's current s; repeating the current value is idempotent, a higher one is refused, and each lowering appends an amendment with its reason (`registered_by: explicit`); a sealed selected pool refuses it. Like `fail-case`, the command reads the live checkout, freezes no source and starts no job. Replacements are fresh admissions. A Dataset plan run without `--max-admission-cases` records `max_admission_cases: null` and derives its cap at every feed: each material's count plus the failures it tolerates at its batch's current s, so a lowered s raises it; `status` shows it as `max_admission_cases=N (plan rates)`. An explicit `--max-admission-cases` (or a `max_admission_cases` override) is recorded and stays as given, even when it equals that default; a campaign without a plan records the execution configuration's value. A rerun may not lower the effective cap (a derived one at the current rates), also not when it switches between an explicit and a derived cap. When an explicit cap holds back a candidate that a remaining success slot needs after a recorded failure, and every admitted candidate is decided, the campaign ends `blocked`; `status` and the driver name the cap: rerun the same command with a larger `--max-admission-cases`.

## Logs

Every Generation and Dataset job writes its log under `runtime/logs/`, submitted with `--open-mode=append`, so a rerun or requeue appends and never truncates a linked log. Once the campaign directory exists, the logs are also hard-linked under their own names into the campaign's `logs/`: a link is the same file, so it grows while the job writes and survives a cleanup of `runtime/logs/`. Nothing is copied, moved or deleted.

| Log | In `runtime/logs/` | Linked into the campaign |
| --- | --- | --- |
| Driver of `run` | `generation/generation-TASK-JOB.log` | as soon as input preparation has created the campaign, also when preparation or `submit-campaign` then fails |
| Dataset driver | `datasets/dataset-TASK-JOB.log` | as the Generation driver; never into the Dataset folder, which nothing changes after its atomic publication ([Datasets](03_datasets.md)) |
| Node job | `generation/node-JOB.out`, `generation/node-JOB.err` | when the node worker starts |
| Helper step | `generation/python-OPERATION.XXXXXXXX.out` (and `.err` when it ran through `srun`) | when the step ends, or when the campaign becomes known for steps that ran before it existed; a successful step whose output is one JSON object (the result the driver receives) leaves only `OPERATION: JSON result returned to the driver (N bytes, sha256 …)` with its `state`, any other output and every failed step keep the raw output |

A job finds its own log from the same `--output`/`--error` template the pipeline gave its `sbatch` call, with `%j` replaced by `SLURM_JOB_ID`: `generation/node-%j.out` and `.err` for a node job, `AREA/NAME-%j.log` for a driver, where `NAME` is its `SLURM_JOB_NAME`; the code that builds those arguments owns the templates. Only an existing regular file is linked. After the link is created, its device and inode are compared with the file that was named; on a mismatch (the file was replaced meanwhile) the link is removed again with a warning. A link never creates the campaign folder and never fails a job: a missing folder, a symbolic link or a file-system error is one warning on stderr, and a different file under the same name in `logs/` stays and is reported.

- **Driver log:** each stage starts with `[n/N] STAGE` and ends with `[n/N] STAGE done in DURATION`. While cases run, a live board (counts, elapsed and estimated remaining time, one line per node job and per case) appears at the first poll, at every state change and otherwise every 300 s; a case that changes its group first gets one line `TIMESTAMP  case MATERIAL/case_NNNN: OLD -> NEW (REASON)` (a case that leaves `failed` or `verifying` adds `; was REASON`), so a case that failed and then settled to done is traceable. The estimate is printed only after one case ended and only while it stays below 10 times the larger of the driver's elapsed time and the done cases' total run time per seat (so a restarted driver estimates from the campaign's measured work), otherwise the header says `ETA: not yet estimable`; a board that cannot be rendered prints `status board unavailable: MESSAGE` once per distinct message and changes nothing else. A failure ends with `FAILED: STAGE: MESSAGE` and, when a command continues the run, `Next: COMMAND`: the rerun of a started campaign (`status` for an authored campaign), and the `--dry-run` of the same config for a failed plan resolution or preflight of a run (none for an authored campaign or a `status`).
- **Node log:** one preflight report per node job (each case step repeats the checks and reports only a failing one), `CASE START …` and `CASE END MATERIAL case_NNNN STATUS at TIMESTAMP after DURATION` per case step (a failed step appends `: REASON`; a stall, which only a transient case can have, names the threshold and the simulated time reached, and any other runtime stop names the solver progress of its task as above), the plain line `MATERIAL case_NNNN stopped: REASON` of a failed step in `node-JOB.err` (followed by a JSON failure record only when the runtime did not classify the stop as stalled, timed out or a solver error; Slurm adds its own `srun: error: … Exited with exit code` line), and `Node worker ended: …` at the end.
- **Progress and estimate:** a running case shows its simulated time past the start of its physical window against the horizon (`25 %  1 h / 4 h`) and, once a case ended, an ETA extrapolated from it. `status` reports both per case as `physical_start_hours` and `physical_horizon_hours` (stop minus start) from the `time` section of its batch's persisted `MATERIAL/_record/REGIME/resolved_generation_config.json`, so the configured horizon and a `--duration-hours` override both show progress; a running case always has that file. The remaining time keeps each running case on its seat and gives each waiting case, with the median duration of the finished cases, whole to the seat that frees first (one waiting case of 1 h on eight free seats still takes 1 h).
- **Failed cases:** the board shows the typed reason and detail, then `replaced by case_NNNN` for the case that took the slot (from the replacement's `parent_failed_case_id` in `status`), `decision recorded`, or `no decision recorded`. A case whose step ended `COMPLETED` but whose result evidence the driver cannot see yet (`completed_without_valid_case_evidence`, typically the cross-node file-system lag) shows as `verifying` with `step ended, result not yet visible` for the driver's 120 s settle window, and as failed only afterwards; the header counts it as `verifying N` while there is one.

## Source storage

Sources live under `storage/01_generation/TASK/SOURCE_NAME/`; controller and scheduler logs live under `runtime/logs/` with hard links in the campaign's `logs/` (see [Logs](#logs)). The printed Source name is the catalogue selector; the execution handle is `source~TASK~SOURCE_NAME`. The name is computed from the original design: `MATERIAL-COUNT` per material in registry order joined by `+`, then `__DOMAIN` (`ref-range` or `ref-param-ext`), `__METHOD-seed-SEED` and for transient drying the physical horizon `__h-HOURS` only when `--duration-hours` makes it differ from the horizon configured in the common config, for example `lentil-3+chickpea-3+kidney-bean-2__ref-range__lhs-seed-9200` and `lentil-3+chickpea-3+kidney-bean-2__ref-range__lhs-seed-9200__h-4`; a deviating horizon keeps the seeds but is another Source. Internal scientific digests establish compatibility. A compatible repeated request reuses its Source; different science under the same name fails before writing, and the message names the fix. Same name, changed science: add `--revision LABEL` to the command you run. It never changes seeds, never makes identical science new, and never re-simulates a Source unless you also pass `--new-source` to the Dataset run. Without a Dataset, `run --revision LABEL` names the Source `SOURCE_NAME__rev-LABEL`; a Dataset plan run does the same only with `--revision LABEL --new-source` ([Datasets](03_datasets.md)), while `--revision` alone names just the Dataset. A campaign config may author the same label as its optional key `revision`; there is no free Source label (`source_name` was removed).

Candidate and replacement seeds derive from a `seed_key` frozen in `campaign_plan.json`: a digest of the task, the original design size per material, the domain profile, the sampling method with its seed base and the campaign purpose. The name and the storage layout are not inputs, so renaming a Source keeps its random fields.

Human files come first: a campaign shows its materials, each material its cases, and each case its `inputs/` and `result/`. Machine records sit in the reserved `_record/` folders at campaign, material and case level, and the campaign's job logs in `logs/`, and the derived `cases.csv` view beside them. `_record`, `logs` and `cases.csv` are the only reserved names: every reader recognizes them explicitly and still rejects any other entry beside the cases, a symbolic link, a missing case number or an incomplete case. A campaign in the earlier layout (`campaign_plan.json` directly in the campaign, `design/`, `cases/`, `execution/` beside the materials) is not read: listing the task's campaigns stops with "superseded storage layout", and naming it fails because `_record/campaign_plan.json` is missing.

```text
01_generation/
  TASK/SOURCE_NAME/
    MATERIAL/
      case_NNNN/
        inputs/               # case.json, fields.csv, scalars.csv, schedule.csv, input_reference.json
        result/               # case.h5, provenance.json, _SUCCESS
        result/evidence/      # execution_provenance, processing_provenance, status, timing, solver.log
        _record/
          reservation.json    # the only reservation copy, published atomically with its case directory
          decision.json       # immutable terminal decision
          attempts/           # attempt_NNNN.json classified attempts; run_NNNN/ failure bundles
      _record/REGIME/         # batch_manifest, input_generation_manifest, resolved_generation_config, _SUCCESS
        replacements/         # amendment_NNNN.json: minimum success rate amendments
    logs/                     # hard links of the campaign's driver, node-job and helper logs (see Logs)
    cases.csv                 # derived one-row-per-candidate view, written when the Source completes
    _record/
      campaign_plan.json      # frozen plan with its seed_key
      selected_pool.json      # sealed success membership (absent until the target is met)
      execution/              # run.json, cancellations.json, epochs/, slurm/, interruptions/, license/, reconciliation/,
                              # reconciliation_failures/, node_jobs/
    .state/                   # controller.lock, submission.lock, batch locks, publication staging, quarantine
  .state/                     # raw-inputs/ (input transactions, lock anchors)
```

Operational records below `_record/execution/` are keyed by `MATERIAL-REGIME` (for example `license/lentil-natural/case_0001/`), which must be unique among the frozen batches of one Source.

Each atomic publication is one rename of a complete directory: a case appears with its `_record/reservation.json` (staged in `.state/`), `inputs/` and `result/` are each renamed into the existing case, and an attempt bundle `run_NNNN/` is renamed inside `_record/attempts/`. Locks, result staging and quarantine stay in the campaign's `.state/`, input staging in the storage-wide `01_generation/.state/raw-inputs/transactions/INPUT_GENERATION_ID/`, so every rename stays on one file system. Inputs and results have separate recovery boundaries:

- **Inputs:** an input generation stages its new cases and batch metadata in its transaction, writes the journal `transaction.json` (manifest hash, new case IDs) before any final path changes, renames each new `inputs/` into its case, admits the staged metadata and only then publishes `input_generation_manifest.json`. A transaction without a journal is abandoned staging and is deleted; one with a journal is completed (missing renames, then the manifest) by the next input generation of that batch under its input lock.
- **Results:** a staged `result/` is renamed in only after an incomplete `result/` without `_SUCCESS` has been moved into the campaign's `.state/` quarantine; a complete one is never replaced. Quarantined directories are kept and counted, never published.

<details>
<summary><strong>Stored records</strong></summary>

| Record | Content |
| --- | --- |
| `_record/campaign_plan.json` | Immutable effective science, original counts, seeds and `seed_key`, block designs and permutations, configuration content, physical mapping, producer identity, model reference and hash |
| `_record/execution/run.json` | Mutable ledger: node jobs (seat, token, allocation, lease, draining, node directory `_record/execution/node_jobs/TOKEN`), dispatch records (submission index, node job, ticket, step name, attempt, status) and the admitted source binding |
| `_record/execution/node_jobs/TOKEN/` | Node directory of one node job: `tickets/`, worker `receipts/`, `control/cancel`, `node.json`, `node_exit.json` |
| `_record/execution/cancellations.json` | Receipts of graceful and forced cancellations |
| `MATERIAL/_record/REGIME/resolved_generation_config.json` | Batch document validated by input admission and batch publication; its `time` section gives the physical start and horizon `status` reports per case |
| `MATERIAL/_record/REGIME/replacements/` | Append-only minimum-success-rate amendments with rate, registrant (`plan` or `explicit`) and reason |
| `_record/execution/epochs/` | Ordered execution-resource and source admission history |
| `case_NNNN/_record/reservation.json` | Durable allocation; the case directory appears only with it |
| `_record/execution/reconciliation/`, `_record/execution/reconciliation_failures/` | Scheduler timestamp normalization and case classification evidence, per `MATERIAL-REGIME/case_NNNN/` |
| `_record/execution/slurm/` | Phase and counters per dispatch (`JOB-SUBMISSION_INDEX.json`, for example `7001-3.json`) for live status |
| `_record/execution/interruptions/` | Worker-interruption receipts per dispatch (`JOB-SUBMISSION_INDEX.json`) |
| `_record/execution/license/` | Per case below `MATERIAL-REGIME/case_NNNN/`: the licence wait `wait.json`, checkout windows `windows/DISPATCH.json` and status recoveries `status_recoveries/` |
| `.state/` | Locks, input transaction staging and quarantine |
| `case_NNNN/inputs/` | Case definition, sampled and derived parameters, spatial and schedule inputs, seeds, rejection evidence, input reference and hashes |
| `case_NNNN/_record/attempts/attempt_NNNN.json`, `case_NNNN/_record/decision.json` | Append-only classified attempts (a solver attempt's `reason_code` is `stalled`, `timed_out`, `solver_error` or `unclassified`) and immutable terminal decisions |
| `case_NNNN/_record/attempts/run_NNNN/` | Evidence bundle of one failed execution: receipt, retained payload and predecessor hashes |
| `result/evidence/status.json`, `timing.json` | Execution outcome and measured timing boundaries |
| `result/provenance.json` | Content inventory (paths relative to `result/`) and conversion provenance |
| `result/evidence/` provenance files | Execution environment and arguments, processing provenance |
| `result/case.h5`, `result/_SUCCESS` | Canonical physical output and hash-bound publication markers |
| `_record/selected_pool.json` | Sealed successful Source membership and artifact hashes |
| `logs/` | Hard links of the campaign's job logs; the originals stay in `runtime/logs/` (see Logs) |
| `cases.csv` | Derived view, rewritten atomically and byte-identically whenever the campaign settles complete: one row per candidate in material and case order with `material`, `case_id`, `origin` (`original` or `replacement`), `outcome`, `reason_code`, `detail`, `duration_s`, `simulated_end_h`, `final_bulk_moisture_wb`, every sampled parameter as its own column (sorted by name), and `solver_log`, the campaign-relative path of the last failed attempt's retained COMSOL log (`stdout.log`); empty cells where a value does not apply (a steady case has no simulated end, a failed attempt may have written no record). A successful row reads what admission requires of it, for both tasks: the `status.json` and `timing.json` of its evidence and, for a transient case, its canonical final bulk moisture; an unreadable one stops the rendering with an error that names the case and the file, and a successful case without its processed status is refused the same way when the campaign status is read |
| `source_snapshots/` (storage root, outside `01_generation/`) | Exact admitted source bytes shared by Generation, Datasets and ML |

Lock and staging records coordinate concurrent workers; inputs, decisions, outputs and snapshots are provenance.

</details>

<details>
<summary><strong>Source snapshots</strong></summary>

A snapshot records the base Git commit, a binary-capable diff of tracked changes, non-ignored untracked bytes, safe links and the permission inventory. Recovery needs the base Git objects plus that diff and payload. The tracked COMSOL templates `simulation/steady_flow/steady_flow_template.mph` and `simulation/transient_drying/transient_drying_template.mph` recover from the commit or the diff; the Source also records template path and hash, raw plan content and the physical mapping.

Identical commit and fingerprint reuse one snapshot, stored as `storage/source_snapshots/COMMIT12-FINGERPRINT12/` (the first twelve hexadecimal digits of each); `snapshot.json` keeps the full commit and fingerprint. A different snapshot under the same short name fails closed. Sources and cases reference it. Runtime environments, scientific storage and ignored files are outside the snapshot. Environment records describe Python, packages, platform, COMSOL module and locks; regeneration needs a compatible provisioned environment.

Every executing command (`./scripts/generation run` without `--dry-run`, `./scripts/generation inputs` without `--dry-run`, and `python -m src.datasets.dataset_packages run`), also from a clean checkout, captures its snapshot and restores it once into the read-only source tree `runtime/sources/COMMIT12-FINGERPRINT12/`: the commit's files plus the recorded diff and untracked bytes, without `.git`, directories without write bits. `./scripts/generation run` and the Dataset command submit their drivers as their own Slurm jobs that run from that tree ([Datasets](03_datasets.md)), and `inputs` runs its Slurm input step from it. The controller, Dataset assembly, Slurm workers and case jobs therefore never run live code; later edits in `repo/` do not reach submitted work, and configurations are read from the tree. Freezing runs only on the login node: the three commands start there, and inside a Slurm allocation they stop with "Freeze the source on the login node", so compute nodes never run the capture and archive Git commands. Read-only commands (`status`, `--dry-run`, `cancel`, `fail-case`, `amend-replacements`, `--preview`) stay in the live checkout and leave no snapshot or tree.

Uncommitted changes are always admitted: the snapshot records them exactly, and the `Source:` line the Generation commands print on stderr states whether the snapshot carries any (`snapshot carries uncommitted changes` or `snapshot is clean`).

An existing campaign continues in the tree of its latest execution epoch, which is restored from its snapshot when it was pruned; a new campaign uses the current code. `--continue-with-current-code` (on `run` and the Dataset command) continues an existing campaign with the current code instead: the driver job runs from the current tree, and the campaign admits the switch only when no job of the campaign is queued or running and records it as a new execution epoch. A change to the producer implementation changes the campaign identity and is never admitted by this switch: finish the campaign in its bound source tree or start a new Source with `--revision <label>` (see [Code identity](07_reproducibility_and_validation.md#code-identity)). Each submitted job leases its tree (`runtime/sources/.leases/TREE/JOB_ID`): case jobs and the Generation and Dataset driver jobs under the job ID `sbatch` returned. A lease that fails after a successful submission only prints one warning; the job ID is still recorded and the tree stays protected by its 24-hour recency. A launcher deletes trees that no queued or running job leases and that nobody used for 24 hours, and an unanswered `squeue` deletes nothing. Parallel launches that freeze the same commit and fingerprint share one snapshot and one tree: the process that loses the publication race accepts the identical result, and different content under the same name fails closed. A deleted tree is restored from its snapshot and base commit when needed. The tree is checked against its snapshot without Git (`scripts/source_fingerprint.py --verify-tree TREE STORAGE/source_snapshots`: bytes, links, permissions and membership).

A process running from a tree is in tree mode (`SOURCE_TREE_ID` set). It requires explicit `PROJECT_ROOT` (equal to `RUNTIME_ROOT/sources/SOURCE_TREE_ID`), `STORAGE_ROOT` and `RUNTIME_ROOT`, infers no sibling default and never calls Git. Compute nodes need no Git: `scripts/generation_node.sh` and `scripts/generation_prerequisites.sh` verify the tree before any project Python runs and again after the command, compare the snapshot's commit and fingerprint with `GENERATION_GIT_COMMIT` and `GENERATION_SOURCE_SHA256`, and require `GENERATION_NATIVE_VENV` to be `RUNTIME_ROOT/venvs/native` with an executable `python`. The Python publication guard repeats the verification in-process. A Dataset built inside a tree binds the admitted snapshot instead of capturing another.

Physical identity binds the recorded producer digest and the numerical environment. The producer implementation is the declared Generation input projection (sampling, fields, schedules, materials, vocabulary) for input reuse and the Generation source projection (the input projection plus case admission, solver invocation, export conversion and acceptance policy) for the Source identity; both are declared in `generation_contracts_source.py` (see [Reproducibility and validation](07_reproducibility_and_validation.md)). Comment or documentation edits and Dataset roles leave it unchanged; a change inside the producer's implementation invalidates cached inputs. The Git commit is not part of input-batch identity: it stays in the input manifest as provenance, and reusing a batch keeps and reports its original commit. Continuing a frozen campaign requires its recorded Source projection digest to equal the running one, otherwise it is refused with "finish it in its bound source tree or start a new Source with --revision <label>"; `--continue-with-current-code` switches the execution tree only and does not waive this check, and inspection of a campaign stays unguarded ([Code identity](07_reproducibility_and_validation.md#code-identity)).

</details>

<details>
<summary><strong>Retention</strong></summary>

`retention: compact` keeps grid, horizon, output fields and validation unchanged and limits stored execution payloads:

- Successful cases keep canonical HDF5, solver log, timing, execution and processing provenance and publication evidence; COMSOL exports and the solved `.mph` model are omitted. Full retention keeps both in `result/evidence/`.
- Failed attempts of every failure stage, conversion and publication included, keep status, stop, timing, provenance, the transient time-stepping history `runtime/solver_steps.csv` (step, simulated time and step size in seconds, solver counts, time-step and nonlinear failures) and the logs: a log above 1 MiB keeps a bounded head and tail copy plus the complete log as `NAME.log.gz`; the omitted exports, converted HDF5 and solved model are recorded. Full retention keeps them.

Cleanup acts only on owned scratch after its publication and recovery checks. A failed attempt never carries a success marker.

COMSOL always runs with `-usebatchlic`: it uses the separate batch licence pool (COMSOLBATCH, CFDBATCH, HEATTRANSFERBATCH, two seats each for CFD and heat transfer) and never the GUI features COMSOL, CFD and HEATTRANSFER of users at OST workstations. Measured on 27 and 28 October 2026: all batch checkouts of one user count as one seat, across two node jobs and also across five hosts at once (probe job 873137, no licence error), so the licence does not limit the number of node jobs; the limit is the cluster (five `standard` nodes with 32 cores). Every user running COMSOL (batch or GUI) also holds one of the five COMSOLUSER seats, again once per user across hosts (the five hosts of probe 873137 held one), so Generation costs GUI users one COMSOLUSER seat while it runs; with four other users in the GUI this fills the pool. `-usebatchlic`, `-ckl` and `-keeplicenses` cannot be set through `extra_arguments`. COMSOL also runs with `-autosave off -recoverydir WORK/comsol/recovery -tmpdir WORK/comsol/tmp` before the `batch` target, so recovery files and temporaries stay inside the case workspace instead of `~/.comsol` and are removed with it. Node jobs request `--signal=B:TERM@120`; the node worker forwards TERM to each running case step (`scancel --signal=TERM JOB.STEP`), whose trap cleans its scratch before the time limit; a forced cancellation or an out-of-memory kill still leaves scratch behind. Each case worker therefore removes stale `vp2-generation-JOBID.*` directories of the same user on its node before it creates its own, but only when the directory carries the worker workspace marker and its job is no longer in the user's queue. Each removal is logged to the job's error log.

</details>

<details>
<summary><strong>Timing fields</strong></summary>

| Field | Boundary |
| --- | --- |
| `comsol_stationary_airflow_seconds`, `comsol_transient_drying_seconds` | COMSOL's top-level `Solution time` per solver block |
| `comsol_scientific_solver_seconds` | Sum of the required solver-phase records |
| `comsol_process_seconds` | Wall time of the COMSOL process wrapper |
| `export_conversion_seconds` | Export validation and conversion |
| `publication_seconds`, validation and I/O counters | Publication and validation work |
| `complete_execution_s` | Complete case execution, without queue time |

Missing or ambiguous solver records stay unavailable with diagnostics. Startup, model loading and meshing have no separate clocks; queue wait and the Generation-to-Dataset duration are not measured by the case timer. Physical simulation hours are unrelated to wall-clock time.

</details>
