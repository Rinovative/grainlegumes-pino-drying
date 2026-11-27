# Generation

> **Normal entry point:** a Dataset plan run ([Datasets](04_datasets.md#normal-workflow)) resolves its `generation` block into one Source campaign.
> This guide explains what runs inside it, how cases fail and are replaced, and what a Source stores.
> Physics and sampling: [Physical model](02_physical_model.md). Learning data: [Datasets](04_datasets.md).
> Notebook: [`01_eda_generation_inputs`](../notebooks/01_eda_generation_inputs.ipynb) shows the prepared inputs, design coverage and failures by parameter.

[Documentation index](README.md)

## Campaign driver

A Dataset plan run is the campaign entry: it resolves the plan's `generation` block into one physical Source campaign and runs it through the driver and node jobs below. Sources carry no Train, validation, test or OOD roles. `./scripts/generation run CONFIG` starts the same driver for a campaign YAML (`schema_kind: generation_source_campaign`) without a Dataset; the repository ships no such file. The resolved plan records the overrides and the effective values.

```mermaid
flowchart LR
  A["Login node: run or plan"] --> B["Freeze source"]
  B --> C["Driver job"]
  C --> D["Node jobs"]
  D --> E["Case steps"]
  E --> F["Slurm accounting"]
  F --> G{"Decision"}
  G -->|successful| C
  G -->|failed| H["Replacement"]
  H --> C
  G -.->|"unclassified"| I["fail-case"]
  I --> G
```

On the login node `run` freezes the source and submits the controller as its own CPU Slurm job (2 CPUs, 8 GB, `--driver-partition` `standard` or `long`, `--driver-time` default 06:00:00) that runs from the frozen tree, then prints the submission. `--dry-run` prints the resolved plan and checks execution readiness (node launcher, native imports and the COMSOL module) without starting work or freezing the source.

<details>
<summary>Printed submission</summary>

```json
{
  "job_id": "4711",
  "log_path": "RUNTIME/logs/generation/generation-transient_drying-4711.log",
  "status": "SUBMITTED",
  "tree_id": "COMMIT12-FINGERPRINT12"
}
```

</details>

Options of `run`: `--materials`, `--duration-hours`, `--revision`, `--max-admission-cases`, `--node-jobs`, `--cores-per-case`, `--case-timeout-seconds`, `--no-retry`, `--driver-partition`, `--driver-time`, `--continue-with-current-code`, `--dry-run`.

<details>
<summary>Options</summary>

| Option | Effect | Default when omitted |
| --- | --- | --- |
| `CONFIG` | Campaign YAML | required, no default |
| `--materials` | Material families (a subset of the campaign's; their design sizes stay as authored) | none: the campaign config's `materials` apply |
| `--duration-hours` | Physical horizon; regular output intervals must stay exact | none: the `time` horizon of `configs/01_generation/physical_model/common.yaml` applies |
| `--revision LABEL` | Source campaigns only: names the Source `SOURCE_NAME__rev-LABEL` beside the existing one when its science changed ([Revision rule](07_reproducibility_and_validation.md#revision-rule)) | none: no revision (unless the campaign config authors `revision`) |
| `--max-admission-cases` | Distinct fresh solver candidates, recorded and fixed; input preparation still covers the full design ([Replacements](#replacements)) | none: a campaign without a plan records `run.max_admission_cases` of `execution.yaml` (4); a Dataset plan run derives it from the plan rates; a value below 1 is refused |
| `--node-jobs` | Node-job slots the run may use (1 to 4; a per-user cap on the shared `standard` nodes); with `--cores-per-case` k it caps the concurrent cases at slots × floor(`site.cores_per_node`/k) | `run.node_jobs` of `execution.yaml` (4) |
| `--cores-per-case` | CPUs per case (k) | `run.cores_per_case` of `execution.yaml` (4) |
| `--case-timeout-seconds` | Solver runtime limit | `run.timeout_seconds` of `execution.yaml` (3600) |
| `--no-retry` | No licence retries (neither the node wait nor the retry within an allocation) | off: `rules.license_wait.enabled` of `execution.yaml` (`true`) applies |
| `--driver-partition` | Partition of the driver job (`standard`, `long`) | `standard` |
| `--driver-time` | Time limit `[D-]HH:MM:SS` of the driver job | `06:00:00` |
| `--continue-with-current-code` | Continue an existing campaign with the current code instead of its bound source tree ([Source snapshots and frozen trees](07_reproducibility_and_validation.md#source-snapshots-and-frozen-trees)); refused with `--dry-run` | off: an existing campaign continues in its bound source tree |
| `--dry-run` | Print the plan and check readiness; start nothing | off: the run starts |

</details>

<details>
<summary><strong>Driver and node jobs</strong></summary>

- **Node-job slots:** one Slurm job per slot (`gen-seat-1` to `gen-seat-4`, Slurm singleton, at most `run.node_jobs` across all Generation runs of the user). Slurm picks a `standard` node with at least `site.cores_per_node` cores; node jobs are not exclusive, so the remaining CPUs of a larger node stay available to other jobs.
- **Node worker:** a generic worker inside each node job starts the cases as job steps of `cores_per_case` cores each, `node_job.cores_per_node // run.cores_per_case` at a time (32 // 4 = 8 with the shipped `execution.yaml`), and refills a free slot as soon as a step ends.
- **Dispatch:** the driver dispatches each case as a ticket into a node job and records the dispatch in `_record/execution/run.json` before the ticket exists. It reconciles every dispatch from the step's own Slurm accounting (state, exit code, times, node), so an out-of-memory or failed step is attributed to its case and a step that never started is dispatched again without consuming an attempt.
- **End of a node job:** when no case remains, the driver ends each node job with a stop ticket; an orphaned node job ends after 15 idle minutes.
- **Driver lifetime:** it runs until the campaign is complete, staged at its admission cap, or stopped by an error; its log is the `log_path`. Node jobs and `run.json` survive the driver, so rerunning the same command adopts the open node jobs and continues the campaign after an error or after the driver's time limit.
- **Monitoring and cancel:** `status` (one line per node job) and `cancel RUN_ID [--force]` run on the login node at any time; `--force` is off by default (graceful cancellation). A cancellation is a persisted desired state that every later driver call repeats until all node jobs have ended.
- **Core count:** `run.cores_per_case` is 4. It was chosen on 2026-10-28 from the measured node throughput per core count (k = 2 to 32): about 15 successful cases per node-hour at k = 4 against about 11 at k = 8.

Case states (names from `src/generation/generation_campaign_status.py` L45–53 and L74–80, dispatch statuses from `generation_campaign.py` L1075 and L1142):

```mermaid
stateDiagram-v2
  [*] --> never_started
  never_started --> admission_waiting: admitted
  admission_waiting --> scheduler_pending: dispatched
  scheduler_pending --> running: step started
  scheduler_pending --> scheduler_pending: step never started, dispatched again
  running --> successful
  running --> failed
  running --> license_blocked: licence window exhausted
  license_blocked --> scheduler_pending: redispatch after backoff
  failed --> never_started: replacement takes the slot
  successful --> [*]
  note right of failed
    reason_code: stalled, timed_out,
    solver_error, negative_water_state
    or unclassified
  end note
  note right of license_blocked
    at the wait cap the case stays license_blocked
    with license_wait_exhausted set, and the
    campaign settles as license_wait_exhausted
  end note
```

`unclassified` stays failed until `fail-case` declares it. `license_wait_exhausted` settles the campaign, not a single case.

</details>

<details>
<summary><strong>Execution settings, timeout and stall rule</strong></summary>

`configs/01_generation/execution.yaml` sets:

| Setting | Value | Meaning |
| --- | --- | --- |
| Case timeout | 3600 s | The orderly stop starts 300 s earlier (after 55 minutes); each case has an outer time limit of the timeout plus 600 s for publication |
| Stall rule `rules.stall_stop` | `minimum_simulated_seconds` 3600 s, `check_after_seconds` 900 s | Stops a transient case whose COMSOL simulated time is still below the minimum after 900 s of solving |
| `site.cores_per_node` | 32 | CPUs of one node job, a whole 32-core `standard` node or most of a 36-core one. The user's COMSOL batch seat is held as long as any case runs, whatever the number of parallel cases (see [Retention and licence](#retention-and-licence)), so wider node jobs shorten the time the seat is held. A whole free node can wait in the queue (more than a day on 12 November 2026); a pending job holds no licence, so submit large runs when whole nodes are idle (`sinfo -p standard -N`) |
| `run.cores_per_case` k | 4 | Eight cases run at a time per node job |
| `run.node_jobs` | 4, at most 4 | Per-user cap across all Generation runs, 128 cores in total; the batch licence counts once per user and does not limit it (see [Retention and licence](#retention-and-licence)) |
| `node_job.memory_per_case` | 20G, provisional | The largest observed transient case peak was 15.8 GB |
| `node_job.node_job_wall_time` | see file | Wall limit of a node job |
| Derived `concurrent_cases_per_node` | floor(`site.cores_per_node`/k) | Computed by the loader |
| Derived `node_job_memory` | floor(`site.cores_per_node`/k) × `memory_per_case` + `node_memory_overhead` | A configuration that would exceed `node_memory_limit` is rejected instead of lowering the concurrency; at 32 cores, k = 4 needs 168G (the 32-core `standard` nodes have 256 GB) and k = 2 needs 328G |

`--mem`, `--dependency`, `--comment`, `--exclude`, GPU and node-selection options are owned by the pipeline and rejected in `scheduler_options`. Without `stall_stop` only the timeout applies. The admission cap counts candidates, not allocations: continuations can request further allocations.

**Timeout.** In the core-scaling measurement of 28 October 2026 the longest success at k = 4 took 29.8 minutes, the same case took 20.0 and 29.8 minutes in its two replicates, and no case timed out. The generous limit costs time only for cases that progress but never finish and keeps slowly converging cases in the data.

**Stall rule.**

- Transient cases only (the profile with a time horizon). A stationary steady solve never reports a simulated time, so a steady case builds no stall rule even though the shared execution file sets one, records `stall_progress` as null, and only the timeout bounds it.
- The stop is orderly, escalates after the same reserve and is recorded as a solver failure with the typed reason `stalled`, not as a timeout. Detail, for example: `stalled: 0.98 h of 168 h simulated after 900 s solving (threshold: 1 h simulated after 900 s)`.
- An unknown simulated time never triggers it; the timeout stays the backstop for slow cases that still progress.
- The 900 s stays deliberately short so that cases that never converge stop early. In the measurement of 28 October 2026 the stuck cases stood at 0 to 63 s simulated after 15.6 minutes at k = 2 to 8; at k = 4 it stopped one case.
- A case stuck at its start (2026-10-26: 54 s simulated in 45 minutes) is stopped after 15 instead of 55 minutes and frees its COMSOL licence. Successful cases had passed 1 h simulated well before 10 minutes, one of them only after a slow first 5 minutes.
- A stall is not always reproducible (one case stalled in 1 of its 10 measured runs); a stalled case is replaced, not retried.

**Calibration.** Every case records in `runtime/timing.json` (and its evidence copy) `stall_progress`: the rule's `minimum_simulated_seconds` and `reached_after_seconds`, the solving time, on the clock of `check_after_seconds`, at which COMSOL's simulated time was first seen at or above that minimum. It is null when it was not seen (for example because the case stalled or its solver ended before stop control started), without a stall rule and for every steady case. A separate uncached reader of the solver output records it after each poll, so the value is accurate to the poll interval and never changes a stop decision. The timing reader exposes both values with the component availability `stall_progress`. After the first production campaign, `check_after_seconds` is set from the largest `reached_after_seconds` of the successful cases.

</details>

## Input-only generation and status

`inputs` prepares selected physical inputs of a campaign YAML without COMSOL through the Slurm input path (a Dataset plan run prepares its own inputs first). It needs a batch and a case selection; actual execution freezes the source like `run`, with or without uncommitted changes. `status` reports solver campaign state.

```bash
./scripts/generation inputs CAMPAIGN_YAML \
  --only-batch transient_drying__lentil__train \
  --case-start 1 --case-count 10 --dry-run
```

<details>
<summary><strong>Selection flags and inspection</strong></summary>

| Option | Effect | Default when omitted |
| --- | --- | --- |
| `CONFIG` | Campaign YAML | required, no default |
| `--only-batch NAME` | Prepare one batch | none: all batches |
| `--only-regime KIND` | Filter batches by design kind (`train`, `validation`, `test` or `parameter_ood`) | none: all design kinds |
| `--case-count N` | Number of cases per selected batch | none: all cases of each selected batch |
| `--case-start N` | First case of the selection; only together with `--case-count` | none: the first configured member of the batch |
| `--dry-run` | Resolve and report the selection without preparing inputs | off: inputs are prepared |

`--storage-root` and `--git-commit` are supplied by the controller and refused as options. Ten cases from a fifty-point design keep that design's seeds and positions. Inspect prepared inputs in `notebooks/01_eda_generation_inputs.ipynb`.

</details>

## Counts and case identity

Case IDs are material-local (`case_0001`, …) and follow candidate positions, not completion order. A retry keeps its ID and appends attempt evidence; an explicit replacement receives a new ID and keeps its parent design and seeds. A Source seals its selected pool when its success target is met; a short Source stays incomplete.

<details>
<summary><strong>Count definitions</strong></summary>

| Count | Meaning |
| --- | --- |
| Original design size | Points taken from the start of one fixed design (material and design kind) |
| Prepared inputs | Cases with generated and admitted physical inputs |
| Fresh admitted candidates | Distinct candidates charged to the admission limit |
| Attempts | Recorded execution and recovery attempts; can exceed the candidates |
| Successful candidates | Validated physical output with content hashes and a successful terminal decision |
| Source selected pool | Sealed successful cases in original-design order, replacements after primaries |
| Dataset selected count | Successful cases chosen by the Dataset |
| Training count | Dataset members assigned to Train |

</details>

## Failures and continuation

Failures end in a retry, a replacement or a stop that names its next step.

| # | Failure | Handling |
| --- | --- | --- |
| 1 | Spatial inputs: texture violates the mean-porosity support | Retry a complete stochastic realization with up to 32 deterministic seed plans; the packing scatter seed stays fixed. Structural and coupling failures stop at once |
| 2 | Schedules: degenerate or infeasible | Retry with up to 32 seed plans; values are never clipped; exhaustion raises an error. Both retries keep the candidate and its design coordinates |
| 3 | Licence capacity | Not a case failure: retried inside the allocation (300 s window, 5 s between attempts), then `wait.json` and redispatch after 60 s doubling to at most 300 s; the cumulative wait is capped at 3600 s, beyond it the campaign settles as `license_wait_exhausted` |
| 4 | Solve the runtime stopped itself; transient conversion that finds a negative water state | Decided automatically with the typed reason `stalled`, `timed_out`, `solver_error` or `negative_water_state`; the replacement is planned once every original candidate of the batch is terminal |
| 5 | Other export, conversion and validation failures, unclassified solver failures | No automatic retry: the driver stops with the `fail-case` instruction. Inspect the evidence and declare the case failed; its replacement is solved again with COMSOL |
| 6 | Controller or publication errors | Automatic feeding stops; rerun the same campaign after inspection. The rerun solves the case again with COMSOL; nothing is reused from the retained output |

<details>
<summary><strong>Licence wait and typed solver reasons</strong></summary>

**Licence capacity (class 3).**

- The worker retries the checkout inside its running allocation for a fixed 300 s window (5 s between attempts). When the window is exhausted it records a per-case `wait.json` and ends the case; the controller redispatches it after a backoff of 60 s doubling to at most 300 s.
- The cumulative wait is counted per case and capped by `rules.license_wait.maximum_wait_seconds` (3600 s); beyond it the campaign settles as `license_wait_exhausted`.
- Two keys are configurable, `rules.license_wait.enabled` (`--no-retry` sets it to false) and `maximum_wait_seconds`; the window and backoff timing are fixed constants of `generation_runtime_license.py`.
- With the protection off, a capacity failure is an ordinary solver failure: the attempt is recorded as `failed`, with no `wait.json`, no allocation window and no redispatch.

**Typed reasons (class 4).**

- `stalled` (the stall rule), `timed_out` (the case timeout) or `solver_error` (COMSOL exited non-zero, except by a signal, for example a memory-limit kill, which stays unclassified and is declared by hand). An attempt that the runtime did not classify reads `unclassified`.
- The attempt stores it as its `reason_code` with a human `detail` that names the solver progress of the task and the solving time, and keeps the exception type in its evidence:
  - transient: simulated hours of the horizon, `solver_error: COMSOL exited with status 1 at 0.98 h of 168 h simulated after 120 s solving`;
  - steady: the last stationary nonlinear iteration, `solver_error: COMSOL exited with status 1 at nonlinear iteration 37 after 120 s solving`;
  - `no progress reported` only when the solver output holds none (a transient case that ends during its initial stationary airflow solve also names that iteration).
- `negative_water_state` (transient only): a grain water state `w_surf` or `w_int` is below zero in any regular state, in the exact stop state or in the post-horizon state the conversion otherwise excludes. The rule is strict, without tolerance, and values are never clipped: the conversion fails before any HDF5 is written, so the case is never published. The `validation` attempt (stage `conversion`, not retryable) keeps the numbers in `evidence.negative_water_state`, because compact retention deletes the exports: per compartment the `unit` (kg/m^3), the `minimum`, the `negative_value_count` over all stored states and the `negative_state_count`, plus `first_negative_time_h`, `last_negative_time_h`, `minimum_time_h`, `stored_state_count` (regular and exact stop states only) and `post_horizon_state_checked`; the counts, minima and times include the post-horizon state when one was checked. Its `detail` is a summary rounded to 6 significant digits that names a checked post-horizon state separately; the exact values are in the attempt evidence. For example `negative_water_state: grain water states go below zero (w_surf minimum -0.5 kg/m^3, 1 negative cell values in 1 states; w_int minimum 17.2 kg/m^3, 0 negative cell values in 0 states) from t=172.289 h to t=172.289 h, lowest at t=172.289 h, over 169 stored states and the post-horizon state at t=172.289 h; the case is neither clipped nor published.`
- Once the step of the case's latest dispatch has terminal Slurm accounting and no dispatch of the case is active, the next feed records the failed decision with that reason and detail. That feed returns without draining a node job; once every original candidate of the batch is terminal, a following feed plans the replacement, unless the batch's minimum success rate has stopped it (see [Replacements](#replacements)).

</details>

Record an inspected failure:

```bash
./scripts/generation fail-case "$CAMPAIGN_RUN_ID" MATERIAL case_0001 \
  --reason-code nonconvergence --detail "inspected terminal solver evidence"
```

`fail-case` takes `RUN_ID`, `MATERIAL`, `CASE_ID`, `--reason-code` and `--detail`, all required without a default. It accepts original and replacement candidates and shares one decision path with the automatic decision. Terminal decisions are immutable: repeating the same decision is idempotent, a conflicting one fails.

<details>
<summary><strong>Decision rules, status and driver reconciliation</strong></summary>

- A replacement is checked against its own one-case batch, not its parent batch.
- A validly published case whose step ended without writing its success decision, for example after a scratch-cleanup error, receives it from the next scheduled feed; a metadata-only feed asks for that feed instead of hashing the case HDF5.
- Resume uses the frozen campaign, validated artifacts and exact scheduler evidence; missing scheduler evidence blocks duplicate admission.
- `status` reports each case's recorded `terminal_decision` (`successful`, `failed` or `null`).
- While no job is pending or running, the driver (`run`, and the Dataset command that wraps it) keeps reconciling with `resume-campaign` before it acts on a failure, because a status read right after the jobs ended can predate their evidence on the driver's node (shared-filesystem caching) and their reconciliation. `resume-campaign` prints one JSON summary line with the state, a compute request with its `continuation_reason`, the submission-record counts and the dispatches this resume reclassified as not started; the full manifest stays in the campaign's execution record `_record/execution/run.json`.
- The driver stops only if the condition has held for two minutes:
  - a failed case with `terminal_decision` `null`: the message names the cases and the `fail-case` route; rerun the same command after recording them;
  - a success deficit that no planned replacement covers because a stopped batch admits none: the message names the batch with k/n, its bound and s, and the ways out under [Replacements](#replacements).
- A recorded failure whose slot is covered by a replacement never stops the run.

</details>

## Replacements

Once every original candidate of the material, over all its design kinds, is terminal, each terminal failure takes the next unused point of its own design, and a failed replacement frees its target slot for the next one. The material's minimum success rate $s$ bounds this. With $k$ successful and $n$ decided candidates of the material (all design kinds, originals and replacements alike), the one-sided 95 % Clopper–Pearson upper bound of the success rate is

$$U(k,n)=\mathrm{Beta}^{-1}(0.95;\,k+1,\,n-k)\quad(k<n),\qquad U(n,n)=1 .$$

Once $n\ge1$ and $U(k,n)<s$, every batch of the material is stopped: it admits no fresh candidate of its own, neither original nor replacement, while its running cases finish and other materials continue. Implemented as `beta.ppf(0.95, k + 1, n - k)` in `src/generation/cases/generation_cases_selection.py` (L38 and L89–97).

- **Dataset plans** set $s$ as `generation.minimum_success_rate` ([Datasets](04_datasets.md)). Campaign creation records it once per material as amendment 1 (`registered_by: plan`); a rerun must repeat it exactly and a different value is refused. $s$ changes neither the Source name nor the seeds.
- **Campaigns without a plan** use $s=1$, so their first recorded failure stops the material until $s$ is lowered.
- **Blocked campaign:** when every admitted candidate is decided and a stopped batch is below its count, the campaign ends `blocked`; `status` and the driver name the batch with k/n, the bound and $s$, and the two ways out: fix the cause and start a new Source (`--revision LABEL`), or lower $s$ and rerun the same command.

```bash
./scripts/generation amend-replacements "$CAMPAIGN_RUN_ID" --material MATERIAL \
  --minimum-success-rate 0.8 --reason "first failure inspected; accept a lower success rate"
```

`RUN_ID`, `--material`, `--minimum-success-rate` and `--reason` are all required; none has a default. `--minimum-success-rate` must lie at or below the material's current $s$; repeating the current value is idempotent, a higher one is refused. Like `fail-case`, the command reads the live checkout, freezes no source and starts no job.

<details>
<summary><strong>Replacement mechanics, amendments and admission cap</strong></summary>

- **Points:** a batch of N points holds points 0 to N − 1, and its replacements take points N, N + 1, … in `candidate_position` order. They are prefix-stable and every per-case seed still derives from the design seed and the point index. Each replacement keeps its target slot and `parent_failed_case_id`. Originals come before replacements in the selected pool, and every failure stays in the Source as a candidate. Replacements are fresh admissions. A replacement's identity binds its parent batch science, design point, seeds, failed case and slot, but not the code fingerprint or the plan bytes, so it stays stable under code changes like a primary's.
- **Counts:** $k$ and $n$ count the batch's material over all its design kinds; undecided candidates, licence waits and cases without a decision do not count.
- **Evaluation:** every feed evaluates the stop from the recorded decisions, so a success recorded meanwhile lifts a stop at once and never blocks. A decision recorded while a replacement is planned that stops its batch or closes its deficit ends planning for that batch only, and the feed goes on.
- **One value per material:** the stop is decided per material, and every original batch of a material reads the same $s$, the rate of the material's latest amendment record in `materials/MATERIAL/_record/amendments/` ($s=1$ without one).
- **Amendment records:** a lowering publishes the material's next numbered amendment record (`amendment_NNNN.json` with rate, registrant `explicit` and reason) atomically under the controller lock and never rewrites an earlier one, so every design of the material reads either the old or the new $s$, never a mix; a sealed selected pool refuses it.
- **Admission cap:**
  - A Dataset plan run without `--max-admission-cases` records `max_admission_cases: null` and derives its cap at every feed: each original batch's count plus the failures it tolerates at its current $s$, so a lowered $s$ raises it; `status` shows it as `max_admission_cases=N (plan rates)`.
  - An explicit `--max-admission-cases` (or a `max_admission_cases` override) is recorded and stays as given, even when it equals that default; a campaign without a plan records the execution configuration's value.
  - A rerun may not lower the effective cap (a derived one at the current rates), also not when it switches between an explicit and a derived cap.
  - When an explicit cap holds back a candidate that a remaining success slot needs after a recorded failure, and every admitted candidate is decided, the campaign ends `blocked`; `status` and the driver name the cap: rerun the same command with a larger `--max-admission-cases`.

</details>

## Field tests

A field test reruns published cases of one Source with one declared parameter change each, so the effect of every sampled parameter can be judged against a tolerance (stage A screens one row at a time, stage B confirms removals, merges and the joint reduction at fresh bases). It is a static roster run on the unchanged node worker: no campaign, no lifecycle, no campaign replacement, and nothing below `01_generation/`.

```bash
FT="$STORAGE_ROOT/04_field_tests/TASK/TEST_ID"
./scripts/generation field-test plan "$FT" --source SOURCE_NAME          # stage A
./scripts/generation field-test plan "$FT" --from-stage-a STAGE_A_ID     # stage B
./scripts/generation field-test plan "$FT" --replace-failed-bases        # next roster version after failed gates
./scripts/generation field-test submit "$FT" --dry-run                   # print the sbatch commands
./scripts/generation field-test submit "$FT"                             # gates, later the variants of passed bases
./scripts/generation field-test resume "$FT"
./scripts/generation field-test status "$FT"
```

`submit` and `resume` freeze the source like `./scripts/generation run` and run from that frozen tree with its bindings (`GENERATION_GIT_COMMIT`, `GENERATION_SOURCE_SHA256`, the snapshot and the native venv), because their node tickets execute that tree; they submit node jobs and need the user's approval. `--dry-run`, `plan`, `status` and `cancel` freeze nothing; `--dry-run` writes nothing. `cancel` publishes the cancel marker of every running field-test node job.

Resources: the case shape of a step (`run.cores_per_case`, `run.timeout_seconds`, `node_job.memory_per_case`) is the base Source's frozen one, because it shapes a case's solve, and so is `site.comsol_executable`, because the solve reads it from the case's frozen configuration. Everything else about the node job comes from the current `execution.yaml` that the Source names (resolved with the Source's case values, so the same resolver validates the combination and derives the node-job memory): `site.cores_per_node`, `run.node_jobs`, `node_job.node_job_wall_time`, `node_job.scheduler_options`, `site.partition`, `site.comsol_module` and `site.python_executable`. A field test therefore follows the maintained node size without rewriting the published Source. Because the derived node-job memory uses the current `node_job.memory_per_case`, a current value other than the Source's is refused. `submit` and `resume` split their eligible members in roster order into at most `run.node_jobs` contiguous chunks of near-equal size (10 members on 4 seats: 3, 3, 2, 2), and each seat runs one node job over its chunk, so a job may hold more steps than its wall time fits. The node worker starts no step without enough time left, and `resume` redispatches the tickets that never started. `--dry-run` prints this chunking and these resources.

<details>
<summary><strong>Roster, bases, references, record, dispositions, gate, resume and licence</strong></summary>

- **Boundaries (D′).** A roster version is frozen once written and bound by its digest; no member is ever changed or removed. A step regenerates its base's batch from the Source's frozen plan, applies the intervention after the base values and before derived values, and runs the unchanged `execute_prepared_case` on a workspace prepared from that explicit bundle. The campaign controller, the lifecycle, the licence module, the node worker and the batch runtime are unchanged; a step refuses to run when `GENERATION_CAMPAIGN_RUN_ID` is set.
- **Intervention hook.** `sample_case(..., intervention=...)` accepts only sampled interval, integer and simplex primitives. Each value must lie in its support and, for a conditioned primitive, in its conditional support at the case's other values; derived values are recomputed and nothing else may change. An unknown name is refused, an out-of-support value is `infeasible`. `feasible_range` returns the conditional range of one primitive at a background.
- **Bases.** Lentil, chickpea and every material with Train cases take bases only from their original first-attempt Train successes without support or schedule retry (transient ones with a reached target), ordered per material by maximin distance over the keep rows. A material without Train cases, such as kidney bean in the demo (material OOD), takes its bases from its first-attempt successes of any design kind: bases are simulation inputs and never select a model. Each base records its `design_kind` in the roster. Every material of the Source design (its batches) must supply its full quota of admitted bases, or planning refuses the roster; a material whose every case is excluded never silently drops out.
- **References.** `R0` sets every tested row to its nominal. Rows that act through a partner pair with a reference that moves the partner to an active value: `R_sin` and `R_gauss` (largest feasible amplitude magnitude), `R_texture` (upper initial-moisture amplitude), `R_event` (event weight at its simplex upper bound) for the event duration and width, and `R_jitter` (upper orientation jitter) for orientation smoothing. The perturbation shape rows are screened against `R0` (nominal amplitude) and re-checked against `R_perturb` (upper amplitude); those re-check members end in `-at-r_perturb`.
- **Merges.** The only prespecified merge formula is the anisotropy pair: the permeability map reads strength and max_ratio only through strength·(max_ratio − 1), so stage B moves both along that constant product (level `merge`, row `merge:permeability.anisotropy.max_ratio+permeability.anisotropy.strength`). A displacement's intervention is its `R0` reference's intervention with exactly these two parameters moved. Merges are prespecified (E1): stage B adds the displacements of every prespecified merge whose two rows stage A tested, whatever their screening verdicts, and refuses stage-A verdicts that carry a merge verdict. The fine-texture triple (`fine_len_rel` with `fine_ani_x`, `fine_ani_y`, which enter only as their products) stays an exact E1 merge recorded from the code, without runs.
- **Base replacement.** A base whose gate seals a failure is never repaired in place: `plan --replace-failed-bases` writes the next roster version (`roster.vN.json`) with every earlier member unchanged, the failed base's rows, merges and joint test rebuilt on the material's next unused base in the same order, and the replacement recorded in `replacements`; `supersedes` binds it to the previous version's digest. `roster_sha256` is the canonical digest of the whole roster payload except itself, so it binds `supersedes`, `replacements` and `derived_from` as well as the members. The effective roster is the newest verified version: `field_tests.roster.load_roster` checks the whole chain (regular files `roster.json`, `roster.v2.json`, … without a gap, no other `roster.v*.json` and no symbolic link), that each version keeps every earlier member and replacement unchanged, and that its appended members are exactly its new replacement bases, each mirroring the failed base's planned members (member-id suffixes, roles, links, rows, levels, reference kinds and the joint values). Every command, the Analysis verdicts and stage B read the roster only through it. A verdict drops a replaced base only through `field_tests.record.effective_members`, which requires the failed gate's seal to be a failure with the recorded `failed_disposition`. `verdicts.json` names the `roster_version` and `roster_sha256` it judged, and stage B refuses verdicts of any other version.
- **Record.** `STORAGE_ROOT/04_field_tests/TASK/TEST_ID/` holds `roster.json` (and later versions), `planning.json`, `members/MEMBER/{seal.json, case.h5, attempts/N/{submission.json, outcome.json, case.h5, bundle/, runtime/}}`, `nodes/TOKEN/` (tickets, receipts, `job.json` and a resume's `job.recovered.json` of one node job; its tickets end with a stop ticket, so the node job ends after its last step), `analysis/` and `locks/`. The field-test area, the task and test directories and every record component below them must not be symbolic links (checked with lstat before the containment check), so no record can be redirected out of the field test. A step checks every component it writes (`bundle`, `base_regenerated`, `runtime`, `case.h5`) the same way; a refused one ends the attempt as `binding_refused` (`record_path`) before anything is written through it. The runtime files are the exception: a refused or failed copy into `runtime/` is recorded in the outcome's `notes` as missing evidence, the solve's own result or failure stands, and the scratch cleanup always runs. `cancel` publishes the marker of every `nodes/TOKEN` directory whose entry and control folder pass the check, then reports each refused entry and exits non-zero. Records are written atomically and write-once; one member lock covers an attempt from the gate check to the seal, and a sealed member never runs again. A step keeps its output in `attempts/N/case.h5` and commits the outcome with its `case_h5_sha256` first; settling then hard-links that file into `members/MEMBER/case.h5`, idempotently and digest-checked, before the seal, so a member `case.h5` never exists without a committed success. A crash before the outcome commit stays an attempt that `resume` records as `infrastructure_stop`; a committed outcome without a seal is settled by the next `resume`. Stage A's analysis writes `analysis/verdicts.json`, which stage B reads.
- **Dispositions.** `succeeded`, `solver_failure`, `infrastructure_stop`, `gate_blocked`, `binding_refused` and `infeasible`. `succeeded`, `gate_blocked`, `binding_refused` and `infeasible` seal at once; a reproduced `solver_failure` seals at its second occurrence; otherwise a member seals at the roster's `max_attempts`. A step that never started (`not_started`) is not an attempt. Only the sampling hook's typed infeasible refusal makes a member `infeasible`; any other generator `ValueError` or `RuntimeError` seals `binding_refused` (`bundle_generation`) for every role, which the verdicts count as broken.
- **Gate.** Each base has one gate member that reruns the unchanged base. numpy's SIMD loops and OpenBLAS change generated maps and schedules in their last digits between ICE's CPU types ([Reproducibility](07_reproducibility_and_validation.md), producer identity), so the gate compares the regenerated inputs with the published `inputs.h5` per array with the last-digit tolerance `rtol = 1e-12`: each field map within `max|a − b| ≤ rtol · max|b|`, each sampled scalar within `|a − b| ≤ rtol · |b|` (a zero stays exactly zero), and each schedule column within `rtol` times that column's largest magnitude. It records the observed maxima in `input_comparison` and fails on a larger difference. The output must then match the published `case.h5` structurally: the same numeric datasets with the same shapes, no non-finite value where the published one is finite, the same reached flag, and the stored hour axis within the published file's time classification tolerance. The transient solve does not reproduce the exact stop time bitwise (on ICE the gates measured differences up to about 1e-4 h), so a stop-time drift passes when these checks pass. A drift that changes the layout still diverges: a stop that moves from a stored hour to after it adds the `exact_stop` datasets and a `global/values` row, and one that passes the next stored hour changes the hour axis. Any structural divergence seals `binding_refused` (`gate_outputs_differ`, listed in `gate.output_divergences`), so the base is replaced. The largest difference per dataset is recorded in `gate.output_max_abs_diff` as evidence only; the analysis measures the gate's `case.h5` against the base's with the endpoint code as the base's replicate envelope, the duration from both exact stop times included when both reached the target ([Field tests](06_evaluation.md#field-tests)). Variants of a base run only after its gate succeeded; a failed gate seals its variants `gate_blocked` without a run.
- **Binding.** Every submission (`submission.json`) records the roster version, the roster digest and the canonical digest of the member's roster entry. The step loads the effective roster and seals `binding_refused` (`roster_binding`) unless its member's entry still has that digest; a newer version that keeps the member unchanged is fine. Every attempt checks the base digests, unchanged seeds and design row, and that only declared and derived primitive values differ from the base (exactly: they are sampled scalars). A variant also regenerates its base on the same node: arrays that differ bitwise between the two bundles are the ones the intervention touched (`input_comparison.touched`); every other array must match the published base within the gate tolerance. Any mismatch seals `binding_refused`; an I/O error while reading the base binding or the inputs is a retryable `infrastructure_stop` (`base_io`, `input_io`).
- **CPU model.** Every seal records `cpu_model` as evidence only. Node jobs never choose a CPU type, and a variant may pair with a reference or gate that ran on another CPU type; nothing is blocked for a CPU mismatch.
- **Resume.** `resume` reconciles each member's latest attempt with the queue: an active job is left alone, a job that ended after the step started without an outcome records `infrastructure_stop`, and one that ended before the step started counts as `not_started`. A submission whose `job.json` is missing or holds a null job id (sbatch failed or its answer was lost) is looked up by its token in the active queue and by the node worker record; a job found that way is recorded in `job.recovered.json` and treated like a recorded one. Without either trace it waits 300 s after submission and then counts as never started only when its ticket has no started receipt; a started receipt makes it an `infrastructure_stop`. Only then is a new attempt submitted. This also picks up the tickets of a seat's node job that ended before its worker could start them (wall time, cancel or a dead job).
- **Licence.** The in-allocation licence wait is off for field-test steps: a licence shortage ends the attempt as `infrastructure_stop` (`license_capacity`), which `resume` retries; it never counts as a solver failure.
- **Never a Source.** A field-test case changes one parameter away from its design point, so it is not a sample of the Source's design and must never enter a Source, a Dataset input or the EDA catalog; the separate storage area keeps every Source listing, digest and catalog byte-identical.

</details>

## Logs

Every Generation and Dataset job writes its log under `runtime/logs/`, submitted with `--open-mode=append`, so a rerun or requeue appends and never truncates a linked log. Once the campaign directory exists, the logs are also hard-linked under their own names into the campaign's `_record/logs/`: a link is the same file, so it grows while the job writes and survives a cleanup of `runtime/logs/`. Nothing is copied, moved or deleted.

<details>
<summary><strong>Log files, linking and log formats</strong></summary>

| Log | In `runtime/logs/` | Linked into the campaign |
| --- | --- | --- |
| Driver of `run` | `generation/generation-TASK-JOB.log` | as soon as input preparation has created the campaign, also when preparation or `submit-campaign` then fails |
| Dataset driver | `datasets/dataset-TASK-JOB.log` | as the Generation driver; never into the Dataset folder, which nothing changes after its atomic publication ([Datasets](04_datasets.md)) |
| Node job | `generation/node-JOB.out`, `generation/node-JOB.err` | when the node worker starts (the `.out` link grows with the file) and again when it ends, which links an `.err` that holds text by then; Slurm teardown lines written after the worker process exits, and the interpreter's own report of an uncaught `KeyboardInterrupt` or `SystemExit`, are written after the final link and stay in `runtime/logs/` |
| Helper step | `generation/python-OPERATION.XXXXXXXX.out` (and `.err` when it ran through `srun`) | when the step ends, or when the campaign becomes known for steps that ran before it existed; a successful step whose output is one JSON object (the result the driver receives) writes no file and prints only `OPERATION: JSON result returned to the driver (N bytes, sha256 …)` with its `state` to the driver log, any other output and every failed step keep the raw output in the file |

A job finds its own log from the same `--output`/`--error` template the pipeline gave its `sbatch` call, with `%j` replaced by `SLURM_JOB_ID`: `generation/node-%j.out` and `.err` for a node job, `AREA/NAME-%j.log` for a driver, where `NAME` is its `SLURM_JOB_NAME`; the code that builds those arguments owns the templates. Only an existing regular file is linked, and an `.err` file that is empty at that moment is not. After the link is created, its device and inode are compared with the file that was named; on a mismatch (the file was replaced meanwhile) the link is removed again with a warning. A link never creates the campaign folder and never fails a job: a missing folder, a symbolic link or a file-system error is one warning on stderr, and a different file under the same name in `_record/logs/` stays and is reported.

- **Driver log:** each stage starts with `[n/N] STAGE` and ends with `[n/N] STAGE done in DURATION`.
  - While cases run, a live board (counts, elapsed time, one line per node job and per case) appears at the first poll and at every state change; otherwise a running case gets one compact line every 60 s (`TIMESTAMP  MATERIAL/case_NNNN  simulated 56.8 h  step 4.176 h  Tfail 2  NLfail 2  updated 10 s ago`, with `no progress for N min` once its solver evidence stayed unchanged for over 10 minutes), and a finished case shows its moisture and target.
  - A case that changes its group first gets one line `TIMESTAMP  case MATERIAL/case_NNNN: OLD -> NEW (REASON)` (a case that leaves `failed` or `verifying` adds `; was REASON`; a failed case whose detail (reason, decision or replacement) changes gets `TIMESTAMP  case MATERIAL/case_NNNN: failed (NEW; was OLD)`), so a case that failed and then settled to done is traceable.
  - The board estimates no remaining time and shows no percentage, because a drying case ends at its target moisture, mostly long before the horizon; a board that cannot be rendered prints `status board unavailable: MESSAGE` once per distinct message and changes nothing else.
  - A failure ends with `FAILED: STAGE: MESSAGE` and, when a command continues the run, `Next: COMMAND`: the rerun of a started campaign (`status` for an authored campaign), and the `--dry-run` of the same config for a failed plan resolution or preflight of a run (none for an authored campaign or a `status`).
- **Node log:** one preflight report per node job (each case step repeats the checks and reports only a failing one), `CASE START …` and `CASE END MATERIAL case_NNNN STATUS at TIMESTAMP after DURATION` per case step (a failed step appends `: REASON`; a stall, which only a transient case can have, names the threshold and the simulated time reached, and any other runtime stop names the solver progress of its task as above), the plain line `MATERIAL case_NNNN stopped: REASON` of a failed step in `node-JOB.err` (followed by a JSON failure record only when the runtime did not classify the stop as stalled, timed out or a solver error; Slurm adds its own `srun: error: … Exited with exit code` line), and `Node worker ended: …` at the end.
- **Progress:** a running case shows its simulated time past the start of its physical window (`simulated 1 h`), the solver step, the time and nonlinear failure counts and the age of that evidence; the solver reporter writes on every change it sees and at least every 300 s as a heartbeat when nothing changed, so the compact line may repeat values with a growing `updated N s ago`. `status` reports `physical_start_hours` and `physical_horizon_hours` (stop minus start) per case from the `time` section of its batch's persisted `MATERIAL/_record/REGIME/resolved_generation_config.json`; a running case always has that file.
- **Failed cases:** the board shows the typed reason and detail, then `replaced by case_NNNN` for the case that took the slot (from the replacement's `parent_failed_case_id` in `status`), `decision recorded`, or `no decision recorded`. A case whose step ended `COMPLETED` but whose result evidence the driver cannot see yet (`completed_without_valid_case_evidence`, typically the cross-node file-system lag) shows as `verifying` with `step ended, result not yet visible` for the driver's 120 s settle window, and as failed only afterwards; the header counts it as `verifying N` while there is one.

</details>

## Source storage

Sources live under `storage/01_generation/TASK/SOURCE_NAME/`. A campaign holds exactly three entries: the derived `cases.csv` view, `materials/` and `_record/`; machine records sit in the reserved `_record/` folders. The execution handle is `source~TASK~SOURCE_NAME`. Exact source bytes and frozen trees: [Source snapshots and frozen trees](07_reproducibility_and_validation.md#source-snapshots-and-frozen-trees).

<details>
<summary>Source tree</summary>

```text
01_generation/
  TASK/SOURCE_NAME/
    cases.csv                 # derived one-row-per-candidate view, written when the Source completes
    materials/
      MATERIAL/
        case_NNNN/
          inputs/             # case.json, inputs.h5 (spatial maps; transient: scalars and schedule), input_reference.json
          result/             # case.h5, provenance.json, _SUCCESS
          result/evidence/    # execution_provenance, processing_provenance, status, timing, solver.log
          _record/
            reservation.json  # the only reservation copy, published atomically with its case directory
            decision.json     # immutable terminal decision
            attempts/         # attempt_NNNN.json classified attempts; run_NNNN/ failure bundles
        _record/REGIME/       # batch_manifest, input_generation_manifest, resolved_generation_config, _SUCCESS
        _record/amendments/   # amendment_NNNN.json: the material's minimum success rate amendments
    _record/
      campaign_plan.json      # frozen plan with its seed_key
      selected_pool.json      # sealed success membership (absent until the target is met)
      logs/                   # hard links of the campaign's driver, node-job and helper logs (see Logs)
      execution/              # run.json, cancellations.json, epochs/, slurm/, interruptions/, license/, reconciliation/,
                              # reconciliation_failures/, node_jobs/
  .state/                     # the stage's only state; no task or campaign folder keeps any
    locks/                    # empty lock anchors, one file per lock (names below)
    staging/TASK~SOURCE_NAME/ # MATERIAL/reservation-staging/ and MATERIAL/REGIME/{publications,quarantine,input-transactions,cleanup_failures}/
                              # record-staging/: every immutable JSON record (plan, epoch, reservation, attempt, decision,
                              # amendment, selected pool) is written here, then hard-linked to its name
```

</details>

<details>
<summary><strong>Source name, seeds and identity</strong></summary>

The printed Source name is the catalogue selector. The name is computed from the original designs: per material in registry order its slug followed by its design sizes in design-kind order (`MATERIAL-TEST` for a material with a test design alone, `MATERIAL-TRAIN-VALIDATION-TEST[-PARAMETER_OOD]` for the ID designs), joined by `+`, then `__DOMAIN` (`ref-range`, or `ref-param-ext` when any material has a parameter-OOD design), `__sobol-seed-SEED` and for transient drying the physical horizon `__h-HOURS` only when `--duration-hours` makes it differ from the horizon configured in the common config, for example `lentil-1-1-1-1+chickpea-1-1-1-1+kidney-bean-2__ref-param-ext__sobol-seed-9200` and the same name with `__h-4`; a deviating horizon keeps the seeds but is another Source.

Internal scientific digests establish compatibility. A compatible repeated request reuses its Source; different science under the same name fails before writing, and the message names the fix ([Revision rule](07_reproducibility_and_validation.md#revision-rule)).

Each design's seed derives from the seed base, the material and the design kind only, and every per-case seed from the design seed and the point index, so point i of a design is the same physical case in every campaign that requests it ([Datasets](04_datasets.md#fixed-designs-and-case-roles); the designs themselves: [Sampling](02_physical_model.md#sampling)). The design is one scrambled Sobol sequence over all sampled block dimensions (recipe `scrambled_sobol_single_sequence`); each batch records its design seed, the dimension count, its block columns and a digest of its points. The campaign's `seed_key` frozen in `campaign_plan.json` (a digest of the task, the design sizes per material and design kind, the domain profile, the design recipe with its seed base and the campaign purpose) keys candidate reservations and the campaign identity. The name and the storage layout are not inputs, so renaming a Source keeps its random fields.

</details>

<details>
<summary><strong>Layout rules, locks and atomic publication</strong></summary>

Each case stores its COMSOL inputs as one HDF5 file `inputs/inputs.h5`: one float64 map per spatial column under `fields/` (column order in its `columns` attribute), and for transient drying the `scalars` (name, value, unit) and `schedule` tables. COMSOL gets `fields.csv`, `scalars.csv` and `schedule.csv` rendered from it into its work directory; each rendered file must match the sha256 and size that `case.json` records under `input_files` before COMSOL starts, and these CSV hashes, not the HDF5 bytes, are the case identity. The sha256 and size of `inputs.h5` itself are storage integrity in `input_generation_manifest.json`. Open `inputs.h5` in VS Code with the H5Web extension to view the maps and tables. A case directory with CSV files instead of `inputs.h5` predates this layout and is refused; regenerate its inputs. The stored `inputs.h5` is the authoritative input: campaigns, Datasets and resumes reuse it, and regenerating the same case is bit-identical only on the same CPU type ([Reproducibility](07_reproducibility_and_validation.md#source-snapshots-and-frozen-trees)).

Human files come first: a campaign holds exactly three entries, the derived `cases.csv` view, `materials/` (each material with its cases, each case with its `inputs/` and `result/`) and `_record/`. Machine records sit in the reserved `_record/` folders at campaign, material and case level (a material's `_record/` stays with it below `materials/MATERIAL/`), and the campaign's job logs in `_record/logs/`. A material holds only its case directories and its `_record/`: every reader recognizes `_record` explicitly and still rejects any other entry beside the cases, a symbolic link, a missing case number or an incomplete case. Controller and scheduler logs live under `runtime/logs/` with hard links in the campaign's `_record/logs/` (see [Logs](#logs)).

A lock file name joins its identity with `~`, which no task, Source, material or regime name contains, so names cannot collide: `TASK~SOURCE_NAME~controller.lock` and `TASK~SOURCE_NAME~submission.lock` per campaign, `TASK~SOURCE_NAME~MATERIAL~REGIME~inputs.lock` per input batch and `TASK~SOURCE_NAME~MATERIAL~REGIME~case_NNNN.lock` per case. A name that would exceed the 255-byte path-component limit is refused, never shortened. A deleted campaign leaves only empty anchors behind.

Operational records below `_record/execution/` are keyed by `MATERIAL-REGIME`, the regime being the design kind or a replacement batch's `KIND_replacement_NNNN` (for example `license/lentil-train/case_0001/`), which must be unique among the frozen batches of one Source.

Each atomic publication is one rename of a complete directory: a case appears with its `_record/reservation.json` (staged in `.state/staging/`), `inputs/` and `result/` are each renamed into the existing case, and an attempt bundle `run_NNNN/` is renamed inside `_record/attempts/`. Every immutable JSON record is written in `record-staging/` of its task and Source and then hard-linked to its name: the link never replaces an existing record (a second writer must hold identical bytes), a record's directory never holds anything but final names, and a writer that dies leaves its temporary only in staging; a publication whose staging and target are on different file systems fails before it writes. Locks, record, reservation, result and input staging and quarantine stay in the stage's single `01_generation/.state/` (staging by task and Source, then material and regime), never inside a campaign folder, so every rename and link stays on one file system. Inputs and results have separate recovery boundaries:

- **Inputs:** an input generation stages its new cases and batch metadata in its transaction, writes the journal `transaction.json` (manifest hash, new case IDs) before any final path changes, renames each new `inputs/` into its case, admits the staged metadata and only then publishes `input_generation_manifest.json`. A transaction without a journal is abandoned staging and is deleted; one with a journal is completed (missing renames, then the manifest) by the next input generation of that batch under its input lock.
- **Results:** a staged `result/` is renamed in only after an incomplete `result/` without `_SUCCESS` has been moved into the batch's quarantine in `.state/staging/`; a complete one is never replaced. Quarantined directories are kept and counted, never published.

</details>

<details>
<summary><strong>Stored records</strong></summary>

| Record | Content |
| --- | --- |
| `_record/campaign_plan.json` | Immutable effective science, design recipe, original design sizes per material and design kind, seeds and `seed_key`, per-batch design seeds and point digests, configuration content, physical mapping, producer identity, model reference and hash |
| `_record/execution/run.json` | Mutable ledger: node jobs (seat, token, allocation, lease, draining, node directory `_record/execution/node_jobs/TOKEN`), dispatch records (submission index, node job, ticket, step name, attempt, status) and the admitted source binding |
| `_record/execution/node_jobs/TOKEN/` | Node directory of one node job: `tickets/`, worker `receipts/`, `control/cancel`, `node.json`, `node_exit.json` |
| `_record/execution/cancellations.json` | Receipts of graceful and forced cancellations |
| `materials/MATERIAL/_record/REGIME/resolved_generation_config.json` | Batch document validated by input admission and batch publication; its `time` section gives the physical start and horizon `status` reports per case |
| `materials/MATERIAL/_record/amendments/` | Append-only minimum-success-rate amendments of the material, one atomic numbered record each, with rate, registrant (`plan` or `explicit`) and reason |
| `_record/execution/epochs/` | Ordered execution-resource and source admission history |
| `case_NNNN/_record/reservation.json` | Durable allocation; the case directory appears only with it |
| `_record/execution/reconciliation/`, `_record/execution/reconciliation_failures/` | Scheduler timestamp normalization and case classification evidence, per `MATERIAL-REGIME/case_NNNN/` |
| `_record/execution/slurm/` | Phase and counters per dispatch (`JOB-SUBMISSION_INDEX.json`, for example `7001-3.json`) for live status |
| `_record/execution/interruptions/` | Worker-interruption receipts per dispatch (`JOB-SUBMISSION_INDEX.json`) |
| `_record/execution/license/` | Per case below `MATERIAL-REGIME/case_NNNN/`: the licence wait `wait.json`, checkout windows `windows/DISPATCH.json` and status recoveries `status_recoveries/` |
| `01_generation/.state/` (stage level, outside every campaign) | Lock anchors in `locks/` and staging in `staging/TASK~SOURCE_NAME/` (immutable records, reservations, publications, input transactions, quarantine), named by task, Source, material and regime |
| `case_NNNN/inputs/` | Case definition, sampled and derived parameters, seeds, rejection evidence, input reference and hashes in `case.json`; the spatial, scalar and schedule inputs in `inputs.h5` |
| `case_NNNN/_record/attempts/attempt_NNNN.json`, `case_NNNN/_record/decision.json` | Append-only classified attempts (a solver attempt's `reason_code` is `stalled`, `timed_out`, `solver_error` or `unclassified`; a conversion that found a negative water state records `negative_water_state` with its undershoot numbers in `evidence.negative_water_state`) and immutable terminal decisions |
| `case_NNNN/_record/attempts/run_NNNN/` | Evidence bundle of one failed execution: receipt, retained payload and predecessor hashes |
| `result/evidence/status.json`, `timing.json` | Execution outcome and measured timing boundaries |
| `result/provenance.json` | Content inventory (paths relative to `result/`) and conversion provenance |
| `result/evidence/` provenance files | Execution environment and arguments, processing provenance |
| `result/case.h5`, `result/_SUCCESS` | Canonical physical output and hash-bound publication markers |
| `_record/selected_pool.json` | Sealed successful Source membership and artifact hashes |
| `_record/logs/` | Hard links of the campaign's job logs; the originals stay in `runtime/logs/` (see Logs) |
| `cases.csv` | Derived view, rewritten atomically and byte-identically whenever the campaign settles complete: one row per candidate in material and case order with `material`, `case_id`, `origin` (`original` or `replacement`), `outcome`, `reason_code`, `detail`, `duration_s`, for a transient campaign `simulated_end_h` and `final_bulk_moisture_wb` (a steady one has no time axis and shows neither), every parameter that the frozen plan's registry classifies as sampled as its own column (in registry order; no constants such as `T_flow_ref`, `p_ref`, `p_out`, `eps_min_global`, `eps_max_global`, no derived values), and `solver_log`, the campaign-relative path of the last failed attempt's retained COMSOL log (`stdout.log`); empty cells where a value does not apply (a failed attempt may have written no record, a parameter that is fixed in a material's registry is empty for that material). A successful row reads what admission requires of it, for both tasks: the `status.json` and `timing.json` of its evidence and, for a transient case, its canonical final bulk moisture; an unreadable one stops the rendering with an error that names the case and the file, and a successful case without its processed status is refused the same way when the campaign status is read |
| `source_snapshots/` (storage root, outside `01_generation/`) | Exact admitted source bytes shared by Generation, Datasets and ML |

Lock and staging records coordinate concurrent workers; inputs, decisions, outputs and snapshots are provenance.

</details>

## Retention and licence

`retention: compact` keeps grid, horizon, output fields and validation unchanged and limits stored execution payloads. COMSOL always runs with `-usebatchlic` (separate batch licence pool, never the GUI features of users at OST workstations).

<details>
<summary><strong>Retention, licence and cleanup</strong></summary>

**Retention.**

- Successful cases keep canonical HDF5, solver log, timing, execution and processing provenance and publication evidence; COMSOL exports and the solved `.mph` model are omitted. Full retention keeps both in `result/evidence/`.
- Failed attempts of every failure stage, conversion and publication included, keep status, stop, timing, provenance, the transient time-stepping history `runtime/solver_steps.csv` (step, simulated time and step size in seconds, solver counts, time-step and nonlinear failures) and the logs: a log above 1 MiB keeps a bounded head and tail copy plus the complete log as `NAME.log.gz`; the omitted exports, converted HDF5 and solved model are recorded. Full retention keeps them.
- Cleanup acts only on owned scratch after its publication and recovery checks. A failed attempt never carries a success marker.

**Licence.**

- COMSOL always runs with `-usebatchlic`: it uses the separate batch licence pool (COMSOLBATCH, CFDBATCH, HEATTRANSFERBATCH, two seats each for CFD and heat transfer) and never the GUI features COMSOL, CFD and HEATTRANSFER of users at OST workstations.
- Measured on 27 and 28 October 2026: all batch checkouts of one user count as one seat, across two node jobs and also across five hosts at once (probe job 873137, no licence error), so the licence does not limit the number of node jobs; the limit is the cluster (five `standard` nodes with 32 cores).
- Every user running COMSOL (batch or GUI) also holds one of the five COMSOLUSER seats, again once per user across hosts (the five hosts of probe 873137 held one), so Generation costs GUI users one COMSOLUSER seat while it runs; with four other users in the GUI this fills the pool.
- `-usebatchlic`, `-ckl` and `-keeplicenses` cannot be set through `extra_arguments`.

**Scratch.**

- COMSOL also runs with `-autosave off -recoverydir WORK/comsol/recovery -tmpdir WORK/comsol/tmp` before the `batch` target, so recovery files and temporaries stay inside the case workspace instead of `~/.comsol` and are removed with it.
- Node jobs request `--signal=B:TERM@120`; the node worker forwards TERM to each running case step (`scancel --signal=TERM JOB.STEP`), whose trap cleans its scratch before the time limit; a forced cancellation or an out-of-memory kill still leaves scratch behind.
- Each case worker therefore removes stale `vp2-generation-JOBID.*` directories of the same user on its node before it creates its own, but only when the directory carries the worker workspace marker and its job is no longer in the user's queue. Each removal is logged to the job's error log.

</details>

## Timing fields

Retained timing is the measured execution time of a case, without queue time; physical simulation hours are unrelated to wall-clock time.

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
