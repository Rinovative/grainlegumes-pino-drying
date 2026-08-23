# COMSOL phase timing probe

This temporary diagnostic runs exactly one transient case from an ordinary
`technical_runtime_smoke` campaign. Its purpose is to retain evidence about
whether COMSOL text can support phase-specific timing. It does not establish a
production timing method or a performance claim.

No real COMSOL execution was performed while implementing or testing this
repair. Focused tests use a fake executable and therefore cannot validate real
COMSOL log grammar, phase boundaries, or timing values. The earlier failed probe
left no usable timing evidence.

## Execution contract

Use the maintained transient Technical Smoke campaign directly:

```bash
./scripts/generation_workflow.sh timing-probe configs/generation/campaigns/transient_drying/technical_smoke.yaml
```

Foreground execution is the default. Managed background execution changes only
controller ownership:

```bash
./scripts/generation_workflow.sh timing-probe configs/generation/campaigns/transient_drying/technical_smoke.yaml --background
```

The campaign remains authoritative for the case, template, scientific inputs,
COMSOL executable, Slurm scheduler and partition, scheduler options, wall time,
modules, and `cores_per_case`. The shell requests one node, one task, and the
configured cores with `srun`; the normal Generation command passes the same
count to COMSOL with `-np`. The public probe runner also requires a numeric
`SLURM_JOB_ID` and an exact `SLURM_CPUS_PER_TASK`, so direct CLI or Python API
use outside an admitted allocation fails before input generation or COMSOL
inspection. COMSOL never runs on the login shell.

The probe has no alternate case builder or subprocess runner. It performs this
single path:

1. `generation_cases_input.generate_input_cases(..., case_count=1)` publishes
   exactly one canonical input case in probe-isolated storage.
2. `generation_runtime_batch.run_case` invokes ordinary workspace preparation.
   That preparation admits the canonical case and copies every declared input
   file, including `fields.csv`, into the COMSOL working-directory root.
3. The ordinary runtime owns command construction, licensing, process control,
   stdout and stderr, export collection, conversion, publication or failure
   attempt recording, and marked scratch cleanup.
4. An optional diagnostic observer adds one runtime-owned
   `-batchlog PATH -batchlogout` pair to that normal command and copies complete
   bounded logs before normal scratch cleanup.
5. The probe records compact normal publication, failure-attempt, or
   temporary-license wait evidence. Exactly one authoritative normal outcome is
   required before immutable bundle publication. The probe then validates the
   bundle and removes its isolated canonical and processed case storage. It
   never builds a Dataset package or leaves a normal case in production
   Generation storage.

The previous implementation bypassed step 2. It generated a case bundle under
an `inputs/` subdirectory and launched COMSOL directly from another directory,
so the model could not resolve `fields.csv` at its normal working-directory
location. The repair removes that parallel path instead of adding an ad-hoc file
copy.

A caller-supplied work root remains caller-owned. The normal runner removes only
its marked case workspace; the probe does not delete the work root or unrelated
content.

## Result and transfer

The remote command announces `PROBE_ID` before case execution. Once a normal
case reaches a retained success, failure-attempt, or temporary-license deferral
result, it prints:

```text
PROBE_CASE_STATE=successful|failed
PROBE_CASE_EXIT_CODE=0|nonzero
PROBE_CPU_BUNDLE=REMOTE_STORAGE/03_experiments/comsol_phase_timing_probe/PROBE_ID
```

The host workflow validates those fields, transfers only the exact diagnostic
bundle into an existing marked transfer staging directory, validates every file
and digest, and atomically publishes it at:

```text
LOCAL_STORAGE/03_experiments/comsol_phase_timing_probe/PROBE_ID
```

It then prints `PROBE_BUNDLE` with that local path and removes the marked
transfer staging. Identical repeated publication is reused; a corrupt existing
bundle, an identity mismatch, an unexpected file, or an HDF5 payload fails
closed. The compact CPU bundle is retained. A normal-case failure is transferred
before the workflow returns the nonzero case exit code. If bundle validation or
transfer fails, active or staging evidence is retained for diagnosis.

Validate and inspect the printed local bundle with:

```bash
python -m src.generation.cli.cli_generation validate-timing-probe "$PROBE_BUNDLE"
python -m json.tool "$PROBE_BUNDLE/manifest.json"
python -m json.tool "$PROBE_BUNDLE/method_verdicts.json"
python -m json.tool "$PROBE_BUNDLE/batch_log_candidates.json"
less "$PROBE_BUNDLE/comsol_batch.log"
less "$PROBE_BUNDLE/stdout.log"
less "$PROBE_BUNDLE/stderr.log"
```

The immutable inventory is exactly:

```text
README.md
batch_log_candidates.json
comsol_batch.log
environment.json
exact_command.json
manifest.json
method_verdicts.json
observed_wall_timing.json
parser_summary.json
phase_events.jsonl
sha256sums.txt
stderr.log
stdout.log
stdout_candidates.json
```

`manifest.json` binds the source commit and campaign hash, batch and case
identities, input-generation identity, canonical and scratch input hashes,
template, configured resources, COMSOL version query, exact command, normal
publication, failure-attempt, or temporary-license wait evidence, process
result, host, and bundle file hashes.
No `case.h5`, solved model, export, Dataset package, PT shard, readiness evidence,
or production timing field belongs in the bundle.

## Timing interpretation

Candidate A retains possible `Solution time`, `Elapsed time`, and
`Computation time` lines with exact source, byte offset, line number, context,
phase association, parsed unit, duplicate classification, and ambiguity
reasons. Synthetic tests prove only parser behavior. Even a single ordered
candidate for both phases is labelled candidate evidence; it is never reported
as validated real COMSOL grammar or selected automatically.

Candidate B remains
`not_implementable_from_current_source_boundary`. The exact top-level stationary
and transient solve calls are inside the binary MPH execution boundary. The
probe does not introduce a second COMSOL process, edit the model, or invent
shell markers as substitutes for those boundaries.

Candidate C records host-observed intervals between retained markers. Polling,
COMSOL buffering, process buffering, filesystem visibility, parser latency, and
non-solver work may all affect those intervals. Candidate C is diagnostic only.

Accordingly, every bundle records:

```text
recommendation = unresolved_pending_real_probe_review
real_comsol_grammar_validated = false
production_timing_fields_updated = false
```

If later real evidence proves exact semantics, the intended distinct fields are
`comsol_stationary_airflow_seconds`,
`comsol_transient_drying_seconds`, and their sum
`comsol_scientific_solver_seconds`. Keep `comsol_process_seconds` separate.
Never infer scientific solver time from Slurm elapsed time, simulated physical
time, file timestamps, process time minus estimated overhead, observed marker
intervals, or missing values filled with zero.

## Exact future removal inventory

After any needed real diagnostic evidence is retained, remove the temporary
surface together:

- Delete `src/generation/generation_timing_probe.py`,
  `tests/generation/test_generation_timing_probe.py`, and this guide. There is no
  probe-specific production YAML to remove.
- Remove `timing_probe_service`, the `timing-probe`,
  `validate-timing-probe`, and `publish-transferred-timing-probe` CLI parsers and
  dispatch branches from `src/generation/cli/cli_generation.py`.
- Remove `diagnostic_observer` from
  `generation_runtime_batch.execute_prepared_case` and `run_case`. Remove
  `diagnostic_batchlog` and its conflict checks and flags from
  `generation_runtime_comsol.build_comsol_command`. Preserve the normal command
  and case lifecycle.
- Remove the optional batch-log behavior used by the fake COMSOL fixture and
  the probe tests from `tests/generation/conftest.py`.
- Remove `TIMING_PROBE_SCHEDULER_OPTIONS`,
  `resolve_timing_probe_contract`, `run_timing_probe_remote`, the
  `timing-probe CONFIG` usage and dispatch, and its background allow-list entries
  from the Generation shell/background services.
- Remove the timing-probe subsection from `docs/simulation_generation.md`.
- Retire the `comsol_phase_timing_probe` bundle and its module-owned observer,
  candidate, observed-wall, parser, verdict, environment, command, manifest,
  file-evidence, and checksum payloads.
- After evidence-retention decisions, remove only probe-owned stored bundles
  under `03_experiments/comsol_phase_timing_probe`. Do not reinterpret or remove
  an existing bundle merely because code is deleted.

Do not create or update `_codex_handoff` for this diagnostic.
