# COMSOL phase timing probe

This temporary, opt-in diagnostic exists only to collect evidence about whether
COMSOL's retained text can support phase-specific solver timing. It runs exactly
one configured `technical_runtime_smoke` transient case:
`transient_drying__lentil__natural`, case 1. It does not publish Generation
cases, Dataset packages, PT shards, readiness evidence, replacement work, or
cleanup decisions. Cases generated before final timing integration may not have
recoverable phase logs.

No real COMSOL execution was performed while implementing or testing this probe.
The synthetic parser and lifecycle tests therefore do not validate real COMSOL
log grammar or support a performance claim.

The shell admits the repository-local probe and campaign configurations. The
referenced campaign remains authoritative for the transient template, scientific
case construction, Slurm partition, site capacity, wall time, modules, and
virtual environment. The probe independently owns `cores_per_case`; this
configuration requests 4 cores while the referenced campaign requests 8. The
shell validates the probe count against the configured 32-core node, passes the
same 4 cores to `srun --cpus-per-task`, and the probe passes 4 to COMSOL with
`-np`. COMSOL never runs on the login shell.

The 4-core result is only log-format and timing-source evidence. It is not a
performance benchmark and is not directly comparable to later 8-core production
timing.

Run it through that configured CPU-host allocation:

    ./scripts/generation_workflow.sh timing-probe configs/generation/diagnostics/comsol_phase_timing_probe.yaml

Run the same isolated diagnostic in the managed background session:

    ./scripts/generation_workflow.sh timing-probe configs/generation/diagnostics/comsol_phase_timing_probe.yaml --background

The background launcher prints `workflow_session_id=<id>`. Inspect its controller
output with the emitted value:

    ./scripts/generation_workflow.sh background-status <workflow_session_id>

The child announces `PROBE_ID`, `PROBE_ACTIVE`, and `PROBE_WORK` before
COMSOL starts. While the process is active, follow the runtime-owned batch log
using that exact announced work path:

    tail -f PROBE_WORK/runtime/comsol_batch.log

The live work tree is not the final bundle. On completion, the command prints
`PROBE_BUNDLE`, atomically publishes the immutable bundle at
`STORAGE_ROOT/03_experiments/comsol_phase_timing_probe/PROBE_ID`, validates it,
and removes only its validated active/work owner. With container-visible storage
at `/workspace/storage`, the expected path is:

    /workspace/storage/03_experiments/comsol_phase_timing_probe/PROBE_ID/

Use the exact printed path when configured storage is elsewhere. Revalidate and
inspect a retained bundle with:

    python -m src.generation.cli.cli_generation validate-timing-probe "$PROBE_BUNDLE"
    python -m json.tool "$PROBE_BUNDLE/method_verdicts.json"
    python -m json.tool "$PROBE_BUNDLE/batch_log_candidates.json"
    python -m json.tool "$PROBE_BUNDLE/stdout_candidates.json"
    less "$PROBE_BUNDLE/comsol_batch.log"
    less "$PROBE_BUNDLE/stdout.log"
    less "$PROBE_BUNDLE/stderr.log"

`method_verdicts.json` summarizes the diagnostic candidates. The candidate JSON
files retain every possible timing line with its source, location, exact text,
context, detected phase, parsed value, unit, and ambiguity classification. The
three complete logs remain the authoritative evidence; do not rely only on a
summary excerpt.

Resume is fail-closed. The active run key binds the exact probe and campaign
configuration, campaign digest, source commit, scheduler kind, and requested
work root. Before starting or adopting the child, the controller reconstructs
the exact case payload, generated input inventory, copied model digest, scalar
handoff, COMSOL command, attempt identity, child control, and control digest.
The child independently rechecks its admitted paths, identities, input/model
inventory, run key, and control digest. Mutation preserves the active evidence
and prevents a second launch.

Candidate A retains and parses COMSOL `Solution time` text lines. It records
every candidate, context, unit, duplicate, and ambiguity status; it confirms a
phase only when one finite, nonnegative, top-level, unambiguous candidate is
found in stationary-then-transient order. Its sum remains diagnostic until real
COMSOL evidence proves the grammar. Candidate B is deliberately unavailable
(`not_implementable_from_current_source_boundary`): both solve boundaries are
inside the MPH model, so no in-process or shell split timing is invented.
Candidate C records observed wall markers with polling and buffering caveats; it
is diagnostic only and can never establish solver time. No production timing
method is selected by this probe, even if one diagnostic candidate is reported
as confirmed.

The intended final production semantics, if a later investigation proves exact
boundaries, are:

    comsol_stationary_airflow_seconds
    comsol_transient_drying_seconds
    comsol_scientific_solver_seconds =
        comsol_stationary_airflow_seconds
        + comsol_transient_drying_seconds

Keep `comsol_process_seconds` separate. Never derive scientific solver time
from process time minus estimated overhead, Slurm elapsed time, simulated
physical time, file timestamps, observed-wall marker intervals, or missing
values filled with zero.

To hand evidence to a later fresh Codex session, make the complete
`PROBE_BUNDLE` directory available without editing it, identify the exact
source commit and probe command, and ask that session to run the validator before
reading `manifest.json`, `method_verdicts.json`, the candidate records, and
the retained logs. The later session must consume and validate the real bundle,
treat Candidate C as diagnostic, and avoid promoting any field until the real
log proves the exact semantics above. After that investigation preserves any
needed evidence, that session must remove this temporary probe using the inventory
below.

## Exact future removal inventory

Remove all of the following together after retaining any diagnostic evidence
that is still needed:

- Delete
  `configs/generation/diagnostics/comsol_phase_timing_probe.yaml`,
  `src/generation/generation_timing_probe.py`,
  `tests/generation/test_generation_timing_probe.py`, and this document.
  Deleting the module removes all of its constants, regular expressions,
  `ProbeObservationState`, parsing/observation helpers, session/child helpers,
  `parse_solution_times`, `summarize_solution_times`,
  `observe_appended_bytes`, `run_timing_probe`, `validate_probe_bundle`,
  `_execute_child`, and `_module_main`; no symbol from that module is
  permanent.
- In `src/generation/cli/cli_generation.py`, remove the
  `timing_probe_service` import, the `timing_probe` and
  `validate_timing_probe` parsers, both dispatch branches, their nested
  `announce_probe`, and the emitted `PROBE_ID`, `PROBE_ACTIVE`,
  `PROBE_WORK`, `PROBE_BUNDLE`, and `PROBE_BUNDLE_VALID` fields.
- In `scripts/generation_workflow.sh`, remove the `timing-probe CONFIG`
  usage/dispatch path, `TIMING_PROBE_CONFIG_PATH`,
  `TIMING_PROBE_RELATIVE_PATH`, `resolve_timing_probe_contract`,
  `run_timing_probe_remote`, and both `timing-probe` background allow-list
  branches. Remove `timing-probe` from
  `src/generation/generation_background.py::_SUPPORTED_SUBCOMMANDS`.
- In
  `src/generation/runtime/generation_runtime_comsol.py::build_comsol_command`,
  remove the probe-only `diagnostic_batchlog` parameter, its conflict
  validation, and its `-batchlog ... -batchlogout` arguments. Preserve the
  ordinary production command behavior.
- Remove the “Bounded COMSOL phase-timing diagnostic” subsection and link from
  `docs/simulation_generation.md`.
- Retire every probe-only version-1 schema:
  `generation_timing_probe` (configuration),
  `comsol_phase_timing_probe_session` (active session),
  `comsol_phase_timing_probe_child` (child control), and
  `comsol_phase_timing_probe` (immutable bundle), plus the module-owned
  version-1 status, child-start/exit, observer/event, candidate, observed-wall,
  parser-summary, method-verdict, environment, command, file-evidence, and
  checksum payloads.
- After evidence retention is decided, remove only probe-owned stored data under
  `STORAGE_ROOT/03_experiments/comsol_phase_timing_probe` and any external
  `WORK_ROOT/comsol_phase_timing_probe` owner. Active-session files include
  `session.json`, `status.json`, `child_control.json`,
  `controller_child.json`, `child_started.json`, `child_exit.json`,
  `observer.json`, the isolated case/model/input tree, and runtime logs.
  Immutable bundle inventory is exactly `manifest.json`,
  `method_verdicts.json`, `exact_command.json`, `environment.json`,
  `comsol_batch.log`, `stdout.log`, `stderr.log`,
  `phase_events.jsonl`, `batch_log_candidates.json`,
  `stdout_candidates.json`, `observed_wall_timing.json`,
  `parser_summary.json`, `sha256sums.txt`, and `README.md`.

Do not remove or reinterpret existing probe bundles merely by deleting code, and
do not create or update `_codex_handoff`.
