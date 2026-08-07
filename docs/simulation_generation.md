# Dual-profile Python generation and COMSOL execution

VP2 supports exactly two reference-simulation profiles through one Python-owned
generation, execution, locking, resume, publication, cluster, and transfer
lifecycle:

| `simulation_profile` | Immutable template | Learning views | Airflow source |
|---|---|---|---|
| `steady_flow` | `simulation/steady_flow/template_brinkman.mph` | `steady_flow` | `comsol_steady_reference` |
| `transient_drying` | `simulation/transient_drying/template_brinkman_temp_moist.mph` | `steady_flow`, raw `transient_drying` | `comsol_coupled_reference` |

The standalone template produces pressure and velocity references. The coupled
template calculates its own stationary airflow together with configured
transient heat and moisture outputs; it does not require a separate airflow
simulation. The final transient tensor fields, temporal windows, normalization,
models, and physics losses are deliberately undefined.

The templates are immutable repository resources:

```text
simulation/steady_flow/template_brinkman.mph
  size: 4,413,253 bytes
  SHA-256: c3363528f49a29774cbf7f48948d5216022f1bac14f4f6c635e7b912985ba976

simulation/transient_drying/template_brinkman_temp_moist.mph
  size: 5,231,788 bytes
  SHA-256: 0cd4ae1a0be0d60a9c8617b4f888eb4b7a3d255518d6e64685a7f12ed6516003
```

The transient digest is owned by `simulation/transient_drying/template.sha256`.
The steady digest is owned by the simulation-profile registry. There is no
second competing identity source for either template. Each case receives a
private `model.mph` copy in its work directory; a repository template is never
used as a solver output.

## Profile and configuration contract

`simulation_profile` is mandatory. It is never inferred from a filename,
directory, schedule, export, or learning task. The selected profile owns its
template identity, exact generated spatial adapter, allowed scalar and schedule
adapters, required export roles, available learning views, and airflow
provenance. Unknown profiles fail during preflight.

Both profiles use the same generated spatial columns, in order:

```text
x, y, Kxx, Kxy, Kyy, eps, p_bc
```

These are mapped to the registered `steady_flow` TaskSpec. The standalone
profile permits no scalar or schedule file. The coupled profile permits generic
configured scalars and a generic configured schedule, without inventing drying
parameters.

A configuration has this structural shape:

```yaml
schema_version: 1
simulation_profile: steady_flow  # or transient_drying
cases:
  indices: [1]
  seed_base: <configured-seed>
  overrides: {}
generator:
  version: python_multiscale_v1
  domain: <explicit-domain-mapping>
  parameters: <complete-explicit-generator-mapping>
sampling: null
inputs:
  spatial_files:
    - filename: case_0001.csv
      delimiter: ";"
      columns: [x, y, Kxx, Kxy, Kyy, eps, p_bc]
exports:
  root: exports
  contracts: <profile-owned-explicit-role-mappings>
execution:
  executable: null
  timeout_seconds: <positive-seconds>
  retain_solved_model: false
  extra_arguments: []
cluster:
  config_path: configs/generation/cluster_cpu.yaml
```

`configs/generation/steady_flow/infrastructure_smoke.yaml` is a maintained
non-scientific schema/preflight fixture. Its values are not a grain-family
definition and its `airflow.csv` mapping has not been validated against a real
COMSOL run. Family means, anchors, ranges, correlations, and final ID/OOD
distributions remain future scientific decisions.

The transient profile requires two roles:

- `steady_flow_fields`: exactly one numeric export with explicit header mappings
  for `x`, `y`, `p`, `u`, and `v`; an optional configured time header permits
  repeated stationary airflow rows.
- `transient_fields`: one or more configured raw numeric exports. This role has
  no tensor-column mapping because the final transient learning contract is not
  defined.

Actual COMSOL export names and expressions must be confirmed from the active
template and supplied explicitly. Preflight fails when a required role or field
mapping is absent; the pipeline does not guess model tags or filenames.

## Deterministic generation and case identity

The shared generator owns coordinate grids, case seeds, optional uniform/LHS/
Sobol sampling, multiscale structure, permeability tensors, porosity, inlet
pressure, generic scalars, and optional schedules. It preserves the existing
formulas, `meshgrid(x, y)` orientation, Fortran-order table flattening,
permeability symmetry and positive definiteness, porosity bounds, units, and
locale-independent 17-digit output. Reproducibility binds the resolved
configuration, `python_multiscale_v1`, `seed_base + case_index`, every input
file digest, the template digest, and the export contract. No cross-language
bitwise-equivalence claim is made.

`case.json` records `simulation_profile`, template, views, airflow source,
generator and seed evidence, resolved parameters, adapters, and input hashes.
Batch and case identities are profile-qualified.

## Isolated execution, publication, and resume

The local command is an argument vector, never a shell string:

```text
${COMSOL_EXECUTABLE} batch -inputfile model.mph -outputfile solved.mph -np <cores-per-case>
```

Each case runs with a private working directory as `cwd`. Successful work is
removed; failed work is retained unless `--cleanup-failed` is explicit. Raw
exports are copied unchanged. The coupled profile's transient exports remain
under `exports/` without normalization or reduction.

The canonical airflow table is published at:

```text
learning_views/steady_flow/fields.csv
```

When an export repeats airflow over time, raw bytes are preserved and `p`, `u`,
and `v` must remain stationary at each coordinate within relative and absolute
tolerance `1e-10`. Variation beyond that tolerance fails the case.

Final storage is:

```text
STORAGE_ROOT/01_generation/
├── meta/<batch_id>/
│   ├── resolved_config.json
│   ├── batch_manifest.json
│   └── _SUCCESS
├── raw/<batch_id>/<case_id>/
├── processed/<batch_id>/<case_id>/
│   ├── exports/
│   ├── learning_views/steady_flow/fields.csv
│   ├── timing.json
│   ├── solver.log
│   ├── provenance.json
│   └── _SUCCESS
└── .state/<simulation_profile>/<batch_id>/
```

Locks, deterministic identities, exact membership, hashes, atomic directory
renames, and success markers own resume. A valid completed case is skipped;
corrupt completion fails closed. The final batch manifest explicitly records
`available_learning_views`, `airflow_source`, profile, template, and each case
publication digest.

## Commands

```bash
export STORAGE_ROOT=/absolute/storage/root
export COMSOL_EXECUTABLE=/absolute/path/to/comsol
export GENERATION_CONFIG=/absolute/path/to/generation.yaml

python -m src.generation.cli.cli_generation validate-config "$GENERATION_CONFIG"
python -m src.generation.cli.cli_generation generate-case "$GENERATION_CONFIG" 1 /tmp/case-inputs
python -m src.generation.cli.cli_generation prepare-case "$GENERATION_CONFIG" 1
python -m src.generation.cli.cli_generation print-command "$GENERATION_CONFIG" 1 --cores-per-case 8
python -m src.generation.cli.cli_generation run-case "$GENERATION_CONFIG" 1 --cores-per-case 8
python -m src.generation.cli.cli_generation validate-case "$GENERATION_CONFIG" 1
python -m src.generation.cli.cli_generation finalize-batch "$GENERATION_CONFIG"
```

The same commands serve both profiles. Real COMSOL execution has not been
validated in this development container.

## Bounded cluster execution

One case stays on one node. The shared cluster configuration declares 32 cores
per node, and validation enforces:

```text
cases_per_node * cores_per_case <= 32
max_parallel_cases <= max_nodes * cases_per_node
```

All four resource controls are required: `--max-nodes`, `--cases-per-node`,
`--cores-per-case`, and `--max-parallel-cases`. Example dry-run submission:

```bash
python -m src.generation.cli.cli_generation print-submit "$GENERATION_CONFIG" \
  --max-nodes 4 --cases-per-node 4 --cores-per-case 8 --max-parallel-cases 16
```

The common node worker is `scripts/generation_node.sh`. Slurm site settings,
COMSOL licensing, and real multi-node behavior remain unvalidated.

## Transfer and steady-flow dataset publication

The common transfer helper excludes private state and never uses `--delete`:

```bash
./scripts/generation_transfer.sh cpu-host /absolute/cpu/storage /absolute/gpu/storage
./scripts/generation_transfer.sh cpu-host /absolute/cpu/storage /absolute/gpu/storage --execute
python -m src.generation.cli.cli_generation validate-transfer "$GENERATION_CONFIG" \
  --storage-root /absolute/gpu/storage
```

Omitting `--execute` is a dry run. Real transfer has not been executed here.

A completed batch from either profile can publish a `steady_flow` dataset:

```bash
python -m src.datasets.dataset_build <batch_id>
python -m src.datasets.dataset_build <batch_id> --dataset-id <dataset_id>
```

The one builder reads the canonical airflow view, preserves TaskSpec field
order and units, applies the established permeability representations, and
atomically publishes tensors plus a terminal-manifest metadata snapshot. Source
profile, template digest, views, and airflow source remain explicit. A coupled
batch can therefore train the existing airflow operator now, while its future
transient dataset remains deferred.

VP2 does not maintain historical persisted-data formats. No MATLAB or COMSOL
LiveLink runtime is required; the former producer exists only in the separate
[VP1 repository](https://github.com/Rinovative/grainlegumes-pino-airflow).
