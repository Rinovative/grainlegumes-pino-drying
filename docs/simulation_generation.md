# VP2 Simulation Generation and Campaign Workflow

Audience: researchers operating or adapting the VP2 reference-simulation pipeline. This is the sole operational guide for generation architecture, configuration, native CPU execution, collection, HDF5 publication, and manual COMSOL adaptation. Canonical parameter definitions and report symbols live in the [parameter reference](generation_parameter_reference.md).

Production generation is intentionally fail-closed. Material/operation evidence, production counts and seeds, and real COMSOL mappings remain unresolved; no value or mapping is inferred from a template binary.

## Ownership and configuration

Generation owns reference simulations and generated scientific data. Learning owns task-specific training, tuning, acceptance, and later learning-task choices. The source-owned task registry remains under `src/domain`; only `steady_flow` is registered.

```text
configs/
├── generation/
│   ├── campaigns/
│   │   ├── steady_flow/
│   │   │   └── family_generalization.yaml
│   │   └── transient_drying/
│   │       ├── family_generalization.yaml
│   │       └── lentil_pilot.yaml
│   ├── execution/
│   │   └── cluster_cpu.yaml
│   ├── materials/
│   │   ├── chickpea.yaml
│   │   ├── lentil.yaml
│   │   ├── kidney_bean.yaml
│   │   ├── almond.yaml
│   │   └── field_pea.yaml
│   ├── operations/
│   │   └── fixed_bed.yaml
│   ├── profiles/
│   │   ├── steady_flow.yaml
│   │   └── transient_drying.yaml
│   ├── common.yaml
│   └── registry.yaml
└── learning/
    └── steady_flow/
        ├── acceptance/
        ├── experiments/
        └── optuna/
```

A campaign resolves exact paths to six disjoint layers:

| Layer | Sole responsibility |
| --- | --- |
| `registry.yaml` | Names, units, kinds, transforms, sampling blocks, OOD groups, derivations, and report symbols |
| `common.yaml` | Global geometry/time contracts, fixed physical constraints, formulas, adapter schemas, and HDF5 settings |
| `materials/<family>.yaml` | Role-neutral material ranges/values, product scope, coupled records, moisture bounds, and evidence |
| `operations/fixed_bed.yaml` | Material-independent pressure and inlet/ambient operation distributions |
| `profiles/<profile>.yaml` | Template-owned logical export roles and manually confirmed COMSOL mappings |
| `execution/cluster_cpu.yaml` | Native runtime, site, Slurm resources, timeout, and retention only |

The loader validates exact schemas and resolves one typed campaign; there is no fallback path discovery or compatibility alias. Scientific identities exclude nodes, cores, wall time, timeout, retention, and scheduler settings. CLI overrides may change execution resources or select an already declared batch, never campaign science.

## Material, batch, campaign, and dataset

| Concept | Meaning and owner |
| --- | --- |
| Material family | One role-neutral scientific/evidence definition; no seen/OOD role |
| Batch | One simulation profile × one material family × one neutral sampling regime |
| Campaign | Profiles, role assignments, counts/seeds, planned batches, and dataset declarations |
| Dataset package | Exact terminal cases assembled for one learning task and evaluation regime |

Visible names remain readable while immutable IDs bind full digests:

```text
batch_name   = <simulation_profile>__<material_family>__<sampling_regime>
dataset_name = <learning_task>__<ordered-material-list>__<evaluation-regime>
```

Names omit versions, timestamps, seeds, counts, source profiles, and digests. `batch_id`, `dataset_id`, `campaign_id`, and `campaign_run_id` append digest prefixes and manifests retain full identities. Material order follows the campaign declaration deterministically.

## Simulation profiles and canonical cases

| Profile | Immutable source template | Case content | Learning availability |
| --- | --- | --- | --- |
| `steady_flow` | `simulation/steady_flow/template_brinkman.mph` | coordinates, scalar provenance, static fields | registered `steady_flow` view |
| `transient_drying` | `simulation/transient_drying/template_brinkman_temp_moist.mph` | steady/static fields plus time, transient states, schedule, globals, and status | steady-flow view plus an unregistered physical transition view |

A template is verified by SHA-256, then copied to an isolated case work directory as `model.mph`; the repository file is never an output target. The transient source digest is additionally pinned by `simulation/transient_drying/template.sha256`.

`case.h5` schema version 2 is the sole canonical numeric case payload:

```text
/
├── coords/x, coords/y                 float64; unit = m
├── static/fields                      float32; names and units attributes
├── scalar/values                      float64; names and units attributes
├── time                               float64; unit = h                 [transient]
├── transient/fields                   float32; names and units          [transient]
├── schedule/values                    float64; names and units          [transient]
└── global/values                      float64; names and units          [transient]
```

Root attributes bind profile, material family, sampling regime, case input/simulation identities, config and template digests, export-contract digest, airflow source, Git commit, learning views, and raw-export hashes. JSON sidecars retain case, execution, timing, status, failure/resume, and batch/campaign evidence. Publication is atomic and validates finite Cartesian axes, exact names/units, shapes, dtypes, compression, chunks, schedule nodes, and source identities.

The transient index admits only consecutive regular one-hour states, records `time`/`dt` with `time_unit: h`, excludes transitions across an irregular exact-stop state, and derives increment targets without changing canonical absolute states. It does not register a transient learning task.

## Native CPU and Docker responsibilities

| Environment | Responsibility | Runtime |
| --- | --- | --- |
| CPU cluster (`sricehpc01`) | COMSOL generation and terminal batch/campaign evidence | Native `Python/3.10` venv, `Comsol/v6.4`, Slurm; no Docker |
| GPU/development host (`hpc115`) | Repository control, collection, dataset construction, learning, analysis | Maintained Docker image; dataset construction is CPU-only and bypasses the GPU queue |

The native venv is installed from the exact detached repository commit with `.[generation-cpu]`. Every Slurm worker uses isolated scratch, one shared campaign-wide concurrency plan, and the same generation runner. The Docker dataset primitive is only:

```bash
./scripts/docker_job.sh build-datasets <generation-campaign-config>
```

It invokes the existing package builder with the repository read-only and configured storage read/write; it uses neither COMSOL nor the GPU queue.

## Host workflow

Run these commands on the GPU host, outside the development container.

### Prepare the CPU checkout

Dry-run the exact setup first:

```bash
./scripts/generation_workflow.sh setup-cpu --cpu-host sricehpc01
```

Then explicitly perform the detached checkout, native venv install, module checks, and compute-node smoke:

```bash
./scripts/generation_workflow.sh setup-cpu --cpu-host sricehpc01 --execute
```

### Launch

```bash
./scripts/generation_workflow.sh launch \
  configs/generation/campaigns/transient_drying/family_generalization.yaml \
  --max-nodes 4 \
  --cases-per-node 4 \
  --cores-per-case 8 \
  --max-parallel-cases 16
```

Launch requires a clean local HEAD reachable from the configured origin and the same exact commit in the detached CPU checkout. It validates the campaign in the maintained container, verifies the remote runtime, persists submission intent before `sbatch`, submits one shared Slurm pool, prints the immutable run ID, and returns without waiting. `--only-batch <declared-batch-name>` narrows execution without changing campaign science; `--wall-time` and resource flags are execution-only.

### Inspect, collect, and build

```bash
./scripts/generation_workflow.sh status <campaign-run-id>
./scripts/generation_workflow.sh collect <campaign-run-id>
./scripts/generation_workflow.sh build-datasets <campaign-run-id>
```

`status` reconstructs scheduler and per-case state and can recover a lost submission receipt by the persisted scheduler job name. `collect` first requires terminally valid remote campaign/batch manifests, transfers only declared public directories into staging, revalidates them in Docker, and publishes only new or identical local identities. It never uses destructive synchronization. The terminal campaign evidence carries the exact repository-relative campaign config used by standalone `build-datasets`; the run ID is not used to guess a filename.

For one foreground convenience flow:

```bash
./scripts/generation_workflow.sh all \
  configs/generation/campaigns/transient_drying/family_generalization.yaml \
  --max-nodes 4 --cases-per-node 4 --cores-per-case 8 --max-parallel-cases 16 \
  --wait --build-datasets
```

Without `--wait`, `all` returns after launch and prints the resume command. A case is skipped only when its complete persisted identity and success evidence validate. Failures are isolated and durable; retry uses the same case identity. Campaign finalization requires every selected terminal batch, the exact Git commit, and immutable manifest hashes. Existing conflicting local or remote identities fail closed rather than overwrite.

## Manual COMSOL 6.4 adaptation checklist

Adapt reviewed working copies manually. Keep both profile files at `template_ready: false`, with null export patterns and null logical headers, until every item below is confirmed. Never infer tags or formulas from binary strings, preserve aliases, or embed absolute developer/storage paths.

### Case-local inputs

All adapters are semicolon-delimited paths relative to the disposable `model.mph` directory.

| File | Exact header | Units |
| --- | --- | --- |
| `fields.csv` | `x;y;Kxx;Kxy;Kyy;eps_bed;p_bc;X_0_db_field` | `m; m; m^2; m^2; m^2; 1; Pa; kg/kg` |
| `scalars.csv` | `name;value;unit` | row-specific units below |
| `schedule.csv` | `t;T_in;omega_in;phi_in` | `h; K; kg/kg; 1` |

Scalar rows, in order:

```text
T_init [K]                 T_amb [K]                 T_in_ref [K]
eps_bed_cal_ref [1]        rho_bu_dry_ref [kg/m^3]  k_gr [W/(m*K)]
cp_gr_dry [J/(kg*K)]       X_target_wb [1]           r_surf_0 [1/s]
r_int_surf [1]             f_surf [1]                A_osw [1]
B_osw [1]                  C_osw [1]                 f_wet_dm_max [1]
```

Confirm coordinates, Cartesian orientation, column order, extrapolation, scalar lookup timing/destinations, and unit parsing. Configure reviewed linear interpolation functions `T_in_fun(t)` and `phi_in_fun(t)` with tabulated hours and explicit argument/value units. Confirm COMSOL's conversion from solver time to hours; do not guess a conversion. `omega_in` is archived and may enter only a separately reviewed expression. `T_in_ref` is the full planned 0–168 h time-average inlet temperature and is unchanged by an early solver stop.

### Physical and initial-state definitions

Use these logical definitions once, with COMSOL units verified:

```text
w_gr = w_surf + w_int
X_db = w_gr / rho_bu_dry
X_wb = w_gr / (rho_bu_dry + w_gr)
X_wb = X_db / (1 + X_db)
X_db = X_wb / (1 - X_wb)

rho_bu_dry = rho_bu_dry_ref * (1 - eps_bed) / (1 - eps_bed_cal_ref)
w_gr_0 = rho_bu_dry * X_0_db_field
w_surf(t=0) = f_surf * w_gr_0
w_int(t=0) = (1 - f_surf) * w_gr_0
r_surf = r_surf_0
r_int = r_int_surf * r_surf
```

`X_wb_bulk` is integrated water divided by integrated dry-plus-water mass, never an unweighted spatial mean. `cp_gr_eff = cp_gr_dry + <reviewed moisture contribution>`; the moisture term, basis, coefficient, and units remain unresolved. Do not cache full `X_db` or `X_wb` fields in HDF5 because `w_surf`, `w_int`, and `rho_bu_dry` recover them.

For the two-state moisture ODE, verify dependent-variable order before applying the damping/mass matrix `[f_surf, 0; 0, 1-f_surf]`. Correct the equation ordering rather than transposing silently.

### Studies and stop event

1. Study 1 solves stationary airflow.
2. Study 2 starts only after Study 1 succeeds and uses its explicitly reviewed frozen within-case airflow solution.
3. Each case initializes only from its own adapters and Study 1 result; disable previous-case, sweep, or stale-solution reuse.
4. Apply `T_init`, `w_surf(t=0)`, and `w_int(t=0)` on the reviewed variables/domains.
5. Preserve exact regular one-hour outputs. Export an irregular exact-stop state only as diagnostic evidence.

Define the dry-mass-weighted fraction and stop condition exactly:

```text
comp1.f_wet_dm =
    intop1(rho_bu_dry * if(X_wb > X_target_wb, 1, 0))
    / intop1(rho_bu_dry)

comp1.f_wet_dm <= f_wet_dm_max
```

`f_wet_dm_max` remains `0.05`. Confirm the component/integration tags, domain, Boolean semantics, evaluation point, event direction, and tolerance. Do not substitute an area fraction or unweighted mean.

### Required output roles

Every path must remain relative beneath case-local `exports/`; populate an exact deterministic pattern and logical-to-COMSOL header only after manual confirmation.

| Role | Logical names (ordered) | Units (ordered) |
| --- | --- | --- |
| `steady_flow_fields` | `x, y, Kxx, Kxy, Kyy, eps_bed, p_bc, X_0_db_field, u, v, p, rho_bu_dry` | `m, m, m^2, m^2, m^2, 1, Pa, kg/kg, m/s, m/s, Pa, kg/m^3` |
| `transient_fields` | `x, y, t, T, phi, w_surf, w_int` | `m, m, h, K, 1, kg/m^3, kg/m^3` |
| `global_time_series` | `t, X_wb_bulk, X_wb_max, X_wb_q95_mass, f_wet_dm, T_out_mean, phi_out_mean, m_w_gr, m_v_gas, m_dot_evap, m_dot_v_in, m_dot_v_out` | `h, 1, 1, 1, 1, K, 1, kg, kg, kg/s, kg/s, kg/s` |
| `final_status` | `t_final, f_wet_dm_final, X_target_wb, X_wb_bulk, X_wb_max, X_wb_q95_mass, T_min_final, T_max_final, phi_min_final, phi_max_final` | `h, 1, 1, 1, 1, 1, K, K, 1, 1` |

Confirm source datasets, table orientation/sort order, delimiter, overwrite behavior, terminal-state consistency, and exact regular/irregular time handling. Manually verify permeability-tensor orientation; pressure and vapor-flow normals/signs; all dependent-variable, physics, study, solver, solution, dataset, import, interpolation, integration, averaging, event, and export tags; the Oswin equation convention; local `m_evap` versus integrated `m_dot_evap`; the dry-mass-weighted `X_wb_q95_mass` implementation; outlet averages; and every filename/header mapping. Only then populate profile YAML mappings and perform a separately approved template-update/validation workflow.

## Blockers before real generation

- Resolve every material taxonomy/product scope, parameter range/value, coupled record, evidence source, confidence, and validity range.
- Resolve operation ranges, production campaign counts/seeds, wall time, and site-approved resource choices.
- Complete the COMSOL checklist, smoke the converter against deliberate exports, and approve any template-byte change separately.
- Keep common/material/operation templates non-executable and profile mappings fail-closed until all required owners validate together.
