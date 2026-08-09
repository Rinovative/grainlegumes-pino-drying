# VP2 Simulation Generation and Campaign Workflow

Audience: researchers operating or adapting the VP2 reference-simulation pipeline. This is the permanent guide for generation profiles, COMSOL handoff, HDF5 publication, dataset packages, and runtime validation. Canonical parameter definitions and report symbols live in the [parameter reference](generation_parameter_reference.md).

Production generation is intentionally fail-closed. Material and operation evidence, production counts and seeds, COMSOL export mappings, and the exhaustive stationary-conditioning audits remain unresolved. Both maintained `.mph` files have validated byte identities, but their model-tree content is runtime-unverified until opened and exercised with COMSOL 6.4. A SHA-256 pass is not physical, structural, import, solve, or export validation.

## Ownership and configuration

Generation owns reference simulations and generated scientific data. Learning owns task-specific training, tuning, and evaluation. Only `steady_flow` is a registered learning task; the transient physical view is buildable and loadable but has no model, trainer, normalizer, rollout, Optuna study, EDA page, or evaluator.

The layered generation configuration is:

```text
configs/generation/
├── campaigns/{steady_flow,transient_drying}/
├── execution/cluster_cpu.yaml
├── materials/<family>.yaml
├── operations/fixed_bed.yaml
├── profiles/{steady_flow,transient_drying}.yaml
├── common.yaml
└── registry.yaml
```

| Layer | Sole responsibility |
| --- | --- |
| `registry.yaml` | Names, units, kinds, transforms, sampling blocks, OOD groups, and derivations |
| `common.yaml` | Geometry, time, fixed physics, formulas, adapter schemas, and HDF5 settings |
| `materials/<family>.yaml` | Role-neutral material values/ranges, coupled records, bounds, and evidence |
| `operations/fixed_bed.yaml` | Pressure-profile and inlet/ambient operation distributions |
| `profiles/<profile>.yaml` | Template-owned export roles and explicitly reviewed COMSOL mappings |
| `execution/cluster_cpu.yaml` | Native runtime, site, Slurm resources, timeout, and retention only |

There is no fallback path discovery, legacy schema reader, old boundary-name alias, or alternate template lookup. Scientific identity excludes execution resources and retention settings.

## Canonical templates and inspection rule

| Profile | Canonical template | Canonical sidecar | Learning views |
| --- | --- | --- | --- |
| `steady_flow` | `simulation/steady_flow/steady_flow_template.mph` | `simulation/steady_flow/steady_flow_template.sha256` | `steady_flow` |
| `transient_drying` | `simulation/transient_drying/transient_drying_template.mph` | `simulation/transient_drying/transient_drying_template.sha256` | `steady_flow`, `transient_drying` |

The source template is hash-validated and copied byte-for-byte into an isolated case directory as `model.mph`; the repository file is never an output target. Exactly one authoritative `.mph` exists per maintained profile.

Raw extraction with `strings`, `rg`, or similar tools may show that text occurs in an `.mph`, but absence from raw extraction is not evidence that a function, variable, study, physics node, file binding, expression, export, or result node is absent or inactive. Do not change the Python contract, add a fail-closed check, rename a variable, or edit an `.mph` from such negative evidence. COMSOL itself and the real sentinel are authoritative for model-tree and runtime behavior.

The steady template is the distinct reduced profile template. Its reduction, Study-1 equivalence to the transient superset, Air-material evaluation, solver settings, mappings, and export behavior remain runtime-unverified in this environment. The profile stays `template_ready: false` until that audit succeeds.

## Geometry, grid, and stationary airflow

The frozen two-dimensional geometry and output grid are:

```text
Lx = 1.2 m       Nx = 401       dx = Lx/(Nx-1) = 0.003 m
Ly = 0.75 m      Ny = 251       dy = Ly/(Ny-1) = 0.003 m
```

Axes include both boundaries, increase from zero, and form tensors in `[ny, nx] = [251, 401]` order. The base COMSOL mesh contract is 400 elements in x and 250 in y before the existing boundary-layer refinement. Output points and mesh elements remain distinct concepts; no redundant output/element count parameters are introduced. `Lz = 0.8 m` is package-fixed geometry/depth provenance, not a runtime CSV field or neural channel. Its use must be identical wherever Study 1 or a stationary export depends on it; transient-only volume diagnostics also use it.

The frozen stationary temperature is:

```text
T_flow_ref = 300.65 K  # 27.5 degC
```

`T_flow_ref = 300.65 K`, `p_ref = 101325 Pa`, and `p_out = 0 Pa` are package-fixed scientific provenance and canonical steady-template conditioning. They are not steady runtime-file inputs or neural channels. The transient adapter also persists them in its scalar handoff. `T_flow_ref` is never derived from `T_in_ref`, `T_init`, `T_amb`, or schedule values.

The steady learning contract is:

```text
inputs  = x, y, Kxx, Kxy, Kyy, eps_bed, p_in_bc
outputs = p, u, v
shape   = input float32 [7,251,401], target float32 [3,251,401]
```

The established steady preprocessing retains log10 diagonal permeability and normalized cross-permeability representations. Their identity remains bound to the renamed raw field `p_in_bc` and the task contract.

Stationary equivalence between profiles requires the same two-dimensional `Lx`/`Ly` geometry and output grid, mesh, Brinkman/Stokes/incompressible/no-gravity/no-slip physics, selections, Air material evaluation, solver, fixed parameters, permeability tensor, stationary exports, and identical `Lz` use wherever Study 1 depends on it. Transient-only drying integrals do not enter the stationary output comparison. These COMSOL-internal facts are runtime-unverified until the real equivalence sentinel.

## Exact case-local adapters

All adapters are semicolon-delimited and relative to the disposable case directory. The only external boundary names are `p_in_bc`, `T_in_bc`, `omega_in_bc`, and `phi_in_bc`. The corresponding COMSOL file-function contract is `p_in_bc_file`, `T_in_bc_file`, `omega_in_bc_file`, and `phi_in_bc_file`. Their actual bindings and activation are runtime-unverified.

### Standalone `steady_flow`

`fields.csv` contains exactly:

```text
x;y;Kxx;Kxy;Kyy;eps_bed;p_in_bc
m;m;m^2;m^2;m^2;1;Pa
```

This is the complete steady runtime input. The steady profile creates no `scalars.csv`, `schedule.csv`, `X_0_db_field`, transient dummy, or drying scalar. `T_flow_ref`, `p_ref`, and `p_out` remain exact package-fixed values in the canonical template and in case, provenance, HDF5, and package identity; they are not a file-backed steady handoff.

### `transient_drying`

`fields.csv` contains exactly:

```text
x;y;Kxx;Kxy;Kyy;eps_bed;p_in_bc;X_0_db_field
m;m;m^2;m^2;m^2;1;Pa;kg/kg
```

`scalars.csv` uses `name;value;unit` and the exact ordered rows:

```text
T_flow_ref [K]            p_ref [Pa]                  p_out [Pa]
T_init [K]                T_amb [K]                  T_in_ref [K]
eps_bed_cal_ref [1]       rho_bu_dry_ref [kg/m^3]   k_gr [W/(m*K)]
cp_gr_dry [J/(kg*K)]      X_target_wb [1]            r_surf_0 [1/s]
r_int_surf [1]            f_surf [1]                 A_osw [1]
B_osw [1/K]               C_osw [1]                  f_wet_dm_max [1]
```

`T_in_ref` remains the full planned 0–168 h schedule mean and scalar provenance; it does not own stationary `T_flow_ref`.

`schedule.csv` contains exactly:

```text
t;T_in_bc;omega_in_bc;phi_in_bc
h;K;kg/kg;1
```

Python generation is the only owner of `omega_in_bc -> phi_in_bc`. The exact package-fixed `p_ref` used by the conversion is persisted. `omega_in_bc` remains complete schedule provenance and a plausibility/analysis quantity; it is not a second moisture-PDE boundary condition and is not a baseline boundary channel.

Every adapter is generated fresh per case, hashed into `case.json`, and revalidated before conversion. The scalar parser rejects missing, duplicate, unknown, obsolete, reordered, mis-unitized, non-finite, or provenance-inconsistent entries. Whether COMSOL actually consumes every intended row and reloads it for every case must be proven by the two-case real sentinel.

## Frozen transient physics

The two moisture states are intensive compartment concentrations. The permanent storage fraction is `f_surf`:

```text
f_surf*d(w_surf)/dt       = j_int - m_evap
(1-f_surf)*d(w_int)/dt    = -j_int
w_gr                       = f_surf*w_surf + (1-f_surf)*w_int
w_gr_0                     = rho_bu_dry*X_0_db_field
w_surf(0)                  = w_gr_0
w_int(0)                   = w_gr_0
r_surf                     = r_surf_0
r_int                      = r_int_surf*r_surf
j_int                      = (1-f_surf)*r_int*(w_int-w_surf)
m_evap                     = f_surf*r_surf*max(w_surf-w_eq,0)
d(w_gr)/dt                 = -m_evap
```

There is no unweighted `w_surf + w_int` alternative and no initial split by `f_surf`.

Moisture, density, heat capacity, conductivity, and Oswin conventions are:

```text
X_db       = w_gr/rho_bu_dry
X_wb       = X_db/(1+X_db) = w_gr/(rho_bu_dry+w_gr)
rho_bu_dry = rho_bu_dry_ref*(1-eps_bed)/(1-eps_bed_cal_ref)

solid phase density       = rho_bu_dry/(1-eps_bed)
solid phase heat capacity = cp_gr_eff
cp_gr_eff                 = cp_gr_dry + X_db*cp_w
volumetric contribution   = rho_bu_dry*cp_gr_eff

k_eff = k_gr*(2*k_gr+k_air-2*eps_bed*(k_gr-k_air))
              /(2*k_gr+k_air+eps_bed*(k_gr-k_air))

phi_eff = min(max(phi,1e-6),0.999)
X_eq_db = 0.01*(A_osw+B_osw*(T-273.15[K]))
                *(phi_eff/(1-phi_eff))^C_osw
w_eq = rho_bu_dry*X_eq_db

osw_ratio_0 = (100*X_0_db_field
               /(A_osw+B_osw*(T_init-273.15[K])))^(1/C_osw)
phi_init = osw_ratio_0/(1+osw_ratio_0)
```

The factor `0.01`, Celsius shift, dry-basis convention, and `B_osw [1/K]` are part of the contract. `k_air` comes from the Air material, and `k_eff` is the single packed-bed equivalent conductivity. Do not mix the same gas and solid conductivities again.

Source signs are:

```text
Moisture Source = +m_evap
Q_evap          = -h_fg*m_evap
```

Positive evaporation removes granular water, adds gas-phase moisture, and supplies a negative latent-heat sink. Integrated `m_dot_evap` is internal phase transfer and is not an external total-water balance term.

These formulas are the repository contract. Their actual COMSOL expressions, variables, physics nodes, and activation state are runtime-unverified until the real sentinel.

## Diagnostics, stop, and time ownership

The complete global series order is:

```text
t [h]
X_wb_bulk [1]
f_wet_dm [1]
m_w_gr [kg]
m_v_gas [kg]
m_dot_evap [kg/s]
m_dot_v_in [kg/s]
m_dot_v_out [kg/s]
mt_mass_balance [kg/s]
T_out_mean [K]
phi_out_mean [1]
```

`m_dot_v_in` and `m_dot_v_out` use positive external inflow/outflow signs; `abs()` must not hide an orientation error. Native `mt.massBalance` maps externally to `mt_mass_balance` and is QA, not a neural channel. Total water is checked as `d/dt(m_w_gr+m_v_gas) = m_dot_v_in-m_dot_v_out`, with hours converted to seconds. No scientific pass tolerance is encoded; real validation reports observed residuals.

The stop contract is `f_wet_dm_max = 0.05`, condition `comp1.f_wet_dm <= f_wet_dm_max`, and COMSOL output `Step after stop`. Final Status is exactly:

```text
t_final [h]
f_wet_dm_final [1]
X_wb_bulk_final [1]
X_wb_max_final [1]
T_min_final [K]
T_max_final [K]
phi_min_final [1]
phi_max_final [1]
```

Python classifies solution states as regular `k*1 h` states within
`16*float64_epsilon*168 h`, a floating-point classification tolerance only, plus at most one optional final irregular exact-stop state. An exact regular-hour stop remains regular. The optional irregular state is diagnostic and never creates a training transition. The last row is never dropped unconditionally.

## Canonical HDF5 and publication

`case.h5` schema version 1 is the sole canonical numeric case payload:

```text
/
├── coords/{x,y}                         float64
├── static/fields                        float32, stored once
├── stationary_fixed/values              float64 package-fixed [steady]
├── scalar/values                        float64 handoff         [transient]
├── provenance/*                         canonical JSON records
├── time                                 float64 regular hours       [transient]
├── transient/fields                     float32 absolute states     [transient]
├── exact_stop/{time,fields}             optional diagnostic         [transient]
├── schedule/values                      float64 complete schedule   [transient]
├── global/values                        float64 complete series     [transient]
└── final_status/values                  float64 actual final point  [transient]
```

The artifact binds schema/converter identity, source profile, case identities, scientific digest, canonical template path and sidecar digest, input/export hashes, package-fixed ownership, exact field/unit orders, and profile-specific content. Only transient cases bind the scalar handoff, schedule conversion pressure, regular-time classification, optional stop state, globals, and final status. Field data are finite and use the configured gzip/shuffle/chunk contract. Any schema, generator, or converter version other than integer `1` is rejected; no migration readers exist.

Conversion validates all required studies/exports, adapters, coordinates, shapes, units, hashes, and formula-level consistency before a private temporary HDF5 file is atomically renamed. With raw retention disabled, successful publication does not keep raw CSV exports. Failure evidence follows the existing isolated failure contract.

## Dataset packages and DataLoader runtime

`src/datasets/dataset_packages.py` is the sole package builder. `dataset_factory.py` is the authoritative Dataset/DataLoader factory, and `dataset_transient.py` owns lazy read-only transient HDF5 access. Static fields and trajectories are not duplicated per transition.

The transient physical item is:

```text
state    = T, phi, w_surf, w_int                         float32 [4,251,401]
static   = x, y, u, v, p, eps_bed, rho_bu_dry           float32 [7,251,401]
boundary = T_in_bc(t_n), T_in_bc(t_n+1),
           phi_in_bc(t_n), phi_in_bc(t_n+1), T_amb       float32 [5]
scalars  = r_surf_0, r_int_surf, f_surf, A_osw,
           B_osw, C_osw, k_gr, cp_gr_dry                float32 [8]
target   = delta_T, delta_phi, delta_w_surf, delta_w_int float32 [4,251,401]
dt       = 1 h
```

`omega_in_bc` remains schedule provenance. `Kxx`, `Kxy`, `Kyy`, `p_in_bc`, and `X_0_db_field` remain source/ablation provenance rather than baseline transient inputs. Absolute states remain in HDF5; only consecutive regular states form increments.

Case-level membership precedes temporal expansion, so every transition from one case has one membership. ID/OOD physical and simulation identities are checked for leakage. Duplicate-source selection is deterministic and explicit. Steady normalizers are fit only on ID-training membership and remain identity-bound.

Transient files open lazily and read only requested slices. Serialization discards handles, PID changes close inherited state, every worker owns process-local handles, positive cache capacity uses bounded LRU eviction with close, zero capacity closes after each item, and Dataset cleanup closes remaining handles. Loader configuration validates batch size, worker count, pinning, persistence, prefetch, shuffle/sampler ownership, `drop_last`, and HDF5 cache capacity. `drop_last` affects incomplete DataLoader batches only; it is unrelated to simulation stop rows.

Package commands are:

```bash
./scripts/generation_workflow.sh build-datasets "$GENERATION_RUN_ID"
python -m src.datasets.dataset_packages inspect <dataset-id> --storage-root /workspace/storage
python -m src.datasets.dataset_packages smoke <dataset-id> --storage-root /workspace/storage --membership train --num-workers 0
python -m src.datasets.dataset_packages smoke <dataset-id> --storage-root /workspace/storage --membership train --num-workers 2 --persistent-workers --prefetch-factor 1
```

## Native CPU execution, operations, and real sentinel

The GPU/development-host checkout and scientific storage are sibling owners:

```text
<project-parent>/repo
<project-parent>/storage
```

Maintained containers expose the same paths as `/workspace/repo` and `/workspace/storage`. `STORAGE_ROOT` defaults to the canonical repository sibling after path resolution; generated data are never placed in the repository. GPU `01_generation` is the retained high-fidelity simulation archive, GPU `02_datasets` is the immutable learning-view layer, and `03_experiments` owns training, tuning, and analysis runs. Disposable GPU collection state is restricted to `01_generation/.state/transfer-staging`.

The binding simulation path is native on `sricehpc01`: `Python/3.10`, a persistent virtual environment, `Comsol/v6.4`, and Slurm. The workflow is run on the GPU/development host outside the container. COMSOL and the GPU queue are not used for dataset construction. The remote persistent layout is `$HOME/grainlegumes-generation/{repo,storage,venv}`; Slurm-local `TMPDIR`/`mktemp` paths remain disposable compute scratch.

Set the exact campaign and reviewed resource request:

```bash
cd /path/to/project-parent/repo
export GENERATION_COMMIT="$(git rev-parse HEAD)"
export GENERATION_CAMPAIGN="configs/generation/campaigns/transient_drying/lentil_pilot.yaml"
export GENERATION_BATCH="transient_drying__lentil__natural"
```

One-time CPU setup is dry-run first, then explicit execution:

```bash
./scripts/generation_workflow.sh setup-cpu   --cpu-host sricehpc01 --git-commit "$GENERATION_COMMIT"
./scripts/generation_workflow.sh setup-cpu   --cpu-host sricehpc01 --git-commit "$GENERATION_COMMIT" --execute
```

The normal foreground lifecycle is one command:

```bash
./scripts/generation_workflow.sh all "$GENERATION_CAMPAIGN"   --cpu-host sricehpc01 --git-commit "$GENERATION_COMMIT"   --only-batch "$GENERATION_BATCH" --wall-time 01:00:00   --max-nodes 1 --cases-per-node 1 --cores-per-case 8   --max-parallel-cases 1
```

Before launch, the resolved plan prints `CPU source cleanup after complete success: enabled`. `all` validates the clean local checkout and exact remote commit/setup, prints the non-mutating campaign plan, launches the deterministic run ID, waits for terminal Slurm and batch/case evidence, collects and revalidates every file/hash through marked staging, atomically publishes GPU `01_generation`, builds or exactly reuses every declared package, inspects every package, runs bounded Dataset/DataLoader smokes, writes the durable pre-cleanup workflow receipt, removes the authorized CPU source, records cleanup, validates the terminal receipt, and returns success. No run ID is copied between these foreground steps.

To retain the CPU source, use the sole normal-lifecycle opt-out:

```bash
./scripts/generation_workflow.sh all "$GENERATION_CAMPAIGN"   --cpu-host sricehpc01 --git-commit "$GENERATION_COMMIT"   --only-batch "$GENERATION_BATCH" --wall-time 01:00:00   --max-nodes 1 --cases-per-node 1 --cores-per-case 8   --max-parallel-cases 1 --keep-cpu-source
```

The resolved plan then prints `CPU source cleanup after complete success: disabled`. `--detach` submits and returns without claiming collection, package building, loader validation, or cleanup; it prints the exact `resume` command for the immutable run.

The independently callable primitives are:

```bash
./scripts/generation_workflow.sh preflight "$GENERATION_CAMPAIGN"   --cpu-host sricehpc01 --git-commit "$GENERATION_COMMIT"   --only-batch "$GENERATION_BATCH"   --max-nodes 1 --cases-per-node 1 --cores-per-case 8 --max-parallel-cases 1

./scripts/generation_workflow.sh plan "$GENERATION_CAMPAIGN"   --cpu-host sricehpc01 --git-commit "$GENERATION_COMMIT"   --only-batch "$GENERATION_BATCH" --wall-time 01:00:00   --max-nodes 1 --cases-per-node 1 --cores-per-case 8 --max-parallel-cases 1

./scripts/generation_workflow.sh launch "$GENERATION_CAMPAIGN"   --cpu-host sricehpc01 --git-commit "$GENERATION_COMMIT"   --only-batch "$GENERATION_BATCH" --wall-time 01:00:00   --max-nodes 1 --cases-per-node 1 --cores-per-case 8 --max-parallel-cases 1
```

`launch` prints the immutable campaign-run ID. With that exact returned ID assigned to `GENERATION_RUN_ID`, advanced operations are:

```bash
./scripts/generation_workflow.sh status "$GENERATION_RUN_ID" --cpu-host sricehpc01
./scripts/generation_workflow.sh accounting "$GENERATION_RUN_ID" --cpu-host sricehpc01
./scripts/generation_workflow.sh cancel "$GENERATION_RUN_ID" --cpu-host sricehpc01
./scripts/generation_workflow.sh validate "$GENERATION_RUN_ID" --cpu-host sricehpc01
./scripts/generation_workflow.sh collect "$GENERATION_RUN_ID" --cpu-host sricehpc01
./scripts/generation_workflow.sh build-datasets "$GENERATION_RUN_ID"
./scripts/generation_workflow.sh resume "$GENERATION_RUN_ID" --cpu-host sricehpc01
```

`collect` is always non-destructive: it uses neither `rsync --delete` nor `rsync --remove-source-files`, publishes only after exact inventory validation, removes marked GPU staging only after publication, and retains the CPU source. `build-datasets` independently builds or validates/reuses every package declared by terminal campaign evidence, then performs package inspection and a bounded loader smoke. `resume` reconstructs scheduler state, resumes incomplete validated generation only when prior attempts are inactive, reuses an existing valid GPU publication and package identities, and continues at the first incomplete gate. Repeating a successful lifecycle validates the existing terminal receipt and performs no transfer, build, or cleanup again.

`status` is the sole host workflow status operation. It reports GPU generation/package/staging sizes, packages by view and regime, missing transient sources, and CPU run/source sizes, scheduler state, failure/incomplete state, transfer state, cleanup eligibility, and reclaimable bytes.

Standalone CPU cleanup is a dry run unless explicitly confirmed:

```bash
./scripts/generation_workflow.sh cleanup "$GENERATION_RUN_ID" --cpu-host sricehpc01
./scripts/generation_workflow.sh cleanup "$GENERATION_RUN_ID" --cpu-host sricehpc01 --confirm
```

It requires the validated GPU transfer receipt, dataset receipt, package inspections, loader smokes, and durable pre-cleanup workflow receipt. The dry run prints exact run-owned paths and bytes. Confirmation revalidates the inactive, successful remote run and the complete authorized source inventory, then transactionally removes all eligible directories or rolls back all moved directories on failure. If the process is interrupted, a confirmed retry authenticates the durable transaction: it restores a partially executed planned detach before retrying, or finishes disposal and receipt publication after all source directories were already detached. An unconfirmed dry run never advances an interrupted cleanup transaction.

After successful default cleanup, CPU batch `meta`, `raw`, and `processed` directories for that exact run are removed. The CPU campaign directory remains compact and retains run ID, Git commit, scheduler/execution history, terminal evidence, destination identity/hashes/bytes, and cleanup receipt. The CPU repository, virtual environment, templates, storage root, other runs, active/incomplete/failed/untransferred runs, and failure evidence remain. GPU transfer staging is removed; GPU `01_generation`, `02_datasets`, and `03_experiments` remain. Transient packages continue to resolve storage-relative `case.h5` sources in GPU `01_generation`.

If any gate fails, no persistent CPU source directory is deleted, no cleanup-complete state is recorded, valid immutable GPU publications/packages remain, and the command exits nonzero with the failed stage, retained CPU bytes, and exact resume command. Cleanup occurs only after terminal generation, remote validation, transfer and destination revalidation, every package build/reuse, every inspection, every loader smoke, and durable receipt publication.

The command effects are:

| Command | Persistent result | Slurm work | Deletion capability |
| --- | --- | --- | --- |
| `setup-cpu` | Dry-run plan or exact checkout/venv/storage setup with `--execute` | None | None |
| `preflight` | Compact non-solving environment evidence | One short probe | Marked temporary probe only |
| `plan` | None | None | None |
| `launch` | Immutable run intent/history and scheduler logs | One bounded worker array | None |
| `status` | Read-only GPU/CPU storage and lifecycle report | None | None |
| `collect` | Exact GPU generation publication and transfer receipt | None | Marked GPU transfer staging only; never CPU source |
| `build-datasets` | All declared immutable packages and dataset-gate receipt | None | None |
| `resume` | First-incomplete-stage continuation | Only when incomplete generation requires a new inactive-attempt pool | CPU source only after all gates |
| `cleanup` | Dry-run report or compact cleanup receipt | None | Exact authorized CPU run source only with `--confirm` |
| `all` | Terminal workflow receipt across every distinct stage | One bounded worker array unless already launched | Default verified CPU cleanup after all gates |

The preflight reports environment readiness separately from production configuration readiness. A hash pass remains `runtime_unverified`; it never turns on `template_ready` or proves a file binding, study, export, isolation, or physical result.

Current production configurations are deliberately non-executable. After evidence, counts/seeds, exact export mappings, `template_ready`, and both exhaustive stationary-conditioning audits are resolved through COMSOL, the direct real sentinel commands are:

```bash
export STORAGE_ROOT="$HOME/grainlegumes-generation-results"
export COMSOL_SENTINEL_PARENT="${TMPDIR:-/tmp}"
export COMSOL_SENTINEL_WORK="$(mktemp -d "${COMSOL_SENTINEL_PARENT%/}/vp2-comsol-sentinel.XXXXXX")"

python -m src.generation.cli.cli_generation validate-config \
  configs/generation/campaigns/transient_drying/lentil_pilot.yaml \
  --only-batch transient_drying__lentil__natural

python -m src.generation.cli.cli_generation run-case \
  configs/generation/campaigns/transient_drying/lentil_pilot.yaml 1 \
  --only-batch transient_drying__lentil__natural \
  --cores-per-case 8 --scheduler-kind slurm \
  --storage-root "$STORAGE_ROOT" --work-root "$COMSOL_SENTINEL_WORK"

python -m src.generation.cli.cli_generation run-case \
  configs/generation/campaigns/transient_drying/lentil_pilot.yaml 2 \
  --only-batch transient_drying__lentil__natural \
  --cores-per-case 8 --scheduler-kind slurm \
  --storage-root "$STORAGE_ROOT" --work-root "$COMSOL_SENTINEL_WORK"

python -m src.generation.cli.cli_generation validate-case \
  configs/generation/campaigns/transient_drying/lentil_pilot.yaml 1 \
  --only-batch transient_drying__lentil__natural \
  --storage-root "$STORAGE_ROOT"

python -m src.datasets.dataset_packages build \
  configs/generation/campaigns/transient_drying/lentil_pilot.yaml \
  --storage-root "$STORAGE_ROOT"

rmdir "$COMSOL_SENTINEL_WORK"
```

Run these direct sentinel commands only inside an interactive Slurm allocation. Use two scientifically valid but distinguishable sentinel cases already produced by the resolved campaign sampler; do not override values outside registry bounds. Then smoke both published views with the package CLI at zero and multiple workers.

The sentinel must prove fresh fields, schedule, and scalar reload; fresh Study 1 per case; Study 2 dependency on the same-case Study-1 solution; no airflow re-solve in Study 2; failure gating; isolated work directories; complete exports; grid/orientation/shape/finite checks; initial weighted moisture identities; stop/time classification; signs; and observed mass-balance residuals. It must report native `mt_mass_balance` without inventing a pass tolerance.

Run the standalone steady case through the same `run-case` CLI with `configs/generation/campaigns/steady_flow/family_generalization.yaml` and batch `steady_flow__lentil__natural`. Compare its `p`, `u`, and `v` against Study 1 of the transient template under identical generated airflow inputs, reporting maximum absolute/relative differences, RMSE, L2 difference, and the location of the maximum. No dedicated equivalence CLI exists; do not claim equivalence until those values are calculated from real published outputs.

## Manual COMSOL 6.4 audit

Open reviewed working copies in COMSOL; never binary-patch either template.

1. Verify the exact boundary/file-function/project-variable names and each case-local file binding.
2. Verify fresh scalar assignment and schedule reload, explicit units, interpolation, and solver-time-to-hours conversion.
3. Verify Study 1 stationary physics, Air material, mesh, selections, fixed `T_flow_ref/p_ref/p_out`, solver, and exact stationary exports.
4. In the transient superset, verify weighted ODE storage, initial states, Oswin/heat/conductivity/source expressions, Study-1-to-Study-2 solution ownership, stop event, globals, Final Status, and export mappings.
5. In the reduced steady template, retain only Study-1-relevant geometry, fixed airflow parameters, input functions/variables, Air material, Brinkman physics, stationary study/solver, mesh, and stationary exports. Remove drying-only schedules, initial moisture, heat/moisture/ODE physics, stop/transient nodes, drying couplings/results/exports only after tracing dependencies.
6. Populate profile YAML patterns/headers and the exhaustive conditioning records only from this COMSOL audit.
7. Save any approved model change normally in COMSOL, regenerate the matching sidecar from exact bytes, and rerun both static validation and the real sentinel.

Until these steps and the full real chain succeed, COMSOL internals and steady/transient stationary equivalence remain runtime-unverified.

## Remaining production blockers

- Resolve material taxonomy, parameter evidence/ranges, coupled records, and validity domains.
- Resolve operation ranges, campaign counts/seeds/memberships, wall time, and approved resources.
- Complete both COMSOL audits, export mappings, fixed stationary-conditioning records, two-case reload sentinel, and steady equivalence comparison.
- Keep common/material/operation/campaign configurations non-executable and both profiles fail-closed until all owners validate together.
