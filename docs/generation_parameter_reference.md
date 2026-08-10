# VP2 Generation Parameter and Config Reference

YAML and Python validation are the executable authority. This reference gives
the compact scientific contract, canonical names and units, ownership rules,
and inspection commands. Numeric material records and their source evidence
remain in `configs/generation/materials/<family>.yaml`; they are not duplicated
here.

## Scientific parameter note

> The configured values and ranges are executable modelling and sampling
> decisions, not universal experimentally validated material constants. A value
> may be directly reported, refitted, convention- or unit-converted, transferred,
> engineering-calculated or inverted, selected as a calibration prior, or
> selected as synthetic generator design. A citation therefore does not imply
> that the final configured number appears directly in that source. The
> machine-readable `status`, `derivation`, `confidence`, and `validity` fields
> define the authoritative interpretation. Software, COMSOL, smoke, and pilot
> passes validate runtime and data flow only; they do not experimentally validate
> the configured science.

## Owners at a glance

| Owner | Scientific decision |
| --- | --- |
| `configs/generation/sources.yaml` | One supplied bibliographic record per source key |
| `configs/generation/registry.yaml` | Decision identity; parameter name, unit, kind, transform, block, sampling order, OOD group, components, derivation |
| `configs/generation/common.yaml` | Grid, time, shared fixed physics, formulas, adapter and HDF5 contracts |
| `configs/generation/operations/<operation>.yaml` | Apparatus and operation supports and constraints |
| `configs/generation/materials/<family>.yaml` | Role-neutral family values, natural supports, coupled records, targets, evidence |
| `configs/generation/profiles/<profile>.yaml` | Template, export schema, explicit native mappings, profile conditioning |
| `configs/generation/campaigns/<profile>/<campaign>.yaml` | Roles, counts, seeds, memberships, package requests |
| `configs/generation/execution/<site>.yaml` | Execution and retention only; excluded from scientific identity |

Python resolution is the only layer that combines these owners. It validates
cross-layer compatibility, projects the registry into the selected profile,
derives identities and allocations, and persists the effective scientific
configuration. Documentation is not a second parameter registry.

## Inspect the resolved scientific contract

Show the complete fail-closed campaign view without changing unresolved launch
state:

```bash
cd /workspace/repo
python -m src.generation.cli.cli_generation validate-config \
  configs/generation/campaigns/transient_drying/family_generalization.yaml \
  --allow-incomplete
```

The output exposes the effective material inventory and roles, source-case
counts and derived total, memberships, campaign and derived seed plan, package
requests and expanded package inventory, profile-projected OOD units and exact
allocation, purpose-specific pilot or smoke scope, bounded static-sentinel
workload, template identity, execution resources, and readiness gates.

Inspect individual values or complete atomic records through their inherited
provenance chain:

```bash
python -m src.generation.cli.cli_generation validate-config \
  configs/generation/campaigns/transient_drying/family_generalization.yaml \
  --allow-incomplete \
  --inspect-parameter A_osw \
  --inspect-parameter density_calibration \
  --inspect-parameter cp_w \
  --inspect-parameter grid.dx \
  --inspect-parameter time.stop \
  --inspect-parameter physical_formulas.rho_bu_dry \
  --inspect-parameter packing_porosity_mean_support
```

Inspection distinguishes authored, inherited, selected, and derived evidence.
It also distinguishes generator consumption, scalar-adapter handoff, and fixed
values that Python binds to a hashed native template without setting at runtime.
Use the same command after every valid configuration edit; derived totals,
dimensions, memberships, packages, and identities must change from the edited
owners without synchronized Python or documentation edits.

## Registry, profile, and channel ownership

The registry owns the active sampling blocks, effective dimensions, names,
units, kinds, transforms, physical OOD groups, and atomic-record components.
The profile projection determines which registry entries participate in one
simulation profile. Do not maintain those inventories in prose.

The schema-only public profile contract exposes adapter fields and available
learning views without loading campaign YAML or native templates:

```bash
python - <<'PY'
import json
from src import generation

def fields(items):
    return [field.as_dict() for field in items]

for profile_id in generation.contracts.available_profile_ids():
    contract = generation.contracts.get_profile_contract(profile_id)
    print(json.dumps({
        "profile": contract.id,
        "available_learning_views": contract.available_learning_views,
        "airflow_source": contract.airflow_source,
        "coordinate_fields": fields(contract.coordinate_fields),
        "stationary_fixed_fields": fields(contract.stationary_fixed_fields),
        "static_fields": fields(contract.static_fields),
        "transient_fields": fields(contract.transient_fields),
        "schedule_fields": fields(contract.schedule_fields),
        "scalar_inputs": fields(contract.scalar_inputs),
    }, indent=2))
PY
```

Dataset task and step contracts remain the authoritative tensor-channel owners.
Generation profile fields are source and provenance contracts; they must not be
copied into a separate learning-channel table. A material family is metadata,
not an implicit one-hot channel, and an evaluation category does not create a
new model, equation, sampling coordinate, or native profile.

## Scientific bindings

Exact formula strings, numerical fixed values, grids, time nodes, storage
settings, and provenance records are owned by `common.yaml` and displayed by
parameter inspection. The stable interpretation is:

- granular water is split between surface and internal compartments;
- internal transfer exchanges those compartments, while evaporation removes
  surface water toward the configured equilibrium relation;
- dry-basis and wet-basis moisture remain explicitly distinguished;
- dry bulk density couples its selected reference record to the realized
  porosity field;
- effective heat capacity and conductivity couple material and bed state;
- latent heat closes the thermal source term;
- total-water diagnostics compare stored water change with inlet and outlet
  vapour flow.

These relationships are executable bindings, not a claim that one cited source
reported every final configured coefficient or range.

### Material-calibrated porosity

There is one physical porosity field. Its scalar calibration reference comes
from the active complete density-calibration record and is not another field.
The material YAML owns the natural representative-porosity support; common
configuration owns pointwise numerical guards.

The Kozeny-Carman anchor deterministically couples the selected permeability,
calibration reference, and sampled anchor factor. Natural sampling resolves a
case- and material-specific conditional support. Parameter OOD resolves disjoint
transformed tails using the registry-owned separation policy. Tail separation
and width are not restated here; inspect `porosity.kc_anchor_factor` and the
resolved `parameter_ood` allocation for the exact active contract.

Local morphology is generated from the shared background field and the resolved
porosity texture settings. Permeability and porosity share spatial structure,
but realized local permeability components do not directly become porosity
texture inputs. The same realized porosity is consumed by airflow, transient
transport and heat transfer, material density, water inventory, native solves,
HDF5, and maintained dataset views.

### Inlet schedule

The schedule contract and constraints come from the common and operation owners.
The generator validates the complete schedule, including temperature, humidity,
source-air, and interpolation rules, and retries it deterministically as one
unit. It does not clip individual time samples. Schedule diagnostics, names, and
units come from the schedule/profile contracts rather than a documentation
column list.

## Material records and atomic selection

Material files are role-neutral. Their schema and evidence validation are owned
by the material resolver; campaign files assign roles and memberships. Each
independent value or complete record retains its source reference, evidence
status, derivation, confidence, and validity. Runtime smoke or pilot success
never upgrades that scientific evidence.

Atomic selection rules are stable:

- density OOD selects one complete configured density-calibration record;
- sorption selection keeps one complete record with its equation convention and
  validity domain;
- kinetics selection keeps one complete kinetic record;
- simplex-valued schedule weights remain one complete constrained vector;
- components from different records are never mixed;
- parameter OOD selects a scalar tail or a complete atomic record, never an
  accidental component-wise hybrid.

## Family and parameter OOD

Campaign roles determine natural-support learning and evaluation eligibility.
Inspect `material_roles`, `material_memberships`, and
`dataset_package_inventory` for the current families and regimes. Family OOD
uses natural material support only and never invokes parameter OOD.

Each parameter-OOD case activates one profile-eligible unit. Eligibility comes
from registry classification, profile projection, parameter kind, and actual
configured tails or alternate records. Deterministic round-robin allocation
covers every eligible unit when capacity permits and balances remaining cases.
The resolved summary and persisted provenance provide, per batch:

- eligible unit identifiers, physical groups, and unit kinds;
- per-unit allocation counts and exact per-case assignment;
- selected record or tail, realized value, natural and OOD support, transform,
  and transform-space distance at case generation time.

Scalar tails are nonempty, separated from natural support in the declared
transform, and fail closed when invalid. Natural cases have no active OOD unit.

## Adapters and persistence

Grid shape, geometry, regular times, exact-stop handling, adapter fields, export
roles, units, HDF5 layout, chunks, compression, and tolerances are resolved from
common and profile configuration. Use `validate-config`, parameter inspection,
and the public profile contract instead of copying their current values into
prose.

Canonical HDF5 binds scientific and case identities, material role, evaluation
and sampling regimes, natural-support state, sampled values and units, seed and
block evidence, complete OOD and coupled-record provenance, inputs, outputs,
diagnostics, hashes, template identity, source-export identity, and the embedded
resolved scientific configuration. Dataset package admission validates this
evidence against the requested source role and regime.

## Source provenance

`configs/generation/sources.yaml` is the sole bibliographic catalogue. Parameter,
material, operation, common, and profile records refer to its keys; the resolver
rejects unknown references. Full citations, identifiers, and locators therefore
remain in one machine-readable owner rather than a generated Markdown table.

Use `--inspect-parameter` to follow an executable value to its exact source
record and evidence interpretation. Supporting records with no active executable
reference remain visible in `sources.yaml` without speculative assignments in
documentation.

## Scientific and runtime gates

Run the bounded, non-COMSOL scientific checks:

```bash
python -m src.generation.cli.cli_generation static-sentinels \
  configs/generation/campaigns/steady_flow/family_generalization.yaml \
  configs/generation/campaigns/transient_drying/family_generalization.yaml
```

Show current production, scientific, mapping, and runtime blockers:

```bash
python -m src.generation.cli.cli_generation readiness-report \
  configs/generation/campaigns/steady_flow/family_generalization.yaml \
  configs/generation/campaigns/transient_drying/family_generalization.yaml \
  --run-static-sentinels
```

Technical checks validate software, native mappings, conversion, persistence,
and data flow. They do not experimentally validate configured science. For
execution, transfer, smoke, pilot analysis, resume, and cleanup, use the
[generation workflow](simulation_generation.md). The project gateway is the
[README](../README.md).
