"""Resolve and validate layered scientific and execution generation configurations.

Responsibilities:
  - Resolve common, material-family, profile, dataset, and execution YAML owners
  - Validate authoritative grid, time, adapter, storage, split, and runtime contracts
  - Derive deterministic scientific and exact case-input identities

Design principles:
  - Scientific and execution settings remain physically and cryptographically separate
  - Resolved scientific configuration contains no inheritance or unresolved defaults
  - Unknown, ambiguous, or non-executable configuration fails before case preparation

This module does NOT:
  - Invent material ranges, coupled sets, production counts, or COMSOL mappings
  - Generate samples or fields, execute COMSOL, or modify templates

Notes:
  Execution node-job contract:
    - The authored groups run, rules, node_job, site and internals resolve into the runtime, submission, cluster and
      site values that consumers read; the resolved cluster holds the derived fields concurrent_cases_per_node
      (site.cores_per_node // cores_per_case) and node_job_memory (<int>G, concurrent_cases_per_node
      * memory_per_case + node_memory_overhead), next to the copied cores_per_node and partition
    - A node job above node_memory_limit, or one whose node_job_wall_time does not exceed
      timeout_seconds plus 600 s, is rejected; the concurrency is never reduced silently
    - The COMSOL batch licence arguments and the scheduler allocation options are pipeline-owned
    - A Dataset plan's campaign without an explicit max_admission_cases resolves submission.max_admission_cases to null
      (derived from its rates); every other campaign keeps the validated positive integer

"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

import yaml

from src import common
from src.generation.contracts import generation_contracts_campaign_identity as campaign_identity
from src.generation.contracts import generation_contracts_materials as materials
from src.generation.contracts import generation_contracts_profiles as profiles
from src.generation.contracts import generation_contracts_provenance as provenance_service
from src.generation.contracts import generation_contracts_source as source_service
from src.generation.contracts import generation_contracts_templates as templates
from src.generation.contracts.generation_contracts_vocabulary import (
    CAMPAIGN_PURPOSES,
    DESIGN_KINDS,
    MATERIAL_ROLES,
)

from . import generation_cases_sampling as sampling_service
from . import generation_cases_schedule as schedule_service
from . import generation_cases_seeding as seeding

_MAX_FILESYSTEM_COMPONENT_BYTES = 255
CASE_ID_WIDTH = 4
_MAXIMUM_GZIP_COMPRESSION_LEVEL = 9
_MAXIMUM_TRANSIENT_INITIAL_STATE_RTOL = 1.0e-4
_MAXIMUM_TRANSIENT_INITIAL_STATE_ATOL = 1.0e-10
_MAXIMUM_TRANSIENT_BULK_MOISTURE_RTOL = 1.0e-5
_MAXIMUM_TRANSIENT_BULK_MOISTURE_ATOL = 1.0e-9
_WALL_TIME_PATTERN = re.compile(r"([0-9]+):([0-5][0-9]):([0-5][0-9])")
_SLURM_MEMORY_PATTERN = re.compile(r"([1-9][0-9]*)([MGT])")
_SLURM_MEMORY_UNIT_MIB = {"M": 1, "G": 1_024, "T": 1_024 * 1_024}
# Per-user cap of node jobs (slots gen-seat-1 to gen-seat-4) on the five shared 32-core standard nodes, so one node always stays free for
# others; a cluster-share limit, the COMSOL batch licence counts once per user. The single source of the slot range for submission and records.
MAXIMUM_NODE_JOBS: Final = 4
# A fresh node job must fit one complete case timeout plus its publication.
_NODE_PUBLICATION_MARGIN_SECONDS = 600
_MAXIMUM_NODE_JOB_WALL_SECONDS = 24 * 3_600
_FINAL_PHYSICAL_FORMULAS = {
    "w_surf_balance": "f_surf*d(w_surf)/dt = j_int - m_evap",
    "w_int_balance": "(1-f_surf)*d(w_int)/dt = -j_int",
    "w_gr": "f_surf*w_surf + (1-f_surf)*w_int",
    "w_gr_0": "rho_bu_dry*X_0_db_field; w_surf(0)=w_int(0)=w_gr_0",
    "r_surf": "r_surf_0",
    "r_int": "r_int_surf*r_surf",
    "j_int": "(1-f_surf)*r_int*(w_int-w_surf)",
    "m_evap": "f_surf*r_surf*max(w_surf-w_eq,0)",
    "X_db": "w_gr/rho_bu_dry",
    "X_wb": "w_gr/(rho_bu_dry+w_gr)",
    "X_wb_from_X_db": "X_db/(1+X_db)",
    "X_db_from_X_wb": "X_wb/(1-X_wb)",
    "X_wb_bulk": "integral(w_gr)/(integral(rho_bu_dry)+integral(w_gr))",
    "rho_bu_dry": "rho_bu_dry_ref*(1-eps_bed)/(1-eps_bed_ref)",
    "solid_phase_density": "rho_bu_dry/(1-eps_bed)",
    "cp_gr_eff": "cp_gr_dry + X_db*cp_w",
    "volumetric_heat_capacity": "rho_bu_dry*cp_gr_eff",
    "k_eff": "k_gr*(2*k_gr+k_air-2*eps_bed*(k_gr-k_air))/(2*k_gr+k_air+eps_bed*(k_gr-k_air))",
    "phi_eff": "min(max(phi,1e-6),0.999)",
    "X_eq_db": "0.01*(A_osw+B_osw*(T-273.15[K]))*(phi_eff/(1-phi_eff))^C_osw",
    "w_eq": "rho_bu_dry*X_eq_db",
    "osw_ratio_0": "(100*X_0_db_field/(A_osw+B_osw*(T_init-273.15[K])))^(1/C_osw)",
    "phi_init": "osw_ratio_0/(1+osw_ratio_0)",
    "Q_evap": "-h_fg*m_evap",
    "f_wet_dm": "integral(rho_bu_dry*indicator(X_wb>X_target_wb))/integral(rho_bu_dry)",
    "total_water_balance": "d/dt(m_w_gr+m_v_gas)=m_dot_v_in-m_dot_v_out",
}
# Contract blocks the loader owns instead of mirroring them in YAML: they enter every scientific record from here, with this provenance.
_CODE_CONTRACT_PROVENANCE: Final = {"evidence": "project_baseline", "source_refs": ["transient_comsol_model_report"]}
_FIXED_TIME_FIELDS: Final = {"internal_steps": "adaptive", "irregular_stop_state": "diagnostic_only"}
_FIXED_STORAGE_FIELDS: Final = {"compression": "gzip"}
_INPUT_FILENAMES: Final = {"spatial": "fields.csv", "scalar": "scalars.csv", "schedule": "schedule.csv"}
_INPUT_DELIMITER: Final = ";"
_OPERATION_ID: Final = "fixed_bed"
_OPERATION_CONSTRAINTS: Final = {
    "heater_only": True,
    "humidity_ratio_conserved": "omega_source_air(t)=omega_in_bc(t)",
    "source_relative_humidity": "RH(T_amb,omega_in_bc(t),p_ref)",
    "inlet_relative_humidity": "RH(T_in_bc(t),omega_in_bc(t),p_ref)",
    "infeasible_schedule_policy": "reject_complete_schedule_and_deterministically_resample",
    "porosity_natural_support_policy": "realized_mean_must_retain_natural_or_kappa_mean_ood_state",
}
# Registry coordinates fixed_bed.yaml owns; bed_fields.yaml owns every other operator-owned coordinate (bed, permeability, porosity,
# inlet-pressure shape and initial-moisture structure), and the material files own these.
_OPERATION_BLOCK: Final = "operation"
_MATERIAL_OWNED_PARAMETERS: Final = frozenset(
    {
        "kappa_mean",
        "initial_moisture.mean_db",
        "initial_moisture.amplitude_db",
        "rho_bu_dry_ref",
        "eps_bed_ref",
        "k_gr",
        "cp_gr_dry",
        "X_target_wb",
        "oswin",
        "r_surf_0",
        "r_int_surf",
        "f_surf",
    }
)
_GLOBAL_PARAMETERS: Final = frozenset({"eps_min_global", "eps_max_global"})
# The layered configuration files a campaign (and so a Dataset plan's generation block) references.
_CAMPAIGN_REFERENCES: Final = (
    "sources_config",
    "registry_config",
    "common_config",
    "operations_config",
    "bed_fields_config",
    "profile_config",
    "execution_config",
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_COMSOL_OWNED_ARGUMENTS = (
    "-autosave",
    "-recoverydir",
    "-tmpdir",
    "-inputfile",
    "-job",
    "-outputfile",
    "-nosave",
    "-np",
    "-nn",
    "-nnhost",
    "-mpihosts",
    "-pname",
    "-plist",
    "-pindex",
    "-paramfile",
    "-batchlog",
    "-batchlogout",
    "-usebatchlic",
    "-ckl",
    "-keeplicenses",
)
_SCHEDULER_OWNED_OPTIONS = (
    "--parsable",
    "--nodes",
    "--ntasks",
    "--ntasks-per-node",
    "--cpus-per-task",
    "--array",
    "--chdir",
    "--job-name",
    "--partition",
    "--time",
    "--output",
    "--error",
    "--exclusive",
    "--export",
    "--nodelist",
    "--exclude",
    "--reservation",
    "--qos",
    "--dependency",
    "--comment",
    "--signal",
    "--wrap",
)
# Whole option families: memory (--mem, --mem-per-cpu, ...) is derived from the case steps, and a CPU-only pipeline takes no GPUs.
_SCHEDULER_OWNED_OPTION_PREFIXES = ("--mem", "--gres", "--gpu")


def build_batch_name(
    simulation_profile: str,
    material_family: str,
    batch_kind: str,
) -> str:
    """Build the canonical profile, material, and semantic-kind batch name."""
    components = {
        "simulation_profile": simulation_profile,
        "material_family": material_family,
        "batch_kind": batch_kind,
    }
    for label, value in components.items():
        common.paths.validate_logical_name(value, label=label)
        if "__" in value:
            message = f"{label} cannot contain the canonical batch separator '__'."
            raise ValueError(message)
    return "__".join(components.values())


def build_batch_id(batch_name: str, scientific_config_digest: str) -> str:
    """Build the canonical immutable batch identifier from scientific identity."""
    name = common.paths.validate_logical_name(batch_name, label="batch_name")
    if _SHA256_PATTERN.fullmatch(scientific_config_digest) is None:
        message = "scientific_config_digest must be one lowercase SHA-256 digest."
        raise ValueError(message)
    return f"{name}__{scientific_config_digest[:16]}"


def resolve_batch_locators(
    simulation_profile: str,
    material_family: str,
    sampling_regime: str,
    scientific: Mapping[str, Any],
) -> tuple[str, str]:
    """
    Resolve the canonical batch name and storage-locator regime of one batch.

    Parameters
    ----------
    simulation_profile : str
        Simulation profile identifier of the batch.
    material_family : str
        Material family of the batch.
    sampling_regime : str
        Design kind of the original batch; a replacement keeps its parent's kind.
    scientific : Mapping[str, Any]
        Resolved scientific values whose optional ``replacement_design`` marks a
        canonical replacement.

    Returns
    -------
    tuple[str, str]
        Batch name and storage-locator regime. An original batch yields the plain
        name and ``sampling_regime``; a canonical replacement yields the name
        suffixed ``__replacement_NNNN`` and the regime ``<regime>_replacement_NNNN``.

    Raises
    ------
    ValueError
        If a name component is invalid or a canonical replacement lacks an integer
        ``candidate_ordinal`` of at least one.

    """
    batch_name = build_batch_name(simulation_profile, material_family, sampling_regime)
    design = scientific.get("replacement_design")
    if design is None:
        return batch_name, sampling_regime
    ordinal = design.get("candidate_ordinal") if isinstance(design, Mapping) else None
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 1:
        message = "Canonical replacement design has an invalid candidate ordinal."
        raise ValueError(message)
    return f"{batch_name}__replacement_{ordinal:04d}", f"{sampling_regime}_replacement_{ordinal:04d}"


def scientific_config_identity_payload(scientific: Mapping[str, Any]) -> dict[str, Any]:
    """Return the current semantic configuration identity payload."""
    return _scientific_identity_config(scientific)


def compute_scientific_config_digest(scientific: Mapping[str, Any]) -> str:
    """Compute the current scientific configuration identity."""
    return common.serialization.canonical_json_sha256(scientific_config_identity_payload(scientific))


def compute_case_input_config_digest(scientific: Mapping[str, Any]) -> str:
    """Compute the current adapter input identity."""
    return common.serialization.canonical_json_sha256(_input_identity_config(scientific))


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """One resolved material/regime generation batch plus separate execution."""

    source_path: Path
    profile: profiles.SimulationProfile
    material_family: str
    material_role: str
    evaluation_regime: str
    sampling_regime: str
    batch_name: str
    scientific_values: dict[str, Any]
    execution_values: dict[str, Any]
    template: templates.ResolvedTemplateIdentity
    case_indices: tuple[int, ...]
    seed_base: int | None
    assignments: dict[int, dict[str, Any]]
    scientific_config_digest: str
    case_input_config_digest: str
    batch_identity: str
    batch_id: str
    batch_storage_name: str

    @property
    def template_path(self) -> Path:
        """Return the configured repository template path."""
        return self.template.absolute_path

    @property
    def template_relative_path(self) -> str:
        """Return the configured repository-relative template locator."""
        return self.template.relative_path

    @property
    def template_sha256(self) -> str:
        """Return the verified configured template byte digest."""
        return self.template.sha256

    def case_id(self, case_index: int) -> str:
        """Return the canonical directory identifier for one batch member."""
        if case_index not in self.case_indices:
            message = f"Case index {case_index} is not a member of batch {self.batch_id}."
            raise ValueError(message)
        return f"case_{case_index:0{CASE_ID_WIDTH}d}"

    def case_seed(self, case_index: int) -> int:
        """Return the master seed of one executable case from its design seed and design point index, never its case index."""
        self.case_id(case_index)
        if self.seed_base is None:
            message = f"Batch {self.batch_name!r} has no resolved seed."
            raise ValueError(message)
        return seeding.derive_seed(self.seed_base, "case", str(self.assignments[case_index]["design_index"]))

    def case_assignment(self, case_index: int) -> dict[str, Any]:
        """Return an isolated material, regime, and OOD assignment."""
        self.case_id(case_index)
        return copy.deepcopy(self.assignments[case_index])


@dataclass(frozen=True, slots=True)
class CampaignConfig:
    """One resolved campaign plan with immutable scientific subbatches."""

    source_path: Path
    campaign_name: str
    campaign_digest: str
    package_request_digest: str
    campaign_id: str
    campaign_purpose: str
    domain_profile: str
    evaluation_regimes: tuple[str, ...]
    profile: profiles.SimulationProfile
    template: templates.ResolvedTemplateIdentity
    material_roles: dict[str, tuple[str, ...]]
    material_memberships: dict[str, tuple[str, ...]]
    membership: dict[str, Any]
    source_registry: dict[str, dict[str, Any]]
    source_files: tuple[Path, ...]
    total_case_count: int
    batches: tuple[GenerationConfig, ...]
    dataset_packages: tuple[dict[str, Any], ...]
    duplicate_case_input_policy: str
    execution_values: dict[str, Any]
    root_seed: int
    designs: dict[str, dict[str, int]]
    selected_targets: dict[str, int]
    composition_sources: tuple[dict[str, Any], ...] = ()
    scientific_overrides: Mapping[str, Any] | None = None
    producer_identity: Mapping[str, str] | None = None
    execution_overrides: Mapping[str, Any] | None = None
    source_label: str | None = None
    authored_value: Mapping[str, Any] | None = None

    @property
    def source_name(self) -> str:
        """Return the readable Source folder without changing seed labels."""
        return campaign_identity.source_name(self.source_label or self.campaign_name, self.campaign_digest)

    @property
    def template_path(self) -> Path:
        """Return the configured repository template path."""
        return self.template.absolute_path

    @property
    def template_relative_path(self) -> str:
        """Return the configured repository-relative template locator."""
        return self.template.relative_path

    @property
    def template_sha256(self) -> str:
        """Return the verified configured template byte digest."""
        return self.template.sha256

    @property
    def material_inventory(self) -> tuple[str, ...]:
        """Return configured materials in canonical role and authored item order."""
        return tuple(material_family for role in MATERIAL_ROLES for material_family in self.material_roles[role])

    def batch(self, batch_name: str) -> GenerationConfig:
        """Select one predeclared generation batch by its human-readable name."""
        for batch in self.batches:
            if batch.batch_name == batch_name:
                return batch
        available = ", ".join(item.batch_name for item in self.batches)
        message = f"Batch {batch_name!r} is not declared by campaign {self.campaign_name!r}. Available: {available}."
        raise ValueError(message)

    def find_batch(
        self,
        *,
        material_family: str,
        sampling_regime: str,
        simulation_profile: str | None = None,
    ) -> GenerationConfig | None:
        """Return one structurally selected batch, or ``None`` when absent."""
        profile_id = self.profile.id if simulation_profile is None else simulation_profile
        matches = tuple(
            batch
            for batch in self.batches
            if batch.profile.id == profile_id and batch.material_family == material_family and batch.sampling_regime == sampling_regime
        )
        matches = tuple(batch for batch in matches if not isinstance(batch.scientific_values.get("replacement_design"), dict))
        if len(matches) > 1:
            message = (
                "Campaign batch identities are ambiguous for "
                f"profile={profile_id!r}, material_family={material_family!r}, "
                f"sampling_regime={sampling_regime!r}."
            )
            raise RuntimeError(message)
        return None if not matches else matches[0]

    def require_batch(
        self,
        *,
        material_family: str,
        sampling_regime: str,
        simulation_profile: str | None = None,
    ) -> GenerationConfig:
        """Return one structurally selected batch and fail clearly when absent."""
        batch = self.find_batch(
            material_family=material_family,
            sampling_regime=sampling_regime,
            simulation_profile=simulation_profile,
        )
        if batch is not None:
            return batch
        profile_id = self.profile.id if simulation_profile is None else simulation_profile
        available = ", ".join(f"({item.profile.id}, {item.material_family}, {item.sampling_regime})" for item in self.batches)
        message = (
            "No campaign batch is declared for "
            f"profile={profile_id!r}, material_family={material_family!r}, "
            f"sampling_regime={sampling_regime!r}. Available: {available}."
        )
        raise ValueError(message)

    def select_batches(self, batch_names: tuple[str, ...] | None) -> CampaignConfig:
        """Return an internally complete execution view over declared batches."""
        if batch_names is None:
            return self
        if not batch_names or len(batch_names) != len(set(batch_names)):
            message = "Campaign batch selection must be non-empty and duplicate-free."
            raise ValueError(message)
        selected = tuple(self.batch(name) for name in batch_names)
        available_sources = {(batch.material_family, batch.sampling_regime) for batch in selected}
        packages = tuple(
            copy.deepcopy(package)
            for package in self.dataset_packages
            if all(
                (
                    material_family,
                    "parameter_ood" if package["evaluation_regime"] == "parameter_ood" else "natural",
                )
                in available_sources
                for material_family in package["materials"]
            )
        )
        return replace(
            self,
            total_case_count=sum(len(batch.case_indices) for batch in selected),
            batches=selected,
            evaluation_regimes=tuple(dict.fromkeys(str(package["evaluation_regime"]) for package in packages)),
            dataset_packages=packages,
            package_request_digest=common.serialization.canonical_json_sha256(list(packages)),
        )


class GenerationConfigError(ValueError):
    """Report one invalid layered generation configuration."""


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    """Return one isolated mapping with string keys."""
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        message = f"{label} must be a mapping with string keys."
        raise TypeError(message)
    return copy.deepcopy(dict(value))


def _exact_keys(value: Mapping[str, Any], *, required: set[str], optional: set[str], label: str) -> None:
    """Require all declared keys and reject unknown configuration."""
    missing = sorted(required.difference(value))
    unknown = sorted(set(value).difference(required | optional))
    if missing or unknown:
        message = f"{label} keys are invalid: missing={missing}, unknown={unknown}."
        raise GenerationConfigError(message)


def _finite(value: Any, *, label: str) -> float:
    """Return one finite non-boolean real value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        message = f"{label} must be a finite real value, got {value!r}."
        raise GenerationConfigError(message)
    return float(value)


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    """Return one bounded non-boolean integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        message = f"{label} must be an integer >= {minimum}, got {value!r}."
        raise GenerationConfigError(message)
    return value


def _delimiter(value: Any, *, label: str) -> str:
    """Return one supported deterministic text-table delimiter."""
    if value == "\t":
        return "	"
    if value not in {",", ";", "	"}:
        message = f"{label} must be ',', ';', or '\t'."
        raise GenerationConfigError(message)
    return str(value)


def _load_yaml(path: Path, *, label: str) -> dict[str, Any]:
    """Load one required YAML mapping."""
    if not path.is_file():
        message = f"Required {label} does not exist: {path}"
        raise FileNotFoundError(message)
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        message = f"Could not load {label}: {path}"
        raise GenerationConfigError(message) from error
    return _mapping(loaded, label=label)


def _diagnostic_path(path: Path) -> str:
    """Return a repository-relative owner path when possible."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(common.paths.get_project_root().resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _diagnostic_key(message: str) -> str:
    """Extract the most specific authored key named by one validation error."""
    candidates = re.findall(
        r"(?:campaign|common|operations|profile|execution|registry|material(?:\[[a-z_]+\])?)"
        r"(?:\.[A-Za-z_][A-Za-z0-9_-]*|\[[0-9]+\])+",
        message,
    )
    if candidates:
        return max(candidates, key=len)
    for key in (
        "steady_flow_conditioning",
        "simulation_profile",
        "material_family",
        "campaign_purpose",
    ):
        if key in message:
            layer = "profile" if key in {"steady_flow_conditioning", "simulation_profile"} else "campaign"
            return f"{layer}.{key}"
    return "$"


def _diagnostic_value(value: Any, key: str) -> Any:
    """Look up one dotted/indexed diagnostic key without mutating authored data."""
    if key == "$":
        return value
    current = value
    for name, index in re.findall(r"([A-Za-z_][A-Za-z0-9_-]*)(?:\[([0-9]+)\])?", key):
        if not isinstance(current, Mapping) or name not in current:
            return "<missing>"
        current = current[name]
        if index:
            if not isinstance(current, list) or int(index) >= len(current):
                return "<missing>"
            current = current[int(index)]
    return current


def validation_error_details(path: Path | str, error: Exception) -> dict[str, Any]:
    """Return exact file, key, rule, value, and authored owner for a CLI error."""
    campaign_path = Path(path).expanduser().resolve()
    try:
        campaign_raw = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        campaign_raw = None
    message = str(error)
    key = _diagnostic_key(message)
    layer = key.split(".", maxsplit=1)[0].split("[", maxsplit=1)[0]
    owner_path = campaign_path
    local_key = key
    owner_raw = campaign_raw
    references = {
        "common": "common_config",
        "sources": "sources_config",
        "registry": "registry_config",
        "operations": "operations_config",
        "bed_fields": "bed_fields_config",
        "profile": "profile_config",
        "execution": "execution_config",
    }
    if layer == "campaign":
        local_key = key.removeprefix("campaign.")
    elif isinstance(campaign_raw, Mapping) and layer in references:
        reference = campaign_raw.get(references[layer])
        if isinstance(reference, str):
            owner_path = _reference_path(
                reference,
                source_path=campaign_path,
                label=f"campaign.{references[layer]}",
            )
            try:
                owner_raw = yaml.safe_load(owner_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, yaml.YAMLError):
                owner_raw = None
            local_key = key.removeprefix(f"{layer}.")
    elif layer == "material":
        match = re.match(r"material\[([a-z_]+)\]", key)
        if match is not None:
            family = match.group(1)
            owner_path = materials.material_config_directory() / f"{family}.yaml"
            try:
                owner_raw = yaml.safe_load(owner_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, yaml.YAMLError):
                owner_raw = None
            local_key = key.removeprefix(f"material[{family}].")
    actual = _diagnostic_value(owner_raw, local_key)
    file_name = _diagnostic_path(owner_path)
    return {
        "error": "generation_config_validation",
        "file": file_name,
        "key": local_key,
        "expected_type_or_rule": message,
        "actual_value": actual,
        "owner_to_edit": file_name,
    }


def _reference_path(value: Any, *, source_path: Path, label: str) -> Path:
    """Resolve one explicit layered configuration reference."""
    if not isinstance(value, str) or not value:
        message = f"{label} must be non-empty path text."
        raise GenerationConfigError(message)
    configured = Path(value).expanduser()
    if configured.is_absolute():
        return configured.resolve()
    local = (source_path.parent / configured).resolve()
    if local.is_file():
        return local
    return (common.paths.get_project_root() / configured).resolve()


def _validate_grid(value: Any) -> dict[str, Any]:
    """Validate a boundary-inclusive Cartesian grid and derive its spacings."""
    grid = _mapping(value, label="common.grid")
    authored = {"nx", "ny", "Lx", "Ly", "Lz", "boundaries_included"}
    _exact_keys(grid, required=authored, optional=set(), label="common.grid")
    normalized = {
        "nx": _integer(grid["nx"], label="common.grid.nx", minimum=2),
        "ny": _integer(grid["ny"], label="common.grid.ny", minimum=2),
        "Lx": _finite(grid["Lx"], label="common.grid.Lx"),
        "Ly": _finite(grid["Ly"], label="common.grid.Ly"),
        "Lz": _finite(grid["Lz"], label="common.grid.Lz"),
        "boundaries_included": grid["boundaries_included"],
    }
    if any(normalized[name] <= 0.0 for name in ("Lx", "Ly", "Lz")):
        message = "common.grid lengths must be finite and strictly positive."
        raise GenerationConfigError(message)
    if normalized["boundaries_included"] is not True:
        message = "common.grid.boundaries_included must be true for the maintained Cartesian adapter."
        raise GenerationConfigError(message)
    normalized["dx"] = normalized["Lx"] / (normalized["nx"] - 1)
    normalized["dy"] = normalized["Ly"] / (normalized["ny"] - 1)
    return normalized


def _validate_time(value: Any) -> dict[str, Any]:
    """Validate a regular output-time contract, add the fixed adaptive-solver fields and derive its exact nodes."""
    time = _mapping(value, label="common.time")
    _exact_keys(time, required={"start", "stop", "interval"}, optional=set(), label="common.time")
    normalized: dict[str, Any] = {
        "start": _finite(time["start"], label="common.time.start"),
        "stop": _finite(time["stop"], label="common.time.stop"),
        "interval": _finite(time["interval"], label="common.time.interval"),
        **_FIXED_TIME_FIELDS,
    }
    if normalized["start"] < 0.0 or normalized["stop"] <= normalized["start"]:
        message = "common.time must have 0 <= start < stop."
        raise GenerationConfigError(message)
    if normalized["interval"] <= 0.0:
        message = "common.time.interval must be strictly positive."
        raise GenerationConfigError(message)
    step_count = (normalized["stop"] - normalized["start"]) / normalized["interval"]
    regular_intervals = round(step_count)
    tolerance = 64.0 * math.ulp(max(abs(step_count), 1.0))
    if regular_intervals < 1 or not math.isclose(
        step_count,
        float(regular_intervals),
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        message = "common.time interval must divide the configured start-to-stop span exactly."
        raise GenerationConfigError(message)
    normalized["regular_times"] = [float(normalized["start"] + index * normalized["interval"]) for index in range(regular_intervals + 1)]
    return normalized


def _validate_scientific_fixed(
    value: Any,
    *,
    sources: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Validate fixed thermodynamic, humidity, and stopping records."""
    fixed = _mapping(value, label="common.scientific_fixed_values")
    units = {
        "T_flow_ref": "K",
        "p_ref": "Pa",
        "p_out": "Pa",
        "T_in_min": "K",
        "T_in_max": "K",
        "omega_min": "kg/kg",
        "omega_max": "kg/kg",
        "phi_operational_min": "1",
        "phi_operational_max": "1",
        "phi_clip_min": "1",
        "phi_clip_max": "1",
        "cp_w": "J/(kg*K)",
        "h_fg": "J/kg",
        "D_v_air": "m^2/s",
        "M_v": "kg/mol",
        "d_wall": "m",
        "k_wall": "W/(m*K)",
        "h_ext": "W/(m^2*K)",
        "U_wall": "W/(m^2*K)",
        "f_wet_dm_max": "1",
        "schedule_interpolation": "method",
    }
    _exact_keys(fixed, required=set(units), optional=set(), label="common.scientific_fixed_values")
    records: dict[str, dict[str, Any]] = {}
    resolved_values: dict[str, Any] = {}
    for name, expected_unit in units.items():
        record = materials.resolve_value_record(
            fixed[name],
            sources=sources,
            label=f"common.scientific_fixed_values.{name}",
        )
        optional = {"boundary"} if name in {"omega_min", "omega_max"} else set()
        required = {"unit", "provenance"} if name == "U_wall" else {"value", "unit", "provenance"}
        _exact_keys(
            record,
            required=required,
            optional=optional,
            label=f"common.scientific_fixed_values.{name}",
        )
        if record["unit"] != expected_unit:
            message = f"common.scientific_fixed_values.{name}.unit must be {expected_unit!r}."
            raise GenerationConfigError(message)
        if "boundary" in record:
            boundary = _mapping(record["boundary"], label=f"common.scientific_fixed_values.{name}.boundary")
            _exact_keys(
                boundary,
                required={"boundary_kind", "boundary_basis", "hard_boundary"},
                optional=set(),
                label=f"common.scientific_fixed_values.{name}.boundary",
            )
            if (
                boundary["boundary_kind"] != "engineering_source_air_envelope"
                or boundary["boundary_basis"] != "source_air_dew_point_range"
                or boundary["hard_boundary"] is not True
            ):
                message = f"common.scientific_fixed_values.{name}.boundary violates the configured hard-envelope contract."
                raise GenerationConfigError(message)
            record["boundary"] = boundary
        if name == "U_wall":
            provenance = record["provenance"]
            if provenance["evidence"] != "derived" or provenance.get("verification") != "mathematically_reproduced":
                message = "common.scientific_fixed_values.U_wall must retain its mathematically reproduced derivation."
                raise GenerationConfigError(message)
        elif name == "schedule_interpolation":
            if record["value"] != "linear":
                message = "The maintained schedule adapter supports linear interpolation between regular nodes."
                raise GenerationConfigError(message)
            resolved_values[name] = "linear"
        else:
            resolved_values[name] = _finite(record["value"], label=f"common.scientific_fixed_values.{name}.value")
        records[name] = record

    positive_names = (
        "T_flow_ref",
        "p_ref",
        "T_in_min",
        "T_in_max",
        "cp_w",
        "h_fg",
        "D_v_air",
        "M_v",
        "d_wall",
        "k_wall",
        "h_ext",
    )
    if any(resolved_values[name] <= 0.0 for name in positive_names):
        message = f"Scientific fixed values must be strictly positive for {list(positive_names)}."
        raise GenerationConfigError(message)
    if resolved_values["T_in_min"] >= resolved_values["T_in_max"]:
        message = "common.scientific_fixed_values must satisfy T_in_min < T_in_max."
        raise GenerationConfigError(message)
    resolved_values["U_wall"] = 1.0 / (resolved_values["d_wall"] / resolved_values["k_wall"] + 1.0 / resolved_values["h_ext"])
    records["U_wall"]["value"] = resolved_values["U_wall"]
    humidity = tuple(
        resolved_values[name]
        for name in (
            "omega_min",
            "omega_max",
            "phi_operational_min",
            "phi_operational_max",
            "phi_clip_min",
            "phi_clip_max",
        )
    )
    if not (0.0 <= humidity[0] < humidity[1] and 0.0 < humidity[4] < humidity[2] < humidity[3] < humidity[5] < 1.0):
        message = "Operational humidity bounds and numerical Oswin clipping limits are invalid or conflated."
        raise GenerationConfigError(message)
    if not 0.0 < resolved_values["f_wet_dm_max"] < 1.0:
        message = "common.scientific_fixed_values.f_wet_dm_max must lie strictly inside (0, 1)."
        raise GenerationConfigError(message)
    return resolved_values, records


def _input_contract() -> dict[str, Any]:
    """Return the canonical spatial, scalar, and schedule adapters the simulation templates read."""
    return {
        "spatial": {
            "filename": _INPUT_FILENAMES["spatial"],
            "delimiter": _INPUT_DELIMITER,
            "columns_by_profile": {profile_id: list(profiles.spatial_input_fields(profile_id)) for profile_id in profiles.available_profiles()},
        },
        "scalar": {"filename": _INPUT_FILENAMES["scalar"], "delimiter": _INPUT_DELIMITER, "columns": ["name", "value", "unit"]},
        "schedule": {"filename": _INPUT_FILENAMES["schedule"], "delimiter": _INPUT_DELIMITER, "columns": list(profiles.SCHEDULE_FIELDS)},
    }


def _validate_storage(value: Any) -> dict[str, Any]:
    """Validate the configured HDF5 layout settings and add the fixed compression."""
    authored = _mapping(value, label="common.storage")
    expected = {"compression_level", "shuffle", "float32_rtol", "float32_atol", "chunk_time", "chunk_y", "chunk_x"}
    _exact_keys(authored, required=expected, optional=set(), label="common.storage")
    storage = {**_FIXED_STORAGE_FIELDS, **authored}
    storage["compression_level"] = _integer(
        storage["compression_level"],
        label="common.storage.compression_level",
    )
    if storage["compression_level"] > _MAXIMUM_GZIP_COMPRESSION_LEVEL:
        message = "common.storage.compression_level must be a gzip level from 0 through 9."
        raise GenerationConfigError(message)
    if not isinstance(storage["shuffle"], bool):
        message = "common.storage.shuffle must be boolean."
        raise TypeError(message)
    for key in ("float32_rtol", "float32_atol"):
        storage[key] = _finite(storage[key], label=f"common.storage.{key}")
        if storage[key] < 0:
            msg = f"common.storage.{key} must be non-negative."
            raise GenerationConfigError(msg)
    for key in ("chunk_time", "chunk_y", "chunk_x"):
        storage[key] = _integer(storage[key], label=f"common.storage.{key}", minimum=1)
    return storage


def _validate_validation(value: Any) -> dict[str, Any]:
    """Validate configured scientific-admission tolerances."""
    validation = _mapping(value, label="common.validation")
    _exact_keys(
        validation,
        required={"transient_initial_state", "transient_bulk_moisture"},
        optional=set(),
        label="common.validation",
    )
    specifications = {
        "transient_initial_state": (
            _MAXIMUM_TRANSIENT_INITIAL_STATE_RTOL,
            _MAXIMUM_TRANSIENT_INITIAL_STATE_ATOL,
        ),
        "transient_bulk_moisture": (
            _MAXIMUM_TRANSIENT_BULK_MOISTURE_RTOL,
            _MAXIMUM_TRANSIENT_BULK_MOISTURE_ATOL,
        ),
    }
    result: dict[str, Any] = {}
    for name, (maximum_rtol, maximum_atol) in specifications.items():
        label = f"common.validation.{name}"
        tolerance = _mapping(validation[name], label=label)
        _exact_keys(
            tolerance,
            required={"rtol", "atol"},
            optional=set(),
            label=label,
        )
        tolerance["rtol"] = _finite(
            tolerance["rtol"],
            label=f"{label}.rtol",
        )
        tolerance["atol"] = _finite(
            tolerance["atol"],
            label=f"{label}.atol",
        )
        if not 0.0 < tolerance["rtol"] <= maximum_rtol:
            message = f"{label}.rtol must be positive and no greater than {maximum_rtol}."
            raise GenerationConfigError(message)
        if not 0.0 <= tolerance["atol"] <= maximum_atol:
            message = f"{label}.atol must be non-negative and no greater than {maximum_atol}."
            raise GenerationConfigError(message)
        result[name] = tolerance
    return result


def _validate_provenance_owners(
    value: Any,
    *,
    expected_names: set[str],
    sources: Mapping[str, Mapping[str, Any]],
    label: str,
) -> dict[str, dict[str, Any]]:
    """Resolve one exact keyed provenance owner without fabricating absent fields."""
    owners = _mapping(value, label=label)
    _exact_keys(owners, required=expected_names, optional=set(), label=label)
    return {
        name: provenance_service.resolve_provenance(
            provenance,
            sources=sources,
            label=f"{label}.{name}",
        )
        for name, provenance in owners.items()
    }


def _validate_common(
    value: Any,
    *,
    sources: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate the global material-independent scientific owner."""
    common_config = _mapping(value, label="generation common configuration")
    expected = {
        "schema_kind",
        "parameter_values",
        "scientific_fixed_values",
        "grid",
        "grid_provenance",
        "time",
        "time_provenance",
        "validation",
        "storage",
    }
    _exact_keys(common_config, required=expected, optional={"schedule_time", "schedule_time_provenance"}, label="generation common configuration")
    if common_config["schema_kind"] != "generation_common":
        message = "Unsupported generation common configuration schema."
        raise GenerationConfigError(message)
    common_config["parameter_values"] = _mapping(common_config["parameter_values"], label="common.parameter_values")
    if set(common_config["parameter_values"]) != {"eps_min_global", "eps_max_global"}:
        message = "common.parameter_values must own exactly the two global porosity guards."
        raise GenerationConfigError(message)
    for name, record in common_config["parameter_values"].items():
        materials.resolve_value_record(record, sources=sources, label=f"common.parameter_values.{name}")

    contract_provenance = provenance_service.resolve_provenance(_CODE_CONTRACT_PROVENANCE, sources=sources, label="code contract provenance")
    common_config["physical_formulas"] = copy.deepcopy(_FINAL_PHYSICAL_FORMULAS)
    common_config["physical_formulas_provenance"] = contract_provenance
    common_config["input_contract"] = _input_contract()
    fixed_time_provenance = {name: copy.deepcopy(contract_provenance) for name in _FIXED_TIME_FIELDS}

    fixed_values, fixed_records = _validate_scientific_fixed(
        common_config["scientific_fixed_values"],
        sources=sources,
    )
    common_config["scientific_fixed_values"] = fixed_values
    common_config["_scientific_fixed_records"] = fixed_records
    common_config["grid"] = _validate_grid(common_config["grid"])
    common_config["grid_provenance"] = _validate_provenance_owners(
        common_config["grid_provenance"],
        expected_names=set(common_config["grid"]),
        sources=sources,
        label="common.grid_provenance",
    )
    common_config["time"] = _validate_time(common_config["time"])
    common_config["time_provenance"] = _validate_provenance_owners(
        common_config["time_provenance"],
        expected_names={"start", "stop", "interval"},
        sources=sources,
        label="common.time_provenance",
    ) | copy.deepcopy(fixed_time_provenance)
    schedule_keys = {"schedule_time", "schedule_time_provenance"}
    if schedule_keys.intersection(common_config):
        if not schedule_keys.issubset(common_config):
            message = "common.schedule_time requires common.schedule_time_provenance and vice versa."
            raise GenerationConfigError(message)
        schedule_time = _validate_time(common_config["schedule_time"])
        output_time = common_config["time"]
        if (
            schedule_time["start"] != output_time["start"]
            or schedule_time["interval"] != output_time["interval"]
            or schedule_time["stop"] < output_time["stop"]
        ):
            message = "common.schedule_time must share output start and interval and cover the complete output horizon."
            raise GenerationConfigError(message)
        common_config["schedule_time"] = schedule_time
        common_config["schedule_time_provenance"] = _validate_provenance_owners(
            common_config["schedule_time_provenance"],
            expected_names={"start", "stop", "interval"},
            sources=sources,
            label="common.schedule_time_provenance",
        ) | copy.deepcopy(fixed_time_provenance)
    common_config["validation"] = _validate_validation(common_config["validation"])
    common_config["storage"] = _validate_storage(common_config["storage"])
    return common_config


def _steady_flow_conditioning(fixed_values: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exhaustive stationary-airflow conditioning both profiles share: the fields.csv inputs and the fixed common values."""
    static_units: dict[str, str] = dict(zip(profiles.STEADY_STATIC_FIELD_NAMES, profiles.STEADY_STATIC_FIELD_UNITS, strict=True))
    model_inputs = [
        {"name": name, "affects_stationary_solution": True, "owner": "model_input", "unit": static_units[name]}
        for name in profiles.STEADY_SPATIAL_INPUT_FIELDS
        if name not in {"x", "y"}
    ]
    package_fixed = [
        {"name": name, "affects_stationary_solution": True, "owner": "package_fixed", "unit": unit, "fixed_value": float(fixed_values[name])}
        for name, unit in zip(profiles.STATIONARY_FIXED_FIELDS, profiles.STATIONARY_FIXED_UNITS, strict=True)
    ]
    return {
        "schema_kind": "steady_flow_conditioning",
        "exhaustive": True,
        "stationary_solution_contract_id": "vp2_stationary_airflow",
        "dependencies": [*model_inputs, *package_fixed],
        "additional_case_varying_solver_scalars": [],
    }


def _validate_mapping_value(
    value: Any,
    *,
    label: str,
    allow_missing: bool,
    source_pattern: bool,
) -> str | None:
    """Validate one explicit source filename or source-header declaration."""
    if value is None:
        if allow_missing:
            return None
        message = f"{label} is incomplete; an explicit executable mapping declaration is required."
        raise GenerationConfigError(message)
    if isinstance(value, Mapping) and "state" in value:
        message = f"{label} uses obsolete persistent export-mapping verification state; the generation profile requires a scalar declaration."
        raise GenerationConfigError(message)
    if not isinstance(value, str) or not value or value.strip() != value or any(character in value for character in ("\x00", "\n", "\r")):
        message = f"{label} must be safe non-empty text."
        raise GenerationConfigError(message)
    if source_pattern:
        candidate = Path(value)
        if (
            candidate.name != value
            or candidate.is_absolute()
            or value in {".", ".."}
            or any(character in value for character in ("*", "?", "[", "]"))
        ):
            message = f"{label} must be one exact case-local filename."
            raise GenerationConfigError(message)
    return value


def _validate_profile_config(
    value: Any,
    *,
    profile: profiles.SimulationProfile,
    require_executable: bool,
) -> tuple[dict[str, Any], templates.ResolvedTemplateIdentity]:
    """Validate configured template identity and explicit export mappings."""
    if not isinstance(require_executable, bool):
        message = "require_executable must be boolean."
        raise TypeError(message)
    config = _mapping(value, label="generation profile configuration")
    expected = {"schema_kind", "simulation_profile", "template", "exports"}
    _exact_keys(
        config,
        required=expected,
        optional=set(),
        label="generation profile configuration",
    )
    if config["schema_kind"] != "generation_profile" or config["simulation_profile"] != profile.id:
        message = "Generation profile configuration schema or profile identity is invalid."
        raise GenerationConfigError(message)
    template = templates.resolve_template_identity(config["template"])
    config["template"] = template.relative_path
    exports = config["exports"]
    if not isinstance(exports, list):
        message = "generation profile exports must be an ordered list."
        raise TypeError(message)
    expected_roles = tuple(spec.role for spec in profile.export_roles)
    if tuple(raw.get("role") if isinstance(raw, Mapping) else None for raw in exports) != expected_roles:
        message = f"Profile {profile.id!r} must declare export roles exactly {list(expected_roles)} in order."
        raise GenerationConfigError(message)
    temporal_kinds = {
        profiles.STEADY_FLOW_EXPORT_ROLE: "stationary",
        profiles.TRANSIENT_RAW_EXPORT_ROLE: "regular_time_series",
        profiles.GLOBAL_EXPORT_ROLE: "regular_time_series",
        profiles.FINAL_STATUS_EXPORT_ROLE: "final_status",
    }
    validated: list[dict[str, Any]] = []
    for index, raw in enumerate(exports):
        label = f"generation profile exports[{index}]"
        export = _mapping(raw, label=label)
        _exact_keys(
            export,
            required={
                "role",
                "temporal_kind",
                "source",
                "delimiter",
                "columns",
            },
            optional=set(),
            label=label,
        )
        role = str(export["role"])
        role_spec = profile.export_role(role)
        temporal_kind = export["temporal_kind"]
        if temporal_kind != temporal_kinds[role]:
            message = f"{label}.temporal_kind must be {temporal_kinds[role]!r}."
            raise GenerationConfigError(message)
        source_pattern = _validate_mapping_value(
            export["source"],
            label=f"{label}.source",
            allow_missing=not require_executable,
            source_pattern=True,
        )
        columns = _mapping(export["columns"], label=f"{label}.columns")
        if tuple(columns) != role_spec.logical_fields:
            message = f"{label}.columns must map exact logical fields {list(role_spec.logical_fields)} in order."
            raise GenerationConfigError(message)
        column_mappings = {
            logical: _validate_mapping_value(
                mapping,
                label=f"{label}.columns.{logical}",
                allow_missing=not require_executable,
                source_pattern=False,
            )
            for logical, mapping in columns.items()
        }
        configured_sources = [mapping for mapping in column_mappings.values() if mapping is not None]
        if len(configured_sources) != len(set(configured_sources)):
            message = f"{label}.columns must map known fields to distinct source headers."
            raise GenerationConfigError(message)
        normalized: dict[str, Any] = {
            "role": role,
            "temporal_kind": temporal_kind,
            "delimiter": _delimiter(
                export["delimiter"],
                label=f"{label}.delimiter",
            ),
            "columns": {logical: mapping for logical, mapping in column_mappings.items() if mapping is not None},
            "required": role_spec.required,
            "allow_multiple": role_spec.allow_multiple,
            "units": dict(
                zip(
                    role_spec.logical_fields,
                    role_spec.units,
                    strict=True,
                )
            ),
        }
        if source_pattern is not None:
            normalized["pattern"] = source_pattern
        validated.append(normalized)
    if require_executable:
        unresolved = []
        for export in validated:
            if not export["required"]:
                continue
            missing_columns = [logical for logical in profile.export_role(str(export["role"])).logical_fields if logical not in export["columns"]]
            if "pattern" not in export or missing_columns:
                unresolved.append(
                    {
                        "role": export["role"],
                        "source_missing": "pattern" not in export,
                        "missing_source_headers": missing_columns,
                    }
                )
        if unresolved:
            message = f"Executable generation profile has incomplete required export declarations: {unresolved}."
            raise GenerationConfigError(message)
    config["exports"] = validated
    return config, template


def _validate_operations(
    value: Any,
    *,
    sources: Mapping[str, Mapping[str, Any]],
    definitions: Mapping[str, Mapping[str, Any]],
    time_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the operating and inlet-schedule distributions and add the fixed operation identity and rules."""
    operations = _mapping(value, label="generation operations configuration")
    _exact_keys(
        operations,
        required={"schema_kind", "parameter_values", "boundary_schedule"},
        optional=set(),
        label="generation operations configuration",
    )
    if operations["schema_kind"] != "generation_operations":
        message = "Unsupported generation operations schema."
        raise GenerationConfigError(message)
    operations["operation_id"] = _OPERATION_ID
    operations["constraints"] = copy.deepcopy(_OPERATION_CONSTRAINTS)

    boundary_schedule = _mapping(
        operations["boundary_schedule"],
        label="operations.boundary_schedule",
    )
    _exact_keys(
        boundary_schedule,
        required={"startup_ramp"},
        optional=set(),
        label="operations.boundary_schedule",
    )
    startup_ramp = _mapping(
        boundary_schedule["startup_ramp"],
        label="operations.boundary_schedule.startup_ramp",
    )
    _exact_keys(
        startup_ramp,
        required={"enabled", "duration_h", "initial_equilibrium_rh_dry_margin", "max_relative_humidity"},
        optional=set(),
        label="operations.boundary_schedule.startup_ramp",
    )
    if not isinstance(startup_ramp["enabled"], bool):
        message = "operations.boundary_schedule.startup_ramp.enabled must be boolean."
        raise GenerationConfigError(message)
    duration_h = _finite(
        startup_ramp["duration_h"],
        label="operations.boundary_schedule.startup_ramp.duration_h",
    )
    regular_interval = float(time_contract["interval"])
    if not 0.0 < duration_h < regular_interval:
        message = "operations.boundary_schedule.startup_ramp.duration_h must be strictly positive and shorter than common.time.interval."
        raise GenerationConfigError(message)
    dry_margin = _finite(
        startup_ramp["initial_equilibrium_rh_dry_margin"],
        label="operations.boundary_schedule.startup_ramp.initial_equilibrium_rh_dry_margin",
    )
    if not 0.0 < dry_margin < 1.0:
        message = "operations.boundary_schedule.startup_ramp.initial_equilibrium_rh_dry_margin must lie strictly inside (0, 1)."
        raise GenerationConfigError(message)
    startup_relative_humidity_maximum = _finite(
        startup_ramp["max_relative_humidity"],
        label="operations.boundary_schedule.startup_ramp.max_relative_humidity",
    )
    if not 0.0 < startup_relative_humidity_maximum <= 1.0:
        message = "operations.boundary_schedule.startup_ramp.max_relative_humidity must lie inside (0, 1]."
        raise GenerationConfigError(message)
    boundary_schedule["startup_ramp"] = {
        "enabled": startup_ramp["enabled"],
        "duration_h": duration_h,
        "initial_equilibrium_rh_dry_margin": dry_margin,
        "max_relative_humidity": startup_relative_humidity_maximum,
    }
    operations["boundary_schedule"] = boundary_schedule

    parameter_values = _validate_parameter_owner(
        operations["parameter_values"], definitions=definitions, operation_block=True, sources=sources, label="operations.parameter_values"
    )
    try:
        schedule_service.validate_temporal_support_resolution(
            parameter_values,
            time_contract,
        )
    except ValueError as error:
        message = f"Temporal schedule support is incompatible with common.time: {error}"
        raise GenerationConfigError(message) from error
    operations["parameter_values"] = parameter_values
    return operations


def _validate_parameter_owner(
    value: Any,
    *,
    definitions: Mapping[str, Mapping[str, Any]],
    operation_block: bool,
    sources: Mapping[str, Mapping[str, Any]],
    label: str,
) -> dict[str, Any]:
    """Validate the exact operator-owned registry coordinates inside or outside the operation block, with their provenance."""
    parameter_values = _mapping(value, label=label)
    expected = {
        name
        for name, definition in definitions.items()
        if definition.get("kind") != "derived"
        and name not in _MATERIAL_OWNED_PARAMETERS | _GLOBAL_PARAMETERS
        and (definition.get("block") == _OPERATION_BLOCK) == operation_block
    }
    if set(parameter_values) != expected:
        missing = sorted(expected.difference(parameter_values))
        unknown = sorted(set(parameter_values).difference(expected))
        message = f"{label} ownership mismatch: missing={missing}, unknown={unknown}."
        raise GenerationConfigError(message)
    for name, record in parameter_values.items():
        materials.resolve_value_record(record, sources=sources, label=f"{label}.{name}")
    return parameter_values


def _validate_bed_fields(
    value: Any,
    *,
    sources: Mapping[str, Mapping[str, Any]],
    definitions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate the bed, permeability, porosity, inlet-pressure and initial-moisture field statistics."""
    label = "generation bed-field configuration"
    bed_fields = _mapping(value, label=label)
    _exact_keys(bed_fields, required={"schema_kind", "parameter_values"}, optional=set(), label=label)
    if bed_fields["schema_kind"] != "generation_bed_fields":
        message = "Unsupported generation bed-field schema."
        raise GenerationConfigError(message)
    return _validate_parameter_owner(
        bed_fields["parameter_values"], definitions=definitions, operation_block=False, sources=sources, label="bed_fields.parameter_values"
    )


def _name_list(value: Any, *, label: str, allow_empty: bool = False) -> tuple[str, ...]:
    """Return one unique ordered list of non-empty identifiers."""
    if not isinstance(value, list) or (not value and not allow_empty):
        message = f"{label} must be {'an' if allow_empty else 'a non-'}empty ordered list."
        raise GenerationConfigError(message)
    names = tuple(value)
    if any(not isinstance(name, str) or not name for name in names) or len(names) != len(set(names)):
        message = f"{label} must contain unique non-empty identifiers."
        raise GenerationConfigError(message)
    return names


def _uint32(value: Any, *, label: str) -> int:
    """Return one explicit uint32 seed."""
    number = _integer(value, label=label)
    if number > seeding.UINT32_MAX:
        message = f"{label} exceeds the uint32 range."
        raise GenerationConfigError(message)
    return number


def _validate_designs(value: Any, *, materials_in_order: tuple[str, ...]) -> dict[str, dict[str, int]]:
    """Validate one positive design size per configured material and design kind, in material and design-kind order."""
    designs = _mapping(value, label="campaign.sampling.designs")
    _exact_keys(designs, required=set(materials_in_order), optional=set(), label="campaign.sampling.designs")
    normalized: dict[str, dict[str, int]] = {}
    for material_family in materials_in_order:
        kinds = _mapping(designs[material_family], label=f"campaign.sampling.designs.{material_family}")
        _exact_keys(kinds, required=set(), optional=set(DESIGN_KINDS), label=f"campaign.sampling.designs.{material_family}")
        normalized[material_family] = {
            kind: _integer(kinds[kind], label=f"campaign.sampling.designs.{material_family}.{kind}", minimum=1)
            for kind in DESIGN_KINDS
            if kind in kinds
        }
    return normalized


def _validate_parameter_ood_policy(
    value: Any,
    *,
    groups: tuple[str, ...],
) -> dict[str, Any]:
    """Validate generic eligible-unit allocation over profile-active OOD units."""
    policy = _mapping(value, label="campaign.sampling.parameter_ood")
    expected = {"groups", "units_per_case", "allocation_strategy", "eligibility_source"}
    _exact_keys(policy, required=expected, optional=set(), label="campaign.sampling.parameter_ood")
    if policy["groups"] != list(groups):
        message = f"Parameter-OOD groups for this profile must be exactly {list(groups)}."
        raise GenerationConfigError(message)
    policy["units_per_case"] = _integer(
        policy["units_per_case"],
        label="campaign.sampling.parameter_ood.units_per_case",
        minimum=1,
    )
    if policy["units_per_case"] != 1:
        message = "Exactly one parameter-OOD scalar unit or complete atomic record must be active per case."
        raise GenerationConfigError(message)
    if policy["allocation_strategy"] != "canonical_eligible_unit_round_robin":
        message = "Parameter OOD must use canonical eligible-unit round-robin allocation."
        raise GenerationConfigError(message)
    if policy["eligibility_source"] != "resolved_profile_projected_registry":
        message = "Parameter-OOD eligibility must come from the resolved profile-projected registry."
        raise GenerationConfigError(message)
    return policy


def design_point_assignment(
    material_family: str,
    sampling_regime: str,
    *,
    material_role: str,
    case_index: int,
    design_index: int,
    eligible_units: tuple[Mapping[str, str], ...],
) -> dict[str, Any]:
    """Return the assignment of design point ``design_index`` executed as material-local case ``case_index``.

    A parameter-OOD point takes eligible unit ``design_index mod n``, so the point carries the same unit in every plan.
    """
    allocated = (
        sampling_service.allocate_ood_units(eligible_units, case_count=1, first_point=design_index)[0] if sampling_regime == "parameter_ood" else None
    )
    return {
        "case_index": case_index,
        "design_index": design_index,
        "material_family": material_family,
        "sampling_regime": sampling_regime,
        "assignment_role": material_role if allocated is None else str(allocated["unit_id"]),
        "ood_group": None if allocated is None else allocated["ood_group"],
        "ood_unit_id": None if allocated is None else allocated["unit_id"],
        "ood_units_per_case": 0 if allocated is None else 1,
    }


def _safe_text_or_none(value: Any, *, label: str) -> str | None:
    """Return optional safe single-line execution text."""
    if value is None:
        return None
    if not isinstance(value, str) or not value or any(character in value for character in ("\x00", "\n", "\r")):
        msg = f"{label} must be null or safe non-empty text."
        raise GenerationConfigError(msg)
    return value


def parse_slurm_memory_mib(value: Any, *, label: str) -> int:
    """
    Return a Slurm memory size ``<int>M|G|T`` in MiB.

    Slurm reads ``M`` as MiB, ``G`` as 1024 MiB and ``T`` as 1024 GiB; every other form
    (decimals, ``K``, a bare number, a sign) is refused so that one parser owns the grammar.
    """
    match = _SLURM_MEMORY_PATTERN.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        message = f"{label} must be a positive memory size <int>M, <int>G or <int>T, got {value!r}."
        raise GenerationConfigError(message)
    return int(match.group(1)) * _SLURM_MEMORY_UNIT_MIB[match.group(2)]


def parse_wall_time_seconds(value: Any, *, label: str) -> int:
    """Return an ``HH:MM:SS`` Slurm time in seconds."""
    match = _WALL_TIME_PATTERN.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        message = f"{label} must use HH:MM:SS, got {value!r}."
        raise GenerationConfigError(message)
    hours, minutes, seconds = (int(part) for part in match.groups())
    return hours * 3_600 + minutes * 60 + seconds


def _validated_node_job_wall_time(value: Any, *, timeout_seconds: float) -> str:
    """Return the HH:MM:SS node-job time that fits one complete case and stays within a day."""
    label = "execution.node_job.node_job_wall_time"
    text = _safe_text_or_none(value, label=label)
    if text is None:
        message = f"{label} must use HH:MM:SS."
        raise GenerationConfigError(message)
    wall_seconds = parse_wall_time_seconds(text, label=label)
    if wall_seconds > _MAXIMUM_NODE_JOB_WALL_SECONDS:
        message = f"{label} must not exceed 24:00:00."
        raise GenerationConfigError(message)
    required_seconds = timeout_seconds + _NODE_PUBLICATION_MARGIN_SECONDS
    if wall_seconds <= required_seconds:
        message = (
            f"{label} ({text}) must exceed execution.run.timeout_seconds plus a {_NODE_PUBLICATION_MARGIN_SECONDS} s publication margin "
            f"({required_seconds:g} s), so that one case fits into a fresh node job."
        )
        raise GenerationConfigError(message)
    return text


def validated_license_wait(value: Any, *, label: str) -> dict[str, Any]:
    """Return the validated licence-wait rule: ``enabled`` and a positive finite ``maximum_wait_seconds``."""
    wait = _mapping(value, label=label)
    _exact_keys(wait, required={"enabled", "maximum_wait_seconds"}, optional=set(), label=label)
    if not isinstance(wait["enabled"], bool):
        message = f"{label}.enabled must be boolean."
        raise TypeError(message)
    wait["maximum_wait_seconds"] = _finite(wait["maximum_wait_seconds"], label=f"{label}.maximum_wait_seconds")
    if wait["maximum_wait_seconds"] <= 0.0:
        message = f"{label}.maximum_wait_seconds must be positive."
        raise GenerationConfigError(message)
    return wait


def _require_execution_schema(execution: Mapping[str, Any]) -> None:
    """Refuse every execution configuration that is not the current node-job schema."""
    if execution.get("schema_kind") != "generation_execution":
        msg = "Unsupported generation execution configuration schema."
        raise GenerationConfigError(msg)


def _derive_node_job_resources(cluster: dict[str, Any], *, cores_per_node: int) -> None:
    """Derive concurrency and memory of one node job from the authored case step; reject what does not fit."""
    if cluster["cores_per_case"] > cores_per_node:
        msg = "execution.run.cores_per_case must not exceed execution.site.cores_per_node."
        raise GenerationConfigError(msg)
    concurrent_cases = cores_per_node // cluster["cores_per_case"]
    case_mib = parse_slurm_memory_mib(cluster["memory_per_case"], label="execution.node_job.memory_per_case")
    overhead_mib = parse_slurm_memory_mib(cluster["node_memory_overhead"], label="execution.node_job.node_memory_overhead")
    limit_mib = parse_slurm_memory_mib(cluster["node_memory_limit"], label="execution.node_job.node_memory_limit")
    node_gib = math.ceil((concurrent_cases * case_mib + overhead_mib) / _SLURM_MEMORY_UNIT_MIB["G"])
    if node_gib * _SLURM_MEMORY_UNIT_MIB["G"] > limit_mib:
        msg = (
            f"execution.node_job needs {node_gib}G per node job ({concurrent_cases} concurrent cases x {cluster['memory_per_case']} "
            f"+ {cluster['node_memory_overhead']}) but node_memory_limit is {cluster['node_memory_limit']}; "
            "raise cores_per_case or lower memory_per_case, the concurrency is never reduced silently."
        )
        raise GenerationConfigError(msg)
    cluster["concurrent_cases_per_node"] = concurrent_cases
    cluster["node_job_memory"] = f"{node_gib}G"


_EXECUTION_GROUPS: Final = {
    "run": ({"node_jobs", "cores_per_case", "timeout_seconds", "retention", "max_admission_cases"}, set()),
    "rules": ({"license_wait", "extra_arguments"}, {"stall_stop"}),
    "node_job": (
        {"node_job_wall_time", "memory_per_case", "node_memory_overhead", "node_memory_limit", "scheduler_options"},
        set(),
    ),
    "site": ({"partition", "cores_per_node", "comsol_module", "comsol_executable", "python_executable"}, set()),
    "internals": ({"poll_interval_seconds", "graceful_stop_reserve_seconds"}, set()),
}


def _require_execution_layout(execution: Mapping[str, Any]) -> None:
    """Require the exact top-level groups and the exact keys of every group; removed keys and the old layout fail here."""
    _exact_keys(
        execution,
        required={"schema_kind", *_EXECUTION_GROUPS},
        optional=set(),
        label="generation execution configuration",
    )
    for group, (required, optional) in _EXECUTION_GROUPS.items():
        _exact_keys(_mapping(execution[group], label=f"execution.{group}"), required=required, optional=optional, label=f"execution.{group}")


def _apply_execution_overrides(value: Any, requested: Mapping[str, Any] | None) -> dict[str, Any]:
    """Apply explicit normal resource choices before authoritative validation."""
    execution = copy.deepcopy(_mapping(value, label="generation execution configuration"))
    _require_execution_schema(execution)
    _require_execution_layout(execution)
    overrides = dict(requested or {})
    allowed = {"node_jobs", "max_admission_cases", "cores_per_case", "case_timeout_seconds", "no_retry"}
    if set(overrides) - allowed:
        message = "Unsupported normal execution overrides."
        raise GenerationConfigError(message)
    if "node_jobs" in overrides:
        execution["run"]["node_jobs"] = _integer(overrides["node_jobs"], label="run.node_jobs", minimum=1)
    if "max_admission_cases" in overrides:
        execution["run"]["max_admission_cases"] = _integer(overrides["max_admission_cases"], label="run.max_admission_cases", minimum=1)
    if "cores_per_case" in overrides:
        execution["run"]["cores_per_case"] = _integer(overrides["cores_per_case"], label="run.cores_per_case", minimum=1)
    if "case_timeout_seconds" in overrides:
        timeout = _finite(overrides["case_timeout_seconds"], label="run.case_timeout_seconds")
        if timeout <= 0:
            message = "run.case_timeout_seconds must be positive."
            raise GenerationConfigError(message)
        reserve_label = "execution.internals.graceful_stop_reserve_seconds"
        reserve = min(_finite(execution["internals"]["graceful_stop_reserve_seconds"], label=reserve_label), timeout / 5)
        execution["run"]["timeout_seconds"] = timeout
        execution["internals"]["graceful_stop_reserve_seconds"] = reserve
    if "no_retry" in overrides:
        if not isinstance(overrides["no_retry"], bool):
            message = "run.no_retry must be boolean."
            raise GenerationConfigError(message)
        if overrides["no_retry"]:
            validated_license_wait(execution["rules"]["license_wait"], label="execution.rules.license_wait")
            execution["rules"]["license_wait"]["enabled"] = False
    return execution


def _validate_execution(value: Any, *, campaign_purpose: str) -> dict[str, Any]:
    """Validate the authored execution groups and resolve them into the runtime, submission, cluster and site values consumers read."""
    authored = _mapping(value, label="generation execution configuration")
    _require_execution_schema(authored)
    if campaign_purpose not in CAMPAIGN_PURPOSES:
        message = f"execution retention has no supported campaign purpose {campaign_purpose!r}."
        raise GenerationConfigError(message)
    run, rules, node_job = authored["run"], authored["rules"], authored["node_job"]
    site, internals = authored["site"], authored["internals"]

    for key in ("partition", "comsol_module", "python_executable", "comsol_executable"):
        site[key] = _safe_text_or_none(site[key], label=f"execution.site.{key}")
        if site[key] is None:
            msg = f"execution.site.{key} must be configured."
            raise GenerationConfigError(msg)
    site["cores_per_node"] = _integer(site["cores_per_node"], label="execution.site.cores_per_node", minimum=1)

    runtime: dict[str, Any] = {
        "timeout_seconds": _finite(run["timeout_seconds"], label="execution.run.timeout_seconds"),
        "graceful_stop_reserve_seconds": _finite(
            internals["graceful_stop_reserve_seconds"],
            label="execution.internals.graceful_stop_reserve_seconds",
        ),
        "extra_arguments": rules["extra_arguments"],
        "license_wait": validated_license_wait(rules["license_wait"], label="execution.rules.license_wait"),
    }
    if not 0.0 < runtime["graceful_stop_reserve_seconds"] < runtime["timeout_seconds"]:
        message = "execution.internals.graceful_stop_reserve_seconds must be positive and shorter than run.timeout_seconds."
        raise GenerationConfigError(message)
    if "stall_stop" in rules:
        runtime["stall_stop"] = _validated_stall_stop(rules["stall_stop"])

    arguments = runtime["extra_arguments"]
    if not isinstance(arguments, list) or not all(
        isinstance(item, str) and item and not any(char in item for char in ("\x00", "\n", "\r")) for item in arguments
    ):
        msg = "execution.rules.extra_arguments must be an ordered list of safe arguments."
        raise GenerationConfigError(msg)
    if any(item.casefold() == owned or item.casefold().startswith(f"{owned}=") for item in arguments for owned in _COMSOL_OWNED_ARGUMENTS):
        msg = "execution.rules.extra_arguments cannot override runtime-owned COMSOL arguments."
        raise GenerationConfigError(msg)
    runtime["executable"] = site["comsol_executable"]
    runtime["module_initialization"] = [f"module load {site['comsol_module']}"]

    retention = run["retention"]
    if retention not in {"full", "compact"}:
        message = "execution.run.retention must be full or compact."
        raise GenerationConfigError(message)

    submission = {
        "max_admission_cases": _integer(run["max_admission_cases"], label="execution.run.max_admission_cases", minimum=1),
        "poll_interval_seconds": _integer(internals["poll_interval_seconds"], label="execution.internals.poll_interval_seconds", minimum=1),
    }

    cluster = {
        "cores_per_case": _integer(run["cores_per_case"], label="execution.run.cores_per_case", minimum=1),
        "node_jobs": _integer(run["node_jobs"], label="execution.run.node_jobs", minimum=1),
        **{key: node_job[key] for key in ("memory_per_case", "node_memory_overhead", "node_memory_limit")},
    }
    if cluster["node_jobs"] > MAXIMUM_NODE_JOBS:
        msg = (
            f"execution.run.node_jobs must be between 1 and {MAXIMUM_NODE_JOBS}, the per-user cap of node jobs on the shared 32-core "
            f"standard nodes, got {cluster['node_jobs']}."
        )
        raise GenerationConfigError(msg)
    for key in ("memory_per_case", "node_memory_overhead", "node_memory_limit"):
        parse_slurm_memory_mib(cluster[key], label=f"execution.node_job.{key}")
    cluster["node_job_wall_time"] = _validated_node_job_wall_time(
        node_job["node_job_wall_time"],
        timeout_seconds=float(runtime["timeout_seconds"]),
    )
    options = node_job["scheduler_options"]
    if not isinstance(options, list) or not all(isinstance(item, str) and item.startswith("--") for item in options):
        msg = "execution.node_job.scheduler_options must be long scheduler arguments."
        raise GenerationConfigError(msg)
    option_names = [re.split(r"[=\s]", option, maxsplit=1)[0] for option in options]
    if any(name in _SCHEDULER_OWNED_OPTIONS or name.startswith(_SCHEDULER_OWNED_OPTION_PREFIXES) for name in option_names):
        msg = "execution.node_job.scheduler_options cannot override pipeline-owned allocation directives."
        raise GenerationConfigError(msg)
    cluster["scheduler_options"] = options
    cluster["cores_per_node"] = site["cores_per_node"]
    cluster["partition"] = site["partition"]
    _derive_node_job_resources(cluster, cores_per_node=site["cores_per_node"])

    return {
        "schema_kind": authored["schema_kind"],
        "site": site,
        "runtime": runtime,
        "retention_policy": retention,
        "submission": submission,
        "cluster": cluster,
    }


def resolve_execution_values(
    path: Path | str,
    *,
    campaign_purpose: str = "source_generation",
    execution_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Load one execution YAML, apply explicit run overrides and return the validated execution values.

    This is the single resolver of the execution contract: campaign loading and read-only
    previews of the scheduler footprint use it, so both see the same derived node-job values.
    """
    raw = _load_yaml(Path(path).expanduser().resolve(), label="generation execution configuration")
    return _validate_execution(_apply_execution_overrides(raw, execution_overrides), campaign_purpose=campaign_purpose)


_MATERIAL_PARAMETER_SEMANTIC_KEYS: Final = frozenset(
    {
        "alpha",
        "atomic_record",
        "block",
        "classification",
        "components",
        "derivation",
        "distribution",
        "kind",
        "lower",
        "maximum_each",
        "minimum_each",
        "nominal",
        "ood",
        "ood_group",
        "profile_applicability",
        "selection",
        "sets",
        "sources",
        "transform",
        "unit",
        "units",
        "upper",
        "value",
    }
)
_MATERIAL_PARAMETER_PROVENANCE_KEYS: Final = frozenset({"provenance", "ood_provenance"})
_OPERATION_PARAMETER_SEMANTIC_KEYS: Final = _MATERIAL_PARAMETER_SEMANTIC_KEYS | {"record_id"}
_ATOMIC_RECORD_SEMANTIC_KEYS: Final = frozenset(
    {
        "components",
        "equation",
        "ood_records",
        "record_id",
        "reference",
        "rho_bu_dry_ref_support",
        "selection_rule",
        "temperature_variable",
        "units",
    }
)
_COUPLED_RECORD_SEMANTIC_KEYS: Final = frozenset(
    {
        "block",
        "components",
        "natural_packing_support",
        "ood_group",
        "records",
        "selection_rule",
        "units",
    }
)
_MATERIAL_PROVENANCE_KEYS: Final = frozenset({"effective_parameter_provenance", "material_scope"})
_SCIENTIFIC_IDENTITY_KEYS: Final = (
    "schema_kind",
    "simulation_profile",
    "reference_template",
    "campaign_seed",
    "domain_profile",
    "material",
    "operation_config_digest",
    "sampling_regime",
    "case_count",
    "case_index_start",
    "sampling",
    "composition_sources",
    "parameter_ood",
    "assignments",
    "scientific_fixed_values",
    "stationary_fixed_ownership",
    "grid",
    "input_contract",
    "output_contract",
    "storage",
    "available_learning_views",
    "airflow_source",
    "replacement_design",
    "boundary_schedule",
    "physical_formulas",
    "validation",
    "time",
    "schedule_time",
    "operation_constraints",
    "steady_flow_conditioning",
)
_SCIENTIFIC_PROVENANCE_KEYS: Final = frozenset(
    {
        "campaign_id",
        "campaign_purpose",
        "natural_support_state",
        "material_config_digest",
        "registry_metadata",
        "scientific_fixed_records",
        "grid_provenance",
        "physical_formulas_provenance",
        "time_provenance",
        "schedule_time_provenance",
    }
)
_CASE_INPUT_IDENTITY_KEYS: Final = (
    "campaign_seed",
    "material",
    "operation_config_digest",
    "sampling_regime",
    "case_count",
    "sampling",
    "composition_sources",
    "parameter_ood",
    "assignments",
    "scientific_fixed_values",
    "stationary_fixed_ownership",
    "grid",
    "input_contract",
    "replacement_design",
    "boundary_schedule",
    "physical_formulas",
    "time",
    "schedule_time",
    "operation_constraints",
)


def _selected_mapping(
    value: Mapping[str, Any],
    *,
    allowed: frozenset[str],
    ignored: frozenset[str],
    label: str,
) -> dict[str, Any]:
    """Select declared semantic fields and reject unclassified dependencies."""
    unknown = sorted(set(value).difference(allowed | ignored))
    if unknown:
        message = f"{label} contains unclassified identity fields: {unknown}."
        raise GenerationConfigError(message)
    return {key: copy.deepcopy(value[key]) for key in value if key in allowed}


def _boundary_schedule_identity(value: Any) -> dict[str, Any]:
    """Return only active startup-ramp semantics for scientific identity."""
    schedule = _mapping(value, label="resolved boundary schedule identity")
    ramp = _mapping(
        schedule.get("startup_ramp"),
        label="resolved boundary schedule identity startup_ramp",
    )
    enabled = ramp.get("enabled")
    if not isinstance(enabled, bool):
        message = "Resolved boundary-schedule identity requires boolean startup-ramp enabled."
        raise GenerationConfigError(message)
    if not enabled:
        return {"startup_ramp": {"enabled": False}}
    return {"startup_ramp": copy.deepcopy(dict(ramp))}


def _operation_identity_config(value: Any) -> dict[str, Any]:
    """Return active operation values and rules without evidence provenance."""
    operation = _mapping(value, label="resolved operation identity")
    _exact_keys(
        operation,
        required={"operation_id", "parameter_values", "constraints"},
        optional={"boundary_schedule"},
        label="resolved operation identity",
    )
    parameter_values = _mapping(
        operation["parameter_values"],
        label="resolved operation parameter values",
    )
    result = {
        "operation_id": copy.deepcopy(operation["operation_id"]),
        "parameter_values": {
            name: _selected_mapping(
                _mapping(record, label=f"resolved operation parameter {name}"),
                allowed=_OPERATION_PARAMETER_SEMANTIC_KEYS,
                ignored=_MATERIAL_PARAMETER_PROVENANCE_KEYS,
                label=f"resolved operation parameter {name}",
            )
            for name, record in parameter_values.items()
        },
        "constraints": copy.deepcopy(operation["constraints"]),
    }
    if "boundary_schedule" in operation:
        result["boundary_schedule"] = _boundary_schedule_identity(operation["boundary_schedule"])
    return result


def _material_identity_config(value: Any) -> dict[str, Any]:
    """Return exact material values and rules without source-description provenance."""
    material = _mapping(value, label="resolved material identity")
    required = frozenset(
        {
            "material_family",
            "packing_porosity_mean_support",
            "porosity_coupling",
            "parameter_registry",
            "active_sampling_blocks",
            "active_coordinate_names",
            "coupled_ood_records",
            "atomic_records",
        }
    )
    optional = frozenset({"initial_moisture_bounds", "initial_moisture_field_constraint"})
    unknown = sorted(set(material).difference(required | optional | _MATERIAL_PROVENANCE_KEYS))
    missing = sorted(required.difference(material))
    if missing or unknown:
        message = f"Resolved material identity fields are unclassified: missing={missing}, unknown={unknown}."
        raise GenerationConfigError(message)
    registry = _mapping(material["parameter_registry"], label="resolved material parameter registry")
    material["parameter_registry"] = {
        name: _selected_mapping(
            _mapping(spec, label=f"resolved material parameter {name}"),
            allowed=_MATERIAL_PARAMETER_SEMANTIC_KEYS,
            ignored=_MATERIAL_PARAMETER_PROVENANCE_KEYS,
            label=f"resolved material parameter {name}",
        )
        for name, spec in registry.items()
    }
    atomic = _mapping(material["atomic_records"], label="resolved material atomic records")
    material["atomic_records"] = {
        name: _selected_mapping(
            _mapping(record, label=f"resolved atomic record {name}"),
            allowed=_ATOMIC_RECORD_SEMANTIC_KEYS,
            ignored=_MATERIAL_PARAMETER_PROVENANCE_KEYS,
            label=f"resolved atomic record {name}",
        )
        for name, record in atomic.items()
    }
    coupled = _mapping(material["coupled_ood_records"], label="resolved material coupled OOD records")
    material["coupled_ood_records"] = {
        name: _selected_mapping(
            _mapping(record, label=f"resolved coupled OOD record {name}"),
            allowed=_COUPLED_RECORD_SEMANTIC_KEYS,
            ignored=_MATERIAL_PARAMETER_PROVENANCE_KEYS,
            label=f"resolved coupled OOD record {name}",
        )
        for name, record in coupled.items()
    }
    support = _mapping(
        material["packing_porosity_mean_support"],
        label="resolved packing porosity support",
    )
    material["packing_porosity_mean_support"] = _selected_mapping(
        support,
        allowed=frozenset({"lower", "upper", "unit"}),
        ignored=frozenset({"provenance"}),
        label="resolved packing porosity support",
    )
    return {key: copy.deepcopy(material[key]) for key in (*sorted(required), *sorted(optional)) if key in material}


def _assignment_identity(value: Any, *, index: int) -> dict[str, Any]:
    """Return one case assignment projected to seed and input semantics."""
    assignment = _mapping(value, label=f"resolved assignment {index}")
    semantic = frozenset(
        {
            "case_index",
            "design_index",
            "material_family",
            "sampling_regime",
            "assignment_role",
            "ood_group",
            "ood_unit_id",
            "ood_units_per_case",
        }
    )
    return _selected_mapping(
        assignment,
        allowed=semantic,
        ignored=frozenset(),
        label=f"resolved assignment {index}",
    )


def _scientific_identity_config(scientific: Mapping[str, Any]) -> dict[str, Any]:
    """Return the positive dependency payload for generated scientific content."""
    value = _mapping(scientific, label="resolved scientific batch")
    allowed = frozenset(_SCIENTIFIC_IDENTITY_KEYS)
    unknown = sorted(set(value).difference(allowed | _SCIENTIFIC_PROVENANCE_KEYS))
    if unknown:
        message = f"Resolved scientific batch contains unclassified identity fields: {unknown}."
        raise GenerationConfigError(message)
    missing = sorted(
        {
            "schema_kind",
            "simulation_profile",
            "reference_template",
            "campaign_seed",
            "material",
            "sampling_regime",
            "case_count",
            "sampling",
            "assignments",
            "scientific_fixed_values",
            "grid",
            "input_contract",
            "output_contract",
            "storage",
            "operation_constraints",
        }.difference(value)
    )
    if missing:
        message = f"Resolved scientific batch identity is incomplete: {missing}."
        raise GenerationConfigError(message)
    result: dict[str, Any] = {key: copy.deepcopy(value[key]) for key in _SCIENTIFIC_IDENTITY_KEYS if key in value}
    result["material"] = _material_identity_config(value["material"])
    if "boundary_schedule" in result:
        result["boundary_schedule"] = _boundary_schedule_identity(value["boundary_schedule"])
    assignments = value["assignments"]
    if not isinstance(assignments, list):
        message = "Resolved scientific batch assignments must be a list."
        raise TypeError(message)
    result["assignments"] = [_assignment_identity(assignment, index=index) for index, assignment in enumerate(assignments)]
    return result


def _input_identity_config(scientific: Mapping[str, Any]) -> dict[str, Any]:
    """Return the positive dependency payload that determines adapter inputs."""
    semantic = _scientific_identity_config(scientific)
    return {key: copy.deepcopy(semantic[key]) for key in _CASE_INPUT_IDENTITY_KEYS if key in semantic}


def _campaign_references(campaign: dict[str, Any], *, source_path: Path) -> None:
    """Resolve each explicit layered campaign reference in place."""
    for key in _CAMPAIGN_REFERENCES:
        campaign[key] = _reference_path(campaign[key], source_path=source_path, label=f"campaign.{key}")


def _validate_campaign_header(
    value: Any,
    *,
    source_path: Path,
) -> dict[str, Any]:
    """Validate one concise campaign and derive its canonical material union."""
    campaign = _mapping(value, label="generation campaign configuration")
    if campaign.get("schema_kind") != "generation_source_campaign":
        message = (
            f"Unsupported generation campaign schema in {source_path}: found {campaign.get('schema_kind')!r}, expected 'generation_source_campaign'."
        )
        raise GenerationConfigError(message)
    required = {"schema_kind", *_CAMPAIGN_REFERENCES, "materials", "sampling"}
    _exact_keys(campaign, required=required, optional={"revision", "minimum_success_rate"}, label="generation campaign configuration")
    campaign["campaign_purpose"] = "source_generation"
    _campaign_references(campaign, source_path=source_path)
    campaign["materials"] = _name_list(campaign["materials"], label="campaign.materials", allow_empty=False)
    return campaign


def _build_batch(
    *,
    source_path: Path,
    profile: profiles.SimulationProfile,
    template: templates.ResolvedTemplateIdentity,
    material: Mapping[str, Any],
    sampling_regime: str,
    case_count: int,
    case_start: int,
    campaign_seed: int,
    parameter_ood: Mapping[str, Any],
    common_config: Mapping[str, Any],
    registry_metadata: Mapping[str, Mapping[str, str]],
    sampling_block_contract: Mapping[str, tuple[str, ...]],
    operations_digest: str,
    material_digest: str,
    execution: Mapping[str, Any],
    campaign_id: str,
    campaign_name: str,
    campaign_purpose: str,
    material_role: str,
    evaluation_regime: str,
) -> GenerationConfig:
    """Build one immutable profile-projected batch: the first ``case_count`` points of one fixed design (its ``sampling_regime`` kind)."""
    material_family = str(material["material_family"])
    batch_name = build_batch_name(profile.id, material_family, sampling_regime)
    batch_seed = seeding.design_seed(campaign_seed, material_family, sampling_regime)
    registry = material["parameter_registry"]
    groups = materials.active_ood_groups(registry, profile.id)
    eligible_units = sampling_service.eligible_ood_units(material, groups=groups) if sampling_regime == "parameter_ood" else ()
    assignments = {
        case_start + index: design_point_assignment(
            material_family,
            sampling_regime,
            material_role=material_role,
            case_index=case_start + index,
            design_index=index,
            eligible_units=eligible_units,
        )
        for index in range(case_count)
    }
    ood_allocation = sampling_service.allocate_ood_units(eligible_units, case_count=case_count) if sampling_regime == "parameter_ood" else ()
    resolved_parameter_ood = copy.deepcopy(dict(parameter_ood))
    resolved_parameter_ood["eligible_units"] = [copy.deepcopy(dict(unit)) for unit in eligible_units]
    resolved_parameter_ood["case_allocation"] = [
        {"case_index": case_start + index, **copy.deepcopy(unit)} for index, unit in enumerate(ood_allocation)
    ]
    allocation_counts = dict.fromkeys((unit["unit_id"] for unit in eligible_units), 0)
    for unit in ood_allocation:
        allocation_counts[unit["unit_id"]] += 1
    resolved_parameter_ood["allocation_counts"] = allocation_counts
    case_indices = tuple(assignments)
    active_blocks = tuple(material["active_sampling_blocks"])
    active_block_parameters = {block: tuple(name for name in sampling_block_contract[block] if name in registry) for block in active_blocks}

    sampling_plan = sampling_service.build_sampling_plan(
        registry=registry,
        case_count=case_count,
        design_seed=batch_seed,
        blocks=active_blocks,
        block_parameters=active_block_parameters,
    )
    input_contract = copy.deepcopy(dict(common_config["input_contract"]))
    spatial_contract = input_contract["spatial"]
    spatial_contract["columns"] = spatial_contract.pop("columns_by_profile")[profile.id]
    if profile.id == profiles.STEADY_FLOW_PROFILE:
        input_contract.pop("scalar")
        input_contract.pop("schedule")
    output_contract = {
        "exports_root": "exports",
        "exports": common_config["_profile_exports"],
        "static_field_names": list(profiles.static_field_names(profile.id)),
        "static_field_units": list(profiles.static_field_units(profile.id)),
        "transient_field_names": (list(profiles.TRANSIENT_FIELD_NAMES) if profile.id == profiles.TRANSIENT_DRYING_PROFILE else []),
        "transient_field_units": (list(profiles.TRANSIENT_FIELD_UNITS) if profile.id == profiles.TRANSIENT_DRYING_PROFILE else []),
        "global_field_names": (list(profiles.GLOBAL_FIELD_NAMES) if profile.id == profiles.TRANSIENT_DRYING_PROFILE else []),
        "global_field_units": (list(profiles.GLOBAL_FIELD_UNITS) if profile.id == profiles.TRANSIENT_DRYING_PROFILE else []),
    }
    fixed_names = profiles.STATIONARY_FIXED_FIELDS if profile.id == profiles.STEADY_FLOW_PROFILE else tuple(common_config["scientific_fixed_values"])
    fixed_values = {name: copy.deepcopy(common_config["scientific_fixed_values"][name]) for name in fixed_names}
    fixed_records = {name: copy.deepcopy(common_config["_scientific_fixed_records"][name]) for name in fixed_names}
    active_registry_metadata = {name: copy.deepcopy(registry_metadata[name]) for name in registry}
    scientific: dict[str, Any] = {
        "schema_kind": "resolved_generation_batch",
        "simulation_profile": profile.id,
        "reference_template": {"sha256": template.sha256},
        "campaign_id": campaign_id,
        "campaign_purpose": campaign_purpose,
        "campaign_seed": campaign_seed,
        "domain_profile": evaluation_regime,
        "natural_support_state": ("parameter_ood" if sampling_regime == "parameter_ood" else "natural"),
        "material": copy.deepcopy(dict(material)),
        "material_config_digest": material_digest,
        "operation_config_digest": operations_digest,
        "sampling_regime": sampling_regime,
        "case_count": case_count,
        "sampling": sampling_plan,
        "parameter_ood": resolved_parameter_ood,
        "assignments": [assignments[index] for index in case_indices],
        "registry_metadata": active_registry_metadata,
        "scientific_fixed_values": fixed_values,
        "scientific_fixed_records": fixed_records,
        "stationary_fixed_ownership": {
            name: {
                "owner": "package_fixed",
                "unit": unit,
                "fixed_value": float(common_config["scientific_fixed_values"][name]),
            }
            for name, unit in zip(
                profiles.STATIONARY_FIXED_FIELDS,
                profiles.STATIONARY_FIXED_UNITS,
                strict=True,
            )
        },
        "grid": copy.deepcopy(dict(common_config["grid"])),
        "grid_provenance": copy.deepcopy(dict(common_config["grid_provenance"])),
        "input_contract": input_contract,
        "output_contract": output_contract,
        "storage": copy.deepcopy(dict(common_config["storage"])),
        "available_learning_views": list(profile.available_learning_views),
        "airflow_source": profile.airflow_source,
    }
    scientific["case_index_start"] = case_start
    if profile.id == profiles.TRANSIENT_DRYING_PROFILE:
        scientific.update(
            {
                "boundary_schedule": copy.deepcopy(dict(common_config["_boundary_schedule"])),
                "physical_formulas": copy.deepcopy(dict(common_config["physical_formulas"])),
                "physical_formulas_provenance": copy.deepcopy(dict(common_config["physical_formulas_provenance"])),
                "validation": copy.deepcopy(dict(common_config["validation"])),
                "time": copy.deepcopy(dict(common_config["time"])),
                "time_provenance": copy.deepcopy(dict(common_config["time_provenance"])),
                "operation_constraints": copy.deepcopy(dict(common_config["_operation_constraints"])),
            }
        )
        if "schedule_time" in common_config:
            scientific["schedule_time"] = copy.deepcopy(dict(common_config["schedule_time"]))
            scientific["schedule_time_provenance"] = copy.deepcopy(dict(common_config["schedule_time_provenance"]))
    else:
        scientific["operation_constraints"] = {
            "porosity_natural_support_policy": common_config["_operation_constraints"]["porosity_natural_support_policy"]
        }
    scientific["steady_flow_conditioning"] = copy.deepcopy(common_config["_steady_flow_conditioning"])

    scientific_digest = compute_scientific_config_digest(scientific)
    case_input_digest = compute_case_input_config_digest(scientific)
    batch_id = build_batch_id(batch_name, scientific_digest)
    batch_storage_name = f"source~{profile.id}~{campaign_name}~{material_family}~{sampling_regime}"
    if len(batch_storage_name.encode("ascii")) > _MAX_FILESYSTEM_COMPONENT_BYTES:
        message = "Campaign batch locator exceeds the filesystem component limit."
        raise GenerationConfigError(message)
    return GenerationConfig(
        source_path=source_path,
        profile=profile,
        material_family=material_family,
        material_role=material_role,
        evaluation_regime=evaluation_regime,
        sampling_regime=sampling_regime,
        batch_name=batch_name,
        scientific_values=scientific,
        execution_values=copy.deepcopy(dict(execution)),
        template=template,
        case_indices=case_indices,
        seed_base=batch_seed,
        assignments=assignments,
        scientific_config_digest=scientific_digest,
        case_input_config_digest=case_input_digest,
        batch_identity=scientific_digest,
        batch_id=batch_id,
        batch_storage_name=batch_storage_name,
    )


def _apply_design_overrides(campaign: dict[str, Any], overrides: Mapping[str, Any]) -> None:
    """Apply an explicit material subset and Source revision."""
    if "revision" in overrides:
        campaign["revision"] = overrides["revision"]
    if "materials" in overrides:
        selected = _name_list(overrides["materials"], label="run.materials", allow_empty=False)
        if not set(selected).issubset(campaign["materials"]):
            message = "Run materials must select configured campaign materials."
            raise GenerationConfigError(message)
        campaign["materials"] = tuple(material for material in campaign["materials"] if material in selected)
        designs = _mapping(campaign["sampling"].get("designs"), label="campaign.sampling.designs")
        campaign["sampling"]["designs"] = {material: kinds for material, kinds in designs.items() if material in selected}


def validate_minimum_success_rate(value: Any, *, label: str) -> float:
    """Return one admitted minimum success rate s, a real number with 0 < s <= 1 that is neither a boolean nor a mapping.

    Raises
    ------
    GenerationConfigError
        If ``value`` is not such a number.

    """
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 < float(value) <= 1.0:
        message = f"{label} must be a number s with 0 < s <= 1, got {value!r}."
        raise GenerationConfigError(message)
    return float(value)


def authored_minimum_success_rate(authored: Mapping[str, Any] | None) -> float | None:
    """Return the minimum success rate a Dataset plan pre-registers for every original batch, or ``None`` without one."""
    if authored is None or "minimum_success_rate" not in authored:
        return None
    return validate_minimum_success_rate(authored["minimum_success_rate"], label="campaign.minimum_success_rate")


def _validated_stall_stop(value: Any) -> dict[str, float]:
    """Validate the optional stall rule; a check time at or after the timeout's orderly stop leaves only the timeout in effect."""
    rule = _mapping(value, label="execution.rules.stall_stop")
    _exact_keys(rule, required={"check_after_seconds", "minimum_simulated_seconds"}, optional=set(), label="execution.rules.stall_stop")
    check_after = _finite(rule["check_after_seconds"], label="execution.rules.stall_stop.check_after_seconds")
    minimum = _finite(rule["minimum_simulated_seconds"], label="execution.rules.stall_stop.minimum_simulated_seconds")
    if check_after <= 0.0 or minimum <= 0.0:
        message = "execution.rules.stall_stop needs positive check_after_seconds and minimum_simulated_seconds."
        raise GenerationConfigError(message)
    return {"check_after_seconds": check_after, "minimum_simulated_seconds": minimum}


def physical_horizon_hours(time: Mapping[str, Any]) -> float:
    """Return the physical horizon in hours of one validated or authored common ``time`` section."""
    return float(time["stop"]) - float(time["start"])


def _apply_duration_override(common_config: dict[str, Any], overrides: Mapping[str, Any]) -> None:
    """Change the physical horizon while retaining the sampled schedule horizon."""
    if "duration_hours" in overrides:
        duration = _finite(overrides["duration_hours"], label="run.duration_hours")
        if duration <= 0:
            message = "Run duration_hours must be strictly positive."
            raise GenerationConfigError(message)
        original_time = copy.deepcopy(common_config["time"])
        original_schedule = copy.deepcopy(common_config.get("schedule_time", original_time))
        shortened = {key: original_time[key] for key in ("start", "stop", "interval")}
        shortened["stop"] = shortened["start"] + duration
        common_config["time"] = _validate_time(shortened)
        if common_config["time"]["stop"] < original_schedule["stop"]:
            common_config["schedule_time"] = original_schedule
            common_config["schedule_time_provenance"] = copy.deepcopy(common_config.get("schedule_time_provenance", common_config["time_provenance"]))
        else:
            common_config.pop("schedule_time", None)
            common_config.pop("schedule_time_provenance", None)


def load_campaign_config(  # noqa: PLR0915 -- one centralized campaign resolver
    path: Path | str,
    *,
    require_executable: bool = True,
    authored_value: Mapping[str, Any] | None = None,
    scientific_overrides: Mapping[str, Any] | None = None,
    execution_overrides: Mapping[str, Any] | None = None,
) -> CampaignConfig:
    """Resolve one campaign, every subbatch, and every immutable dataset plan."""
    source_path = Path(path).expanduser().resolve()
    campaign = _validate_campaign_header(
        _load_yaml(source_path, label="generation campaign configuration") if authored_value is None else copy.deepcopy(dict(authored_value)),
        source_path=source_path,
    )
    overrides = {} if scientific_overrides is None else copy.deepcopy(dict(scientific_overrides))
    unknown_overrides = set(overrides).difference({"materials", "duration_hours", "revision"})
    if unknown_overrides:
        message = f"Unsupported scientific run overrides: {sorted(unknown_overrides)}."
        raise GenerationConfigError(message)
    _apply_design_overrides(campaign, overrides)
    if authored_value is None and "minimum_success_rate" in campaign:
        message = "minimum_success_rate is pre-registered by a Dataset plan; lower it for other campaigns with amend-replacements."
        raise GenerationConfigError(message)
    plan_rate = authored_minimum_success_rate(campaign)
    registry_config = _load_yaml(
        campaign["registry_config"],
        label="generation parameter registry",
    )
    definitions, registry_metadata, sampling_block_contract = materials.validate_semantic_registry(registry_config)
    source_config = _load_yaml(
        campaign["sources_config"],
        label="generation scientific source registry",
    )
    sources = provenance_service.validate_source_registry(source_config)
    common_config = _validate_common(
        _load_yaml(
            campaign["common_config"],
            label="generation common configuration",
        ),
        sources=sources,
    )
    configured_horizon = physical_horizon_hours(common_config["time"])
    _apply_duration_override(common_config, overrides)
    operations = _validate_operations(
        _load_yaml(
            campaign["operations_config"],
            label="generation operations configuration",
        ),
        sources=sources,
        definitions=definitions,
        time_contract=common_config.get("schedule_time", common_config["time"]),
    )
    bed_fields = _validate_bed_fields(
        _load_yaml(campaign["bed_fields_config"], label="generation bed-field configuration"),
        sources=sources,
        definitions=definitions,
    )
    operations["parameter_values"] = {**bed_fields, **operations["parameter_values"]}
    profile_raw = _load_yaml(
        campaign["profile_config"],
        label="generation profile configuration",
    )
    profile_id = profile_raw.get("simulation_profile")
    if not isinstance(profile_id, str):
        message = "generation profile simulation_profile must be text."
        raise TypeError(message)
    profile = profiles.resolve_profile(profile_id)
    if ("schedule_time" in common_config or "duration_hours" in overrides) and profile.id != profiles.TRANSIENT_DRYING_PROFILE:
        message = "common.schedule_time is supported only for transient_drying."
        raise GenerationConfigError(message)
    profile_config, template = _validate_profile_config(
        profile_raw,
        profile=profile,
        require_executable=require_executable,
    )
    campaign_purpose = str(campaign["campaign_purpose"])
    execution = resolve_execution_values(
        campaign["execution_config"],
        campaign_purpose=campaign_purpose,
        execution_overrides=execution_overrides,
    )
    if plan_rate is not None and "max_admission_cases" not in (execution_overrides or {}):
        # Without an explicit cap a Dataset plan's campaign records none; it derives the cap from its batches' current rates.
        execution["submission"]["max_admission_cases"] = None
    campaign_name = ""

    resolved_materials: dict[str, dict[str, Any]] = {}
    material_digests: dict[str, str] = {}
    for material_family in campaign["materials"]:
        material_path = materials.material_config_directory() / f"{material_family}.yaml"
        material_raw = _load_yaml(
            material_path,
            label=f"material {material_family} configuration",
        )
        if material_raw.get("material_family") != material_family:
            message = f"Material filename and material_family disagree for {material_path}."
            raise GenerationConfigError(message)
        full_material = materials.resolve_material_definition(
            definitions,
            registry_metadata,
            common_config["parameter_values"],
            operations["parameter_values"],
            material_raw,
            sources=sources,
        )
        projected = materials.project_material_for_profile(
            full_material,
            profile.id,
            sampling_block_contract=sampling_block_contract,
        )
        resolved_materials[material_family] = projected
        material_digests[material_family] = common.serialization.canonical_json_sha256(projected)

    sampling = _mapping(campaign["sampling"], label="campaign.sampling")
    _exact_keys(sampling, required={"seed_base", "designs"}, optional=set(), label="campaign.sampling")
    campaign_seed = _uint32(
        sampling["seed_base"],
        label="campaign.sampling.seed_base",
    )
    material_roles = {"seen": campaign["materials"], "near_family_ood": (), "far_family_ood": (), "extreme_family_ood": ()}
    designs = _validate_designs(sampling["designs"], materials_in_order=campaign["materials"])
    domain_profile = campaign_identity.domain_profile(designs)
    sampling_regimes = tuple(kind for kind in DESIGN_KINDS if any(kind in kinds for kinds in designs.values()))
    selected_targets = {material: sum(kinds.values()) for material, kinds in designs.items()}
    name_horizon = (
        campaign_identity.deviating_horizon_hours(physical_horizon_hours(common_config["time"]), configured_horizon)
        if profile.id == profiles.TRANSIENT_DRYING_PROFILE
        else None
    )
    try:
        campaign_name = campaign_identity.canonical_campaign_name(designs, root_seed=campaign_seed, horizon_hours=name_horizon)
    except ValueError as error:
        raise GenerationConfigError(str(error)) from error
    composition_sources: tuple[dict[str, Any], ...] = ()
    campaign_id = campaign_name

    membership: dict[str, Any] = {}
    if campaign_purpose == "source_generation":
        ood_groups = materials.active_ood_groups(definitions, profile.id)
        parameter_ood = _validate_parameter_ood_policy(
            {
                "groups": list(ood_groups),
                "units_per_case": 1,
                "allocation_strategy": "canonical_eligible_unit_round_robin",
                "eligibility_source": "resolved_profile_projected_registry",
            },
            groups=ood_groups,
        )
    else:
        parameter_ood = {
            "groups": [],
            "units_per_case": 0,
            "allocation_strategy": "not_applicable",
            "eligibility_source": "not_applicable",
        }
    dataset_packages: tuple[dict[str, Any], ...] = ()
    evaluation_regimes: tuple[str, ...] = ()
    material_memberships: dict[str, tuple[str, ...]] = {}

    common_with_profile = copy.deepcopy(common_config)
    common_with_profile["_profile_exports"] = profile_config["exports"]
    common_with_profile["_steady_flow_conditioning"] = _steady_flow_conditioning(common_config["scientific_fixed_values"])
    common_with_profile["_operation_constraints"] = operations["constraints"]
    common_with_profile["_boundary_schedule"] = operations["boundary_schedule"]
    active_names = set(materials.profile_parameter_names(definitions, profile.id))
    active_operation = {name: operations["parameter_values"][name] for name in operations["parameter_values"] if name in active_names}
    active_constraints = (
        operations["constraints"]
        if profile.id == profiles.TRANSIENT_DRYING_PROFILE
        else {"porosity_natural_support_policy": operations["constraints"]["porosity_natural_support_policy"]}
    )
    operation_identity = {
        "operation_id": operations["operation_id"],
        "parameter_values": active_operation,
        "constraints": active_constraints,
        **({"boundary_schedule": operations["boundary_schedule"]} if profile.id == profiles.TRANSIENT_DRYING_PROFILE else {}),
    }
    operations_digest = common.serialization.canonical_json_sha256(_operation_identity_config(operation_identity))

    batches: list[GenerationConfig] = []
    material_case_offsets: dict[str, int] = {}
    for sampling_regime in sampling_regimes:
        for material_family, kinds in designs.items():
            if sampling_regime not in kinds:
                continue
            count = kinds[sampling_regime]
            material_role = "source"
            evaluation_regime = "parameter_extension" if sampling_regime == "parameter_ood" else "reference_range"
            batches.append(
                _build_batch(
                    source_path=source_path,
                    profile=profile,
                    template=template,
                    material=resolved_materials[material_family],
                    sampling_regime=sampling_regime,
                    case_count=count,
                    case_start=material_case_offsets.get(material_family, 0) + 1,
                    campaign_seed=campaign_seed,
                    parameter_ood=parameter_ood,
                    common_config=common_with_profile,
                    registry_metadata=registry_metadata,
                    sampling_block_contract=sampling_block_contract,
                    operations_digest=operations_digest,
                    material_digest=material_digests[material_family],
                    execution=execution,
                    campaign_id=campaign_id,
                    campaign_name=campaign_name,
                    campaign_purpose=campaign_purpose,
                    material_role=material_role,
                    evaluation_regime=evaluation_regime,
                )
            )
            material_case_offsets[material_family] = material_case_offsets.get(material_family, 0) + count
    batch_names = [batch.batch_name for batch in batches]
    if len(batch_names) != len(set(batch_names)):
        message = "Campaign planned duplicate batch names."
        raise GenerationConfigError(message)
    batch_ids = [batch.batch_id for batch in batches]
    if len(batch_ids) != len(set(batch_ids)):
        message = "Campaign planned duplicate scientific batch identities."
        raise GenerationConfigError(message)
    total_case_count = sum(len(batch.case_indices) for batch in batches)

    duplicate_case_input_policy = "reject_duplicates"
    producer_identity = source_service.source_producer_identity()
    campaign_scientific = {
        "schema_kind": "resolved_generation_campaign",
        "producer_identity": producer_identity,
        "campaign_name": campaign_name,
        "campaign_id": campaign_id,
        "campaign_purpose": campaign_purpose,
        "simulation_profile": profile.id,
        "materials": list(campaign["materials"]),
        "domain_profile": domain_profile,
        "design_recipe": sampling_service.DESIGN_RECIPE,
        "designs": copy.deepcopy(designs),
        "sampling_regimes": list(sampling_regimes),
        "parameter_ood_policy": parameter_ood,
        "parameter_ood_allocations": {
            batch.batch_name: copy.deepcopy(batch.scientific_values["parameter_ood"]) for batch in batches if batch.sampling_regime == "parameter_ood"
        },
        "campaign_seed": campaign_seed,
        "total_case_count": total_case_count,
        "batch_ids": [batch.batch_id for batch in batches],
        "duplicate_case_input_policy": duplicate_case_input_policy,
    }
    campaign_digest = common.serialization.canonical_json_sha256(campaign_scientific)
    source_label = None
    if campaign.get("revision") is not None:
        try:
            source_label = campaign_identity.revised_source_label(campaign_name, campaign["revision"])
        except ValueError as error:
            raise GenerationConfigError(str(error)) from error
    source_directory = campaign_identity.source_name(campaign_name if source_label is None else source_label, campaign_digest)
    batches = [
        replace(batch, batch_storage_name=f"source~{profile.id}~{source_directory}~{batch.material_family}~{batch.sampling_regime}")
        for batch in batches
    ]
    package_request_digest = common.serialization.canonical_json_sha256(list(dataset_packages))
    return CampaignConfig(
        source_path=source_path,
        campaign_name=campaign_name,
        campaign_digest=campaign_digest,
        package_request_digest=package_request_digest,
        campaign_id=campaign_id,
        campaign_purpose=campaign_purpose,
        domain_profile=domain_profile,
        evaluation_regimes=evaluation_regimes,
        profile=profile,
        template=template,
        material_roles=copy.deepcopy(material_roles),
        material_memberships=material_memberships,
        membership=copy.deepcopy(membership),
        source_registry=copy.deepcopy(sources),
        source_files=(
            source_path,
            *(campaign[key] for key in _CAMPAIGN_REFERENCES),
            *(materials.material_config_directory() / f"{material_family}.yaml" for material_family in campaign["materials"]),
        ),
        total_case_count=total_case_count,
        batches=tuple(batches),
        dataset_packages=dataset_packages,
        duplicate_case_input_policy=duplicate_case_input_policy,
        execution_values=execution,
        root_seed=campaign_seed,
        designs=designs,
        selected_targets=selected_targets,
        composition_sources=composition_sources,
        scientific_overrides=overrides,
        producer_identity=producer_identity,
        execution_overrides=copy.deepcopy(dict(execution_overrides or {})),
        source_label=source_label,
        authored_value=None if authored_value is None else copy.deepcopy(dict(authored_value)),
    )


def discover_profile_template_identities(
    repository_root: Path | str | None = None,
) -> dict[str, templates.ResolvedTemplateIdentity]:
    """
    Discover configured template identities without a profile-to-path registry.

    Parameters
    ----------
    repository_root : Path | str | None, optional
        Repository tree containing maintained Generation profile YAML.

    Returns
    -------
    dict[str, ResolvedTemplateIdentity]
        Verified identities keyed by the profile IDs declared by configuration.

    Raises
    ------
    FileNotFoundError
        If the maintained profile configuration directory is absent.
    ValueError
        If declarations are missing, duplicated, unsupported, or invalid.

    """
    root = common.paths.get_project_root().resolve() if repository_root is None else Path(repository_root).expanduser().resolve()
    profile_root = root / "configs" / "01_generation" / "physical_model" / "profiles"
    if not profile_root.is_dir() or profile_root.is_symlink():
        message = f"Generation profile configuration directory is missing or unsafe: {profile_root}"
        raise FileNotFoundError(message)
    identities: dict[str, templates.ResolvedTemplateIdentity] = {}
    for candidate in sorted(profile_root.rglob("*.yaml")):
        if not candidate.is_file() or candidate.is_symlink():
            continue
        raw = _load_yaml(candidate, label="generation profile configuration")
        if raw.get("schema_kind") != "generation_profile":
            continue
        profile_id = raw.get("simulation_profile")
        if not isinstance(profile_id, str):
            message = f"Generation profile simulation_profile must be text: {candidate}"
            raise TypeError(message)
        profiles.resolve_profile(profile_id)
        if profile_id in identities:
            message = f"Generation profile {profile_id!r} is declared more than once under {profile_root}."
            raise ValueError(message)
        identities[profile_id] = templates.resolve_template_identity(
            raw.get("template"),
            repository_root=root,
        )
    if not identities:
        message = f"No maintained Generation profile templates were discovered under {profile_root}."
        raise ValueError(message)
    return dict(sorted(identities.items()))


def load_generation_config(
    path: Path | str,
    *,
    require_executable: bool = True,
    only_batch: str | None = None,
) -> GenerationConfig:
    """Resolve one predeclared batch from a campaign configuration."""
    campaign = load_campaign_config(path, require_executable=require_executable)
    if only_batch is not None:
        return campaign.batch(only_batch)
    if len(campaign.batches) != 1:
        message = f"Campaign {campaign.campaign_name!r} declares {len(campaign.batches)} batches; select one with only_batch."
        raise GenerationConfigError(message)
    return campaign.batches[0]
