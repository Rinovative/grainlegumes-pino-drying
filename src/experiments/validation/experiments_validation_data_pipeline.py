"""Validate the complete production data pipeline of a steady-flow experiment before it is run.

Responsibilities:
  - Refuse a transient-drying request, whose group scaling the transient loaders admit
  - Admit complete task-bound ID and OOD datasets, whose manifests the production loaders already bound
  - Verify deterministic split membership, leakage, and loader coverage
  - Independently check train-fitted and normalized per-channel statistics
  - Verify finite transforms, inverse transforms, and sampler-state restoration
  - Return typed scientific records without notebook presentation dependencies

Design principles:
  - Production dataset, split, normalizer, and loader services remain authoritative
  - Independent checks fail closed and never persist validation state
  - The module is imported only for explicit full-data acceptance

This module does NOT:
  - Construct models, losses, optimizers, schedulers, or training runs
  - Initialize W&B, allocate paths, write checkpoints, or generate artifacts
  - Build pandas tables or own notebook display behavior
"""

from __future__ import annotations

import gc
import resource
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
from torch import Tensor
from torch.utils.data import SequentialSampler, Subset

from src import datasets, domain
from src.experiments import experiments_run as run_service
from src.experiments.config import experiments_config_loader as config_loader

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.datasets.preprocessing.dataset_preprocessing_normalization import TrainScaling
    from src.domain.tasks.domain_task_spec import TaskSpec

_FULL_VALIDATION_RTOL = 3e-4
_FULL_VALIDATION_ATOL = 3e-5
_NORMALIZED_MEAN_ATOL = 8e-4
_NORMALIZED_SCALE_RTOL = 8e-4
_NORMALIZED_SCALE_ATOL = 8e-4
_TENSOR_RANK = 4
_OOD_LABELS: dict[datasets.preprocessing.splits.OodRole, str] = {"material_ood": "Material OOD", "parameter_ood": "Parameter OOD"}
ResultLabel = Literal["PASS", "INFO"]


@dataclass(frozen=True, slots=True)
class ValidationCheck:
    """Record one overall validation result."""

    check: str
    evidence: str
    result: ResultLabel


@dataclass(frozen=True, slots=True)
class DatasetMembershipRecord:
    """Record identity and membership evidence for one dataset role."""

    role: str
    dataset_id: str
    full_samples: int
    expected: int
    observed: int
    duplicates: int
    missing: int
    shape: str
    dtype: str
    fingerprint: str
    data_contract_digest: str
    finite: bool
    policy: str
    result: ResultLabel


@dataclass(frozen=True, slots=True)
class ChannelStatisticsRecord:
    """Record independent fitted and normalized statistics for one channel."""

    tensor_role: str
    channel: str
    fitted_mean: float
    fitted_scale: float
    normalized_mean: float
    normalized_scale: float
    finite: bool
    result: ResultLabel


@dataclass(frozen=True, slots=True)
class LoaderCoverageRecord:
    """Record complete traversal and preprocessing evidence for one loader."""

    loader: str
    sampler: str
    batches: int
    batch_size: int | None
    final_batch: int
    drop_last: bool
    inverse_checked: bool
    finite: bool
    result: ResultLabel


@dataclass(frozen=True, slots=True)
class FullDataValidationResult:
    """Return compact typed evidence from complete read-only validation."""

    overall: tuple[ValidationCheck, ...]
    dataset_membership: tuple[DatasetMembershipRecord, ...]
    channels: tuple[ChannelStatisticsRecord, ...]
    coverage: tuple[LoaderCoverageRecord, ...]
    elapsed_seconds: float
    peak_gib: float


@dataclass(slots=True)
class _ChannelMoments:
    """Accumulate float64 per-channel moments without retaining batches."""

    count: int = 0
    total: Tensor | None = None
    total_sq: Tensor | None = None
    finite: bool = True


@dataclass(frozen=True, slots=True)
class _DatasetContract:
    """Retain one fully admitted source dataset and its identity evidence."""

    source: Any
    identity: Any
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    dtype: str
    finite: bool


@dataclass(frozen=True, slots=True)
class _LoaderPass:
    """Retain one complete loader traversal result."""

    observed_ids: tuple[str, ...]
    final_batch_size: int
    inverse_checked: bool


def _require(condition: bool, message: str) -> None:
    """Fail one scientific invariant with an actionable validation error."""
    if not condition:
        detail = f"Full data-pipeline validation failed: {message}"
        raise RuntimeError(detail)


def _update_channel_moments(state: _ChannelMoments, tensor: Tensor) -> None:
    """Accumulate BCHW channel moments without retaining normalized batches."""
    _require(
        tensor.ndim == _TENSOR_RANK,
        f"expected BCHW tensor, got shape {tuple(tensor.shape)}",
    )
    state.finite = bool(state.finite and torch.isfinite(tensor).all().item())
    values = tensor.detach().to(dtype=torch.float64)
    reduced_sum = values.sum(dim=(0, 2, 3)).cpu()
    reduced_sum_sq = values.square().sum(dim=(0, 2, 3)).cpu()
    state.total = reduced_sum if state.total is None else state.total + reduced_sum
    state.total_sq = reduced_sum_sq if state.total_sq is None else state.total_sq + reduced_sum_sq
    state.count += int(values.shape[0] * values.shape[2] * values.shape[3])


def _finalize_channel_moments(state: _ChannelMoments) -> tuple[Tensor, Tensor]:
    """Return mean and population standard deviation, the definition the Train scaling fits."""
    _require(state.count > 0, "channel statistics require at least one element")
    _require(state.total is not None and state.total_sq is not None, "channel moment totals are absent")
    total = cast("Tensor", state.total)
    total_sq = cast("Tensor", state.total_sq)
    mean = total / state.count
    variance = (total_sq - total.square() / state.count) / state.count
    return mean, variance.clamp_min(0).sqrt()


def _batch_case_ids(batch: Mapping[str, Any]) -> list[str]:
    """Extract compact admitted case identities from one production batch."""
    metadata = batch.get("meta")
    _require(isinstance(metadata, Mapping), "batch metadata must be a mapping")
    metadata_mapping = cast("Mapping[str, Any]", metadata)
    raw_ids = metadata_mapping.get("case_id")
    if isinstance(raw_ids, Tensor):
        values: Iterable[Any] = raw_ids.tolist()
    elif isinstance(raw_ids, (list, tuple)):
        values = raw_ids
    else:
        values = (raw_ids,)
    case_ids = [str(value) for value in values]
    _require(all(case_ids), "batch case IDs must be non-empty")
    return case_ids


def _full_source_dataset(loader: Any) -> Any:
    """Return the production dataset beneath one admitted Subset."""
    selected = loader.dataset
    _require(isinstance(selected, Subset), "production loader dataset must be a torch Subset")
    return selected.dataset


def _components(source: Any) -> tuple[Any, ...]:
    """Return the verified packages beneath one possibly combined evaluation source."""
    return tuple(getattr(source, "sources", (source,)))


def _role_sample_ids(parts: Sequence[Any]) -> tuple[str, ...]:
    """Return the ordered case IDs a loader over admitted split parts yields in its batch metadata."""
    return tuple(case_id for part in parts for case_id in part.case_ids)


def _validate_tensor_contract(
    *,
    loader: Any,
    role: str,
    task: TaskSpec,
) -> _DatasetContract:
    """Validate a complete mounted dataset in bounded tensor chunks."""
    source = _full_source_dataset(loader)
    identity = source.identity
    inputs = source.data["inputs"]
    outputs = source.data["outputs"]
    expected_input_shape = (len(source), task.in_channels, *identity.spatial_shape)
    expected_output_shape = (len(source), task.out_channels, *identity.spatial_shape)
    _require(tuple(inputs.shape) == expected_input_shape, f"{role} input shape disagrees with TaskSpec")
    _require(tuple(outputs.shape) == expected_output_shape, f"{role} output shape disagrees with TaskSpec")
    _require(
        inputs.dtype == torch.float32 and outputs.dtype == torch.float32,
        f"{role} tensors must be float32",
    )
    _require(source.input_fields == list(task.input_names), f"{role} input field order changed")
    _require(source.output_fields == list(task.output_names), f"{role} output field order changed")
    _require(identity.task == task.id, f"{role} task identity changed")
    try:
        datasets.contracts.identity.validate_dataset_data_contract_digest(
            identity.data_contract_digest,
            task=task,
            label=f"{role} dataset data_contract_digest",
        )
    except (TypeError, ValueError) as error:
        detail = f"Full data-pipeline validation failed: {role} learned-data contract changed"
        raise RuntimeError(detail) from error
    _require(
        len(identity.sample_ids) == len(source),
        f"{role} sample IDs do not match sample count",
    )
    _require(
        len(set(identity.sample_ids)) == len(identity.sample_ids),
        f"{role} sample IDs contain duplicates",
    )
    metadata_ids = tuple(str(item["case_id"]) for item in source.data["source_metadata"])
    _require(metadata_ids == identity.sample_ids, f"{role} metadata case IDs are misaligned")
    finite = True
    for start in range(0, len(source), 16):
        finite = finite and bool(torch.isfinite(inputs[start : start + 16]).all().item())
        finite = finite and bool(torch.isfinite(outputs[start : start + 16]).all().item())
    _require(finite, f"{role} complete tensors contain NaN or infinity")
    return _DatasetContract(
        source=source,
        identity=identity,
        input_shape=tuple(inputs.shape),
        output_shape=tuple(outputs.shape),
        dtype=str(inputs.dtype).removeprefix("torch."),
        finite=finite,
    )


def _validate_loader_pass(
    *,
    loader: Any,
    role: str,
    expected_ids: tuple[str, ...],
    scaling: TrainScaling,
    raw_moments: dict[str, _ChannelMoments] | None = None,
    normalized_moments: dict[str, _ChannelMoments] | None = None,
) -> _LoaderPass:
    """Traverse one complete production membership and validate preprocessing."""
    observed: list[str] = []
    final_batch_size = 0
    inverse_checked = False
    input_mean, input_scale = scaling.input_affine()
    affines = {
        "input": (input_mean, input_scale, scaling.decode_inputs),
        "output": (scaling.group("state").mean, scaling.state_scale, scaling.decode_state),
    }
    boundary_names = datasets.preprocessing.normalization.STEADY_BOUNDARY_STATES
    input_names = domain.tasks.registry.get_task(scaling.task).input_names
    boundary = torch.tensor([name in boundary_names for name in input_names], device=scaling.device).view(1, -1, 1, 1)
    for batch in loader:
        x = batch["x"]
        y = batch["y"]
        _require(
            x.ndim == _TENSOR_RANK and y.ndim == _TENSOR_RANK,
            f"{role} batch is not BCHW",
        )
        _require(
            x.dtype == torch.float32 and y.dtype == torch.float32,
            f"{role} batch dtype changed",
        )
        _require(x.shape[1] == input_mean.numel(), f"{role} input channels changed")
        _require(y.shape[1] == scaling.state_scale.numel(), f"{role} output channels changed")
        reference = scaling.state_reference(x)
        x_normalized = scaling.encode_inputs(x, reference=reference)
        y_normalized = scaling.encode_state(y, reference=reference)
        # Physical units of the encoding's mean and scale per case: P for the states and the boundary inputs.
        per_case = reference.view(-1, 1, 1, 1)
        units = {"input": torch.where(boundary, per_case, torch.ones_like(per_case)), "output": per_case}
        _require(
            bool(torch.isfinite(x_normalized).all().item()),
            f"{role} normalized inputs are non-finite",
        )
        _require(
            bool(torch.isfinite(y_normalized).all().item()),
            f"{role} normalized outputs are non-finite",
        )
        if raw_moments is not None:
            _update_channel_moments(raw_moments["input"], x)
            _update_channel_moments(raw_moments["output"], y)
        if normalized_moments is not None:
            _update_channel_moments(normalized_moments["input"], x_normalized)
            _update_channel_moments(normalized_moments["output"], y_normalized)
        if not inverse_checked:
            for tensor_name, original, normalized in (("input", x, x_normalized), ("output", y, y_normalized)):
                mean, scale, decode = affines[tensor_name]
                reconstructed = decode(normalized, reference=reference)
                rounding_scale = original.abs() + (mean.abs().view(1, -1, 1, 1) + scale.view(1, -1, 1, 1)) * units[tensor_name] + 1.0
                rounding_bound = 32 * torch.finfo(original.dtype).eps * rounding_scale
                _require(
                    bool(torch.all((reconstructed - original).abs() <= rounding_bound).item()),
                    f"{role} {tensor_name} inverse transform exceeded the float32 rounding bound",
                )
            inverse_checked = True
        observed.extend(_batch_case_ids(batch))
        final_batch_size = int(x.shape[0])
    _require(len(observed) == len(expected_ids), f"{role} loader count changed")
    _require(len(set(observed)) == len(observed), f"{role} loader produced duplicate case IDs")
    _require(set(observed) == set(expected_ids), f"{role} loader omitted or added case IDs")
    return _LoaderPass(
        observed_ids=tuple(observed),
        final_batch_size=final_batch_size,
        inverse_checked=inverse_checked,
    )


def _fitted_channel_statistics(scaling: TrainScaling, names: tuple[str, ...]) -> tuple[Tensor, Tensor, Tensor]:
    """
    Return float64 Train mean, raw std and a fitted-channel mask of physical channels.

    An output carries its physical state statistics; a boundary input carries the dimensionless statistics of its state
    that encode it after the division by P.
    """
    boundary = datasets.preprocessing.normalization.STEADY_BOUNDARY_STATES
    static = scaling.group("static")
    means: list[float] = []
    stds: list[float] = []
    for name in names:
        source = scaling.group("state") if name in boundary else static if name in static.names else scaling.physical_state
        channel = boundary.get(name, name)
        index = source.names.index(channel)
        means.append(float(source.mean[index]))
        stds.append(float(source.std[index]))
    fitted = torch.tensor([name not in boundary for name in names])
    return torch.tensor(means, dtype=torch.float64), torch.tensor(stds, dtype=torch.float64), fitted


def _membership_contract(
    label: str,
    *,
    id_contract: _DatasetContract,
    ood_contracts: Mapping[str, tuple[_DatasetContract, ...]],
) -> _DatasetContract:
    """Return the tensor contract that owns one membership row: an OOD part's first source, otherwise the ID Dataset."""
    return ood_contracts[label][0] if label in ood_contracts else id_contract


def _membership_record(
    *,
    role: str,
    contract: _DatasetContract,
    expected_ids: tuple[str, ...],
    observed_ids: tuple[str, ...],
    shape: str,
    policy: str,
) -> DatasetMembershipRecord:
    """Build one exact membership record after validation succeeds."""
    duplicates = len(observed_ids) - len(set(observed_ids))
    missing = len(set(expected_ids).difference(observed_ids))
    return DatasetMembershipRecord(
        role=role,
        dataset_id=contract.identity.dataset_id,
        full_samples=len(contract.source),
        expected=len(expected_ids),
        observed=len(observed_ids),
        duplicates=duplicates,
        missing=missing,
        shape=shape,
        dtype=contract.dtype,
        fingerprint=contract.identity.fingerprint[:16],
        data_contract_digest=contract.identity.data_contract_digest[:16],
        finite=contract.finite,
        policy=policy,
        result="PASS",
    )


def validate_full_data_pipeline(config: Mapping[str, Any]) -> FullDataValidationResult:
    """
    Execute complete read-only production data admission and validation of a steady-flow experiment.

    The independent checks recompute per-sample ``(x, y)`` per-channel
    standardization and input-field identity. Every steady case covers the same
    grid, so the pooled independent moments equal the case-equal Train fit; the
    boundary input ``p_in_bc`` carries the statistics of ``p``. Transient drying scales
    deduplicated states, transitions and increments of trajectory items instead,
    so a transient request is refused before any data is loaded; its partition
    and scaling membership are admitted when the transient loaders are built.
    """
    started = time.perf_counter()
    peak_before_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
    effective = config_loader.validate_resolved_config(config)
    if effective["task"] == "transient_drying":
        message = (
            "The full data-pipeline validator checks per-sample steady_flow standardization; "
            "transient_drying admits its group scaling when its loaders are built."
        )
        raise ValueError(message)
    task = config_loader.validate_resolved_task_contract(effective)
    _require(
        task.tensor_layout == ("batch", "channel", "y", "x"),
        "TaskSpec layout must be BCHW",
    )
    _require(
        task.normalization_axes == (0, 2, 3),
        "normalization axes must preserve channels",
    )
    _require(
        task.preprocessing.fit_split == "train",
        "normalizer ownership must be train-only",
    )
    # A steady run has no stage; a transient request was refused above.
    seed_plan = run_service.build_seed_plan(int(effective["run"]["seed"]), stage=None)
    # The acceptance checks every role's loader coverage without a model; training never gets these held-out loaders.
    built = config_loader.create_dataloaders_from_config(
        effective,
        seed_plan=seed_plan,
        held_out_loaders=True,
    )
    train_loader = built["train"]
    evaluation_loaders = {
        "ID validation": built["eval"],
        "ID test": built["id_test"],
        **{label: built[role] for role, label in _OOD_LABELS.items() if built[role] is not None},
    }
    scaling: TrainScaling = built["scaling"]
    split = built["split_indices"]

    id_contract = _validate_tensor_contract(loader=train_loader, role="ID", task=task)
    ood_contracts: dict[str, tuple[_DatasetContract, ...]] = {
        label: tuple(
            _validate_tensor_contract(loader=SimpleNamespace(dataset=Subset(component, [])), role=f"{label} {index}", task=task)
            for index, component in enumerate(_components(_full_source_dataset(built[role])))
        )
        for role, label in _OOD_LABELS.items()
        if built[role] is not None
    }
    split_contract = datasets.preprocessing.splits.admit_split(split)
    _require(split_contract.train.dataset["fingerprint"] == id_contract.identity.fingerprint, "split Train identity differs from the ID Dataset")
    role_parts = {
        "ID train": (split_contract.train,),
        "ID validation": (split_contract.validation,),
        "ID test": (split_contract.id_test,),
        **{label: split_contract.ood_parts(role) for role, label in _OOD_LABELS.items()},
    }
    expected = {label: _role_sample_ids(role_parts[label]) for label in ("ID train", *evaluation_loaders)}
    id_sets = [set(split_contract.role(role).case_ids) for role in datasets.preprocessing.splits.ID_SPLIT_ROLES]
    _require(
        not any(left & right for index, left in enumerate(id_sets) for right in id_sets[index + 1 :]),
        "ID train/validation/test overlap",
    )

    sampler = train_loader.batch_sampler
    _require(
        isinstance(sampler, datasets.runtime.units.UnitSampler),
        "training loader must be driven by the resumable UnitSampler",
    )
    for label, loader in evaluation_loaders.items():
        _require(isinstance(loader.sampler, SequentialSampler), f"{label} loader must be stable")
    sampler_state = sampler.state_dict()
    sampler.begin_epoch(0)
    first_epoch_batches = tuple(tuple(batch) for batch in sampler)
    sampler_after = sampler.state_dict()
    flat_positions = [position for batch in first_epoch_batches for position in batch]
    _require(
        sorted(flat_positions) == list(range(len(train_loader.dataset))),
        "training sampler epoch is not an exact permutation",
    )
    sampler.load_state_dict(sampler_state)
    sampler.begin_epoch(0)
    replay_batches = tuple(tuple(batch) for batch in sampler)
    _require(
        replay_batches == first_epoch_batches,
        "training sampler state restore is not deterministic",
    )
    _require(sampler.state_dict() == sampler_after, "restored sampler continuation state changed")
    sampler.load_state_dict(sampler_state)
    sampler.begin_epoch(0)

    raw_moments = {
        "input": _ChannelMoments(),
        "output": _ChannelMoments(),
    }
    normalized_moments = {
        "input": _ChannelMoments(),
        "output": _ChannelMoments(),
    }
    passes = {
        "ID train": _validate_loader_pass(
            loader=train_loader,
            role="ID train",
            expected_ids=expected["ID train"],
            scaling=scaling,
            raw_moments=raw_moments,
            normalized_moments=normalized_moments,
        )
    }
    for label, loader in evaluation_loaders.items():
        passes[label] = _validate_loader_pass(loader=loader, role=label, expected_ids=expected[label], scaling=scaling)

    channel_records: list[ChannelStatisticsRecord] = []
    for tensor_role, names in (("Input", task.input_names), ("Output", task.output_names)):
        key = tensor_role.lower()
        raw_mean, raw_std = _finalize_channel_moments(raw_moments[key])
        normalized_mean, normalized_std = _finalize_channel_moments(normalized_moments[key])
        fitted_mean, fitted_std, fitted = _fitted_channel_statistics(scaling, names)
        _require(
            torch.allclose(
                raw_mean[fitted],
                fitted_mean[fitted],
                rtol=_FULL_VALIDATION_RTOL,
                atol=_FULL_VALIDATION_ATOL,
            ),
            f"{tensor_role} train-only fitted means disagree with independent pass",
        )
        _require(
            torch.allclose(
                raw_std[fitted],
                fitted_std[fitted],
                rtol=_FULL_VALIDATION_RTOL,
                atol=_FULL_VALIDATION_ATOL,
            ),
            f"{tensor_role} train-only fitted scales disagree with independent pass",
        )
        floor = datasets.preprocessing.normalization.SCALE_FLOOR
        # The normalized target follows the statistics that encode a channel: steady outputs the dimensionless state std.
        encoding_std = scaling.group("state").std.double() if key == "output" else fitted_std
        # A channel below the floor keeps its target std / floor < 1, so channels on both sides of the floor are compared.
        expected_scale = encoding_std / encoding_std.clamp_min(floor)
        _require(
            bool(torch.all(torch.abs(normalized_mean[fitted]) <= _NORMALIZED_MEAN_ATOL).item()),
            f"{tensor_role} normalized means are not near zero",
        )
        _require(
            torch.allclose(
                normalized_std[fitted],
                expected_scale[fitted],
                rtol=_NORMALIZED_SCALE_RTOL,
                atol=_NORMALIZED_SCALE_ATOL,
            ),
            f"{tensor_role} normalized scales are not near the maintained target",
        )
        for index, name in enumerate(names):
            finite = raw_moments[key].finite and normalized_moments[key].finite
            channel_records.append(
                ChannelStatisticsRecord(
                    tensor_role=tensor_role,
                    channel=name,
                    fitted_mean=float(fitted_mean[index]),
                    fitted_scale=float(fitted_std[index]),
                    normalized_mean=float(normalized_mean[index]),
                    normalized_scale=float(normalized_std[index]),
                    finite=finite,
                    result="PASS",
                )
            )

    normalization = datasets.preprocessing.normalization
    persisted_state = scaling.state_dict()
    _require(
        normalization.TrainScaling.from_state_dict(persisted_state).digest == scaling.digest,
        "persisted scaling reconstruction changed state",
    )
    for group in persisted_state["groups"].values():
        group["std"] = [0.0] * len(group["std"])
    zero_variance = normalization.TrainScaling.from_state_dict(persisted_state)
    # A unit reference leaves the encoded means in their own units, so a zero-variance channel encodes its mean to zero.
    unit_reference = torch.ones(1, dtype=zero_variance.dtype)
    zero_input = zero_variance.encode_inputs(zero_variance.input_affine()[0].view(1, -1, 1, 1), reference=unit_reference)
    zero_output = zero_variance.encode_state(zero_variance.group("state").mean.view(1, -1, 1, 1), reference=unit_reference)
    _require(
        bool(torch.isfinite(zero_input).all().item()) and torch.equal(zero_input, torch.zeros_like(zero_input)),
        "input zero-variance scale floor failed",
    )
    _require(
        bool(torch.isfinite(zero_output).all().item()) and torch.equal(zero_output, torch.zeros_like(zero_output)),
        "output zero-variance scale floor failed",
    )

    coverage = tuple(
        LoaderCoverageRecord(
            loader=label,
            sampler=type(loader.batch_sampler if label == "ID train" else loader.sampler).__name__,
            batches=len(loader),
            batch_size=loader.batch_size,
            final_batch=passes[label].final_batch_size,
            drop_last=loader.drop_last,
            inverse_checked=passes[label].inverse_checked,
            finite=True,
            result="PASS",
        )
        for label, loader in {"ID train": train_loader, **evaluation_loaders}.items()
    )
    dataset_membership = (
        _membership_record(
            role="ID full",
            contract=id_contract,
            expected_ids=id_contract.identity.sample_ids,
            observed_ids=id_contract.identity.sample_ids,
            shape=f"x{id_contract.input_shape}, y{id_contract.output_shape}",
            policy="complete identity and metadata artifact binding",
        ),
        *(
            _membership_record(
                role=label,
                contract=_membership_contract(label, id_contract=id_contract, ood_contracts=ood_contracts),
                expected_ids=expected[label],
                observed_ids=passes[label].observed_ids,
                shape="BCHW",
                policy="UnitSampler permutation" if label == "ID train" else "SequentialSampler",
            )
            for label in passes
        ),
    )

    first_split = split_contract
    del built, train_loader, evaluation_loaders, scaling, id_contract, ood_contracts
    gc.collect()
    repeated = config_loader.create_dataloaders_from_config(
        effective,
        seed_plan=seed_plan,
    )
    _require(
        datasets.preprocessing.splits.admit_split(repeated["split_indices"]) == first_split,
        "repeated production split changed membership",
    )
    del repeated
    gc.collect()

    elapsed = time.perf_counter() - started
    peak_after_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
    overall = (
        ValidationCheck(
            "Production APIs",
            "dataset admission, split, normalizer, loaders, preprocessing",
            "PASS",
        ),
        ValidationCheck(
            "Metadata binding",
            "complete artifact hashes and metadata snapshots admitted",
            "PASS",
        ),
        ValidationCheck(
            "Train-only normalizer",
            "independent float64 complete-pass agreement",
            "PASS",
        ),
        ValidationCheck(
            "Persisted state",
            "split schema and scaling reconstruction admitted",
            "PASS",
        ),
        ValidationCheck(
            "Zero-variance policy",
            "Train scale floor returns finite zeros",
            "PASS",
        ),
        ValidationCheck(
            "Sampler restore",
            "captured, advanced, restored, and replayed",
            "PASS",
        ),
        ValidationCheck(
            "Read-only boundary",
            "no run, W&B, checkpoint, or artifact operation",
            "PASS",
        ),
        ValidationCheck("Elapsed seconds", f"{elapsed:.1f}", "INFO"),
        ValidationCheck(
            "Observed process peak GiB",
            f"{peak_after_gib:.2f} (baseline {peak_before_gib:.2f})",
            "INFO",
        ),
    )
    return FullDataValidationResult(
        overall=overall,
        dataset_membership=dataset_membership,
        channels=tuple(channel_records),
        coverage=coverage,
        elapsed_seconds=elapsed,
        peak_gib=peak_after_gib,
    )
