"""Prepare typed, read-only state and tables for the training control notebook.

Responsibilities:
  - Resolve steady and transient protocol children, TaskSpec, metadata, and pure paths
  - List the built Datasets of a configuration's task and apply explicit training-Dataset and Stage-B-seconds selections
  - Prepare deterministic table specifications from already resolved values
  - Inspect saved configuration and summary identities with truthful artifact paths
  - Represent optional completed-run inspection and validation displays

Design principles:
  - Notebook preparation is atomic so failed re-execution cannot reuse stale state
  - Scientific admission delegates to config, task, metadata, run, and validation owners
  - Table specifications remain ordered, immutable, and independent of pandas

This module does NOT:
  - Load final tensor datasets, fit normalizers, or construct dataloaders
  - Allocate, resume, train, infer, initialize W&B, or generate artifacts
  - Reimplement scientific configuration, path, identity, or validation semantics
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src import common, datasets

from .config import experiments_config_loader as config_loader
from .config import experiments_config_naming as config_naming
from .config import experiments_config_transient_plan as transient_plan

if TYPE_CHECKING:
    from src.domain.tasks.domain_task_spec import TaskSpec
    from src.experiments.validation.experiments_validation_data_pipeline import FullDataValidationResult


@dataclass(frozen=True, slots=True)
class DatasetPreview:
    """Describe one task-owned dataset role from compact metadata only."""

    role: str
    dataset_id: str
    path: Path
    exists: bool
    sample_count: int | None
    dataset_digest: str | None
    metadata_validated: bool


@dataclass(frozen=True, slots=True)
class NotebookContext:
    """Store the atomic lightweight state used by notebook display cells."""

    config_path: Path
    official_config: dict[str, Any]
    task: TaskSpec
    objective: dict[str, Any]
    official_config_digest: str
    output_root: Path
    run_dir: Path
    dataset_previews: tuple[DatasetPreview, ...]
    protocol: str = "direct"
    stage_configs: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class RunInspection:
    """Store validated summary fields and expected-path existence records."""

    run_dir: Path
    summary_rows: tuple[tuple[str, object], ...]
    existence_rows: tuple[tuple[str, bool], ...]


@dataclass(frozen=True, slots=True)
class NotebookTable:
    """Store one ordered notebook table without a pandas dependency."""

    title: str | None
    columns: tuple[str, ...]
    rows: tuple[tuple[object, ...], ...]


@dataclass(frozen=True, slots=True)
class ValidationPresentation:
    """Store complete full-validation tables and their concise conclusion."""

    tables: tuple[NotebookTable, ...]
    conclusion: str


def _dataset_preview(
    *,
    role: str,
    dataset_id: str,
    task: TaskSpec,
    dataset_root: Path,
    metadata_root: Path,
) -> DatasetPreview:
    """Return one validated metadata preview or an explicit absent marker."""
    dataset_path = common.paths.resolve_dataset_path(task.id, dataset_id, dataset_root=dataset_root)
    metadata_directory = common.paths.resolve_dataset_metadata_dir(
        task.id,
        dataset_id,
        metadata_root=metadata_root,
    )
    if not metadata_directory.is_dir():
        return DatasetPreview(
            role=role,
            dataset_id=dataset_id,
            path=dataset_path,
            exists=dataset_path.is_file() and not dataset_path.is_symlink(),
            sample_count=None,
            dataset_digest=None,
            metadata_validated=False,
        )

    manifest = datasets.packages.manifest.load_package_manifest_evidence(
        task.id,
        dataset_id,
        storage_root=dataset_root.parent,
    )
    view = datasets.contracts.views.get_view(task.id)
    if manifest["registered_task_id"] != view.registered_task_id or manifest["dataset_view"] != view.id:
        msg = "Dataset manifest does not match the selected task."
        raise ValueError(msg)
    return DatasetPreview(
        role=role,
        dataset_id=dataset_id,
        path=dataset_path,
        exists=True,
        sample_count=manifest["sample_count"],
        dataset_digest=manifest["dataset_digest"],
        metadata_validated=True,
    )


def list_built_datasets(config_path: Path | str) -> tuple[str, ...]:
    """Return the sorted IDs of the built Datasets of the task a configuration belongs to."""
    task_id = config_naming.config_source_from_path(config_path).task
    dataset_root = common.paths.get_datasets_root()
    manifests = (dataset_root / task_id).glob(f"*/{common.paths.DATASET_MANIFEST_FILENAME}")
    built = tuple(sorted(manifest.parent.name for manifest in manifests))
    if not built:
        msg = f"No built {task_id} Dataset exists below {dataset_root}; build one before selecting it."
        raise FileNotFoundError(msg)
    return built


def prepare_notebook_context(
    config_path: Path | str,
    *,
    stage: str | None = None,
    train_dataset: str | None = None,
    train_seconds: float | None = None,
) -> NotebookContext:
    """Resolve configuration, TaskSpec, metadata summaries, and paths atomically.

    ``train_dataset`` replaces ``data.train_dataset`` of every stage and ``train_seconds``
    sets Stage B's measured target of a matched plan, exactly as the ``--train-dataset``
    and ``--train-seconds`` training options do; ``None`` keeps the configured value.
    """
    selected_path = Path(config_path)
    raw = config_loader.apply_training_overrides(
        config_loader.load_authored_config(selected_path),
        train_dataset=train_dataset,
        train_seconds=train_seconds,
    )
    config_source = config_naming.config_source_from_path(selected_path)
    if transient_plan.is_matched_a1_b_config(raw):
        plan = transient_plan.resolve_matched_a1_b_plan(raw, source=config_source)
        protocol = "matched_a1_b"
        stage_configs = tuple(dict(config) for config in plan.stages)
    else:
        protocol = "direct"
        stage_configs = (config_loader.resolve_config(raw, source=config_source),)
    if stage is None:
        official_config = stage_configs[0]
    else:
        matches = [config for config in stage_configs if config["training"].get("comparison_arm") == stage]
        if len(matches) != 1:
            msg = f"Stage {stage!r} is unavailable in the selected {protocol} configuration."
            raise ValueError(msg)
        official_config = matches[0]
    task = config_loader.validate_resolved_task_contract(official_config)
    objective = config_loader.get_resolved_objective(official_config)

    dataset_root = Path(official_config["paths"]["dataset_root"])
    metadata_root = Path(official_config["paths"]["dataset_root"])
    data = official_config["data"]
    roles = [
        ("ID train / validation / test source", data["train_dataset"]),
        *(("Dataset-declared OOD", dataset_id) for dataset_id in data["additional_evaluation_datasets"]),
    ]
    previews = tuple(
        _dataset_preview(
            role=role,
            dataset_id=dataset_id,
            task=task,
            dataset_root=dataset_root,
            metadata_root=metadata_root,
        )
        for role, dataset_id in roles
    )

    output_root = Path(official_config["paths"]["output_root"])
    run_dir = common.paths.resolve_config_run_dir(official_config)
    return NotebookContext(
        config_path=selected_path,
        official_config=official_config,
        task=task,
        objective=objective,
        official_config_digest=common.serialization.canonical_json_sha256(official_config),
        output_root=output_root,
        run_dir=run_dir,
        dataset_previews=previews,
        protocol=protocol,
        stage_configs=stage_configs,
    )


def prepare_run_inspection(run_dir: Path | str) -> RunInspection:
    """Inspect saved configuration and summary without opening tensor artifacts."""
    from . import experiments_run as run_service  # noqa: PLC0415 -- optional run inspection stays lazy

    selected_run = Path(run_dir).expanduser().resolve()
    if not selected_run.is_dir():
        msg = f"Run directory not found: {selected_run}"
        raise FileNotFoundError(msg)
    summary = run_service.read_run_summary(selected_run)
    config = config_loader.validate_resolved_config(
        config_loader.load_yaml(common.paths.resolve_run_config_path(selected_run)),
    )
    if summary.get("task") != config["task"] or summary.get("run_name") != config["run"]["name"]:
        msg = "Saved run summary and configuration identity disagree."
        raise ValueError(msg)
    objective = config_loader.get_resolved_objective(config)
    data = config["data"]
    summary_rows: tuple[tuple[str, object], ...] = (
        ("Run directory", selected_run),
        ("Saved task", config["task"]),
        ("Saved run name", config["run"]["name"]),
        ("Stage / comparison arm", config["training"].get("comparison_arm", "direct")),
        ("Status", summary.get("status", "not available")),
        ("Objective", objective["id"]),
        ("Completed epoch", summary.get("completed_epoch", "not available")),
        ("Best epoch", summary.get("best_epoch", "not available")),
        ("Best metric", summary.get("best_metric", "not available")),
    )
    expected_paths = [(name, selected_run / name) for name in common.paths.CURRENT_RUN_REQUIRED_FILES]
    expected_paths.extend(
        [
            ("wandb/", selected_run / "wandb"),
            ("analysis/id/", common.paths.resolve_id_analysis_dir(selected_run)),
        ]
    )
    if config["task"] == "transient_drying":
        expected_paths.append(("history.json", selected_run / "history.json"))
    additional = data["additional_evaluation_datasets"]
    if additional:
        identity = datasets.contracts.identity.combined_dataset_id(additional)
        expected_paths.append(
            (
                f"analysis/additional_evaluation/{identity}/",
                common.paths.resolve_additional_evaluation_analysis_dir(selected_run, identity),
            )
        )
    return RunInspection(
        run_dir=selected_run,
        summary_rows=summary_rows,
        existence_rows=tuple((label, path.exists()) for label, path in expected_paths),
    )


def prepare_protocol_table(context: NotebookContext) -> NotebookTable:
    """Preview authored stage dependencies and budgets without allocating runs."""
    rows = []
    for config in context.stage_configs or (context.official_config,):
        training = config["training"]
        matched = training.get("matched_a1_b", {})
        teacher = training.get("teacher_handoff")
        rows.append(
            (
                training.get("comparison_arm", "direct"),
                config["run"]["name"],
                training["epochs"],
                teacher["source_run_name"] if teacher else "none",
                matched.get("train_seconds"),
                matched.get("receipt_source_run_name"),
            )
        )
    return _table(
        title=f"Training protocol: {context.protocol}",
        columns=("Arm", "Run name", "Epoch cap", "Teacher source", "B train seconds", "Measured budget source"),
        rows=rows,
    )


def _short_digest(value: str | None, *, length: int = 16) -> str:
    """Return a readable prefix while preserving the full value in its owner."""
    return "not available" if value is None else value[:length]


def _readable_value(value: object) -> str:
    """Format one scalar or short sequence for a compact notebook cell."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return "none"
    if isinstance(value, float):
        return f"{value:.8g}"
    if isinstance(value, Mapping):
        return "; ".join(f"{key}={_readable_value(item)}" for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return ", ".join(_readable_value(item) for item in value) if value else "none"
    return str(value)


def display_path(
    path: Path | str,
    *,
    project_root: Path,
    storage_root: Path,
) -> str:
    """Render a configured path without exposing its host-mounted prefix."""
    candidate = Path(path).expanduser().resolve()
    roots = (
        (storage_root.resolve(), "$STORAGE_ROOT"),
        (project_root.resolve(), "$PROJECT_ROOT"),
    )
    for root, label in roots:
        if candidate == root:
            return label
        if candidate.is_relative_to(root):
            return f"{label}/{candidate.relative_to(root).as_posix()}"
    return f"<explicit external path>/{candidate.name}"


def _table(
    *,
    columns: tuple[str, ...],
    rows: Iterable[Iterable[object]],
    title: str | None = None,
) -> NotebookTable:
    """Freeze one deterministic notebook table specification."""
    return NotebookTable(
        title=title,
        columns=columns,
        rows=tuple(tuple(row) for row in rows),
    )


def _settings_table(
    title: str,
    rows: Iterable[tuple[str, object, str]],
) -> NotebookTable:
    """Format one ordered settings table without repeated section labels."""
    return _table(
        title=title,
        columns=("Setting", "Resolved value", "Meaning"),
        rows=((setting, _readable_value(value), meaning) for setting, value, meaning in rows),
    )


def _prepare_transient_configuration_tables(context: NotebookContext) -> tuple[NotebookTable, ...]:
    """Render resolved transient settings without borrowing steady physics semantics."""
    config = context.official_config
    data = config["data"]
    training = config["training"]
    model = config["model"]
    return (
        _settings_table(
            "Task and data",
            (
                ("Task ID", context.task.id, "Registered scientific task"),
                ("ID dataset", data["train_dataset"], "Disjoint train, validation, and untouched ID-test cases"),
                ("Case partition", "Dataset manifest", "Fixed case membership independent of training seed"),
                ("Additional Evaluation datasets", data["additional_evaluation_datasets"], "Optional explicitly assigned scientific roles"),
                ("Input profile", config["input_profile"], "Resolved transient input semantics"),
                ("Output fields", context.task.output_names, "Ordered physical state channels"),
                ("Backend", data["transient_backend_preference"], "Requested maintained data backend"),
                ("Backend required", data["transient_backend_required"], "Fail-closed backend requirement"),
                ("Spatial stride", data["spatial_stride"], "Declared model spatial support"),
            ),
        ),
        _settings_table(
            "Model",
            (
                ("Variant", config_loader.resolved_model_variant(config), "Production semantic model identity"),
                ("Kind", model["kind"], "Maintained neural-operator family"),
                *((key, value, "Resolved architecture parameter") for key, value in model["params"].items()),
            ),
        ),
        _settings_table(
            "Temporal protocol and training",
            (
                ("Protocol", context.protocol, "Direct or matched A0/B/A1 request"),
                ("Arm", training["comparison_arm"], "Selected child configuration"),
                ("Sampling", config["temporal"]["sampling"], "Authoritative transition or rollout sampling"),
                ("Time conditioning", config["temporal"]["temporal_conditioning"], "Configured temporal features"),
                ("Epoch cap", training["epochs"], "Operational cap; does not establish measured budget completion"),
                ("Curriculum", training["curriculum"], "Configured rollout lengths and milestones"),
                ("Teacher handoff", training["teacher_handoff"], "Common Stage-A checkpoint dependency"),
                ("Matched protocol", training.get("matched_a1_b"), "A1 consumes B measured train-time receipt"),
                ("ID evaluation interval", training["evaluation_interval"], "Validation cadence for model selection"),
                ("Additional evaluation interval", training["additional_evaluation_interval"], "Diagnostic cadence"),
            ),
        ),
        _settings_table(
            "Runtime, optimization, and loss",
            (
                ("Device policy", config["run"]["device"], "Resolved policy; no device probe"),
                ("Seed", config["run"]["seed"], "Saved deterministic seed"),
                ("Deterministic", config["run"]["deterministic"], "Deterministic runtime policy"),
                ("Batch size", data["batch_size"], "Training batch size"),
                ("Workers", data["num_workers"], "Dataloader worker count"),
                ("Mixed precision", training["mixed_precision"], "Configured precision policy"),
                ("Gradient accumulation", training["gradient_accumulation_steps"], "Batches per optimizer step"),
                ("Optimizer", config["optimizer"], "Production optimization settings"),
                ("Scheduler", config["scheduler"], "Validation-objective learning-rate policy"),
                ("Scaling", config["scaling"], "Training-only fitted scaling contract"),
                ("Data loss", config["loss"]["data"], "Saved increment and auxiliary state objective"),
                ("Physics loss", config["loss"]["physics"], "Resolved transient physics selection"),
            ),
        ),
        _settings_table(
            "Evaluation and tracking",
            (
                ("Primary objective", context.objective["id"], "Validation model-selection metric"),
                ("Direction", context.objective["direction"], "Production objective direction"),
                ("Metrics", [metric["id"] for metric in config["evaluation"]["metrics"]], "Resolved normalized and physical metrics"),
                ("W&B", config["tracking"]["wandb"], "Configured observer; inspection never initializes it"),
            ),
        ),
    )


def prepare_configuration_tables(context: NotebookContext) -> tuple[NotebookTable, ...]:
    """Prepare ordered presentation tables from one resolved configuration."""
    config = context.official_config
    if context.task.id == "transient_drying":
        return _prepare_transient_configuration_tables(context)
    task = context.task
    model = config["model"]
    model_params = model["params"]
    training = config["training"]
    data = config["data"]
    optimizer = config["optimizer"]
    scheduler = config["scheduler"]
    loss_data = config["loss"]["data"]
    physics = config["loss"]["physics"]
    wandb = config["tracking"]["wandb"]
    metrics = config["evaluation"]["metrics"]
    variant = config_loader.resolved_model_variant(config)

    model_rows: list[tuple[str, object, str]] = [
        ("Variant", variant, "Semantic model variant used in names and W&B tags"),
        ("Kind", model["kind"], "Neural-operator family"),
        ("Input channels", model_params["in_channels"], "Task-owned ordered input fields"),
        ("Output channels", model_params["out_channels"], "Task-owned ordered output fields"),
    ]
    if model["kind"] == "fno":
        model_rows.extend(
            [
                ("Fourier modes", model_params["n_modes"], "Retained modes on the two operator axes"),
                ("Hidden channels", model_params["hidden_channels"], "Latent channel width"),
                ("Layers", model_params["n_layers"], "Fourier operator layers"),
                ("Implementation", model_params["implementation"], "FNO contraction implementation"),
                ("FNO skip", model_params["fno_skip"], "Operator-layer skip mapping"),
                ("Channel MLP skip", model_params["channel_mlp_skip"], "Channel MLP skip mapping"),
                ("Lifting ratio", model_params["lifting_channel_ratio"], "Input lifting width ratio"),
                ("Projection ratio", model_params["projection_channel_ratio"], "Output projection width ratio"),
            ]
        )
    else:
        model_rows.extend(
            [
                (
                    "Fourier modes",
                    [model_params["modes_x"], model_params["modes_y"]],
                    "UNO modes on both operator axes",
                ),
                ("Hidden channels", model_params["hidden_channels"], "Latent channel width"),
                ("Layers", model_params["n_layers"], "UNO operator layers"),
                ("Mode ratio", model_params["mode_ratio"], "Mode scaling across UNO layers"),
                ("Channel MLP skip", model_params["channel_mlp_skip"], "Channel MLP skip mapping"),
            ]
        )

    physics_rows: list[tuple[str, object, str]] = [
        ("Physics enabled", physics["enabled"], "Whether PINO contributions affect training"),
        (
            "Derivative mode",
            f"{physics['derivatives']['kind']} with {physics['derivatives']['extension']} extension",
            "Spatial derivative policy used by diagnostics and active PINO loss",
        ),
        (
            "Continuity",
            "cross-section flux balance",
            "Task-owned mass balance: integral of v across x is constant over y",
        ),
        ("Interior crop", physics["interior_crop"], "Boundary cells excluded from residual diagnostics"),
    ]
    if physics["enabled"]:
        for term in ("momentum", "continuity", "boundary"):
            weight = physics[f"{term}_weight"]
            physics_rows.extend(
                [
                    (f"{term.capitalize()} weight", weight["target"], f"Active target weight of the dimensionless {term} term"),
                    (
                        f"{term.capitalize()} warmup",
                        f"{weight['warmup']['kind']} over {weight['warmup']['epochs']} epochs",
                        f"Active {term}-weight schedule",
                    ),
                ]
            )
    else:
        physics_rows.extend(
            [
                (
                    "Residual training contribution",
                    "inactive",
                    "Configured diagnostics remain visible but do not affect optimization",
                ),
                (
                    "Boundary training contribution",
                    "inactive",
                    "Configured diagnostics remain visible but do not affect optimization",
                ),
            ]
        )

    return (
        _settings_table(
            "Task and data",
            (
                ("Task ID", task.id, "Registered scientific task"),
                ("Task schema", task.schema_version, "Task contract schema version"),
                ("Task digest", _short_digest(task.contract_digest), "Short display of the full TaskSpec digest"),
                ("ID dataset", data["train_dataset"], "Fixed Train/validation/ID-test partition"),
                ("Additional evaluation", data["additional_evaluation_datasets"], "Dataset-declared OOD holdouts"),
                ("Input fields", task.input_names, "Ordered model input channels"),
                ("Output fields", task.output_names, "Ordered prediction channels"),
                ("Tensor layout", task.tensor_layout, "Saved tensor dimension order"),
                ("Normalizer fit role", task.preprocessing.fit_split, "Only ID training membership is fitted"),
                ("Role ownership", "TaskSpec", "Dataset IDs are not supplied manually"),
            ),
        ),
        _settings_table("Model", model_rows),
        _settings_table(
            "Runtime and training",
            (
                ("Device policy", config["run"]["device"], "YAML policy used when the CLI has no override"),
                ("Batch size", data["batch_size"], "Samples per optimization batch"),
                ("Workers", data["num_workers"], "Dataloader worker processes"),
                ("Pin memory", data["pin_memory"], "Host-memory transfer policy"),
                ("Persistent workers", data["persistent_workers"], "Worker lifetime policy"),
                ("Epochs", training["epochs"], "Official target duration"),
                (
                    "ID evaluation interval",
                    training["evaluation_interval"],
                    "Completed-epoch cadence for objective, scheduler, and checkpoint selection",
                ),
                (
                    "Additional evaluation interval",
                    training["additional_evaluation_interval"],
                    "Completed-epoch cadence for additional evaluation",
                ),
                ("Mixed precision", training["mixed_precision"], "CUDA mixed-precision policy"),
                ("Seed", config["run"]["seed"], "Root deterministic seed"),
                ("Deterministic", config["run"]["deterministic"], "Deterministic runtime policy"),
            ),
        ),
        _settings_table(
            "Optimization, loss, and physics",
            (
                ("Optimizer", optimizer["kind"], "Optimization algorithm"),
                ("Learning rate", optimizer["lr"], "Initial learning rate"),
                ("Weight decay", optimizer["weight_decay"], "AdamW regularization"),
                ("Betas", optimizer["betas"], "First and second moment factors"),
                ("Scheduler", scheduler["kind"], "ID-objective scheduler"),
                ("Scheduler factor", scheduler["factor"], "Learning-rate reduction factor"),
                ("Scheduler patience", scheduler["patience"], "ID evaluations before reduction"),
                ("Minimum learning rate", scheduler["min_lr"], "Scheduler floor"),
                (
                    "Data loss",
                    f"{loss_data['kind']} in {loss_data['space']} space",
                    "Supervised training objective",
                ),
                ("Data-loss weight", loss_data["weight"], "Supervised contribution weight"),
                *physics_rows,
            ),
        ),
        _settings_table(
            "Evaluation and W&B",
            (
                (
                    "Primary objective",
                    context.objective["id"],
                    "ID metric used for scheduling and checkpoint selection",
                ),
                ("Objective direction", context.objective["direction"], "Selection direction"),
                (
                    "Normalized metrics",
                    [metric["id"] for metric in metrics if metric["space"] == "normalized"],
                    "Dimensionless evaluation group",
                ),
                (
                    "Physical metrics",
                    [metric["id"] for metric in metrics if metric["space"] == "physical"],
                    "Per-field physical-unit evaluation group",
                ),
                ("W&B mode", wandb["mode"], "Online, fail-closed observer policy"),
                ("W&B workflow", wandb["workflow"], "Run organization workflow"),
                ("W&B project", wandb["project"], "Repository-owned W&B project"),
                ("W&B entity", wandb["entity"], "Configured W&B entity"),
                ("W&B tags", wandb["tags"], "Model variant tags"),
                (
                    "Automatic categories",
                    "Overview, Accuracy, Physics, Diagnostics",
                    "Personal-workspace metric prefixes",
                ),
                (
                    "Monitor policy",
                    f"every {wandb['monitor']['interval']} epochs, at most {wandb['monitor']['max_cases']} cases",
                    "Terminal-inclusive completed-epoch cadence. No epoch-zero history",
                ),
                (
                    "Evaluation artifact upload",
                    wandb["upload"]["evaluation_artifacts"],
                    "Curated artifact upload remains opt-in",
                ),
            ),
        ),
    )


def prepare_dataset_table(
    context: NotebookContext,
    *,
    project_root: Path,
    storage_root: Path,
) -> NotebookTable:
    """Prepare compact metadata-only dataset role previews."""
    rows = (
        (
            preview.role,
            preview.dataset_id,
            display_path(
                preview.path,
                project_root=project_root,
                storage_root=storage_root,
            ),
            _readable_value(preview.exists),
            _readable_value(preview.sample_count) if preview.metadata_validated else "not available",
            _short_digest(preview.dataset_digest),
        )
        for preview in context.dataset_previews
    )
    return _table(
        columns=("Role", "Dataset ID", "Resolved path", "Exists", "Sample count", "Dataset digest"),
        rows=rows,
    )


def prepare_run_preview_table(
    context: NotebookContext,
    *,
    project_root: Path,
    storage_root: Path,
) -> NotebookTable:
    """Prepare deterministic run identity and destination evidence."""
    config = context.official_config
    wandb = config["tracking"]["wandb"]
    variant = config_loader.resolved_model_variant(config)
    rows = (
        (
            "Experiment configuration",
            display_path(
                context.config_path,
                project_root=project_root,
                storage_root=storage_root,
            ),
            "Selected YAML request",
        ),
        ("Model variant", variant, "Semantic model identity"),
        ("Deterministic run name", config["run"]["name"], "Generated by the production config loader"),
        ("Device policy", config["run"]["device"], "Used when the CLI has no override"),
        ("W&B mode", wandb["mode"], "Production observer mode"),
        ("Target epochs", config["training"]["epochs"], "Official training duration"),
        ("ID dataset", config["data"]["train_dataset"], "Train and evaluation source"),
        (
            "Additional evaluation",
            config["data"]["additional_evaluation_datasets"],
            "Optional saved diagnostic datasets",
        ),
        (
            "Resolved config digest",
            _short_digest(context.official_config_digest),
            "Short display of the full in-memory digest",
        ),
        (
            "Output root",
            display_path(
                context.output_root,
                project_root=project_root,
                storage_root=storage_root,
            ),
            "Task-owned run root",
        ),
        (
            "Expected run directory",
            display_path(
                context.run_dir,
                project_root=project_root,
                storage_root=storage_root,
            ),
            "Pure path preview with no allocation",
        ),
        (
            "Run directory exists",
            _readable_value(context.run_dir.exists()),
            "An existing run continues only when its saved science matches",
        ),
    )
    return _table(columns=("Setting", "Value", "Meaning"), rows=rows)


_RUN_OUTPUT_ROWS = (
    ("config.yaml", "Immutable fully resolved run configuration", "yes", "Validation, resume, inference, and artifacts"),
    (
        "summary.json",
        "Lifecycle, objective, provenance, and selected-result summary",
        "yes",
        "CLI and notebook inspection, W&B summary",
    ),
    (
        "split_indices.pt",
        "Exact train, validation, ID-test, and additional evaluation membership",
        "yes",
        "Resume, inference, and artifacts",
    ),
    (
        "normalizer.pt",
        "Normalizer fitted only on ID training membership",
        "yes",
        "Resume, inference, and artifacts",
    ),
    (
        "best_checkpoint.pt",
        "Best finite ID-objective model state",
        "completed run",
        "Inference, evaluation, and artifacts",
    ),
    ("last_checkpoint.pt", "Latest exact continuation state", "yes", "Resume"),
    ("wandb/", "Local SDK state for online or offline tracking", "no", "W&B SDK and local diagnostics"),
    ("analysis/id/", "ID evaluation artifacts", "no", "Single-model and comparison evaluation"),
    (
        "analysis/additional_evaluation/<combined dataset ID>/",
        "Optional additional evaluation artifacts",
        "no",
        "Single-model and comparison evaluation",
    ),
    (
        "artifact_provenance.json and case files",
        "Analysis completion marker, Parquet table, and case NPZ payloads",
        "no",
        "Evaluation DataFrames and plots",
    ),
)


def prepare_run_output_inventory_table(*, task_id: str = "steady_flow") -> NotebookTable:
    """Prepare the maintained saved-run file inventory."""
    rows: tuple[tuple[str, str, str, str], ...] = _RUN_OUTPUT_ROWS
    if task_id == "transient_drying":
        rows = (
            *rows,
            ("history.json", "Durable completed-epoch transient history", "yes", "Training and Evaluation"),
        )
    return _table(
        columns=("File or directory", "Purpose", "Required", "Consumer"),
        rows=rows,
    )


def prepare_run_inspection_tables(
    inspection: RunInspection,
    *,
    project_root: Path,
    storage_root: Path,
) -> tuple[NotebookTable, NotebookTable]:
    """Prepare lightweight summary and expected-path tables for one run."""
    summary_rows: list[tuple[str, str]] = []
    for field, value in inspection.summary_rows:
        if field == "Run directory":
            if not isinstance(value, (Path, str)):
                msg = f"Run directory summary value must be path-like, got {type(value).__name__}."
                raise TypeError(msg)
            rendered = display_path(
                value,
                project_root=project_root,
                storage_root=storage_root,
            )
        else:
            rendered = _readable_value(value)
        summary_rows.append((field, rendered))

    existence_rows = ((label, _readable_value(exists)) for label, exists in inspection.existence_rows)
    return (
        _table(columns=("Summary field", "Value"), rows=summary_rows),
        _table(columns=("File or directory", "Exists"), rows=existence_rows),
    )


def prepare_validation_presentation(
    result: FullDataValidationResult,
) -> ValidationPresentation:
    """Prepare complete display tables without changing validation evidence."""
    tables = (
        _table(
            title="Overall status",
            columns=("Check", "Evidence", "Result"),
            rows=((record.check, record.evidence, record.result) for record in result.overall),
        ),
        _table(
            title="Dataset membership",
            columns=(
                "Role",
                "Dataset ID",
                "Full samples",
                "Expected",
                "Observed",
                "Duplicates",
                "Missing",
                "Shape",
                "Dtype",
                "Fingerprint",
                "Data contract digest",
                "Finite",
                "Policy",
                "Result",
            ),
            rows=(
                (
                    record.role,
                    record.dataset_id,
                    record.full_samples,
                    record.expected,
                    record.observed,
                    record.duplicates,
                    record.missing,
                    record.shape,
                    record.dtype,
                    record.fingerprint,
                    record.data_contract_digest,
                    record.finite,
                    record.policy,
                    record.result,
                )
                for record in result.dataset_membership
            ),
        ),
        _table(
            title="Channel normalization",
            columns=(
                "Tensor role",
                "Channel",
                "Fitted mean",
                "Fitted scale",
                "Normalized mean",
                "Normalized scale",
                "Finite",
                "Result",
            ),
            rows=(
                (
                    record.tensor_role,
                    record.channel,
                    record.fitted_mean,
                    record.fitted_scale,
                    record.normalized_mean,
                    record.normalized_scale,
                    record.finite,
                    record.result,
                )
                for record in result.channels
            ),
        ),
        _table(
            title="Loader coverage",
            columns=(
                "Loader",
                "Sampler",
                "Batches",
                "Batch size",
                "Final batch",
                "Drop last",
                "Inverse",
                "Finite",
                "Result",
            ),
            rows=(
                (
                    record.loader,
                    record.sampler,
                    record.batches,
                    record.batch_size,
                    record.final_batch,
                    record.drop_last,
                    record.inverse_checked,
                    record.finite,
                    record.result,
                )
                for record in result.coverage
            ),
        ),
        _table(
            title="Execution footprint",
            columns=("Measure", "Value"),
            rows=(("Elapsed seconds", result.elapsed_seconds), ("Peak memory (GiB)", result.peak_gib)),
        ),
    )
    conclusion = (
        "**PASS:** complete mounted data and metadata, production split, "
        "train-only normalization, preprocessing, loader coverage, inverse "
        "transforms, and sampler restoration satisfy the maintained contracts."
    )
    return ValidationPresentation(tables=tables, conclusion=conclusion)
