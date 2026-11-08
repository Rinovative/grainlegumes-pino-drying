"""Render bounded line-oriented reporting for experiment lifecycles.

Responsibilities:
  - Report the run start block, one startup line, one line per epoch, and one end line with epochs run, stop reason and duration parts
  - Report one line per finished Optuna trial plus the study start and end
  - Redact secrets from messages and complete traceback chains

Design principles:
  - Scientific values are consumed from lifecycle payloads without recomputation
  - Reporting uses genuine completed epochs and emits no default batch progress
  - Credential-bearing values are redacted and paths are shown relative to the storage root

This module does NOT:
  - Train models, persist checkpoints, or initialize W&B
  - Decide experiment success or reconstruct missing scientific observations
"""

from __future__ import annotations

import math
import re
import sys
import time
import traceback
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src import common

from .config import experiments_config_loader as config_loader

_SECRET_PATTERN = re.compile(r"(?i)(api[_-]?key|password|secret|token)(\s*[:=]\s*)([^\s,;]+)")
_STARTUP_PHASE_EVENTS = frozenset(
    {
        "data_setup_start",
        "dataset_resolved",
        "scaler_cache",
        "scaler_fit_start",
        "scaler_fit_progress",
        "scaler_fit_complete",
        "dataloader_ready",
        "model_build_start",
        "model_cuda_ready",
        "training_start",
    }
)


def _redact_secrets(value: object) -> str:
    """Redact credential-shaped values while preserving surrounding text."""
    return _SECRET_PATTERN.sub(r"\1\2<redacted>", str(value))


def _clean(value: object) -> str:
    """Return one bounded, single-line, secret-redacted display value."""
    text = _redact_secrets(value).replace("\r", " ").replace("\n", " ").strip()
    return text[:600]


def display_path(value: object) -> str:
    """Return an absolute path relative to the storage root, else the project root; any other value unchanged."""
    text = str(value)
    path = Path(text)
    if not path.is_absolute():
        return text
    for root in (common.paths.get_storage_root(), common.paths.get_project_root()):
        if path.is_relative_to(root):
            return path.relative_to(root).as_posix()
    return text


def duration_summary(total_seconds: float, parts: Mapping[str, float]) -> str:
    """Return ``total <t> (<name> <d>, ..., other <rest>)``, where ``other`` is the part of the total no named part covers."""
    named = {**parts, "other": max(0.0, total_seconds - sum(parts.values()))}
    inner = ", ".join(f"{name} {common.console.format_duration(seconds)}" for name, seconds in named.items())
    return f"total {common.console.format_duration(total_seconds)} ({inner})"


def _num(value: object, *, digits: int = 3) -> str:
    """Format one scalar in scientific notation with ``digits + 1`` significant digits; ``-`` when absent."""
    if value is None:
        return "-"
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return f"{value:.{digits}e}"
    return str(value)


def _text(value: object) -> str:
    """Format one display value: four significant digits, relative paths, sorted mappings, bounded redacted text."""
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, Mapping):
        return " ".join(f"{key}={_text(item)}" for key, item in sorted(value.items()))
    return display_path(_clean(value))


def _line(text: str, *, to_stderr: bool = False) -> None:
    """Print and promptly flush one secret-redacted lifecycle line."""
    print(_redact_secrets(text), file=sys.stderr if to_stderr else sys.stdout, flush=True)


def _loss_composition(config: Mapping[str, Any]) -> str:
    """Describe the implemented active total-loss composition."""
    loss = config["loss"]
    data = loss["data"]
    terms = [f"{data['weight']}*{data['kind']}[{data['space']}]"]
    physics = loss["physics"]
    if bool(physics["enabled"]):
        for term, residual in (("momentum", "momentum"), ("continuity", "flux_balance"), ("boundary", "pressure_boundary")):
            weight = physics[f"{term}_weight"]
            terms.append(f"{term}_weight(epoch,target={weight['target']},warmup={weight['warmup']['epochs']})*{residual}")
    return "+".join(terms)


def _loss_components(metrics: Mapping[str, float]) -> str:
    """Return compact loss components, shown only when more than one remains besides the total."""
    total = metrics.get("train/loss_total")
    components = {
        key.removeprefix("physics/train/").removeprefix("train/").removeprefix("loss_").removeprefix("data_"): value
        for key, value in metrics.items()
        if key.startswith(("train/", "physics/train/")) and key != "train/loss_total"
    }
    if "data" in components and components["data"] == total:
        del components["data"]
    if len(components) < 2:  # noqa: PLR2004 -- a single remaining component repeats the total
        return ""
    return " [" + " ".join(f"{name} {_num(value, digits=2)}" for name, value in components.items()) + "]"


@dataclass(slots=True)
class ConsoleReporter:
    """Render the start block, one line per epoch, the end line, and failures of one run."""

    config: Mapping[str, Any]
    run_dir: Path
    resume: bool = False
    study_name: str | None = None
    trial_number: int | None = None
    _started: float = field(default_factory=time.perf_counter)
    _phases: dict[str, Mapping[str, Any]] = field(default_factory=dict, init=False)
    _training_seconds: float = field(default=0.0, init=False)
    _validation_seconds: float = field(default=0.0, init=False)

    @property
    def objective_id(self) -> str:
        """Return the explicitly resolved selection metric ID."""
        return str(self.config["evaluation"]["objective"]["id"])

    def startup_phase(self, event: str, fields: Mapping[str, Any] | None = None) -> None:
        """Collect one startup phase and print the single startup line when training starts."""
        if event not in _STARTUP_PHASE_EVENTS:
            message = f"Unsupported startup phase event: {event!r}."
            raise ValueError(message)
        self._phases.pop(event, None)
        self._phases[event] = {**(fields or {}), "elapsed_seconds": time.perf_counter() - self._started}
        if event == "training_start":
            _line(self._startup_summary())

    def _startup_summary(self) -> str:
        """Summarize the collected startup phases in one line."""
        data = self._phases.get("dataset_resolved", {})
        scaler = self._phases.get("scaler_cache", {})
        fit = self._phases.get("scaler_fit_complete")
        parts = [f"startup {common.console.format_duration(self._phases['training_start']['elapsed_seconds'])}"]
        if data:
            parts.append(f"data {data.get('cases')} Train cases {data.get('items')} items grid {data.get('grid')}")
        if scaler:
            parts.append(f"scaler {scaler.get('state')}" + (f" (fit {common.console.format_duration(fit['seconds'])})" if fit else ""))
        if "model_cuda_ready" in self._phases:
            parts.append(f"model {common.console.format_duration(self._phases['model_cuda_ready'].get('seconds', 0.0))}")
        parts.append(f"{self._phases['training_start'].get('batches_per_epoch')} batches/epoch")
        return "  ".join(parts)

    def startup(self, *, resolved_device: str) -> None:
        """Print the four-line run start block: identity, data, model and optimization, loss and selection."""
        run = self.config["run"]
        data = self.config["data"]
        training = self.config["training"]
        optimizer = self.config["optimizer"]
        objective = self.config["evaluation"]["objective"]
        scheduler = "no scheduler" if self.config.get("scheduler") is None else self.config["scheduler"]["kind"]
        origin = f" | study {self.study_name} trial {self.trial_number}" if self.study_name is not None else ""
        model = f"{self.config['model']['kind']} {config_loader.resolved_model_variant(self.config)}"
        _line(
            f"run {_text(run['name'])} | {self.config['task']} | {run['layout']['family']} | {resolved_device} | seed {run['seed']}"
            f" | {'resumed run' if self.resume else 'new run'}{origin}"
        )
        workers = config_loader.effective_num_workers(int(data["num_workers"]))
        _line(f"data {_text(data['train_dataset'])} | batch {data['batch_size']} | workers {workers}")
        _line(f"model {model} | {optimizer['kind']} lr {_num(optimizer['lr'], digits=2)} {scheduler} | {training['epochs']} epochs")
        _line(
            f"loss {_loss_composition(self.config)} | select {self.objective_id} ({objective['direction']})"
            f" | val every {training['evaluation_interval']}"
        )

    def epoch(self, epoch: int, metrics: dict[str, float]) -> None:
        """Print one line per completed epoch with the evaluation results that fell due."""
        target = int(self.config["training"]["epochs"])
        id_key = f"id/{self.objective_id}"
        parts = [f"epoch {epoch:>{len(str(target))}}/{target}", f"loss {_num(metrics.get('train/loss_total'))}{_loss_components(metrics)}"]
        if id_key in metrics:
            parts.append(f"val {_num(metrics[id_key])}{' *best' if metrics.get('checkpoint/new_best') == 1.0 else ''}")
        additional_key = f"additional_evaluation/{self.objective_id}"
        if additional_key in metrics:
            parts.append(f"add-val {_num(metrics[additional_key])}")
        if "physics/id/momentum_residual_mse" in metrics:
            residuals = (metrics[f"physics/id/{name}"] for name in ("momentum_residual_mse", "flux_balance_mse", "pressure_boundary_mse"))
            parts.append("phys mom/flux/bnd " + "/".join(_num(value, digits=2) for value in residuals))
        learning_rate = f"lr {_num(metrics.get('optimization/learning_rate'), digits=2)}"
        if "optimization/scheduler_new_learning_rate" in metrics:
            learning_rate += f"->{_num(metrics['optimization/scheduler_new_learning_rate'], digits=2)}"
        parts.append(learning_rate)
        if "system/epoch_duration_seconds" in metrics:
            parts.append(common.console.format_duration(metrics["system/epoch_duration_seconds"]))
        if "system/train_samples_per_second" in metrics:
            parts.append(f"{metrics['system/train_samples_per_second']:.1f} samples/s")
        self._training_seconds += metrics.get("system/train_duration_seconds", 0.0)
        self._validation_seconds += sum(
            value
            for key, value in metrics.items()
            if key.endswith("_evaluation_duration_seconds") or key == "system/physics_monitor_duration_seconds"
        )
        _line("  ".join(parts))

    def final(self, result: Mapping[str, Any], *, total_wall_seconds: float) -> dict[str, float]:
        """Print the end line with the epochs run, why training stopped and the duration parts; return the parts."""
        target = result.get("target_train_seconds")
        reason = "epoch budget" if target is None else f"train_seconds budget {common.console.format_duration(target)}"
        parts = {"training": self._training_seconds, "validation": self._validation_seconds}
        if "training_start" in self._phases:
            parts = {"startup": self._phases["training_start"]["elapsed_seconds"], **parts}
        _line(
            f"{result.get('status', 'completed')} after {result['completed_epoch']} of {self.config['training']['epochs']} epochs ({reason})"
            f"  best epoch {result['best_epoch']}  {self.objective_id} {_num(result['best_metric'])}"
            f"  {duration_summary(total_wall_seconds, parts)}  run {display_path(self.run_dir)}"
        )
        return parts

    def failure(self, error: BaseException, *, status: str, phase: str | None = None) -> None:
        """Print the sanitized failure, the last reached startup phase, and how to continue; the caller owns any traceback."""
        epoch = getattr(error, "completed_epoch", None)
        checkpoint = common.paths.resolve_last_checkpoint_file(self.run_dir).is_file()
        _line(
            f"{'interrupted' if status == 'interrupted' else 'FAILED'} {_text(self.config['run']['name'])}"
            f" in {phase or getattr(error, 'training_phase', 'run_lifecycle')}"
            f"{'' if epoch is None else f' after epoch {epoch}'}: {type(error).__name__}: {_clean(error)}",
            to_stderr=True,
        )
        if self._phases and "training_start" not in self._phases:
            _line(f"  startup stopped after {next(reversed(self._phases))}: {_text(list(self._phases.values())[-1])}", to_stderr=True)
        _line(
            "  next: "
            + ("rerun the same command to continue from the last checkpoint" if checkpoint else "rerun the same command, no checkpoint saved yet")
            + f" | W&B {status} | run {display_path(self.run_dir)}",
            to_stderr=True,
        )


def sanitized_exception_message(error: BaseException) -> str:
    """Return one bounded, single-line, secret-redacted exception message."""
    return _clean(error)


def print_sanitized_traceback(error: BaseException) -> None:
    """Print the complete exception chain and frames with secrets redacted."""
    rendered = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    sys.stderr.write(_redact_secrets(rendered))
    sys.stderr.flush()


def print_error_notes(error: BaseException) -> None:
    """Print the notes attached to an error whose outcome prints no traceback, with secrets redacted."""
    for note in getattr(error, "__notes__", ()):
        sys.stderr.write(_redact_secrets(f"note: {note}\n"))
    sys.stderr.flush()


def run_admission_event(status: str, **fields: object) -> None:
    """Print one stable line about how a start that names an existing run proceeds."""
    _line(f"run {status}: " + "  ".join(f"{key} {_text(value)}" for key, value in fields.items() if value is not None))


def optuna_study_event(status: str, **fields: Any) -> None:
    """Print the Optuna study start line or end line."""
    if status == "started":
        _line(
            f"study {_text(fields['study'])} started | {fields['task']} {fields['model_family']} | {fields['sampler']}/{fields['pruner']}"
            f" | {fields['invocation_trials']} of {fields['configured_trials']} trials, max {fields['maximum_epochs']} epochs"
            f" | {fields['continuation']} | storage {_text(fields['storage'])}"
        )
        return
    best = "no completed trial" if fields["best_trial"] is None else f"best trial {fields['best_trial']} {_num(fields['best_objective'])}"
    _line(
        f"study {_text(fields['study'])} {status} | {fields['attempted_trials']} trials ({fields['completed_trials']} completed,"
        f" {fields['pruned_trials']} pruned, {fields['failed_trials']} failed) | {best}"
        f" | {common.console.format_duration(fields['duration_seconds'])} | {_text(fields['best_parameters'])} | storage {_text(fields['storage'])}"
        + ("" if fields.get("summary_file") is None else f" | summary {fields['summary_file']}"),
        to_stderr=status == "failed",
    )


def optuna_trial_event(
    status: str,
    *,
    trial: int,
    sampled: Mapping[str, Any],
    duration_seconds: float,
    objective: float | None = None,
    step: int | None = None,
) -> None:
    """Print one line for a finished Optuna trial; pruning and duplicates are normal outcomes, any other state goes to stderr."""
    state = status if step is None or status == "completed" else f"{status}@{step}"
    _line(
        f"trial {trial:>3}  {state}  val {_num(objective)}  {common.console.format_duration(duration_seconds)}  {_text(sampled)}",
        to_stderr=status not in {"completed", "duplicate"} and not status.endswith("pruned"),
    )
