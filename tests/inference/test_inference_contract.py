"""Protect completed-run admission and strict inference versus resume checkpoint roles.

Responsibilities:
  - Check completed-run admission, required files, checkpoint roles, and exact normalizer reuse
  - Check that weights recorded by other model code are refused before any inference state is built
  - Check that inference batches steady items with absent metadata exactly as training does

Design principles:
  - Incomplete bundles fail before reconstruction or redundant loading

This module does NOT:
  - Restore full resume state or perform production inference
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import pytest
import torch
from support import configs, run_state
from torch import nn
from torch.utils.data import Dataset

from src import analysis, common, datasets, experiments, learning

if TYPE_CHECKING:
    from pathlib import Path

_REQUIRED_PAYLOAD_FILES = (
    "config.yaml",
    "normalizer.pt",
    "split_indices.pt",
    "best_checkpoint.pt",
    "last_checkpoint.pt",
)


class _SyntheticDataset(Dataset[dict[str, Any]]):
    """
    Model the minimal field-aware dataset needed for context reconstruction.

    The single zero sample deliberately omits production storage and scientific
    meaning. It only exercises validated normalizer and split wiring.
    """

    input_fields: ClassVar[list[str]] = ["source"]
    output_fields: ClassVar[list[str]] = ["target"]

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index != 0:
            raise IndexError(index)
        return {
            "x": torch.zeros(1, 1, 1),
            "y": torch.zeros(1, 1, 1),
            "meta": {},
        }


def _status_run(tmp_path: Path, status: str, *, touch_payloads: bool) -> Path:
    """
    Create one synthetic run leaf at a requested lifecycle status.

    Optional empty payload files test that filename presence cannot override the
    authoritative status marker. No payload is intended to be deserialized.
    """
    run_dir = experiments.run.allocate_run_directory(tmp_path / status)
    experiments.run.transition_run_status(run_dir, "initializing")
    if status != "initializing":
        experiments.run.transition_run_status(run_dir, "running")
    if status not in {"initializing", "running"}:
        experiments.run.transition_run_status(run_dir, status)
    if touch_payloads:
        for filename in _REQUIRED_PAYLOAD_FILES:
            (run_dir / filename).touch()
    return run_dir


@pytest.mark.parametrize("status", ["running", "interrupted", "budget_unmet"])
def test_inference_and_artifacts_reject_non_completed_status(tmp_path: Path, status: str) -> None:
    """
    Create a non-completed run leaf containing every required payload filename.

    Inference and artifact planning must both reject it by lifecycle status and
    tell the user to resume the run to completion, proving apparent file
    completeness cannot bypass completion admission.
    """
    run_dir = _status_run(tmp_path, status, touch_payloads=True)

    with pytest.raises(experiments.run.RunLifecycleError, match=rf"{status}.*Resume the run to completion"):
        learning.inference.context.load_inference_context(run_dir=run_dir, device_policy="cpu")
    with pytest.raises(experiments.run.RunLifecycleError, match=rf"{status}.*Resume the run to completion"):
        analysis.artifacts.service.load_run_artifact_plan(run_dir)


@pytest.mark.parametrize("status", ["initializing", "interrupted"])
def test_non_completed_run_without_payload_files_gets_the_status_diagnostic(tmp_path: Path, status: str) -> None:
    """
    Create a non-completed run leaf without any required payload files.

    Every admission entry point must name the lifecycle status and the resume
    hint before it requires payload files, so a run interrupted before its
    first best checkpoint is not reported as merely missing a checkpoint.
    """
    run_dir = _status_run(tmp_path, status, touch_payloads=False)

    with pytest.raises(experiments.run.RunLifecycleError, match=rf"{status}.*Resume the run to completion"):
        learning.inference.context.load_inference_context(run_dir=run_dir, device_policy="cpu")
    with pytest.raises(experiments.run.RunLifecycleError, match=rf"{status}.*Resume the run to completion"):
        analysis.artifacts.service.load_run_artifact_plan(run_dir)
    with pytest.raises(experiments.run.RunLifecycleError, match=rf"{status}.*Resume the run to completion"):
        experiments.run.validate_completed_run(run_dir)


def test_completed_run_without_payload_files_is_rejected_before_reconstruction(tmp_path: Path) -> None:
    """Reject a completed summary whose evaluation payload files are absent at run admission."""
    run_dir = _status_run(tmp_path, "completed", touch_payloads=False)

    with pytest.raises(experiments.run.RunLifecycleError, match=r"evaluation evidence|best checkpoint"):
        learning.inference.context.load_inference_context(run_dir=run_dir, device_policy="cpu")
    with pytest.raises(experiments.run.RunLifecycleError, match=r"evaluation evidence|best checkpoint"):
        analysis.artifacts.service.load_run_artifact_plan(run_dir)


def test_continuation_requires_last_checkpoint_even_when_other_files_exist(tmp_path: Path) -> None:
    """
    Remove ``last_checkpoint.pt`` from an otherwise populated running run.

    Rerunning the request must be refused on the continuation artifact instead of
    substituting the selection-only best checkpoint, preserving exact optimizer/RNG
    continuation.
    """
    output_root = tmp_path / "outputs"
    config = configs.resolve(configs.direct_config(), output_root=output_root)
    run_dir = common.paths.resolve_config_run_dir(config)
    run_state.write_lightweight_run_state(run_dir, config, status="running", completed_epoch=1)
    common.paths.resolve_last_checkpoint_file(run_dir).unlink()
    request = configs.write_config(tmp_path, configs.direct_config(), canonical_root=tmp_path)

    with pytest.raises(experiments.run.ExistingRunAdmissionError, match=r"last_checkpoint\.pt"):
        experiments.run.run_experiment(request, output_root=output_root)

    assert common.paths.resolve_best_checkpoint_file(run_dir).is_file()


def _producer(task_id: str, *, model_sha256: str | None = None) -> dict[str, str]:
    """Return run producer evidence recording the running model code unless another digest is given."""
    digest = learning.training.checkpoint.model_implementation_digest(task_id) if model_sha256 is None else model_sha256
    return {"training_sha256": "a" * 64, "model_sha256": digest}


def test_inference_refuses_weights_of_other_model_code_before_building_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Loading persisted weights continues the trained computation, so a changed model digest is refused before seeding or model setup."""
    context = learning.inference.context
    completed_run = {"scientific_run_name": "run", "config": {"task": "steady_flow", "producer": _producer("steady_flow", model_sha256="0" * 64)}}
    monkeypatch.setattr(experiments.run, "validate_evaluable_run", lambda _run_dir: completed_run)
    monkeypatch.setattr(context, "_field_contract", lambda _config: (["source"], ["target"]))
    monkeypatch.setattr(experiments.run, "configure_reproducibility", lambda *_args, **_kwargs: pytest.fail("seeded before the model guard"))
    monkeypatch.setattr(context, "_build_model_from_config", lambda *_args, **_kwargs: pytest.fail("built a model before the guard"))

    with pytest.raises(ValueError, match=r"^Run run: model_steady_flow implementation changed .*--revision <label>$"):
        context.load_inference_context(run_dir=tmp_path / "run", device_policy="cpu")


def test_inference_uses_exact_normalizer_state_returned_by_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Return validated normalizer tensors directly and forbid every later ``torch.load``.

    Context reconstruction must use those exact objects and still build the selected
    loader/model, preventing a time-of-check/time-of-use artifact reopen.
    """
    context = learning.inference.context
    model = nn.Conv2d(1, 1, kernel_size=1, bias=False)
    normalizer_state = {
        "in_normalizer.mean": torch.zeros(1, 1, 1, 1),
        "in_normalizer.std": torch.ones(1, 1, 1, 1),
        "out_normalizer.mean": torch.zeros(1, 1, 1, 1),
        "out_normalizer.std": torch.ones(1, 1, 1, 1),
    }
    completed_run = {
        "scientific_run_name": "run",
        "config": {"task": "steady_flow", "producer": _producer("steady_flow")},
        "split_indices": {},
        "best_checkpoint": {"model_state_dict": model.state_dict()},
        "normalizer_state": normalizer_state,
    }
    evidence = datasets.preprocessing.splits.SplitRoleEvidence(
        name="id_test",
        dataset={"dataset_id": "synthetic", "case_ids": ["sample-0"]},
        case_ids=("sample-0",),
        membership_digest="e" * 64,
    )
    selection = context.SplitSelection(role="id_test", evidence=(evidence,), train_subset=None)
    dataset = _SyntheticDataset()

    monkeypatch.setattr(experiments.run, "validate_evaluable_run", lambda _run_dir: completed_run)
    monkeypatch.setattr(context, "_field_contract", lambda _config: (["source"], ["target"]))

    def configure_reproducibility(_config: dict[str, Any], *, device: torch.device) -> dict[str, int]:
        assert device == torch.device("cpu")
        return {"model_init": 1}

    def seed_process(_seed: int, *, device: torch.device) -> None:
        assert device == torch.device("cpu")

    monkeypatch.setattr(experiments.run, "configure_reproducibility", configure_reproducibility)
    monkeypatch.setattr(experiments.run, "seed_process", seed_process)
    monkeypatch.setattr(context, "_select_split", lambda **_kwargs: selection)

    def build_model(_config: dict[str, Any], *, device: torch.device) -> nn.Module:
        assert device == torch.device("cpu")
        return model

    monkeypatch.setattr(context, "_build_model_from_config", build_model)
    monkeypatch.setattr(context, "_open_split_sources", lambda _selection, **_kwargs: [dataset])
    monkeypatch.setattr(context.datasets.preprocessing.splits, "combine_identity_datasets", lambda _sources: dataset)

    def unexpected_torch_load(*_args: Any, **_kwargs: Any) -> Any:
        msg = "Inference reopened a validated Torch artifact"
        raise AssertionError(msg)

    monkeypatch.setattr(torch, "load", unexpected_torch_load)

    loaded_model, loader, processor, device = context.load_inference_context(
        run_dir=tmp_path / "run",
        device_policy="cpu",
    )

    assert loaded_model is model
    selected_dataset = loader.dataset
    assert isinstance(selected_dataset, context.IndexedSubset)
    assert len(selected_dataset) == 1
    assert device.type == "cpu"
    in_normalizer = processor.in_normalizer
    out_normalizer = processor.out_normalizer
    assert in_normalizer is not None
    assert out_normalizer is not None
    assert torch.equal(in_normalizer.mean, normalizer_state["in_normalizer.mean"])
    assert torch.equal(out_normalizer.std, normalizer_state["out_normalizer.std"])


def test_inference_loader_batches_steady_items_whose_metadata_holds_absent_values() -> None:
    """Evaluation artifacts must accept the same steady metadata that the training loader accepts."""
    items = [
        {
            "x": torch.zeros(2, 3, 3),
            "y": torch.ones(1, 3, 3),
            "meta": {"case": index, "inlet_profile": None},
            "source_index": index,
            "split_local_index": index,
        }
        for index in range(3)
    ]

    batch = next(iter(learning.inference.context._build_eval_loader(items, batch_size=2)))  # type: ignore[arg-type]

    assert batch["x"].shape == (2, 2, 3, 3)
    assert batch["meta"]["inlet_profile"] == [None, None]
    assert batch["source_index"].tolist() == [0, 1]
