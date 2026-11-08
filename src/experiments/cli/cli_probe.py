"""Check the ML container, its command modules and the visible device inside a Slurm worker allocation.

Responsibilities:
  - Import the train, Optuna, and artifact command modules
  - Require the worker's Slurm provenance with a job ID and an image digest
  - Require one usable GPU (the V100 model for ``gpu:v100:1``) in ``gpu`` mode and none in ``cpu`` mode, then print one pass line

Design principles:
  - A failed check raises, so the job fails without a pass line

This module does NOT:
  - Validate the complete provenance schema. ``experiments_run`` owns that boundary
  - Open Datasets, configurations, or run directories
"""

from __future__ import annotations

import importlib
import json
import os
import sys

import torch


def _require(condition: object, message: str) -> None:
    """Raise ``RuntimeError`` with the message unless the probe condition holds."""
    if not condition:
        raise RuntimeError(message)


def main(argv: list[str]) -> int:
    """Run the probe for the one allocation mode ``cpu`` or ``gpu`` and print its pass line."""
    (mode,) = argv
    for name in ("cli_train", "cli_optuna", "cli_build_artifacts"):
        importlib.import_module(f"src.experiments.cli.{name}")
    provenance = json.loads(os.environ["ML_SLURM_PROVENANCE"])
    _require(provenance["job_id"] and provenance["sif_sha256"], "ML_SLURM_PROVENANCE must name the job and the image digest.")
    if mode == "gpu":
        _require(torch.cuda.device_count() == 1, "Expected exactly one framework-visible GPU.")
        device_name = torch.cuda.get_device_name(0)
        _require(provenance["gres"] != "gpu:v100:1" or device_name == "Tesla V100-PCIE-32GB", f"Unexpected V100 model: {device_name}")
        torch.empty(1, device="cuda").sum().item()
    else:
        _require(mode == "cpu", f"Unknown probe mode: {mode}")
        device_name = "none"
    visible_gpus = torch.cuda.device_count()
    _require(mode == "gpu" or visible_gpus == 0, "Expected no framework-visible GPUs in CPU mode.")
    print(f"ML probe passed: mode={mode} job={provenance['job_id']} visible_gpus={visible_gpus} device={device_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
