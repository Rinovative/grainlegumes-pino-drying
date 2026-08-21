# ruff: noqa: EM101, EM102, TRY003
"""Generate one validated, non-overwriting transient matched-compute YAML config."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

from src import experiments
from src.experiments.config import experiments_config_loader as config_loader
from src.learning.transient.learning_transient_curriculum import RolloutCurriculum


def generate_matched_config(
    *,
    source_run_dir: Path | str,
    output_path: Path | str,
    arm: str,
    budget: float | None = None,
    b_run_dir: Path | str | None = None,
) -> Path:
    """Generate one non-overwriting B or completed-B-derived A+ configuration."""
    source = experiments.run.validate_completed_run(source_run_dir)
    a0_config = source["config"]
    if a0_config.get("task") != "transient_drying" or a0_config["training"]["comparison_arm"] != "a0":
        raise ValueError("Matched config sources must be completed transient A0 runs.")
    if arm not in {"b", "a_plus"}:
        raise ValueError("arm must be b or a_plus.")
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"Matched config destination already exists: {output}")
    source_cuda = str(source["summary"].get("resolved_device", a0_config["run"].get("device", "cpu"))).startswith("cuda")
    if arm == "b":
        if budget is None or budget <= 0:
            raise ValueError("B requires a positive planned matched-compute budget.")
        if not source_cuda and float(budget) != float(int(budget)):
            raise ValueError("CPU B budgets must be exact integer optimizer-step counts.")
        generated = copy.deepcopy(a0_config)
        generated["temporal"]["sampling"] = {
            "mode": "rollout_window",
            "rollout_length": 32,
            "window_stride": 32,
            "window_offset": 0,
        }
        generated["training"].update(
            {
                "stage": "b",
                "comparison_arm": "b",
                "gradient_accumulation_steps": a0_config["training"]["gradient_accumulation_steps"],
                "fixed_evaluation_horizon": 32,
                "curriculum": {
                    "lengths": [2, 4, 8, 16, 32],
                    "milestone_fractions": list(RolloutCurriculum.DEFAULT_MILESTONE_FRACTIONS),
                    "seed": a0_config["training"]["curriculum"]["seed"],
                },
                "teacher_handoff": {"source_run_name": a0_config["run"]["name"]},
                "matched_compute": {
                    "planned_seconds": float(budget) if source_cuda else None,
                    "planned_steps": None if source_cuda else int(budget),
                    "rollout_reference_seconds": None,
                    "rollout_reference_steps": None,
                },
            }
        )
    else:
        if b_run_dir is None:
            raise ValueError("A+ requires a completed B run directory.")
        rollout = experiments.run.validate_completed_run(b_run_dir)
        b_config = rollout["config"]
        if b_config.get("task") != "transient_drying" or b_config["training"]["comparison_arm"] != "b":
            raise ValueError("A+ must be derived from a completed transient B run.")
        if b_config["training"]["teacher_handoff"] != {"source_run_name": a0_config["run"]["name"]}:
            raise ValueError("Completed B run does not reference the exact requested A0 source.")
        if b_config["model"] != a0_config["model"] or b_config["input_profile"] != a0_config["input_profile"]:
            raise ValueError("Completed B model/profile differs from its requested A0 source.")
        b_cuda = str(rollout["summary"].get("resolved_device", b_config["run"].get("device", "cpu"))).startswith("cuda")
        if b_cuda != source_cuda:
            raise ValueError("Completed B device clock kind differs from the A0 source.")
        terminal = rollout["summary"].get("terminal_controller")
        if not isinstance(terminal, dict):
            raise ValueError("Completed B run lacks terminal matched-compute evidence.")
        evidence_key = "post_handoff_optimizer_device_seconds" if b_cuda else "successful_optimizer_steps"
        evidence = terminal.get(evidence_key)
        if not isinstance(evidence, (int, float)) or isinstance(evidence, bool) or evidence <= 0:
            raise ValueError("Completed B run lacks positive terminal matched-compute evidence.")
        if not b_cuda and int(evidence) != evidence:
            raise ValueError("Completed B CPU evidence must be an exact optimizer-step count.")
        if budget is not None and float(budget) != float(evidence):
            raise ValueError("A+ budget, when supplied, must exactly equal completed-B terminal compute evidence.")
        generated = copy.deepcopy(b_config)
        generated["training"].update(
            {
                "stage": "a",
                "comparison_arm": "a_plus",
                "teacher_handoff": {"source_run_name": a0_config["run"]["name"]},
                "matched_compute": {
                    "planned_seconds": float(evidence) if b_cuda else None,
                    "planned_steps": None if b_cuda else int(evidence),
                    "rollout_reference_seconds": float(evidence) if b_cuda else None,
                    "rollout_reference_steps": None if b_cuda else int(evidence),
                },
            }
        )
    generated["run"]["suffix"] = arm
    generated["run"]["name"] = config_loader.generate_run_name(generated)
    generated.pop("_transient_tensorizer", None)
    resolved = config_loader.validate_resolved_config(generated)
    config_loader.save_yaml(resolved, output)
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a transient B or A+ matched-compute config.")
    parser.add_argument("source_run_dir")
    parser.add_argument("output_path")
    parser.add_argument("--arm", choices=("b", "a_plus"), required=True)
    parser.add_argument("--budget", type=float)
    parser.add_argument("--b-run-dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the matched-config command and report the generated path."""
    args = _parser().parse_args(argv)
    try:
        result = generate_matched_config(
            source_run_dir=args.source_run_dir,
            output_path=args.output_path,
            arm=args.arm,
            budget=args.budget,
            b_run_dir=args.b_run_dir,
        )
    except Exception as error:  # noqa: BLE001
        print(str(error), file=sys.stderr)
        return 2
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
