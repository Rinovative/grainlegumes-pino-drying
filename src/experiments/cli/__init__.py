"""
Command-line entry points for experiment workflows.

Executable modules:
- cli_build_artifacts: artifact-generation command
- cli_config_preflight: configuration-preflight command
- cli_device: shared --device argument of the training, Optuna and artifact commands (no command of its own)
- cli_optuna: Optuna-study command
- cli_probe: container and device probe command
- cli_train: model-training command
"""

__all__: list[str] = []
