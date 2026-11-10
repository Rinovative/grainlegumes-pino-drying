"""Protect the one-authority layout of the maintained learning configurations.

Responsibilities:
  - Check that the transient A0 studies, which cannot inherit base.yaml, restate exactly its shared values
  - Check that experiments and studies state nothing their base or the model factory already supplies
  - Check that every physics-informed config names its derivative backend
  - Check that no code default holds a leaf of a task's base.yaml

Design principles:
  - Only the layout of the maintained files is checked, never their scientific values
  - Each check reads the YAML text, so a failing path names the file and the key

This module does NOT:
  - Resolve configs or exercise the loader
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from src.experiments.config import experiments_config_defaults as config_defaults
from src.learning.models import learning_models_factory as model_factory

if TYPE_CHECKING:
    from collections.abc import Iterator

_CONFIGS = Path(__file__).resolve().parents[2] / "configs"
_LEARNING = _CONFIGS / "03_learning"
_TASKS = ("steady_flow", "transient_drying")
# The only shared leaves an A0 study may set differently from base.yaml; model, lr, sampling, curriculum and training are not base leaves.
_A0_DIFFERENCES = {("run", "device"), ("tracking", "wandb", "workflow")}
_IDENTITY_KEYS = ("task", "base")
_MODEL_PARAMS_DEPTH = 3


def _load(path: Path) -> dict[str, Any]:
    """Return one YAML mapping."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _leaves(node: Any, prefix: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], Any]]:
    """Yield every key path whose value is a scalar, a list or an empty mapping."""
    if isinstance(node, dict) and node:
        for key, value in node.items():
            yield from _leaves(value, (*prefix, str(key)))
    else:
        yield prefix, node


def _base_leaves(task: str) -> dict[tuple[str, ...], Any]:
    """Return the leaves of one task's base.yaml."""
    return dict(_leaves(_load(_LEARNING / task / "base.yaml")))


def _authored(task: str) -> list[tuple[str, bool, dict[str, Any]]]:
    """Return name, whether it is a transient A0 study, and the authored experiment block of every experiment and study."""
    blocks = []
    for path in sorted((_LEARNING / task).rglob("*.yaml")):
        if path.name == "base.yaml":
            continue
        raw = _load(path)
        block = raw.get("experiment", raw)
        stated = {key: value for key, value in block.items() if key not in _IDENTITY_KEYS}
        blocks.append((path.relative_to(_LEARNING / task).as_posix(), "experiment" in raw and task == "transient_drying", stated))
    return blocks


def _is_tuned(path: tuple[str, ...]) -> bool:
    """Return whether an A0 study may state a leaf base.yaml does not: its model, lr, sampling, training block, device and workflow."""
    return path[0] in {"model", "training"} or path[:2] in {("optimizer", "lr"), ("temporal", "sampling")} or path in _A0_DIFFERENCES


def _has_path(mapping: dict[str, Any], dotted: str) -> bool:
    """Return whether a dotted key path exists in a nested mapping."""
    node: Any = mapping
    for token in dotted.split("."):
        if not isinstance(node, dict) or token not in node:
            return False
        node = node[token]
    return True


def test_a0_studies_restate_exactly_the_shared_base_values() -> None:
    """A0 studies cannot inherit base.yaml, so every non-training base value must reappear in them unchanged."""
    base = _base_leaves("transient_drying")
    shared = {path: value for path, value in base.items() if path[0] != "training" and path not in _A0_DIFFERENCES}
    studies = [(name, dict(_leaves(block))) for name, is_study, block in _authored("transient_drying") if is_study]

    assert studies
    for name, study in studies:
        missing = sorted(".".join(path) for path in shared if path not in study)
        changed = sorted(".".join(path) for path, value in shared.items() if path in study and study[path] != value)
        foreign = sorted(".".join(path) for path in study if path not in base and not _is_tuned(path))
        assert (name, missing, changed, foreign) == (name, [], [], [])


@pytest.mark.parametrize("task", _TASKS)
def test_experiments_and_studies_state_only_what_base_and_the_model_factory_do_not(task: str) -> None:
    """Restating a base value or a model-factory default hides the one place that owns it."""
    base = _base_leaves(task)
    restated = []
    for name, is_a0_study, block in _authored(task):
        defaults = model_factory.model_defaults(block["model"]["kind"])
        for path, value in _leaves(block):
            if not is_a0_study and path in base and base[path] == value:
                restated.append(f"{name}: {'.'.join(path)} equals base.yaml")
            if len(path) == _MODEL_PARAMS_DEPTH and path[:2] == ("model", "params") and path[2] in defaults and defaults[path[2]] == value:
                restated.append(f"{name}: {'.'.join(path)} equals the {block['model']['kind']} default")

    assert restated == []


def test_physics_informed_configs_state_their_derivative_backend() -> None:
    """The __fd or __spec file name promises a backend, so the file states it instead of leaving it to a default."""
    physics_informed = []
    for name, _, block in _authored("steady_flow"):
        physics = block.get("loss", {}).get("physics", {})
        if physics.get("enabled") is True:
            physics_informed.append(name)
            assert physics.get("derivatives", {}).get("kind") is not None, name

    assert physics_informed


@pytest.mark.parametrize("task", _TASKS)
def test_code_defaults_hold_no_leaf_the_base_states(task: str) -> None:
    """One authority per value: the code defaults hold no leaf of the task's base.yaml, and the required paths are base leaves."""
    base = _base_leaves(task)
    defaults = config_defaults.get_task_defaults(task)

    assert [".".join(path) for path in base if _has_path(defaults, ".".join(path))] == []
    assert [path for path in config_defaults.REQUIRED_PATHS[task] if tuple(path.split(".")) not in base] == []
