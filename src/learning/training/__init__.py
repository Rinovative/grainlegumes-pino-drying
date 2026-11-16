"""
Budget-driven training, optimization, and persistence services.

Provides:
- adapter: task-owned training-step, evaluation-view and optimizer-work contracts
- budget: stage budgets in units or supervised transitions and their validation points
- checkpoint: atomic checkpoint and continuation-state persistence
- history: durable completed-epoch training history of both tasks
- loop: training and evaluation execution
- optim: optimizer and learning-rate schedule construction
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import learning_training_adapter as adapter
    from . import learning_training_budget as budget
    from . import learning_training_checkpoint as checkpoint
    from . import learning_training_history as history
    from . import learning_training_loop as loop
    from . import learning_training_optim as optim

_MODULES = {
    "adapter": "learning_training_adapter",
    "budget": "learning_training_budget",
    "checkpoint": "learning_training_checkpoint",
    "history": "learning_training_history",
    "loop": "learning_training_loop",
    "optim": "learning_training_optim",
}
__all__ = ["adapter", "budget", "checkpoint", "history", "loop", "optim"]


def __getattr__(name: str) -> object:
    """Resolve one declared public name on first access."""
    module_name = _MODULES.get(name)
    if module_name is None:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{module_name}")
    globals()[name] = module
    return module
