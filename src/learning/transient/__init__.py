"""
Transient task-owned learning adaptation.

Provides:
- adapter: task-owned transient training adaptation
- contracts: immutable transient tensor contracts
- curriculum: rollout and matched-compute contracts
- rollout: differentiable teacher-forced and self-fed execution
- tensorizer: physical-batch tensorization and reconstruction
- scaling: train-only transient scaling artifacts
- handoff: immutable transient teacher handoffs
- history: durable transient completed-epoch history
"""

from . import learning_transient_adapter as adapter
from . import learning_transient_contracts as contracts
from . import learning_transient_curriculum as curriculum
from . import learning_transient_handoff as handoff
from . import learning_transient_history as history
from . import learning_transient_rollout as rollout
from . import learning_transient_scaling as scaling
from . import learning_transient_tensorizer as tensorizer

__all__ = ["adapter", "contracts", "curriculum", "handoff", "history", "rollout", "scaling", "tensorizer"]
