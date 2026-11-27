"""
Paired per-parameter field-test analysis: endpoints, frozen tolerances and verdicts.

Provides:
- endpoints: case reading, field-test regions and the paired endpoint measures of both tasks
- tolerances: the validation-population yardstick of trained runs, its freeze and its admission
- verdicts: pairing validity, verdict rules and the published verdict files
- workflow: the two orchestration calls of a field test

Imported as ``from src.analysis import field_tests``; it is deliberately not listed in ``src.analysis``'s surface,
because the analysis code projections hash that initializer and the field test must not change the recorded identity
of evaluation artifacts.
"""

from . import field_tests_endpoints as endpoints
from . import field_tests_tolerances as tolerances
from . import field_tests_verdicts as verdicts
from . import field_tests_workflow as workflow

__all__ = ["endpoints", "tolerances", "verdicts", "workflow"]
