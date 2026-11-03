"""Define canonical permeability and porosity field names.

Responsibilities:
  - Declare the canonical permeability and porosity field-name tuples

Design principles:
  - Field names are declarative constants independent of model architecture
  - The tuples provide reusable vocabulary, not learned channel order
  - Registered TaskSpec objects remain authoritative for learned ordering

This module does NOT:
  - Define task-specific tensor layouts or model input/output membership
  - Define permeability storage conversion or source-field mapping
"""

from __future__ import annotations

PERMEABILITY_FIELDS = ("Kxx", "Kxy", "Kyy")
POROSITY_FIELDS = ("eps_bed",)
