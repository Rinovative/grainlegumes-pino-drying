"""
Scientific field, moisture, permeability, physics, and task contracts.

Provides:
- field_sets: named input and output field collections
- fields: field definitions, units, and tensor semantics
- moisture: canonical dry-basis, wet-basis, and bulk-moisture conversions
- permeability: permeability representations and validation
- physics: boundary, derivative, and Brinkman operator services
- tasks: registered scientific task specifications
"""

from . import domain_field_sets as field_sets
from . import domain_fields as fields
from . import domain_moisture as moisture
from . import domain_permeability as permeability
from . import physics, tasks

__all__ = ["field_sets", "fields", "moisture", "permeability", "physics", "tasks"]
