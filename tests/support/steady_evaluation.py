"""Provide a compact synthetic steady Evaluation frame with payload files and complete provenance.

Responsibilities:
  - Re-export the two-case steady artifact frame, which the plot catalog owns, for panel, session, and dataset-cut tests

Design principles:
  - One builder serves the tests and the catalog, so the fixture shape cannot drift between them

This module does NOT:
  - Build the frame itself; ``src.analysis.catalog.analysis_catalog_evaluation`` does
"""

from __future__ import annotations

from src.analysis.catalog.analysis_catalog_evaluation import steady_frame

__all__ = ["steady_frame"]
