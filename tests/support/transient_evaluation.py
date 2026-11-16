"""Provide compact synthetic transient Evaluation records, frames, and scaling state.

Responsibilities:
  - Expose the in-memory sequence records, artifact frames and scaling state that the plot catalog owns, for session and artifact tests

Design principles:
  - One builder serves the tests and the plot catalog, because ``src`` must never import from ``tests``
  - States are constant over the smallest grid with an H1 interior, so per-case errors equal the offsets

This module does NOT:
  - Define the synthetic inputs; ``src.analysis.catalog.analysis_catalog_transient`` does
  - Run inference, write artifacts, or represent production results
"""

from __future__ import annotations

from src.analysis.catalog.analysis_catalog_transient import (
    ANALYSIS_SHA256,
    GRID_SHAPE,
    production_frame,
    scaling_state,
    sequence_frame,
    sequence_record,
    static_conditioning,
)

__all__ = ["ANALYSIS_SHA256", "GRID_SHAPE", "production_frame", "scaling_state", "sequence_frame", "sequence_record", "static_conditioning"]
