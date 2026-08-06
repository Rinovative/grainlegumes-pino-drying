"""
===============================================================================
common_queue_log_cli.py
===============================================================================
Resolve one queue-log directory for non-Python launchers.

Responsibilities:
  - Parse one experiment queue-log scope from a maintained launcher
  - Delegate validated storage-path derivation to ``common.paths``
  - Print the resolved directory as one machine-readable line

Design principles:
  - The CLI contains no storage naming or composition logic
  - ``STORAGE_ROOT`` remains the sole environment-level storage override
  - Shell launchers consume the same authoritative path contract as Python code

This module does NOT:
  - Create directories, log files, runs, studies, or queue jobs
  - Validate Docker, GPU, scheduler, or experiment configuration state
  - Translate container paths to host mount paths
===============================================================================
"""

from __future__ import annotations

import argparse

from . import common_paths as paths


def _build_parser() -> argparse.ArgumentParser:
    """Build the single-purpose queue-log path parser."""
    parser = argparse.ArgumentParser(
        description="Resolve a validated queue-log directory below STORAGE_ROOT",
    )
    parser.add_argument("scope", help="Safe experiment or artifact log scope")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Print one queue-log directory resolved by the common path owner."""
    args = _build_parser().parse_args(argv)
    print(paths.resolve_queue_log_dir(args.scope))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
