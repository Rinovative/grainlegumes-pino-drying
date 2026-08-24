# ruff: noqa: S101
"""Protect console redaction without freezing reporting prose."""

from __future__ import annotations

from typing import TYPE_CHECKING

from support import configs

from src import experiments

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_console_failure_does_not_disclose_secrets(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Redact credentials from both startup and failure diagnostics."""
    secret = "synthetic-credential-value"  # noqa: S105 - artificial redaction sentinel
    monkeypatch.setenv("WANDB_API_KEY", secret)
    config = experiments.config.loader.resolve_config(configs.direct_config())
    reporter = experiments.console.ConsoleReporter(config=config, run_dir=tmp_path)

    reporter.startup(resolved_device="cpu")
    message = f"api_key={secret}"
    reporter.failure(RuntimeError(message), status="failed")

    captured = capsys.readouterr()
    visible = captured.out + captured.err
    assert secret not in visible
    assert "<redacted>" in captured.err


def test_console_emits_bounded_semantic_startup_progress(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Expose shard-level startup progress without sample-level output."""
    config = experiments.config.loader.resolve_config(configs.direct_config())
    reporter = experiments.console.ConsoleReporter(config=config, run_dir=tmp_path)

    reporter.startup_phase(
        "scaler_fit_progress",
        {"shards_processed": 2, "shards_total": 5},
    )

    output = capsys.readouterr().out
    assert "event=scaler_fit_progress" in output
    assert "shards_processed=2" in output
    assert "shards_total=5" in output
    assert "elapsed_seconds=" in output
