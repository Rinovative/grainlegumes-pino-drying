"""Protect the shared number and text name tokens that compose readable path names.

Responsibilities:
  - Check number tokens for integers, decimals, signs, fixed precision and rejected values
  - Check text tokens for booleans, numbers and free text

Design principles:
  - Tokens are deterministic and path-safe so that names never depend on runtime state

This module does NOT:
  - Compose complete names or validate path components
"""

from __future__ import annotations

import math

import pytest

from src import common


@pytest.mark.parametrize(
    ("value", "precision", "expected"),
    [
        (4, None, "4"),
        (1.0, None, "1"),
        (0.495, None, "0p495"),
        (-0.5, None, "minus0p5"),
        (0.49513182, 3, "0p495"),
        (2.0, 3, "2"),
        (1e-05, None, "1eminus05"),
    ],
)
def test_number_token_strips_trailing_zeros_and_replaces_dot_and_sign(value: float, precision: int | None, expected: str) -> None:
    """Numbers become dot-free, sign-free tokens with optional fixed precision."""
    assert common.names.number_token(value, precision=precision) == expected


@pytest.mark.parametrize("value", [math.nan, math.inf, True, "3", None])
def test_number_token_rejects_non_finite_and_non_numbers(value: object) -> None:
    """Booleans, text and non-finite floats never become ambiguous name tokens."""
    with pytest.raises(ValueError, match="finite numbers"):
        common.names.number_token(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(True, "yes"), (False, "no"), (0.25, "0p25"), (8, "8"), ("Soft Gating!", "soft-gating"), ("", "none"), ("--", "none")],
)
def test_text_token_is_lexical(value: object, expected: str) -> None:
    """Switch values render as lowercase hyphenated tokens, numbers as number tokens."""
    assert common.names.text_token(value) == expected
