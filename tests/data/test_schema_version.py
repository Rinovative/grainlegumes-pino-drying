"""Protect the strict schema-version predicate shared by every document reader.

Responsibilities:
  - Check that only an exact integer equal to the expected version matches

Design principles:
  - JSON booleans and floats that Python equates with an integer are never accepted as a schema version

This module does NOT:
  - Exercise individual document readers
"""

import pytest

from src import common


@pytest.mark.parametrize(
    ("value", "expected", "matches"),
    [
        (1, 1, True),
        (2, 1, False),
        (True, 1, False),
        (False, 0, False),
        (1.0, 1, False),
        ("1", 1, False),
        (None, 1, False),
    ],
)
def test_is_schema_version_accepts_only_the_exact_integer(value: object, expected: int, matches: bool) -> None:
    """Python equates True with 1 and 1.0 with 1; the predicate must not."""
    assert common.serialization.is_schema_version(value, expected) is matches
