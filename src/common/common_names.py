"""Render numbers and text as compact, path-safe name tokens.

Responsibilities:
  - Render finite numeric settings as lexical tokens without dots or signs
  - Render switch values and labels as lowercase hyphenated text tokens

Design principles:
  - Tokens are deterministic pure functions of their input
  - Tokens are readable name parts, never scientific identity
  - Non-finite or non-numeric numbers fail instead of producing ambiguous names

This module does NOT:
  - Compose, validate, or bound complete path components. ``paths`` owns that
  - Decide which settings appear in a name. Naming owners in their domains do
"""

import math
import re


def number_token(value: object, *, precision: int | None = None) -> str:
    """Render a finite numeric setting as a safe compact name token.

    Parameters
    ----------
    value : object
        Finite ``int`` or ``float``; booleans are rejected.
    precision : int | None, optional
        Fixed number of decimals applied before trailing zeros are stripped.

    Returns
    -------
    str
        Token without trailing zeros where ``.`` becomes ``p`` and ``-`` becomes
        ``minus``, for example ``0.495`` -> ``0p495``.

    Raises
    ------
    ValueError
        If ``value`` is not a finite number.

    """
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        message = f"Name numbers must be finite numbers, got {value!r}."
        raise ValueError(message)
    text = format(value, f".{precision}f") if precision is not None else str(value)
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text.replace(".", "p").replace("-", "minus")


def text_token(value: object) -> str:
    """Render a switch value lexically; scientific identity retains the exact value.

    Booleans become ``yes``/``no`` and numbers use :func:`number_token`. Other
    values are lowercased, runs outside ``[a-z0-9-]`` collapse to one hyphen, and
    an empty result becomes ``none``.
    """
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return number_token(value)
    return re.sub(r"[^a-z0-9-]+", "-", str(value).lower()).strip("-") or "none"
