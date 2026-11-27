"""Compare two campaigns of one task on the same designs, paired per case.

Responsibilities:
  - Pair the primary candidates of two campaigns on material, design key and case, and count the designs one campaign lacks
  - Verify that a shared design key binds equal sampled values and one case id, and refuse the comparison when it does not
  - Report the success-rate difference B - A with the Newcombe hybrid-score interval and the exact McNemar test on the discordant pairs
  - Tabulate paired per-case quantities on the pairs both campaigns completed successfully
  - Name the campaigns that share a design, so the view is offered only when a comparison exists

Design principles:
  - A design key is a label of regime and case index, not of the inputs: only equal sampled values make a pair the same design
  - A pair whose candidates carry no sampled values is unverified and left out; a shared key bound to other values is a conflict
  - The success statistics take the four paired counts only; a pair with a pending candidate is undecided and drawn nowhere
  - A result that needs at least one decided pair is ``None`` without one, never zero
  - A replacement candidate belongs to no design of the original Sobol sequence and never pairs
  - A campaign that repeats a design key is refused, because a pair would not be unique

This module does NOT:
  - Draw figures, build widgets or read campaigns from storage
  - Read completed outputs: the candidate rows carry the case durations, so those are the paired quantities

Notes:
  Newcombe (1998), Statistics in Medicine 17, 2635-2650, method 10: with the counts a (both), b (only A), c (only B), d (neither),
  n = a + b + c + d, p_A = (a + b) / n, p_B = (a + c) / n and the Wilson score intervals (l, u) of both proportions, the interval for
  p_B - p_A is the difference minus and plus the square roots
  ``(p_B - l_B)^2 + (u_A - p_A)^2 - 2 phi (p_B - l_B)(u_A - p_A)`` and ``(u_B - p_B)^2 + (p_A - l_A)^2 - 2 phi (u_B - p_B)(p_A - l_A)``.
  phi = C / sqrt((a + b)(c + d)(a + c)(b + d)), or 0 when a margin is empty, with B = ad - bc and C = B - n / 2 for B > n / 2, 0 for
  0 <= B <= n / 2 and B for B < 0.

"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from statistics import NormalDist
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

import pandas as pd
from scipy.stats import binom

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

type Campaign = tuple[str, str]  # (task, source name)

QUANTITIES: Final[Mapping[str, str]] = MappingProxyType({"duration_seconds": "Candidate duration [s]"})
_PRIMARY: Final = "primary"
_DESIGN: Final = ["material_family", "design_key"]
_KEY: Final = [*_DESIGN, "case_id"]
_DECIDED: Final = ("successful", "failed")
_Z: Final = NormalDist().inv_cdf(0.975)


@dataclass(frozen=True, slots=True)
class SuccessComparison:
    """Hold the paired success counts and, from at least one decided pair on, the rates, the B - A difference, its interval and McNemar's p."""

    both: int
    only_a: int
    only_b: int
    neither: int
    rate_a: float | None
    rate_b: float | None
    difference: float | None
    low: float | None
    high: float | None
    p_value: float | None

    @property
    def pairs(self) -> int:
        """Return the number of decided pairs."""
        return self.both + self.only_a + self.only_b + self.neither


@dataclass(frozen=True, slots=True)
class TemplateComparison:
    """Bind the decided pairs of two campaigns with their success statistics and the counts of what was left out."""

    campaign_a: Campaign
    campaign_b: Campaign
    quantities: Mapping[str, str]
    pairs: pd.DataFrame
    unmatched: int
    unverified: int
    pending: int
    success: SuccessComparison


def _wilson(successes: int, n: int) -> tuple[float, float]:
    """Return the Wilson score interval of ``successes`` in ``n`` at 95 %."""
    centre = (successes + _Z**2 / 2) / (n + _Z**2)
    half = _Z * math.sqrt(successes * (n - successes) / n + _Z**2 / 4) / (n + _Z**2)
    return max(0.0, centre - half), min(1.0, centre + half)


def newcombe_paired_interval(both: int, only_a: int, only_b: int, neither: int) -> tuple[float, float]:
    """Return the 95 % hybrid-score interval of p_B - p_A for paired proportions, Newcombe (1998) method 10; see the module notes."""
    n = both + only_a + only_b + neither
    if n < 1:
        message = "A paired interval needs at least one pair."
        raise ValueError(message)
    p_a, p_b = (both + only_a) / n, (both + only_b) / n
    (l_a, u_a), (l_b, u_b) = _wilson(both + only_a, n), _wilson(both + only_b, n)
    margins = (both + only_a) * (only_b + neither) * (both + only_b) * (only_a + neither)
    cross = both * neither - only_a * only_b
    corrected = cross - n / 2 if cross > n / 2 else min(cross, 0)
    phi = corrected / math.sqrt(margins) if margins else 0.0
    below_b, above_a = p_b - l_b, u_a - p_a  # the limits of the lower end
    above_b, below_a = u_b - p_b, p_a - l_a  # and of the upper end
    return (
        p_b - p_a - math.sqrt(below_b**2 + above_a**2 - 2 * phi * below_b * above_a),
        p_b - p_a + math.sqrt(above_b**2 + below_a**2 - 2 * phi * above_b * below_a),
    )


def mcnemar_exact_p(only_a: int, only_b: int) -> float:
    """Return the two-sided exact McNemar p: the Binomial(b + c, 1/2) tail at the smaller discordant count, doubled and capped at one."""
    return min(1.0, 2 * float(binom.cdf(min(only_a, only_b), only_a + only_b, 0.5)))


def compare_success(both: int, only_a: int, only_b: int, neither: int) -> SuccessComparison:
    """Return the paired success statistics of the four counts; without a pair every statistic is ``None``."""
    pairs = both + only_a + only_b + neither
    if not pairs:
        return SuccessComparison(both, only_a, only_b, neither, None, None, None, None, None, None)
    low, high = newcombe_paired_interval(both, only_a, only_b, neither)
    return SuccessComparison(
        both,
        only_a,
        only_b,
        neither,
        (both + only_a) / pairs,
        (both + only_b) / pairs,
        (only_b - only_a) / pairs,
        low,
        high,
        mcnemar_exact_p(only_a, only_b),
    )


def _number(value: object) -> float:
    """Return a finite real ``value`` as a float, else NaN."""
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else math.nan


def _sample(row: Mapping[str, Any]) -> str | None:
    """Return the sampled values and units of ``row`` as one order-independent text, or ``None`` when the row carries none."""
    values, units = row.get("parameters"), row.get("parameter_units")
    if not isinstance(values, Mapping) or not isinstance(units, Mapping) or not values:
        return None
    return json.dumps([values, units], sort_keys=True, default=str)


def _primaries(rows: Iterable[Mapping[str, Any]], campaign: Campaign, quantities: Mapping[str, str]) -> pd.DataFrame:
    """Return the primary candidates of ``campaign`` by pairing key with their sampled values, outcome and quantities."""
    records = [
        {**{key: row[key] for key in _KEY}, "sample": _sample(row), "outcome": row["outcome"], **{key: _number(row.get(key)) for key in quantities}}
        for row in rows
        if (row["task"], row["source_name"]) == campaign and row["origin"] == _PRIMARY
    ]
    frame = pd.DataFrame(records, columns=[*_KEY, "sample", "outcome", *quantities])
    if frame.duplicated(_DESIGN).any():
        message = f"Campaign {campaign[0]}/{campaign[1]} repeats a material and design key among its primary candidates."
        raise ValueError(message)
    return frame


def _refuse(conflicts: pd.DataFrame, reason: str) -> None:
    """Raise when ``conflicts`` holds designs the two campaigns bind differently, naming their count and the first."""
    if not conflicts.empty:
        first = conflicts.iloc[0]
        message = f"Shared designs with {reason}: {len(conflicts)}; first: {first['material_family']}/{first['design_key']}."
        raise ValueError(message)


def _merge_designs(rows: Sequence[Mapping[str, Any]], campaign_a: Campaign, campaign_b: Campaign, quantities: Mapping[str, str]) -> pd.DataFrame:
    """
    Outer-merge the primaries of two campaigns on the pairing key, refusing a shared design the campaigns bind differently.

    A material and design key that meet under two case ids, or whose candidates carry different sampled values, are one design
    that disagrees with itself. The campaigns' scrambled Sobol designs depend on their seed, so equal keys alone prove nothing.
    """
    merged = _primaries(rows, campaign_a, quantities).merge(
        _primaries(rows, campaign_b, quantities), on=_KEY, how="outer", suffixes=("_a", "_b"), indicator=True
    )
    both = merged["_merge"] == "both"
    one_sided = merged[~both]
    _refuse(one_sided[one_sided.duplicated(_DESIGN, keep=False)].drop_duplicates(_DESIGN), "different case ids")
    known = both & merged["sample_a"].notna() & merged["sample_b"].notna()
    _refuse(merged[known & (merged["sample_a"] != merged["sample_b"])], "different sampled values")
    return merged


def comparable_campaigns(rows: Iterable[Mapping[str, Any]]) -> dict[Campaign, tuple[Campaign, ...]]:
    """Return, per campaign, the other campaigns of its task that agree on every shared primary design and verify at least one."""
    candidates = tuple(rows)
    designs: dict[Campaign, set[tuple[str, str]]] = {}
    for row in candidates:
        if row["origin"] == _PRIMARY:
            designs.setdefault((row["task"], row["source_name"]), set()).add((row["material_family"], row["design_key"]))
    partners: dict[Campaign, tuple[Campaign, ...]] = {}
    for campaign, own in designs.items():
        agreeing = []
        for other, shared in designs.items():
            if other == campaign or other[0] != campaign[0] or not shared & own:
                continue
            try:
                merged = _merge_designs(candidates, campaign, other, {})
            except ValueError:  # a conflict, or a campaign that repeats a design key
                continue
            if ((merged["_merge"] == "both") & merged["sample_a"].notna() & merged["sample_b"].notna()).any():  # a verified shared design
                agreeing.append(other)
        if agreeing:
            partners[campaign] = tuple(agreeing)
    return partners


def compare_templates(
    rows: Iterable[Mapping[str, Any]], campaign_a: Campaign, campaign_b: Campaign, *, quantities: Mapping[str, str]
) -> TemplateComparison:
    """
    Pair the primary candidates of two campaigns of one task and compare their success and ``quantities``.

    Candidates pair on material, design key and case id; a design in one campaign only is counted in ``unmatched``, a pair whose
    candidates do not both carry sampled values in ``unverified``, a verified pair with a pending candidate in ``pending``, and the
    remaining decided pairs carry the success counts. ``quantities`` maps a numeric candidate key to its label.

    Raises
    ------
    ValueError
        If the campaigns are not two different ones of one task, or a shared design key binds another case id or other sampled values.

    """
    if campaign_a[0] != campaign_b[0] or campaign_a == campaign_b:
        message = "A template comparison pairs two different campaigns of one task."
        raise ValueError(message)
    merged = _merge_designs(tuple(rows), campaign_a, campaign_b, quantities)
    matched = merged[merged["_merge"] == "both"].drop(columns="_merge")
    verified = matched["sample_a"].notna() & matched["sample_b"].notna()
    decided = verified & matched["outcome_a"].isin(_DECIDED) & matched["outcome_b"].isin(_DECIDED)
    pairs = matched[decided].reset_index(drop=True)
    won_a, won_b = pairs["outcome_a"] == "successful", pairs["outcome_b"] == "successful"
    success = compare_success(int((won_a & won_b).sum()), int((won_a & ~won_b).sum()), int((~won_a & won_b).sum()), int((~won_a & ~won_b).sum()))
    unmatched = int((merged["_merge"] != "both").sum())
    return TemplateComparison(
        campaign_a, campaign_b, dict(quantities), pairs, unmatched, int((~verified).sum()), int((verified & ~decided).sum()), success
    )


def quantity_pairs(comparison: TemplateComparison, quantity: str) -> pd.DataFrame:
    """Return material, ``a`` and ``b`` of the pairs both campaigns completed successfully with a finite ``quantity`` in both."""
    pairs = comparison.pairs
    done = pairs[(pairs["outcome_a"] == "successful") & (pairs["outcome_b"] == "successful")]
    values = done.assign(a=done[f"{quantity}_a"], b=done[f"{quantity}_b"])
    return values.loc[values["a"].notna() & values["b"].notna(), ["material_family", "a", "b"]]


def quantity_table(comparison: TemplateComparison) -> pd.DataFrame:
    """Return per quantity the paired count, both medians and the median and P10 / P90 of the paired difference B - A."""
    records = []
    for quantity, label in comparison.quantities.items():
        pairs = quantity_pairs(comparison, quantity)
        difference = pairs["b"] - pairs["a"]
        records.append(
            {
                "quantity": label,
                "n paired": len(pairs),
                "median A": pairs["a"].median(),
                "median B": pairs["b"].median(),
                "median B - A": difference.median(),
                "P10 B - A": difference.quantile(0.1),
                "P90 B - A": difference.quantile(0.9),
            }
        )
    return pd.DataFrame(records)


def success_table(comparison: TemplateComparison) -> pd.DataFrame:
    """Return the one-row table of the decided pairs, both success fractions, the B - A difference with its interval and the McNemar test."""
    success = comparison.success
    return pd.DataFrame(
        [
            {
                "decided pairs": success.pairs,
                "success A": success.rate_a,
                "success B": success.rate_b,
                "B - A": success.difference,
                "95 % interval low": success.low,
                "95 % interval high": success.high,
                "only A successful": success.only_a,
                "only B successful": success.only_b,
                "McNemar p": success.p_value,
            }
        ]
    )
