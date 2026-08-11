# ruff: noqa: S101, PLR2004, SLF001
"""Focused checks for configuration-derived Generation validation evidence."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from src import generation
from src.generation import generation_smoke as smoke


def _evidence(
    case_index: int,
    *,
    shared_offset: float,
    output_offset: float = 0.0,
) -> smoke._CaseEvidence:
    shape = (2, 3)
    static = {name: np.full(shape, shared_offset, dtype=np.float64) for name in smoke._SHARED_FIELD_NAMES}
    static.update({name: np.full(shape, shared_offset + output_offset, dtype=np.float64) for name in smoke._AIRFLOW_FIELD_NAMES})
    return smoke._CaseEvidence(
        record={
            "case_index": case_index,
            "simulation_case_id": f"simulation-{case_index}-{output_offset}",
            "input_files": [{"case_index": case_index}],
        },
        static=static,
        stationary_fixed={"T_flow_ref": 300.0},
        scalars={},
        schedule=None,
        global_values=None,
        initial_state={},
    )


def _batch(
    material_family: str,
    case_indices: tuple[int, ...],
    *,
    grid: dict[str, Any] | None = None,
) -> generation.cases.config.GenerationConfig:
    return cast(
        "generation.cases.config.GenerationConfig",
        SimpleNamespace(
            material_family=material_family,
            material_role="seen",
            case_indices=case_indices,
            scientific_values={"grid": grid} if grid is not None else {},
        ),
    )


def test_sentinel_workload_is_independent_of_production_case_counts() -> None:
    """Keep bounded validation mechanics separate from campaign allocation."""
    campaign = generation.cases.config.load_campaign_config(
        Path("configs/generation/campaigns/steady_flow/family_generalization.yaml"),
        require_executable=False,
    )
    baseline = generation.validation.sentinels.inspect_sentinel_workload(campaign)
    reduced_batches = tuple(replace(batch, case_indices=(1,)) for batch in campaign.batches)
    reduced = replace(
        campaign,
        total_case_count=len(reduced_batches),
        batches=reduced_batches,
    )

    assert generation.validation.sentinels.inspect_sentinel_workload(reduced) == baseline
    assert baseline["natural_materials"] == list(campaign.material_inventory)
    assert baseline["natural_case_count"] == (baseline["natural_cases_per_material"] * len(campaign.material_inventory))
    assert baseline["parameter_ood_case_count"] == sum(evidence["case_count"] for evidence in baseline["parameter_ood"].values())
    assert baseline["production_case_count_independent"] is True


def test_smoke_variation_accepts_configured_case_inventory() -> None:
    """Measure contrasts across every configured case, not a fixed pair."""
    report = smoke._variation_report(
        (
            _evidence(1, shared_offset=0.0),
            _evidence(2, shared_offset=1.0),
            _evidence(3, shared_offset=3.0),
        ),
        profile_id="steady_flow",
    )

    assert report["case_count"] == 3
    assert report["spatial_maximum_absolute_differences"]["Kxx"] == 3.0
    assert report["input_hash_sets_distinct"] is True


def test_smoke_pairing_derives_material_count_seed_and_grid() -> None:
    """Pair arbitrary configured material/count and report its resolved grid."""
    grid = {
        "nx": 3,
        "ny": 2,
        "Lx": 2.0,
        "Ly": 1.0,
        "Lz": 0.5,
        "boundaries_included": True,
        "dx": 1.0,
        "dy": 1.0,
    }
    indices = (1, 2, 3)
    steady_batch = _batch("configured_material", indices, grid=grid)
    transient_batch = _batch("configured_material", indices, grid=grid)
    steady_campaign = cast(
        "generation.cases.config.CampaignConfig",
        SimpleNamespace(
            batches=(steady_batch,),
            paired_equivalence_seed=42,
        ),
    )
    transient_campaign = cast(
        "generation.cases.config.CampaignConfig",
        SimpleNamespace(
            batches=(transient_batch,),
            paired_equivalence_seed=42,
        ),
    )

    assert smoke._paired_smoke_batches(
        steady_campaign,
        transient_campaign,
    ) == (steady_batch, transient_batch)

    steady_cases = tuple(_evidence(index, shared_offset=float(index)) for index in indices)
    transient_cases = tuple(
        _evidence(
            index,
            shared_offset=float(index),
            output_offset=0.25,
        )
        for index in indices
    )
    report = smoke._equivalence_report(
        steady_cases,
        transient_cases,
        steady_config=steady_batch,
        transient_config=transient_batch,
    )

    assert report["pair_count"] == 3
    assert report["grid_shape"] == [2, 3]
    assert report["grid_spacing_by_axis_m"] == {"x": 1.0, "y": 1.0}
    assert report["grid_extent_m"] == {"x": 2.0, "y": 1.0, "z": 0.5}

    mismatched_campaign = cast(
        "generation.cases.config.CampaignConfig",
        SimpleNamespace(
            batches=(_batch("different_material", indices),),
            paired_equivalence_seed=42,
        ),
    )
    with pytest.raises(ValueError, match="pair one material"):
        smoke._paired_smoke_batches(
            steady_campaign,
            mismatched_campaign,
        )
