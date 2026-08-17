# ruff: noqa: S101
"""Config-owned COMSOL template locator and byte-identity contracts."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest
import yaml

from src import common, generation
from src.generation.contracts import generation_contracts_templates as templates

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any


def _write_template(
    repository_root: Path,
    relative_path: str,
    *,
    sidecar: str | None = None,
) -> tuple[Path, str]:
    """Create one test-owned template and adjacent digest sidecar."""
    template_path = repository_root / relative_path
    template_path.parent.mkdir(parents=True, exist_ok=True)
    template_path.write_bytes(b"synthetic COMSOL template bytes\n")
    digest = hashlib.sha256(template_path.read_bytes()).hexdigest()
    template_path.with_suffix(".sha256").write_text(
        f"{digest}\n" if sidecar is None else sidecar,
        encoding="utf-8",
    )
    return template_path, digest


def test_digest_only_sidecar_is_derived_from_configured_template(tmp_path: Path) -> None:
    """Accept one digest-only sidecar mechanically adjacent to a fake MPH file."""
    template_path, digest = _write_template(tmp_path, "renamed/template_variant.mph")

    resolved = templates.resolve_template_identity(
        "renamed/template_variant.mph",
        repository_root=tmp_path,
    )

    assert resolved.absolute_path == template_path
    assert resolved.sidecar_path == template_path.with_suffix(".sha256")
    assert resolved.sha256 == digest


@pytest.mark.parametrize(
    ("relative_path", "sidecar", "expected_exception"),
    [
        ("missing/template.mph", None, FileNotFoundError),
        ("templates/no_sidecar.mph", "delete", FileNotFoundError),
        ("templates/malformed.mph", "not-a-digest\n", ValueError),
        ("templates/mismatch.mph", "0" * 64 + "\n", ValueError),
    ],
)
def test_template_identity_rejects_missing_or_invalid_bytes(
    tmp_path: Path,
    relative_path: str,
    sidecar: str | None,
    expected_exception: type[Exception],
) -> None:
    """Reject missing templates and sidecars plus malformed or stale digests."""
    if relative_path != "missing/template.mph":
        template_path, _digest = _write_template(tmp_path, relative_path, sidecar=sidecar)
        if sidecar == "delete":
            template_path.with_suffix(".sha256").unlink()

    with pytest.raises(expected_exception):
        templates.resolve_template_identity(relative_path, repository_root=tmp_path)


def test_template_identity_rejects_unsafe_configured_paths(tmp_path: Path) -> None:
    """Reject absolute, traversing, and non-MPH configured paths before lookup."""
    for configured_path in (
        "/absolute/template.mph",
        "../escape.mph",
        "templates/not_a_template.txt",
    ):
        with pytest.raises(ValueError, match=r"repository-relative|normalized"):
            templates.resolve_template_identity(configured_path, repository_root=tmp_path)


def test_template_identity_rejects_symlink_escape(tmp_path: Path) -> None:
    """Reject a configured template symlink whose target escapes the repository."""
    external_template = tmp_path.parent / "external.mph"
    external_template.write_bytes(b"outside repository\n")
    configured_template = tmp_path / "templates/escape.mph"
    configured_template.parent.mkdir(parents=True)
    configured_template.symlink_to(external_template)
    configured_template.with_suffix(".sha256").write_text(
        f"{hashlib.sha256(external_template.read_bytes()).hexdigest()}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="escapes the repository"):
        templates.resolve_template_identity("templates/escape.mph", repository_root=tmp_path)


def test_template_locator_is_provenance_while_bytes_bind_simulation_identity(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """Keep path relocation semantic-neutral and template bytes simulation-bound."""
    config_path, original_template = generation_config_factory(simulation_profile="steady_flow")
    original_campaign = generation.cases.config.load_campaign_config(config_path)
    original = original_campaign.batches[0]
    original_bundle = generation.cases.case.generate_case_input_bundle(
        original,
        1,
        tmp_path / "original-inputs",
    )

    project_root = original_template.parents[1]
    relocated_template = project_root / "relocated/reference_template.mph"
    relocated_template.parent.mkdir(parents=True)
    relocated_template.write_bytes(original_template.read_bytes())
    relocated_template.with_suffix(".sha256").write_text(
        f"{common.serialization.file_sha256(relocated_template)}\n",
        encoding="utf-8",
    )
    profile_path = config_path.parent / "profile.yaml"
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    profile["template"] = relocated_template.relative_to(project_root).as_posix()
    profile_path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")

    relocated_campaign = generation.cases.config.load_campaign_config(config_path)
    relocated = relocated_campaign.batches[0]
    relocated_bundle = generation.cases.case.generate_case_input_bundle(
        relocated,
        1,
        tmp_path / "relocated-inputs",
    )
    assert relocated.template_relative_path != original.template_relative_path
    assert relocated.template_sha256 == original.template_sha256
    assert relocated.scientific_config_digest == original.scientific_config_digest
    assert relocated.batch_id == original.batch_id
    assert relocated_campaign.campaign_digest == original_campaign.campaign_digest
    assert relocated_bundle.case_input_id == original_bundle.case_input_id
    assert relocated_bundle.simulation_case_id == original_bundle.simulation_case_id

    relocated_template.write_bytes(b"changed synthetic COMSOL template bytes\n")
    relocated_template.with_suffix(".sha256").write_text(
        f"{common.serialization.file_sha256(relocated_template)}\n",
        encoding="utf-8",
    )
    changed_campaign = generation.cases.config.load_campaign_config(config_path)
    changed = changed_campaign.batches[0]
    changed_bundle = generation.cases.case.generate_case_input_bundle(
        changed,
        1,
        tmp_path / "changed-inputs",
    )
    assert changed.scientific_config_digest != relocated.scientific_config_digest
    assert changed.batch_id != relocated.batch_id
    assert changed_campaign.campaign_digest != relocated_campaign.campaign_digest
    assert changed.case_input_config_digest == relocated.case_input_config_digest
    assert changed_bundle.case_input_id == relocated_bundle.case_input_id
    assert changed_bundle.simulation_case_id != relocated_bundle.simulation_case_id
