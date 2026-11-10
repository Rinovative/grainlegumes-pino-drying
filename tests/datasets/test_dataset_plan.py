"""Protect the ordinary Dataset plan and CLI handoffs.

Responsibilities:
  - Preserve the authored plan, its derived designs and domain, and the role boundary between Sources and Datasets
  - Check the computed Dataset name, the power-of-two counts, the role and key schema, the Dataset-only revision and the
    explicit new Source that reuses its label
  - Check that a design point keeps its seeds and coordinates in every plan that requests it
  - Check that both tasks move every parameter-OOD point outside natural support, over the groups their Dataset view declares
  - Verify preview, staged continuation, publication, PT, and the public CLI allocation boundary
  - Verify that a login-node run freezes the source and submits its own driver job, which refuses live code inside an allocation
  - Verify that dirty sources are always admitted
  - Verify that the driver ends as a plain summary or one plain failure report, never a JSON dump or traceback,
    while preview and launcher output stay JSON
  - Verify that the driver links its own log into the Dataset folder only after publication

Design principles:
  - Exercise maintained plan and partition services with a tiny test-owned plan, never the maintained Dataset plan YAMLs
  - Replace only Generation, Dataset publication, and PT stage boundaries

This module does NOT:
  - Admit Sources, run COMSOL, publish scientific data, or create Slurm jobs
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from src import common, datasets, generation
from src.datasets import dataset_packages as cli
from src.datasets import dataset_plan
from src.datasets.packages import dataset_packages_builder as builder
from src.datasets.packages import dataset_packages_manifest as manifests
from src.generation import generation_controller
from src.generation.runtime import generation_runtime_host as host

if TYPE_CHECKING:
    from collections.abc import Callable

    from support.source_tree import FrozenProject


_PHYSICAL_MODEL = "configs/01_generation/physical_model"


def _base_plan(task: str = "transient_drying") -> dict[str, Any]:
    """Return a fresh test-owned Dataset plan; it only references the maintained Generation configs the loader reads."""
    return {
        "schema_kind": "dataset_plan",
        "task": task,
        "materials": {
            "lentil": {"role": "id", "train": 1, "validation": 1, "test": 1},
            "chickpea": {"role": "id", "train": 1, "validation": 1, "test": 1},
        },
        "generation": {
            "seed_base": 41,
            "minimum_success_rate": 0.8,
            "sources_config": f"{_PHYSICAL_MODEL}/sources.yaml",
            "registry_config": f"{_PHYSICAL_MODEL}/registry.yaml",
            "common_config": f"{_PHYSICAL_MODEL}/common.yaml",
            "operations_config": f"{_PHYSICAL_MODEL}/fixed_bed.yaml",
            "bed_fields_config": f"{_PHYSICAL_MODEL}/bed_fields.yaml",
            "profile_config": f"{_PHYSICAL_MODEL}/profiles/{task}.yaml",
            "execution_config": "configs/01_generation/execution.yaml",
        },
    }


def _write_plan(tmp_path: Path, mutate: Callable[[dict[str, Any]], object] | None = None, *, task: str = "transient_drying") -> Path:
    """Write the test-owned base plan, optionally mutated, and return its path."""
    plan = _base_plan(task)
    if mutate is not None:
        mutate(plan)
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def plan_path(tmp_path: Path) -> Path:
    """Return the unmodified test-owned transient plan file."""
    return _write_plan(tmp_path)


def _read_preview(capsys: pytest.CaptureFixture[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split preview stdout into its first-line footprint object and the effective-plan JSON that follows."""
    first, _, rest = capsys.readouterr().out.partition("\n")
    return json.loads(first), json.loads(rest)


def _small_plan(path: Path, *, duration_hours: float | None = None) -> dict[str, Any]:
    return dataset_plan.load_dataset_plan(path, duration_hours=duration_hours)


def _pood_and_ood(authored: dict[str, Any]) -> None:
    """Give lentil a parameter-OOD design and hold chickpea out as material OOD."""
    authored["materials"] = {
        "lentil": {"role": "id", "train": 4, "validation": 2, "test": 2, "parameter_ood": 1},
        "chickpea": {"role": "ood", "test": 8},
    }


def test_loader_derives_designs_and_domain_and_preserves_the_authored_plan(plan_path: Path, tmp_path: Path) -> None:
    """The plan yields its design counts per kind and its domain; loading never rewrites the authored file."""
    original_bytes = plan_path.read_bytes()
    plan = dataset_plan.load_dataset_plan(plan_path)
    assert plan["designs"] == {"lentil": {"train": 1, "validation": 1, "test": 1}, "chickpea": {"train": 1, "validation": 1, "test": 1}}
    assert plan["domain_profile"] == "reference_range"
    assert plan_path.read_bytes() == original_bytes
    extended = dataset_plan.load_dataset_plan(_write_plan(tmp_path, _pood_and_ood))
    assert extended["designs"] == {"lentil": {"train": 4, "validation": 2, "test": 2, "parameter_ood": 1}, "chickpea": {"test": 8}}
    assert extended["domain_profile"] == "reference_and_parameter_extension"


def test_dataset_name_is_computed_from_design_counts_domain_and_seed(tmp_path: Path) -> None:
    """The name states the resolved content: design counts of ID materials, held-out materials, domain and Generation seed."""

    def mutate(authored: dict[str, Any]) -> None:
        _pood_and_ood(authored)
        authored["generation"]["seed_base"] = 123

    path = _write_plan(tmp_path, mutate)

    assert dataset_plan.load_dataset_plan(path)["name"] == "lentil-train4-val2-test2-pood1+chickpea-ood8__ref-param-ext__sobol-seed-123"
    # Only a horizon that deviates from the configured one is named, so an override equal to it adds no token.
    common_config = yaml.safe_load((common.paths.get_project_root() / f"{_PHYSICAL_MODEL}/common.yaml").read_text(encoding="utf-8"))
    configured_hours = generation.cases.config.physical_horizon_hours(common_config["time"])
    assert dataset_plan.load_dataset_plan(path, duration_hours=configured_hours)["name"] == dataset_plan.load_dataset_plan(path)["name"]
    assert dataset_plan.load_dataset_plan(path, duration_hours=4)["name"].endswith("__sobol-seed-123__h-4")


@pytest.mark.parametrize("count", [3, 0, -2, True, 2.0, "4", None])
def test_every_design_count_must_be_a_power_of_two(tmp_path: Path, count: object) -> None:
    """Counts 1, 2, 4, ... keep each Sobol prefix balanced; any other count, a bool or a non-integer is refused."""
    path = _write_plan(tmp_path, lambda authored: authored["materials"]["lentil"].update(validation=count))
    with pytest.raises(ValueError, match=r"materials\.lentil\.validation must be an integer power of two"):
        dataset_plan.load_dataset_plan(path)


@pytest.mark.parametrize(
    ("specification", "message"),
    [
        ({"role": "id", "train": 1, "test": 1}, "requires"),
        ({"role": "id", "train": 1, "validation": 1, "test": 1, "count": 1}, "admits"),
        ({"role": "ood", "train": 1, "test": 1}, "admits"),
        ({"role": "ood", "test": 1, "parameter_ood": 1}, "admits"),
        ({"train": 1, "validation": 1, "test": 1}, "role must be id or ood"),
    ],
)
def test_material_roles_admit_only_their_design_kinds(tmp_path: Path, specification: dict[str, Any], message: str) -> None:
    """An ID material needs Train, validation and test and may add parameter OOD; an OOD material takes only its test design."""
    path = _write_plan(tmp_path, lambda authored: authored["materials"].update(lentil=specification))
    with pytest.raises(ValueError, match=message):
        dataset_plan.load_dataset_plan(path)


def test_minimum_success_rate_is_authored_policy_that_changes_neither_identity_nor_the_execution_overrides(tmp_path: Path) -> None:
    """The rate changes neither name nor seeds; only an explicit cap is an override, the campaign derives the default from the rate."""

    def with_rate(rate: float) -> dict[str, Any]:
        return dataset_plan.load_dataset_plan(_write_plan(tmp_path, lambda authored: authored["generation"].update(minimum_success_rate=rate)))

    strict, tolerant = with_rate(1.0), with_rate(0.6)
    assert strict["name"] == tolerant["name"]
    assert dataset_plan.effective_execution_overrides() == {}
    assert dataset_plan.effective_execution_overrides({"max_admission_cases": 5}) == {"max_admission_cases": 5}
    with pytest.raises(ValueError, match="positive integer"):
        dataset_plan.effective_execution_overrides({"max_admission_cases": None})
    campaigns = [dataset_plan.authored_campaign(plan) for plan in (strict, tolerant)]
    assert [campaign["minimum_success_rate"] for campaign in campaigns] == [1.0, 0.6]
    assert campaigns[0]["sampling"] == campaigns[1]["sampling"]
    preview = dataset_plan.preview_dataset_plan(tolerant)
    assert (preview["minimum_success_rate"], preview["execution_overrides"]) == (0.6, {})


@pytest.mark.parametrize("value", [None, 0, 1.5, True, "0.9", {"lentil": 0.9}])
def test_minimum_success_rate_is_required_and_rejects_unsupported_values(tmp_path: Path, value: object) -> None:
    """Require one rate 0 < s <= 1 per plan and reject a missing, boolean, textual or per-material value before any Generation work."""

    def mutate(authored: dict[str, Any]) -> None:
        authored["generation"].pop("minimum_success_rate")
        if value is not None:
            authored["generation"]["minimum_success_rate"] = value

    with pytest.raises(ValueError, match="minimum_success_rate"):
        dataset_plan.load_dataset_plan(_write_plan(tmp_path, mutate))


def test_revision_names_only_the_dataset_and_reuses_the_default_source(plan_path: Path) -> None:
    """A Dataset revision keeps a second Dataset of the same plan; the physical Source keeps its default name and is reused."""
    base = dataset_plan.load_dataset_plan(plan_path)
    revised = dataset_plan.load_dataset_plan(plan_path, revision="converter-fix")
    source = dataset_plan.derive_source_name(base)

    assert revised["name"] == f"{base['name']}__rev-converter-fix"
    assert dataset_plan.derive_source_name(revised) == source
    assert "revision" not in revised["generation"]
    assert "revision" not in dataset_plan.authored_campaign(revised)
    assert dataset_plan.authored_campaign(revised) == dataset_plan.authored_campaign(base)
    preview = dataset_plan.preview_dataset_plan(revised)
    assert preview["source_name"] == source
    assert "new_source" not in preview


def test_new_source_names_the_source_under_the_dataset_revision_label(plan_path: Path) -> None:
    """``new_source`` carries the revision into the authored campaign, so the Source is the default name plus ``__rev-LABEL``."""
    base = dataset_plan.load_dataset_plan(plan_path)
    fresh = dataset_plan.load_dataset_plan(plan_path, revision="producer-fix", new_source=True)
    source = dataset_plan.derive_source_name(base)

    assert fresh["name"] == f"{base['name']}__rev-producer-fix"
    assert fresh["generation"]["revision"] == "producer-fix"
    assert dataset_plan.derive_source_name(fresh) == f"{source}__rev-producer-fix"
    assert dataset_plan.authored_campaign(fresh)["revision"] == "producer-fix"
    assert "source_name" not in dataset_plan.authored_campaign(fresh)
    preview = dataset_plan.preview_dataset_plan(fresh)
    assert preview["source_name"] == f"{source}__rev-producer-fix"
    assert preview["new_source"].startswith(f"Source {source}__rev-producer-fix will be simulated afresh")
    with pytest.raises(ValueError, match="Revision"):
        dataset_plan.load_dataset_plan(plan_path, new_source=True)


def test_source_name_is_the_generation_default_and_a_new_source_changes_only_the_source_locator(plan_path: Path) -> None:
    """The derived Source name is what Generation resolves; a new Source keeps seeds, batch identities and the campaign digest."""
    plain = dataset_plan.load_dataset_plan(plan_path)
    dataset_only = dataset_plan.load_dataset_plan(plan_path, revision="converter-fix")
    fresh = dataset_plan.load_dataset_plan(plan_path, revision="converter-fix", new_source=True)

    def resolve(plan: dict[str, Any]) -> Any:
        return generation.cases.config.load_campaign_config(plan_path, require_executable=False, authored_value=dataset_plan.authored_campaign(plan))

    base_campaign, dataset_only_campaign, revised_campaign = resolve(plain), resolve(dataset_only), resolve(fresh)

    assert base_campaign.source_name == base_campaign.campaign_name == dataset_plan.derive_source_name(plain)
    assert dataset_only_campaign.source_name == base_campaign.source_name
    assert revised_campaign.source_name == dataset_plan.derive_source_name(fresh)
    assert revised_campaign.source_name == f"{base_campaign.source_name}__rev-converter-fix"
    assert revised_campaign.campaign_name == base_campaign.campaign_name
    assert revised_campaign.campaign_digest == base_campaign.campaign_digest
    assert [(batch.batch_id, batch.seed_base, batch.case_indices, batch.assignments) for batch in revised_campaign.batches] == [
        (batch.batch_id, batch.seed_base, batch.case_indices, batch.assignments) for batch in base_campaign.batches
    ]
    assert all(
        revised_batch.batch_storage_name == base_batch.batch_storage_name.replace(base_campaign.source_name, revised_campaign.source_name)
        for base_batch, revised_batch in zip(base_campaign.batches, revised_campaign.batches, strict=True)
    )
    assert revised_campaign.batches[0].batch_storage_name != base_campaign.batches[0].batch_storage_name


@pytest.mark.parametrize("label", ["fix", "converter-fix", "v2_a+b", "Upper", "dot.v2", "-lead", "trail-", "a--b", "nested/label"])
def test_revision_label_follows_one_grammar_for_dataset_and_source(label: str, plan_path: Path) -> None:
    """Dataset and Source names accept exactly the labels of the Generation revision grammar, refused at load time before any source is frozen."""

    def accepted(build: Callable[[], object]) -> bool:
        try:
            build()
        except ValueError:
            return False
        return True

    plan = _small_plan(plan_path)
    expected = accepted(lambda: generation.contracts.campaign_identity.revision_label(label))

    assert accepted(lambda: dataset_plan.derive_dataset_name(plan, revision=label)) is expected
    assert accepted(lambda: dataset_plan.load_dataset_plan(plan_path, revision=label)) is expected
    assert accepted(lambda: dataset_plan.load_dataset_plan(plan_path, revision=label, new_source=True)) is expected


def test_plan_schema_rejects_removed_unknown_and_missing_keys_and_other_versions(tmp_path: Path) -> None:
    """The plan computes its name, domain and roles; removed, unknown or missing keys are refused."""
    for extra in ({"name": "authored-name"}, {"split": {"seed": 7}}):
        with pytest.raises(ValueError, match="exactly the fields"):
            dataset_plan.load_dataset_plan(_write_plan(tmp_path, lambda authored, extra=extra: authored.update(extra)))
    # The Source name and the domain are derived; only --revision with --new-source varies the Source name, never the YAML.
    for key in ("sampling", "domain_profile", "source_name", "revision"):
        with pytest.raises(ValueError, match="generation must use exactly the fields"):
            dataset_plan.load_dataset_plan(_write_plan(tmp_path, lambda authored, key=key: authored["generation"].update({key: "authored"})))
    with pytest.raises(ValueError, match="generation must use exactly the fields"):
        dataset_plan.load_dataset_plan(_write_plan(tmp_path, lambda authored: authored["generation"].pop("seed_base")))


def test_generation_projection_contains_seed_and_designs_but_no_dataset_roles(plan_path: Path) -> None:
    """Keep material roles and case roles in Dataset identity, not Source design."""
    plan = _small_plan(plan_path)
    campaign = dataset_plan.authored_campaign(plan)

    assert campaign["sampling"] == {"seed_base": plan["generation"]["seed_base"], "designs": plan["designs"]}
    assert set(campaign["materials"]) == {"lentil", "chickpea"}
    assert not {"name", "dataset_name", "selection", "roles", "split", "domain_profile"} & campaign.keys()


@pytest.mark.parametrize("task", ["steady_flow", "transient_drying"])
def test_both_tasks_move_each_parameter_ood_point_outside_natural_support(tmp_path: Path, task: str) -> None:
    """A parameter-OOD design resolves for both tasks over the groups the task's Dataset view declares; each point moves its allocated unit."""
    path = _write_plan(tmp_path, lambda authored: authored["materials"]["lentil"].update(parameter_ood=2), task=task)
    plan = dataset_plan.load_dataset_plan(path)
    campaign = generation.cases.config.load_campaign_config(path, require_executable=False, authored_value=dataset_plan.authored_campaign(plan))
    (batch,) = (batch for batch in campaign.batches if batch.sampling_regime == "parameter_ood")
    assert (batch.material_family, batch.scientific_values["natural_support_state"]) == ("lentil", "parameter_ood")
    declared = datasets.contracts.views.get_view(task).parameter_ood_blocks
    assert declared
    assert tuple(batch.scientific_values["parameter_ood"]["groups"]) == declared
    allocation = batch.scientific_values["parameter_ood"]["case_allocation"]
    assert [entry["case_index"] for entry in allocation] == list(batch.case_indices)
    for entry in allocation:
        provenance = generation.cases.sampling.sample_case(batch, entry["case_index"]).ood_provenance
        assert (provenance["natural_support_state"], provenance["active_unit_id"]) == ("parameter_ood", entry["unit_id"])


def test_a_design_point_keeps_its_seeds_and_coordinates_in_every_plan(tmp_path: Path) -> None:
    """Point i of a design is the same case in a small and a large plan, whatever the other materials and design kinds request."""
    small = dataset_plan.load_dataset_plan(_write_plan(tmp_path, lambda authored: authored["materials"].pop("chickpea")))
    large_path = tmp_path / "large.yaml"
    large_path.write_bytes(_write_plan(tmp_path, _pood_and_ood).read_bytes())
    large = dataset_plan.load_dataset_plan(large_path)

    def batches(plan: dict[str, Any], path: Path) -> dict[tuple[str, str], Any]:
        campaign = generation.cases.config.load_campaign_config(path, require_executable=False, authored_value=dataset_plan.authored_campaign(plan))
        return {(batch.material_family, batch.sampling_regime): batch for batch in campaign.batches}

    first, second = batches(small, tmp_path / "plan.yaml"), batches(large, large_path)
    for kind in ("train", "validation", "test"):
        one, four = first[("lentil", kind)], second[("lentil", kind)]
        design = one.scientific_values["sampling"]
        assert design["design_seed"] == four.scientific_values["sampling"]["design_seed"]
        assert one.case_seed(one.case_indices[0]) == four.case_seed(four.case_indices[0])
        prefix = generation.cases.sampling.design_points(seed=design["design_seed"], dimensions=design["dimensions"], count=len(four.case_indices))
        assert (generation.cases.sampling.design_points(seed=design["design_seed"], dimensions=design["dimensions"], count=1) == prefix[:1]).all()


def test_preview_does_not_construct_generation_controller(tmp_path: Path, plan_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep plan preview read-only even when no scientific storage exists."""
    storage = tmp_path / "not-created"
    monkeypatch.setenv("STORAGE_ROOT", str(storage))
    monkeypatch.setattr(dataset_plan, "_controller", lambda: pytest.fail("preview constructed Generation"))

    summary = dataset_plan.preview_dataset_plan(
        _small_plan(plan_path, duration_hours=4),
        execution_overrides={"node_jobs": 1, "max_admission_cases": 6, "cores_per_case": 8},
    )

    assert summary["directory"] == f"02_datasets/transient_drying/{summary['dataset']}"
    assert "--revision" in summary["collision"]
    assert summary["actual_outcomes"] == "not measured by preview"
    assert summary["physical_duration_hours_override"] == 4
    assert summary["execution_overrides"]["max_admission_cases"] == 6
    assert not storage.exists()


def test_run_without_slurm_fails_before_generation_controller(plan_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject execution before constructing the Generation runtime outside Slurm."""
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(dataset_plan, "_controller", lambda: pytest.fail("controller constructed before Slurm guard"))

    with pytest.raises(RuntimeError, match="CPU Slurm allocation"):
        dataset_plan.run_dataset_plan(plan_path, _small_plan(plan_path), invocation=("python", "run", "plan"))


_TREE_ID = "a" * 12 + "-" + "b" * 12
_SUBMITTED = host.SubmittedLauncher("4711", _TREE_ID, Path("/runtime/logs/datasets/dataset-transient_drying-4711.log"))
_DRIVER: dict[str, Any] = {"driver_partition": "long", "driver_time": "12:00:00"}


@pytest.mark.parametrize("continue_with_current_code", [False, True])
def test_live_run_submits_its_driver_from_the_campaign_tree_before_any_generation(
    plan_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    continue_with_current_code: bool,
) -> None:
    """A run started on the login node resolves its campaign, then submits its driver from that campaign's tree without running Generation."""
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    invocation = ("python", "-m", "src.datasets.dataset_packages", "run", "plan.yaml", "--continue-with-current-code")
    submitted: list[dict[str, Any]] = []
    campaign_plan = SimpleNamespace(kind="campaign", identity="source~transient_drying~lentil-chickpea__abc")
    plan = _small_plan(plan_path)

    class Controller:
        def authored_campaign_plan(self, _path: str, campaign: dict[str, Any], **kwargs: Any) -> SimpleNamespace:
            assert campaign == dataset_plan.authored_campaign(plan)
            # Without an explicit cap the campaign derives its cap from the plan rate, so no override carries one.
            assert kwargs == {"scientific_overrides": None, "execution_overrides": {}}
            return campaign_plan

        def submit_run_tree(self, plan: SimpleNamespace, **kwargs: Any) -> host.SubmittedLauncher:
            submitted.append({"plan": plan, **kwargs})
            return _SUBMITTED

        def run_authored_campaign(self, *_args: Any, **_kwargs: Any) -> None:
            pytest.fail("live code ran Generation instead of submitting the driver")

    monkeypatch.setattr(dataset_plan, "_controller", Controller)

    result = dataset_plan.submit_dataset_plan(
        plan_path, plan, invocation=invocation, continue_with_current_code=continue_with_current_code, **_DRIVER
    )

    assert result == {"status": "SUBMITTED", "job_id": "4711", "tree_id": _TREE_ID, "log_path": str(_SUBMITTED.log_path)}
    assert submitted == [
        {
            "plan": campaign_plan,
            "module": "src.datasets.dataset_packages",
            "arguments": list(invocation[3:]),
            "continue_with_current_code": continue_with_current_code,
            "job": host.LauncherJob(name="dataset-transient_drying", log_area="datasets", partition="long", wall_time="12:00:00"),
        }
    ]


def test_submission_rejects_unsupported_driver_options_before_admitting_the_source(plan_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The driver job is validated first, so a mistyped partition freezes nothing."""
    monkeypatch.setattr(dataset_plan, "_controller", lambda: pytest.fail("the source was admitted with an invalid driver job"))

    with pytest.raises(ValueError, match="partition must be one of"):
        dataset_plan.submit_dataset_plan(
            plan_path, _small_plan(plan_path), invocation=("python", "-m", "m", "run"), **{**_DRIVER, "driver_partition": "rsm"}
        )


def test_live_run_inside_an_allocation_is_refused_before_any_git_command(
    plan_path: Path, frozen_project: FrozenProject, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Freezing happens only on the login node: a live run inside an allocation fails on admission, before Git or any submission."""
    plan = _small_plan(plan_path)
    failing_git, git_log = frozen_project.failing_git()
    for key, value in {
        "SLURM_JOB_ID": "4242",
        "PATH": f"{failing_git}{os.pathsep}{os.environ['PATH']}",
        "STORAGE_ROOT": str(frozen_project.storage),
        "RUNTIME_ROOT": str(frozen_project.runtime),
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("SOURCE_TREE_ID", raising=False)
    monkeypatch.setattr(common.paths, "get_project_root", lambda: frozen_project.repository)

    with pytest.raises(
        RuntimeError, match=re.escape("Freeze the source on the login node: start this command there; it submits its own Slurm work.")
    ):
        dataset_plan.submit_dataset_plan(plan_path, plan, invocation=("python", "-m", "m", "run"), **_DRIVER)

    assert not git_log.exists()
    assert not frozen_project.snapshots.exists()


def test_staged_generation_publishes_neither_dataset_nor_pt_and_retains_invocation(plan_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep incomplete candidate admission resumable without partial Dataset publication."""
    monkeypatch.setenv("SLURM_JOB_ID", "test-allocation")
    invocation = (
        "python",
        "-m",
        "src.datasets.dataset_packages",
        "run",
        "plan.yaml",
        "--duration-hours",
        "4",
        "--node-jobs",
        "1",
        "--max-admission-cases",
        "6",
        "--cores-per-case",
        "8",
    )
    generation_calls: list[dict[str, Any]] = []

    class Controller:
        def run_authored_campaign(self, path: str, campaign: dict[str, Any], **kwargs: Any) -> SimpleNamespace:
            generation_calls.append({"path": path, "campaign": campaign, **kwargs})
            return SimpleNamespace(result="STAGED", run_id="source~transient_drying~lentil-chickpea__abc")

    monkeypatch.setattr(dataset_plan, "_controller", Controller)
    monkeypatch.setattr(builder, "build_dataset_selection", lambda **_kwargs: pytest.fail("STAGED run published a Dataset"))
    execution = {"node_jobs": 1, "max_admission_cases": 6, "cores_per_case": 8}

    result = dataset_plan.run_dataset_plan(
        plan_path,
        _small_plan(plan_path, duration_hours=4),
        invocation=invocation,
        execution_overrides=execution,
    )

    assert result["status"] == "STAGED"
    assert result["dataset"] is None
    assert result["next"] == shlex.join(invocation)
    assert len(generation_calls) == 1
    assert generation_calls[0]["scientific_overrides"] == {"duration_hours": 4}
    assert generation_calls[0]["execution_overrides"] == execution
    # Admission is always dirty-capable: the driver forwards no snapshot or strictness switches; the explicit cap is an override.
    assert set(generation_calls[0]) == {"path", "campaign", "scientific_overrides", "execution_overrides"}


def test_complete_run_forwards_exact_selection_material_roles_and_invocation(plan_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bind the one publication to the plan's design counts, material roles, and invocation."""
    monkeypatch.setenv("SLURM_JOB_ID", "test-allocation")
    plan = _small_plan(plan_path, duration_hours=4)
    invocation = ("python", "-m", "src.datasets.dataset_packages", "run", "plan.yaml", "--max-admission-cases", "6")
    execution = {"node_jobs": 1, "max_admission_cases": 6, "cores_per_case": 8}
    generation_calls: list[dict[str, Any]] = []
    publication_calls: list[dict[str, Any]] = []

    class Controller:
        def run_authored_campaign(self, path: str, campaign: dict[str, Any], **kwargs: Any) -> SimpleNamespace:
            generation_calls.append({"path": path, "campaign": campaign, **kwargs})
            return SimpleNamespace(result="OK", run_id="source-run", source_name="lentil-chickpea__abc")

    def publish(request: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        publication_calls.append({"request": request, **kwargs})
        return {"dataset_id": "dataset-id", "manifest_path": "manifest.json"}

    monkeypatch.setattr(dataset_plan, "_controller", Controller)
    monkeypatch.setattr(builder, "build_dataset_selection", publish)

    result = dataset_plan.run_dataset_plan(
        plan_path,
        plan,
        invocation=invocation,
        execution_overrides=execution,
    )

    assert result["status"] == "OK"
    assert result["dataset"]["dataset_id"] == "dataset-id"
    assert generation_calls[0]["scientific_overrides"] == {"duration_hours": 4}
    assert generation_calls[0]["execution_overrides"] == execution
    assert publication_calls[0]["request"]["selection"] == {
        "kind": "design_order",
        "source_names": ["lentil-chickpea__abc"],
        "target_counts": plan["designs"],
    }
    assert publication_calls[0]["request"]["source_cases"] == []
    assert publication_calls[0]["material_roles"] == {"lentil": "id", "chickpea": "id"}
    assert publication_calls[0]["plan_invocation"] == list(invocation)
    assert len(publication_calls) == 1


def test_cli_preview_and_run_preserve_readonly_and_allocation_boundaries(
    tmp_path: Path,
    plan_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exercise the documented CLI without admitting any scientific stage: the login node submits, the tree driver needs its allocation."""
    storage = tmp_path / "not-created"
    monkeypatch.setenv("STORAGE_ROOT", str(storage))
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(dataset_plan, "_controller", lambda: pytest.fail("CLI admitted Generation"))
    monkeypatch.setattr(builder, "build_dataset_selection", lambda **_kwargs: pytest.fail("CLI published Dataset"))
    arguments = [
        "dataset_packages",
        "run",
        str(plan_path),
        "--revision",
        "smoke",
        "--duration-hours",
        "4",
        "--node-jobs",
        "1",
        "--max-admission-cases",
        "6",
        "--cores-per-case",
        "8",
    ]
    monkeypatch.setattr(sys, "argv", [*arguments, "--dry-run"])
    assert cli.main() == 0
    footprint, preview = _read_preview(capsys)
    assert footprint["identity"] == f"source~transient_drying~{dataset_plan.derive_source_name(_small_plan(plan_path))}__h-4"
    scheduler = footprint["scheduler_footprint"]
    # The override requests one node job; the other node-job resources come from the configured execution values.
    assert scheduler["node_jobs"] == 1
    assert re.fullmatch(r"[1-9][0-9]*G", scheduler["node_job_memory"])
    assert re.fullmatch(r"[0-9]{2}:[0-5][0-9]:[0-5][0-9]", scheduler["node_job_wall_time"])
    assert scheduler["cores_per_node_job"] >= 8
    assert (scheduler["driver_cpus"], scheduler["driver_time"]) == (2, "06:00:00")
    # The --duration-hours 4 horizon is part of the computed name, before the revision.
    assert preview["dataset"] == _small_plan(plan_path)["name"] + "__h-4__rev-smoke"
    authored = _base_plan()
    assert preview["materials"] == authored["materials"]
    assert preview["generation_seed_base"] == authored["generation"]["seed_base"]
    assert (preview["domain_profile"], preview["selection"], preview["partition"]) == ("reference_range", {"kind": "design_order"}, "design_kind")
    assert preview["physical_duration_hours_override"] == 4
    assert preview["execution_overrides"] == {"node_jobs": 1, "max_admission_cases": 6, "cores_per_case": 8}
    assert not storage.exists()
    submissions: list[dict[str, Any]] = []

    def submit(path: Path, plan: dict[str, Any], **kwargs: Any) -> dict[str, str]:
        submissions.append({"path": path, "dataset": plan["name"], "source": dataset_plan.derive_source_name(plan), **kwargs})
        return {"status": "SUBMITTED", "job_id": "4711"}

    monkeypatch.setattr(dataset_plan, "submit_dataset_plan", submit)
    monkeypatch.delenv("SOURCE_TREE_ID", raising=False)
    monkeypatch.setattr(sys, "argv", arguments)
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out) == {"status": "SUBMITTED", "job_id": "4711"}
    monkeypatch.setattr(sys, "argv", [*arguments, "--driver-partition", "long", "--driver-time", "12:00:00"])
    assert cli.main() == 0
    capsys.readouterr()
    defaults, explicit = submissions
    assert {key: defaults[key] for key in _DRIVER} == {"driver_partition": "standard", "driver_time": "06:00:00"}
    assert {key: explicit[key] for key in _DRIVER} == _DRIVER
    assert defaults["dataset"] == _small_plan(plan_path)["name"] + "__h-4__rev-smoke"
    assert defaults["source"] == dataset_plan.derive_source_name(_small_plan(plan_path)) + "__h-4"
    assert defaults["execution_overrides"] == {"node_jobs": 1, "max_admission_cases": 6, "cores_per_case": 8}
    assert not defaults["continue_with_current_code"]
    tree_plan = _small_plan(plan_path)
    monkeypatch.setattr(dataset_plan, "load_dataset_plan", lambda *_args, **_kwargs: tree_plan)
    monkeypatch.setattr(common.paths, "source_tree_id", lambda *_args, **_kwargs: _TREE_ID)
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(RuntimeError, match="CPU Slurm allocation"):
        cli.main()
    assert len(submissions) == 2
    assert not storage.exists()


def test_cli_new_source_needs_a_revision_and_names_the_source_under_it(
    tmp_path: Path,
    plan_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--new-source`` alone is a usage error; with ``--revision`` the preview shows the new Source, without ``--new-source`` the default one."""
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "not-created"))
    monkeypatch.setattr(dataset_plan, "_controller", lambda: pytest.fail("CLI admitted Generation"))
    arguments = [
        "dataset_packages",
        "run",
        str(plan_path),
        "--dry-run",
    ]
    default_source = dataset_plan.derive_source_name(_small_plan(plan_path))

    monkeypatch.setattr(sys, "argv", [*arguments, "--new-source"])
    with pytest.raises(SystemExit) as refusal:
        cli.main()
    assert refusal.value.code == 2
    assert "--new-source requires --revision" in capsys.readouterr().err

    monkeypatch.setattr(sys, "argv", [*arguments, "--revision", "fix", "--new-source"])
    assert cli.main() == 0
    _footprint, preview = _read_preview(capsys)
    assert preview["dataset"].endswith("__rev-fix")
    assert preview["source_name"] == f"{default_source}__rev-fix"
    assert "simulated afresh" in preview["new_source"]

    monkeypatch.setattr(sys, "argv", [*arguments, "--revision", "fix"])
    assert cli.main() == 0
    _footprint, preview = _read_preview(capsys)
    assert preview["dataset"].endswith("__rev-fix")
    assert preview["source_name"] == default_source
    assert "new_source" not in preview


def _as_driver(monkeypatch: pytest.MonkeyPatch, plan_path: Path) -> None:
    """Run the CLI as the driver job inside a source tree for the small test-owned plan."""
    tree_plan = _small_plan(plan_path)
    monkeypatch.setattr(dataset_plan, "load_dataset_plan", lambda *_args, **_kwargs: tree_plan)
    monkeypatch.setattr(common.paths, "source_tree_id", lambda *_args, **_kwargs: _TREE_ID)
    monkeypatch.setattr(sys, "argv", ["dataset_packages", "run", str(plan_path)])


def test_cli_driver_reports_an_expected_failure_as_one_plain_report_with_the_unchanged_exit_status(
    monkeypatch: pytest.MonkeyPatch, plan_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An expected controller failure ends as FAILED and Next without a traceback or stdout; an unexpected error still raises."""
    failure = generation_controller.GenerationControllerError("campaign failed", exit_code=2)
    failure.stage, failure.next_command = "campaign submission and reconciliation", "./scripts/generation status source~example"

    def run(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise failure

    _as_driver(monkeypatch, plan_path)
    monkeypatch.setattr(dataset_plan, "run_dataset_plan", run)
    assert cli.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "FAILED: campaign submission and reconciliation: campaign failed\nNext: ./scripts/generation status source~example\n"

    def crash(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        message = "unexpected"
        raise KeyError(message)

    monkeypatch.setattr(dataset_plan, "run_dataset_plan", crash)
    with pytest.raises(KeyError):
        cli.main()


def test_cli_reports_an_invalid_plan_as_one_plain_line(monkeypatch: pytest.MonkeyPatch, plan_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A plan the loader refuses is an expected error of the command, not a traceback."""

    def refuse(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        message = "unknown material"
        raise ValueError(message)

    monkeypatch.setattr(dataset_plan, "load_dataset_plan", refuse)
    monkeypatch.setattr(sys, "argv", ["dataset_packages", "run", str(plan_path)])
    assert cli.main() == 1
    assert capsys.readouterr().err == "FAILED: Dataset plan: unknown material\n"


def test_cli_driver_summarizes_the_published_dataset_instead_of_dumping_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, plan_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The finished driver states the Dataset path relative to the storage root, the cases per role, each failure with its replacement, the time."""
    storage = tmp_path / "storage"
    published = {
        "dataset_id": "example-dataset",
        "dataset_view": "transient_drying",
        "dependency_storage_root": str(storage),
        "manifest_path": str(storage / "02_datasets/transient_drying/example-dataset/dataset_manifest.json"),
        "source_case_count": 6,
        "sample_count": 618,
        "transition_count": 618,
    }
    failed = {"material_family": "lentil", "case_id": "case_0001", "origin": "primary", "parent_failed_case_id": None, "outcome": "failed"}
    failed_replacement = {**failed, "case_id": "case_0003", "origin": "replacement", "parent_failed_case_id": "case_0001"}
    replacement = {
        "material_family": "lentil",
        "case_id": "case_0004",
        "origin": "replacement",
        "parent_failed_case_id": "case_0001",
        "outcome": "successful",
    }
    manifest = {
        "case_partition": {
            "train": ["a", "b", "c", "d"],
            "validation": ["e"],
            "id_test": ["f"],
            "parameter_ood": [],
            "ood": [],
            "material_roles": {"lentil": "id", "chickpea": "id"},
        },
        "source_outcomes": [
            {
                "candidates": [
                    {**failed, "failure_stage": "solver", "reason_code": "stalled"},
                    {**failed_replacement, "failure_stage": "solver", "reason_code": "stalled"},
                    replacement,
                ]
            }
        ],
    }
    seen: list[tuple[str, str, Path]] = []

    def load(task: str, dataset_id: str, *, storage_root: Path) -> dict[str, Any]:
        seen.append((task, dataset_id, storage_root))
        return manifest

    _as_driver(monkeypatch, plan_path)
    monkeypatch.setattr(dataset_plan, "run_dataset_plan", lambda *_args, **_kwargs: {"status": "OK", "source_run": "run", "dataset": published})
    monkeypatch.setattr(manifests, "load_package_manifest_evidence", load)
    assert cli.main() == 0
    assert seen == [("transient_drying", "example-dataset", storage)]
    lines = capsys.readouterr().out.splitlines()
    assert lines[:4] == [
        "DATASET COMPLETE: example-dataset (transient_drying)",
        "Published: 02_datasets/transient_drying/example-dataset",
        "Contents: 6 Source cases; 618 samples; 618 transitions",
        "Split: train 4  validation 1  id_test 1  parameter_ood 0  ood 0  (lentil=id, chickpea=id)",
    ]
    # A slot whose first replacement failed too reads as a chain, as on the live board.
    assert lines[4:7] == [
        "Failures:",
        "  lentil case_0001: solver failure (stalled), replaced by case_0003",
        "  lentil case_0003: solver failure (stalled), replaced by case_0004",
    ]
    assert re.fullmatch(r"Duration: \d+ s", lines[-1])


def test_cli_driver_names_the_next_command_of_an_incomplete_source(
    monkeypatch: pytest.MonkeyPatch, plan_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A staged Source publishes no Dataset; the summary says so and gives the command that continues."""
    _as_driver(monkeypatch, plan_path)
    staged = {"status": "STAGED", "source_run": "source~example", "dataset": None, "next": "python -m src.datasets.dataset_packages run plan.yaml"}
    monkeypatch.setattr(dataset_plan, "run_dataset_plan", lambda *_args, **_kwargs: staged)
    assert cli.main() == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[:2] == [
        "DATASET STAGED: Source source~example is not complete, no Dataset published",
        "Next: python -m src.datasets.dataset_packages run plan.yaml",
    ]


def test_controller_factory_freezes_the_source_and_binds_its_evidence_to_every_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admission always freezes the live source, uncommitted changes included, and binds the tree and snapshot evidence of every child."""
    layout = host.HostLayout(tmp_path / "repo", tmp_path / "storage", tmp_path / "runtime", tmp_path / "venv", {})
    snapshot_path = str(layout.storage / "source_snapshots/exact/snapshot.json")
    tree_id = "a" * 12 + "-" + "b" * 12
    admission = host.SourceAdmission("a" * 40, "b" * 64, snapshot_path, "c" * 64, tree_id, layout.runtime / "sources" / tree_id)
    calls: list[dict[str, Any]] = []

    def admit_source(admitted_layout: host.HostLayout, *, freeze: bool) -> host.SourceAdmission:
        assert admitted_layout is layout
        calls.append({"freeze": freeze})
        return admission

    monkeypatch.setattr(host.HostLayout, "resolve", classmethod(lambda _cls, _repository: layout))
    monkeypatch.setattr(host, "admit_source", admit_source)
    controller = dataset_plan._controller()

    assert isinstance(controller, generation_controller.GenerationController)
    assert isinstance(controller.port, host.GenerationHost)
    assert calls == [{"freeze": True}]
    assert controller.port.admission is admission
    environment = host._child_environment(layout, controller.port.admission)
    assert environment["GENERATION_GIT_COMMIT"] == admission.commit
    assert environment["GENERATION_SOURCE_SHA256"] == admission.fingerprint
    assert environment["GENERATION_SOURCE_SNAPSHOT"] == snapshot_path
    assert environment["GENERATION_SOURCE_SNAPSHOT_SHA256"] == admission.snapshot_sha256
    assert environment["PROJECT_ROOT"] == str(admission.tree_path)
    assert environment["SOURCE_TREE_ID"] == tree_id
    assert not layout.storage.exists()


def test_steady_plan_requires_matching_profile(tmp_path: Path) -> None:
    """Load the steady plan, refuse a time horizon it does not have, and reject a task that contradicts its profile."""
    plan = dataset_plan.load_dataset_plan(_write_plan(tmp_path, task="steady_flow"))
    assert plan["task"] == "steady_flow"
    assert "__h-" not in plan["name"]
    with pytest.raises(ValueError, match=r"--duration-hours.*'steady_flow'"):
        dataset_plan.load_dataset_plan(_write_plan(tmp_path, task="steady_flow"), duration_hours=4)
    assert dataset_plan.authored_campaign(plan)["profile_config"].endswith("steady_flow.yaml")
    mismatched = _write_plan(tmp_path, lambda authored: authored.update(task="transient_drying"), task="steady_flow")
    with pytest.raises(ValueError, match="does not simulate the plan task"):
        dataset_plan.load_dataset_plan(mismatched)


def test_complete_steady_run_publishes_partition_and_keeps_driver_log_outside(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Publish the steady payload with its fixed partition and never write into the published folder afterwards."""
    monkeypatch.setenv("SLURM_JOB_ID", "test-allocation")
    steady_path = _write_plan(tmp_path, task="steady_flow")
    plan = dataset_plan.load_dataset_plan(steady_path)
    publication_calls: list[dict[str, Any]] = []
    driver_log = tmp_path / "runtime/logs/datasets/dataset-steady_flow-4711.log"
    driver_log.parent.mkdir(parents=True)
    driver_log.write_text("[1/4] Run plan\n", encoding="utf-8")
    published = tmp_path / "storage/02_datasets/steady_flow/steady-dataset"

    class Controller:
        def run_authored_campaign(self, _path: str, _campaign: dict[str, Any], **_kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(result="OK", run_id="source-run", source_name="steady-source")

    def publish(request: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        # The builder treats any existing Dataset folder as a publication, so nothing may exist before it publishes.
        assert not published.exists()
        publication_calls.append({"request": request, **kwargs})
        published.mkdir(parents=True)
        return {"dataset_id": "steady-dataset", "manifest_path": published / "dataset_manifest.json"}

    monkeypatch.setattr(dataset_plan, "_controller", Controller)
    monkeypatch.setattr(builder, "build_dataset_selection", publish)
    monkeypatch.setattr(host, "current_job_log_files", lambda: (driver_log,))

    result = dataset_plan.run_dataset_plan(steady_path, plan, invocation=("python", "-m", "src.datasets.dataset_packages", "run"))

    assert result["status"] == "OK"
    # The driver log keeps growing until the job ends, so it stays in its log area.
    assert list(published.iterdir()) == []
    assert driver_log.is_file()
    assert publication_calls[0]["request"]["dataset_view"] == "steady_flow"
    assert publication_calls[0]["material_roles"] == {material: specification["role"] for material, specification in plan["materials"].items()}
