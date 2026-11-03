"""Protect the ordinary Dataset plan and CLI handoffs.

Responsibilities:
  - Preserve plan defaults, explicit small overrides, and the role boundary between Sources and Datasets
  - Check the computed Dataset name, the plan schema, the Dataset-only revision and the explicit new Source that reuses its label
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

from src import common, generation
from src.datasets import dataset_packages as cli
from src.datasets import dataset_plan
from src.datasets.contracts import dataset_contracts_partition as partition
from src.datasets.packages import dataset_packages_builder as builder
from src.datasets.packages import dataset_packages_manifest as manifests
from src.datasets.packages import dataset_packages_transient_shards as shards
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
        "schema_version": 2,
        "task": task,
        "generation": {
            "sources_config": f"{_PHYSICAL_MODEL}/sources.yaml",
            "registry_config": f"{_PHYSICAL_MODEL}/registry.yaml",
            "common_config": f"{_PHYSICAL_MODEL}/common.yaml",
            "operations_config": f"{_PHYSICAL_MODEL}/fixed_bed.yaml",
            "profile_config": f"{_PHYSICAL_MODEL}/profiles/{task}.yaml",
            "execution_config": "configs/01_generation/execution.yaml",
            "sampling": {"method": "lhs", "seed_base": 41},
            "replacement_capacity": 1,
            "domain_profile": "reference_range",
        },
        "materials": {"lentil": {"count": 10, "role": "id"}, "chickpea": {"count": 10, "role": "id"}},
        "selection_seed": 42,
        "split": {"seed": 7, "train_fraction": 0.5, "validation_fraction": 0.25, "id_test_fraction": 0.25},
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


def _small_materials() -> dict[str, dict[str, Any]]:
    """Return the three-case override with each material's authored role."""
    return {material: {"count": 3, "role": specification["role"]} for material, specification in _base_plan()["materials"].items()}


def _small_plan(path: Path) -> dict[str, Any]:
    return dataset_plan.load_dataset_plan(path, counts=("lentil=3", "chickpea=3"), split_fractions="1/3,1/3,1/3")


def test_documented_defaults_and_small_overrides_preserve_authored_plan(plan_path: Path) -> None:
    """Keep public scientific defaults and temporary overrides distinct from YAML."""
    original_bytes = plan_path.read_bytes()

    authored = _base_plan()

    defaults = dataset_plan.load_dataset_plan(plan_path)
    preview = dataset_plan.preview_dataset_plan(defaults)
    assert defaults["materials"] == authored["materials"]
    assert defaults["generation"]["sampling"]["seed_base"] == authored["generation"]["sampling"]["seed_base"]
    assert defaults["selection_seed"] == authored["selection_seed"]
    assert defaults["split"] == authored["split"]
    assert preview["split"]["seed"] == common.random.derive_subseed(authored["split"]["seed"], "split")

    small = dataset_plan.load_dataset_plan(
        plan_path,
        counts=("lentil=3", "chickpea=3"),
        split_fractions="1/3,1/3,1/3",
    )
    assert small["materials"] == _small_materials()
    assert tuple(small["split"][key] for key in ("train_fraction", "validation_fraction", "id_test_fraction")) == pytest.approx((1 / 3, 1 / 3, 1 / 3))
    assert plan_path.read_bytes() == original_bytes

    case_materials = {
        f"{material}-{index}": material for material, specification in small["materials"].items() for index in range(specification["count"])
    }
    membership = partition.resolve_case_partition(
        tuple(case_materials),
        {
            "kind": "seeded_material_fraction",
            "train_fraction": 1 / 3,
            "validation_fraction": 1 / 3,
            "id_test_fraction": 1 / 3,
        },
        split_seed=common.random.derive_subseed(small["split"]["seed"], "split"),
        material_by_case=case_materials,
    )
    assert {role: len(cases) for role, cases in membership.items()} == {"train": 2, "validation": 2, "id_test": 2}
    assert all(sum(case_materials[case] == material for case in cases) == 1 for cases in membership.values() for material in small["materials"])


def test_dataset_name_is_computed_from_materials_split_domain_and_seed(tmp_path: Path) -> None:
    """The name states the resolved content: split counts of ID materials, held-out materials, domain and Generation seed."""

    def mutate(authored: dict[str, Any]) -> None:
        authored["materials"] = {"lentil": {"count": 10, "role": "id"}, "chickpea": {"count": 7, "role": "ood"}}
        authored["generation"]["domain_profile"] = "reference_and_parameter_extension"
        authored["generation"]["sampling"]["seed_base"] = 123
        authored["split"].update(train_fraction=0.5, validation_fraction=0.25, id_test_fraction=0.25)

    path = _write_plan(tmp_path, mutate)

    assert dataset_plan.load_dataset_plan(path)["name"] == "lentil-train5-val2-test3+chickpea-ood7__ref-param-ext__lhs-seed-123"
    overridden = dataset_plan.load_dataset_plan(path, counts=("lentil=4",), split_fractions="1/2,1/4,1/4")
    assert overridden["name"] == "lentil-train2-val1-test1+chickpea-ood7__ref-param-ext__lhs-seed-123"
    # Only a horizon that deviates from the configured one is named, so an override equal to it adds no token.
    common_config = yaml.safe_load((common.paths.get_project_root() / f"{_PHYSICAL_MODEL}/common.yaml").read_text(encoding="utf-8"))
    configured_hours = generation.cases.config.physical_horizon_hours(common_config["time"])
    assert dataset_plan.load_dataset_plan(path, duration_hours=configured_hours)["name"] == dataset_plan.load_dataset_plan(path)["name"]
    assert dataset_plan.load_dataset_plan(path, duration_hours=4)["name"].endswith("__lhs-seed-123__h-4")
    with pytest.raises(ValueError, match="without a case in every role"):
        dataset_plan.load_dataset_plan(path, counts=("lentil=2",))


def test_replacement_capacity_is_authored_policy_that_sizes_admission_but_not_identity(tmp_path: Path) -> None:
    """Capacity zero is the default; one integer or a mapping pre-registers replacements without changing name, seeds or split."""

    def with_capacity(value: object | None) -> dict[str, Any]:
        def mutate(authored: dict[str, Any]) -> None:
            authored["generation"].pop("replacement_capacity", None)
            if value is not None:
                authored["generation"]["replacement_capacity"] = value

        return dataset_plan.load_dataset_plan(_write_plan(tmp_path, mutate), counts=("lentil=3", "chickpea=4"), split_fractions="1/3,1/3,1/3")

    absent, everywhere, mapped = with_capacity(None), with_capacity(2), with_capacity({"lentil": 3})
    assert absent["generation"]["replacement_capacity"] == {"lentil": 0, "chickpea": 0}
    assert everywhere["generation"]["replacement_capacity"] == {"lentil": 2, "chickpea": 2}
    assert mapped["generation"]["replacement_capacity"] == {"lentil": 3, "chickpea": 0}
    assert absent["name"] == everywhere["name"] == mapped["name"]
    assert [dataset_plan.effective_execution_overrides(plan)["max_admission_cases"] for plan in (absent, everywhere, mapped)] == [7, 11, 10]
    assert dataset_plan.effective_execution_overrides(mapped, {"max_admission_cases": 5})["max_admission_cases"] == 5
    campaigns = [dataset_plan.authored_campaign(plan) for plan in (absent, everywhere, mapped)]
    assert [campaign["replacement_capacity"] for campaign in campaigns] == [
        plan["generation"]["replacement_capacity"] for plan in (absent, everywhere, mapped)
    ]
    assert all(campaign["sampling"] == campaigns[0]["sampling"] for campaign in campaigns)
    preview = dataset_plan.preview_dataset_plan(mapped)
    assert preview["replacement_capacity"] == {"lentil": 3, "chickpea": 0}
    assert preview["execution_overrides"]["max_admission_cases"] == 10
    assert preview["split"] == dataset_plan.preview_dataset_plan(absent)["split"]


@pytest.mark.parametrize("value", [-1, True, 1.5, "2", {"lentil": -1}, {"lentil": "1"}, {"field_pea": 1}])
def test_replacement_capacity_rejects_unsupported_values(tmp_path: Path, value: object) -> None:
    """Reject negative, non-integer and unknown-material capacities before any Generation work."""
    path = _write_plan(tmp_path, lambda authored: authored["generation"].update(replacement_capacity=value))
    with pytest.raises(ValueError, match="replacement_capacity"):
        dataset_plan.load_dataset_plan(path)


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
    overrides = {"counts": ("lentil=3", "chickpea=3"), "split_fractions": "1/3,1/3,1/3"}
    plain = dataset_plan.load_dataset_plan(plan_path, **overrides)
    dataset_only = dataset_plan.load_dataset_plan(plan_path, revision="converter-fix", **overrides)
    fresh = dataset_plan.load_dataset_plan(plan_path, revision="converter-fix", new_source=True, **overrides)

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


def test_plan_schema_rejects_an_authored_name_and_the_old_version(tmp_path: Path) -> None:
    """Schema 2 computes the name; a plan that still authors one is refused with a clear message."""
    with pytest.raises(ValueError, match="exactly the fields"):
        dataset_plan.load_dataset_plan(_write_plan(tmp_path, lambda authored: authored.update(name="authored-name")))
    with pytest.raises(ValueError, match=r"schema_version must be 2.*removed the authored name"):
        dataset_plan.load_dataset_plan(_write_plan(tmp_path, lambda authored: authored.update(schema_version=1)))
    # The Source name is derived like the Dataset name; only --revision with --new-source varies it, never the YAML.
    for key in ("source_name", "revision"):
        with pytest.raises(ValueError, match="six maintained configs"):
            dataset_plan.load_dataset_plan(_write_plan(tmp_path, lambda authored, key=key: authored["generation"].update({key: "authored-source"})))


def test_generation_projection_contains_physics_but_no_dataset_roles_or_split(plan_path: Path) -> None:
    """Keep material roles and case partition in Dataset identity, not Source design."""
    plan = _small_plan(plan_path)
    campaign = dataset_plan.authored_campaign(plan)

    assert campaign["sampling"]["seed_base"] == plan["generation"]["sampling"]["seed_base"]
    assert campaign["sampling"]["counts"] == {"natural": {"lentil": 3, "chickpea": 3}}
    assert set(campaign["materials"]) == {"lentil", "chickpea"}
    assert not {"name", "dataset_name", "selection_seed", "selection", "roles", "split"} & campaign.keys()
    assert all(isinstance(count, int) for count in campaign["sampling"]["counts"]["natural"].values())


def test_preview_does_not_construct_generation_controller(tmp_path: Path, plan_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep plan preview read-only even when no scientific storage exists."""
    storage = tmp_path / "not-created"
    monkeypatch.setenv("STORAGE_ROOT", str(storage))
    monkeypatch.setattr(dataset_plan, "_controller", lambda: pytest.fail("preview constructed Generation"))

    summary = dataset_plan.preview_dataset_plan(
        _small_plan(plan_path),
        duration_hours=4,
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
    plan["generation"]["replacement_capacity"] = {"lentil": 1, "chickpea": 0}

    class Controller:
        def authored_campaign_plan(self, _path: str, campaign: dict[str, Any], **kwargs: Any) -> SimpleNamespace:
            assert campaign == dataset_plan.authored_campaign(plan)
            # Three plus three original candidates and the one pre-registered lentil replacement.
            assert kwargs == {"scientific_overrides": None, "execution_overrides": {"max_admission_cases": 7}}
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
        "--count",
        "lentil=3",
        "--count",
        "chickpea=3",
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
    monkeypatch.setattr(shards, "build_transient_shards", lambda *_args, **_kwargs: pytest.fail("STAGED run built PT"))
    execution = {"node_jobs": 1, "max_admission_cases": 6, "cores_per_case": 8}

    result = dataset_plan.run_dataset_plan(
        plan_path,
        _small_plan(plan_path),
        invocation=invocation,
        duration_hours=4,
        execution_overrides=execution,
    )

    assert result["status"] == "STAGED"
    assert result["dataset"] is None
    assert result["next"] == shlex.join(invocation)
    assert len(generation_calls) == 1
    assert generation_calls[0]["scientific_overrides"] == {"duration_hours": 4}
    assert generation_calls[0]["execution_overrides"] == execution
    assert generation_calls[0]["max_admission_cases"] == 6
    # Admission is always dirty-capable: the driver forwards no snapshot or strictness switches.
    assert set(generation_calls[0]) == {"path", "campaign", "scientific_overrides", "execution_overrides", "max_admission_cases"}


def test_complete_run_forwards_exact_selection_partition_and_builds_pt(plan_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bind publication to the plan's selection, fixed partition, and invocation before PT."""
    monkeypatch.setenv("SLURM_JOB_ID", "test-allocation")
    plan = _small_plan(plan_path)
    invocation = ("python", "-m", "src.datasets.dataset_packages", "run", "plan.yaml", "--max-admission-cases", "6")
    execution = {"node_jobs": 1, "max_admission_cases": 6, "cores_per_case": 8}
    generation_calls: list[dict[str, Any]] = []
    publication_calls: list[dict[str, Any]] = []
    pt_calls: list[str] = []

    class Controller:
        def run_authored_campaign(self, path: str, campaign: dict[str, Any], **kwargs: Any) -> SimpleNamespace:
            generation_calls.append({"path": path, "campaign": campaign, **kwargs})
            return SimpleNamespace(result="OK", run_id="source-run", source_name="lentil-chickpea__abc")

    def publish(request: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        publication_calls.append({"request": request, **kwargs})
        return {"dataset_id": "dataset-id", "manifest_path": "manifest.json"}

    def build_pt(dataset_id: str) -> dict[str, str]:
        pt_calls.append(dataset_id)
        return {"receipt_path": "transient_pt_shards/receipt.json"}

    monkeypatch.setattr(dataset_plan, "_controller", Controller)
    monkeypatch.setattr(builder, "build_dataset_selection", publish)
    monkeypatch.setattr(shards, "build_transient_shards", build_pt)

    result = dataset_plan.run_dataset_plan(
        plan_path,
        plan,
        invocation=invocation,
        duration_hours=4,
        execution_overrides=execution,
    )

    assert result["status"] == "OK"
    assert result["dataset"]["dataset_id"] == "dataset-id"
    assert result["pt"]["receipt_path"] == "transient_pt_shards/receipt.json"
    assert generation_calls[0]["scientific_overrides"] == {"duration_hours": 4}
    assert generation_calls[0]["execution_overrides"] == execution
    assert generation_calls[0]["max_admission_cases"] == 6
    assert publication_calls[0]["request"]["selection"] == {
        "kind": "seed_ranked",
        "root_seed": _base_plan()["selection_seed"],
        "source_names": ["lentil-chickpea__abc"],
        "target_counts": {"lentil": 3, "chickpea": 3},
    }
    assert publication_calls[0]["request"]["source_cases"] == []
    split_seed = _base_plan()["split"]["seed"]
    expected_seed = common.random.derive_subseed(split_seed, "split")
    assert publication_calls[0]["partition_spec"] == {
        "seed_input": split_seed,
        "seed": expected_seed,
        "recipe": {
            "kind": "seeded_material_fraction",
            "train_fraction": pytest.approx(1 / 3),
            "validation_fraction": pytest.approx(1 / 3),
            "id_test_fraction": pytest.approx(1 / 3),
        },
        "material_roles": {material: specification["role"] for material, specification in _small_materials().items()},
    }
    assert publication_calls[0]["plan_invocation"] == list(invocation)
    assert pt_calls == ["dataset-id"]


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
    monkeypatch.setattr(shards, "build_transient_shards", lambda *_args, **_kwargs: pytest.fail("CLI built PT"))
    arguments = [
        "dataset_packages",
        "run",
        str(plan_path),
        "--count",
        "lentil=3",
        "--count",
        "chickpea=3",
        "--revision",
        "smoke",
        "--split-fractions",
        "1/3,1/3,1/3",
        "--duration-hours",
        "4",
        "--node-jobs",
        "1",
        "--max-admission-cases",
        "6",
        "--cores-per-case",
        "8",
    ]
    monkeypatch.setattr(sys, "argv", [*arguments, "--preview"])
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
    assert preview["materials"] == _small_materials()
    assert preview["generation_seed_base"] == authored["generation"]["sampling"]["seed_base"]
    assert preview["selection"] == {"kind": "seed_ranked", "seed": authored["selection_seed"]}
    assert preview["split"]["seed"] == common.random.derive_subseed(authored["split"]["seed"], "split")
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
        "--count",
        "lentil=3",
        "--count",
        "chickpea=3",
        "--split-fractions",
        "1/3,1/3,1/3",
        "--preview",
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
            "ood": [],
            "material_roles": {"lentil": "id", "chickpea": "id"},
        },
        "source_outcomes": [{"candidates": [{**failed, "failure_stage": "solver", "reason_code": "stalled"}, replacement]}],
    }
    seen: list[tuple[str, str, Path]] = []

    def load(task: str, dataset_id: str, *, storage_root: Path) -> dict[str, Any]:
        seen.append((task, dataset_id, storage_root))
        return manifest

    _as_driver(monkeypatch, plan_path)
    monkeypatch.setattr(
        dataset_plan, "run_dataset_plan", lambda *_args, **_kwargs: {"status": "OK", "source_run": "run", "dataset": published, "pt": None}
    )
    monkeypatch.setattr(manifests, "load_package_manifest_evidence", load)
    assert cli.main() == 0
    assert seen == [("transient_drying", "example-dataset", storage)]
    lines = capsys.readouterr().out.splitlines()
    assert lines[:4] == [
        "DATASET COMPLETE: example-dataset (transient_drying)",
        "Published: 02_datasets/transient_drying/example-dataset",
        "Contents: 6 Source cases; 618 samples; 618 transitions",
        "Split: train 4  validation 1  id_test 1  ood 0  (lentil=id, chickpea=id)",
    ]
    assert lines[4:6] == ["Failures:", "  lentil case_0001: solver failure (stalled), replaced by case_0004"]
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
    """Load the steady plan and reject a plan whose task contradicts its Generation profile."""
    plan = dataset_plan.load_dataset_plan(_write_plan(tmp_path, task="steady_flow"))
    assert plan["task"] == "steady_flow"
    assert "__h-" not in plan["name"]
    assert "__h-" not in dataset_plan.load_dataset_plan(_write_plan(tmp_path, task="steady_flow"), duration_hours=4)["name"]
    assert dataset_plan.authored_campaign(plan)["profile_config"].endswith("steady_flow.yaml")
    mismatched = _write_plan(tmp_path, lambda authored: authored.update(task="transient_drying"), task="steady_flow")
    with pytest.raises(ValueError, match="does not simulate the plan task"):
        dataset_plan.load_dataset_plan(mismatched)


def test_complete_steady_run_publishes_partition_without_transient_shards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Publish the steady payload with its fixed partition, never build transient PT shards, and link the driver log only afterwards."""
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
    monkeypatch.setattr(shards, "build_transient_shards", lambda _dataset_id: pytest.fail("steady plans build no transient shards"))
    monkeypatch.setattr(host, "current_job_log_files", lambda: (driver_log,))

    result = dataset_plan.run_dataset_plan(steady_path, plan, invocation=("python", "-m", "src.datasets.dataset_packages", "run"))

    assert result["status"] == "OK"
    assert result["pt"] is None
    assert (published / "logs" / driver_log.name).samefile(driver_log)
    assert publication_calls[0]["request"]["dataset_view"] == "steady_flow"
    assert publication_calls[0]["partition_spec"]["material_roles"] == {
        material: specification["role"] for material, specification in plan["materials"].items()
    }
