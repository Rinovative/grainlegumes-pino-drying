"""
check_package_install.py

Verify editable and wheel installation contracts from unrelated directories.

Responsibilities:
  - Build isolated editable and wheel installations from a scoped source copy
  - Probe the sole ``src`` package and all seven public domains off-checkout
  - Reject missing Python modules and unintended non-package payloads

Design principles:
  - Every build, install, and probe artifact lives in one temporary directory
  - Probes run without checkout fallbacks or dependency installation
  - Wheel inventory and runtime import origin are verified independently

This module does NOT:
  - Mutate the checkout, environment, or production data directories
  - Download project dependencies or contact external services
  - Validate scientific runtime behavior beyond packaging and CLI help contracts
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _run(
    arguments: list[str],
    *,
    cwd: Path,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one bounded package command and raise with captured diagnostics."""
    completed = subprocess.run(  # noqa: S603 -- arguments are maintained local commands
        arguments,
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        message = f"Command failed with exit {completed.returncode}: {arguments!r}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        raise RuntimeError(message)
    return completed


def _copy_source(destination: Path) -> None:
    """Copy only files needed to build the maintained root package."""
    destination.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE.md"):
        shutil.copy2(_REPOSITORY_ROOT / name, destination / name)
    shutil.copytree(
        _REPOSITORY_ROOT / "src",
        destination / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )


def _bootstrap(editable_target: Path | None) -> str:
    """Return child-process path initialization without checkout fallbacks."""
    if editable_target is None:
        return "import sys, sysconfig; sys.path.append(sysconfig.get_paths()['purelib']); "
    return f"import site, sys, sysconfig; site.addsitedir({str(editable_target)!r}); sys.path.append(sysconfig.get_paths()['purelib']); "


def _probe_install(
    *,
    python: Path,
    expected_root: Path,
    cwd: Path,
    editable_target: Path | None,
    wheel_target: Path | None,
) -> None:
    """Import every public domain and dataset service from one install root."""
    bootstrap = _bootstrap(editable_target)
    if wheel_target is not None:
        bootstrap += f"sys.path.insert(0, {str(wheel_target)!r}); "
    probe = (
        bootstrap
        + f"""
from pathlib import Path
import h5py
import src
from src import analysis, common, datasets, domain, experiments, generation, learning
from src.analysis.eda import eda_dataframe
from src.experiments import config as experiment_config
from src.experiments.config import experiments_config_temporal
from src.generation import contracts as generation_contracts
from src.generation.contracts import generation_contracts_scalar_handoff
from src.learning import learning_temporal
from src.datasets import contracts as dataset_contracts
from src.datasets import dataset_packages
from src.datasets import packages as dataset_package_services
from src.datasets import preprocessing as dataset_preprocessing
from src.datasets import runtime as dataset_runtime
from src.datasets.contracts import (
    dataset_contracts_identity,
    dataset_contracts_metadata,
    dataset_contracts_transient,
    dataset_contracts_views,
)
from src.datasets.packages import (
    dataset_packages_builder,
    dataset_packages_generated_batch,
    dataset_packages_manifest,
    dataset_packages_planning,
    dataset_packages_references,
    dataset_packages_transient_shards,
)
from src.datasets.preprocessing import (
    dataset_preprocessing_normalization,
    dataset_preprocessing_splits,
)
from src.datasets.runtime import (
    dataset_runtime_factory,
    dataset_runtime_package_validation,
    dataset_runtime_steady,
    dataset_runtime_training,
    dataset_runtime_transient,
)

expected = Path({str(expected_root)!r}).resolve()
modules = (
    src,
    analysis,
    common,
    datasets,
    domain,
    experiments,
    generation,
    learning,
    dataset_contracts,
    dataset_package_services,
    dataset_preprocessing,
    dataset_runtime,
    dataset_contracts_identity,
    dataset_contracts_metadata,
    dataset_contracts_transient,
    dataset_contracts_views,
    dataset_packages,
    dataset_packages_builder,
    dataset_packages_generated_batch,
    dataset_packages_manifest,
    dataset_packages_planning,
    dataset_packages_references,
    dataset_packages_transient_shards,
    dataset_preprocessing_normalization,
    dataset_preprocessing_splits,
    dataset_runtime_factory,
    dataset_runtime_package_validation,
    dataset_runtime_steady,
    dataset_runtime_training,
    dataset_runtime_transient,
    eda_dataframe,
    experiment_config,
    experiments_config_temporal,
    generation_contracts,
    generation_contracts_scalar_handoff,
    learning_temporal,
)
for module in modules:
    path = Path(module.__file__).resolve()
    assert path.is_relative_to(expected), (module.__name__, path, expected)
assert datasets.runtime.factory.create_dataset is dataset_runtime_factory.create_dataset
assert datasets.packages.generated_batch.load_generated_batch is dataset_packages_generated_batch.load_generated_batch
assert datasets.contracts.identity.DatasetIdentity is dataset_contracts_identity.DatasetIdentity
assert datasets.contracts.metadata.load_dataset_metadata is dataset_contracts_metadata.load_dataset_metadata
assert datasets.preprocessing.normalization.build_normalizer_artifact is dataset_preprocessing_normalization.build_normalizer_artifact
assert datasets.packages.build_campaign_packages is dataset_packages.build_campaign_packages
assert datasets.preprocessing.splits.admit_split_contract is dataset_preprocessing_splits.admit_split_contract
assert datasets.runtime.steady.SteadyFlowDataset is dataset_runtime_steady.SteadyFlowDataset
assert datasets.runtime.training.create_dataloaders is dataset_runtime_training.create_dataloaders
assert datasets.runtime.transient.TransientPhysicalDataset is dataset_runtime_transient.TransientPhysicalDataset
assert datasets.contracts.transient.transient_contract_digest is dataset_contracts_transient.transient_contract_digest
assert datasets.contracts.views.inspect_contract is dataset_contracts_views.inspect_contract
assert dataset_contracts.identity.DatasetIdentity is dataset_contracts_identity.DatasetIdentity
assert dataset_contracts.metadata.load_dataset_metadata is dataset_contracts_metadata.load_dataset_metadata
assert dataset_contracts.transient.transient_contract_digest is dataset_contracts_transient.transient_contract_digest
assert dataset_contracts.views.inspect_contract is dataset_contracts_views.inspect_contract
assert dataset_package_services.builder.build_campaign_packages is dataset_packages_builder.build_campaign_packages
assert dataset_package_services.generated_batch.load_generated_batch is dataset_packages_generated_batch.load_generated_batch
assert dataset_package_services.manifest.load_package_manifest is dataset_packages_manifest.load_package_manifest
assert dataset_package_services.planning.prepare_campaign_packages is dataset_packages_planning.prepare_campaign_packages
assert dataset_package_services.references is dataset_packages_references
assert dataset_package_services.publish_dataset_reference is dataset_packages.publish_dataset_reference
assert dataset_package_services.resolve_dataset_reference is dataset_packages.resolve_dataset_reference
assert dataset_package_services.transient_shards is dataset_packages_transient_shards
assert dataset_package_services.transient_shards.build_transient_shards is dataset_packages_transient_shards.build_transient_shards
assert dataset_package_services.inspect_dataset_package is dataset_packages.inspect_dataset_package
assert dataset_preprocessing.normalization.build_normalizer_artifact is dataset_preprocessing_normalization.build_normalizer_artifact
assert dataset_preprocessing.splits.admit_split_contract is dataset_preprocessing_splits.admit_split_contract
assert dataset_runtime.factory.create_dataset is dataset_runtime_factory.create_dataset
assert dataset_runtime.package_validation.inspect_dataset_package is dataset_runtime_package_validation.inspect_dataset_package
assert dataset_runtime.steady.SteadyFlowDataset is dataset_runtime_steady.SteadyFlowDataset
assert dataset_runtime.training.create_dataloaders is dataset_runtime_training.create_dataloaders
assert dataset_runtime.transient.TransientPhysicalDataset is dataset_runtime_transient.TransientPhysicalDataset
assert dataset_packages.DATASET_PACKAGE_SCHEMA_KIND == dataset_packages_manifest.DATASET_PACKAGE_SCHEMA_KIND
assert dataset_packages.DATASET_PACKAGE_SCHEMA_VERSION == dataset_packages_manifest.DATASET_PACKAGE_SCHEMA_VERSION
assert dataset_packages.build_campaign_packages.__module__ == "src.datasets.dataset_packages"
assert not hasattr(datasets, "base")
assert datasets.contracts.views.inspect_contract("steady_flow").contract_digest
assert generation.contracts.get_profile_contract("transient_drying").id == "transient_drying"
assert experiments.config.temporal is experiments_config_temporal
assert experiment_config.temporal is experiments_config_temporal
assert generation.contracts.scalar_handoff is generation_contracts_scalar_handoff
assert generation_contracts.scalar_handoff is generation_contracts_scalar_handoff
assert learning.temporal is learning_temporal
assert h5py.version.version
print(src.__file__)
print(dataset_packages.__file__)
print(dataset_packages_builder.__file__)
print(dataset_runtime_factory.__file__)
print(eda_dataframe.__file__)
"""
    )
    _run([str(python), "-S", "-B", "-c", probe], cwd=cwd)

    help_probe = bootstrap
    if wheel_target is not None:
        help_probe += f"sys.path.insert(0, {str(wheel_target)!r}); "
    help_probe += (
        "import runpy, sys; sys.argv=['src.datasets.dataset_packages', '--help']; "
        "runpy.run_module('src.datasets.dataset_packages', run_name='__main__')"
    )
    completed = _run([str(python), "-S", "-B", "-c", help_probe], cwd=cwd)
    commands = "{build,inspect,smoke,refs,resolve,inspect-ref}"
    if commands not in completed.stdout:
        message = "Dataset package help did not expose the maintained package and reference commands."
        raise RuntimeError(message)


def main() -> int:
    """Run editable and isolated-wheel checks and remove all temporary material."""
    with tempfile.TemporaryDirectory(prefix="grainlegumes-package-contract-") as temporary:
        scratch = Path(temporary)
        source = scratch / "source"
        _copy_source(source)
        neutral = scratch / "neutral"
        neutral.mkdir()

        editable_target = scratch / "editable-install"
        _run(
            [
                sys.executable,
                "-B",
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-build-isolation",
                "--target",
                str(editable_target),
                "--editable",
                str(source),
            ],
            cwd=scratch,
        )
        _probe_install(
            python=Path(sys.executable),
            expected_root=source,
            cwd=neutral,
            editable_target=editable_target,
            wheel_target=None,
        )

        wheel_dir = scratch / "wheel"
        wheel_dir.mkdir()
        _run(
            [
                sys.executable,
                "-B",
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-build-isolation",
                "--wheel-dir",
                str(wheel_dir),
                str(source),
            ],
            cwd=scratch,
        )
        wheels = sorted(wheel_dir.glob("*.whl"))
        if len(wheels) != 1:
            message = f"Expected one wheel, found {wheels!r}."
            raise RuntimeError(message)
        wheel = wheels[0]
        with zipfile.ZipFile(wheel) as archive:
            names = set(archive.namelist())
            top_level_path = next(name for name in names if name.endswith(".dist-info/top_level.txt"))
            top_levels = set(archive.read(top_level_path).decode("utf-8").splitlines())
            if top_levels != {"src"}:
                message = f"Wheel top-level packages are incomplete: {sorted(top_levels)!r}."
                raise RuntimeError(message)
            expected_python = {path.relative_to(source).as_posix() for path in (source / "src").rglob("*.py")}
            wheel_python = {name for name in names if name.startswith("src/") and name.endswith(".py")}
            missing = sorted(expected_python.difference(wheel_python))
            unexpected = sorted(wheel_python.difference(expected_python))
            if missing or unexpected:
                message = f"Wheel Python inventory mismatch: missing={missing!r}, unexpected={unexpected!r}."
                raise RuntimeError(message)
            unintended = sorted(name for name in names if name.startswith(("configs/", "notebooks/", "simulation/", "tests/")))
            if unintended:
                message = f"Wheel contains unintended repository payloads: {unintended!r}."
                raise RuntimeError(message)

        wheel_target = scratch / "wheel-install"
        _run(
            [
                sys.executable,
                "-B",
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--target",
                str(wheel_target),
                str(wheel),
            ],
            cwd=scratch,
        )
        _probe_install(
            python=Path(sys.executable),
            expected_root=wheel_target,
            cwd=neutral,
            editable_target=None,
            wheel_target=wheel_target,
        )

        print("Editable install: root src, all seven public domains, dataset services, and EDA passed.")
        print("Wheel install: sole src package, inventory, imports, and dataset-builder help passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
