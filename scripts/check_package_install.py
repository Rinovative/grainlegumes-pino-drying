"""
===============================================================================
check_package_install.py
===============================================================================
Verify editable and wheel installation contracts from unrelated directories.

Responsibilities:
  - Build isolated editable and wheel installations from a scoped source copy
  - Probe the sole ``src`` package and all six public domains off-checkout
  - Reject missing Python modules and unintended non-package payloads

Design principles:
  - Every build, install, and probe artifact lives in one temporary directory
  - Probes run without checkout fallbacks or dependency installation
  - Wheel inventory and runtime import origin are verified independently

This module does NOT:
  - Mutate the checkout, environment, or production data directories
  - Download project dependencies or contact external services
  - Validate scientific runtime behavior beyond packaging and CLI help contracts
===============================================================================
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
import src
from src import analysis, common, datasets, domain, experiments, learning
from src.analysis.eda import eda_dataframe
from src.datasets import dataset_build, dataset_generated_batch

expected = Path({str(expected_root)!r}).resolve()
modules = (
    src,
    analysis,
    common,
    datasets,
    domain,
    experiments,
    learning,
    dataset_build,
    dataset_generated_batch,
    eda_dataframe,
)
for module in modules:
    path = Path(module.__file__).resolve()
    assert path.is_relative_to(expected), (module.__name__, path, expected)
assert datasets.build is dataset_build
assert datasets.generated_batch is dataset_generated_batch
assert datasets.generated_batch.load_generated_batch is dataset_generated_batch.load_generated_batch
print(src.__file__)
print(dataset_generated_batch.__file__)
print(dataset_build.__file__)
print(eda_dataframe.__file__)
"""
    )
    _run([str(python), "-S", "-B", "-c", probe], cwd=cwd)

    help_probe = bootstrap
    if wheel_target is not None:
        help_probe += f"sys.path.insert(0, {str(wheel_target)!r}); "
    help_probe += (
        "import runpy, sys; sys.argv=['src.datasets.dataset_build', '--help']; runpy.run_module('src.datasets.dataset_build', run_name='__main__')"
    )
    completed = _run([str(python), "-S", "-B", "-c", help_probe], cwd=cwd)
    if "Completed generated batch and final dataset identifier" not in completed.stdout:
        message = "Builder help did not expose the maintained positional batch contract."
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

        print("Editable install: root src, all six public domains, dataset services, and EDA passed.")
        print("Wheel install: sole src package, inventory, imports, and dataset-builder help passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
