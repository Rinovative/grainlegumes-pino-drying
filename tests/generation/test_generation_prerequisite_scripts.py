"""Native Generation worker source, environment, and scratch contracts.

Responsibilities:
  - Check allocation requirements, spooled workers, source-tree binding, version admission, and scratch
  - Check that workers verify a Git-less frozen source tree against its snapshot and refuse any drift

Design principles:
  - Workers run from a read-only source tree with explicit storage, runtime, and native-environment roots
  - Source admission never needs Git: a failing Git stub is installed and its call log must stay empty

This module does NOT:
  - Contact real Slurm or launch licensed production solves
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest
from support.fake_git import FAKE_GIT_HEAD
from support.source_tree import FrozenProject, tampered

_CAMPAIGN_RUN_ID = "source~steady_flow~lentil-350+chickpea-350+kidney-bean-350__ref-range__lhs-seed-9300"
_CANONICAL_CAMPAIGN_RUN_ID = "source~transient_drying~lentil-50+chickpea-50+kidney-bean-20__ref-range__lhs-seed-9200"
_BENCHMARK_RUN_ID = "core_scaling_transient__0123456789abcdef"
_NODE_ARGUMENTS = ("campaign-case", _CAMPAIGN_RUN_ID, "batch", "1", "2")


@pytest.mark.parametrize("arguments", [(), ("--candidate",)])
def test_native_provisioning_requires_slurm(arguments: tuple[str, ...]) -> None:
    """Refuse environment creation outside a scheduler allocation."""
    environment = os.environ.copy()
    environment.pop("SLURM_JOB_ID", None)
    script = Path(__file__).resolve().parents[2] / "scripts/provision_native.sh"
    result = subprocess.run(["bash", str(script), *arguments], env=environment, capture_output=True, text=True, check=False)  # noqa: S603, S607 -- test-owned executable and controlled argument vector; no shell interpolation
    assert result.returncode == 2
    assert "Slurm allocation" in result.stderr


def _write_executable(path: Path, contents: str) -> None:
    path.write_text("#!/bin/bash\nset -euo pipefail\n" + contents, encoding="utf-8")
    path.chmod(0o755)


@pytest.fixture
def native_worker(frozen_project: FrozenProject, tmp_path: Path) -> tuple[Path, dict[str, str], Path]:
    """Freeze a checkout into a Git-less source tree and prepare inert site commands and a logging venv."""
    receipt = frozen_project.materialize()
    tree = Path(receipt["tree_path"])
    assert receipt["git_commit"] == FAKE_GIT_HEAD
    assert not (tree / ".git").exists()
    failing_git, _git_log = frozen_project.failing_git()
    binary = tmp_path / "bin"
    binary.mkdir()
    command_log = tmp_path / "commands.log"
    _write_executable(binary / "module", 'printf "module <%s>\\n" "$*" >> "$COMMAND_LOG"\n')
    _write_executable(binary / "comsol", 'printf "COMSOL Multiphysics 6.4.0.293\\n"\n')
    _write_executable(binary / "squeue", 'printf "%s" "${FAKE_SQUEUE_ACTIVE:-}"\n')
    _write_executable(
        frozen_project.venv / "bin" / "python",
        'printf "python <%s>\\n" "$*" >> "$COMMAND_LOG"\nif [[ "${3:-}" == cleanup-worker-workspace ]]; then rmdir -- "$4"; fi\n',
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    environment = {
        **frozen_project.tree_environment(receipt, git_binary=failing_git),
        "COMMAND_LOG": str(command_log),
        "SLURM_JOB_ID": "12345",
        "SLURM_CPUS_PER_TASK": "2",
        "TMPDIR": str(scratch),
        "GENERATION_COMSOL_MODULE": "Comsol/v6.4",
        "GENERATION_PYTHON_EXECUTABLE": "python3",
        "GENERATION_COMSOL_EXECUTABLE": "comsol",
        "GENERATION_ATTEMPT_INDEX": "1",
        "GENERATION_CAMPAIGN_RUN_ID": _CAMPAIGN_RUN_ID,
        "GENERATION_BENCHMARK_RUN_ID": _BENCHMARK_RUN_ID,
    }
    environment["PATH"] = f"{binary}{os.pathsep}{environment['PATH']}"
    return tree, environment, command_log


def _run_node(tree: Path, environment: dict[str, str], *arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the tree's node launcher with the given mode and arguments."""
    return subprocess.run(  # noqa: S603 -- test-owned executable and controlled argument vector; no shell interpolation
        ["/bin/bash", str(tree / "scripts" / "generation_node.sh"), str(tree), *arguments],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("script", "arguments", "expected_cli"),
    [
        ("campaign-case", (_CAMPAIGN_RUN_ID, "batch", "1", "2"), "run-campaign-case"),
        ("campaign-case", (_CANONICAL_CAMPAIGN_RUN_ID, "transient_drying__lentil__natural", "1", "2"), "run-campaign-case"),
        ("benchmark-case", (_BENCHMARK_RUN_ID, "sweep_1", "cores_02", "1", "case_01"), "run-core-benchmark-case"),
    ],
)
def test_spooled_worker_uses_shared_source_native_modules_and_cleans_scratch(
    native_worker: tuple[Path, dict[str, str], Path],
    tmp_path: Path,
    script: str,
    arguments: tuple[str, ...],
    expected_cli: str,
) -> None:
    """Run the worker from Slurm's spool path with only shared source beside it."""
    repository, environment, command_log = native_worker
    if script == "campaign-case":
        environment["GENERATION_CAMPAIGN_RUN_ID"] = arguments[0]
    else:
        # A benchmark case runs only as a step of its sweep node job.
        environment["GENERATION_BENCHMARK_RUN_ID"] = arguments[0]
        environment["SLURM_STEP_ID"] = "7"
    spooled = tmp_path / "slurm_script"
    shutil.copy2(repository / "scripts/generation_node.sh", spooled)
    result = subprocess.run(  # noqa: S603 -- test-owned executable and controlled argument vector; no shell interpolation
        ["/bin/bash", str(spooled), str(repository), script, *arguments],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Preflight" not in result.stdout
    commands = command_log.read_text(encoding="utf-8")
    assert "module <load Comsol/v6.4>" in commands
    assert expected_cli in commands
    assert "cleanup-worker-workspace" in commands
    assert not tuple(Path(environment["TMPDIR"]).iterdir())


def test_worker_start_sweeps_only_own_marked_scratch_of_finished_jobs(
    native_worker: tuple[Path, dict[str, str], Path],
) -> None:
    """Remove marked worker roots whose job ended and whose lifetime lock is free; leave every other directory alone."""
    repository, environment, _command_log = native_worker
    scratch = Path(environment["TMPDIR"])

    def make(name: str, *, marked: bool = True, locked: bool = True) -> Path:
        directory = scratch / name
        directory.mkdir()
        (directory / "payload.dat").write_text("unpublished", encoding="utf-8")
        if marked:
            (directory / ".generation-worker-workspace.json").write_text("{}", encoding="utf-8")
        if locked:
            Path(f"{directory}.lock").write_text("", encoding="utf-8")
        return directory

    stale_case = make("vp2-generation-111.AAAAAA")
    stale_benchmark = make("vp2-benchmark-112-nominal.BBBBBB")
    active = make("vp2-generation-222.CCCCCC")
    unmarked = make("vp2-generation-333.DDDDDD", marked=False)
    foreign = make("other-444.EEEEEE")
    without_lock = make("vp2-generation-555.FFFFFF", locked=False)
    # A worker that started after the queue snapshot still holds its lifetime lock and must survive.
    starting = make("vp2-generation-666.GGGGGG")
    holder = subprocess.Popen(["flock", "-x", f"{starting}.lock", "sleep", "60"])  # noqa: S603, S607 -- test-owned helper process
    try:
        for _ in range(100):
            probe = subprocess.run(["flock", "-n", "-x", f"{starting}.lock", "true"], check=False)  # noqa: S603, S607
            if probe.returncode != 0:
                break
            time.sleep(0.05)
        environment["FAKE_SQUEUE_ACTIVE"] = "222\n"
        result = subprocess.run(  # noqa: S603 -- test-owned executable and controlled argument vector; no shell interpolation
            [
                "/bin/bash",
                str(repository / "scripts" / "generation_node.sh"),
                str(repository),
                "campaign-case",
                _CAMPAIGN_RUN_ID,
                "batch",
                "1",
                "2",
            ],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
    finally:
        holder.kill()
        holder.wait()
    assert result.returncode == 0, result.stderr
    assert not stale_case.exists()
    assert not Path(f"{stale_case}.lock").exists()
    assert not stale_benchmark.exists()
    assert active.exists()
    assert unmarked.exists()
    assert foreign.exists()
    assert without_lock.exists()
    assert starting.exists()
    assert str(stale_case) in result.stderr
    assert str(stale_benchmark) in result.stderr
    assert str(active) not in result.stderr
    assert str(starting) not in result.stderr


@pytest.mark.parametrize(
    ("run_id", "environment_override"),
    [
        ("source~transient_drying~bad/name", {}),
        ("source~transient_drying~lentil-50__ref-range__lhs-seed-9200~extra", {}),
        ("campaign-v2~transient_drying~seen_lentil-n1__lhs-seed-9200", {}),
        (_CANONICAL_CAMPAIGN_RUN_ID, {"GENERATION_CAMPAIGN_RUN_ID": _CAMPAIGN_RUN_ID}),
        (_CANONICAL_CAMPAIGN_RUN_ID, {"SLURM_CPUS_PER_TASK": "1"}),
    ],
)
def test_canonical_worker_rejects_malformed_identity_environment_and_allocation(
    native_worker: tuple[Path, dict[str, str], Path], run_id: str, environment_override: dict[str, str]
) -> None:
    """The real shell wrapper keeps canonical identity and Slurm guards intact."""
    repository, environment, command_log = native_worker
    environment["GENERATION_CAMPAIGN_RUN_ID"] = _CANONICAL_CAMPAIGN_RUN_ID
    environment.update(environment_override)
    result = subprocess.run(  # noqa: S603 -- test-owned executable and controlled argument vector; no shell interpolation
        ["/bin/bash", str(repository / "scripts" / "generation_node.sh"), str(repository), "campaign-case", run_id, "batch", "1", "2"],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Campaign case identity or Slurm allocation is inconsistent." in result.stderr
    assert "CASE START" not in result.stdout
    assert not command_log.exists()


def test_worker_admits_a_git_less_tree_without_calling_git(
    native_worker: tuple[Path, dict[str, str], Path], frozen_project: FrozenProject, tmp_path: Path
) -> None:
    """A tree without Git metadata passes the worker validation although every Git call would fail."""
    tree, environment, command_log = native_worker
    result = _run_node(tree, environment, *_NODE_ARGUMENTS)
    assert result.returncode == 0, result.stderr
    assert "run-campaign-case" in command_log.read_text(encoding="utf-8")
    assert not (frozen_project.root / "failing-bin/git-calls.log").exists()
    reported = _run_node(tree, environment, "node-worker", str(_node_directory(tmp_path)), "2")
    assert f"{tree}@{FAKE_GIT_HEAD}:{environment['GENERATION_SOURCE_SHA256']}" in reported.stdout
    assert not (frozen_project.root / "failing-bin/git-calls.log").exists()


@pytest.mark.parametrize(
    ("tamper", "expected"),
    [("scripts/source_fingerprint.py", "does not match"), ("src/module.py", "does not match")],
)
def test_worker_rejects_a_changed_tree_before_python_or_comsol(native_worker: tuple[Path, dict[str, str], Path], tamper: str, expected: str) -> None:
    """Fail closed when one byte of the frozen tree differs from its snapshot."""
    tree, environment, command_log = native_worker
    tampered(tree, tamper)
    result = _run_node(tree, environment, *_NODE_ARGUMENTS)
    assert result.returncode != 0
    assert expected in result.stderr
    assert "source tree verified against its snapshot" in result.stderr
    assert not command_log.exists()


def test_worker_rejects_an_added_tree_file(native_worker: tuple[Path, dict[str, str], Path]) -> None:
    """A file that the snapshot does not list is drift, not harmless extra content."""
    tree, environment, command_log = native_worker
    (tree / "src/extra.py").parent.chmod(0o755)
    (tree / "src/extra.py").write_text("VALUE = 2\n", encoding="utf-8")
    result = _run_node(tree, environment, *_NODE_ARGUMENTS)
    assert result.returncode != 0
    assert "Source membership differs" in result.stderr
    assert not command_log.exists()


@pytest.mark.parametrize(
    ("variable", "value", "expected"),
    [
        ("GENERATION_GIT_COMMIT", "c" * 40, "source commit " + "c" * 40),
        ("GENERATION_GIT_COMMIT", "short", "exact launch commit"),
        ("GENERATION_SOURCE_SHA256", "d" * 64, "matching the submission fingerprint"),
        ("GENERATION_SOURCE_SHA256", "not-a-digest", "GENERATION_SOURCE_SHA256 launch fingerprint"),
    ],
)
def test_worker_rejects_mismatched_launch_commit_or_fingerprint(
    native_worker: tuple[Path, dict[str, str], Path], variable: str, value: str, expected: str
) -> None:
    """The snapshot's commit and fingerprint must equal the values admitted at submission."""
    tree, environment, command_log = native_worker
    environment[variable] = value
    result = _run_node(tree, environment, *_NODE_ARGUMENTS)
    assert result.returncode != 0
    assert expected in result.stderr
    assert not command_log.exists()


def test_comsol_version_rejects_zero_exit_with_invalid_launcher_output(
    native_worker: tuple[Path, dict[str, str], Path],
) -> None:
    """Do not report a ready solver when its launcher prints an error and exits zero."""
    repository, environment, _command_log = native_worker
    fake_comsol = Path(environment["PATH"].split(os.pathsep)[0]) / "comsol"
    _write_executable(fake_comsol, 'printf "Invalid Configuration Location\\n"\n')
    result = subprocess.run(  # noqa: S603 -- test-owned executable and controlled argument vector; no shell interpolation
        [
            "/bin/bash",
            "-c",
            'source "$1"; generation_comsol_version comsol',
            "bash",
            str(repository / "scripts" / "generation_prerequisites.sh"),
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "invalid v6.4 evidence" in result.stderr
    assert not tuple(Path(environment["TMPDIR"]).iterdir())


@pytest.mark.parametrize(
    ("variable", "value", "expected"),
    [
        ("STORAGE_ROOT", None, "explicit physical STORAGE_ROOT directory"),
        ("STORAGE_ROOT", "relative/storage", "explicit physical STORAGE_ROOT directory"),
        ("STORAGE_ROOT", "{tmp}/other-storage", "explicit physical STORAGE_ROOT directory"),
        ("RUNTIME_ROOT", None, "explicit physical RUNTIME_ROOT directory"),
        ("SOURCE_TREE_ID", "f" * 12 + "-" + "f" * 12, "as REPOSITORY and PROJECT_ROOT"),
        ("PROJECT_ROOT", "{tmp}/elsewhere", "as REPOSITORY and PROJECT_ROOT"),
        ("GENERATION_NATIVE_VENV", "{tmp}/other-venv", "native Generation venv RUNTIME_ROOT/venvs/native"),
    ],
)
def test_worker_rejects_roots_that_are_not_explicit_and_consistent(
    native_worker: tuple[Path, dict[str, str], Path], tmp_path: Path, variable: str, value: str | None, expected: str
) -> None:
    """Storage, runtime, tree identity and the native environment are explicit; nothing is inferred beside the tree."""
    tree, environment, command_log = native_worker
    if value is None:
        del environment[variable]
    else:
        environment[variable] = value.format(tmp=tmp_path)
    result = _run_node(tree, environment, *_NODE_ARGUMENTS)
    assert result.returncode != 0
    assert expected in result.stderr
    assert not command_log.exists()


def test_worker_rejects_a_tree_outside_the_runtime_source_root(
    native_worker: tuple[Path, dict[str, str], Path], frozen_project: FrozenProject
) -> None:
    """A copy of the tree elsewhere is not the tree the launcher admitted."""
    tree, environment, command_log = native_worker
    copy = frozen_project.root / "copied" / tree.name
    shutil.copytree(tree, copy, symlinks=True)
    environment["PROJECT_ROOT"] = str(copy)
    result = _run_node(copy, environment, *_NODE_ARGUMENTS)
    assert result.returncode != 0
    assert "as REPOSITORY and PROJECT_ROOT" in result.stderr
    assert not command_log.exists()


def test_worker_requires_an_executable_native_python(native_worker: tuple[Path, dict[str, str], Path], frozen_project: FrozenProject) -> None:
    """The native venv must provide an executable interpreter before any project Python runs."""
    tree, environment, command_log = native_worker
    (frozen_project.venv / "bin/python").chmod(0o644)
    result = _run_node(tree, environment, *_NODE_ARGUMENTS)
    assert result.returncode != 0
    assert "native Generation venv" in result.stderr
    assert not command_log.exists()


def _node_directory(tmp_path: Path) -> Path:
    directory = tmp_path / "node-directory"
    (directory / "tickets").mkdir(parents=True)
    return directory


def test_node_worker_mode_runs_the_python_worker_in_the_foreground(native_worker: tuple[Path, dict[str, str], Path], tmp_path: Path) -> None:
    """The launcher admits the tree, then hands the node directory and core count to the generic worker."""
    tree, environment, command_log = native_worker
    directory = _node_directory(tmp_path)
    result = _run_node(tree, environment, "node-worker", str(directory), "2")
    assert result.returncode == 0, result.stderr
    assert "check=source-tree status=pass" in result.stdout
    assert f"run-node-worker {directory} --cores 2" in command_log.read_text(encoding="utf-8")


def test_node_worker_mode_returns_the_worker_status(
    native_worker: tuple[Path, dict[str, str], Path], tmp_path: Path, frozen_project: FrozenProject
) -> None:
    """A non-zero worker status (invalid ticket, error) is the status of the Slurm batch script."""
    tree, environment, _command_log = native_worker
    _write_executable(frozen_project.venv / "bin" / "python", "exit 3\n")
    assert _run_node(tree, environment, "node-worker", str(_node_directory(tmp_path)), "2").returncode == 3


@pytest.mark.parametrize(
    ("arguments", "environment_override"),
    [
        (("{node}",), {}),
        (("{node}", "2", "extra"), {}),
        (("{node}", "4"), {}),
        (("{node}", "0"), {}),
        (("{node}", "two"), {}),
        (("relative/node", "2"), {}),
        (("{tmp}/missing", "2"), {}),
        (("{node}", "2"), {"SLURM_ARRAY_TASK_ID": "1"}),
    ],
)
def test_node_worker_mode_rejects_inconsistent_directory_and_allocation(
    native_worker: tuple[Path, dict[str, str], Path], tmp_path: Path, arguments: tuple[str, ...], environment_override: dict[str, str]
) -> None:
    """Directory, core count, and Slurm allocation must agree before any project Python runs."""
    tree, environment, command_log = native_worker
    directory = _node_directory(tmp_path)
    environment.update(environment_override)
    result = _run_node(tree, environment, "node-worker", *(argument.format(node=directory, tmp=tmp_path) for argument in arguments))
    assert result.returncode == 2
    assert "Node worker" in result.stderr or "Usage" in result.stderr
    assert not command_log.exists()


def test_node_worker_mode_rejects_a_symlinked_directory(native_worker: tuple[Path, dict[str, str], Path], tmp_path: Path) -> None:
    """The node directory itself must not be a symlink."""
    tree, environment, command_log = native_worker
    link = tmp_path / "node-link"
    link.symlink_to(_node_directory(tmp_path))
    assert _run_node(tree, environment, "node-worker", str(link), "2").returncode == 2
    assert not command_log.exists()


def test_node_worker_mode_forwards_term_to_the_python_worker_and_returns_its_status(
    native_worker: tuple[Path, dict[str, str], Path], tmp_path: Path, frozen_project: FrozenProject
) -> None:
    """TERM to the batch script reaches the Python worker process, and the script waits for it and passes its status on."""
    tree, environment, command_log = native_worker
    _write_executable(
        frozen_project.venv / "bin" / "python",
        '[[ "${1:-}" == -c ]] && exit 0\n'  # the node start's venv check, not the worker
        'trap \'printf "TERM-received\\n" >> "$COMMAND_LOG"; exit 143\' TERM\nprintf "ready\\n" >> "$COMMAND_LOG"\nwhile :; do sleep 0.05; done\n',
    )
    process = subprocess.Popen(  # noqa: S603 -- test-owned executable and controlled argument vector; no shell interpolation
        ["/bin/bash", str(tree / "scripts" / "generation_node.sh"), str(tree), "node-worker", str(_node_directory(tmp_path)), "2"],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and not (command_log.exists() and "ready" in command_log.read_text(encoding="utf-8")):
            assert process.poll() is None, process.communicate()
            time.sleep(0.05)
        process.send_signal(15)
        process.communicate(timeout=30)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
    assert process.returncode == 143
    assert "TERM-received" in command_log.read_text(encoding="utf-8")


def test_node_job_reports_its_preflight_once_and_its_case_steps_check_silently(
    native_worker: tuple[Path, dict[str, str], Path], tmp_path: Path
) -> None:
    """The node worker prints the complete preflight report at node start; the case steps of that job print nothing while the checks pass."""
    tree, environment, _command_log = native_worker
    node = _run_node(tree, environment, "node-worker", str(_node_directory(tmp_path)), "2")
    assert node.returncode == 0, node.stderr
    for check in ("source-tree", "command:module", "command:mktemp", "python-version:", "comsol-version:", "Generation-venv-runtime"):
        assert f"check={check}" in node.stdout
    assert node.stdout.count("check=comsol-version:") == 1
    step = _run_node(tree, environment, *_NODE_ARGUMENTS)
    assert step.returncode == 0, step.stderr
    assert "Preflight" not in step.stdout
    assert "Preflight" not in step.stderr


def test_case_step_stays_loud_about_a_failing_check_although_passing_checks_are_silent(native_worker: tuple[Path, dict[str, str], Path]) -> None:
    """A missing prerequisite is reported on stderr and ends the step before its Python worker starts."""
    tree, environment, command_log = native_worker
    environment["GENERATION_COMSOL_EXECUTABLE"] = "missing-comsol"
    result = _run_node(tree, environment, *_NODE_ARGUMENTS)
    assert result.returncode != 0
    assert "prerequisite missing: missing-comsol" in result.stderr
    assert "Preflight" not in result.stdout
    assert "run-campaign-case" not in command_log.read_text(encoding="utf-8")


def test_case_step_scratch_names_the_step_inside_a_node_job(native_worker: tuple[Path, dict[str, str], Path]) -> None:
    """A campaign case that runs as a step of a node job keeps its scratch attributable to job and step; without a step the name is unchanged."""
    tree, environment, command_log = native_worker
    environment["SLURM_STEP_ID"] = "5"
    assert _run_node(tree, environment, *_NODE_ARGUMENTS).returncode == 0
    assert re.search(r"initialize-worker-workspace \S*/vp2-generation-12345-5\.[A-Za-z0-9]{6} ", command_log.read_text(encoding="utf-8"))
    command_log.unlink()
    del environment["SLURM_STEP_ID"]
    assert _run_node(tree, environment, *_NODE_ARGUMENTS).returncode == 0
    assert re.search(r"initialize-worker-workspace \S*/vp2-generation-12345\.[A-Za-z0-9]{6} ", command_log.read_text(encoding="utf-8"))


def test_worker_start_sweeps_stale_scratch_of_step_named_roots(native_worker: tuple[Path, dict[str, str], Path]) -> None:
    """The sweep recognizes job-step scratch names: a finished job's root goes, an active node job's root stays."""
    tree, environment, _command_log = native_worker
    scratch = Path(environment["TMPDIR"])
    roots = {}
    for name in ("vp2-generation-111-3.AAAAAA", "vp2-generation-222-4.BBBBBB"):
        directory = scratch / name
        directory.mkdir()
        (directory / ".generation-worker-workspace.json").write_text("{}", encoding="utf-8")
        Path(f"{directory}.lock").write_text("", encoding="utf-8")
        roots[name] = directory
    environment["FAKE_SQUEUE_ACTIVE"] = "222\n"
    assert _run_node(tree, environment, *_NODE_ARGUMENTS).returncode == 0
    assert not roots["vp2-generation-111-3.AAAAAA"].exists()
    assert roots["vp2-generation-222-4.BBBBBB"].exists()
