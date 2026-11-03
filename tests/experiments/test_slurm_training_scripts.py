"""Exercise shared ML submission and worker contracts without contacting Slurm.

Responsibilities:
  - Check submission arguments, resource admission, worker dispatch, and SIF drift rejection
  - Check that every job runs from a frozen source tree: current or bound tree choice, explicit transition, lease, and Git-free worker verification
  - Check that the trained stage selection reaches the config preflight that discovers the bound tree
  - Check that the Apptainer executor admits exactly the frozen-tree overlap with the runtime root
  - Check that the job's Slurm logs are named once, reach the executor, and are linked into the recorded folders without changing its exit status

Design principles:
  - Fake executors exercise worker boundaries without contacting Slurm
  - The source freezing tool runs for real against a disposable Git checkout; only Slurm, the queue and Apptainer are inert

This module does NOT:
  - Submit real jobs or load production datasets
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from support.fake_git import install_failing_git
from support.source_tree import tampered


@dataclass(frozen=True)
class _Harness:
    repo: Path
    storage: Path
    runtime: Path
    environment: dict[str, str]
    sbatch_capture: Path
    executor_capture: Path
    provenance_capture: Path
    preflight_capture: Path
    sbatch_environment_capture: Path
    job_log_capture: Path

    @property
    def trees(self) -> Path:
        """Return the runtime root of frozen source trees."""
        return self.runtime / "sources"


def _unfreeze(trees: Path) -> None:
    """Restore owner write access to frozen trees so the temporary folder can be removed."""
    for current, directories, _files in os.walk(trees):
        for name in directories:
            path = Path(current) / name
            if not path.is_symlink():
                path.chmod(stat.S_IMODE(path.stat().st_mode) | stat.S_IWUSR)


@pytest.fixture
def harness(request: pytest.FixtureRequest) -> _Harness:
    """Build a disposable Git checkout with inert sbatch, queue and Apptainer boundaries."""
    temporary = TemporaryDirectory(prefix="slurm-ml-test-", dir="/tmp")
    tmp_path = Path(temporary.name)

    def cleanup() -> None:
        _unfreeze(tmp_path / "project" / "runtime" / "sources")
        temporary.cleanup()

    request.addfinalizer(cleanup)  # noqa: PT021 -- clean the checkout if fixture setup fails
    source_scripts = Path(__file__).resolve().parents[2] / "scripts"
    repo = tmp_path / "project" / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    storage = repo.parent / "storage"
    runtime = repo.parent / "runtime"
    storage.mkdir()
    (runtime / "containers").mkdir(parents=True)
    (runtime / "containers" / "grainlegumes-pino-drying.sif").write_bytes(b"test image")
    for name in ("slurm_ml.sh", "slurm_ml_worker.sh", "source_fingerprint.py"):
        shutil.copy2(source_scripts / name, scripts / name)
    # The fake executor must hide host GPUs as the CPU Apptainer path does; its preflight answer ends in the bound tree.
    executor = scripts / "apptainer_exec.sh"
    executor.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'if [[ "${1:-}" == python && "${2:-}" == -m && "${3:-}" == src.experiments.cli.cli_config_preflight ]]; then\n'
        '  printf \'%s\\0\' "$@" > "${PREFLIGHT_CAPTURE}"\n'
        '  if [[ "$4" == optuna ]]; then\n'
        '    printf \'optuna\\tsteady_flow\\tconfig.yaml\\tstudy\\t%s%s\\n\' "${PREFLIGHT_BOUND_TREE:--}" "${PREFLIGHT_EXTRA:-}"\n'
        "  else\n"
        '    printf \'experiment\\tsteady_flow\\tconfig.yaml\\trun\\t%s%s\\n\' "${PREFLIGHT_BOUND_TREE:--}" "${PREFLIGHT_EXTRA:-}"\n'
        "  fi\n"
        "else\n"
        '  printf \'%s\\0\' "$@" > "${EXECUTOR_CAPTURE}"\n'
        '  printf \'%s\' "${APPTAINERENV_ML_SLURM_PROVENANCE:-}" > "${PROVENANCE_CAPTURE}"\n'
        '  printf \'%s\' "${JOB_LOG_FILES:-}" > "${JOB_LOG_CAPTURE}"\n'
        '  if [[ "${1:-}" == python && "${2:-}" == -c ]]; then\n'
        '    CUDA_VISIBLE_DEVICES="" ML_SLURM_PROVENANCE="${APPTAINERENV_ML_SLURM_PROVENANCE}" \\\n'
        '      PYTHONPATH="${SOURCE_REPO}" "${TEST_PYTHON}" -c "$3" "$4"\n'
        "  fi\n"
        "fi\n",
        encoding="utf-8",
    )
    executor.chmod(0o755)
    (repo / "config.yaml").write_text("task: steady_flow\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)  # noqa: S603, S607 -- test-owned executable and controlled argument vector; no shell interpolation
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)  # noqa: S603, S607 -- test-owned executable and controlled argument vector; no shell interpolation
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"], check=True)  # noqa: S603, S607 -- test-owned executable and controlled argument vector; no shell interpolation
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)  # noqa: S603, S607 -- test-owned executable and controlled argument vector; no shell interpolation
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)  # noqa: S603, S607 -- test-owned executable and controlled argument vector; no shell interpolation
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sbatch = bin_dir / "sbatch"
    sbatch.write_text(
        '#!/usr/bin/env bash\nprintf \'%s\\0\' "$@" > "${SBATCH_CAPTURE}"\nenv -0 > "${SBATCH_ENVIRONMENT_CAPTURE}"\nprintf \'12345\\n\'\n',
        encoding="utf-8",
    )
    sbatch.chmod(0o755)
    # The queue is empty: pruning removes only old trees nobody leased.
    squeue = bin_dir / "squeue"
    squeue.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    squeue.chmod(0o755)
    sbatch_capture = tmp_path / "sbatch.args"
    executor_capture = tmp_path / "executor.args"
    provenance_capture = tmp_path / "provenance.json"
    preflight_capture = tmp_path / "preflight.args"
    sbatch_environment_capture = tmp_path / "sbatch.env"
    job_log_capture = tmp_path / "job-log-files"
    environment = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "STORAGE_ROOT": str(storage),
        "RUNTIME_ROOT": str(runtime),
        "SBATCH_CAPTURE": str(sbatch_capture),
        "EXECUTOR_CAPTURE": str(executor_capture),
        "PROVENANCE_CAPTURE": str(provenance_capture),
        "PREFLIGHT_CAPTURE": str(preflight_capture),
        "SBATCH_ENVIRONMENT_CAPTURE": str(sbatch_environment_capture),
        "JOB_LOG_CAPTURE": str(job_log_capture),
        "SOURCE_REPO": str(Path(__file__).resolve().parents[2]),
        "TEST_PYTHON": sys.executable,
    }
    return _Harness(
        repo,
        storage,
        runtime,
        environment,
        sbatch_capture,
        executor_capture,
        provenance_capture,
        preflight_capture,
        sbatch_environment_capture,
        job_log_capture,
    )


def _args(path: Path) -> list[str]:
    """Decode exact argument boundaries captured by an inert shell executable."""
    return [part.decode() for part in path.read_bytes().split(b"\0") if part]


def _submit(harness: _Harness, *arguments: str) -> subprocess.CompletedProcess[str]:
    """Call the real submit wrapper against the inert sbatch executable."""
    return subprocess.run(  # noqa: S603 -- test-owned executable and controlled argument vector; no shell interpolation
        [str(harness.repo / "scripts" / "slurm_ml.sh"), *arguments],
        cwd=harness.repo,
        env=harness.environment,
        text=True,
        capture_output=True,
        check=False,
    )


def _resources(mode: str) -> list[str]:
    """Return one small test-owned CPU or GPU allocation request."""
    partition = "standard" if mode == "cpu" else "gpu"
    return ["--mode", mode, "--partition", partition, "--cpus-per-task", "2", "--mem", "4G", "--time", "00:10:00"]


def _sbatch_environment(harness: _Harness) -> dict[str, str]:
    """Decode the environment the inert sbatch received, which the job inherits through ``--export=ALL``."""
    entries = [part.decode() for part in harness.sbatch_environment_capture.read_bytes().split(b"\0") if part]
    return dict(entry.split("=", 1) for entry in entries)


def _submitted_tree(harness: _Harness) -> Path:
    """Return the frozen source tree the last submission runs from."""
    return Path(_sbatch_environment(harness)["PROJECT_ROOT"])


def _worker_arguments(harness: _Harness) -> list[str]:
    """Return the positional worker arguments of the last submission."""
    submission = _args(harness.sbatch_capture)
    start = next(index for index, value in enumerate(submission) if value.endswith("/scripts/slurm_ml_worker.sh"))
    return submission[start + 1 :]


def _run_worker(
    harness: _Harness,
    *,
    gpu: bool,
    arguments: list[str] | None = None,
    environment_override: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Replay a Slurm-spooled worker copy of the submitted tree inside a fake allocation without any working Git."""
    submitted = _sbatch_environment(harness)
    tree = Path(submitted["PROJECT_ROOT"])
    spooled_worker = harness.runtime / "slurm_script"
    shutil.copy2(tree / "scripts" / "slurm_ml_worker.sh", spooled_worker)
    failing_git = install_failing_git(harness.runtime.parent.parent / "failing-bin")
    environment = {
        **{key: value for key, value in harness.environment.items() if key != "ML_SOURCE_TRANSITION"},
        **{key: submitted[key] for key in ("PROJECT_ROOT", "SOURCE_TREE_ID", "STORAGE_ROOT", "RUNTIME_ROOT", "ML_JOB_LOG_TEMPLATES")},
        "PATH": f"{failing_git.parent}{os.pathsep}{harness.environment['PATH']}",
        "SLURM_JOB_ID": "12345",
        "SLURM_JOB_PARTITION": "gpu" if gpu else "standard",
        "SLURM_CPUS_PER_TASK": "2",
    }
    if "ML_SOURCE_TRANSITION" in submitted:
        environment["ML_SOURCE_TRANSITION"] = submitted["ML_SOURCE_TRANSITION"]
    environment.pop("SLURM_JOB_GPUS", None)
    environment.pop("SLURM_STEP_GPUS", None)
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    if gpu:
        environment.update({"SLURM_JOB_GPUS": "0", "CUDA_VISIBLE_DEVICES": "0"})
    environment.update(environment_override or {})
    return subprocess.run(  # noqa: S603 -- test-owned executable and controlled argument vector; no shell interpolation
        [str(spooled_worker), *(arguments if arguments is not None else _worker_arguments(harness))],
        cwd=tree,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def _git_calls(harness: _Harness) -> str:
    """Return the Git invocations a replayed worker attempted; the stub fails every one of them."""
    log = harness.runtime.parent.parent / "failing-bin" / "git-calls.log"
    return log.read_text(encoding="utf-8") if log.exists() else ""


def test_cpu_submission_preserves_training_arguments(harness: _Harness) -> None:
    """CPU Slurm construction has no GRES and forwards every training override unchanged."""
    result = _submit(
        harness,
        *_resources("cpu"),
        "train",
        "config.yaml",
        "--epochs",
        "5",
        "--rollout-length",
        "8",
        "--train-dataset",
        "synthetic_train",
        "--lr-scale",
        "0.5",
        "--train-seconds",
        "25",
        "--revision",
        "fixed-loss",
        "--artifacts",
        "none",
    )

    assert result.returncode == 0, result.stderr
    assert "Slurm job ID: 12345" in result.stdout
    arguments = _args(harness.sbatch_capture)
    assert "--gres" not in " ".join(arguments)
    assert f"--output={harness.runtime}/logs/ml/slurm-%j.out" in arguments
    assert f"--error={harness.runtime}/logs/ml/slurm-%j.err" in arguments
    assert arguments[-17:] == [
        "/workspace/repo/config.yaml",
        "--epochs",
        "5",
        "--rollout-length",
        "8",
        "--train-dataset",
        "synthetic_train",
        "--lr-scale",
        "0.5",
        "--train-seconds",
        "25",
        "--revision",
        "fixed-loss",
        "--artifacts",
        "none",
        "--device",
        "cpu",
    ]
    assert _args(harness.preflight_capture) == [
        "python",
        "-m",
        "src.experiments.cli.cli_config_preflight",
        "train",
        "/workspace/repo/config.yaml",
        "--epochs",
        "5",
        "--rollout-length",
        "8",
        "--train-dataset",
        "synthetic_train",
        "--lr-scale",
        "0.5",
        "--train-seconds",
        "25",
        "--revision",
        "fixed-loss",
    ]
    worker_result = _run_worker(harness, gpu=False)
    assert worker_result.returncode == 0, worker_result.stderr
    assert _args(harness.executor_capture) == [
        "python",
        "-m",
        "src.experiments.cli.cli_train",
        *arguments[-17:],
    ]
    provenance = json.loads(harness.provenance_capture.read_text(encoding="utf-8"))
    assert provenance["mode"] == "cpu"
    assert provenance["job_id"] == "12345"
    assert provenance["gres"] == "none"
    assert provenance["source_transition"] == "none"
    assert _git_calls(harness) == ""
    assert provenance["sif_path"] == "/workspace/runtime/containers/grainlegumes-pino-drying.sif"


def test_config_combined_with_resume_is_rejected_before_submission(harness: _Harness) -> None:
    """Only the saved-config form takes --resume; a CONFIG continues by rerunning it, so the combination exits 2."""
    result = _submit(harness, *_resources("cpu"), "train", "config.yaml", "--resume", str(harness.storage / "run"))

    assert result.returncode == 2
    assert "--resume takes a stage directory instead of CONFIG" in result.stderr
    assert not harness.sbatch_capture.exists()


def test_train_stage_selection_reaches_the_preflight_but_the_artifact_selection_does_not(harness: _Harness) -> None:
    """Only the trained stages decide which leaves bind the source tree, so the preflight sees --train-stages alone."""
    result = _submit(harness, *_resources("cpu"), "train", "config.yaml", "--artifacts", "all", "--train-stages", "b,a1")

    assert result.returncode == 0, result.stderr
    assert _args(harness.preflight_capture) == [
        "python",
        "-m",
        "src.experiments.cli.cli_config_preflight",
        "train",
        "/workspace/repo/config.yaml",
        "--train-stages",
        "b,a1",
    ]
    assert _args(harness.sbatch_capture)[-7:] == [
        "/workspace/repo/config.yaml",
        "--artifacts",
        "all",
        "--train-stages",
        "b,a1",
        "--device",
        "cpu",
    ]


def test_gpu_submission_uses_named_gres_and_worker_forwards_strict_cuda(harness: _Harness) -> None:
    """Slurm selects a named GPU; no host physical index enters the request."""
    result = _submit(harness, *_resources("gpu"), "--gres", "gpu:rtx6000ada:1", "train", "config.yaml")

    assert result.returncode == 0, result.stderr
    assert _args(harness.preflight_capture) == [
        "python",
        "-m",
        "src.experiments.cli.cli_config_preflight",
        "train",
        "/workspace/repo/config.yaml",
    ]
    arguments = _args(harness.sbatch_capture)
    assert "--gres=gpu:rtx6000ada:1" in arguments
    assert not any(argument.startswith("--gpu-bind") for argument in arguments)
    assert arguments[-3:] == ["/workspace/repo/config.yaml", "--device", "cuda"]
    worker_result = _run_worker(harness, gpu=True)
    assert worker_result.returncode == 0, worker_result.stderr
    assert _args(harness.executor_capture) == [
        "--gpu",
        "python",
        "-m",
        "src.experiments.cli.cli_train",
        "/workspace/repo/config.yaml",
        "--device",
        "cuda",
    ]
    assert json.loads(harness.provenance_capture.read_text(encoding="utf-8"))["gres"] == "gpu:rtx6000ada:1"


def test_resume_only_submission_binds_the_run_owned_config(harness: _Harness) -> None:
    """The concise saved-run form sends its exact config path through worker hashing."""
    run_dir = harness.storage / "saved run with spaces"
    run_dir.mkdir()
    config = run_dir / "config.yaml"
    config.write_text("resolved: true\n", encoding="utf-8")

    result = _submit(
        harness,
        *_resources("cpu"),
        "train",
        "--resume",
        str(run_dir),
        "--max-epochs",
        "9",
    )

    assert result.returncode == 0, result.stderr
    arguments = _args(harness.sbatch_capture)
    assert arguments[-7:] == [
        "/workspace/storage/saved run with spaces/config.yaml",
        "--resume",
        "/workspace/storage/saved run with spaces",
        "--max-epochs",
        "9",
        "--device",
        "cpu",
    ]
    assert _args(harness.preflight_capture) == [
        "python",
        "-m",
        "src.experiments.cli.cli_config_preflight",
        "train",
        "/workspace/storage/saved run with spaces/config.yaml",
        "--resume",
        "/workspace/storage/saved run with spaces",
    ]
    worker_result = _run_worker(harness, gpu=False)
    assert worker_result.returncode == 0, worker_result.stderr
    assert _args(harness.executor_capture) == [
        "python",
        "-m",
        "src.experiments.cli.cli_train",
        "/workspace/storage/saved run with spaces/config.yaml",
        "--resume",
        "/workspace/storage/saved run with spaces",
        "--max-epochs",
        "9",
        "--device",
        "cpu",
    ]
    provenance = json.loads(harness.provenance_capture.read_text(encoding="utf-8"))
    assert provenance["config_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()


@pytest.mark.parametrize("mode", ["cpu", "gpu"])
def test_optuna_submission_preserves_arguments_and_worker_dispatch(harness: _Harness, mode: str) -> None:
    """Optuna uses the shared allocation and passes study options to its CLI."""
    resources = _resources(mode)
    if mode == "gpu":
        resources += ["--gres", "gpu:rtx6000ada:1"]
    result = _submit(
        harness,
        *resources,
        "optuna",
        "config.yaml",
        "--epochs",
        "5",
        "--train-dataset",
        "synthetic_train",
        "--n-trials",
        "2",
        "--revision",
        "short-study",
    )
    assert result.returncode == 0, result.stderr
    assert _args(harness.preflight_capture) == [
        "python",
        "-m",
        "src.experiments.cli.cli_config_preflight",
        "optuna",
        "/workspace/repo/config.yaml",
        "--epochs",
        "5",
        "--train-dataset",
        "synthetic_train",
        "--revision",
        "short-study",
    ]
    arguments = _args(harness.sbatch_capture)
    if mode == "cpu":
        assert not any(argument.startswith("--gres=") for argument in arguments)
    else:
        assert "--gres=gpu:rtx6000ada:1" in arguments
    worker_result = _run_worker(harness, gpu=mode == "gpu")
    assert worker_result.returncode == 0, worker_result.stderr
    expected = [
        "/workspace/repo/config.yaml",
        "--epochs",
        "5",
        "--train-dataset",
        "synthetic_train",
        "--n-trials",
        "2",
        "--revision",
        "short-study",
        "--device",
        "cpu" if mode == "cpu" else "cuda",
    ]
    assert _args(harness.executor_capture) == [
        *(["--gpu"] if mode == "gpu" else []),
        "python",
        "-m",
        "src.experiments.cli.cli_optuna",
        *expected,
    ]


@pytest.mark.parametrize("mode", ["cpu", "gpu"])
def test_artifact_submission_preserves_arguments_and_worker_dispatch(harness: _Harness, mode: str) -> None:
    """Artifact selection and publication flags reach the existing Python CLI."""
    resources = _resources(mode)
    if mode == "gpu":
        resources += ["--gres", "gpu:v100:1"]
    run_dir = harness.storage / "run with spaces"
    result = _submit(
        harness,
        *resources,
        "artifacts",
        "--run-dir",
        str(run_dir),
        "--run-dir",
        str(harness.storage / "second"),
        "--rebuild",
        "--evaluation-spatial-stride",
        "2",
    )
    assert result.returncode == 0, result.stderr
    arguments = _args(harness.sbatch_capture)
    if mode == "cpu":
        assert not any(argument.startswith("--gres=") for argument in arguments)
    else:
        assert "--gres=gpu:v100:1" in arguments
    worker_result = _run_worker(harness, gpu=mode == "gpu")
    assert worker_result.returncode == 0, worker_result.stderr
    assert _args(harness.executor_capture) == [
        *(["--gpu"] if mode == "gpu" else []),
        "python",
        "-m",
        "src.experiments.cli.cli_build_artifacts",
        "--run-dir",
        "/workspace/storage/run with spaces",
        "--run-dir",
        "/workspace/storage/second",
        "--rebuild",
        "--evaluation-spatial-stride",
        "2",
        "--device",
        "cpu" if mode == "cpu" else "cuda",
    ]


@pytest.mark.parametrize(
    ("operation", "arguments"),
    [
        ("optuna", ["config.yaml", "--device", "cuda"]),
        ("optuna", ["config.yaml", "--resume", "run"]),
        ("artifacts", ["--task", "steady_flow", "--device", "cuda"]),
        ("artifacts", ["--run-dir", "/workspace/storage/../runtime/run"]),
        ("artifacts", ["--rebuild"]),
    ],
)
def test_invalid_shared_ml_arguments_fail_before_submission(
    harness: _Harness,
    operation: str,
    arguments: list[str],
) -> None:
    """Wrong device and invalid durable-path arguments never reach Slurm."""
    result = _submit(harness, *_resources("cpu"), operation, *arguments)
    assert result.returncode != 0
    assert not harness.sbatch_capture.exists()


@pytest.mark.parametrize(
    "arguments",
    [
        [*_resources("cpu"), "--gres", "gpu:v100:1", "train", "config.yaml"],
        [*_resources("gpu"), "train", "config.yaml"],
        [*_resources("gpu"), "--gres", "gpu:0:1", "train", "config.yaml"],
        [*_resources("gpu"), "--gres", "gpu:v100:1", "train", "config.yaml", "--device", "cpu"],
        [*_resources("cpu"), "train", "--resume"],
        [*_resources("cpu"), "train", "config.yaml", "--train-seconds", "0"],
        [*_resources("cpu"), "train", "config.yaml", "--rollout-length", "0"],
        *[[*_resources("cpu"), "train", "config.yaml", "--lr-scale", value] for value in ("0", "0.0", "-1", "nan", "x")],
        [*_resources("cpu"), "train", "config.yaml", "--lr-scale", "0.5", "--lr-scale", "0.25"],
        [*_resources("cpu"), "optuna", "config.yaml", "--lr-scale", "0.5"],
        [*_resources("cpu"), "optuna", "config.yaml", "--rollout-length", "8"],
    ],
)
def test_invalid_resource_or_training_arguments_fail_before_submission(harness: _Harness, arguments: list[str]) -> None:
    """Reject mismatched resources and malformed resume before any sbatch call."""
    result = _submit(harness, *arguments)

    assert result.returncode != 0
    assert not harness.sbatch_capture.exists()


def test_overlapping_roots_fail_before_submission(harness: _Harness) -> None:
    """Runtime logs cannot be placed inside source or durable storage."""
    harness.environment["RUNTIME_ROOT"] = str(harness.repo / "runtime")

    result = _submit(harness, *_resources("cpu"), "probe")

    assert result.returncode != 0
    assert "must not overlap" in result.stderr
    assert not harness.sbatch_capture.exists()


def test_storage_path_traversal_fails_before_submission(harness: _Harness) -> None:
    """Container paths cannot escape durable storage through parent segments."""
    result = _submit(
        harness,
        *_resources("cpu"),
        "train",
        "--resume",
        "/workspace/storage/../runtime/run",
    )

    assert result.returncode != 0
    assert not harness.sbatch_capture.exists()


@pytest.mark.parametrize("operation", ["train", "optuna"])
@pytest.mark.parametrize("revision", ["", "-lead", ".dot", "a/b", "with space", "x" * 65])
def test_revision_label_must_be_a_safe_name_before_submission(harness: _Harness, operation: str, revision: str) -> None:
    """A revision becomes a directory suffix, so unsafe labels never reach Slurm."""
    result = _submit(harness, *_resources("cpu"), operation, "config.yaml", "--revision", revision)

    assert result.returncode != 0
    assert "--revision" in result.stderr
    assert not harness.sbatch_capture.exists()


def test_worker_rejects_sif_content_drift_with_unchanged_size_and_mtime(harness: _Harness) -> None:
    """A queued job must run the exact submitted SIF bytes."""
    assert _submit(harness, *_resources("cpu"), "probe").returncode == 0
    sif = harness.runtime / "containers" / "grainlegumes-pino-drying.sif"
    original_stat = sif.stat()
    sif.write_bytes(b"evil image")
    os.utime(sif, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    result = _run_worker(harness, gpu=False)

    assert result.returncode != 0
    assert "SIF content changed" in result.stderr
    assert not harness.executor_capture.exists()


def test_worker_rejects_missing_gpu_allocation(harness: _Harness) -> None:
    """The worker checks the Slurm allocation before execution."""
    assert _submit(harness, *_resources("gpu"), "--gres", "gpu:rtxa6000:1", "probe").returncode == 0

    missing_gpu = _run_worker(harness, gpu=False)

    assert missing_gpu.returncode != 0
    assert not harness.executor_capture.exists()


def test_probe_uses_same_apptainer_executor_without_dataset_access(harness: _Harness) -> None:
    """Acceptance probes traverse the same worker and executor as training."""
    assert _submit(harness, *_resources("cpu"), "probe").returncode == 0
    result = _run_worker(harness, gpu=False)

    assert result.returncode == 0, result.stderr
    assert _args(harness.executor_capture)[:2] == ["python", "-c"]
    header = result.stdout.splitlines()[:3]
    assert header[0].startswith("job 12345 | partition standard | gres none")
    assert "2 CPUs" in header[0]
    assert "4G" in header[0]
    assert "00:10:00" in header[0]
    assert _submitted_tree(harness).name in header[1]
    assert "transition none" in header[1]
    assert re.search(r"commit [0-9a-f]{12}\b", header[1])
    assert "grainlegumes-pino-drying.sif" in header[2]
    assert re.search(r"sha256 [0-9a-f]{12}$", header[2])
    assert "/" not in header[2]
    assert not result.stdout.startswith(("Slurm job:", "SIF SHA256:"))


_OTHER_TREE = "aaaaaaaaaaaa-bbbbbbbbbbbb"


def _live_state(harness: _Harness) -> tuple[str, str]:
    """Return commit and source fingerprint of the live checkout the launcher freezes."""
    commit = subprocess.run(  # noqa: S603 -- test-owned executable and controlled argument vector
        ["git", "-C", str(harness.repo), "rev-parse", "HEAD"],  # noqa: S607 -- Git resolved from PATH
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    fingerprint = subprocess.run(  # noqa: S603 -- test-owned interpreter and controlled argument vector
        [sys.executable, str(harness.repo / "scripts" / "source_fingerprint.py"), str(harness.repo)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return commit, fingerprint


def _train(harness: _Harness, *extra: str, operation: str = "train") -> subprocess.CompletedProcess[str]:
    """Submit one CPU training or Optuna job of the fixture config."""
    return _submit(harness, *_resources("cpu"), operation, "config.yaml", *extra)


def test_new_unit_runs_from_a_frozen_tree_of_the_current_code_and_is_leased(harness: _Harness) -> None:
    """A unit without history freezes the current checkout, submits the tree's worker and leases the tree for the job."""
    result = _train(harness)

    assert result.returncode == 0, result.stderr
    commit, fingerprint = _live_state(harness)
    environment = _sbatch_environment(harness)
    tree = _submitted_tree(harness)
    assert tree == harness.trees / f"{commit[:12]}-{fingerprint[:12]}"
    assert environment["SOURCE_TREE_ID"] == tree.name
    assert environment["STORAGE_ROOT"] == str(harness.storage)
    assert environment["RUNTIME_ROOT"] == str(harness.runtime)
    assert "ML_SOURCE_TRANSITION" not in environment
    submission = _args(harness.sbatch_capture)
    assert f"--chdir={tree}" in submission
    assert str(tree / "scripts" / "slurm_ml_worker.sh") in submission
    worker_arguments = _worker_arguments(harness)
    assert worker_arguments[2:4] == [commit, fingerprint]
    assert worker_arguments[4] == hashlib.sha256((tree / "config.yaml").read_bytes()).hexdigest()
    assert not (tree / ".git").exists()
    assert stat.S_IMODE(tree.stat().st_mode) & 0o222 == 0
    assert (harness.storage / "source_snapshots" / tree.name / "snapshot.json").is_file()
    assert (harness.trees / ".leases" / tree.name / "12345").is_file()
    assert f"Source tree: {tree.name} (current code)" in result.stdout


def test_container_repository_path_is_accepted_and_read_from_the_tree(harness: _Harness) -> None:
    """A config named by its container path resolves to the frozen tree, not the checkout, for the submitted digest."""
    result = _submit(harness, *_resources("cpu"), "train", "/workspace/repo/config.yaml")

    assert result.returncode == 0, result.stderr
    assert _worker_arguments(harness)[4] == hashlib.sha256((_submitted_tree(harness) / "config.yaml").read_bytes()).hexdigest()
    assert _args(harness.sbatch_capture)[-3:] == ["/workspace/repo/config.yaml", "--device", "cpu"]


def test_launcher_prunes_unused_old_trees_before_freezing(harness: _Harness) -> None:
    """Pruning removes an unleased tree nobody touched for a day and keeps the freshly frozen one."""
    stale = harness.trees / _OTHER_TREE
    stale.mkdir(parents=True)
    old = time.time() - 2 * 86400
    os.utime(stale, (old, old))

    assert _submit(harness, *_resources("cpu"), "probe").returncode == 0

    assert not stale.exists()
    assert _submitted_tree(harness).is_dir()


@pytest.mark.parametrize("operation", ["train", "optuna"])
def test_existing_unit_continues_in_its_bound_tree_after_the_code_changed(harness: _Harness, operation: str) -> None:
    """The bound tree is reused, or restored from its snapshot once pruned, and the authored config comes from it."""
    assert _train(harness, operation=operation).returncode == 0
    first_tree = _submitted_tree(harness)
    first_arguments = _worker_arguments(harness)
    original_config = (harness.repo / "config.yaml").read_bytes()
    (harness.repo / "new-source.py").write_text("changed = True\n", encoding="utf-8")
    (harness.repo / "config.yaml").write_text("task: steady_flow\nchanged: true\n", encoding="utf-8")
    harness.environment["PREFLIGHT_BOUND_TREE"] = first_tree.name

    reused = _train(harness, operation=operation)

    assert reused.returncode == 0, reused.stderr
    assert _submitted_tree(harness) == first_tree
    assert "ML_SOURCE_TRANSITION" not in _sbatch_environment(harness)
    assert f"Source tree: {first_tree.name} (bound to the unit)" in reused.stdout
    assert _worker_arguments(harness)[2:5] == first_arguments[2:5]
    assert hashlib.sha256(original_config).hexdigest() == _worker_arguments(harness)[4]
    assert not (first_tree / "new-source.py").exists()

    _unfreeze(harness.trees)
    shutil.rmtree(first_tree)
    restored = _train(harness, operation=operation)

    assert restored.returncode == 0, restored.stderr
    assert _submitted_tree(harness) == first_tree
    assert (first_tree / "config.yaml").read_bytes() == original_config
    assert not (first_tree / "new-source.py").exists()
    assert _worker_arguments(harness)[2:5] == first_arguments[2:5]


def test_bound_tree_without_snapshot_fails_before_submission(harness: _Harness) -> None:
    """A unit whose recorded tree cannot be restored never silently runs other code."""
    harness.environment["PREFLIGHT_BOUND_TREE"] = _OTHER_TREE

    result = _train(harness)

    assert result.returncode == 1
    assert f"Cannot restore the source tree {_OTHER_TREE}" in result.stderr
    assert not harness.sbatch_capture.exists()


@pytest.mark.parametrize("operation", ["train", "optuna"])
def test_continue_with_current_code_switches_the_tree_and_marks_the_transition(harness: _Harness, operation: str) -> None:
    """The flag moves a bound unit to the current code; the marker reaches the job and its provenance."""
    assert _train(harness, operation=operation).returncode == 0
    first_tree = _submitted_tree(harness)
    (harness.repo / "new-source.py").write_text("changed = True\n", encoding="utf-8")
    harness.environment["PREFLIGHT_BOUND_TREE"] = first_tree.name

    result = _train(harness, "--continue-with-current-code", operation=operation)

    assert result.returncode == 0, result.stderr
    tree = _submitted_tree(harness)
    assert tree != first_tree
    assert (tree / "new-source.py").is_file()
    assert _sbatch_environment(harness)["ML_SOURCE_TRANSITION"] == "explicit"
    assert f"explicit transition from {first_tree.name}" in result.stdout
    assert "--continue-with-current-code" not in _args(harness.sbatch_capture)
    assert "--continue-with-current-code" not in _args(harness.preflight_capture)
    assert (harness.trees / ".leases" / tree.name / "12345").is_file()
    worker_result = _run_worker(harness, gpu=False)
    assert worker_result.returncode == 0, worker_result.stderr
    provenance = json.loads(harness.provenance_capture.read_text(encoding="utf-8"))
    assert provenance["source_transition"] == "explicit"
    assert provenance["source_commit"] == _live_state(harness)[0]
    assert "transition explicit" in worker_result.stdout.splitlines()[1]
    assert _git_calls(harness) == ""


@pytest.mark.parametrize("bound", ["new-unit", "unchanged-code"])
def test_continue_with_current_code_is_not_a_transition_without_a_different_bound_tree(harness: _Harness, bound: str) -> None:
    """A new unit, or a unit already in the current tree, records no transition."""
    if bound == "unchanged-code":
        assert _train(harness).returncode == 0
        harness.environment["PREFLIGHT_BOUND_TREE"] = _submitted_tree(harness).name
    result = _train(harness, "--continue-with-current-code")

    assert result.returncode == 0, result.stderr
    assert "ML_SOURCE_TRANSITION" not in _sbatch_environment(harness)
    assert "explicit transition" not in result.stdout
    worker_result = _run_worker(harness, gpu=False)
    assert worker_result.returncode == 0, worker_result.stderr
    assert json.loads(harness.provenance_capture.read_text(encoding="utf-8"))["source_transition"] == "none"


def test_ambient_transition_marker_never_reaches_a_job(harness: _Harness) -> None:
    """Only the launcher's own decision sets the marker, never the login environment."""
    harness.environment["ML_SOURCE_TRANSITION"] = "explicit"

    assert _train(harness).returncode == 0

    assert "ML_SOURCE_TRANSITION" not in _sbatch_environment(harness)


def test_artifacts_and_probe_always_run_the_current_code(harness: _Harness) -> None:
    """A downstream execution is new work and never inherits a bound tree."""
    assert _submit(harness, *_resources("cpu"), "probe").returncode == 0
    first_tree = _submitted_tree(harness)
    (harness.repo / "new-source.py").write_text("changed = True\n", encoding="utf-8")
    harness.environment["PREFLIGHT_BOUND_TREE"] = first_tree.name

    assert _submit(harness, *_resources("cpu"), "artifacts", "--run-dir", str(harness.storage / "run")).returncode == 0

    assert _submitted_tree(harness) != first_tree
    assert "ML_SOURCE_TRANSITION" not in _sbatch_environment(harness)


@pytest.mark.parametrize(
    ("operation", "arguments"),
    [
        ("artifacts", ["--task", "steady_flow", "--continue-with-current-code"]),
        ("probe", ["--continue-with-current-code"]),
    ],
)
def test_continue_with_current_code_is_rejected_outside_train_and_optuna(harness: _Harness, operation: str, arguments: list[str]) -> None:
    """Downstream operations have no bound unit to switch."""
    result = _submit(harness, *_resources("cpu"), operation, *arguments)

    assert result.returncode == 2
    assert "applies only to train and optuna" in result.stderr
    assert not harness.sbatch_capture.exists()


def test_duplicate_continue_with_current_code_is_rejected(harness: _Harness) -> None:
    """The switch is stated once."""
    result = _train(harness, "--continue-with-current-code", "--continue-with-current-code")

    assert result.returncode == 2
    assert not harness.sbatch_capture.exists()


@pytest.mark.parametrize(
    ("variable", "value", "message"),
    [
        ("PREFLIGHT_BOUND_TREE", "not-a-tree", "invalid source tree"),
        ("PREFLIGHT_EXTRA", "\textra", "invalid summary"),
    ],
)
def test_preflight_summary_must_end_in_exactly_one_valid_tree_field(harness: _Harness, variable: str, value: str, message: str) -> None:
    """The launcher parses the five-field preflight summary strictly."""
    harness.environment[variable] = value

    result = _train(harness)

    assert result.returncode == 1
    assert message in result.stderr
    assert not harness.sbatch_capture.exists()


def test_worker_verifies_an_untouched_tree_without_git(harness: _Harness) -> None:
    """A compute node needs no Git: the tree is checked against its snapshot alone."""
    assert _train(harness).returncode == 0

    result = _run_worker(harness, gpu=False)

    assert result.returncode == 0, result.stderr
    assert _submitted_tree(harness).name in result.stdout.splitlines()[1]
    assert _git_calls(harness) == ""


def test_worker_rejects_a_tampered_tree(harness: _Harness) -> None:
    """A queued job must run the exact frozen bytes."""
    assert _train(harness).returncode == 0
    tampered(_submitted_tree(harness), "config.yaml")

    result = _run_worker(harness, gpu=False)

    assert result.returncode != 0
    assert "failed verification against its snapshot" in result.stderr
    assert not harness.executor_capture.exists()


@pytest.mark.parametrize(("position", "message"), [(2, "commit differs"), (3, "fingerprint differs")])
def test_worker_rejects_commit_or_fingerprint_that_differs_from_the_submission(harness: _Harness, position: int, message: str) -> None:
    """The verified tree must be the one that was admitted at submission."""
    assert _train(harness).returncode == 0
    arguments = _worker_arguments(harness)
    arguments[position] = ("c" if position == 2 else "d") * (40 if position == 2 else 64)

    result = _run_worker(harness, gpu=False, arguments=arguments)

    assert result.returncode != 0
    assert message in result.stderr
    assert not harness.executor_capture.exists()


@pytest.mark.parametrize(
    "override",
    [
        pytest.param({"SOURCE_TREE_ID": "cccccccccccc-dddddddddddd"}, id="tree-id"),
        pytest.param({"SOURCE_TREE_ID": ""}, id="unset-tree-id"),
        pytest.param({"STORAGE_ROOT": ""}, id="unset-storage"),
        pytest.param({"RUNTIME_ROOT": ""}, id="unset-runtime"),
        pytest.param({"ML_SOURCE_TRANSITION": "implicit"}, id="transition"),
    ],
)
def test_worker_requires_explicit_tree_roots_and_a_known_transition(harness: _Harness, override: dict[str, str]) -> None:
    """A tree never infers its roots, and its identity must agree with its location."""
    assert _train(harness).returncode == 0

    result = _run_worker(harness, gpu=False, environment_override=override)

    assert result.returncode != 0
    assert not harness.executor_capture.exists()


def test_worker_rejects_the_live_checkout_as_project_root(harness: _Harness) -> None:
    """Only RUNTIME_ROOT/sources/<tree id> is an admissible repository for a job."""
    assert _train(harness).returncode == 0

    result = _run_worker(harness, gpu=False, environment_override={"PROJECT_ROOT": str(harness.repo)})

    assert result.returncode != 0
    assert "must be the source tree" in result.stderr
    assert not harness.executor_capture.exists()


def test_job_log_names_are_defined_once_appended_to_and_reach_the_executor(harness: _Harness) -> None:
    """The sbatch log templates, the append mode and the job's own log paths all come from one definition."""
    assert _train(harness).returncode == 0
    stdout_template = f"{harness.runtime}/logs/ml/slurm-%j.out"
    stderr_template = f"{harness.runtime}/logs/ml/slurm-%j.err"
    arguments = _args(harness.sbatch_capture)
    assert {f"--output={stdout_template}", f"--error={stderr_template}", "--open-mode=append"} <= set(arguments)

    worker_result = _run_worker(harness, gpu=False)

    assert worker_result.returncode == 0, worker_result.stderr
    assert harness.job_log_capture.read_text(encoding="utf-8").split("\n") == [
        f"{harness.runtime}/logs/ml/slurm-12345.out",
        f"{harness.runtime}/logs/ml/slurm-12345.err",
    ]


@pytest.fixture
def executor_roots(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Provide storage, runtime and an inert apptainer for the real executor script."""
    storage = tmp_path / "storage"
    runtime = tmp_path / "runtime"
    storage.mkdir()
    (runtime / "containers").mkdir(parents=True)
    (runtime / "containers" / "grainlegumes-pino-drying.sif").write_bytes(b"test image")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    apptainer = bin_dir / "apptainer"
    apptainer.write_text('#!/usr/bin/env bash\nprintf \'%s\\0\' "$@" > "${APPTAINER_CAPTURE}"\n', encoding="utf-8")
    apptainer.chmod(0o755)
    return storage, runtime, bin_dir


_FROZEN_TREE = "0123456789ab-fedcba987654"


@pytest.mark.parametrize(
    ("relative_root", "admitted"),
    [
        (f"runtime/sources/{_FROZEN_TREE}", True),
        ("runtime/sources/0123456789ab-fedcba98765", False),
        ("runtime/sources/0123456789AB-fedcba987654", False),
        ("runtime/sources/current", False),
        (f"runtime/sources/{_FROZEN_TREE}/scripts", False),
        ("runtime/sources", False),
        ("runtime", False),
        (f"runtime/other/{_FROZEN_TREE}", False),
        (f"storage/sources/{_FROZEN_TREE}", False),
    ],
)
def test_apptainer_executor_admits_only_a_frozen_tree_below_the_runtime_sources(
    tmp_path: Path, executor_roots: tuple[Path, Path, Path], relative_root: str, admitted: bool
) -> None:
    """The repository may overlap the runtime root only as RUNTIME_ROOT/sources/<commit12>-<fingerprint12>."""
    storage, runtime, bin_dir = executor_roots
    project_root = tmp_path / relative_root
    project_root.mkdir(parents=True, exist_ok=True)
    capture = tmp_path / "apptainer.args"
    environment = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "PROJECT_ROOT": str(project_root),
        "STORAGE_ROOT": str(storage),
        "RUNTIME_ROOT": str(runtime),
        "SOURCE_TREE_ID": _FROZEN_TREE,
        "APPTAINER_CAPTURE": str(capture),
    }

    result = subprocess.run(  # noqa: S603 -- test-owned executable and controlled argument vector; no shell interpolation
        [str(Path(__file__).resolve().parents[2] / "scripts" / "apptainer_exec.sh"), "python", "-V"],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    if not admitted:
        assert result.returncode == 1
        assert "must not overlap" in result.stderr
        assert not capture.exists()
        return
    assert result.returncode == 0, result.stderr
    arguments = _args(capture)
    assert f"{project_root}:/workspace/repo:ro" in arguments
    assert "PROJECT_ROOT=/workspace/repo" in arguments
    assert "STORAGE_ROOT=/workspace/storage" in arguments
    assert "RUNTIME_ROOT=/workspace/runtime" in arguments
    assert not any("SOURCE_TREE_ID" in argument for argument in arguments)


@pytest.mark.parametrize("container_status", [0, 3])
def test_apptainer_executor_keeps_temporary_files_on_node_scratch_and_removes_them(
    tmp_path: Path, executor_roots: tuple[Path, Path, Path], container_status: int
) -> None:
    """Container temporaries go to a private node-scratch directory that is gone after exit, whatever the status."""
    storage, runtime, bin_dir = executor_roots
    (bin_dir / "apptainer").write_text(
        f'#!/usr/bin/env bash\nprintf \'%s\\0\' "$@" > "${{APPTAINER_CAPTURE}}"\nexit {container_status}\n', encoding="utf-8"
    )
    node_scratch = tmp_path / "node-scratch"
    node_scratch.mkdir()
    capture = tmp_path / "apptainer.args"
    project_root = runtime / "sources" / _FROZEN_TREE
    project_root.mkdir(parents=True)
    environment = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "PROJECT_ROOT": str(project_root),
        "STORAGE_ROOT": str(storage),
        "RUNTIME_ROOT": str(runtime),
        "TMPDIR": str(node_scratch),
        "APPTAINER_CAPTURE": str(capture),
    }

    result = subprocess.run(  # noqa: S603 -- test-owned executable and controlled argument vector; no shell interpolation
        [str(Path(__file__).resolve().parents[2] / "scripts" / "apptainer_exec.sh"), "python", "-V"],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == container_status, result.stderr
    arguments = _args(capture)
    binds = [argument for argument in arguments if argument.endswith(":/workspace/tmp:rw")]
    assert len(binds) == 1
    assert Path(binds[0].removesuffix(":/workspace/tmp:rw")).parent == node_scratch
    assert "TMPDIR=/workspace/tmp" in arguments
    assert not any(argument.startswith("HOME=") for argument in arguments)
    assert not tuple(node_scratch.iterdir())
    assert not (runtime / "apptainer" / "tmp").exists()


@pytest.mark.parametrize("queue_readable", [True, False])
def test_apptainer_executor_removes_scratch_left_by_ended_jobs_only(
    tmp_path: Path, executor_roots: tuple[Path, Path, Path], queue_readable: bool
) -> None:
    """Scratch of a killed job goes once it is an hour old and its job left the queue; nothing goes when the queue is unreadable."""
    storage, runtime, bin_dir = executor_roots
    squeue = bin_dir / "squeue"
    squeue.write_text("#!/usr/bin/env bash\nprintf '111\\n'\n" if queue_readable else "#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")
    squeue.chmod(0o755)
    node_scratch = tmp_path / "node-scratch"
    node_scratch.mkdir()
    an_hour_and_more = time.time() - 7200
    leftovers = {name: node_scratch / name for name in ("vp2-apptainer.222.a", "vp2-apptainer.111.b", "vp2-apptainer.333.c", "vp2-apptainer.local.d")}
    for name, path in leftovers.items():
        path.mkdir()
        if name != "vp2-apptainer.333.c":
            os.utime(path, (an_hour_and_more, an_hour_and_more))
    project_root = runtime / "sources" / _FROZEN_TREE
    project_root.mkdir(parents=True)
    environment = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "PROJECT_ROOT": str(project_root),
        "STORAGE_ROOT": str(storage),
        "RUNTIME_ROOT": str(runtime),
        "TMPDIR": str(node_scratch),
        "SLURM_JOB_ID": "999",
        "APPTAINER_CAPTURE": str(tmp_path / "apptainer.args"),
    }

    result = subprocess.run(  # noqa: S603 -- test-owned executable and controlled argument vector; no shell interpolation
        [str(Path(__file__).resolve().parents[2] / "scripts" / "apptainer_exec.sh"), "python", "-V"],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    removed = {"vp2-apptainer.222.a"} if queue_readable else set()
    assert {path.name for path in node_scratch.iterdir()} == set(leftovers) - removed


def _make_job_logs(runtime: Path) -> tuple[Path, Path]:
    """Create the stdout and stderr logs of one finished test job below the runtime root."""
    logs = runtime / "logs" / "ml"
    logs.mkdir(parents=True)
    paths = (logs / "slurm-77.out", logs / "slurm-77.err")
    for path in paths:
        path.write_text(path.name, encoding="utf-8")
    return paths


def _run_executor_after_job(
    tmp_path: Path,
    roots: tuple[Path, Path, Path],
    *,
    owners: list[str] | None,
    job_logs: list[Path] | None,
    locked_scratch: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run the real executor around a fake container that records ``owners`` as job log owners and exits with status 3."""
    storage, runtime, bin_dir = roots
    (bin_dir / "apptainer").write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\\0\' "$@" > "${APPTAINER_CAPTURE}"\n'
        'for argument in "$@"; do\n'
        '  [[ "${argument}" == *:/workspace/tmp:rw ]] && scratch="${argument%:/workspace/tmp:rw}"\n'
        "done\n"
        '[[ -z "${APPTAINER_OWNERS+x}" ]] || printf \'%s\\n\' "${APPTAINER_OWNERS}" > "${scratch}/job-log-owners"\n'
        '[[ -z "${APPTAINER_LOCKED+x}" ]] || { mkdir -p "${scratch}/locked/inner"; chmod 0555 "${scratch}/locked"; }\n'
        "exit 3\n",
        encoding="utf-8",
    )
    (tmp_path / "node-scratch").mkdir()
    project_root = runtime / "sources" / _FROZEN_TREE
    project_root.mkdir(parents=True)
    environment = {
        **{key: value for key, value in os.environ.items() if key != "JOB_LOG_FILES"},
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "PROJECT_ROOT": str(project_root),
        "STORAGE_ROOT": str(storage),
        "RUNTIME_ROOT": str(runtime),
        "TMPDIR": str(tmp_path / "node-scratch"),
        "APPTAINER_CAPTURE": str(tmp_path / "apptainer.args"),
    }
    if owners is not None:
        environment["APPTAINER_OWNERS"] = "\n".join(owners)
    if job_logs is not None:
        environment["JOB_LOG_FILES"] = "\n".join(str(path) for path in job_logs)
    if locked_scratch:
        environment["APPTAINER_LOCKED"] = "1"
    return subprocess.run(  # noqa: S603 -- test-owned executable and controlled argument vector; no shell interpolation
        [str(Path(__file__).resolve().parents[2] / "scripts" / "apptainer_exec.sh"), "python", "-V"],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def _warnings(result: subprocess.CompletedProcess[str], tmp_path: Path) -> list[str]:
    """Return the executor's warnings and require that the container status and the scratch cleanup survived them."""
    assert result.returncode == 3, result.stderr
    assert not tuple((tmp_path / "node-scratch").iterdir())
    assert all(line.startswith("Warning: ") for line in result.stderr.splitlines())
    return result.stderr.splitlines()


def test_apptainer_executor_links_the_recorded_job_logs_once_and_keeps_the_container_status(
    tmp_path: Path, executor_roots: tuple[Path, Path, Path]
) -> None:
    """Each recorded folder gets both logs as the same files, a link from an earlier attempt and a vanished log are silent."""
    storage, runtime, _ = executor_roots
    stdout_log, stderr_log = _make_job_logs(runtime)
    first, second = storage / "runs" / "one", storage / "runs" / "two"
    for owner in (first, second):
        owner.mkdir(parents=True)
    (first / "logs").mkdir()
    (first / "logs" / stderr_log.name).hardlink_to(stderr_log)

    result = _run_executor_after_job(
        tmp_path,
        executor_roots,
        owners=["/workspace/storage/runs/one", "/workspace/storage/runs/two", "/workspace/storage/runs/one"],
        job_logs=[stdout_log, stderr_log, runtime / "logs" / "ml" / "slurm-77.missing"],
    )

    assert _warnings(result, tmp_path) == []
    for owner in (first, second):
        assert sorted(path.name for path in (owner / "logs").iterdir()) == ["slurm-77.err", "slurm-77.out"]
        assert all((owner / "logs" / log.name).samefile(log) for log in (stdout_log, stderr_log))
    assert "JOB_LOG_OWNERS_FILE=/workspace/tmp/job-log-owners" in _args(tmp_path / "apptainer.args")


@pytest.mark.parametrize("missing", ["job-logs", "record"])
def test_apptainer_executor_links_nothing_and_says_nothing_without_job_logs_or_a_record(
    tmp_path: Path, executor_roots: tuple[Path, Path, Path], missing: str
) -> None:
    """A manual call or a preflight, which have no job logs or record no owner, leave every folder untouched."""
    storage, runtime, _ = executor_roots
    job_logs = list(_make_job_logs(runtime))
    (storage / "run").mkdir()

    result = _run_executor_after_job(
        tmp_path,
        executor_roots,
        owners=None if missing == "record" else ["/workspace/storage/run"],
        job_logs=None if missing == "job-logs" else job_logs,
    )

    assert _warnings(result, tmp_path) == []
    assert not (storage / "run" / "logs").exists()


@pytest.mark.parametrize("problem", ["outside-storage", "dot-dot", "missing", "symlinked-owner", "symlink-in-path", "symlinked-logs"])
def test_apptainer_executor_refuses_an_unsafe_log_owner_with_one_warning(
    tmp_path: Path, executor_roots: tuple[Path, Path, Path], problem: str
) -> None:
    """An owner outside storage, absent, or reached through `..` or a symbolic link gets no link and one warning."""
    storage, runtime, _ = executor_roots
    job_logs = list(_make_job_logs(runtime))
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "sub").mkdir(parents=True)
    (storage / "run").mkdir()
    (storage / "linked").symlink_to(elsewhere)
    (storage / "run" / "logs").symlink_to(elsewhere)
    owner = {
        "outside-storage": "/elsewhere/run",
        "dot-dot": "/workspace/storage/../elsewhere",
        "missing": "/workspace/storage/missing",
        "symlinked-owner": "/workspace/storage/linked",
        "symlink-in-path": "/workspace/storage/linked/sub",
        "symlinked-logs": "/workspace/storage/run",
    }[problem]

    result = _run_executor_after_job(tmp_path, executor_roots, owners=[owner], job_logs=job_logs)

    assert len(_warnings(result, tmp_path)) == 1
    assert [path.name for path in elsewhere.rglob("*")] == ["sub"]


def test_apptainer_executor_leaves_a_different_file_at_the_link_target_alone(tmp_path: Path, executor_roots: tuple[Path, Path, Path]) -> None:
    """A foreign file keeps its place and bytes with one warning; the job's other log is still linked."""
    storage, runtime, _ = executor_roots
    stdout_log, stderr_log = _make_job_logs(runtime)
    (storage / "run" / "logs").mkdir(parents=True)
    foreign = storage / "run" / "logs" / stdout_log.name
    foreign.write_text("not the job log", encoding="utf-8")

    result = _run_executor_after_job(tmp_path, executor_roots, owners=["/workspace/storage/run"], job_logs=[stdout_log, stderr_log])

    assert len(_warnings(result, tmp_path)) == 1
    assert foreign.read_text(encoding="utf-8") == "not the job log"
    assert (foreign.parent / stderr_log.name).samefile(stderr_log)


def test_apptainer_executor_cleans_up_and_keeps_the_status_when_logs_cannot_be_written(
    tmp_path: Path, executor_roots: tuple[Path, Path, Path]
) -> None:
    """A read-only logs/ costs one warning per log and nothing else: the scratch goes and the container status stays."""
    storage, runtime, _ = executor_roots
    job_logs = list(_make_job_logs(runtime))
    logs = storage / "run" / "logs"
    logs.mkdir(parents=True)
    logs.chmod(0o555)
    try:
        if os.access(logs, os.W_OK):
            pytest.skip("directory permissions do not bind this user")
        result = _run_executor_after_job(tmp_path, executor_roots, owners=["/workspace/storage/run"], job_logs=job_logs)
        warnings = _warnings(result, tmp_path)
        assert not tuple(logs.iterdir())
    finally:
        logs.chmod(0o755)
    assert len(warnings) == len(job_logs)


def test_apptainer_executor_keeps_the_container_status_when_its_scratch_cannot_be_removed(
    tmp_path: Path, executor_roots: tuple[Path, Path, Path]
) -> None:
    """A scratch the container left undeletable ends the cleanup with a warning, not with a changed exit status."""
    result = _run_executor_after_job(tmp_path, executor_roots, owners=None, job_logs=None, locked_scratch=True)
    locked = list((tmp_path / "node-scratch").glob("*/locked"))
    for directory in locked:
        directory.chmod(0o755)
    if not locked:
        pytest.skip("directory permissions do not bind this user")
    assert result.returncode == 3, result.stderr
    assert result.stderr.splitlines()[-1].startswith("Warning: ")
