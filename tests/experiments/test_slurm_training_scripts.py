# ruff: noqa: S101, S603, S607
"""Exercise shared ML submission and worker contracts without contacting Slurm."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest


@dataclass(frozen=True)
class _Harness:
    repo: Path
    storage: Path
    runtime: Path
    environment: dict[str, str]
    sbatch_capture: Path
    executor_capture: Path
    provenance_capture: Path


@pytest.fixture
def harness(request: pytest.FixtureRequest) -> _Harness:
    """Build a disposable Git checkout with inert sbatch and Apptainer boundaries."""
    temporary = TemporaryDirectory(prefix="slurm-ml-test-", dir="/tmp")
    request.addfinalizer(temporary.cleanup)  # noqa: PT021 -- clean the checkout if fixture setup fails
    tmp_path = Path(temporary.name)
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
    # The fake executor must hide host GPUs as the CPU Apptainer path does.
    executor = scripts / "apptainer_exec.sh"
    executor.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'if [[ "${1:-}" == python && "${2:-}" == -m && "${3:-}" == src.experiments.cli.cli_config_preflight ]]; then\n'
        '  if [[ "$4" == optuna ]]; then\n'
        "    printf 'optuna\\tsteady_flow\\t/workspace/repo/config.yaml\\tstudy\\n'\n"
        "  else\n"
        "    printf 'experiment\\tsteady_flow\\t/workspace/repo/config.yaml\\trun\\n'\n"
        "  fi\n"
        "else\n"
        '  printf \'%s\\0\' "$@" > "${EXECUTOR_CAPTURE}"\n'
        '  printf \'%s\' "${APPTAINERENV_ML_SLURM_PROVENANCE:-}" > "${PROVENANCE_CAPTURE}"\n'
        '  if [[ "${1:-}" == python && "${2:-}" == -c ]]; then\n'
        '    CUDA_VISIBLE_DEVICES="" ML_SLURM_PROVENANCE="${APPTAINERENV_ML_SLURM_PROVENANCE}" PYTHONPATH="${SOURCE_REPO}" python -c "$3" "$4"\n'
        "  fi\n"
        "fi\n",
        encoding="utf-8",
    )
    executor.chmod(0o755)
    (repo / "config.yaml").write_text("task: steady_flow\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sbatch = bin_dir / "sbatch"
    sbatch.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\0' \"$@\" > \"${SBATCH_CAPTURE}\"\nprintf '12345\\n'\n",
        encoding="utf-8",
    )
    sbatch.chmod(0o755)
    sbatch_capture = tmp_path / "sbatch.args"
    executor_capture = tmp_path / "executor.args"
    provenance_capture = tmp_path / "provenance.json"
    environment = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "STORAGE_ROOT": str(storage),
        "RUNTIME_ROOT": str(runtime),
        "SBATCH_CAPTURE": str(sbatch_capture),
        "EXECUTOR_CAPTURE": str(executor_capture),
        "PROVENANCE_CAPTURE": str(provenance_capture),
        "SOURCE_REPO": str(Path(__file__).resolve().parents[2]),
    }
    return _Harness(repo, storage, runtime, environment, sbatch_capture, executor_capture, provenance_capture)


def _args(path: Path) -> list[str]:
    """Decode exact argument boundaries captured by an inert shell executable."""
    return [part.decode() for part in path.read_bytes().split(b"\0") if part]


def _submit(harness: _Harness, *arguments: str) -> subprocess.CompletedProcess[str]:
    """Call the real submit wrapper against the inert sbatch executable."""
    return subprocess.run(
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


def _run_worker(harness: _Harness, *, gpu: bool) -> subprocess.CompletedProcess[str]:
    """Replay a Slurm-spooled worker copy inside a fake allocation."""
    submission = _args(harness.sbatch_capture)
    worker = str(harness.repo / "scripts" / "slurm_ml_worker.sh")
    start = submission.index(worker)
    spooled_worker = harness.runtime / "slurm_script"
    shutil.copy2(worker, spooled_worker)
    environment = {
        **harness.environment,
        "PROJECT_ROOT": str(harness.repo),
        "SLURM_JOB_ID": "12345",
        "SLURM_JOB_PARTITION": "gpu" if gpu else "standard",
        "SLURM_CPUS_PER_TASK": "2",
    }
    environment.pop("SLURM_JOB_GPUS", None)
    environment.pop("SLURM_STEP_GPUS", None)
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    if gpu:
        environment.update({"SLURM_JOB_GPUS": "0", "CUDA_VISIBLE_DEVICES": "0"})
    return subprocess.run(
        [str(spooled_worker), *submission[start + 1 :]],
        cwd=harness.repo,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_cpu_submission_and_resume_preserve_training_arguments(harness: _Harness) -> None:
    """CPU Slurm construction has no GRES and keeps explicit resume semantics."""
    run_dir = harness.storage / "run with spaces"
    result = _submit(harness, *_resources("cpu"), "train", "config.yaml", "--resume", str(run_dir), "--no-build-artifacts")

    assert result.returncode == 0, result.stderr
    assert "Slurm job ID: 12345" in result.stdout
    arguments = _args(harness.sbatch_capture)
    assert "--gres" not in " ".join(arguments)
    assert f"--output={harness.runtime}/logs/ml/slurm-%j.out" in arguments
    assert f"--error={harness.runtime}/logs/ml/slurm-%j.err" in arguments
    assert arguments[-6:] == [
        "/workspace/repo/config.yaml",
        "--resume",
        "/workspace/storage/run with spaces",
        "--no-build-artifacts",
        "--device",
        "cpu",
    ]
    worker_result = _run_worker(harness, gpu=False)
    assert worker_result.returncode == 0, worker_result.stderr
    assert _args(harness.executor_capture) == [
        "python",
        "-m",
        "src.experiments.cli.cli_train",
        *arguments[-6:],
    ]
    provenance = json.loads(harness.provenance_capture.read_text(encoding="utf-8"))
    assert provenance["mode"] == "cpu"
    assert provenance["job_id"] == "12345"
    assert provenance["gres"] == "none"
    assert provenance["sif_path"] == "/workspace/runtime/containers/grainlegumes-pino-drying.sif"


def test_gpu_submission_uses_named_gres_and_worker_forwards_strict_cuda(harness: _Harness) -> None:
    """Slurm selects a named GPU; no host physical index enters the request."""
    result = _submit(harness, *_resources("gpu"), "--gres", "gpu:rtx6000ada:1", "train", "config.yaml")

    assert result.returncode == 0, result.stderr
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
        "--n-trials",
        "2",
        "--output-root",
        str(harness.storage / "study with spaces"),
        "--show-progress-bar",
    )
    assert result.returncode == 0, result.stderr
    arguments = _args(harness.sbatch_capture)
    if mode == "cpu":
        assert not any(argument.startswith("--gres=") for argument in arguments)
    else:
        assert "--gres=gpu:rtx6000ada:1" in arguments
    worker_result = _run_worker(harness, gpu=mode == "gpu")
    assert worker_result.returncode == 0, worker_result.stderr
    expected = [
        "/workspace/repo/config.yaml",
        "--n-trials",
        "2",
        "--output-root",
        "/workspace/storage/study with spaces",
        "--show-progress-bar",
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
        [*_resources("cpu"), "train", "config.yaml", "--resume"],
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


@pytest.mark.parametrize("option", ["--resume", "--output-root"])
def test_storage_path_traversal_fails_before_submission(harness: _Harness, option: str) -> None:
    """Container paths cannot escape durable storage through parent segments."""
    result = _submit(
        harness,
        *_resources("cpu"),
        "train",
        "config.yaml",
        option,
        "/workspace/storage/../runtime/run",
    )

    assert result.returncode != 0
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


def test_worker_rejects_missing_gpu_allocation_and_source_drift(harness: _Harness) -> None:
    """The worker checks Slurm allocation and queued source identity before execution."""
    assert _submit(harness, *_resources("gpu"), "--gres", "gpu:rtxa6000:1", "probe").returncode == 0
    missing_gpu = _run_worker(harness, gpu=False)
    assert missing_gpu.returncode != 0
    assert not harness.executor_capture.exists()

    (harness.repo / "new-source.py").write_text("changed = True\n", encoding="utf-8")
    drifted = _run_worker(harness, gpu=True)
    assert drifted.returncode != 0
    assert "source state changed" in drifted.stderr
    assert not harness.executor_capture.exists()


def test_probe_uses_same_apptainer_executor_without_dataset_access(harness: _Harness) -> None:
    """Acceptance probes traverse the same worker and executor as training."""
    assert _submit(harness, *_resources("cpu"), "probe").returncode == 0
    result = _run_worker(harness, gpu=False)

    assert result.returncode == 0, result.stderr
    assert _args(harness.executor_capture)[:2] == ["python", "-c"]
    assert "SIF SHA256:" in result.stdout
