# ruff: noqa: PLR2004, S101, S603
"""
Verify Docker and cluster launchers through isolated command stubs, never submission.

The harness covers canonical mounts/environment roots, GPU discovery/selection,
quoting, queue arguments, logging, wrapper validation, and early rejection of
invalid CPU/fallback requests. It deliberately does not run Docker, Slurm,
``runTSGPU.py``, training, or a real GPU workload.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from support import configs

pytestmark = pytest.mark.integration

_REPOSITORY_ROOT = Path(__file__).parents[2]
_DIRECT_CONFIG_RELATIVE = "configs/learning/steady_flow/experiments/synthetic/direct.yaml"
_DIRECT_REPOSITORY_RELATIVE = _DIRECT_CONFIG_RELATIVE
_OPTUNA_CONFIG_RELATIVE = "configs/learning/steady_flow/optuna/synthetic.yaml"
_CAMPAIGN_CONFIG_RELATIVE = "configs/generation/campaigns/test_support/docker_wrapper.yaml"


@dataclass(frozen=True)
class _Harness:
    """
    Hold one immutable isolated launcher-test environment.

    Attributes
    ----------
    repository : pathlib.Path
        Temporary repository copy containing only the launcher scripts and configs.
    environment : dict[str, str]
        Subprocess environment that routes external commands to local stubs.
    binary_dir : pathlib.Path
        Directory containing the fake ``docker``, ``nvidia-smi``, and queue commands.
    home : pathlib.Path
        Isolated home used for optional credential-file precedence.
    launcher capture paths : pathlib.Path
        NUL-delimited argument/environment and invocation capture files written by stubs.

    """

    repository: Path
    environment: dict[str, str]
    binary_dir: Path
    home: Path
    runtsgpu_capture: Path
    docker_capture: Path
    preflight_docker_capture: Path
    path_docker_capture: Path
    docker_environment_capture: Path
    tail_capture: Path
    nvidia_capture: Path
    host_python_capture: Path


def _write_executable(path: Path, content: str) -> None:
    """Write one isolated command stub with executable permissions."""
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _capture_arguments(path: Path) -> list[str]:
    """Decode one NUL-delimited command capture."""
    return [part.decode() for part in path.read_bytes().split(b"\0") if part]


def _harness(
    tmp_path: Path,
    *,
    docker_exit_code: int = 0,
    exported_key: str | None = "mock API key with spaces",
    file_key: str | None = None,
    gpu_report: str | None = None,
    preflight_summary: str | None = None,
) -> _Harness:
    """
    Create safe command stubs around copied launcher scripts and minimal configs.

    The returned harness records argv, environment, output streams, and exit codes
    without invoking Docker, a scheduler, or GPU hardware. Each call owns an isolated
    repository, storage root, home, and PATH.
    """
    repository = tmp_path / "repository with spaces"
    scripts = repository / "scripts"
    scripts.mkdir(parents=True)
    for name in (
        "docker_job.sh",
        "docker_python.sh",
        "_docker_run.sh",
        "config_preflight_runtime.py",
    ):
        shutil.copy2(_REPOSITORY_ROOT / "scripts" / name, scripts / name)

    shutil.copytree(_REPOSITORY_ROOT / "src", repository / "src")
    direct_payload = configs.direct_config(device="auto")
    configs.write_yaml(
        repository / _DIRECT_CONFIG_RELATIVE,
        direct_payload,
    )
    optuna_payload = configs.optuna_config()
    optuna_payload["experiment"]["run"]["device"] = "auto"
    configs.write_yaml(
        repository / _OPTUNA_CONFIG_RELATIVE,
        optuna_payload,
    )
    configs.write_yaml(
        repository / _CAMPAIGN_CONFIG_RELATIVE,
        {"schema_kind": "generation_campaign", "schema_version": 1},
    )

    binary_dir = tmp_path / "stub commands"
    binary_dir.mkdir()
    _write_executable(
        binary_dir / "python",
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\0' "$@" > "${HOST_PYTHON_CAPTURE}"
if [[ "${1-}" == "-c" ]]; then
  printf '%s\\n' "${HOST_PYTHON_STUB_VERSION:-3.9.19}"
  exit "${HOST_PYTHON_VERSION_EXIT_CODE:-0}"
fi
echo 'host Python was asked to import project code directly' >&2
exit 97
""",
    )
    report = gpu_report if gpu_report is not None else ("0, Cluster GPU A, 20, 7000, 24000\n2, Cluster GPU B, 5, 1000, 24000\n")
    _write_executable(
        binary_dir / "nvidia-smi",
        f"""#!/usr/bin/env bash
set -euo pipefail
printf 'called\n' >> "${{NVIDIA_CAPTURE}}"
if [[ "${{1-}}" == "-L" ]]; then
  printf 'GPU 0: Cluster GPU A\\nGPU 2: Cluster GPU B\\n'
elif [[ "$*" == *"--query-gpu=index,name,utilization.gpu,memory.used,memory.total"* ]]; then
  printf '%b' {report!r}
else
  printf 'unexpected nvidia-smi arguments: %s\\n' "$*" >&2
  exit 64
fi
""",
    )
    _write_executable(
        binary_dir / "runTSGPU.py",
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\0' "$@" >> "${RUNTSGPU_CAPTURE}"
if (( ${QUEUE_SUBMISSION_EXIT:-0} != 0 )); then
  printf '%b' "${QUEUE_PARTIAL_OUTPUT-}"
  echo 'stub queue submission refused' >&2
  exit "${QUEUE_SUBMISSION_EXIT}"
fi
while (( $# > 0 )); do
  if [[ "$1" == "--" ]]; then
    shift
    set +e
    "$@"
    set -e
    if [[ -n "${QUEUE_SUBMISSION_OUTPUT+x}" ]]; then
      printf '%b' "${QUEUE_SUBMISSION_OUTPUT}"
    else
      printf 'TS socket: %s\n' "${TS_SOCKET:-/etc/ts/socket_unknown}"
      printf '%s\n' "${QUEUE_JOB_ID:-25}"
    fi
    exit 0
  fi
  shift
done
echo 'runTSGPU stub did not receive --' >&2
exit 65
""",
    )
    _write_executable(
        binary_dir / "docker",
        """#!/usr/bin/env bash
set -euo pipefail
case "${1-}" in
  info)
    if [[ "$*" == *"--format"* ]]; then
      printf '{"nvidia": {}}\\n'
    fi
    exit "${DOCKER_INFO_EXIT_CODE:-0}"
    ;;
  image)
    if [[ "${DOCKER_IMAGE_AVAILABLE:-true}" == true ]]; then
      exit 0
    fi
    exit 1
    ;;
  ps)
    if [[ "$*" == *" -a "* || "$*" == *"ps -a"* ]]; then
      printf '%s' "${DOCKER_ALL_NAMES-}"
    else
      printf '%s' "${DOCKER_RUNNING_NAMES-}"
    fi
    exit 0
    ;;
  start)
    exit 0
    ;;
  run)
    arguments=("$@")
    path_module_index=-1
    for index in "${!arguments[@]}"; do
      if [[ "${arguments[index]}" == "src.common.common_queue_log_cli" ]]; then
        path_module_index="${index}"
        break
      fi
    done
    if (( path_module_index >= 0 )); then
      printf '%s\\0' "$@" > "${PATH_DOCKER_CAPTURE}"
      scope="${arguments[path_module_index + 1]}"
      printf '/workspace/storage/03_experiments/%s/logs/queue\n' "${scope}"
      exit 0
    fi
    bootstrap_index=-1
    for index in "${!arguments[@]}"; do
      if [[ "${arguments[index]}" == "/workspace/repo/scripts/config_preflight_runtime.py" ]]; then
        bootstrap_index="${index}"
        break
      fi
    done
    if (( bootstrap_index >= 0 )); then
      printf '%s\\0' "$@" > "${PREFLIGHT_DOCKER_CAPTURE}"
      if (( ${PREFLIGHT_CONTAINER_EXIT_CODE:-0} != 0 )); then
        printf '%s' "${PREFLIGHT_CONTAINER_STDOUT-}"
        printf '%s' "${PREFLIGHT_CONTAINER_STDERR-}" >&2
        exit "${PREFLIGHT_CONTAINER_EXIT_CODE}"
      fi
      if [[ -n "${PREFLIGHT_STUB_SUMMARY-}" ]]; then
        printf '%s\n' "${PREFLIGHT_STUB_SUMMARY}"
        exit 0
      fi
      workflow="${arguments[bootstrap_index + 1]}"
      config="${arguments[bootstrap_index + 2]}"
      case "${config}" in
        /workspace/repo/*)
          host_config="${PROJECT_ROOT}/${config#/workspace/repo/}"
          ;;
        /workspace/storage/*)
          host_config="${STORAGE_ROOT}/${config#/workspace/storage/}"
          ;;
        *)
          host_config="${config}"
          ;;
      esac
      (
        cd "${PROJECT_ROOT}"
        export PYTHONPATH="${PROJECT_ROOT}"
        export PROJECT_ROOT
        export STORAGE_ROOT
        "${CONTAINER_PYTHON}" "${PROJECT_ROOT}/scripts/config_preflight_runtime.py" "${workflow}" "${host_config}"
      )
      exit $?
    fi
    printf '%s\\0' "$@" >> "${DOCKER_CAPTURE}"
    printf '%s' "${WANDB_API_KEY-<unset>}" > "${DOCKER_ENV_CAPTURE}"
    printf 'captured Docker stdout with spaces\\n'
    printf 'captured Docker stderr with spaces\\n' >&2
    exit "${DOCKER_EXIT_CODE:-0}"
    ;;
  build)
    exit 0
    ;;
  *)
    printf 'unexpected Docker command: %s\\n' "$*" >&2
    exit 66
    ;;
esac
""",
    )

    _write_executable(
        binary_dir / "tail",
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\0' "$@" > "${TAIL_CAPTURE}"
case "${TAIL_MODE:-return}" in
  return)
    exit 0
    ;;
  interrupt)
    kill -INT "$PPID"
    exit 130
    ;;
  fail)
    exit 42
    ;;
  *)
    echo "unknown TAIL_MODE: ${TAIL_MODE}" >&2
    exit 64
    ;;
esac
""",
    )

    home = tmp_path / "home without ssh"
    home.mkdir()
    if file_key is not None:
        (home / "wandb_key.txt").write_text(file_key, encoding="utf-8")

    fallback_binary_dir = tmp_path / "fallback commands"
    fallback_binary_dir.mkdir()
    for command in (
        "bash",
        "basename",
        "cat",
        "chmod",
        "date",
        "dirname",
        "env",
        "grep",
        "id",
        "mkdir",
        "mktemp",
        "realpath",
    ):
        resolved = shutil.which(command)
        assert resolved is not None
        (fallback_binary_dir / command).symlink_to(resolved)

    storage_root = tmp_path / "storage root with spaces"
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{binary_dir}{os.pathsep}{fallback_binary_dir}",
            "HOME": str(home),
            "PROJECT_ROOT": str(repository),
            "STORAGE_ROOT": str(storage_root),
            "RUNTSGPU_CAPTURE": str(tmp_path / "runtsgpu.args"),
            "DOCKER_CAPTURE": str(tmp_path / "docker.args"),
            "PREFLIGHT_DOCKER_CAPTURE": str(tmp_path / "preflight-docker.args"),
            "PATH_DOCKER_CAPTURE": str(tmp_path / "path-docker.args"),
            "DOCKER_ENV_CAPTURE": str(tmp_path / "docker.env"),
            "TAIL_CAPTURE": str(tmp_path / "tail.args"),
            "NVIDIA_CAPTURE": str(tmp_path / "nvidia.called"),
            "HOST_PYTHON_CAPTURE": str(tmp_path / "host-python.args"),
            "HOST_PYTHON_STUB_VERSION": "3.9.19",
            "CONTAINER_PYTHON": sys.executable,
            "QUEUE_JOB_ID": "25",
            "DOCKER_EXIT_CODE": str(docker_exit_code),
        }
    )
    if exported_key is None:
        environment.pop("WANDB_API_KEY", None)
    else:
        environment["WANDB_API_KEY"] = exported_key
    if preflight_summary is not None:
        environment["PREFLIGHT_STUB_SUMMARY"] = preflight_summary
    return _Harness(
        repository=repository,
        environment=environment,
        binary_dir=binary_dir,
        home=home,
        runtsgpu_capture=Path(environment["RUNTSGPU_CAPTURE"]),
        docker_capture=Path(environment["DOCKER_CAPTURE"]),
        preflight_docker_capture=Path(environment["PREFLIGHT_DOCKER_CAPTURE"]),
        path_docker_capture=Path(environment["PATH_DOCKER_CAPTURE"]),
        docker_environment_capture=Path(environment["DOCKER_ENV_CAPTURE"]),
        tail_capture=Path(environment["TAIL_CAPTURE"]),
        nvidia_capture=Path(environment["NVIDIA_CAPTURE"]),
        host_python_capture=Path(environment["HOST_PYTHON_CAPTURE"]),
    )


def _run_job(
    harness: _Harness,
    *arguments: str,
    selection: str = "",
) -> subprocess.CompletedProcess[str]:
    """Run the copied submission wrapper against isolated command stubs."""
    return subprocess.run(
        [str(harness.repository / "scripts" / "docker_job.sh"), *arguments],
        cwd=harness.repository,
        env=harness.environment,
        input=selection,
        text=True,
        capture_output=True,
        check=False,
    )


def _queue_log_dir(harness: _Harness, scope: str = "steady_flow") -> Path:
    """Return one host-visible experiment queue-log directory."""
    return Path(harness.environment["STORAGE_ROOT"]) / "03_experiments" / scope / "logs" / "queue"


def _assert_preflight_container(
    harness: _Harness,
    *,
    workflow: str,
    config_path: str,
) -> None:
    """Verify one read-only, CPU-only, network-disabled authoritative preflight."""
    arguments = _capture_arguments(harness.preflight_docker_capture)
    assert arguments[:2] == ["run", "--rm"]
    assert arguments[arguments.index("--network") + 1] == "none"
    assert arguments[arguments.index("--workdir") + 1] == "/workspace/repo"
    assert "--gpus" not in arguments
    assert "WANDB_API_KEY" not in arguments
    assert not any(argument.startswith("WANDB_") for argument in arguments)
    assert f"type=bind,source={harness.repository},target=/workspace/repo,readonly" in arguments
    assert f"type=bind,source={harness.environment['STORAGE_ROOT']},target=/workspace/storage" in arguments
    assert arguments[arguments.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
    assert "STORAGE_ROOT=/workspace/storage" in arguments
    assert not harness.host_python_capture.exists()
    assert arguments[-4:] == [
        "python",
        "/workspace/repo/scripts/config_preflight_runtime.py",
        workflow,
        config_path,
    ]
    assert arguments[-5]
    assert not arguments[-5].startswith("-")


def _assert_queue_path_container(harness: _Harness, *, scope: str) -> None:
    """Verify queue logs are resolved by common paths in a bounded container."""
    arguments = _capture_arguments(harness.path_docker_capture)
    assert arguments[:2] == ["run", "--rm"]
    assert arguments[arguments.index("--network") + 1] == "none"
    assert arguments[arguments.index("--workdir") + 1] == "/workspace/repo"
    assert "--gpus" not in arguments
    assert f"type=bind,source={harness.repository},target=/workspace/repo,readonly" in arguments
    assert f"type=bind,source={harness.environment['STORAGE_ROOT']},target=/workspace/storage" in arguments
    assert arguments[arguments.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
    assert "STORAGE_ROOT=/workspace/storage" in arguments
    assert not harness.host_python_capture.exists()
    assert arguments[-4:] == [
        "python",
        "-m",
        "src.common.common_queue_log_cli",
        scope,
    ]


def _without_device(arguments: list[str]) -> list[str]:
    """
    Remove the already-validated semantic device option from forwarded arguments.

    Split and equals spellings are handled here. Wrapper validation owns malformed
    or duplicate cases before the inner command is normalized to strict CUDA.
    """
    cleaned: list[str] = []
    index = 0
    while index < len(arguments):
        if arguments[index] == "--device":
            index += 2
        elif arguments[index].startswith("--device="):
            index += 1
        else:
            cleaned.append(arguments[index])
            index += 1
    return cleaned


def _assert_wandb_forwarding(
    harness: _Harness,
    result: subprocess.CompletedProcess[str],
    *,
    expected_key: str | None,
    possible_keys: tuple[str | None, ...],
) -> None:
    """
    Verify name-only Docker credential forwarding and output redaction.

    Captured stub environment may contain the selected key for assertion, but queue,
    Docker argv, launcher streams, and logs must never expose any candidate value.
    """
    docker = _capture_arguments(harness.docker_capture)
    _assert_queue_path_container(harness, scope="steady_flow")
    expected_capture = expected_key if expected_key is not None else "<unset>"
    assert harness.docker_environment_capture.read_text(encoding="utf-8") == expected_capture
    assert ("WANDB_API_KEY" in docker) is (expected_key is not None)
    assert not any(argument.startswith("WANDB_API_KEY=") for argument in docker)

    processed_root = Path(harness.environment["STORAGE_ROOT"]) / "03_experiments"
    log_text = "\n".join(path.read_text(encoding="utf-8") for path in processed_root.glob("*/logs/queue/*.log"))
    visible_text = "\n".join(("\0".join(docker), result.stdout, result.stderr, log_text))
    for key in possible_keys:
        if key:
            assert key not in visible_text


def _assert_submission(
    harness: _Harness,
    result: subprocess.CompletedProcess[str],
    *,
    gpu: str,
    workflow: str,
    module: str,
    semantic_arguments: list[str],
) -> Path:
    """Assert the scheduler and maintained CLI receive semantic arguments."""
    assert result.returncode == 0, result.stderr
    queued = _capture_arguments(harness.runtsgpu_capture)
    assert queued[:3] == [
        f"-g{gpu}",
        "--",
        str(harness.repository / "scripts" / "_docker_run.sh"),
    ]
    assert queued[3:5] == [gpu, workflow]
    log_path = Path(queued[5])
    assert log_path.parent == _queue_log_dir(harness)
    forwarded_arguments = list(semantic_arguments)
    if workflow in {"train", "optuna"} and not forwarded_arguments[0].startswith("/workspace/"):
        forwarded_arguments[0] = f"/workspace/repo/{forwarded_arguments[0]}"
    assert queued[6:] == forwarded_arguments
    _assert_queue_path_container(harness, scope="steady_flow")
    if workflow in {"train", "optuna"}:
        _assert_preflight_container(
            harness,
            workflow=workflow,
            config_path=forwarded_arguments[0],
        )

    docker = _capture_arguments(harness.docker_capture)
    assert docker[docker.index("--gpus") + 1] == f"device={gpu}"
    assert docker[docker.index("--workdir") + 1] == "/workspace/repo"
    assert "STORAGE_ROOT=/workspace/storage" in docker
    assert f"{harness.repository}:/workspace/repo:rw" in docker
    assert f"{harness.environment['STORAGE_ROOT']}:/workspace/storage:rw" in docker
    inner_arguments = [*_without_device(forwarded_arguments), "--device", "cuda"]
    assert docker[-(len(inner_arguments) + 3) :] == [
        "python",
        "-m",
        module,
        *inner_arguments,
    ]
    assert "--queue-gpu" not in queued
    assert "--queue-gpu" not in docker

    assert log_path.is_file()
    log_text = log_path.read_text(encoding="utf-8")
    assert "captured Docker stdout with spaces" in log_text
    assert "captured Docker stderr with spaces" in log_text
    return log_path


def test_dataset_builder_runs_synchronously_without_gpu_queue(
    tmp_path: Path,
) -> None:
    """Run the dataset builder in the bounded CPU-only container path."""
    harness = _harness(tmp_path)

    result = _run_job(
        harness,
        "build-datasets",
        _CAMPAIGN_CONFIG_RELATIVE,
    )

    assert result.returncode == 0, result.stderr
    docker = _capture_arguments(harness.docker_capture)
    assert docker[:2] == ["run", "--rm"]
    assert docker[docker.index("--network") + 1] == "none"
    assert docker[docker.index("--user") + 1] == (f"{os.getuid()}:{os.getgid()}")
    assert docker[docker.index("--workdir") + 1] == "/workspace/repo"
    assert "--gpus" not in docker
    assert f"type=bind,source={harness.repository},target=/workspace/repo,readonly" in docker
    assert f"type=bind,source={harness.environment['STORAGE_ROOT']},target=/workspace/storage" in docker
    assert docker[-7:] == [
        "python",
        "-m",
        "src.datasets.dataset_packages",
        "build",
        f"/workspace/repo/{_CAMPAIGN_CONFIG_RELATIVE}",
        "--storage-root",
        "/workspace/storage",
    ]
    assert not harness.nvidia_capture.exists()
    assert not harness.runtsgpu_capture.exists()
    assert not harness.path_docker_capture.exists()


def test_direct_submission_uses_automatic_gpu_and_forwards_arguments(
    tmp_path: Path,
) -> None:
    """Forward a valid direct request while stripping wrapper-only options."""
    harness = _harness(tmp_path)
    arguments = [
        _DIRECT_CONFIG_RELATIVE,
        "--output-root",
        "/workspace/storage/03_experiments/synthetic output",
    ]

    result = _run_job(
        harness,
        "--queue-gpu",
        "auto",
        "train",
        *arguments,
    )

    _assert_submission(
        harness,
        result,
        gpu="2",
        workflow="train",
        module="src.experiments.cli.cli_train",
        semantic_arguments=arguments,
    )


def test_optuna_submission_uses_explicit_gpu_and_forwards_arguments(
    tmp_path: Path,
) -> None:
    """Forward a valid Optuna request to the explicitly selected device."""
    harness = _harness(
        tmp_path,
        preflight_summary=f"optuna\tsteady_flow\t{_OPTUNA_CONFIG_RELATIVE}",
    )
    arguments = [
        _OPTUNA_CONFIG_RELATIVE,
        "--n-trials",
        "1",
        "--show-progress-bar",
    ]

    result = _run_job(
        harness,
        "optuna",
        *arguments,
        "--queue-gpu",
        "0",
    )

    _assert_submission(
        harness,
        result,
        gpu="0",
        workflow="optuna",
        module="src.experiments.cli.cli_optuna",
        semantic_arguments=arguments,
    )


def test_malformed_gpu_report_fails_before_submission(tmp_path: Path) -> None:
    """Reject malformed automatic-selection evidence without queuing."""
    harness = _harness(tmp_path, gpu_report="not a gpu record\n")

    result = _run_job(
        harness,
        "--queue-gpu",
        "auto",
        "train",
        _DIRECT_CONFIG_RELATIVE,
    )

    assert result.returncode != 0
    assert not harness.runtsgpu_capture.exists()
    assert not harness.docker_capture.exists()


@pytest.mark.parametrize("missing_command", ["nvidia-smi", "runTSGPU.py"])
def test_missing_queue_infrastructure_fails_before_submission(
    tmp_path: Path,
    missing_command: str,
) -> None:
    """Fail clearly before queuing when a required executable is unavailable."""
    harness = _harness(tmp_path)
    (harness.binary_dir / missing_command).unlink()

    result = _run_job(
        harness,
        "--queue-gpu",
        "auto",
        "train",
        _DIRECT_CONFIG_RELATIVE,
    )

    assert result.returncode != 0
    assert not harness.runtsgpu_capture.exists()


@pytest.mark.parametrize(
    ("workflow", "config_path"),
    [
        ("train", _OPTUNA_CONFIG_RELATIVE),
        ("optuna", _DIRECT_CONFIG_RELATIVE),
    ],
)
def test_wrong_config_family_fails_before_gpu_and_queue(
    tmp_path: Path,
    workflow: str,
    config_path: str,
) -> None:
    """Reject workflow/config-family mismatches before GPU allocation."""
    harness = _harness(tmp_path)

    result = _run_job(harness, workflow, config_path)

    assert result.returncode == 2
    assert not harness.nvidia_capture.exists()
    assert not harness.runtsgpu_capture.exists()
    assert not _queue_log_dir(harness).exists()


def test_host_paths_are_translated_to_container_domains(tmp_path: Path) -> None:
    """Translate repository configs and host storage paths to container paths."""
    harness = _harness(
        tmp_path,
        preflight_summary=f"experiment\tsteady_flow\t{_DIRECT_CONFIG_RELATIVE}",
    )
    resume = Path(harness.environment["STORAGE_ROOT"]) / "03_experiments/steady_flow/runs/synthetic run"
    resume.mkdir(parents=True)
    output = Path(harness.environment["STORAGE_ROOT"]) / "03_experiments/new output"
    arguments = [
        _DIRECT_REPOSITORY_RELATIVE,
        "--resume",
        str(resume),
        "--output-root",
        str(output),
    ]
    expected = [
        _DIRECT_CONFIG_RELATIVE,
        "--resume",
        "/workspace/storage/03_experiments/steady_flow/runs/synthetic run",
        "--output-root",
        "/workspace/storage/03_experiments/new output",
    ]

    result = _run_job(
        harness,
        "--queue-gpu",
        "auto",
        "train",
        *arguments,
    )

    _assert_submission(
        harness,
        result,
        gpu="2",
        workflow="train",
        module="src.experiments.cli.cli_train",
        semantic_arguments=expected,
    )


def test_scheduler_submission_failure_is_propagated(tmp_path: Path) -> None:
    """Return the scheduler failure without claiming successful admission."""
    harness = _harness(
        tmp_path,
        preflight_summary=f"experiment\tsteady_flow\t{_DIRECT_CONFIG_RELATIVE}",
    )
    harness.environment["QUEUE_SUBMISSION_EXIT"] = "37"
    harness.environment["QUEUE_PARTIAL_OUTPUT"] = "synthetic scheduler diagnostic\n"

    result = _run_job(
        harness,
        "--queue-gpu",
        "auto",
        "train",
        _DIRECT_CONFIG_RELATIVE,
    )

    assert result.returncode == 37
    assert "synthetic scheduler diagnostic" in result.stderr
    assert harness.runtsgpu_capture.exists()
    assert not harness.docker_capture.exists()


def test_wandb_credential_is_forwarded_by_name_without_disclosure(
    tmp_path: Path,
) -> None:
    """Forward a credential through environment indirection without printing it."""
    secret = "synthetic-queue-credential"  # noqa: S105 - redaction sentinel
    harness = _harness(tmp_path, exported_key=secret)

    result = _run_job(
        harness,
        "--queue-gpu",
        "auto",
        "artifacts",
        "--task",
        "steady_flow",
    )

    assert result.returncode == 0, result.stderr
    _assert_wandb_forwarding(
        harness,
        result,
        expected_key=secret,
        possible_keys=(secret,),
    )
