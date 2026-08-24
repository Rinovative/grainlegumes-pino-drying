#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
  echo "Usage: $0 <scope/queue-label>" >&2
  exit 2
fi

QUEUE_TOKEN="$1"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd -P)"
HOST_STORAGE_ROOT="${STORAGE_ROOT:-${PROJECT_DIR}/../storage}"
STORAGE_DIR="$(cd "${HOST_STORAGE_ROOT}" && pwd -P)"

if [[ ! "${QUEUE_TOKEN}" =~ ^([A-Za-z0-9][A-Za-z0-9._-]{0,63})/([A-Za-z0-9][A-Za-z0-9._+-]{0,95})$ ]]; then
  echo "Queue descriptor token is invalid." >&2
  exit 2
fi
QUEUE_SCOPE="${BASH_REMATCH[1]}"
QUEUE_LABEL="${BASH_REMATCH[2]}"
DESCRIPTOR_PATH="${STORAGE_DIR}/03_experiments/${QUEUE_SCOPE}/logs/queue/${QUEUE_LABEL}.queue.json"
if [[ -L "${DESCRIPTOR_PATH}" || ! -f "${DESCRIPTOR_PATH}" || ! -r "${DESCRIPTOR_PATH}" ]]; then
  echo "Queue descriptor must be a readable non-symlink regular file." >&2
  exit 2
fi
DESCRIPTOR_REALPATH="$(realpath -e -- "${DESCRIPTOR_PATH}")"
if [[ "${DESCRIPTOR_REALPATH}" != "${STORAGE_DIR}/"* ]]; then
  echo "Queue descriptor must remain inside the configured storage root." >&2
  exit 2
fi

mapfile -d '' -t EXECUTION_ARGV < <(
  python -c '
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

path = Path(sys.argv[1])
project_root = sys.argv[2]
storage_root = sys.argv[3]
expected = {
    "schema_version", "created_utc", "workflow", "task", "canonical_config",
    "config_sha256", "host_gpu", "container_gpu", "task_spooler_socket", "host_log",
    "project_root", "storage_root", "queue_log_root", "source_commit",
    "source_worktree_sha256", "execution_argv",
}
try:
    file_status = path.stat()
    if file_status.st_uid != os.getuid() or stat.S_IMODE(file_status.st_mode) & 0o222:
        raise SystemExit("Queue descriptor permissions are unsafe.")
    payload = json.loads(path.read_text(encoding="utf-8"))
except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
    raise SystemExit(f"Queue descriptor is malformed: {error}") from error
if not isinstance(payload, dict) or set(payload) != expected:
    raise SystemExit("Queue descriptor has an unsupported schema.")
if payload["schema_version"] != 1:
    raise SystemExit("Queue descriptor schema version is unsupported.")
if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", payload.get("created_utc", "")):
    raise SystemExit("Queue descriptor creation time is invalid.")
if payload["project_root"] != project_root or payload["storage_root"] != storage_root:
    raise SystemExit("Queue descriptor roots do not match this checkout.")
for field in (
    "created_utc", "workflow", "task", "canonical_config", "host_gpu", "container_gpu",
    "task_spooler_socket", "host_log", "queue_log_root", "source_commit", "source_worktree_sha256",
):
    if not isinstance(payload[field], str) or not payload[field]:
        raise SystemExit(f"Queue descriptor field {field!r} is invalid.")
if payload["workflow"] not in {"train", "optuna", "artifacts"}:
    raise SystemExit("Queue descriptor workflow is unsupported.")
if not re.fullmatch(r"(?:0|[1-9][0-9]*)", payload["host_gpu"]):
    raise SystemExit("Queue descriptor GPU is invalid.")
if payload["container_gpu"] != "0" or payload["task_spooler_socket"] != "/etc/ts/socket_{}".format(payload["host_gpu"]):
    raise SystemExit("Queue descriptor GPU mapping is invalid.")
if not re.fullmatch(r"[0-9a-f]{40}", payload["source_commit"]):
    raise SystemExit("Queue descriptor source commit is invalid.")
if not re.fullmatch(r"[0-9a-f]{64}", payload["source_worktree_sha256"]):
    raise SystemExit("Queue descriptor source-worktree fingerprint is invalid.")
if payload["config_sha256"] is not None and not re.fullmatch(r"[0-9a-f]{64}", payload["config_sha256"]):
    raise SystemExit("Queue descriptor config hash is invalid.")
def source_worktree_sha256(root: Path) -> str:
    tracked_diff = subprocess.run(
        ["git", "-C", str(root), "diff", "--no-ext-diff", "--no-textconv", "--binary", "HEAD", "--"],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    untracked_output = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard", "-z"],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    digest = hashlib.sha256()
    digest.update(b"tracked-diff\0")
    digest.update(len(tracked_diff).to_bytes(8, "big"))
    digest.update(tracked_diff)
    for relative_raw in sorted(part for part in untracked_output.split(b"\0") if part):
        candidate = root / os.fsdecode(relative_raw)
        digest.update(b"untracked\0")
        digest.update(len(relative_raw).to_bytes(8, "big"))
        digest.update(relative_raw)
        if candidate.is_symlink():
            target = os.fsencode(os.readlink(candidate))
            digest.update(b"symlink\0")
            digest.update(len(target).to_bytes(8, "big"))
            digest.update(target)
        elif candidate.is_file():
            digest.update(b"file\0")
            digest.update((candidate.stat().st_mode & stat.S_IXUSR).to_bytes(1, "big"))
            with candidate.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        else:
            raise SystemExit(f"Unable to fingerprint untracked source path: {candidate}")
    return digest.hexdigest()


current_commit = subprocess.run(
    ["git", "-C", project_root, "rev-parse", "HEAD"],
    check=True,
    stdout=subprocess.PIPE,
    text=True,
).stdout.strip()
if current_commit != payload["source_commit"]:
    raise SystemExit("Queue descriptor source commit no longer matches this checkout.")
if source_worktree_sha256(Path(project_root)) != payload["source_worktree_sha256"]:
    raise SystemExit("Queue descriptor source worktree no longer matches this checkout.")

queue_log_root = Path(payload["queue_log_root"])
host_log = Path(payload["host_log"])
if not queue_log_root.is_absolute() or not host_log.is_absolute() or queue_log_root not in host_log.parents:
    raise SystemExit("Queue descriptor host log is outside its queue-log root.")
if path.parent != queue_log_root:
    raise SystemExit("Queue descriptor is outside its declared queue-log root.")
if not str(queue_log_root).startswith(storage_root + "/") or not str(host_log).startswith(storage_root + "/"):
    raise SystemExit("Queue descriptor paths are outside the storage root.")
if payload["workflow"] == "artifacts":
    if payload["config_sha256"] is not None:
        raise SystemExit("Artifact queue descriptors must not contain a config hash.")
else:
    if payload["config_sha256"] is None:
        raise SystemExit("Training queue descriptors require a config hash.")
    if payload["canonical_config"].startswith("/workspace/repo/"):
        config_path = Path(project_root, payload["canonical_config"].removeprefix("/workspace/repo/"))
    elif payload["canonical_config"].startswith("/workspace/storage/"):
        config_path = Path(storage_root, payload["canonical_config"].removeprefix("/workspace/storage/"))
    else:
        raise SystemExit("Queue descriptor config path is outside the mounted roots.")
    if not config_path.is_file() or hashlib.sha256(config_path.read_bytes()).hexdigest() != payload["config_sha256"]:
        raise SystemExit("Queue descriptor config no longer matches its recorded hash.")
argv = payload["execution_argv"]
if not isinstance(argv, list) or len(argv) < 4 or any(not isinstance(value, str) or "\x00" in value for value in argv):
    raise SystemExit("Queue descriptor execution arguments are invalid.")
if argv[:4] != [f"{project_root}/scripts/_docker_run.sh", payload["host_gpu"], payload["workflow"], payload["host_log"]]:
    raise SystemExit("Queue descriptor execution contract is invalid.")
sys.stdout.buffer.write(b"\0".join(value.encode("utf-8") for value in argv) + b"\0")
' "${DESCRIPTOR_REALPATH}" "${PROJECT_DIR}" "${STORAGE_DIR}"
)

if (( ${#EXECUTION_ARGV[@]} < 4 )); then
  echo "Queue descriptor execution arguments are missing." >&2
  exit 2
fi
exec "${EXECUTION_ARGV[@]}"
