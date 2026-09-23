"""
source_fingerprint.py

Fingerprint the current Git worktree for scheduler admission.

Responsibilities:
  - Hash tracked changes against HEAD and untracked source contents
  - Expose one command-line fingerprint for submission and worker checks

Design principles:
  - Preserve exact path and content ordering in the existing ML fingerprint
  - Reject unreadable or unsupported untracked source paths

This module does NOT:
  - Establish scientific run identity or persist provenance
"""

import hashlib
import os
import stat
import subprocess
import sys
from pathlib import Path


def source_fingerprint(root: Path) -> str:
    """Return the SHA256 of tracked changes and untracked source contents."""
    tracked = subprocess.run(
        ["git", "-C", str(root), "diff", "--no-ext-diff", "--no-textconv", "--binary", "HEAD", "--"],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    untracked = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard", "-z"],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    digest = hashlib.sha256()
    digest.update(b"tracked-diff\0")
    digest.update(len(tracked).to_bytes(8, "big"))
    digest.update(tracked)
    for raw in sorted(part for part in untracked.split(b"\0") if part):
        path = root / os.fsdecode(raw)
        digest.update(b"untracked\0")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
        if path.is_symlink():
            value = os.fsencode(os.readlink(path))
            digest.update(b"symlink\0")
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
        elif path.is_file():
            digest.update(b"file\0")
            digest.update((path.stat().st_mode & stat.S_IXUSR).to_bytes(1, "big"))
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        else:
            raise ValueError(f"Cannot fingerprint source path: {path}")
    return digest.hexdigest()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} REPOSITORY")
    print(source_fingerprint(Path(sys.argv[1])))
