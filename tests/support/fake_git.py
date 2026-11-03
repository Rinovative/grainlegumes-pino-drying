"""Provide controlled read-only Git responses for source-admission fixtures.

Responsibilities:
  - Install a fake Git executable and report source-byte, mode, untracked-file, and HEAD changes
  - Serve the tracked file list and a tar of the installation-time bytes so snapshots and source trees can be materialized
  - Install a Git stub that fails and records every call, proving a code path never reaches Git

Design principles:
  - Only supported read queries are admitted; source drift is measured from real fixture bytes

This module does NOT:
  - Mutate real Git indexes/history or test Git diff semantics
"""

from __future__ import annotations

import io
import json
import os
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

FAKE_GIT_HEAD = "b" * 40


def _inventory(repository: Path) -> dict[str, str]:
    files = {}
    for path in sorted(repository.rglob("*")):
        relative = path.relative_to(repository).as_posix()
        if relative == ".git" or relative.startswith(".git/"):
            continue
        if path.is_symlink():
            files[relative] = "link:" + str(path.readlink())
        elif path.is_file():
            files[relative] = f"{bool(path.stat().st_mode & 0o111)}:" + path.read_bytes().hex()
    return files


@dataclass
class FakeGit:
    """Control only the commit identity; source drift is observed from real bytes."""

    state_path: Path

    def change_head(self, head: str) -> None:
        """Simulate a different admitted commit without writing Git history."""
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        state["head"] = head
        self.state_path.write_text(json.dumps(state), encoding="utf-8")


def install_fake_git(repository: Path, binary: Path) -> FakeGit:
    """Install a fail-closed fake for the exact read queries used by workers."""
    binary.mkdir(parents=True, exist_ok=True)
    (repository / ".git").mkdir(exist_ok=True)
    state_path = binary / "git-state.json"
    state_path.write_text(
        json.dumps({"root": str(repository), "head": FAKE_GIT_HEAD, "baseline": _inventory(repository)}),
        encoding="utf-8",
    )
    executable = binary / "git"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import runpy, sys\n"
        f"sys.argv.insert(1, {str(state_path)!r})\n"
        f"runpy.run_path({str(Path(__file__).resolve())!r}, run_name='__main__')\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return FakeGit(state_path)


def install_failing_git(binary: Path) -> Path:
    """Install a Git stub that records its arguments and fails; return the call log."""
    binary.mkdir(parents=True, exist_ok=True)
    log = binary / "git-calls.log"
    executable = binary / "git"
    executable.write_text(f'#!/bin/sh\nprintf "%s\\n" "$*" >> {str(log)!r}\nexit 99\n', encoding="utf-8")
    executable.chmod(0o755)
    return log


def _write_archive(baseline: dict[str, str]) -> None:
    """Write the installation-time tracked bytes as the tar that ``git archive`` would produce."""
    with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as archive:
        for name, value in sorted(baseline.items()):
            member = tarfile.TarInfo(name)
            if value.startswith("link:"):
                member.type = tarfile.SYMTYPE
                member.linkname = value.removeprefix("link:")
                archive.addfile(member)
                continue
            executable, _, content = value.partition(":")
            payload = bytes.fromhex(content)
            member.size = len(payload)
            member.mode = 0o755 if executable == "True" else 0o644
            archive.addfile(member, io.BytesIO(payload))


def _main(state_path: Path, arguments: list[str]) -> None:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    repository = Path(state["root"])
    args = list(arguments)
    if args and args[0] == "--no-optional-locks":
        args.pop(0)
    if args[:1] == ["-C"]:
        if Path(args[1]).resolve() != repository.resolve():
            message = "Fake Git received a different repository"
            raise SystemExit(message)
        del args[:2]
    if args == ["rev-parse", "HEAD"]:
        print(state["head"])
        return
    if args == ["rev-parse", "--show-toplevel"]:
        print(repository)
        return
    current = _inventory(repository)
    baseline = state["baseline"]
    changed = {name: current.get(name) for name, value in baseline.items() if current.get(name) != value}
    added = sorted(set(current) - set(baseline))
    if args == ["diff", "--no-ext-diff", "--no-textconv", "--binary", "HEAD", "--"]:
        if changed:
            print(json.dumps(changed, sort_keys=True))
    elif args == ["ls-files", "--others", "--exclude-standard", "-z"]:
        sys.stdout.buffer.write(b"".join(os.fsencode(name) + b"\0" for name in added))
    elif args in (["ls-files", "-z"], ["ls-tree", "-r", "-z", "--name-only", state["head"]]):
        sys.stdout.buffer.write(b"".join(os.fsencode(name) + b"\0" for name in sorted(baseline)))
    elif args == ["ls-files", "--others", "--ignored", "--exclude-standard", "--directory", "-z"]:
        return
    elif args == ["archive", "--format=tar", state["head"]]:
        _write_archive(baseline)
    elif args == ["status", "--porcelain=v1", "--untracked-files=all"]:
        for name in sorted(changed):
            print(f" M {name}")
        for name in added:
            print(f"?? {name}")
    else:
        message = f"Unsupported fake Git read command: {args!r}"
        raise SystemExit(message)


if __name__ == "__main__":
    _main(Path(sys.argv[1]), sys.argv[2:])
