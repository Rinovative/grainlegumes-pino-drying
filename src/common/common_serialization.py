"""Publish authoritative files atomically and compute stable content digests.

Responsibilities:
  - Write unique sibling temporary files and atomically replace destinations
  - Flush file data and parent-directory metadata where the platform supports it
  - Remove unpublished temporary files after handled failures
  - Provide canonical JSON and file SHA-256 helpers for lifecycle identities
  - Open files whose digest was verified on the same handle the caller parses

Design principles:
  - A failed publication never damages the published destination
  - Same-directory temporary files keep replacement on one filesystem
  - Callers decide whether replacement is permitted by their lifecycle contract

This module does NOT:
  - Enforce checkpoint, experiment, dataset, or artifact schemas
  - Refuse overwrite of an existing destination on the caller's behalf
  - Turn several independently published files into one transaction
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import tempfile
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

PathWriter = Callable[[Path], None]

_ATOMIC_TEMPORARY_SUFFIX = ".tmp"
_FILE_HASH_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class VerifiedFile:
    """One open binary file and the identity under which its complete bytes were verified."""

    stream: BinaryIO
    identity: tuple[int, int, int, int]

    def require_unchanged(self) -> None:
        """Refuse further reads once the open file's device, inode, size or modification time differ from the verified identity."""
        if _descriptor_identity(self.stream) != self.identity:
            msg = f"File changed after its digest check: {self.stream.name}"
            raise OSError(msg)


def _descriptor_identity(stream: BinaryIO) -> tuple[int, int, int, int]:
    """Return device, inode, size and modification time of one open file."""
    status = os.fstat(stream.fileno())
    return (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns)


def atomic_write_temporary_destination(candidate: Path | str) -> Path | None:
    """
    Return the destination named by one atomic-write temporary sibling.

    Parameters
    ----------
    candidate : Path | str
        Candidate sibling path to classify lexically.

    Returns
    -------
    Path | None
        The same-directory destination encoded by atomic_path_write, or None
        when the candidate does not match that naming contract.

    Notes
    -----
    This function classifies only the common atomic naming contract. Callers
    must still validate the candidate file type and require the derived
    destination name to belong to their authoritative domain.

    """
    candidate_path = Path(candidate)
    name = candidate_path.name
    if not name.startswith(".") or not name.endswith(_ATOMIC_TEMPORARY_SUFFIX):
        return None
    body = name[1 : -len(_ATOMIC_TEMPORARY_SUFFIX)]
    destination_name, separator, random_name = body.rpartition(".")
    if not separator or not destination_name or not random_name:
        return None
    return candidate_path.with_name(destination_name)


def _fsync_file(path: Path) -> None:
    """
    Flush one completed temporary file before it becomes authoritative.

    Flush failures propagate so publication cannot silently replace the
    destination with content whose file data was not synchronized.
    """
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_parent(path: Path) -> None:
    """
    Best-effort flush a parent directory after atomic replacement.

    Some supported filesystems cannot open or synchronize directory
    descriptors. Those platform limitations are ignored after ``os.replace``.
    Content atomicity still holds, but crash durability of the directory entry
    is then filesystem-dependent.
    """
    flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY
    try:
        descriptor = os.open(path, flags)
    except OSError:
        # Directory descriptors are not available on every supported filesystem.
        return
    try:
        os.fsync(descriptor)
    except OSError:
        # Atomic replacement still holds when directory durability flushing is unsupported.
        return
    finally:
        os.close(descriptor)


def atomic_path_write(destination: Path | str, writer: PathWriter) -> Path:
    """
    Publish a file produced by ``writer`` through an atomic sibling replace.

    Parameters
    ----------
    destination : Path | str
        Final file path. Its parent is created when necessary.
    writer : Callable[[Path], None]
        Callback that must completely write the supplied temporary path.

    Returns
    -------
    Path
        The published destination.

    Raises
    ------
    RuntimeError
        If the writer removes or fails to create the temporary file.
    Exception
        Any writer, flush, or replacement failure after best-effort cleanup.

    Notes
    -----
    An existing destination is replaced. This helper owns no overwrite guard.
    Replacement is atomic when the destination filesystem honors ``os.replace``.
    The temporary file is always allocated in the same parent, flushed before
    publication, and removed on handled failure. Parent-directory flushing is
    best effort on filesystems that do not support directory descriptors.

    """
    final_path = Path(destination)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp_path = tempfile.mkstemp(
        dir=final_path.parent,
        prefix=f".{final_path.name}.",
        suffix=_ATOMIC_TEMPORARY_SUFFIX,
    )
    os.close(descriptor)
    temp_path = Path(raw_temp_path)
    try:
        writer(temp_path)
        if not temp_path.is_file():
            msg = f"Atomic writer did not produce its temporary file: {temp_path}"
            raise RuntimeError(msg)  # noqa: TRY301
        _fsync_file(temp_path)
        temp_path.replace(final_path)
        _fsync_parent(final_path.parent)
    except BaseException:
        with suppress(OSError):
            temp_path.unlink(missing_ok=True)
        raise
    return final_path


def atomic_write_bytes(destination: Path | str, payload: bytes) -> Path:
    """
    Atomically publish a bytes payload.

    Parameters
    ----------
    destination : Path | str
        Final file path.
    payload : bytes
        Complete serialized payload.

    Returns
    -------
    Path
        Published destination.

    Notes
    -----
    An existing destination is replaced through ``atomic_path_write``. Write,
    flush, and replacement failures propagate after temporary-file cleanup.

    """

    def write_bytes(temp_path: Path) -> None:
        temp_path.write_bytes(payload)

    return atomic_path_write(destination, write_bytes)


def atomic_write_text(destination: Path | str, payload: str) -> Path:
    """
    Atomically publish a UTF-8 text payload.

    Parameters
    ----------
    destination : Path | str
        Final file path.
    payload : str
        Complete text payload.

    Returns
    -------
    Path
        Published destination.

    Notes
    -----
    An existing destination is replaced through ``atomic_path_write``. Encoding,
    write, flush, and replacement failures propagate after temporary cleanup.

    """

    def write_text(temp_path: Path) -> None:
        temp_path.write_text(payload, encoding="utf-8")

    return atomic_path_write(destination, write_text)


def atomic_write_json(destination: Path | str, payload: Mapping[str, Any]) -> Path:
    """
    Atomically publish a JSON object with a final newline.

    Parameters
    ----------
    destination : Path | str
        Final file path.
    payload : Mapping[str, Any]
        JSON-serializable object.

    Returns
    -------
    Path
        Published destination.

    Raises
    ------
    TypeError, ValueError
        If ``payload`` cannot be serialized by ``json.dumps``.

    Notes
    -----
    Keys are sorted, non-ASCII is escaped, and one final newline is appended.
    An existing destination is replaced atomically. Serialization and filesystem
    failures propagate without damaging the previous published file.

    """
    serialized = json.dumps(
        dict(payload),
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )
    return atomic_write_text(destination, f"{serialized}\n")


def atomic_torch_save(payload: Any, destination: Path | str) -> Path:
    """
    Atomically publish one ``torch.save`` payload.

    Parameters
    ----------
    payload : Any
        PyTorch-serializable object.
    destination : Path | str
        Final checkpoint or tensor-artifact path.

    Returns
    -------
    Path
        Published destination.

    Notes
    -----
    An existing destination is replaced through ``atomic_path_write``. PyTorch
    serialization, flush, and filesystem failures propagate after cleanup. This
    helper does not define or validate checkpoint/artifact schema.

    """
    import torch  # noqa: PLC0415

    def write_torch(temp_path: Path) -> None:
        with temp_path.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())

    return atomic_path_write(destination, write_torch)


def canonical_json_sha256(payload: Any) -> str:
    """
    Return a SHA-256 digest of canonical JSON-compatible content.

    Parameters
    ----------
    payload : Any
        JSON-compatible content.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256 digest.

    Raises
    ------
    TypeError, ValueError
        If ``payload`` cannot be encoded by ``json.dumps``.

    Notes
    -----
    Keys are sorted with compact separators and non-ASCII escaping. This helper
    follows Python JSON encoding semantics, including its default handling of
    non-finite floating values. Callers needing a stricter persistence schema
    must validate it before hashing.

    """
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _stream_sha256(stream: BinaryIO) -> str:
    """Return the SHA-256 digest of the bytes a binary stream yields from its current position."""
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(_FILE_HASH_CHUNK_SIZE), b""):
        digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: Path | str) -> str:
    """
    Return the SHA-256 digest of one file's exact bytes, streamed in one-MiB chunks.

    Parameters
    ----------
    path : Path | str
        Existing file path.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256 digest.

    Raises
    ------
    OSError
        If the file cannot be opened or read.

    """
    with Path(path).open("rb") as stream:
        return _stream_sha256(stream)


_FILE_DIGESTS: dict[str, tuple[tuple[int, int, int, int], str]] = {}


def _descriptor_sha256(file: Path, stream: BinaryIO) -> tuple[tuple[int, int, int, int], str]:
    """Return the identity and digest of one open file, reusing the cache only for that descriptor's identity."""
    identity = _descriptor_identity(stream)
    cached = _FILE_DIGESTS.get(str(file))
    if cached is not None and cached[0] == identity:
        return cached
    digest = _stream_sha256(stream)
    if _descriptor_identity(stream) != identity:
        msg = f"File changed while it was hashed: {file}"
        raise OSError(msg)
    _FILE_DIGESTS[str(file)] = (identity, digest)
    return identity, digest


def cached_file_sha256(path: Path | str) -> str:
    """Return a file's SHA-256 digest, hashing each file identity at most once per process.

    The digest and the identity it is cached under come from one open descriptor, and the
    digest is reused while the path, device, inode, size and modification time are unchanged.
    A same-size in-place edit that restores the modification time would evade this; no writer does that.
    """
    file = Path(path)
    with file.open("rb") as stream:
        return _descriptor_sha256(file, stream)[1]


@contextmanager
def open_verified_file(path: Path | str, expected_sha256: str, mismatch_error: Exception) -> Iterator[VerifiedFile]:
    """Yield one open file whose complete bytes were checked against ``expected_sha256``.

    The digest belongs to the open descriptor, so a path replaced after the check cannot
    change the bytes the caller parses from the handle. An in-place rewrite of the same file
    is visible only through its identity, so the caller calls ``require_unchanged`` after each
    read it relies on. The cache is shared with ``cached_file_sha256`` and is reused only for
    the descriptor's own identity.

    Raises
    ------
    Exception
        ``mismatch_error`` when the digest differs; the caller owns its type and message.
    OSError
        If the file cannot be opened or read, or changed while it was hashed.

    """
    file = Path(path)
    with file.open("rb") as stream:
        identity, digest = _descriptor_sha256(file, stream)
        if digest != expected_sha256:
            raise mismatch_error
        stream.seek(0)
        yield VerifiedFile(stream, identity)


def python_implementation_sha256(root: Path, paths: Sequence[str]) -> str:
    """Hash selected Python implementations without presentation-only changes.

    Parameters
    ----------
    root : Path
        Source root containing every selected file.
    paths : Sequence[str]
        Nonempty selection of relative Python file paths. The domain caller
        selects its scientific dependencies; this helper does not discover them.

    Returns
    -------
    str
        Digest of relative names and syntax trees, excluding source positions,
        comments and module, class and function docstrings.

    Raises
    ------
    ValueError
        The selection is empty or a path is not a contained relative Python file.
    OSError, SyntaxError
        A selected implementation cannot be read or parsed.

    Notes
    -----
    This detects changes in the selected implementation, not mathematical
    equivalence or dependencies omitted by the caller. Exact producing bytes
    and environments remain separate reproduction evidence.

    """
    if not paths:
        msg = "Python implementation identity requires selected files."
        raise ValueError(msg)
    source_root = root.resolve(strict=True)
    implementations = {}
    for name in sorted(set(paths)):
        relative = Path(name)
        source = source_root / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.suffix != ".py"
            or not source.resolve(strict=True).is_relative_to(source_root)
        ):
            msg = f"Implementation path must be a contained relative Python file: {name}"
            raise ValueError(msg)
        with warnings.catch_warnings():
            # Third-party sources such as neuraloperator docstrings contain LaTeX
            # escapes (a backslash before "hat") whose SyntaxWarning does not change the parsed tree.
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                body = node.body
                if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                    del body[0]
        implementations[relative.as_posix()] = ast.dump(tree, include_attributes=False)
    return canonical_json_sha256(implementations)
