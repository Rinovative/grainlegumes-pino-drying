"""Fit or reuse the Train-only scaling state of both tasks through one identity-bound cache and one run artifact.

Responsibilities:
  - Derive the scaling identity from the admitted split's manifest binding, Train membership and preprocessing specification
  - Reuse a cached scaling artifact only under its exact identity and model implementation, and publish new entries atomically under a lock
  - Build and admit the run's persisted scaling artifact (``normalizer.pt``) the same way for both tasks, with its producer evidence
  - Read every scaling artifact file, run artifact and cache entry alike, through one safe loader

Design principles:
  - Scaling is fitted on Train membership only; the identity binds the Dataset manifest digest and that membership
  - The envelope records the model implementation digest that fitted it; the cache key binds the identity and that digest, so
    changed fit or application code refits once per Dataset while a loss-only change reuses the entry
  - The artifact identity stays data-only: a restored run artifact is admitted against its recorded evidence, never the running code
  - A cache entry whose identity or producer disagrees with its key is refused, never reused or silently refitted
  - The task owns its fitted state and preprocessing specification; this module owns identity, cache and envelope

This module does NOT:
  - Compute statistics or reconstruct processors (normalization owns steady, learning owns transient state)
  - Choose split membership or construct DataLoaders
  - Write run directories; the run lifecycle persists the admitted artifact
"""

from __future__ import annotations

import copy
import pickle
import re
import time
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any, Final

import torch

from src import common

if TYPE_CHECKING:
    from pathlib import Path

    from src.datasets.preprocessing import dataset_preprocessing_splits as splits

SCALING_ARTIFACT_SCHEMA_KIND: Final = "train_scaling_artifact"
_ARTIFACT_KEYS: Final = frozenset({"schema_kind", "identity", "identity_sha256", "producer", "state"})
_SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}")
StartupCallback = Callable[[str, Mapping[str, Any]], None]


def scaling_identity(split: splits.SplitContract, preprocessing: Mapping[str, Any]) -> dict[str, Any]:
    """Return the scaling identity: task contract, training manifest and payload identity, Train membership and preprocessing."""
    return copy.deepcopy(
        {
            "task": split.task,
            "task_contract_digest": split.task_contract_digest,
            "dataset": {key: value for key, value in split.train.dataset.items() if key != "case_ids"},
            "train_membership_digest": split.train.membership_digest,
            "preprocessing": dict(preprocessing),
        }
    )


def build_scaling_artifact(identity: Mapping[str, Any], state: Mapping[str, Any], *, model_sha256: str) -> dict[str, Any]:
    """Wrap one task-fitted scaling state in its identity-bound envelope with the model digest of the code that fitted it."""
    return {
        "schema_kind": SCALING_ARTIFACT_SCHEMA_KIND,
        "identity": copy.deepcopy(dict(identity)),
        "identity_sha256": common.serialization.canonical_json_sha256(dict(identity)),
        "producer": {"model_sha256": model_sha256},
        "state": dict(state),
    }


def _cache_key(identity: Mapping[str, Any], producer: Mapping[str, Any]) -> str:
    """Return the cache entry name of one scaling identity and producer."""
    return common.serialization.canonical_json_sha256({"identity": dict(identity), "producer": dict(producer)})


def admit_scaling_artifact(value: Any, *, identity: Mapping[str, Any]) -> dict[str, Any]:
    """
    Admit one persisted or cached scaling artifact under its exact expected identity.

    Parameters
    ----------
    value : Any
        Loaded artifact envelope.
    identity : Mapping[str, Any]
        Expected scaling identity (:func:`scaling_identity`).

    Returns
    -------
    dict[str, Any]
        The task-fitted state; the task reconstructs and validates it.

    Raises
    ------
    ValueError
        If the envelope schema, its identity digest, its recorded producer evidence or its identity differ from the
        expectation. The recorded model digest is evidence and is never compared with the running code here.

    """
    if (
        not isinstance(value, dict)
        or set(value) != _ARTIFACT_KEYS
        or value["schema_kind"] != SCALING_ARTIFACT_SCHEMA_KIND
        or not isinstance(value["identity"], dict)
        or value["identity_sha256"] != common.serialization.canonical_json_sha256(value["identity"])
        or not isinstance(value["producer"], dict)
        or set(value["producer"]) != {"model_sha256"}
        or not isinstance(value["producer"]["model_sha256"], str)
        or _SHA256_PATTERN.fullmatch(value["producer"]["model_sha256"]) is None
        or not isinstance(value["state"], dict)
    ):
        msg = "Scaling artifact does not match the current train_scaling_artifact schema."
        raise ValueError(msg)
    if value["identity"] != dict(identity):
        msg = "Scaling artifact identity does not match the Dataset manifest, Train membership and preprocessing of this run."
        raise ValueError(msg)
    return value["state"]


def read_scaling_artifact(path: Path) -> dict[str, Any]:
    """
    Read one scaling artifact file (run ``normalizer.pt``, stage handoff copy or cache entry) under the safe unpickling policy.

    Raises
    ------
    ValueError
        If the file is unreadable or does not hold a mapping; the envelope itself is admitted by :func:`admit_scaling_artifact`.

    """
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except (RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as error:
        msg = f"Scaling artifact is unreadable: {path}."
        raise ValueError(msg) from error
    if not isinstance(value, Mapping):
        msg = f"Scaling artifact must contain a mapping: {path}."
        raise ValueError(msg)  # noqa: TRY004 -- one contract error type for every unreadable scaling artifact
    return dict(value)


def _emit(callback: StartupCallback | None, event: str, **values: Any) -> None:
    if callback is not None:
        callback(event, values)


def _fit(identity: Mapping[str, Any], fit: Callable[[], Mapping[str, Any]], callback: StartupCallback | None, *, model_sha256: str) -> dict[str, Any]:
    """Fit once and report the fit phase."""
    _emit(callback, "scaler_fit_start")
    started = time.perf_counter()
    artifact = build_scaling_artifact(identity, fit(), model_sha256=model_sha256)
    _emit(callback, "scaler_fit_complete", seconds=time.perf_counter() - started)
    return artifact


def fit_or_load_scaling(
    identity: Mapping[str, Any],
    *,
    model_sha256: str,
    fit: Callable[[], Mapping[str, Any]],
    storage_root: Path | str | None,
    restored: Mapping[str, Any] | None = None,
    startup_callback: StartupCallback | None = None,
) -> dict[str, Any]:
    """
    Admit a restored run artifact, or reuse the cached artifact of one scaling identity and model, or fit and cache it atomically.

    Parameters
    ----------
    identity : Mapping[str, Any]
        Scaling identity (:func:`scaling_identity`); it stays data-only and is the only identity a stored artifact is admitted under.
    model_sha256 : str
        The run's recorded ``producer.model_sha256``; a new artifact records it, and together with ``identity`` it forms the
        cache key, so changed fit or application code misses the cache and refits once. A restored artifact does not depend on it.
    fit : Callable[[], Mapping[str, Any]]
        Fits the task's state from Train membership only; called on a miss.
    storage_root : Path | str | None
        Storage root of the cache.
    restored : Mapping[str, Any] | None, optional
        Artifact of a resumed run or stage handoff; admitted under ``identity`` instead of the cache.
    startup_callback : Callable or None, optional
        Receives ``scaler_cache`` events (``hit``, ``miss`` or ``unavailable``) and the fit phase.

    Returns
    -------
    dict[str, Any]
        The admitted artifact envelope.

    Raises
    ------
    ValueError
        If the restored artifact or an existing cache entry is unreadable or carries another identity, or a cache entry's
        envelope does not reproduce the key it is stored under.

    """
    if restored is not None:
        admit_scaling_artifact(restored, identity=identity)
        _emit(startup_callback, "scaler_cache", state="hit", source="restored_run")
        return dict(restored)
    key = _cache_key(identity, {"model_sha256": model_sha256})
    cache_root = common.paths.get_scaling_cache_root(storage_root=storage_root)
    path = cache_root / f"{key}.pt"
    stack = ExitStack()
    try:
        stack.enter_context(common.locking.exclusive_file_lock(cache_root / "locks" / f"{key}.lock", blocking=True))
    except OSError as error:
        stack.close()
        _emit(startup_callback, "scaler_cache", state="unavailable", reason=type(error).__name__)
        return _fit(identity, fit, startup_callback, model_sha256=model_sha256)
    with stack:
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_file():
                msg = f"Scaling cache entry is unsafe: {path}."
                raise ValueError(msg)
            cached = read_scaling_artifact(path)
            admit_scaling_artifact(cached, identity=identity)
            if _cache_key(cached["identity"], cached["producer"]) != key:
                msg = f"Scaling cache entry was produced under another key than the one it is stored under: {path}."
                raise ValueError(msg)
            _emit(startup_callback, "scaler_cache", state="hit", key=key)
            return cached
        _emit(startup_callback, "scaler_cache", state="miss", key=key)
        artifact = _fit(identity, fit, startup_callback, model_sha256=model_sha256)
        try:
            common.serialization.atomic_torch_save(artifact, path)
        except OSError as error:
            _emit(startup_callback, "scaler_cache", state="unavailable", reason=type(error).__name__)
        return artifact
