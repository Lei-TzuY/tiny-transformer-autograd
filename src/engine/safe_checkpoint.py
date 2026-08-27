"""Non-executable NumPy/JSON checkpoint persistence.

The historical checkpoint format uses pickle and therefore must only be read
from trusted files. This module provides an opt-in format built from a strict
JSON manifest plus ordinary ``.npy`` arrays inside an ``.npz`` container.
Loading always uses ``allow_pickle=False`` and never imports or executes data
from the checkpoint.

The format protects against pickle-style arbitrary code execution; it is not a
resource-exhaustion sandbox for adversarially huge compressed files.
"""

from collections.abc import Mapping
import json
import os
import tempfile

import numpy as np

from .checkpoint import (
    CHECKPOINT_VERSION,
    _fsync_parent_directory,
    _nonnegative_checkpoint_step,
    _validate_checkpoint_envelope,
)


SAFE_CHECKPOINT_VERSION = 1
_MANIFEST_KEY = "__manifest__"
_MAX_MANIFEST_BYTES = 8 * 1024 * 1024


def _reject_duplicate_json_keys(pairs):
    """Build one JSON object while rejecting ambiguous duplicate keys."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(
                f"safe checkpoint manifest contains duplicate JSON object key {key!r}"
            )
        result[key] = value
    return result


def save_safe_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    step=0,
    metadata=None,
):
    """Save training state atomically without using pickle.

    The decoded state has the same structure and envelope contract as
    ``read_checkpoint`` returns, so callers can pass ``read_safe_checkpoint``
    output directly to the existing transactional ``restore_checkpoint``.
    """
    step = _nonnegative_checkpoint_step(step)
    if metadata is None:
        metadata = {}
    elif not isinstance(metadata, Mapping):
        raise TypeError("checkpoint metadata must be a mapping or None")
    else:
        metadata = dict(metadata)

    state = {
        "format_version": CHECKPOINT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "optimizer_type": (
            optimizer.__class__.__name__ if optimizer is not None else None
        ),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "rng_state": np.random.get_state(),
        "step": step,
        "metadata": metadata,
    }
    _validate_checkpoint_envelope(state)
    _write_safe_state(path, state)


def read_safe_checkpoint(path):
    """Read and envelope-validate a non-executable safe checkpoint."""
    path = os.fspath(path)
    try:
        archive = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid safe checkpoint container: {exc}") from exc

    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError("safe checkpoint container must be an NPZ archive")

    with archive:
        files = list(archive.files)
        if len(files) != len(set(files)):
            raise ValueError("safe checkpoint contains duplicate archive members")
        if _MANIFEST_KEY not in files:
            raise ValueError("safe checkpoint is missing its manifest")

        manifest_array = _load_manifest_array(archive)
        try:
            manifest_text = manifest_array.tobytes().decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("safe checkpoint manifest must be UTF-8") from exc
        try:
            manifest = json.loads(
                manifest_text,
                object_pairs_hook=_reject_duplicate_json_keys,
            )
        except json.JSONDecodeError as exc:
            raise ValueError("safe checkpoint manifest is not valid JSON") from exc
        except RecursionError as exc:
            raise ValueError("safe checkpoint manifest nesting is too deep") from exc

        _validate_manifest(manifest)
        used_arrays = set()
        try:
            state = _decode_node(
                manifest["state"],
                archive,
                used_arrays,
                path="state",
            )
        except RecursionError as exc:
            raise ValueError("safe checkpoint manifest nesting is too deep") from exc

        expected_files = used_arrays | {_MANIFEST_KEY}
        actual_files = set(files)
        if actual_files != expected_files:
            unexpected = sorted(actual_files - expected_files)
            missing = sorted(expected_files - actual_files)
            raise ValueError(
                "safe checkpoint archive members do not match the manifest: "
                f"missing={missing}, unexpected={unexpected}"
            )
        if not isinstance(state, dict):
            raise ValueError("safe checkpoint root state must decode to a dictionary")
        _validate_checkpoint_envelope(state)
        return state


def _write_safe_state(path, state):
    """Encode one checkpoint-state dictionary and atomically replace ``path``."""
    arrays = {}
    encoded = _encode_node(state, arrays, path="state")
    manifest = {
        "safe_checkpoint_version": SAFE_CHECKPOINT_VERSION,
        "state": encoded,
    }
    try:
        manifest_bytes = json.dumps(
            manifest,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TypeError(f"safe checkpoint manifest is not serializable: {exc}") from exc
    if len(manifest_bytes) > _MAX_MANIFEST_BYTES:
        raise ValueError(
            f"safe checkpoint manifest exceeds {_MAX_MANIFEST_BYTES} bytes"
        )

    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    prefix = f".{os.path.basename(path)}."
    descriptor, temporary = tempfile.mkstemp(
        dir=directory,
        prefix=prefix,
        suffix=".tmp",
    )
    try:
        payload = {
            _MANIFEST_KEY: np.frombuffer(manifest_bytes, dtype=np.uint8),
            **arrays,
        }
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        _fsync_parent_directory(directory)
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
        raise


def _encode_node(value, arrays, *, path):
    """Encode supported Python/NumPy values into a strict JSON tree."""
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError(
                f"safe checkpoint does not support object arrays at {path}"
            )
        key = f"array_{len(arrays):08d}"
        arrays[key] = np.array(value, copy=True)
        return {"type": "array", "key": key}

    if isinstance(value, np.generic):
        value = value.item()

    if value is None:
        return {"type": "none"}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": value}
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(
                f"safe checkpoint requires finite scalar floats at {path}"
            )
        return {"type": "float", "value": value}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, list):
        return {
            "type": "list",
            "items": [
                _encode_node(item, arrays, path=f"{path}[{index}]")
                for index, item in enumerate(value)
            ],
        }
    if isinstance(value, tuple):
        return {
            "type": "tuple",
            "items": [
                _encode_node(item, arrays, path=f"{path}[{index}]")
                for index, item in enumerate(value)
            ],
        }
    if isinstance(value, dict):
        items = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"safe checkpoint dictionary keys must be strings at {path}"
                )
            items.append(
                [
                    key,
                    _encode_node(item, arrays, path=f"{path}.{key}"),
                ]
            )
        return {"type": "dict", "items": items}

    raise TypeError(
        f"safe checkpoint does not support {type(value).__name__} at {path}"
    )


def _load_manifest_array(archive):
    try:
        manifest_array = archive[_MANIFEST_KEY]
    except ValueError as exc:
        raise ValueError("safe checkpoint manifest cannot require pickle") from exc
    if manifest_array.dtype != np.uint8 or manifest_array.ndim != 1:
        raise ValueError("safe checkpoint manifest must be a one-dimensional uint8 array")
    if manifest_array.size > _MAX_MANIFEST_BYTES:
        raise ValueError(
            f"safe checkpoint manifest exceeds {_MAX_MANIFEST_BYTES} bytes"
        )
    return manifest_array


def _validate_manifest(manifest):
    if not isinstance(manifest, dict):
        raise ValueError("safe checkpoint manifest root must be a JSON object")
    expected = {"safe_checkpoint_version", "state"}
    if set(manifest) != expected:
        raise ValueError(
            "safe checkpoint manifest keys must be exactly "
            f"{sorted(expected)}"
        )
    version = manifest["safe_checkpoint_version"]
    if type(version) is not int or version != SAFE_CHECKPOINT_VERSION:
        raise ValueError(f"unsupported safe checkpoint format version: {version}")


def _decode_node(node, archive, used_arrays, *, path):
    if not isinstance(node, dict) or "type" not in node:
        raise ValueError(f"malformed safe checkpoint node at {path}")
    kind = node["type"]

    if kind == "array":
        _require_keys(node, {"type", "key"}, path)
        key = node["key"]
        if not isinstance(key, str) or not key.startswith("array_"):
            raise ValueError(f"invalid array reference at {path}")
        if key in used_arrays:
            raise ValueError(f"duplicate array reference {key!r} at {path}")
        if key not in archive.files:
            raise ValueError(f"missing array {key!r} referenced at {path}")
        try:
            array = archive[key]
        except ValueError as exc:
            raise ValueError(
                f"array {key!r} at {path} cannot require pickle"
            ) from exc
        if array.dtype.hasobject:
            raise ValueError(f"object array {key!r} is not allowed at {path}")
        used_arrays.add(key)
        return np.array(array, copy=True)

    if kind == "none":
        _require_keys(node, {"type"}, path)
        return None
    if kind == "bool":
        _require_keys(node, {"type", "value"}, path)
        if type(node["value"]) is not bool:
            raise ValueError(f"invalid boolean at {path}")
        return node["value"]
    if kind == "int":
        _require_keys(node, {"type", "value"}, path)
        if type(node["value"]) is not int:
            raise ValueError(f"invalid integer at {path}")
        return node["value"]
    if kind == "float":
        _require_keys(node, {"type", "value"}, path)
        value = node["value"]
        if type(value) not in {int, float}:
            raise ValueError(f"invalid finite float at {path}")
        try:
            value = float(value)
        except OverflowError as exc:
            raise ValueError(f"invalid finite float at {path}") from exc
        if not np.isfinite(value):
            raise ValueError(f"invalid finite float at {path}")
        return value
    if kind == "str":
        _require_keys(node, {"type", "value"}, path)
        if not isinstance(node["value"], str):
            raise ValueError(f"invalid string at {path}")
        return node["value"]
    if kind in {"list", "tuple"}:
        _require_keys(node, {"type", "items"}, path)
        items = node["items"]
        if not isinstance(items, list):
            raise ValueError(f"invalid {kind} items at {path}")
        decoded = [
            _decode_node(item, archive, used_arrays, path=f"{path}[{index}]")
            for index, item in enumerate(items)
        ]
        return decoded if kind == "list" else tuple(decoded)
    if kind == "dict":
        _require_keys(node, {"type", "items"}, path)
        items = node["items"]
        if not isinstance(items, list):
            raise ValueError(f"invalid dictionary items at {path}")
        result = {}
        for index, pair in enumerate(items):
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or not isinstance(pair[0], str)
            ):
                raise ValueError(
                    f"invalid dictionary entry {index} at {path}"
                )
            key, child = pair
            if key in result:
                raise ValueError(f"duplicate dictionary key {key!r} at {path}")
            result[key] = _decode_node(
                child,
                archive,
                used_arrays,
                path=f"{path}.{key}",
            )
        return result

    raise ValueError(f"unknown safe checkpoint node type {kind!r} at {path}")


def _require_keys(node, expected, path):
    if set(node) != expected:
        raise ValueError(
            f"malformed safe checkpoint node at {path}: expected keys "
            f"{sorted(expected)}, got {sorted(node)}"
        )
