"""Deterministic semantic digests for non-executable safe checkpoints.

The NPZ container itself is not a stable content identifier: ZIP metadata and
compression details may change when the same checkpoint state is rewritten.
This module hashes the validated decoded state instead, with explicit type and
length framing so equivalent safe checkpoints receive the same SHA-256 digest.
It also reports deterministic paths for semantic differences between checkpoints.
"""

import hashlib
import hmac
import struct

import numpy as np

from .safe_checkpoint import read_safe_checkpoint


_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


def safe_checkpoint_digest(path):
    """Return a SHA-256 hex digest of one validated safe checkpoint's state.

    The digest is independent of archive member order, ZIP metadata, compression,
    and dictionary insertion order. Sequence order, scalar types, array dtype,
    shape, and exact array bytes remain significant.
    """
    state = read_safe_checkpoint(path)
    digest = hashlib.sha256()
    try:
        _hash_value(digest, state)
    except RecursionError as exc:
        raise ValueError("safe checkpoint state nesting is too deep to digest") from exc
    return digest.hexdigest()


def verify_safe_checkpoint_digest(path, expected):
    """Return whether a safe checkpoint matches an expected semantic SHA-256 digest.

    ``expected`` must be a 64-character hexadecimal string. Validation happens
    before the checkpoint is opened so a malformed expected digest cannot trigger
    file I/O. Uppercase hexadecimal is accepted and normalized for comparison.
    """
    if not isinstance(expected, str):
        raise TypeError("expected safe checkpoint digest must be a string")
    if len(expected) != 64 or any(character not in _HEX_DIGITS for character in expected):
        raise ValueError("expected safe checkpoint digest must be 64 hexadecimal characters")

    actual = safe_checkpoint_digest(path)
    return hmac.compare_digest(actual, expected.lower())


def safe_checkpoints_equal(first, second):
    """Return whether two validated safe checkpoints contain identical semantic state.

    Both inputs are read through the non-executable safe-checkpoint reader. The first
    checkpoint is validated before the second is opened, giving deterministic error
    precedence when both inputs are invalid.
    """
    first_digest = safe_checkpoint_digest(first)
    second_digest = safe_checkpoint_digest(second)
    return hmac.compare_digest(first_digest, second_digest)


def safe_checkpoint_differences(first, second):
    """Return deterministic paths whose decoded checkpoint values differ.

    Paths use ``$`` for the decoded checkpoint root, ``['key']`` for mappings, and
    ``[index]`` for sequences. Mapping insertion order is ignored. Array identity
    follows the digest contract exactly: dtype, shape, and C-order bytes all matter.
    Missing mapping keys or sequence elements are reported at their missing path.

    The first checkpoint is fully validated before the second is opened, matching
    :func:`safe_checkpoints_equal` error precedence.
    """
    first_state = read_safe_checkpoint(first)
    second_state = read_safe_checkpoint(second)
    differences = []
    try:
        _collect_differences(first_state, second_state, "$", differences)
    except RecursionError as exc:
        raise ValueError("safe checkpoint state nesting is too deep to compare") from exc
    return tuple(differences)


def _write_bytes(digest, tag, payload=b""):
    digest.update(tag)
    digest.update(struct.pack(">Q", len(payload)))
    digest.update(payload)


def _array_identity(value):
    array = np.ascontiguousarray(value)
    return (
        array.dtype.str,
        repr(array.dtype.descr),
        array.shape,
        array.tobytes(order="C"),
    )


def _scalar_equal(first, second):
    if type(first) is not type(second):
        return False
    if isinstance(first, float):
        return struct.pack(">d", first) == struct.pack(">d", second)
    return first == second


def _collect_differences(first, second, path, differences):
    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        if not isinstance(first, np.ndarray) or not isinstance(second, np.ndarray):
            differences.append(path)
            return
        if _array_identity(first) != _array_identity(second):
            differences.append(path)
        return

    scalar_types = (type(None), bool, int, float, str)
    if isinstance(first, scalar_types) or isinstance(second, scalar_types):
        if not isinstance(first, scalar_types) or not isinstance(second, scalar_types):
            differences.append(path)
            return
        if not _scalar_equal(first, second):
            differences.append(path)
        return

    if isinstance(first, dict) or isinstance(second, dict):
        if not isinstance(first, dict) or not isinstance(second, dict):
            differences.append(path)
            return
        for key in sorted(first.keys() | second.keys()):
            child_path = f"{path}[{key!r}]"
            if key not in first or key not in second:
                differences.append(child_path)
            else:
                _collect_differences(first[key], second[key], child_path, differences)
        return

    sequence_types = (list, tuple)
    if isinstance(first, sequence_types) or isinstance(second, sequence_types):
        if type(first) is not type(second):
            differences.append(path)
            return
        common = min(len(first), len(second))
        for index in range(common):
            _collect_differences(first[index], second[index], f"{path}[{index}]", differences)
        for index in range(common, max(len(first), len(second))):
            differences.append(f"{path}[{index}]")
        return

    raise TypeError(
        "unsupported decoded safe-checkpoint value: "
        f"{type(first).__name__}/{type(second).__name__}"
    )


def _hash_value(digest, value):
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        _write_bytes(digest, b"A", array.dtype.str.encode("ascii"))
        _write_bytes(digest, b"D", repr(array.dtype.descr).encode("utf-8"))
        shape = ",".join(str(int(size)) for size in array.shape).encode("ascii")
        _write_bytes(digest, b"H", shape)
        _write_bytes(digest, b"B", array.tobytes(order="C"))
        return

    if value is None:
        _write_bytes(digest, b"N")
        return
    if isinstance(value, bool):
        _write_bytes(digest, b"T" if value else b"F")
        return
    if isinstance(value, int):
        _write_bytes(digest, b"I", str(value).encode("ascii"))
        return
    if isinstance(value, float):
        _write_bytes(digest, b"R", struct.pack(">d", value))
        return
    if isinstance(value, str):
        _write_bytes(digest, b"S", value.encode("utf-8"))
        return
    if isinstance(value, list):
        _write_bytes(digest, b"L", str(len(value)).encode("ascii"))
        for item in value:
            _hash_value(digest, item)
        return
    if isinstance(value, tuple):
        _write_bytes(digest, b"U", str(len(value)).encode("ascii"))
        for item in value:
            _hash_value(digest, item)
        return
    if isinstance(value, dict):
        _write_bytes(digest, b"M", str(len(value)).encode("ascii"))
        for key in sorted(value):
            _write_bytes(digest, b"K", key.encode("utf-8"))
            _hash_value(digest, value[key])
        return

    # read_safe_checkpoint() validates this contract before returning. Keep a
    # loud internal guard here so future codec expansion cannot silently create
    # ambiguous digests without adding an explicit type encoding above.
    raise TypeError(f"unsupported decoded safe-checkpoint value: {type(value).__name__}")
