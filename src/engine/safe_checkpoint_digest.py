"""Deterministic semantic digests for non-executable safe checkpoints.

The NPZ container itself is not a stable content identifier: ZIP metadata and
compression details may change when the same checkpoint state is rewritten.
This module hashes the validated decoded state instead, with explicit type and
length framing so equivalent safe checkpoints receive the same SHA-256 digest.
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
    _hash_value(digest, state)
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


def _write_bytes(digest, tag, payload=b""):
    digest.update(tag)
    digest.update(struct.pack(">Q", len(payload)))
    digest.update(payload)


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
