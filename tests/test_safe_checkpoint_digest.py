"""Semantic safe-checkpoint digests identify decoded training state, not ZIP bytes."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.safe_checkpoint import save_safe_checkpoint
from engine.safe_checkpoint_digest import (
    safe_checkpoint_digest,
    safe_checkpoints_equal,
    verify_safe_checkpoint_digest,
)


class _Model:
    def __init__(self, weight):
        self.weight = np.array(weight, dtype=np.float64)

    def state_dict(self):
        return {
            "weight": self.weight.copy(),
            "ids": np.array([1, 2, 3], dtype=np.int64),
        }


def _save(path, *, weight=(1.0, 2.0), step=7, metadata=None):
    np.random.seed(123)
    save_safe_checkpoint(
        path,
        _Model(weight),
        step=step,
        metadata={"name": "run", "nested": [1, (True, None)]}
        if metadata is None
        else metadata,
    )


def test_digest_is_stable_across_equivalent_rewrites_and_mapping_order(tmp_path):
    first = tmp_path / "first.safe.npz"
    second = tmp_path / "second.safe.npz"

    _save(
        first,
        metadata={"name": "run", "nested": [1, (True, None)]},
    )
    _save(
        second,
        metadata={"nested": [1, (True, None)], "name": "run"},
    )

    first_digest = safe_checkpoint_digest(first)
    second_digest = safe_checkpoint_digest(second)

    assert first_digest == second_digest
    assert len(first_digest) == 64
    assert all(character in "0123456789abcdef" for character in first_digest)


def test_digest_changes_when_tensor_content_changes(tmp_path):
    first = tmp_path / "first.safe.npz"
    second = tmp_path / "second.safe.npz"
    _save(first, weight=(1.0, 2.0))
    _save(second, weight=(1.0, 3.0))

    assert safe_checkpoint_digest(first) != safe_checkpoint_digest(second)


def test_digest_changes_when_scalar_state_changes(tmp_path):
    first = tmp_path / "first.safe.npz"
    second = tmp_path / "second.safe.npz"
    _save(first, step=7)
    _save(second, step=8)

    assert safe_checkpoint_digest(first) != safe_checkpoint_digest(second)


def test_digest_distinguishes_list_and_tuple_metadata(tmp_path):
    first = tmp_path / "first.safe.npz"
    second = tmp_path / "second.safe.npz"
    _save(first, metadata={"value": [1, 2]})
    _save(second, metadata={"value": (1, 2)})

    assert safe_checkpoint_digest(first) != safe_checkpoint_digest(second)


def test_digest_preserves_array_dtype_as_part_of_identity(tmp_path):
    class DtypeModel:
        def __init__(self, dtype):
            self.dtype = dtype

        def state_dict(self):
            return {"value": np.array([1, 2], dtype=self.dtype)}

    first = tmp_path / "first.safe.npz"
    second = tmp_path / "second.safe.npz"
    np.random.seed(5)
    save_safe_checkpoint(first, DtypeModel(np.int32))
    np.random.seed(5)
    save_safe_checkpoint(second, DtypeModel(np.int64))

    assert safe_checkpoint_digest(first) != safe_checkpoint_digest(second)


def test_digest_normalizes_recursive_hash_walk_failures(monkeypatch):
    state = []
    cursor = state
    for _ in range(sys.getrecursionlimit() + 100):
        nested = []
        cursor.append(nested)
        cursor = nested

    monkeypatch.setattr(
        "engine.safe_checkpoint_digest.read_safe_checkpoint",
        lambda path: state,
    )

    with pytest.raises(
        ValueError,
        match="safe checkpoint state nesting is too deep to digest",
    ):
        safe_checkpoint_digest("ignored.safe.npz")


def test_verify_accepts_exact_and_uppercase_expected_digest(tmp_path):
    path = tmp_path / "checkpoint.safe.npz"
    _save(path)
    expected = safe_checkpoint_digest(path)

    assert verify_safe_checkpoint_digest(path, expected) is True
    assert verify_safe_checkpoint_digest(path, expected.upper()) is True


def test_verify_returns_false_for_well_formed_mismatch(tmp_path):
    path = tmp_path / "checkpoint.safe.npz"
    _save(path)
    expected = safe_checkpoint_digest(path)
    replacement = "0" if expected[-1] != "0" else "1"

    assert verify_safe_checkpoint_digest(path, expected[:-1] + replacement) is False


@pytest.mark.parametrize("expected", [None, b"0" * 64, 123])
def test_verify_rejects_non_string_expected_digest_before_file_io(tmp_path, expected):
    missing = tmp_path / "missing.safe.npz"

    with pytest.raises(TypeError, match="expected safe checkpoint digest must be a string"):
        verify_safe_checkpoint_digest(missing, expected)


@pytest.mark.parametrize(
    "expected",
    ["", "0" * 63, "0" * 65, "g" * 64, ("0" * 63) + "!"],
)
def test_verify_rejects_malformed_expected_digest_before_file_io(tmp_path, expected):
    missing = tmp_path / "missing.safe.npz"

    with pytest.raises(
        ValueError,
        match="expected safe checkpoint digest must be 64 hexadecimal characters",
    ):
        verify_safe_checkpoint_digest(missing, expected)


def test_verify_preserves_reader_error_for_valid_expected_digest(tmp_path):
    path = tmp_path / "broken.safe.npz"
    path.write_bytes(b"not an npz")

    with pytest.raises(ValueError, match="invalid safe checkpoint container"):
        verify_safe_checkpoint_digest(path, "0" * 64)


def test_safe_checkpoints_equal_uses_semantic_state_not_archive_bytes(tmp_path):
    first = tmp_path / "first.safe.npz"
    second = tmp_path / "second.safe.npz"
    _save(first, metadata={"name": "run", "nested": [1, (True, None)]})
    _save(second, metadata={"nested": [1, (True, None)], "name": "run"})

    assert first.read_bytes() != second.read_bytes()
    assert safe_checkpoints_equal(first, second) is True


def test_safe_checkpoints_equal_detects_semantic_difference(tmp_path):
    first = tmp_path / "first.safe.npz"
    second = tmp_path / "second.safe.npz"
    _save(first, step=7)
    _save(second, step=8)

    assert safe_checkpoints_equal(first, second) is False


def test_safe_checkpoints_equal_validates_first_before_opening_second(tmp_path):
    first = tmp_path / "first.safe.npz"
    first.write_bytes(b"not an npz")
    second = tmp_path / "missing.safe.npz"

    with pytest.raises(ValueError, match="invalid safe checkpoint container"):
        safe_checkpoints_equal(first, second)


def test_safe_checkpoints_equal_preserves_second_reader_error(tmp_path):
    first = tmp_path / "first.safe.npz"
    second = tmp_path / "second.safe.npz"
    _save(first)
    second.write_bytes(b"not an npz")

    with pytest.raises(ValueError, match="invalid safe checkpoint container"):
        safe_checkpoints_equal(first, second)


def test_invalid_safe_checkpoint_keeps_reader_error_contract(tmp_path):
    path = tmp_path / "broken.safe.npz"
    path.write_bytes(b"not an npz")

    with pytest.raises(ValueError, match="invalid safe checkpoint container"):
        safe_checkpoint_digest(path)
