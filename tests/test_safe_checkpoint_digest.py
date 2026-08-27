"""Semantic safe-checkpoint digests identify decoded training state, not ZIP bytes."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.safe_checkpoint import save_safe_checkpoint
from engine.safe_checkpoint_digest import safe_checkpoint_digest


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


def test_invalid_safe_checkpoint_keeps_reader_error_contract(tmp_path):
    path = tmp_path / "broken.safe.npz"
    path.write_bytes(b"not an npz")

    with pytest.raises(ValueError, match="invalid safe checkpoint container"):
        safe_checkpoint_digest(path)
