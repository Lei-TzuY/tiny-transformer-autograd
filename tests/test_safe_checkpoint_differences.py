"""Semantic checkpoint differences report deterministic decoded-state paths."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.safe_checkpoint import save_safe_checkpoint
from engine.safe_checkpoint_digest import safe_checkpoint_differences


class _Model:
    def __init__(self, weight=(1.0, 2.0), ids_dtype=np.int64):
        self.weight = np.array(weight, dtype=np.float64)
        self.ids_dtype = ids_dtype

    def state_dict(self):
        return {
            "weight": self.weight.copy(),
            "ids": np.array([1, 2, 3], dtype=self.ids_dtype),
        }


def _save(path, *, model=None, step=7, metadata=None):
    np.random.seed(123)
    save_safe_checkpoint(
        path,
        _Model() if model is None else model,
        step=step,
        metadata={"name": "run", "nested": [1, (True, None)]}
        if metadata is None
        else metadata,
    )


def test_differences_ignore_archive_and_mapping_order(tmp_path):
    first = tmp_path / "first.safe.npz"
    second = tmp_path / "second.safe.npz"
    _save(first, metadata={"name": "run", "nested": [1, (True, None)]})
    _save(second, metadata={"nested": [1, (True, None)], "name": "run"})

    assert first.read_bytes() != second.read_bytes()
    assert safe_checkpoint_differences(first, second) == ()


def test_differences_report_deterministic_nested_paths(tmp_path):
    first = tmp_path / "first.safe.npz"
    second = tmp_path / "second.safe.npz"
    _save(
        first,
        model=_Model(weight=(1.0, 2.0)),
        step=7,
        metadata={"name": "run", "nested": [1, (True, None)], "old": 5},
    )
    _save(
        second,
        model=_Model(weight=(1.0, 3.0)),
        step=8,
        metadata={"name": "changed", "nested": [1, (False, None), 9], "new": 6},
    )

    assert safe_checkpoint_differences(first, second) == (
        "$['metadata']['name']",
        "$['metadata']['nested'][1][0]",
        "$['metadata']['nested'][2]",
        "$['metadata']['new']",
        "$['metadata']['old']",
        "$['model']['weight']",
        "$['step']",
    )


def test_differences_treat_array_dtype_as_semantic_identity(tmp_path):
    first = tmp_path / "first.safe.npz"
    second = tmp_path / "second.safe.npz"
    _save(first, model=_Model(ids_dtype=np.int32))
    _save(second, model=_Model(ids_dtype=np.int64))

    assert safe_checkpoint_differences(first, second) == ("$['model']['ids']",)


def test_differences_report_sequence_type_at_parent_path(tmp_path):
    first = tmp_path / "first.safe.npz"
    second = tmp_path / "second.safe.npz"
    _save(first, metadata={"value": [1, 2]})
    _save(second, metadata={"value": (1, 2)})

    assert safe_checkpoint_differences(first, second) == ("$['metadata']['value']",)


def test_differences_validate_first_before_opening_second(tmp_path):
    first = tmp_path / "broken.safe.npz"
    second = tmp_path / "missing.safe.npz"
    first.write_bytes(b"not an npz")

    with pytest.raises(ValueError, match="invalid safe checkpoint container"):
        safe_checkpoint_differences(first, second)


def test_differences_preserve_second_reader_error(tmp_path):
    first = tmp_path / "first.safe.npz"
    second = tmp_path / "broken.safe.npz"
    _save(first)
    second.write_bytes(b"not an npz")

    with pytest.raises(ValueError, match="invalid safe checkpoint container"):
        safe_checkpoint_differences(first, second)


def test_differences_normalize_recursive_compare_failure(monkeypatch):
    first = []
    second = []
    first_cursor = first
    second_cursor = second
    for _ in range(sys.getrecursionlimit() + 100):
        first_nested = []
        second_nested = []
        first_cursor.append(first_nested)
        second_cursor.append(second_nested)
        first_cursor = first_nested
        second_cursor = second_nested

    states = iter((first, second))
    monkeypatch.setattr(
        "engine.safe_checkpoint_digest.read_safe_checkpoint",
        lambda path: next(states),
    )

    with pytest.raises(
        ValueError,
        match="safe checkpoint state nesting is too deep to compare",
    ):
        safe_checkpoint_differences("first.safe.npz", "second.safe.npz")
