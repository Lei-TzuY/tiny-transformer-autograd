"""Regression tests for one-way trusted-pickle checkpoint migration."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.checkpoint import read_checkpoint, save_checkpoint
from engine.checkpoint_convert import convert_checkpoint_to_safe
from engine.safe_checkpoint import read_safe_checkpoint


class _StateOwner:
    def __init__(self, state):
        self._state = state

    def state_dict(self):
        return self._state


class _DemoOptimizer(_StateOwner):
    pass


def _assert_state_equal(left, right):
    assert type(left) is type(right)
    if isinstance(left, np.ndarray):
        assert left.dtype == right.dtype
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert list(left) == list(right)
        for key in left:
            _assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_state_equal(left_item, right_item)
    else:
        assert left == right


def _write_source(path, *, step=7):
    model = _StateOwner(
        {
            "weight": np.arange(6.0, dtype=np.float64).reshape(2, 3),
            "bias": np.array([-1.5, 2.25], dtype=np.float64),
        }
    )
    optimizer = _DemoOptimizer(
        {
            "lr": 1e-3,
            "t": 4,
            "m": [np.array([0.25, -0.5], dtype=np.float64)],
        }
    )
    scheduler = _StateOwner({"last_step": 3, "min_lr": 1e-5})
    metadata = {
        "name": "migration-probe",
        "shape": (2, 3),
        "flags": [True, None, 5],
    }
    save_checkpoint(
        path,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=step,
        metadata=metadata,
    )


def test_conversion_preserves_complete_checkpoint_state(tmp_path):
    source = tmp_path / "legacy.pkl"
    destination = tmp_path / "converted.safe.npz"
    _write_source(source)
    expected = read_checkpoint(source)

    completed_step = convert_checkpoint_to_safe(source, destination)
    actual = read_safe_checkpoint(destination)

    assert completed_step == 7
    _assert_state_equal(actual, expected)


def test_conversion_does_not_change_process_rng_state(tmp_path):
    source = tmp_path / "legacy.pkl"
    destination = tmp_path / "converted.safe.npz"
    _write_source(source)
    np.random.seed(991)
    before = np.random.get_state()

    convert_checkpoint_to_safe(source, destination)
    after = np.random.get_state()

    _assert_state_equal(after, before)


def test_conversion_can_replace_source_path_after_reading_it(tmp_path):
    path = tmp_path / "checkpoint.bin"
    _write_source(path, step=11)

    assert convert_checkpoint_to_safe(path, path) == 11

    state = read_safe_checkpoint(path)
    assert state["step"] == 11
    assert state["metadata"]["name"] == "migration-probe"


def test_invalid_source_cannot_replace_existing_destination(tmp_path):
    source = tmp_path / "broken.pkl"
    destination = tmp_path / "keep-me.safe.npz"
    source.write_bytes(b"not a pickle checkpoint")
    original = b"existing destination bytes"
    destination.write_bytes(original)

    with pytest.raises(Exception):
        convert_checkpoint_to_safe(source, destination)

    assert destination.read_bytes() == original


def test_path_protocol_objects_are_supported(tmp_path):
    source = tmp_path / "legacy.pkl"
    destination = tmp_path / "converted.safe.npz"
    _write_source(source, step=13)

    assert convert_checkpoint_to_safe(source, destination) == 13
    assert read_safe_checkpoint(destination)["step"] == 13
