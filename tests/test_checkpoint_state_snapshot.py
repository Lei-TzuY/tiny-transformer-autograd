"""Checkpoint readers must freeze dynamic top-level mappings once."""

from collections.abc import Mapping
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.checkpoint import read_checkpoint, restore_checkpoint


class _ChangingMapping(Mapping):
    """Mapping whose selected values change on repeated reads."""

    def __init__(self, values, changes=None):
        self._values = dict(values)
        self._changes = {
            key: list(sequence) for key, sequence in (changes or {}).items()
        }
        self.reads = {}

    def __getitem__(self, key):
        if key not in self._values:
            raise KeyError(key)
        count = self.reads.get(key, 0)
        self.reads[key] = count + 1
        sequence = self._changes.get(key)
        if sequence:
            return sequence[min(count, len(sequence) - 1)]
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)


class _StateModel:
    def __init__(self, value):
        self.value = np.array([value], dtype=np.float64)

    def state_dict(self):
        return {"value": self.value.copy()}

    def load_state_dict(self, state, strict=True):
        self.value[:] = state["value"]


class _StateOptimizer:
    def __init__(self, value):
        self.value = np.array([value], dtype=np.float64)

    def state_dict(self):
        return {"value": self.value.copy()}

    def load_state_dict(self, state):
        self.value[:] = state["value"]


def _base_state(model_state):
    return {
        "format_version": 2,
        "model": model_state,
        "optimizer": None,
        "optimizer_type": None,
        "scheduler": None,
        "rng_state": None,
        "step": 0,
        "metadata": {},
    }


def _assert_rng_equal(actual, expected):
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2:] == expected[2:]


def test_restore_uses_one_optimizer_type_snapshot():
    source = _StateModel(3.0)
    target = _StateModel(-1.0)
    optimizer = _StateOptimizer(-2.0)
    values = _base_state(source.state_dict())
    values.update(
        optimizer={"value": np.array([7.0], dtype=np.float64)},
        optimizer_type="_StateOptimizer",
        step=4,
    )
    state = _ChangingMapping(
        values,
        changes={"optimizer_type": ["_StateOptimizer", "DifferentOptimizer"]},
    )

    assert restore_checkpoint(state, target, optimizer=optimizer) == 4

    np.testing.assert_array_equal(target.value, source.value)
    np.testing.assert_array_equal(optimizer.value, np.array([7.0]))
    assert state.reads["optimizer_type"] == 1


def test_restore_uses_rng_state_that_was_validated():
    source = _StateModel(5.0)
    target = _StateModel(0.0)
    rng_first = np.random.RandomState(11).get_state()
    rng_later = np.random.RandomState(29).get_state()
    values = _base_state(source.state_dict())
    values["rng_state"] = rng_first
    state = _ChangingMapping(
        values,
        changes={"rng_state": [rng_first, rng_later, rng_later]},
    )

    np.random.seed(101)
    restore_checkpoint(state, target)

    _assert_rng_equal(np.random.get_state(), rng_first)
    np.testing.assert_array_equal(target.value, source.value)
    assert state.reads["rng_state"] == 1


def test_read_checkpoint_returns_frozen_plain_dictionary(tmp_path):
    path = tmp_path / "changing.ckpt"
    values = _base_state({"value": np.array([2.0], dtype=np.float64)})
    values["step"] = 3
    state = _ChangingMapping(values, changes={"step": [3, 9, 12]})
    with path.open("wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)

    loaded = read_checkpoint(path)

    assert type(loaded) is dict
    assert loaded["step"] == 3
    assert loaded["step"] == 3
    np.testing.assert_array_equal(loaded["model"]["value"], np.array([2.0]))
