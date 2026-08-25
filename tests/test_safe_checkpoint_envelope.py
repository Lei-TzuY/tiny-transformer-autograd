"""Envelope parity regressions for the non-pickle checkpoint format."""

import os
import sys
from types import MappingProxyType

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.safe_checkpoint import (
    _write_safe_state,
    read_safe_checkpoint,
    save_safe_checkpoint,
)
from nn.layers import Linear


class _StateDictMustNotRun:
    def __init__(self):
        self.calls = 0

    def state_dict(self):
        self.calls += 1
        raise AssertionError("model serialization must not run")


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"step": True}, TypeError, "step"),
        ({"step": -1}, ValueError, "step"),
        ({"metadata": []}, TypeError, "metadata"),
        ({"metadata": ""}, TypeError, "metadata"),
    ],
)
def test_safe_save_preflights_envelope_before_model_serialization(
    tmp_path, kwargs, error, message
):
    model = _StateDictMustNotRun()
    destination = tmp_path / "never-created.safe.npz"

    with pytest.raises(error, match=message):
        save_safe_checkpoint(destination, model, **kwargs)

    assert model.calls == 0
    assert not destination.exists()


def test_safe_save_accepts_and_snapshots_general_mapping_metadata(tmp_path):
    np.random.seed(3)
    model = Linear(2, 2)
    metadata = MappingProxyType({"kind": "test", "count": 2})
    destination = tmp_path / "mapping.safe.npz"

    save_safe_checkpoint(destination, model, step=np.int64(4), metadata=metadata)
    state = read_safe_checkpoint(destination)

    assert state["step"] == 4
    assert type(state["step"]) is int
    assert state["metadata"] == {"kind": "test", "count": 2}
    assert type(state["metadata"]) is dict


def _valid_outer_state():
    return {
        "format_version": 2,
        "model": {},
        "optimizer": None,
        "optimizer_type": None,
        "scheduler": None,
        "rng_state": None,
        "step": 0,
        "metadata": {},
    }


@pytest.mark.parametrize(
    ("mutate", "error", "message"),
    [
        (
            lambda state: state.__setitem__("format_version", True),
            TypeError,
            "format_version",
        ),
        (
            lambda state: state.__setitem__("step", -1),
            ValueError,
            "step",
        ),
        (
            lambda state: state.__setitem__("metadata", []),
            TypeError,
            "metadata",
        ),
        (
            lambda state: state.update(optimizer={"m": []}, optimizer_type=None),
            ValueError,
            "requires optimizer_type",
        ),
        (
            lambda state: state.__setitem__(
                "rng_state",
                ("MT19937", np.array([1], dtype=np.uint32), 0, 0, 0.0),
            ),
            ValueError,
            "RNG state",
        ),
    ],
)
def test_safe_reader_rejects_valid_container_with_invalid_checkpoint_envelope(
    tmp_path, mutate, error, message
):
    state = _valid_outer_state()
    mutate(state)
    destination = tmp_path / "malformed-envelope.safe.npz"
    _write_safe_state(destination, state)

    with pytest.raises(error, match=message):
        read_safe_checkpoint(destination)
