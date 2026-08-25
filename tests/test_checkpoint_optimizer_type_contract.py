"""Optimizer-type envelope invariants for versioned checkpoints."""

import os
import pickle
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.checkpoint import read_checkpoint, restore_checkpoint
from engine.optim import Adam, AdamW
from engine.tensor import Tensor


class _StateModel:
    def __init__(self, value=0.0):
        self.value = np.array([value], dtype=np.float64)

    def state_dict(self):
        return {"value": self.value.copy()}

    def load_state_dict(self, state, strict=True):
        self.value[:] = state["value"]


class _SnapshotBomb:
    def state_dict(self):
        raise AssertionError("malformed envelope must fail before caller snapshots")

    def load_state_dict(self, state, strict=True):
        raise AssertionError("malformed envelope must not mutate caller state")


def _adam_with_state(*, lr=0.02, weight_decay=0.3):
    parameter = Tensor([1.5], requires_grad=True)
    optimizer = Adam([parameter], lr=lr, weight_decay=weight_decay)
    parameter.grad = np.array([0.25], dtype=np.float64)
    optimizer.step()
    return parameter, optimizer


def test_v2_optimizer_payload_requires_optimizer_type_before_snapshots():
    _, source = _adam_with_state()
    target_parameter = Tensor([9.0], requires_grad=True)
    target = AdamW([target_parameter], lr=0.5, weight_decay=0.8)
    before = target.state_dict()

    state = {
        "format_version": 2,
        "model": {},
        "optimizer": source.state_dict(),
        "step": 1,
    }

    with pytest.raises(ValueError, match="requires optimizer_type"):
        restore_checkpoint(state, _SnapshotBomb(), optimizer=target)

    after = target.state_dict()
    assert after["lr"] == before["lr"]
    assert after["betas"] == before["betas"]
    assert after["eps"] == before["eps"]
    assert after["weight_decay"] == before["weight_decay"]
    assert after["t"] == before["t"]
    assert after["steps"] == before["steps"]
    for actual, expected in zip(after["m"], before["m"]):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(after["v"], before["v"]):
        np.testing.assert_array_equal(actual, expected)


def test_read_rejects_v2_optimizer_payload_without_type(tmp_path):
    _, source = _adam_with_state()
    path = tmp_path / "missing-optimizer-type.ckpt"
    state = {
        "format_version": 2,
        "model": {"value": np.array([1.0])},
        "optimizer": source.state_dict(),
        "optimizer_type": None,
    }
    with open(path, "wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with pytest.raises(ValueError, match="requires optimizer_type"):
        read_checkpoint(path)


def test_v1_optimizer_payload_without_type_remains_compatible():
    source_model = _StateModel(4.0)
    _, source_optimizer = _adam_with_state(lr=0.03, weight_decay=0.2)
    target_model = _StateModel(-7.0)
    target_parameter = Tensor([8.0], requires_grad=True)
    target_optimizer = Adam([target_parameter], lr=0.9, weight_decay=0.0)

    state = {
        "format_version": 1,
        "model": source_model.state_dict(),
        "optimizer": source_optimizer.state_dict(),
        "step": 6,
    }

    assert restore_checkpoint(state, target_model, optimizer=target_optimizer) == 6
    np.testing.assert_array_equal(target_model.value, source_model.value)

    expected = source_optimizer.state_dict()
    actual = target_optimizer.state_dict()
    assert actual["lr"] == expected["lr"]
    assert actual["betas"] == expected["betas"]
    assert actual["eps"] == expected["eps"]
    assert actual["weight_decay"] == expected["weight_decay"]
    assert actual["t"] == expected["t"]
    assert actual["steps"] == expected["steps"]
    for actual_buffer, expected_buffer in zip(actual["m"], expected["m"]):
        np.testing.assert_array_equal(actual_buffer, expected_buffer)
    for actual_buffer, expected_buffer in zip(actual["v"], expected["v"]):
        np.testing.assert_array_equal(actual_buffer, expected_buffer)


def test_v2_model_only_checkpoint_does_not_require_optimizer_type():
    source = _StateModel(2.0)
    target = _StateModel(-1.0)
    state = {
        "format_version": 2,
        "model": source.state_dict(),
        "step": 3,
    }

    assert restore_checkpoint(state, target) == 3
    np.testing.assert_array_equal(target.value, source.value)
