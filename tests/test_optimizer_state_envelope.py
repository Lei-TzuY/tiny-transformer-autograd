"""Regression tests for optimizer state-dict envelope validation."""

import os
import sys
from collections import UserDict

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.optim import Adam, AdamW, SGD
from engine.tensor import Tensor


_CASES = [
    (
        SGD,
        {"lr": 0.05, "momentum": 0.6, "weight_decay": 0.1},
        {"lr", "momentum", "weight_decay", "v"},
    ),
    (
        Adam,
        {
            "lr": 0.02,
            "betas": (0.7, 0.8),
            "eps": 1e-7,
            "weight_decay": 0.1,
        },
        {"lr", "betas", "eps", "weight_decay", "t", "m", "v"},
    ),
    (
        AdamW,
        {
            "lr": 0.02,
            "betas": (0.7, 0.8),
            "eps": 1e-7,
            "weight_decay": 0.1,
        },
        {"lr", "betas", "eps", "weight_decay", "t", "m", "v"},
    ),
]


def _make_optimizer(optimizer_cls, kwargs):
    parameter = Tensor([1.5, -2.0], requires_grad=True)
    optimizer = optimizer_cls([parameter], **kwargs)
    parameter.grad[:] = [0.25, -0.5]
    optimizer.step()
    return optimizer


def _assert_state_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        actual_value = actual[key]
        expected_value = expected[key]
        if isinstance(actual_value, list) and actual_value and isinstance(
            actual_value[0], np.ndarray
        ):
            assert len(actual_value) == len(expected_value)
            for actual_array, expected_array in zip(actual_value, expected_value):
                np.testing.assert_array_equal(actual_array, expected_array)
        else:
            assert actual_value == expected_value


@pytest.mark.parametrize("optimizer_cls,kwargs,_required", _CASES)
@pytest.mark.parametrize("bad_state", [None, [], (), "state", 3])
def test_non_mapping_state_is_rejected_transactionally(
    optimizer_cls,
    kwargs,
    _required,
    bad_state,
):
    optimizer = _make_optimizer(optimizer_cls, kwargs)
    before = optimizer.state_dict()

    with pytest.raises(TypeError, match="state must be a mapping"):
        optimizer.load_state_dict(bad_state)

    _assert_state_equal(optimizer.state_dict(), before)


@pytest.mark.parametrize("optimizer_cls,kwargs,required", _CASES)
def test_each_missing_required_key_is_rejected_transactionally(
    optimizer_cls,
    kwargs,
    required,
):
    optimizer = _make_optimizer(optimizer_cls, kwargs)
    before = optimizer.state_dict()

    for missing_key in sorted(required):
        malformed = optimizer.state_dict()
        del malformed[missing_key]

        with pytest.raises(ValueError, match="state missing keys") as exc_info:
            optimizer.load_state_dict(malformed)

        assert missing_key in str(exc_info.value)
        _assert_state_equal(optimizer.state_dict(), before)


@pytest.mark.parametrize("optimizer_cls,kwargs,required", _CASES)
def test_multiple_missing_keys_have_deterministic_sorted_error(
    optimizer_cls,
    kwargs,
    required,
):
    optimizer = _make_optimizer(optimizer_cls, kwargs)
    malformed = optimizer.state_dict()
    missing = sorted(required)[:2]
    for key in missing:
        del malformed[key]

    with pytest.raises(ValueError) as exc_info:
        optimizer.load_state_dict(malformed)

    assert str(missing) in str(exc_info.value)


@pytest.mark.parametrize("optimizer_cls,kwargs,_required", _CASES)
def test_general_mapping_state_and_extra_keys_are_accepted(
    optimizer_cls,
    kwargs,
    _required,
):
    source = _make_optimizer(optimizer_cls, kwargs)
    state = source.state_dict()
    state["future_metadata"] = {"schema": 2}

    target_parameter = Tensor([9.0, 8.0], requires_grad=True)
    target = optimizer_cls([target_parameter])
    target.load_state_dict(UserDict(state))

    restored = target.state_dict()
    expected = source.state_dict()
    _assert_state_equal(restored, expected)


@pytest.mark.parametrize("optimizer_cls", [Adam, AdamW])
def test_adam_legacy_missing_parameter_steps_remains_supported(optimizer_cls):
    source = _make_optimizer(
        optimizer_cls,
        {
            "lr": 0.03,
            "betas": (0.6, 0.9),
            "eps": 1e-8,
            "weight_decay": 0.0,
        },
    )
    state = source.state_dict()
    total_step = state["t"]
    del state["steps"]

    target = optimizer_cls([Tensor([4.0, 5.0], requires_grad=True)])
    target.load_state_dict(state)

    assert target.t == total_step
    assert target._steps == [total_step]


def test_missing_key_validation_precedes_field_value_validation():
    optimizer = _make_optimizer(SGD, {"lr": 0.1, "momentum": 0.4})
    before = optimizer.state_dict()
    malformed = optimizer.state_dict()
    malformed["lr"] = float("nan")
    del malformed["v"]

    with pytest.raises(ValueError, match="state missing keys.*v"):
        optimizer.load_state_dict(malformed)

    _assert_state_equal(optimizer.state_dict(), before)
