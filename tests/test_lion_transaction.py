import numpy as np
import pytest

from engine.lion import Lion
from engine.tensor import Tensor


def test_state_roundtrip_reproduces_next_lion_update_exactly():
    first_parameter = Tensor([1.5, -2.5], requires_grad=True)
    first = Lion(
        [first_parameter],
        lr=0.02,
        betas=(0.7, 0.95),
        weight_decay=0.1,
    )
    first_parameter.grad[...] = [0.4, -0.8]
    first.step()

    resumed_parameter = Tensor(first_parameter.data.copy(), requires_grad=True)
    resumed = Lion([resumed_parameter], lr=0.5)
    resumed.load_state_dict(first.state_dict())

    next_gradient = np.array([-0.3, 0.6])
    first_parameter.grad[...] = next_gradient
    resumed_parameter.grad[...] = next_gradient
    first.step()
    resumed.step()

    np.testing.assert_array_equal(resumed_parameter.data, first_parameter.data)
    first_state = first.state_dict()
    resumed_state = resumed.state_dict()
    assert resumed_state["step_count"] == first_state["step_count"]
    assert resumed_state["steps"] == first_state["steps"]
    assert resumed_state["lr"] == first_state["lr"]
    assert resumed_state["betas"] == first_state["betas"]
    assert resumed_state["weight_decay"] == first_state["weight_decay"]
    np.testing.assert_array_equal(
        resumed_state["momentum"][0], first_state["momentum"][0]
    )


def test_unexpected_late_parameter_write_failure_rolls_back_prior_commits(monkeypatch):
    first = Tensor([1.0], requires_grad=True)
    second = Tensor([2.0], requires_grad=True)
    first.grad[...] = [1.0]
    second.grad[...] = [-1.0]
    optimizer = Lion([first, second], lr=0.1)
    before = optimizer.state_dict()
    first_version = first._version
    second_version = second._version

    array_type = type(second.data)
    original_setitem = array_type.__setitem__

    def fail_second_parameter_write(array, key, value):
        owner_ref = getattr(array, "_owner_ref", None)
        owner = None if owner_ref is None else owner_ref()
        if owner is second:
            raise RuntimeError("injected Lion write failure")
        return original_setitem(array, key, value)

    monkeypatch.setattr(array_type, "__setitem__", fail_second_parameter_write)

    with pytest.raises(RuntimeError, match="injected Lion write failure"):
        optimizer.step()

    np.testing.assert_array_equal(first.data, [1.0])
    np.testing.assert_array_equal(second.data, [2.0])
    after = optimizer.state_dict()
    for actual, expected in zip(after["momentum"], before["momentum"]):
        np.testing.assert_array_equal(actual, expected)
    assert after["step_count"] == 0
    assert after["steps"] == [0, 0]

    # The first parameter really was written and then restored, so its mutation
    # history must remain visible to autograd instead of rewinding its version.
    assert first._version >= first_version + 2
    # The failing second parameter write never committed, so fallback restore
    # should only occur if needed; its original numerical value remains intact.
    assert second._version >= second_version


def test_constructor_rejects_invalid_hyperparameters_before_training():
    parameter = Tensor([1.0], requires_grad=True)

    for bad_lr in (0.0, -1.0, np.nan, np.inf, True, "0.1"):
        with pytest.raises((TypeError, ValueError)):
            Lion([parameter], lr=bad_lr)

    for bad_betas in (
        (1.0, 0.9),
        (-0.1, 0.9),
        (0.9, 1.0),
        (0.9,),
        (0.9, 0.99, 0.999),
        (True, 0.9),
    ):
        with pytest.raises((TypeError, ValueError)):
            Lion([parameter], betas=bad_betas)

    for bad_decay in (-1.0, np.nan, np.inf, True, "0.1"):
        with pytest.raises((TypeError, ValueError)):
            Lion([parameter], weight_decay=bad_decay)
