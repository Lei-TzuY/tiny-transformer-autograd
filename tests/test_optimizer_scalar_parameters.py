import numpy as np
import pytest

from engine.optim import Adam, AdamW, SGD
from engine.tensor import Tensor


def test_sgd_momentum_updates_scalar_parameter_across_steps():
    parameter = Tensor(5.0, requires_grad=True)
    optimizer = SGD([parameter], lr=0.1, momentum=0.9)

    parameter.grad = np.array(2.0)
    optimizer.step()

    assert parameter.shape == ()
    assert optimizer._v[0].shape == ()
    assert optimizer._v[0] == pytest.approx(2.0)
    assert parameter.data == pytest.approx(4.8)

    parameter.grad = np.array(1.0)
    optimizer.step()

    assert optimizer._v[0] == pytest.approx(2.8)
    assert parameter.data == pytest.approx(4.52)


@pytest.mark.parametrize(
    "optimizer_factory",
    [
        lambda parameter: SGD([parameter], lr=0.1, momentum=0.9),
        lambda parameter: Adam([parameter], lr=0.1),
        lambda parameter: AdamW([parameter], lr=0.1, weight_decay=0.1),
    ],
)
def test_optimizer_zero_grad_clears_scalar_buffer_in_place(optimizer_factory):
    parameter = Tensor(3.0, requires_grad=True)
    optimizer = optimizer_factory(parameter)
    loss = parameter * parameter
    loss.backward()
    original_grad = parameter.grad

    assert original_grad.shape == ()
    assert original_grad == pytest.approx(6.0)

    optimizer.zero_grad()

    assert parameter.grad is original_grad
    assert parameter.grad.shape == ()
    assert parameter.grad == pytest.approx(0.0)


def test_adam_updates_scalar_moments_and_parameter():
    parameter = Tensor(5.0, requires_grad=True)
    optimizer = Adam([parameter], lr=0.1)
    parameter.grad = np.array(2.0)

    optimizer.step()

    assert optimizer._m[0].shape == ()
    assert optimizer._v[0].shape == ()
    assert optimizer._m[0] == pytest.approx(0.2)
    assert optimizer._v[0] == pytest.approx(0.004)
    assert optimizer.t == 1
    assert optimizer._steps == [1]
    assert parameter.data == pytest.approx(4.9000000005)


def test_adamw_updates_scalar_moments_weight_decay_and_parameter():
    parameter = Tensor(5.0, requires_grad=True)
    optimizer = AdamW([parameter], lr=0.1, weight_decay=0.1)
    parameter.grad = np.array(2.0)

    optimizer.step()

    assert optimizer._m[0].shape == ()
    assert optimizer._v[0].shape == ()
    assert optimizer._m[0] == pytest.approx(0.2)
    assert optimizer._v[0] == pytest.approx(0.004)
    assert optimizer.t == 1
    assert optimizer._steps == [1]
    assert parameter.data == pytest.approx(4.8500000005)


def test_sgd_scalar_velocity_state_round_trip_and_next_step_match():
    first_parameter = Tensor(5.0, requires_grad=True)
    first = SGD([first_parameter], lr=0.1, momentum=0.9)
    first_parameter.grad = np.array(2.0)
    first.step()
    state = first.state_dict()

    second_parameter = Tensor(first_parameter.data.copy(), requires_grad=True)
    second = SGD([second_parameter], lr=0.5, momentum=0.0)
    second.load_state_dict(state)

    assert second._v[0].shape == ()
    assert second._v[0] == pytest.approx(first._v[0])

    first_parameter.grad = np.array(1.25)
    second_parameter.grad = np.array(1.25)
    first.step()
    second.step()

    assert second_parameter.data == pytest.approx(first_parameter.data)
    assert second._v[0] == pytest.approx(first._v[0])


@pytest.mark.parametrize("optimizer_type", [Adam, AdamW])
def test_adam_family_scalar_state_round_trip_and_next_step_match(optimizer_type):
    kwargs = {"lr": 0.05}
    if optimizer_type is AdamW:
        kwargs["weight_decay"] = 0.02

    first_parameter = Tensor(5.0, requires_grad=True)
    first = optimizer_type([first_parameter], **kwargs)
    first_parameter.grad = np.array(2.0)
    first.step()
    state = first.state_dict()

    second_parameter = Tensor(first_parameter.data.copy(), requires_grad=True)
    second = optimizer_type([second_parameter], lr=0.5)
    second.load_state_dict(state)

    assert second._m[0].shape == ()
    assert second._v[0].shape == ()
    assert second._m[0] == pytest.approx(first._m[0])
    assert second._v[0] == pytest.approx(first._v[0])
    assert second.t == first.t
    assert second._steps == first._steps

    first_parameter.grad = np.array(-0.75)
    second_parameter.grad = np.array(-0.75)
    first.step()
    second.step()

    assert second_parameter.data == pytest.approx(first_parameter.data)
    assert second._m[0] == pytest.approx(first._m[0])
    assert second._v[0] == pytest.approx(first._v[0])
    assert second.t == first.t
    assert second._steps == first._steps


def test_scalar_zero_grad_set_to_none_still_uses_historical_contract():
    parameter = Tensor(2.0, requires_grad=True)
    optimizer = Adam([parameter], lr=0.1)
    (parameter * parameter).backward()

    optimizer.zero_grad(set_to_none=True)

    assert parameter.grad is None
