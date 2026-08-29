import numpy as np
import pytest

from engine.adafactor import Adafactor
from engine.tensor import Tensor


_TINY = np.nextafter(0.0, 1.0)


def _matrix_direction(gradient):
    squared = gradient * gradient
    row = np.mean(squared, axis=-1)
    col = np.mean(squared, axis=-2)
    row_mean = np.mean(row, axis=-1, keepdims=True)
    row_factor = np.sqrt(row_mean / row)
    return gradient / np.expand_dims(np.sqrt(col), -2) * np.expand_dims(
        row_factor, -1
    )


def _physical_full_moment(state):
    return state["v"] * (state["scale"] * state["scale"])


def test_factored_matrix_first_step_matches_hand_calculation():
    parameter = Tensor([[10.0, 20.0], [30.0, 40.0]], requires_grad=True)
    gradient = np.array([[1.0, 2.0], [3.0, 4.0]])
    parameter.grad = gradient.copy()
    optimizer = Adafactor(
        parameter,
        lr=0.1,
        beta2=0.0,
        eps=_TINY,
        clip_threshold=10.0,
    )

    expected_direction = _matrix_direction(gradient)
    expected = parameter.data.copy() - 0.1 * expected_direction

    assert optimizer.step() is None

    np.testing.assert_allclose(parameter.data, expected, rtol=1e-14, atol=0.0)
    state = optimizer.state_dict()["states"][0]
    assert state["kind"] == "factored"
    assert state["step"] == 1
    assert state["row"].shape == (2,)
    assert state["col"].shape == (2,)
    assert max(np.max(state["row"]), np.max(state["col"])) == pytest.approx(1.0)


def test_vector_uses_unfactored_second_moment_and_first_step_is_sign_update():
    parameter = Tensor([5.0, -5.0, 1.0], requires_grad=True)
    parameter.grad = np.array([3.0, -4.0, 2.0])
    optimizer = Adafactor(
        parameter,
        lr=0.25,
        beta2=0.0,
        eps=_TINY,
        clip_threshold=10.0,
    )

    optimizer.step()

    np.testing.assert_allclose(parameter.data, [4.75, -4.75, 0.75])
    state = optimizer.state_dict()["states"][0]
    assert state["kind"] == "full"
    assert state["v"].shape == (3,)
    assert state["step"] == 1


def test_scalar_parameter_is_supported():
    parameter = Tensor(2.0, requires_grad=True)
    parameter.grad = np.array(-7.0)
    optimizer = Adafactor(
        parameter,
        lr=0.1,
        beta2=0.0,
        eps=_TINY,
        clip_threshold=10.0,
    )

    optimizer.step()

    assert parameter.shape == ()
    assert parameter.data.item() == pytest.approx(2.1)
    state = optimizer.state_dict()["states"][0]
    assert state["kind"] == "full"
    assert state["v"].shape == ()


def test_rms_update_clipping_scales_unfactored_direction():
    parameter = Tensor([0.0, 0.0], requires_grad=True)
    parameter.grad = np.array([1.0, -2.0])
    optimizer = Adafactor(
        parameter,
        lr=1.0,
        beta2=0.0,
        eps=_TINY,
        clip_threshold=0.25,
    )

    optimizer.step()

    np.testing.assert_allclose(parameter.data, [-0.25, 0.25], rtol=1e-14, atol=0.0)


def test_unfactored_beta2_second_moment_matches_ema():
    parameter = Tensor([0.0, 0.0], requires_grad=True)
    optimizer = Adafactor(
        parameter,
        lr=0.01,
        beta2=0.5,
        eps=_TINY,
        clip_threshold=100.0,
    )

    parameter.grad = np.array([1.0, 2.0])
    optimizer.step()
    first = optimizer.state_dict()["states"][0]
    np.testing.assert_allclose(
        _physical_full_moment(first),
        0.5 * np.array([1.0, 4.0]),
        rtol=2e-15,
        atol=0.0,
    )

    parameter.grad = np.array([3.0, 4.0])
    optimizer.step()
    second = optimizer.state_dict()["states"][0]
    expected = 0.5 * (0.5 * np.array([1.0, 4.0])) + 0.5 * np.array([9.0, 16.0])
    np.testing.assert_allclose(
        _physical_full_moment(second), expected, rtol=3e-15, atol=0.0
    )
    assert second["step"] == 2


def test_grad_none_skips_parameter_and_does_not_advance_its_state():
    first = Tensor([1.0], requires_grad=True)
    second = Tensor([2.0], requires_grad=True)
    first.grad = np.array([1.0])
    second.grad = None
    optimizer = Adafactor(
        [first, second],
        lr=0.1,
        beta2=0.0,
        eps=_TINY,
        clip_threshold=10.0,
    )

    second_version = second._version
    optimizer.step()

    assert first.data.item() == pytest.approx(0.9)
    assert second.data.item() == pytest.approx(2.0)
    assert second._version == second_version
    assert optimizer.steps == (1, 0)


def test_zero_gradient_advances_moment_state_but_is_parameter_noop():
    parameter = Tensor([3.0, -4.0], requires_grad=True)
    parameter.grad = np.zeros(2)
    optimizer = Adafactor(
        parameter,
        lr=0.5,
        beta2=0.5,
        eps=1e-12,
        clip_threshold=1.0,
    )
    version = parameter._version

    optimizer.step()

    np.testing.assert_array_equal(parameter.data, [3.0, -4.0])
    assert parameter._version == version
    state = optimizer.state_dict()["states"][0]
    assert state["step"] == 1
    assert state["scale"] > 0.0
    assert np.any(state["v"] > 0.0)


def test_three_dimensional_parameter_factors_last_two_axes():
    parameter = Tensor(np.zeros((2, 3, 4)), requires_grad=True)
    parameter.grad = np.arange(1.0, 25.0).reshape(2, 3, 4)
    optimizer = Adafactor(
        parameter,
        lr=0.01,
        beta2=0.0,
        eps=_TINY,
        clip_threshold=10.0,
    )

    optimizer.step()

    state = optimizer.state_dict()["states"][0]
    assert state["kind"] == "factored"
    assert state["row"].shape == (2, 3)
    assert state["col"].shape == (2, 4)
    assert np.all(np.isfinite(parameter.data))


def test_empty_parameter_is_a_valid_noop():
    parameter = Tensor(np.empty((0, 3)), requires_grad=True)
    parameter.grad = np.empty((0, 3))
    optimizer = Adafactor(parameter)
    version = parameter._version

    optimizer.step()

    assert parameter.shape == (0, 3)
    assert parameter._version == version
    assert optimizer.steps == (0,)


def test_zero_grad_supports_in_place_and_set_to_none_for_scalar_and_vector():
    scalar = Tensor(1.0, requires_grad=True)
    vector = Tensor([1.0, 2.0], requires_grad=True)
    scalar.grad = np.array(3.0)
    vector.grad = np.array([4.0, 5.0])
    scalar_reference = scalar.grad
    vector_reference = vector.grad
    optimizer = Adafactor([scalar, vector])

    optimizer.zero_grad()

    assert scalar.grad is scalar_reference
    assert vector.grad is vector_reference
    assert scalar.grad.item() == 0.0
    np.testing.assert_array_equal(vector.grad, [0.0, 0.0])

    scalar.grad[...] = 7.0
    vector.grad[...] = [8.0, 9.0]
    optimizer.zero_grad(set_to_none=True)
    assert scalar.grad is None
    assert vector.grad is None


def test_step_preserves_gradient_values_and_references():
    parameter = Tensor([1.0, 2.0], requires_grad=True)
    gradient = np.array([3.0, 4.0])
    parameter.grad = gradient
    optimizer = Adafactor(parameter, beta2=0.0, eps=_TINY)
    before = gradient.copy()

    optimizer.step()

    assert parameter.grad is gradient
    np.testing.assert_array_equal(gradient, before)


def test_constructor_and_step_do_not_consume_numpy_global_rng():
    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad = np.array([2.0])
    before = np.random.get_state()

    optimizer = Adafactor(parameter)
    optimizer.step()

    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
