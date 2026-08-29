import numpy as np
import pytest

from engine.grad_mode import no_grad
from engine.tensor import Tensor
from engine.value_and_grad import value_and_grad


def _assert_rng_state_equal(left, right):
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def test_unselected_reachable_tensor_grad_buffer_is_restored():
    x = Tensor(2.0, requires_grad=True)
    y = Tensor(3.0, requires_grad=True)
    x.grad[...] = 11.0
    y.grad[...] = -5.0
    x_grad = x.grad
    y_grad = y.grad

    transformed = value_and_grad(lambda left, right: left * right, argnums=0)
    value, gradient = transformed(x, y)

    assert float(value.data) == pytest.approx(6.0)
    assert float(gradient) == pytest.approx(3.0)
    assert x.grad is x_grad
    assert y.grad is y_grad
    assert float(x.grad) == pytest.approx(11.0)
    assert float(y.grad) == pytest.approx(-5.0)


def test_intermediate_and_output_grad_objects_are_restored_exactly():
    x = Tensor(2.0, requires_grad=True)
    captured = {}

    def loss(value):
        intermediate = value * value
        intermediate.grad[...] = 31.0
        intermediate_grad = intermediate.grad
        output = intermediate * value
        output.grad[...] = -19.0
        output_grad = output.grad
        captured.update(
            intermediate=intermediate,
            intermediate_grad=intermediate_grad,
            output=output,
            output_grad=output_grad,
        )
        return output

    value, gradient = value_and_grad(loss)(x)

    assert value is captured["output"]
    assert float(value.data) == pytest.approx(8.0)
    assert float(gradient) == pytest.approx(12.0)
    assert captured["intermediate"].grad is captured["intermediate_grad"]
    assert captured["output"].grad is captured["output_grad"]
    assert float(captured["intermediate"].grad) == pytest.approx(31.0)
    assert float(captured["output"].grad) == pytest.approx(-19.0)


def test_transform_does_not_mutate_tensor_data_or_versions():
    x = Tensor(2.0, requires_grad=True)
    y = Tensor(4.0, requires_grad=True)
    x_data = np.array(x.data, copy=True)
    y_data = np.array(y.data, copy=True)
    x_version = x._version
    y_version = y._version

    _, gradients = value_and_grad(
        lambda left, right: left * right + left * left,
        argnums=(0, 1),
    )(x, y)

    assert float(gradients[0]) == pytest.approx(8.0)
    assert float(gradients[1]) == pytest.approx(2.0)
    np.testing.assert_array_equal(x.data, x_data)
    np.testing.assert_array_equal(y.data, y_data)
    assert x._version == x_version
    assert y._version == y_version


def test_stale_graph_failure_restores_persistent_gradient_buffer():
    x = Tensor(2.0, requires_grad=True)
    x.grad[...] = 7.0
    original_grad = x.grad
    captured = {}

    def stale_loss(value):
        intermediate = value * value
        intermediate.grad[...] = 13.0
        captured["intermediate"] = intermediate
        captured["grad"] = intermediate.grad
        output = intermediate * value
        value.data[...] = 3.0
        return output

    with pytest.raises(RuntimeError, match="tensor data was modified after forward"):
        value_and_grad(stale_loss)(x)

    assert x.grad is original_grad
    assert float(x.grad) == pytest.approx(7.0)
    assert captured["intermediate"].grad is captured["grad"]
    assert float(captured["intermediate"].grad) == pytest.approx(13.0)
    assert float(x.data) == pytest.approx(3.0)
    assert x._version == 1


def test_no_grad_scope_is_respected_instead_of_silently_reenabled():
    x = Tensor(2.0, requires_grad=True)
    x.grad[...] = 9.0
    original_grad = x.grad
    calls = []

    transformed = value_and_grad(
        lambda value: calls.append(value) or value * value,
    )
    with no_grad():
        with pytest.raises(ValueError, match="output must require gradients"):
            transformed(x)

    assert calls == [x]
    assert x.grad is original_grad
    assert float(x.grad) == pytest.approx(9.0)


def test_preforward_validation_failure_is_rng_neutral():
    np.random.seed(2026)
    before = np.random.get_state()
    calls = []

    transformed = value_and_grad(
        lambda value: calls.append(np.random.random()) or value,
        argnums=1,
    )
    with pytest.raises(ValueError, match="index is out of range"):
        transformed(Tensor(1.0, requires_grad=True))

    after = np.random.get_state()
    assert calls == []
    _assert_rng_state_equal(before, after)


def test_reachability_failure_does_not_change_rng_beyond_single_forward():
    x = Tensor(2.0, requires_grad=True)
    y = Tensor(3.0, requires_grad=True)

    np.random.seed(99)
    expected_draw = np.random.random()
    expected_next = np.random.random()
    np.random.seed(99)

    calls = []

    def loss(left, right):
        calls.append(np.random.random())
        return left * left

    transformed = value_and_grad(loss, argnums=(0, 1))
    with pytest.raises(ValueError, match="requested inputs must be reachable"):
        transformed(x, y)

    assert calls == [expected_draw]
    assert np.random.random() == pytest.approx(expected_next)


def test_auxiliary_tensor_outside_value_graph_is_not_touched():
    x = Tensor(3.0, requires_grad=True)
    aux_source = Tensor(5.0, requires_grad=True)
    aux_source.grad[...] = 41.0
    original_grad = aux_source.grad

    def loss(value, source):
        aux = source * source
        aux.grad[...] = -17.0
        return value * value, aux

    (value, aux), gradient = value_and_grad(loss, argnums=0, has_aux=True)(
        x, aux_source
    )

    assert float(value.data) == pytest.approx(9.0)
    assert float(gradient) == pytest.approx(6.0)
    assert float(aux.data) == pytest.approx(25.0)
    assert float(aux.grad) == pytest.approx(-17.0)
    assert aux_source.grad is original_grad
    assert float(aux_source.grad) == pytest.approx(41.0)
