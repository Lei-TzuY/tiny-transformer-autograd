import numpy as np
import pytest

from engine.tensor import Tensor
from engine.value_and_grad import value_and_grad


def test_single_arg_value_and_grad_runs_forward_once_and_restores_grad_buffer():
    x = Tensor(3.0, requires_grad=True)
    x.grad[...] = 17.0
    original_grad = x.grad
    calls = []

    def loss(value, *, scale):
        calls.append(float(value.data))
        return scale * value * value

    transformed = value_and_grad(loss)
    value, gradient = transformed(x, scale=2.0)

    assert calls == [3.0]
    assert isinstance(value, Tensor)
    assert value.shape == ()
    assert float(value.data) == pytest.approx(18.0)
    assert isinstance(gradient, np.ndarray)
    assert gradient.shape == ()
    assert float(gradient) == pytest.approx(12.0)
    assert x.grad is original_grad
    assert float(x.grad) == pytest.approx(17.0)

    gradient[...] = -999.0
    assert float(x.grad) == pytest.approx(17.0)


def test_tuple_argnums_preserve_requested_gradient_order():
    x = Tensor(2.0, requires_grad=True)
    y = Tensor(5.0, requires_grad=True)

    def loss(left, label, right):
        assert label == "metadata"
        return left * right + left * left

    transformed = value_and_grad(loss, argnums=(2, 0))
    value, gradients = transformed(x, "metadata", y)

    assert float(value.data) == pytest.approx(14.0)
    assert isinstance(gradients, tuple)
    assert len(gradients) == 2
    assert float(gradients[0]) == pytest.approx(2.0)
    assert float(gradients[1]) == pytest.approx(9.0)


def test_negative_single_argnum_uses_python_positional_indexing():
    x = Tensor(2.0, requires_grad=True)
    y = Tensor(7.0, requires_grad=True)

    transformed = value_and_grad(lambda left, right: left * right, argnums=-1)
    value, gradient = transformed(x, y)

    assert float(value.data) == pytest.approx(14.0)
    assert float(gradient) == pytest.approx(2.0)


def test_numpy_integer_argnums_are_normalized():
    x = Tensor(4.0, requires_grad=True)
    y = Tensor(3.0, requires_grad=True)

    transformed = value_and_grad(
        lambda left, right: left * right,
        argnums=(np.int64(1), np.int32(0)),
    )
    _, gradients = transformed(x, y)

    assert float(gradients[0]) == pytest.approx(4.0)
    assert float(gradients[1]) == pytest.approx(3.0)


def test_unselected_positional_arguments_may_be_non_tensors():
    x = Tensor(2.0, requires_grad=True)

    def loss(value, coefficient, payload):
        assert payload == {"kind": "metadata"}
        return coefficient * value * value

    transformed = value_and_grad(loss, argnums=0)
    value, gradient = transformed(x, 3.5, {"kind": "metadata"})

    assert float(value.data) == pytest.approx(14.0)
    assert float(gradient) == pytest.approx(14.0)


def test_has_aux_returns_auxiliary_payload_unchanged():
    x = Tensor(3.0, requires_grad=True)
    captured = {}

    def loss(value):
        aux_tensor = value + 1.0
        aux_tensor.grad[...] = 23.0
        aux = {
            "tensor": aux_tensor,
            "label": "diagnostic",
            "array": np.array([1.0, 2.0]),
        }
        captured["aux"] = aux
        return value * value, aux

    transformed = value_and_grad(loss, has_aux=True)
    (value, aux), gradient = transformed(x)

    assert aux is captured["aux"]
    assert aux["label"] == "diagnostic"
    np.testing.assert_array_equal(aux["array"], np.array([1.0, 2.0]))
    assert float(aux["tensor"].data) == pytest.approx(4.0)
    assert float(aux["tensor"].grad) == pytest.approx(23.0)
    assert float(value.data) == pytest.approx(9.0)
    assert float(gradient) == pytest.approx(6.0)


def test_has_aux_accepts_numpy_boolean_flag():
    x = Tensor(2.0, requires_grad=True)
    transformed = value_and_grad(
        lambda value: (value * value, "aux"),
        has_aux=np.bool_(True),
    )

    (value, aux), gradient = transformed(x)
    assert aux == "aux"
    assert float(value.data) == pytest.approx(4.0)
    assert float(gradient) == pytest.approx(4.0)


def test_output_tensor_is_still_usable_after_functional_gradient():
    x = Tensor(2.0, requires_grad=True)
    transformed = value_and_grad(lambda value: value * value)

    value, gradient = transformed(x)
    assert float(gradient) == pytest.approx(4.0)

    x.zero_grad()
    value.backward()
    assert float(x.grad) == pytest.approx(4.0)


def test_existing_gradients_on_multiple_selected_inputs_are_restored_exactly():
    x = Tensor(2.0, requires_grad=True)
    y = Tensor(5.0, requires_grad=True)
    x.grad[...] = 11.0
    y.grad[...] = -7.0
    x_grad = x.grad
    y_grad = y.grad

    transformed = value_and_grad(lambda left, right: left * right, argnums=(0, 1))
    _, gradients = transformed(x, y)

    assert float(gradients[0]) == pytest.approx(5.0)
    assert float(gradients[1]) == pytest.approx(2.0)
    assert x.grad is x_grad
    assert y.grad is y_grad
    assert float(x.grad) == pytest.approx(11.0)
    assert float(y.grad) == pytest.approx(-7.0)


def test_gradient_arrays_are_independent_between_calls():
    x = Tensor(2.0, requires_grad=True)
    transformed = value_and_grad(lambda value: value * value)

    _, first = transformed(x)
    _, second = transformed(x)
    assert first is not second
    assert float(first) == pytest.approx(4.0)
    assert float(second) == pytest.approx(4.0)

    first[...] = 100.0
    assert float(second) == pytest.approx(4.0)


def test_forward_randomness_is_consumed_exactly_once():
    x = Tensor(2.0, requires_grad=True)
    calls = []

    np.random.seed(12345)
    expected_scale = np.random.random()
    expected_next = np.random.random()

    np.random.seed(12345)

    def loss(value):
        scale = np.random.random()
        calls.append(scale)
        return value * scale

    transformed = value_and_grad(loss)
    value, gradient = transformed(x)
    actual_next = np.random.random()

    assert calls == [expected_scale]
    assert float(value.data) == pytest.approx(2.0 * expected_scale)
    assert float(gradient) == pytest.approx(expected_scale)
    assert actual_next == pytest.approx(expected_next)


def test_kwargs_are_forwarded_without_becoming_differentiation_targets():
    x = Tensor(3.0, requires_grad=True)

    def loss(value, *, offset, multiplier=1.0):
        return multiplier * value * value + offset

    transformed = value_and_grad(loss)
    value, gradient = transformed(x, offset=4.0, multiplier=2.5)

    assert float(value.data) == pytest.approx(26.5)
    assert float(gradient) == pytest.approx(15.0)


def test_callable_object_is_supported():
    class Loss:
        def __init__(self):
            self.calls = 0

        def __call__(self, value):
            self.calls += 1
            return value * value

    loss = Loss()
    transformed = value_and_grad(loss)
    value, gradient = transformed(Tensor(4.0, requires_grad=True))

    assert loss.calls == 1
    assert float(value.data) == pytest.approx(16.0)
    assert float(gradient) == pytest.approx(8.0)
