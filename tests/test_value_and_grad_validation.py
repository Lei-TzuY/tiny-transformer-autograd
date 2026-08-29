import numpy as np
import pytest

from engine.tensor import Tensor
from engine.value_and_grad import value_and_grad


def test_transform_creation_rejects_non_callable():
    with pytest.raises(TypeError, match="function must be callable"):
        value_and_grad(123)


@pytest.mark.parametrize(
    "argnums",
    [True, np.bool_(False), (), [], [0], (0.5,), (True,), (np.bool_(True),)],
)
def test_argnums_type_validation(argnums):
    with pytest.raises(TypeError, match="argnums must be an integer"):
        value_and_grad(lambda value: value, argnums=argnums)


def test_literal_duplicate_argnums_are_rejected_at_transform_creation():
    with pytest.raises(ValueError, match="must not contain duplicate indices"):
        value_and_grad(lambda left, right: left * right, argnums=(0, 0))


@pytest.mark.parametrize("has_aux", [0, 1, "yes", None, object()])
def test_has_aux_requires_boolean(has_aux):
    with pytest.raises(TypeError, match="has_aux must be a boolean"):
        value_and_grad(lambda value: value, has_aux=has_aux)


def test_out_of_range_argnum_is_rejected_before_forward():
    calls = []
    transformed = value_and_grad(lambda value: calls.append(value) or value, argnums=1)

    with pytest.raises(ValueError, match="index is out of range"):
        transformed(Tensor(1.0, requires_grad=True))
    assert calls == []


def test_too_negative_argnum_is_rejected_before_forward():
    calls = []
    transformed = value_and_grad(lambda value: calls.append(value) or value, argnums=-2)

    with pytest.raises(ValueError, match="index is out of range"):
        transformed(Tensor(1.0, requires_grad=True))
    assert calls == []


def test_negative_alias_duplicate_is_rejected_before_forward():
    calls = []

    def loss(left, right):
        calls.append((left, right))
        return left * right

    transformed = value_and_grad(loss, argnums=(-1, 1))
    with pytest.raises(ValueError, match="resolve to duplicate positional arguments"):
        transformed(
            Tensor(2.0, requires_grad=True),
            Tensor(3.0, requires_grad=True),
        )
    assert calls == []


def test_selected_non_tensor_is_rejected_before_forward():
    calls = []

    def loss(value):
        calls.append(value)
        return Tensor(1.0, requires_grad=True)

    transformed = value_and_grad(loss)
    with pytest.raises(TypeError, match="positional argument 0 must be a Tensor"):
        transformed(3.0)
    assert calls == []


def test_selected_frozen_tensor_is_rejected_before_forward():
    calls = []

    def loss(value):
        calls.append(value)
        return value

    transformed = value_and_grad(loss)
    with pytest.raises(ValueError, match="positional argument 0 must require gradients"):
        transformed(Tensor(3.0, requires_grad=False))
    assert calls == []


def test_same_tensor_selected_through_multiple_positions_is_rejected_before_forward():
    x = Tensor(2.0, requires_grad=True)
    calls = []

    def loss(left, right):
        calls.append((left, right))
        return left * right

    transformed = value_and_grad(loss, argnums=(0, 1))
    with pytest.raises(ValueError, match="distinct Tensor objects"):
        transformed(x, x)
    assert calls == []


def test_same_tensor_may_appear_in_unselected_position():
    x = Tensor(2.0, requires_grad=True)
    transformed = value_and_grad(lambda selected, alias: selected * alias, argnums=0)

    value, gradient = transformed(x, x)
    assert float(value.data) == pytest.approx(4.0)
    # The engine records one Tensor identity, so selecting only that identity
    # intentionally yields the total derivative of the live graph.
    assert float(gradient) == pytest.approx(4.0)


def test_non_tensor_function_value_is_rejected_after_exactly_one_forward():
    calls = []

    def loss(value):
        calls.append(value)
        return 3.0

    transformed = value_and_grad(loss)
    with pytest.raises(TypeError, match="function value must be a Tensor"):
        transformed(Tensor(1.0, requires_grad=True))
    assert len(calls) == 1


def test_non_scalar_tensor_value_is_rejected():
    x = Tensor([1.0, 2.0], requires_grad=True)
    transformed = value_and_grad(lambda value: value)

    with pytest.raises(ValueError, match="function value must be a scalar Tensor"):
        transformed(x)


def test_has_aux_requires_exact_pair_result():
    x = Tensor(2.0, requires_grad=True)

    with pytest.raises(TypeError, match=r"must return a \(value, aux\) tuple"):
        value_and_grad(lambda value: value * value, has_aux=True)(x)

    with pytest.raises(TypeError, match=r"must return a \(value, aux\) tuple"):
        value_and_grad(
            lambda value: (value * value, "a", "b"), has_aux=True
        )(x)

    with pytest.raises(TypeError, match=r"must return a \(value, aux\) tuple"):
        value_and_grad(lambda value: [value * value, "aux"], has_aux=True)(x)


def test_has_aux_still_requires_tensor_value():
    x = Tensor(2.0, requires_grad=True)
    transformed = value_and_grad(lambda value: (4.0, {"x": value}), has_aux=True)

    with pytest.raises(TypeError, match="function value must be a Tensor"):
        transformed(x)


def test_selected_input_must_be_reachable_from_scalar_value():
    x = Tensor(2.0, requires_grad=True)
    y = Tensor(3.0, requires_grad=True)
    x.grad[...] = 8.0
    y.grad[...] = 9.0
    x_grad = x.grad
    y_grad = y.grad

    transformed = value_and_grad(lambda left, right: left * left, argnums=(0, 1))
    with pytest.raises(ValueError, match="requested inputs must be reachable"):
        transformed(x, y)

    assert x.grad is x_grad
    assert y.grad is y_grad
    assert float(x.grad) == pytest.approx(8.0)
    assert float(y.grad) == pytest.approx(9.0)


def test_unrelated_trainable_scalar_output_is_rejected_as_unreachable():
    x = Tensor(2.0, requires_grad=True)
    transformed = value_and_grad(lambda value: Tensor(7.0, requires_grad=True))

    with pytest.raises(ValueError, match="requested inputs must be reachable"):
        transformed(x)


def test_plain_tuple_result_without_has_aux_is_not_implicitly_unpacked():
    x = Tensor(2.0, requires_grad=True)
    transformed = value_and_grad(lambda value: (value * value, "aux"))

    with pytest.raises(TypeError, match="function value must be a Tensor"):
        transformed(x)


def test_function_exception_propagates_without_a_second_call():
    x = Tensor(2.0, requires_grad=True)
    calls = []

    def loss(value):
        calls.append(value)
        raise RuntimeError("forward failed")

    transformed = value_and_grad(loss)
    with pytest.raises(RuntimeError, match="forward failed"):
        transformed(x)
    assert calls == [x]


def test_wrapper_preserves_standard_function_metadata():
    def named_loss(value):
        """A named loss for wrapper metadata testing."""
        return value * value

    transformed = value_and_grad(named_loss)
    assert transformed.__name__ == named_loss.__name__
    assert transformed.__doc__ == named_loss.__doc__
    assert transformed.__wrapped__ is named_loss
