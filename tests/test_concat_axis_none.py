import numpy as np

import engine.ops as ops
from engine.tensor import Tensor


def test_concat_axis_none_flattens_heterogeneous_inputs_and_restores_grad_shapes():
    matrix = Tensor(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        requires_grad=True,
    )
    vector = Tensor(np.array([5.0, 6.0, 7.0]), requires_grad=True)
    scalar = Tensor(np.array(8.0), requires_grad=True)

    out = ops.concat((matrix, vector, scalar), axis=None)

    expected = np.concatenate(
        [matrix.data, vector.data, scalar.data],
        axis=None,
    )
    np.testing.assert_array_equal(out.data, expected)
    assert out.shape == (8,)

    upstream = np.arange(1.0, 9.0)
    out.backward(upstream)

    np.testing.assert_array_equal(matrix.grad, upstream[:4].reshape(2, 2))
    np.testing.assert_array_equal(vector.grad, upstream[4:7])
    np.testing.assert_array_equal(scalar.grad, np.array(upstream[7]))
    assert matrix.grad.shape == matrix.shape
    assert vector.grad.shape == vector.shape
    assert scalar.grad.shape == scalar.shape


def test_concat_axis_none_handles_empty_inputs():
    empty = Tensor(np.empty((0, 2)), requires_grad=True)
    values = Tensor(np.array([2.0, 4.0, 6.0]), requires_grad=True)

    out = ops.concat((empty, values), axis=None)
    np.testing.assert_array_equal(out.data, values.data)

    upstream = np.array([0.5, 1.5, 2.5])
    out.backward(upstream)

    assert empty.grad.shape == (0, 2)
    assert empty.grad.size == 0
    np.testing.assert_array_equal(values.grad, upstream)


def test_concat_axis_none_accumulates_repeated_tensor_slices():
    x = Tensor(np.array([[10.0, 20.0]]), requires_grad=True)

    out = ops.concat((x, x), axis=None)
    out.backward(np.array([1.0, 2.0, 3.0, 4.0]))

    np.testing.assert_array_equal(x.grad, np.array([[4.0, 6.0]]))
    assert x.grad.shape == x.shape


def test_concat_axis_none_preserves_non_trainable_input_behavior():
    trainable = Tensor(np.array([[1.0], [2.0]]), requires_grad=True)
    frozen = Tensor(np.array([3.0, 4.0]), requires_grad=False)

    out = ops.concat((trainable, frozen), axis=None)
    out.backward(np.array([5.0, 6.0, 7.0, 8.0]))

    np.testing.assert_array_equal(trainable.grad, np.array([[5.0], [6.0]]))
    assert frozen.grad is None


def test_concat_regular_axis_backward_is_unchanged():
    left = Tensor(np.array([[1.0], [2.0]]), requires_grad=True)
    right = Tensor(
        np.array([[3.0, 4.0], [5.0, 6.0]]),
        requires_grad=True,
    )

    out = ops.concat((left, right), axis=1)
    upstream = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out.backward(upstream)

    np.testing.assert_array_equal(left.grad, upstream[:, :1])
    np.testing.assert_array_equal(right.grad, upstream[:, 1:])
