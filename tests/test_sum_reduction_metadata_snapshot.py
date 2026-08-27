import numpy as np
import pytest

from engine import ops
from engine.tensor import Tensor


def test_sum_snapshots_mutable_scalar_axis_for_backward():
    x = Tensor(np.arange(4.0).reshape(2, 2), requires_grad=True)
    axis = np.array(0, dtype=np.int64)

    out = ops.sum(x, axis=axis)
    np.testing.assert_array_equal(out.data, np.array([2.0, 4.0]))

    axis[...] = 1
    out.backward(np.array([10.0, 20.0]))

    np.testing.assert_array_equal(
        x.grad,
        np.array([[10.0, 20.0], [10.0, 20.0]]),
    )


def test_sum_snapshots_mutable_axis_entries_inside_tuple():
    x = Tensor(np.arange(8.0).reshape(2, 2, 2), requires_grad=True)
    first_axis = np.array(0, dtype=np.int64)
    last_axis = np.array(2, dtype=np.int64)
    axis = (first_axis, last_axis)

    out = ops.sum(x, axis=axis)
    np.testing.assert_array_equal(out.data, np.array([10.0, 18.0]))

    first_axis[...] = 1
    out.backward(np.array([3.0, 7.0]))

    expected = np.broadcast_to(
        np.array([3.0, 7.0]).reshape(1, 2, 1),
        x.shape,
    )
    np.testing.assert_array_equal(x.grad, expected)


def test_sum_snapshots_mutable_keepdims_for_backward():
    x = Tensor(np.arange(4.0).reshape(2, 2), requires_grad=True)
    keepdims = np.array(0, dtype=np.int64)

    out = ops.sum(x, axis=1, keepdims=keepdims)
    np.testing.assert_array_equal(out.data, np.array([1.0, 5.0]))

    keepdims[...] = 1
    out.backward(np.array([5.0, 9.0]))

    np.testing.assert_array_equal(
        x.grad,
        np.array([[5.0, 5.0], [9.0, 9.0]]),
    )


def test_sum_preserves_numpy_integer_like_metadata_acceptance():
    x = Tensor(np.arange(6.0).reshape(2, 3), requires_grad=True)

    out = ops.sum(
        x,
        axis=np.array(1, dtype=np.int64),
        keepdims=np.array(1, dtype=np.int64),
    )
    np.testing.assert_array_equal(out.data, np.array([[3.0], [12.0]]))

    out.backward(np.array([[2.0], [4.0]]))
    np.testing.assert_array_equal(
        x.grad,
        np.array([[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]),
    )


@pytest.mark.parametrize(
    ("axis", "keepdims"),
    [
        (np.array([0], dtype=np.int64), False),
        (0, np.array([0, 1], dtype=np.int64)),
    ],
)
def test_sum_keeps_numpy_rejection_for_non_scalar_metadata(axis, keepdims):
    x = Tensor(np.arange(6.0).reshape(2, 3), requires_grad=True)

    with pytest.raises(TypeError):
        ops.sum(x, axis=axis, keepdims=keepdims)
