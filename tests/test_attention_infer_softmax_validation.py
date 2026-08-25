"""NumPy attention softmax must match the graph non-finite contract."""

import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.tensor import Tensor
import engine.ops as ops
from nn.attention import SelfAttention, _softmax


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_numpy_softmax_rejects_invalid_nonfinite_like_graph(bad):
    values = np.array([[0.0, bad]])

    with pytest.raises(ValueError, match="must not contain NaN or \\+inf"):
        ops.softmax(Tensor(values))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="must not contain NaN or \\+inf"):
            _softmax(values)


def test_numpy_softmax_preserves_valid_negative_infinity_semantics():
    values = np.array(
        [
            [0.0, -np.inf, -1.0],
            [-np.inf, -np.inf, -np.inf],
            [1000.0, 999.0, -np.inf],
        ]
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = _softmax(values)
        expected = ops.softmax(Tensor(values)).data

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual[1], np.zeros(3))
    assert actual[0, 1] == 0.0
    assert actual[2, 2] == 0.0


def test_attention_forward_and_infer_reject_unrepresentable_score_consistently():
    attention = SelfAttention(1)
    attention.W_q.weight.data[:] = 1.0
    attention.W_k.weight.data[:] = 1.0
    data = np.array([[[1e308]]])
    message = "softmax inputs must not contain NaN or \\+inf"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match=message):
            attention(Tensor(data))
        with pytest.raises(ValueError, match=message):
            attention.infer(data)
