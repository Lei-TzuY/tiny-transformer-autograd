"""NumPy GELU inference must match the graph path for large finite activations."""

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.tensor import Tensor
from nn.transformer import FeedForward, _gelu


def _historical_gelu(values):
    values = np.asarray(values, dtype=np.float64)
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * values * (
        1.0 + np.tanh(c * (values + 0.044715 * values ** 3))
    )


def test_numpy_gelu_large_finite_saturates_without_warning():
    values = np.array([1e155, -1e155, 1e200, -1e200], dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = _gelu(values)

    np.testing.assert_array_equal(
        result,
        np.array([1e155, -0.0, 1e200, -0.0], dtype=np.float64),
    )
    assert np.isfinite(result).all()


def test_numpy_gelu_keeps_ordinary_historical_arithmetic_exactly():
    values = np.linspace(-6.0, 6.0, 121, dtype=np.float64)

    np.testing.assert_array_equal(_gelu(values), _historical_gelu(values))


def test_feedforward_infer_matches_graph_for_large_finite_activation():
    layer = FeedForward(d_model=1, d_ff=1, dropout=0.0)
    layer.fc1.weight.data[:] = 1.0
    layer.fc1.bias.data[:] = 0.0
    layer.fc2.weight.data[:] = 1.0
    layer.fc2.bias.data[:] = 0.0
    values = np.array([[[1e200]], [[-1e200]]], dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        graph = layer(Tensor(values)).data
        inferred = layer.infer(values)

    np.testing.assert_array_equal(inferred, graph)
    np.testing.assert_array_equal(
        inferred,
        np.array([[[1e200]], [[-0.0]]], dtype=np.float64),
    )
