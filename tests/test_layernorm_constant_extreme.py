"""LayerNorm must retain epsilon semantics for constant extreme finite rows."""

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.tensor import Tensor
from nn.layers import LayerNorm, RMSNorm


def _constant_layernorm_dx(upstream, gamma, eps):
    weighted = upstream * gamma
    return (weighted - weighted.mean(axis=-1, keepdims=True)) / np.sqrt(eps)


def test_constant_extreme_layernorm_preserves_epsilon_and_jacobian():
    norm = LayerNorm(4)
    norm.gamma.data[:] = np.array([0.5, 1.0, 1.5, 2.0])
    norm.beta.data[:] = np.array([-0.25, 0.5, 0.75, -1.0])
    data = np.array(
        [
            [1e308, 1e308, 1e308, 1e308],
            [-1e308, -1e308, -1e308, -1e308],
        ],
        dtype=np.float64,
    )
    upstream = np.array(
        [[0.7, -0.2, 0.3, 1.1], [-0.4, 0.6, -0.8, 0.2]],
        dtype=np.float64,
    )
    expected_dx = _constant_layernorm_dx(
        upstream, norm.gamma.data.copy(), norm.eps
    )
    expected_beta_grad = upstream.sum(axis=0)

    x = Tensor(data, requires_grad=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = norm(x)
        inferred = norm.infer(data)
        out.backward(upstream)

    expected_output = np.broadcast_to(norm.beta.data, data.shape)
    np.testing.assert_array_equal(out.data, expected_output)
    np.testing.assert_array_equal(inferred, expected_output)
    np.testing.assert_allclose(x.grad, expected_dx, rtol=3e-15, atol=0.0)
    np.testing.assert_array_equal(norm.gamma.grad, np.zeros(4))
    np.testing.assert_array_equal(norm.beta.grad, expected_beta_grad)


def test_constant_extreme_row_stays_stable_when_sibling_requires_scaling():
    norm = LayerNorm(4)
    norm.gamma.data[:] = np.array([0.7, 1.1, 1.3, 0.9])
    norm.beta.data[:] = np.array([-0.2, 0.4, 0.1, -0.5])
    constant = np.full(4, 1e308, dtype=np.float64)
    varying = 1e200 * np.array([1.0, 0.5, -0.2, -0.8], dtype=np.float64)
    data = np.stack([constant, varying])
    upstream = np.array([[0.3, -0.6, 0.2, 0.8], [0.0, 0.0, 0.0, 0.0]])
    expected_dx = _constant_layernorm_dx(
        upstream[:1], norm.gamma.data.copy(), norm.eps
    )[0]

    x = Tensor(data, requires_grad=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = norm(x)
        inferred = norm.infer(data)
        out.backward(upstream)

    np.testing.assert_array_equal(out.data[0], norm.beta.data)
    np.testing.assert_array_equal(inferred[0], norm.beta.data)
    np.testing.assert_allclose(x.grad[0], expected_dx, rtol=3e-15, atol=0.0)
    assert np.isfinite(out.data[1]).all()
    assert np.isfinite(inferred[1]).all()


def test_rmsnorm_constant_extreme_row_keeps_existing_scaled_path():
    norm = RMSNorm(4)
    norm.gamma.data[:] = np.array([0.5, 1.0, 1.5, 2.0])
    data = np.full((1, 4), 1e308, dtype=np.float64)

    x = Tensor(data, requires_grad=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = norm(x)
        inferred = norm.infer(data)

    np.testing.assert_allclose(out.data[0], norm.gamma.data, rtol=2e-15, atol=0.0)
    np.testing.assert_allclose(inferred[0], norm.gamma.data, rtol=2e-15, atol=0.0)
