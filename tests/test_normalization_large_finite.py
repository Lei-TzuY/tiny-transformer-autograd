"""LayerNorm and RMSNorm must stay correct for large finite activations."""

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.tensor import Tensor
from nn.layers import LayerNorm, RMSNorm


def _row_scale(x):
    return np.max(np.abs(x), axis=-1, keepdims=True)


def _layernorm_reference(x, gamma, beta, upstream, eps):
    scale = _row_scale(x)
    z = x / scale
    eps_scaled = (eps / scale) / scale
    mean = z.mean(axis=-1, keepdims=True)
    diff = z - mean
    variance = (diff ** 2).mean(axis=-1, keepdims=True)
    inv_std = (variance + eps_scaled) ** -0.5
    x_hat = diff * inv_std
    output = x_hat * gamma + beta

    weighted = upstream * gamma
    width = x.shape[-1]
    grad_z = (
        inv_std
        / width
        * (
            width * weighted
            - weighted.sum(axis=-1, keepdims=True)
            - x_hat * (weighted * x_hat).sum(axis=-1, keepdims=True)
        )
    )
    grad_x = grad_z / scale
    grad_gamma = (upstream * x_hat).sum(axis=0)
    grad_beta = upstream.sum(axis=0)
    return output, grad_x, grad_gamma, grad_beta


def _rmsnorm_reference(x, gamma, upstream, eps):
    scale = _row_scale(x)
    z = x / scale
    eps_scaled = (eps / scale) / scale
    mean_square = (z ** 2).mean(axis=-1, keepdims=True)
    inv_rms = (mean_square + eps_scaled) ** -0.5
    normalised = z * inv_rms
    output = normalised * gamma

    weighted = upstream * gamma
    grad_z = weighted * inv_rms - z * (inv_rms ** 3) * (
        weighted * z
    ).mean(axis=-1, keepdims=True)
    grad_x = grad_z / scale
    grad_gamma = (upstream * normalised).sum(axis=0)
    return output, grad_x, grad_gamma


def test_layernorm_large_finite_forward_infer_and_backward():
    norm = LayerNorm(4)
    norm.gamma.data[:] = [0.5, 1.0, 1.5, 2.0]
    norm.beta.data[:] = [-0.25, 0.5, 0.75, -1.0]
    data = 1e200 * np.array(
        [[1.0, 0.5, -0.2, -0.8], [-0.7, 0.1, 0.4, 1.0]],
        dtype=np.float64,
    )
    upstream = np.array(
        [[0.7, -0.2, 0.3, 1.1], [-0.4, 0.6, -0.8, 0.2]],
        dtype=np.float64,
    )
    expected, expected_dx, expected_dgamma, expected_dbeta = _layernorm_reference(
        data,
        norm.gamma.data.copy(),
        norm.beta.data.copy(),
        upstream,
        norm.eps,
    )

    x = Tensor(data, requires_grad=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = norm(x)
        inferred = norm.infer(data)
        out.backward(upstream)

    assert np.isfinite(out.data).all()
    assert np.isfinite(inferred).all()
    assert np.isfinite(x.grad).all()
    np.testing.assert_allclose(out.data, expected, rtol=2e-15, atol=0.0)
    np.testing.assert_allclose(inferred, expected, rtol=2e-15, atol=0.0)
    np.testing.assert_allclose(x.grad, expected_dx, rtol=3e-14, atol=0.0)
    np.testing.assert_allclose(norm.gamma.grad, expected_dgamma, rtol=2e-15, atol=0.0)
    np.testing.assert_allclose(norm.beta.grad, expected_dbeta, rtol=0.0, atol=0.0)


def test_rmsnorm_large_finite_forward_infer_and_backward():
    norm = RMSNorm(4)
    norm.gamma.data[:] = [0.5, 1.0, 1.5, 2.0]
    data = 1e200 * np.array(
        [[1.0, 0.5, -0.2, -0.8], [-0.7, 0.1, 0.4, 1.0]],
        dtype=np.float64,
    )
    upstream = np.array(
        [[0.7, -0.2, 0.3, 1.1], [-0.4, 0.6, -0.8, 0.2]],
        dtype=np.float64,
    )
    expected, expected_dx, expected_dgamma = _rmsnorm_reference(
        data, norm.gamma.data.copy(), upstream, norm.eps
    )

    x = Tensor(data, requires_grad=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = norm(x)
        inferred = norm.infer(data)
        out.backward(upstream)

    assert np.isfinite(out.data).all()
    assert np.isfinite(inferred).all()
    assert np.isfinite(x.grad).all()
    np.testing.assert_allclose(out.data, expected, rtol=2e-15, atol=0.0)
    np.testing.assert_allclose(inferred, expected, rtol=2e-15, atol=0.0)
    np.testing.assert_allclose(x.grad, expected_dx, rtol=3e-14, atol=0.0)
    np.testing.assert_allclose(norm.gamma.grad, expected_dgamma, rtol=2e-15, atol=0.0)


def test_extreme_row_does_not_change_ordinary_row_layernorm_arithmetic():
    mixed = LayerNorm(4)
    single = LayerNorm(4)
    gamma = np.array([0.7, 1.1, 1.3, 0.9])
    beta = np.array([-0.2, 0.4, 0.1, -0.5])
    mixed.gamma.data[:] = single.gamma.data[:] = gamma
    mixed.beta.data[:] = single.beta.data[:] = beta

    ordinary = np.array([0.25, -0.75, 1.5, -0.5])
    extreme = 1e200 * np.array([1.0, 0.5, -0.2, -0.8])
    mixed_x = Tensor(np.stack([extreme, ordinary]), requires_grad=True)
    single_x = Tensor(ordinary[None, :], requires_grad=True)
    ordinary_upstream = np.array([[0.3, -0.6, 0.2, 0.8]])

    mixed_out = mixed(mixed_x)
    single_out = single(single_x)
    mixed_out.backward(np.vstack([np.zeros(4), ordinary_upstream]))
    single_out.backward(ordinary_upstream)

    np.testing.assert_array_equal(mixed_out.data[1], single_out.data[0])
    np.testing.assert_array_equal(mixed_x.grad[1], single_x.grad[0])


def test_extreme_row_does_not_change_ordinary_row_rmsnorm_arithmetic():
    mixed = RMSNorm(4)
    single = RMSNorm(4)
    gamma = np.array([0.7, 1.1, 1.3, 0.9])
    mixed.gamma.data[:] = single.gamma.data[:] = gamma

    ordinary = np.array([0.25, -0.75, 1.5, -0.5])
    extreme = 1e200 * np.array([1.0, 0.5, -0.2, -0.8])
    mixed_x = Tensor(np.stack([extreme, ordinary]), requires_grad=True)
    single_x = Tensor(ordinary[None, :], requires_grad=True)
    ordinary_upstream = np.array([[0.3, -0.6, 0.2, 0.8]])

    mixed_out = mixed(mixed_x)
    single_out = single(single_x)
    mixed_out.backward(np.vstack([np.zeros(4), ordinary_upstream]))
    single_out.backward(ordinary_upstream)

    np.testing.assert_array_equal(mixed_out.data[1], single_out.data[0])
    np.testing.assert_array_equal(mixed_x.grad[1], single_x.grad[0])
