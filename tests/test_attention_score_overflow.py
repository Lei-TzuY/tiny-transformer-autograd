"""Scaled attention scores must not overflow before their scale is applied."""

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.tensor import Tensor
import engine.ops as ops
from nn.attention import (
    MultiHeadAttention,
    SelfAttention,
    _scaled_dot_product_scores,
    _scaled_dot_product_scores_np,
)


_EXTREME = 8e153


def _set_identity_attention(module):
    for projection in (module.W_q, module.W_k, module.W_v, module.out_proj):
        projection.weight.data[:] = np.eye(module.d_model)
        if projection.bias is not None:
            projection.bias.data[:] = 0.0


def test_scaled_dot_product_recovers_representable_score_and_vjp():
    query = Tensor(np.full((1, 1, 4), _EXTREME), requires_grad=True)
    key_t = Tensor(np.full((1, 4, 1), _EXTREME), requires_grad=True)
    upstream = np.array([[[1e-308]]])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        score = _scaled_dot_product_scores(query, key_t, 0.5)
        score.backward(upstream)

    expected_score = (np.full((1, 1, 4), _EXTREME) * 0.5) @ np.full(
        (1, 4, 1), _EXTREME
    )
    assert np.isfinite(score.data).all()
    np.testing.assert_array_equal(score.data, expected_score)
    np.testing.assert_allclose(
        query.grad, np.full(query.shape, 4e-155), rtol=2e-15, atol=0.0
    )
    np.testing.assert_allclose(
        key_t.grad, np.full(key_t.shape, 4e-155), rtol=2e-15, atol=0.0
    )


def test_numpy_scaled_dot_product_recovers_same_representable_score():
    query = np.full((1, 1, 4), _EXTREME)
    key_t = np.full((1, 4, 1), _EXTREME)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = _scaled_dot_product_scores_np(query, key_t, 0.5)

    expected = (query * 0.5) @ key_t
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual, expected)


def test_ordinary_tensor_scores_preserve_historical_bits_and_gradients():
    rng = np.random.default_rng(1234)
    query_data = rng.standard_normal((2, 3, 4))
    key_data = rng.standard_normal((2, 4, 5))
    upstream = rng.standard_normal((2, 3, 5))
    scale = 0.5

    query_new = Tensor(query_data, requires_grad=True)
    key_new = Tensor(key_data, requires_grad=True)
    new = _scaled_dot_product_scores(query_new, key_new, scale)
    new.backward(upstream)

    query_old = Tensor(query_data, requires_grad=True)
    key_old = Tensor(key_data, requires_grad=True)
    old = ops.matmul(query_old, key_old) * scale
    old.backward(upstream)

    np.testing.assert_array_equal(new.data, old.data)
    np.testing.assert_array_equal(query_new.grad, query_old.grad)
    np.testing.assert_array_equal(key_new.grad, key_old.grad)
    np.testing.assert_array_equal(
        _scaled_dot_product_scores_np(query_data, key_data, scale),
        query_data @ key_data * scale,
    )


def test_overflowing_batch_does_not_change_safe_batch_score_bits_or_gradients():
    rng = np.random.default_rng(2026)
    scale = 3.0 ** -0.5
    ordinary_query = rng.standard_normal((1, 1, 3))
    ordinary_key = rng.standard_normal((1, 3, 2))
    ordinary_upstream = rng.standard_normal((1, 1, 2))

    query_data = np.concatenate(
        [np.full((1, 1, 3), _EXTREME), ordinary_query], axis=0
    )
    key_data = np.concatenate(
        [np.full((1, 3, 2), _EXTREME), ordinary_key], axis=0
    )
    upstream = np.concatenate([np.zeros((1, 1, 2)), ordinary_upstream], axis=0)

    query_mixed = Tensor(query_data, requires_grad=True)
    key_mixed = Tensor(key_data, requires_grad=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mixed = _scaled_dot_product_scores(query_mixed, key_mixed, scale)
        mixed.backward(upstream)
        mixed_np = _scaled_dot_product_scores_np(query_data, key_data, scale)

    query_old = Tensor(ordinary_query, requires_grad=True)
    key_old = Tensor(ordinary_key, requires_grad=True)
    old = ops.matmul(query_old, key_old) * scale
    old.backward(ordinary_upstream)

    assert np.isfinite(mixed.data[0]).all()
    np.testing.assert_array_equal(mixed.data[1], old.data[0])
    np.testing.assert_array_equal(query_mixed.grad[1], query_old.grad[0])
    np.testing.assert_array_equal(key_mixed.grad[1], key_old.grad[0])
    np.testing.assert_array_equal(
        mixed_np[1], (ordinary_query @ ordinary_key * scale)[0]
    )


def test_self_attention_extreme_finite_single_token_is_warning_free():
    attention = SelfAttention(4)
    _set_identity_attention(attention)
    data = np.full((1, 1, 4), _EXTREME)
    upstream = np.array([[[0.25, -0.5, 0.75, 1.0]]])
    x = Tensor(data, requires_grad=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = attention(x)
        inferred, cache = attention.infer(data)
        out.backward(upstream)

    assert np.isfinite(out.data).all()
    assert np.isfinite(inferred).all()
    assert np.isfinite(x.grad).all()
    np.testing.assert_allclose(out.data, data, rtol=2e-15, atol=0.0)
    np.testing.assert_allclose(inferred, data, rtol=2e-15, atol=0.0)
    np.testing.assert_allclose(x.grad, upstream, rtol=2e-15, atol=0.0)
    assert np.isfinite(cache["k"]).all()
    assert np.isfinite(cache["v"]).all()


def test_multihead_attention_extreme_finite_single_token_is_warning_free():
    attention = MultiHeadAttention(8, 2)
    _set_identity_attention(attention)
    data = np.full((1, 1, 8), _EXTREME)
    upstream = np.array(
        [[[0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8]]]
    )
    x = Tensor(data, requires_grad=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = attention(x)
        inferred, cache = attention.infer(data)
        out.backward(upstream)

    assert np.isfinite(out.data).all()
    assert np.isfinite(inferred).all()
    assert np.isfinite(x.grad).all()
    np.testing.assert_allclose(out.data, data, rtol=2e-15, atol=0.0)
    np.testing.assert_allclose(inferred, data, rtol=2e-15, atol=0.0)
    np.testing.assert_allclose(x.grad, upstream, rtol=2e-15, atol=0.0)
    assert np.isfinite(cache["k"]).all()
    assert np.isfinite(cache["v"]).all()
