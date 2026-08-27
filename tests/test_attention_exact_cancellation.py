"""Recover finite scaled attention dots after both float64 paths overflow."""

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.tensor import Tensor
from nn.attention import _scaled_dot_product_scores, _scaled_dot_product_scores_np


_SCALE = 2.0 ** -0.5
_EXTREME = 1e308


def test_exact_cancellation_recovers_score_and_existing_vjp_path():
    query = Tensor(np.array([[[ _EXTREME, _EXTREME ]]]), requires_grad=True)
    key_t = Tensor(np.array([[[3.0], [-3.0]]]), requires_grad=True)
    upstream = np.array([[[4e-308]]])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        score = _scaled_dot_product_scores(query, key_t, _SCALE)
        score.backward(upstream)

    np.testing.assert_array_equal(score.data, np.zeros((1, 1, 1)))
    expected_query_grad = upstream * _SCALE * np.array([[[3.0, -3.0]]])
    expected_key_grad = (
        (np.array([[[ _EXTREME, _EXTREME ]]]) * _SCALE).transpose(0, 2, 1)
        * upstream
    )
    np.testing.assert_allclose(query.grad, expected_query_grad, rtol=2e-15, atol=0.0)
    np.testing.assert_allclose(key_t.grad, expected_key_grad, rtol=2e-15, atol=0.0)


def test_numpy_exact_cancellation_recovers_same_zero_score():
    query = np.array([[[ _EXTREME, _EXTREME ]]])
    key_t = np.array([[[3.0], [-3.0]]])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        score = _scaled_dot_product_scores_np(query, key_t, _SCALE)

    np.testing.assert_array_equal(score, np.zeros((1, 1, 1)))


def test_exact_recovery_supports_broadcast_batches_and_unbroadcasts_vjp():
    query = Tensor(np.array([[[ _EXTREME, _EXTREME ]]]), requires_grad=True)
    key_t = Tensor(
        np.array(
            [
                [[3.0], [-3.0]],
                [[-3.0], [3.0]],
            ]
        ),
        requires_grad=True,
    )
    upstream = np.array([[[4e-308]], [[-2e-308]]])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        score = _scaled_dot_product_scores(query, key_t, _SCALE)
        score_np = _scaled_dot_product_scores_np(query.data, key_t.data, _SCALE)
        score.backward(upstream)

    np.testing.assert_array_equal(score.data, np.zeros((2, 1, 1)))
    np.testing.assert_array_equal(score_np, np.zeros((2, 1, 1)))

    expected_query_grad = np.array(
        [[[18.0 * _SCALE * 1e-308, -18.0 * _SCALE * 1e-308]]]
    )
    scaled_query = _EXTREME * _SCALE
    expected_key_grad = np.array(
        [
            [[scaled_query * 4e-308], [scaled_query * 4e-308]],
            [[scaled_query * -2e-308], [scaled_query * -2e-308]],
        ]
    )
    np.testing.assert_allclose(query.grad, expected_query_grad, rtol=2e-15, atol=0.0)
    np.testing.assert_allclose(key_t.grad, expected_key_grad, rtol=2e-15, atol=0.0)


def test_genuinely_unrepresentable_scaled_dot_remains_nonfinite():
    query_data = np.array([[[ _EXTREME, _EXTREME ]]])
    key_data = np.array([[[3.0], [3.0]]])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        score = _scaled_dot_product_scores(
            Tensor(query_data), Tensor(key_data), _SCALE
        )
        score_np = _scaled_dot_product_scores_np(query_data, key_data, _SCALE)

    assert np.isposinf(score.data).all()
    assert np.isposinf(score_np).all()
