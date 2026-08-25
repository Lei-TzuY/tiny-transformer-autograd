"""Softmax and cross-entropy must fail loudly on invalid scored logits."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.ops import cross_entropy, softmax
from engine.tensor import Tensor


@pytest.mark.parametrize(
    "bad_row",
    [
        [0.0, np.inf, -1.0],
        [0.0, np.nan, -1.0],
    ],
)
def test_softmax_rejects_nan_and_positive_infinity(bad_row):
    x = Tensor(bad_row, requires_grad=True)
    before = x.grad.copy()
    with pytest.raises(ValueError, match=r"NaN|\+inf"):
        softmax(x)
    np.testing.assert_array_equal(x.grad, before)


def test_softmax_still_allows_negative_infinity_masks():
    x = Tensor([[0.0, -np.inf, 1.0], [-np.inf, -np.inf, -np.inf]], requires_grad=True)
    out = softmax(x)

    expected_first = np.exp([0.0, -np.inf, 1.0])
    expected_first /= expected_first.sum()
    np.testing.assert_allclose(out.data[0], expected_first)
    np.testing.assert_array_equal(out.data[1], np.zeros(3))

    out.backward(np.ones_like(out.data))
    np.testing.assert_allclose(x.grad, np.zeros_like(x.data), atol=1e-15)


@pytest.mark.parametrize("bad_value", [np.inf, np.nan])
def test_cross_entropy_rejects_nonfinite_scored_logits(bad_value):
    logits = Tensor([[0.0, bad_value, -1.0]], requires_grad=True)
    before = logits.grad.copy()
    with pytest.raises(ValueError, match=r"NaN|\+inf"):
        cross_entropy(logits, np.array([0]))
    np.testing.assert_array_equal(logits.grad, before)


def test_cross_entropy_ignores_nonfinite_values_in_ignored_rows():
    logits = Tensor(
        [
            [0.0, 1.0, -1.0],
            [np.nan, np.inf, -np.inf],
        ],
        requires_grad=True,
    )
    loss = cross_entropy(logits, np.array([1, -1]), ignore_index=-1)

    expected = np.log(np.exp(-1.0) + 1.0 + np.exp(-2.0))
    np.testing.assert_allclose(loss.data, expected)
    loss.backward()
    np.testing.assert_array_equal(logits.grad[1], np.zeros(3))
    assert np.isfinite(logits.grad[0]).all()


def test_cross_entropy_allows_negative_infinity_for_impossible_non_targets():
    logits = Tensor([[0.0, -np.inf, 1.0]], requires_grad=True)
    loss = cross_entropy(logits, np.array([2]))
    expected = np.log1p(np.exp(-1.0))
    np.testing.assert_allclose(loss.data, expected)
    loss.backward()
    assert np.isfinite(logits.grad).all()
