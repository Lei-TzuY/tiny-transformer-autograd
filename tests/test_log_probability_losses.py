"""Stable log-softmax, NLL, and label smoothing regression coverage."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine import (
    Tensor,
    cross_entropy,
    gradcheck,
    label_smoothed_cross_entropy,
    log_softmax,
    nll_loss,
    no_grad,
)


def _reference_log_softmax(data, axis=-1):
    row_max = np.max(data, axis=axis, keepdims=True)
    shifted = data - row_max
    return shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))


def test_log_softmax_matches_stable_reference_on_arbitrary_axis():
    data = np.array(
        [
            [[1.0, -2.0], [3.0, 0.5], [-4.0, 2.0]],
            [[-1.5, 4.0], [0.0, 2.5], [5.0, -3.0]],
        ]
    )
    x = Tensor(data, requires_grad=True)

    result = log_softmax(x, axis=1)

    np.testing.assert_allclose(result.data, _reference_log_softmax(data, axis=1))
    np.testing.assert_allclose(np.exp(result.data).sum(axis=1), np.ones((2, 2)))


def test_log_softmax_backward_matches_closed_form_vjp():
    data = np.array([[1.0, -0.5, 2.0], [3.0, 0.25, -4.0]])
    seed = np.array([[2.0, -3.0, 5.0], [7.0, 11.0, -13.0]])
    x = Tensor(data, requires_grad=True)
    output = log_softmax(x)

    output.backward(seed)

    probabilities = np.exp(_reference_log_softmax(data))
    expected = seed - probabilities * seed.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(x.grad, expected)


def test_log_softmax_accepts_negative_infinity_masks():
    x = Tensor([[2.0, -np.inf, 1.0]], requires_grad=True)

    result = log_softmax(x)

    assert np.isneginf(result.data[0, 1])
    np.testing.assert_allclose(np.exp(result.data), [[0.7310585786300049, 0.0, 0.2689414213699951]])


def test_log_softmax_handles_opposite_sign_extreme_finite_logits_warning_free():
    x = Tensor([[1e308, -1e308]], requires_grad=True)

    with np.errstate(all="raise"):
        result = log_softmax(x)
        result.backward(np.array([[3.0, -5.0]]))

    np.testing.assert_array_equal(np.exp(result.data), [[1.0, 0.0]])
    assert result.data[0, 0] == 0.0
    assert np.isneginf(result.data[0, 1])
    np.testing.assert_allclose(x.grad, [[5.0, -5.0]])


@pytest.mark.parametrize(
    "data, message",
    [
        ([[0.0, np.nan]], "NaN or \\+inf"),
        ([[0.0, np.inf]], "NaN or \\+inf"),
        ([[-np.inf, -np.inf]], "at least one finite"),
    ],
)
def test_log_softmax_rejects_invalid_probability_slices(data, message):
    with pytest.raises(ValueError, match=message):
        log_softmax(Tensor(data))


def test_log_softmax_rejects_invalid_axes_and_scalar_input():
    with pytest.raises(TypeError, match="axis must be an integer"):
        log_softmax(Tensor([1.0, 2.0]), axis=True)
    with pytest.raises(ValueError, match="out of bounds"):
        log_softmax(Tensor([1.0, 2.0]), axis=2)
    with pytest.raises(ValueError, match="at least one dimension"):
        log_softmax(Tensor(1.0))
    with pytest.raises(ValueError, match="normalization axis must be non-empty"):
        log_softmax(Tensor(np.empty((2, 0))))


def test_nll_loss_mean_value_and_gradient():
    log_probs_data = np.log(
        np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]], dtype=np.float64)
    )
    log_probs = Tensor(log_probs_data, requires_grad=True)
    targets = np.array([0, 2])

    loss = nll_loss(log_probs, targets)
    loss.backward()

    expected_loss = -0.5 * (np.log(0.7) + np.log(0.6))
    np.testing.assert_allclose(loss.data, expected_loss)
    expected_grad = np.zeros_like(log_probs_data)
    expected_grad[0, 0] = -0.5
    expected_grad[1, 2] = -0.5
    np.testing.assert_array_equal(log_probs.grad, expected_grad)


def test_nll_loss_none_and_sum_reductions():
    data = np.log(np.array([[0.8, 0.2], [0.25, 0.75]]))
    targets = np.array([1, 0])

    none_input = Tensor(data, requires_grad=True)
    none_loss = nll_loss(none_input, targets, reduction="none")
    np.testing.assert_allclose(none_loss.data, [-np.log(0.2), -np.log(0.25)])
    none_loss.backward(np.array([2.0, 3.0]))
    np.testing.assert_array_equal(none_input.grad, [[0.0, -2.0], [-3.0, 0.0]])

    sum_input = Tensor(data, requires_grad=True)
    sum_loss = nll_loss(sum_input, targets, reduction="sum")
    np.testing.assert_allclose(sum_loss.data, -np.log(0.2) - np.log(0.25))
    sum_loss.backward(np.array(4.0))
    np.testing.assert_array_equal(sum_input.grad, [[0.0, -4.0], [-4.0, 0.0]])


def test_nll_loss_ignore_index_excludes_values_and_gradients():
    data = np.array(
        [
            [np.log(0.6), np.log(0.4)],
            [np.nan, np.inf],
            [np.log(0.3), np.log(0.7)],
        ]
    )
    log_probs = Tensor(data, requires_grad=True)
    targets = np.array([0, -100, 1])

    loss = nll_loss(log_probs, targets, ignore_index=-100, reduction="none")
    np.testing.assert_allclose(loss.data, [-np.log(0.6), 0.0, -np.log(0.7)])
    loss.backward(np.array([2.0, 99.0, 3.0]))

    expected = np.zeros_like(data)
    expected[0, 0] = -2.0
    expected[2, 1] = -3.0
    np.testing.assert_array_equal(log_probs.grad, expected)


def test_nll_loss_snapshots_mutable_targets_for_backward():
    data = np.log(np.array([[0.6, 0.4], [0.3, 0.7]]))
    targets = np.array([0, 1])
    log_probs = Tensor(data, requires_grad=True)
    loss = nll_loss(log_probs, targets, reduction="sum")

    targets[:] = [1, 0]
    loss.backward()

    np.testing.assert_array_equal(log_probs.grad, [[-1.0, 0.0], [0.0, -1.0]])


def test_nll_loss_stable_mean_recovers_sum_before_divide_overflow():
    log_probs = Tensor([[-1e308], [-1e308]], requires_grad=True)

    with np.errstate(all="raise"):
        loss = nll_loss(log_probs, np.array([0, 0]), reduction="mean")
        loss.backward()

    assert loss.data == 1e308
    np.testing.assert_array_equal(log_probs.grad, [[-0.5], [-0.5]])


def test_label_smoothing_zero_matches_existing_cross_entropy():
    logits_data = np.array([[2.0, -1.0, 0.5], [0.25, 3.0, -2.0]])
    targets = np.array([2, 1])

    existing_logits = Tensor(logits_data, requires_grad=True)
    existing = cross_entropy(existing_logits, targets)
    existing.backward()

    smoothed_logits = Tensor(logits_data, requires_grad=True)
    smoothed = label_smoothed_cross_entropy(
        smoothed_logits, targets, smoothing=0.0
    )
    smoothed.backward()

    np.testing.assert_allclose(smoothed.data, existing.data, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(smoothed_logits.grad, existing_logits.grad, rtol=1e-14, atol=1e-14)


def test_label_smoothed_cross_entropy_matches_manual_distribution_gradient():
    logits_data = np.array([[2.0, 0.0, -1.0], [-0.5, 1.5, 0.25]])
    targets = np.array([0, 2])
    smoothing = 0.2
    logits = Tensor(logits_data, requires_grad=True)

    loss = label_smoothed_cross_entropy(logits, targets, smoothing=smoothing)
    loss.backward()

    log_probs = _reference_log_softmax(logits_data)
    probabilities = np.exp(log_probs)
    target_distribution = np.full_like(probabilities, smoothing / 3.0)
    target_distribution[np.arange(2), targets] += 1.0 - smoothing
    expected_loss = -np.mean(np.sum(target_distribution * log_probs, axis=-1))
    expected_grad = (probabilities - target_distribution) / 2.0

    np.testing.assert_allclose(loss.data, expected_loss)
    np.testing.assert_allclose(logits.grad, expected_grad)


def test_label_smoothed_cross_entropy_uniform_target_at_smoothing_one():
    logits_data = np.array([[4.0, 1.0, -2.0]])
    logits = Tensor(logits_data, requires_grad=True)

    loss = label_smoothed_cross_entropy(
        logits, np.array([0]), smoothing=1.0, reduction="sum"
    )
    loss.backward()

    log_probs = _reference_log_softmax(logits_data)
    probabilities = np.exp(log_probs)
    np.testing.assert_allclose(loss.data, -np.mean(log_probs))
    np.testing.assert_allclose(logits.grad, probabilities - 1.0 / 3.0)


def test_label_smoothed_cross_entropy_passes_gradcheck():
    logits = Tensor([[0.2, -0.3, 1.1], [1.5, 0.7, -0.9]])
    targets = np.array([2, 0])

    assert gradcheck(
        lambda value: label_smoothed_cross_entropy(
            value, targets, smoothing=0.15, reduction="mean"
        ),
        logits,
        eps=1e-6,
        atol=1e-6,
        rtol=1e-5,
    )


def test_label_smoothed_cross_entropy_ignore_index_has_zero_gradient():
    logits = Tensor(
        [[2.0, 0.0, -1.0], [100.0, -100.0, 0.0], [0.5, 1.0, -0.5]],
        requires_grad=True,
    )
    targets = np.array([0, -100, 1])

    loss = label_smoothed_cross_entropy(
        logits,
        targets,
        smoothing=0.1,
        ignore_index=-100,
        reduction="mean",
    )
    loss.backward()

    np.testing.assert_array_equal(logits.grad[1], np.zeros(3))
    assert np.isfinite(loss.data)


def test_probability_losses_respect_no_grad_mode():
    x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)

    with no_grad():
        log_probs = log_softmax(x)
        loss = nll_loss(log_probs, np.array([2]))

    assert not log_probs.requires_grad
    assert not loss.requires_grad
    assert log_probs._children == ()
    assert loss._children == ()


@pytest.mark.parametrize("reduction", [None, "bad", 3])
def test_nll_loss_validates_reduction(reduction):
    log_probs = Tensor(np.log([[0.5, 0.5]]))
    error = TypeError if not isinstance(reduction, str) else ValueError
    with pytest.raises(error, match="reduction"):
        nll_loss(log_probs, np.array([0]), reduction=reduction)


@pytest.mark.parametrize("smoothing", [True, -0.1, 1.1, np.inf, "0.1"])
def test_label_smoothed_cross_entropy_validates_smoothing(smoothing):
    error = TypeError if isinstance(smoothing, (bool, str)) else ValueError
    with pytest.raises(error, match="smoothing"):
        label_smoothed_cross_entropy(
            Tensor([[1.0, 2.0]]), np.array([0]), smoothing=smoothing
        )


def test_nll_loss_validates_targets_and_ignore_index():
    log_probs = Tensor(np.log([[0.5, 0.5]]))
    with pytest.raises(TypeError, match="integers"):
        nll_loss(log_probs, np.array([0.0]))
    with pytest.raises(ValueError, match="shape mismatch"):
        nll_loss(log_probs, np.array([[0]]))
    with pytest.raises(ValueError, match=r"\[0, 2\)"):
        nll_loss(log_probs, np.array([2]))
    with pytest.raises(TypeError, match="ignore_index"):
        nll_loss(log_probs, np.array([0]), ignore_index=True)
    with pytest.raises(ValueError, match="no scored target"):
        nll_loss(log_probs, np.array([-100]), ignore_index=-100)
