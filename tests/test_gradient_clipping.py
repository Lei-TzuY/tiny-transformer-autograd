"""Regression tests for stable global gradient norm clipping."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.tensor import Tensor
from train import clip_grad_norm_


def _parameter(gradient):
    parameter = Tensor(np.zeros_like(np.asarray(gradient, dtype=np.float64)), requires_grad=True)
    parameter.grad[:] = np.asarray(gradient, dtype=np.float64)
    return parameter


def test_clips_generator_parameters_instead_of_consuming_them_during_norm():
    first = _parameter([3.0, 0.0])
    second = _parameter([0.0, 4.0])

    total = clip_grad_norm_((value for value in (first, second)), max_norm=1.0)

    assert total == pytest.approx(5.0)
    np.testing.assert_allclose(first.grad, [0.6, 0.0], atol=1e-15, rtol=0.0)
    np.testing.assert_allclose(second.grad, [0.0, 0.8], atol=1e-15, rtol=0.0)


def test_huge_finite_gradients_clip_without_square_overflow():
    parameter = _parameter([1e308, -1e308])

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        total = clip_grad_norm_([parameter], max_norm=1.0)

    assert np.isfinite(total)
    assert total == pytest.approx(np.sqrt(2.0) * 1e308, rel=1e-15)
    assert np.linalg.norm(parameter.grad) == pytest.approx(1.0, rel=1e-15)
    np.testing.assert_allclose(
        parameter.grad,
        [1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0)],
        rtol=1e-15,
        atol=0.0,
    )


def test_unrepresentable_norm_still_uses_a_finite_clipping_ratio():
    largest = np.finfo(np.float64).max
    parameter = _parameter([largest, largest, largest])

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        total = clip_grad_norm_([parameter], max_norm=1.0)

    assert np.isposinf(total)
    assert np.isfinite(parameter.grad).all()
    assert np.linalg.norm(parameter.grad) == pytest.approx(1.0, rel=2e-15)
    np.testing.assert_allclose(
        parameter.grad,
        np.full(3, 1.0 / np.sqrt(3.0)),
        rtol=2e-15,
        atol=0.0,
    )


def test_zero_threshold_measures_huge_norm_without_modifying_gradients():
    parameter = _parameter([1e308, -1e308])
    before = parameter.grad.copy()

    with np.errstate(over="raise", invalid="raise"):
        total = clip_grad_norm_([parameter], max_norm=0.0)

    assert np.isfinite(total)
    np.testing.assert_array_equal(parameter.grad, before)


@pytest.mark.parametrize("bad_max_norm", [-1.0, np.nan, np.inf, -np.inf, True, np.bool_(False), "1.0"])
def test_rejects_invalid_max_norm_before_mutating_gradients(bad_max_norm):
    parameter = _parameter([3.0, 4.0])
    before = parameter.grad.copy()

    with pytest.raises((TypeError, ValueError), match="max_norm"):
        clip_grad_norm_([parameter], max_norm=bad_max_norm)

    np.testing.assert_array_equal(parameter.grad, before)


def test_nonfinite_late_gradient_fails_transactionally():
    first = _parameter([3.0, 4.0])
    second = _parameter([1.0, np.nan])
    first_before = first.grad.copy()
    second_before = second.grad.copy()

    with pytest.raises(ValueError, match="gradient 1.*finite"):
        clip_grad_norm_([first, second], max_norm=1.0)

    np.testing.assert_array_equal(first.grad, first_before)
    np.testing.assert_array_equal(second.grad, second_before)


def test_ignores_missing_gradients_and_empty_parameter_sets():
    missing = _parameter([1.0, 2.0])
    missing.grad = None

    assert clip_grad_norm_([missing], max_norm=1.0) == 0.0
    assert clip_grad_norm_([], max_norm=1.0) == 0.0


def test_requires_parameter_iterable():
    with pytest.raises(TypeError, match="params must be an iterable"):
        clip_grad_norm_(None, max_norm=1.0)
