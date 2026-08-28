"""Regression tests for reusable gradient norm and clipping helpers."""

import numpy as np
import pytest

from engine.grad_utils import clip_grad_norm_, global_grad_norm
from engine.tensor import Tensor


def _parameter(gradient, *, dtype=np.float64):
    values = np.asarray(gradient, dtype=dtype)
    parameter = Tensor(np.zeros(values.shape, dtype=np.float64), requires_grad=True)
    parameter.grad = np.array(values, dtype=dtype, copy=True)
    return parameter


def test_global_norm_accepts_generator_without_mutating_gradients():
    first = _parameter([3.0, 0.0])
    second = _parameter([0.0, 4.0])
    first_before = first.grad.copy()
    second_before = second.grad.copy()

    total = global_grad_norm(value for value in (first, second))

    assert total == pytest.approx(5.0)
    np.testing.assert_array_equal(first.grad, first_before)
    np.testing.assert_array_equal(second.grad, second_before)


def test_clip_norm_matches_global_l2_scale():
    first = _parameter([3.0, 0.0])
    second = _parameter([0.0, 4.0])

    total = clip_grad_norm_((value for value in (first, second)), max_norm=1.0)

    assert total == pytest.approx(5.0)
    np.testing.assert_allclose(first.grad, [0.6, 0.0], atol=1e-15, rtol=0.0)
    np.testing.assert_allclose(second.grad, [0.0, 0.8], atol=1e-15, rtol=0.0)


def test_huge_finite_gradients_do_not_overflow_during_norm():
    parameter = _parameter([1e308, -1e308])

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        total = global_grad_norm([parameter])

    assert np.isfinite(total)
    assert total == pytest.approx(np.sqrt(2.0) * 1e308, rel=1e-15)


def test_unrepresentable_norm_still_clips_with_finite_ratio():
    largest = np.finfo(np.float64).max
    parameter = _parameter([largest, largest, largest])

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        total = clip_grad_norm_([parameter], max_norm=1.0)

    assert np.isposinf(total)
    assert np.isfinite(parameter.grad).all()
    assert np.linalg.norm(parameter.grad) == pytest.approx(1.0, rel=2e-15)


def test_zero_threshold_measures_without_mutating():
    parameter = _parameter([1e308, -1e308])
    before = parameter.grad.copy()

    total = clip_grad_norm_([parameter], max_norm=0.0)

    assert np.isfinite(total)
    np.testing.assert_array_equal(parameter.grad, before)


@pytest.mark.parametrize("bad_max_norm", [10**400, -(10**400), np.inf, np.nan])
def test_invalid_max_norm_fails_before_consuming_parameters(bad_max_norm):
    class MustNotIterate:
        def __iter__(self):
            raise AssertionError("parameters were consumed before max_norm validation")

    with pytest.raises(ValueError, match="max_norm must be finite"):
        clip_grad_norm_(MustNotIterate(), max_norm=bad_max_norm)


@pytest.mark.parametrize("bad_max_norm", [True, np.bool_(False), "1.0", None])
def test_non_real_max_norm_has_explicit_type_error(bad_max_norm):
    with pytest.raises(TypeError, match="max_norm must be a real number"):
        clip_grad_norm_([], max_norm=bad_max_norm)


def test_duplicate_parameter_references_fail_without_mutation():
    parameter = _parameter([3.0, 4.0])
    before = parameter.grad.copy()

    with pytest.raises(ValueError, match="duplicate Tensor references"):
        clip_grad_norm_([parameter, parameter], max_norm=1.0)

    np.testing.assert_array_equal(parameter.grad, before)


def test_late_nonfinite_gradient_fails_transactionally():
    first = _parameter([3.0, 4.0])
    second = _parameter([1.0, np.nan])
    first_before = first.grad.copy()
    second_before = second.grad.copy()

    with pytest.raises(ValueError, match="gradient 1.*finite"):
        clip_grad_norm_([first, second], max_norm=1.0)

    np.testing.assert_array_equal(first.grad, first_before)
    np.testing.assert_array_equal(second.grad, second_before)


def test_readonly_late_gradient_fails_before_any_clipping():
    first = _parameter([3.0, 4.0])
    second = _parameter([6.0, 8.0])
    second.grad.flags.writeable = False
    first_before = first.grad.copy()
    second_before = second.grad.copy()

    with pytest.raises(ValueError, match="gradient 1.*writeable"):
        clip_grad_norm_([first, second], max_norm=1.0)

    np.testing.assert_array_equal(first.grad, first_before)
    np.testing.assert_array_equal(second.grad, second_before)


def test_clipping_underflow_is_warning_neutral():
    parameter = _parameter([1e308, 1.0])

    with np.errstate(under="raise", over="raise", invalid="raise"):
        total = clip_grad_norm_([parameter], max_norm=np.finfo(np.float64).tiny)

    assert total == pytest.approx(1e308)
    assert parameter.grad[0] == pytest.approx(np.finfo(np.float64).tiny)
    assert parameter.grad[1] == 0.0


def test_gradient_shape_and_dtype_are_validated_before_mutation():
    valid = _parameter([3.0, 4.0])
    valid_before = valid.grad.copy()
    bad_shape = _parameter([1.0, 2.0])
    bad_shape.grad = np.zeros((1, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="gradient 1 shape mismatch"):
        clip_grad_norm_([valid, bad_shape], max_norm=1.0)
    np.testing.assert_array_equal(valid.grad, valid_before)

    bad_dtype = _parameter([1.0, 2.0])
    bad_dtype.grad = np.array([1, 2], dtype=np.int64)
    with pytest.raises(TypeError, match="gradient 0.*floating dtype"):
        global_grad_norm([bad_dtype])


def test_missing_gradients_empty_sets_and_float32_are_supported():
    missing = _parameter([1.0, 2.0])
    missing.grad = None
    float32_parameter = _parameter([3.0, 4.0], dtype=np.float32)

    assert global_grad_norm([]) == 0.0
    assert global_grad_norm([missing]) == 0.0
    total = clip_grad_norm_([float32_parameter], max_norm=1.0)
    assert total == pytest.approx(5.0)
    assert float32_parameter.grad.dtype == np.float32
    np.testing.assert_allclose(float32_parameter.grad, [0.6, 0.8], rtol=1e-6)


def test_requires_tensor_parameter_iterable():
    with pytest.raises(TypeError, match="iterable of Tensors"):
        global_grad_norm(None)
    with pytest.raises(TypeError, match="parameter 1 must be a Tensor"):
        global_grad_norm([_parameter([1.0]), object()])
