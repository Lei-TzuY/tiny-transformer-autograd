import json

import numpy as np
import pytest

from engine.gradient_noise_scale import GradientNoiseScaleEstimator
from engine.tensor import Tensor


def _capture_scalar(estimator, parameter, value):
    parameter.grad = None if value is None else np.array(value, dtype=np.float64)
    estimator.capture()


def test_known_scalar_noise_scale_matches_unbiased_sample_covariance():
    p = Tensor(0.0, requires_grad=True)
    estimator = GradientNoiseScaleEstimator(p, batch_size=2)

    _capture_scalar(estimator, p, 1.0)
    _capture_scalar(estimator, p, 3.0)
    report = estimator.report()

    assert estimator.mean_gradients()[0].item() == 2.0
    assert report["mean_gradient_l2"] == 2.0
    assert report["gradient_variance_trace"] == 2.0
    assert report["noise_scale"] == 1.0
    assert report["noise_scale_defined"] is True
    assert report["noise_scale_reason"] == "ok"


def test_vector_noise_scale_uses_global_parameter_geometry():
    p = Tensor([0.0, 0.0], requires_grad=True)
    estimator = GradientNoiseScaleEstimator(p, batch_size=4)
    for gradient in ([1.0, 2.0], [3.0, 4.0], [5.0, 6.0]):
        p.grad = np.array(gradient, dtype=np.float64)
        estimator.capture()

    report = estimator.report()
    np.testing.assert_allclose(estimator.mean_gradients()[0], [3.0, 4.0])
    assert report["mean_gradient_l2"] == 5.0
    assert report["gradient_variance_trace"] == 8.0
    assert report["noise_scale"] == pytest.approx(32.0 / 25.0)


def test_missing_gradient_is_an_exact_zero_sample():
    p = Tensor(0.0, requires_grad=True)
    estimator = GradientNoiseScaleEstimator(p, batch_size=1)
    _capture_scalar(estimator, p, None)
    _capture_scalar(estimator, p, 2.0)

    report = estimator.report()
    assert estimator.mean_gradients()[0].item() == 1.0
    assert report["gradient_variance_trace"] == 2.0
    assert report["noise_scale"] == 2.0


def test_identical_gradients_have_zero_noise_scale():
    p = Tensor([2.0, -3.0], requires_grad=True)
    estimator = GradientNoiseScaleEstimator(p, batch_size=16)
    for _ in range(3):
        p.grad = np.array([2.0, -3.0])
        estimator.capture()

    report = estimator.report()
    assert report["gradient_variance_trace"] == 0.0
    assert report["noise_scale"] == 0.0
    assert report["noise_scale_defined"] is True
    assert report["noise_scale_reason"] == "ok"


def test_zero_signal_with_positive_noise_is_reported_as_infinite_without_json_inf():
    p = Tensor(0.0, requires_grad=True)
    estimator = GradientNoiseScaleEstimator(p, batch_size=8)
    _capture_scalar(estimator, p, 1.0)
    _capture_scalar(estimator, p, -1.0)

    report = estimator.report()
    assert report["mean_gradient_l2"] == 0.0
    assert report["gradient_variance_trace"] == 2.0
    assert report["noise_scale"] is None
    assert report["noise_scale_defined"] is False
    assert report["noise_scale_infinite"] is True
    assert report["noise_scale_reason"] == "zero_signal"
    json.dumps(report, allow_nan=False)


def test_all_zero_samples_have_undefined_zero_signal_and_noise():
    p = Tensor(0.0, requires_grad=True)
    estimator = GradientNoiseScaleEstimator(p, batch_size=8)
    _capture_scalar(estimator, p, None)
    _capture_scalar(estimator, p, 0.0)

    report = estimator.report()
    assert report["mean_gradient_l2"] == 0.0
    assert report["gradient_variance_trace"] == 0.0
    assert report["noise_scale"] is None
    assert report["noise_scale_infinite"] is False
    assert report["noise_scale_reason"] == "zero_signal_and_noise"


def test_one_sample_reports_mean_but_not_variance_or_noise_scale():
    p = Tensor(0.0, requires_grad=True)
    estimator = GradientNoiseScaleEstimator(p, batch_size=4)
    _capture_scalar(estimator, p, -3.0)

    report = estimator.report()
    assert report["mean_gradient_l2"] == 3.0
    assert report["gradient_variance_trace"] is None
    assert report["noise_scale"] is None
    assert report["noise_scale_reason"] == "insufficient_samples"


def test_opposite_float64_extremes_keep_variance_overflow_explicit():
    limit = np.finfo(np.float64).max
    p = Tensor(0.0, requires_grad=True)
    estimator = GradientNoiseScaleEstimator(p, batch_size=2)

    with np.errstate(all="raise"):
        _capture_scalar(estimator, p, limit)
        _capture_scalar(estimator, p, -limit)
        report = estimator.report()

    assert report["mean_gradient_l2"] == 0.0
    assert report["gradient_variance_trace"] is None
    assert report["gradient_variance_trace_overflow"] is True
    assert report["noise_scale_infinite"] is True
    json.dumps(report, allow_nan=False)


def test_same_sign_float64_maximum_is_warning_free_and_zero_noise():
    limit = np.finfo(np.float64).max
    p = Tensor(0.0, requires_grad=True)
    estimator = GradientNoiseScaleEstimator(p, batch_size=2)

    with np.errstate(all="raise"):
        _capture_scalar(estimator, p, limit)
        _capture_scalar(estimator, p, limit)
        report = estimator.report()

    assert report["mean_gradient_l2"] == limit
    assert report["mean_gradient_l2_overflow"] is False
    assert report["gradient_variance_trace"] == 0.0
    assert report["noise_scale"] == 0.0


def test_empty_report_and_empty_parameter_collection_are_supported():
    estimator = GradientNoiseScaleEstimator([], batch_size=1)
    empty = estimator.report()
    assert empty["sample_count"] == 0
    assert empty["noise_scale_reason"] == "empty"
    with pytest.raises(RuntimeError, match="no samples"):
        estimator.mean_gradients()

    estimator.capture()
    estimator.capture()
    report = estimator.report()
    assert estimator.mean_gradients() == ()
    assert report["parameter_count"] == 0
    assert report["noise_scale_reason"] == "zero_signal_and_noise"


def test_samples_and_mean_results_are_independent_copies():
    p = Tensor([1.0, 2.0], requires_grad=True)
    estimator = GradientNoiseScaleEstimator(p, batch_size=1)
    p.grad = np.array([3.0, 4.0])
    estimator.capture()

    samples = estimator.sample_gradients()
    means = estimator.mean_gradients()
    samples[0][0][0] = 99.0
    means[0][1] = 88.0

    np.testing.assert_array_equal(estimator.sample_gradients()[0][0], [3.0, 4.0])
    np.testing.assert_array_equal(estimator.mean_gradients()[0], [3.0, 4.0])


def test_capture_and_report_preserve_model_and_rng_state():
    p = Tensor([1.0, -2.0], requires_grad=True)
    p.grad[...] = [5.0, 7.0]
    grad_ref = p.grad
    data = p.data.copy()
    version = p._version
    rng = np.random.get_state()
    estimator = GradientNoiseScaleEstimator(p, batch_size=2)

    estimator.capture()
    estimator.report()

    np.testing.assert_array_equal(p.data, data)
    assert p._version == version
    assert p.grad is grad_ref
    np.testing.assert_array_equal(p.grad, [5.0, 7.0])
    after = np.random.get_state()
    assert rng[0] == after[0]
    np.testing.assert_array_equal(rng[1], after[1])
    assert rng[2:] == after[2:]


def test_float32_gradients_are_normalized_to_independent_float64_samples():
    p = Tensor([0.0, 0.0], requires_grad=True)
    estimator = GradientNoiseScaleEstimator(p, batch_size=2)
    source = np.array([1.5, -2.5], dtype=np.float32)
    p.grad = source
    estimator.capture()
    source[...] = 0.0

    captured = estimator.sample_gradients()[0][0]
    assert captured.dtype == np.float64
    np.testing.assert_array_equal(captured, [1.5, -2.5])


def test_reset_forgets_samples_without_touching_live_gradients():
    p = Tensor(0.0, requires_grad=True)
    estimator = GradientNoiseScaleEstimator(p, batch_size=1)
    _capture_scalar(estimator, p, 2.0)
    grad_ref = p.grad

    assert estimator.reset() is estimator
    assert estimator.sample_count == 0
    assert p.grad is grad_ref
    assert p.grad.item() == 2.0
