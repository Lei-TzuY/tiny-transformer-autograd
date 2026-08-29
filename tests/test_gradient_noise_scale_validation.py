import sys

import numpy as np
import pytest

from engine.gradient_noise_scale import GradientNoiseScaleEstimator
from engine.tensor import Tensor


def test_batch_size_validation_precedes_parameter_generator_consumption():
    consumed = []

    def parameters():
        consumed.append(True)
        yield Tensor([1.0], requires_grad=True)

    with pytest.raises(TypeError, match="positive integer"):
        GradientNoiseScaleEstimator(parameters(), batch_size=True)
    assert consumed == []

    with pytest.raises(ValueError, match="positive integer"):
        GradientNoiseScaleEstimator(parameters(), batch_size=0)
    assert consumed == []

    with pytest.raises(ValueError, match="sys.maxsize"):
        GradientNoiseScaleEstimator(parameters(), batch_size=sys.maxsize + 1)
    assert consumed == []


def test_numpy_integer_batch_size_is_accepted():
    estimator = GradientNoiseScaleEstimator([], batch_size=np.int64(7))
    assert estimator.batch_size == 7


def test_parameter_generator_is_materialized_once():
    p = Tensor([1.0], requires_grad=True)
    calls = []

    def parameters():
        calls.append("start")
        yield p
        calls.append("end")

    estimator = GradientNoiseScaleEstimator(parameters(), batch_size=1)
    assert calls == ["start", "end"]
    assert estimator.parameter_count == 1


def test_parameter_collection_rejects_non_tensor_duplicates_and_frozen_tensors():
    p = Tensor([1.0], requires_grad=True)
    with pytest.raises(TypeError, match="parameter 0"):
        GradientNoiseScaleEstimator([object()], batch_size=1)
    with pytest.raises(ValueError, match="duplicate"):
        GradientNoiseScaleEstimator([p, p], batch_size=1)
    with pytest.raises(ValueError, match="must require gradients"):
        GradientNoiseScaleEstimator([Tensor([1.0])], batch_size=1)


def test_capture_rejects_parameter_shape_and_trainability_drift():
    p = Tensor([1.0, 2.0], requires_grad=True)
    estimator = GradientNoiseScaleEstimator(p, batch_size=1)

    p.data = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError, match="shape changed"):
        estimator.capture()
    assert estimator.sample_count == 0

    p = Tensor([1.0], requires_grad=True)
    estimator = GradientNoiseScaleEstimator(p, batch_size=1)
    p.requires_grad = False
    with pytest.raises(ValueError, match="continue to require"):
        estimator.capture()
    assert estimator.sample_count == 0


def test_capture_gradient_validation_is_transactional_across_parameters():
    p1 = Tensor([1.0], requires_grad=True)
    p2 = Tensor([2.0], requires_grad=True)
    estimator = GradientNoiseScaleEstimator([p1, p2], batch_size=1)
    p1.grad = np.array([3.0])
    p2.grad = np.array([4.0])
    estimator.capture()
    before = estimator.sample_gradients()

    p1.grad = np.array([9.0])
    p2.grad = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="gradient 1 shape"):
        estimator.capture()

    assert estimator.sample_count == 1
    after = estimator.sample_gradients()
    np.testing.assert_array_equal(after[0][0], before[0][0])
    np.testing.assert_array_equal(after[0][1], before[0][1])


def test_gradient_type_dtype_shape_and_finiteness_validation():
    p = Tensor([0.0, 0.0], requires_grad=True)
    estimator = GradientNoiseScaleEstimator(p, batch_size=1)

    bad_values = [
        ([1.0, 2.0], TypeError, "NumPy array"),
        (np.array([1, 2], dtype=np.int64), TypeError, "floating dtype"),
        (np.array([True, False]), TypeError, "floating dtype"),
        (np.array([1 + 2j, 3 + 4j]), TypeError, "floating dtype"),
        (np.array([np.nan, 0.0]), ValueError, "finite"),
        (np.array([np.inf, 0.0]), ValueError, "finite"),
        (np.array([-np.inf, 0.0]), ValueError, "finite"),
    ]
    for value, error, pattern in bad_values:
        p.grad = value
        with pytest.raises(error, match=pattern):
            estimator.capture()
        assert estimator.sample_count == 0


def test_none_gradient_captures_independent_exact_zero_array():
    p = Tensor([1.0, 2.0], requires_grad=True)
    estimator = GradientNoiseScaleEstimator(p, batch_size=1)
    p.grad = None
    estimator.capture()

    captured = estimator.sample_gradients()[0][0]
    assert captured.dtype == np.float64
    np.testing.assert_array_equal(captured, [0.0, 0.0])


def test_extended_precision_gradient_outside_float64_is_rejected_when_available():
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("platform longdouble is not wider than float64")

    p = Tensor([0.0], requires_grad=True)
    estimator = GradientNoiseScaleEstimator(p, batch_size=1)
    p.grad = np.array([np.finfo(np.float64).max], dtype=np.longdouble) * np.longdouble(2)

    with pytest.raises(ValueError, match="fit float64"):
        estimator.capture()
    assert estimator.sample_count == 0


def test_smallest_subnormal_opposite_samples_preserve_variance_underflow_signal():
    tiny = np.nextafter(np.float64(0.0), np.float64(1.0))
    p = Tensor(0.0, requires_grad=True)
    estimator = GradientNoiseScaleEstimator(p, batch_size=2)

    with np.errstate(all="raise"):
        p.grad = np.array(tiny)
        estimator.capture()
        p.grad = np.array(-tiny)
        estimator.capture()
        report = estimator.report()

    assert report["gradient_variance_trace"] == 0.0
    assert report["gradient_variance_trace_underflow"] is True
    assert report["noise_scale_infinite"] is True


def test_tiny_identical_signal_stays_representable_with_zero_noise():
    tiny = np.nextafter(np.float64(0.0), np.float64(1.0))
    p = Tensor(0.0, requires_grad=True)
    estimator = GradientNoiseScaleEstimator(p, batch_size=2)
    p.grad = np.array(tiny)
    estimator.capture()
    estimator.capture()

    report = estimator.report()
    assert report["mean_gradient_l2"] == tiny
    assert report["gradient_variance_trace"] == 0.0
    assert report["noise_scale"] == 0.0


def test_per_parameter_scaling_keeps_unrelated_extreme_tensor_warning_free():
    limit = np.finfo(np.float64).max
    large = Tensor([0.0], requires_grad=True)
    small = Tensor([0.0], requires_grad=True)
    estimator = GradientNoiseScaleEstimator([large, small], batch_size=1)

    with np.errstate(all="raise"):
        large.grad = np.array([limit])
        small.grad = np.array([1.0])
        estimator.capture()
        large.grad = np.array([limit])
        small.grad = np.array([-1.0])
        estimator.capture()
        report = estimator.report()

    assert report["mean_gradient_l2"] == limit
    assert report["mean_gradient_l2_overflow"] is False
    assert report["gradient_variance_trace"] == 2.0
    assert report["noise_scale"] == 0.0
    assert report["noise_scale_underflow"] is True
