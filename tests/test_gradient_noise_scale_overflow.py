import numpy as np

from engine.gradient_noise_scale import GradientNoiseScaleEstimator
from engine.tensor import Tensor


def test_nonzero_tiny_signal_reports_noise_scale_overflow_not_infinity():
    limit = np.finfo(np.float64).max
    tiny = np.nextafter(np.float64(0.0), np.float64(1.0))
    p = Tensor([0.0, 0.0], requires_grad=True)
    estimator = GradientNoiseScaleEstimator(p, batch_size=1)

    p.grad = np.array([limit, tiny])
    estimator.capture()
    p.grad = np.array([-limit, tiny])
    estimator.capture()

    report = estimator.report()
    assert report["mean_gradient_l2"] == tiny
    assert report["noise_scale"] is None
    assert report["noise_scale_defined"] is False
    assert report["noise_scale_infinite"] is False
    assert report["noise_scale_overflow"] is True
    assert report["noise_scale_reason"] == "overflow"
