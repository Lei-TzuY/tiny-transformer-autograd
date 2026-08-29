import numpy as np
import pytest

from engine.gradient_noise_scale import GradientNoiseScaleEstimator
from engine.tensor import Tensor


def test_batch_size_multiplies_batch_gradient_covariance_into_noise_scale():
    p = Tensor(0.0, requires_grad=True)
    small_batch = GradientNoiseScaleEstimator(p, batch_size=2)
    large_batch = GradientNoiseScaleEstimator(p, batch_size=8)

    for value in (1.0, 3.0):
        p.grad = np.array(value)
        small_batch.capture()
        large_batch.capture()

    assert small_batch.report()["gradient_variance_trace"] == 2.0
    assert large_batch.report()["gradient_variance_trace"] == 2.0
    assert small_batch.report()["noise_scale"] == pytest.approx(1.0)
    assert large_batch.report()["noise_scale"] == pytest.approx(4.0)
