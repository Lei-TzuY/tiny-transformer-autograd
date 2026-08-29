import json

import numpy as np

from engine.gradient_noise_scale import GradientNoiseScaleEstimator
from engine.tensor import Tensor


def test_every_report_state_is_strict_json_safe():
    p = Tensor(0.0, requires_grad=True)
    estimator = GradientNoiseScaleEstimator(p, batch_size=4)

    json.dumps(estimator.report(), allow_nan=False)

    p.grad = np.array(1.0)
    estimator.capture()
    json.dumps(estimator.report(), allow_nan=False)

    p.grad = np.array(-1.0)
    estimator.capture()
    json.dumps(estimator.report(), allow_nan=False)

    limit = np.finfo(np.float64).max
    estimator.reset()
    p.grad = np.array(limit)
    estimator.capture()
    p.grad = np.array(-limit)
    estimator.capture()
    json.dumps(estimator.report(), allow_nan=False)
