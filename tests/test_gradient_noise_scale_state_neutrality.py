import copy

import numpy as np
import pytest

from engine.gradient_noise_scale import GradientNoiseScaleEstimator
from engine.tensor import Tensor


def test_rejected_state_restore_leaves_existing_samples_exactly_unchanged():
    p = Tensor([0.0], requires_grad=True)
    estimator = GradientNoiseScaleEstimator(p, batch_size=2)
    p.grad = np.array([3.0])
    estimator.capture()
    before = estimator.state_dict()

    broken = copy.deepcopy(before)
    broken["samples"][0][0] = np.array([np.inf])
    with pytest.raises(ValueError, match="finite"):
        estimator.load_state_dict(broken)

    after = estimator.state_dict()
    assert after["version"] == before["version"]
    assert after["type"] == before["type"]
    assert after["batch_size"] == before["batch_size"]
    assert after["sample_count"] == before["sample_count"]
    np.testing.assert_array_equal(after["samples"][0][0], before["samples"][0][0])
