import json

import numpy as np

from engine.directional_curvature import directional_curvature
from engine.tensor import Tensor


def test_huge_equal_losses_cancel_exactly_without_intermediate_overflow():
    parameter = Tensor([0.0], requires_grad=True)
    maximum = np.finfo(np.float64).max
    with np.errstate(all="raise"):
        report = directional_curvature(
            lambda: maximum,
            parameter,
            np.array([1.0]),
            step=0.5,
        )
    assert report["curvature"] == 0.0
    assert report["curvature_overflow"] is False
    assert report["curvature_underflow"] is False
    assert report["curvature_sign"] == 0
    json.dumps(report, allow_nan=False)


def test_positive_curvature_overflow_is_reported_without_json_infinity():
    parameter = Tensor([0.0], requires_grad=True)
    maximum = np.finfo(np.float64).max

    def loss():
        return 0.0 if parameter.data[0] == 0.0 else maximum

    with np.errstate(all="raise"):
        report = directional_curvature(
            loss,
            parameter,
            np.array([1.0]),
            step=1e-200,
        )
    assert report["curvature"] is None
    assert report["curvature_overflow"] is True
    assert report["curvature_underflow"] is False
    assert report["curvature_sign"] == 1
    json.dumps(report, allow_nan=False)


def test_negative_curvature_overflow_retains_sign():
    parameter = Tensor([0.0], requires_grad=True)
    maximum = np.finfo(np.float64).max

    def loss():
        return 0.0 if parameter.data[0] == 0.0 else -maximum

    report = directional_curvature(
        loss,
        parameter,
        np.array([1.0]),
        step=1e-200,
    )
    assert report["curvature"] is None
    assert report["curvature_overflow"] is True
    assert report["curvature_sign"] == -1


def test_positive_curvature_underflow_is_reported_separately():
    parameter = Tensor([0.0], requires_grad=True)
    smallest = np.nextafter(0.0, 1.0)

    def loss():
        return 0.0 if parameter.data[0] == 0.0 else smallest

    with np.errstate(all="raise"):
        report = directional_curvature(
            loss,
            parameter,
            np.array([1.0]),
            step=1e150,
        )
    assert report["curvature"] == 0.0
    assert report["curvature_overflow"] is False
    assert report["curvature_underflow"] is True
    assert report["curvature_sign"] == 1


def test_opposite_large_losses_produce_finite_negative_curvature_when_exact_ratio_fits():
    parameter = Tensor([0.0], requires_grad=True)
    large = 1e300

    def loss():
        if parameter.data[0] > 0:
            return large
        if parameter.data[0] < 0:
            return -large
        return 0.0

    report = directional_curvature(
        loss,
        parameter,
        np.array([1.0]),
        step=1e150,
    )
    # Odd loss samples cancel in the second difference despite huge endpoints.
    assert report["curvature"] == 0.0
    assert report["curvature_overflow"] is False
