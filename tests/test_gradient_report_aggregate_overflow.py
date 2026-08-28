"""Regression coverage for aggregate gradient magnitude overflow reporting."""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.gradient_report import gradient_report
from engine.tensor import Tensor


def test_aggregate_max_does_not_hide_unrepresentable_finite_magnitude():
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("platform longdouble has no wider finite range than float64")

    huge = Tensor([1.0], requires_grad=True)
    ordinary = Tensor([2.0], requires_grad=True)
    huge.grad = np.array([np.finfo(np.longdouble).max], dtype=np.longdouble)
    ordinary.grad[...] = [1.0]

    with np.errstate(all="raise"):
        report = gradient_report([huge, ordinary])

    assert report["entries"][0]["max_finite_abs"] is None
    assert report["entries"][0]["magnitude_overflow"] is True
    assert report["entries"][1]["max_finite_abs"] == 1.0
    assert report["entries"][1]["magnitude_overflow"] is False
    assert report["max_finite_abs_gradient"] is None
    assert report["max_finite_abs_gradient_overflow"] is True
    json.dumps(report, allow_nan=False)


def test_aggregate_max_overflow_flag_is_false_when_all_magnitudes_fit():
    first = Tensor([1.0, 2.0], requires_grad=True)
    second = Tensor([3.0], requires_grad=True)
    first.grad[...] = [-4.0, 2.0]
    second.grad[...] = [7.0]

    report = gradient_report([first, second])

    assert report["max_finite_abs_gradient"] == 7.0
    assert report["max_finite_abs_gradient_overflow"] is False
