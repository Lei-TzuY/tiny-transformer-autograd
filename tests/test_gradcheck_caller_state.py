"""Gradcheck must leave caller-owned Tensor state exactly reusable."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import engine.ops as ops
from engine.gradcheck import gradcheck
from engine.tensor import Tensor


def test_gradcheck_restores_parameter_version_so_existing_graph_stays_valid():
    parameter = Tensor([2.0, -3.0], requires_grad=True)
    existing_graph = ops.sum(parameter * parameter)
    version_before = parameter._version

    assert gradcheck(lambda: parameter * parameter, parameters=[parameter])

    assert parameter._version == version_before
    existing_graph.backward()
    np.testing.assert_array_equal(parameter.grad, [4.0, -6.0])


def test_gradcheck_restores_existing_grad_buffer_identity_and_values():
    parameter = Tensor([1.5, -0.5], requires_grad=True)
    grad_buffer = parameter.grad
    grad_buffer[:] = [7.0, -11.0]

    assert gradcheck(lambda: parameter * parameter, parameters=[parameter])

    assert parameter.grad is grad_buffer
    np.testing.assert_array_equal(parameter.grad, [7.0, -11.0])
