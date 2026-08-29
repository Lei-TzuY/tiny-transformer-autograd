import numpy as np

from engine.diagonal_fisher import DiagonalFisherEstimator
from engine.tensor import Tensor


def test_capture_is_mean_of_squared_observed_gradients_not_square_of_mean_gradient():
    parameter = Tensor([0.0], requires_grad=True)
    estimator = DiagonalFisherEstimator(parameter)

    parameter.grad = np.array([1.0])
    estimator.capture()
    parameter.grad = np.array([-1.0])
    estimator.capture()

    # Empirical diagonal Fisher is E[g^2] = 1 here. Squaring the mean gradient
    # would incorrectly produce zero, so this regression pins the definition.
    np.testing.assert_allclose(estimator.diagonals()[0], [1.0])


def test_microbatch_gradient_capture_is_not_claimed_to_equal_per_example_fisher():
    # Two per-example gradients +1 and +3 have per-example E[g^2] = 5.
    per_example_parameter = Tensor([0.0], requires_grad=True)
    per_example = DiagonalFisherEstimator(per_example_parameter)
    per_example_parameter.grad = np.array([1.0])
    per_example.capture()
    per_example_parameter.grad = np.array([3.0])
    per_example.capture()

    # A caller that instead captures the mean microbatch gradient observes 2,
    # so that observation contributes 2^2 = 4. Both are correct for their
    # stated observation granularity; the estimator does not silently equate them.
    microbatch_parameter = Tensor([0.0], requires_grad=True)
    microbatch_parameter.grad = np.array([2.0])
    microbatch = DiagonalFisherEstimator(microbatch_parameter).capture()

    np.testing.assert_allclose(per_example.diagonals()[0], [5.0])
    np.testing.assert_allclose(microbatch.diagonals()[0], [4.0])
