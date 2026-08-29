import numpy as np
import pytest

from engine.gradient_accumulator import GradientAccumulator
from engine.tensor import Tensor


def test_constructor_materializes_generator_once_and_preserves_order():
    first = Tensor([0.0], requires_grad=True)
    second = Tensor([0.0], requires_grad=True)
    visits = []

    def parameters():
        visits.append("start")
        yield first
        yield second

    accumulator = GradientAccumulator(parameters())
    assert accumulator.parameters == (first, second)
    assert visits == ["start"]


def test_constructor_rejects_invalid_parameter_collections():
    with pytest.raises(TypeError, match="Tensor or iterable"):
        GradientAccumulator(123)
    with pytest.raises(TypeError, match="only Tensors"):
        GradientAccumulator([Tensor([0.0], requires_grad=True), object()])

    parameter = Tensor([0.0], requires_grad=True)
    with pytest.raises(ValueError, match="duplicate"):
        GradientAccumulator([parameter, parameter])
    with pytest.raises(ValueError, match="all parameters must require gradients"):
        GradientAccumulator(Tensor([0.0], requires_grad=False))


@pytest.mark.parametrize("weight", [True, np.bool_(False), "1", None, 1 + 2j])
def test_weight_type_validation(weight):
    parameter = Tensor([0.0], requires_grad=True)
    accumulator = GradientAccumulator(parameter)
    with pytest.raises(TypeError, match="weight must be a real number"):
        accumulator.accumulate(weight=weight)


@pytest.mark.parametrize("weight", [0.0, -1.0, -np.float32(2.0)])
def test_weight_must_be_positive(weight):
    parameter = Tensor([0.0], requires_grad=True)
    accumulator = GradientAccumulator(parameter)
    with pytest.raises(ValueError, match="weight must be positive"):
        accumulator.accumulate(weight=weight)


@pytest.mark.parametrize("weight", [np.inf, -np.inf, np.nan, 10**400])
def test_weight_must_be_finite_binary64(weight):
    parameter = Tensor([0.0], requires_grad=True)
    accumulator = GradientAccumulator(parameter)
    with pytest.raises(ValueError, match="weight must be finite"):
        accumulator.accumulate(weight=weight)


def test_weight_validation_precedes_gradient_inspection():
    parameter = Tensor([0.0], requires_grad=True)
    parameter.grad = "bad"
    accumulator = GradientAccumulator(parameter)
    with pytest.raises(ValueError, match="weight must be positive"):
        accumulator.accumulate(weight=0.0)


def test_gradient_type_dtype_shape_and_finiteness_validation():
    parameter = Tensor([0.0, 0.0], requires_grad=True)
    accumulator = GradientAccumulator(parameter)

    parameter.grad = [1.0, 2.0]
    with pytest.raises(TypeError, match="NumPy array"):
        accumulator.accumulate()

    parameter.grad = np.array([1, 2], dtype=np.int64)
    with pytest.raises(TypeError, match="floating-point"):
        accumulator.accumulate()

    parameter.grad = np.array([1.0], dtype=np.float64)
    with pytest.raises(ValueError, match="shape mismatch"):
        accumulator.accumulate()

    parameter.grad = np.array([1.0, np.inf], dtype=np.float64)
    with pytest.raises(ValueError, match="only finite"):
        accumulator.accumulate()


def test_extended_precision_gradient_that_cannot_fit_float64_is_rejected():
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("platform longdouble is not wider than float64")
    parameter = Tensor([0.0], requires_grad=True)
    accumulator = GradientAccumulator(parameter)
    parameter.grad = np.array(
        [np.longdouble(np.finfo(np.float64).max) * np.longdouble(2)],
        dtype=np.longdouble,
    )
    assert np.isfinite(parameter.grad).all()

    with np.errstate(all="raise"):
        with pytest.raises(ValueError, match="must fit in float64"):
            accumulator.accumulate()
    assert accumulator.accumulation_count == 0
    assert accumulator.total_weight == 0.0


def test_total_weight_overflow_is_rejected_transactionally():
    parameter = Tensor([0.0], requires_grad=True)
    parameter.grad[...] = [1.0]
    accumulator = GradientAccumulator(parameter)
    accumulator.accumulate(weight=1e308)
    before = accumulator.state_dict()

    with pytest.raises(ValueError, match="total accumulated weight must remain finite"):
        accumulator.accumulate(weight=1e308)

    after = accumulator.state_dict()
    assert after["total_weight"] == before["total_weight"]
    assert after["accumulation_count"] == before["accumulation_count"]
    np.testing.assert_array_equal(after["averages"][0], before["averages"][0])


def test_parameter_shape_and_trainability_drift_are_rejected():
    parameter = Tensor([0.0, 0.0], requires_grad=True)
    accumulator = GradientAccumulator(parameter)
    parameter.data = np.array([0.0])
    with pytest.raises(ValueError, match="parameter 0 shape changed"):
        accumulator.accumulate()

    other = Tensor([0.0], requires_grad=True)
    accumulator = GradientAccumulator(other)
    other.requires_grad = False
    with pytest.raises(ValueError, match="no longer requires gradients"):
        accumulator.accumulate()


def test_average_gradient_copies_are_independent():
    parameter = Tensor([0.0], requires_grad=True)
    parameter.grad[...] = [7.0]
    accumulator = GradientAccumulator(parameter)
    accumulator.accumulate()
    first = accumulator.average_gradients()
    second = accumulator.average_gradients()

    first[0][...] = 99.0
    np.testing.assert_array_equal(second[0], [7.0])
    np.testing.assert_array_equal(accumulator.average_gradients()[0], [7.0])


def test_copy_to_grads_does_not_reset_accumulator():
    parameter = Tensor([0.0], requires_grad=True)
    parameter.grad[...] = [2.0]
    accumulator = GradientAccumulator(parameter)
    accumulator.accumulate(weight=3.0)
    accumulator.copy_to_grads()

    assert accumulator.accumulation_count == 1
    assert accumulator.total_weight == 3.0
    np.testing.assert_array_equal(accumulator.average_gradients()[0], [2.0])
