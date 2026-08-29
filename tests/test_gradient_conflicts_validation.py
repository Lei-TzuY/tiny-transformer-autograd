import numpy as np
import pytest

from engine.gradient_conflicts import GradientConflictAnalyzer
from engine.tensor import Tensor


def test_constructor_materializes_parameter_generator_once():
    first = Tensor([1.0], requires_grad=True)
    second = Tensor([2.0], requires_grad=True)
    consumed = []

    def parameters():
        for parameter in (first, second):
            consumed.append(parameter)
            yield parameter

    analyzer = GradientConflictAnalyzer(parameters())
    assert analyzer.parameters == (first, second)
    assert consumed == [first, second]


def test_constructor_rejects_malformed_and_duplicate_parameters():
    trainable = Tensor([1.0], requires_grad=True)
    frozen = Tensor([1.0], requires_grad=False)

    with pytest.raises(TypeError, match="parameters must be a Tensor or iterable"):
        GradientConflictAnalyzer(123)
    with pytest.raises(TypeError, match="parameters must contain only Tensors"):
        GradientConflictAnalyzer([trainable, object()])
    with pytest.raises(ValueError, match="duplicate Tensors"):
        GradientConflictAnalyzer([trainable, trainable])
    with pytest.raises(ValueError, match="all parameters must require gradients"):
        GradientConflictAnalyzer([frozen])


def test_task_name_validation_is_state_neutral():
    parameter = Tensor([0.0], requires_grad=True)
    analyzer = GradientConflictAnalyzer(parameter)

    for bad in (1, True, [], object()):
        with pytest.raises(TypeError, match="task name must be a string or None"):
            analyzer.capture(bad)
        assert analyzer.task_count == 0

    with pytest.raises(ValueError, match="task name must not be empty"):
        analyzer.capture("")
    assert analyzer.task_count == 0

    analyzer.capture("task")
    with pytest.raises(ValueError, match="duplicate task name"):
        analyzer.capture("task")
    assert analyzer.task_count == 1


def test_failed_auto_named_capture_does_not_consume_the_name():
    parameter = Tensor([0.0], requires_grad=True)
    analyzer = GradientConflictAnalyzer(parameter)
    parameter.grad = "bad"

    with pytest.raises(TypeError, match="gradient 0 must be a NumPy array or None"):
        analyzer.capture()
    assert analyzer.task_count == 0

    parameter.grad = np.array([1.0])
    assert analyzer.capture() == "task_0"


def test_capture_rejects_late_bad_gradient_without_partial_task():
    first = Tensor([0.0], requires_grad=True)
    second = Tensor([0.0], requires_grad=True)
    analyzer = GradientConflictAnalyzer([first, second])
    first.grad[...] = [2.0]
    second.grad[...] = [3.0]
    analyzer.capture("baseline")
    baseline = analyzer.task_gradients()

    first.grad[...] = [7.0]
    second.grad = np.array([np.nan])
    with pytest.raises(ValueError, match="gradient 1 must contain only finite values"):
        analyzer.capture("bad")

    assert analyzer.task_count == 1
    after = analyzer.task_gradients()
    assert after[0][0] == baseline[0][0]
    np.testing.assert_array_equal(after[0][1][0], baseline[0][1][0])
    np.testing.assert_array_equal(after[0][1][1], baseline[0][1][1])


@pytest.mark.parametrize(
    "gradient, error_type, message",
    [
        ([1.0], TypeError, "NumPy array or None"),
        (np.array([1]), TypeError, "floating-point values"),
        (np.array([True]), TypeError, "floating-point values"),
        (np.array([1.0 + 0.0j]), TypeError, "floating-point values"),
        (np.array([np.nan]), ValueError, "only finite values"),
        (np.array([np.inf]), ValueError, "only finite values"),
    ],
)
def test_gradient_type_and_finiteness_validation(gradient, error_type, message):
    parameter = Tensor([0.0], requires_grad=True)
    analyzer = GradientConflictAnalyzer(parameter)
    parameter.grad = gradient

    with pytest.raises(error_type, match=message):
        analyzer.capture("bad")
    assert analyzer.task_count == 0


def test_gradient_shape_mismatch_is_rejected():
    parameter = Tensor([0.0, 0.0], requires_grad=True)
    analyzer = GradientConflictAnalyzer(parameter)
    parameter.grad = np.array([1.0])

    with pytest.raises(ValueError, match="gradient 0 shape mismatch"):
        analyzer.capture("bad")


def test_float32_gradients_are_normalized_into_independent_float64_snapshots():
    parameter = Tensor([0.0, 0.0], requires_grad=True)
    analyzer = GradientConflictAnalyzer(parameter)
    source = np.array([1.25, -2.5], dtype=np.float32)
    parameter.grad = source
    analyzer.capture("task")

    source[...] = 99.0
    captured = analyzer.task_gradients()[0][1][0]
    assert captured.dtype == np.float64
    np.testing.assert_allclose(captured, [1.25, -2.5])


def test_extended_precision_gradient_outside_float64_is_rejected_when_available():
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("platform longdouble has no wider range than float64")

    parameter = Tensor([0.0], requires_grad=True)
    analyzer = GradientConflictAnalyzer(parameter)
    too_large = np.longdouble(np.finfo(np.float64).max) * np.longdouble(2)
    parameter.grad = np.array([too_large], dtype=np.longdouble)

    with pytest.raises(ValueError, match="gradient 0 must fit in float64"):
        analyzer.capture("bad")
    assert analyzer.task_count == 0


def test_parameter_shape_and_trainability_drift_are_rejected():
    parameter = Tensor([0.0], requires_grad=True)
    analyzer = GradientConflictAnalyzer(parameter)
    analyzer.capture("baseline")

    parameter.data = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="parameter 0 shape changed"):
        analyzer.report()

    parameter.data = np.array([0.0])
    parameter.requires_grad = False
    with pytest.raises(ValueError, match="parameter 0 no longer requires gradients"):
        analyzer.report()
    with pytest.raises(ValueError, match="parameter 0 no longer requires gradients"):
        analyzer.capture("next")


def test_extreme_opposite_gradients_have_finite_negative_cosine_under_strict_errors():
    parameter = Tensor([0.0, 0.0], requires_grad=True)
    analyzer = GradientConflictAnalyzer(parameter)

    parameter.grad[...] = [1.3e308, -1.3e308]
    analyzer.capture("a")
    parameter.grad[...] = [-1.3e308, 1.3e308]
    analyzer.capture("b")

    with np.errstate(all="raise"):
        report = analyzer.report()

    assert report["cosine_similarity_matrix"][0][1] == pytest.approx(-1.0)
    assert report["conflict_pair_count"] == 1
    assert report["pairs"][0]["status"] == "conflict"


def test_subnormal_gradients_remain_comparable_without_warning_leaks():
    parameter = Tensor([0.0], requires_grad=True)
    analyzer = GradientConflictAnalyzer(parameter)
    tiny = np.finfo(np.float64).smallest_subnormal

    parameter.grad[...] = [tiny]
    analyzer.capture("positive")
    parameter.grad[...] = [-tiny]
    analyzer.capture("negative")

    with np.errstate(all="raise"):
        report = analyzer.report()

    assert report["tasks"][0]["l2_norm"] == tiny
    assert report["tasks"][1]["l2_norm"] == tiny
    assert report["cosine_similarity_matrix"][0][1] == -1.0
