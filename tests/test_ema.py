"""Regression tests for parameter exponential moving averages."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import engine.ops as ops
from engine.ema import ExponentialMovingAverage
from engine.tensor import Tensor


def _rng_state_equal(first, second):
    return (
        first[0] == second[0]
        and np.array_equal(first[1], second[1])
        and first[2:] == second[2:]
    )


def _assert_ema_state_equal(first, second):
    assert first["decay"] == second["decay"]
    assert first["num_updates"] == second["num_updates"]
    assert len(first["averages"]) == len(second["averages"])
    for left, right in zip(first["averages"], second["averages"]):
        np.testing.assert_array_equal(left, right)


def test_initial_state_accepts_direct_tensor_and_returns_independent_averages():
    parameter = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    ema = ExponentialMovingAverage(parameter, decay=0.9)

    assert ema.decay == 0.9
    assert ema.num_updates == 0
    averages = ema.averages()
    assert len(averages) == 1
    np.testing.assert_array_equal(averages[0], parameter.data)

    averages[0][0, 0] = 999.0
    np.testing.assert_array_equal(
        ema.averages()[0], np.array([[1.0, 2.0], [3.0, 4.0]])
    )


def test_parameter_generator_is_materialized_once_in_order():
    parameters = [Tensor([1.0]), Tensor([2.0])]
    yielded = []

    def items():
        for index, parameter in enumerate(parameters):
            yielded.append(index)
            yield parameter

    ema = ExponentialMovingAverage(items(), decay=np.float64(0.5))

    assert yielded == [0, 1]
    assert [value.item() for value in ema.averages()] == [1.0, 2.0]


def test_decay_validation_happens_before_parameter_generator_consumption():
    consumed = []

    def items():
        consumed.append(True)
        yield Tensor([1.0])

    with pytest.raises(TypeError, match="EMA decay must be a real number"):
        ExponentialMovingAverage(items(), decay=True)
    assert consumed == []

    with pytest.raises(ValueError, match="EMA decay must be finite"):
        ExponentialMovingAverage(items(), decay=10**400)
    assert consumed == []

    with pytest.raises(ValueError, match=r"EMA decay must be in \[0, 1\]"):
        ExponentialMovingAverage(items(), decay=-0.1)
    assert consumed == []


def test_parameter_collection_validation_is_explicit():
    parameter = Tensor([1.0])

    with pytest.raises(TypeError, match="EMA parameters must be a Tensor or iterable"):
        ExponentialMovingAverage(123)
    with pytest.raises(ValueError, match="at least one Tensor"):
        ExponentialMovingAverage([])
    with pytest.raises(TypeError, match="EMA parameter 1 must be a Tensor"):
        ExponentialMovingAverage([parameter, object()])
    with pytest.raises(ValueError, match="must not contain duplicate Tensors"):
        ExponentialMovingAverage([parameter, parameter])
    with pytest.raises(ValueError, match="EMA parameter 0 must contain only finite"):
        ExponentialMovingAverage(Tensor([np.inf]))


def test_update_uses_exact_ema_formula_without_touching_tensor_state_or_rng():
    first = Tensor([2.0, 4.0], requires_grad=True)
    second = Tensor([10.0], requires_grad=True)
    first.grad[...] = [7.0, 8.0]
    second.grad[...] = [9.0]
    ema = ExponentialMovingAverage([first, second], decay=0.75)

    first.data[...] = [6.0, 8.0]
    second.data[...] = [2.0]
    first_version = first._version
    second_version = second._version
    first_grad = first.grad
    second_grad = second.grad
    rng_before = np.random.get_state()

    returned = ema.update()

    rng_after = np.random.get_state()
    assert returned is ema
    assert ema.num_updates == 1
    np.testing.assert_array_equal(ema.averages()[0], [3.0, 5.0])
    np.testing.assert_array_equal(ema.averages()[1], [8.0])
    assert first._version == first_version
    assert second._version == second_version
    assert first.grad is first_grad
    assert second.grad is second_grad
    np.testing.assert_array_equal(first.grad, [7.0, 8.0])
    np.testing.assert_array_equal(second.grad, [9.0])
    assert _rng_state_equal(rng_before, rng_after)

    first.data[...] = [11.0, 13.0]
    second.data[...] = [4.0]
    ema.update()
    np.testing.assert_allclose(ema.averages()[0], [5.0, 7.0])
    np.testing.assert_allclose(ema.averages()[1], [7.0])
    assert ema.num_updates == 2


def test_decay_zero_and_one_have_well_defined_endpoint_behavior():
    zero_parameter = Tensor([1.0, 2.0])
    zero = ExponentialMovingAverage(zero_parameter, decay=0.0)
    zero_parameter.data[...] = [9.0, 8.0]
    zero.update()
    np.testing.assert_array_equal(zero.averages()[0], [9.0, 8.0])
    assert zero.num_updates == 1

    one_parameter = Tensor([1.0, 2.0])
    one = ExponentialMovingAverage(one_parameter, decay=1.0)
    one_parameter.data[...] = [9.0, 8.0]
    one.update()
    np.testing.assert_array_equal(one.averages()[0], [1.0, 2.0])
    assert one.num_updates == 1


def test_failed_update_is_transactional_for_shadow_state():
    first = Tensor([1.0])
    second = Tensor([2.0])
    ema = ExponentialMovingAverage([first, second], decay=0.5)
    before = ema.state_dict()

    first.data[...] = [3.0]
    second.data[...] = [np.inf]

    with pytest.raises(ValueError, match="EMA parameter 1 must contain only finite"):
        ema.update()

    _assert_ema_state_equal(ema.state_dict(), before)


def test_shape_changes_are_rejected_before_shadow_or_parameter_writes():
    parameter = Tensor([1.0, 2.0])
    ema = ExponentialMovingAverage(parameter, decay=0.5)
    before = ema.state_dict()
    parameter.data = np.array([[1.0, 2.0]])
    version = parameter._version

    with pytest.raises(ValueError, match="EMA parameter 0 shape changed"):
        ema.update()
    with pytest.raises(ValueError, match="EMA parameter 0 shape changed"):
        ema.copy_to()

    _assert_ema_state_equal(ema.state_dict(), before)
    assert parameter._version == version
    np.testing.assert_array_equal(parameter.data, [[1.0, 2.0]])


def test_copy_to_installs_shadow_values_and_invalidates_existing_graph():
    parameter = Tensor([1.0, 2.0], requires_grad=True)
    ema = ExponentialMovingAverage(parameter, decay=0.5)
    parameter.data[...] = [3.0, 4.0]
    ema.update()  # shadow = [2, 3]
    parameter.data[...] = [5.0, 6.0]

    graph = parameter * parameter
    version = parameter._version
    grad = parameter.grad

    assert ema.copy_to() is ema
    np.testing.assert_array_equal(parameter.data, [2.0, 3.0])
    assert parameter._version == version + 1
    assert parameter.grad is grad

    with pytest.raises(RuntimeError, match="modified after forward"):
        graph.backward()


def test_copy_to_skips_identical_values_without_spurious_version_bump():
    parameter = Tensor([1.0, 2.0], requires_grad=True)
    ema = ExponentialMovingAverage(parameter)
    version = parameter._version

    ema.copy_to()

    assert parameter._version == version


def test_copy_to_preflights_all_destination_writability_before_any_write():
    first = Tensor([1.0])
    second = Tensor([2.0])
    ema = ExponentialMovingAverage([first, second], decay=0.5)
    first.data[...] = [10.0]
    second.data[...] = [20.0]
    first_before = first.data.copy()
    first_version = first._version
    second.data.flags.writeable = False

    with pytest.raises(ValueError, match="EMA parameter 1 data is not writeable"):
        ema.copy_to()

    np.testing.assert_array_equal(first.data, first_before)
    assert first._version == first_version


def test_average_parameters_restores_values_and_gradients_after_normal_exit():
    parameter = Tensor([1.0, 2.0], requires_grad=True)
    parameter.grad[...] = [7.0, 8.0]
    grad = parameter.grad
    ema = ExponentialMovingAverage(parameter, decay=0.5)
    parameter.data[...] = [3.0, 4.0]
    ema.update()  # [2, 3]
    parameter.data[...] = [5.0, 6.0]
    original = parameter.data.copy()
    version = parameter._version

    with ema.average_parameters() as active:
        assert active is ema
        np.testing.assert_array_equal(parameter.data, [2.0, 3.0])
        assert parameter.grad is grad
        np.testing.assert_array_equal(parameter.grad, [7.0, 8.0])

    np.testing.assert_array_equal(parameter.data, original)
    assert parameter.grad is grad
    np.testing.assert_array_equal(parameter.grad, [7.0, 8.0])
    assert parameter._version == version + 2


def test_average_parameters_restores_entry_values_and_shape_after_exception():
    parameter = Tensor([1.0, 2.0], requires_grad=True)
    ema = ExponentialMovingAverage(parameter, decay=0.5)
    parameter.data[...] = [3.0, 4.0]
    ema.update()
    parameter.data[...] = [5.0, 6.0]
    original = parameter.data.copy()

    with pytest.raises(RuntimeError, match="body failed"):
        with ema.average_parameters():
            np.testing.assert_array_equal(parameter.data, [2.0, 3.0])
            parameter.data = np.array([[99.0]])
            raise RuntimeError("body failed")

    assert parameter.shape == (2,)
    np.testing.assert_array_equal(parameter.data, original)


def test_noop_average_context_does_not_invalidate_graph():
    parameter = Tensor([2.0, 3.0], requires_grad=True)
    ema = ExponentialMovingAverage(parameter)
    graph = ops.sum(parameter * parameter)
    version = parameter._version

    with ema.average_parameters():
        np.testing.assert_array_equal(parameter.data, [2.0, 3.0])

    assert parameter._version == version
    graph.backward()
    np.testing.assert_array_equal(parameter.grad, [4.0, 6.0])


def test_state_dict_round_trip_is_independent_from_caller_mutation():
    first = Tensor([1.0, 2.0])
    second = Tensor([[3.0], [4.0]])
    source = ExponentialMovingAverage([first, second], decay=0.8)
    first.data[...] = [6.0, 7.0]
    second.data[...] = [[8.0], [9.0]]
    source.update()
    state = source.state_dict()

    target = ExponentialMovingAverage(
        [Tensor([100.0, 200.0]), Tensor([[300.0], [400.0]])], decay=0.1
    )
    assert target.load_state_dict(state) is target

    assert target.decay == 0.8
    assert target.num_updates == 1
    for actual, expected in zip(target.averages(), source.averages()):
        np.testing.assert_array_equal(actual, expected)

    state["averages"][0][0] = 999.0
    assert target.averages()[0][0] != 999.0
    exported = target.state_dict()
    exported["averages"][1][0, 0] = 999.0
    assert target.averages()[1][0, 0] != 999.0


def test_load_state_dict_failure_leaves_existing_ema_state_unchanged():
    parameters = [Tensor([1.0, 2.0]), Tensor([3.0])]
    ema = ExponentialMovingAverage(parameters, decay=0.7)
    parameters[0].data[...] = [5.0, 6.0]
    parameters[1].data[...] = [7.0]
    ema.update()
    before = ema.state_dict()

    bad_states = [
        {"decay": 0.2, "num_updates": 9, "averages": [np.ones(2)]},
        {"decay": 0.2, "num_updates": 9, "averages": [np.ones(2), np.ones(2)]},
        {"decay": 0.2, "num_updates": 9, "averages": [np.ones(2), [np.nan]]},
        {"decay": True, "num_updates": 9, "averages": [np.ones(2), np.ones(1)]},
        {"decay": 0.2, "num_updates": -1, "averages": [np.ones(2), np.ones(1)]},
    ]

    for state in bad_states:
        with pytest.raises((TypeError, ValueError)):
            ema.load_state_dict(state)
        _assert_ema_state_equal(ema.state_dict(), before)


def test_load_state_dict_validates_envelope_and_accepts_numeric_arrays():
    ema = ExponentialMovingAverage([Tensor([1.0, 2.0]), Tensor([3.0])])

    with pytest.raises(TypeError, match="EMA state must be a mapping"):
        ema.load_state_dict([])
    with pytest.raises(ValueError, match="missing required keys"):
        ema.load_state_dict({"decay": 0.5})
    with pytest.raises(TypeError, match="EMA averages must be a list or tuple"):
        ema.load_state_dict({"decay": 0.5, "num_updates": 0, "averages": {}})
    with pytest.raises(TypeError, match="EMA average 0 must contain real numeric"):
        ema.load_state_dict(
            {"decay": 0.5, "num_updates": 0, "averages": [[True, False], [1.0]]}
        )

    ema.load_state_dict(
        {
            "decay": np.float64(0.25),
            "num_updates": np.int64(4),
            "averages": [np.array([1, 2], dtype=np.int32), [3.0]],
            "future_metadata": "ignored",
        }
    )
    assert ema.decay == 0.25
    assert ema.num_updates == 4
    assert all(value.dtype == np.float64 for value in ema.averages())


def test_extended_precision_state_overflow_is_rejected_transactionally():
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("platform longdouble has no wider finite range than float64")

    ema = ExponentialMovingAverage(Tensor([1.0]), decay=0.5)
    before = ema.state_dict()
    huge = np.array([np.finfo(np.longdouble).max], dtype=np.longdouble)

    with pytest.raises(ValueError, match="EMA average 0 must fit in float64"):
        ema.load_state_dict(
            {"decay": 0.25, "num_updates": 3, "averages": [huge]}
        )

    _assert_ema_state_equal(ema.state_dict(), before)


def test_state_operations_and_context_do_not_consume_numpy_rng():
    parameter = Tensor([1.0, 2.0])
    ema = ExponentialMovingAverage(parameter, decay=0.5)
    parameter.data[...] = [3.0, 4.0]
    rng_before = np.random.get_state()

    ema.update()
    state = ema.state_dict()
    ema.load_state_dict(state)
    with ema.average_parameters():
        pass

    rng_after = np.random.get_state()
    assert _rng_state_equal(rng_before, rng_after)
