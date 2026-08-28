import copy

import numpy as np
import pytest

from engine.lion import Lion
from engine.tensor import Tensor


def test_lion_exact_two_step_arithmetic_and_state_progression():
    parameter = Tensor([1.0, -2.0], requires_grad=True)
    optimizer = Lion(
        [parameter], lr=0.1, betas=(0.9, 0.99), weight_decay=0.1
    )

    parameter.grad[...] = [0.5, -0.25]
    returned = optimizer.step()

    assert returned is optimizer
    np.testing.assert_allclose(parameter.data, [0.89, -1.88], rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(
        optimizer.state_dict()["momentum"][0], [0.005, -0.0025], rtol=0.0, atol=1e-15
    )
    assert optimizer.step_count == 1
    assert optimizer.state_dict()["steps"] == [1]

    parameter.grad[...] = [-1.0, 1.0]
    optimizer.step()

    np.testing.assert_allclose(
        parameter.data, [0.9811, -1.9612], rtol=0.0, atol=1e-15
    )
    np.testing.assert_allclose(
        optimizer.state_dict()["momentum"][0],
        [-0.00505, 0.007525],
        rtol=0.0,
        atol=1e-15,
    )
    assert optimizer.step_count == 2
    assert optimizer.state_dict()["steps"] == [2]


def test_missing_gradient_skips_parameter_specific_state_but_global_step_advances():
    first = Tensor([1.0], requires_grad=True)
    second = Tensor([2.0], requires_grad=True)
    optimizer = Lion([first, second], lr=0.1)
    first.grad[...] = [1.0]
    second.grad = None

    optimizer.step()

    np.testing.assert_array_equal(first.data, [0.9])
    np.testing.assert_array_equal(second.data, [2.0])
    assert optimizer.step_count == 1
    assert optimizer.state_dict()["steps"] == [1, 0]
    np.testing.assert_array_equal(optimizer.state_dict()["momentum"][1], [0.0])


def test_empty_optimizer_step_has_defined_counter_semantics():
    optimizer = Lion([], lr=0.1)

    optimizer.step()

    assert optimizer.step_count == 1
    assert optimizer.state_dict()["steps"] == []
    assert optimizer.state_dict()["momentum"] == []


def test_zero_gradient_without_weight_decay_is_noop_for_tensor_version():
    parameter = Tensor([2.0], requires_grad=True)
    parameter.grad[...] = [0.0]
    optimizer = Lion([parameter], lr=0.1)
    version = parameter._version

    optimizer.step()

    np.testing.assert_array_equal(parameter.data, [2.0])
    assert parameter._version == version
    assert optimizer.step_count == 1
    assert optimizer.state_dict()["steps"] == [1]


def test_weight_decay_applies_even_when_active_gradient_is_zero():
    parameter = Tensor([2.0], requires_grad=True)
    parameter.grad[...] = [0.0]
    optimizer = Lion([parameter], lr=0.1, weight_decay=0.5)
    version = parameter._version

    optimizer.step()

    np.testing.assert_allclose(parameter.data, [1.9], rtol=0.0, atol=1e-15)
    assert parameter._version == version + 1


def test_all_active_writes_are_preflighted_before_first_parameter_changes():
    first = Tensor([1.0], requires_grad=True)
    second = Tensor([2.0], requires_grad=True)
    first.grad[...] = [1.0]
    second.grad[...] = [1.0]
    second.data.flags.writeable = False
    optimizer = Lion([first, second], lr=0.1)
    first_version = first._version
    before = optimizer.state_dict()

    with pytest.raises(ValueError, match="parameter 1 storage must be writeable"):
        optimizer.step()

    np.testing.assert_array_equal(first.data, [1.0])
    np.testing.assert_array_equal(second.data, [2.0])
    assert first._version == first_version
    assert optimizer.step_count == 0
    np.testing.assert_array_equal(
        optimizer.state_dict()["momentum"][0], before["momentum"][0]
    )
    np.testing.assert_array_equal(
        optimizer.state_dict()["momentum"][1], before["momentum"][1]
    )


def test_extreme_finite_gradients_do_not_overflow_momentum_blends():
    first = Tensor([1.0], requires_grad=True)
    second = Tensor([-1.0], requires_grad=True)
    first.grad[...] = [1.3e308]
    second.grad[...] = [-1.3e308]
    optimizer = Lion([first, second], lr=0.01, betas=(0.9, 0.99))

    with np.errstate(all="raise"):
        optimizer.step()

    np.testing.assert_allclose(first.data, [0.99], rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(second.data, [-0.99], rtol=0.0, atol=1e-15)
    state = optimizer.state_dict()
    assert np.isfinite(state["momentum"][0]).all()
    assert np.isfinite(state["momentum"][1]).all()


def test_unrepresentable_parameter_candidate_rejects_whole_step():
    first = Tensor([1.0], requires_grad=True)
    second = Tensor([np.finfo(np.float64).max], requires_grad=True)
    first.grad[...] = [1.0]
    second.grad[...] = [-1.0]
    optimizer = Lion([first, second], lr=1e308)
    versions = (first._version, second._version)

    with pytest.raises(ValueError, match="unrepresentable finite update"):
        optimizer.step()

    np.testing.assert_array_equal(first.data, [1.0])
    np.testing.assert_array_equal(second.data, [np.finfo(np.float64).max])
    assert (first._version, second._version) == versions
    assert optimizer.step_count == 0


def test_unrepresentable_weight_decay_product_rejects_before_writes():
    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad[...] = [1.0]
    optimizer = Lion([parameter], lr=1e308, weight_decay=1e308)
    version = parameter._version

    with pytest.raises(ValueError, match=r"lr \* weight_decay"):
        optimizer.step()

    np.testing.assert_array_equal(parameter.data, [1.0])
    assert parameter._version == version
    assert optimizer.step_count == 0


def test_nonfinite_parameter_and_gradient_are_rejected():
    parameter = Tensor([1.0], requires_grad=True)
    optimizer = Lion([parameter], lr=0.1)

    parameter.grad[...] = [np.nan]
    with pytest.raises(ValueError, match="gradient.*finite"):
        optimizer.step()

    parameter.grad[...] = [1.0]
    parameter.data[...] = [np.inf]
    with pytest.raises(ValueError, match="parameter 0.*finite"):
        optimizer.step()


def test_gradient_shape_and_dtype_are_validated_before_state_mutation():
    parameter = Tensor([1.0, 2.0], requires_grad=True)
    optimizer = Lion([parameter], lr=0.1)

    parameter.grad = np.array([1.0])
    with pytest.raises(ValueError, match="gradient.*shape mismatch"):
        optimizer.step()
    assert optimizer.step_count == 0

    parameter.grad = np.array([True, False])
    with pytest.raises(TypeError, match="real numeric dtype"):
        optimizer.step()
    assert optimizer.step_count == 0


def test_float32_gradient_is_supported_without_changing_parameter_storage_dtype():
    parameter = Tensor([1.0, 2.0], requires_grad=True)
    parameter.grad = np.array([1.0, -1.0], dtype=np.float32)
    optimizer = Lion([parameter], lr=0.1)

    optimizer.step()

    assert parameter.data.dtype == np.float64
    np.testing.assert_array_equal(parameter.data, [0.9, 2.1])


def test_longdouble_gradient_that_cannot_fit_float64_is_rejected_transactionally():
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("platform longdouble has no wider finite range")
    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad = np.array(
        [np.longdouble(np.finfo(np.float64).max) * np.longdouble(2.0)],
        dtype=np.longdouble,
    )
    optimizer = Lion([parameter], lr=0.1)
    version = parameter._version

    with pytest.raises(ValueError, match="not representable as float64"):
        optimizer.step()

    np.testing.assert_array_equal(parameter.data, [1.0])
    assert parameter._version == version
    assert optimizer.step_count == 0


def test_zero_grad_supports_inplace_and_set_to_none_modes():
    parameter = Tensor([1.0, 2.0], requires_grad=True)
    parameter.grad[...] = [3.0, 4.0]
    optimizer = Lion([parameter])
    gradient = parameter.grad

    assert optimizer.zero_grad() is optimizer
    assert parameter.grad is gradient
    np.testing.assert_array_equal(parameter.grad, [0.0, 0.0])

    parameter.grad[...] = [5.0, 6.0]
    assert optimizer.zero_grad(set_to_none=True) is optimizer
    assert parameter.grad is None


def test_zero_grad_preflights_all_writable_gradient_buffers():
    first = Tensor([1.0], requires_grad=True)
    second = Tensor([2.0], requires_grad=True)
    first.grad[...] = [3.0]
    second.grad[...] = [4.0]
    second.grad.flags.writeable = False
    optimizer = Lion([first, second])

    with pytest.raises(ValueError, match="parameter 1 must be writeable"):
        optimizer.zero_grad()

    np.testing.assert_array_equal(first.grad, [3.0])
    np.testing.assert_array_equal(second.grad, [4.0])


def test_zero_grad_flag_validation_precedes_mutation():
    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad[...] = [2.0]
    optimizer = Lion([parameter])

    with pytest.raises(TypeError, match="set_to_none must be a boolean"):
        optimizer.zero_grad(set_to_none=1)

    np.testing.assert_array_equal(parameter.grad, [2.0])


def test_state_dict_round_trip_restores_hyperparameters_counters_and_momentum():
    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad[...] = [2.0]
    source = Lion([parameter], lr=0.02, betas=(0.8, 0.95), weight_decay=0.1)
    source.step()
    saved = source.state_dict()

    other_parameter = Tensor([7.0], requires_grad=True)
    restored = Lion([other_parameter], lr=0.5, betas=(0.1, 0.2), weight_decay=0.0)
    returned = restored.load_state_dict(saved)

    assert returned is restored
    assert restored.lr == pytest.approx(0.02)
    assert restored.betas == pytest.approx((0.8, 0.95))
    assert restored.weight_decay == pytest.approx(0.1)
    assert restored.step_count == 1
    assert restored.state_dict()["steps"] == [1]
    np.testing.assert_array_equal(
        restored.state_dict()["momentum"][0], saved["momentum"][0]
    )
    np.testing.assert_array_equal(other_parameter.data, [7.0])


def test_state_dict_returns_independent_nested_copies():
    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad[...] = [2.0]
    optimizer = Lion([parameter])
    optimizer.step()

    state = optimizer.state_dict()
    state["momentum"][0][...] = 123.0
    state["steps"][0] = 999

    fresh = optimizer.state_dict()
    assert fresh["steps"] == [1]
    assert not np.array_equal(fresh["momentum"][0], state["momentum"][0])


def test_rejected_state_load_is_fully_transactional():
    first = Tensor([1.0], requires_grad=True)
    second = Tensor([2.0], requires_grad=True)
    optimizer = Lion([first, second], lr=0.03, betas=(0.8, 0.9), weight_decay=0.2)
    first.grad[...] = [1.0]
    second.grad[...] = [-1.0]
    optimizer.step()
    before = optimizer.state_dict()

    bad = copy.deepcopy(before)
    bad["lr"] = 0.7
    bad["betas"] = (0.1, 0.2)
    bad["weight_decay"] = 0.9
    bad["momentum"][1] = np.zeros((2,), dtype=np.float64)

    with pytest.raises(ValueError, match=r"momentum\[1\] shape mismatch"):
        optimizer.load_state_dict(bad)

    after = optimizer.state_dict()
    assert after["lr"] == before["lr"]
    assert after["betas"] == before["betas"]
    assert after["weight_decay"] == before["weight_decay"]
    assert after["step_count"] == before["step_count"]
    assert after["steps"] == before["steps"]
    for actual, expected in zip(after["momentum"], before["momentum"]):
        np.testing.assert_array_equal(actual, expected)


def test_state_load_rejects_step_counter_inconsistency():
    parameter = Tensor([1.0], requires_grad=True)
    optimizer = Lion([parameter])
    state = optimizer.state_dict()
    state["step_count"] = 1
    state["steps"] = [2]

    with pytest.raises(ValueError, match="cannot exceed step_count"):
        optimizer.load_state_dict(state)


def test_dynamic_hyperparameter_properties_validate_before_assignment():
    optimizer = Lion([Tensor([1.0], requires_grad=True)])

    optimizer.lr = np.float32(0.2)
    optimizer.betas = (np.float32(0.7), np.float64(0.8))
    optimizer.weight_decay = np.float32(0.3)
    assert optimizer.lr == pytest.approx(0.2)
    assert optimizer.betas == pytest.approx((0.7, 0.8))
    assert optimizer.weight_decay == pytest.approx(0.3)

    baseline = (optimizer.lr, optimizer.betas, optimizer.weight_decay)
    with pytest.raises(ValueError, match="positive"):
        optimizer.lr = 0.0
    with pytest.raises(ValueError, match="less than 1.0"):
        optimizer.betas = (1.0, 0.9)
    with pytest.raises(ValueError, match="at least 0.0"):
        optimizer.weight_decay = -1.0
    assert (optimizer.lr, optimizer.betas, optimizer.weight_decay) == baseline


def test_constructor_materializes_generator_once_and_rejects_duplicates_and_non_tensors():
    first = Tensor([1.0], requires_grad=True)
    second = Tensor([2.0], requires_grad=True)
    visits = []

    def parameters():
        visits.append("start")
        yield first
        yield second

    optimizer = Lion(parameters())
    assert visits == ["start"]
    assert optimizer.parameters == [first, second]

    with pytest.raises(ValueError, match="duplicate"):
        Lion([first, first])
    with pytest.raises(TypeError, match="parameter 1 must be a Tensor"):
        Lion([first, object()])


def test_parameter_list_replacement_reorder_and_shape_drift_are_rejected():
    first = Tensor([1.0], requires_grad=True)
    second = Tensor([2.0], requires_grad=True)
    optimizer = Lion([first, second])

    optimizer.parameters = list(optimizer.parameters)
    with pytest.raises(RuntimeError, match="list was replaced"):
        optimizer.step()

    optimizer = Lion([first, second])
    optimizer.parameters[:] = [second, first]
    with pytest.raises(RuntimeError, match="identity/order changed"):
        optimizer.step()

    optimizer = Lion([first])
    first.data = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="parameter shape changed"):
        optimizer.step()


def test_successful_parameter_write_keeps_stale_graph_invalid_even_if_value_is_restored_later():
    parameter = Tensor(2.0, requires_grad=True)
    old_loss = parameter * parameter
    parameter.grad[...] = 1.0
    optimizer = Lion([parameter], lr=0.1)

    optimizer.step()
    parameter.data[...] = 2.0

    with pytest.raises(RuntimeError, match="modified after forward"):
        old_loss.backward()


def test_optimizer_does_not_consume_numpy_global_rng():
    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad[...] = [1.0]
    optimizer = Lion([parameter], lr=0.1)
    np.random.seed(1234)
    before = np.random.get_state()

    optimizer.step()
    optimizer.state_dict()
    optimizer.zero_grad()

    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
