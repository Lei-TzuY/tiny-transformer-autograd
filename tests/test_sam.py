import copy
import threading

import numpy as np
import pytest

from engine.optim import Adam, AdamW, SGD
from engine.sam import SAM
from engine.tensor import Tensor


def _assert_nested_equal(left, right):
    assert type(left) is type(right)
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
        return
    if isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_nested_equal(a, b)
        return
    if isinstance(left, np.ndarray):
        assert left.dtype == right.dtype
        assert left.shape == right.shape
        np.testing.assert_array_equal(left, right)
        return
    assert left == right


def test_sgd_two_phase_sam_has_exact_perturbation_and_update():
    parameter = Tensor([1.0, 2.0], requires_grad=True)
    parameter.grad[...] = [3.0, 4.0]
    inner = SGD([parameter], lr=0.1)
    optimizer = SAM(inner, rho=0.5)
    version = parameter._version

    assert optimizer.first_step() is optimizer
    np.testing.assert_allclose(parameter.data, [1.3, 2.4], rtol=0.0, atol=1e-15)
    assert optimizer.phase == "perturbed"
    assert optimizer.step_count == 0
    assert parameter._version == version + 1

    optimizer.zero_grad()
    parameter.grad[...] = [1.0, 2.0]
    assert optimizer.second_step() is None

    np.testing.assert_allclose(parameter.data, [0.9, 1.8], rtol=0.0, atol=1e-15)
    assert optimizer.phase == "ready"
    assert optimizer.step_count == 1
    # first perturbation, base restore, then SGD's in-place update
    assert parameter._version == version + 3


@pytest.mark.parametrize("factory", [Adam, AdamW])
def test_adam_family_state_advances_only_on_second_step(factory):
    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad[...] = [2.0]
    inner = factory([parameter], lr=0.01)
    optimizer = SAM(inner, rho=0.1)

    optimizer.first_step()
    assert inner.t == 0
    assert inner._steps == [0]

    optimizer.zero_grad()
    parameter.grad[...] = [0.5]
    optimizer.second_step()

    assert inner.t == 1
    assert inner._steps == [1]
    assert optimizer.step_count == 1


def test_extreme_finite_gradients_normalize_without_norm_overflow():
    parameter = Tensor([0.0, 0.0], requires_grad=True)
    parameter.grad[...] = [1.3e308, -1.3e308]
    optimizer = SAM(SGD([parameter], lr=0.1), rho=0.5)

    with np.errstate(all="raise"):
        optimizer.first_step()

    expected = 0.5 / np.sqrt(2.0)
    np.testing.assert_allclose(
        parameter.data, [expected, -expected], rtol=1e-15, atol=0.0
    )
    optimizer.restore()
    np.testing.assert_array_equal(parameter.data, [0.0, 0.0])


def test_zero_rho_enters_second_pass_without_spurious_tensor_write():
    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad[...] = [4.0]
    optimizer = SAM(SGD([parameter], lr=0.1), rho=0.0)
    version = parameter._version

    optimizer.first_step()

    np.testing.assert_array_equal(parameter.data, [1.0])
    assert parameter._version == version
    assert optimizer.phase == "perturbed"

    optimizer.zero_grad()
    parameter.grad[...] = [2.0]
    optimizer.second_step()
    np.testing.assert_allclose(parameter.data, [0.8], rtol=0.0, atol=1e-15)


def test_first_step_requires_at_least_one_present_gradient():
    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad = None
    optimizer = SAM(SGD([parameter]))

    with pytest.raises(ValueError, match="requires at least one gradient"):
        optimizer.first_step()

    assert optimizer.phase == "ready"
    assert optimizer.step_count == 0


@pytest.mark.parametrize(
    "gradient, error, message",
    [
        ([1.0], TypeError, "must be a NumPy array"),
        (np.ones((2,), dtype=np.float64), ValueError, "shape mismatch"),
        (np.array([1 + 2j]), TypeError, "real numeric dtype"),
        (np.array([np.nan]), ValueError, "only finite values"),
        (np.array([np.inf]), ValueError, "only finite values"),
    ],
)
def test_first_step_rejects_malformed_gradients_without_parameter_write(
    gradient, error, message
):
    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad = gradient
    optimizer = SAM(SGD([parameter]), rho=0.1)
    version = parameter._version

    with pytest.raises(error, match=message):
        optimizer.first_step()

    np.testing.assert_array_equal(parameter.data, [1.0])
    assert parameter._version == version
    assert optimizer.phase == "ready"


def test_first_step_preflights_all_destination_writability():
    first = Tensor([1.0], requires_grad=True)
    second = Tensor([2.0], requires_grad=True)
    first.grad[...] = [1.0]
    second.grad[...] = [1.0]
    optimizer = SAM(SGD([first, second]), rho=0.2)
    first_version = first._version
    second_version = second._version
    second.data.setflags(write=False)

    with pytest.raises(ValueError, match="parameter 1 must be writeable"):
        optimizer.first_step()

    np.testing.assert_array_equal(first.data, [1.0])
    np.testing.assert_array_equal(second.data, [2.0])
    assert first._version == first_version
    assert second._version == second_version
    assert optimizer.phase == "ready"


def test_first_step_rejects_parameter_addition_overflow_before_any_write():
    first = Tensor([1.0], requires_grad=True)
    second = Tensor([np.finfo(np.float64).max], requires_grad=True)
    first.grad[...] = [0.0]
    second.grad[...] = [1.0]
    optimizer = SAM(SGD([first, second]), rho=np.finfo(np.float64).max)
    first_version = first._version
    second_version = second._version

    with np.errstate(all="raise"):
        with pytest.raises(ValueError, match="would make parameter 1 non-finite"):
            optimizer.first_step()

    np.testing.assert_array_equal(first.data, [1.0])
    np.testing.assert_array_equal(second.data, [np.finfo(np.float64).max])
    assert first._version == first_version
    assert second._version == second_version


def test_second_step_requires_a_completed_first_step():
    parameter = Tensor([1.0], requires_grad=True)
    optimizer = SAM(SGD([parameter]))

    with pytest.raises(RuntimeError, match="requires phase 'perturbed'"):
        optimizer.second_step()


def test_second_step_detects_parameter_drift_and_restore_recovers_base():
    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad[...] = [1.0]
    optimizer = SAM(SGD([parameter]), rho=0.2)
    optimizer.first_step()
    parameter.data[...] = [9.0]

    with pytest.raises(RuntimeError, match="changed before second_step"):
        optimizer.second_step()

    assert optimizer.phase == "perturbed"
    optimizer.restore()
    np.testing.assert_array_equal(parameter.data, [1.0])
    assert optimizer.phase == "ready"


def test_restore_recovers_shape_and_read_only_storage_after_aborted_second_pass():
    parameter = Tensor([1.0, 2.0], requires_grad=True)
    parameter.grad[...] = [3.0, 4.0]
    optimizer = SAM(SGD([parameter]), rho=0.5)
    optimizer.first_step()

    parameter.data = [7.0]
    parameter.data.setflags(write=False)
    optimizer.restore()

    np.testing.assert_array_equal(parameter.data, [1.0, 2.0])
    assert parameter.data.flags.writeable
    assert optimizer.phase == "ready"


def test_second_step_failure_rolls_inner_state_and_parameters_back_to_perturbed_point():
    class FailingSGD(SGD):
        def step(self):
            self._v[0][...] = [7.0]
            self.parameters[0].data[...] = [123.0]
            raise RuntimeError("synthetic inner failure")

    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad[...] = [2.0]
    inner = FailingSGD([parameter], lr=0.1, momentum=0.9)
    optimizer = SAM(inner, rho=0.2)
    optimizer.first_step()
    perturbed = parameter.data.copy()
    optimizer.zero_grad()
    parameter.grad[...] = [1.0]
    state_before = copy.deepcopy(inner.state_dict())

    with pytest.raises(RuntimeError, match="synthetic inner failure"):
        optimizer.second_step()

    assert optimizer.phase == "perturbed"
    assert optimizer.step_count == 0
    np.testing.assert_array_equal(parameter.data, perturbed)
    _assert_nested_equal(inner.state_dict(), state_before)
    optimizer.restore()
    np.testing.assert_array_equal(parameter.data, [1.0])


def test_second_step_rejects_silent_nonfinite_inner_result_and_remains_retryable():
    class NonfiniteSGD(SGD):
        def step(self):
            self.parameters[0].data[...] = [np.inf]
            return None

    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad[...] = [1.0]
    optimizer = SAM(NonfiniteSGD([parameter]), rho=0.1)
    optimizer.first_step()
    perturbed = parameter.data.copy()
    optimizer.zero_grad()
    parameter.grad[...] = [1.0]

    with pytest.raises(ValueError, match="non-finite parameter"):
        optimizer.second_step()

    assert optimizer.phase == "perturbed"
    assert optimizer.step_count == 0
    np.testing.assert_array_equal(parameter.data, perturbed)


def test_second_step_rejects_missing_neighbourhood_gradient_without_restoring_base():
    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad[...] = [1.0]
    optimizer = SAM(SGD([parameter]), rho=0.1)
    optimizer.first_step()
    perturbed = parameter.data.copy()
    parameter.grad = None

    with pytest.raises(ValueError, match="requires at least one gradient"):
        optimizer.second_step()

    np.testing.assert_array_equal(parameter.data, perturbed)
    assert optimizer.phase == "perturbed"


def test_zero_grad_forwards_during_both_phases():
    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad[...] = [3.0]
    gradient = parameter.grad
    optimizer = SAM(SGD([parameter]), rho=0.1)

    optimizer.zero_grad()
    assert parameter.grad is gradient
    np.testing.assert_array_equal(parameter.grad, [0.0])

    parameter.grad[...] = [1.0]
    optimizer.first_step()
    optimizer.zero_grad(set_to_none=True)
    assert parameter.grad is None
    optimizer.restore()


def test_state_dict_round_trip_restores_sam_and_inner_optimizer_state():
    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad[...] = [1.0]
    inner = Adam([parameter], lr=0.01)
    optimizer = SAM(inner, rho=0.2)
    optimizer.first_step()
    optimizer.zero_grad()
    parameter.grad[...] = [0.5]
    optimizer.second_step()
    saved = optimizer.state_dict()

    optimizer.rho = 0.9
    parameter.grad[...] = [1.0]
    optimizer.first_step()
    optimizer.zero_grad()
    parameter.grad[...] = [1.0]
    optimizer.second_step()
    assert optimizer.step_count == 2

    assert optimizer.load_state_dict(saved) is optimizer
    assert optimizer.rho == 0.2
    assert optimizer.step_count == 1
    assert inner.t == 1
    assert inner._steps == [1]


def test_state_dict_returns_independent_inner_state_copy():
    parameter = Tensor([1.0], requires_grad=True)
    optimizer = SAM(Adam([parameter]))

    state = optimizer.state_dict()
    state["optimizer"]["m"][0][...] = [9.0]

    np.testing.assert_array_equal(optimizer.optimizer._m[0], [0.0])


def test_state_save_and_load_are_rejected_while_parameters_are_perturbed():
    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad[...] = [1.0]
    optimizer = SAM(SGD([parameter]), rho=0.1)
    saved = optimizer.state_dict()
    optimizer.first_step()

    with pytest.raises(RuntimeError, match="state_dict requires phase 'ready'"):
        optimizer.state_dict()
    with pytest.raises(RuntimeError, match="load_state_dict requires phase 'ready'"):
        optimizer.load_state_dict(saved)

    optimizer.restore()


def test_malformed_inner_state_load_is_transactional_for_sam_metadata():
    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad[...] = [0.25]
    inner = Adam([parameter], lr=0.01)
    optimizer = SAM(inner, rho=0.3)
    optimizer.first_step()
    optimizer.zero_grad()
    parameter.grad[...] = [0.5]
    optimizer.second_step()

    before = optimizer.state_dict()
    malformed = copy.deepcopy(before)
    malformed["rho"] = 0.8
    malformed["step_count"] = 9
    malformed["optimizer"]["m"][0] = np.zeros((2,), dtype=np.float64)

    with pytest.raises(ValueError, match="shape mismatch"):
        optimizer.load_state_dict(malformed)

    _assert_nested_equal(optimizer.state_dict(), before)


def test_rho_is_dynamically_validated_and_cannot_change_mid_step():
    parameter = Tensor([1.0], requires_grad=True)
    optimizer = SAM(SGD([parameter]), rho=0.05)

    optimizer.rho = np.float32(0.25)
    assert optimizer.rho == 0.25

    for bad in (True, "0.5", 1 + 0j):
        with pytest.raises(TypeError):
            optimizer.rho = bad
    for bad in (-0.1, np.inf, 10**400):
        with pytest.raises(ValueError):
            optimizer.rho = bad

    parameter.grad[...] = [1.0]
    optimizer.first_step()
    with pytest.raises(RuntimeError, match="cannot change while parameters are perturbed"):
        optimizer.rho = 0.1
    optimizer.restore()


def test_parameter_collection_replacement_and_reordering_are_rejected():
    first = Tensor([1.0], requires_grad=True)
    second = Tensor([2.0], requires_grad=True)
    inner = SGD([first, second])
    optimizer = SAM(inner)
    first.grad[...] = [1.0]
    second.grad[...] = [1.0]

    inner.parameters = [first, second]
    with pytest.raises(RuntimeError, match="parameter collection changed"):
        optimizer.first_step()

    inner.parameters = optimizer._parameter_container
    inner.parameters[:] = [second, first]
    with pytest.raises(RuntimeError, match="parameter collection changed"):
        optimizer.first_step()


def test_constructor_rejects_unsupported_duplicate_non_tensor_and_nonfinite_parameters():
    class Unsupported:
        parameters = []

    with pytest.raises(TypeError, match="must be SGD, Adam, or AdamW"):
        SAM(Unsupported())

    parameter = Tensor([1.0], requires_grad=True)
    inner = SGD([parameter])
    inner.parameters.append(parameter)
    with pytest.raises(ValueError, match="must not contain duplicates"):
        SAM(inner)

    inner = SGD([parameter])
    inner.parameters.append(object())
    with pytest.raises(TypeError, match="parameter 1 must be a Tensor"):
        SAM(inner)

    nonfinite = Tensor([1.0], requires_grad=True)
    nonfinite.data[...] = [np.inf]
    with pytest.raises(ValueError, match="only finite values"):
        SAM(SGD([nonfinite]))


def test_empty_optimizer_constructs_but_first_step_requires_gradient():
    optimizer = SAM(Adam([]))
    assert optimizer.parameters == []
    assert optimizer.phase == "ready"

    with pytest.raises(ValueError, match="requires at least one gradient"):
        optimizer.first_step()


def test_second_phase_operations_are_owned_by_first_step_thread():
    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad[...] = [1.0]
    optimizer = SAM(SGD([parameter]), rho=0.1)
    optimizer.first_step()
    errors = []

    def worker():
        for operation in (optimizer.zero_grad, optimizer.second_step, optimizer.restore):
            try:
                operation()
            except BaseException as exc:
                errors.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert len(errors) == 3
    assert all(isinstance(error, RuntimeError) for error in errors)
    assert all("thread that called first_step" in str(error) for error in errors)
    assert optimizer.phase == "perturbed"
    optimizer.restore()


def test_numpy_rng_is_unchanged_by_perturb_and_restore():
    parameter = Tensor([1.0, 2.0], requires_grad=True)
    parameter.grad[...] = [3.0, 4.0]
    optimizer = SAM(SGD([parameter]), rho=0.5)
    np.random.seed(12345)
    before = np.random.get_state()

    optimizer.first_step()
    optimizer.restore()

    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
