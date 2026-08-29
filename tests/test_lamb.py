import copy

import numpy as np
import pytest

from engine.lamb import LAMB
from engine.tensor import Tensor


_MAX = np.finfo(np.float64).max
_TINY = np.nextafter(0.0, 1.0)


def test_first_step_trust_ratio_hand_calculation():
    parameter = Tensor([3.0, 4.0], requires_grad=True)
    parameter.grad = np.array([1.0, 0.0])
    optimizer = LAMB(
        parameter,
        lr=0.1,
        betas=(0.9, 0.999),
        eps=1.0,
        weight_decay=0.0,
    )

    optimizer.step()

    # First corrected Adam update is [0.5, 0]. Its norm is 0.5 while
    # ||parameter|| is 5, so trust ratio is 10 and the trusted step is [0.5, 0].
    np.testing.assert_allclose(parameter.data, [2.5, 4.0], rtol=0.0, atol=1e-15)
    assert optimizer.steps == (1,)


def test_pure_weight_decay_is_inside_trust_ratio_update():
    parameter = Tensor([3.0, 4.0], requires_grad=True)
    parameter.grad = np.zeros(2)
    optimizer = LAMB(
        parameter,
        lr=0.1,
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=0.25,
    )

    optimizer.step()

    # u = wd*p is parallel to p, so ||p||/||u|| cancels wd exactly.
    np.testing.assert_allclose(parameter.data, [2.7, 3.6], rtol=1e-15, atol=0.0)
    assert optimizer.steps == (1,)


def test_two_step_state_matches_bias_corrected_ema_definition():
    parameter = Tensor([8.0, 9.0], requires_grad=True)
    optimizer = LAMB(
        parameter,
        lr=1e-6,
        betas=(0.5, 0.5),
        eps=1e-6,
        weight_decay=0.0,
    )

    parameter.grad = np.array([2.0, 4.0])
    optimizer.step()
    parameter.grad = np.array([4.0, 2.0])
    optimizer.step()

    state = optimizer.state_dict()["states"][0]
    np.testing.assert_allclose(state["m"], [10.0 / 3.0, 8.0 / 3.0])
    physical_v = (state["v_scale"] ** 2) * state["v"]
    np.testing.assert_allclose(physical_v, [12.0, 8.0], rtol=2e-15, atol=0.0)
    assert state["step"] == 2


def test_scalar_parameter_is_supported_without_slice_indexing():
    parameter = Tensor(2.0, requires_grad=True)
    parameter.grad = np.array(7.0)
    optimizer = LAMB(
        parameter,
        lr=0.1,
        betas=(0.0, 0.0),
        eps=_TINY,
        weight_decay=0.0,
    )

    optimizer.step()

    assert parameter.shape == ()
    assert parameter.data == pytest.approx(1.8)
    optimizer.zero_grad()
    assert parameter.grad.shape == ()
    assert parameter.grad == 0.0


def test_zero_parameter_norm_uses_trust_ratio_one():
    parameter = Tensor([0.0], requires_grad=True)
    parameter.grad = np.array([2.0])
    optimizer = LAMB(
        parameter,
        lr=0.1,
        betas=(0.0, 0.0),
        eps=1.0,
        weight_decay=0.0,
    )

    optimizer.step()

    # Adam update is 2/(2+1)=2/3 and trust ratio falls back to 1.
    np.testing.assert_allclose(parameter.data, [-1.0 / 15.0], rtol=1e-15, atol=0.0)


def test_grad_none_skips_moments_and_weight_decay():
    parameter = Tensor([3.0, 4.0], requires_grad=True)
    optimizer = LAMB(parameter, lr=0.1, weight_decay=10.0)
    parameter.grad = None
    before = parameter.data.copy()

    optimizer.step()

    np.testing.assert_array_equal(parameter.data, before)
    assert optimizer.steps == (0,)


def test_zero_gradient_without_weight_decay_advances_state_without_data_write():
    parameter = Tensor([3.0, 4.0], requires_grad=True)
    parameter.grad = np.zeros(2)
    optimizer = LAMB(parameter, lr=0.1, weight_decay=0.0)
    before_version = parameter._version

    optimizer.step()

    np.testing.assert_array_equal(parameter.data, [3.0, 4.0])
    assert parameter._version == before_version
    assert optimizer.steps == (1,)


def test_true_parameter_norm_overflow_does_not_require_materialized_trust_ratio():
    parameter = Tensor([_MAX, -_MAX], requires_grad=True)
    parameter.grad = np.array([_MAX, -_MAX])
    optimizer = LAMB(
        parameter,
        lr=0.1,
        betas=(0.0, 0.0),
        eps=_TINY,
        weight_decay=0.0,
    )

    with np.errstate(all="raise"):
        optimizer.step()

    expected = np.array([0.9 * _MAX, -0.9 * _MAX])
    np.testing.assert_allclose(parameter.data, expected, rtol=3e-16, atol=0.0)


def test_weight_decay_product_may_overflow_raw_float64_but_trusted_step_remains_valid():
    parameter = Tensor([_MAX, -_MAX], requires_grad=True)
    parameter.grad = np.zeros(2)
    optimizer = LAMB(
        parameter,
        lr=0.1,
        betas=(0.0, 0.0),
        eps=1e-6,
        weight_decay=2.0,
    )

    with np.errstate(all="raise"):
        optimizer.step()

    expected = np.array([0.9 * _MAX, -0.9 * _MAX])
    np.testing.assert_allclose(parameter.data, expected, rtol=3e-16, atol=0.0)


def test_opposite_sign_extreme_gradients_do_not_overflow_first_moment_interpolation():
    parameter = Tensor([5.0, 6.0], requires_grad=True)
    optimizer = LAMB(
        parameter,
        lr=1e-12,
        betas=(0.5, 0.5),
        eps=1.0,
        weight_decay=0.0,
    )
    parameter.grad = np.array([_MAX, -_MAX])
    with np.errstate(all="raise"):
        optimizer.step()
    parameter.grad = np.array([-_MAX, _MAX])
    with np.errstate(all="raise"):
        optimizer.step()

    state = optimizer.state_dict()["states"][0]
    assert np.all(np.isfinite(state["m"]))
    assert np.all(np.isfinite(state["v"]))
    assert np.isfinite(state["v_scale"])


def test_smallest_subnormal_gradient_is_warning_free():
    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad = np.array([_TINY])
    optimizer = LAMB(
        parameter,
        lr=1e-3,
        betas=(0.0, 0.0),
        eps=1.0,
        weight_decay=0.0,
    )

    with np.errstate(all="raise"):
        optimizer.step()

    assert np.isfinite(parameter.data).all()
    assert optimizer.steps == (1,)


def test_zero_grad_set_to_none_matches_optimizer_contract():
    parameter = Tensor([1.0, 2.0], requires_grad=True)
    parameter.grad = np.array([3.0, 4.0])
    optimizer = LAMB(parameter)

    optimizer.zero_grad(set_to_none=True)

    assert parameter.grad is None


def test_state_and_model_reads_are_rng_neutral():
    parameter = Tensor([1.0, 2.0], requires_grad=True)
    parameter.grad = np.array([3.0, 4.0])
    optimizer = LAMB(parameter)
    before = copy.deepcopy(np.random.get_state())

    optimizer.step()
    optimizer.state_dict()

    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_empty_parameter_collection_is_valid():
    optimizer = LAMB([])
    optimizer.step()
    assert optimizer.steps == ()
    assert optimizer.state_dict()["states"] == []
