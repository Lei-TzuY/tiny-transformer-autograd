import numpy as np
import pytest

from engine.optim import AdamW, SGD
from engine.trainability import freeze_, set_trainable_, unfreeze_
from engine.tensor import Tensor
from nn.module import Module


class PairModule(Module):
    def __init__(self):
        self.left = Tensor(2.0, requires_grad=True)
        self.right = Tensor(3.0, requires_grad=True)


def test_freeze_clears_gradient_and_bumps_version_without_touching_data():
    parameter = Tensor([1.0, -2.0], requires_grad=True)
    parameter.grad[:] = [7.0, 8.0]
    before = parameter.data.copy()
    version = parameter._version

    assert freeze_(parameter) == 1

    assert parameter.requires_grad is False
    assert parameter.grad is None
    np.testing.assert_array_equal(parameter.data, before)
    assert parameter._version == version + 1


def test_unfreeze_rebuilds_gradient_on_next_backward():
    parameter = Tensor(3.0, requires_grad=True)
    freeze_(parameter)
    frozen_version = parameter._version

    assert unfreeze_(parameter) == 1
    assert parameter.requires_grad is True
    assert parameter.grad is None
    assert parameter._version == frozen_version + 1

    loss = parameter * parameter
    loss.backward()
    assert parameter.grad == pytest.approx(6.0)


def test_noop_unfreeze_preserves_existing_gradient_and_version():
    parameter = Tensor([1.0, 2.0], requires_grad=True)
    original_grad = parameter.grad
    original_grad[:] = [4.0, 5.0]
    version = parameter._version

    assert unfreeze_(parameter) == 0

    assert parameter.grad is original_grad
    np.testing.assert_array_equal(parameter.grad, [4.0, 5.0])
    assert parameter._version == version


def test_repeated_freeze_removes_stale_gradient_without_false_version_bump():
    parameter = Tensor(2.0, requires_grad=True)
    freeze_(parameter)
    version = parameter._version
    parameter.grad = np.array(123.0)

    assert freeze_(parameter) == 0

    assert parameter.requires_grad is False
    assert parameter.grad is None
    assert parameter._version == version


def test_set_trainable_returns_number_of_actual_transitions():
    active = Tensor(1.0, requires_grad=True)
    frozen = Tensor(2.0, requires_grad=False)

    assert set_trainable_([active, frozen], False) == 1
    assert active.requires_grad is False
    assert frozen.requires_grad is False

    assert set_trainable_([active, frozen], True) == 2
    assert active.requires_grad is True
    assert frozen.requires_grad is True


def test_frozen_parameter_disappears_from_module_parameters_and_can_be_restored():
    module = PairModule()
    saved = list(module.parameters())
    assert saved == [module.left, module.right]

    freeze_(saved)
    assert module.parameters() == []

    unfreeze_(saved)
    assert module.parameters() == [module.left, module.right]


def test_optimizer_bound_before_freeze_skips_parameter_and_weight_decay():
    parameter = Tensor(10.0, requires_grad=True)
    optimizer = AdamW([parameter], lr=0.1, weight_decay=0.5)
    parameter.grad = np.array(4.0)
    freeze_(parameter)
    before = parameter.data.copy()
    version = parameter._version

    optimizer.step()

    np.testing.assert_array_equal(parameter.data, before)
    assert parameter._version == version
    assert optimizer._steps == [0]


def test_sgd_bound_before_freeze_skips_existing_momentum_update():
    parameter = Tensor(5.0, requires_grad=True)
    optimizer = SGD([parameter], lr=0.1, momentum=0.9)
    parameter.grad = np.array(2.0)
    optimizer.step()
    velocity = optimizer._v[0].copy()

    parameter.grad = np.array(9.0)
    freeze_(parameter)
    before = parameter.data.copy()
    optimizer.step()

    np.testing.assert_array_equal(parameter.data, before)
    np.testing.assert_array_equal(optimizer._v[0], velocity)


def test_graph_built_before_freeze_is_rejected_as_stale():
    left = Tensor(2.0, requires_grad=True)
    right = Tensor(3.0, requires_grad=True)
    output = left * right
    left_grad = left.grad.copy()
    right_grad = right.grad.copy()

    freeze_(left)

    with pytest.raises(RuntimeError, match="tensor data was modified after forward"):
        output.backward()

    assert left.grad is None
    np.testing.assert_array_equal(right.grad, right_grad)
    np.testing.assert_array_equal(left_grad, np.zeros_like(left_grad))


def test_graph_built_while_frozen_is_rejected_if_unfrozen_before_backward():
    frozen = Tensor(2.0, requires_grad=True)
    trainable = Tensor(3.0, requires_grad=True)
    freeze_(frozen)
    output = frozen * trainable
    trainable_grad = trainable.grad.copy()

    unfreeze_(frozen)

    with pytest.raises(RuntimeError, match="tensor data was modified after forward"):
        output.backward()

    assert frozen.grad is None
    np.testing.assert_array_equal(trainable.grad, trainable_grad)


def test_forward_and_backward_after_freeze_only_updates_trainable_peer():
    frozen = Tensor(2.0, requires_grad=True)
    peer = Tensor(3.0, requires_grad=True)
    freeze_(frozen)

    output = frozen * peer
    output.backward()

    assert frozen.grad is None
    assert peer.grad == pytest.approx(2.0)


def test_unfreeze_then_fresh_forward_restores_gradient_flow():
    parameter = Tensor(2.0, requires_grad=True)
    freeze_(parameter)
    unfreeze_(parameter)

    output = parameter * 4.0
    output.backward()

    assert parameter.grad == pytest.approx(4.0)


def test_one_shot_generator_is_materialized_once():
    parameters = [Tensor(1.0, requires_grad=True), Tensor(2.0, requires_grad=True)]
    seen = []

    def values():
        for parameter in parameters:
            seen.append(id(parameter))
            yield parameter

    assert freeze_(values()) == 2
    assert seen == [id(parameter) for parameter in parameters]


def test_empty_collection_is_a_valid_noop():
    assert freeze_([]) == 0
    assert unfreeze_([]) == 0


def test_trainability_helpers_do_not_consume_numpy_rng():
    parameter = Tensor(1.0, requires_grad=True)
    np.random.seed(2026)
    expected = np.random.get_state()

    freeze_(parameter)
    unfreeze_(parameter)

    actual = np.random.get_state()
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2:] == expected[2:]
