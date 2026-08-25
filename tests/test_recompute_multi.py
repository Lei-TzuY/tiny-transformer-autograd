"""Regression tests for multi-output activation recomputation."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import engine.ops as ops
from engine.grad_mode import no_grad
from engine.recompute import recompute
from engine.tensor import Tensor
from nn.layers import Dropout, Linear


class TestMultiOutputRecompute:
    def test_matches_plain_forward_and_backward_with_parameters(self):
        np.random.seed(11)
        layer = Linear(3, 3)
        data = np.random.randn(4, 3)

        def section(inp):
            hidden = ops.tanh(layer(inp))
            return hidden, hidden * hidden

        plain_input = Tensor(data, requires_grad=True)
        plain_a, plain_b = section(plain_input)
        plain_values = (plain_a.data.copy(), plain_b.data.copy())
        ops.sum(plain_a * 2.0 + plain_b * 3.0).backward()
        expected_input_grad = plain_input.grad.copy()
        expected_weight_grad = layer.weight.grad.copy()
        expected_bias_grad = layer.bias.grad.copy()

        layer.zero_grad()
        checkpointed_input = Tensor(data, requires_grad=True)
        checkpointed_a, checkpointed_b = recompute(section, checkpointed_input)
        ops.sum(checkpointed_a * 2.0 + checkpointed_b * 3.0).backward()

        np.testing.assert_allclose(checkpointed_a.data, plain_values[0], atol=1e-15)
        np.testing.assert_allclose(checkpointed_b.data, plain_values[1], atol=1e-15)
        np.testing.assert_allclose(
            checkpointed_input.grad, expected_input_grad, atol=1e-15
        )
        np.testing.assert_allclose(layer.weight.grad, expected_weight_grad, atol=1e-15)
        np.testing.assert_allclose(layer.bias.grad, expected_bias_grad, atol=1e-15)

    def test_replays_once_when_multiple_outputs_contribute(self):
        calls = {"count": 0}
        x = Tensor([[1.0, 2.0]], requires_grad=True)

        def section(inp):
            calls["count"] += 1
            squared = inp * inp
            return squared, squared + inp

        first, second = recompute(section, x)
        assert calls["count"] == 1

        ops.sum(first + second).backward()

        assert calls["count"] == 2
        np.testing.assert_allclose(x.grad, [[5.0, 9.0]], atol=1e-15)

    def test_unused_output_contributes_zero_cotangent(self):
        x = Tensor([[2.0, 3.0]], requires_grad=True)
        squared, cubed = recompute(lambda inp: (inp * inp, inp ** 3), x)

        ops.sum(squared).backward()

        np.testing.assert_allclose(x.grad, [[4.0, 6.0]], atol=1e-15)
        assert cubed.shape == x.shape

    def test_outputs_support_separate_repeated_backward_calls(self):
        x = Tensor([[2.0]], requires_grad=True)
        squared, cubed = recompute(lambda inp: (inp * inp, inp ** 3), x)

        ops.sum(squared).backward()
        np.testing.assert_allclose(x.grad, [[4.0]], atol=1e-15)

        ops.sum(cubed).backward()
        np.testing.assert_allclose(x.grad, [[16.0]], atol=1e-15)

    def test_replays_rng_once_and_restores_the_random_stream(self):
        drop = Dropout(0.5)
        x = Tensor(np.arange(1.0, 13.0).reshape(3, 4), requires_grad=True)

        np.random.seed(17)
        first, second = recompute(
            lambda inp: (drop(inp), drop(inp * 2.0)),
            x,
        )
        state_before = np.random.get_state()

        ops.sum(first + second).backward()

        state_after = np.random.get_state()
        np.testing.assert_array_equal(state_before[1], state_after[1])
        assert state_before[2] == state_after[2]
        expected = first.data / x.data + second.data / x.data
        np.testing.assert_allclose(x.grad, expected, atol=1e-15)

    def test_is_a_plain_tuple_call_under_no_grad(self):
        x = Tensor([[1.0, 2.0]], requires_grad=True)

        with no_grad():
            outputs = recompute(lambda inp: (inp * 2.0, inp + 1.0), x)

        assert isinstance(outputs, tuple)
        assert len(outputs) == 2
        assert all(not value.requires_grad for value in outputs)
        assert all(value._children == set() for value in outputs)

    @pytest.mark.parametrize(
        ("function", "error"),
        [
            (lambda inp: (), ValueError),
            (lambda inp: (inp, np.ones(1)), TypeError),
            (lambda inp: [inp, inp], TypeError),
        ],
    )
    def test_validates_multi_output_structure(self, function, error):
        with pytest.raises(error):
            recompute(function, Tensor([1.0], requires_grad=True))

    def test_replay_must_preserve_output_structure(self):
        calls = {"count": 0}
        x = Tensor([2.0], requires_grad=True)

        def unstable(inp):
            calls["count"] += 1
            if calls["count"] == 1:
                return inp * inp, inp + 1.0
            return (inp * inp,)

        first, _ = recompute(unstable, x)
        with pytest.raises(RuntimeError, match="different output structure"):
            ops.sum(first).backward()

    def test_replay_must_preserve_output_shapes(self):
        calls = {"count": 0}
        x = Tensor([[1.0, 2.0]], requires_grad=True)

        def unstable(inp):
            calls["count"] += 1
            if calls["count"] == 1:
                return inp, inp * 2.0
            return inp.reshape((-1,)), inp * 2.0

        first, second = recompute(unstable, x)
        with pytest.raises(RuntimeError, match="changed shape"):
            ops.sum(first + second).backward()
