"""
test_recompute.py — Gradient checkpointing (activation recomputation).

The contract is strict: recomputing must change *nothing* observable except
memory and time. These tests pin that down — identical forward values, identical
gradients (with and without dropout), an untouched random stream, and no
retained activations — plus the input validation and the interaction with
no_grad().
"""

import gc
import os
import sys
import weakref

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import engine.ops as ops
from engine.grad_mode import no_grad
from engine.optim import Adam
from engine.recompute import recompute
from engine.tensor import Tensor
from nn.layers import Linear
from nn.transformer import GPT


def _model(**overrides):
    config = dict(
        vocab_size=12, context_len=8,
        d_model=16, num_heads=2, d_ff=32, num_layers=3,
    )
    config.update(overrides)
    np.random.seed(5)
    return GPT(**config)


def _tokens():
    return np.array([[1, 4, 7, 2], [3, 6, 9, 0]], dtype=np.int64)


def _gradients(model):
    return {name: p.grad.copy() for name, p in model.named_parameters()}


class TestPrimitive:
    def test_matches_a_plain_call_forward_and_backward(self):
        np.random.seed(0)
        layer = Linear(4, 3)
        data = np.random.randn(5, 4)

        plain_input = Tensor(data, requires_grad=True)
        ops.sum(ops.gelu(layer(plain_input))).backward()
        plain_weight_grad = layer.weight.grad.copy()
        plain_input_grad = plain_input.grad.copy()

        layer.zero_grad()
        checkpointed_input = Tensor(data, requires_grad=True)
        checkpointed = recompute(
            lambda inp: ops.gelu(layer(inp)), checkpointed_input
        )
        ops.sum(checkpointed).backward()

        np.testing.assert_allclose(
            checkpointed.data, ops.gelu(layer(Tensor(data))).data, atol=1e-15
        )
        np.testing.assert_allclose(layer.weight.grad, plain_weight_grad, atol=1e-15)
        np.testing.assert_allclose(checkpointed_input.grad, plain_input_grad, atol=1e-15)

    def test_does_not_disturb_gradients_from_other_consumers(self):
        """A residual connection reuses its input; the replay must only add."""
        np.random.seed(1)
        layer = Linear(3, 3)
        data = np.random.randn(2, 3)

        plain = Tensor(data, requires_grad=True)
        ops.sum(plain + layer(plain)).backward()
        expected = plain.grad.copy()

        layer.zero_grad()
        shared = Tensor(data, requires_grad=True)
        ops.sum(shared + recompute(layer, shared)).backward()

        np.testing.assert_allclose(shared.grad, expected, atol=1e-15)

    def test_accumulates_over_repeated_backward(self):
        layer = Linear(2, 2)
        x = Tensor(np.ones((1, 2)), requires_grad=True)
        out = ops.sum(recompute(layer, x))

        out.backward()
        first = x.grad.copy()
        out.backward()

        np.testing.assert_allclose(x.grad, 2.0 * first, atol=1e-15)

    def test_releases_the_sections_intermediates(self):
        x = Tensor(np.ones((2, 2)), requires_grad=True)
        captured = {}

        def section(inp):
            hidden = inp * inp
            captured["ref"] = weakref.ref(hidden)
            return hidden + inp

        out = recompute(section, x)
        gc.collect()

        assert captured["ref"]() is None, "recompute retained an intermediate"
        # The gradient still arrives, because backward replays the section.
        ops.sum(out).backward()
        np.testing.assert_allclose(x.grad, [[3.0, 3.0], [3.0, 3.0]], atol=1e-15)

    def test_is_a_plain_call_under_no_grad(self):
        layer = Linear(2, 2)
        x = Tensor(np.ones((1, 2)))
        with no_grad():
            out = recompute(layer, x)
        assert not out.requires_grad
        assert out._children == set()

    def test_leaves_the_random_stream_untouched(self):
        from nn.layers import Dropout

        drop = Dropout(0.5)
        x = Tensor(np.ones((4, 4)), requires_grad=True)
        np.random.seed(3)
        out = ops.sum(recompute(lambda inp: drop(inp) * inp, x))
        state_before = np.random.get_state()

        out.backward()

        state_after = np.random.get_state()
        np.testing.assert_array_equal(state_before[1], state_after[1])
        assert state_before[2] == state_after[2]

    def test_replays_the_same_dropout_mask(self):
        from nn.layers import Dropout

        drop = Dropout(0.5)
        x = Tensor(np.arange(1.0, 17.0).reshape(4, 4), requires_grad=True)
        np.random.seed(4)
        out = recompute(drop, x)
        ops.sum(out).backward()

        # d/dx of (x * mask) is the mask itself, so the gradient must equal the
        # kept-and-rescaled pattern the forward pass produced.
        np.testing.assert_allclose(x.grad, out.data / x.data, atol=1e-12)

    @pytest.mark.parametrize(
        ("call", "error"),
        [
            (lambda: recompute(lambda: None), ValueError),
            (lambda: recompute(lambda inp: inp, np.ones(2)), TypeError),
            (lambda: recompute(lambda inp: inp.data, Tensor([1.0])), TypeError),
        ],
    )
    def test_validates_its_arguments(self, call, error):
        with pytest.raises(error):
            call()


class TestModelIntegration:
    def test_forward_values_are_unchanged(self):
        plain = _model()
        checkpointed = _model(grad_checkpoint=True)
        tokens = _tokens()

        np.testing.assert_array_equal(plain(tokens).data, checkpointed(tokens).data)

    def test_gradients_are_unchanged(self):
        tokens = _tokens()
        targets = np.roll(tokens, -1, axis=1)

        plain = _model()
        ops.cross_entropy(plain(tokens), targets).backward()
        checkpointed = _model(grad_checkpoint=True)
        ops.cross_entropy(checkpointed(tokens), targets).backward()

        plain_grads = _gradients(plain)
        for name, gradient in _gradients(checkpointed).items():
            np.testing.assert_allclose(
                gradient, plain_grads[name], atol=1e-14, err_msg=f"gradient {name}"
            )

    def test_gradients_are_unchanged_with_dropout(self):
        tokens = _tokens()
        targets = np.roll(tokens, -1, axis=1)

        np.random.seed(9)
        plain = _model(dropout=0.3)
        np.random.seed(9)
        ops.cross_entropy(plain(tokens), targets).backward()

        np.random.seed(9)
        checkpointed = _model(dropout=0.3, grad_checkpoint=True)
        np.random.seed(9)
        ops.cross_entropy(checkpointed(tokens), targets).backward()

        plain_grads = _gradients(plain)
        for name, gradient in _gradients(checkpointed).items():
            np.testing.assert_allclose(
                gradient, plain_grads[name], atol=1e-14, err_msg=f"gradient {name}"
            )

    def test_training_trajectories_match(self):
        tokens = _tokens()
        targets = np.roll(tokens, -1, axis=1)
        losses = {}

        for label, flag in (("plain", False), ("checkpointed", True)):
            np.random.seed(2)
            model = _model(dropout=0.2, grad_checkpoint=flag)
            optimizer = Adam(model.parameters(), lr=1e-2)
            np.random.seed(2)
            history = []
            for _ in range(5):
                optimizer.zero_grad()
                loss = ops.cross_entropy(model(tokens), targets)
                loss.backward()
                optimizer.step()
                history.append(float(loss.data))
            losses[label] = history

        np.testing.assert_allclose(
            losses["checkpointed"], losses["plain"], atol=1e-14
        )

    def test_retains_fewer_tensors_after_a_forward_pass(self):
        tokens = _tokens()

        def live_tensors(model):
            gc.collect()
            before = sum(1 for obj in gc.get_objects() if isinstance(obj, Tensor))
            logits = model(tokens)
            gc.collect()
            after = sum(1 for obj in gc.get_objects() if isinstance(obj, Tensor))
            del logits
            return after - before

        plain = live_tensors(_model())
        checkpointed = live_tensors(_model(grad_checkpoint=True))
        assert checkpointed < plain

    def test_toggle_is_not_part_of_the_architecture_config(self):
        model = _model(grad_checkpoint=True)
        assert "grad_checkpoint" not in model.config()
        assert "grad_checkpoint" in repr(model)

        # Flipping it at runtime is supported and changes nothing observable.
        logits = model(_tokens()).data
        model.grad_checkpoint = False
        np.testing.assert_array_equal(model(_tokens()).data, logits)

    def test_works_with_lora_frozen_backbone(self):
        tokens = _tokens()
        targets = np.roll(tokens, -1, axis=1)

        plain = _model(lora_rank=2, lora_alpha=4)
        ops.cross_entropy(plain(tokens), targets).backward()
        checkpointed = _model(lora_rank=2, lora_alpha=4, grad_checkpoint=True)
        ops.cross_entropy(checkpointed(tokens), targets).backward()

        plain_grads = _gradients(plain)
        assert plain_grads and all("lora_" in name for name in plain_grads)
        for name, gradient in _gradients(checkpointed).items():
            np.testing.assert_allclose(
                gradient, plain_grads[name], atol=1e-14, err_msg=f"gradient {name}"
            )

    def test_composes_with_a_padding_mask_and_ignore_index(self):
        tokens = np.array([[1, 4, 7, 0], [3, 6, 0, 0]], dtype=np.int64)
        mask = np.array([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=np.int64)
        targets = np.array([[4, 7, -1, -1], [6, -1, -1, -1]], dtype=np.int64)

        plain = _model()
        ops.cross_entropy(
            plain(tokens, attention_mask=mask), targets, ignore_index=-1
        ).backward()
        checkpointed = _model(grad_checkpoint=True)
        ops.cross_entropy(
            checkpointed(tokens, attention_mask=mask), targets, ignore_index=-1
        ).backward()

        plain_grads = _gradients(plain)
        for name, gradient in _gradients(checkpointed).items():
            np.testing.assert_allclose(
                gradient, plain_grads[name], atol=1e-14, err_msg=f"gradient {name}"
            )

    def test_inference_paths_are_unaffected(self):
        tokens = _tokens()
        model = _model(grad_checkpoint=True)

        with no_grad():
            detached = model(tokens)
        inferred, _ = model.infer(tokens)

        assert not detached.requires_grad
        np.testing.assert_allclose(detached.data, inferred, atol=1e-12)
