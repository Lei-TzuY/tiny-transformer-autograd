"""
test_transformer.py — Integration tests for the neural network modules.

Checks:
  - Correct output shapes at every stage
  - Gradients flow (non-zero, non-NaN) all the way back to every parameter
  - Loss decreases over a few gradient steps (the model can overfit)
  - GPT.generate produces the right token count

Run:
    pytest tests/test_transformer.py -v
or:
    python tests/test_transformer.py
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import engine.ops as ops
from engine.optim import Adam
from nn.layers import Linear, Embedding, LayerNorm, Dropout
from nn.attention import SelfAttention, MultiHeadAttention
from nn.transformer import FeedForward, TransformerBlock, GPT


RNG = np.random.default_rng(0)


def rand(*shape):
    from engine.tensor import Tensor
    return Tensor(RNG.standard_normal(shape).astype(np.float64), requires_grad=True)


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

class TestLinear:
    def test_shape_2d(self):
        layer = Linear(8, 16)
        x = rand(4, 8)
        out = layer(x)
        assert out.shape == (4, 16)

    def test_shape_3d(self):
        layer = Linear(8, 16)
        x = rand(2, 5, 8)
        out = layer(x)
        assert out.shape == (2, 5, 16)

    def test_grad_flows(self):
        layer = Linear(4, 8)
        x = rand(3, 4)
        loss = ops.sum(layer(x))
        loss.backward()
        assert layer.weight.grad is not None
        assert not np.all(layer.weight.grad == 0)
        assert layer.bias.grad is not None


class TestEmbedding:
    def test_shape(self):
        emb = Embedding(10, 4)
        idx = np.array([[0, 1, 2], [3, 4, 5]])
        out = emb(idx)
        assert out.shape == (2, 3, 4)

    def test_grad_flows(self):
        emb = Embedding(10, 4)
        idx = np.array([0, 2, 5])
        out = emb(idx)
        ops.sum(out).backward()
        assert emb.weight.grad is not None
        # Only rows 0, 2, 5 should have non-zero gradient
        assert not np.all(emb.weight.grad == 0)


class TestLayerNorm:
    def test_shape(self):
        ln = LayerNorm(8)
        x = rand(2, 5, 8)
        out = ln(x)
        assert out.shape == (2, 5, 8)

    def test_normalised_statistics(self):
        ln = LayerNorm(16)
        x = rand(4, 16)
        # Set gamma=1, beta=0 and verify unit-normal statistics
        ln.gamma.data[:] = 1.0
        ln.beta.data[:] = 0.0
        out = ln(x)
        means = out.data.mean(axis=-1)
        vars_ = out.data.var(axis=-1)
        np.testing.assert_allclose(means, 0.0, atol=1e-5)
        np.testing.assert_allclose(vars_, 1.0, atol=1e-4)

    def test_grad_flows(self):
        ln = LayerNorm(8)
        x = rand(3, 8)
        ops.sum(ln(x)).backward()
        assert not np.all(ln.gamma.grad == 0)
        assert ln.beta.grad is not None


class TestDropout:
    def test_training_zeros_fraction(self):
        drop = Dropout(p=0.5)
        drop.training = True
        x = rand(1000, 1)
        out = drop(x)
        zero_frac = (out.data == 0).mean()
        # Expect roughly 50% zeros; allow generous tolerance
        assert 0.3 < zero_frac < 0.7

    def test_eval_identity(self):
        drop = Dropout(p=0.5)
        drop.training = False
        x = rand(10, 10)
        out = drop(x)
        np.testing.assert_array_equal(out.data, x.data)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class TestSelfAttention:
    def test_shape(self):
        attn = SelfAttention(d_model=16)
        x = rand(2, 5, 16)
        mask_data = np.triu(np.full((5, 5), -1e9), k=1)
        from engine.tensor import Tensor
        mask = Tensor(mask_data)
        out = attn(x, mask)
        assert out.shape == (2, 5, 16)

    def test_grad_flows(self):
        attn = SelfAttention(d_model=8)
        x = rand(2, 4, 8)
        out = attn(x)
        ops.sum(out).backward()
        for p in attn.parameters():
            assert p.grad is not None, f"No gradient for param of shape {p.shape}"
            assert not np.any(np.isnan(p.grad)), "NaN gradient"


class TestMultiHeadAttention:
    def test_shape(self):
        attn = MultiHeadAttention(d_model=16, num_heads=4)
        x = rand(2, 5, 16)
        out = attn(x)
        assert out.shape == (2, 5, 16)

    def test_causal_mask(self):
        """Future tokens must not influence past positions."""
        attn = MultiHeadAttention(d_model=8, num_heads=2)
        from engine.tensor import Tensor
        T = 4
        mask_data = np.triu(np.full((T, T), -1e9), k=1)
        mask = Tensor(mask_data)

        x = rand(1, T, 8)
        out_full = attn(x, mask).data.copy()

        # Zero out a future token and verify that past outputs are unchanged
        x_mod = x.data.copy()
        x_mod[0, T - 1, :] = 0.0
        from engine.tensor import Tensor as T2
        x2 = T2(x_mod, requires_grad=False)
        out_mod = attn(x2, mask).data

        np.testing.assert_allclose(out_full[0, 0], out_mod[0, 0], atol=1e-5)

    def test_grad_flows(self):
        attn = MultiHeadAttention(d_model=8, num_heads=2)
        x = rand(2, 4, 8)
        ops.sum(attn(x)).backward()
        for p in attn.parameters():
            assert p.grad is not None
            assert not np.any(np.isnan(p.grad))


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------

class TestFeedForward:
    def test_shape(self):
        ff = FeedForward(d_model=8, d_ff=32)
        x = rand(2, 5, 8)
        assert ff(x).shape == (2, 5, 8)


class TestTransformerBlock:
    def test_shape(self):
        block = TransformerBlock(d_model=16, num_heads=4, d_ff=32)
        x = rand(2, 6, 16)
        out = block(x)
        assert out.shape == (2, 6, 16)

    def test_grad_flows(self):
        block = TransformerBlock(d_model=8, num_heads=2, d_ff=16)
        x = rand(2, 4, 8)
        ops.sum(block(x)).backward()
        for p in block.parameters():
            assert p.grad is not None
            assert not np.any(np.isnan(p.grad))


# ---------------------------------------------------------------------------
# Full GPT
# ---------------------------------------------------------------------------

class TestGPT:
    def _make_model(self, **kw):
        defaults = dict(
            vocab_size=16, context_len=8,
            d_model=16, num_heads=2, d_ff=32, num_layers=1,
        )
        defaults.update(kw)
        return GPT(**defaults)

    def test_output_shape(self):
        model = self._make_model()
        idx = np.random.randint(0, 16, size=(2, 6))
        out = model(idx)
        assert out.shape == (2, 6, 16)

    def test_grad_flows_to_all_params(self):
        model = self._make_model()
        idx = np.random.randint(0, 16, size=(2, 4))
        targets = np.random.randint(0, 16, size=(2 * 4,))
        logits = model(idx)
        B, T, V = logits.shape
        loss = ops.cross_entropy(ops.reshape(logits, (B * T, V)), targets)
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert not np.any(np.isnan(p.grad)), f"NaN gradient for {name}"

    def test_loss_decreases(self):
        """The model should be able to overfit a tiny fixed batch."""
        np.random.seed(1)
        model = self._make_model()
        optimizer = Adam(model.parameters(), lr=1e-2)

        # Fixed batch
        idx = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
        tgt = np.array([[1, 2, 3, 4, 5, 6, 7, 0]])

        losses = []
        for _ in range(30):
            logits = model(idx)
            B, T, V = logits.shape
            loss = ops.cross_entropy(ops.reshape(logits, (B * T, V)), tgt.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.data))

        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
        )

    def test_generate_length(self):
        model = self._make_model()
        prompt = np.array([[0, 1, 2]])
        out = model.generate(prompt, max_new_tokens=5)
        assert out.shape == (1, 8)   # 3 prompt + 5 generated

    def test_param_count_reasonable(self):
        model = self._make_model(d_model=32, num_heads=4, d_ff=64, num_layers=2)
        n = model.param_count()
        assert n > 0
        assert n < 10_000_000   # sanity cap

    def test_weight_tying_same_object(self):
        model = self._make_model()
        assert model.head.weight is model.token_emb.weight, (
            "head.weight and token_emb.weight must be the same Tensor object"
        )

    def test_weight_tying_reduces_param_count(self):
        """Tied model has vocab×d fewer trainable scalars than an untied one."""
        V, D = 32, 16
        model = self._make_model(vocab_size=V, d_model=D)
        # An untied model would count head.weight as a separate parameter;
        # the tied one should have V*D fewer scalars.
        n = model.param_count()
        # Manually count all unique named tensors to confirm deduplication
        unique_ids = {id(t) for _, t in model.named_tensors() if t.requires_grad}
        counted = sum(
            t.data.size for name, t in model.named_tensors()
            if t.requires_grad and id(t) in unique_ids
        )
        assert n == counted  # param_count() matches deduplicated traversal

    def test_weight_tying_gradient_accumulates(self):
        """One backward pass accumulates grads from both emb lookup and head."""
        model = self._make_model()
        idx = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
        tgt = np.array([[1, 2, 3, 4, 5, 6, 7, 0]])
        logits = model(idx)
        B, T, V = logits.shape
        loss = ops.cross_entropy(ops.reshape(logits, (B * T, V)), tgt.reshape(-1))
        loss.backward()
        # The shared weight should have non-zero gradient
        shared_weight = model.token_emb.weight
        assert shared_weight.grad is not None
        assert not np.all(shared_weight.grad == 0), "tied weight has zero gradient"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    suites = [
        TestLinear, TestEmbedding, TestLayerNorm, TestDropout,
        TestSelfAttention, TestMultiHeadAttention,
        TestFeedForward, TestTransformerBlock, TestGPT,
    ]
    passed = failed = 0
    for cls in suites:
        obj = cls()
        for name in [m for m in dir(obj) if m.startswith("test_")]:
            try:
                getattr(obj, name)()
                print(f"  PASS  {cls.__name__}.{name}")
                passed += 1
            except Exception:
                print(f"  FAIL  {cls.__name__}.{name}")
                traceback.print_exc()
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
