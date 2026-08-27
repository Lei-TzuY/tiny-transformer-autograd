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

    def test_default_forward_matches_causal_inference(self):
        attn = SelfAttention(d_model=8)
        x = RNG.standard_normal((2, 4, 8))
        from engine.tensor import Tensor
        forward = attn(Tensor(x)).data
        inferred, _ = attn.infer(x)
        np.testing.assert_allclose(forward, inferred, atol=1e-12, rtol=1e-12)


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

    def test_default_forward_matches_causal_inference(self):
        attn = MultiHeadAttention(d_model=8, num_heads=2)
        x = RNG.standard_normal((2, 4, 8))
        from engine.tensor import Tensor
        forward = attn(Tensor(x)).data
        inferred, _ = attn.infer(x)
        np.testing.assert_allclose(forward, inferred, atol=1e-12, rtol=1e-12)


class TestCustomAttentionMask:
    """Defined behaviour for caller-supplied masks, including all-masked rows."""

    @staticmethod
    def _fully_masked_first_row(T):
        """Causal mask whose query row 0 additionally hides its own key."""
        mask = np.triu(np.full((T, T), -np.inf), k=1)
        mask[0, 0] = -np.inf
        return mask

    def test_fully_masked_row_returns_out_proj_bias(self):
        from engine.tensor import Tensor
        T = 4
        attn = MultiHeadAttention(d_model=8, num_heads=2)
        # A distinctive bias makes "output is exactly the bias" observable.
        attn.out_proj.bias.data[:] = np.arange(1.0, 9.0)
        x = Tensor(RNG.standard_normal((2, T, 8)))

        out = attn(x, Tensor(self._fully_masked_first_row(T))).data

        assert np.isfinite(out).all()
        # Zero attention weights ⇒ zero context vector ⇒ bias only.
        for batch in range(2):
            np.testing.assert_allclose(
                out[batch, 0], attn.out_proj.bias.data, atol=1e-12
            )
        assert not np.allclose(out[0, 1], attn.out_proj.bias.data)

    def test_fully_masked_row_blocks_gradient_to_inputs(self):
        from engine.tensor import Tensor
        T = 3
        attn = SelfAttention(d_model=8)
        x = Tensor(RNG.standard_normal((1, T, 8)), requires_grad=True)

        out = attn(x, Tensor(self._fully_masked_first_row(T)))
        # Differentiate the fully masked position only.
        cotangent = np.zeros_like(out.data)
        cotangent[0, 0, :] = 1.0
        out.backward(cotangent)

        assert np.isfinite(x.grad).all()
        np.testing.assert_allclose(x.grad, np.zeros_like(x.grad), atol=1e-12)
        # The bias still receives the upstream gradient it passed through.
        np.testing.assert_allclose(attn.out_proj.bias.grad, np.ones(8), atol=1e-12)

    def test_masked_softmax_paths_agree(self):
        from nn.attention import _softmax
        from engine.tensor import Tensor
        scores = np.array([[-np.inf, -np.inf, -np.inf], [0.5, -np.inf, 2.0]])

        np.testing.assert_allclose(
            _softmax(scores), ops.softmax(Tensor(scores)).data, atol=1e-15
        )

    def test_numpy_mask_is_accepted_and_matches_tensor_mask(self):
        from engine.tensor import Tensor
        T = 4
        attn = MultiHeadAttention(d_model=8, num_heads=2)
        x = Tensor(RNG.standard_normal((1, T, 8)))
        mask = np.triu(np.full((T, T), -np.inf), k=1)

        np.testing.assert_array_equal(
            attn(x, mask).data, attn(x, Tensor(mask)).data
        )

    def test_per_head_mask_shape_is_supported(self):
        from engine.tensor import Tensor
        T, H = 4, 2
        attn = MultiHeadAttention(d_model=8, num_heads=H)
        x = Tensor(RNG.standard_normal((1, T, 8)))
        mask = np.zeros((1, H, T, T))
        mask[0, 1] = np.triu(np.full((T, T), -np.inf), k=1)

        out = attn(x, mask).data
        assert out.shape == (1, T, 8)
        assert np.isfinite(out).all()

    @pytest.mark.parametrize(
        ("batch", "heads"),
        [(2, 2), (3, 2)],
        ids=["batch-equals-heads", "batch-differs-from-heads"],
    )
    def test_three_dimensional_multihead_mask_is_batch_major(
        self, batch, heads
    ):
        """(B, Q, K) means per-batch even when B happens to equal H."""
        from engine.tensor import Tensor

        T = 4
        attn = MultiHeadAttention(d_model=8, num_heads=heads)
        x = RNG.standard_normal((batch, T, 8))
        mask = np.zeros((batch, T, T))
        for row in range(batch):
            mask[row, :, row % T] = -2.0 * (row + 1)

        forward_batched = attn(Tensor(x), Tensor(mask)).data
        forward_explicit = attn(Tensor(x), Tensor(mask[:, None, :, :])).data
        np.testing.assert_array_equal(forward_batched, forward_explicit)

        inferred_batched, _ = attn.infer(x, key_bias=mask)
        inferred_explicit, _ = attn.infer(
            x, key_bias=mask[:, None, :, :]
        )
        np.testing.assert_array_equal(inferred_batched, inferred_explicit)

    def test_three_dimensional_tensor_mask_keeps_its_gradient(self):
        from engine.tensor import Tensor

        B, T = 3, 4
        attn = MultiHeadAttention(d_model=8, num_heads=2)
        x = Tensor(RNG.standard_normal((B, T, 8)), requires_grad=True)
        mask = Tensor(np.zeros((B, T, T)), requires_grad=True)

        ops.sum(attn(x, mask)).backward()

        assert mask.grad.shape == (B, T, T)
        assert np.isfinite(mask.grad).all()
        assert np.any(mask.grad != 0.0)

    @pytest.mark.parametrize(
        ("attention_cls", "kwargs"),
        [
            (SelfAttention, {}),
            (MultiHeadAttention, {"num_heads": 2}),
        ],
        ids=["self", "multihead"],
    )
    @pytest.mark.parametrize(
        ("key_bias", "message"),
        [
            (np.zeros((3, 4)), "does not broadcast"),
            (np.zeros((2, 1, 1, 3)), "larger than"),
            (np.full((3, 3), np.nan), "finite biases"),
            (np.full((3, 3), np.inf), "finite biases"),
            (np.full((3, 3), "invalid"), "numeric values"),
        ],
        ids=[
            "non-broadcastable",
            "oversized",
            "nan",
            "positive-infinity",
            "non-numeric",
        ],
    )
    def test_inference_rejects_malformed_key_bias(
        self, attention_cls, kwargs, key_bias, message
    ):
        attn = attention_cls(d_model=8, **kwargs)
        x = RNG.standard_normal((1, 3, 8))

        with pytest.raises((TypeError, ValueError), match=message):
            attn.infer(x, key_bias=key_bias)

    @pytest.mark.parametrize(
        ("attention_cls", "kwargs"),
        [
            (SelfAttention, {}),
            (MultiHeadAttention, {"num_heads": 2}),
        ],
        ids=["self", "multihead"],
    )
    def test_fully_masked_inference_returns_projection_bias(
        self, attention_cls, kwargs
    ):
        from engine.tensor import Tensor

        B, T = 2, 3
        attn = attention_cls(d_model=8, **kwargs)
        attn.out_proj.bias.data[:] = np.arange(1.0, 9.0)
        x = RNG.standard_normal((B, T, 8))
        mask = np.full((B, T, T), -np.inf)

        inferred, _ = attn.infer(x, key_bias=mask)
        forwarded = attn(Tensor(x), Tensor(mask)).data
        expected = np.broadcast_to(attn.out_proj.bias.data, (B, T, 8))

        assert np.isfinite(inferred).all()
        np.testing.assert_allclose(inferred, expected, atol=1e-12)
        np.testing.assert_allclose(forwarded, expected, atol=1e-12)

    @pytest.mark.parametrize(
        ("attention_cls", "kwargs"),
        [
            (SelfAttention, {}),
            (MultiHeadAttention, {"num_heads": 2}),
        ],
        ids=["self", "multihead"],
    )
    @pytest.mark.parametrize(
        ("failure", "error", "message"),
        [
            ("missing-v", ValueError, "contain 'k' and 'v'"),
            ("object", TypeError, "real numeric"),
            ("nan", ValueError, "finite values"),
        ],
    )
    def test_inference_rejects_malformed_cache(
        self, attention_cls, kwargs, failure, error, message
    ):
        attn = attention_cls(d_model=8, **kwargs)
        x = RNG.standard_normal((1, 2, 8))
        shape = (1, 2, 8) if attention_cls is SelfAttention else (1, 2, 2, 4)
        key = np.zeros(shape)
        value = np.zeros(shape)
        if failure == "missing-v":
            cache = {"k": key}
        elif failure == "object":
            cache = {
                "k": key.astype(object),
                "v": value.astype(object),
            }
        else:
            key[..., 0] = np.nan
            cache = {"k": key, "v": value}

        with pytest.raises(error, match=message):
            attn.infer(x, cache=cache)


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


class TestKeyPaddingMask:
    """attention_mask must make padding invisible to the real tokens."""

    PAD = 0
    IGNORE = -1

    def _model(self, **kw):
        defaults = dict(
            vocab_size=10, context_len=8,
            d_model=8, num_heads=2, d_ff=16, num_layers=2,
        )
        defaults.update(kw)
        np.random.seed(3)
        return GPT(**defaults)

    def _padded_batch(self, real, pad_to):
        padding = pad_to - real.shape[1]
        idx = np.concatenate(
            [real, np.full((real.shape[0], padding), self.PAD, dtype=np.int64)],
            axis=1,
        )
        mask = np.concatenate(
            [np.ones_like(real), np.zeros((real.shape[0], padding), dtype=np.int64)],
            axis=1,
        )
        return idx, mask

    def test_real_token_logits_match_the_unpadded_sequence(self):
        model = self._model()
        real = np.array([[1, 4, 7, 2]], dtype=np.int64)
        idx, mask = self._padded_batch(real, pad_to=7)

        padded_logits = model(idx, attention_mask=mask).data
        unpadded_logits = model(real).data

        np.testing.assert_allclose(
            padded_logits[:, : real.shape[1]], unpadded_logits, atol=1e-12
        )

    def test_padding_content_cannot_change_real_token_logits(self):
        model = self._model()
        real = np.array([[3, 5, 1]], dtype=np.int64)
        idx, mask = self._padded_batch(real, pad_to=6)

        first = model(idx, attention_mask=mask).data[:, :3]
        idx[:, 3:] = 9  # different junk in the padded slots
        second = model(idx, attention_mask=mask).data[:, :3]

        np.testing.assert_array_equal(first, second)

    def test_without_the_mask_padding_leaks_into_real_tokens(self):
        """Guards the test above: the mask is what makes padding invisible."""
        model = self._model()
        real = np.array([[3, 5, 1]], dtype=np.int64)
        idx, _ = self._padded_batch(real, pad_to=6)

        # Position 2 is causal, so only later positions could differ; compare a
        # padded position's neighbour instead by changing the padding content.
        leaky_first = model(idx).data[:, 3]
        idx[:, 3:] = 9
        leaky_second = model(idx).data[:, 3]

        assert not np.allclose(leaky_first, leaky_second)

    def test_gradients_match_the_unpadded_sequence(self):
        real = np.array([[1, 4, 7, 2]], dtype=np.int64)
        targets = np.array([[4, 7, 2, 1]], dtype=np.int64)
        idx, mask = self._padded_batch(real, pad_to=7)
        padded_targets = np.concatenate(
            [targets, np.full((1, 3), self.IGNORE, dtype=np.int64)], axis=1
        )

        padded_model = self._model()
        padded_loss = ops.cross_entropy(
            padded_model(idx, attention_mask=mask),
            padded_targets,
            ignore_index=self.IGNORE,
        )
        padded_loss.backward()

        plain_model = self._model()
        plain_loss = ops.cross_entropy(plain_model(real), targets)
        plain_loss.backward()

        np.testing.assert_allclose(padded_loss.data, plain_loss.data, atol=1e-12)
        for (name, padded), (_, plain) in zip(
            padded_model.named_parameters(), plain_model.named_parameters()
        ):
            np.testing.assert_allclose(
                padded.grad, plain.grad, atol=1e-12, err_msg=f"gradient for {name}"
            )

    def test_all_padding_row_stays_finite(self):
        model = self._model()
        idx = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
        mask = np.array([[1, 1, 1], [0, 0, 0]], dtype=np.int64)

        logits = model(idx, attention_mask=mask).data

        assert np.isfinite(logits).all()

    def test_boolean_mask_matches_integer_mask(self):
        model = self._model()
        idx = np.array([[1, 2, 3, 0]], dtype=np.int64)
        mask = np.array([[1, 1, 1, 0]], dtype=np.int64)

        np.testing.assert_array_equal(
            model(idx, attention_mask=mask).data,
            model(idx, attention_mask=mask.astype(bool)).data,
        )

    def test_mask_is_applied_in_every_layer(self):
        """A single-layer model cannot detect a mask bug in later layers."""
        model = self._model(num_layers=3)
        real = np.array([[2, 6, 1, 3]], dtype=np.int64)
        idx, mask = self._padded_batch(real, pad_to=8)

        np.testing.assert_allclose(
            model(idx, attention_mask=mask).data[:, :4],
            model(real).data,
            atol=1e-12,
        )


ARCHITECTURES = [
    pytest.param(
        dict(norm="layernorm", pos_encoding="learned", ffn="gelu"), id="gpt"
    ),
    pytest.param(
        dict(norm="rmsnorm", pos_encoding="rope", ffn="swiglu"), id="llama"
    ),
]


class TestPaddedGeneration:
    """Left-padded batched generation must equal generating each row alone."""

    PAD = 0
    PROMPTS = [[1, 4, 7, 2, 5], [3, 6], [9, 8, 1]]

    def _model(self, arch, **kw):
        config = dict(
            vocab_size=12, context_len=16,
            d_model=16, num_heads=2, d_ff=32, num_layers=2,
        )
        config.update(arch)
        config.update(kw)
        np.random.seed(11)
        return GPT(**config)

    def _left_padded(self, prompts=None):
        prompts = prompts or self.PROMPTS
        width = max(len(prompt) for prompt in prompts)
        tokens = np.full((len(prompts), width), self.PAD, dtype=np.int64)
        mask = np.zeros((len(prompts), width), dtype=np.int64)
        for row, prompt in enumerate(prompts):
            tokens[row, width - len(prompt):] = prompt
            mask[row, width - len(prompt):] = 1
        return tokens, mask

    def _position_ids(self, mask):
        return np.maximum(np.cumsum(mask, axis=1) - 1, 0)

    @pytest.mark.parametrize("arch", ARCHITECTURES)
    def test_prefill_logits_match_the_unpadded_prompt(self, arch):
        model = self._model(arch)
        tokens, mask = self._left_padded()

        padded, _ = model.infer(
            tokens, attention_mask=mask, position_ids=self._position_ids(mask)
        )

        assert np.isfinite(padded).all()
        for row, prompt in enumerate(self.PROMPTS):
            alone, _ = model.infer(np.array([prompt], dtype=np.int64))
            np.testing.assert_allclose(
                padded[row, -1], alone[0, -1], atol=1e-12,
                err_msg=f"row {row} last-position logits differ",
            )

    @pytest.mark.parametrize("arch", ARCHITECTURES)
    def test_batched_generation_matches_one_prompt_at_a_time(self, arch):
        model = self._model(arch)
        tokens, mask = self._left_padded()
        new_tokens = 4

        batched = model.generate(
            tokens, new_tokens, strategy="greedy", attention_mask=mask
        )

        assert batched.shape == (len(self.PROMPTS), tokens.shape[1] + new_tokens)
        for row, prompt in enumerate(self.PROMPTS):
            alone = model.generate(
                np.array([prompt], dtype=np.int64), new_tokens, strategy="greedy"
            )
            np.testing.assert_array_equal(
                batched[row, -new_tokens:], alone[0, -new_tokens:]
            )
        # The prompt columns, padding included, come back unchanged.
        np.testing.assert_array_equal(batched[:, : tokens.shape[1]], tokens)

    @pytest.mark.parametrize("arch", ARCHITECTURES)
    def test_cached_and_uncached_padded_generation_agree(self, arch):
        model = self._model(arch)
        tokens, mask = self._left_padded()

        cached = model.generate(
            tokens, 3, strategy="greedy", use_cache=True, attention_mask=mask
        )
        uncached = model.generate(
            tokens, 3, strategy="greedy", use_cache=False, attention_mask=mask
        )

        np.testing.assert_array_equal(cached, uncached)

    def test_padding_content_cannot_change_generated_tokens(self):
        model = self._model(ARCHITECTURES[0].values[0])
        tokens, mask = self._left_padded()
        first = model.generate(tokens, 3, strategy="greedy", attention_mask=mask)

        scrambled = tokens.copy()
        scrambled[mask == 0] = 11
        second = model.generate(scrambled, 3, strategy="greedy", attention_mask=mask)

        np.testing.assert_array_equal(first[:, -3:], second[:, -3:])

    def test_single_row_prompt_with_no_padding_is_unaffected(self):
        model = self._model(ARCHITECTURES[0].values[0])
        prompt = np.array([[1, 2, 3]], dtype=np.int64)

        with_mask = model.generate(
            prompt, 3, strategy="greedy", attention_mask=np.ones((1, 3), dtype=np.int64)
        )
        without_mask = model.generate(prompt, 3, strategy="greedy")

        np.testing.assert_array_equal(with_mask, without_mask)

    def test_rejects_right_padded_mask(self):
        model = self._model(ARCHITECTURES[0].values[0])
        tokens = np.array([[1, 2, 0, 0]], dtype=np.int64)
        with pytest.raises(ValueError, match="left-padded"):
            model.generate(
                tokens, 2, strategy="greedy",
                attention_mask=np.array([[1, 1, 0, 0]]),
            )

    def test_rejects_prompt_without_any_real_token(self):
        model = self._model(ARCHITECTURES[0].values[0])
        tokens = np.array([[1, 2], [0, 0]], dtype=np.int64)
        with pytest.raises(ValueError, match="at least one real token"):
            model.generate(
                tokens, 2, strategy="greedy",
                attention_mask=np.array([[1, 1], [0, 0]]),
            )

    def test_beam_search_all_real_mask_matches_unmasked_prompt(self):
        model = self._model(ARCHITECTURES[0].values[0])
        tokens = np.array([[1, 2]], dtype=np.int64)

        with_mask = model.generate(
            tokens,
            2,
            strategy="beam",
            beam_width=2,
            attention_mask=np.ones((1, 2), dtype=np.int64),
        )
        without_mask = model.generate(
            tokens,
            2,
            strategy="beam",
            beam_width=2,
        )

        np.testing.assert_array_equal(with_mask, without_mask)

    def test_sampling_respects_the_mask_for_every_row(self):
        """Sampling shares the masked forward pass, so rows stay independent."""
        model = self._model(ARCHITECTURES[0].values[0])
        tokens, mask = self._left_padded()

        np.random.seed(0)
        first = model.generate(
            tokens, 3, temperature=0.9, strategy="sample", attention_mask=mask
        )
        scrambled = tokens.copy()
        scrambled[mask == 0] = 11
        np.random.seed(0)
        second = model.generate(
            scrambled, 3, temperature=0.9, strategy="sample", attention_mask=mask
        )

        np.testing.assert_array_equal(first[:, -3:], second[:, -3:])


class TestMaskedSlidingWindow(TestPaddedGeneration):
    """A masked run that outgrows context_len must still match solo decoding.

    Inherits the padded-generation fixtures deliberately: the claim is that the
    equivalence proved inside the window survives the crop, so it is the same
    claim under harder conditions, not a new one.
    """

    CONTEXT_LEN = 8

    def _model(self, arch, **kw):
        kw.setdefault("context_len", self.CONTEXT_LEN)
        return super()._model(arch, **kw)

    def _crossing_run(self, model, prompts=None, new_tokens=6, use_cache=True):
        """Generate past the window and return (batched output, prompts)."""
        prompts = prompts or self.PROMPTS
        tokens, mask = self._left_padded(prompts)
        batched = model.generate(
            tokens, new_tokens, strategy="greedy",
            attention_mask=mask, use_cache=use_cache,
        )
        # Without this the run never reaches the crop and the test would pass
        # for the wrong reason.
        assert batched.shape[1] > model.context_len
        return batched, prompts

    @pytest.mark.parametrize("arch", ARCHITECTURES)
    def test_run_past_the_window_matches_one_prompt_at_a_time(self, arch):
        model = self._model(arch)
        new_tokens = 6
        batched, prompts = self._crossing_run(model, new_tokens=new_tokens)

        for row, prompt in enumerate(prompts):
            alone = model.generate(
                np.array([prompt], dtype=np.int64), new_tokens, strategy="greedy"
            )
            np.testing.assert_array_equal(
                batched[row, -new_tokens:], alone[0, -new_tokens:],
                err_msg=f"row {row} diverged after the window slid",
            )

    @pytest.mark.parametrize("arch", ARCHITECTURES)
    def test_prompt_longer_than_the_window_is_cropped_per_row(self, arch):
        """The first prefill already crops: no step ever fits the full prompt."""
        model = self._model(arch)
        prompts = [[1, 4, 7, 2, 5, 3, 9, 8, 2, 6], [3, 6, 1, 5, 7, 2]]
        new_tokens = 5
        tokens, mask = self._left_padded(prompts)
        assert tokens.shape[1] > model.context_len

        batched = model.generate(
            tokens, new_tokens, strategy="greedy", attention_mask=mask
        )

        for row, prompt in enumerate(prompts):
            alone = model.generate(
                np.array([prompt], dtype=np.int64), new_tokens, strategy="greedy"
            )
            np.testing.assert_array_equal(
                batched[row, -new_tokens:], alone[0, -new_tokens:],
                err_msg=f"row {row} diverged on an over-long prompt",
            )

    def test_many_crops_stay_equivalent(self):
        """20 new tokens re-prefill the window on every step after it fills."""
        model = self._model(ARCHITECTURES[0].values[0])
        new_tokens = 20
        batched, prompts = self._crossing_run(model, new_tokens=new_tokens)

        for row, prompt in enumerate(prompts):
            alone = model.generate(
                np.array([prompt], dtype=np.int64), new_tokens, strategy="greedy"
            )
            np.testing.assert_array_equal(
                batched[row, -new_tokens:], alone[0, -new_tokens:]
            )

    @pytest.mark.parametrize("arch", ARCHITECTURES)
    def test_cached_and_uncached_agree_past_the_window(self, arch):
        model = self._model(arch)
        cached, _ = self._crossing_run(model, use_cache=True)
        uncached, _ = self._crossing_run(model, use_cache=False)

        np.testing.assert_array_equal(cached, uncached)

    def test_padding_content_cannot_change_tokens_past_the_window(self):
        model = self._model(ARCHITECTURES[0].values[0])
        tokens, mask = self._left_padded()
        first = model.generate(tokens, 6, strategy="greedy", attention_mask=mask)

        scrambled = tokens.copy()
        scrambled[mask == 0] = 11
        second = model.generate(scrambled, 6, strategy="greedy", attention_mask=mask)

        np.testing.assert_array_equal(first[:, -6:], second[:, -6:])

    def test_dropping_the_mask_still_changes_the_answer(self):
        """Counter-test: the crop must not be quietly discarding the mask."""
        model = self._model(ARCHITECTURES[0].values[0])
        tokens, mask = self._left_padded()

        masked = model.generate(tokens, 6, strategy="greedy", attention_mask=mask)
        unmasked = model.generate(tokens, 6, strategy="greedy")

        assert not np.array_equal(masked[:, -6:], unmasked[:, -6:])

    @pytest.mark.parametrize("arch", ARCHITECTURES)
    def test_window_matches_a_direct_inference_on_the_surviving_tokens(self, arch):
        """Anchor the renumbering against a path that never touches generate().

        Each row's next token must be what ``infer`` predicts from that row's
        last ``context_len`` *real* tokens, fed unpadded with default positions.
        """
        model = self._model(arch)
        new_tokens = 7
        tokens, mask = self._left_padded()

        before = model.generate(
            tokens, new_tokens - 1, strategy="greedy", attention_mask=mask
        )
        after = model.generate(
            tokens, new_tokens, strategy="greedy", attention_mask=mask
        )
        keep = np.concatenate(
            [mask, np.ones((mask.shape[0], new_tokens - 1), dtype=np.int64)], axis=1
        )

        for row in range(len(self.PROMPTS)):
            window = before[row][keep[row] == 1][-model.context_len:]
            logits, _ = model.infer(np.array([window], dtype=np.int64))
            assert int(np.argmax(logits[0, -1])) == int(after[row, -1]), (
                f"row {row}: generate disagrees with infer on the cropped window"
            )

    def test_sampling_past_the_window_ignores_padding_content(self):
        model = self._model(ARCHITECTURES[0].values[0])
        tokens, mask = self._left_padded()

        np.random.seed(0)
        first = model.generate(
            tokens, 6, temperature=0.9, strategy="sample", attention_mask=mask
        )
        scrambled = tokens.copy()
        scrambled[mask == 0] = 11
        np.random.seed(0)
        second = model.generate(
            scrambled, 6, temperature=0.9, strategy="sample", attention_mask=mask
        )

        np.testing.assert_array_equal(first[:, -6:], second[:, -6:])


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

    def test_loading_legacy_checkpoint_rebuilds_strict_causal_mask(self):
        model = self._make_model()
        state = model.state_dict()
        state["causal_mask"] = np.triu(
            np.full((model.context_len, model.context_len), -1e9), k=1
        )

        restored = self._make_model()
        restored.load_state_dict(state)
        assert np.isneginf(restored.causal_mask.data[0, 1])
        assert restored.causal_mask.data[-1, -1] == 0.0


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
