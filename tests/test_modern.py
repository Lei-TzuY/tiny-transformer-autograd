"""
test_modern.py — Tests for the Llama-style architecture options and the
training additions built on top of them:

  - silu op (numerical gradient + stability)
  - RMSNorm (correctness, gradients, infer parity)
  - RoPE (rotation invariants, relative-position property, cache parity)
  - SwiGLU (gradient flow, infer parity)
  - GPT(arch=llama) end-to-end (forward/infer parity, generation, checkpoint)
  - AdamW decoupled weight decay
  - gradient accumulation equivalence

Run:
    pytest tests/test_modern.py -v
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.tensor import Tensor
import engine.ops as ops
from engine.optim import Adam, AdamW
from engine.checkpoint import read_checkpoint, restore_checkpoint, save_checkpoint
from nn.layers import RMSNorm, Linear
from nn.attention import RotaryEmbedding, _rotate_half_np
from nn.transformer import GPT, SwiGLU

LLAMA = {"norm": "rmsnorm", "pos_encoding": "rope", "ffn": "swiglu"}


def make(shape, seed=0):
    rng = np.random.default_rng(seed)
    return Tensor(rng.standard_normal(shape) * 0.5, requires_grad=True)


def numeric_grad(fn, x: Tensor, eps=1e-5):
    """Central finite differences of sum(fn(x)) w.r.t. x."""
    grad = np.zeros_like(x.data)
    it = np.nditer(x.data, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = float(x.data[idx])
        x.data[idx] = orig + eps
        fp = float(fn(x).data.sum())
        x.data[idx] = orig - eps
        fm = float(fn(x).data.sum())
        x.data[idx] = orig
        grad[idx] = (fp - fm) / (2 * eps)
        it.iternext()
    return grad


def check_grad(fn, x: Tensor, tol=1e-5):
    x.grad = np.zeros_like(x.data)
    fn(x).backward()
    analytical = x.grad.copy()
    numerical = numeric_grad(fn, x)
    err = float(np.max(np.abs(analytical - numerical)))
    assert err < tol, f"gradient mismatch: {err:.2e}"


def tiny_gpt(seed=0, **overrides):
    np.random.seed(seed)
    config = {
        "vocab_size": 31,
        "context_len": 16,
        "d_model": 16,
        "num_heads": 2,
        "d_ff": 32,
        "num_layers": 2,
        **overrides,
    }
    return GPT(**config)


# ---------------------------------------------------------------------------
# silu
# ---------------------------------------------------------------------------
class TestSiLU:
    def test_matches_definition(self):
        x = make((4, 5), 1)
        expected = x.data / (1.0 + np.exp(-x.data))
        np.testing.assert_allclose(ops.silu(x).data, expected, atol=1e-12)

    def test_gradient(self):
        check_grad(ops.silu, make((3, 4), 2))

    def test_extreme_values_are_stable(self):
        x = Tensor(np.array([-1e4, -50.0, 0.0, 50.0, 1e4]), requires_grad=True)
        with np.errstate(over="raise"):
            out = ops.silu(x)
            out.backward(np.ones_like(x.data))
        np.testing.assert_allclose(out.data[[0, 1]], [0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(out.data[[3, 4]], [50.0, 1e4], atol=1e-8)
        assert np.all(np.isfinite(x.grad))


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------
class TestRMSNorm:
    def test_output_has_unit_rms(self):
        norm = RMSNorm(8)
        x = make((4, 6, 8), 3)
        out = norm(x)
        rms = np.sqrt((out.data ** 2).mean(axis=-1))
        np.testing.assert_allclose(rms, 1.0, atol=1e-3)

    def test_no_mean_centering(self):
        """Unlike LayerNorm, a constant offset changes the output."""
        norm = RMSNorm(8)
        x = make((2, 8), 4)
        shifted = Tensor(x.data + 5.0)
        assert not np.allclose(norm(x).data, norm(shifted).data)

    def test_gradient_wrt_input(self):
        norm = RMSNorm(6)
        check_grad(lambda t: norm(t), make((3, 6), 5))

    def test_gradient_wrt_gamma(self):
        norm = RMSNorm(6)
        x = make((3, 6), 6)
        check_grad(lambda g: x * ((ops.mean(x ** 2, axis=-1, keepdims=True)
                                   + norm.eps) ** -0.5) * g, norm.gamma)

    def test_infer_matches_forward(self):
        norm = RMSNorm(8)
        norm.gamma.data[:] = np.linspace(0.5, 1.5, 8)
        x = make((2, 4, 8), 7)
        np.testing.assert_allclose(norm.infer(x.data), norm(x).data, atol=1e-12)

    def test_single_parameter(self):
        assert [name for name, _ in RMSNorm(4).named_parameters()] == ["gamma"]


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------
class TestRoPE:
    def test_rejects_odd_dim(self):
        with pytest.raises(ValueError):
            RotaryEmbedding(5, 16)

    def test_rotation_preserves_norm(self):
        rope = RotaryEmbedding(8, 32)
        x = np.random.default_rng(0).standard_normal((2, 3, 10, 8))
        rotated = rope.rotate_np(x)
        np.testing.assert_allclose(
            np.linalg.norm(rotated, axis=-1), np.linalg.norm(x, axis=-1), atol=1e-10
        )

    def test_position_zero_is_identity(self):
        rope = RotaryEmbedding(8, 32)
        x = np.random.default_rng(1).standard_normal((1, 1, 1, 8))
        np.testing.assert_allclose(rope.rotate_np(x), x, atol=1e-12)

    def test_scores_depend_only_on_relative_offset(self):
        """q·k after rotation at (m, n) must equal the same at (m+s, n+s)."""
        rope = RotaryEmbedding(16, 64)
        rng = np.random.default_rng(2)
        q = rng.standard_normal(16)
        k = rng.standard_normal(16)

        def score(pos_q, pos_k):
            rq = rope.rotate_np(q[None, :], offset=pos_q)[0]
            rk = rope.rotate_np(k[None, :], offset=pos_k)[0]
            return float(rq @ rk)

        np.testing.assert_allclose(score(5, 2), score(25, 22), atol=1e-10)
        np.testing.assert_allclose(score(0, 3), score(40, 43), atol=1e-10)

    def test_tensor_path_matches_numpy_path(self):
        rope = RotaryEmbedding(8, 32)
        x = make((2, 2, 6, 8), 8)
        np.testing.assert_allclose(
            rope.rotate(x, offset=3).data, rope.rotate_np(x.data, offset=3), atol=1e-12
        )

    def test_gradient_through_rotation(self):
        rope = RotaryEmbedding(6, 16)
        check_grad(lambda t: rope.rotate(t, offset=2), make((2, 4, 6), 9))

    def test_rotate_half_layout(self):
        x = np.arange(6.0)[None, :]
        np.testing.assert_allclose(
            _rotate_half_np(x)[0], [-3.0, -4.0, -5.0, 0.0, 1.0, 2.0]
        )


# ---------------------------------------------------------------------------
# SwiGLU
# ---------------------------------------------------------------------------
class TestSwiGLU:
    def test_shapes_and_no_biases(self):
        ff = SwiGLU(8, 16)
        out = ff(make((2, 4, 8), 10))
        assert out.shape == (2, 4, 8)
        assert ff.fc_gate.bias is None and ff.fc_up.bias is None and ff.fc_down.bias is None

    def test_all_weights_receive_gradients(self):
        ff = SwiGLU(6, 12)
        ops.mean(ff(make((2, 3, 6), 11))).backward()
        for name, p in ff.named_parameters():
            assert p.grad is not None and np.any(p.grad != 0), f"{name} has no gradient"

    def test_infer_matches_forward(self):
        ff = SwiGLU(8, 16)
        x = make((2, 4, 8), 12)
        np.testing.assert_allclose(ff.infer(x.data), ff(x).data, atol=1e-12)

    def test_gradient_wrt_input(self):
        ff = SwiGLU(4, 8)
        check_grad(lambda t: ff(t), make((2, 4), 13), tol=1e-4)


# ---------------------------------------------------------------------------
# GPT with Llama-style architecture
# ---------------------------------------------------------------------------
class TestLlamaGPT:
    def test_forward_shape_and_no_pos_emb(self):
        model = tiny_gpt(**LLAMA)
        assert model.pos_emb is None
        assert not any("pos_emb" in name for name in model.state_dict())
        idx = np.random.randint(0, 31, size=(2, 12))
        assert model(idx).shape == (2, 12, 31)

    def test_rejects_odd_head_dim_with_rope(self):
        with pytest.raises(ValueError):
            tiny_gpt(d_model=6, num_heads=2, **LLAMA)  # d_k = 3

    def test_rejects_unknown_arch_options(self):
        with pytest.raises((ValueError, KeyError)):
            tiny_gpt(norm="batchnorm")
        with pytest.raises(ValueError):
            tiny_gpt(pos_encoding="alibi")
        with pytest.raises((ValueError, KeyError)):
            tiny_gpt(ffn="relu2")

    def test_all_parameters_receive_gradients(self):
        model = tiny_gpt(**LLAMA)
        idx = np.random.randint(0, 31, size=(2, 8))
        targets = np.random.randint(0, 31, size=(2 * 8,))
        logits = model(idx)
        loss = ops.cross_entropy(ops.reshape(logits, (16, 31)), targets)
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None and np.any(p.grad != 0), f"{name} has no gradient"

    def test_infer_matches_forward(self):
        model = tiny_gpt(**LLAMA).eval()
        idx = np.random.randint(0, 31, size=(2, 10))
        logits, _ = model.infer(idx)
        np.testing.assert_allclose(logits, model(idx).data, atol=1e-10)

    def test_cached_inference_matches_full_inference(self):
        model = tiny_gpt(**LLAMA).eval()
        idx = np.random.randint(0, 31, size=(1, 10))
        full_logits, _ = model.infer(idx)

        logits, cache = model.infer(idx[:, :4])
        for t in range(4, 10):
            logits, cache = model.infer(idx[:, t : t + 1], cache)
        np.testing.assert_allclose(logits[0, -1], full_logits[0, -1], atol=1e-10)

    def test_greedy_generation_cache_parity(self):
        model = tiny_gpt(**LLAMA).eval()
        prompt = np.random.randint(0, 31, size=(1, 4))
        cached = model.generate(prompt, 20, strategy="greedy", use_cache=True)
        uncached = model.generate(prompt, 20, strategy="greedy", use_cache=False)
        np.testing.assert_array_equal(cached, uncached)

    def test_checkpoint_roundtrip(self, tmp_path):
        model = tiny_gpt(**LLAMA)
        path = str(tmp_path / "llama.pkl")
        save_checkpoint(path, model, step=7, metadata={"model_config": model.config()})

        state = read_checkpoint(path)
        config = state["metadata"]["model_config"]
        assert (config["norm"], config["pos_encoding"], config["ffn"]) == (
            "rmsnorm", "rope", "swiglu",
        )
        restored = GPT(**config)
        assert restore_checkpoint(state, restored) == 7

        idx = np.random.randint(0, 31, size=(1, 6))
        np.testing.assert_allclose(restored(idx).data, model(idx).data, atol=1e-12)

    def test_old_config_without_arch_keys_still_loads(self):
        """Checkpoints from before the arch options must construct a GPT model."""
        config = {
            "vocab_size": 31, "context_len": 16, "d_model": 16,
            "num_heads": 2, "d_ff": 32, "num_layers": 1,
            "dropout": 0.0, "lora_rank": 0, "lora_alpha": 1.0,
        }
        model = GPT(**config)
        assert (model.norm, model.pos_encoding, model.ffn) == (
            "layernorm", "learned", "gelu",
        )

    def test_training_reduces_loss(self):
        np.random.seed(0)
        model = tiny_gpt(**LLAMA)
        optimizer = AdamW(model.parameters(), lr=1e-2, weight_decay=1e-3)
        data = np.tile(np.arange(31), 8)

        def step_loss():
            offsets = np.random.randint(0, len(data) - 9, size=(8,))
            x = np.stack([data[i : i + 8] for i in offsets])
            y = np.stack([data[i + 1 : i + 9] for i in offsets])
            logits = model(x)
            loss = ops.cross_entropy(ops.reshape(logits, (64, 31)), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return float(loss.data)

        first = np.mean([step_loss() for _ in range(5)])
        for _ in range(60):
            step_loss()
        last = np.mean([step_loss() for _ in range(5)])
        assert last < first * 0.7, f"loss did not improve: {first:.3f} → {last:.3f}"

    def test_lora_works_with_llama_arch(self):
        model = tiny_gpt(**LLAMA)
        total = model.param_count()
        model.enable_lora(2)
        assert model.param_count() < total
        idx = np.random.randint(0, 31, size=(1, 6))
        loss = ops.mean(model(idx))
        loss.backward()
        lora_names = [n for n, _ in model.named_parameters()]
        assert lora_names and all("lora" in n for n in lora_names)


# ---------------------------------------------------------------------------
# AdamW
# ---------------------------------------------------------------------------
class TestAdamW:
    def test_pure_decay_with_zero_gradient(self):
        """With g=0 the update must be exactly p ← p·(1 − lr·λ)."""
        p = Tensor(np.full(4, 2.0), requires_grad=True)
        p.grad = np.zeros(4)
        opt = AdamW([p], lr=0.1, weight_decay=0.5)
        opt.step()
        np.testing.assert_allclose(p.data, 2.0 * (1 - 0.1 * 0.5), atol=1e-12)

    def test_adam_couples_decay_through_moments(self):
        """
        Under Adam, L2 decay flows through m/v where bias-corrected
        normalisation makes the first-step update ≈ lr·sign(g) regardless of
        weight magnitude.  AdamW's decay stays proportional to the weight.
        """
        pa = Tensor(np.full(4, 3.0), requires_grad=True); pa.grad = np.zeros(4)
        pw = Tensor(np.full(4, 3.0), requires_grad=True); pw.grad = np.zeros(4)
        Adam([pa], lr=0.1, weight_decay=0.5).step()
        AdamW([pw], lr=0.1, weight_decay=0.5).step()
        np.testing.assert_allclose(pa.data, 3.0 - 0.1, atol=1e-6)   # ≈ lr
        np.testing.assert_allclose(pw.data, 3.0 * (1 - 0.05), atol=1e-12)
        assert not np.allclose(pa.data, pw.data)

    def test_no_decay_matches_adam(self):
        rng = np.random.default_rng(14)
        data = rng.standard_normal(6)
        grad = rng.standard_normal(6)
        p1 = Tensor(data.copy(), requires_grad=True); p1.grad = grad.copy()
        p2 = Tensor(data.copy(), requires_grad=True); p2.grad = grad.copy()
        Adam([p1], lr=1e-2).step()
        AdamW([p2], lr=1e-2).step()
        np.testing.assert_allclose(p1.data, p2.data, atol=1e-12)

    def test_state_dict_roundtrip(self):
        p = Tensor(np.ones(3), requires_grad=True)
        p.grad = np.full(3, 0.5)
        opt = AdamW([p], lr=1e-3, weight_decay=1e-2)
        opt.step()
        state = opt.state_dict()

        fresh = AdamW([p], lr=1e-3, weight_decay=1e-2)
        fresh.load_state_dict(state)
        assert fresh.t == 1
        np.testing.assert_allclose(fresh._m[0], opt._m[0])
        np.testing.assert_allclose(fresh._v[0], opt._v[0])


# ---------------------------------------------------------------------------
# Gradient accumulation
# ---------------------------------------------------------------------------
class TestGradAccumulation:
    def test_accumulated_grads_match_large_batch(self):
        """Two half-batches (grads averaged) ≡ one combined batch."""
        rng = np.random.default_rng(15)
        layer = Linear(4, 3)
        xa, xb = rng.standard_normal((8, 4)), rng.standard_normal((8, 4))
        ya = rng.integers(0, 3, size=8)
        yb = rng.integers(0, 3, size=8)

        # Accumulated: backward twice, then average.
        layer.zero_grad()
        ops.cross_entropy(layer(Tensor(xa)), ya).backward()
        ops.cross_entropy(layer(Tensor(xb)), yb).backward()
        accumulated = {id(p): p.grad / 2 for p in layer.parameters()}

        # Single big batch.
        layer.zero_grad()
        big_x = np.concatenate([xa, xb])
        big_y = np.concatenate([ya, yb])
        ops.cross_entropy(layer(Tensor(big_x)), big_y).backward()

        for p in layer.parameters():
            np.testing.assert_allclose(accumulated[id(p)], p.grad, atol=1e-12)

    def test_train_cli_validates_grad_accum(self):
        import train

        class Args:
            pass

        args = Args()
        for key, value in dict(
            iters=10, batch=2, ctx=8, d=16, heads=2, layers=1, arch="llama",
            eval_interval=5, eval_iters=1, warmup_iters=0, save_every=0,
            val_frac=0.1, lr=1e-3, min_lr=0.0, dropout=0.0, weight_decay=0.0,
            grad_clip=1.0, bpe_merges=10, lora_rank=0, lora_alpha=1.0,
            sample=0, temperature=1.0, beam_width=1, top_k=None, top_p=None,
            prompt=None, prompt_file=None, eval_only=False, generate_only=False,
            grad_accum=0, optimizer="adamw", data_format="text", jsonl_field="text",
        ).items():
            setattr(args, key, value)
        with pytest.raises(ValueError, match="grad-accum"):
            train._validate_args(args)
        args.grad_accum = 2
        train._validate_args(args)

        args.d, args.heads = 6, 2  # d_k = 3 is odd → rope must reject
        with pytest.raises(ValueError, match="llama"):
            train._validate_args(args)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
