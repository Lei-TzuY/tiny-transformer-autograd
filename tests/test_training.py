"""
test_training.py — Tests for scheduler, tokenizer, checkpoint, and LoRA.

Run:
    pytest tests/test_training.py -v
or:
    python tests/test_training.py
"""

import sys
import os
import tempfile
import traceback

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.optim import Adam
from engine.scheduler import WarmupCosineScheduler
from engine.checkpoint import read_checkpoint, restore_checkpoint, save_checkpoint
from nn.transformer import GPT
from tokenizer import build_tokenizer, tokenizer_from_state_dict, CharTokenizer, BPETokenizer
import engine.ops as ops


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class TestWarmupCosineScheduler:
    def _make(self, total=100, warmup=10, min_lr=0.0, base_lr=1e-3):
        from engine.tensor import Tensor
        dummy_params = [Tensor(np.zeros(1), requires_grad=True)]
        opt = Adam(dummy_params, lr=base_lr)
        return opt, WarmupCosineScheduler(opt, total, warmup, min_lr)

    def test_warmup_linear(self):
        opt, sched = self._make(total=100, warmup=10, base_lr=1.0)
        lr0 = sched.step(0)
        assert abs(lr0 - 0.1) < 1e-9, f"expected 0.1, got {lr0}"
        lr9 = sched.step(9)
        assert abs(lr9 - 1.0) < 1e-9, f"expected 1.0, got {lr9}"

    def test_cosine_decay_reaches_min(self):
        opt, sched = self._make(total=100, warmup=0, min_lr=0.01, base_lr=1.0)
        lr_last = sched.step(99)
        # At the very last step, should be close to min_lr
        assert abs(lr_last - 0.01) < 1e-6, f"expected ~0.01, got {lr_last}"

    def test_cosine_monotone(self):
        opt, sched = self._make(total=50, warmup=5, min_lr=0.0, base_lr=1.0)
        lrs = [sched.step(i) for i in range(5, 50)]
        for i in range(len(lrs) - 1):
            assert lrs[i] >= lrs[i + 1] - 1e-12, (
                f"LR increased at step {i + 5}: {lrs[i]:.6f} < {lrs[i+1]:.6f}"
            )

    def test_state_dict_round_trip(self):
        opt, sched = self._make(total=100, warmup=10)
        for i in range(30):
            sched.step(i)
        state = sched.state_dict()
        opt2, sched2 = self._make(total=100, warmup=10)
        sched2.load_state_dict(state)
        assert abs(sched.get_lr(40) - sched2.get_lr(40)) < 1e-12

    def test_no_warmup(self):
        opt, sched = self._make(total=10, warmup=0, base_lr=1.0, min_lr=0.0)
        lr0 = sched.step(0)
        # At step 0 with no warmup, should be base_lr (progress=0 → cosine=1)
        assert abs(lr0 - 1.0) < 1e-9


    def test_single_step_no_warmup_uses_base_lr(self):
        opt, sched = self._make(total=1, warmup=0, base_lr=1.0, min_lr=0.0)
        assert abs(sched.step(0) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_SAMPLE = "hello world foo bar baz hello world"


class TestCharTokenizer:
    def test_vocab_size(self):
        tok = CharTokenizer.train(_SAMPLE)
        expected = len(set(_SAMPLE))
        assert tok.vocab_size == expected

    def test_roundtrip(self):
        tok = CharTokenizer.train(_SAMPLE)
        assert tok.decode(tok.encode(_SAMPLE)) == _SAMPLE

    def test_state_dict_roundtrip(self):
        tok = CharTokenizer.train(_SAMPLE)
        tok2 = tokenizer_from_state_dict(tok.state_dict())
        assert tok2.decode(tok2.encode("hello")) == "hello"

    def test_build_tokenizer(self):
        tok = build_tokenizer("char", _SAMPLE)
        assert tok.kind == "char"
        assert tok.vocab_size == len(set(_SAMPLE))


class TestBPETokenizer:
    def test_encode_decode(self):
        tok = BPETokenizer.train(_SAMPLE, num_merges=5)
        decoded = tok.decode(tok.encode(_SAMPLE))
        assert decoded == _SAMPLE, f"roundtrip failed: {decoded!r}"

    def test_merges_reduce_token_count(self):
        tok0 = CharTokenizer.train(_SAMPLE)
        bpe = BPETokenizer.train(_SAMPLE, num_merges=10)
        n_char = len(tok0.encode(_SAMPLE))
        n_bpe = len(bpe.encode(_SAMPLE))
        assert n_bpe <= n_char, (
            f"BPE should have fewer tokens: char={n_char}, bpe={n_bpe}"
        )

    def test_state_dict_roundtrip(self):
        tok = BPETokenizer.train(_SAMPLE, num_merges=5)
        tok2 = tokenizer_from_state_dict(tok.state_dict())
        assert tok2.decode(tok2.encode("hello")) == "hello"

    def test_build_tokenizer(self):
        tok = build_tokenizer("bpe", _SAMPLE, bpe_merges=10)
        assert tok.kind == "bpe"
        assert tok.vocab_size >= len(set(_SAMPLE))   # vocab grows with merges

    def test_vocab_grows_with_merges(self):
        tok5 = BPETokenizer.train(_SAMPLE, num_merges=5)
        tok10 = BPETokenizer.train(_SAMPLE, num_merges=10)
        assert tok10.vocab_size >= tok5.vocab_size


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

class TestCheckpoint:
    def _small_model_and_opt(self):
        model = GPT(vocab_size=10, context_len=4, d_model=8,
                    num_heads=2, d_ff=16, num_layers=1)
        opt = Adam(model.parameters(), lr=1e-3)
        return model, opt

    def test_save_and_load(self):
        model, opt = self._small_model_and_opt()
        sched = WarmupCosineScheduler(opt, total_steps=100, warmup_steps=10)

        # Train a bit to accumulate optimizer state
        idx = np.array([[0, 1, 2, 3]])
        tgt = np.array([[1, 2, 3, 0]])
        for _ in range(3):
            logits = model(idx)
            B, T, V = logits.shape
            loss = ops.cross_entropy(ops.reshape(logits, (B * T, V)), tgt.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step(opt.t)

        original_params = {name: t.data.copy() for name, t in model.named_parameters()}
        original_t = opt.t

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.pkl")
            save_checkpoint(path, model, opt, sched, step=5, metadata={"test": True})
            assert os.path.exists(path)

            # Restore into a fresh model
            model2, opt2 = self._small_model_and_opt()
            sched2 = WarmupCosineScheduler(opt2, total_steps=100, warmup_steps=10)
            ckpt = read_checkpoint(path)
            step = restore_checkpoint(ckpt, model2, opt2, sched2)

            assert step == 5
            assert ckpt["metadata"]["test"] is True
            assert opt2.t == original_t
            for name, t in model2.named_parameters():
                np.testing.assert_array_equal(
                    t.data, original_params[name],
                    err_msg=f"Parameter {name} differs after restore"
                )

    def test_training_continues_after_resume(self):
        """Loss after resume should equal loss if we'd never stopped."""
        np.random.seed(7)
        model, opt = self._small_model_and_opt()
        sched = WarmupCosineScheduler(opt, total_steps=20, warmup_steps=2)

        idx = np.array([[0, 1, 2, 3]])
        tgt = np.array([[1, 2, 3, 0]])

        def one_step(m, o, s, step_num):
            logits = m(idx)
            B, T, V = logits.shape
            loss = ops.cross_entropy(ops.reshape(logits, (B * T, V)), tgt.reshape(-1))
            o.zero_grad()
            loss.backward()
            s.step(step_num)
            o.step()
            return float(loss.data)

        # 5 steps without stopping
        np.random.seed(7)
        model_ref, opt_ref = self._small_model_and_opt()
        sched_ref = WarmupCosineScheduler(opt_ref, total_steps=20, warmup_steps=2)
        for i in range(5):
            one_step(model_ref, opt_ref, sched_ref, i)
        loss_ref = one_step(model_ref, opt_ref, sched_ref, 5)

        # 3 steps, checkpoint, restore, then 2 more steps
        np.random.seed(7)
        model_a, opt_a = self._small_model_and_opt()
        sched_a = WarmupCosineScheduler(opt_a, total_steps=20, warmup_steps=2)
        for i in range(3):
            one_step(model_a, opt_a, sched_a, i)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.pkl")
            save_checkpoint(path, model_a, opt_a, sched_a, step=3)
            model_b, opt_b = self._small_model_and_opt()
            sched_b = WarmupCosineScheduler(opt_b, total_steps=20, warmup_steps=2)
            ckpt = read_checkpoint(path)
            restore_checkpoint(ckpt, model_b, opt_b, sched_b)
            for i in range(3, 5):
                one_step(model_b, opt_b, sched_b, i)
            loss_resumed = one_step(model_b, opt_b, sched_b, 5)

        assert abs(loss_ref - loss_resumed) < 1e-8, (
            f"loss diverged after resume: {loss_ref:.6f} vs {loss_resumed:.6f}"
        )


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

class TestLoRA:
    def _model(self, lora=0):
        return GPT(vocab_size=10, context_len=4, d_model=8,
                   num_heads=2, d_ff=16, num_layers=1, lora_rank=lora)

    def test_lora_freezes_backbone(self):
        model = self._model(lora=2)
        trainable = [name for name, p in model.named_parameters()]
        frozen = [name for name, t in model.named_tensors()
                  if not t.requires_grad]
        assert trainable, "no trainable parameters with LoRA"
        assert frozen, "no frozen parameters with LoRA"
        # Only lora_A and lora_B should be trainable
        for name in trainable:
            assert "lora" in name, f"non-LoRA param is trainable: {name}"

    def test_lora_output_shape(self):
        model = self._model(lora=2)
        idx = np.array([[0, 1, 2, 3]])
        logits = model(idx)
        assert logits.shape == (1, 4, 10)

    def test_lora_grads_only_on_adapters(self):
        model = self._model(lora=2)
        idx = np.array([[0, 1, 2, 3]])
        tgt = np.array([[1, 2, 3, 0]])
        logits = model(idx)
        B, T, V = logits.shape
        loss = ops.cross_entropy(ops.reshape(logits, (B * T, V)), tgt.reshape(-1))
        loss.backward()
        for name, t in model.named_tensors():
            if not t.requires_grad:
                assert t.grad is None or np.all(t.grad == 0), (
                    f"frozen param {name} received gradient"
                )

    def test_lora_fewer_params_than_full(self):
        full = self._model(lora=0)
        lora = self._model(lora=2)
        assert lora.param_count() < full.param_count()

    def test_no_lora_trains_all(self):
        model = self._model(lora=0)
        all_tensors = list(model.named_tensors())
        trainable = [n for n, t in all_tensors if t.requires_grad]
        # causal_mask should be frozen, everything else trainable
        assert len(trainable) > 0
        for name, t in model.named_tensors():
            if "causal_mask" not in name:
                assert t.requires_grad, f"{name} should be trainable"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    suites = [
        TestWarmupCosineScheduler,
        TestCharTokenizer,
        TestBPETokenizer,
        TestCheckpoint,
        TestLoRA,
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
