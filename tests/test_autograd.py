"""
test_autograd.py — Numerical gradient checks for all autograd primitives.

Each test computes the analytical gradient via .backward() and then
independently estimates it with central finite differences.  Passing means
the two agree to within a tolerance that is well above numerical noise.

Run:
    pytest tests/test_autograd.py -v
or:
    python tests/test_autograd.py
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.tensor import Tensor
import engine.ops as ops


# ---------------------------------------------------------------------------
# Utility: central finite-difference gradient check
# ---------------------------------------------------------------------------
def grad_check(fn, x: Tensor, eps: float = 1e-4, tol: float = 1e-5) -> float:
    """
    Compare fn's analytical gradient on x (via backward) with
    central finite differences.  Returns the max absolute error.
    """
    # Analytical grad
    x.grad = np.zeros_like(x.data)
    out = fn(x)
    out.backward()
    analytical = x.grad.copy()

    # Numerical grad
    numerical = np.zeros_like(x.data)
    it = np.nditer(x.data, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = float(x.data[idx])

        x.data[idx] = orig + eps
        fp = float(fn(x).data.sum())   # scalar-ise arbitrary shaped output

        x.data[idx] = orig - eps
        fm = float(fn(x).data.sum())

        x.data[idx] = orig
        numerical[idx] = (fp - fm) / (2 * eps)
        it.iternext()

    err = float(np.max(np.abs(analytical - numerical)))
    assert err < tol, (
        f"Gradient check failed: max error = {err:.2e} > tol {tol:.2e}\n"
        f"  analytical = {analytical}\n  numerical  = {numerical}"
    )
    return err


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def make(shape, seed=0):
    """Reproducible small random Tensor with requires_grad."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal(shape) * 0.5
    return Tensor(data, requires_grad=True)


class TestBasicOps:
    def test_add(self):
        a, b = make((3, 4), 0), make((3, 4), 1)
        grad_check(lambda x: ops.add(x, b), a)
        grad_check(lambda x: ops.add(a, x), b)

    def test_add_broadcast(self):
        a = make((3, 4), 0)
        b = make((4,), 1)
        grad_check(lambda x: ops.add(a, x), b)

    def test_mul(self):
        a, b = make((3, 4), 0), make((3, 4), 1)
        grad_check(lambda x: ops.mul(x, b), a)
        grad_check(lambda x: ops.mul(a, x), b)

    def test_matmul_2d(self):
        a = make((3, 4), 0)
        b = make((4, 5), 1)
        grad_check(lambda x: ops.matmul(x, b), a)
        grad_check(lambda x: ops.matmul(a, x), b)

    def test_matmul_batched(self):
        # (B, M, K) @ (K, N) — tests the batch-dim summation fix
        a = make((2, 3, 4), 0)
        b = make((4, 5), 1)
        grad_check(lambda x: ops.matmul(x, b), a)
        grad_check(lambda x: ops.matmul(a, x), b)

    def test_pow(self):
        a = make((3,), 0)
        a.data = np.abs(a.data) + 1.0   # keep ≥1 so 1/x gradients stay bounded
        for exp in [2, 3, 0.5, -1]:
            grad_check(lambda x, e=exp: x ** e, a)

    def test_neg(self):
        a = make((3, 3), 0)
        grad_check(lambda x: -x, a)

    def test_sub(self):
        a, b = make((3,), 0), make((3,), 1)
        grad_check(lambda x: x - b, a)

    def test_div_scalar(self):
        a = make((4,), 0)
        grad_check(lambda x: x / 3.0, a)

    def test_getitem_fancy(self):
        a = make((5, 3), 0)
        idx = np.array([0, 2, 4])
        grad_check(lambda x: x[idx], a)

    def test_getitem_slice(self):
        a = make((6, 4), 0)
        grad_check(lambda x: x[2:5], a)


class TestActivations:
    def test_relu(self):
        a = make((4, 5), 0)
        grad_check(lambda x: ops.relu(x), a)

    def test_sigmoid(self):
        a = make((4, 5), 0)
        grad_check(lambda x: ops.sigmoid(x), a)

    def test_sigmoid_extremes(self):
        """Numerically stable: very large / very small inputs should not NaN."""
        a = make((3,), 0)
        a.data = np.array([-100.0, 0.0, 100.0])
        out = ops.sigmoid(a)
        assert np.all(np.isfinite(out.data)), "sigmoid output not finite"

    def test_exp(self):
        a = make((3, 3), 0)
        a.data = np.clip(a.data, -2, 2)   # prevent overflow
        grad_check(lambda x: ops.exp(x), a)

    def test_log(self):
        a = make((3, 3), 0)
        a.data = np.abs(a.data) + 0.5
        grad_check(lambda x: ops.log(x), a)

    def test_tanh(self):
        a = make((4, 4), 0)
        grad_check(lambda x: ops.tanh(x), a)

    def test_gelu(self):
        a = make((4, 4), 0)
        grad_check(lambda x: ops.gelu(x), a)

    def test_softmax(self):
        a = make((3, 5), 0)
        grad_check(lambda x: ops.softmax(x), a)


class TestReductions:
    def test_sum_all(self):
        a = make((3, 4), 0)
        grad_check(lambda x: ops.sum(x), a)

    def test_sum_axis(self):
        a = make((3, 4), 0)
        grad_check(lambda x: ops.sum(x, axis=0), a)
        grad_check(lambda x: ops.sum(x, axis=1), a)
        grad_check(lambda x: ops.sum(x, axis=-1), a)

    def test_sum_keepdims(self):
        a = make((3, 4), 0)
        grad_check(lambda x: ops.sum(x, axis=1, keepdims=True), a)

    def test_mean_all(self):
        a = make((3, 4), 0)
        grad_check(lambda x: ops.mean(x), a)

    def test_mean_axis(self):
        a = make((3, 4), 0)
        grad_check(lambda x: ops.mean(x, axis=-1), a)
        grad_check(lambda x: ops.mean(x, axis=-1, keepdims=True), a)


class TestShapeOps:
    def test_reshape(self):
        a = make((12,), 0)
        grad_check(lambda x: ops.reshape(x, (3, 4)), a)

    def test_transpose_2d(self):
        a = make((3, 4), 0)
        grad_check(lambda x: ops.transpose(x, (1, 0)), a)

    def test_transpose_4d(self):
        a = make((2, 3, 4, 5), 0)
        grad_check(lambda x: ops.transpose(x, (0, 2, 1, 3)), a)

    def test_concat(self):
        a, b = make((3, 4), 0), make((3, 4), 1)
        grad_check(lambda x: ops.concat([x, b], axis=0), a)
        grad_check(lambda x: ops.concat([a, x], axis=1), b)


class TestLoss:
    def test_cross_entropy(self):
        rng = np.random.default_rng(42)
        logits = Tensor(rng.standard_normal((8, 5)), requires_grad=True)
        targets = rng.integers(0, 5, size=(8,))
        grad_check(lambda x: ops.cross_entropy(x, targets), logits)


class TestComposed:
    def test_layer_norm_manual(self):
        """LayerNorm via primitive ops — checks chain through mean/var/pow."""
        a = make((4, 8), 0)
        gamma = Tensor(np.ones(8))
        beta = Tensor(np.zeros(8))
        eps = 1e-5

        def layer_norm(x):
            mu = ops.mean(x, axis=-1, keepdims=True)
            diff = x - mu
            var = ops.mean(diff ** 2, axis=-1, keepdims=True)
            return diff * ((var + eps) ** -0.5) * gamma + beta

        grad_check(layer_norm, a)

    def test_attention_scores(self):
        """Scaled dot-product scores Q @ Kᵀ / √d."""
        rng = np.random.default_rng(7)
        B, T, d = 2, 4, 8
        Q = Tensor(rng.standard_normal((B, T, d)), requires_grad=True)
        K = Tensor(rng.standard_normal((B, T, d)))
        scale = d ** -0.5

        def scores(q):
            return ops.matmul(q, ops.transpose(K, (0, 2, 1))) * scale

        grad_check(scores, Q)


# ---------------------------------------------------------------------------
# Allow running directly with  python tests/test_autograd.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import traceback

    suites = [
        TestBasicOps, TestActivations, TestReductions,
        TestShapeOps, TestLoss, TestComposed,
    ]
    passed = failed = 0
    for suite_cls in suites:
        suite = suite_cls()
        for name in [m for m in dir(suite) if m.startswith("test_")]:
            try:
                getattr(suite, name)()
                print(f"  PASS  {suite_cls.__name__}.{name}")
                passed += 1
            except Exception:
                print(f"  FAIL  {suite_cls.__name__}.{name}")
                traceback.print_exc()
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
