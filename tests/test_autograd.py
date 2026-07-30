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


def vjp_grad_check(
    fn, x: Tensor, upstream, eps: float = 1e-5, tol: float = 2e-5
) -> float:
    """Check a vector-Jacobian product using a non-uniform cotangent."""
    x.zero_grad()
    out = fn(x)
    upstream = np.asarray(upstream, dtype=np.float64)
    assert upstream.shape == out.shape
    out.backward(upstream)
    analytical = x.grad.copy()

    numerical = np.zeros_like(x.data)
    for idx in np.ndindex(x.shape):
        original = float(x.data[idx])
        x.data[idx] = original + eps
        plus = float(np.sum(fn(x).data * upstream))
        x.data[idx] = original - eps
        minus = float(np.sum(fn(x).data * upstream))
        x.data[idx] = original
        numerical[idx] = (plus - minus) / (2.0 * eps)

    err = float(np.max(np.abs(analytical - numerical)))
    assert err < tol, (
        f"VJP check failed: max error = {err:.2e} > tol {tol:.2e}\n"
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

    def test_matmul_numpy_shapes_with_random_vjp(self):
        """Vectors and singleton batch axes follow full np.matmul semantics."""
        shape_pairs = [
            ((3,), (3,)),
            ((3,), (3, 2)),
            ((2, 3), (3,)),
            ((3,), (2, 3, 4)),
            ((2, 4, 3), (3,)),
            ((1, 2, 3), (4, 3, 2)),
            ((4, 2, 3), (1, 3, 2)),
            ((2, 1, 2, 3), (1, 4, 3, 2)),
        ]
        rng = np.random.default_rng(123)
        for a_shape, b_shape in shape_pairs:
            a_data = rng.standard_normal(a_shape)
            b_data = rng.standard_normal(b_shape)
            expected = np.matmul(a_data, b_data)
            upstream = rng.standard_normal(expected.shape)

            a = Tensor(a_data, requires_grad=True)
            b_constant = Tensor(b_data)
            vjp_grad_check(
                lambda x: ops.matmul(x, b_constant), a, upstream, tol=3e-5
            )

            a_constant = Tensor(a_data)
            b = Tensor(b_data, requires_grad=True)
            vjp_grad_check(
                lambda x: ops.matmul(a_constant, x), b, upstream, tol=3e-5
            )

            np.testing.assert_allclose(
                ops.matmul(Tensor(a_data), Tensor(b_data)).data, expected
            )

    def test_tensor_operators_interoperate_with_numpy_arrays(self):
        rng = np.random.default_rng(456)
        left_data = rng.standard_normal((2, 3))
        right_data = rng.standard_normal((3, 2))
        upstream = rng.standard_normal((2, 2))

        left = Tensor(left_data, requires_grad=True)
        vjp_grad_check(lambda x: x @ right_data, left, upstream)

        right = Tensor(right_data, requires_grad=True)
        vjp_grad_check(lambda x: left_data @ x, right, upstream)

        denominator = Tensor([-2.0, 0.5, 4.0], requires_grad=True)
        quotient = np.array([1.0, 2.0, 3.0]) / denominator
        assert isinstance(quotient, Tensor)
        quotient.backward(np.array([0.5, -1.0, 2.0]))
        np.testing.assert_allclose(
            denominator.grad,
            -np.array([1.0, 2.0, 3.0])
            / denominator.data ** 2
            * np.array([0.5, -1.0, 2.0]),
        )

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

    def test_div_tensor_broadcast_and_negative_denominator(self):
        rng = np.random.default_rng(321)
        a_data = rng.standard_normal((2, 3))
        b_data = np.array([-0.75, 1.25, -2.5])
        upstream = rng.standard_normal((2, 3))

        a = Tensor(a_data, requires_grad=True)
        vjp_grad_check(lambda x: x / Tensor(b_data), a, upstream)

        b = Tensor(b_data, requires_grad=True)
        vjp_grad_check(lambda x: Tensor(a_data) / x, b, upstream)
        np.testing.assert_allclose((Tensor(a_data) / Tensor(b_data)).data,
                                   a_data / b_data)

    def test_reverse_division_gradient(self):
        denominator = Tensor(np.array([-2.0, 0.5, 4.0]), requires_grad=True)
        upstream = np.array([0.5, -1.0, 2.0])
        vjp_grad_check(lambda x: 3.0 / x, denominator, upstream)

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
        """Large magnitudes keep both selected and unselected branches finite."""
        a = Tensor(np.array([-1e4, 0.0, 1e4]), requires_grad=True)
        with np.errstate(over="raise", invalid="raise"):
            out = ops.sigmoid(a)
            out.backward(np.array([0.25, -0.5, 2.0]))
        np.testing.assert_allclose(out.data, [0.0, 0.5, 1.0])
        assert np.all(np.isfinite(a.grad))

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


class TestMaskedSoftmax:
    """A row of all -inf is defined as zero weights, not NaN."""

    def test_fully_masked_row_is_all_zeros(self):
        x = Tensor(
            [[-np.inf, -np.inf, -np.inf], [1.0, 2.0, 3.0]],
            requires_grad=True,
        )
        weights = ops.softmax(x)

        np.testing.assert_array_equal(weights.data[0], np.zeros(3))
        assert not np.isnan(weights.data).any()
        np.testing.assert_allclose(weights.data[1].sum(), 1.0, atol=1e-12)

    def test_fully_masked_row_has_zero_gradient(self):
        x = Tensor(
            [[-np.inf, -np.inf], [0.5, -0.5]],
            requires_grad=True,
        )
        weights = ops.softmax(x)
        weights.backward(np.array([[3.0, -2.0], [1.0, 4.0]]))

        np.testing.assert_array_equal(x.grad[0], np.zeros(2))
        assert np.isfinite(x.grad).all()

    def test_partially_masked_row_renormalises_over_visible_keys(self):
        x = Tensor([[1.0, -np.inf, 3.0]])
        weights = ops.softmax(x).data[0]

        visible = np.exp(np.array([1.0, 3.0]) - 3.0)
        visible /= visible.sum()
        np.testing.assert_allclose(weights, [visible[0], 0.0, visible[1]], atol=1e-12)

    def test_unmasked_softmax_is_bitwise_unchanged(self):
        rng = np.random.default_rng(11)
        data = rng.standard_normal((4, 6)) * 30.0
        shifted = data - data.max(axis=-1, keepdims=True)
        reference = np.exp(shifted)
        reference /= reference.sum(axis=-1, keepdims=True)

        np.testing.assert_array_equal(ops.softmax(Tensor(data)).data, reference)


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

    def test_transpose_negative_axes_random_vjp(self):
        a = make((2, 3, 4), 9)
        upstream = np.random.default_rng(10).standard_normal((2, 4, 3))
        vjp_grad_check(lambda x: ops.transpose(x, (0, -1, -2)), a, upstream)

    def test_concat(self):
        a, b = make((3, 4), 0), make((3, 4), 1)
        grad_check(lambda x: ops.concat([x, b], axis=0), a)
        grad_check(lambda x: ops.concat([a, x], axis=1), b)

    def test_concat_materialises_generator(self):
        a, b = make((2, 3), 11), make((1, 3), 12)
        out = ops.concat((tensor for tensor in (a, b)), axis=0)
        upstream = np.arange(9.0).reshape(3, 3)
        out.backward(upstream)
        np.testing.assert_array_equal(a.grad, upstream[:2])
        np.testing.assert_array_equal(b.grad, upstream[2:])

    def test_concat_rejects_empty_iterable(self):
        with pytest.raises(ValueError, match="at least one"):
            ops.concat(iter(()))


class TestLoss:
    def test_cross_entropy(self):
        rng = np.random.default_rng(42)
        logits = Tensor(rng.standard_normal((8, 5)), requires_grad=True)
        targets = rng.integers(0, 5, size=(8,))
        grad_check(lambda x: ops.cross_entropy(x, targets), logits)

    def test_cross_entropy_generalised_leading_dims(self):
        rng = np.random.default_rng(43)
        logits = Tensor(rng.standard_normal((2, 3, 5)), requires_grad=True)
        targets = rng.integers(0, 5, size=(2, 3))
        vjp_grad_check(
            lambda x: ops.cross_entropy(x, targets),
            logits,
            np.asarray(1.7),
        )

    def test_cross_entropy_extreme_logits_matches_logsumexp(self):
        data = np.array([[-1000.0, 0.0], [1000.0, -1000.0]])
        targets = np.array([0, 1])
        logits = Tensor(data, requires_grad=True)
        loss = ops.cross_entropy(logits, targets)

        reference = np.mean(
            np.logaddexp.reduce(data, axis=-1)
            - data[np.arange(targets.size), targets]
        )
        np.testing.assert_allclose(loss.data, reference, atol=1e-12)
        loss.backward()
        np.testing.assert_allclose(
            logits.grad,
            np.array([[-0.5, 0.5], [0.5, -0.5]]),
            atol=1e-12,
        )

        shifted = ops.cross_entropy(Tensor(data + 1e6), targets)
        np.testing.assert_allclose(shifted.data, loss.data, atol=1e-10)

    def test_cross_entropy_accepts_one_dimensional_logits(self):
        logits = Tensor([2.0, -1.0, 0.5], requires_grad=True)
        target = np.asarray(2, dtype=np.int64)
        loss = ops.cross_entropy(logits, target)
        loss.backward()
        assert loss.shape == ()
        assert logits.grad.shape == logits.shape

    def test_cross_entropy_snapshots_mutable_targets_for_backward(self):
        logits = Tensor([[2.0, -1.0], [-0.5, 1.5]], requires_grad=True)
        targets = np.array([0, 1], dtype=np.int64)
        loss = ops.cross_entropy(logits, targets)
        targets[:] = [1, 0]
        loss.backward()

        shifted = logits.data - logits.data.max(axis=-1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        expected = probabilities
        expected[np.arange(2), [0, 1]] -= 1.0
        expected /= 2.0
        np.testing.assert_allclose(logits.grad, expected, atol=1e-12)

    def test_ignore_index_excludes_positions_from_the_mean(self):
        rng = np.random.default_rng(5)
        data = rng.standard_normal((5, 4))
        targets = np.array([1, -100, 3, -100, 0])
        logits = Tensor(data, requires_grad=True)

        loss = ops.cross_entropy(logits, targets, ignore_index=-100)

        scored = np.array([0, 2, 4])
        reference = ops.cross_entropy(
            Tensor(data[scored]), targets[scored]
        )
        np.testing.assert_allclose(loss.data, reference.data, atol=1e-14)

    def test_unused_ignore_index_matches_the_plain_loss(self):
        """The no-gather fast path must be exactly the ordinary computation."""
        rng = np.random.default_rng(8)
        data = rng.standard_normal((4, 3))
        targets = np.array([0, 2, 1, 2])

        plain = Tensor(data, requires_grad=True)
        ops.cross_entropy(plain, targets).backward()
        with_flag = Tensor(data, requires_grad=True)
        ops.cross_entropy(with_flag, targets, ignore_index=-1).backward()

        np.testing.assert_array_equal(plain.grad, with_flag.grad)

    def test_ignore_index_gives_ignored_positions_zero_gradient(self):
        rng = np.random.default_rng(6)
        data = rng.standard_normal((4, 3))
        targets = np.array([2, -1, 0, -1])
        logits = Tensor(data, requires_grad=True)

        ops.cross_entropy(logits, targets, ignore_index=-1).backward()

        np.testing.assert_array_equal(logits.grad[1], np.zeros(3))
        np.testing.assert_array_equal(logits.grad[3], np.zeros(3))
        # Scored rows match a loss computed over those rows alone.
        scored = Tensor(data[[0, 2]], requires_grad=True)
        ops.cross_entropy(scored, np.array([2, 0])).backward()
        np.testing.assert_allclose(logits.grad[[0, 2]], scored.grad, atol=1e-14)

    def test_ignore_index_is_gradient_checked(self):
        rng = np.random.default_rng(7)
        logits = Tensor(rng.standard_normal((3, 2, 4)), requires_grad=True)
        targets = np.array([[1, -1], [-1, 3], [0, 2]])
        vjp_grad_check(
            lambda x: ops.cross_entropy(x, targets, ignore_index=-1),
            logits,
            np.asarray(0.75),
        )

    def test_ignore_index_tolerates_masked_logits_on_ignored_rows(self):
        logits = Tensor([[0.5, 1.0], [-np.inf, -np.inf]], requires_grad=True)
        loss = ops.cross_entropy(logits, np.array([0, -1]), ignore_index=-1)
        loss.backward()

        assert np.isfinite(loss.data)
        np.testing.assert_array_equal(logits.grad[1], np.zeros(2))

    def test_cross_entropy_rejects_all_ignored_targets(self):
        logits = Tensor(np.zeros((2, 3)), requires_grad=True)
        with pytest.raises(ValueError, match="no scored target"):
            ops.cross_entropy(logits, np.array([-1, -1]), ignore_index=-1)

    def test_cross_entropy_rejects_non_integer_ignore_index(self):
        logits = Tensor(np.zeros((2, 3)))
        with pytest.raises(TypeError, match="ignore_index"):
            ops.cross_entropy(logits, np.array([0, 1]), ignore_index=1.5)

    def test_cross_entropy_still_rejects_out_of_range_scored_targets(self):
        logits = Tensor(np.zeros((2, 3)))
        with pytest.raises(ValueError, match=r"\[0, 3\)"):
            ops.cross_entropy(logits, np.array([5, -1]), ignore_index=-1)

    def test_cross_entropy_rejects_fully_masked_row(self):
        logits = Tensor([[0.5, 1.0], [-np.inf, -np.inf]], requires_grad=True)
        with pytest.raises(ValueError, match="finite logit"):
            ops.cross_entropy(logits, np.array([0, 1]))

    def test_cross_entropy_allows_masked_classes_beside_a_finite_one(self):
        logits = Tensor([[0.0, -np.inf]], requires_grad=True)
        loss = ops.cross_entropy(logits, np.array([0]))
        loss.backward()

        np.testing.assert_allclose(loss.data, 0.0, atol=1e-12)
        np.testing.assert_allclose(logits.grad, [[0.0, 0.0]], atol=1e-12)

    def test_cross_entropy_validates_inputs(self):
        cases = [
            (Tensor(1.0), np.asarray(0), ValueError),
            (Tensor(np.empty((0, 3))), np.empty((0,), dtype=int), ValueError),
            (Tensor(np.empty((2, 0))), np.zeros(2, dtype=int), ValueError),
            (Tensor(np.zeros((2, 3))), np.zeros((2, 1), dtype=int), ValueError),
            (Tensor(np.zeros((2, 3))), np.array([0.0, 1.0]), TypeError),
            (Tensor(np.zeros((2, 3))), np.array([-1, 0]), ValueError),
            (Tensor(np.zeros((2, 3))), np.array([0, 3]), ValueError),
        ]
        for logits, targets, error in cases:
            with pytest.raises(error):
                ops.cross_entropy(logits, targets)


class TestBackwardLifecycle:
    def test_repeated_backward_accumulates_only_into_leaves(self):
        x = Tensor(2.0, requires_grad=True)
        intermediate = x * x
        loss = intermediate * x

        loss.backward()
        np.testing.assert_allclose(x.grad, 12.0)
        np.testing.assert_allclose(intermediate.grad, 2.0)

        loss.backward()
        np.testing.assert_allclose(x.grad, 24.0)
        np.testing.assert_allclose(intermediate.grad, 2.0)

    def test_leaf_backward_accumulates(self):
        x = Tensor(2.0, requires_grad=True)
        x.backward()
        x.backward()
        np.testing.assert_allclose(x.grad, 2.0)

    def test_non_scalar_default_is_sum_vjp(self):
        x = Tensor([1.0, 2.0, -3.0], requires_grad=True)
        (x * x).backward()
        np.testing.assert_allclose(x.grad, [2.0, 4.0, -6.0])

    def test_backward_rejects_wrong_cotangent_shape(self):
        x = Tensor([1.0, 2.0], requires_grad=True)
        with pytest.raises(ValueError, match="gradient shape mismatch"):
            (x * x).backward(np.ones((2, 1)))

    def test_gradient_free_node_inside_a_graph_is_skipped(self):
        """A node with no gradient of its own must not try to propagate one."""
        constants = ops.concat([Tensor([1.0]), Tensor([2.0])])
        assert not constants.requires_grad
        weight = Tensor([1.0, 1.0], requires_grad=True)

        ops.sum(constants * weight).backward()
        np.testing.assert_allclose(weight.grad, [1.0, 2.0])
        assert constants.grad is None

    def test_iterative_topology_supports_deep_graph(self):
        x = Tensor(1.0, requires_grad=True)
        result = x
        for _ in range(5000):
            result = result + 1.0
        result.backward()
        np.testing.assert_allclose(x.grad, 1.0)


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
        TestBasicOps, TestActivations, TestMaskedSoftmax, TestReductions,
        TestShapeOps, TestLoss, TestBackwardLifecycle, TestComposed,
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
