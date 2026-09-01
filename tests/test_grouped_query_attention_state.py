import numpy as np
import pytest

from engine.tensor import Tensor
from nn import GroupedQueryAttention


def test_parameter_count_reflects_compact_kv_projections():
    attention = GroupedQueryAttention(8, 4, num_kv_heads=2, dropout=0.0)

    # Q:64, K:32, V:32, out weight:64, out bias:8.
    assert attention.param_count() == 200
    names = [name for name, _ in attention.named_parameters()]
    assert names == [
        "W_q.weight",
        "W_k.weight",
        "W_v.weight",
        "out_proj.weight",
        "out_proj.bias",
    ]


def test_mqa_parameter_count_is_smaller_than_mha_endpoint():
    mqa = GroupedQueryAttention(12, 6, num_kv_heads=1)
    mha = GroupedQueryAttention(12, 6, num_kv_heads=6)
    assert mqa.param_count() < mha.param_count()
    assert mha.param_count() - mqa.param_count() == 2 * (12 - 2) * 12


def test_state_dict_round_trip_restores_exact_forward():
    np.random.seed(801)
    source = GroupedQueryAttention(8, 4, num_kv_heads=2, dropout=0.0)
    target = GroupedQueryAttention(8, 4, num_kv_heads=2, dropout=0.0)
    x = Tensor(np.random.randn(2, 3, 8))
    expected = source(x).data.copy()
    state = source.state_dict()

    target.load_state_dict(state)
    actual = target(Tensor(x.data)).data

    np.testing.assert_array_equal(actual, expected)
    for name, value in state.items():
        np.testing.assert_array_equal(dict(target.state_dict())[name], value)


def test_state_dict_arrays_are_independent_copies():
    np.random.seed(802)
    attention = GroupedQueryAttention(8, 4, num_kv_heads=2)
    state = attention.state_dict()
    original = attention.W_k.weight.data.copy()

    state["W_k.weight"][...] = 123.0

    np.testing.assert_array_equal(attention.W_k.weight.data, original)


def test_strict_load_rejects_different_kv_projection_shapes_transactionally():
    np.random.seed(803)
    compact = GroupedQueryAttention(8, 4, num_kv_heads=2)
    full = GroupedQueryAttention(8, 4, num_kv_heads=4)
    before = full.state_dict()

    with pytest.raises(ValueError):
        full.load_state_dict(compact.state_dict())

    after = full.state_dict()
    for name in before:
        np.testing.assert_array_equal(after[name], before[name])


def test_eval_mode_disables_attention_dropout_recursively():
    np.random.seed(804)
    attention = GroupedQueryAttention(8, 4, num_kv_heads=2, dropout=0.5)
    x = np.random.randn(2, 3, 8)
    attention.eval()

    forward = attention(Tensor(x)).data
    inferred, _ = attention.infer(x)

    assert attention.training is False
    assert attention.attn_drop.training is False
    np.testing.assert_allclose(forward, inferred, rtol=0, atol=2e-14)


def test_train_mode_reenables_dropout_and_consumes_rng():
    np.random.seed(805)
    attention = GroupedQueryAttention(8, 4, num_kv_heads=2, dropout=0.5)
    attention.eval().train()
    x = Tensor(np.random.randn(1, 3, 8))
    before = np.random.get_state()

    attention(x)

    after = np.random.get_state()
    assert attention.training is True
    assert attention.attn_drop.training is True
    assert not np.array_equal(before[1], after[1]) or before[2:] != after[2:]


def test_zero_grad_clears_all_compact_projection_gradients():
    np.random.seed(806)
    attention = GroupedQueryAttention(8, 4, num_kv_heads=2, dropout=0.0)
    x = Tensor(np.random.randn(1, 3, 8), requires_grad=True)
    attention(x).backward(np.ones((1, 3, 8)))
    assert any(np.any(parameter.grad != 0.0) for parameter in attention.parameters())

    attention.zero_grad()

    for parameter in attention.parameters():
        np.testing.assert_array_equal(parameter.grad, np.zeros(parameter.shape))
