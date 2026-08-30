import numpy as np
import pytest

import nn.attention as attention_module
from nn import (
    KVCacheBuffer,
    MultiHeadAttention,
    SelfAttention,
    infer_with_kv_buffer,
)


def _snapshot(buffer):
    return None if not buffer.initialized else buffer.snapshot()


def _assert_snapshot_equal(buffer, before, length):
    assert buffer.length == length
    if before is None:
        assert not buffer.initialized
        return
    after = buffer.snapshot()
    np.testing.assert_array_equal(after["k"], before["k"])
    np.testing.assert_array_equal(after["v"], before["v"])


def test_rejects_wrong_public_types_before_buffer_mutation():
    np.random.seed(1210)
    attention = MultiHeadAttention(8, 2)
    x = np.random.randn(1, 1, 8)
    buffer = KVCacheBuffer(4)

    with pytest.raises(TypeError, match="buffer must be a KVCacheBuffer"):
        infer_with_kv_buffer(attention, x, {})
    with pytest.raises(TypeError, match="attention must be"):
        infer_with_kv_buffer(object(), x, buffer)
    with pytest.raises(TypeError, match="positions are only supported"):
        infer_with_kv_buffer(SelfAttention(8), x, buffer, positions=np.array([0]))

    assert not buffer.initialized
    assert buffer.length == 0


def test_capacity_overflow_fails_before_projection_or_mutation(monkeypatch):
    np.random.seed(1211)
    attention = MultiHeadAttention(8, 2)
    buffer = KVCacheBuffer(2)
    infer_with_kv_buffer(attention, np.random.randn(1, 2, 8), buffer)
    before = buffer.snapshot()

    called = False

    def forbidden(_):
        nonlocal called
        called = True
        raise AssertionError("projection should not run")

    monkeypatch.setattr(attention.W_q, "infer", forbidden)
    with pytest.raises(OverflowError, match="cache capacity"):
        infer_with_kv_buffer(attention, np.random.randn(1, 1, 8), buffer)

    assert not called
    _assert_snapshot_equal(buffer, before, 2)


def test_late_key_bias_failure_rolls_back_existing_buffer():
    np.random.seed(1212)
    attention = MultiHeadAttention(8, 2)
    buffer = KVCacheBuffer(4)
    infer_with_kv_buffer(attention, np.random.randn(1, 2, 8), buffer)
    before = buffer.snapshot()

    with pytest.raises(ValueError, match="does not broadcast"):
        infer_with_kv_buffer(
            attention,
            np.random.randn(1, 1, 8),
            buffer,
            key_bias=np.zeros((1, 9, 9)),
        )

    _assert_snapshot_equal(buffer, before, 2)


def test_first_call_failure_leaves_buffer_uninitialized():
    np.random.seed(1213)
    attention = SelfAttention(4)
    buffer = KVCacheBuffer(4)

    with pytest.raises(ValueError, match="does not broadcast"):
        infer_with_kv_buffer(
            attention,
            np.random.randn(1, 2, 4),
            buffer,
            key_bias=np.zeros((7, 7)),
        )

    assert not buffer.initialized
    assert buffer.length == 0
    assert buffer.storage_nbytes == 0


def test_late_output_projection_failure_rolls_back_visible_state(monkeypatch):
    np.random.seed(1214)
    attention = MultiHeadAttention(8, 2)
    buffer = KVCacheBuffer(4)
    infer_with_kv_buffer(attention, np.random.randn(1, 1, 8), buffer)
    before = buffer.snapshot()

    def fail(_):
        raise RuntimeError("injected output failure")

    monkeypatch.setattr(attention.out_proj, "infer", fail)
    with pytest.raises(RuntimeError, match="injected output failure"):
        infer_with_kv_buffer(attention, np.random.randn(1, 1, 8), buffer)

    _assert_snapshot_equal(buffer, before, 1)


def test_incompatible_initialized_buffer_is_rejected_before_projection(monkeypatch):
    np.random.seed(1215)
    attention = MultiHeadAttention(8, 2)
    buffer = KVCacheBuffer(4)
    # Wrong head count for the target attention.
    buffer.append(
        np.zeros((1, 1, 1, 4), dtype=np.float64),
        np.zeros((1, 1, 1, 4), dtype=np.float64),
    )
    before = buffer.snapshot()

    called = False

    def forbidden(_):
        nonlocal called
        called = True
        raise AssertionError("projection should not run")

    monkeypatch.setattr(attention.W_q, "infer", forbidden)
    with pytest.raises(ValueError, match="head count"):
        infer_with_kv_buffer(attention, np.random.randn(1, 1, 8), buffer)

    assert not called
    _assert_snapshot_equal(buffer, before, 1)


def test_non_rope_buffered_path_does_not_call_numpy_concatenate(monkeypatch):
    np.random.seed(1216)
    attention = MultiHeadAttention(8, 2)
    prefix = np.random.randn(1, 2, 8)
    token = np.random.randn(1, 1, 8)
    buffer = KVCacheBuffer(4)
    infer_with_kv_buffer(attention, prefix, buffer)

    def forbidden(*args, **kwargs):
        raise AssertionError("np.concatenate should not be used by buffered decode")

    monkeypatch.setattr(attention_module.np, "concatenate", forbidden)
    output, live = infer_with_kv_buffer(attention, token, buffer)

    assert output.shape == (1, 1, 8)
    assert live["k"].shape == (1, 2, 3, 4)
    assert buffer.length == 3


def test_legacy_dict_path_still_uses_its_historical_concatenate(monkeypatch):
    np.random.seed(1217)
    attention = MultiHeadAttention(8, 2)
    _, cache = attention.infer(np.random.randn(1, 1, 8))

    def forbidden(*args, **kwargs):
        raise AssertionError("legacy concatenate reached")

    monkeypatch.setattr(attention_module.np, "concatenate", forbidden)
    with pytest.raises(AssertionError, match="legacy concatenate reached"):
        attention.infer(np.random.randn(1, 1, 8), cache=cache)


def test_numpy_rng_state_is_unchanged():
    np.random.seed(1218)
    attention = MultiHeadAttention(8, 2)
    x = np.arange(8, dtype=np.float64).reshape(1, 1, 8)
    buffer = KVCacheBuffer(2)
    state = np.random.get_state()

    infer_with_kv_buffer(attention, x, buffer)

    after = np.random.get_state()
    assert state[0] == after[0]
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]
