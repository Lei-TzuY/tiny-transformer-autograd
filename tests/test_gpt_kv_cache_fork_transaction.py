import numpy as np
import pytest

from nn import GPT, GPTKVCache, fork_gpt_kv_cache, infer_gpt_with_kv_cache
from nn.kv_cache import KVCacheBuffer


def _model():
    np.random.seed(1501)
    return GPT(
        vocab_size=17,
        context_len=8,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        dropout=0.0,
    )


def _rng_state_equal(left, right):
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def test_mid_fork_append_failure_leaves_source_exactly_unchanged(monkeypatch):
    model = _model()
    source = GPTKVCache(model)
    infer_gpt_with_kv_cache(model, np.array([[1, 2, 3]], dtype=np.int64), source)

    before_snapshot = source.snapshot()
    before_versions = source._model_versions
    before_pointers = [
        (
            buffer.view()["k"].__array_interface__["data"][0],
            buffer.view()["v"].__array_interface__["data"][0],
        )
        for buffer in source._buffers
    ]
    np.random.seed(6677)
    before_rng = np.random.get_state()

    original_append = KVCacheBuffer.append
    calls = 0

    def failing_append(self, key, value):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected child append failure")
        return original_append(self, key, value)

    monkeypatch.setattr(KVCacheBuffer, "append", failing_append)
    with pytest.raises(RuntimeError, match="injected child append failure"):
        fork_gpt_kv_cache(source)

    assert source.length == 3
    assert source._model_versions == before_versions
    after_pointers = [
        (
            buffer.view()["k"].__array_interface__["data"][0],
            buffer.view()["v"].__array_interface__["data"][0],
        )
        for buffer in source._buffers
    ]
    assert after_pointers == before_pointers
    for expected, actual in zip(before_snapshot, source.snapshot()):
        np.testing.assert_array_equal(actual["k"], expected["k"])
        np.testing.assert_array_equal(actual["v"], expected["v"])
    assert _rng_state_equal(before_rng, np.random.get_state())


def test_successful_fork_preserves_source_storage_pointers():
    model = _model()
    source = GPTKVCache(model)
    infer_gpt_with_kv_cache(model, np.array([[4, 5]], dtype=np.int64), source)
    before = [
        (
            buffer.view()["k"].__array_interface__["data"][0],
            buffer.view()["v"].__array_interface__["data"][0],
        )
        for buffer in source._buffers
    ]
    child = fork_gpt_kv_cache(source)
    after = [
        (
            buffer.view()["k"].__array_interface__["data"][0],
            buffer.view()["v"].__array_interface__["data"][0],
        )
        for buffer in source._buffers
    ]
    assert before == after
    assert child.length == source.length
