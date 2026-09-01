import numpy as np
import pytest

from nn import GPT, GPTKVCache, fork_gpt_kv_cache, infer_gpt_with_kv_cache


def _model(*, rope=False):
    np.random.seed(1101)
    kwargs = {}
    if rope:
        kwargs.update(norm="rmsnorm", pos_encoding="rope", ffn="swiglu")
    return GPT(
        vocab_size=23,
        context_len=9,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        dropout=0.0,
        **kwargs,
    )


def _rng_state_equal(left, right):
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def test_fork_copies_live_prefix_without_sharing_storage():
    model = _model()
    source = GPTKVCache(model)
    prompt = np.array([[1, 2, 3]], dtype=np.int64)
    infer_gpt_with_kv_cache(model, prompt, source)

    source_snapshot = source.snapshot()
    source_versions = source._model_versions
    child = fork_gpt_kv_cache(source)

    assert child is not source
    assert child.length == source.length == 3
    assert child.capacity == source.capacity
    assert child._model_versions == source_versions
    child_snapshot = child.snapshot()
    for source_entry, child_entry, source_buffer, child_buffer in zip(
        source_snapshot, child_snapshot, source._buffers, child._buffers
    ):
        np.testing.assert_array_equal(child_entry["k"], source_entry["k"])
        np.testing.assert_array_equal(child_entry["v"], source_entry["v"])
        assert not np.shares_memory(source_buffer.view()["k"], child_buffer.view()["k"])
        assert not np.shares_memory(source_buffer.view()["v"], child_buffer.view()["v"])


def test_sibling_forks_append_independently_and_leave_parent_unchanged():
    model = _model(rope=True)
    parent = GPTKVCache(model)
    infer_gpt_with_kv_cache(model, np.array([[4, 5]], dtype=np.int64), parent)
    parent_before = parent.snapshot()

    left = fork_gpt_kv_cache(parent)
    right = fork_gpt_kv_cache(parent)
    infer_gpt_with_kv_cache(model, np.array([[6]], dtype=np.int64), left)
    infer_gpt_with_kv_cache(model, np.array([[7]], dtype=np.int64), right)

    assert parent.length == 2
    assert left.length == right.length == 3
    for before, after in zip(parent_before, parent.snapshot()):
        np.testing.assert_array_equal(after["k"], before["k"])
        np.testing.assert_array_equal(after["v"], before["v"])

    # The siblings retain the common parent prefix but own distinct appended tails.
    for left_entry, right_entry in zip(left.snapshot(), right.snapshot()):
        np.testing.assert_array_equal(left_entry["k"][..., :2, :], right_entry["k"][..., :2, :])
        np.testing.assert_array_equal(left_entry["v"][..., :2, :], right_entry["v"][..., :2, :])
        assert not np.shares_memory(left_entry["k"], right_entry["k"])
        assert not np.shares_memory(left_entry["v"], right_entry["v"])


def test_empty_and_cleared_cache_fork_to_fresh_empty_cache():
    model = _model()
    empty = GPTKVCache(model)
    child = fork_gpt_kv_cache(empty)
    assert child.length == 0
    assert not child.initialized
    assert child.storage_nbytes == 0

    infer_gpt_with_kv_cache(model, np.array([[1, 2]], dtype=np.int64), empty)
    allocated = empty.storage_nbytes
    empty.clear()
    assert empty.initialized
    assert empty.length == 0
    assert empty.storage_nbytes == allocated

    child = fork_gpt_kv_cache(empty)
    assert child.length == 0
    assert not child.initialized
    assert child.storage_nbytes == 0
    assert empty.storage_nbytes == allocated


def test_fork_rejects_stale_model_cache_before_copying():
    model = _model()
    cache = GPTKVCache(model)
    infer_gpt_with_kv_cache(model, np.array([[1, 2]], dtype=np.int64), cache)
    before = cache.snapshot()

    model.token_emb.weight.data[0, 0] += 0.25
    with pytest.raises(RuntimeError, match="model tensors changed"):
        fork_gpt_kv_cache(cache)

    assert cache.length == 2
    for expected, actual in zip(before, cache.snapshot()):
        np.testing.assert_array_equal(actual["k"], expected["k"])
        np.testing.assert_array_equal(actual["v"], expected["v"])


def test_fork_rejects_partially_initialized_source():
    model = _model()
    cache = GPTKVCache(model)
    key = np.zeros((1, model.num_heads, 1, model.d_model // model.num_heads))
    cache._buffers[0].append(key, key)
    with pytest.raises(RuntimeError, match="partially initialized"):
        fork_gpt_kv_cache(cache)


def test_fork_preserves_numpy_rng_state():
    model = _model()
    cache = GPTKVCache(model)
    infer_gpt_with_kv_cache(model, np.array([[3, 4]], dtype=np.int64), cache)
    np.random.seed(8877)
    before = np.random.get_state()
    fork_gpt_kv_cache(cache)
    after = np.random.get_state()
    assert _rng_state_equal(before, after)
