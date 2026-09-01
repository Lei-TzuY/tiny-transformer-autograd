import numpy as np
import pytest

from nn import (
    GPT,
    PersistentGPTKVCache,
    fork_persistent_gpt_kv_cache,
    infer_gpt_with_persistent_kv_cache,
)


def _model():
    np.random.seed(1502)
    return GPT(
        vocab_size=19,
        context_len=10,
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


def test_fork_shares_exact_immutable_prefix_nodes_without_kv_copy():
    model = _model()
    root = PersistentGPTKVCache(model)
    infer_gpt_with_persistent_kv_cache(
        model,
        np.array([[1, 2, 3, 4]], dtype=np.int64),
        root,
    )
    child = fork_persistent_gpt_kv_cache(root)

    assert child is not root
    assert child._lock is not root._lock
    assert child.length == root.length == 4
    assert child.segment_count == root.segment_count == 1
    assert child._model_versions == root._model_versions
    for root_layer, child_layer in zip(root._layers, child._layers):
        assert child_layer is not root_layer
        assert child_layer.head is root_layer.head
        assert child_layer.head.key is root_layer.head.key
        assert child_layer.head.value is root_layer.head.value
        assert not root_layer.head.key.flags.writeable
        assert not root_layer.head.value.flags.writeable


def test_sibling_appends_add_private_heads_while_retaining_shared_ancestor():
    model = _model()
    root = PersistentGPTKVCache(model)
    infer_gpt_with_persistent_kv_cache(model, np.array([[1, 2]], dtype=np.int64), root)
    root_heads = [layer.head for layer in root._layers]

    left = root.fork()
    right = root.fork()
    infer_gpt_with_persistent_kv_cache(model, np.array([[3]], dtype=np.int64), left)
    infer_gpt_with_persistent_kv_cache(model, np.array([[4]], dtype=np.int64), right)

    assert root.length == 2
    assert root.segment_count == 1
    assert left.length == right.length == 3
    assert left.segment_count == right.segment_count == 2
    for index, root_head in enumerate(root_heads):
        assert root._layers[index].head is root_head
        assert left._layers[index].head is not right._layers[index].head
        assert left._layers[index].head.prev is root_head
        assert right._layers[index].head.prev is root_head
        assert not np.shares_memory(
            left._layers[index].head.key,
            right._layers[index].head.key,
        )
        assert not np.shares_memory(
            left._layers[index].head.value,
            right._layers[index].head.value,
        )

    _, root_legacy = model.infer(np.array([[1, 2]], dtype=np.int64))
    _, left_legacy = model.infer(np.array([[1, 2, 3]], dtype=np.int64))
    _, right_legacy = model.infer(np.array([[1, 2, 4]], dtype=np.int64))
    for actual, expected in zip(root.snapshot(), root_legacy):
        np.testing.assert_allclose(actual["k"], expected["k"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual["v"], expected["v"], rtol=1e-12, atol=1e-12)
    for actual, expected in zip(left.snapshot(), left_legacy):
        np.testing.assert_allclose(actual["k"], expected["k"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual["v"], expected["v"], rtol=1e-12, atol=1e-12)
    for actual, expected in zip(right.snapshot(), right_legacy):
        np.testing.assert_allclose(actual["k"], expected["k"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual["v"], expected["v"], rtol=1e-12, atol=1e-12)


def test_clearing_one_fork_does_not_touch_shared_parent_or_sibling():
    model = _model()
    root = PersistentGPTKVCache(model)
    infer_gpt_with_persistent_kv_cache(model, np.array([[5, 6, 7]], dtype=np.int64), root)
    left = root.fork()
    right = root.fork()
    shared_heads = [layer.head for layer in root._layers]

    left.clear()
    assert left.length == 0
    assert left.segment_count == 0
    assert root.length == right.length == 3
    for index, head in enumerate(shared_heads):
        assert root._layers[index].head is head
        assert right._layers[index].head is head


def test_empty_fork_is_independent_empty_cache():
    model = _model()
    root = PersistentGPTKVCache(model)
    child = root.fork()
    assert child.length == root.length == 0
    assert child.segment_count == root.segment_count == 0
    infer_gpt_with_persistent_kv_cache(model, np.array([[1]], dtype=np.int64), child)
    assert child.length == 1
    assert root.length == 0


def test_fork_rejects_stale_model_before_child_publication():
    model = _model()
    root = PersistentGPTKVCache(model)
    infer_gpt_with_persistent_kv_cache(model, np.array([[1, 2]], dtype=np.int64), root)
    model.head.weight.data[0, 0] += 0.25
    with pytest.raises(RuntimeError, match="model tensors changed"):
        fork_persistent_gpt_kv_cache(root)
    assert root.length == 2


def test_fork_type_validation_and_rng_neutrality():
    with pytest.raises(TypeError, match="PersistentGPTKVCache"):
        fork_persistent_gpt_kv_cache(None)

    model = _model()
    root = PersistentGPTKVCache(model)
    infer_gpt_with_persistent_kv_cache(model, np.array([[1, 2]], dtype=np.int64), root)
    np.random.seed(7771)
    before = np.random.get_state()
    child = root.fork()
    after = np.random.get_state()
    assert _rng_state_equal(before, after)
    assert child.length == root.length


def test_fork_state_validation_does_not_walk_ancestor_chain(monkeypatch):
    model = _model()
    cache = PersistentGPTKVCache(model)
    infer_gpt_with_persistent_kv_cache(model, np.array([[1, 2]], dtype=np.int64), cache)
    for token in (3, 4, 5):
        infer_gpt_with_persistent_kv_cache(
            model,
            np.array([[token]], dtype=np.int64),
            cache,
        )
    assert cache.segment_count == 4

    # Fork is allowed to inspect only each layer head plus scalar metadata.  Walking
    # older nodes would touch this poisoned second ancestor and fail the regression.
    poisoned = []
    for layer in cache._layers:
        node = layer.head.prev
        poisoned.append((node, node.total_length))
        node.total_length = -123
    try:
        child = fork_persistent_gpt_kv_cache(cache)
        assert child.length == cache.length
        assert child.segment_count == cache.segment_count
    finally:
        for node, total in poisoned:
            node.total_length = total

    # A real inference must walk the chain and therefore fail closed on that same kind
    # of corruption rather than publishing another segment.
    bad = cache._layers[0].head.prev
    original = bad.total_length
    bad.total_length = -123
    try:
        with pytest.raises(RuntimeError, match="total-length"):
            infer_gpt_with_persistent_kv_cache(
                model,
                np.array([[6]], dtype=np.int64),
                cache,
            )
    finally:
        bad.total_length = original
