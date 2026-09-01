import numpy as np

from nn import GPT, PersistentGPTKVCache, infer_gpt_with_persistent_kv_cache


def _model():
    np.random.seed(1508)
    return GPT(
        vocab_size=17,
        context_len=12,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=3,
        dropout=0.0,
    )


def _unique_storage_bytes(caches):
    arrays = {}
    nodes = set()
    for cache in caches:
        for layer in cache._layers:
            node = layer.head
            while node is not None:
                nodes.add(id(node))
                arrays[id(node.key)] = node.key
                arrays[id(node.value)] = node.value
                node = node.prev
    return sum(array.nbytes for array in arrays.values()), len(nodes), len(arrays)


def test_forks_add_no_physical_kv_storage_before_divergence():
    model = _model()
    root = PersistentGPTKVCache(model)
    prompt = np.array([[1, 2, 3, 4]], dtype=np.int64)
    infer_gpt_with_persistent_kv_cache(model, prompt, root)

    children = [root.fork() for _ in range(5)]
    unique_bytes, node_count, array_count = _unique_storage_bytes([root] + children)
    layers = len(model.blocks)

    assert unique_bytes == root.live_nbytes
    assert node_count == layers
    assert array_count == 2 * layers
    for child in children:
        assert child.live_nbytes == root.live_nbytes


def test_diverged_children_allocate_only_their_new_token_segments():
    model = _model()
    root = PersistentGPTKVCache(model)
    prompt = np.array([[1, 2, 3, 4]], dtype=np.int64)
    infer_gpt_with_persistent_kv_cache(model, prompt, root)
    prefix_bytes = root.live_nbytes
    assert prefix_bytes % prompt.shape[1] == 0
    bytes_per_token = prefix_bytes // prompt.shape[1]

    children = [root.fork() for _ in range(3)]
    for child, token in zip(children, (5, 6, 7)):
        infer_gpt_with_persistent_kv_cache(
            model,
            np.array([[token]], dtype=np.int64),
            child,
        )

    unique_bytes, node_count, array_count = _unique_storage_bytes([root] + children)
    layers = len(model.blocks)
    assert unique_bytes == prefix_bytes + 3 * bytes_per_token
    assert node_count == layers * 4
    assert array_count == node_count * 2
    assert root.segment_count == 1
    assert all(child.segment_count == 2 for child in children)


def test_deeper_branch_reuses_both_root_and_parent_segments():
    model = _model()
    root = PersistentGPTKVCache(model)
    infer_gpt_with_persistent_kv_cache(model, np.array([[1, 2]], dtype=np.int64), root)
    parent = root.fork()
    infer_gpt_with_persistent_kv_cache(model, np.array([[3]], dtype=np.int64), parent)
    child = parent.fork()

    for root_layer, parent_layer, child_layer in zip(
        root._layers,
        parent._layers,
        child._layers,
    ):
        assert parent_layer.head.prev is root_layer.head
        assert child_layer.head is parent_layer.head
        assert child_layer.head.prev is root_layer.head

    infer_gpt_with_persistent_kv_cache(model, np.array([[4]], dtype=np.int64), child)
    for root_layer, parent_layer, child_layer in zip(
        root._layers,
        parent._layers,
        child._layers,
    ):
        assert child_layer.head.prev is parent_layer.head
        assert parent_layer.head.prev is root_layer.head
