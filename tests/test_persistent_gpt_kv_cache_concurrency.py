import threading
import time

import numpy as np

from nn import (
    GPT,
    PersistentGPTKVCache,
    fork_persistent_gpt_kv_cache,
    infer_gpt_with_persistent_kv_cache,
)


def _model():
    np.random.seed(1507)
    return GPT(
        vocab_size=17,
        context_len=8,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        dropout=0.0,
    )


def test_fork_waits_for_source_inference_and_observes_committed_head(monkeypatch):
    model = _model()
    cache = PersistentGPTKVCache(model)
    entered = threading.Event()
    release = threading.Event()
    fork_done = threading.Event()
    errors = []
    children = []
    original = model.token_emb.infer

    def blocking(ids):
        entered.set()
        release.wait()
        return original(ids)

    monkeypatch.setattr(model.token_emb, "infer", blocking)

    def run_infer():
        try:
            infer_gpt_with_persistent_kv_cache(
                model,
                np.array([[1, 2]], dtype=np.int64),
                cache,
            )
        except BaseException as exc:
            errors.append(exc)

    def run_fork():
        try:
            children.append(fork_persistent_gpt_kv_cache(cache))
        except BaseException as exc:
            errors.append(exc)
        finally:
            fork_done.set()

    thread_a = threading.Thread(target=run_infer, daemon=True)
    thread_b = None
    thread_a.start()
    try:
        assert entered.wait(timeout=5)

        thread_b = threading.Thread(target=run_fork, daemon=True)
        thread_b.start()
        time.sleep(0.05)
        assert not fork_done.is_set()
    finally:
        # Always unblock the worker before allowing an assertion failure to escape;
        # otherwise pytest can hang forever waiting for a live non-daemon thread.
        release.set()
        thread_a.join(timeout=5)
        if thread_b is not None:
            thread_b.join(timeout=5)

    assert not thread_a.is_alive()
    assert thread_b is not None and not thread_b.is_alive()
    assert errors == []
    assert cache.length == 2
    assert len(children) == 1
    child = children[0]
    assert child.length == 2
    for source_layer, child_layer in zip(cache._layers, child._layers):
        assert child_layer.head is source_layer.head


def test_sibling_forks_do_not_share_one_decode_lock(monkeypatch):
    model = _model()
    root = PersistentGPTKVCache(model)
    infer_gpt_with_persistent_kv_cache(model, np.array([[1, 2]], dtype=np.int64), root)
    left = root.fork()
    right = root.fork()

    left_entered = threading.Event()
    left_release = threading.Event()
    right_entered = threading.Event()
    errors = []
    original = model.token_emb.infer

    class RightProbe(Exception):
        pass

    def selective(ids):
        token = int(np.asarray(ids)[0, 0])
        if token == 3:
            left_entered.set()
            left_release.wait()
        elif token == 4:
            right_entered.set()
            raise RightProbe("right sibling entered model inference")
        return original(ids)

    monkeypatch.setattr(model.token_emb, "infer", selective)

    def advance(cache, token):
        try:
            infer_gpt_with_persistent_kv_cache(
                model,
                np.array([[token]], dtype=np.int64),
                cache,
            )
        except BaseException as exc:
            errors.append(exc)

    thread_left = threading.Thread(target=advance, args=(left, 3), daemon=True)
    thread_right = None
    thread_left.start()
    try:
        assert left_entered.wait(timeout=5)

        thread_right = threading.Thread(target=advance, args=(right, 4), daemon=True)
        thread_right.start()
        assert right_entered.wait(timeout=5)
        thread_right.join(timeout=5)
        assert not thread_right.is_alive()
        assert right.length == 2
        assert left.length == 2
    finally:
        # A failed right-entry assertion must still release the intentionally blocked
        # left branch so the test process can terminate and report the real failure.
        left_release.set()
        thread_left.join(timeout=5)
        if thread_right is not None:
            thread_right.join(timeout=5)

    assert not thread_left.is_alive()
    assert thread_right is not None and not thread_right.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], RightProbe)
    assert left.length == 3
    assert right.length == 2
    assert root.length == 2

    monkeypatch.setattr(model.token_emb, "infer", original)
    infer_gpt_with_persistent_kv_cache(model, np.array([[4]], dtype=np.int64), right)
    assert right.length == 3


def test_concurrent_sibling_successes_keep_shared_ancestor_immutable():
    model = _model()
    root = PersistentGPTKVCache(model)
    infer_gpt_with_persistent_kv_cache(model, np.array([[1, 2]], dtype=np.int64), root)
    left = root.fork()
    right = root.fork()
    ancestor_heads = [layer.head for layer in root._layers]
    errors = []

    def advance(cache, token):
        try:
            infer_gpt_with_persistent_kv_cache(
                model,
                np.array([[token]], dtype=np.int64),
                cache,
            )
        except BaseException as exc:
            errors.append(exc)

    thread_a = threading.Thread(target=advance, args=(left, 3), daemon=True)
    thread_b = threading.Thread(target=advance, args=(right, 4), daemon=True)
    thread_a.start()
    thread_b.start()
    thread_a.join(timeout=10)
    thread_b.join(timeout=10)
    assert not thread_a.is_alive()
    assert not thread_b.is_alive()
    assert errors == []

    for index, ancestor in enumerate(ancestor_heads):
        assert root._layers[index].head is ancestor
        assert left._layers[index].head.prev is ancestor
        assert right._layers[index].head.prev is ancestor
        assert not ancestor.key.flags.writeable
        assert not ancestor.value.flags.writeable
    assert root.length == 2
    assert left.length == right.length == 3
