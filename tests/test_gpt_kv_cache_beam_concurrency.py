import threading
import time

import numpy as np

from nn import GPT, GPTKVCache, fork_gpt_kv_cache, infer_gpt_with_kv_cache


def _model():
    np.random.seed(1401)
    return GPT(
        vocab_size=17,
        context_len=8,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        dropout=0.0,
    )


def test_fork_waits_for_inflight_inference_and_sees_committed_prefix(monkeypatch):
    model = _model()
    cache = GPTKVCache(model)
    entered = threading.Event()
    release = threading.Event()
    fork_done = threading.Event()
    errors = []
    child_holder = []

    original = model.token_emb.infer

    def blocking(ids):
        entered.set()
        if not release.wait(timeout=5):
            raise RuntimeError("test timed out waiting to release inference")
        return original(ids)

    monkeypatch.setattr(model.token_emb, "infer", blocking)

    def run_infer():
        try:
            infer_gpt_with_kv_cache(model, np.array([[1, 2]], dtype=np.int64), cache)
        except BaseException as exc:
            errors.append(exc)

    def run_fork():
        try:
            child_holder.append(fork_gpt_kv_cache(cache))
        except BaseException as exc:
            errors.append(exc)
        finally:
            fork_done.set()

    thread_a = threading.Thread(target=run_infer)
    thread_a.start()
    assert entered.wait(timeout=5)

    thread_b = threading.Thread(target=run_fork)
    thread_b.start()
    time.sleep(0.05)
    assert not fork_done.is_set()

    release.set()
    thread_a.join(timeout=5)
    thread_b.join(timeout=5)
    assert not thread_a.is_alive()
    assert not thread_b.is_alive()
    assert errors == []
    assert cache.length == 2
    assert len(child_holder) == 1
    assert child_holder[0].length == 2


def test_independent_forks_can_advance_concurrently(monkeypatch):
    model = _model()
    parent = GPTKVCache(model)
    infer_gpt_with_kv_cache(model, np.array([[1, 2]], dtype=np.int64), parent)
    left = fork_gpt_kv_cache(parent)
    right = fork_gpt_kv_cache(parent)

    left_entered = threading.Event()
    left_release = threading.Event()
    right_done = threading.Event()
    errors = []
    original = model.token_emb.infer

    def selective_block(ids):
        token = int(np.asarray(ids)[0, 0])
        if token == 3:
            left_entered.set()
            if not left_release.wait(timeout=5):
                raise RuntimeError("test timed out waiting to release left branch")
        return original(ids)

    monkeypatch.setattr(model.token_emb, "infer", selective_block)

    def advance(cache, token, done=None):
        try:
            infer_gpt_with_kv_cache(
                model,
                np.array([[token]], dtype=np.int64),
                cache,
            )
        except BaseException as exc:
            errors.append(exc)
        finally:
            if done is not None:
                done.set()

    thread_left = threading.Thread(target=advance, args=(left, 3))
    thread_left.start()
    assert left_entered.wait(timeout=5)

    thread_right = threading.Thread(target=advance, args=(right, 4, right_done))
    thread_right.start()
    assert right_done.wait(timeout=5)
    assert right.length == 3
    assert left.length == 2

    left_release.set()
    thread_left.join(timeout=5)
    thread_right.join(timeout=5)
    assert errors == []
    assert left.length == right.length == 3
    assert parent.length == 2
