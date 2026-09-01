import threading

import numpy as np

from nn import GPT, GPTKVCache, infer_gpt_with_kv_cache


def _model():
    np.random.seed(5501)
    return GPT(
        vocab_size=17,
        context_len=8,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        dropout=0.0,
    )


def test_same_gpt_cache_serializes_complete_inference_calls():
    model = _model()
    cache = GPTKVCache(model)
    first = np.array([[1]], dtype=np.int64)
    second = np.array([[2]], dtype=np.int64)

    entered = threading.Event()
    release = threading.Event()
    second_projected = threading.Event()
    original = model.token_emb.infer
    call_count = {"value": 0}
    guard = threading.Lock()

    def blocked(idx):
        with guard:
            call_count["value"] += 1
            number = call_count["value"]
        if number == 1:
            entered.set()
            assert release.wait(timeout=5.0)
        elif number == 2:
            second_projected.set()
        return original(idx)

    model.token_emb.infer = blocked
    results = {}
    errors = []

    def worker(name, tokens):
        try:
            results[name], _ = infer_gpt_with_kv_cache(model, tokens, cache)
        except BaseException as exc:
            errors.append(exc)

    try:
        thread_a = threading.Thread(target=worker, args=("a", first))
        thread_b = threading.Thread(target=worker, args=("b", second))
        thread_a.start()
        assert entered.wait(timeout=5.0)
        thread_b.start()
        assert not second_projected.wait(timeout=0.2)
        release.set()
        thread_a.join(timeout=5.0)
        thread_b.join(timeout=5.0)
    finally:
        model.token_emb.infer = original
        release.set()

    assert not thread_a.is_alive()
    assert not thread_b.is_alive()
    assert errors == []
    assert second_projected.is_set()
    assert cache.length == 2

    reference = _model()
    # Copy exact parameters rather than relying on construction RNG in case the
    # fixture evolves later.
    reference.load_state_dict(model.state_dict())
    expected_a, legacy = reference.infer(first)
    expected_b, legacy = reference.infer(second, legacy)
    np.testing.assert_allclose(results["a"], expected_a, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(results["b"], expected_b, rtol=0.0, atol=0.0)
    snapshot = cache.snapshot()
    for actual, expected in zip(snapshot, legacy):
        np.testing.assert_array_equal(actual["k"], expected["k"])
        np.testing.assert_array_equal(actual["v"], expected["v"])


def test_different_gpt_caches_do_not_share_a_global_inference_lock():
    model = _model()
    cache_a = GPTKVCache(model)
    cache_b = GPTKVCache(model)
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()
    original = model.token_emb.infer
    lock = threading.Lock()
    calls = {"value": 0}

    def observed(idx):
        with lock:
            calls["value"] += 1
            number = calls["value"]
        if number == 1:
            first_entered.set()
            assert release_first.wait(timeout=5.0)
        else:
            second_entered.set()
        return original(idx)

    model.token_emb.infer = observed
    errors = []

    def run(cache, token):
        try:
            infer_gpt_with_kv_cache(model, np.array([[token]], dtype=np.int64), cache)
        except BaseException as exc:
            errors.append(exc)

    try:
        a = threading.Thread(target=run, args=(cache_a, 1))
        b = threading.Thread(target=run, args=(cache_b, 2))
        a.start()
        assert first_entered.wait(timeout=5.0)
        b.start()
        assert second_entered.wait(timeout=2.0)
        release_first.set()
        a.join(timeout=5.0)
        b.join(timeout=5.0)
    finally:
        model.token_emb.infer = original
        release_first.set()

    assert not a.is_alive()
    assert not b.is_alive()
    assert errors == []
    assert cache_a.length == 1
    assert cache_b.length == 1
