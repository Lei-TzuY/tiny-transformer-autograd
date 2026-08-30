import threading

import numpy as np

from nn import KVCacheBuffer, MultiHeadAttention, infer_with_kv_buffer


def test_same_buffer_inference_is_serialized_across_complete_calls(monkeypatch):
    np.random.seed(1220)
    attention = MultiHeadAttention(8, 2)
    reference = MultiHeadAttention(8, 2)
    reference.load_state_dict(attention.state_dict())

    token_a = np.random.randn(1, 1, 8)
    token_b = np.random.randn(1, 1, 8)

    reference_buffer = KVCacheBuffer(4)
    expected_a, _ = infer_with_kv_buffer(reference, token_a, reference_buffer)
    expected_b, expected_cache = infer_with_kv_buffer(
        reference,
        token_b,
        reference_buffer,
    )

    original = attention.W_q.infer
    first_entered = threading.Event()
    release_first = threading.Event()
    call_order = []
    order_lock = threading.Lock()

    def blocking_infer(x):
        name = threading.current_thread().name
        with order_lock:
            call_order.append(name)
        if name == "buffer-a":
            first_entered.set()
            assert release_first.wait(timeout=5)
        return original(x)

    monkeypatch.setattr(attention.W_q, "infer", blocking_infer)

    buffer = KVCacheBuffer(4)
    results = {}
    failures = []

    def worker(name, token):
        try:
            results[name] = infer_with_kv_buffer(attention, token, buffer)
        except BaseException as exc:  # pragma: no cover - surfaced below
            failures.append(exc)

    thread_a = threading.Thread(
        target=worker,
        args=("a", token_a),
        name="buffer-a",
    )
    thread_b = threading.Thread(
        target=worker,
        args=("b", token_b),
        name="buffer-b",
    )

    thread_a.start()
    assert first_entered.wait(timeout=5)
    thread_b.start()

    # B is already running, but the complete-call buffer lock prevents it from even
    # entering Q projection while A owns the temporary append/score transaction.
    thread_b.join(timeout=0.05)
    with order_lock:
        assert call_order == ["buffer-a"]
    assert thread_b.is_alive()

    release_first.set()
    thread_a.join(timeout=5)
    thread_b.join(timeout=5)

    assert not thread_a.is_alive()
    assert not thread_b.is_alive()
    assert failures == []
    with order_lock:
        assert call_order == ["buffer-a", "buffer-b"]

    actual_a, _ = results["a"]
    actual_b, live = results["b"]
    np.testing.assert_allclose(actual_a, expected_a)
    np.testing.assert_allclose(actual_b, expected_b)
    np.testing.assert_allclose(live["k"], expected_cache["k"])
    np.testing.assert_allclose(live["v"], expected_cache["v"])
    assert buffer.length == 2


def test_different_buffers_do_not_share_a_global_inference_lock(monkeypatch):
    np.random.seed(1221)
    attention = MultiHeadAttention(8, 2)
    token = np.random.randn(1, 1, 8)
    buffer_a = KVCacheBuffer(2)
    buffer_b = KVCacheBuffer(2)

    original = attention.W_q.infer
    entered = threading.Event()
    release = threading.Event()
    second_entered = threading.Event()

    def blocking_infer(x):
        if threading.current_thread().name == "independent-a":
            entered.set()
            assert release.wait(timeout=5)
        else:
            second_entered.set()
        return original(x)

    monkeypatch.setattr(attention.W_q, "infer", blocking_infer)
    failures = []

    def worker(buffer):
        try:
            infer_with_kv_buffer(attention, token, buffer)
        except BaseException as exc:  # pragma: no cover - surfaced below
            failures.append(exc)

    first = threading.Thread(target=worker, args=(buffer_a,), name="independent-a")
    second = threading.Thread(target=worker, args=(buffer_b,), name="independent-b")

    first.start()
    assert entered.wait(timeout=5)
    second.start()
    assert second_entered.wait(timeout=5)
    second.join(timeout=5)
    assert not second.is_alive()

    release.set()
    first.join(timeout=5)
    assert not first.is_alive()
    assert failures == []
    assert buffer_a.length == 1
    assert buffer_b.length == 1
