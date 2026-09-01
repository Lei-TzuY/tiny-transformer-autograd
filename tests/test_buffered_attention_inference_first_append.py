import numpy as np
import pytest

from nn import KVCacheBuffer, MultiHeadAttention, infer_with_kv_buffer


def test_first_output_failure_does_not_initialize_buffer(monkeypatch):
    np.random.seed(1225)
    attention = MultiHeadAttention(8, 2)
    buffer = KVCacheBuffer(4)

    def fail(_):
        raise RuntimeError("injected first output failure")

    monkeypatch.setattr(attention.out_proj, "infer", fail)
    with pytest.raises(RuntimeError, match="injected first output failure"):
        infer_with_kv_buffer(attention, np.random.randn(1, 2, 8), buffer)

    assert not buffer.initialized
    assert buffer.length == 0
    assert buffer.storage_nbytes == 0


def test_first_success_publishes_exactly_one_complete_kv_chunk():
    np.random.seed(1226)
    attention = MultiHeadAttention(8, 2)
    buffer = KVCacheBuffer(4)
    x = np.random.randn(2, 2, 8)

    output, live = infer_with_kv_buffer(attention, x, buffer)

    assert output.shape == (2, 2, 8)
    assert buffer.initialized
    assert buffer.length == 2
    assert live["k"].shape == (2, 2, 2, 4)
    assert live["v"].shape == (2, 2, 2, 4)
    assert np.isfinite(live["k"]).all()
    assert np.isfinite(live["v"]).all()
