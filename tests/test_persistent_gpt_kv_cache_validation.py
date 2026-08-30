import numpy as np
import pytest

import nn.persistent_kv_cache as persistent_module
from nn import (
    GPT,
    PersistentGPTKVCache,
    beam_generate_gpt_with_persistent_kv_cache,
    fork_persistent_gpt_kv_cache,
    infer_gpt_with_persistent_kv_cache,
)


def _model(seed=1505):
    np.random.seed(seed)
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


def test_persistent_cache_public_type_and_model_binding_validation():
    model = _model()
    other = _model(1506)
    cache = PersistentGPTKVCache(model)
    tokens = np.array([[1, 2]], dtype=np.int64)

    with pytest.raises(TypeError, match="model must be a GPT"):
        PersistentGPTKVCache(object())
    with pytest.raises(TypeError, match="model must be a GPT"):
        infer_gpt_with_persistent_kv_cache(object(), tokens, cache)
    with pytest.raises(TypeError, match="PersistentGPTKVCache"):
        infer_gpt_with_persistent_kv_cache(model, tokens, object())
    with pytest.raises(ValueError, match="different GPT"):
        infer_gpt_with_persistent_kv_cache(other, tokens, cache)


def test_persistent_beam_validates_options_and_batch_contract():
    model = _model()
    prompt = np.array([[1, 2]], dtype=np.int64)

    with pytest.raises(TypeError, match="non-negative integer"):
        beam_generate_gpt_with_persistent_kv_cache(model, prompt, True)
    with pytest.raises(ValueError, match="non-negative integer"):
        beam_generate_gpt_with_persistent_kv_cache(model, prompt, -1)
    with pytest.raises(TypeError, match="positive integer"):
        beam_generate_gpt_with_persistent_kv_cache(
            model, prompt, 1, beam_width=np.bool_(True)
        )
    with pytest.raises(ValueError, match="positive integer"):
        beam_generate_gpt_with_persistent_kv_cache(model, prompt, 1, beam_width=0)
    with pytest.raises(TypeError, match="real number"):
        beam_generate_gpt_with_persistent_kv_cache(model, prompt, 1, temperature=True)
    with pytest.raises(ValueError, match="finite"):
        beam_generate_gpt_with_persistent_kv_cache(
            model, prompt, 1, temperature=np.inf
        )

    multi = np.array([[1, 2], [3, 4]], dtype=np.int64)
    with pytest.raises(ValueError, match="batch size 1"):
        beam_generate_gpt_with_persistent_kv_cache(model, multi, 1)


def test_persistent_beam_uses_normal_left_padding_validation():
    model = _model()
    prompt = np.array([[1, 2, 3]], dtype=np.int64)
    bad_mask = np.array([[1, 0, 1]], dtype=np.int64)
    with pytest.raises(ValueError, match="left-padded"):
        beam_generate_gpt_with_persistent_kv_cache(
            model,
            prompt,
            1,
            attention_mask=bad_mask,
        )


def test_incremental_persistent_attention_never_concatenates_historical_kv(monkeypatch):
    model = _model()
    cache = PersistentGPTKVCache(model)
    infer_gpt_with_persistent_kv_cache(model, np.array([[1, 2]], dtype=np.int64), cache)
    infer_gpt_with_persistent_kv_cache(model, np.array([[3]], dtype=np.int64), cache)
    assert cache.segment_count == 2

    original_numpy = persistent_module.np

    class NumpyProxy:
        def __getattr__(self, name):
            return getattr(original_numpy, name)

        def concatenate(self, arrays, *args, **kwargs):
            arrays = list(arrays)
            if arrays:
                first = original_numpy.asarray(arrays[0])
                if original_numpy.issubdtype(first.dtype, original_numpy.floating) and first.ndim >= 3:
                    raise AssertionError("historical floating K/V concatenate reached")
            return original_numpy.concatenate(arrays, *args, **kwargs)

    monkeypatch.setattr(persistent_module, "np", NumpyProxy())
    logits, _ = infer_gpt_with_persistent_kv_cache(
        model,
        np.array([[4]], dtype=np.int64),
        cache,
    )
    assert logits.shape == (1, 1, model.vocab_size)
    assert cache.length == 4
    assert cache.segment_count == 3


def test_beam_path_does_not_materialize_snapshots(monkeypatch):
    model = _model()

    def forbidden_snapshot(self):
        raise AssertionError("persistent beam materialized a cache snapshot")

    monkeypatch.setattr(persistent_module._PersistentLayerCache, "snapshot", forbidden_snapshot)
    sequence, cache = beam_generate_gpt_with_persistent_kv_cache(
        model,
        np.array([[1, 2]], dtype=np.int64),
        3,
        beam_width=2,
    )
    assert sequence.shape == (1, 5)
    assert cache.length == 5


def test_partial_layer_corruption_fails_closed():
    model = _model()
    cache = PersistentGPTKVCache(model)
    infer_gpt_with_persistent_kv_cache(model, np.array([[1, 2]], dtype=np.int64), cache)
    cache._layers[1].head = None
    with pytest.raises(RuntimeError, match="stale length metadata|partially initialized"):
        _ = cache.length


def test_head_segment_writeability_corruption_fails_closed():
    model = _model()
    cache = PersistentGPTKVCache(model)
    infer_gpt_with_persistent_kv_cache(model, np.array([[1, 2]], dtype=np.int64), cache)
    key = cache._layers[0].head.key
    key.flags.writeable = True
    try:
        with pytest.raises(RuntimeError, match="read-only"):
            _ = cache.length
    finally:
        key.flags.writeable = False


def test_persistent_inference_and_beam_preserve_global_rng():
    model = _model()
    cache = PersistentGPTKVCache(model)
    np.random.seed(9911)
    before = np.random.get_state()
    infer_gpt_with_persistent_kv_cache(model, np.array([[1, 2]], dtype=np.int64), cache)
    fork_persistent_gpt_kv_cache(cache)
    beam_generate_gpt_with_persistent_kv_cache(
        model,
        np.array([[1, 2]], dtype=np.int64),
        2,
        beam_width=2,
    )
    after = np.random.get_state()
    assert _rng_state_equal(before, after)


def test_model_version_metadata_corruption_is_rejected():
    model = _model()
    cache = PersistentGPTKVCache(model)
    infer_gpt_with_persistent_kv_cache(model, np.array([[1, 2]], dtype=np.int64), cache)
    cache._model_versions = None
    with pytest.raises(RuntimeError, match="missing model-version"):
        infer_gpt_with_persistent_kv_cache(
            model,
            np.array([[3]], dtype=np.int64),
            cache,
        )
