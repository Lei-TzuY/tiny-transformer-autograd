"""GPT inference and generation backed by fixed-capacity per-layer KV buffers.

This module is an opt-in companion to :meth:`GPT.infer` / :meth:`GPT.generate`.
The historical APIs continue to return ordinary per-layer ``{"k", "v"}`` mappings.
``GPTKVCache`` instead owns one :class:`KVCacheBuffer` per transformer block and
``infer_gpt_with_kv_cache`` routes each block through ``infer_with_kv_buffer`` so
incremental decoding copies only newly projected K/V chunks.

A cache is bound to one concrete GPT instance.  Once it contains live tokens, a
snapshot of every model Tensor mutation version is retained; mutating/reloading the
model before the next decode therefore fails before projection instead of silently
combining stale K/V with new weights.  Clearing/truncating to zero removes that version
binding because no live cached values remain.
"""

import threading

import numpy as np

from .buffered_inference import infer_with_kv_buffer
from .kv_cache import KVCacheBuffer
from .transformer import (
    GPT,
    _left_padded_positions,
    _sample,
    _validate_non_negative_int,
    _validate_sampling_options,
    _validate_selection_logits,
)


class GPTKVCache:
    """One fixed-capacity KV buffer per block, bound to a specific GPT instance."""

    def __init__(self, model):
        if not isinstance(model, GPT):
            raise TypeError("model must be a GPT")
        self._model = model
        self._capacity = model.context_len
        self._buffers = [KVCacheBuffer(self._capacity) for _ in model.blocks]
        self._model_versions = None
        self._lock = threading.RLock()

    @property
    def capacity(self):
        return self._capacity

    @property
    def num_layers(self):
        return len(self._buffers)

    @property
    def initialized(self):
        with self._lock:
            initialized, _ = self._state_unlocked()
            return initialized

    @property
    def length(self):
        with self._lock:
            _, length = self._state_unlocked()
            return length

    @property
    def remaining(self):
        return self.capacity - self.length

    @property
    def storage_nbytes(self):
        with self._lock:
            self._state_unlocked()
            return sum(buffer.storage_nbytes for buffer in self._buffers)

    @property
    def live_nbytes(self):
        with self._lock:
            self._state_unlocked()
            return sum(buffer.live_nbytes for buffer in self._buffers)

    def snapshot(self):
        """Return legacy-compatible independent per-layer cache mappings.

        ``None`` represents a cache that has never been initialized.  After a
        storage-retaining ``clear()``, the result is a list of empty cache arrays.
        """
        with self._lock:
            initialized, _ = self._state_unlocked()
            if not initialized:
                return None
            return [buffer.snapshot() for buffer in self._buffers]

    def clear(self):
        """Drop all live tokens while retaining every layer's backing allocation."""
        with self._lock:
            initialized, _ = self._state_unlocked()
            if initialized:
                for buffer in self._buffers:
                    buffer.clear()
            self._model_versions = None
            return self

    def truncate(self, length):
        """Shorten every layer to ``length`` without reallocating backing storage."""
        if isinstance(length, (bool, np.bool_)) or not isinstance(length, (int, np.integer)):
            raise TypeError("length must be a non-negative integer")
        length = int(length)
        if length < 0:
            raise ValueError("length must be non-negative")
        with self._lock:
            initialized, current = self._state_unlocked()
            if not initialized:
                if length == 0:
                    self._model_versions = None
                    return self
                raise ValueError("cannot truncate an uninitialized GPT cache to non-zero length")
            if length > current:
                raise ValueError("cannot extend a GPT cache with truncate()")
            for buffer in self._buffers:
                buffer.truncate(length)
            if length == 0:
                self._model_versions = None
            return self

    def _assert_model(self, model):
        if model is not self._model:
            raise ValueError("GPTKVCache is bound to a different GPT instance")

    def _state_unlocked(self):
        if len(self._buffers) != len(self._model.blocks):
            raise RuntimeError("GPT cache layer count is inconsistent")
        if not self._buffers:
            raise RuntimeError("GPT cache must contain at least one layer buffer")
        initialized = [buffer.initialized for buffer in self._buffers]
        if any(initialized) and not all(initialized):
            raise RuntimeError("GPT cache layers are only partially initialized")
        if not any(initialized):
            if self._model_versions is not None:
                raise RuntimeError("uninitialized GPT cache has stale model-version metadata")
            return False, 0
        lengths = [buffer.length for buffer in self._buffers]
        if any(length != lengths[0] for length in lengths[1:]):
            raise RuntimeError("GPT cache layers have different live lengths")
        if any(buffer.capacity != self._capacity for buffer in self._buffers):
            raise RuntimeError("GPT cache layer capacity is inconsistent")
        if lengths[0] == 0 and self._model_versions is not None:
            raise RuntimeError("empty GPT cache has stale model-version metadata")
        return True, lengths[0]

    def _entry_state_unlocked(self):
        return [
            (buffer.initialized, buffer.length if buffer.initialized else 0)
            for buffer in self._buffers
        ]

    def _restore_unlocked(self, states, versions):
        errors = []
        for index, (was_initialized, length) in enumerate(states):
            buffer = self._buffers[index]
            try:
                if was_initialized:
                    if not buffer.initialized:
                        raise RuntimeError("initialized layer buffer was lost")
                    current = buffer.length
                    if current < length:
                        raise RuntimeError("layer buffer cannot be extended during rollback")
                    if current != length:
                        buffer.truncate(length)
                elif buffer.initialized:
                    # Layer buffers are private to GPTKVCache. Replacing a newly
                    # initialized buffer restores the exact pre-call uninitialized state.
                    self._buffers[index] = KVCacheBuffer(self._capacity)
            except BaseException as exc:  # pragma: no cover - guarded by injected tests
                errors.append(exc)
        self._model_versions = versions
        if errors:
            raise RuntimeError("GPT KV cache rollback failed") from errors[0]

    def __len__(self):
        return self.length

    def __repr__(self):
        with self._lock:
            initialized, length = self._state_unlocked()
            state = "initialized" if initialized else "uninitialized"
            return (
                f"GPTKVCache(layers={self.num_layers}, capacity={self.capacity}, "
                f"length={length}, {state})"
            )


def infer_gpt_with_kv_cache(
    model,
    idx,
    cache,
    *,
    attention_mask=None,
    position_ids=None,
):
    """Run GPT NumPy inference while appending each layer into ``GPTKVCache``.

    The complete multi-layer operation is transactional with respect to cache-visible
    state. If any later block, final norm, LM head, or postcondition fails, all layers
    return to their entry initialization/length state.
    """
    if not isinstance(model, GPT):
        raise TypeError("model must be a GPT")
    if not isinstance(cache, GPTKVCache):
        raise TypeError("cache must be a GPTKVCache")

    with cache._lock:
        cache._assert_model(model)
        _, past_len = cache._state_unlocked()
        idx = model._validate_token_batch(idx, max_time=model.context_len)
        batch, time = idx.shape
        if past_len + time > model.context_len:
            raise ValueError("inference input and cache exceed context_len")
        if time > cache.remaining:
            raise OverflowError("GPT KV cache capacity exceeded")

        before_versions = _model_versions(model)
        if past_len > 0:
            if cache._model_versions is None:
                raise RuntimeError("non-empty GPT cache is missing model-version metadata")
            if before_versions != cache._model_versions:
                raise RuntimeError("model tensors changed while GPT KV cache was live")

        key_bias = None
        if attention_mask is not None:
            keep = _validate_keep_mask_through_model(
                attention_mask,
                (batch, past_len + time),
            )
            key_bias = np.where(keep, 0.0, -np.inf)[:, np.newaxis, np.newaxis, :]

        if position_ids is None:
            positions = np.arange(past_len, past_len + time, dtype=np.int64)
            rope_positions = None
        else:
            positions = model._validate_position_ids(position_ids, (batch, time))
            rope_positions = positions[:, np.newaxis, :]

        entry_states = cache._entry_state_unlocked()
        entry_versions = cache._model_versions
        try:
            x = model.token_emb.infer(idx)
            if model.pos_emb is not None:
                x = x + model.pos_emb.infer(positions)

            for block, layer_buffer in zip(model.blocks, cache._buffers):
                normalized = block.ln1.infer(x)
                attended, _ = infer_with_kv_buffer(
                    block.attn,
                    normalized,
                    layer_buffer,
                    key_bias=key_bias,
                    positions=rope_positions,
                )
                x = x + attended
                x = x + block.ff.infer(block.ln2.infer(x))

            logits = model.head.infer(model.ln_f.infer(x))
            after_versions = _model_versions(model)
            if after_versions != before_versions:
                raise RuntimeError("model tensors changed during GPT KV-cache inference")

            initialized, final_len = cache._state_unlocked()
            if not initialized or final_len != past_len + time:
                raise RuntimeError("GPT KV cache postcondition failed")
            cache._model_versions = after_versions if final_len > 0 else None
            return logits, cache
        except BaseException:
            try:
                cache._restore_unlocked(entry_states, entry_versions)
            except BaseException as rollback_exc:
                raise RuntimeError("GPT KV cache rollback failed") from rollback_exc
            raise


def generate_gpt_with_kv_cache(
    model,
    idx,
    max_new_tokens,
    *,
    temperature=1.0,
    top_k=None,
    top_p=None,
    strategy="sample",
    attention_mask=None,
    cache=None,
):
    """Generate with sample/greedy decoding using a reusable ``GPTKVCache``.

    Returns ``(tokens, cache)``. The cache must be empty at entry; when a context
    window saturates it is cleared and refilled from the cropped window, retaining
    the already allocated layer storage just like the historical cache path resets
    its dict cache at a sliding-window boundary.
    """
    if not isinstance(model, GPT):
        raise TypeError("model must be a GPT")
    max_new_tokens = _validate_non_negative_int(max_new_tokens, "max_new_tokens")
    if not isinstance(strategy, str):
        raise TypeError("strategy must be a string")
    if strategy not in {"sample", "greedy"}:
        raise ValueError("strategy must be 'sample' or 'greedy'")
    temperature, top_k, top_p = _validate_sampling_options(temperature, top_k, top_p)

    if cache is None:
        cache = GPTKVCache(model)
    elif not isinstance(cache, GPTKVCache):
        raise TypeError("cache must be a GPTKVCache or None")
    cache._assert_model(model)
    if cache.length != 0:
        raise ValueError("generation cache must be empty at entry")

    idx = np.array(model._validate_token_batch(idx), dtype=np.int64, copy=True)
    mask = positions = None
    if attention_mask is not None:
        mask = model._validate_generation_mask(attention_mask, idx.shape)
        positions = np.zeros(idx.shape, dtype=np.int64)

    has_cache = False
    for _ in range(max_new_tokens):
        if has_cache and cache.length > 0:
            step_mask = step_positions = None
            if mask is not None:
                cached = cache.length
                step_mask = mask[:, -(cached + 1):]
                step_positions = positions[:, -1:]
            logits, _ = infer_gpt_with_kv_cache(
                model,
                idx[:, -1:],
                cache,
                attention_mask=step_mask,
                position_ids=step_positions,
            )
        else:
            window = idx[:, -model.context_len:]
            window_mask = window_positions = None
            if mask is not None:
                width = window.shape[1]
                window_mask = mask[:, -width:]
                positions[:, -width:] = _left_padded_positions(window_mask)
                window_positions = positions[:, -width:]
            logits, _ = infer_gpt_with_kv_cache(
                model,
                window,
                cache,
                attention_mask=window_mask,
                position_ids=window_positions,
            )
            has_cache = True

        logits_last = _validate_selection_logits(logits[:, -1, :], "generation logits")
        if strategy == "greedy":
            next_token = np.argmax(logits_last, axis=-1)
        else:
            next_token = np.array(
                [
                    _sample(logit, temperature=temperature, top_k=top_k, top_p=top_p)
                    for logit in logits_last
                ]
            )
        idx = np.concatenate([idx, next_token[:, None]], axis=1)
        if mask is not None:
            mask = np.concatenate(
                [mask, np.ones((mask.shape[0], 1), dtype=bool)], axis=1
            )
            positions = np.concatenate(
                [positions, positions[:, -1:] + 1], axis=1
            )

        if cache.length >= model.context_len:
            cache.clear()
            has_cache = False

    return idx, cache


def _model_versions(model):
    result = []
    seen = set()
    for name, tensor in model.named_tensors():
        identity = id(tensor)
        if identity in seen:
            continue
        seen.add(identity)
        version = getattr(tensor, "_version", None)
        if isinstance(version, bool) or not isinstance(version, int) or version < 0:
            raise RuntimeError(f"model tensor {name!r} has invalid mutation-version metadata")
        result.append((name, identity, version))
    return tuple(result)


def _validate_keep_mask_through_model(attention_mask, expected_shape):
    # Keep the public wording/semantics aligned with GPT.infer without copying its
    # module-private validator into a second implementation.
    from .transformer import _validate_keep_mask

    return _validate_keep_mask(
        attention_mask,
        expected_shape,
        "covering the cached and current keys",
    )
