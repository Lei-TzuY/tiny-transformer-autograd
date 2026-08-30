"""Persistent shared-prefix GPT KV caches for branching inference.

``GPTKVCache`` provides excellent single-path decoding by appending into one fixed
capacity allocation, while ``fork_gpt_kv_cache()`` deliberately makes an eager copy
so every mutable branch owns independent storage.  Beam search can create many sibling
branches, however, and copying the complete historical K/V prefix for every selected
child makes branching O(prefix_bytes).

This module provides a complementary persistent representation.  Every layer stores an
immutable linked chain of K/V segments.  Forking a cache copies only small per-layer
metadata and shares the immutable segment heads; appending a child token allocates only
that new K/V segment.  Attention scores are computed directly against each segment and
placed into one ordinary score tensor before the existing softmax is applied.  Historical
K/V arrays are therefore never concatenated or copied during normal fork/decode/beam
operations.  ``snapshot()`` is the explicit ownership boundary that materializes legacy
contiguous per-layer K/V arrays.

The representation is optimized for branching rather than single-path decode.  Each
appended segment is a separate immutable allocation and inference walks the live segment
chain, so callers that do not need branching should continue to prefer ``GPTKVCache``.
"""

import threading

import numpy as np

from .attention import (
    MultiHeadAttention,
    _causal_mask,
    _prepare_mask,
    _scaled_dot_product_scores_np,
    _softmax,
)
from .gpt_kv_cache import _model_versions, _validate_keep_mask_through_model
from .transformer import (
    GPT,
    _left_padded_positions,
    _log_softmax,
    _temperature_scale_logits,
    _validate_non_negative_int,
    _validate_positive_finite_real,
    _validate_positive_int,
)


class _KVSegment:
    __slots__ = ("key", "value", "prev", "length", "total_length")

    def __init__(self, key, value, prev, total_length):
        self.key = key
        self.value = value
        self.prev = prev
        self.length = int(key.shape[-2])
        self.total_length = int(total_length)


class _PersistentLayerCache:
    __slots__ = (
        "capacity",
        "head",
        "length",
        "segment_count",
        "key_layout",
        "value_layout",
        "key_dtype",
        "value_dtype",
    )

    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.head = None
        self.length = 0
        self.segment_count = 0
        self.key_layout = None
        self.value_layout = None
        self.key_dtype = None
        self.value_dtype = None

    def entry_state(self):
        return (
            self.head,
            self.length,
            self.segment_count,
            self.key_layout,
            self.value_layout,
            self.key_dtype,
            self.value_dtype,
        )

    def restore_state(self, state):
        (
            self.head,
            self.length,
            self.segment_count,
            self.key_layout,
            self.value_layout,
            self.key_dtype,
            self.value_dtype,
        ) = state

    def clear(self):
        self.head = None
        self.length = 0
        self.segment_count = 0
        self.key_layout = None
        self.value_layout = None
        self.key_dtype = None
        self.value_dtype = None

    def append(self, key, value):
        key, value = self._snapshot_candidate(key, value)
        chunk_len = key.shape[-2]
        end = self.length + chunk_len
        if end > self.capacity:
            raise OverflowError(
                f"persistent cache capacity {self.capacity} exceeded by append ending at {end}"
            )

        key_layout = _layout_without_time(key.shape)
        value_layout = _layout_without_time(value.shape)
        if self.head is not None:
            self._validate_internal_state()
            if key_layout != self.key_layout:
                raise ValueError("key cache layout does not match the persistent layer")
            if value_layout != self.value_layout:
                raise ValueError("value cache layout does not match the persistent layer")
            if key.dtype != self.key_dtype:
                raise TypeError("key dtype does not match the persistent layer")
            if value.dtype != self.value_dtype:
                raise TypeError("value dtype does not match the persistent layer")

        key.flags.writeable = False
        value.flags.writeable = False
        node = _KVSegment(key, value, self.head, end)

        if self.head is None:
            self.key_layout = key_layout
            self.value_layout = value_layout
            self.key_dtype = key.dtype
            self.value_dtype = value.dtype
        self.head = node
        self.length = end
        self.segment_count += 1
        return node

    def segments_oldest(self):
        self._validate_internal_state()
        nodes = []
        node = self.head
        while node is not None:
            nodes.append(node)
            node = node.prev
        nodes.reverse()
        return nodes

    def snapshot(self):
        nodes = self.segments_oldest()
        if not nodes:
            raise RuntimeError("persistent cache layer is empty")
        if len(nodes) == 1:
            return {
                "k": np.array(nodes[0].key, copy=True, order="C", subok=False),
                "v": np.array(nodes[0].value, copy=True, order="C", subok=False),
            }
        return {
            "k": np.concatenate([node.key for node in nodes], axis=-2),
            "v": np.concatenate([node.value for node in nodes], axis=-2),
        }

    def live_nbytes(self):
        total = 0
        node = self.head
        seen = 0
        while node is not None:
            total += int(node.key.nbytes + node.value.nbytes)
            seen += 1
            if seen > self.segment_count:
                raise RuntimeError("persistent cache segment chain contains a cycle")
            node = node.prev
        if seen != self.segment_count:
            raise RuntimeError("persistent cache segment count metadata is inconsistent")
        return total

    def _snapshot_candidate(self, key, value):
        key = _validate_kv_array("key", key)
        value = _validate_kv_array("value", value)
        if key.ndim != value.ndim:
            raise ValueError("key and value must have the same rank")
        if key.shape[:-2] != value.shape[:-2]:
            raise ValueError("key and value leading cache dimensions must match")
        if key.shape[-2] != value.shape[-2]:
            raise ValueError("key and value must contain the same number of time steps")
        if key.shape[-2] == 0:
            raise ValueError("persistent cache append must contain at least one time step")
        return (
            np.array(key, copy=True, order="C", subok=False),
            np.array(value, copy=True, order="C", subok=False),
        )

    def _validate_internal_state(self):
        if not isinstance(self.capacity, int) or isinstance(self.capacity, bool):
            raise RuntimeError("persistent cache capacity metadata is invalid")
        if self.capacity <= 0:
            raise RuntimeError("persistent cache capacity metadata is invalid")
        if not isinstance(self.length, int) or isinstance(self.length, bool):
            raise RuntimeError("persistent cache length metadata is invalid")
        if self.length < 0 or self.length > self.capacity:
            raise RuntimeError("persistent cache length metadata is outside capacity")
        if not isinstance(self.segment_count, int) or isinstance(self.segment_count, bool):
            raise RuntimeError("persistent cache segment count metadata is invalid")
        if self.segment_count < 0:
            raise RuntimeError("persistent cache segment count metadata is invalid")

        if self.head is None:
            if self.length != 0 or self.segment_count != 0:
                raise RuntimeError("empty persistent cache layer has stale length metadata")
            if any(
                item is not None
                for item in (
                    self.key_layout,
                    self.value_layout,
                    self.key_dtype,
                    self.value_dtype,
                )
            ):
                raise RuntimeError("empty persistent cache layer has stale layout metadata")
            return

        if self.length == 0 or self.segment_count == 0:
            raise RuntimeError("non-empty persistent cache layer has empty metadata")
        if self.key_layout is None or self.value_layout is None:
            raise RuntimeError("persistent cache layer is missing layout metadata")
        if self.key_dtype is None or self.value_dtype is None:
            raise RuntimeError("persistent cache layer is missing dtype metadata")

        node = self.head
        seen = 0
        expected_total = self.length
        while node is not None:
            if not isinstance(node, _KVSegment):
                raise RuntimeError("persistent cache chain contains an invalid segment")
            key = node.key
            value = node.value
            if type(key) is not np.ndarray or type(value) is not np.ndarray:
                raise RuntimeError("persistent cache segments must use ordinary NumPy arrays")
            if key.flags.writeable or value.flags.writeable:
                raise RuntimeError("persistent cache segment storage must remain read-only")
            if key.ndim < 2 or value.ndim < 2 or key.ndim != value.ndim:
                raise RuntimeError("persistent cache segment rank metadata is inconsistent")
            if key.shape[:-2] != value.shape[:-2] or key.shape[-2] != value.shape[-2]:
                raise RuntimeError("persistent cache segment K/V shapes are inconsistent")
            if _layout_without_time(key.shape) != self.key_layout:
                raise RuntimeError("persistent key segment layout metadata is inconsistent")
            if _layout_without_time(value.shape) != self.value_layout:
                raise RuntimeError("persistent value segment layout metadata is inconsistent")
            if key.dtype != self.key_dtype or value.dtype != self.value_dtype:
                raise RuntimeError("persistent cache segment dtype metadata is inconsistent")
            if node.length != key.shape[-2] or node.length <= 0:
                raise RuntimeError("persistent cache segment length metadata is inconsistent")
            if node.total_length != expected_total:
                raise RuntimeError("persistent cache segment total-length metadata is inconsistent")
            if not np.isfinite(key).all() or not np.isfinite(value).all():
                raise RuntimeError("persistent cache segment contains non-finite values")

            expected_total -= node.length
            seen += 1
            if seen > self.segment_count:
                raise RuntimeError("persistent cache segment chain contains a cycle")
            node = node.prev

        if seen != self.segment_count or expected_total != 0:
            raise RuntimeError("persistent cache segment chain metadata is inconsistent")


class PersistentGPTKVCache:
    """GPT KV cache whose immutable layer segments can be shared across branches."""

    def __init__(self, model):
        if not isinstance(model, GPT):
            raise TypeError("model must be a GPT")
        self._model = model
        self._capacity = model.context_len
        self._layers = [_PersistentLayerCache(self._capacity) for _ in model.blocks]
        self._model_versions = None
        self._lock = threading.RLock()

    @property
    def capacity(self):
        return self._capacity

    @property
    def num_layers(self):
        return len(self._layers)

    @property
    def initialized(self):
        return self.length > 0

    @property
    def length(self):
        with self._lock:
            _, length, _ = self._state_unlocked()
            return length

    @property
    def remaining(self):
        return self.capacity - self.length

    @property
    def segment_count(self):
        """Number of immutable K/V segments per layer in this logical branch."""
        with self._lock:
            _, _, count = self._state_unlocked()
            return count

    @property
    def live_nbytes(self):
        """Reachable live K/V bytes for this branch, including shared ancestors."""
        with self._lock:
            self._state_unlocked()
            return sum(layer.live_nbytes() for layer in self._layers)

    @property
    def storage_nbytes(self):
        # Persistent segments have no spare capacity.  For one logical branch, every
        # reachable byte is live; across siblings, shared ancestors are intentionally
        # counted once per branch by this per-cache property.
        return self.live_nbytes

    def snapshot(self):
        """Materialize independent legacy-compatible contiguous per-layer K/V arrays."""
        with self._lock:
            initialized, _, _ = self._state_unlocked()
            if not initialized:
                return None
            return [layer.snapshot() for layer in self._layers]

    def clear(self):
        """Drop this branch's complete segment chain without affecting sibling forks."""
        with self._lock:
            self._state_unlocked()
            for layer in self._layers:
                layer.clear()
            self._model_versions = None
            return self

    def fork(self):
        return fork_persistent_gpt_kv_cache(self)

    def _assert_model(self, model):
        if model is not self._model:
            raise ValueError("PersistentGPTKVCache is bound to a different GPT instance")

    def _state_unlocked(self):
        if len(self._layers) != len(self._model.blocks):
            raise RuntimeError("persistent GPT cache layer count is inconsistent")
        if not self._layers:
            raise RuntimeError("persistent GPT cache must contain at least one layer")

        lengths = []
        counts = []
        initialized = []
        for layer in self._layers:
            if not isinstance(layer, _PersistentLayerCache):
                raise RuntimeError("persistent GPT cache contains an invalid layer cache")
            if layer.capacity != self._capacity:
                raise RuntimeError("persistent GPT cache layer capacity is inconsistent")
            layer._validate_internal_state()
            lengths.append(layer.length)
            counts.append(layer.segment_count)
            initialized.append(layer.head is not None)

        if any(initialized) and not all(initialized):
            raise RuntimeError("persistent GPT cache layers are only partially initialized")
        if any(length != lengths[0] for length in lengths[1:]):
            raise RuntimeError("persistent GPT cache layers have different live lengths")
        if any(count != counts[0] for count in counts[1:]):
            raise RuntimeError("persistent GPT cache layers have different segment counts")

        if not any(initialized):
            if lengths[0] != 0 or counts[0] != 0:
                raise RuntimeError("empty persistent GPT cache has stale layer metadata")
            if self._model_versions is not None:
                raise RuntimeError("empty persistent GPT cache has stale model-version metadata")
            return False, 0, 0

        if lengths[0] <= 0 or counts[0] <= 0:
            raise RuntimeError("non-empty persistent GPT cache has invalid metadata")
        return True, lengths[0], counts[0]

    def _entry_state_unlocked(self):
        return [layer.entry_state() for layer in self._layers]

    def _restore_unlocked(self, states, versions):
        if len(states) != len(self._layers):
            raise RuntimeError("persistent GPT cache rollback state count is inconsistent")
        for layer, state in zip(self._layers, states):
            layer.restore_state(state)
        self._model_versions = versions
        self._state_unlocked()

    def __len__(self):
        return self.length

    def __repr__(self):
        with self._lock:
            initialized, length, segments = self._state_unlocked()
            state = "initialized" if initialized else "empty"
            return (
                f"PersistentGPTKVCache(layers={self.num_layers}, capacity={self.capacity}, "
                f"length={length}, segments={segments}, {state})"
            )


def fork_persistent_gpt_kv_cache(cache):
    """Fork ``cache`` without copying historical K/V arrays.

    The returned cache owns new mutable layer metadata and a new outer lock, but each
    layer initially points at the same immutable segment head as the source.  Future
    appends add new child-owned nodes and never modify shared ancestors.
    """
    if not isinstance(cache, PersistentGPTKVCache):
        raise TypeError("cache must be a PersistentGPTKVCache")

    with cache._lock:
        _, length, _ = cache._state_unlocked()
        if length > 0:
            if cache._model_versions is None:
                raise RuntimeError("non-empty persistent GPT cache is missing model-version metadata")
            current_versions = _model_versions(cache._model)
            if current_versions != cache._model_versions:
                raise RuntimeError("model tensors changed while persistent GPT KV cache was live")

        states = cache._entry_state_unlocked()
        versions = cache._model_versions
        model = cache._model

    child = PersistentGPTKVCache(model)
    with child._lock:
        for layer, state in zip(child._layers, states):
            layer.restore_state(state)
        child._model_versions = versions
        child._state_unlocked()
    return child


def infer_gpt_with_persistent_kv_cache(
    model,
    idx,
    cache,
    *,
    attention_mask=None,
    position_ids=None,
):
    """Run GPT inference while appending immutable shared-prefix K/V segments."""
    if not isinstance(model, GPT):
        raise TypeError("model must be a GPT")
    if not isinstance(cache, PersistentGPTKVCache):
        raise TypeError("cache must be a PersistentGPTKVCache")

    with cache._lock:
        cache._assert_model(model)
        _, past_len, _ = cache._state_unlocked()
        idx = model._validate_token_batch(idx, max_time=model.context_len)
        batch, time = idx.shape
        if past_len + time > model.context_len:
            raise ValueError("inference input and cache exceed context_len")
        if time > cache.remaining:
            raise OverflowError("persistent GPT KV cache capacity exceeded")

        before_versions = _model_versions(model)
        if past_len > 0:
            if cache._model_versions is None:
                raise RuntimeError("non-empty persistent GPT cache is missing model-version metadata")
            if before_versions != cache._model_versions:
                raise RuntimeError("model tensors changed while persistent GPT KV cache was live")

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

            for block, layer in zip(model.blocks, cache._layers):
                normalized = block.ln1.infer(x)
                attended = _infer_segmented_attention(
                    block.attn,
                    normalized,
                    layer,
                    key_bias=key_bias,
                    positions=rope_positions,
                )
                x = x + attended
                x = x + block.ff.infer(block.ln2.infer(x))

            logits = model.head.infer(model.ln_f.infer(x))
            after_versions = _model_versions(model)
            if after_versions != before_versions:
                raise RuntimeError("model tensors changed during persistent GPT KV-cache inference")

            initialized, final_len, _ = cache._state_unlocked()
            if not initialized or final_len != past_len + time:
                raise RuntimeError("persistent GPT KV cache postcondition failed")
            cache._model_versions = after_versions
            return logits, cache
        except BaseException:
            try:
                cache._restore_unlocked(entry_states, entry_versions)
            except BaseException as rollback_exc:
                raise RuntimeError("persistent GPT KV cache rollback failed") from rollback_exc
            raise


def beam_generate_gpt_with_persistent_kv_cache(
    model,
    idx,
    max_new_tokens,
    *,
    beam_width=3,
    temperature=1.0,
    attention_mask=None,
):
    """Batch-size-one beam search whose sibling branches share immutable K/V prefixes."""
    if not isinstance(model, GPT):
        raise TypeError("model must be a GPT")
    max_new_tokens = _validate_non_negative_int(max_new_tokens, "max_new_tokens")
    beam_width = _validate_positive_int(beam_width, "beam_width")
    temperature = _validate_positive_finite_real(temperature, "temperature")

    idx = np.array(model._validate_token_batch(idx), dtype=np.int64, copy=True)
    if idx.shape[0] != 1:
        raise ValueError("persistent buffered beam search currently supports batch size 1")

    mask = None
    if attention_mask is not None:
        mask = model._validate_generation_mask(attention_mask, idx.shape).copy()

    root = PersistentGPTKVCache(model)
    logits = _prefill(model, idx, mask, root)
    beams = [(idx, 0.0, logits, root, mask)]

    for _ in range(max_new_tokens):
        candidates = []
        for sequence, score, beam_logits, parent_cache, beam_mask in beams:
            scaled = _temperature_scale_logits(beam_logits[0, -1], temperature)
            log_probs = _log_softmax(scaled)
            best = np.argsort(log_probs)[-beam_width:]
            for token in best:
                extended = np.concatenate([sequence, [[token]]], axis=1)
                extended_mask = None
                if beam_mask is not None:
                    extended_mask = np.concatenate(
                        [beam_mask, np.ones((1, 1), dtype=bool)],
                        axis=1,
                    )
                candidates.append(
                    (
                        extended,
                        score + float(log_probs[token]),
                        parent_cache,
                        extended_mask,
                    )
                )

        selected = sorted(candidates, key=lambda item: item[1], reverse=True)[:beam_width]
        if not selected:
            raise RuntimeError("persistent beam search produced no candidates")

        advanced = []
        for sequence, score, parent_cache, beam_mask in selected:
            child = fork_persistent_gpt_kv_cache(parent_cache)
            child_logits = _advance_child(model, sequence, beam_mask, child)
            advanced.append((sequence, score, child_logits, child, beam_mask))
        beams = advanced

    best_sequence, _, _, best_cache, _ = beams[0]
    return best_sequence, best_cache


def _infer_segmented_attention(attention, x, layer, *, key_bias, positions):
    if not isinstance(attention, MultiHeadAttention):
        raise TypeError("persistent GPT KV cache requires MultiHeadAttention blocks")

    layer._validate_internal_state()
    B, T, C = x.shape
    H, d_k = attention.num_heads, attention.d_k
    if T > layer.capacity - layer.length:
        raise OverflowError("persistent attention KV cache capacity exceeded")

    Q = attention.W_q.infer(x).reshape(B, T, H, d_k).transpose(0, 2, 1, 3)
    K_new = attention.W_k.infer(x).reshape(B, T, H, d_k).transpose(0, 2, 1, 3)
    V_new = attention.W_v.infer(x).reshape(B, T, H, d_k).transpose(0, 2, 1, 3)

    past_len = layer.length
    if attention.rope is not None:
        if positions is None:
            Q = attention.rope.rotate_np(Q, offset=past_len)
            K_new = attention.rope.rotate_np(K_new, offset=past_len)
        else:
            Q = attention.rope.rotate_np(Q, positions=positions)
            K_new = attention.rope.rotate_np(K_new, positions=positions)

    historical = layer.segments_oldest() if layer.head is not None else []
    parts = [(node.key, node.value, node.length) for node in historical]
    parts.append((K_new, V_new, T))
    total_len = past_len + T

    scores = None
    offset = 0
    for key, _, part_len in parts:
        part_scores = _scaled_dot_product_scores_np(
            Q,
            key.transpose(0, 1, 3, 2),
            attention.scale,
        )
        if scores is None:
            scores = np.empty(
                part_scores.shape[:-1] + (total_len,),
                dtype=part_scores.dtype,
            )
        scores[..., offset : offset + part_len] = part_scores
        offset += part_len
    if scores is None or offset != total_len:
        raise RuntimeError("persistent attention score assembly postcondition failed")

    scores += _causal_mask(T, total_len, past_len)
    if key_bias is not None:
        key_bias = _prepare_mask(
            key_bias,
            scores.shape,
            as_tensor=False,
            name="attention key_bias",
        )
        scores = scores + key_bias
    weights = _softmax(scores)

    attended = None
    offset = 0
    for _, value, part_len in parts:
        contribution = weights[..., offset : offset + part_len] @ value
        attended = contribution if attended is None else attended + contribution
        offset += part_len
    if attended is None or offset != total_len:
        raise RuntimeError("persistent attention value assembly postcondition failed")

    attended = attended.transpose(0, 2, 1, 3).reshape(B, T, C)
    output = attention.out_proj.infer(attended)

    # Publish the immutable node only after projection, attention, softmax, value
    # reduction and output projection all succeeded.  A later GPT-layer failure can
    # roll back by restoring only the layer head/metadata tuple.
    layer.append(K_new, V_new)
    return output


def _prefill(model, sequence, mask, cache):
    window = sequence[:, -model.context_len :]
    window_mask = positions = None
    if mask is not None:
        width = window.shape[1]
        window_mask = mask[:, -width:]
        positions = _left_padded_positions(window_mask)
    logits, _ = infer_gpt_with_persistent_kv_cache(
        model,
        window,
        cache,
        attention_mask=window_mask,
        position_ids=positions,
    )
    return logits


def _advance_child(model, sequence, mask, cache):
    if cache.length < model.context_len:
        step_mask = step_positions = None
        if mask is not None:
            cached = cache.length
            step_mask = mask[:, -(cached + 1) :]
            step_positions = _left_padded_positions(step_mask)[:, -1:]
        logits, _ = infer_gpt_with_persistent_kv_cache(
            model,
            sequence[:, -1:],
            cache,
            attention_mask=step_mask,
            position_ids=step_positions,
        )
        return logits

    cache.clear()
    return _prefill(model, sequence, mask, cache)


def _validate_kv_array(name, value):
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.ndim < 2:
        raise ValueError(f"{name} must have at least two dimensions")
    if not np.issubdtype(value.dtype, np.floating):
        raise TypeError(f"{name} must have a real floating dtype")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value


def _layout_without_time(shape):
    return tuple(shape[:-2]) + (int(shape[-1]),)
