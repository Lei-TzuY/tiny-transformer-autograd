"""Preallocated key/value cache storage for incremental attention inference.

The existing attention inference paths represent a cache as ``{"k": ..., "v": ...}``
and grow it with ``np.concatenate``. That simple representation is useful, but repeated
single-token appends copy the complete history on every decode step. ``KVCacheBuffer``
provides a fixed-capacity alternative whose storage is allocated once and whose append
cost is proportional only to the newly appended chunk.

The time dimension is always the penultimate axis. This matches both cache layouts used
by the repository:

* single-head: ``(batch, time, key_or_value_dim)``
* multi-head/GQA: ``(batch, heads, time, key_or_value_dim)``

``view()`` returns read-only live slices using the same ``{"k", "v"}`` mapping contract
as the existing attention APIs. Those views share this buffer's storage and are intended
for immediate inference use. Call ``snapshot()`` when an independently owned cache is
needed across later ``clear()``, ``truncate()``, or reuse operations.
"""

from numbers import Integral
import threading

import numpy as np


class KVCacheBuffer:
    """Fixed-capacity append-only K/V storage with O(new_tokens) append copies."""

    def __init__(self, capacity):
        if isinstance(capacity, (bool, np.bool_)) or not isinstance(capacity, Integral):
            raise TypeError("capacity must be a positive integer")
        capacity = int(capacity)
        if capacity <= 0:
            raise ValueError("capacity must be positive")

        self._capacity = capacity
        self._length = 0
        self._k_storage = None
        self._v_storage = None
        self._k_layout = None
        self._v_layout = None
        self._k_dtype = None
        self._v_dtype = None
        self._lock = threading.RLock()

    @property
    def capacity(self):
        return self._capacity

    @property
    def length(self):
        with self._lock:
            return self._length

    @property
    def remaining(self):
        with self._lock:
            return self._capacity - self._length

    @property
    def initialized(self):
        with self._lock:
            return self._k_storage is not None

    @property
    def storage_nbytes(self):
        """Bytes reserved by the backing arrays, or zero before first append."""
        with self._lock:
            if self._k_storage is None:
                return 0
            return int(self._k_storage.nbytes + self._v_storage.nbytes)

    @property
    def live_nbytes(self):
        """Bytes occupied by the currently visible prefix."""
        with self._lock:
            if self._k_storage is None:
                return 0
            return int(
                _live_nbytes(self._k_storage, self._length)
                + _live_nbytes(self._v_storage, self._length)
            )

    def append(self, key, value):
        """Append one non-empty cache chunk and return the resulting live view.

        Both sources are snapshotted into ordinary independent NumPy arrays before any
        buffer write. This makes self-append from one of this buffer's own live views
        well-defined and keeps validation failures transactional.
        """
        with self._lock:
            key_copy, value_copy = self._snapshot_chunk(key, value)
            chunk_len = key_copy.shape[-2]
            end = self._length + chunk_len
            if end > self._capacity:
                raise OverflowError(
                    f"cache capacity {self._capacity} exceeded by append ending at {end}"
                )

            if self._k_storage is None:
                k_storage, v_storage = self._allocate_storage(key_copy, value_copy)
                start = 0
                k_storage[..., start:end, :] = key_copy
                v_storage[..., start:end, :] = value_copy
                self._k_storage = k_storage
                self._v_storage = v_storage
                self._k_layout = _layout_without_time(key_copy.shape)
                self._v_layout = _layout_without_time(value_copy.shape)
                self._k_dtype = key_copy.dtype
                self._v_dtype = value_copy.dtype
                self._length = end
                return self._view_unlocked()

            self._validate_internal_state()
            self._validate_layout(key_copy, value_copy)
            if not self._k_storage.flags.writeable or not self._v_storage.flags.writeable:
                raise RuntimeError("cache backing storage is unexpectedly read-only")

            start = self._length
            # The destinations are plain owned ndarrays and both source candidates were
            # independently snapshotted above, so these two assignments cannot alias.
            self._k_storage[..., start:end, :] = key_copy
            self._v_storage[..., start:end, :] = value_copy
            self._length = end
            return self._view_unlocked()

    def view(self):
        """Return read-only live cache slices sharing this buffer's backing storage."""
        with self._lock:
            if self._k_storage is None:
                raise RuntimeError("cache buffer is not initialized")
            self._validate_internal_state()
            return self._view_unlocked()

    def snapshot(self):
        """Return independent writable arrays containing the current live cache."""
        with self._lock:
            if self._k_storage is None:
                raise RuntimeError("cache buffer is not initialized")
            self._validate_internal_state()
            return {
                "k": np.array(self._live_slice(self._k_storage), copy=True, subok=False),
                "v": np.array(self._live_slice(self._v_storage), copy=True, subok=False),
            }

    def truncate(self, length):
        """Shorten the visible prefix without reallocating or rewriting storage."""
        if isinstance(length, (bool, np.bool_)) or not isinstance(length, Integral):
            raise TypeError("length must be a non-negative integer")
        length = int(length)
        if length < 0:
            raise ValueError("length must be non-negative")

        with self._lock:
            if self._k_storage is None:
                if length == 0:
                    return self
                raise ValueError("cannot truncate an uninitialized cache to non-zero length")
            self._validate_internal_state()
            if length > self._length:
                raise ValueError("cannot extend a cache with truncate()")
            self._length = length
            return self

    def clear(self):
        """Reset logical length to zero while retaining the allocated storage."""
        with self._lock:
            if self._k_storage is not None:
                self._validate_internal_state()
            self._length = 0
            return self

    def _snapshot_chunk(self, key, value):
        key = _validate_array("key", key)
        value = _validate_array("value", value)
        if key.ndim != value.ndim:
            raise ValueError("key and value must have the same rank")
        if key.shape[:-2] != value.shape[:-2]:
            raise ValueError("key and value leading cache dimensions must match")
        if key.shape[-2] != value.shape[-2]:
            raise ValueError("key and value must contain the same number of time steps")
        if key.shape[-2] == 0:
            raise ValueError("cache append must contain at least one time step")

        # Always detach from caller storage before capacity/layout preflight and commit.
        return (
            np.array(key, copy=True, order="C", subok=False),
            np.array(value, copy=True, order="C", subok=False),
        )

    def _allocate_storage(self, key, value):
        k_shape = key.shape[:-2] + (self._capacity, key.shape[-1])
        v_shape = value.shape[:-2] + (self._capacity, value.shape[-1])
        # Allocate both successfully before publishing either as object state.
        k_storage = np.empty(k_shape, dtype=key.dtype)
        v_storage = np.empty(v_shape, dtype=value.dtype)
        return k_storage, v_storage

    def _validate_layout(self, key, value):
        if _layout_without_time(key.shape) != self._k_layout:
            raise ValueError("key cache layout does not match the initialized buffer")
        if _layout_without_time(value.shape) != self._v_layout:
            raise ValueError("value cache layout does not match the initialized buffer")
        if key.dtype != self._k_dtype:
            raise TypeError("key dtype does not match the initialized buffer")
        if value.dtype != self._v_dtype:
            raise TypeError("value dtype does not match the initialized buffer")

    def _validate_internal_state(self):
        if self._k_storage is None or self._v_storage is None:
            raise RuntimeError("cache backing storage is incomplete")
        if type(self._k_storage) is not np.ndarray or type(self._v_storage) is not np.ndarray:
            raise RuntimeError("cache backing storage must remain ordinary NumPy arrays")
        if self._k_storage.dtype != self._k_dtype or self._v_storage.dtype != self._v_dtype:
            raise RuntimeError("cache backing dtype metadata is inconsistent")
        if _layout_without_time(self._k_storage.shape) != self._k_layout:
            raise RuntimeError("key backing layout metadata is inconsistent")
        if _layout_without_time(self._v_storage.shape) != self._v_layout:
            raise RuntimeError("value backing layout metadata is inconsistent")
        if self._k_storage.shape[-2] != self._capacity:
            raise RuntimeError("key backing capacity is inconsistent")
        if self._v_storage.shape[-2] != self._capacity:
            raise RuntimeError("value backing capacity is inconsistent")
        if not isinstance(self._length, int) or isinstance(self._length, bool):
            raise RuntimeError("cache length metadata is invalid")
        if self._length < 0 or self._length > self._capacity:
            raise RuntimeError("cache length metadata is outside capacity")

    def _view_unlocked(self):
        key = self._live_slice(self._k_storage).view()
        value = self._live_slice(self._v_storage).view()
        key.flags.writeable = False
        value.flags.writeable = False
        return {"k": key, "v": value}

    def _live_slice(self, array):
        return array[..., : self._length, :]

    def __len__(self):
        return self.length

    def __repr__(self):
        with self._lock:
            status = "initialized" if self._k_storage is not None else "uninitialized"
            return (
                f"KVCacheBuffer(capacity={self._capacity}, length={self._length}, "
                f"{status})"
            )


def _validate_array(name, value):
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


def _live_nbytes(array, length):
    elements_per_time = int(np.prod(array.shape[:-2], dtype=np.int64)) * array.shape[-1]
    return int(elements_per_time * length * array.dtype.itemsize)
