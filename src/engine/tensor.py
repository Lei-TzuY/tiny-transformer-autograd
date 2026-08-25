"""
tensor.py — Core Tensor class with automatic differentiation.

A Tensor wraps a NumPy array and records the computational graph
so that backward() can propagate gradients via reverse-mode autodiff.

Math recap
----------
Given a scalar loss L and a node y = f(x₁, x₂, ...):
  ∂L/∂xᵢ += (∂L/∂y) · (∂y/∂xᵢ)

We implement this by storing a _backward closure at each node that
accumulates into its parents' .grad fields.  backward() executes
closures in reverse topological order (like reverse-mode AD).
"""

from numbers import Real
import weakref

import numpy as np

from .grad_mode import is_grad_enabled


def _no_backward():
    """Shared do-nothing closure for nodes that cannot propagate a gradient."""
    return None


def _snapshot_index(index):
    """Copy mutable NumPy indexing state captured by a backward closure."""
    if isinstance(index, np.ndarray):
        return index.copy()
    if isinstance(index, tuple):
        return tuple(_snapshot_index(item) for item in index)
    if isinstance(index, list):
        return [_snapshot_index(item) for item in index]
    return index


def _ordered_unique_children(children):
    """Deduplicate Tensor parents by identity while preserving first-seen order."""
    unique = []
    seen = set()
    for child in children:
        child_id = id(child)
        if child_id in seen:
            continue
        seen.add(child_id)
        unique.append(child)
    return tuple(unique)


class _VersionedArray(np.ndarray):
    """ndarray view that increments its owning Tensor version on normal writes."""

    def __new__(cls, data, owner=None):
        array = np.array(data, dtype=np.float64, copy=True).view(cls)
        array._owner_ref = None if owner is None else weakref.ref(owner)
        return array

    def __array_finalize__(self, source):
        owner_ref = getattr(source, "_owner_ref", None)
        if owner_ref is None:
            self._owner_ref = None
            return
        # Preserve ownership only for views that still alias the Tensor's storage.
        try:
            aliases_owner = np.shares_memory(self, source)
        except ValueError:
            aliases_owner = False
        self._owner_ref = owner_ref if aliases_owner else None

    def _mark_modified(self):
        owner_ref = getattr(self, "_owner_ref", None)
        owner = None if owner_ref is None else owner_ref()
        if owner is not None:
            owner._version += 1

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._mark_modified()

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # Compute on ordinary ndarray views so read-only ufuncs return normal
        # NumPy arrays and do not accidentally propagate Tensor ownership.
        raw_inputs = tuple(
            np.asarray(value) if isinstance(value, _VersionedArray) else value
            for value in inputs
        )
        outputs = kwargs.get("out")
        if outputs is not None:
            kwargs["out"] = tuple(
                np.asarray(value) if isinstance(value, _VersionedArray) else value
                for value in outputs
            )

        result = getattr(ufunc, method)(*raw_inputs, **kwargs)

        # ufunc.at mutates its first input but does not use an ``out`` argument.
        if method == "at" and inputs and isinstance(inputs[0], _VersionedArray):
            inputs[0]._mark_modified()

        if outputs is not None:
            seen_owners = set()
            for output in outputs:
                if not isinstance(output, _VersionedArray):
                    continue
                owner_ref = getattr(output, "_owner_ref", None)
                owner = None if owner_ref is None else owner_ref()
                if owner is not None and id(owner) not in seen_owners:
                    owner._version += 1
                    seen_owners.add(id(owner))
            return outputs[0] if len(outputs) == 1 else outputs
        return result

    def copy(self, order="C"):
        """Return an independent ordinary ndarray with no Tensor ownership."""
        return np.array(self, dtype=self.dtype, copy=True, order=order, subok=False)

    def fill(self, value):
        super().fill(value)
        self._mark_modified()

    def put(self, indices, values, mode="raise"):
        super().put(indices, values, mode=mode)
        self._mark_modified()

    def sort(self, axis=-1, kind=None, order=None):
        super().sort(axis=axis, kind=kind, order=order)
        self._mark_modified()

    def partition(self, kth, axis=-1, kind="introselect", order=None):
        super().partition(kth, axis=axis, kind=kind, order=order)
        self._mark_modified()

    def resize(self, new_shape, refcheck=True):
        super().resize(new_shape, refcheck=refcheck)
        self._mark_modified()

    def setfield(self, val, dtype, offset=0):
        super().setfield(val, dtype, offset=offset)
        self._mark_modified()

    def byteswap(self, inplace=False):
        result = super().byteswap(inplace=inplace)
        if inplace:
            self._mark_modified()
        return result


class Tensor:
    """
    A multi-dimensional array that participates in automatic differentiation.

    Parameters
    ----------
    data : array-like
        The underlying numerical data.
    requires_grad : bool
        If True, gradients will be computed for this tensor.
    _children : tuple[Tensor, ...]
        Parent tensors in the computational graph (set by ops, not users).
    _op : str
        Human-readable label of the op that produced this tensor.

    Inside a ``no_grad()`` block an op result (a tensor with ``_children``)
    is created detached: no parents, no gradient buffer, no backward closure.
    Explicitly constructed leaves keep the ``requires_grad`` they were given.
    """

    # Ask NumPy binary operators to defer to Tensor's reflected methods.
    __array_priority__ = 1000

    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        children = _ordered_unique_children(_children)
        is_op_result = bool(children)
        recording = is_grad_enabled()
        requested_grad = bool(requires_grad)

        # Remember why a result became detached before discarding its parents.
        # The marker propagates through further constant-only arithmetic so a
        # value derived from a suppressed forward remains a loud backward
        # error. Explicit constants and operations on constants never acquire
        # it and retain their historical no-op backward behaviour.
        parent_was_suppressed = any(
            getattr(child, "_detached_by_no_grad", False) for child in children
        )
        detached_by_no_grad = is_op_result and (
            parent_was_suppressed or (not recording and requested_grad)
        )

        if is_op_result and not recording:
            # Recording is off: keep the value, drop the graph.
            requires_grad = False
        if is_op_result and not requires_grad:
            # A result that cannot receive a gradient has no reason to retain
            # its parents. This also prunes constant/frozen-only branches while
            # allowing their values to feed a later trainable operation.
            children = ()

        self._version = 0
        self._data = _VersionedArray(data, owner=self)
        self.requires_grad = requires_grad
        self.grad = (
            np.zeros(self._data.shape, dtype=np.float64) if requires_grad else None
        )
        self._detached_by_no_grad = detached_by_no_grad and not requires_grad

        # Keep parent traversal deterministic. A set would deduplicate identities
        # but would also make reverse-mode accumulation order depend on hashes / ids.
        self._children = children
        self._parent_versions = {
            child: child._version for child in self._children
        }
        self._forward_version = self._version
        self._op = _op
        self._backward = lambda: None  # filled by each op

    @property
    def data(self):
        """Tracked NumPy storage; normal in-place writes invalidate old graphs."""
        return self._data

    @data.setter
    def data(self, value):
        # Augmented attribute assignment writes the same in-place-mutated array
        # back to the property. The ufunc already bumped the version, so do not
        # replace storage or bump twice in that common optimizer-style path.
        if hasattr(self, "_data") and value is self._data:
            return
        self._data = _VersionedArray(value, owner=self)
        if hasattr(self, "_version"):
            self._version += 1

    # ------------------------------------------------------------------
    # Backward closure
    # ------------------------------------------------------------------
    @property
    def _backward(self):
        """Closure that pushes this node's gradient into its parents."""
        return self._backward_fn

    @_backward.setter
    def _backward(self, function):
        # A node with no gradient of its own can never contribute one, so its
        # closure is dropped instead of stored.  That releases every forward
        # intermediate the closure captured — the reason no_grad() is cheap —
        # and guarantees a detached node is never asked to propagate a
        # gradient it does not have.
        graph_node = self.requires_grad and bool(self._children)
        self._backward_fn = function if graph_node else _no_backward

    def _validate_graph_versions(self):
        """Reject graph reuse after any tracked Tensor data mutation."""
        if self._children and self._version != self._forward_version:
            raise RuntimeError(
                "tensor data was modified after forward; rebuild the forward "
                "graph before calling backward()"
            )
        for child in self._children:
            expected = self._parent_versions[child]
            if child._version != expected:
                raise RuntimeError(
                    "tensor data was modified after forward; rebuild the forward "
                    "graph before calling backward()"
                )

    # ------------------------------------------------------------------
    # Shape helpers
    # ------------------------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def T(self):
        """Full matrix transpose, registered in the computational graph."""
        from .ops import transpose
        return transpose(self, None)

    def reshape(self, *shape):
        """Reshape registered in graph. Accepts reshape(a, b) or reshape((a, b))."""
        from .ops import reshape
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return reshape(self, shape)

    def transpose(self, axes=None):
        """Permute axes, registered in graph."""
        from .ops import transpose as _t
        return _t(self, axes)

    def __repr__(self):
        return (
            f"Tensor(shape={self.shape}, op='{self._op}', "
            f"requires_grad={self.requires_grad})"
        )

    # ------------------------------------------------------------------
    # Gradient utilities
    # ------------------------------------------------------------------
    def zero_grad(self):
        """Reset gradient to zero (in-place)."""
        if self.grad is not None:
            self.grad = np.zeros(self.data.shape, dtype=np.float64)

    def _ensure_grad(self):
        """Lazily initialise grad if it is None (e.g. after a detach)."""
        if self.grad is None and self.requires_grad:
            self.grad = np.zeros(self.data.shape, dtype=np.float64)

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------
    def backward(self, grad=None):
        """
        Compute gradients for all ancestor tensors via reverse-mode AD.

        Builds the topological order of the computational graph rooted at
        self, then calls each node's _backward closure in reverse order.
        Leaf gradients accumulate across calls; non-leaf gradients are reset
        on every call so stale intermediate values are never propagated.

        Parameters
        ----------
        grad : array-like or None
            Gradient flowing into this tensor. If None, defaults to an array
            of ones with this tensor's shape. For non-scalar tensors this is
            equivalent to differentiating ``self.sum()``. Explicit seeds must
            contain real finite numeric values and match this tensor's shape.

        Raises
        ------
        RuntimeError
            If this tensor was produced by an operation while recording was
            disabled, or if tracked tensor data used by this graph was mutated
            after the forward pass.
        TypeError
            If an explicit backward gradient is not real numeric data.
        ValueError
            If an explicit backward gradient has the wrong shape or contains
            NaN or infinity.
        """
        if not self.requires_grad:
            if self._detached_by_no_grad:
                raise RuntimeError(
                    "backward() called on a tensor detached by no_grad(); "
                    "build the forward pass outside no_grad() (or re-enable "
                    "it with enable_grad())"
                )
            return

        # Prepare and validate the incoming vector-Jacobian product seed before
        # traversing the graph or touching any existing gradient buffers.
        if grad is None:
            incoming = np.ones_like(self.data, dtype=np.float64)
        else:
            raw = np.asarray(grad)
            is_integer = np.issubdtype(raw.dtype, np.integer)
            is_floating = np.issubdtype(raw.dtype, np.floating)
            if np.issubdtype(raw.dtype, np.bool_) or not (
                is_integer or is_floating
            ):
                raise TypeError("backward gradient must contain real numeric values")
            incoming = np.asarray(raw, dtype=np.float64)

        if incoming.shape != self.shape:
            raise ValueError(
                f"backward gradient shape mismatch: expected {self.shape}, "
                f"got {incoming.shape}"
            )
        if not np.isfinite(incoming).all():
            raise ValueError("backward gradient must contain only finite values")

        # Build a post-order topological traversal iteratively. A recursive
        # DFS fails on perfectly valid long chains at Python's recursion limit.
        topo = []
        visited = set()
        stack = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            node_id = id(node)
            if expanded:
                topo.append(node)
                continue
            if node_id in visited:
                continue
            visited.add(node_id)
            stack.append((node, True))
            for child in node._children:
                if id(child) not in visited:
                    stack.append((child, False))

        # Validate the whole tape before mutating any caller-owned gradient.
        # This keeps mutation failures transactional even when the bad tensor is
        # buried deep in a large graph.
        for node in topo:
            node._validate_graph_versions()

        # Intermediates are implementation details of this VJP and must start
        # clean on each call. Leaves intentionally retain their gradients so
        # independent graphs and repeated backward calls accumulate normally.
        for node in topo:
            if node._children and node.requires_grad:
                node.grad = np.zeros(node.data.shape, dtype=np.float64)

        self._ensure_grad()
        self.grad += incoming

        # Propagate gradients in reverse topological order
        for node in reversed(topo):
            node._backward()

    # ------------------------------------------------------------------
    # Operator overloads (delegate to ops module via late import)
    # ------------------------------------------------------------------
    def __add__(self, other):
        from .ops import add
        other = other if isinstance(other, Tensor) else Tensor(other)
        return add(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        from .ops import mul
        other = other if isinstance(other, Tensor) else Tensor(other)
        return mul(self, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        from .ops import matmul
        other = other if isinstance(other, Tensor) else Tensor(other)
        return matmul(self, other)

    def __rmatmul__(self, other):
        from .ops import matmul
        other = other if isinstance(other, Tensor) else Tensor(other)
        return matmul(other, self)

    def __neg__(self):
        from .ops import mul
        return mul(self, Tensor(np.full_like(self.data, -1.0)))

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __truediv__(self, other):
        from .ops import div
        other = other if isinstance(other, Tensor) else Tensor(other)
        return div(self, other)

    def __rtruediv__(self, other):
        from .ops import div
        other = other if isinstance(other, Tensor) else Tensor(other)
        return div(other, self)

    def __pow__(self, exponent):
        """Raise this tensor elementwise to one finite real scalar exponent."""
        if isinstance(exponent, (bool, np.bool_)) or not isinstance(exponent, Real):
            raise TypeError("power exponent must be a real scalar")
        exponent = float(exponent)
        if not np.isfinite(exponent):
            raise ValueError("power exponent must be finite")

        out = Tensor(
            self.data ** exponent,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op=f"**{exponent:g}",
        )

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                if exponent == 0.0:
                    # d(x**0)/dx = 0 everywhere under NumPy's x**0 == 1
                    # convention, including x=0. Avoid evaluating 0**-1,
                    # which would otherwise turn the exact zero into NaN.
                    return
                self.grad += exponent * (self.data ** (exponent - 1)) * out.grad

        out._backward = _backward
        return out

    def __getitem__(self, idx):
        """Slice / index; gradient flows back via scatter-add."""
        # Advanced NumPy indices are mutable caller-owned arrays/lists. The VJP
        # must use the exact indexing state that produced this forward value.
        saved_idx = _snapshot_index(idx)
        out = Tensor(
            self.data[saved_idx],
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="getitem",
        )

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                # Scatter gradient back to the original forward indices.
                np.add.at(self.grad, saved_idx, out.grad)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Detach / clone
    # ------------------------------------------------------------------
    def detach(self):
        """
        Return a graph-free copy of this tensor.

        The constructor copies, so the result owns its own storage: later
        writes to ``detach().data`` do not touch this tensor.  Use it to read
        a value out of the graph; use ``no_grad()`` to avoid building one.
        """
        return Tensor(self.data, requires_grad=False)

    def numpy(self):
        """Return the tracked underlying NumPy array."""
        return self.data
