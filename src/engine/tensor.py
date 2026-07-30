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

import numpy as np

from .grad_mode import is_grad_enabled


def _no_backward():
    """Shared do-nothing closure for nodes that cannot propagate a gradient."""
    return None


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
        if _children and not is_grad_enabled():
            # Recording is off: keep the value, drop the graph.
            requires_grad = False
            _children = ()

        self.data = np.array(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None

        # Computational graph bookkeeping
        self._children = set(_children)
        self._op = _op
        self._backward = lambda: None  # filled by each op

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
            self.grad = np.zeros_like(self.data)

    def _ensure_grad(self):
        """Lazily initialise grad if it is None (e.g. after a detach)."""
        if self.grad is None and self.requires_grad:
            self.grad = np.zeros_like(self.data)

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
            equivalent to differentiating ``self.sum()``.

        Raises
        ------
        RuntimeError
            If this tensor carries no gradient while recording is disabled.
            Differentiating a graph that was built outside the block is still
            allowed; this only catches the common mistake of building the
            forward pass *inside* ``no_grad()`` and then expecting gradients.
        """
        if not self.requires_grad:
            if not is_grad_enabled():
                raise RuntimeError(
                    "backward() called on a tensor with no gradient while grad "
                    "recording is disabled; build the forward pass outside "
                    "no_grad() (or re-enable it with enable_grad())"
                )
            return

        # Prepare and validate the incoming vector-Jacobian product seed.
        if grad is None:
            grad = np.ones_like(self.data, dtype=np.float64)
        incoming = np.asarray(grad, dtype=np.float64)
        if incoming.shape != self.shape:
            raise ValueError(
                f"backward gradient shape mismatch: expected {self.shape}, "
                f"got {incoming.shape}"
            )

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

        # Intermediates are implementation details of this VJP and must start
        # clean on each call. Leaves intentionally retain their gradients so
        # independent graphs and repeated backward calls accumulate normally.
        for node in topo:
            if node._children and node.requires_grad:
                node.grad = np.zeros_like(node.data)

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
        """Scalar-exponent power: x ** n."""
        assert isinstance(exponent, (int, float)), "only scalar exponents"
        out = Tensor(
            self.data ** exponent,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op=f"**{exponent}",
        )

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += exponent * (self.data ** (exponent - 1)) * out.grad

        out._backward = _backward
        return out

    def __getitem__(self, idx):
        """Slice / index; gradient flows back via scatter-add."""
        out = Tensor(
            self.data[idx],
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="getitem",
        )

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                # Scatter gradient back to the original indices
                np.add.at(self.grad, idx, out.grad)

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
        """Return the underlying NumPy array."""
        return self.data
