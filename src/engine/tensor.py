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
    """

    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        self.data = np.array(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None

        # Computational graph bookkeeping
        self._children = set(_children)
        self._op = _op
        self._backward = lambda: None  # filled by each op

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

        Parameters
        ----------
        grad : array-like or None
            Gradient flowing into this tensor.  If None and this tensor is
            scalar, it defaults to 1.0 (standard loss.backward() call).
        """
        if not self.requires_grad:
            return

        # Initialise incoming gradient
        if grad is None:
            if self.data.shape == ():
                grad = np.ones((), dtype=np.float64)
            else:
                grad = np.ones_like(self.data, dtype=np.float64)
        self.grad = np.array(grad, dtype=np.float64)

        # Build topological ordering via DFS
        topo = []
        visited = set()

        def _build(node):
            if id(node) not in visited:
                visited.add(id(node))
                for child in node._children:
                    _build(child)
                topo.append(node)

        _build(self)

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
        return matmul(self, other)

    def __neg__(self):
        from .ops import mul
        return mul(self, Tensor(np.full_like(self.data, -1.0)))

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __truediv__(self, other):
        from .ops import mul, exp, log
        if isinstance(other, (int, float)):
            return mul(self, Tensor(np.full_like(self.data, 1.0 / other)))
        # a / b = a * exp(-log(b))  (works for tensor divisors)
        return mul(self, exp(-log(other)))

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
        """Return a new Tensor sharing data but excluded from the graph."""
        return Tensor(self.data, requires_grad=False)

    def numpy(self):
        """Return the underlying NumPy array."""
        return self.data
