"""
module.py — Base Module class for all neural network components.

Provides:
  - parameters()  : recursively collect all leaf Tensors with requires_grad=True
  - zero_grad()   : reset all parameter gradients
  - __call__()    : invoke forward()
  - named_parameters() : same but returns (name, tensor) pairs for inspection

Design choice: modules register child modules by simply assigning them as
attributes.  parameters() traverses all attributes that are Module instances
or Tensor instances with requires_grad=True.
"""

from collections.abc import Mapping
from reprlib import recursive_repr

import numpy as np

from engine.tensor import Tensor


class Module:
    """Base class for all neural network modules."""

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    # ------------------------------------------------------------------
    # Parameter collection
    # ------------------------------------------------------------------
    def modules(self):
        """Yield this module and all nested child modules once."""
        found = []
        seen_modules = set()
        seen_containers = set()

        def _collect(obj):
            if isinstance(obj, Module):
                marker = id(obj)
                if marker in seen_modules:
                    return
                seen_modules.add(marker)
                found.append(obj)
                for val in vars(obj).values():
                    _collect(val)
            elif isinstance(obj, (list, tuple)):
                marker = id(obj)
                if marker in seen_containers:
                    return
                seen_containers.add(marker)
                for item in obj:
                    _collect(item)
            elif isinstance(obj, dict):
                marker = id(obj)
                if marker in seen_containers:
                    return
                seen_containers.add(marker)
                for item in obj.values():
                    _collect(item)

        _collect(self)
        return found

    def parameters(self):
        """
        Recursively collect all trainable Tensor parameters.
        Returns a flat list with no duplicates.
        """
        params = []
        seen_tensors = set()
        seen_modules = set()
        seen_containers = set()

        def _collect(obj):
            if isinstance(obj, Tensor) and obj.requires_grad:
                marker = id(obj)
                if marker not in seen_tensors:
                    seen_tensors.add(marker)
                    params.append(obj)
            elif isinstance(obj, Module):
                marker = id(obj)
                if marker in seen_modules:
                    return
                seen_modules.add(marker)
                for val in vars(obj).values():
                    _collect(val)
            elif isinstance(obj, (list, tuple)):
                marker = id(obj)
                if marker in seen_containers:
                    return
                seen_containers.add(marker)
                for item in obj:
                    _collect(item)
            elif isinstance(obj, dict):
                marker = id(obj)
                if marker in seen_containers:
                    return
                seen_containers.add(marker)
                for item in obj.values():
                    _collect(item)

        _collect(self)
        return params

    def named_tensors(self, prefix=""):
        """Yield all persistent tensors, including frozen tensors and buffers."""
        seen_tensors = set()
        seen_modules = set()
        seen_containers = set()

        def _collect(obj, pfx):
            if isinstance(obj, Tensor):
                marker = id(obj)
                if marker not in seen_tensors:
                    seen_tensors.add(marker)
                    yield pfx, obj
            elif isinstance(obj, Module):
                marker = id(obj)
                if marker in seen_modules:
                    return
                seen_modules.add(marker)
                for name, val in vars(obj).items():
                    full = f"{pfx}.{name}" if pfx else name
                    yield from _collect(val, full)
            elif isinstance(obj, (list, tuple)):
                marker = id(obj)
                if marker in seen_containers:
                    return
                seen_containers.add(marker)
                for i, item in enumerate(obj):
                    yield from _collect(item, f"{pfx}[{i}]")
            elif isinstance(obj, dict):
                marker = id(obj)
                if marker in seen_containers:
                    return
                seen_containers.add(marker)
                for key, item in obj.items():
                    yield from _collect(item, f"{pfx}[{key}]")

        yield from _collect(self, prefix)

    def named_parameters(self, prefix=""):
        """Yield (name, tensor) pairs for all trainable parameters."""
        seen_tensors = set()
        seen_modules = set()
        seen_containers = set()

        def _collect(obj, pfx):
            if isinstance(obj, Tensor) and obj.requires_grad:
                marker = id(obj)
                if marker not in seen_tensors:
                    seen_tensors.add(marker)
                    yield pfx, obj
            elif isinstance(obj, Module):
                marker = id(obj)
                if marker in seen_modules:
                    return
                seen_modules.add(marker)
                for name, val in vars(obj).items():
                    full = f"{pfx}.{name}" if pfx else name
                    yield from _collect(val, full)
            elif isinstance(obj, (list, tuple)):
                marker = id(obj)
                if marker in seen_containers:
                    return
                seen_containers.add(marker)
                for i, item in enumerate(obj):
                    yield from _collect(item, f"{pfx}[{i}]")
            elif isinstance(obj, dict):
                marker = id(obj)
                if marker in seen_containers:
                    return
                seen_containers.add(marker)
                for k, item in obj.items():
                    yield from _collect(item, f"{pfx}[{k}]")

        yield from _collect(self, prefix)

    def zero_grad(self):
        """Reset all parameter gradients to zero."""
        for p in self.parameters():
            p.zero_grad()

    def train(self, mode=True):
        """Set training mode recursively and return self."""
        for module in self.modules():
            module.training = mode
        return self

    def eval(self):
        """Set evaluation mode recursively and return self."""
        return self.train(False)

    def state_dict(self):
        """Return copies of all persistent tensor values."""
        return {name: tensor.data.copy() for name, tensor in self.named_tensors()}

    def load_state_dict(self, state, strict=True):
        """Validate then atomically copy persistent tensor values from ``state``."""
        if not isinstance(state, Mapping):
            raise TypeError("state_dict must be a mapping")
        if not isinstance(strict, (bool, np.bool_)):
            raise TypeError("state_dict strict flag must be boolean")
        for key in state:
            if not isinstance(key, str):
                raise TypeError("state_dict keys must be strings")

        tensors = dict(self.named_tensors())
        missing = sorted(set(tensors) - set(state))
        unexpected = sorted(set(state) - set(tensors))
        if strict and (missing or unexpected):
            raise ValueError(
                f"state_dict mismatch: missing={missing}, unexpected={unexpected}"
            )

        # Validate every destination before copying any value so a malformed
        # late entry cannot leave the module partially restored.
        for name, value in state.items():
            if name not in tensors:
                continue
            _validate_state_tensor(name, tensors[name], value)

        for name, value in state.items():
            if name not in tensors:
                continue
            tensors[name].data[:] = value

    def param_count(self):
        """Total number of trainable scalar parameters."""
        return sum(p.data.size for p in self.parameters())

    @recursive_repr(fillvalue="...")
    def __repr__(self):
        lines = [f"{self.__class__.__name__}("]
        for name, val in vars(self).items():
            if isinstance(val, (Module, Tensor)):
                lines.append(f"  ({name}): {val}")
        lines.append(")")
        return "\n".join(lines)


def _validate_state_tensor(name, tensor, value):
    if not isinstance(value, np.ndarray):
        raise TypeError(f"state_dict value for {name} must be a NumPy array")
    if tensor.data.shape != value.shape:
        raise ValueError(
            f"shape mismatch for {name}: expected {tensor.data.shape}, "
            f"got {value.shape}"
        )
    if (
        not np.issubdtype(value.dtype, np.number)
        or np.issubdtype(value.dtype, np.complexfloating)
    ):
        raise TypeError(
            f"state_dict value for {name} must have a real numeric dtype"
        )

    # Normal model tensors start finite and must stay finite. Some persistent
    # deterministic buffers intentionally contain -inf (GPT's causal mask) and
    # may have legacy migration rules in their owning module, so those buffers
    # are allowed to carry non-finite sentinels through this generic boundary.
    if np.isfinite(tensor.data).all() and not np.isfinite(value).all():
        raise ValueError(
            f"state_dict value for {name} must contain only finite values"
        )
