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
        seen = set()

        def _collect(obj):
            if isinstance(obj, Module):
                if id(obj) in seen:
                    return
                seen.add(id(obj))
                found.append(obj)
                for val in vars(obj).values():
                    _collect(val)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    _collect(item)
            elif isinstance(obj, dict):
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
        seen = set()

        def _collect(obj):
            if isinstance(obj, Tensor) and obj.requires_grad:
                if id(obj) not in seen:
                    seen.add(id(obj))
                    params.append(obj)
            elif isinstance(obj, Module):
                for name in vars(obj):
                    _collect(getattr(obj, name))
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    _collect(item)
            elif isinstance(obj, dict):
                for item in obj.values():
                    _collect(item)

        _collect(self)
        return params

    def named_tensors(self, prefix=""):
        """Yield all persistent tensors, including frozen tensors and buffers."""
        seen = set()

        def _collect(obj, pfx):
            if isinstance(obj, Tensor):
                if id(obj) not in seen:
                    seen.add(id(obj))
                    yield pfx, obj
            elif isinstance(obj, Module):
                for name, val in vars(obj).items():
                    full = f"{pfx}.{name}" if pfx else name
                    yield from _collect(val, full)
            elif isinstance(obj, (list, tuple)):
                for i, item in enumerate(obj):
                    yield from _collect(item, f"{pfx}[{i}]")
            elif isinstance(obj, dict):
                for key, item in obj.items():
                    yield from _collect(item, f"{pfx}[{key}]")

        yield from _collect(self, prefix)

    def named_parameters(self, prefix=""):
        """Yield (name, tensor) pairs for all trainable parameters."""
        seen = set()

        def _collect(obj, pfx):
            if isinstance(obj, Tensor) and obj.requires_grad:
                if id(obj) not in seen:
                    seen.add(id(obj))
                    yield pfx, obj
            elif isinstance(obj, Module):
                for name, val in vars(obj).items():
                    full = f"{pfx}.{name}" if pfx else name
                    yield from _collect(val, full)
            elif isinstance(obj, (list, tuple)):
                for i, item in enumerate(obj):
                    yield from _collect(item, f"{pfx}[{i}]")
            elif isinstance(obj, dict):
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
        """Load tensor values from a state dictionary."""
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
            tensor = tensors[name]
            if tensor.data.shape != value.shape:
                raise ValueError(
                    f"shape mismatch for {name}: expected {tensor.data.shape}, "
                    f"got {value.shape}"
                )

        for name, value in state.items():
            if name not in tensors:
                continue
            tensor = tensors[name]
            tensor.data[:] = value

    def param_count(self):
        """Total number of trainable scalar parameters."""
        return sum(p.data.size for p in self.parameters())

    def __repr__(self):
        lines = [f"{self.__class__.__name__}("]
        for name, val in vars(self).items():
            if isinstance(val, (Module, Tensor)):
                lines.append(f"  ({name}): {val}")
        lines.append(")")
        return "\n".join(lines)
