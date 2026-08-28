"""Deterministic, read-only summaries for live ``nn.Module`` trees.

The report deliberately describes persistent Tensor structure rather than numerical
values. This keeps it cheap, strict-JSON-friendly, and useful for checkpoint/model
inspection without mutating module, Tensor, gradient, or NumPy RNG state.
"""

from nn.module import Module


def _shape_list(shape):
    return [int(size) for size in shape]


def module_summary(module):
    """Return a deterministic strict-JSON-friendly summary of one ``Module`` tree.

    Persistent tensors come from ``named_tensors()`` so frozen buffers are included.
    Trainability is determined from each Tensor's current ``requires_grad`` flag.
    Logical byte counts use the underlying NumPy array ``nbytes`` and intentionally
    exclude Python object, closure, and allocator overhead.
    """
    if not isinstance(module, Module):
        raise TypeError("module_summary module must be an nn.Module")

    tensors = tuple(module.named_tensors())
    modules = tuple(module.modules())

    entries = []
    persistent_elements = 0
    persistent_bytes = 0
    trainable_elements = 0
    trainable_bytes = 0
    trainable_tensors = 0

    for name, tensor in tensors:
        element_count = int(tensor.data.size)
        byte_count = int(tensor.data.nbytes)
        trainable = bool(tensor.requires_grad)

        persistent_elements += element_count
        persistent_bytes += byte_count
        if trainable:
            trainable_tensors += 1
            trainable_elements += element_count
            trainable_bytes += byte_count

        entries.append(
            {
                "name": name,
                "shape": _shape_list(tensor.shape),
                "dtype": str(tensor.data.dtype),
                "element_count": element_count,
                "byte_count": byte_count,
                "requires_grad": trainable,
                "has_grad": tensor.grad is not None,
                "mutation_version": int(tensor._version),
            }
        )

    frozen_tensors = len(entries) - trainable_tensors
    frozen_elements = persistent_elements - trainable_elements
    frozen_bytes = persistent_bytes - trainable_bytes

    return {
        "module_type": type(module).__name__,
        "module_count": len(modules),
        "persistent_tensor_count": len(entries),
        "trainable_tensor_count": trainable_tensors,
        "frozen_tensor_count": frozen_tensors,
        "persistent_element_count": persistent_elements,
        "trainable_element_count": trainable_elements,
        "frozen_element_count": frozen_elements,
        "persistent_byte_count": persistent_bytes,
        "trainable_byte_count": trainable_bytes,
        "frozen_byte_count": frozen_bytes,
        "tensors": entries,
    }
