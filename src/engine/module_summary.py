"""Deterministic, read-only summaries for live ``nn.Module`` trees.

The report describes persistent Tensor structure plus allocated gradient-buffer memory.
This keeps it cheap, strict-JSON-friendly, and useful for checkpoint/model/training
inspection without mutating module, Tensor, gradient, or NumPy RNG state.
"""

from nn.module import Module


def _shape_list(shape):
    return [int(size) for size in shape]


def module_summary(module):
    """Return a deterministic strict-JSON-friendly summary of one ``Module`` tree.

    Persistent tensors come from ``named_tensors()`` so frozen buffers are included.
    Trainability is determined from each Tensor's current ``requires_grad`` flag.
    Logical byte counts use NumPy array ``nbytes`` and intentionally exclude Python
    object, closure, and allocator overhead. Gradient totals count only buffers that
    are currently allocated; they are not persistent checkpoint state.
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
    gradient_tensors = 0
    gradient_elements = 0
    gradient_bytes = 0

    for name, tensor in tensors:
        element_count = int(tensor.data.size)
        byte_count = int(tensor.data.nbytes)
        trainable = bool(tensor.requires_grad)
        has_grad = tensor.grad is not None
        grad_element_count = int(tensor.grad.size) if has_grad else 0
        grad_byte_count = int(tensor.grad.nbytes) if has_grad else 0

        persistent_elements += element_count
        persistent_bytes += byte_count
        if trainable:
            trainable_tensors += 1
            trainable_elements += element_count
            trainable_bytes += byte_count
        if has_grad:
            gradient_tensors += 1
            gradient_elements += grad_element_count
            gradient_bytes += grad_byte_count

        entries.append(
            {
                "name": name,
                "shape": _shape_list(tensor.shape),
                "dtype": str(tensor.data.dtype),
                "element_count": element_count,
                "byte_count": byte_count,
                "requires_grad": trainable,
                "has_grad": has_grad,
                "gradient_element_count": grad_element_count,
                "gradient_byte_count": grad_byte_count,
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
        "gradient_tensor_count": gradient_tensors,
        "persistent_element_count": persistent_elements,
        "trainable_element_count": trainable_elements,
        "frozen_element_count": frozen_elements,
        "gradient_element_count": gradient_elements,
        "persistent_byte_count": persistent_bytes,
        "trainable_byte_count": trainable_bytes,
        "frozen_byte_count": frozen_bytes,
        "gradient_byte_count": gradient_bytes,
        "tensors": entries,
    }
