"""Convert GPT attention projections between MHA, GQA, and MQA layouts."""

from numbers import Integral

import numpy as np

from .transformer import GPT


_KV_ROW_SUFFIXES = (
    ".attn.W_k.weight",
    ".attn.W_v.weight",
    ".attn.W_k.lora_B",
    ".attn.W_v.lora_B",
)


def convert_gpt_kv_heads(model: GPT, num_kv_heads: int) -> GPT:
    """Return an independent GPT with a different number of K/V heads.

    Expansion repeats each source K/V head contiguously, which preserves the
    represented attention function exactly (up to normal floating-point
    ordering differences). Compression averages contiguous source K/V heads in
    equal-sized groups. The same row transform is applied to LoRA ``lora_B``
    matrices, so adapters remain structurally separate from the base weights.

    Source and target K/V head counts must divide one another. This keeps every
    target group aligned with complete source groups instead of inventing an
    ambiguous remapping for crossing partitions such as 3 -> 4 heads.

    The source model, gradients, Tensor mutation versions, and NumPy global RNG
    state are left unchanged. The returned model has independent Tensor storage,
    preserves model/trainability mode and the runtime gradient-checkpoint flag,
    and starts with no copied live gradient buffers.
    """
    if not isinstance(model, GPT):
        raise TypeError("model must be a GPT")
    num_kv_heads = _positive_int("num_kv_heads", num_kv_heads)
    if model.num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")

    source_kv_heads = _positive_int("model.num_kv_heads", model.num_kv_heads)
    if source_kv_heads != num_kv_heads and not (
        source_kv_heads % num_kv_heads == 0
        or num_kv_heads % source_kv_heads == 0
    ):
        raise ValueError(
            "source and target num_kv_heads must divide one another"
        )

    source_state = model.state_dict()
    source_tensors = dict(model.named_tensors())
    source_gradients = {
        name: (tensor.grad, None if tensor.grad is None else tensor.grad.copy())
        for name, tensor in source_tensors.items()
    }
    source_versions = {
        name: getattr(tensor, "_version", None)
        for name, tensor in source_tensors.items()
    }
    rng_state = np.random.get_state()

    config = dict(model.config())
    if num_kv_heads == model.num_heads:
        config.pop("num_kv_heads", None)
    else:
        config["num_kv_heads"] = num_kv_heads

    try:
        target = GPT(**config)
        target_state = target.state_dict()
        if set(target_state) != set(source_state):
            missing = sorted(set(target_state) - set(source_state))
            unexpected = sorted(set(source_state) - set(target_state))
            raise RuntimeError(
                "converted GPT state keys differ from source: "
                f"missing={missing}, unexpected={unexpected}"
            )

        converted = {}
        head_dim = model.d_model // model.num_heads
        for name, destination in target_state.items():
            source = source_state[name]
            if source.shape == destination.shape:
                converted[name] = source.copy()
                continue
            if not name.endswith(_KV_ROW_SUFFIXES):
                raise RuntimeError(
                    f"unexpected shape change while converting {name}: "
                    f"{source.shape} -> {destination.shape}"
                )
            converted[name] = _resize_kv_rows(
                source,
                source_kv_heads,
                num_kv_heads,
                head_dim,
                name,
            )
            if converted[name].shape != destination.shape:
                raise RuntimeError(
                    f"internal KV conversion shape mismatch for {name}: "
                    f"expected {destination.shape}, got {converted[name].shape}"
                )

        target.load_state_dict(converted, strict=True)
        target.grad_checkpoint = bool(getattr(model, "grad_checkpoint", False))
        _copy_trainability(source_tensors, target)
        target.train(bool(getattr(model, "training", True)))
        return target
    finally:
        np.random.set_state(rng_state)
        _assert_source_unchanged(
            model,
            source_state,
            source_tensors,
            source_gradients,
            source_versions,
        )


def _positive_int(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _resize_kv_rows(values, source_heads, target_heads, head_dim, name):
    values = np.asarray(values)
    expected_rows = source_heads * head_dim
    if values.ndim < 1 or values.shape[0] != expected_rows:
        raise RuntimeError(
            f"source KV rows for {name} must start with {expected_rows}, "
            f"got {values.shape}"
        )
    if (
        not np.issubdtype(values.dtype, np.number)
        or np.issubdtype(values.dtype, np.complexfloating)
    ):
        raise TypeError(f"source KV rows for {name} must be real numeric")
    if not np.isfinite(values).all():
        raise ValueError(f"source KV rows for {name} must contain only finite values")

    tail = values.shape[1:]
    headed = np.array(values, dtype=np.float64, copy=True).reshape(
        (source_heads, head_dim) + tail
    )
    if source_heads == target_heads:
        resized = headed
    elif target_heads > source_heads:
        if target_heads % source_heads != 0:
            raise ValueError("target KV head count must be a multiple of source")
        resized = np.repeat(headed, target_heads // source_heads, axis=0)
    else:
        if source_heads % target_heads != 0:
            raise ValueError("source KV head count must be a multiple of target")
        group = source_heads // target_heads
        pieces = []
        for target_head in range(target_heads):
            start = target_head * group
            pieces.append(_stable_equal_mean(headed[start : start + group]))
        resized = np.stack(pieces, axis=0)
    return np.array(
        resized.reshape((target_heads * head_dim,) + tail),
        dtype=np.float64,
        copy=True,
    )


def _stable_equal_mean(values):
    """Equal mean over axis 0 without overflowing same/opposite-sign extremes."""
    if values.shape[0] == 0:
        raise ValueError("cannot average an empty KV-head group")
    mean = np.array(values[0], dtype=np.float64, copy=True)
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        for count, sample in enumerate(values[1:], start=2):
            sample = np.asarray(sample, dtype=np.float64)
            same_sign = np.signbit(mean) == np.signbit(sample)
            candidate = mean.copy()
            if np.any(same_sign):
                candidate[same_sign] = mean[same_sign] + (
                    sample[same_sign] - mean[same_sign]
                ) / count
            opposite = ~same_sign
            if np.any(opposite):
                previous_weight = (count - 1) / count
                sample_weight = 1.0 / count
                candidate[opposite] = (
                    mean[opposite] * previous_weight
                    + sample[opposite] * sample_weight
                )
            mean = candidate
    if not np.isfinite(mean).all():
        raise OverflowError("finite KV-head mean became non-finite")
    return mean


def _copy_trainability(source_tensors, target):
    target_tensors = dict(target.named_tensors())
    if set(target_tensors) != set(source_tensors):
        raise RuntimeError("converted GPT tensor names differ while copying trainability")
    for name, source in source_tensors.items():
        target_tensors[name].requires_grad = bool(source.requires_grad)
        target_tensors[name].grad = None


def _assert_source_unchanged(
    model,
    source_state,
    source_tensors,
    source_gradients,
    source_versions,
):
    """Fail loudly if conversion code ever mutates its source model."""
    current_tensors = dict(model.named_tensors())
    if set(current_tensors) != set(source_tensors):
        raise RuntimeError("GPT KV-head conversion mutated source tensor bindings")
    for name, tensor in current_tensors.items():
        if tensor is not source_tensors[name]:
            raise RuntimeError(f"GPT KV-head conversion rebound source tensor {name}")
        if not np.array_equal(tensor.data, source_state[name], equal_nan=True):
            raise RuntimeError(f"GPT KV-head conversion mutated source tensor {name}")
        if getattr(tensor, "_version", None) != source_versions[name]:
            raise RuntimeError(f"GPT KV-head conversion changed source version for {name}")
        original_grad, original_values = source_gradients[name]
        if tensor.grad is not original_grad:
            raise RuntimeError(f"GPT KV-head conversion rebound source gradient {name}")
        if original_grad is not None and not np.array_equal(
            original_grad, original_values, equal_nan=True
        ):
            raise RuntimeError(f"GPT KV-head conversion mutated source gradient {name}")
