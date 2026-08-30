"""Checkpoint-level GPT migration between MHA, GQA, and MQA layouts."""

from collections.abc import Mapping

import numpy as np

from engine.checkpoint import _validate_checkpoint_envelope
from engine.optim import Adam, AdamW, SGD

from .attention_conversion import convert_gpt_kv_heads, _positive_int
from .transformer import GPT


_MIGRATION_KEY = "_tiny_transformer_migrations"
_SUPPORTED_OPTIMIZERS = {
    "SGD": SGD,
    "Adam": Adam,
    "AdamW": AdamW,
}


def convert_gpt_checkpoint_kv_heads(checkpoint, num_kv_heads):
    """Return an independent checkpoint migrated to ``num_kv_heads``.

    The checkpoint must contain ``metadata['model_config']`` so the source GPT
    architecture can be reconstructed before its model state is validated and
    converted with :func:`convert_gpt_kv_heads`.

    When the KV-head count actually changes, saved optimizer moments cannot in
    general be transformed exactly. In particular, a grouped/shared parameter's
    future gradient is a sum of per-query-head contributions, while Adam-style
    second-moment state stores only per-coordinate squared-gradient history and
    therefore lacks the cross terms required for an exact reparameterization.
    For the built-in SGD/Adam/AdamW optimizers this helper consequently preserves
    optimizer type and scalar hyperparameters but resets parameter-local state to
    fresh zero buffers with the target model's shapes. The scheduler, checkpoint
    step, and NumPy RNG state remain unchanged, so training can intentionally
    continue from the same schedule point with fresh optimizer moments.

    A same-head conversion is an exact checkpoint clone: optimizer state is
    preserved rather than reset. Every returned NumPy array is detached from the
    input checkpoint, and the process-global NumPy RNG state is restored on both
    success and failure.
    """
    num_kv_heads = _positive_int("num_kv_heads", num_kv_heads)
    state = _snapshot_checkpoint(checkpoint)
    _validate_checkpoint_envelope(state)

    metadata = state.get("metadata", {})
    if not isinstance(metadata, Mapping):
        # The envelope normally catches this, but keep the public error local if
        # checkpoint validation changes in the future.
        raise TypeError("checkpoint metadata must be a mapping")
    metadata = _snapshot_mapping(metadata, path="checkpoint metadata")
    model_config = metadata.get("model_config")
    if not isinstance(model_config, Mapping):
        raise ValueError("checkpoint metadata must contain a model_config mapping")
    model_config = _snapshot_mapping(model_config, path="model_config")

    rng_before = np.random.get_state()
    try:
        source_model = GPT(**model_config)
        source_model.load_state_dict(state["model"], strict=True)
        source_kv_heads = _positive_int(
            "checkpoint model num_kv_heads", source_model.num_kv_heads
        )

        target_model = convert_gpt_kv_heads(source_model, num_kv_heads)
        changed = source_kv_heads != num_kv_heads

        result = _snapshot_checkpoint(state)
        result["model"] = target_model.state_dict()
        metadata["model_config"] = dict(target_model.config())

        if changed:
            result["optimizer"] = _fresh_optimizer_state(
                state.get("optimizer_type"),
                state.get("optimizer"),
                source_model,
                target_model,
            )
            _append_migration_record(
                metadata,
                source_kv_heads=source_kv_heads,
                target_kv_heads=num_kv_heads,
                optimizer_present=state.get("optimizer") is not None,
            )

        result["metadata"] = metadata
        _validate_checkpoint_envelope(result)
        return result
    finally:
        np.random.set_state(rng_before)


def convert_gpt_checkpoint_file(
    source_path,
    destination_path,
    num_kv_heads,
    *,
    source_format="pickle",
    destination_format=None,
):
    """Atomically migrate a checkpoint file and return the converted state.

    ``source_format`` and ``destination_format`` are ``'pickle'`` or ``'safe'``.
    The destination defaults to the source format. Reading pickle checkpoints is
    executable and must therefore be restricted to trusted local files, exactly
    like :func:`engine.checkpoint.read_checkpoint`.

    The destination writer is the same durable atomic writer used by the normal
    checkpoint codecs. Source and destination may be the same path, allowing an
    in-place architecture migration after the full source state has been read and
    validated.
    """
    source_format = _checkpoint_format("source_format", source_format)
    if destination_format is None:
        destination_format = source_format
    else:
        destination_format = _checkpoint_format(
            "destination_format", destination_format
        )

    if source_format == "pickle":
        from engine.checkpoint import read_checkpoint

        state = read_checkpoint(source_path)
    else:
        from engine.safe_checkpoint import read_safe_checkpoint

        state = read_safe_checkpoint(source_path)

    converted = convert_gpt_checkpoint_kv_heads(state, num_kv_heads)

    if destination_format == "pickle":
        from engine.checkpoint import _atomic_pickle_dump

        _atomic_pickle_dump(destination_path, converted)
    else:
        from engine.safe_checkpoint import _write_safe_state

        _write_safe_state(destination_path, converted)
    return converted


def _fresh_optimizer_state(optimizer_type, saved_state, source_model, target_model):
    if saved_state is None:
        return None
    optimizer_class = _SUPPORTED_OPTIMIZERS.get(optimizer_type)
    if optimizer_class is None:
        raise ValueError(
            "cannot safely reset unsupported checkpoint optimizer type "
            f"{optimizer_type!r} after KV-head conversion"
        )

    source_optimizer = optimizer_class(source_model.parameters())
    source_optimizer.load_state_dict(saved_state)

    if optimizer_class is SGD:
        target_optimizer = SGD(
            target_model.parameters(),
            lr=source_optimizer.lr,
            momentum=source_optimizer.momentum,
            weight_decay=source_optimizer.weight_decay,
        )
    else:
        target_optimizer = optimizer_class(
            target_model.parameters(),
            lr=source_optimizer.lr,
            betas=(source_optimizer.beta1, source_optimizer.beta2),
            eps=source_optimizer.eps,
            weight_decay=source_optimizer.weight_decay,
        )
    return target_optimizer.state_dict()


def _append_migration_record(
    metadata,
    *,
    source_kv_heads,
    target_kv_heads,
    optimizer_present,
):
    existing = metadata.get(_MIGRATION_KEY)
    if existing is None:
        history = []
    elif isinstance(existing, list):
        history = _snapshot_value(existing, path=_MIGRATION_KEY, active=set())
    else:
        raise TypeError(f"checkpoint metadata {_MIGRATION_KEY!r} must be a list")

    history.append(
        {
            "kind": "gpt_kv_heads",
            "source_num_kv_heads": int(source_kv_heads),
            "target_num_kv_heads": int(target_kv_heads),
            "optimizer_state": "reset" if optimizer_present else "absent",
        }
    )
    metadata[_MIGRATION_KEY] = history


def _checkpoint_format(name, value):
    if not isinstance(value, str):
        raise TypeError(f"{name} must be 'pickle' or 'safe'")
    value = value.lower()
    if value not in {"pickle", "safe"}:
        raise ValueError(f"{name} must be 'pickle' or 'safe'")
    return value


def _snapshot_checkpoint(checkpoint):
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a mapping")
    return _snapshot_mapping(checkpoint, path="checkpoint")


def _snapshot_mapping(mapping, *, path):
    active = set()
    return _snapshot_value(mapping, path=path, active=active)


def _snapshot_value(value, *, path, active):
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True, subok=False)
    if isinstance(value, np.generic):
        return value.copy()
    if isinstance(value, Mapping):
        marker = id(value)
        if marker in active:
            raise ValueError(f"cyclic mapping is not supported at {path}")
        active.add(marker)
        try:
            items = tuple(value.items())
            return {
                key: _snapshot_value(
                    item,
                    path=f"{path}.{key}",
                    active=active,
                )
                for key, item in items
            }
        finally:
            active.remove(marker)
    if isinstance(value, list):
        marker = id(value)
        if marker in active:
            raise ValueError(f"cyclic list is not supported at {path}")
        active.add(marker)
        try:
            return [
                _snapshot_value(item, path=f"{path}[{index}]", active=active)
                for index, item in enumerate(value)
            ]
        finally:
            active.remove(marker)
    if isinstance(value, tuple):
        marker = id(value)
        if marker in active:
            raise ValueError(f"cyclic tuple is not supported at {path}")
        active.add(marker)
        try:
            return tuple(
                _snapshot_value(item, path=f"{path}[{index}]", active=active)
                for index, item in enumerate(value)
            )
        finally:
            active.remove(marker)

    # Checkpoint metadata is intentionally extensible. Immutable/opaque trusted
    # pickle metadata is preserved by reference rather than invoking arbitrary
    # user-defined deepcopy hooks. All built-in checkpoint arrays/containers are
    # detached above; safe-checkpoint output will independently reject any value
    # its non-executable format cannot encode.
    return value
