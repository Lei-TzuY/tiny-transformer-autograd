import numpy as np
import pytest

from nn.layers import Linear
from nn.transformer import FeedForward, GPT, SwiGLU, TransformerBlock


def _forbid_rng(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("invalid constructor consumed model-initialization RNG")

    monkeypatch.setattr(np.random, "uniform", fail)
    monkeypatch.setattr(np.random, "randn", fail)


def _tiny_gpt(**overrides):
    config = {
        "vocab_size": 8,
        "context_len": 4,
        "d_model": 4,
        "num_heads": 2,
        "d_ff": 8,
        "num_layers": 1,
        "dropout": 0.0,
    }
    config.update(overrides)
    return GPT(**config)


@pytest.mark.parametrize(
    ("factory", "error"),
    [
        (lambda: FeedForward(4, 8, dropout="bad"), TypeError),
        (lambda: SwiGLU(4, 8, dropout=np.nan), ValueError),
        (lambda: TransformerBlock(4, 2, True, dropout=0.0), TypeError),
        (lambda: _tiny_gpt(num_layers=np.float64(1.0)), TypeError),
        (lambda: _tiny_gpt(dropout=False), TypeError),
        (lambda: _tiny_gpt(lora_rank=1.5), TypeError),
        (lambda: _tiny_gpt(lora_alpha=np.nan), ValueError),
        (lambda: _tiny_gpt(grad_checkpoint="false"), TypeError),
    ],
)
def test_invalid_composite_hyperparameters_fail_before_rng(monkeypatch, factory, error):
    _forbid_rng(monkeypatch)
    with pytest.raises(error):
        factory()


def test_valid_numpy_scalar_hyperparameters_are_canonicalized():
    model = GPT(
        vocab_size=np.int64(8),
        context_len=np.int64(4),
        d_model=np.int64(4),
        num_heads=np.int64(2),
        d_ff=np.int64(8),
        num_layers=np.int64(1),
        dropout=np.float32(0.0),
        lora_rank=np.int64(0),
        lora_alpha=np.float32(1.0),
        grad_checkpoint=np.bool_(True),
    )

    assert type(model.vocab_size) is int
    assert type(model.context_len) is int
    assert type(model.d_model) is int
    assert type(model.num_heads) is int
    assert type(model.d_ff) is int
    assert type(model.num_layers) is int
    assert type(model.dropout) is float
    assert type(model.lora_alpha) is float
    assert type(model.grad_checkpoint) is bool
    assert model.grad_checkpoint is True


def _snapshot_model(model):
    snapshots = []
    for name, tensor in model.named_tensors():
        snapshots.append(
            (
                name,
                tensor,
                tensor.data.copy(),
                tensor.requires_grad,
                None if tensor.grad is None else tensor.grad.copy(),
            )
        )
    return snapshots


def _assert_model_snapshot_unchanged(model, snapshots):
    current = list(model.named_tensors())
    assert [name for name, _ in current] == [item[0] for item in snapshots]
    for (name, tensor), (_, original, data, requires_grad, grad) in zip(
        current, snapshots
    ):
        assert tensor is original, name
        np.testing.assert_array_equal(tensor.data, data)
        assert tensor.requires_grad is requires_grad
        if grad is None:
            assert tensor.grad is None
        else:
            np.testing.assert_array_equal(tensor.grad, grad)

    for module in model.modules():
        if isinstance(module, Linear):
            assert module.lora_A is None
            assert module.lora_B is None


@pytest.mark.parametrize(
    ("rank", "alpha", "error"),
    [
        (1.5, 1.0, TypeError),
        (True, 1.0, TypeError),
        (2, np.nan, ValueError),
        (2, "bad", TypeError),
    ],
)
def test_invalid_enable_lora_is_transactional(rank, alpha, error):
    model = _tiny_gpt()
    for _, tensor in model.named_tensors():
        if tensor.requires_grad:
            tensor.grad = np.full_like(tensor.data, 0.25)

    before = _snapshot_model(model)
    before_rank = model.lora_rank
    before_alpha = model.lora_alpha

    with pytest.raises(error):
        model.enable_lora(rank, alpha)

    assert model.lora_rank == before_rank
    assert model.lora_alpha == before_alpha
    _assert_model_snapshot_unchanged(model, before)


def test_enable_lora_accepts_numpy_scalars_and_normalizes_state():
    model = _tiny_gpt()

    returned = model.enable_lora(np.int64(2), np.float32(4.0))

    assert returned is model
    assert type(model.lora_rank) is int
    assert model.lora_rank == 2
    assert type(model.lora_alpha) is float
    assert model.lora_alpha == 4.0
    assert any(
        isinstance(module, Linear) and module.lora_A is not None
        for module in model.modules()
    )
