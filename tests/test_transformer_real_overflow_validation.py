import numpy as np
import pytest

from nn.layers import Linear
from nn.transformer import FeedForward, GPT, SwiGLU, TransformerBlock


_HUGE = 10**400


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


def _rng_state():
    state = np.random.get_state()
    return state[0], state[1].copy(), state[2], state[3], state[4]


def _assert_rng_state_equal(left, right):
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: FeedForward(4, 8, dropout=_HUGE), "dropout must be finite"),
        (lambda: SwiGLU(4, 8, dropout=-_HUGE), "dropout must be finite"),
        (
            lambda: TransformerBlock(4, 2, 8, dropout=_HUGE),
            "dropout must be finite",
        ),
        (lambda: _tiny_gpt(dropout=-_HUGE), "dropout must be finite"),
        (lambda: _tiny_gpt(lora_alpha=_HUGE), "lora_alpha must be finite"),
        (lambda: _tiny_gpt(lora_alpha=-_HUGE), "lora_alpha must be finite"),
    ],
)
def test_constructor_conversion_overflow_is_finite_error_without_rng(factory, match):
    np.random.seed(12345)
    before = _rng_state()

    with pytest.raises(ValueError, match=match):
        factory()

    _assert_rng_state_equal(before, _rng_state())


def _snapshot_model(model):
    return [
        (
            name,
            tensor,
            tensor.data.copy(),
            tensor.requires_grad,
            None if tensor.grad is None else tensor.grad.copy(),
        )
        for name, tensor in model.named_tensors()
    ]


def _assert_model_unchanged(model, snapshot):
    current = list(model.named_tensors())
    assert [name for name, _ in current] == [item[0] for item in snapshot]
    for (name, tensor), (_, original, data, requires_grad, grad) in zip(
        current, snapshot
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


@pytest.mark.parametrize("alpha", [_HUGE, -_HUGE])
def test_enable_lora_overflow_is_transactional_and_preserves_rng(alpha):
    model = _tiny_gpt()
    for _, tensor in model.named_tensors():
        if tensor.requires_grad:
            tensor.grad = np.full_like(tensor.data, 0.25)

    snapshot = _snapshot_model(model)
    before_rank = model.lora_rank
    before_alpha = model.lora_alpha
    np.random.seed(9876)
    before_rng = _rng_state()

    with pytest.raises(ValueError, match="LoRA alpha must be finite"):
        model.enable_lora(2, alpha)

    assert model.lora_rank == before_rank
    assert model.lora_alpha == before_alpha
    _assert_model_unchanged(model, snapshot)
    _assert_rng_state_equal(before_rng, _rng_state())


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"temperature": _HUGE}, "temperature must be finite"),
        ({"temperature": -_HUGE}, "temperature must be finite"),
        ({"top_p": _HUGE}, "top_p must be finite"),
        ({"top_p": -_HUGE}, "top_p must be finite"),
    ],
)
def test_generate_overflow_fails_before_token_validation_inference_or_rng(
    monkeypatch, overrides, match
):
    model = _tiny_gpt()

    def forbidden(*args, **kwargs):
        pytest.fail("generation reached model input handling before option validation")

    monkeypatch.setattr(model, "_validate_token_batch", forbidden)
    monkeypatch.setattr(model, "infer", forbidden)
    options = {
        "max_new_tokens": 1,
        "temperature": 1.0,
        "top_k": None,
        "top_p": None,
        "strategy": "sample",
        "beam_width": 2,
        "use_cache": True,
    }
    options.update(overrides)

    np.random.seed(2468)
    before = _rng_state()
    with pytest.raises(ValueError, match=match):
        model.generate([[1, 2]], **options)
    _assert_rng_state_equal(before, _rng_state())


@pytest.mark.parametrize("temperature", [_HUGE, -_HUGE])
def test_direct_generate_beam_overflow_fails_before_token_validation(
    monkeypatch, temperature
):
    model = _tiny_gpt()

    def forbidden(*args, **kwargs):
        pytest.fail("beam search reached token validation before temperature validation")

    monkeypatch.setattr(model, "_validate_token_batch", forbidden)
    monkeypatch.setattr(model, "infer", forbidden)

    with pytest.raises(ValueError, match="temperature must be finite"):
        model.generate_beam([[1, 2]], 1, beam_width=2, temperature=temperature)


def test_large_representable_values_keep_existing_semantics():
    model = _tiny_gpt(lora_alpha=1e300)
    assert model.lora_alpha == 1e300

    with pytest.raises(ValueError, match=r"dropout must be in \[0, 1\)"):
        _tiny_gpt(dropout=1e300)

    prompt = np.array([[1, 2]], dtype=np.int64)
    generated = model.generate(
        prompt,
        max_new_tokens=0,
        temperature=1e300,
        strategy="greedy",
    )
    np.testing.assert_array_equal(generated, prompt)
    assert not np.shares_memory(generated, prompt)

    beamed = model.generate_beam(
        prompt,
        max_new_tokens=0,
        beam_width=2,
        temperature=1e300,
    )
    np.testing.assert_array_equal(beamed, prompt)

    with pytest.raises(ValueError, match=r"top_p must be in \(0, 1\]"):
        model.generate(prompt, max_new_tokens=0, top_p=1e300)
