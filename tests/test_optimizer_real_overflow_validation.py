"""Public optimizer real-valued options normalize binary64 conversion overflow."""

import pytest

from engine.optim import Adam, AdamW, SGD


HUGE = 10**400


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: SGD([], lr=HUGE), "lr must be finite"),
        (lambda: SGD([], momentum=HUGE), "momentum must be finite"),
        (lambda: SGD([], weight_decay=-HUGE), "weight_decay must be finite"),
        (lambda: Adam([], lr=-HUGE), "lr must be finite"),
        (lambda: Adam([], betas=(HUGE, 0.999)), "betas[0] must be finite"),
        (lambda: Adam([], betas=(0.9, -HUGE)), "betas[1] must be finite"),
        (lambda: Adam([], eps=HUGE), "eps must be finite"),
        (lambda: Adam([], weight_decay=HUGE), "weight_decay must be finite"),
        (lambda: AdamW([], lr=HUGE), "lr must be finite"),
    ],
)
def test_constructor_normalizes_real_conversion_overflow(factory, message):
    with pytest.raises(ValueError) as exc_info:
        factory()
    assert str(exc_info.value) == message


def test_sgd_state_overflow_is_transactional():
    optimizer = SGD([], lr=0.25, momentum=0.5, weight_decay=0.125)
    before = optimizer.state_dict()
    malformed = optimizer.state_dict()
    malformed["momentum"] = HUGE

    with pytest.raises(ValueError, match=r"^SGD momentum must be finite$"):
        optimizer.load_state_dict(malformed)

    assert optimizer.state_dict() == before


def test_adam_state_overflow_is_transactional():
    optimizer = Adam([], lr=0.25, betas=(0.5, 0.75), eps=0.125, weight_decay=0.0625)
    before = optimizer.state_dict()
    malformed = optimizer.state_dict()
    malformed["betas"] = (0.5, HUGE)

    with pytest.raises(ValueError, match=r"^Adam betas\[1\] must be finite$"):
        optimizer.load_state_dict(malformed)

    assert optimizer.state_dict() == before


def test_large_representable_values_keep_existing_validation_paths():
    optimizer = Adam([], lr=1e300, eps=1e300, weight_decay=1e300)
    assert optimizer.lr == 1e300
    assert optimizer.eps == 1e300
    assert optimizer.weight_decay == 1e300

    with pytest.raises(ValueError, match=r"^betas\[0\] must be less than 1.0$"):
        Adam([], betas=(1e300, 0.999))
