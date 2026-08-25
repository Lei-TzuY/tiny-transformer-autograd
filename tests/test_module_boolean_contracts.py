"""Regression tests for explicit module/layer boolean contracts."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nn.layers import Dropout, Linear
from nn.module import Module


@pytest.mark.parametrize(
    "bad_bias",
    [None, 0, 1, np.int64(1), "yes", [], object()],
)
def test_linear_rejects_non_boolean_bias_before_rng(monkeypatch, bad_bias):
    calls = []

    def unexpected_uniform(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("weight initialization must not run")

    monkeypatch.setattr(np.random, "uniform", unexpected_uniform)

    with pytest.raises(TypeError, match="bias must be boolean"):
        Linear(3, 2, bias=bad_bias)

    assert calls == []


@pytest.mark.parametrize(
    ("bias", "has_bias"),
    [
        (True, True),
        (False, False),
        (np.bool_(True), True),
        (np.bool_(False), False),
    ],
)
def test_linear_accepts_boolean_bias_scalars(bias, has_bias):
    layer = Linear(3, 2, bias=bias)

    assert (layer.bias is not None) is has_bias
    assert ("bias" in dict(layer.named_parameters())) is has_bias


@pytest.mark.parametrize(
    "bad_mode",
    [None, 0, 1, np.int64(0), "eval", [], object()],
)
def test_train_rejects_non_boolean_mode_without_partial_mutation(bad_mode):
    root = Module()
    child = Dropout(0.5)
    sibling = Dropout(0.25)
    root.child = child
    root.group = {"sibling": sibling}
    root.loop = root

    root.training = False
    child.training = True
    sibling.training = False
    before = (root.training, child.training, sibling.training)

    with pytest.raises(TypeError, match="training mode must be boolean"):
        root.train(bad_mode)

    assert (root.training, child.training, sibling.training) == before


def test_train_normalizes_numpy_boolean_recursively_and_cycle_safely():
    root = Module()
    child = Dropout(0.5)
    sibling = Dropout(0.25)
    root.child = child
    root.group = [sibling]
    root.loop = root

    result = root.train(np.bool_(False))

    assert result is root
    for module in (root, child, sibling):
        assert module.training is False
        assert type(module.training) is bool

    root.train(np.bool_(True))
    for module in (root, child, sibling):
        assert module.training is True
        assert type(module.training) is bool


def test_eval_still_routes_through_validated_train_contract():
    root = Module()
    child = Dropout(0.5)
    root.child = child

    result = root.eval()

    assert result is root
    assert root.training is False
    assert child.training is False
