"""Regression tests for the public gradient-norm utility exports."""

import engine
from engine import clip_grad_norm_, global_grad_norm
from engine.grad_utils import (
    clip_grad_norm_ as implementation_clip_grad_norm_,
    global_grad_norm as implementation_global_grad_norm,
)


def test_gradient_norm_helpers_are_exported_from_engine():
    assert engine.global_grad_norm is implementation_global_grad_norm
    assert global_grad_norm is implementation_global_grad_norm
    assert engine.clip_grad_norm_ is implementation_clip_grad_norm_
    assert clip_grad_norm_ is implementation_clip_grad_norm_
