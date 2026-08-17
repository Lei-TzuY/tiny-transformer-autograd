# CLAUDE.md

Guidance for working in this repository. Read `PROJECT_STATE.md` for the full state
handoff (completed rounds, design rationale, next-round candidates).

## What this is

A pure-NumPy teaching project: a reverse-mode autograd engine plus a decoder-only
Transformer that trains and generates text. No PyTorch, no GPU, no C extensions —
NumPy is the only runtime dependency. Every addition must stay readable as a
*teaching* implementation; correctness and clarity outrank features.

## Commands

```bash
python -m pytest -q -W error          # full suite — must stay green (380 tests, ~1s)
python -m pytest tests/test_data.py -q # one module
python -m compileall -q src tests plot_loss.py
python src/train.py --iters 20 --no-sample          # smoke-train
pip install -e ".[dev]" && tiny-train --help        # installed entry point
```

Pytest warnings are errors in CI and locally (`-W error`). A NumPy overflow/divide
warning is a test failure, so stable formulations are mandatory, not stylistic.

## Layout

```
src/engine/    tensor.py  ops.py  grad_mode.py  recompute.py
               optim.py  scheduler.py  checkpoint.py
src/nn/        module.py  layers.py  attention.py  transformer.py
src/           train.py (CLI)  tokenizer.py  benchmark.py
tests/         9 modules, 380 tests
```

Tests add `src` to `sys.path` themselves; scripts run from the repo root cannot
`import engine` without doing the same.

## Non-negotiable invariants

Break any of these and the project is silently wrong. Each has dedicated tests.

1. **Every op result is built by `Tensor.__init__`.** That constructor is the single
   place recording is gated (`is_grad_enabled()`), so a new op must construct its
   result there rather than mutating an existing tensor. Never bypass it.
2. **`no_grad()` suppresses op results, never explicit leaves.** A model built inside
   `no_grad()` must still be trainable. Suppressed results retain creation provenance,
   so delayed `backward()` misuse still raises after the scope exits.
3. **A backward closure is stored only on a node that can hold a gradient.** The
   `_backward` property setter enforces this; it is also what prevents a gradient-less
   node from being asked to split a `None` gradient. Such op results also drop parents.
4. **A fully masked (`all -inf`) softmax row is zero weights, not NaN.** Defined
   identically in `ops.softmax` and `nn/attention._softmax` — the graph and NumPy
   inference paths must agree. Rounds 3 and 4 depend on this.
5. **`cross_entropy` raises on a scored row with no finite logit.** Zero weights are
   meaningful for attention; a loss over an impossible distribution is not. The check
   applies to *scored* rows only — an `ignore_index` row may hold anything.
6. **`cross_entropy(ignore_index=…)` divides by the scored count** and leaves exactly
   zero gradient at ignored positions.
7. **Causal masks use true `-inf`**, never `-1e9`. A finite bias can be overcome by a
   large enough logit. `GPT.load_state_dict` rebuilds the buffer after every load so a
   legacy checkpoint cannot reintroduce a finite mask.
8. **`forward(mask=None)` on `SelfAttention`/`MultiHeadAttention` is causal**, matching
   `infer()`. A parity test pins them together.
9. **Training masks are right-padded; generation masks are left-padded.** Enforced at
   the call sites (`_key_padding_bias`, `_validate_generation_mask`), not assumed.
10. **`recompute` replays from detached copies of its inputs** and captures/restores the
    NumPy RNG state. Differentiating into the originals discards a residual's already
    accumulated gradient; replaying without the RNG state differentiates a different
    function than the forward computed.
11. **`grad_checkpoint` stays out of `GPT.config()`.** It changes memory and time,
    never weights, outputs, or gradients.
12. **Checkpoint restore is transactional.** A malformed later section must leave the
    caller's model, optimizer, scheduler, and RNG state untouched.
13. **The default `text` (token-stream) training path is trajectory-stable.** Refactors
    must consume the RNG in the same order; verify by re-running a seeded CLI command
    and comparing losses and gradient norms step for step.
14. **A generation window crop renumbers positions from 0.** `generate` recomputes
    per-row positions from the *cropped* mask on every re-prefill, never carrying
    absolute numbering across a crop. Carrying it over pushes `position_ids` past
    `context_len` — `_validate_position_ids` catches that, so the failure is loud
    rather than a silent position drift.
15. **A 3-D multi-head additive mask is batch-major `(B,Q,K)`.** It is normalized to
    `(B,1,Q,K)` before broadcasting; per-head masks must be explicit 4-D tensors.
    Graph forward and NumPy inference share the same value/shape contract.
16. **Caller-provided KV caches are validated before their past length is used.** Every
    layer must carry matching rank-4, finite, real-valued `k`/`v` arrays with the
    model's batch, head count, head width, and one common past length.

## Conventions

- Tensors are `float64` everywhere. There is no dtype system and no mixed precision.
- Module state travels through `state_dict()`/`load_state_dict()`; `named_parameters()`
  drives optimizers. New submodules must be reachable from `modules()`.
- Public API validation raises `ValueError`/`TypeError` with a message naming the
  offending argument — never let a caller mistake surface as an opaque NumPy error.
- Line length ≤ 95 characters.
- Comments explain *why* (a convention, a trap, a numerical choice), not *what*.
- Docstrings at module top describe the contract the tests pin.

## Verification style expected here

Claims in this repo are measured, not asserted. When adding behavior:

1. Probe the current behavior first and record the number.
2. Assert the strongest true relation — bitwise equality where it holds (padded vs.
   unpadded logits differ by exactly `0.0`), `1e-12`–`1e-14` where BLAS rounding is
   involved.
3. Add a counter-test so the equivalence cannot pass for the wrong reason (e.g. show
   the logits *do* change when the mask is removed).

## Planning artifacts

`task_plan.md` (phases, decisions, acceptance criteria), `findings.md` (per-round
findings and rationale), `progress.md` (verification log, test results, error log),
`PROJECT_STATE.md` (handoff snapshot). Keep them current as work lands — they are the
memory across sessions. Log every command/test failure in `progress.md` and use a
different approach on retry.

## Gotchas

- The worktree carries a large pre-existing user diff (Llama/RoPE/SwiGLU/AdamW work)
  on top of a single `Initial commit`. Inspect diffs before editing; preserve it.
- Windows console rendering garbles non-ASCII in file output. Anchor edits on ASCII
  context only — the files themselves are valid UTF-8.
- `git diff --check` reports pre-existing LF/CRLF notices; those are expected noise.
- Checkpoints are pickles: only load trusted files. `plot_loss.py` is a
  source-checkout utility and is not part of the installed package.
- Grad-mode decorators support ordinary synchronous functions only; coroutine and
  generator targets fail explicitly instead of silently recording a graph.
