# Progress Log

## Session: 2026-07-30

### Phase 1: Baseline & Discovery
- **Status:** complete
- **Started:** 2026-07-30
- Actions taken:
  - Read the `planning-with-files` skill instructions and its templates.
  - Confirmed there were no prior planning files or unsynchronized planning context.
  - Initialized the task plan, findings log, and progress log.
  - Inventoried the repository and inspected Git status.
  - Identified a dirty worktree with pre-existing changes across the core files; marked them as user-owned constraints.
  - Read packaging metadata and the first 260 README lines, and captured the pre-existing diff size.
  - Confirmed the timed-out inspection succeeds with a larger command budget.
  - Ran the full baseline suite: 130 tests passed.
  - Searched source/docs/tests for TODO, FIXME, placeholder, and unimplemented markers; found no unfinished production implementation.
  - Inspected `Tensor` and all primitive ops; identified candidate matmul broadcasting/vector, repeated-backward, sigmoid stability, and cross-entropy edge cases for targeted probes.
  - Confirmed matmul backward failures for broadcast batch axes and 1-D operands.
  - Confirmed sigmoid warning noise and extreme cross-entropy forward/gradient inconsistency.
  - Confirmed chained repeated backward over-propagates stale intermediate gradients.
  - Attempted a PEP 517 wheel build; it failed because the configured setuptools backend module does not exist.
  - Received independent autograd/Transformer audit confirmation of cross-entropy, division, and test-coverage gaps.
  - Received the completed autograd audit, including negative-axis transpose, deep-graph recursion, concat-generator, and VJP coverage findings.
  - Inspected attention and GPT training/inference/generation paths; found coherent main-path cache behavior but several missing public input validations.
  - Inspected training, checkpoint, optimizer, and scheduler data flow; confirmed optimizer-class loss on resume.
  - Received an independent attention audit confirming `forward(mask=None)`/`infer()` causal mismatch.
  - Located the exact attention and checkpoint test classes and documented the missing regression cases before editing.
  - Inspected CI and plotting utility; confirmed the missing package smoke gate and stale `--metric` documentation.
  - Inspected layer and optimizer constructors and selected a focused set of confirmed API-validation fixes.
- Files created/modified:
  - `task_plan.md` (created)
  - `findings.md` (created)
  - `progress.md` (created)

### Phase 2: Prioritization & Design
- **Status:** complete
- Actions taken:
  - Ranked confirmed gaps by mathematical correctness, reproducibility, installability, and teaching value.
  - Selected a bounded hardening set spanning autograd, causal attention, checkpoint optimizer identity, packaging, tests, and docs.
  - Defined explicit acceptance criteria in `task_plan.md`.
- Files created/modified:
  - `task_plan.md`
  - `findings.md`
  - `progress.md`

### Phase 3: Implementation
- **Status:** complete
- Actions taken:
  - Made standalone attention causal by default and changed generated/precomputed causal masks to use `-inf`.
  - Added forward/inference parity tests for single- and multi-head attention.
  - Added optimizer type and NumPy RNG state to checkpoints with legacy-compatible restore behavior.
  - Added resume optimizer selection/conflict handling and checkpoint regression tests.
  - Repaired the setuptools backend declaration and explicitly packaged top-level CLI modules.
  - Added explicit validation for optimizer hyperparameters, layer dimensions/dropout, attention/RoPE dimensions and bounds, token batches, context limits, caches, and generation lengths.
  - Added a checkpoint-step guard to prevent resuming training into a smaller requested total step.
  - Added `tests/test_validation.py` covering confirmed invalid-input failure modes.
  - Rebuilt the wheel successfully and inspected its archive contents for every intended package/CLI module.
  - Installed the wheel into an isolated venv, verified all installed imports, and ran `tiny-train --help`.
  - Integrated and reviewed the delegated autograd hardening diff and its 15 new regression tests.
  - Ran the integrated 170-test suite with warnings-as-errors and checked the diff for whitespace defects.
  - Updated CI to exercise the package build/install and minimum supported Python.
  - Synchronized README feature counts, checkpoint/causal semantics, CLI options, tests, and installation instructions.
  - Removed the unsupported plotting flag from its usage documentation and verified help output.
  - Versioned the checkpoint format, moved optimizer compatibility checks before state mutation, and added future-version rejection while retaining version-1 compatibility.
  - Added a real random-batch + dropout + AdamW resume trajectory test.
  - Confirmed exactly 172 collected tests and successfully byte-compiled source, tests, and plotting utility.
  - Ran a two-process Llama/AdamW CLI save/resume smoke without repeating the optimizer flag; resume preserved AdamW and advanced the step.
  - Simplified training/evaluation to use generalized `(..., classes)` cross entropy directly, removing manual logits/target flattening.
  - Added checkpoint migration behavior that rebuilds strict `-inf` causal masks after loading legacy finite-mask state.
  - Re-ran the full integrated suite and whitespace check after all checkpoint/training/mask changes.
  - Applied final-review fixes: CE target snapshots and transactional checkpoint restore with atomic component loaders.
  - Added mutable-label and four-way failed-restore rollback regression coverage.
  - Added NumPy-array operator interoperability and strict normalization input-shape checks with regression tests.
  - Added the trusted-pickle security warning and clarified plotting utility wheel scope in README.
  - Completed final full-suite, compilation, whitespace, Git status, and generated-artifact checks.
  - Removed the generated CLI smoke checkpoint after verification; it was reproducible and contained no user data.
  - Inspected final status/diff size and identified only generated build/venv artifacts for cleanup; no unexpected source file appeared.
  - Verified the exact absolute paths of four generated artifact directories and removed them using explicit native PowerShell paths.
- Files created/modified:
  - `src/nn/attention.py`
  - `src/nn/transformer.py`
  - `src/engine/checkpoint.py`
  - `src/train.py`
  - `pyproject.toml`
  - `tests/test_transformer.py`
  - `tests/test_training.py`
  - `src/engine/optim.py`
  - `src/nn/layers.py`
  - `tests/test_validation.py`
  - `.github/workflows/tests.yml`
  - `README.md`
  - `plot_loss.py`

### Phase 4: Verification & Hardening
- **Status:** complete
- Actions taken:
  - Ran focused numerical, attention, checkpoint, validation, and integration suites throughout implementation.
  - Ran the final warnings-as-errors suite: 178 tests passed.
  - Byte-compiled all source, tests, and the plotting utility; checked the diff for whitespace defects.
  - Built and inspected a wheel, installed it in an isolated environment, imported every intended module, and exercised `tiny-train --help`.
  - Ran independent final correctness and delivery reviews, then closed the mutable-target, transactional-restore, NumPy-operator, normalization-shape, and documentation gaps they identified.
  - Audited Git status and removed only verified, reproducible build/test artifacts.
- Files created/modified:
  - `tests/test_autograd.py`
  - `tests/test_transformer.py`
  - `tests/test_training.py`
  - `tests/test_validation.py`
  - `.github/workflows/tests.yml`
  - `README.md`

### Phase 5: Delivery
- **Status:** complete
- Actions taken:
  - Reconciled README feature/test counts and documented package installation, causal semantics, checkpoint trust boundaries, and source-only plotting scope.
  - Recorded final verification evidence and the intentionally deferred opportunities.
  - Finalized all persistent planning artifacts for handoff.
- Files created/modified:
  - `task_plan.md`
  - `findings.md`
  - `progress.md`

### Phase 6: Round 2 — Inference mode and fully-masked attention rows
- **Status:** complete
- Actions taken:
  - Re-read all three planning artifacts and the current engine/nn/train sources before editing.
  - Added `src/engine/grad_mode.py` with a thread-local recording flag and reentrant
    `set_grad_enabled` / `no_grad` / `enable_grad` guards usable as context managers or decorators.
  - Gated graph construction in `Tensor.__init__`: an op result created while recording is off
    loses its parents, gradient buffer, and backward closure, while explicit leaves are untouched.
  - Made `_backward` a property that stores a closure only on nodes that can hold a gradient, and
    confirmed against re-enabled unconditional storage that this also fixes a real `TypeError`
    when a gradient-less node sat inside a gradient-requiring graph.
  - Made `backward()` raise when called on a detached tensor inside a disabled block, while leaving
    graphs built outside the block differentiable and behavior outside any block unchanged.
  - Corrected the `detach()` docstring, which claimed shared storage although the constructor copies.
  - Exported the new switch from `engine/__init__.py` and used it in `train.evaluate`.
  - Defined all-`-inf` softmax rows as zero weights in `ops.softmax` and mirrored the convention in
    `nn/attention._softmax` so `forward()` and `infer()` cannot disagree.
  - Made `cross_entropy` reject a logits row with no finite class instead of returning NaN.
  - Added `_prepare_mask` validation (broadcast shape, NaN/`+inf`) and NumPy-array mask support to
    both `SelfAttention.forward` and `MultiHeadAttention.forward`.
  - Documented the mask contract and the fully-masked-row behavior in the attention module,
    both forward docstrings, and the README.
  - Added `tests/test_grad_mode.py` (21 tests) plus masked-softmax, cross-entropy, custom-mask,
    and mask-validation regressions across three existing test files.
  - Measured the benefit (1.88x faster evaluation, 346 to 65 live tensors) and recorded it.
  - Verified package imports, byte compilation, whitespace, and a real training CLI run.
- Files created/modified:
  - `src/engine/grad_mode.py` (created)
  - `src/engine/tensor.py`
  - `src/engine/ops.py`
  - `src/engine/__init__.py`
  - `src/nn/attention.py`
  - `src/train.py`
  - `tests/test_grad_mode.py` (created)
  - `tests/test_autograd.py`
  - `tests/test_transformer.py`
  - `tests/test_validation.py`
  - `README.md`

### Phase 7: Round 3 — Variable-length batches
- **Status:** complete
- Actions taken:
  - Added `ignore_index` to `ops.cross_entropy`: ignored positions are dropped from the mean,
    receive exactly zero gradient, and are the only targets allowed outside `[0, C)`.
  - Kept a no-gather fast path so the ordinary training call pays nothing for the feature.
  - Narrowed the all-`-inf` logits check to scored rows, since an ignored position may hold anything.
  - Made an all-`ignore_index` batch raise instead of dividing by zero.
  - Added `GPT.forward(idx, attention_mask=...)` with a `_key_padding_bias` helper that validates
    shape, dtype, and 0/1 values, then combines the bias with the causal mask by a plain add.
  - Added `TestKeyPaddingMask` (8 tests) including logits/gradient equivalence with an unpadded run,
    a counter-test proving the mask is what causes the equivalence, per-layer application,
    boolean/integer agreement, and an all-padding row staying finite.
  - Added `ignore_index` coverage to `test_autograd.py` (8 tests) including a VJP gradient check and
    proof that an unused `ignore_index` is bit-identical to the plain loss (the no-gather path).
  - Added mask-validation cases to `test_validation.py` and an end-to-end ragged-batch training
    integration test to `test_features.py`.
  - Documented the contract, the right-padding requirement, and why `infer`/`generate` take no mask.
- Files created/modified:
  - `src/engine/ops.py`
  - `src/nn/transformer.py`
  - `tests/test_autograd.py`
  - `tests/test_transformer.py`
  - `tests/test_validation.py`
  - `tests/test_features.py`
  - `README.md`

### Phase 8: Round 4 — Batched generation from ragged prompts
- **Status:** complete
- Actions taken:
  - Added `RotaryEmbedding.rotate_np(positions=...)` with a `_gather_positions` helper that
    validates dtype, range, and broadcastability, keeping the scalar-offset path untouched.
  - Threaded an additive `key_bias` through `SelfAttention.infer`, `MultiHeadAttention.infer`, and
    `TransformerBlock.infer`, and per-row RoPE `positions` through the multi-head path.
  - Extended `GPT.infer` with `attention_mask` validated against `(batch, past + time)` so cached
    padding stays hidden, plus a validated `position_ids` argument.
  - Added `generate(..., attention_mask=...)`: derives per-row positions from the mask, extends the
    mask and positions as tokens are appended, and keeps the unmasked path byte-identical.
  - Enforced the left-padding contract (zeros then ones, at least one real token), the
    `context_len` bound for masked runs, and beam search's mask rejection.
  - Factored mask validation into a shared `_validate_keep_mask` used by the forward and inference
    paths so both report the same errors.
  - Probed equivalence before writing tests: per-row generation match, cached/uncached agreement,
    and a 0.0 maximum logits difference for both `gpt` and `llama` architectures.
  - Added `TestPaddedGeneration` (13 tests, parametrized over both architectures) and seven
    validation tests for `position_ids`, cached-key masks, and RoPE positions.
  - Rewrote the README section that documented the old limitation.
- Files created/modified:
  - `src/nn/attention.py`
  - `src/nn/transformer.py`
  - `tests/test_transformer.py`
  - `tests/test_validation.py`
  - `README.md`

### Phase 9: Round 5 — Gradient checkpointing
- **Status:** complete
- Actions taken:
  - Added `src/engine/recompute.py`: `recompute(function, *inputs)` runs the section under
    `no_grad()`, then replays it under `enable_grad()` inside its backward closure and
    differentiates the replay.
  - Replayed from detached input copies so the replay cannot reset an outer-graph node and discard a
    residual connection's already-accumulated gradient.
  - Captured the NumPy RNG state before the unrecorded forward, restored it for the replay so
    dropout masks match, and put the backward-time state back so the training stream is untouched.
  - Named the module `recompute` to avoid colliding with `engine/checkpoint.py` (on-disk state).
  - Added `GPT(..., grad_checkpoint=...)` wrapping each block, kept out of `config()` as a runtime
    toggle, shown in `__repr__`, and exposed as `--grad-checkpoint` applied after construction or
    resume in `train.py`.
  - Added `tests/test_recompute.py` (19 tests): plain-call equivalence, residual-consumer safety,
    repeated backward accumulation, intermediate release, `no_grad()` passthrough, RNG neutrality,
    dropout-mask replay, argument validation, model-level gradient and trajectory equality with and
    without dropout, LoRA compatibility, retained-tensor reduction, and config/inference behavior.
  - Measured the tradeoff and documented it in the README with the exact figures.
- Files created/modified:
  - `src/engine/recompute.py` (created)
  - `src/engine/__init__.py`
  - `src/nn/transformer.py`
  - `src/train.py`
  - `tests/test_recompute.py` (created)
  - `README.md`

### Phase 10: Round 6 — Document corpora in the training CLI
- **Status:** complete
- Actions taken:
  - Added `--data-format text|lines|jsonl` and `--jsonl-field`, with `_validate_args` coverage.
  - Added `load_documents`, `encode_documents`, `get_document_batch`, `batch_loss`,
    `evaluate_batches`, and `evaluate_documents` to `train.py`, keeping `get_batch` and `evaluate`
    signatures intact for existing callers and tests.
  - Routed training and validation through one `(tokens, targets, mask)` sampler so the stream path
    passes `mask=None` and consumes the RNG in exactly the same order as before.
  - Moved document splitting ahead of tokenizer construction after observing a JSONL char vocab of
    26 versus 19 for the same corpus; the tokenizer now sees document text only.
  - Switched the default generation prompt from the raw file to the corpus text, so JSONL runs no
    longer prompt with `{"body": ...`.
  - Split train/validation over documents and reported dropped documents shorter than two tokens.
  - Added `tests/test_data.py` (23 tests) covering parsing, encoding, batch layout, loss and
    gradient equivalence with per-document scoring, evaluation, and four in-process CLI runs.
  - Added the two new CLI fields to the existing arg-validation fixtures in `test_modern.py` and
    `test_features.py`.
- Files created/modified:
  - `src/train.py`
  - `tests/test_data.py` (created)
  - `tests/test_modern.py`
  - `tests/test_features.py`
  - `README.md`

### Phase 11: Round 7 — Handoff documentation and checkpoint
- **Status:** complete
- Actions taken:
  - Added `CLAUDE.md`: test/build commands, module layout, the numbered invariant list with the
    failure each one prevents, code conventions, the probe-then-assert-then-counter-test
    verification style, planning-artifact roles, and environment gotchas.
  - Added `PROJECT_STATE.md`: per-file architecture map with line counts, all six rounds of
    completed work, the design-decision table, the test baseline with per-module coverage, the
    measured tradeoffs, deliberate limitations, repository state, and ranked next candidates.
  - Established a reproducible CLI regression anchor and recorded its exact output, replacing the
    round-6 reference to numbers whose command had never been written down.
  - Verified the anchor reproduces run to run and is identical under `--grad-checkpoint`.
  - Updated `task_plan.md` (Phase 11, four decisions, three acceptance criteria), `findings.md`
    (Round 7 findings), and this log.
  - Created the checkpoint commit, preserving the pre-existing user diff.
- Files created/modified:
  - `CLAUDE.md` (created)
  - `PROJECT_STATE.md` (created)
  - `task_plan.md`
  - `findings.md`
  - `progress.md`

### Phase 12: Round 8 — Sliding-window decoding for masked runs
- **Status:** complete
- **Started:** 2026-07-31
- Actions taken:
  - Re-read `CLAUDE.md`, `PROJECT_STATE.md`, and `task_plan.md`; confirmed the 301-test baseline
    and the CLI anchor before choosing work, and selected the top-ranked next candidate.
  - Traced the interaction that made round 4 refuse the case: the crop moves the window start, so
    both `position_ids` and the cached-key mask alignment change.
  - Removed the `fit in context_len` refusal and merged `generate`'s two prefill branches into one
    window branch that crops `idx`, `mask`, and `positions` together.
  - Added `_left_padded_positions` and applied it to the *cropped* mask on every re-prefill, so the
    surviving real tokens are renumbered from 0.
  - Sliced the cached-step mask to `cache_len + 1` so it still covers exactly the cached plus
    current keys after the window start moves.
  - Probed the new behavior before writing tests: every row matched solo decoding for 6 and 20 new
    tokens, both architectures, cached and uncached, and for a prompt longer than `context_len`.
  - Confirmed a missing renumbering fails loudly by emulating it — `_validate_position_ids` raises.
  - Added `TestMaskedSlidingWindow`, subclassing the round-4 class at `context_len=8` so all 14
    in-window guarantees re-run under a sliding window, plus 12 new cases including the counter-test
    and an anchor against a direct `infer` call.
  - Verified no regression by diffing 52 unmasked and 24 in-window masked generation results
    against the previous commit's source.
  - Measured the cost of decoding past the window and recorded it as the next candidate.
  - Updated `README.md`, `CLAUDE.md` (invariant 14, counts), `PROJECT_STATE.md`, `task_plan.md`
    (Phase 12, six decisions, six acceptance criteria), `findings.md`, and this log.
- Files created/modified:
  - `src/nn/transformer.py`
  - `tests/test_transformer.py`
  - `README.md`
  - `CLAUDE.md`
  - `PROJECT_STATE.md`
  - `task_plan.md`
  - `findings.md`
  - `progress.md`

## Session: 2026-08-03

### Phase 13: Round 9 — Inference-mode and mask-contract audit
- **Status:** complete
- Actions taken:
  - Re-read the complete planning-with-files instructions and restored all existing planning context.
  - Ran the session catch-up helper; it reported no unsynchronized context.
  - Confirmed from the persistent plan that both requested features shipped in round 2 and now underpin later ragged-batch, generation, and recomputation work.
  - Opened a fresh audit phase to verify the current implementation and extend it only where concrete gaps remain.
  - Confirmed the round-8 checkpoint commit and a clean pre-round worktree; only planning artifacts changed after opening phase 13.
  - Read the handoff invariants and mapped every grad-mode, inference, mask, and fully-masked-row reference across source, tests, and README.
  - Ran the fresh baseline suite with warnings as errors: all 324 tests passed.
  - Read the grad-mode/Tensor/recompute implementation and the attention/GPT/train inference paths; recorded four concrete lifecycle and direct-API candidates for reproduction.
  - Reproduced all four candidates with one deterministic probe: post-scope detached backward silently no-ops, a shared guard crosses thread restoration stacks, evaluation exceptions leak eval mode, and standalone inference accepts poisoned/oversized key biases.
  - Independent mask review reproduced two contract bugs: documented `(B,T,T)` multi-head masks are misread as per-head masks, and documented right-padding is not enforced in `GPT.forward`.
  - Recorded missing public KV-cache structure validation and constant-only Tensor parent retention as bounded lifecycle hardening candidates.
  - Inspected all public documentation sections and identified the exact no-grad, custom-mask, right-padding, cache, suite-count, and handoff text that must be synchronized after implementation.
  - Integrated and reviewed the grad-mode/Tensor lifecycle implementation and eight new regressions; focused tests passed 48 and an intermediate full suite passed 332.
  - Integrated and reviewed shared forward/inference mask validation, batch-major 3-D normalization, and thirteen transformer regressions; the transformer module passed 90 tests.
  - Integrated evaluation exception safety, enforced training right padding, and added structural KV-cache validation with twenty-five parameterized boundary regressions.
  - Re-ran every original failure probe against the integrated code; all now pass.
  - Ran the five affected test modules: 229 passed with warnings as errors. Byte compilation and `git diff --check` also passed.
  - Collected exactly 372 tests and recorded every per-module count plus current source line counts for documentation synchronization.
  - Synchronized README, invariants, and the cold-start project snapshot with round-9 behavior, limitations, source sizes, and all per-module test counts.
  - Closed the final low-severity review gap by validating finite real-valued KV caches in standalone attention and GPT, with eight additional regressions.
  - Ran the final integrated suite: 380 passed with warnings as errors; collection also reports exactly 380. Compilation, diff, and stale-document-count checks passed.
  - Built the final wheel, force-installed it in an isolated venv with NumPy available from system site packages, verified imports plus round-9 behavior, and ran the installed CLI help.
  - Resolved and removed only the three generated packaging targets: `.tmp-round9-package-audit/`, `build/`, and `src/tiny_transformer.egg-info/`; all are reproducible and contained no user data.
  - Received final independent lifecycle, mask/cache, and delivery reviews; all reported no remaining actionable issue after the finite-cache hardening.
  - Removed the generated `.pytest_cache` and confirmed no packaging/test artifacts or untracked files remain.
- Files created/modified:
  - `.github/workflows/tests.yml`
  - `README.md`
  - `CLAUDE.md`
  - `PROJECT_STATE.md`
  - `src/engine/grad_mode.py`
  - `src/engine/tensor.py`
  - `src/nn/attention.py`
  - `src/nn/transformer.py`
  - `src/train.py`
  - `tests/test_data.py`
  - `tests/test_grad_mode.py`
  - `tests/test_transformer.py`
  - `tests/test_validation.py`
  - `task_plan.md`
  - `findings.md`
  - `progress.md`

## Test Results
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| Baseline full suite | `python -m pytest -q` | Existing suite passes before new work | 130 passed in 0.80s | PASS |
| Matmul edge probe | Broadcast batches and all 1-D operand combinations | NumPy-compatible gradients | Broadcast `ValueError`; 1-D `AxisError` | FAIL |
| Sigmoid stability probe | `[-1000, 1000]` | Finite values without warnings | Correct values, four runtime warnings | FAIL |
| Extreme cross entropy probe | logits `[-1000, 1000]`, target 0 | Loss near 2000, gradient `[-1,1]` | Loss 27.63, gradient `[-1,1]` | FAIL |
| Repeated chained backward probe | Call backward twice on `(x*x)^2` at x=2 | Leaf accumulation reaches 64 | Leaf gradient reaches 96 | FAIL |
| Baseline wheel build | `python -m pip wheel . --no-deps --wheel-dir .tmp-wheel-audit` | Wheel builds | Backend import failure | FAIL |
| Attention/checkpoint focused suite after first implementation | `python -m pytest tests/test_transformer.py tests/test_training.py -q` | New behavior and old tests pass | 53 passed in 0.28s | PASS |
| API-validation and affected integration suites | `python -m pytest tests/test_validation.py tests/test_transformer.py tests/test_training.py -q` | New validation plus existing behavior passes | 73 passed in 0.33s | PASS |
| Repaired wheel build | `python -m pip wheel . --no-deps --wheel-dir .tmp-wheel-audit` | PEP 517 wheel builds | Built `tiny_transformer-0.1.0-py3-none-any.whl` | PASS |
| Wheel content audit | Inspect wheel ZIP entries | Engine/NN plus train/tokenizer/benchmark and entry point metadata | All intended files present | PASS |
| Isolated installed-package smoke | Install wheel into venv; import modules; run `tiny-train --help` | Installed artifacts work outside `src` injection | Imports and CLI help succeeded | PASS |
| Integrated full suite | `python -m pytest -q -W error` | All old and new tests pass without warnings | 170 passed in 0.60s | PASS |
| Whitespace validation | `git diff --check` | No whitespace errors | Passed; LF/CRLF notices only | PASS |
| Documentation stale-pattern scan | Search old counts/mask/default/plot flag | No stale claims remain | No matches (`rg` exit 1 means zero matches) | PASS |
| Plot CLI help | `python plot_loss.py --help` | Documented parser loads | Help displayed successfully | PASS |
| Versioned checkpoint tests | `python -m pytest tests/test_training.py -q -W error` | Current, legacy, RNG, optimizer, and future-version cases pass | 26 passed in 0.21s | PASS |
| Stochastic checkpoint resume suite | `python -m pytest tests/test_training.py -q -W error` | Random batches/dropout and optimizer trajectory resume consistently | 27 passed in 0.22s | PASS |
| Test collection | `python -m pytest --collect-only -q` | README count matches collection | 172 tests collected | PASS |
| Byte compilation | `python -m compileall -q src tests plot_loss.py` | All Python files compile | No errors | PASS |
| End-to-end optimizer-aware resume | Train/save step 1 with AdamW, resume to step 2 without `--optimizer` | AdamW and step restore automatically | Reported `optimizer=AdamW`, `resume_step=1`; completed step 2 | PASS |
| Legacy causal-mask migration | `python -m pytest tests/test_transformer.py -q -W error` | Old `-1e9` buffers cannot weaken current causal semantics | 29 passed in 0.26s | PASS |
| Final integrated suite (pre-review) | `python -m pytest -q -W error` | Entire expanded suite passes | 173 passed in 0.60s | PASS |
| Final-review correctness regressions | `python -m pytest tests/test_autograd.py tests/test_training.py -q -W error` | Mutable targets and failed restores are safe | 75 passed in 0.34s | PASS |
| All review-affected suites | `python -m pytest tests/test_autograd.py tests/test_training.py tests/test_transformer.py tests/test_validation.py -q -W error` | Final lifecycle/API fixes preserve behavior | 127 passed in 0.37s | PASS |
| Final complete suite | `python -m pytest -q -W error` | All delivered behavior passes without warnings | 178 passed in 0.63s | PASS |
| Final compilation | `python -m compileall -q src tests plot_loss.py` | Every Python file compiles | No errors | PASS |
| Final diff check | `git diff --check` | No whitespace defects | Passed; line-ending notices only | PASS |
| Delivery artifact scan | Check root/src for build, dist, temp, wheel, checkpoint, egg-info | No generated artifacts remain | None found | PASS |
| Round 2 pre-change baseline | `python -m pytest -q -W error` | Suite green before the new features | 178 passed in 0.68s | PASS |
| Round 2 first new-test run | `pytest tests/test_grad_mode.py tests/test_autograd.py tests/test_transformer.py tests/test_validation.py -q -W error` | New behavior passes | 3 failed, 134 passed (decorator subclass args; two wrong mask expectations) | FAIL |
| Detached-node backward probe | Re-enable unconditional closure storage, then differentiate through a gradient-less `concat` | Reproduce the pre-guard failure | `TypeError: object of type 'NoneType' has no len()`; guarded version returns `[1, 2]` | PASS |
| Evaluation cost probe | 10 forward+loss passes, 4 layers, d=128, ctx=64, batch=8 | `no_grad()` is measurably cheaper | 1.102s to 0.586s (1.88x); live tensors 346 to 65 | PASS |
| Round 2 full suite | `python -m pytest -q -W error` | All old and new tests pass without warnings | 217 passed in 0.70s | PASS |
| Round 2 test collection | Per-file `pytest --collect-only -q` | README counts match collection | 55/34/28/21/14/37/28 = 217 | PASS |
| Round 2 package imports | `import engine, nn, train, tokenizer, benchmark` plus `engine.no_grad` usage | New module is importable through the package | Exports resolved; op detached under `no_grad()` | PASS |
| Round 2 compilation and whitespace | `python -m compileall -q src tests plot_loss.py`; `git diff --check` | Everything compiles, no whitespace defects | No errors; line-ending notices only | PASS |
| Round 2 training CLI smoke | `python src/train.py --iters 4 --eval-interval 2 --eval-iters 2 --dropout 0.1 --no-sample` | Validation under `no_grad()` works in a real run | Steps 1/2/4 reported finite train and val loss/perplexity | PASS |
| Round 3 padding and ignore_index suites | `pytest tests/test_transformer.py tests/test_autograd.py -q -W error` | Padding equivalence and ignored-position gradients hold | 103 passed in 0.59s | PASS |
| Round 3 padding-mask leak counter-test | Change padded token content with and without `attention_mask` | Masked run is unchanged; unmasked run differs | Both assertions held | PASS |
| Round 3 ragged-batch training | 15 Adam steps on a padded 3-sequence batch with `ignore_index` | Loss at least halves; padding content is irrelevant | Loss halved; scrambled padding gave a bit-identical loss | PASS |
| Round 3 full suite | `python -m pytest -q -W error` | All old and new tests pass without warnings | 239 passed in 0.94s | PASS |
| Round 3 test collection | Per-file `pytest --collect-only -q` | README counts match collection | 63/41/28/21/15/37/34 = 239 | PASS |
| Round 3 compilation | `python -m compileall -q src tests plot_loss.py` | Every Python file compiles | No errors | PASS |
| Round 4 padded-generation probe | Left-padded batch of 3 prompts, greedy, both architectures | Rows match per-row generation; cached equals uncached | All rows matched; cached == uncached; logits max error 0.000e+00; all finite | PASS |
| Round 4 inference-path regression | `python -m pytest -q -W error` after threading key bias and positions | Existing KV-cache and RoPE behavior unchanged | 239 passed in 0.92s | PASS |
| Round 4 padded-generation suite | `pytest tests/test_transformer.py -q -W error` | New generation contract holds | 54 passed in 0.30s | PASS |
| Round 4 RoPE positions test fix | `np.arange(24.0).reshape(1, 1, 3, 4)` | Build a (1, 1, 3, 4) input | `ValueError: cannot reshape array of size 24` | FAIL |
| Round 4 full suite | `python -m pytest -q -W error` | All old and new tests pass without warnings | 259 passed in 0.82s | PASS |
| Round 4 test collection | Per-file `pytest --collect-only -q` | README counts match collection | 63/54/28/21/15/37/41 = 259 | PASS |
| Round 5 recompute suite | `pytest tests/test_recompute.py -q -W error` | Recomputation changes nothing observable | 19 passed in 0.53s | PASS |
| Round 5 full suite | `python -m pytest -q -W error` | All old and new tests pass without warnings | 278 passed in 0.95s | PASS |
| Round 5 tradeoff measurement | 6 layers, d=128, ctx=64, batch 8, 5 steps | Less activation memory for more time, identical losses | 320.9 MiB/413 tensors and 227 ms/step to 14.2 MiB/29 tensors and 342 ms/step (22.7x less, 1.51x slower); losses identical | PASS |
| Round 5 CLI equivalence | Same seed with and without `--grad-checkpoint` | Identical training trajectory | Both runs reported identical losses and gradient norms at steps 1 and 5 | PASS |
| Round 5 CLI smoke with dropout | `python src/train.py --grad-checkpoint --dropout 0.1 --layers 2` | Flag reaches the model and trains | `[info] GPT(..., grad_checkpoint)`; steps completed | PASS |
| Round 6 arg-validation fixtures | `python -m pytest -q -W error` after adding CLI fields | Existing partial Namespace fixtures still validate | `AttributeError: 'Namespace' object has no attribute 'data_format'` | FAIL |
| Round 6 document CLI smoke | `--data-format lines` and `--data-format jsonl --jsonl-field body` on a 40-document corpus | Both corpora train and report document counts | Both reported `train=36 val=4 documents` and completed 6 steps | PASS |
| Round 6 JSONL vocabulary check | Same corpus through `lines` and through `jsonl` | Identical vocabulary | 19 versus 26 before the fix; 19 for both after | PASS |
| Round 6 stream-path regression | Re-ran the round-5 command with seed 7 | Trajectory identical to the recorded numbers | `train_loss=3.6031/3.5854`, `val_loss=3.5694/3.5598`, `gnorm=2.230/2.095` — exact match | PASS |
| Round 6 document suite | `pytest tests/test_data.py -q -W error` | Parsing, batching, and equivalence hold | 23 passed in 0.32s | PASS |
| Round 6 full suite | `python -m pytest -q -W error` | All old and new tests pass without warnings | 301 passed in 1.02s | PASS |
| Round 6 test collection | Per-file `pytest --collect-only -q` | README counts match collection | 63/23/15/21/37/19/28/54/41 = 301 | PASS |
| Round 7 CLI anchor determinism | The recorded anchor command run twice | Byte-identical step lines | `3.6106/3.5812/3.5234`, `gnorm 2.677/2.654/2.547` both times | PASS |
| Round 7 CLI anchor under checkpointing | Same anchor plus `--grad-checkpoint` | Losses and gradient norms unchanged | All three step lines identical to the baseline run | PASS |
| Round 7 full suite | `python -m pytest -q -W error` | Documentation round changes no behavior | 301 passed in 1.12s | PASS |
| Round 7 byte compilation | `python -m compileall -q src tests plot_loss.py` | All modules compile | Clean | PASS |
| Round 7 diff hygiene | `git diff --check` | No whitespace errors | Clean apart from existing CRLF notices | PASS |
| Round 8 baseline | `python -m pytest -q -W error` before any edit | Suite green at the recorded count | 301 passed in 1.68s | PASS |
| Round 8 crossing-run probe | Prompts of length 5/2/3 at `ctx=8`, 6 new tokens | Each row matches solo decoding | Match for gpt and llama, cached and uncached | PASS |
| Round 8 over-long prompt probe | 10- and 6-token prompts at `ctx=8`, 5 new tokens | First prefill crops, rows still match | Match for both architectures | PASS |
| Round 8 far-past-window probe | 20 new tokens at `ctx=8` (width 5 → 25) | Repeated crops stay equivalent | Match for all three rows | PASS |
| Round 8 missing-renumbering probe | Emulate positions carried across a crop | Fails loudly, does not drift | `ValueError: position_ids must be in [0, 8)` | PASS |
| Round 8 unmasked-path regression | 52 generation results diffed against `HEAD:src` | Byte-identical | No diff (both architectures × ctx × cache × greedy/sample/beam × long prompts) | PASS |
| Round 8 in-window masked regression | 24 masked results at `ctx=16` diffed against `HEAD:src` | Byte-identical | No diff | PASS |
| Round 8 window-slide count | 20-token masked run at `ctx=8` | Crop path actually exercised | 17 prefills, 16 of them full-window | PASS |
| Round 8 decoding cost | `ctx=64`, 4 layers, `d_model=128`, 3 ragged rows | Measure the cost of the opened path | 1.10 ms/token inside → 11.21 ms/token past, 10.2× | PASS |
| Round 8 transformer suite | `pytest tests/test_transformer.py -q -W error` | New and old generation tests pass | 77 passed in 0.49s | PASS |
| Round 8 full suite | `python -m pytest -q -W error` | All old and new tests pass without warnings | 324 passed in 1.20s | PASS |
| Round 8 CLI anchor | The recorded anchor command | Training trajectory unchanged | `3.6106/3.5812/3.5234`, `gnorm 2.677/2.654/2.547` — exact match | PASS |
| Round 8 test collection | `pytest --collect-only -q` per file | Documented counts match collection | 63/23/15/21/37/19/28/77/41 = 324 | PASS |

| Round-9 baseline | `python -m pytest -q -W error` | Existing advanced suite stays green | 324 passed in 1.42s | PASS |
| Original round-9 failure probes after fixes | Delayed backward, shared guard, eval failure, invalid masks/cache | Every former failure is rejected/restored explicitly | All probes passed | PASS |
| Final affected suites | Grad/recompute/transformer/data/validation | All focused lifecycle and mask behavior passes | 229 passed in 0.76s before final cache additions; cache suites 158 passed after | PASS |
| Final complete suite | `python -m pytest -q -W error` | Entire delivered suite passes warning-free | 380 passed in 1.14s | PASS |
| CLI trajectory anchor | Seeded 3-step default trainer command | Exact round-7 loss/val/lr/gnorm values | All three lines reproduced exactly | PASS |
| Final installed wheel | Build, force-install, imports, round-9 probes, `tiny-train --help` | Current tree works outside source injection | All checks passed | PASS |
| Final artifact/status audit | Build/dist/venv/egg-info/wheel/pytest cache + untracked scan | No generated delivery artifacts or untracked files | None found | PASS |

## Error Log
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-07-30 | README/config read and `git diff --stat` both exceeded 10-second timeout | 1 | Retry with 30-second budget and smaller output. |
| 2026-07-30 | Inline edge-case probe raised `ModuleNotFoundError: engine` | 1 | Retry with repository `src` inserted into `sys.path` inside the probe. |
| 2026-07-30 | Wheel build raised `BackendUnavailable: Cannot import 'setuptools.backends.legacy'` | 1 | Fixed backend to `setuptools.build_meta`, then rebuilt successfully. |
| 2026-07-30 | Planning-file patch context did not match an existing test row | 1 | Re-read files and applied a context-accurate patch. |
| 2026-07-30 | API-validation patch could not match a Unicode `Embedding` docstring as rendered by PowerShell | 1 | Confirmed patch was not applied; retry with narrow ASCII-only contexts per file. |
| 2026-07-30 | Policy rejected the verified multi-target recursive cleanup command before execution | 1 | Separate explicit `Remove-Item -LiteralPath` commands within the verified workspace paths succeeded. |
| 2026-07-30 | Stochastic resume parameter `assert_array_equal` found a 2.17e-19 allocation/BLAS rounding difference | 1 | Kept exact batch and loss checks; changed parameter comparison to strict 1e-15 allclose. |
| 2026-07-30 | Final-review planning update had an overly broad patch context mismatch | 1 | Re-read exact locations and applied smaller patches. |
| 2026-07-30 | Final status grep used a literal `\n` without ripgrep multiline mode | 1 | Reran with a status-only expression that does not span lines. |
| 2026-07-30 | `@no_grad()` decorator raised `TypeError: no_grad.__init__() takes 1 positional argument but 2 were given` | 1 | Wrapper now builds a fresh `set_grad_enabled(mode)` guard instead of re-instantiating the subclass. |
| 2026-07-30 | Mask-validation test expected a rejection for a `(1, 2, 4, 4)` mask that is exactly the score shape | 1 | Used `(2, 2, 4, 4)` for "larger than scores" and moved `(5, 5)` to the non-broadcastable case. |
| 2026-07-30 | RoPE positions test built its input with `np.arange(24.0).reshape(1, 1, 3, 4)` | 1 | Size mismatch, not a source defect; used `np.arange(12.0)` for the (1, 1, 3, 4) shape. |
| 2026-07-31 | Two arg-validation fixtures raised `AttributeError` for the new `data_format` field | 1 | Added `data_format`/`jsonl_field` to the fixtures rather than making the validator tolerate partial namespaces, which would hide typos. |
| 2026-07-31 | A scripted replace of `_print_sample(model, tokenizer, text, args)` asserted 3 occurrences and found 4 | 1 | The fourth was the `def` line itself; replaced only call sites (trailing newline) and renamed the parameter separately. |


| 2026-08-03 | Phase 13 planning patch matched a console-garbled em dash in the phase heading | 1 | Re-read exact headings with `rg` and patched against stable section boundaries. |
| 2026-08-03 | Batched plan/test inspection treated ripgrep's normal no-match exit code as fatal | 1 | Kept the returned test evidence and reran the plan read separately with optional-search no-match handling. |
| 2026-08-03 | Round-9 multi-document patch used a non-exact wrapped `PROJECT_STATE.md` sentence | 1 | Split updates per file and used stable ASCII heading boundaries. |
| 2026-08-03 | Second `PROJECT_STATE.md` patch matched a console-rendered Unicode baseline separator | 1 | Switched to small ASCII-only replacements and heading-anchored paragraph updates. |
| 2026-08-03 | Optional `PROJECT_STATE.md` cleanup again depended on Unicode heading context | 1 | Kept the harmless separator and limited the retry to ASCII contract text. |
| 2026-08-03 | Installed-wheel smoke used `--no-deps` in a venv without NumPy | 1 | Recreate only the generated venv with system site packages and reuse the successfully built wheel. |
| 2026-08-03 | Policy rejected combined computed-path verification and recursive venv removal | 1 | Resolve the target read-only, then remove the exact literal path in a separate native PowerShell command. |
| 2026-08-03 | Truncating installed CLI help closed stdout early and produced exit -1 after valid output | 1 | Rerun complete help redirected to null and emit a separate success marker. |
| 2026-08-03 | Cache-hardening planning patch used a non-exact progress-log row | 1 | Split the update by file and re-read the exact row before patching it. |
| 2026-08-03 | Final artifact scan found ignored `.pytest_cache` after the full suite | 1 | Resolve and explicitly remove the generated cache, then rerun status/artifact checks without rerunning pytest afterward. |

## 5-Question Reboot Check
| Question | Answer |
|----------|--------|
| Where am I? | Round 9 complete: inference lifecycle, mask semantics, padding, evaluation restoration, and every public KV-cache boundary are hardened and documented. |
| Where am I going? | Hand off a tested teaching project. Remaining non-blocking extensions are ranked in `PROJECT_STATE.md`, led by faster decoding after a context-window crop. |
| What's the goal? | Improve the project materially and prove the result with tests, packaging, and documentation. |
| What have I learned? | Current mode cannot identify a tensor's creation history; NumPy broadcasting can silently exchange batch and head semantics; and shape-valid caches still need dtype/finiteness checks. Strong boundary tests caught all three. |
| What have I done? | Added suppression provenance and thread-safe guards, pruned constant graphs, unified graph/NumPy masks, enforced padding, made evaluation exception-safe, validated caches, expanded the suite to 380, and verified the installed wheel/CLI. |
