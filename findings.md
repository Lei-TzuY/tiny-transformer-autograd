# Findings & Decisions

## Requirements
- Continue improving, expanding, and implementing `Tiny Transformer & Autograd` rather than assuming it is finished.
- Determine from the repository whether meaningful improvement remains.
- Make concrete changes, not only provide suggestions.
- Preserve a small, understandable educational implementation while raising correctness and completeness.

## Research Findings
- No prior planning artifacts existed when this session began.
- The repository is a compact Python project with `src/engine`, `src/nn`, training/benchmark scripts, and five test modules.
- The worktree already contained extensive uncommitted changes before this audit: `README.md`, core engine/optimizer files, neural-network files, training/benchmark code, and `tests/test_features.py` are modified; `tests/test_modern.py` is untracked. These changes must be treated as user-owned and preserved.
- Current branch is `main`, tracking `origin/main`.
- The pre-existing diff is substantial: 389 insertions and 60 deletions across 11 tracked files, plus an untracked 37-test `tests/test_modern.py`; it already appears to add a Llama-style architecture, RMSNorm/RoPE/SwiGLU, AdamW, gradient accumulation, and related documentation.
- The current README presents a feature-rich 130-test project: dynamic NumPy autograd, SGD/Adam/AdamW, schedulers, GPT/Llama variants, LoRA, BPE, checkpointing, KV-cache generation, and benchmarking. The new work therefore needs to find actual correctness/usability gaps rather than duplicate advertised features.
- Packaging metadata declares `tiny-train = "train:main"` while setuptools is configured only to discover packages beneath `src`; whether the top-level `src/train.py` module is installed must be verified.
- README output contains visibly garbled box-drawing/math glyphs in this PowerShell rendering. This may be console decoding rather than file corruption and requires a byte/UTF-8 check before editing.
- Baseline suite is green: all 130 tests pass in 0.80 seconds on Python/pytest in the current environment.
- A repository-wide unfinished-marker search found no actionable TODO/FIXME/placeholder in source; the only `NotImplementedError` is the expected abstract `Module.forward` contract.
- Manual autograd inspection found a likely real correctness gap in `ops.matmul`: backward only reduces extra leading dimensions, not same-rank broadcast dimensions of size 1. For example, `(1,M,K) @ (B,K,N)` produces an `(B,M,K)` gradient that cannot be accumulated in-place into the `(1,M,K)` operand.
- `matmul` delegates forward to NumPy (which accepts 1-D operands) but its backward unconditionally swaps the final two axes, so vector-matrix, matrix-vector, and dot-product backward cases are likely broken.
- `Tensor.backward()` overwrites the root gradient but does not clear intermediate node gradients. Reusing the same graph for a second backward likely over-accumulates intermediate gradients and produces more than the expected leaf accumulation.
- Stable sigmoid computes both `np.where` branches eagerly; extreme magnitudes can still emit overflow warnings even though the selected values are finite. The SiLU implementation already demonstrates an `exp(-abs(x))` formulation that avoids this.
- `cross_entropy` is restricted implicitly to `(N,C)` logits and uses `log(softmax + 1e-12)`, which caps extreme losses and can make its forward value inconsistent with the otherwise stable analytical gradient. Shape/range validation is also absent.
- Targeted probes confirmed `matmul` backward fails both same-rank batch broadcasting (`ValueError` for `(1,2,3) @ (4,3,5)`) and every tested 1-D case (`AxisError` for vector-matrix, matrix-vector, and vector dot products).
- Extreme sigmoid inputs return the correct `[0, 1]` values but emit multiple overflow/invalid warnings because both `np.where` branches are evaluated.
- Extreme cross entropy is numerically wrong in the forward pass: logits `[-1000, 1000]` with target class 0 return about 27.63 instead of about 2000, while the gradient remains `[-1, 1]`.
- A direct repeated backward on the root operation (`x*x`) accumulates leaves correctly (4 then 8); the suspected intermediate-node case still needs a chained-graph probe.
- Chained repeated backward is confirmed incorrect: for `loss=(x*x)^2` at `x=2`, the first derivative is 32, but a second call leaves 96 instead of the expected accumulated 64 because the intermediate gradient grows from 8 to 16 and is propagated again.
- The project initially could not build a wheel under PEP 517. `python -m pip wheel . --no-deps` failed while loading the nonexistent backend `setuptools.backends.legacy`; the delivered configuration uses `setuptools.build_meta`.
- The wheel audit created an empty `.tmp-wheel-audit` directory that should be removed after verification.
- Transformer/training audit independently reproduced the extreme cross-entropy inconsistency and found that negative target indices are silently treated as valid NumPy indexing. The model/training-focused subset still passes 99/99, confirming a coverage gap rather than a currently detected regression.
- Autograd audit also found tensor division is implemented through `exp(-log(b))`; negative tensor divisors therefore produce NaN instead of valid signed quotients/gradients.
- The autograd audit additionally reproduced backward failure for transpose permutations containing negative axes, a recursive topological-sort `RecursionError` around 1,100 chained operations, and generator exhaustion in `concat` because `any()` consumes the iterable before concatenation.
- Current tests mostly seed non-scalar outputs with implicit all-ones gradients; that leaves important vector-Jacobian products under-tested (for example, `sum(softmax)` is always constant and cannot meaningfully validate its Jacobian).
- `detach()` documentation claims shared storage although the Tensor constructor copies; evaluation builds graphs because there is no `no_grad` mechanism. These are lower-priority API/performance gaps.
- Manual attention/Transformer inspection found the main cached causal-attention flow internally coherent, including learned-position cache resets at the context boundary; no obvious main-path attention math regression was found.
- User-input boundaries are weak: `GPT.forward` does not explicitly reject sequences longer than `context_len`; token arrays can use negative NumPy indices; `generate` does not validate rank, non-empty prompts, or non-negative `max_new_tokens`; constructors use an `assert` for head divisibility and omit several positive/range validations. Failures therefore surface as incidental NumPy errors or silently select unintended tokens.
- RoPE slice bounds and KV-cache structure are not validated, so malformed direct API calls can produce opaque broadcasting/index errors. These are usability-hardening opportunities after mathematical correctness.
- Both `SelfAttention` and `MultiHeadAttention` advertise causal self-attention, yet training `forward(mask=None)` is unmasked while `infer()` always creates a causal mask. A direct parity probe found a maximum error around 0.51 without an explicit mask and exact parity when one is supplied; custom callers can therefore train with future-token leakage and decode causally.
- Checkpoints persist optimizer buffers/hyperparameters but not the optimizer class. Resume constructs Adam/AdamW from the current CLI (default Adam) before loading generic state, so an AdamW run resumed using the documented command silently changes update semantics.
- `save_checkpoint`/`restore_checkpoint` can safely gain an optional optimizer type field with backward compatibility; `train.py` should select a recorded optimizer before constructing it and reject an explicit conflicting override.
- Training CLI validation is already fairly thorough, including positive shapes/steps, RoPE parity, sampling filters, prompt conflicts, and data splits. Direct library APIs remain less guarded than the CLI.
- Existing attention tests pass `mask=None` in shape/gradient cases but only verify causality when manually supplying a mask; they do not compare the documented default forward path with inference.
- Existing checkpoint tests cover Adam state round-trips and uninterrupted/resumed loss parity, but not optimizer-class identity, mismatch rejection, or legacy checkpoint compatibility.
- CI only installs NumPy/pytest and runs source-tree tests on Python 3.11/3.12; it has no wheel/build/entry-point smoke test and does not exercise the declared Python 3.10 minimum.
- `plot_loss.py` documents an unsupported `--metric val_ppl` example. The implementation plots train/validation loss plus optional learning rate and accepts only `--out`, `--title`, and `--no-lr`.
- The README is valid UTF-8; garbled glyphs observed in PowerShell output are a terminal rendering issue, not file corruption.

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Use repository-local evidence to choose scope | The user did not prescribe a feature, so tests, APIs, and missing invariants should drive the work. |
| Treat the passing baseline as necessary but not sufficient | Numerical/autoregressive edge cases and installation behavior may be absent from the 130 tests. |
| Include optimizer identity in checkpoint hardening | Silent algorithm changes on resume invalidate training reproducibility even when state arrays happen to load. |
| Make causal attention causal by default | The classes and inference path already promise causal semantics; forward parity is safer than requiring every custom caller to construct a mask. |
| Scope this pass around confirmed failures | Fix autograd edge semantics, causal parity, optimizer-aware resume, and packaging; defer larger framework abstractions until these foundations are trustworthy. |
| Use true negative infinity for causal masking | A finite `-1e9` bias can be overcome by sufficiently large logits and does not enforce the advertised invariant. |
| Save global NumPy RNG state in new checkpoints | Batch sampling, dropout, and token sampling all depend on it; restoring it makes stochastic resume reproducible while the field can remain optional for legacy files. |
| Include focused public-API validation | Confirmed NaNs/opaque NumPy failures for invalid dropout, Adam betas, attention dimensions, RoPE bounds, token inputs, and generation bounds are cheap to replace with explicit errors. |

## Implementation Findings
- Focused attention, checkpoint, and validation suites pass 73/73 after the first implementation batch.
- Parameter iterables are now materialized by optimizers, preventing generator exhaustion in optimizer state buffers and updates.
- The repaired PEP 517 configuration now builds `tiny_transformer-0.1.0-py3-none-any.whl` successfully.
- Wheel inspection confirms it contains `train.py`, `tokenizer.py`, `benchmark.py`, both `engine` and `nn` packages, and entry-point metadata.
- An isolated virtual environment installed the built wheel with `--no-deps`, imported `engine`, `nn`, `train`, `tokenizer`, and `benchmark`, and ran the generated `tiny-train --help` command successfully.
- Autograd hardening now covers full NumPy matmul promotion/broadcasting, generalized stable cross entropy, repeated-backward lifecycle, iterative deep-graph traversal, direct division, stable sigmoid, negative-axis transpose, and generator-safe concat.
- Diff review confirmed the pre-existing SiLU and AdamW exports remain intact alongside the new `div` export; the added VJP tests use non-uniform cotangents rather than only implicit output sums.
- The integrated suite passes all 170 tests with Python warnings promoted to errors; `git diff --check` reports no whitespace errors (only existing Windows LF-to-CRLF notices).
- README was synchronized throughout implementation and now reflects 18 ops, 178 tests, default causal masking, optimizer/RNG-aware checkpoints, the installed CLI, and final test-file counts.
- Plotting documentation no longer advertises the nonexistent `--metric` flag, and `plot_loss.py --help` succeeds with the documented options.
- CI now installs the package through PEP 517 on Python 3.10/3.11/3.12, smoke-tests installed imports and `tiny-train`, then runs tests with warnings-as-errors.
- Checkpoints now carry format version 2; restore treats missing versions as legacy version 1, rejects unsupported future formats, and validates optimizer compatibility before loading model/optimizer state.
- The stochastic resume probe reproduced the same sampled batch and identical loss after restore. Fresh model allocations can change a final BLAS rounding bit (observed maximum 2.17e-19), so parameter trajectory equivalence is asserted at a strict 1e-15 tolerance rather than byte identity.
- An intermediate review gate collected and passed 173 tests; the final expanded suite contains 178 passing tests, and all source/test/plot modules compile successfully.
- Legacy checkpoints persist the old finite `causal_mask` buffer in model state. `GPT.load_state_dict()` now rebuilds this deterministic buffer with `-inf` after every load so old files cannot reintroduce future-token leakage.
- After generalized training CE, checkpoint versioning, stochastic resume, and mask migration were integrated, the full warnings-as-errors suite passed 173/173 and `git diff --check` remained clean apart from line-ending notices.
- Final correctness review found that CE backward retained a writable view of integer labels. Targets are now copied during forward, so later caller mutation cannot change the gradient's class selection.
- Restore is now transactional across model, optimizer, scheduler, and NumPy RNG. Module, optimizer, and scheduler loaders validate complete shapes/state before copying, while checkpoint restore snapshots and rolls back all components if any later component fails.
- Regression coverage injects malformed model, optimizer, scheduler, and RNG checkpoint sections and proves every caller-owned state remains unchanged after each failed restore.
- Final public-API review gaps were also closed: Tensor operators now interoperate with NumPy arrays for matmul/reflected division, and LayerNorm/RMSNorm explicitly reject a mismatched final feature dimension instead of silently broadcasting parameters.
- README now warns that pickle checkpoints must be trusted and explicitly marks `plot_loss.py` as a source-checkout-only utility.
- The affected autograd, checkpoint, Transformer, and validation suites pass 127/127 after all review fixes.
- Final complete verification passes 178/178 tests with warnings-as-errors; byte compilation and `git diff --check` both pass.
- Final artifact scan found no build/dist/temp/wheel/checkpoint/egg-info deliverables left behind. Remaining untracked files are intentional project additions (`tests/test_validation.py`, existing `tests/test_modern.py`, and the three planning artifacts).
- A real two-process CLI smoke trained and saved a tiny Llama/AdamW model at step 1, then resumed without `--optimizer`; the second process reported `optimizer=AdamW`, `resume_step=1`, and completed step 2. The temporary checkpoint was removed afterward.
- Build verification temporarily created `.tmp-wheel-audit/`, `build/`, `src/tiny_transformer.egg-info/`, and an ignored `.tmp-wheel-venv/`; all were removed after exact-path verification.
- The combined tracked diff currently spans 19 files; much of it predates this session (Llama/RoPE/SwiGLU/AdamW work), so final review must distinguish and preserve those user changes.
- All four generated build/venv artifact directories were removed after resolving and confirming each absolute path was inside the workspace; they are fully reproducible and no source/user data was deleted.

## Round 2 Findings (inference mode and fully-masked rows)
- The two deferred items were confirmed as genuine gaps, not blockers: the default training and
  generation paths were already correct, so both are additive.
- Evaluation previously built a full backward graph. A measured 4-layer, `d_model=128`, `ctx=64`,
  batch-8 forward-plus-loss ran 10 passes in 1.102 s with recording and 0.586 s under `no_grad()`
  (1.88x), and live `Tensor` objects after a forward pass dropped from 346 to 65 (the model's own
  parameters), so 281 retained nodes per pass disappear.
- Gating recording inside `Tensor.__init__` covers all 18 ops at once because every op constructs
  its result there. Suppressing only tensors that arrive with `_children` reproduces PyTorch's
  rule that explicit leaves keep their `requires_grad`.
- Storing a backward closure only on nodes that can hold a gradient also fixed a latent crash.
  Verified by re-enabling unconditional closure storage: a gradient-less `concat` node used inside
  a gradient-requiring graph raised `TypeError: object of type 'NoneType' has no len()` because
  its `_backward` split a `None` gradient. With the guard, the same graph differentiates correctly.
- A silent no-op was the worst possible response to `backward()` on a tensor built inside
  `no_grad()`, so that specific case now raises. Differentiating a graph built outside the block
  is still allowed, matching PyTorch, and behavior outside any block is unchanged.
- Fully masked rows were confirmed broken before this round: `-inf - -inf` produced NaN weights
  that spread through the entire batch and every gradient. Defining the row as zero weights keeps
  the forward finite and makes the softmax VJP vanish without any backward special case.
- The zero-row convention had to be duplicated in `nn/attention._softmax`, otherwise the
  graph-tracked `forward()` and NumPy `infer()` paths would disagree exactly where a caller is
  most likely to be debugging.
- Unmasked softmax rows are bitwise unchanged: the only difference is shifting by `0` instead of
  `-inf` when a row has no finite maximum, plus clamping a zero normaliser to avoid a
  divide-by-zero warning.
- `cross_entropy` needed the opposite decision. Zero attention weights are meaningful, but a row
  with no finite logit has no distribution at all, so it now raises instead of silently returning
  NaN. The check reuses the row maximum that the stable loss already computes.
- Custom masks were previously unvalidated: an oversized mask silently broadcast the scores to a
  larger output, and NaN or `+inf` entries produced NaN attention. Both now raise, and masks may
  be passed as plain NumPy arrays as well as Tensors.
- A fully masked query row returns exactly the output projection's bias (zero context vector), so
  that position is a constant rather than a prediction; the documentation states it must be
  excluded from the loss rather than trusted.
- Remaining opportunity, addressed in round 3: `GPT.forward` accepted no caller mask, so padding
  masks required using the attention modules directly.

## Round 3 Findings (variable-length batches)
- Hiding padded keys and dropping padded targets are inseparable. With only the mask, padded
  positions still contribute a loss term and gradients; with only `ignore_index`, real tokens still
  attend to padding. Both together make a ragged batch exactly equivalent to unpadded runs.
- Equivalence is now asserted directly rather than assumed: real-token logits match an unpadded
  forward pass to 1e-12, and loss plus every parameter gradient match to 1e-12. A deliberate
  counter-test confirms the logits *do* change without the mask, so the equivalence test cannot
  pass for the wrong reason.
- Combining masks needed no new machinery: the causal slice is a Tensor and the key-padding bias is
  `(batch, 1, 1, time)`, so a plain add broadcasts to `(batch, heads, time, time)`. `-inf + -inf`
  stays `-inf`, and because round 2 rejects `+inf` masks there is no NaN path.
- Round 2's fully-masked-row definition turned out to be load-bearing here: an all-padding row
  makes every query row fully masked. That now yields finite output instead of NaN, so the
  behavior is allowed and tested rather than special-cased.
- The `-inf` logits check had to be narrowed to *scored* rows. An ignored position may legitimately
  hold anything, including a fully masked row, so checking all rows would reject valid batches.
- Dividing by the scored count rather than the total matters more than it looks: with the total,
  loss would shrink as padding grows and gradient magnitude would depend on batch shape rather
  than content.
- The no-`ignore_index` path keeps a fast route (no row gathering, no scatter), so the common
  training call does not pay for the feature.
- Right padding is a real constraint, not an oversight: positions start at 0, so left padding would
  shift every learned and rotary position. Fixing that properly means threading position ids
  through the embeddings, RoPE, and the KV cache — which is also why `infer`/`generate` still take
  no mask.
- Remaining opportunity, addressed in round 4: batched generation from padded prompts (per-row
  position offsets plus cache bookkeeping).

## Round 4 Findings (batched generation from ragged prompts)
- Training and generation want opposite paddings. Training reads every slot and numbers position i
  at slot i, so it pads right; decoding reads slot -1 for the next distribution, so every row's
  newest token must sit there, which means padding left. Supporting one padding for both would
  have been wrong for whichever path lost.
- Causality did not need per-row handling. Slot order still encodes arrival order, so
  `_causal_mask` is unchanged; only *absolute positions* (learned embeddings and RoPE) and *which
  slots are real* differ per row.
- The mask had to be defined over all keys, not the current step. Padded prompt slots live in the
  KV cache for the rest of the run, so a per-step mask would unhide them right after the prefill.
  `GPT.infer` therefore validates `attention_mask` against `(batch, past + time)`.
- RoPE needed genuine per-element positions. The old scalar `offset` assumed every row sat at the
  cache length; `rotate_np(positions=…)` gathers cos/sin rows so indexing appends the rotation
  axis and the result broadcasts against `(batch, heads, time, d_k)` directly. A regression test
  pins the explicit form to the scalar form for a uniform batch.
- Deriving `position_ids` inside `generate` (`cumsum(mask) - 1`, clamped at 0) removes the main
  caller mistake; padded slots collapse to position 0, which is harmless because they are never
  attended to.
- Equivalence is bitwise, not approximate: the measured maximum difference between a padded row's
  last-position logits and that prompt's own inference run is exactly 0.0 for both architectures.
  Masked keys contribute `exp(-inf) = 0`, so they add exact zeros to both the softmax denominator
  and the weighted value sum.
- Round 2's fully-masked-row definition is load-bearing a second time: a left-padded row's first
  query attends only to padding. Without the zero-weight definition the prefill would be NaN and
  the entire scheme would be impossible.
- Refusing to run when prompt + `max_new_tokens` exceeds `context_len` was the honest choice: the
  sliding-window crop resets the cache and renumbers slots, so per-row positions would silently
  drift. The unmasked path keeps its existing cropping behavior untouched.
- Beam search takes no mask by construction — it already decodes one sequence at a time, so there
  is nothing to pad; passing one now raises instead of being ignored.
- Remaining opportunities, none blocking: exposing ragged prompts through the training CLI (which
  currently samples fixed-length contiguous chunks), and sliding-window decoding for masked runs.

## Round 5 Findings (gradient checkpointing)
- Round 2's `no_grad`/`enable_grad` pair turned out to be exactly the primitive this needs: the
  unrecorded forward is `with no_grad()`, and the replay is `with enable_grad()` inside the backward
  closure. No new machinery in the engine was required.
- The replay must differentiate *copies* of the inputs. Differentiating into the original tensor
  looked simpler and is wrong: `Tensor.backward` resets the gradient of any node that has parents,
  so a replay reaching the outer graph's node would wipe contributions that an earlier consumer had
  already accumulated. A residual connection makes this happen on the very first block, which is
  why there is a dedicated regression test for it.
- Dropout would have silently broken correctness. The replay draws fresh masks, so it would
  differentiate a different function than the forward computed. Capturing `np.random.get_state()`
  before the unrecorded forward and replaying it fixes that; restoring the backward-time state
  afterwards keeps the surrounding training loop's stream untouched, which is what makes an
  identical trajectory possible rather than merely a similar one.
- Measured tradeoff on a 6-layer, `d_model=128`, `ctx=64`, batch-8 step: retained activations fall
  from 320.9 MiB (413 tensors) to 14.2 MiB (29 tensors) — 22.7x less — for 227 ms/step to
  342 ms/step, 1.51x slower. That is the textbook "one extra forward pass" cost.
- Correctness is bit-level, not approximate: five training steps with dropout 0.2 produce the same
  losses to 1e-14, and two CLI runs with the same seed report identical losses *and* gradient norms.
- `grad_checkpoint` is deliberately not part of `config()`. It changes memory and time but never
  weights, outputs, or gradients, so persisting it in a checkpoint would wrongly pin a machine's
  memory budget to a model file; `train.py` applies the flag after construction or resume instead.
- Naming mattered: `engine/checkpoint.py` already means on-disk training state, so the new module is
  `engine/recompute.py` and the docs use "gradient checkpointing (activation recomputation)".
- Remaining opportunities: sliding-window decoding for masked runs, and multi-output/multi-section
  `recompute` (today it wraps functions returning a single Tensor, which is all the block structure
  needs). Ragged prompts in the training CLI were addressed in round 6.

## Round 6 Findings (document corpora)
- Rounds 3 and 4 built masking and `ignore_index` but nothing in the shipped CLI used them; the
  default trainer still sampled fixed-length windows from one token stream. This round makes the
  CLI a real consumer of that path rather than leaving it library-only.
- Routing both corpus kinds through one `(tokens, targets, mask)` sampler kept the change small:
  the stream path passes `mask=None` and consumes the RNG in exactly the same order, which is why
  the previously recorded trajectory reproduces step for step.
- Building the tokenizer before splitting documents was a real bug, caught by comparing modes: the
  same corpus gave a char vocab of 26 through JSONL and 19 through `lines`, because `{`, `"`, `:`
  and friends were being learned as tokens. The tokenizer is now built from the joined documents.
- The same mistake extended to generation: the default prompt was the first `context_len`
  characters of the *file*, which for JSONL is `{"body": ...` — nonsense as a prompt and possibly
  outside the vocabulary. Prompts now default to the corpus text.
- Truncating to `context_len + 1` rather than `context_len` is what keeps a full-length document
  usable: the input drops the last token and the target drops the first, so both are exactly
  `context_len` long.
- Documents shorter than two tokens have no (input, target) pair at all. They are dropped and the
  count is reported rather than silently ignored, and a corpus with fewer than two usable documents
  is refused outright.
- The equivalence test is the one that matters: a padded batch's loss equals the scored-position
  mean of the per-document losses to 1e-12, and both loss and gradients are bit-identical when the
  padded slots are overwritten with junk.

## Round 7 Findings (handoff documentation)
- The planning artifacts record *how the work happened*, chronologically. That is the wrong shape
  for a cold start, which needs *what is true now*. Hence two new documents rather than a longer
  `progress.md`: `CLAUDE.md` for rules that apply to every edit, `PROJECT_STATE.md` for the state
  snapshot read on demand.
- Writing the invariants down exposed how many of them are cross-round couplings rather than
  file-local rules. Round 2's zero-weight masked row is load-bearing in rounds 3 and 4; round 2's
  `no_grad`/`enable_grad` pair is exactly what round 5's replay needs. A future edit that "simplifies"
  the softmax row definition would break two later features with no local test failure, so each
  invariant is stated with the failure it prevents.
- The round-6 stream-path regression cited exact numbers (`train_loss=3.6031/3.5854`) but never
  recorded the command that produced them. Four candidate hyperparameter sets were tried and none
  reproduced those values, so the check was not rerunnable from a cold start. Replaced with a named
  anchor command whose output is recorded verbatim, verified reproducible run to run.
- That anchor also re-confirmed the round-5 claim at CLI level: adding `--grad-checkpoint` leaves all
  three step lines — losses, perplexity, learning rate, and gradient norms — byte-identical.
- `evaluate_batches` takes an unweighted mean over per-batch losses. For the token stream every batch
  scores the same number of positions, so this is exact; for document corpora each batch is already a
  scored-position mean, so batches with different real-token counts are weighted equally. Recorded as
  a ranked next candidate rather than fixed here, since changing it changes reported numbers.
- Committing as one checkpoint was the honest option: the pre-existing user diff (Llama/RoPE/SwiGLU/
  AdamW) and six rounds of work touch the same files, so splitting the history after the fact would
  risk misattributing or dropping user changes.

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| Existing implementation changes overlap likely improvement areas | Inspect both working-tree content and diffs before editing; build on rather than overwrite user changes. |
| PowerShell repository reads timed out at 10 seconds | Increase command budget and split large README/diff output into smaller inspections. |
| Initial read commands need more than 10 seconds in this environment | Retried successfully with a 30-second command budget. |
| Autograd baseline passes despite uncovered edge cases | Add focused failing probes before choosing fixes; current tests do not cover full NumPy matmul broadcasting/vector semantics or repeated graph backward. |
| Direct root-level Python invocation does not import `engine` | Tests manipulate their import path; diagnostics must explicitly add `src`, and packaging/import ergonomics merit verification. |
| Wheel build fails before metadata/package-content inspection | Correct the invalid backend as part of the selected scope, then rebuild into the already isolated audit directory. |
| Patches must not match rendered non-ASCII source text from PowerShell | Use narrow ASCII-only context because console glyph corruption does not reflect the UTF-8 file bytes. |

## Resources
- Project root: `C:\Users\User\Desktop\All project\computer-science-labs\Tiny Transformer & Autograd`

## Visual/Browser Findings
- No browser or visual inspection performed.

---
*Update after every two view/search operations.*
