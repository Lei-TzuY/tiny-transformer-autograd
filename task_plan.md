# Task Plan: Continue Tiny Transformer & Autograd

## Goal
Assess the current project honestly, implement the highest-value missing capabilities and quality improvements, and leave the repository with verified behavior, tests, and clear documentation.

## Current Phase
Complete (round 9: inference-mode and mask-contract hardening)

## Phases

### Phase 1: Baseline & Discovery
- [x] Inventory repository structure, documentation, and current API surface
- [x] Run the existing test/lint/type-check baseline
- [x] Audit autograd, neural-network/Transformer, and user-facing completeness
- [x] Record concrete gaps and risks in `findings.md`
- **Status:** complete

### Phase 2: Prioritization & Design
- [x] Rank gaps by educational value, correctness risk, and implementation cost
- [x] Select a coherent, bounded improvement set
- [x] Define acceptance criteria and regression tests
- **Status:** complete

### Phase 3: Implementation
- [x] Implement the selected autograd/core improvements
- [x] Make standalone attention causal by default and preserve forward/infer parity
- [x] Preserve optimizer identity across checkpoint save/resume
- [x] Preserve NumPy RNG state across stochastic checkpoint resume
- [x] Repair PEP 517 packaging and the installed `tiny-train` entry point
- [x] Add low-cost public API validation for invalid dimensions, probabilities, token inputs, and generation bounds
- [x] Update examples and documentation where behavior changes
- [x] Add focused regression and integration tests
- **Status:** complete

### Phase 4: Verification & Hardening
- [x] Run focused tests during implementation
- [x] Run the complete available quality suite
- [x] Inspect the final diff for unintended or unrelated changes
- [x] Resolve discovered regressions or document genuine limitations
- **Status:** complete

### Phase 5: Delivery
- [x] Summarize what was improved and why
- [x] Report verification evidence and remaining opportunities
- [x] Ensure planning artifacts reflect the completed state
- **Status:** complete

### Phase 6: Round 2 — Inference mode and fully-masked attention rows
- [x] Add a formal `no_grad` / `enable_grad` / `set_grad_enabled` recording switch
- [x] Suppress parents, gradient buffers, and backward closures for detached op results
- [x] Keep explicitly created leaves trainable so models can be built inside a block
- [x] Make an unusable `backward()` call inside a disabled block fail loudly
- [x] Apply the switch to the training script's validation loop
- [x] Define fully-masked (`all -inf`) softmax rows as zero attention weights in both
      the autograd and NumPy inference paths
- [x] Validate caller-supplied attention masks (broadcast shape, NaN/`+inf`)
- [x] Reject a cross-entropy logits row with no finite class
- [x] Add regression tests for both features and update README/CI-visible counts
- **Status:** complete

### Phase 7: Round 3 — Variable-length batches
- [x] Add `ignore_index` to `cross_entropy` with zero gradient at ignored positions
- [x] Divide the loss by the scored-position count and reject an all-ignored batch
- [x] Add `GPT.forward(idx, attention_mask=...)` combining key padding with the causal mask
- [x] Validate the mask shape, dtype, and 0/1 values
- [x] Prove padded keys are invisible: real-token logits and every parameter gradient match an
      unpadded run, and padded content cannot change either
- [x] Add an end-to-end ragged-batch training integration test
- [x] Document the contract, the right-padding requirement, and the generation limitation
- **Status:** complete

### Phase 8: Round 4 — Batched generation from ragged prompts
- [x] Add explicit per-element RoPE positions for the inference path
- [x] Thread an additive key bias through block/attention inference
- [x] Extend `GPT.infer` with `attention_mask` (covering cached keys) and `position_ids`
- [x] Add `generate(..., attention_mask=…)` deriving per-row positions from a left-padded mask
- [x] Enforce the left-padding contract and the context-window bound instead of assuming them
- [x] Prove a row's generated tokens are bitwise identical to generating that prompt alone, for
      both learned positions and RoPE, cached and uncached
- [x] Update the README section that documented the old limitation
- **Status:** complete

### Phase 9: Round 5 — Gradient checkpointing
- [x] Add `engine/recompute.py`: run a section unrecorded, replay and differentiate it in backward
- [x] Replay from detached copies so other consumers' gradients survive
- [x] Capture and replay the NumPy RNG state so dropout masks match, then restore the stream
- [x] Expose it per block on `GPT` as a runtime toggle plus a `--grad-checkpoint` CLI flag
- [x] Prove forward values, gradients, and full training trajectories are unchanged, with dropout
- [x] Measure the memory/time tradeoff and document it honestly
- **Status:** complete

### Phase 10: Round 6 — Document corpora in the training CLI
- [x] Add `--data-format text|lines|jsonl` and `--jsonl-field` with validation
- [x] Parse, encode, truncate, and drop unusable documents with a reported count
- [x] Sample right-padded batches carrying an attention mask and ignored padding targets
- [x] Route training and validation through one sampler interface shared by both corpus kinds
- [x] Build the tokenizer from document text rather than the raw file
- [x] Prove a padded batch's loss and gradients equal scoring each document alone
- [x] Confirm the default token-stream path is unchanged, step for step
- **Status:** complete

### Phase 11: Round 7 — Handoff documentation and checkpoint
- [x] Add `CLAUDE.md`: commands, layout, conventions, verification style, and the
      numbered list of invariants that must not break
- [x] Add `PROJECT_STATE.md`: architecture map, per-round completed work, design
      decisions, test baseline, measured tradeoffs, limitations, next-round candidates
- [x] Record a reproducible CLI regression anchor with its exact expected output
- [x] Update `task_plan.md`, `findings.md`, and `progress.md` to the delivered state
- [x] Verify the full suite, byte compilation, and diff hygiene
- [x] Create the checkpoint commit preserving the pre-existing user diff
- **Status:** complete

### Phase 12: Round 8 — Sliding-window decoding for masked runs
- [x] Remove the refusal when prompt + `max_new_tokens` exceeds `context_len`
- [x] Crop the shared array per window and renumber the surviving real tokens from 0
- [x] Slice the cached-step mask to the cache length so it still covers cached + current keys
- [x] Accept a prompt that is already longer than `context_len`
- [x] Prove a cropped masked run still matches generating each prompt alone, both
      architectures, cached and uncached, and far past the window
- [x] Anchor the renumbering against a direct `infer` call that never touches `generate`
- [x] Prove unmasked and in-window masked generation are byte-identical to round 7
- [x] Measure the cost of decoding past the window and record it as the next candidate
- **Status:** complete

### Phase 13: Round 9 — Inference-mode and mask-contract audit
- [x] Verify the existing `no_grad` and fully-masked-row implementations against their documented contracts
- [x] Audit every public inference/evaluation entry point for accidental graph construction
- [x] Audit custom attention-mask normalization, broadcasting, and fully-masked behavior across training and cached inference
- [x] Implement any confirmed lifecycle or mask-contract gaps without changing default numerical behavior
- [x] Add focused regression/integration tests and synchronize public documentation
- [x] Run the complete quality, packaging, and artifact-hygiene checks
- **Status:** complete

## Key Questions
1. Which correctness or capability gaps remain in the current autograd engine?
2. Which Transformer features are missing for a coherent tiny-but-real implementation?
3. What do the existing tests fail to cover, especially edge cases and numerical correctness?
4. What improvement set gives the best value without turning a teaching project into a large framework?

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Audit before selecting changes | The request is intentionally broad; repository evidence should determine what is worth implementing. |
| Preserve the project's educational scale | Improvements should make the tiny implementation more correct and usable without obscuring its core ideas. |
| Verify behavior with tests and numerical checks | Autograd and attention code can appear plausible while silently producing incorrect gradients or masking behavior. |
| Implement a correctness-and-reliability hardening set | Stable CE, NumPy-compatible matmul gradients, repeatable backward, real division, axis/generator/stability fixes, causal parity, optimizer-aware checkpoints, and working packaging address confirmed defects without bloating the teaching model. |
| Make checkpoint restoration transactional | A malformed later section must not leave any caller-owned model, optimizer, scheduler, or RNG state partially restored. |
| Defer broad framework features such as a full no-grad system | They are useful, but confirmed correctness and delivery failures have higher value in this bounded pass. |
| Round 2: gate recording in the `Tensor` constructor | Every op funnels through it, so one thread-local check covers all 18 primitives with no per-op edits and no duplicated logic. |
| Round 2: suppress only op results, never explicit leaves | Matches PyTorch and avoids the trap where building a model inside `no_grad()` silently produces an untrainable model. |
| Round 2: drop the backward closure whenever a node cannot hold a gradient | Releases captured intermediates (the actual memory win) and removes a latent crash where a gradient-less node was still asked to propagate. |
| Round 2: define a fully masked row as zero attention weights | It is the standard masked-attention convention, keeps forward and backward finite, and is far more useful than a NaN or a hard error for padded positions. |
| Round 2: keep `cross_entropy` strict about all-`-inf` rows | Zero weights are meaningful for attention; a loss over an impossible distribution is not, and a silent NaN loss would corrupt every weight. |
| Round 3: expose a `(batch, time)` keep/pad mask rather than a raw score bias | Callers should not have to build a `(B, 1, T, T)` additive tensor; the model already owns the causal mask and can combine the two correctly. |
| Round 3: pair the mask with `cross_entropy(ignore_index=…)` | Hiding padded *keys* is only half the job — padded *targets* would still be scored, so the two must ship together to make a ragged batch equivalent to unpadded runs. |
| Round 3: divide by the scored count, not the total | Otherwise the loss silently shrinks as padding grows and gradients scale with batch shape rather than content. |
| Round 3: require right padding instead of adding position offsets | Position ids would have to be threaded through embeddings, RoPE, and the KV cache; documenting the requirement keeps the round bounded and honest. |
| Round 3: keep `infer`/`generate` mask-free for now | Batched padded generation additionally needs per-row position offsets and cache bookkeeping; a half-working mask there would be worse than none. |
| Round 4: left-pad for generation, right-pad for training | Decoding reads slot -1, so every row's newest token must sit there; training assigns position i to slot i. The two paddings serve different reads, and each is enforced where it applies. |
| Round 4: make `infer`'s mask cover cached keys, not just current ones | Padded prompt slots stay in the KV cache forever, so a mask describing only the current step would silently unhide them after the prefill. |
| Round 4: derive `position_ids` inside `generate` rather than asking the caller | `cumsum(mask) - 1` is the only correct answer for a left-padded row, and deriving it removes a whole class of caller mistakes. |
| Round 4: keep per-element RoPE positions optional | The scalar-offset path stays byte-identical for unmasked decoding, and a regression test pins the two forms together. |
| Round 4: require a masked run to fit in `context_len` | The sliding-window crop resets the cache and renumbers slots, which would invalidate per-row positions; refusing is honest, silently drifting is not. |
| Round 5: name the module `recompute`, not `checkpoint` | `engine/checkpoint.py` already means "save training state to disk"; reusing the word for activation recomputation would make both harder to read. |
| Round 5: replay from detached copies of the inputs | Differentiating into the original would let the replay's `backward()` reset a node that has parents in the outer graph, discarding a residual connection's already-accumulated gradient. |
| Round 5: capture and replay the RNG state | Otherwise the replay draws different dropout masks and differentiates a different function than the forward pass computed. Restoring afterwards keeps the training trajectory bit-identical. |
| Round 5: treat `grad_checkpoint` as a runtime toggle, not architecture | It changes memory and time, never weights or outputs, so it stays out of `config()` and can be flipped on a resumed run. |
| Round 5: always record a node when grad is enabled | The wrapped function's own parameters may need gradients even when none of its inputs do, and the closure cannot inspect them; `no_grad()` remains the way to ask for no graph. |
| Round 6: make the CLI use the ragged-batch path rather than adding a parallel one | Rounds 3 and 4 built masking and `ignore_index`; a document corpus is exactly their intended consumer, so the CLI exercises the same code the library exposes. |
| Round 6: one sampler interface returning `(tokens, targets, mask)` | The stream path passes `mask=None` and stays byte-identical, while the document path adds the mask, so training and evaluation have a single code path instead of two. |
| Round 6: split documents before building the tokenizer | Training the tokenizer on raw JSONL puts braces and quotes in the vocabulary; the observed char vocab was 26 instead of 19 until this was fixed. |
| Round 6: default the generation prompt to the corpus, not the file | With JSONL the raw file starts with `{"body": ...`, which is both a nonsense prompt and possibly outside the vocabulary. |
| Round 7: split handoff docs into `CLAUDE.md` and `PROJECT_STATE.md` | `CLAUDE.md` is loaded into every session, so it holds only rules and invariants; the detailed state snapshot belongs in a file read on demand. |
| Round 7: state invariants as a numbered list with the failure they prevent | Most traps here are interactions between rounds, not bugs inside one file; a rule without its reason gets refactored away. |
| Round 7: pin a CLI regression anchor with exact expected output | The round-6 stream-path check referenced numbers whose command was never recorded, so it could not be rerun from a cold start. A named command plus its output can. |
| Round 7: commit as a checkpoint rather than splitting the history | The pre-existing user diff (Llama/RoPE/SwiGLU/AdamW) is interleaved with six rounds of work across the same files; reconstructing separate commits would risk misattributing or losing it. |
| Round 8: crop the shared array rather than per row | Left padding already puts every row's newest token at slot −1, so cropping the last `context_len` columns is simultaneously correct for every row — one that has not filled the window loses padding, one that has loses its oldest tokens. Per-row cropping would produce ragged rows that no longer share a cache. |
| Round 8: renumber positions from 0 inside each window | Carrying absolute numbering across a crop immediately exceeds `context_len`. Renumbering also matches what the unmasked path already does by re-prefilling with default positions, so masked and unmasked runs stay one behaviour rather than two. |
| Round 8: recompute positions from the cropped mask, never track an offset | An offset has to be maintained correctly at every crop *and* every cached step; `cumsum` over the mask that is actually being passed to `infer` is derived from the state that decides the answer, so the two cannot drift apart. |
| Round 8: keep the existing cache-drop trigger | Dropping a full cache is exactly what forces the re-prefill that renumbers the window. Reusing it means the masked path inherits an already-tested reset rule instead of adding a second one. |
| Round 8: accept prompts longer than `context_len` | It falls out of cropping rather than being added: the first prefill already crops, and the unmasked path has always accepted them. Refusing would have been an extra rule with no reason behind it. |
| Round 8: test by subclassing the round-4 test class | The claim is that round 4's equivalence survives the crop, not a new claim, so re-running the same assertions with a small `context_len` verifies exactly that and cannot drift from the original wording. |
| Round 9: remember whether an op result was detached by grad mode | Current mode at `backward()` time cannot reveal how a tensor was created; provenance lets a delayed misuse raise while preserving the project's intentional no-op for explicit constants. |
| Round 9: make each guard's restoration stack thread-local | The enabled flag is already per-thread, but a shared guard instance can interleave exits; its saved previous modes must have the same isolation. |
| Round 9: share the additive-mask contract across forward and standalone inference | Both public paths accept additive masks and promise the same fully-masked behavior, so shape and value validation must not stop at the graph path. |
| Round 9: restore evaluation mode in `finally` | Validation failures must not leak `eval()` into the caller's subsequent training loop. |
| Round 9: interpret a 3-D multi-head mask as `(B,T,T)` | This is the documented public shape; per-head masks remain available unambiguously as `(1,H,T,T)` or `(B,H,T,T)`. |
| Round 9: enforce right padding at training forward | Learned absolute positions make left/interior padding numerically different from an unpadded sequence, so silently accepting it violates the documented equivalence. |
| Round 9: validate caller-provided KV caches before using their length | Mask and position validation depend on one coherent past length; malformed per-layer structures should fail at the public boundary, not deep in NumPy broadcasting. |
| Round 9: reject async/generator decorators explicitly | A synchronous wrapper disables recording only while creating a lazy coroutine/generator, not while its body runs; a clear unsupported error is safer than silently recording a graph. |
| Round 9: require finite real-valued KV caches at every public inference boundary | Correct shapes are insufficient: object arrays fail opaquely and NaNs poison logits, so standalone attention and GPT should reject both before attention math. |

## Acceptance Criteria
- Matmul gradients pass finite differences for singleton batch broadcasting and all 1-D NumPy matmul combinations.
- Cross entropy matches a log-sum-exp reference at extreme logits and rejects malformed targets.
- Repeated backward preserves leaf accumulation without reusing stale intermediate gradients; deep graphs do not recurse through Python call depth.
- Tensor division handles negative/broadcast denominators and reverse division with correct gradients.
- Negative-axis transpose, generator concat, and extreme sigmoid cases are regression-tested.
- Attention `forward(mask=None)` matches causal inference.
- Causal masks use true negative infinity so arbitrarily large future scores cannot leak.
- New checkpoints restore the saved optimizer algorithm; mismatches are explicit and old checkpoints remain readable.
- New checkpoints restore NumPy RNG state while old checkpoints remain readable.
- Failed checkpoint restores leave model, optimizer, scheduler, and RNG state unchanged.
- A wheel builds, contains `engine`, `nn`, `train`, `tokenizer`, and `benchmark`, and its `tiny-train` entry point executes.
- Invalid dropout/optimizer hyperparameters, attention dimensions, RoPE bounds, token indices, context lengths, and generation lengths fail early with clear exceptions.
- The complete test suite remains green and documentation matches the delivered CLI/API.
- Round 2: op results created under `no_grad()` have no parents, no gradient buffer, and no
  backward closure, and their forward intermediates become collectable.
- Round 2: leaves created with `requires_grad=True` inside a block stay trainable, and a model
  built there can still be trained afterwards.
- Round 2: blocks nest, restore on exception, work as decorators, and are thread-local.
- Round 2: `backward()` on a detached tensor inside a disabled block raises; differentiating a
  graph built outside the block still works.
- Round 2: forward values are identical with and without recording.
- Round 2: an all-`-inf` softmax row yields zero weights and zero gradient, unmasked rows are
  bitwise unchanged, and the autograd and inference softmax agree on masked rows.
- Round 2: a fully masked attention row returns exactly the output projection bias and blocks
  gradients to Q/K/V.
- Round 2: malformed custom masks (too large, non-broadcastable, NaN, `+inf`) raise explicitly,
  while NumPy arrays and per-head mask shapes are accepted.
- Round 3: with `attention_mask`, a real token's logits equal the unpadded run's logits and are
  unchanged by rewriting the padded slots; without the mask they demonstrably differ.
- Round 3: loss and every parameter gradient from a padded batch with `ignore_index` match an
  unpadded run to 1e-12; ignored positions have exactly zero gradient.
- Round 3: the mask applies in every layer, boolean and 0/1 masks agree, an all-padding row stays
  finite, and malformed masks raise.
- Round 3: an all-`ignore_index` batch raises, a non-integer `ignore_index` raises, out-of-range
  scored targets still raise, and `ignore_index` rows may hold masked logits.
- Round 3: a ragged batch trains end to end (loss at least halves) and its value is bit-identical
  under scrambled padding.
- Round 4: a left-padded batch's prefill logits at each row's last real token match that prompt's
  own inference run, for both learned positions and RoPE.
- Round 4: each row's generated tokens are identical to generating that prompt alone, with cached
  and uncached decoding agreeing and prompt columns returned unchanged.
- Round 4: scrambling the padding changes nothing, under greedy and seeded sampling.
- Round 4: a fully unmasked single-row call is identical to passing no mask at all.
- Round 4: right-padded masks, all-padding rows, runs exceeding `context_len`, and beam search with
  a mask all raise; `position_ids` and RoPE positions are shape/dtype/range validated.
- Round 4: explicit RoPE positions equal the scalar-offset path for a uniform batch.
- Round 5: a recomputed section's forward values, input gradients, and parameter gradients equal a
  plain call's, including when the input feeds a residual connection.
- Round 5: model-level checkpointing leaves forward values identical, gradients equal to 1e-14 with
  and without dropout, and a five-step training trajectory identical.
- Round 5: the section's intermediates are collectable right after the forward pass, and fewer
  tensors are retained than without checkpointing.
- Round 5: the random stream after backward is exactly where it was before, and the replay uses the
  forward pass's dropout mask.
- Round 5: `recompute` is a plain call under `no_grad()`, accumulates over repeated backward,
  validates its arguments, works with a LoRA-frozen backbone, and stays out of `config()`.
- Round 5: two CLI runs with the same seed, one with `--grad-checkpoint`, report identical losses
  and gradient norms.
- Round 6: `lines` and `jsonl` parsing skip blank documents, select the requested field, and report
  the offending line number for malformed input.
- Round 6: batches are right-padded, targets are the inputs shifted by one, padded targets hold
  `ignore_index`, and sampling works when the batch exceeds the corpus size.
- Round 6: a padded batch's loss equals the scored-position mean of per-document losses, and both
  loss and gradients are unchanged when the padding content is scrambled.
- Round 6: the CLI trains on both formats, saves checkpoints, reports document counts, rejects a
  corpus with fewer than two usable documents, and builds the vocabulary from document text only.
- Round 6: the `text` corpus path reproduces the previously recorded trajectory exactly.
- Round 7: a session with no prior context can name the test command, the module layout, the
  invariants, the limitations, and the next candidates from the checked-in documents alone.
- Round 7: the recorded CLI anchor reproduces run to run and is identical with
  `--grad-checkpoint`.
- Round 7: the full suite passes with warnings as errors, sources compile, and the checkpoint
  commit contains the pre-existing user diff alongside the six rounds of work.
- Round 8: a masked run whose length exceeds `context_len` completes, and every row's generated
  tokens still equal generating that prompt alone — both architectures, cached and uncached,
  and with the run continued far past the window.
- Round 8: a prompt that is already longer than `context_len` is cropped per row and matches
  the same prompt decoded alone.
- Round 8: the run provably reaches the crop (output width exceeds `context_len`), scrambling
  the padding changes nothing, and dropping the mask demonstrably changes the answer.
- Round 8: the next token after a crop equals what `infer` predicts from that row's last
  `context_len` real tokens fed unpadded, a path that never enters `generate`.
- Round 8: unmasked generation and masked generation inside the window are byte-identical to
  the previous round across architectures, cache modes, strategies, and over-long prompts.
- Round 8: the CLI regression anchor reproduces exactly.
- Round 9: an op result created under `no_grad()` raises on `backward()` both inside and after
  the disabled scope; explicit constant leaves remain a no-op, and an outer recorded graph may
  still be differentiated inside `no_grad()`.
- Round 9: one guard instance can be shared by interleaving threads without either thread restoring
  the other's previous mode; ordinary nesting, recursion, decorators, and exception restoration stay green.
- Round 9: standalone attention inference rejects non-broadcastable, oversized, NaN, and `+inf`
  key biases with explicit errors, while valid broadcast shapes and all-`-inf` rows retain the
  documented zero-weight/output-bias behavior.
- Round 9: evaluation restores the caller's exact training/eval mode even when sampling or model
  execution raises, and grad recording is enabled again afterward.
- Round 9: constant-only op results retain no parents, and delayed no-grad errors do not prevent
  such constants from remaining intentional backward no-ops.
- Round 9: documented `(B,T,T)` multi-head masks behave exactly like `(B,1,T,T)`, including when
  `batch == heads`; explicit 4-D per-head masks remain supported.
- Round 9: `GPT.forward` rejects left padding and interior holes while accepting right padding and
  all-padding rows, and `GPT.infer` rejects malformed or inconsistent per-layer KV caches before
  deriving mask/position shapes.
- Round 9: decorator use on coroutine, generator, or async-generator functions fails immediately
  with a clear sync-only message instead of appearing to disable recording while doing nothing.
- Round 9: standalone Self/MultiHead attention and GPT reject cache entries that are malformed,
  non-real/non-numeric, or non-finite, while caches returned by valid inference remain accepted.
- Round 9: all pre-existing numerical paths remain bitwise unchanged for valid inputs, and the
  complete warnings-as-errors suite, byte compilation, packaging smoke, and diff checks pass.

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| Two read-only PowerShell inspections exceeded a 10-second command timeout | 1 | Logged immediately; retry with a longer timeout and smaller output slices because shell startup is consuming most of the budget. |
| Inline autograd probe could not import `engine` from repository root | 1 | Do not repeat unchanged; prepend the resolved `src` directory to `sys.path` inside the diagnostic script. |
| PEP 517 wheel build failed because backend `setuptools.backends.legacy` cannot be imported | 1 | Treat as a packaging defect; change to `setuptools.build_meta` and verify the wheel after scope selection. |
| Planning-file update patch used a non-exact expected test-table row | 1 | Re-read all planning files and retry with exact current context. |
| Multi-file API-validation patch matched a Unicode docstring through a garbled console representation | 1 | No source changes were applied; retry as small per-file patches anchored only on stable ASCII code. |
| Policy rejected a verified recursive PowerShell cleanup script | 1 | Used native PowerShell with one explicit, pre-verified literal workspace path at a time; cleanup succeeded. |
| Stochastic resume regression required bitwise-equal parameters across fresh NumPy allocations | 1 | Batches and loss were exact; one parameter differed by 2.17e-19 from BLAS rounding. Use a strict 1e-15 numerical tolerance while retaining exact RNG batch/loss assertions. |
| Final-review planning update used an overly broad multi-file patch context | 1 | Re-read exact heading/row locations and apply narrow, context-accurate updates. |
| Final status grep used a literal newline without ripgrep multiline mode | 1 | Rerun with a simple status-only expression that does not span lines. |
| Round 2 decorator form raised `TypeError` because `self.__class__(self.mode)` does not fit the zero-argument `no_grad`/`enable_grad` subclasses | 1 | Construct the base `set_grad_enabled(mode)` guard inside the wrapper instead of re-instantiating the subclass. |
| Round 2 mask-validation test expected "larger than" for a `(1, 2, 4, 4)` mask | 1 | That is exactly the score shape and therefore valid; used `(2, 2, 4, 4)` for the too-large case and `(5, 5)` for the non-broadcastable case. |

| Phase 13 planning patch matched a console-garbled em dash in the phase heading | 1 | Re-read exact headings with `rg` and patch against stable section boundaries instead. |
| Batched plan/test inspection treated ripgrep's normal no-match exit code as a fatal batch error | 1 | Preserve partial results and rerun the plan read separately; use an explicit no-match handler for optional searches. |
| Round-9 multi-document patch used a non-exact wrapped sentence in `PROJECT_STATE.md` | 1 | Split the documentation update per file and anchor the round insertion on stable ASCII headings. |
| Second `PROJECT_STATE.md` batch patch matched a rendered Unicode separator in the baseline line | 1 | Use smaller ASCII-only patches and replace the baseline paragraph via surrounding headings. |
| Optional `PROJECT_STATE.md` cleanup again depended on a Unicode round heading and wrapped context | 1 | Stop matching the heading; keep the harmless section separator and patch only ASCII contract text. |
| Installed-wheel smoke used a dependency-free venv, so importing the package could not find NumPy | 1 | Keep the already verified wheel, recreate only the generated venv with system site packages, and rerun imports/CLI without downloading dependencies. |
| Policy rejected a script combining computed-path verification with recursive venv deletion | 1 | Resolve the exact generated venv read-only first, then use a separate explicit literal-path PowerShell removal before recreating it. |
| Installed CLI help was piped through `Select-Object -First`, closing stdout early and yielding exit -1 after valid output | 1 | Rerun the installed executable with complete help redirected to null, then print an independent success marker. |
| Cache-hardening planning patch used a non-exact progress-log row | 1 | Split the update by file and inspect the exact progress row before patching it. |
| Final artifact scan included pytest's ignored `.pytest_cache` and failed after an otherwise clean status | 1 | Resolve the exact cache directory, remove that generated test cache explicitly, then rerun the artifact/status scan. |

## Notes
- Existing user changes must be preserved and unrelated files left untouched.
- The initial worktree is dirty in nearly every likely implementation area; inspect diffs before any source edit.
- Re-read this plan before committing to the implementation scope.
- Log every command/test failure and use a different approach on retries.
