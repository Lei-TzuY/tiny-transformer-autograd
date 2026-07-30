# Project State

Handoff snapshot for `Tiny Transformer & Autograd`. Written so a session with no prior
context can resume without re-deriving anything.

**Last updated:** 2026-07-31 · **Branch:** `main` · **Suite:** 301 passing (`-W error`)

Companions: `CLAUDE.md` (working rules and invariants), `task_plan.md` (phases,
decisions, acceptance criteria), `findings.md` (per-round findings), `progress.md`
(verification log, test results, error log).

---

## 1. What exists

Pure-NumPy reverse-mode autograd engine plus a decoder-only Transformer that trains and
generates text. NumPy is the only runtime dependency. Python ≥ 3.10; CI runs 3.10/3.11/3.12
on Linux, development is on Windows.

| Path | Lines | Responsibility |
|------|-------|----------------|
| `src/engine/tensor.py` | 311 | `Tensor`: data + grad, `_children`, `_backward` closures, iterative topological `backward()`, operators, `detach`. Recording is gated here. |
| `src/engine/ops.py` | 593 | 18 differentiable ops: add/mul/div/matmul, relu/sigmoid/exp/log/tanh/gelu/silu, softmax, cross_entropy, sum/mean, reshape/transpose/concat. |
| `src/engine/grad_mode.py` | 108 | `no_grad` / `enable_grad` / `set_grad_enabled` / `is_grad_enabled`. Thread-local, reentrant, usable as decorators. |
| `src/engine/recompute.py` | 112 | `recompute(function, *inputs)`: gradient checkpointing (activation recomputation). |
| `src/engine/optim.py` | 238 | `SGD` (momentum, weight decay), `Adam`, `AdamW`, with `state_dict`/`load_state_dict`. |
| `src/engine/scheduler.py` | 72 | `WarmupCosineScheduler`. |
| `src/engine/checkpoint.py` | 78 | `save_checkpoint` / `read_checkpoint` / `restore_checkpoint`, format version 2, transactional restore. |
| `src/nn/module.py` | 180 | `Module` base: `modules`, `parameters`, `named_parameters`, `state_dict`, `train`/`eval`, `param_count`. |
| `src/nn/layers.py` | 235 | `Linear` (+LoRA), `Embedding`, `LayerNorm`, `RMSNorm`, `Dropout`, each with a NumPy `infer`. |
| `src/nn/attention.py` | 393 | `SelfAttention`, `MultiHeadAttention`, `RotaryEmbedding`, mask preparation/validation, zero-row softmax. |
| `src/nn/transformer.py` | 606 | `FeedForward`, `SwiGLU`, `TransformerBlock`, `GPT` (forward / infer / generate / generate_beam / LoRA / config). |
| `src/train.py` | 630 | Training CLI: corpora, batching, loop, evaluation, checkpointing, sampling, arg validation. |
| `src/tokenizer.py` | 123 | `CharTokenizer`, `BPETokenizer`, build/restore helpers. |
| `src/benchmark.py` | 122 | Throughput/timing harness. |
| `plot_loss.py` | 105 | JSONL log plotting. Source-checkout only, not packaged. |

**Model options.** `GPT(norm="layernorm"|"rmsnorm", pos_encoding="learned"|"rope",
ffn="gelu"|"swiglu")` — defaults give GPT-2 style, the alternatives together give
Llama style. Weight tying between the embedding table and the LM head is always on.
`lora_rank`/`lora_alpha` freeze the backbone and train low-rank adapters.
`grad_checkpoint` is a runtime toggle, deliberately not in `config()`.

---

## 2. Completed work

Round 1 predates the current planning artifacts; rounds 2–6 are recorded phase by
phase in `task_plan.md` and `progress.md`.

**Round 1 — correctness and delivery hardening.**
Fixed `matmul` backward for same-rank broadcasting and all 1-D NumPy cases; generalized
and stabilized `cross_entropy` (extreme logits returned ~27.63 instead of ~2000);
made repeated `backward()` on a chained graph accumulate correctly (was 96 instead of
64); replaced recursive topological sort with an iterative one (`RecursionError` at
~1,100 ops); implemented real division (was `exp(-log(b))`, NaN for negative divisors);
fixed stable sigmoid warnings, negative-axis transpose, generator-exhausting `concat`.
Made `forward(mask=None)` causal to match `infer()` (measured leak: 0.51 max error).
Made causal masks true `-inf`. Added optimizer-class and NumPy-RNG state to checkpoints
with legacy compatibility, and made restore transactional. Repaired PEP 517 packaging
(`setuptools.build_meta`) so the wheel builds and `tiny-train` runs. Added public-API
validation across constructors, token inputs, RoPE bounds, and generation arguments.

**Round 2 — inference mode and fully-masked rows.**
`no_grad`/`enable_grad`/`set_grad_enabled`, gated once inside `Tensor.__init__` so all
18 ops are covered without per-op edits. Op results under `no_grad` lose parents,
gradient buffer, and backward closure; explicit leaves stay trainable. `backward()` on a
detached tensor inside a disabled block raises instead of silently doing nothing.
Defined an all-`-inf` softmax row as zero weights in both the autograd and NumPy paths.
Added custom-mask validation (broadcast shape, NaN, `+inf`). `cross_entropy` rejects a
scored row with no finite logit.

**Round 3 — variable-length batches.**
`cross_entropy(ignore_index=…)` with scored-count divisor and exact zero gradient at
ignored positions; `GPT.forward(idx, attention_mask=…)` combining a `(batch, time)`
keep mask with the causal mask. Right padding required. Proved a padded batch is exactly
equivalent to unpadded runs.

**Round 4 — batched generation from ragged prompts.**
Per-element RoPE positions (`rotate_np(positions=…)`), additive key bias threaded
through block/attention inference, `GPT.infer(attention_mask=…, position_ids=…)` with the
mask covering cached *and* current keys, and `generate(attention_mask=…)` deriving
per-row positions as `cumsum(mask) - 1`. Left padding required and enforced; a masked run
must fit inside `context_len`; beam search rejects a mask.

**Round 5 — gradient checkpointing.**
`engine/recompute.py`: unrecorded forward under `no_grad()`, recorded replay under
`enable_grad()` inside the backward closure, replaying from detached input copies and
restoring the NumPy RNG state. Exposed as `GPT(grad_checkpoint=…)` per block and
`--grad-checkpoint`.

**Round 6 — document corpora in the training CLI.**
`--data-format text|lines|jsonl` and `--jsonl-field`; `load_documents`,
`encode_documents`, `get_document_batch`, `batch_loss`, `evaluate_batches`,
`evaluate_documents`. Both corpus kinds flow through one `(tokens, targets, mask)`
sampler, so the stream path is byte-identical. Tokenizer is built from document text,
not the raw file.

---

## 3. Design decisions worth knowing

| Decision | Why |
|----------|-----|
| Gate recording in `Tensor.__init__` | Every op constructs its result there, so one thread-local check covers all 18 primitives with no duplicated logic. |
| Suppress op results, never explicit leaves | Matches PyTorch; avoids silently producing an untrainable model built inside `no_grad()`. |
| Drop the backward closure on a node that cannot hold a gradient | Releases captured intermediates (the real memory win) and removes a latent crash where a gradient-less node was asked to split a `None` gradient. |
| Fully masked row → zero weights | Standard masked-attention convention; keeps forward and backward finite. Load-bearing in rounds 3 and 4. |
| `cross_entropy` stays strict about all-`-inf` rows | Zero weights are meaningful for attention; a loss over an impossible distribution is not, and a silent NaN corrupts every weight. |
| Mask and `ignore_index` ship together | Only masking leaves padded targets scored; only `ignore_index` leaves real tokens attending to padding. |
| Divide by the scored count, not the total | Otherwise loss shrinks as padding grows and gradient magnitude tracks batch shape rather than content. |
| Right-pad training, left-pad generation | Training numbers position *i* at slot *i*; decoding reads slot −1, so the newest token must sit there. One padding could not serve both. |
| `infer`'s mask covers cached keys | Padded prompt slots stay in the KV cache for the whole run; a per-step mask would unhide them right after the prefill. |
| Derive `position_ids` inside `generate` | `cumsum(mask) - 1` is the only correct answer for a left-padded row; deriving it removes a class of caller mistakes. |
| Refuse masked runs exceeding `context_len` | The sliding-window crop resets the cache and renumbers slots, so per-row positions would silently drift. |
| Module named `recompute`, not `checkpoint` | `engine/checkpoint.py` already means on-disk training state. |
| Replay from detached copies | `Tensor.backward` resets the gradient of any node with parents, so replaying into the outer graph discards a residual's accumulated gradient — this fires on the very first block. |
| Capture and replay the RNG state | Otherwise the replay draws fresh dropout masks and differentiates a different function; restoring afterwards keeps the training trajectory bit-identical. |
| `grad_checkpoint` out of `config()` | It changes memory and time, never weights or outputs; persisting it would pin a machine's memory budget to a model file. |
| One sampler interface for both corpora | The stream path passes `mask=None` and consumes the RNG identically, so training and evaluation have one code path. |
| Split documents before building the tokenizer | Training on raw JSONL put `{`, `"`, `:` in the vocabulary — an observed char vocab of 26 instead of 19. |

---

## 4. Test baseline

`python -m pytest -q -W error` → **301 passed in ~1.0s**.

| Module | Tests | Covers |
|--------|-------|--------|
| `tests/test_autograd.py` | 63 | Ops, VJPs with non-uniform cotangents, matmul broadcasting/1-D cases, stable CE, repeated backward, deep graphs, division, transpose/concat edge cases. |
| `tests/test_transformer.py` | 54 | Attention parity, causality, masks, KV cache, RoPE, generation, ragged batches, batched masked generation. |
| `tests/test_validation.py` | 41 | Public-API argument validation across constructors, tokens, RoPE, generation, norms. |
| `tests/test_modern.py` | 37 | Llama-style stack: RMSNorm, RoPE, SwiGLU, AdamW, gradient accumulation, LoRA. |
| `tests/test_training.py` | 28 | Training loop, checkpoint save/resume/transactionality, optimizer identity, RNG state, CLI. |
| `tests/test_data.py` | 23 | Document corpora: parsing, encoding, batch layout, loss/gradient equivalence, evaluation, 4 in-process CLI runs. |
| `tests/test_grad_mode.py` | 21 | Suppression, leaf semantics, nesting/exceptions/decorators/thread-locality, backward errors. |
| `tests/test_recompute.py` | 19 | Plain-call equivalence, residual-consumer safety, RNG neutrality, dropout replay, model-level trajectory equality, LoRA. |
| `tests/test_features.py` | 15 | Tokenizers, schedulers, sampling filters, misc CLI features. |

**Canonical CLI regression anchor** (verified 2026-07-31, reproducible run to run, and
identical with `--grad-checkpoint`):

```bash
python src/train.py --iters 3 --eval-interval 1 --eval-iters 2 \
  --ctx 32 --d 64 --heads 4 --layers 2 --batch 8 --seed 7 --no-sample
```

```text
step 1/3  train_loss=3.6106  val_loss=3.5996  val_ppl=36.58  lr=0.0001  gnorm=2.677
step 2/3  train_loss=3.5812  val_loss=3.5620  val_ppl=35.23  lr=0.0002  gnorm=2.654
step 3/3  train_loss=3.5234  val_loss=3.4988  val_ppl=33.07  lr=0.0003  gnorm=2.547
```

Any refactor of the default token-stream path must reproduce these numbers exactly.

**Measured tradeoffs** (recorded when the features landed):

| Feature | Measurement |
|---------|-------------|
| `no_grad()` evaluation | 4 layers, `d_model=128`, `ctx=64`, batch 8, 10 passes: 1.102 s recorded → 0.586 s (1.88×). Live tensors after a forward pass: 346 → 65 (the model's own parameters). |
| Gradient checkpointing | 6 layers, `d_model=128`, `ctx=64`, batch 8: retained activations 320.9 MiB (413 tensors) → 14.2 MiB (29 tensors), 22.7× less; 227 ms/step → 342 ms/step, 1.51× slower. |
| Padded vs. unpadded equivalence | Logit difference exactly `0.0`; loss and every parameter gradient within 1e-12–1e-14. |

---

## 5. Known limitations (deliberate, documented)

These are stated in the README and enforced with clear errors — none is an open bug.

- **`float64` only.** No dtype system, no mixed precision, no GPU.
- **Right padding for training masks, left padding for generation masks.** Enforced, not assumed.
- **A masked generation run must fit in `context_len`.** No sliding-window decoding for
  masked runs; the unmasked path still crops as before.
- **`recompute` wraps functions returning a single `Tensor`.** Multi-output and
  multi-section forms are not implemented (block structure does not need them).
- **Beam search takes no attention mask.** It decodes one sequence at a time.
- **A fully masked query row returns exactly the output projection's bias** — a constant,
  not a prediction. Such positions must be excluded from the loss.
- **Checkpoints are pickles.** Load trusted files only.
- **`plot_loss.py` is source-checkout only**, not part of the installed wheel.
- **BPE is a teaching implementation** — training is O(merges × corpus) and not intended
  for large corpora.

## 6. Repository state

- History: `021dda5 Initial commit`, then one checkpoint commit on branch
  `checkpoint/rounds-1-7` containing everything described above. That single commit also
  carries the pre-existing user diff (Llama/RoPE/SwiGLU/AdamW), which was interleaved
  with six rounds of work across the same files and could not be separated after the
  fact — see the round 7 decision in `task_plan.md`.
- The checkpoint branch is not merged into `main`. To put it there:
  `git checkout main && git merge --ff-only checkpoint/rounds-1-7`.
- `git diff --check` is clean apart from pre-existing LF/CRLF notices on Windows.
- No build/dist/venv/checkpoint artifacts are left in the tree.

---

## 7. Next-round candidates

Ranked. None blocks the default training or generation path.

1. **Sliding-window decoding for masked runs.** Today `generate(attention_mask=…)` refuses
   when prompt + `max_new_tokens` exceeds `context_len`, because cropping resets the KV
   cache and renumbers slots while per-row positions must stay absolute. Doing this
   properly means cropping per row, rebuilding the cache, and remapping `position_ids` —
   then proving a cropped masked run still matches generating that prompt alone.
   *Highest value: it removes the only hard refusal in the generation API.*
2. **Multi-output / multi-section `recompute`.** Currently one function returning one
   `Tensor`. A tuple-returning form would let a caller checkpoint a section that also
   emits a cache or auxiliary loss. Needs care: the replay must seed each output's
   cotangent, and the RNG capture must still cover the whole section.
3. **Evaluation throughput.** Evaluation runs under `no_grad()` but still goes through the
   graph-building `forward`. Routing it through the existing NumPy `infer` path would be
   faster; the risk is that the two paths could drift, so it needs a parity test first.
4. **Perplexity-correct evaluation for document corpora.** `evaluate_documents` averages
   per-batch scored means; a token-weighted aggregate across batches would be the more
   defensible number for ragged data.

Before starting any of these, re-read §5 of this file and the invariant list in
`CLAUDE.md` — most of the traps in this codebase are interactions between rounds, not
bugs inside a single file.
