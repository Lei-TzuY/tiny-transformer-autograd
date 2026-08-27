# Project State

Operational handoff for `tiny-transformer-autograd`. This file is intentionally concise:
it records the current stable baseline, load-bearing invariants, and active review map so a
new session can resume without re-deriving repository history.

**Snapshot date:** 2026-08-28
**Current `main`:** `a011903671efb97db0f73b50e081f4d45f5eab11`
**Main tree:** `e97cda08c06fd0ce4ba288a6e03f6fa42dfb7286`
**Latest main CI:** GitHub Actions run `33069878874`, green on Python 3.10–3.13.
Python 3.10 (CPython 3.10.21): **1399 passed, 22 subtests passed**.

> **Live GitHub state is authoritative.** This document is a handoff snapshot, not a lock.
> Before coding, always refresh `main`, open PRs, exact PR heads, compare ahead/behind, and
> Actions status. Do not reimplement work merely because it is absent from `main` if an open
> green PR already owns the same behavior or path.

Companions:
- `CLAUDE.md` — coding rules and semantic invariants.
- `task_plan.md`, `findings.md`, `progress.md` — historical planning and investigation logs;
  useful context, but not live branch/PR authority.

---

## 1. Stable baseline on main

The repository is a pure-NumPy educational deep-learning stack with reverse-mode autograd,
a decoder-only Transformer, training/generation CLIs, streaming generation, checkpointing,
and numerical regression coverage. NumPy is the runtime dependency; CI covers Python
3.10, 3.11, 3.12, and 3.13.

### Autograd / Tensor

- `Tensor` records graph parents and backward closures; graph traversal is iterative.
- `no_grad()` suppresses operation-result recording while explicit leaves remain trainable.
- Functional mutation/version checks reject stale graphs after supported in-place storage
  mutations.
- Core ops include arithmetic, matmul, activations, softmax, cross entropy, reductions,
  reshape/transpose, and concat.
- Numerically sensitive reductions and normalization paths contain explicit overflow and
  non-finite handling rather than relying on warning-prone incidental NumPy behavior.
- Functional `gradcheck()` exists on main; dense Jacobian/JVP/JVP-check APIs are still under
  review in the #83–#85 stack.

### Model / attention / generation

- GPT-style and Llama-style configurations are supported: LayerNorm/RMSNorm,
  learned positions/RoPE, GELU/SwiGLU, tied token embedding/LM-head weights, LoRA,
  dropout, and gradient checkpointing.
- Attention supports graph and NumPy inference paths, causal masks, custom masks,
  batched masks, KV caches, and fully masked attention rows with zero weights.
- `GPT.forward(mask=None)` is causal like inference.
- Training masks are right-padded; generation masks are left-padded.
- Masked generation derives per-row positions from the mask and safely renumbers positions
  after sliding-window crops.
- Generation supports sample, greedy, and beam strategies, with cached and uncached paths.
- RoPE streaming generation with bounded shifted KV-cache semantics is present on main;
  incremental iterator/stop-token/live-output extensions are the open #92–#94 stack.

### Training / evaluation

- `tiny-train` supports text, line-document, and JSONL corpora.
- Padded document training uses attention masks plus ignored targets and scored-token
  weighting; padding does not dilute the loss or gradients.
- Evaluation restores model mode transactionally and uses NumPy inference where applicable.
- Gradient accumulation, clipping, Adam/AdamW, warmup+cosine scheduling, LoRA, checkpoint
  resume, JSONL metrics, and sampling are supported.
- Default training arithmetic/RNG order is a compatibility surface: opt-in features must not
  silently change the historical default trajectory.

### Checkpoints / operational tooling

- Trusted historical pickle checkpoints remain supported by `engine.checkpoint`.
- A non-executable NPZ/JSON safe checkpoint format is supported by `engine.safe_checkpoint`;
  safe reads use `allow_pickle=False` internally.
- `tiny-train-safe` routes training through the safe checkpoint format.
- Checkpoint restore validates envelopes and is transactional with respect to caller-owned
  model/optimizer/scheduler/RNG state.
- Safe checkpoint inspection, trusted-pickle conversion, semantic digesting, and additional
  hardening are currently under review (#106, #107, #110, #117, #124).

### Benchmarks / CI

Every normal Actions matrix job performs:
1. install `.[dev]`,
2. installed-package/import smoke,
3. CLI help smokes,
4. general benchmark JSON smoke,
5. streaming benchmark JSON smoke,
6. `python -m pytest tests -q -W error`,
7. `python -m compileall -q src tests`.

The exact current-main Python 3.10 baseline is **1399 passed + 22 subtests**. Open feature
branches naturally report larger counts because they add focused regressions.

---

## 2. Load-bearing invariants

These rules are compatibility constraints, not suggestions:

1. Every differentiable op result is constructed through `Tensor.__init__`.
2. `no_grad()` suppresses op results, not explicit leaves.
3. Backward closures operate only on nodes that can hold gradients.
4. A fully `-inf` attention-softmax row yields zero weights.
5. A scored cross-entropy row must contain a finite logit.
6. `ignore_index` mean reduction divides by scored targets and ignored rows receive exact
   zero gradient.
7. Causal masks use true `-inf`.
8. `forward(mask=None)` remains causal and agrees with inference semantics.
9. Training padding is right-sided; generation padding is left-sided.
10. Activation recomputation replays detached copies under restored NumPy RNG state.
11. `grad_checkpoint` is a runtime memory/time choice and does not belong in model config.
12. Checkpoint restore is transactional.
13. Default text training must preserve historical RNG order and seeded trajectory unless a
    behavior is explicitly opt-in.
14. Generation window crops renumber surviving positions from zero.
15. Three-dimensional multi-head masks are batch-major.
16. Caller-provided KV caches are completely validated before a past length is trusted.
17. Token id `0` is a real vocabulary id; do not introduce `Embedding.padding_idx` semantics
    that would suppress it.
18. Public validation should raise deliberate `TypeError`/`ValueError` diagnostics instead
    of leaking implementation-level Python/NumPy exceptions.

---

## 3. Active review map

Refresh this list before acting; heads can move at any time.

### Stacked feature lines

- **#83 → #84 → #85** — dense Jacobian → functional JVP → directional JVP checker.
  **#119** is stacked on #85 for JVP-check tolerance overflow normalization.
- **#86 → #87 → #88** — stable log-probability losses → training label smoothing → optional
  validation RNG isolation. This stack was clean-replayed onto current main and each layer is
  one commit on its dependency.
- **#92 → #93 → #94** — incremental streaming iterator → stop-token termination → live CLI
  output. This stack was also clean-replayed onto current main.
- **#102 → #121** — Linear LoRA reconfiguration contract → layer real-value overflow
  normalization.
- **#114 → #116** — `concat(axis=None)` autograd → immutable snapshot of mutable sum-reduction
  metadata.

### Independent / mostly independent reviews

- #95 optimizer state-envelope validation.
- #96 tokenizer encoding/merge-count validation.
- #100 gradcheck public target validation.
- #101 Tensor flat-iterator mutation tracking.
- #103 Module traversal through general mappings.
- #104 paired benchmark measurement-order correction.
- #105 custom-awaitable rejection in grad-mode decorators.
- #106 safe checkpoint inspection CLI and packaging/workflow smoke.
- #107 safe-checkpoint deep-manifest recursion normalization.
- #108 scheduler real conversion overflow normalization.
- #109 streaming benchmark NumPy-integer normalization.
- #110 checkpoint RNG-state overflow normalization.
- #111 recompute with unused trainable inputs.
- #112 gradcheck tolerance conversion overflow normalization.
- #113 attention real-value conversion overflow normalization.
- #115 direct beam temperature overflow normalization.
- #117 trusted pickle → safe checkpoint conversion helper.
- #118 serialization of embedded safe-training checkpoint swaps.
- #120 `plot_loss.py` JSONL record validation.
- #122 Transformer/GPT real-value conversion overflow normalization.
- #123 optimizer constructor parameter-collection validation.
- #124 semantic digests for safe checkpoints.

### High-conflict paths while those PRs remain open

Treat these paths as occupied unless deliberately stacking on the owning PR:

- `src/engine/autograd.py`, `src/engine/__init__.py` — #83–#85.
- `src/engine/ops.py` — #114/#116.
- `src/engine/tensor.py` — #101.
- `src/engine/optim.py` — #95/#123.
- `src/engine/scheduler.py` — #108.
- `src/engine/checkpoint.py` — #110.
- `src/engine/safe_checkpoint.py` — #107.
- `src/engine/recompute.py` — #111.
- `src/engine/grad_mode.py` — #105.
- `src/engine/gradcheck.py`, `src/engine/_gradcheck_impl.py` — #100/#112.
- `src/nn/layers.py` — #102/#121.
- `src/nn/module.py` — #103.
- `src/nn/attention.py` — #113.
- `src/nn/transformer.py` — #122.
- `src/nn/beam.py` — #115.
- `src/nn/streaming.py`, `src/nn/__init__.py`, `src/streaming_cli.py` — #92–#94.
- `src/train.py` — #87/#88.
- `src/tokenizer.py` — #96.
- `src/benchmark.py` — #104.
- `src/streaming_benchmark.py` — #109.
- `src/safe_train_cli.py` — #118.
- `plot_loss.py` — #120.
- `pyproject.toml`, `.github/workflows/tests.yml` — #106.

New standalone modules can still be reasonable when they solve a real problem and do not
silently duplicate an open PR (for example #117 and #124 intentionally add new files only).

---

## 4. Development procedure

For every new engineering round:

1. Fetch live `main` SHA and the latest open PR list.
2. Search open PR titles/bodies and changed paths for overlap before editing.
3. Reproduce a real bug or define a concrete API/operational gap; avoid speculative changes.
4. Base directly on exact live `main`, or deliberately on the exact head of a dependency PR
   when stacking is necessary.
5. Keep the production delta narrow and add a regression that fails for the old behavior.
6. Preserve validation ordering, caller-owned state, NumPy RNG state, and default trajectory
   where those are part of the contract.
7. Run exact-head GitHub Actions and require Python 3.10–3.13 success.
8. Inspect at least one full job log for the exact test count and all smoke/compile steps.
9. Compare base → head and require the expected ahead/behind and file list.
10. Update PR metadata when a branch is replayed so reviewers do not see stale SHAs/CI counts.
11. **Never merge automatically.** Leave reviewed green PRs open unless the user explicitly
    requests a merge.

For large-file rewrites, reconstruct the exact current blob first, verify its Git blob SHA,
compute/verify the intended replacement blob, and immediately inspect the resulting compare.

---

## 5. Deliberate limits / non-goals

- The project is educational and NumPy-first: no GPU backend, mixed precision, distributed
  training, or production-scale kernel fusion on main.
- A tokenizer OOV error is an error; do not silently add or map to an `<unk>` token unless the
  language/model contract is deliberately redesigned.
- Do not weaken mask, finite-value, mutation, checkpoint, or transactional validation merely
  to accept malformed inputs.
- Do not change default CLI output, option defaults, checkpoint semantics, or seeded training
  arithmetic as collateral damage from an opt-in feature.

When this snapshot disagrees with GitHub, **GitHub wins**. Refresh this file when `main`
advances materially or the review map changes enough that it would misdirect future work.
