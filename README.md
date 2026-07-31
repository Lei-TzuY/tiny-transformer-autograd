# Tiny Transformer & Autograd

A from-scratch implementation of reverse-mode automatic differentiation and a
GPT-style language model — **pure NumPy, zero ML frameworks**.

The goal is to be a readable, runnable reference that makes the internals of
autograd and transformers concrete.  Every operation records a backward closure;
every module is built from those primitives.

---

## Features

| Component | What it shows |
| --- | --- |
| **Autograd engine** | Reverse-mode AD via dynamic computation graph; 18 differentiable ops with NumPy-compatible broadcasting |
| **Inference mode** | `no_grad()` / `enable_grad()` suppress graph recording — ~1.9× faster evaluation, no retained intermediates |
| **Grad checkpointing** | `recompute()` replays a block in backward instead of storing it — 23× less activation memory for 1.5× the time |
| **Optimizers** | SGD (+ momentum), Adam, and AdamW (decoupled weight decay) |
| **LR scheduler** | Linear warmup + cosine decay |
| **Layers** | Linear, Embedding, LayerNorm, RMSNorm, Dropout — all with fast `infer()` paths |
| **Attention** | Multi-head causal self-attention with KV-cache inference; optional RoPE; validated custom masks |
| **Transformer** | Pre-norm decoder block; beam/greedy/nucleus sampling; weight tying |
| **Ragged batches** | `attention_mask` hides padded keys, `cross_entropy(..., ignore_index=…)` drops padded targets, and left-padded batched generation matches one-at-a-time decoding, even past the context window |
| **Architectures** | `--arch gpt` (LayerNorm + learned pos + GELU) or `--arch llama` (RMSNorm + RoPE + SwiGLU) |
| **LoRA** | Low-rank adapter fine-tuning (freeze backbone, train A/B matrices) |
| **Grad accumulation** | `--grad-accum N` simulates N×-larger batches at constant memory |
| **Tokenizers** | Character-level and byte-pair encoding (BPE) |
| **Corpora** | One token stream (`--data-format text`) or one document per line / JSONL record, batched with padding + masking |
| **Checkpointing** | Versioned atomic save/resume of model + optimizer type + scheduler + NumPy RNG + arch config |
| **Benchmark** | Measures tokens/s and KV-cache speedup for either architecture |

---

## Project structure

```text
src/
├── engine/
│   ├── tensor.py       # Tensor class — data + grad + backward closure
│   ├── grad_mode.py    # no_grad / enable_grad / set_grad_enabled (graph recording)
│   ├── recompute.py    # gradient checkpointing (replay a section in backward)
│   ├── ops.py          # 18 differentiable primitives (add, div, matmul, gelu, silu, …)
│   ├── optim.py        # SGD, Adam, AdamW (state_dict / load_state_dict)
│   ├── scheduler.py    # WarmupCosineScheduler
│   └── checkpoint.py   # save_checkpoint / read_checkpoint / restore_checkpoint
├── nn/
│   ├── module.py       # Module base (parameters, train/eval, state_dict)
│   ├── layers.py       # Linear (+ LoRA), Embedding, LayerNorm, RMSNorm, Dropout
│   ├── attention.py    # SelfAttention, MultiHeadAttention, RotaryEmbedding (+ KV-cache)
│   └── transformer.py  # FeedForward, SwiGLU, TransformerBlock, GPT
├── tokenizer.py        # CharTokenizer, BPETokenizer
├── train.py            # Full training script
└── benchmark.py        # Inference throughput benchmark
tests/
├── test_autograd.py    # Numerical/VJP gradient checks and graph lifecycle (63 tests)
├── test_transformer.py # Shape, gradients, masks, padded generation (77 tests)
├── test_training.py    # Scheduler, tokenizer, checkpoint, LoRA tests (28 tests)
├── test_grad_mode.py   # no_grad graph suppression and scope semantics (21 tests)
├── test_data.py        # document corpora, padded batching, training CLI (23 tests)
├── test_recompute.py   # gradient checkpointing equivalence and memory (19 tests)
├── test_features.py    # Integration: KV-cache, generation, ragged batches (15 tests)
├── test_modern.py      # RMSNorm, RoPE, SwiGLU, AdamW, grad accumulation (37 tests)
└── test_validation.py  # Public API and invalid-input regression tests (41 tests)
plot_loss.py            # Plot training curves from a --log-jsonl file
CLAUDE.md               # Working rules, commands, and the invariants that must hold
PROJECT_STATE.md        # Architecture map, design decisions, baseline, limitations
```

---

## Quickstart

```bash
# Install the package and `tiny-train` command (only NumPy is required)
pip install .

# Train on the built-in Shakespeare excerpt
tiny-train

# Train on a custom file with BPE tokenization
python src/train.py --data path/to/book.txt --tokenizer bpe --bpe-merges 200

# Train on a corpus of short documents (one per line, or one JSON record per line)
python src/train.py --data corpus.txt   --data-format lines
python src/train.py --data corpus.jsonl --data-format jsonl --jsonl-field body

# Train a Llama-style model (RMSNorm + RoPE + SwiGLU) with AdamW
python src/train.py --arch llama --optimizer adamw --weight-decay 0.1

# Simulate a 4×-larger batch with gradient accumulation
python src/train.py --batch 8 --grad-accum 4

# Resume from a checkpoint (architecture, optimizer, and RNG are restored)
python src/train.py --resume run/ckpt.pkl --iters 2000

# Generate text from a saved checkpoint (no training)
python src/train.py --resume run/ckpt.pkl --generate-only --sample 400

# Run all 324 tests
pip install -r requirements-dev.txt
pytest tests/ -v
```

> **Checkpoint safety:** checkpoints use Python `pickle`. Only load files you
> created yourself or obtained from a fully trusted source; untrusted pickle
> files can execute arbitrary code while loading.

---

## Training CLI reference

```text
python src/train.py [OPTIONS]

Data
  --data FILE          plain-text training file (default: built-in Shakespeare)
  --data-format FMT    text: one token stream, random windows (default)
                       lines: one document per line
                       jsonl: one JSON record per line
  --jsonl-field NAME   document field for --data-format jsonl (default: text)
  --tokenizer char|bpe tokenizer type (default: char)
  --bpe-merges N       BPE merge count (default: 100)
  --val-frac F         validation fraction (default: 0.1)

Model
  --ctx N              context length / block size (default: 32)
  --d N                d_model — embedding & hidden width (default: 64)
  --heads N            attention heads (default: 4)
  --layers N           transformer blocks (default: 2)
  --dropout F          dropout probability (default: 0.0)
  --arch gpt|llama     gpt: LayerNorm + learned pos + GELU
                       llama: RMSNorm + RoPE + SwiGLU (default: gpt)

LoRA fine-tuning
  --lora-rank R        adapter rank; 0 = full training (default: 0)
  --lora-alpha A       LoRA scaling α (default: 1.0)

Optimisation
  --iters N            training steps (default: 1000)
  --optimizer NAME     adam (L2-coupled decay) | adamw (decoupled)
                       default: Adam for new runs; saved type when resuming
  --lr F               peak learning rate (default: 3e-4)
  --min-lr F           cosine-decay floor LR (default: 0.0)
  --warmup-iters N     linear warmup steps (default: 100)
  --weight-decay F     weight decay strength (default: 1e-2)
  --grad-clip F        gradient norm clipping; 0 = disabled (default: 1.0)
  --grad-checkpoint    recompute block activations in backward (saves memory)
  --batch N            batch size (default: 8)
  --grad-accum N       micro-batches accumulated per step (default: 1)
  --seed N             random seed (default: 42)

Logging & checkpoints
  --eval-interval N    validation every N steps (default: 100)
  --eval-iters N       validation batches (default: 10)
  --log-jsonl FILE     append metrics to a JSONL file
  --save FILE          checkpoint path
  --resume FILE        checkpoint to resume from
  --save-every N       save periodically every N steps (default: 0 = end only)
  --eval-only          evaluate and sample without training
  --generate-only      sample from checkpoint without evaluating
  --no-sample          skip final generation

Generation
  --sample N           tokens to generate after training (default: 200)
  --strategy           sample | greedy | beam (default: sample)
  --temperature F      softmax temperature (default: 0.8)
  --top-k N            top-k filtering
  --top-p F            nucleus (top-p) filtering
  --beam-width N       beam width for beam search (default: 3)
  --prompt TEXT        custom generation prompt
  --prompt-file FILE   read prompt from file
  --no-kv-cache        disable KV-cache during generation
```

---

## Architecture

```text
Input (B, T)  integer token ids
│
├─ token_emb   Embedding(vocab, d)     ┐  weight-tied:
├─ pos_emb     Embedding(ctx, d)       │  token_emb.weight == head.weight
└─ emb_drop    Dropout                 │
                                       │
┌─ TransformerBlock ×L ─────────────┐ │
│  x = x + MHA(LN(x), causal_mask) │ │
│  x = x + FFN(LN(x))              │ │
│                                   │ │
│  MHA:  W_q, W_k, W_v, out_proj   │ │
│  FFN:  fc1 → GELU → fc2          │ │
└───────────────────────────────────┘ │
                                       │
├─ ln_f        LayerNorm               │
└─ head        Linear(d, vocab) ───────┘

Output (B, T, vocab)  raw logits
```

**Llama variant** (`--arch llama`) — swaps each component for its modern
counterpart: LayerNorm → RMSNorm, learned positional embeddings → RoPE
(rotation applied to Q/K inside each attention head, no `pos_emb` table),
and the GELU FFN → SwiGLU (`W_down(SiLU(W_gate·x) ⊙ W_up·x)`).  Everything
else (causal mask, KV-cache, weight tying, LoRA) works identically.

**Causal mask** — a `(ctx, ctx)` upper-triangular matrix of negative infinity is
precomputed once, sliced to `(T, T)` per forward pass, and broadcast over
`(B, H, T, T)`. Standalone attention modules create the same mask by default,
so their graph-tracked `forward()` and NumPy `infer()` paths agree.

**Custom masks** — `attn(x, mask)` takes any additive bias that broadcasts up to
the score shape (`(T, T)`, `(B, T, T)`, or `(B, H, T, T)`), as a Tensor or a
plain NumPy array: `0.0` keeps a key, `-inf` removes it. A mask that is larger
than the scores, does not broadcast, or contains NaN/`+inf` is rejected with an
explicit error instead of quietly producing NaN.

**Fully masked rows** — if every key of a query row is `-inf` (a padded query, or
an over-restrictive custom mask) there is no distribution to normalise. Both
`ops.softmax` and the inference softmax define that row as **all-zero weights**
rather than NaN, so:

- the row's context vector is exactly zero and the layer output for that
  position is only `out_proj`'s bias — a constant, not a prediction;
- no gradient flows from that position back to Q, K, or V;
- the rest of the batch is unaffected.

Nothing turns into NaN, but such a position carries no information: exclude it
from the loss rather than trusting it. `cross_entropy` is deliberately stricter
— a logits row with no finite entry raises, because no target can be scored
against an impossible distribution.

**Variable-length batches** — pass `model(idx, attention_mask=…)` with a
`(batch, time)` mask where `1`/`True` marks a real token and `0`/`False` marks
padding. It is combined with the causal mask inside every layer, so padded keys
are invisible: a real token's logits are identical to running its sequence
unpadded, and rewriting the padded slots cannot change them (both are asserted
in `test_transformer.py`). Pair it with `ignore_index` so padded *targets* are
not scored either:

```python
loss = ops.cross_entropy(
    model(tokens, attention_mask=attention_mask),
    targets,                 # padded positions hold ignore_index
    ignore_index=-1,
)
```

`ignore_index` positions are dropped from the mean (the divisor is the number of
scored positions) and receive exactly zero gradient, so the loss and every
parameter gradient match a run on the unpadded sequence. An all-`ignore_index`
batch raises rather than dividing by zero.

For training, positions are numbered from 0, so **pad on the right** — the
forward pass assigns position *i* to slot *i*.

The training CLI uses exactly this path for document corpora:

```bash
python src/train.py --data corpus.txt   --data-format lines
python src/train.py --data corpus.jsonl --data-format jsonl --jsonl-field body
```

Each line (or JSON record) is one document. Documents are truncated to
`context_len + 1` tokens — input is the document without its last token, target
without its first — and anything shorter than two tokens is dropped and
reported. A sampled batch is right-padded to its longest document, with the
mask and `ignore_index` doing the rest, so **a padded batch's loss equals the
mean of scoring each document on its own** (asserted in `test_data.py`, along
with the gradients being unchanged when the padding content is scrambled). The
train/validation split is over documents rather than tokens, and the tokenizer
is built from the document text — not the JSON scaffolding around it, which
would otherwise put braces and quotes in the vocabulary.

**Batched generation from ragged prompts** — decoding reads the next
distribution from slot −1, so here you **pad on the left** and pass the mask to
`generate`:

```python
tokens = np.array([[0, 0, 1, 4, 7],     # 0 = pad, left-aligned padding
                   [0, 3, 6, 2, 8]])
mask   = np.array([[0, 0, 1, 1, 1],
                   [0, 1, 1, 1, 1]])

out = model.generate(tokens, max_new_tokens=8, strategy="greedy",
                     attention_mask=mask)
generated = out[:, -8:]                 # prompt columns come back unchanged
```

`generate` derives per-row `position_ids` (`cumsum(mask) − 1`, clamped at 0), so
each row is numbered from its own first real token, and the padded slots stay
masked for the whole run — including inside the KV-cache, which is why
`infer`'s `attention_mask` covers *cached and current* keys. The result is
exact: a row's generated tokens are bitwise identical to generating that prompt
on its own, cached and uncached agree, and this holds for both `--arch gpt`
(learned positions) and `--arch llama` (RoPE, rotated per row). Scrambling the
padding changes nothing.

One rule the implementation enforces rather than assumes: the mask must be
genuinely left-padded (zeros then ones, at least one real token per row). Beam
search takes no mask, since it decodes one sequence at a time.

A masked run is free to outgrow `context_len`. It then slides the same window
an unmasked run does — keeping each row's newest `context_len` slots — and
**renumbers the surviving real tokens from 0 inside the new window**, which is
the step that makes the crop safe for both learned positions and RoPE. Because
every row's newest token sits at slot −1, one crop of the shared array is
simultaneously right for every row: a row that has not filled the window yet
loses padding, and a row that has loses its oldest tokens. The equivalence
survives the crop — a row's tokens still match generating that prompt alone,
including when the prompt is *already* longer than `context_len`.

```python
model.context_len          # 8
out = model.generate(tokens, max_new_tokens=20, strategy="greedy",
                     attention_mask=mask)
out.shape[1]               # 25 — the window was re-cropped 16 times
```

A left-padded row's first query attends only to padding, which is exactly the
fully-masked row defined above: it yields a zero context vector instead of NaN,
which is what makes the whole scheme work.

**KV-cache** — `model.infer(idx, kv_cache)` skips recomputing keys and values
for already-processed positions, giving ~1.5× speedup for autoregressive
generation.  The cache is automatically reset when the context window fills,
because learned absolute positional embeddings would be invalid after a
sliding-window crop.

**Weight tying** — `head.weight` and `token_emb.weight` are the same Python
object (GPT-2 style).  Gradients from both the embedding lookup and the output
projection accumulate into a single array.

---

## Using the autograd engine directly

```python
import sys
sys.path.insert(0, "src")

import numpy as np
from engine.tensor import Tensor
import engine.ops as ops

x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
y = Tensor([[0.5, -1.0], [-0.5, 1.0]], requires_grad=True)

z    = ops.relu(ops.matmul(x, y.T))
loss = ops.mean(z)
loss.backward()

print(x.grad)   # ∂loss/∂x
print(y.grad)   # ∂loss/∂y
```

Available ops: `add`, `mul`, `div`, `matmul`, `sigmoid`, `relu`, `exp`, `log`,
`tanh`, `gelu`, `silu`, `softmax`, `cross_entropy`, `sum`, `mean`, `reshape`,
`transpose`, `concat`. `cross_entropy` takes `(..., C)` logits with matching
integer targets and an optional `ignore_index`; `softmax` defines a fully masked
row as zero weights (see [Architecture](#architecture)).

### Inference mode: `no_grad()`

Every op stores its parents and keeps a backward closure alive, and that closure
holds the forward intermediates it will need. For evaluation or generation that
bookkeeping is pure overhead, so wrap the work in `no_grad()`:

```python
from engine.grad_mode import no_grad, enable_grad, is_grad_enabled

with no_grad():
    logits = model(x)          # detached: no parents, no closures, no .grad
    assert not logits.requires_grad

@no_grad()                     # also usable as a decorator
def score(batch):
    return ops.cross_entropy(model(batch[0]), batch[1]).data
```

- **Op results** created while recording is off come back detached.
- **Leaves are untouched:** `Tensor(x, requires_grad=True)` inside a `no_grad()`
  block stays trainable, so constructing a model there does not freeze it.
- **Nesting works:** `enable_grad()` re-enables recording inside a `no_grad()`
  block, blocks restore the previous mode even if the body raises, and the flag
  is thread-local.
- **Mistakes are loud:** calling `backward()` on a detached tensor inside a
  `no_grad()` block raises instead of silently doing nothing. Differentiating a
  graph that was built *outside* the block is still allowed.

Measured on a 4-layer, `d_model=128`, `ctx=64`, batch-8 forward + loss: **1.88×
faster** (1.10 s → 0.59 s for 10 passes) and 281 fewer live tensors per pass.
`train.py`'s validation loop uses it — `eval()` only disables dropout, while
`no_grad()` is what stops the graph from being built.

### Gradient checkpointing: `recompute()`

Training memory is dominated by activations kept for the backward pass.
`recompute()` trades that memory for one extra forward pass: the section runs
with recording off, only its input and output are kept, and when the backward
pass reaches it the section is replayed *with* recording and differentiated.

```python
from engine.recompute import recompute

x = recompute(lambda inp: block(inp, mask), x)   # instead of block(x, mask)
```

`GPT` exposes it per block, as a runtime toggle rather than architecture — it is
absent from `config()`, so it can be flipped on a resumed run:

```python
model = GPT(..., grad_checkpoint=True)
model.grad_checkpoint = False        # or flip it at any time
```

```bash
python src/train.py --grad-checkpoint
```

Measured on a 6-layer, `d_model=128`, `ctx=64`, batch-8 training step:

| | activations retained | time per step |
| --- | --- | --- |
| plain | 320.9 MiB (413 tensors) | 227 ms |
| `--grad-checkpoint` | 14.2 MiB (29 tensors) | 342 ms |
| | **22.7× less** | **1.51× slower** |

Nothing else changes: forward values are identical, gradients agree to 1e-14,
and a five-step training run produces the same losses to the last digit — with
dropout enabled. Two details make that true:

- **The replay uses detached copies of the inputs.** Differentiating into the
  original tensor would let `backward()` reset a node that has parents in the
  outer graph, discarding gradient another consumer (a residual connection) had
  already accumulated. Input gradients are added into the originals instead.
- **The NumPy RNG state is captured and replayed,** so the replay draws the same
  dropout masks as the forward pass — then the state in effect when backward
  started is restored, leaving the training loop's random stream untouched.

---

## Building a custom model

```python
import sys
sys.path.insert(0, "src")

import numpy as np
from engine.tensor import Tensor
from engine.optim import Adam
import engine.ops as ops
from nn.module import Module
from nn.layers import Linear

class MLP(Module):
    def __init__(self):
        self.fc1 = Linear(2, 16)
        self.fc2 = Linear(16, 1)

    def forward(self, x):
        return self.fc2(ops.relu(self.fc1(x)))

model = MLP()
opt   = Adam(model.parameters(), lr=1e-3)

for step in range(500):
    x = Tensor(np.random.randn(32, 2))
    y = (x.data[:, 0] > x.data[:, 1]).astype(float)

    logits = model(x)
    loss   = ops.mean((logits - Tensor(y[:, None])) ** 2)

    opt.zero_grad()
    loss.backward()
    opt.step()
```

---

## Plotting training curves

`plot_loss.py` is a source-checkout utility (it is not included in the wheel).

```bash
# Train with JSONL logging
python src/train.py --iters 1000 --log-jsonl runs/tiny.jsonl

# Plot loss + learning rate
python plot_loss.py runs/tiny.jsonl

# Save to a file instead of opening a window
python plot_loss.py runs/tiny.jsonl --out loss.png
```

Requires `matplotlib` (`pip install matplotlib`), which is not in `requirements.txt`
because training and inference need only NumPy.

---

## Benchmark

```text
python src/benchmark.py            # or --arch llama

Tiny GPT benchmark
  arch: gpt
  shape: vocab=128 ctx=32 d=64 heads=4 layers=2 batch=4
  infer:                ~48 000 tokens/s
  generate cached:       ~6 600 tokens/s
  generate uncached:     ~2 200 tokens/s
  cache speedup:             ~2.9×
```

---

## Design notes

- **No external ML dependency** — only `numpy`; the autograd engine stays compact
  enough to read end-to-end.
- **Reverse-mode AD** — each op stores a `_backward` closure that accumulates
  `∂L/∂input` into `input.grad`.  `Tensor.backward()` executes closures in
  reverse topological order (DFS topo-sort; no cycles possible by construction).
- **Broadcasting** — `_unbroadcast` in `ops.py` sums over axes that NumPy
  expanded implicitly, so gradients flow correctly through broadcast adds.
- **Batched matmul fix** — the matmul backward sums over extra batch dims so a
  2-D weight `(K, N)` used with a 3-D batched input `(B, M, K)` receives a
  correctly-shaped gradient without silent shape errors.
- **Weight tying** — `head.weight is token_emb.weight` (same Python object).
  `parameters()` deduplicates by `id()`, so the tied weight is counted once and
  initialised once.
- **RoPE and the KV-cache** — cached keys are stored *already rotated* at their
  absolute positions, so each decode step only rotates the new Q/K slice
  (`offset=past_len`).  Because rotation preserves norms and the score
  `q·k` depends only on the relative offset, RoPE needs no learned table —
  it is built from `_rotate_half` (slice + negate + concat), all existing
  autograd primitives.
- **AdamW vs Adam** — Adam's L2 term flows through the m/v moments, where the
  adaptive denominator normalises it away (the first-step update is ≈ lr
  regardless of weight magnitude).  AdamW decays weights directly
  (`p ← p·(1−lr·λ)` before the moment update), keeping regularisation
  proportional to the weight — `test_modern.py` demonstrates the difference.
- **Gradient accumulation** — `backward()` *adds* into `.grad`, so running N
  micro-batches before `optimizer.step()` and dividing the grads by N is
  exactly equivalent to one N×-larger batch (verified in tests).
- **Numerically stable sigmoid** — computes `z = exp(-abs(x))` first, then
  selects `1/(1+z)` or `z/(1+z)`. Both eagerly evaluated NumPy branches remain
  finite even for very large magnitudes.
- **Graph suppression** — `no_grad()` flips one thread-local flag that the
  `Tensor` constructor reads: an op result created while it is off gets no
  parents, no `.grad` buffer, and no backward closure, which releases the
  captured intermediates. A node that cannot receive a gradient never stores a
  closure at all, so a gradient-less op inside a larger graph is skipped rather
  than asked to propagate a gradient it does not have.
- **Masked softmax** — the row shift is `0` where the row maximum is `-inf` and
  the normaliser is clamped away from `0`, so a fully masked row becomes all
  zeros instead of `-inf − -inf = nan`. Unmasked rows are bitwise identical to
  the plain stable formula, and the zero row makes the softmax VJP vanish on its
  own — no special case in the backward pass.
