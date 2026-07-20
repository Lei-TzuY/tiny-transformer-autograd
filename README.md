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
| **Autograd engine** | Reverse-mode AD via dynamic computation graph; 15 differentiable ops |
| **Optimizers** | SGD (+ momentum) and Adam with bias correction |
| **LR scheduler** | Linear warmup + cosine decay |
| **Layers** | Linear, Embedding, LayerNorm, Dropout — all with fast `infer()` paths |
| **Attention** | Multi-head causal self-attention with KV-cache inference |
| **Transformer** | Pre-LN GPT block; beam/greedy/nucleus sampling; weight tying |
| **LoRA** | Low-rank adapter fine-tuning (freeze backbone, train A/B matrices) |
| **Tokenizers** | Character-level and byte-pair encoding (BPE) |
| **Checkpointing** | Atomic save/resume of model + optimizer + scheduler |
| **Benchmark** | Measures tokens/s and KV-cache speedup |

---

## Project structure

```text
src/
├── engine/
│   ├── tensor.py       # Tensor class — data + grad + backward closure
│   ├── ops.py          # 15 differentiable primitives (add, matmul, gelu, …)
│   ├── optim.py        # SGD, Adam (state_dict / load_state_dict)
│   ├── scheduler.py    # WarmupCosineScheduler
│   └── checkpoint.py   # save_checkpoint / read_checkpoint / restore_checkpoint
├── nn/
│   ├── module.py       # Module base (parameters, train/eval, state_dict)
│   ├── layers.py       # Linear (+ LoRA), Embedding, LayerNorm, Dropout
│   ├── attention.py    # SelfAttention, MultiHeadAttention (+ KV-cache)
│   └── transformer.py  # FeedForward, TransformerBlock, GPT
├── tokenizer.py        # CharTokenizer, BPETokenizer
├── train.py            # Full training script
└── benchmark.py        # Inference throughput benchmark
tests/
├── test_autograd.py    # Numerical gradient checks for every op (31 tests)
├── test_transformer.py # Shape, gradient-flow, causal-mask tests (26 tests)
├── test_training.py    # Scheduler, tokenizer, checkpoint, LoRA tests (22 tests)
└── test_features.py    # Integration: KV-cache, generation, JSONL logging (14 tests)
plot_loss.py            # Plot training curves from a --log-jsonl file
```

---

## Quickstart

```bash
# Install (only numpy is required at runtime)
pip install -r requirements.txt

# Train on the built-in Shakespeare excerpt
python src/train.py

# Train on a custom file with BPE tokenization
python src/train.py --data path/to/book.txt --tokenizer bpe --bpe-merges 200

# Resume from a checkpoint
python src/train.py --resume run/ckpt.pkl --iters 2000

# Generate text from a saved checkpoint (no training)
python src/train.py --resume run/ckpt.pkl --generate-only --sample 400

# Run all 93 tests
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

## Training CLI reference

```text
python src/train.py [OPTIONS]

Data
  --data FILE          plain-text training file (default: built-in Shakespeare)
  --tokenizer char|bpe tokenizer type (default: char)
  --bpe-merges N       BPE merge count (default: 100)
  --val-frac F         validation fraction (default: 0.1)

Model
  --ctx N              context length / block size (default: 32)
  --d N                d_model — embedding & hidden width (default: 64)
  --heads N            attention heads (default: 4)
  --layers N           transformer blocks (default: 2)
  --dropout F          dropout probability (default: 0.0)

LoRA fine-tuning
  --lora-rank R        adapter rank; 0 = full training (default: 0)
  --lora-alpha A       LoRA scaling α (default: 1.0)

Optimisation
  --iters N            training steps (default: 1000)
  --lr F               peak learning rate (default: 3e-4)
  --min-lr F           cosine-decay floor LR (default: 0.0)
  --warmup-iters N     linear warmup steps (default: 100)
  --weight-decay F     Adam L2 regularisation (default: 0.0)
  --grad-clip F        gradient norm clipping; 0 = disabled (default: 1.0)
  --batch N            batch size (default: 8)
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

**Causal mask** — a `(ctx, ctx)` upper-triangular matrix of −10⁹ is precomputed
once, sliced to `(T, T)` per forward pass, and broadcast over `(B, H, T, T)`.

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

Available ops: `add`, `mul`, `matmul`, `sigmoid`, `relu`, `exp`, `log`,
`tanh`, `gelu`, `softmax`, `cross_entropy`, `sum`, `mean`, `reshape`,
`transpose`, `concat`.

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
python src/benchmark.py

Tiny GPT benchmark
  shape: vocab=128 ctx=32 d=64 heads=4 layers=2 batch=4
  infer:                ~44 000 tokens/s
  generate cached:       ~2 500 tokens/s
  generate uncached:     ~1 600 tokens/s
  cache speedup:             ~1.5×
```

---

## Design notes

- **No external ML dependency** — only `numpy`.  The entire autograd engine is ~300 lines.
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
- **Numerically stable sigmoid** — uses the `np.where` trick:
  `1/(1+exp(-x))` for `x≥0` and `exp(x)/(1+exp(x))` for `x<0`, avoiding
  overflow in both branches.
