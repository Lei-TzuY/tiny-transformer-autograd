"""Feed-forward, transformer block, and tiny GPT modules."""

import numpy as np

from engine.tensor import Tensor
import engine.ops as ops
from .module import Module
from .layers import Linear, Embedding, LayerNorm, Dropout
from .attention import MultiHeadAttention


class FeedForward(Module):
    """Position-wise feed-forward network with GELU activation."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)
        self.drop = Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = ops.gelu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)

    def infer(self, x):
        return self.fc2.infer(_gelu(self.fc1.infer(x)))


class TransformerBlock(Module):
    """One decoder-only pre-LayerNorm transformer block."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
    ):
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ln2 = LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        x = x + self.attn(self.ln1(x), mask)
        return x + self.ff(self.ln2(x))

    def infer(self, x, cache=None):
        attn, cache = self.attn.infer(self.ln1.infer(x), cache)
        x = x + attn
        return x + self.ff.infer(self.ln2.infer(x)), cache


class GPT(Module):
    """Tiny decoder-only GPT language model."""

    def __init__(
        self,
        vocab_size: int,
        context_len: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        dropout: float = 0.0,
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
    ):
        self.vocab_size = vocab_size
        self.context_len = context_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.dropout = dropout
        self.lora_rank = 0
        self.lora_alpha = lora_alpha

        self.token_emb = Embedding(vocab_size, d_model)
        self.pos_emb = Embedding(context_len, d_model)
        self.emb_drop = Dropout(dropout)
        self.blocks = [
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ]
        self.ln_f = LayerNorm(d_model)
        self.head = Linear(d_model, vocab_size, bias=False)
        # Weight tying: the LM head reuses the embedding table (GPT-2 style).
        # head computes x @ W.T, embedding lookup returns W[idx] — inverse ops
        # that benefit from sharing the same matrix.  Saves vocab_size×d_model
        # parameters and usually improves perplexity.
        self.head.weight = self.token_emb.weight

        mask_data = np.triu(
            np.full((context_len, context_len), -1e9, dtype=np.float64), k=1
        )
        self.causal_mask = Tensor(mask_data)

        self._init_weights()
        if lora_rank:
            self.enable_lora(lora_rank, lora_alpha)

    def _init_weights(self):
        """Apply the small normal initialization used by GPT models."""
        for parameter in self.parameters():
            if parameter.data.ndim >= 2:
                parameter.data[:] = np.random.randn(*parameter.data.shape) * 0.02
            else:
                parameter.data[:] = 0.0
        for name, parameter in self.named_parameters():
            if "gamma" in name:
                parameter.data[:] = 1.0

    def forward(self, idx) -> Tensor:
        """Return raw logits with shape (batch, time, vocab)."""
        _, time = idx.shape
        positions = np.arange(time, dtype=np.int64)
        x = self.emb_drop(self.token_emb(idx) + self.pos_emb(positions))
        mask = self.causal_mask[:time, :time]
        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.ln_f(x))

    def infer(self, idx, kv_cache=None):
        """NumPy-only inference path, returning logits and an updated KV cache."""
        _, time = idx.shape
        past_len = 0 if kv_cache is None else kv_cache[0]["k"].shape[2]
        if past_len + time > self.context_len:
            raise ValueError("inference input and cache exceed context_len")
        positions = np.arange(past_len, past_len + time, dtype=np.int64)

        x = self.token_emb.infer(idx) + self.pos_emb.infer(positions)
        caches = []
        for i, block in enumerate(self.blocks):
            block_cache = None if kv_cache is None else kv_cache[i]
            x, block_cache = block.infer(x, block_cache)
            caches.append(block_cache)
        return self.head.infer(self.ln_f.infer(x)), caches

    def generate(
        self,
        idx,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
        strategy: str = "sample",
        beam_width: int = 3,
        use_cache: bool = True,
    ):
        """Generate tokens with sampling, greedy decoding, or beam search."""
        if strategy == "beam":
            return self.generate_beam(idx, max_new_tokens, beam_width, temperature)
        if strategy not in {"sample", "greedy"}:
            raise ValueError("strategy must be 'sample', 'greedy', or 'beam'")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        idx = np.array(idx, dtype=np.int64, copy=True)
        kv_cache = None
        for _ in range(max_new_tokens):
            if use_cache:
                if kv_cache is None:
                    logits, kv_cache = self.infer(idx[:, -self.context_len:])
                else:
                    logits, kv_cache = self.infer(idx[:, -1:], kv_cache)
            else:
                logits, _ = self.infer(idx[:, -self.context_len:])

            logits_last = logits[:, -1, :]
            if strategy == "greedy":
                next_token = np.argmax(logits_last, axis=-1)
            else:
                next_token = np.array(
                    [
                        _sample(logit, temperature=temperature, top_k=top_k, top_p=top_p)
                        for logit in logits_last
                    ]
                )
            idx = np.concatenate([idx, next_token[:, None]], axis=1)

            # Sliding-window crops shift learned absolute positions.
            if kv_cache is not None and kv_cache[0]["k"].shape[2] >= self.context_len:
                kv_cache = None
        return idx

    def generate_beam(self, idx, max_new_tokens, beam_width=3, temperature=1.0):
        """Return the highest-scoring sequence from deterministic beam search."""
        if beam_width <= 0:
            raise ValueError("beam_width must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        idx = np.array(idx, dtype=np.int64, copy=True)
        if idx.shape[0] != 1:
            raise ValueError("beam search currently supports batch size 1")

        beams = [(idx, 0.0)]
        for _ in range(max_new_tokens):
            candidates = []
            for sequence, score in beams:
                logits, _ = self.infer(sequence[:, -self.context_len:])
                log_probs = _log_softmax(logits[0, -1] / temperature)
                best = np.argsort(log_probs)[-beam_width:]
                for token in best:
                    extended = np.concatenate([sequence, [[token]]], axis=1)
                    candidates.append((extended, score + float(log_probs[token])))
            beams = sorted(candidates, key=lambda item: item[1], reverse=True)[:beam_width]
        return beams[0][0]

    def enable_lora(self, rank, alpha=1.0):
        """Freeze the backbone and train only low-rank adapters on linear layers."""
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if self.lora_rank:
            if rank == self.lora_rank and alpha == self.lora_alpha:
                return self
            raise ValueError("LoRA adapters are already enabled")
        for _, tensor in self.named_tensors():
            tensor.requires_grad = False
            tensor.grad = None
        for module in self.modules():
            if isinstance(module, Linear):
                module.enable_lora(rank, alpha)
        self.lora_rank = rank
        self.lora_alpha = alpha
        return self

    def config(self):
        """Return constructor arguments needed to recreate this model."""
        return {
            "vocab_size": self.vocab_size,
            "context_len": self.context_len,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
        }

    @property
    def weight_tied(self):
        return self.head.weight is self.token_emb.weight

    def __repr__(self):
        tied = ", tied" if self.weight_tied else ""
        return (
            f"GPT(vocab={self.vocab_size}, ctx={self.context_len}, "
            f"d={self.d_model}, trainable_params={self.param_count():,}{tied})"
        )


def _gelu(x):
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))


def _log_softmax(logits):
    shifted = logits - logits.max()
    return shifted - np.log(np.exp(shifted).sum())


def _sample(logits, temperature=1.0, top_k=None, top_p=None):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive")
    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    filtered = np.array(logits, dtype=np.float64, copy=True) / temperature
    if top_k is not None and top_k < len(filtered):
        threshold = np.partition(filtered, -top_k)[-top_k]
        filtered[filtered < threshold] = -np.inf

    if top_p is not None and top_p < 1.0:
        order = np.argsort(filtered)[::-1]
        sorted_logits = filtered[order]
        sorted_probs = np.exp(sorted_logits - np.max(sorted_logits))
        sorted_probs /= sorted_probs.sum()
        remove = np.cumsum(sorted_probs) > top_p
        remove[1:] = remove[:-1].copy()
        remove[0] = False
        filtered[order[remove]] = -np.inf

    probs = np.exp(filtered - np.max(filtered))
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)
