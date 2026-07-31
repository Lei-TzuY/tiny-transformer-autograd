"""Feed-forward, transformer block, and tiny GPT modules."""

import numpy as np

from engine.tensor import Tensor
import engine.ops as ops
from engine.recompute import recompute
from .module import Module
from .layers import Linear, Embedding, LayerNorm, RMSNorm, Dropout
from .attention import MultiHeadAttention, RotaryEmbedding


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


class SwiGLU(Module):
    """
    Gated feed-forward network (Shazeer 2020), used by Llama.

    y = W_down( SiLU(W_gate·x) ⊙ W_up·x )

    The gate path modulates the up path element-wise; no biases, matching
    modern practice.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        self.fc_gate = Linear(d_model, d_ff, bias=False)
        self.fc_up = Linear(d_model, d_ff, bias=False)
        self.fc_down = Linear(d_ff, d_model, bias=False)
        self.drop = Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        h = ops.silu(self.fc_gate(x)) * self.fc_up(x)
        return self.fc_down(self.drop(h))

    def infer(self, x):
        g = self.fc_gate.infer(x)
        return self.fc_down.infer(g * _sigmoid(g) * self.fc_up.infer(x))


_NORMS = {"layernorm": LayerNorm, "rmsnorm": RMSNorm}
_FFNS = {"gelu": FeedForward, "swiglu": SwiGLU}


class TransformerBlock(Module):
    """One decoder-only pre-norm transformer block."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        norm: str = "layernorm",
        ffn: str = "gelu",
        rope: RotaryEmbedding = None,
    ):
        if norm not in _NORMS:
            raise ValueError(f"norm must be one of {sorted(_NORMS)}")
        if ffn not in _FFNS:
            raise ValueError(f"ffn must be one of {sorted(_FFNS)}")
        norm_cls = _NORMS[norm]
        self.ln1 = norm_cls(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout, rope=rope)
        self.ln2 = norm_cls(d_model)
        self.ff = _FFNS[ffn](d_model, d_ff, dropout)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        x = x + self.attn(self.ln1(x), mask)
        return x + self.ff(self.ln2(x))

    def infer(self, x, cache=None, key_bias=None, positions=None):
        attn, cache = self.attn.infer(
            self.ln1.infer(x), cache, key_bias=key_bias, positions=positions
        )
        x = x + attn
        return x + self.ff.infer(self.ln2.infer(x)), cache


class GPT(Module):
    """
    Tiny decoder-only GPT language model.

    Architecture knobs (defaults reproduce GPT-2 style; the alternatives
    together give a Llama-style model):
      norm         : "layernorm" | "rmsnorm"
      pos_encoding : "learned"   | "rope"
      ffn          : "gelu"      | "swiglu"
    """

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
        norm: str = "layernorm",
        pos_encoding: str = "learned",
        ffn: str = "gelu",
        grad_checkpoint: bool = False,
    ):
        dimensions = {
            "vocab_size": vocab_size,
            "context_len": context_len,
            "d_model": d_model,
            "num_heads": num_heads,
            "d_ff": d_ff,
            "num_layers": num_layers,
        }
        invalid = [name for name, value in dimensions.items() if value <= 0]
        if invalid:
            raise ValueError(f"{', '.join(invalid)} must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if norm not in _NORMS:
            raise ValueError(f"norm must be one of {sorted(_NORMS)}")
        if ffn not in _FFNS:
            raise ValueError(f"ffn must be one of {sorted(_FFNS)}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if lora_rank < 0 or lora_alpha <= 0:
            raise ValueError("lora_rank must be non-negative and lora_alpha positive")
        if pos_encoding not in {"learned", "rope"}:
            raise ValueError("pos_encoding must be 'learned' or 'rope'")
        if pos_encoding == "rope" and (d_model // num_heads) % 2 != 0:
            raise ValueError("RoPE requires an even head dimension")
        self.vocab_size = vocab_size
        self.context_len = context_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.dropout = dropout
        self.lora_rank = 0
        self.lora_alpha = lora_alpha
        self.norm = norm
        self.pos_encoding = pos_encoding
        self.ffn = ffn
        # Runtime memory/compute toggle, not part of the architecture: each
        # block's activations are recomputed in the backward pass instead of
        # being kept. Weights, outputs, and gradients are unaffected, so it is
        # deliberately absent from config() and safe to flip on a resumed run.
        self.grad_checkpoint = bool(grad_checkpoint)

        self.token_emb = Embedding(vocab_size, d_model)
        if pos_encoding == "learned":
            self.pos_emb = Embedding(context_len, d_model)
            rope = None
        else:
            self.pos_emb = None
            rope = RotaryEmbedding(d_model // num_heads, context_len)
        self.rope = rope
        self.emb_drop = Dropout(dropout)
        self.blocks = [
            TransformerBlock(
                d_model, num_heads, d_ff, dropout, norm=norm, ffn=ffn, rope=rope
            )
            for _ in range(num_layers)
        ]
        self.ln_f = _NORMS[norm](d_model)
        self.head = Linear(d_model, vocab_size, bias=False)
        # Weight tying: the LM head reuses the embedding table (GPT-2 style).
        # head computes x @ W.T, embedding lookup returns W[idx] — inverse ops
        # that benefit from sharing the same matrix.  Saves vocab_size×d_model
        # parameters and usually improves perplexity.
        self.head.weight = self.token_emb.weight

        mask_data = np.triu(
            np.full((context_len, context_len), -np.inf, dtype=np.float64), k=1
        )
        self.causal_mask = Tensor(mask_data)

        self._init_weights()
        if lora_rank:
            self.enable_lora(lora_rank, lora_alpha)

    def load_state_dict(self, state, strict=True):
        """Load weights and rebuild the deterministic strict causal buffer."""
        super().load_state_dict(state, strict=strict)
        self.causal_mask.data[:] = np.triu(
            np.full(
                (self.context_len, self.context_len),
                -np.inf,
                dtype=np.float64,
            ),
            k=1,
        )

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

    def forward(self, idx, attention_mask=None) -> Tensor:
        """
        Return raw logits with shape (batch, time, vocab).

        Parameters
        ----------
        idx : integer array, shape (batch, time)
            Token ids, at most ``context_len`` long.
        attention_mask : array-like or None, shape (batch, time)
            Optional key-padding mask for variable-length batches, using the
            usual convention: 1/True marks a real token, 0/False marks padding.
            Padded keys are removed from every layer's attention in addition to
            the causal mask, so a real token's output is identical to running
            its sequence unpadded.  Pass the matching ``ignore_index`` to
            ``cross_entropy`` so padded *targets* are not scored either.

            Positions are taken to start at 0, so pad on the right — left
            padding would shift every learned/rotary position.
        """
        idx = self._validate_token_batch(idx, max_time=self.context_len)
        _, time = idx.shape
        x = self.token_emb(idx)
        if self.pos_emb is not None:
            positions = np.arange(time, dtype=np.int64)
            x = x + self.pos_emb(positions)
        x = self.emb_drop(x)
        mask = self.causal_mask[:time, :time]
        if attention_mask is not None:
            # (batch, 1, 1, time) broadcasts over (batch, heads, time, time);
            # -inf + -inf stays -inf, so combining with the causal mask is
            # just an add.
            mask = mask + Tensor(self._key_padding_bias(attention_mask, idx.shape))
        for block in self.blocks:
            if self.grad_checkpoint:
                x = recompute(lambda inp, blk=block: blk(inp, mask), x)
            else:
                x = block(x, mask)
        return self.head(self.ln_f(x))

    @staticmethod
    def _key_padding_bias(attention_mask, idx_shape):
        """Turn a (batch, time) keep/pad mask into an additive key bias."""
        keep = _validate_keep_mask(
            attention_mask, idx_shape, "matching the token ids"
        )
        return np.where(keep, 0.0, -np.inf)[:, np.newaxis, np.newaxis, :]

    def infer(self, idx, kv_cache=None, attention_mask=None, position_ids=None):
        """
        NumPy-only inference path, returning logits and an updated KV cache.

        Parameters
        ----------
        idx : integer array, shape (batch, time)
            The tokens to process now — the whole prompt, or one step when a
            cache is supplied.
        kv_cache : list or None
            Per-block key/value cache returned by a previous call.
        attention_mask : array-like or None, shape (batch, past + time)
            Keep/pad mask covering **every key**, cached ones included: padded
            prompt slots stay in the cache and must stay hidden on every step.
        position_ids : integer array or None, shape (batch, time)
            Absolute position of each new token. Required for left-padded
            batches, where rows share a slot index but not a position. Defaults
            to ``arange(past, past + time)`` for every row.
        """
        idx = self._validate_token_batch(idx, max_time=self.context_len)
        if kv_cache is not None and len(kv_cache) != len(self.blocks):
            raise ValueError("kv_cache must contain one entry per transformer block")
        batch, time = idx.shape
        past_len = 0 if kv_cache is None else kv_cache[0]["k"].shape[2]
        if past_len + time > self.context_len:
            raise ValueError("inference input and cache exceed context_len")

        key_bias = None
        if attention_mask is not None:
            keep = _validate_keep_mask(
                attention_mask,
                (batch, past_len + time),
                "covering the cached and current keys",
            )
            key_bias = np.where(keep, 0.0, -np.inf)[:, np.newaxis, np.newaxis, :]

        if position_ids is None:
            positions = np.arange(past_len, past_len + time, dtype=np.int64)
            rope_positions = None
        else:
            positions = self._validate_position_ids(position_ids, (batch, time))
            # (batch, 1, time) broadcasts over the (batch, heads, time, d_k) heads.
            rope_positions = positions[:, np.newaxis, :]

        x = self.token_emb.infer(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb.infer(positions)
        caches = []
        for i, block in enumerate(self.blocks):
            block_cache = None if kv_cache is None else kv_cache[i]
            x, block_cache = block.infer(
                x, block_cache, key_bias=key_bias, positions=rope_positions
            )
            caches.append(block_cache)
        return self.head.infer(self.ln_f.infer(x)), caches

    def _validate_position_ids(self, position_ids, expected_shape):
        positions = np.asarray(position_ids)
        if positions.shape != tuple(expected_shape):
            raise ValueError(
                f"position_ids must have shape {tuple(expected_shape)}, "
                f"got {positions.shape}"
            )
        if not np.issubdtype(positions.dtype, np.integer):
            raise TypeError("position_ids must be integers")
        if positions.min() < 0 or positions.max() >= self.context_len:
            raise ValueError(f"position_ids must be in [0, {self.context_len})")
        return positions

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
        attention_mask=None,
    ):
        """
        Generate tokens with sampling, greedy decoding, or beam search.

        ``attention_mask`` enables batched generation from prompts of different
        lengths. Pad each prompt on the **left** and mark the padding with 0;
        every row's newest token is then at slot -1, which is where the next
        distribution is read from. Padded slots stay hidden for the whole run
        and each row gets its own ``position_ids``, so a row's output is the
        same as generating that prompt on its own.

        The returned array keeps the left padding, so a row's generated tokens
        are the last ``max_new_tokens`` columns. A masked run may outgrow
        ``context_len``: it then slides the same window an unmasked run does,
        keeping each row's newest ``context_len`` slots and renumbering the
        surviving real tokens from 0 — a row that has not filled the window
        yet loses padding, one that has loses its oldest tokens.
        """
        if not isinstance(max_new_tokens, (int, np.integer)) or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a non-negative integer")
        idx = self._validate_token_batch(idx)
        if strategy == "beam":
            if attention_mask is not None:
                raise ValueError(
                    "beam search runs one sequence at a time and takes no "
                    "attention_mask"
                )
            return self.generate_beam(idx, max_new_tokens, beam_width, temperature)
        if strategy not in {"sample", "greedy"}:
            raise ValueError("strategy must be 'sample', 'greedy', or 'beam'")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        idx = np.array(idx, dtype=np.int64, copy=True)
        mask = positions = None
        if attention_mask is not None:
            mask = self._validate_generation_mask(attention_mask, idx.shape)
            # Filled per window below: a crop renumbers the surviving tokens,
            # so positions are only meaningful inside the current window.
            positions = np.zeros(idx.shape, dtype=np.int64)

        kv_cache = None
        for _ in range(max_new_tokens):
            if use_cache and kv_cache is not None:
                step_mask = step_positions = None
                if mask is not None:
                    # The mask must span the cached keys plus the new one, and
                    # the cache may start after the prompt if the window slid.
                    cached = kv_cache[0]["k"].shape[2]
                    step_mask = mask[:, -(cached + 1):]
                    step_positions = positions[:, -1:]
                logits, kv_cache = self.infer(
                    idx[:, -1:],
                    kv_cache,
                    attention_mask=step_mask,
                    position_ids=step_positions,
                )
            else:
                window = idx[:, -self.context_len:]
                window_mask = window_positions = None
                if mask is not None:
                    width = window.shape[1]
                    window_mask = mask[:, -width:]
                    # A crop restarts the window at position 0, exactly as the
                    # unmasked path does; each row is numbered from its own
                    # first real token that survived the crop.
                    positions[:, -width:] = _left_padded_positions(window_mask)
                    window_positions = positions[:, -width:]
                logits, cache = self.infer(
                    window,
                    attention_mask=window_mask,
                    position_ids=window_positions,
                )
                kv_cache = cache if use_cache else None

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
            if mask is not None:
                # Generated tokens are real, and continue each row's own count.
                mask = np.concatenate(
                    [mask, np.ones((mask.shape[0], 1), dtype=bool)], axis=1
                )
                positions = np.concatenate(
                    [positions, positions[:, -1:] + 1], axis=1
                )

            # A full cache means the next step must crop, and a crop shifts
            # absolute positions. Dropping the cache forces the re-prefill that
            # renumbers the window instead of decoding against stale positions.
            if kv_cache is not None and kv_cache[0]["k"].shape[2] >= self.context_len:
                kv_cache = None
        return idx

    @staticmethod
    def _validate_generation_mask(attention_mask, idx_shape):
        keep = _validate_keep_mask(
            attention_mask, idx_shape, "matching the prompt tokens"
        )
        if not keep.any(axis=1).all():
            raise ValueError(
                "every generation prompt must contain at least one real token"
            )
        if np.any(np.diff(keep.astype(np.int8), axis=1) < 0):
            raise ValueError(
                "generation attention_mask must be left-padded: each row must "
                "be zeros followed by ones"
            )
        return keep

    def generate_beam(self, idx, max_new_tokens, beam_width=3, temperature=1.0):
        """Return the highest-scoring sequence from deterministic beam search."""
        if not isinstance(max_new_tokens, (int, np.integer)) or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a non-negative integer")
        if beam_width <= 0:
            raise ValueError("beam_width must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        idx = np.array(self._validate_token_batch(idx), dtype=np.int64, copy=True)
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
            "norm": self.norm,
            "pos_encoding": self.pos_encoding,
            "ffn": self.ffn,
        }

    def _validate_token_batch(self, idx, max_time=None):
        idx = np.asarray(idx)
        if idx.ndim != 2:
            raise ValueError("token ids must have shape (batch, time)")
        if idx.shape[0] == 0 or idx.shape[1] == 0:
            raise ValueError("token batch and prompt must not be empty")
        if max_time is not None and idx.shape[1] > max_time:
            raise ValueError(f"token sequence exceeds context_len={max_time}")
        if not np.issubdtype(idx.dtype, np.integer):
            raise TypeError("token ids must be integers")
        if np.any(idx < 0) or np.any(idx >= self.vocab_size):
            raise ValueError(f"token ids must be in [0, {self.vocab_size})")
        return idx

    @property
    def weight_tied(self):
        return self.head.weight is self.token_emb.weight

    def __repr__(self):
        tied = ", tied" if self.weight_tied else ""
        arch = ""
        if (self.norm, self.pos_encoding, self.ffn) != ("layernorm", "learned", "gelu"):
            arch = f", {self.norm}+{self.pos_encoding}+{self.ffn}"
        recomputing = ", grad_checkpoint" if self.grad_checkpoint else ""
        return (
            f"GPT(vocab={self.vocab_size}, ctx={self.context_len}, "
            f"d={self.d_model}, trainable_params={self.param_count():,}"
            f"{tied}{arch}{recomputing})"
        )


def _validate_keep_mask(attention_mask, expected_shape, description):
    """Validate a keep/pad mask (1/True keeps) and return it as booleans."""
    mask = np.asarray(attention_mask)
    expected_shape = tuple(expected_shape)
    if mask.shape != expected_shape:
        raise ValueError(
            f"attention_mask must have shape {expected_shape} {description}, "
            f"got {mask.shape}"
        )
    if mask.dtype != bool:
        if not np.issubdtype(mask.dtype, np.number):
            raise TypeError("attention_mask must be boolean or numeric")
        if not np.all((mask == 0) | (mask == 1)):
            raise ValueError(
                "attention_mask entries must be 0/False (padding) or "
                "1/True (real token)"
            )
    return mask.astype(bool)


def _left_padded_positions(keep):
    """Number each row's real tokens from 0; padded slots collapse to 0.

    Padded slots are never attended to, so their position is arbitrary — 0 is
    the only value guaranteed to be inside the learned/rotary position table.
    """
    return np.maximum(np.cumsum(keep, axis=1) - 1, 0)


def _gelu(x):
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))


def _sigmoid(x):
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


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
