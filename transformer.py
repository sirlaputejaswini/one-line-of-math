"""
Transformer, implemented from scratch in pure NumPy — a teaching
reimplementation of "Attention Is All You Need" (Vaswani et al., 2017).

Why NumPy: PyTorch could not be installed in this environment (its
download CDN stalled repeatedly behind the sandbox proxy), so this file
implements everything — forward AND backward passes — with explicit
NumPy array operations. That turns out to suit the purpose: no autograd
magic anywhere, every gradient is a formula you can read. A numerical
gradient check in the test file proves the backward passes are right.

Design rules:
  * Every mechanism the paper introduces is written out by hand:
    scaled dot-product attention, multi-head attention, sinusoidal
    positional encodings, encoder/decoder layers, masking.
  * Comments cite the paper section each piece implements.
  * Each module exposes params_and_grads() -> list of (param, grad)
    array pairs, which the hand-written Adam in train.py consumes.

Shapes: B = batch, S = source length, T = target length, D = d_model,
H = heads, dk = D / H.
"""

import numpy as np

MASK_FILL = -1e9  # stands in for -inf in masked softmax scores (see below)


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)  # stability: subtract row max
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# Primitives with hand-written gradients.
# ---------------------------------------------------------------------------
class Linear:
    """y = x @ W + b, applied over the last axis. (Paper's projections.)"""

    def __init__(self, rng, fan_in, fan_out):
        scale = np.sqrt(2.0 / (fan_in + fan_out))  # Xavier-style init
        self.W = rng.normal(0.0, scale, size=(fan_in, fan_out))
        self.b = np.zeros(fan_out)
        self.gW = np.zeros_like(self.W)
        self.gb = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dy):
        x2 = self.x.reshape(-1, self.x.shape[-1])
        dy2 = dy.reshape(-1, dy.shape[-1])
        self.gW += x2.T @ dy2
        self.gb += dy2.sum(axis=0)
        return dy @ self.W.T

    def params_and_grads(self):
        return [(self.W, self.gW), (self.b, self.gb)]


class Embedding:
    """Lookup table; backward scatter-adds gradients (np.add.at)."""

    def __init__(self, rng, vocab, d_model, pad_id=0):
        self.W = rng.normal(0.0, 1.0 / np.sqrt(d_model), size=(vocab, d_model))
        self.gW = np.zeros_like(self.W)
        self.pad_id = pad_id

    def forward(self, ids):
        self.ids = ids
        return self.W[ids]

    def backward(self, dy):
        np.add.at(self.gW, self.ids, dy)
        # The <pad> embedding is pinned to zero gradient (padding_idx idea):
        # pad positions are masked out of attention and ignored by the loss,
        # so nothing should ever train them.
        self.gW[self.pad_id] = 0.0
        return None  # ids are integers; no gradient flows further back

    def params_and_grads(self):
        return [(self.W, self.gW)]


class LayerNorm:
    """LayerNorm over the last axis (paper section 3.1, inside Add & Norm).
    Learnable gain gamma and bias beta, as in the paper's formulation."""

    def __init__(self, d_model, eps=1e-6):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.ggamma = np.zeros_like(self.gamma)
        self.gbeta = np.zeros_like(self.beta)
        self.eps = eps

    def forward(self, x):
        self.mu = x.mean(axis=-1, keepdims=True)
        self.var = x.var(axis=-1, keepdims=True)
        self.std = np.sqrt(self.var + self.eps)
        self.xhat = (x - self.mu) / self.std
        return self.gamma * self.xhat + self.beta

    def backward(self, dy):
        n = dy.shape[-1]
        self.ggamma += (dy * self.xhat).reshape(-1, n).sum(axis=0)
        self.gbeta += dy.reshape(-1, n).sum(axis=0)
        dxhat = dy * self.gamma
        # Standard LayerNorm input gradient:
        dx = (1.0 / (n * self.std)) * (
            n * dxhat
            - dxhat.sum(axis=-1, keepdims=True)
            - self.xhat * (dxhat * self.xhat).sum(axis=-1, keepdims=True)
        )
        return dx

    def params_and_grads(self):
        return [(self.gamma, self.ggamma), (self.beta, self.gbeta)]


# ---------------------------------------------------------------------------
# Paper section 3.2.1 — Scaled Dot-Product Attention
#   Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V
# Masked scores are set to -1e9 BEFORE the softmax (rather than -inf,
# which would make fully-masked rows NaN in NumPy); exp(-1e9 - max) is
# exactly 0.0 in float64, so blocked positions get exactly zero weight
# and — because softmax backward multiplies by the weight — exactly
# zero gradient. The test file asserts both properties numerically.
# ---------------------------------------------------------------------------
def _alibi_slopes(num_heads):
    """ALiBi per-head slopes (Press et al., 2022): for H a power of two,
    m_h = 2^(-8h/H), h = 1..H. (The general non-power-of-two recipe in
    the paper is not needed here: this codebase uses H in {2, 4}.)"""
    assert num_heads & (num_heads - 1) == 0, "ALiBi slopes here assume H a power of two"
    return 2.0 ** (-8.0 * np.arange(1, num_heads + 1) / num_heads)


class ScaledDotProductAttention:
    """Attention with a switchable score function — the ONLY thing the
    layer-math experiment changes. Modes:

      "base"   : scores = Q K^T / sqrt(dk)                     (baseline)
      "alibi"  : scores = Q K^T / sqrt(dk) - m_h * |i - j|     (V1)
                 per-head linear distance penalty, subtracted BEFORE
                 the softmax; m_h from _alibi_slopes. No new params:
                 the penalty is a constant, so the backward pass is
                 identical to baseline.
      "qknorm" : scores = tau * cos(Q, K)                      (V2)
                 Q and K rows L2-normalized; the 1/sqrt(dk) factor is
                 replaced by ONE learnable scalar temperature tau per
                 attention module, initialized to sqrt(dk).

    Masking is applied after the score function in every mode, exactly
    as in the baseline, so masked positions keep exactly zero weight
    and exactly zero gradient in all three modes.
    """

    def __init__(self, mode="base", dk=None, num_heads=None):
        assert mode in ("base", "alibi", "qknorm")
        self.mode = mode
        self.slopes = _alibi_slopes(num_heads) if mode == "alibi" else None
        if mode == "qknorm":
            # Learnable temperature (replaces the fixed 1/sqrt(dk)).
            self.tau = np.array([np.sqrt(float(dk))])
            self.gtau = np.zeros_like(self.tau)
        else:
            self.tau = None
            self.gtau = None

    def forward(self, q, k, v, mask=None):
        # q: (B,H,Lq,dk)  k: (B,H,Lk,dk)  v: (B,H,Lk,dv)
        self.dk = q.shape[-1]
        if self.mode == "qknorm":
            # L2-normalize each Q/K row: y = x / r, r = sqrt(||x||^2 + eps).
            self.rq = np.sqrt((q * q).sum(axis=-1, keepdims=True) + 1e-12)
            self.rk = np.sqrt((k * k).sum(axis=-1, keepdims=True) + 1e-12)
            self.qn = q / self.rq
            self.kn = k / self.rk
            self.cos = self.qn @ self.kn.transpose(0, 1, 3, 2)
            scores = self.tau * self.cos
        else:
            scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(self.dk)
            if self.mode == "alibi":
                lq, lk = q.shape[2], k.shape[2]
                dist = np.abs(
                    np.arange(lq)[:, None] - np.arange(lk)[None, :]
                ).astype(np.float64)  # (Lq, Lk)
                scores = scores - self.slopes[None, :, None, None] * dist[None, None, :, :]
        if mask is not None:
            scores = np.where(mask == 0, MASK_FILL, scores)
        self.weights = softmax(scores, axis=-1)
        self.k, self.v, self.q = k, v, q
        return self.weights @ v

    def backward(self, dout):
        w = self.weights
        dweights = dout @ self.v.transpose(0, 1, 3, 2)
        dv = w.transpose(0, 1, 3, 2) @ dout
        # Softmax Jacobian: ds = w * (dw - sum(dw * w))
        dscores = w * (dweights - (dweights * w).sum(axis=-1, keepdims=True))
        if self.mode == "qknorm":
            # scores = tau * cos(qn, kn): d tau = sum(ds * cos),
            # d cos = ds * tau, then back through the row normalization:
            # for y = x / r, dx = (dy - y * (dy . y)) / r.
            self.gtau += np.array([(dscores * self.cos).sum()])
            dcos = dscores * self.tau
            dqn = dcos @ self.kn
            dkn = dcos.transpose(0, 1, 3, 2) @ self.qn
            dq = (dqn - self.qn * (dqn * self.qn).sum(axis=-1, keepdims=True)) / self.rq
            dk = (dkn - self.kn * (dkn * self.kn).sum(axis=-1, keepdims=True)) / self.rk
            return dq, dk, dv
        # "base" and "alibi" share this path: the ALiBi penalty is a
        # constant offset to the scores, so it contributes no gradient.
        dscores = dscores / np.sqrt(self.dk)
        dq = dscores @ self.k
        dk = dscores.transpose(0, 1, 3, 2) @ self.q
        return dq, dk, dv

    def params_and_grads(self):
        if self.mode == "qknorm":
            return [(self.tau, self.gtau)]
        return []


# ---------------------------------------------------------------------------
# Paper section 3.2.2 — Multi-Head Attention
#   MultiHead(Q,K,V) = Concat(head_1..head_h) W^O,
#   head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
# One Linear per role (the W_i stacked), split along features into heads.
# ---------------------------------------------------------------------------
class MultiHeadAttention:
    def __init__(self, rng, d_model, num_heads, attention_mode="base"):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.h = num_heads
        self.dk = d_model // num_heads
        self.wq = Linear(rng, d_model, d_model)
        self.wk = Linear(rng, d_model, d_model)
        self.wv = Linear(rng, d_model, d_model)
        self.wo = Linear(rng, d_model, d_model)
        self.attn = ScaledDotProductAttention(
            mode=attention_mode, dk=self.dk, num_heads=num_heads
        )

    def _split(self, x):
        b, l, _ = x.shape
        return x.reshape(b, l, self.h, self.dk).transpose(0, 2, 1, 3)

    def _merge(self, x):
        b, _, l, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(b, l, self.d_model)

    def forward(self, query, key, value, mask=None):
        self.q_in, self.k_in, self.v_in = query, key, value
        q = self._split(self.wq.forward(query))
        k = self._split(self.wk.forward(key))
        v = self._split(self.wv.forward(value))
        out = self.attn.forward(q, k, v, mask=mask)
        return self.wo.forward(self._merge(out))

    def backward(self, dy):
        d = self.wo.backward(dy)
        d = d.reshape(d.shape[0], d.shape[1], self.h, self.dk).transpose(0, 2, 1, 3)
        dq, dk, dv = self.attn.backward(d)
        dq = self.wq.backward(dq.transpose(0, 2, 1, 3).reshape(dq.shape[0], -1, self.d_model))
        dk = self.wk.backward(dk.transpose(0, 2, 1, 3).reshape(dk.shape[0], -1, self.d_model))
        dv = self.wv.backward(dv.transpose(0, 2, 1, 3).reshape(dv.shape[0], -1, self.d_model))
        # Self-attention passes the same tensor as q/k/v input: gradients add.
        if self.q_in is self.k_in and self.k_in is self.v_in:
            return dq + dk + dv
        return dq, dk, dv

    def params_and_grads(self):
        out = []
        for m in (self.wq, self.wk, self.wv, self.wo):
            out += m.params_and_grads()
        out += self.attn.params_and_grads()  # tau in qknorm mode, else empty
        return out


# ---------------------------------------------------------------------------
# Paper section 3.2.3 — Position-wise Feed-Forward Network
#   FFN(x) = max(0, x W1 + b1) W2 + b2
# ---------------------------------------------------------------------------
class FeedForward:
    def __init__(self, rng, d_model, d_ff):
        self.l1 = Linear(rng, d_model, d_ff)
        self.l2 = Linear(rng, d_ff, d_model)

    def forward(self, x):
        self.pre = self.l1.forward(x)
        self.act = np.maximum(0.0, self.pre)  # ReLU
        return self.l2.forward(self.act)

    def backward(self, dy):
        d = self.l2.backward(dy)
        d = d * (self.pre > 0)  # ReLU derivative
        return self.l1.backward(d)

    def params_and_grads(self):
        return self.l1.params_and_grads() + self.l2.params_and_grads()


# ---------------------------------------------------------------------------
# Paper section 3.5 — Sinusoidal Positional Encoding (fixed, not learned)
#   PE(pos, 2i) = sin(pos / 10000^(2i/D)), PE(pos, 2i+1) = cos(...)
# ---------------------------------------------------------------------------
class PositionalEncoding:
    def __init__(self, d_model, max_len=512):
        pe = np.zeros((max_len, d_model))
        pos = np.arange(max_len)[:, None]
        div = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.pe = pe[None, :, :]  # (1, max_len, D)

    def forward(self, x):
        return x + self.pe[:, : x.shape[1]]

    def backward(self, dy):
        return dy  # fixed table: gradient passes straight through the add


# ---------------------------------------------------------------------------
# Masks (paper sections 3.1 / 5.1). 1 = may attend, 0 = blocked.
# ---------------------------------------------------------------------------
def pad_mask(ids, pad_id):
    """(B, L) -> (B, 1, 1, L) mask over KEY positions."""
    return (ids != pad_id)[:, None, None, :].astype(np.float64)


def look_ahead_mask(size):
    """(1, 1, T, T) lower-triangular: position i may see j <= i."""
    return np.tril(np.ones((size, size)))[None, None, :, :]


def tgt_mask(tgt_ids, pad_id):
    return pad_mask(tgt_ids, pad_id) * look_ahead_mask(tgt_ids.shape[1])


# ---------------------------------------------------------------------------
# Paper section 3.1 — Encoder layer: self-attention, Add & Norm, FFN,
# Add & Norm. (Post-LN ordering, exactly as the original paper: residual
# add first, LayerNorm applied to the sum.)
# ---------------------------------------------------------------------------
class EncoderLayer:
    def __init__(self, rng, d_model, num_heads, d_ff, attention_mode="base"):
        self.attn = MultiHeadAttention(rng, d_model, num_heads, attention_mode)
        self.ffn = FeedForward(rng, d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x, src_mask=None):
        a = self.attn.forward(x, x, x, mask=src_mask)
        self._after_attn = x + a                       # residual branch 1
        h = self.norm1.forward(self._after_attn)
        f = self.ffn.forward(h)
        self._after_ffn = h + f                        # residual branch 2
        return self.norm2.forward(self._after_ffn)

    def backward(self, dy):
        d = self.norm2.backward(dy)
        dh = d + self.ffn.backward(d)                  # split at the add
        d = self.norm1.backward(dh)
        dx = d + self.attn.backward(d)
        return dx

    def params_and_grads(self):
        return (
            self.attn.params_and_grads()
            + self.ffn.params_and_grads()
            + self.norm1.params_and_grads()
            + self.norm2.params_and_grads()
        )


# ---------------------------------------------------------------------------
# Paper section 3.1 — Decoder layer: MASKED self-attention, cross-attention
# over the encoder output, FFN — each with Add & Norm.
# ---------------------------------------------------------------------------
class DecoderLayer:
    def __init__(self, rng, d_model, num_heads, d_ff, attention_mode="base"):
        self.self_attn = MultiHeadAttention(rng, d_model, num_heads, attention_mode)
        self.cross_attn = MultiHeadAttention(rng, d_model, num_heads, attention_mode)
        self.ffn = FeedForward(rng, d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

    def forward(self, x, enc_out, t_mask=None, s_mask=None):
        a = self.self_attn.forward(x, x, x, mask=t_mask)
        h1 = self.norm1.forward(x + a)
        c = self.cross_attn.forward(h1, enc_out, enc_out, mask=s_mask)
        h2 = self.norm2.forward(h1 + c)
        f = self.ffn.forward(h2)
        return self.norm3.forward(h2 + f)

    def backward(self, dy):
        d = self.norm3.backward(dy)
        dh2 = d + self.ffn.backward(d)
        d = self.norm2.backward(dh2)
        dh1, denc_from_cross, _ = _split_cross_grads(self.cross_attn, d)
        dh1_total = d + dh1
        d = self.norm1.backward(dh1_total)
        dx = d + self.self_attn.backward(d)
        return dx, denc_from_cross

    def params_and_grads(self):
        return (
            self.self_attn.params_and_grads()
            + self.cross_attn.params_and_grads()
            + self.ffn.params_and_grads()
            + self.norm1.params_and_grads()
            + self.norm2.params_and_grads()
            + self.norm3.params_and_grads()
        )


def _split_cross_grads(attn_module, d):
    """Cross-attention backward returns (d_query, d_key, d_value); keys and
    values are the same encoder tensor, so their gradients add."""
    dq, dk, dv = attn_module.backward(d)
    return dq, dk + dv, None


# ---------------------------------------------------------------------------
# The full model (paper Figure 1): N encoder layers, N decoder layers,
# embedding scaling by sqrt(d_model) (section 3.4), final LayerNorm and
# linear projection to vocabulary logits.
# ---------------------------------------------------------------------------
class Transformer:
    def __init__(
        self,
        src_vocab,
        tgt_vocab,
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=512,
        max_len=512,
        pad_id=0,
        seed=0,
        attention_mode="base",
    ):
        rng = np.random.default_rng(seed)
        self.d_model = d_model
        self.pad_id = pad_id
        self.attention_mode = attention_mode
        self.src_emb = Embedding(rng, src_vocab, d_model, pad_id)
        self.tgt_emb = Embedding(rng, tgt_vocab, d_model, pad_id)
        self.pos = PositionalEncoding(d_model, max_len)
        self.enc_layers = [EncoderLayer(rng, d_model, num_heads, d_ff, attention_mode) for _ in range(num_layers)]
        self.dec_layers = [DecoderLayer(rng, d_model, num_heads, d_ff, attention_mode) for _ in range(num_layers)]
        self.final_norm = LayerNorm(d_model)
        self.out = Linear(rng, d_model, tgt_vocab)

    # -- forward pieces ----------------------------------------------------
    def encode(self, src_ids, s_mask):
        x = self.pos.forward(self.src_emb.forward(src_ids) * np.sqrt(self.d_model))
        for layer in self.enc_layers:
            x = layer.forward(x, s_mask)
        self._enc_out = x
        return x

    def decode(self, tgt_ids, enc_out, t_mask, s_mask):
        x = self.pos.forward(self.tgt_emb.forward(tgt_ids) * np.sqrt(self.d_model))
        for layer in self.dec_layers:
            x = layer.forward(x, enc_out, t_mask, s_mask)
        return self.final_norm.forward(x)

    def forward(self, src_ids, tgt_ids):
        s_mask = pad_mask(src_ids, self.pad_id)
        t_mask = tgt_mask(tgt_ids, self.pad_id)
        enc = self.encode(src_ids, s_mask)
        dec = self.decode(tgt_ids, enc, t_mask, s_mask)
        return self.out.forward(dec)  # logits (B, T, V)

    # -- loss + full backward ----------------------------------------------
    def loss_and_backward(self, src_ids, tgt_in, tgt_out, pad_id=0):
        """Mean softmax cross-entropy over non-pad label positions, then a
        full backward pass. Returns the scalar loss."""
        logits = self.forward(src_ids, tgt_in)
        b, t, v = logits.shape
        probs = softmax(logits, axis=-1)
        mask = tgt_out != pad_id
        n = mask.sum()
        # Loss: -mean log p(correct token) over real label positions.
        correct_logp = np.log(np.take_along_axis(probs, tgt_out[:, :, None], axis=-1)[:, :, 0] + 1e-12)
        loss = -(correct_logp * mask).sum() / n
        # d loss / d logits = (probs - onehot) / n at real positions, else 0.
        dlogits = probs.copy()
        rows, cols = np.nonzero(mask)
        dlogits[rows, cols, tgt_out[rows, cols]] -= 1.0
        dlogits = dlogits * (mask[:, :, None] / n)

        # Backward: output projection -> final norm -> decoder stack
        # (accumulating gradients into the encoder via cross-attention)
        # -> encoder stack -> embeddings.
        d = self.out.backward(dlogits)
        d = self.final_norm.backward(d)
        d_enc_total = np.zeros_like(self._enc_out)
        d_tgt = d
        for layer in reversed(self.dec_layers):
            d_tgt, d_enc = layer.backward(d_tgt)
            d_enc_total += d_enc
        d_tgt_scaled = self.pos.backward(d_tgt) * np.sqrt(self.d_model)
        self.tgt_emb.backward(d_tgt_scaled)
        d_src = d_enc_total
        for layer in reversed(self.enc_layers):
            d_src = layer.backward(d_src)
        d_src_scaled = self.pos.backward(d_src) * np.sqrt(self.d_model)
        self.src_emb.backward(d_src_scaled)
        return loss

    # -- inference ----------------------------------------------------------
    def predict_logits(self, src_ids, tgt_ids):
        return self.forward(src_ids, tgt_ids)

    def generate(self, src_ids, bos_id, eos_id, max_len=32):
        """Greedy autoregressive decoding of ONE source sequence."""
        src = np.array([src_ids], dtype=np.int64)
        s_mask = pad_mask(src, self.pad_id)
        enc = self.encode(src, s_mask)
        tgt = [bos_id]
        for _ in range(max_len):
            t = np.array([tgt], dtype=np.int64)
            dec = self.decode(t, enc, tgt_mask(t, self.pad_id), s_mask)
            logits = self.out.forward(dec)
            nxt = int(logits[0, -1].argmax())
            tgt.append(nxt)
            if nxt == eos_id:
                break
        return tgt

    # -- bookkeeping ---------------------------------------------------------
    def params_and_grads(self):
        pairs = []
        pairs += self.src_emb.params_and_grads()
        pairs += self.tgt_emb.params_and_grads()
        for layer in self.enc_layers + self.dec_layers:
            pairs += layer.params_and_grads()
        pairs += self.final_norm.params_and_grads()
        pairs += self.out.params_and_grads()
        return pairs

    def zero_grads(self):
        for _, g in self.params_and_grads():
            g[...] = 0.0

    def num_params(self):
        return int(sum(p.size for p, _ in self.params_and_grads()))
