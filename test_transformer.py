"""
Behavioural tests for the from-scratch NumPy Transformer.

Run:  python3 test_transformer.py     (plain asserts, no framework needed)
  or  python3 -m pytest test_transformer.py -v

These test *properties the paper's mechanisms must have*, not just shapes:
  1. Scaled dot-product attention: right shapes; a masked key position
     gets exactly zero attention weight (and zero gradient).
  2. Positional encodings match the closed-form sin/cos values.
  3. CAUSALITY: changing a FUTURE target token must not change logits at
     earlier positions. If this fails, the look-ahead mask is broken and
     teacher forcing would be cheating.
  4. Full model end-to-end output shape.
  5. GRADIENT CHECK: analytic gradients from the hand-written backward
     pass vs central finite differences on a tiny model. This is the
     test that makes "implemented from scratch" mean something — if any
     backward formula is wrong, this catches it.
"""

import numpy as np

import data as D
from transformer import (
    MultiHeadAttention,
    PositionalEncoding,
    ScaledDotProductAttention,
    Transformer,
    look_ahead_mask,
)

rng = np.random.default_rng(0)


def small_model(seed=1):
    return Transformer(
        src_vocab=D.VOCAB_SIZE,
        tgt_vocab=D.VOCAB_SIZE,
        d_model=32,
        num_heads=4,
        num_layers=2,
        d_ff=64,
        pad_id=D.PAD,
        seed=seed,
    )


def test_attention_shapes_and_mask_zero_weight():
    attn = ScaledDotProductAttention()
    q = rng.normal(size=(2, 4, 5, 8))
    k = rng.normal(size=(2, 4, 7, 8))
    v = rng.normal(size=(2, 4, 7, 8))
    mask = np.ones((2, 1, 5, 7))
    mask[:, :, :, 6] = 0  # last key blocked for every query
    out = attn.forward(q, k, v, mask=mask)
    assert out.shape == (2, 4, 5, 8), out.shape
    assert attn.weights.shape == (2, 4, 5, 7), attn.weights.shape
    assert np.all(attn.weights[..., 6] == 0), "masked position got nonzero weight"
    assert np.allclose(attn.weights.sum(-1), 1.0, atol=1e-9)
    # And zero gradient flows to the blocked key/value rows:
    dq, dk, dv = attn.backward(rng.normal(size=(2, 4, 5, 8)))
    assert np.allclose(dk[..., 6, :], 0.0, atol=1e-9), "gradient leaked through mask (K)"
    assert np.allclose(dv[..., 6, :], 0.0, atol=1e-9), "gradient leaked through mask (V)"


def test_multihead_attention_shape():
    mha = MultiHeadAttention(np.random.default_rng(0), d_model=32, num_heads=4)
    x = rng.normal(size=(3, 6, 32))
    assert mha.forward(x, x, x).shape == (3, 6, 32)


def test_positional_encoding_values():
    pe = PositionalEncoding(d_model=16, max_len=10).pe[0]  # (max_len, D)
    assert np.allclose(pe[0, 0::2], 0.0, atol=1e-12)  # sin(0)
    assert np.allclose(pe[0, 1::2], 1.0, atol=1e-12)  # cos(0)
    angle = 1.0 / (10000.0 ** (2.0 / 16.0))           # pos=1, i=1
    assert abs(pe[1, 2] - np.sin(angle)) < 1e-12
    assert abs(pe[1, 3] - np.cos(angle)) < 1e-12
    assert np.abs(pe).max() <= 1.0 + 1e-12


def test_look_ahead_mask_is_lower_triangular():
    m = look_ahead_mask(4)[0, 0]
    expected = np.array(
        [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]], dtype=float
    )
    assert np.array_equal(m, expected)


def test_causality_future_token_cannot_change_past_logits():
    model = small_model()
    src = np.array([[3, 5, 7, 9]])
    tgt_a = np.array([[D.BOS, 4, 6, 8, 10]])
    tgt_b = np.array([[D.BOS, 4, 6, 8, 12]])  # only the LAST token differs
    logits_a = model.predict_logits(src, tgt_a)
    logits_b = model.predict_logits(src, tgt_b)
    assert np.allclose(logits_a[:, :4], logits_b[:, :4], atol=1e-10), (
        "future target token changed earlier logits — look-ahead mask broken"
    )


def test_full_model_output_shape():
    model = small_model()
    src = np.array([[3, 4, 5], [6, 7, D.PAD]])
    tgt = np.array([[D.BOS, 5, 4], [D.BOS, 7, 6]])
    logits = model.predict_logits(src, tgt)
    assert logits.shape == (2, 3, D.VOCAB_SIZE), logits.shape


def test_gradient_check_tiny_model():
    """Central finite differences vs analytic gradients, all parameters."""
    model = Transformer(
        src_vocab=D.VOCAB_SIZE,
        tgt_vocab=D.VOCAB_SIZE,
        d_model=16,
        num_heads=2,
        num_layers=1,
        d_ff=32,
        pad_id=D.PAD,
        seed=3,
    )
    src = np.array([[3, 4, 5, 6], [7, 8, D.PAD, D.PAD]])
    tgt = np.array([[D.BOS, 6, 5, 4, 3, D.EOS], [D.BOS, 8, 7, D.EOS, D.PAD, D.PAD]])
    tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]

    model.zero_grads()
    model.loss_and_backward(src, tgt_in, tgt_out, pad_id=D.PAD)
    pairs = model.params_and_grads()
    # Snapshot analytic grads NOW: loss_and_backward accumulates into the
    # grad arrays, and the finite-difference calls below would corrupt
    # them if we read them lazily inside the loop.
    analytic_grads = [g.copy() for _, g in pairs]

    eps = 1e-6
    checked, worst = 0, 0.0
    check_rng = np.random.default_rng(7)
    for (p, _), g in zip(pairs, analytic_grads):
        # Check every element of small arrays, a random sample of big ones.
        idxs = list(np.ndindex(p.shape))
        if len(idxs) > 60:
            idxs = [idxs[i] for i in check_rng.choice(len(idxs), 60, replace=False)]
        for idx in idxs:
            orig = p[idx]
            p[idx] = orig + eps
            lp = model.loss_and_backward(src, tgt_in, tgt_out, pad_id=D.PAD)
            p[idx] = orig - eps
            lm = model.loss_and_backward(src, tgt_in, tgt_out, pad_id=D.PAD)
            p[idx] = orig
            numeric = (lp - lm) / (2 * eps)
            analytic = g[idx]
            rel = abs(numeric - analytic) / max(1.0, abs(numeric), abs(analytic))
            worst = max(worst, rel)
            checked += 1
    print(f"  gradient check: {checked} elements, worst relative error {worst:.2e}")
    assert worst < 1e-4, f"gradient check failed: worst rel error {worst}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed")
