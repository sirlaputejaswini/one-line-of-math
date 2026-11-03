"""
Gradient checks for the layer-math variants.

The math changes touch the attention backward pass (QK-norm especially:
gradients flow through two row normalizations and a learnable
temperature), so each attention mode gets the SAME two checks the
baseline implementation was validated with:

  1. Attention level: a masked key position gets exactly zero weight
     and exactly zero gradient (K and V rows), in this mode.
  2. Full model: analytic gradients from the hand-written backward pass
     vs central finite differences on the tiny model used by
     test_transformer.py (d_model 16, 2 heads, 1+1 layers). Reports the
     number of elements checked and the worst relative error.

A variant whose gradient check fails does not get its accuracy numbers
reported as valid — that rule is enforced in RESULTS-LAYER-MATH.md.

Run:  python3 gradcheck.py base|alibi|qknorm
Writes logs/gradcheck_<mode>.log and prints to stdout.
"""

import sys

import numpy as np

import data as D
from transformer import ScaledDotProductAttention, Transformer


def check_attention_mask(mode, rng):
    attn = ScaledDotProductAttention(mode=mode, dk=8, num_heads=4)
    q = rng.normal(size=(2, 4, 5, 8))
    k = rng.normal(size=(2, 4, 7, 8))
    v = rng.normal(size=(2, 4, 7, 8))
    mask = np.ones((2, 1, 5, 7))
    mask[:, :, :, 6] = 0  # last key blocked for every query
    out = attn.forward(q, k, v, mask=mask)
    assert out.shape == (2, 4, 5, 8)
    assert np.all(attn.weights[..., 6] == 0), f"[{mode}] masked position got nonzero weight"
    assert np.allclose(attn.weights.sum(-1), 1.0, atol=1e-9)
    dq, dk, dv = attn.backward(rng.normal(size=(2, 4, 5, 8)))
    assert np.allclose(dk[..., 6, :], 0.0, atol=1e-9), f"[{mode}] gradient leaked through mask (K)"
    assert np.allclose(dv[..., 6, :], 0.0, atol=1e-9), f"[{mode}] gradient leaked through mask (V)"
    return "mask/weight/gradient assertions passed"


def check_full_model_gradients(mode):
    model = Transformer(
        src_vocab=D.VOCAB_SIZE,
        tgt_vocab=D.VOCAB_SIZE,
        d_model=16,
        num_heads=2,
        num_layers=1,
        d_ff=32,
        pad_id=D.PAD,
        seed=3,
        attention_mode=mode,
    )
    src = np.array([[3, 4, 5, 6], [7, 8, D.PAD, D.PAD]])
    tgt = np.array([[D.BOS, 6, 5, 4, 3, D.EOS], [D.BOS, 8, 7, D.EOS, D.PAD, D.PAD]])
    tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]

    model.zero_grads()
    model.loss_and_backward(src, tgt_in, tgt_out, pad_id=D.PAD)
    pairs = model.params_and_grads()
    analytic_grads = [g.copy() for _, g in pairs]  # snapshot before FD calls

    eps = 1e-6
    checked, worst = 0, 0.0
    check_rng = np.random.default_rng(7)
    for (p, _), g in zip(pairs, analytic_grads):
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
    return checked, worst


def main():
    mode = sys.argv[1]
    lines = []

    def log(msg=""):
        print(msg, flush=True)
        lines.append(msg)

    rng = np.random.default_rng(0)
    log(f"Gradient check — attention mode: {mode}")
    log(f"  attention-level: {check_attention_mask(mode, rng)}")
    checked, worst = check_full_model_gradients(mode)
    status = "PASS" if worst < 1e-4 else "FAIL"
    log(f"  full-model gradient check: {checked} elements checked, "
        f"worst relative error {worst:.3e}  [{status}]")
    with open(f"logs/gradcheck_{mode}.log", "w") as f:
        f.write("\n".join(lines) + "\n")
    if worst >= 1e-4:
        sys.exit(1)


if __name__ == "__main__":
    main()
