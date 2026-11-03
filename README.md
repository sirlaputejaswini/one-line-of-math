# One Line of Math

A from-scratch Transformer — pure NumPy, every gradient written by hand —
learns to reverse short sequences perfectly: **500/500 exact match** on
held-out sequences of the lengths it trained on. Show it a sequence just a
few tokens longer than anything it has ever seen and it collapses to
**0/500**. Not worse. Zero.

This repo asks a narrow, honest question: can you fix that cliff by changing
**one line of math** inside the model — and nothing else?

## WHY change the math in a layer?

**Length generalization** is a model's ability to handle inputs longer than
its training examples. It is one of the cleanest failure modes in deep
learning: the model has, by every in-distribution measure, "learned the
task" — reversing a sequence is a simple, mechanical rule — yet the learned
solution secretly depends on the lengths it saw, and breaks the moment
length changes.

The prime suspects live in the layer math:

- **Positional information.** This model adds learned absolute positional
  embeddings. Positions beyond the training maximum exist in the table but
  never received a single gradient update. Length 11 may be failing before
  attention even starts.
- **Attention score geometry.** Raw dot products grow with vector magnitude,
  so attention can quietly encode "how big" instead of "what matches what."
  Two standard repairs target exactly this: penalize distant pairs (ALiBi),
  or throw magnitude away entirely (cosine attention).
- **The optimizer.** Maybe the model simply lands in a sharp, length-bound
  minimum. A different optimizer with decoupled weight decay and a longer
  warmup might find a flatter, more general solution.

**Why one variable at a time?** Because anything else is storytelling. If
you change three things and the score moves, you know nothing about which
change mattered. Each arm below changes exactly one piece of math against a
fresh baseline re-run in the same harness, with the same seed, the same
data splits, and the same evaluation. Negative results are reported as
measured — an arm that changes nothing is a finding, not an embarrassment.

## HOW — the three changes

### V1 — ALiBi-style distance penalty

```
scores_h(i, j) = (Q_h · K_h) / sqrt(d_k)  −  m_h · |i − j|,
m_h = 2^(−8h/H)  for head h of H
```

In plain words: before the softmax, each head subtracts a penalty that
grows linearly with the distance between two tokens, so attention is biased
toward nearby tokens by construction — no new parameters, nothing learned.

Run it: `python3 run_arm.py alibi` (attention_mode = "alibi")

### V2 — QK-normalization (cosine attention)

```
scores(i, j) = tau · cos(Q_i, K_j)
             = tau · (Q_i · K_j) / (||Q_i|| · ||K_j||)
```

In plain words: normalize every query and key to unit length so attention
depends only on the *angle* between vectors, never their size, and let one
learnable temperature tau per attention module (initialized to sqrt(d_k))
set the sharpness — the fixed 1/sqrt(d_k) scaling is gone.

Run it: `python3 run_arm.py qknorm` (attention_mode = "qknorm")

### V3 — the optimizer arm (AdamW + longer warmup)

```
AdamW:  p ← p − lr · ( adam_direction(p) + wd · p ),   wd = 0.1
Noam warmup: 200 → 400 steps
```

In plain words: keep the attention math exactly as the baseline, but decay
the weights directly (decoupled from the gradient, as in AdamW) and warm
the learning rate up over twice as many steps. The baseline uses plain
Adam, so this is a genuine change, not a rename.

Run it: `python3 run_arm.py adamw` (optimizer = "adamw", warmup = 400)

Baseline control: `python3 run_arm.py baseline`

## Reproduce in 3 commands

```bash
git clone <this-repo> && cd one-line-of-math
pip install numpy
python3 run_arm.py qknorm   # or: baseline | alibi | adamw  (~20 min per arm on 2 CPU cores)
```

Then `python3 collect_results.py && python3 gen_results.py` rebuilds
`results.json` and `RESULTS-LAYER-MATH.md` from the raw logs, and
`python3 gradcheck.py <base|alibi|qknorm>` re-runs the numerical gradient
check for a variant's backward pass.

## Results

Every number below is generated from `results.json`, which was merged from
the raw run logs in `logs/` — nothing is typed from memory. Exact-match is
greedy decoding on 500 held-out sequences per band; lengths 4–10 are the
trained range, 11–14 and 15–18 were never trained on. Before any accuracy
number is trusted, each modified backward pass must pass a numerical
gradient check — all of them did.

| Arm | Train time | Val token acc | Exact 4–10 | Exact 11–14 | Exact 15–18 | Gradient check |
|---|---|---|---|---|---|---|
| Baseline (fresh re-run) | 1361s | 0.9992 | 499/500 | **0/500** | **0/500** | PASS — worst rel err 4.91e-10 (1,633 elements) |
| V1 — ALiBi distance penalty | 1120s | 0.9996 | 495/500 | **0/500** | **0/500** | PASS — worst rel err 5.21e-10 (1,633 elements) |
| V2 — QK-norm (cosine attention) | 1276s | 1.0000 | 500/500 | **0/500** | **0/500** | PASS — worst rel err 5.43e-10 (1,636 elements) |
| V3 — AdamW + warmup 400 | 1143s | 1.0000 | 500/500 | **0/500** | **0/500** | PASS — worst rel err 4.91e-10 (baseline attention math, covered by the baseline check) |

**The honest headline: none of the three changes fixed length
generalization — at all.** All four arms score exactly zero beyond the
trained lengths. Two arms (QK-norm, AdamW) reached a *perfect*
in-distribution fit — validation accuracy 1.0000, 500/500 exact — and still
scored 0/500 at length 11. QK-norm's final training loss fell to 0.0002,
versus the baseline's 0.0110. So at this scale the wall is not an
optimization problem and not a fit-quality problem; the evidence points at
the one component none of these changes touches — the learned absolute
positional embeddings for positions the model never trained on. That is a
hypothesis for the next experiment, not a conclusion of this one.

**What this does not show:** anything about real-scale models, real
language data, other tasks, or other positional schemes (rotary, relative,
none). Each arm ran one seed, so in-distribution differences of a few
sequences in 500 (baseline 499, ALiBi 495) are within seed noise and are
not claimed as effects. Combinations of changes are untested by design.
Train times were measured on a shared 2-core VM; the baseline's first
epoch overlapped another process and its total is inflated by roughly
200s — the per-epoch times in the logs show this openly.

Full per-arm discussion: `RESULTS-LAYER-MATH.md`. Paper plan:
`PAPER-OUTLINE.md`.

## Citation

```bibtex
@misc{sirlapu2026onelineofmath,
  author       = {Tejaswini Sirlapu},
  title        = {One Line of Math: Three Single-Variable Changes vs. the Length-Generalization Wall},
  year         = {2026},
  howpublished = {GitHub repository: one-line-of-math},
  note         = {From-scratch NumPy Transformer; all results generated from logged runs in results.json}
}
```

## License

MIT — see `LICENSE`.
