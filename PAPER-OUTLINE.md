# PAPER OUTLINE — the layer-math experiment

All numbers below come from `results.json` in this folder (generated from the
run logs by `collect_results.py`; summarized in `RESULTS-LAYER-MATH.md`).

## Title candidates

1. **One Line of Math Is Not Enough: Three Single-Variable Changes Fail to Give a From-Scratch Transformer Length Generalization**
2. Perfect Fit, Zero Reach: Why Cleaner Attention Math Does Not Teach a Small Transformer to Reverse Longer Sequences
3. The Length Wall: A Controlled Negative Result on ALiBi, QK-Norm, and AdamW in a From-Scratch NumPy Transformer

## The claim the results actually support (do not exceed this)

At teaching scale (d_model 128, 4 heads, 2+2 layers, 930,957 parameters,
synthetic sequence reversal, one seed, 10 epochs), **none** of the three
single-variable math changes — an ALiBi-style distance penalty (V1),
QK-normalization with a learnable temperature (V2), or AdamW with decoupled
weight decay and doubled warmup (V3) — produced **any** length
generalization. Every arm, including a fresh baseline re-run in the same
harness, scored **0/500** greedy exact-match at lengths 11–14 and **0/500**
at lengths 15–18, while in-distribution exact-match (lengths 4–10) stayed at
495–500/500 and final validation token accuracy at 0.9992–1.0000.

Two arms (QK-norm and AdamW) reached a *perfect* in-distribution fit —
validation token accuracy 1.0000 and exact-match 500/500 — and still scored
zero out of distribution. QK-norm's final training loss collapsed to 0.0002
(baseline: 0.0110). So at this scale the length wall is demonstrably **not**
an optimization or fit-quality problem: a model can fit the trained lengths
perfectly and have literally nothing at length 11.

The suspect the evidence points to next is the one component none of the
three changes touches: the learned absolute positional embeddings.
Positions beyond the trained maximum receive no gradient during training,
so a length-11 input feeds the network effectively untrained positional
vectors. All three changes operate on score geometry or optimization; none
gives the model trained representations for unseen positions.

## What is NOT claimed

- Nothing about real-scale models, real language data, other tasks, or other
  positional schemes (rotary, relative, NoPE). The result is local to this
  harness, deliberately.
- No ranking of the three variants on out-of-distribution performance — all
  are tied at zero. In-distribution differences of a few sequences in 500
  (baseline 499, ALiBi 495) are within single-seed noise and are not claimed
  as effects. QK-norm's loss collapse is reported as an observation.
- No claim that combinations fail: each arm changed exactly one variable, so
  interactions (e.g. ALiBi + QK-norm) are untested by design.

## Section plan

1. **Introduction** — the cliff as the phenomenon: 500/500 in-distribution,
   0/500 one length beyond. Why a clean, total failure is worth a controlled
   study instead of another architecture tweak.
2. **Setup and trust basis** — from-scratch NumPy Transformer, hand-written
   forward/backward; numerical gradient checks on every modified backward
   pass (worst relative error 4.906e-10 baseline, 5.207e-10 ALiBi,
   5.432e-10 QK-norm; AdamW arm shares the baseline attention math and its
   check); reversal task, seeded 20k/2k/2k splits; identical protocol per
   arm; the fresh-baseline rule (variants judged against the same-harness
   re-run — 499/500 — never against the recorded original 500/500; the
   one-sequence residue is floating-point nondeterminism, stated openly).
3. **The three changes** — one equation and one plain sentence each:
   distance penalty before softmax; cosine attention with learned
   temperature; AdamW (decoupled decay 0.1) with warmup 200 → 400.
4. **Results** — the four-arm table (train time, val accuracy, exact-match
   in the three bands, gradient-check status) and one paragraph per arm,
   including ALiBi's slight in-distribution cost (495/500) and QK-norm's
   perfect fit.
5. **Analysis** — why zero movement in three mechanistically different arms
   localizes the failure: score-geometry changes (V1, V2) and optimization
   changes (V3) all leave the positional pathway untouched; the positional
   hypothesis and how it would be falsified.
6. **Limitations** — one seed per arm; tiny scale; synthetic task; greedy
   decoding only; train times measured on a shared 2-core VM (baseline
   epoch 1 inflated ~200s by a concurrent process — disclosed in the logs).
7. **Next experiments** — rotary or relative positions, or removing
   positional embeddings from the encoder (NoPE); position interpolation;
   training-length curricula; multi-seed replication before any stronger
   wording.
8. **Reproducibility** — public repo (`one-line-of-math`), three commands
   to re-run an arm, `results.json` as the single source of every number
   in the paper.

## Framing decision

Because no variant fixed length generalization, this is written as a
**negative-result + analysis** paper, not a method paper. The contribution
is the controlled comparison itself: three plausible one-line repairs,
gradient-verified, run under an identical protocol, all failing completely
— plus the perfect-fit observation that rules out "the model just didn't
learn the task well enough" as an explanation at this scale.
