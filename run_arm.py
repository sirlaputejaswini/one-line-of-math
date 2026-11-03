"""
Run ONE arm of the layer-math experiment.

Arms (everything else — seed, splits, d_model 128, 2+2 layers, d_ff 512,
batch 64, 10 epochs, Noam schedule — is IDENTICAL to the baseline
reimplementation-01 run, so any difference is the arm's doing):

  baseline : attention mode "base", Adam, warmup 200   (fresh re-run in
             this harness; the control every variant is compared to)
  alibi    : V1 — ALiBi-style per-head linear distance penalty subtracted
             from attention scores before softmax. No new parameters.
  qknorm   : V2 — QK-normalization (cosine attention): Q/K rows
             L2-normalized, learnable temperature tau (init sqrt(dk),
             one scalar per attention module = +6 params) replaces
             1/sqrt(dk).
  adamw    : V3 — optimizer arm: baseline attention math, Adam -> AdamW
             (decoupled weight decay 0.1, applied uniformly to all
             parameters) and Noam warmup extended 200 -> 400 steps.
             (The baseline uses plain Adam, so AdamW is a genuine
             change; no need for the SGD fallback.)

Evaluation protocol (identical for every arm, mirrors train.py +
eval_lengthgen.py of reimplementation-01 exactly):
  * final validation token accuracy (teacher-forced, 2,000 sequences)
  * greedy exact-match on the first 500 held-out test sequences (4-10)
  * greedy exact-match on 500 fresh sequences at lengths 11-14
    (RNG seeded SEED+11) and 15-18 (seeded SEED+15) — never trained on

Outputs: logs/<arm>.log, weights_<arm>.npz, results_<arm>.json.

Run:  python3 run_arm.py baseline|alibi|qknorm|adamw
"""

import json
import random
import sys
import time

import numpy as np

import data as D
from transformer import Transformer

CONFIG = dict(
    seed=20261028,
    d_model=128,
    num_heads=4,
    num_layers=2,
    d_ff=512,
    batch_size=64,
    epochs=10,
    train_size=20_000,
    val_size=2_000,
    test_size=2_000,
    eval_decode_n=500,
    min_len=4,
    max_len=10,
)

ARMS = {
    "baseline": dict(attention_mode="base", optimizer="adam", warmup=200, weight_decay=None),
    "alibi": dict(attention_mode="alibi", optimizer="adam", warmup=200, weight_decay=None),
    "qknorm": dict(attention_mode="qknorm", optimizer="adam", warmup=200, weight_decay=None),
    "adamw": dict(attention_mode="base", optimizer="adamw", warmup=400, weight_decay=0.1),
}

ARM_BLURB = {
    "baseline": "Control: scaled dot-product attention, Adam, Noam warmup 200.",
    "alibi": "V1 ALiBi: scores -= m_h * |i-j| per head before softmax; no new parameters.",
    "qknorm": "V2 QK-norm: scores = tau * cos(Q,K), tau learnable per attention module (init sqrt(dk)).",
    "adamw": "V3 AdamW: decoupled weight decay 0.1 (uniform), Noam warmup 200 -> 400.",
}


class Adam:
    """Adam as in reimplementation-01 (beta1=0.9, beta2=0.98, eps=1e-9)."""

    def __init__(self, params_and_grads_fn, beta1=0.9, beta2=0.98, eps=1e-9):
        self.fn = params_and_grads_fn
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.t = 0
        self.m = None
        self.v = None

    def _adam_direction(self, pairs):
        if self.m is None:
            self.m = [np.zeros_like(p) for p, _ in pairs]
            self.v = [np.zeros_like(p) for p, _ in pairs]
        self.t += 1
        bc1 = 1 - self.beta1 ** self.t
        bc2 = 1 - self.beta2 ** self.t
        dirs = []
        for i, (p, g) in enumerate(pairs):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g * g
            mhat = self.m[i] / bc1
            vhat = self.v[i] / bc2
            dirs.append(mhat / (np.sqrt(vhat) + self.eps))
        return dirs

    def step(self, lr):
        pairs = self.fn()
        for (p, _), d in zip(pairs, self._adam_direction(pairs)):
            p -= lr * d


class AdamW(Adam):
    """AdamW (Loshchilov & Hutter, 2019): the Adam step PLUS decoupled
    weight decay p -= lr * wd * p, applied to the pre-update weights and
    NOT folded into the gradient (that folding is what made L2-in-Adam
    not equivalent to weight decay). Uniform over all parameters."""

    def __init__(self, params_and_grads_fn, weight_decay=0.1, **kw):
        super().__init__(params_and_grads_fn, **kw)
        self.weight_decay = weight_decay

    def step(self, lr):
        pairs = self.fn()
        for (p, _), d in zip(pairs, self._adam_direction(pairs)):
            p -= lr * (d + self.weight_decay * p)


def noam_lr(step, d_model, warmup):
    return d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)


def evaluate(model, dataset):
    total_loss, total_tokens, total_correct = 0.0, 0, 0
    for start in range(0, len(dataset), CONFIG["batch_size"]):
        batch = dataset[start : start + CONFIG["batch_size"]]
        src, tgt_in, tgt_out = D.collate(batch)
        logits = model.predict_logits(src, tgt_in)
        shifted = logits - logits.max(axis=-1, keepdims=True)
        logp = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        mask = tgt_out != D.PAD
        picked = np.take_along_axis(logp, tgt_out[:, :, None], axis=-1)[:, :, 0]
        total_loss += float(-(picked * mask).sum())
        total_tokens += int(mask.sum())
        total_correct += int(((logits.argmax(-1) == tgt_out) & mask).sum())
    return total_loss / total_tokens, total_correct / total_tokens


def train_one_epoch(model, dataset, opt, step_num, warmup):
    order = list(range(len(dataset)))
    random.Random(CONFIG["seed"] * 1000 + step_num).shuffle(order)
    total_loss, total_tokens, total_correct = 0.0, 0, 0
    for start in range(0, len(order), CONFIG["batch_size"]):
        batch = [dataset[i] for i in order[start : start + CONFIG["batch_size"]]]
        src, tgt_in, tgt_out = D.collate(batch)
        model.zero_grads()
        loss = model.loss_and_backward(src, tgt_in, tgt_out, pad_id=D.PAD)
        logits = model.predict_logits(src, tgt_in)  # pre-update weights
        mask = tgt_out != D.PAD
        total_loss += loss * int(mask.sum())
        total_tokens += int(mask.sum())
        total_correct += int(((logits.argmax(-1) == tgt_out) & mask).sum())
        step_num += 1
        opt.step(noam_lr(step_num, CONFIG["d_model"], warmup))
    return total_loss / total_tokens, total_correct / total_tokens, step_num


def exact_match(model, dataset, n):
    exact = 0
    for i in range(n):
        src, tgt = dataset[i]
        pred = model.generate(src, D.BOS, D.EOS, max_len=len(tgt) + 6)
        exact += list(pred) == list(tgt)
    return exact


def main():
    arm = sys.argv[1]
    spec = ARMS[arm]
    lines = []

    def log(msg=""):
        print(msg, flush=True)
        lines.append(msg)

    np.random.seed(CONFIG["seed"])
    random.seed(CONFIG["seed"])

    log("=" * 72)
    log(f"Layer-math experiment — arm: {arm}")
    log(f"Arm definition: {ARM_BLURB[arm]}")
    log(f"Config: {CONFIG} | arm spec: {spec}")
    log(f"NumPy version: {np.__version__}")
    log("=" * 72)

    rng = random.Random(CONFIG["seed"])
    train_data = D.make_dataset(rng, CONFIG["train_size"], CONFIG["min_len"], CONFIG["max_len"])
    val_data = D.make_dataset(rng, CONFIG["val_size"], CONFIG["min_len"], CONFIG["max_len"])
    test_data = D.make_dataset(rng, CONFIG["test_size"], CONFIG["min_len"], CONFIG["max_len"])

    model = Transformer(
        src_vocab=D.VOCAB_SIZE,
        tgt_vocab=D.VOCAB_SIZE,
        d_model=CONFIG["d_model"],
        num_heads=CONFIG["num_heads"],
        num_layers=CONFIG["num_layers"],
        d_ff=CONFIG["d_ff"],
        pad_id=D.PAD,
        seed=CONFIG["seed"],
        attention_mode=spec["attention_mode"],
    )
    n_params = model.num_params()
    log(f"Parameter count: {n_params:,}")

    if spec["optimizer"] == "adamw":
        opt = AdamW(model.params_and_grads, weight_decay=spec["weight_decay"])
    else:
        opt = Adam(model.params_and_grads)

    step_num = 0
    epoch_rows = []
    t0 = time.time()
    for epoch in range(1, CONFIG["epochs"] + 1):
        ep_t0 = time.time()
        tr_loss, tr_acc, step_num = train_one_epoch(model, train_data, opt, step_num, spec["warmup"])
        va_loss, va_acc = evaluate(model, val_data)
        epoch_rows.append(
            dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc,
                 val_loss=va_loss, val_acc=va_acc)
        )
        log(
            f"epoch {epoch:2d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f} | {time.time() - ep_t0:.1f}s"
        )
    train_seconds = time.time() - t0
    log(f"Training wall time: {train_seconds:.1f}s for {CONFIG['epochs']} epochs ({step_num} optimizer steps)")

    final_val_loss, final_val_acc = epoch_rows[-1]["val_loss"], epoch_rows[-1]["val_acc"]

    n_eval = CONFIG["eval_decode_n"]
    em_in = exact_match(model, test_data, n_eval)
    log(f"Greedy exact-match, held-out lengths 4-10: {em_in}/{n_eval} = {em_in / n_eval:.4f}")

    em = {}
    for lo, hi in ((11, 14), (15, 18)):
        rng2 = random.Random(CONFIG["seed"] + lo)
        ds = D.make_dataset(rng2, 500, lo, hi)
        n = exact_match(model, ds, 500)
        em[f"{lo}_{hi}"] = n
        log(f"Greedy exact-match, lengths {lo}-{hi} (never trained on): {n}/500 = {n / 500:.4f}")

    np.savez(
        f"weights_{arm}.npz",
        **{f"p{i}": p for i, (p, _) in enumerate(model.params_and_grads())},
    )
    log(f"Saved model weights to weights_{arm}.npz")

    result = dict(
        arm=arm,
        description=ARM_BLURB[arm],
        attention_mode=spec["attention_mode"],
        optimizer=spec["optimizer"],
        warmup_steps=spec["warmup"],
        weight_decay=spec["weight_decay"],
        params=n_params,
        train_seconds=round(train_seconds, 1),
        optimizer_steps=step_num,
        final_val_loss=round(final_val_loss, 4),
        final_val_acc=round(final_val_acc, 4),
        epochs=epoch_rows,
        exact_match_4_10=dict(correct=em_in, total=n_eval),
        exact_match_11_14=dict(correct=em["11_14"], total=500),
        exact_match_15_18=dict(correct=em["15_18"], total=500),
    )
    with open(f"results_{arm}.json", "w") as f:
        json.dump(result, f, indent=2)
    log(f"Wrote results_{arm}.json")

    with open(f"logs/{arm}.log", "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
