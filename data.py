"""
Local, offline data for the reimplementation: the sequence-reversal task.

Why reversal? It is the smallest task that forces the model to actually
use attention: the output at position i depends on the input at position
n-1-i, so a model that cannot route information across positions cannot
solve it. It needs no dataset download — sequences are sampled from a
seeded RNG, so train / validation / test are disjoint draws from the
same generator, and anyone can reproduce the exact splits.

Vocabulary (shared by source and target):
    0 = <pad>   1 = <bos>   2 = <eos>   3..12 = symbols 'a'..'j'

Source:  a random symbol sequence, e.g.  c a j d
Target:  the same sequence reversed, wrapped:  <bos> d j a c <eos>
"""

import random

import numpy as np

PAD, BOS, EOS = 0, 1, 2
SYMBOLS = list("abcdefghij")
VOCAB_SIZE = 3 + len(SYMBOLS)


def token_of(ch):
    return 3 + SYMBOLS.index(ch)


def render(ids):
    """Token ids -> readable string (special tokens shown as words)."""
    out = []
    for i in ids:
        i = int(i)
        if i == PAD:
            out.append("<pad>")
        elif i == BOS:
            out.append("<bos>")
        elif i == EOS:
            out.append("<eos>")
        else:
            out.append(SYMBOLS[i - 3])
    return " ".join(out)


def make_example(rng, min_len=4, max_len=10):
    n = rng.randint(min_len, max_len)
    seq = [rng.choice(SYMBOLS) for _ in range(n)]
    src = [token_of(c) for c in seq]
    tgt = [BOS] + [token_of(c) for c in reversed(seq)] + [EOS]
    return src, tgt


def make_dataset(rng, size, min_len=4, max_len=10):
    return [make_example(rng, min_len, max_len) for _ in range(size)]


def collate(batch):
    """Pad a list of (src, tgt) pairs into NumPy arrays.

    Returns src (B, S), tgt_in (B, T-1), tgt_out (B, T-1) where tgt_in is
    the target shifted right (teacher-forcing input, starts with <bos>)
    and tgt_out is the target shifted left (what must be predicted).
    """
    max_src = max(len(s) for s, _ in batch)
    max_tgt = max(len(t) for _, t in batch)
    src = np.full((len(batch), max_src), PAD, dtype=np.int64)
    tgt = np.full((len(batch), max_tgt), PAD, dtype=np.int64)
    for i, (s, t) in enumerate(batch):
        src[i, : len(s)] = s
        tgt[i, : len(t)] = t
    return src, tgt[:, :-1], tgt[:, 1:]
