"""Merge results_<arm>.json (one per arm, written by run_arm.py) and the
gradient-check logs into a single results.json — the machine-readable
record RESULTS-LAYER-MATH.md is generated from.

Run:  python3 collect_results.py
"""

import json
import re

ARMS = ["baseline", "alibi", "qknorm", "adamw"]
GRADCHECK_MODE = {"baseline": "base", "alibi": "alibi", "qknorm": "qknorm", "adamw": "base"}


def gradcheck_for(arm):
    mode = GRADCHECK_MODE[arm]
    try:
        text = open(f"logs/gradcheck_{mode}.log").read()
    except FileNotFoundError:
        return None
    m = re.search(r"(\d+) elements checked, worst relative error ([\d.e-]+)  \[(PASS|FAIL)\]", text)
    if not m:
        return None
    return dict(
        mode=mode,
        elements_checked=int(m.group(1)),
        worst_rel_error=float(m.group(2)),
        status=m.group(3),
        note=("optimizer arm: attention math identical to baseline arm, "
              "covered by the baseline (base) gradient check" if arm == "adamw" else ""),
    )


def main():
    arms = {}
    for arm in ARMS:
        with open(f"results_{arm}.json") as f:
            arms[arm] = json.load(f)
        arms[arm]["gradient_check"] = gradcheck_for(arm)
    out = dict(
        experiment="layer-math: three single-variable math changes vs a fresh baseline re-run",
        task="token-level sequence reversal, symbols a-j, train lengths 4-10, 20k/2k/2k seeded splits",
        harness="reimplementation-02-layer-math (fork of reimplementation-01-transformer, pure NumPy)",
        arms=arms,
    )
    with open("results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results.json with arms:", ", ".join(arms))


if __name__ == "__main__":
    main()
