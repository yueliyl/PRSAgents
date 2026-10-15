"""Compare a candidate and baseline on the SAME independent replicate rows.

Input CSV columns: scenario, replicate, method, score (the older names
dataset, rep are accepted too). Scores may be validation or
instructor-returned test scores; never pool them or mix different metrics.

Pairing matters. The ten replicates per scenario differ a lot in overall
signal, so an unpaired difference of means is dominated by that variation
rather than by the methods. Pairing within replicate removes it. Comparing
methods that were evaluated on DIFFERENT subsets of replicates can reverse the
apparent sign of an effect, which is why this script drops to common support
and reports how many pairs it used.
"""
import argparse
import numpy as np
import pandas as pd

ALIASES = {"dataset": "scenario", "rep": "replicate", "regime": "scenario"}


def _normalise(frame):
    frame = frame.rename(columns={k: v for k, v in ALIASES.items()
                                  if k in frame.columns and v not in frame.columns})
    missing = {"scenario", "replicate", "method", "score"} - set(frame.columns)
    if missing:
        raise ValueError(f"missing column(s): {sorted(missing)}")
    return frame


def summarize(frame, baseline, candidate):
    frame = _normalise(frame)
    keys = ["scenario", "replicate", "method"]
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate method/replicate rows")
    idx = ["scenario", "replicate"]
    a = frame[frame.method == baseline].set_index(idx).score.rename("baseline")
    b = frame[frame.method == candidate].set_index(idx).score.rename("candidate")
    paired = pd.concat([a, b], axis=1).dropna()
    if paired.empty:
        raise ValueError("No common replicate support")
    rows = []
    for name, g in [("pooled", paired)] + list(paired.groupby(level="scenario")):
        delta = g.candidate - g.baseline
        n = len(delta)
        se = delta.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        t = delta.mean() / se if n > 1 and se > 0 else np.nan
        rows.append({"scenario": name, "n_pairs": n, "mean_gain": delta.mean(),
                     "se": se, "t": t})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv")
    ap.add_argument("--baseline", default="stage2_moe")
    ap.add_argument("--candidate", default="qwen_agent")
    a = ap.parse_args()
    out = summarize(pd.read_csv(a.csv), a.baseline, a.candidate)
    print(out.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
