"""Stage 1 reference export: five source weights, transfer table, target scores.

    python3 fit_stage1.py data/shared_effects --replicate 0 --out outputs/stage1
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from harness import Dataset
from stages.stage1_fit import fit_sources, transfer_table, SOURCES


def main(scenario, replicate, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    d = Dataset(scenario, replicate)
    weights = fit_sources(d)
    transfer_table(d, weights).to_csv(out / "transfer_matrix.csv")

    # Score the target cohort. Standardisation is learned on TRAINING
    # individuals only and reused unchanged on validation and test.
    x = d.dosage.astype(float)
    mu = x[d.idx_train].mean(0); sd = x[d.idx_train].std(0); sd[sd < 1e-8] = 1.0
    standardized = (x - mu) / sd
    raw = np.column_stack([standardized @ weights[pop] for pop in SOURCES])
    smu = raw[d.idx_train].mean(0); ssd = raw[d.idx_train].std(0); ssd[ssd < 1e-8] = 1.0
    scores = (raw - smu) / ssd
    frame = pd.DataFrame(scores, columns=[f"{pop}_viprs" for pop in SOURCES])
    frame.insert(0, "iid", np.arange(len(scores)))
    frame.to_csv(out / "target_scores.csv", index=False)
    np.savez_compressed(out / "source_weights.npz", populations=np.array(SOURCES),
                        weights=np.vstack([weights[pop] for pop in SOURCES]),
                        genotype_mean=mu, genotype_sd=sd,
                        score_mean=smu, score_sd=ssd)
    print(f"Stage 1 reference exports for {d.scenario} replicate {replicate}:",
          out.resolve())
    print(pd.read_csv(out / "transfer_matrix.csv", index_col=0).round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scenario", nargs="?", default="data/shared_effects")
    ap.add_argument("--replicate", type=int, default=0)
    ap.add_argument("--out", default="outputs/stage1_demo")
    a = ap.parse_args(); main(a.scenario, a.replicate, a.out)
