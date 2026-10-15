"""Stage 1 reference: fit a PRS per ancestry from summary statistics, then
measure within- versus cross-ancestry transfer.

Students do this themselves with their Assignment 1 VIPRS, summary-statistic
LASSO, or the published viprs package. This file exists so the instructor can
check the data supports the exercise, and as a fallback.

    python3 stages/stage1_fit.py data/shared_effects --replicate 0
"""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulate.prs import viprs
from models import incremental_r2
from harness import Dataset

SOURCES = ["EUR", "AFR", "EAS", "SAS", "AMR"]


def fit_sources(d, h2=0.6):
    """One PRS per ancestry, from that ancestry's summary statistics and LD.

    The ten LD blocks are unlinked by construction, so the variational sweep
    runs within each block against that block's 100 x 100 submatrix. Entries
    between blocks in the supplied matrices are sampling noise, not linkage.
    """
    W = {}
    for pop in SOURCES:
        s = d.sumstats(pop)
        W[pop] = viprs(s["beta"], d.ld(pop), s["n"],
                       block_of_snp=s["block_of_snp"], h2=h2)
    return W


def transfer_table(d, W):
    """Rows: the PRS's training ancestry. Columns: the evaluation cohort."""
    evals = d.eval_populations
    out = pd.DataFrame(index=SOURCES, columns=evals, dtype=float)
    for ep in evals:
        ev = d.eval_cohort(ep)
        X = ev["dosage"]
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        for sp in SOURCES:
            out.loc[sp, ep] = incremental_r2(ev["y"], X @ W[sp], ev["covariates"])
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scenario", nargs="?", default="data/shared_effects")
    ap.add_argument("--replicate", type=int, default=0)
    a = ap.parse_args()
    d = Dataset(a.scenario, a.replicate)
    W = fit_sources(d)
    tab = transfer_table(d, W)
    pd.set_option("display.width", 160)
    print(f"Stage 1, {d.scenario} replicate {d.replicate}")
    print("Incremental R2 of each source PRS (rows) in each cohort (columns)\n")
    print(tab.round(4).to_string())
    print("\n  diagonal (ancestry-matched):",
          {p: round(float(tab.loc[p, p]), 4) for p in SOURCES if p in tab.columns})
