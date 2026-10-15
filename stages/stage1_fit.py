"""Stage 1 reference: fit a PRS per ancestry from summary statistics, then
measure within- versus cross-ancestry transfer.

Students do this themselves with their Assignment 1 VIPRS, summary-statistic
LASSO, or the published viprs package. This file exists so the instructor can
check the data supports the exercise, and as a fallback.
"""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulate.prs import viprs
from models import incremental_r2

SOURCES = ["EUR", "AFR", "EAS", "SAS", "AMR"]


def load_stage1(dataset, root="data"):
    ss = np.load(os.path.join(dataset, "stage1", "sumstats.npz"))
    ev_g = np.load(os.path.join(root, "stage1", "eval_genotypes.npz"))
    ev_y = np.load(os.path.join(dataset, "stage1", "eval_phenotypes.npz"))
    ld = {p: np.load(os.path.join(root, "stage1", "ld", f"ld_{p}.npz"))["R"]
          for p in SOURCES}
    return ss, ld, ev_g, ev_y


def fit_sources(ss, ld, block_of_snp, h2=0.6):
    """One PRS per ancestry, from that ancestry's summary statistics and LD."""
    W = {}
    for pop in SOURCES:
        W[pop] = viprs(ss[f"{pop}_beta"], ld[pop].astype(float),
                       int(ss[f"{pop}_n"][0]), block_of_snp=block_of_snp, h2=h2)
    return W


def transfer_table(W, ev_g, ev_y):
    """Rows: the PRS's training ancestry. Columns: the evaluation cohort."""
    evals = [str(p) for p in ev_g["populations"]]
    out = pd.DataFrame(index=SOURCES, columns=evals, dtype=float)
    for ep in evals:
        X = ev_g[f"dosage_{ep}"].astype(float)
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        C = np.column_stack([ev_g[f"age_{ep}"], ev_g[f"sex_{ep}"]])
        y = ev_y[f"y_{ep}"]
        for sp in SOURCES:
            out.loc[sp, ep] = incremental_r2(y, X @ W[sp], C)
    return out


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "data/datasets/dataset_01/rep00"
    ss, ld, ev_g, ev_y = load_stage1(ds)
    W = fit_sources(ss, ld, ss["block_of_snp"])
    tab = transfer_table(W, ev_g, ev_y)
    pd.set_option("display.width", 160)
    print(f"Stage 1, {ds}")
    print("Incremental R2 of each source PRS (rows) in each cohort (columns)\n")
    print(tab.round(4).to_string())
    print("\n  diagonal (ancestry-matched):",
          {p: round(float(tab.loc[p, p]), 4) for p in SOURCES if p in tab.columns})
