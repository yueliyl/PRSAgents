"""Student-facing harness: load one replicate, evaluate on validation, diagnose.

Data layout (see data/README.md):

    data/<scenario>/      shared_effects, ancestry_dependent,
                          locus_varying, nonlinear_link
    data/common/          LD matrices and the transfer-assessment cohorts

Each scenario holds TEN independent replicates stacked along a leading axis.
`Dataset(scenario, replicate)` gives you one of them.  Replicates are not
cross-validation folds; they are independent samples, and comparisons between
methods should be paired within replicate and averaged across the ten.

The evaluation budget is enforced here.  `evaluate` scores only the validation
split; the test split has no labels on your side.  Submit predictions.
"""
import numpy as np
import pandas as pd

from models import incremental_r2
from prsdata import load, n_replicates, POPS

DEFAULT_SCENARIO = "data/ancestry_dependent"
SCENARIOS = ["shared_effects", "ancestry_dependent", "locus_varying", "nonlinear_link"]


class Dataset:
    """One replicate of one scenario: 4,000 individuals, 1,000 variants.

        d = Dataset("data/locus_varying", replicate=3)
        S, Z, C, y, idx = d.part("train")
    """

    def __init__(self, scenario, replicate=0):
        r = load(scenario, replicate)
        self.path = scenario
        self.scenario = r.scenario
        self.replicate = replicate
        self.dosage = r.dosage
        self.S = r.S                      # (N, K) the ten source PRS
        self.Z = r.Z                      # (N, K*W) window contrasts
        self.C = r.C                      # (N, 12) age, sex, PC1..PC10
        self.y = r.y                      # NaN on test
        self.panel = r.panel
        self.cov_names = r.cov_names
        self.block_of_snp = r.block_of_snp
        self.source_weights = r.source_weights
        self.n_windows = r.n_windows
        self.idx_train, self.idx_val, self.idx_test = (
            r.idx_train, r.idx_val, r.idx_test)
        self.info = r.meta
        self._r = r

    # ---- Stage 1 inputs -----------------------------------------------------
    def sumstats(self, pop):
        """GWAS summary statistics from one discovery cohort."""
        return self._r.sumstats(pop)

    def ld(self, pop):
        """That population's LD matrix (block diagonal up to sampling noise)."""
        return self._r.ld(pop)

    def eval_cohort(self, pop):
        """A held-out single-ancestry cohort, for transfer assessment."""
        return self._r.eval_cohort(pop)

    @property
    def eval_populations(self):
        return self._r.eval_populations

    # ---- convenience slices -------------------------------------------------
    def part(self, which):
        i = {"train": self.idx_train, "val": self.idx_val,
             "test": self.idx_test}[which]
        return self.S[i], self.Z[i], self.C[i], self.y[i], i

    def __repr__(self):
        return (f"<Dataset {self.scenario} replicate={self.replicate} "
                f"train/val/test={len(self.idx_train)}/{len(self.idx_val)}/"
                f"{len(self.idx_test)}>")


def all_replicates(scenario):
    """Iterate a scenario's ten replicates, for paired comparisons."""
    for r in range(n_replicates(scenario)):
        yield Dataset(scenario, r)


class Harness:
    """Wraps a Dataset and enforces the evaluation budget."""

    def __init__(self, dataset, budget=12):
        self.d = dataset
        self.budget = budget
        self.used = 0
        self.log = []

    def evaluate(self, predict, note=""):
        """Incremental R2 on the VALIDATION split.  Costs one unit of budget."""
        if self.used >= self.budget:
            raise RuntimeError(f"evaluation budget of {self.budget} exhausted")
        self.used += 1
        S, Z, C, y, _ = self.d.part("val")
        pred = predict(S, Z, C)
        r2 = incremental_r2(y, np.asarray(pred, float), C)
        self.log.append({"call": self.used, "note": note, "val_incremental_r2": r2})
        return r2

    def diagnose(self, predict, on="val"):
        """Free. Residual structure, per-stratum accuracy, calibration."""
        S, Z, C, y, _ = self.d.part(on)
        pred = np.asarray(predict(S, Z, C), float)
        resid = y - pred
        out = {}
        for j, name in enumerate(self.d.cov_names):
            out[f"resid_corr_{name}"] = float(np.corrcoef(resid, C[:, j])[0, 1])
        # Terciles of PC1, a proxy for the ancestry axis.
        pc1 = C[:, self.d.cov_names.index("PC1")]
        q = np.quantile(pc1, [1 / 3, 2 / 3])
        for t, mask in enumerate([pc1 <= q[0], (pc1 > q[0]) & (pc1 <= q[1]), pc1 > q[1]]):
            out[f"r2_pc1_tercile{t+1}"] = float(incremental_r2(y[mask], pred[mask], C[mask]))
        sl = np.polyfit(pred, y, 1)[0] if pred.std() > 1e-9 else float("nan")
        out["calibration_slope"] = float(sl)
        K = self.d.S.shape[1]
        W = self.d.Z.shape[1] // K
        idx = self.d.part(on)[4]
        wc = []
        for w in range(W):
            blk = self.d.Z[idx][:, w * K:(w + 1) * K]
            wc.append(float(np.abs([np.corrcoef(resid, blk[:, k])[0, 1]
                                    for k in range(K)]).max()))
        out["max_resid_corr_per_window"] = wc
        return out

    def submit(self, predict, outfile):
        S, Z, C, _, idx = self.d.part("test")
        pred = np.asarray(predict(S, Z, C), float)
        pd.DataFrame({"iid": idx, "prediction": pred}).to_csv(outfile, index=False)
        return outfile


def demo(scenario=DEFAULT_SCENARIO, replicate=0):
    from models import fit_linear_stack
    d = Dataset(scenario, replicate)
    h = Harness(d)
    print(d)
    S, Z, C, y, _ = d.part("train")
    f = fit_linear_stack(S, Z, C, y)
    print("linear stacking, validation incremental R2 =", round(h.evaluate(f, "stack"), 4))
    diag = h.diagnose(f)
    print("residual corr with PC1 =", round(diag["resid_corr_PC1"], 4))
    print("R2 by PC1 tercile      =",
          [round(diag[f"r2_pc1_tercile{i}"], 4) for i in (1, 2, 3)])
    print("target test labels and oracle metadata are withheld")


if __name__ == "__main__":
    import sys
    a = sys.argv[1:]
    demo(a[0] if a else DEFAULT_SCENARIO, int(a[1]) if len(a) > 1 else 0)
