"""Student-facing harness: load a dataset, evaluate on validation, diagnose.

The evaluation budget is enforced here.  `evaluate` scores only the validation
split; the test split has no labels on your side.  Submit predictions.
"""
import json, os
import numpy as np
import pandas as pd

from models import incremental_r2


class Dataset:
    def __init__(self, path):
        self.path = path
        g = np.load(os.path.join(path, "genotypes.npz"))
        self.dosage = g["dosage"]
        w = np.load(os.path.join(path, "source_weights.npz"))
        self.source_weights = w["weights"]
        self.panel = [str(x) for x in w["panel"]]
        self.block_of_snp = w["block_of_snp"]
        self.Z = np.load(os.path.join(path, "window_scores.npz"))["Z"].astype(float)
        sc = pd.read_csv(os.path.join(path, "scores.csv"))
        self.S = sc[self.panel].to_numpy(float)
        cv = pd.read_csv(os.path.join(path, "covariates.csv"))
        self.cov_names = [c for c in cv.columns if c != "iid"]
        self.C = cv[self.cov_names].to_numpy(float)
        lab = pd.read_csv(os.path.join(path, "labels_train.csv"))
        self.info = json.load(open(os.path.join(path, "dataset_info.json")))
        self.idx_train = lab.loc[lab.split == 0, "iid"].to_numpy()
        self.idx_val = lab.loc[lab.split == 1, "iid"].to_numpy()
        self.idx_test = np.array(self.info["test_iids"])
        self.y = np.full(len(self.C), np.nan)
        self.y[lab.iid.to_numpy()] = lab.phenotype.to_numpy()
        self.n_windows = self.info["n_sources"] and int(self.Z.shape[1] / self.S.shape[1])

    # ---- convenience slices -------------------------------------------------
    def part(self, which):
        i = {"train": self.idx_train, "val": self.idx_val, "test": self.idx_test}[which]
        return self.S[i], self.Z[i], self.C[i], self.y[i], i


class Harness:
    """Wraps a Dataset and enforces the evaluation budget."""

    def __init__(self, dataset, budget=40):
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
        wc = []
        for w in range(W):
            blk = self.d.Z[self.d.part(on)[4]][:, w * K:(w + 1) * K]
            wc.append(float(np.abs([np.corrcoef(resid, blk[:, k])[0, 1]
                                    for k in range(K)]).max()))
        out["max_resid_corr_per_window"] = wc
        return out

    def submit(self, predict, outfile):
        S, Z, C, _, idx = self.d.part("test")
        pred = np.asarray(predict(S, Z, C), float)
        pd.DataFrame({"iid": idx, "prediction": pred}).to_csv(outfile, index=False)
        return outfile


def demo(path):
    from models import fit_linear_stack
    d = Dataset(path)
    h = Harness(d)
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
    demo(sys.argv[1] if len(sys.argv) > 1 else "data/datasets/dataset_01/rep00")
