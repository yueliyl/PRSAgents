"""Loader for the PRSAgents release layout.

    from prsdata import load
    d = load("data/release/ancestry_dependent", replicate=0)
    S, Z, C, y, idx = d.part("train")

One scenario folder holds ten independent replicates stacked along a leading
axis. `load` gives you one replicate as a flat object with the same attribute
names the per-replicate layout used, so analysis code only changes where the
data is opened.
"""
import json, os
import numpy as np

POPS = ["EUR", "AFR", "EAS", "SAS", "AMR"]
_CACHE = {}


class Replicate:
    """One simulated cohort: 4,000 individuals under a single generating rule."""

    def __init__(self, root, r, bundle):
        self.root, self.replicate = root, r
        self.meta = bundle["meta"]
        self.scenario = self.meta["scenario"]
        self.panel = bundle["panel"]
        self.block_of_snp = bundle["block_of_snp"]
        self.n_windows = bundle["n_windows"]
        self.cov_names = bundle["cov_names"]

        self.dosage = bundle["dosage"][r]          # (N, M) int8
        self.S = bundle["S"][r].astype(float)      # (N, K) standardised scores
        self.Z = bundle["Z"][r].astype(float)      # (N, K*W) window contrasts
        self.C = bundle["C"][r].astype(float)      # (N, 12) age, sex, PC1..PC10
        self.source_weights = bundle["W"][r].astype(float)   # (K, M)
        self.y = bundle["y"][r].astype(float)      # (N,) NaN on test
        sp = bundle["split"][r]
        self.split = sp
        self.idx_train = np.where(sp == 0)[0]
        self.idx_val = np.where(sp == 1)[0]
        self.idx_test = np.where(sp == 2)[0]
        self._ss = bundle["sumstats"]
        self._ev = bundle["evalpheno"]

    def part(self, which):
        i = {"train": self.idx_train, "val": self.idx_val,
             "test": self.idx_test}[which]
        return self.S[i], self.Z[i], self.C[i], self.y[i], i

    def sumstats(self, pop):
        """GWAS summary statistics from one discovery cohort."""
        s = self._ss
        return {"beta": s[f"{pop}_beta"][self.replicate].astype(float),
                "se": s[f"{pop}_se"][self.replicate].astype(float),
                "z": s[f"{pop}_z"][self.replicate].astype(float),
                "maf": s[f"{pop}_maf"][self.replicate].astype(float),
                "n": int(s[f"{pop}_n"][self.replicate]),
                "block_of_snp": self.block_of_snp}

    def ld(self, pop):
        """Block-diagonal LD matrix for one population (shared by all scenarios)."""
        p = os.path.join(os.path.dirname(self.root.rstrip("/")), "common",
                         f"ld_{pop}.npz")
        return np.load(p)["R"].astype(float)

    def eval_cohort(self, pop):
        """A held-out single-ancestry cohort for transfer assessment."""
        p = os.path.join(os.path.dirname(self.root.rstrip("/")), "common",
                         "eval_genotypes.npz")
        g = np.load(p)
        return {"dosage": g[f"dosage_{pop}"].astype(float),
                "covariates": np.column_stack([g[f"age_{pop}"], g[f"sex_{pop}"]]),
                "y": self._ev[f"y_{pop}"][self.replicate].astype(float)}

    @property
    def eval_populations(self):
        return [str(x) for x in self._ev["populations"]]

    def __repr__(self):
        return (f"<Replicate {self.scenario} rep={self.replicate} "
                f"n={len(self.y)} train/val/test="
                f"{len(self.idx_train)}/{len(self.idx_val)}/{len(self.idx_test)}>")


def _bundle(root):
    if root in _CACHE:
        return _CACHE[root]
    g = np.load(os.path.join(root, "genotypes.npz"))
    sc = np.load(os.path.join(root, "scores.npz"))
    zz = np.load(os.path.join(root, "window_scores.npz"))
    sw = np.load(os.path.join(root, "source_weights.npz"))
    cv = np.load(os.path.join(root, "covariates.npz"))
    lb = np.load(os.path.join(root, "labels.npz"))
    b = {"dosage": g["dosage"], "S": sc["S"], "Z": zz["Z"], "W": sw["weights"],
         "C": cv["C"], "y": lb["y"], "split": lb["split"],
         "panel": [str(x) for x in sc["panel"]],
         "block_of_snp": sw["block_of_snp"].astype(int),
         "n_windows": int(zz["n_windows"]),
         "cov_names": [str(x) for x in cv["names"]],
         "sumstats": np.load(os.path.join(root, "sumstats.npz")),
         "evalpheno": np.load(os.path.join(root, "eval_phenotypes.npz")),
         "meta": json.load(open(os.path.join(root, "scenario.json")))}
    _CACHE[root] = b
    return b


def load(scenario_dir, replicate=0):
    """Open one replicate of one scenario."""
    b = _bundle(scenario_dir)
    n = b["meta"]["n_replicates"]
    if not 0 <= replicate < n:
        raise IndexError(f"replicate must be 0..{n-1}, got {replicate}")
    return Replicate(scenario_dir, replicate, b)


def n_replicates(scenario_dir):
    return _bundle(scenario_dir)["meta"]["n_replicates"]


def all_replicates(scenario_dir):
    """Iterate every replicate in a scenario, for paired comparisons."""
    for r in range(n_replicates(scenario_dir)):
        yield load(scenario_dir, r)
