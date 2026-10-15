"""Check the complete student release without needing Ollama or test labels.

    python3 check_package.py            # sizes
    python3 check_package.py --full     # sizes and SHA256
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

SCENARIOS = ["shared_effects", "ancestry_dependent", "locus_varying", "nonlinear_link"]
POPS = ["EUR", "AFR", "EAS", "SAS", "AMR"]
EVAL_POPS = ["AFR", "EUR", "SAS", "EAS", "NAT", "AMR"]
SCENARIO_FILES = ["genotypes.npz", "scores.npz", "window_scores.npz",
                  "source_weights.npz", "covariates.npz", "labels.npz",
                  "sumstats.npz", "eval_phenotypes.npz", "windows.json",
                  "scenario.json"]
FORBIDDEN = ("labels_test", "regime.json", "_instructor", "founders.npz",
             "alias_map", "dataset_mapping")


def main(root, full=False):
    root = Path(root).resolve()
    if sys.version_info < (3, 10):
        sys.exit(f"Python 3.10 or newer required, found {sys.version.split()[0]}")

    manifest = json.loads((root / "manifest.json").read_text())
    for item in manifest["files"]:
        path = root / item["path"]
        assert path.is_file(), path
        assert path.stat().st_size == item["bytes"], path
        if full:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], path

    for word in FORBIDDEN:
        hits = [p for p in root.rglob("*") if word in p.relative_to(root).as_posix()]
        assert not hits, hits[:3]

    common = root / "data" / "common"
    assert (common / "eval_genotypes.npz").is_file(), common
    for pop in POPS:
        assert (common / f"ld_{pop}.npz").is_file(), pop

    sys.path.insert(0, str(root))
    from harness import Dataset
    from prsdata import n_replicates

    total = 0
    for scen in SCENARIOS:
        sdir = root / "data" / scen
        assert sdir.is_dir(), sdir
        for f in SCENARIO_FILES:
            assert (sdir / f).is_file(), sdir / f
        R = n_replicates(str(sdir))
        assert R == 10, (scen, R)
        for r in range(R):
            d = Dataset(str(sdir), r)
            assert d.dosage.shape == (4000, 1000), (scen, r)
            assert d.S.shape == (4000, 10), (scen, r)
            assert d.Z.shape == (4000, 100), (scen, r)
            assert d.C.shape == (4000, 12), (scen, r)
            assert d.source_weights.shape == (10, 1000), (scen, r)
            assert (len(d.idx_train), len(d.idx_val), len(d.idx_test)) == (2000, 800, 1200)
            assert np.isnan(d.y[d.idx_test]).all(), (scen, r)
            assert np.isfinite(d.y[d.idx_train]).all(), (scen, r)
            assert set(d.idx_train).isdisjoint(d.idx_val)
            assert set(d.idx_train).isdisjoint(d.idx_test)
            assert set(d.idx_val).isdisjoint(d.idx_test)
            assert np.isfinite(d.S).all() and np.isfinite(d.C).all(), (scen, r)
            assert d.block_of_snp.shape == (1000,), (scen, r)
            for pop in POPS:
                ss = d.sumstats(pop)
                assert ss["beta"].shape == (1000,) and ss["n"] > 0, (scen, r, pop)
            for pop in EVAL_POPS:
                ev = d.eval_cohort(pop)
                assert ev["dosage"].shape == (1500, 1000), (scen, r, pop)
                assert ev["y"].shape == (1500,), (scen, r, pop)
            assert "oracle" not in json.dumps(d.info).lower(), (scen, r)
            total += 1

    print(f"PASS: {len(SCENARIOS)} scenarios x 10 replicates = {total} datasets; "
          "shapes, fixed splits and withheld test labels verified")
    print(f"PASS: {len(manifest['files'])} manifest files"
          + (" with SHA256 checks" if full else " checked"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=Path(__file__).resolve().parent)
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args(); main(a.root, a.full)
