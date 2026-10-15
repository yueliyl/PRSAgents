"""Minimal Stage 2 fit/predict example; run from Student_Package."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from harness import Dataset
from stage2_baselines import load_experts
from moeprs import MoEPRS
from models import incremental_r2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('data', nargs='?', default='data/shared_effects')
    ap.add_argument('--replicate', type=int, default=0)
    ap.add_argument('--scores', help='Optional Stage 1 target_scores.csv')
    ap.add_argument('--out', default='outputs/stage2_minimal')
    args = ap.parse_args()
    d = load_experts(Dataset(args.data, args.replicate), args.scores)
    S, _, C, y, _ = d.part('train')
    model = MoEPRS(n_pc=10, l2_gate=0.01).fit(S, C, y)
    Sv, _, Cv, yv, _ = d.part('val')
    pred = model.predict(Sv, Cv)
    weights = model.mixing(Cv)
    assert np.isfinite(weights).all() and np.allclose(weights.sum(axis=1), 1)
    print('Validation incremental R2:', incremental_r2(yv, pred, Cv))
    print('Validation MSE:', np.mean((yv - pred) ** 2))
    St, _, Ct, _, ids = d.part('test')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'iid': ids, 'prediction': model.predict(St, Ct)}).to_csv(out/'predictions.csv', index=False)


if __name__ == '__main__':
    main()
