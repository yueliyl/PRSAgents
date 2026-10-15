"""Stage 2 reference check using five fixed VIPRS experts and PC-only EM.

All tuning uses training CV. Validation is assessed once; test labels are never
loaded. Students use the provided model; implementing EM from scratch is not required.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_predict, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from harness import Dataset
from models import incremental_r2
from moeprs import MoEPRS


def load_experts(d, scores=None):
    names = [name for name in d.panel if name.endswith('_viprs')]
    if len(names) != 5:
        raise ValueError('Expected five VIPRS columns')
    columns = [d.panel.index(name) for name in names]
    old_k = len(d.panel)
    d.Z = d.Z[:, [w*old_k+k for w in range(d.n_windows) for k in columns]]
    d.source_weights = d.source_weights[columns]
    if scores:
        frame = pd.read_csv(scores).set_index('iid')
        d.S = frame.loc[np.arange(len(d.C)), names].to_numpy(float)
    else:
        d.S = d.S[:, columns]
    d.panel = names
    return d


def run(data, out, scores=None, train_size=None, replicate=0):
    d = load_experts(Dataset(str(data), replicate), scores)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    S, _, C, y, _ = d.part('train')
    if train_size:
        if train_size > len(y) or train_size < 10:
            raise ValueError('Invalid training subset size')
        take = np.random.default_rng(0).permutation(len(y))[:train_size]
        S, C, y = S[take], C[take], y[take]
        dosage = d.dosage[d.idx_train[take]].astype(float)
    else:
        dosage = d.dosage[d.idx_train].astype(float)
    cv = KFold(3, shuffle=True, random_state=0)
    single_losses = []
    for k in range(S.shape[1]):
        pred = cross_val_predict(LinearRegression(), np.column_stack([C, S[:, k]]), y, cv=cv)
        single_losses.append(np.mean((y - pred)**2))
    best_k = int(np.argmin(single_losses))
    single = LinearRegression().fit(np.column_stack([C, S[:, best_k]]), y)
    stack = GridSearchCV(make_pipeline(StandardScaler(), Ridge()),
                         {'ridge__alpha': np.logspace(-1, 3, 9)},
                         scoring='neg_mean_squared_error', cv=cv)
    stack.fit(np.column_stack([C, S]), y)
    # Scaling of variants/covariates is refit inside each tuning fold.
    ridge = GridSearchCV(make_pipeline(StandardScaler(), Ridge()),
                         {'ridge__alpha': np.logspace(1, 5, 5)},
                         scoring='neg_mean_squared_error', cv=cv)
    ridge.fit(np.column_stack([C, dosage]), y)
    moe = MoEPRS(n_pc=10, l2_gate=0.01).fit(S, C, y)
    pd.DataFrame({'iteration': np.arange(len(moe.loglik_history_)),
                  'training_log_likelihood': moe.loglik_history_}).to_csv(out/'moe_training_loglik.csv', index=False)
    (out/'stage2_config.json').write_text(json.dumps({
        'implementation': 'provided compact teaching MoEPRS (moeprs.py)',
        'n_pc': 10, 'l2_gate': 0.01, 'n_em': moe.n_em, 'seed': moe.seed,
        'n_iter': moe.n_iter_, 'shared_residual_variance': moe.sigma2,
        'experts': d.panel, 'n_train': len(y),
    }, indent=2) + '\n')
    models = {
        'best_single': lambda s, c, x: single.predict(np.column_stack([c, s[:, best_k]])),
        'multiprs': lambda s, c, x: stack.predict(np.column_stack([c, s])),
        'genotype_ridge': lambda s, c, x: ridge.predict(np.column_stack([c, x])),
        'stage2_moe': lambda s, c, x: moe.predict(s, c),
    }
    Sv, _, Cv, yv, iv = d.part('val')
    St, _, Ct, _, it = d.part('test')
    rows=[]
    for name, predict in models.items():
        pv = predict(Sv, Cv, d.dosage[iv].astype(float))
        pt = predict(St, Ct, d.dosage[it].astype(float))
        rows.append({'method': name, 'validation_incremental_r2': incremental_r2(yv,pv,Cv),
                     'validation_mse': np.mean((yv-pv)**2), 'n_train': len(y)})
        pd.DataFrame({'iid': it, 'prediction': pt}).to_csv(out/f'{name}_predictions.csv',index=False)
    result=pd.DataFrame(rows); result.to_csv(out/'validation_metrics.csv',index=False)
    # A plotting-ready gate table; actual plotting remains student work.
    weights=moe.mixing(Cv)
    gate=pd.DataFrame(weights,columns=d.panel)
    gate.insert(0,'iid',iv); gate['PC1']=Cv[:,2]; gate['PC2']=Cv[:,3]
    gate.to_csv(out/'validation_gate_weights.csv',index=False)
    print(result.to_string(index=False))
    print('Best single source selected using training CV:',d.panel[best_k])
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('data',nargs='?',default='data/shared_effects')
    ap.add_argument('--replicate', type=int, default=0)
    ap.add_argument('--out',default='outputs/stage2_demo')
    ap.add_argument('--scores',help='Optional Stage 1 target_scores.csv')
    ap.add_argument('--train-size',type=int)
    args=ap.parse_args()
    run(args.data,args.out,args.scores,args.train_size,args.replicate)
