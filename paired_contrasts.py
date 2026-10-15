"""Compare a candidate and baseline on the SAME independent replicate rows.

Input CSV columns: dataset, rep, method, score. Scores may be validation or
instructor-returned test scores; never pool them or mix different metrics.
"""
import argparse
import numpy as np
import pandas as pd


def summarize(frame, baseline, candidate):
    keys=['dataset','rep','method']
    if frame.duplicated(keys).any(): raise ValueError('Duplicate method/replicate rows')
    a=frame[frame.method==baseline].set_index(['dataset','rep']).score.rename('baseline')
    b=frame[frame.method==candidate].set_index(['dataset','rep']).score.rename('candidate')
    paired=pd.concat([a,b],axis=1).dropna()
    if paired.empty: raise ValueError('No common replicate support')
    rows=[]
    for name,g in [('pooled',paired)]+list(paired.groupby(level='dataset')):
        delta=g.candidate-g.baseline; n=len(delta)
        se=delta.std(ddof=1)/np.sqrt(n) if n>1 else np.nan
        rows.append({'dataset':name,'n_pairs':n,'mean_gain':delta.mean(),'se':se})
    return pd.DataFrame(rows)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('csv');ap.add_argument('--baseline',default='stage2_moe')
    ap.add_argument('--candidate',default='qwen_agent')
    a=ap.parse_args(); print(summarize(pd.read_csv(a.csv),a.baseline,a.candidate).to_string(index=False))
