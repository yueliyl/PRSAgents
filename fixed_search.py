"""Fixed seeded random-search comparison using the same CV tool and budget."""
import argparse
from itertools import product
import random
from harness import Dataset
from stage2_baselines import load_experts
from cv_tools import TrainingCVEvaluator


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('data',nargs='?',default='data/datasets/dataset_01/rep00')
    ap.add_argument('--budget',type=int,default=12)
    ap.add_argument('--scores');ap.add_argument('--out',default='outputs/fixed_search')
    a=ap.parse_args()
    tool=TrainingCVEvaluator(load_experts(Dataset(a.data),a.scores),budget=a.budget)
    configs=[dict(n_pc=pc,l2_gate=penalty,gate=gate,expert=expert)
             for pc,penalty,gate,expert in product([2,3,5,10],[0.001,0.01,0.1,1.0],
                                                   ['linear','quadratic'],['linear','power'])]
    random.Random(0).shuffle(configs)
    for config in configs:
        if tool.used>=tool.budget:break
        print(tool.evaluate(**config,reason='Fixed seeded random order'))
    best=min((r for r in tool.records.values() if r['ok']),key=lambda r:r['oof_mse'])
    print(tool.finalize(best['name'],a.out))
