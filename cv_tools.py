"""Stage 3 tool scaffold: common training folds, budget and one confirmation.

Wire evaluate/history/finalize into your own Ollama controller. This is an
evaluation tool, not an AI controller. No test labels or oracle are read.
"""
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from stages.stage1_moeprs import MoEPRS
from models import incremental_r2


class FrozenMoE:
    def __init__(self, config):
        self.config=config
        self.sscale=StandardScaler();self.cscale=StandardScaler()

    def fit(self,S,C,y):
        s=self.sscale.fit_transform(S);c=self.cscale.fit_transform(C)
        self.model=MoEPRS(**self.config).fit(s,c,y)
        return self

    def predict(self,S,C):
        return self.model.predict(self.sscale.transform(S),self.cscale.transform(C))

    def genetic(self,S,C):
        s=self.sscale.transform(S);c=self.cscale.transform(C)
        return (self.model.mixing(c)*self.model._expert_basis(s)*self.model.gamma).sum(1)


class TrainingCVEvaluator:
    BASELINE={'n_pc':10,'l2_gate':0.01,'gate':'linear','expert':'linear'}

    def __init__(self,dataset,budget=12,seed=0):
        if budget<1:raise ValueError('Budget must be positive')
        self.d=dataset;self.budget=budget;self.used=0;self.fit_count=0
        self.records={};self.predictions={};self.finalized=False
        self.S,_,self.C,self.y,_=dataset.part('train')
        self.folds=list(KFold(3,shuffle=True,random_state=seed).split(self.y))
        self.baseline_name=self.evaluate(reason='Required Stage 2 baseline')['name']

    def evaluate(self,n_pc=10,l2_gate=0.01,gate='linear',expert='linear',reason=''):
        if self.finalized:raise RuntimeError('Search is finalized')
        if n_pc not in [2,3,5,10] or l2_gate not in [0.001,0.01,0.1,1.0]:
            raise ValueError('Unsupported gate setting')
        if gate not in ['linear','quadratic'] or expert not in ['linear','power']:
            raise ValueError('Unsupported family')
        config=dict(n_pc=n_pc,l2_gate=l2_gate,gate=gate,expert=expert)
        name=f'pc{n_pc}_l2{l2_gate:g}_{gate}_{expert}'
        if name in self.records:return self.records[name]
        if self.used>=self.budget:raise RuntimeError('Candidate budget exhausted')
        self.used+=1;start=time.monotonic()
        pred=np.empty(len(self.y));genetic=np.empty(len(self.y))
        try:
            for tr,va in self.folds:
                self.fit_count+=1
                m=FrozenMoE(config).fit(self.S[tr],self.C[tr],self.y[tr])
                pred[va]=m.predict(self.S[va],self.C[va])
                genetic[va]=m.genetic(self.S[va],self.C[va])
            if not np.isfinite(pred).all():raise ValueError('Nonfinite predictions')
            baseline=self.predictions[next(iter(self.predictions))] if self.predictions else pred
            gain=(self.y-baseline)**2-(self.y-pred)**2
            residual=self.y-pred
            def corr(a,b):
                return float(np.corrcoef(a,b)[0,1]) if np.std(a)>1e-12 and np.std(b)>1e-12 else 0.0
            rec={'name':name,'ok':True,'config':config,'reason':reason,
                 'oof_incremental_r2':float(incremental_r2(self.y,pred,self.C)),
                 'oof_mse':float(np.mean((self.y-pred)**2)),
                 'paired_mse_gain':float(gain.mean()),
                 'conditional_gain_se':float(gain.std(ddof=1)/np.sqrt(len(gain))),
                 'residual_pc1_correlation':corr(residual,self.C[:,2]),
                 'residual_genetic_squared_correlation':corr(residual,genetic**2)}
            self.predictions[name]=pred
        except Exception as exc:
            rec={'name':name,'ok':False,'config':config,'reason':reason,'error':str(exc)}
            if not self.predictions:raise RuntimeError('Baseline fitting failed') from exc
        rec.update(seconds=time.monotonic()-start,candidate_evaluations_used=self.used,
                   total_fit_attempts=self.fit_count)
        self.records[name]=rec
        return rec

    def history(self):
        return {'experiments':list(self.records.values()),'remaining':self.budget-self.used}

    def finalize(self,name,out):
        if self.finalized:raise RuntimeError('Confirmation is available only once')
        if name not in self.predictions:raise ValueError('Choose an evaluated successful candidate')
        self.finalized=True
        out=Path(out);out.mkdir(parents=True,exist_ok=True)
        S,_,C,y,_=self.d.part('train');Sv,_,Cv,yv,_=self.d.part('val')
        St,_,Ct,_,it=self.d.part('test')
        final_models={};metrics={}
        for key in dict.fromkeys([self.baseline_name,name]):
            self.fit_count+=1
            m=FrozenMoE(self.records[key]['config']).fit(S,C,y)
            pv=m.predict(Sv,Cv)
            metrics[key]={'validation_mse':float(np.mean((yv-pv)**2)),
                          'validation_incremental_r2':float(incremental_r2(yv,pv,Cv))}
            final_models[key]=m
        # Predeclared confirmation policy: reject a worse validation MSE.
        chosen=name if metrics[name]['validation_mse']<=metrics[self.baseline_name]['validation_mse'] else self.baseline_name
        for label,key in [('stage2_moe',self.baseline_name),('selected',chosen)]:
            pd.DataFrame({'iid':it,'prediction':final_models[key].predict(St,Ct)}).to_csv(out/f'{label}_predictions.csv',index=False)
        result={'proposed':name,'selected':chosen,'confirmation':metrics,
                'confirmation_rule':'fallback if validation MSE is worse than Stage 2',
                'candidate_evaluations':self.used,'fit_attempts':self.fit_count,
                'experiments':list(self.records.values())}
        (out/'cv_results.json').write_text(json.dumps(result,indent=2))
        return result
