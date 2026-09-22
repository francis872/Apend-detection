from __future__ import annotations

import numpy as np
import pandas as pd

from .probability import fit_distributions


def bootstrap_distribution_stability(series: pd.Series, n_boot:int=20, sample_size:int=3000, random_state:int=42) -> dict:
    x=pd.to_numeric(series,errors="coerce").dropna()
    if len(x)<50:
        return {"runs":0,"winner_counts":{},"stability":None}
    rng=np.random.default_rng(random_state)
    winners=[]
    n=min(sample_size,len(x))
    for _ in range(max(1,n_boot)):
        sample=x.iloc[rng.integers(0,len(x),size=n)]
        try:
            fits=fit_distributions(sample,bins=60)
            winners.append(str(fits.iloc[0]["distribution"]))
        except Exception:
            continue
    counts=pd.Series(winners).value_counts().to_dict() if winners else {}
    stability=(max(counts.values())/len(winners)) if winners else None
    return {"runs":len(winners),"winner_counts":{str(k):int(v) for k,v in counts.items()},"stability":stability}


def ablation_sensitivity(components: pd.DataFrame, weights: dict[str,float]) -> dict:
    cols=[c for c in components.columns if c in weights]
    if not cols:
        return {}
    w=np.array([weights[c] for c in cols],float)
    w=w/w.sum()
    base=components[cols].to_numpy()@w
    base99=np.quantile(base,.99)
    base_mask=base>=base99
    out={}
    for j,c in enumerate(cols):
        keep=[i for i in range(len(cols)) if i!=j]
        wk=w[keep]
        wk=wk/wk.sum()
        score=components[[cols[i] for i in keep]].to_numpy()@wk
        q=np.quantile(score,.99)
        mask=score>=q
        union=(base_mask|mask).sum()
        jaccard=float((base_mask&mask).sum()/union) if union else 1.0
        out[c]={"jaccard_vs_full_99":jaccard,"threshold_99":float(q)}
    return out
