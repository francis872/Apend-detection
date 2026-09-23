from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


def euclidean_intelligence(
    frame:pd.DataFrame,
    features:list[str],
    spatial_columns:tuple[str,str]|None=None,
    spatial_weight:float=.25,
    feature_weight:float=.75,
    baseline_mask:np.ndarray|None=None,
)->tuple[pd.DataFrame,dict]:
    """Robust Euclidean-distance intelligence model.

    Distances are computed in robustly scaled feature space so variables with different
    units do not dominate. Optional spatial distance is also robustly scaled and combined
    only as a contextual component.
    """
    if len(features)<2:
        raise ValueError("At least two features are required")
    missing=[c for c in features if c not in frame.columns]
    if missing: raise KeyError("Missing features: "+", ".join(missing))

    x=frame[features].apply(pd.to_numeric,errors="coerce")
    med=x.median()
    x=x.fillna(med)
    scaler=RobustScaler()
    z=scaler.fit_transform(x)

    if baseline_mask is None:
        baseline=np.ones(len(frame),dtype=bool)
    else:
        baseline=np.asarray(baseline_mask,dtype=bool)
        if len(baseline)!=len(frame):raise ValueError("baseline_mask length mismatch")
        if baseline.sum()<3:raise ValueError("Baseline requires at least 3 observations")

    center=np.median(z[baseline],axis=0)
    feature_distance=np.linalg.norm(z-center,axis=1)
    f_med=float(np.median(feature_distance[baseline]))
    f_mad=float(np.median(np.abs(feature_distance[baseline]-f_med)))+1e-12
    feature_robust=(feature_distance-f_med)/(1.4826*f_mad)
    feature_score=1/(1+np.exp(-feature_robust))

    spatial_distance=np.zeros(len(frame),float)
    spatial_score=np.zeros(len(frame),float)
    if spatial_columns:
        a,b=spatial_columns
        if a not in frame.columns or b not in frame.columns:
            raise KeyError("Spatial columns not found")
        coords=frame[[a,b]].apply(pd.to_numeric,errors="coerce").fillna(0).to_numpy(float)
        cscale=RobustScaler()
        cz=cscale.fit_transform(coords)
        ccenter=np.median(cz[baseline],axis=0)
        spatial_distance=np.linalg.norm(cz-ccenter,axis=1)
        s_med=float(np.median(spatial_distance[baseline]))
        s_mad=float(np.median(np.abs(spatial_distance[baseline]-s_med)))+1e-12
        spatial_robust=(spatial_distance-s_med)/(1.4826*s_mad)
        spatial_score=1/(1+np.exp(-spatial_robust))

    fw=max(0.0,float(feature_weight)); sw=max(0.0,float(spatial_weight if spatial_columns else 0.0))
    total=fw+sw
    if total<=0:raise ValueError("At least one weight must be positive")
    fw/=total; sw/=total
    combined=fw*feature_score+sw*spatial_score

    q90,q95,q99=[float(np.quantile(combined,q)) for q in (.90,.95,.99)]
    out=frame.copy()
    out["euclidean_feature_distance"]=feature_distance
    out["euclidean_spatial_distance"]=spatial_distance
    out["euclidean_feature_score"]=feature_score
    out["euclidean_spatial_score"]=spatial_score
    out["euclidean_intelligence_score"]=combined
    out["euclidean_outlier_90"]=combined>=q90
    out["euclidean_outlier_95"]=combined>=q95
    out["euclidean_outlier_99"]=combined>=q99

    contribution={}
    scale=np.abs(z-center)
    denom=np.maximum(scale.sum(axis=1,keepdims=True),1e-12)
    rel=scale/denom
    for i,f in enumerate(features):
        out[f"euclidean_contrib_{f}"]=rel[:,i]
        contribution[f]=float(np.mean(rel[:,i]))

    summary={
        "method":"Robust Euclidean Distance Intelligence",
        "features":features,
        "baseline_rows":int(baseline.sum()),
        "weights":{"feature":fw,"spatial":sw},
        "thresholds":{"90":q90,"95":q95,"99":q99},
        "counts":{"90":int((combined>=q90).sum()),"95":int((combined>=q95).sum()),"99":int((combined>=q99).sum())},
        "feature_contribution_mean":contribution,
        "center":dict(zip(features,center.tolist())),
        "policy":{
            "distance_is_not_probability":True,
            "robust_scaling":True,
            "causal_claims":False,
        },
    }
    return out,summary


def compare_to_baseline(current:dict,baseline:dict,features:list[str])->dict:
    """Euclidean distance between two EO/territorial feature summaries."""
    a=[];b=[];used=[]
    for f in features:
        if f in current and f in baseline:
            try:
                av=float(current[f]);bv=float(baseline[f])
                if np.isfinite(av) and np.isfinite(bv):
                    a.append(av);b.append(bv);used.append(f)
            except Exception:
                pass
    if len(used)<2:raise ValueError("At least two shared numeric features are required")
    a=np.asarray(a);b=np.asarray(b)
    scale=np.maximum(np.abs(b),1e-6)
    normalized=(a-b)/scale
    distance=float(np.linalg.norm(normalized))
    return {
        "distance":distance,
        "features":used,
        "per_feature_delta":{f:float(current[f])-float(baseline[f]) for f in used},
        "normalized_delta":dict(zip(used,normalized.tolist())),
        "interpretation":"Higher distance means the current feature vector is farther from the baseline under relative Euclidean geometry; it is not a probability.",
    }
