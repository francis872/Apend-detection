from __future__ import annotations

import ast
import numpy as np
import pandas as pd

from .probability import fit_distributions
from .quadrature import quadrature_tail_score
from .temporal import temporal_validation
from .spatial import spatial_validation


def compare_variables(gdf, variables:list[str], coords:np.ndarray, quadrature_order:int=32) -> list[dict]:
    rows=[]
    for var in variables:
        if var not in gdf.columns:
            continue
        values=pd.to_numeric(gdf[var],errors="coerce")
        finite=values.dropna()
        if len(finite)<30:
            rows.append({"variable":var,"status":"insufficient_data","n":int(len(finite))})
            continue
        try:
            fits=fit_distributions(values)
            best=fits.iloc[0]
            params=ast.literal_eval(best["params"])
            tail,score=quadrature_tail_score(values.to_numpy(float),best["distribution"],params,n=quadrature_order)
            spatial,_=spatial_validation(coords,values.to_numpy(float),k=12)
            temporal,_=temporal_validation(gdf,var)
            rows.append({
                "variable":var,"status":"ok","n":int(len(finite)),
                "distribution":str(best["distribution"]),
                "ks":float(best["ks_stat"]),"bhattacharyya":float(best["bhattacharyya"]),
                "fisher_rao":float(best["fisher_rao"]),
                "probability_anomalies_95":int(np.nansum(score>=np.nanquantile(score,.95))),
                "spatial_anomalies_95":int(np.nansum(spatial>=.95)),
                "temporal_anomalies_95":int(np.nansum(temporal>=.95)),
                "median_tail_area":float(np.nanmedian(tail)),
            })
        except Exception as e:
            rows.append({"variable":var,"status":"error","error":str(e)})
    return rows
