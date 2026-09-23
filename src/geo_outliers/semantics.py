from __future__ import annotations

import re
from difflib import SequenceMatcher
import pandas as pd

ROLE_ALIASES={
    "latitude":("lat","latitude","y","latitud"),
    "longitude":("lon","lng","long","longitude","x","longitud"),
    "timestamp":("timestamp","datetime","date_time","fecha_hora","acq_datetime"),
    "date":("date","fecha","acq_date"),
    "time":("time","hora","acq_time"),
    "id":("id","objectid","fid","record_id","asset_id","vehicle_id","property_id"),
}

def _norm(s:str)->str:
    s=str(s).strip().lower()
    s=re.sub(r"[^a-z0-9]+","_",s).strip("_")
    return s

def _similarity(a:str,b:str)->float:
    a,b=_norm(a),_norm(b)
    if not a or not b:return 0.0
    if a==b:return 1.0
    if a in b or b in a:return .90
    return SequenceMatcher(None,a,b).ratio()

def profile_columns(df:pd.DataFrame)->list[dict]:
    rows=[]
    n=max(len(df),1)
    for col in df.columns:
        if str(col)=="geometry":continue
        s=df[col]
        numeric=pd.to_numeric(s,errors="coerce")
        dt=pd.to_datetime(s,errors="coerce") if not pd.api.types.is_numeric_dtype(s) else pd.Series(pd.NaT,index=s.index)
        numeric_ratio=float(numeric.notna().sum()/n)
        datetime_ratio=float(dt.notna().sum()/n)
        rows.append({
            "column":str(col),
            "dtype":str(s.dtype),
            "non_null":int(s.notna().sum()),
            "missing_ratio":float(s.isna().sum()/n),
            "unique":int(s.nunique(dropna=True)),
            "numeric_ratio":numeric_ratio,
            "datetime_ratio":datetime_ratio,
            "usable_numeric":numeric.notna().sum()>=30,
        })
    return rows

def semantic_contract(df:pd.DataFrame, domain_pack:dict|None=None)->dict:
    profiles=profile_columns(df)
    columns=[x["column"] for x in profiles]
    mappings=[]
    used=set()

    # Structural roles first.
    for role,aliases in ROLE_ALIASES.items():
        best=None
        for c in columns:
            score=max(_similarity(c,a) for a in aliases)
            if best is None or score>best[1]:best=(c,score)
        if best and best[1]>=.78:
            mappings.append({"column":best[0],"semantic":role,"role":"structural","confidence":round(best[1],3),"source":"alias"})
            used.add(best[0])

    # Domain semantics.
    if domain_pack:
        variables=list(domain_pack.get("variables") or [])
        alias_groups=list(domain_pack.get("aliases") or [])
        groups={}
        for v in variables:groups[v]=[v]
        for group in alias_groups:
            if group:
                canonical=group[0];groups.setdefault(canonical,[])
                groups[canonical]=list(dict.fromkeys(groups[canonical]+list(group)))
        for semantic,aliases in groups.items():
            best=None
            for c in columns:
                if c in used:continue
                score=max(_similarity(c,a) for a in aliases)
                if best is None or score>best[1]:best=(c,score)
            if best and best[1]>=.62:
                mappings.append({"column":best[0],"semantic":semantic,"role":"measure","confidence":round(best[1],3),"source":"domain_pack"})
                used.add(best[0])

    # Generic numeric measures.
    for p in profiles:
        if p["column"] not in used and p["usable_numeric"]:
            mappings.append({"column":p["column"],"semantic":_norm(p["column"]),"role":"measure","confidence":.5,"source":"generic_numeric"})

    issues=[]
    if not any(m["semantic"]=="latitude" for m in mappings) and not hasattr(df,"geometry"):
        issues.append("No latitude field or geometry detected")
    if not any(m["semantic"]=="longitude" for m in mappings) and not hasattr(df,"geometry"):
        issues.append("No longitude field or geometry detected")
    measures=[m for m in mappings if m["role"]=="measure"]
    if len(measures)<2:issues.append("Fewer than two usable numeric measures detected")
    high_missing=[p["column"] for p in profiles if p["missing_ratio"]>.5]
    if high_missing:issues.append("High missingness (>50%): "+", ".join(high_missing[:8]))

    return {
        "version":"1.0",
        "domain":(domain_pack or {}).get("id","general"),
        "rows":int(len(df)),
        "columns":profiles,
        "mappings":mappings,
        "measures":[m["column"] for m in measures],
        "semantic_measures":[m["semantic"] for m in measures],
        "issues":issues,
        "ready":len(measures)>=2,
    }

def apply_contract(df:pd.DataFrame, contract:dict):
    """Add canonical semantic aliases without deleting original source columns."""
    out=df.copy()
    applied={}
    for m in contract.get("mappings",[]):
        src=m.get("column"); dst=m.get("semantic")
        if not src or not dst or src not in out.columns or src==dst:continue
        if dst not in out.columns:
            out[dst]=out[src]
            applied[dst]=src
    return out,applied
