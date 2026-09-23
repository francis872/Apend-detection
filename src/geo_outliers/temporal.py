from __future__ import annotations

import numpy as np
import pandas as pd


def parse_timestamp(df: pd.DataFrame) -> pd.Series:
    if "ACQ_DATE" not in df.columns:
        return pd.Series(pd.NaT,index=df.index,dtype="datetime64[ns]")
    date=pd.to_datetime(df["ACQ_DATE"],errors="coerce")
    if "ACQ_TIME" in df.columns:
        raw=df["ACQ_TIME"].astype(str).str.replace(r"\.0$","",regex=True).str.zfill(4)
        hh=pd.to_numeric(raw.str[:2],errors="coerce").fillna(0)
        mm=pd.to_numeric(raw.str[2:4],errors="coerce").fillna(0)
        date=date+pd.to_timedelta(hh,unit="h")+pd.to_timedelta(mm,unit="m")
    return date


def temporal_validation(df: pd.DataFrame, value_col: str, window: int = 101) -> tuple[np.ndarray, dict]:
    ts=parse_timestamp(df)
    values=pd.to_numeric(df[value_col],errors="coerce")
    valid=ts.notna() & values.notna()
    out=np.zeros(len(df),float)
    if valid.sum()<20:
        return out,{"valid_rows":int(valid.sum()),"window":window,"method":"insufficient temporal data","change_points":[]}

    tmp=pd.DataFrame({"idx":np.flatnonzero(valid.to_numpy()),"ts":ts[valid].to_numpy(),"v":values[valid].to_numpy()}).sort_values("ts")
    w=min(window,max(11,len(tmp)//5*2+1))
    if w%2==0:w+=1
    s=pd.Series(tmp["v"].to_numpy())
    med=s.rolling(w,center=True,min_periods=max(5,w//5)).median()
    absdev=(s-med).abs()
    mad=absdev.rolling(w,center=True,min_periods=max(5,w//5)).median()
    rz=absdev/(1.4826*mad.replace(0,np.nan))
    rz=rz.fillna(absdev/(absdev.median()+1e-12)).to_numpy()
    rank=np.argsort(np.argsort(rz,kind="mergesort"),kind="mergesort")/max(len(rz)-1,1)
    out[tmp["idx"].to_numpy(int)]=rank

    # Lightweight change-point proxy based on robust first differences.
    dif=np.abs(np.diff(s.to_numpy(float),prepend=s.iloc[0]))
    med_d=float(np.median(dif)); mad_d=float(np.median(np.abs(dif-med_d)))+1e-12
    cp_score=(dif-med_d)/(1.4826*mad_d)
    cp_idx=np.flatnonzero(cp_score>3.5)
    cps=[{"timestamp":str(tmp.iloc[i]["ts"]),"value":float(tmp.iloc[i]["v"]),"score":float(cp_score[i])} for i in cp_idx[:25]]

    # Hourly aggregation for UI.
    hourly=(tmp.set_index("ts")["v"].resample("1h").agg(["count","mean","median"]).dropna().reset_index())
    series=[{"timestamp":str(r.ts),"count":int(r["count"]),"mean":float(r["mean"]),"median":float(r["median"])} for _,r in hourly.tail(240).iterrows()]

    trend=float(np.polyfit(np.arange(len(s)),s.to_numpy(float),1)[0]) if len(s)>1 else 0.0
    return out,{
        "valid_rows":int(valid.sum()),
        "window":int(w),
        "start":str(tmp["ts"].min()),
        "end":str(tmp["ts"].max()),
        "method":"rolling median/MAD temporal deviation",
        "change_points":cps,
        "change_point_count":int(len(cp_idx)),
        "trend_slope_per_observation":trend,
        "hourly_series":series
    }
