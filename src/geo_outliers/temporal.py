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
    if valid.sum() < 20:
        return out,{"valid_rows":int(valid.sum()),"window":window,"method":"insufficient temporal data"}
    tmp=pd.DataFrame({"idx":np.flatnonzero(valid.to_numpy()),"ts":ts[valid].to_numpy(),"v":values[valid].to_numpy()}).sort_values("ts")
    w=min(window, max(11, len(tmp)//5*2+1))
    if w%2==0:w+=1
    s=pd.Series(tmp["v"].to_numpy())
    med=s.rolling(w,center=True,min_periods=max(5,w//5)).median()
    absdev=(s-med).abs()
    mad=absdev.rolling(w,center=True,min_periods=max(5,w//5)).median()
    rz=absdev/(1.4826*mad.replace(0,np.nan))
    rz=rz.fillna(absdev/(absdev.median()+1e-12)).to_numpy()
    denom=max(len(rz)-1,1)
    rank=np.argsort(np.argsort(rz,kind="mergesort"),kind="mergesort")/denom
    out[tmp["idx"].to_numpy(int)]=rank
    return out,{
        "valid_rows":int(valid.sum()),"window":int(w),
        "start":str(tmp["ts"].min()),"end":str(tmp["ts"].max()),
        "method":"rolling median/MAD temporal deviation"
    }
