from __future__ import annotations

import ast
import hashlib
import json
import shutil
import threading
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from scipy import stats

from .comparison import compare_variables
from .derivatives import find_inflection_points
from .detector import detect_outliers
from .exports import export_analysis
from .io import load_and_append, basic_clean
from .integrations import integration_status, fetch_open_meteo, fetch_nasa_firms
from .report_html import write_html_report
from .storage import get_analysis, list_analyses, save_analysis
from .validation import ablation_sensitivity, bootstrap_distribution_stability

VERSION="1.1.0"
app=FastAPI(title="Apend Detection API",version=VERSION)
RUNTIME=Path("runtime/jobs"); RUNTIME.mkdir(parents=True,exist_ok=True)
PROGRESS:dict[str,dict]={}


def _progress(job_id,percent,stage,message):
    PROGRESS[job_id]={"job_id":job_id,"percent":percent,"stage":stage,"message":message}


def _safe_extract(zpath:Path,dest:Path):
    with zipfile.ZipFile(zpath) as z:
        names=z.namelist()
        shp=[n for n in names if n.lower().endswith(".shp")]
        if not shp: raise ValueError("ZIP has no .shp file")
        lower={n.lower() for n in names}
        for s in shp:
            stem=s[:-4].lower()
            missing=[ext for ext in (".dbf",".shx") if stem+ext not in lower]
            if missing: raise ValueError(f"Incomplete Shapefile {s}: missing {', '.join(missing)}")
        root=dest.resolve()
        for member in z.infolist():
            target=(dest/member.filename).resolve()
            if root not in target.parents and target!=root: raise ValueError("Unsafe path in ZIP")
        z.extractall(dest)


def _hash_files(paths):
    h=hashlib.sha256()
    for p in paths:
        with p.open("rb") as f:
            for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()


def _map_points(scored,max_points=5000):
    if scored.crs is None:return []
    geo=scored.to_crs(4326)
    if len(geo)>max_points:
        priority=geo.loc[geo["outlier_90"]]
        rest=geo.loc[~geo["outlier_90"]]
        take=max(0,max_points-len(priority))
        if take<len(rest): rest=rest.sample(take,random_state=42)
        geo=pd.concat([priority,rest]).head(max_points)
    pts=[]
    for idx,row in geo.iterrows():
        if row.geometry is None or row.geometry.is_empty: continue
        pts.append({"id":int(idx),"lat":float(row.geometry.y),"lon":float(row.geometry.x),
          "level":str(row.get("outlier_level","normal")),"status":str(row.get("record_status","normal")),
          "score":float(row.get("outlier_score",0)),"reason":str(row.get("anomaly_reason",""))})
    return pts


def _table_rows(scored,limit=300):
    cols=["outlier_score","probability_tail_area","score_mahal","score_probability","score_spd",
          "score_procrustes","score_spatial","score_temporal","distance_to_inflection",
          "consensus_methods","anomaly_reason","record_status","outlier_level"]
    top=scored.sort_values("outlier_score",ascending=False).head(limit)
    out=[]
    for idx,row in top.iterrows():
        item={"id":int(idx)}
        for col in cols:
            v=row.get(col)
            if isinstance(v,(np.floating,float)): item[col]=None if not np.isfinite(v) else float(v)
            elif isinstance(v,(np.integer,int)): item[col]=int(v)
            else:item[col]=str(v)
        out.append(item)
    return out


def _run_job(job_id,inputs,results,probability_feature,variables,noise_level,quadrature_order,weights,input_hash):
    started=time.perf_counter()
    params={"probability_feature":probability_feature,"variables":variables,"noise_level":noise_level,
            "quadrature_order":quadrature_order,"weights":weights}
    save_analysis(job_id,"running",input_hash=input_hash,probability_feature=probability_feature,params=params)
    try:
        _progress(job_id,10,"loading","Loading and appending Shapefiles")
        gdf=load_and_append(inputs)
        if gdf.crs is None: raise ValueError("Dataset has no CRS. Define the source CRS before analysis.")
        if len(gdf)<30: raise ValueError("Dataset is too small; at least 30 valid records are required.")
        numeric=[c for c in gdf.columns if c!="geometry" and pd.to_numeric(gdf[c],errors="coerce").notna().sum()>=30]
        if probability_feature not in numeric: raise ValueError(f"{probability_feature} is not a usable numeric variable")

        _progress(job_id,20,"cleaning","Cleaning geometry, duplicates and invalid values")
        gdf,cleaning=basic_clean(gdf)
        _progress(job_id,35,"probability","Fitting probability distributions and Gaussian quadrature")
        scored,summary,fits=detect_outliers(gdf,probability_feature=probability_feature,
            quadrature_order=quadrature_order,component_weights=weights)

        _progress(job_id,68,"validation","Bootstrap, ablation and multi-variable comparison")
        component_data=summary.pop("_components_for_validation",{})
        components=pd.DataFrame(component_data)
        summary["model_validation"]={
            "bootstrap":bootstrap_distribution_stability(gdf[probability_feature],n_boot=3,sample_size=1000),
            "ablation":ablation_sensitivity(components,summary["ensemble_weights"])
        }
        projected=gdf.to_crs(gdf.estimate_utm_crs())
        coords=np.c_[projected.geometry.x,projected.geometry.y]
        chosen=[v for v in variables if v in numeric] or [probability_feature]
        comparison=compare_variables(gdf,chosen,coords,quadrature_order=min(24,quadrature_order),max_sample=10000)
        summary["variable_comparison"]=comparison

        _progress(job_id,78,"plots","Preparing PDF, derivatives, map and explainability table")
        vals=pd.to_numeric(scored[probability_feature],errors="coerce").dropna().to_numpy(float)
        hist,edges=np.histogram(vals,bins="fd",density=True)
        centers=((edges[:-1]+edges[1:])/2).astype(float)
        best=fits.iloc[0]; fit_params=ast.literal_eval(best["params"])
        pdf=stats.__dict__[best["distribution"]].pdf(centers,*fit_params)
        deriv=find_inflection_points(best["distribution"],fit_params,lower=float(edges[0]),upper=float(edges[-1]),grid_size=max(512,len(centers)*8))
        qa=summary["gaussian_quadrature"]["confidence_interval_areas"]
        plot={"x":centers.tolist(),"histogram":hist.astype(float).tolist(),
              "pdf":np.nan_to_num(pdf).astype(float).tolist(),"distribution":best["distribution"],
              "first_derivative":np.interp(centers,deriv["x"],deriv["first_derivative"]).tolist(),
              "second_derivative":np.interp(centers,deriv["x"],deriv["second_derivative"]).tolist(),
              "inflection_points":deriv["inflection_points"],
              "intervals":qa}

        reproducibility={"software_version":VERSION,"timestamp_utc":datetime.now(timezone.utc).isoformat(),
          "input_sha256":input_hash,"parameters":params,"crs":str(gdf.crs),"rows_input":int(len(gdf)),
          "random_state":42}
        summary["reproducibility"]=reproducibility

        _progress(job_id,88,"exporting","Exporting GeoPackage, CSV, JSON and reports")
        outputs=export_analysis(scored,fits,cleaning,summary,results,noise_level=noise_level)
        write_html_report(results,cleaning,summary,plot,_map_points(scored))
        outputs["report_html"]="REPORT.html"

        elapsed=time.perf_counter()-started
        payload={"job_id":job_id,"summary":{"cleaning":cleaning,"detector":summary},
          "outputs":{k:f"/jobs/{job_id}/files/{v}" for k,v in outputs.items()},
          "plot":plot,"map_points":_map_points(scored,max_points=5000),"anomalies":_table_rows(scored,limit=300),
          "numeric_variables":numeric,"comparison":comparison,"elapsed_seconds":elapsed,
          "analysis_status":{"numeric_detected":len(numeric),"variables_compared":len(comparison),
            "map_points":len(_map_points(scored,max_points=5000)),"anomaly_rows":len(_table_rows(scored,limit=300))}}
        
        (results/"response.json").write_text(json.dumps(payload,ensure_ascii=False),encoding="utf-8")
        save_analysis(job_id,"complete",input_hash=input_hash,probability_feature=probability_feature,
          rows=len(scored),crs=str(gdf.crs),params=params,summary=summary,outputs=outputs,elapsed_seconds=elapsed)
        _progress(job_id,100,"complete",f"Analysis completed in {elapsed:.1f}s")
    except Exception as e:
        save_analysis(job_id,"failed",input_hash=input_hash,probability_feature=probability_feature,params=params,error=str(e))
        _progress(job_id,100,"failed",str(e))


@app.get("/health")
def health(): return {"status":"ok","service":"Apend Detection","version":VERSION}

@app.get("/integrations")
def integrations(): return integration_status()

@app.get("/external/weather")
def external_weather(lat:float,lon:float):
    try: return fetch_open_meteo(lat,lon)
    except Exception as e: raise HTTPException(502,str(e))

@app.get("/external/firms")
def external_firms(west:float,south:float,east:float,north:float,days:int=1):
    try: return fetch_nasa_firms(west,south,east,north,days)
    except Exception as e: raise HTTPException(502,str(e))


@app.post("/inspect")
async def inspect(files:List[UploadFile]=File(...)):
    job=RUNTIME/("_inspect_"+uuid.uuid4().hex); inputs=job/"input"; inputs.mkdir(parents=True)
    try:
        for i,f in enumerate(files):
            if not (f.filename or "").lower().endswith(".zip"): raise HTTPException(400,"Only ZIP Shapefiles are accepted")
            zp=job/f"u{i}.zip"; zp.write_bytes(await f.read())
            dest=inputs/f"s{i}";dest.mkdir();_safe_extract(zp,dest)
        gdf=load_and_append(inputs)
        numeric=[c for c in gdf.columns if c!="geometry" and pd.to_numeric(gdf[c],errors="coerce").notna().sum()>=30]
        return {"rows":len(gdf),"crs":str(gdf.crs) if gdf.crs else None,"numeric_variables":numeric,
                "valid":bool(gdf.crs and len(gdf)>=30)}
    except Exception as e: raise HTTPException(422,str(e))
    finally: shutil.rmtree(job,ignore_errors=True)


@app.post("/analyze")
async def analyze(files:List[UploadFile]=File(...), probability_feature:str=Form("FRP"),
    variables:str=Form(""), noise_level:int=Form(99), quadrature_order:int=Form(48),
    weights_json:str=Form("")):
    if noise_level not in (90,95,99): raise HTTPException(400,"noise_level must be 90, 95 or 99")
    try: weights=json.loads(weights_json) if weights_json else None
    except json.JSONDecodeError: raise HTTPException(400,"Invalid weights_json")
    selected=[x.strip() for x in variables.split(",") if x.strip()]
    job_id=uuid.uuid4().hex; job=RUNTIME/job_id;inputs=job/"input";results=job/"results"
    inputs.mkdir(parents=True);results.mkdir()
    uploads=[]
    try:
        for i,f in enumerate(files):
            if not (f.filename or "").lower().endswith(".zip"): raise HTTPException(400,"Only ZIP Shapefiles are accepted")
            zp=job/f"upload_{i}.zip"
            with zp.open("wb") as out: shutil.copyfileobj(f.file,out)
            uploads.append(zp);dest=inputs/f"shape_{i}";dest.mkdir();_safe_extract(zp,dest)
    except Exception as e:
        shutil.rmtree(job,ignore_errors=True)
        raise HTTPException(422,str(e))
    input_hash=_hash_files(uploads)
    _progress(job_id,2,"queued","Analysis queued")
    threading.Thread(target=_run_job,args=(job_id,inputs,results,probability_feature,selected,noise_level,quadrature_order,weights,input_hash),daemon=True).start()
    return {"job_id":job_id,"status":"queued","progress_url":f"/jobs/{job_id}/progress"}


@app.get("/jobs/{job_id}/progress")
def progress(job_id:str):
    p=PROGRESS.get(job_id)
    if not p:
        record=get_analysis(job_id)
        if not record: raise HTTPException(404,"Job not found")
        return {"job_id":job_id,"percent":100 if record["status"] in ("complete","failed") else 0,"stage":record["status"],"message":record.get("error") or record["status"]}
    if p["stage"]=="complete":
        p=p|{"result_url":f"/jobs/{job_id}/result"}
    return p


@app.get("/jobs/{job_id}/result")
def result(job_id:str):
    p=RUNTIME/job_id/"results"/"response.json"
    if not p.exists():
        status=PROGRESS.get(job_id,{})
        raise HTTPException(409,status.get("message","Analysis not complete"))
    return json.loads(p.read_text(encoding="utf-8"))


@app.get("/history")
def history(limit:int=50):
    rows=list_analyses(min(max(limit,1),100))
    for r in rows:
        for k in ("params_json","summary_json","outputs_json"): r.pop(k,None)
    return rows


@app.get("/jobs/{job_id}/files/{filename}")
def job_file(job_id:str,filename:str):
    if "/" in filename or "\\" in filename: raise HTTPException(400,"Invalid filename")
    p=RUNTIME/job_id/"results"/filename
    if not p.exists(): raise HTTPException(404,"File not found")
    return FileResponse(p)

ui=Path("ui")
if ui.exists(): app.mount("/ui",StaticFiles(directory=ui,html=True),name="ui")
@app.get("/")
def root(): return RedirectResponse("/ui/" if ui.exists() else "/docs")
