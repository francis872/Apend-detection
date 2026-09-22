from __future__ import annotations

import shutil
import uuid
import zipfile
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .io import load_and_append, basic_clean
from .detector import detect_outliers
from .exports import export_analysis
import ast
import numpy as np
import pandas as pd
from scipy import stats
from .derivatives import find_inflection_points

app=FastAPI(title="Apend Detection API",version="1.0.0")
RUNTIME=Path("runtime/jobs")
RUNTIME.mkdir(parents=True,exist_ok=True)


def _safe_extract(zpath: Path, dest: Path):
    with zipfile.ZipFile(zpath) as z:
        root=dest.resolve()
        for member in z.infolist():
            target=(dest/member.filename).resolve()
            if root not in target.parents and target!=root:
                raise HTTPException(400,"Unsafe path in ZIP")
        z.extractall(dest)


@app.get("/health")
def health():
    return {"status":"ok","service":"Apend Detection","version":"1.0.0"}


@app.post("/analyze")
async def analyze(
    files: List[UploadFile]=File(...),
    probability_feature: str=Form("FRP"),
    noise_level: int=Form(99),
    quadrature_order: int=Form(48),
):
    if noise_level not in (90,95,99):
        raise HTTPException(400,"noise_level must be 90, 95 or 99")
    job_id=uuid.uuid4().hex
    job=RUNTIME/job_id; inputs=job/"input"; results=job/"results"
    inputs.mkdir(parents=True); results.mkdir(parents=True)
    for i,f in enumerate(files):
        if not (f.filename or "").lower().endswith(".zip"):
            raise HTTPException(400,"Upload ZIP Shapefiles")
        zpath=job/f"upload_{i}.zip"
        with zpath.open("wb") as out:
            shutil.copyfileobj(f.file,out)
        dest=inputs/f"shape_{i}"; dest.mkdir()
        _safe_extract(zpath,dest)
    try:
        gdf=load_and_append(inputs)
        gdf,cleaning=basic_clean(gdf)
        scored,summary,fits=detect_outliers(
            gdf,probability_feature=probability_feature,quadrature_order=quadrature_order
        )
        outputs=export_analysis(scored,fits,cleaning,summary,results,noise_level=noise_level)
        vals=pd.to_numeric(scored[probability_feature],errors="coerce").dropna().to_numpy(float)
        hist,edges=np.histogram(vals,bins="fd",density=True)
        centers=((edges[:-1]+edges[1:])/2).astype(float)
        best=fits.iloc[0]
        params=ast.literal_eval(best["params"])
        pdf=stats.__dict__[best["distribution"]].pdf(centers,*params)
        deriv=find_inflection_points(best["distribution"],params,lower=float(edges[0]),upper=float(edges[-1]),grid_size=max(512,len(centers)*8))
        d1=np.interp(centers,deriv["x"],deriv["first_derivative"])
        d2=np.interp(centers,deriv["x"],deriv["second_derivative"])
        plot_data={
            "x":centers.tolist(),"histogram":hist.astype(float).tolist(),
            "pdf":np.nan_to_num(pdf,nan=0.0,posinf=0.0,neginf=0.0).astype(float).tolist(),
            "distribution":best["distribution"],
            "first_derivative":d1.astype(float).tolist(),
            "second_derivative":d2.astype(float).tolist(),
            "inflection_points":deriv["inflection_points"]
        }
    except Exception as e:
        raise HTTPException(422,f"Analysis failed: {e}") from e
    return {
        "job_id":job_id,"summary":{"cleaning":cleaning,"detector":summary},
        "outputs":{k:f"/jobs/{job_id}/files/{v}" for k,v in outputs.items()}, "plot":plot_data
    }


@app.get("/jobs/{job_id}/files/{filename}")
def job_file(job_id: str, filename: str):
    if "/" in filename or "\\" in filename:
        raise HTTPException(400,"Invalid filename")
    p=RUNTIME/job_id/"results"/filename
    if not p.exists():
        raise HTTPException(404,"File not found")
    return FileResponse(p)


ui=Path("ui")
if ui.exists():
    app.mount("/ui",StaticFiles(directory=ui,html=True),name="ui")


@app.get("/")
def root():
    return RedirectResponse("/ui/" if ui.exists() else "/docs")
