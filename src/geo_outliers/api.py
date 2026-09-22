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

app=FastAPI(title="Apend Detection API",version="0.2.0")
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
    return {"status":"ok","service":"Apend Detection","version":"0.2.0"}


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
    except Exception as e:
        raise HTTPException(422,f"Analysis failed: {e}") from e
    return {
        "job_id":job_id,"summary":{"cleaning":cleaning,"detector":summary},
        "outputs":{k:f"/jobs/{job_id}/files/{v}" for k,v in outputs.items()}
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
