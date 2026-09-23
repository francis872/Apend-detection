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
from .domains import get_domain_pack, list_domain_packs, configure_domain
from .intelligence import build_intelligence
from .semantics import semantic_contract, apply_contract
from .automation import build_autopilot_plan
from .enrichment import plan_auto_enrichment, execute_auto_enrichment
from .orchestrator import MeridianOrchestrator
from .event_store import append_event, list_events, event_stats
from .governance import ensure_champion, get_champion, register_challenger, list_strategies, evaluate_strategy, promote, rollback, experiment_history, monitor_drift, promotion_gate, governance_dashboard
from .exports import export_analysis
from .io import load_and_append, basic_clean
from .integrations import integration_status, fetch_open_meteo, fetch_nasa_firms, fetch_firms_area, sample_weather_enrichment
from .report_html import write_html_report
from .storage import get_analysis, list_analyses, save_analysis
from .validation import ablation_sensitivity, bootstrap_distribution_stability
from .incidents import get_incident_case, update_incident_management, add_incident_note, build_intelligence_brief, operations_cases
from .migrations import migrate_all, migration_status
from .data_quality import assess_data_quality, quality_gate
from .earth_observation import sensor_catalog, index_catalog, search_scenes, search_geometry, build_index_plan
from .computational_intelligence import euclidean_intelligence, compare_to_baseline
from .raster_processing import process_scene
from .socio_ecological import register_dataset, list_datasets, lineage, create_event, territorial_snapshot, snapshots, territorial_indicators, correlation, distance_model, hotspot, detect_corridors
from .network_corridor import build_territorial_graph, shortest_path, detect_network_corridors
from .spatial_operations import spatial_operation
from .corridor_store import save_corridors, list_corridors, get_corridor
from .territorial_fusion import fuse_features, feature_history, temporal_profile, land_use_divergence
from .advanced_spatial import getis_ord_gi_star, bivariate_moran, spatial_association_matrix, temporal_spatial_comparison
from .spatial_dynamics import trajectory, compare_trajectories, cluster_trajectories
from .trajectory_watch import watch_territory, assessment_history
from .territorial import load_analysis_layer, select_polygon, summarize_region, compare_regions, save_layer, list_layers, delete_layer, create_territorial_object, list_territorial_objects, object_versions, buffer_geometry, corridor_geometry, intersect_geometries, evaluate_object, list_territorial_alerts, monitor_object, monitor_all_objects, monitoring_history, territorial_monitoring_dashboard, create_watchlist, list_watchlists, subscribe_object, create_alert_rule, list_alert_rules, process_watchlist_alerts, list_incidents, operations_center

VERSION="4.7.0"
app=FastAPI(title="Meridian API",version=VERSION)
RUNTIME=Path("runtime/jobs"); RUNTIME.mkdir(parents=True,exist_ok=True)
MIGRATION_RESULT=migrate_all()
PROGRESS:dict[str,dict]={}


def _progress(job_id,percent,stage,message):
    PROGRESS[job_id]={"job_id":job_id,"percent":percent,"stage":stage,"message":message}


def _safe_extract(zpath:Path,dest:Path):
    with zipfile.ZipFile(zpath) as z:
        names=z.namelist()
        supported=(".shp",".gpkg",".geojson",".json",".csv",".xlsx",".xls",".parquet")
        if not any(n.lower().endswith(supported) for n in names): raise ValueError("ZIP has no supported Meridian data source")
        lower={n.lower() for n in names}
        for s in [n for n in names if n.lower().endswith(".shp")]:
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


def _intelligence_feed(summary:dict,domain:str="general"):
    return [{"severity":x["severity"],"type":x["dimension"],"title":x["title"],"detail":x["statement"],"evidence":x["evidence"],"domain":x["domain"]} for x in build_intelligence(summary,domain)]


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
        ts=None
        for tc in ("timestamp","datetime","ACQ_DATE","date","fecha"):
            if tc in row.index:
                try:
                    parsed=pd.to_datetime(row.get(tc),errors="coerce")
                    if pd.notna(parsed): ts=parsed.isoformat()
                except Exception: pass
                if ts: break
        pts.append({"id":int(idx),"lat":float(row.geometry.y),"lon":float(row.geometry.x),
          "level":str(row.get("outlier_level","normal")),"status":str(row.get("record_status","normal")),
          "score":float(row.get("outlier_score",0)),"reason":str(row.get("anomaly_reason","")),
          "lisa_cluster":str(row.get("lisa_cluster","NS")),"lisa_pvalue":float(row.get("local_moran_pvalue",1.0)),"density":float(row.get("spatial_density_rank",0)),
          "timestamp":ts,"spatial":float(row.get("score_spatial",0)),"temporal":float(row.get("score_temporal",0)),
          "probability":float(row.get("score_probability",0)),"consensus":int(row.get("consensus_methods",0))})
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


def _run_job(job_id,inputs,results,probability_feature,variables,noise_level,quadrature_order,weights,input_hash,source_gdf=None,source_meta=None,domain='general'):
    started=time.perf_counter()
    params={"probability_feature":probability_feature,"variables":variables,"noise_level":noise_level,
            "quadrature_order":quadrature_order,"weights":weights}
    save_analysis(job_id,"running",input_hash=input_hash,probability_feature=probability_feature,params=params)
    orchestrator=MeridianOrchestrator(job_id)
    try:
        _progress(job_id,10,"orchestrating","Orchestrator is validating the source")
        gdf=source_gdf.copy() if source_gdf is not None else load_and_append(inputs)
        if not orchestrator.gate_source(gdf)["ok"]: raise ValueError(orchestrator.finalize()["decisions"][-1]["reason"])
        quality=assess_data_quality(gdf)
        qgate=quality_gate(quality)
        orchestrator.record("data_quality",qgate["status"],"continue" if qgate["ok"] else "block",qgate["reason"])
        if not qgate["ok"]: raise ValueError(qgate["reason"])
        autopilot=build_autopilot_plan(gdf,domain)
        if not orchestrator.gate_autopilot(autopilot)["ok"]: raise ValueError(orchestrator.finalize()["decisions"][-1]["reason"])
        if domain in ("","auto",None): domain=autopilot["selected_domain"]
        # Autopilot readiness is enforced by the orchestrator quality gate.
        pack=get_domain_pack(domain)
        contract=semantic_contract(gdf,pack)
        gdf,semantic_aliases=apply_contract(gdf,contract)
        numeric=[c for c in gdf.columns if c!="geometry" and pd.to_numeric(gdf[c],errors="coerce").notna().sum()>=30]
        profile=configure_domain(domain,numeric,probability_feature,variables,weights)
        profile["data_contract"]=contract
        profile["data_quality"]=quality
        profile["semantic_aliases"]=semantic_aliases
        profile["autopilot"]=autopilot
        enrichment_plan=plan_auto_enrichment(gdf,domain,contract)
        profile["enrichment_plan"]=enrichment_plan
        probability_feature=profile["probability_feature"]
        variables=profile["variables"]
        weights=profile["weights"]
        champion=ensure_champion(domain,weights)
        champion_weights=(champion.get("config") or {}).get("weights")
        if champion_weights:
            weights=champion_weights
            profile["weights"]=weights
        profile["algorithm_governance"]={"champion":champion}
        strategy=orchestrator.choose_analysis_strategy(len(gdf),variables,quadrature_order)
        quadrature_order=strategy["quadrature_order"]
        variables=strategy.get("variables",variables)
        profile["orchestration_strategy"]=strategy
        if probability_feature not in numeric: raise ValueError("No usable numeric probability feature was found")

        _progress(job_id,20,"cleaning","Cleaning geometry, duplicates and invalid values")
        gdf,cleaning=basic_clean(gdf)
        _progress(job_id,35,"probability","Fitting probability distributions and Gaussian quadrature")
        scored,summary,fits=detect_outliers(gdf,probability_feature=probability_feature,
            quadrature_order=quadrature_order,component_weights=weights)

        _progress(job_id,68,"validation","Bootstrap, ablation and multi-variable comparison")
        component_data=summary.pop("_components_for_validation",{})
        components=pd.DataFrame(component_data)
        summary["model_validation"]={
            "bootstrap":bootstrap_distribution_stability(gdf[probability_feature],n_boot=strategy["bootstrap_n"],sample_size=1000),
            "ablation":ablation_sensitivity(components,summary["ensemble_weights"])
        }
        projected=gdf.to_crs(gdf.estimate_utm_crs())
        coords=np.c_[projected.geometry.x,projected.geometry.y]
        chosen=[v for v in variables if v in numeric] or [probability_feature]
        comparison=compare_variables(gdf,chosen,coords,quadrature_order=min(24,quadrature_order),max_sample=strategy["comparison_sample"])
        orchestrator.evaluate_distribution(summary)
        orchestrator.evaluate_validation(summary)
        summary["variable_comparison"]=comparison
        summary["domain_profile"]=profile

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
          "random_state":42,"source":source_meta or {"type":"uploaded_shapefile"}}
        summary["reproducibility"]=reproducibility
        _progress(job_id,84,"enrichment","Autopilot is evaluating and retrieving relevant external context")
        enrichment=orchestrator.run_with_retry("enrichment",lambda:execute_auto_enrichment(scored,enrichment_plan,max_weather_points=8),max_attempts=2,fallback=lambda err:{"policy":"context_only","score_mutation":False,"causal_claims":False,"results":[],"status":"degraded","error":str(err),"provenance_note":"External context unavailable after retry; core analysis preserved."})
        summary["external_context"]={"auto_enrichment":enrichment}
        orchestrator.evaluate_enrichment(enrichment)
        summary["orchestration"]=orchestrator.finalize()

        _progress(job_id,88,"exporting","Exporting GeoPackage, CSV, JSON and reports")
        outputs=export_analysis(scored,fits,cleaning,summary,results,noise_level=noise_level)
        write_html_report(results,cleaning,summary,plot,_map_points(scored))
        outputs["report_html"]="REPORT.html"

        elapsed=time.perf_counter()-started
        governance=evaluate_strategy(job_id,domain,summary,elapsed)
        drift=monitor_drift(job_id,domain,summary)
        governance["drift"]=drift
        governance["promotion_gate"]=promotion_gate(domain)
        summary["algorithm_governance"]=governance
        payload={"job_id":job_id,"summary":{"cleaning":cleaning,"detector":summary},
          "outputs":{k:f"/jobs/{job_id}/files/{v}" for k,v in outputs.items()},
          "plot":plot,"map_points":_map_points(scored,max_points=5000),"anomalies":_table_rows(scored,limit=300),
          "numeric_variables":numeric,"comparison":comparison,"elapsed_seconds":elapsed,
          "analysis_status":{"numeric_detected":len(numeric),"variables_compared":len(comparison),
            "map_points":len(_map_points(scored,max_points=5000)),"anomaly_rows":len(_table_rows(scored,limit=300))},
          "domain_profile":profile,"orchestration":summary.get("orchestration"),"algorithm_governance":governance,"intelligence_feed":_intelligence_feed(summary,domain)}
        
        (results/"response.json").write_text(json.dumps(payload,ensure_ascii=False),encoding="utf-8")
        save_analysis(job_id,"complete",input_hash=input_hash,probability_feature=probability_feature,
          rows=len(scored),crs=str(gdf.crs),params=params,summary=summary,outputs=outputs,elapsed_seconds=elapsed)
        territorial_monitoring=monitor_all_objects(results.parent)
        summary["territorial_monitoring"]=territorial_monitoring
        territorial_alerting=process_watchlist_alerts(job_id,territorial_monitoring["results"])
        summary["territorial_alerting"]=territorial_alerting
        append_event(job_id,"territorial_alerting_completed",stage="territorial_alerting",severity="warning" if territorial_alerting["active"] else "info",payload=territorial_alerting)
        append_event(job_id,"territorial_monitoring_completed",stage="territorial_monitoring",payload={"objects":territorial_monitoring["objects"],"changed":territorial_monitoring["changed"]})
        append_event(job_id,"analysis_completed",stage="runtime",payload={"elapsed_seconds":elapsed,"rows":len(scored),"domain":domain})
        _progress(job_id,100,"complete",f"Analysis completed in {elapsed:.1f}s")
    except Exception as e:
        orchestrator.record("runtime","blocked","stop",str(e),False,"manual_review")
        append_event(job_id,"analysis_failed",stage="runtime",severity="error",payload={"error":str(e)})
        save_analysis(job_id,"failed",input_hash=input_hash,probability_feature=probability_feature,params=params,error=str(e))
        _progress(job_id,100,"failed",str(e))


@app.get("/health")
def health():
    return {"status":"ok","service":"Meridian","version":VERSION,"event_store":event_stats(),"migrations":migration_status(),"active_jobs":sum(1 for x in PROGRESS.values() if x.get("percent",100)<100)}















@app.post("/v1/territorial/trajectory/watch/{territory_id}")
def trajectory_watch_api(territory_id:str,baseline_window:int=4):
    hist=feature_history(territory_id)
    try:return watch_territory(hist,baseline_window,True)
    except ValueError as e:raise HTTPException(400,str(e))

@app.get("/v1/territorial/trajectory/watch/{territory_id}/history")
def trajectory_watch_history_api(territory_id:str,limit:int=100):
    return {"territory_id":territory_id,"assessments":assessment_history(territory_id,min(max(limit,1),1000))}

@app.get("/v1/territorial/trajectory/{territory_id}")
def territorial_trajectory_api(territory_id:str,baseline_window:int=4):
    hist=feature_history(territory_id)
    try:return trajectory(hist,baseline_window)
    except ValueError as e:raise HTTPException(400,str(e))

@app.post("/v1/territorial/trajectories/compare")
def territorial_trajectories_compare_api(payload:dict):
    try:
        items=[]
        for item in payload["territories"]:
            snaps=item.get("snapshots") or feature_history(str(item["territory_id"]))
            items.append({"snapshots":snaps,"baseline_window":item.get("baseline_window",4)})
        return compare_trajectories(items)
    except (ValueError,KeyError) as e:raise HTTPException(400,str(e))

@app.post("/v1/territorial/trajectories/cluster")
def territorial_trajectories_cluster_api(payload:dict):
    try:
        items=[]
        for item in payload["territories"]:
            snaps=item.get("snapshots") or feature_history(str(item["territory_id"]))
            items.append({"snapshots":snaps,"baseline_window":item.get("baseline_window",4)})
        return cluster_trajectories(items,int(payload.get("n_clusters",3)))
    except (ValueError,KeyError) as e:raise HTTPException(400,str(e))

@app.post("/v1/spatial/hotspots/gi-star")
def spatial_gistar_api(payload:dict):
    try:return getis_ord_gi_star(payload["points"],payload.get("value_key","value"),int(payload.get("k",8)),int(payload.get("permutations",199)))
    except (ValueError,KeyError) as e:raise HTTPException(400,str(e))

@app.post("/v1/spatial/bivariate-moran")
def spatial_bivariate_moran_api(payload:dict):
    try:return bivariate_moran(payload["points"],payload["x_key"],payload["y_key"],int(payload.get("k",8)),int(payload.get("permutations",199)))
    except (ValueError,KeyError) as e:raise HTTPException(400,str(e))

@app.post("/v1/spatial/association-matrix")
def spatial_association_matrix_api(payload:dict):
    try:return spatial_association_matrix(payload["points"],payload["variables"],int(payload.get("k",8)),int(payload.get("permutations",99)))
    except (ValueError,KeyError) as e:raise HTTPException(400,str(e))

@app.post("/v1/spatial/temporal-compare")
def spatial_temporal_compare_api(payload:dict):
    try:return temporal_spatial_comparison(payload["t0"],payload["t1"],payload.get("value_key","value"),payload.get("method","gi_star"),int(payload.get("k",8)),int(payload.get("permutations",199)))
    except (ValueError,KeyError) as e:raise HTTPException(400,str(e))

@app.post("/v1/territorial/fusion")
def territorial_fusion_api(payload:dict):
    try:return fuse_features(payload)
    except ValueError as e:raise HTTPException(400,str(e))

@app.get("/v1/territorial/profile/{territory_id}")
def territorial_profile_api(territory_id:str,window:int=8):
    return temporal_profile(territory_id,window)

@app.get("/v1/territorial/features/{territory_id}")
def territorial_features_api(territory_id:str):
    return {"territory_id":territory_id,"snapshots":feature_history(territory_id)}

@app.post("/v1/territorial/land-use/divergence")
def land_use_divergence_api(payload:dict):
    try:return land_use_divergence(payload,True)
    except ValueError as e:raise HTTPException(400,str(e))

@app.post("/v1/spatial/operations")
def spatial_operations_api(payload:dict):
    try:return spatial_operation(payload["left"],payload.get("right") or [],payload.get("operation","intersects"),payload.get("left_crs","EPSG:4326"),payload.get("right_crs","EPSG:4326"),payload.get("distance_m"))
    except (ValueError,KeyError) as e:raise HTTPException(400,str(e))

@app.get("/v1/geodata")
def geodata_catalog(limit:int=200):
    return {"datasets":list_datasets(limit)}

@app.post("/v1/geodata/register")
async def geodata_register(files:List[UploadFile]=File(...), metadata_json:str=Form("{}")):
    job_dir=RUNTIME/("_geodata_"+uuid.uuid4().hex[:10]);job_dir.mkdir(parents=True,exist_ok=True)
    try:
        paths=_store_uploads(files,job_dir);gdf=load_and_append(paths)
        metadata=json.loads(metadata_json or "{}")
        return register_dataset(gdf,metadata)
    except (ValueError,json.JSONDecodeError) as e:raise HTTPException(422,str(e))
    finally:shutil.rmtree(job_dir,ignore_errors=True)

@app.get("/v1/geodata/{dataset_id}/lineage")
def geodata_lineage(dataset_id:str):
    return {"dataset_id":dataset_id,"lineage":lineage(dataset_id)}

@app.post("/v1/spatial/events")
def spatial_event_create(payload:dict):
    try:return create_event(payload)
    except (ValueError,KeyError) as e:raise HTTPException(400,str(e))

@app.post("/v1/territorial/snapshots")
def territorial_snapshot_create(payload:dict):
    try:return territorial_snapshot(payload)
    except ValueError as e:raise HTTPException(400,str(e))

@app.get("/v1/territorial/snapshots/{territory_id}")
def territorial_snapshot_history(territory_id:str):
    return {"territory_id":territory_id,"snapshots":snapshots(territory_id)}

@app.post("/v1/territorial/indicators")
def territorial_indicator_api(payload:dict):
    return territorial_indicators(payload)

@app.post("/v1/territorial/correlation")
def territorial_correlation_api(payload:dict):
    try:
        out=correlation(payload["x"],payload["y"],payload.get("method","pearson"))
        out.update({"period":payload.get("period"),"spatial_unit":payload.get("spatial_unit")})
        return out
    except (ValueError,KeyError) as e:raise HTTPException(400,str(e))

@app.post("/v1/territorial/distance")
def territorial_distance_api(payload:dict):
    try:return distance_model(payload["a"],payload["b"],payload.get("reference"))
    except (ValueError,KeyError) as e:raise HTTPException(400,str(e))

@app.post("/v1/territorial/hotspots")
def territorial_hotspots_api(payload:dict):
    try:return hotspot(payload["points"],payload.get("value_key","value"),payload.get("method","lisa"),payload.get("bandwidth"))
    except (ValueError,KeyError) as e:raise HTTPException(400,str(e))

@app.post("/v1/territorial/network/build")
def territorial_network_build(payload:dict):
    try:return build_territorial_graph(payload["nodes"],payload.get("edges"),payload.get("proximity_m"))
    except (ValueError,KeyError) as e:raise HTTPException(400,str(e))

@app.post("/v1/territorial/network/shortest-path")
def territorial_network_shortest_path(payload:dict):
    try:return shortest_path(payload["graph"],str(payload["source"]),str(payload["target"]))
    except (ValueError,KeyError) as e:raise HTTPException(400,str(e))

@app.post("/v1/territorial/corridors/detect")
def territorial_corridors_api(payload:dict):
    try:
        # Backward-compatible geographic threshold mode remains available.
        if "distance_threshold" in payload and "proximity_m" not in payload:
            return {"corridors":detect_corridors(payload["points"],float(payload["distance_threshold"]),payload.get("period"),payload.get("data_sources")),
                    "methodology":{"legacy":True,"causal_claims":False}}
        result=detect_network_corridors(
            payload.get("nodes") or payload["points"],payload.get("edges"),
            float(payload.get("proximity_m",5000)),payload.get("bandwidth_m"),int(payload.get("min_nodes",3)),
            float(payload.get("temporal_weight",.15)),float(payload.get("density_weight",.30)),
            float(payload.get("connectivity_weight",.35)),float(payload.get("continuity_weight",.20)),
            int(payload.get("permutations",99)))
        if payload.get("persist",True):
            result["persisted"]=save_corridors(result,payload.get("period"),payload.get("data_sources"))
        return result
    except (ValueError,KeyError) as e:raise HTTPException(400,str(e))


@app.get("/v1/territorial/corridors")
def territorial_corridor_list(limit:int=100):
    return {"corridors":list_corridors(limit)}

@app.get("/v1/territorial/corridors/{corridor_id}")
def territorial_corridor_detail(corridor_id:str):
    item=get_corridor(corridor_id)
    if not item:raise HTTPException(404,"Corridor not found")
    return item

@app.post("/v1/earth-observation/process")
def earth_observation_process(payload:dict):
    try:
        return process_scene(payload["scene"],payload["geometry"],payload.get("indices",["NDVI"]),int(payload.get("max_pixels",1000000)))
    except KeyError as e:raise HTTPException(400,f"Missing {e}")
    except ValueError as e:raise HTTPException(400,str(e))
    except Exception as e:raise HTTPException(502,str(e))

@app.post("/v1/earth-observation/territorial/{object_key}/intelligence")
def earth_observation_territorial_intelligence(object_key:str,payload:dict):
    obj=next((x for x in list_territorial_objects(False,500) if x["object_key"]==object_key),None)
    if not obj:raise HTTPException(404,"Territorial object not found")
    try:
        indices=payload.get("indices",["NDVI","NDMI","NBR","NDWI"])
        current=process_scene(payload["scene"],obj["geometry"],indices,int(payload.get("max_pixels",1000000)))
        response={"object_key":object_key,"current":current}
        baseline=payload.get("baseline")
        if baseline:
            shared=[x for x in indices if x in current["feature_vector"] and x in baseline]
            if len(shared)>=2:
                response["euclidean_change"]=compare_to_baseline(current["feature_vector"],baseline,shared)
        return response
    except KeyError as e:raise HTTPException(400,f"Missing {e}")
    except ValueError as e:raise HTTPException(400,str(e))
    except Exception as e:raise HTTPException(502,str(e))

@app.post("/v1/intelligence/euclidean")
def euclidean_model(payload:dict):
    try:
        rows=payload.get("rows") or []
        features=payload.get("features") or []
        if not rows:raise ValueError("rows are required")
        frame=pd.DataFrame(rows)
        spatial=payload.get("spatial_columns")
        spatial_columns=tuple(spatial) if spatial else None
        scored,summary=euclidean_intelligence(
            frame,features,spatial_columns,
            float(payload.get("spatial_weight",.25)),
            float(payload.get("feature_weight",.75)),
        )
        return {"summary":summary,"rows":scored.replace({np.nan:None}).to_dict(orient="records")}
    except (ValueError,KeyError) as e:raise HTTPException(400,str(e))

@app.post("/v1/intelligence/euclidean/baseline")
def euclidean_baseline_compare(payload:dict):
    try:return compare_to_baseline(payload["current"],payload["baseline"],payload["features"])
    except (ValueError,KeyError) as e:raise HTTPException(400,str(e))

@app.get("/v1/earth-observation/catalog")
def earth_observation_catalog():
    return {"sensors":sensor_catalog(),"indices":index_catalog()}

@app.post("/v1/earth-observation/search")
def earth_observation_search(payload:dict):
    try:
        sensor=payload.get("sensor","sentinel-2-l2a")
        start=payload["start_date"]; end=payload["end_date"]
        cloud=payload.get("cloud_cover_max",30); limit=int(payload.get("limit",20))
        if payload.get("geometry"):
            return search_geometry(sensor,payload["geometry"],start,end,cloud,limit)
        return search_scenes(sensor,payload["bbox"],start,end,cloud,limit)
    except KeyError as e:raise HTTPException(400,f"Missing {e}")
    except ValueError as e:raise HTTPException(400,str(e))
    except Exception as e:raise HTTPException(502,str(e))

@app.post("/v1/earth-observation/index-plan")
def earth_observation_index_plan(payload:dict):
    try:return build_index_plan(payload.get("sensor","sentinel-2-l2a"),payload.get("indices",["NDVI"]))
    except ValueError as e:raise HTTPException(400,str(e))

@app.post("/v1/earth-observation/territorial/{object_key}/search")
def earth_observation_territorial_search(object_key:str,payload:dict):
    obj=next((x for x in list_territorial_objects(False,500) if x["object_key"]==object_key),None)
    if not obj:raise HTTPException(404,"Territorial object not found")
    try:
        return search_geometry(payload.get("sensor","sentinel-2-l2a"),obj["geometry"],payload["start_date"],payload["end_date"],payload.get("cloud_cover_max",30),int(payload.get("limit",20)))
    except KeyError as e:raise HTTPException(400,f"Missing {e}")
    except ValueError as e:raise HTTPException(400,str(e))
    except Exception as e:raise HTTPException(502,str(e))

@app.post("/v1/data-quality")
async def data_quality(files:List[UploadFile]=File(...)):
    job_dir=RUNTIME/("_quality_"+uuid.uuid4().hex[:10]); job_dir.mkdir(parents=True,exist_ok=True)
    try:
        paths=_store_uploads(files,job_dir)
        gdf=load_and_append(paths)
        report=assess_data_quality(gdf)
        return {"report":report,"gate":quality_gate(report)}
    except ValueError as e:raise HTTPException(422,str(e))
    finally:shutil.rmtree(job_dir,ignore_errors=True)

@app.get("/v1/system/migrations")
def system_migrations():
    return migration_status()

@app.post("/v1/system/migrations")
def system_run_migrations():
    return migrate_all()

@app.get("/v1/operations")
def territorial_operations_center():
    return operations_center()

@app.get("/v1/watchlists")
def territorial_watchlists():
    return {"watchlists":list_watchlists()}

@app.post("/v1/watchlists")
def territorial_create_watchlist(payload:dict):
    if not payload.get("name"):raise HTTPException(400,"name is required")
    try:return create_watchlist(payload["name"],payload.get("domain"))
    except Exception as e:raise HTTPException(409,str(e))

@app.post("/v1/watchlists/{watchlist_id}/objects/{object_key}")
def territorial_subscribe(watchlist_id:int,object_key:str):
    try:return subscribe_object(watchlist_id,object_key)
    except KeyError:raise HTTPException(404,"Territorial object not found")

@app.get("/v1/watchlists/{watchlist_id}/rules")
def territorial_rules(watchlist_id:int):
    return {"rules":list_alert_rules(watchlist_id)}

@app.post("/v1/watchlists/{watchlist_id}/rules")
def territorial_create_rule(watchlist_id:int,payload:dict):
    try:return create_alert_rule(watchlist_id,payload["name"],payload["metric"],payload["operator"],float(payload["threshold"]),payload["severity"])
    except KeyError as e:raise HTTPException(400,f"Missing {e}")
    except ValueError as e:raise HTTPException(400,str(e))


@app.get("/v1/incidents/{incident_id}")
def incident_case(incident_id:int):
    try:return get_incident_case(incident_id)
    except KeyError:raise HTTPException(404,"Incident not found")

@app.patch("/v1/incidents/{incident_id}")
def incident_update(incident_id:int,payload:dict):
    try:return update_incident_management(incident_id,payload.get("owner"),payload.get("priority"),payload.get("acknowledged"),payload.get("investigation_status"),payload.get("resolution_code"),payload.get("resolution_note"))
    except KeyError:raise HTTPException(404,"Incident not found")
    except ValueError as e:raise HTTPException(400,str(e))

@app.post("/v1/incidents/{incident_id}/notes")
def incident_add_note(incident_id:int,payload:dict):
    try:return add_incident_note(incident_id,payload.get("note",""),payload.get("author"))
    except KeyError:raise HTTPException(404,"Incident not found")
    except ValueError as e:raise HTTPException(400,str(e))

@app.get("/v1/incidents/{incident_id}/brief")
def incident_brief(incident_id:int):
    try:return build_intelligence_brief(incident_id)
    except KeyError:raise HTTPException(404,"Incident not found")

@app.get("/v1/operations/cases")
def operations_incident_cases(status:str="",limit:int=200):
    return {"cases":operations_cases(status or None,limit)}

@app.get("/v1/incidents")
def territorial_incidents(status:str="",limit:int=200):
    return {"incidents":list_incidents(status or None,limit)}

@app.get("/v1/territorial/monitoring/{object_key}")
def territorial_monitoring(object_key:str):
    try:return territorial_monitoring_dashboard(object_key)
    except KeyError:raise HTTPException(404,"Territorial object not found")

@app.get("/v1/territorial/monitoring/{object_key}/history")
def territorial_monitoring_history(object_key:str,limit:int=100):
    return {"object_key":object_key,"history":monitoring_history(object_key,limit)}

@app.post("/v1/territorial/monitoring/{object_key}/run/{job_id}")
def territorial_monitoring_run(object_key:str,job_id:str):
    objs=[x for x in list_territorial_objects(True,500) if x["object_key"]==object_key]
    if not objs:raise HTTPException(404,"Territorial object not found")
    try:
        result=monitor_object(RUNTIME/job_id,objs[0])
        append_event(job_id,"territorial_object_monitored",stage="territorial_monitoring",severity="warning" if result["status"]=="changed" else "info",payload={"object_key":object_key,"status":result["status"],"change":result["change"]})
        return result
    except FileNotFoundError as e:raise HTTPException(404,str(e))

@app.post("/v1/territorial/monitoring/run/{job_id}")
def territorial_monitoring_run_all(job_id:str):
    try:
        result=monitor_all_objects(RUNTIME/job_id)
        append_event(job_id,"territorial_monitoring_completed",stage="territorial_monitoring",payload={"objects":result["objects"],"changed":result["changed"]})
        return result
    except FileNotFoundError as e:raise HTTPException(404,str(e))

@app.get("/v1/territorial/objects")
def territorial_objects(active_only:bool=True,limit:int=200):
    return {"objects":list_territorial_objects(active_only,limit)}

@app.post("/v1/territorial/objects")
def territorial_create_object(payload:dict):
    for k in ("name","object_type","geometry"):
        if not payload.get(k):raise HTTPException(400,f"{k} is required")
    try:return create_territorial_object(payload["name"],payload["object_type"],payload["geometry"],payload.get("job_id"),payload.get("metadata"),payload.get("object_key"))
    except ValueError as e:raise HTTPException(400,str(e))

@app.get("/v1/territorial/objects/{object_key}/versions")
def territorial_object_versions(object_key:str):
    return {"versions":object_versions(object_key)}

@app.post("/v1/territorial/objects/{object_key}/evaluate/{job_id}")
def territorial_evaluate_object(object_key:str,job_id:str):
    objs=[x for x in list_territorial_objects(True,500) if x["object_key"]==object_key]
    if not objs:raise HTTPException(404,"Territorial object not found")
    try:return evaluate_object(RUNTIME/job_id,objs[0])
    except FileNotFoundError as e:raise HTTPException(404,str(e))

@app.post("/v1/territorial/geometry/buffer")
def territorial_buffer(payload:dict):
    try:return {"geometry":buffer_geometry(payload["geometry"],float(payload["distance_m"]))}
    except (KeyError,ValueError) as e:raise HTTPException(400,str(e))

@app.post("/v1/territorial/geometry/corridor")
def territorial_corridor(payload:dict):
    try:return {"geometry":corridor_geometry(payload["coordinates"],float(payload["distance_m"]))}
    except (KeyError,ValueError) as e:raise HTTPException(400,str(e))

@app.post("/v1/territorial/geometry/intersection")
def territorial_intersection(payload:dict):
    try:return {"geometry":intersect_geometries(payload["a"],payload["b"])}
    except (KeyError,ValueError) as e:raise HTTPException(400,str(e))

@app.get("/v1/territorial/alerts")
def territorial_alerts(object_key:str="",limit:int=200):
    return {"alerts":list_territorial_alerts(object_key or None,limit)}

@app.post("/v1/territorial/query/{job_id}")
def territorial_query(job_id:str,geometry:dict):
    try:
        gdf=load_analysis_layer(RUNTIME/job_id)
        selected=select_polygon(gdf,geometry)
        return {"job_id":job_id,"summary":summarize_region(selected),"selected_ids":[int(x) for x in selected.index[:5000]],"selection_truncated":len(selected)>5000}
    except FileNotFoundError as e:raise HTTPException(404,str(e))
    except ValueError as e:raise HTTPException(400,str(e))

@app.post("/v1/territorial/compare/{job_id}")
def territorial_compare(job_id:str,payload:dict):
    if "region_a" not in payload or "region_b" not in payload:raise HTTPException(400,"region_a and region_b are required")
    try:
        gdf=load_analysis_layer(RUNTIME/job_id)
        a=select_polygon(gdf,payload["region_a"]); b=select_polygon(gdf,payload["region_b"])
        return {"job_id":job_id,**compare_regions(a,b)}
    except FileNotFoundError as e:raise HTTPException(404,str(e))
    except ValueError as e:raise HTTPException(400,str(e))

@app.get("/v1/territorial/layers")
def territorial_layers(limit:int=100):
    return {"layers":list_layers(limit)}

@app.post("/v1/territorial/layers")
def territorial_save_layer(payload:dict):
    if not payload.get("name") or not payload.get("geometry"):raise HTTPException(400,"name and geometry are required")
    try:return save_layer(payload["name"],payload["geometry"],payload.get("job_id"),payload.get("metadata"))
    except ValueError as e:raise HTTPException(400,str(e))

@app.delete("/v1/territorial/layers/{layer_id}")
def territorial_delete_layer(layer_id:int):
    if not delete_layer(layer_id):raise HTTPException(404,"Layer not found")
    return {"deleted":True,"id":layer_id}

@app.get("/v1/events")
def events(job_id:str="",limit:int=200):
    return {"events":list_events(job_id or None,limit)}

@app.get("/v1/health/system")
def system_health():
    history=list_analyses(100)
    running=sum(1 for x in history if x.get("status")=="running")
    failed=sum(1 for x in history if x.get("status")=="failed")
    complete=sum(1 for x in history if x.get("status")=="complete")
    return {"service":"Meridian","version":VERSION,"jobs":{"running":running,"failed":failed,"complete":complete},"events":event_stats(),"integrations":integration_status()}

@app.get("/v1/domains")
def domains(): return {"engine":"Meridian","version":VERSION,"domains":list_domain_packs()}

@app.get("/v1/domains/{domain_id}")
def domain(domain_id:str):
    try:return get_domain_pack(domain_id)
    except KeyError:raise HTTPException(404,"Unknown Meridian domain pack")

@app.post("/v1/enrichment/plan")
def enrichment_plan(domain:str="general", columns:str=""):
    cols=[x.strip() for x in columns.split(",") if x.strip()]
    preview=pd.DataFrame({x:pd.Series([1.0]*30) for x in cols})
    if "latitude" not in preview: preview["latitude"]=pd.Series([0.0]*30)
    if "longitude" not in preview: preview["longitude"]=pd.Series([0.0]*30)
    import geopandas as gpd
    gdf=gpd.GeoDataFrame(preview,geometry=gpd.points_from_xy(preview["longitude"],preview["latitude"]),crs="EPSG:4326")
    try:return plan_auto_enrichment(gdf,domain)
    except KeyError:raise HTTPException(404,"Unknown Meridian domain pack")

@app.post("/v1/autopilot/plan")
def autopilot_plan(domain:str="auto", columns:str=""):
    cols=[x.strip() for x in columns.split(",") if x.strip()]
    preview=pd.DataFrame({x:pd.Series([1.0]*30) for x in cols})
    try:return build_autopilot_plan(preview,domain)
    except KeyError:raise HTTPException(404,"Unknown Meridian domain pack")

@app.post("/v1/semantic/contract")
def semantic_contract_endpoint(domain_id:str="general", columns:str=""):
    """Preview a semantic contract from column names when full source inspection is not required."""
    try:pack=get_domain_pack(domain_id)
    except KeyError:raise HTTPException(404,"Unknown Meridian domain pack")
    cols=[x.strip() for x in columns.split(",") if x.strip()]
    preview=pd.DataFrame({x:pd.Series([1.0]*30) for x in cols})
    return semantic_contract(preview,pack)

@app.post("/v1/domains/{domain_id}/configure")
def configure_domain_endpoint(domain_id:str, columns:str="", probability_feature:str=""):
    numeric=[x.strip() for x in columns.split(",") if x.strip()]
    try:return configure_domain(domain_id,numeric,probability_feature or None)
    except KeyError:raise HTTPException(404,"Unknown Meridian domain pack")

@app.get("/v1/governance/dashboard/{domain}")
def governance_domain_dashboard(domain:str):
    return governance_dashboard(domain)

@app.get("/v1/governance/experiments")
def governance_experiments(domain:str="",limit:int=100):
    return {"experiments":experiment_history(domain or None,limit)}

@app.get("/v1/governance/drift/{domain}")
def governance_drift(domain:str):
    history=experiment_history(domain,20)
    return {"domain":domain,"experiments":len(history),"note":"Drift is evaluated during completed analyses and stored in governance/Event Store."}

@app.get("/v1/governance/promotion-gate/{domain}")
def governance_promotion_gate(domain:str,min_runs:int=5):
    return promotion_gate(domain,min_runs)

@app.get("/v1/governance/strategies")
def governance_strategies(domain:str=""):
    return {"strategies":list_strategies(domain or None)}

@app.get("/v1/governance/champion/{domain}")
def governance_champion(domain:str):
    return get_champion(domain) or ensure_champion(domain)

@app.post("/v1/governance/challenger/{domain}")
def governance_challenger(domain:str,name:str,version:str,weights_json:str=""):
    try:weights=json.loads(weights_json) if weights_json else {}
    except Exception:raise HTTPException(400,"weights_json must be valid JSON")
    return register_challenger(domain,name,version,{"name":name,"version":version,"weights":weights})

@app.post("/v1/governance/promote/{domain}/{strategy_id}")
def governance_promote(domain:str,strategy_id:int):
    try:return promote(domain,strategy_id)
    except KeyError:raise HTTPException(404,"Strategy not found")

@app.post("/v1/governance/rollback/{domain}")
def governance_rollback(domain:str):
    try:return rollback(domain)
    except ValueError as e:raise HTTPException(409,str(e))

@app.get("/v1/orchestrator/policy")
def orchestrator_policy():
    return {"engine":"Meridian Orchestrator","version":VERSION,"principles":["deterministic decisions","hard quality gates","adaptive runtime strategy","external context is optional","no silent score mutation","no causal claims from correlation"],"large_dataset_threshold":150000,"balanced_threshold":60000,"max_multivariate_variables":8}

@app.get("/v1/capabilities")
def capabilities():
    return {"engine":"Meridian","version":VERSION,"analysis":["socio_ecological_spatialization","territorial_snapshots","geodata_catalog","data_lineage","spatial_events","territorial_indicators","gini_concentration","spatial_correlation","territorial_distance_model","corridor_candidates","territorial_network_graph","network_centrality","shortest_path","validated_corridor_detection","trajectory_watch","trajectory_assessment_history","territorial_trajectory","trajectory_velocity","trajectory_acceleration","trajectory_change_points","trajectory_clustering","getis_ord_gi_star","bivariate_moran","spatial_association_matrix","spatial_temporal_comparison","territorial_fusion","temporal_feature_store","land_use_divergence","territorial_profile","spatial_join_service","overlay","nearest_neighbor","metric_buffer","raster_processing","remote_cog_window","cloud_masking","spectral_indices","zonal_statistics","eo_territorial_intelligence","euclidean_intelligence","euclidean_baseline_distance","earth_observation_stac","sentinel_2","landsat","modis","spectral_index_planning","territorial_scene_search","data_quality_engine","geometry_quality","coordinate_validation","quality_gate","permutation_lisa","significant_hotspots","spatial_density","spatial_outliers","incident_management","territorial_intelligence_briefs","incident_ownership","acknowledgement","investigation_timeline","operations_center","territorial_watchlists","alert_rules","alert_deduplication","alert_escalation","alert_resolution","territorial_monitoring","territorial_baselines","territorial_change_detection","automatic_object_linking","territorial_objects","versioned_geometries","buffers","corridors","layer_intersections","territorial_findings","territorial_alerts","polygon_queries","statistical_region_compare","persistent_territorial_layers","territorial_workspace","region_selection","region_compare","timeline_filters","alert_rules","experiment_history","drift_monitoring","promotion_gates","algorithm_governance","champion_challenger","rollback","event_store","health_monitoring","bounded_retry","fallback_recovery","orchestrator","quality_gates","adaptive_strategy","autopilot","auto_enrichment","provenance","domain_inference","semantic_mapping","data_contract","probability","spatial","temporal","compare","explain"],"delivery":["workspace","api","exports"],"ingestion":["zip_shapefile","shapefile","geopackage","geojson","json","csv","excel","parquet"],"domain_packs":[x["id"] for x in list_domain_packs()]}

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


@app.post("/live/firms/analyze")
def live_firms_analyze(
    west:float=Form(...), south:float=Form(...), east:float=Form(...), north:float=Form(...),
    days:int=Form(1), source:str=Form("VIIRS_NOAA21_NRT"), date:str=Form(""),
    probability_feature:str=Form("FRP"), variables:str=Form("FRP,BRIGHTNESS,BRIGHT_T31,SCAN,TRACK"),
    noise_level:int=Form(99), quadrature_order:int=Form(32), weights_json:str=Form("")
):
    if noise_level not in (90,95,99): raise HTTPException(400,"noise_level must be 90, 95 or 99")
    try: weights=json.loads(weights_json) if weights_json else None
    except json.JSONDecodeError: raise HTTPException(400,"Invalid weights_json")
    try:
        gdf=fetch_firms_area(west,south,east,north,days,source,date or None)
    except Exception as e:
        raise HTTPException(502,str(e))
    if len(gdf)<30:
        raise HTTPException(422,f"NASA FIRMS returned only {len(gdf)} observations; at least 30 are required for this analysis.")
    selected=[x.strip() for x in variables.split(",") if x.strip()]
    job_id=uuid.uuid4().hex
    job=RUNTIME/job_id; inputs=job/"input"; results=job/"results"
    inputs.mkdir(parents=True); results.mkdir()
    signature=json.dumps({"provider":"NASA FIRMS","bbox":[west,south,east,north],"days":days,"source":source,"date":date},sort_keys=True).encode()
    input_hash=hashlib.sha256(signature).hexdigest()
    source_meta={"type":"live_api","provider":"NASA FIRMS","product":source,"bbox":[west,south,east,north],"days":days,"date":date or None}
    _progress(job_id,2,"queued",f"NASA FIRMS loaded {len(gdf)} live observations")
    threading.Thread(target=_run_job,args=(job_id,inputs,results,probability_feature,selected,noise_level,quadrature_order,weights,input_hash,gdf,source_meta),daemon=True).start()
    return {"job_id":job_id,"status":"queued","rows":len(gdf),"source":source,"progress_url":f"/jobs/{job_id}/progress"}


def _store_upload(data:bytes,filename:str,dest:Path,index:int):
    name=Path(filename or f"source_{index}").name
    ext=Path(name).suffix.lower()
    if ext==".zip":
        zp=dest/f"u{index}.zip";zp.write_bytes(data);folder=dest/f"s{index}";folder.mkdir();_safe_extract(zp,folder);return
    if ext not in {".shp",".gpkg",".geojson",".json",".csv",".xlsx",".xls",".parquet"}:
        raise ValueError(f"Unsupported format: {ext or 'unknown'}")
    (dest/f"{index}_{name}").write_bytes(data)

@app.post("/inspect")
async def inspect(files:List[UploadFile]=File(...), domain:str=Form("auto")):
    job=RUNTIME/("_inspect_"+uuid.uuid4().hex); inputs=job/"input"; inputs.mkdir(parents=True)
    try:
        for i,f in enumerate(files):
            _store_upload(await f.read(),f.filename or "",inputs,i)
        gdf=load_and_append(inputs)
        numeric=[c for c in gdf.columns if c!="geometry" and pd.to_numeric(gdf[c],errors="coerce").notna().sum()>=30]
        plan=build_autopilot_plan(gdf,domain)
        contract=plan["data_contract"]
        return {"rows":len(gdf),"crs":str(gdf.crs) if gdf.crs else None,"numeric_variables":numeric,
                "data_contract":contract,"autopilot":plan,"valid":bool(gdf.crs and len(gdf)>=30 and plan["ready"])}
    except Exception as e: raise HTTPException(422,str(e))
    finally: shutil.rmtree(job,ignore_errors=True)


@app.post("/analyze")
async def analyze(files:List[UploadFile]=File(...), probability_feature:str=Form("FRP"),
    variables:str=Form(""), domain:str=Form("auto"), noise_level:int=Form(99), quadrature_order:int=Form(48),
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
            data=await f.read()
            original=job/f"upload_{i}_{Path(f.filename or 'source').name}";original.write_bytes(data);uploads.append(original)
            _store_upload(data,f.filename or "",inputs,i)
    except Exception as e:
        shutil.rmtree(job,ignore_errors=True)
        raise HTTPException(422,str(e))
    input_hash=_hash_files(uploads)
    _progress(job_id,2,"queued","Analysis queued")
    threading.Thread(target=_run_job,args=(job_id,inputs,results,probability_feature,selected,noise_level,quadrature_order,weights,input_hash,None,None,domain),daemon=True).start()
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
