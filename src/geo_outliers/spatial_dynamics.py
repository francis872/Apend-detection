from __future__ import annotations
import math
from datetime import datetime
import numpy as np
from sklearn.cluster import KMeans

def _time(ts):
    try:return datetime.fromisoformat(str(ts).replace("Z","+00:00")).timestamp()/86400.0
    except Exception:raise ValueError(f"Invalid timestamp: {ts}")

def _series(snapshots):
    if len(snapshots)<2:raise ValueError("At least two territorial snapshots are required")
    s=sorted(snapshots,key=lambda x:_time(x["timestamp"]));keys=sorted(set.intersection(*[set(x.get("vector",{})) for x in s]))
    if not keys:raise ValueError("Snapshots have no common numeric features")
    X=np.asarray([[float(x["vector"][k]) for k in keys] for x in s],float)
    if not np.isfinite(X).all():raise ValueError("Trajectory features must be finite")
    t=np.asarray([_time(x["timestamp"]) for x in s],float);dt=np.diff(t)
    if np.any(dt<=0):raise ValueError("Snapshot timestamps must be unique")
    return s,keys,X,t,dt

def trajectory(snapshots,baseline_window=4):
    s,keys,X,t,dt=_series(snapshots)
    scale=np.nanstd(X,axis=0,ddof=1);scale=np.where(np.isfinite(scale)&(scale>1e-9),scale,1.0);Z=(X-X[0])/scale
    velocity=np.diff(Z,axis=0)/dt[:,None];speed=np.linalg.norm(velocity,axis=1)
    acceleration=np.diff(velocity,axis=0)/((dt[1:]+dt[:-1])[:,None]/2) if len(velocity)>1 else np.empty((0,len(keys)))
    accel_mag=np.linalg.norm(acceleration,axis=1) if len(acceleration) else np.array([])
    path=float(np.sum(np.linalg.norm(np.diff(Z,axis=0),axis=1)));displacement=float(np.linalg.norm(Z[-1]-Z[0]))
    straightness=float(displacement/path) if path>1e-12 else 1.0
    w=min(max(2,int(baseline_window)),len(Z)-1);base=Z[:w];center=np.median(base,axis=0);mad=np.median(np.abs(base-center),axis=0);robust=np.maximum(1.4826*mad,.25)
    departure=np.linalg.norm((Z-center)/robust,axis=1);change_points=[]
    if len(speed)>=3:
        med=float(np.median(speed));m=float(np.median(np.abs(speed-med)));thr=med+3.5*max(1.4826*m,1e-9)
        change_points=[{"timestamp":s[i+1]["timestamp"],"speed":float(v),"threshold":float(thr)} for i,v in enumerate(speed) if v>thr]
    return {"territory_id":s[-1].get("territory_id"),"features":keys,"timestamps":[x["timestamp"] for x in s],
      "speed":[float(x) for x in speed],"acceleration":[float(x) for x in accel_mag],"departure_from_baseline":[float(x) for x in departure],
      "path_length":path,"net_displacement":displacement,"trajectory_straightness":straightness,"change_points":change_points,
      "current_departure":float(departure[-1]),"current_speed":float(speed[-1]),
      "methodology":{"standardization":"per-feature trajectory scale","baseline":"median/MAD of initial window","change_point":"robust speed threshold 3.5 MAD","distance":"standardized Euclidean","causal_claims":False,"scores_are_not_probabilities":True}}

def compare_trajectories(territories):
    results=[trajectory(x["snapshots"],x.get("baseline_window",4)) for x in territories]
    if len(results)<2:raise ValueError("At least two territories required")
    m=min(len(r["speed"]) for r in results);A=np.asarray([r["speed"][-m:] for r in results])
    D=np.linalg.norm(A[:,None,:]-A[None,:,:],axis=2)
    return {"territories":[r["territory_id"] for r in results],"speed_distance_matrix":D.tolist(),"trajectories":results,"distance_is_not_probability":True}

def cluster_trajectories(territories,n_clusters=3):
    results=[trajectory(x["snapshots"],x.get("baseline_window",4)) for x in territories]
    if len(results)<2:raise ValueError("At least two territories required")
    k=min(max(2,int(n_clusters)),len(results))
    F=np.asarray([[r["current_speed"],r["current_departure"],r["path_length"],r["trajectory_straightness"]] for r in results],float)
    sd=F.std(axis=0);Z=(F-F.mean(axis=0))/np.where(sd>1e-9,sd,1)
    labels=KMeans(n_clusters=k,random_state=42,n_init=10).fit_predict(Z)
    return {"clusters":[{"territory_id":r["territory_id"],"cluster":int(labels[i]),"features":F[i].tolist()} for i,r in enumerate(results)],
      "feature_names":["current_speed","current_departure","path_length","trajectory_straightness"],"n_clusters":k,
      "methodology":{"algorithm":"KMeans on standardized trajectory descriptors","cluster_is_descriptive_not_causal":True}}
