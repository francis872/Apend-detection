from __future__ import annotations
import math, uuid
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString, mapping
from shapely.ops import unary_union
from sklearn.neighbors import KernelDensity, NearestNeighbors

def _frames(nodes):
    rows=[{**n,"node_id":str(n.get("id",i)),"geometry":Point(float(n["longitude"]),float(n["latitude"]))} for i,n in enumerate(nodes)]
    geo=gpd.GeoDataFrame(rows,geometry="geometry",crs=4326)
    return geo,geo.to_crs(geo.estimate_utm_crs()) if len(geo)>1 else geo

def build_territorial_graph(nodes,edges=None,proximity_m=None):
    if not nodes: raise ValueError("nodes are required")
    geo,proj=_frames(nodes);G=nx.Graph()
    for _,r in geo.iterrows():G.add_node(str(r.node_id),longitude=float(r.geometry.x),latitude=float(r.geometry.y),node_type=r.get("node_type","node"),metadata=r.get("metadata",{}))
    if edges:
        for e in edges:
            a,b=str(e["source"]),str(e["target"])
            if a in G and b in G:G.add_edge(a,b,weight=float(e.get("weight",1)),edge_type=e.get("edge_type","relation"))
    if proximity_m is not None and len(proj)>1:
        xy=np.c_[proj.geometry.x,proj.geometry.y];nn=NearestNeighbors(radius=float(proximity_m)).fit(xy);ds,js=nn.radius_neighbors(xy,return_distance=True);ids=proj.node_id.astype(str).tolist()
        for i,(dd,jj) in enumerate(zip(ds,js)):
            for d,j in zip(dd,jj):
                j=int(j)
                if i<j:G.add_edge(ids[i],ids[j],weight=float(max(d,1e-6)),edge_type="proximity")
    degree=nx.degree_centrality(G);between=nx.betweenness_centrality(G,weight="weight");close=nx.closeness_centrality(G,distance="weight")
    comps=list(nx.connected_components(G));ci={n:i for i,c in enumerate(comps) for n in c}
    ns=[{**G.nodes[n],"id":n,"degree_centrality":float(degree[n]),"betweenness_centrality":float(between[n]),"closeness_centrality":float(close[n]),"component":ci[n]} for n in G]
    es=[{"source":a,"target":b,**d} for a,b,d in G.edges(data=True)]
    return {"nodes":ns,"edges":es,"metrics":{"node_count":len(G),"edge_count":G.number_of_edges(),"connected_components":len(comps),"network_density":float(nx.density(G)) if len(G)>1 else 0},"methodology":{"causal_claims":False}}

def shortest_path(graph,source,target):
    G=nx.Graph()
    for n in graph.get("nodes",[]):G.add_node(str(n["id"]))
    for e in graph.get("edges",[]):G.add_edge(str(e["source"]),str(e["target"]),weight=float(e.get("weight",1)))
    if source not in G or target not in G:raise ValueError("source and target must exist")
    try:return {"path":nx.shortest_path(G,source,target,weight="weight"),"distance":float(nx.shortest_path_length(G,source,target,weight="weight")),"connected":True}
    except nx.NetworkXNoPath:return {"path":[],"distance":None,"connected":False}

def detect_network_corridors(nodes,edges=None,proximity_m=5000,bandwidth_m=None,min_nodes=3,temporal_weight=.15,density_weight=.30,connectivity_weight=.35,continuity_weight=.20,permutations=99):
    graph=build_territorial_graph(nodes,edges,proximity_m);geo,proj=_frames(nodes);ids=geo.node_id.astype(str).tolist()
    G=nx.Graph();G.add_nodes_from(ids)
    for e in graph["edges"]:G.add_edge(str(e["source"]),str(e["target"]),weight=float(e.get("weight",1)))
    xy=np.c_[proj.geometry.x,proj.geometry.y];bw=float(bandwidth_m or proximity_m or 1);kde=KernelDensity(bandwidth=max(bw,1)).fit(xy);dens=np.exp(kde.score_samples(xy));dens=(dens-dens.min())/(dens.max()-dens.min()+1e-12);dmap=dict(zip(ids,dens))
    rng=np.random.default_rng(42);out=[]
    for comp in nx.connected_components(G):
        if len(comp)<int(min_nodes):continue
        sub=G.subgraph(comp);mst=nx.minimum_spanning_tree(sub,weight="weight");lookup={str(r.node_id):r.geometry for _,r in proj.iterrows()}
        parts=[LineString([lookup[a],lookup[b]]) for a,b in mst.edges() if a in lookup and b in lookup]
        if not parts:continue
        geom=gpd.GeoSeries([unary_union(parts)],crs=proj.crs).to_crs(4326).iloc[0];members=list(comp)
        density=float(np.mean([dmap[x] for x in members]));connect=float(sub.number_of_edges()/max(len(members)*(len(members)-1)/2,1));lengths=np.array([d.get("weight",1) for *_,d in mst.edges(data=True)],float);continuity=float(1/(1+np.std(lengths)/(np.mean(lengths)+1e-9))) if len(lengths)>1 else 1
        dates=[pd.to_datetime(next((n.get("date") or n.get("timestamp") for n in nodes if str(n.get("id"))==x),None),errors="coerce") for x in members];dates=[d for d in dates if pd.notna(d)];temporal=.5 if len(dates)<2 else float(min(1,len(dates)/(max((max(dates)-min(dates)).days,1)+1)*30))
        w=np.array([density_weight,connectivity_weight,continuity_weight,temporal_weight],float);w/=w.sum();score=float(w@[density,connect,continuity,temporal]);null=[float(w@[float(rng.choice(dens,len(members),replace=True).mean()),connect,continuity,temporal]) for _ in range(int(permutations))];p=float((sum(v>=score for v in null)+1)/(len(null)+1))
        out.append({"corridor_id":"cor_"+uuid.uuid4().hex,"geometry":mapping(geom),"nodes":members,"confidence":score,"validation_pvalue":p,"metrics":{"density":density,"connectivity":connect,"continuity":continuity,"temporal_support":temporal},"method":"density + connectivity + MST continuity + temporal support","limitations":["Confidence is a support score, not probability","No causal or conflict interpretation"]})
    return {"corridors":sorted(out,key=lambda x:x["confidence"],reverse=True),"graph":graph,"methodology":{"permutations":int(permutations),"causal_claims":False,"conflict_classification":False}}
