from dataclasses import dataclass
from math import asin, cos, radians, sin, sqrt
@dataclass(frozen=True)
class SpatialPoint: id:str; lat:float; lon:float; kind:str="generic"
@dataclass(frozen=True)
class Edge: source:str; target:str; distance_km:float; weight:float
@dataclass(frozen=True)
class SpatialGraph: nodes:list[SpatialPoint]; edges:list[Edge]
class SpatialEngine:
    R=6371.0088
    @staticmethod
    def encode(p):
        la,lo=radians(p.lat),radians(p.lon);return (cos(la)*cos(lo),cos(la)*sin(lo),sin(la))
    @classmethod
    def distance(cls,a,b):
        la1,la2=radians(a.lat),radians(b.lat);dlat=la2-la1;dlon=radians(b.lon-a.lon)
        h=sin(dlat/2)**2+cos(la1)*cos(la2)*sin(dlon/2)**2
        return 2*cls.R*asin(min(1,sqrt(h)))
    @classmethod
    def graph(cls,nodes,radius_km):
        edges=[]
        for i,a in enumerate(nodes):
            for b in nodes[i+1:]:
                d=cls.distance(a,b)
                if d<=radius_km:
                    w=1/(1+d);edges += [Edge(a.id,b.id,d,w),Edge(b.id,a.id,d,w)]
        return SpatialGraph(list(nodes),edges)
