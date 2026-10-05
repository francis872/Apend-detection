from dataclasses import dataclass
@dataclass(frozen=True)
class Corridor: source:str; target:str; distance_km:float; connectivity:float; score:float
class CorridorEngine:
    @staticmethod
    def rank(graph,max_results=None):
        seen=set();out=[]
        for e in graph.edges:
            key=tuple(sorted((e.source,e.target)))
            if key in seen:continue
            seen.add(key);score=e.weight
            out.append(Corridor(e.source,e.target,e.distance_km,e.weight,score))
        out.sort(key=lambda x:x.score,reverse=True)
        return out[:max_results] if max_results else out
