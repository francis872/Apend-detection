from dataclasses import dataclass
from statistics import mean,pstdev
@dataclass(frozen=True)
class ChangeResult: baseline:float; current:float; delta:float; relative_delta:float; volatility:float; z_score:float
class BaselineEngine:
    @staticmethod
    def detect(history,current):
        values=list(map(float,history))
        if not values:return ChangeResult(current,current,0,0,0,0)
        b=mean(values);v=pstdev(values) if len(values)>1 else 0.;d=current-b
        return ChangeResult(b,current,d,0 if abs(b)<1e-12 else d/abs(b),v,0 if v<1e-12 else d/v)
