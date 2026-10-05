from dataclasses import dataclass
@dataclass(frozen=True)
class RiskResult: score:float; band:str; components:dict
class RiskEngine:
    def __init__(self,weights=None):
        self.weights=weights or {"isolation":.30,"sparsity":.20,"change":.30,"volatility":.20}
        if abs(sum(self.weights.values())-1)>1e-9: raise ValueError("Risk weights must sum to 1")
    def score(self,connectivity,density,relative_change,relative_volatility):
        c={"isolation":1-max(0,min(1,connectivity)),"sparsity":1-max(0,min(1,density)),"change":max(0,min(1,abs(relative_change))),"volatility":max(0,min(1,relative_volatility))}
        s=sum(self.weights[k]*c[k] for k in self.weights);return RiskResult(s,"High" if s>=.7 else "Medium" if s>=.4 else "Low",c)
