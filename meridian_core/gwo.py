from dataclasses import dataclass
import random
@dataclass(frozen=True)
class GWOResult: position:list[float]; objective:float; convergence:list[float]; iterations:int
class GreyWolfOptimizer:
    def __init__(self,objective,bounds,population=24,iterations=80,seed=872):
        if population<3:raise ValueError("population must be >= 3")
        self.f=objective;self.bounds=bounds;self.population=population;self.iterations=iterations;self.r=random.Random(seed)
    def _clip(self,x):return [max(lo,min(hi,v)) for v,(lo,hi) in zip(x,self.bounds)]
    def run(self):
        wolves=[[self.r.uniform(lo,hi) for lo,hi in self.bounds] for _ in range(self.population)];curve=[]
        for t in range(self.iterations):
            wolves.sort(key=self.f);alpha,beta,delta=wolves[:3];curve.append(self.f(alpha));a=2*(1-t/max(1,self.iterations-1));new=[]
            for wolf in wolves:
                pos=[]
                for j,x in enumerate(wolf):
                    estimates=[]
                    for leader in (alpha,beta,delta):
                        r1,r2=self.r.random(),self.r.random();A=2*a*r1-a;C=2*r2;D=abs(C*leader[j]-x);estimates.append(leader[j]-A*D)
                    pos.append(sum(estimates)/3)
                new.append(self._clip(pos))
            wolves=new
        wolves.sort(key=self.f);return GWOResult(wolves[0],self.f(wolves[0]),curve,self.iterations)
