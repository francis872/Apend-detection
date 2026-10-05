from dataclasses import dataclass
from math import sqrt
@dataclass(frozen=True)
class Forecast: next_value:float; trend:float; confidence:float; rmse:float
class ForecastingEngine:
    @staticmethod
    def linear(values):
        y=list(map(float,values));n=len(y)
        if not n:return Forecast(0,0,0,0)
        if n==1:return Forecast(y[0],0,.25,0)
        mx=(n-1)/2;my=sum(y)/n;den=sum((i-mx)**2 for i in range(n));slope=sum((i-mx)*(v-my) for i,v in enumerate(y))/den if den else 0
        pred=[my+slope*(i-mx) for i in range(n)];rmse=sqrt(sum((v-p)**2 for v,p in zip(y,pred))/n);nxt=my+slope*(n-mx);conf=1/(1+rmse/(abs(my)+1e-12))
        return Forecast(nxt,slope,max(0,min(1,conf)),rmse)
