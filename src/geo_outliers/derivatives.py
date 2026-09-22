from __future__ import annotations

import numpy as np
from scipy import stats


def pdf_derivatives(
    distribution: str,
    params: tuple[float, ...],
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate fitted PDF plus first and second numerical derivatives on x."""
    grid=np.asarray(x,dtype=float)
    if grid.ndim!=1 or len(grid)<5:
        raise ValueError("x must be a one-dimensional grid with at least 5 points")
    order=np.argsort(grid)
    xs=grid[order]
    dist=getattr(stats,distribution)
    pdf=np.asarray(dist.pdf(xs,*params),dtype=float)
    pdf=np.nan_to_num(pdf,nan=0.0,posinf=0.0,neginf=0.0)
    d1=np.gradient(pdf,xs,edge_order=2)
    d2=np.gradient(d1,xs,edge_order=2)

    inv=np.empty_like(order)
    inv[order]=np.arange(len(order))
    return pdf[inv],d1[inv],d2[inv]


def find_inflection_points(
    distribution: str,
    params: tuple[float, ...],
    lower: float | None=None,
    upper: float | None=None,
    grid_size: int=2048,
) -> dict:
    """Detect PDF inflection points where f'' changes sign.

    Roots are estimated by linear interpolation between adjacent second-derivative
    samples. Only sign changes are accepted; touching zero without a concavity
    change is not considered an inflection point.
    """
    dist=getattr(stats,distribution)
    if lower is None:
        lower=float(dist.ppf(1e-5,*params))
    if upper is None:
        upper=float(dist.ppf(1-1e-5,*params))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower>=upper:
        raise ValueError("Could not determine finite derivative bounds")

    x=np.linspace(lower,upper,max(256,int(grid_size)))
    pdf,d1,d2=pdf_derivatives(distribution,params,x)

    points=[]
    for i in range(len(x)-1):
        a,b=d2[i],d2[i+1]
        if not np.isfinite(a) or not np.isfinite(b) or a*b>=0:
            continue
        denom=abs(a)+abs(b)
        frac=abs(a)/denom if denom else .5
        xr=float(x[i]+frac*(x[i+1]-x[i]))
        points.append(xr)

    # Remove near-duplicate roots caused by numerical jitter.
    tol=(upper-lower)/max(grid_size,1)*3
    dedup=[]
    for p in points:
        if not dedup or abs(p-dedup[-1])>tol:
            dedup.append(p)

    return {
        "distribution":distribution,
        "bounds":[float(lower),float(upper)],
        "x":x,
        "pdf":pdf,
        "first_derivative":d1,
        "second_derivative":d2,
        "inflection_points":dedup,
    }


def observation_derivative_features(
    values: np.ndarray,
    distribution: str,
    params: tuple[float, ...],
    grid_size: int=2048,
) -> tuple[dict[str,np.ndarray],dict]:
    """Interpolate derivative diagnostics for observations."""
    v=np.asarray(values,dtype=float)
    result=find_inflection_points(distribution,params,grid_size=grid_size)
    x=result["x"]

    pdf=np.interp(v,x,result["pdf"],left=0.0,right=0.0)
    d1=np.interp(v,x,result["first_derivative"],left=0.0,right=0.0)
    d2=np.interp(v,x,result["second_derivative"],left=0.0,right=0.0)

    infl=np.asarray(result["inflection_points"],dtype=float)
    if len(infl):
        distance=np.min(np.abs(v[:,None]-infl[None,:]),axis=1)
        scale=max(float(np.nanstd(v)),1e-12)
        near=distance <= 0.05*scale
    else:
        distance=np.full(len(v),np.nan)
        near=np.zeros(len(v),dtype=bool)

    finite=np.isfinite(v)
    pdf[~finite]=np.nan; d1[~finite]=np.nan; d2[~finite]=np.nan
    distance[~finite]=np.nan; near[~finite]=False

    summary={
        "method":"numerical first/second derivative of fitted PDF",
        "inflection_points":[float(z) for z in result["inflection_points"]],
        "bounds":[float(z) for z in result["bounds"]],
        "grid_size":int(len(x)),
    }
    features={
        "pdf_value":pdf,
        "pdf_derivative_1":d1,
        "pdf_derivative_2":d2,
        "distance_to_inflection":distance,
        "is_inflection_zone":near,
    }
    return features,summary
