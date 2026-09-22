from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from scipy.spatial import procrustes
from sklearn.neighbors import NearestNeighbors


def nearest_spd(a: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    a = np.asarray(a, float)
    a = (a + a.T) / 2
    vals, vecs = eigh(a)
    vals = np.maximum(vals, eps)
    return (vecs * vals) @ vecs.T


def spd_affine_invariant_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Affine-invariant Riemannian distance on SPD matrices."""
    a, b = nearest_spd(a), nearest_spd(b)
    vals = eigh(b, a, eigvals_only=True)
    vals = np.clip(vals, 1e-15, None)
    return float(np.sqrt(np.sum(np.log(vals)**2)))


def local_spd_scores(features: np.ndarray, k: int = 25) -> np.ndarray:
    z = np.asarray(features, float)
    n, d = z.shape
    k = min(max(k, d+2), n)
    nbrs = NearestNeighbors(n_neighbors=k).fit(z)
    idx = nbrs.kneighbors(return_distance=False)
    global_cov = nearest_spd(np.cov(z, rowvar=False) + np.eye(d)*1e-6)
    scores = np.empty(n)
    for i, ids in enumerate(idx):
        local = z[ids]
        cov = nearest_spd(np.cov(local, rowvar=False) + np.eye(d)*1e-6)
        scores[i] = spd_affine_invariant_distance(cov, global_cov)
    return scores


def procrustes_neighborhood_scores(coords: np.ndarray, k: int = 12) -> np.ndarray:
    xy = np.asarray(coords, float)
    n = len(xy)
    k = min(max(k, 4), n)
    nn = NearestNeighbors(n_neighbors=k).fit(xy)
    _, ids = nn.kneighbors(xy)
    clouds = []
    for i in range(n):
        p = xy[ids[i]] - xy[i]
        ang = np.arctan2(p[:,1], p[:,0])
        p = p[np.argsort(ang)]
        s = np.sqrt((p*p).sum())
        if s > 0: p = p/s
        clouds.append(p)
    ref = np.median(np.stack(clouds), axis=0)
    if np.allclose(ref, 0):
        ref = clouds[0]
    scores = np.empty(n)
    for i,p in enumerate(clouds):
        try:
            _, _, disparity = procrustes(ref, p)
        except Exception:
            disparity = 1.0
        scores[i] = float(disparity)
    return scores
