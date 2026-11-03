import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN, AgglomerativeClustering

from harness.metrics.verification import l2_normalize

CANNOT_LINK_DISTANCE = 1e6


def distance_matrix(emb, metric="cosine"):
    e = l2_normalize(emb)
    if metric == "cosine":
        d = 1.0 - e @ e.T
    elif metric == "euclidean_l2":
        sq = np.sum(e ** 2, axis=1)
        d = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * e @ e.T, 0))
    else:
        raise ValueError(f"métrica desconocida: {metric}")
    d = np.maximum(d, 0)
    np.fill_diagonal(d, 0)
    return (d + d.T) / 2


def apply_cannot_link(d, pairs):
    d = d.copy()
    for i, j in pairs:
        d[i, j] = d[j, i] = CANNOT_LINK_DISTANCE
    return d


def cluster(d, algo, params):
    n = len(d)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0])
    if algo == "dbscan":
        return DBSCAN(eps=params["eps"], min_samples=params["min_samples"],
                      metric="precomputed").fit_predict(d)
    if algo == "hdbscan":
        if n < max(2, params["min_cluster_size"]):
            return np.full(n, -1)
        return HDBSCAN(min_cluster_size=params["min_cluster_size"], min_samples=params["min_samples"],
                       cluster_selection_epsilon=params["cluster_selection_epsilon"],
                       metric="precomputed", copy=True).fit_predict(d)
    if algo == "agglomerative":
        if params["linkage"] == "ward":
            raise ValueError("ward necesita características euclidianas, no una matriz de distancias")
        return AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage=params["linkage"],
                                       distance_threshold=params["distance_threshold"]).fit_predict(d)
    raise ValueError(f"algoritmo desconocido: {algo}")
