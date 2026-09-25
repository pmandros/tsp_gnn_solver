"""Random TSP instance generators.

Every generator returns only a dense symmetric distance matrix ``D`` of shape
``(n, n)`` with a zero diagonal. The model never sees coordinates, so the same
code path handles Euclidean, non-Euclidean and non-metric instances.
"""
import numpy as np


def _pairwise(points, p):
    diff = points[:, None, :] - points[None, :, :]
    if p == 2:
        return np.sqrt((diff ** 2).sum(-1))
    if p == 1:
        return np.abs(diff).sum(-1)
    if p == np.inf:
        return np.abs(diff).max(-1)
    raise ValueError(p)


def euclidean(n, rng):
    return _pairwise(rng.random((n, 2)), 2)


def clustered(n, rng):
    """Gaussian clusters (as in the DIMACS TSP challenge generator)."""
    k = rng.integers(3, 9)
    centers = rng.random((k, 2))
    assign = rng.integers(0, k, n)
    sigma = rng.uniform(0.02, 0.08)
    return _pairwise(centers[assign] + sigma * rng.standard_normal((n, 2)), 2)


def manhattan(n, rng):
    return _pairwise(rng.random((n, 2)), 1)


def chebyshev(n, rng):
    return _pairwise(rng.random((n, 2)), np.inf)


def random_matrix(n, rng):
    """Symmetric i.i.d. U(0,1) entries: non-metric, no geometry at all."""
    a = rng.random((n, n))
    d = np.triu(a, 1)
    return d + d.T


def shortest_path(n, rng):
    """Metric but non-Euclidean: shortest-path closure of a random matrix."""
    d = random_matrix(n, rng)
    for k in range(n):  # Floyd-Warshall
        d = np.minimum(d, d[:, k:k + 1] + d[k:k + 1, :])
    return d


GENERATORS = {
    "euclidean": euclidean,
    "clustered": clustered,
    "manhattan": manhattan,
    "chebyshev": chebyshev,
    "random": random_matrix,
    "shortest_path": shortest_path,
}

# Types seen in training; the rest are held out to test distance-type transfer.
TRAIN_TYPES = ("euclidean", "clustered", "random")
HELDOUT_TYPES = ("manhattan", "chebyshev", "shortest_path")


def generate(kind, n, rng, scale=1.0):
    d = GENERATORS[kind](n, rng) * scale
    np.fill_diagonal(d, 0.0)
    return d
