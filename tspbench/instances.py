"""TSP instances and distance functions.

An :class:`Instance` is either coordinate-based (``coords`` + ``metric``) or
matrix-based (``matrix``, possibly asymmetric). Distances are computed on
demand from coordinates so that large instances (n = 10k) never need a dense
n x n matrix unless a solver asks for one.

Tour lengths are always computed in float64 from the instance's own distance
function, never from the (possibly rounded or rescaled) values a solver used
internally. TSPLIB metrics (``EUC_2D``, ``ATT``, ``GEO``, ...) are integer
valued by definition, so published TSPLIB optima are directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

# Metrics defined on coordinates. Float metrics are used for generated data;
# upper-case names follow the TSPLIB95 definitions exactly (integer valued).
FLOAT_METRICS = ("euclidean", "manhattan", "chebyshev")
TSPLIB_METRICS = ("EUC_2D", "CEIL_2D", "MAN_2D", "MAX_2D", "ATT", "GEO")


def _nint(x):
    return np.floor(x + 0.5)


def _geo_radians(v):
    pi = 3.141592  # TSPLIB95 uses this truncated value
    deg = np.trunc(v)
    minutes = v - deg
    return pi * (deg + 5.0 * minutes / 3.0) / 180.0


def pairwise(metric: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distances between broadcastable coordinate arrays ``a[..., 2]`` and ``b[..., 2]``."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if metric == "GEO":
        lat_a, lon_a = _geo_radians(a[..., 0]), _geo_radians(a[..., 1])
        lat_b, lon_b = _geo_radians(b[..., 0]), _geo_radians(b[..., 1])
        q1 = np.cos(lon_a - lon_b)
        q2 = np.cos(lat_a - lat_b)
        q3 = np.cos(lat_a + lat_b)
        arg = np.clip(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3), -1.0, 1.0)
        d = np.trunc(6378.388 * np.arccos(arg) + 1.0)
        # TSPLIB's formula gives 1 for identical points; the diagonal is 0.
        return np.where(np.all(a == b, axis=-1), 0.0, d)
    diff = a - b
    if metric in ("euclidean", "EUC_2D", "CEIL_2D", "ATT"):
        sq = np.einsum("...i,...i->...", diff, diff)
        if metric == "euclidean":
            return np.sqrt(sq)
        if metric == "EUC_2D":
            return _nint(np.sqrt(sq))
        if metric == "CEIL_2D":
            return np.ceil(np.sqrt(sq))
        r = np.sqrt(sq / 10.0)
        t = _nint(r)
        return np.where(t < r, t + 1.0, t)
    ad = np.abs(diff)
    if metric == "manhattan":
        return ad.sum(-1)
    if metric == "MAN_2D":
        return _nint(ad.sum(-1))
    if metric == "chebyshev":
        return ad.max(-1)
    if metric == "MAX_2D":
        return np.maximum(_nint(ad[..., 0]), _nint(ad[..., 1]))
    raise ValueError(f"unknown metric {metric!r}")


@dataclass
class Instance:
    """A single TSP / ATSP instance.

    Exactly one of ``coords`` and ``matrix`` defines the distances. ``coords``
    may still be set for a matrix instance (e.g. for plotting) when
    ``metric == "explicit"``.
    """

    name: str
    coords: Optional[np.ndarray] = None
    matrix: Optional[np.ndarray] = None
    metric: str = "euclidean"
    optimum: Optional[float] = None
    meta: Dict = field(default_factory=dict)

    def __post_init__(self):
        if self.matrix is not None:
            self.matrix = np.asarray(self.matrix, dtype=np.float64)
            if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
                raise ValueError("matrix must be square")
            self.metric = "explicit"
        elif self.coords is not None:
            self.coords = np.asarray(self.coords, dtype=np.float64)
            if self.metric not in FLOAT_METRICS + TSPLIB_METRICS:
                raise ValueError(f"unknown metric {self.metric!r}")
        else:
            raise ValueError("instance needs coords or matrix")
        self._symmetric: Optional[bool] = None

    @property
    def n(self) -> int:
        return len(self.matrix) if self.matrix is not None else len(self.coords)

    @property
    def symmetric(self) -> bool:
        if self._symmetric is None:
            self._symmetric = self.matrix is None or bool(np.array_equal(self.matrix, self.matrix.T))
        return self._symmetric

    @property
    def integral(self) -> bool:
        """True when all distances are integers (TSPLIB metrics, integer matrices)."""
        if self.metric in TSPLIB_METRICS:
            return True
        return self.matrix is not None and bool(np.all(self.matrix == np.round(self.matrix)))

    def dist(self, i, j) -> np.ndarray:
        """Vectorised distance from node(s) ``i`` to node(s) ``j``."""
        if self.matrix is not None:
            return self.matrix[i, j]
        return pairwise(self.metric, self.coords[i], self.coords[j])

    def row(self, i: int) -> np.ndarray:
        """Distances from ``i`` to every node."""
        if self.matrix is not None:
            return self.matrix[i]
        return pairwise(self.metric, self.coords[i][None, :], self.coords)

    def col(self, j: int) -> np.ndarray:
        """Distances from every node to ``j``."""
        if self.matrix is not None:
            return self.matrix[:, j]
        return self.row(j)

    def full_matrix(self, max_n: int = 20000) -> np.ndarray:
        """Dense float64 distance matrix. Refuses above ``max_n`` nodes."""
        if self.matrix is not None:
            return self.matrix
        if self.n > max_n:
            raise MemoryError(f"refusing to build a dense {self.n}x{self.n} matrix")
        c = self.coords
        return pairwise(self.metric, c[:, None, :], c[None, :, :])

    def tour_length(self, tour) -> float:
        tour = check_tour(tour, self.n)
        return float(np.sum(self.dist(tour, np.roll(tour, -1)), dtype=np.float64))

    def knn(self, k: int) -> np.ndarray:
        """Indices of the ``k`` nearest successors of every node, shape ``(n, k)``."""
        k = min(k, self.n - 1)
        out = np.empty((self.n, k), dtype=np.int64)
        for i in range(self.n):
            r = self.row(i).copy()
            r[i] = np.inf
            idx = np.argpartition(r, k - 1)[:k]
            out[i] = idx[np.argsort(r[idx], kind="stable")]
        return out


def check_tour(tour, n: int) -> np.ndarray:
    """Return ``tour`` as an int array, raising if it is not a permutation of ``range(n)``."""
    t = np.asarray(tour, dtype=np.int64).ravel()
    if len(t) == n + 1 and t[0] == t[-1]:
        t = t[:-1]  # accept closed tours
    if len(t) != n or not np.array_equal(np.sort(t), np.arange(n)):
        raise ValueError(f"invalid tour for n={n}")
    return t


def is_valid_tour(tour, n: int) -> bool:
    try:
        check_tour(tour, n)
        return True
    except ValueError:
        return False


DistanceFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
