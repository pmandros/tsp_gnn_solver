"""Random instance generators.

``kool_uniform`` reproduces the test-set generation of Kool et al. (2019),
"Attention, Learn to Solve Routing Problems!" (``generate_data.py``: the legacy
NumPy RNG seeded with 1234, then ``np.random.uniform(size=(N, n, 2))``). The
other generators cover the distance types the project wants to generalise to:
other norms, clustered points, symmetric non-metric matrices and asymmetric
matrices that satisfy the triangle inequality (the "tmat" class used by
MatNet, Kwon et al. 2021).
"""

from __future__ import annotations

from typing import List

import numpy as np

from .instances import Instance

KOOL_TEST_SEED = 1234
KOOL_VAL_SEED = 4321


def kool_uniform(n: int, num: int = 10000, seed: int = KOOL_TEST_SEED, prefix: str = "") -> List[Instance]:
    """Uniform points in the unit square, generated exactly like Kool et al.'s TSP test sets."""
    pts = np.random.RandomState(seed).uniform(size=(num, n, 2))
    prefix = prefix or f"tsp{n}"
    return [Instance(f"{prefix}_{i:05d}", coords=p, meta={"family": "uniform", "seed": seed}) for i, p in enumerate(pts)]


def uniform(n: int, num: int, seed: int, metric: str = "euclidean", prefix: str = "") -> List[Instance]:
    rng = np.random.default_rng(seed)
    prefix = prefix or f"{metric}{n}"
    return [
        Instance(f"{prefix}_{i:05d}", coords=rng.random((n, 2)), metric=metric, meta={"family": metric, "seed": seed})
        for i in range(num)
    ]


def clustered(n: int, num: int, seed: int, num_clusters: int = 3, std: float = 0.07, prefix: str = "") -> List[Instance]:
    """Gaussian mixture: cluster centres uniform in [0.2, 0.8]^2, points clipped to the unit square."""
    rng = np.random.default_rng(seed)
    prefix = prefix or f"clustered{n}"
    out = []
    for i in range(num):
        centres = rng.uniform(0.2, 0.8, size=(num_clusters, 2))
        assign = rng.integers(num_clusters, size=n)
        pts = np.clip(centres[assign] + std * rng.standard_normal((n, 2)), 0.0, 1.0)
        out.append(
            Instance(
                f"{prefix}_{i:05d}",
                coords=pts,
                meta={"family": "clustered", "seed": seed, "num_clusters": num_clusters, "std": std},
            )
        )
    return out


def nonmetric(n: int, num: int, seed: int, prefix: str = "") -> List[Instance]:
    """Symmetric matrices with i.i.d. U(0, 1) entries; the triangle inequality generally fails."""
    rng = np.random.default_rng(seed)
    prefix = prefix or f"nonmetric{n}"
    out = []
    for i in range(num):
        m = np.triu(rng.random((n, n)), 1)
        out.append(Instance(f"{prefix}_{i:05d}", matrix=m + m.T, meta={"family": "nonmetric", "seed": seed}))
    return out


def metric_closure(m: np.ndarray) -> np.ndarray:
    """All-pairs shortest paths (Floyd-Warshall), i.e. the fixed point MatNet iterates to."""
    m = m.copy()
    for k in range(len(m)):
        np.minimum(m, m[:, k : k + 1] + m[k : k + 1, :], out=m)
    return m


def atsp_tmat(n: int, num: int, seed: int, int_max: int = 1_000_000, prefix: str = "") -> List[Instance]:
    """MatNet-style asymmetric instances ("tmat").

    Random integers in ``[0, int_max)``, zero diagonal, then closed under
    shortest paths so the triangle inequality holds, then divided by
    ``int_max``. The distribution matches MatNet; the RNG stream does not, so
    these are not MatNet's released test instances.
    """
    rng = np.random.default_rng(seed)
    prefix = prefix or f"atsp{n}"
    out = []
    for i in range(num):
        m = rng.integers(0, int_max, size=(n, n)).astype(np.float64)
        np.fill_diagonal(m, 0.0)
        m = metric_closure(m) / int_max
        out.append(Instance(f"{prefix}_{i:05d}", matrix=m, meta={"family": "atsp_tmat", "seed": seed}))
    return out
