"""Exact symmetric TSP via a MILP with lazily added subtour-elimination cuts.

Used as a drop-in replacement for Concorde where pyconcorde cannot be built
(it downloads QSopt at build time). Solved with SciPy's HiGHS backend, so it
optimizes the float distance matrix directly, with no integer rounding.
"""
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix


def _components(n, edges):
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in edges:
        parent[find(i)] = find(j)
    comps = {}
    for v in range(n):
        comps.setdefault(find(v), []).append(v)
    return list(comps.values())


def solve_tsp_exact(dist, max_rounds=500):
    """Return (tour, length) for a symmetric distance matrix.

    The tour is a permutation starting at node 0; length is computed on `dist`.
    """
    dist = np.asarray(dist, dtype=float)
    n = dist.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    m = len(iu)
    c = dist[iu, ju]

    # Degree constraints: every node has exactly two incident tour edges.
    rows = np.concatenate([iu, ju])
    cols = np.concatenate([np.arange(m), np.arange(m)])
    A_deg = coo_matrix((np.ones(2 * m), (rows, cols)), shape=(n, m)).tocsr()
    constraints = [LinearConstraint(A_deg, 2, 2)]
    cuts, cut_sets = [], []

    for _ in range(max_rounds):
        cons = list(constraints)
        if cuts:
            cons.append(LinearConstraint(np.array(cuts), -np.inf, [len(s) - 1 for s in cut_sets]))
        res = milp(c, constraints=cons, integrality=np.ones(m), bounds=Bounds(0, 1),
                   options={"mip_rel_gap": 1e-9})
        if res.status != 0:
            raise RuntimeError(f"MILP failed: {res.message}")
        chosen = np.flatnonzero(res.x > 0.5)
        edges = list(zip(iu[chosen], ju[chosen]))
        comps = _components(n, edges)
        if len(comps) == 1:
            return _edges_to_tour(n, edges), float(c[chosen].sum())
        for comp in comps:
            s = set(comp)
            row = np.array([1.0 if (a in s and b in s) else 0.0 for a, b in zip(iu, ju)])
            cuts.append(row)
            cut_sets.append(comp)
    raise RuntimeError("subtour elimination did not converge")


def _edges_to_tour(n, edges):
    nbrs = [[] for _ in range(n)]
    for i, j in edges:
        nbrs[i].append(j)
        nbrs[j].append(i)
    tour, prev, cur = [0], -1, 0
    while len(tour) < n:
        nxt = nbrs[cur][0] if nbrs[cur][0] != prev else nbrs[cur][1]
        tour.append(nxt)
        prev, cur = cur, nxt
    return np.array(tour)


def tour_length(tour, dist):
    tour = np.asarray(tour)
    return float(dist[tour, np.roll(tour, -1)].sum())
