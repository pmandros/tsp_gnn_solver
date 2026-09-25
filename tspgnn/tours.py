"""Tour utilities: exact-length scoring, LKH reference, classic heuristics, 2-opt."""
import numpy as np
import numba


def tour_length(d, tour):
    """Length of a closed tour, always computed on the float matrix."""
    tour = np.asarray(tour)
    return float(d[tour, np.roll(tour, -1)].sum())


def is_valid_tour(tour, n):
    return len(tour) == n and np.array_equal(np.sort(tour), np.arange(n))


def tour_to_adjacency(tour, n):
    """Symmetric 0/1 adjacency (both i->j and j->i are positives)."""
    tour = np.asarray(tour)
    a = np.zeros((n, n), dtype=np.float32)
    nxt = np.roll(tour, -1)
    a[tour, nxt] = 1.0
    a[nxt, tour] = 1.0
    return a


def lkh_tour(d, runs=1):
    """Reference tour from LKH-3 (via elkai).

    LKH needs integer costs, so the matrix is scaled to ~1e6 resolution first.
    The returned tour is always re-scored on the float matrix by the caller,
    so the rounding never leaks into reported gaps.
    """
    import elkai

    n = d.shape[0]
    if n <= 3:
        return np.arange(n)
    scaled = np.rint(d / d.max() * 1e6).astype(np.int64)
    tour = elkai.DistanceMatrix(scaled.tolist()).solve_tsp(runs=runs)
    return np.asarray(tour[:-1], dtype=np.int64)


@numba.njit(cache=True)
def nearest_neighbor(d, start=0):
    n = d.shape[0]
    visited = np.zeros(n, dtype=np.bool_)
    tour = np.empty(n, dtype=np.int64)
    cur = start
    visited[cur] = True
    tour[0] = cur
    for t in range(1, n):
        best, best_j = np.inf, -1
        for j in range(n):
            if not visited[j] and d[cur, j] < best:
                best, best_j = d[cur, j], j
        cur = best_j
        visited[cur] = True
        tour[t] = cur
    return tour


@numba.njit(cache=True)
def _find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


@numba.njit(cache=True)
def greedy_edge_tour(n, src, dst, order, d):
    """Build a tour by inserting candidate edges in ``order`` (best first).

    An edge is accepted if both endpoints still have degree < 2 and it closes
    no subtour. Remaining path fragments are then joined greedily by distance.
    Works for any edge ranking: model scores, or plain distances.
    """
    deg = np.zeros(n, dtype=np.int64)
    parent = np.arange(n)
    nb = -np.ones((n, 2), dtype=np.int64)
    added = 0
    for idx in order:
        i, j = src[idx], dst[idx]
        if i == j or deg[i] >= 2 or deg[j] >= 2:
            continue
        ri, rj = _find(parent, i), _find(parent, j)
        if ri == rj:
            continue
        parent[ri] = rj
        nb[i, deg[i]] = j
        nb[j, deg[j]] = i
        deg[i] += 1
        deg[j] += 1
        added += 1
        if added == n - 1:
            break
    # Join fragments: repeatedly link the closest pair of endpoints in
    # different components until a Hamiltonian path remains.
    while added < n - 1:
        best, bi, bj = np.inf, -1, -1
        for i in range(n):
            if deg[i] >= 2:
                continue
            ri = _find(parent, i)
            for j in range(i + 1, n):
                if deg[j] >= 2 or _find(parent, j) == ri:
                    continue
                if d[i, j] < best:
                    best, bi, bj = d[i, j], i, j
        parent[_find(parent, bi)] = _find(parent, bj)
        nb[bi, deg[bi]] = bj
        nb[bj, deg[bj]] = bi
        deg[bi] += 1
        deg[bj] += 1
        added += 1
    # Walk the Hamiltonian path from one endpoint.
    start = 0
    for i in range(n):
        if deg[i] < 2:
            start = i
            break
    tour = np.empty(n, dtype=np.int64)
    prev, cur = -1, start
    for t in range(n):
        tour[t] = cur
        nxt = nb[cur, 0] if nb[cur, 0] != prev else nb[cur, 1]
        prev, cur = cur, nxt
    return tour


@numba.njit(cache=True)
def _reverse(tour, pos, lo, hi):
    """Reverse tour[lo .. hi] cyclically, keeping ``pos`` in sync."""
    n = tour.shape[0]
    length = (hi - lo) % n + 1
    for s in range(length // 2):
        u, v = (lo + s) % n, (hi - s) % n
        tu, tv = tour[u], tour[v]
        tour[u], tour[v] = tv, tu
        pos[tv], pos[tu] = u, v


@numba.njit(cache=True)
def two_opt(d, tour, neighbors, max_passes=1000):
    """First-improvement 2-opt restricted to a candidate neighbor list.

    ``neighbors[i]`` holds the candidate partners of city i sorted by distance
    (e.g. its k nearest neighbors), which allows the standard early exit.
    Both neighbor-list move types (via successors and via predecessors) are
    tried. Symmetric matrices only. Runs until a local optimum or
    ``max_passes`` full sweeps.
    """
    n = tour.shape[0]
    tour = tour.copy()
    pos = np.empty(n, dtype=np.int64)
    for i in range(n):
        pos[tour[i]] = i
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n):
            # Move type 1: new edges (a, c) and (succ a, succ c).
            a = tour[i]
            b = tour[(i + 1) % n]
            for c in neighbors[a]:
                if d[a, c] >= d[a, b]:
                    break
                j = pos[c]
                e = tour[(j + 1) % n]
                if c == b or e == a:
                    continue
                if d[a, c] + d[b, e] - d[a, b] - d[c, e] < -1e-10:
                    _reverse(tour, pos, (pos[a] + 1) % n, j)
                    improved = True
                    a = tour[pos[a]]
                    b = tour[(pos[a] + 1) % n]
            # Move type 2: new edges (a, c) and (pred a, pred c).
            a = tour[i]
            p = tour[(i - 1) % n]
            for c in neighbors[a]:
                if d[a, c] >= d[p, a]:
                    break
                j = pos[c]
                q = tour[(j - 1) % n]
                if c == p or q == a:
                    continue
                if d[a, c] + d[p, q] - d[p, a] - d[q, c] < -1e-10:
                    _reverse(tour, pos, j, (pos[a] - 1) % n)
                    improved = True
                    p = tour[(pos[a] - 1) % n]
    return tour


def knn_lists(d, k):
    """Indices of the k nearest other cities for every city."""
    n = d.shape[0]
    k = min(k, n - 1)
    masked = d + np.diag(np.full(n, np.inf))
    idx = np.argpartition(masked, k - 1, axis=1)[:, :k]
    order = np.take_along_axis(masked, idx, 1).argsort(1)
    return np.take_along_axis(idx, order, 1).astype(np.int64)
