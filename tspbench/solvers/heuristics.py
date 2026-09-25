"""Construction heuristics, 2-opt local search and a tiny exact solver.

All constructions work on arbitrary (including asymmetric) distances through
``Instance.row``/``Instance.col`` and never need a dense matrix, so they run on
TSP10000. 2-opt reverses segments and is therefore symmetric-only.
"""

from __future__ import annotations

import time

import numpy as np

from ..instances import Instance, check_tour
from .base import Solver, Unsupported, make_solver, register

# --------------------------------------------------------------------------- #
# Constructions
# --------------------------------------------------------------------------- #


def nearest_neighbor(inst: Instance, start: int = 0) -> np.ndarray:
    n = inst.n
    visited = np.zeros(n, dtype=bool)
    tour = np.empty(n, dtype=np.int64)
    cur = start
    for step in range(n):
        tour[step] = cur
        visited[cur] = True
        if step == n - 1:
            break
        r = inst.row(cur).astype(np.float64, copy=True)
        r[visited] = np.inf
        cur = int(np.argmin(r))
    return tour


def insertion(inst: Instance, mode: str = "farthest", start: int = 0, rng=None) -> np.ndarray:
    """Nearest / farthest / random insertion.

    The node to insert is chosen by ``mode``; it is placed at the position of
    least added cost. For asymmetric instances the node-to-tour distance is the
    smaller of the two directions.
    """
    n = inst.n
    in_tour = np.zeros(n, dtype=bool)
    in_tour[start] = True
    tour = np.array([start], dtype=np.int64)
    edge = np.zeros(1)  # edge[p] = d(tour[p], tour[p+1 mod len])
    mind = np.minimum(inst.row(start), inst.col(start)).astype(np.float64)
    order = rng.permutation(n) if mode == "random" else None
    order_pos = 0
    for _ in range(n - 1):
        if mode == "random":
            while in_tour[order[order_pos]]:
                order_pos += 1
            k = int(order[order_pos])
        else:
            cand = np.where(in_tour, np.nan, mind)
            k = int(np.nanargmin(cand) if mode == "nearest" else np.nanargmax(cand))
        to_k = inst.col(k)[tour]
        from_k = inst.row(k)[np.roll(tour, -1)]
        cost = to_k + from_k - edge
        p = int(np.argmin(cost))
        tour = np.insert(tour, p + 1, k)
        edge = np.concatenate([edge[:p], [to_k[p], from_k[p]], edge[p + 1 :]])
        in_tour[k] = True
        if mode != "random":
            np.minimum(mind, np.minimum(inst.row(k), inst.col(k)), out=mind)
    return tour


def cheapest_insertion(inst: Instance, start: int = 0) -> np.ndarray:
    """Insert the (node, position) pair of least added cost at every step. O(n^3)."""
    d = inst.full_matrix()
    n = inst.n
    tour = [start]
    rest = np.ones(n, dtype=bool)
    rest[start] = False
    for _ in range(n - 1):
        t = np.array(tour)
        nxt = np.roll(t, -1)
        u = np.flatnonzero(rest)
        cost = d[t][:, u] + d[u][:, nxt].T - d[t, nxt][:, None]
        p, j = np.unravel_index(int(np.argmin(cost)), cost.shape)
        tour.insert(p + 1, int(u[j]))
        rest[u[j]] = False
    return np.array(tour, dtype=np.int64)


# --------------------------------------------------------------------------- #
# 2-opt
# --------------------------------------------------------------------------- #


def two_opt_dense(d: np.ndarray, tour: np.ndarray, time_limit: float = np.inf, max_iters: int = 10**7) -> np.ndarray:
    """Best-improvement 2-opt on a dense symmetric matrix (vectorised, for n up to ~1000)."""
    t = np.array(tour, dtype=np.int64)
    n = len(t)
    if n < 4:
        return t
    iu = np.triu(np.ones((n, n), dtype=bool), 2)
    iu[0, n - 1] = False  # these two edges are adjacent in the cycle
    t0 = time.perf_counter()
    for _ in range(max_iters):
        a = d[np.ix_(t, t)]
        e = np.diagonal(np.roll(a, -1, axis=1)).copy()  # e[i] = d(t_i, t_{i+1})
        delta = a + np.roll(np.roll(a, -1, axis=0), -1, axis=1) - e[:, None] - e[None, :]
        delta[~iu] = 0.0
        i, j = np.unravel_index(int(np.argmin(delta)), delta.shape)
        if delta[i, j] >= -1e-10 or time.perf_counter() - t0 > time_limit:
            break
        t[i + 1 : j + 1] = t[i + 1 : j + 1][::-1]
    return t


def two_opt_neighbors(inst: Instance, tour: np.ndarray, k: int = 10, time_limit: float = np.inf) -> np.ndarray:
    """First-improvement 2-opt restricted to k-nearest-neighbour candidates, with don't-look bits."""
    t = np.array(tour, dtype=np.int64)
    n = len(t)
    if n < 4:
        return t
    nbrs = inst.knn(k)
    pos = np.empty(n, dtype=np.int64)
    pos[t] = np.arange(n)

    def dist(i, j):
        return float(inst.dist(i, j))

    def reverse(i, j):
        # reverse the cyclic segment t[i..j]; reverse the complement when shorter
        length = (j - i) % n + 1
        if 2 * length > n:
            i, j = (j + 1) % n, (i - 1) % n
            length = n - length
        for _ in range(length // 2):
            ci, cj = t[i], t[j]
            t[i], t[j] = cj, ci
            pos[cj], pos[ci] = i, j
            i = (i + 1) % n
            j = (j - 1) % n

    active = list(range(n))
    in_queue = np.ones(n, dtype=bool)
    t0 = time.perf_counter()
    while active:
        if time.perf_counter() - t0 > time_limit:
            break
        a = active.pop()
        in_queue[a] = False
        improved = False
        for direction in (1, -1):
            pa = pos[a]
            b = t[(pa + direction) % n]
            dab = dist(a, b)
            for c in nbrs[a]:
                dac = dist(a, c)
                if dac >= dab:
                    break
                pc = pos[c]
                dd = t[(pc + direction) % n]
                if c == b or dd == a:
                    continue
                delta = dac + dist(b, dd) - dab - dist(c, dd)
                if delta < -1e-10:
                    if direction == 1:
                        reverse((pa + 1) % n, pc)
                    else:
                        reverse(pc, (pa - 1) % n)
                    for x in (a, b, c, dd):
                        if not in_queue[x]:
                            in_queue[x] = True
                            active.append(x)
                    improved = True
                    break
            if improved:
                break
    return t


def two_opt(inst: Instance, tour: np.ndarray, dense_max_n: int = 1000, **kw) -> np.ndarray:
    if not inst.symmetric:
        raise Unsupported("2-opt needs a symmetric instance")
    if inst.n <= dense_max_n:
        return two_opt_dense(inst.full_matrix(), tour, time_limit=kw.get("time_limit", np.inf))
    return two_opt_neighbors(inst, tour, **kw)


# --------------------------------------------------------------------------- #
# Exact dynamic programme (tests and tiny instances)
# --------------------------------------------------------------------------- #


def held_karp(d: np.ndarray) -> np.ndarray:
    """Exact Held-Karp DP; works for asymmetric matrices. O(2^n n^2), n <= 13."""
    n = len(d)
    if n > 13:
        raise Unsupported("held_karp is limited to n <= 13")
    if n <= 3:
        return np.arange(n)
    m = n - 1  # node 0 is the fixed start; subsets over nodes 1..n-1
    full = 1 << m
    dp = np.full((full, m), np.inf)
    parent = np.full((full, m), -1, dtype=np.int64)
    for j in range(m):
        dp[1 << j, j] = d[0, j + 1]
    for mask in range(1, full):
        for j in range(m):
            if not mask & (1 << j) or not np.isfinite(dp[mask, j]):
                continue
            base = dp[mask, j]
            for k in range(m):
                if mask & (1 << k):
                    continue
                nm = mask | (1 << k)
                v = base + d[j + 1, k + 1]
                if v < dp[nm, k]:
                    dp[nm, k] = v
                    parent[nm, k] = j
    last = int(np.argmin(dp[full - 1] + d[1:, 0]))
    tour = []
    mask = full - 1
    while last != -1:
        tour.append(last + 1)
        prev = parent[mask, last]
        mask ^= 1 << last
        last = prev
    return np.array([0] + tour[::-1], dtype=np.int64)


# --------------------------------------------------------------------------- #
# Registered solvers
# --------------------------------------------------------------------------- #


@register("nearest_neighbor")
class NearestNeighbor(Solver):
    """Nearest neighbour from a random start node (``start=<int>`` fixes it)."""

    stochastic = True

    def solve(self, inst, seed=0):
        start = self.params.get("start")
        if start is None:
            start = int(np.random.default_rng(seed).integers(inst.n))
        return nearest_neighbor(inst, start)


@register("nearest_insertion")
class NearestInsertion(Solver):
    def solve(self, inst, seed=0):
        return insertion(inst, "nearest", start=self.params.get("start", 0))


@register("farthest_insertion")
class FarthestInsertion(Solver):
    def solve(self, inst, seed=0):
        return insertion(inst, "farthest", start=self.params.get("start", 0))


@register("random_insertion")
class RandomInsertion(Solver):
    stochastic = True

    def solve(self, inst, seed=0):
        rng = np.random.default_rng(seed)
        return insertion(inst, "random", start=int(rng.integers(inst.n)), rng=rng)


@register("cheapest_insertion")
class CheapestInsertion(Solver):
    def solve(self, inst, seed=0):
        if inst.n > self.params.get("max_n", 1000):
            raise Unsupported("cheapest_insertion is O(n^3); raise max_n to force it")
        return cheapest_insertion(inst, start=self.params.get("start", 0))


@register("two_opt")
class TwoOpt(Solver):
    """2-opt on top of a construction: ``two_opt:init=farthest_insertion,time_limit=10``."""

    def __init__(self, **params):
        super().__init__(**params)
        self.init = make_solver(params.get("init", "nearest_neighbor"))
        self.stochastic = self.init.stochastic

    def solve(self, inst, seed=0):
        tour = self.init.solve(inst, seed)
        kw = {"time_limit": float(self.params.get("time_limit", np.inf))}
        if inst.n > self.params.get("dense_max_n", 1000):
            kw["k"] = int(self.params.get("k", 10))
        return two_opt(inst, tour, dense_max_n=self.params.get("dense_max_n", 1000), **kw)


@register("exact_dp")
class ExactDP(Solver):
    exact = True

    def solve(self, inst, seed=0):
        if inst.n > 13:
            raise Unsupported("exact_dp is limited to n <= 13")
        return check_tour(held_karp(inst.full_matrix()), inst.n)
