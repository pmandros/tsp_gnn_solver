"""Adapters that plug any model into the harness without the harness knowing about it.

Two entry points, both loaded by import path ``package.module:function``:

* ``callable:fn=pkg.mod:solve`` calls ``solve(distance_matrix) -> tour``
  (``input=coords`` or ``input=instance`` passes those instead).
* ``heatmap:fn=pkg.mod:predict,decode=greedy_edge,two_opt=true`` calls
  ``predict(distance_matrix) -> (n, n) edge scores`` and decodes the scores
  here, so decoding strategies can be compared on equal footing.

Extra ``key=value`` pairs in the spec are passed on as keyword arguments
(e.g. ``checkpoint=runs/best.pt``), except the adapter's own keys. If the
function accepts ``seed`` it also receives the run seed.
"""

from __future__ import annotations

import importlib
import inspect

import numpy as np

from ..instances import Instance
from .base import Solver, Unsupported, register
from .heuristics import two_opt

_OWN_KEYS = {"fn", "input", "decode", "two_opt", "time_limit", "name"}


def load_callable(path: str):
    module, _, attr = path.partition(":")
    if not attr:
        raise ValueError("fn must look like 'package.module:function'")
    obj = importlib.import_module(module)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def _call(fn, arg, seed, extra):
    kw = dict(extra)
    try:
        if "seed" in inspect.signature(fn).parameters:
            kw["seed"] = seed
    except (TypeError, ValueError):
        pass
    return fn(arg, **kw)


def _model_input(inst: Instance, kind: str):
    if kind == "matrix":
        return inst.full_matrix()
    if kind == "coords":
        if inst.coords is None:
            raise Unsupported("model wants coordinates but the instance only has a matrix")
        return inst.coords
    if kind == "instance":
        return inst
    raise ValueError("input must be matrix, coords or instance")


class _ModelSolver(Solver):
    def __init__(self, **params):
        super().__init__(**params)
        if "fn" not in params:
            raise ValueError(f"{self.name} needs fn=package.module:function")
        self.fn = load_callable(params["fn"])
        self.extra = {k: v for k, v in params.items() if k not in _OWN_KEYS}
        self.stochastic = bool(params.get("stochastic", False))
        self.extra.pop("stochastic", None)

    @property
    def label(self):
        return self.params.get("name") or super().label


@register("callable")
class CallableSolver(_ModelSolver):
    def solve(self, inst, seed=0):
        tour = _call(self.fn, _model_input(inst, self.params.get("input", "matrix")), seed, self.extra)
        return np.asarray(tour, dtype=np.int64)


# --------------------------------------------------------------------------- #
# Heatmap decoding
# --------------------------------------------------------------------------- #


def greedy_walk(scores: np.ndarray, start: int = 0) -> np.ndarray:
    """Follow the highest-scoring edge to an unvisited node (the notebook's decoder)."""
    n = len(scores)
    visited = np.zeros(n, dtype=bool)
    tour = [start]
    visited[start] = True
    for _ in range(n - 1):
        r = np.where(visited, -np.inf, scores[tour[-1]])
        nxt = int(np.argmax(r))
        tour.append(nxt)
        visited[nxt] = True
    return np.array(tour, dtype=np.int64)


def greedy_edge(scores: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Symmetric greedy edge selection (Kruskal-style, degree <= 2, no subtours).

    Edges are taken in order of ``scores[i, j] + scores[j, i]``; the path
    fragments left at the end are chained by nearest endpoint using ``dist``.
    """
    n = len(scores)
    s = scores + scores.T
    iu, ju = np.triu_indices(n, 1)
    order = np.argsort(-s[iu, ju], kind="stable")
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    deg = np.zeros(n, dtype=np.int64)
    adj = [[] for _ in range(n)]
    added = 0
    for e in order:
        i, j = int(iu[e]), int(ju[e])
        if deg[i] >= 2 or deg[j] >= 2:
            continue
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        parent[ri] = rj
        deg[i] += 1
        deg[j] += 1
        adj[i].append(j)
        adj[j].append(i)
        added += 1
        if added == n - 1:
            break

    # Walk fragments; hop from each fragment's end to the nearest free endpoint.
    seen = np.zeros(n, dtype=bool)
    tour = []
    cur = next(v for v in range(n) if deg[v] <= 1)
    while True:
        prev = -1
        while True:
            tour.append(cur)
            seen[cur] = True
            nxt = [v for v in adj[cur] if v != prev and not seen[v]]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
        if len(tour) == n:
            break
        ends = np.flatnonzero(~seen & (deg <= 1))
        cur = int(ends[np.argmin(dist[cur, ends])])
    return np.array(tour, dtype=np.int64)


@register("heatmap")
class HeatmapSolver(_ModelSolver):
    def solve(self, inst, seed=0):
        scores = np.asarray(_call(self.fn, _model_input(inst, self.params.get("input", "matrix")), seed, self.extra))
        if scores.shape != (inst.n, inst.n):
            raise ValueError(f"heatmap has shape {scores.shape}, expected {(inst.n, inst.n)}")
        decode = self.params.get("decode", "greedy_edge")
        if decode == "greedy_walk":
            tour = greedy_walk(scores)
        elif decode == "greedy_edge":
            if not inst.symmetric:
                raise Unsupported("greedy_edge is symmetric-only; use decode=greedy_walk")
            tour = greedy_edge(scores, inst.full_matrix())
        else:
            raise ValueError("decode must be greedy_walk or greedy_edge")
        if self.params.get("two_opt", False) and inst.symmetric:
            tour = two_opt(inst, tour, time_limit=float(self.params.get("time_limit", np.inf)))
        return tour


def inverse_distance(matrix: np.ndarray) -> np.ndarray:
    """Reference 'model': score = -distance. With greedy_walk this is nearest neighbour."""
    return -np.asarray(matrix, dtype=np.float64)
