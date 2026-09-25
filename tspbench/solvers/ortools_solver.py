"""Google OR-Tools routing solver (guided local search by default)."""

from __future__ import annotations

import numpy as np

from ..instances import Instance
from .base import Solver, register

_METAHEURISTICS = ("GUIDED_LOCAL_SEARCH", "SIMULATED_ANNEALING", "TABU_SEARCH", "GREEDY_DESCENT", "AUTOMATIC")


def _int_scale(inst: Instance, max_int: float = 1e6) -> float:
    """Scale factor so the largest arc cost is about ``max_int`` (OR-Tools needs integer costs)."""
    if inst.integral:
        return 1.0
    if inst.matrix is not None:
        top = float(inst.matrix.max())
    else:
        ext = inst.coords.max(0) - inst.coords.min(0)
        top = float(np.abs(ext).sum()) or 1.0  # >= every euclidean/manhattan/chebyshev distance
    return max_int / top if top > 0 else 1.0


@register("ortools")
class ORTools(Solver):
    """``ortools:time_limit=<s>,metaheuristic=GUIDED_LOCAL_SEARCH,dense_max_n=3000``.

    OR-Tools' search is not seeded; results can vary slightly with machine
    load because the stopping rule is a wall-clock limit.
    """

    @classmethod
    def available(cls):
        try:
            import ortools  # noqa: F401
        except ImportError:
            return False, "pip install ortools"
        return True, ""

    def solve(self, inst, seed=0):
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2

        n = inst.n
        scale = _int_scale(inst)
        manager = pywrapcp.RoutingIndexManager(n, 1, 0)
        routing = pywrapcp.RoutingModel(manager)
        if n <= self.params.get("dense_max_n", 3000):
            mat = np.rint(inst.full_matrix() * scale).astype(np.int64).tolist()

            def cost(i, j):
                return mat[manager.IndexToNode(i)][manager.IndexToNode(j)]
        else:

            def cost(i, j):
                return int(round(float(inst.dist(manager.IndexToNode(i), manager.IndexToNode(j))) * scale))

        cb = routing.RegisterTransitCallback(cost)
        routing.SetArcCostEvaluatorOfAllVehicles(cb)
        p = pywrapcp.DefaultRoutingSearchParameters()
        p.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        meta = str(self.params.get("metaheuristic", "GUIDED_LOCAL_SEARCH")).upper()
        if meta not in _METAHEURISTICS:
            raise ValueError(f"metaheuristic must be one of {_METAHEURISTICS}")
        p.local_search_metaheuristic = getattr(routing_enums_pb2.LocalSearchMetaheuristic, meta)
        p.time_limit.FromMilliseconds(int(1000 * float(self.params.get("time_limit", 1.0))))
        sol = routing.SolveWithParameters(p)
        if sol is None:
            raise RuntimeError("OR-Tools found no solution")
        tour, idx = [], routing.Start(0)
        while not routing.IsEnd(idx):
            tour.append(manager.IndexToNode(idx))
            idx = sol.Value(routing.NextVar(idx))
        return np.array(tour, dtype=np.int64)
