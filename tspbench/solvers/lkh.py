"""LKH-3 (Helsgaun), through the ``elkai`` package or an ``LKH`` binary.

``elkai`` ships LKH-3.0.8 compiled in, so ``pip install elkai`` is enough. Set
``backend=binary`` (and ``LKH_BIN`` or put ``LKH`` on PATH) to use a separately
built LKH-3 release instead. Handles symmetric and asymmetric instances.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import numpy as np

from .base import Solver, SolverUnavailable, register
from ._export import to_int_problem


def _lkh_binary():
    return os.environ.get("LKH_BIN") or shutil.which("LKH")


def _have_elkai() -> bool:
    try:
        import elkai._elkai  # noqa: F401
    except ImportError:
        return False
    return True


@register("lkh")
class LKH(Solver):
    """``lkh:runs=1,max_trials=<n>,time_limit=<s>,backend=elkai|binary``.

    Seeded: LKH's ``SEED`` is set from the run seed, so ``lkh`` is run once per seed.
    """

    stochastic = True

    @classmethod
    def available(cls):
        if _have_elkai() or _lkh_binary():
            return True, ""
        return False, "pip install elkai, or build LKH-3 and set LKH_BIN"

    def _params(self, n: int, seed: int) -> str:
        p = {
            "RUNS": int(self.params.get("runs", 1)),
            "MAX_TRIALS": int(self.params.get("max_trials", n)),
            "SEED": int(seed) + 1,  # LKH treats SEED = 0 as "use the time"
            "TRACE_LEVEL": 0,
        }
        if "time_limit" in self.params:
            p["TIME_LIMIT"] = float(self.params["time_limit"])
        if "max_candidates" in self.params:
            p["MAX_CANDIDATES"] = int(self.params["max_candidates"])
        return "".join(f"{k} = {v}\n" for k, v in p.items())

    def solve(self, inst, seed=0):
        prob = to_int_problem(inst, max_int=float(self.params.get("max_int", 1e6)))
        text = prob.tsplib_text(inst.name)
        backend = self.params.get("backend", "elkai" if _have_elkai() else "binary")
        if backend == "elkai":
            from elkai import _elkai

            tour = _elkai.solve_problem(self._params(inst.n, seed) + "PROBLEM_FILE = :stdin:\n", text)
            return np.asarray(tour, dtype=np.int64) - 1
        exe = _lkh_binary()
        if not exe:
            raise SolverUnavailable("LKH binary not found (set LKH_BIN)")
        with tempfile.TemporaryDirectory() as tmp:
            prob_path = os.path.join(tmp, "p.tsp")
            tour_path = os.path.join(tmp, "p.tour")
            par_path = os.path.join(tmp, "p.par")
            with open(prob_path, "w") as f:
                f.write(text)
            with open(par_path, "w") as f:
                f.write(f"PROBLEM_FILE = {prob_path}\nOUTPUT_TOUR_FILE = {tour_path}\n" + self._params(inst.n, seed))
            subprocess.run([exe, par_path], check=True, stdin=subprocess.DEVNULL, capture_output=True)
            from ..io import read_tsplib_tour

            return read_tsplib_tour(tour_path)
