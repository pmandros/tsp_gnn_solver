"""Concorde (Applegate et al.), through ``pyconcorde`` or a ``concorde`` binary.

Concorde solves the *integer* problem exactly. Float instances are scaled (see
``_export``) and the returned tour is re-scored in float by the harness, which
fixes the notebook's habit of comparing a float tour against Concorde's
rounded ``optimal_value``.

Asymmetric instances go through the standard 2n-node symmetric transformation
(Jonker & Volgenant 1983); that needs ``(n+1)^2 * max_d < 2^31``, so large or
wide-range ATSP instances are rejected and LKH-3 should be used for them.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import numpy as np

from .base import Solver, SolverUnavailable, Unsupported, register
from ._export import IntProblem, to_int_problem

_INT_MAX = 2**31 - 1


def _concorde_binary():
    return os.environ.get("CONCORDE_BIN") or shutil.which("concorde")


def _have_pyconcorde() -> bool:
    try:
        from concorde.tsp import TSPSolver  # noqa: F401
    except ImportError:
        return False
    return True


def atsp_to_stsp(d: np.ndarray) -> np.ndarray:
    """Symmetric 2n x 2n integer matrix whose optimal tours map to optimal ATSP tours.

    Node ``i`` is paired with a ghost ``n + i`` by a zero-cost edge; arc ``i -> j``
    becomes edge ``(n + i, j)`` with cost ``d[i, j] + M``; all other edges cost
    ``BIG``. Any tour avoiding ``BIG`` edges alternates real and ghost nodes and
    must use all n zero edges, so it reads as ``i -> j -> ...`` in the original.
    """
    d = np.asarray(d, dtype=np.int64)
    n = len(d)
    m = n * int(d.max()) + 1
    big = (n + 1) * m
    s = np.full((2 * n, 2 * n), big, dtype=np.int64)
    s[n:, :n] = d + m  # ghost of i -> j
    s[:n, n:] = (d + m).T
    idx = np.arange(n)
    s[idx, n + idx] = 0
    s[n + idx, idx] = 0
    s[np.arange(2 * n), np.arange(2 * n)] = 0
    return s


def stsp_tour_to_atsp(tour: np.ndarray, n: int) -> np.ndarray:
    """Read the ATSP tour back from a tour of the transformed 2n-node problem."""
    t = [int(v) for v in tour]
    k = t.index(0)
    t = t[k:] + t[:k]
    # The zero edge from 0 goes to its ghost n; follow the direction ghost -> next real node.
    if t[1] != n:
        t = [t[0]] + t[1:][::-1]
    real = [v for v in t if v < n]
    return np.array(real, dtype=np.int64)


@register("concorde")
class Concorde(Solver):
    """``concorde:time_bound=-1,max_int=1e6``. Exact on the scaled integer problem."""

    exact = True

    @classmethod
    def available(cls):
        if _have_pyconcorde() or _concorde_binary():
            return True, ""
        return False, "install pyconcorde (pip install 'pyconcorde @ git+https://github.com/jvkersch/pyconcorde') or set CONCORDE_BIN"

    def solve(self, inst, seed=0):
        prob = to_int_problem(inst, max_int=float(self.params.get("max_int", 1e6)))
        n = inst.n
        asym = prob.asymmetric
        if asym:
            limit = _INT_MAX // ((n + 1) ** 2 + 1)
            m = prob.matrix
            if int(m.max()) > limit:
                if inst.integral:
                    raise Unsupported("ATSP weights too large for Concorde's transformation; use lkh")
                m = np.rint(inst.matrix * (limit / float(inst.matrix.max()))).astype(np.int64)
            prob = IntProblem(2 * n, False, matrix=atsp_to_stsp(m))
        tour = self._run(prob, inst.name, seed)
        return stsp_tour_to_atsp(tour, n) if asym else tour

    def _run(self, prob: IntProblem, name: str, seed: int) -> np.ndarray:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "p.tsp")
            with open(path, "w") as f:
                f.write(prob.tsplib_text(name))
            if _have_pyconcorde() and self.params.get("backend", "pyconcorde") == "pyconcorde":
                from concorde.tsp import TSPSolver

                cwd = os.getcwd()
                os.chdir(tmp)  # Concorde litters the working directory
                try:
                    sol = TSPSolver.from_tspfile(path).solve(
                        time_bound=float(self.params.get("time_bound", -1)), verbose=False, random_seed=int(seed)
                    )
                finally:
                    os.chdir(cwd)
                if not sol.found_tour:
                    raise RuntimeError("Concorde found no tour")
                return np.asarray(sol.tour, dtype=np.int64)
            exe = _concorde_binary()
            if not exe:
                raise SolverUnavailable("concorde binary not found (set CONCORDE_BIN)")
            out = os.path.join(tmp, "p.sol")
            subprocess.run([exe, "-x", "-s", str(int(seed)), "-o", out, path], cwd=tmp, check=True, capture_output=True)
            with open(out) as f:
                vals = f.read().split()
            return np.array(vals[1 : 1 + int(vals[0])], dtype=np.int64)
