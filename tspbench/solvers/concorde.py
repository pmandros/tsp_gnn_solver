"""Concorde (Applegate et al.), through ``pyconcorde`` or a ``concorde`` binary.

Concorde solves the *integer* problem exactly. Float instances are scaled (see
``_export``) and the returned tour is re-scored in float by the harness, which
fixes the notebook's habit of comparing a float tour against Concorde's
rounded ``optimal_value``.

Asymmetric instances go through the standard 2n-node symmetric transformation
(Jonker & Volgenant 1983). Its penalty weights grow like ``n * tour length``;
Concorde segfaults well before weights reach 2^31, so the matrix is scaled to keep
every weight below ``_MAX_WEIGHT``. Integer instances that would need scaling
are rejected (use LKH-3); float instances lose a little precision, so Concorde
on generated ATSP is near-exact rather than exact.

A crashed Concorde prints ``FATAL ERROR`` and then sleeps for an hour "to permit
debugger access"; the binary backend watches for that and kills it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import numpy as np

from .base import Solver, SolverUnavailable, Unsupported, register
from ._export import IntProblem, to_int_problem

_MAX_WEIGHT = 2**25  # Concorde segfaulted on a 50-node ATSP between 2^26 and 2^29


def _concorde_binary():
    return os.environ.get("CONCORDE_BIN") or shutil.which("concorde")


def _have_pyconcorde() -> bool:
    try:
        from concorde.tsp import TSPSolver  # noqa: F401
    except ImportError:
        return False
    return True


def _nn_upper_bound(d: np.ndarray) -> int:
    n = len(d)
    seen = np.zeros(n, dtype=bool)
    cur, total = 0, 0
    for _ in range(n - 1):
        seen[cur] = True
        row = np.where(seen, np.iinfo(np.int64).max, d[cur])
        nxt = int(np.argmin(row))
        total += int(d[cur, nxt])
        cur = nxt
    return total + int(d[cur, 0])


def atsp_penalties(d: np.ndarray):
    """(M, BIG) for :func:`atsp_to_stsp`: M exceeds any tour's length, BIG any valid 2n-tour."""
    n = len(d)
    ub = _nn_upper_bound(d)
    m = ub + 1
    return m, n * m + ub + 1


def atsp_to_stsp(d: np.ndarray) -> np.ndarray:
    """Symmetric 2n x 2n integer matrix whose optimal tours map to optimal ATSP tours.

    Node ``i`` is paired with a ghost ``n + i`` by a zero-cost edge; arc ``i -> j``
    becomes edge ``(n + i, j)`` with cost ``d[i, j] + M``; all other edges cost
    ``BIG``. With ``M`` above an upper bound on the ATSP optimum, a tour that
    skips a zero edge costs at least ``(n + 1) M``, more than any real tour
    (``n M + length``); ``BIG`` is above that too. So optimal tours alternate
    real and ghost nodes, use every zero edge, and read as ``i -> j -> ...``.
    """
    d = np.asarray(d, dtype=np.int64)
    n = len(d)
    m, big = atsp_penalties(d)
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
    """``concorde:backend=binary|pyconcorde,time_bound=-1,max_int=1e6``. Exact on the scaled integer problem.

    The binary backend (``CONCORDE_BIN``) is preferred when set; it is the one tested in CI.
    """

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
            m = prob.matrix
            big = atsp_penalties(m)[1]
            if big > _MAX_WEIGHT:
                if inst.integral:
                    raise Unsupported("ATSP weights too large for Concorde's transformation; use lkh")
                m = np.rint(m * (0.9 * _MAX_WEIGHT / big)).astype(np.int64)
            prob = IntProblem(2 * n, False, matrix=atsp_to_stsp(m))
        tour = self._run(prob, inst.name, seed)
        return stsp_tour_to_atsp(tour, n) if asym else tour

    def _run(self, prob: IntProblem, name: str, seed: int) -> np.ndarray:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "p.tsp")
            with open(path, "w") as f:
                f.write(prob.tsplib_text(name))
            backend = self.params.get("backend", "binary" if _concorde_binary() else "pyconcorde")
            if backend == "pyconcorde" and _have_pyconcorde():
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
            # No -x: with it Concorde exits 255 after solving (it fails to delete its
            # temp files); the temporary directory is cleaned up anyway.
            proc = subprocess.Popen([exe, "-s", str(int(seed)), "-o", out, path], cwd=tmp, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            tail = []
            for line in proc.stdout:
                tail = (tail + [line])[-20:]
                if "FATAL ERROR" in line:
                    proc.kill()
                    break
            proc.wait()
            if proc.returncode != 0 or not os.path.exists(out):
                raise RuntimeError(f"concorde exited {proc.returncode}: {''.join(tail)[-600:]}")
            with open(out) as f:
                vals = f.read().split()
            return np.array(vals[1 : 1 + int(vals[0])], dtype=np.int64)
