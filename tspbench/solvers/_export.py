"""Turn an instance into the integer TSPLIB problem LKH and Concorde need.

TSPLIB-metric instances are passed through unchanged, so published optima stay
comparable. Float coordinates are scaled so the bounding box spans ``max_int``
and written with the matching TSPLIB rounding metric; matrices are scaled and
rounded. The solver's tour is then re-scored in float by the caller, so the
rounding only costs a relative error of roughly ``n / max_int`` per tour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..instances import TSPLIB_METRICS, Instance

_FLOAT_TO_TSPLIB = {"euclidean": "EUC_2D", "manhattan": "MAN_2D", "chebyshev": "MAX_2D"}


@dataclass
class IntProblem:
    n: int
    asymmetric: bool
    metric: str = "EXPLICIT"
    coords: Optional[np.ndarray] = None
    matrix: Optional[np.ndarray] = None

    def tsplib_text(self, name: str = "problem") -> str:
        lines = [
            f"NAME : {name}",
            f"TYPE : {'ATSP' if self.asymmetric else 'TSP'}",
            f"DIMENSION : {self.n}",
        ]
        if self.matrix is not None:
            lines += ["EDGE_WEIGHT_TYPE : EXPLICIT", "EDGE_WEIGHT_FORMAT : FULL_MATRIX", "EDGE_WEIGHT_SECTION"]
            lines += [" ".join(map(str, row)) for row in self.matrix.tolist()]
        else:
            lines += [f"EDGE_WEIGHT_TYPE : {self.metric}", "NODE_COORD_SECTION"]
            lines += [f"{i + 1} {x:.10g} {y:.10g}" for i, (x, y) in enumerate(self.coords.tolist())]
        lines.append("EOF")
        return "\n".join(lines) + "\n"


def to_int_problem(inst: Instance, max_int: float = 1e6) -> IntProblem:
    if inst.matrix is None and inst.metric in TSPLIB_METRICS:
        return IntProblem(inst.n, False, metric=inst.metric, coords=inst.coords)
    if inst.matrix is None:
        c = inst.coords - inst.coords.min(0)
        ext = float(c.max()) or 1.0
        return IntProblem(inst.n, False, metric=_FLOAT_TO_TSPLIB[inst.metric], coords=np.rint(c * (max_int / ext)))
    m = inst.matrix
    if not inst.integral:
        top = float(m.max()) or 1.0
        m = m * (max_int / top)
    return IntProblem(inst.n, not inst.symmetric, matrix=np.rint(m).astype(np.int64))
