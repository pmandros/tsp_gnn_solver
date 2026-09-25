"""Readers and writers for TSPLIB95 files and the text format of the learned-TSP test sets.

The text format (one instance per line, ``x1 y1 x2 y2 ... output t1 t2 ... tn t1``
with 1-indexed tours) is the one used by Joshi et al. (2019) for TSP20/50/100
and by Fu et al. (2021) for TSP500/1000/10000, and reused by DIMES, DIFUSCO and
most later work.
"""

from __future__ import annotations

import gzip
import os
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from .instances import TSPLIB_METRICS, Instance


def _open(path: str):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def _is_number(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


def _read_sections(path: str) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    header: Dict[str, str] = {}
    sections: Dict[str, List[str]] = {}
    current: Optional[str] = None
    with _open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            first = line.split()[0]
            if not _is_number(first) and first != "-1":
                key = line.split(":", 1)[0].strip().upper()
                if key == "EOF":
                    break
                if key.endswith("_SECTION"):
                    current = key
                    sections[current] = []
                    rest = line[len(first):].replace(":", " ").split()
                    sections[current].extend(rest)
                else:
                    current = None
                    if ":" in line:
                        header[key] = line.split(":", 1)[1].strip()
                continue
            if current is not None:
                sections[current].extend(line.split())
    return header, sections


def _triangle_indices(n: int, fmt: str) -> Iterator[Tuple[int, int]]:
    # Column-wise formats of a symmetric matrix enumerate the same entries as
    # the opposite row-wise format.
    fmt = {
        "UPPER_COL": "LOWER_ROW",
        "LOWER_COL": "UPPER_ROW",
        "UPPER_DIAG_COL": "LOWER_DIAG_ROW",
        "LOWER_DIAG_COL": "UPPER_DIAG_ROW",
    }.get(fmt, fmt)
    for i in range(n):
        if fmt == "UPPER_ROW":
            js = range(i + 1, n)
        elif fmt == "LOWER_ROW":
            js = range(0, i)
        elif fmt == "UPPER_DIAG_ROW":
            js = range(i, n)
        elif fmt == "LOWER_DIAG_ROW":
            js = range(0, i + 1)
        else:
            raise ValueError(f"unsupported EDGE_WEIGHT_FORMAT {fmt}")
        for j in js:
            yield i, j


def read_tsplib(path: str, optimum: Optional[float] = None) -> Instance:
    """Parse a TSPLIB95 ``.tsp`` / ``.atsp`` file (optionally gzipped)."""
    header, sections = _read_sections(path)
    name = header.get("NAME", os.path.basename(path).split(".")[0]).strip()
    n = int(header["DIMENSION"])
    ewt = header.get("EDGE_WEIGHT_TYPE", "EXPLICIT").upper()
    ptype = header.get("TYPE", "TSP").split()[0].upper()
    meta = {"family": "tsplib", "type": ptype, "edge_weight_type": ewt, "path": str(path)}
    if ewt == "EXPLICIT":
        fmt = header.get("EDGE_WEIGHT_FORMAT", "FULL_MATRIX").upper()
        vals = np.array(sections["EDGE_WEIGHT_SECTION"], dtype=np.float64)
        if fmt == "FULL_MATRIX":
            m = vals[: n * n].reshape(n, n)
        else:
            m = np.zeros((n, n))
            idx = list(_triangle_indices(n, fmt))
            if len(vals) < len(idx):
                raise ValueError(f"{path}: expected {len(idx)} weights, found {len(vals)}")
            for (i, j), v in zip(idx, vals):
                m[i, j] = m[j, i] = v
        m = m.copy()
        np.fill_diagonal(m, 0.0)  # some ATSP files store 9999999 on the diagonal
        meta["edge_weight_format"] = fmt
        coords = None
        if "DISPLAY_DATA_SECTION" in sections:
            coords = np.array(sections["DISPLAY_DATA_SECTION"], dtype=np.float64).reshape(n, -1)[:, 1:3]
        inst = Instance(name, matrix=m, optimum=optimum, meta=meta)
        inst.meta["display_coords"] = coords
        return inst
    if ewt not in TSPLIB_METRICS:
        raise ValueError(f"{path}: unsupported EDGE_WEIGHT_TYPE {ewt}")
    vals = np.array(sections["NODE_COORD_SECTION"], dtype=np.float64).reshape(n, -1)
    return Instance(name, coords=vals[:, 1:3], metric=ewt, optimum=optimum, meta=meta)


def read_tsplib_tour(path: str) -> np.ndarray:
    """Parse a TSPLIB ``.tour`` file into a 0-indexed tour."""
    _, sections = _read_sections(path)
    ids = [int(float(t)) for t in sections["TOUR_SECTION"]]
    if -1 in ids:
        ids = ids[: ids.index(-1)]
    return np.array(ids, dtype=np.int64) - 1


def write_tsplib(path: str, name: str, n: int, *, coords=None, metric: str = "EUC_2D", matrix=None, asymmetric=False):
    """Write a TSPLIB file. ``matrix`` must already be integer valued."""
    with open(path, "w") as f:
        f.write(f"NAME : {name}\nTYPE : {'ATSP' if asymmetric else 'TSP'}\nDIMENSION : {n}\n")
        if matrix is not None:
            f.write("EDGE_WEIGHT_TYPE : EXPLICIT\nEDGE_WEIGHT_FORMAT : FULL_MATRIX\nEDGE_WEIGHT_SECTION\n")
            for row in np.asarray(matrix, dtype=np.int64):
                f.write(" ".join(map(str, row)) + "\n")
        else:
            f.write(f"EDGE_WEIGHT_TYPE : {metric}\nNODE_COORD_SECTION\n")
            for i, (x, y) in enumerate(np.asarray(coords)):
                f.write(f"{i + 1} {x:.10g} {y:.10g}\n")
        f.write("EOF\n")


def read_text_tsp(path: str, limit: Optional[int] = None, prefix: Optional[str] = None) -> List[Instance]:
    """Read the Joshi / Fu et al. text format.

    When a line carries a reference tour, its float Euclidean length becomes
    ``meta["reference_length"]`` and the tour ``meta["reference_tour"]``.
    """
    prefix = prefix or os.path.basename(path).split(".")[0]
    out = []
    with _open(path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            if " output " in f" {line} ":
                left, right = line.split("output", 1)
            else:
                left, right = line, ""
            coords = np.array(left.split(), dtype=np.float64).reshape(-1, 2)
            inst = Instance(f"{prefix}_{i:05d}", coords=coords, meta={"family": "uniform", "source": str(path)})
            if right.strip():
                tour = np.array(right.split(), dtype=np.int64) - 1
                if len(tour) == inst.n + 1:
                    tour = tour[:-1]
                # Some published files carry placeholder tours (TSP10000 ships the
                # identity permutation); keep those out of the references.
                if not np.array_equal(tour, np.arange(inst.n)):
                    inst.meta["reference_tour"] = tour
                    inst.meta["reference_length"] = inst.tour_length(tour)
            out.append(inst)
    return out


def write_text_tsp(path: str, instances: List[Instance], tours: Optional[List[np.ndarray]] = None):
    with open(path, "w") as f:
        for k, inst in enumerate(instances):
            line = " ".join(f"{v:.10g}" for v in inst.coords.ravel())
            if tours is not None:
                t = np.asarray(tours[k]) + 1
                line += " output " + " ".join(map(str, list(t) + [t[0]]))
            f.write(line + "\n")
