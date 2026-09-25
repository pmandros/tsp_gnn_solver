"""Named benchmark suites.

Suite specs use the same ``name:key=value`` syntax as solvers. Built in:

=====================  =====================================================
``tsp20|50|100``       Kool et al. test sets, 10,000 instances (regenerated
                       bit-for-bit from seed 1234)
``tsp500|1000|10000``  Fu et al. test sets (128 / 128 / 16 instances) read
                       from ``<data>/fu/tsp{n}_test_concorde.txt``
``tsp{n}_gen``         same sizes regenerated from seed 1234 when the files
                       are unavailable (not the published instances)
``tsplib``             every ``<data>/tsplib/*.tsp[.gz]`` with a published
                       optimum; ``min_n=`` / ``max_n=`` filter by size
``atsp_tsplib``        every ``<data>/tsplib_atsp/*.atsp[.gz]`` with an optimum
``uniform{n}``         uniform unit square, Euclidean
``manhattan{n}``,      uniform unit square, L1 / L-infinity
``chebyshev{n}``
``clustered{n}``       Gaussian mixture (``num_clusters=``, ``std=``)
``nonmetric{n}``       symmetric U(0,1) matrices, no triangle inequality
``atsp{n}``            MatNet-style asymmetric matrices with the triangle
                       inequality
``text:path=F``        any file in the Joshi / Fu text format
``tsplib_dir:path=D``  any directory of TSPLIB files
=====================  =====================================================

Generated families take ``num=`` and ``seed=`` (default 1234). ``num``
defaults to 1000 for n <= 100, 128 for n <= 1000 and 16 above.
"""

from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import generators as gen
from .instances import Instance
from .io import read_text_tsp, read_tsplib
from .solvers.base import parse_spec
from .tsplib_optima import optimum as tsplib_optimum

KOOL_SIZES = (20, 50, 100)
FU_SIZES = {500: 128, 1000: 128, 10000: 16}
_FAMILY = re.compile(r"^(uniform|manhattan|chebyshev|clustered|nonmetric|atsp)(\d+)$")


@dataclass
class Suite:
    name: str
    instances: List[Instance]
    description: str = ""
    params: Dict = field(default_factory=dict)


def default_count(n: int) -> int:
    return 1000 if n <= 100 else 128 if n <= 1000 else 16


def _tsplib_dir(path: str, min_n: int, max_n: int, require_optimum: bool = True) -> List[Instance]:
    files = sorted(glob.glob(os.path.join(path, "*.tsp")) + glob.glob(os.path.join(path, "*.tsp.gz"))
                   + glob.glob(os.path.join(path, "*.atsp")) + glob.glob(os.path.join(path, "*.atsp.gz")))
    if not files:
        raise FileNotFoundError(f"no TSPLIB files in {path}; run scripts/download_benchmarks.py or see BENCHMARKS.md")
    out = []
    for f in files:
        name = os.path.basename(f).split(".")[0]
        opt = tsplib_optimum(name)
        if require_optimum and opt is None:
            continue
        try:
            inst = read_tsplib(f, optimum=opt)
        except (ValueError, KeyError) as e:  # e.g. EUC_3D, special formats
            print(f"skipping {f}: {e}")
            continue
        if min_n <= inst.n <= max_n:
            out.append(inst)
    return sorted(out, key=lambda i: (i.n, i.name))


def load_suite(spec: str, data_dir: str = "data", limit: Optional[int] = None) -> Suite:
    name, p = parse_spec(spec)
    seed = int(p.get("seed", gen.KOOL_TEST_SEED))
    m = re.fullmatch(r"tsp(\d+)(_gen)?", name)
    if m and not m.group(2) and int(m.group(1)) in KOOL_SIZES:
        n = int(m.group(1))
        num = int(p.get("num", 10000))
        insts = gen.kool_uniform(n, num, seed=seed)
        desc = f"Kool et al. uniform TSP{n} test set ({num} instances, seed {seed})"
    elif m and not m.group(2) and int(m.group(1)) in FU_SIZES:
        n = int(m.group(1))
        path = p.get("path", os.path.join(data_dir, "fu", f"tsp{n}_test_concorde.txt"))
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found; see BENCHMARKS.md, or use suite tsp{n}_gen")
        insts = read_text_tsp(path, limit=limit, prefix=f"tsp{n}")
        desc = f"Fu et al. TSP{n} test set ({path})"
    elif m and m.group(2):
        n = int(m.group(1))
        num = int(p.get("num", FU_SIZES.get(n, default_count(n))))
        insts = gen.kool_uniform(n, num, seed=seed, prefix=f"tsp{n}_gen")
        desc = f"uniform TSP{n}, {num} instances regenerated from seed {seed}"
    elif name in ("tsplib", "atsp_tsplib", "tsplib_dir"):
        default = {"tsplib": "tsplib", "atsp_tsplib": "tsplib_atsp"}.get(name)
        path = p.get("path") or os.path.join(data_dir, default)
        insts = _tsplib_dir(path, int(p.get("min_n", 0)), int(p.get("max_n", 10**9)),
                            require_optimum=name != "tsplib_dir")
        desc = f"TSPLIB instances from {path}"
    elif name == "text":
        insts = read_text_tsp(p["path"], limit=limit)
        desc = f"text-format instances from {p['path']}"
    elif _FAMILY.match(name):
        family, n = _FAMILY.match(name).groups()
        n = int(n)
        num = int(p.get("num", default_count(n)))
        if family in ("uniform", "manhattan", "chebyshev"):
            metric = "euclidean" if family == "uniform" else family
            insts = gen.uniform(n, num, seed, metric=metric, prefix=name)
        elif family == "clustered":
            insts = gen.clustered(n, num, seed, int(p.get("num_clusters", 3)), float(p.get("std", 0.07)), prefix=name)
        elif family == "nonmetric":
            insts = gen.nonmetric(n, num, seed, prefix=name)
        else:
            insts = gen.atsp_tmat(n, num, seed, prefix=name)
        desc = f"{family}, n={n}, {num} instances, seed {seed}"
    else:
        raise KeyError(f"unknown suite {spec!r}; see tspbench.suites docstring")
    if limit is not None:
        insts = insts[:limit]
    return Suite(spec, insts, desc, p)


# --------------------------------------------------------------------------- #
# Reference tour lengths for suites without published optima
# --------------------------------------------------------------------------- #


def _ref_path(data_dir: str, suite: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.=-]+", "_", suite)
    return os.path.join(data_dir, "refs", f"{safe}.json")


def load_references(data_dir: str, suite: str) -> Dict:
    path = _ref_path(data_dir, suite)
    if not os.path.exists(path):
        return {"solver": None, "lengths": {}}
    with open(path) as f:
        return json.load(f)


def save_references(data_dir: str, suite: str, solver: str, lengths: Dict[str, float]):
    path = _ref_path(data_dir, suite)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    old = load_references(data_dir, suite)
    if old["solver"] not in (None, solver):
        old = {"solver": solver, "lengths": {}}  # never mix references from two solvers
    old["solver"] = solver
    old["lengths"].update(lengths)
    with open(path, "w") as f:
        json.dump(old, f, indent=0, sort_keys=True)
    return path


def reference_for(inst: Instance, refs: Dict):
    """(length, source) for an instance: published optimum, file tour, or cached reference solver."""
    if inst.optimum is not None:
        return float(inst.optimum), "optimum"
    if inst.name in refs.get("lengths", {}):
        return float(refs["lengths"][inst.name]), refs["solver"]
    if "reference_length" in inst.meta:
        return float(inst.meta["reference_length"]), "file"
    return None, None
