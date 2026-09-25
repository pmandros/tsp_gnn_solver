"""Solver interface and the spec-string registry.

A solver spec is ``name`` or ``name:key=value,key=value``, e.g.
``ortools:time_limit=2`` or ``callable:fn=my_pkg.model:solve``. Values are
parsed as int, float or bool when possible. Specs (not solver objects) are
what the evaluation runner ships to worker processes.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple, Type

import numpy as np

from ..instances import Instance


class SolverUnavailable(RuntimeError):
    """Raised when a solver's backend (binary or package) is not installed."""


class Unsupported(ValueError):
    """Raised when a solver cannot handle an instance (e.g. asymmetric, too large)."""


class Solver:
    name: str = "solver"
    #: Stochastic solvers are run once per seed; deterministic ones once in total.
    stochastic: bool = False
    #: Exact solvers produce optimal tours, so they can serve as references.
    exact: bool = False

    def __init__(self, **params):
        self.params = params

    @classmethod
    def available(cls) -> Tuple[bool, str]:
        return True, ""

    def solve(self, inst: Instance, seed: int = 0) -> np.ndarray:
        raise NotImplementedError

    @property
    def label(self) -> str:
        if not self.params:
            return self.name
        return self.name + ":" + ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))


_REGISTRY: Dict[str, Type[Solver]] = {}


def register(name: str) -> Callable[[Type[Solver]], Type[Solver]]:
    def deco(cls):
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def _parse_value(v: str):
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def parse_spec(spec: str) -> Tuple[str, dict]:
    name, _, rest = spec.partition(":")
    params = {}
    if rest:
        for part in rest.split(","):
            k, _, v = part.partition("=")
            params[k.strip()] = _parse_value(v.strip())
    return name.strip(), params


def make_solver(spec: str) -> Solver:
    from . import load_all

    load_all()
    name, params = parse_spec(spec)
    if name not in _REGISTRY:
        raise KeyError(f"unknown solver {name!r}; known: {sorted(_REGISTRY)}")
    cls = _REGISTRY[name]
    ok, why = cls.available()
    if not ok:
        raise SolverUnavailable(f"{name}: {why}")
    return cls(**params)


def registry() -> Dict[str, Type[Solver]]:
    from . import load_all

    load_all()
    return dict(_REGISTRY)
