"""Baseline solvers and model adapters. See ``base`` for the spec-string format."""

from .base import Solver, SolverUnavailable, Unsupported, make_solver, parse_spec, register, registry

_LOADED = False


def load_all():
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    from . import concorde, heuristics, lkh, matnet, model, ortools_solver  # noqa: F401


__all__ = ["Solver", "SolverUnavailable", "Unsupported", "make_solver", "parse_spec", "register", "registry", "load_all"]
