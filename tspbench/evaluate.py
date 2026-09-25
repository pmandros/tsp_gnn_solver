"""Evaluation runner: solvers x suites x seeds -> per-instance records and summaries.

Every tour is validated and re-scored with the instance's own float distance
function. Gaps are reported against the instance reference (published optimum,
the reference tour shipped with the test file, or a cached reference-solver
run; see ``suites.reference_for``) and, separately, against the best tour any
solver found in this run.

Stochastic solvers run once per seed. The summary reports, per suite and
solver, the mean gap over instances for each seed, then the mean and a 95%
t-interval across seeds. Deterministic solvers run once, and their interval is
taken across instances instead (column ``ci_over``).
"""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import subprocess
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .instances import Instance, check_tour
from .solvers import SolverUnavailable, Unsupported, make_solver
from .suites import Suite, load_references, reference_for

RECORD_FIELDS = [
    "suite", "instance", "n", "solver", "seed", "status", "length", "reference",
    "reference_source", "gap", "gap_best", "time_s", "error",
]

_SOLVER_CACHE: Dict[str, object] = {}


def _solver(spec: str):
    if spec not in _SOLVER_CACHE:
        _SOLVER_CACHE[spec] = make_solver(spec)
    return _SOLVER_CACHE[spec]


def run_one(spec: str, inst: Instance, seed: int) -> Dict:
    rec = {"instance": inst.name, "n": inst.n, "solver": spec, "seed": seed, "status": "ok",
           "length": None, "time_s": None, "error": ""}
    try:
        solver = _solver(spec)
        t0 = time.perf_counter()
        tour = solver.solve(inst, seed=seed)
        rec["time_s"] = time.perf_counter() - t0
        rec["length"] = inst.tour_length(check_tour(tour, inst.n))
    except Unsupported as e:
        rec.update(status="unsupported", error=str(e))
    except SolverUnavailable as e:
        rec.update(status="unavailable", error=str(e))
    except ValueError as e:
        rec.update(status="invalid" if "invalid tour" in str(e) else "error", error=str(e))
    except Exception as e:  # keep the run going; the record carries the failure
        rec.update(status="error", error=f"{type(e).__name__}: {e}")
        if os.environ.get("TSPBENCH_DEBUG"):
            traceback.print_exc()
    return rec


def _run_task(args):
    return run_one(*args)


def _tasks(suite: Suite, specs: Sequence[str], seeds: Sequence[int]):
    for spec in specs:
        stochastic = _solver(spec).stochastic
        for s in (seeds if stochastic else seeds[:1]):
            for inst in suite.instances:
                yield spec, inst, int(s)


def evaluate(
    suites: Iterable[Suite],
    specs: Sequence[str],
    seeds: Sequence[int] = (0,),
    workers: int = 1,
    data_dir: str = "data",
    progress: bool = True,
) -> List[Dict]:
    for spec in specs:  # fail fast on typos and missing backends
        _solver(spec)
    records: List[Dict] = []
    for suite in suites:
        tasks = list(_tasks(suite, specs, seeds))
        t0 = time.time()
        if workers > 1:
            with ProcessPoolExecutor(workers) as ex:
                out = []
                for k, r in enumerate(ex.map(_run_task, tasks, chunksize=max(1, len(tasks) // (workers * 16)))):
                    out.append(r)
                    if progress and (k + 1) % max(1, len(tasks) // 10) == 0:
                        print(f"  {suite.name}: {k + 1}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)
        else:
            out = []
            for k, t in enumerate(tasks):
                out.append(run_one(*t))
                if progress and (k + 1) % max(1, len(tasks) // 10) == 0:
                    print(f"  {suite.name}: {k + 1}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)
        refs = load_references(data_dir, suite.name)
        by_name = {i.name: i for i in suite.instances}
        best: Dict[str, float] = {}
        for r in out:
            r["suite"] = suite.name
            ref, src = reference_for(by_name[r["instance"]], refs)
            r["reference"], r["reference_source"] = ref, src
            cands = [x for x in (ref, r["length"]) if x is not None]
            if cands:
                best[r["instance"]] = min([best.get(r["instance"], math.inf)] + cands)
        for r in out:
            if r["length"] is not None:
                r["gap"] = r["length"] / r["reference"] - 1.0 if r["reference"] else None
                b = best.get(r["instance"])
                r["gap_best"] = r["length"] / b - 1.0 if b else None
            else:
                r["gap"] = r["gap_best"] = None
        records.extend(out)
    return records


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #


def _t95(dof: int) -> float:
    try:
        from scipy.stats import t

        return float(t.ppf(0.975, dof))
    except ImportError:
        return 1.96


def _mean_ci(values: Sequence[float]):
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0:
        return None, None
    if len(v) == 1:
        return float(v[0]), None
    return float(v.mean()), _t95(len(v) - 1) * float(v.std(ddof=1)) / math.sqrt(len(v))


def summarize(records: Sequence[Dict]) -> List[Dict]:
    groups = defaultdict(list)
    for r in records:
        groups[(r["suite"], r["solver"])].append(r)
    rows = []
    for (suite, solver), recs in groups.items():
        ok = [r for r in recs if r["status"] == "ok"]
        instances = {r["instance"] for r in recs}
        seeds = sorted({r["seed"] for r in recs})
        row = {
            "suite": suite, "solver": solver, "instances": len(instances), "seeds": len(seeds),
            "ok": len(ok), "failed": len(recs) - len(ok),
            "failures": ";".join(sorted({r["status"] for r in recs if r["status"] != "ok"})),
        }
        # A mean gap is only meaningful when every instance was solved for that seed.
        complete = [s for s in seeds if sum(1 for r in ok if r["seed"] == s) == len(instances)]
        for key in ("gap", "gap_best"):
            per_seed = []
            for s in complete:
                vals = [r[key] for r in ok if r["seed"] == s]
                if vals and all(v is not None for v in vals):
                    per_seed.append(float(np.mean(vals)))
            if len(per_seed) >= 2:
                mean, ci = _mean_ci(per_seed)
                row["ci_over"] = "seeds"
            elif len(per_seed) == 1:
                vals = [r[key] for r in ok if r["seed"] == complete[0]]
                mean, ci = _mean_ci(vals)
                row["ci_over"] = "instances"
            else:
                mean, ci = None, None
            row[f"{key}_mean"], row[f"{key}_ci95"] = mean, ci
        if ok:
            row["length_mean"] = float(np.mean([r["length"] for r in ok]))
            row["time_mean_s"] = float(np.mean([r["time_s"] for r in ok]))
            row["time_total_s"] = float(np.sum([r["time_s"] for r in ok]))
        refs = sorted({str(r["reference_source"]) for r in recs if r["reference_source"]})
        row["reference"] = "+".join(refs) if refs else "none"
        rows.append(row)
    return rows


SUMMARY_FIELDS = [
    "suite", "solver", "instances", "seeds", "ok", "failed", "failures", "reference", "ci_over",
    "gap_mean", "gap_ci95", "gap_best_mean", "gap_best_ci95", "length_mean", "time_mean_s", "time_total_s",
]


def _pct(v):
    return "" if v is None else f"{100 * v:.3f}%"


def summary_markdown(rows: Sequence[Dict]) -> str:
    lines = [
        "| suite | solver | n inst | seeds | gap vs ref | gap vs best in run | time / inst | ref | failed |",
        "|---|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for r in rows:
        def cell(key):
            m, ci = r.get(f"{key}_mean"), r.get(f"{key}_ci95")
            if m is None:
                return "n/a"
            return _pct(m) + (f" ± {_pct(ci)}" if ci is not None else "")

        t = r.get("time_mean_s")
        lines.append(
            f"| {r['suite']} | `{r['solver']}` | {r['instances']} | {r['seeds']} | {cell('gap')} | "
            f"{cell('gap_best')} | {'' if t is None else f'{t:.3g}s'} | {r['reference']} | {r['failed']} |"
        )
    lines.append("")
    lines.append("± is a 95% t-interval across seeds for stochastic solvers, across instances otherwise.")
    return "\n".join(lines) + "\n"


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def write_results(out_dir: str, records: Sequence[Dict], rows: Sequence[Dict], config: Dict):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "records.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, RECORD_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(summary_markdown(rows))
    cfg = dict(config)
    cfg.update(git_commit=_git_commit(), python=platform.python_version(), platform=platform.platform(),
               cpu_count=os.cpu_count(), numpy=np.__version__)
    with open(os.path.join(out_dir, "run.json"), "w") as f:
        json.dump(cfg, f, indent=2)
