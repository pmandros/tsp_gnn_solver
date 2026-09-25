"""Command line: ``python -m tspbench {list,run,reference,export}``."""

from __future__ import annotations

import argparse
import sys
import time

from .evaluate import evaluate, summarize, summary_markdown, write_results
from .io import write_text_tsp
from .solvers import registry
from .suites import load_suite, save_references


def _suites(args):
    for spec in args.suite:
        s = load_suite(spec, data_dir=args.data_dir, limit=args.limit)
        print(f"{s.name}: {len(s.instances)} instances. {s.description}")
        yield s


def cmd_list(args):
    print("Solvers:")
    for name, cls in sorted(registry().items()):
        ok, why = cls.available()
        kind = "exact" if cls.exact else "stochastic" if cls.stochastic else "deterministic"
        print(f"  {name:20s} {kind:13s} {'available' if ok else 'MISSING: ' + why}")
    print("\nSuites: see `python -c 'import tspbench.suites as s; print(s.__doc__)'`")


def cmd_run(args):
    suites = list(_suites(args))
    t0 = time.time()
    records = evaluate(suites, args.solver, seeds=args.seeds, workers=args.workers, data_dir=args.data_dir)
    rows = summarize(records)
    out = args.out or time.strftime("results/%Y%m%d-%H%M%S")
    write_results(out, records, rows, {"argv": sys.argv, "suites": args.suite, "solvers": args.solver,
                                       "seeds": args.seeds, "limit": args.limit, "workers": args.workers,
                                       "wall_s": time.time() - t0})
    print()
    print(summary_markdown(rows))
    print(f"wrote {out}/records.csv, summary.csv, summary.md, run.json")


def cmd_reference(args):
    if len(args.solver) != 1:
        sys.exit("reference takes exactly one --solver")
    spec = args.solver[0]
    for suite in _suites(args):
        recs = evaluate([suite], [spec], seeds=[args.seeds[0]], workers=args.workers, data_dir=args.data_dir)
        good = {r["instance"]: r["length"] for r in recs if r["status"] == "ok"}
        path = save_references(args.data_dir, suite.name, spec, good)
        bad = len(recs) - len(good)
        print(f"{suite.name}: {len(good)} reference lengths -> {path}" + (f" ({bad} failed)" if bad else ""))


def cmd_export(args):
    for suite in _suites(args):
        if any(i.coords is None or i.metric != "euclidean" for i in suite.instances):
            sys.exit(f"{suite.name}: the text format only holds Euclidean coordinates")
        path = f"{args.out}/{suite.name}.txt" if args.out else f"{suite.name}.txt"
        write_text_tsp(path, suite.instances)
        print(f"wrote {path}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="tspbench", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list solvers and whether their backends are installed").set_defaults(fn=cmd_list)
    for name, fn, hlp in (
        ("run", cmd_run, "evaluate solvers on suites"),
        ("reference", cmd_reference, "compute and cache reference tour lengths with one solver"),
        ("export", cmd_export, "write a generated suite in the Joshi / Fu text format"),
    ):
        sp = sub.add_parser(name, help=hlp)
        sp.add_argument("--suite", action="append", required=True, help="suite spec, repeatable")
        sp.add_argument("--solver", action="append", default=[], help="solver spec, repeatable")
        sp.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
        sp.add_argument("--limit", type=int, default=None, help="use only the first N instances of each suite")
        sp.add_argument("--workers", type=int, default=1)
        sp.add_argument("--data-dir", default="data")
        sp.add_argument("--out", default=None)
        sp.set_defaults(fn=fn)
    args = p.parse_args(argv)
    if args.cmd == "run" and not args.solver:
        p.error("run needs at least one --solver")
    args.fn(args)


if __name__ == "__main__":
    main()
