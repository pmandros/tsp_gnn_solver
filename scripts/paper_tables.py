"""Build the markdown tables in results/BENCHMARK.md from results/bench/*/summary.csv.

    python scripts/paper_tables.py results/bench > results/bench/tables.md
"""
import csv
import glob
import os
import re
import sys

LABELS = {
    "gnn+2opt": "tspgnn greedy + 2-opt",
    "gnn": "tspgnn greedy",
    "greedy(dist)+2opt": "ablation: greedy(dist) + 2-opt",
    "lkh": "LKH-3",
    "ortools:time_limit=1": "OR-Tools GLS (1 s)",
    "ortools:time_limit=10": "OR-Tools GLS (10 s)",
    "two_opt:init=farthest_insertion": "farthest insertion + 2-opt",
    "farthest_insertion": "farthest insertion",
    "random_insertion": "random insertion",
    "nearest_insertion": "nearest insertion",
    "nearest_neighbor": "nearest neighbour",
    "lkh:time_limit=60": "LKH-3",
}
ORDER = list(LABELS.values())


def label(spec):
    m = re.search(r"name=([^,]+)$", spec)
    return LABELS.get(m.group(1) if m else spec, spec)


def suite_name(s):
    s = s.replace(":num=128", "")
    return {"tsplib:max_n=10000": "TSPLIB"}.get(s, s)


def load(root):
    rows = []
    for path in glob.glob(os.path.join(root, "*", "summary.csv")):
        with open(path) as f:
            rows += list(csv.DictReader(f))
    return rows


def fmt(r):
    if not r["gap_mean"]:
        return f"failed ({r['failed']})"
    t = float(r["time_mean_s"])
    ts = f"{t:.2g}s" if t < 10 else f"{t:.0f}s"
    return f"{100 * float(r['gap_mean']):.2f} ± {100 * float(r['gap_ci95']):.2f} ({ts})"


def table(rows, suites):
    cells = {}
    for r in rows:
        s = suite_name(r["suite"])
        if s in suites:
            cells[(label(r["solver"]), s)] = fmt(r)
    solvers = sorted({k[0] for k in cells}, key=lambda x: ORDER.index(x) if x in ORDER else 99)
    out = ["| solver | " + " | ".join(suites) + " |", "|---|" + "---:|" * len(suites)]
    for sv in solvers:
        out.append(f"| {sv} | " + " | ".join(cells.get((sv, s), "") for s in suites) + " |")
    return "\n".join(out)


GROUPS = [
    ("Uniform Euclidean, Kool et al. test set (first 1280 instances), gap to Concorde",
     ["tsp20", "tsp50", "tsp100"]),
    ("Uniform Euclidean, Fu et al. test sets, gap to LKH-3 reference", ["tsp500", "tsp1000", "tsp10000"]),
    ("Manhattan (held out)", ["manhattan100", "manhattan500", "manhattan1000"]),
    ("Chebyshev (held out)", ["chebyshev100", "chebyshev500", "chebyshev1000"]),
    ("Clustered", ["clustered100", "clustered500", "clustered1000"]),
    ("Non-metric symmetric U(0,1)", ["nonmetric100", "nonmetric500", "nonmetric1000"]),
    ("TSPLIB, n ≤ 10000, gap to published optimum", ["TSPLIB"]),
]

if __name__ == "__main__":
    rows = load(sys.argv[1] if len(sys.argv) > 1 else "results/bench")
    for title, suites in GROUPS:
        print(f"### {title}\n\n{table(rows, suites)}\n")
