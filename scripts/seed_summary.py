"""Aggregate tspbench results over independently trained model seeds.

    python scripts/seed_summary.py --eval-dir runs/eval --out runs/SEEDS.md

Expects one tspbench output directory per model, named after it
(``seed0``, ``seed1``, ..., plus any model-free baselines such as ``baseline``),
each holding a ``summary.csv``. ``--eval-dir`` can be repeated to merge runs
over different suites. For every suite and method it reports the mean
gap to the reference over the ``seed*`` models, the standard deviation across
seeds, and each seed's own gap.
"""
import argparse
import csv
import glob
import os
import re
from collections import defaultdict

import numpy as np


def method_name(spec):
    m = re.search(r"name=([^,]+)", spec)
    return m.group(1) if m else spec


def read(eval_dirs):
    rows = defaultdict(dict)  # (suite, method) -> {model: row}
    paths = [p for d in eval_dirs for p in sorted(glob.glob(os.path.join(d, "*", "summary.csv")))]
    for path in paths:
        model = os.path.basename(os.path.dirname(path))
        for r in csv.DictReader(open(path)):
            rows[(r["suite"], method_name(r["solver"]))][model] = r
    return rows


def table(rows):
    seeds = sorted({m for v in rows.values() for m in v if m.startswith("seed")},
                   key=lambda s: int(s[4:]) if s[4:].isdigit() else s)
    others = sorted({m for v in rows.values() for m in v if not m.startswith("seed")})
    head = ["suite", "method", "ref", "gap, mean over seeds", "std across seeds",
            *seeds, *others, "time / inst"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * 2 + "---|" + "---:|" * (len(head) - 3)]
    for (suite, method), by in sorted(rows.items()):
        gaps = [float(by[s]["gap_mean"]) for s in seeds if s in by and by[s]["gap_mean"]]
        any_row = next(iter(by.values()))
        mean = f"{100 * np.mean(gaps):.3f}%" if gaps else ""
        std = f"{100 * np.std(gaps, ddof=1):.3f}%" if len(gaps) > 1 else ""
        cells = [f"{100 * float(by[m]['gap_mean']):.3f}%" if m in by and by[m]["gap_mean"] else ""
                 for m in seeds + others]
        t = np.mean([float(r["time_mean_s"]) for r in by.values()])
        lines.append("| " + " | ".join([suite, method, any_row["reference"], mean, std, *cells,
                                        f"{t:.3g}s"]) + " |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", required=True, action="append", help="repeatable")
    p.add_argument("--out")
    a = p.parse_args()
    md = table(read(a.eval_dir))
    print(md)
    if a.out:
        with open(a.out, "w") as f:
            f.write("Gap to reference per model seed. Std is the sample standard deviation "
                    "across independently trained seeds.\n\n" + md + "\n")


if __name__ == "__main__":
    main()
