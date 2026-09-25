"""Print a markdown table of the old vs new results in results/*.json."""
import glob
import json
import os
from collections import defaultdict

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
groups = defaultdict(list)
for path in sorted(glob.glob(os.path.join(here, "results", "*.json"))):
    r = json.load(open(path))
    name = "old (notebook as was)" if r["mode"] == "old" else \
        ("new, distance weights" if r["edge_weight"] else "new, unweighted mean")
    groups[name].append(r)


def fmt(xs):
    xs = 100 * np.asarray(xs)
    return f"{xs.mean():.2f} ± {xs.std(ddof=1) if len(xs) > 1 else 0:.2f}"


print("| setup | seeds | test gap, n=10–30 (reported ref) | test gap vs float optimum | gap at n=100 vs float optimum | epoch used |")
print("|---|---|---|---|---|---|")
for name in ["old (notebook as was)", "new, distance weights", "new, unweighted mean"]:
    rs = groups.get(name)
    if not rs:
        continue
    print(f"| {name} | {len(rs)} | {fmt([r['test']['gap'] for r in rs])} | "
          f"{fmt([r['test']['gap_float_ref'] for r in rs])} | {fmt([r['tsp100']['gap_float_ref'] for r in rs])} | "
          f"{', '.join(str(r['chosen_epoch'] + 1) for r in rs)} |")
