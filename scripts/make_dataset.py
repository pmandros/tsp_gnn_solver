"""Generate uniform random Euclidean TSP instances with exact reference tours.

For every instance we store two optimal tours:
  * `tour_float` / `len_float`: optimal for the true float distances (the correct reference).
  * `tour_round` / `len_round`: optimal for nint-rounded distances, i.e. what Concorde's
    EUC_2D norm returns on raw [0, 100] coordinates. Used only to reproduce the old metric.

Usage: python make_dataset.py OUT.npz --sizes 10 15 20 25 30 --per-size 1000 --seed 0
"""
import argparse
from multiprocessing import Pool

import numpy as np

from exact_tsp import solve_tsp_exact


def _solve(points):
    d = np.sqrt(((points[:, None] - points) ** 2).sum(2))
    tf, lf = solve_tsp_exact(d)
    tr, lr = solve_tsp_exact(np.rint(d))
    return tf, lf, tr, lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--sizes", type=int, nargs="+", default=[10, 15, 20, 25, 30])
    ap.add_argument("--per-size", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    points = [100.0 * rng.random((n, 2)) for n in args.sizes for _ in range(args.per_size)]
    with Pool(args.workers) as pool:
        sols = pool.map(_solve, points, chunksize=4)

    out = {"sizes": np.array([len(p) for p in points])}
    for i, (p, (tf, lf, tr, lr)) in enumerate(zip(points, sols)):
        out[f"points_{i}"] = p
        out[f"tour_float_{i}"] = tf
        out[f"tour_round_{i}"] = tr
    out["len_float"] = np.array([s[1] for s in sols])
    out["len_round"] = np.array([s[3] for s in sols])
    np.savez_compressed(args.out, **out)
    print(f"wrote {len(points)} instances to {args.out}")


if __name__ == "__main__":
    main()
