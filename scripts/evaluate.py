"""Evaluate the GNN and classical baselines against LKH-3 across sizes and types.

    python scripts/evaluate.py --ckpt runs/main/best.pt --out results/main.json

Test instances use seeds disjoint from training and are cached (with their
LKH reference tours) under --cache. Every tour, including the reference, is
scored on the float distance matrix.
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tspgnn import instances  # noqa: E402
from tspgnn.graph import EDGE_DIM, NODE_DIM  # noqa: E402
from tspgnn.model import TSPGNN  # noqa: E402
from tspgnn.solve import (solve_gnn, solve_greedy_distance,  # noqa: E402
                          solve_nearest_neighbor, timed)
from tspgnn.tours import is_valid_tour, lkh_tour, tour_length  # noqa: E402

DEFAULT_COUNTS = {20: 128, 50: 128, 100: 128, 200: 64, 500: 32, 1000: 16}
TEST_SEED = 10_000


def _make_ref(args):
    kind, n, i = args
    rng = np.random.default_rng([TEST_SEED, list(instances.GENERATORS).index(kind), n, i])
    d = instances.generate(kind, n, rng)
    tour, t = timed(lkh_tour, d, runs=1 if n > 200 else 5)
    return d.astype(np.float64), tour, t


def test_set(kind, n, count, cache, workers):
    path = os.path.join(cache, f"{kind}_{n}_{count}.pt")
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    with Pool(workers) as pool:
        out = pool.map(_make_ref, [(kind, n, i) for i in range(count)], chunksize=1)
    os.makedirs(cache, exist_ok=True)
    torch.save(out, path)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt")
    p.add_argument("--refs-only", action="store_true", help="only build the cached test sets")
    p.add_argument("--out")
    p.add_argument("--cache", default="data/test")
    p.add_argument("--types", default=",".join(instances.GENERATORS))
    p.add_argument("--sizes", default=",".join(map(str, DEFAULT_COUNTS)))
    p.add_argument("--count-scale", type=float, default=1.0)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()
    types, sizes = a.types.split(","), list(map(int, a.sizes.split(",")))

    def count_for(n):
        return max(4, int(DEFAULT_COUNTS.get(n, 16) * a.count_scale))

    if a.refs_only:
        for n in sizes:
            for kind in types:
                test_set(kind, n, count_for(n), a.cache, a.workers)
                print("cached", kind, n, flush=True)
        return

    ck = torch.load(a.ckpt, weights_only=False)
    cfg = ck["config"]
    model = TSPGNN(NODE_DIM, EDGE_DIM, cfg["hidden"], cfg["layers"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    torch.set_num_threads(a.workers)

    methods = {
        "nearest_neighbor+2opt": lambda d: solve_nearest_neighbor(d),
        "greedy_distance": lambda d: solve_greedy_distance(d, a.k, use_two_opt=False),
        "greedy_distance+2opt": lambda d: solve_greedy_distance(d, a.k),
        "gnn_greedy": lambda d: solve_gnn(model, d, a.k, use_two_opt=False),
        "gnn+2opt": lambda d: solve_gnn(model, d, a.k),
    }
    results = []
    for kind in types:
        for n in sizes:
            data = test_set(kind, n, count_for(n), a.cache, a.workers)
            row = {"type": kind, "n": n, "count": len(data),
                   "heldout_type": kind not in instances.TRAIN_TYPES,
                   "lkh_time_s": float(np.mean([t for _, _, t in data]))}
            for name, fn in methods.items():
                gaps, times = [], []
                for d, ref, _ in data:
                    tour, t = timed(fn, d)
                    assert is_valid_tour(tour, n), name
                    gaps.append(tour_length(d, tour) / tour_length(d, ref) - 1)
                    times.append(t)
                row[name] = {"gap_pct": 100 * float(np.mean(gaps)),
                             "gap_sem_pct": 100 * float(np.std(gaps) / np.sqrt(len(gaps))),
                             "time_s": float(np.mean(times))}
            results.append(row)
            print(kind, n, {k: round(v["gap_pct"], 2) for k, v in row.items()
                            if isinstance(v, dict)}, flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"checkpoint": a.ckpt, "config": cfg, "results": results},
              open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
