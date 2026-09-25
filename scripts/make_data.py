"""Generate labelled training/validation instances (distance matrices only).

Sizes, distance types and scales are mixed. Labels are LKH-3 tours.

    python scripts/make_data.py --out data/train.pt --num 40000 --seed 1  # -> train_000.pt ...
    python scripts/make_data.py --out data/val.pt --num 600 --seed 2 --store-d
"""
import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tspgnn import instances  # noqa: E402
from tspgnn.graph import build_graph, label_coverage  # noqa: E402
from tspgnn.tours import lkh_tour  # noqa: E402


def make_one(args):
    seed, n_min, n_max, types, k, store_d = args
    rng = np.random.default_rng(seed)
    kind = types[rng.integers(len(types))]
    n = int(rng.integers(n_min, n_max + 1))
    scale = float(np.exp(rng.uniform(np.log(1e-2), np.log(1e3))))
    d = instances.generate(kind, n, rng, scale=scale)
    g = build_graph(d, k=k, tour=lkh_tour(d))
    if store_d:
        g.d = d.astype(np.float32)
    return kind, g


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--num", type=int, default=30000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n-min", type=int, default=20)
    p.add_argument("--n-max", type=int, default=100)
    p.add_argument("--types", default=",".join(instances.TRAIN_TYPES))
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--store-d", action="store_true", help="keep matrices (for validation gaps)")
    p.add_argument("--shard-size", type=int, default=5000)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()
    types = a.types.split(",")
    base = np.random.SeedSequence(a.seed).generate_state(a.num)
    jobs = [(int(s), a.n_min, a.n_max, types, a.k, a.store_d) for s in base]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    covs = []
    # Shards keep peak memory low; elkai also leaks a little, hence maxtasksperchild.
    with Pool(a.workers, maxtasksperchild=500) as pool:
        for s, lo in enumerate(range(0, a.num, a.shard_size)):
            out = pool.map(make_one, jobs[lo:lo + a.shard_size], chunksize=16)
            graphs = [g for _, g in out]
            covs += [label_coverage(g) for g in graphs]
            path = a.out if a.num <= a.shard_size else a.out.replace(".pt", f"_{s:03d}.pt")
            torch.save({"graphs": graphs, "kinds": [k for k, _ in out], "args": vars(a)}, path)
            print(f"saved {len(graphs)} instances to {path}", flush=True)
    print(f"tour-edge coverage of kNN graph {np.mean(covs):.4f}")


if __name__ == "__main__":
    main()
