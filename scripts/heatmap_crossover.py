"""Run published heatmaps and ours through the same guided search (tspgnn.search).

    python scripts/import_rethink_heatmaps.py ...   # once, writes heatmaps/<method><n>.npz
    python scripts/heatmap_crossover.py --n 500 --heatmaps heatmaps --out results/crossover

Each external heatmap (DIFUSCO, DIMES, Att-GCN, as released with Xia et al. 2024)
is used exactly like tspgnn's: top-5 candidates per city, sampled in proportion to
the heatmap value, greedy edge decode + 2-opt as the start tour, then the guided
search for ``time_per_node * n`` seconds. Model inference time is not included for
any method, since external heatmaps are precomputed (tspgnn's is measured separately).
Gaps are relative to the LKH-3 references in data/refs/.
"""
import argparse
import csv
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from tspbench.suites import load_references, load_suite, reference_for
from tspgnn.api import _guide
from tspgnn.search import candidate_lists, guide_weights, guided_search
from tspgnn.tours import greedy_edge_tour, is_valid_tour, knn_lists, tour_length, two_opt

SOFTDIST_TAU = {500: 0.0066, 1000: 0.0051, 10000: 0.0018}  # tuned values from Xia et al.
G = {}


def edges_from_dense_topk(idx, val):
    """Undirected edges (i<j) from per-row top-K lists, with the max value seen."""
    n, k = idx.shape
    i = np.repeat(np.arange(n), k)
    j = idx.ravel()
    v = val.ravel().astype(np.float64)
    keep = (i != j) & (v > 0)
    i, j, v = i[keep], j[keep], v[keep]
    a, b = np.minimum(i, j), np.maximum(i, j)
    key = a.astype(np.int64) * n + b
    order = np.lexsort((-v, key))
    key, a, b, v = key[order], a[order], b[order], v[order]
    first = np.ones(key.size, bool)
    first[1:] = key[1:] != key[:-1]
    return a[first], b[first], v[first]


def run(job):
    method, k_inst, seed = job
    d, ref, heat = G["d"][k_inst], G["ref"][k_inst], G["heat"]
    n = d.shape[0]
    t0 = time.perf_counter()
    if method in heat:
        src, dst, p = edges_from_dense_topk(heat[method]["idx"][k_inst], heat[method]["val"][k_inst])
        score, kind = np.log(np.maximum(p, 1e-300)), "prob"
    else:
        tau = SOFTDIST_TAU.get(n) if method == "softdist" else None
        src, dst, score, p, kind = _guide(d, method, None if method != "gnn" else G["ckpt"], 20, tau)
    t_heat = time.perf_counter() - t0
    tour = greedy_edge_tour(n, src, dst, np.argsort(-score, kind="stable"), d)
    tour = two_opt(d, tour, knn_lists(d, 20))
    start = tour_length(d, tour)
    cand, val = candidate_lists(n, src, dst, p, 5)
    w = guide_weights(cand, val, kind)
    tour = guided_search(d, tour, cand, w, time_limit=G["tpn"] * n, seed=seed)
    assert is_valid_tour(tour, n)
    return dict(method=method, instance=k_inst, seed=seed, start_gap=100 * (start / ref - 1),
                gap=100 * (tour_length(d, tour) / ref - 1), heat_time=t_heat,
                time=time.perf_counter() - t0)


def init(g):
    G.update(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--heatmaps", default="heatmaps")
    ap.add_argument("--methods", default="gnn,softdist,dist,difusco,dimes,attgcn")
    ap.add_argument("--time-per-node", type=float, default=0.002)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results/crossover")
    a = ap.parse_args()

    from tspgnn.api import DEFAULT_CHECKPOINT
    suite = load_suite(f"tsp{a.n}", limit=a.limit)
    refs = load_references("data", f"tsp{a.n}")
    ds, rs = [], []
    for inst in suite.instances:
        ds.append(inst.full_matrix())
        rs.append(reference_for(inst, refs)[0])
    methods = a.methods.split(",")
    heat = {}
    for m in methods:
        f = os.path.join(a.heatmaps, f"{m}{a.n}.npz")
        if os.path.exists(f):
            z = np.load(f)
            heat[m] = {"idx": z["idx"], "val": z["val"]}
    jobs = [(m, k, s) for s in a.seeds for k in range(len(ds)) for m in methods]
    g = dict(d=ds, ref=rs, heat=heat, tpn=a.time_per_node, ckpt=DEFAULT_CHECKPOINT)
    with Pool(a.workers, initializer=init, initargs=(g,)) as pool:
        rows = pool.map(run, jobs, chunksize=1)

    os.makedirs(a.out, exist_ok=True)
    tag = f"tsp{a.n}_t{a.time_per_node:g}"
    with open(os.path.join(a.out, f"{tag}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {}
    for m in methods:
        r = [x for x in rows if x["method"] == m]
        gaps = np.array([x["gap"] for x in r])
        summary[m] = dict(gap=gaps.mean(), ci95=1.96 * gaps.std(ddof=1) / np.sqrt(len(gaps)),
                          start_gap=np.mean([x["start_gap"] for x in r]),
                          time=np.mean([x["time"] for x in r]), n=len(r))
        print(f"{m:10s} gap {summary[m]['gap']:.3f} ± {summary[m]['ci95']:.3f}  "
              f"start {summary[m]['start_gap']:.2f}  time {summary[m]['time']:.2f}s")
    with open(os.path.join(a.out, f"{tag}.json"), "w") as f:
        json.dump(dict(args=vars(a), summary=summary), f, indent=1)


if __name__ == "__main__":
    main()
