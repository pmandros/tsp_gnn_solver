"""Write tspgnn's heatmap and the model-free ones for a Fu et al. test set, in the
format of scripts/import_rethink_heatmaps.py (per-city top-50 neighbours, symmetric).

    python scripts/export_heatmaps.py --n 500 --out heatmaps

``gnn`` is sigmoid(logit) on the kNN graph, ``softdist`` is Xia et al.'s
softmax(-d / tau) with their tuned tau, and ``dist`` is a rank prior
1/(rank+1) over the 20 nearest neighbours, normalized per row (similar in
spirit to the GT-Prior of Pan et al. 2026).
"""
import argparse
import os
import time

import numpy as np

from tspbench.suites import load_suite
from tspgnn.api import DEFAULT_CHECKPOINT, _guide
from tspgnn.search import candidate_lists

K = 50
TAU = {500: 0.0066, 1000: 0.0051, 10000: 0.0018}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", default="heatmaps")
    ap.add_argument("--methods", default="gnn,softdist,dist")
    a = ap.parse_args()
    insts = load_suite(f"tsp{a.n}").instances
    os.makedirs(a.out, exist_ok=True)
    for m in a.methods.split(","):
        idx = np.zeros((len(insts), a.n, K), np.int32)
        val = np.zeros((len(insts), a.n, K), np.float32)
        secs = []
        for i, inst in enumerate(insts):
            d = inst.full_matrix()
            t = time.perf_counter()
            src, dst, _, p, kind = _guide(d, m, DEFAULT_CHECKPOINT, 20, TAU.get(a.n) if m == "softdist" else None)
            if kind == "rank":
                # Rank prior: 1/(rank+1) of j among i's neighbours, normalized, then symmetrized.
                cand, _ = candidate_lists(a.n, src, dst, p, 20)
                w = np.where(cand >= 0, 1.0 / (1.0 + np.arange(20)), 0.0)
                w /= w.sum(1, keepdims=True)
                h = np.zeros((a.n, a.n))
                rows = np.repeat(np.arange(a.n), 20)
                ok = cand.ravel() >= 0
                h[rows[ok], cand.ravel()[ok]] = w.ravel()[ok]
                h = 0.5 * (h + h.T)
                c, v = candidate_lists(a.n, *np.nonzero(np.triu(h, 1)), h[np.triu(h, 1) > 0], K)
            else:
                c, v = candidate_lists(a.n, src, dst, p, K)
            secs.append(time.perf_counter() - t)
            idx[i] = np.maximum(c, 0)
            val[i] = np.where(c >= 0, v, 0.0)
        np.savez_compressed(os.path.join(a.out, f"{m}{a.n}.npz"), idx=idx, val=val)
        print(f"{m}{a.n}: {len(insts)} heatmaps, {np.mean(secs):.3f}s each")


if __name__ == "__main__":
    main()
