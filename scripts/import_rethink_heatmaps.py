"""Import the DIFUSCO, DIMES and Att-GCN heatmaps released with Xia et al. (ICML 2024).

    git clone https://github.com/xyfffff/rethink_mcts_for_tsp tools/rethink
    python scripts/import_rethink_heatmaps.py --repo tools/rethink --n 500 --out heatmaps

The released heatmaps are dense n x n text files (about 1 GB per method at
n = 1000, and far more at n = 10000), so they are streamed out of the zip one
file at a time. Each is symmetrized as their MCTS does ((h + h^T) / 2) and stored
as per-city top-50 neighbours. Every value the MCTS would use (>= 1e-4) is kept.
Heatmap i belongs to the i-th instance of the Fu et al. test file.
"""
import argparse
import os
import re
import zipfile

import numpy as np

K = 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--methods", default="difusco,dimes,attgcn")
    ap.add_argument("--out", default="heatmaps")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pat = re.compile(rf"heatmap/tsp{a.n}/heatmaptsp{a.n}_(\d+)\.txt$")
    for m in a.methods.split(","):
        with zipfile.ZipFile(os.path.join(a.repo, "all_heatmap", m, "heatmap.zip")) as z:
            files = {int(pat.search(f).group(1)): f for f in z.namelist() if pat.search(f)}
            num = len(files)
            idx = np.zeros((num, a.n, K), np.int32)
            val = np.zeros((num, a.n, K), np.float32)
            for i in range(num):
                arr = np.array(z.read(files[i]).split(), dtype=np.float64)
                assert int(arr[0]) == a.n and arr.size == a.n * a.n + 1
                h = arr[1:].reshape(a.n, a.n)
                h = 0.5 * (h + h.T)
                np.fill_diagonal(h, -1.0)
                top = np.argpartition(-h, K, 1)[:, :K]
                order = np.argsort(-np.take_along_axis(h, top, 1), 1)
                idx[i] = np.take_along_axis(top, order, 1)
                val[i] = np.take_along_axis(h, idx[i], 1)
        assert not (val[:, :, -1] >= 1e-4).any(), "top-50 cut off values the MCTS would use"
        np.savez_compressed(os.path.join(a.out, f"{m}{a.n}.npz"), idx=idx, val=val)
        print(f"{m}{a.n}: {num} heatmaps, mean {np.mean((val >= 1e-4).sum(2)):.1f} candidates per city")


if __name__ == "__main__":
    main()
