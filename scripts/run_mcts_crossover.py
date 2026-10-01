"""Run heatmaps through the heatmap-guided MCTS used by Att-GCN, DIMES and DIFUSCO
(the default-parameter C++ code released with Xia et al., ICML 2024).

    python scripts/run_mcts_crossover.py --repo tools/rethink --n 500 --param-t 0.01 \
        --methods gnn,softdist,dist,difusco,dimes,attgcn --out results/crossover

The MCTS stops after ``param_t * n`` seconds of CPU time per instance (their default is
0.1, i.e. 50 s at n = 500). The time includes reading the heatmap file. Heatmaps come
from heatmaps/<method><n>.npz (scripts/export_heatmaps.py, scripts/import_rethink_heatmaps.py)
and are written back as dense text in the format the MCTS reads. Tours are re-scored
against the LKH-3 references in data/refs/, not the tours in the Fu et al. file.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile

import numpy as np

from tspbench.suites import load_references, load_suite, reference_for
from tspgnn.tours import is_valid_tour, tour_length


def build(repo, workdir):
    src = os.path.join(workdir, "mcts")
    shutil.copytree(os.path.join(repo, "default_mcts"), src)
    cpp = os.path.join(src, "code", "TSP.cpp")
    s = open(cpp).read()
    s = s.replace("Inst_Num_Per_Batch=atoi(argv[5]);",
                  "Inst_Num_Per_Batch=atoi(argv[5]);\n\tif(argc>6) Param_T=atof(argv[6]);")
    s = s.replace("\tgetchar();\n\n\treturn 0;", "\treturn 0;")
    open(cpp, "w").write(s)
    # Read_Heatmap() is declared bool but never returns, which GCC 13 at -O3 turns into
    # an illegal instruction.
    io = os.path.join(src, "code", "TSP_IO.h")
    s = open(io).read().rstrip()
    assert s.endswith("}")
    open(io, "w").write(s[:-1] + "    return true;\n}\n")
    subprocess.run(["make", "-s"], cwd=src, check=True, capture_output=True)
    return src


def write_heatmaps(npz, n, dirname, limit):
    z = np.load(npz)
    os.makedirs(dirname, exist_ok=True)
    for i in range(min(limit, z["idx"].shape[0])):
        h = np.zeros((n, n), np.float32)
        rows = np.repeat(np.arange(n), z["idx"].shape[2])
        h[rows, z["idx"][i].ravel()] = np.maximum(z["val"][i].ravel(), 0)
        h = np.maximum(h, h.T)
        with open(os.path.join(dirname, f"heatmaptsp{n}_{i}.txt"), "w") as f:
            f.write(f"{n}\n")
            for row in h:
                f.write(" ".join("0" if x == 0 else f"{x:.6g}" for x in row) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--param-t", type=float, default=0.1)
    ap.add_argument("--methods", default="gnn,softdist,dist,difusco,dimes,attgcn")
    ap.add_argument("--heatmaps", default="heatmaps")
    ap.add_argument("--limit", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results/crossover")
    a = ap.parse_args()

    insts = load_suite(f"tsp{a.n}", limit=a.limit).instances
    refs = load_references("data", f"tsp{a.n}")
    fu = os.path.abspath(os.path.join("data", "fu", f"tsp{a.n}_test_concorde.txt"))
    per = -(-len(insts) // a.workers)
    summary = {}
    with tempfile.TemporaryDirectory() as tmp:
        exe = os.path.join(build(os.path.abspath(a.repo), tmp), "test")
        for m in a.methods.split(","):
            run = os.path.join(tmp, m)
            write_heatmaps(os.path.join(a.heatmaps, f"{m}{a.n}.npz"), a.n,
                           os.path.join(run, "heatmap", f"tsp{a.n}"), len(insts))
            procs = [subprocess.Popen([exe, str(b), f"res{b}.txt", fu, str(a.n), str(per), str(a.param_t)],
                                      cwd=run, stdout=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
                     for b in range(a.workers)]
            for p in procs:
                p.wait()
            gaps, times = [], []
            for b in range(a.workers):
                text = open(os.path.join(run, f"res{b}.txt")).read()
                for idx, secs, sol in re.findall(
                        r"Inst_Index:(\d+).*?Time:([\d.]+) Seconds\s*\nSolution: ([\d ]+)", text):
                    k = int(idx) - 1
                    if k >= len(insts):
                        continue
                    tour = np.array(sol.split(), dtype=np.int64) - 1
                    d = insts[k].full_matrix()
                    assert is_valid_tour(tour, a.n)
                    ref = reference_for(insts[k], refs)[0]
                    gaps.append(100 * (tour_length(d, tour) / ref - 1))
                    times.append(float(secs))
            g = np.array(gaps)
            summary[m] = dict(gap=g.mean(), ci95=1.96 * g.std(ddof=1) / np.sqrt(len(g)),
                              time=float(np.mean(times)), n=len(g))
            print(f"{m:10s} gap {g.mean():.3f} ± {summary[m]['ci95']:.3f}  time {np.mean(times):.1f}s  ({len(g)} inst)", flush=True)
            shutil.rmtree(run)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"mcts_tsp{a.n}_T{a.param_t:g}_n{len(insts)}.json"), "w") as f:
        json.dump(dict(args=vars(a), summary=summary), f, indent=1)


if __name__ == "__main__":
    main()
