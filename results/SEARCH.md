# Stronger decoding: sampling and guided search, GNN heatmap vs distance

[BENCHMARK.md](BENCHMARK.md) found that with greedy decoding + 2-opt, the GNN heatmap was worth only 0.4–0.9 gap
points over ranking the same kNN edges by distance on Euclidean-like instances. This file asks whether that changes
under a stronger search. It does: with the same search and the same time budget, the GNN-guided search's gap is 30–70% lower
on every metric set and 6–7× lower on non-metric ones.

Same checkpoint (`checkpoints/tspgnn.pt`, trained on n ≤ 100), same suites and references as BENCHMARK.md, same
4-core Xeon, one core per solver process, no GPU. Reproduce with `bash scripts/run_search_bench.sh`. Per-instance records are in
`results/search/*/records.csv.gz`.

## Decoders

Both decoders take a *guide*: a score for each kNN candidate edge (k = 20). `guide=gnn` uses the heatmap, and
`guide=dist` ranks the same edges by distance. Everything else is identical, so the difference is the model's.

- **Sampling** (`tspgnn.api:solve_sample`): the best of 16 greedy edge decodes, each on Gumbel-perturbed scores
  (τ = 1; the first decode unperturbed), each followed by 2-opt. This is the "sampling + 2-opt" of DIMES and DIFUSCO.
- **Guided search** (`tspgnn.api:solve_search`, code in `tspgnn/search.py`): starts from the guide's greedy + 2-opt
  tour. It then runs Lin–Kernighan-style chains of up to 6 sequential 2-opt steps. Each step's new edge is
  *sampled* from the city's top-5 guide candidates, weighted by the GNN edge probability or by 1/rank for distance.
  Improving chains are committed. It then iterates local double-bridge kicks (swap two short adjacent segments),
  re-optimizing around each kick and keeping it if the tour got no longer, until the time budget
  of 2 ms per city runs out (0.2 s at n = 100, 2 s at n = 1000, 20 s at n = 10000). This is the same role as the
  heatmap-guided MCTS of Fu et al. (2021) used by DIMES and DIFUSCO: k-opt moves sampled from the heatmap. It is not a
  reimplementation of their code, and it does not keep a search tree. The search is stochastic; seeds 0–2 on TSP500/1000, seed 0 elsewhere.

Times include GNN inference. The budget starts after the first full local optimization, so the actual time per
instance is 10–15% over budget at n ≤ 1000 and 2.3× over at n = 10000, where the first pass is long. The tables
show the measured times.

## Results

Gap to the same references as BENCHMARK.md (Concorde for n ≤ 100, LKH-3 above, published optima for TSPLIB), in %,
mean ± 95% interval, with mean time per instance on one core.

### Euclidean (Kool et al. TSP20–100, first 1280 instances; Fu et al. TSP500/1000/10000)

| decoder | TSP20 | TSP50 | TSP100 | TSP500 | TSP1000 | TSP10000 |
|---|---:|---:|---:|---:|---:|---:|
| **gnn + guided search** | **0.000** (0.05s) | **0.002** (0.12s) | **0.068** (0.23s) | **0.454** ± 0.025 (1.2s) | **0.529** ± 0.023 (2.2s) | **0.793** ± 0.030 (47s) |
| dist + guided search | 0.000 (0.04s) | 0.013 (0.10s) | 0.156 (0.20s) | 0.669 ± 0.015 (1.0s) | 0.824 ± 0.041 (2.1s) | 1.137 ± 0.045 (41s) |
| gnn + sampling ×16 + 2-opt | 0.036 | 0.457 | 1.308 | 3.091 (0.14s) | 3.551 (0.25s) | — |
| dist + sampling ×16 + 2-opt | 0.043 | 0.792 | 2.457 | 4.538 (0.03s) | 4.540 (0.10s) | — |
| gnn greedy + 2-opt (BENCHMARK.md) | 1.17 | 2.73 | 3.50 | 4.05 | 4.08 | 3.90 |
| dist greedy + 2-opt (BENCHMARK.md) | 2.01 | 3.67 | 4.30 | 4.60 | 4.54 | 4.31 |

### Distance types (128 instances each; Manhattan and Chebyshev held out of training)

| decoder | n | Manhattan | Chebyshev | clustered | non-metric |
|---|---:|---:|---:|---:|---:|
| gnn + guided search | 100 | **0.049** | **0.063** | **0.034** | **0.090** |
| dist + guided search | 100 | 0.147 | 0.135 | 0.190 | 0.668 |
| gnn + guided search | 500 | **0.404** | **0.430** | **0.377** | **1.519** |
| dist + guided search | 500 | 0.633 | 0.624 | 0.815 | 10.067 |
| gnn + guided search | 1000 | **0.532** | **0.494** | **0.553** | **3.456** |
| dist + guided search | 1000 | 0.835 | 0.785 | 1.016 | 21.119 |
| gnn + sampling ×16 + 2-opt | 1000 | 3.854 | 3.765 | 3.752 | 21.496 |
| dist + sampling ×16 + 2-opt | 1000 | 5.117 | 5.007 | 4.960 | 64.166 |
| gnn greedy + 2-opt (BENCHMARK.md) | 1000 | 4.52 | 4.34 | 4.40 | 24.84 |

Search times are 0.23 s / 1.1 s / 2.3 s at n = 100 / 500 / 1000. The n = 100 and 500 sampling rows are in
`results/search/types/summary.md`.

### TSPLIB (103 symmetric instances, n ≤ 10000, gap to published optima)

| decoder | gap | time |
|---|---:|---:|
| gnn + guided search | **0.342 ± 0.089** | 2.2s |
| dist + guided search | 1.125 ± 0.518 | 1.8s |
| gnn greedy + 2-opt (BENCHMARK.md) | 4.18 | |

### Is it just the extra time? (TSP1000 and non-metric n = 1000, seed 0)

| decoder | TSP1000 | non-metric 1000 |
|---|---:|---:|
| gnn + guided search, 1 ms/city | **0.629** (1.2s) | **4.534** (1.2s) |
| dist + guided search, 2 ms/city | 0.824 (2.1s) | 21.119 (2.0s) |
| dist + guided search, 4 ms/city | 0.743 (4.1s) | 16.641 (4.1s) |

With less than a third of the time, the GNN guide still beats the distance guide.

## Against published learned solvers

Same test sets as the papers, using their published gaps (as tabulated in BENCHMARK.md, from Sun & Yang 2023, Table 2).

| method | TSP500 | TSP1000 | TSP10000 | hardware |
|---|---:|---:|---:|---|
| **tspgnn + guided search (2 ms/city)** | **0.45** | **0.53** | **0.79** | 1 CPU core, 1.2 s / 2.2 s / 47 s per instance |
| DIFUSCO greedy + MCTS | 0.46 | 1.17 | 2.58 | GPU |
| DIFUSCO sampling + 2-opt | 0.57 | 1.43 | 2.95 | GPU |
| DIMES RL + AS + MCTS | 1.76 | 2.46 | 3.19 | GPU |
| Att-GCN + MCTS | 2.54 | 3.22 | 4.39 | GPU |
| LKH-3 (BENCHMARK.md) | 0.00 (1.8s) | 0.00 (7.6s) | | 1 CPU core |

Caveats before claiming anything here:

- **Times are not comparable.** The published MCTS numbers come with different, GPU-based time budgets, which we did
  not match. A fair comparison runs their released MCTS code with our heatmap under their budget, or our search
  with their heatmaps. Our search is also closer to Or-opt/LK local search than to their MCTS implementation.
- **The search does a lot of the work.** Distance-guided search alone (0.67 / 0.82 / 1.14%) also beats the published
  MCTS numbers, so these rows mainly say that this search is strong. The GNN's contribution is the gap between the gnn and dist rows,
  which is consistent, significant, and survives giving the distance guide 4× the budget.
- **LKH-3 is still better.** At n = 1000 it reaches the reference in 7.6 s on one core.
- Sampling τ = 1 was not tuned. A 32-instance sweep on TSP1000 gave 3.24 / 3.09 / 3.57% for the GNN at
  τ = 0.1 / 0.3 / 1 and 4.42–4.49% for distance, so tuning does not change the picture.
- Seed variance is across search seeds only. There is still one training run.

## What this means for the paper

The GNN heatmap is a much better guide for local search than distance. It gives 30–70% lower gaps on every metric set,
including held-out metrics and TSPLIB, and 6–7× lower on non-metric matrices. This holds for a model trained only on n ≤ 100 and
evaluated up to n = 10000. That is a stronger and cleaner claim than the greedy + 2-opt result, where the GNN looked nearly redundant.
The natural next steps are:

1. **A matched comparison with DIFUSCO and DIMES.** Run their MCTS code with the tspgnn heatmap, or give our search their
   heatmaps, under the same time.
2. **Gap vs time curves** instead of one budget.
3. **GPU training with several seeds.**
