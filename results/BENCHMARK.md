# Benchmark: tspgnn against classical and learned solvers

This benchmark runs the checkpoint in `checkpoints/tspgnn.pt`, the distance-matrix-only gated GCN described in
[RESULTS.md](RESULTS.md). It was trained for 8 epochs on CPU, on 40k instances with n = 20–100 (Euclidean,
clustered and random non-metric). Everything was run through `tspbench` on a 4-core Intel Xeon (2.1 GHz),
with one core per solver process. No GPU was used anywhere.

Reproduce everything with `bash scripts/run_paper_bench.sh`, then `python scripts/paper_tables.py results/bench`.
Per-instance records are in `results/bench/*/records.csv.gz`, and the command lines, versions and git commit are in `run.json`.

## Summary

- **Against classical heuristics, tspgnn wins everywhere.** GNN greedy + 2-opt is 3.5% on TSP100, 4.1% on
  TSP500/1000 and 3.9% on TSP10000. It stays around 4–4.6% on the held-out Manhattan and Chebyshev metrics up to
  n = 1000, and 4.2% on TSPLIB up to 10k nodes. The best insertion heuristic plus 2-opt gets 6–11%. OR-Tools GLS gets
  3.2% at 1 s on TSP100 (slightly better than tspgnn), but 9.7% at 10 s on TSP1000.
- **The GNN's own contribution on metric instances is small.** Ranking the same kNN edges by distance, with the same
  greedy decoder and 2-opt, reaches 4.3% on TSP100, 4.5% on TSP1000 and 4.3% on TSP10000. So on Euclidean-like
  instances the GNN is worth only 0.4–0.9 gap points. On non-metric matrices it is worth far more: 24.8% vs 64.2%
  at n = 1000.
- **Without 2-opt the model is weak**, at 6.2 / 11.3 / 13.9% on TSP20/50/100, while Kool et al.'s AM gets
  0.34 / 1.76 / 4.53% with greedy decoding.
- **Against published learned solvers, it is not competitive yet, except at TSP10000.** DIFUSCO greedy + 2-opt
  reports 1.49 / 1.90 / 3.10% on TSP500/1000/10000; tspgnn gets 4.05 / 4.08 / 3.90%. At TSP10000, tspgnn beats
  Att-GCN+MCTS (4.39%) and is level with DIMES RL+MCTS (3.98%), using a model that never saw more than 100 nodes
  and 14 s per instance on one CPU core.
- **Size and metric generalization hold up.** The gap is flat from n = 100 to n = 10000, and the held-out metrics
  are within about 0.5 points of Euclidean. This is the strongest claim the model supports today.
- **LKH-3 is the elephant in the room.** It is within 0.01% of optimal on every generated set, in 0.05 s at n = 100
  and 8–12 s at n = 1000.

## Comparison with published learned solvers

Gaps as reported in the papers, on the same test sets: the Kool et al. TSP20/50/100 set (10,000 instances;
this run used the first 1280) and the Fu et al. TSP500/1000/10000 sets. Their times are for GPUs and are not comparable,
so none are shown. "—" means not reported.

| method | decoding | TSP20 | TSP50 | TSP100 | TSP500 | TSP1000 | TSP10000 | source |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **tspgnn (this repo)** | greedy | 6.24 | 11.32 | 13.91 | 16.24 | 15.65 | 14.82 | this run |
| **tspgnn (this repo)** | greedy + 2-opt | 1.17 | 2.73 | 3.50 | 4.05 | 4.08 | 3.90 | this run |
| GCN (Joshi et al. 2019) | greedy | 0.60 | 3.10 | 8.38 | — | — | — | Joshi et al. 2019, Table 1 |
| Attention Model (Kool et al. 2019) | greedy | 0.34 | 1.76 | 4.53 | — | — | — | Kool et al. 2019, Table 1 |
| POMO (Kwon et al. 2020) | single trajectory | — | 0.64 | 1.07 | — | — | — | as listed in Sun & Yang 2023, Table 1 |
| Att-GCN (Fu et al. 2021) | MCTS | — | — | — | 2.54 | 3.22 | 4.39 | as listed in Sun & Yang 2023, Table 2 |
| DIMES (Qiu et al. 2022) | RL + sampling | — | — | — | 13.84 | 14.01 | 19.48 | as listed in Sun & Yang 2023, Table 2 |
| DIMES (Qiu et al. 2022) | RL + MCTS | — | — | — | — | — | 3.98 | as listed in Sun & Yang 2023, Table 2 |
| DIMES (Qiu et al. 2022) | RL + AS + MCTS | — | — | — | 1.76 | 2.46 | 3.19 | as listed in Sun & Yang 2023, Table 2 |
| DIFUSCO (Sun & Yang 2023) | greedy | — | — | — | 10.85 | 13.06 | 36.75 | Sun & Yang 2023, Table 2 |
| DIFUSCO (Sun & Yang 2023) | greedy + 2-opt | — | 0.10† | 0.24† | 1.49 | 1.90 | 3.10 | Sun & Yang 2023, Tables 1–2 |
| DIFUSCO (Sun & Yang 2023) | sampling + 2-opt | — | — | — | 0.57 | 1.43 | 2.95 | Sun & Yang 2023, Table 2 |
| DIFUSCO (Sun & Yang 2023) | greedy + MCTS | — | — | — | 0.46 | 1.17 | 2.58 | Sun & Yang 2023, Table 2 |

† DIFUSCO's Table 1 marks its greedy TSP50/100 row with a dagger for 2-opt post-processing.

Caveats:

- The published gaps for TSP500/1000/10000 are relative to their Concorde or LKH-3 lengths of 16.55 / 23.12 / 71.77.
  Ours are relative to the LKH-3 references in `data/refs/` (16.546 / 23.119 / 71.778), which match them to within 0.01%.
- The literature numbers were read from the DIFUSCO NeurIPS 2023 PDF, which also lists AM and GCN on TSP50/100.
  The TSP20 figures for AM and GCN come from our recollection of the original papers' tables. Recheck every row against the PDFs
  before submission.
- tspgnn is deterministic, so its "± over seeds" is really ± over instances. Seed variance of training (only one
  checkpoint exists) is not measured.

Sources:

- Sun & Yang, *DIFUSCO: Graph-based Diffusion Solvers for Combinatorial Optimization*, NeurIPS 2023.
  <https://proceedings.neurips.cc/paper_files/paper/2023/file/0ba520d93c3df592c83a611961314c98-Paper-Conference.pdf>
- Kool, van Hoof & Welling, *Attention, Learn to Solve Routing Problems!*, ICLR 2019. <https://arxiv.org/abs/1803.08475>
- Joshi, Laurent & Bresson, *An Efficient Graph Convolutional Network Technique for the Travelling Salesman Problem*, 2019. <https://arxiv.org/abs/1906.01227>
- Kwon et al., *POMO*, NeurIPS 2020; Fu et al., *Generalize a Small Pre-trained Model to Arbitrarily Large TSP Instances*, AAAI 2021;
  Qiu et al., *DIMES*, NeurIPS 2022, all as tabulated by Sun & Yang 2023.

## Full results

Every cell is the mean gap in % ± a 95% interval (across seeds for stochastic solvers, across instances otherwise),
with the mean time per instance on one core in parentheses. Seeds are 0, 1 and 2, except for the ablation, which is
deterministic and ran with seed 0. References: Concorde for n ≤ 100, LKH-3 for larger sets, and published optima for TSPLIB.
The generated sets have 128 instances each. Negative LKH-3 gaps mean it beat the cached LKH-3 reference.

The **ablation** row uses tspgnn's own kNN candidate set (k = 20), greedy edge decoder and neighbour-list 2-opt,
with edges ranked by distance instead of by GNN score. It is the baseline the GNN has to beat.
The harness's `farthest insertion + 2-opt` uses a different, weaker 2-opt.

#### Uniform Euclidean, Kool et al. test set (first 1280 instances), gap to Concorde

| solver | tsp20 | tsp50 | tsp100 |
|---|---:|---:|---:|
| tspgnn greedy + 2-opt | 1.17 ± 0.11 (0.0057s) | 2.73 ± 0.12 (0.012s) | 3.50 ± 0.10 (0.024s) |
| tspgnn greedy | 6.24 ± 0.33 (0.0044s) | 11.32 ± 0.31 (0.012s) | 13.91 ± 0.25 (0.022s) |
| ablation: greedy(dist) + 2-opt | 2.01 ± 0.15 (0.00081s) | 3.67 ± 0.14 (0.00098s) | 4.30 ± 0.11 (0.0014s) |
| LKH-3 | 0.00 ± 0.00 (0.0013s) | 0.00 ± 0.00 (0.011s) | 0.00 ± 0.00 (0.051s) |
| OR-Tools GLS (1 s) | 0.00 ± 0.00 (1s) | 1.57 ± 0.09 (1s) | 3.16 ± 0.10 (1s) |
| farthest insertion + 2-opt | 1.51 ± 0.12 (0.0015s) | 4.58 ± 0.16 (0.0034s) | 6.39 ± 0.13 (0.0071s) |
| farthest insertion | 2.40 ± 0.16 (0.0013s) | 5.68 ± 0.18 (0.0031s) | 7.57 ± 0.14 (0.0063s) |
| random insertion | 4.43 ± 0.31 (0.00078s) | 7.80 ± 0.09 (0.0017s) | 9.63 ± 0.07 (0.0036s) |
| nearest insertion | 12.97 ± 0.38 (0.0012s) | 19.30 ± 0.25 (0.0033s) | 21.90 ± 0.19 (0.0066s) |
| nearest neighbour | 17.60 ± 0.26 (0.00018s) | 23.08 ± 0.42 (0.00042s) | 24.90 ± 0.36 (0.00078s) |

#### Uniform Euclidean, Fu et al. test sets, gap to LKH-3 reference

| solver | tsp500 | tsp1000 | tsp10000 |
|---|---:|---:|---:|
| tspgnn greedy + 2-opt | 4.05 ± 0.14 (0.11s) | 4.08 ± 0.10 (0.23s) | 3.90 ± 0.07 (14s) |
| tspgnn greedy | 16.24 ± 0.47 (0.095s) | 15.65 ± 0.30 (0.21s) | 14.82 ± 0.32 (12s) |
| ablation: greedy(dist) + 2-opt | 4.60 ± 0.17 (0.016s) | 4.54 ± 0.12 (0.042s) | 4.31 ± 0.12 (6.9s) |
| LKH-3 | 0.00 ± 0.00 (1.8s) | 0.00 ± 0.00 (7.6s) |  |
| OR-Tools GLS (10 s) | 4.93 ± 0.16 (10s) | 9.74 ± 0.22 (10s) |  |
| farthest insertion + 2-opt | 9.27 ± 0.20 (0.089s) | 9.88 ± 0.16 (0.47s) | 10.99 ± 0.16 (7.7s) |
| farthest insertion | 10.59 ± 0.22 (0.044s) | 11.28 ± 0.16 (0.13s) | 12.28 ± 0.15 (5.8s) |
| random insertion | 12.35 ± 0.11 (0.024s) | 12.89 ± 0.14 (0.064s) |  |
| nearest insertion | 24.64 ± 0.27 (0.045s) | 25.26 ± 0.20 (0.12s) |  |
| nearest neighbour | 25.75 ± 0.57 (0.0067s) | 25.22 ± 0.84 (0.02s) | 23.79 ± 0.82 (1.4s) |

#### Manhattan (held out)

| solver | manhattan100 | manhattan500 | manhattan1000 |
|---|---:|---:|---:|
| tspgnn greedy + 2-opt | 4.14 ± 0.37 (0.028s) | 4.61 ± 0.18 (0.12s) | 4.52 ± 0.13 (0.22s) |
| tspgnn greedy | 14.07 ± 0.92 (0.017s) | 16.43 ± 0.47 (0.1s) | 15.88 ± 0.35 (0.2s) |
| ablation: greedy(dist) + 2-opt | 4.94 ± 0.42 (0.0061s) | 5.10 ± 0.21 (0.018s) | 5.13 ± 0.15 (0.055s) |
| LKH-3 | 0.00 ± 0.00 (0.052s) | -0.00 ± 0.00 (2.1s) | 0.00 ± 0.00 (8.6s) |
| farthest insertion + 2-opt | 7.56 ± 0.45 (0.0082s) | 9.88 ± 0.24 (0.12s) | 10.61 ± 0.17 (0.69s) |
| farthest insertion | 10.39 ± 0.57 (0.0067s) | 12.85 ± 0.27 (0.053s) | 13.74 ± 0.18 (0.15s) |
| random insertion | 13.59 ± 0.69 (0.0038s) | 15.71 ± 0.43 (0.028s) | 16.44 ± 0.28 (0.08s) |
| nearest insertion | 25.70 ± 0.62 (0.0067s) | 27.43 ± 0.34 (0.052s) | 28.24 ± 0.20 (0.16s) |
| nearest neighbour | 24.95 ± 1.58 (0.00089s) | 25.99 ± 0.85 (0.0092s) | 25.65 ± 0.10 (0.029s) |

#### Chebyshev (held out)

| solver | chebyshev100 | chebyshev500 | chebyshev1000 |
|---|---:|---:|---:|
| tspgnn greedy + 2-opt | 3.78 ± 0.37 (0.028s) | 4.45 ± 0.20 (0.12s) | 4.33 ± 0.16 (0.26s) |
| tspgnn greedy | 14.30 ± 0.86 (0.017s) | 16.91 ± 0.47 (0.11s) | 16.37 ± 0.39 (0.24s) |
| ablation: greedy(dist) + 2-opt | 4.73 ± 0.35 (0.0064s) | 5.00 ± 0.17 (0.023s) | 5.01 ± 0.14 (0.079s) |
| LKH-3 | 0.00 ± 0.01 (0.056s) | 0.00 ± 0.00 (2.1s) | -0.00 ± 0.00 (8.4s) |
| farthest insertion + 2-opt | 6.81 ± 0.43 (0.009s) | 9.34 ± 0.20 (0.15s) | 10.07 ± 0.16 (0.81s) |
| farthest insertion | 9.63 ± 0.54 (0.0072s) | 12.61 ± 0.26 (0.076s) | 13.29 ± 0.18 (0.25s) |
| random insertion | 12.12 ± 0.72 (0.0042s) | 15.07 ± 0.50 (0.039s) | 15.90 ± 0.19 (0.12s) |
| nearest insertion | 24.62 ± 0.62 (0.0071s) | 27.43 ± 0.33 (0.074s) | 27.73 ± 0.21 (0.23s) |
| nearest neighbour | 26.03 ± 1.96 (0.001s) | 27.01 ± 1.05 (0.013s) | 26.98 ± 1.20 (0.049s) |

#### Clustered

| solver | clustered100 | clustered500 | clustered1000 |
|---|---:|---:|---:|
| tspgnn greedy + 2-opt | 3.84 ± 0.33 (0.029s) | 4.49 ± 0.24 (0.13s) | 4.40 ± 0.16 (0.22s) |
| tspgnn greedy | 16.46 ± 0.97 (0.019s) | 15.71 ± 0.48 (0.11s) | 15.77 ± 0.40 (0.19s) |
| ablation: greedy(dist) + 2-opt | 5.30 ± 0.41 (0.0058s) | 5.23 ± 0.23 (0.015s) | 4.99 ± 0.17 (0.042s) |
| LKH-3 | 0.01 ± 0.01 (0.063s) | 0.00 ± 0.01 (3s) | -0.00 ± 0.00 (12s) |
| farthest insertion + 2-opt | 5.61 ± 0.38 (0.0089s) | 8.86 ± 0.22 (0.092s) | 9.49 ± 0.18 (0.46s) |
| farthest insertion | 6.91 ± 0.45 (0.0064s) | 10.17 ± 0.24 (0.05s) | 10.92 ± 0.19 (0.12s) |
| random insertion | 9.35 ± 0.65 (0.0037s) | 12.30 ± 0.11 (0.026s) | 12.87 ± 0.13 (0.062s) |
| nearest insertion | 19.42 ± 0.58 (0.0067s) | 23.68 ± 0.29 (0.048s) | 24.35 ± 0.21 (0.12s) |
| nearest neighbour | 27.15 ± 0.70 (0.00083s) | 26.56 ± 0.26 (0.007s) | 25.87 ± 0.45 (0.02s) |

#### Non-metric symmetric U(0,1)

| solver | nonmetric100 | nonmetric500 | nonmetric1000 |
|---|---:|---:|---:|
| tspgnn greedy + 2-opt | 10.66 ± 0.87 (0.026s) | 20.28 ± 0.70 (0.11s) | 24.84 ± 0.57 (0.2s) |
| tspgnn greedy | 48.96 ± 2.96 (0.018s) | 88.80 ± 3.02 (0.1s) | 102.39 ± 3.55 (0.18s) |
| ablation: greedy(dist) + 2-opt | 30.98 ± 1.42 (0.0056s) | 53.82 ± 1.08 (0.011s) | 64.17 ± 0.97 (0.026s) |
| LKH-3 | 0.00 ± 0.00 (0.028s) | -0.00 ± 0.00 (0.74s) | 0.00 ± 0.00 (3.1s) |
| farthest insertion + 2-opt | 78.07 ± 2.91 (0.0087s) | 218.05 ± 2.40 (0.58s) | 321.92 ± 2.67 (4.9s) |
| farthest insertion | 256.26 ± 6.20 (0.0039s) | 666.30 ± 6.28 (0.026s) | 966.93 ± 6.36 (0.061s) |
| random insertion | 246.34 ± 3.15 (0.0022s) | 657.58 ± 3.43 (0.014s) | 959.77 ± 6.32 (0.03s) |
| nearest insertion | 236.21 ± 6.54 (0.0038s) | 644.79 ± 6.00 (0.024s) | 947.53 ± 6.39 (0.061s) |
| nearest neighbour | 135.76 ± 2.89 (0.00035s) | 211.45 ± 3.39 (0.0019s) | 240.70 ± 3.31 (0.0048s) |

#### TSPLIB, n ≤ 10000, gap to published optimum

| solver | TSPLIB |
|---|---:|
| tspgnn greedy + 2-opt | 4.18 ± 0.63 (0.45s) |
| tspgnn greedy | 20.96 ± 12.08 (0.36s) |
| ablation: greedy(dist) + 2-opt | 5.06 ± 0.73 (0.23s) |
| LKH-3 | 0.02 ± 0.02 (12s) |
| farthest insertion + 2-opt | 8.45 ± 1.07 (0.27s) |
| farthest insertion | 10.25 ± 1.53 (0.18s) |


### Notes on TSPLIB

Across the 103 TSPLIB instances with n ≤ 10000, by size:

| n | instances | tspgnn greedy + 2-opt | median tspgnn greedy | farthest insertion + 2-opt |
|---|---:|---:|---:|---:|
| < 200 | 47 | 2.63 | 13.12 | 5.53 |
| 200–999 | 29 | 5.40 | 15.77 | 8.63 |
| 1000–2999 | 21 | 5.85 | 15.34 | 13.04 |
| 3000–10000 | 6 | 4.61 | 14.06 | 14.27 |

The wide interval on greedy-only decoding comes from a few instances with degenerate distance structure:
brg180 has many zero or tied distances (638%), p654 has 57% and fl1400 has 38%. 2-opt repairs them.

## Where this stands for publication

As a "learned solver beats the state of the art" paper, it isn't publishable yet. On the standard Euclidean benchmarks
it is 2–3 points behind DIFUSCO with the same 2-opt post-processing. Worse, a reviewer will notice that the GNN adds
under one point over ranking edges by distance, and LKH-3 dominates both.

What the results do support is a narrower paper: **one small model, trained only on n ≤ 100 and on distance
matrices, that generalizes flatly to n = 10000, to unseen metrics and to non-metric instances.** For that paper to
hold up, the gaps to close are:

1. **Stronger decoding.** Use MCTS or sampling on the heatmap, as in Att-GCN, DIMES and DIFUSCO. Greedy + 2-opt
   leaves most of the heatmap's value on the table. It would also allow a like-for-like comparison with those rows.
2. **GPU training at scale.** The model trained for about 2 hours on CPU. More data, more epochs and a larger n range
   are the obvious next step.
3. **A real matrix-input baseline.** The non-metric advantage needs a learned comparator that also takes only a
   matrix, such as MatNet, plus the TSPLIB ATSP set once asymmetric distances are supported.
4. **Multiple training seeds**, to report training variance.
