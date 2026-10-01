# Heatmap crossover: tspgnn vs SoftDist, DIFUSCO, DIMES and Att-GCN under the same searches

[SEARCH.md](SEARCH.md) compared tspgnn's heatmap only against plain distance. This file puts every published
heatmap through the **same** search with the **same** time budget. It does this for our guided search and for the
heatmap-guided MCTS that Att-GCN, DIMES and DIFUSCO report. The question is whether tspgnn's heatmap is
competitive with the published ones, and whether a learned heatmap beats the model-free SoftDist baseline of Xia et
al. (ICML 2024). That paper argues learned heatmaps add little.

All runs use the Fu et al. TSP500/1000 test sets (128 instances each), report gaps to the LKH-3 references in
`data/refs/`, and run on one CPU core per instance.

## Where the heatmaps come from

| heatmap | source | trained on |
|---|---|---|
| **tspgnn** | `checkpoints/tspgnn.pt` (this repo), sigmoid of edge logits on the 20-NN graph | one model, n = 20–100, CPU, 8 epochs, distance matrix only |
| DIFUSCO | released by Xia et al. (regenerated with DIFUSCO's code) | separate models per size, on n = 500 / 1000, GPU, coordinates |
| DIMES | released by Xia et al. (from DIMES's repository) | per size, coordinates |
| Att-GCN | released by Xia et al. (from Att-GCN's repository) | n = 20–100 plus sub-graph sampling, coordinates |
| SoftDist | softmax(−d / τ) per row, τ = 0.0066 / 0.0051 (Xia et al.'s tuned values) | no training |
| plain distance | our search: neighbours ranked by distance (weights 1/rank); MCTS: a 1/(rank+1) prior over the 20 nearest neighbours | no training |

Heatmap inference time is not counted for any method. The published heatmaps were precomputed on GPUs. tspgnn's takes
0.09 s (n = 500) and 0.17 s (n = 1000) on one CPU core.

## 1. Our guided search (`tspgnn/search.py`), 2 ms per city

Seeds 0–2, mean gap in %, with the start gap (greedy edge decode + 2-opt) in parentheses.

| heatmap | TSP500 (1 s) | TSP1000 (2 s) | TSP1000, 0.5 ms/city (0.5 s, seed 0) |
|---|---:|---:|---:|
| **tspgnn** | **0.47** (4.05) | **0.54** (4.08) | **0.74** |
| DIFUSCO | **0.47** (2.00) | 0.62 (2.67) | 0.76 |
| SoftDist | 0.63 (4.54) | 0.72 (4.44) | 0.95 |
| plain distance | 0.69 (4.60) | 0.75 (4.44) | 1.00 |
| Att-GCN | 0.75 (5.32) | 0.95 (5.65) | 1.27 |
| DIMES | 0.84 (5.21) | 1.03 (5.52) | 1.35 |

Paired per-instance differences (mean over seeds, 95% interval):

- tspgnn − DIFUSCO: −0.005 ± 0.055 on TSP500 (a tie), and **−0.077 ± 0.046 on TSP1000**. tspgnn is better on 62% of instances.
- tspgnn − SoftDist: −0.16 ± 0.06 on TSP500 and −0.18 ± 0.04 on TSP1000.

DIFUSCO's heatmap gives a much better *starting* tour (2.0% vs 4.1%), but under this search tspgnn ends level at
n = 500 and ahead at n = 1000. At TSP500 at 0.5 ms/city, DIFUSCO is slightly ahead (0.56 vs 0.60, one seed).

## 2. The heatmap-guided MCTS of Fu et al. / DIMES / DIFUSCO

This is the default-parameter C++ MCTS released by Xia et al., with one fix: `Read_Heatmap()` never returns, which
crashes under GCC 13 at -O3. It stops after 0.01 · n s of CPU time per instance (5 s / 10 s), a tenth of the default
0.1 · n. Seed 0 only.

| heatmap | TSP500 (5 s) | TSP1000 (10 s) |
|---|---:|---:|
| DIFUSCO | **0.69** | **1.72** |
| **tspgnn** | 1.62 | 2.28 |
| SoftDist | 2.22 | 2.78 |
| Att-GCN | 2.24 | 2.95 |
| DIMES | 2.35 | 3.06 |
| plain distance (rank prior) | 3.00 | 3.78 |

Here tspgnn is second, clearly behind DIFUSCO and clearly ahead of SoftDist and the other learned heatmaps.
This MCTS uses the heatmap more literally than our search does. Its candidate set is every edge with value ≥ 1e-4,
and its initial tours are sampled in proportion to e^heatmap. tspgnn's per-edge sigmoids are not normalized per
city: a city often has 5–6 edges above 0.5, against two edges near 0.45 for DIFUSCO. So we tried simple
reshapings of tspgnn's heatmap (TSP500, same budget):

| tspgnn heatmap variant | TSP500 (5 s) |
|---|---:|
| as is | 1.62 |
| row-normalized, then symmetrized | 1.63 |
| top-10 per city | 1.52 |
| squared | 1.34 |
| ^4 | 1.35 |
| **top-5 per city** | **1.32** |

These variants were chosen on the test set itself, so the best one is optimistic. Sharpening helps by about 0.3 points but does not close the gap to DIFUSCO (0.69). Under this MCTS, DIFUSCO's heatmap is
genuinely better than ours.

## What this says about novelty

Checked against the literature (October 2026):

- **SoftDist** (Xia et al., ICML 2024) and **"Beyond the Heatmap"** (Pan et al., ICLR 2026) argue that, with a well-tuned
  search, learned heatmaps add little over distance-based priors. Pan et al.'s zero-parameter k-NN prior reaches
  0.50 / 0.85 / 2.14% on TSP500/1000/10000 with tuned MCTS, against 0.33 / 0.53 / 2.37% for DIFUSCO. Our results
  partly contradict the claim, and partly support it. Under both searches, a learned heatmap (ours and DIFUSCO's)
  beats SoftDist and distance consistently. The size of the gain depends on the search: 0.16–0.18 points with ours,
  0.5–0.6 points with the MCTS.
- **Matrix-only, any-metric neural solvers exist.** MatNet (NeurIPS 2021) and UniCO / MatPOENet / MatDIFFNet
  (ICLR 2025) take a distance matrix, including non-metric and asymmetric ones. GREAT (2024) is an edge-based GNN for
  non-Euclidean and asymmetric routing. In this project MatNet runs only to n ≤ 256 and fails on our non-metric sets
  (see the MatNet thread). We have not yet run GREAT or UniCO, so "first matrix-only solver" cannot be claimed.
- **Size generalization from small training** is well studied: Att-GCN, LEHD, BQ-NCO, INViT (ICML 2024; 5.99 / 7.86% on
  TSP1000 / 10000 when trained on n = 100) and L2R (2025). These use coordinates and mostly constructive decoders.

The defensible claim, given these numbers, is this. **One small, coordinate-free GNN, trained on n ≤ 100 on a CPU,
gives a heatmap that matches DIFUSCO under a fast guided search at n = 500, beats it at n = 1000, and beats SoftDist
under both searches. The same model also works on held-out metrics and non-metric matrices (SEARCH.md), where
coordinate-based heatmaps do not apply.** It is weaker than DIFUSCO when plugged into the original MCTS.

Gaps that a paper would still need to close:

1. TSP10000 crossover. The released 10k heatmaps are dense text files of several GB each and did not fit in this sandbox.
2. MCTS at its default budget (0.1 · n s) and with tuned parameters, as Pan et al. do. This takes hours of CPU per heatmap.
3. Comparisons with GREAT and UniCO on non-metric matrices.
4. Multi-seed training (the Colab notebook in PR #7).

## Reproduce

```bash
git clone https://github.com/xyfffff/rethink_mcts_for_tsp tools/rethink
for n in 500 1000; do
  python scripts/import_rethink_heatmaps.py --repo tools/rethink --n $n   # DIFUSCO, DIMES, Att-GCN
  python scripts/export_heatmaps.py --n $n                                # tspgnn, SoftDist, distance
  python scripts/heatmap_crossover.py --n $n --seeds 0 1 2                # our search
  python scripts/heatmap_crossover.py --n $n --time-per-node 0.0005
  python scripts/run_mcts_crossover.py --repo tools/rethink --n $n --param-t 0.01   # their MCTS
done
```

Results are in `results/crossover/`. The TSP1000 runs of our search are split by seed into `s0/`–`s2/`.

Sources: Xia et al., *Position: Rethinking Post-Hoc Search-Based Neural Approaches for Solving Large-Scale TSP*,
ICML 2024 (<https://github.com/xyfffff/rethink_mcts_for_tsp>). Pan et al., *Beyond the Heatmap*, ICLR 2026
(<https://proceedings.iclr.cc/paper_files/paper/2026/file/e3bf2f0f10774c474de22a12cb060e2c-Paper-Conference.pdf>).
*A GREAT Architecture for Edge-Based Graph Problems Like TSP* (<https://arxiv.org/abs/2408.16717>).
*UniCO*, ICLR 2025 (<https://mlanthology.org/iclr/2025/pan2025iclr-unico/>). *INViT*, ICML 2024 (<https://arxiv.org/abs/2402.02317>).
