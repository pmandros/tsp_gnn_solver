# Results: distance-matrix-only GNN

Checkpoint: `checkpoints/tspgnn.pt` (257k parameters, 12 gated-GCN layers, hidden size 64). It was trained for 8 epochs on CPU, about 2 hours, on 40k LKH-labelled instances with n=20–100. The training types were `euclidean`, `clustered` and `random` (a non-metric symmetric matrix), with random global scales between 1e-2 and 1e3. The test instances use seeds disjoint from training.

Entries are the mean gap to LKH-3 in %, ± the standard error of the mean. Every tour is scored on the float matrix. Instance counts are 128 for n ≤ 100, 64 for n=200, 32 for n=500 and 16 for n=1000. *(held out)* marks distance types never seen in training. Sizes above 100 were also never seen in training. Times are per instance, on one CPU, and do not include numba JIT warm-up.

Columns:

- **NN+2opt:** nearest neighbour followed by 2-opt.
- **greedy(dist):** greedy edge insertion on the same kNN candidate edges, ranked by distance.
- **greedy(GNN):** the same greedy decoder, but with edges ranked by the GNN score.
- **+2opt:** neighbour-list 2-opt run after decoding.

Reproduce with:

```bash
python scripts/evaluate.py --ckpt checkpoints/tspgnn.pt --out results/main.json
python scripts/report.py results/main.json
```

| type | n | NN+2opt | greedy(dist) | greedy(GNN) | greedy(dist)+2opt | greedy(GNN)+2opt | GNN+2opt time | LKH time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| euclidean | 20 | 2.21 ± 0.25 | 11.97 ± 0.71 | 5.91 ± 0.60 | 2.10 ± 0.23 | **1.21 ± 0.16** | 0.006s | 0.00s |
| euclidean | 50 | 4.11 ± 0.23 | 17.03 ± 0.60 | 12.26 ± 0.50 | 3.58 ± 0.22 | **2.95 ± 0.20** | 0.008s | 0.04s |
| euclidean | 100 | 5.00 ± 0.20 | 17.87 ± 0.48 | 13.21 ± 0.37 | 4.28 ± 0.19 | **3.48 ± 0.18** | 0.011s | 0.20s |
| euclidean | 200 | 5.62 ± 0.21 | 17.68 ± 0.54 | 15.28 ± 0.47 | 4.83 ± 0.21 | **3.96 ± 0.15** | 0.018s | 1.38s |
| euclidean | 500 | 5.87 ± 0.15 | 17.44 ± 0.42 | 16.49 ± 0.41 | 4.73 ± 0.15 | **4.03 ± 0.13** | 0.042s | 1.96s |
| euclidean | 1000 | 5.70 ± 0.13 | 17.13 ± 0.38 | 15.37 ± 0.35 | 4.69 ± 0.14 | **3.95 ± 0.13** | 0.083s | 9.28s |
| clustered | 20 | 1.09 ± 0.14 | 9.44 ± 0.56 | 8.24 ± 0.81 | 0.87 ± 0.12 | **0.54 ± 0.08** | 0.006s | 0.00s |
| clustered | 50 | 2.77 ± 0.20 | 14.85 ± 0.59 | 16.96 ± 1.07 | 2.76 ± 0.21 | **2.25 ± 0.17** | 0.007s | 0.04s |
| clustered | 100 | 4.99 ± 0.23 | 18.65 ± 0.57 | 19.36 ± 0.61 | 4.97 ± 0.25 | **4.32 ± 0.26** | 0.010s | 0.48s |
| clustered | 200 | 6.88 ± 0.33 | 19.85 ± 0.61 | 18.98 ± 0.60 | 7.07 ± 0.45 | **5.76 ± 0.33** | 0.018s | 3.17s |
| clustered | 500 | 7.52 ± 0.44 | 18.52 ± 0.64 | 17.52 ± 0.54 | 7.23 ± 0.47 | **6.53 ± 0.57** | 0.049s | 5.87s |
| clustered | 1000 | 7.29 ± 0.38 | 19.67 ± 0.53 | 16.60 ± 0.64 | 7.04 ± 0.47 | **6.06 ± 0.25** | 0.096s | 31.59s |
| manhattan *(held out)* | 20 | 2.75 ± 0.32 | 13.20 ± 0.78 | 7.44 ± 0.62 | 2.46 ± 0.30 | **1.13 ± 0.16** | 0.005s | 0.00s |
| manhattan *(held out)* | 50 | 5.05 ± 0.31 | 16.59 ± 0.64 | 13.38 ± 0.58 | 4.32 ± 0.26 | **3.63 ± 0.22** | 0.007s | 0.04s |
| manhattan *(held out)* | 100 | 6.04 ± 0.22 | 17.32 ± 0.44 | 15.04 ± 0.39 | 4.97 ± 0.20 | **3.86 ± 0.17** | 0.010s | 0.22s |
| manhattan *(held out)* | 200 | 6.56 ± 0.28 | 17.34 ± 0.48 | 16.00 ± 0.49 | 4.92 ± 0.21 | **4.17 ± 0.17** | 0.016s | 1.36s |
| manhattan *(held out)* | 500 | 6.83 ± 0.24 | 17.13 ± 0.45 | 16.39 ± 0.58 | 5.07 ± 0.17 | **4.37 ± 0.20** | 0.038s | 2.24s |
| manhattan *(held out)* | 1000 | 6.96 ± 0.21 | 15.82 ± 0.38 | 16.10 ± 0.35 | 5.58 ± 0.19 | **4.45 ± 0.18** | 0.083s | 8.93s |
| chebyshev *(held out)* | 20 | 2.72 ± 0.31 | 15.63 ± 0.71 | 7.02 ± 0.52 | 2.10 ± 0.23 | **1.35 ± 0.16** | 0.005s | 0.00s |
| chebyshev *(held out)* | 50 | 4.54 ± 0.23 | 18.07 ± 0.57 | 11.99 ± 0.54 | 3.55 ± 0.24 | **3.18 ± 0.21** | 0.008s | 0.04s |
| chebyshev *(held out)* | 100 | 5.72 ± 0.21 | 19.96 ± 0.46 | 14.77 ± 0.38 | 4.70 ± 0.21 | **4.03 ± 0.19** | 0.010s | 0.20s |
| chebyshev *(held out)* | 200 | 6.18 ± 0.20 | 18.11 ± 0.54 | 15.36 ± 0.38 | 4.62 ± 0.19 | **4.17 ± 0.17** | 0.019s | 1.36s |
| chebyshev *(held out)* | 500 | 6.18 ± 0.22 | 18.75 ± 0.71 | 16.76 ± 0.57 | 5.23 ± 0.20 | **4.35 ± 0.20** | 0.037s | 2.93s |
| chebyshev *(held out)* | 1000 | 6.80 ± 0.26 | 18.29 ± 0.34 | 15.50 ± 0.45 | 5.00 ± 0.21 | **4.24 ± 0.16** | 0.084s | 9.98s |
| random | 20 | 14.48 ± 0.90 | 43.01 ± 2.01 | 20.14 ± 1.53 | 11.90 ± 0.70 | **3.41 ± 0.35** | 0.005s | 0.00s |
| random | 50 | 24.17 ± 0.71 | 72.86 ± 2.32 | 39.44 ± 1.64 | 21.40 ± 0.74 | **7.25 ± 0.42** | 0.007s | 0.02s |
| random | 100 | 39.95 ± 0.83 | 98.69 ± 1.91 | 54.65 ± 1.75 | 30.08 ± 0.66 | **9.89 ± 0.36** | 0.010s | 0.09s |
| random | 200 | 53.99 ± 1.13 | 116.69 ± 2.71 | 69.68 ± 2.27 | 39.66 ± 0.82 | **14.69 ± 0.54** | 0.017s | 0.35s |
| random | 500 | 73.27 ± 1.24 | 152.20 ± 4.31 | 91.32 ± 3.56 | 54.76 ± 1.05 | **21.02 ± 0.91** | 0.041s | 0.83s |
| random | 1000 | 85.75 ± 1.52 | 179.16 ± 6.36 | 103.10 ± 4.44 | 63.40 ± 1.22 | **24.57 ± 0.76** | 0.081s | 3.90s |
| shortest_path *(held out)* | 20 | 2.81 ± 0.23 | 10.59 ± 0.52 | 4.05 ± 0.31 | 2.32 ± 0.20 | **1.26 ± 0.15** | 0.005s | 0.01s |
| shortest_path *(held out)* | 50 | 4.37 ± 0.19 | 12.51 ± 0.29 | 4.27 ± 0.20 | 3.96 ± 0.18 | **2.03 ± 0.12** | 0.007s | 0.12s |
| shortest_path *(held out)* | 100 | 5.98 ± 0.15 | 12.93 ± 0.21 | 4.58 ± 0.14 | 4.88 ± 0.13 | **2.32 ± 0.09** | 0.011s | 0.55s |
| shortest_path *(held out)* | 200 | 7.31 ± 0.20 | 13.09 ± 0.26 | 4.72 ± 0.12 | 5.75 ± 0.17 | **2.58 ± 0.09** | 0.020s | 2.87s |
| shortest_path *(held out)* | 500 | 8.92 ± 0.16 | 13.52 ± 0.19 | 4.77 ± 0.11 | 6.64 ± 0.17 | **3.06 ± 0.08** | 0.048s | 5.35s |
| shortest_path *(held out)* | 1000 | 10.03 ± 0.22 | 13.54 ± 0.20 | 4.59 ± 0.09 | 7.06 ± 0.08 | **3.09 ± 0.08** | 0.095s | 21.56s |
