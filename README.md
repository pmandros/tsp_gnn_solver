The Traveling Salesman Problem (TSP) is a classic optimization problem in the field of operations research and computer science. It poses a straightforward yet computationally challenging question: given a list of cities and the distances between each pair of them, what is the shortest possible route that visits each city exactly once and returns to the origin city? Despite its apparent simplicity, the TSP is known for its combinatorial explosion of possible routes as the number of cities increases, making it a quintessential NP-hard problem. This repository contains a notebook that solves the TSP using graph neural networks.

## Train on a GPU in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pmandros/tsp_gnn_solver/blob/main/colab_gpu_training.ipynb)

[`colab_gpu_training.ipynb`](colab_gpu_training.ipynb) trains `tspgnn` with several seeds on a Colab GPU and evaluates every seed with `tspbench`, both greedy + 2-opt and guided search against the distance-guided baseline. Choose a GPU runtime and press *Run all*. Data, checkpoints and results go to Google Drive, and re-running after a disconnect resumes where it stopped.

## Benchmarks and baselines

`tspbench/` is a model-agnostic evaluation harness. It includes:

- Test suites: Kool TSP20/50/100, Fu TSP500/1000/10000, TSPLIB, TSPLIB ATSP, and generated clustered, non-metric and asymmetric sets of any size.
- Baselines: Concorde, LKH-3, OR-Tools, nearest-neighbor, insertion heuristics and 2-opt.
- A runner that reports gaps with 95% confidence intervals over seeds.

A model plugs in as a `solve(distance_matrix) -> tour` function. See [BENCHMARKS.md](BENCHMARKS.md).

```bash
pip install -r requirements-bench.txt && pip install -e . --no-deps
python -m tspbench run --suite tsp50 --limit 100 --solver farthest_insertion --solver lkh --seeds 0 1 2
```

## Distance-matrix-only GNN (`tspgnn/`)

`tspgnn` is a rewrite of the notebook model. Its only input is a distance matrix, so the same model runs on Euclidean, non-Euclidean and non-metric instances of any size.

- **Input.** The input is a sparse kNN graph with k=20, symmetrised. The model gets no coordinates. Edge features are the distance over the instance's mean kNN distance, the distance over each endpoint's local kNN scale, and the two kNN ranks. Node features are the local scales. Every feature is invariant to rescaling the matrix.
- **Model.** The model is an anisotropic gated GCN (Joshi et al. 2019) with LayerNorm instead of BatchNorm. Neighbour aggregation is gate-normalised over a bounded neighbourhood, and there is no global pooling, so no statistic depends on n or on the batch. Edge scores are symmetrised.
- **Training.** Training is supervised with symmetric LKH-3 labels (via `elkai`) and a held-out validation set used for checkpoint selection. The training data mixes sizes (20–100), distance types and scales in the same batch.
- **Decoding.** Greedy edge insertion on the GNN scores builds the tour, and neighbour-list 2-opt then improves it.

```bash
pip install -r requirements.txt
python -m pytest tests
python scripts/make_data.py --out data/train.pt --num 40000 --seed 1
python scripts/make_data.py --out data/val.pt --num 600 --seed 2 --store-d
python scripts/train.py --train 'data/train_*.pt' --val data/val.pt --out runs/main
python scripts/evaluate.py --ckpt runs/main/best.pt --out results/main.json
```

`tspgnn.api.solve(distance_matrix) -> tour` and `tspgnn.api.predict(distance_matrix) -> heatmap` are plain-function entry points that other code, such as a benchmark harness, can call. Results up to n=1000 are in [results/RESULTS.md](results/RESULTS.md). The full benchmark against classical and published learned solvers (TSP20–10000, TSPLIB, held-out metrics) is in [results/BENCHMARK.md](results/BENCHMARK.md).
