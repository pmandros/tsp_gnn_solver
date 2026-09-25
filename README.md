The Traveling Salesman Problem (TSP) is a classic optimization problem in the field of operations research and computer science. It poses a straightforward yet computationally challenging question: given a list of cities and the distances between each pair of them, what is the shortest possible route that visits each city exactly once and returns to the origin city? Despite its apparent simplicity, the TSP is known for its combinatorial explosion of possible routes as the number of cities increases, making it a quintessential NP-hard problem. This repository contains a notebook that solves the TSP using graph neural networks.

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
