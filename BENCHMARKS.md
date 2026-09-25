# tspbench: benchmarks and baselines

`tspbench` is a model-agnostic evaluation harness for TSP solvers. It covers
three things:

- the standard test sets
- the classical baselines
- a runner that scores every solver the same way, with multiple seeds and
  confidence intervals

The GNN itself lives elsewhere. The harness talks to it only through a function
that takes a distance matrix.

## Install

```bash
pip install -r requirements-bench.txt   # numpy, scipy, OR-Tools, elkai (bundles LKH-3.0.8), pytest
pip install -e . --no-deps
python -m tspbench list                 # shows which solver backends are available
```

Concorde is optional:
`pip install 'pyconcorde @ git+https://github.com/jvkersch/pyconcorde'`, or
point `CONCORDE_BIN` at a `concorde` executable. Setting `LKH_BIN` to a
separately built LKH-3 selects it with `lkh:backend=binary`.

Download TSPLIB (symmetric and asymmetric) into `data/`:

```bash
python scripts/download_benchmarks.py
```

## Suites

| spec | instances | reference |
|---|---|---|
| `tsp20`, `tsp50`, `tsp100` | Kool et al. (2019) test sets: 10,000 uniform instances each, from the same generator and seed (1234) as their `generate_data.py` | `tspbench reference` (Concorde or LKH-3) |
| `tsp500`, `tsp1000`, `tsp10000` | Fu et al. (2021) test sets (128 / 128 / 16), read from `data/fu/tsp{n}_test_concorde.txt` | tours shipped in the file |
| `tsp500_gen`, … | same sizes, regenerated from seed 1234 (**not** the published instances) | `tspbench reference` |
| `tsplib` (`min_n=`, `max_n=`) | every TSPLIB95 `.tsp` with a published optimum | published optimum |
| `atsp_tsplib` | every TSPLIB95 `.atsp` with a published optimum | published optimum |
| `uniform{n}`, `manhattan{n}`, `chebyshev{n}` | unit square, L2 / L1 / L∞ | `tspbench reference` |
| `clustered{n}` (`num_clusters=`, `std=`) | Gaussian mixture | `tspbench reference` |
| `nonmetric{n}` | symmetric U(0,1) matrices (no triangle inequality) | `tspbench reference` |
| `atsp{n}` | MatNet-style asymmetric matrices with the triangle inequality | `tspbench reference` |
| `text:path=…`, `tsplib_dir:path=…` | any file or directory in those formats | file tours / optimum |

Generated families take `num=` and `seed=` options, e.g.
`clustered200:num=256,seed=7`. They accept any n, which is what the size
generalization sweeps need.

The Fu et al. files are not fetched automatically because they have no stable
URL. They are the `tsp{500,1000,10000}_test_concorde.txt` files published with
Att-GCRN+MCTS and reused by DIMES and DIFUSCO. Copy them into `data/fu/`.

Kool et al. regenerate their sets from seeds, and the harness does the same
(`tests/test_bench.py` checks the RNG stream). Before quoting TSP20/50/100
numbers in a paper, check a few instances against the released
`tsp*_test_seed1234.pkl` files.

## Solvers

A solver is named by a spec string: `name` or `name:key=value,...`.

| spec | type | notes |
|---|---|---|
| `concorde` | exact | Requires pyconcorde or `CONCORDE_BIN`. Solves ATSP through the 2n-node transformation when the weights fit in 32 bits. |
| `lkh` (`runs`, `max_trials`, `time_limit`, `backend`) | stochastic, seeded | LKH-3. Handles TSP and ATSP. |
| `ortools` (`time_limit`, `metaheuristic`) | deterministic | Guided local search by default. The stopping rule is wall-clock time. |
| `nearest_neighbor` | stochastic | Starts from a random node. |
| `nearest_insertion`, `farthest_insertion` | deterministic | |
| `random_insertion` | stochastic | |
| `cheapest_insertion` | deterministic | O(n³), so capped at `max_n=1000`. |
| `two_opt` (`init=…`, `time_limit`) | inherits `init` | Symmetric instances only. Uses dense vectorized search up to n=1000 and neighbor lists above that. |
| `exact_dp` | exact | Held–Karp for n ≤ 13, used in tests. |
| `callable:fn=pkg.mod:solve` | model | `solve(distance_matrix) -> tour` |
| `heatmap:fn=pkg.mod:predict,decode=greedy_edge,two_opt=true` | model | `predict(distance_matrix) -> (n, n) scores`, decoded by the harness |

LKH and Concorde work on integers. Float instances are scaled to about 1e6
before they are handed over. TSPLIB metrics pass through unchanged, so
published optima remain comparable.

## Plugging in a model

```python
# my_model/bench.py
def solve(dist):            # dist: (n, n) float64 numpy array
    ...                     # load a checkpoint once at import time, run the GNN, decode
    return tour             # permutation of range(n)
```

```bash
python -m tspbench run --suite tsp100 --suite tsplib:max_n=1000 --suite atsp50 \
    --solver "callable:fn=my_model.bench:solve,name=gnn" \
    --solver farthest_insertion --solver lkh --seeds 0 1 2 --workers 1
```

Extra `key=value` pairs are passed to the function as keyword arguments.
If the function has a `seed` parameter, it receives the run seed.
`input=coords` passes coordinates instead of the matrix.

## Running

```bash
# cache reference lengths for generated suites once (Concorde if available, otherwise LKH-3)
python -m tspbench reference --suite tsp100 --solver concorde --workers 8
python -m tspbench reference --suite atsp100 --solver "lkh:runs=10"

python -m tspbench run --suite tsp100 --solver farthest_insertion --solver "ortools:time_limit=1" \
    --solver lkh --seeds 0 1 2 3 4 --workers 8 --out results/tsp100
```

Each run writes four files:

- `records.csv`: one row per (instance, solver, seed), with the status, the float tour length, the reference, the gap and the time.
- `summary.csv` and `summary.md`: the mean gap, with a 95% t-interval across seeds for stochastic solvers and across instances for deterministic ones.
- `run.json`: the command line, git commit, versions and CPU count.

## Scoring

The scoring rules follow from the problems found in the notebook:

- Every tour is checked to be a permutation. It is then re-scored in float64
  with the instance's own distance function. Nothing is compared against
  Concorde's rounded `optimal_value`.
- Gaps are reported two ways: against the reference (optimum, file, or cached
  reference solver, named in the `reference` column), and against the best tour
  any solver found in the same run.
- A failure (unsupported instance, error, invalid tour) is recorded as a status,
  never dropped. A seed's mean gap is only reported when every instance
  succeeded for that seed.
- Times are wall-clock per instance. With `--workers > 1`, solvers share the
  CPU, so use `--workers 1` for runtime tables.
