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

Download TSPLIB (symmetric and asymmetric, with optimal tours), the Fu et al.
TSP500/1000/10000 files, and optionally a prebuilt Linux Concorde binary:

```bash
python scripts/download_benchmarks.py --concorde tools/concorde
export CONCORDE_BIN=$PWD/tools/concorde
```

The binary is the one pyconcorde ships; the build from source needs QSopt from
the Waterloo site. `pip install 'pyconcorde @ git+https://github.com/jvkersch/pyconcorde'`
also works where that site is reachable. Setting `LKH_BIN` to a separately
built LKH-3 selects it with `lkh:backend=binary`.

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

The Fu et al. files are fetched from the authors' repository (Spider-scnu/TSP)
at a pinned commit. **Do not use their tours as references.** The TSP10000 file
carries the identity permutation as a placeholder, so the loader ignores it.
The TSP500 and TSP1000 tours average 16.584 and 23.227, while the optima
reported in the literature are 16.55 and 23.12, so the shipped tours are about
0.2–0.5% longer than optimal. The repository ships LKH-3 references in `data/refs/`:

- TSP500 and TSP1000 use default settings. They average 16.546 and 23.119,
  matching the literature, and on 16 TSP500 instances they are within 0.004%
  of Concorde.
- TSP10000 uses `max_trials=1000,time_limit=300` and averages 71.778; the
  literature's LKH-3 figure is 71.77.

A cached reference takes precedence over the file tour.

Kool et al. regenerate their sets from seeds. Their `generate_data.py` calls
`np.random.seed(1234)` and then draws `np.random.uniform(size=(10000, n, 2))`
for each size, and `tspbench` does the same. `tests/test_bench.py` checks that
the two RNG streams match.

## Solvers

A solver is named by a spec string: `name` or `name:key=value,...`.

| spec | type | notes |
|---|---|---|
| `concorde` | exact | Requires `CONCORDE_BIN` or pyconcorde. Exact on symmetric TSPLIB. For ATSP it goes through the 2n-node transformation: exact on TSPLIB ATSP, near-exact on float matrices, which are rescaled so Concorde doesn't crash. |
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
