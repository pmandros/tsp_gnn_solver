#!/usr/bin/env bash
# Full benchmark of tspgnn against classical solvers, as reported in results/BENCHMARK.md.
# Prerequisites: scripts/download_benchmarks.py --concorde tools/concorde, and the
# references in data/refs/ (see the `reference` commands at the bottom of this file).
set -euo pipefail
export CONCORDE_BIN=${CONCORDE_BIN:-$PWD/tools/concorde}
# One core per solver process, so per-instance times are single-core times.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
W=${WORKERS:-4}
# SECTIONS="kool fu" runs only those sections (default: all).
SEEDS="0 1 2"
OUT=${OUT:-results/bench}

GNN=(--solver "callable:fn=tspgnn.api:solve,name=gnn+2opt"
     --solver "callable:fn=tspgnn.api:solve_without_two_opt,name=gnn")
CLASSIC=(--solver nearest_neighbor --solver nearest_insertion --solver farthest_insertion
         --solver random_insertion --solver "two_opt:init=farthest_insertion")

want() { [ -z "${SECTIONS:-}" ] || [[ " $SECTIONS " == *" $1 "* ]]; }
types() { for f in manhattan chebyshev clustered nonmetric; do echo "--suite ${f}$1:num=128"; done; }

# Kool et al. TSP20/50/100: the first 1280 of the 10,000 test instances.
want kool && python -m tspbench run --suite tsp20 --suite tsp50 --suite tsp100 --limit 1280 \
  "${GNN[@]}" "${CLASSIC[@]}" --solver lkh --solver "ortools:time_limit=1" \
  --seeds $SEEDS --workers $W --out $OUT/kool

# Fu et al. TSP500/1000 (128 instances each).
want fu && python -m tspbench run --suite tsp500 --suite tsp1000 \
  "${GNN[@]}" "${CLASSIC[@]}" --solver lkh --solver "ortools:time_limit=10" \
  --seeds $SEEDS --workers $W --out $OUT/fu

# Distance types; manhattan and chebyshev were held out of training.
want types && python -m tspbench run $(types 100) $(types 500) $(types 1000) \
  "${GNN[@]}" "${CLASSIC[@]}" --solver lkh \
  --seeds $SEEDS --workers $W --out $OUT/types

# TSPLIB up to 10k nodes (published optima).
want tsplib && python -m tspbench run --suite "tsplib:max_n=10000" \
  "${GNN[@]}" --solver farthest_insertion --solver "two_opt:init=farthest_insertion" \
  --solver "lkh:time_limit=60" --seeds $SEEDS --workers $W --out $OUT/tsplib

# Fu et al. TSP10000 (16 instances; reference is LKH-3 with max_trials=1000, 300 s).
want tsp10000 && python -m tspbench run --suite tsp10000 \
  "${GNN[@]}" --solver nearest_neighbor --solver farthest_insertion \
  --solver "two_opt:init=farthest_insertion" \
  --seeds $SEEDS --workers 2 --out $OUT/tsp10000

# References used above (run once; cached under data/refs/):
#   python -m tspbench reference --suite tsp20 --suite tsp50 --suite tsp100 --limit 1280 --solver concorde --workers 4
#   python -m tspbench reference $(types 100) --solver concorde --workers 4
#   python -m tspbench reference $(types 500) $(types 1000) --solver lkh --workers 4

# Ablation: tspgnn's decoder and 2-opt with edges ranked by distance, on every suite above.
want ablation && python -m tspbench run --suite tsp20 --suite tsp50 --suite tsp100 --limit 1280 \
  --suite tsp500 --suite tsp1000 $(types 100) $(types 500) $(types 1000) --suite "tsplib:max_n=10000" \
  --suite tsp10000 --solver "callable:fn=tspgnn.api:solve_distance_greedy,name=greedy(dist)+2opt" \
  --seeds 0 --workers 2 --out $OUT/ablation

true
