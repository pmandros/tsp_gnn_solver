#!/usr/bin/env bash
# Sampling and guided-search decoding, GNN heatmap vs distance, as reported in results/SEARCH.md.
# Same suites and references as scripts/run_paper_bench.sh. The search runs for 2 ms per city
# (0.2 s at n = 100, 2 s at n = 1000, 20 s at n = 10000) on one core, after the greedy + 2-opt start.
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
W=${WORKERS:-4}
OUT=${OUT:-results/search}

S=(--solver "callable:fn=tspgnn.api:solve_search,guide=gnn,name=gnn+search,stochastic=true"
   --solver "callable:fn=tspgnn.api:solve_search,guide=dist,name=dist+search,stochastic=true")
SAMPLE=(--solver "callable:fn=tspgnn.api:solve_sample,guide=gnn,name=gnn+sample16,stochastic=true"
        --solver "callable:fn=tspgnn.api:solve_sample,guide=dist,name=dist+sample16,stochastic=true")

want() { [ -z "${SECTIONS:-}" ] || [[ " $SECTIONS " == *" $1 "* ]]; }
types() { for f in manhattan chebyshev clustered nonmetric; do echo "--suite ${f}$1:num=128"; done; }

want fu && python -m tspbench run --suite tsp500 --suite tsp1000 "${S[@]}" "${SAMPLE[@]}" \
  --seeds 0 1 2 --workers $W --out $OUT/fu

want kool && python -m tspbench run --suite tsp20 --suite tsp50 --suite tsp100 --limit 1280 \
  "${S[@]}" "${SAMPLE[@]}" --seeds 0 --workers $W --out $OUT/kool

want types && python -m tspbench run $(types 100) $(types 500) $(types 1000) "${S[@]}" "${SAMPLE[@]}" \
  --seeds 0 --workers $W --out $OUT/types

want tsplib && python -m tspbench run --suite "tsplib:max_n=10000" "${S[@]}" \
  --seeds 0 --workers $W --out $OUT/tsplib

want tsp10000 && python -m tspbench run --suite tsp10000 "${S[@]}" \
  --seeds 0 --workers 2 --out $OUT/tsp10000

# Time check: distance guide with twice the budget, GNN guide with half.
want timecheck && python -m tspbench run --suite tsp1000 --suite nonmetric1000:num=128 \
  --solver "callable:fn=tspgnn.api:solve_search,guide=dist,time_per_node=0.004,name=dist+search(2x),stochastic=true" \
  --solver "callable:fn=tspgnn.api:solve_search,guide=gnn,time_per_node=0.001,name=gnn+search(0.5x),stochastic=true" \
  --seeds 0 --workers $W --out $OUT/timecheck

true
