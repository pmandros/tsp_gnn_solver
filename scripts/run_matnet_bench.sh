#!/usr/bin/env bash
# tspgnn vs MatNet on non-metric and asymmetric instances, as reported in results/MATNET.md.
# Prerequisites:
#   python scripts/download_benchmarks.py --no-fu --matnet tools/MatNet   (TSPLIB ATSP + MatNet)
#   references in data/refs/ (see the `reference` command at the bottom)
set -euo pipefail
export MATNET_DIR=${MATNET_DIR:-$PWD/tools/MatNet}
# One core per solver process, so per-instance times are single-core times.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
W=${WORKERS:-4}
OUT=${OUT:-results/bench}
want() { [ -z "${SECTIONS:-}" ] || [[ " $SECTIONS " == *" $1 "* ]]; }

NONMETRIC=(--suite nonmetric20:num=128 --suite nonmetric50:num=128 --suite nonmetric100:num=128)
ATSP=(--suite atsp20:num=128 --suite atsp50:num=128 --suite atsp100:num=128)

# Symmetric non-metric matrices: tspgnn trained on this family; MatNet did not (it saw only ATSP "tmat").
want nonmetric && python -m tspbench run "${NONMETRIC[@]}" \
  --solver "callable:fn=tspgnn.api:solve,name=gnn+2opt" \
  --solver "callable:fn=tspgnn.api:solve_distance_greedy,name=greedy(dist)+2opt" \
  --solver matnet --solver nearest_neighbor --solver lkh \
  --seeds 0 1 2 --workers $W --out $OUT/matnet_nonmetric

# Does MatNet's non-metric collapse come from input scale? (U(0,1) entries are ~10x tmat entries at n=100.)
want scaled && python -m tspbench run "${NONMETRIC[@]}" \
  --solver "matnet:scale=0.3" --solver "matnet:scale=0.1" --solver "matnet:scale=0.03" \
  --seeds 0 --workers $W --out $OUT/matnet_nonmetric_scaled

# Asymmetric "tmat" matrices: MatNet's training distribution. tspgnn is symmetric-only, so it runs
# on (D + D^T) / 2 and keeps the cheaper direction (tspgnn.api:solve_symmetrized).
want atsp && python -m tspbench run "${ATSP[@]}" \
  --solver "callable:fn=tspgnn.api:solve_symmetrized,name=gnn(sym)+2opt" \
  --solver "callable:fn=tspgnn.api:solve_symmetrized,gnn=false,name=greedy(dist-sym)+2opt" \
  --solver matnet --solver nearest_neighbor --solver lkh \
  --seeds 0 1 2 --workers $W --out $OUT/matnet_atsp

# TSPLIB ATSP (published optima, integer costs). MatNet gets the matrix divided by its maximum.
# It cannot take n > 256, so this drops rbg323/358/403/443.
want tsplib && python -m tspbench run --suite "atsp_tsplib:max_n=256" \
  --solver "callable:fn=tspgnn.api:solve_symmetrized,name=gnn(sym)+2opt" \
  --solver "matnet:scale=max" \
  --solver nearest_neighbor --solver lkh \
  --seeds 0 1 2 --workers $W --out $OUT/matnet_atsp_tsplib

# MatNet with x128 augmentation, as in its paper. It costs about 150 s per n=100 instance on one CPU
# core, so this runs on the first 32 instances of each suite with one seed.
want aug && python -m tspbench run "${NONMETRIC[@]}" "${ATSP[@]}" --limit 32 \
  --solver "matnet:aug=128" --seeds 0 --workers $W --out $OUT/matnet_aug128
want aug && python -m tspbench run --suite "atsp_tsplib:max_n=256" \
  --solver "matnet:aug=128,scale=max" --seeds 0 --workers $W --out $OUT/matnet_aug128_tsplib

# References used above (run once; cached under data/refs/; nonmetric100 is shared with run_paper_bench.sh):
#   python -m tspbench reference --suite nonmetric20:num=128 --suite nonmetric50:num=128 "${ATSP[@]}" --solver "lkh:runs=10" --workers 2

true
