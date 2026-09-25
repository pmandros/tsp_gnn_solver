#!/usr/bin/env bash
# Old vs new comparison over three seeds. Expects the datasets from make_dataset.py.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
for seed in 0 1 2; do
  python compare_old_new.py --mode old --seed $seed --out results/old_s$seed.json
  python compare_old_new.py --mode new --seed $seed --out results/new_s$seed.json
  python compare_old_new.py --mode new --no-edge-weight --seed $seed --out results/new_noweight_s$seed.json
done
