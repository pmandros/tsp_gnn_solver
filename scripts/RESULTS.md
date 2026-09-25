# Old vs new results

Produced by `run_all.sh` then `summarize.py` on CPU. The data are 5,000 uniform instances with n in {10,15,20,25,30} (70/10/20 split), plus 100 instances with n=100. Every reference tour is exact, from `exact_tsp.py`. Values are the mean relative gap of the greedy tour in %, ± the std over seeds 0, 1 and 2.

- **old:** the test gap is taken at the last epoch, which is how the notebook reported it. Its reference is the optimum for integer-rounded distances.
- **new:** the test gap is taken at the epoch with the best held-out validation gap. Its reference is the float optimum.

| setup | seeds | test gap, n=10–30 (reported ref) | test gap vs float optimum | gap at n=100 vs float optimum | epoch used |
|---|---|---|---|---|---|
| old (notebook as was) | 3 | 11.73 ± 0.84 | 11.71 ± 0.82 | 22.56 ± 0.47 | 17, 17, 17 |
| new, distance weights | 3 | 8.77 ± 0.48 | 8.77 ± 0.48 | 21.34 ± 0.20 | 17, 15, 15 |
| new, unweighted mean | 3 | 9.03 ± 0.60 | 9.03 ± 0.60 | 21.36 ± 0.80 | 12, 13, 15 |

Notes
- The rounded reference barely moves the average: the rounded-distance optimum differs from the float optimum by 0.27% per instance in absolute terms, but by only -0.01% on average, because the errors cancel. The fix is still needed for per-instance numbers and larger scales.
- Distance-weighted aggregation beats the unweighted mean by about 0.3 points at n=10–30 and ties at n=100. That is within seed noise.
- The validation gap swings by 2–4 points from epoch to epoch, so a single-epoch number is noisy. Report several seeds.
