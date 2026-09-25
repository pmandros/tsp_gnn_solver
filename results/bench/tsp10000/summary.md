| suite | solver | n inst | seeds | gap vs ref | gap vs best in run | time / inst | ref | failed |
|---|---|---:|---:|---:|---:|---:|---|---:|
| tsp10000 | `callable:fn=tspgnn.api:solve,name=gnn+2opt` | 16 | 1 | 3.899% ± 0.068% | 3.899% ± 0.068% | 14s | lkh:max_trials=1000,time_limit=300 | 0 |
| tsp10000 | `callable:fn=tspgnn.api:solve_without_two_opt,name=gnn` | 16 | 1 | 14.822% ± 0.317% | 14.822% ± 0.317% | 11.6s | lkh:max_trials=1000,time_limit=300 | 0 |
| tsp10000 | `nearest_neighbor` | 16 | 3 | 23.789% ± 0.821% | 23.789% ± 0.821% | 1.37s | lkh:max_trials=1000,time_limit=300 | 0 |
| tsp10000 | `farthest_insertion` | 16 | 1 | 12.279% ± 0.152% | 12.279% ± 0.152% | 5.85s | lkh:max_trials=1000,time_limit=300 | 0 |
| tsp10000 | `two_opt:init=farthest_insertion` | 16 | 1 | 10.992% ± 0.162% | 10.992% ± 0.162% | 7.72s | lkh:max_trials=1000,time_limit=300 | 0 |

± is a 95% t-interval across seeds for stochastic solvers, across instances otherwise.
