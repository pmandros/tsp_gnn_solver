| suite | solver | n inst | seeds | gap vs ref | gap vs best in run | time / inst | ref | failed |
|---|---|---:|---:|---:|---:|---:|---|---:|
| tsp10000 | `callable:fn=tspgnn.api:solve_search,guide=gnn,name=gnn+search,stochastic=true` | 16 | 1 | 0.793% ± 0.030% | 0.793% ± 0.030% | 46.6s | lkh:max_trials=1000,time_limit=300 | 0 |
| tsp10000 | `callable:fn=tspgnn.api:solve_search,guide=dist,name=dist+search,stochastic=true` | 16 | 1 | 1.137% ± 0.045% | 1.137% ± 0.045% | 40.8s | lkh:max_trials=1000,time_limit=300 | 0 |

± is a 95% t-interval across seeds for stochastic solvers, across instances otherwise.
