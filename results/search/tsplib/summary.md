| suite | solver | n inst | seeds | gap vs ref | gap vs best in run | time / inst | ref | failed |
|---|---|---:|---:|---:|---:|---:|---|---:|
| tsplib:max_n=10000 | `callable:fn=tspgnn.api:solve_search,guide=gnn,name=gnn+search,stochastic=true` | 103 | 1 | 0.342% ± 0.089% | 0.342% ± 0.089% | 2.17s | optimum | 0 |
| tsplib:max_n=10000 | `callable:fn=tspgnn.api:solve_search,guide=dist,name=dist+search,stochastic=true` | 103 | 1 | 1.125% ± 0.518% | 1.125% ± 0.518% | 1.84s | optimum | 0 |

± is a 95% t-interval across seeds for stochastic solvers, across instances otherwise.
