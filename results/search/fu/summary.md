| suite | solver | n inst | seeds | gap vs ref | gap vs best in run | time / inst | ref | failed |
|---|---|---:|---:|---:|---:|---:|---|---:|
| tsp500 | `callable:fn=tspgnn.api:solve_search,guide=gnn,name=gnn+search,stochastic=true` | 128 | 3 | 0.454% ± 0.025% | 0.454% ± 0.025% | 1.15s | lkh | 0 |
| tsp500 | `callable:fn=tspgnn.api:solve_search,guide=dist,name=dist+search,stochastic=true` | 128 | 3 | 0.669% ± 0.015% | 0.669% ± 0.015% | 1.02s | lkh | 0 |
| tsp500 | `callable:fn=tspgnn.api:solve_sample,guide=gnn,name=gnn+sample16,stochastic=true` | 128 | 3 | 3.091% ± 0.156% | 3.091% ± 0.156% | 0.138s | lkh | 0 |
| tsp500 | `callable:fn=tspgnn.api:solve_sample,guide=dist,name=dist+sample16,stochastic=true` | 128 | 3 | 4.538% ± 0.052% | 4.538% ± 0.052% | 0.0331s | lkh | 0 |
| tsp1000 | `callable:fn=tspgnn.api:solve_search,guide=gnn,name=gnn+search,stochastic=true` | 128 | 3 | 0.529% ± 0.023% | 0.529% ± 0.023% | 2.23s | lkh | 0 |
| tsp1000 | `callable:fn=tspgnn.api:solve_search,guide=dist,name=dist+search,stochastic=true` | 128 | 3 | 0.824% ± 0.041% | 0.824% ± 0.041% | 2.05s | lkh | 0 |
| tsp1000 | `callable:fn=tspgnn.api:solve_sample,guide=gnn,name=gnn+sample16,stochastic=true` | 128 | 3 | 3.551% ± 0.005% | 3.552% ± 0.005% | 0.248s | lkh | 0 |
| tsp1000 | `callable:fn=tspgnn.api:solve_sample,guide=dist,name=dist+sample16,stochastic=true` | 128 | 3 | 4.540% ± 0.004% | 4.541% ± 0.004% | 0.103s | lkh | 0 |

± is a 95% t-interval across seeds for stochastic solvers, across instances otherwise.
