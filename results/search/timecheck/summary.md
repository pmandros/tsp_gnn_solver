| suite | solver | n inst | seeds | gap vs ref | gap vs best in run | time / inst | ref | failed |
|---|---|---:|---:|---:|---:|---:|---|---:|
| tsp1000 | `callable:fn=tspgnn.api:solve_search,guide=dist,time_per_node=0.004,name=dist+search(2x),stochastic=true` | 128 | 1 | 0.743% ± 0.040% | 0.743% ± 0.040% | 4.07s | lkh | 0 |
| tsp1000 | `callable:fn=tspgnn.api:solve_search,guide=gnn,time_per_node=0.001,name=gnn+search(0.5x),stochastic=true` | 128 | 1 | 0.629% ± 0.035% | 0.629% ± 0.035% | 1.24s | lkh | 0 |
| nonmetric1000:num=128 | `callable:fn=tspgnn.api:solve_search,guide=dist,time_per_node=0.004,name=dist+search(2x),stochastic=true` | 128 | 1 | 16.641% ± 0.229% | 16.641% ± 0.229% | 4.05s | lkh | 0 |
| nonmetric1000:num=128 | `callable:fn=tspgnn.api:solve_search,guide=gnn,time_per_node=0.001,name=gnn+search(0.5x),stochastic=true` | 128 | 1 | 4.534% ± 0.124% | 4.534% ± 0.124% | 1.21s | lkh | 0 |

± is a 95% t-interval across seeds for stochastic solvers, across instances otherwise.
