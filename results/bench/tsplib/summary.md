| suite | solver | n inst | seeds | gap vs ref | gap vs best in run | time / inst | ref | failed |
|---|---|---:|---:|---:|---:|---:|---|---:|
| tsplib:max_n=10000 | `callable:fn=tspgnn.api:solve,name=gnn+2opt` | 103 | 1 | 4.180% ± 0.635% | 4.180% ± 0.635% | 0.447s | optimum | 0 |
| tsplib:max_n=10000 | `callable:fn=tspgnn.api:solve_without_two_opt,name=gnn` | 103 | 1 | 20.965% ± 12.084% | 20.965% ± 12.084% | 0.362s | optimum | 0 |
| tsplib:max_n=10000 | `farthest_insertion` | 103 | 1 | 10.248% ± 1.528% | 10.248% ± 1.528% | 0.183s | optimum | 0 |
| tsplib:max_n=10000 | `two_opt:init=farthest_insertion` | 103 | 1 | 8.445% ± 1.069% | 8.445% ± 1.069% | 0.269s | optimum | 0 |
| tsplib:max_n=10000 | `lkh:time_limit=60` | 103 | 3 | 0.021% ± 0.018% | 0.021% ± 0.018% | 12.1s | optimum | 0 |

± is a 95% t-interval across seeds for stochastic solvers, across instances otherwise.
