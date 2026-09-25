| suite | solver | n inst | seeds | gap vs ref | gap vs best in run | time / inst | ref | failed |
|---|---|---:|---:|---:|---:|---:|---|---:|
| tsp20 | `callable:fn=tspgnn.api:solve,name=gnn+2opt` | 1280 | 1 | 1.166% ± 0.107% | 1.166% ± 0.107% | 0.0057s | concorde | 0 |
| tsp20 | `callable:fn=tspgnn.api:solve_without_two_opt,name=gnn` | 1280 | 1 | 6.243% ± 0.333% | 6.243% ± 0.333% | 0.00437s | concorde | 0 |
| tsp20 | `nearest_neighbor` | 1280 | 3 | 17.600% ± 0.265% | 17.600% ± 0.265% | 0.000183s | concorde | 0 |
| tsp20 | `nearest_insertion` | 1280 | 1 | 12.970% ± 0.381% | 12.970% ± 0.381% | 0.00119s | concorde | 0 |
| tsp20 | `farthest_insertion` | 1280 | 1 | 2.398% ± 0.155% | 2.398% ± 0.155% | 0.00129s | concorde | 0 |
| tsp20 | `random_insertion` | 1280 | 3 | 4.426% ± 0.306% | 4.426% ± 0.306% | 0.000779s | concorde | 0 |
| tsp20 | `two_opt:init=farthest_insertion` | 1280 | 1 | 1.510% ± 0.116% | 1.510% ± 0.116% | 0.00146s | concorde | 0 |
| tsp20 | `lkh` | 1280 | 3 | 0.000% ± 0.000% | 0.000% ± 0.000% | 0.00127s | concorde | 0 |
| tsp20 | `ortools:time_limit=1` | 1280 | 1 | 0.002% ± 0.002% | 0.002% ± 0.002% | 1s | concorde | 0 |
| tsp50 | `callable:fn=tspgnn.api:solve,name=gnn+2opt` | 1280 | 1 | 2.728% ± 0.115% | 2.728% ± 0.115% | 0.0125s | concorde | 0 |
| tsp50 | `callable:fn=tspgnn.api:solve_without_two_opt,name=gnn` | 1280 | 1 | 11.319% ± 0.308% | 11.319% ± 0.308% | 0.0121s | concorde | 0 |
| tsp50 | `nearest_neighbor` | 1280 | 3 | 23.082% ± 0.425% | 23.082% ± 0.425% | 0.000423s | concorde | 0 |
| tsp50 | `nearest_insertion` | 1280 | 1 | 19.298% ± 0.251% | 19.298% ± 0.251% | 0.0033s | concorde | 0 |
| tsp50 | `farthest_insertion` | 1280 | 1 | 5.683% ± 0.177% | 5.683% ± 0.177% | 0.00313s | concorde | 0 |
| tsp50 | `random_insertion` | 1280 | 3 | 7.796% ± 0.095% | 7.796% ± 0.095% | 0.00175s | concorde | 0 |
| tsp50 | `two_opt:init=farthest_insertion` | 1280 | 1 | 4.585% ± 0.156% | 4.585% ± 0.156% | 0.00343s | concorde | 0 |
| tsp50 | `lkh` | 1280 | 3 | 0.003% ± 0.002% | 0.003% ± 0.002% | 0.0105s | concorde | 0 |
| tsp50 | `ortools:time_limit=1` | 1280 | 1 | 1.571% ± 0.092% | 1.571% ± 0.092% | 1s | concorde | 0 |
| tsp100 | `callable:fn=tspgnn.api:solve,name=gnn+2opt` | 1280 | 1 | 3.504% ± 0.099% | 3.504% ± 0.099% | 0.0238s | concorde | 0 |
| tsp100 | `callable:fn=tspgnn.api:solve_without_two_opt,name=gnn` | 1280 | 1 | 13.906% ± 0.250% | 13.906% ± 0.250% | 0.0222s | concorde | 0 |
| tsp100 | `nearest_neighbor` | 1280 | 3 | 24.899% ± 0.356% | 24.899% ± 0.356% | 0.000776s | concorde | 0 |
| tsp100 | `nearest_insertion` | 1280 | 1 | 21.896% ± 0.193% | 21.896% ± 0.193% | 0.00656s | concorde | 0 |
| tsp100 | `farthest_insertion` | 1280 | 1 | 7.571% ± 0.141% | 7.571% ± 0.141% | 0.00634s | concorde | 0 |
| tsp100 | `random_insertion` | 1280 | 3 | 9.631% ± 0.070% | 9.631% ± 0.070% | 0.00359s | concorde | 0 |
| tsp100 | `two_opt:init=farthest_insertion` | 1280 | 1 | 6.389% ± 0.129% | 6.389% ± 0.129% | 0.00714s | concorde | 0 |
| tsp100 | `lkh` | 1280 | 3 | 0.004% ± 0.001% | 0.004% ± 0.001% | 0.0512s | concorde | 0 |
| tsp100 | `ortools:time_limit=1` | 1280 | 1 | 3.158% ± 0.098% | 3.158% ± 0.098% | 1s | concorde | 0 |

± is a 95% t-interval across seeds for stochastic solvers, across instances otherwise.
