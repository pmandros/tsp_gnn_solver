| suite | solver | n inst | seeds | gap vs ref | gap vs best in run | time / inst | ref | failed |
|---|---|---:|---:|---:|---:|---:|---|---:|
| nonmetric20:num=128 | `callable:fn=tspgnn.api:solve,name=gnn+2opt` | 128 | 1 | 4.576% ± 0.933% | 4.576% ± 0.933% | 0.0201s | lkh:runs=10 | 0 |
| nonmetric20:num=128 | `callable:fn=tspgnn.api:solve_distance_greedy,name=greedy(dist)+2opt` | 128 | 1 | 10.665% ± 1.334% | 10.665% ± 1.334% | 0.000336s | lkh:runs=10 | 0 |
| nonmetric20:num=128 | `matnet` | 128 | 3 | 10.355% ± 2.169% | 10.355% ± 2.169% | 0.122s | lkh:runs=10 | 0 |
| nonmetric20:num=128 | `nearest_neighbor` | 128 | 3 | 59.547% ± 3.757% | 59.547% ± 3.757% | 9.51e-05s | lkh:runs=10 | 0 |
| nonmetric20:num=128 | `lkh` | 128 | 3 | 0.000% ± 0.000% | 0.000% ± 0.000% | 0.00153s | lkh:runs=10 | 0 |
| nonmetric50:num=128 | `callable:fn=tspgnn.api:solve,name=gnn+2opt` | 128 | 1 | 7.445% ± 0.874% | 7.445% ± 0.874% | 0.0269s | lkh:runs=10 | 0 |
| nonmetric50:num=128 | `callable:fn=tspgnn.api:solve_distance_greedy,name=greedy(dist)+2opt` | 128 | 1 | 21.925% ± 1.415% | 21.925% ± 1.415% | 0.000438s | lkh:runs=10 | 0 |
| nonmetric50:num=128 | `matnet` | 128 | 3 | 55.508% ± 3.313% | 55.508% ± 3.313% | 0.397s | lkh:runs=10 | 0 |
| nonmetric50:num=128 | `nearest_neighbor` | 128 | 3 | 94.825% ± 4.373% | 94.825% ± 4.373% | 0.000203s | lkh:runs=10 | 0 |
| nonmetric50:num=128 | `lkh` | 128 | 3 | 0.000% ± 0.001% | 0.000% ± 0.001% | 0.0101s | lkh:runs=10 | 0 |
| nonmetric100:num=128 | `callable:fn=tspgnn.api:solve,name=gnn+2opt` | 128 | 1 | 10.661% ± 0.866% | 10.661% ± 0.866% | 0.0415s | concorde | 0 |
| nonmetric100:num=128 | `callable:fn=tspgnn.api:solve_distance_greedy,name=greedy(dist)+2opt` | 128 | 1 | 30.983% ± 1.424% | 30.983% ± 1.424% | 0.000785s | concorde | 0 |
| nonmetric100:num=128 | `matnet` | 128 | 3 | 314.427% ± 13.989% | 314.427% ± 13.989% | 0.975s | concorde | 0 |
| nonmetric100:num=128 | `nearest_neighbor` | 128 | 3 | 135.756% ± 2.892% | 135.756% ± 2.892% | 0.000433s | concorde | 0 |
| nonmetric100:num=128 | `lkh` | 128 | 3 | 0.000% ± 0.000% | 0.000% ± 0.000% | 0.0298s | concorde | 0 |

± is a 95% t-interval across seeds for stochastic solvers, across instances otherwise.
