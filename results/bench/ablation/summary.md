| suite | solver | n inst | seeds | gap vs ref | gap vs best in run | time / inst | ref | failed |
|---|---|---:|---:|---:|---:|---:|---|---:|
| tsp20 | `callable:fn=tspgnn.api:solve_distance_greedy,name=greedy(dist)+2opt` | 1280 | 1 | 2.007% ± 0.148% | 2.007% ± 0.148% | 0.000809s | concorde | 0 |
| tsp50 | `callable:fn=tspgnn.api:solve_distance_greedy,name=greedy(dist)+2opt` | 1280 | 1 | 3.669% ± 0.145% | 3.669% ± 0.145% | 0.000977s | concorde | 0 |
| tsp100 | `callable:fn=tspgnn.api:solve_distance_greedy,name=greedy(dist)+2opt` | 1280 | 1 | 4.297% ± 0.113% | 4.297% ± 0.113% | 0.00143s | concorde | 0 |
| tsp500 | `callable:fn=tspgnn.api:solve_distance_greedy,name=greedy(dist)+2opt` | 128 | 1 | 4.605% ± 0.170% | 4.605% ± 0.170% | 0.0163s | lkh | 0 |
| tsp1000 | `callable:fn=tspgnn.api:solve_distance_greedy,name=greedy(dist)+2opt` | 128 | 1 | 4.541% ± 0.117% | 4.541% ± 0.117% | 0.0423s | lkh | 0 |
| manhattan100:num=128 | `callable:fn=tspgnn.api:solve_distance_greedy,name=greedy(dist)+2opt` | 128 | 1 | 4.943% ± 0.419% | 4.943% ± 0.419% | 0.00611s | concorde | 0 |
| chebyshev100:num=128 | `callable:fn=tspgnn.api:solve_distance_greedy,name=greedy(dist)+2opt` | 128 | 1 | 4.730% ± 0.348% | 4.730% ± 0.348% | 0.00637s | concorde | 0 |
| clustered100:num=128 | `callable:fn=tspgnn.api:solve_distance_greedy,name=greedy(dist)+2opt` | 128 | 1 | 5.297% ± 0.407% | 5.297% ± 0.407% | 0.00579s | concorde | 0 |
| nonmetric100:num=128 | `callable:fn=tspgnn.api:solve_distance_greedy,name=greedy(dist)+2opt` | 128 | 1 | 30.983% ± 1.424% | 30.983% ± 1.424% | 0.00556s | concorde | 0 |
| manhattan500:num=128 | `callable:fn=tspgnn.api:solve_distance_greedy,name=greedy(dist)+2opt` | 128 | 1 | 5.100% ± 0.215% | 5.100% ± 0.215% | 0.0178s | lkh | 0 |
| chebyshev500:num=128 | `callable:fn=tspgnn.api:solve_distance_greedy,name=greedy(dist)+2opt` | 128 | 1 | 4.996% ± 0.173% | 4.996% ± 0.173% | 0.0234s | lkh | 0 |
| clustered500:num=128 | `callable:fn=tspgnn.api:solve_distance_greedy,name=greedy(dist)+2opt` | 128 | 1 | 5.235% ± 0.228% | 5.235% ± 0.228% | 0.0154s | lkh | 0 |
| nonmetric500:num=128 | `callable:fn=tspgnn.api:solve_distance_greedy,name=greedy(dist)+2opt` | 128 | 1 | 53.816% ± 1.079% | 53.816% ± 1.079% | 0.0111s | lkh | 0 |
| manhattan1000:num=128 | `callable:fn=tspgnn.api:solve_distance_greedy,name=greedy(dist)+2opt` | 128 | 1 | 5.125% ± 0.146% | 5.125% ± 0.146% | 0.0547s | lkh | 0 |
| chebyshev1000:num=128 | `callable:fn=tspgnn.api:solve_distance_greedy,name=greedy(dist)+2opt` | 128 | 1 | 5.012% ± 0.140% | 5.012% ± 0.140% | 0.0785s | lkh | 0 |
| clustered1000:num=128 | `callable:fn=tspgnn.api:solve_distance_greedy,name=greedy(dist)+2opt` | 128 | 1 | 4.992% ± 0.173% | 4.992% ± 0.173% | 0.042s | lkh | 0 |
| nonmetric1000:num=128 | `callable:fn=tspgnn.api:solve_distance_greedy,name=greedy(dist)+2opt` | 128 | 1 | 64.166% ± 0.971% | 64.166% ± 0.971% | 0.0263s | lkh | 0 |
| tsplib:max_n=10000 | `callable:fn=tspgnn.api:solve_distance_greedy,name=greedy(dist)+2opt` | 103 | 1 | 5.057% ± 0.730% | 5.057% ± 0.730% | 0.229s | optimum | 0 |
| tsp10000 | `callable:fn=tspgnn.api:solve_distance_greedy,name=greedy(dist)+2opt` | 16 | 1 | 4.314% ± 0.117% | 4.314% ± 0.117% | 6.9s | lkh:max_trials=1000,time_limit=300 | 0 |

± is a 95% t-interval across seeds for stochastic solvers, across instances otherwise.
