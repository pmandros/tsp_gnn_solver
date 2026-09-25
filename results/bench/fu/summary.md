| suite | solver | n inst | seeds | gap vs ref | gap vs best in run | time / inst | ref | failed |
|---|---|---:|---:|---:|---:|---:|---|---:|
| tsp500 | `callable:fn=tspgnn.api:solve,name=gnn+2opt` | 128 | 1 | 4.048% ± 0.139% | 4.050% ± 0.139% | 0.109s | lkh | 0 |
| tsp500 | `callable:fn=tspgnn.api:solve_without_two_opt,name=gnn` | 128 | 1 | 16.244% ± 0.472% | 16.245% ± 0.472% | 0.0949s | lkh | 0 |
| tsp500 | `nearest_neighbor` | 128 | 3 | 25.746% ± 0.566% | 25.748% ± 0.566% | 0.00674s | lkh | 0 |
| tsp500 | `nearest_insertion` | 128 | 1 | 24.638% ± 0.268% | 24.639% ± 0.268% | 0.045s | lkh | 0 |
| tsp500 | `farthest_insertion` | 128 | 1 | 10.585% ± 0.222% | 10.587% ± 0.222% | 0.0439s | lkh | 0 |
| tsp500 | `random_insertion` | 128 | 3 | 12.348% ± 0.113% | 12.349% ± 0.113% | 0.0242s | lkh | 0 |
| tsp500 | `two_opt:init=farthest_insertion` | 128 | 1 | 9.270% ± 0.202% | 9.271% ± 0.202% | 0.0885s | lkh | 0 |
| tsp500 | `lkh` | 128 | 3 | 0.000% ± 0.000% | 0.001% ± 0.000% | 1.77s | lkh | 0 |
| tsp500 | `ortools:time_limit=10` | 128 | 1 | 4.925% ± 0.165% | 4.927% ± 0.165% | 10s | lkh | 0 |
| tsp1000 | `callable:fn=tspgnn.api:solve,name=gnn+2opt` | 128 | 1 | 4.079% ± 0.104% | 4.082% ± 0.105% | 0.231s | lkh | 0 |
| tsp1000 | `callable:fn=tspgnn.api:solve_without_two_opt,name=gnn` | 128 | 1 | 15.650% ± 0.304% | 15.653% ± 0.304% | 0.209s | lkh | 0 |
| tsp1000 | `nearest_neighbor` | 128 | 3 | 25.224% ± 0.839% | 25.228% ± 0.839% | 0.0204s | lkh | 0 |
| tsp1000 | `nearest_insertion` | 128 | 1 | 25.264% ± 0.200% | 25.267% ± 0.200% | 0.125s | lkh | 0 |
| tsp1000 | `farthest_insertion` | 128 | 1 | 11.275% ± 0.164% | 11.278% ± 0.164% | 0.126s | lkh | 0 |
| tsp1000 | `random_insertion` | 128 | 3 | 12.885% ± 0.137% | 12.889% ± 0.137% | 0.0639s | lkh | 0 |
| tsp1000 | `two_opt:init=farthest_insertion` | 128 | 1 | 9.882% ± 0.157% | 9.885% ± 0.157% | 0.472s | lkh | 0 |
| tsp1000 | `lkh` | 128 | 3 | 0.000% ± 0.000% | 0.003% ± 0.000% | 7.59s | lkh | 0 |
| tsp1000 | `ortools:time_limit=10` | 128 | 1 | 9.736% ± 0.221% | 9.740% ± 0.222% | 10.1s | lkh | 0 |

± is a 95% t-interval across seeds for stochastic solvers, across instances otherwise.
