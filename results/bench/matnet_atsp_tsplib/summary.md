| suite | solver | n inst | seeds | gap vs ref | gap vs best in run | time / inst | ref | failed |
|---|---|---:|---:|---:|---:|---:|---|---:|
| atsp_tsplib:max_n=256 | `callable:fn=tspgnn.api:solve_symmetrized,name=gnn(sym)+2opt` | 15 | 1 | 23.462% ± 11.526% | 23.462% ± 11.526% | 0.131s | optimum | 0 |
| atsp_tsplib:max_n=256 | `matnet:scale=max` | 15 | 3 | 58.567% ± 19.676% | 58.567% ± 19.676% | 0.557s | optimum | 0 |
| atsp_tsplib:max_n=256 | `nearest_neighbor` | 15 | 3 | 31.914% ± 7.771% | 31.914% ± 7.771% | 0.000315s | optimum | 0 |
| atsp_tsplib:max_n=256 | `lkh` | 15 | 3 | 0.004% ± 0.013% | 0.004% ± 0.013% | 0.0442s | optimum | 0 |

± is a 95% t-interval across seeds for stochastic solvers, across instances otherwise.
