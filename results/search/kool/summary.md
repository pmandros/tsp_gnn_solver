| suite | solver | n inst | seeds | gap vs ref | gap vs best in run | time / inst | ref | failed |
|---|---|---:|---:|---:|---:|---:|---|---:|
| tsp20 | `callable:fn=tspgnn.api:solve_search,guide=gnn,name=gnn+search,stochastic=true` | 1280 | 1 | 0.000% ± 0.000% | 0.000% ± 0.000% | 0.047s | concorde | 0 |
| tsp20 | `callable:fn=tspgnn.api:solve_search,guide=dist,name=dist+search,stochastic=true` | 1280 | 1 | 0.000% ± 0.000% | 0.000% ± 0.000% | 0.0407s | concorde | 0 |
| tsp20 | `callable:fn=tspgnn.api:solve_sample,guide=gnn,name=gnn+sample16,stochastic=true` | 1280 | 1 | 0.036% ± 0.011% | 0.036% ± 0.011% | 0.00599s | concorde | 0 |
| tsp20 | `callable:fn=tspgnn.api:solve_sample,guide=dist,name=dist+sample16,stochastic=true` | 1280 | 1 | 0.043% ± 0.011% | 0.043% ± 0.011% | 0.00101s | concorde | 0 |
| tsp50 | `callable:fn=tspgnn.api:solve_search,guide=gnn,name=gnn+search,stochastic=true` | 1280 | 1 | 0.002% ± 0.002% | 0.002% ± 0.002% | 0.116s | concorde | 0 |
| tsp50 | `callable:fn=tspgnn.api:solve_search,guide=dist,name=dist+search,stochastic=true` | 1280 | 1 | 0.013% ± 0.005% | 0.013% ± 0.005% | 0.101s | concorde | 0 |
| tsp50 | `callable:fn=tspgnn.api:solve_sample,guide=gnn,name=gnn+sample16,stochastic=true` | 1280 | 1 | 0.457% ± 0.033% | 0.457% ± 0.033% | 0.0145s | concorde | 0 |
| tsp50 | `callable:fn=tspgnn.api:solve_sample,guide=dist,name=dist+sample16,stochastic=true` | 1280 | 1 | 0.792% ± 0.042% | 0.792% ± 0.042% | 0.00181s | concorde | 0 |
| tsp100 | `callable:fn=tspgnn.api:solve_search,guide=gnn,name=gnn+search,stochastic=true` | 1280 | 1 | 0.068% ± 0.012% | 0.068% ± 0.012% | 0.231s | concorde | 0 |
| tsp100 | `callable:fn=tspgnn.api:solve_search,guide=dist,name=dist+search,stochastic=true` | 1280 | 1 | 0.156% ± 0.018% | 0.156% ± 0.018% | 0.202s | concorde | 0 |
| tsp100 | `callable:fn=tspgnn.api:solve_sample,guide=gnn,name=gnn+sample16,stochastic=true` | 1280 | 1 | 1.308% ± 0.040% | 1.308% ± 0.040% | 0.0281s | concorde | 0 |
| tsp100 | `callable:fn=tspgnn.api:solve_sample,guide=dist,name=dist+sample16,stochastic=true` | 1280 | 1 | 2.457% ± 0.056% | 2.457% ± 0.056% | 0.00377s | concorde | 0 |

± is a 95% t-interval across seeds for stochastic solvers, across instances otherwise.
