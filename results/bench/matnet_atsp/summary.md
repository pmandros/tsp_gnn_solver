| suite | solver | n inst | seeds | gap vs ref | gap vs best in run | time / inst | ref | failed |
|---|---|---:|---:|---:|---:|---:|---|---:|
| atsp20:num=128 | `callable:fn=tspgnn.api:solve_symmetrized,name=gnn(sym)+2opt` | 128 | 1 | 40.790% ± 2.656% | 40.790% ± 2.656% | 0.0215s | lkh:runs=10 | 0 |
| atsp20:num=128 | `callable:fn=tspgnn.api:solve_symmetrized,gnn=false,name=greedy(dist-sym)+2opt` | 128 | 1 | 41.585% ± 2.520% | 41.585% ± 2.520% | 0.000353s | lkh:runs=10 | 0 |
| atsp20:num=128 | `matnet` | 128 | 3 | 0.552% ± 0.089% | 0.552% ± 0.089% | 0.123s | lkh:runs=10 | 0 |
| atsp20:num=128 | `nearest_neighbor` | 128 | 3 | 32.750% ± 1.075% | 32.750% ± 1.075% | 9.81e-05s | lkh:runs=10 | 0 |
| atsp20:num=128 | `lkh` | 128 | 3 | 0.002% ± 0.009% | 0.002% ± 0.009% | 0.0041s | lkh:runs=10 | 0 |
| atsp50:num=128 | `callable:fn=tspgnn.api:solve_symmetrized,name=gnn(sym)+2opt` | 128 | 1 | 66.659% ± 2.128% | 66.659% ± 2.128% | 0.0332s | lkh:runs=10 | 0 |
| atsp50:num=128 | `callable:fn=tspgnn.api:solve_symmetrized,gnn=false,name=greedy(dist-sym)+2opt` | 128 | 1 | 69.249% ± 2.107% | 69.249% ± 2.107% | 0.000549s | lkh:runs=10 | 0 |
| atsp50:num=128 | `matnet` | 128 | 3 | 1.448% ± 0.120% | 1.448% ± 0.120% | 0.362s | lkh:runs=10 | 0 |
| atsp50:num=128 | `nearest_neighbor` | 128 | 3 | 34.041% ± 1.336% | 34.041% ± 1.336% | 0.000221s | lkh:runs=10 | 0 |
| atsp50:num=128 | `lkh` | 128 | 3 | 0.001% ± 0.002% | 0.001% ± 0.002% | 0.0262s | lkh:runs=10 | 0 |
| atsp100:num=128 | `callable:fn=tspgnn.api:solve_symmetrized,name=gnn(sym)+2opt` | 128 | 1 | 87.703% ± 1.626% | 87.703% ± 1.626% | 0.0535s | lkh:runs=10 | 0 |
| atsp100:num=128 | `callable:fn=tspgnn.api:solve_symmetrized,gnn=false,name=greedy(dist-sym)+2opt` | 128 | 1 | 91.014% ± 1.918% | 91.014% ± 1.918% | 0.0011s | lkh:runs=10 | 0 |
| atsp100:num=128 | `matnet` | 128 | 3 | 3.329% ± 0.134% | 3.329% ± 0.134% | 0.975s | lkh:runs=10 | 0 |
| atsp100:num=128 | `nearest_neighbor` | 128 | 3 | 36.383% ± 0.382% | 36.383% ± 0.382% | 0.000369s | lkh:runs=10 | 0 |
| atsp100:num=128 | `lkh` | 128 | 3 | 0.002% ± 0.003% | 0.002% ± 0.003% | 0.0864s | lkh:runs=10 | 0 |

± is a 95% t-interval across seeds for stochastic solvers, across instances otherwise.
