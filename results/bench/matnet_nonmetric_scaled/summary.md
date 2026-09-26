| suite | solver | n inst | seeds | gap vs ref | gap vs best in run | time / inst | ref | failed |
|---|---|---:|---:|---:|---:|---:|---|---:|
| nonmetric20:num=128 | `matnet:scale=0.3` | 128 | 1 | 12.339% ± 1.385% | 12.339% ± 1.385% | 0.124s | lkh:runs=10 | 0 |
| nonmetric20:num=128 | `matnet:scale=0.1` | 128 | 1 | 15.638% ± 1.478% | 15.638% ± 1.478% | 0.12s | lkh:runs=10 | 0 |
| nonmetric20:num=128 | `matnet:scale=0.03` | 128 | 1 | 32.373% ± 2.970% | 32.373% ± 2.970% | 0.121s | lkh:runs=10 | 0 |
| nonmetric50:num=128 | `matnet:scale=0.3` | 128 | 1 | 56.734% ± 3.170% | 56.734% ± 3.170% | 0.365s | lkh:runs=10 | 0 |
| nonmetric50:num=128 | `matnet:scale=0.1` | 128 | 1 | 58.352% ± 2.916% | 58.352% ± 2.916% | 0.366s | lkh:runs=10 | 0 |
| nonmetric50:num=128 | `matnet:scale=0.03` | 128 | 1 | 59.177% ± 3.256% | 59.177% ± 3.256% | 0.362s | lkh:runs=10 | 0 |
| nonmetric100:num=128 | `matnet:scale=0.3` | 128 | 1 | 139.260% ± 3.819% | 139.260% ± 3.819% | 0.966s | concorde | 0 |
| nonmetric100:num=128 | `matnet:scale=0.1` | 128 | 1 | 132.427% ± 3.678% | 132.427% ± 3.678% | 0.968s | concorde | 0 |
| nonmetric100:num=128 | `matnet:scale=0.03` | 128 | 1 | 176.238% ± 4.361% | 176.238% ± 4.361% | 0.972s | concorde | 0 |

± is a 95% t-interval across seeds for stochastic solvers, across instances otherwise.
