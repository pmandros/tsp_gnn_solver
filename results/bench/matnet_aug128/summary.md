| suite | solver | n inst | seeds | gap vs ref | gap vs best in run | time / inst | ref | failed |
|---|---|---:|---:|---:|---:|---:|---|---:|
| nonmetric20:num=128 | `matnet:aug=128` | 32 | 1 | 0.034% ± 0.053% | 0.034% ± 0.053% | 13.4s | lkh:runs=10 | 0 |
| nonmetric50:num=128 | `matnet:aug=128` | 32 | 1 | 20.561% ± 2.177% | 20.561% ± 2.177% | 48s | lkh:runs=10 | 0 |
| nonmetric100:num=128 | `matnet:aug=128` | 32 | 1 | 214.178% ± 9.248% | 214.178% ± 9.248% | 142s | concorde | 0 |
| atsp20:num=128 | `matnet:aug=128` | 32 | 1 | 0.000% ± 0.000% | 0.000% ± 0.000% | 13.3s | lkh:runs=10 | 0 |
| atsp50:num=128 | `matnet:aug=128` | 32 | 1 | 0.114% ± 0.078% | 0.114% ± 0.078% | 48.2s | lkh:runs=10 | 0 |
| atsp100:num=128 | `matnet:aug=128` | 32 | 1 | 1.132% ± 0.213% | 1.132% ± 0.213% | 142s | lkh:runs=10 | 0 |

± is a 95% t-interval across seeds for stochastic solvers, across instances otherwise.
