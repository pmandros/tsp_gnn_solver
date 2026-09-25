"""tspbench: benchmark suites, baselines and an evaluation harness for TSP solvers.

The harness is model-agnostic. A learned solver plugs in through
``callable:fn=pkg.mod:solve`` (``solve(distance_matrix) -> tour``) or
``heatmap:fn=pkg.mod:predict`` (``predict(distance_matrix) -> (n, n) scores``).
"""

from .instances import Instance, check_tour, is_valid_tour

__version__ = "0.1.0"
__all__ = ["Instance", "check_tour", "is_valid_tour"]
