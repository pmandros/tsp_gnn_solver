"""Plain-function entry points, e.g. for ``tspbench``.

    solve(distance_matrix) -> tour
    predict(distance_matrix) -> (n, n) edge scores

Both take the checkpoint path as a keyword argument and cache the loaded model.
"""
import os
from functools import lru_cache

import numpy as np
import torch

from .graph import EDGE_DIM, NODE_DIM
from .model import TSPGNN
from .solve import gnn_heatmap, solve_gnn, solve_greedy_distance

DEFAULT_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "tspgnn.pt"
)


@lru_cache(maxsize=4)
def load_model(checkpoint=DEFAULT_CHECKPOINT):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = TSPGNN(NODE_DIM, EDGE_DIM, cfg["hidden"], cfg["layers"])
    model.load_state_dict(ck["state_dict"])
    return model.eval()


def solve(distance_matrix, checkpoint=DEFAULT_CHECKPOINT, two_opt=True, k=20):
    d = np.asarray(distance_matrix, dtype=np.float64)
    if d.shape[0] <= 3:
        return np.arange(d.shape[0])
    if isinstance(two_opt, str):
        two_opt = two_opt.lower() == "true"
    return solve_gnn(load_model(checkpoint), d, int(k), use_two_opt=two_opt)


def solve_without_two_opt(distance_matrix, checkpoint=DEFAULT_CHECKPOINT, k=20):
    """GNN greedy decoding only, for ablations (``tspbench`` keeps ``two_opt`` for itself)."""
    return solve(distance_matrix, checkpoint=checkpoint, two_opt=False, k=k)


def solve_distance_greedy(distance_matrix, two_opt=True, k=20):
    """Ablation: the same decoder and 2-opt, with edges ranked by distance instead of the GNN."""
    d = np.asarray(distance_matrix, dtype=np.float64)
    if d.shape[0] <= 3:
        return np.arange(d.shape[0])
    return solve_greedy_distance(d, int(k), use_two_opt=two_opt)


def solve_symmetrized(distance_matrix, checkpoint=DEFAULT_CHECKPOINT, two_opt=True, k=20, gnn=True):
    """Naive ATSP fallback: solve (D + D^T) / 2, then keep the cheaper direction on D.

    The model and 2-opt are symmetric-only, so this ignores the asymmetry
    everywhere except the final choice of direction. It is a floor for what
    the current model can do on ATSP, not ATSP support. ``gnn=False`` ranks
    edges by the symmetrized distance instead (the distance ablation).
    """
    d = np.asarray(distance_matrix, dtype=np.float64)
    if isinstance(gnn, str):
        gnn = gnn.lower() == "true"
    sym = (d + d.T) / 2
    tour = solve(sym, checkpoint, two_opt, k) if gnn else solve_distance_greedy(sym, two_opt, k)
    tour = np.asarray(tour)
    rev = tour[::-1]
    fwd_len = d[tour, np.roll(tour, -1)].sum()
    rev_len = d[rev, np.roll(rev, -1)].sum()
    return tour if fwd_len <= rev_len else rev


def predict(distance_matrix, checkpoint=DEFAULT_CHECKPOINT, k=20):
    """Symmetric edge probabilities; pairs outside the kNN graph get 0."""
    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]
    (src, dst), score = gnn_heatmap(load_model(checkpoint), d, int(k))
    heat = np.zeros((n, n), dtype=np.float32)
    p = 1.0 / (1.0 + np.exp(-score))
    heat[src, dst] = p
    heat[dst, src] = p
    return heat
