"""Plain-function entry points, e.g. for ``tspbench``.

    solve(distance_matrix) -> tour
    solve_sample(distance_matrix, samples=16) -> tour     (sampling + 2-opt)
    solve_search(distance_matrix, guide="gnn") -> tour    (guided k-opt search)
    predict(distance_matrix) -> (n, n) edge scores

All take the checkpoint path as a keyword argument and cache the loaded model.
"""
import os
from functools import lru_cache

import numpy as np
import torch

from .graph import EDGE_DIM, NODE_DIM, build_graph
from .model import TSPGNN
from .search import candidate_lists, guide_weights, guided_search, sample_decode
from .solve import gnn_heatmap, solve_gnn, solve_greedy_distance
from .tours import greedy_edge_tour, knn_lists, two_opt as _two_opt

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


def _guide(d, guide, checkpoint, k):
    """Candidate kNN edges, a score per edge for ranking and perturbing (GNN logit or
    -distance), and the per-edge sampling prior with its kind (see ``guide_weights``)."""
    if guide == "gnn":
        (src, dst), logit = gnn_heatmap(load_model(checkpoint), d, k)
        return src, dst, logit, 1.0 / (1.0 + np.exp(-logit)), "prob"
    if guide == "dist":
        g = build_graph(d, k=k)
        m = g.edge_index.shape[1] // 2
        src, dst = g.edge_index[:, :m]
        score = -d[src, dst]
        return src, dst, score, score, "rank"
    raise ValueError("guide must be gnn or dist")


def solve_search(distance_matrix, checkpoint=DEFAULT_CHECKPOINT, guide="gnn", time_limit=None,
                 time_per_node=0.002, m=5, depth=6, kick_len=30, k=20, seed=0):
    """Guided k-opt search (see ``tspgnn.search``) from the guide's greedy + 2-opt tour.

    ``guide=gnn`` samples moves from the GNN heatmap; ``guide=dist`` is the
    ablation that ranks the same candidate edges by distance. The search runs
    for ``time_limit`` seconds, by default ``time_per_node * n``.
    """
    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]
    if n <= 3:
        return np.arange(n)
    src, dst, score, prior, kind = _guide(d, guide, checkpoint, int(k))
    tour = greedy_edge_tour(n, src, dst, np.argsort(-score, kind="stable"), d)
    tour = _two_opt(d, tour, knn_lists(d, 20))
    cand, val = candidate_lists(n, src, dst, prior, int(m))
    w = guide_weights(cand, val, kind)
    limit = float(time_limit) if time_limit is not None else float(time_per_node) * n
    return guided_search(d, tour, cand, w, time_limit=limit, depth=int(depth),
                         kick_len=int(kick_len), seed=int(seed))


def solve_sample(distance_matrix, checkpoint=DEFAULT_CHECKPOINT, guide="gnn", samples=16, tau=1.0,
                 k=20, seed=0):
    """Best of ``samples`` Gumbel-perturbed greedy decodes, each with 2-opt."""
    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]
    if n <= 3:
        return np.arange(n)
    src, dst, score, _, _ = _guide(d, guide, checkpoint, int(k))
    if guide == "dist":
        score = score / np.abs(score).mean()  # unit-scale noise, like the GNN's logits
    return sample_decode(d, src, dst, score, samples=int(samples), tau=float(tau), seed=int(seed))


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
