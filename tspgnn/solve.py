"""End-to-end solvers: the GNN and the classical baselines it is compared with."""
import time

import numpy as np
import torch

from .graph import build_graph, collate
from .tours import greedy_edge_tour, knn_lists, nearest_neighbor, two_opt

TWO_OPT_K = 20


def _finish(d, tour, use_two_opt):
    if use_two_opt:
        tour = two_opt(d, tour, knn_lists(d, TWO_OPT_K))
    return tour


@torch.no_grad()
def gnn_heatmap(model, d, k=20):
    """Edge scores for the undirected kNN candidate edges of one instance."""
    g = build_graph(d, k=k)
    logits = model(collate([g]))
    m = g.edge_index.shape[1] // 2
    return g.edge_index[:, :m], logits[:m].numpy()


def solve_gnn(model, d, k=20, use_two_opt=True):
    (src, dst), score = gnn_heatmap(model, d, k)
    tour = greedy_edge_tour(d.shape[0], src, dst, np.argsort(-score, kind="stable"), d)
    return _finish(d, tour, use_two_opt)


def solve_greedy_distance(d, k=20, use_two_opt=True):
    """Same greedy edge decoder, ranked by distance instead of GNN score."""
    g = build_graph(d, k=k)
    m = g.edge_index.shape[1] // 2
    src, dst = g.edge_index[:, :m]
    tour = greedy_edge_tour(d.shape[0], src, dst, np.argsort(d[src, dst], kind="stable"), d)
    return _finish(d, tour, use_two_opt)


def solve_nearest_neighbor(d, use_two_opt=True):
    return _finish(d, nearest_neighbor(d, 0), use_two_opt)


def timed(fn, *args, **kw):
    t = time.perf_counter()
    out = fn(*args, **kw)
    return out, time.perf_counter() - t
