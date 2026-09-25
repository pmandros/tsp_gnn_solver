"""Turn a distance matrix into a sparse, scale-free graph for the GNN.

The only input is the matrix ``D``. Nodes carry no coordinates. Every feature
is invariant to rescaling ``D`` by a positive constant, so instances on any
scale (unit square, kilometres, arbitrary costs) look the same to the model.
"""
from dataclasses import dataclass

import numpy as np
import torch

from .tours import knn_lists

NODE_DIM = 2
EDGE_DIM = 5


@dataclass
class TSPGraph:
    n: int
    x: np.ndarray           # (n, NODE_DIM) float32
    edge_index: np.ndarray  # (2, 2m) int32; edges [0, m) are i<j, [m, 2m) are their reverses
    edge_attr: np.ndarray   # (2m, EDGE_DIM) float32
    y: np.ndarray = None    # (2m,) float32, 1 if the undirected edge is in the reference tour


def build_graph(d, k=20, tour=None):
    n = d.shape[0]
    k = min(k, n - 1)
    nbrs = knn_lists(d, k)

    rank = np.full((n, n), k, dtype=np.int64)
    rank[np.repeat(np.arange(n), k), nbrs.ravel()] = np.tile(np.arange(k), n)

    # Undirected candidate set: j in kNN(i) or i in kNN(j).
    cand = rank < k
    cand = cand | cand.T
    src, dst = np.nonzero(np.triu(cand, 1))
    ei = np.concatenate([np.stack([src, dst]), np.stack([dst, src])], 1)
    i, j = ei

    knn_d = np.take_along_axis(d, nbrs, 1)       # (n, k)
    mean_i = knn_d.mean(1) + 1e-12                  # local scale per city
    scale = mean_i.mean()                           # global scale per instance
    dij = d[i, j]
    edge_attr = np.stack([
        dij / scale,
        np.minimum(dij / mean_i[i], 10.0),
        np.minimum(dij / mean_i[j], 10.0),
        rank[i, j] / k,
        rank[j, i] / k,
    ], 1).astype(np.float32)
    x = np.stack([mean_i / scale, knn_d[:, 0] / scale], 1).astype(np.float32)

    y = None
    if tour is not None:
        tour = np.asarray(tour)
        nxt = np.roll(tour, -1)
        adj = np.zeros((n, n), dtype=bool)
        adj[tour, nxt] = True
        adj[nxt, tour] = True
        y = adj[i, j].astype(np.float32)
    return TSPGraph(n, x, ei.astype(np.int32), edge_attr, y)


def label_coverage(g):
    """Fraction of reference-tour edges that survive kNN sparsification."""
    return float(g.y.sum() / (2 * g.n))


def collate(graphs):
    """Batch graphs of any sizes into one disjoint union."""
    xs, eis, eas, ys, rev, node_batch = [], [], [], [], [], []
    node_off = edge_off = 0
    for b, g in enumerate(graphs):
        m2 = g.edge_index.shape[1]
        m = m2 // 2
        xs.append(g.x)
        eis.append(g.edge_index.astype(np.int64) + node_off)
        eas.append(g.edge_attr)
        if g.y is not None:
            ys.append(g.y)
        r = np.arange(m2)
        rev.append((r + m) % m2 + edge_off)
        node_batch.append(np.full(g.n, b))
        node_off += g.n
        edge_off += m2
    out = {
        "x": torch.from_numpy(np.concatenate(xs)),
        "edge_index": torch.from_numpy(np.concatenate(eis, 1)).long(),
        "edge_attr": torch.from_numpy(np.concatenate(eas)),
        "rev": torch.from_numpy(np.concatenate(rev)),
        "batch": torch.from_numpy(np.concatenate(node_batch)),
    }
    if ys:
        out["y"] = torch.from_numpy(np.concatenate(ys))
    return out
