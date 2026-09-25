"""Edge-scoring GNN that reads only distances.

Anisotropic gated GCN (Bresson & Laurent 2017; Joshi et al. 2019) on the
sparse kNN graph, with choices aimed at size generalization:

* LayerNorm instead of BatchNorm, so no statistic depends on the batch or on
  the number of nodes;
* gate-normalized neighbor aggregation over a bounded kNN neighborhood, so the
  message scale does not grow or shrink with n;
* no global pooling or any other graph-level readout.
"""
import torch
from torch import nn


def scatter_sum(src, index, n):
    out = src.new_zeros((n,) + src.shape[1:])
    return out.index_add_(0, index, src)


class GatedGCNLayer(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.U = nn.Linear(h, h)
        self.V = nn.Linear(h, h)
        self.A = nn.Linear(h, h)
        self.B = nn.Linear(h, h)
        self.C = nn.Linear(h, h)
        self.norm_h = nn.LayerNorm(h)
        self.norm_e = nn.LayerNorm(h)

    def forward(self, h, e, edge_index):
        src, dst = edge_index  # message flows src -> dst
        e_new = self.A(h)[dst] + self.B(h)[src] + self.C(e)
        gate = torch.sigmoid(e_new)
        n = h.shape[0]
        num = scatter_sum(gate * self.V(h)[src], dst, n)
        den = scatter_sum(gate, dst, n) + 1e-6
        h_new = self.U(h) + num / den
        h = h + torch.relu(self.norm_h(h_new))
        e = e + torch.relu(self.norm_e(e_new))
        return h, e


class TSPGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden=64, layers=12):
        super().__init__()
        self.node_in = nn.Linear(node_dim, hidden)
        self.edge_in = nn.Linear(edge_dim, hidden)
        self.layers = nn.ModuleList(GatedGCNLayer(hidden) for _ in range(layers))
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, batch):
        h = self.node_in(batch["x"])
        e = self.edge_in(batch["edge_attr"])
        for layer in self.layers:
            h, e = layer(h, e, batch["edge_index"])
        logit = self.head(e).squeeze(-1)
        # Symmetric TSP: score i-j and j-i identically.
        return 0.5 * (logit + logit[batch["rev"]])
