"""Re-run the notebook's experiment before and after the training/evaluation fixes.

Modes
  old : the notebook as it was. Directed (asymmetric) labels, `use_edge_weight=True`
        silently replaced by ones, the validation split trained on every epoch, and the
        gap measured against the optimum for integer-rounded distances (Concorde EUC_2D).
  new : symmetric labels (y = A + A^T), real distance weights in the aggregation
        (or ones with --no-edge-weight), a validation split that is never trained on
        and is used to pick the epoch, and the gap measured against the float optimum.

The model, optimizer, batching and greedy decoding are otherwise identical to the notebook.

Usage: python compare_old_new.py --mode new --seed 0 --out results/new_s0.json
"""
import argparse
import json
import random
import time
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn import BatchNorm1d, Linear as Lin, ReLU, Sequential as Seq
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter

from exact_tsp import tour_length

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------- data

def load_instances(path):
    z = np.load(path)
    out = []
    for i, n in enumerate(z["sizes"]):
        out.append(dict(points=z[f"points_{i}"], tour_float=z[f"tour_float_{i}"],
                        tour_round=z[f"tour_round_{i}"], len_float=float(z["len_float"][i]),
                        len_round=float(z["len_round"][i]), n=int(n)))
    return out


def tour_to_adjacency(tour, symmetric):
    n = len(tour)
    a = np.zeros((n, n))
    a[tour, np.roll(tour, -1)] = 1  # closes the tour back to tour[0], whatever it is
    if symmetric:
        a = np.maximum(a, a.T)
    return a


def to_data(inst, mode):
    n = inst["n"]
    p = inst["points"]
    dist = np.sqrt(((p[:, None] - p) ** 2).sum(2))
    if mode == "old":
        tour, ref = inst["tour_round"], inst["len_round"]
    else:
        tour, ref = inst["tour_float"], inst["len_float"]
    y = tour_to_adjacency(tour, symmetric=(mode == "new"))
    num_pos = y.sum()
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    edge_index = torch.tensor(np.stack([ii.ravel(), jj.ravel()]), dtype=torch.long)
    d = torch.tensor(dist.ravel()).float().unsqueeze(1)
    data = Data(x=torch.tensor(p).float(), edge_index=edge_index, edge_attr=d,
                y=torch.tensor(y.ravel()).float().unsqueeze(1))
    data.edge_weight = d.clone()
    data.true_path = torch.tensor(tour).float()
    data.true_distance = torch.tensor([[ref]]).float()
    data.true_distance_float = torch.tensor([[inst["len_float"]]]).float()
    data.num_nodes = n
    data.pos_class_weight = float((n * n - num_pos) / num_pos)
    return data


def same_size_batches(data_list, max_batch_nodes=2048):
    batches = []
    for d in sorted(data_list, key=lambda d: d.num_nodes):
        if batches and batches[-1][-1].num_nodes == d.num_nodes and \
                sum(x.num_nodes for x in batches[-1]) + d.num_nodes <= max_batch_nodes:
            batches[-1].append(d)
        else:
            batches.append([d])
    return batches


# ---------------------------------------------------------------------------- model

class EdgeModel(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        c = 3 * hidden_dim
        self.edge_mlp = Seq(BatchNorm1d(c), ReLU(), Lin(c, hidden_dim, bias=False))

    def forward(self, src, dst, edge_attr, edge_weight, u, batch):
        return self.edge_mlp(torch.cat([src, edge_attr, dst], 1)) + edge_attr


class NodeModel(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        c = 2 * hidden_dim
        self.node_mlp_1 = Seq(BatchNorm1d(c), ReLU(), Lin(c, hidden_dim))
        self.node_mlp_2 = Seq(BatchNorm1d(c), ReLU(), Lin(c, hidden_dim))

    def forward(self, x, edge_index, edge_attr, edge_weight, u, batch):
        row, col = edge_index
        out = self.node_mlp_1(torch.cat([x[row], edge_attr], dim=1))
        out = scatter(edge_weight * out, col, dim=0, dim_size=x.size(0), reduce="mean")
        return self.node_mlp_2(torch.cat([x, out], dim=1)) + x


class MetaLayer(torch.nn.Module):
    def __init__(self, edge_model, node_model):
        super().__init__()
        self.edge_model, self.node_model = edge_model, node_model

    def forward(self, x, edge_index, edge_attr, edge_weight, u, batch):
        row, col = edge_index
        edge_attr = self.edge_model(x[row], x[col], edge_attr, edge_weight, u, batch)
        x = self.node_model(x, edge_index, edge_attr, edge_weight, u, batch)
        return x, edge_attr, u


class METALAYER_TSP_SOLVER(torch.nn.Module):
    """The notebook's model with the node_inner_product decoder."""

    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, use_edge_weight, legacy_ones_bug):
        super().__init__()
        self.lin0_edge = Seq(Lin(edge_input_dim, hidden_dim, bias=False), BatchNorm1d(hidden_dim), ReLU())
        self.lin0_node = Seq(Lin(node_input_dim, hidden_dim, bias=False), BatchNorm1d(hidden_dim), ReLU())
        self.metalayer = MetaLayer(EdgeModel(hidden_dim), NodeModel(hidden_dim))
        self.metalayer2 = MetaLayer(EdgeModel(hidden_dim), NodeModel(hidden_dim))
        self.use_edge_weight = use_edge_weight
        self.legacy_ones_bug = legacy_ones_bug

    def forward(self, x, edge_index, edge_attr, edge_weight, u, batch):
        edge_attr = self.lin0_edge(edge_attr)
        x = self.lin0_node(x)
        if self.legacy_ones_bug or not self.use_edge_weight or edge_weight is None:
            # legacy: the notebook did this exactly when use_edge_weight was True
            edge_weight = torch.ones(edge_attr.shape[0], 1, device=edge_attr.device)
        x, edge_attr, u = self.metalayer(x, edge_index, edge_attr, edge_weight, None, batch)
        x, edge_attr, u = self.metalayer2(x, edge_index, edge_attr, edge_weight, None, batch)
        row, col = edge_index
        return (x[row] * x[col]).sum(dim=1).unsqueeze(1)


# ------------------------------------------------------------------------- training

def weighted_bce(out, data, criterion):
    loss = criterion(out, data.y)
    w = torch.ones_like(loss)
    w[data.y.bool()] = data.pos_class_weight[0]
    return (loss * w).sum() / w.sum()


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    losses = []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.edge_weight, None, data.batch)
        loss = weighted_bce(out, data, criterion)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    return float(np.mean(losses))


def greedy_tour(heat):
    n = heat.shape[0]
    visited = np.zeros(n, dtype=bool)
    tour, cur = [0], 0
    visited[0] = True
    for _ in range(n - 1):
        row = np.where(visited, -np.inf, heat[cur])
        cur = int(row.argmax())
        tour.append(cur)
        visited[cur] = True
    return np.array(tour)


@torch.no_grad()
def evaluate(model, loader, criterion):
    """Per-instance gaps of the greedy tour vs the mode's reference and vs the float optimum."""
    model.eval()
    losses, gaps, gaps_float = [], [], []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.edge_weight, None, data.batch)
        losses.append(weighted_bce(out, data, criterion).item())
        g = data.num_graphs
        n = data.num_nodes // g
        heat = out.view(g, n, n).cpu().numpy()
        dist = data.edge_weight.view(g, n, n).cpu().numpy()
        ref = data.true_distance.view(g).cpu().numpy()
        ref_f = data.true_distance_float.view(g).cpu().numpy()
        for k in range(g):
            L = tour_length(greedy_tour(heat[k]), dist[k])
            gaps.append((L - ref[k]) / ref[k])
            gaps_float.append((L - ref_f[k]) / ref_f[k])
    return dict(loss=float(np.mean(losses)), gap=float(np.mean(gaps)), gap_float_ref=float(np.mean(gaps_float)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["old", "new"], required=True)
    ap.add_argument("--no-edge-weight", action="store_true", help="new mode: aggregate with ones")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=17)  # the notebook ran range(0, 16 + 1)
    ap.add_argument("--train-data", default="/home/claude/data/train_10_30.npz")
    ap.add_argument("--big-data", default="/home/claude/data/test_100.npz")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_list = [to_data(i, args.mode) for i in load_instances(args.train_data)]
    shuffled = random.sample(data_list, len(data_list))
    n_tr, n_va = int(0.7 * len(shuffled)), int(0.1 * len(shuffled))
    train_list, val_list, test_list = shuffled[:n_tr], shuffled[n_tr:n_tr + n_va], shuffled[n_tr + n_va:]

    train_loader = DataLoader(same_size_batches(train_list), batch_size=None, shuffle=True)
    val_loader = DataLoader(same_size_batches(val_list), batch_size=None, shuffle=False)
    test_loader = DataLoader(same_size_batches(test_list), batch_size=None, shuffle=False)
    big_loader = DataLoader([to_data(i, args.mode) for i in load_instances(args.big_data)],
                            batch_size=1, shuffle=False)

    model = METALAYER_TSP_SOLVER(2, 1, 128, use_edge_weight=not args.no_edge_weight,
                                 legacy_ones_bug=(args.mode == "old")).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00005)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    history, best_val, best_state = [], float("inf"), None
    for epoch in range(args.epochs):
        t0 = time.time()
        tr_loss = train_epoch(model, train_loader, optimizer, criterion)
        if args.mode == "old":
            tr_loss = train_epoch(model, val_loader, optimizer, criterion)  # the leak
        val = evaluate(model, val_loader, criterion)
        test = evaluate(model, test_loader, criterion)
        history.append(dict(epoch=epoch, train_loss=tr_loss, val=val, test=test))
        if val["gap"] < best_val:
            best_val = val["gap"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        print(f"epoch {epoch:2d} {time.time() - t0:5.1f}s train_loss {tr_loss:.4f} "
              f"val_gap {val['gap']:.4f} test_gap {test['gap']:.4f} (float ref {test['gap_float_ref']:.4f})",
              flush=True)

    if args.mode == "old":
        # the notebook reported the last epoch; its "val" was trained on, so no selection
        final_test, chosen_epoch = history[-1]["test"], args.epochs - 1
    else:
        model.load_state_dict(best_state)
        final_test, chosen_epoch = evaluate(model, test_loader, criterion), best_epoch
    big = evaluate(model, big_loader, criterion)
    result = dict(mode=args.mode, edge_weight=(args.mode == "new" and not args.no_edge_weight),
                  seed=args.seed, chosen_epoch=chosen_epoch, test=final_test, tsp100=big, history=history)
    print(json.dumps({k: result[k] for k in ["mode", "edge_weight", "seed", "chosen_epoch", "test", "tsp100"]}))
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
