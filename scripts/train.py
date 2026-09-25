"""Supervised training of the distance-only GNN on mixed sizes and types.

    python scripts/train.py --train 'data/train_*.pt' --val data/val.pt --out runs/main
"""
import argparse
import glob
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tspgnn.graph import EDGE_DIM, NODE_DIM, collate  # noqa: E402
from tspgnn.model import TSPGNN  # noqa: E402
from tspgnn.tours import greedy_edge_tour, tour_length  # noqa: E402


def edge_budget_batches(graphs, budget, shuffle):
    """Group graphs of any size into batches of roughly ``budget`` edges."""
    order = list(range(len(graphs)))
    if shuffle:
        random.shuffle(order)
    batch, edges = [], 0
    for i in order:
        batch.append(graphs[i])
        edges += graphs[i].edge_index.shape[1]
        if edges >= budget:
            yield batch
            batch, edges = [], 0
    if batch:
        yield batch


def loss_fn(logits, y):
    pos = y.sum().clamp(min=1)
    pos_weight = (y.numel() - pos) / pos
    return F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)


@torch.no_grad()
def validate(model, graphs, budget):
    """Validation loss and greedy-decode gap (no 2-opt) against LKH."""
    model.eval()
    losses, gaps = [], []
    for batch in edge_budget_batches(graphs, budget, shuffle=False):
        b = collate(batch)
        logits = model(b)
        losses.append(loss_fn(logits, b["y"]).item() * len(batch))
        off = 0
        for g in batch:
            m2 = g.edge_index.shape[1]
            score = logits[off:off + m2 // 2].numpy()
            src, dst = g.edge_index[:, :m2 // 2]
            tour = greedy_edge_tour(g.n, src, dst, np.argsort(-score, kind="stable"), g.d)
            ref = np.nonzero(g.y[: m2 // 2])[0]
            ref_len = g.d[src[ref], dst[ref]].sum()
            gaps.append(tour_length(g.d, tour) / ref_len - 1)
            off += m2
    model.train()
    return sum(losses) / len(graphs), float(np.mean(gaps))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="file or glob of shards")
    p.add_argument("--val", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--edge-budget", type=int, default=60000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=os.cpu_count())
    a = p.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)

    train = [g for f in sorted(glob.glob(a.train))
             for g in torch.load(f, weights_only=False)["graphs"]]
    val = torch.load(a.val, weights_only=False)["graphs"]
    # Valid tours only: the val gap needs every reference edge in the kNN graph.
    val = [g for g in val if g.y.sum() == 2 * g.n]

    model = TSPGNN(NODE_DIM, EDGE_DIM, a.hidden, a.layers)
    print(f"params {sum(p.numel() for p in model.parameters())}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
    steps_per_epoch = sum(1 for _ in edge_budget_batches(train, a.edge_budget, False))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.epochs * steps_per_epoch, pct_start=0.05
    )

    best, log = float("inf"), []
    t0 = time.time()
    for epoch in range(a.epochs):
        tot, cnt = 0.0, 0
        for step, batch in enumerate(edge_budget_batches(train, a.edge_budget, True)):
            b = collate(batch)
            loss = loss_fn(model(b), b["y"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(batch)
            cnt += len(batch)
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{steps_per_epoch} loss {loss.item():.4f} "
                      f"{time.time() - t0:.0f}s", flush=True)
        vloss, vgap = validate(model, val, a.edge_budget)
        rec = {"epoch": epoch, "train_loss": tot / cnt, "val_loss": vloss,
               "val_greedy_gap": vgap, "time_s": time.time() - t0}
        log.append(rec)
        print(json.dumps(rec), flush=True)
        if vgap < best:
            best = vgap
            torch.save({"state_dict": model.state_dict(), "config": vars(a), "epoch": epoch},
                       os.path.join(a.out, "best.pt"))
        json.dump(log, open(os.path.join(a.out, "log.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
