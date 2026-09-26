"""Supervised training of the distance-only GNN on mixed sizes and types.

    python scripts/train.py --train 'data/train_*.pt' --val data/val.pt --out runs/main

Runs on the GPU when one is available (``--device``). Every epoch also writes
``last.pt`` with the optimizer and scheduler state, so ``--resume`` continues
an interrupted run (e.g. a disconnected Colab session) from the last finished epoch.
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


def to_device(b, device):
    return {k: v.to(device, non_blocking=True) for k, v in b.items()}


@torch.no_grad()
def validate(model, graphs, budget, device="cpu"):
    """Validation loss and greedy-decode gap (no 2-opt) against LKH."""
    model.eval()
    losses, gaps = [], []
    for batch in edge_budget_batches(graphs, budget, shuffle=False):
        b = to_device(collate(batch), device)
        logits = model(b)
        losses.append(loss_fn(logits, b["y"]).item() * len(batch))
        logits = logits.cpu()
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
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume", action="store_true", help="continue from OUT/last.pt if it exists")
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

    model = TSPGNN(NODE_DIM, EDGE_DIM, a.hidden, a.layers).to(a.device)
    print(f"params {sum(p.numel() for p in model.parameters())} device {a.device}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
    steps_per_epoch = sum(1 for _ in edge_budget_batches(train, a.edge_budget, False))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.epochs * steps_per_epoch, pct_start=0.05
    )

    best, log, start, t_prev = float("inf"), [], 0, 0.0
    last = os.path.join(a.out, "last.pt")
    if a.resume and os.path.exists(last):
        ck = torch.load(last, map_location=a.device, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        best, log, start = ck["best"], ck["log"], ck["epoch"] + 1
        t_prev = log[-1]["time_s"] if log else 0.0
        random.setstate(ck["rng"]["python"])
        np.random.set_state(ck["rng"]["numpy"])
        torch.set_rng_state(ck["rng"]["torch"])
        print(f"resumed from {last} at epoch {start}", flush=True)
    config = {k: v for k, v in vars(a).items() if k not in ("device", "resume")}
    t0 = time.time() - t_prev
    for epoch in range(start, a.epochs):
        tot, cnt = 0.0, 0
        for step, batch in enumerate(edge_budget_batches(train, a.edge_budget, True)):
            b = to_device(collate(batch), a.device)
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
        vloss, vgap = validate(model, val, a.edge_budget, a.device)
        rec = {"epoch": epoch, "train_loss": tot / cnt, "val_loss": vloss,
               "val_greedy_gap": vgap, "time_s": time.time() - t0}
        log.append(rec)
        print(json.dumps(rec), flush=True)
        if vgap < best:
            best = vgap
            state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save({"state_dict": state, "config": config, "epoch": epoch},
                       os.path.join(a.out, "best.pt"))
        json.dump(log, open(os.path.join(a.out, "log.json"), "w"), indent=1)
        # Written last and atomically, so a crash never leaves a half-saved resume point.
        torch.save({"state_dict": model.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "epoch": epoch, "best": best, "log": log,
                    "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                            "torch": torch.get_rng_state()}}, last + ".tmp")
        os.replace(last + ".tmp", last)


if __name__ == "__main__":
    main()
