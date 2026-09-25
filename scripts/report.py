"""Render evaluate.py JSON as Markdown tables (gap to LKH-3, % ± s.e.m.)."""
import json
import sys

COLS = ["nearest_neighbor+2opt", "greedy_distance", "gnn_greedy",
        "greedy_distance+2opt", "gnn+2opt"]
HEAD = ["NN+2opt", "greedy(dist)", "greedy(GNN)", "greedy(dist)+2opt", "greedy(GNN)+2opt"]


def main(path):
    res = json.load(open(path))["results"]
    print("| type | n | " + " | ".join(HEAD) + " | GNN+2opt time | LKH time |")
    print("|---|---:|" + "---:|" * (len(COLS) + 2))
    for r in res:
        best = min(COLS, key=lambda c: r[c]["gap_pct"])
        cells = []
        for c in COLS:
            s = f"{r[c]['gap_pct']:.2f} ± {r[c]['gap_sem_pct']:.2f}"
            cells.append(f"**{s}**" if c == best else s)
        kind = r["type"] + (" *(held out)*" if r["heldout_type"] else "")
        print(f"| {kind} | {r['n']} | " + " | ".join(cells)
              + f" | {r['gnn+2opt']['time_s']:.3f}s | {r['lkh_time_s']:.2f}s |")


if __name__ == "__main__":
    main(sys.argv[1])
