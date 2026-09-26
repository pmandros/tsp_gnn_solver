"""Stronger decoders over an edge heatmap: sampling, and guided k-opt search.

Both work from a *guide*: a score for every candidate edge of the kNN graph.
The guide is either the GNN heatmap or plain distance, so the same search can
be run with and without the model and the difference is the model's own
contribution.

* ``sample_decode``: best of S greedy edge decodes, each on Gumbel-perturbed
  scores, each followed by 2-opt (the "sampling + 2-opt" decoding of DIMES
  and DIFUSCO).
* ``guided_search``: a Lin-Kernighan-style local search whose moves are
  sequential chains of 2-opt steps (depth up to ``depth``), with each chain's
  next edge sampled among the node's top-``m`` guide candidates in
  proportion to the guide's weight. Improving chains are committed.
  Weights of edges that entered an improving chain are reinforced, as in the
  MCTS of Fu et al. (2021) used by DIMES and DIFUSCO. Once no chain improves,
  the search perturbs the tour with a local segment swap (a double bridge on
  nearby cities), re-optimizes around it and keeps the result only if the
  tour got no longer. It runs until a time limit.

Symmetric matrices only.
"""
import time

import numpy as np
from numba import njit

from .tours import greedy_edge_tour, knn_lists, tour_length, two_opt

LOG_CAP = 1 << 16


# --------------------------------------------------------------------------- #
# Candidate lists
# --------------------------------------------------------------------------- #


def candidate_lists(n, src, dst, score, m):
    """Per-city top-``m`` neighbours by ``score`` (higher is better), padded with -1.

    Returns ``(cand, rank_score)`` where ``rank_score[i, r]`` is the score of
    ``cand[i, r]``.
    """
    s = np.concatenate([src, dst])
    t = np.concatenate([dst, src])
    v = np.concatenate([score, score]).astype(np.float64)
    order = np.lexsort((-v, s))
    s, t, v = s[order], t[order], v[order]
    starts = np.searchsorted(s, np.arange(n))
    counts = np.bincount(s, minlength=n)
    cand = -np.ones((n, m), dtype=np.int64)
    val = np.full((n, m), -np.inf)
    for r in range(m):
        ok = counts > r
        idx = starts[ok] + r
        cand[ok, r] = t[idx]
        val[ok, r] = v[idx]
    return cand, val


def guide_weights(cand, val, kind):
    """Sampling weights for the candidate lists.

    ``prob``: the guide values are probabilities (the GNN heatmap).
    ``rank``: weight 1/(rank+1), which uses only the guide's ordering.
    """
    if kind == "prob":
        w = np.clip(val, 1e-3, None)
    elif kind == "rank":
        w = np.broadcast_to(1.0 / (1.0 + np.arange(cand.shape[1])), cand.shape).copy()
    else:
        raise ValueError("kind must be prob or rank")
    w[cand < 0] = 0.0
    return np.ascontiguousarray(w, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #


def sample_decode(d, src, dst, score, samples=16, tau=1.0, seed=0, use_two_opt=True, time_limit=np.inf):
    """Best of ``samples`` Gumbel-perturbed greedy decodes (the first one unperturbed)."""
    n = d.shape[0]
    rng = np.random.default_rng(seed)
    nbrs = knn_lists(d, 20) if use_two_opt else None
    best, best_len = None, np.inf
    t0 = time.perf_counter()
    for s in range(samples):
        noisy = score if s == 0 else score + tau * rng.gumbel(size=score.shape)
        tour = greedy_edge_tour(n, src, dst, np.argsort(-noisy, kind="stable"), d)
        if use_two_opt:
            tour = two_opt(d, tour, nbrs)
        length = tour_length(d, tour)
        if length < best_len:
            best, best_len = tour, length
        if time.perf_counter() - t0 > time_limit:
            break
    return best


# --------------------------------------------------------------------------- #
# Guided k-opt search
# --------------------------------------------------------------------------- #


@njit(cache=True)
def _rev(tour, pos, i, j):
    """Reverse tour positions i..j (cyclic, inclusive), or the complementary
    range when that is shorter; both give the same undirected tour. Reversing
    the same (i, j) again undoes it."""
    n = tour.shape[0]
    inner = (j - i) % n + 1
    if 2 * inner > n:
        i, j = (j + 1) % n, (i - 1) % n
        inner = n - inner
    for s in range(inner // 2):
        u, v = (i + s) % n, (j - s) % n
        a, b = tour[u], tour[v]
        tour[u], tour[v] = b, a
        pos[b], pos[a] = u, v


@njit(cache=True)
def _undo(tour, pos, log, lo, hi):
    for k in range(hi - 1, lo - 1, -1):
        _rev(tour, pos, log[k, 0], log[k, 1])


@njit(cache=True)
def _pick(weights, k):
    tot = 0.0
    for i in range(k):
        tot += weights[i]
    r = np.random.random() * tot
    for i in range(k):
        r -= weights[i]
        if r <= 0.0:
            return i
    return k - 1


@njit(cache=True)
def _chain(d, tour, pos, t1, direction, cand, w, depth, log, logn, mark, stamp,
           touched, feas, fw):
    """One sampled LK chain from edge (t1, next(t1)). Keeps the best closed
    prefix if it shortens the tour; returns (gain, logn, n_touched)."""
    n = tour.shape[0]
    m = cand.shape[1]
    t2 = tour[(pos[t1] + direction) % n]
    G = d[t1, t2]
    mark[t1] = stamp
    mark[t2] = stamp
    touched[0], touched[1] = t1, t2
    nt = 2
    start = logn
    best_gain, best_logn, best_nt = 0.0, logn, nt
    for _ in range(depth):
        if logn >= log.shape[0]:
            break
        nxt2 = tour[(pos[t2] + direction) % n]
        k = 0
        for r in range(m):
            t3 = cand[t2, r]
            if t3 < 0 or mark[t3] == stamp or t3 == nxt2:
                continue
            g1 = G - d[t2, t3]
            if g1 <= 1e-12:
                continue
            t4 = tour[(pos[t3] - direction) % n]
            if mark[t4] == stamp:
                continue
            feas[k] = t3
            fw[k] = w[t2, r]
            k += 1
        if k == 0:
            break
        t3 = feas[_pick(fw, k)]
        t4 = tour[(pos[t3] - direction) % n]
        if direction == 1:
            i, j = pos[t2], pos[t4]
        else:
            i, j = pos[t4], pos[t2]
        _rev(tour, pos, i, j)
        log[logn, 0], log[logn, 1] = i, j
        logn += 1
        G = G - d[t2, t3] + d[t3, t4]
        mark[t3] = stamp
        mark[t4] = stamp
        touched[nt], touched[nt + 1] = t3, t4
        nt += 2
        direction = 1 if tour[(pos[t1] + 1) % n] == t4 else -1
        t2 = t4
        close = G - d[t2, t1]
        if close > best_gain + 1e-10:
            best_gain, best_logn, best_nt = close, logn, nt
    _undo(tour, pos, log, best_logn, logn)
    if best_gain <= 0.0:
        return 0.0, start, 0
    return best_gain, best_logn, best_nt


@njit(cache=True)
def _reinforce(cand, w, touched, nt, amount):
    # Edges added by a chain are (t2, t3) pairs: touched[1::2] -> touched[2::2].
    for k in range(1, nt - 1, 2):
        a, b = touched[k], touched[k + 1]
        for r in range(cand.shape[1]):
            if cand[a, r] == b:
                w[a, r] += amount
            if cand[b, r] == a:
                w[b, r] += amount


@njit(cache=True)
def _local_opt(d, tour, pos, cand, w, depth, trials, log, logn, mark, stamp,
               queue, inq, qn, touched, feas, fw, beta, length):
    """Chains from queued cities until none improves. Returns (gain, logn, stamp)."""
    total = 0.0
    while qn > 0:
        qn -= 1
        t1 = queue[qn]
        inq[t1] = False
        for _ in range(trials):
            improved = False
            for direction in (1, -1):
                stamp += 1
                gain, logn, nt = _chain(d, tour, pos, t1, direction, cand, w, depth,
                                        log, logn, mark, stamp, touched, feas, fw)
                if gain > 0.0:
                    total += gain
                    if beta > 0.0:
                        _reinforce(cand, w, touched, nt, beta * gain / length)
                    for k in range(nt):
                        c = touched[k]
                        if not inq[c]:
                            inq[c] = True
                            queue[qn] = c
                            qn += 1
                    improved = True
                    break
            if improved:
                break
        if logn >= log.shape[0] - depth:
            break
    return total, logn, stamp


@njit(cache=True)
def _search(d, tour, cand, w, depth, trials, kicks, kick_len, beta, seed, first):
    n = tour.shape[0]
    np.random.seed(seed)
    tour = tour.copy()
    pos = np.empty(n, dtype=np.int64)
    for i in range(n):
        pos[tour[i]] = i
    length = 0.0
    for i in range(n):
        length += d[tour[i], tour[(i + 1) % n]]
    mark = np.zeros(n, dtype=np.int64)
    stamp = 0
    queue = np.empty(n, dtype=np.int64)
    inq = np.zeros(n, dtype=np.bool_)
    touched = np.empty(2 * depth + 2, dtype=np.int64)
    feas = np.empty(cand.shape[1], dtype=np.int64)
    fw = np.empty(cand.shape[1], dtype=np.float64)
    log = np.empty((LOG_CAP, 2), dtype=np.int64)

    if first:
        # Full local optimization; the log is not needed, so reuse it freely.
        qn = 0
        for c in np.random.permutation(n):
            queue[qn] = c
            inq[c] = True
            qn += 1
        while qn > 0:
            gain, logn, stamp = _local_opt(d, tour, pos, cand, w, depth, trials, log, 0, mark,
                                           stamp, queue, inq, qn, touched, feas, fw, beta, length)
            length -= gain
            qn = 0
            for c in range(n):
                if inq[c]:
                    queue[qn] = c
                    qn += 1

    if n < 8:
        return tour, length
    kick_len = min(kick_len, (n - 2) // 2)
    for _ in range(kicks):
        # Segment swap A B C D -> A C B D with short B and C: a local double bridge.
        p = np.random.randint(n)
        l1 = 1 + np.random.randint(kick_len)
        l2 = 1 + np.random.randint(kick_len)
        a, b1 = tour[p], tour[(p + 1) % n]
        b2, c1 = tour[(p + l1) % n], tour[(p + l1 + 1) % n]
        c2, e = tour[(p + l1 + l2) % n], tour[(p + l1 + l2 + 1) % n]
        delta = d[a, c1] + d[c2, b1] + d[b2, e] - d[a, b1] - d[b2, c1] - d[c2, e]
        logn = 0
        for (i, j) in (((p + 1) % n, (p + l1) % n), ((p + l1 + 1) % n, (p + l1 + l2) % n),
                       ((p + 1) % n, (p + l1 + l2) % n)):
            _rev(tour, pos, i, j)
            log[logn, 0], log[logn, 1] = i, j
            logn += 1
        qn = 0
        for c in (a, b1, b2, c1, c2, e):
            if not inq[c]:
                inq[c] = True
                queue[qn] = c
                qn += 1
        gain, logn, stamp = _local_opt(d, tour, pos, cand, w, depth, trials, log, logn, mark,
                                       stamp, queue, inq, qn, touched, feas, fw, 0.0, length)
        for c in range(n):
            inq[c] = False
        if delta - gain <= 1e-10:
            length += delta - gain
        else:
            _undo(tour, pos, log, 0, logn)
    return tour, length


def guided_search(d, tour, cand, w, time_limit=1.0, depth=6, trials=1, kick_len=30,
                  beta=0.0, seed=0):
    """Improve ``tour`` for about ``time_limit`` seconds; see the module docstring.

    The first full local optimization always runs to completion; the
    perturbation phase then runs in chunks sized to end close to the limit.
    """
    d = np.ascontiguousarray(d, dtype=np.float64)
    n = d.shape[0]
    if n < 5:
        return np.asarray(tour)
    w = w.copy()
    t0 = time.perf_counter()
    tour, length = _search(d, np.asarray(tour, dtype=np.int64), cand, w, depth, trials,
                           0, kick_len, beta, seed, True)
    chunk, it = 50, 1
    while True:
        left = time_limit - (time.perf_counter() - t0)
        if left <= 0:
            break
        t = time.perf_counter()
        tour, length = _search(d, tour, cand, w, depth, trials, chunk, kick_len,
                               beta, seed + it, False)
        per_kick = (time.perf_counter() - t) / chunk
        # Aim for about 20 chunks per budget, never past the remaining time.
        target = min(left - (time.perf_counter() - t), time_limit / 20)
        chunk = int(max(10, min(4 * chunk, target / max(per_kick, 1e-9))))
        it += 1
    return tour
