"""Line-item metric ported from DocILE's LIR task (Study 34), plus a cell-only score that ignores the grouping.

A row is a collection of cells, a cell is any hashable key such as (field, start, end). Predicted and gold rows are
matched one-to-one so that the number of exactly matching cells is maximal (DocILE: maximum matching, maximising
recall); a predicted cell is right only if it is in the row matched to its gold row. `cell_f1` pools every row of a
document, so `cell_f1 - lir_f1` is the loss from grouping cells into rows alone.

    python scripts/lir_metric.py     # self-check
"""
from __future__ import annotations

import itertools

import numpy as np


def _assign(w: np.ndarray) -> list[tuple[int, int]]:
    """Maximum-weight one-to-one assignment (Hungarian algorithm, minimising -w); rectangular input is padded."""
    n = max(w.shape)
    c = np.zeros((n + 1, n + 1))
    c[1:w.shape[0] + 1, 1:w.shape[1] + 1] = -w
    u, v, p, way = np.zeros(n + 1), np.zeros(n + 1), np.zeros(n + 1, int), np.zeros(n + 1, int)
    for i in range(1, n + 1):
        p[0], j0, minv, used = i, 0, np.full(n + 1, np.inf), np.zeros(n + 1, bool)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], np.inf, 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = c[i0, j] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            p[j0] = p[way[j0]]
            j0 = way[j0]
    return [(p[j] - 1, j - 1) for j in range(1, n + 1) if p[j] - 1 < w.shape[0] and j - 1 < w.shape[1]]


def _hits(p: set, g: set, same) -> int:
    """Predicted cells with a partner in `g`: set equality, or one-to-one greedy pairing under `same(pred_cell, gold_cell)`."""
    if same is None:
        return len(p & g)
    free, n = list(g), 0
    for c in p:
        k = next((i for i, x in enumerate(free) if same(c, x)), None)
        if k is not None:
            n += 1
            free.pop(k)
    return n


def lir_counts(pred: list[set], gold: list[set], same=None) -> tuple[int, int, int]:
    """(tp, n_pred_cells, n_gold_cells) with predicted rows matched to gold rows by maximum overlap (`same`: value-level cell test)."""
    n_pred, n_gold = sum(map(len, pred)), sum(map(len, gold))
    if not pred or not gold:
        return 0, n_pred, n_gold
    w = np.array([[_hits(p, g, same) for g in gold] for p in pred], float)
    return int(sum(w[i, j] for i, j in _assign(w))), n_pred, n_gold


def cell_counts(pred: list[set], gold: list[set], same=None) -> tuple[int, int, int]:
    if same is None:
        p, g = set().union(*pred), set().union(*gold)
    else:   # value-level cells repeat across rows (same channel text in many rows), so pool them as lists
        p, g = [c for r in pred for c in r], [c for r in gold for c in r]
    return _hits(p, g, same), sum(map(len, pred)), sum(map(len, gold))


def f1(c: tuple[int, int, int]) -> float:
    tp, n_pred, n_gold = c
    p, r = (tp / n_pred if n_pred else 0.0), (tp / n_gold if n_gold else 0.0)
    return 2 * p * r / (p + r) if p + r else 0.0


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    gold = [{("a", i, i + 1), ("b", i + 10, i + 11)} for i in range(0, 4)]
    assert f1(lir_counts(gold, gold)) == 1.0 and f1(cell_counts(gold, gold)) == 1.0
    assert f1(lir_counts(gold[::-1], gold)) == 1.0                      # row order does not matter
    swapped = [{("a", 0, 1), ("b", 11, 12)}, {("a", 1, 2), ("b", 10, 11)}] + gold[2:]   # two rows trade their b cells
    assert f1(cell_counts(swapped, gold)) == 1.0 and abs(f1(lir_counts(swapped, gold)) - 0.75) < 1e-9
    assert f1(lir_counts(gold[:3], gold)) == f1(cell_counts(gold[:3], gold)) < 1.0   # a missing row costs both
    merged = [gold[0] | gold[1]] + gold[2:]                              # two rows fused into one
    assert f1(cell_counts(merged, gold)) == 1.0 and f1(lir_counts(merged, gold)) < 1.0
    for _ in range(200):                                                 # matches brute force on random small cases
        npred, ngold = rng.integers(1, 5, 2)
        cells = [(f, int(s), int(s) + 1) for f in "ab" for s in range(8)]
        pred = [set(rng.choice(len(cells), rng.integers(1, 5), replace=False).tolist()) for _ in range(npred)]
        gl = [set(rng.choice(len(cells), rng.integers(1, 5), replace=False).tolist()) for _ in range(ngold)]
        best = max(sum(len(pred[i] & gl[j]) for i, j in zip(*pair)) for k in range(1, min(npred, ngold) + 1)
                   for a in itertools.combinations(range(npred), k) for b in itertools.permutations(range(ngold), k) for pair in [(a, b)])
        assert lir_counts(pred, gl)[0] == best, (pred, gl)
    same = lambda a, b: a[0] == b[0] and a[1].lower() == b[1].lower()      # value-level test: same field, text equal up to case
    g2, p2 = [{("a", "X"), ("b", "Y")}, {("a", "Z")}], [{("a", "x"), ("b", "y")}, {("a", "z")}]
    dup = [{("a", "x")}, {("a", "x")}]                                 # the same value in two rows counts twice
    assert f1(cell_counts(dup, dup, same)) == 1.0 and cell_counts(dup[:1], dup, same)[0] == 1
    assert f1(lir_counts(p2, g2)) == 0.0 and f1(lir_counts(p2, g2, same)) == f1(cell_counts(p2, g2, same)) == 1.0
    print("lir_metric self-check passed")
