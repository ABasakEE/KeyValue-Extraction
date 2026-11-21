"""Row grouping shared by the BROS line-item baselines (ad-buy, DocILE): runs of consecutive words, strict IOB2 spans, and the anchor rule that groups spans into rows.

No dataset is read here, so it imports anywhere (`adbuy_bros.py` reads the VRDU metadata at import and cannot run on a machine without that data).
"""
from __future__ import annotations

from collections import defaultdict


def runs(ws) -> list[tuple[int, int]]:
    out, ws = [], sorted(ws)
    for i in ws:
        if out and i == out[-1][1]:
            out[-1] = (out[-1][0], i + 1)
        else:
            out.append((i, i + 1))
    return out


def extract_spans(names: list[str]) -> list[tuple[str, int, int]]:
    """(field, start, end) chunks of a strict IOB2 tag sequence, as src/evaluate._extract_spans (not imported: it needs seqeval)."""
    out, cur, start = [], None, 0
    for i, t in enumerate(names + ["O"]):
        pre, _, typ = t.partition("-")
        if cur is not None and (pre != "I" or typ != cur):
            out.append((cur, start, i))
            cur = None
        if pre == "B":
            cur, start = typ, i
    return out


def group_rows(spans: list[list], doc: dict, tau: float = 0.5, merge: bool = True) -> list[set]:
    """Rows of (field, word set) from predicted spans [field, start, end, text]: per page, anchor field = the one with most spans;
    every span goes to the closest anchor at or above it (within `tau` span heights below counts as 'at'), else the closest below."""
    fb, pg = doc["fboxes"], doc["pages"]
    by_page = defaultdict(list)
    for f, s, e, _ in spans:
        y0, y1 = min(fb[i][1] for i in range(s, e)), max(fb[i][3] for i in range(s, e))
        by_page[pg[s]].append((f, frozenset(range(s, e)), (y0 + y1) / 2, y1 - y0))
    rows = []
    for p, items in by_page.items():
        counts = defaultdict(int)
        for f, _, _, _ in items:
            counts[f] += 1
        anchor = max(counts, key=lambda f: (counts[f], f))
        anchors = sorted(((y, k) for k, (f, _, y, _) in enumerate(items) if f == anchor))
        cluster: dict[int, set] = {k: set() for _, k in anchors}
        for k, (f, ws, y, h) in enumerate(items):
            above = [(y - ya, ka) for ya, ka in anchors if ya <= y + tau * h]
            below = [(ya - y, ka) for ya, ka in anchors if ya > y + tau * h]
            _, target = min(above) if above else min(below)
            cluster[target].add((f, ws))
        for cells in cluster.values():
            if merge:   # a row holds at most one cell per field (checked on the gold: no row repeats a field), so same-field pieces in one row are one cell
                byf: dict[str, frozenset] = {}
                for f, ws in cells:
                    byf[f] = byf.get(f, frozenset()) | ws
                cells = set(byf.items())
            rows.append(cells)
    return rows
