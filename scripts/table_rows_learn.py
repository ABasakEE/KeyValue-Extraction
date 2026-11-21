"""Study 35, step 1 (CPU): group perfect spans into rows with a learned pair classifier, against the anchor rule.

    <python with scikit-learn> scripts/table_rows_learn.py

Input to both groupers: the spans a perfect tagger would output (runs of consecutive words of every gold cell, as in studies 34's oracles). Output: rows, scored with strict row F1
(`lir_metric`). Candidate pairs: spans of one page whose vertical gap is at most 4 median span heights; classifier = gradient boosting on geometry, field types and (DocILE only) the
gold header above; rows = single-link clusters over pairs above a threshold chosen on the split's validation documents (a row never holds two cells of a field that never repeats in a
row: ad-buy). Trained only on training documents; evaluated on the DocILE test documents (200) and the ad-buy primary documents (143); bootstrap over documents.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "src"), str(ROOT)]
import adbuy_bros as B  # noqa: E402
import docile_bros as DB  # noqa: E402
import docile_rows as D  # noqa: E402
from lir_metric import f1, lir_counts  # noqa: E402

THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
N_BOOT = 1000


# ------------------------------------------------------------------ documents
def prep(dataset: str, ids: list[str], ad_docs: dict | None = None) -> list[dict]:
    """Per document: words' boxes, pages, spans [field, start, end], gold rows, true row of every span, headers (DocILE)."""
    out = []
    for i in ids:
        if dataset == "docile":
            d = DB.doc_dict(i)
            ann = DB.G.ann_of(i)
            d["headers"] = [(h["page"], h["bbox"], h["fieldtype"]) for h in ann["line_item_headers"]]
            spans = DB.spans_of(d)
        else:
            d = ad_docs[i]
            d["headers"] = []
            spans = sorted([t, s, e, ""] for t, s, e in B.extract_spans([B.ID2LABEL[k] for k in B.tags_of(d)]))
        if not d["rows"] or not spans:
            continue
        true_row = {(f, w): k for k, r in enumerate(d["rows"]) for f, ws, _ in r for w in ws}
        out.append({"id": i, "d": d, "spans": spans, "row_of": [true_row.get((f, s), -1) for f, s, e, _ in spans]})
    return out


def box(d: dict, s: int, e: int) -> tuple:
    b = d["fboxes"][s:e]
    return (min(x[0] for x in b), min(x[1] for x in b), max(x[2] for x in b), max(x[3] for x in b))


def xo(a: tuple, b: tuple) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) / max(1e-9, min(a[2] - a[0], b[2] - b[0]))


# ------------------------------------------------------------------ pairs
def pairs(doc: dict, fields: list[str], use_header: bool):
    """Candidate pairs (i, j), features, labels (same true row)."""
    d, spans = doc["d"], doc["spans"]
    boxes = [box(d, s, e) for _, s, e, _ in spans]
    pages = [d["pages"][s] for _, s, _, _ in spans]
    hdr = [-1] * len(spans)
    for k, bx in enumerate(boxes):
        cy = (bx[1] + bx[3]) / 2
        cands = [(cy - h[1][3], hi) for hi, h in enumerate(d["headers"]) if h[0] == pages[k] and h[1][3] <= cy and xo(bx, tuple(h[1])) >= 0.3]
        hdr[k] = min(cands)[1] if cands else -1
    K = len(fields)
    fi = {f: n for n, f in enumerate(fields)}
    by_page = defaultdict(list)
    for k, p in enumerate(pages):
        by_page[p].append(k)
    I, J, X, Y = [], [], [], []
    for p, ks in by_page.items():
        H = float(np.median([boxes[k][3] - boxes[k][1] for k in ks])) or 1e-3
        for a, b in combinations(ks, 2):
            if boxes[a][1] + boxes[a][3] > boxes[b][1] + boxes[b][3]:
                a, b = b, a                     # a is the upper one
            A, Bx = boxes[a], boxes[b]
            gap = max(0.0, max(A[1], Bx[1]) - min(A[3], Bx[3]))
            if gap > 4 * H:
                continue
            hA, hB = A[3] - A[1], Bx[3] - Bx[1]
            fa, fb = sorted((fi[spans[a][0]], fi[spans[b][0]]))
            f = [((Bx[1] + Bx[3]) - (A[1] + A[3])) / 2 / H, gap / H, max(0.0, min(A[3], Bx[3]) - max(A[1], Bx[1])) / max(1e-9, min(hA, hB)),
                 ((Bx[0] + Bx[2]) - (A[0] + A[2])) / 2 / 0.1, xo(A, Bx), float(spans[a][0] == spans[b][0]), hA / H, hB / H, fa * K + fb]
            if use_header:
                f += [float(hdr[a] >= 0 and hdr[a] == hdr[b]), float(hdr[a] >= 0), float(hdr[b] >= 0)]
            I.append(a); J.append(b); X.append(f); Y.append(int(doc["row_of"][a] >= 0 and doc["row_of"][a] == doc["row_of"][b]))
    return I, J, np.array(X, float).reshape(len(X), 12 if use_header else 9), np.array(Y, int)


def cluster(n: int, I: list, J: list, prob: np.ndarray, tau: float, fields_of: list[str], unique: bool) -> list[list[int]]:
    parent = list(range(n))
    members = {k: {fields_of[k]} for k in range(n)}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for t in np.argsort(-prob):
        if prob[t] < tau:
            break
        ra, rb = find(I[t]), find(J[t])
        if ra == rb or (unique and members[ra] & members[rb]):
            continue
        parent[rb] = ra
        members[ra] |= members.pop(rb)
    groups = defaultdict(list)
    for k in range(n):
        groups[find(k)].append(k)
    return list(groups.values())


def doc_counts(doc: dict, rows: list[list[int]]):
    spans = doc["spans"]
    pred = [{(spans[k][0], frozenset(range(spans[k][1], spans[k][2]))) for k in g} for g in rows]
    gold = [{(f, ws) for f, ws, _ in r} for r in doc["d"]["rows"]]
    return lir_counts(pred, gold)


def rule_rows(doc: dict) -> list[list[int]]:
    spans = doc["spans"]
    idx = {(f, frozenset(range(s, e))): k for k, (f, s, e, _) in enumerate(spans)}
    return [[idx[c] for c in r] for r in B.group_rows(spans, doc["d"], 0.5, merge=False)]


# ------------------------------------------------------------------ experiment
def fit(train: list[dict], fields: list[str], use_header: bool):
    Xs, Ys = [], []
    for doc in train:
        _, _, X, Y = pairs(doc, fields, use_header)
        if len(Y):
            Xs.append(X); Ys.append(Y)
    X, Y = np.vstack(Xs), np.concatenate(Ys)
    cat = np.zeros(X.shape[1], bool)
    cat[8] = True
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, categorical_features=cat, random_state=0).fit(X, Y)
    return m, len(Y), float(Y.mean())


def predict_counts(model, docs: list[dict], fields: list[str], use_header: bool, tau: float, unique: bool) -> list:
    out = []
    for doc in docs:
        I, J, X, _ = pairs(doc, fields, use_header)
        prob = model.predict_proba(X)[:, 1] if len(I) else np.zeros(0)
        out.append(doc_counts(doc, cluster(len(doc["spans"]), I, J, prob, tau, [s[0] for s in doc["spans"]], unique)))
    return out


def micro(c: list) -> float:
    return f1([sum(x[i] for x in c) for i in range(3)]) * 100


def boot_diff(a: list, b: list, seed: int = 0) -> tuple[float, float, float]:
    A, Bc = np.array(a, float), np.array(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(A), size=(N_BOOT, len(A)))
    fa = lambda M: np.array([f1(tuple(M[ix].sum(axis=0))) for ix in idx]) * 100
    d = fa(A) - fa(Bc)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def run_dataset(name: str, fields: list[str], train_sets: dict, val: list, test: list, unique: bool, header_variants: list[bool]) -> dict:
    rule = [doc_counts(d, rule_rows(d)) for d in test]
    print(f"\n== {name}: {len(test)} test documents; anchor rule row F1 {micro(rule):.2f}")
    res = {"rule": round(micro(rule), 2), "runs": {}}
    for tname, train in train_sets.items():
        for uh in header_variants:
            model, npairs, pos = fit(train, fields, uh)
            best = max(THRESHOLDS, key=lambda t: micro(predict_counts(model, val, fields, uh, t, unique)))
            c = predict_counts(model, test, fields, uh, best, unique)
            diff, lo, hi = boot_diff(c, rule)
            key = f"{tname} header={uh}"
            res["runs"][key] = {"row_f1": round(micro(c), 2), "threshold": best, "diff_vs_rule": [round(diff, 2), round(lo, 2), round(hi, 2)], "train_pairs": npairs, "positive_rate": round(pos, 3)}
            print(f"  {key}: row F1 {micro(c):.2f} (tau {best}) | minus rule {diff:+.2f} [{lo:+.2f}, {hi:+.2f}] | {npairs} training pairs, {100 * pos:.1f}% same-row")
    return res


def main() -> None:
    out = {}
    # ---- DocILE
    sp = json.loads((ROOT / "results" / "docile_rows" / "splits.json").read_text(encoding="utf-8"))
    fields = sorted(DB.FIELDS)
    test = prep("docile", sp["test"])
    val = prep("docile", sp["val"])
    trains = {f"N={n}": prep("docile", sp["train"]["0"][:n]) for n in (200, 1000)}
    out["docile"] = run_dataset("DocILE (layout-disjoint test)", fields, trains, val, test, unique=False, header_variants=[True, False])
    # ---- ad-buy
    ad = B.load_docs()
    split = lambda n: json.loads(next((B.DATA / "few_shot-splits").glob(f"DeepForm-unk_template-train_{n}-*-SD_0.json")).read_text(encoding="utf-8"))
    primary = json.loads(B.PRIMARY.read_text(encoding="utf-8"))
    s200 = split(200)
    pick = lambda ids: prep("adbuy", [g for g in ids if g in ad], ad)
    out["adbuy"] = run_dataset("VRDU ad-buy (unseen templates, primary documents)", B.FIELDS,
                               {f"N={n}": pick(split(n)["train"]) for n in (100, 200)}, pick(s200["valid"]), pick(primary), unique=True, header_variants=[False])
    p = ROOT / "results" / "table_linking" / "step1.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\nregistered rule (primary run = the largest N, with the header feature for DocILE): adopt if the gain is >= +1.0 with an interval above 0 on BOTH sets")
    for name, key in (("docile", "N=1000 header=True"), ("adbuy", "N=200 header=False")):
        g = out[name]["runs"][key]["diff_vs_rule"]
        print(f"  {name}: {g[0]:+.2f} [{g[1]:+.2f}, {g[2]:+.2f}] -> {'meets' if g[0] >= 1.0 and g[1] > 0 else 'does not meet'} the rule")


if __name__ == "__main__":
    main()
