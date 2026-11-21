"""Study 35, step 1b: other ways to turn the pair scores of step 1 into rows (registered in STUDIES.md before this ran).

    <python with scikit-learn> scripts/table_rows_cluster.py

Same classifiers, spans and test sets as `table_rows_learn.py`; variants V0 single link, V1 average link, V2 greedy correlation clustering, V3 anchor assignment with learned scores.
Selection by row F1 on the validation documents only; the selected setting is scored once on the test documents.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict

import numpy as np

import table_rows_learn as T

GRIDS = {
    "V0 single link": [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98, 0.99, 0.995],
    "V1 average link": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "V2 correlation": [-3, -2, -1, 0, 1, 2, 3],
    "V3 anchors": [0.0, 0.1, 0.3, 0.5],
}


def agglomerate(n: int, I: list, J: list, prob: np.ndarray, fields_of: list[str], unique: bool, score, threshold: float, init) -> list[list[int]]:
    """Greedy merging of the two clusters with the best `score(sum, count, size_a, size_b)` while it is at least `threshold`."""
    members = {k: [k] for k in range(n)}
    fields = {k: {fields_of[k]} for k in range(n)}
    nb: dict[int, dict[int, list]] = {k: {} for k in range(n)}   # cluster -> neighbour -> [sum of pair values, number of candidate pairs]
    for a, b, p in zip(I, J, prob):
        v = init(float(p))
        nb[a][b] = [v, 1]
        nb[b][a] = [v, 1]
    while True:
        best, pair = threshold, None
        for a, d in nb.items():
            for b, (sm, cnt) in d.items():
                if a < b and not (unique and fields[a] & fields[b]):
                    sc = score(sm, cnt, len(members[a]), len(members[b]))
                    if sc >= best and (pair is None or sc > best):
                        best, pair = sc, (a, b)
        if pair is None:
            break
        a, b = pair
        members[a] += members.pop(b)
        fields[a] |= fields.pop(b)
        nbb = nb.pop(b)
        nb[a].pop(b, None)
        for c, (sm, cnt) in nbb.items():
            if c == a:
                continue
            nb[c].pop(b, None)
            cur = nb[a].setdefault(c, [0.0, 0])
            cur[0] += sm
            cur[1] += cnt
            nb[c][a] = cur
    return list(members.values())


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def v3_anchors(doc: dict, I: list, J: list, prob: np.ndarray, tau: float) -> list[list[int]]:
    spans, d = doc["spans"], doc["d"]
    page = [d["pages"][s] for _, s, _, _ in spans]
    pair = {}
    for a, b, p in zip(I, J, prob):
        pair[(a, b)] = pair[(b, a)] = float(p)
    cy = [(T.box(d, s, e)[1] + T.box(d, s, e)[3]) / 2 for _, s, e, _ in spans]
    groups: dict[int, list[int]] = {}
    for p in set(page):
        ks = [k for k in range(len(spans)) if page[k] == p]
        counts = defaultdict(int)
        for k in ks:
            counts[spans[k][0]] += 1
        anchor = max(counts, key=lambda f: (counts[f], f))
        anchors = [k for k in ks if spans[k][0] == anchor]
        for a in anchors:
            groups[a] = [a]
        for k in ks:
            if spans[k][0] == anchor:
                continue
            cands = [(pair[(a, k)], a) for a in anchors if (a, k) in pair]
            if cands and max(cands)[0] >= tau:
                groups[max(cands)[1]].append(k)
            else:
                groups[min(anchors, key=lambda a: abs(cy[a] - cy[k]))].append(k)
    return list(groups.values())


def groups_for(variant: str, param: float, doc: dict, I: list, J: list, prob: np.ndarray, unique: bool) -> list[list[int]]:
    n, fields_of = len(doc["spans"]), [s[0] for s in doc["spans"]]
    if variant == "V0 single link":
        return T.cluster(n, I, J, prob, param, fields_of, unique)
    if variant == "V1 average link":
        return agglomerate(n, I, J, prob, fields_of, unique, lambda sm, cnt, sa, sb: sm / (sa * sb), param, float)
    if variant == "V2 correlation":
        w0 = logit(0.01) + param
        return agglomerate(n, I, J, prob, fields_of, unique, lambda sm, cnt, sa, sb: sm + (sa * sb - cnt) * w0, 1e-9, lambda p: logit(p) + param)
    return v3_anchors(doc, I, J, prob, param)


def cached(model, docs: list, fields: list, use_header: bool) -> list:
    out = []
    for doc in docs:
        I, J, X, _ = T.pairs(doc, fields, use_header)
        out.append((I, J, model.predict_proba(X)[:, 1] if len(I) else np.zeros(0)))
    return out


def evaluate(variant: str, param: float, docs: list, probs: list, unique: bool) -> list:
    return [T.doc_counts(doc, groups_for(variant, param, doc, *pr, unique)) for doc, pr in zip(docs, probs)]


def run(name: str, model, fields: list, use_header: bool, val: list, test: list, unique: bool) -> dict:
    pv, pt = cached(model, val, fields, use_header), cached(model, test, fields, use_header)
    rule = [T.doc_counts(d, T.rule_rows(d)) for d in test]
    print(f"\n== {name}: validation {len(val)} documents (rule row F1 {T.micro([T.doc_counts(d, T.rule_rows(d)) for d in val]):.2f}), test {len(test)} documents (rule {T.micro(rule):.2f})")
    res, best_overall = {}, None
    for variant, grid in GRIDS.items():
        vals = {g: T.micro(evaluate(variant, g, val, pv, unique if variant != "V3 anchors" else False)) for g in grid}
        g = max(vals, key=vals.get)
        c = evaluate(variant, g, test, pt, unique if variant != "V3 anchors" else False)
        diff, lo, hi = T.boot_diff(c, rule)
        res[variant] = {"param": g, "val_row_f1": round(vals[g], 2), "test_row_f1": round(T.micro(c), 2), "minus_rule": [round(diff, 2), round(lo, 2), round(hi, 2)]}
        print(f"  {variant:16s} best validation setting {g:>6} (validation {vals[g]:.2f}) -> test {T.micro(c):.2f}, minus rule {diff:+.2f} [{lo:+.2f}, {hi:+.2f}]")
        if best_overall is None or vals[g] > res[best_overall]["val_row_f1"]:
            best_overall = variant
    res["selected"] = best_overall
    print(f"  selected by validation: {best_overall} -> test minus rule {res[best_overall]['minus_rule']}")
    return res


def main() -> None:
    out = {}
    sp = json.loads((T.ROOT / "results" / "docile_rows" / "splits.json").read_text(encoding="utf-8"))
    fields = sorted(T.DB.FIELDS)
    test, val, train = T.prep("docile", sp["test"]), T.prep("docile", sp["val"]), T.prep("docile", sp["train"]["0"][:1000])
    model, _, _ = T.fit(train, fields, True)
    out["docile"] = run("DocILE, N = 1000, header feature", model, fields, True, val, test, unique=False)
    ad = T.B.load_docs()
    split = json.loads(next((T.B.DATA / "few_shot-splits").glob("DeepForm-unk_template-train_200-*-SD_0.json")).read_text(encoding="utf-8"))
    pick = lambda ids: T.prep("adbuy", [g for g in ids if g in ad], ad)
    model_a, _, _ = T.fit(pick(split["train"]), T.B.FIELDS, False)
    out["adbuy"] = run("ad-buy, N = 200", model_a, T.B.FIELDS, False, pick(split["valid"]), pick(json.loads(T.B.PRIMARY.read_text(encoding="utf-8"))), unique=True)
    (T.ROOT / "results" / "table_linking" / "step1b.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\nregistered rule: adopt the selected variant if the gain is >= +1.0 with an interval above 0 on BOTH sets")
    for k in ("docile", "adbuy"):
        g = out[k][out[k]["selected"]]["minus_rule"]
        print(f"  {k}: {out[k]['selected']} {g[0]:+.2f} [{g[1]:+.2f}, {g[2]:+.2f}] -> {'meets' if g[0] >= 1.0 and g[1] > 0 else 'does not meet'} the rule")


if __name__ == "__main__":
    main()
