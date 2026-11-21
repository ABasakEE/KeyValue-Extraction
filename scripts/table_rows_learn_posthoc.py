"""Study 35, step 1, POST HOC checks (written after seeing the registered result; labelled as such in STUDIES.md).

    <python with scikit-learn> scripts/table_rows_learn_posthoc.py

The registered run chose the threshold at the edge of its grid on both datasets (0.9 on DocILE, 0.2 on ad-buy) and never constrained DocILE clusters. Here: (1) pair-level quality of the
classifier and the recall of the candidate band, (2) a wider threshold grid, (3) DocILE with the "no repeated field in a row" constraint that ad-buy used. Same data, same training sets,
same bootstrap; only the largest N of each dataset.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import roc_auc_score

import table_rows_learn as T


def pair_quality(model, docs: list, fields: list, use_header: bool) -> dict:
    ys, ps, same_row_total, same_row_in_band = [], [], 0, 0
    for doc in docs:
        I, J, X, Y = T.pairs(doc, fields, use_header)
        if len(I):
            ys.append(Y); ps.append(model.predict_proba(X)[:, 1])
        n = {}
        for r in doc["row_of"]:
            if r >= 0:
                n[r] = n.get(r, 0) + 1
        same_row_total += sum(c * (c - 1) // 2 for c in n.values())
        same_row_in_band += int(Y.sum())
    y, p = np.concatenate(ys), np.concatenate(ps)
    pred = p >= 0.5
    return {"auc": round(float(roc_auc_score(y, p)), 4), "precision_at_0.5": round(float((y[pred] == 1).mean()), 4), "recall_at_0.5": round(float(pred[y == 1].sum() / max(1, y.sum())), 4),
            "candidate_band_recall": round(same_row_in_band / max(1, same_row_total), 4), "pairs": int(len(y))}


def main() -> None:
    out = {}
    sp = json.loads((T.ROOT / "results" / "docile_rows" / "splits.json").read_text(encoding="utf-8"))
    fields = sorted(T.DB.FIELDS)
    test, val, train = T.prep("docile", sp["test"]), T.prep("docile", sp["val"]), T.prep("docile", sp["train"]["0"][:1000])
    model, _, _ = T.fit(train, fields, True)
    rule = [T.doc_counts(d, T.rule_rows(d)) for d in test]
    out["docile_pairs"] = pair_quality(model, test, fields, True)
    print("DocILE test pairs:", out["docile_pairs"])
    grid = [0.5, 0.7, 0.9, 0.95, 0.98, 0.99, 0.995]
    for unique in (False, True):
        scores = {t: T.micro(T.predict_counts(model, val, fields, True, t, unique)) for t in grid}
        best = max(scores, key=scores.get)
        c = T.predict_counts(model, test, fields, True, best, unique)
        diff, lo, hi = T.boot_diff(c, rule)
        out[f"docile unique={unique}"] = {"tau": best, "val_row_f1": {str(k): round(v, 2) for k, v in scores.items()}, "test_row_f1": round(T.micro(c), 2), "minus_rule": [round(diff, 2), round(lo, 2), round(hi, 2)]}
        print(f"DocILE N=1000 header=True, unique={unique}: tau {best}, validation row F1 by tau {({k: round(v, 1) for k, v in scores.items()})} -> test {T.micro(c):.2f}, minus rule {diff:+.2f} [{lo:+.2f}, {hi:+.2f}]")
    # ad-buy
    ad = T.B.load_docs()
    split = json.loads(next((T.B.DATA / "few_shot-splits").glob("DeepForm-unk_template-train_200-*-SD_0.json")).read_text(encoding="utf-8"))
    pick = lambda ids: T.prep("adbuy", [g for g in ids if g in ad], ad)
    test_a, val_a, train_a = pick(json.loads(T.B.PRIMARY.read_text(encoding="utf-8"))), pick(split["valid"]), pick(split["train"])
    model_a, _, _ = T.fit(train_a, T.B.FIELDS, False)
    rule_a = [T.doc_counts(d, T.rule_rows(d)) for d in test_a]
    out["adbuy_pairs"] = pair_quality(model_a, test_a, T.B.FIELDS, False)
    print("ad-buy test pairs:", out["adbuy_pairs"])
    grid_a = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    scores = {t: T.micro(T.predict_counts(model_a, val_a, T.B.FIELDS, False, t, True)) for t in grid_a}
    best = max(scores, key=scores.get)
    c = T.predict_counts(model_a, test_a, T.B.FIELDS, False, best, True)
    diff, lo, hi = T.boot_diff(c, rule_a)
    out["adbuy wider grid"] = {"tau": best, "val_row_f1": {str(k): round(v, 2) for k, v in scores.items()}, "test_row_f1": round(T.micro(c), 2), "minus_rule": [round(diff, 2), round(lo, 2), round(hi, 2)]}
    print(f"ad-buy N=200: tau {best}, validation row F1 by tau {({k: round(v, 1) for k, v in scores.items()})} -> test {T.micro(c):.2f}, minus rule {diff:+.2f} [{lo:+.2f}, {hi:+.2f}]")
    p = T.ROOT / "results" / "table_linking" / "step1_posthoc.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
