"""Stage B: grounded, matched comparison of the zero-shot VLM against Track-1 BROS on unseen templates.

Both systems are scored on the same documents (the VLM's 100 sampled documents of each held-out template,
which lie inside every UTL test set), against the same gold spans (BROS's error dumps), with the same metric
(exact entity-span micro F1 = seqeval strict on word-level BIO).

Grounding turns each VLM value (one per field, `--one-per-field` decoding) into one word span. Rule, fixed
before any grounded result was seen and based on gold properties only:
  1. exact match of the word-normalised value on the page it was extracted from, first occurrence;
  2. else the first exact match anywhere in the document;
  3. else the best fuzzy window (token F1 >= 0.5), extraction page first.
One span per value because VRDU annotates one occurrence per entity (repeated text is annotated once in
190/1437 entities of these documents), and the annotated copy is not reliably the first or last (93 vs 84).

Also reported, labelled as an upper bound: "any occurrence", where a predicted value counts as correct if its
grounded text equals a gold span's text for that field, wherever the gold copy sits.

Usage: python scripts/matched_eval.py --out results/matched_eval/metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zero_shot_kv import ROOT, norm, prf, sbert_embedder, tok_f1  # noqa: E402
from zero_shot_kv_v2 import assign_fields, read_rows, schema_fields, schema_texts  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
FOLDS = {0: "Amendment", 1: "Dissemination_Report", 2: "Short-Form"}
SEEDS = (13, 42)
N_BOOT = 1000


# ---------------------------------------------------------------- grounding
class DocText:
    """Word-normalised token stream of one document, with the word index and page of every token."""

    def __init__(self, words: list[str], page_ids: list[int]):
        self.words = words
        self.tokens, self.owner = [], []
        for i, w in enumerate(words):
            for t in norm(w).split():
                self.tokens.append(t)
                self.owner.append(i)
        self.page = [page_ids[i] for i in self.owner]

    def span(self, i: int, n: int) -> tuple[int, int]:
        return self.owner[i], self.owner[i + n - 1] + 1

    def ground(self, value: str, page: int | None) -> tuple[int, int] | None:
        vt = norm(value).split()
        n = len(vt)
        if not n:
            return None
        exact = [i for i in range(len(self.tokens) - n + 1) if self.tokens[i:i + n] == vt]
        if exact:
            on_page = [i for i in exact if self.page[i] == page]
            return self.span((on_page or exact)[0], n)
        vocab = set(vt)
        best: dict[bool, tuple[float, int, int]] = {}
        for i, t in enumerate(self.tokens):
            if t not in vocab:
                continue
            for L in range(max(1, n - 2), n + 3):
                if i + L > len(self.tokens):
                    break
                s = tok_f1(" ".join(self.tokens[i:i + L]), value)
                key = self.page[i] == page
                if s >= 0.5 and s > best.get(key, (0.0,))[0]:
                    best[key] = (s, i, L)
        pick = best.get(True) or best.get(False)
        return self.span(pick[1], pick[2]) if pick else None


def vlm_spans(text: DocText, assigned: dict) -> list[list]:
    """Grounded [field, start, end, text] spans; a span overlapping a better-matched field's span is dropped."""
    used: set[int] = set()
    out = []
    for f, (sim, value, page) in sorted(((f, c[0]) for f, c in assigned.items()), key=lambda x: -x[1][0]):
        sp = text.ground(value, page)
        if sp and not used.intersection(range(*sp)):
            used.update(range(*sp))
            out.append([f, sp[0], sp[1], " ".join(text.words[sp[0]:sp[1]])])
    return out


# ---------------------------------------------------------------- scoring
def span_counts(pred: list[list], gold: list[list], any_occurrence: bool = False) -> dict:
    """Per-field (tp, n_pred, n_gold) plus '_micro'. Exact spans, or field+text matches for any_occurrence."""
    key = (lambda s: (s[0], norm(s[3]))) if any_occurrence else (lambda s: (s[0], s[1], s[2]))
    p, g = Counter(key(s) for s in pred), Counter(key(s) for s in gold)
    out: dict = defaultdict(lambda: [0, 0, 0])
    for k, n in p.items():
        out[k[0]][0] += min(n, g.get(k, 0)); out[k[0]][1] += n
    for k, n in g.items():
        out[k[0]][2] += n
    out["_micro"] = [sum(v[i] for f, v in out.items() if f != "_micro") for i in range(3)]
    return {f: tuple(v) for f, v in out.items()}


def f1_of(c) -> float:
    tp, n_pred, n_gold = c
    p, r = (tp / n_pred if n_pred else 0.0), (tp / n_gold if n_gold else 0.0)
    return 2 * p * r / (p + r) if p + r else 0.0


def paired_bootstrap(per_doc: dict[str, list[dict]], key: str, seed: int = 0) -> dict[str, np.ndarray]:
    """F1 of every system over N_BOOT document resamples (same resample for all systems -> paired)."""
    names = list(per_doc)
    arr = np.array([[per_doc[s][d].get(key, (0, 0, 0)) for d in range(len(per_doc[names[0]]))] for s in names],
                   dtype=float)  # (systems, docs, 3)
    rng = np.random.default_rng(seed)
    out = {s: np.empty(N_BOOT) for s in names}
    for b in range(N_BOOT):
        idx = rng.integers(0, arr.shape[1], arr.shape[1])
        for j, s in enumerate(names):
            out[s][b] = f1_of(arr[j, idx].sum(axis=0))
    return out


def ci(x: np.ndarray) -> list[float]:
    return [round(float(v), 4) for v in np.percentile(x, [2.5, 97.5])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2-dir", default="results/zero_shot_kv_v2")
    ap.add_argument("--bros-dir", default="results/vrdu_track1_bros_errdump")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from data.vrdu_loader import load_vrdu

    examples, _, _ = load_vrdu(ROOT / "data" / "vrdu", "registration-form")
    text = {e.guid: DocText(e.words, e.page_ids) for e in examples}

    v2 = Path(args.v2_dir)
    docs = {d["doc_id"].split("/", 1)[1]: d for d in json.loads((v2 / "manifest.json").read_text(encoding="utf-8"))
            if d["dataset"] == "registration-form"}
    rows = read_rows([str(p) for p in sorted(v2.glob("preds_*.jsonl"))])
    desc = json.loads((v2 / "schema_desc.json").read_text(encoding="utf-8")).get("registration-form", {})
    fields = schema_fields("registration-form")
    embed = sbert_embedder()
    field_emb = {mode: embed(schema_texts(mode, fields, desc)) for mode in ("key", "named")}

    bros_metrics = json.loads((Path(args.bros_dir) / "metrics.json").read_text())
    report: dict = {"protocol": "UTL folds of the matched-size protocol (N=200), seeds 13/42 for BROS; "
                                "VLM zero-shot, one answer per field; exact entity-span micro F1",
                    "sanity": {}, "folds": {}}
    pooled: dict[str, list[dict]] = defaultdict(list)
    grounding = Counter()

    for fold, template in FOLDS.items():
        bros = {}
        for s in SEEDS:
            path = Path(args.bros_dir) / f"UTL_fold{fold}_seed{s}" / "test_entities.jsonl"
            bros[s] = {r["guid"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}
            # sanity: exact-span F1 over the full test set must reproduce the seqeval F1 BROS reported
            full = [0, 0, 0]
            for r in bros[s].values():
                c = span_counts(r["pred"], r["gold"])["_micro"]
                full = [a + b for a, b in zip(full, c)]
            reported = next(r["test"]["f1"] for r in bros_metrics["UTL"] if r["fold"] == fold and r["seed"] == s)
            report["sanity"][f"fold{fold}_seed{s}"] = {"recomputed_f1": round(f1_of(full), 4), "reported_f1": reported}

        matched = sorted(g for g in docs if docs[g]["template"] == template and g in bros[SEEDS[0]]
                         and len(rows.get(docs[g]["doc_id"], [])) == len(docs[g]["pages"]))
        per_doc: dict[str, list[dict]] = defaultdict(list)
        for g in matched:
            gold = bros[SEEDS[0]][g]["gold"]
            for s in SEEDS:
                per_doc[f"bros_seed{s}"].append(span_counts(bros[s][g]["pred"], gold))
            for mode in ("key", "named"):
                assigned = assign_fields(rows[docs[g]["doc_id"]], mode, fields, field_emb[mode], embed, True)
                spans = vlm_spans(text[g], assigned)
                grounding[f"{mode}_values"] += len(assigned)
                grounding[f"{mode}_grounded"] += len(spans)
                per_doc[f"vlm_{mode}"].append(span_counts(spans, gold))
                per_doc[f"vlm_{mode}_any_occurrence"].append(span_counts(spans, gold, any_occurrence=True))
        for s, v in per_doc.items():
            pooled[s] += v

        boot = paired_bootstrap(per_doc, "_micro")
        bros_mean = (boot["bros_seed13"] + boot["bros_seed42"]) / 2
        systems = {}
        for s, v in per_doc.items():
            c = [sum(x.get("_micro", (0, 0, 0))[i] for x in v) for i in range(3)]
            per_field = {f: round(f1_of([sum(x.get(f, (0, 0, 0))[i] for x in v) for i in range(3)]), 4)
                         for f in fields if any(x.get(f, (0, 0, 0))[2] for x in v)}
            systems[s] = {**prf(*c), "f1_ci95": ci(boot[s]), "per_field_f1": per_field}
            if s.startswith("vlm"):
                systems[s]["minus_bros_mean_f1"] = round(float(np.mean(boot[s] - bros_mean)), 4)
                systems[s]["minus_bros_mean_ci95"] = ci(boot[s] - bros_mean)
        report["folds"][f"fold{fold}_{template}"] = {"n_docs": len(matched), "systems": systems}

    boot = paired_bootstrap(pooled, "_micro")
    bros_mean = (boot["bros_seed13"] + boot["bros_seed42"]) / 2
    report["pooled"] = {s: {"f1": round(f1_of([sum(x.get("_micro", (0, 0, 0))[i] for x in v) for i in range(3)]), 4),
                            "f1_ci95": ci(boot[s]),
                            **({"minus_bros_mean_ci95": ci(boot[s] - bros_mean)} if s.startswith("vlm") else {})}
                        for s, v in pooled.items()}
    report["grounding"] = dict(grounding)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("sanity", "pooled", "grounding")}, indent=2))


def _selftest() -> None:
    t = DocText(["1.", "Name", "Acme", "Corp", "2.", "Signed", "Acme", "Corp,", "Jan", "5,", "2020"],
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    assert t.ground("Acme Corp", 0) == (2, 4)            # first exact match on the extraction page
    assert t.ground("Acme Corp", 1) == (6, 8)            # extraction page wins over an earlier copy
    assert t.ground("acme corp.", 5) == (2, 4)           # page not found -> first match anywhere
    assert t.ground("January 5, 2020", 1) == (9, 11)     # fuzzy window (token F1 >= 0.5)
    assert t.ground("Nothing like it", 0) is None
    gold = [["registrant_name", 6, 8, "Acme Corp,"]]
    assert span_counts([["registrant_name", 2, 4, "Acme Corp"]], gold)["_micro"] == (0, 1, 1)
    assert span_counts([["registrant_name", 2, 4, "Acme Corp"]], gold, any_occurrence=True)["_micro"] == (1, 1, 1)
    spans = vlm_spans(t, {"registrant_name": [(0.9, "Acme Corp", 1)], "signer_name": [(0.5, "Acme Corp", 1)]})
    assert spans == [["registrant_name", 6, 8, "Acme Corp,"]], spans  # weaker overlapping field is dropped
    print("[+] matched_eval selftest passed")


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        _selftest()
    else:
        main()
