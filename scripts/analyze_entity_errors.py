"""Summarise test_entities.jsonl dumps written by src/train.py.

For each (template, entity type): exact matches, gold spans the model labelled
as another type (role confusion), gold spans missed entirely, and the most
common text of wrong predictions.

Usage: python scripts/analyze_entity_errors.py [--out errors.json] results/<exp>/UTL_fold2_seed13/test_entities.jsonl [...]
"""

import argparse
import json
from collections import Counter, defaultdict


def overlaps(a, b):
    return a[1] < b[2] and b[1] < a[2]


def analyze(paths):
    stats = defaultdict(Counter)
    wrong_text = defaultdict(Counter)
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                doc = json.loads(line)
                gold, pred = doc["gold"], doc["pred"]
                for g in gold:
                    key = (doc["template"], g[0])
                    stats[key]["gold"] += 1
                    hits = [p for p in pred if overlaps(p, g)]
                    if any(p[:3] == g[:3] for p in hits):
                        stats[key]["exact"] += 1
                    elif any(p[0] == g[0] for p in hits):
                        stats[key]["partial"] += 1
                    elif hits:
                        stats[key]["labelled_as_other_type"] += 1
                    else:
                        stats[key]["missed"] += 1
                for p in pred:
                    if not any(p[:3] == g[:3] for g in gold):
                        key = (doc["template"], p[0])
                        stats[key]["wrong_pred"] += 1
                        # what the wrong span really was: a gold entity of another type, or unlabelled text
                        other = [g[0] for g in gold if overlaps(p, g) and g[0] != p[0]]
                        stats[key][f"wrong_pred_on_{other[0]}" if other else "wrong_pred_on_O"] += 1
                        wrong_text[key][p[3][:50]] += 1

    report: dict = defaultdict(dict)
    for template, etype in sorted(stats):
        report[template][etype] = {
            **stats[(template, etype)],
            "top_wrong_preds": wrong_text[(template, etype)].most_common(5),
        }
    return dict(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--out", default=None, help="Also write the report as JSON")
    args = parser.parse_args()
    report = analyze(args.paths)
    for template, by_type in report.items():
        for etype, row in by_type.items():
            counts = {k: v for k, v in row.items() if k != "top_wrong_preds"}
            print(f"\n== {template} / {etype}: {counts}")
            for text, n in row["top_wrong_preds"]:
                print(f"     wrong pred x{n}: {text!r}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
