"""Study 27: score raw VLM answers (xfund_vlm_generate.py) with the LiLT end-to-end rules of src/e2e_link.py (CPU).

    python scripts/xfund_vlm_score.py --raw results/xfund_vlm/V2_raw.jsonl --out results/xfund_vlm/V2_e2e.json
    python scripts/xfund_vlm_score.py --raw results/xfund_vlm/V1_raw.jsonl results/xfund_vlm/V2_raw.jsonl \
        --common --out results/xfund_vlm/common_V1_V2.json

Keys follow scripts/xfund_e2e_eval.py with mode ``vlm``: ``{all,sibling_free}|vlm|{entity,link,link_qa}`` and
``|group|<name>`` (recall only).  Every form in the raw file is scored; an answer that fails to parse or a generation
error counts as an empty prediction.  ``--common`` scores each file on the forms all files share.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT / "src"), str(Path(__file__).resolve().parent)):
    sys.path.insert(0, p)

from data.xfund_loader import load_xfund_split  # noqa: E402
from e2e_link import add_counts, gold_qa_links, gold_structure, prf, qa_pred_links  # noqa: E402
from vlm_link import parse  # noqa: E402
from xfund_e2e_eval import link_groups  # noqa: E402
from xfund_link_eval import LANGS, sibling_free  # noqa: E402


def read_rows(path: Path) -> dict[str, dict]:
    rows = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["guid"]] = r  # a later row for the same form wins
    return rows


def bootstrap(per_form: list[tuple[int, int, int]], n: int = 1000, seed: int = 0) -> list[float]:
    if not per_form:
        return [0.0, 0.0]
    rng = random.Random(seed)
    f1s = []
    for _ in range(n):
        tp = fp = fn = 0
        for _ in per_form:
            a, b, c = per_form[rng.randrange(len(per_form))]
            tp, fp, fn = tp + a, fp + b, fn + c
        f1s.append(prf(tp, fp, fn)["f1"])
    f1s.sort()
    return [f1s[int(0.025 * n)], f1s[int(0.975 * n) - 1]]


def score(rows: dict[str, dict], examples: dict, keep: set[str], guids: list[str]) -> dict:
    counts: dict[str, list[int]] = {}
    per_form = {"entity": [], "link": [], "link_qa": []}
    stats, by_lang, models, seconds = Counter(), {}, Counter(), []
    arm = next(iter(rows.values()))["arm"] if rows else "?"
    for g in guids:
        ex, row = examples[g], rows[g]
        models[row.get("model", "?")] += 1
        seconds.append(row.get("seconds", 0.0))
        stats["errors"] += bool(row.get("error"))
        stats["truncated"] += bool(row.get("truncated"))
        ents, links, st = parse(arm, row.get("text") or "", ex)
        stats.update(st)
        g_ents, g_links = gold_structure(ex)
        g_qa, groups = gold_qa_links(ex), link_groups(ex)
        cells = {"entity": (ents, g_ents), "link": (links, g_links), "link_qa": (qa_pred_links(ents, links), g_qa)}
        for name, (pred, gold) in cells.items():
            per_form[name].append((len(pred & gold), len(pred - gold), len(gold - pred)))
        add_counts(by_lang.setdefault(g.split("_")[0], [0, 0, 0]), links, g_links)
        for sub in ("all", "sibling_free"):
            if sub == "sibling_free" and g not in keep:
                continue
            for name, (pred, gold) in cells.items():
                add_counts(counts.setdefault(f"{sub}|vlm|{name}", [0, 0, 0]), pred, gold)
            for name, grp in groups.items():
                add_counts(counts.setdefault(f"{sub}|vlm|group|{name}", [0, 0, 0]), links & grp, grp)
    out = {k: prf(*v) for k, v in counts.items()}
    n = len(guids)
    out.update({
        "arm": arm, "n_forms": n, "n_sibling_free": sum(g in keep for g in guids), "models": dict(models),
        "bootstrap_ci_f1": {k: bootstrap(v) for k, v in per_form.items()},
        "link_f1_by_language": {k: prf(*v)["f1"] for k, v in sorted(by_lang.items())},
        "parse_fail_rate": round(stats["parse_fail"] / n, 4) if n else 0.0,
        "error_rate": round(stats["errors"] / n, 4) if n else 0.0,
        "truncated_rate": round(stats["truncated"] / n, 4) if n else 0.0,
        "item_counts": {k: stats[k] for k in ("entity_lines", "link_lines", "bad_ranges", "unmatched", "dangling_links")},
        "mean_seconds_per_form": round(sum(seconds) / n, 2) if n else 0.0,
    })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="data/xfund")
    ap.add_argument("--common", action="store_true", help="score every file on the forms they all contain")
    a = ap.parse_args()
    val, train = load_xfund_split(a.data, "val", LANGS), load_xfund_split(a.data, "train", LANGS)
    examples, keep = {ex.guid: ex for ex in val}, sibling_free(val, train)
    files = {Path(p): read_rows(Path(p)) for p in a.raw}
    for p, rows in files.items():
        unknown = set(rows) - set(examples)
        if unknown:
            raise SystemExit(f"{p}: {len(unknown)} guids are not XFUND val forms, e.g. {sorted(unknown)[:2]}")
    shared = set.intersection(*(set(r) for r in files.values())) if files else set()
    res = {}
    for p, rows in files.items():
        guids = sorted(shared if a.common else rows)
        res[p.stem.replace("_raw", "")] = score(rows, examples, keep, guids) if guids else {"n_forms": 0}
    payload = res if (a.common or len(res) > 1) else next(iter(res.values()))
    if a.common:
        payload = {"common_forms": len(shared), "arms": res}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    for name, r in res.items():
        if r.get("n_forms"):
            print(f"{name}: {r['n_forms']} forms | entity {r['all|vlm|entity']['f1']:.4f} | link {r['all|vlm|link']['f1']:.4f} "
                  f"CI {r['bootstrap_ci_f1']['link']} | link_qa {r['all|vlm|link_qa']['f1']:.4f} | parse_fail {r['parse_fail_rate']}")
        else:
            print(f"{name}: no forms")


if __name__ == "__main__":
    main()
