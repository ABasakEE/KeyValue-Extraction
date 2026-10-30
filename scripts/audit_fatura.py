"""Study 13 (CPU): is FATURA usable for the held-out-template protocol? Writes results/fatura_audit/summary.json.

Checks: (1) template structure and the official template-disjoint split, (2) which fields exist in how many
templates and how their wording varies, (3) annotation granularity (blocks vs words), (4) how the Hugging Face-format
files relate to the original blocks (box convention, which classes share a tag id), (5) image size vs box coordinates.
Reads only JSON/JPEG headers from data/fatura/raw (downloaded from Zenodo 10371464, CC-BY-4.0).
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "data" / "fatura" / "raw" / "invoices_dataset_final"
SKIP = {"OTHER", "TABLE", "LOGO", "INVOICE_INFO"}  # no per-block text in Original_Format


def main(out: Path) -> None:
    from PIL import Image

    orig = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (R / "Annotations/Original_Format").glob("*.json")}
    hf = {re.sub(r"_hugg_\w+$", "", p.stem): json.loads(p.read_text(encoding="utf-8"))
          for p in (R / "Annotations/layoutlm_HF_format").glob("*.json")}
    assert set(hf) == set(orig), f"{len(set(hf) ^ set(orig))} documents differ between annotation formats"
    tmpl = {k: int(re.match(r"Template(\d+)_", k).group(1)) for k in orig}
    per_t = Counter(tmpl.values())
    report: dict = {"documents": len(orig), "templates": len(per_t), "instances_per_template": sorted(set(per_t.values()))}

    split = (R / "Strat2_Split.txt").read_text()
    lists = {m.group(1): json.loads(m.group(2)) for m in re.finditer(r"(\w+_inds) = (\[.*?\])", split)}
    report["official_unseen_template_split"] = {k: len(v) for k, v in lists.items()}
    report["official_split_disjoint"] = not set(lists["train_inds"]) & set(lists["dev_inds"])

    present: dict[str, set] = defaultdict(set)
    wording: dict[str, dict] = defaultdict(dict)
    for k, d in orig.items():
        for cls, v in d.items():
            if cls not in SKIP and isinstance(v, dict) and v.get("text"):
                present[cls].add(tmpl[k])
                wording[cls].setdefault(tmpl[k], v["text"].split("\n")[0][:40])
    report["fields_templates_with_text"] = {c: len(t) for c, t in sorted(present.items(), key=lambda x: -len(x[1]))}
    report["core_fields_ge_29_templates"] = [c for c, t in present.items() if len(t) >= 29]
    report["label_prefix_examples"] = {c: [wording[c][t] for t in sorted(wording[c])[:4]]
                                       for c in ("DATE", "NUMBER", "TOTAL", "BUYER")}

    # HF-format boxes are the original block boxes with y flipped (page height 841): match blocks by box to see
    # which tag id each field class carries in the HF files
    H = 841
    close = lambda a, b: all(abs(p - q) <= 2.5 for p, q in zip(a, b))
    found, total = Counter(), Counter()
    ids: dict[str, Counter] = defaultdict(Counter)
    runs_words = []
    for k, d in hf.items():
        boxes = {tuple(b): t for b, t in zip(d["bboxes"], d["ner_tags"])}
        for cls, v in orig[k].items():
            if cls in SKIP or not (isinstance(v, dict) and v.get("text") and v.get("bbox")):
                continue
            (xa, ya), (xb, yb) = v["bbox"]
            fb = (min(xa, xb), H - max(ya, yb), max(xa, xb), H - min(ya, yb))
            total[cls] += 1
            hit = next((bx for bx in boxes if close(bx, fb)), None)
            if hit:
                found[cls] += 1
                ids[cls][boxes[hit]] += 1
        runs, prev = [], None
        for w, b, t in zip(d["words"], d["bboxes"], d["ner_tags"]):
            key = (tuple(b), t)
            if key != prev:
                runs.append(0)
                prev = key
            runs[-1] += 1
        runs_words += runs
    report["hf_box_is_flipped_original_block_box_rate"] = round(sum(found.values()) / sum(total.values()), 4)
    by_tag: dict[int, list] = defaultdict(list)
    for c, t in ids.items():
        by_tag[t.most_common(1)[0][0]].append(c)
    report["hf_tag_id_to_classes"] = {int(t): sorted(c) for t, c in sorted(by_tag.items())}
    report["hf_tag_ids_lumping_several_classes"] = [int(t) for t, c in by_tag.items() if len(c) > 1]
    report["mean_words_per_block"] = round(sum(runs_words) / len(runs_words), 2)

    sizes, extents = Counter(), []
    for k in sorted(hf)[:: len(hf) // 50]:
        sizes[Image.open(R / "images" / f"{k}.jpg").size] += 1
        extents.append((max(b[2] for b in hf[k]["bboxes"]), max(b[3] for b in hf[k]["bboxes"])))
    report["image_sizes"] = {f"{w}x{h}": n for (w, h), n in sizes.most_common(5)}
    report["max_box_extent_seen"] = [max(e[0] for e in extents), max(e[1] for e in extents)]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(ROOT / "results" / "fatura_audit" / "summary.json")
