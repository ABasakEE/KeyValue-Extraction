"""Qualitative example: one unseen Short-Form page drawn from its OCR word boxes, with the BROS span and the
zero-shot reader's grounded span for `registrant_name` (Study 31 confirm set, saved predictions only).

    python poster/scripts/make_example.py [--guid <pdf name>]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style as S  # noqa: E402

sys.path[:0] = [str(S.ROOT / "src"), str(S.ROOT)]
from data.vrdu_loader import load_vrdu  # noqa: E402

FIELD = "registrant_name"


def spans(rec, key):
    return [(s, e, t) for f, s, e, t in rec.get(key, []) if f == FIELD]


def pick(records):
    """Short-Form doc where BROS (seed 13) picks a wrong registrant and the reader picks the gold one."""
    for r in records:
        if r["template"] != "Short-Form":
            continue
        g = {(s, e) for s, e, _ in spans(r, "gold")}
        b = {(s, e) for s, e, _ in spans(r, "bros_seed13")}
        v = {(s, e) for s, e, _ in spans(r, "vlm_V2A2")}
        if g and v and v <= g and b and not (b & g):
            yield r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--guid", default=None)
    a = ap.parse_args()
    src = S.ROOT / "results/schema_guided_kv_v2/hybrid_input_V2A2.jsonl"
    recs = [json.loads(line) for line in open(src, encoding="utf-8")]
    cands = list(pick(recs))
    rec = next(r for r in cands if r["guid"] == a.guid) if a.guid else cands[0]
    print(f"{len(cands)} candidate docs; using {rec['guid']}")
    for k in ("gold", "bros_seed13", "vlm_V2A2"):
        print(f"  {k:<12} {spans(rec, k)}")

    exs, _, _ = load_vrdu(S.ROOT / "data" / "vrdu", "registration-form")
    ex = next(e for e in exs if e.guid == rec["guid"])
    b_s, b_e, _ = spans(rec, "bros_seed13")[0]
    v_s, v_e, _ = spans(rec, "vlm_V2A2")[0]
    page = ex.page_ids[v_s]
    idx = [i for i, p in enumerate(ex.page_ids) if p == page]
    lo, hi = min(b_s, v_s), max(b_e, v_e)
    top = max(0, min(ex.boxes[i][1] for i in range(lo, hi)) - 60)
    bot = min(1000, max(ex.boxes[i][3] for i in range(lo, hi)) + 60)
    A = 11 / 8.5  # US letter aspect: y units are stretched to page proportions
    X0, X1 = 0, 620  # crop: the two blocks sit in the left part of the page
    W = S.HALF_W
    pt_per_unit = W * 72 / (X1 - X0)

    S.apply()
    fig, ax = plt.subplots(figsize=(W, W * (bot - top) * A / (X1 - X0)))
    for i in idx:
        x0, y0, x1, y1 = ex.boxes[i]
        if y1 < top or y0 > bot or x0 > X1:
            continue
        word = "".join(c for c in ex.words[i] if ord(c) < 0x2000)
        ax.add_patch(Rectangle((x0, y0 * A), x1 - x0, (y1 - y0) * A, fc="#F1F1F1", ec="none"))
        fs = max(3.0, 0.62 * (y1 - y0) * A * pt_per_unit)
        ax.text(x0, (y0 + y1) / 2 * A, word, fontsize=fs, va="center", ha="left", color="#4A4A4A", clip_on=True)

    def outline(s, e, color, ls):
        x0 = min(ex.boxes[i][0] for i in range(s, e)) - 5
        y0 = min(ex.boxes[i][1] for i in range(s, e)) * A - 5
        x1 = max(ex.boxes[i][2] for i in range(s, e)) + 5
        y1 = max(ex.boxes[i][3] for i in range(s, e)) * A + 5
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="none", ec=color, lw=2.2, ls=ls))
        return x1, (y0 + y1) / 2

    for (s_, e_, color, ls, lab) in ((b_s, b_e, S.GREY_DARK, (0, (3, 1.5)), "BROS"), (v_s, v_e, S.ACCENT, "-", "reader")):
        x, y = outline(s_, e_, color, ls)
        ax.annotate(lab, xy=(x, y), xytext=(X1 + 10, y), color=color, fontsize=S.FONT_SIZE - 4, fontweight="semibold",
                    va="center", ha="left", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=color, lw=1.2, shrinkA=2, shrinkB=0))
    ax.set_xlim(X0, X1)
    ax.set_ylim(bot * A, top * A)
    ax.set_aspect("equal")
    ax.axis("off")
    S.save(fig, "f6_example_shortform")
    (S.FIG / "f6_example_shortform.json").write_text(json.dumps({"guid": rec["guid"], "page": page, "source": str(src.relative_to(S.ROOT))}, indent=1))


if __name__ == "__main__":
    main()
