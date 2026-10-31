"""Draw page 1 of one real VRDU registration form per template.

Uses the OCR word boxes released with VRDU (no page images are needed): every
word is drawn as a grey box and words inside a labelled field are coloured by
field type. The point is to show how differently the three templates place the
same six fields.

Run from anywhere:  python intensive_assessment/render_template_examples.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data.vrdu_loader import load_vrdu  # noqa: E402

OUTPUT = Path(__file__).resolve().parent / "figures" / "fig8_template_examples.png"
TEMPLATES = ["Amendment", "Dissemination_Report", "Short-Form"]
# Categorical slots 1-6 of the reference palette used by generate_presentation_figures.py
FIELD_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
WORD_FILL, WORD_EDGE, INK, SURFACE = "#e1e0d9", "#c3c2b7", "#0b0b0b", "#fcfcfb"


def first_page(ex):
    pages = ex.page_ids or [0] * len(ex.words)
    p0 = min(pages)
    return [(b, t) for b, t, p in zip(ex.boxes, ex.ner_tags, pages) if p == p0]


def main():
    examples, _, id2label = load_vrdu(ROOT / "data" / "vrdu")
    fields = sorted({id2label[i][2:] for i in id2label if id2label[i] != "O"})
    color = {f: FIELD_COLORS[i] for i, f in enumerate(fields)}

    picks = {}
    for tmpl in TEMPLATES:
        docs = [e for e in examples if e.template_id == tmpl]
        # the page-1 view that shows the most distinct labelled fields reads best
        picks[tmpl] = max(docs, key=lambda e: len({id2label[t][2:] for _, t in first_page(e) if t}))

    fig, axes = plt.subplots(1, 3, figsize=(15, 7.4), facecolor=SURFACE)
    for ax, tmpl in zip(axes, TEMPLATES):
        for (x0, y0, x1, y1), tag in first_page(picks[tmpl]):
            label = id2label[tag]
            if label == "O":
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=WORD_FILL,
                                       edgecolor=WORD_EDGE, linewidth=0.3))
            else:
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=color[label[2:]],
                                       edgecolor=color[label[2:]], linewidth=0.6))
        ax.set_xlim(0, 1000)
        ax.set_ylim(1000, 0)
        ax.set_aspect(1.294)  # US Letter page, 11 / 8.5
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_edgecolor(WORD_EDGE)
        ax.set_facecolor("#ffffff")
        ax.set_title(tmpl.replace("_", " "), fontsize=15, fontweight="bold", color=INK, pad=10)

    fig.legend(handles=[Patch(color=color[f], label=f) for f in fields] +
               [Patch(facecolor=WORD_FILL, edgecolor=WORD_EDGE, label="other OCR words")],
               loc="lower center", ncol=4, frameon=False, fontsize=12)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.12, wspace=0.08)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=200, facecolor=SURFACE)
    print(f"[+] {OUTPUT.relative_to(ROOT)}")
    for tmpl in TEMPLATES:
        print(f"    {tmpl}: {picks[tmpl].guid}")


if __name__ == "__main__":
    main()
