"""Contact sheet of four synthetic form families with their labels drawn on.

First generate forms (needs faker and augraphy):
    python src/data/synthetic_generator.py --output_dir <dir> --n_train 14 --n_test 0
then:
    python intensive_assessment/render_synthetic_examples.py <dir>
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from PIL import Image

OUTPUT = Path(__file__).resolve().parent / "figures" / "fig9_synthetic_examples.png"
PICKS = [("syn_invoice_0_0000", "Invoice"), ("syn_intake_0_0004", "Medical intake"),
         ("syn_manifest_0_0003", "Shipping manifest"), ("syn_fara_0_0006", "FARA registration")]
COL = {"question": "#2a78d6", "answer": "#eb6834", "header": "#1baf7a"}
LINK, SURFACE = "#52514e", "#fcfcfb"


def main(synth_dir):
    base = Path(synth_dir) / "training_data"
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6), facecolor=SURFACE)
    for ax, (stem, title) in zip(axes, PICKS):
        ax.imshow(Image.open(base / "images" / f"{stem}.png").convert("RGB"))
        form = json.load(open(base / "annotations" / f"{stem}.json"))["form"]
        by_id = {e["id"]: e for e in form}
        for e in form:
            if e["label"] in COL:
                x0, y0, x1, y1 = e["box"]
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                       edgecolor=COL[e["label"]], linewidth=1.2))
            for a, b in e.get("linking", []):
                if a == e["id"] and b in by_id:
                    ka, kb = e["box"], by_id[b]["box"]
                    ax.plot([(ka[0] + ka[2]) / 2, (kb[0] + kb[2]) / 2],
                            [(ka[1] + ka[3]) / 2, (kb[1] + kb[3]) / 2], color=LINK, lw=0.6)
        # crop the empty lower part of the page
        ax.set_ylim(max(e["box"][3] for e in form) + 120, 0)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axis("off")
    fig.legend(handles=[Patch(facecolor="none", edgecolor=c, label=l) for l, c in COL.items()] +
               [Line2D([], [], color=LINK, label="key-value link")],
               loc="lower center", ncol=4, frameon=False, fontsize=12)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.12, wspace=0.04)
    fig.savefig(OUTPUT, dpi=180, facecolor=SURFACE)
    print(f"[+] {OUTPUT}")


if __name__ == "__main__":
    main(sys.argv[1])
