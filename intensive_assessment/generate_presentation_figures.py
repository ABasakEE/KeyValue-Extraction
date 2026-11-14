"""Generate figures for the intensive-assessment slides and report.

Every number is read from the committed result files under ``results/`` (or, for
the XFUND knockout suite, from the saved output of
``notebooks/kaggle_ablation_adbuy.ipynb``). The only hardcoded values are the
GNN matched-UTL numbers in ``GNN_UTL_FALLBACK``, used until
``results/gnn_matched/metrics.json`` and ``results/typed_gnn_matched/metrics.json``
are committed.

Run from anywhere:  python intensive_assessment/generate_presentation_figures.py
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# fold index -> held-out template (leave_one_template_out iterates sorted(templates))
TEMPLATES = ["Amendment", "Dissemination_Report", "Short-Form"]

# TODO(Arjoe): replace once results/gnn_matched/ and results/typed_gnn_matched/ are committed.
# Values as reported in BENCHMARK_METRICS.md (Kaggle dual-T4 run).
GNN_UTL_FALLBACK = {
    "rsa_gnn": {"mtl": 85.97, "utl": 42.05, "metric": "entity F1"},
    "ner_fused": {"mtl": 81.40, "utl": 30.48, "metric": "edge link F1"},
}

# Reference palette (dataviz skill, light mode)
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 11,
    "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "text.color": INK,
    "legend.frameon": False,
    "legend.fontsize": 10,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save(fig, name):
    out = OUTPUT_DIR / name
    fig.savefig(out)
    plt.close(fig)
    print(f"[+] {out.relative_to(ROOT)}")


def label_bars(ax, bars, fmt="{:.1f}", errs=None, dy=3):
    """Value labels above each bar, clearing the error bar when one is drawn."""
    for i, b in enumerate(bars):
        h = b.get_height()
        top = h + (errs[i] if errs is not None else 0)
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, top), xytext=(0, dy),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9.5, color=INK)


def run_dirs(exp, regime):
    """Per-run result.json files of a matched MTL/UTL experiment, grouped by fold."""
    by_fold = {}
    for d in sorted((RESULTS / exp).glob(f"{regime}_fold*_seed*")):
        fold = int(re.search(r"fold(\d+)", d.name).group(1))
        by_fold.setdefault(fold, []).append(load_json(d / "result.json"))
    return by_fold


def parse_relative_coordinate_table():
    """Rows of results/RelativeCoordinate_ablation.txt -> {name: [cells...]}."""
    rows = {}
    for line in (RESULTS / "RelativeCoordinate_ablation.txt").read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 3 and cells[0] and not cells[0].startswith(("Experiment", "-")):
            rows[cells[0]] = cells[1:]
    return rows


def pct(cell):
    return float(cell.rstrip("%"))


# ==============================================================================
# Fig 1: Registration forms - UTL F1 per held-out template vs MTL
# ==============================================================================
def fig_gap_per_template():
    gap = load_json(RESULTS / "vrdu_lilt" / "metrics.json")["gap"]
    utl = run_dirs("vrdu_lilt", "UTL")
    means = [100 * np.mean([r["test"]["f1"] for r in utl[i]]) for i in range(3)]
    stds = [100 * np.std([r["test"]["f1"] for r in utl[i]], ddof=1) for i in range(3)]

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    x = np.arange(3)
    bars = ax.bar(x, means, yerr=stds, width=0.5, color=ORANGE, capsize=4,
                  error_kw={"ecolor": INK2, "elinewidth": 1}, label="UTL: template held out (mean of 3 seeds)")
    mtl = 100 * gap["mtl_mean_f1"]
    ax.axhline(mtl, color=BLUE, lw=2, label=f"MTL: all templates seen ({mtl:.1f})")
    label_bars(ax, bars, errs=stds)
    ax.set_xticks(x, [t.replace("_", " ") for t in TEMPLATES])
    ax.set_ylabel("Entity F1 (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"VRDU Registration: unseen templates lose {100 * gap['gap_f1']:.1f} F1 points (N=200)")
    ax.legend(loc="lower left")
    save(fig, "fig1_gap_per_template.png")


# ==============================================================================
# Fig 2: Per-entity MTL -> UTL collapse (dumbbell)
# ==============================================================================
def fig_per_entity():
    def entity_means(regime):
        runs = [r for fold in run_dirs("vrdu_lilt", regime).values() for r in fold]
        ents = sorted(runs[0]["test"]["per_type"])
        # entities absent from a held-out template's test set are skipped, not counted as 0
        return {e: 100 * np.mean([r["test"]["per_type"][e]["f1"] for r in runs if e in r["test"]["per_type"]])
                for e in ents}

    mtl, utl = entity_means("MTL"), entity_means("UTL")
    ents = sorted(mtl, key=lambda e: mtl[e] - utl.get(e, 0))
    y = np.arange(len(ents))

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, e in enumerate(ents):
        ax.plot([utl[e], mtl[e]], [i, i], color=AXIS, lw=2, zorder=1)
    ax.scatter([mtl[e] for e in ents], y, s=70, color=BLUE, zorder=2, label="MTL (seen)",
               edgecolor=SURFACE, linewidth=2)
    ax.scatter([utl[e] for e in ents], y, s=70, color=ORANGE, zorder=2, label="UTL (unseen)",
               edgecolor=SURFACE, linewidth=2)
    for i, e in enumerate(ents):
        ax.annotate(f"{utl[e]:.1f}", (utl[e], i), xytext=(-8, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=9, color=INK2)
        ax.annotate(f"{mtl[e]:.1f}", (mtl[e], i), xytext=(8, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=9, color=INK2)
    ax.set_yticks(y, ents)
    ax.set_xlim(-12, 108)
    ax.set_xlabel("Entity F1 (%), mean over 9 runs")
    ax.set_title("Which fields break on unseen templates")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    ax.grid(axis="y", visible=False)
    save(fig, "fig2_per_entity_collapse.png")


# ==============================================================================
# Fig 3: Ad-Buy vs Registration - layout regularity decides the gap
# ==============================================================================
def fig_adbuy_vs_registration():
    reg = load_json(RESULTS / "vrdu_lilt" / "metrics.json")["gap"]
    adb = load_json(RESULTS / "vrdu_adbuy_lilt" / "metrics.json")["gap"]
    names = ["Ad-Buy forms\n(DeepForm)", "Registration forms\n(FARA)"]
    mtl = [100 * adb["mtl_mean_f1"], 100 * reg["mtl_mean_f1"]]
    utl = [100 * adb["utl_mean_f1"], 100 * reg["utl_mean_f1"]]
    mtl_sd = [100 * adb["mtl_std"], 100 * reg["mtl_std"]]
    utl_sd = [100 * adb["utl_std"], 100 * reg["utl_std"]]
    gaps = [100 * adb["gap_f1"], 100 * reg["gap_f1"]]

    fig, ax = plt.subplots(figsize=(7, 4.4))
    x, w = np.arange(2), 0.34
    err = {"ecolor": INK2, "elinewidth": 1}
    b1 = ax.bar(x - w / 2 - 0.01, mtl, w, yerr=mtl_sd, capsize=4, error_kw=err, color=BLUE, label="MTL (seen)")
    b2 = ax.bar(x + w / 2 + 0.01, utl, w, yerr=utl_sd, capsize=4, error_kw=err, color=ORANGE, label="UTL (unseen)")
    label_bars(ax, b1, errs=mtl_sd)
    label_bars(ax, b2, errs=utl_sd)
    for i, g in enumerate(gaps):
        ax.text(x[i], 108, f"gap {g:.2f}", ha="center", fontsize=10, color=INK, fontweight="bold")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Entity F1 (%), 3 folds x 3 seeds")
    ax.set_title("Same model, same N=200: the gap depends on layout variety")
    ax.legend(loc="lower left")
    save(fig, "fig3_adbuy_vs_registration.png")


# ==============================================================================
# Fig 4: GNN architecture ablations (random document split)
# ==============================================================================
def fig_gnn_ablations():
    configs = [
        ("gnn_ablation_standard_gat", "Standard GAT\n(no edge features)"),
        ("gnn_ablation_full_16d", "Edge-GAT\n16-D edges, 2 layers"),
        ("gnn_ablation_depth_1", "Edge-GAT\n1 layer"),
        ("gnn_ablation_depth_3", "Edge-GAT\n3 layers"),
    ]
    data = [load_json(RESULTS / c / "metrics.json") for c, _ in configs]
    runs = [[100 * r["test"]["f1"] for r in d["runs"]] for d in data]
    means = [100 * d["mean_f1"] for d in data]
    stds = [np.std(r, ddof=1) for r in runs]

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    x = np.arange(len(configs))
    bars = ax.bar(x, means, yerr=stds, width=0.5, capsize=4, color=[AXIS, BLUE, BLUE, BLUE],
                  error_kw={"ecolor": INK2, "elinewidth": 1})
    label_bars(ax, bars, fmt="{:.2f}", errs=stds)
    ax.set_xticks(x, [n for _, n in configs])
    ax.set_ylim(85, 92.5)
    ax.set_ylabel("Entity F1 (%), mean of 2 seeds")
    ax.set_title("Delaunay + Edge-GAT on seen layouts (random document split)")
    save(fig, "fig4_gnn_ablations.png")


# ==============================================================================
# Fig 5: XFUND spatial-feature and head knockouts (link F1)
# ==============================================================================
def fig_xfund_knockouts():
    nb = load_json(ROOT / "notebooks" / "kaggle_ablation_adbuy.ipynb")
    text = ""
    for cell in nb["cells"]:
        for o in cell.get("outputs", []):
            t = "".join(o.get("text", []))
            if "Full 16-D Upgraded Model" in t:
                text = t
    rows = []
    for line in text.splitlines():
        m = re.match(r"\|\s*([^|]+?)\s*\|\s*([\d.]+)%[^|]*\|\s*([\d.]+)%\s*\S+\s*([\d.]+)%", line)
        if m and "Ref" not in m.group(1):
            rows.append((m.group(1), float(m.group(3)), float(m.group(4))))
    if not rows:
        print("[!] XFUND knockout table not found in notebook output; skipping fig5")
        return
    full = rows[0][1]

    fig, ax = plt.subplots(figsize=(8, 4.4))
    y = np.arange(len(rows))
    colors = [BLUE] + [ORANGE if full - r[1] > 5 else "#86b6ef" for r in rows[1:]]
    ax.barh(y, [r[1] for r in rows], xerr=[r[2] for r in rows], height=0.6, color=colors,
            capsize=3, error_kw={"ecolor": INK2, "elinewidth": 1})
    for i, (_, f1, sd) in enumerate(rows):
        txt = f"{f1:.2f}" if i == 0 else f"{f1:.2f}  ({f1 - full:+.2f})"
        ax.annotate(txt, (f1 + sd, i), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9.5, color=INK)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(50, 75)
    ax.set_xlabel("Key-value linking F1 (%), 3 seeds")
    ax.set_title("XFUND: what each spatial feature group and decoding step is worth")
    ax.grid(axis="y", visible=False)
    save(fig, "fig5_xfund_knockouts.png")


# ==============================================================================
# Fig 6: Matched-protocol UTL comparison - LiLT vs GNNs
# ==============================================================================
def gnn_matched(name, key):
    path = RESULTS / name / "metrics.json"
    if path.exists():
        g = load_json(path)["gap"]
        return 100 * g["mtl_mean_f1"], 100 * g["utl_mean_f1"]
    fb = GNN_UTL_FALLBACK[key]
    return fb["mtl"], fb["utl"]


def fig_gnn_utl():
    rc = parse_relative_coordinate_table()["vrdu_track1_lilt"]
    lilt = (pct(rc[0]), pct(rc[1]))
    rsa = gnn_matched("gnn_matched", "rsa_gnn")
    ner = gnn_matched("typed_gnn_matched", "ner_fused")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"width_ratios": [2, 1]})
    w = 0.34
    for ax, names, vals, ylabel in (
        (a1, ["LiLT\n(sliding window)", "RSA-GNN\n(Delaunay + Edge-GAT)"], [lilt, rsa], "Entity F1 (%)"),
        (a2, ["NER-fused\njoint GNN"], [ner], "Edge link F1 (%)"),
    ):
        x = np.arange(len(names))
        b1 = ax.bar(x - w / 2 - 0.01, [v[0] for v in vals], w, color=BLUE, label="MTL (seen)")
        b2 = ax.bar(x + w / 2 + 0.01, [v[1] for v in vals], w, color=ORANGE, label="UTL (unseen)")
        label_bars(ax, b1)
        label_bars(ax, b2)
        for i, (m, u) in enumerate(vals):
            ax.text(x[i], 106, f"-{m - u:.1f} pts", ha="center", fontsize=10, fontweight="bold")
        ax.set_xticks(x, names)
        ax.set_ylim(0, 112)
        ax.set_ylabel(ylabel)
    a2.set_xlim(-0.8, 0.8)
    a1.set_title("Matched N=200, VRDU Registration: graph priors widen the gap")
    fig.legend(*a1.get_legend_handles_labels(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    save(fig, "fig6_gnn_utl_comparison.png")


# ==============================================================================
# Fig 7: Training dynamics - validation cannot see the template shift; curriculum
# ==============================================================================
def fig_training_curves():
    run = load_json(RESULTS / "vrdu_lilt" / "UTL_fold0_seed42" / "result.json")
    xf = load_json(RESULTS / "xfund_joint_lilt" / "metrics.json")["runs"][0]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ep = [h["epoch"] + 1 for h in run["history"]]
    a1.plot(ep, [100 * h["val_f1"] for h in run["history"]], color=BLUE, lw=2, label="Validation F1 (seen templates)")
    a1.axhline(100 * run["test"]["f1"], color=ORANGE, lw=2, ls="--",
               label=f"Test F1 on held-out Amendment ({100 * run['test']['f1']:.1f})")
    a1.set_xlabel("Epoch")
    a1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    a1.set_ylabel("Entity F1 (%)")
    a1.set_ylim(0, 100)
    a1.set_title("Validation never sees the shift")
    a1.legend(loc="lower right")

    hist = xf["history"]
    step = np.arange(1, len(hist) + 1)
    ent = [100 * h.get("val_entity_f1", np.nan) for h in hist]
    lnk = [100 * h["val_linking_f1"] if "val_linking_f1" in h else np.nan for h in hist]
    a2.plot(step, ent, color=BLUE, lw=2, marker="o", ms=4, label="Val entity F1")
    a2.plot(step, lnk, color=AQUA, lw=2, marker="o", ms=4, label="Val linking F1")
    phases = [h["phase"] for h in hist]
    for p in sorted(set(phases)):
        idx = [i + 1 for i, q in enumerate(phases) if q == p]
        a2.axvspan(idx[0] - 0.5, idx[-1] + 0.5, color=["#f0efec", SURFACE, "#f0efec"][(p - 1) % 3], zorder=0)
        a2.text((idx[0] + idx[-1]) / 2, 2, ["tagging", "linking head", "joint"][p - 1],
                ha="center", fontsize=9, color=MUTED)
    a2.set_xlabel("Epoch (across curriculum phases)")
    a2.set_ylabel("F1 (%)")
    a2.set_ylim(0, 100)
    a2.set_title("XFUND 3-phase curriculum")
    a2.legend(loc="center right")
    fig.tight_layout()
    save(fig, "fig7_training_curves.png")


if __name__ == "__main__":
    for fn in (fig_gap_per_template, fig_per_entity, fig_adbuy_vs_registration, fig_gnn_ablations,
               fig_xfund_knockouts, fig_gnn_utl, fig_training_curves):
        fn()
    print(f"[+] Figures written to {OUTPUT_DIR}")
