"""Regenerate every poster plot from the result files under results/.

    python poster/scripts/make_plots.py

Prints each plotted value next to its source so the poster numbers can be checked.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S  # noqa: E402

R = S.ROOT / "results"


def load(rel):
    return json.loads((R / rel).read_text(encoding="utf-8"))


def log(name, value, src):
    print(f"  {name:<44} {value:>8}   <- {src}")


# --------------------------------------------------------------------------- F1
def vrdu_templates():
    """Unseen-template exact-span F1 on the confirm set: BROS vs reader vs router."""
    src_m = "schema_guided_kv_v2/vsS25/metrics_V2A2.json"
    src_h = "schema_guided_kv_v2/hybrid_V2A2.json"
    m, h = load(src_m), load(src_h)
    keys = [("fold0_Amendment", "Amendment", "Amendment"),
            ("fold1_Dissemination_Report", "Dissemination_Report", "Dissem.\nReport"),
            ("fold2_Short-Form", "Short-Form", "Short-Form")]
    bros, reader, router, labels = [], [], [], []
    for fk, hk, lab in keys:
        f = m["folds"][fk]["f1"]
        bros.append(100 * (f["bros_seed13"] + f["bros_seed42"]) / 2)
        reader.append(100 * f["vlm_V2A2"])
        router.append(100 * h["primary"][hk]["f1"])
        labels.append(lab)
    p = m["pooled"]["f1"]
    bros.append(100 * (p["bros_seed13"] + p["bros_seed42"]) / 2)
    reader.append(100 * p["vlm_V2A2"])
    router.append(100 * h["primary"]["_pooled"]["f1"])
    labels.append("Pooled")
    print("F1 VRDU unseen templates (confirm set)")
    for lab, b, r, ro in zip(labels, bros, reader, router):
        log(f"{lab.replace(chr(10), ' ')}: BROS / reader / router", f"{b:.1f}/{r:.1f}/{ro:.1f}", f"{src_m}, {src_h}")

    fig, ax = plt.subplots(figsize=(S.COL_W, 4.1))
    x = np.arange(len(labels))
    w = 0.26
    series = [(bros, "BROS Track-1 (trained)", S.GREY), (reader, "Zero-shot reader", S.ACCENT),
              (router, "Router: reader + BROS", S.ACCENT_LIGHT)]
    for i, (vals, name, c) in enumerate(series):
        bars = ax.bar(x + (i - 1) * w, vals, w, color=c, label=name, linewidth=0)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=S.FONT_SIZE - 4)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Exact-span F1")
    ax.set_ylim(0, 100)
    ax.axvline(2.5, color=S.GREY, lw=0.8, ls=(0, (2, 2)))
    ax.legend(loc="upper left", ncol=3, bbox_to_anchor=(0, 1.16), handlelength=1.0, columnspacing=1.0)
    S.save(fig, "f1_vrdu_templates")


# --------------------------------------------------------------------------- F2
def fatura_crossover():
    src = "fatura_sg/run/report.json"
    r = load(src)
    ks = [1, 2, 5, 10, 40]
    bros = [r["per_k"][str(k)]["bros_f1"] for k in ks]
    reader = r["per_k"]["1"]["A2"]["f1"]
    print("F2 FATURA crossover")
    for k, b in zip(ks, bros):
        log(f"K={k}: BROS F1, reader minus BROS", f"{b} {r['per_k'][str(k)]['A2']['minus_bros']:+}", src)
    log("reader (zero-shot, no training)", reader, src)

    fig, ax = plt.subplots(figsize=(S.HALF_W, 3.0))
    ax.plot(ks, bros, color=S.GREY_DARK, marker="o", ms=7, lw=2)
    ax.axhline(reader, color=S.ACCENT, lw=2.5)
    ax.text(6.5, reader - 3, f"reader {reader}", color=S.ACCENT, fontsize=S.FONT_SIZE - 2, va="top")
    ax.text(ks[0] * 1.15, bros[0] - 2, "BROS", color=S.GREY_DARK, fontsize=S.FONT_SIZE - 2, va="center")
    ax.set_xscale("log")
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.minorticks_off()
    ax.set_xlabel("Templates seen in training (K)")
    ax.set_ylabel("Field F1 (unseen)")
    ax.set_ylim(25, 105)
    S.save(fig, "f2_fatura_crossover")


# --------------------------------------------------------------------------- F3
def xfund_links():
    rows = [("LiLT-en", "table_links/A0_e2e.json", S.GREY_DARK),
            ("LiLT-XLM", None, S.GREY),
            ("+ crops", "table_links/A2_e2e.json", S.ACCENT_LIGHT),
            ("BROS joint", "bros_joint/J1_e2e.json", S.ACCENT)]
    print("F3 XFUND end-to-end link F1 (windowed, 250 val forms)")
    names, means, seeds, colors = [], [], [], []
    for name, src, c in rows:
        if src is None:
            src = "xfund_e2e_xlm/summary.json"
            j = load(src)
            mean = 100 * j["cells"]["all|windowed|link"]["windows"]
            sd = []  # per-seed absolute values are not stored in the summary, only deltas
        else:
            j = load(src)
            sd = [100 * j["seeds"][s]["all|windowed|link"]["f1"] for s in j["seeds"]]
            mean = float(np.mean(sd))
        log(name, f"{mean:.2f}", src)
        names.append(name)
        means.append(mean)
        seeds.append(sd)
        colors.append(c)

    fig, ax = plt.subplots(figsize=(S.HALF_W, 3.0))
    y = np.arange(len(names))[::-1]
    ax.barh(y, means, color=colors, height=0.62, linewidth=0)
    for yi, m, sd in zip(y, means, seeds):
        if sd:
            ax.plot(sd, [yi] * len(sd), "o", color="white", ms=5, mec=S.INK, mew=0.8)
        ax.text(max([m] + sd) + 0.6, yi, f"{m:.2f}", va="center", fontsize=S.FONT_SIZE - 3)
    ax.set_yticks(y, names)
    ax.set_xlim(30, 62)
    ax.set_xlabel("End-to-end link F1")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    S.save(fig, "f3_xfund_links")


# --------------------------------------------------------------------------- F4
def docile_curve():
    src = "docile_bros/run/report.json"
    r = load(src)
    sizes = [50, 200, 1000]
    mean = [r["per_size"][str(n)]["row_f1"] for n in sizes]
    per_seed = {n: [v["row_f1"] for k, v in r["runs"].items() if k.startswith(f"N{n}_")] for n in sizes}
    full = [load(f"docile_bros_full/run/report_{s}.json")["runs"][f"Nfull_T{s}"]["row_f1"] for s in (13, 21)]
    mtl = [load(f"docile_bros_full/run/report_mtl_{s}.json")["runs"][f"Nfull_T{s}"]["row_f1"] for s in (13, 21)]
    n_full = 2908  # whole training pool (STUDIES.md, Study 41)
    print("F4 DocILE line-item row F1, unseen layouts")
    for n, m in zip(sizes, mean):
        log(f"N={n} mean (seeds {per_seed[n]})", m, src)
    log(f"whole pool N={n_full} seeds 13/21", f"{full}", "docile_bros_full/run/report_{13,21}.json")
    log("seen layouts, whole pool, seeds 13/21", f"{mtl} mean {np.mean(mtl):.2f}", "docile_bros_full/run/report_mtl_*.json")

    fig, ax = plt.subplots(figsize=(S.HALF_W, 3.0))
    xs = sizes + [n_full]
    ys = mean + [float(np.mean(full))]
    ax.plot(xs, ys, color=S.ACCENT, marker="o", ms=7, lw=2.5, label="unseen layouts")
    for n in sizes:
        ax.plot([n] * len(per_seed[n]), per_seed[n], "o", color=S.ACCENT, alpha=0.35, ms=5, mew=0)
    ax.plot([n_full] * 2, full, "o", color=S.ACCENT, alpha=0.35, ms=5, mew=0)
    ax.plot([n_full], [np.mean(mtl)], marker="s", color=S.GREY_DARK, ms=8, ls="none", label="seen layouts")
    ax.plot([n_full] * 2, mtl, "s", color=S.GREY_DARK, alpha=0.35, ms=5, mew=0)
    ax.set_xscale("log")
    ax.set_xticks(xs, ["50", "200", "1000", "all\n(2908)"])
    ax.minorticks_off()
    ax.set_xlabel("Training documents")
    ax.set_ylabel("Line-item row F1")
    ax.set_ylim(35, 75)
    ax.legend(loc="upper left", handlelength=1.0)
    S.save(fig, "f4_docile_curve")


# --------------------------------------------------------------------------- F5
def docile_taxonomy():
    src = "docile_taxonomy/taxonomy.json"
    t = load(src)
    groups = [("Right", ["right"], S.ACCENT),
              ("Missed", ["missed_field_absent_in_doc", "missed_split_gold_cell", "missed_other"], S.GREY_DARK),
              ("Wrong field", ["wrong_field_money", "wrong_field_other"], S.GREY),
              ("Wrong extent", ["extent_too_short", "extent_too_long", "extent_other_overlap", "wrong_field_and_extent"], "#D9D9D9")]

    def budget(runs):
        return [float(np.mean([sum(runs[r][k] for k in keys) for r in runs])) for _, keys, _ in groups]

    unseen = budget({k: t["loss_budget_utl"][k] for k in ("Nfull_T13", "Nfull_T21")})
    seen = budget({k: t["loss_budget_mtl"][k] for k in ("Nfull_T13", "Nfull_T21")})
    all5 = budget({k: v for k, v in t["loss_budget_utl"].items() if k != "mean"})
    print("F5 DocILE share of gold spans (%)")
    for (g, _, _), u, s, a in zip(groups, unseen, seen, all5):
        log(f"{g}: unseen / seen (pool runs) | unseen 5 runs", f"{u:.1f}/{s:.1f}|{a:.1f}", src)

    fig, ax = plt.subplots(figsize=(S.COL_W, 1.9))
    for yi, vals in ((1, unseen), (0, seen)):
        left = 0
        for (g, _, c), v in zip(groups, vals):
            ax.barh(yi, v, left=left, color=c, height=0.6, linewidth=0)
            if v > 4:
                ax.text(left + v / 2, yi, f"{v:.0f}", ha="center", va="center", fontsize=S.FONT_SIZE - 3,
                        color="white" if c in (S.ACCENT, S.GREY_DARK) else S.INK)
            left += v
    ax.set_yticks([1, 0], ["unseen layouts", "seen layouts"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of gold line-item cells (%)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [Rectangle((0, 0), 1, 1, color=c) for _, _, c in groups]
    ax.legend(handles, [g for g, _, _ in groups], ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0),
              handlelength=1.0, columnspacing=1.2)
    S.save(fig, "f5_docile_taxonomy")


if __name__ == "__main__":
    S.apply()
    vrdu_templates()
    fatura_crossover()
    xfund_links()
    docile_curve()
    docile_taxonomy()
