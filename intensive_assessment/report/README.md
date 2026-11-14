# Intensive assessment report (draft)

## Build the PDF on Overleaf

1. Zip this folder (`report.tex` + `figures/`).
2. Overleaf → New Project → Upload Project → choose the zip.
3. Menu → Compiler: pdfLaTeX, Main document: `report.tex`. Recompile.
4. Download the PDF. Only PDF is accepted for submission.

## Before submitting

- [ ] **Rewrite every paragraph in your own words.** The course runs a plagiarism and AI-content check, and a report that "simply contains AI generated content" scores zero. Keep the numbers, tables and figures; rewrite the prose.
- [ ] Confirm the division of work in the Workflow section (marked `\todo`). Grading is individual.
- [ ] Add a screenshot of the running Gradio demo to the Interface section (marked `\todo`).
- [ ] RSA-GNN (85.97 → 42.05) and NER-fused GNN (81.40 → 30.48) come from `BENCHMARK_METRICS.md`; their `metrics.json` files are not in the repo yet. When Arjoe commits `results/gnn_matched/` and `results/typed_gnn_matched/`, rerun `python intensive_assessment/generate_presentation_figures.py` and check the numbers in the Graph models subsection.
- [ ] Check the access dates in the website references.
- [ ] The `\todo` macro prints red text; search the PDF for "TODO" before submitting.

## Where the numbers come from

| Number | Source |
|---|---|
| LiLT registration 87.74 / 66.66, per template, per field | `results/vrdu_lilt/` |
| LiLT Ad-Buy 92.46 / 90.61 | `results/vrdu_adbuy_lilt/metrics.json` |
| LiLT with windows + anchoring 91.56 / 64.24 | `results/RelativeCoordinate_ablation.txt` |
| FUNSD 79.29; joint 79.10 / 32.05 | `results/funsd_lilt/`, `results/funsd_joint_lilt/` |
| XFUND 79.09 / 65.51 (12-D) | `results/xfund_joint_lilt/metrics.json` |
| XFUND knockouts (67.21 etc.) | saved output of `notebooks/kaggle_ablation_adbuy.ipynb` |
| Edge-GAT ablations 89.01–90.50 | `results/gnn_ablation_*/metrics.json` (random split) |
| GNN matched UTL | `BENCHMARK_METRICS.md` (result files pending) |
| Error taxonomy 97.2% | prep deck, `slides/figures/error_taxonomy.png` |

Figures are produced by `intensive_assessment/generate_presentation_figures.py` (fig1–7) and `intensive_assessment/render_template_examples.py` (fig8).
