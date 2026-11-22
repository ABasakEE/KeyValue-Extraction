# Best models, seeds, weights and how to reproduce the report figures and the demo

Written 2026-11-22 from the study log (`STUDIES.md`). A model is listed for the demo only when its metric is good enough to show without apology; everything else is reported as a result, not demonstrated.

## 1. Which model to show for which kind of document

| Format | Model (config) | Best run to use | Metric (our protocol) | Demo? | Weights |
|---|---|---|---|---|---|
| Forms, entity boxes and field names (VRDU registration) | BROS Track-1 (`configs/vrdu_track1_bros_demo.yaml`) | MTL fold 0 and UTL fold 0 (`Amendment` held out), seed 13 | MTL 92.4, UTL 71.3 (3 folds, 3 seeds); fold 0: 92.2 / 83.4 | **Yes** | **New run needed:** `notebooks/kaggle_demo_weights.ipynb` ("Save & Run All"), download `results/vrdu_track1_bros_demo/{MTL,UTL}_fold0_seed13/best_model.pt` from the notebook Output tab (about 500 MB each) |
| Forms, zero-shot recipe (Qwen3-VL-8B, grounded answers) | schema-guided recipe, prompt v2 | none to train | 76.5 / 75.2 exact-span F1 on confirm / dev, against BROS 70.5 / 69.8 | Show saved outputs | Nothing to download from us; a live demo needs the Hub model (4-bit, about 6 GB) and a GPU |
| Key-value linking (XFUND) | BROS joint tagger + biaffine linker, arm J1 (`configs/xfund_joint_bros_J1.yaml`) | **seed 42** (link F1 54.01 windowed; seeds 13 / 21: 53.10 / 52.93) | entity 80.2, link 53.35, link_qa 59.6 (3 seeds); LiLT A0: 75.8 / 47.4 / 53.2 | **Yes** (needs a small change in `src/demo.py`, which hard-codes a LiLT backbone) | Probably on Kaggle: notebook `kaggle_xfund_bros_joint`, Output tab, `results/bros_joint/J1/seed_42/best_model.pt` (training saved weights by default; not verified). Fallback: the local LiLT joint checkpoints `results/xfund_joint_16d/seed_21`, `results/xfund_joint_lilt` already used by `demo.py` |
| Tables, line items (DocILE) | BROS line-item tagger + anchor-rule row grouping, full pool | seed 21 (UTL 60.96 / MTL 63.02) or seed 13 (59.84 / 68.87) | UTL 60.4, MTL 66.0 (2 seeds); published unseen-layout baseline 59.8 | **Yes, as a replay of saved predictions** | None: `results/docile_bros_full/run/Nfull_T*/seed_*/test_entities.jsonl` + the local DocILE PDFs; figures by `scripts/make_example_figures.py` |

## 2. Models that are results, not demos (metric too low or negative)

Zero-shot VLM linking on XFUND (link F1 3 to 24, study 27); the recipe on DocILE tables (pilot row F1 26.6, study 38); the detector-and-typer on DocILE (-7.45 row F1, study 43); the LoRA-tuned 3B VLM (46.2 against 70.1 for BROS, study 29); LiLT variants (BROS beats them in every comparison); the graph and adversarial variants and the GNN models; the column rules (study 40). LayoutLMv3 (study 42): pixels help unseen templates by about 3 points but it is 14-17 points below BROS Track-1 on the same folds, so it is a result, not a demo model.

## 3. Reproduce the figures without any weights

`python scripts/make_example_figures.py` writes `figures/example_{table,form}_{good,failure}.png` from saved predictions: DocILE line items (BROS full pool seed 21, boxes coloured by field, numbered rows next to the gold) and VRDU registration forms (BROS Track-1, unseen templates, green = exactly right, red = wrong or extra, blue dashed = missed). Needs the local `data/docile` and `data/vrdu` folders and `pymupdf`.
Key-to-value arrows come from the Gradio linking demo (`python demo.py`) and need a joint-model checkpoint (the local LiLT one works now; BROS J1 is better).

## 4. Reproduce the headline numbers (seeds fixed)

| Result | Command or notebook | Seeds |
|---|---|---|
| BROS Track-1 on VRDU registration forms, MTL / UTL | `python src/train.py --config configs/vrdu_track1_bros.yaml --protocol matched` | 13, 21, 42 |
| BROS joint on XFUND (arms J1, J2) | `notebooks/kaggle_xfund_bros_joint.ipynb` | 13, 21, 42 |
| BROS line items on DocILE, N = 10 to 1000 | `notebooks/kaggle_docile_bros.ipynb` | split seeds 0, 1, 2, training seed 13 |
| BROS line items on DocILE, whole pool, UTL and MTL | `notebooks/kaggle_docile_bros_full.ipynb` (`TRAIN_SEEDS`) | 13, 21 |
| LayoutLMv3 with and without pixels, UTL folds 0 and 1 | `notebooks/kaggle_layoutlmv3.ipynb` (`REGIMES="UTL"`, `FOLDS`, `ARMS`) | 13, 42 |
| Detector and typer on DocILE | `notebooks/kaggle_docile_typer.ipynb` | detector runs `Nfull_T21`, `N1000_S0` |
| DocILE failure taxonomy | `python scripts/docile_failure_taxonomy.py` (CPU, 7 processes) | five saved runs |
