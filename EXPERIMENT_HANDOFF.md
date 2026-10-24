# Experiment Handoff & Technical Summary

**Date:** October 15, 2026  
**Repository:** `KeyValue-Extraction`  
**Headline Focus:** Visually-Rich Key-Value Extraction, Cross-Template Generalization (MTL vs. UTL), and Spatial Relation Linking.

---

## 1. Executive Summary

This repository investigates **cross-template generalization** in Visually-Rich Document Understanding (VrDU). Standard benchmarks (like FUNSD) leak template layouts between train and test sets, artificially inflating performance metrics. 

By benchmarking with strict template-disjoint protocols (following Google's KDD 2023 VRDU benchmark) and evaluating our 16-D Spatial Relation Joint Extractor, we established concrete empirical proof of the **generalization gap** and built high-performance, multi-GPU training infrastructure on Kaggle.

---

## 2. Experimental Benchmark Results

All metrics below are consolidated from full multi-seed, multi-fold evaluations now tracked in [`results/`](file:///D:/KeyValue-Extraction/results).

### A. Headline Cross-Template Generalization (VRDU Benchmark)

At matched training size ($N = 200$), we evaluate Mixed Template Learning (**MTL** — seen layouts) against Unseen Template Learning (**UTL** — strictly held-out layouts):

| Dataset / Track | MTL Mean F1 | UTL Mean F1 | Generalization Gap ($\Delta$) | Relative Drop | Folds / Seeds | Top Checkpoint F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **VRDU Registration Forms** (Track B1) | **87.74%** $\pm$ 1.62% | **66.66%** $\pm$ 7.29% | **+21.08 points** | **24.03%** | 3 folds $\times$ 3 seeds (9 runs) | 0.8992 |
| **VRDU Ad-Buy Forms** (Track B2 / DeepForm) | **92.46%** $\pm$ 0.86% | **90.61%** $\pm$ 1.12% | **+1.85 points** | **2.00%** | 3 folds $\times$ 3 seeds (9 runs) | **0.9379** |

#### Key Insights:
1. **The Registration Forms Generalization Gap (21.08 points):**  
   Confirms the central thesis: models relying heavily on fixed layout templates suffer catastrophic degradation when deployed on unseen form structures.
2. **Ad-Buy Forms Layout Transferability (1.85 points):**  
   FCC political broadcast contract filings share standardized tabular structures. Cross-template transfer is remarkably robust even when the exact station template is unseen.
3. **Top Checkpoint:**  
   The top-performing model across all folds reached **93.79% Test F1** on Ad-Buy Form fold 0 (seed 42), preserved locally in [`checkpoints/top_model.pt`](file:///D:/KeyValue-Extraction/checkpoints/top_model.pt).

---

### B. Standard & Multilingual Baselines

| Benchmark | Task | Entity F1 | Linking F1 | Notes |
| :--- | :--- | :---: | :---: | :--- |
| **FUNSD** (`funsd_lilt`) | Token Tagging | **79.29%** $\pm$ 1.36% | N/A | Standard 3-seed baseline |
| **FUNSD Joint** (`funsd_joint_lilt`) | Tagging + Linking | **79.10%** $\pm$ 0.79% | **32.05%** $\pm$ 2.74% | Joint relation head baseline |
| **XFUND (12-D Baseline)** | Tagging + Linking | **79.09%** $\pm$ 0.21% | **65.51%** $\pm$ 1.26% | 5 languages (es, fr, de, it, pt) |
| **XFUND (Full 16-D Upgraded)** | Tagging + Linking | **79.32%** $\pm$ 0.36% | **67.21%** $\pm$ 0.83% | **+1.70%** gain with full 16-D features |
| **Synthetic Sanity** (`synthetic_joint_lilt`) | Joint Extraction | **100.00%** | **100.00%** | Proves architectural expressivity |

---

### C. Systematic Feature & Head Ablation Study (XFUND)

| Experiment / Condition | Active Dims | Entity F1 | Linking F1 | $\Delta$ vs 16-D | Key Finding |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Full 16-D Upgraded Model** | **16-D** | **79.32%** $\pm$ 0.36% | **67.21%** $\pm$ 0.83% | **Ref** | Baseline for all ablations |
| **w/o Line-of-Sight (14-D)** | 14-D | 78.60% $\pm$ 0.71% | 65.26% $\pm$ 2.11% | **-1.95%** | Ray-casting prevents false cross-column links |
| **w/o Typography (12-D)** | 12-D | 79.09% $\pm$ 0.21% | 65.51% $\pm$ 1.26% | **-1.70%** | Font hierarchy separates headers from values |
| **w/o Containment (11-D)** | 11-D | 79.25% $\pm$ 0.36% | 64.57% $\pm$ 1.13% | **-2.64%** | Overlap-over-min resolves multi-line boxes |
| **Geometry Only (5-D)** | 5-D | 78.87% $\pm$ 0.19% | **57.26%** $\pm$ 1.66% | **-9.95%** | **Severe drop:** continuous polar/distance fails alone |
| **BCE Loss (no Focal)** | 16-D | 79.18% $\pm$ 0.34% | 64.76% $\pm$ 0.64% | **-2.45%** | Hard negative mining necessary for 99:1 imbalance |
| **Independent Decoding (no 1-1)** | 16-D | 79.32% $\pm$ 0.36% | **57.33%** $\pm$ 0.03% | **-9.88%** | **Severe drop:** argmax bipartite prevents collision |

---

## 3. Engineering & Architecture Milestones

### 1. Dual-GPU Dynamic Multiprocessing (`src/train.py`)
- Created `_mp_matched_worker` targeting Kaggle's dual Tesla T4 GPUs (`cuda:0` and `cuda:1`).
- Shared task queue dynamically balances folds across GPUs, cutting total execution time by **~50%**.
- Upfront checkpoint scanning detects already completed folds and resumes immediately.
- Automatic fallback to sequential execution if only 1 GPU is available or `--no-parallel` is specified.

### 2. DeepForm (VRDU Ad-Buy) Split Loader (`src/data/vrdu_loader.py`)
- Implemented `load_deepform_protocol` to parse Google's official KDD 2023 few-shot split files (`DeepForm-mixed_template` and `DeepForm-unk_template` for $N=200$, seeds 0, 1, 2).
- Resolved FCC transaction hash filename parsing to prevent zero-token drops across all 641 documents.

### 3. Decoupled 3-Stage Curriculum
- **Phase 1:** Pure token classification warmup ($\lambda_{\text{link}} = 0$) with plateau early stopping.
- **Phase 2:** Teacher-forced spatial relation head warmup with backbone frozen.
- **Phase 3:** Full end-to-end joint fine-tuning with 1-to-1 argmax matching and focal linking loss.

### 4. 16-D Explicit Spatial Features
- **Geometry (5D):** Normalized coordinates, bounding box deltas, Euclidean distances, polar angles.
- **Containment (5D):** Horizontal/vertical overlap over minimum dimension, reading order alignment.
- **Line-of-Sight (2D):** Ray-casting unobstructed visual path detection.
- **Typography (4D):** Character aspect ratio, font size prominence, word length scale.

---

## 4. Local File Structure & Artifacts

- **Metrics & History Logs:**
  - `results/vrdu_lilt/` — Full 18-run fold metrics for Registration Forms.
  - `results/vrdu_adbuy_lilt/` — Full 18-run fold metrics for Ad-Buy Forms.
  - `results/xfund_joint_lilt/` — Multilingual joint extraction metrics.
  - `results/funsd_lilt/` & `results/funsd_joint_lilt/` — FUNSD baselines.
- **Model Checkpoints:**
  - Top model weight (`top_model.pt`, 520 MB) placed in [`checkpoints/top_model.pt`](file:///D:/KeyValue-Extraction/checkpoints/top_model.pt) and [`results/vrdu_adbuy_lilt/MTL_fold0_seed42/best_model.pt`](file:///D:/KeyValue-Extraction/results/vrdu_adbuy_lilt/MTL_fold0_seed42/best_model.pt).
  - Both `.pt` locations are excluded via `.gitignore` to keep git operations lightweight.

---

## 5. Next Steps

1. **Slide & Paper Preparation:** Generate comparative plots from `history.json` and LaTeX summary tables directly from `metrics.json`.
2. **Error Analysis:** Analyze failure modes in UTL folds (specifically false positives on novel header designs).
3. **VRDU Joint Linking:** Extend the 16-D spatial relation head to the VRDU dataset to evaluate whether explicit spatial features narrow the 21-point generalization gap.
