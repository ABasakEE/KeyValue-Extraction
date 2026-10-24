# Comprehensive Project Handoff & Technical Roadmap

**Date:** October 24, 2026  
**Repository:** `KeyValue-Extraction` (Branch: `main`)  
**Project:** IE 643 Course Project — Cross-Template Key-Value Field Extraction from Scanned Forms  
**Team:** Unemployed and Unsupervised (Param Mehta 23b2439, Arjoe Basak 23b1295) \cdot IIT Bombay  

---

## 1. Executive Summary & Research Context

Standard Visually-Rich Document Understanding (VrDU) benchmarks (like FUNSD) leak layout templates between training and test sets, masking whether models truly understand relative document topology or merely memorize fixed $(x, y)$ coordinate positions.

This project investigates **zero-same-template cross-template generalization** using:
1. Google's **VRDU benchmark** (KDD 2023) under the **Matched-Size Protocol ($N=200$ training forms)** comparing:
   - **MTL (Mixed Template Learning):** Train and test drawn from the same pool of templates.
   - **UTL (Unseen Template Learning):** Train and test template sets are strictly disjoint.
2. A **16-D Explicit Structural Spatial Relation Vector** (Geometry, Containment, Line-of-Sight, Typography) for bipartite Key-Value linking on **XFUND** and **FUNSD**.

---

## 2. Current Empirical Benchmark Status (Consolidated Numbers)

All experimental logs, histories, and metrics are committed locally under [`results/`](file:///D:/KeyValue-Extraction/results) and documented in [`BENCHMARK_METRICS.md`](file:///D:/KeyValue-Extraction/BENCHMARK_METRICS.md).

### A. Headline Cross-Template Generalization (VRDU Benchmark)

| Benchmark / Subcorpus | Protocol | MTL Mean F1 | UTL Mean F1 | Generalization Gap ($\Delta$) | Relative Drop (%) | Top Checkpoint F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **VRDU Registration Forms** (`FARA`) | Matched ($N=200$) | **87.74%** $\pm$ 1.62% | **66.66%** $\pm$ 7.29% | **+21.08 points** | **24.03%** | 89.92% (MTL fold 0, seed 42) |
| **VRDU Ad-Buy Forms** (`DeepForm`) | Matched ($N=200$) | **92.46%** $\pm$ 0.86% | **90.61%** $\pm$ 1.12% | **+1.85 points** | **2.00%** | **93.79%** (MTL fold 0, seed 42) |

#### Critical Findings from the Comparison:
1. **The 21.08-Point Registration Collapse:**  
   In Registration Forms (FARA), the three held-out templates (`Amendment`, `Dissemination Report`, `Short-Form`) are completely distinct legal documents. When layouts change, models relying on absolute coordinate embeddings fail catastrophically:
   - `file_date`: **0.0% F1** (0/88 detected) across multiple seeds.
   - `foreign_principle_name`: **0.0% F1** (0/15 detected).
   - `signer_name`: **6.98% F1** (3/78 detected).
   - `signer_title`: **25.71% F1**.
2. **The Ad-Buy Invariance (1.85-Point Gap):**  
   In Ad-Buy Forms (DeepForm), cross-template transfer succeeds with minimal loss because:
   - Target entities have high lexical salience (`advertiser`, `agency`, `flight_to` date ranges).
   - US broadcast stations share standard commercial traffic & billing ERP software (WideOrbit, Marketron, OSI), creating a consistent underlying column/tabular format across templates.
   - Preserved checkpoint: [`checkpoints/top_model.pt`](file:///D:/KeyValue-Extraction/checkpoints/top_model.pt) (**93.79% Test F1**).

---

### B. Systematic 6-Condition Feature Ablation Suite (XFUND Multilingual)

Conducted on the 5-language XFUND benchmark (~745 train forms, ~250 test forms) across all seeds. Executed and recorded in [`notebooks/kaggle_ablation_adbuy.ipynb`](file:///D:/KeyValue-Extraction/notebooks/kaggle_ablation_adbuy.ipynb) and [`BENCHMARK_METRICS.md#L75-L93`](file:///D:/KeyValue-Extraction/BENCHMARK_METRICS.md#L75-L93):

| Condition / Knockout | Active Dims | Entity F1 | Linking F1 | $\Delta$ vs Full 16-D | Component Role & Key Finding |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Full 16-D Upgraded Model** | **16-D** | **79.32%** $\pm$ 0.36% | **67.21%** $\pm$ 0.83% | **Reference** | Geometry (5D) + Containment (5D) + LoS (2D) + Typography (4D) + Focal Loss + 1-to-1 Argmax. |
| **Ablation 1: w/o Line-of-Sight** | 14-D | 78.60% $\pm$ 0.71% | 65.26% $\pm$ 2.11% | **-1.95%** | Disabling ray-casting corridor creates false positive links across distant table columns. |
| **Ablation 2: w/o Typography** | 12-D | 79.09% $\pm$ 0.21% | 65.51% $\pm$ 1.26% | **-1.70%** | Removing font scale log-ratios impairs header-vs-value hierarchical distinction. |
| **Ablation 3: w/o Containment Overlap** | 11-D | 79.25% $\pm$ 0.36% | 64.57% $\pm$ 1.13% | **-2.64%** | Removing overlap-over-min damages multi-line address and value entity grouping. |
| **Ablation 4: Continuous Geometry Only** | 5-D | 78.87% $\pm$ 0.19% | **57.26%** $\pm$ 1.66% | **-9.95%** | **Catastrophic drop:** pure polar distance $(\Delta x, \Delta y, d, \theta)$ lacks document inductive bias. |
| **Ablation 5: BCE Loss (No Focal Loss)** | 16-D | 79.18% $\pm$ 0.34% | 64.76% $\pm$ 0.64% | **-2.45%** | Hard negative mining via focal loss ($\gamma=2$) is essential for the 99:1 link class imbalance. |
| **Ablation 6: Independent Decoding (No 1-1)** | 16-D | 79.32% $\pm$ 0.36% | **57.33%** $\pm$ 0.03% | **-9.88%** | **Catastrophic drop:** independent thresholding causes multiple questions to collide over the same value. |

---

### C. Standard Baselines
- **FUNSD Token Tagging Baseline:** **79.29%** $\pm$ 1.36% F1 ([`results/funsd_lilt/metrics.json`](file:///D:/KeyValue-Extraction/results/funsd_lilt/metrics.json)).
- **FUNSD Joint Extraction (12-D Baseline):** Entity F1 **79.10%** $\pm$ 0.79%, Linking F1 **32.05%** $\pm$ 2.74% ([`results/funsd_joint_lilt/metrics.json`](file:///D:/KeyValue-Extraction/results/funsd_joint_lilt/metrics.json)).
- **Synthetic Sanity Verification:** Entity F1 **100.0%**, Linking F1 **100.0%** ([`results/synthetic_joint_lilt/metrics.json`](file:///D:/KeyValue-Extraction/results/synthetic_joint_lilt/metrics.json)).

---

## 3. Core Diagnostic Discoveries (The Bottlenecks to Fix)

1. **The Multi-Page Sequence Truncation Bottleneck:**
   - In [`results/vrdu_lilt/UTL_fold0_seed42/result.json#L6-L7`](file:///D:/KeyValue-Extraction/results/vrdu_lilt/UTL_fold0_seed42/result.json#L6-L7):
     - `support`: 314,090 tokens
     - `truncated_words`: **261,888 tokens (83.3% of the corpus dropped!)**
   - Because VRDU forms are multi-page PDFs (2–6 pages), and `max_length: 512` cuts off after Page 1, signature blocks, execution clauses, and filing dates on Pages 2–3 were never fed to the model.
2. **Absolute Coordinate Memorization:**
   - Standard LiLT/LayoutLM architectures use absolute 2D bounding-box coordinates $(x_0, y_0, x_1, y_1) \in [0, 1000]$.
   - The model learns *spatial rote memory* rather than *structural semantic relationships*, causing the 0.0% F1 collapse on held-out templates.

---

## 4. The Two-Track Implementation Plan (Agreed Roadmap)

To address these setbacks within a 48-hour development cycle, two complementary tracks have been formulated:

### Track 1: Patching the Existing Architecture (LiLT + Bipartite Head)
*Goal: Remove coordinate memorization from the existing pipeline without changing the base model class.*
1. **Relative Coordinate Injection:** Strip out absolute $[0, 1000]$ box coordinates. Compute document/cluster centroid $(\bar{x}, \bar{y})$ and feed normalized relative shifts $(\Delta x, \Delta y)$ from the anchor.
2. **Aggressive Coordinate Jittering:** Randomly translate ($\pm 80$ px), scale ($0.85\times - 1.15\times$), and aspect-stretch document boxes during training. This prevents memorizing spatial positions and forces reliance on relative 16-D features.
3. **Sliding Window Chunking:** Slice documents into 384-token windows with 128-token overlap. Soft-vote prediction distributions across overlapping regions during inference to eliminate the 83.3% truncation drop.

### Track 2: The New Architecture (RSA-GNN with Delaunay Visibility) — Recommended
*Goal: Replace the Euclidean grid with an Attributed Relational Graph.*
1. **Step 1: Offline Feature Caching (`src/cache_features.py`):**
   - Run chunked VRDU forms through the LiLT backbone **once** using PyTorch `DataParallel` on dual T4 GPUs.
   - Cache extracted $(N, 768)$ node embeddings, boxes, labels, and masks to disk (`.pt`).
   - Bypasses transformer forward/backward passes during GNN training, dropping VRAM to <1 GB and accelerating training epochs from 2 minutes to ~3 seconds.
2. **Step 2: Delaunay Visibility Graph Builder (`src/models/delaunay_graph.py`):**
   - Discard basic $k$-NN. Use `scipy.spatial.Delaunay` on token centroids for planar triangulation.
   - Run OpenCV morphological line detection (`cv2.morphologyEx`) on document images to detect visual borders/rules.
   - Prune any Delaunay edge that crosses a visual line, preventing false links across table borders.
   - Assign the 16-D structural vector (Geometry, Containment, LoS, Typography) to every retained edge.
3. **Step 3: Edge-Augmented Graph Attention Network (`src/models/edge_gat.py`):**
   - Train an Edge-Augmented GAT layer (`EdgeGATLayer`) directly on the cached `.pt` tensors.
   - Update node representations by message-passing over relative edges, followed by token classification.

---

## 5. Repository File Map & Navigation

```
KeyValue-Extraction/
├── configs/
│   ├── funsd_joint_lilt.yaml       # FUNSD Joint 16-D config
│   ├── funsd_lilt.yaml             # FUNSD tagging baseline config
│   ├── vrdu_lilt.yaml              # VRDU Registration Forms (Track B1) matched protocol
│   ├── vrdu_adbuy_lilt.yaml        # VRDU Ad-Buy Forms (Track B2) matched protocol
│   └── xfund_joint_lilt.yaml       # XFUND multilingual joint 16-D config
├── src/
│   ├── train.py                    # Master training entrypoint (multi-GPU matched & standard)
│   ├── evaluate.py                 # Strict seqeval IOB2, independent cross-check, linking_f1, generalization_gap
│   ├── data/
│   │   ├── vrdu_loader.py          # VRDU loader (FARA + DeepForm official KDD splits)
│   │   ├── splits.py               # Template-disjoint split construction (assert_no_template_leakage)
│   │   ├── funsd_loader.py         # FUNSD JSON parser & box normalizer
│   │   └── xfund_loader.py         # XFUND multilingual JSON parser
│   └── models/
│       ├── token_clf.py            # TokenClassifier (LiLT, LayoutLMv3, BROS wrappers)
│       ├── joint_extractor.py      # JointFormExtractor (Decoupled 3-phase curriculum)
│       └── linking_head.py         # 16-D SpatialRelationClassifier & compute_spatial_features
├── notebooks/
│   ├── kaggle_ablation_adbuy.ipynb # FULLY EXECUTED master notebook (6 ablations + Ad-Buy run)
│   ├── kaggle_ablation.ipynb       # Ablation runner notebook template
│   └── kaggle_XFUND_run.ipynb      # XFUND runner notebook
├── results/                        # All metrics.json, result.json, history.json files
├── checkpoints/
│   └── top_model.pt                # 93.79% F1 top model weights (ignored by git)
├── BENCHMARK_METRICS.md            # Living benchmark tracker with full comparison tables
├── EXPERIMENT_HANDOFF.md           # High-level technical milestone summary
└── PROJECT_HANDOFF.md              # THIS FILE — complete state transfer document
```

---

## 6. Exact Action Items for the Next Session

When picking up work in the next session, follow these sequential steps:

### Immediate Next Step 1: Implement Track 1 Multi-Page Sliding Window
1. Open [`src/data/vrdu_loader.py`](file:///D:/KeyValue-Extraction/src/data/vrdu_loader.py).
2. Implement chunking of `VrduExample` into windows of 384 words with 128 overlap.
3. Update prediction aggregation in [`src/train.py`](file:///D:/KeyValue-Extraction/src/train.py) so multiple windows for the same `guid` are soft-voted before computing F1.
4. Verify locally using `tests/test_train_pipeline.py`.

### Immediate Next Step 2: Implement Track 2 GNN Modules
1. Create `src/models/delaunay_graph.py`:
   - Function `build_delaunay_graph(boxes, image=None)` using `scipy.spatial.Delaunay`.
   - Line intersection check using `cv2.findContours` or bounding line segments.
   - Vectorized call to `compute_spatial_features` from [`src/models/linking_head.py`](file:///D:/KeyValue-Extraction/src/models/linking_head.py#L150) to produce edge feature tensor $(E, 16)$.
2. Create `src/models/edge_gat.py`:
   - Class `EdgeGATLayer(nn.Module)` with multi-head attention conditioned on 16-D edges.
   - Class `GNNTokenClassifier(nn.Module)` wrapping the GAT layers and classification head.
3. Create `src/cache_features.py`:
   - Script taking a config (e.g. `vrdu_lilt.yaml`), running the LiLT encoder once, and saving tensors to `feature_cache/`.

### Immediate Next Step 3: Build Master Kaggle Evaluation Notebook
1. Create `notebooks/kaggle_gnn_evaluation.ipynb` with sections for:
   - Git pull & environment check.
   - Track 1 execution (LiLT + Jitter + Sliding Window).
   - Track 2 execution (Feature Caching $\rightarrow$ Delaunay Graph $\rightarrow$ Edge-GAT Training).
   - Comparative evaluation script outputting before/after delta tables.
