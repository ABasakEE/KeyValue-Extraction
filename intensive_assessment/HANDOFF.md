# Intensive Midterm Assessment Handoff Guide

> **Project Title:** Visually-Rich Key-Value Extraction Under Zero-Same-Template Training Conditions  
> **Course / Benchmark:** IE 643 Course Project — Google VRDU, FUNSD, and XFUND Evaluations  
> **Repository:** `https://github.com/ABasakEE/KeyValue-Extraction.git`  
> **Active Branch:** `main` (Latest commit: `7e8f5d4`)  

---

## 1. Executive Summary & Core Research Story

This repository contains the complete codebase, trained model checkpoints, and empirical evaluation logs for investigating **Unseen Template Learning (UTL)** in Visually-Rich Document Understanding (VrDU).

### The Central Research Question
> *"When a document understanding model is deployed on completely unseen form templates under zero-same-template training conditions, does introducing geometric Graph Neural Networks (GNNs) with relative spatial coordinates close the generalization collapse observed in coordinate-based Transformers?"*

### The Headline Finding: The "Spatial Inductive Bias Dilemma"
Under a strict, template-disjoint 3-fold cross-validation protocol ($N=200$ matched training sample size, with zero validation/test leakage mathematically verified via `assert_no_template_leakage`):
1. **Standard LiLT Baseline (Track 1):** Drops from **91.56% (Seen)** to **64.24% (Unseen)** — a **27.31-point / 29.8% relative collapse**. The pretrained language model backbone (XLM-RoBERTa) cushions the drop because lexical tokens remain semantically meaningful.
2. **Standalone RSA-GNN (Track 2):** Drops from **85.97% (Seen)** to **42.05% (Unseen)** — a **43.92-point / 51.1% relative collapse**. Rigid planar Delaunay triangulation hardcodes local spatial edges based on Euclidean distance. When layout shifts from horizontal rows to vertical table grids, message passing scrambles across incorrect neighbors.
3. **NER-Fused Joint GNN (Track 3):** Drops from **81.40% (Seen)** to **30.48% (Unseen)** — a **50.92-point / 62.6% relative collapse**. Discrete entity one-hot tags coupled with spatial link priors overfit to template-specific coordinate co-occurrences.

**Conclusion:** Explicit geometric graph priors create brittle layout dependencies. In unseen templates, layout perturbations disrupt graph topology, whereas transformer self-attention over text tokens provides greater robustness to template shifts.

---

## 2. Metric Source Registry (Where to Pull Data for Charts & Visuals)

The next agent or evaluator can pull exact, verified JSON metrics and logs from the following canonical files:

| Figure / Chart to Produce | Exact Metric File Path | Key Data Fields | Description |
|:---|:---|:---|:---|
| **Master Generalization Gap (Bar Chart)** | [`BENCHMARK_METRICS.md`](file:///D:/KeyValue-Extraction/BENCHMARK_METRICS.md#L18-L30) and [`results/RelativeCoordinate_ablation.txt`](file:///D:/KeyValue-Extraction/results/RelativeCoordinate_ablation.txt) | `Seen (MTL F1)`, `Unseen (UTL F1)`, `Gap (Δ)`, `Relative Drop` | Compares Baseline LiLT (91.56% vs 64.24%), Standalone RSA-GNN (85.97% vs 42.05%), and NER-Fused GNN (81.40% vs 30.48%). |
| **Track 2 GNN Architecture Ablations** | [`results/gnn_ablation_standard_gat/metrics.json`](file:///D:/KeyValue-Extraction/results/gnn_ablation_standard_gat/metrics.json)<br>[`results/gnn_ablation_full_16d/metrics.json`](file:///D:/KeyValue-Extraction/results/gnn_ablation_full_16d/metrics.json)<br>[`results/gnn_ablation_depth_1/metrics.json`](file:///D:/KeyValue-Extraction/results/gnn_ablation_depth_1/metrics.json)<br>[`results/gnn_ablation_depth_3/metrics.json`](file:///D:/KeyValue-Extraction/results/gnn_ablation_depth_3/metrics.json) | `mean_f1`, `std_f1`, `runs` | Standard GAT (89.01%) $\rightarrow$ Full 16-D (89.91%) $\rightarrow$ Depth 1 (90.12%) $\rightarrow$ Depth 3 (90.50%). Demonstrates +1.49% gain on seen layouts. |
| **FARA Registration Forms vs DeepForm Ad-Buy** | [`results/RelativeCoordinate_ablation.txt`](file:///D:/KeyValue-Extraction/results/RelativeCoordinate_ablation.txt) | `MTL F1`, `UTL F1`, `Gap (Δ)` | Ad-Buy: 92.12% vs 90.72% (Gap 1.40%). Registration Forms: 91.56% vs 64.24% (Gap 27.31%). Proves why radical structural shifts cause collapse while standardized layouts do not. |
| **Per-Entity Breakdown (FARA Legal Collapse)** | [`BENCHMARK_METRICS.md`](file:///D:/KeyValue-Extraction/BENCHMARK_METRICS.md#L42-L58) | `MTL F1`, `UTL F1`, `Drop` | Total collapse of `file_date` (0.00%) and `foreign_principle_name` (0.00%) vs resilience of `registration_num` (89.97%). |
| **16-D Spatial Feature Ablation (Radar/Bar Chart)** | [`BENCHMARK_METRICS.md`](file:///D:/KeyValue-Extraction/BENCHMARK_METRICS.md#L90-L108) | `Active Spatial Dims`, `Linking F1`, `Δ vs Full 16-D` | Full 16-D (67.21%) vs w/o Line-of-Sight (-1.95%) vs w/o Typography (-1.70%) vs w/o Containment (-2.64%) vs Continuous Geometry Only (-9.95%). |
| **Multilingual Generalization (FUNSD vs XFUND)** | [`results/funsd_joint_lilt/metrics.json`](file:///D:/KeyValue-Extraction/results/funsd_joint_lilt/metrics.json)<br>[`results/xfund_joint_lilt/metrics.json`](file:///D:/KeyValue-Extraction/results/xfund_joint_lilt/metrics.json) | `entity_f1`, `linking_f1` | FUNSD: 79.10% entity / 32.05% link. XFUND: 79.09% entity / 65.51% link. |
| **Trained Checkpoint Weights** | [`checkpoints/best_gnn_model.pt`](file:///D:/KeyValue-Extraction/checkpoints/best_gnn_model.pt)<br>[`checkpoints/best_typed_gnn.pt`](file:///D:/KeyValue-Extraction/checkpoints/best_typed_gnn.pt)<br>[`checkpoints/top_model.pt`](file:///D:/KeyValue-Extraction/checkpoints/top_model.pt) | State dict tensors, layer shapes | Checkpoints ready for live inference and demonstration. |

---

## 3. Deliverable A: Midterm Presentation Slides (Deck Outline)

A 10-slide high-impact structure designed for an academic / technical panel:

* **Slide 1: Title & Motivation**
  - *Title:* Generalization Collapse in Visually-Rich Key-Value Extraction Under Zero-Same-Template Conditions.
  - *Subtitle:* A Rigorous Empirical Study of Transformers, Relative Coordinates, and Delaunay GNNs.
  - *Presenter:* Team KeyValue-Extraction (IE 643).
  - *Hook:* "Most document AI models boast 90%+ accuracy in research papers. What happens when you deploy them on a form template they have never seen before?"

* **Slide 2: The Core Problem — Unseen Template Learning (UTL)**
  - Difference between **MTL (Mixed Template Learning)** (random document shuffle, seen layouts) and **UTL (Unseen Template Learning)** (template-disjoint holdout).
  - The zero-template leakage contract: Strict training, validation, and test disjointness ($N=200$ matched).

* **Slide 3: Baseline Discovery — The 27.31-Point Generalization Collapse**
  - Show side-by-side: VRDU Ad-Buy Forms (1.40% gap) vs VRDU Registration Forms (27.31% gap).
  - Diagnostic: Why Ad-Buy generalized (standardized ERP billing layouts) while legal registration forms collapsed (radical layout shifts across `Amendment`, `Short-Form`, `Dissemination Report`).
  - Per-entity collapse: Total failure of `file_date` (0.00%) and `foreign_principle_name` (0.00%).

* **Slide 4: The Proposed Hypothesis — Can GNNs Close the Gap?**
  - Why hypothesize GNNs? Transformers memorize absolute coordinate grid $[0, 1000]^4$.
  - Our approach: Coordinate-free document understanding via **Attributed Relational Graphs (ARG)**:
    - Planar Delaunay Visibility Triangulation.
    - 16-D Continuous Spatial Edge Features (Geometry, Containment, Line-of-Sight, Typography).
    - Multi-layer Edge-GAT with asymmetric spatial message passing.

* **Slide 5: Track 2 Architectural Ablations (Seen Templates)**
  - Standard GAT (89.01%) $\rightarrow$ 16-D Edge-GAT (89.91%) $\rightarrow$ 3-Layer Deep GNN (90.50%).
  - Demonstrates that on seen templates, 16-D spatial edges and deep message passing improve local context aggregation.

* **Slide 6: The Master Benchmark (Deliverable 4 Headline Results)**
  - Display the Master Comparison Table (LiLT Baseline vs Standalone RSA-GNN vs NER-Fused Joint GNN).
  - Highlight the shocking empirical finding:
    - LiLT Baseline: 27.31% drop
    - Standalone RSA-GNN: 43.92% drop
    - NER-Fused Joint GNN: 50.92% drop

* **Slide 7: Deep Failure Analysis — The "Spatial Inductive Bias Dilemma"**
  - Visual diagram showing how Delaunay triangulation rearranges edges when moving from horizontal rows to vertical tables.
  - Why Language Models win under layout shift: Text tokens retain semantic identity; geometric graphs scramble their neighborhood.

* **Slide 8: Relation Linking & Feature Knockout Ablation (XFUND)**
  - 6-condition ablation chart: Line-of-Sight (-1.95%), Typography (-1.70%), Containment (-2.64%), Pure Geometry (-9.95%), 1-to-1 argmax matching (-9.88%).

* **Slide 9: Path Forward — Next Architectural Horizons**
  - Solution 1: **Template-Adversarial Representation Learning (DANN-LiLT)** — Gradient reversal layer to mathematically enforce template-invariant representations.
  - Solution 2: **BROS 2D Relative Positional Attention (2D RoPE)** — Relative spatial coordinates in self-attention rather than absolute boxes.
  - Solution 3: **Semantic-Gated Dynamic Adjacency** — Replacing rigid Euclidean Delaunay graphs with learned semantic affinity.

* **Slide 10: Conclusion & Deliverables Summary**
  - Summary of contributions: Strict matched benchmark protocol, open-source reproducible codebase, dual-GPU training harness, identification of the spatial inductive bias dilemma.

---

## 4. Deliverable B: Midterm Project Report (IEEE/ACL Template Outline)

1. **Title & Abstract:** Formulates the zero-same-template problem, states the matched $N=200$ protocol, summarizes the 27.31 to 50.92 point generalization drops, and outlines the spatial overfitting dilemma.
2. **Introduction:** Real-world motivation (invoices, tax forms, FARA filings). Limitations of random-split benchmarks.
3. **Related Work:** Pretrained Document Transformers (LayoutLM, LiLT, BROS), Graph Neural Networks for Document IE (GraphIE, Edge-GAT), Benchmark Protocols (VRDU, FUNSD, XFUND).
4. **Methodology:**
   - Pretrained Multimodal Backbone (LiLT).
   - Planar Delaunay Visibility Graph Construction & Line Pruning.
   - 16-D Asymmetric Spatial Feature Formulation.
   - Dual-Head Edge-GAT with Focal Loss and 1-to-1 Bipartite Argmax Matching.
5. **Experimental Setup:**
   - Strict Matched-Size Protocol ($N=200$, 3-fold template disjoint, mathematical leakage assertion).
   - Dual-GPU Acceleration Pipeline (Kaggle T4 $\times$ 2 sharding).
6. **Results & Ablation Studies:**
   - Deliverable 4: Master Generalization Benchmark Table.
   - Deliverable 5: Per-entity breakdown and error taxonomy.
   - Track 2 GNN Architecture Ablation Table (Depth 1 vs 3, standard GAT vs 16-D Edge-GAT).
   - Multilingual 16-D Spatial Feature Knockout Table.
7. **Discussion: The Spatial Inductive Bias Dilemma:** Theoretical and empirical analysis of why rigid geometric graphs amplify layout overfitting.
8. **Future Work & Conclusion:** Adversarial domain adaptation (DANN), 2D RoPE, and dynamic soft adjacency.

---

## 5. Deliverable C: Code Walkthrough Video Guide (Script & Flow)

* **Target Duration:** 5–7 Minutes.
* **Format:** Screen capture of IDE / VS Code with voiceover.

### Step-by-Step Walkthrough Flow:
1. **0:00 - 1:00 | Project Structure & Configuration (`configs/`):**
   - Show [`configs/vrdu_track1_lilt.yaml`](file:///D:/KeyValue-Extraction/configs/vrdu_track1_lilt.yaml).
   - Explain hyperparameters: sliding window (384 window, 128 overlap), coordinate jittering, relative coordinates, batch size, and seed list (`[13, 21, 42]`).
2. **1:00 - 2:30 | Graph Construction & 16-D Spatial Features (`src/models/`):**
   - Open [`src/models/delaunay_graph.py`](file:///D:/KeyValue-Extraction/src/models/delaunay_graph.py).
   - Walk through `build_delaunay_graph()`: Delaunay triangulation via `scipy.spatial.Delaunay`, OpenCV morphological line extraction, ray-segment intersection pruning, and 16-D feature calculation (Geometry, Containment, Line-of-Sight, Typography).
   - Open [`src/models/edge_gat.py`](file:///D:/KeyValue-Extraction/src/models/edge_gat.py): Show how asymmetric edge attributes modulate attention logits: $\alpha_{ij} = \text{LeakyReLU}(a^T [Wh_i \parallel Wh_j \parallel W_e e_{ij}])$.
3. **2:30 - 4:00 | Strict Matched Protocol & Zero-Leakage Proof (`src/data/`):**
   - Open [`src/data/splits.py`](file:///D:/KeyValue-Extraction/src/data/splits.py).
   - Walk through `matched_size_protocol()`: Explain how sample size is clamped to $N=200$ across both MTL and UTL to eliminate dataset size bias.
   - Walk through `assert_no_template_leakage()`: Show the set-disjointness assertions between `train_templates` and `test_templates`.
4. **4:00 - 5:30 | Dual-GPU Training Harness (`src/`):**
   - Open [`src/train_gnn_matched.py`](file:///D:/KeyValue-Extraction/src/train_gnn_matched.py) and [`src/train_typed_gnn_matched.py`](file:///D:/KeyValue-Extraction/src/train_typed_gnn_matched.py).
   - Show the `ThreadPoolExecutor` worker mapping: Worker 0 runs Seed 13 on `cuda:0`, Worker 1 runs Seed 42 on `cuda:1` concurrently.
   - Show batch collation: `collate_disjoint_graphs()` handling disjoint node/edge offsets.
5. **5:30 - 6:30 | Master Benchmark Notebook (`notebooks/`):**
   - Open [`notebooks/kaggle_gnn_utl_benchmark.ipynb`](file:///D:/KeyValue-Extraction/notebooks/kaggle_gnn_utl_benchmark.ipynb).
   - Show concurrent feature caching (Cell 8) launching `cache_features.py` and `cache_fused_features.py` simultaneously.
   - Show final summary table generation formatting the Master Deliverable 4 table.

---

## 6. Deliverable D: Code Demo Video Guide (Live Run & Inference)

* **Target Duration:** 3–5 Minutes.
* **Format:** Live terminal or Kaggle notebook screen recording demonstrating active execution.

### Live Demo Flow:
1. **0:00 - 0:45 | Environment & GPU Setup:**
   - Run `nvidia-smi` to display Dual NVIDIA Tesla T4 GPUs.
   - Verify repo branch: `git status` on branch `main`.
2. **0:45 - 2:00 | Live Feature Caching & Training Demonstration:**
   - Execute Step 2 benchmark command:
     ```bash
     python src/train_gnn_matched.py --cache-dir feature_cache/registration-form --output-dir results/gnn_matched --seeds 13,42 --epochs 20 --n-train 200
     ```
   - Highlight the console output:
     - `Preloading graph tensors into memory...`
     - `PROTOCOL VERIFIED: Matched N=200 across MTL and UTL`
     - Concurrent worker logs streaming from `[cuda:0 | MTL/fold0/seed13]` and `[cuda:1 | MTL/fold0/seed42]`.
3. **2:00 - 3:00 | Inspecting Checkpoint Inference:**
   - Load [`checkpoints/best_gnn_model.pt`](file:///D:/KeyValue-Extraction/checkpoints/best_gnn_model.pt) via python interactive shell:
     ```python
     import torch
     ckpt = torch.load("checkpoints/best_gnn_model.pt", map_location="cpu")
     print(f"Loaded {len(ckpt)} model weight tensors successfully!")
     ```
   - Run inference on a sample document graph tensor and print predicted entity labels.
4. **3:00 - 3:45 | Output Artifacts & Deliverables:**
   - Display the generated Deliverable 4 summary table and `metrics.json` outputs in `results/`.
   - Conclude with the diagnostic summary.

---

## 7. Python Figure Generator (`generate_presentation_figures.py`)

A standalone script is provided in [`intensive_assessment/generate_presentation_figures.py`](file:///D:/KeyValue-Extraction/intensive_assessment/generate_presentation_figures.py) that reads all committed metric files and outputs high-resolution PNG charts directly into `intensive_assessment/figures/` for use in slides and reports:
1. `fig1_master_generalization_gap.png` (Deliverable 4 Head-to-Head Comparison)
2. `fig2_gnn_architecture_ablations.png` (Track 2 Depth & Edge-GAT ablations)
3. `fig3_adbuy_vs_registration_gap.png` (Structural comparison between subcorpora)
4. `fig4_spatial_feature_knockouts.png` (16-D feature group impact)
