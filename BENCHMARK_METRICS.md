# Visually-Rich Key-Value Extraction: Living Benchmark Tracker

> [!NOTE]  
> **Living Document Notice:** This file is a centralized, living benchmark tracker for the `KeyValue-Extraction` project (IE 643 Course Project). It documents empirical results across all datasets, training regimes, and architectural configurations. **It is updated continuously as new experiments, feature ablations, and dataset evaluations are completed.**

---

## 1. Executive Summary: The Master Benchmark (Deliverable 4)

To resolve the central research question—*Can geometric graph neural networks close the generalization collapse on unseen form templates?*—we evaluated all three project architectures under a strict template-disjoint 3-fold cross-validation protocol ($N=200$ training documents per fold, zero template leakage verified via `assert_no_template_leakage`).

### Master Head-to-Head Generalization Benchmark (VRDU Registration Forms, Matched $N=200$)

| # | Architecture / Pipeline | Architectural Inductive Bias | Seen (MTL F1) | Unseen (UTL F1) | Generalization Gap ($\Delta$) | Relative Drop (%) | Generalization Diagnosis |
|---|:---|:---|:---:|:---:|:---:|:---:|:---|
| **1** | **Standard LiLT Baseline (Track 1)** | Absolute 2D BBoxes + XLM-RoBERTa backbone | **91.56%** $\pm$ 1.13% | **64.24%** $\pm$ 12.71% | **27.31 points** | **29.8% drop** | Severe Coordinate Memorization; language model prior cushions collapse. |
| **2** | **Standalone RSA-GNN (Track 2)** | Planar Delaunay Triangulation + 16-D Asymmetric Spatial Edges | **85.97%** | **42.05%** | **43.92 points** | **51.1% drop** | **The Spatial Overfitting Dilemma:** Geometric message passing scrambles across layout shifts. |
| **3** | **NER-Fused Joint GNN (Track 3 Link F1)** | 16-D Edges + Discrete 8-D Entity One-Hot Concatenation | **81.40%** | **30.48%** | **50.92 points** | **62.6% drop** | **The Coupling Trap:** Discrete entity tags rigidly coupled with spatial link priors backfire on unseen layouts. |

---

## 2. Track 2: RSA-GNN Architecture & Feature Ablations (Seen / Random Split)

Before the template-disjoint benchmark, controlled architectural ablations were conducted on the Delaunay Visibility Graph and Edge-GAT message passing formulation on VRDU to isolate the impact of graph depth and edge feature representations:

| Experiment / Configuration | Architecture Details | Active Spatial Dims | Mean Test F1 | Std F1 | Result Location | Key Finding |
|:---|:---|:---:|:---:|:---:|:---|:---|
| **Standard GAT Baseline** | 3-Layer GAT (No edge features) | 0-D (Topology only) | **89.01%** | $\pm$ 0.42% | [`results/gnn_ablation_standard_gat/metrics.json`](file:///D:/KeyValue-Extraction/results/gnn_ablation_standard_gat/metrics.json) | Graph attention alone without edge features lacks geometric orientation. |
| **Full 16-D Edge-GAT (Shallow)** | 1-Layer RSA-GNN + 16-D Spatial Edges | 16-D (Full) | **90.12%** | $\pm$ 0.35% | [`results/gnn_ablation_depth_1/metrics.json`](file:///D:/KeyValue-Extraction/results/gnn_ablation_depth_1/metrics.json) | Local 1-hop spatial exchange provides +1.11% boost over standard GAT. |
| **Full 16-D Edge-GAT (Deep)** | 3-Layer RSA-GNN + 16-D Spatial Edges | 16-D (Full) | **90.50%** | $\pm$ 0.28% | [`results/gnn_ablation_depth_3/metrics.json`](file:///D:/KeyValue-Extraction/results/gnn_ablation_depth_3/metrics.json) | **Optimal configuration on seen layouts (+1.49% over GAT).** |
| **Full 16-D Coordinate Feature Set** | 3-Layer RSA-GNN (Alternative Run) | 16-D (Full) | **89.91%** | $\pm$ 0.31% | [`results/gnn_ablation_full_16d/metrics.json`](file:///D:/KeyValue-Extraction/results/gnn_ablation_full_16d/metrics.json) | Robust performance across varied random seed initializations. |

---

## 3. Deep Diagnosis: Why Did the GNNs Suffer a Larger UTL Collapse than LiLT?

A critical, publishable discovery of this project is the **"Spatial Inductive Bias Dilemma" (Layout Overfitting Paradox)**:

```
[Less Geometric Bias] -----------------------------------------------------> [Heavy Geometric Bias]
Standard LiLT (2D BBoxes)           Standalone RSA-GNN (Delaunay 16-D)      NER-Fused GNN (16-D + Tags)
UTL Gap: 27.31 points               UTL Gap: 43.92 points                   UTL Gap: 50.92 points
Relative Drop: 29.8%                Relative Drop: 51.1%                    Relative Drop: 62.6%
```

### The Three Root Mechanisms of the Collapse:

1. **The Language Backbone Safety Net in LiLT:**
   - LiLT processes text sequences through pretrained XLM-RoBERTa self-attention.
   - When a template changes from `Amendment` to `Short-Form`, the text tokens (e.g., `"Date:"`, `"Registrant Name:"`, `"Address:"`) retain their linguistic and lexical salience. Even when coordinates shift unpredictably, semantic similarity allows the model to correctly identify target fields with **64.24% F1**.

2. **Delaunay Graph Instability Under Layout Perturbations (Track 2):**
   - Planar Delaunay Triangulation connects nodes based purely on 2D Euclidean proximity of bounding box centroids.
   - In `Amendment`, target values sit horizontally adjacent to field labels.
   - In `Short-Form` and `Dissemination Report`, the document structure reorganizes into **vertical multi-column grids**, where keys are placed *above* values or separated by wide margins.
   - As a result, the Delaunay triangulation connects completely different neighbor pairs! During message passing, nodes aggregate representations from irrelevant geometric neighbors, poisoning the token classification heads and dropping F1 to **42.05%**.

3. **The Coupling Trap of Pseudo-Linking and Discrete Tags (Track 3):**
   - In Track 3, 8-D discrete entity type one-hot encodings were concatenated with token embeddings, and edge links were trained with directional spatial priors (Key sits to the left of Value).
   - The edge classifier memorized that a valid Key $\rightarrow$ Value link requires a horizontal positive displacement $(\Delta x > 0)$.
   - When evaluated on unseen forms containing vertical table cells or top-down alignments, precision and recall collapsed, resulting in a **30.48% Link F1**.

---

## 4. Cross-Template Generalization on VRDU Subcorpora (Baseline LiLT)

*Protocol:* Matched-size protocol ($N = 200$ training documents per fold), 3 template-disjoint folds $\times$ 3 seeds (`[13, 21, 42]`) = **18 runs per subcorpus**. Backbone: `LiLT-RoBERTa-base`.

### A. Headline Summary & Generalization Gap ($\Delta$)

| Dataset / Subcorpus | Protocol | MTL Mean F1 | UTL Mean F1 | Generalization Gap ($\Delta$) | Relative Drop (%) | Top Fold Checkpoint |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **VRDU Registration Forms** (`FARA`) | Matched ($N=200$) | **87.74%** $\pm$ 1.62% | **66.66%** $\pm$ 7.29% | **+21.08 points** | **24.03%** | 89.92% (MTL fold 0, seed 42) |
| **VRDU Registration Forms (Track 1 Patched)** | Matched ($N=200$) | **91.56%** $\pm$ 1.13% | **64.24%** $\pm$ 12.71% | **+27.31 points** | **29.80%** | 92.40% (MTL fold 1, seed 13) |
| **VRDU Ad-Buy Forms** (`DeepForm`) | Matched ($N=200$) | **92.46%** $\pm$ 0.86% | **90.61%** $\pm$ 1.12% | **+1.85 points** | **2.00%** | **93.79%** (MTL fold 0, seed 42) |
| **VRDU Ad-Buy Forms (Track 1 Patched)** | Matched ($N=200$) | **92.12%** $\pm$ 0.94% | **90.72%** $\pm$ 1.05% | **+1.40 points** | **1.52%** | **93.55%** (MTL fold 0, seed 21) |

---

### B. Per-Entity Breakdown: Why Ad-Buy Forms Generalized but Registration Forms Collapsed

1. **VRDU Registration Forms (FARA):** The held-out templates (`Amendment`, `Dissemination Report`, `Short-Form`) are distinct legal instruments with radically altered field positions.
   - Lexically weak fields (`file_date`, `foreign_principle_name`, `signer_name`) experienced **near-total or total collapse (0.00% to 6.98% F1)**.
   - Only structurally salient numerical fields (`registration_num`: 89.97%) survived the layout shift.

2. **VRDU Ad-Buy Forms (DeepForm):** Target fields carry rigid lexical triggers (`"Gross Total"`, `"Advertiser"`, `"Flight Dates"`) and US broadcast stations share standardized traffic & billing ERP software (WideOrbit, Marketron), bounding the generalization drop to only **1.40% - 1.85%**.

---

## 5. Joint Entity Extraction & Spatial Relation Linking Benchmarks (FUNSD / XFUND)

*Architecture:* `JointFormExtractor` with decoupled 3-stage curriculum (Token Tagging Warmup $\rightarrow$ Frozen Relation Head Warmup $\rightarrow$ Joint Fine-tuning) and 1-to-1 argmax bipartite matching.

| Benchmark Dataset | Languages | Task | Entity F1 | Linking F1 | Loss / Head Config | Output Directory |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **FUNSD** (`funsd_joint_lilt`) | English | Tagging + Linking | **79.10%** $\pm$ 0.79% | **32.05%** $\pm$ 2.74% | 16-D Spatial Vector, Focal Loss ($\gamma=2, \alpha=0.75$) | [`results/funsd_joint_lilt`](file:///D:/KeyValue-Extraction/results/funsd_joint_lilt) |
| **XFUND** (`xfund_joint_lilt`) | 5 Languages (`es`, `fr`, `de`, `it`, `pt`) | Tagging + Linking | **79.09%** $\pm$ 0.21% | **65.51%** $\pm$ 1.26% | 16-D Spatial Vector, 1-to-1 Matching | [`results/xfund_joint_lilt`](file:///D:/KeyValue-Extraction/results/xfund_joint_lilt) |
| **FUNSD (Tagging Only)** | English | Token Tagging | **79.29%** $\pm$ 1.36% | N/A | Standard LiLT Token Classifier | [`results/funsd_lilt`](file:///D:/KeyValue-Extraction/results/funsd_lilt) |
| **Synthetic Forms** | Synthetic | Joint Extraction | **100.00%** | **100.00%** | Sanity verification of relation head & loss | [`results/synthetic_joint_lilt`](file:///D:/KeyValue-Extraction/results/synthetic_joint_lilt) |

---

## 6. Controlled Spatial Feature Ablation Suite (XFUND)

| Experiment / Condition | Active Spatial Dims | Entity F1 | Linking F1 | $\Delta$ vs Full 16-D | Component Role & Key Finding |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Full 16-D Upgraded Model** | **16-D** | **79.32%** $\pm$ 0.36% | **67.21%** $\pm$ 0.83% | **Reference** | Full feature set + Focal Loss + 1-to-1 argmax decoding. |
| **Ablation 1: w/o Line-of-Sight** | 14-D | 78.60% $\pm$ 0.71% | 65.26% $\pm$ 2.11% | **-1.95%** | Ray-casting corridor suppresses cross-column false positives. |
| **Ablation 2: w/o Typography** | 12-D | 79.09% $\pm$ 0.21% | 65.51% $\pm$ 1.26% | **-1.70%** | Font scale log-ratios enable header vs value distinction. |
| **Ablation 3: w/o Containment Overlap** | 11-D | 79.25% $\pm$ 0.36% | 64.57% $\pm$ 1.13% | **-2.64%** | Overlap-over-min preserves multi-line address entity grouping. |
| **Ablation 4: Continuous Geometry Only** | 5-D | 78.87% $\pm$ 0.19% | **57.26%** $\pm$ 1.66% | **-9.95%** | **Severe drop:** polar distance alone lacks document layout semantics. |
| **Ablation 5: BCE Loss (No Focal Loss)** | 16-D | 79.18% $\pm$ 0.34% | 64.76% $\pm$ 0.64% | **-2.45%** | Hard negative mining via focal loss is essential for 99:1 link imbalance. |
| **Ablation 6: Independent Decoding (No 1-1)** | 16-D | 79.32% $\pm$ 0.36% | **57.33%** $\pm$ 0.03% | **-9.88%** | **Severe drop:** independent thresholding causes multiple keys to claim one value. |

---

## 7. Current Technical Issues & Architectural Roadblocks

1. **Rigid Geometric Triangulation Fails Across Modalities:**
   - Planar Delaunay Triangulation hardcodes graph edges based on Euclidean distance rather than semantic compatibility. In vertical forms, keys and values separated by whitespace are left disconnected.
2. **Discrete Entity Coupling Causes Layout Memorization:**
   - Concatenating discrete one-hot tags directly couples the edge classifier to the spatial coordinates of seen forms.
3. **Fixed Global Thresholding in Edge Classification:**
   - Applying `sigmoid(logits) >= 0.5` fails when graph density changes across templates. Dynamic candidate ranking (softmax over incident candidate values per key) is required.

---

## 8. Proposed Architectural Solutions (Next Steps)

1. **Template-Adversarial Representation Learning (DANN-LiLT):**
   - Integrate a Gradient Reversal Layer (GRL) and template discriminator head to penalize the LiLT backbone for encoding template-identifying spatial features, mathematically guaranteeing template-invariant representations.
2. **BROS-Style 2D Relative Spatial Positional Embeddings (2D RoPE):**
   - Replace absolute coordinate bounding boxes $[x_0, y_0, x_1, y_1]$ with pairwise relative spatial displacements $(\Delta x_{ij}, \Delta y_{ij})$ injected directly into the self-attention matrix.
3. **Semantic-Gated Dynamic Graph Adjacency:**
   - Transition from fixed Delaunay graphs to a dense learnable adjacency matrix gated by semantic compatibility:
     $$A_{ij} = \text{Sigmoid}\left(\text{CosineSim}(h_i, h_j) \cdot \text{MLP}(e_{ij}^{\text{spatial}})\right)$$
