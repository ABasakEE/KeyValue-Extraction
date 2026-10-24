# Visually-Rich Key-Value Extraction: Living Benchmark Tracker

> [!NOTE]  
> **Living Document Notice:** This file is a centralized, living benchmark tracker for the `KeyValue-Extraction` project. It documents empirical results across all datasets, training regimes, and architectural configurations. **It is updated continuously as new experiments, feature ablations, and dataset evaluations are completed.**

---

## 1. Overview & Experimental Tracks

This project investigates two core questions in Visually-Rich Document Understanding (VrDU):
1. **Cross-Template Generalization:** Quantifying performance degradation when models transition from seen templates (**MTL** — Mixed Template Learning) to strictly unseen layouts (**UTL** — Unseen Template Learning) at identical training sample size ($N=200$).
2. **Explicit Spatial Relation Linking:** Evaluating a 16-D continuous spatial feature vector (geometry, directional containment, line-of-sight visibility, typography) for bipartite Key-Value linking.

---

## 2. Track B: Cross-Template Generalization (VRDU Benchmark)

*Protocol:* Matched-size protocol ($N = 200$ training documents per fold), 3 template-disjoint folds $\times$ 3 seeds (`[13, 21, 42]`) = **18 runs per subcorpus**. Backbone: `LiLT-RoBERTa-base`.

### A. Headline Summary & Generalization Gap ($\Delta$)

| Dataset / Subcorpus | Protocol | MTL Mean F1 | UTL Mean F1 | Generalization Gap ($\Delta$) | Relative Drop (%) | Top Fold Checkpoint |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **VRDU Registration Forms** (`FARA`) | Matched ($N=200$) | **87.74%** $\pm$ 1.62% | **66.66%** $\pm$ 7.29% | **+21.08 points** | **24.03%** | 89.92% (MTL fold 0, seed 42) |
| **VRDU Ad-Buy Forms** (`DeepForm`) | Matched ($N=200$) | **92.46%** $\pm$ 0.86% | **90.61%** $\pm$ 1.12% | **+1.85 points** | **2.00%** | **93.79%** (MTL fold 0, seed 42) |

---

### B. Per-Entity Dissection: Why the Drop Differs Radically

The contrast between Registration Forms (21.08-point collapse) and Ad-Buy Forms (1.85-point minor drop) reveals how document structure affects layout overfitting:

#### 1. VRDU Registration Forms: Severe Layout Overfitting
In Registration Forms (FARA), the held-out templates (`Amendment`, `Dissemination Report`, `Short-Form`) are **entirely different legal instruments**. When target fields migrate to novel coordinate areas, models relying on absolute 2D coordinate embeddings collapse on key fields:

| Entity Class | MTL F1 | UTL F1 (Held-out) | Drop ($\Delta$) | Support (Tokens) | Failure Diagnosis |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `registration_num` | 0.9718 | 0.8997 | -0.0721 | 360 | Salient numerical pattern; partial spatial shift tolerance. |
| `registrant_name` | 0.8916 | 0.7539 | -0.1377 | 376 | Partial semantic identification from organization suffixes. |
| `signer_title` | 0.6667 | 0.2571 | **-0.4096** | 50 | Heavy shift from table header to signature block. |
| `signer_name` | 0.7797 | 0.0698 | **-0.7099** | 78 | Location moves from body to execution block; near-total collapse. |
| `file_date` | 0.6792 | **0.0000** | **-0.6792** | 88 | Date formats share format with other dates; **complete collapse** when layout changes. |
| `foreign_principle_name` | 0.7143 | **0.0000** | **-0.7143** | 15 | Highly template-dependent field placement; **complete collapse**. |

#### 2. VRDU Ad-Buy Forms: High Semantic Salience & Software Standardization
In Ad-Buy Forms (DeepForm), target entities carry high lexical salience and US television stations share standard traffic & billing ERP software (WideOrbit, Marketron, OSI):

| Entity Class | MTL F1 | UTL F1 (Held-out) | Drop ($\Delta$) | Support (Tokens) | Robustness Rationale |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `flight_to` | 0.9718 | 0.9615 | -0.0103 | 128 | Strict date pattern preceded by standard trigger keywords. |
| `flight_from` | 0.9421 | 0.9237 | -0.0184 | 130 | Standard date format preceded by trigger keywords. |
| `advertiser` | 0.9562 | 0.9484 | -0.0078 | 155 | Strong lexical signals ("PAC", "Campaign", "for Congress"). |
| `agency` | 0.9579 | 0.9516 | -0.0063 | 63 | Corporate naming suffixes ("Media LLC", "Group Inc."). |
| `contract_num` | 0.9499 | 0.9251 | -0.0248 | 154 | Distinct 5–8 digit numerical structure. |
| `tv_address` | 0.9286 | 0.9132 | -0.0154 | 131 | Typical address token patterns. |
| `property` | 0.9240 | 0.9039 | -0.0201 | 251 | Broadcast call signs ("WNBC", "KABC"). |
| `product` | 0.9217 | 0.8940 | -0.0277 | 150 | Domain-specific naming conventions. |
| `gross_amount` | 0.8920 | 0.8805 | -0.0115 | 84 | Currency symbols and decimal formatting adjacent to total headers. |

---

## 3. Joint Entity Extraction & Spatial Relation Linking Benchmarks

*Architecture:* `JointFormExtractor` with decoupled 3-stage curriculum (Token Tagging Warmup $\rightarrow$ Frozen Relation Head Warmup $\rightarrow$ Joint Fine-tuning) and 1-to-1 argmax bipartite matching.

| Benchmark Dataset | Languages | Task | Entity F1 | Linking F1 | Loss / Head Config | Output Directory |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **FUNSD** (`funsd_joint_lilt`) | English | Tagging + Linking | **79.10%** $\pm$ 0.79% | **32.05%** $\pm$ 2.74% | 16-D Spatial Vector, Focal Loss ($\gamma=2, \alpha=0.75$) | [`results/funsd_joint_lilt`](file:///D:/KeyValue-Extraction/results/funsd_joint_lilt) |
| **XFUND** (`xfund_joint_lilt`) | 5 Languages (`es`, `fr`, `de`, `it`, `pt`) | Tagging + Linking | **79.09%** $\pm$ 0.21% | **65.51%** $\pm$ 1.26% | 16-D Spatial Vector, 1-to-1 Matching | [`results/xfund_joint_lilt`](file:///D:/KeyValue-Extraction/results/xfund_joint_lilt) |
| **FUNSD (Tagging Only)** | English | Token Tagging | **79.29%** $\pm$ 1.36% | N/A | Standard LiLT Token Classifier | [`results/funsd_lilt`](file:///D:/KeyValue-Extraction/results/funsd_lilt) |
| **Synthetic Forms** | Synthetic | Joint Extraction | **100.00%** | **100.00%** | Sanity verification of relation head & loss | [`results/synthetic_joint_lilt`](file:///D:/KeyValue-Extraction/results/synthetic_joint_lilt) |

---

## 4. Systematic Ablation Study (XFUND Multilingual Key-Value Extraction)

To isolate the exact contribution of each spatial feature group, loss formulation, and decoding constraint, a 6-condition controlled ablation suite was executed across all seeds.

| Experiment / Condition | Active Spatial Dims | Entity F1 | Linking F1 | $\Delta$ vs Full 16-D | Component Role & Key Finding |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Full 16-D Upgraded Model** | **16-D** | **79.32%** $\pm$ 0.36% | **67.21%** $\pm$ 0.83% | **Reference** | Full feature set + Focal Loss + 1-to-1 argmax decoding. |
| **Ablation 1: w/o Line-of-Sight** | 14-D | 78.60% $\pm$ 0.71% | 65.26% $\pm$ 2.11% | **-1.95%** | Disabling ray-casting corridor creates false positive links across distant table columns. |
| **Ablation 2: w/o Typography** | 12-D | 79.09% $\pm$ 0.21% | 65.51% $\pm$ 1.26% | **-1.70%** | Removing font scale log-ratios impairs header-vs-value hierarchical distinction. |
| **Ablation 3: w/o Containment Overlap** | 11-D | 79.25% $\pm$ 0.36% | 64.57% $\pm$ 1.13% | **-2.64%** | Removing overlap-over-min damages multi-line address and value entity grouping. |
| **Ablation 4: Continuous Geometry Only** | 5-D | 78.87% $\pm$ 0.19% | **57.26%** $\pm$ 1.66% | **-9.95%** | **Catastrophic drop:** pure polar distance $(\Delta x, \Delta y, d, \cos\theta, \sin\theta)$ lacks document inductive bias. |
| **Ablation 5: BCE Loss (No Focal Loss)** | 16-D | 79.18% $\pm$ 0.34% | 64.76% $\pm$ 0.64% | **-2.45%** | Hard negative mining via focal loss ($\gamma=2$) is essential for the 99:1 link class imbalance. |
| **Ablation 6: Independent Decoding (No 1-1)** | 16-D | 79.32% $\pm$ 0.36% | **57.33%** $\pm$ 0.03% | **-9.88%** | **Catastrophic drop:** independent thresholding causes multiple questions to collide over the same value. |

---

## 5. Implementation Status of 16-D Spatial Relation Head

The 16-D Spatial Feature Vector computes pairwise geometric and structural relationships:
- **Geometry (5D):** Normalized coordinates, box deltas, Euclidean center distances, polar angles.
- **Containment (5D):** Horizontal & vertical overlap over minimum dimension, reading order alignment.
- **Line-of-Sight (2D):** Ray-casting unobstructed visual path detection.
- **Typography (4D):** Character aspect ratio, font size prominence, word length scale.

| Dataset | Contains Ground-Truth K-V Links? | 16-D Head Evaluated? | Status & Next Action |
| :--- | :---: | :---: | :--- |
| **FUNSD** | Yes (`question` $\rightarrow$ `answer`) | **Yes** | Complete (Entity F1 79.10%, Link F1 32.05%) |
| **XFUND** | Yes (`question` $\rightarrow$ `answer`) | **Yes** | Complete (Entity F1 79.09%, Link F1 65.51%) |
| **Synthetic Forms** | Yes (Engineered links) | **Yes** | Complete (100% verification) |
| **VRDU Registration Forms** | No (Extractive NER only) | **No** (Token Tagging only) | Potential for pseudo-linking (pairing prompts to values) |
| **VRDU Ad-Buy Forms** | No (Extractive NER only) | **No** (Token Tagging only) | Potential for pseudo-linking (pairing headers to values) |
| **CORD** | Hierarchical groupings | **No** | Planned receipt hierarchy linking |
| **SROIE** | 4 slot-filling fields | **No** | Planned prompt-to-value matching |

---

## 6. Model Checkpoints & Storage Policy

- **Weight Checkpoints:** Model weights (`*.pt`) are preserved locally and excluded from git tracking via [`.gitignore`](file:///D:/KeyValue-Extraction/.gitignore).
- **Top Checkpoint:**
  - File: [`checkpoints/top_model.pt`](file:///D:/KeyValue-Extraction/checkpoints/top_model.pt) (and duplicate at [`results/vrdu_adbuy_lilt/MTL_fold0_seed42/best_model.pt`](file:///D:/KeyValue-Extraction/results/vrdu_adbuy_lilt/MTL_fold0_seed42/best_model.pt))
  - Performance: **93.79% Test F1** on VRDU Ad-Buy Forms.
- **Metrics Logs:** All per-epoch histories (`history.json`), per-seed results (`result.json`), and summary statistics (`metrics.json`) are committed directly to [`results/`](file:///D:/KeyValue-Extraction/results).

---

## 7. Planned Experiment Changelog & Next Horizons

- [x] Establish dual-GPU matched protocol for VRDU Ad-Buy Forms (Kaggle 2 $\times$ T4).
- [x] Evaluate cross-template generalization on FARA Registration Forms (21.08 pt drop confirmed).
- [x] Evaluate cross-template generalization on DeepForm Ad-Buy Forms (1.85 pt drop confirmed).
- [x] Evaluate 16-D spatial relation head on multilingual XFUND (67.21% link F1).
- [x] **Ablation Study Matrix:** Completed all 6 feature knockouts (Line-of-Sight, Containment, Typography, Focal vs. BCE Loss, 1-to-1 Matching).
- [ ] **Multi-Page Sliding Window Loader:** Eliminate sequence truncation (currently 83.3% of tokens truncated on multi-page forms).
- [ ] **VRDU Pseudo-Linking / Relational Attention Adapter:** Inject 16-D relative spatial bias to narrow the 21-point Registration Form gap.
- [ ] **Evaluation on CORD / SROIE:** Expanding spatial linking to scanned commercial receipts.
