# NOVELTY ASSESSMENT & PROJECT HANDOFF

**Project:** Cross-template key-value extraction and linking from scanned forms (IE 643 course project, IIT Bombay)
**Branch:** `feat/bros-relative-layout` · **Written:** 2026-11-04 · **Source of truth for every number below:** [`STUDIES.md`](STUDIES.md) (Studies P and 1-23) and the `results/` folders it cites.

This document is meant to let a new person (or a new session) pick the project up cold. Read sections 1-3 for what the problem is and what we found, section 5 for where we are stuck, and sections 7-8 for what to do next. Section 11 lists what has *not* been verified.

---

## 0. One-paragraph summary

We set out to close the gap between how well document models extract key-value fields from form templates they have **seen** and templates they have **never seen**. A strong layout-aware tagger (Track-1 BROS) reaches 92.4 F1 on seen templates but only 71.3 on unseen ones on VRDU; the loss is concentrated in two fields (`foreign_principle_name`, and `registrant_name` on the Short-Form template). Three graph-neural-network ideas (a standalone GNN, an NER-fused GNN, and a graph bias inside BROS attention) did not beat the plain BROS tagger, and one of them was never validly tested. A zero-shot vision-language model (VLM) solves the fields BROS cannot but is worse overall, and a hybrid that routes per field between the two gains **+3.8 F1 on VRDU** (confirmed on fresh documents) but that gain comes almost entirely from one field and does not generalize to a second dataset where BROS already saturates. In the second half of the project we turned to key-value **linking** on XFUND and found that the reported link F1 (65.5) was inflated by two evaluation artefacts; the honest end-to-end figure with a model that finds its own entities is about **47.7**, and the remaining errors are structural (downward and heading-to-label links), not close-neighbour geometry mistakes. The next open questions are whether a multilingual text encoder, a BROS linking head, or a stronger VLM moves the numbers that cheap decoding changes (worth about +2 to +3) cannot.

---

## 1. Problem statement (where we started)

**Goal.** Extract key-value fields from scanned forms and generalize to **form templates with zero same-template training data** - no labelled or unlabelled data from held-out templates at any point.

**Why it is a real problem.** Entity-level extraction has been measured under template shift (VRDU, DocILE, Do-GOOD). Key-value **linking** has not: every strong linking number in the literature (for example GeoLayoutLM's 80.35 -> 89.45 relation-extraction jump on FUNSD) comes from a split where train and test share templates. The project's framing is to measure both entity extraction and linking under one template-disjoint protocol and to identify which structural assumptions break.

**Primary benchmark protocol (VRDU `registration-form`, Matched-Size).**
- Equal training budget of **N = 200 documents** in both regimes, so the two are directly comparable:
  - **MTL (Mixed Template Learning):** train and test drawn from the same pool of templates.
  - **UTL (Unseen Template Learning):** train and test template sets strictly disjoint.
- Three registration templates: `Amendment`, `Dissemination_Report`, `Short-Form`. Fields include `registrant_name`, `foreign_principle_name`, `signer_name`, `signer_title`, `file_date`, `registration_num`.
- Metric: entity-level strict span F1 (seqeval-compatible). Gap = MTL F1 - UTL F1.

**Secondary datasets we brought in** (all under a held-out-template idea where possible):

| Dataset | Used for | Notes |
|---|---|---|
| VRDU (registration forms, ad-buy) | main benchmark; zero-shot VLM; hybrid | 3 registration templates; ad-buy line items excluded from gold |
| FUNSD | pipeline sanity; linking | 149 train / 50 test forms; tiny |
| XFUND (es, fr, de, it, pt) | multilingual linking | 745 train / 250 val forms; see section 4.4 |
| FATURA | second dataset for the hybrid | 50 labelled invoice templates, 9 core fields, official unseen-template folds (Study 13 audit) |

**Compute.** Kaggle 2 x Tesla T4 (16 GB each); the 20 GB `/kaggle/working` disk limit once crashed runs that saved thirty 500 MB checkpoints. Local machine: CPU only (i5-1235U, 15.7 GB RAM; see section 9 for the memory pitfalls).

---

## 2. Approaches and results before the study log (the "P" row and Studies 1-5)

All numbers: VRDU registration-form, matched N = 200 (source: `NOVELTY.md`, `BENCHMARK_METRICS.md`, `results/*/metrics.json`).

| # | Model | MTL F1 | UTL F1 | Gap | What it taught us |
|---|---|---|---|---|---|
| 1 | LiLT baseline (512-token truncation) | 87.74 | 66.66 | 21.1 | the reference point; later Track-1 LiLT 91.56 / 64.24 (gap 27.3) |
| 2 | Standalone RSA-GNN (Delaunay graph over frozen encoder) | 85.97 | 42.05 | 43.9 | graph **outside** the transformer loses pretrained semantics and memorizes reading direction |
| 3 | BROS raw (512-token truncation) | 22.70 +/- 17.0 | 7.48 +/- 13.1 | - | VRDU forms are 1,200-2,500 words; truncation discards over 75% of the document |
| 4 | **Track-1 BROS** (384-word sliding windows + soft voting + coordinate jitter + content-box anchoring) | **92.36 +/- 0.3** | **71.27 +/- 11.2** | **21.1** | best tagger so far; BROS's pairwise relative position encoding beats LiLT's absolute coordinates by +7.0 UTL |
| 5 | DANN-LiLT / DANN-BROS (template-adversarial) | 91.12 / 91.80 | 64.79 / 69.59 | 26.3 / 22.2 | adversarial template regularization did **not** beat plain Track-1 BROS |
| 6 | NER-fused joint GNN (link F1, **not** entity F1) | 81.40 | 30.48 | 50.9 | see Study 5: the labels, not the model, drove this drop |
| 7 | Graph-BROS v1 (LoRA + frozen base + graph bias in attention) | 26.67 +/- 30.8 | 39.79 +/- 12.4 | - | collapsed: four concrete bugs (below) |

**Graph-BROS v1 post-mortem (`NOVELTY.md` section 5):** a -17.1 attention logit shock on every head at step 0 (softplus gate x log(1e-9)); 99.73% of the model frozen so multi-token BIO state could not propagate (30.6% malformed tags); unweighted focal loss gave a 3-4 epoch `val_f1 = 0` dead zone; early stopping counted those zero epochs against patience. A v2 with the fixes (commit `5eeaaa4`) was designed and **reviewed as not worth running** (Study 1) and never run.

**The design idea behind the GNN work (still the project's original hypothesis).** Inject a direction-agnostic beta-skeleton (Gabriel) graph into BROS self-attention so that attention is pruned to topologically adjacent tokens, with isotropic edge features (log-radial distance, log area ratio, y-containment). The argument: BROS's signed relative positions still over-fit to reading direction, and dense O(L^2) attention leaks across columns. This was never validated; see section 5.

---

## 3. The study log in one table (Studies 1-23)

Full detail, intervals and limits: [`STUDIES.md`](STUDIES.md). GPU = Kaggle T4 x2; CPU = local.

| # | Study | Type | Main finding |
|---|---|---|---|
| 1 | Graph-BROS v2 design review | CPU | not worth running: edge features broken, half the gates frozen, baseline mis-stated |
| 2 | Per-entity UTL failure analysis | CPU | the gap is two fields: `foreign_principle_name` (0-13% everywhere) and `registrant_name` on Short-Form (2-6%) |
| 3 | VRDU template inspection | CPU | Short-Form has a decoy name block; each template words `foreign_principle_name` differently |
| 4 | BROS error dump | GPU | 95% of wrong Short-Form registrants are the item-1 filer; `foreign_principle_name` is never predicted |
| 5 | NER-fused GNN audit | CPU | the "NER" is a regex; the KV link labels are a template-specific reading-order heuristic; semantic matching was **never tested** |
| 6 | Zero-shot v1 (page 1, VLM vs text LM) | GPU | Qwen2.5-VL-3B beats the text LM; key-to-field naming is the bottleneck (value recall 98%, F1 2.2 on Short-Form `foreign_principle_name`) |
| 7 | Zero-shot v2 (whole document + LM naming) | GPU | naming fixes wording (Short-Form `foreign_principle_name` 4 -> 85) but lowers precision |
| 8 | One-answer-per-field decoding | CPU | raises F1 on every template; Short-Form decoy solved (filer picked 86 -> 0 docs) |
| 9 | Stage B: grounded matched VLM vs BROS | CPU | under strict matching BROS wins overall (pooled delta -6 to -15); **errors are complementary**: VLM wins role/wording fields, BROS wins signature/date fields and span boundaries |
| 10 | Hybrid feasibility | CPU | oracle ceiling +16 F1, but neither pre-registered rule passes |
| 11 | Mode-free VLM mapping, hybrids re-tested | CPU | static per-field router passes: +3.9 F1 over BROS pooled, never significantly worse |
| 12 | Confirmation on fresh documents | GPU | **confirmed:** router +3.8 F1 [+2.6, +5.0]; almost all from `foreign_principle_name` |
| 13 | Dataset audit for the held-out-template protocol | CPU | FATURA is compatible; the other candidates fail |
| 14 | Hybrid on FATURA | GPU | **uninformative**: BROS already scores 99.9% on unseen templates (oracle ceiling +0.0); the router correctly chose BROS everywhere |
| 15 | Data-starved BROS on FATURA (K = 1, 2, 5, 10 seen templates) | GPU | oracle headroom +16.0 / +5.1 / +2.3 at K = 2 / 5 / 10 but a static router captures none; the apparent K = 2 loss (-2.6) was later shown to be seed noise |
| 16 | Per-document routing signal (BROS confidence / endorsement), 5 seeds | GPU | VRDU hybrid H +3.8 [+2.0, +5.6] (criterion met); gate G alone -0.2; FATURA K = 2 gate +1.5 [+0.1, +3.2] (narrow pass); K >= 5 neutral; **static router at K = 2 is +2.3 [-2.1, +7.3], so Study 15's loss did not replicate** |
| 17 | XFUND link F1 of saved checkpoints, truncation and windowing | CPU | reported 65.5 -> **49.9** when cut-off links count as misses -> 59.8 with link-aware inference windows (gold entities) |
| 18 | BROS/VLM error correlation | CPU | errors are positively correlated: VRDU lift 0.46 [0.43, 0.50]; FATURA 0.81 / 0.67 / 0.51 at K = 2 / 5 / 10 |
| 19 | Windowed joint training, **end-to-end** (predicted entities) link F1 | GPU | windows beat truncated training: **43.5 -> 47.7 link F1** (+4.2, 3/3 seeds); predicted entities cost ~16-22 points vs gold entities |
| 20 | Cross-window entity dedup | CPU | no better: entity F1 +1.4 but link F1 -0.35 (negative in all seeds) |
| 21 | Step-matched control for Study 19 | GPU | coverage matters: windows beat the control by **+3.1** link F1; extra steps alone gave +1.2; the control stopped early at 35-43 of 53 epochs |
| 22 | Where do gold-entity link errors come from? | CPU | 72% of misses point downward; 30% are question->question; 25% unreachable by any window; close-neighbour confusion is not the main error |
| 23 | Decode-time levers (threshold, one-to-one, type prior, nearness) | CPU | +2.4 link F1 by forbidding type-violating pairs - **later corrected** (section 5.4) |

---

## 4. The story in four threads

### 4.1 Thread A - Tagging and the UTL gap (Studies 2-5, P)
The UTL gap is not spread across fields. BROS never predicts `foreign_principle_name` (each template words it differently, so a tagger trained on other templates sees a different cue), and on Short-Form it picks the wrong name block (the item-1 filer, not the registrant) 95% of the time. LiLT, DANN and BROS variants all fail the same way, so the problem is a **semantic/wording** gap, not a spatial one that better attention would fix. Plausible ways to close it need semantics that generalize across wordings, which is what a language model supplies.

### 4.2 Thread B - Zero-shot VLM and the hybrid (Studies 6-16, 18)
- **VLM:** Qwen2.5-VL-3B, no training, no field list. It extracts key-value pairs on unseen templates, and a naming step (generated field name + description, matched to the schema) plus one-answer-per-field decoding fixes most wording mismatches: Short-Form `registrant_name` 33.2 -> 96.5 and the decoy is never picked (Study 8).
- **Honest comparison (Study 9):** with exact-span matching the VLM is *worse* than BROS overall (57.9 / 61.2 vs ~70). Its loss is mostly **span boundaries**, then location. The two systems fail on different fields (VLM: `foreign_principle_name` 50-61, Short-Form `registrant_name` 62; BROS: signature and date fields).
- **Hybrid:** a static per-field router (choose BROS or VLM per field using other templates' documents only) gains +3.8 F1 pooled on VRDU, confirmed on 300 fresh documents (Study 12). The gain is almost all `foreign_principle_name`.
- **Where it stops working:** on FATURA BROS reaches 99.9% so there is nothing to route (Study 14); with BROS starved of data the oracle shows large headroom (+16 at K = 2) but no deployable rule captures it (Studies 15-16): a per-document BROS-confidence gate gains only +1.5 at K = 2 and nothing at K >= 5. Study 18 explains why: the VLM is right on 32% of VRDU spans BROS misses against 70% where BROS is right (lift 0.46), so the two fail together more often than apart.

### 4.3 Thread C - GNN and graph ideas (P, Studies 1 and 5)
Every graph model we built or audited either lost to plain BROS or was invalid: standalone RSA-GNN (42.05 UTL), NER-fused GNN (the 81 -> 30 drop came from heuristic labels, so it says nothing about semantic matching), Graph-BROS v1 (collapsed), Graph-BROS v2 (reviewed, never run). The clean statement is "unproven", not "disproven": the in-attention graph idea itself has never had a valid test.

### 4.4 Thread D - Key-value *linking* on XFUND/FUNSD (Studies 17-23)
We trained a joint LiLT model (token tagging + a biaffine relation head over a 16-D pairwise spatial vector) and found, in order:
1. The saved XFUND link F1 of **65.5** (legacy checkpoints, 3 seeds) scored only the 61% of links visible in the first 512 tokens (38% of words were cut off) **and** gave the linker the annotated entities. Counting cut-off links as misses: 49.9. Inference-only link-aware windows: 59.8 (Study 17).
2. With the model's **own** entities (Study 19) link F1 is much lower. Windowed training improves it (43.5 -> 47.7) and the step-matched control (Study 21) shows about 70% of that gain is the windows, about 30% extra training steps.
3. Entity errors, not the linker, cost the most (same checkpoint: 60.9 with gold entities, 38.6 with predicted ones), and merge rules cannot fix it (Study 20).
4. With gold entities the linker is still weak (about 60 F1 against a reachable ceiling near 90). Errors are structural (Study 22): downward partners (58% of gold, recall 49) and section-heading -> field-label links (question -> question: 18% of gold, recall 34). Wrong partners are rarely the nearest candidate (only 10% are at rank 0-1), so this does not look like a short-range geometry deficit.
5. Decode-time rules are worth about +2 to +3 (Study 23) and do not touch the main weakness.

---

## 5. Where we are stuck (and what the evidence does and does not say)

### 5.1 The tagging gap is semantic, and the only thing that fixes it (a VLM) is imprecise
BROS cannot learn wording it has never seen; the VLM understands wording but misses annotation span conventions. The hybrid takes the best of both for one field. Everything else (signature fields, dates, span boundaries) still favours BROS.

### 5.2 The hybrid does not generalize beyond VRDU's one-field story
Gain on VRDU is +3.8 but concentrated in `foreign_principle_name`. On FATURA BROS saturates, so there is no gap to close. Per-document routing is weak because errors are correlated (Study 18). **Without a second dataset with a genuine wording gap, the hybrid is a result about VRDU, not about cross-template extraction.**

### 5.3 Linking is far below the headline number once the evaluation is honest
End-to-end all-links F1 is **47.7** (best arm), against 65.5 originally reported. Two things limit it: entity errors (precision 67-71 vs recall 78-83) and the linker's recall on downward and heading links. Neither is solved by decoding or merge rules.

### 5.4 A correction we made to our own work (important for anyone reading Study 23)
Study 23 first attributed its +2.4 gain to dropping question->question links and called the model's question->question predictions "net harmful". **That was wrong.** Forbidding question->question alone *lowers* F1 by 2.6; those predictions are 44% precise, above the roughly F1/2 break-even. Most of the mask's gain came from forbidding type-violating pairs that touch gold `other` entities, which the end-to-end pipeline never creates. The corrected analysis is the addendum at the end of Study 23.

### 5.5 Metric decision
We score XFUND linking on **all gold links** (including question->question) for comparability with the literature and Studies 17-23. A key->value-only metric was considered and rejected; see memory note `xfund-link-metric-all-links`. If you revisit it, note that the transferable decode rules differ by mode (section 7.2).

### 5.6 Things that look like findings but are not (avoid repeating these)
- Study 14 is uninformative, not negative: BROS had no gap to close.
- Study 15's K = 2 router loss was seed noise (Study 16 with 5 seeds: +2.3 [-2.1, +7.3]).
- The 81 -> 30 NER-fused link F1 drop is a property of heuristic labels (Study 5).
- Study 9 first suggested "VLM matches BROS" under lenient text matching; under exact spans it does not.

---

## 6. Repository map (what exists and where)

| Area | Path | Notes |
|---|---|---|
| Study log (source of truth) | `STUDIES.md` | one entry per study, with registered criteria, results, limits |
| Tagging/training | `src/train.py`, `src/models/token_clf.py`, `src/models/graph_bros.py`, `src/models/spatial_graph.py` | `TrainConfig` has `form_window`, `form_window_stride` (Study 19) |
| Joint tagging + linking | `src/models/joint_extractor.py`, `src/models/linking_head.py` | biaffine relation head, soft type prior (-5 for pairs outside {q->a, h->q, h->a}) |
| End-to-end linking | `src/e2e_link.py`, `src/data/xfund_windows.py` | `predict_windows`, `merge_windows` (strategies `edge_drop`, `center`), `gold_structure` |
| Zero-shot VLM + hybrid | `scripts/zero_shot_kv.py`, `scripts/zero_shot_kv_v2.py`, `scripts/matched_eval.py`, `scripts/hybrid_eval.py`, `scripts/routing_eval.py`, `scripts/fatura_eval.py` | Qwen2.5-VL-3B |
| XFUND link studies | `scripts/xfund_link_eval.py`, `scripts/xfund_e2e_eval.py`, `scripts/xfund_e2e_summary.py`, `scripts/xfund_dedupe_eval.py`, `scripts/xfund_link_errors.py`, `scripts/xfund_decode_study.py`, `scripts/error_correlation.py` | each has a self-test or smoke path |
| Tests | `tests/test_e2e_link.py`, `tests/test_predict_probs.py`, `tests/test_graph_bros.py` | `python tests/test_e2e_link.py` needs `data/xfund` and cached LiLT weights |
| Kaggle notebooks | `notebooks/kaggle_windowed_e2e.ipynb` (Study 19), `kaggle_matched_steps.ipynb` (Study 21), `kaggle_routing_signal.ipynb` (Study 16), others from earlier studies | all follow the 7-cell contract below |
| Configs | `configs/xfund_joint_lilt.yaml`, `xfund_joint_lilt_windows.yaml`, `xfund_joint_lilt_matched.yaml`, `vrdu_track1_bros.yaml`, ... | |
| Results | `results/<study>/` | weights are **not** committed (only Kaggle output holds them) |
| Older handoffs | `NOVELTY.md` (Graph-BROS spec, gitignored), `PROJECT_HANDOFF.md`, `EXPERIMENT_HANDOFF.md`, `intensive_assessment/HANDOFF.md` | superseded by this file for status; still useful for the architecture spec |

**Kaggle notebook contract (7 cells, every notebook):** (1) clone the repo with the `GITHUB_TOKEN` secret and install deps; (2) fetch data; (3) smoke test, raising on failure; (4) run the experiment (resumable, never raises so cell 7 can push); (5) write metrics as JSON; (6) display; (7) push JSON/JSONL only, never weights.

**Working conventions that have mattered:**
- Register hypotheses, variants and success criteria in `STUDIES.md` **before** running anything; commit the registration first.
- Report seed-matched deltas; with 3 seeds do not claim intervals.
- Treat any variant conceived after seeing report-split numbers as post hoc and say so.

---

## 7. What remains (prioritised)

### 7.0 Housekeeping first
1. **Push** the 8 local study commits (Studies 20-23) - `git push`; nothing depends on it but it protects the work.
2. **Review and commit** the uncommitted `STUDIES.md` edit that fills in Study 16's results (it matches `results/routing_v3/*.json`). Stage by file name; the working tree also holds unrelated demo and slide files.
3. **Download Kaggle checkpoints** for the Study 19 windows arm: `results/xfund_e2e/windows/seed_{13,21,42}/best_model.pt` (about 500 MB each; optionally the `base` arm and the Study 21 control). They unlock gold-entity scoring and header-rule tests on CPU.

### 7.1 Study 24 - does a multilingual text encoder help linking? (design drafted, awaiting sign-off)
- **Question:** under the same windowed recipe, does LiLT with a multilingual text stream beat the English one on end-to-end all-links F1?
- **Arms:** A0 = Study 19 `windows` arm (reused); A1 = `SCUT-DLVCLab/lilt-infoxlm-base`, 3 seeds, identical recipe.
- **Verified:** the checkpoint exists (MIT, tokenizer included); its tokenizer is a `LayoutXLMTokenizer` that accepts `boxes=`, so wiring may need only a `BACKBONES` entry in `src/models/token_clf.py`. On XFUND val, InfoXLM tokenization is shorter: median 484 vs 651 tokens, 137 vs 84 forms fit in 512, and 93.2% vs 89.7% of links are reachable by 510/128 windows.
- **Confound to report:** part of any gain is the tokenizer's coverage; A1 also trains on fewer windows per epoch (fewer steps). Add a gold-entity link F1 on the set of links reachable under both tokenizations.
- **Proposed criteria:** text side matters if mean seed-matched delta >= +2.0 link F1 and every seed positive; no better if |mean| < 1.0 or signs mix.
- **Before spending GPU:** confirm the checkpoint loads with `transformers==4.51.3`, run the windowing/e2e tests with the new backbone, and a CPU smoke run. Memory is larger (250k-token vocabulary).

### 7.2 Decode-time rules (cheap, partly done)
- All-links mode: forbidding links from `header` entities gave +1.9 in an end-to-end emulation (post hoc, XFUND-specific; gold has only 17 header links). Needs confirmation with predicted types on the Study 19 models.
- Key-value-only mode (not our primary metric): allowing only question->answer gave +8.2 in the emulation.
- Neither should be made a default without a registered confirmation.

### 7.3 BROS linking head (the project's original hypothesis, still untested for linking)
Add the biaffine head to BROS and train it with the windowed pipeline. Open questions before building: does `JointFormExtractor` accept BROS's `bbox` format and expose hidden states the way `e2e_link.py` expects (the Track-1 code builds BROS boxes from `word_ids`); BROS is English-only and uncased (accents stripped), so on XFUND it is handicapped relative to LiLT. Step 0 (Study 22) found errors are not short-range geometry, so this arm is **less** supported than 7.1; a fair test needs the same handicap on both arms or an English linking dataset with enough data (FUNSD has only 149 train forms).

### 7.4 Stronger VLM side (the only lever on the hybrid's ceiling)
Oracle headroom (+10.6 on VRDU, +18 at K = 2 on FATURA) only matters if the VLM is right more often than the ~58-60% it scores now. Ideas, in rough order of cost:
1. **Few-shot or schema-conditioned prompts** using seen-template examples and the field descriptions, instead of generating descriptions from field names alone (the 3B model followed only simple one-question prompts reliably).
2. **Larger VLMs** than Qwen2.5-VL-3B; Studies 6-12 never varied the model. The 3B size was a compute choice.
3. **Grounded output:** ask the VLM for boxes or OCR line IDs directly so span boundaries come from the OCR layout, attacking the largest measured loss (Study 9: span boundaries, then location).
4. **Span snapping:** after the VLM picks the entity, snap its extent to BROS-style boundaries on existing dumps (CPU-testable on Study 12/16 data).
5. **Learned selector** on top of both systems' outputs using features beyond BROS confidence (Study 18 says confidence alone is weak because errors correlate).

### 7.5 Tabular and table-structured key-value linking (new direction)
Our work so far treats forms as pairs of boxes; tables (line items, header-cell relations) are mostly absent: VRDU ad-buy line items are excluded from gold, and FATURA's 9 core fields leave out its line-item tables. A tabular direction would:
- add a **table detection / structure recognition** stage and treat header->cell relations as key-value links, scored with a graph-aware metric;
- evaluate on a dataset with line items and a held-out-template split (FATURA is the only local candidate; DocILE's line-item task is a likely candidate from memory, not verified here);
- compare a graph-structured table model against our relation head on the same cells.
See the reading list in section 8; none of those papers has been read in full.

### 7.6 Infrastructure
- **No mixed precision anywhere in the training code** (no `autocast`, `GradScaler`, bf16, fp16): all Kaggle runs so far were FP32 on T4s. Adding AMP is a small change that probably speeds training substantially, but it changes numerics, so any new arm must be re-run in the same precision as its control.
- **Cloud GPUs (RunPod / Vast.ai):** an A100 80GB or an RTX 4090 is the sensible pick for base-size models at batch 4; an H100 is not worth the premium. Cell 1's token handling is portable, `WORK_DIR` is hard-coded to `/kaggle/working`. Price snapshots (third-party, volatile): A100 80GB about $1.2-1.4/h on RunPod, about $0.4-0.9/h on Vast.ai. Benchmark one hour before committing.
- **Resume is per seed only:** an interrupted seed restarts from scratch.

### 7.7 Reporting
`STUDIES.md` is the report source. `intensive_assessment/` already holds slide scripts and Q&A drafts (written outside this thread); reconcile their numbers with the corrections in section 5 (in particular the 65.5 link F1 and the Study 23 attribution).

---

## 8. Literature pointers for the next steps

**Caveat:** everything in this section comes from search-result snippets or my recall, not from reading the papers. Verify metrics and dataset details before citing.

**Key-value linking / relation extraction in forms**
- GeoLayoutLM (CVPR 2023, [arXiv 2304.10759](https://arxiv.org/pdf/2304.10759)): geometric pre-training with a relation head; reports 89.45 FUNSD relation-extraction F1 vs 80.35 for LayoutLMv3-large (snippet).
- KVPFormer (AAAI 2023, [paper](https://ojs.aaai.org/index.php/AAAI/article/view/26516)): key-value pairing as question answering with a spatial attention bias; reports 82.23 FUNSD entity-linking F1 (snippet). Evaluation setups differ from GeoLayoutLM's, so the two numbers are not a head-to-head.
- FUNSD revisited for key-value detection ([arXiv 2010.05322](https://arxiv.org/pdf/2010.05322)); relational representation learning in visually-rich documents ([arXiv 2205.02411](https://arxiv.org/pdf/2205.02411)), which evaluates table structure recognition and entity linking together on FUNSD forms (snippet).

**Zero-shot / VLM key information extraction (our section 7.4)**
- VDInstruct ([arXiv 2507.09531](https://arxiv.org/pdf/2507.09531)): layout-aware zero-shot KIE; reports +5.5 F1 over DocOwl 1.5 (authors' claim).
- UniKIE-Bench ([arXiv 2602.07038](https://arxiv.org/html/2602.07038)): benchmark of large multimodal models with constrained and open-category tracks; reports persistent weaknesses in schema grounding and layout-aware reasoning.
- BLOCKIE ([arXiv 2505.13535](https://arxiv.org/pdf/2505.13535v1)): LLM over semantic blocks; claims layout-based approaches struggle on unseen formats - related to our span-snapping idea.
- The VRDU benchmark itself ([arXiv 2211.15421](https://arxiv.org/pdf/2211.15421)) defines the unseen-template task we use.

**Tables and table-structured relations (our section 7.5)**
- PubTables-v2 ([arXiv 2512.10888](https://arxiv.org/html/2512.10888)): full-page and multi-page table extraction; proposes a model that maps images to graph-structured table representations (snippet).
- PulseBench-Tab ([arXiv 2606.07534](https://arxiv.org/pdf/2606.07534)): multilingual table benchmark with a graph-based metric over cell adjacencies (snippet).
- Spatial ModernBERT (2025): joint key-value, table-structure and spatial relation learning on financial documents, reporting 95.49 F1 on CORD (search summary; arXiv ID not verified).
- DocILE (ICDAR 2023, from memory): key information localisation and **line-item recognition** tasks with held-out layouts. Not searched or verified in this thread; worth checking as the held-out-template table dataset.

**What the search did not find:** a benchmark that jointly scores key-value pairs, table structure and cell-relation linking under template-disjoint splits - consistent with this project's framing that linking under template shift is under-measured.

---

## 9. Environment and operational pitfalls

- **Windows memory:** the local page file was 512 MB; LiLT training on CPU crashed with `not enough memory` for tiny allocations until the page file was raised to 16 GB on D:. Two LiLT processes together need that headroom. Free RAM alone is not the limit - commit limit is.
- **Background jobs die with the session:** a restart kills any `nohup` job; scripts therefore save results after each finished seed. A "completed" notification from a launcher can mean only the launcher exited - read the log's `*_EXIT` line.
- **Kaggle:** the `GITHUB_TOKEN` expired mid-run once and lost a push (cell 7 never raises for that reason). Weights are saved on Kaggle but never pushed; download them from the notebook Output tab.
- **XFUND download:** the official GitHub release host failed TLS from this machine; the same JSON files came from the Hugging Face mirror `FrancophonIA/XFUND`. The visible-word count matched the saved metrics' `support` exactly (46,180), which is how we checked the data is the same. `data/` is gitignored.
- **Saved legacy checkpoints** (`results/xfund_joint_lilt`) were trained with an older 12-D spatial feature set; current code builds 16-D features, so a fair baseline needs retraining (Study 19 does this).
- **Conventions in git:** stage files by name; the working tree contains unrelated demo/slide files and an uncommitted `.gitignore` change.

---

## 10. Honest novelty assessment

**Defensible contributions**
1. **A measurement correction.** Reported XFUND/FUNSD-style link F1 can be inflated by input truncation (dropped links are not counted as misses) and by giving the linker annotated entities. In our case 65.5 became 49.9 / 59.8 (gold entities) and 47.7 (end to end). This is a methodological point that generalizes beyond our model.
2. **A template-disjoint view of linking.** Using XFUND val (about 94% of forms have no train sibling by a question-text proxy) as an approximately template-disjoint linking test, plus the characterisation of where links fail (downward, heading-to-label, unreachable).
3. **Complementary-error hybrid on VRDU:** confirmed on fresh documents (+3.8 [+2.6, +5.0]) with a careful chain of studies, plus the finding that per-document routing signals are weak because the two systems' errors correlate.
4. **Negative results with diagnoses:** graph-in-attention and NER-fused GNNs; dedupe; decode rules; the 70/30 split between windows and steps.

**Weaknesses a reviewer will raise**
- The hybrid's gain is concentrated in one field and one dataset; the second dataset (FATURA) was uninformative.
- The original architectural hypothesis (direction-agnostic graph bias in BROS attention) was never validly tested; Graph-BROS v2 was reviewed and shelved.
- Linking results are LiLT-only with 3 seeds and legacy checkpoints in several studies; no BROS linking number exists.
- Several analyses are post hoc (flagged in `STUDIES.md`), and the VLM is a single 3B model.
- The final linking figure (47.7) is modest, and the decode-time gains (+2 to +3) do not address the structural errors.

**Where novelty could still grow:** a BROS or multilingual-encoder linking result under the honest end-to-end protocol; a stronger or grounded VLM that lifts the hybrid ceiling on a dataset with a real wording gap; or a table-structured template-disjoint evaluation.

---

## 11. What has not been verified

- Study 16's FATURA gate pass is narrow (lower bound +0.08); the VRDU intervals resample documents of only 3 templates.
- Studies 19 and 21 have 3 seeds each and no intervals; Study 21's control stopped early, so steps are only approximately matched.
- Studies 20, 22 and 23 use the legacy checkpoints, not the Study 19 models; the Study 23 addendum variants were conceived after seeing the report split.
- The speed-up estimates for A100/H100 over T4 are my estimates, not measurements; no GPU timing exists for any Kaggle run.
- The InfoXLM checkpoint has been located and its tokenizer measured, but it has not been loaded into the joint model.
- Whether BROS plugs into `JointFormExtractor` and `e2e_link.py` has not been tried.
- The literature in section 8 was not read in full.
- **A labelling inconsistency in the repo's own docs, unresolved:** `STUDIES.md` row P calls 91.56 / 64.24 the "LiLT baseline (512-token truncation)", while `NOVELTY.md` and `README.md` call 91.56 / 64.24 "Track-1 LiLT" (sliding windows + jitter + anchoring) and give 87.74 / 66.66 as the truncated LiLT. Section 2 follows `NOVELTY.md`. Check `results/vrdu_lilt/metrics.json` and `results/RelativeCoordinate_ablation.txt` before quoting either label in a report.
- Local state at writing time: 8 study commits are unpushed, and `STUDIES.md` carries an uncommitted Study 16 results block that was not written in this thread.
