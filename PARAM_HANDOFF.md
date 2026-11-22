# PARAM_HANDOFF.md — poster and 2-page report for the novelty assessment

Written 2026-11-22 for Param. **Deadline: today, 22 Nov, 11:59 PM IST** — upload `Teamname_IE643_CourseProject_Poster.pdf` and `Teamname_IE643_NoveltyAssessment_report.pdf` (max 2 pages) to the shared Drive folder. Poster is printed in person on **A1**; presentation on 23 or 25 Nov, 10 AM–1 PM. Rules are in `Novelty.txt` (read it once; a badly designed poster scores very low).

Everything below is traceable to `STUDIES.md` (the full log; every number here has a study number next to it). Raw result files are under `results/`. Figures under `figures/`. How to reproduce / which weights to use: `docs/DEMO_AND_REPRODUCTION.md`.

**`[RUNNING]` = the result depends on a Kaggle run that has not finished yet.** Leave the tag in the text until the number is filled; section 9 lists every `[RUNNING]` item with the file to read when it lands (I will push results to the same branch, `feat/bros-relative-layout`; `git pull` to get them).

---

## 0. What to hand in (checklist from Novelty.txt)

| Item | Notes |
|---|---|
| Poster (A1, PDF) | Header: line 1 "IE 643 Course Project"; line 2 project title; line 3 team name; lines 4–5 members (names, roll numbers, e-mails); IITB logo left, IE 643 logo right (template). Sections 2–13 in the list below. |
| Report (PDF, ≤ 2 pages) | "Approaches tried and experiments conducted for novelty assessment" (see section 7). |
| QR codes | GitHub repo (`ABasakEE/KeyValue-Extraction`, branch `feat/bros-relative-layout` — **check the repo is visible to the evaluators**) and the demo video **[TODO: record the Gradio demo; link not created yet]**. |
| References | Format in Novelty.txt (authors. Title, venue, year. URL). Candidate list in section 10 (verify each URL before printing). |

Team (from the Track-1 notebook header): **Team Unemployed and Unsupervised** — Param Mehta (23b2439), Arjoe Basak (23b1295), IIT Bombay. E-mail addresses: fill in yourselves.
Title suggestion: *Cross-Template Key-Value Field Extraction from Scanned Forms: Generalizing to Unseen Templates* (the allotted project title; the novelty work adds tables and a vision-language reader).

---

## 1. The project in ten lines

- **Task (Project description.txt):** extract key-value fields from scanned forms and generalise to **completely unseen templates**, never using labelled or unlabelled data of the held-out templates in training; measure the seen-vs-unseen gap (MTL vs UTL), analyse failures, give a demo.
- **Protocol:** matched-size (N = 200 training documents) on VRDU `registration-form`, 3 folds (fold 0 = Amendment, 1 = Dissemination_Report, 2 = Short-Form held out for UTL). MTL = all templates seen, UTL = one template unseen. Metric: entity-level micro F1 (exact span).
- **Datasets used:** VRDU registration-form (primary), VRDU ad-buy (line items), FATURA (invoices, 50 templates), XFUND (key-value *linking*, 5 languages), FUNSD (early), DocILE (tables / line items, 5,680 labelled invoices), synthetic generator (early).
- **Before the initial submission (intensive assessment):** LiLT baseline, Track-1 BROS (windows + jitter + anchoring), DANN, graph models. Prior-benchmark row P in `STUDIES.md`: LiLT MTL 91.6 / UTL 64.2; BROS raw 22.7 / 7.5 (truncation); **Track-1 BROS 92.4 / 71.3 ± 11.2 (gap 21.1)**; DANN-BROS 91.8 / 69.6; Delaunay GNN 86.0 / 42.1; NER-fused GNN link F1 81.4 / 30.5.
- **After the initial submission (this handoff):** studies 1–44 in STUDIES.md (28 was reserved and never used; "initial submission" is read here as the intensive-assessment deliverables, i.e. everything up to the prior-benchmarks row P). Four threads, in order of emphasis: **(A) a vision-language reader (VLM) used zero-shot with grounded answers, and its hybrid with BROS; (B) a multilingual text encoder for key-value linking; (C) BROS as the tagger backbone (VRDU, XFUND linking, tables); (D) tabular data — line items — folded into the same pipeline.**
- **Three headline messages:** (1) a *zero-shot* VLM with grounded, schema-guided answers beats the trained tagger on unseen templates of *forms* (76.5 vs 70.5 exact-span F1) and the two are complementary; (2) on *tables* the opposite holds — the trained BROS tagger wins at every training size, and the zero-shot reader fails; (3) the multilingual text stream did **not** help linking, the better backbone (BROS, relative geometry) did (+5.9 link F1).
- **What is new relative to the literature (honest list, section 8):** the measurement correction for linking, the template-disjoint linking protocol, the hybrid with a confirmed gain, the form-vs-table crossover, and a documented set of negative results with diagnoses.

---

## 2. System overview (use for the workflow flowchart and the architecture figure)

```
page image + OCR words/boxes
        │
        ├─► Tagger: BROS Track-1  (BROS-base; 384-word windows, overlap 128, soft voting across windows;
        │            coordinate jitter; content-box anchoring) ──► entity spans (BIO), per field
        │
        │   FORMS (key-value)                                  TABLES (line items)
        │   ├─ joint tagger + biaffine linker (XFUND):         ├─ cells = word sets (19 DocILE fields / 5 ad-buy fields)
        │   │   key ─► value links, windowed inference         ├─ anchor-rule row grouping (per page: anchor field =
        │   └─ schema-guided VLM (Qwen3-VL-8B, 4-bit):         │   most frequent field; each cell joins the closest anchor
        │       OCR lines with ids + field descriptions        │   at/above it) ─► rows of cells ("cell-to-row link")
        │       ─► JSON answer citing line ids ─► grounded      └─ metric: DocILE line-item F1 (ported, `scripts/lir_metric.py`)
        │       to OCR words
        │
        └─► Hybrid on forms: per-field router / abstaining vote between BROS and the VLM (complementary errors)
```
Key design point for the poster: **a key-value pair and a table row are both "linking" decisions between detected cells** (key→value, cell→row), so one tagger plus a cheap linker covers forms and tables; the VLM is the reader that fixes *wording* mismatches the tagger cannot learn from 200 documents.

---

## 3. Thread A — the vision-language reader (main emphasis)

### 3.1 Motivation (Studies 2–5, CPU/GPU, 28 Oct)
The UTL gap is two fields: `foreign_principle_name` (0–13% F1 for every trained model) and `registrant_name` on Short-Form (2–6%; 95% of wrong predictions are the item-1 *filer*, a decoy block). Each template words `foreign_principle_name` differently, so a tagger trained on 200 documents cannot know which text the field name refers to. Study 5 showed the earlier "NER" in the graph model was a regex and the link labels a template-specific heuristic — semantic matching had never been tested. → try a model that *reads field names*.

### 3.2 Zero-shot VLM, step by step (Studies 6–12)
| Study | What | Result |
|---|---|---|
| 6 | Zero-shot v1 (page 1, Qwen2.5-VL-3B vs a text-only LM) | VLM beats the text LM; finds values BROS missed; mapping keys to fields is the bottleneck |
| 7 | v2: whole document + LM naming step (generated field name + description, matched to schema) | Naming fixes wording (Short-Form `foreign_principle_name` 4 → 85) but lowers precision |
| 8 | One-answer-per-field decoding (post hoc) | Raises F1 on every template; Short-Form registrant 96.5, decoy picked 0/100 |
| 9 | **Matched, grounded comparison with BROS** | Under the strict matched metric **BROS wins overall (~70 vs 61)**; errors are complementary: VLM wins role fields, BROS wins signature/date fields and span boundaries |
| 10–11 | Hybrid feasibility (oracle, fill-in, cross-template router) | Oracle +16 F1 ceiling; the router with a *mode-free* mapping passes (+3.9 pooled) |
| 12 | **Confirmation on fresh documents** | **Router beats BROS by +3.8 F1 [+2.6, +5.0] pooled**; almost all from `foreign_principle_name` |

### 3.3 Schema-guided, grounded extraction — "the recipe" (Studies 25, 31)
- **Recipe (LMDX-style):** one call per page. Prompt = the page's OCR lines each with an id (`XX|YY` position code) + field descriptions (written from the field names alone, frozen before any run) + the JSON schema; the model answers `{field: "text copied from the page + line id"}`; the answer is **grounded** back to OCR words by the cited line and quoted text (invalid citations are dropped; a repair step moves a cell to the piece that holds its text). One answer per field, earliest page wins ties. Image optional (pixel ablation).
- **Study 25 (registered 5 Nov, finished 18 Nov):** first version, 3B and Qwen3-VL-8B. **Not an improvement by the registered rule**: the line-id JSON format failed (fell back to one call per field); 8B A2 60.2 (+2.5 over reference, below the +3.0 bar), ~10 below BROS (70.5). Post hoc: image adds +13.7 (3B) / +3.3 (8B); a static router with the 8B spans 76.2 vs BROS 70.5 (+5.7 [+4.4, +7.0]); snapping spans to BROS boundaries 60.5 → 65.5.
- **Study 31 — prompt v2 (copy only the value, an id example, three rewritten field descriptions), Qwen3-VL-8B 4-bit:** **helps, clears the bar, replicates.** V2A2 (image + text) **76.5 (confirm set) and 75.2 (dev set) exact-span F1**, against 60.5 / 58.1 for the Study-25 prompt (+16.0 / +17.0, positive on every template) and **BROS 70.5 / 69.8**. Router with BROS 79.2 vs 70.5 (+8.7 [+6.9, +10.6]). Image adds +3.2 / +3.4; text-only already 73.3 / 71.7. Weak spots: still below BROS on Amendment; `foreign_principle_name` regressed (70 → 52–57); `signer_name` below BROS.
- **Caveat the audience will raise:** the prompt and descriptions were chosen after seeing the dev/confirm results. **Study 32** (registered 20 Nov) replicates prompt v2 plus two post hoc rules (latest-page tie-break; abstaining vote with BROS) on **300 fresh forms nobody looked at** — **[RUNNING / NOT YET STARTED: the Kaggle commit kept queueing; if it never runs, say on the poster that prompt v2 was tuned on dev/confirm and the fresh-set test is pending]**. Result file when done: `results/fresh_set3/` (`notebooks/kaggle_fresh_set.ipynb`).

### 3.4 Where the VLM works and stops working
- **FATURA (Study 33):** recipe A2 **91.7** vs zero-shot 3B VLM 59.6 (+32.1). **Crossover with BROS trained on K seen templates:** LLM wins at K = 1 (+59.4) and 2 (+22.5), tie at K = 5 (+0.9 [−1.8, +3.7]), **BROS wins at K = 10 (−4.2) and 40 (−8.2)**; image +3.8. Earlier (Studies 14–15): BROS alone is 99.9% on FATURA unseen templates, so the router had nothing to route; with data-starved BROS (K = 1–10) headroom exists but no static router captures it. Study 36: a perfect per-field router would gain +4.2 / +5.4 / +5.8 / +4.1 / +0.1 over the better single system at K = 1 / 2 / 5 / 10 / 40 (no learned router trained).
- **Errors are positively correlated (Study 18):** the VLM is right on 32% of VRDU spans BROS misses vs 70% where BROS is right (lift 0.46) → per-document confidence gating gains little (Study 16: +3.8 from the static per-field router; the gate alone −0.2).
- **Fine-tuning the VLM did not help (Study 29, LoRA on Qwen2.5-VL-3B, 180 docs/fold, word-number output):** unseen-template F1 **46.2** vs BROS 70.1 (−23.9 [−25.9, −21.6], negative on all templates and both seeds), 11 below the zero-shot pipeline; MTL 76.1 (gap 29.9). ~25% of gold fields found with wrong boundaries, 22–30% not predicted. **Study 30** (does an OCR-segment output format close the gap?) was registered but **not run** (too long for the time left).
- **VLM for *linking* (Study 27, zero-shot Qwen2.5-VL-7B on XFUND, 5 languages):** V3 (gold entities given) link F1 **23.7**, V2 (end to end) **3.0**, V4 (LiLT tagger's entities, VLM links; first 50 forms only) **23.9** — against **47.4** for the trained biaffine linker. The VLM over-links (precision ≈ 19%). V1 (image only) not run.
- **VLM on tables:** recipe on VRDU ad-buy line items **38.2** row F1 vs BROS 68.9–85.7 (1.71× too many rows); on DocILE the pilot gate **failed** (cited ids that exist 88.3%, bar 90%) and the pilot row F1 is **26.6** (Study 38). **VLM as field typer on top of BROS's DocILE cells (Study 43): −7.45 row F1 [−10.0, −5.1]**; typing accuracy 46–52% vs BROS 77%; it fixes a third of BROS's errors but breaks 47% of what BROS had right; only on net-only documents is it better (26.9% vs 7.7%).
- **Take-away sentence for the poster:** *A zero-shot VLM that must copy and cite OCR text is a strong reader of "which text does this field name mean" on forms, complementary to a trained tagger; it is not a good reader of dense tables or of links, and fine-tuning it on 200 documents made it worse.*

---

## 4. Thread B — the multilingual encoder (Study 24) and what we did instead

- **Question:** the joint tagger-plus-linker used an English-only text stream on Spanish, French, German, Italian and Portuguese XFUND forms. Does LiLT's **multilingual** text stream (`SCUT-DLVCLab/lilt-infoxlm-base`, same layout stream, same recipe) raise end-to-end entity and link F1?
- **Setup (registered 6 Nov, run 14 Nov):** 250 XFUND val forms, 3 seeds, windowed (510/128) end-to-end inference, entities from the model's own tags. A0 = LiLT-en (Study 19 arm), A1 = LiLT-InfoXLM. Side-effects measured beforehand: XLM-R tokenization is shorter (median 484 vs 651 tokens per form; 137 vs 84 of 250 forms fit in 512 tokens; 93.2% vs 89.7% of links reachable by windows).
- **Result: worse (registered rule).** Link F1 (windowed, primary) **47.71 → 44.22 (−3.49; −4.06 / −2.94 / −3.49 per seed)**; entity F1 75.9 → 73.2; single-window link F1 unchanged (41.69 vs 41.88), so the loss appears only under windowed inference (windows add +6.0 for A0 but +2.3 for A1); question→answer-only link F1 50.36 for A1.
- **Reading and limits:** "no gain with this recipe", not "multilingual encoders cannot help" (one backbone, English hyper-parameters unchanged, tokenizer coverage and step count not separated by a control).
- **Why language was not the bottleneck (supporting studies):** Study 22 — 72% of missed links point *downward* (recall 49 vs 79 for same-line partners), 30% of misses are question→question links, 25% are unreachable by any window; wrong partners are rarely the nearest candidate. Errors are **structural/geometric**. Then:
  - **Study 26 (link-preserving crops, relation-rich synthetic pretraining):** crops help, link F1 **47.4 → 49.7** (+1.4 / +2.3 / +3.2), gains on partner-above (+4.0) and far-partner links; synthetic pretraining alone +0.1.
  - **Study 37 — swap the backbone to BROS (relative layout), keep the linker: link F1 47.42 → 53.35** (see 5.3). An *English-only* backbone with better geometry beat the *multilingual* one on multilingual forms.
- **Poster sentence:** *Multilingual text did not help key-value linking (−3.5 link F1); better relative geometry did (+5.9).* Honest footnote: BROS is English-only and XFUND is multilingual, so the gain appears despite the language mismatch.

---

## 5. Thread C — BROS as the tagger backbone

### 5.1 History
- Plain BROS (512-token truncation) **collapsed: MTL 22.7 / UTL 7.5** — multi-page VRDU documents lose most words past 512 tokens (83% of corpus tokens dropped for LiLT). **Track-1 fixes** (384-word sliding windows with overlap 128 + soft voting, ±80 px coordinate jitter, aspect-preserving content-box anchoring) turned it into **MTL 92.36 ± 0.3 / UTL 71.27 ± 11.2** (gap 21.1) — the best trained model before this phase. UTL variance is driven by the fold (83 / 72 / 58), not the seed. DANN (template-adversarial) did not help; Graph-BROS v2 was reviewed and shelved (Study 1: broken edge features, inert gates).
- **Engineering lesson to mention in the report:** a CPU check under transformers 5.17 hid a tokenizer bug that only appears under the Kaggle pin 4.51.3 (windows built wrongly; found when the BROS XFUND run failed); fixed in `src/data/xfund_windows.py`.

### 5.2 BROS across datasets (where it wins)
| Dataset | Result |
|---|---|
| VRDU registration (forms) | Track-1 BROS 92.4 / 71.3 — baseline for the VLM comparison; VLM recipe 76.5 / 75.2 on dev/confirm vs BROS 70.5 / 69.8 |
| FATURA | **99.9%** on unseen templates at K = 40; data-starved crossover with the VLM recipe (K ≈ 5, Study 33) |
| VRDU ad-buy (line items) | **BROS row F1 68.9 / 83.2 / 83.8 / 85.7 at N = 10 / 50 / 100 / 200**; recipe 38.2; **BROS wins at every N ≥ 10** (Study 34) |
| DocILE (line items) | 0.3 / 44.7 / 52.9 / **59.6** at N = 10 / 50 / 200 / 1000 layouts-disjoint (Study 39); whole pool **60.4** (Study 41) |
| XFUND (linking) | see 5.3 |

### 5.3 BROS inside the joint tagger + biaffine linker (Study 37, XFUND, 250 val forms, 3 seeds, windowed end to end)
| Arm | Entity F1 | Link F1 | Link_qa F1 | Link F1 per seed (13 / 21 / 42) | Minus LiLT A0, same seeds |
|---|---|---|---|---|---|
| LiLT A0 (reference) | 75.84 | 47.42 | 53.15 | 48.33 / 47.01 / 46.93 | |
| **BROS J1** | **80.20** | **53.35** | **59.59** | 53.10 / 52.93 / 54.01 | **+4.77 / +5.92 / +7.08** |
| BROS J2 (+ coordinate jitter) | 79.63 | 53.00 | 59.66 | 53.10 / 52.72 / 53.19 | +4.77 / +5.71 / +6.26 |
Registered rule (≥ +1.5 and positive in all seed pairs): **BROS better**. Recall by group (A0 / J1): above 34.8 / 50.2 (+15.4), below 39.8 / 46.5, same line 63.5 / 67.4, question→question 31.2 / 32.2. Jitter neither helps nor hurts. Caveats: A0 trained in an earlier session; BROS is English-only.

### 5.4 LayoutLMv3 (text + layout + **page image**) vs BROS — "does adding pixels help?" (Study 42, deliverable 2 asks for text + visual + layout)
- Plumbing built this week: page images (224², one image per window; windows cut at page boundaries so each window has exactly one page image), blank-image control arm (same model, constant image), no jitter/anchoring (they would move boxes but not pixels). `configs/vrdu_layoutlmv3.yaml`, `..._blank.yaml`.
- **Seen templates (MTL), 3 folds × 2 seeds, F1:** pages 92.08 / 91.16, 93.33 / 92.49, 92.10 / 92.25 (mean **92.24**); blank 91.26 / 90.97, 93.07 / 91.97, 93.30 / 93.54 (mean **92.35**); BROS Track-1 **92.36**. Pixel effect on seen templates: **−0.12** (six paired tasks) — none.
- **Unseen templates (UTL, folds 0 and 1, seeds 13 and 42; registered rule: pixels help if mean difference ≥ +2.0, positive in ≥ 3 of 4 pairs and both fold means): partly in, verdict `[RUNNING]`.** Fold 0 (Amendment held out): **pages 69.98 (70.49 / 69.47) vs blank 66.10 (65.96 / 66.23): +3.89, positive in both seeds**; BROS Track-1 83.38. Fold 1 (Dissemination_Report): **blank 54.70 (55.21 / 54.18); pages `[RUNNING]`**; BROS 72.07. Pixels help most on the signature-block fields of fold 0 (`signer_title` 36.6 vs 9.7, `signer_name` 71.9 vs 60.6) and not on `registrant_name` (50.0 vs 53.3; BROS 91.1). Gap fold 0: pages 21.6, blank 25.0, BROS 8.8. **LayoutLMv3 is 13–17 points below BROS on unseen templates; these runs have no jitter/anchoring, so this is a recipe-plus-backbone difference, not a clean backbone comparison.** Fill when done: pages fold 1 `[RUNNING]`, final difference `[RUNNING]`, verdict `[RUNNING]`. BROS Track-1 on the same two folds: UTL **77.72** (83.38 / 72.07), MTL 92.29. MTL–UTL gap for LayoutLMv3 `[RUNNING]`. The hardest template (Short-Form, fold 2) is **not** covered by the UTL runs (time) — say so.
- Files when done: `results/vrdu_layoutlmv3_utl_f0|f1/`, `results/vrdu_layoutlmv3_blank_utl_f0|f1/`, summaries `results/vrdu_layoutlmv3_summary_*.json`.

---

## 6. Thread D — tabular data inside the same pipeline

**Why tables:** the project's forms (VRDU, FATURA) mostly leave out line-item tables; real forms have them. We added them as a second "linking" problem: detect cells, then link cells into rows (the same detect-then-link logic as key→value).

### 6.1 Design (Studies 34, 35, 39)
- **Cells are word sets** (a cell can span two OCR lines); fields: 5 on ad-buy, 19 on DocILE (`line_item_description`, `quantity`, `unit_price_gross/net`, `amount_gross/net`, `code`, `date`, …). Tagging = BIO over the 19 fields with the *unchanged* Track-1 BROS recipe.
- **Rows by the anchor rule:** per page, the most frequent predicted field is the anchor; each cell joins the closest anchor at or above it (0.5 span-height tolerance). Metric = DocILE line-item F1 (rows matched one-to-one to maximise matching cells; a cell is right only if its word set equals the gold's; ported in `scripts/lir_metric.py`).
- **Ceilings (oracle, gold tags decoded):** DocILE **94.4 row F1 / 97.8 cell F1**; ad-buy 90.0 / 94.9. 3.5% of DocILE line-item cells contain no OCR word, so no OCR-based method reaches 100.
- **A learned pair classifier for grouping did not beat the rule (Study 35):** +2.38 [+0.42, +4.71] on ad-buy but −4.32 [−9.28, −0.90] on DocILE (pair AUC 0.97–0.998; clustering is the weak part) → the rule stays.

### 6.2 DocILE results (layout-disjoint: 200 test layouts, 100 validation layouts; vendors also disjoint; Studies 39, 41)
| Training documents | UTL row F1 | cell F1 |
|---|---|---|
| 10 | 0.3 (training never left the all-`O` state; not a real measurement) | 0.3 |
| 50 | 44.7 | 46.4 |
| 200 | 52.9 | 54.7 |
| 1000 (about 285 layouts) | **59.6** | 61.8 |
| whole pool, 2,908 docs, 562 layouts (2 seeds: 13 / 21) | **60.4** (59.84 / 60.96) | |
- **Flattening:** whole pool gains ≤ 1.4 over N = 1000.
- **Seen vs unseen (Study 41; 200 held-out documents of *seen* layouts):** MTL row F1 **66.0** (68.87 / 63.02) vs UTL 60.4 → gap **2 to 9 points depending on the seed**, vs **21 points on VRDU forms**. A longer-trained seed (11 epochs) improved on seen layouts (68.9) but not unseen ones (59.8) — the signature of memorising layout conventions.
- **Literature position (read from the PDFs, Study 39):** DocILE's official line-item F1 is the same metric. RoBERTa-base baseline **68.6** overall (**59.8 on unseen layouts**, 56.8 few-shot, 75.6 many-shot); LayoutLMv3-base 72.1 (IIT-CDIP pre-training, prohibited by the benchmark); winner GraphDoc **75.9** (pre-trained on 932k *unlabeled* DocILE documents — which would violate our zero-same-template rule). **Our 59.6 ≈ the published unseen-layout baseline with ≤ 1,000 documents and no unlabeled pre-training.** Test sets differ; comparison is rough.

### 6.3 Failure analysis of the tabular tagger (Studies 40, 43, 44 — deliverable 5)
- **Loss budget (unseen layouts, share of gold spans, mean of 5 runs):** right 64.0%; **missed 21.4%** (13.6% a field predicted elsewhere in the document but not here; 6.9% a field never predicted in the document; 1.0% split gold cell); wrong field 8.2% (monetary family 5.9%); wrong extent 4.8%; wrong field and extent 1.5%. False spans = 22.6% of predictions (15.5% inside the table's extent, 7.1% outside: headers, totals, addresses).
- **Template-dependent vs intrinsic (same two full-pool runs, seen minus unseen):** shrinks on seen layouts — fields absent from a document −4.3, partly tagged fields −3.1, monetary typing −2.1, in-table false spans 13.9% → 7.2%. Does **not** shrink — false spans outside the table (6.6% vs 7.0%), boundary errors (go *up*, +1.4), ~27% of gold spans still wrong/missed on seen layouts.
- **Gross vs net is not a function of the layout:** 63% of net-only documents (266 of 421) sit in layouts that also contain gross-only documents; 58 of 241 layouts are mixed → the label depends on the document (does it include tax?), so a layout-learner types it better on seen layouts only. Net amount: 9.7% right (81% labelled another monetary field); net unit price 17.7% right.
- **Which documents fail (row F1; five runs pooled):** net-only amount documents **43.9** [35.2, 52.9] vs gross 63.5; `structured_rows` table structure 47.9 [37.0, 59.8] vs normal 62.2; 6+ fields used 50.4; 1–2 rows 53.8 vs 7–15 rows 65.2. **No association with border type** (all borders 58.7, column borders 61.5, none 60.0). Rare fields are most often never touched (currency 45%, units of measure 41%, code 38%).
- **What did not recover the errors:** column field-vote **−2.7** row F1 (harmful), table-extent rule **+0.0** (Study 40); VLM typer **−7.45** (Study 43); learned grouper (Study 35); recipe on tables (Studies 34, 38). Upper bound of perfect monetary typing on exactly-found cells: +5.9.
- **Assumptions that fail (poster box):** (1) *a field's label follows from the layout* (gross/net); (2) *every table has the secondary fields* (currency/units/code often absent); (3) *errors are wrong labels* — they are mostly omissions (21% vs 8%); (4) *a zero-shot reader can type cells from the page* (VLM 46–52% vs 77%).

---

## 7. XFUND key-value linking: the measurement correction (Studies 17–23) — the other novelty

- **Study 17:** the reported **65.5** link F1 scores only the 61% of links visible in 512 tokens and gives the linker *gold* entities; counting cut-off links as misses gives **49.9**; link-aware windows (510/128) at inference give **59.8** (gold entities); end to end (predicted entities) **47.7** (Study 19; entity F1 75.9). XFUND val is ~94% sibling-free (no train form with the same questions), so it is an approximately template-disjoint linking test (siblings cost < 1 point).
- **Study 19:** windowed joint training helps (end-to-end link F1 43.5 → 47.7, all 3 seeds); predicted entities cost ~22 points vs gold entities. Study 21: a quarter to a third of the gain is extra gradient steps. Study 22: errors are structural (downward links, question→question, unreachable). Studies 20/23: dedup does not help; decode levers help modestly (+2.4) — with a correction found the same day (gain mostly from forbidding pairs the end-to-end pipeline never creates).
- Use for the report: "**our end-to-end link F1 on an approximately template-disjoint split is 47.7 (LiLT) → 53.4 (BROS joint), not the 65.5 a gold-entity, truncated evaluation reports.**"

---

## 8. Claims to make, and claims not to make

**Safe to claim (all with study numbers above):**
1. Zero-shot grounded VLM > trained tagger on unseen *form* templates (76.5/75.2 vs 70.5/69.8; replicated on dev and confirm; **fresh-set test `[RUNNING]`**), errors complementary, hybrid +3.8 confirmed on fresh documents (Study 12), +8.7 with prompt v2 (Study 31).
2. Form vs table crossover: the recipe wins with few seen templates and loses to BROS with ≥ 10 (FATURA), and loses on tables at every size (ad-buy, DocILE).
3. Multilingual encoder: −3.5 link F1; BROS: +5.9.
4. Tables: 59.6–60.4 row F1 on unseen DocILE layouts ≈ published baseline; small seen/unseen gap; failure taxonomy.
5. Measurement correction for XFUND linking.
6. Pixels do not change seen-template F1 (−0.12); unseen-template effect `[RUNNING]`.

**Do not claim:**
- that the VLM prompt was not tuned (it was tuned on dev/confirm; fresh set pending);
- that multilingual encoders cannot help, or that BROS is better than LiLT in general (one recipe, English BROS, XFUND);
- the 3-fold UTL mean for LayoutLMv3 (only folds 0 and 1 run);
- anything from the DocILE leaderboard comparison as a head-to-head (different test documents);
- the MTL–UTL gap on DocILE as a stable number (2 seeds, 2–9 points).

**Weaknesses a reviewer will raise (and the honest answer):** the hybrid's gain is concentrated in one field and one dataset (FATURA was uninformative; DocILE KILE fields would be the second wording-gap dataset — not done); several analyses are post hoc (each flagged in STUDIES.md); the VLM is a single model; Study 30, 32 (if still queued) and the V1/V4 arms are not run; tables use one tagger recipe not tuned for DocILE.

---

## 9. `[RUNNING]` / open register (fill when results land)

| Item | Where it goes | File to read when done | Status |
|---|---|---|---|
| LayoutLMv3 unseen templates, folds 0–1, pages vs blank, seeds 13/42 (Study 42 primary cell) | Results table, section 5.4, Conclusions | `results/vrdu_layoutlmv3_utl_f0\|f1/`, `..._blank_utl_f0\|f1/`, `results/vrdu_layoutlmv3_summary_*.json`; STUDIES.md Study 42 | `[RUNNING]` (four Kaggle notebooks, ~1.4 h each) |
| LayoutLMv3 MTL–UTL gap (folds 0–1; MTL values already above) | Seen-vs-unseen table | same | `[RUNNING]` |
| Study 32 — prompt v2 + rules on 300 fresh forms | Claim 1 caveat | `results/fresh_set3/` | `[RUNNING / NOT STARTED]` (commit queue) |
| BROS Track-1 demo weights (MTL + UTL fold 0) | Live demo / demo video | download from `notebooks/kaggle_demo_weights.ipynb` Output tab | `[RUNNING / NOT STARTED]` |
| BROS joint J1 weights for the linking demo | arrow figure / demo | Kaggle Output of `kaggle_xfund_bros_joint` (`results/bros_joint/J1/seed_42/best_model.pt`, not verified) | `[TODO check]` |
| Key→value **arrow** figure | Poster results panel | screenshot of `python demo.py` (works now with the local LiLT linker; BROS J1 would be better) | `[TODO]` |
| Demo video + QR code | Poster | — | `[TODO]` |
| Study 30, V1/V4-full, DocILE recipe test run | — | — | **not run (cut for time)**; say so |

---

## 10. Figures and tables available now

- `figures/example_table_good.png`, `example_table_failure.png` — DocILE line items, BROS full pool (seed 21) on unseen layouts: boxes by predicted field, numbered rows, gold alongside.
- `figures/example_form_good.png`, `example_form_failure.png` — VRDU registration forms, BROS Track-1 on unseen templates (Amendment good case; Short-Form failure: green exactly right, red wrong/extra, blue dashed missed). Regenerate / change documents: `python scripts/make_example_figures.py`.
- Tables to build from sections 3–6: (a) prior benchmarks vs Track-1 BROS (P row); (b) recipe vs BROS on VRDU (70.5 / 69.8 vs 76.5 / 75.2) and hybrid; (c) FATURA crossover K = 1…40; (d) XFUND link F1: LiLT-en 47.7 → LiLT-XLM 44.2 → crops 49.7 → BROS 53.4; (e) DocILE N-curve and published baselines; (f) failure taxonomy bars; (g) LayoutLMv3 pages vs blank (MTL done, UTL `[RUNNING]`).
- Suggested plots: FATURA crossover curve (Study 33 numbers); DocILE UTL F1 vs number of training layouts (46 → 110 → 285 → 562: 44.7 → 52.9 → 59.6 → 60.4); stacked loss-budget bars unseen vs seen (section 6.3).

---

## 11. Poster plan (A1; map to the 13 required parts)

1. **Header** per template.
2. **Abstract (≤ 100 words):** unseen-template key-value extraction; BROS tagger + grounded VLM hybrid; linking and tables; headline numbers (76.5 vs 70.5; 47.7 → 53.4; 59.6 row F1; multilingual −3.5).
3. **Introduction:** why cross-template matters (new vendor, new form → retraining), why deep learning (layout-aware encoders, VLMs), what is hard (wording gap, linking, tables).
4. **Workflow flowchart:** prior work (LiLT/BROS Track-1/DANN/GNN) → failure analysis (two fields) → VLM zero-shot → hybrid/router → FATURA crossover → linking correction (XFUND) → multilingual / BROS / crops → tables (ad-buy → DocILE) → pixels (LayoutLMv3).
5. **Methods and approaches:** section 2 diagram; recipe box (section 3.3); BROS Track-1 box.
6. **Dataset table:** VRDU registration-form (1,915 documents in the loader, 6 entity types, 3 templates), ad-buy (641 documents, 9,163 line-item rows), FATURA (50 templates, 9 core fields), XFUND (5 languages, 250 val forms), DocILE (5,680 docs, 36 KILE + 19 LIR fields, 1,063 layouts, test = 200 layouts). *Check the counts against `STUDIES.md` (Studies 13, 34, 39) before printing; the poster template link is in the course announcement, not in the repo.*
7. **Novelty assessment panel:** the four threads (A–D) + pixels; what is new (section 1).
8. **Results:** tables/plots above + the four example figures + arrow figure `[TODO]`.
9. **Conclusions and future work:** crossover is the story; future: learned router, document-level tax question for DocILE, a second wording-gap dataset (DocILE KILE), unlabeled-data-free pre-training, a multilingual backbone *with* relative geometry.
10. **References** (below). 11. **Acknowledgements.** 12–13. **QR codes** (GitHub, demo video `[TODO]`).

## 12. 2-page report plan (Teamname_IE643_NoveltyAssessment_report.pdf)
- Page 1: problem and protocol (5 lines); what existed at submission; the four novelty threads with one headline number each; one table (VRDU recipe vs BROS vs hybrid) and one figure (FATURA crossover).
- Page 2: linking (47.7 → 53.4, correction 65.5 → 47.7); tables (DocILE N-curve, published baseline, taxonomy, negative results list); pixels/LayoutLMv3 (`[RUNNING]`); limitations and what was not run; references (compact).

## 13. References (author lists are written from memory: check every author list and URL against the paper page before printing; format per Novelty.txt)
- Teakgyu Hong, Donghyun Kim, Mingi Ji, Wonseok Hwang, Daehyun Nam and Sungrae Park. BROS: A Pre-trained Language Model Focusing on Text and Layout for Better Key Information Extraction from Documents, AAAI, 2022. https://arxiv.org/abs/2108.04539
- Jiapeng Wang, Lianwen Jin and Kai Ding. LiLT: A Simple yet Effective Language-Independent Layout Transformer for Structured Document Understanding, ACL, 2022. https://arxiv.org/abs/2202.13669
- Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu and Furu Wei. LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking, ACM Multimedia, 2022. https://arxiv.org/abs/2204.08387
- Yiheng Xu, Tengchao Lv, Lei Cui, et al. XFUND / LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding, 2021. https://arxiv.org/abs/2104.08836
- Zilong Wang, Yichao Zhou, Wei Wei, Chen-Yu Lee and Sandeep Tata. VRDU: A Benchmark for Visually-rich Document Understanding, KDD, 2023. https://arxiv.org/abs/2211.15421
- Štěpán Šimsa, Milan Šulc, Michal Uřičář, et al. DocILE Benchmark for Document Information Localization and Extraction, ICDAR, 2023. https://arxiv.org/abs/2302.05658
- Štěpán Šimsa, Michal Uřičář, Milan Šulc, et al. Extended Overview of DocILE 2023, CLEF Working Notes (CEUR-WS Vol-3497), 2023. https://ceur-ws.org/Vol-3497/paper-049.pdf
- Yan Wang, Jun Du, Jiefeng Ma, et al. USTC-iFLYTEK at DocILE: A Multi-modal Approach Using Domain-specific GraphDoc, CLEF Working Notes, 2023. https://ceur-ws.org/Vol-3497/paper-052.pdf
- Mahmoud Limam, et al. [check author list] FATURA: A Multi-Layout Invoice Image Dataset for Document Analysis and Understanding, 2023. https://arxiv.org/abs/2311.11856
- Shuai Bai, et al. Qwen2.5-VL Technical Report, 2025. https://arxiv.org/abs/2502.13923 (add the Qwen3-VL report/model card for the 8B model).
- Vincent Perot, et al. LMDX: Language Model-based Document Information Extraction and Localization, 2023. https://arxiv.org/abs/2309.10952
- Guillaume Jaume, Hazim Kemal Ekenel and Jean-Philippe Thiran. FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents, ICDAR-W, 2019. https://arxiv.org/abs/1905.13538
- Yaroslav Ganin, et al. Domain-Adversarial Training of Neural Networks, JMLR, 2016. https://arxiv.org/abs/1505.07818
- Jiabang He, et al. [check author list] Do-GOOD: Towards Distribution Shift Evaluation for Pre-Trained Visual Document Understanding Models, SIGIR, 2023. https://arxiv.org/abs/2306.02623 (as cited in README.md; check author list)
- Hugging Face Hub model cards: `jinho8345/bros-base-uncased`, `SCUT-DLVCLab/lilt-roberta-en-base`, `SCUT-DLVCLab/lilt-infoxlm-base`, `microsoft/layoutlmv3-base`, `Qwen/Qwen3-VL-8B-Instruct`. Accessed on: 22nd November, 2026.
(Licence note for the report: LayoutLMv3 is CC-BY-NC-SA-4.0; BROS Apache-2.0; LiLT MIT; DocILE terms of use were read and results may be pushed to GitHub.)

## 14. Where things are (repo map for the friend)
- `STUDIES.md` — full log, one entry per study with question, setup, rule fixed in advance, results, limits (read the summary table first).
- `NOVELTY_ASSESSMENT.md`, `NOVELTY.md`, `ARJOE_HANDOFF.md`, `intensive_assessment/` — pre-submission material and the earlier threads.
- `docs/DEMO_AND_REPRODUCTION.md` — which model/seed/weights for which demo; commands and notebooks per result.
- `notebooks/kaggle_*.ipynb` — one per GPU study; `scripts/` — analysis and scoring; `src/` — models, training; `tests/` — unit checks; `configs/` — recipes; `results/` — pushed result files; `figures/` — example figures; demos: `demo_bros.py` (VRDU entity boxes), `demo.py` (key-value linking arrows, LiLT checkpoint present locally), `demo_gnn.py` (weakest model, not recommended).
- Pitfalls: Kaggle pins `transformers==4.51.3` for BROS/LayoutLMv3 notebooks; do not trust CPU checks under another version (see 5.1).
