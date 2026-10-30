# Studies Log

Every experiment and analysis on the cross-template key-value extraction problem, in order, so results can be
lifted into reports and slides. One entry per study: question, type (GPU run or CPU analysis), setup, key
numbers, conclusion, and where the evidence lives. **Add an entry for every new study.**

Conventions:
- **MTL / UTL** = seen- / unseen-template regimes of the matched-size protocol (`src/data/splits.py`,
  N = 200 training documents in both). Test sets of UTL fold *f* = all documents of the held-out template.
- **F1** = entity-level micro F1 in %. `[a–b]` = 95% bootstrap interval (resampling documents/pages within
  each template). Non-overlapping intervals = a real difference; overlapping = not shown to differ.
- **"Indicative"** = systems scored on different documents or with different metrics; only **matched**
  comparisons (same documents, same gold, same metric) support claims of one system beating another.
- Templates (VRDU `registration-form`): fold 0 = Amendment, fold 1 = Dissemination_Report, fold 2 = Short-Form.

## Summary

| # | Study | Type | Date | Main finding |
|---|---|---|---|---|
| P | Prior benchmarks (before this log) | GPU | ≤ 2026-10-28 | Best UTL: Track-1 BROS 71.3 ± 11.2; every model has a 21–51 pt MTL–UTL gap |
| 1 | Graph-BROS v2 design review | CPU | 2026-10-28 | Not worth running: edge features broken, half the gates frozen, fast-track baseline mis-stated |
| 2 | Per-entity UTL failure analysis | CPU | 2026-10-28 | The gap is two fields: `foreign_principle_name` (0–13% everywhere), `registrant_name` on Short-Form (2–6%) |
| 3 | VRDU template inspection | CPU | 2026-10-28 | Short-Form has a decoy name block; each template words `foreign_principle_name` differently |
| 4 | BROS error dump (diagnostic) | GPU | 2026-10-28 | Confirmed: 95% of wrong Short-Form registrants are the item-1 filer; `foreign_principle_name` is never predicted |
| 5 | NER-fused GNN audit | CPU | 2026-10-28 | The "NER" is a regex and the KV link labels are a template-specific heuristic; semantic matching was never tested |
| 6 | Zero-shot v1 (page 1, VLM vs text LM) | GPU | 2026-10-28 | VLM beats text LM; finds the values BROS missed; mapping keys to fields is the bottleneck |
| 7 | Zero-shot v2 (whole doc + LM naming) | GPU | 2026-10-29 | Naming fixes wording mismatches (Short-Form `foreign_principle_name` 4 → 85) but lowers precision |
| 8 | One-answer-per-field decoding (post-hoc) | CPU | 2026-10-29 | Raises F1 on every template; Short-Form registrant 96.5, decoy picked 0/100 |
| 9 | Stage B: grounded, matched VLM vs BROS | CPU | 2026-10-29 | Under the strict matched metric BROS wins overall (~70 vs 61); errors are complementary: VLM wins role fields, BROS wins signature/date fields and span boundaries |
| 10 | Hybrid feasibility (oracle, fill-in, cross-template router) | CPU | 2026-10-29 | Ceiling is large (oracle +16 F1) but neither pre-registered rule passes; the VLM key/named choice does not transfer across templates |
| 11 | Mode-free VLM mapping + hybrids re-tested | CPU | 2026-10-29 | The cross-template router with the mode-free mapping passes: +3.9 F1 over BROS pooled, never significantly worse; needs confirmation on fresh documents |
| 12 | Confirmation of the study-11 hybrid on fresh documents | GPU | 2026-10-30 | **Confirmed:** router beats BROS by +3.8 F1 [+2.6, +5.0] pooled (study 11: +3.9); almost all of it from `foreign_principle_name` |
| 13 | Dataset audit for the held-out-template protocol (FATURA etc.) | CPU | 2026-10-30 | FATURA is compatible (50 labelled templates, 9 core fields, official unseen-template split) but needs block-level scoring and tokens rebuilt from the original annotations; the other candidates fail |
| 14 | BROS + VLM hybrid on FATURA held-out template folds | GPU | 2026-10-31 | **Not passed, uninformative:** BROS already scores 99.9% on unseen templates (oracle ceiling +0.0), so the router correctly sent every field to BROS (0.0 gain, no harm); the VLM alone scores 59.6% and naive fill-in hurts (-2.2) |

---

## P. Prior benchmarks (before this log)

VRDU registration-form, matched protocol, N = 200. Source: `NOVELTY.md`, `results/*/metrics.json`.

| Model | MTL F1 | UTL F1 | Gap |
|---|---|---|---|
| LiLT baseline (512-token truncation) | 91.56 | 64.24 | 27.3 |
| Standalone RSA-GNN (Delaunay graph over frozen encoder) | 85.97 | 42.05 | 43.9 |
| BROS raw (512-token truncation) | 22.70 ± 17.0 | 7.48 ± 13.1 | — |
| **Track-1 BROS** (384-token sliding window + coordinate jitter + relative anchor) | **92.36 ± 0.3** | **71.27 ± 11.2** | 21.1 |
| DANN-LiLT (template-adversarial) | 91.12 ± 1.4 | 64.79 ± 13.7 | 26.3 |
| DANN-BROS | 91.80 ± 0.9 | 69.59 ± 13.1 | 22.2 |
| NER-fused joint GNN (link F1, not entity F1) | 81.40 | 30.48 | 50.9 |
| Graph-BROS v1 (collapsed run) | 26.67 ± 30.8 | 39.79 ± 12.4 | — |

---

## 1. Graph-BROS v2 design review — CPU, 2026-10-28

**Question:** is the β-skeleton graph-in-attention design (Graph-BROS v2, commit `5eeaaa4`) worth a GPU run?

**Method:** code review of `src/models/graph_bros.py`, `src/models/spatial_graph.py`, the config, and the
per-run results of Track-1 BROS.

**Findings:**
- `nn.LayerNorm(3)` normalises each edge's 3 heterogeneous features against each other, destroying the distance
  signal; all padding edges collapse to one bias.
- Half the per-head graph gates start at exactly 0 behind a ReLU (zero gradient), so they never train; the rest
  start at a −0.09 penalty, so the graph mask is nearly inert.
- The direction-dependent BROS term is still present in every layer; LoRA sits on a fully trainable backbone. A
  v2 result could not attribute any gain to the graph.
- **Evaluation bias:** the fast-track protocol (folds 0–1) drops the hardest fold. Track-1 BROS on exactly those
  runs scores **76.6** UTL, not the 71.27 used as the baseline (~5 pts in Graph-BROS's favour). UTL spread is
  driven by the fold (83 / 72 / 58), not the seed.
- Validation is drawn from training templates (UTL best-val 93–95% vs test 58–84%), so model selection rewards
  fitting seen templates.

**Conclusion:** do not run v2 as designed; first establish what the UTL errors actually are (studies 2–4).

**Evidence:** conversation analysis; `results/vrdu_track1_bros/metrics.json`.

---

## 2. Per-entity UTL failure analysis — CPU, 2026-10-28

**Question:** where exactly is the MTL–UTL gap?

**Method:** per-entity F1 for every UTL run of every model in `results/*/metrics.json`.

**Findings (Track-1 BROS, mean of 3 seeds):**

| Held-out template | `registration_num` | `file_date` | `registrant_name` | `foreign_principle_name` | `signer_name` |
|---|---|---|---|---|---|
| Seen (MTL) | ~97 | ~91 | ~95 | ~91 | ~87 |
| Amendment | 96.8 | 75.8 | 91.2 | **8.7** (n = 20) | 75.7 |
| Dissemination_Report | 97.6 | 84.0 | 87.7 | **5.7** | 51.2 |
| Short-Form | 97.4 | 91.5 | **3.3** | **0.0** | 82.6 |

- Two fields make up most of the gap: `foreign_principle_name` (0–13% on every unseen template, for every model)
  and `registrant_name` on Short-Form (precision ≈ recall ≈ 2%: predicted, but on the wrong text).
- Short, self-identifying fields (`registration_num`, `file_date`) transfer almost perfectly.

**Conclusion:** the failure is role/semantic confusion, not geometry; the direction-invariance premise of
Graph-BROS is not supported.

**Evidence:** `results/vrdu_track1_bros/metrics.json`, `results/vrdu_dann_*`, `results/vrdu_lilt`.

---

## 3. VRDU template inspection — CPU, 2026-10-28

**Question:** what in the data causes the two failures?

**Method:** read gold annotations and the surrounding OCR text in each template (`data/vrdu`).

**Findings:**
- **Short-Form decoy:** its top block reads `1. Name | 2. Registration No.` and holds the *individual filer's*
  name, the same wording and position as `1. Name of Registrant` in the other templates. The real registrant is
  under `8. What is the name and address of the primary registrant?`.
- **`foreign_principle_name` has one-template supervision per fold:** Amendment has only 20 instances, and the
  prompt is worded differently per template ("5. Name of foreign principal on whose behalf this material was
  transmitted" vs "10. List every foreign principal to whom you will render services…").

**Conclusion:** both failures need reading the field's question text, which spatial invariance cannot supply.

---

## 4. BROS error dump (diagnostic) — GPU (Kaggle T4 × 2), 2026-10-28

**Question:** what does Track-1 BROS actually predict for the failing fields?

**Setup:** Track-1 BROS retrained on the 3 UTL folds × seeds 13, 42; per-document gold/predicted spans dumped.
Notebook `notebooks/kaggle_entity_error_diagnostic.ipynb`; analysis `scripts/analyze_entity_errors.py`.

**Findings:**
- The rerun reproduced the original fold F1 exactly (82.8 / 84.4, 65.8 / 73.5, 58.1 / 57.6).
- Short-Form `registrant_name`: 1,565 wrong predictions; **1,481 (95%) sit in the item-1 filer block**, 519 have
  the same text as the document's signer; only 27 are near item 8.
- `foreign_principle_name`: **1,463 / 1,468 missed with almost no predictions** (never fires), not confused.
- ~150 `signer_title` false positives per Short-Form run are ticked role boxes ("director", "partner"), while
  Short-Form labels `signer_title` in only 3 documents (annotation artefact).
- Peak GPU memory 7.9 GB per run (two runs per T4 would not fit).

**Conclusion:** decoy and wording hypotheses (study 3) confirmed from the model's own predictions.

**Evidence:** `results/vrdu_track1_bros_errdump/` (commit `267bc11`); code `2f6db5f`.

---

## 5. NER-fused GNN audit — CPU, 2026-10-28

**Question:** why did "semantic matching with a frozen NER tagger" not help key-value linking?

**Findings:**
- The 8-D "NER" vector is a word-level regex (`classify_text_semantic_type`): `Registrant`, `Statement` →
  LOCATION (substring "st"); `Name`, `Date`, `Signature` → PERSON/ORG; all-caps names → TEXT.
- The node features come from the base LiLT checkpoint, not fine-tuned, carrying absolute coordinates.
- KV link labels are generated by a reading-order heuristic whose "key" token changes meaning per template
  (`registrant_name` key: `No.` 80% on Amendment, `registrant` 74% on Dissemination, `Name` 92% on Short-Form).
- The only real semantic model (Sentence-BERT) was a standalone demo cell, never connected to training or metrics.
- Even a perfect NER tagger would tag the decoy and the real registrant identically (both organisation/person names).

**Conclusion:** semantic matching was never actually tested; the 81 → 30 link-F1 drop is largely built into the
heuristic labels.

**Evidence:** `src/cache_fused_features.py`, `notebooks/kaggle_typed_delaunay_gnn.ipynb`.

---

## 6. Zero-shot v1: open-schema KV extraction, page 1 — GPU (Kaggle T4 × 2), 2026-10-28

**Question:** can an off-the-shelf model with no training and no field list find key-value pairs, and does it
avoid the BROS failures?

**Setup:** Qwen2.5-VL-3B (page image) vs Qwen2.5-3B (OCR lines with positions), same prompt, tagged output
`<key>…</key><value>…</value>`. FUNSD test (50 pages) + page 1 of 100 VRDU documents per registration template +
100 ad-buy. Keys mapped to VRDU fields by SBERT similarity to the field name (threshold 0.4, fixed a priori).
Notebook `notebooks/kaggle_zero_shot_kv_baseline.ipynb`; code `scripts/zero_shot_kv.py`. ~25 s/page.

**Results (after the two scorer fixes below):**

| | VLM (image) | Text LM |
|---|---|---|
| FUNSD pair F1 | **51.7** [41.6–61.6] | 30.1 [23.4–36.9] |
| Amendment (page 1) | 76.7 [71.9–81.3] | 69.4 [63.5–75.4] |
| Dissemination_Report | **79.8** [77.6–81.9] | 49.7 [43.7–55.3] |
| Short-Form | 27.2 [24.6–29.6] | 20.0 [16.6–23.6] |
| Ad-buy | 34.5 [31.5–37.6] | 25.4 [21.8–28.7] |

- Dissemination `foreign_principle_name` 84.1 (BROS UTL 5.6); the Short-Form decoy was **never** picked.
- Short-Form `foreign_principle_name`: value recall **98%** but F1 **2.2**; the right value is extracted under the
  right key text, but the key text doesn't map to the field name ⇒ **key→field naming is the bottleneck**.
- Page 1 covers only 60% (registration) / 46% (ad-buy) of gold values; 75% of ad-buy outputs hit the 768-token cap.

**Bugs found and fixed (results above are post-fix):**
- Parser required a `<pair>` wrapper the VLM often omits (37/50 FUNSD pages lost; VLM FUNSD 16.4 → 51.7). `0ba9126`.
- Ad-buy line-item fields were excluded from gold but still used as mapping targets, absorbing predictions. `f08b348`.

**Evidence:** `results/zero_shot_kv/` (raw outputs, metrics). Commits `cfd349c`, `cb13001`, `ae107fa`, `0ba9126`.

---

## 7. Zero-shot v2: whole documents + LM naming step — GPU (Kaggle T4 × 2), 2026-10-29

**Question:** does reading every page and naming each pair with an LM fix the bottlenecks of v1?

**Setup:** VLM only, same 400 VRDU documents as v1 + FUNSD; up to 4 pages per document (covers 100% / 97% of
gold); 2,048-token extraction; naming = one short prompt per pair, batched per page, prefilled with `<field>` and
one worked example, answering `<field>/<desc>/<value>` (splitting combined values); schema descriptions written by
the model from field names only. Two mapping modes: **key** (raw key text vs field name) and **named** (generated
name + description vs field name + description). 897 page images, 0 errors, ~4.9 GPU-hours of generation.
Notebook `notebooks/kaggle_zero_shot_kv_v2.ipynb`; code `scripts/zero_shot_kv_v2.py`.

**Results (micro F1, whole documents):**

| Template | key | named |
|---|---|---|
| Amendment | 70.5 [67.0–73.8] | 60.2 [57.0–63.5] |
| Dissemination_Report | 74.1 [72.1–76.3] | 67.9 [65.6–70.3] |
| Short-Form | 37.5 [34.8–40.0] | **48.5** [46.4–50.6] |
| Ad-buy | 33.0 [30.7–35.5] | 34.4 [32.4–36.4] |
| FUNSD pair F1 | 49.4 [37.3–61.5] | — |

- Reading all pages recovered off-page-1 fields: Amendment `file_date` 0 → 72.5, `signer_name` 0 → 68.0.
- Naming fixed the wording mismatch: Short-Form `foreign_principle_name` 4.3 → **85.1**; ad-buy `flight_to` 0 → 42.4
  (date ranges split); `agency` 22.9 → 50.6.
- Naming lowered precision where keys already matched (Amendment P 65 → 50%, recall unchanged): more items per field.
- Decoy only half-solved: the real registrant is found in 97/100 Short-Form documents, but the filer is *also*
  matched to registrant in 86.

**Process notes (prompt-format robustness on a 3B model):** a multi-item tagged answer format was not followed
(0/6 and 0/14 schema descriptions; 0 naming items); one short tagged answer per question with a prefilled tag
and a worked example was followed reliably (19/20 descriptions, clean naming output).

**Evidence:** `results/zero_shot_kv_v2/` (manifest, raw outputs, schema descriptions, metrics). Commits `cc5435b`
… `1c48717` (code), `f08b348` (results).

---

## 8. One-answer-per-field decoding (post-hoc) — CPU, 2026-10-29

**Question:** does limiting each field to its single best-matching candidate recover the precision lost in v2?

**Setup:** re-scoring of the v2 predictions; per (document, field) keep the highest-similarity candidate.
**Decided after seeing v2 results** ⇒ reported as a post-hoc rule, applied identically to both modes. Cost: only
28/802 ad-buy gold values (none in registration) have a second distinct value per document.
`scripts/zero_shot_kv_v2.py score --one-per-field`.

**Results (micro F1; default → one answer per field; all intervals non-overlapping with their default):**

| Template | key | named |
|---|---|---|
| Amendment | 70.5 → **82.8** [79.3–86.6] | 60.2 → 70.8 [67.0–74.6] |
| Dissemination_Report | 74.1 → **82.9** [80.7–85.2] | 67.9 → **80.9** [78.1–83.7] |
| Short-Form | 37.5 → 48.2 [44.8–51.9] | 48.5 → **71.3** [68.8–74.0] |
| Ad-buy | 33.0 → 47.6 [44.0–51.2] | 34.4 → **53.6** [49.3–57.6] |

- **Decoy solved:** Short-Form `registrant_name` (named) 33.2 → **96.5** [92.4–99.5]; filer picked in 86 → **0** docs.
- Which mode wins depends on wording: key on Amendment (labels already resemble field names), named on Short-Form
  and ad-buy, tie on Dissemination.
- Still weak (indicative vs BROS UTL): signature-block fields, e.g. Short-Form `signer_name` 53 vs 83.

**Evidence:** `results/zero_shot_kv_v2/metrics_v2_top1.json`; commit `71b09ca`.

---

## 9. Stage B: grounded, matched comparison VLM vs Track-1 BROS — CPU, 2026-10-29

**Question:** on the *same* unseen-template documents, with the *same* gold spans and metric, does the zero-shot
VLM beat template-trained BROS?

**Setup:** `scripts/matched_eval.py`. Per UTL fold, the VLM's 100 sampled documents of the held-out template (all
inside BROS's test set). Gold and BROS predictions from the study-4 dumps (seeds 13, 42). VLM values (one answer
per field, key and named modes) grounded to word spans: exact match on the extraction page → anywhere → fuzzy
(token F1 ≥ 0.5); one span per value, because VRDU annotates one occurrence per entity (repeated text annotated
once in 190/1,437 entities; the annotated copy is first in 93, last in 84). Exact entity-span micro F1 (= seqeval
strict); paired bootstrap over documents for VLM − mean(BROS seeds). Also an "any occurrence" upper bound for the
VLM (correct text for the field, wherever the gold copy sits). Rule fixed before grounded results were seen.

**Validity checks:** recomputing BROS's F1 from its dumped spans reproduces all 6 reported fold F1s exactly, so
the span metric equals the seqeval metric BROS was evaluated with. 100 matched documents per fold. Grounding
placed 1,183/1,258 (key) and 1,312/1,405 (named) VLM values; the rest found no match ≥ 0.5 token F1.

**Results (exact-span micro F1, same 100 documents per fold; Δ = VLM − mean of BROS seeds, paired bootstrap):**

| Held-out template | BROS s13 / s42 | VLM key | Δ key | VLM named | Δ named |
|---|---|---|---|---|---|
| Amendment | 83.6 / 84.2 | 72.2 [68.2–75.8] | −11.6 [−16.0, −7.3] | 65.3 [61.3–69.2] | −18.6 [−22.9, −14.0] |
| Dissemination_Report | 62.8 / 70.0 | 67.6 [64.9–70.5] | +1.4 [−2.3, +4.9] | 62.2 [59.2–65.5] | −4.0 [−8.0, −0.5] |
| Short-Form | 60.2 / 60.1 | 33.4 [30.5–36.7] | −26.7 [−30.5, −22.9] | 56.6 [53.2–60.4] | −3.4 [−7.6, +0.5] |
| **Pooled (300 docs)** | 68.6 / 70.9 | 57.9 [55.2–60.6] | [−14.5, −9.3] | 61.2 [59.1–63.3] | [−11.1, −6.0] |

"Any occurrence" upper bound (named, pooled): 64.1 [62.2–66.1], Δ [−8.2, −3.2]: location explains little of the gap.

**Per-field: the errors are complementary** (exact-span F1, BROS seed 13 vs VLM named):

| Field | Amendment | Dissemination | Short-Form |
|---|---|---|---|
| `foreign_principle_name` | 0 vs **50** | 0 vs **61** | 0 vs **58** |
| `registrant_name` | 86 vs 88 | 78 vs 76 | 2 vs **62** |
| `signer_name` | **79** vs 36 | 31 vs 36 | **85** vs 23 |
| `file_date` | **76** vs 43 | 76 vs 78 | **91** vs 51 |
| `signer_title` | **71** vs 14 | **47** vs 6 | — |

**Where the VLM loses points (named mode, F1 under increasingly strict matching):**

| Field | text match (token F1 ≥ 0.5) | exact text, any location | exact span |
|---|---|---|---|
| Short-Form `registrant_name` | 96.5 | 67.7 | 61.6 |
| Short-Form `foreign_principle_name` | 87.5 | 76.3 | 57.8 |
| Dissemination `registrant_name` | 100.0 | 76.4 | 76.4 |

The largest drop is **span boundaries** (the VLM finds the right entity but its text extent differs from the
annotation convention, e.g. extra words or a combined address), then location.

**Conclusions:**
- Under the standard, strict matched metric, template-trained BROS beats the zero-shot VLM overall (pooled
  Δ −6 to −15); on Dissemination and Short-Form the better VLM mode ties BROS (Δ intervals include 0).
- Study 8's "VLM matches or beats BROS" was indicative only; it held under lenient text matching, not exact spans.
- The two systems fail on different fields: the VLM solves the role/wording fields BROS cannot
  (`foreign_principle_name`, the Short-Form decoy), BROS wins signature-block and date fields and knows VRDU's span
  boundaries. ⇒ A hybrid (VLM for which entity, a trained span model for its boundaries) is the natural next step.
- Caveats: VLM mode choice (key vs named) differs by template; picking per template after seeing results would be
  test-set selection. MTL-side comparison not run (BROS per-document MTL predictions were never dumped).

**Evidence:** `results/matched_eval/metrics.json`; `scripts/matched_eval.py` (selftest: `python scripts/matched_eval.py selftest`).

---

## 10. Hybrid feasibility: oracle, fill-in, cross-template router — CPU, 2026-10-29

**Question:** study 9 found complementary errors. Is a BROS + VLM hybrid worth pursuing, and can its gain be
captured without looking at the test template?

**Setup:** `scripts/hybrid_eval.py` on the per-document spans saved by `scripts/matched_eval.py` (same 300
documents, gold, and exact-span F1 as study 9). Each hybrid is built with BROS seed 13 and seed 42 and compared
with the same seed; paired bootstrap over documents. **Rules and pass criterion registered here before running:**
- **Oracle** (upper bound, uses gold): per (document, field), the source with more correct spans.
- **Fill-in** (no parameters): BROS spans, plus VLM spans for fields BROS predicted nothing for in that document;
  VLM mode (key / named) chosen by pooled VLM F1 on the other two templates.
- **Router** (leave-one-template-out): per field, the source (BROS, VLM key, VLM named) with the best pooled F1 on
  the other two templates, applied to the held-out template.
- **Pass:** fill-in or router beats BROS pooled with a 95% interval above 0, and no template's interval lies
  entirely below 0.

**Results (exact-span F1, mean of the two BROS seeds; Δ = hybrid − BROS, paired bootstrap 95% CI):**

| Template | BROS | Fill-in | Δ fill-in | Router | Δ router | Oracle (bound) | Δ oracle |
|---|---|---|---|---|---|---|---|
| Amendment | 83.9 | 82.3 | −1.5 [−2.6, −0.5] | 83.3 | −0.6 [−2.4, +1.3] | 91.0 | +7.2 [+5.3, +9.1] |
| Dissemination_Report | 66.4 | 71.8 | **+5.5** [+3.7, +7.3] | 70.0 | +3.6 [+1.6, +5.5] | 83.2 | +16.9 [+14.5, +19.1] |
| Short-Form | 60.1 | 60.2 | +0.1 [−0.6, +0.8] | 49.2 | −10.9 [−13.6, −8.0] | 84.6 | +24.5 [+21.8, +27.2] |
| **Pooled (300 docs)** | 69.8 | 71.4 | **+1.6** [+0.7, +2.5] | 67.4 | −2.3 [−3.8, −0.9] | **85.9** | **+16.2** [+14.7, +17.8] |

**Verdict under the pre-registered rule:** neither usable hybrid passes. Fill-in gains significantly pooled
(+1.6) but is significantly worse on Amendment (−1.5); the router loses pooled and badly on Short-Form.

**Why (from the recorded decisions):**
- **The VLM mode choice does not transfer across templates.** Both rules pick the VLM mode (key vs named) on the
  other two templates. For Short-Form, Amendment + Dissemination favour *key*, but Short-Form needs *named*
  (study 8: key 48.2 vs named 71.3). So fill-in adds key-mode `foreign_principle_name` on Short-Form (4% F1
  there) instead of named (58%), and the router routes registrant/file_date to the wrong mode (−10.9).
- **BROS's Short-Form errors are confident wrong predictions, not abstentions.** It predicts the decoy as
  registrant, so fill-in (which only fills fields BROS left empty) cannot replace it.
- **On Amendment, fill-in's extra VLM spans are mostly false positives** where BROS abstained correctly.

**Conclusions:**
- **The hybrid is worth pursuing:** a perfect per-field choice would gain +16 F1 pooled (+24.5 on Short-Form).
  The gain exists, but the simple rules here can't capture it on an unseen template.
- What is missing is (1) a VLM mapping that needs no per-template mode choice, and (2) a signal for when BROS is
  confidently wrong (agreement between the two systems, or BROS token confidences, which the dumps don't contain).
- Caveat: studies 8–10 all evaluate post-hoc choices on the same 300 documents. Before any final hybrid claim, a
  confirmatory run on documents not used in these studies is needed (the UTL test sets contain 1,615 other
  documents with BROS predictions already dumped; only the VLM would need to run on them).

**Evidence:** `results/hybrid_eval/metrics.json` (includes every per-fold decision); `results/matched_eval/spans.jsonl`.

---

## 11. Mode-free VLM mapping, hybrids re-tested — CPU, 2026-10-29

**Question:** study 10 failed partly because the VLM key/named mode chosen on two templates was wrong for the
third. Does a mapping that needs no mode choice make the hybrid rules work?

**Setup (registered before running):** `vlm_both` = candidates from key mode (raw key vs field name) and named
mode (generated name + description vs field name + description), each scored in its own mode, pooled per field;
under one-answer-per-field the single most similar candidate wins regardless of mode (ties: key).
`assign_both` in `scripts/zero_shot_kv_v2.py`; grounded and scored by `scripts/matched_eval.py`; hybrids
`fill_in_both` and `router_both` (router chooses per field between BROS and `vlm_both` on the other two
templates) in `scripts/hybrid_eval.py`. **Same pass rule as study 10.** Caveat stated up front: this is another
post-hoc variant on the same 300 documents as studies 8–10.

**Validity checks:** all study 9 and study 10 numbers re-computed identically (sanity F1s, every system).

**VLM alone (exact-span F1):** the mode-free mapping is *not* better than the single modes. Pooled 57.2
[54.8–59.7] vs key 57.9 and named 61.2. On Short-Form it scores 41.6, between key (33.4) and named (56.6):
pooling candidates lets the raw-key "Name" (the filer) out-match the named item-8 registrant, so Short-Form
`registrant_name` falls to 6.0 (named alone: 61.6). **The decoy comes back** under max-similarity pooling.

**Hybrids (Δ vs BROS, mean of seeds, paired bootstrap 95% CI):**

| Template | BROS | Fill-in (both) | Δ | **Router (both)** | **Δ** |
|---|---|---|---|---|---|
| Amendment | 83.9 | 82.2 | −1.6 [−2.7, −0.5] | 83.2 | −0.6 [−2.5, +1.2] |
| Dissemination_Report | 66.4 | 72.2 | +5.9 [+4.2, +7.6] | 73.0 | **+6.6** [+5.0, +8.4] |
| Short-Form | 60.1 | 61.0 | +0.9 [−1.2, +3.0] | 66.1 | **+6.0** [+3.6, +8.3] |
| **Pooled (300 docs)** | 69.8 | 71.2 | +1.5 [+0.4, +2.5] | **73.6** | **+3.9** [+2.8, +5.1] |

**Verdict under the pre-registered rule: `router_both` passes** (pooled gain significant; no template
significantly worse). Fill-in (both) still fails (significantly worse on Amendment).

**What the router learned** (fitted on two templates, applied to the third; identical across all three folds
for both seeds, apart from `registration_num` on Amendment): **VLM for `foreign_principle_name` and
`registrant_name`; BROS for `file_date`, `signer_name`, `signer_title`** (and `registration_num`). That is the
division suggested by studies 9–10: the VLM for role/wording fields, BROS for layout-regular fields.

**Conclusions:**
- A simple, interpretable hybrid (field-level routing learned on other templates) beats template-trained BROS on
  unseen templates: **+3.9 F1 pooled**, +6.6 on Dissemination and +6.0 on Short-Form, no significant loss on
  Amendment. Most of the gain is `foreign_principle_name`, which BROS never predicts.
- It still captures only a quarter of the oracle ceiling (+16.2): Short-Form registrant stays poor (the mode-free
  mapping reintroduces the decoy), and signature-block fields remain BROS-only.
- **This is the fourth post-hoc variant on the same 300 documents** (studies 8–11). The result is a hypothesis
  until a confirmatory run on fresh documents (the 1,615 unused UTL test documents) with every rule frozen as
  is, so it must not be reported as a final number yet.

**Evidence:** `results/matched_eval/metrics.json` (adds `vlm_both`), `results/matched_eval/spans.jsonl`,
`results/hybrid_eval/metrics.json` (verdicts and per-fold router decisions).

---

## 12. Confirmation of the study-11 hybrid on fresh documents — GPU (Kaggle T4 × 2), run 2026-10-30

**Question:** does the study-11 router (VLM for role fields, BROS for the rest; +3.9 F1 on the 300 documents it
was chosen on) still beat Track-1 BROS on unseen-template documents never used before?

**Setup (frozen before the run):** 100 fresh documents per registration template (seed 1), excluding the 300 used
in studies 6–11; all inside the UTL test sets, whose BROS predictions are already dumped (study 4). The VLM
pipeline is unchanged from study 7 (same prompts and code, schema descriptions copied from v2), with
one-answer-per-field, the mode-free `vlm_both` mapping, the study-9 grounding, and the study-10 pass rule.
- **Primary:** the study-11 per-template router tables (each learned on the other two templates' old documents),
  applied unchanged.
- **Secondary:** the same routing procedure re-fitted on the fresh documents of the other two templates.
- **Validity checks:** BROS on the fresh sample vs its full test-set F1; VLM on the fresh sample vs the old sample.

Notebook `notebooks/kaggle_confirm_hybrid.ipynb`; scoring `scripts/confirm_eval.py` (a dry run on the old spans
reproduces study 11 exactly).

**Run:** 609 page images, 300/300 documents complete, 0 errors, ~20–25 s/page. The Kaggle push failed
(expired token); the output files were downloaded and everything below was re-checked locally.

**Validity (all passed):**
- All 300 documents are new (0 overlap with the studies 6–11 sample); `confirm.json` recomputes identically from
  the saved spans.
- BROS on the fresh sample: F1 lies inside the fresh 95% interval of its full-test-set F1 for all 6 runs
  (e.g. Dissemination 65.7 vs 65.8 and 73.5 vs 73.5), so the sample is representative.
- VLM on the fresh sample overlaps the old sample's interval for all 9 (template × mode) cases
  (e.g. Short-Form `vlm_both` 40.4 vs 41.6), so the pipeline behaved as in study 11.

**Results (exact-span F1, mean of the two BROS seeds; Δ = router − BROS, paired bootstrap 95% CI):**

| Template | BROS | Router (primary) | Δ primary | Δ secondary |
|---|---|---|---|---|
| Amendment | 85.2 | 84.1 | −1.0 [−2.7, +0.5] | −0.3 [−1.8, +1.3] |
| Dissemination_Report | 69.6 | 75.9 | **+6.3** [+4.5, +8.0] | **+6.3** [+4.5, +8.0] |
| Short-Form | 57.2 | 63.6 | **+6.4** [+4.1, +8.8] | **+6.4** [+4.1, +8.8] |
| **Pooled (300 new docs)** | 70.5 | **74.3** | **+3.8** [+2.6, +5.0] | **+4.0** [+2.9, +5.2] |

Study 11 on the original 300 documents: +3.9 [+2.8, +5.1]. **The effect replicated almost exactly.**

**Verdict:** the pre-registered rule passes for both the primary (frozen tables) and the secondary (router
re-fitted on the fresh other-template documents): pooled gain significant, no template significantly worse.

**Where the gain comes from (pooled, primary router, exact-span F1):**

| Field | BROS | Router | Δ |
|---|---|---|---|
| `foreign_principle_name` | 5.1 | 55.3 | **+50.2** |
| `registrant_name` | 59.6 | 63.6 | +3.9 |
| `registration_num` | 97.8 | 96.8 | −1.0 |
| `file_date`, `signer_name`, `signer_title` | (BROS) | (BROS) | 0.0 |

**Conclusions:**
- **The hybrid is confirmed:** a per-field router learned on other templates, sending `foreign_principle_name`
  and `registrant_name` to the zero-shot VLM and the other fields to template-trained BROS, beats BROS by about
  +4 F1 on unseen templates, on documents not used to choose any rule.
- **The gain is essentially one field.** `foreign_principle_name` goes from 5% to 55% (BROS almost never
  predicts it); `registrant_name` adds only +3.9 (Short-Form registrant stays weak after grounding and
  span-boundary mismatch, study 9). The gap to the study-10 oracle (+16) is mostly still open.
- **The router is almost a two-field switch:** the re-fitted secondary tables equal the frozen primary tables in
  5 of 6 fold × seed cases, so the secondary result shows the procedure is stable, but it is not an
  independent test of a richer rule.

**Limits:** the same 3 VRDU registration templates (new documents, not new templates); ad-buy and FUNSD not
included; BROS is the N = 200 Track-1 model. The study-10 oracle ceiling was not re-measured on the fresh documents.

**Evidence:** `results/confirm_v2/` (manifest, raw outputs, schema descriptions, grounded spans, `confirm.json`).
Notebook `notebooks/kaggle_confirm_hybrid.ipynb`; scoring `scripts/confirm_eval.py`.

---

## 13. Dataset audit: can the held-out-template protocol be reproduced elsewhere? — CPU, 2026-10-30

**Question:** the hybrid (studies 10–12) was only tested on 3 VRDU registration templates. Which public dataset
supports a leave-one-template-out test with a named-field schema and gold annotations?

**Requirements:** known template labels; ≥ 4 templates (so each fold's router sees ≥ 3 others); a shared named
field schema; gold annotations that can be scored against both BROS tokens and VLM output.

**Candidates checked (data downloaded to `data/`, gitignored; `scripts/audit_fatura.py`):**

| Dataset | Result |
|---|---|
| **VRDU ad-buy** (already local) | **Rejected.** No template labels; the three official unknown-template test sets share ~95% of their documents (146 of ~155), so they cannot serve as folds. LiLT's gap there is only 1.85 F1 (UTL 90.6), so little to fix. |
| **FATURA** (Zenodo 10371464, CC-BY-4.0, 691 MB, 10,000 invoices) | **Compatible with adaptations**, see below. |
| CORD-v2 (Hugging Face, CC-BY-4.0) | Rejected: no template/merchant labels; nested menu-item schema. |
| SROIE, WildReceipt (Hugging Face) | Rejected: the hub entries are loading scripts with no data. |
| FUNSD / XFUND | Rejected: question/answer labels only, no named-field schema or template labels. |

**FATURA findings:**
- **Templates:** 50 templates × 200 instances (all IDs 1–50, uniform), 595 × 841 px images, and an official
  template-disjoint split (`Strat2_Split.txt`: 40 train / 10 test templates, disjoint).
- **Shared schema:** 31 field classes with text, but only **9 core fields exist in ≥ 29 of 50 templates**: DATE (49),
  NUMBER (44), SELLER_ADDRESS (41), TOTAL (40), TITLE (37), SELLER_NAME (34), SUB_TOTAL (34), BUYER (31),
  DUE_DATE (29). The rest are rarer (e.g. DISCOUNT 12, GSTIN 7).
- **Wording varies by template, as in VRDU:** DATE is `Date:` or `Invoice Date:`; NUMBER is `INVOICE #` or `INVOICE ID`;
  BUYER is `Bill to:` or `Buyer :`, and the similar blocks `BILL_TO` / `SEND_TO` are separate classes.
- **Gold is block-level, and the block text includes its label** (`"TOTAL : 734.33 EUR"`, `"Date: 20-Mar-2008"`),
  one box per class per document (mean 5.4 words per block). There are no word-level boxes, and no unlabelled
  words other than the TABLE/LOGO regions. VRDU gold is value-only with word-level boxes, so scoring must be at
  block level (predicted value → grounded words → enclosing block).
- **The Hugging Face-format files are unusable for the schema:** tag id 13 lumps 21 classes (SELLER_ADDRESS, TITLE,
  SUB_TOTAL, NOTE, TAX, …). Tokens must be rebuilt from `Original_Format` (all classes, block text split on
  whitespace, block box). The conversion is exact: HF boxes equal the original boxes with y flipped
  (y_img = 841 − y_orig), matched for 100% of 300,000+ blocks.
- **Risk:** data is synthetic (Faker values, clean rendering), so scores may saturate for both systems.

**Conclusion:** FATURA is the only checked dataset that supports leave-one-template-out with a shared field
schema, with far more templates (50) than VRDU registration (3), which would also allow k-fold template groups
(e.g. 5 folds × 10 templates). Required adaptations: tokens from `Original_Format`; block-level scoring;
restrict to the 9 core fields; BROS sees block-level boxes (a coarser layout signal than VRDU).

**Evidence:** `results/fatura_audit/summary.json`; `scripts/audit_fatura.py`.

---

## 14. BROS + zero-shot VLM hybrid on a new dataset: FATURA, held-out template folds — GPU (Kaggle T4 × 2), registered 2026-10-30

**Question:** does the study 10–12 hybrid (a per-field router between template-trained BROS and a zero-shot VLM)
still beat BROS on unseen templates when the dataset, schema and layout style all change?

**Setup (registered before running; code in `src/data/fatura_loader.py`, `scripts/fatura_bros.py`,
`scripts/fatura_eval.py`; notebook `notebooks/kaggle_fatura_hybrid.ipynb`; dataset audit in study 13):**
- **Data:** FATURA, 50 templates × 200 invoices. Tokens rebuilt from `Original_Format` (words of each text block
  share the block's box; verified against the independent HF files on 2,734 blocks); 9 core fields present in ≥ 29
  templates: buyer, date, due_date, number, seller_address, seller_name, sub_total, title, total.
- **Folds:** 5 × 10 templates. Fold 0 = the dataset's official unseen-template test set (templates 2, 4, 7, 10, 12,
  25, 32, 39, 49, 50); folds 1–4 = the other 40 templates shuffled with seed 0.
- **BROS (trained):** Track-1 recipe unchanged (`configs/fatura_bros.yaml`). Per fold and seed (13, 42): 200 training
  invoices spread evenly over the 40 seen templates, 50 validation invoices from the same seen templates (disjoint),
  test on all 2,000 invoices of the 10 held-out templates. 10 runs. No held-out-template invoice is ever used.
- **VLM (zero-shot, frozen from study 7):** same extraction and per-pair naming prompts and code, schema
  descriptions written by the model from the 9 field names alone, one answer per field, mode-free `vlm_both`
  mapping. 20 invoices per template (seed 1) = 1,000 pages: 10 *dev* (debugging the new block-level scoring only)
  and 10 *confirm* (all claims) per template.
- **Block-level scoring (new for this dataset, since gold blocks include their label):** a VLM value is grounded to
  words (exact match, else fuzzy ≥ 0.5 token F1) and then to the whole text block containing it; one field per
  block (higher similarity wins). Metric: exact block-span micro F1. BROS spans are scored as emitted.
- **Router:** per field, BROS or `vlm_both`, whichever has the higher pooled F1 on the *other four folds'* invoices
  (dev and confirm); applied to the held-out fold's confirm invoices. Also fill-in (BROS + VLM blocks for fields BROS
  left empty), the oracle ceiling, and the VLM alone.
- **Intervals:** the primary interval resamples *templates* (clusters of 10 per fold, 50 pooled); invoice-level
  intervals are reported alongside for comparison with studies 8–12.
- **Pass (registered):** pooled router − BROS (mean of the two seeds) has a 95% template-bootstrap interval above 0,
  and no fold's interval lies entirely below 0.
- **Validity checks:** BROS's F1 on the confirm invoices vs its F1 on all 2,000 held-out invoices; BROS dump gold
  equals the loader gold; VLM dev vs confirm F1.

**Known risks stated up front:** FATURA is synthetic (generated values, clean rendering), so BROS and/or the VLM may
saturate, leaving the router little to gain; BROS sees block-level boxes only; five folds of ten templates give a
coarser fold structure than VRDU's leave-one-template-out.

**Run:** 10 BROS runs (~390 s each) and 1,000 VLM pages (~38 s/page averaged over templates), 0 errors; 28
extraction and 175 naming outputs hit their token limits. The Kaggle push worked. Scores below are from a local
re-score (see "Process notes").

**Validity:**
- BROS on the confirm invoices vs all 2,000 held-out invoices: 99.6–100% vs 99.50–99.98% for all 10 runs. The
  interval is degenerate (collapses to 100%) because BROS is saturated, so the "inside interval" flag is
  uninformative here.
- The BROS dump's gold equals the loader's gold on all 2,000 documents (an order-sensitive comparison in my check
  first reported 2,000 mismatches; the dump lists spans sorted by field, the loader in reading order; fixed).
- VLM dev vs confirm F1 agree per fold (68.9/68.4, 66.0/62.8, 58.3/58.3, 51.6/54.1, 52.9/53.0).

**Results (exact block-span micro F1, 500 confirm invoices; Δ = system − BROS, template-bootstrap 95% CI):**

| System | Pooled F1 | Δ vs BROS [95% CI] |
|---|---|---|
| **BROS** (mean of 2 seeds) | **99.9** | — |
| **Router** | 99.9 | **+0.0** [0.0, 0.0] |
| Oracle (ceiling) | 99.9 | +0.0 [0.0, +0.1] |
| Fill-in | 97.6 | **−2.2** [−3.3, −1.3] |
| VLM alone (`vlm_both` / key / named) | 59.6 / 56.4 / 51.9 | −41 / −43 / −48 (all CIs below −36) |

Per fold: BROS 100.0 / 99.9 / 99.9 / 99.7 / 99.8; VLM `vlm_both` 68.4 / 62.8 / 58.3 / 54.1 / 53.0.
In every fold the router sent **no field to the VLM** (BROS had the higher pooled F1 on the other folds for all 9).

**Per field (pooled F1, BROS vs VLM):** date 100.0 vs 85.0, sub_total 100.0 vs 85.1, due_date 100.0 vs 76.8,
total 99.9 vs 68.2, seller_address 100.0 vs 66.5, buyer 99.1 vs 54.3, number 100.0 vs 42.5, seller_name 99.6 vs 14.2,
title 100.0 vs 0.0.

**Verdict under the registered rule: not passed** (pooled gain 0.0, interval [0.0, 0.0]; no fold significantly worse).

**Interpretation:**
- **A ceiling effect, as registered as a risk:** at 200 training invoices from 40 seen templates, BROS has no
  template-shift problem on FATURA (99.9% on unseen templates, against 58–85% on VRDU registration templates).
  The oracle shows nothing can be gained, so the experiment cannot confirm or refute the hybrid's value on a new
  schema. It is not evidence against it.
- **The router did no harm.** Fitted on other folds' invoices it learned that BROS beats the VLM on every field and
  reduced to BROS exactly. The naive fill-in lost 2.2 F1 (significant in all five folds): when BROS predicts
  nothing for a field it is usually because that template has no such block, and the VLM then fills a false one.
- **Why the VLM scores low (diagnostic, not a registered test):** it reads the content (91–97% of the gold block's
  words appear in its raw output for date, totals, number, due date, address) but the key-value formulation has no
  slot for blocks without a label: `title` 0.0 and `seller_name` 14.2 (only 26% of its words reach the extracted
  pairs; in the inspected example the company name is never output). Exact whole-block matching is also strict.
  A block-oriented prompt was not tested.
- **Generality of studies 10–12:** the VRDU hybrid gain applies where the trained model has a real template gap.
  Where it has none, the learned router degrades safely to the trained model.

**Process notes:** VLM-only scores differ by ≤ 0.32 points between the Kaggle scoring and the local re-score (sentence
embeddings on GPU vs CPU flip a few near-ties); router, fill-in, oracle, BROS numbers and the verdict are identical.
The committed `eval.json` is the local re-score.

**Next (not run):** make BROS data-starved so a template gap appears (few invoices and/or few seen templates) and
re-use the existing VLM outputs, so only cheap BROS runs are needed; a block-oriented VLM prompt; a dataset with real
scans (FATURA is synthetic).

**Evidence:** `results/fatura_v1/` (`eval.json`, `spans.jsonl`, `manifest.json`, raw VLM outputs, BROS runs and span
dumps); notebook `notebooks/kaggle_fatura_hybrid.ipynb`.
