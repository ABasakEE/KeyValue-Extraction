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
