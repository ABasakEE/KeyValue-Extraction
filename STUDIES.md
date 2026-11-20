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
| 15 | Router with data-starved BROS on FATURA (K = 1, 2, 5, 10 seen templates) | GPU | 2026-10-31 | Registered rule passes only at K=1, where the router merely matches the VLM alone; at K=2 it is significantly worse than BROS (-2.6); K>=5 neutral. Oracle headroom exists but a static per-field router captures none |
| 16 | Per-document routing signal (BROS confidence / endorsement) on FATURA and VRDU, plus seed replication | GPU | registered 2026-10-31 | **Registered criteria met, but not by the gate:** VRDU hybrid H (static router, then gate) +3.8 [+2.0, +5.6] over BROS (passes); the confidence gate alone is -0.2 [-0.6, +0.04] on VRDU and gains only at FATURA K = 2 (+1.5 [+0.1, +3.2]); the study-15 K = 2 loss of the static router did not replicate with 5 seeds (+2.3 [-2.1, +7.3]) |
| 17 | Link F1 of saved XFUND joint-LiLT checkpoints on held-out forms: truncation and windowing | CPU | 2026-10-31 | Reported 65.5 link F1 scores only the 61% of links visible in 512 tokens and gives the linker gold entities; counting cut-off links as misses gives 49.9; link-aware windows (510/128) at inference give 59.8. XFUND val is ~94% sibling-free, so siblings cost <1 point |
| 18 | Are BROS's and the VLM's errors correlated? (error-correlation check on the study-16 dumps) | CPU | 2026-10-31 | Yes, positively: the VLM is right on 32% of VRDU spans BROS misses vs 70% where BROS is right (lift 0.46 [0.43, 0.50]); FATURA lift 0.81 / 0.67 / 0.51 at K = 2 / 5 / 10. The VLM still rescues 25-42% of BROS misses, so a selector has something to find, but a confidence gate sees it only weakly |
| 19 | Windowed joint training and end-to-end (predicted-entity) XFUND link F1 | GPU | 2026-10-31 | **Windows help (registered criterion met):** end-to-end link F1 43.5 -> 47.7 (+4.2, all 3 seeds positive, windowed inference); entity F1 73.0 -> 75.9. Predicted entities cost ~22 points vs gold entities for the same checkpoint (seed 13: 60.9 -> 38.6); windowed inference over-predicts entities by ~21% |
| 20 | Cross-window entity deduplication for end-to-end XFUND linking | CPU | 2026-11-01 | **No better (registered criterion):** dedup raises entity F1 by +1.4 (precision 66.5 -> 70.6, recall -2.2) but link F1 changes by -0.35 (negative in all 3 seeds); with the edge rule kept it is +0.06. Over-prediction is only part of the entity problem: 14.4k entities still predicted vs 13.0k gold. Legacy checkpoints only; confirmation on study-19 models pending |
| 21 | Step-matched control for windowed joint training (is the study-19 gain just more gradient steps?) | GPU | 2026-11-01 | **Coverage matters (registered criterion met), but a quarter to a third of the gain was steps:** windows beat the control by +3.1 end-to-end link F1 (+4.0 / +3.4 / +1.8); extra steps alone gave the control +1.2 over base. Caveat: early stopping ended the control at 35-43 of 53 epochs, so it got 66-81% of the windows arm's steps |
| 22 | Where do gold-entity link errors come from? (step 0 for the encoder question) | CPU | 2026-11-01 | 72% of missed links point downward (recall 49 vs 79 for same-line partners); 30% of misses are question->question links (recall 34); 25% are unreachable by any window. Wrong partners are rarely the nearest candidate (10% at rank 0-1; 79% are far away or of the wrong type), so close-neighbour confusion is not the main error and a layout-attention encoder is not clearly indicated |
| 23 | Decode-time levers for gold-entity linking: threshold, one-to-one, type prior, nearness limit | CPU | 2026-11-01 | **Decode levers matter, modestly (registered criterion met):** the variant chosen on the selection split (threshold 0.4, hard type mask, one-to-one) gains +2.4 link F1 on the 250 forms (+1.9 / +2.2 / +3.2), mostly a precision gain (60.3 -> 67.7); it forbids every question->question link (recall 34 -> 0) and does not recover the downward links (recall 50 -> 46). **Correction (same day, see addendum): the gain comes mainly from forbidding type-violating pairs that touch gold `other` entities, which the end-to-end pipeline never creates; forbidding question->question alone costs 2.6 points. End-to-end emulation: forbidding `header` sources +1.9 (structure mode), question->answer only +8.2 (key-value mode, gold without question->question)** |
| 24 | Multilingual text stream for joint XFUND linking: LiLT-InfoXLM vs LiLT-en, end to end | GPU | 2026-11-14 | **Worse (registered rule):** end-to-end link F1 47.7 -> 44.2 (-3.5, negative in all 3 seeds), entity F1 75.9 -> 73.2; single-window link F1 is unchanged (41.7 vs 41.9), so the loss is in windowed inference. The multilingual stream did not help |
| 25 | Schema-guided, grounded VLM extraction (LMDX-style) on VRDU unseen templates, with a pixel ablation (3B; Qwen3-VL-8B arm added 2026-11-18) | GPU | run finished 2026-11-18 (registered 2026-11-05) | **Not an improvement (registered rule):** line-id JSON format failed for both models (fell back to one call per field); 8B A2 60.2 vs reference 57.7 (+2.5 [0.0, 5.2], below +3.0) and about 10 below BROS (70.5); 3B A2 scored 4.0 only because of a decode flaw (post hoc fallback: 56.6). Post hoc: image +13.7 (3B) / +3.3 (8B); static router with 8B spans 76.2 vs BROS 70.5 (+5.7 [+4.4, +7.0]); snapping to BROS 60.5 -> 65.5 |
| 26 | Table-type links: link-preserving crops and relation-rich synthetic pretraining on XFUND, end to end | GPU | 2026-11-09 | **Crops help (registered criterion met):** link F1 47.4 -> 49.7 (+1.4 / +2.3 / +3.2); gains sit on partner-above (+4.0), partner-below and far-partner links; synthetic pretraining alone is no better (+0.1); both together 49.9. Crops get ~2x the steps (confound) and lower link F1 on synthetic held-out families |
| 27 | Zero-shot VLM (Qwen2.5-VL) key-value linking on XFUND, scored with the LiLT end-to-end scorer | GPU | registered 2026-11-13 | *pending (Kaggle run)* |
| 29 | LoRA-tuned VLM with grounded word-span output: VRDU unseen- vs mixed-template learning (Track E, primary) and XFUND end-to-end linking (Track L) | GPU | run finished (sessions A and B; registered 2026-11-15) | **Worse (registered rule):** tuned VLM on unseen templates scores 46.2 exact-span F1 (replication 44.7) against BROS 70.1, i.e. -23.9 [-25.9, -21.6], negative on all three templates and in both seeds; it is also 11 points below the zero-shot pipeline of studies 8-12 (57.4). Mixed-template F1 is 76.1, so the seen/unseen gap is 29.9 (rule: <= 10.5, BROS 21.1). About 25% of gold fields are found with the wrong boundaries and 22-30% are not predicted at all. E3 (no image) not run; Track L not built |
| 30 | Output format of the tuned VLM on VRDU unseen templates: word numbers (study 29) vs OCR-segment numbers vs copied text with line identifiers | GPU | registered 2026-11-16 | *pending (Kaggle run)* |
| 31 | Prompt v2 (copy only the value, id example, three rewritten field descriptions) for the schema-guided grounded extractor, Qwen3-VL-8B 4-bit, VRDU unseen templates, confirm set | GPU | confirm and dev runs finished (registered 2026-11-19) | **Helps, clears the bar, and replicates (registered rules):** V2A2 76.5 (confirm) and 75.2 (dev) exact-span F1 vs 60.5 / 58.1 for study 25's prompt (+16.0 and +17.0, positive on all templates); router with BROS 79.2 vs 70.5 (+8.7 [+6.9, +10.6]); image adds +3.2 / +3.4; text-only already 73.3 / 71.7 (BROS 70.5 / 69.8). Still below BROS on Amendment; `foreign_principle_name` regressed (70 -> 52-57) and `signer_name` is below BROS |

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

---

## 15. Router with data-starved BROS on FATURA — GPU (Kaggle T4 × 2), registered 2026-10-31

**Question:** study 14 found no headroom for a hybrid because BROS scored 99.9% on unseen templates with 200 training
invoices from 40 templates. If BROS has seen few templates, a template gap should appear: does the router then beat
BROS, as it did on VRDU (studies 10–12)?

**Setup (registered before running; `scripts/fatura_bros.py --k-list`, `scripts/fatura_sweep.py`, notebook
`notebooks/kaggle_fatura_sweep.ipynb`):**
- **BROS (trained):** Track-1 recipe unchanged, trained on **K = 1, 2, 5, 10 seen templates** (chosen at random from
  the fold's 40 seen templates by seed), **10 invoices per template** (10, 20, 50, 100 invoices), 50 validation
  invoices from the same seen templates (disjoint), seeds 13 and 42, the 5 folds of study 14: 40 runs. Tested on the 200
  evaluated invoices (dev + confirm) of each fold's 10 held-out templates. K = 40 with 200 invoices is study 14.
- **VLM:** not re-run; the study-14 VLM outputs (`results/fatura_v1/spans.jsonl`) are re-used unchanged.
- **Scoring (CPU):** the study-14 analysis per K: router fitted on the other four folds' invoices (dev + confirm),
  applied to the held-out fold's confirm invoices; exact block-span F1; template-bootstrap intervals; oracle,
  fill-in, VLM alone.
- **Registered success rule:** at least one K with BROS pooled F1 below 90% (a template gap) where the router beats BROS
  (pooled 95% template-bootstrap interval above 0, and no fold's interval entirely below 0). If BROS stays at or above
  90% for every K, FATURA cannot test the hybrid. Per-K and per-field detail is descriptive.
- **Pipeline check done before the run:** with fake BROS dumps whose weaknesses were planted (K = 1 never predicts
  date, due_date, total; K = 2 never predicts date) the scoring routed exactly those fields to the real VLM and
  reported a positive, significant gain; with perfect fake BROS (K = 5, 10) it reported 0.0.

**Limits stated up front:** FATURA is synthetic; the choice of which K templates BROS sees varies only over 2 seeds
(template-bootstrap intervals cover the held-out templates, not the seen-template draw); BROS sees block-level boxes
only; the VLM's key-value formulation cannot express unlabeled blocks (study 14), which caps what the router can gain on
`title` and `seller_name`.

**Run:** all 40 BROS runs completed; 0 alignment mismatches between the dumps and the loader gold; 1,000 invoices
scored per setting. The scoring dry run with planted faults (above) behaved as the real run did.

**Results (exact block-span micro F1, 500 confirm invoices; Δ = router − BROS, template-bootstrap 95% CI):**

| Seen templates K | Train invoices | BROS F1 | Router F1 | Δ vs BROS [95% CI] | Registered "gap" (BROS < 90%) | Router passes | Oracle − BROS | Fill-in − BROS |
|---|---|---|---|---|---|---|---|---|
| 1 | 10 | **32.5** | 58.5 | **+26.0** [+19.8, +31.9] | yes | **yes** | +39.0 | +25.6 |
| 2 | 20 | 69.2 | 66.6 | **−2.6** [−5.0, −0.5] | yes | **no (worse)** | +16.0 | +2.0 |
| 5 | 50 | 90.8 | 90.8 | −0.0 [−0.4, +0.4] | no (just above) | no | +5.1 | −1.2 |
| 10 | 100 | 95.9 | 95.9 | +0.0 [0.0, 0.0] | no | no | +2.3 | −1.8 |
| 40 (study 14) | 200 | 99.9 | 99.9 | +0.0 [0.0, 0.0] | no | no | +0.0 | −2.2 |

VLM alone: 59.6 at every K (its outputs are fixed). **Registered rule: satisfied** (K = 1 has a template gap and the
router passes: `helpful_in_starved_regime = true`).

**Exploratory post-hoc analysis (written after seeing the results; `scripts/fatura_sweep_posthoc.py`,
`results/fatura_v2_sweep/posthoc.json`):**

| K | BROS | VLM alone | Router | Router − VLM alone [95% CI] | Better single system | Router share of oracle headroom | BROS per-run F1 min / max / std |
|---|---|---|---|---|---|---|---|
| 1 | 32.5 | 59.6 | 58.5 | −1.1 [−2.5, +0.4] | VLM | 67% | 0.0 / 57.1 / 19.8 |
| 2 | 69.2 | 59.6 | 66.6 | +6.9 [+2.9, +11.1] | BROS | −17% | 57.4 / 82.3 / 7.6 |
| 5 | 90.8 | 59.6 | 90.8 | +31.1 | BROS | 0% | 86.4 / 96.7 / 3.4 |
| 10 | 95.9 | 59.6 | 95.9 | +36.2 | BROS | 0% | 91.1 / 99.1 / 2.7 |
| 40 | 99.9 | 59.6 | 99.9 | +40.2 | BROS | 0% | 99.6 / 100.0 / 0.2 |

**What the results show:**
1. **BROS's unseen-template F1 is a smooth function of the number of seen templates** (32.5 → 69.2 → 90.8 → 95.9 → 99.9),
   and its run-to-run spread collapses with it (std 19.8 → 0.2). One BROS run trained on a single template scored 0.0.
2. **The K = 1 "pass" is the VLM beating a near-useless BROS, not a hybrid advantage.** The router sends 7–8 of 9 fields
   to the VLM and ends at 58.5, indistinguishable from the VLM alone (−1.1 [−2.5, +0.4]). It does not beat the better single
   system at any K; at K = 2 it is worse than BROS and at K ≥ 5 it equals BROS.
3. **At K = 2, where BROS has a clear gap (69.2%) the router hurts (−2.6, interval entirely below 0).** There BROS and the VLM
   are close on most fields (e.g. date 79.3 vs 85.0, due_date 77.8 vs 76.8, total 68.3 vs 68.2, buyer 52.3 vs 54.3) and BROS's
   quality depends on which two templates it saw, so field choices fitted on other folds transfer poorly (different
   field sets are routed in different folds).
4. **Headroom exists, but a static per-field router captures none of it** (oracle +16.0, +5.1, +2.3 at K = 2, 5, 10;
   router −17%, 0%, 0%). The complementarity is per document, not per field.
5. **Naive fill-in helps only when BROS is nearly broken** (+25.6 at K = 1) and hurts from K = 5 up (−1.2 to −2.2).

**Comparison with VRDU (studies 10–12):** at a similar coverage (VRDU: BROS ≈ 70% after seeing two templates; FATURA K = 2:
69.2%) the router helped on VRDU (+3.8 on fresh documents) but not on FATURA (−2.6). Interpretation (a hypothesis
consistent with the per-field tables, not tested directly): the VRDU gain came from fields where BROS fails and the
VLM wins *consistently across templates* (`foreign_principle_name` 5 → 55). FATURA's failures with few seen templates are
diffuse (every field degrades somewhat), the VLM is weaker on several fields (`title` 0.0, `seller_name` 14.2,
`number` 42.5), and no field is reliably VLM-favoured.

**Conclusions for the generalization claim:**
- The hybrid is **safe where the trained model is strong** (K ≥ 5: no harm, repeated across 5 folds and 50 templates) and
  **not reliable where it is mid-strength** (K = 2: significantly worse than BROS).
- The VRDU gain does **not** generalise as "a router helps whenever BROS is weak"; it needs a field that is consistently
  better handled by the zero-shot VLM. Report the VRDU result with that scope.
- Where BROS is almost useless (K = 1) the useful finding is that a zero-shot VLM (59.6) beats it, not that routing adds value.

**Limits:** FATURA is synthetic; 2 seeds, so which K templates BROS sees varies little and the intervals cover only the
held-out templates; the 90% gap threshold is arbitrary (K = 5 sits at 90.8%); the exploratory table was produced after
seeing the registered results.

**Next (not run):** a per-document routing signal (agreement between systems or BROS token confidences; BROS confidences
are not stored in the current dumps), a margin-based router that sends a field to the VLM only when its advantage on other
folds is clearly positive, and a real-scan dataset with a field the VLM handles consistently better than a trained model.

**Evidence:** `results/fatura_v2_sweep/` (`sweep.json`, `posthoc.json`, 40 BROS run directories with span dumps).

---

## 16. Per-document routing signal on FATURA and VRDU, with seed replication — GPU (Kaggle T4 × 2), registered 2026-10-31

**Questions:** (1) Study 15 left oracle headroom (+16.0 / +5.1 / +2.3 F1 at K = 2 / 5 / 10 seen templates on FATURA;
about +16 on VRDU) that a static per-field router captured none of. Does BROS's own per-word field probability give a
per-document signal that captures some of it, on both datasets? (2) Is the static router's K = 2 loss (−2.6) real, or noise
from which two templates BROS happened to see?

**Signals** (from BROS's soft-voted field probabilities, B and I tags merged; `test_probs.jsonl`, integer percents):
*confidence* = mean probability of field f over BROS's own predicted span; *endorsement* = mean probability of f over the
VLM's proposed span (how plausible BROS itself finds it).

**Rules (registered before any run; `scripts/routing_eval.py`):** with one VLM span per field (`vlm_both`):
- **G (gate):** if BROS predicted a span for f with confidence < τ, replace it with the VLM span; BROS silent → stay silent.
- **E (endorse):** G, plus fill a silent field with the VLM span when endorsement ≥ κ.
- **H (hybrid):** the static per-field router of studies 10–15 first, then G on the remaining fields. **Added because VRDU's
  main gain is a field BROS never predicts (`foreign_principle_name`), which G cannot touch by construction** (decided before
  any result, from the structure of studies 10–12).
- τ ∈ {.5,.6,.7,.8,.9,.95,.99}, κ ∈ {.1,.3,.5}, chosen per seed only on the **other** folds' documents (dev + confirm) by pooled
  F1; ties go to fewer replacements. Claims use the held-out fold's confirm documents only.

**Runs:**
- **FATURA:** BROS (Track-1 recipe unchanged) on K = 2 and 5 seen templates with 5 seeds (13, 42, 7, 21, 99) and K = 10 with seeds
  13, 42; 10 invoices per template; the 5 folds of study 14; tested on the 200 evaluated invoices per fold: 60 runs. VLM
  outputs re-used from study 14 (`results/fatura_v1/spans.jsonl`).
- **VRDU:** Track-1 BROS (`configs/vrdu_track1_bros_probs.yaml`, identical recipe, seeds 13, 42) on the 3 unseen-template folds, saving probabilities
  for the 600 VLM-evaluated documents: the 300 original (dev, used in studies 8–11) and the 300 fresh (confirm, study 12). VLM outputs
  re-used from studies 7 and 12.
- Smoke tests for both datasets (real BROS, one epoch, probability dump checked) run first in the notebook
  `notebooks/kaggle_routing_signal.ipynb`.

**Intervals:** two-level bootstrap, 1000 draws: seeds are resampled (the draw of seen templates / training run) and clusters
(FATURA: templates; VRDU: documents, since it has only 3 templates).

**Registered success criteria:**
- **FATURA, rule G:** pooled G − BROS lower interval bound > 0 at K = 2 or K = 5, **and** upper bound ≥ 0 at every K. Seed replication
  (question 2): the static router's K = 2 loss counts as replicated only if its interval upper bound is < 0 with the 5 seeds.
- **VRDU, rule H:** pooled H − BROS lower bound > 0 **and** no template's upper bound < 0.
- G, E and H are also reported on both datasets as descriptive results, with the static router, fill-in, the oracle and the VLM alone.

**Descriptive diagnostics (not criteria):** AUROC of BROS's confidence for "this span is correct"; share of oracle headroom
captured; per-seed deltas; chosen τ/κ; per-fold results.

**Checks done before the run:** unit test of the soft-vote and dump (`tests/test_predict_probs.py`); self-test of the rules,
AUROC and verdicts; a dry run on the real VLM outputs with fake BROS runs whose behaviour was planted (K = 2 gate +13.8, endorse
+17.5; K = 5, 10 exactly 0.0; VRDU hybrid +16.1; alignment 0 mismatches) and both command-line paths.

**Limits stated up front:** FATURA is synthetic; the *family* of rules was chosen after seeing studies 14–15 (the BROS runs, the
confidences and the parameter fits are new); BROS confidences under template shift may be poorly calibrated (the AUROC will
show it); 5 seeds is still small and VRDU has 2; VRDU's intervals cover documents of 3 templates, not new templates in
general; the VRDU dev documents were used when choosing among variants in earlier studies.

**Results (`results/routing_v3/{vrdu,fatura}.json`; Δ = system − BROS, two-level bootstrap 95% interval):**

| VRDU (600 docs, 2 seeds), BROS 70.5 | F1 | Δ vs BROS |
|---|---|---|
| static router | 74.5 | +4.0 [+2.2, +5.9] |
| gate G | 70.3 | **−0.2** [−0.6, +0.04] |
| endorse E | 70.9 | +0.4 [−0.4, +1.3] |
| **hybrid H (static, then gate)** | 74.3 | **+3.8** [+2.0, +5.6] |
| fill-in | 72.1 | +1.6 [+0.4, +2.9] |
| oracle | 81.1 | +10.6 [+8.8, +12.4] |
| VLM alone | 57.7 | −12.8 |

The static router and H capture about 38% and 36% of the oracle headroom; G captures none (−2%). BROS's confidence separates correct from wrong spans well (AUROC 0.86) but that does not translate into gains (see study 18).

| FATURA | K = 2 (5 seeds), BROS 66.6 | K = 5 (5 seeds), BROS 89.8 | K = 10 (2 seeds), BROS 95.9 |
|---|---|---|---|
| static router Δ | +2.3 [−2.1, +7.3] | −0.3 [−1.3, +0.2] | 0.0 |
| gate G Δ | **+1.5 [+0.1, +3.2]** | −0.1 [−0.7, +0.3] | −0.1 [−0.4, 0.0] |
| endorse E Δ | +3.0 [+1.3, +4.8] | 0.0 | −0.1 |
| fill-in Δ | +2.8 [+1.0, +4.6] | −1.1 [−1.8, −0.3] | −1.8 [−2.9, −0.8] |
| oracle Δ | +18.2 | +5.9 | +2.3 |
| AUROC of BROS confidence | 0.84 | 0.76 | 0.70 |

**Verdict (registered criteria):** VRDU rule H passes (pooled gain significant, no template significantly worse). FATURA rule G passes (lower bound > 0 at K = 2; never significantly worse at any K). Seed replication: the static router's K = 2 loss from study 15 is **not** replicated (`static_router_K2_loss_replicated = false`; with 5 seeds the point estimate is positive), so that loss was seed noise about which two templates BROS saw.

**Reading:** the VRDU gain is the static two-field routing (`foreign_principle_name`, `registrant_name`) that studies 10-12 already established; the per-document gate adds nothing there. The gate and endorsement rules help only in the one FATURA setting where BROS is mid-strength and diffusely wrong (K = 2) and are neutral or slightly harmful where BROS is strong. FATURA is synthetic and the 'passes' for G rests on a lower bound of +0.1.

**Evidence:** `results/routing_v3/` (`vrdu.json`, `fatura.json`, per-run probability dumps).

---

## 17. Link F1 of the saved XFUND joint-LiLT checkpoints on held-out forms — CPU, 2026-10-31

**Question:** how good is a real key-value *linker* on unseen forms, and how much of the reported 65.5 is an artefact of truncation or of the evaluation set-up?

**Data:** XFUND es/fr/de/it/pt val (250 documents, 9,064 unique gold links), from the Hugging Face mirror `FrancophonIA/XFUND` (the official GitHub release host failed TLS here); the visible-word count equals the `support` in the saved metrics (46,180). Checkpoints: `results/xfund_joint_lilt/seed_{13,21,42}`.

**Method (`scripts/xfund_link_eval.py`):** the linker is given the annotation's entity spans and types, as `predict_joint` does (so these are upper bounds for end-to-end linking). *single* = one 512-token window; *windowed* = entity-aligned 510-token windows, stride 128, predicted links unioned. Gold sets: *visible* (what metrics.json scored: 5,605 links) and *full* (all 9,062 links with distinct endpoints; cut-off links count as misses). Sibling-free = 234 of 250 val documents with no train document of question-text Jaccard >= 0.5 in the same language.

**Results (link F1, mean ± sd over 3 seeds):**

| Condition | All 250 docs | Sibling-free (234) |
|---|---|---|
| single window, visible gold (reproduces metrics.json: 65.5) | 65.5 ± 1.3 | 64.9 ± 1.4 |
| single window, **full gold** | **49.9 ± 0.9** | 49.0 ± 1.0 |
| **windowed, full gold** | **59.8 ± 1.0** | 59.1 ± 1.1 |

**Findings:** (1) the 65.5 reproduces exactly but ignores the 38% of links cut off by truncation; counting them as misses gives about 50. (2) Inference-only link-aware windowing recovers about 10 points (59.8); the token-space ceiling was 89.6% of links. (3) XFUND val is already nearly template-disjoint by the text proxy; removing the 16 documents with a train sibling costs under 1 point. (4) 62% of val documents do not fit in 512 tokens (84 of 250 do).

**Limits:** entity spans and types are gold, so these are not end-to-end link F1; the sibling test is text-only and within-language; no retraining on windows; LiLT, not BROS. Seeds 13 and 21 F1 values were recovered from the log of a run that crashed (OOM) before writing its output, so only F1 is stored for them (`results/xfund_link_eval/summary.json`); seed 42 is in `seed42.json`. The "windowed, visible gold" cells in that file are invalid (the code intersects predictions with the visible gold, forcing precision to 1.0) and must not be used.

---

## 18. Are BROS's and the VLM's errors correlated? — CPU, 2026-10-31

**Question:** study 16's gate gained little. Hypothesis (stated before the run): when BROS is wrong the VLM is wrong too, so no per-document signal can help much. Is the VLM less often right where BROS is wrong?

**Method (`scripts/error_correlation.py`, reuses the study-16 dumps; no new model runs):** per gold span, BROS right = exact (field, start, end) match, VLM right likewise; lift = P(VLM right | BROS missed) / P(VLM right | BROS right) (1 = independent). Per BROS-predicted span: wrong/right, the VLM's span for that field right/wrong, and BROS's confidence (mean field probability over its span; split at 0.7). Intervals: cluster bootstrap (FATURA: templates; VRDU: documents), seeds pooled within cluster, 500 draws.

**Results:**

| Setting | Gold spans | BROS recall | VLM recall | P(VLM right ∣ BROS missed) | P(VLM right ∣ BROS right) | Lift [95% CI] | BROS-only / VLM-only / both wrong |
|---|---|---|---|---|---|---|---|
| VRDU (600 docs, 2 seeds) | 5,750 | 63.7 | 56.4 | 32.4 | 70.1 | **0.46 [0.43, 0.50]** | 19.0 / 11.8 / 24.6 |
| VRDU confirm only | 2,870 | 64.2 | 56.7 | 32.3 | 70.3 | 0.46 [0.41, 0.51] | 19.1 / 11.6 / 24.2 |
| FATURA K=2 | 33,900 | 63.7 | 48.0 | 41.7 | 51.7 | 0.81 [0.74, 0.88] | 30.8 / 15.1 / 21.2 |
| FATURA K=5 | 33,900 | 91.5 | 48.0 | 33.3 | 49.4 | 0.67 [0.57, 0.79] | 46.3 / 2.9 / 5.7 |
| FATURA K=10 | 13,560 | 96.7 | 48.0 | 25.1 | 48.8 | 0.51 [0.27, 0.82] | 49.5 / 0.8 / 2.5 |

Where BROS's prediction is wrong, the VLM's span for that field is right 22% (VRDU) / 38% (K=2) / 25% (K=5) / 19% (K=10) of the time, against 70% / 52% / 49% / 49% where BROS is right. Spans with BROS confidence < 0.7 are 5.1% of VRDU predictions (68% of them wrong, VLM right on 26% of those) and 32% / 7.5% / 2.0% of FATURA K = 2 / 5 / 10 predictions.

**Findings:** (1) The errors are positively correlated at every setting (lift < 1, interval excludes 1 except K=10's upper end of 0.82): the VLM is less reliable exactly where BROS fails, strongest on VRDU (lift 0.46). (2) The correlation is partial, not total: the VLM is still right on a quarter to two fifths of BROS's misses, which is the headroom the oracle measures. (3) Plausible reading of the weak gate (an inference from these rates, not tested): replacing a low-confidence span gains only when BROS is wrong and the VLM right, which is about a quarter to two fifths of the wrong spans, while the low-confidence spans BROS got right can only be lost. (4) FATURA K=2 is the least correlated setting (lift 0.81), consistent with it being the one setting where the hybrid gained.

**Limits:** exact-span matching; the VLM output is one fixed run per document; FATURA is synthetic and VRDU has 3 templates (intervals resample documents there); correlation is not causation (hard documents may hurt both models); descriptive study, no registered criteria.

**Evidence:** `results/error_corr/{fatura,vrdu}.json`, `scripts/error_correlation.py` (self-test included).

---

## 19. Windowed joint training and end-to-end XFUND link F1 — GPU (Kaggle T4 × 2), registered 2026-10-31

**Question:** study 17 measured link F1 on the *saved* joint-LiLT checkpoints with **gold** entity spans and types: 65.5 on the links visible in 512 tokens, 49.9 counting the cut-off links as misses, 59.8 with inference-only windows. What is the *end-to-end* link F1 (entities from the model's own tags), and does **training** on entity-aligned windows improve it?

**Arms (3 seeds 13 / 21 / 42, same code and config, only `form_window` differs):**
- `base`: `configs/xfund_joint_lilt.yaml` (3-stage curriculum, documents truncated at 512 tokens).
- `windows`: `configs/xfund_joint_lilt_windows.yaml` (training and validation on 510-subword entity-aligned windows, stride 255; test stays whole-document).
- The old checkpoints were trained with an older feature set (12-D spatial features; current code gives 16-D), so they are *not* the base arm; they are scored with the same end-to-end code as a CPU reference (`results/xfund_e2e/legacy_checkpoints.json`).

**Evaluation (`scripts/xfund_e2e_eval.py`):** 250 val forms (es/fr/de/it/pt); entities are the BIO spans of the model's own predictions; the relation head links those spans (`src/e2e_link.py`, checked equal to `JointFormExtractor.forward` by `tests/test_e2e_link.py`). Two readings: `single` (first 512 tokens) and `windowed` (510-token windows, stride 128; an entity touching a window edge that is not the document edge is dropped; links unioned). Gold = every annotated link (links touching an untagged "other" entity can never be found; their count is reported). A link is correct if both endpoints match the word ranges of a linked gold pair; entity F1 is typed.

**Registered criteria (primary cell: link F1, windowed inference, all documents, seed-matched differences):**
- *windows help*: mean delta (windows - base) >= +2.0 points and > 0 in every seed.
- *no better*: |mean delta| < 1.0 point, or the sign differs across seeds.
- Secondary, descriptive: entity F1; `single` vs `windowed` inference within each arm (the inference-only effect on end-to-end linking); the gap to study 17's gold-entity numbers; the sibling-free subset (234 of 250 documents).

**Checks done before the run:** unit tests of windowing and of the relation half against `forward` (`tests/test_e2e_link.py`); end-to-end evaluation smoke-run on 4 documents; summary self-test.

**Limits stated up front:** 3 seeds (no interval is claimed, only seed-matched deltas); LiLT, not BROS; entity-type mismatches are ignored for links; the window-edge rule is a heuristic; the base arm is retrained with the current code, so absolute numbers are not comparable to the earlier 65.5; sibling-free is a text-only proxy; training time is unmeasured.

**Results (3 seeds, F1 in points; Kaggle T4 x2, both arms 22 epochs, all return codes 0):**

| Cell (all 250 val forms) | base | windows | delta mean | delta per seed (13 / 21 / 42) |
|---|---|---|---|---|
| **link F1, windowed inference (primary)** | 43.5 | **47.7** | **+4.2** | +4.8 / +4.7 / +3.2 |
| link F1, single 512 window | 39.6 | 41.7 | +2.1 | +1.9 / +1.9 / +2.3 |
| entity F1 (typed), windowed inference | 73.0 | 75.9 | +2.9 | +2.8 / +3.2 / +2.6 |
| entity F1, single 512 window | 62.8 | 64.1 | +1.3 | +1.1 / +1.5 / +1.3 |
| link F1, windowed, sibling-free (234 forms) | 42.6 | 46.6 | +4.0 | +4.5 / +4.3 / +3.2 |

Precision / recall, windowed inference, mean of 3 seeds: links base 41.2 / 46.0, windows 47.2 / 48.2; entities base 66.4 / 81.2, windows 70.2 / 82.7. The train-time whole-document test metric (gold entities, first 512 tokens, visible links only) is 66.4 link F1 (base) vs 70.2 (windows). 36 of 9,062 gold links touch an untagged "other" entity and can never be found.

**Verdict (registered criteria):** *windows help* is met (mean delta +4.2 >= +2.0 and positive in every seed); *no better* is not.

**Reference (CPU, legacy checkpoints trained with the older 12-D feature set; only seed 13 finished before the run was killed, `results/xfund_e2e/legacy_checkpoints.json`):** end-to-end link F1 34.2 (single) / 38.6 (windowed), entity F1 62.9 / 73.5. The same seed-13 checkpoint scored 50.9 / 60.9 link F1 with gold entities in study 17, so predicted entities cost about 16 to 22 points. The retrained base arm (39.6 / 43.5) is not weaker than the legacy one.

**Findings:** (1) Training on entity-aligned windows improves end-to-end link F1 by about 4 points over truncated training, consistently across seeds; the gain is larger with windowed inference (+4.2) than with the single window (+2.1), so the two fixes add up. (2) End-to-end linking from the model's own entities reaches only 47.7 F1 (precision 47, recall 48): far below the 65.5 originally reported (see study 17) and below the 59.8 gold-entity windowed figure. (3) Entity errors are the larger cost: windowed inference predicts many more entities than exist (legacy seed 13: 15,707 predicted vs 13,007 gold, precision 67 at recall 81), consistent with the same entity being predicted with slightly different spans in overlapping windows; exact-span matching then counts those as wrong. (4) Even among the links whose both endpoints exist, roughly a third are still missed: with entity recall 83%, about 69% of links have both endpoints found if the two events are independent, versus a link recall of 48% (an arithmetic estimate, not measured).

**Limits:** 3 seeds, so only seed-matched deltas, no interval; the windows arm sees about 3x as many training examples per epoch (3.25x in the 16-form smoke sample; overlapping windows also act as augmentation) over the same 22 epochs, so "more gradient steps" is not separated from "better coverage"; LiLT, not BROS; link endpoints are matched on word range only; the edge-dropping rule is a heuristic and the cross-window duplicate hypothesis in finding 3 is not tested; the new checkpoints' weights were not pushed, so no gold-entity score exists for them; the legacy reference is one seed.

**Evidence:** `results/xfund_e2e/{summary,base_e2e,windows_e2e,run_status}.json`, per-seed `base/` and `windows/`, `legacy_checkpoints.json` (seed 13).

---

## 20. Cross-window entity deduplication for end-to-end XFUND linking — CPU, registered 2026-11-01

**Question:** study 19 found that windowed inference over-predicts entities (legacy seed 13: 15,707 predicted vs 13,007 gold, precision 67 at recall 81) and conjectured that the same entity is predicted with slightly different spans by overlapping windows, which exact-span scoring counts as wrong. Does collapsing those duplicates raise end-to-end link and entity F1?

**Strategies (`merge_windows` in `src/e2e_link.py`; all use the same cached per-window model output, so only the merge differs; stride 128, windows of 510 tokens):**
- `edge_drop` (baseline = the rule used so far): drop an entity touching a window edge that is not the document edge; keep everything else.
- `center` (primary): drop nothing; same-type predictions that share a word form a cluster; the one furthest from its window edge represents it (ties: earliest window); links are remapped to the representative and self-links removed.
- `edge_drop+center` (secondary): both.

**Data and checkpoints:** XFUND val, 250 forms, the same gold as studies 17 and 19 (all annotated links; entities typed). Primary run: the three legacy checkpoints (`results/xfund_joint_lilt/seed_{13,21,42}`) on this machine. Confirmation, with the same criteria and no new variants: the six study-19 checkpoints once the Kaggle outputs are downloaded.

**Registered criteria (primary cell: link F1, windowed inference, all forms, `center` minus `edge_drop`, seed-matched):**
- *dedupe helps*: mean delta >= +1.0 point and >= 0 in every seed.
- *no better*: mean delta < +0.5 point or a negative seed.
- Descriptive: entity precision / recall / F1, number of predicted entities, `edge_drop+center`, the sibling-free subset.

**Sanity check (must pass before reading anything else):** `edge_drop` on the cached output reproduces the earlier legacy seed-13 numbers in `results/xfund_e2e/legacy_checkpoints.json` (link F1 0.3857, entity F1 0.7347).

**Checks done before the run:** model-free unit test of the three strategies on a synthetic two-window form (`tests/test_e2e_link.py::test_merge`), including that a single window is unaffected.

**Limits stated up front:** the legacy checkpoints are not the study-19 models; clustering on shared words can merge two distinct gold entities that a fragment overlaps; the centrality rule is one heuristic among several (confidence-based choice needs probabilities the model code does not return); link endpoints are matched on word range; the dedupe conjecture is tested only through its effect on F1.

**Results (legacy checkpoints, 3 seeds, 250 forms, F1 in points; sanity check passed: `edge_drop` equals the earlier seed-13 numbers 38.57 / 73.47 exactly):**

| Strategy | link F1 | link P / R | entity F1 | entity P / R | predicted entities (gold 13,007) |
|---|---|---|---|---|---|
| `edge_drop` (baseline) | 38.2 | 36.9 / 39.6 | 72.8 | 66.5 / 80.4 | 15,730 |
| `center` (primary) | 37.9 | 37.2 / 38.6 | 74.2 | 70.6 / 78.2 | 14,399 |
| `edge_drop+center` | 38.3 | 38.2 / 38.4 | 74.5 | 71.0 / 78.2 | 14,318 |

Seed-matched deltas versus `edge_drop` (seeds 13 / 21 / 42): `center` link F1 -0.37 / -0.34 / -0.34 (mean -0.35), entity F1 +1.18 / +1.51 / +1.64; `edge_drop+center` link F1 +0.00 / +0.09 / +0.10 (mean +0.06), entity F1 +1.42 / +1.71 / +1.84.

**Verdict (registered criteria, primary cell = link F1, `center` minus `edge_drop`):** *dedupe helps* is not met (mean delta -0.35 < +1.0, every seed negative); *no better* is met.

**Findings:** (1) The duplication conjecture is partly right for entities: collapsing overlapping same-type predictions removes about 1,300 entities and raises entity precision by about 4 points, at the cost of about 2 points of recall (a representative with the wrong span replaces a correct one, or two distinct entities are merged). (2) It does not move link F1: the precision gain and the recall loss cancel, and with the edge rule kept the change is +0.06. (3) Over-prediction is not the main bottleneck: even after deduplication about 11% more entities are predicted than exist (14.3k vs 13.0k) at precision 71 and recall 78, so boundary and type errors dominate the remaining entity gap, and link F1 stays between 37.9 and 38.3 whatever the merge rule. (4) The cheap merge-rule route to better end-to-end linking is closed on these checkpoints; gains have to come from the model (training, or a stronger encoder or linker).

**Limits:** legacy checkpoints (older feature set), not the study-19 models; centrality is one heuristic (confidence-based selection, which could keep the better span, needs probabilities the model code does not return); clustering on shared words can merge distinct gold entities; the registered confirmation on the study-19 checkpoints has not been run because their weights are not in the repo.

**Evidence:** `results/xfund_dedupe/legacy.json` (the per-window cache under `results/xfund_dedupe/cache/` is local, not committed).

---

## 21. Step-matched control for windowed joint training — GPU (Kaggle T4 × 2), registered 2026-11-01

**Question:** study 19 found that training on entity-aligned windows beats truncated training by +4.2 end-to-end link F1 (3/3 seeds) but flagged a confound: the windows arm saw 1,537 windows per epoch against 634 forms (2.42x; measured on the actual train split with the study-19 windowing) over the same 22 epochs, about 8,470 gradient steps against about 3,500. Is the gain better coverage of the long forms, or just more steps (and overlapping windows acting as augmentation)?

**Control (`configs/xfund_joint_lilt_matched.yaml`, notebook `notebooks/kaggle_matched_steps.ipynb`):** the study-19 `base` recipe (truncated forms, 3-stage curriculum, batch 4) with the curriculum epoch caps raised from 8 / 6 / 8 to 19 / 15 / 19 (53 epochs of 159 steps, about 8,430 steps, within 1% of the windows arm). Seeds 13 / 21 / 42, same code, same end-to-end evaluation (`scripts/xfund_e2e_eval.py`). The study-19 `base` and `windows` results are read from the repo, not retrained. In study 19 both arms ran all 22 epochs (early stopping never fired); with longer schedules the control may stop early, and the epochs actually run are reported.

**Registered criteria (primary cell: link F1, windowed inference, all 250 forms; seed-matched differences, study-19 windows minus control):**
- *coverage matters*: mean delta >= +2.0 points and > 0 in every seed.
- *steps explain it*: |mean delta| < 1.0 point, or the sign differs across seeds, or the control is ahead.
- Descriptive: control minus study-19 base (what the extra steps alone buy); entity F1; single-window link F1; the sibling-free subset.
- If the control matches or beats the windows arm, the study-19 claim "windows help" is withdrawn as a coverage claim and reduces to "longer training helps".

**Checks done before the run:** the matched config differs from `xfund_joint_lilt.yaml` only in the epoch caps, `epochs` and `output_dir` (diffed); notebook cell 3 asserts the schedule and that the study-19 reference files exist; cells 5-6 executed locally against a fake control (windows minus a copy of base reproduced +4.2; control minus base = 0.0); the training and evaluation code paths are those smoke-tested in study 19.

**Limits stated up front:** 3 seeds (no interval, only seed-matched deltas); the control and the study-19 arms are separate Kaggle runs; matching the number of steps does not match the number of distinct examples or the learning-rate schedule shape across phases; the control is the legacy-recipe feature set retrained with current code, as in study 19; LiLT, not BROS; training time unmeasured (about the same compute as the windows arm).

**Results (3 seeds, F1 in points; Kaggle T4 x2; train and eval return codes 0):**

| Cell (all 250 forms) | base (s19) | control | windows (s19) | windows - control | per seed (13 / 21 / 42) | control - base |
|---|---|---|---|---|---|---|
| **link F1, windowed inference (primary)** | 43.5 | 44.6 | **47.7** | **+3.1** | +4.0 / +3.4 / +1.8 | +1.2 |
| link F1, single 512 window | 39.6 | 40.4 | 41.7 | +1.3 | +1.2 / +1.4 / +1.4 | +0.7 |
| entity F1, windowed inference | 73.0 | 73.3 | 75.9 | +2.7 | +4.0 / +1.7 / +2.3 | +0.2 |
| entity F1, single 512 window | 62.8 | 63.2 | 64.1 | +0.9 | +1.1 / +1.0 / +0.7 | +0.4 |
| link F1, windowed, sibling-free | 42.6 | 43.7 | 46.6 | +2.8 | +3.9 / +2.9 / +1.8 | +1.1 |

Control, windowed inference, mean of 3 seeds: link precision / recall 42.8 / 46.7, entity precision / recall 66.9 / 81.0. The control's curriculum ended by early stopping after 35 / 38 / 43 of 53 epochs (phases 8+15+12, 17+12+9, 19+15+9), i.e. about 5,570 / 6,040 / 6,840 steps against about 8,470 for the windows arm. Train-time whole-document test link F1 (gold entities, first 512 tokens): base 66.3, control 67.9, windows 70.2.

**Verdict (registered criteria):** *coverage matters* is met (mean +3.1 >= +2.0, positive in every seed); *steps explain it* is not.

**Findings:** (1) Most of the study-19 gain survives a longer-trained control: windows add about +3.1 end-to-end link F1 and +2.7 entity F1 beyond what extra training gives. (2) Extra training alone is worth about +1.2 link F1 (consistent across seeds, 0.8 to 1.4) and almost nothing for entity F1 (+0.2), so about 30% of the headline +4.2 was steps and about 70% was windows. (3) Entity F1 is where windows clearly win; the link-level gain is smaller and the weakest seed (42) is +1.8, just under the +2.0 bar on its own. (4) The control stopped early on validation, so more steps beyond about 35-43 epochs did not improve it; the matching of steps is therefore approximate (66 to 81% of the windows arm's), and part of the remaining +3.1 could still be steps the control chose not to take.

**Limits:** 3 seeds, no interval; the control and the study-19 arms are separate Kaggle runs; step matching is incomplete because of early stopping; overlapping windows also act as augmentation, which this control does not separate from coverage; LiLT, not BROS.

**Evidence:** `results/xfund_e2e_matched/{base_matched_e2e,summary_vs_windows,summary_steps_only,run_status}.json`, `base_matched/` (per-seed results and histories).

---

## 22. Where do gold-entity link errors come from? — CPU, 2026-11-01

**Question (step 0 before designing a "stronger encoder" for linking):** with perfect entity spans and types, why does the joint-LiLT linker reach only about 60 F1? In particular, are the misses close-neighbour confusions (which a better layout mechanism such as BROS's relative-position attention could plausibly fix), long-range or vertical relations, or something about the link scheme?

**Method (`scripts/xfund_link_errors.py`, descriptive, no registered criteria):** the three legacy checkpoints, gold entity spans and types, windowed inference (510 tokens, stride 128) as in study 17, all 250 val forms (9,062 gold links per seed, 5,368 found, 3,536 spurious on average). Every gold link is classified by endpoint types, by the geometry of the true partner relative to the source (centre offsets; "same line" if the vertical offset is under half the larger box height), by the rank of the true partner among same-type entities ordered by centre distance from the source (0 = nearest), and by whether any window contains both endpoints. Every spurious predicted link is classified as reversed direction, source with a different true partner (with the rank of the predicted partner, or "another type"), or source with no true partner.

**Sanity check (passed):** seed 13 gives precision 61.9, recall 59.9, F1 60.88, equal to the study-17 windowed full-gold figure 0.6088; seeds 21 and 42 give 59.5 and 58.9 (pooled 59.8).

**Gold links (pooled over the 3 seeds; "of misses" = share of the 3,694 missed links per seed):**

| Group | Share of gold | Recall | Share of misses |
|---|---|---|---|
| partner below | 58.2% | **49.2** | 72% |
| partner on the same line, to the right | 36.6% | 79.4 | 19% |
| partner above | 5.1% | 29.7 | 9% |
| type question -> answer | 80.8% | 65.6 | 68% |
| type question -> question | 18.3% | **33.9** | 30% |
| header -> question, answer -> answer, links touching "other" (rare) | 1.0% | 0 to 14 | ~2% |
| endpoints not in one window (unreachable) | 10.3% | 0.0 | 25% |
| reachable links | 89.7% | 66.1 | 75% |
| true partner is the nearest same-type entity (rank 0) | 25.4% | 64.7 | |
| true partner rank 1 | 14.0% | 66.8 | |
| true partner rank 2-4 | 22.8% | 65.2 | |
| true partner rank 5+ | 37.8% | 49.2 | |

(The group rows overlap: geometry, type and reachability are separate cuts of the same links.)

**Spurious links (per seed, 3,536):** 66% come from a source that has a different true partner, 34% from a source with no true partner at all, ~0% are reversed. For the wrong-partner cases (2,323), the predicted partner is at rank 0 in 4.8%, rank 1 in 5.4%, rank 2-4 in 11.1%, rank 5+ in 41.6%, and of another type than any true partner in 37.1%.

**Findings:** (1) Vertical relations are the main loss: 58% of gold links point downward, recall 49 against 79 for same-line partners, and they account for 72% of all misses. (2) Question->question links (hierarchical keys, 18% of gold) are found only a third of the time and give 30% of the misses; they are also where the soft type prior is least informative, and 37% of wrong-partner predictions land on an entity of the wrong type, consistent with the model hedging between question and answer targets. (3) Close-neighbour confusion is not the main error: the nearest candidate is found only 65% of the time, the same as candidates 2-4, and only 10% of wrong partners are the nearest or second nearest; most wrong partners are far away (42%) or of the wrong type (37%). (4) About a quarter of the misses (10% of gold) cannot be found by any window, which no encoder fixes. (5) Taken together this does not point to a short-range geometry deficit that a relative-position attention encoder would repair; it points to structural disambiguation (which of several candidates down the page, and which link kind) and to the link scheme itself.

**Limits:** legacy checkpoints (older feature set), gold entities, LiLT only; rank and direction are defined by box-centre geometry (not text); the "another type" class uses the predicted partner's gold type; the analysis cannot say whether a different encoder would change these proportions, only where the current errors sit; correlated categories (vertical links are also the question->question links and the far partners), so the rows are not independent causes.

**Evidence:** `results/xfund_link_errors/legacy.json`, `scripts/xfund_link_errors.py`.

---

## 23. Decode-time levers for gold-entity linking — CPU, registered 2026-11-01

**Question:** study 22 showed that gold-entity link errors are mostly downward links (72% of misses) and question->question links (30%), that wrong partners are rarely the nearest candidate, and that 37% of wrong partners are of the wrong type. Are the missing links already scored correctly or nearly so and lost in the decoding and type rules (then a fix needs no retraining and the encoder is not the bottleneck), or is the information not in the scores (then the encoder arms are justified)?

**Suspect found in the code:** the soft type prior added inside the relation head (`logits + soft_priors`) allows only question->answer, header->question and header->answer; every other pair gets -5. Question->question is 18% of the gold links and is penalised by it. The prior is therefore baked into the logits the model returns, and can be undone at decode time.

**Data:** the three legacy checkpoints (`results/xfund_joint_lilt/seed_{13,21,42}`), gold entity spans and types, windowed (510 tokens, stride 128) as in studies 17 and 22. The model runs once per window and its relation logits are cached; every variant is a re-decode of the same cache (`scripts/xfund_decode_study.py`). **Selection split:** the 111 forms carved from the training file (the same carve the checkpoints used, `load_xfund_dataset` seed 42). **Report split:** the 250 forms of the XFUND val file, as in all earlier studies.

**Variants (fixed grid, 5 x 2 x 4 x 3 = 120; default V0 = 0.5 / one-to-one / prior as trained / no limit):**
- threshold on the link probability: 0.3, 0.4, 0.5, 0.6, 0.7;
- one-to-one decoding (each value keeps only its best-scoring key): on, off;
- type prior: *as trained*; *none* (the -5 undone for every pair); *qq_free* (the -5 undone only for question->question pairs); *hard* (pairs outside the allowed set forbidden);
- nearness limit: none, or only the K nearest same-type candidates by box-centre distance (the rank used in study 22), K = 5 or 10.

**Selection and criteria (registered):** the variant with the highest pooled link F1 on the selection split (counts summed over the 3 seeds; ties: fewest non-default factors, then threshold closest to 0.5) is applied to all seeds on the report split. Primary cell: link F1 on the 250 forms, full gold (all annotated links), selected variant minus V0, seed-matched.
- *decode-time levers matter* (decoding or type handling is the first thing to fix): mean delta >= +2.0 points and no negative seed.
- *representation-limited* (the encoder arms are justified): mean delta < +1.0 point.
- Between the two: inconclusive, reported as such.
- Descriptive: one-factor sweeps on the report split (each factor varied alone from V0); recall of the question->question, question->answer, downward and same-line groups for V0 and the selected variant.

**Sanity check (must pass first):** V0 on the report split reproduces the study-22 per-seed F1 (seed 13: 0.6088; seeds 21 and 42: 0.5948 and 0.5894).

**Checks done before the run:** model-free self-test that the numpy decoder equals `BiaffineRelationClassifier.decode_predictions` on random logits (one-to-one on and off, several thresholds, diagonal included), that the prior algebra reproduces `build_soft_type_priors_from_labels`, and that the nearness rank equals study 22's.

**Limits stated up front:** legacy checkpoints (older feature set) and gold entities only; the checkpoints were trained with the current decoding and prior, so a retrained model could respond differently (removing the prior at decode time only removes what the biaffine part learned to compensate); the selection split was used for early stopping of these checkpoints; no decode change can reach the 10% of links that no window contains; 3 seeds, no interval; 120 variants are searched on the selection split, which is why the report split decides.

**Results (legacy checkpoints, gold entities, windowed; sanity check passed: V0 on the report split = 0.6088 / 0.5948 / 0.5894, equal to study 22):**

Selected on the 111-form selection split (pooled F1 63.76 for V0 -> 64.70): **threshold 0.4, one-to-one on, hard type mask, no nearness limit.**

| Report split (250 forms) | V0 (0.5, one-to-one, prior as trained) | selected | delta per seed (13 / 21 / 42) |
|---|---|---|---|
| link F1, mean of 3 seeds | 59.8 | **62.2** | **+2.4** (+1.85 / +2.16 / +3.20) |
| per-seed link F1 | 60.9 / 59.5 / 58.9 | 62.7 / 61.6 / 62.1 | |
| pooled precision / recall | 60.3 / 59.2 | **67.7** / 57.4 | |

**Verdict (registered criteria):** *decode-time levers matter* is met (mean +2.4 >= +2.0, no negative seed); *representation-limited* (< +1.0) is not.

One-factor sweeps on the report split (link F1, mean of 3 seeds, others at V0):

| Factor | Values |
|---|---|
| threshold | 0.3: 56.2 · 0.4: 59.0 · **0.5: 59.8** · 0.6: 57.1 · 0.7: 49.3 |
| one-to-one | on: 59.8 · **off: 47.6** |
| type prior | trained: 59.8 · **none: 29.1** · question->question unpenalised: 50.2 · **hard: 61.1** |
| nearness limit (K nearest same-type) | none: 59.8 · K=10: 57.5 · K=5: 51.9 |

Recall by group, seed 13, V0 -> selected: question->answer 66.4 -> 71.6; question->question 33.9 -> **0.0**; same-line partner on the right 79.2 -> 79.8; partner above 30.1 -> 37.0; **partner below 50.4 -> 45.9**.

**Findings:** (1) A decode-only change gains +2.4 link F1 with no retraining, consistently across seeds, but it is a precision gain bought by giving up a link class: the hard mask forbids all question->question pairs (18% of the gold links, recall 34 before), and (CORRECTED in the addendum below: this claim was wrong. The `qq_free` result only shows that removing the penalty is harmful; forbidding question->question alone lowers F1 by 2.6, because those predictions are 44% precise, above the roughly F1/2 break-even). (2) The selected variant does not fix the weakness found in study 22: recall for partners below the source falls from 50 to 46. (3) One-to-one decoding is essential (off: -12.2), the trained type prior is doing real work (none: -30.7), and the nearness limit hurts because many true partners are far (K=10: -2.3, K=5: -7.9). (4) The threshold 0.5 is near-optimal for the trained model; 0.4 helps only together with the hard mask. (5) So a cheap +2.4 exists, but the bulk of the distance to the reachable ceiling (recall 66% of reachable links, precision 60%) is not recovered by any decode rule in this grid; whether an encoder change would recover it is untested.

**Limits:** legacy checkpoints and gold entities only; the checkpoints were trained with the current decoding and prior, so a retrained model could respond differently; the selection split was also used for these checkpoints' early stopping; 120 variants searched on 111 forms (the report split decides, and it agrees in sign and size on every seed); group recalls are for seed 13 only; dropping question->question links changes what is predicted, and whether that is acceptable is a task-definition choice (they are scored in this study); no interval.

**Evidence:** `results/xfund_decode/legacy.json`, `scripts/xfund_decode_study.py`.

**Addendum (post hoc, 2026-11-01, same day): what the mask really does, and what question->question links are.** Variants below were conceived after the report-split results were seen, so they are exploratory; where a selection-split check exists it is given and agrees in direction. Same cache, legacy checkpoints, gold entity spans and types, 3 seeds, one-to-one decoding. Scripts were scratch analyses (not committed); numbers are reproducible from `scripts/xfund_decode_study.py` helpers.

*What question->question links are (250 forms, per `data/xfund`):* 1,657 of 9,062 gold links (18%; 5% in Italian to 27.5% in German), present in 143 of 250 forms. 94% point downward; the source rarely has an answer of its own (7%), the target usually does (95%); a source has 1 to 24 targets. Examples: "DADOS PROFISSIONAIS" -> "Formacao:", "Section 5 - Completed qualifications" -> "Name of school, college or university". They encode section membership (heading -> field label), not key -> value pairs.

*Where V0's false links sit (report split, per seed, true / false):* question->answer 4,802 / 1,399; question->question 562 / 704 (precision 44%); other->question 0 / 480; header->question 2 / 425; other->other 0 / 234; question->other 2 / 211; the rest < 30. About 925 false links per seed involve `other` entities (never produced by the tagger end to end) and 425 are `header->question`, which the prior allows but which are 99.5% false (gold has 17 header links in total).

*Report split, link F1 (mean of 3 seeds) with full gold ("structure mode"), thr 0.5:* V0 59.8; forbid question->question only 57.2 (-2.6); forbid every non-allowed pair except question->question 63.1 (+3.3; thr 0.4 chosen on the selection split 63.3 = +3.5; selection split 66.7 vs 63.8); study-23 selected variant (forbids question->question too) 62.2.

*End-to-end emulation (every pair touching an `other` entity forbidden in all variants, gold restricted to links between tagged entities; types are still the gold types):*

| Mode | Variant | Selection split | Report split (per seed) | Mean | P / R |
|---|---|---|---|---|---|
| structure (full gold) | V0, no `other` pairs | 66.8 | 64.1 / 62.5 / 62.9 | 63.2 | 67.2 / 59.5 |
| structure | + type rules, question->question kept | 66.9 | 64.2 / 62.6 / 63.0 | 63.3 | 67.5 / 59.5 |
| structure | + no links from `header` entities | 67.5 | 65.9 / 64.9 / 64.6 | **65.1** | 71.3 / 59.9 |
| key-value (gold without question->question) | V0, no `other` pairs | 65.4 | 63.6 / 62.4 / 61.7 | 62.6 | 60.1 / 65.2 |
| key-value | question->answer only | 72.9 | 71.7 / 70.2 / 70.4 | **70.8** | 77.3 / 65.3 |

(Thr 0.4 adds about +0.1 to +0.9 in these rows; thr 0.5 shown.)

**Corrected findings:** (1) Question->question links are heading -> field-label links. They are scored in XFUND (18% of the gold) and the model finds a third of them with 44% precision, which is above the break-even for F1, so predicting them helps in structure mode; they are not key -> value pairs and are irrelevant if the task is pairing a field name with its value. (2) Most of the study-23 "hard mask" gain was removing type-violating pairs that involve gold `other` entities; that part cannot transfer to the end-to-end pipeline, where it is already implicit (+3.4 for free in the emulation). (3) Transferable rules in the emulation: forbid links from `header` entities (+1.9 in structure mode, positive in every seed, XFUND-specific because its gold has almost no header links), and in key-value mode allow question->answer only (+8.2, because every other predicted link is then a false positive). (4) The earlier statement that question->question predictions were net harmful was a misreading of the `qq_free` variant and is withdrawn.

**Limits of the addendum:** post hoc; gold entity types (predicted types are noisier, so the type rules will pay less end to end); legacy checkpoints; header rule is a property of XFUND's annotation, not a general law; key-value mode changes the task being scored (the +8.2 compares systems on a metric that excludes question->question gold); no interval; a confirmation on the study-19 checkpoints or with predicted entities is still needed.

---

## 24. Multilingual text stream for joint XFUND linking: LiLT-InfoXLM vs LiLT-en, end to end — GPU (Kaggle T4 × 2), registered 2026-11-06

**Question.** Study 19 left entity precision (67-71%) as the largest cost of end-to-end linking, and the text stream of the joint model is English-only on Spanish, French, German, Italian and Portuguese forms. Does swapping in LiLT's multilingual text stream (`SCUT-DLVCLab/lilt-infoxlm-base`, same layout stream, same recipe) raise end-to-end entity and link F1?

**Arms (3 seeds 13 / 21 / 42, all with the windowed recipe `form_window: 510, stride 255`, 3-stage curriculum, batch 4, lr 5e-5, 22 epochs; nothing is tuned for the new backbone):**
- **A0** (reference, reused, not retrained): the study-19 `windows` arm, `results/xfund_e2e/windows_e2e.json` (LiLT-en text stream, LayoutLMv3 tokenizer).
- **A1**: identical config with `backbone: lilt_xlm` (LiLT-InfoXLM, XLM-R sentencepiece tokenizer through the LayoutXLM tokenizer, which accepts `boxes=` like the current path).

**Evaluation.** `scripts/xfund_e2e_eval.py` unchanged in meaning: 250 XFUND val forms, entities from the model's own tags, `single` and `windowed` readings (510-token windows, stride 128, edge-cut entities dropped), full gold = all annotated links, typed entity F1, sibling-free subset. Added for this study, as agreed: **question->answer-only link F1** (`link_qa`: gold = links from a question entity to an answer entity; predictions = predicted links whose endpoints are predicted question and answer; this filters predictions after decoding, it is not a decode-time mask). Published XFUND relation extraction scores question->answer links only, with gold entity labels; ours still predicts its own entities.

**What differs besides the encoder (measured before the run, on XFUND val).** The XLM-R tokenization is shorter: median 484 vs 651 tokens per form, 137 vs 84 of 250 forms fit in 512 tokens, 93.2% vs 89.7% of the 9,062 links are reachable by 510/128 windows, 2.26 vs 3.45 windows per form. Part of any gain is therefore tokenizer coverage, and A1 trains on fewer windows per epoch (fewer steps; study 21 found that extra steps alone were worth about +1.2 link F1, so fewer steps cannot explain a gain). The embedding table is 250k x 768 (about 278M parameters in total against about 130M).

**Registered criteria (primary cell: link F1, windowed inference, all 250 forms, all links, seed-matched A1 minus A0):**
- *text side matters:* mean delta >= +2.0 points and > 0 in every seed.
- *no better:* |mean delta| < 1.0 point, or the sign differs across seeds. A negative mean with consistent sign is reported as *worse*.
- Secondary, descriptive: entity F1 (typed); `link_qa` F1 (A0's `link_qa` is not in the study-19 files and is filled in only if the study-19 windows checkpoints are scored with the same evaluator); single-window readings; the sibling-free subset; the reachability ceiling of each arm; and, only if the study-19 checkpoints are available, gold-entity link F1 on the links reachable under both tokenizers (separates representation from coverage).

**Checks before any GPU spend.** The InfoXLM checkpoint loads in the joint model and the tokenizer gives word ids; the windowing, end-to-end evaluation and unit tests of `tests/test_e2e_link.py` pass with the new backbone; a model-free test of the question->answer filter and gold; the summary self-test; the notebook cell 3 runs the 3-phase mini-training smoke test on the GPU with the new backbone and raises on failure.

**Limits stated up front.** 3 seeds (seed-matched deltas only, no interval); one new backbone with the same hyper-parameters (the larger embedding table may want a different learning rate, untested); A0 comes from a separate Kaggle run of study 19; the coverage and step differences above are not separated by a control; `link_qa` for A0 may be missing; LiLT-InfoXLM is much less used than the English checkpoint; GPU time and memory unmeasured (about 2x the parameters).

**Checks done before any GPU spend (2026-11-06):** the InfoXLM checkpoint loads in the joint model (`lilt_xlm` in `BACKBONES`); `tests/test_e2e_link.py lilt_xlm` passes (every window within 512 tokens under the XLM-R tokenizer, `link_entities` equal to `forward`, windowed end-to-end prediction, the question->answer filter and gold); the default backbone still passes; the summary self-test passes and the summary skips cells missing from the study-19 file; and `scripts/smoke_xfund_e2e.py lilt_xlm` ran the 3-phase mini-training (16 forms -> **36 training windows**, against 52 for the English stream, which matches the shorter tokenization) and the end-to-end evaluation on CPU, writing every cell including `link_qa`. The smoke scores are meaningless at that size (one epoch per phase; phase-1 validation entity F1 was 0.0 with the new backbone against 0.13 for LiLT-en on the same smoke, consistent with slower early learning from the larger embedding table at the same learning rate, so the per-epoch history of the real run should be inspected).

**Results (run 2026-11-14, Kaggle T4 x2, 3 seeds; A0 = study-19 `windows` arm, seed-matched):**

| Cell (all 250 forms) | A0 (en) | A1 (xlm) | Delta per seed (13 / 21 / 42) |
|---|---|---|---|
| **Link F1, windowed (primary)** | 47.71 | **44.22** | -4.06 / -2.94 / -3.49 (mean -3.49) |
| Entity F1, windowed | 75.94 | 73.23 | -1.94 / -2.23 / -3.97 |
| Entity precision / recall, windowed | 70.2 / 82.7 | 68.6 / 78.6 | |
| Link precision / recall, windowed | 47.3 / 48.2 | 44.4 / 44.1 | |
| Link F1, single window | 41.69 | 41.88 | +0.2 (precision 57.0 -> 51.9, recall 32.9 -> 35.1) |
| Link F1, windowed, sibling-free | 46.58 | 43.52 | -3.1 |
| Question->answer-only link F1, windowed | not in the study-19 file | 50.36 (51.11 / 50.63 / 49.34) | A1 only |

**Verdict (registered rule): worse.** The mean delta is -3.5 points with the same sign in every seed, so neither *text side matters* nor *no better* applies; the multilingual stream is consistently worse in the windowed end-to-end reading, which is the registered primary cell.

**Other observations (descriptive):**
- In the single-window reading the two arms tie (41.7 vs 41.9). Windowed inference adds +6.0 link F1 for A0 but only +2.3 for A1, so the loss appears where windows are used.
- The entity side explains part of it: A1 finds fewer entities (recall 78.6 vs 82.7) and precision is also lower (68.6 vs 70.2). Links are scored on the entities found, so entity errors carry over.
- A1's recall per link type (windowed): same line 63.7, rank 0 (nearest partner) 54.2, key above value 29.4, key below value 33.0, rank 5+ 30.7, question->question 25.5. A0's per-type recalls are not in the study-19 file, so there is no A0 comparison for these.
- Training memory: peak 11.8-12.4 GB per GPU (seed-parallel on two T4). Training steps and wall-clock were not recorded in the pushed files.
- Metric cross-check passed for all three seeds (seqeval F1 equals the independent F1).

**Post hoc (2026-11-15): question->answer link F1 against a retrained A0.** The study-19 checkpoints exist only on Kaggle, so A0's `link_qa` was not scored. Study 26's A0 is the same recipe retrained with the same seeds (link F1 47.42 vs 47.71 for study 19, inside its tolerance) and its file contains `link_qa`, so it is used as a stand-in (a different training run, not the study-19 weights). Windowed, all forms, link_qa F1: A0 **53.15** (53.41 / 53.06 / 52.97) against A1 **50.36** (51.11 / 50.63 / 49.34), delta **-2.79** (-2.30 / -2.43 / -3.63); sibling-free 52.03 vs 49.43 (-2.60). Single-window: 46.20 vs 47.53 (+1.33). The multilingual stream is worse on the question->answer links too. Same-seed deltas against this A0 on all-links F1 are -3.21 (-3.81 / -2.37 / -3.44), close to the registered -3.49.

**Read with the limits above:** one new backbone with the English recipe unchanged (lr 5e-5, 22 epochs, same curriculum), so this says "no gain with this recipe", not "multilingual encoders cannot help". The study cannot separate the tokenizer-coverage effect from the representation effect. Fewer training windows (about 36 vs 52 on the smoke subset) means fewer gradient steps, which may contribute to the drop.

**Evidence:** `results/xfund_e2e_xlm/` (`xlm_e2e.json`, `summary.json`, `run_status.json`, `xlm/seed_*/`). Pulled from the Kaggle push (commit bfd097d).

---

## 25. Schema-guided, grounded VLM extraction on VRDU unseen templates — GPU (Kaggle T4 × 2), registered 2026-11-05

(Study 24, the multilingual text-encoder study for XFUND linking, is designed but not registered; this study is numbered 25 to avoid reusing it.)

**Question.** Does an LMDX-style recipe ([arXiv 2309.10952](https://arxiv.org/pdf/2309.10952): schema in the prompt, OCR lines with coordinate identifiers, answers tied to cited OCR lines and verified against them, per-field voting) beat our open-schema pipeline (studies 6-12) for the same checkpoint on unseen VRDU templates, and how much do the page pixels add? It targets the two largest VRDU losses measured so far: key->field naming (study 6) and span boundaries (study 9).

**Model and data.** Qwen2.5-VL-3B-Instruct (the checkpoint of studies 6-12), greedy decoding, up to 4 pages per document. VRDU `registration-form`, three templates (Amendment, Dissemination_Report, Short-Form). Document sets: the 300 documents of studies 7-9 (**dev**, 100 per template) and the 300 fresh documents of study 12 (**confirm**). BROS (Track-1, seeds 13 and 42) is the reference on the same documents, from the existing error dumps. A fresh **pilot** set of 30 documents (10 per template, excluding dev and confirm) is the only place the prompt or output format may be changed; every change is logged, then the recipe is frozen before dev and confirm are run once each.

**Recipe (frozen after the pilot).**
1. *Segments:* the VRDU OCR lines. Each gets an identifier from its centre coordinates in 100 buckets, `XX|YY`, with the page index; a suffix disambiguates collisions, and the number of collisions is reported.
2. *Prompt (per page):* `<Document>` the lines, each followed by its identifier; `<Task>` the instruction plus the schema as JSON keys, each with its one-line description from `results/zero_shot_kv_v2/schema_desc.json` (written by the model from field names only, so no template knowledge); `<Extraction>` for the model to complete.
3. *Output:* a JSON object, `null` for absent fields; each value is the extracted text followed by the identifier of the line it came from (several segments allowed).
4. *Grounding decode:* the cited line must exist and contain the text exactly (after the same normalisation as `matched_eval.py`); otherwise the value is dropped as ungrounded. A verified value becomes a word span inside that line.
5. *One answer per field:* majority over pages (and samples), ties to the earliest page.

**Arms (run in this order; an arm that does not finish within the quota is reported as not run):**
- **A0** (reference, reused): the open-schema pipeline of studies 8 and 12, image only, naming step, post-hoc fuzzy grounding (the best existing VLM spans for these documents).
- **A2** (**primary**): page image + OCR lines with identifiers + schema with descriptions + grounded decoding.
- **A1** (pixel ablation): A2 without the image.
- **A3** (descriptive): the better of A1 and A2 on dev, with 8 samples (temperature 0.5, top-k 40) and a vote.

**Format gate (automatic, on the pilot).** If fewer than 90% of page outputs parse as JSON, or fewer than 90% of returned values ground, the recipe switches to the documented fallback (one field per call, prefilled tag, as in study 7) and the switch is recorded in the results.

**Metric.** Exact entity-span micro F1 on word spans (identical to `scripts/matched_eval.py`), pooled and per template, per field; paired bootstrap over documents (1,000 draws) for differences.

**Registered criteria (primary cell: pooled exact-span micro F1 on the 300 confirm documents, A2 minus A0):**
- *Strong success:* delta >= **+5.0** F1, bootstrap lower bound > 0, and no template's upper bound < 0.
- *Weaker success* (looked at in addition, as requested): delta >= **+3.0** F1 with bootstrap lower bound > 0 and no template's upper bound < 0.
- *Competitive with BROS:* pooled A2 within 3.0 F1 of BROS, or the paired interval for A2 - BROS contains 0.
- *Not an improvement:* delta < +3.0, or the lower bound <= 0.
- Descriptive: A2 - A1 (value of the pixels); A3 versus its base arm; per-field F1 (notably `foreign_principle_name` and the Short-Form `registrant_name` decoy); grounding rate, hallucination (ungrounded) rate, identifier collision count; the dev-set results; and the **static hybrid router rerun with the new spans** (same rule as studies 11 and 12) to see whether a stronger VLM enlarges the +3.8 hybrid gain.

**Checks before any GPU spend.** Unit tests: identifier quantisation and collision handling; the JSON parser (valid, invalid, nulls, multi-segment values); the grounding verifier (a hallucinated value must be rejected, a correct one accepted); the vote logic; and an **oracle grounding check** that feeds the gold values with their true line identifiers through the whole decode path and must score near 100% on real documents, which validates the line-to-word-span mapping before the model is involved.

**Limits stated up front.** The 3B model may not follow the JSON-with-identifiers format (what the pilot gate catches); LMDX's strength came from first tuning the model on diverse synthetic form templates, which is not done here, so this tests the prompting and grounding recipe only; A2 - A0 bundles three changes (schema, OCR text, grounding) and cannot attribute the effect; the prompt is tuned on a 30-document pilot only; the dev documents were used in earlier VLM studies; VRDU registration forms only (no ad-buy, no FATURA); GPU time unmeasured; 1 model, 1 recipe.

**Clarifications made while building, before any model output exists (2026-11-05; derived from gold properties and code, not from results):**
1. *Reference arm:* A0 is `vlm_both`, the mapping used in the confirmed hybrid of studies 11 and 12 (`results/matched_eval/spans.jsonl`, `results/confirm_v2/matched/spans.jsonl`). `vlm_key` and `vlm_named` are not scored in the verdict.
2. *Line -> word mapping:* VRDU OCR line `segments` are character offsets into the document text, the same space as the gold annotations. A verified value is mapped to words by character overlap inside its cited line, which is how the gold itself is aligned (so punctuation-only tokens such as `)` and `'` at the edges are included). If the exact text is not found in the cited line, the registered normalised-token containment is the fallback and is counted separately (`grounded_lenient`).
3. *Multi-segment values:* segments are merged while contiguous; a later disjoint segment is ignored, because the gold alignment keeps only the first contiguous run of a disjoint value (counted as `grounded_disjoint_truncated`).
4. *Verdict keys:* `strong_ge_5`, `weaker_ge_3`, `competitive_with_bros` as registered, plus `not_worse_than_bros_by_more_than_3` (a model far above BROS is not "within 3.0" but is not a failure either).
5. *Samples* for A3 are drawn with temperature 0.5 and top-k 40; the deadline for the whole run is 11.5 hours (changed from 10 on 2026-11-18, before any run).

**Checks done before any GPU spend:** the self-test (identifier quantisation and collisions, parser, grounding verifier including hallucination rejection, punctuation-only words, disjoint segments, vote and overlap rules) passes; the **oracle grounding check** on 100 dev documents (gold values with their true line identifiers through the decode path) gives exact-span F1 **0.9969** (478 of 478 predicted spans correct against 481 gold; the 3 missing gold values consist only of whitespace); the pilot manifest has 30 documents (10 per template) with zero overlap with dev or confirm; the `run` loop (sharding, resume, field-format fallback) was exercised with a stub model; `gate` and `score` were exercised on a fabricated perfect model (0.9964); the notebook cells 5 and 6 were run on that data.

**Pre-run amendment (2026-11-18, before any GPU run): a second model, two simultaneous sessions.** Motivation: the literature check of 2026-11-17 (UniKIE-Bench, VAREX, OCRBench v2; see the novelty assessment) puts Qwen3-VL-8B as the strongest open model on field extraction (UniKIE constrained 79.05, VAREX image 96.6), and all LMDX-type gains were with models far larger than 3B. (1) `SESSION` in the notebook: **A** = the registered arms above, unchanged (Qwen2.5-VL-3B, `results/schema_guided_kv`); **B** = the same recipe, pilot, documents, grounding, scorer and verdict rules with **Qwen/Qwen3-VL-8B-Instruct** loaded in 4-bit nf4 (bitsandbytes, fp16 compute), arms A2-8B, A1-8B, A3-8B, output `results/schema_guided_kv_q3vl8b`. (2) The 3B primary cell (A2 minus A0, confirm set) and its verdicts are unchanged; the 8B cell is graded by the same rules against A0 and BROS, and A2-8B minus A2-3B (paired) is reported as the effect of scale, A2 minus A1 within each model as the effect of pixels. (3) Descriptive additions, computed after the run with the Study 29 diagnostics script: boundary-tolerance table, snapping to BROS, and router complementarity. (4) Primary metric stays entity (exact-span) F1; XFUND link F1 is a separate track. (5) Needs `transformers>=4.57,<5` for session B (A keeps 4.51.3); the smoke cell imports `Qwen3VLForConditionalGeneration` and checks that the processor honours the pixel budget (at most 1,600 visual tokens per page image), and raises otherwise. **Limits added:** 4-bit quantisation and fp16 compute on a T4 (Qwen3-VL is usually run in bf16) may cost quality or produce non-finite outputs, so a bad 8B result could be a loading artefact (the format gate and smoke outputs are the check); the 8B arm has not been run on a GPU and the `Qwen3VLForConditionalGeneration` loader path is untested here.

**Results (sessions A = 3B, 7.9 h; B = 8B, 11.5 h; both pushed).** *Format gate:* with both models the JSON-per-page format failed (parse 100%, but 0% of 3B values and 5.5% of 8B values carried a valid line id), so both switched to one call per field (the registered fallback). Exact-span micro F1, 300 confirm documents, **registered decode**: 3B A2 (image + OCR lines + schema) **4.0**, 3B A1 (no image) 42.8, 8B A2 **60.2**, 8B A1 57.4; reference A0 57.7; BROS 69.0 / 72.0 (seeds 13 / 42). **Registered verdicts:** 3B A2 - A0 = -53.7 (*not an improvement*); 8B A2 - A0 = +2.5 [+0.0, +5.2], below the +3.0 bar (*not an improvement*), and 8B is about 10 points below BROS (*not competitive*). A3 (8 samples) is incomplete (3B: 258 / 287 documents; 8B: 2 confirm documents, no dev: out of time) and is not interpreted. GPU cost (sum of per-unit seconds over both GPUs, per 300 documents): 3B A1 0.65 h, 3B A2 2.65 h, 8B A1 2.0 h, 8B A2 4.4 h; 8-sample vote 3B 3.7 h, 8B about 8.2 h.

**Stage 0 post hoc re-analysis (CPU, 2026-11-19; written after seeing the results; `scripts/study25_stage0.py`, `results/study25_stage0/`).**
- *Decode flaw found:* `ground()` discards a value when its line id is missing or malformed. The 3B model with the image wrote truncated ids ('Jeff Surrell 1', '06/15/2016 43' for '43|96'), so 1,360 of 1,517 values were lost. A post hoc `--fallback` (look the text up in the page's lines when the id is unusable) restores 1,257 of them. Exact-span F1 with fallback, confirm (dev): **3B A2 56.6 (56.3), 3B A1 42.9 (42.5), 8B A2 60.5 (58.1), 8B A1 57.3 (55.4)**. The 3B score of 4.0 was a decode artefact, not model accuracy; the 8B was barely affected (1.6% of values lost). The registered verdicts stand; this is not the registered decode.
- *Paired effects (confirm; dev in brackets; exact-span):* image, 3B **+13.7 [+11.2, +16.3]** ([+13.9]); image, 8B **+3.3 [+1.8, +4.8]** ([+2.8, CI above 0]); scale, A2 8B - 3B +3.8 [+1.9, +5.8] ([+1.8, [-0.2, +4.0]], so the scale effect with the image is not firm); scale with text only is large (8B A1 57.3 vs 3B A1 42.9).
- *Metric check:* an official-style text-match scorer (rules of the VRDU `match_utils.py`, read through a subagent's summary and not run with the official tool, one gold entity per field assumed) gives almost the same F1 as exact spans (8B A2 60.6, BROS 69.6 / 72.0), so the exact-span rule is not what separates us from LMDX's roughly 90; the gap is model scale/tuning, as before.
- *Where 8B A2 loses (confirm):* `file_date` 22.3 (BROS 84.0): the model picks the received stamp, while the Amendment gold is the written 'day of ... year' near the notary line (183 of 243 predictions differ in text, not only in span); `signer_name` 41.1: it copies 'Name , Title' (107 of 222 predictions too long), and snapping to BROS lifts it to **72.6**, above BROS (64.9); `registrant_name` 62.3: it sometimes returns the signer instead of the organisation; `foreign_principle_name` 70.4 against BROS 2.0; `signer_title` 73.5 against 50.3. Part of the `foreign_principle_name` misses are annotation convention (gold includes the extra line 'Tokyo, Japan').
- *Snapping and routing (confirm; exact-span; snapping = replace each VLM span by the overlapping BROS span of the same field, seed 42 in the table):* 8B A2 60.5 -> **65.5** snapped -> 67.9 snapped and filled with BROS spans for fields the VLM leaves empty (BROS 72.0). The static cross-template router (as studies 11-12) with the fallback spans: **76.2 against BROS 70.5, +5.7 [+4.4, +7.0]**, primary criterion passes (registered decode: 75.7, +6.0); routing snapped spans gives the same 76.2; with the 3B A2 spans 72.5 (+2.0).
- *Reading:* the model is a strong, complementary source for semantic fields and a weak one for boundaries and for 'which date'; snapping alone does not reach the router (76.2). The LMDX line-id recipe was not reproduced zero-shot by either model.

**Evidence:** `results/schema_guided_kv/`, `results/schema_guided_kv_q3vl8b/` (registered runs), `results/study25_stage0/` (fallback rescoring, `analysis_confirm.json`, `analysis_dev.json`, router inputs and outputs).

---

## 26. Table-type links: link-preserving crops and relation-rich synthetic pretraining — GPU (Kaggle T4 × 2), registered 2026-11-08

**Question.** Study 22 showed where the joint LiLT linker loses links on XFUND: partners below the source (recall 49, against 79 for same-line partners), partners five or more entities away (recall 49) and question->question links, the heading->label links that tables are built from (recall 34). Study 19's best end-to-end link F1 is 47.7. Do two training-data changes raise end-to-end link F1, and the recall on downward, far-partner and question->question links, with the model and decoding left as they are?

**Sources.** Two recent papers give the ideas:
- *Identify, Locate, Link* (ICDAR 2026, arXiv 2608.20868): crop the page around groups of linked key-value pairs, merge groups that sit close together, add random padding. It also reports that synthetic filling raised XFUND relation extraction from 0.171 to 0.206.
- *RIDGE* (CVPR 2025, arXiv 2504.10659): synthetic documents whose links come from nested headings; pretraining on them helped LayoutLMv3 few-shot entity tagging.

Neither paper shows a relation-extraction gain for a LiLT-style linker, and neither has a no-crop baseline or a template-disjoint split. This study is a first test of the ideas for our model, and a null or negative result is reported as it comes out.

**Naming.** The crops are called link-preserving crop augmentation. They change the training data and leave the model unchanged, which keeps them apart from the earlier graph models (RSA-GNN, NER-fused GNN, Graph-BROS).

**Arms (XFUND es / fr / de / it / pt, seeds 13 / 21 / 42, the Study 19 `windows` recipe from `configs/xfund_joint_lilt_windows.yaml`, FP32):**
- **A0**: the Study 19 recipe, retrained in the same run. Study 19's weights are not available locally and the group recalls need predictions. A0 should land near Study 19's 47.7 link F1.
- **A1**: synthetic pretraining stage, then the A0 recipe.
- **A2**: the A0 recipe with link-preserving crops added to the training set.
- **A3**: A1 and A2 together.

Run order: A0, A2, A1, A3. Runs are resumable per seed. An arm that does not finish within the Kaggle quota is reported as not run.

**Evaluation.** `scripts/xfund_e2e_eval.py`: 250 XFUND val forms, entities from the model's own tags, windowed inference (510 tokens, stride 128), all gold links. Seed-matched delta against A0.

**Registered criteria (primary cell: end-to-end link F1, windowed inference, all 250 val forms, all gold links, seed-matched delta vs A0, in F1 points):**
- *helps:* mean delta >= +2.0 and positive in every seed.
- *no_better:* |mean delta| < 1.0, or the signs differ across seeds.
- *worse:* negative mean with the same sign in every seed.
- *inconclusive:* anything that is not helps, no_better or worse.
- Verdicts are checked in this order: helps, no_better, worse, inconclusive; the first that matches applies.
- **A0 sanity gate:** A0 must reproduce Study 19's windowed end-to-end link F1 (47.7) within 1.5 points; if it does not, the other arms are reported but no verdict is drawn until the difference is explained.

**Secondary cells (reported for every arm):**
- End-to-end recall of gold links by group: `below`, `same_line`, `above`; `rank_0`, `rank_1`, `rank_2_4`, `rank_5plus`; `q_to_q`.
- Entity F1 (typed), `link_qa` F1, and the sibling-free subset.
- Link F1 on a synthetic test set made of the two held-out families.

**Crop parameters.** Up to 4 crops per training form. Links form a graph over the form's entities; connected components are merged when their bounding boxes are within 50 of 1000 units; each region gets random padding of 10-60 units, clamped to the page. A crop keeps the words whose centre lies inside the region, keeps an entity only if all its words are kept, and keeps a link only if both ends are kept. Boxes are re-normalised to the crop frame (0-1000). Crops are added to the full-form windows, never replacing them, are built from the training set only, and are fixed once per seed.

**Synthetic data.** About 3,000 training documents from five families (invoice, financial_statement, procurement_order, adbuy_contract, fara_registration) and about 300 test documents from two held-out families (shipping_manifest, medical_intake) that never appear in training. New relation structure: nested sections whose label links to child field labels (question->question), a key-above-value layout for downward links, and taller tables so a header links to far-away cells. Generated with `--no_augraphy` (LiLT reads no pixels). Values use Faker locales es_ES / fr_FR / de_DE / it_IT / pt_PT; keys stay in English. Pretraining uses the three-stage curriculum shortened to 2 / 2 / 3 epochs, validated on a 15% share carved from the training families by the funsd loader (the held-out families shipping_manifest and medical_intake are used only as the synthetic test set, never for checkpoint selection), then the A0 recipe starts from the pretrained weights.

**Limits stated up front.** 3 seeds and no intervals; synthetic forms have English keys while XFUND forms do not; the papers do not support a relation-extraction gain for this kind of model; the crop and synthetic-data parameters are our own choices; A1 and A3 also get more training steps than A0, and no step-matched control is run (Study 21 found extra steps alone were worth about +1.2 link F1).

**Checks done before any GPU spend.** The unit tests pass: test_table_crops (8), test_table_train_hooks (4), test_synthetic_generator (11), test_table_eval (3) and test_e2e_link. A CPU smoke run of all four arms on seed 13 also passed (3281 s). It trained a tiny synthetic pretrain, loaded its checkpoint through the real `init_checkpoint` path for A1 and A3, filled every group cell on 3 val forms per arm, and ran the synthetic-test evaluation. Each arm config was diffed against `configs/xfund_joint_lilt_windows.yaml` and differs only in `output_dir`, `crop_aug` and `init_checkpoint`. None of this says anything about the arms' scores, only that the pipeline runs end to end.

**Pre-run amendment (2026-11-08, before any GPU run).** A final review of the branch found three things the registration above should have said. First, A2 and A3 train on about 2.13 times as many windows per epoch as A0 (3,276 against 1,537, measured with the real tokenizer), because the crops are added to the forms and the epoch count stays at 22. They therefore take about twice the gradient steps. Study 21 found that extra steps alone are worth roughly +1.2 link F1, so this is a confound for A2 and A3, on top of the extra pretraining steps that A1 and A3 get. A gain for A2 or A3 cannot be read as the effect of crops alone. Second, a crop is rescaled to the 0-1000 frame separately in x and y, so word boxes inside crops come out much taller than on full pages (median height about 60 against about 12). LiLT's absolute layout embeddings and the geometry features are not scale-free, so a weak A2 could reflect that mismatch and not a failure of crop augmentation. Third, the A0 sanity gate is now computed automatically: summary.json holds an a0_gate entry (A0 primary F1, the 47.7 reference, the 1.5 tolerance, pass or fail), and when it fails every arm's verdict is written as no_verdict_a0_gate while the deltas are still reported. The summary also records n_seeds per arm and warns when fewer than 3 seeds are matched.

**Results (run 2026-11-08 to 2026-11-09, three Kaggle sessions).**

A0 gate: A0 scored 47.42 against Study 19's 47.7, inside the 1.5 point tolerance, so verdicts are drawn.

| Arm | Link F1 (primary) | Delta vs A0 per seed (13 / 21 / 42) | Mean delta | Verdict |
|---|---|---|---|---|
| A0 | 47.42 | | | reference |
| A1 (synthetic pretraining) | 47.50 | -1.17 / -0.34 / +1.73 | +0.07 | no_better |
| A2 (crops) | 49.70 | +1.36 / +2.27 / +3.19 | +2.27 | helps |
| A3 (both) | 49.89 | +0.89 / +2.76 / +3.76 | +2.47 | helps |

Group recall (end to end, windowed, all forms; mean delta vs A0 in points):

| Group | A0 recall | A1 | A2 | A3 |
|---|---|---|---|---|
| below | 39.77 | +0.33 | +1.78 | +2.68 |
| same_line | 63.47 | -0.87 | +0.66 | +1.19 |
| above | 34.77 | +0.00 | +4.04 | +4.55 |
| rank_0 | 57.05 | -0.95 | +1.01 | +1.44 |
| rank_1 | 55.92 | -1.13 | +1.60 | +2.86 |
| rank_2_4 | 54.22 | -0.00 | +1.02 | +1.93 |
| rank_5plus | 35.83 | +0.72 | +2.03 | +2.71 |
| q_to_q | 31.20 | +0.50 | +0.84 | +2.43 |

Other secondary cells (means over 3 seeds):

| Cell | A0 | A1 | A2 | A3 |
|---|---|---|---|---|
| Entity F1 | 75.84 | 75.32 | 76.95 | 76.57 |
| link_qa F1 | 53.15 | 53.03 | 54.89 | 55.13 |
| Sibling-free link F1 | 46.34 | 46.46 | 48.37 | 48.62 |
| Synthetic held-out link F1 | 49.11 | 45.76 | 38.49 | 46.16 |
| Synthetic held-out entity F1 | 80.34 | 73.28 | 75.40 | 73.45 |

What this says:
- The crops carry the gain. A2 clears the registered bar (mean +2.27, positive in every seed), though not by much. The groups that improve most are the ones Study 22 flagged: partners above (+4.0), far partners (+2.0) and partners below (+1.8). Same-line pairs barely move.
- Synthetic pretraining alone does nothing measurable on XFUND (A1: +0.07, seeds disagree).
- A3 against A2 is a post hoc comparison: per seed -0.47 / +0.49 / +0.57, mean +0.20, which would read no_better under the registered rule. A3's group recalls are a little higher than A2's across the board, but with 3 seeds that is not evidence of an added effect from pretraining.
- On the synthetic held-out families every trained arm is below A0, and A2 is far below (-10.6). The per-seed spread is large (A2: -1.5 / -16.9 / -13.5; A3: +4.0 / -1.7 / -11.2), so this cell is noisy, but the direction is consistent. Crops help on real XFUND forms and hurt on unseen synthetic layouts. One untested explanation is the crop rescaling, which makes word boxes about five times taller than on full pages. A1's drop is also odd, since it was pretrained on synthetic forms; fine-tuning on XFUND afterwards may simply overwrite what it learned, and the held-out families differ from the training families.
- The step confound from the pre-run amendment applies to A2 and A3: they took about twice the gradient steps of A0, and this study cannot separate crops from extra steps. Study 21 found extra steps alone worth about +1.2 for the windows recipe, so a step-matched control is the obvious follow-up.

Run notes:
- Sessions: A0 and A2 in session 1 (about 3.4 h and 5.8 h), pretrain and A1 in session 2, A3 in session 3. Pretrain weights are never pushed, so session 3 re-ran pretraining; A3 therefore starts from a different pretraining run than A1 (same seeds and config, GPU non-determinism aside).
- Sessions 2 and 3 could not push: a conflicted `git pull --rebase` left `.git/rebase-merge` behind and every later push failed. The four evaluation JSONs for A1 and A3 were downloaded from the sessions' output by hand. They passed through a file-format converter on the way, so they were checked before use: the same key structure as A0's files, 9,062 gold links per seed, 250 / 300 forms, and the per-seed values agree with the means. The A1 and A3 training folders (`metrics.json`, per-seed `history.json`) and those sessions' `run_status.json` were not recovered. The push step was fixed afterwards (commit 0f35b99).
- `summary.json` was rebuilt locally from the four `*_e2e.json` files.

**Evidence:** `results/table_links/` (A0 to A3 `*_e2e.json` and `*_synth_e2e.json`, `summary.json`; training folders for A0 and A2 only).

## 27. Zero-shot VLM key-value linking on XFUND, scored with the LiLT end-to-end scorer — GPU (Kaggle T4 × 2), registered 2026-11-13

**Question.** Studies 6-18 and 25 used a VLM (Qwen2.5-VL-3B) for field extraction on VRDU and FATURA, but never on the task the LiLT work is measured on: finding the header / question / answer entities of an XFUND form and linking each key to its value. How well does a zero-shot open VLM do on that task, scored by exactly the same end-to-end rules as LiLT (Studies 19 and 26)? The goal is a fair attempt at the VLM route, not a win; no "helps" verdict is drawn.

**Levers from `NOVELTY_ASSESSMENT.md` §7.4.** A larger VLM (7B instead of 3B) and grounded output (the VLM cites OCR word indices instead of retyping text), neither tried before.

**Arms (XFUND val es / fr / de / it / pt, zero-shot, no training, greedy decoding):**
- **V2 (main): image + numbered OCR words, end to end.** The prompt shows the page image and the form's words in `ex.words` order (the order LiLT reads), grouped into visual lines written as `[a-b] (x,y) word word ...`. The VLM lists entities as `E<n> <H|Q|A> <a>-<b>` (inclusive word indices) and links as `E<n>>E<m>`.
- **V3: image + numbered correct entities, links only.** The prompt shows the image and every annotated entity (including `other`, as the linker got in Study 17) as `e<n> <label> (x,y) text`. The VLM lists links as `e<n>>e<m>`. Entity F1 is not reported (it is given).
- **V1: image only, end to end.** The VLM reads the page itself and lists entities as `E<n> <H|Q|A> | <text>` plus links. Each text is mapped onto the first unused exact match of its words in `ex.words` (case, accent-insensitive punctuation-stripped comparison); an entity that matches nowhere is dropped and counted.

Each prompt has one short worked example made up by us, not taken from XFUND. A link written value->key (A>Q, or Q>H) is flipped to key->value before scoring, using the VLM's own labels (V3: the given labels); this is part of the decode rule, the same for every arm.

**Model and budget.** Qwen2.5-VL-7B-Instruct in float16, split over both T4s (`device_map="auto"`), `max_pixels` 1280·28·28. If the smoke cell's timing projects V2 on 250 forms above 6 h, Qwen2.5-VL-3B-Instruct is used instead and the write-up says so. Forms run in one fixed order (seed 0 shuffle, cycling through the five languages) so a run cut short stays balanced. Order and caps: V2 on all 250 forms (cap about 6 h), V3 on all 250, V1 on the first 100 in the order. Whatever finishes is reported with its form count; comparisons between arms use the forms they share.

**Registered cells.**
- **Primary:** V2 end-to-end link F1 with `src/e2e_link.py` unchanged: entities need the exact word range and label, links the exact source and destination ranges; gold is every annotated link. 95% interval by bootstrap over forms (1,000 resamples, seed 0). Reported next to LiLT A0 47.4 and A2 49.7 (Study 26, 250 forms, mean of 3 seeds); if V2 covers fewer than 250 forms, that is stated next to the comparison.
- **Secondary, every arm:** entity F1 (V1, V2), `link_qa` F1, the sibling-free subset, recall by group (below / same_line / above, rank_0 / rank_1 / rank_2_4 / rank_5plus, q_to_q), the parse-failure rate (outputs with no parseable entity or link line) and the share of out-of-range or unmatched items. Failed outputs count as empty predictions; no form is dropped.
- V3 link F1 is read against LiLT with gold entities (Study 17, about 59.8 with windowed inference).

**Limits stated up front.** One model, one prompt per arm, greedy decoding; the prompts and the output format are ours and were not tuned on XFUND val; V2 gets the words in annotation order, the same hint LiLT gets; a VLM can be strong on content and weak on exact spans (Study 9), and this scorer counts every boundary miss as wrong.

**Evidence:** `results/xfund_vlm/` (after the run): `{V1,V2,V3}_raw.jsonl` (raw generations), `{V1,V2,V3}_e2e.json`, `run_status.json`.

**Pre-run amendment (2026-11-13, before any GPU run).** Three details changed while the pipeline was built and checked on CPU (none of them used a VLM answer), followed by the measured ceilings and the checks done.
- **V2 output format.** Measured with the Qwen tokenizer, a perfect answer in the registered `E<n> ...` format has a median of 746 tokens (90th percentile 1,545), and because entities come before links, an answer cut off by the token limit would lose every link. V2 now writes one self-contained item per line: a linked pair `Q 2-2 > A 3-5` (both ends become entities) or a lone entity `H 0-1`; the first label written for a word range wins. A perfect answer is now 549 tokens median (mean 670), 13 of 250 forms exceed the 1,536-token limit, and a cut-off answer keeps all its complete lines. V1 keeps the `E<n>` format (limit 2,048 tokens) and V3 writes `e<n>>e<m>` (limit 768).
- **V1 text matching.** A V1 entity text is matched to its first free exact occurrence searching forward from the previous entity (reading order, wrapping around), after removing case, accents and punctuation. Symbol-only words that the VLM cannot retype are then attached by rule: checkbox marks and bullets to the entity on their right, lone punctuation (`:`, `)`) to the entity on their left. Measured ceiling with perfect answers on the 250 forms: 93% of tagged entities recovered exactly; on 60 forms the perfect V1 answer scores about 90 link F1. This ceiling is part of the image-only arm and is reported with it.
- **Model choice rule, made exact.** The smoke cell runs 3 V2 forms with the 7B model; projected V2 hours = 250 × (smoke seconds per generated token) × 670 tokens / 3600. Above 6 h, the run uses Qwen2.5-VL-3B-Instruct. `results/xfund_vlm/model_choice.json` records the projection and the model.
- **Ceilings for reference.** A perfect V2 answer scores 99.8 link F1, not 100, because links touching untagged `other` entities cannot be written (the same holds for LiLT end to end); V3 can reach 100. Forms run in chunks of 50, each pushed before the next starts.
- **Checks done.** `tests/test_vlm_link.py` passes (form order, line grouping, perfect answers score perfectly, bad or cut-off answers never crash). A CPU run of notebook cells 4-6 with a stub that writes perfect answers produced every output file. The official `{lang}.val.zip` image archives download and every val form's image is found by name (checked for es).

---

## 29. LoRA-tuned VLM with grounded word-span output: VRDU unseen- vs mixed-template learning (Track E, primary) and XFUND end-to-end linking (Track L) — GPU (Kaggle T4 × 2), registered 2026-11-15

(Study 28 is reserved for the step-matched control of study 26; it is not registered yet.)

**Question.** Studies 6-12 and 27 used a VLM zero-shot. The nearest published system, *Identify, Locate, Link* (arXiv 2608.20868), reports that fine-tuning a small VLM beats zero-shot Qwen2.5-VL-7B on XFUND links under a layout-aware metric (0.206 vs 0.123; their metric, not ours). Nobody on this project has fine-tuned one. Does a Qwen2.5-VL-3B tuned with LoRA on N = 200 documents, reading the page image plus the numbered OCR words and answering with word spans, (a) beat or match BROS on **unseen templates** (UTL), and (b) **shrink the MTL-UTL gap** (BROS: 92.4 vs 71.3, gap 21.1)?

**Track E (primary): VRDU `registration-form`, Matched-Size protocol.** Same documents, gold and metric as studies 9 and 12; BROS Track-1 (seeds 13 and 42) is the reference on the same documents.
- *Training sets:* the official split files `FARA-lv3-unk_<T>-train_200-...-SD_0.json` for the three held-out templates T (Amendment, Dissemination_Report, Short-Form), and `FARA-lv2-mixed_template-train_200-...-SD_0.json` for MTL. The strict protocol is applied (no held-out-template document in the validation part). Checked on the local files: for every T and every split seed, the 100 dev documents and the 100 confirm documents of T share no document with the training part.
- *UTL test set:* the 100 dev documents (`results/matched_eval/spans.jsonl`) and the 100 confirm documents (`results/confirm_v2/matched/spans.jsonl`) of the held-out template, 600 documents over the three folds. They hold BROS's predictions and the gold spans.
- *MTL test set:* the 300 documents of the lv2 split's test part (gold from the loader tags; BROS MTL is the study-9-era figure 92.4, mean of 9 runs, which is not paired with this set).
- *Up to 4 pages per document* (as study 25); gold words on later pages are unreachable and still count as misses.

**Prompt (frozen before any run; the same text is used zero-shot and for tuning).** One call per page. Words of the page are numbered from 0 on that page and grouped into visual lines (the grouping of `src/vlm_link.py`), `[first-last] (x,y) words`, x and y on the 0-1000 scale. The schema is the six field names with the one-line descriptions of `results/zero_shot_kv_v2/schema_desc.json` (written from field names only). The answer is one line per field, `field | a-b, c-d` (word ranges, several allowed) or `field | NONE`. Gold targets come from the loader's BIO tags. The image is the PDF page at 150 dpi, at most 640 x 28 x 28 pixels.

**Tuning recipe (fixed, nothing tuned on test data).** Qwen2.5-VL-3B-Instruct, LoRA r = 16, alpha = 32, dropout 0.05 on the attention and MLP projections of the language model, vision tower frozen; fp16 autocast with fp32 adapter weights and a gradient scaler; AdamW, lr 2e-4, cosine, 3 warm-up percent, 3 epochs, batch 1 with 8 accumulation steps, loss on answer tokens only, seed 13; the last epoch is the checkpoint. Greedy decoding. If the loss goes non-finite or the output is garbage in fp16, the recipe may be changed only by a dated amendment made before any test document is scored.

**Arms:**
- **E0** zero-shot control: the same prompt, untuned model, confirm documents only (300).
- **E1 (primary)** LoRA, UTL: three trainings (one per held-out template, split seed 0), scored on the 600 documents.
- **E2** LoRA, MTL: one training, scored on the lv2 test part.
- **E3** (ablation, runs last, descriptive): E1 without the image (words only), same recipe.

**Decode.** Page answers are mapped to document word indices; one span per field per document by majority over pages, ties to the earliest page (the rule of study 25); overlapping spans are dropped; an unparseable page counts as empty. Scored with `schema_guided_kv.py score`, i.e. exact-span micro F1 (`matched_eval.py`) with a paired bootstrap over documents (1,000 draws).

**Gate (before any F1 is read).** Page-level parse rate >= 90% and fewer than 5% of pages with an error; otherwise the failure is reported as the finding.

**Registered criteria (primary cell: pooled UTL exact-span micro F1, 600 documents, E1 minus the mean of BROS seeds 13 and 42, paired):**
- *Beats BROS:* delta >= +3.0 points, bootstrap lower bound > 0, and no template's upper bound < 0.
- *Competitive:* |delta| <= 3.0 or the interval contains 0.
- *Worse:* otherwise.
- *Gap closes:* (E2 MTL F1) - (E1 UTL F1) <= 10.5 points, half of BROS's 21.1. Reported with the caveat that the two test sets differ.
- Descriptive: E1 vs E0 (value of tuning), E1 vs E3 (value of pixels), per-field F1 (notably `foreign_principle_name` and the Short-Form `registrant_name` decoy), per-template F1, parse and grounding rates, training loss, GPU time and memory.

**Track L (secondary, a separate Kaggle session): XFUND end-to-end links.** The V2 prompt and parser of study 27 (`src/vlm_link.py`), unchanged, with LoRA tuning on the XFUND train split (745 forms; the same recipe, 2 epochs), scored on the 250 val forms with the study-26 scorer, compared with LiLT A0 (47.4) and A2 (49.7), plus the zero-shot V2 numbers of study 27 if they exist. Registered rule: *helps* = link F1 >= LiLT A0 + 3.0 with the bootstrap lower bound above A0; *competitive* = within 3.0; otherwise *worse*. Extra generalization arms, descriptive: train on four languages and test on the fifth (the held-out language); train on XFUND and score FUNSD test. Its scripts are built after Track E has run.

**Checks before any GPU spend.** Unit tests (no model): the page builder and word numbering, line grouping, the gold-target round trip (BIO tags -> target text -> parsed spans reproduce the dev-set gold of `results/matched_eval/spans.jsonl` exactly), the parser on valid, invalid and cut-off answers, and the page-to-document decode; the notebook's smoke cell trains 3 steps and generates 2 pages on the GPU and raises on failure.

**Limits stated up front.** One model size, one recipe, one training seed per cell (split seed 0), so no seed variance; BROS used its own seeds on possibly different training documents; the MTL arm is compared with a 9-run BROS mean on a different test set; N = 200 documents (about 500 page samples) may overfit; Qwen may have seen public XFUND/FUNSD pages (VRDU much less likely), which is why Track E is primary; word indices rely on the loader's OCR order; the image helps only if E1 beats E3; GPU hours are an estimate (about 1.5-2 h per training and 1 h per fold of scoring, unmeasured).

**Pre-run amendment (2026-11-15, before any GPU run): two simultaneous sessions and a replication seed.** (1) Track E runs as two Kaggle sessions at the same time with the same notebook (`SESSION` = A or B): A trains E1 (split seed 0) and then runs E0 and E3; B trains E1r and E2. (2) E1r is a replication of E1 with split seed 1 (the official `SD_1` split files) and training seed 14, everything else identical. The registered primary cell, verdicts and the *gap closes* rule are unchanged and drawn on E1 (seed 0). New rule: a verdict of *beats BROS* is reported as **replicated** only if E1r has the same sign of delta against BROS and its bootstrap lower bound is above 0; otherwise it is reported as not replicated. Seeds per cell: 2 for E1, 1 for E2, E0 and E3 (BROS has 9 runs per regime, so this is still thinner than the reference). Reason: the earlier registration gave one training seed per cell, which cannot show whether a difference is seed noise.

**Pre-run amendment 2 (2026-11-15, still before any GPU run): a larger tuned part, a lower learning rate, a vision-adapter arm and a collapse check.** The tuning recipe above is replaced for every arm by: LoRA **rank 64, alpha 64** (adapter scale 1, was rank 16, alpha 32) on the attention and MLP projections of the language model; the **vision-to-language projector (`visual.merger`) trained in full** (fp32 weights, saved with the adapter); vision tower otherwise frozen; **learning rate 1e-4** (was 2e-4, because four times the adapter capacity at the old scale and fp16 on a T4 make larger steps riskier; the only comparable published setup we found used 5e-5). Reasons: avoid underfitting (the first recipe trained about 1% of the model, the same order as the Graph-BROS v1 run that collapsed, though that run failed for other reasons) without a second round of GPU time. New arm **E4**: E1 plus LoRA adapters (rank 64) on the vision tower's attention and MLP; descriptive, runs last in session B, so only folds 0 and 1 are expected to finish, and it carries its own gradient check (PEFT issue 2880: with gradient checkpointing, vision-side adapters on Qwen2.5-VL can train with zero gradient). Safeguards added to every training: (i) the training aborts if no gradient reaches the LoRA weights, the projector or the vision adapters within the first 3 optimizer steps; (ii) the trainer checks that the layer-name patterns matched language-model modules (and vision modules only for E4); (iii) **collapse check**: after training, 20 validation documents from the strict split (no held-out-template document) are decoded with the model in memory and scored; the training counts as finished only if their F1 is at least 0.10, otherwise the run raises and no scoring uses it. The primary cell, the verdicts, the gap rule and amendment 1 are unchanged. The earlier-stated reference to "the prompt is frozen" still holds: no prompt change.

**Clarification and smoke-test findings (2026-11-15/16, from the first GPU smoke run, no test document scored).** (1) The strict protocol leaves **180 training documents** per UTL fold, not 200: the official validation part of each `lv3` split holds only held-out-template documents, so 20 training documents are carved out as validation (the collapse check uses them). BROS's matched-size runs build their own 200-document training sets (`matched_size_protocol`), so the two reference systems differ by 10% in training data; this is a limit. (2) The smoke run trained on 3 steps and hit **out-of-memory on 2 of 6 pages** at peak 13.2 GB, caused by transformers 4.51.3 building full-sequence float32 logits for the loss. Fixed before any real run: the LM head now computes logits only for the answer positions and the loss is cross-entropy on them (the same loss as before); the fraction of skipped samples is recorded in `train_done.json` and printed in the notebook. (3) Measured: 156.4M trainable parameters (4.04% of 3.87B), 252 language-model LoRA modules, 0 vision modules (E4 adds vision modules), gradients reach the adapters (norm about 3) and the projector (about 0.4), and the zero-shot untuned model does not follow the number format (it writes names), so E0 will have a high parse-failure rate by design of the prompt, not of the run.

**Results (Track E; Kaggle sessions A and B; E3 not run; Track L not built).** Gate first: page parse rate 100% and 0% errors for E1, E1r, E2 and E4; **E0 (untuned model, same prompt) parses only 63% of pages and scores 0.13 F1**, so the untuned model does not follow the word-number format and the tuning is what makes the format work at all. Every training finished (130-153 optimizer steps, 1.1-1.5 h each, E4 1.8-1.9 h; peak 10.8-11.7 GB; no skipped samples; training loss 0.11 -> 0.035 -> 0.017 over the 3 epochs; collapse-check F1 on the 20 validation documents, which come from the training templates, 72-87).

| Arm | Documents | Exact-span micro F1 [95% CI] | Reference |
|---|---|---|---|
| **E1** (UTL, split seed 0) | 600 | **46.24** [44.53, 47.97] | BROS (mean of seeds 13 and 42) 70.13; study-8-12 zero-shot VLM pipeline 57.44 |
| **E1r** (UTL, split seed 1) | 600 | **44.66** [43.11, 46.36] | same |
| **E2** (MTL) | 300 (lv2 test part) | **76.12** [73.58, 78.70] | BROS MTL 92.4 (9-run mean, different test set) |
| E4 (UTL + vision adapters, folds 0 and 1 only) | 400 | 48.65 [46.64, 50.80] | BROS on the same two folds 75.6 |
| E0 (untuned) | 300 (confirm) | 0.13 [0.0, 0.41] | parse rate 63% |

**Primary cell: Worse.** E1 minus BROS = -23.9 points, interval of the difference [-25.9, -21.6]; E1r minus BROS = -25.5 [-27.5, -23.4]. By template (E1 / E1r / BROS seeds 13 and 42 / zero-shot VLM pipeline): Amendment 42.3 / 38.3 / 84.0 and 85.0 / 65.1; Dissemination_Report 45.7 / 48.4 / 64.2 and 71.8 / 67.2; Short-Form 50.3 / 46.3 / 58.7 and 58.7 / 41.0. The tuned VLM is below BROS on every template and in both seeds; the only place it beats a reference is Short-Form against the zero-shot VLM pipeline (+9 and +5). **Gap rule: not met.** E2 - E1 = 29.9 points (rule: at most 10.5; BROS 21.1, measured on a different test set), because the tuned VLM is also weak on seen templates (76.1 against BROS 92.4), not because it generalizes worse than expected.

**Per field, pooled UTL (E1 / E1r / BROS seed 13 / zero-shot VLM pipeline):** file_date 60.9 / 68.5 / 82.6 / 59.9; foreign_principle_name **48.7 / 7.8** / 1.0 / 57.3; registrant_name 33.9 / 26.8 / 56.1 / 63.7; registration_num 74.3 / 82.5 / 96.9 / 95.0; signer_name 14.9 / 18.6 / 66.0 / 27.9; signer_title 15.8 / 18.8 / 50.8 / 5.6. The one field where a tuned seed beats BROS by a wide margin is foreign_principle_name, the field BROS never predicts, but it does not replicate: 48.7 in E1 and 7.8 in E1r (E1r predicts nothing for 89% of gold occurrences), and the zero-shot pipeline (57.3) is better than both.

**Post hoc diagnostic (written after seeing the results; computed from `spans_E1.jsonl` and `spans_E1r.jsonl`, same documents).** Each gold span classified by what the model predicted for that field (E1 / E1r): exact **42.1% / 38.9%**; overlaps the gold span but not exact **24.5% / 21.6%**; a span in the wrong place 12.0% / 9.6%; **no prediction for the field 21.5% / 29.9%**. A lenient (any overlap) recall would therefore be 66.6% / 60.5% against the 42.1% / 38.9% exact one. By field the pattern differs: registrant_name is found almost always (overlap or exact 95.1% / 73.1%) but is exact only 33.9% / 24.8% of the time, i.e. a boundary problem; signer_title and foreign_principle_name are mostly not predicted at all (75% / 67% and 39% / 89%), i.e. a recall problem. 441 / 394 predictions overlap no gold span.

**Post hoc diagnostics 2 (`scripts/vlm_sft_diagnostics.py`, `results/vlm_sft/diagnostics.json`; CPU, existing outputs; written after the results).**
- **D1, boundary tolerance (all 600 documents).** Micro F1 when a prediction counts as correct if both of its boundaries are within k words of a gold span of the same field (E1 / E1r / BROS seed 13 / BROS seed 42 / zero-shot VLM pipeline): k = 0: 46.2 / 44.7 / 68.8 / 71.5 / 57.4; k = 1: 58.1 / 57.4 / 69.9 / 72.8 / 60.8; k = 2: 63.8 / 62.9 / 70.7 / 73.8 / 64.0; k = 3: **71.2 / 70.1 / 71.8 / 74.5** / 66.6; any word overlap: **73.2 / 69.4 / 72.1 / 75.1** / 71.6. BROS barely moves (its boundaries are already exact); the tuned VLM gains 25 points. E1 near-misses (704): 380 predicted spans are too long, 188 too short, 136 shifted; 36% are within 1 word in total, 70% within 3, 85% within 5. Typical errors: a start that is too late ('Tourist Department' for 'Austrian State Tourist Department Austrian Federal Railways'; 'FJ Macrory' for 'Patrick FJ Macrory') or trailing extra words ('Caroline Cushing 2327'; 'Jared Genser , Associate').
- **D2, deployable snapping (no gold used).** Replacing each E1 / E1r span by the overlapping BROS span of the same field gives 60.3 / 60.2 (from 46.2 / 44.7); by the zero-shot VLM's span 57.0 / 55.3; snapping to visual lines hurts (36.1 / 32.4) because only 46.4% of gold spans are whole visual lines. Snapping to BROS is a hybrid in disguise (the span becomes BROS's), so it shows that boundaries are the problem, not that it is a method: it stays below BROS alone.
- **D3, does the tuned VLM add to the router?** Cross-template router (per-field system chosen on the dev documents of the other two templates, scored on the confirm documents): BROS + zero-shot VLM 73.4 (BROS seed 13) and 75.1 (seed 42); adding E1 as a third system gives 72.0 and 71.4 (differences -1.4 [-2.0, -0.7] and -3.8 [-4.8, -2.8]); adding E1r changes nothing (it is never chosen). An oracle that picks the best system per field per document rises from 80.5 to 83.9 (seed 13) and 81.7 to 85.5 (seed 42) with E1, so it holds some complementary correct spans, but when BROS is wrong the tuned VLM is right in only 14% (E1) and 6-8% (E1r) of cases, against 18-20% for the zero-shot pipeline.

**Reading of the diagnostics.** The tuned VLM finds the right region about as often as BROS does (any-overlap F1 73.2 and 69.4 against 72.1 and 75.1) and loses its 24 points mostly on exact word boundaries, which BROS gets from token-level tagging and our model has to write as word numbers. The benchmark is strict-boundary, so the benchmark rewards exactly the thing this output format is worst at. This is a reading of post hoc numbers, not a controlled test: the any-overlap rule is generous to a model that predicts long spans, and 22-30% of gold fields still receive no prediction. The registered verdict (worse) stands.

**What this shows (and does not).** (1) With this recipe (LoRA on 180 documents, word-number output, exact-span scoring) a fine-tuned 3B VLM does not close the unseen-template gap; it is far below BROS and below our own zero-shot VLM pipeline, so it is not a step forward on this benchmark. (2) The losses are of two kinds: boundary errors on fields the model finds, and missing predictions on the hard semantic fields; the second is unstable across seeds. (3) The very low training loss (0.017) with 72-87 F1 on seen-template validation documents against 44-46 on unseen templates is consistent with the model memorising template-specific positions from 180 documents of two templates; this is a hypothesis, not tested here. (4) Adding vision adapters (E4, one seed, two folds) gave 43.3 and 53.2 on folds 0 and 1 against 42.3 / 38.3 and 45.7 / 48.4 for E1 / E1r; the seed spread is as large as the difference, so there is no demonstrated gain. (5) The result is for the registered recipe only; it does not rule out text-out-and-ground prompting (as in studies 8-9), more data, or other tuning settings.

**Run notes.** `run_status.json` in the repo is session B's (7.72 h); session A's was overwritten by the final rebase (`-X theirs`), but its metrics, gates and rows are all present. Intermediate pushes failed during the run (git identity was set only in cell 7, fixed in `ff9c4fa`), and the E3 arm failed at startup because the projector-gradient guard was wrong for runs without an image (also fixed); neither affects the E1, E1r, E2, E0 or E4 numbers. Training used 180 documents per UTL fold (strict protocol), BROS 200.

**Evidence:** `results/vlm_sft/` (`metrics_*.json`, `gate_*.json`, `spans_*.jsonl`, `rows_*.jsonl`, per-training `train_done.json` and `collapse_check.json`).

---

## 30. Output format of the tuned VLM on VRDU unseen templates: word numbers vs OCR-segment numbers vs copied text with line identifiers — GPU (Kaggle T4 × 2), registered 2026-11-16

**Question.** Study 29's tuned VLM, which answers with word numbers, scored 46.2 exact-span F1 on unseen templates against BROS's 70.1. Post hoc diagnostics on the same outputs showed that it finds the right region about as often as BROS (any-overlap F1 73.2 against 72.1 and 75.1) and that most of the loss is exact word boundaries: with a 3-word tolerance it scores 71.2 (BROS 71.8 and 74.5). If that reading is right, an output format that takes the boundaries from the OCR instead of from the model's word counting should close most of the gap. Does it? Nothing else changes: same model, recipe, training documents, test documents and scorer.

**Arms (UTL only; three folds; split seed 0 is the primary, split seed 1 the replication; run as two simultaneous Kaggle sessions, A = seed 0 and B = seed 1, each training S and T on the three folds):**
- **P** (reference, reused, not retrained): the study-29 arms E1 (seed 0) and E1r (seed 1): word numbers.
- **S** (segment numbers): the words of each page are split into visual lines (`lines_of` of `src/vlm_sft.py`) and every line is cut where the horizontal gap between two consecutive words exceeds **1.0 x the page's median word height** (rule below). Segments are numbered from 0 on each page in reading order and listed as `S<i> (x,y) text`. The answer is, per field, `field | S<a>-S<b>` (a run of consecutive segments, several runs separated by commas) or `field | NONE`. A run maps to the word span from the first word of S<a> to the last word of S<b>. Training target for a gold span: the run of segments it overlaps if it coincides with segment boundaries, otherwise the smallest run of segments that covers it (so a model that learns this format cannot reproduce the boundaries of the gold spans that do not coincide with segment boundaries; this ceiling is reported).
- **T** (copied text with line identifiers, the format of study 25 unchanged): the OCR lines of the page with identifiers `XX|YY` and the JSON schema prompt of `scripts/schema_guided_kv.py`; the answer is a JSON object whose values are the text copied from the line followed by that line's identifier (one `text identifier` per line for a value spanning lines). Training targets come from the gold spans through `gold_occurrences`; decoding is study 25's grounding (the cited line must exist and contain the text after the normalisation of `matched_eval.py`, otherwise the value is dropped), one answer per field by majority over pages (ties to the earliest page).

**Gap rule for S, chosen on training documents only.** The multiple of the median word height was chosen as the one that maximises the share of gold spans that are exact runs of consecutive segments on the 180 training documents of each fold (candidates 1.0, 1.5, 2.0, 2.5, 3.0, 4.0 and no split): 1.0 gives 79.0 / 79.2 / 86.1% on folds 0 / 1 / 2 against 76.8 / 77.3 / 84.1% for 2.0, at about 20% more segments per page. History to disclose: before this check, a coverage table on the 600 test documents (post hoc, gold used) had suggested 2.0; that value is not used.

**Recipe (identical to study 29 after amendment 2):** Qwen2.5-VL-3B-Instruct, LoRA rank 64 / alpha 64 on the language model, projector trained in full, vision tower otherwise frozen, lr 1e-4, 3 epochs, batch 1 with 8 accumulation steps, fp16 autocast, training seed 13 + split seed, answer-position loss, images at 150 dpi up to 640 x 28 x 28 pixels, 180 training documents per fold, collapse check on the 20 validation documents (F1 >= 0.10 or the run raises), gradient guards. Greedy decoding, up to 4 pages per document. Only the prompt, the answer format and the decoder differ between arms.

**Metric and test set.** The 600 UTL documents of study 29 (100 dev + 100 confirm of the held-out template, three folds), exact-span micro F1 as in `matched_eval.py`, paired bootstrap over documents (1,000 draws), BROS (seeds 13 and 42) as the reference.

**Gate (before any F1 is read).** Page parse rate >= 90%; for T, at least 90% of the returned values must ground; error rate below 5%. A failing gate is reported as the finding.

**Registered criteria (primary cell: pooled UTL exact-span micro F1 on the 600 documents, split seed 0, for each of S and T):**
- *Format matters:* the arm's F1 minus P's (E1) >= +5.0 points with bootstrap lower bound > 0, and **replicated**: the same arm in split seed 1 minus E1r has a positive mean difference with lower bound > 0.
- *Beats BROS:* the arm minus the mean of BROS seeds 13 and 42 >= +3.0 with lower bound > 0, and replicated in the same sense.
- *Competitive with BROS:* |difference| <= 3.0 or the interval contains 0.
- *No better than word numbers:* the arm minus P is below +5.0 or its lower bound is <= 0.
- Verdicts are given per arm; if both pass, the larger difference is named.
- Descriptive: per-field F1 (notably `foreign_principle_name` and the Short-Form `registrant_name` decoy), the boundary-tolerance table of `scripts/vlm_sft_diagnostics.py` (k = 0, 1, 2, 3, any overlap), the share of test gold spans that are exact runs of segments (the ceiling of S), the share of values that ground (T), per-template F1, seed spread, and the cross-template router and oracle with BROS and the zero-shot VLM pipeline (as diagnostics 3 of study 29).

**Checks before any GPU spend.** Unit tests without a model: the segment builder and the span <-> run mapping (gold spans that coincide with segment boundaries survive the round trip), the S parser (valid, invalid, cut-off answers), the S ceiling on the training and test documents reproduced from the oracle (gold answers through the S decode path score exactly the coverage figure), and the T oracle (`schema_guided_kv.py oracle`, at least 98 F1 on real documents, which validates the line-to-word mapping). Notebook smoke cell as in study 29: 3-step training, in-memory collapse check, generation.

**Limits stated up front.** One recipe and one training seed per fold per split seed; the tuning set is 180 documents of two templates (the same as study 29); S cannot reach about 20% of the gold spans exactly by construction and handles annotation conventions (parentheses, trailing punctuation) only where they fall on segment boundaries; T inherits study 9's finding that the annotated copy of repeated text is not reliably the first or the last, so grounding can pick the wrong copy; the three arms differ in prompt length as well as in format (S lists segments, T lists lines); P was trained under the same recipe but in an earlier session; no MTL arm (run only for a format that passes); the tolerance and overlap diagnostics that motivate the study were computed after seeing the study-29 results.

**Checks done before any GPU spend (CPU, no model).** (1) Format S: the segments partition every page of the 600 test documents (57.2 segments per page on average); **2,361 of the 2,875 gold spans (82.1%) are exact runs of segments**; the gold answers written in format S and decoded by the same path as a model answer score **83.0 F1** (2,355 correct of 2,798 predicted: the other 17.9% of the gold spans can only be answered with a covering run), so 83 is the exact-span ceiling of S on these documents under the one-span-per-field decode. The S parser handles valid, out-of-range and malformed answers. (2) Format T: study 25's oracle (`schema_guided_kv.py oracle`, gold occurrences through the grounding) gives **99.74 F1** on the 600 documents; the training targets (gold values with line identifiers as JSON) decoded through the same parser, grounding and vote give **99.56 F1**, and every answer parses. (3) The training sets are built for all three formats (361 page samples for the fold-0 training; prompts of about 4-7k characters in every format, answers of 130-210 characters). (4) New code: `src/vlm_sft.py` (segments, prompt, targets, parser for S), `scripts/vrdu_vlm_sft.py` (`--format word|seg|text`), `scripts/study30_verdict.py` (the registered rules, tested here by feeding study 29's own spans in place of an arm), `tests/test_vlm_sft.py`, `notebooks/kaggle_vlm_format.ipynb` (cell 3 trains 3 steps for each new format on the GPU and raises on failure). None of this says anything about the model's scores.

**Results:** *pending*

**Evidence:** `results/vlm_sft30/` (after the run).

---

## 31. Prompt v2 for the schema-guided grounded extractor: Qwen3-VL-8B (4-bit) on VRDU unseen templates — GPU (Kaggle T4 × 2), registered 2026-11-19

**Question.** Study 25's 8B arm (A2: image + OCR lines with ids + schema, grounded decoding) scored 60.5 exact-span F1 with the post hoc fallback decode, against BROS 70.5; its router with BROS gave 76.2 (+5.7). The error review (study 25, stage 0) found prompt-level causes: whole-line copies (`signer_name` too long in 107 of 222 predictions, snapping to BROS lifts it from 41 to 73), the received stamp chosen for `file_date` (22.3 against BROS 84), the signer returned for `registrant_name`, and no line ids in the one-call-per-page JSON format (5.5% of values). Does a prompt that (i) says to copy only the value, (ii) shows one worked example of the id format, (iii) rewrites three field descriptions, raise the 8B's F1 and the router gain? Nothing else changes.

**Prompt v2 (frozen before any run; `build_prompt` / `field_prompt` with `v2=True` in `scripts/schema_guided_kv.py`, descriptions in `results/schema_guided_kv_v2/schema_desc_v2.json`).** Rules in the prompt: copy the value exactly from ONE line followed by that line's id (example `"Jane Q. Public 42|17"`, an invented value); copy only the value, not the label in front of it nor other words on the line (for a person's name leave out the title); several lines as one `text identifier` per line; null if absent; one worked answer for a different, invented document (`EXAMPLE_V2`); the answer skeleton no longer uses empty strings (`"<text> <identifier>" or null`). Descriptions changed (the rest as in study 25): `file_date` = the date on which the form was signed or dated by its signer, as written beside the signature (e.g. 'the 5th day of March, 2012'), not a received or filed stamp; `registrant_name` = the name of the organisation (firm, company or office) that files the registration, not the person who signs it; `signer_name` = the full name of the person who signs, without title or position. **Disclosure:** the `file_date` wording was chosen after seeing study 25's errors on the confirm documents (it names the signing date, which is generic domain knowledge, not a template); the user approved putting it in the main arm. The same applies to the other two descriptions and the copy-only-the-value rule. The confirm set was used to see these errors, so it is no longer a clean untouched test of this prompt; it is the same set on which study 25 was scored, which is what makes the comparison paired.

**Model, data, decode.** Qwen3-VL-8B-Instruct in 4-bit nf4, fp16 compute, as study 25 session B; OCR lines with ids, grounding, vote, up to 4 pages, one sample (K = 1, which the sampling code decodes greedily), the post hoc fallback decode (`--fallback`) for every scored arm, exact-span micro F1 as in `matched_eval.py`, paired bootstrap over documents (1,000 draws), the 300 confirm documents only (no dev set, no multi-sample vote).

**Arms (`notebooks/kaggle_prompt_v2.ipynb`, one session):** **V2A2** (primary): image + OCR lines + schema, prompt v2. **V2A1**: the same without the image (descriptive: V2A2 - V2A1). Reference arm (not re-run): the study-25 8B A2 spans decoded with the fallback (`results/study25_stage0/8B_A2_confirm/spans_A2.jsonl`).

**Pilot and format gate.** The 30-document pilot of study 25 (same manifest, checked identical in the notebook). The JSON-per-page format is kept if at least 90% of outputs parse and at least 90% of values ground with a valid id (strict decode, no fallback), otherwise the one-call-per-field format is used, as in study 25. **Pilot guard:** the pilot is decoded with the fallback and scored per field with the official-style text match (`scripts/pilot_fieldwise.py`); study 25's prompt on the same 30 documents gives micro 60.9 (`results/schema_guided_kv_v2/pilot_v1_baseline.json`: file_date 25.5, foreign_principle_name 73.7, registrant_name 56.6, registration_num 100, signer_name 41.9, signer_title 58.8). If v2's pilot micro is more than 5.0 points below 60.9 the main arms are not run (a guard against a broken prompt, not a significance test: 30 documents).

**Registered criteria (primary cell: pooled exact-span micro F1, 300 confirm documents, V2A2 minus the study-25 8B A2 arm, both with the fallback decode, paired):**
- *Helps:* delta >= +3.0 points, bootstrap lower bound > 0, and no template's upper bound < 0.
- *Clears the bar:* V2A2 alone >= 66.0 exact-span F1, or the static cross-template router (as studies 11-12, `confirm_eval.py`) with the V2A2 spans gains >= +7.0 over BROS (study 25: 60.5 and +5.7).
- *No better:* otherwise.
- Descriptive: per-field F1 (notably `signer_name`, `registrant_name`, `file_date`), V2A2 - V2A1, JSON gate and decode counters (ids, grounded, dropped overlaps), comparison with A0 and BROS, GPU time per arm.

**Checks before any GPU spend.** The self-test of `schema_guided_kv.py` passes with new assertions (the v2 prompt contains the id example, the invented example, the descriptions and the line ids; the v1 prompt is unchanged); the run loop with the new flags was exercised with the stub model; the pilot baseline was computed on the saved study-25 rows; the notebook cells compile. The model path is the study-25 session-B path (Qwen3-VL-8B 4-bit), which ran to completion there.

**Limits stated up front.** One model, one prompt variant, one sample: a change in F1 cannot be attributed to the individual rules, the example or the descriptions; the prompt and descriptions were written after seeing study-25 errors on the same confirm documents (see above); the 4-bit and fp16 loading effect is not measured; the `file_date` description names the signing date, which may not hold for every VRDU template and was not tested on other datasets; the decode fallback is post hoc (introduced in study 25's stage 0); no dev set, so no selection between prompts was made.

**Results (confirm set; Kaggle run, 4.03 h; commit `d39dd5d`).** *Format gate:* the JSON-per-page format parsed 100% but only 84.4% of values grounded with a valid id (146 of 173; 22 not in the cited line, 5 unknown ids), up from 5.5% in study 25 and just below the 90% threshold, so the registered rule switched to one call per field. *Pilot guard:* v2 scores 73.9 micro on the 30 pilot documents against 60.9 for study 25's prompt (guard passed). The 4-hour deadline cut the text-only arm off: **V2A2 has all 300 documents, V2A1 only 212**. GPU time (sum over both GPUs): V2A2 5.28 h, V2A1 1.72 h for 2,617 units, pilots 0.8 h.

| Arm (confirm) | Exact-span micro F1 | Reference |
|---|---|---|
| **V2A2** (prompt v2, image) | **76.5** | study-25 8B A2 (same fallback decode) 60.5; BROS 69.0 / 72.0; A0 57.7 |
| V2A1 (212 documents) | 73.6 (on those documents V2A2: 76.6) | |

**Registered criteria (primary: V2A2 minus study-25 8B A2, paired):** delta **+16.0 [+13.7, +18.4]**, positive on every template (Amendment 75.7 vs 59.3, Dissemination_Report 86.2 vs 73.8, Short-Form 65.5 vs 44.3): **helps**. **Clears the bar:** V2A2 alone 76.5 >= 66.0, and the static router with the V2A2 spans scores 79.2 against BROS 70.5, **+8.7 [+6.9, +10.6]** (>= +7.0; study 25: +5.7). V2A2 minus BROS (mean of seeds 13 / 42): +6.0 [+3.5, +8.5] (the interval of `new_minus_bros`), but **below BROS on Amendment** (75.7 vs 84.5 / 85.9), so the study 11-12 router rule 'no template significantly worse' is not met (`hybrid_V2A2.json` primary passes = false). Image effect on the 212 documents both arms cover: V2A2 - V2A1 = **+3.1 [+1.2, +5.0]**; the full text-only comparison is not available.

**Per field (V2A2 / study-25 8B A2 / BROS seed 13):** file_date 79.5 / 22.3 / 84.0; foreign_principle_name **52.5 / 70.4** / 2.0; registrant_name **90.4** / 62.3 / 56.9; registration_num 97.2 / 97.3 / 97.4; signer_name 53.4 / 41.1 / 64.9; signer_title 75.1 / 73.5 / 50.3. The prompt fixed the field-choice errors (`registrant_name`, `file_date`) but **regressed `foreign_principle_name`** (54 predictions too short, 64 predicted where the field is not annotated; the copy-one-line rule truncates multi-line names) and `signer_name` is still below BROS (55 predictions in the wrong place, 57 where no gold field exists, 48 missing). Decode counters (V2A2): 1,892 values, 1,734 grounded (45 through the fallback), 136 not in the cited line, 81 dropped for overlap.

**Reading and limits.** A large, paired, per-template-consistent gain from prompt wording alone, with the same model. The prompt and the three descriptions were written after seeing study-25 errors on these same confirm documents, so the confirm set is not an untouched test of this prompt (the dev documents were not used to write it; see the amendment below). One prompt variant: the rules, the example and the descriptions are confounded. The `file_date` gain (22 -> 80) rests on a description that names the signing date, which is Amendment-style knowledge; it was not tested on other templates or datasets. The decode fallback is post hoc (study 25 stage 0). `V2A1` incomplete.

**Pre-run amendment (2026-11-19, after the confirm results, before any dev run): dev replication.** The recipe is frozen as it ran on confirm (prompt v2, descriptions v2, one call per field, fallback decode, 4-bit Qwen3-VL-8B, no pilot) and scored once on the 300 dev documents (`results/zero_shot_kv_v2/manifest.json`, the study-8/9 documents; their error profile was not used to write the prompt, but their study-25 results were seen in the stage-0 summary). Runs: V2A2 on dev, V2A1 on dev, then the 88 cut-off documents of V2A1 on confirm. *Replicates* = V2A2 minus the study-25 8B A2 arm on the dev documents (same decode) >= +3.0 with a paired lower bound > 0, and V2A2 alone >= 66.0. Descriptive: V2A2 - V2A1 on dev and on the completed confirm arm, per-field F1, V2A2 against BROS on dev. No router on dev (the router's fit uses the confirm documents). Notebook: `notebooks/kaggle_prompt_v2.ipynb` (edited in place; the confirm version is in `d39dd5d`), deadline 5.0 h.

**Results of the dev replication (commit `621117d`; 3.93 h; all three runs complete, 300 documents each).** Exact-span micro F1 (fallback decode):

| Arm | Dev | Confirm (completed) |
|---|---|---|
| **V2A2** (image) | **75.2** | 76.5 |
| V2A1 (text only) | 71.7 | 73.3 |
| study-25 8B A2 | 58.1 | 60.5 |
| BROS (seeds 13 / 42) | 68.6 / 70.9 | 69.0 / 72.0 |

**Replicates (registered rule met):** V2A2 minus study-25 8B A2 on dev = **+17.0 [+14.6, +19.3]** (>= +3.0, lower bound > 0, and 75.2 >= 66.0); positive on all three dev templates (Amendment 72.5 vs 56.7, Dissemination_Report 86.2 vs 71.7, Short-Form 65.0 vs 42.9). V2A2 minus BROS on dev: +5.5 (interval of the difference [+2.9, +7.9]); below BROS on Amendment again (72.5 vs 83.6 / 84.2). **Image effect (paired, same documents):** dev **+3.4 [+1.5, +5.4]**, confirm (completed text-only arm) **+3.2 [+1.6, +4.8]**; the earlier +3.1 on 212 documents stands. **Text-only is already competitive with BROS:** V2A1 minus BROS = +1.0 on dev (interval [-0.4, +4.4]) and +2.8 on confirm [+0.5, +5.1], at about 45% of the GPU time (V2A1 2.2-2.3 GPU-h per 300 documents against 4.8-5.3 for V2A2). Per field on dev (V2A2 / V2A1): file_date 82.6 / 77.0, foreign_principle_name 57.4 / 53.6, registrant_name 86.3 / 87.4, registration_num 96.3 / 90.0, signer_name 48.9 / 47.0, signer_title 64.5 / 63.0; the confirm-set weaknesses repeat (`foreign_principle_name` below study 25's 70, `signer_name` below BROS). The image mainly helps `registration_num` (+6.3 on dev, +6.9 on confirm).

**Reading.** The gain from prompt v2 holds on documents the prompt was not written on, with a similar size (+17.0 and +16.0), so it is not a confirm-set artefact; the remaining limits are those stated above (one prompt variant, confounded changes, post hoc fallback decode, `file_date` description names the signing date, one model, 4-bit loading, one sample per page). The dev documents were still used in studies 7-9, and the stage-0 summary of study 25 showed dev scores before the prompt was written, so 'untouched' is not claimed.

**Post hoc stage 0 (CPU; written after seeing the results; `scripts/study31_router.py`, `results/schema_guided_kv_v2/stage0_router.json`): snapping and cross-set routers with the V2A2 spans.** Per-field source choices are fitted on the OTHER document set's documents of the other two templates and scored on the held-out template of this set (dev fitted on confirm, confirm fitted on dev); BROS seeds 13 and 42 averaged; difference to BROS with a paired bootstrap. Exact-span F1 (dev / confirm), V2A2 spans alone 75.2 / 76.5 (+5.4 [+2.9, +7.9] / +5.9 [+3.5, +8.5] over BROS 69.8 / 70.5); **snapped to BROS** 74.3 / 75.5 (lower than the unsnapped spans: the study-25 `signer_name` gain from snapping (41 -> 73) came from whole-line copies, which prompt v2 removed; under v2 `signer_name` is 47 / 53 snapped against 49 / 53 unsnapped, its errors are wrong place and missing, not boundaries); **VLM spans plus BROS spans for fields the VLM leaves empty (no fitted choice): 76.2 / 77.2, +6.5 [+4.5, +8.6] / +6.7 [+4.7, +8.8]**, about +1 over the VLM alone; **per-field routers** (BROS vs VLM, with or without the snapped source) **72.9 / 72.3, +3.2 [+1.6, +4.9] / +1.8 [-0.1, +3.6]: below the VLM alone**, because the per-field winners differ across templates (the VLM wins most fields on most templates, BROS wins `file_date` and `signer_name` on Amendment) and the fitted choices send fields to BROS that the VLM would have scored better. **Correction to the reading above:** the router figure of +8.7 (79.2) quoted for the confirm run is the *primary* cell of `confirm_eval.py`, whose per-field choices were frozen in study 11 from the old VLM's behaviour on the other templates' dev documents (BROS for `file_date`, `signer_name`, `signer_title`, mostly `registration_num`; the VLM for `foreign_principle_name`, `registrant_name`); the same script's *secondary* cell, which refits on the confirm documents of the other two templates with the V2A2 spans, gives 75.2, +4.7 [+3.0, +6.4], with Amendment significantly worse than BROS (-3.8), and the two refits above give +1.8 to +3.2. The registered bar ('router gain >= +7') was defined on the primary cell and is met there only; the stable statements are that the VLM alone is +5 to +6 over BROS on both sets, and that filling its empty fields from BROS adds about +1.

**Post hoc CPU experiments on the saved outputs (written after seeing the results; `scripts/study31_cpu.py`, `results/study31_cpu/`; dev and confirm, 300 documents each; the variants were listed before scoring but chosen after seeing the study-31 results, so a gain is a hypothesis for a fresh set).** Two metrics: exact span (as registered) and an *official-style text match* (any annotated occurrence of the field; the match rules were read from the official `match_utils.py` and mirrored, the official code was not executed). V2A2 alone: exact 75.2 / 76.5, text match 77.6 / 78.7 (dev / confirm); BROS 69.8 / 70.5 exact, 69.7 / 70.8 text match, so the V2A2 lead over BROS is about +8 under text match. (1) **Decode tie-break on the saved rows** (one call per page and field gives up to four candidate spans per field; ties between pages went to the earliest page): choosing the **latest page for `file_date`, `signer_name`, `signer_title`** (generic: the signature block is at the end of the form) gives exact **81.3 / 81.7 (+6.2 [+4.7, +7.6] / +5.2 [+4.0, +6.6])** and text match **80.9 / 81.9 (+3.4 [+2.4, +4.4] / +3.2 [+2.2, +4.2])**; about half of the exact-span gain is the annotation's choice of occurrence (the right text on another page counts only under exact span). The same tie-break for all fields is not better (+0.7 / +0.5 exact, -1.5 / -1.4 text match), 'richest page' is worse (-3.1 / -2.5), taking the next-best span when the best overlaps another field adds +0.2 to +0.3. (2) **Span-level voting across systems** (V2A2, V2A1, BROS, the zero-shot reference A0, study-25 8B A2; one answer per field; mean over BROS seeds): V2A2 plus BROS fill-in of empty fields +2.0 / +1.6 exact (+1.6 / +1.5 text); plain 3-system vote +1.8 / +2.0; **abstaining unless at least 2 of the 5 systems propose the same span: 79.5 / 80.2 exact (+4.3 [+3.1, +5.5] / +3.7 [+2.5, +4.9]), 82.0 / 82.2 text match (+4.4 / +3.5)**, an increase in precision at some cost in recall; overlap-clustered votes are about the same. The two ideas were not combined. (3) **Error taxonomy of the V2A2 spans (600 documents):** many remaining errors are annotation conventions, not model errors: `foreign_principle_name` is predicted in 139 documents where it is not annotated, 119 of them Amendment forms (Amendment carries 5 annotations of it in 200 documents); `signer_name` is predicted without annotation in 85 Short-Form documents, and is missing in 67 of 200 Amendment documents; `signer_title` is predicted without annotation 109 times; `signer_name` has 72 predictions with the right text on another occurrence; `foreign_principle_name` has 54 'gold continues after it' (the gold includes the address line) and `file_date` 59 'other text' (25 Amendment, 28 Short-Form). (4) **Fresh third set:** 300 documents (100 per template, seed 11) outside dev, confirm and the pilot (630 documents excluded) are listed in `results/fresh_set3/manifest.json`; BROS spans for them do not exist (checkpoints are in the repository, but whether they are the Track-1 models, and whether inference is practical on CPU, was not checked).

**Evidence:** `results/schema_guided_kv_v2/` (`vsS25/`, `vsA0/`, `dev_vsS25/`, `dev_vsA0/`, `confirm_full_*`, `hybrid_*.json`, `gate_*.json`, `pilot_v2_fieldwise.json`, rows and spans).
