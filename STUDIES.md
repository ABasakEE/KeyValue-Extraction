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
| 32 | Prompt v2 and two post hoc rules (latest-page tie-break, abstaining vote with BROS) on a fresh third document set, Qwen3-VL-8B 4-bit | GPU | registered 2026-11-20 | *pending (Kaggle run)* |
| 33 | Schema-guided grounded recipe (prompt v2, Qwen3-VL-8B 4-bit) on FATURA unseen templates against BROS trained on K = 1, 2, 5, 10, 40 seen templates | GPU | 2026-11-20 | **Recipe helps (rule met):** A2 91.7 vs zero-shot 3B VLM 59.6 (+32.1 [27.1, 36.9]). **Crossover:** LLM wins at K = 1 (+59.4) and 2 (+22.5), tie at K = 5 (+0.9 [-1.8, 3.7]), BROS wins at K = 10 (-4.2 [-6.5, -1.9]) and 40 (-8.2). Image adds about 3.8 F1 (A2 91.7 vs A1 87.9); neither combination beats the better system alone |
| 34 | Table-heavy extension: line items (rows of cells) on VRDU ad-buy unseen templates, schema-guided grounded recipe vs BROS plus row grouping; DocILE LIR metric ported | CPU steps done, GPU pending | registered 2026-11-20 | **Ad-buy result (2026-11-22):** BROS row F1 68.9 / 83.2 / 83.8 / 85.7 at N = 10 / 50 / 100 / 200; recipe A2 38.2 (128 documents, 1.71x too many rows); **BROS wins at every N, no crossover for N >= 10**; DocILE runs not done |
| 35 | Efficient linking on tables: grouping cells into rows (and typing them from headers) with a learned pair classifier on candidate pairs, against the anchor heuristic; DocILE and ad-buy | CPU first | registered 2026-11-21 | **Not adopted (registered rule):** learned pair classifier beats the anchor rule on ad-buy (+2.38 [+0.42, +4.71]) but loses on DocILE (-4.32 [-9.28, -0.90]); the pair scores are good (AUC 0.97 to 0.998), the clustering is the weak part; step 1b (average link, correlation, anchors): ad-buy +2.3 to +2.6 for every variant, DocILE best -1.9 [-4.0, +0.03]; the rule stays |
| 36 | A learned router between the recipe and BROS (which system to trust, per document or per field): headroom first | CPU | groundwork 2026-11-22 | **Headroom exists:** on the FATURA confirm invoices a perfect per-field router gains +4.2 / +5.4 / +5.8 / +4.1 / +0.1 over the better single system at K = 1 / 2 / 5 / 10 / 40; no router has been trained |
| 37 | The joint tagger-plus-biaffine-linker with a BROS backbone against LiLT on XFUND (end to end, 250 val forms, 3 seeds) | GPU | registered 2026-11-22 | *pending (Kaggle run); CPU checks done* |
| 38 | The schema-guided grounded recipe on DocILE line items (layout-disjoint test documents), pilot-gated | GPU | registered 2026-11-22 | *pending (Kaggle run)* |
| 39 | The BROS line-item tagger with row grouping on DocILE layout-disjoint splits (the baseline for study 38) | GPU | registered 2026-11-22 | *pending (Kaggle run); CPU checks done* |

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


**Pre-run amendment 2 (2026-11-22, before any GPU run; study 27 has still not been run).** Made after the study 33 to 35 work and a focused literature check; the model, the V1 to V3 prompts, the parsers and the scorer are unchanged.
- **Order and reading.** V3 runs first, then V2, V4, V1. V3 (the VLM links the given entities) is the cleanest answer to 'can the VLM link at all' and is read against LiLT with gold entities (about 59.8); V2 and V4 are the end-to-end arms. Reason: published zero-shot results on FUNSD and XFUND are low and a large part of the gap is exact localisation (the [Identify, Locate, Link](https://arxiv.org/pdf/2608.20868) paper: zero-shot Qwen2.5-VL-7B scores 0.123 relation F1 on XFUND under a layout-aware metric against 0.206 for a tuned 256M model, and relaxing the localisation gate raises its scores; our own study 9 found VLM spans weak). The registered primary cell (V2 end-to-end link F1, interval over forms, next to LiLT) is kept, and no 'helps' verdict is drawn.
- **New arm V4 (hybrid): predicted entities, VLM links.** The numbered entities of the V3 prompt are the entities a trained LiLT tagger predicts (`src/vlm_link.py: prompt_v4`, `parse_v4`; the prompt says the tagger can be wrong and to use the page image too). The tagger is LiLT A0 (`configs/xfund_table_A0.yaml`, the study 26 baseline), seed 13, trained in the notebook (about 35 minutes on one T4, from study 19's 1.7 h per arm for 3 seeds) and run on the 250 val forms with entity-safe 510-token windows, stride 128 (`scripts/xfund_dump_entities.py`; the predicted entities are stored in every raw row, so scoring needs no extra file). Scoring is end to end with `src/e2e_link.py` unchanged: entity F1 is the tagger's, a link counts only if both ends are exact predicted entities and the direction is right. **Reference for V4: LiLT A0 seed 13 on the same 250 forms, entity 75.79, link 48.33, link_qa 53.41** (`results/table_links/A0_e2e.json`; the mean of 3 seeds, 47.42, is the reference for V2). Why: the tagger supplies exact spans and the VLM only the relations, which is the efficient split (an answer is about 400 tokens of id pairs); it is the only arm in which exact boundaries are not asked of the VLM, so it shows what a VLM linker adds to a tagger. Limit: one tagger seed, so V4 and its reference share the entities and differ only in who links.
- **Attention fix.** `xfund_vlm_generate.py` applies the guarded `patch_sdpa()` of study 34 (repeat keys and values instead of the GQA flag) before loading the model. On the notebook's pinned transformers 4.51.3 the function probably does not exist and the patch is a no-op (the failure was seen on 4.57 in the ad-buy runs); it protects a run if the version is changed.
- **Branch.** The notebook now clones `feat/bros-relative-layout`, which carries the V4 arm and the fix (it cloned `feat/xfund-vlm`).
- **Time.** Caps per arm V3 2.5 h, V2 6 h, V4 2.5 h plus the tagger training, V1 2.5 h inside the 11 h budget; the order puts V3, V2 and V4 first, so a cut run loses V1 first.
- **Checks done on CPU (no model).** `tests/test_vlm_link.py` passes including a V4 test (a perfect answer for the gold entities scores perfectly, a tagger that missed an entity loses only that entity's links, malformed answers never raise); the stub pipeline `xfund_dump_entities.py --stub` then `xfund_vlm_generate.py --arm V4 --stub` then `xfund_vlm_score.py` runs and scores 1.0 (the stub entities are the gold ones, so that 1.0 only proves the plumbing); every notebook cell compiles. **Not checked:** training LiLT inside this notebook, the V4 prompt with a real model, and the dump script on a real checkpoint (no checkpoint is available here).

**Results (2026-11-22; the session finished at 10.6 h of its 11 h budget; scored with `scripts/xfund_vlm_score.py` from the pushed raw answers; `results/xfund_vlm/{V3,V2,V4}_e2e.json`; Qwen2.5-VL-7B-Instruct, chosen because the projected V2 time was 3.87 h, below the 6 h switch; zero parse failures).** **V3 ran on 232 of 250 forms (its 2.5 h cap), V2 on all 250, V4 on its first 50 forms only (the session budget ended), V1 not at all.** *Correction:* an earlier version of this entry, written while the session was still running, said V2 had stopped at 200 forms for an unrecorded reason; it had not stopped, `run_status.json` is written when an arm ends.
- **V3 (VLM links the given gold entities), 232 of 250 forms:** entity F1 100 (gold given), **link F1 23.69, 95% interval over forms [21.86, 25.66]**, link_qa F1 41.34. Link precision 19.1% and recall 31.2%: it wrote 13,448 links for 8,243 gold ones. Group recall: below 20.0, same line 47.6, above 36.4. **Read against the registered reference (LiLT with gold entities, windowed, about 59.8; a different set of forms: 250 against 232):** the zero-shot VLM is about 36 points lower even when it is handed the entities.
- **V2 (end to end, the VLM finds entities and links), 250 forms (5.96 h):** entity F1 13.57, **link F1 2.97, interval [2.24, 3.81]**, link_qa F1 3.40, against LiLT A0 47.42 (mean of 3 seeds). The registered primary cell has no 'helps' verdict by design; the interval is far below A0's and the order of the arms is not in doubt.
- **V4 (the LiLT A0 seed 13 tagger's entities, the VLM links them), first 50 forms only:** entity F1 77.87 (the tagger's, not the VLM's), **link F1 23.93, interval [20.35, 27.46]**, link_qa F1 33.83; link precision 19.5%, recall 30.9% (2,286 links for 1,442 gold). The reference is the same tagger's own biaffine linker on all 250 forms, link F1 48.33 (seed 13); on these 50 forms the tagger's entity F1 is about the same as over all 250 (77.9 against 75.8), so the subset is not obviously harder. **A VLM linking a trained tagger's entities scores about half of the trained linker**, and over-links exactly as in V3 (precision about 19%). Not computed: the tagger's link F1 on exactly these 50 forms, and V1.
- **Not checked:** whether part of V2's very low entity F1 comes from the word-range output format and numbering rather than from the VLM's reading; the CPU checks used perfect stub answers only.
- **What this shows:** with this prompt, a zero-shot 7B VLM does not link XFUND entities competitively, whether it finds the entities itself (V2), is given gold entities (V3) or a tagger's (V4), and in V3 and V4 it over-links (precision about 19%). The trained biaffine linker is about twice as good as the best VLM arm. **What it does not show:** anything about larger or tuned VLMs, about prompts other than ours, or about V1 (not run); V4 rests on 50 forms.

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



---

## 32. Prompt v2 and two post hoc rules on a fresh third document set — GPU (Kaggle T4 × 2), registered 2026-11-20

**Question.** Study 31 (prompt v2, Qwen3-VL-8B 4-bit) gave 76.5 and 75.2 exact-span F1 on the confirm and dev documents against 60.5 / 58.1 for study 25's prompt, and BROS 70.5 / 69.8. Post hoc CPU experiments on the saved study-31 outputs then found two rules that helped on both sets: (i) when the one-call-per-page-and-field decode finds a field on several pages with equal votes, take the **latest page** for `file_date`, `signer_name`, `signer_title` (study 31: +6.2 / +5.2 exact span, +3.4 / +3.2 official-style text match); (ii) **abstaining voting** across systems (+4.3 / +3.7 exact, +4.4 / +3.5 text match for five systems; the three-system version tested here was +2.4 / +1.9 exact with abstention). The rules, the prompt and the descriptions were all chosen after seeing the dev and confirm results. Do they hold, with the recipe frozen, on 300 documents nobody has looked at?

**Data.** 300 registration forms (100 per template, seed 11) outside the dev set, the confirm set and the study-25 pilot (630 documents excluded): `results/fresh_set3/manifest.json`. Their BROS predictions already exist: the saved Track-1 dumps (`results/vrdu_track1_bros_errdump/UTL_fold*_seed{13,42}/test_entities.jsonl`) cover the whole held-out template of each fold, which contains all 300 documents; `results/fresh_set3/bros_spans.jsonl` holds them with the gold (BROS exact-span F1 on these documents: 68.6 seed 13, 71.4 seed 42, in line with dev and confirm). The BROS models were trained before these documents were chosen and saved no weights, so nothing was selected on them. The documents are sampled from the held-out template's test pool like dev and confirm.

**Recipe (frozen; no pilot, no format gate, no tuning).** Qwen3-VL-8B-Instruct in 4-bit nf4 (fp16 compute), prompt v2 and descriptions v2 exactly as in study 31 (`results/schema_guided_kv_v2/schema_desc_v2.json`), one call per page and field, one sample per page (greedy), up to 4 pages, fallback decode. Arms: **V2A2** (image + OCR lines with ids + schema) and **V2A1** (the same without the image), both on all 300 documents. Notebook: `notebooks/kaggle_fresh_set.ipynb`; deadline 5.0 h.

**Rules applied after decoding (frozen before the run; `scripts/study32_fresh.py score`).** *Tie-break:* for the fields `file_date`, `signer_name`, `signer_title` the latest page wins ties between candidate spans with equal votes; all other fields keep the earliest page (the registered decode of study 31). *Vote:* per field, candidates from V2A2, V2A1 (both with the tie-break decode) and BROS (seed 13 or seed 42, scored separately and averaged); the span proposed by most systems wins, ties go to V2A2 then V2A1 then BROS; **abstain** (no prediction for the field) unless at least two systems propose exactly the same span.

**Metrics.** Official-style text match (any annotated occurrence of the field counts; match rules of the VRDU `match_utils.py`, read and mirrored, not executed) and exact span (as in studies 25-31). Mean over BROS seeds 13 and 42, paired bootstrap over documents (1,000 draws), the same 300 documents for every system.

**Registered rules (text-match F1 is primary; exact span reported next to it):**
- *Prompt replicates:* V2A2 (earliest-page decode) minus BROS >= +3.0, interval above 0.
- *Tie-break helps:* V2A2 with the latest-page rule minus V2A2 earliest-page >= +2.0, interval above 0.
- *Vote helps:* the three-system abstaining vote minus V2A2 with the tie-break >= +1.0, interval above 0.
- Descriptive: V2A1 against V2A2 (value of the image), plain three-system vote without abstention, V2A2 with BROS filling empty fields, vote without the tie-break, per template and per field.

**Checks before any GPU spend.** The scorer ran end to end on stub rows (all-empty answers: BROS 69.6 text-match, VLM arms 0, every rule false, as expected); the BROS spans for all 300 documents were built from the dumps (a missing document would have stopped the script); the notebook cells compile; the model path is the study-31 path (Qwen3-VL-8B 4-bit, prompt v2) that ran to completion on dev and confirm.

**Limits stated up front.** One model, one prompt, one sample per page; the prompt and rules come from the dev and confirm sets, which makes this a replication of two post hoc rules and one prompt rather than a first test; the fresh documents are drawn from the same corpus and templates as dev and confirm (an unseen-template test inside registration forms, not a new corpus); the vote's abstention helps precision at the cost of recall, so the F1 gain depends on the operating point; the five-system vote of study 31 (+4.3 / +3.7) needs two more VLM arms and is not tested here; BROS is the Track-1 recipe with two seeds; the official scorer was not executed.

**Results:** *pending*

**Evidence:** `results/fresh_set3/` (after the run).


---

## 33. The schema-guided grounded recipe on FATURA unseen templates, against BROS trained on K seen templates — GPU (Kaggle T4 × 2), registered 2026-11-20

**Question.** Study 15 measured BROS's unseen-template F1 on FATURA as a function of the number K of seen templates (32.5 / 69.2 / 90.8 / 95.9 / 99.9 for K = 1 / 2 / 5 / 10 / 40, 500 confirm invoices) and compared it with a zero-shot 3B VLM in a key-value formulation (59.6, flat in K). That formulation could not express unlabeled blocks (`title` 0.0, `seller_name` 14.2) and the model was small. The VRDU studies (25, 31) showed that the schema-guided grounded recipe (OCR lines with ids, schema with descriptions, answers that cite a line) with Qwen3-VL-8B reaches 75-77 exact-span F1 on unseen templates, above BROS trained on two templates (about 70). **At which number of seen templates does BROS stop being worse than this training-free recipe on FATURA?** This is the measurement the VRDU data cannot give (VRDU has three templates, so only K = 2 exists).

**Data.** The 500 confirm invoices of study 14 (10 per template, 50 templates, `results/fatura_sg/confirm.json`, from `results/fatura_v1/manifest.json`). Gold = the annotated blocks of the nine core fields (block text includes its label). The OCR lines of a page are its annotated text blocks (`fatura_loader.parse_doc`); the same block tokens are what BROS read, so both systems start from the same 'perfect OCR' (a limit: no OCR noise). A cited block id becomes the whole block (gold is block level). 30 dev invoices of study 14 (`results/fatura_sg/pilot.json`, seed 5) are used only for the answer-format gate.

**Recipe (frozen; nothing tuned on FATURA).** Qwen3-VL-8B-Instruct, 4-bit nf4 (fp16 compute), prompt v2 of study 31 with the nine field descriptions of study 14 (`results/fatura_v1/schema_desc.json`, written by a model from the field names alone) and a worked example in FATURA's field names (invented values and ids); one sample per page, greedy; fallback by text when an id is missing or unknown; one block per field, a block cited by two fields goes to the first field in alphabetical order. **Format gate on the pilot:** JSON per page if at least 90% of outputs parse and at least 90% of values cite an existing block id (strict), else one call per field. **Arms (order):** **A2** (page image + blocks + schema; primary) then **A1** (no image). Notebook: `notebooks/kaggle_fatura_sg.ipynb`, deadline 5.0 h.

**BROS reference.** The saved dumps of studies 14 and 15 (`results/fatura_v1/bros` for K = 40; `results/fatura_v2_sweep/K{1,2,5,10}`), two seeds each, same 500 invoices; the scorer reproduces their F1 (32.5, 69.2, 90.8, 95.9, 99.9) on stub rows.

**Metric.** Exact block-span micro F1 (a predicted span counts only if it equals a gold block). Intervals: bootstrap over the 50 templates (clusters of 10 invoices), 1,000 draws, as in studies 14-16; differences to BROS are against the mean of the two seeds.

**Registered rules (primary arm A2; A1 reported alongside):**
- *Recipe helps on FATURA:* A2 minus the study-14 zero-shot VLM (59.6) >= +10.0, template-bootstrap interval above 0.
- *Crossover:* for each K, **LLM wins** if A2 minus BROS(K) has its interval above 0, **tie** if the interval contains 0, **BROS wins** if the interval is below 0. Report the largest K where the LLM wins and the smallest K where BROS wins.
- Descriptive: per-field and per-template F1; two combinations per K (BROS with the LLM filling fields BROS left empty; the LLM with BROS filling its empty fields); A2 against A1 (value of the image).

**Checks before any GPU spend.** The oracle (gold block cited by id, through the decode path, both answer formats) scores 100.00 on 100 invoices; the scorer ran on stub rows and reproduced BROS's study-15 numbers; the run loop was exercised with the stub model; the notebook cells compile. Not tested here: the 8B model on FATURA pages (the notebook's smoke cell runs 3 invoices per arm before anything else and checks the visual-token budget).

**Limits stated up front.** FATURA is synthetic (generated values, clean rendering) and the OCR is the annotation, so both systems are favoured; the model may have seen the public dataset (not checkable); BROS has two seeds per K and the K seen templates vary by seed, so the BROS curve has seed noise that the template bootstrap does not cover; one prompt, one model, one sample; block-level exact scoring is strict for a model that must pick whole blocks including labels; the field descriptions come from a model, not from FATURA's wording; no ablation separates prompt, ids and model from the study-14 baseline (the comparison with 59.6 bundles them).

**Results (run 2026-11-20, 2.64 h of the 5 h deadline, no failed shards; format gate: JSON kept, parse 1.0, id-valid 0.968, 220 values on 30 pilot pages):**

| K | BROS | A2 (image) | A2 - BROS [95% CI] | A1 (no image) | A1 - BROS [95% CI] | verdict (A2) |
|---|---|---|---|---|---|---|
| 1 | 32.5 | 91.7 | +59.4 [54.7, 64.6] | 87.9 | +55.6 [49.8, 61.2] | LLM wins |
| 2 | 69.2 | 91.7 | +22.5 [18.0, 26.9] | 87.9 | +18.8 [13.7, 23.8] | LLM wins |
| 5 | 90.8 | 91.7 | +0.9 [-1.8, 3.7] | 87.9 | -2.9 [-5.9, 0.3] | tie |
| 10 | 95.9 | 91.7 | -4.2 [-6.5, -1.9] | 87.9 | -8.0 [-10.7, -5.2] | BROS wins |
| 40 | 99.9 | 91.7 | -8.2 [-10.2, -6.3] | 87.9 | -11.9 [-14.7, -9.4] | BROS wins |

- **Recipe helps:** A2 minus the study-14 VLM (59.6) = +32.1 [27.1, 36.9] (A1: +28.4 [24.0, 32.9]); rule met.
- **Crossover (registered form):** largest K where the LLM wins = 2; smallest K where BROS wins = 10; K = 5 is a tie. Same classification for A1.
- **Combinations (descriptive):** BROS with the LLM filling empty fields: 77.8 / 76.3 / 91.2 / 94.8 / 96.6 (minus BROS: +45.6 at K = 1, +7.2 at K = 2, +0.4 tie at K = 5, -1.0 tie at K = 10, -3.2 at K = 40); at K = 1 and 2 it is below the LLM alone. The LLM with BROS filling its empty fields is within about 1 point of the LLM alone at every K (92.0 at K = 2). Neither beats the better single system.
- **Per field (F1, A2 / A1):** buyer 70.2 / 78.4, number 95.5 / 72.9, title 98.8 / 72.7, seller_name 89.5 / 89.1, seller_address 85.3 / 87.9, date 98.3 / 99.8, due_date 96.1 / 96.5, sub_total 96.6 / 100.0, total 93.7 / 93.5. The image helps `title` and `number` (about +26 each) and is neutral or slightly negative on buyer.
- **Why `buyer` is weak (post hoc, from the rows):** (1) only 31 of 50 templates annotate a buyer (310 of 500 invoices); the other 19 still print a `BILL_TO` / `SHIP_TO` block, which the model fills on 144 of those 190 invoices (A2) and 118 (A1), so buyer precision is 0.59 (A2) / 0.68 (A1) against recall 0.865 / 0.926; on the annotated templates alone buyer F1 is 86.5 (A2) / 93.5 (A1). The gold has no buyer there, so the penalty comes from the annotation, not from reading error. (2) In A2, 37 of the 42 wrong-block buyer answers quote the right buyer text but cite another line's id (invoice number, date, address); decode trusts a valid id, so they score as wrong (A1: 11). Not re-scored with a text-first decode (would be post hoc).
- **Text-first decode (post hoc, not registered; `scripts/fatura_sg_textfirst.py`, `results/fatura_sg/run/report_textfirst.json`):** when the cited id's block does not hold the quoted text but exactly one block does, use that block; nothing else changes, no new model calls. A2 94.0 (+2.3 on the registered 91.7), A1 88.7. Minus BROS (A2): K = 1 +61.6 [57.0, 66.6], K = 2 +24.8 [20.9, 28.7], **K = 5 +3.2 [1.1, 5.5] (registered: tie)**, K = 10 -1.9 [-3.8, -0.1], K = 40 -5.9 [-7.4, -4.4]. Largest K where the LLM wins becomes 5 (registered: 2); smallest K where BROS wins stays 10. A1 verdicts unchanged (K = 5 -2.1 [-5.2, 1.1], tie). A2 minus the zero-shot VLM: +34.4 [29.9, 38.8]. A2 per field: buyer 79.8, date 100.0, due_date 96.8, number 97.8, seller_address 88.0, seller_name 91.4, sub_total 96.6, title 99.6, total 94.2. Of the A2 answers that changed, 105 went wrong to right and 4 right to wrong; the four losses are one seller_address and three seller_name answers (A1: 37 gained, 1 lost). The rule was written after reading the rows, so the registered result (91.7, K = 5 tie) stays the headline.
- **Per template (A2):** min 75.6, nine of 50 templates at 100.0.
- Not computed: a template-bootstrap interval for A2 minus A1; the image effect (+3.8) is a point estimate.

**Evidence:** `results/fatura_sg/run/` (`report.json`, `rows_A{1,2}_{0,1}.jsonl`, `gate_json.json`, `run_status.json`); notebook `notebooks/kaggle_fatura_sg.ipynb`.


---

## 34. Table-heavy extension: line items on VRDU ad-buy unseen templates — CPU steps done, GPU not run, registered 2026-11-20

**Question.** Studies 25-33 only extract one block per field. Does the schema-guided grounded recipe still beat a trained tagger when the target is a table (repeated rows of cells), and where does the loss come from: reading the cells or grouping them into rows?

**Data (checked 2026-11-20, `data/vrdu/ad-buy-form/main/dataset.jsonl.gz`).** 641 documents; 636 have at least one line-item group, 9,163 gold rows in total (mean 14.3, median 12, maximum 110 per document); a group is `[[field names], [rows]]` and each row is a list of cells (`[text, [page, x0, y0, x1, y1], [[char_start, char_end]]]`), one per field name (no row has a different cell count from its header: 0 of 9,163). Five line-item fields: `channel`, `program_desc`, `program_start_date`, `program_end_date`, `sub_amount`; 4 or 5 cells per row in 7,126 of the 9,163 rows. OCR words per document: median 846, 90th percentile 2,611, maximum 11,519. So row grouping is annotated here (an assumption I made earlier, that it might not be, was wrong). Splits: `few_shot-splits/DeepForm-unk_template-train_{10,50,100,200}-test_*-valid_100-SD_{0,1,2}.json` (unseen templates; test sizes 293 to 153; 3 seeds each). Scored on the unseen-template test documents only.

**DocILE (looked up, not used yet; sources: the DocILE README and paper, via a fetch agent whose read of the paper was truncated, so figures marked "paper" are partly a summary).** No key-value linking task or model. KILE = fixed field classes, metric COCO-style AP; LIR = grouping the same classes into line items, metric micro F1 over line-item fields, where a cell is right only if its type and location match and it sits in the correct line item (maximum matching of predicted to gold rows, maximising recall). Baselines: RoBERTa-base and LayoutLMv3 NER models (plus a DETR line-item detector for LIR); a joint model tags tokens as inside a line item and groups them. Best permitted test scores in its README: KILE AP 0.539, LIR F1 0.698 (RoBERTa-base with synthetic pretraining). Checkpoints and metric code (`docile/evaluation/evaluate.py`, MIT) are released. Data needs a token from docile.rossum.ai; the dataset licence terms are not stated on the pages read. 6,680 annotated documents (train 5,180, validation 500, test 1,000); validation and test are 25% zero-shot layouts, 25% few-shot, 50% many-shot. DocILE is the second dataset, after the ad-buy probe.

**Metric ported (done, `scripts/lir_metric.py`, self-checked).** `lir_f1`: predicted rows are matched one-to-one to gold rows for the maximum number of exactly matching cells (Hungarian algorithm, checked against brute force on 200 random cases); a cell counts only in its matched row; micro F1 over cells. `cell_f1` ignores rows. **cell_f1 minus lir_f1 isolates the cost of grouping.** Cell identity here is exact (field, start, end) like studies 25-33, a stricter test than DocILE's pseudo-character-centre match.

**Plan and registered rules (draft; to be frozen before the GPU run).**
1. *Oracle (CPU, next):* gold cells cited by id through the decode path score 100.00 in `lir_f1` and `cell_f1`; token-budget table per document (page image plus lines) for Qwen3-VL-8B 4-bit on a T4; decide chunking (table region, or row bands) for the 10% of documents above 2,600 words.
2. *Recipe extension:* the model returns a list of row objects, each cell citing a line id; rows come from the JSON structure; a field may repeat across rows; the text-first id repair of study 33 is part of the frozen decode.
3. *Trained baseline:* BROS tagging of the five fields (existing pipeline) plus row grouping by vertical clustering of the predicted cells; the link head of study 26 is a later alternative.
4. *Pilot (30 dev documents, format gate):* at least 90% of outputs parse, at least 90% of cited ids exist, and at least one row on at least 90% of documents that have gold rows; otherwise switch to one call per table region.
5. *Primary comparison:* recipe (A2 and A1) against BROS plus clustering on `lir_f1`, at 10 / 50 / 100 / 200 training documents of the unseen-template splits (BROS trained; recipe untrained), 3 seeds, template-agnostic document bootstrap (templates are not labelled in the data; interval over documents, stated as a limit). LLM wins if the interval of the difference is above 0, ties if it contains 0, BROS wins if below. Also report `cell_f1`, per-field F1, row-count error, and the unseen-template gap.
6. *DocILE:* only after step 5 shows the recipe is viable, and after the licence and data access are checked.

**Limits stated up front.** Ad-buy templates are not labelled in the data, so the bootstrap is over documents; the document OCR is the dataset's own OCR (noisy), unlike the FATURA annotation OCR of study 33; a single 8B 4-bit model; exact cell matching is stricter than DocILE's.

**Results so far (step 1, oracle, 2026-11-20; no model involved).** The oracle cites each gold cell by line id and quoted text and decodes it the way the recipe would (`scripts/adbuy_rows.py oracle`, results in `results/adbuy_rows/oracle_lines.json` and `oracle_gap0.5.json`, all 641 documents, 9,163 rows, 38,204 cells). **It does not reach 100, so the ceiling of any line-citing method on this data is below 100.**
- **Cell identity changed from a word range to a word set** (a cell can span two OCR lines, with other columns' words between them; the range scored 91.3 on the first 40 documents for this reason). A set also matches DocILE's pseudo-character-centre match in spirit. `lir_metric.py` already accepts any hashable cell.
- **Whole OCR lines as citation units: oracle `lir_f1` = `cell_f1` = 92.97** (2,686 of 38,204 cells wrong; 258 of 641 documents have a miss). Cause: one OCR line often holds several cells with the same text (for example a start and an end date that are equal), and a line id plus text cannot say which one.
- **Lines cut where neighbouring words are more than 0.5 line heights apart: 97.80** (842 wrong; 75 documents). Gaps of 0.5, 0.35 and 0.25 scored the same on the first 40 documents (98.95); 0.1 scored 99.47 there with about 30% more prompt tokens. 0.5 was chosen on those 40 oracle runs only, as the cheapest setting that reaches the plateau.
- **The remaining error is one field:** on documents 41-160, `program_end_date` is wrong in 259 of 1,564 cells (16.6%); `program_start_date` 1 of 1,843; `sub_amount`, `channel` and `program_desc` 0. Example (one document of that sample): the OCR text reads `04/15/20-04/21/20`, but the gold start date is only `04/15/` and the gold end date only `/21/20`, so the gold cells are partial strings inside glued tokens and no word-level citation reproduces them.
- **Value-level metric added next to the strict one** (`value_lir_f1`, `value_cell_f1` in the oracle output; a cell is right if the field is the same and the text matches under the field's rule in the ad-buy `meta.json`: GeneralStringMatch alphanumerics only, NumericalStringMatch digits only, DateMatch parsed date, PriceMatch numeric value; study 25's reading of the VRDU rules, not run with the official tool, and PriceMatch is my addition). **Oracle with split lines: value `lir_f1` = `cell_f1` = 97.79, against 97.80 strict, so the value-level metric does not remove the ceiling.** I had expected a date-aware match to accept those end dates; it does not, because the gold text is itself partial. The secondary metric is still reported (it will matter for model outputs whose boundaries differ slightly from gold), but the ceiling of about 97.8 holds under both.
- **Prompt size (estimated at 3 characters per token, no tokenizer available locally):** per page, whole lines median 1,214, 90th percentile 1,898, 99th 3,803, maximum 8,745 tokens; split lines median 1,399, 90th 2,431, 99th 4,417, maximum 9,163; 6 of 1,598 pages exceed 6,000. Image tokens (up to 1,280 per page at the study-33 pixel limit) come on top. Per-page calls are therefore feasible for nearly all pages; no table chunking is needed except for those 6 pages. Rows are single-page in this data (documents have up to 12 pages), so the model is called per page and the rows are concatenated.

**Step 2 done (CPU, 2026-11-21): row-list format, decode, repair, run/gate/score code (`scripts/adbuy_sg.py`).**
- **Format:** one call per page; the model answers `{"rows": [{field: "text identifier" | null, ...}, ...]}` for the five line-item fields; cells cite the pieces of lines cut at 0.5 line heights. Field descriptions are written from the field names alone and frozen (`DESC` in the script); one worked example with invented values. The parser takes every complete flat object after `"rows"`, so a truncated answer for a long table keeps the rows it finished. `max_new_tokens` 3000 (gold rows per page with rows: median 7, 90th percentile 14, maximum 47).
- **Decode:** a cell becomes the set of words of its quoted text on the cited piece; cells that cannot be grounded and rows with no grounded cell are dropped; rows are concatenated over the pages of a document. **Repair** (replaces the study-33 unique-piece rule after the repair check below): a cell whose cited piece does not hold its text moves to the piece that holds it and sits on the row's line (rule `row`).
- **Oracle through the decode path (first 30 documents):** `lir_f1` = `cell_f1` = 98.86 (consistent with the all-document ceiling of 97.8). With every cited id shifted to the next piece: cell F1 3.85 without repair, **30.99 with repair**. So the repair helps far less on tables than on FATURA: cell texts repeat across rows (the same channel, the same dates), and the repair only fires when exactly one piece on the page holds the text.
- **Can a wrong cited id be repaired? (CPU check, 2026-11-21, `scripts/adbuy_repair.py`, no model run; `results/adbuy_sg/repair_simulation.json`).** Two parts.
  - *Real slips on FATURA (study 33 rows):* 58 A2 cells (16 for A1) cite a block that does not hold their text while another block does; the text is found in exactly one other block in 57 of the 58, and in 48 of 58 (all 16 for A1) the cited block is the gold block of another field. So the real model's slip is "right text, id of a neighbouring field's block". FATURA has one block per field, so the text is unique there; on tables the same text recurs in many rows, which is why the unique-only rule recovered little (31 against 84 with ids shifted on every cell).
  - *Simulation on the ad-buy oracle (20 documents, wrong ids put in by hand):* four slip types (next piece, random piece, same field in the adjacent row, another cell of the same row) at 0, 10 and 30% of the cells, five rules: none; unique (study 33); nearest (the piece closest to the cited id's coordinates); **row** (the piece closest to the vertical position of the row's other, correctly cited cells, ties by the cited coordinates; falls back to nearest, then unique); row+ (row, and also moves correctly grounded cells that sit more than 1.5 piece heights off their row's line). Cell F1: with no slips every rule except row+ is 98.3 (row+ 94.5: it moves correct multi-line cells, so it is rejected). At 10% wrong ids, none / unique / nearest / **row**: next 89.3 / 90.9 / 95.5 / **96.0**; random 87.7 / 90.1 / 87.0 / **94.4**; same row, other cell 91.1 / 92.6 / 93.0 / **96.3**; adjacent row, same field 87.0 / 88.7 / 89.3 / 89.3 (row+ 91.8). At 30%: row is 92.7 / 92.0 / 82.3 / 93.1 against none 76.4 / 77.6 / 75.6 / 80.1. Nearest alone is worse than unique for random slips (87.0 against 90.1 at 10%). **The `row` rule is the frozen decode repair.**
  - *What it does not fix:* a cell that cites the same field's piece in a neighbouring row has the right text on the cited piece, so nothing looks wrong except its vertical position; that is the weakest case (89.3 at 10%) and row+ helps there only at the price above.
  - *What the check cannot say:* the simulated slips are put in by hand, and the real model's slips on tables are unknown until a pilot; the FATURA slip rate was about 2% of cells (buyer 12%), where the gain of the row rule over no repair would be roughly a point or two of cell F1 (about +5 to +7 at 10%).
- **Evaluation sets (`results/adbuy_sg/primary.json`, `pilot.json`, from `adbuy_sg.py manifests`):** the test documents of the twelve unseen-template splits (train sizes 10 / 50 / 100 / 200, three seeds) are all drawn from a common pool of 300 documents; **143 documents (367 pages) are in the test set of every split**. The primary set is those 143, so BROS at every training size and seed is scored on the same documents as the recipe. The pilot set is 30 documents (74 pages, seed 5) outside every test set.
- **Gate (pilot, as in study 33, changed from the draft):** at least 90% of outputs parse, at least 90% of cited ids exist, and at least one row on at least 90% of pages that have gold rows; if not, the script reports `stop_and_review` instead of switching format automatically (the draft's "one call per table region" fallback is not implemented).
- **Plumbing checked with a stub model:** run loop, resume, gate (correctly says `stop_and_review` for empty answers) and scorer (0.0 on empty answers). **Not checked locally:** the image path (`pymupdf` is not installed here, so the PDF page rendering and the 8B model are untested until the notebook's smoke cell); no model has been run on ad-buy pages.

**DocILE probe (CPU, 2026-11-21; `scripts/docile_rows.py`, `scripts/download_docile.sh`, `results/docile_rows/`).** Access granted (token kept in the gitignored `.secrets/docile_token`); `annotated-trainval` downloaded (1.14 GB zip, 1.7 GB unzipped, `data/docile/`, gitignored). The dataset terms are not yet read by us; the pages we fetched state none.
- **Content:** 5,680 labelled documents (train 5,180, validation 500), all English; 3,850 tax invoices, 1,440 orders, the rest purchase orders, receipts and sales orders; 1 to 3 pages (7,394 pages). 36 key-information field types and 19 line-item field types. **5,325 documents have line items; 38,657 rows with at least one OCR word, median 4 rows per document (maximum 110), 2 to 8 cells per row** (4 or fewer in most rows). 45 rows have cells on two pages. The OCR (DocTR) comes with blocks, lines and words, so OCR lines exist (the earlier summary said word level only).
- **Layouts:** each document carries a `cluster_id`: 1,063 layouts in total, 1,003 in train, 204 in validation. Validation = 124 documents of layouts unseen in train, 126 of layouts with at most 3 train documents, 250 of seen layouts. Train layouts are small: median 1 document per layout, 558 of 1,003 layouts are singletons, the largest has 79. So layout-disjoint splits and a training-layouts curve (K layouts) can be built from all 5,680 documents; the 124 official zero-shot documents alone are too few.
- **Oracle (same method as ad-buy: cells as word sets, cited by piece id and text, grounded by the first matching word run; a word belongs to a gold cell if a pseudo-character centre of it lies in the gold box):** whole OCR lines as pieces `lir_f1` 98.89, `cell_f1` 98.66 (592 documents with a miss); **pieces cut at 0.5 line heights `lir_f1` 99.94, `cell_f1` 99.71 (38 documents)**. This oracle only scores cells that contain a word. **5,563 of 156,765 gold line-item cells (3.5%) contain no OCR word**, so under DocILE's own metric (every gold cell counts) no OCR-citing method can exceed about 96.5% recall on cells; this is a data limit, not a model one.
- **Prompt size (estimated, 3 characters per token):** per page, cut pieces median 740, 90th percentile 1,647, 99th 6,584, maximum 8,995; 82 of 7,394 pages above 6,000. Gold rows per page with rows: median 3, 90th percentile 15, maximum 61 (shorter outputs than ad-buy).
- **What this says:** the cell-as-word-set and piece-citation design transfers to DocILE (oracle 99.9 on the reachable cells), pages are shorter than in ad-buy, and the layout structure allows the K-layouts curve. Not yet done on DocILE: any model run, the 19 line-item field descriptions, and the layout-disjoint split definition.

**DocILE layout splits (frozen 2026-11-21, before any model saw DocILE; `scripts/docile_splits.py`, `results/docile_rows/splits.json`).**
- **Rules:** unit = documents with at least one line item (5,325 documents, 934 layouts by `cluster_id`); all 5,680 labelled documents (DocILE train and validation) form one pool, because the 124 official unseen-layout validation documents are too few. Reserved first, one random document per layout, layouts drawn uniformly at random (seed 34): **test 200 layouts (200 documents)**, validation 100 layouts (BROS early stopping, so stopping is judged on unseen layouts too), pilot 30 layouts (prompt format gate). Stricter than the cluster id: every pool document whose normalised `vendor_name` equals that of a reserved document is dropped from the pool (169 documents; the same company can sit in two clusters). **Training sets: for seeds 0, 1, 2 a random order of the pool; the set of size N is the first N documents (nested in N), N = 10, 50, 100, 200, 500, 1000.**
- **Resulting sizes:** pool 3,108 documents in 562 layouts. Test: 200 documents, 236 pages, 886 rows (median 2 rows per document, maximum 64). Pilot: 30 documents, 38 pages. Training layouts per set (seeds 0 / 1 / 2): N = 10: 10 / 9 / 10; 50: 46 / 42 / 46; 100: 74 / 65 / 78; 200: 110 / 107 / 121; 500: 188 / 178 / 207; 1000: 282 / 286 / 294. Training rows (seed 0): 113, 321, 704, 1,494, 3,601, 7,392 (seed 2 with N = 10 has only 55). So the curve covers about 10 to 290 training layouts, against 1 to 40 on FATURA.
- **Checks passed in the script:** test, validation and pilot documents and layouts are disjoint; no training document shares a layout or a known vendor with a reserved document; training sets have no repeats.
- **Why test layouts are one document each, drawn per layout:** each test document is a different unseen layout, so a bootstrap over documents is a bootstrap over layouts (study 33's design), and large layouts do not dominate the test. A consequence: the test favours small, rare layouts, as in the wild; large layouts contribute to training only through the pool.
- **Limits:** the vendor check covers only the 165 of 200 test documents with an annotated `vendor_name`, so 35 test documents rely on the cluster id alone; layout clusters are the benchmark's own, near-duplicate layouts across clusters are possible; test documents have few rows (median 2), so row-grouping differences will be small per document (886 rows, about 3.5k cells in total); the pool lacks the 169 vendor-overlap documents and the whole of every reserved layout (pool 3,108 of 5,325 documents).

**Step 3 built (BROS line-item tagger plus row grouping, 2026-11-21; `scripts/adbuy_bros.py`, `configs/adbuy_rows_bros.yaml`, `notebooks/kaggle_adbuy_bros.ipynb`; CPU parts checked, training not yet run).**
- **Design:** Track-1 BROS (unchanged recipe, one training seed 13) tags the five fields; the existing ad-buy loader ignored the nested line-item annotations, so tags are built from the gold cells here. A cell on two OCR lines is not a run of consecutive words and BIO cannot hold it, so each run of consecutive words is tagged as its own span (B at the start of every run). Predicted spans are grouped into rows by `group_rows`: per page the field with the most spans is the anchor, every span goes to the closest anchor at or above it (tolerance 0.5 span heights), so a second line of a cell joins the row above. Runs: N = 10, 50, 100, 200 training documents x official unseen-template splits SD 0, 1, 2 = 12 runs (early stopping on the split's validation documents), each predicting the 143 primary documents; scored with `lir_metric` (strict word-set and value level) by `adbuy_bros.py score`, which writes per-document counts for the later bootstrap against the recipe.
- **CPU oracle (`adbuy_bros.py oracle`), gold cells through the tag and decode path:** 3.3% of the 38,204 gold cells (4.0% in the training documents of split 100/SD_0, 3.5% in the primary documents) are not consecutive words. **Perfect tags decoded to spans, with the true row of each span: `lir_f1` = `cell_f1` = 95.18 (all 641 documents), 94.89 (primary 143); value level 95.7 / 95.4.** So the tagger's own ceiling on the strict metric is about 95, below the recipe's oracle of 97.8, because a two-line cell becomes two spans. **Perfect spans with rows by `group_rows`: `lir_f1` 91.63 (all), 90.01 (primary), 90.09 (training documents of split 100/SD_0) against cell F1 95.18 / 94.89 / 94.13; value level 93.8 / 93.1 / 92.7.** The grouping step alone costs about 4 to 5 row-F1 points with perfect spans.
- **Choices made on the 100 training documents of split 100/SD_0 only (no model, primary set not used for selection):** tolerance 0 / 0.25 / 0.5 / 1.0 / 1.5 span heights gave row F1 72.6 / 89.9 / 90.1 / 90.2 / 88.4, so 0.5 was kept; merging same-field pieces of one row into one cell (justified by the gold: no line-item header repeats a field) lowered row F1 to 89.5 and was rejected.
- **How to read the later comparison:** the row F1 of this BROS baseline is capped near 90 by a crude grouper and near 95 on cells by the tagging; report the cell F1 next to the row F1 so that a gap to the recipe is not read as only a grouping effect. A stronger grouper (for example the study-26 link head) is untested.
- **Checked here:** `adbuy_bros.py score` on span dumps made from the gold tags reproduces the oracle numbers exactly (90.01 / 94.89 on the primary set); the BROS span extraction is a local copy of the strict IOB2 rule of `src/evaluate.py` (that module needs seqeval, not installed locally), so the Kaggle dump uses the original. **Not checked locally (no torch or transformers here):** the `train` command, the config and the notebook beyond compiling every cell; the first test is the notebook's smoke cell (one epoch on 12 documents).

**Step 4 prepared (the recipe on the 143 primary documents; `notebooks/kaggle_adbuy_sg.ipynb`, no model run yet).** Seven cells in the usual order: clone and install; VRDU ad-buy clone and input checks (641 documents and pdfs, 143 primary, 30 pilot); smoke (metric self-check, oracle through the decode path on 5 documents, the real 8B model on 3 pages per arm, the visual-token budget of one page image); run (pilot of 30 documents / 74 pages with the image arm, then the format gate, then A2 and A1 on the primary documents; **a failed gate skips the primary runs and pushes the pilot only, no automatic fallback**); score (`adbuy_sg.py score`: row F1 and cell F1, strict and value level); display; push. Two shards, one per T4, resumable, deadline 5 h. Time not measured: estimate 0.4 h for the pilot and 1.5 to 2.5 h per arm (367 pages; answers of a few hundred tokens, up to 3,000 for long tables; a page whose JSON never closes costs the full 3,000 tokens). The comparison with BROS (per-document counts of both sides, bootstrap over documents) is a separate CPU step once both results are in.

**Prepared on CPU while the two ad-buy runs train (2026-11-21).**
- **Comparison script (`scripts/adbuy_compare.py`):** per-document counts of both sides, bootstrap over the 143 documents (1,000 resamples, the same resample for every system), BROS at each N = mean over its splits; recipe minus BROS with a 95% interval and the registered verdict (LLM wins if the interval is above 0, tie if it contains 0, BROS wins if below); primary metric strict row F1, the other three reported; crossover = largest N where the recipe wins and smallest N where BROS wins. Tested end to end on fake data (oracle-made BROS dumps with spans dropped at a known rate, oracle-made recipe answers with and without wrong ids); the word indexing of the BROS side and the recipe side is asserted equal per document. One of the 143 primary documents has no line items, so the comparison runs on 142.
- **DocILE rows repeat fields, ad-buy rows never do.** In DocILE 26.4% of all rows (17.4% of the test rows) hold two or more cells of the same field (a second description line, two dates), so a one-value-per-field row object loses 8.4% of all cells (5.7% on the test set): with that format the oracle through the decode path scored 97.57 row F1 on the test documents. The DocILE answer format therefore lets a field hold a list of cells (a multi-line cell stays one string with one 'text identifier' per line). With it the **oracle through the decode path on the 200 test documents is 99.98 row F1 and 99.79 cell F1** (`scripts/docile_sg.py oracle`). In ad-buy no line-item header repeats a field (0 of 9,163 groups), so the ad-buy format is unchanged.
- **DocILE recipe code (`scripts/docile_sg.py`, `results/docile_sg/{test,pilot}.json`; no model run):** same method as `adbuy_sg.py` (one call per page, pieces cut at 0.5 line heights, row-position repair, run, gate and scorer with per-document counts) with the 19 line-item fields. Field descriptions are written from the field names alone and frozen (`DESC`). Stub run, gate and scorer checked on CPU. Value-level scoring is not defined for DocILE (its own rule is the word-set match).
- **DocILE BROS ceilings (`scripts/docile_bros.py oracle`, CPU, training code deliberately not written until the ad-buy results are in):** 19 fields; 1.0% of the test cells (1.6% in 200 training documents) are not consecutive words; **perfect tags decoded to spans: 97.83 cell F1 on the test documents; with rows from the same grouper as ad-buy (tolerance 0.5, chosen on ad-buy training documents only): row F1 94.43 (test), 95.65 (training documents), against cell F1 97.83 and 97.11.** The grouping step costs about 3.4 row-F1 points with perfect spans (ad-buy: about 5).

**First Kaggle smoke runs of both ad-buy notebooks (2026-11-21): two failures, both fixed before any result.**
- **BROS notebook:** `src/smoke_test.py` with no arguments smoke-tests five configs, including the shelved `vrdu_graph_bros.yaml`; its `GraphBrosSelfAttention.forward()` does not accept the `head_mask` keyword that the installed transformers passes (`TypeError`). That is not the Track-1 BROS used here. The notebook now runs `src/smoke_test.py --configs configs/adbuy_rows_bros.yaml` (one forward and backward step of the recipe actually trained); the shelved model is untouched. Inferred from the traceback and the script's default config list; the traceback's earlier lines were not seen.
- **Recipe notebook:** the first two pages took 263 s and 242 s (pages with 13 and 11 gold rows) and the third page (a document whose gold has 3 rows on that page) ran out of memory (`Tried to allocate 3.53 GiB`, 12.25 GiB in use, 1.61 GiB reserved but unallocated).
  - *Prompt size was underestimated.* With the real Qwen3-VL tokenizer (public `tokenizer.json`, run locally) the text prompts of the 367 primary pages are median 3,020, 90th percentile 5,109, maximum 8,088 tokens (the 3-characters-per-token estimate gave 1,790 / 2,768 / 9,623 in characters/3 terms): digits are one token each and these pages are mostly numbers. With the image tokens a page reaches 6 to 9k tokens, which fits the 3.53 GiB allocation (about 6,000 positions of float32 logits, or attention at that length).
  - *Time was underestimated.* The gold answers of the 367 pages total 170,551 tokens as compact JSON (188,397 with indent 2, 134,712 as plain row lines, 68,207 as ids only). The smoke pages imply about 4 to 5 output tokens per second per T4 for the 4-bit 8B model (study 33's pages implied about 11), so one arm needs about 5 h on two T4s (3.9 h at 6 tokens per second, 2.2 h at 11); my earlier estimate of 1.5 to 2.5 h per arm was wrong.
  - *Citing ids only would halve the answer but is not viable:* 22.6% of the gold cells are strictly inside their piece, so ids alone cap row F1 at 77.4.
  - *Why the image budget was not cut (decided 2026-11-21 after a first version of these changes had cut it to 640 tokens):* the page that ran out of memory has a text prompt of 4,508 tokens (page 1 of the same document: 5,099); the old budget allows at most 980 visual tokens (one per 32 x 32 pixels), so its sequence is about 5,500 positions, and 3.53 GiB is what one fp32 attention matrix of 32 heads costs at that length (32 x 5,440^2 x 4 bytes). Memory grows with the square of the length, so halving the image to 490 tokens would cut it only by about 17% (3.6 to 3.0 GiB) and would not have saved that page, while it would change the A2 setting of studies 25 to 33 and lower what the image can show (a 490-token page is about 0.5 megapixels). The memory problem is therefore not an image-budget problem and is handled per page instead.
  - *Cause of the out-of-memory, checked on CPU (2026-11-21, `scripts/attn_probe_cpu.py`, venv with transformers 4.57.6 and torch 2.15 CPU, the version range the notebooks install; no GPU here, so memory itself was not measured).* With the real Qwen3-VL processor the failing page (page 0 of document 0617c152) is **5,463 positions** at the 1,280 budget (945 visual tokens) and page 1 is 6,054; at the 640 budget 4,993 and 5,584. One fp32 attention matrix over 32 heads costs 3.56 GiB at 5,463 positions (the failed allocation was 3.53 GiB, the size for 5,442 positions) and 4.37 GiB at 6,054; fp32 logits for every position would cost 3.09 GiB at 5,463 and do not match. A tiny random Qwen3-VL confirms that prefill computes logits for one position only (`logits_to_keep`), and that `attn_implementation="sdpa"` reaches the text and vision sub-models. It also shows that transformers 4.57 calls `scaled_dot_product_attention(..., is_causal=True, enable_gqa=True)` with no mask. **Hypothesis:** a T4 has no flash kernel and the memory-efficient kernel does not take `enable_gqa` (not verified; the probe below measures it), so PyTorch falls back to the math kernel, which builds the full L x L matrix, so memory grows with the square of the length and every page above about 5k positions is at risk (the median text prompt alone is 3,020 tokens).
  - *Fix:* `adbuy_sg.patch_sdpa()` makes transformers repeat the keys and values instead of passing the flag (checked on the tiny model: keys then have 4 heads and `enable_gqa` is None), which should keep the memory-efficient kernel; it is applied in every `run` process before the model loads. **Kaggle probe** `scripts/attn_probe_gpu.py`, run in the smoke cell before the model smoke: bare attention at 2,000, 5,500 and 9,500 positions (default dispatch with the flag, efficient kernel only with the flag, efficient kernel only with repeated keys), then the real model prefilling random tokens of 3,000 to 9,500 positions with and without the fix, printing peak memory; it exits with an error only if the fixed model cannot prefill 5,500 positions. If the fix does not work, the table shows where memory runs out and the next options are the text-only cap or splitting a page in two.
  - *Probe result on the Kaggle T4 (2026-11-21, `attn_probe_gpu.py`, real model, text-only prefill; attention implementation `sdpa`; 6.27 GiB allocated after loading, 7.16 GiB free of 14.56): as shipped, 3,000 positions peak 9.43 GiB and **5,500, 7,500 and 9,500 positions run out of memory**; with the fix, peak 7.01 GiB at 3,000, 7.58 at 5,500, 8.05 at 7,500 and 8.53 at 9,500.* Memory with the fix grows by about 0.23 GiB per 1,000 positions, so the hypothesis (math fallback with the GQA flag) is supported by the failure pattern; the bare-attention table (kernel support) was not read. The longest primary page (8,110 text tokens plus about 975 image and template positions) peaks near 8.6 GiB, leaving about 6 GiB free, which also leaves room to batch pages if throughput needs it.
  - *Image cap changed from 5,500 to 8,500 text tokens (2026-11-21):* 22 of the 367 primary pages (6.0%) exceed 5,500, and the fix makes the cap unnecessary; the out-of-memory retry stays as the safety net. A Kaggle session started before this change used 5,500: its rows carry `image_dropped`, so A2 can be reported with and without those pages.
  - *Changes (frozen before any result):* the image budget stays at 1,280 (at most 980 visual tokens); a page whose text prompt exceeds 5,500 tokens, or whose generation runs out of memory, is answered from the text alone (`image_dropped`, and `oom_retry` for the memory case; peak GPU memory per page is saved too, so the share of such pages and the memory use are known after the pilot); answers requested as one row object per line without indentation; at most 2,400 new tokens per page (the longest gold answer is 1,965); `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and an `empty_cache` after every page; per-page prompt and output token counts and tokens per second are logged and saved; **one arm per session** (`RUN_ARMS`, A2 first, then A1 in a second session with the same document order); a gate kept in an earlier session is not repeated; the scorer and `adbuy_compare.py` score only documents with every page answered, so a run cut by the deadline is compared with BROS on the documents that finished (and A2 and A1 on the documents both finished).
  - *Still unmeasured:* the throughput of the changed setup (the smoke cell and the pilot print it) and the A2 and A1 documents-finished counts.

**DocILE links and the header rule baseline (CPU, 2026-11-21; `scripts/docile_headers.py`, `results/docile_rows/header_rule.json`; annotations only, no model).**
- **What can be linked on DocILE:** the 36 key-information fields come as value boxes only (the label text is not annotated, so there are no key-value links to score for them). Line items have row ids (cells of one row) and annotated header boxes, each carrying the field type it names, so the links that exist are cell to row and cell to column header. In 2,000 documents 85.7% of the line-item cells have a header of their own field on the page; 65% of key-information values are a whole OCR piece (label elsewhere) and 30% share the piece with other text (a label in the same line).
- **Rule baseline:** give each gold line-item cell the type of a header of its page, using the gold type of the header (so only the linking geometry is tested). R1 = the nearest header above whose horizontal extent overlaps the cell's by at least 30% of the narrower box; R2 = R1, else the header with the largest horizontal overlap, else the nearest by centre distance. **All 5,680 documents (157,469 cells): R2 gives the right type for 99.0% of the cells whose type has a header on the page (R1 94.6%), but 13.3% of the cells have no header of their type on the page, so R2 is right for 85.8% of all cells (R1 82.0%). Test documents (200, 3,225 cells): 97.9% of the recoverable cells, but 20.6% are not recoverable, so 77.8% of all cells.** Pages where a field has several headers (several tables or repeats) are as good as the others (99.2% against 98.8%), so the nearest header above picks the right table. Weakest fields: currency 46% (1,196 cells in all documents), units of measure 91%.
- **Reading:** column geometry is not the hard part; the missing headers are (continuation pages, unlabelled columns, rare layouts have fewer annotated headers), so a link-only design would be capped near 78% (test) to 86% (all) of the cells against 97.8% for typing cells from their content (tagger oracle). The header is a useful extra signal (our prompt already shows it to the model as text) but not a replacement for typing from content. Not computed: link F1 on predicted cells and headers, and whether the 12% of cells whose nearest header is below were repeated headers or section headers.

**Results of steps 3 to 5 on VRDU ad-buy (runs of 2026-11-22; `results/adbuy_bros/run`, `results/adbuy_sg/run`; comparison `results/adbuy_sg/run/comparison.json`).**
- **Run health.** BROS: 12 runs, 3.4 h, no failed shard. Recipe (A2 only, session of 5.03 h): pilot gate `keep` (parse 1.00, cited ids that exist 0.9025, 98.3% of pages with gold rows answered); 335 pages answered, 0 errors, **0 out-of-memory retries** (the attention fix works; peak 8.2 GB), 21 pages answered without the image (the old 5,500-token cap), 9.4 output tokens per second, mean 85 s per page, mean 804 output tokens per page (the gold answers average about 465); **32 pages (9.5%) hit the 2,400-token output cap**; **128 of the 143 primary documents were complete** when the deadline stopped the run, and only those are scored (A1, the text arm, has not been run).
- **Registered comparison (128 documents, row F1 strict, bootstrap over documents; BROS = mean of three splits per training size N):** recipe A2 **38.16**; BROS 68.94 (N = 10), 82.87 (50), 83.53 (100), 85.30 (200); **A2 minus BROS -30.78 [-35.29, -25.65] at N = 10, -44.71 at 50, -45.38 at 100, -47.15 at 200 (all intervals below 0): BROS wins at every training size tested; there is no N >= 10 at which the recipe wins.** Cell F1: A2 39.45 against BROS 72.20 / 87.01 / 87.97 / 89.98; value level (row): A2 46.76 against 72.23 / 86.07 / 86.58 / 88.76 (all BROS wins, intervals below 0).
- **BROS alone (all 143 primary documents, the three splits' mean):** row F1 68.84 / 83.15 / 83.75 / 85.65 and cell F1 72.45 / 87.47 / 88.32 / 90.32 at N = 10 / 50 / 100 / 200; the N = 200 row F1 is below the grouping-rule ceiling with perfect spans (90.01), the N = 200 cell F1 about 4.6 points below the tagger ceiling (94.89).
- **Where the recipe loses (CPU, `scripts/adbuy_sg_errors.py`):** it predicts **3,082 rows against 1,807 gold rows (1.71x)**; strict cell precision 31% and recall 49% (counts 3,578 right of 11,437 predicted and 7,318 gold); per field precision 27 to 36%, recall 30 to 60% (`program_desc` worst: 27 / 30). Of the 7,738 predicted cells that are not exact gold cells, 77.3% touch no gold cell, 8.5% overlap a gold cell of the same field (boundary), 8.0% repeat an already matched gold cell, 4.0% are the words of a gold cell of another field. **The 30 pages that hit the output cap produced 1,005 rows against 245 gold rows** (runaway repetition); 20 pages without any gold row received 256 predicted rows. Decode: 1,168 of 12,605 cells (9.3%) could not be grounded and 1,920 (15%) needed the repair.
- **Post hoc, how much of that is fixable by cleaning (`scripts/adbuy_sg_posthoc.py`; written after the result):** row F1 38.2 as is, 40.2 after removing repeated rows, 43.0 after also dropping the capped pages, 43.8 after also dropping pages without gold rows (precision 41.8, recall 45.9). **Even with that oracle-style cleanup the recipe stays about 25 points below BROS at N = 10 (68.9)**, so the gap is not a decoding accident.
- **Reading and limits.** On dense, numeric, scanned line-item tables the zero-shot recipe (Qwen3-VL-8B in 4 bits, one call per page, long row output) is far below a BROS tagger trained on only 10 documents; the trained tagger's data efficiency is high here (N = 10 already 69). Not tested: why (untested hypotheses: OCR tokens that glue a channel and a date while the gold cells are sub-token, long structured outputs from a 4-bit 8B model, no worked example from the domain, rows with several lines), the text-only arm, other models or prompts, and N below 10 (where BROS must fall below the recipe's 38 somewhere: a crossover, if any, lies at one to a few training documents). 15 of 143 documents are not scored for the recipe because of the deadline. The ad-buy pages are harder than the tables of published benchmarks (median 7 gold rows per page with rows, up to 47; VAREX tables have a median of 3 rows on synthetic digital forms), so published table scores for similar models are not a reference for this number.

**Next:** rerun both notebooks (separate sessions, any order): `kaggle_adbuy_sg.ipynb` smoke cell first (it now prints prompt tokens, output tokens, tokens per second and whether the image was dropped), then `kaggle_adbuy_bros.ipynb`; then `adbuy_compare.py`.

**Evidence:** `scripts/lir_metric.py` (self-check passes); dataset profile as above (no result file written yet).


---

## 35. Efficient linking on tables: row grouping and header typing with a learned pair classifier — CPU first, registered 2026-11-21

**Question.** Our table pipeline needs two links once cells are found: cell to row (which cells form one line item) and cell to column header (which field a cell is). Study 34 groups rows with a hand-made anchor rule (`adbuy_bros.group_rows`) and types cells from content. With perfect cells the rule gives row F1 94.43 on the DocILE test documents (cell F1 97.83) and 90.01 on the ad-buy primary documents (cell F1 94.89). **Can a learned pair classifier over cheap candidate pairs group rows better than the rule, at a cost that does not grow with the square of the page?** The rule baseline for header linking is already measured (study 34: the nearest header above recovers the type of 99.0% of the cells that have a header of their type, but 13.3% of all cells, 20.6% of the DocILE test cells, have none).

**What the literature does (focused search, 2026-11-21; read from search summaries and abstracts, not in full).**
- *Token tagging for membership, then grouping:* the DocILE baselines tag every token as inside a line item and group by the predicted line-item tags ([DocILE benchmark](https://arxiv.org/pdf/2302.05658)); [Holeček et al. 2019](https://arxiv.org/abs/1904.12577) classify each word box of an invoice graph as line item or not and leave the table to clustering; [Spatial ModernBERT 2025](https://arxiv.org/abs/2507.08865) uses one head for semantic labels, one for column indices and one that separates item rows from header rows.
- *Merger or pair modules on a text-box graph:* the DocILE CLEF 2023 notebook of USTC and iFLYTEK ([paper 052](https://ceur-ws.org/Vol-3497/paper-052.pdf), GraphDoc backbone, ranked first on key-information localisation in the leaderboard summary I found) classifies text boxes, then runs an instance-merge and a line-item-merge process; the exact merger design and its line-item score were not in the summaries. [PEneo 2024](https://arxiv.org/html/2401.03472v3) does line extraction, line grouping and entity linking in one pass for key-value pairs, against the usual two-step recognition then pair classification of every candidate pair.
- *Relation prediction over cells:* table-structure recognisers link cells (GraphTSR; [TSRFormer](https://arxiv.org/abs/2208.04921) forms a grid and a relation network merges adjacent cells to recover spanning cells); these work on table images or detected tables, not on invoice pages with unbordered tables.
- *Generative:* [LMDX](https://arxiv.org/abs/2309.10952) outputs hierarchical entities (line items with their fields) from an LLM, which is our recipe; a 2026 WACV paper benchmarks zero-shot table detection and structure recognition on DocILE-QUEST ([poster](https://wacv.thecvf.com/virtual/2026/poster/338)); I found no source testing row-structure prompting on DocILE.
- *Rules:* patents cluster cells into header and value regions; the [DoSA](https://arxiv.org/pdf/2211.04934) annotation tool links a value to the nearest preceding key and reports noisy pairs.
- *Gaps in what I found:* no top-team line-item score, no paper comparing groupers at different numbers of training layouts, none that scores header-to-cell links as links.

**Efficiency argument (ours).** An all-pairs head over n cells costs n squared; a page of 150 cells gives 22,000 pairs. Rows are local: a cell and its row-mates sit inside a vertical band. Candidate pairs are therefore restricted to cells of the same page whose vertical distance is at most a few row heights (measured on the training documents), which is linear in n, and the classifier is a small model on pair features, so no GPU is needed.

**Step 1 (CPU, the study's first run).** Group gold cells into rows; compare, by strict row F1 with `lir_metric`:
- (a) the anchor rule (reference: 94.43 DocILE test, 90.01 ad-buy primary);
- (b) a pair classifier on candidate pairs (same row or not) with features of the two cells: relative vertical and horizontal offsets in units of the cell heights, overlap of the vertical extents, order, field types, whether the two types occur together in a row in the training documents, horizontal overlap with the nearest header above; clusters = connected components of the pairs scored above a threshold, with a split rule for components that hold two cells of a field that never repeats in a row (ad-buy);
- (c) (b) without the header feature (what the feature adds).
Training only on the training documents of each split (DocILE: the pool order of `splits.json`, N = 200 and 1000, seed 0; ad-buy: the training documents of the unseen-template splits, N = 100 and 200); threshold chosen on the validation documents of the same split; evaluation on the 200 DocILE test documents and the 143 ad-buy primary documents; bootstrap over documents (layouts for DocILE).

**Registered rule.** The learned grouper replaces the anchor rule in the BROS baselines if its row F1 minus the rule's is at least +1.0 on both test sets with an interval above 0; if it wins on one set only, it is reported and not adopted; if it does not win, the rule stays and the grouping ceiling is stated as it is. The rule baseline for headers is already the reference; step 1 does not retype cells.

**Later steps (not registered yet).** (2) End to end: the pair head inside the BROS tagger (study 26 style) with predicted cells, link F1 and row F1 against tag-then-group, on a GPU. (3) The header text as an input to the recipe prompt as an ablation (the prompt already contains it as text).

**Limits stated up front.** The classifier sees gold cells (perfect cells), so the gain is an upper bound for a tagger's output; the header feature uses the gold type of the header (as in the rule baseline); DocILE test documents have a median of 2 rows, so row differences are small per document (886 rows); one classifier family only; the literature summary rests on search snippets.

**Results (step 1, 2026-11-21; `scripts/table_rows_learn.py`, `results/table_linking/step1.json`; input = the spans a perfect tagger outputs; strict row F1; bootstrap over documents):**

| Set | Anchor rule | Learned, training documents | Row F1 | Threshold | Learned minus rule [95% CI] |
|---|---|---|---|---|---|
| DocILE test (200 documents) | 94.43 | N = 200, with header feature | 86.00 | 0.9 | -8.21 [-13.55, -3.11] |
| | | N = 200, without | 86.61 | 0.9 | -7.61 [-12.84, -2.64] |
| | | N = 1000, with (primary) | 90.10 | 0.9 | **-4.32 [-9.28, -0.90]** |
| | | N = 1000, without | 90.59 | 0.9 | -3.83 [-9.26, -0.32] |
| Ad-buy primary (142 documents) | 90.01 | N = 100 | 92.64 | 0.2 | +2.61 [+0.53, +4.95] |
| | | N = 200 (primary) | 92.39 | 0.2 | **+2.38 [+0.42, +4.71]** |

- **Registered rule: not met.** The learned grouper beats the rule on ad-buy only (+2.38, interval above 0) and loses on DocILE (-4.32, interval below 0); a win on one set only is reported and not adopted, so **the anchor rule stays in the BROS baselines**. The header feature does not help on DocILE (with the feature is 0.5 points lower at both N).
- **Post hoc checks (written after the result; `scripts/table_rows_learn_posthoc.py`, `step1_posthoc.json`):** the chosen thresholds sat at the edge of the registered grid (0.9 on DocILE, 0.2 on ad-buy), so a wider grid was tried. *Pair quality:* DocILE test AUC 0.9725, precision 0.961 and recall 0.892 at 0.5, the candidate band holds 98.9% of the same-row pairs; ad-buy AUC 0.998, precision 0.990, recall 0.984, band recall 97.3%. *DocILE, N = 1000, wider grid:* best validation threshold 0.995 (validation row F1 93.5) gives test 91.13, still **-3.31 [-5.96, -1.14]** against the rule; with the 'no repeated field in a row' constraint (right for ad-buy, wrong for DocILE where 26% of rows repeat a field) test 86.93 (-7.56). *Ad-buy, wider grid:* validation row F1 is flat from 0.02 to 0.5 (92.0 to 91.8), test 92.41, +2.40 [+0.46, +4.73] (so that win does not depend on the threshold). *Same documents, validation:* the rule scores 94.76 on the 100 DocILE validation documents against the learned grouper's 93.5 (-1.3), and 91.40 on the 100 ad-buy validation documents against 92.0 (+0.6), so the sign is the same on selection and test data.
- **Reading:** the pair classifier is good (AUC 0.97 to 0.998); the loss on DocILE comes from turning pair scores into rows. Single-link clustering lets one false pair merge two rows, and DocILE tables are dense and repeat fields, so the constraint that fixes ad-buy cannot be used. The anchor rule uses structure the classifier does not see (one anchor field per row), which helps on DocILE and costs on ad-buy, where the learned model recovers about 2.4 points of the 4.9 that grouping loses. Untried: a different clustering over the same pair scores (average-link, correlation clustering, or assigning every span to its best anchor with learned scores), features of neighbouring rows, and the dataset difference in training size (ad-buy N = 200 documents against DocILE N = 1000 documents with few rows each).
- **Decision for the baselines:** keep the rule on both datasets (the registered rule). The ad-buy gain (+2.4 on the perfect-span ceiling) is reported as a possible improvement of the ad-buy BROS baseline that was not registered as an adoption.
- **Caveat on the post hoc work:** the DocILE test documents were used again for the post hoc rows; the threshold was chosen on validation only, and no variant was picked by its test score.

**Step 1b: a different clustering over the same pair scores (registered 2026-11-21, before it was run; `scripts/table_rows_cluster.py`).**
- **Why:** step 1 showed good pair scores (AUC 0.97 to 0.998) but single-link clusters that let one false pair merge two rows. The classifiers, training sets (DocILE N = 1000 with the header feature, ad-buy N = 200), candidate pairs, inputs (perfect-tag spans) and test sets are those of step 1; only the way pair scores become rows changes.
- **Variants (each with its grid fixed here):** V0 = single link, as in step 1 with the wider threshold grid {0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98, 0.99, 0.995}. V1 = average link: repeatedly merge the two clusters with the highest mean pair probability (pairs outside the candidate band count as 0) while it is at least tau, tau in {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}. V2 = greedy correlation clustering: edge weight logit(p) + b, pairs outside the band weight logit(0.01) + b, merge the two clusters with the largest positive total weight between them, b in {-3, -2, -1, 0, 1, 2, 3}. V3 = anchor assignment with learned scores: the anchor field of a page is the one with most spans (as in the rule), each other span joins the anchor with the highest pair probability if that is at least tau, else the nearest anchor by vertical distance, tau in {0, 0.1, 0.3, 0.5}. On ad-buy V0, V1 and V2 keep the rule 'no two cells of one field in a row'; on DocILE no constraint.
- **Selection (fixed before running):** per dataset, the variant and its hyperparameter are the ones with the highest row F1 on the split's validation documents (DocILE: the 100 validation layouts; ad-buy: the 100 validation documents of the N = 200 split, SD 0); the selected configuration is scored once on the test documents; all variants at their own best validation setting are also printed (descriptive).
- **Registered rule:** as in step 1: the selected learned grouper replaces the anchor rule in the BROS baselines if its row F1 minus the rule's is at least +1.0 with a document-bootstrap interval above 0 on both the DocILE test documents and the ad-buy primary documents; a win on one set only is reported, not adopted.
- **Limits:** the classifier sees gold-derived spans (an upper bound for a tagger's output); four variants and about 30 settings are searched on 100 validation documents per dataset, so the selected validation score is optimistic; the DocILE test documents were already used in step 1 and its post hoc checks, though never for selection.

**Step 1b results (2026-11-22; `scripts/table_rows_cluster.py`, `results/table_linking/step1b.json`; same classifiers and sets as step 1; each variant at its best validation setting; test row F1 minus the anchor rule with a document-bootstrap interval):**

| Variant | DocILE: validation / test, minus rule [95% CI] | Ad-buy: validation / test, minus rule [95% CI] |
|---|---|---|
| V0 single link | 93.54 / 91.13, -3.31 [-5.96, -1.14] | 91.99 / 92.41, +2.40 [+0.46, +4.73] |
| V1 average link | 94.71 / 92.54, -1.89 [-4.01, +0.03] | 91.74 / 92.29, +2.28 [+0.33, +4.58] |
| V2 correlation | **94.71** / 92.57, -1.86 [-3.97, +0.03] | 91.91 / 92.47, +2.46 [+0.51, +4.76] |
| V3 anchors | 91.37 / 87.70, -6.75 [-10.91, -2.58] | **92.17** / 92.58, **+2.57 [+0.67, +4.89]** |
| anchor rule (reference) | 94.76 / 94.43 | 91.40 / 90.01 |

- **Selected by validation:** V2 on DocILE (tie on validation, 94.71 for V1 and V2), V3 on ad-buy.
- **Registered rule: not met.** DocILE: V2, -1.86 [-3.97, +0.03] (the interval just reaches 0, the point estimate is below the rule); ad-buy: V3, +2.57 [+0.67, +4.89]. A gain on one set only is reported and not adopted, so **the anchor rule stays in the BROS baselines on both datasets.**
- **Reading:** the clustering was the weak part of step 1. Average link and correlation clustering recover most of the DocILE gap (single link -3.3 to -1.9 points) and tie the rule on the DocILE validation documents (94.71 against 94.76), but on the test layouts they stay about 1.9 points below it; no clustering beats the rule on DocILE. On ad-buy all four variants gain 2.3 to 2.6 points, so that gain does not depend on the clustering. The ad-buy gain is largest for the variant that keeps the rule's anchor structure (V3), the opposite of DocILE where V3 is the worst, which fits the structure of the two tables: ad-buy rows have one anchor-field cell and no repeated fields, DocILE rows repeat fields and have a variable set of fields.
- **Limits:** about 30 settings and four variants were searched on 100 validation documents per dataset (the selected validation scores are optimistic); validation and test disagree on DocILE (a tie on validation, -1.9 on test), consistent with layout shift between document sets; the classifier still sees gold-derived spans (a ceiling for a tagger's output); the DocILE test set has a median of 2 rows per document.
- **Decision and consequence:** no change to the baselines; the learned grouper is a possible dataset-specific improvement for ad-buy (about +2.4 to +2.6 on perfect spans), not adopted because it does not generalise to DocILE under the registered rule. The end-to-end pair head inside the tagger is not worth building on this evidence: with perfect spans the learned grouping gains at most about 2.5 points on one dataset and loses on the other.

**Next:** back to the main line: the ad-buy recipe and BROS results, then `adbuy_compare.py`.

**Evidence:** `scripts/table_rows_learn.py`, `scripts/table_rows_learn_posthoc.py`, `results/table_linking/step1.json`, `results/table_linking/step1_posthoc.json`; `scripts/docile_headers.py`, `results/docile_rows/header_rule.json` (rule baseline for headers, study 34).


---

## 36. A learned router between the schema-guided recipe and BROS — groundwork (CPU), 2026-11-22

**Question.** Can a small model, given a form (or a field of it), pick the system that will be right, the recipe (Qwen3-VL-8B, schema-guided, grounded) or a trained BROS tagger? Earlier evidence on routers is mixed: a static per-field router with BROS gave VRDU +3.8 [+2.0, +5.6] over BROS (study 16, old 3B VLM), the per-document confidence gate alone -0.2 on VRDU and +1.5 at FATURA K = 2 (study 16), the errors of BROS and the old VLM were positively correlated (lift 0.46 on VRDU, 0.81 / 0.67 / 0.51 on FATURA at K = 2 / 5 / 10, study 18), and with the new recipe a router with BROS reached 79.2 against 70.5 for BROS and 76.5 for the recipe alone on VRDU (study 31). FATURA with the new recipe: filling one system's empty fields with the other never beat the better system alone (study 33).

**Groundwork result (`scripts/router_headroom_fatura.py`, `results/router/headroom_fatura.json`; the 500 FATURA confirm invoices, recipe A2 with the text-first decode, BROS dumps of study 15 averaged over seeds 13 and 42; the BROS and recipe numbers reproduce the registered ones: 32.5 / 69.2 / 90.8 / 95.9 / 99.9 and 94.0).** An oracle router picks, per invoice and field, the system whose block is right, abstains when the field has no gold, and otherwise keeps the recipe's answer:

| K seen templates | BROS | Recipe | Oracle router | Headroom over the better system | Gold fields right only for the recipe / only for BROS |
|---|---|---|---|---|---|
| 1 | 32.5 | 94.0 | 98.1 | +4.2 | 74.1% / 0.3% |
| 2 | 69.2 | 94.0 | 99.4 | +5.4 | 31.8% / 2.2% |
| 5 | 90.8 | 94.0 | 99.8 | +5.8 | 7.0% / 3.0% |
| 10 | 95.9 | 94.0 | 100.0 | +4.1 | 3.3% / 3.3% |
| 40 | 99.9 | 94.0 | 100.0 | +0.1 | 0.2% / 3.4% |

- **Reading:** there is room for a router (+4 to +6 points around the crossover, K = 2 to 10), and the share of fields where only BROS is right stays near 3% even at K = 40, so the two systems fail on different fields to a useful degree with the new recipe. The oracle also knows when a field is absent, so part of its gain is abstention (the recipe predicts `buyer` for templates whose gold has none); I did not separate the two (a first attempt at a no-abstention oracle forced an answer and was discarded as unfair).
- **Not tested:** any deployable router; features (agreement of the two systems, BROS span confidence, similarity of the layout to the training layouts, field type, K); labels for VRDU, ad-buy and DocILE (they need the recipe run on those sets and BROS predictions on held-out layouts).

**Plan (not registered for a run yet).** Leave-templates-out cross-validation on the FATURA dumps for K = 2, 5, 10: logistic regression and gradient boosting on the features above, at field level, against the better single system and the oracle; the rule to be fixed before running. A neural network is not needed for a handful of features; a small model is enough and is cheaper to validate.

**Evidence:** `scripts/router_headroom_fatura.py`, `results/router/headroom_fatura.json`.


---

## 37. The joint tagger-plus-biaffine-linker with a BROS backbone, against LiLT — GPU (Kaggle T4 x 2), registered 2026-11-22

**Question.** Every linking result so far (studies 19 to 26) comes from the joint model with a LiLT backbone: token tags plus a biaffine link head (`src/models/joint_extractor.py`). Attaching that head to BROS was the project's original, untested hypothesis (`NOVELTY_ASSESSMENT.md` section 7.3), and BROS tags unseen layouts better than LiLT on VRDU (Track-1 BROS 71.3 against LiLT 64.2 on unseen templates). **With the study 26 A0 recipe unchanged, does a BROS backbone raise end-to-end entity and link F1 on XFUND?** The aim of the project is both tagging and linking, so one model that does both is the natural test.

**Setup.** XFUND val es / fr / de / it / pt (250 forms), seeds 13, 21, 42, FP32, entity-safe 510-subword windows with stride 255 for training and the 'windowed' reading (stride 128) for scoring, entities from the model's own tags, `src/e2e_link.py` unchanged (a link counts only if both ends are exact predicted entities of the right type and the direction is right). Scored by `scripts/xfund_e2e_eval.py`.

**Arms.** J1 = `configs/xfund_table_A0.yaml` with `backbone: bros` and nothing else changed (`configs/xfund_joint_bros_J1.yaml`; checked: the two files differ only in backbone and output directory). J2 = J1 plus `coordinate_jitter: true` (`configs/xfund_joint_bros_J2.yaml`), the one Track-1 geometry option that is safe here: **anchoring is excluded because `e2e_link.predict_window` feeds unanchored boxes while the training collator would anchor them, a train/test mismatch** found in the CPU check. Notebook: `notebooks/kaggle_xfund_bros_joint.ipynb` (J1 on GPU 0 and J2 on GPU 1 in parallel, each scored and pushed).

**Reference (LiLT A0, `results/table_links/A0_e2e.json`; the same code and data).** Mean of 3 seeds: entity 75.84, link 47.42, link_qa 53.15. Per seed 13 / 21 / 42: entity 75.79 / 75.85 / 75.88, link 48.33 / 47.01 / 46.93, link_qa 53.41 / 53.06 / 52.97. Recall by group, mean of 3 seeds: below 39.8, same line 63.5, above 34.8, rank 0 / 1 / 2-4 / 5+ 57.1 / 55.9 / 54.2 / 35.8, question to question 31.2.

**Registered rule (primary cell: link F1, each BROS arm against A0 with seeds matched).** *Better* if the mean difference is at least +1.5 points and positive in all three seed pairs; *worse* if at most -1.5 and negative in all three; otherwise *no clear difference*. Entity F1, link_qa and the group recalls are reported, not tested. No interval is computed (three seeds); the seed-to-seed spread of A0 is about 1.4 points of link F1. If any seed does not finish, that arm is reported as incomplete.

**CPU checks done before any GPU run (2026-11-22; conda env with transformers 5.17 and torch 2.14; no GPU).**
- The joint model builds with `backbone: bros` (the BROS checkpoint `jinho8345/bros-base-uncased`; the classifier layer is new, as expected) and the existing training loop runs: a 6-form CPU run went through phase 1, phase 2 and into phase 3 (token classification, linking, joint) with the real early-stopping validation; the run then finished: the test pass ran and `metrics.json`, `result.json`, `history.json` and `best_model.pt` were written (test entity F1 0.006 and link F1 0.029, as expected after one epoch per phase on 6 forms; only the chain is tested), and **`scripts/xfund_e2e_eval.py` loaded that BROS checkpoint and scored 3 val forms (all keys written, scores 0.0 for the nearly untrained model)**, so the whole chain from training to the end-to-end scorer works with BROS.
- `e2e_link.predict_document` works with a BROS model on real XFUND forms (entities and links come out; the head was untrained, so only the plumbing was tested).
- In the training collator, coordinate jitter and content-box anchoring both change the boxes BROS receives; `predict_window` does not anchor (above).
- **Correction after the first Kaggle attempt (2026-11-22).** The Kaggle run of this notebook stopped with `TypeError: PreTrainedTokenizerFast._batch_encode_plus() got an unexpected keyword argument 'boxes'` (transformers 4.51.3). Cause: `src/data/xfund_windows.word_windows`, which cuts forms into entity-safe windows for training and scoring, passed `boxes=` to every tokenizer, and a plain BERT tokenizer (BROS) rejects it in 4.x. **My CPU check above missed this because it ran under transformers 5.17, which ignores the keyword and then reads the word list as a batch of texts, so the check silently built one oversized window per form** (6 training and 111 validation windows for the 6 forms; the checks of the prediction path and of the geometry options were valid, the window counts were not). Fix: `word_windows` now asks the tokenizer's signature whether it takes `boxes` and otherwise tokenizes with `is_split_into_words=True` (the same fallback `TokenClassifier.encode` already had); on a 520-word test form BROS now gets 6 windows of at most 510 subwords under both 4.57.6 and 5.17, and LiLT's windows are unchanged. Re-checked under transformers 4.57.6 (the closest local match to the pin): the end-to-end scorer runs on the BROS smoke checkpoint and the training start builds proper windows (24 training and 244 validation windows from the same 6 forms). Not checked locally: transformers 4.51.3 itself, and any further BROS-specific path the GPU run reaches.
- BROS uses an uncased English vocabulary and XFUND is not English; LiLT-en has the same limit (LiLT with a multilingual text stream was worse, study 24), so the comparison is fair but both may be capped.

**Limits stated up front.** One recipe, written for LiLT and not tuned for BROS (learning rate, epochs, window sizes); three seeds; the BROS arms' runtime is unmeasured; XFUND val is only approximately template-disjoint (about 94% of forms have no train sibling by a question-text proxy); the A0 reference was trained in an earlier session, so a session effect cannot be excluded; FP32 only.

**Results (scored 2026-11-22 from `results/bros_joint/{J1,J2}_e2e.json`; windowed inference, all 250 val forms, mean of 3 seeds; 5.75 h on two T4s, all 6 trainings and both scoring passes returned 0).**

| Arm | Entity F1 | Link F1 | Link_qa F1 | Link F1 per seed 13 / 21 / 42 | Link F1 minus A0, same seeds |
|---|---|---|---|---|---|
| LiLT A0 (reference) | 75.84 | 47.42 | 53.15 | 48.33 / 47.01 / 46.93 | |
| BROS J1 | 80.20 | 53.35 | 59.59 | 53.10 / 52.93 / 54.01 | +4.77 / +5.92 / +7.08 (mean +5.93) |
| BROS J2 (+ jitter) | 79.63 | 53.00 | 59.66 | 53.10 / 52.72 / 53.19 | +4.77 / +5.71 / +6.26 (mean +5.58) |

**Registered rule: both BROS arms are *better* than A0** (mean difference at least +1.5 and positive in all three seed pairs; the smallest pair difference is +4.77). Not tested, reported: entity F1 +4.4 (J1) and +3.8 (J2), link_qa +6.4 and +6.5; sibling-free forms link F1 52.52 (J1), 52.24 (J2) against 46.34; single-window link F1 47.47 / 47.13 against 41.54. Recall by group (mean of 3 seeds), A0 / J1 / J2: below 39.8 / 46.5 / 47.3, same line 63.5 / 67.4 / 67.1, above 34.8 / 50.2 / 50.1, rank 5+ 35.8 / 43.3 / 43.5, question to question 31.2 / 32.2 / 32.0. J1 against J2 was not registered; the difference is 0.35 link F1 (J1 minus J2 per seed: 0.00 / +0.21 / +0.82), so jitter neither helps nor hurts visibly.

**What this shows.** With the same joint code and data, replacing the LiLT backbone with BROS adds about 6 link F1 points and 4 entity F1 points, in every seed, and the recall gain is largest for answers above their question (+15.4 points; below +6.7) and smallest where the pair is on the same line (+3.9) or both ends are questions (+1.0). **What it does not show:** that BROS is better than LiLT in general (one recipe tuned for LiLT, one dataset); the size of the effect with a matched training session (A0 was trained in an earlier session); anything about languages, since BROS is English-only and XFUND is not (the gain appears despite that, so the advantage is probably the relative-position encoding, which is a guess, not tested here).

**Evidence:** `results/bros_joint/` (`J1_e2e.json`, `J2_e2e.json`, `J1/`, `J2/`, `run_status.json`).


---

## 38. The schema-guided grounded recipe on DocILE line items — GPU (Kaggle T4 x 2), registered 2026-11-22

**Question.** Study 34 found that on VRDU ad-buy line items (dense scanned tables, median 7 gold rows per page with rows, up to 47) the recipe scores 38.2 row F1 against 68.9 to 85.3 for a trained BROS tagger, and that cleaning its answers (repeated rows, runaway pages) only reaches 43.8. **Is that failure general to tables, or specific to the ad-buy scans?** DocILE pages are shorter (median 3 gold rows per page with rows, maximum 61), its oracle through the same decode path scores 99.98 row F1 and 99.79 cell F1 on the test documents, and its layouts are split so that no test layout is in any training set.

**Data and inputs.** DocILE `annotated-trainval` (5,680 documents; terms of use not yet read by us), the layout-disjoint test documents of `results/docile_rows/splits.json` (200 documents, 236 pages, 886 gold rows) as the only scored set, and the 30 pilot documents (38 pages) that are in no test set. Pieces: OCR lines cut at 0.5 line heights (`docile_rows.Doc`). The 19 line-item fields have descriptions written from their names alone and frozen in `docile_sg.DESC`. Several cells of one field in a row are written as a list (26% of DocILE rows repeat a field).

**Method (same as study 34, `scripts/docile_sg.py`).** Qwen3-VL-8B-Instruct, 4-bit, greedy, one call per page, answer `{"rows": [...]}` with every cell citing a piece id plus its text, at most 2,400 new tokens, image budget 1,280 (at most 980 visual tokens), the row-position repair, the attention fix, an out-of-memory retry without the image, a page that failed is retried on a re-run, and only documents with every page answered are scored. Arm A2 (image + pieces) first; A1 (text only) only if A2 is informative. **Gate on the pilot (image arm):** at least 90% of answers parse, at least 90% of cited ids exist, and at least one row on at least 90% of the pages that have gold rows; if it fails the test run is skipped and the pilot is reviewed (no automatic fallback).

**Scoring.** Strict row F1 (DocILE's line-item rule: predicted rows matched one-to-one to gold rows for the most matching cells, a cell right only if its word set equals the gold word set) and cell F1; per-document counts are saved. The BROS baseline on the same test documents is study 39 (comparison with `scripts/docile_compare.py` once both have run); on its own the result is read against the oracle, the ad-buy recipe and the stated rule below.

**Registered reading (fixed before the run, on the A2 row F1 of the test documents):** 80 or more: the recipe is competitive on typical invoice tables and the ad-buy failure is specific to dense scans; 50 to 80: partial, the failure is partly general; below 50: tables are hard for the recipe in general. These thresholds are judgement calls, not tested rules, and a number inside a band says nothing about BROS.

**Limits stated up front.** One model, one prompt, no tuning on DocILE (the descriptions and the prompt were written before any DocILE page went through the model); the test set has a median of 2 gold rows per document, so row F1 rests on few rows per document; the oracle ceiling counts only gold cells that contain an OCR word (3.5% of DocILE line-item cells do not, so under the official every-cell rule the ceiling is lower); the answers contain document text and are not pushed to GitHub by default until the terms are read (scores and counts are); the run time on DocILE is not measured (estimate 2 to 3 h for the 236 pages on two T4s plus 0.3 h for the pilot).

**Pre-run checks done on CPU (2026-11-22).** The shared repair helpers moved to `scripts/row_repair.py` and both recipes were re-tested (ad-buy oracle 100.00 on 3 documents with the repair table unchanged, 83.72 with repair after shifted ids; DocILE oracle 99.98 / 99.79 on the 200 test documents); stub run, gate and scorer work, including a partly finished run (26 of 30 documents scored) and a re-run that retries failed pages (30 of 30). **Not checked:** the image path and the 8B model on DocILE pages (the notebook's smoke cell is the first test), and the PDF download inside Kaggle.

**Results (2026-11-22): the pilot gate failed, so the test run did not start; the registered reading (A2 row F1 on the 200 test documents) has no value.**
- **Gate (`results/docile_sg/run/gate.json`; 30 pilot documents, 38 pages, image arm; the pilot took 0.44 h):** parse rate 100% (pass); pages with at least one row, 30 of 33 pages with gold rows = 90.9% (pass, bar 90%); **cited ids that exist 88.28% (565 of 640 segments; fail, bar 90%)**. Decision recorded by the script: `stop_and_review`. Also: 621 cells, 177 (28.5%) not grounded in any OCR piece, 46 row-position repairs, 23 empty rows.
- **Pilot score (computed afterwards with `docile_sg.py score` on the pilot rows; not a registered cell, not the test documents, and the pilot documents were the gate's, so read it as a signal only):** strict row F1 **26.56**, cell F1 26.97; counts (matched, predicted, gold) rows 130 / 444 / 535. Precision about 29% and recall about 24%.
- **How it reads:** the pilot sits at the "below 50" end of the registered bands (tables hard for the recipe), and next to study 39 it is far below BROS at N = 200 and N = 1000 (52.9 and 59.6 on the test documents), but the pilot and the test sets differ, so no interval and no verdict are given. The registered test run (about 2 to 3 h, 236 pages) was not made; its only purpose would have been to confirm a gap of this size.
- **Not examined:** why 12% of cited ids do not exist (the gate says only that they do not), and the A1 (text-only) arm, which was conditional on A2 being informative.

**Evidence:** `results/docile_sg/run/` (`gate.json`, `rows_pilot_*.jsonl`, `run_status.json`); notebook `notebooks/kaggle_docile_sg.ipynb`. The pilot answers contain document text; the DocILE terms of use were read by the project owner on 2026-11-22 and pushing the results to GitHub was approved.


---

## 39. The BROS line-item tagger with row grouping on DocILE, layout-disjoint splits — GPU (Kaggle T4 x 2), registered 2026-11-22

**Question.** Study 38 runs the recipe on DocILE line items but has no trained baseline to be read against. How well does a trained Track-1 BROS tagger with the anchor row rule do on the same layout-disjoint test documents as a function of the number of training documents, and where does it cross the recipe? On ad-buy (study 34) BROS scored 68.9 / 83.2 / 83.8 / 85.7 row F1 at N = 10 / 50 / 100 / 200 against 38.2 for the recipe.

**Setup (`scripts/docile_bros.py`, `configs/docile_rows_bros.yaml`, `notebooks/kaggle_docile_bros.ipynb`).** Track-1 BROS, the recipe unchanged from `configs/vrdu_track1_bros.yaml` (20 epochs, windows of 384 words with overlap 128, coordinate jitter, content-box anchoring, learning rate 5e-5), one training seed (13); the 19 line-item fields as BIO labels, every run of consecutive words of a cell is a span (a cell on two lines is two spans, ceiling in the oracle below). Training sets: for split seed s in {0, 1, 2} the first N documents of the registered order of `results/docile_rows/splits.json` (nested in N), N in {10, 50, 200, 1000}; early stopping on the 100 validation layouts; prediction on the 200 test layouts (layouts and vendors disjoint from every training set). **12 runs.** Rows from the predicted spans by the anchor rule (`row_group.group_rows`, tolerance 0.5, chosen on ad-buy training documents only, not on DocILE). Scoring: strict row F1 (DocILE's line-item rule) and cell F1 (`lir_metric`), per-document counts saved; the comparison with the recipe is `scripts/docile_compare.py` (bootstrap over documents, each a different layout; for each N the BROS value is the mean over its three split seeds; rule: the recipe wins if the interval of recipe minus BROS is above 0, ties if it contains 0, BROS wins if below).

**Ceilings (CPU oracle, test documents).** Perfect tags decoded to spans: 97.83 cell F1 (1.0% of the cells are not consecutive words); with rows from the anchor rule: 94.43 row F1. A learned grouper did not beat the rule on DocILE (study 35).

**CPU checks done before any GPU run (2026-11-22).** The ad-buy oracle gives the same numbers after the shared row-grouping code moved to `scripts/row_group.py` (95.18 / 91.63 on all 641 documents, 94.89 / 90.01 on the primary set); the DocILE oracle gives 97.83 / 94.43 on the test documents and 97.11 / 95.65 on 200 training documents; `docile_bros.py score` on span dumps made from the gold tags reproduces 94.43 row F1 and 97.83 cell F1 on the 200 test documents; a real CPU training smoke (`train --smoke`: 6 training documents, 2 validation, 3 test, one epoch, BROS from the Hub) wrote the result files and `test_entities.jsonl`, and `score` read them (scores 0.0 for the untrained model); `docile_compare.py` was tested on made-up reports. **Not checked:** the GPU run, the time per run on DocILE, and the DocILE download inside Kaggle.

**Limits stated up front.** One training seed per run and three split seeds, one recipe not tuned for DocILE (learning rate, epochs, window size are those of VRDU), BROS has an English vocabulary and DocILE is English, the row rule was not tuned on DocILE, the test documents have a median of 2 gold rows each (886 rows in total), runs at N = 1000 are the least certain in time (a deadline of 11 h stops the notebook cleanly and finished runs are kept), and the predictions contain document text so they stay out of GitHub until the DocILE terms are read (scores and per-document counts are pushed).

**Results (2026-11-22; `results/docile_bros/run/report.json`; 12 runs, all finished, 2.16 h on two T4s, no failed shards; scored on the 200 test documents).**

| N training documents | Row F1, split seeds 0 / 1 / 2 | Mean row F1 | Mean cell F1 |
|---|---|---|---|
| 10 | 0.25 / 0.18 / 0.34 | 0.3 | 0.3 |
| 50 | 45.34 / 39.92 / 48.97 | 44.7 | 46.4 |
| 200 | 54.12 / 52.57 / 51.91 | 52.9 | 54.7 |
| 1000 | 60.40 / 58.90 / 59.38 | 59.6 | 61.8 |

- **N = 10 is a training failure, not a measurement of 10 documents.** Training loss was still about 0.9 to 1.1 and validation F1 0.00 when early stopping ended each run after 7 epochs (best epoch 1). With so few windows an epoch is a handful of steps. On ad-buy the same recipe gave 68.9 at N = 10. Treat the 0.3 as "did not train", and note that **N = 50 is probably also step-limited**: one run (split 2) used all 20 epochs and was still improving (best epoch 20), two others peaked at epochs 13 and 16.
- **From N = 200, the runs overfit the training layouts:** training loss 0.02 to 0.07 at the end, best validation epoch 4 to 11 of at most 14 run. The limit is generalisation to new layouts, not optimisation.
- **Scaling:** +8.2 from N = 50 to 200, +6.7 from 200 to 1000 (five times the data), so more data helps but slowly; the oracle ceiling with these tags and the anchor rule is 94.4 row F1, and N = 1000 is 34.8 below it. No extrapolation is made.
- **The comparison with the recipe (`docile_compare.py`) was not made:** study 38 has no test-set scores. Against the 30-document pilot only: recipe 26.6, BROS at N = 200 and N = 1000 about 53 and 60 on a different document set.

**Post hoc error analysis (2026-11-22, after the results; CPU, saved predictions only; `scripts/docile_bros_errors.py`, `results/docile_bros/errors.json`; N = 200 and 1000, three split seeds each).** Each oracle below changes one stage and leaves the rest as predicted, so the gaps show where the row F1 points go. Row F1 means, N = 200 / N = 1000: **reported 52.9 / 59.6**; with the true row of every predicted span (grouping oracle) 54.7 / 61.8; with predicted extents relabelled by the gold field where the extent is exact (typing oracle, then the anchor rule) 62.4 / 67.2. Span F1 against the gold spans, N = 200 / 1000: typed 55.1 / 62.6, field ignored 64.9 / 70.1.
- **Row grouping costs about 2 points** (the anchor rule is not the bottleneck; the grouping-oracle row F1 came out equal to the cell F1 in every run).
- **Where the gold spans of N = 1000 end up (mean of 3 runs, 3,113 gold spans per run):** exactly right 64.2%; no predicted span touches it 21.6%; right extent but wrong field 7.8%; right field, wrong extent 4.8%; wrong field and extent 1.6%. At N = 200: 53.7 / 30.0 / 9.6 / 4.7 / 1.9.
- **The field errors are mostly gross against net:** at N = 1000 the largest confusions (summed over 3 runs) are net amount predicted as gross amount 192, gross unit price predicted as gross amount 129, net unit price predicted as gross unit price 99. Recall of `amount_net` is 6.8% (78 gold spans per run) and of `code` 44.9%; description, quantity, unit price and amount are 65 to 77% recall with 55 to 66% precision (amount_gross: precision 54.8%, recall 77.1%).
- **False spans:** about 2,000 predicted spans per run touch no gold word (largest: amount_gross 676, quantity 472, description 406 at N = 1000); 66 to 72% of those sit inside the vertical extent of the gold table on their page (580 / 539 / 506 of 809 / 779 / 767 for the three runs), the rest outside it (totals, headers).
- **Misses are not whole-document failures:** at N = 1000, only 9 to 14 of the 200 test documents have under 25% of their gold spans touched (about 8 to 10% of the misses); 38 to 44 documents with 25 to 75% touched hold 59 to 74% of the misses. A look at three such documents showed three different causes: a column never tagged at all (a repeated code and a unit price on every row), OCR words that fuse two gold cells (`6431308/20/20` is a code and a date in one word), and a value column tagged as a different field in some rows. Three documents is an anecdote, not a count.
- **What this says:** the loss is in tagging (detection first, then the field of an amount), not in row grouping; giving every exactly-found span its true field would still reach only 67 at N = 1000, because about 22% of gold spans are never touched by a predicted span. **What it does not say:** whether more data, more epochs or a different input would close the gap (not tested), or whether gross against net is learnable from the page at all (only 8.5% of the test documents have more than 384 words and the median has 133, so the window is unlikely to hide the column headers for most of them; the cause is not identified).

**Published comparison (read 2026-11-22: DocILE benchmark paper arXiv 2302.05658, Table 3 and supplementary Table 3; CLEF 2023 extended overview, CEUR-WS Vol-3497 paper-049; USTC-iFLYTEK notebook, paper-052; the PDFs were extracted to text and read, the web fetch tool returned only raw bytes).** The official LIR F1 matches a predicted field to a gold field when the field type is the same and the same text is covered, after matching predicted to gold line items so that the number of matched fields is maximal; our `row F1` is a port of it (`lir_metric.py`), with word-set equality in place of pseudo-character coverage (the paper's NER upper bound is 96.1 against our oracle 94.4). The official test set is private and mixes layouts seen in training with unseen ones; ours is 200 layout-disjoint documents, which is closest to the official zero-shot subset.
- **Official LIR F1 on the full test set, trained on the DocILE annotated training data (size not checked):** RoBERTa-base 68.6 (precision 69.5, recall 67.8); RoBERTa with synthetic pre-training 69.8; LayoutLMv3-base 72.1 (pre-trained on IIT-CDIP, which the benchmark prohibits); LayoutLMv3 pre-trained on the 932k unlabeled DocILE documents 66.1; **best submission, GraphDoc (USTC-iFLYTEK; pre-training on the 932k unlabeled documents, a merger module, an ensemble): 75.93** (precision 80.82, recall 71.60); the organisers state that the best results on both tasks are below 80%.
- **By layout (RoBERTa-base baseline, LIR F1):** unseen layouts 59.8, layouts with 1 to 3 training documents 56.8, layouts with 4 or more 75.6. The overview says all methods are significantly worse on unseen layouts, and that GraphDoc stays first there (its zero-shot number is in a bar chart only; not read numerically).
- **Our BROS (layout-disjoint test, no pre-training on DocILE data):** 52.9 at N = 200 and 59.6 at N = 1000 (about 285 training layouts of the 562 in the pool).
- **Reading.** At N = 1000 the tagger is level with the published RoBERTa baseline on unseen layouts (59.6 against 59.8) while using at most 1,000 training documents and no unlabeled pre-training; it is well below the top published system (75.9 overall, a mix of seen and unseen layouts). So 60 is an ordinary result for unseen layouts, not a bad one and not a strong one. **The comparison is rough:** different test documents (their subset has its own size and mix; ours has a median of 2 rows per document), a slightly different matching of text, an unknown training-set size for the baseline.

**Evidence:** `results/docile_bros/run/` (`report.json`, `run_status.json`, per-run `result.json`, `history.json`, `test_entities.jsonl`), `results/docile_bros/errors.json`.


---

## 40. Two column rules on the saved BROS predictions for DocILE line items — CPU, registered 2026-11-22 (before any run of the rules)

**Question.** The error analysis of study 39 put the loss in tagging: at N = 1000, 21.6% of gold spans have no predicted span touching them, 7.8% are found with the wrong field (mostly net predicted as gross), and about 2,000 predicted spans per run touch no gold word (about two thirds of them inside the table's vertical extent, the rest above or below it). **Can two cheap, parameter-free rules that use the column structure of the page, applied to the predictions after the fact, raise row F1 without retraining?** If yes, they are a free addition to the table pipeline and say what a better tagger would need to learn; if no, the tagger itself has to change (the GPU plan).

**Data.** The saved predictions of the 12 study 39 runs: N = 1000 (primary) and N = 200 (secondary), split seeds 0, 1, 2, on the 200 layout-disjoint test documents (`results/docile_bros/run/N*_S*/seed_13/test_entities.jsonl`). N = 10 and N = 50 are not used (untrained and step-limited).

**Rules (fixed here; `scripts/docile_column_rules.py`, constants OVERLAP 0.5, MIN_COL 3, MIN_ANCHORS 3, none tuned; no other value is run).**
- **R1, column vote.** Per page, predicted spans whose horizontal extents overlap by at least 0.5 of the narrower span are chained into columns; a column of at least 3 spans gets the field most of its spans have (a tie changes nothing). Aim: consistent field per column.
- **R2, table extent.** Per page with at least 3 spans of the anchor field (the field the anchor rule uses), a span whose vertical centre is more than one median anchor pitch above the first anchor or below the last anchor is dropped. Aim: remove spans in headers, totals and addresses that the anchor rule currently attaches to the nearest row as false cells.
- **R1+R2** (R1 first) is reported, not tested.
- Rows are built with the anchor rule (`row_group.group_rows`, tolerance 0.5) and scored with the same strict row F1 and cell F1 as study 39. The rules never see gold.

**Registered rule (primary cell: row F1 at N = 1000, each of R1 and R2 against the reported predictions, same runs).** A rule is *adopted provisionally* if the mean difference over the three split seeds is at least +1.0, positive in all three seeds, the 95% interval of a paired bootstrap over the 200 test documents (2,000 resamples, the three runs averaged) is above 0, and the difference at N = 200 is positive in all three seeds. A rule that meets some of this is reported with the numbers and not adopted; a rule with mean difference at most -1.0 is called *harmful*. Two rules are tested without a multiplicity correction. Also reported, not tested: cell F1, the difference in the two halves of the test documents (even and odd position in the manifest), and for each rule the number of spans changed and how many exactly-right spans there were before and after.

**Stated expectation, before running.** R1 will not reach +1.0: the field errors are systematic (the model almost never predicts net amount, recall 6.8%), and a vote cannot restore a label that the whole column lacks. R2 may reach about +1 because the false spans outside the table add false cells; it can also hurt by dropping the wrapped lines of the last row. Not a prediction of any size beyond that.

**Limits stated up front.** The rules were written after the error analysis of the same 200 test documents, so the test set is not independent of them; an adopted rule is provisional until it is checked on predictions for documents the rules have not seen (the 100 validation layouts, whose predictions are not saved; the next GPU run should save them). Only three split seeds of one training seed per size; the test documents have a median of 2 gold rows each, so intervals over documents are wide; the 0.5 overlap and the one-pitch margin are judgement calls; chaining by overlap can merge two neighbouring columns on crowded pages (R1 then votes across both).

**Checks done before the run.** `tests/test_docile_column_rules.py` (a made-up page: the minority field in a column is relabelled, a tie is left alone, spans far above and below the table are dropped and a span half a pitch below the last row is kept) passes.

**Results (2026-11-22; `results/docile_column_rules/results.json`; one run of each rule, as registered).** Row F1, mean of the three split seeds; difference to the reported predictions per seed, 95% interval over documents:

| N | Arm | Row F1 | Difference (per seed) | Interval | Test halves (even / odd) |
|---|---|---|---|---|---|
| 1000 | reported | 59.56 | | | |
| 1000 | R1 column vote | 56.85 | **-2.71** (-3.15 / -3.65 / -1.33) | [-4.10, -1.31] | -2.75 / -2.66 |
| 1000 | R2 table extent | 59.57 | **+0.01** (-0.14 / +0.01 / +0.17) | [-0.17, +0.19] | -0.04 / +0.06 |
| 1000 | R1+R2 (not tested) | 56.85 | -2.71 | [-4.03, -1.37] | |
| 200 | reported | 52.87 | | | |
| 200 | R1 | 51.05 | -1.82 (-2.03 / -2.33 / -1.10) | [-3.84, -0.39] | -1.26 / -2.41 |
| 200 | R2 | 52.98 | +0.11 (+0.07 / +0.19 / +0.07) | [-0.06, +0.27] | +0.19 / +0.03 |

- **R1 is harmful** under the registered rule (mean difference at most -1.0, interval below 0, negative in every seed and at both sizes). Exactly-right spans fell from 5,996 to 5,717 at N = 1000 (summed over the three runs; 9,828 predicted spans, none added or removed): the vote changed more right labels to wrong than it fixed. The registered expectation (R1 cannot reach +1.0) holds, and it is worse than "no effect".
- **R2 changes nothing** (+0.01 at N = 1000, +0.11 at N = 200, intervals contain 0): it dropped 152 spans (1.5%) of which **66 were exactly right**, so it removed about as many right cells as false ones. The registered guess that it might reach +1 is not met. It means the spans above or below the anchor rows are not mostly false: many are real cells of rows the anchor field does not mark (for example a row whose anchor field was not predicted).
- **Neither rule is adopted.** The anchor rule and the plain predictions stay as they are.
- **What this says:** the lost points are not recoverable from column structure by a rule applied to these predictions. The wrong fields are not randomly scattered inside consistent columns (a vote makes it worse), and the false spans are not mostly outside the table. The tagger itself has to be better, which points at the training side (more layouts, a different loss, different inputs) and not at post-processing. **What it does not say:** that no column-aware method can help (a learned column typer, or a header-conditioned tagger, was not tested; the header rule of study 35 needs gold header types), or anything about parameter values other than the registered ones (none were tried; 0.5 and one pitch were fixed in advance).

**Evidence:** `results/docile_column_rules/results.json`; `scripts/docile_column_rules.py`; `tests/test_docile_column_rules.py`.


---

## 41. The BROS line-item tagger on the whole DocILE training pool — GPU (Kaggle T4 x 2), registered 2026-11-22

**Amendment 1 (2026-11-22, before any run of this study): the run also measures seen layouts (MTL).** 200 pool documents are held out of training, one from each of 200 pool layouts that have at least two documents (`docile_splits.py`, random seed 41, `results/docile_rows/mtl.json`; 259 such layouts exist). Training therefore uses **2,908 documents, still covering all 562 pool layouts** (a held-out document's layout always has another document in training), and the run is scored on two test sets: the 200 unseen-layout documents of study 39 (**UTL**, the primary cell, unchanged) and the 200 held-out documents of seen layouts (**MTL**). The MTL minus UTL gap is reported, not tested. The held-out documents are a different set from study 39's test documents, so the MTL number is not a re-score of anything earlier; its oracle ceiling (gold tags, anchor rule) is 91.37 row F1 and 94.14 cell F1, against 94.43 and 97.83 for the UTL set. The project's earlier MTL-UTL gaps (VRDU, matched training size) are not directly comparable: this gap is measured at the largest training size, with one held-out document per seen layout. Validation for early stopping stays on the 100 unseen-layout validation documents. Not checked: whether a seen-layout test set drawn this way is easier or harder than the earlier MTL folds. The text below describes the original design (3,108 training documents); read 2,908 wherever the pool size is used for training, and the time and disk estimates fall by 6%.

**Question.** In study 39, row F1 rose from 44.7 (N = 50, 46 layouts) to 52.9 (N = 200, about 110 layouts) to 59.6 (N = 1000, about 285 layouts): roughly +6 to +8 points per doubling of layouts, with 562 layouts in the pool. **Does it keep rising when the whole pool (3,108 documents, 562 layouts, about twice the layouts of N = 1000) is used, or does it flatten?** It also gives a full-data number to put beside the published baselines (RoBERTa 59.8 on unseen layouts; study 39's comparison).

**Setup (`scripts/docile_bros.py train --full-pool`, `notebooks/kaggle_docile_bros_full.ipynb`).** The recipe of study 39 unchanged (`configs/docile_rows_bros.yaml`: 20 epochs at most, early stopping with patience 3 on the 100 validation layouts, lr 5e-5, windows of 384 words with overlap 128, jitter, anchoring), trained on every document of the pool (`results/docile_rows/pool.json`, written by `docile_splits.py`; 3,108 documents, disjoint from the validation, test and pilot documents; `splits.json` is byte-identical after regeneration), **two training seeds, 13 and 21** (the pool is the same, so the seeds measure initialisation and ordering only), prediction on the same 200 test layouts, rows by the anchor rule, scored as in study 39 (`docile_bros.py score`). One process per seed on its own T4; the notebook can also be run as two notebooks, one per seed (`TRAIN_SEEDS`).

**Registered reading (primary cell: row F1 of each seed against the study 39 N = 1000 mean, 59.56).** *Keeps rising* if both seeds are at least +3.0 above it; *flattening* if both are below +1.5 above it; otherwise *mixed*. The reference is a mean over three different 1,000-document sets and the two new seeds share one document set, so the comparison carries both kinds of variation unequally; the thresholds are judgement calls, not a test. Reported, not tested: cell F1, the error decomposition of study 39 (`docile_bros_errors.py --sizes full`), per-epoch validation F1.

**Budget and failure modes (checked 2026-11-22).**
- **Time.** From study 39 (12 runs, 37,560 training-document-epochs, 2.16 h on two T4s) about 0.4 s per training document per epoch including validation, an upper-side figure because the shards were not perfectly balanced. Whole pool: about 2.8 h per run at 8 epochs (the earlier runs stopped between epochs 7 and 14 for N >= 200 and training loss was already about 0.03), at most about 7 h at the 20-epoch limit; both seeds in parallel, so about 3 h expected and 7 h worst case, inside the 12 h session limit. **A run cannot resume part-way** (`train_single_seed` resumes only finished runs); a killed session loses that seed's training. The per-epoch validation F1 in the notebook log is the only trace of an unfinished run.
- **Disk.** The DocILE download is 1.7 GB (a zip plus its extraction, about 3.5 GB at the peak); no model weights are written (`save_weights: false`; the best state is held in CPU memory, about 0.5 GB per process); outputs are about 0.35 MB per seed. Far below the 20 GB working-directory limit.
- **Memory.** GPU memory per process is set by the batch (4 windows of at most 512 tokens), peak 7.9 GB at N = 1000, so it does not grow with N; CPU memory holds the tokenised pool in each process (a few hundred MB: median 133 words per document).
- **Stored (pushed):** `result.json` (test metrics, per-type scores, history), `history.json` (per-epoch train loss and validation F1), `best_meta.json`, `test_entities.jsonl` (the test predictions, needed for scoring and error analysis; pushing approved by the project owner after reading the DocILE terms), `run_status_<seeds>.json`, `report_<seeds>.json`. **Not stored:** weights, validation-document predictions (the trainer does not write them; the column rules of study 40 that would have needed them were not adopted).
- **Two notebooks.** Files are named by seed and suffix so two pushes do not touch the same path; the push retries up to three times after a rebase.

**Checks done on CPU (2026-11-22).** `docile_bros.py train --smoke --full-pool --train-seeds 13,21` trains one epoch on 6 pool documents and writes the four result files; `score` reads that directory (`N=full`); the original path (`--sizes 10 --seeds 0 --smoke`) still passes; the pool is 3,108 distinct documents, contains every study 39 training set and none of the validation, test and pilot documents; every notebook code cell compiles; the notebook's cell 2 repeats the pool checks on Kaggle. **Not checked:** the GPU run, the true time per epoch on 3,108 documents, and the Kaggle download inside the new notebook (the same cell as study 39, which worked).

**Limits stated up front.** One recipe tuned for neither DocILE nor the pool size; two seeds on one document set; test documents have a median of 2 gold rows (886 rows in total); the study 39 reference runs may have had under-trained smaller sizes, but N = 1000 stopped on validation, not on the epoch limit.

**Results (2026-11-22; two sessions, one training seed each; `results/docile_bros_full/run/report_{13,21}.json`, `report_mtl_{13,21}.json`; no failed shard).** Training set 2,908 documents (562 layouts); both test sets of 200 documents; strict row F1 (cell F1 in brackets).

| Training seed | Epochs run (best) | Training hours | UTL row F1 | MTL row F1 | MTL minus UTL |
|---|---|---|---|---|---|
| 13 | 11 (10) | 2.30 | 59.84 (61.56) | 68.87 (71.75) | +9.03 |
| 21 | 5 (2) | 1.16 | 60.96 (62.82) | 63.02 (65.39) | +2.06 |
| mean | | | **60.40** | **65.95** | +5.55 |

- **Registered reading (UTL against the study 39 N = 1000 mean, 59.56): flattening.** The differences are +0.28 (seed 13) and +1.40 (seed 21), both below +1.5. Three times the training documents and about twice the layouts (562 against about 285) bought at most 1.4 row F1 on unseen layouts. For scale: N = 50 to 200 gave +8.2, N = 200 to 1000 gave +6.7. The reference is a mean of three different 1,000-document sets and the new seeds share one document set, so this comparison carries both kinds of variation unequally; two seeds only.
- **The MTL number moves with the seed, the UTL number does not.** Seed 13 trained 11 epochs (validation F1 still creeping up to 0.704) and gained on seen layouts (68.87) but not on unseen ones (59.84); seed 21 stopped after 5 epochs (best epoch 2, validation 0.677) and sits at 63.02 / 60.96. A longer-trained model is better on layouts it has seen and no better on layouts it has not, which is what memorising layout conventions would look like. With two seeds and early stopping decided on 100 validation layouts, the size of the gap (+9.0 against +2.1) is not pinned down; the sign (seen at least as good as unseen) holds in both. For comparison the VRDU gap with BROS Track-1 is 21 points.
- **Error decomposition (`scripts/docile_bros_errors.py`, with `--mtl` for seen layouts; `errors_{utl,mtl}_all.json`, both seeds).** Gold spans exactly right, unseen / seen: seed 13 60.6% / 74.3%, seed 21 66.8% / 71.1%. Never touched by a predicted span: seed 13 24.0% / 15.4%, seed 21 18.5% / 13.9%. **Right extent, wrong field: seed 13 9.1% / 4.0%, seed 21 8.8% / 6.9%.** Typed span F1 (unseen / seen): 62.6 / 74.7 and 63.6 / 68.5. The typing oracle (true field on exactly-found spans) gives row F1 69.07 / 72.47 (seed 13) and 69.29 / 69.34 (seed 21).
- **Correction to an earlier remark.** After seed 21 alone I wrote that the net-versus-gross confusion is as large on seen layouts as on unseen ones. That holds for seed 21 (8.8% against 6.9% wrong-field) but not for seed 13 (9.1% against 4.0%): a model trained longer types the monetary fields better on layouts it has seen. The layout-independence of the label (`scripts/docile_netgross_layouts.py`: 63% of net-only documents sit in layouts that also have gross-only documents) is unchanged; what changed is that the seen-layout error rate depends on the training run.
- **What this shows:** on DocILE line items, more training layouts (to 562 of the pool) raised unseen-layout row F1 by at most 1.4 over N = 1000, and the seen-versus-unseen gap is between 2 and 9 points depending on the seed, far below the 21 points of VRDU registration forms. The loss is dominated by detection and monetary typing, not by template shift. **What it does not show:** a value for the gap that is stable (two seeds, early stopping decided on a small validation set), or what a fully different recipe (more epochs, other learning rate, pre-training) would do.

**Evidence:** `results/docile_bros_full/run/` (`report_*.json`, `run_status_*.json`, per-seed `result.json`, `history.json`, `test_entities.jsonl` for both test sets), `errors_utl_all.json`, `errors_mtl_all.json`.


---

## 42. Does the page image help a fine-tuned encoder across templates? LayoutLMv3 with and without pixels on VRDU registration forms (UTL and MTL) — GPU (Kaggle T4 x 2), registered 2026-11-22

**Question.** The project description (deliverable 2) asks for a representation that jointly captures textual, visual and layout information. The fine-tuned encoders so far (LiLT, BROS) see text and layout only; the systems that see pixels are vision-language models (the zero-shot recipe, 76.5 exact-span F1 on registration forms; a LoRA-tuned 3B model, 46.2 against 70.1 for BROS, study 29). LayoutLMv3 is a fine-tuned encoder that reads the page image next to the words and boxes; its wrapper existed (`src/models/token_clf.py`) but no VRDU or DocILE run was ever made. **Does adding the page image help on unseen templates (UTL), and does it change the seen-versus-unseen gap (MTL minus UTL)?**

**Amendment 1 (2026-11-22, before any run of this study): windows are cut at page boundaries in both arms (`window_by_page: true`, `chunk_vrdu_example_by_page`).** The original design gave each window the image of the page holding most of its words; measured on the 1,915 registration forms (1,777 with more than one page), 36.7% of the default windows (384 words, overlap 128) contain words from more than one page and 11.1% of word slots would have faced the wrong page's image, which would have biased `pages` against `blank`. With page-aligned windows (7,648 windows instead of 7,353, none crossing a page, every word covered, tested) each window has exactly one page and one image, and `blank` uses the same windows. The cost: a window no longer carries text from the previous page, which BROS Track-1 windows do (the comparison with BROS Track-1 gets one more difference). Read the 'window of words is paired with the image of the page that holds most of its words' sentence below as: the window's own page.

**Amendment 2 (2026-11-22, before any result): a third seed may be run as separate sessions** (`SEEDS = "21"` in the notebook; output in `results/vrdu_layoutlmv3_s21*`; one seed's 6 fold-and-regime tasks are shared by the two GPUs). It is reported as spread and reproducibility next to the registered pairs; **the registered rule below uses seeds 13 and 42 only** and is not changed by it.

**Amendment 3 (2026-11-22, before any result of this study was read): the primary cell is rerun as unseen templates only, folds 0 and 1, seeds 13 and 42** (`REGIMES = "UTL"`, `MAX_FOLDS = "2"` in the notebook; output in `results/vrdu_layoutlmv3*_utl2`). Reason: the first sessions ran the full 3-fold, both-regime protocol, which runs all MTL tasks before any UTL task, and after about 3 hours they were in the third MTL fold with about 1.3 to 1.5 hours per fold-and-seed pair (my estimate of 2 to 4 hours per arm was about half of what it needs); they could not have reached the unseen-template tasks, the primary cell, inside the time available. A run now pushes its finished runs every 20 minutes. **Consequences for the rule below:** the pairs are 2 folds x 2 seeds = 4; *pixels help* if the mean UTL difference is at least +2.0, positive in at least 3 of the 4 pairs and positive in both fold means; *pixels hurt* if it is at most -2.0 and negative in at least 3 of 4; otherwise no clear difference. Folds 0 and 1 hold out `Amendment` and `Dissemination_Report`; **the hardest template, `Short-Form` (fold 2), is not covered**, so the UTL mean is not comparable with the 3-fold BROS Track-1 value (71.27); the BROS Track-1 reference on the same two folds is **UTL 77.72** (83.38 and 72.07 per fold) and MTL 92.29. MTL, if any MTL fold of the first sessions finishes, is reported as a bonus and is not part of the rule.

**Amendment 4 (same day, before any result): the 8 tasks of the primary cell may be spread over notebooks** (`FOLDS = "0"` or `"1"` with `REGIMES = "UTL"` and `ARMS = "pages"` or `"blank"`: four notebooks, each running seeds 13 and 42 on its two GPUs; output in `results/vrdu_layoutlmv3[_blank]_utl_f0` and `_f1`). The tasks, seeds and folds are the same as in one notebook, so the rule and the reading are unchanged; the summary is assembled from the per-run `result.json` files. A new `--folds` option of `src/train.py` selects fold indices (checked: `--regimes UTL --folds 1 --seeds 13,42` lists exactly UTL fold 1 for both seeds).

**Arms (matched-size protocol of the headline benchmark: N = 200 training documents, 3 folds, MTL and UTL, training seeds 13 and 42; 12 runs per arm).**
- **`pages`** (`configs/vrdu_layoutlmv3.yaml`): LayoutLMv3-base; each window of words is paired with the image of the page that holds most of its words (the earliest page on a tie), rendered from the PDF at 100 dpi, resized to 224 x 224 and normalised (mean = std = 0.5), cached once per (document, page) (`src/data/page_images.py`).
- **`blank`** (`configs/vrdu_layoutlmv3_blank.yaml`): the same model, recipe and seeds with a constant image (zeros) for every window, i.e. the visual stream switched off. This is the control: `pages` minus `blank` isolates the pixels.
- **Recipe (identical in both arms; the Track-1 BROS recipe except the geometry options):** 20 epochs at most, early stopping on the validation documents, lr 5e-5, batch 4, windows of 384 words with overlap 128 and soft voting. **No coordinate jitter and no content-box anchoring**, because both move the boxes while the image stays put, so a box and its pixels would stop agreeing. Nothing tuned for LayoutLMv3.

**References (same documents, folds, N; `results/vrdu_track1_bros/metrics.json`).** BROS Track-1 (windows, jitter, anchoring; 3 seeds): MTL 92.36, UTL 71.27 (std 11.24), gap 21.1. Plain BROS without windows reached only 22.7 / 7.5 because the 512-token limit drops most words of multi-page documents, so it is not a reference. The comparison of either arm with BROS Track-1 changes the backbone and the geometry options together and is reported, not tested.

**Registered rule (primary cell: UTL F1, mean over folds and seeds, `pages` minus `blank`, paired by fold and seed, 6 pairs).** *Pixels help* if the mean difference is at least +2.0, positive in at least 5 of the 6 pairs and positive in all three fold means. *Pixels hurt* if it is at most -2.0 and negative in at least 5 of 6. Otherwise *no clear difference*. Reported, not tested: MTL (same quantities), the gap, the UTL fields `file_date`, `foreign_principle_name`, `signer_name` (the fields that failed for earlier models), per-fold values, both arms against BROS Track-1. No interval (6 pairs); the fold-to-fold spread of UTL is large (std 11 for BROS), which the pairing removes only partly.

**Plumbing checked on CPU before any GPU run (2026-11-22; `tests/test_layoutlmv3_plumbing.py`, real VRDU documents and the real `microsoft/layoutlmv3-base` weights, under transformers 5.17 and 4.57.6).** The page of a window (majority, earliest on a tie, 0 when there are no page ids); the image tensor (3 x 224 x 224, in [-1, 1], not constant, cached, identical for repeated windows); the blank control (zeros, equal for every window); the collator adds `pixel_values` next to `bbox` (boxes at most 1000); one forward and backward step gives a finite loss and a gradient in the patch embedding; the image changes the logits; a backbone that needs images refuses `image_mode: none`. `src/smoke_test.py` passes on the blank config, and the training loop `train_one` (windows, validation, early stopping, prediction with soft voting, `test_entities.jsonl`) ran one epoch on 4 real VRDU documents in both arms under transformers 4.57.6 and wrote the four result files (test F1 0.0, as expected after one epoch on 4 documents; only the chain is tested). The two configs differ only in `image_mode` and `output_dir`.

**Not checked.** transformers 4.51.3 (the Kaggle pin of the other notebooks; the CPU check under 5.x hid a tokenizer problem in study 37, so this is a real gap); GPU memory with 4 windows of up to 512 tokens plus 197 image patches (about 709 positions per window); the time per run (my guess is 2 to 4 h per arm on two T4s); whether LayoutLMv3 trains stably at lr 5e-5 (copied from BROS; a collapse to all-`O` would show as validation F1 near 0 in the first epochs of the log); that the VRDU PDFs are all present after the Kaggle clone (cell 2 of the notebook counts them against the documents).

**Limits stated up front.** One recipe, two training seeds, one corpus (registration forms; English); the image enters at 224 x 224, too coarse for small print, so a negative result says only that this input does not help, not that pixels never do; LayoutLMv3 and BROS were both pre-trained on IIT-CDIP (external, not data of the held-out templates, so the zero-same-template rule is respected); the BROS reference uses three seeds and different geometry options; the weights are CC-BY-NC-SA (declared in the report).

**Interim results: seen templates (MTL), from the logs of the first two sessions** (`results/LayoutLMv3/MTL_Pages.txt`, `MTL_Blank.txt`: the `-> Test F1` line of each finished task; micro entity F1, no per-field scores; copied by hand because those sessions push only at their end). Not part of the registered rule, which is about unseen templates.

| Arm | Fold 0 (seed 13 / 42) | Fold 1 (13 / 42) | Fold 2 (13 / 42) | Mean of the finished tasks |
|---|---|---|---|---|
| `pages` | 92.08 / 91.16 | 93.33 / 92.49 | 92.10 / 92.25 | **92.24** (6 tasks) |
| `blank` | 91.26 / 90.97 | 93.07 / 91.97 | 93.30 / 93.54 | **92.35** (6 tasks) |
| BROS Track-1 (3 seeds) | 92.21 | 92.37 | 92.50 | 92.36 |

- **Pixels, seen templates (the six paired tasks):** `pages` minus `blank` = +0.82, +0.19 (fold 0), +0.26, +0.52 (fold 1), **-1.20, -1.29 (fold 2)**; mean **-0.12**. The first four pairs alone suggested +0.45, and the two fold 2 tasks, which arrived last, reverse it: on seen templates the page image makes no difference (the spread between folds, up to 1.4 points, is larger than any arm effect). Not part of the registered rule.
- **LayoutLMv3 against BROS Track-1 on seen templates (three folds):** `pages` 92.24, `blank` 92.35, BROS 92.36: all three within 0.15 points. For seen templates the ceiling is about 92 whatever the input; the interest is in the unseen-template result, which is still running.
- **MTL-minus-UTL gap:** to be computed from the unseen-template runs (folds 0 and 1 are being run; fold 2, `Short-Form`, is not, so the gap is compared fold by fold with these MTL values).

**Results (unseen templates, the registered cell):** *pending*

**Evidence:** `results/vrdu_layoutlmv3*/`, `results/vrdu_layoutlmv3_summary_*.json` (after the runs); `results/LayoutLMv3/` (MTL logs); notebook `notebooks/kaggle_layoutlmv3.ipynb`.


---

## 43. Detector and typer on DocILE line items: BROS finds the monetary cells, a VLM says which field each one is — GPU (Kaggle T4 x 2), registered 2026-11-22

**Question.** The error analysis of studies 39 and 41 put the loss in tagging. Among the cells BROS finds with the right extent, the field is wrong for 8 to 9% of the gold spans, mostly the monetary fields (net amount predicted as gross, unit price as amount, net unit price as gross); this is the largest field error on unseen layouts and on seen layouts alike. Net-only documents sit in layouts that also have gross-only documents (266 of 421 net-only documents; 58 of 241 layouts with at least two such documents are mixed; `scripts/docile_netgross_layouts.py`), so the label depends on the document (whether its prices carry tax), not on the layout. That is a reading question, a VLM's kind of task; the BROS windows see the page words and boxes but were never asked to reason about totals and tax columns. **Does a VLM that is told where the monetary cells are, and sees the page, type them better than BROS does?**

**Method (`scripts/docile_typer.py`, `notebooks/kaggle_docile_typer.ipynb`).**
- **Detector:** a saved BROS run on the 200 unseen-layout test documents (`test_entities.jsonl`): `Nfull_T21` (study 41, seed 21, 2,908 training documents) and `N1000_S0` (study 39). Everything BROS predicted stays, except the field of the **monetary candidates**: predicted spans of `line_item_unit_price_gross`, `line_item_unit_price_net`, `line_item_amount_gross`, `line_item_amount_net`.
- **Typer:** Qwen3-VL-8B-Instruct, 4-bit, greedy, one call per page with at least one candidate: the page image, the page's OCR lines with their identifiers (`docile_rows.Doc` pieces), the four field descriptions of `docile_sg.DESC` (written from the names alone, before any DocILE page was read), and the numbered candidates with the line identifier each sits on; **the BROS field is not shown**. The answer gives each candidate one of the four fields. An unanswered page, a failed call or an invalid field keeps the BROS field.
- **Score:** rows by the anchor rule (`row_group.group_rows`, 0.5) on the retyped spans; strict row F1 on the 200 documents, same as study 39.
- **Upper bound (computed before the run, on CPU with gold fields):** giving every candidate whose extent is exactly a gold span its true field raises row F1 from 60.40 to 66.38 (`N1000_S0`, +5.98) and from 60.96 to 66.82 (`Nfull_T21`, +5.86). BROS's own typing accuracy on those exactly-found candidates is 77.1% (n = 860) and 76.6% (n = 847). A rough exchange rate from these two numbers: about 0.27 row F1 per point of typing accuracy, so +1.0 row F1 needs about +3.7 points of typing accuracy over BROS.

**Registered rule (primary cell: row F1, typer minus BROS, over the tags run, paired by tag).** *Typer helps* if the mean difference is at least +1.0, positive for every tag, and the 95% interval of a paired bootstrap over the 200 documents (tags averaged, 2,000 resamples) is above 0. *Typer hurts* if the mean difference is at most -1.0. Otherwise *no clear difference*. Reported, not tested: typing accuracy of the VLM and of BROS on the exactly-found candidates, how many BROS fields the VLM changes, pages answered, the result for each tag.

**Expectation, before running.** Moderate to low. The VLM has to beat BROS's 77% on a task whose gold labels may follow annotator habit as much as tax logic; if the net/gross label is inconsistent even for a human, the typer cannot help. What a positive result would mean: the field error is a document-level reading problem and a detector-plus-reader split is the right design for line items; what a negative result would mean: the labels are not recoverable from the page by this reader.

**Checks done on CPU before any GPU run (2026-11-22).** Unit tests (`tests/test_docile_typer.py`: candidate numbering, retyping, parsing of valid, unknown and cut-off answers); the whole chain with a stub typer that echoes the BROS fields reproduces the BROS scores exactly (60.40 and 60.96, difference 0.00); a stub that answers the gold field gives +5.73 and +5.52 (close to the upper bound; the rest is candidates whose gold field is not one of the four monetary fields, which the answer format cannot express). **Not checked:** the real model on these prompts (the notebook's cell 3 runs it on 3 pages), the time per page (my estimate is about 20 s, 197 pages per tag, two GPUs: 1 to 1.5 h for two tags), and the prompt's token length on the longest pages.

**Limits stated up front.** One model and one prompt, no tuning; the candidates are BROS's, so the 18 to 22% of gold spans BROS never touches are out of reach; only the four monetary fields are retyped (about two thirds of the wrong-field errors); two detector runs (a second session can add `N1000_S1` and `N1000_S2`); the test documents have a median of 2 gold rows each; the typer sees the document text as OCR and the image at the 1,280 visual-token budget; a candidate that is not a gold cell at all cannot be dropped by the typer.

**Results (2026-11-22; one session, tags `Nfull_T21` and `N1000_S0`; 394 pages, 0 errors, 0 out-of-memory retries, 0.55 h on two T4s, median 5.9 s per page, median prompt 1,312 tokens, longest 7,830, peak 8.4 GB; `results/docile_typer/scores_Nfull_T21_N1000_S0.json`).**

| Tag | BROS row F1 | Typer row F1 | Difference | Upper bound | Typing accuracy on exactly-found monetary spans, BROS / VLM | Fields changed by the VLM | Candidates answered |
|---|---|---|---|---|---|---|---|
| `Nfull_T21` | 60.96 | 52.93 | **-8.03** | 66.82 | 76.6% / **45.9%** (n = 847) | 381 | 1,081 of 1,163 |
| `N1000_S0` | 60.40 | 53.54 | **-6.86** | 66.38 | 77.1% / **51.9%** (n = 860) | 371 | 1,112 of 1,225 |

**Registered rule: the typer hurts.** The mean difference is **-7.45**, 95% interval over documents [-10.03, -5.11], negative for both tags and far beyond the -1.0 line. The expected payoff (upper bound about +5.9) was not approached: the VLM labels fewer candidates correctly than BROS does, by 25 to 30 points.

**Post hoc diagnosis (`scripts/docile_typer_diagnose.py`, `results/docile_typer/diagnosis.json`; both tags pooled, 1,524 exactly-found candidates with a valid answer; not registered).**
- **Agreement with BROS:** both right 655; BROS right and VLM wrong 575; BROS wrong and VLM right 98; both wrong 196. The VLM fixes a third of BROS's errors (98 of 294) and breaks 47% of what BROS had right (575 of 1,230).
- **By gold field (BROS / VLM accuracy):** `amount_gross` 98.3% / 61.1% (n = 889), `unit_price_gross` 84.5% / 36.7% (n = 401), `amount_net` 4.9% / 16.0% (n = 144), `unit_price_net` 11.1% / 44.4% (n = 90). The VLM is better on the rare net fields and much worse on the common gross ones. It calls 249 gold gross amounts net (28% of them) and 215 gold gross unit prices net (54%); over all candidates (some match no gold span) its answers are 534 amount_net and 406 unit_price_net, against 234 gold net spans among the exactly-found ones, so it over-uses net.
- **By document kind:** on net-only documents (234 candidates) BROS is right 7.7% of the time and the VLM 26.9% (3.5 times better); on gross-only documents (1,256) BROS 95.1% and the VLM 52.9%. **There is signal in the VLM for net documents, but it cannot tell from the page which kind of document it is reading well enough to avoid damaging the gross majority** (gross-only documents are about 85% of the monetary candidates).
- **What this shows:** a VLM given the page and the candidates does not type DocILE's monetary fields better than the tagger, and replacing the tagger's field with the VLM's costs 7 row F1 points. The label is not recovered by reading the page with this prompt: the upper bound of +5.9 stays an upper bound. **What it does not show:** that no reader can do it. The one lead is document-level: the VLM does better exactly where BROS fails (net-only documents). A different design (one question per document about whether the prices include tax, then a fixed rule for the fields; or the BROS field shown to the VLM as a prior) was **not** tested and would be a new, post hoc study; one prompt, one model, one arm were registered and run.
- **Against the registered expectation (moderate to low prior):** the expectation was that the typer would not beat BROS's 77%, and it did not; it was worse than 'no clear difference', not better.

**Evidence:** `results/docile_typer/` (`run/rows_*.jsonl` field-name answers, `scores_*.json`, `diagnosis.json`).


---

## 44. Failure taxonomy of the BROS line-item tagger on DocILE (unseen and seen layouts) — CPU, registered 2026-11-22 (before the script was run)

**Question (deliverable 5 of the project: which layout, structural, textual or spatial assumptions fail and why).** Studies 39 and 41 gave the totals (row F1 about 60 on unseen layouts, a seen-versus-unseen gap of 2 to 9 points) and a first decomposition (22% of gold spans never touched, 8 to 9% with the wrong field, 5% with the wrong extent). **Which kinds of errors make up the loss, which documents carry it, and which of it is template-dependent (shrinks on seen layouts) and which is intrinsic?**

**Data (saved predictions, no model run).** Unseen-layout test documents (200): `N1000_S0`, `N1000_S1`, `N1000_S2` (study 39) and `Nfull_T13`, `Nfull_T21` (study 41); seen-layout test documents (200): `Nfull_T13`, `Nfull_T21`. Seven jobs run in parallel (`scripts/docile_failure_taxonomy.py`, one process each).

**Taxonomy (fixed here).** Every gold span gets exactly one outcome: *right*; *wrong field, same extent* (monetary family: unit price and amount, gross and net; or other); *right field, wrong extent* (predicted span inside the gold span, around it, or other overlap); *wrong field and extent*; *missed* (no predicted span touches it), split into field absent in the whole document, gold cell made of non-consecutive words, other. Every predicted span that touches no gold word is a *false span*, inside or outside the vertical extent of the gold table. Reported as a share of gold spans (false spans: of predicted spans), per run and mean; per gold field; and for unseen against seen layouts (the categories that shrink are the template-dependent part).
**Document properties (fixed here; from the annotation metadata and the gold, diagnostic only: a deployed system would not have them).** table border type, table structure, document type, source, language, pages, gold rows, table columns, fields used, words, amount columns used (net, gross, both, neither), share of multi-line cells, whether a row repeats a field, missing columns. Row F1 by property (documents pooled over the runs, 95% bootstrap over the documents of the group, only for groups of five or more documents) and Spearman correlations of the per-document row F1 with the numeric properties.

**What is and is not tested.** Descriptive: no registered threshold, no correction for the many groups and correlations; an interval or correlation is a pointer for the report, not a finding. **Expectation, before looking:** monetary typing and fields absent from a whole document dominate the loss; borderless tables, many columns and many rows go with low row F1; seen layouts reduce typing and detection errors more than boundary errors. If the data disagree, that goes in the report as it stands.

**Limits stated up front.** Five runs share the same 200 test documents, so run-level means are not independent samples; properties are correlated with each other (many rows and many columns, a scanned source and a table structure) and a group difference is not a cause; groups are small (200 documents, 886 gold rows); the seen-layout documents are a different set from the unseen-layout ones, so the shrinkage is read from the two full-pool runs only.

**Results (2026-11-22; seven jobs in parallel; `results/docile_taxonomy/taxonomy.json`, `per_document.jsonl`, `miss_position.json`; one follow-up script `scripts/docile_miss_position.py` was added after the first output).**

**Loss budget, share of the 3,113 gold spans per run (unseen layouts: mean of the five runs; the seen-layout column and its unseen comparison use the same two full-pool runs).**

| Outcome | Unseen, 5 runs | Unseen, 2 full runs | Seen, 2 full runs | Seen minus unseen |
|---|---|---|---|---|
| right | 64.0 | 63.7 | 72.7 | +9.0 |
| wrong field, same extent: monetary family | 5.9 | 6.4 | 4.3 | -2.1 |
| wrong field, same extent: other | 2.4 | 2.5 | 1.2 | -1.3 |
| right field, extent too short | 3.4 | 3.8 | 5.2 | +1.4 |
| right field, extent too long | 1.4 | 1.0 | 1.0 | 0.0 |
| wrong field and extent | 1.5 | 1.3 | 1.0 | -0.3 |
| missed: field absent in the whole document | 6.9 | 7.6 | 3.3 | -4.3 |
| missed: gold cell of non-consecutive words | 1.0 | 0.9 | 1.7 | +0.8 |
| missed: other (a partly tagged field) | 13.6 | 12.7 | 9.6 | -3.1 |
| *false spans, % of predicted: inside the table's extent* | 15.5 | 13.9 | 7.2 | -6.7 |
| *false spans, % of predicted: outside it (headers, totals, addresses)* | 7.1 | 6.6 | 7.0 | +0.4 |

- **The largest single bucket is not typing: it is spans that BROS never touches although the field is predicted elsewhere in the document ('missed, other', 13.6%)**, then monetary typing (5.9%) and fields never predicted in a whole document (6.9%). In total 21.4% of gold spans are missed, 8.2% have the wrong field, 4.8% the wrong extent, 1.5% both.
- **Seen against unseen layouts (the same two runs), what shrinks is template-dependent:** fields absent from the document (-4.3), partly tagged fields (-3.1), monetary typing (-2.1), other typing (-1.3), and false spans inside the table (13.9% to 7.2% of predictions). **What does not shrink is intrinsic:** false spans outside the table (6.6% against 7.0%), boundary errors (too-short extents and split cells go **up**, +1.4 and +0.8, because cells that were missed are now found, at the wrong extent), and about 27% of gold spans still wrong or missed on layouts the model has seen.
- **By field (unseen, share of the field's gold spans):** `amount_net` 9.7% right (80.8% given another monetary field); `unit_price_net` 17.7% right (72.1% another monetary field); `unit_price_gross` 54.9% right (22.5% another monetary field; 10.6% on seen layouts); `code` 48.4% right (37.8% never touched in the follow-up); `units_of_measure` 39.5% (27.2% field absent in the document); `currency` 50.5% (29.5% field absent); `description` 64.3% right with 13.0% wrong boundaries (13.9% too short on seen layouts); `amount_gross` 76.5%, `quantity` 72.8%, `date` 82.4% (their errors are almost all 'missed, other').
- **Row F1 by document property (unseen, the five runs pooled per document; 95% bootstrap over documents; groups of five or more).** Table structure: `normal` 62.2 [58.5, 66.7] (171 documents) against `structured_rows` 47.9 [37.0, 59.8] (28). Amount columns used: gross 63.5 [58.9, 69.2] (134), neither 51.9 [38.7, 61.2] (28), **net only 43.9 [35.2, 52.9] (38)**. Fields used: 4 to 5 fields 64.1, 6 or more 50.4 [38.9, 55.1] (18). Gold rows: 1 to 2 rows 53.8, 7 to 15 rows 65.2 [57.4, 71.7]. Multi-line cells: none 66.5, 30% or more 57.7. Source: UCSF 61.2 (176), PIF 53.7 [44.1, 65.4] (24). **No association with the table border type** (all borders 58.7, column borders 61.5, no borders 60.0, row borders 60.7) or with repeated fields in a row (59.4 and 61.0); all documents are English. Spearman correlations of the per-document F1 with numeric properties are weak: words -0.165 (p = 0.02), pages -0.139 (p = 0.049), rows, fields, table columns and multi-line share all within +-0.08. No correction for the many comparisons.
- **Where the misses fall (follow-up script, share of gold spans untouched by any predicted span, five runs pooled):** by field: `currency` 44.7%, `units_of_measure` 40.8%, `code` 37.8%, `quantity` 22.7%, `description` 19.6%, `amount_gross` 18.5%, `unit_price_gross` 17.6%, `date` 15.7%, `position` 13.7%, `amount_net` 9.4%, `unit_price_net` 4.2% (the net fields' spans are mostly touched, by a span with another field). By row position: first row 17.0%, only row 16.6%, middle rows 22.9%, last row 23.6%. Multi-line rows 25.6% against 20.4%; tables of 7 or more rows 23.9% against 17.1% for 1 to 2 rows. The misses are spread over the table, not concentrated at its ends; rare and secondary fields are the ones most often never touched.
- **Against the registered expectation:** the expectation that monetary typing and absent fields dominate was too narrow (partly tagged fields are the largest bucket), that borderless tables fail more was **not** supported (no association), that many columns and many rows go with low F1 was weak or reversed (more rows went with higher F1), and that seen layouts reduce typing and detection errors more than boundary errors was supported.
- **What this says about the assumptions that limit generalisation (deliverable 5), as far as these runs show:** (1) a field's label is not a function of the layout (gross against net; the same layout labels the amounts either way), so a tagger that learns layout conventions types them better on seen layouts and not on unseen ones; (2) 'every table has this field' fails for the secondary fields (currency, units, code), which are the most often absent; (3) the unseen-layout loss is mostly omissions (21%) and not wrong labels (8%), so a post hoc relabelling can recover little (study 40, and the upper bound of study 43); (4) about 7% of predicted spans, outside the table, are structural false positives that seen layouts do not remove either (7.0% on them). **What it does not say:** causes. The properties are correlated, the groups are small, and nothing here is an intervention.

**Evidence:** `results/docile_taxonomy/taxonomy.json`, `results/docile_taxonomy/per_document.jsonl`, `results/docile_taxonomy/miss_position.json`; `scripts/docile_failure_taxonomy.py`, `scripts/docile_miss_position.py`.
