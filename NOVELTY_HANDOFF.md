# Novelty Assessment Handoff: what we did after the intensive assessment, and how to defend it

> **Who this is for:** anyone standing at the poster (Param, Arjoe). After reading it you should not be blank on any line of the poster or the 2-page report.
> **Structure:** copied from `intensive_assessment/ENDTERM_PREP.md` (verdict, story, numbers, weak points, Q&A bank, claims, landmines, delivery).
> **Source of truth for every number:** `STUDIES.md` (one entry per study, with the registered rule written before the run). Numbers below were re-read from it and from `results/` on 2026-11-25. If a number here and `STUDIES.md` disagree, `STUDIES.md` wins; tell the other person.
> **Branch:** `main` (the feature branches were merged on 2026-11-22; the poster QR points to the repository, default branch `main`).
> **Study numbers are internal.** The poster and report deliberately do not show them. In conversation say "the fresh-forms run", "the FATURA sweep", etc.; use the numbers below only to look things up.

---

## 0. The verdict, before you read anything else

1. **The one-sentence result.** When templates are new and few, a zero-shot vision-language reader that must *cite OCR lines* beats a trained tagger (BROS) on wording-dependent fields (76.5 vs 70.5 F1 on VRDU forms, replicated on 300 untouched forms at 76.5 vs 70.0). Once about 10 templates have been seen, or on tables, the trained tagger wins. A per-field router (79.2) and an abstaining vote (82.4 on the fresh set) combine the two. For key-value *linking*, relative layout in the encoder (BROS) helps (47.4 to 53.4) and a multilingual text stream hurts (about 3 points).
2. **What is solid:** the per-template matched protocol (N = 200, three folds, no leakage), the prompt-v2 gain (+16.0 / +17.0 on two sets, +6.5 over BROS on a third set, all with intervals), the FATURA crossover (K = 1, 2 reader wins; K = 5 tie; K = 10, 40 BROS wins), the ad-buy and DocILE table results, the BROS-vs-LiLT linking result (all three seeds positive).
3. **What is soft (say so before they do):** the reader's prompt and three field descriptions were written *after* looking at errors on the confirm set (replicated on dev and on a fresh set, but all three sets are the same three VRDU templates); the router needs "no template significantly worse" and does not meet it (Amendment, -4.6); pixels help LayoutLMv3 by +0.8 over three unseen folds with a 25-point seed swing; only BROS has two seeds per number and UTL has only three folds.
4. **What we did not do:** open-schema decoding (the reader always got our field list), field names it has never been shown, other languages/domains for the reader, other VLM families, a learned router, an unlabeled-data method (forbidden by the zero-same-template rule).
5. **Do not quote:** "65.5 link F1" (a mistake from our side; see Landmines), "60.5" for Study 25's prompt as if it were registered (registered is 60.2), any DocILE comparison against GraphDoc 75.9 without the pre-training caveat.

---

## 1. The 90-second story (every answer should hang off this arc)

**Problem.** Form models memorise where fields sit. On forms from a template they never saw, our best trained model (BROS with sliding windows, jitter, anchoring: "Track 1") drops from 92.4 to 71.3 F1 (a 21-point gap). The loss is two fields: `foreign_principle_name` (0 to 13 F1 for every trained model, because each template words it differently) and the `registrant_name` on Short-Form (95% of BROS's errors there are the filer block, a decoy).

**Idea 1: read, do not train (wording).** A trained tagger cannot learn which text a *field name* refers to from 200 documents; a language-vision model can read the name. So: give a VLM the schema and the page's OCR lines (each with an id), make it answer by *citing a line*, and verify the cited text against the line (grounding). That is the "grounded zero-shot reader". It reached 76.5 vs 70.5 (BROS) on unseen templates.

**Idea 2: the advantage reverses with training templates.** On FATURA invoices (50 templates) we trained BROS on K = 1, 2, 5, 10, 40 templates: 32.5, 69.2, 90.8, 95.9, 99.9. The reader scores 91.7 flat. Reader wins at K = 1, 2; tie at 5; BROS wins at 10, 40. On tables (ad-buy line items, DocILE) the reader loses at every size.

**Idea 3: combine.** The two fail on different fields (reader: wording; BROS: boundaries, dates, signatures). A per-field router fitted on *other* templates gives 79.2; a three-way abstaining vote with BROS gives 82.4 on the fresh set.

**Tables.** A table row is a link between cells: detect cells with the same BROS tagger, group them into rows by an anchor rule. DocILE unseen layouts: 44.7 / 52.9 / 59.6 row F1 at 50 / 200 / 1,000 training documents, 60.4 on the whole pool, level with the published RoBERTa baseline on unseen layouts (59.8). Seen-unseen gap 2 to 9 points (vs 21 on forms).

**Pixels.** LayoutLMv3 with the page image vs with a blank image: no effect on seen templates (-0.1), +3.4 on two unseen folds, +0.8 over three (a 25-point seed swing on Short-Form). Both are about 14 points below BROS Track 1. In the VLM reader the image adds +3 to +4 (consistently).

**Linking.** End-to-end (the model finds the entities itself): LiLT 47.4; multilingual LiLT 44.2; link-preserving crops 49.7; BROS joint 53.4 (+5.9, all three seeds positive).

**Message.** *Whether to read or train depends on how many templates you have seen and on whether the target is a field or a table.*

### 1a. Three-minute spoken version (poster order)

"Form readers lose 21 F1 points on templates they have not seen; we measured it with whole templates held out. Two fields cause it. We tried a vision-language model that reads the field names, must cite the OCR line it copied from, and is checked against that line. It beats our trained tagger by about six points on unseen templates, and we repeated that on 300 forms nobody had touched. But train the tagger on ten templates and it wins, as our invoice sweep shows; on tables it wins at every size. The two make different mistakes, so a vote between them is the best system we measured, 82 F1. For linking questions to answers, the thing that helped was relative layout in the encoder, not a multilingual text stream. Pixels help a little and unreliably."

---

## 2. Poster map: every claim, where it comes from, how sure we are

| Poster text | Number(s) | Source (`STUDIES.md`; internal number in brackets) | Confidence |
|---|---|---|---|
| "lose 21 F1 points on unseen templates" | 92.4 to 71.3, gap 21.1 | Track-1 BROS benchmark `results/vrdu_track1_bros/metrics.json` [P] | High (3 seeds; UTL std 11.2) |
| Abstract "76.5 vs 70.5" | confirm set, 300 docs | Prompt v2 on confirm [31] | High (replicated, below) |
| "The tagger wins again once it has seen 10 templates, and on tables" | K = 10: -4.2 [-6.5, -1.9]; ad-buy reader 38.2 vs 68.9-85.7 | FATURA sweep [33], ad-buy [34] | High on FATURA (template bootstrap); medium on tables (ad-buy: 143 documents, document bootstrap) |
| "End-to-end linking improves with layout geometry, not multilingual text" | 47.4 to 53.4; 47.7 to 44.2 | [37], [24] | High (3 seeds each) |
| Table "Starting point" | LiLT 91.6/64.2/27.3; Delaunay graph 86.0/42.1/43.9; BROS template-adversarial 91.8/69.6/22.2; BROS Track 1 92.4/71.3/21.1 | prior benchmarks [P] | High; this is the intensive-phase work |
| Footnote "BROS Track-1 UTL by fold 83.4 / 72.1 / 58.4" | Amendment / Dissemination / Short-Form | `results/vrdu_track1_bros/metrics.json` | High |
| "Two fields cause most of the loss ... 0-13 F1 ... 95% filer block" | | error analysis and BROS dump [2], [4] | High |
| New Approaches 1 | 76.5 vs 70.5; 76.5 vs 70.0 [+6.5: 4.0, 9.1] on 300 fresh forms; router 79.2; vote 82.4 | [31], [32] | High (vote +1.5 over tie-break reader is small) |
| New Approaches 2 "The advantage reverses" | K = 10 and line items | [33], [34] | High |
| New Approaches 3 "Linking, measured end to end" | 47.4, 53.4, +5.9; multilingual costs 3 | [37], [24] | High |
| "Prior / intensive phase vs Ours" table | trained tagger 70.5 / reader 76.5; router 79.2; crossover K = 5; LiLT 47.4, multilingual 44.2 / BROS 53.4; tables "not covered" / DocILE row F1 59.6 | as above | High |
| Methods flowchart | BROS-base: 384-word windows, overlap 128, soft vote; Qwen3-VL-8B 4-bit zero-shot; heads: BIO tagger, biaffine linker, anchor rule; JSON answer grounded back to OCR words; per-field router | code listed in section 9 | High |
| Workflow nodes | intensive phase; error analysis (gap = 2 fields); zero-shot reader + router; linking end to end 47.4 to 53.4; FATURA crossover K = 1-40; grounded prompt VRDU 76.5; link backbone XLM, crops, BROS; tables ad-buy, DocILE; page pixels LayoutLMv3 | |
| Dataset table | VRDU forms: 6 fields, 1,915 docs, 3 templates; VRDU ad-buy: line items, 641 docs, N = 10-200; FATURA 9 fields, 50 templates, K = 1-40 seen; XFUND links, 5 languages, 250 val forms, 94% unseen; DocILE 19 line-item fields, 5,680 docs, 200 new layouts | dataset audits [13], [17], [34], [39] | High |
| "Results: unseen form templates" bars | BROS Track 1 / zero-shot reader / router per template: Amendment 85.2/75.7/80.5; Dissemination 69.6/86.2/75.5; Short-Form 57.2/65.5/82.4; pooled 70.5/76.5/79.2 | confirm set [31] (`hybrid_V2A2.json`) | High |
| Caption "Router +8.7 [+6.9, +10.6] pooled but -4.6 on Amendment. Dev set 75.2 vs 69.8" | | [31] | High |
| FATURA figure | crossover; "Tie at K = 5 (+0.9); BROS ahead by 4.2 at K = 10" | [33] | High |
| XFUND figure | LiLT-en 47.42, LiLT-XLM 44.22, + crops 49.70, BROS joint 53.35 | [24], [26], [37] | High |
| DocILE curve | unseen 44.7 / 52.9 / 59.6 / 60.4; seen 66.0; "Gap 2-9 points (VRDU 21); 21.4% of cells missed" | [39], [41], [44] | High |
| Conclusion: "ad-buy: reader 38.2 vs BROS 68.8-85.7 F1" | | [34] (68.8 from the file, 68.9 in the log; see Landmines) | High |
| Conclusion: pixels | "none on seen templates, +0.8 over three unseen (unstable seeds), 14 below BROS" | [42] | High as a negative, low as a positive |
| Conclusion: not checked | open-schema decoding, field names never shown, other domains and languages | | |

---

## 3. Single source of truth for numbers

### 3a. VRDU registration forms (6 fields; 3 templates; N = 200 training documents; MTL = seen, UTL = unseen)

| Model | MTL | UTL | Gap |
|---|---|---|---|
| LiLT baseline | 91.6 | 64.2 | 27.3 |
| Delaunay graph network over a frozen encoder | 86.0 | 42.1 | 43.9 |
| DANN-LiLT / **DANN-BROS** (template-adversarial) | 91.1 / **91.8** | 64.8 / **69.6** | 26.3 / **22.2** |
| **BROS Track 1** (windows + jitter + anchoring, 3 seeds) | **92.4** | **71.3** (std 11.2) | **21.1** |
| LayoutLMv3 without image (blank) | 92.4 | 56.6 | 35.8 |
| LayoutLMv3 with image | 92.2 | 57.4 | 34.8 |
| Qwen3-VL-8B grounded reader (zero-shot) | n/a | 76.5 (confirm), 75.2 (dev), 76.5 (fresh) | no split |

UTL by held-out template for BROS Track 1: Amendment 83.4, Dissemination_Report 72.1, Short-Form 58.4.

### 3b. The reader, set by set (exact-span F1; "text match" = official-style rule, about 2 points higher)

| Set (300 docs each) | BROS (mean of seeds 13, 42) | Prior prompt (v1) | Prompt v2 reader | Router with BROS |
|---|---|---|---|---|
| Dev | 69.8 | 58.1 | **75.2** | not computed |
| Confirm | 70.5 | 60.5 (post hoc decode; registered decode 60.2) | **76.5** | **79.2** (+8.7 [6.9, 10.6]) |
| Fresh | 70.0 (text match 69.6) | n/a | **76.5** (text match 78.4); +6.5 [4.0, 9.1] | not run |

Fresh set only (the replication): tie-break (latest page for `file_date`, `signer_name`, `signer_title`) 80.9 exact; abstaining three-way vote 82.4; vote minus tie-break reader +1.5 [0.4, 2.5]; text-only reader 76.9, so the image adds +4.1 [2.6, 5.7].

### 3c. FATURA (9 fields, 50 templates, 500 confirm invoices; exact block F1, template bootstrap)

| K templates seen | 1 | 2 | 5 | 10 | 40 |
|---|---|---|---|---|---|
| BROS | 32.5 | 69.2 | 90.8 | 95.9 | 99.9 |
| Reader (with image) | 91.7 | 91.7 | 91.7 | 91.7 | 91.7 |
| Reader minus BROS | +59.4 | +22.5 | +0.9 (tie) | -4.2 | -8.2 |
| Reader without image | 87.9 (so image +3.8) | | | | |

Post hoc text-first decode (not registered): reader 94.0; then K = 5 becomes +3.2 [1.1, 5.5], K = 10 -1.9. The oracle per-field router would add +4.2 to +5.8 over the better system at K = 1..10; no router was trained.

### 3d. Tables

| | Metric | Result |
|---|---|---|
| Ad-buy line items, BROS + anchor rule | row F1 at N = 10 / 50 / 100 / 200 | 68.9 / 83.2 / 83.8 / 85.7 |
| Ad-buy, reader | row F1 (128 docs) | 38.2 (1.71x too many rows) |
| Ad-buy oracle ceiling (line-citing) | | 97.8 |
| DocILE, BROS + anchor rule, unseen layouts | row F1 at N = 10 / 50 / 200 / 1,000 / whole pool (2,908 docs) | 0.3 (did not train) / 44.7 / 52.9 / 59.6 / **60.4** |
| DocILE, seen layouts (whole-pool runs) | row F1 | 66.0 (68.9 / 63.0 by seed) |
| DocILE gap | | 5.6 (2 to 9 by seed) |
| DocILE ceilings | perfect tags + anchor rule | 94.4 row F1 (cell F1 97.8) |
| DocILE, reader (pilot) | row F1 on 30 docs | 26.6; gate failed (88.3% of cited ids exist, bar 90%) |
| DocILE, VLM typer on BROS cells | row F1 vs BROS | -7.45 [-10.0, -5.1]; typing accuracy 46-52% vs BROS 77% |
| Published baselines (official LIR F1) | | RoBERTa-base 68.6 overall, 59.8 on unseen layouts; winner GraphDoc 75.9 (uses 932k unlabeled documents) |
| Learned pair classifier vs anchor rule | | ad-buy +2.38 [0.42, 4.71]; DocILE -4.32 [-9.28, -0.90]; not adopted |
| Column rules | | field vote -2.7; table-extent +0.0; neither adopted |
| DocILE loss budget (unseen, 3,113 gold spans per run) | | right 64.0%, missed 21.4% (13.6% partly tagged field, 6.9% field absent in the document), wrong field 8.2%, wrong extent 4.8% |

### 3e. Pixels

| Arm | Seen | Unseen |
|---|---|---|
| LayoutLMv3 with image vs blank, folds 0, 1 | -0.12 (6 pairs) | +3.35 (pages 63.75 vs blank 60.39; registered rule met) |
| Same, all three folds | | **+0.83** (57.4 vs 56.6); six pair differences +4.5, +3.2, +8.3, -2.6, +14.5, -22.9 |
| VLM reader, image vs text-only | | +3.3 (8B, confirm), +3.2 / +3.4 (v2 confirm/dev), +4.1 (fresh), +3.8 (FATURA) |
| BROS Track 1 | 92.4 | 71.3 (14 points above LayoutLMv3 either arm) |

### 3f. Linking (XFUND validation, 250 forms, 5 languages, end to end, 3 seeds)

| System | Link F1 | Entity F1 |
|---|---|---|
| LiLT-en (joint tagger + biaffine linker) | 47.4 (47.7 in the earlier run) | 75.8 |
| LiLT-InfoXLM (multilingual text stream) | 44.2 (-3.5 vs its own run; negative in all 3 seeds) | 73.2 |
| LiLT + link-preserving crops | 49.7 (+1.4 / +2.3 / +3.2) | |
| **BROS joint** | **53.4** (+4.8 / +5.9 / +7.1 per seed vs LiLT) | 80.2 |
| BROS joint + jitter | 53.0 | |
| Zero-shot Qwen2.5-VL-7B links | 23.7 given gold entities, 3.0 end to end | |

---

## 4. Deep dive A: the VLM prompt sequences (zero-shot, grounding, router)

> This is the section to know cold. The sequence is a *chain of failures that each fixed the previous failure*. Dates are 2026.

### 4.0 What "zero-shot" means here (say this first)

No weight of the VLM is trained on any form, labelled or not. Nothing about the held-out template enters the model. What *is* given: the field names and one-line descriptions of the six VRDU fields (the **schema**), written from the field names alone (study 7: written by the model; v2: three of them rewritten by us). The descriptions are the only template-adjacent knowledge, and that is the honest caveat (see W2 below).

### 4.1 Stage 1 (Oct 28): open-schema, page 1, image-only vs text-only (Study 6)

- Models: Qwen2.5-VL-3B (page image) vs Qwen2.5-3B (OCR text with [x,y] positions on a 0-1000 grid).
- **Exact prompt** (`scripts/zero_shot_kv.py`, `PROMPT`):
  > "List every field on this form page as key-value pairs. A key is a label or question printed on the form; its value is the information filled in for it. Copy text exactly as it appears. Skip keys that have no filled-in value. Output only lines of the form: `<pair><key>KEY TEXT</key><value>VALUE TEXT</value></pair>`"
  For the text model the same prompt plus "Page text, one line per row, with the [x,y] position of each line on a 0-1000 grid:" and the lines.
- **Mapping to our fields:** SBERT similarity between the generated *key text* and the field name, threshold 0.4, fixed in advance.
- **Result:** VLM beats the text model (Dissemination 79.8 vs 49.7; FUNSD pair F1 51.7 vs 30.1); the Short-Form decoy was never picked; but Short-Form `foreign_principle_name` had 98% value recall and 2.2 F1: the value was found under a key text that did not map to the field name. **Bottleneck: key to field naming.**
- Two bugs were found and fixed (parser required a `<pair>` wrapper the model often omits; ad-buy line-item fields absorbed predictions).

### 4.2 Stage 2 (Oct 29): whole documents + a naming step (Study 7)

- All pages (up to 4), 2,048 tokens, plus one extra prompt per extracted pair (`NAMING_PROMPT`):
  > "A key-value pair was extracted from a document. Give its field a short canonical name in snake_case and a one-sentence description of the information it holds, in your own words. If the value combines separate pieces of information, write one line per piece with the part of the value it covers." + one worked example (Ship To / Jane Smith) + "Now this pair / Key: {key} / Value: {value} / Answer:", with the answer **prefilled** with `<field>`.
- Schema descriptions for the six fields: generated by the model from the field names (`FIELD_DESC` prompt: "A document data-extraction schema has these fields: ... In one sentence, describe the information that the field 'X' holds ... Output only: `<desc>DESCRIPTION</desc>`").
- Two mapping modes: **key** (raw key vs field name) and **named** (generated name + description vs field name + description).
- **Process lesson:** the 3B model ignored a many-pairs tagged format (0/6 and 0/14 parses) but followed *one short tagged answer per question with a prefilled tag and a worked example* (19/20). That robustness trick is reused in every later prompt (one call per field, prefill `ANSWER: `).
- **Result:** reading all pages recovered fields beyond page 1 (Amendment `file_date` 0 to 72.5); naming fixed wording (Short-Form `foreign_principle_name` 4.3 to 85.1) but cost precision.

### 4.3 Stage 3: one answer per field (Study 8, post hoc), grounding, matched comparison (Study 9)

- **One answer per field** (keep the candidate with highest similarity): Short-Form `registrant_name` 33.2 to 96.5; decoy picked 0/100. Decided after seeing results, so labelled post hoc.
- **Grounding to word spans** (`scripts/matched_eval.py`): exact match on the extraction page, then anywhere, then fuzzy (token F1 at least 0.5); one span per value because VRDU annotates one occurrence.
- **Matched comparison with BROS on the same 100 docs per template** (exact-span F1): pooled VLM 57.9 (key) / 61.2 (named) vs BROS 68.6 / 70.9. BROS wins overall; **errors are complementary**:

| Field | Amendment BROS / VLM | Dissemination | Short-Form |
|---|---|---|---|
| `foreign_principle_name` | 0 / **50** | 0 / **61** | 0 / **58** |
| `registrant_name` | 86 / 88 | 78 / 76 | 2 / **62** |
| `signer_name` | **79** / 36 | 31 / 36 | **85** / 23 |
| `file_date` | **76** / 43 | 76 / 78 | **91** / 51 |
| `signer_title` | **71** / 14 | **47** / 6 | n/a |

  Where the VLM loses points: span *boundaries* (extra words, combined address), then location.
- This is why we pursued a hybrid rather than "the VLM replaces BROS".

### 4.4 Stage 4: the router (Studies 10, 11, 12) and why a first version failed

- **Router definition (leave-one-template-out):** per field, pick the source (BROS, VLM key mode, VLM named mode; later `vlm_both`) with the best pooled F1 on the *other two templates*, and apply it to the held-out template. So the router never sees the test template.
- **Study 10 (pre-registered pass rule: pooled interval above 0 and no template significantly worse):** neither fill-in nor router passed. Why: the key/named mode chosen on two templates was wrong for the third (Short-Form needs named; the other two prefer key). Oracle ceiling: +16 F1.
- **Study 11:** a mode-free mapping (`vlm_both`: candidates from both modes, the most similar wins). Router passes: +3.9 pooled. What it learned (identical for all folds): **VLM for `foreign_principle_name` and `registrant_name`; BROS for `file_date`, `signer_name`, `signer_title`** (and `registration_num`). It is a two-field switch.
- **Study 12 (frozen rules on 300 fresh documents):** +3.8 [+2.6, +5.0] (study 11: +3.9). Almost all of it is `foreign_principle_name` (5.1 to 55.3).
- This was the first confirmation run in the project. Same pattern repeats in studies 31 and 32.

### 4.5 Stage 5: the schema-guided grounded recipe (Study 25, LMDX-style)

- Inspiration: LMDX (arXiv 2309.10952): schema in the prompt, OCR lines each carrying a coordinate identifier, the model answers by citing a line, the answer is verified against the cited line.
- **Line identifier:** the line's centre as `XX|YY`, 100 buckets each (so `42|17` means 42% across, 17% down); a `#2` suffix resolves collisions.
- **Per-page JSON prompt (v1)** (`build_prompt`, `v2=False`):
  > `<Document>` one line per OCR line, each followed by its id `</Document>` `<Task>` "From the document, extract the text values of the following fields. Every document line ends with its identifier (two numbers separated by |). Write each value as the exact text copied from the line, followed by that line's identifier. A value that spans several lines is written as one 'text identifier' per line, separated by a newline. Use null for a field that does not occur on this page. Field descriptions: ... Answer with this JSON object: {"file_date": "", ...}" `</Task>` `<Extraction>`
- **Arms:** A0 = the earlier open-schema pipeline (reference); **A2** = image + OCR lines + schema (primary); A1 = no image (pixel ablation); A3 = 8-sample vote (incomplete).
- **Format gate (pilot of 30 documents):** at least 90% of outputs must parse and at least 90% of values must cite a valid line id, otherwise fall back to one call per field. Result: **JSON failed the gate with both models** (parse 100%, but valid ids only 0% for 3B and 5.5% for 8B), so the registered fallback ran: *one call per field, answer prefilled with `ANSWER: `*.
- **Per-field prompt (v1)** (`field_prompt`):
  > `<Document>` ... `</Document>` `<Task>` From the document, extract the value of the field "{field}" ({description}). Every document line ends with its identifier. Answer with the exact text copied from the line, followed by that line's identifier. If the field does not occur on this page, answer NONE. `</Task>` `ANSWER: `
- **Model:** 3B first; an 8B arm (Qwen3-VL-8B-Instruct, 4-bit nf4, fp16 compute so it fits a T4) was added on 2026-11-18 after a literature check. Image goes first in the chat message, then the text.
- **Grounding decode (`ground()`):** a value is *grounded* only if its text occurs inside the cited line (exact whitespace-insensitive match, else normalised); mapped to words by character overlap; several cited segments are merged while contiguous; a later disjoint segment is dropped. A hallucinated value is rejected. **Oracle check:** feeding gold values with their true line ids through the whole decode path scores 99.7 F1 (so the line-to-word mapping is validated before any model is involved).
- **Voting:** per field, majority over pages; ties to the earliest page. Overlapping spans of different fields are resolved by votes then field order.
- **Registered result: not an improvement.** 8B A2 60.2 vs reference 57.7 (+2.5 [0.0, 5.2], below the +3.0 bar), about 10 below BROS. 3B A2 scored 4.0 because of a decode flaw (truncated ids such as "43" for "43|96"); with a text lookup fallback 56.6. Post hoc: **image adds +13.7 (3B) / +3.3 (8B)**; the static router with the 8B spans gave **76.2 vs 70.5 (+5.7)**.
- **What was wrong (error review, the basis for prompt v2):** `file_date` 22.3 (the model picked the *received stamp*, not the written signing date); `signer_name` 41.1 (it copied "Name, Title"; 107 of 222 too long; snapping to BROS lifted it to 72.6); `registrant_name` 62.3 (sometimes returned the signer). Also LMDX's ~90 was not reproduced: LMDX tuned a large model on synthetic forms.

### 4.6 Stage 6: prompt v2 (Study 31): the biggest single jump

- **Exact v2 per-field prompt (the one that ran)** (`field_prompt(..., v2=True)`):
  > `<Document>` {lines, each "text XX|YY"} `</Document>` `<Task>` From the document, extract the value of the field "{field}" ({description}). Every document line ends with its identifier. **Copy the value exactly from ONE line and write it followed by that line's identifier, for example "Jane Q. Public 42|17". Copy only the value itself: not the label in front of it and not other words on the same line (for a person's name, leave out their title or position). If the value spans several lines, write one 'text identifier' per line. If the field does not occur on this page, answer NONE.** `</Task>` `ANSWER: `
  (The per-page JSON v2 has the same four rules plus an invented worked answer `{"file_date": "March 5, 2012 31|12", "registration_num": "6123 08|40", "signer_name": null}`; it reached 84.4% valid ids, still below the 90% gate, so the per-field format again ran.)
- **Three descriptions rewritten** (`results/schema_guided_kv_v2/schema_desc_v2.json`):
  - `file_date`: "The date on which the form was signed or dated by its signer, as written beside the signature (for example 'the 5th day of March, 2012'). It is not a received or filed stamp."
  - `registrant_name`: "The name of the organisation (firm, company or office) that files the registration, not the name of the person who signs it."
  - `signer_name`: "The full name of the person who signs the document, written without their title or position."
  (Unchanged: `foreign_principle_name`, `registration_num`, `signer_title`.)
- **Result (confirm):** 76.5 vs 60.5 for the old prompt: **+16.0 [13.7, 18.4]**, positive on every template (75.7 vs 59.3; 86.2 vs 73.8; 65.5 vs 44.3). **Dev replication:** 75.2 vs 58.1, **+17.0 [14.6, 19.3]**. Router with BROS 79.2 (+8.7 [6.9, 10.6]) on confirm. Image adds +3.2 / +3.4. Text-only (no image) is already 73.3 / 71.7, above BROS 70.5 / 69.8.
- **Per field (confirm; v2 / old / BROS seed 13):** `file_date` 79.5 / 22.3 / 84.0; `foreign_principle_name` 52.5 / 70.4 / 2.0; `registrant_name` 90.4 / 62.3 / 56.9; `registration_num` 97.2 / 97.3 / 97.4; `signer_name` 53.4 / 41.1 / 64.9; `signer_title` 75.1 / 73.5 / 50.3. **The prompt fixed the field-choice errors but regressed `foreign_principle_name`** (the "copy ONE line" rule truncates multi-line names) and `signer_name` stays below BROS.
- **Honest limits written in the study before the result:** the prompt and three descriptions were written after seeing study-25 errors on the same confirm documents; the rules, the example and the descriptions are confounded in one change; the `file_date` description names the signing date (Amendment-style knowledge); the decode fallback is post hoc.

### 4.7 Stage 7: the fresh-forms replication and two post hoc rules (Study 32, run 2026-11-23)

- **Data:** 300 registration forms (100 per template, seed 11) outside dev, confirm and the pilot (630 documents excluded); BROS predictions existed already from the saved Track-1 dumps (trained before these documents were chosen; no weights saved, so nothing was selected on them).
- **Recipe frozen:** Qwen3-VL-8B 4-bit, prompt v2, descriptions v2, one call per page and field, one sample per page, fallback decode, up to 4 pages. Runs: V2A2 (image) and V2A1 (text only). 3.31 h on two T4s, no failed shard.
- **Rules registered before the run** (`scripts/study32_fresh.py`):
  1. *Prompt replicates:* V2A2 minus BROS at least +3.0 (text-match), interval above 0.
  2. *Tie-break:* when a field is found on several pages with equal votes, take the **latest page** for `file_date`, `signer_name`, `signer_title` (the signature block is at the end of the form); other fields keep the earliest page. Needed: at least +2.0 over the earliest-page reader.
  3. *Abstaining vote:* candidates from V2A2, V2A1 and BROS; the span proposed by most systems wins; **abstain unless at least two systems propose exactly the same span**. Needed: at least +1.0 over the tie-break reader.
- **Results (exact span / text match):** BROS 70.0 / 69.6; V2A2 earliest page **76.5 / 78.4** (+6.5 [4.0, 9.1] / +8.8 [6.3, 11.3]); with tie-break **80.9 / 81.0** (+4.4 [3.1, 5.6]); V2A1 76.9 / 77.5; abstaining vote **82.4 / 82.6** (+1.5 [0.4, 2.5] over the tie-break reader). **All three rules met.**
- **By template (exact):** Amendment BROS 80.9 / 84.3 vs reader 75.6 (earliest) / 78.2 (latest): BROS still wins Amendment. Dissemination 69.5 / 74.9 vs 87.7 / 87.0. Short-Form 54.9 / 55.2 vs 64.1 / 76.0.
- **By field:** `foreign_principle_name` 0.0 to 55.7; `registrant_name` 58.0 to 89.1; `signer_name` 63.3 vs 55.0 (earliest) / 69.3 (latest); `file_date` 80.8 vs 77.9 / 83.7; `registration_num` equal.
- **What it does not show:** a new corpus or language (same three templates); the router was not rerun here; vote gain is small; BROS has two seeds.

### 4.8 The other VLM probes (for "did you try ...?")

| Try | Result | Why it matters |
|---|---|---|
| LoRA-tuned Qwen2.5-VL-3B with a word-range output (r = 16, N = 200, 3 epochs) [29] | UTL 46.2 (replication 44.7) vs BROS 70.1; MTL 76.1; gap 29.9; 11 below the zero-shot pipeline | Tuning a small VLM on 200 documents makes it worse; the output format forces boundary guesses. 25% of gold fields found with wrong boundaries, 22-30% not predicted. |
| Output format of the tuned VLM [30] | never run | The tuned VLM already lost; dropped. |
| Zero-shot VLM *linking* on XFUND (Qwen2.5-VL-7B) [27] | 23.7 given gold entities, 3.0 end to end, 23.9 with the trained tagger's entities, vs 47.4 trained | The VLM over-links (precision about 19%). Linking is not a prompt-only task. |
| VLM as field *typer* on BROS cells in DocILE [43] | -7.45 row F1 | The VLM does better only on net-only documents (27% vs 8%) but cannot tell the document kind. |
| Recipe on DocILE line items [38] | pilot gate failed; 26.6 row F1 | See tables. |
| Recipe on ad-buy line items [34] | 38.2 vs 68.9-85.7 | See tables. |
| Per-document confidence gate for the router [16], error correlation [18] | gate alone -0.2 on VRDU; errors positively correlated (VLM right on 32% of BROS misses vs 70% of BROS hits) | Why a "smart" router is not trivial. |
| Learned router [36] | only headroom measured: perfect per-field router +4 to +6 over the better system at K = 2..10 on FATURA | Honest "future work". |

---

## 5. Deep dive B: how we incorporated tables

> **Core idea.** A table row is a *link problem*: detect cells (sets of words), then attach each cell to the row it belongs to. We did it with the same BROS tagger plus a simple geometric grouping rule, and compared it with the grounded reader.

### 5.1 The row-grouping rule (the "anchor rule", `scripts/row_group.py`)

1. The tagger predicts BIO spans for the line-item fields (ad-buy: 5 fields; DocILE: 19).
2. **Per page, pick the anchor field** = the field with the most predicted spans.
3. Each predicted cell goes to the **closest anchor at or above it** (a cell whose centre is within 0.5 of its own height *below* the anchor still counts as "at"), else the closest anchor below. Each anchor starts one row.
4. A row holds at most one cell per field on ad-buy (checked on the gold: no gold row repeats a field), so same-field pieces in one row are merged; 26% of DocILE rows do repeat a field, an accepted approximation there (the taxonomy runs without merging).
5. No learning. A learned pair classifier (AUC 0.97-0.998) beat the rule on ad-buy (+2.4) and lost on DocILE (-4.3), because *clustering*, not pair scoring, was the weak part, so the rule stays.

### 5.2 Metric ported (`scripts/lir_metric.py`)

- **Row F1 (LIR):** predicted rows are matched one-to-one to gold rows to maximise exactly matching cells (Hungarian algorithm, checked against brute force on 200 random cases); a cell counts only inside its matched row; micro F1 over cells. Cell identity = a *set of words* (a cell can span two OCR lines).
- **Cell F1** ignores rows; **cell F1 minus row F1 isolates the cost of grouping.**
- Our port differs from DocILE's official scorer (word-set equality vs pseudo-character coverage); the official one was not run.

### 5.3 Ceilings (computed before any model ran; the "what is the best possible" step)

- Ad-buy: citing gold cells by OCR line gives only 93.0; **cutting lines where neighbouring words are more than 0.5 line heights apart** gives 97.8; the remaining error is one field (`program_end_date`) because gold cells are partial strings inside glued tokens.
- DocILE: lines cut at 0.5 give 99.94 row F1; 3.5% of gold cells contain no OCR word, so no citing method can exceed about 96.5% recall; BROS-style tags with the anchor rule: **94.4** row F1 (our ceiling).

### 5.4 Ad-buy (Study 34; 641 documents, 9,163 gold rows, DeepForm templates unlabeled)

- Training sizes N = 10 / 50 / 100 / 200 (3 split seeds); primary test = 143 documents in every split's test set.
- **BROS + anchor rule: 68.9 / 83.2 / 83.8 / 85.7 row F1. Reader (A2): 38.2** (128 documents; 1.71x too many rows). **BROS wins at every N**, no crossover for N >= 10. Cleaning the reader's rows only reaches 43.8.
- Reader format: one call per page; answer `{"rows": [{field: "text identifier" | null, ...}]}`; a "row-position repair" moves a cell citing a piece that does not hold its text to the piece that does and sits on the row's line. (A simulation with wrong ids put in by hand: at 10% wrong ids, cell F1 96.0 with the row repair vs 89.3 with no repair.)

### 5.5 DocILE (Studies 38 to 44; 5,680 labelled documents; layouts split via `cluster_id`)

- **Data:** train 5,180 + validation 500 documents; 36 key-information and 19 line-item field types; 1,063 layouts (median 1 document per layout; 558 singletons). Layout-disjoint test: 200 documents (886 gold rows; median 2 rows per document). Training order registered; early stopping on 100 validation layouts.
- **Reader on DocILE [38]:** pilot of 30 documents: parse 100%, rows on 90.9% of pages, **cited ids that exist 88.3% (bar 90%): gate failed**, test run not made; pilot row F1 26.6. Reading: below the registered "below 50" band, i.e. tables are hard for the recipe in general (not only ad-buy scans).
- **BROS tagger + anchor rule [39, 41]:**

| N training docs | 10 | 50 | 200 | 1,000 | whole pool (2,908) |
|---|---|---|---|---|---|
| Unseen layouts, row F1 | 0.3 | 44.7 | 52.9 | 59.6 | 60.4 |
| Seen layouts | | | | | 66.0 |

  N = 10 is a **training failure** (loss still 0.9-1.1 and validation F1 0.0 at early stop after 7 epochs; epochs are a handful of steps), not a measurement of ten documents. N = 50 is probably step-limited too. From N = 200 the runs overfit the training layouts (loss 0.02-0.07). The curve **flattens**: +8.2 (50 to 200), +6.7 (200 to 1,000), +0.8 (1,000 to whole pool).
- **Published comparison:** RoBERTa-base LIR F1 68.6 overall, **59.8 on unseen layouts**, 56.8 on layouts with 1-3 training documents, 75.6 on layouts with 4+; winner GraphDoc 75.9 uses 932k unlabeled documents (not allowed under the zero-same-template rule). So our 59.6-60.4 is "level with the published baseline on unseen layouts, with at most 1,000 labelled documents and no pre-training": ordinary, not strong. Different test documents; rough comparison.
- **Failure taxonomy [44]** (3,113 gold spans per run; unseen): **right 64.0%**; missed 21.4% (13.6% a field partly tagged, 6.9% a field absent in the whole document, 1.0% non-consecutive words); wrong field 8.2% (mostly gross vs net); wrong extent 4.8%; both 1.5%. False spans: 22.6% of predictions (15.5% inside the table, 7.1% outside). By field: `amount_net` 9.7% right, `unit_price_net` 17.7%, `unit_price_gross` 54.9%.
- **What breaks generalisation (deliverable 5):** (1) *gross vs net is a property of the document, not the layout* (266 of 421 net-only documents sit in layouts that also have gross-only documents; 58 of 241 layouts with at least two such documents are mixed); (2) "every table has this field" fails for secondary fields (currency, units, code are most often absent); (3) the unseen-layout loss is mostly *omission* (partly tagged fields), not mislabelling.
- **Seen vs unseen:** what shrinks on seen layouts is template-dependent (absent fields -4.3, partly tagged -3.1, monetary typing -2.1, false spans in the table 13.9% to 7.2%); what does not shrink is intrinsic (false spans outside the table, boundary errors).
- **Two things tried to fix it, neither adopted:** column vote -2.7 row F1 [40]; table-extent rule +0.0 [40]; VLM typer -7.45 [43].
- **Why the DocILE seen-unseen gap is small (2 to 9) vs forms (21):** rows share structure across layouts (column order and header words), so a tagger transfers most of the table; the loss is field typing and missed cells, not layout shift. (Seed spread is large: gap 9.1 for seed 13, 2.0 for seed 21.)

### 5.6 The "Link" connection (why the poster says tables and key-value are both links)

The methods diagram has one encoder and three light heads: BIO tagger (entities), biaffine linker (question to answer, trained on XFUND), anchor rule (cell to row). The claim is *architectural economy*: a link between two detected objects covers both a key to its value and a cell to its row. We did **not** train a single model on both; the tables use the BROS tagger plus the rule. Be ready to say so.

---

## 6. Deep dive C: effect of the number of training templates (and the work on it)

> **Question the course asks:** "how does performance change with the number of training templates?" We answered it on three corpora with different answers.

### 6.1 VRDU forms: fixed N = 200, three folds (the protocol)

- Matched-size protocol: N = 200 documents in both regimes so MTL and UTL differ only in *which* templates are present. MTL: random documents from all templates. UTL: one template wholly held out. 3-fold template-disjoint; `assert_no_template_leakage` verifies no leakage (zero validation/test leakage); validation set contains no held-out template.
- Only three templates exist, so K = 2 is the only K on VRDU. That is why we needed FATURA.

### 6.2 FATURA: the clean K sweep (Studies 13 to 16, 33)

- **Dataset audit [13]:** FATURA qualifies (50 labelled templates, 9 core fields, official unseen-template split) but needs block-level scoring and tokens rebuilt from the original annotations. It is synthetic (generated values, clean rendering) and the OCR is the annotation, so both systems are favoured.
- **First test [14]:** BROS scores 99.9 on unseen templates (K = 40) so the router had nothing to fix; the 3B zero-shot VLM scored 59.6. **Uninformative at K = 40.**
- **Data-starved BROS [15, 16]:** K = 1, 2, 5, 10 seen templates (BROS Track-1 recipe unchanged): 32.5 / 69.2 / 90.8 / 95.9 / 99.9 (K = 1, 2, 5, 10, 40); 5 seeds at K = 2 and 5. The old 3B VLM (59.6) was beaten by BROS from K = 2. Oracle headroom was +16 / +5 / +2 at K = 2 / 5 / 10, but a static per-field router captured none; the confidence gate gained only at K = 2 (+1.5). (Study 15's K = 2 router loss did not replicate with 5 seeds: +2.3 [-2.1, +7.3]).
- **Grounded reader on FATURA [33]:** same recipe as VRDU (prompt v2, nine field descriptions written from the field names alone, worked example in FATURA's names). Gate: JSON kept (parse 100%, ids valid 96.8%). **A2 91.7 / A1 87.9** (image +3.8): +32.1 [27.1, 36.9] over the 3B zero-shot pipeline. **Crossover:** K = 1 +59.4, K = 2 +22.5, K = 5 +0.9 (tie), K = 10 -4.2, K = 40 -8.2.
- **Why the reader's line is flat:** it uses no training; its 91.7 is a property of the model + prompt on this layout family. Its misses: `buyer` (only 31 of 50 templates annotate a buyer; the model fills a "BILL TO" block on 144 of the 190 un-annotated invoices; so buyer precision 0.59 is an *annotation* penalty); 37 of 42 wrong-block buyer answers quote the right text but cite another line's id.
- **Text-first decode (post hoc):** use the block that holds the quoted text when the cited id's block does not. 94.0 (+2.3); K = 5 becomes +3.2 [1.1, 5.5] (registered: a tie); K = 10 -1.9.
- **Combinations** (BROS with the LLM filling empty fields; LLM with BROS filling): **neither beats the better single system** at any K. At K = 1, 2 the BROS-first fill is *below* the LLM alone.
- **Learned router [36]:** only the oracle headroom was measured (+4.2 / +5.4 / +5.8 / +4.1 / +0.1 at K = 1 / 2 / 5 / 10 / 40) and an unfair no-abstention oracle was discarded. No router trained.
- **Therefore, the crossover is "about 5 templates" (reader wins to 2, tie at 5, BROS from 10).** Where *you* say "10 seen templates", you mean "the tagger has clearly overtaken by 10".

### 6.3 Ad-buy and DocILE: N training *documents* (a different axis)

- **Ad-buy** (DeepForm templates are unlabeled in the data): BROS 68.9 / 83.2 / 83.8 / 85.7 at N = 10 / 50 / 100 / 200: saturates by N = 50.
- **DocILE:** unseen-layout row F1 0.3 / 44.7 / 52.9 / 59.6 / 60.4 at N = 10 / 50 / 200 / 1,000 / 2,908. +6 to +8 points per doubling of layouts at first, then a plateau: "layouts" not documents is the unit (the pool has 562 layouts with at least one line-item document; N = 1,000 covers about 285).
- **The LoRA-tuned VLM** is the only experiment that varied the *tuned* data: seen 76.1, unseen 46.2.

### 6.4 What the training-templates work *means*

1. The wording problem (`foreign_principle_name`) cannot be fixed by BROS at N = 200 on three templates because the wording variants do not repeat; it *can* be fixed by reading the field name (the reader) or by seeing many templates (FATURA K >= 10).
2. For a field family that is layout-regular (dates, signatures, line-item columns) training transfers well *once enough templates have been seen*.
3. For a business user: "if you have <= 5 labelled templates for a new form type, use the reader; at ~10 or more, train; for tables, train from the start".

---

## 7. Deep dive D: effect of pixels

### 7.1 In the zero-shot VLM reader (descriptive arm "no image" in every VLM study)

| Study | Setting | Image effect (exact-span F1) |
|---|---|---|
| 25 | 8B, confirm | **+3.3 [1.8, 4.8]**; dev +2.8 |
| 25 | 3B, confirm, with fallback decode | +13.7 [11.2, 16.3] (read with caution: the 3B with the image wrote truncated ids, so its image-arm score depends heavily on the fallback decode) |
| 31 | prompt v2, confirm / dev | **+3.2 [1.6, 4.8] / +3.4 [1.5, 5.4]** |
| 32 | fresh set, latest-page rule | **+4.1 [2.6, 5.7]** (text match +3.5 [2.1, 5.0]) |
| 33 | FATURA | +3.8 (A2 91.7 vs A1 87.9; `title` and `number` +26 each; `buyer` slightly negative) |

**Reading:** image helps consistently by about +3 to +4 F1; text-only is already competitive with BROS (73.3 / 71.7 vs 70.5 / 69.8). Cost: V2A2 5.3 GPU-hours per 300 documents (both GPUs summed) vs 1.7-2.3 for text-only.

### 7.2 In a fine-tuned encoder: LayoutLMv3 with and without pixels (Study 42)

- **Why:** the project description (deliverable 2) asks for a representation that jointly uses text, visual and layout information; LiLT and BROS see text and layout only. LayoutLMv3 is a fine-tuned encoder that reads the page image next to the words and boxes.
- **Arms** (identical recipe, seeds 13 and 42, N = 200, three folds): **pages** = each window of words is paired with the image of the page that holds most of its words, rendered at 100 dpi, resized to 224 x 224, normalised; **blank** = the same model and recipe with a constant zero image (the visual stream switched off). `pages` minus `blank` isolates the pixels.
- **Recipe:** Track-1 BROS recipe *except* geometry options: 20 epochs max, early stopping, lr 5e-5, batch 4, 384-word windows with overlap 128, soft voting; **no coordinate jitter and no content-box anchoring** (they move the boxes but not the pixels, so box and image would stop agreeing).
- **Design fix before any run:** 36.7% of default windows cross a page boundary and 11.1% of word slots would have faced the wrong page's image; so windows are cut at page boundaries in both arms. (A trade-off: a window no longer carries text from the previous page.)
- **Results:**
  - **Seen templates:** pages 92.24, blank 92.35, BROS 92.36 (six paired differences, mean -0.12). On seen templates the page image makes no difference; the ceiling is about 92 whatever the input.
  - **Unseen, folds 0 and 1 (the registered primary cell: needs mean >= +2.0, 3 of 4 positive, both fold means positive):** pages 63.75 vs blank 60.39, **+3.35**, 3 of 4 positive: *pixels help* (rule met). Fold 1 seed 42 (pages) scored 51.57, below its own blank pair.
  - **Fold 2 (Short-Form) added after the verdict:** pages 44.7 vs blank 48.9; seeds 25 and 12 points apart. **Three folds: pages 57.39, blank 56.56, +0.83.** The registered two-fold verdict stands as registered but does not survive the third fold as a claim: "small and unstable".
  - **Where the image helps** (folds 0 and 1, per field): `signer_title` +28.5, `signer_name` +7.9 (signature-block fields); it costs `registrant_name` -4.4 and `file_date` -2.6 (fields text and position settle).
  - **Gap (MTL minus UTL, mean of folds):** pages 34.9, blank 35.8, BROS Track 1 21.1; on folds 0-1 only: 28.5 / 31.4 / 14.6.
  - **LayoutLMv3 is 14 points below BROS Track 1 on unseen templates either way** (-14.0 pages, -17.3 blank on folds 0-1). Confound: no jitter, no anchoring.
- **Limits:** one recipe, two seeds, English, 224 x 224 too coarse for small print (a negative result says only that *this* input does not help); weights CC-BY-NC-SA (declared).

### 7.3 What pixels mean for the project

Pixels carry a small, consistent benefit for a model that *reads* (the VLM, +3 to +4) and a small, unreliable one for a model that *tags* with a fixed 224-pixel image (+0.8). The best unseen-template model remains the geometry-aware text encoder (BROS). Pixels do not close the seen-unseen gap.

---

## 8. Deep dive E: linking (the other half of the poster)

- **Task and metric:** XFUND validation forms, 250 forms, 5 languages (the XFUND validation set); entities typed header / question / answer; links directed question to answer (and header to question). **End to end** = the model must find the entities itself, long forms are read in windows (510 tokens, overlap 128), and a link counts only if both ends exactly match and the direction is right. 94% of the validation forms have no training form of the same template (sibling-free).
- **Model:** joint LiLT (or BROS) tagger + biaffine link head (`src/models/joint_extractor.py`).
- **The 65.5 mistake (do not quote):** the earlier "65.5 link F1" scored only the 61% of links visible in a 512-token window and gave the linker *gold* entities. Counting cut-off links as misses gave 49.9 [17]; link-aware windows 59.8 [17]; and training with windows and scoring with *predicted* entities gave 43.5 to 47.7 [19]. The poster and report now report only the end-to-end values.
- **Windows help** (+4.2 link F1, all 3 seeds; a quarter to a third of it is just more gradient steps: step-matched control +1.2 of the +4.2 [21]); **entity deduplication across windows** did not help (-0.35 [20]).
- **Error analysis (gold entities):** 72% of missed links point downward (recall 49 vs 79 for same-line partners); 30% of misses are question to question; 25% unreachable by any window [22].
- **Decode levers** (threshold, one-to-one, type prior): +2.4, mostly precision; a later correction showed much of that came from forbidding pairs that touch gold `other` entities, which the end-to-end pipeline never creates [23].
- **Multilingual text stream [24]:** LiLT-InfoXLM replaces the English text stream: 47.7 to 44.2 end to end (-3.5, negative in all 3 seeds); single-window link F1 is unchanged (41.7 vs 41.9), so the loss is in windowed inference. (Poster: 44.2 vs 47.4; the 47.4 is the retrained LiLT-en of the later runs.)
- **Link-preserving crops [26]:** crop long forms around links instead of cutting arbitrarily: 47.4 to 49.7 (+1.4 / +2.3 / +3.2); gains on partner-above (+4.0). Synthetic relation-rich pre-training alone +0.1; both 49.9. Crops get about 2x the steps (a confound) and lower link F1 on synthetic held-out families.
- **BROS backbone [37]:** BROS's relative-position attention in place of LiLT's absolute-box stream: **47.4 to 53.4** (+4.8 / +5.9 / +7.1), entity F1 75.8 to 80.2, link_qa 53.2 to 59.6; **jitter adds nothing** (53.0). BROS is English-only, so the gain appears *despite* the language mismatch.
- **Zero-shot VLM for links [27]:** 23.7 (gold entities given), 3.0 end to end, 23.9 with the trained tagger's entities; precision about 19%. So the claim "zero-shot beats training" is *field-extraction-specific*, not link-specific.
- **How to say it:** "geometry helps linking; multilingual text and prompt-only VLMs do not".

---

## 9. Code and result map (where to point if asked "show me")

| Topic | Code | Results |
|---|---|---|
| Prompts, grounding, vote | `scripts/schema_guided_kv.py` (`build_prompt`, `field_prompt`, `ground`, `decode_doc`); open-schema: `scripts/zero_shot_kv.py`, `zero_shot_kv_v2.py` | `results/schema_guided_kv_v2/` (`schema_desc_v2.json`, rows, `hybrid_V2A2.json`), `results/zero_shot_kv_v2/`, `results/confirm_v2/` |
| Router | `scripts/hybrid_eval.py`, `confirm_eval.py`, `study31_router.py` | `results/hybrid_eval/`, `results/schema_guided_kv_v2/hybrid_V2A2.json` |
| Fresh-forms rules | `scripts/study32_fresh.py` | `results/fresh_set3/` (`manifest.json`, `bros_spans.jsonl`, `run/fresh_report.json`) |
| FATURA | `scripts/fatura_sg.py`, `fatura_sweep.py` | `results/fatura_sg/`, `results/fatura_v2_sweep/`, `results/router/headroom_fatura.json` |
| Tables | `scripts/row_group.py`, `lir_metric.py`, `adbuy_*.py`, `docile_*.py` | `results/adbuy_rows/`, `results/docile_bros/`, `results/docile_bros_full/`, `results/docile_taxonomy/`, `results/docile_typer/` |
| Pixels | `src/models/token_clf.py`, `src/data/page_images.py`, `configs/vrdu_layoutlmv3*.yaml` | `results/vrdu_layoutlmv3*_utl_f{0,1,2}/`, `results/LayoutLMv3/` |
| Linking | `src/models/joint_extractor.py`, `src/e2e_link.py`, `src/vlm_link.py`, `src/train.py` | `results/table_links/`, `results/bros_joint/J1/` |
| Demo | `demo.py`, `src/demo.py` | BROS joint J1 seed 42 (`results/bros_joint/J1/seed_42/best_model.pt`) |

**Quick re-checks (CPU, a minute each):**
- Fresh-forms scoring: `python scripts/study32_fresh.py score --rows-dir results/fresh_set3/run`.
- Oracle check (grounding path validity): `python scripts/schema_guided_kv.py oracle --manifest results/zero_shot_kv_v2/manifest.json --ref results/matched_eval/spans.jsonl`.
- Self-test of the decode path: `python scripts/schema_guided_kv.py selftest`.

---

## 10. Weak points, ranked by how much they can cost

**W1. The reader's prompt and three descriptions were written after seeing errors on the confirm set (cost: high if they find it).**
They *will* find it if they read the report carefully, so lead with it. Rebuttal: stated in the study before the result; replicated on dev (+17.0) and then on 300 fresh documents never used for any prompt or rule (+6.5 over BROS, +10.9 with the tie-break); every rule that mattered was registered before the fresh run. Limit to concede: dev and fresh forms are the same three VRDU templates, so this is a replication on new documents, not new templates.

**W2. "Zero-shot" is not "no template knowledge" (cost: high).**
The field descriptions are template-adjacent: the `file_date` description names the *signing date*, which is Amendment-style knowledge we learned from error analysis. Rebuttal: no weights saw any form; descriptions are four sentences in a schema file, exactly what a deployer would write; on FATURA we used *model-written descriptions from field names alone* with the same prompt and got 91.7. Concede: the `file_date` result (22 to 80) rests on a description we tuned.

**W3. The router "passes" only if you ignore its worst template (cost: medium-high).**
With prompt v2 the router gains +8.7 pooled but loses 4.6 on Amendment, so the registered "no template significantly worse" rule fails (`hybrid_V2A2.json`: `passes: false`). The vote on the fresh set is a different, later rule (all three rules met). Be precise: say "the router gains pooled but costs Amendment 4.6; the vote is the more robust combination" and show the per-template bars (they are on the poster).

**W4. The router is nearly a two-field switch (cost: medium).**
Study 12 showed the fitted tables equal the frozen tables in 5 of 6 folds: it routes `foreign_principle_name` and `registrant_name` to the reader. Rebuttal: that *is* the finding (wording fields vs layout-regular fields); the oracle gap (+16) is mostly open, which is why the learned router is future work.

**W5. Few seeds, three folds (cost: medium).**
BROS has two seeds (Track 1 benchmark: three); the VRDU UTL std is 11; folds are three templates. Rebuttal: pairing by document/fold, bootstrap intervals on the reader-vs-BROS differences, and the effect replicated on three document sets. Concede: no hypothesis test across templates (n = 3).

**W6. FATURA is synthetic and its OCR is the annotation (cost: medium).**
Both systems are favoured. Rebuttal: stated up front in the study; the crossover is a *shape* (reader flat, tagger rising) that matters even if the absolute numbers move; the same recipe was used unchanged. Concede: the "model may have seen the public dataset" is not checkable.

**W7. Tables: our trained system is only level with the published baseline (cost: medium).**
60.4 vs RoBERTa's 59.8 on unseen layouts; GraphDoc 75.9 is far above. Rebuttal: GraphDoc uses 932k unlabeled documents, which the zero-same-template rule forbids; our test documents are layout-disjoint and a different subset; we used no DocILE-specific tuning. Concede: we did not beat the field; our contribution is the table-as-link framing, the failure taxonomy, and the finding that a grounded reader cannot do tables.

**W8. Pixels: the positive was not robust (cost: low if stated).**
We registered the rule, passed it on two folds (+3.35), then ran the third fold and reported it failed to hold (+0.83). Say it as a *method strength* ("we ran the third fold and told you it reversed").

**W9. LayoutLMv3 is a confounded comparison to BROS (cost: low).**
No jitter/anchoring. We only claim pages vs blank. We say "14 below BROS" as a reported difference, not a test.

**W10. Linking compares different backbones (cost: medium).**
LiLT vs BROS differ in pre-training and in how they encode geometry. Rebuttal: same joint head, data, windows, seeds, scorer; per-seed gain positive; the claim is "relative layout helps", not "BROS is better in general". BROS is English-only and still wins on a 5-language set, which cuts against a language explanation.

**W11. Two metrics in the same paper (cost: low).**
Exact-span F1 (headline; the strict rule used in every study) and an official-style text match (about 2 points higher; mirrored from `match_utils.py` but the official code was not run). Always say which. Poster numbers 76.5 / 70.5 / 79.2 are exact-span.

**W12. Compute (cost: low).**
The 8B reader at 4-bit takes 5.3 GPU-hours per 300 documents (two T4s summed), about 3 times BROS inference. It is a trade-off the router/vote does not change.

**W13. "Novelty" vs literature (cost: medium).**
LMDX is the recipe; Qwen3-VL is off the shelf. Our novelty is the *crossover curve* (reader vs trained tagger as a function of seen templates), the per-field complementarity, and the table/link generalisation. Do not claim a new model.

---

## 11. Q&A bank

### 11A. The reader (prompts, grounding)

**Q: What exactly is the prompt?** Section 4.6 verbatim. In one line: *the page's OCR lines each tagged with an id (`XX|YY`), the field name and a one-line description, rules to copy the value from ONE line and cite its id, "NONE" if absent; the answer prefilled with `ANSWER: `.* One call per page and field.

**Q: Why one call per field instead of one JSON per page?** The JSON format failed the gate in both runs (0% / 5.5% of values carried a valid line id in v1; 84.4% in v2, still below 90%). One call per field with a prefilled answer was reliable for 3B and 8B. (A 3B model also ignored multi-item tagged formats in study 7.)

**Q: What is "grounding" and why?** A value counts only if its text occurs inside the OCR line it cites. That removes hallucinations, ties each answer to a word span, and lets us score it with the same span metric as BROS. Check: gold values through the same path score 99.7 F1 (so the machinery is not the limit).

**Q: What happens if the model cites a wrong id but quotes the right text?** The registered decode drops it. A post hoc fallback looks the text up in the page (used for the 3B truncated ids and for the numbers we report: 60.5, 76.5). On FATURA a unique-text repair gives +2.3.

**Q: Why Qwen3-VL-8B in 4-bit?** Strongest open model for field extraction in the literature check (Nov 17), and 4-bit nf4 fits a 16 GB T4 so both GPUs each run a shard. The effect of 4-bit loading is not measured.

**Q: Is the reader zero-shot if you tuned the prompt?** See W1 and W2. The prompt was tuned on one set, then replicated on dev and fresh forms with every rule frozen before the run. No model weights were updated on any form.

**Q: Why is the reader worse on Amendment?** BROS is strong there (83-85), where labels resemble the field names and layouts are regular. The reader picks "received" stamps and whole-line copies. It is better where the wording or the decoy defeats BROS (Dissemination +16, Short-Form +9 to +21).

**Q: Why does the reader beat BROS on `foreign_principle_name` so much (55.7 vs 0)?** Each template words it differently (and the Short-Form value hides next to a decoy), so BROS trained on 200 documents never learns a rule; the reader reads the description. Part of the remaining misses are annotation convention (gold includes an extra line such as "Tokyo, Japan").

**Q: What does the image add?** About +3 to +4 F1 in every VLM study (exact numbers in 7.1). Text-only is already above BROS.

**Q: Why not fine-tune the VLM?** We did: LoRA on 200 documents scored 46.2 vs 70.1 for BROS (worse than zero-shot), because it must guess word ranges and overfits 200 documents.

**Q: Why not use GPT-4-class models?** Out of budget and not comparable under a fixed-model protocol; the claim concerns the *crossover*, which depends on the training-set size of the trained system, not on the largest reader. We tested one 3B and one 8B model (scale +3.8 with image).

### 11B. The router and the vote

**Q: How does the router decide?** Per field, pick BROS or the reader by their pooled F1 on the *other two* templates (leave-one-template-out). It never sees the test template. It ends up sending `foreign_principle_name` and `registrant_name` to the reader and the rest to BROS.

**Q: Why not decide per document?** We tried a confidence gate: it gained nothing on VRDU (-0.2) and +1.5 only at one FATURA setting; errors of the two systems are positively correlated, so a confidence signal is weak.

**Q: What is the abstaining vote?** Take the reader with image, the reader without image and BROS; per field choose the span most systems propose; if no two propose exactly the same span, output nothing for that field. 82.4 on the fresh set (+1.5 over the tie-break reader, 12.4 over BROS).

**Q: Why is the vote better than the router?** It does not need fitting; it exploits the fact that errors are not identical. The router tops out at the two-field switch. But the vote gain is small (+1.5) and costs 3 systems of inference.

**Q: Did you test the router on the fresh set?** No: the router was fitted on confirm/dev, so the fresh set would test it; we ran the vote and the tie-break (registered). 79.2 is from confirm.

**Q: Isn't 79.2 vs 82.4 an apples-to-oranges comparison?** Yes: different sets (confirm vs fresh) and different rules. The poster labels each.

### 11C. Training templates

**Q: What is the crossover and why does it matter?** The reader beats a tagger trained on K templates until K is about 5; the tagger overtakes at 10. It tells a deployer when labelling is worth it.

**Q: Why is the reader flat?** It never trains, so it does not change with K; it scores 91.7 whatever K.

**Q: Can't you just train BROS on more templates for VRDU?** VRDU has only three templates; FATURA gives the K sweep.

**Q: Why does DocILE flatten at 60?** The error is tagging (21% of gold spans never touched; 8% typed wrongly) and a document-level label (net vs gross), not the amount of data; the oracle with true fields reaches only 67.

**Q: Why was N = 10 a "training failure"?** At 10 documents an epoch is a handful of steps; the validation F1 was 0 when early stopping fired after 7 epochs. We report it as "did not train".

### 11D. Tables

**Q: How did you incorporate tables?** Section 5. In one line: *the BROS tagger finds cell spans for the line-item fields; an anchor rule attaches each cell to the nearest anchor cell above it; we score row F1 (DocILE's LIR rule).*

**Q: Why not the VLM for tables?** We tried: ad-buy 38.2 vs 68.9-85.7; DocILE pilot 26.6, gate failed. The model cites ids that do not exist (12%) and over-produces rows (1.71x). As a typer on BROS cells it lost 7.45 points.

**Q: Why is the DocILE seen-unseen gap so small?** Rows share structure across layouts; the loss is typing and missed cells (taxonomy). A seed spread of 2.0 to 9.1.

**Q: Did you beat DocILE's best?** No; GraphDoc 75.9 uses 932k unlabeled documents. Our 60.4 on layout-disjoint documents is level with the RoBERTa baseline on unseen layouts (59.8).

**Q: What fails?** Section 5.5 taxonomy; the biggest bucket is a field that is partly tagged (13.6%); gross vs net is a document property.

### 11E. Pixels

**Q: Do pixels help?** In the reader yes (+3 to +4); in a fine-tuned encoder no (+0.8, unstable). Section 7.

**Q: Why did you remove jitter for LayoutLMv3?** Jitter moves boxes but not pixels, so box and image would disagree.

**Q: Why is LayoutLMv3 14 points worse than BROS?** Different pre-training and no jitter/anchoring; we do not claim a clean backbone comparison.

### 11F. Linking

**Q: Why did multilingual hurt?** The windowed inference loses entity precision; single-window link F1 is unchanged. We report the effect, not a mechanism.

**Q: Why is BROS good at linking?** Its relative-position attention encodes geometry between tokens, and most linking errors are geometric (downward, question-to-question links). Note BROS is English-only.

**Q: Why 47.4 and 47.7?** Two LiLT trainings: 47.7 in the earlier run (windowed joint training, study 19) and 47.4 in the later controlled runs. The poster uses 47.4, the like-for-like retrain.

**Q: What was the 65.5?** A mistake: gold entities plus only the links visible in a 512-token window. We do not report it.

### 11G. General

**Q: What is the single most important result?** The crossover: a reader that cites evidence beats training until about 5-10 templates, and a vote of both is best.

**Q: What would you do next?** A learned router (headroom +4 to +6 on FATURA), open-schema decoding, more VLM families, a fresh-template test on a new corpus.

**Q: What did you not check?** Open-schema decoding (the model always got our field list), field names it has never seen, other domains and languages for the reader, other VLM families, sampling/ensembling.

**Q: Ethics / data:** DocILE terms read by the project owner (2026-11-22); weights CC-BY-NC-SA (LayoutLMv3) declared; unlabeled-data methods excluded by the zero-same-template rule.

---

## 12. Hypothesis scorecard (answer to "any hypotheses you proved?")

| Hypothesis (before running) | Verdict |
|---|---|
| H1: A zero-shot reader with a schema beats a trained tagger on unseen wording | **Supported** (76.5 vs 70.5; replicated: 75.2 vs 69.8 dev, 76.5 vs 70.0 fresh); not on boundaries/dates |
| H2: The advantage reverses with training templates | **Supported** (FATURA K = 10: -4.2; K = 40: -8.2) |
| H3: A per-field router combines both | **Partly supported** (+8.7 pooled but -4.6 on Amendment; the vote is more robust) |
| H4: The recipe transfers to tables | **Not supported** (38.2 vs 68.9-85.7; DocILE gate failed) |
| H5: A VLM typer repairs gross/net | **Not supported** (-7.45) |
| H6: A learned pair classifier beats the anchor rule | **Not supported** (mixed: +2.4 / -4.3) |
| H7: Pixels help a fine-tuned encoder on unseen templates | **Not firm** (+3.35 on two folds, +0.83 on three) |
| H8: Template-adversarial training closes the gap | **Not supported** (BROS 69.6 vs 71.3; LiLT 64.8 vs 64.2) |
| H9: Relative-geometry encoders improve linking | **Supported** (+5.9; 3/3 seeds) |
| H10: A multilingual text stream helps linking | **Not supported** (-3.5; 3/3 seeds) |
| H11: A LoRA-tuned VLM beats zero-shot | **Not supported** (46.2 vs 57.4 zero-shot pipeline vs 70.1 BROS) |
| H12: The seen-unseen gap is smaller for tables | **Supported** (2 to 9 vs 21) |

---

## 13. Landmines (numbers and wording that will contradict each other)

1. **76.5 vs 70.5 vs 70.0.** 76.5 is the reader on both confirm *and* fresh; 70.5 is BROS on confirm; 70.0 is BROS on fresh. The poster's abstract uses confirm; point 1 of New Approaches gives both.
2. **60.2 vs 60.5.** Registered decode 60.2; with the post hoc fallback decode 60.5. Say "about 60".
3. **47.4 vs 47.7 vs 44.2 vs 44.22.** LiLT-en link F1 is 47.4 (retrain) or 47.7 (earlier run); multilingual is 44.2. The multilingual delta is "about 3" (3.2 on the chart, 3.5 against its own run).
4. **68.8 vs 68.9.** Ad-buy BROS at N = 10: 68.84 in the file, 68.9 in the log. Poster/report use 68.8-85.7.
5. **65.5 link F1:** removed from poster and report. If you hear it, it is the gold-entity, 512-token figure.
6. **"10 seen templates" vs "K = 5 tie".** The crossover is at about 5; BROS wins clearly at 10. The poster says "wins again once it has seen 10 templates".
7. **DocILE seen 65.9 vs 66.0.** 65.9 appears in an older text; the whole-pool seen figure is 66.0 (68.9 / 63.0 by seed).
8. **Study numbers on slides:** do not show. If asked "which study", say "the registered fresh-forms run".
9. **Router claims:** do not say "the router beats BROS on every template". It does not (Amendment -4.6).
10. **Do not say** "we beat DocILE". We are level with the published RoBERTa baseline on unseen layouts, below GraphDoc.
11. **"Pixels help":** only "in the reader" or "on two folds". Over three folds +0.8.
12. **Demo model:** the Gradio demo runs BROS joint (J1, seed 42); the banner shows the loaded backbone.

---

## 14. Delivery coaching and checklist

- **Lead with the arc, not the studies.** Gap, read-vs-train crossover, combine, tables, pixels, linking, limits.
- **Point at one figure per question:** reader vs BROS bars (reader), FATURA crossover (training templates), DocILE curve (tables), French linking example (linking).
- **Concede first on W1, W2 and W3** before they ask; each one has a prepared rebuttal.
- **Never answer with a study number.** Use "the FATURA sweep", "the fresh-forms run".
- **If asked something not in this file:** say what you measured, what you did not, and what the nearest measured thing is. (E.g. "open-schema: not measured; the closed-schema number is 76.5".)

**Before presenting:**
- [ ] Poster printed at A1, actual size, no scaling. QR codes (repo, demo video) decode.
- [ ] Both of you can state 76.5 / 70.5 / 70.0 / 79.2 / 82.4 and what each is on.
- [ ] Both can draw the FATURA crossover from memory (K = 1, 2 reader; 5 tie; 10, 40 BROS).
- [ ] Both can recite the prompt in one breath (Section 4.6 one-liner).
- [ ] Demo: BROS joint J1 weights in place, `DEMO_CKPT_DIR` set, French form ready.
- [ ] Know that `file_date` and `signer_name` are where the reader is weakest, and `foreign_principle_name` where it is strongest.
