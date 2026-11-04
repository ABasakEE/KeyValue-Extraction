# demo_pdfs/: documents for the live demo

Upload any file here to `python demo.py` (Gradio). FUNSD and XFUND files come as both an image (`.png` / `.jpg`) and a `.pdf` copy.
All documents are **held out**: FUNSD test forms, XFUND validation forms, and VRDU Registration documents from the unseen-template test folds. None was used to train the checkpoint the demo loads.

## Two groups, and how each was chosen

| Folder | Purpose | How the files were chosen |
|---|---|---|
| `showcase/` | Show what the system does when it works | **Cherry-picked.** Best of the 53 scored FUNSD and XFUND forms (first 25 FUNSD test forms, 28 XFUND candidates) by the demo's own output score (16-D model, word-level pipeline). VRDU files are the best BROS-entity-F1 documents of each template. Say so when presenting. |
| `unseen_template/` | Show what happens in general | **Seeded random draw** (seed 13), not by score: 3 FUNSD forms, 4 XFUND forms, 2 VRDU documents per template (Amendment, Dissemination Report, Short-Form). Some look bad. That is the point. |

## Ground truth

`GROUND_TRUTH.md` (readable) and `ground_truth.json` (machine-readable) list, for every file here, the true entities, the true key-value links (FUNSD and XFUND) or the true field values (VRDU, which has no link labels), plus the entity and link F1 for that file. Use it to check what the demo shows against the answer key.

## File name suffixes

`_g` good, `_m` moderate, `_b` bad (limitations). Assigned from the demo's own text-matched score on that file (entity / link F1): **g** = entity >= 60 and link >= 70; **b** = link < 25 or entity < 35; **m** = everything else. The Squire Patton Boggs VRDU file is `_m` from looking at its output (no gold links exist for VRDU). The other VRDU files have no suffix: the demo has no score for them and I did not inspect them one by one.

## What the demo does (so the output is read correctly)

1. Renders the page (first page of a PDF) and OCRs it with RapidOCR. It gets text *lines* with line boxes, not the dataset's gold words.
2. Cuts each line into words and splits the line box in proportion to word length, so the model gets one token per word, as in training.
3. Runs the trained joint model (`results/xfund_joint_16d/seed_21`, entity F1 79.3 and link F1 67.2 on the XFUND benchmark with gold input) over windows of up to 510 subwords: it tags header / question / answer spans itself, then its biaffine head links key to value (`src/e2e_link.py`, the same code path as the end-to-end evaluation).
4. It does **not** extract the six named VRDU fields and does not run BROS. On VRDU documents it shows generic key-value pairs from the form text. The VRDU entity F1 in the tables below is the BROS benchmark number, not something the demo computes.

The earlier version (`DEMO_PIPELINE=blocks`) fed whole OCR lines to the model as single tokens and used distance rules for links. It is kept as an option.

## Why demo numbers are below the benchmark numbers

The benchmark feeds the model gold words and boxes and, for linking, gold entities (XFUND link F1 65.5, FUNSD 32.05). End to end with the model's own entities, the same checkpoints score about 38 to 48. The demo has no gold input: it OCRs pixels, and OCR lines rarely match annotated blocks. The "demo output vs gold" column is an **approximate text match** (spaces, punctuation and case ignored, similarity >= 0.75, labels must agree), so multi-line blocks count as misses.

## Measured on the cached forms (same OCR; 25 FUNSD + 28 XFUND forms; text-matched)

| Demo version | FUNSD entity / link F1 | XFUND entity / link F1 |
|---|---|---|
| Original demo (FUNSD checkpoint, OCR lines as tokens, rules only) | 45% / 30% | 47% / 30% |
| Now (16-D XFUND model run as trained: word tokens, windows, own tags, trained link head) | 55% / 44% | 63% / 49% |

XFUND link F1 here (49%) is in the same range as the end-to-end benchmark for the same kind of model (44 to 48 on XFUND val), though the scorers differ. Entity F1 is below the benchmark's end-to-end 73 to 76. Dense tables, blank forms and low-quality scans still come out poorly.

## Showcase (cherry-picked)

| File | Source | Demo output vs gold (what you will see) | Benchmark score for the same document |
|---|---|---|---|
| `funsd/82504862_g.pdf` | FUNSD test | entity 77% / link 77% (7 pairs shown, 6 gold) | entity 54% / link 32% (gold entities) |
| `funsd/83443897_g.pdf` | FUNSD test | entity 68% / link 72% (11 pairs shown, 14 gold) | entity 91% / link 40% (gold entities) |
| `funsd/82253245_3247_m.pdf` | FUNSD test | entity 60% / link 61% (34 pairs shown, 41 gold) | entity 84% / link 34% (gold entities) |
| `xfund/it_val_38_g.pdf` | XFUND val (it) | entity 82% / link 90% (20 pairs shown, 18 gold) | entity 100% / link 88% (end to end) |
| `xfund/fr_val_7_g.pdf` | XFUND val (fr) | entity 69% / link 100% (8 pairs shown, 8 gold) | entity 100% / link 98% (end to end) |
| `xfund/de_val_13_g.pdf` | XFUND val (de) | entity 76% / link 88% (8 pairs shown, 8 gold) | entity 96% / link 96% (end to end) |
| `xfund/es_val_38_g.pdf` | XFUND val (es) | entity 80% / link 78% (47 pairs shown, 30 gold) | entity 96% / link 89% (end to end) |
| `vrdu_registration/20140520_Squire Patton Boggs, LLP_Amendment_Amendment_m.pdf` | VRDU Registration (Amendment) | 17 pairs shown (no gold links in VRDU). e.g. 1. Name of Registrant -> 2. Registration No.; 1. Name of Registrant -> Patton Boggs LLP; 2. Registration No. -> 2165 | BROS entity F1 100% on 5 gold fields |
| `vrdu_registration/19911001_Arnold _ Porter Kaye Scholer, LLP_Dissemination Report_Dissemination Report.pdf` | VRDU Registration (Dissemination_Report) | 36 pairs shown (no gold links in VRDU). e.g. 1. Name of registrant -> Arnold & Porter; 2. Registration No. -> 1750; 3. Nature of material (A concise account of the nature of the propaganda material filed) -> United | BROS entity F1 96% on 6 gold fields |
| `vrdu_registration/20111209_Akin, Gump, Strauss, Hauer _ Feld, LLP_Johnson IV, Charles W._Short-Form.pdf` | VRDU Registration (Short-Form) | 20 pairs shown (no gold links in VRDU). e.g. 1. Name -> 2. Registration No.; 1. Name -> Charles W. Johnson IV; 2. Registration No. -> 3492 | BROS entity F1 100% on 4 gold fields |

## Unseen-template / typical (seeded random)

| File | Source | Demo output vs gold (what you will see) | Benchmark score for the same document |
|---|---|---|---|
| `funsd/83635935_m.pdf` | FUNSD test | entity 36% / link 57% (3 pairs shown, 4 gold) | entity 67% / link 80% (gold entities) |
| `funsd/83772145_m.pdf` | FUNSD test | entity 44% / link 40% (14 pairs shown, 11 gold) | entity 72% / link 44% (gold entities) |
| `funsd/89856243_b.pdf` | FUNSD test | entity 32% / link 12% (48 pairs shown, 17 gold) | entity 90% / link 43% (gold entities) |
| `xfund/pt_val_41_m.pdf` | XFUND val (pt) | entity 58% / link 39% (108 pairs shown, 60 gold) | entity 73% / link 32% (end to end) |
| `xfund/it_val_34_m.pdf` | XFUND val (it) | entity 62% / link 25% (121 pairs shown, 58 gold) | entity 80% / link 37% (end to end) |
| `xfund/pt_val_17_b.pdf` | XFUND val (pt) | entity 29% / link 11% (82 pairs shown, 26 gold) | entity 61% / link 39% (end to end) |
| `xfund/pt_val_28_m.pdf` | XFUND val (pt) | entity 84% / link 50% (101 pairs shown, 55 gold) | entity 95% / link 64% (end to end) |
| `vrdu_registration/19980220_KOTRA_Amendment_Amendment.pdf` | VRDU Registration (Amendment) | 25 pairs shown (no gold links in VRDU). e.g. statmn,sotfomrgistrationstatemensuppmntal -> otherdocumentorinfomationfiledwiththeAttoey; 1. Name of Registrant -> Korea Trade Promotion Center,New York; 1. Name of Registrant -> To cor | BROS entity F1 83% on 3 gold fields; missed: signer_name |
| `vrdu_registration/20180308_S.G.R. LLC Government Relations and Lobbying_Amendment_Amendment.pdf` | VRDU Registration (Amendment) | 13 pairs shown (no gold links in VRDU). e.g. 1. Naime of Registrant -> 2. Registration No.; 2. Registration No. -> 6379; 3. This amendment is fied to accomplish the foliowing indicaled purpose or purposes: -> To give a 10-day noti | BROS entity F1 86% on 4 gold fields; missed: signer_name |
| `vrdu_registration/19920401_JETRO, Los Angeles_Dissemination Report_Dissemination Report.pdf` | VRDU Registration (Dissemination_Report) | 63 pairs shown (no gold links in VRDU). e.g. 1. Name of registrant -> 2. Registration No.; 1. Name of registrant -> JETRO,Los Angeles; 2. Registration No. -> 1833 | BROS entity F1 91% on 6 gold fields; missed: foreign_principle_name |
| `vrdu_registration/19931201_JETRO, Los Angeles_Dissemination Report_Dissemination Report.pdf` | VRDU Registration (Dissemination_Report) | 27 pairs shown (no gold links in VRDU). e.g. INSUCTIONS:epotmustbesubmted -> induplicatetothegistrationUnit,IntalScurltyconCrminalivisonDparumontfJuseWashington; INSUCTIONS:epotmustbesubmted -> 23Theoriginalmustbesgndbyronbehalfft | BROS entity F1 63% on 6 gold fields; missed: foreign_principle_name, signer_name, signer_title |
| `vrdu_registration/20101206_India Tourism, New York__Short-Form.pdf` | VRDU Registration (Short-Form) | 34 pairs shown (no gold links in VRDU). e.g. 1.Name -> 2. Registration No.; 1.Name -> MR..SENTHUR KUMARAN VELAYUTHAM; 3. Residence Address(es) -> 43-44. KiSSENA. BLVD ' | BROS entity F1 54% on 4 gold fields; missed: foreign_principle_name, registrant_name |
| `vrdu_registration/20120327_Caribbean Tourism Organization, USA Inc. _Bramble, Sylma Brown_Short-Form.pdf` | VRDU Registration (Short-Form) | 21 pairs shown (no gold links in VRDU). e.g. 1. Name -> Sylma Brown Bramble; 2. Registration No. -> 991; 3. Residence Address(es) -> 14 Fairview Avenue | BROS entity F1 67% on 3 gold fields; missed: registrant_name |

For VRDU the benchmark column is the Track-1 BROS exact-span entity F1 on that document in its unseen-template fold (mean of seeds 13 and 42). The demo does not run BROS.

## Regenerating

`scripts/demo_select_examples.py` (benchmark per-document scores) and `scripts/demo_pipeline_eval.py` (demo-output scores). Demo settings: environment variables `DEMO_CKPT_DIR`, `DEMO_LINK_MODE` (rules | neural | hybrid), `DEMO_CLEAN` (see `src/demo.py`). Sources: FUNSD (`guillaumejaume.github.io/FUNSD`), XFUND val (`github.com/doc-analysis/XFUND` releases), VRDU (`data/vrdu`).
