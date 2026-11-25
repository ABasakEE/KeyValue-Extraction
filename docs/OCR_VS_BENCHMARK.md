# Why the demo scores below the benchmark: the input is not the benchmark's input

Measured 2026-11-25 on the 8 cached XFUND forms (J1 seed 42) and on 10 curated VRDU forms (BROS Track 1, UTL/MTL fold 0, seed 13). Scripts: `scripts/ocr_gap_analysis.py`, `scripts/vrdu_ocr_effect.py`.

## Where each dataset's words and boxes come from (what the benchmark numbers assume)

| Dataset | Source of the words and boxes | Basis |
|---|---|---|
| XFUND | Microsoft Read API OCR tokens, then annotators grouped tokens into entities; annotation granularity is mixed (entity-level for some content, line-level for other) | XFUND paper (ACL Findings 2022); RFUND/SRFUND papers on the granularity |
| FUNSD | crowd-sourced text-detection boxes plus an annotation tool; whether the text was transcribed by hand or OCR'd was not settled by my search | FUNSD paper (arXiv 1905.13538) |
| VRDU | OCR shipped with the dataset (token / line / paragraph / block), kept on purpose so OCR variability is out of the benchmark | Google Research write-up; `data/vrdu/README.md` |
| DocILE | OCR files in docTR's export format (blocks, lines, words, `confidence`, `snapped_geometry`); 200 dpi pages | file schema on disk; the DocILE paper was not re-read |
| FATURA | synthetic: boxes and text come from the generator (exact) | FATURA paper / Zenodo record |

The tagger tabs of `demo_all.py` read dataset examples with the dataset's own words (identical to training). Only uploads go through RapidOCR.

## The ladder on the 8 XFUND forms (link F1)

| Input | F1 |
|---|---|
| gold words and boxes, benchmark rule (exact word ranges) | 67.8 |
| gold words and boxes, the demo's text-matching rule | 68.5 |
| RapidOCR words (demo) | 34.9 |
| docTR words (Hub MobileNet weights) | 35.7 |

The scorer is not the cause. Resolution is not the cause (RapidOCR at 3x render size: word recall 70% vs 70%).

## What the OCR output contains (515 gold entities)

| OCR | found exactly | found ignoring accents | scattered | partial | missing | gold-word recall |
|---|---|---|---|---|---|---|
| RapidOCR | 54% | 68% | 6% | 15% | 10% | 70% (78% ignoring accents) |
| docTR | 55% | 72% | 10% | 9% | 9% | 82% (90%) |

(found / scattered / partial / missing are the accent-insensitive rows after the first column.) BROS's tokenizer is uncased and the demo's pair matcher accepts a one-letter difference, so accents explain the exact-match loss better than the demo's F1 loss.

## VRDU tagger, field-level text match against gold (10 curated forms)

| Input | micro F1 |
|---|---|
| dataset OCR | 96.6 |
| RapidOCR | 84.7 |

The loss concentrates on the unseen-template forms (100/100/100/86/86 with dataset OCR; 75/67/67/33/100 with RapidOCR). The forms are curated (good, typical and failure cases), so the absolute values are not a benchmark.

## Reading

Published numbers use the dataset's own words (and, for relation extraction, I believe, the gold entities, which I did not re-check). Our end-to-end numbers (53.4 on XFUND) already drop the gold entities. The demo then replaces the gold words with OCR output, which contains only about 70-80% of them, and loses about half of the link F1 on these forms. Not tested: DeepSeek-OCR (needs a GPU, no word boxes), docTR's larger default models (their download host redirects to GitHub release assets, blocked on this network), OCR on FUNSD, FATURA and DocILE uploads.
