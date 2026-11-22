# Arjoe handoff: poster and 2-page report

Written 22 Nov 2026 by Param, with Claude, in reply to `PARAM_HANDOFF.md`. Both deliverables are built from the repo on branch `feat/bros-relative-layout`. **Deadline for upload: tonight, 22 Nov, 11:59 PM IST**, to the shared Drive folder. The poster is printed on A1 and presented on 23 or 25 Nov.

## What exists

| Deliverable | PDF | Source |
|---|---|---|
| Poster, A1 landscape (84.1 × 59.4 cm), 1 page | `poster/Unemployed_and_Unsupervised_IE643_CourseProject_Poster.pdf` | `poster/poster.tex`, theme `poster/beamerthemeie643.sty` |
| Novelty report, exactly 2 pages, A4 | `novelty_report/Unemployed_and_Unsupervised_IE643_NoveltyAssessment_report.pdf` | `novelty_report/report.tex` |

Rename both PDFs if the Drive folder uses a different spelling of the team name (`Teamname_IE643_...`).

Figures are regenerated from `results/` by scripts, never typed in:
- `poster/scripts/make_plots.py`: VRDU per-template bars, FATURA crossover, XFUND link F1, DocILE curve, DocILE taxonomy. It prints every plotted value next to its source file.
- `poster/scripts/make_example.py`: the qualitative Short-Form example (needs `data/vrdu/` locally).
- `poster/scripts/make_qr.py`: QR codes as TikZ.

Every number in `poster.tex` has its source file in a comment next to it.

## How to rebuild

```bash
python poster/scripts/make_plots.py      # from the repo root
python poster/scripts/make_example.py
python poster/scripts/make_qr.py
cd poster && tectonic poster.tex          # XeTeX; downloads packages on first run
cd ../novelty_report && tectonic report.tex
```

Tectonic is a single binary (github.com/tectonic-typesetting/tectonic, releases). Fonts are vendored in `poster/fonts/` (Fira Sans, OFL), so nothing needs installing. `poster/README.md` has more detail.

## Open items (red TODO boxes on the poster)

1. **Demo video QR.** No link exists yet. Record the Gradio demo, then:
   - add the URL to `CODES` in `poster/scripts/make_qr.py` and run it;
   - in `poster.tex`, replace `\fbox{\parbox[c][3.7cm][c]{3.7cm}{...\todo{demo QR}}}` with `\setlength{\qrunit}{3.9cm/\qrmodules}\input{figures/qr_demo.tex}`;
   - rebuild.
2. **Study 42, LayoutLMv3 on unseen templates.** This is in the poster's Conclusion and in report section 2.5.
   - What the Kaggle runner pushed today is `best_meta.json` files. Those hold `best_val_score`, a validation score on seen templates, not the unseen-template test F1, so I didn't use them.
   - When the `results/vrdu_layoutlmv3_summary_*.json` files land, replace `\todo{S42}` in `poster.tex` and the "still going at submission" sentence in `report.tex`.
3. **Check two reference author lists against the paper pages:** FATURA (Limam, Dhiaf, Kessentini) and XFUND (Xu, Lv, Cui, et al., Findings of ACL 2022). The rest I'm confident in.
4. **Check that the repo is public.** The GitHub QR points to `https://github.com/ABasakEE/KeyValue-Extraction/tree/feat/bros-relative-layout`.
5. **Nothing is committed yet:** `poster/`, `novelty_report/` and this file are untracked on `feat/bros-relative-layout`. Commit and push when you're happy with them.

## Numbers I changed from PARAM_HANDOFF.md (checked against the raw files)

| Handoff said | Poster/report use | Why |
|---|---|---|
| Ad-buy BROS row F1 68.9 at N = 10 | **68.8** | `results/adbuy_bros/run/report.json` gives 68.84. The 68.9 came from `adbuy_sg/run/comparison.json`, a 128-document subset. |
| DocILE seen layouts 66.0 | **65.9** | mean of 68.87 and 63.02 is 65.945 |
| XFUND "47.7 → 53.4" | **47.4 → 53.4** | 47.7 (Study 24) and 47.42 (Study 37) are separate LiLT runs. The +5.9 is measured against 47.42. |
| Study 25 8B "60.2" | **60.5** | `results/study25_stage0/8B_A2_confirm/metrics.json` (0.6047). STUDIES.md row 25 still says 60.2 and should be fixed. |
| Section 5.2 table "76.5 / 75.2 on dev/confirm" | 76.5 = confirm, 75.2 = dev | the order was reversed there |

LiLT 91.6/64.2 and the graph network 86.0/42.1 exist only in `results/master_architectural_benchmark.csv`, with no per-run metrics file. They carry an asterisk on the poster. The CSV calls the graph model "Standalone RSA-GNN", which is the name the poster uses.

## Questions to expect at the poster

- **The prompt-v2 router fails its own registered rule.** It gains +8.7 [+6.9, +10.6] pooled but loses 4.6 on Amendment, a significant loss (`results/schema_guided_kv_v2/hybrid_V2A2.json`, `verdict.primary.passes = false`). The poster shows the −4.6. The earlier 3B router (Study 12, +3.8 [+2.6, +5.0]) does pass.
- **The prompt was tuned on dev/confirm.** Study 32 (300 fresh forms) never ran. Both the poster and the report say so.
- **BROS is English-only on multilingual XFUND.** The +5.9 appears despite that. Study 37 is one recipe on one dataset.
- **DocILE vs the published 59.8 RoBERTa baseline:** different test documents, so the comparison is rough.

## What was cut for space

- The poster dropped the DocILE taxonomy bar chart; the "21.4% missed" figure moved into a caption.
- The poster dropped the LMDX reference; the report keeps it.
- The report's table covers 13 study groups; the full log stays in `STUDIES.md`.
