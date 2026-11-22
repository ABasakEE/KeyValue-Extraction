# IE 643 novelty-assessment poster (A1 landscape)

| File | What |
|---|---|
| `poster.tex` | Poster source (beamer + beamerposter, 84.1 × 59.4 cm). Every number has its `results/` source in a comment. |
| `beamerthemeie643.sty` | Theme: one accent colour, Fira Sans (Regular + SemiBold), section titles with a thin rule, the five-line course header. Adapted from Gemini (MIT, `GEMINI_LICENSE.md`). |
| `fonts/` | Fira Sans TTF (SIL OFL, `fonts/OFL.txt`), shared by LaTeX and matplotlib. |
| `scripts/make_plots.py` | F1–F5 from `results/*.json`; prints every plotted value next to its source file. |
| `scripts/make_example.py` | F6, the qualitative Short-Form example (OCR boxes + saved BROS / reader spans). Needs `data/vrdu/` locally. |
| `scripts/make_qr.py` | QR codes as TikZ (`figures/qr_*.tex`). Add the demo-video link there. |
| `figures/` | Logos from the course template, generated PDFs (PNG copies are previews only). |

## Build

```bash
# from the repo root
python poster/scripts/make_plots.py
python poster/scripts/make_example.py
python poster/scripts/make_qr.py
# from poster/
tectonic poster.tex          # XeTeX; downloads packages on first run
```

## Open TODOs (red on the poster)
- Demo-video QR: add the link to `CODES` in `scripts/make_qr.py`, run it, and replace the `\fbox{...\todo{demo QR}}` with `\setlength{\qrunit}{3.9cm/\qrmodules}\input{figures/qr_demo.tex}`.
- Study 42 (LayoutLMv3 on unseen templates): fill in when `results/vrdu_layoutlmv3_summary_*.json` lands. The `best_meta.json` files pushed so far hold validation scores (seen templates), not the unseen-template test F1.
