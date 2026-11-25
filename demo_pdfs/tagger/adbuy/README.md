# demo_pdfs/tagger/adbuy: VRDU Ad-Buy line items (saved predictions, not a live model)

Ad-Buy political-ad invoices (641 documents, dozens of templates, five line-item fields: channel, program description, start date, end date, amount). The files are documents of templates the model did not train on: Track-1 BROS trained on 200 documents (`N200_SD0`, training seed 13), cells grouped into rows by the anchor rule. `_g` = row F1 on this document >= 90, `_m` = 50 to 90. Each PDF has a preview PNG in `previews/` with the predicted cells (coloured by field) and numbered rows; dashed blue boxes are gold cells that were missed.

**Not live:** no Ad-Buy checkpoint was kept (`save_weights: false`), so `demo_all.py` cannot run these; train one with `notebooks/kaggle_adbuy_bros.ipynb` after setting `save_weights: true` if you need that. Over the 143 documents scored in every split this run averages 81% per-document row F1 (the study's pooled figure for N = 200 is 85.7 over three splits).

| File | Row F1 (this document) | Gold rows / predicted rows | Gold rows (first two) |
|---|---|---|---|
| `50e20ed6-1232-4272-a28a-4632d68679fc_g.pdf` | 94% | 19 / 19 | program_desc: Su AM News, program_start_date: 03 / 15 /, program_end_date: / 22 / 20, sub_amount: $ 60.00, channel: MFD / KF ( MFD_KF ); program_desc: Su 5PM News, program_start_date: 03 / 15 /, program_end_date: / 22 / 20, sub_amount: $ 80.00, channel: MFD / KF ( MFD_KF ) |
| `463974-kovr-nrcc-r-multi-ord33720-fed-natl-contract_g.pdf` | 99% | 10 / 10 | program_start_date: 09 / 23 /, program_end_date: / 23 / 12, program_desc: CBS THIS MORNING- SUND, sub_amount: 4500.00; program_start_date: 09 / 21 /, program_end_date: / 21 / 12, program_desc: DR PHIL, sub_amount: 1000.00 |
| `a16f110d-e5c1-1af3-5b26-2553147b5027_g.pdf` | 100% | 5 / 5 | sub_amount: $ 125.00, program_start_date: 03/23/20; sub_amount: $ 125.00, program_start_date: 03/23/20 |
| `dee95bc6-913c-2935-7c65-137add174212_m.pdf` | 52% | 8 / 8 | channel: WJFW, program_desc: Today Show |, program_start_date: 03/30/20, program_end_date: 04/05/20; channel: WJFW, program_desc: Newswatch 12 Morning N, program_start_date: 03/30/20, program_end_date: 04/05/20 |
| `493496-315597-up-obama-10-30-12-13516185156783-_-pdf_m.pdf` | 72% | 11 / 23 | sub_amount: $ 20,000.00, channel: WEWS, program_start_date: 10/24/12, program_end_date: 11/06/12, program_desc: Tuesday Prime B Tuesda; channel: WEWS, program_start_date: 10/24/12, program_end_date: 11/06/12, program_desc: Wednesday Prime A Wedn, sub_amount: $ 32,000.00 |
| `717f24ba-51d7-3daa-89bc-d811e5be1f3a_m.pdf` | 89% | 18 / 18 | program_start_date: 04/30/20, program_desc: News - WGXA Morning Ne, sub_amount: $ 15.00; program_start_date: 04/28/20, program_desc: Couples Court, sub_amount: $ 15.00 |
