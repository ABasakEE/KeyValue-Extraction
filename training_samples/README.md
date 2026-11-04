# training_samples/: documents from the training datasets, grouped by task

A few documents per task, picked at random (seed 13), so you can show what the models were trained on. They are files only: the labels live in `data/` (paths below). For the held-out examples used in the live demo, see `demo_pdfs/` and its `GROUND_TRUTH.md`.

FUNSD and FATURA also come with the original image next to the PDF copy, and synthetic forms with their PNG. The VRDU files are the original PDFs.


## Entity extraction under template shift

VRDU Registration (FARA), the main benchmark. Six fields per document (file_date, foreign_principle_name, registrant_name, registration_num, signer_name, signer_title). Three templates, listed by folder. A document is training data in every fold where its own template is not the held-out one, and test data in the fold where it is. Annotations: `data/vrdu/registration-form/main/dataset.jsonl.gz`.

| File | Dataset | Note |
|---|---|---|
| `1_entity_extraction_vrdu_registration/Amendment/20060127_Hogan Lovells US LLP_Amendment_Amendment.pdf` | VRDU Registration (FARA) | template Amendment |
| `1_entity_extraction_vrdu_registration/Amendment/20080507_British Virgin Islands Tourist Board_Amendment_Amendment.pdf` | VRDU Registration (FARA) | template Amendment |
| `1_entity_extraction_vrdu_registration/Amendment/20180711_Representative of the Turkish Republic of Northern Cyprus_Amendment_Amendment.pdf` | VRDU Registration (FARA) | template Amendment |
| `1_entity_extraction_vrdu_registration/Dissemination_Report/19940914_JETRO, Los Angeles_Dissemination Report_Dissemination Report.pdf` | VRDU Registration (FARA) | template Dissemination_Report |
| `1_entity_extraction_vrdu_registration/Dissemination_Report/19960116_JETRO, Houston_Dissemination Report_Dissemination Report.pdf` | VRDU Registration (FARA) | template Dissemination_Report |
| `1_entity_extraction_vrdu_registration/Dissemination_Report/19920101_Scottish Enterprise_Dissemination Report_Dissemination Report.pdf` | VRDU Registration (FARA) | template Dissemination_Report |
| `1_entity_extraction_vrdu_registration/Short-Form/20171212_Beckerman_Paul, Jordan_Short-Form.pdf` | VRDU Registration (FARA) | template Short-Form |
| `1_entity_extraction_vrdu_registration/Short-Form/20120606_Daniel J. Edelman, Inc._Dreyer, Kate_Short-Form.pdf` | VRDU Registration (FARA) | template Short-Form |
| `1_entity_extraction_vrdu_registration/Short-Form/20180131_BGR Government Affairs, LLC_Osborne, Jason_Short-Form.pdf` | VRDU Registration (FARA) | template Short-Form |

## Entity extraction, control corpus

VRDU Ad-Buy (DeepForm), political advertising contracts, nine fields. Templates are not labelled; we use VRDU's own unseen-template split files. Annotations: `data/vrdu/ad-buy-form/main/dataset.jsonl.gz`.

| File | Dataset | Note |
|---|---|---|
| `2_entity_extraction_vrdu_adbuy/3eb2feea-8655-10c2-3964-ba69eead8ac9.pdf` | VRDU Ad-Buy (DeepForm) | template not labelled |
| `2_entity_extraction_vrdu_adbuy/4e1c4012-34b3-a73b-c40c-85f1bd0c4b32.pdf` | VRDU Ad-Buy (DeepForm) | template not labelled |
| `2_entity_extraction_vrdu_adbuy/462499-wbz-nrcc-richard-tisei-housemacd6-ord58654-fed.pdf` | VRDU Ad-Buy (DeepForm) | template not labelled |
| `2_entity_extraction_vrdu_adbuy/34e3acdc-2c6b-1d50-ebc1-973d2ff6ca09.pdf` | VRDU Ad-Buy (DeepForm) | template not labelled |

## Key-value linking (training split of FUNSD)

FUNSD training forms: header / question / answer entities with annotated key-to-value links. No template labels, so no seen/unseen split. Annotations: `data/funsd/dataset/training_data/annotations/<id>.json`.

| File | Dataset | Note |
|---|---|---|
| `3_key_value_linking_funsd/00093726.pdf` | FUNSD train | question/answer/header entities with links |
| `3_key_value_linking_funsd/92094751.pdf` | FUNSD train | question/answer/header entities with links |
| `3_key_value_linking_funsd/0060214859.pdf` | FUNSD train | question/answer/header entities with links |
| `3_key_value_linking_funsd/01073843.pdf` | FUNSD train | question/answer/header entities with links |

## Second schema for the router study

FATURA synthetic invoices from templates in the official train split (`Strat2_Split.txt`). Nine core fields, block-level boxes. Annotations: `data/fatura/raw/invoices_dataset_final/Annotations/`.

| File | Dataset | Note |
|---|---|---|
| `4_second_schema_fatura/Template3_Instance3.pdf` | FATURA (synthetic invoices) | train template 3 |
| `4_second_schema_fatura/Template35_Instance70.pdf` | FATURA (synthetic invoices) | train template 35 |
| `4_second_schema_fatura/Template14_Instance37.pdf` | FATURA (synthetic invoices) | train template 14 |
| `4_second_schema_fatura/Template47_Instance21.pdf` | FATURA (synthetic invoices) | train template 47 |

## Pipeline sanity check

Our own generated forms, one per family (invoice, financial statement, procurement order, shipping manifest, medical intake, ad-buy contract, FARA registration), with scanner noise added. Train and test share the same families, so scores on them are a code check, not evidence of generalisation. Generator: `src/data/synthetic_generator.py`.

| File | Dataset | Note |
|---|---|---|
| `5_synthetic_pipeline_check/syn_adbuy_0_0005.pdf` | Synthetic (ours) | one of seven generated form families |
| `5_synthetic_pipeline_check/syn_fara_0_0006.pdf` | Synthetic (ours) | one of seven generated form families |
| `5_synthetic_pipeline_check/syn_fin_statement_0_0001.pdf` | Synthetic (ours) | one of seven generated form families |
| `5_synthetic_pipeline_check/syn_intake_0_0004.pdf` | Synthetic (ours) | one of seven generated form families |
| `5_synthetic_pipeline_check/syn_invoice_0_0000.pdf` | Synthetic (ours) | one of seven generated form families |
| `5_synthetic_pipeline_check/syn_manifest_0_0003.pdf` | Synthetic (ours) | one of seven generated form families |
| `5_synthetic_pipeline_check/syn_procure_0_0002.pdf` | Synthetic (ours) | one of seven generated form families |

## Not included: XFUND training forms

The XFUND training images (`fr.train.zip` and the other languages) would belong under task 3, but the download failed on this machine: the release host closed the TLS connection for those files, while the validation zips downloaded fine. The XFUND annotation files are already in `data/xfund/`. To add the images, download `https://github.com/doc-analysis/XFUND/releases/download/v1.0/<lang>.train.zip` from a browser and put a few forms in `3_key_value_linking_xfund/`.

## Note on "training" documents

For VRDU, the same PDF is training data in some folds and test data in another, so these are examples of the training distribution, not a separate set. The files here were chosen not to overlap with the held-out demo files in `demo_pdfs/`.
