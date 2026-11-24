# demo_pdfs/tagger: examples for `python demo_all.py`

Documents of the three tagger datasets, chosen by the demo's own checkpoints (not by file order): a mix of good, moderate and failing cases, so a viewer sees what the model gets right and wrong. All are **held out** from the checkpoint they are paired with.

**Using them:** the dropdown in `demo_all.py` lists the same documents and scores them with the dataset's own words and boxes (what the model was trained on). Uploading a file from this folder instead reads it with RapidOCR, so the scores can differ from the dropdown labels and no gold comparison is shown.

File suffix: `_g` good (F1 >= 90), `_m` moderate (50 to 90), `_b` bad (< 50); the score behind it is the one printed below.
Entity F1 is exact-span on the dataset's words; DocILE rows use the line-item row F1.

## VRDU registration forms (BROS Track 1, six fields)

| File | Template | Checkpoint | Entity F1 on this form | Gold fields (exact text) |
|---|---|---|---|---|
| `vrdu/unseen_template/20090930_Garvey, Schubert _ Barer_Amendment_Amendment_g.pdf` | Amendment | unseen-template model (Amendment held out) | 100% | registrant_name: Garvey Schubert Barer; registration_num: 3047; file_date: September 30 , 2009; signer_name: Harold G. Bailey , Jr. |
| `vrdu/unseen_template/20160720_Glover Park Group, LLC_Amendment_Amendment_g.pdf` | Amendment | unseen-template model (Amendment held out) | 100% | registrant_name: The Glover Park Group , LLC; registration_num: 5666; file_date: July 20 , 2016; signer_name: Joel Johnson |
| `vrdu/unseen_template/20030331_Hogan Lovells US LLP_Amendment_Amendment_g.pdf` | Amendment | unseen-template model (Amendment held out) | 100% | registrant_name: Hogan & Hartson L.L.P.; registration_num: 2244; file_date: March 31 , 2003; signer_name: Mark S. McConnell |
| `vrdu/unseen_template/19991228_Crowell _ Moring International, Ltd._Amendment_Amendment_m.pdf` | Amendment | unseen-template model (Amendment held out) | 86% | registrant_name: Crowell & Moring International , Ltd.; registration_num: 3988; signer_name: Christopher S. Wilson; file_date: 28th day of December .199 |
| `vrdu/unseen_template/20050323_Arnold _ Porter Kaye Scholer, LLP_Amendment_Amendment_m.pdf` | Amendment | unseen-template model (Amendment held out) | 86% | registration_num: 1750; registrant_name: Arnold & Porter LLP; signer_name: Mark H. Stumpf; file_date: March 23 , 2005 |
| `vrdu/unseen_template/19850701_Squire Patton Boggs, LLP_Amendment_Amendment_b.pdf` | Amendment | unseen-template model (Amendment held out) | 25% | registrant_name: Patton , Boggs & Blow; registration_num: 2165; signer_name: imothy J. May , Esquire; signer_title: Managing Partner; file_date: Washington |
| `vrdu/seen_template/20130628_Picard Kentz _ Rowe, LLP_Amendment_Amendment_g.pdf` | Amendment | seen-template model | 100% | registrant_name: Picard Kentz & Rowe LLP; registration_num: 5939; file_date: June 28 , 2013; signer_name: B.Donovan Picard |
| `vrdu/seen_template/19760701_Tourism Authority of Thailand, New York_Amendment_Amendment_m.pdf` | Amendment | seen-template model | 89% | registration_num: 1897; registrant_name: Tourist Organization of Thailand; signer_name: Staporn Sirisinha; signer_title: Assistant - Chief , T.O.T. , N.Y. Office; file_date: 16 day of |
| `vrdu/seen_template/19920801_JETRO, New York_Dissemination Report_Dissemination Report_g.pdf` | Dissemination_Report | seen-template model | 100% | registrant_name: JETRO , New York; registration_num: 1643; foreign_principle_name: Japan External Trade Organization; file_date: August 17 , 1992; signer_name: Kunio Ito; signer_title: Director , Public Affairs |
| `vrdu/seen_template/19920701_JETRO, Houston_Dissemination Report_Dissemination Report_g.pdf` | Dissemination_Report | seen-template model | 100% | registrant_name: JETRO , Houston; registration_num: 2277; foreign_principle_name: Japan External Trade Organization Tokyo , Japan; signer_name: Hisashi Yamada; file_date: July 8 , 1991; signer_title: Director , Public Relations |
| `vrdu/seen_template/20180308_Akin, Gump, Strauss, Hauer _ Feld, LLP_Verhoff, Geoff_Short-Form_g.pdf` | Short-Form | seen-template model | 100% | registration_num: 3492; registrant_name: Akin Gump Strauss Hauer & Feld LLP; foreign_principle_name: Public Investment Fund; signer_name: Geoff Verhoff; file_date: March 08 , 2018 |
| `vrdu/seen_template/20110203_BGR Government Affairs, LLC_Birnbaum, Jeffrey_Short-Form_g.pdf` | Short-Form | seen-template model | 100% | registration_num: 5430; registrant_name: Barbour , Griffith & Rogers , LLC d / b / a BGR Government Affairs; foreign_principle_name: Republic of India; file_date: 2 |

## FATURA invoices (BROS Track 1, nine fields)

`unseen_template/`: invoices of fold-0 held-out templates (never seen by either model). `seen_template/`: other invoices of the two templates the K = 2 model trained on (also among the 40 the K = 40 model trained on); none of these invoices was a training or validation invoice. The suffix follows the K = 2 model; both scores are listed. The files are PDFs made from the dataset's JPG invoices.

| File | Template | K = 40 entity F1 | K = 2 entity F1 | Gold fields (exact text) |
|---|---|---|---|---|
| `fatura/unseen_template/Template2_Instance74_m.pdf` | Template 2 (unseen) | 100% | 71% | title: TAX INVOICE; number: INVOICE # 7Y9M4d-818; date: Date: 17-Jun-2014; seller_name: Barnes, Garcia and Martin; seller_address: Address:13868 Michael Wall North Cindyto; buyer: Buyer :Karen Manning 679 Brown Haven Sui; sub_total: SUB_TOTAL : 881.76 USD; total: TOTAL : 931.61 USD |
| `fatura/unseen_template/Template4_Instance148_m.pdf` | Template 4 (unseen) | 100% | 57% | seller_name: Huff-Bryan; seller_address: Address:2180 Michael Ridges Apt. 385 Por; title: COMMERCIAL INVOICE; total: TOTAL : 1168.66 EUR; due_date: Due Date : 23-Mar-2018; date: Date: 24-Apr-2004; sub_total: SUB_TOTAL : 1157.45 EUR |
| `fatura/unseen_template/Template7_Instance59_m.pdf` | Template 7 (unseen) | 100% | 71% | seller_name: White and Sons; title: INVOICE; number: Invoice number 9696-199; buyer: Buyer :Oscar Roberts 74227 Gabriel Ville; date: Invoice Date: 10-Jan-2023; due_date: Due Date : 16-Jan-2009; sub_total: SUB_TOTAL : 661.24 $; total: TOTAL : 677.84 $ |
| `fatura/unseen_template/Template10_Instance170_b.pdf` | Template 10 (unseen) | 100% | 31% | seller_address: Address:4894 Melanie Pass Apt. 742 Tammy; date: Invoice Date: 26-Aug-2018; number: Invoice number INV/26-68/435; seller_name: Mclean-Cochran; due_date: Due Date : 23-May-2018; buyer: Bill to:James Smith 456 Cox Mall Apt. 43; total: TOTAL : 40.29 EUR |
| `fatura/unseen_template/Template12_Instance44_b.pdf` | Template 12 (unseen) | 100% | 43% | title: TAX INVOICE; number: INVOICE # INV/39-45/178; date: Date: 17-Nov-2005; due_date: Due Date : 13-Jul-2000; buyer: Buyer :Nicholas Holmes 3590 Kyle Locks S; seller_name: Richardson-Davis; seller_address: Address:6479 Smith Causeway East Cameron; sub_total: SUB_TOTAL : 751.85 $; total: TOTAL : 776.92 $ |
| `fatura/unseen_template/Template25_Instance125_m.pdf` | Template 25 (unseen) | 100% | 77% | buyer: Buyer :Brent Lamb 3124 Robert Cape Miran; seller_name: Kelly-Moss; seller_address: Address:6030 Willie Shores Suite 354 Gue; number: Invoice number INV67289395; due_date: Due Date : 31-Mar-2016; date: Date: 06-Dec-2020; sub_total: SUB_TOTAL : 758.83 USD; total: TOTAL : 774.97 USD |
| `fatura/seen_template/Template15_Instance0_g.pdf` | Template 15 (seen) | 100% | 100% | title: INVOICE; date: Date: 27-Aug-1993; seller_address: Address:744 Hernandez Hollow Apt. 459 Ro; number: INVOICE ID INV38499489; due_date: Due Date : 16-Nov-2006; buyer: Buyer :Megan Smith 744 Hernandez Hollow ; sub_total: SUB_TOTAL : 1057.19 USD |
| `fatura/seen_template/Template15_Instance2_g.pdf` | Template 15 (seen) | 100% | 100% | title: TAX INVOICE; date: Invoice Date: 19-Dec-2003; seller_address: Address:744 Hernandez Hollow Apt. 459 Ro; number: INVOICE ID INV/95-26/957; due_date: Due Date : 22-Feb-2007; buyer: Bill to:Rachel Brandt 57355 Haynes Islan; sub_total: SUB_TOTAL : 730.06 USD |
| `fatura/seen_template/Template36_Instance0_g.pdf` | Template 36 (seen) | 100% | 100% | buyer: Bill to:Michael Green 047 Jackson Meadow; seller_address: Address:047 Jackson Meadows Apt. 391 Pea; title: INVOICE; number: INVOICE # INV37478055; date: Invoice Date: 02-Aug-2021; sub_total: SUB_TOTAL : 1113.41 USD |
| `fatura/seen_template/Template36_Instance1_g.pdf` | Template 36 (seen) | 100% | 100% | buyer: Bill to:Tammy Fisher 93159 Anderson Cros; seller_address: Address:047 Jackson Meadows Apt. 391 Pea; title: COMMERCIAL INVOICE; number: INVOICE # 5811-948; date: Date: 27-Jul-1999; sub_total: SUB_TOTAL : 399.25 EUR |

## DocILE line items (BROS Track 1, 19 line-item fields; N = 1,000 split seed 0)

`unseen_layout/`: test documents whose layout and vendor appear in no training document. `seen_layout/`: documents that were NOT in the 1,000 training documents but share a layout (cluster) with some of them. Local only: these are DocILE documents and are not committed (see .gitignore).

| File | Row F1 on this document (this checkpoint) | Gold rows |
|---|---|---|
| `docile/unseen_layout/8bc8897b17ac474a846f6a77_g.pdf` | 100% | 12 rows; first: description: COST OF MENU; amount_gross: $11,250.00 |
| `docile/unseen_layout/9ebdec30843146c584f0a713_g.pdf` | 100% | 5 rows; first: date: 04/22/20 |
| `docile/unseen_layout/ecd1566f99f5422a88f09cba_g.pdf` | 100% | 3 rows; first: description: 2001 Camel Biker Program Print |
| `docile/unseen_layout/c3fb1ae263a34aab94da6f6e_m.pdf` | 64% | 6 rows; first: description: 4-1 Large 5 point twisters - r; amount_gross: $1400.00 |
| `docile/unseen_layout/b434fe4846e64ce48784453f_m.pdf` | 67% | 9 rows; first: date: 04/12/00; amount_gross: 1,413.75 |
| `docile/unseen_layout/685f7be21127477583114af0_b.pdf` | 0% | 3 rows; first: description: Sample size: |
| `docile/seen_layout/fba13e6e07144f8f9056f4ee_g.pdf` | 100% | 3 rows; first: position: 1; unit_price_gross: $20; quantity: 8 |
| `docile/seen_layout/ea05d3675a7944c19a95b2be_g.pdf` | 100% | 5 rows; first: position: 1; date: 10/04/22-10/1022; unit_price_gross: 120.00; quantity: 15; amount_gross: 1,800.00 |
| `docile/seen_layout/72a7e1a870f040b38b6df8e0_m.pdf` | 84% | 7 rows; first: date: 4/20/2021; description: 1:00WABK/WBAK 09:20:12 AM 03:3; quantity: 2; unit_price_gross: $100.00 |
| `docile/seen_layout/4a9d2670f8854db5af301fa0_m.pdf` | 86% | 3 rows; first: description: Spot; quantity: 0; amount_net: $0.00; tax: $0.00; amount_gross: $0.00 |
| `docile/seen_layout/5dd535c7d6e54fd8bb8d077a_b.pdf` | 44% | 3 rows; first: date: Mo, 7/6/20; quantity: 60; code: 20TGIA0008R; amount_gross: $614.00; description: Hurt R |

## XFUND forms (BROS joint tagger + linker, J1 seed 42; entities and key-value links)

The forms the demo's linking tab shows: XFUND validation forms in five languages (about 94% have no training form of the same template). Score = link F1 of J1 through the demo pipeline (RapidOCR words, the model's own entities, text-matched pairs): `_g` >= 60, `_m` 25 to 60, `_b` < 25. This is lower than the 53.4 end-to-end benchmark average because the demo reads the page with OCR and matches pairs by text. The `_g/_m/_b` suffixes of the older `demo_pdfs/showcase` and `unseen_template` copies came from an earlier LiLT demo and are superseded by these.

| File | Language | Link F1 (J1, this form) | Pairs predicted / gold | Gold key-value pairs (first three) |
|---|---|---|---|---|
| `xfund/de_val_13_g.pdf` | de | 67% | 7 / 8 | Vorname, Name: -> Helen Schmitz; Adresse: -> Tauberfranken 8; PLZ, Ort: -> 97922 Lauda-Königshofen |
| `xfund/it_val_38_g.pdf` | it | 64% | 14 / 17 | Responsabile documento *: -> Rivieri Michele; Redazione *: -> de Simone; Redazione *: -> Carlotta |
| `xfund/fr_val_7_g.pdf` | fr | 62% | 8 / 8 | Civilité : -> Madame; Nom : -> Robert; Prénom : -> Katel |
| `xfund/it_val_34_m.pdf` | it | 31% | 86 / 55 | Code: -> AbbPar; Ed. 01 -> Rev. 00-09/2009; Pagina: -> 1 di 2 |
| `xfund/pt_val_28_b.pdf` | pt | 25% | 106 / 48 | Proponente: -> Embraer S.A.; CNPJ: -> 17.115.437/0001-73; Endereço: -> Vila Coelho Xavier 1973 |
| `xfund/pt_val_41_b.pdf` | pt | 12% | 39 / 60 | INFORMAÇÕES SOBRE O CANDIDATO -> Nome Completo:; INFORMAÇÕES SOBRE O CANDIDATO -> RG:; INFORMAÇÕES SOBRE O CANDIDATO -> Local e Data de Nascimento: |
| `xfund/es_val_38_b.pdf` | es | 11% | 42 / 30 | Información del miembro/paciente: -> Nombre completo; Información del miembro/paciente: -> Fecha de nacimiento; Información del miembro/paciente: -> N |
| `xfund/pt_val_17_b.pdf` | pt | 6% | 40 / 24 | Agência -> MC; CPF/CNPJ/MF -> 881.334.667-13; Código da Agência -> MC |
