# Ground truth for every file in demo_pdfs/

True entities and key-value links from the dataset annotations, so you can compare them with what the demo shows. `(question) key -> (answer) value` is a gold link, listed as annotated. FUNSD and XFUND links include the dataset's own annotation quirks, for example heading-to-field links. VRDU has **no link labels**, only the true field values; for those files the BROS prediction is listed next to them.

Scores per file: *demo* = what the Gradio demo scores against these gold labels (approximate text match); *benchmark* = the trained model's per-document score with gold input. File suffix `_g` good, `_m` moderate, `_b` bad.


## showcase


### funsd

#### `82253245_3247_m.pdf`  [m]

- Scores: demo entity 60% / link 61% (34 pairs shown, 41 gold); benchmark entity 84% / link 34% (gold entities)
- Gold entities: 70 (answer 37, header 2, other 6, question 25); gold links: 41
- True links:
  - (question) TO: -> (answer) K. A. Sparrow
  - (question) FROM: -> (answer) R. E. Lane
  - (question) SELL -> (answer) Overall pre- sell efforts were successful Retail accounts that previously stocked
  - (question) SELL -> (answer) Style Full Price accepted the introduction of the low price
  - (header) SUBMISSION DATE -> (question) JUNE 30
  - (header) SUBMISSION DATE -> (question) AUG 11
  - (header) SUBMISSION DATE -> (question) SEP 22
  - (header) SUBMISSION DATE -> (question) NOV 10
  - (question) JUNE 30 -> (answer) X
  - (question) BIGIF 2 FOR 1: -> (answer) Proved to be an excellent tool for pulling the balance of Style Full Price packs through
  - (question) BIGIF 2 FOR 1: -> (answer) the system. This aided the field greatly during the transition
  - (question) $ 7.00 CARTON COUPON/ BUYDOWN: -> (answer) Effective in those retail calls that we could not exchange product out of.
  - (question) $ 7.00 CARTON COUPON/ BUYDOWN: -> (answer) Those situations were limited.
  - (question) NAME OF ACCOUNT -> (answer) Sheetz
  - (question) NAME OF ACCOUNT -> (answer) M. Maskos & Sons
  - (question) NAME OF ACCOUNT -> (answer) Pollock Candy and Cigar
  - (question) NAME OF ACCOUNT -> (answer) McKeesport Candy Co.
  - (question) NAME OF ACCOUNT -> (answer) Sicc Serva
  - (question) NAME OF ACCOUNT -> (answer) Thrift/ Eckerd
  - (question) IND/ LOR VOLUME -> (answer) 104/ 22
  - (question) IND/ LOR VOLUME -> (answer) 521/ 42
  - (question) IND/ LOR VOLUME -> (answer) 137/ 20
  - (question) NO. OF STORES -> (answer) 18
  - (question) NO. OF STORES -> (answer) 150
  - (question) NO. OF STORES -> (answer) 183
  - (question) NAME OF ACCOUNT -> (answer) Kroger
  - (question) NAME OF ACCOUNT -> (answer) cvs
  - (question) NAME OF ACCOUNT -> (answer) Rich Oil
  - (question) NAME OF ACCOUNT -> (answer) Super America
  - (question) NAME OF ACCOUNT -> (answer) W H Smith
  - (question) NAME OF ACCOUNT -> (answer) 7- 11 318 Zon
  - (question) NAME OF ACCOUNT -> (answer) Dairy Marts
  - (question) NAME OF ACCOUNT -> (answer) Widman Drugs
  - (question) NO OF STORES -> (answer) 21
  - (question) NO OF STORES -> (answer) 82
  - (question) NO OF STORES -> (answer) 106
  - (question) NO OF STORES -> (answer) 87
  - (question) NO OF STORES -> (answer) 5
  - (question) NO OF STORES -> (answer) 23
  - (question) NO OF STORES -> (answer) 35
  - (question) NO OF STORES -> (answer) 43

#### `82504862_g.pdf`  [g]

- Scores: demo entity 77% / link 77% (7 pairs shown, 6 gold); benchmark entity 54% / link 32% (gold entities)
- Gold entities: 18 (answer 6, header 1, other 1, question 10); gold links: 6
- True links:
  - (question) COURT: -> (answer) San Francisco Superior Court- No. 996382
  - (question) CASE NAME: -> (answer) Donald D. Sellers and Robin J. Sellers v. Raybestos Manhattan et al.
  - (question) LORILLARD ENTITIES -> (answer) Lorillard Tobacco Company
  - (question) DATE SERVED -> (answer) August 3, 1998
  - (question) CASE TYPE: -> (answer) Asbestos
  - (question) PLAINTIEE'S COUNSEL: -> (answer) Wartnick Chaber, Harowitz, Smith& Tigerman StephenM. Tigerman 101 California Street Suite 2200 San Francisco California 94111

#### `83443897_g.pdf`  [g]

- Scores: demo entity 68% / link 72% (11 pairs shown, 14 gold); benchmark entity 91% / link 40% (gold entities)
- Gold entities: 33 (answer 12, header 2, other 7, question 12); gold links: 14
- True links:
  - (question) DATE: -> (answer) January 11, 1999
  - (question) COMPANY: -> (answer) Lorillard Tobacco Company
  - (question) PHONE: -> (answer) 336/ 373- 6750
  - (question) FROM: -> (answer) Andy Zausner and Rob Mangas
  - (question) PHONE: -> (answer) 202/ 828- 2259 and 202/ 828 2241
  - (question) MESSAGE: -> (answer) JAN 1 2 1999
  - (question) MESSAGE: -> (answer) The following is for your review
  - (question) NO -> (answer) X
  - (question) CLIENT NO.: -> (answer) L8557 002
  - (question) MESSAGE TO: -> (answer) Dewey Tedder
  - (question) FAX NUMBER: -> (answer) 336/ 373- 6917
  - (question) PAGES (including Cover Sheet): -> (answer) 2
  - (header) HARD COPY TO FOLLOW -> (question) YES
  - (header) HARD COPY TO FOLLOW -> (question) NO


### xfund

#### `de_val_13_g.pdf`  [g]

- Scores: demo entity 76% / link 88% (8 pairs shown, 8 gold); benchmark entity 96% / link 96% (end to end)
- Gold entities: 25 (answer 8, header 1, other 8, question 8); gold links: 8
- True links:
  - (question) Vorname, Name: -> (answer) Helen Schmitz
  - (question) Adresse: -> (answer) Tauberfranken 8
  - (question) PLZ, Ort: -> (answer) 97922 Lauda-Königshofen
  - (question) Telefon: -> (answer) 15289203232
  - (question) E-Mail: -> (answer) 15289203232@gmail.com
  - (question) Geburtsdatum: -> (answer) 04.04.1989
  - (question) Berufliche Ausbildung und Berufserfahrung: -> (answer) Bank, Bankerfahrung in Sparkasse
  - (question) Aktuelle Tätigkeit, Arbeitgeber: -> (answer) Sparkasse in München

#### `es_val_38_g.pdf`  [g]

- Scores: demo entity 80% / link 78% (47 pairs shown, 30 gold); benchmark entity 96% / link 89% (end to end)
- Gold entities: 47 (answer 18, header 1, other 13, question 15); gold links: 30
- True links:
  - (question) Información del miembro/paciente: -> (question) Nombre completo
  - (question) Información del miembro/paciente: -> (question) Fecha de nacimiento
  - (question) Información del miembro/paciente: -> (question) Número de ID de Anthem (indicado en la tarjeta de ID)
  - (question) Información del miembro/paciente: -> (question) Número de ID de Medicaid (indicado en la tarjeta de ID)
  - (question) Información del miembro/paciente: -> (question) Número telefónico
  - (question) Información del miembro/paciente: -> (question) Nombre del tutor legal (si es menor de 18 años)
  - (question) Nombre completo -> (answer) Irene Joan
  - (question) Fecha de nacimiento -> (answer) 12-19-1987
  - (question) Número de ID de Anthem (indicado en la tarjeta de ID) -> (answer) 7834 3498 9232
  - (question) Número de ID de Medicaid (indicado en la tarjeta de ID) -> (answer) RED4 776T 187J
  - (question) Número telefónico -> (answer) 34 958 135 343
  - (question) Nombre del tutor legal (si es menor de 18 años) -> (answer) Rosario Mendoza
  - (question) Información del nuevo PCP: -> (question) Fecha de la solicitud (o fecha efectiva)
  - (question) Información del nuevo PCP: -> (question) Nombre del PCP solicitado
  - (question) Información del nuevo PCP: -> (question) Número de ID del proveedor
  - (question) Información del nuevo PCP: -> (question) Dirección del proveedor
  - (question) Información del nuevo PCP: -> (question) Número telefónico del proveedor
  - (question) Información del nuevo PCP: -> (question) Número de fax del proveedor
  - (question) Fecha de la solicitud (o fecha efectiva) -> (answer) 03-12-2019
  - (question) Nombre del PCP solicitado -> (answer) 3478323223
  - (question) Número de ID del proveedor -> (answer) 47834792332
  - (question) Dirección del proveedor -> (answer) Callejón del Ángel, 2, 18006, Granada
  - (question) Número telefónico del proveedor -> (answer) 34 935 135 343
  - (question) Número de fax del proveedor -> (answer) 3847259341
  - (question) Motivo para la solicitud: -> (answer) No elegí mi último PCP
  - (question) Motivo para la solicitud: -> (answer) ¿ No estaba satisfecho con mi último PCP
  - (question) Motivo para la solicitud: -> (answer) _ Tuve problemas para conseguir citas con mi último PCP
  - (question) Motivo para la solicitud: -> (answer) Me mude, o mi PCP se mudo
  - (question) Motivo para la solicitud: -> (answer) LEI consultorio de mi PCP estaba demasiado lejos o era muy difícil ir
  - (question) Motivo para la solicitud: -> (answer) LOtro (explique abajo).

#### `fr_val_7_g.pdf`  [g]

- Scores: demo entity 69% / link 100% (8 pairs shown, 8 gold); benchmark entity 100% / link 98% (end to end)
- Gold entities: 31 (answer 8, header 1, other 14, question 8); gold links: 8
- True links:
  - (question) Civilité : -> (answer) Madame
  - (question) Nom : -> (answer) Robert
  - (question) Prénom : -> (answer) Katel
  - (question) Date de naissance : -> (answer) 09/09/1992
  - (question) Lieu de naissance : -> (answer) Lyon
  - (question) Adresse : -> (answer) 89 rue Alain, Paris
  - (question) Téléphone : -> (answer) 0682732476
  - (question) Courriel : -> (answer) Katel@hotmail.com

#### `it_val_38_g.pdf`  [g]

- Scores: demo entity 82% / link 90% (20 pairs shown, 18 gold); benchmark entity 100% / link 88% (end to end)
- Gold entities: 41 (answer 18, other 11, question 12); gold links: 18
- True links:
  - (question) Responsabile documento *: -> (answer) Rivieri Michele
  - (question) Redazione *: -> (answer) de Simone
  - (question) Redazione *: -> (answer) Carlotta
  - (question) Emissione *: -> (answer) ISEC
  - (question) Data* -> (answer) 09/01/2018
  - (question) Approvazione *: -> (answer) SERV
  - (question) Data -> (answer) 09/01/2018
  - (question) Codice Documento *: -> (answer) IA00Q001-2
  - (question) Stato Documento *: -> (answer) Pubblicato
  - (question) Distribuzione: -> (answer) Pubblica
  - (question) Data Emissione Prima Versione: -> (answer) 12-04-2017
  - (question) Data di Stampa: -> (answer) 14-04-2017
  - (question) La distribuzione è un campo da compilare manualmente e può assumere i valori: -> (answer) • Pubblica, se il documento può circolare senza restrizioni, sia internamente che esternamente;
  - (question) La distribuzione è un campo da compilare manualmente e può assumere i valori: -> (answer) • Riservata, se il documento è distribuibile solo agli Utenti Cedacri o all'interno di un Gruppo di Progetto (specificare il Gruppo
  - (question) La distribuzione è un campo da compilare manualmente e può assumere i valori: -> (answer) di progetto);
  - (question) La distribuzione è un campo da compilare manualmente e può assumere i valori: -> (answer) • Confidenziale, se il documento è distribuibile internamente a livello personale (specificare le persone)
  - (question) La distribuzione è un campo da compilare manualmente e può assumere i valori: -> (answer) • Utente, se il documento è destinato all'Organizzazione del Cliente
  - (question) La distribuzione è un campo da compilare manualmente e può assumere i valori: -> (answer) • <altro>, da specificare.


### vrdu_registration

#### `19911001_Arnold _ Porter Kaye Scholer, LLP_Dissemination Report_Dissemination Report.pdf`  [-]

- BROS (unseen template, mean of seeds 13/42): entity F1 96%, all fields found

- True field values:
  - `registration_num`: 1750
  - `registrant_name`: Arnold & Porter
  - `foreign_principle_name`: Republic of Venezuela
  - `signer_title`: Partner
  - `file_date`: 10/23/91
  - `signer_name`: Lawrence A. Schneider
- BROS seed-13 prediction:
  - `file_date`: 10/23/91
  - `registrant_name`: Arnold & Porter
  - `registration_num`: 1750
  - `signer_name`: Lawrence A. Schneider
  - `signer_title`: Partner

#### `20111209_Akin, Gump, Strauss, Hauer _ Feld, LLP_Johnson IV, Charles W._Short-Form.pdf`  [-]

- BROS (unseen template, mean of seeds 13/42): entity F1 100%, all fields found

- True field values:
  - `registration_num`: 3492
  - `registrant_name`: Charles W. Johnson IV
  - `file_date`: December 08, 2011
  - `signer_name`: Charles W. Johnson
- BROS seed-13 prediction:
  - `file_date`: December 08 , 2011
  - `registrant_name`: Charles W. Johnson IV
  - `registration_num`: 3492
  - `signer_name`: Charles W. Johnson

#### `20140520_Squire Patton Boggs, LLP_Amendment_Amendment_m.pdf`  [m]

- BROS (unseen template, mean of seeds 13/42): entity F1 100%, all fields found

- True field values:
  - `registrant_name`: Patton Boggs LLP
  - `registration_num`: 2165
  - `signer_title`: Managing Partner
  - `file_date`: May 19, 2014
  - `signer_name`: Edward J. Newberry
- BROS seed-13 prediction:
  - `file_date`: May 19 , 2014
  - `registrant_name`: Patton Boggs LLP
  - `registration_num`: 2165
  - `signer_name`: Edward J. Newberry
  - `signer_title`: Managing Partner


## unseen_template


### funsd

#### `83635935_m.pdf`  [m]

- Scores: demo entity 36% / link 57% (3 pairs shown, 4 gold); benchmark entity 67% / link 80% (gold entities)
- Gold entities: 17 (answer 4, header 2, other 7, question 4); gold links: 4
- True links:
  - (question) DATE: -> (answer) June 4, 1997
  - (question) TO: -> (answer) Dr. Spears/ A. J. Stevens/ R. Milstein
  - (question) FROM: -> (answer) Barry Hirsch
  - (question) OUR FAX NUMBER -> (answer) (212) 935 6801

#### `83772145_m.pdf`  [m]

- Scores: demo entity 44% / link 40% (14 pairs shown, 11 gold); benchmark entity 72% / link 44% (gold entities)
- Gold entities: 35 (answer 8, header 1, other 11, question 15); gold links: 9
- True links:
  - (question) #: -> (answer) 392
  - (question) TO: -> (answer) JACK REILLY
  - (question) FROM: -> (answer) JIM DALEY
  - (question) DATE: -> (answer) JANUARY 31, 1997
  - (question) TELECOPY NO: -> (answer) 212/ 545- 3297
  - (question) SHB Client Matter No: -> (answer) LORI 45048
  - (question) COMMENTS/ MESSAGE: -> (question) Jack
  - (question) COMMENTS/ MESSAGE: -> (answer) Per your request, attached is a memo regarding the process of labeling reviewed diskettes for the purpose of later identification Please call me if you have any questions regret I cannot join you and the electronic document collection team for lunch today. I hope you agree they have done an outstanding job! I look forward to seeing you on February 5 for the Legal Edge presentation. Best Regards, Jim Daley
  - (question) Special Instructions: -> (answer) If you experience any problems, please call extension 21000

#### `89856243_b.pdf`  [b]

- Scores: demo entity 32% / link 12% (48 pairs shown, 17 gold); benchmark entity 90% / link 43% (gold entities)
- Gold entities: 51 (answer 17, header 1, other 8, question 25); gold links: 17
- True links:
  - (question) PRODUCT -> (answer) Triumph
  - (question) Date -> (answer) October 17, 1979
  - (question) Incidence -> (answer) -
  - (question) Topline -> (answer) 2 wks from start of fid.
  - (question) Final -> (answer) 4 wks from start of fld
  - (question) PROJECT TITLE -> (answer) Triumph Disaster Check Study
  - (question) Key Criteria for Analysis -> (answer) -Triumph switchers asked why no longer smoking Triumph
  - (question) Key Criteria for Analysis -> (answer) -Present smokers of Triumph asked reasons for switching to Triumph
  - (question) Key Criteria for Analysis -> (answer) -Mentions of "harshness"
  - (question) Research Limitations -> (answer) Due to sample size and experience we feel 25 respondents will presently be Triumph smokers. An action standard of 75% should be utilized for judgments on product negatives only among this group
  - (question) Research Limitations -> (answer) If 75% of Triumph smokers mention harshness we could consider it a 50- 75% problem If of Triumph smokers mention "harshness" we could it a look consider possible problem at other areas).
  - (question) Research Limitations -> (answer) If below 50% of Triumph smokers mention "harshness" we could consider it at a problem
  - (question) Research Firm -> (answer) The Data Group. Inc
  - (question) Lenght Int -> (answer) 10 minutes
  - (question) # Open -> (answer) 4
  - (question) # Reports -> (answer) 1
  - (question) Cost Estimate + - 10% Contingency Final Report Inc. -> (answer) $ 2.250


### xfund

#### `it_val_34_m.pdf`  [m]

- Scores: demo entity 62% / link 25% (121 pairs shown, 58 gold); benchmark entity 80% / link 37% (end to end)
- Gold entities: 114 (answer 52, header 3, other 17, question 42); gold links: 58
- True links:
  - (question) Code: -> (answer) AbbPar
  - (question) Ed. 01 -> (answer) Rev. 00-09/2009
  - (question) Pagina: -> (answer) 1 di 2
  - (question) Rif .: -> (answer) PiaProCPR
  - (question) ORDINARIOP&P Mod_2 -> (answer) COMUNE DI FAENZA (RA)
  - (question) Data di presentazione -> (answer) 20/11/2020
  - (question) Cognome -> (answer) MERLINO
  - (question) Nome -> (answer) ANNA
  - (question) Doc. ID nr. -> (answer) 234
  - (question) Residente in via -> (answer) SAN DONATO 43
  - (question) Civico Interno -> (answer) 34
  - (question) Cap -> (answer) 10291
  - (question) Comune -> (answer) PIEMONTE
  - (question) Provincia -> (answer) TO
  - (question) C.F -> (answer) MEA7738447384773
  - (question) E-mail -> (answer) MERLINO ANNA@GMAIL.COM
  - (question) Data -> (answer) 10/12/2020
  - (question) Firma -> (answer) MERLINO
  - (question) Cognome -> (answer) CONTINO
  - (question) Nome -> (answer) LUIGI
  - (question) Residente in via -> (answer) SAN DONATO 43
  - (question) Civico Interno -> (answer) 44
  - (question) Cap -> (answer) 10283
  - (question) Comune -> (answer) PIEMONTE
  - (question) Provincia -> (answer) TO
  - (question) Tipologia abbonamento -> (answer) MENSILE (30 giorni continuativi)
  - (question) Tipologia abbonamento -> (answer) [X SEMESTRALE (180 giorni continuativi)
  - (question) Tipologia abbonamento -> (answer) □ ANNUALE (365 giorni continuativi)
  - (question) TARIFFA -> (answer) € 30,00
  - (question) TARIFFA -> (answer) € 170,00
  - (question) TARIFFA -> (answer) € 325,00
  - (question) Con decorrenza dal: -> (answer) 12/12/2020
  - (question) e fino al: -> (answer) 30/12/2020
  - (question) Marca -> (answer) MIMOSA
  - (question) Modello -> (answer) VENDITA
  - (question) Targa -> (answer) 782
  - (question) che venga riconosciuta la seguente riduzione alla quale ho diritto: -> (answer) auto elettrica (sconto 50%)
  - (question) dichiara inoltre che sussiste il presupposto per la richiesta di accesso in ZTL per: -> (question) [ G) utilizzare/per accedere ad un garage / cortile privato sito in via
  - (question) dichiara inoltre che sussiste il presupposto per la richiesta di accesso in ZTL per: -> (question) all'interno della ZTL, di proprieta
  - (question) dichiara inoltre che sussiste il presupposto per la richiesta di accesso in ZTL per: -> (answer) ☒ svolgere l'attività di rifornimento della propria attività, sita all'interno della ZTL (S);
  - (question) dichiara inoltre che sussiste il presupposto per la richiesta di accesso in ZTL per: -> (answer) □ svolgere l'attività di rifornimento di attività commerciali ed artigiani all'interno della ZTL (S);
  - (question) dichiara inoltre che sussiste il presupposto per la richiesta di accesso in ZTL per: -> (answer) □ svolgere l'attività di corriere e/o trasportatore di merci per conto terzi (S);
  - (question) dichiara inoltre che sussiste il presupposto per la richiesta di accesso in ZTL per: -> (answer) □ svolgere l'attività di trasporto merci e documentazione amministrativa per conto di Istituto di Credito in ZTL (S);
  - (question) dichiara inoltre che sussiste il presupposto per la richiesta di accesso in ZTL per: -> (answer) □ svolgere l'attività di ambulante con posto fisso assegnato nei mercati cittadini (AM);
  - (question) dichiara inoltre che sussiste il presupposto per la richiesta di accesso in ZTL per: -> (answer) □ effettuare visite domiciliari urgenti in qualità di medico (M);
  - (question) dichiara inoltre che sussiste il presupposto per la richiesta di accesso in ZTL per: -> (answer) □ svolgere attività istituzionali (SP);
  - (question) dichiara inoltre che sussiste il presupposto per la richiesta di accesso in ZTL per: -> (question) [X] raggiungere il cantiere edile sito all'interno della ZTL in Via _
  - (question) dichiara inoltre che sussiste il presupposto per la richiesta di accesso in ZTL per: -> (question) □ potersi recare presso lo Studio Medico/ Ambulatorio_
  - (question) dichiara inoltre che sussiste il presupposto per la richiesta di accesso in ZTL per: -> (question) fino al
  - (question) dichiara inoltre che sussiste il presupposto per la richiesta di accesso in ZTL per: -> (question) sito all'interno della ZTL in Via
  - (question) [ G) utilizzare/per accedere ad un garage / cortile privato sito in via -> (answer) SAN DONATO 23
  - (question) all'interno della ZTL, di proprieta -> (answer) IDENDITA
  - (question) [X] raggiungere il cantiere edile sito all'interno della ZTL in Via _ -> (answer) SAN DONATO 23
  - (question) □ potersi recare presso lo Studio Medico/ Ambulatorio_ -> (answer) N.781234
  - (question) fino al -> (answer) 30/12/2020
  - (question) sito all'interno della ZTL in Via -> (answer) SAN DONATO 23
  - (question) esclusivamente per eseguire cure continuative fino al -> (answer) 30/12/2020
  - (question) Tel. /Cell. -> (answer) 3374464883

#### `pt_val_17_b.pdf`  [b]

- Scores: demo entity 29% / link 11% (82 pairs shown, 26 gold); benchmark entity 61% / link 39% (end to end)
- Gold entities: 84 (answer 26, header 2, other 35, question 21); gold links: 26
- True links:
  - (question) Agência -> (answer) MC
  - (question) CPF/CNPJ/MF -> (answer) 881.334.667-13
  - (question) Código da Agência -> (answer) MC
  - (question) E-mail do Funcionário -> (answer) bgc657@gmail.com
  - (question) Nome, -> (answer) Vitór A Barbosa
  - (question) Data de Nascimento -> (answer) 1975-8-17
  - (question) CPF/MF -> (answer) 798.441.444-00
  - (question) Profissão -> (answer) Medical Translator
  - (question) Sexo -> (answer) Masculino Feminino
  - (question) Estado Civil -> (answer) AM
  - (question) Sua profissão é sua principal Fonte de Renda? -> (answer) Sim Não
  - (question) Finalidade do Imóvel -> (answer) 1 - Comercial 2 - Residencial 3 - Lazer 4 - Lote
  - (question) Discriminação -> (answer) 1 - Esteve ou está afastado de suas atividades profissionais? Por que motivo?
  - (question) Discriminação -> (answer) Esclareça:
  - (question) Discriminação -> (answer) 2 - Está em gozo de benefício previdenciário de invalidez? Por que motivo?
  - (question) Discriminação -> (answer) Esclareça:
  - (question) Sim -> (answer) √
  - (question) Sim -> (answer) √
  - (question) Mod .: -> (answer) 4840-605E
  - (question) Versão: -> (answer) 05/2016
  - (question) Sistema de Amortização Escolhido (Prestação Atualizada) -> (answer) 1 - SAC - Sistema de Amortização Constante
  - (question) Sistema de Amortização Escolhido (Prestação Atualizada) -> (answer) 2 - TP - Tabela Price
  - (question) Funcionário Responsável -> (answer) Bruno G Cunha
  - (question) Díg. -> (answer) 387
  - (question) Conta -> (answer) 7558651504
  - (question) Díg. -> (answer) 526

#### `pt_val_28_m.pdf`  [m]

- Scores: demo entity 84% / link 50% (101 pairs shown, 55 gold); benchmark entity 95% / link 64% (end to end)
- Gold entities: 107 (answer 53, header 1, other 8, question 45); gold links: 55
- True links:
  - (question) Proponente: -> (answer) Embraer S.A.
  - (question) CNPJ: -> (answer) 17.115.437/0001-73
  - (question) Endereço: -> (answer) Vila Coelho Xavier 1973
  - (question) CEP: -> (answer) 15800-3401
  - (question) Município: -> (answer) Fortaleza
  - (question) UF: -> (answer) SP
  - (question) DDD: -> (answer) (85)
  - (question) Telefone: -> (answer) 2501-2316
  - (question) Fax: -> (answer) (85) 8614-3556
  - (question) E-mail: -> (answer) zzu19@linshiyouxiang.net
  - (question) População do município: -> (answer) 1 bilhão
  - (question) Regime jurídico / Esfera administrativa: -> (answer) Direito Público
  - (question) Regime jurídico / Esfera administrativa: -> (answer) Direito Privado
  - (question) Regime jurídico / Esfera administrativa: -> (answer) Federal
  - (question) Regime jurídico / Esfera administrativa: -> (answer) Estadual
  - (question) Regime jurídico / Esfera administrativa: -> (answer) Municipal
  - (question) Regime jurídico / Esfera administrativa: -> (answer) ☑
  - (question) Regime jurídico / Esfera administrativa: -> (answer) ☑
  - (question) Regime jurídico / Esfera administrativa: -> (answer) □
  - (question) Regime jurídico / Esfera administrativa: -> (answer) □
  - (question) Regime jurídico / Esfera administrativa: -> (answer) □
  - (question) CPF: -> (answer) 107.672.200-80
  - (question) Representante legal: -> (answer) Renan R Barros
  - (question) RG / órgão expedidor: -> (answer) 1.234.543 SE
  - (question) Cargo: -> (answer) escretária
  - (question) Função: -> (answer) administração
  - (question) E-mail: -> (answer) kkqg58|@payspun.com
  - (question) Endereço residencial: -> (answer) Rua Teresópolis 294
  - (question) Telefone: -> (answer) (24) 2501-2316
  - (question) CEP: -> (answer) 15800-3567
  - (question) Município: -> (answer) Petrópolis
  - (question) Data: -> (answer) 1\1\2019
  - (question) Assinatura: -> (answer) Renan R Barros
  - (question) UF: -> (answer) RJ
  - (question) Período de execução -> (question) Início:
  - (question) Período de execução -> (question) Término:
  - (question) Título do projeto: -> (answer) Aquisições
  - (question) Início: -> (answer) 8/2018
  - (question) Identificação do objeto: -> (answer) SMOO
  - (question) Justificativa da proposição: -> (answer) A empresa atualmente precisa se expandir, SMOO enfrenta falência
  - (question) Metas: -> (answer) Aquisições com sucesso
  - (question) Nome do Coordenador I: -> (answer) Tiago Lopes
  - (question) Término: -> (answer) 8/2019
  - (question) CPF -> (answer) 603.216.025-44
  - (question) DDD: -> (answer) 85
  - (question) Telefone: -> (answer) 8614-3556
  - (question) E-mail: -> (answer) zzu19@linshiyouxiang.net
  - (question) Assinatura: -> (answer) Tiago Lopes
  - (question) Nome do Coordenador II: -> (answer) Nuno Rodrigues
  - (question) CPF -> (answer) 603.216.025-44
  - (question) DDD: -> (answer) 21
  - (question) Telefone: -> (answer) 5380-6371
  - (question) E-mail: -> (answer) zzu19@linshiyouxiang.com
  - (question) Assinatura: -> (answer) Nuno Rodrigues
  - (question) Site: -> (answer) www.embraersa.br

#### `pt_val_41_m.pdf`  [m]

- Scores: demo entity 58% / link 39% (108 pairs shown, 60 gold); benchmark entity 73% / link 32% (end to end)
- Gold entities: 73 (answer 30, header 1, other 8, question 34); gold links: 60
- True links:
  - (question) INFORMAÇÕES SOBRE O CANDIDATO -> (question) Nome Completo:
  - (question) INFORMAÇÕES SOBRE O CANDIDATO -> (question) RG:
  - (question) INFORMAÇÕES SOBRE O CANDIDATO -> (question) Local e Data de Nascimento:
  - (question) INFORMAÇÕES SOBRE O CANDIDATO -> (question) Nome completo do Pai:
  - (question) INFORMAÇÕES SOBRE O CANDIDATO -> (question) Nome completo da Mãe:
  - (question) INFORMAÇÕES SOBRE O CANDIDATO -> (question) Bairro:
  - (question) INFORMAÇÕES SOBRE O CANDIDATO -> (question) Cidade:
  - (question) INFORMAÇÕES SOBRE O CANDIDATO -> (question) Telefone Fixo:
  - (question) INFORMAÇÕES SOBRE O CANDIDATO -> (question) CEP:
  - (question) INFORMAÇÕES SOBRE O CANDIDATO -> (question) Celular:
  - (question) INFORMAÇÕES SOBRE O CANDIDATO -> (question) Recado:
  - (question) INFORMAÇÕES SOBRE O CANDIDATO -> (question) Ponto de Referência:
  - (question) INFORMAÇÕES SOBRE O CANDIDATO -> (question) Rua e número:
  - (question) Nome Completo: -> (answer) Isabelle M Souza
  - (question) RG: -> (answer) 278.064-1
  - (question) Local e Data de Nascimento: -> (answer) Ponta Grossa 11/2/2003
  - (question) Nome completo do Pai: -> (answer) Pedro Nuno Souza
  - (question) Nome completo da Mãe: -> (answer) Ana Maria Souza
  - (question) Bairro: -> (answer) centro
  - (question) Cidade: -> (answer) Ponta Grossa
  - (question) Telefone Fixo: -> (answer) (42) 4512-8746
  - (question) CEP: -> (answer) 70.675-628
  - (question) Celular: -> (answer) (42) 46512-8746Tel. Para
  - (question) Recado: -> (answer) (24) 2501-2316
  - (question) Ponto de Referência: -> (answer) jardim
  - (question) INFORMAÇÕES SOBRE A FAMÍLIA DO CANDIDATO -> (question) Com quem o candidato reside?
  - (question) INFORMAÇÕES SOBRE A FAMÍLIA DO CANDIDATO -> (question) Situação de trabalho dos pais e ou responsável:
  - (question) INFORMAÇÕES SOBRE A FAMÍLIA DO CANDIDATO -> (question) Se empregado, qual a função?
  - (question) INFORMAÇÕES SOBRE A FAMÍLIA DO CANDIDATO -> (question) Salário do Pai:
  - (question) INFORMAÇÕES SOBRE A FAMÍLIA DO CANDIDATO -> (question) Em caso de residir com responsável que não os genitores qual a situação de
  - (question) INFORMAÇÕES SOBRE A FAMÍLIA DO CANDIDATO -> (question) Mãe:
  - (question) INFORMAÇÕES SOBRE A FAMÍLIA DO CANDIDATO -> (question) trabalho do responsável?
  - (question) Com quem o candidato reside? -> (answer) Pai (1 ) Mãe(v) Padrasto( ) Madrasta( ) Outro Responsável( ) Quem?
  - (question) Situação de trabalho dos pais e ou responsável: -> (answer) Pai: Empregado(V ) Desempregado( ) Aposentado( ) Autônomo( ) Não
  - (question) Situação de trabalho dos pais e ou responsável: -> (answer) sabe informar( )
  - (question) Se empregado, qual a função? -> (answer) professor
  - (question) Salário do Pai: -> (answer) R$ 15000
  - (question) Em caso de residir com responsável que não os genitores qual a situação de -> (answer) Empregada( ) Desempregada( ) Aposentada(V )
  - (question) Em caso de residir com responsável que não os genitores qual a situação de -> (answer) Autonoma( ) Não sabe informar( )
  - (question) Em relação aos demais membros da família -> (question) Número de irmãos:
  - (question) Em relação aos demais membros da família -> (question) Quantos destes irmãos residem com o candidato?
  - (question) Em relação aos demais membros da família -> (question) Quantos destes irmãos trabalham?
  - (question) Em relação aos demais membros da família -> (question) Qual a renda do(a) irmão(ã) que trabalha?
  - (question) Em relação aos demais membros da família -> (question) Ha outro familiar que reside com o candidato:
  - (question) Em relação aos demais membros da família -> (question) Caso haja outro familiar residindo com a família, qual o valor da renda deste
  - (question) Em relação aos demais membros da família -> (question) familiar?
  - (question) Número de irmãos: -> (answer) Fernanda Souza
  - (question) Quantos destes irmãos residem com o candidato? -> (answer) 1
  - (question) Quantos destes irmãos trabalham? -> (answer) 1
  - (question) Qual a renda do(a) irmão(ã) que trabalha? -> (answer) R$3500
  - (question) Ha outro familiar que reside com o candidato: -> (answer) sim( ) não(√ ) quem?
  - (question) Caso haja outro familiar residindo com a família, qual o valor da renda deste -> (answer) R$1000
  - (question) INFORMAÇÕES ACADÊMICAS DO CANDIDATO -> (question) Nome da escola que o candidato estuda: escola de Ponta Grossa
  - (question) INFORMAÇÕES ACADÊMICAS DO CANDIDATO -> (question) Que série está cursando no Ensino Médio:
  - (question) INFORMAÇÕES ACADÊMICAS DO CANDIDATO -> (question) O candidato já frequentou algum tipo de curso? Qual?
  - (question) Nome da escola que o candidato estuda: escola de Ponta Grossa -> (answer) Horário das aulas: Matutino(V ) Vespertino( ) Noturno( )
  - (question) Nome da escola que o candidato estuda: escola de Ponta Grossa -> (answer) 1º( √) 2º( ) 3º( )
  - (question) O candidato já frequentou algum tipo de curso? Qual? -> (answer) sim. cultura portuguesa.
  - (question) Rua e número: -> (answer) Rua Leopoldo Miquez 1174
  - (question) Mãe: -> (answer) Empregada( ) Desempregada( V) Aposentada( ) Autônoma( ) Não


### vrdu_registration

#### `19920401_JETRO, Los Angeles_Dissemination Report_Dissemination Report.pdf`  [-]

- BROS (unseen template, mean of seeds 13/42): entity F1 91%, missed: foreign_principle_name

- True field values:
  - `registrant_name`: JETRO, Los Angeles
  - `registration_num`: 1833
  - `foreign_principle_name`: Japan External Trade Organization Tokyo, Japan
  - `signer_name`: Eiichi Nagata
  - `file_date`: 4/3/92
  - `signer_title`: Executive Director
- BROS seed-13 prediction:
  - `file_date`: 4/3/92
  - `registrant_name`: JETRO , Los Angeles
  - `registration_num`: 1833
  - `signer_name`: Eiichi Nagata
  - `signer_title`: Executive Director

#### `19931201_JETRO, Los Angeles_Dissemination Report_Dissemination Report.pdf`  [-]

- BROS (unseen template, mean of seeds 13/42): entity F1 63%, missed: foreign_principle_name, signer_name, signer_title

- True field values:
  - `registration_num`: 1833
  - `registrant_name`: JETRO, Los Angeles
  - `foreign_principle_name`: Japan External Trade Organization Tokyo, Japan
  - `file_date`: 12/6/93
  - `signer_name`: Harumi Kamekawa
  - `signer_title`: Public Relations Director
- BROS seed-13 prediction:
  - `file_date`: 12/6/93
  - `registrant_name`: JETRO , Los Angeles
  - `registrant_name`: Asahi Homecast Corporation
  - `registration_num`: 1833

#### `19980220_KOTRA_Amendment_Amendment.pdf`  [-]

- BROS (unseen template, mean of seeds 13/42): entity F1 83%, missed: signer_name

- True field values:
  - `registrant_name`: Korea Trade Promotion Center, New York
  - `registration_num`: 1619
  - `signer_name`: Ja Yoon
- BROS seed-13 prediction:
  - `registrant_name`: Korea Trade Promotion Center , New York
  - `registration_num`: 1619
  - `signer_name`: Ja Yoon

#### `20101206_India Tourism, New York__Short-Form.pdf`  [-]

- BROS (unseen template, mean of seeds 13/42): entity F1 54%, missed: foreign_principle_name, registrant_name

- True field values:
  - `registration_num`: 2329
  - `registrant_name`: GOVT OF INDIA MINISTRY OF TOURISM
  - `foreign_principle_name`: every foreign principal to whom you will render services in support of the primary registrant. Promotion of Tourism by assisting travel trade, airlines,tarism exhibitors, schools, associations, conforences, etc.
  - `file_date`: 2 2010.
- BROS seed-13 prediction:
  - `file_date`: 2 2010 .
  - `registrant_name`: MR . SENTHUR KUMARAN VELAYUTHAM
  - `registration_num`: 2329

#### `20120327_Caribbean Tourism Organization, USA Inc. _Bramble, Sylma Brown_Short-Form.pdf`  [-]

- BROS (unseen template, mean of seeds 13/42): entity F1 67%, missed: registrant_name

- True field values:
  - `registration_num`: 991
  - `registrant_name`: Caribbean Tourism Organization USA Inc
  - `file_date`: 3/10/12 12
- BROS seed-13 prediction:
  - `file_date`: 3/10/12 12
  - `registrant_name`: Sylma Brown Bramble
  - `registration_num`: 991

#### `20180308_S.G.R. LLC Government Relations and Lobbying_Amendment_Amendment.pdf`  [-]

- BROS (unseen template, mean of seeds 13/42): entity F1 86%, missed: signer_name

- True field values:
  - `registrant_name`: S.G.R. LLC Government Relations and Lobbying
  - `registration_num`: 6379
  - `file_date`: 3/8/18
  - `signer_name`: GREG LOWMAN
- BROS seed-13 prediction:
  - `file_date`: 3/8/18
  - `registrant_name`: S.G.R. LLC Government Relations and Lobbying
  - `registration_num`: 6379
