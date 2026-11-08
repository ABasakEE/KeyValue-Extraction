# Study 26: table-type links via relation-rich synthetic pretraining and link-preserving crops

## Context

The joint LiLT model (token tags + biaffine link head over 16-D spatial features) handles nearby same-line pairs well, but it misses the links that tables are built from. Study 22 (XFUND, gold entities) measured this:
- partner below: 58% of gold links, recall 49 (vs 79 for same-line partners)
- partner 5+ entities away: recall 49
- question->question (heading->label) links: recall 34
- 10% of links never fall inside a 510-token training window

`demo_pdfs/README.md` also notes that "dense tables ... still come out poorly". The best end-to-end XFUND link F1 is 47.7 (Study 19 `windows` arm).

The user wants to use recent research (2025–26) to improve these results before the 23/25 Nov novelty assessment, skip VLM work for now, and run the Kaggle jobs themselves. We take two ideas from two papers:
- **Link-preserving crops** (the paper calls them graph-based crops) from Identify, Locate, Link (ICDAR 2026, arXiv 2608.20868): crop around connected groups of linked key-value pairs, merge groups that sit close together, add random padding.
- **Relation-rich synthetic pretraining** from RIDGE (CVPR 2025, arXiv 2504.10659): its links come from nested headings, and pretraining helped LayoutLMv3 few-shot entity tagging. Identify, Locate, Link also reports synthetic filling raising XFUND relation extraction from 0.171 to 0.206.

Neither paper shows a relation-extraction gain for a LiLT-style linker. Neither has a no-crop baseline or a template-disjoint split. So this study is the first test of these ideas for our model, and any result, positive or negative, is ours to report. The user agreed to go ahead knowing this.

Work happens on a new branch `feat/table-links` cut from `feat/bros-relative-layout`. The study is registered in `STUDIES.md` and the registration is committed **before** any code is written or anything runs (repo convention).

## Arms (XFUND es/fr/de/it/pt, seeds 13/21/42, Study 19 recipe: `configs/xfund_joint_lilt_windows.yaml`, FP32)

| Arm | What changes | Purpose |
|---|---|---|
| A0 | none (Study 19 `windows` recipe, retrained) | Same-run control. Study 19's weights aren't available locally, and the per-group recall needs predictions. Cross-check: A0 should land near Study 19's 47.7. |
| A1 | synthetic pretraining stage, then the A0 recipe | synthetic idea |
| A2 | A0 recipe + link-preserving crops added to the training set | crop idea |
| A3 | A1 + A2 | combined |

Run order in the notebook: A0, A2, A1, A3. Runs are resumable per seed. If the Kaggle quota runs out, any arm that didn't finish is reported as not run.

## Registered criteria (written into STUDIES.md before running)

- **Primary cell:** end-to-end link F1, windowed inference, all 250 val forms, all gold links. Seed-matched delta vs A0.
  - *helps:* mean ≥ +2.0 and positive in every seed
  - *no better:* |mean| < 1.0, or signs differ across seeds
  - *worse:* negative mean with a consistent sign
  - *inconclusive:* anything else
  - Verdicts are checked in this order: helps, no_better, worse, inconclusive; the first that matches applies.
  - **A0 sanity gate:** A0 must reproduce Study 19's windowed end-to-end link F1 (47.7) within 1.5 points; if it does not, the other arms are reported but no verdict is drawn until the difference is explained.
- **Table cells (secondary, reported for every arm):** end-to-end recall of gold links grouped as
  - partner below / same line / above
  - partner rank 0, 1, 2–4, 5+
  - question->question
- **Other secondary cells:** entity F1, `link_qa`, the sibling-free subset, and link F1 on the synthetic held-out-family test set.
- **Limits stated up front:** 3 seeds and no intervals; synthetic forms have English keys while XFUND has none; papers don't support a relation-extraction gain; the crop and filling parameters are our own choices.

## Implementation

### 1. Link-preserving crops: new `crop_examples()` in `src/data/xfund_windows.py`
In `STUDIES.md` and the write-up, call this "link-preserving crop augmentation", not "graph-based". Say plainly that it changes the training data and leaves the model unchanged. This keeps it separate from the earlier graph models (RSA-GNN, NER-fused GNN, Graph-BROS).
- Per training form:
  - build the link graph from `ex.links` over `ex.entities`
  - take connected components (union-find), ignoring single unlinked entities
  - merge components whose bounding boxes are within `crop_merge_dist` (default 50 of 1000 units)
  - pad each region by a random 10–60 units, clamped to the page
- `crop_example(ex, region)` keeps the words whose box centre is inside the region, in their original order. It:
  - remaps `token_indices` and `entity_ids`
  - keeps only entities that lie wholly inside, and links whose two ends are both kept
  - **re-normalizes boxes to the crop frame (0–1000)**, as the paper crops images; LiLT reads no pixels, so no image work is needed
- `sub_example` (same file) shows the remap pattern to follow, but crops select non-contiguous word indices.
- Crops are **added to** the full-form windows, never replacing them. Up to `crop_max_per_form` (default 4) per form. Crops longer than 510 tokens still go through `window_examples`.
- New `TrainConfig` fields in `src/train.py`: `crop_aug: bool`, `crop_max_per_form`, `crop_merge_dist`.
- Hook: `src/train.py:559`, where `form_window` calls `window_examples`. Crops are built from `train_examples` only, never from validation.
- Crops are fixed once per seed (seeded RNG), so resume is deterministic.

### 2. Relation-rich synthetic data: `src/data/synthetic_generator.py`
- Existing pieces to reuse:
  - 7 families in `GENERATOR_REGISTRY` (line 915)
  - tables with header->cell and summary links (`_render_table`, line 211)
  - FUNSD-format output, which loads through `src/data/funsd_loader.py` into `FormExample`
- Add:
  - (a) nested sections where a section-label question links to its child field-label questions (question->question, the RIDGE-style hierarchy)
  - (b) a key-above-value layout option on `FormField`, for downward links
  - (c) taller tables with more rows, so a header links to far-away cells
- Add `--train_families` and `--test_families` filters to the generator CLI (canonical family keys: invoice, financial_statement, procurement_order, shipping_manifest, medical_intake, adbuy_contract, fara_registration).
  - Train families: invoice, financial_statement, procurement_order, adbuy_contract, fara_registration.
  - **Held out for the synthetic test:** shipping_manifest, medical_intake.
- Generate with `--no_augraphy`, since LiLT ignores pixels and this is much faster: about 3,000 train docs and 300 held-out-family test docs, built into a dataset in the notebook.
- Faker locales es_ES/fr_FR/de_DE/it_IT/pt_PT for **values** (keys stay English), so values look like XFUND text.

### 3. Pretrain-then-fine-tune: `src/train.py`
- New config `configs/xfund_synth_pretrain.yaml`: joint task, 3-stage curriculum shortened (2/2/3 epochs), synthetic train set, validation is carved from the training families (15%, by the funsd loader); the held-out families (shipping_manifest, medical_intake) are used only as the synthetic test set, never for checkpoint selection.
- New `TrainConfig.init_checkpoint: str = ""`. After `build_model`, if it's set, load the state dict (`strict=True`, same architecture). Arms A1 and A3 point it at the per-seed pretrained `best_model.pt`.
- New configs `xfund_table_A0..A3.yaml`, each derived from `xfund_joint_lilt_windows.yaml` with only the arm's fields changed.

### 4. Evaluation: extend `scripts/xfund_e2e_eval.py`
- Add the per-group recall above, computed end to end. For every gold link (from `gold_structure` in `src/e2e_link.py`), group it by:
  - direction (reuse `direction()` from `scripts/xfund_link_errors.py`)
  - partner rank (reuse `rank_of()` / `bucket()`)
  - endpoint types

  Count it found if it's in the predicted link set. Write the counts under `{sub}|{mode}|group|<name>`.
- Add a `--data synthetic` path that scores link F1 on the held-out-family set.
- New `scripts/table_links_summary.py`: seed-matched deltas vs A0 and the verdict keys `helps` / `no_better` / `worse`, modelled on `scripts/xfund_e2e_summary.py`.

### 5. Kaggle notebook: `notebooks/kaggle_table_links.ipynb`
Copy `notebooks/kaggle_windowed_e2e.ipynb` and keep the 7-cell contract:
1. clone with `GITHUB_TOKEN`, install deps
2. fetch XFUND (Hugging Face mirror `FrancophonIA/XFUND`) and generate synthetic data
3. smoke test: crops, pretrain-to-fine-tune loading, and a mini 3-phase run per arm; raise on failure
4. arms in order A0, A2, A1 (pretrain, then fine-tune), A3; resumable; never raises
5. evaluate and write `results/table_links/*.json`
6. display
7. push JSON only, never weights

## Verification

- **Unit tests** (new `tests/test_table_crops.py`, plus additions to `tests/test_synthetic_generator.py`):
  - every link in a crop has both ends inside it; `token_indices` remap to the right words; boxes stay in 0–1000; no validation form is ever cropped
  - generator emits question->question and key-above-value links; held-out families never appear in train output
  - `init_checkpoint` round-trip: the loaded model's link logits equal the saved model's
- **CPU smoke:** extend `scripts/smoke_xfund_e2e.py` to run each arm on 16 forms (A1/A3 via a 10-doc synthetic pretrain) through `xfund_e2e_eval.py`, checking that every output cell, including the group cells, is written. This needs `data/xfund` locally, which is missing; fetch it from the HF mirror (NOVELTY_ASSESSMENT §9).
- **Evaluator sanity:** the new group counts must sum to the existing all-links counts for the same run.
- **Kaggle:** cell 3 smoke passes on the GPU before cell 4 starts. Then confirm that A0 reproduces Study 19 (`results/xfund_e2e/windows_e2e.json`) to within about 1.5 link F1 before reading the other arms.
- **Write-up:** results go into the Study 26 entry with evidence paths. Mark anything post hoc as post hoc.
