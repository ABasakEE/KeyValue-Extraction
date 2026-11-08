# Study 26 (table-type links) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether link-preserving crop augmentation and relation-rich synthetic pretraining (ideas from arXiv 2608.20868 and 2504.10659) raise end-to-end XFUND link F1 and recall on table-type links (downward, far partner, question->question) for the joint LiLT model.

**Architecture:** The model is unchanged. Crops are a training-set augmentation in `src/data/xfund_windows.py`, hooked into `train_one`. Synthetic data comes from the existing generator, extended with nested sections and more stacked and tall-table layouts; a pretrain run's per-seed checkpoint initialises the XFUND run through a new `init_checkpoint` config field. Evaluation extends `scripts/xfund_e2e_eval.py` with per-group recall. A 7-cell Kaggle notebook runs arms A0–A3.

**Tech Stack:** Python 3.11, PyTorch, transformers (LiLT), Faker, PIL; scripts and tests run as plain `python file.py` (the repo has no pytest).

**Spec:** `docs/superpowers/specs/2026-11-08-table-links-design.md`

## Global Constraints

- Branch `feat/table-links` (cut from `feat/bros-relative-layout`). Stage files by name, never `git add -A` or `git add .`.
- Study 26 is registered in `STUDIES.md` and committed **before** any code task.
- In docs and code comments call the crops "link-preserving crop augmentation", never "graph-based".
- Seeds 13 / 21 / 42. XFUND languages es, fr, de, it, pt. FP32 (no AMP), the same as Study 19.
- Every arm config equals `configs/xfund_joint_lilt_windows.yaml` except the arm's own fields.
- Verdict thresholds, in F1 points on the seed-matched delta vs A0: `helps` = mean ≥ +2.0 and every seed > 0; `no_better` = |mean| < 1.0 or the signs differ; `worse` = mean < 0 with a consistent sign; anything else is `inconclusive`.
- Synthetic held-out families: `shipping_manifest`, `medical_intake`. They never appear in synthetic training data.
- The Kaggle notebook pushes JSON/JSONL only, never `.pt` weights.

## Review Focus

1. **A crop with no linked entity, or a form with no links:** `crop_examples` returns nothing for that form and never crashes. Pinned in Task 2 (`test_form_without_links_gives_no_crops`).
2. **A word box partly outside the crop region:** the word is kept or dropped by its centre, and an entity is kept only if all its words are kept. A link to a dropped entity is removed. Pinned in Task 2 (`test_partial_entity_dropped_with_its_links`).
3. **A degenerate crop region (zero width or height after clamping):** box re-normalisation doesn't divide by zero, and boxes stay inside 0–1000. Pinned in Task 2 (`test_degenerate_region_boxes_in_range`).
4. **`init_checkpoint` set but the file is missing for a seed:** training stops with a clear error naming the path instead of silently training from scratch. Pinned in Task 3 (`test_missing_init_checkpoint_raises`).
5. **A gold link whose endpoint has no tokens (cut off or untagged):** it is skipped by `link_groups`, so the group counts still sum to the counted gold links. Pinned in Task 5 (`test_groups_partition_links`).

---

### Task 1: Register Study 26 and set up the local CPU environment

**Files:**
- Modify: `STUDIES.md` (append section `## 26. ...` after Study 25; add a row to the summary table near the top)
- Add: `docs/superpowers/specs/2026-11-08-table-links-design.md` (already copied), `docs/superpowers/plans/2026-11-08-table-links.md`

- [ ] **Step 1: Write the Study 26 registration** in `STUDIES.md`, in the style of Studies 24/25. Include:
  - Heading: `## 26. Table-type links: link-preserving crops and relation-rich synthetic pretraining — GPU (Kaggle T4 × 2), registered 2026-11-08`
  - **Question:** do the two ideas raise end-to-end link F1 and recall on downward, far-partner and question->question links? Cite Study 22's numbers: below 49, rank 5+ 49, question->question 34.
  - **Sources:** the two papers, and the honest note that neither paper shows a relation-extraction gain for a LiLT-style linker or has a no-crop baseline.
  - **Arms:** A0–A3, with the run order A0, A2, A1, A3.
  - **Primary cell and verdict keys:** exactly as in Global Constraints.
  - **Secondary cells:** group recalls (`below`, `same_line`, `above`, `rank_0`, `rank_1`, `rank_2_4`, `rank_5plus`, `q_to_q`), entity F1, `link_qa`, sibling-free subset, synthetic held-out-family link F1.
  - **Crop parameters:** up to 4 crops per form, merge distance 50, random padding 10–60, boxes re-normalised to the crop frame, crops added to (not replacing) full-form windows, training set only.
  - **Synthetic data:** about 3,000 train documents from 5 families and 300 test documents from the 2 held-out families; `--no_augraphy`; Faker locale values with English keys; pretrain curriculum 2/2/3 epochs.
  - **Limits:** as listed in the spec.
  - Results: *pending*.
- [ ] **Step 2: Commit the registration before any code**

```bash
git add STUDIES.md docs/superpowers/specs/2026-11-08-table-links-design.md docs/superpowers/plans/2026-11-08-table-links.md
git commit -m "feat(study-26): register table-type link study (crops + relation-rich synthetic pretraining), before code and run"
```

- [ ] **Step 3: Create a CPU venv** (it lives under `.venv/`, which must stay untracked; add `.venv/` to `.gitignore` if it isn't covered)

Run: `uv venv .venv --python 3.11 && uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cpu && uv pip install --python .venv/Scripts/python.exe -r requirements.txt`
Expected: `.venv/Scripts/python.exe -c "import torch, transformers, faker; print('ok')"` prints `ok`.

- [ ] **Step 4: Fetch XFUND into `data/xfund`** from the Hugging Face mirror `FrancophonIA/XFUND` (NOVELTY_ASSESSMENT §9). Then check:

Run: `.venv/Scripts/python.exe -c "import sys; sys.path.insert(0,'src'); from data.xfund_loader import load_xfund_dataset; tr,va,te=load_xfund_dataset('data/xfund', languages=['es','fr','de','it','pt']); print(len(tr), len(va))"`
Expected: about `745 250`.

- [ ] **Step 5: Run the existing suite to get a baseline:** `.venv/Scripts/python.exe tests/test_synthetic_generator.py` and `.venv/Scripts/python.exe tests/test_e2e_link.py`. Expected: both print their pass lines. Record any failure that already exists before changing code.

---

### Task 2: Link-preserving crops

**Files:**
- Modify: `src/data/xfund_windows.py` (add below `window_examples`)
- Test: `tests/test_table_crops.py` (new; runnable as a script with a `__main__` block that calls every test, like `tests/test_synthetic_generator.py`)

**Interfaces:**
- Consumes: `FormExample` (`src/data/funsd_loader.py:48`), with `entities[i] = {"label": str, "token_indices": list[int], ...}`, `links: list[tuple[int,int]]`, boxes in 0–1000.
- Produces:
  - `link_components(ex: FormExample, merge_dist: float = 50.0) -> list[tuple[int, int, int, int]]` returns one region `(x0, y0, x1, y1)` per merged component.
  - `crop_example(ex: FormExample, region: tuple[int, int, int, int], tag: str) -> FormExample | None` returns `None` when no link survives. The guid is `f"{ex.guid}#crop{tag}"`.
  - `crop_examples(examples: list[FormExample], rng: random.Random, max_per_form: int = 4, merge_dist: float = 50.0, pad: tuple[int, int] = (10, 60)) -> list[FormExample]`

- [ ] **Step 1: Write the failing tests** in `tests/test_table_crops.py`. Build `FormExample`s by hand: a 2-column toy form with two key->value pairs far apart, plus one header->2-cells table group.
  - `test_components_follow_links`: `len(link_components(ex)) == 3`. With `merge_dist=1000`, `len(...) == 1`.
  - `test_crop_keeps_links_whole`: for every crop from `crop_examples(...)`, every `(s, d)` in `crop.links` has `s` and `d` in `crop.entities`. The words at `crop.entities[i]["token_indices"]` equal the original entity's words.
  - `test_crop_boxes_renormalised`: all crop boxes are within `[0, 1000]`, and the crop's leftmost word has `x0 <= 60 * 1000 / region_width + 1`, so it sits near the frame edge.
  - `test_form_without_links_gives_no_crops`: an example with `links=[]` returns `[]`.
  - `test_partial_entity_dropped_with_its_links`: when the region covers only one of an entity's two words, that entity is absent and no link touches it.
  - `test_degenerate_region_boxes_in_range`: `crop_example(ex, (500, 500, 500, 500), "z")` either returns `None` or has boxes within 0–1000, with no `ZeroDivisionError`.
  - `test_deterministic_with_seed`: two calls with `random.Random(13)` give the same guids and boxes.
  - `test_max_per_form`: never more than `max_per_form` crops from one form.

- [ ] **Step 2: Run them to make sure they fail**

Run: `.venv/Scripts/python.exe tests/test_table_crops.py`
Expected: `ImportError: cannot import name 'crop_examples'`.

- [ ] **Step 3: Implement the three functions** in `src/data/xfund_windows.py`.
  - Components: union-find over `ex.links`, keeping only entities that have `token_indices`; single unlinked entities are ignored. Repeatedly merge two component boxes if their gap on both axes is under `merge_dist`.
  - `crop_example`: keep the word indices whose box centre lies in `region`, in their original order, and build the old→new index map. Keep an entity only if all its `token_indices` map; keep a link only if both ends are kept. Re-normalise: `x' = round((x - rx0) * 1000 / max(rx1 - rx0, 1))`, then clamp to [0, 1000], and the same for y.
  - `crop_examples`: pick up to `max_per_form` regions with `rng.sample`, pad each side by `rng.randint(*pad)`, and clamp to [0, 1000].
  - Copy the module docstring's tone: "link-preserving crop augmentation (after arXiv 2608.20868)".

- [ ] **Step 4: Run the tests to make sure they pass**

Run: `.venv/Scripts/python.exe tests/test_table_crops.py`
Expected: every test prints passed.

- [ ] **Step 5: Commit**

```bash
git add src/data/xfund_windows.py tests/test_table_crops.py
git commit -m "feat(study-26): link-preserving crop augmentation for joint training"
```

---

### Task 3: Training hooks (crop fields and `init_checkpoint`)

**Files:**
- Modify: `src/train.py`: `TrainConfig` (fields near line 121); `train_one` form-window block (around line 559); `train_single_seed` (around line 1167)
- Test: `tests/test_table_train_hooks.py` (new, script-runnable)

**Interfaces:**
- Consumes: `crop_examples` (Task 2).
- Produces:
  - `TrainConfig` fields: `crop_aug: bool = False`, `crop_max_per_form: int = 4`, `crop_merge_dist: float = 50.0`, `crop_seed: int = 13`, `init_checkpoint: str = ""` (may contain `{seed}`).
  - `apply_init_checkpoint(model: torch.nn.Module, config: TrainConfig, seed: int) -> str | None` in `src/train.py` returns the resolved path, or `None` when the field is unset.

- [ ] **Step 1: Write the failing tests**
  - `test_init_checkpoint_roundtrip`:
    - save the `state_dict` of a `torch.nn.Linear(4, 3)` with fixed weights to a temp `seed_13/best_model.pt`
    - set `cfg.init_checkpoint = str(tmp / "seed_{seed}" / "best_model.pt")`
    - call `apply_init_checkpoint` on a fresh `Linear(4, 3)` with seed 13
    - check the outputs on a fixed input are equal, and the returned path ends with `seed_13/best_model.pt`
  - `test_missing_init_checkpoint_raises`: with seed 21 and no file, a `FileNotFoundError` is raised whose message contains `seed_21`.
  - `test_unset_init_checkpoint_noop`: with an empty field, the function returns `None` and the weights are unchanged.
  - `test_crop_config_fields_default_off`: `TrainConfig()` has `crop_aug is False` and `init_checkpoint == ""`.

- [ ] **Step 2: Run them to make sure they fail**

Run: `.venv/Scripts/python.exe tests/test_table_train_hooks.py`
Expected: `ImportError` or `AttributeError` on `apply_init_checkpoint`.

- [ ] **Step 3: Implement**
  - Add the fields.
  - `apply_init_checkpoint`: `torch.load(path, map_location="cpu")`, then `model.load_state_dict(state, strict=True)`, and print `[init] loaded <path>`.
  - Call it in `train_single_seed` right after `model = build_model(...)` (line 1167).
  - In `train_one`, **before** the `form_window` block, when `config.crop_aug` is set, run `train_examples = train_examples + crop_examples(train_examples, random.Random(config.crop_seed * 1000 + seed_if_known_else_0), config.crop_max_per_form, config.crop_merge_dist)`, then print `[crops] +N train examples`. Use `config.crop_seed` alone if `train_one` doesn't know the seed; don't add a parameter just for this. Validation and test are never cropped.

- [ ] **Step 4: Run the tests to make sure they pass**, plus the existing `tests/test_e2e_link.py` to show the default path still works.

Run: `.venv/Scripts/python.exe tests/test_table_train_hooks.py && .venv/Scripts/python.exe tests/test_e2e_link.py`
Expected: both pass.

- [ ] **Step 5: Commit**

```bash
git add src/train.py tests/test_table_train_hooks.py
git commit -m "feat(study-26): crop_aug and init_checkpoint training hooks"
```

---

### Task 4: Relation-rich synthetic generator

**Files:**
- Modify: `src/data/synthetic_generator.py`
- Test: `tests/test_synthetic_generator.py` (add tests and register them in its `__main__`)

**Interfaces:**
- Produces:
  - `FormSection(title: str, fields: list[FormField])` dataclass, and `FormDocument.sections: list[FormSection] = field(default_factory=list)`.
  - `FAMILIES: dict[str, Callable]` with the keys `invoice`, `financial_statement`, `procurement_order`, `shipping_manifest`, `medical_intake`, `adbuy_contract`, `fara_registration`.
  - `generate_synthetic_dataset(output_dir, n_train=100, n_test=30, seed=42, use_augraphy=True, train_families: list[str] | None = None, test_families: list[str] | None = None, stacked_ratio: float = 0.0, tall_tables: bool = False, locales: list[str] | None = None, save_images: bool = True) -> dict`. With `None` families it behaves as before (all 7, round-robin).
  - CLI flags: `--train_families`, `--test_families` (comma-separated), `--stacked_ratio`, `--tall_tables`, `--locales`, `--no_images`.

- [ ] **Step 1: Write the failing tests**
  - `test_sections_emit_question_to_question_links`: render a `FormDocument` with one `FormSection("Applicant Details", [3 fields])`. Its annotation has a `question` entity with that title, and it has links to the 3 key entities, all labelled `question`.
  - `test_stacked_ratio_one_puts_values_below_keys`: with `SyntheticFormRenderer(fake, use_augraphy=False, stacked_ratio=1.0)` on a horizontal-field document, every linked value's box `y0` is greater than its key's `y1`.
  - `test_family_split_is_disjoint`: `generate_synthetic_dataset(tmp, n_train=10, n_test=4, use_augraphy=False, train_families=["invoice","fara_registration"], test_families=["medical_intake"], save_images=False)`. Every train annotation filename contains `invoice` or `fara`, and every test filename contains `medical`.
  - `test_tall_tables_have_many_rows`: with `tall_tables=True`, an invoice document's table has ≥ 8 data rows. Also count header->cell links with the destination ≥ 5 entities away by centre-distance rank, and require at least one.
  - `test_locales_change_values_not_keys`: with `locales=["es_ES"]`, key texts are unchanged and the run doesn't error.
  - Existing tests still pass, so the default behaviour is unchanged.

- [ ] **Step 2: Run them to make sure they fail**

Run: `.venv/Scripts/python.exe tests/test_synthetic_generator.py`
Expected: `ImportError: FormSection` (or a `TypeError` on the new kwargs).

- [ ] **Step 3: Implement**
  - The renderer draws each section title as a `question` entity, linked to each child field's key id, followed by the child fields; reuse the existing field-rendering block.
  - Add sections (2–3 each) to `adbuy_contract`, `fara_registration` and `medical_intake`.
  - `stacked_ratio`: per horizontal field, `random.random() < stacked_ratio` switches it to the existing `"stacked"` layout, which already puts the value under the key.
  - `tall_tables`: give the 4 table families an `n_rows: int | None = None` keyword, passed `random.randint(8, 25)`.
  - `locales`: document `i` uses `Faker(locales[i % len(locales)])` seeded with `seed + i`, for values only; key strings stay hard-coded.
  - `save_images=False` skips writing the PNG but still writes the annotation; set the image path in the JSON as before so `load_funsd_split` doesn't break, and check that it doesn't open the image.

- [ ] **Step 4: Run the tests to make sure they pass**

Run: `.venv/Scripts/python.exe tests/test_synthetic_generator.py`
Expected: `ALL SYNTHETIC GENERATOR TESTS PASSED`.

- [ ] **Step 5: Check that the generated data loads end to end**

Run: `.venv/Scripts/python.exe -m src.data.synthetic_generator --output_dir data/synthetic_table --n_train 50 --n_test 10 --no_augraphy --no_images --train_families invoice,financial_statement,procurement_order,adbuy_contract,fara_registration --test_families shipping_manifest,medical_intake --stacked_ratio 0.4 --tall_tables --locales es_ES,fr_FR,de_DE,it_IT,pt_PT`, then `load_funsd_split('data/synthetic_table','train')`.
Expected: 50 examples, each with `links` non-empty.

- [ ] **Step 6: Commit**

```bash
git add src/data/synthetic_generator.py tests/test_synthetic_generator.py
git commit -m "feat(study-26): relation-rich synthetic forms (nested sections, stacked values, tall tables, family split)"
```

---

### Task 5: Evaluation (group recall, synthetic test, summary and verdict)

**Files:**
- Modify: `scripts/xfund_e2e_eval.py`
- Create: `scripts/table_links_summary.py`
- Test: `tests/test_table_eval.py` (new, script-runnable)

**Interfaces:**
- Consumes: `entity_boxes`, `direction`, `rank_of` from `scripts/xfund_link_errors.py`; `gold_structure`, `add_counts`, `prf` from `src/e2e_link.py`.
- Produces:
  - `link_groups(ex) -> dict[str, set[Link]]` in `scripts/xfund_e2e_eval.py`, with keys `below`, `same_line`, `above`, `rank_0`, `rank_1`, `rank_2_4`, `rank_5plus`, `q_to_q`. Word-range links use the same representation as `gold_structure`.
  - New result keys `"{sub}|{mode}|group|{name}"`, each a `prf` dict; computed with `add_counts(acc, links & G, G)`, so only recall is meaningful.
  - CLI: `--data {xfund,synthetic}` (default `xfund`) and `--data-root`. For `synthetic`, docs = `load_funsd_split(data_root, "test")`, and the `sibling_free` subset is skipped.
  - `verdict(deltas_pts: list[float]) -> str` in `scripts/table_links_summary.py` returns `helps`, `no_better`, `worse` or `inconclusive`.
  - Summary CLI: `--arms A0=path A1=path ...` writes per-arm means, seed-matched deltas vs A0 for the primary cell and every group cell, and a verdict per arm. A `selftest` sub-command is included.

- [ ] **Step 1: Write the failing tests**
  - `test_groups_partition_links`: on a hand-built example with 4 links (1 below, 2 same line, 1 above, one of them question->question) plus 1 link to an entity with no tokens, `below | same_line | above` equals the 4 counted links and they don't overlap. The rank groups also partition the same 4. `q_to_q` has exactly 1.
  - `test_group_recall_counts`: when the predicted links are the 2 same-line links, the `same_line` recall is 1.0 and the `below` recall is 0.0.
  - `test_verdict_rules`:
    - `verdict([2.5, 2.1, 3.0]) == "helps"`
    - `verdict([0.5, -0.3, 0.4]) == "no_better"`
    - `verdict([-1.5, -2.0, -1.2]) == "worse"`
    - `verdict([1.5, 1.2, 1.8]) == "inconclusive"`
    - `verdict([2.5, -0.1, 4.0]) == "no_better"` (mixed signs)

- [ ] **Step 2: Run them to make sure they fail**

Run: `.venv/Scripts/python.exe tests/test_table_eval.py`
Expected: `ImportError: link_groups`.

- [ ] **Step 3: Implement**
  - `link_groups` maps entity ids to word ranges exactly as `gold_structure` does.
  - Direction: `direction()` outputs starting with "same line" go to `same_line`.
  - Rank: `rank_of(s, d, boxes, types)` with `types[i] = label`, bucketed into 0, 1, 2–4 and 5+.
  - `q_to_q`: both labels are `question`.
  - In `evaluate_seed`, add the group counts next to the existing link counts, for both modes and both subsets.

- [ ] **Step 4: Run the tests and the summary selftest**

Run: `.venv/Scripts/python.exe tests/test_table_eval.py && .venv/Scripts/python.exe scripts/table_links_summary.py selftest`
Expected: both pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/xfund_e2e_eval.py scripts/table_links_summary.py tests/test_table_eval.py
git commit -m "feat(study-26): per-group end-to-end link recall, synthetic test path, summary and verdict"
```

---

### Task 6: Configs and CPU smoke for every arm

**Files:**
- Create:
  - `configs/xfund_synth_pretrain.yaml`
  - `configs/xfund_table_A0.yaml`, `configs/xfund_table_A1.yaml`, `configs/xfund_table_A2.yaml`, `configs/xfund_table_A3.yaml`
  - `scripts/smoke_table_links.py`

**Interfaces:**
- Consumes: Tasks 2–5.
- Produces:
  - Output dirs `results/table_links/pretrain`, `results/table_links/A0` … `A3`.
  - The A1 and A3 `init_checkpoint` is `results/table_links/pretrain/seed_{seed}/best_model.pt`.

- [ ] **Step 1: Write the configs.**
  - **A0–A3:** each is a copy of `xfund_joint_lilt_windows.yaml` with `output_dir` changed. A2 and A3 add `crop_aug: true`; A1 and A3 add `init_checkpoint`.
  - **Pretrain:** the same joint/link settings, plus:
    - `dataset: funsd`, `data_root: data/synthetic_table`
    - `max_phase1_epochs: 2`, `max_phase2_epochs: 2`, `max_phase3_epochs: 3`, `epochs: 7`
    - `form_window: 510`, `output_dir: results/table_links/pretrain`

    The funsd loader carves validation from train (training families) and uses test as the held-out families.
- [ ] **Step 2: Write `scripts/smoke_table_links.py`**, modelled on `scripts/smoke_xfund_e2e.py`: 1 epoch per phase, batch 1, 16 train and 8 val forms. It runs, in order:
  1. a pretrain on 10 synthetic documents
  2. A0, A2, A1 and A3 mini-runs from the matching configs, calling `train_single_seed` so `init_checkpoint` is exercised
  3. `evaluate_seed` on 3 val documents per arm, asserting that every `all|windowed|group|*` key exists
  4. the synthetic `--data synthetic` evaluation on 3 documents

  It raises on any failure and deletes its temp output dirs.
- [ ] **Step 3: Run it**

Run: `.venv/Scripts/python.exe scripts/smoke_table_links.py`
Expected: it ends with `SMOKE OK`. Scores are meaningless at this size. This downloads LiLT weights the first time.

- [ ] **Step 4: Commit**

```bash
git add configs/xfund_synth_pretrain.yaml configs/xfund_table_A0.yaml configs/xfund_table_A1.yaml configs/xfund_table_A2.yaml configs/xfund_table_A3.yaml scripts/smoke_table_links.py
git commit -m "feat(study-26): arm configs and CPU smoke test"
```

---

### Task 7: Kaggle notebook, final check and push

**Files:**
- Create: `notebooks/kaggle_table_links.ipynb` (start from a copy of `notebooks/kaggle_windowed_e2e.ipynb`)
- Modify: `STUDIES.md` (add a "Checks done before any GPU spend" paragraph to Study 26)

- [ ] **Step 1: Adapt the 7 cells.**
  1. Clone `feat/table-links` with `GITHUB_TOKEN` and install deps.
  2. Fetch XFUND, then run the generator command from Task 4 Step 5 with `--n_train 3000 --n_test 300`.
  3. `python scripts/smoke_table_links.py`; raise on failure.
  4. Run, in order:
     - `train.py --config configs/xfund_table_A0.yaml`
     - then A2
     - then the pretrain config
     - then A1
     - then A3

     Each is wrapped so a failure is logged and the next one still runs; this cell never raises.
  5. Run `xfund_e2e_eval.py --ckpt-dir results/table_links/<arm> --out results/table_links/<arm>_e2e.json` per arm, plus the synthetic evaluation for A0, A1 and A3, then `table_links_summary.py` to `results/table_links/summary.json`.
  6. Display the summary table.
  7. Push `results/table_links/**/*.json` only.
- [ ] **Step 2: Validate the notebook JSON**

Run: `.venv/Scripts/python.exe -c "import json; nb=json.load(open('notebooks/kaggle_table_links.ipynb', encoding='utf-8')); print(len(nb['cells']))"`
Expected: `7` (or the source notebook's cell count, if it has markdown cells too).

- [ ] **Step 3: Rerun every Study 26 test plus the existing suite**

Run: `.venv/Scripts/python.exe tests/test_table_crops.py && .venv/Scripts/python.exe tests/test_table_train_hooks.py && .venv/Scripts/python.exe tests/test_synthetic_generator.py && .venv/Scripts/python.exe tests/test_table_eval.py && .venv/Scripts/python.exe tests/test_e2e_link.py`
Expected: all pass.

- [ ] **Step 4: Record the pre-run checks in Study 26 and commit**

```bash
git add notebooks/kaggle_table_links.ipynb STUDIES.md
git commit -m "feat(study-26): 7-cell Kaggle notebook; record pre-run checks"
```

- [ ] **Step 5: Push the branch, so the notebook's clone works:** `git push -u origin feat/table-links` (ask the user before pushing).

After this the user runs the notebook on Kaggle. Results are analysed in a follow-up: check that A0 is within about 1.5 link F1 of Study 19's 47.7 before reading the other arms.
