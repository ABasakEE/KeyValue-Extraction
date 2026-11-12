# Arjoe handoff

Written 13 Nov 2026 by Param, with Claude. It covers the work done 8–13 Nov on branches `feat/table-links` and `feat/xfund-vlm`, plus what is left before the novelty assessment on 23/25 Nov.

The full record is `STUDIES.md`: Study 26 (results in) and Study 27 (registered, not run yet). This file is the short version, plus what to do next.

---

## 1. Where things stand

**Assessment goal (23/25 Nov).** We want to show two things:
1. We used recent research papers to get better results. **Done: Study 26.**
2. We also tried the VLM route on our main task. **Built and tested on CPU, not run yet: Study 27.**

**Branches.** Each branch is cut from the one before it, so `feat/xfund-vlm` contains everything:

```
main
 └─ feat/bros-relative-layout   Study 24 (multilingual LiLT): code done, marked pending in STUDIES.md
     └─ feat/table-links        Study 26: done, results committed
         └─ feat/xfund-vlm      Study 27: code + notebook done, waiting for the Kaggle run   <-- work here
```

Studies 24 and 25 are also still marked *pending* in `STUDIES.md`. Ask Param whether those runs happened before you count on them.

**Main metric used everywhere below.** End-to-end link F1 on the 250 XFUND val forms (es/fr/de/it/pt):
- The model has to find the entities itself (header / question / answer) and then link each key to its value.
- A link counts only if both ends have exactly the right words.
- Scoring code: `src/e2e_link.py` and `scripts/xfund_e2e_eval.py`.

This is much stricter than published XFUND numbers. Those give the model the correct entities, so our numbers can't be compared with theirs.

---

## 2. Study 26: table-type links (done)

**Problem.** Study 22 showed where our joint LiLT linker fails. Its best score was 47.7 (Study 19). The weak spots were:
- the value sits **below** its key: recall 49, against 79 for same-line pairs
- the partner is **far away** (5+ entities)
- **heading → field-label** links (question→question)

These are exactly the links tables are built from.

**What we tried.** Two training-data ideas from recent papers, with the model left unchanged:
- **Link-preserving crops**, from *Identify, Locate, Link* (ICDAR 2026, arXiv 2608.20868). We cut extra training examples out of each form around groups of linked fields, so the model sees more close-ups of tables and sections.
- **Relation-rich synthetic pretraining**, from *RIDGE* (CVPR 2025, arXiv 2504.10659). We pretrain on 3,000 generated forms with nested sections, keys above values and tall tables, then train on XFUND as usual.

**Arms.** 3 seeds each (13 / 21 / 42), compared seed by seed against A0. A0 reproduced Study 19 (47.42 vs 47.7, inside the 1.5 tolerance), so verdicts could be drawn.

| Arm | What it is | Link F1 | Change vs A0 per seed | Verdict (registered rule) |
|---|---|---|---|---|
| A0 | baseline (Study 19 recipe, retrained) | 47.42 | | reference |
| A1 | synthetic pretraining | 47.50 | −1.17 / −0.34 / +1.73 | no better |
| A2 | crops | **49.70** | +1.36 / +2.27 / +3.19 | **helps** |
| A3 | both | **49.89** | +0.89 / +2.76 / +3.76 | **helps** |

**Where the gain comes from.** Recall gain over A0, in points:

| Link type | A2 (crops) | A3 (both) |
|---|---|---|
| key above value | +4.0 | +4.5 |
| key below value | +1.8 | +2.7 |
| partner 6th-nearest or further | +2.0 | +2.7 |
| heading→label | +0.8 | +2.4 |
| same line | +0.7 | +1.2 |

The gains are largest on the weak spots Study 22 found; same-line pairs, which were already fine, barely move. This is the strongest point to make in the assessment.

**Other results:**
- Entity F1 goes up about 1 point with crops (75.8 → 77.0).
- Question→answer link F1 goes from 53.2 to 54.9.
- On the forms with no near-identical template in training ("sibling-free"), the gain holds: 46.3 → 48.4.

**Caveats. Say these in the assessment; don't hide them.**
1. **Extra training steps.** Crops add examples, so A2 and A3 took about 2× A0's gradient steps. Study 21 found extra steps alone are worth about +1.2. Part of the +2.3 may be steps rather than crops, and this study can't separate the two. Fix: task 4.2 below.
2. **Synthetic held-out test goes the other way.** On 300 synthetic forms from two families never used in training, every trained arm scores below A0, and crops fall most (49.1 → 38.5; per seed −1.5 / −16.9 / −13.5). Our untested guess: crops are stretched back to full-page coordinates, which makes word boxes about 5× taller than on real pages.
3. Only 3 seeds, so there are no confidence intervals.
4. **A3 vs A2.** A3 is only +0.2 over A2, and the seeds disagree. Pretraining adds nothing we can trust.
5. **A1 and A3 started from different pretraining runs.** The weights can't be carried between Kaggle sessions.

**Suggested claim:**
> Link-preserving crop augmentation (Identify, Locate, Link, ICDAR 2026) raised end-to-end XFUND link F1 from 47.4 to 49.7, positive in all three seeds. The gain is largest on vertical, long-range and heading links, the links tables depend on. Synthetic pretraining (RIDGE, CVPR 2025) gave no measurable gain. On unseen synthetic layouts the crops hurt, and part of the gain may come from extra training steps.

**Files:**
- `results/table_links/`: per-arm `*_e2e.json`, `*_synth_e2e.json` and `summary.json`
- notebook: `notebooks/kaggle_table_links.ipynb`
- crop code: `crop_examples()` in `src/data/xfund_windows.py`
- generator changes: `src/data/synthetic_generator.py`

**Run notes.** Sessions 2 and 3 couldn't push their results. A failed `git pull --rebase` left `.git/rebase-merge` behind, and every later push failed. Param downloaded the A1/A3 JSONs by hand and we checked them before use. `push_results` in the notebooks now clears that state first (commit 0f35b99). The A1/A3 training logs are lost; no reported number depends on them.

---

## 3. Study 27: the VLM route (your first job)

**Earlier VLM work.** Studies 6–18 and 25 used Qwen2.5-VL-3B, but only for **field extraction** on VRDU/FATURA. None of them tried it on our actual task, finding XFUND entities and linking them.

**What Study 27 does.** It runs that task zero-shot, with no training, using Qwen2.5-VL-**7B**. It is scored with exactly the same rules as LiLT. Param's brief: the route has to be *tried properly* (fair setup, honest scoring, more than one attempt). It does not have to win.

**The three arms:**

| Arm | What the VLM is given | What it writes | Compare with |
|---|---|---|---|
| **V2 (main)** | page image + the form's words, numbered, grouped into lines with positions | one item per line: `Q 2-2 > A 3-5` (link) or `H 0-1` (lone entity), using word numbers | LiLT end to end: 47.4 (A0) / 49.7 (A2) |
| **V3** | page image + the correct entities, numbered | links only: `e1>e2` | LiLT with correct entities, about 59.8 (Study 17) |
| **V1** | page image only | entity texts, which we match back onto the words | V2, to show what giving it the words adds |

**Design choices:**
- Same forms and gold as LiLT, with the words in the same order LiLT reads them.
- Unparseable answers count as empty predictions; no form is dropped.
- A link written backwards (value→key) is flipped using the VLM's own labels, the same rule for every arm.
- Greedy decoding, so there are no seeds.
- The forms run in a fixed shuffled order that cycles through the five languages, so a run cut short stays balanced.

**Ceilings, measured on CPU with perfect answers:**
- **V2:** 99.8. Links that touch unlabelled "other" text can't be written; LiLT can't find those either.
- **V3:** 100.
- **V1:** about 90 link F1, with 93% of entities recovered. Checkbox marks and bullets can't be retyped from the image. Report this next to V1's score.

**Code (all on `feat/xfund-vlm`):**
- `src/vlm_link.py`: prompts, parsers, the form order, and the "perfect answer" renderers used for testing
- `scripts/xfund_vlm_generate.py` (GPU): runs one arm and appends raw answers to `results/xfund_vlm/{arm}_raw.jsonl`. Re-running skips forms already done. `--stub` writes perfect answers without a model, for testing.
- `scripts/xfund_vlm_score.py` (CPU): scores raw answers with the unchanged `src/e2e_link.py` rules. Writes `{arm}_e2e.json` with:
  - entity / link / question→answer F1
  - recall per link type
  - the sibling-free subset
  - a bootstrap 95% CI
  - parse-failure and truncation rates
  - per-language link F1

  `--common` scores several arms on the forms they share.
- `tests/test_vlm_link.py`: plain Python, run it with `.venv/Scripts/python.exe tests/test_vlm_link.py`. All 9 tests pass.
- `notebooks/kaggle_xfund_vlm.ipynb`: the Kaggle run.

### How to run it

1. In Kaggle, import `notebooks/kaggle_xfund_vlm.ipynb` from the repo, branch `feat/xfund-vlm`.
2. Settings:
   - Accelerator **GPU T4 x2**. Two GPUs are needed, because the 7B model is split across both.
   - Internet **on**.
   - Add-ons → Secrets → attach **`GITHUB_TOKEN`**. It needs push access to `ABasakEE/KeyValue-Extraction`.
3. **Run all.** What each cell does:
   - **Cell 2** downloads the XFUND JSONs and the five `{lang}.val.zip` image archives (about 80 MB each) from the official GitHub release, then checks that all 250 images are there.
   - **Cell 3 (smoke test)** runs 3 V2 forms, then 1 form each of V3 and V1, scores them and prints the first answers. **Read those answers.** They are the first real VLM output. Cell 3 raises an error if a form fails or every V2 answer is unparseable.

     It also applies the registered model rule. If V2 is projected above 6 h on 250 forms, the run switches to Qwen2.5-VL-3B. The choice is saved in `results/xfund_vlm/model_choice.json`.
   - **Cell 4** runs V2 (250 forms, cap 6 h), then V3 (250 forms, cap 2.5 h), then V1 (first 100 forms, cap 2.5 h). It works in chunks of 50 and pushes each chunk, so a killed session loses at most one chunk. Re-running resumes.
   - **Cells 5–7** score each arm and the shared-forms comparison, print a table next to the LiLT numbers, and push.
4. **Env overrides:**
   - `VLM_ARMS=V3,V1` runs only those arms, for example in a second session.
   - `TIME_BUDGET_H` (default 11).
   - `VLM_MODEL=...` forces a model and skips the timing rule.

If V2 eats the whole session, run a second session with `VLM_ARMS=V3,V1`. V2's finished forms are on GitHub and are skipped.

### After the run
1. Before reading any F1, check `parse_fail_rate`, `error_rate` and `truncated_rate`. A high parse-fail rate means the format didn't land; that is a finding, but say so.
2. Write the Results section of Study 27 in `STUDIES.md`, in the same style as Study 26's:
   - the main table with V2's 95% CI
   - recall per link type
   - per-language link F1
   - the V1 ceiling
   - run notes

   Then update its summary row at the top of the file.
3. Add a short paragraph to `NOVELTY_ASSESSMENT.md` §4.2 (the VLM thread). Also update §7.4: it lists "larger VLM" and "grounded output" as untried, and Study 27 tries both.
4. **Don't tune the prompts on these results and re-report.** The study is registered. Any new prompt is a new, labelled post hoc arm.

### If something breaks
- **Out of memory on a form.** The script catches it, records the error, scores that form as empty and moves on. If it happens a lot, lower `--max-pixels` in `scripts/xfund_vlm_generate.py` (default 1280·28·28).
- **Push fails.** The results stay in `/kaggle/working/KeyValue-Extraction/results/xfund_vlm/`. Download them before the session ends.
  - We have hit file converters renaming downloads before (the A1/A3 files came back with a `_MConverter.eu_` prefix). Download the raw files directly.
  - Then commit them locally and re-run cells 5–6 on CPU to score.
- **Garbage output in float16 on T4** (empty or `!!!!` text). Try `VLM_MODEL=Qwen/Qwen2.5-VL-3B-Instruct`. The 3B model ran fine in float16 in Studies 6–18. Record this as an amendment.

---

## 4. Further tasks, in priority order

### 4.1 Run Study 27 (above)

### 4.2 Step-matched control for Study 26 (one Kaggle session, about 4–6 h)
This answers caveat 1: is A2's +2.3 from crops, or from about 2× the gradient steps?
- **Not built yet.** Use Study 21 as the template. It matched steps by raising the curriculum epoch caps: `configs/xfund_joint_lilt_matched.yaml` and `notebooks/kaggle_matched_steps.ipynb`.
- **Here:** copy `configs/xfund_table_A0.yaml` and scale the epoch caps and `epochs` so A0 gets about the same number of steps as A2. A2 sees 3,276 windows per epoch against A0's 1,537, so about 2.13×.

  Watch out: Study 21's control stopped early, at 35–43 of 53 epochs, so it got fewer steps than intended. Report the steps actually taken.
- **Register it in `STUDIES.md` first** (Study 28) with a rule, for example:
  - "crops matter": A2 minus control ≥ +1.0 and positive in all seeds
  - otherwise: "steps explain most of it"
- **Optional second arm, if time allows:** crops rescaled with **one** factor for x and y, so box heights stay realistic. This tests the synthetic held-out drop (caveat 2).

### 4.3 Novelty assessment write-up
- `NOVELTY_ASSESSMENT.md`: add Study 26 (the claim above, the link-type table, the caveats) and Study 27.
- Slides and report for 23/25 Nov:
  - Check the course guidelines in `intensive_assessment/GUIDELINES.md`; they override any handoff file.
  - Keep the text plain, and use real project figures, for example a form with the crop regions drawn on it, or a VLM answer next to the gold links.

### 4.4 Housekeeping
- Merge order once the results are in: `feat/bros-relative-layout` → `feat/table-links` → `feat/xfund-vlm` → `main`. Or open one PR from `feat/xfund-vlm`.
- Delete the local SDD scratch folder `.superpowers/sdd/2026-11-08-table-links/` (it isn't tracked).
- Studies 24 and 25 are still "pending" in `STUDIES.md`. Find out whether they ran and fill them in or mark them not run.

---

## 5. How we work in this repo
- **Register before running.** Each study goes into `STUDIES.md` (question, arms, registered criteria, limits) and is committed before any code or run. If something changes before the GPU run, add a dated **pre-run amendment**. Never quietly edit the registration. Post hoc analyses are labelled post hoc.
- **Kaggle notebooks use a fixed 7-cell layout:** setup/clone, data, smoke (raises), main loop (never raises; resumable; pushes after each unit), score, display, push.
- **Pushes are JSON/JSONL only, never weights or data.** `push_results` adds files with `git add -f`, because `results/` is gitignored.
- **Local environment:**
  - Python is `.venv/Scripts/python.exe` (CPU torch). Plain `python` has no torch.
  - The XFUND JSONs are in `data/xfund/`; the images aren't stored locally.
- **Don't publish numbers that aren't backed by a committed results file.**
