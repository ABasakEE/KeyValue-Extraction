# Video Script — IE 643 Prep Presentation

**Team: Unemployed and Unsupervised** · Param Mehta (23b2439), Arjoe Basak (23b1295)

**Measured length: see the Timing table at the foot of this file.** Hard cap 15:00 —
instruction 15 penalises overrun. Both presenters speak. Facecam on throughout.
**Do not read the slides** — the slides carry the numbers, the speaker carries the argument.

Paragraphs marked **[CUT IF LONG]** are droppable whole, without losing a claim. Drop all four if
your timed rehearsal comes in over 13:30.

**Speaker split.** PM opens and owns the empirical half; AB owns the literature half and closes.

| Block | Slides | Speaker |
|---|---|---|
| Problem and contract | 1–4 | **PM** |
| Background and literature | 5–12 | **AB** |
| Data, protocol, experiments | 13–19 | **PM** |
| Gap, roadmap, close | 20–25 | **AB** |

Recording notes: OBS, 1080p, slides as window capture with facecam bottom-right.
Record in four takes matching the four blocks — a fluffed line costs one block, not the whole video.

---

## BLOCK 1 — PM

### Slide 1 · Title

> Hi, I'm Param Mehta, with Arjoe Basak. We're team Unemployed and Unsupervised, and this is our
> prep presentation for IE 643.
>
> Our topic is cross-template key-value extraction from scanned forms. The hard part is in the
> subtitle: we have to generalise to form templates we have zero training data for. Not a few
> examples — none.

### Slide 2 · Outline

> Quick map. Arjoe takes the background reading and where the literature disagrees. I come back for
> our datasets, our split protocol, and the two experiments we've run. Arjoe closes with the gap
> we're claiming and the roadmap.

*Delivery: gesture at the two columns. Do not read the eight headings.*

### Slide 3 · The Task

> Why this is hard. Train an extractor on one layout and test it on the same layout, and it scores
> well while learning almost nothing useful. It has learned that the invoice number sits top-right,
> and that the value is the box to the right of the key — facts about *that template*, not about
> forms.
>
> Change the template and three priors break at once: absolute position, reading order, and the
> spatial key-value convention. That simultaneity is what separates this from ordinary fine-tuning.

### Slide 4 · The Zero-Data Contract

> "Zero same-template data" is stricter than it sounds. No labelled examples of the held-out
> template, obviously. But also **no unlabelled examples**, which rules out most of domain
> adaptation. And the one people get wrong: **no validation use**. Pick your checkpoint by scoring
> on the held-out template and you have leaked, even though no gradient ever flowed.
>
> This is an assertion in our code, printed every run, tested with a negative control.

> **Handoff:** Over to Arjoe for what we read.

---

## BLOCK 2 — AB

### Slide 5 · The Standard Pipeline

> Thanks Param. The classical pipeline runs along the top: scan, OCR — giving words *and* boxes — a
> layout-aware encoder, per-token tags, then key-value links.
>
> Two things to hold onto. Three modalities: what a token says, where it sits, how it looks. Every
> architectural fight we're about to see is about fusing those. And **tagging and linking are
> different problems** — one is per-token classification, the other is relational. The literature
> measures them under very different conditions.

### Slide 6 · Family 1 — Layout-Aware Encoders

> Family one, the LayoutLM line.
>
> Version one bolts *absolute* 2D position embeddings onto BERT — this token is at x equals four
> hundred. Version two makes it *relative*: the attention score between two tokens picks up a bias
> from the offset between their boxes. The diagram shows why that matters. Shift the form, and every
> absolute coordinate changes while every relative offset survives.
>
> **[CUT IF LONG]** The second axis is vision — CNN regions in v2, ViT patches at token granularity
> in v3, which reaches ninety-point-three on FUNSD.

### Slide 7 · Family 2 — Decoupling and Relative Encoding

> Family two pushes that further, in two directions.
>
> **LiLT**, our primary model, splits into a text tower and a layout tower, coupled by BiACM. The
> critical detail is in the diagram: the gradient from layout back into text is **detached**. The
> argument is that layout is language-agnostic, so don't entangle it with one vocabulary. They prove
> it by swapping the text tower and transferring to seven unseen languages.
>
> Our honest caveat: that is *cross-lingual* evidence. We're betting it helps *cross-template*.
> That's an extrapolation, and testing it is part of the contribution.
>
> **BROS** goes the other way — no image at all — and still beats LayoutLMv2, which has one.

### Slide 8 · Family 3 — Graphs and Parsing

> Family three says: if linking is relational, stop pretending the document is a sentence.
>
> Model it as a graph. Nodes are text regions, so tagging is node classification. Edges are candidate
> relations, so linking is **edge** classification. No reading order anywhere — and reading order is
> exactly what breaks on a new template.
>
> The number I'd point to: a plain long-sequence transformer gets sixty-six. Add Rich Attention and
> graph pooling, eighty-four and a half. **Eighteen points** from structural encoding alone — the
> clearest evidence we found that sequence order is the wrong prior for forms.

### Slide 9 · Family 4 — OCR-Free and Generative

> Family four throws the pipeline out. Donut goes image straight to JSON. DocLLM keeps boxes but
> splits attention four ways.
>
> The cost is on the right. DocLLM's own FUNSD number is **fifty-one-point-eight**, against
> eighty-three to ninety-three for the discriminative encoders, with far more parameters. Free
> generation gives no structural guarantee of a well-formed span.

### Slide 10 · Where the Literature Disagrees

> Two questions we expected settled answers to.
>
> Does vision help? BROS says no, and beats an image-using model without one. LayoutLMv3 and
> FormNetV2 say yes, but only fused at token granularity. Nobody has run the controlled ablation, so
> both sides argue from confounded comparisons.
>
> Do generative models win? LMDX drops under five F1 seen-to-unseen, where LayoutLMv2 drops nineteen
> to twenty-seven. DocLLM says the opposite. They may generalise better while being worse at exact
> span boundaries — never tested in one experiment.

### Slide 11 · What Is Actually Measured

> My most important slide. Three results that changed our plan.
>
> **One** — the gap is real. FormNet on VRDU: ninety-point-five seen, seventy-seven-point-three
> unseen. Thirteen points, and thirteen to seventeen across models. That's our calibration target.
>
> **Two**, and this cost us a design decision. We had justified BROS on the grounds that relative
> encoding should help on new templates. Then we found KNN-Former's unseen-template split, where BROS
> scores **twenty-three** — worse than plain LayoutLM at forty-seven. Our argument was simply wrong.
> BROS stays, but for its linking head.
>
> **Three** — Do-GOOD decomposes the shift, and pure layout novelty is five F1 of a thirty-two point
> drop. Tune positional encodings all semester and you optimise five points of a thirty-two point
> problem.

*Delivery: slow down here. This slide shows we read critically rather than collected citations.*

### Slide 12 · Benchmark Integrity

> And a warning about the numbers themselves.
>
> Someone measured template duplication between train and test. SROIE is **seventy-five percent**
> duplicated, so those generalisation numbers are largely measuring memorisation.
>
> FUNSD is sixteen percent, but has its own flaw: block-level annotation gives every token in an
> entity identical coordinates, so models learn "block boundary equals entity boundary" as a shortcut.
>
> **[CUT IF LONG]** Its linking labels were noisy enough that another group re-annotated the whole thing.
>
> So FUNSD validates our **code**. It cannot validate our **claim**.

> **Handoff:** Back to Param.

---

## BLOCK 3 — PM

### Slide 13 · Dataset Survey

> We surveyed what's available and hit a wall. Look at the last two columns.
>
> The dataset with an official unseen-template protocol — VRDU — has no linking annotation. The
> datasets with linking annotation have no template split. No single dataset lets us ask our question
> directly. That's not a gap in our search; that's the shape of the field, and it's why the question
> is worth asking.

### Slide 14 · Datasets We Will Use

> So, three sources. VRDU primary — the only ready-made unseen-template protocol, and its published
> gap calibrates ours. FUNSD through the corrected RFUND annotation, for the linking half only, never
> as generalisation evidence. And our own synthetic data, which is the next slide.

### Slide 15 · Our Split Protocol

> The contract from slide four, made concrete. Top row, seen: A, B and C in training and in test.
> Bottom row, unseen: train on B and C, test on A alone. One fold per held-out template.
>
> **Matched training size** matters — if the two regimes see different numbers of documents, you
> haven't measured a generalisation gap, you've measured a sample-efficiency curve. Our code errors
> out rather than report one.
>
> And something we didn't expect. When we audited **VRDU's own official unseen-template splits**, the
> held-out template shows up in *validation* on all three folds — in one case, a hundred documents
> out of a hundred. So we implemented both: theirs for comparability, and a strict one. Everything I
> show next uses the strict one.

*Delivery: this is our strongest "we did real work" moment. Land it, don't rush past it.*

### Slide 16 · Data Curation Pipeline

> Our own data, left to right. Real blank fillable PDFs — about fifty-five thousand, plus IRS forms —
> filled programmatically. Because *we* wrote the values in, we get the box, the label and the link as
> exact ground truth for free. Then degrade with Augraphy so it looks scanned. The payoff: we can vary
> one factor at a time — same template, different noise.
>
> **[CUT IF LONG]** Open risk, stated honestly — whether synthetic diversity transfers to *real*
> unseen templates isn't cleanly measured anywhere we found. We treat it as a hypothesis.

### Slide 17 · Metrics and Failure Taxonomy

> Metrics: strict entity F1, exact span and type, implemented twice independently — they agree to
> zero.
>
> The interesting part is the error breakdown on unseen templates. We expected a spread across four
> buckets. We got OCR at zero-point-two percent, layout association at one percent, and
> **ninety-seven percent field-type and schema errors**.
>
> That's the most informative thing we have. The model finds the right text in the right place, then
> assigns the wrong field type. It isn't failing to *read* the form. It's failing to know *what the
> form is asking for* — semantic, not geometric. That points our ablations away from positional
> encoding.

### Slide 18 · Experiment 1 — Pipeline Validation

> First experiment: LiLT on standard FUNSD, three seeds. Its only job is to prove our training and
> evaluation loop is correct before we trust any cross-template number.
>
> We got **seventy-nine-point-three**. The published number is eighty-eight-point-four.
>
> We'd rather flag that than have it found. We pre-registered a gate saying this had to land near
> eighty-eight, or our box normalisation or label alignment was wrong. It didn't.
>
> We have ruled out one explanation: we checked whether we were scoring more strictly than the
> papers, and the lenient score is actually *lower*. Remaining suspects are training length and the
> HEADER class, which collapses to point-five F1. That's our next action.

*Delivery: don't apologise, don't rush. A TA will spot the 88.41 anyway — owning it beats hiding it.*

### Slide 19 · Experiment 2 — Seen vs Unseen

> The headline. VRDU Registration Forms, leave-one-template-out, three folds by three seeds, matched
> training size at two hundred documents, leakage check passing on every run.
>
> Seen templates: **eighty-eight F1**. Unseen: **sixty-seven**. A **twenty-one point** gap.
>
> Back to the previous slide. Both arms run identical code, data handling and metric. The only thing
> that differs is whether the test template was in training. So even with our absolute level low, the
> *gap* is a valid within-study comparison.
>
> What we should *not* do is set our twenty-one against FormNet's thirteen and conclude LiLT is worse
> — a depressed baseline inflates a gap. That waits on the previous slide's fix.
>
> **[CUT IF LONG]** The short form degrades steepest, eighty-eight to fifty-eight. Degradation isn't
> uniform across templates, and that's a lead.

> **Handoff:** Arjoe takes us through what we're claiming and what's next.

---

## BLOCK 4 — AB

### Slide 20 · The Research Gap

> Here's the gap we're claiming, and it came out of the reading rather than being chosen first.
>
> Entity extraction *has* been measured under template shift — VRDU, DocILE, Do-GOOD. Key-value
> **linking** has not. Every strong result in this table comes from a split where train and test share
> templates. The two template-disjoint entries are in the wrong setting: one is webpages, the other
> does field typing.
>
> So we measure tagging **and** linking under one template-disjoint protocol. We say "to our
> knowledge" deliberately.

### Slide 21 · Roadmap

> Where we are. The loaders, the leakage assertion, the dual metric, the gap measurement and the
> failure analysis are done.
>
> Next: I take the linking head on RFUND — the subtask that actually tests the claim we just made —
> plus scaling to VRDU Ad-buy for four training and two held-out templates, and the synthetic
> pipeline. Param takes the ablations and the demo interface.

### Slide 22 · Models, Interface and Risks

> Four models, licences declared — LayoutLMv3 is non-commercial, fine for coursework, but we'd rather
> state it. The encoders pretrain on IIT-CDIP, about eleven million scanned pages.
>
> The interface will be a Gradio app: upload a form, see predicted key-value pairs highlighted.
>
> Biggest risk is the top one — VRDU Registration has only three templates. That's exactly why Ad-buy
> and the synthetic families are on the roadmap.

### Slides 23–25 · References

> References are on the last three slides in the required format. A few are still flagged unverified
> in our repository, pending a primary-source check before the final report.
>
> That's us — thank you.

*Delivery: do not scroll slowly through three reference slides. One sentence, then stop.*

---

## Timing

Measured on the spoken lines only, excluding delivery notes and handoff cues.

Full script: **1860 spoken words**. With all four **[CUT IF LONG]** paragraphs dropped: **1771**.

| Delivery rate | Full script | After cuts |
|---|---|---|
| 130 wpm (slow, deliberate) | 14:18 | 13:37 |
| 150 wpm (normal) | 12:24 | 11:48 |
| 160 wpm (brisk) | 11:38 | 11:04 |

Rehearse once with a stopwatch. If the rehearsal passes 13:30, drop the four **[CUT IF LONG]**
paragraphs and re-run.

## Pre-flight checklist

- [ ] Compiled PDF last page reads **25**
- [ ] Facecam visible for **both** presenters, inside their own blocks
- [ ] Timed rehearsal under **14:00**
- [ ] Audio levels checked on both mics
- [ ] Upload to `Unemployed and Unsupervised_IE643_CourseProject_Prep`, verify access from a second account
