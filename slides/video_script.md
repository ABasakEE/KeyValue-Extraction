# Video Script — IE 643 Prep Presentation

**Team: Unemployed and Unsupervised** · Param Mehta (23b2439), Arjoe Basak (23b1295)

Target runtime **13:25**, hard cap 15:00 (instruction 15 penalises overrun).
Both presenters speak. Facecam on throughout. **Do not read the slides** — the slides
carry the numbers, the speaker carries the argument.

**Speaker split.** PM opens and owns the empirical half; AB owns the literature half and closes.
Three handoffs, each spoken aloud so the cut is obvious to the viewer.

| Block | Slides | Speaker | Budget |
|---|---|---|---|
| Problem and contract | 1–4 | **PM** | 2:10 |
| Background and literature | 5–12 | **AB** | 5:05 |
| Data, protocol, experiments | 13–19 | **PM** | 4:25 |
| Gap, roadmap, close | 20–25 | **AB** | 1:45 |

Recording notes: OBS, 1080p, slides as window capture with facecam bottom-right.
Record in four takes matching the four blocks — a fluffed line costs one block, not the whole video.

---

## BLOCK 1 — PM

### Slide 1 · Title — 0:25

> Hi, I'm Param Mehta, and I'm here with Arjoe Basak. We're team Unemployed and Unsupervised, and
> this is our prep presentation for the IE 643 course project.
>
> Our allotted topic is cross-template key-value field extraction from scanned forms. The hard part
> is in the subtitle: we have to generalise to form templates for which we have **zero** training
> data. Not a few examples — none.

*Delivery: say the team name with a straight face and move on. Don't linger on the title.*

### Slide 2 · Outline — 0:20

> Quick map of the next thirteen minutes. Arjoe takes the background reading and what the
> literature actually agrees and disagrees on. I'll come back for our datasets, the split protocol
> we bind ourselves to, and the two experiments we've already run. Then Arjoe closes with the
> research gap we're claiming and the roadmap.

*Delivery: gesture at the two columns, don't read the eight headings.*

### Slide 3 · The Task — 0:40

> Here's the intuition for why this is hard.
>
> If you train a form extractor on one layout and test it on the same layout, it will score very
> well — and it will have learned almost nothing useful. It has learned that the invoice number
> lives in the top-right corner. It has learned that the value is whatever box sits to the right of
> the key. Those are facts about *that template*, not facts about forms.
>
> Change the template and all three of these priors break at the same time: the absolute position
> prior, the reading-order assumption, and the spatial key-value convention. That simultaneity is
> what makes this a genuinely different problem from ordinary fine-tuning.

### Slide 4 · The Zero-Data Contract — 0:45

> Before any modelling, we had to pin down what "zero same-template data" actually forbids, because
> it's stricter than it first sounds.
>
> No labelled examples of the held-out template — obviously. But also **no unlabelled examples**,
> which quietly rules out most of the domain-adaptation literature, because almost all of it assumes
> you have unlabelled target data sitting around.
>
> And the one people get wrong: **no validation use**. If you pick your best checkpoint by scoring
> on the held-out template, you have leaked — even though no gradient ever flowed from it. That one
> turns out to matter in practice, and I'll come back to it on the protocol slide.
>
> This isn't a promise in a document. It's an assertion in our code that prints on every run, and we
> tested it with a negative control that deliberately poisons a split, to confirm the check actually fires.

> **Handoff:** Over to Arjoe for what we read.

---

## BLOCK 2 — AB

### Slide 5 · The Standard Pipeline — 0:35

> Thanks Param. Let me set up the machinery everything else sits on.
>
> The classical pipeline runs left to right along the top: scan, OCR — which gives you words *and*
> bounding boxes — then a layout-aware encoder, then per-token tags, then links between keys and values.
>
> Two things to hold onto. First, three modalities: what a token says, where it sits, and what it
> looks like. Every architectural disagreement we're about to see is about how to fuse those three.
> Second, and this becomes our whole project: **tagging and linking are different problems**. Tagging
> is per-token classification. Linking is relational. The literature measures them under very
> different conditions.

### Slide 6 · Family 1 — Layout-Aware Encoders — 0:40

> Family one is the LayoutLM line, and it's a clean two-part trajectory.
>
> Version one bolts *absolute* two-D position embeddings onto BERT — the model is told "this token
> sits at x equals four hundred". Version two replaces that with *relative* encoding: the attention
> score between two tokens picks up a bias computed from the offset between their boxes. The diagram
> shows why that matters for us. Shift the whole form and every absolute coordinate changes, but
> every relative offset survives.
>
> The second axis is vision: CNN region features in v2, versus ViT patches at the same granularity as
> text tokens in v3. v3 gets ninety-point-three on FUNSD, the strongest published encoder number.
>
> One thing worth flagging: the ablation on the right shows word-patch alignment, which is v3's
> headline contribution, is worth zero-point-five-nine F1. Real, but small.

### Slide 7 · Family 2 — Decoupling and Relative Encoding — 0:40

> Family two pushes relative encoding in two different directions.
>
> **LiLT** — our primary model — splits into two towers, one for text and one for layout, coupled at
> every layer by a module called BiACM. The critical detail is in the diagram: during pretraining the
> gradient flowing from layout back into text is **detached**. The argument is that layout is
> language-agnostic, so you shouldn't let it entangle with one vocabulary. They demonstrate it by
> swapping in a different language's text tower and transferring to seven unseen languages.
>
> Here's our honest caveat, and we'd rather say it up front: that is evidence about *cross-lingual*
> transfer. We are betting the same decoupling helps with *cross-template* transfer. That's an
> extrapolation, and testing it is part of what we're contributing.
>
> **BROS** goes the other way — no image at all — and still gets eighty-three-point-oh-five on FUNSD,
> beating LayoutLMv2 *with* an image. Hold that thought; it comes back in two slides.

### Slide 8 · Family 3 — Graphs and Parsing — 0:40

> Family three says: if linking is relational, stop pretending the document is a sentence.
>
> Model it as a graph instead. Nodes are text regions, so tagging becomes node classification. Edges
> are candidate key-value relations, so linking becomes **edge** classification. No reading order is
> assumed anywhere — and reading order is exactly the assumption that breaks on a new template.
>
> The number I'd point to is the FormNet ablation. A plain long-sequence transformer gets
> sixty-five-point-nine. Add Rich Attention — a distance and direction term inside attention — and it
> jumps to eighty-two. Add the graph pooling as well, eighty-four-point-five. That's **eighteen and a
> half F1** from structural encoding alone. It's the clearest single piece of evidence we found that
> sequence order is the wrong prior for forms.
>
> Shared weakness: the graph is built *before* the network runs, so one bad OCR box corrupts
> everything downstream.

### Slide 9 · Family 4 — OCR-Free and Generative — 0:30

> Family four throws the pipeline out entirely. Donut goes straight from image to JSON — no OCR, no
> boxes. DocLLM keeps boxes but splits attention into four text-and-spatial matrices.
>
> The cost is on the right. DocLLM's own reported FUNSD number is **fifty-one-point-eight**, against
> eighty-three to ninety-three for the discriminative encoders, with far more parameters. Free-form
> generation gives you no structural guarantee of a well-formed span, the way a tagging head or a
> graph head does.

### Slide 10 · Where the Literature Disagrees — 0:35

> Two questions we expected to have settled answers to, and don't.
>
> Does vision help? BROS says no — it beats an image-using model without one, and argues serialisation
> was the real bottleneck all along. LayoutLMv3 and FormNetV2 say yes, but only when vision is fused
> at token granularity. Nobody has run the controlled ablation holding backbone size and pretraining
> corpus fixed, so both camps are arguing from confounded comparisons.
>
> Do generative models win? LMDX drops under five F1 going from seen to unseen templates, where
> LayoutLMv2 drops nineteen to twenty-seven. That's a big claim for generation. But DocLLM's
> fifty-one-point-eight says the opposite. Reading both charitably: generative models may generalise
> better while being worse at exact span boundaries. Different quantities — never measured in one
> experiment.

### Slide 11 · What Is Actually Measured — 0:50

> This is the most important slide in my half. Three measured results that changed our plan.
>
> **One** — the gap is real. On VRDU, FormNet scores ninety-point-five on seen templates and
> seventy-seven-point-three on unseen. Thirteen points, and it's thirteen to seventeen across models.
> That's our calibration target.
>
> **Two** — and this one cost us a design decision. We had originally justified using BROS on the
> grounds that relative position encoding should help on new templates. Then we found KNN-Former's
> unseen-template split, where BROS scores **twenty-three** — worse than plain LayoutLM at
> forty-seven. The architectural argument we had made was simply wrong. BROS stays in our plan, but
> now for its linking head, which is a capability argument, not a generalisation one.
>
> **Three** — Do-GOOD decomposes the shift, and pure layout novelty costs about five F1 out of a
> thirty-two-point real-world drop. So if you spend the whole project tuning positional encodings, you
> are optimising five points of a thirty-two point problem. That genuinely reframed what we're looking for.

*Delivery: slow down here. This is the slide that shows we read critically rather than collected citations.*

### Slide 12 · Benchmark Integrity — 0:35

> And a warning about the numbers themselves.
>
> Someone went and measured template duplication between train and test in the standard benchmarks.
> SROIE is **seventy-five percent** duplicated. So published SROIE generalisation numbers are, to a
> large extent, measuring memorisation.
>
> FUNSD is sixteen percent, which is better, but FUNSD has its own problem: block-level annotation
> gives every token inside an entity identical coordinates, so models learn "block boundary equals
> entity boundary" as a shortcut. And its linking ground truth was noisy enough that a separate group
> re-annotated the whole thing into RFUND.
>
> The consequence for us is the line at the bottom: FUNSD can validate our **code**. It cannot
> validate our **claim**. Param is about to show you exactly that distinction being used.

> **Handoff:** Back to Param.

---

## BLOCK 3 — PM

### Slide 13 · Dataset Survey — 0:30

> So we surveyed what's available, and ran straight into a wall.
>
> Look at the last two columns. The dataset with an official unseen-template protocol — VRDU — has
> **no linking annotation**. The datasets with linking annotation — FUNSD, XFUND, KVP10k — have **no
> template split**. There is no single dataset that lets you ask our question directly.
>
> That's not a gap in our search. That's the shape of the field, and it's a large part of why the
> question is worth asking at all.

### Slide 14 · Datasets We Will Use — 0:30

> So we use three things. VRDU as primary, because it's the only ready-made unseen-template protocol
> and its published gap calibrates ours. FUNSD through the corrected RFUND annotation, for the linking
> half only — never as generalisation evidence on its own, for the reasons Arjoe just gave.
>
> And third, we curate our own, which is the next slide but one.

### Slide 15 · Our Split Protocol — 0:40

> This is the contract from slide four, made concrete.
>
> Top row, seen templates: A, B and C in training, A, B and C in test. Bottom row, unseen: train on B
> and C only, test on A alone. One fold per held-out template.
>
> Two details that are easy to get wrong. **Matched training size** — if the seen and unseen regimes
> train on different numbers of documents, what you've measured isn't a generalisation gap, it's a
> sample-efficiency curve. Our code raises an error rather than reporting a gap in that case.
>
> And here's something we didn't expect. When we audited **VRDU's own official unseen-template
> splits**, the held-out template shows up in the *validation* set on all three folds — in one case a
> hundred out of a hundred validation documents. So we implemented both: their official protocol, for
> comparability with published numbers, and a strict protocol where validation is rebuilt from
> training templates only. Everything I'm about to show uses the strict one.

*Delivery: this is our strongest "we did real work" moment. Land it clearly, don't rush past it.*

### Slide 16 · Data Curation Pipeline — 0:30

> Our own data, left to right. Take real blank fillable PDFs — there's a public corpus of about
> fifty-five thousand, plus IRS forms — fill the form widgets programmatically with fake but plausible
> values, and because *we* wrote the values in, we get the box, the label and the key-value link as
> exact ground truth for free. Then degrade it with Augraphy so it looks like it came off a scanner.
>
> The payoff is the third bullet: we can vary exactly one factor at a time. Same template, different
> degradation. That's the Do-GOOD decomposition, but under our control instead of someone else's.
>
> Open risk, stated honestly: whether synthetic template diversity actually transfers to *real* unseen
> templates isn't cleanly measured anywhere we found. We treat it as a hypothesis, not an assumption.

### Slide 17 · Metrics and Failure Taxonomy — 0:35

> Metrics: strict entity F1 — exact span, exact type, no partial credit. We implemented it twice,
> independently, and cross-checked; the two agree to zero.
>
> The interesting part is the error breakdown on the right, run on the unseen-template predictions. We
> expected a spread across four buckets. What we got was OCR errors at zero-point-two percent, layout
> association at one percent, and **ninety-seven percent field-type and schema errors**.
>
> Read that carefully, because it's the most informative thing we've found so far. On a new template
> the model is locating the right text in the right place, and then assigning it the wrong field type.
> It isn't failing to *read* the form. It's failing to know *what the form is asking for*. That's a
> semantic failure, not a geometric one — which lines up with Do-GOOD, and it points our ablations
> somewhere quite different from positional encoding.

### Slide 18 · Experiment 1 — Pipeline Validation — 0:45

> First experiment: LiLT on standard FUNSD, three seeds. Its only purpose is to prove our training and
> evaluation loop is correct before we trust any cross-template number.
>
> We got **seventy-nine-point-three**. The published number for this model is eighty-eight-point-four.
>
> We want to be straight about this, because it's a nine-point shortfall and we'd rather flag it than
> have it found. We had pre-registered a gate in our plan saying this needed to land near eighty-eight,
> or something was wrong with our box normalisation or our label alignment. It didn't land there.
>
> We have ruled out one explanation: we checked whether we were simply scoring more strictly than the
> papers do, and the lenient score is actually *lower*, so that isn't it. Our remaining suspects are
> training length and the HEADER class, which collapses to point-five F1 on very few examples. That's
> the first thing we fix.
>
> What this does **not** invalidate is the next slide, and I'll say why.

*Delivery: don't apologise, don't rush. Owning a missed gate is worth more than hiding it, and a TA will spot the 88.41 anyway.*

### Slide 19 · Experiment 2 — Seen vs Unseen — 0:55

> This is the headline experiment. VRDU Registration Forms, leave-one-template-out, three folds by
> three seeds, matched training size at two hundred documents, leakage check passing on every run.
>
> Seen templates: **eighty-eight F1**. Unseen: **sixty-seven**. A **twenty-one point** gap — a
> twenty-four percent relative drop.
>
> Now, back to the previous slide. Both arms here run through identical code, identical data handling,
> identical metric. The only thing that differs is whether the test template was in training. So even
> with our absolute level depressed, the *gap* is a valid within-study comparison.
>
> What we should *not* do is put our twenty-one points next to FormNet's thirteen and conclude LiLT is
> worse — our baseline is low, so a larger gap is partly expected. We'll make that comparison once
> experiment one passes its gate.
>
> One more thing worth following: the per-template chart shows the short form degrading steepest,
> eighty-eight down to fifty-eight. Degradation isn't uniform across templates, and that's a lead.

> **Handoff:** Arjoe will take us through what we're claiming and what's next.

---

## BLOCK 4 — AB

### Slide 20 · The Research Gap — 0:40

> So here's the gap we're claiming, and it came directly out of the reading rather than being chosen
> first.
>
> Entity extraction **has** been measured under template shift — VRDU, DocILE and Do-GOOD all do it.
> Key-value **linking** has not. Every strong linking result in this table — GeoLayoutLM at
> eighty-nine-point-four, KVPFormer, PEneo — comes from a split where train and test share templates.
>
> The two entries that *are* template-disjoint are in the wrong setting: one is on webpages, the other
> does field typing rather than linking.
>
> So our contribution is to measure tagging **and** linking under one template-disjoint protocol. We
> phrase that as "to our knowledge" deliberately — we haven't exhaustively verified every benchmark's
> task structure, and we'd rather be precise than sweeping.

### Slide 21 · Roadmap — 0:30

> Where we are. The first three subtasks and the failure analysis are done — the loaders, the leakage
> assertion, the dual metric implementation, and the gap measurement Param just showed you.
>
> Next: I take the linking head on RFUND, which is the subtask that actually tests the claim we just
> made, plus scaling to VRDU Ad-buy to get four training and two held-out templates, and the synthetic
> pipeline. Param takes the ablations and the demo interface. The report is joint.

### Slide 22 · Models, Interface and Risks — 0:25

> Four models, licences declared — note LayoutLMv3 is non-commercial, which is perfectly fine for
> coursework but we'd rather state it than not. All the encoders pretrain on IIT-CDIP, about eleven
> million scanned pages.
>
> The interface will be a Gradio app: upload a form, see the predicted key-value pairs highlighted on
> the page.
>
> Biggest risk is the top one — VRDU Registration only has three templates, which is thin for a claim
> about generalisation. That's exactly why Ad-buy and the synthetic families are on the roadmap.

### Slides 23–25 · References — 0:10

> Our references are on the last three slides in the format the course requires. A few are still
> flagged unverified in our repository, pending a primary-source check before the final report.
>
> That's us — thank you.

*Delivery: do not scroll slowly through three reference slides. One sentence, then stop.*

---

## Pre-flight checklist

- [ ] Compiled PDF last page reads **25**
- [ ] Facecam visible for **both** presenters, inside their own blocks
- [ ] Full rehearsal timed under **14:00** before the real take
- [ ] Audio levels checked on both mics
- [ ] Upload to `Unemployed and Unsupervised_IE643_CourseProject_Prep`, verify access from a second account
