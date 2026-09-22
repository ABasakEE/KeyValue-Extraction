# Video script: IE 643 prep presentation

Team: Unemployed and Unsupervised. Param Mehta (23b2439), Arjoe Basak (23b1295).

Measured length is in the Timing table at the foot of this file. The cap is 15:00 and instruction 15
penalises going over. Both of us speak, facecam on throughout. Don't read the slides out loud. The
slides carry the numbers; you carry the argument.

Seven paragraphs are marked [CUT IF LONG]. Each one drops whole without losing a claim. Drop them
all if your timed rehearsal comes in over 13:30, or from the start if either of you speaks slowly.

Param opens and takes the empirical half. Arjoe takes the literature half and closes.

| Block | Slides | Speaker |
|---|---|---|
| Problem and contract | 1 to 4 | PM |
| Background and literature | 5 to 12 | AB |
| Data, protocol, experiments | 13 to 19 | PM |
| Gap, roadmap, close | 20 to 25 | AB |

Recording: OBS at 1080p, slides as a window capture with the facecam bottom right. Record in four
takes matching the four blocks, so a fluffed line costs one block instead of the whole video.

## Block 1: PM

### Slide 1, Title

> Hi, I'm Param Mehta. This is Arjoe Basak. We're team Unemployed and Unsupervised, and this is our
> prep presentation for IE 643.
>
> Our topic is cross-template key-value extraction from scanned forms, and the subtitle is where the
> difficulty is. We generalise to form templates we have zero training data for. Not a handful of
> examples, zero.

### Slide 2, Outline

> Quick map. Arjoe covers the background reading and where the literature disagrees with itself. I
> come back for our datasets, our split protocol, and the two experiments we've run. Arjoe closes
> with the gap we're claiming and the roadmap.

*Delivery: gesture at the two columns. Don't read the eight headings out.*

### Slide 3, The task

> Here's why this is hard. Train an extractor on one layout, test it on that same layout, and it
> scores well while learning almost nothing you can use. It's learned that the invoice number sits
> top right, and that the value is whatever box is to the right of the key. Those are facts about one
> template, not about forms.
>
> Change the template and three assumptions break together: absolute position, reading order, and
> that spatial convention. Everything going at once is what makes this different from ordinary
> fine-tuning.

### Slide 4, The zero-data contract

> "Zero same-template data" is stricter than it sounds. No labelled examples of the held-out
> template, obviously. But also no unlabelled ones, which rules out most of the domain adaptation
> literature. And the one that catches people: no validation use either. Pick your checkpoint by
> scoring on the held-out template and you've leaked, even though no gradient flowed.
>
> So it is an assertion in the code, printed every run, tested with a negative control that
> poisons a split.

> Handoff: over to Arjoe for what we read.

## Block 2: AB

### Slide 5, The standard pipeline

> Thanks Param. The pipeline runs along the top of the slide. Scan, OCR into words and bounding
> boxes, a layout-aware encoder, per-token tags, then links between keys and values.
>
> Three modalities: what a token says, where it sits, what it looks like. Every architectural
> argument I'm about to describe is about combining those. And tagging and linking are separate
> problems. One is per-token classification, the other is relational.

### Slide 6, Family 1, layout-aware encoders

> Family one is the LayoutLM line.
>
> Version one bolts absolute 2D position embeddings onto BERT. The model gets told this token sits at
> x equals four hundred. Version two makes position relative: the attention score between two tokens
> picks up a bias from the offset between their boxes. In the diagram, shift the whole
> form and every absolute coordinate changes, while every relative offset stays put.
>
> [CUT IF LONG] The other axis is vision. CNN regions in v2, ViT patches at token granularity in v3,
> which gets to ninety point three on FUNSD.

### Slide 7, Family 2, decoupling and relative encoding

> Family two takes that further, two ways.
>
> LiLT is our primary model. It splits into a text tower and a layout tower joined by BiACM. The part that matters is in the diagram: the gradient running from layout back into text
> gets detached. The reasoning is that layout doesn't depend on language, so you don't want it
> tangled with one vocabulary. They back it up by swapping the text tower out and transferring to
> seven unseen languages.
>
> Our caveat: that's cross-lingual evidence. We're betting it also helps across templates. That's an
> extrapolation, and testing it is part of what we're contributing.
>
> [CUT IF LONG] BROS goes the opposite way, no image at all, and still beats LayoutLMv2, which has one.

### Slide 8, Family 3, graphs and parsing

> Family three starts from linking being relational, so it models the document as a graph rather
> than a sentence. Nodes are text regions, so tagging becomes node classification. Edges are
> candidate relations, so linking becomes edge classification. Nothing assumes a reading order, and
> reading order is what breaks on a new template.
>
> The number worth remembering: a plain long-sequence transformer gets sixty-six. Add Rich Attention
> and graph pooling and you get eighty-four and a half. Eighteen points from structural encoding
> alone. That's the strongest evidence we found that sequence order is the wrong prior for forms.

### Slide 9, Family 4, OCR-free and generative

> Family four throws the pipeline away. Donut goes from image straight to JSON. DocLLM keeps the
> boxes but splits attention four ways.
>
> The cost is on the right. DocLLM's own FUNSD number is fifty-one point eight, against eighty-three
> to ninety-three for the discriminative encoders, with far more parameters. Free generation gives
> you no structural guarantee that a span comes out well formed.

### Slide 10, Where the literature disagrees

> Two questions without settled answers.
>
> [CUT IF LONG] Does vision help? BROS says no, and beats an image-using model without one. LayoutLMv3 and
> FormNetV2 say yes, but only when vision is fused at token granularity. Nobody has run the
> controlled ablation, so both sides argue from confounded comparisons.
>
> Do generative models win? LMDX loses under five F1 from seen to unseen templates, where LayoutLMv2
> loses nineteen to twenty-seven. DocLLM points the other way. They may generalise better while being
> worse at exact span boundaries, but nobody has put both in one experiment.

### Slide 11, What is actually measured

> Three results that changed our plan.
>
> First, the gap is real. FormNet on VRDU gets ninety point five on seen templates, seventy-seven
> point three on unseen. Thirteen points, and it's thirteen to seventeen across models.
>
> Second, and this cost us a design decision. We'd justified BROS on the grounds that relative
> encoding ought to help on new templates. Then we found KNN-Former's unseen-template split, where
> BROS gets twenty-three, below plain LayoutLM at forty-seven. Our argument was just wrong. BROS
> stays, but for its linking head.
>
> Third, Do-GOOD breaks the shift down, and pure layout novelty accounts for five F1 of a thirty-two
> point drop. You can spend a semester tuning positional encodings and be working on five points of a
> thirty-two point problem.

*Delivery: slow down here. This slide is what shows we read critically instead of collecting citations.*

### Slide 12, Benchmark integrity

> Someone measured template duplication between train and test. SROIE is
> seventy-five percent duplicated, so those generalisation numbers are largely measuring
> memorisation.
>
> FUNSD is sixteen percent, but has its own flaw. Block-level annotation gives every token inside an
> entity the same coordinates, so models learn to treat a block boundary as an entity
> boundary.
>
> [CUT IF LONG] Its linking labels were noisy enough that another group re-annotated the whole thing.
>
> So FUNSD can validate our code. It can't validate our claim.

> Handoff: back to Param.

## Block 3: PM

### Slide 13, Dataset survey

> We went looking for a dataset and hit a wall.
>
> VRDU has an official unseen-template protocol and no linking annotation. The datasets with linking
> annotation have no template split. Nothing lets us ask our question directly. That's the shape of
> the field rather than a hole in our search, and it's why the question is worth asking.

### Slide 14, Datasets we will use

> Three sources. VRDU is primary, the only ready-made unseen-template protocol, and its published gap
> calibrates ours. FUNSD comes in through the corrected RFUND annotation, for the linking half only,
> never as generalisation evidence. And we build our own synthetic data.

### Slide 15, Our split protocol

> Here's slide four made concrete. Top row is seen: A, B and C in training and in
> test. Bottom row is unseen: train on B and C, test on A alone. One fold per held-out template.
>
> Matched training size matters. If the two conditions see different numbers of documents, you
> haven't measured a generalisation gap, you've measured a sample-efficiency curve. Our code errors
> out rather than report one.
>
> Then something we didn't expect. When we audited VRDU's own official unseen-template splits, the
> held-out template turns up in validation on all three folds. On one fold that's a hundred documents
> out of a hundred. So we implemented both: theirs for comparability, and a strict version.
> Everything after this uses the strict one.

*Delivery: this is our strongest evidence that we did real work. Land it, don't rush past it.*

### Slide 16, Data curation pipeline

> Our own data, left to right. Real blank fillable PDFs, about fifty-five thousand plus IRS forms,
> filled in programmatically. Because we write the values in, we get the box, the label and the link
> as exact ground truth for free. Then we degrade it with Augraphy so it looks scanned. The payoff is
> we can vary one factor at a time: same template, different noise.
>
> [CUT IF LONG] One open risk. Whether synthetic diversity transfers to real unseen templates isn't
> cleanly measured anywhere we looked. We're treating it as a hypothesis.

### Slide 17, Metrics and failure taxonomy

> Metrics: strict entity F1, exact span and exact type. We implemented it twice independently
> and the two agree to zero.
>
> The interesting part is the error breakdown on unseen templates. We expected the errors to spread
> across four buckets. We got OCR at zero point two percent, layout association at one percent, and
> ninety-seven percent field-type and schema errors.
>
> The model finds the right text in the right place and gives it the wrong field type. It can read the form. It can't tell what the form is asking for. The
> failure is semantic rather than geometric, which points our ablations away from positional
> encoding.

### Slide 18, Experiment 1, pipeline validation

> First experiment. LiLT on standard FUNSD, three seeds. Its only job is to show our training and
> evaluation loop is correct before we trust any cross-template number.
>
> We got seventy-nine point three. The published figure is eighty-eight point four.
>
> We'd rather flag that ourselves than have it found. We'd pre-registered a gate saying this had to
> land near eighty-eight, or our box normalisation or label alignment was wrong. It didn't.
>
> We've ruled out one explanation. We checked whether we were scoring more strictly than the papers
> do, and the lenient score is actually lower. What's left is training length and the HEADER class,
> which collapses to point five F1. That's what we fix next.

*Delivery: don't apologise and don't rush. A TA will spot the 88.41 anyway, so owning it beats hiding it.*

### Slide 19, Experiment 2, seen vs unseen

> The headline experiment. VRDU Registration Forms, leave-one-template-out, three folds by three
> seeds, matched training size at two hundred documents, leakage check passing every run.
>
> Seen templates give us eighty-eight F1. Unseen gives sixty-seven. A twenty-one point gap.
>
> Go back to the previous slide. Both arms run identical code, identical data handling, identical
> metric. The only thing that differs is whether the test template appeared in
> training. So even with our absolute level low, the gap holds up as a within-study comparison.
>
> What we shouldn't do is set our twenty-one against FormNet's thirteen and conclude LiLT is worse. A
> depressed baseline inflates a gap. That waits until we've fixed the previous slide.
>
> [CUT IF LONG] The short form degrades steepest, eighty-eight down to fifty-eight, so degradation
> isn't uniform across templates. Worth chasing.

> Handoff: Arjoe takes us through what we're claiming and what's next.

## Block 4: AB

### Slide 20, The research gap

> The gap we're claiming came out of the reading rather than being picked first.
>
> Entity extraction has been measured under template shift. VRDU, DocILE and Do-GOOD all do it.
> Key-value linking hasn't. Every strong result in this table comes from a split where train and test
> share templates. The two template-disjoint entries are in the wrong setting: one is webpages, the
> other does field typing.
>
> So we measure tagging and linking under a single template-disjoint protocol. We say "to our
> knowledge" deliberately.

### Slide 21, Roadmap

> The loaders, the leakage assertion, the dual metric, the gap measurement and the failure analysis
> are done.
>
> Next I take the linking head on RFUND, the subtask that actually tests the claim we just made, plus
> scaling to VRDU Ad-buy for four training and two held-out templates, and the synthetic pipeline.
> Param takes the ablations and the demo interface.

### Slide 22, Models, interface and risks

> Four models with licences declared. LayoutLMv3 is non-commercial, fine for coursework, but we'd
> rather state it. The encoders are pretrained on IIT-CDIP, roughly eleven million scanned pages.
>
> [CUT IF LONG] The interface will be a Gradio app. Upload a form, see the predicted key-value pairs
> highlighted on it.
>
> The biggest risk is the one at the top. VRDU Registration has only three templates, thin for a
> claim about generalisation. That's why Ad-buy and the synthetic families are on the roadmap.

### Slides 23 to 25, References

> The references are on the last three slides in the format the course asks for. A few are still
> flagged unverified in our repository, pending a primary-source check before the final report.
>
> That's us. Thank you.

*Delivery: don't scroll slowly through three reference slides. One sentence, then stop.*

## Timing

Measured on the spoken lines only, excluding delivery notes and handoff cues.

Full script: **1933 spoken words**. With all **[CUT IF LONG]** paragraphs dropped: **1761**.

| Delivery rate | Full script | After cuts |
|---|---|---|
| 130 wpm (slow, deliberate) | 14:52 | 13:33 |
| 150 wpm (normal) | 12:53 | 11:44 |
| 160 wpm (brisk) | 12:05 | 11:00 |

Rehearse once with a stopwatch. If you come in over 13:30, drop the [CUT IF LONG] paragraphs and run
it again.

## Pre-flight checklist

- [ ] Compiled PDF last page reads 25
- [ ] Facecam visible for both presenters, inside their own blocks
- [ ] Timed rehearsal under 14:00
- [ ] Audio levels checked on both mics
- [ ] Uploaded to `Unemployed and Unsupervised_IE643_CourseProject_Prep`, access verified from a second account
