# Cross-Template Key-Value Field Extraction from Scanned Forms

**IE 643 Course Project** — generalizing key-value field extraction to form templates with zero same-template training data.

Extract key-value fields from scanned forms and generalize to form templates never seen in training — no labelled *or* unlabelled data from held-out templates at any point. The project measures the seen-vs-unseen performance gap and identifies which structural assumptions break.

## Project framing

Entity-level extraction has been measured under template shift (VRDU, DocILE, Do-GOOD). Key-value **linking** has not — every strong linking number in the literature, including GeoLayoutLM's 80.35 → 89.45 RE jump, comes from a split where train and test share templates. This project measures both under one template-disjoint protocol.

## Setup

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -r requirements.txt
```

Training runs on Kaggle Notebooks (P100 16GB or 2×T4); local installs of `torch` are optional and only needed to run the pipeline on CPU.

### Data

FUNSD is not redistributable through this repo. Download and extract it so the paths below exist:

```
data/funsd/dataset/training_data/{annotations,images}/
data/funsd/dataset/testing_data/{annotations,images}/
```

from <https://guillaumejaume.github.io/FUNSD/>. VRDU (the primary benchmark, with its official unseen-template splits) comes from <https://github.com/google-research-datasets/vrdu>.

## Running

```bash
# Track A — pipeline sanity check on FUNSD's standard split
python src/train.py --config configs/funsd_lilt.yaml
```

Track A targets entity F1 ≈ 0.88. **Read this number as a code-path validation, not as evidence of generalization** — see the caveats below.

## Layout

| Path | Purpose |
|---|---|
| `src/data/funsd_loader.py` | Parses the raw FUNSD release, keeping the `linking` edges the HF mirror drops |
| `src/data/splits.py` | MTL/UTL split construction and the template-leakage assertion |
| `src/models/token_clf.py` | One interface over LiLT / LayoutLMv3 / BROS, with word→subword label alignment |
| `src/evaluate.py` | Entity F1 (seqeval + an independent reimplementation), linking F1, the gap |
| `src/train.py` | Config-driven fine-tuning |
| `configs/` | One YAML per experiment |
| `slides/` | Beamer source for the prep presentation |

## Benchmark caveats that shape the design

These are not footnotes — they determine which numbers mean anything:

- **SROIE is unusable for this question.** Its test set has **75% template duplication** with train (FUNSD: 16%). Published SROIE numbers do not measure cross-template performance. *Laatiri et al. Information Redundancy and Biases in Public Document Information Extraction Benchmarks, ICDAR, 2023. <https://arxiv.org/abs/2304.14936>*
- **FUNSD's annotation leaks entity boundaries.** Block-level annotation gives every token in an entity identical coordinates, so models learn block boundaries as a proxy for entity boundaries rather than semantics. *Zhang et al. Unveiling the Deficiencies of Pre-trained Text-and-Layout Models, 2024. <https://arxiv.org/abs/2402.02379>*
- **FUNSD's linking ground truth is noisy.** PEneo re-annotated FUNSD/XFUND into RFUND precisely because the originals were inadequate for linking evaluation. Use RFUND for any reported linking number. *Lin et al. PEneo, ACM MM, 2024. <https://arxiv.org/abs/2401.03472>*
- **"Unseen template" is not mainly a geometry problem.** For LayoutLMv3 on FUNSD, pure layout shift costs 5.34 F1; real-world combined shift costs 32.41. Designing solely around positional encoding targets the smallest component. *He et al. Do-GOOD, SIGIR, 2023. <https://arxiv.org/abs/2306.02623>*
- **Relative 2D position encoding is not a generalization guarantee.** On MIDV2020's unseen-template country split: KNN-Former 87.88, LayoutLM-base 47.65, **BROS 23.31** — BROS degrades worst despite its relative-encoding design. *Dong et al. KNN-Former, 2024. <https://arxiv.org/abs/2405.06701>*

## Calibration targets

VRDU reports a 13–17 point MTL→UTL gap on Registration Forms; FormNet at 200 training documents scores MTL 90.51 vs UTL 77.29, a **13.22-point gap**. Our measured gap should be read against that. *Wang, Zhou, Wei, Lee, Tata. VRDU: A Benchmark for Visually-rich Document Understanding, KDD, 2023. <https://arxiv.org/abs/2211.15421>*

## Licensing

Model licenses differ and are stated in `src/models/token_clf.py`: LiLT is MIT, BROS is Apache-2.0, **LayoutLMv3 is CC-BY-NC-SA-4.0 (non-commercial)** — acceptable for coursework, but it must be declared in the report.
