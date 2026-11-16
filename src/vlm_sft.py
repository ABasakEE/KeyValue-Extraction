"""Study 29: page prompts, gold targets and answer parsing for the LoRA-tuned grounded VLM on VRDU registration forms.

One call per page.  The page's words are numbered from 0 and grouped into visual lines (`[first-last] (x,y) words`); the
answer is one line per field, `field | a-b, c-d` (inclusive page-local word ranges) or `field | NONE`.  Page answers map back to
document word indices (`base` = index of the page's first word) and are voted into one span per field per document.
Pure functions, no torch, so the whole data path is testable on CPU.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
MAX_PAGES = 4
SUBCORPUS = "registration-form"


def fields_and_desc() -> tuple[list[str], dict[str, str]]:
    """Page-level schema fields (line items excluded, as in studies 6-25) and their field-name-only descriptions."""
    meta = json.loads((ROOT / "data" / "vrdu" / SUBCORPUS / "main" / "meta.json").read_text(encoding="utf-8"))
    pat = meta.get("entity_appearance_pattern", {})
    fields = [n for n in meta["entity_name_to_match_func"] if isinstance(n, str) and pat.get(n) != "line_item"]
    desc = json.loads((ROOT / "results" / "zero_shot_kv_v2" / "schema_desc.json").read_text(encoding="utf-8"))[SUBCORPUS]
    return sorted(fields), desc


# ---------------------------------------------------------------- pages and prompts
def page_units(ex, max_pages: int = MAX_PAGES, gap: float | None = None) -> list[SimpleNamespace]:
    """The words of each page (contiguous in ``ex.words``): base offset, words, boxes; with ``gap`` also the segments (study 30)."""
    units = []
    for p in range(min(ex.n_pages, max_pages)):
        idx = [i for i, q in enumerate(ex.page_ids) if q == p]
        if idx:
            u = SimpleNamespace(page=p, base=idx[0], words=[ex.words[i] for i in idx], boxes=[ex.boxes[i] for i in idx])
            if gap is not None:
                u.segs = segments_of(u.boxes, gap)
            units.append(u)
    return units


def lines_of(boxes: list[list[int]]) -> list[tuple[int, int]]:
    """Runs [a, b) of consecutive words on one visual line (a new line when y jumps or the position moves back left)."""
    if not boxes:
        return []
    tol = 0.6 * median(max(1, b[3] - b[1]) for b in boxes)
    out, start, cy = [], 0, (boxes[0][1] + boxes[0][3]) / 2
    for i in range(1, len(boxes)):
        y = (boxes[i][1] + boxes[i][3]) / 2
        if abs(y - cy) > tol or boxes[i][0] < boxes[i - 1][0] - 5:
            out.append((start, i))
            start, cy = i, y
    out.append((start, len(boxes)))
    return out


def render_page(unit) -> str:
    return "\n".join(f"[{a}-{b - 1}] ({unit.boxes[a][0]},{unit.boxes[a][1]}) " + " ".join(unit.words[a:b]) for a, b in lines_of(unit.boxes))


def build_prompt(unit, fields: list[str], desc: dict[str, str]) -> str:
    descs = "\n".join(f"- {f}: {desc[f]}" for f in fields if desc.get(f))
    return ("This is a page of a scanned form. Extract the values of these fields.\n"
            f"Field descriptions:\n{descs}\n"
            "The words of the page are listed below, numbered, grouped into lines as '[first-last] (x,y) words', "
            "with x,y the top-left corner on a 0-1000 scale. A value is a run of consecutive word numbers, written first-last.\n"
            "Answer with one line per field and nothing else: '<field> | <first>-<last>' (several runs separated by ', '), "
            "or '<field> | NONE' if the field does not occur on this page.\n"
            "The words of this page:\n" + render_page(unit) + "\nAnswer:\n")


# ---------------------------------------------------------------- gold
def gold_spans(ex, id2label: dict[int, str]) -> list[list]:
    """[field, start, end_exclusive, text] from the BIO tags, in document order (strict BIO, the convention of the BROS dumps)."""
    out, cur = [], None
    for i, t in enumerate(ex.ner_tags):
        lab = id2label[t]
        if lab.startswith("I-") and cur is not None and cur[0] == lab[2:]:
            cur[2] = i + 1
            continue
        if cur:
            out.append(cur)
        # strict BIO (seqeval): an I- tag that does not continue a span of its own type starts nothing
        cur = [lab[2:], i, i + 1] if lab.startswith("B-") else None
    if cur:
        out.append(cur)
    return [[f, a, b, " ".join(ex.words[a:b])] for f, a, b in out]


def page_target(spans: list[list], unit, fields: list[str]) -> str:
    """The answer text for one page: gold spans that lie inside the page, as inclusive page-local ranges."""
    n, runs = len(unit.words), defaultdict(list)
    for f, a, b, _ in spans:
        if unit.base <= a and b <= unit.base + n:
            runs[f].append(f"{a - unit.base}-{b - 1 - unit.base}")
    return "\n".join(f"{f} | " + (", ".join(runs[f]) if runs[f] else "NONE") for f in fields)


# ---------------------------------------------------------------- parsing and voting
_ROW = re.compile(r"^\W*([A-Za-z_]+)\W*\|\s*(.+?)\s*$")
_RUN = re.compile(r"^(\d+)\s*(?:-\s*(\d+))?$")


def parse_page(text: str, n_words: int, fields: list[str]) -> tuple[dict[str, list[tuple[int, int]]], dict]:
    """field -> [(first, last_inclusive)] on the page.  Never raises; malformed rows and out-of-range runs are counted."""
    got: dict[str, list[tuple[int, int]]] = {}
    st = Counter(rows=0, bad_rows=0, bad_runs=0)
    for line in text.splitlines():
        if not line.strip():
            continue
        m = _ROW.match(line)
        if not m or m.group(1) not in fields:
            st["bad_rows"] += 1
            continue
        st["rows"] += 1
        runs = []
        for part in m.group(2).split(","):
            part = part.strip()
            if part.upper() == "NONE":
                continue
            r = _RUN.match(part)
            if not r:
                st["bad_runs"] += 1
                continue
            a = int(r.group(1))
            b = int(r.group(2)) if r.group(2) else a
            if a > b or b >= n_words:
                st["bad_runs"] += 1
                continue
            runs.append((a, b))
        if runs:
            got.setdefault(m.group(1), []).extend(runs)
    return got, dict(st)


def decode_pages(units, answers: dict[int, str], fields: list[str], words: list[str]) -> tuple[list[list], Counter]:
    """One span per field: majority over pages (ties to the earliest page); spans that overlap an earlier pick are dropped.

    Runs of one field on one page vote separately, so a field with several runs on a page is reduced to its first run, like
    study 25 (VRDU annotates one occurrence per entity).
    """
    stats: Counter = Counter()
    votes: dict[str, Counter] = defaultdict(Counter)
    first: dict[tuple, int] = {}
    for u in units:
        if u.page not in answers:
            stats["pages_missing"] += 1
            continue
        got, st = parse_page(answers[u.page], len(u.words), fields)
        stats.update(st)
        stats["pages"] += 1
        if not got and not any(l.strip() for l in answers[u.page].splitlines()):
            stats["pages_empty"] += 1
        for f, runs in got.items():
            a, b = runs[0]
            span = (u.base + a, u.base + b + 1)
            votes[f][span] += 1
            first.setdefault((f, span), u.page)
    chosen = [(c[s], f, s) for f, c in votes.items() for s in [max(c, key=lambda s: (c[s], -first[(f, s)], -s[0]))]]
    used: set[int] = set()
    spans = []
    for n, f, (a, b) in sorted(chosen, key=lambda x: (-x[0], fields.index(x[1]))):
        if used.intersection(range(a, b)):
            stats["dropped_overlap"] += 1
            continue
        used.update(range(a, b))
        spans.append([f, a, b, " ".join(words[a:b])])
    return spans, stats


# ---------------------------------------------------------------- study 30, format S: OCR-segment numbers
GAP = 1.0  # chosen on the training documents only (STUDIES.md, study 30)


def segments_of(boxes: list[list[int]], gap: float = GAP) -> list[tuple[int, int]]:
    """Page-local word ranges [a, b): the visual lines, cut where the gap between two consecutive words exceeds gap x the median word height."""
    if not boxes:
        return []
    med = median(max(1, b[3] - b[1]) for b in boxes)
    out = []
    for s, e in lines_of(boxes):
        cut = s
        for i in range(s + 1, e):
            if boxes[i][0] - boxes[i - 1][2] > gap * med:
                out.append((cut, i))
                cut = i
        out.append((cut, e))
    return out


def build_prompt_s(unit, fields: list[str], desc: dict[str, str]) -> str:
    descs = "\n".join(f"- {f}: {desc[f]}" for f in fields if desc.get(f))
    pieces = "\n".join(f"S{i} ({unit.boxes[a][0]},{unit.boxes[a][1]}) " + " ".join(unit.words[a:b]) for i, (a, b) in enumerate(unit.segs))
    return ("This is a page of a scanned form. Extract the values of these fields.\n"
            f"Field descriptions:\n{descs}\n"
            "The page is cut into numbered pieces, listed below as 'S<number> (x,y) text', with x,y the top-left corner on a 0-1000 scale. "
            "A value is a run of consecutive pieces, written S<first>-S<last> (a single piece is written S<number>).\n"
            "Answer with one line per field and nothing else: '<field> | S<first>-S<last>' (several runs separated by ', '), "
            "or '<field> | NONE' if the field does not occur on this page.\n"
            "The pieces of this page:\n" + pieces + "\nAnswer:\n")


def seg_target(spans: list[list], unit, fields: list[str]) -> str:
    """Gold spans on the page as runs of segments: the exact run if the span coincides with segment boundaries, else the smallest covering run."""
    n, runs = len(unit.words), defaultdict(list)
    for f, a, b, _ in spans:
        if unit.base <= a and b <= unit.base + n:
            wa, wb = a - unit.base, b - 1 - unit.base
            i = next(k for k, (s, e) in enumerate(unit.segs) if e > wa)
            j = next(k for k, (s, e) in enumerate(unit.segs) if e > wb)
            runs[f].append(f"S{i}-S{j}" if i != j else f"S{i}")
    return "\n".join(f"{f} | " + (", ".join(runs[f]) if runs[f] else "NONE") for f in fields)


_SRUN = re.compile(r"^S?(\d+)\s*(?:-\s*S?(\d+))?$", re.I)


def s_to_words(raw: str, unit, fields: list[str]) -> tuple[str, dict]:
    """A format-S answer as the equivalent word-run answer (page-local, inclusive), so decode_pages and the scorer apply unchanged."""
    st = Counter(rows=0, bad_rows=0, bad_runs=0)
    lines = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        m = _ROW.match(line)
        if not m or m.group(1) not in fields:
            st["bad_rows"] += 1
            continue
        st["rows"] += 1
        runs = []
        for part in m.group(2).split(","):
            part = part.strip()
            if part.upper() == "NONE":
                continue
            r = _SRUN.match(part)
            if not r:
                st["bad_runs"] += 1
                continue
            i = int(r.group(1))
            j = int(r.group(2)) if r.group(2) else i
            if i > j or j >= len(unit.segs):
                st["bad_runs"] += 1
                continue
            runs.append(f"{unit.segs[i][0]}-{unit.segs[j][1] - 1}")
        lines.append(f"{m.group(1)} | " + (", ".join(runs) if runs else "NONE"))
    return "\n".join(lines), dict(st)
