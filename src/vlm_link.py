"""Zero-shot VLM key-value linking on XFUND (Study 27): prompts, output parsers and gold renderings.

Three arms, all scored with the end-to-end rules of ``e2e_link`` (exact word ranges, typed entities, directed links):

  V2  page image + the form's words (``ex.words`` order, grouped into visual lines with their indices);
      one item per line: a linked pair ``Q 4-6 > A 7-9`` or a lone entity ``H 0-3`` (inclusive word indices).
      Every line stands alone, so an answer cut off by the token limit keeps all its complete lines.
  V3  page image + every annotated entity numbered ``e<n>``; the VLM lists links only, as ``e<n>>e<m>``.
  V1  page image only; the VLM lists entities as ``E<n> <H|Q|A> | <text>`` and links; texts are mapped onto
      ``ex.words`` by exact (normalised) match.

Parsers never raise: malformed lines are skipped and counted.  A link written value->key (by the predicted labels)
is flipped to key->value, the same rule for every arm.  Pure functions, no torch.
"""
from __future__ import annotations

import random
import re
import unicodedata
from statistics import median

try:
    from e2e_link import Entity, Link, gold_structure
except ImportError:
    from src.e2e_link import Entity, Link, gold_structure

LETTER = {"H": "header", "Q": "question", "A": "answer"}
LETTER_OF = {v: k for k, v in LETTER.items()}
# (source label, destination label) pairs that are written backwards: flip them to key -> value
_BACKWARD = {("answer", "question"), ("question", "header"), ("answer", "header")}

_MENTION = re.compile(r"^\W*([HQA])[a-z]*\s*[:.]?\s*(\d+)(?:\s*-\s*(\d+))?\W*$", re.I)
_ENT_V1 = re.compile(r"^\W*E\s*(\d+)\s*[:.]?\s+([HQA])\w*\s*[|:]\s*(.+?)\s*$", re.I)
_LINK = re.compile(r"\b([Ee])\s*(\d+)\s*(?:->|=>|>|→)\s*[Ee]\s*(\d+)\b")


# ---------------------------------------------------------------- form order and layout
def form_order(examples, seed: int = 0) -> list:
    """Shuffle each language with a fixed seed, then take the languages in turn, so any prefix stays balanced."""
    by_lang: dict[str, list] = {}
    for ex in sorted(examples, key=lambda e: e.guid):
        by_lang.setdefault(ex.guid.split("_")[0], []).append(ex)
    rng = random.Random(seed)
    for lang in sorted(by_lang):
        rng.shuffle(by_lang[lang])
    out, queues = [], [by_lang[k] for k in sorted(by_lang)]
    while any(queues):
        for q in queues:
            if q:
                out.append(q.pop(0))
    return out


def ocr_lines(ex) -> list[tuple[int, int]]:
    """Runs [a, b) of consecutive word indices that sit on one visual line (indices stay contiguous)."""
    if not ex.words:
        return []
    heights = [max(1, b[3] - b[1]) for b in ex.boxes]
    tol = 0.6 * median(heights)
    lines, start = [], 0
    cy = (ex.boxes[0][1] + ex.boxes[0][3]) / 2
    for i in range(1, len(ex.words)):
        b, prev = ex.boxes[i], ex.boxes[i - 1]
        y = (b[1] + b[3]) / 2
        if abs(y - cy) > tol or b[0] < prev[0] - 5:  # new visual line, or the reading position jumps back left
            lines.append((start, i))
            start, cy = i, y
    lines.append((start, len(ex.words)))
    return lines


def render_lines(ex) -> str:
    rows = []
    for a, b in ocr_lines(ex):
        x, y = ex.boxes[a][0], ex.boxes[a][1]
        rows.append(f"[{a}-{b - 1}] ({x},{y}) " + " ".join(ex.words[a:b]))
    return "\n".join(rows)


def entity_list(ex) -> list[int]:
    """Annotated entity ids with at least one word, in id order: e1 is the first of these."""
    return [i for i in sorted(ex.entities) if ex.entities[i].get("token_indices")]


def _entity_text(ex, i: int) -> str:
    idx = ex.entities[i]["token_indices"]
    return " ".join(ex.words[w] for w in idx) or ex.entities[i].get("text", "")


# ---------------------------------------------------------------- prompts
_TASK = (
    "This is a scanned form. Find its entities and how they are linked.\n"
    "Entity types:\n"
    "  H = header: a title or section heading.\n"
    "  Q = question: a key or field label, e.g. 'Name:' or 'Date of birth'.\n"
    "  A = answer: the value filled in for a key.\n"
    "Links go from key to value: a question to its answer (Q>A), and a header to the questions under it (H>Q).\n"
    "Text that is none of these (page numbers, footnotes, instructions) is not listed.\n")

PROMPT_V1 = (
    _TASK +
    "Read the page and answer in exactly this format, one item per line, entities first, then links, nothing else:\n"
    "E<number> <H|Q|A> | <exact text of the entity as printed>\n"
    "E<number>>E<number>\n"
    "Example answer for a small form:\n"
    "E1 H | DATOS PERSONALES\n"
    "E2 Q | Nombre:\n"
    "E3 A | Ana García López\n"
    "E4 Q | Fecha\n"
    "E5 A | 12/03/2019\n"
    "E1>E2\n"
    "E1>E4\n"
    "E2>E3\n"
    "E4>E5\n"
    "Now the answer for this form:\n")


def prompt_v1(ex) -> str:
    return PROMPT_V1


def prompt_v2(ex) -> str:
    return (
        _TASK +
        "The form's words are listed below, numbered, grouped into lines as '[first-last] (x,y) words', with x,y the "
        "top-left corner on a 0-1000 scale. An entity is a run of consecutive word numbers, written as its type letter "
        "and its first-last word numbers.\n"
        "Answer with one item per line and nothing else:\n"
        "  a link as '<key entity> > <value entity>', e.g. 'Q 2-2 > A 3-5' or 'H 0-1 > Q 2-2';\n"
        "  an entity with no link on a line of its own, e.g. 'H 0-1'.\n"
        "Example: for the words\n"
        "[0-1] (80,40) DATOS PERSONALES\n"
        "[2-2] (80,90) Nombre:\n"
        "[3-5] (300,90) Ana García López\n"
        "[6-7] (80,140) Fecha 12/03/2019\n"
        "[8-9] (80,190) Firma del solicitante\n"
        "the answer is\n"
        "H 0-1 > Q 2-2\n"
        "H 0-1 > Q 6-6\n"
        "Q 2-2 > A 3-5\n"
        "Q 6-6 > A 7-7\n"
        "Q 8-9\n"
        "The words of this form:\n" + render_lines(ex) + "\nAnswer:\n")


def prompt_v3(ex) -> str:
    rows = []
    for n, i in enumerate(entity_list(ex), 1):
        b = ex.boxes[ex.entities[i]["token_indices"][0]]
        rows.append(f"e{n} {ex.entities[i]['label'].lower()} ({b[0]},{b[1]}) {_entity_text(ex, i)}")
    return (
        "This is a scanned form. Its entities are listed below as 'e<number> <type> (x,y) text', with x,y the top-left "
        "corner on a 0-1000 scale. Types: header (title or section heading), question (a key or field label), answer "
        "(the value filled in for a key), other.\n"
        "List every link from a key to its value: a question to its answer, and a header to the questions under it. "
        "Answer with one link per line as e<number>>e<number>, key first, and nothing else.\n"
        "Example answer:\n"
        "e1>e2\n"
        "e2>e3\n"
        "e4>e5\n"
        "The entities of this form:\n" + "\n".join(rows) + "\nAnswer:\n")


PROMPTS = {"V1": prompt_v1, "V2": prompt_v2, "V3": prompt_v3}


# ---------------------------------------------------------------- parsing
def _orient(src, dst, lab_s: str, lab_d: str):
    return (dst, src) if (lab_s, lab_d) in _BACKWARD else (src, dst)


def _links(text: str, prefix: str, ents: dict, labels: dict, stats: dict) -> set[Link]:
    out: set[Link] = set()
    for m in _LINK.finditer(text):
        if m.group(1).upper() != prefix.upper():
            continue
        s, d = int(m.group(2)), int(m.group(3))
        if s not in ents or d not in ents:
            stats["dangling_links"] += 1
            continue
        if ents[s] == ents[d]:
            continue
        out.add(_orient(ents[s], ents[d], labels[s], labels[d]))
    return out


def _new_stats() -> dict:
    return {"entity_lines": 0, "link_lines": 0, "bad_ranges": 0, "unmatched": 0, "dangling_links": 0, "parse_fail": 0}


def _finish(stats: dict, text: str) -> None:
    stats["link_lines"] = sum(1 for _ in _LINK.finditer(text))
    stats["parse_fail"] = int(stats["entity_lines"] == 0 and stats["link_lines"] == 0)


def _mention(part: str, n_words: int, stats: dict):
    m = _MENTION.match(part)
    if not m:
        return None
    a = int(m.group(2))
    b = int(m.group(3)) if m.group(3) is not None else a
    if not (0 <= a <= b < n_words):
        stats["bad_ranges"] += 1
        return None
    return LETTER[m.group(1).upper()], (a, b + 1)


def parse_v2(text: str, n_words: int) -> tuple[set[Entity], set[Link], dict]:
    """Lines 'X a-b' (entity) or 'X a-b > Y c-d' (link, both ends become entities); the first label seen for a range wins."""
    stats, labels, links = _new_stats(), {}, set()
    for line in text.splitlines():
        parts = [p for p in re.split(r"\s*(?:->|=>|>|→)\s*", line.strip()) if p]
        if not parts or len(parts) > 2:
            continue
        found = [_mention(p, n_words, stats) for p in parts]
        if any(f is None for f in found):
            continue
        for lab, rng in found:
            labels.setdefault(rng, lab)
        stats["entity_lines" if len(found) == 1 else "link_lines"] += 1
        if len(found) == 2 and found[0][1] != found[1][1]:
            s, d = found[0][1], found[1][1]
            links.add(_orient(s, d, labels[s], labels[d]))
    stats["parse_fail"] = int(stats["entity_lines"] == 0 and stats["link_lines"] == 0)
    return {(lab, *rng) for rng, lab in labels.items()}, links, stats


def parse_v3(text: str, ex) -> tuple[set[Link], dict]:
    stats = _new_stats()
    ids = entity_list(ex)
    ents, labels = {}, {}
    for n, i in enumerate(ids, 1):
        idx = ex.entities[i]["token_indices"]
        ents[n], labels[n] = (min(idx), max(idx) + 1), ex.entities[i]["label"].lower()
    links = _links(text, "e", ents, labels, stats)
    _finish(stats, text)
    stats["entity_lines"] = len(ids)  # given, not generated
    stats["parse_fail"] = int(stats["link_lines"] == 0)
    return links, stats


def norm_tokens(s: str) -> list[str]:
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^0-9a-z]+", " ", s).split()


def _word_tokens(ex) -> list[tuple[str, int]]:
    return [(t, w) for w, word in enumerate(ex.words) for t in norm_tokens(word)]


def ground_text(text: str, toks: list[tuple[str, int]], used: list[tuple[int, int]], cursor: int = 0) -> tuple[int, int] | None:
    """Exact run of the text's normalised tokens in the form's words that does not overlap a used span.

    The search starts at token ``cursor`` (just after the previous entity, since the VLM lists entities in reading
    order) and wraps around, so a repeated label like 'Desde:' goes to the next free occurrence."""
    q = norm_tokens(text)
    if not q:
        return None
    n = len(toks) - len(q) + 1
    for i in [*range(min(cursor, max(n, 0)), n), *range(0, min(cursor, max(n, 0)))]:
        if all(toks[i + k][0] == q[k] for k in range(len(q))):
            a, b = toks[i][1], toks[i + len(q) - 1][1] + 1
            if not any(a < ub and b > ua for ua, ub in used):
                return a, b
    return None


def _is_mark(word: str) -> bool:
    """A checkbox, bullet or dash: symbols that sit in front of the label they belong to."""
    return all(unicodedata.category(c) in ("So", "Sm", "Po", "Pd") and c not in ":;.,)" for c in word) and         any(unicodedata.category(c) == "So" or c in "•·-" for c in word)


def _absorb_symbols(ex, span: tuple[int, int], used: list[tuple[int, int]]) -> tuple[int, int]:
    """Widen a matched span over neighbouring words with no letters or digits, which a VLM does not retype but the
    annotation includes: checkbox marks and bullets on the left, punctuation (a lone ':' or ')') on the right."""
    a, b = span
    free = lambda w: not norm_tokens(ex.words[w]) and not any(ua <= w < ub for ua, ub in used)
    while a > 0 and free(a - 1) and _is_mark(ex.words[a - 1]):
        a -= 1
    while b < len(ex.words) and free(b) and not _is_mark(ex.words[b]):
        b += 1
    return a, b


def parse_v1(text: str, ex) -> tuple[set[Entity], set[Link], dict]:
    stats, ents, labels, used = _new_stats(), {}, {}, []
    toks = _word_tokens(ex)
    tok_at = {w: k for k, (_, w) in reversed(list(enumerate(toks)))}  # first token index of each word
    cursor = 0
    for line in text.splitlines():
        m = _ENT_V1.match(line)
        if not m:
            continue
        stats["entity_lines"] += 1
        n, lab = int(m.group(1)), LETTER[m.group(2).upper()]
        if n in ents:
            continue
        span = ground_text(m.group(3), toks, used, cursor)
        if span is None:
            stats["unmatched"] += 1
            continue
        span = _absorb_symbols(ex, span, used)
        ents[n], labels[n] = span, lab
        used.append(span)
        cursor = next((tok_at[w] for w in range(span[1], len(ex.words)) if w in tok_at), len(toks))
    links = _links(text, "E", ents, labels, stats)
    _finish(stats, text)
    return {(labels[n], *ents[n]) for n in ents}, links, stats


def parse(arm: str, text: str, ex) -> tuple[set[Entity], set[Link], dict]:
    """(entities, links, stats) for any arm; V3 returns the gold entities (they were given)."""
    if arm == "V2":
        return parse_v2(text, len(ex.words))
    if arm == "V1":
        return parse_v1(text, ex)
    if arm == "V3":
        links, stats = parse_v3(text, ex)
        return gold_structure(ex)[0], links, stats
    raise ValueError(arm)


# ---------------------------------------------------------------- perfect answers (tests and dry runs)
def render_gold(arm: str, ex) -> str:
    """The answer a perfect VLM would write for this form in the arm's format."""
    if arm == "V3":
        num = {i: n for n, i in enumerate(entity_list(ex), 1)}
        return "\n".join(f"e{num[s]}>e{num[d]}" for s, d in ex.links if s in num and d in num and s != d)
    if arm == "V2":
        rng = {i: (min(e["token_indices"]), max(e["token_indices"])) for i, e in ex.entities.items()
               if e.get("token_indices") and e["label"].lower() in LETTER_OF}
        item = lambda i: f"{LETTER_OF[ex.entities[i]['label'].lower()]} {rng[i][0]}-{rng[i][1]}"
        pairs = [(s, d) for s, d in ex.links if s in rng and d in rng and s != d]
        linked = {i for p in pairs for i in p}
        return "\n".join([f"{item(s)} > {item(d)}" for s, d in pairs] + [item(i) for i in sorted(rng) if i not in linked])
    num, rows = {}, []
    for i in entity_list(ex):
        lab = ex.entities[i]["label"].lower()
        if lab not in LETTER_OF:
            continue
        num[i] = len(num) + 1
        idx = ex.entities[i]["token_indices"]
        rows.append(f"E{num[i]} {LETTER_OF[lab]} | {_entity_text(ex, i)}")
    rows += [f"E{num[s]}>E{num[d]}" for s, d in ex.links if s in num and d in num and s != d]
    return "\n".join(rows)
