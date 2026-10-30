"""Study 16 (CPU scoring): per-document routing between BROS and the zero-shot VLM, on FATURA and VRDU.

Signals, from BROS's own per-word field probabilities (test_probs.jsonl, merged B/I, integer percent):
  confidence   mean probability of field f over BROS's own predicted span
  endorsement  mean probability of field f over the VLM's proposed span (how plausible BROS itself finds it)

Rules (registered before any run), applied per invoice/document and field f, with one VLM span per field:
  G  gate        if BROS predicted a span for f and its confidence < tau, replace it with the VLM span.
                 BROS silent for f -> stay silent (avoids fill-in's harm on fields a template lacks).
  E  endorse     G, plus: if BROS is silent for f and the VLM span's endorsement >= kappa, add the VLM span.
  H  hybrid      the static per-field router of studies 10-15 first (fields BROS loses on other folds go to the
                 VLM), then G on the remaining fields. Needed on VRDU, whose main gain is a field BROS never
                 predicts, which G cannot touch by construction.
tau in {.5,.6,.7,.8,.9,.95,.99}, kappa in {.1,.3,.5} are chosen only on the OTHER folds' documents (dev + confirm),
separately per seed, maximising pooled micro F1; ties go to fewer replacements (lower tau, higher kappa). Claims use
the held-out fold's confirm documents only. Compared with BROS, the static router, fill-in, the oracle and the VLM alone.

Intervals: two-level bootstrap, resampling seeds (the draw of seen templates / training run) and clusters
(FATURA: templates; VRDU: documents, since it has only 3 templates), 1000 draws, percentile 95%.

Registered success criteria:
  FATURA (K = 2, 5, 10 seen templates; primary rule G): G - BROS pooled interval lower bound > 0 at K = 2 or K = 5,
    AND G - BROS interval upper bound >= 0 at every K. Seed replication (#2): the static router's K = 2 loss counts as
    replicated only if its interval upper bound is < 0 with the 5 seeds.
  VRDU (3 templates, fresh confirm documents; primary rule H): H - BROS pooled interval lower bound > 0 AND no
    template's interval upper bound < 0. G, E and H are also reported on both datasets as descriptive results.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fatura_eval import same_spans  # noqa: E402
from hybrid_eval import fill_in, fit_router, oracle_counts  # noqa: E402
from matched_eval import f1_of, span_counts  # noqa: E402

TAUS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
KAPPAS = (0.1, 0.3, 0.5)
N_BOOT = 1000
SYSTEMS = ("bros", "static", "gate", "endorse", "hybrid", "fill_in", "oracle", "vlm")


# ---------------------------------------------------------------- signals and rules
def support(p_field: np.ndarray, span: list) -> float:
    return float(p_field[span[1]:span[2]].mean()) if span[2] > span[1] else 0.0


def apply_rule(bros: list, vlm: list, probs: dict, tau: float | None, kappa: float | None,
               routed: frozenset = frozenset()) -> list:
    b_by, v_by = defaultdict(list), {}
    for sp in bros:
        b_by[sp[0]].append(sp)
    for sp in vlm:
        v_by[sp[0]] = sp
    kept, added = [], []
    for f in sorted(set(b_by) | set(v_by)):
        b, v = b_by.get(f, []), v_by.get(f)
        if f in routed:
            if v:
                added.append(v)
        elif b:
            if tau is not None and v and v not in b and max(support(probs[f], sp) for sp in b) < tau:
                added.append(v)
            else:
                kept += b
        elif kappa is not None and v and support(probs[f], v) >= kappa:
            added.append(v)
    used = {i for sp in kept for i in range(sp[1], sp[2])}
    for sp in added:                                   # a VLM span never overwrites a kept BROS span
        if not used.intersection(range(sp[1], sp[2])):
            kept.append(sp)
            used.update(range(sp[1], sp[2]))
    return kept


def micro(counts: list[dict]) -> float:
    return f1_of([sum(c["_micro"][i] for c in counts) for i in range(3)])


def fit_params(fit: list[dict], seed: int, rule: str, routed: frozenset) -> dict:
    """Grid-search tau (and kappa) on the fit documents; ties -> fewer replacements."""
    best, grid = None, [(t, None) for t in TAUS] if rule != "endorse" else [(t, k) for t in TAUS for k in KAPPAS]
    for tau, kappa in grid:
        f1 = micro([span_counts(apply_rule(r["bros"][seed], r["vlm"], r["probs"][seed], tau, kappa, routed), r["gold"]) for r in fit])
        key = (f1, -tau, kappa if kappa is not None else 0)
        if best is None or key > best[0]:
            best = (key, {"tau": tau, "kappa": kappa})
    return best[1]


def systems_for(test: list[dict], fit: list[dict], seed: int) -> tuple[dict[str, list[dict]], dict]:
    bk = f"bros_seed{seed}"
    views = [{"gold": r["gold"], bk: r["bros"][seed], "vlm_both": r["vlm"]} for r in fit]
    table = fit_router(views, bk, ("vlm_both",))
    routed = frozenset(f for f, src in table.items() if src == "vlm_both")
    params = {"routed": sorted(routed), "gate": fit_params(fit, seed, "gate", frozenset()),
              "endorse": fit_params(fit, seed, "endorse", frozenset()), "hybrid": fit_params(fit, seed, "hybrid", routed)}
    out: dict[str, list[dict]] = defaultdict(list)
    for r in test:
        b, v, p, g = r["bros"][seed], r["vlm"], r["probs"][seed], r["gold"]
        out["bros"].append(span_counts(b, g))
        out["static"].append(span_counts(apply_rule(b, v, p, None, None, routed), g))
        out["gate"].append(span_counts(apply_rule(b, v, p, params["gate"]["tau"], None), g))
        out["endorse"].append(span_counts(apply_rule(b, v, p, params["endorse"]["tau"], params["endorse"]["kappa"]), g))
        out["hybrid"].append(span_counts(apply_rule(b, v, p, params["hybrid"]["tau"], None, routed), g))
        out["fill_in"].append(span_counts(fill_in(b, v), g))
        out["oracle"].append(oracle_counts(g, {"b": b, "v": v}, ["b", "v"]))
        out["vlm"].append(span_counts(v, g))
    return dict(out), params


# ---------------------------------------------------------------- bootstrap and diagnostics
def f1_vec(tot: np.ndarray) -> np.ndarray:
    tp, npred, ngold = tot[..., 0], tot[..., 1], tot[..., 2]
    p = np.divide(tp, npred, out=np.zeros_like(tp), where=npred > 0)
    r = np.divide(tp, ngold, out=np.zeros_like(tp), where=ngold > 0)
    return np.divide(2 * p * r, p + r, out=np.zeros_like(tp), where=(p + r) > 0)


def two_level_bootstrap(per_seed: dict[str, dict[int, list[dict]]], clusters: list, n_boot: int = N_BOOT, seed: int = 0) -> dict[str, np.ndarray]:
    """F1 per system over paired draws: clusters resampled, then a seed multiset resampled."""
    names, seeds = list(per_seed), sorted(next(iter(per_seed.values())))
    arr = np.array([[[d["_micro"] for d in per_seed[n][s]] for s in seeds] for n in names], dtype=float)   # (sys, seed, doc, 3)
    groups: dict = defaultdict(list)
    for i, c in enumerate(clusters):
        groups[c].append(i)
    keys, rng = list(groups), np.random.default_rng(seed)
    out = {n: np.empty(n_boot) for n in names}
    for b in range(n_boot):
        idx = np.concatenate([groups[keys[j]] for j in rng.integers(0, len(keys), len(keys))])
        chosen = rng.integers(0, len(seeds), len(seeds))
        f1 = f1_vec(arr[:, :, idx, :].sum(axis=2))[:, chosen].mean(axis=1)
        for j, n in enumerate(names):
            out[n][b] = f1[j]
    return out


def ci(x: np.ndarray) -> list[float]:
    return [round(float(v), 4) for v in np.percentile(x, [2.5, 97.5])]


def auroc(scores: list[float], labels: list[bool]) -> float | None:
    pos, neg = [s for s, l in zip(scores, labels) if l], [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    s_sorted = np.asarray(scores)[order]
    i = 0
    while i < len(s_sorted):                                   # average ranks over ties
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    r_pos = sum(ranks[k] for k, l in enumerate(labels) if l)
    return round(float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))), 4)


def confidence_auroc(recs: list[dict], seeds: list[int]) -> dict:
    scores, labels = [], []
    for r in recs:
        gold = {tuple(g[:3]) for g in r["gold"]}
        for s in seeds:
            for sp in r["bros"][s]:
                scores.append(support(r["probs"][s][sp[0]], sp))
                labels.append(tuple(sp[:3]) in gold)
    return {"auroc": auroc(scores, labels), "n_spans": len(labels), "share_correct": round(float(np.mean(labels)), 4) if labels else None}


def analyse(recs: list[dict], seeds: list[int], cluster_of) -> dict:
    """recs: dicts with fold, split, template, guid, gold, vlm, bros{seed}, probs{seed}."""
    confirm = [r for r in recs if r["split"] == "confirm"]
    folds = sorted({r["fold"] for r in recs})
    per_seed: dict[str, dict[int, list[dict]]] = {n: {s: [] for s in seeds} for n in SYSTEMS}
    order, params = [], {}
    for f in folds:
        test = [r for r in confirm if r["fold"] == f]
        fit = [r for r in recs if r["fold"] != f]
        order += test
        for s in seeds:
            sysd, params[f"fold{f}_seed{s}"] = systems_for(test, fit, s)
            for n in SYSTEMS:
                per_seed[n][s] += sysd[n]
    clusters = [cluster_of(r) for r in order]
    boot = two_level_bootstrap(per_seed, clusters)

    def point(name):
        return float(np.mean([micro(per_seed[name][s]) for s in seeds]))

    out = {"seeds": seeds, "n_confirm_docs": len(order), "f1": {n: round(point(n), 4) for n in SYSTEMS}, "vs_bros": {},
           "per_seed_delta": {}, "chosen_params": params}
    for n in SYSTEMS:
        if n == "bros":
            continue
        d = boot[n] - boot["bros"]
        out["vs_bros"][n] = {"delta": round(point(n) - point("bros"), 4), "ci95": ci(d)}
        out["per_seed_delta"][n] = {str(s): round(micro(per_seed[n][s]) - micro(per_seed["bros"][s]), 4) for s in seeds}
    head = out["f1"]["oracle"] - out["f1"]["bros"]
    out["oracle_headroom"] = round(head, 4)
    out["share_of_headroom"] = {n: round((out["f1"][n] - out["f1"]["bros"]) / head, 3) if head > 1e-9 else None
                                for n in ("static", "gate", "endorse", "hybrid")}
    out["gate_vs_static"] = {"delta": round(point("gate") - point("static"), 4), "ci95": ci(boot["gate"] - boot["static"])}
    out["hybrid_vs_static"] = {"delta": round(point("hybrid") - point("static"), 4), "ci95": ci(boot["hybrid"] - boot["static"])}
    out["confidence_signal"] = confidence_auroc(recs, seeds)
    by_fold = {}
    for f in folds:                                         # per-fold interval for the 'no template/fold worse' checks
        idx = [i for i, r in enumerate(order) if r["fold"] == f]
        sub = {n: {s: [per_seed[n][s][i] for i in idx] for s in seeds} for n in SYSTEMS}
        b = two_level_bootstrap(sub, [clusters[i] for i in idx])
        by_fold[f"fold{f}"] = {n: {"delta": round(float(np.mean([micro(sub[n][s]) - micro(sub["bros"][s]) for s in seeds])), 4),
                                   "ci95": ci(b[n] - b["bros"])} for n in ("static", "gate", "endorse", "hybrid")}
        by_fold[f"fold{f}"]["bros_f1"] = round(float(np.mean([micro(sub["bros"][s]) for s in seeds])), 4)
    out["by_fold"] = by_fold
    return out


# ---------------------------------------------------------------- verdicts
def fatura_verdict(per_k: dict[str, dict]) -> dict:
    gain = [k for k in ("2", "5") if k in per_k and per_k[k]["vs_bros"]["gate"]["ci95"][0] > 0]
    never_worse = all(v["vs_bros"]["gate"]["ci95"][1] >= 0 for v in per_k.values())
    return {"gate_gains_at_K": gain, "gate_never_significantly_worse": never_worse, "passes": bool(gain) and never_worse,
            "static_router_K2_loss_replicated": "2" in per_k and per_k["2"]["vs_bros"]["static"]["ci95"][1] < 0}


def vrdu_verdict(res: dict) -> dict:
    pooled_ok = res["vs_bros"]["hybrid"]["ci95"][0] > 0
    no_loss = all(v["hybrid"]["ci95"][1] >= 0 for v in res["by_fold"].values())
    return {"hybrid_pooled_gain_significant": pooled_ok, "no_template_significantly_worse": no_loss, "passes": pooled_ok and no_loss}


# ---------------------------------------------------------------- adapters
def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_probs(path: Path) -> dict[str, dict[str, np.ndarray]]:
    out = {}
    for row in read_jsonl(path):
        arr = np.asarray(row["p"], dtype=float).reshape(-1, len(row["fields"])) / 100.0
        out[row["guid"]] = {f: arr[:, j] for j, f in enumerate(row["fields"])}
    return out


def validate_probs(path: Path, n_words_of: dict[str, int], keep: set[str] | None = None) -> int:
    """Smoke-test helper: one row per kept document, shape (n_words, n_fields), integer percents in 0..100."""
    rows = read_jsonl(path)
    assert rows, f"{path} is empty"
    for r in rows:
        assert keep is None or r["guid"] in keep, f"{r['guid']} was not requested"
        p = np.asarray(r["p"])
        assert p.shape == (n_words_of[r["guid"]], len(r["fields"])), (r["guid"], p.shape, n_words_of[r["guid"]], r["fields"])
        assert p.min() >= 0 and p.max() <= 100 and np.issubdtype(p.dtype, np.integer), r["guid"]
    if keep is not None:
        assert {r["guid"] for r in rows} == keep & set(n_words_of), "requested documents missing from the dump"
    return len(rows)


def attach_runs(recs: list[dict], seeds: list[int], run_dir) -> tuple[list[dict], dict]:
    """run_dir(fold, seed) -> directory holding test_entities.jsonl and test_probs.jsonl. Drops incomplete documents."""
    info = {"alignment_mismatches": 0, "missing": 0}
    cache = {}
    for f in sorted({r["fold"] for r in recs}):
        for s in seeds:
            d = run_dir(f, s)
            cache[(f, s)] = ({r["guid"]: r for r in read_jsonl(d / "test_entities.jsonl")}, load_probs(d / "test_probs.jsonl"))
    out = []
    for r in recs:
        new = dict(r, bros={}, probs={})
        ok = True
        for s in seeds:
            ent, pr = cache[(r["fold"], s)]
            if r["guid"] not in ent or r["guid"] not in pr:
                ok = False
                break
            info["alignment_mismatches"] += not same_spans(ent[r["guid"]]["gold"], r["gold"])
            new["bros"][s], new["probs"][s] = ent[r["guid"]]["pred"], pr[r["guid"]]
        info["missing"] += not ok
        if ok:
            out.append(new)
    return out, info


def base_records(rows: list[dict], split: str | None = None) -> list[dict]:
    return [{"fold": r["fold"], "template": r["template"], "guid": r["guid"], "split": split or r["split"],
             "gold": r["gold"], "vlm": r["vlm_both"]} for r in rows]


def detect_seeds(root_of, folds: list[int], candidates=(13, 42, 7, 21, 99)) -> list[int]:
    return [s for s in candidates if all((root_of(f, s) / "test_probs.jsonl").exists() for f in folds)]


def run_fatura(spans: str, routing_dir: str, ks: list[str], out: str) -> dict:
    base = base_records(read_jsonl(Path(spans)))
    per_k = {}
    for k in ks:
        d = lambda f, s, k=k: Path(routing_dir) / f"K{k}" / f"fold{f}" / f"seed_{s}"
        seeds = detect_seeds(d, sorted({r["fold"] for r in base}))
        if not seeds:
            per_k[k] = {"error": "no complete runs found"}
            continue
        recs, info = attach_runs(base, seeds, d)
        per_k[k] = {**analyse(recs, seeds, lambda r: r["template"]), "alignment": info, "invoices": len(recs)}
    ok = {k: v for k, v in per_k.items() if "error" not in v}
    res = {"settings": per_k, "verdict": fatura_verdict(ok) if ok else None}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def run_vrdu(dev_spans: str, confirm_spans: str, bros_dir: str, out: str) -> dict:
    base = base_records(read_jsonl(Path(dev_spans)), "dev") + base_records(read_jsonl(Path(confirm_spans)), "confirm")
    d = lambda f, s: Path(bros_dir) / f"UTL_fold{f}_seed{s}"
    seeds = detect_seeds(d, sorted({r["fold"] for r in base}), candidates=(13, 42))
    if not seeds:
        raise FileNotFoundError("no complete VRDU BROS runs with test_probs.jsonl found")
    recs, info = attach_runs(base, seeds, d)
    res = analyse(recs, seeds, lambda r: r["guid"])        # 3 templates: resample documents, not templates
    res.update({"alignment": info, "documents": len(recs), "verdict": vrdu_verdict(res)})
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


# ---------------------------------------------------------------- self-test
def _selftest() -> None:
    P = lambda **kw: {f: np.array(v, dtype=float) for f, v in kw.items()}
    probs = P(a=[.9, .9, .2, .2, .2], b=[.2, .2, .9, .9, .2], c=[.1, .1, .1, .1, .1])
    bros_a, bros_b = ["a", 0, 2, "t"], ["b", 2, 4, "u"]
    vlm_a2, vlm_b2, vlm_c = ["a", 2, 4, "x"], ["b", 3, 5, "y"], ["c", 4, 5, "z"]
    # G: confident BROS span kept; low-confidence one replaced; silent field stays silent
    out = apply_rule([bros_a, bros_b], [vlm_a2, vlm_c], probs, 0.5, None)
    assert bros_a in out and vlm_c not in out, out
    probs_low = P(a=[.3, .3, .2, .2, .2], b=probs["b"], c=probs["c"])
    out = apply_rule([bros_a], [vlm_a2], probs_low, 0.5, None)
    assert out == [vlm_a2], out                                # support 0.3 < 0.5 -> replaced by the VLM span
    # E: fill a silent field only when BROS endorses the VLM span
    probs_e = P(a=probs["a"], b=probs["b"], c=[.1, .1, .1, .1, .6])
    assert apply_rule([], [vlm_c], probs_e, 0.5, 0.5) == [vlm_c] and apply_rule([], [vlm_c], probs, 0.5, 0.5) == []
    # H: routed fields come from the VLM, the rest are gated
    out = apply_rule([bros_a, bros_b], [vlm_a2, vlm_b2], probs, 0.5, None, frozenset({"b"}))
    assert vlm_b2 in out and bros_a in out and bros_b not in out, out
    # a VLM span never overwrites a kept BROS span
    assert apply_rule([bros_a], [["b", 1, 3, "q"]], probs, None, 0.0) == [bros_a]
    assert auroc([.9, .8, .3, .2], [True, True, False, False]) == 1.0 and auroc([.5, .5], [True, False]) == 0.5
    assert auroc([.5], [True]) is None
    # verdict logic
    mk = lambda lo_hi, st=(-0.01, 0.01): {"vs_bros": {"gate": {"ci95": list(lo_hi)}, "static": {"ci95": list(st)}}}
    assert fatura_verdict({"2": mk((0.01, 0.05)), "5": mk((0, 0.02)), "10": mk((-0.001, 0.001))})["passes"]
    assert not fatura_verdict({"2": mk((0.01, 0.05)), "5": mk((-0.03, -0.01)), "10": mk((0, 0.0))})["passes"]   # worse at K=5
    assert not fatura_verdict({"2": mk((-0.01, 0.05)), "5": mk((0, 0.02)), "10": mk((0, 0.0))})["passes"]       # no gain anywhere
    assert fatura_verdict({"2": mk((0.0, 0.1), st=(-0.05, -0.01))})["static_router_K2_loss_replicated"]
    res = {"vs_bros": {"hybrid": {"ci95": [0.01, 0.05]}}, "by_fold": {"fold0": {"hybrid": {"ci95": [-0.02, 0.04]}}}}
    assert vrdu_verdict(res)["passes"]
    res["by_fold"]["fold1"] = {"hybrid": {"ci95": [-0.05, -0.01]}}
    assert not vrdu_verdict(res)["passes"]
    # end to end on a toy: 2 folds x 3 docs, planted weakness: BROS's field-a span is wrong with low confidence
    def toy(fold, i, split):
        gold = [["a", 0, 2, "g"]]
        bad = ["a", 2, 4, "w"]
        return {"fold": fold, "template": f"T{fold}", "guid": f"d{fold}{i}", "split": split, "gold": gold, "vlm": gold,
                "bros": {1: [bad], 2: [bad]}, "probs": {1: {"a": np.array([.1, .1, .3, .3, .1])}, 2: {"a": np.array([.1, .1, .3, .3, .1])}}}
    recs = [toy(f, i, "confirm" if i else "dev") for f in (0, 1) for i in range(3)]
    res = analyse(recs, [1, 2], lambda r: r["template"])
    assert res["f1"]["bros"] == 0.0 and res["f1"]["gate"] == 1.0 and res["f1"]["oracle"] == 1.0, res["f1"]
    assert res["share_of_headroom"]["gate"] == 1.0 and res["confidence_signal"]["auroc"] is None
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "p.jsonl"
        f.write_text(json.dumps({"guid": "d", "fields": ["a", "b"], "p": [[0, 100], [5, 7], [1, 2]]}) + "\n")
        assert validate_probs(f, {"d": 3}, {"d"}) == 1
        for bad_words, bad_keep in (({"d": 4}, None), ({"d": 3}, {"other"})):
            try:
                validate_probs(f, bad_words, bad_keep)
            except AssertionError:
                continue
            raise AssertionError("validate_probs accepted a wrong dump")
    print("[+] routing_eval selftest passed")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fatura")
    f.add_argument("--spans", default="results/fatura_v1/spans.jsonl")
    f.add_argument("--routing-dir", required=True)
    f.add_argument("--ks", default="2,5,10")
    f.add_argument("--out", required=True)
    v = sub.add_parser("vrdu")
    v.add_argument("--dev-spans", default="results/matched_eval/spans.jsonl")
    v.add_argument("--confirm-spans", default="results/confirm_v2/matched/spans.jsonl")
    v.add_argument("--bros-dir", required=True)
    v.add_argument("--out", required=True)
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "selftest":
        _selftest()
    elif a.cmd == "fatura":
        res = run_fatura(a.spans, a.routing_dir, a.ks.split(","), a.out)
        print(json.dumps({"verdict": res["verdict"], **{f"K{k}": {n: v["f1"][n] for n in ("bros", "static", "gate", "hybrid", "oracle")}
                                                          for k, v in res["settings"].items() if "error" not in v}}, indent=2))
    else:
        res = run_vrdu(a.dev_spans, a.confirm_spans, a.bros_dir, a.out)
        print(json.dumps({"verdict": res["verdict"], "f1": res["f1"]}, indent=2))


if __name__ == "__main__":
    main()
