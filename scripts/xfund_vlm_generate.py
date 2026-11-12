"""Study 27: run a zero-shot VLM over XFUND val forms for one arm and append its raw answers to a jsonl file.

    python scripts/xfund_vlm_generate.py --arm V2 --budget-h 6
    python scripts/xfund_vlm_generate.py --arm V2 --forms 3 --stub      # CPU check: writes perfect answers, no model

One row per form: guid, arm, model, seconds, new_tokens, prompt_tokens, truncated, error, text.  Forms already in
the file are skipped, so a stopped run resumes.  Forms go in ``vlm_link.form_order`` order (balanced prefix).
Scoring is separate and CPU-only: scripts/xfund_vlm_score.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data.xfund_loader import load_xfund_split  # noqa: E402
from vlm_link import PROMPTS, form_order, render_gold  # noqa: E402

LANGS = ["es", "fr", "de", "it", "pt"]
MAX_NEW = {"V1": 2048, "V2": 1536, "V3": 768}


class VLM:
    """Qwen2.5-VL split over every visible GPU (device_map='auto'), greedy decoding."""

    def __init__(self, model_id: str, dtype: str, max_pixels: int):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=getattr(torch, dtype), device_map="auto", attn_implementation="sdpa").eval()
        self.proc = AutoProcessor.from_pretrained(model_id, min_pixels=256 * 28 * 28, max_pixels=max_pixels)

    def __call__(self, prompt: str, image, max_new_tokens: int) -> tuple[str, int, int]:
        import torch

        content = ([{"type": "image"}] if image is not None else []) + [{"type": "text", "text": prompt}]
        text = self.proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        kw = {"text": [text], "return_tensors": "pt"}
        if image is not None:
            kw["images"] = [image]
        inputs = self.proc(**kw).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new = out[:, inputs["input_ids"].shape[1]:]
        return self.proc.batch_decode(new, skip_special_tokens=True)[0], int(new.shape[1]), int(inputs["input_ids"].shape[1])


def image_index(root: Path) -> dict[str, Path]:
    return {p.name: p for p in root.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")} if root.exists() else {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(PROMPTS), required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    ap.add_argument("--max-new-tokens", type=int, default=0, help="default: per arm (V1 2048, V2 1536, V3 768)")
    ap.add_argument("--data", default="data/xfund")
    ap.add_argument("--images", default="data/xfund/images")
    ap.add_argument("--forms", type=int, default=0, help="first N forms of the fixed order (0 = all 250)")
    ap.add_argument("--budget-h", type=float, default=0.0, help="stop cleanly after this many hours (0 = no cap)")
    ap.add_argument("--out", default="", help="default results/xfund_vlm/<arm>_raw.jsonl")
    ap.add_argument("--stub", action="store_true", help="no model: write the perfect answer (pipeline check)")
    a = ap.parse_args()

    start = time.time()
    forms = form_order(load_xfund_split(a.data, "val", LANGS))
    if a.forms:
        forms = forms[: a.forms]
    out = Path(a.out or f"results/xfund_vlm/{a.arm}_raw.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["guid"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [ex for ex in forms if ex.guid not in done]
    print(f"[{a.arm}] {len(todo)} of {len(forms)} forms to do ({len(done)} already in {out})", flush=True)
    if not todo:
        return
    images = image_index(Path(a.images))
    max_new = a.max_new_tokens or MAX_NEW[a.arm]
    gen = None if a.stub else VLM(a.model, a.dtype, a.max_pixels)
    model_name = "stub" if a.stub else a.model
    with open(out, "a", encoding="utf-8") as fh:
        for k, ex in enumerate(todo, 1):
            if a.budget_h and time.time() - start > a.budget_h * 3600:
                print(f"[{a.arm}] budget of {a.budget_h} h reached after {k - 1} forms; stopping (re-run resumes)", flush=True)
                break
            t0 = time.time()
            row = {"guid": ex.guid, "arm": a.arm, "model": model_name, "text": "", "new_tokens": 0, "prompt_tokens": 0,
                   "truncated": False, "error": None}
            try:
                if gen is None:
                    row["text"] = render_gold(a.arm, ex)
                else:
                    from PIL import Image

                    path = images.get(Path(ex.image_path).name)
                    if path is None:
                        raise FileNotFoundError(f"no image {ex.image_path} under {a.images}")
                    image = Image.open(path).convert("RGB")
                    row["text"], row["new_tokens"], row["prompt_tokens"] = gen(PROMPTS[a.arm](ex), image, max_new)
                    row["truncated"] = row["new_tokens"] >= max_new
            except Exception as e:  # one bad form (e.g. out of memory) is recorded and scored as an empty answer
                row["error"] = f"{type(e).__name__}: {e}"[:500]
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass
            row["seconds"] = round(time.time() - t0, 2)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            if k % 10 == 0 or k == len(todo) or row["error"]:
                print(f"[{a.arm}] {k}/{len(todo)} {ex.guid[:24]} {row['seconds']} s, {row['new_tokens']} tokens"
                      + (f", ERROR {row['error']}" if row["error"] else ""), flush=True)


if __name__ == "__main__":
    main()
