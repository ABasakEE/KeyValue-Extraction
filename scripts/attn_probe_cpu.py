"""Study 34, CPU diagnostics for the T4 out-of-memory on ad-buy pages (no GPU here, so memory itself is not measured).

    <venv python> scripts/attn_probe_cpu.py positions   # exact sequence length of a page prompt with the real Qwen3-VL processor
    <venv python> scripts/attn_probe_cpu.py tiny        # tiny random Qwen3-VL: which attention path runs, and does prefill compute logits for every position?

Run in a venv with transformers >= 4.57, < 5 (what the Kaggle notebooks install), torch, pillow, pymupdf. Needs internet only for the
processor files (config and tokenizer, no weights).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def positions() -> None:
    import adbuy_rows as R
    import adbuy_sg as S
    import pymupdf
    from PIL import Image
    from transformers import AutoProcessor

    fn = "0617c152-6638-cd96-409e-8af194730069.pdf"   # the page that ran out of memory (page 0), and page 1 of the same document
    rec = next(r for r in R.load_records() if r["filename"] == fn)
    doc = S.make_doc(rec)
    for max_px in (1280 * 28 * 28, 640 * 28 * 28):
        proc = AutoProcessor.from_pretrained(MODEL, min_pixels=256 * 28 * 28, max_pixels=max_px)
        proc.image_processor.size = {"shortest_edge": 256 * 28 * 28, "longest_edge": max_px}
        for page in (0, 1):
            with pymupdf.open(ROOT / "data" / "vrdu" / "ad-buy-form" / "main" / "pdfs" / fn) as d:
                pix = d[page].get_pixmap(dpi=150)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            prompt = S.build_prompt(doc.lines[page])
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = proc(text=[text], images=[img], return_tensors="pt")
            thw = enc["image_grid_thw"][0]
            n_img = int(thw.prod()) // 4
            n = enc["input_ids"].shape[1]
            print(f"max_pixels {max_px:>8d} page {page}: image {pix.width}x{pix.height} -> {n_img} visual tokens, total positions {n}; "
                  f"fp32 attention over 32 heads {32 * n * n * 4 / 2**30:.2f} GiB, fp32 logits for every position {n * 151936 * 4 / 2**30:.2f} GiB")
    print("the failed allocation was 3.53 GiB = 32 heads fp32 attention at", round((3.53 * 2**30 / (32 * 4)) ** 0.5), "positions, or full fp32 logits at",
          round(3.53 * 2**30 / (151936 * 4)), "positions")


def tiny() -> None:
    import torch
    import transformers
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    print("transformers", transformers.__version__, "torch", torch.__version__)
    cfg = Qwen3VLConfig(
        text_config=dict(vocab_size=2000, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=16, max_position_embeddings=4096, rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3]}),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, in_channels=3, patch_size=16, spatial_merge_size=2,
                           temporal_patch_size=2, out_hidden_size=64, num_position_embeddings=64, deepstack_visual_indexes=[0, 1]),
        image_token_id=1990, video_token_id=1991, vision_start_token_id=1992, vision_end_token_id=1993)
    import transformers.integrations.sdpa_attention as sa
    original = sa.use_gqa_in_sdpa
    for impl in ("sdpa", "sdpa+no-gqa patch", "eager"):
        sa.use_gqa_in_sdpa = (lambda attention_mask, key: False) if "patch" in impl else original
        model = Qwen3VLForConditionalGeneration._from_config(cfg, attn_implementation=impl.split("+")[0], dtype=torch.float32).eval()
        print(f"\n== requested attn_implementation={impl}: top {model.config._attn_implementation} | text {model.config.text_config._attn_implementation} "
              f"| vision {model.config.vision_config._attn_implementation}")
        seen = {"lm_head": [], "sdpa": []}
        h1 = model.lm_head.register_forward_hook(lambda m, i, o: seen["lm_head"].append(tuple(i[0].shape)))
        real_sdpa = torch.nn.functional.scaled_dot_product_attention

        def spy(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, **kw):
            seen["sdpa"].append((tuple(q.shape), tuple(k.shape), None if attn_mask is None else (attn_mask.dtype, tuple(attn_mask.shape)), is_causal, kw.get("enable_gqa")))
            return real_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kw)

        torch.nn.functional.scaled_dot_product_attention = spy
        try:
            ids = torch.randint(0, 1900, (1, 300))
            with torch.no_grad():
                model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=3, do_sample=False)
        finally:
            torch.nn.functional.scaled_dot_product_attention = real_sdpa
            h1.remove()
        print("   lm_head input shapes (prefill first):", seen["lm_head"][:3], "-> prefill computes logits for", "ONE position" if seen["lm_head"][0][1] == 1 else f"ALL {seen['lm_head'][0][1]} positions")
        print("   F.scaled_dot_product_attention calls:", len(seen["sdpa"]), "| first:", seen["sdpa"][0] if seen["sdpa"] else "none (eager matmul path)")


if __name__ == "__main__":
    {"positions": positions, "tiny": tiny}[sys.argv[1]]()
