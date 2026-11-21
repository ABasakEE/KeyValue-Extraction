"""Study 34, GPU probe (Kaggle smoke cell): does the T4 attention memory grow with the square of the prompt length, and does the fix remove it?

Background (CPU diagnostics, `attn_probe_cpu.py`): transformers 4.57 passes `enable_gqa=True` to scaled_dot_product_attention when there is no mask. A T4 has no
flash kernel, and as far as I know the memory-efficient kernel does not accept `enable_gqa`, so PyTorch falls back to the math kernel, which builds the full
L x L attention matrix: 32 heads x 5,463^2 x 4 bytes = 3.56 GiB, the size of the failed 3.53 GiB allocation. The fix is `adbuy_sg.patch_sdpa()` (repeat the keys and
values so the efficient kernel is used). This probe measures both, first on bare attention, then on the real model with text-only prompts.

    python scripts/attn_probe_gpu.py          # exits 1 only if the patched model cannot prefill 5,500 positions (the size of the page that failed)
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "src")]
GIB = 2**30


def micro() -> None:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    print("== bare attention, 32 query heads, 8 key/value heads, head size 128, fp16, causal (extra GiB above the inputs)")
    print(f"{'positions':>10s} | {'GQA flag, default dispatch':>28s} | {'GQA flag, efficient kernel only':>32s} | {'keys repeated, efficient kernel only':>38s}")
    for L in (2000, 5500, 9500):
        q = torch.randn(1, 32, L, 128, device="cuda", dtype=torch.float16)
        k, v = (torch.randn(1, 8, L, 128, device="cuda", dtype=torch.float16) for _ in range(2))
        cells = []
        for gqa, backends in ((True, None), (True, [SDPBackend.EFFICIENT_ATTENTION]), (False, [SDPBackend.EFFICIENT_ATTENTION])):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            try:
                kk, vv = (k, v) if gqa else (k.repeat_interleave(4, 1), v.repeat_interleave(4, 1))
                with (sdpa_kernel(backends) if backends else contextlib.nullcontext()):
                    out = F.scaled_dot_product_attention(q, kk, vv, is_causal=True, **({"enable_gqa": True} if gqa else {}))
                torch.cuda.synchronize()
                cells.append(f"{(torch.cuda.max_memory_allocated() - base) / GIB:.2f}")
                del out
            except torch.OutOfMemoryError:
                cells.append("OOM")
            except RuntimeError as e:
                cells.append("unsupported: " + str(e).splitlines()[0][:40])
        print(f"{L:>10d} | {cells[0]:>28s} | {cells[1]:>32s} | {cells[2]:>38s}")
        del q, k, v


def model_table() -> bool:
    import transformers.integrations.sdpa_attention as sa
    from schema_guided_kv import _vlm_class

    vlm = _vlm_class()("Qwen/Qwen3-VL-8B-Instruct", "cuda:0", "float16", 1280 * 28 * 28, True)
    model = vlm.model
    print(f"\n== real model, text-only prefill of random tokens; attention implementation: {model.config._attn_implementation} / text {model.config.text_config._attn_implementation}")
    print(f"weights and context after loading: {torch.cuda.memory_allocated() / GIB:.2f} GiB allocated, {torch.cuda.mem_get_info()[0] / GIB:.2f} GiB free of {torch.cuda.mem_get_info()[1] / GIB:.2f}")
    original, ok_5500 = sa.use_gqa_in_sdpa, False
    for patched in (False, True):
        sa.use_gqa_in_sdpa = (lambda attention_mask, key: False) if patched else original
        for n in (3000, 5500, 7500, 9500):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            ids = torch.randint(1000, 100000, (1, n), device="cuda:0")
            try:
                with torch.no_grad():
                    model(input_ids=ids, attention_mask=torch.ones_like(ids), logits_to_keep=1)
                status = f"peak {torch.cuda.max_memory_allocated() / GIB:5.2f} GiB"
                ok_5500 |= patched and n == 5500
            except torch.OutOfMemoryError:
                status = "OUT OF MEMORY"
            print(f"  {'with fix (keys repeated)' if patched else 'as shipped (GQA flag)':26s} {n:>5d} positions: {status}")
            del ids
            torch.cuda.empty_cache()
    sa.use_gqa_in_sdpa = original
    return ok_5500


if __name__ == "__main__":
    print("torch", torch.__version__, "|", torch.cuda.get_device_name(0))
    micro()
    sys.exit(0 if model_table() else 1)
