"""Row repair shared by the ad-buy and DocILE recipes (study 34): the piece-choice rule for a cell whose cited id is wrong, and the attention fix.

`_pick` chooses among the pieces that hold a cell's text (rules none/unique/nearest/row, see adbuy_repair.py for the check); `patch_sdpa` stops transformers from
passing the GQA flag to scaled_dot_product_attention (a T4 then falls back to the math kernel whose memory grows with the square of the prompt length).
"""
from __future__ import annotations

import re

def _xy(lid: str | None):
    m = re.match(r"(\d{2})\|(\d{2})", lid or "")
    return (int(m[1]), int(m[2])) if m else None


def _pick(cand: list, lid: str | None, row_y: float | None, repair: str):
    """Choose among the pieces that hold a cell's text. unique: only if one piece holds it. nearest: closest to the cited id's coordinates.
    row: closest to the vertical position of the row's other (correct) cells, ties broken by the cited coordinates."""
    if not cand:
        return None
    if repair == "unique":
        return cand[0] if len(cand) == 1 else None
    xy = _xy(lid)
    d_cited = lambda c: ((c.box[0] + c.box[2]) * 50 - xy[0]) ** 2 + ((c.box[1] + c.box[3]) * 50 - xy[1]) ** 2 if xy else 0.0
    if repair == "row" and row_y is not None:
        key = lambda c: (round(abs((c.box[1] + c.box[3]) / 2 - row_y), 3), d_cited(c))
    elif repair in ("nearest", "row") and xy:
        key = lambda c: (d_cited(c),)
    else:
        return cand[0] if len(cand) == 1 else None
    ranked = sorted(cand, key=key)
    return ranked[0] if len(ranked) == 1 or key(ranked[0]) != key(ranked[1]) else None


def patch_sdpa() -> None:
    """transformers 4.57 passes enable_gqa=True to scaled_dot_product_attention when there is no mask. A T4 has no flash kernel and (as far as is known; the
    Kaggle probe `attn_probe_gpu.py` measures it) the memory-efficient kernel does not take that flag, so PyTorch falls back to the math kernel and builds the
    whole L x L attention matrix: 3.56 GiB for 32 heads at the 5,463 positions of the page that ran out of memory. Repeating the keys and values keeps the
    efficient kernel, whose memory grows linearly with the length. Same results, a little more memory for the repeated keys and values."""
    import transformers.integrations.sdpa_attention as sa
    sa.use_gqa_in_sdpa = lambda attention_mask, key: False
