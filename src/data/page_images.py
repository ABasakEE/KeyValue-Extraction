"""Page images for image-aware backbones (LayoutLMv3): one normalised 3 x S x S tensor per (PDF, page), rendered once and cached as uint8.

A window of words is paired with the image of the page that holds most of its words (the earliest page on a tie): boxes are per page, the model sees one image.
``mode="blank"`` returns the same constant image for every window, so a run with it is the same model with the visual stream switched off (the control arm).
LayoutLMv3's own processor also resizes to 224 x 224 without keeping the aspect ratio, so relative box coordinates stay aligned with the pixels.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import torch


class PageImages:
    def __init__(self, pdf_dir: str | Path, size: int = 224, mode: str = "pages", dpi: int = 100):
        if mode not in ("pages", "blank"):
            raise ValueError(f"image mode {mode!r}: choose 'pages' or 'blank'")
        self.pdf_dir, self.size, self.mode, self.dpi = Path(pdf_dir), size, mode, dpi
        self._cache: dict[tuple[str, int], np.ndarray] = {}

    @staticmethod
    def page_of(example) -> int:
        pages = Counter(getattr(example, "page_ids", None) or [0])
        top = max(pages.values())
        return min(p for p, n in pages.items() if n == top)

    def _render(self, pdf: str, page: int) -> np.ndarray:
        import pymupdf
        from PIL import Image

        with pymupdf.open(self.pdf_dir / pdf) as doc:
            page = min(page, len(doc) - 1)
            pix = doc[page].get_pixmap(dpi=self.dpi)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        return np.asarray(img.resize((self.size, self.size), Image.BILINEAR), dtype=np.uint8)

    def __call__(self, example) -> torch.Tensor:
        """(3, S, S) float tensor in [-1, 1] (mean = std = 0.5, as LayoutLMv3's image processor without OCR)."""
        if self.mode == "blank":
            return torch.zeros(3, self.size, self.size)
        key = (getattr(example, "doc_guid", "") or example.guid, self.page_of(example))
        if key not in self._cache:
            self._cache[key] = self._render(*key)
        return torch.from_numpy(self._cache[key]).permute(2, 0, 1).float().div(127.5).sub(1.0)
