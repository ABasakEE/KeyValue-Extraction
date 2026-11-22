"""QR codes as TikZ (vector, no external image): python poster/scripts/make_qr.py"""
import sys
from pathlib import Path

import segno

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S  # noqa: E402

CODES = {
    "qr_github": "https://github.com/ABasakEE/KeyValue-Extraction/tree/feat/bros-relative-layout",
    # "qr_demo": "<demo video link>",   # TODO: add when the demo video is uploaded
}

for name, url in CODES.items():
    q = segno.make(url, error="m")
    rows = list(q.matrix)
    n = len(rows)
    cells = [f"({x},{n - 1 - y}) rectangle ++(1,1)" for y, row in enumerate(rows) for x, v in enumerate(row) if v]
    tex = (f"% {url}\n\\begin{{tikzpicture}}[x=\\qrunit,y=\\qrunit]\n"
           f"\\fill[ink] {' '.join(cells)};\n\\end{{tikzpicture}}\n")
    (S.FIG / f"{name}.tex").write_text(tex, encoding="ascii")
    (S.FIG / f"{name}.pdf").unlink(missing_ok=True)
    print(name, n, "modules ->", url)
