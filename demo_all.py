"""Root entrypoint for the four-dataset demo (VRDU, FATURA, DocILE, XFUND). See src/demo_all.py."""

import os
import sys

for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    if _var not in os.environ:
        os.environ[_var] = "1"

from pathlib import Path

ROOT = Path(__file__).resolve().parent
for _p in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.demo_all import main

if __name__ == "__main__":
    main()
