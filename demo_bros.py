"""Root entrypoint for the BROS + Track 1 field-extraction demo (VRDU Registration fields). See src/demo_bros.py."""

import os
import sys

for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    if _var not in os.environ:
        os.environ[_var] = "1"

from pathlib import Path

ROOT = Path(__file__).resolve().parent
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.demo_bros import main

if __name__ == "__main__":
    main()
