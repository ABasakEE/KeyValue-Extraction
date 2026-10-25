"""Root entrypoint to launch Track 1 LiLT Joint Key-Value Extraction Gradio Demo."""

import os
import sys
for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    if _var not in os.environ:
        os.environ[_var] = "1"

from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from src.demo import main

if __name__ == "__main__":
    main()
