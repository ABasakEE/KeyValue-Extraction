"""Root entrypoint to launch Track 2 RSA-GNN Gradio Demo."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from src.demo_gnn import main

if __name__ == "__main__":
    main()
