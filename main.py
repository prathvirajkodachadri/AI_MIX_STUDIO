"""Direct Windows/Python launcher for AI Mix Studio.

Run from the extracted repository folder with:
    python main.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai_mix_studio.app import main


if __name__ == "__main__":
    main()
