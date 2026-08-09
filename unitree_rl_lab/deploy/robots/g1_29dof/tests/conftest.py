from __future__ import annotations

import sys
from pathlib import Path

ROBOT_DIR = Path(__file__).resolve().parents[1]
if str(ROBOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROBOT_DIR))
