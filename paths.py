"""Shared paths for the project.

BASE is the folder this file lives in. Everything the app produces
(dataset_<cat>.csv, model_<cat>.pth/.json, runs/, results/,
torch_cache/, app_error.log) is written inside it.

The MVTec source folder is resolved by get_mvtec_root(), in this order:
1) the MVTEC_ROOT environment variable,
2) the "mvtec_root" key in BASE/config.json,
3) BASE/"data"/"mvtec_anomaly_detection".

It is re-read on every call, so a change made in the GUI takes effect
immediately - including in child processes, which read config.json
themselves.
"""

import json
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent
CONFIG = BASE / "config.json"


def get_mvtec_root():
    """Return the current MVTec folder (see the module docstring)."""
    env = os.environ.get("MVTEC_ROOT")
    if env:
        return Path(env)
    try:
        root = json.loads(CONFIG.read_text(encoding="utf-8")).get(
            "mvtec_root")
        if root:
            return Path(root)
    except (OSError, ValueError):
        pass
    return BASE / "data" / "mvtec_anomaly_detection"


def set_mvtec_root(path):
    """Persist the MVTec folder choice into BASE/config.json."""
    CONFIG.write_text(
        json.dumps({"mvtec_root": str(path)}, indent=2),
        encoding="utf-8",
    )
