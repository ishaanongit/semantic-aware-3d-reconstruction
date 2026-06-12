#!/usr/bin/env python3
"""Resume production pipeline from cached artifacts (skip completed frames)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("PYTHONUNBUFFERED", "1")

import config.settings as cfg

cfg.NUM_POINTS_INITIAL = 100_000
cfg.NUM_POINTS_SUBSAMPLE = 50_000
cfg.NUM_FRAMES = None
cfg.FORCE_RECOMPUTE = False  # use caches
cfg.DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
cfg.STEPS = list(range(3, 10))
cfg.AGGREGATION_METHOD = "mean"
cfg.ABLATION_METHODS = ["mean", "frechet"]

if __name__ == "__main__":
    import importlib.util

    import main

    main.main()

    spec = importlib.util.spec_from_file_location(
        "validate_pipeline", ROOT / "scripts" / "validate_pipeline.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(cfg.STEPS)
