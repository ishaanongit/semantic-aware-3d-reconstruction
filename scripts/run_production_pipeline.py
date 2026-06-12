#!/usr/bin/env python3
"""Run full production pipeline (100k points, 2000 frames, ablation)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config.settings as cfg

# Production settings
cfg.NUM_POINTS_INITIAL = 100_000
cfg.NUM_POINTS_SUBSAMPLE = 50_000
cfg.NUM_FRAMES = None  # all 2000 frames
cfg.FORCE_RECOMPUTE = True
cfg.DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
cfg.STEPS = list(range(1, 10))
cfg.AGGREGATION_METHOD = "mean"
cfg.ABLATION_METHODS = ["mean", "frechet"]

if __name__ == "__main__":
    import os

    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    import main

    main.main()

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "validate_pipeline", ROOT / "scripts" / "validate_pipeline.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(cfg.STEPS)
