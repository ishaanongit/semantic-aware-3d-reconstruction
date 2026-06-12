#!/usr/bin/env python3
"""Run pipeline with dev settings for step-by-step testing."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config.settings as cfg

# Dev defaults from testing guide
cfg.NUM_POINTS_INITIAL = 1_000
cfg.NUM_POINTS_SUBSAMPLE = 500
cfg.NUM_FRAMES = 10
cfg.FORCE_RECOMPUTE = True
cfg.DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

if __name__ == "__main__":
    no_viz = "--no-viz" in sys.argv
    step_args = [int(x) for x in sys.argv[1:] if x.isdigit()]
    cfg.STEPS = step_args if step_args else list(range(1, 10))

    import main

    main.main()

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "validate_pipeline", ROOT / "scripts" / "validate_pipeline.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(cfg.STEPS)

    if not no_viz:
        from src.viz.ply_viewer import show_dev_ply_outputs

        show_dev_ply_outputs(cfg.STEPS)
