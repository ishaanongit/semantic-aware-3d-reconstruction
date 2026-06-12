# Pipeline Testing Guide

Run from repo root with `conda activate 3dvision`.

## Quick commands

```bash
# Dev run (1k points, 10 frames) — all steps + validation + PLY viewer
python scripts/run_dev_pipeline.py

# Skip interactive PLY windows (headless / SSH)
python scripts/run_dev_pipeline.py --no-viz

# View PLYs manually after a run
python src/viz/ply_viewer.py

# Single steps
python scripts/run_dev_pipeline.py 1 2

# Validate existing outputs
python scripts/validate_pipeline.py 1 2 3

# Full production (100k points, 2000 frames, ablation)
python scripts/run_production_pipeline.py
```

## Per-step reference

| Step | Script / `main.py` | Key outputs |
|------|-------------------|-------------|
| 1 | `src/data/mesh.py` | `points_*.npy`, `gt_labels_*.npy`, `viz/gt_chair_table.ply` |
| 2 | `src/data/projection.py` | `visibility.npz` |
| 3 | `src/features/dino_backbone.py` | `features_vits16/{frame}.npy` |
| 4 | `src/aggregation/variance.py` | `variance/dispersion.npy`, `variance/heatmap.ply` |
| 5 | `main.py` only | `points_*_idx.npy`, subsampled points |
| 6 | `src/features/dinotxt_features.py` | `features_dinotxt/{frame}.npy` |
| 7 | `src/aggregation/slerp_agg.py` | `aggregated/{method}.npy` |
| 8 | `src/features/dinotxt_text.py` | `text_embeddings/*.npy` |
| 9 | `classify.py` + `metrics.py` + `segmentation_viz.py` | `results/metrics_*.json`, `segmentation_errors_*.ply` |

## GT visualization (step 1)

```bash
python src/viz/gt_labels.py
```

Opens `outputs/room2/viz/gt_chair_table.ply` in MeshLab — green=chair, blue=table, gray=other.

## Dev settings (in `scripts/run_dev_pipeline.py`)

- `NUM_POINTS_INITIAL = 1_000`
- `NUM_POINTS_SUBSAMPLE = 500`
- `NUM_FRAMES = 10`

## Production estimates (GPU)

| Step | Runtime | Disk |
|------|---------|------|
| 1 | ~10 s | ~2 MB |
| 2 | ~30–90 min (with depth cache) | ~200 MB |
| 3 | ~30–90 min | ~10 GB |
| 4 | ~15–40 min | ~5 MB |
| 5 | ~5 s | ~2 MB |
| 6 | ~1–3 h | ~26 GB |
| 7 | ~15–45 min | ~200 MB |
| 8 | ~30 s | ~100 KB |
| 9 | ~5 min | ~5 MB |

Monitor production: `tail -f outputs/room2/production_run.log`
