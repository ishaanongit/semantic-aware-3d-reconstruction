# Semantic Flickering Research

Measure and reduce cross-view semantic inconsistency in DINOv3 patch features on the Replica dataset, then segment the 3D scene with DINOtxt-aligned features.

## Setup

```bash
git submodule update --init --recursive
conda activate 3dvision   # or: python3 -m venv .venv && source .venv/bin/activate
pip install -e .          # installs project + deps; makes config/src importable
pip install -e dinov3/
pip install torch torchvision  # install for your CUDA version if needed
```

Or install deps only: `pip install -r requirements.txt`

## Run

Edit [`config/settings.py`](config/settings.py), then:

```bash
python main.py
```

### Module smoke tests

Any of these work from the repo root:

```bash
python -m src.data.mesh
python src/data/mesh.py
```

### Dev mode (fast iteration)

```python
NUM_POINTS_INITIAL = 1_000
NUM_POINTS_SUBSAMPLE = 500
NUM_FRAMES = 10
STEPS = [1, 2, 3, 4]
FORCE_RECOMPUTE = True
```

## Pipeline steps

| Step | Description |
|------|-------------|
| 1 | Sample 100k mesh points + GT labels |
| 2 | Project to 2000 views, occlusion cull, save visibility |
| 3 | Extract ViT-S/16 patch features (variance analysis) |
| 4 | Compute cosine dispersion + heatmap |
| 5 | Variance-weighted 50k subsample |
| 6 | Extract DINOtxt-aligned patch features (ViT-L) |
| 7 | Aggregate multi-view features (mean / slerp / frechet) |
| 8 | Text prompt ensemble embeddings |
| 9 | Cosine segmentation, IoU metrics, error visualization |

## Module smoke tests

```bash
python -m src.data.mesh
python -m src.data.replica
python -m src.data.projection
python -m src.aggregation.slerp_agg
```

## Outputs

Artifacts saved under `outputs/room2/`. See [`AGENTS.md`]
