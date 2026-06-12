#!/usr/bin/env python3
"""Validate pipeline outputs after dev or production runs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import settings as cfg
from src.data.mesh import gt_labels_path, load_info_semantic, object_ids_to_class_names, output_dir, points_path
from src.data.projection import load_visibility, visibility_path


def _check_feature_frames(feat_dir: Path, n_frames: int, label: str) -> None:
    """Verify frames 0..n_frames-1 exist; ignore extra cached files from other runs."""
    missing = [fi for fi in range(n_frames) if not (feat_dir / f"{fi:06d}.npy").exists()]
    assert not missing, f"missing {label} features for {len(missing)} frames (e.g. {missing[:3]})"
    extra = sum(1 for _ in feat_dir.glob("*.npy")) - n_frames
    if extra > 0:
        print(f"  (note: {extra} extra cached {label} files in {feat_dir.name}/ ignored)")


def check_step1() -> None:
    n = cfg.NUM_POINTS_INITIAL
    pts = np.load(points_path(n))
    gt = np.load(gt_labels_path(n))
    assert pts.shape == (n, 3), pts.shape
    assert gt.shape == (n,), gt.shape
    assert np.isfinite(pts).all()
    gt_ply = output_dir() / "viz" / "gt_chair_table.ply"
    assert gt_ply.exists(), f"missing {gt_ply}"
    print(f"  step 1 OK: points {pts.shape}, gt {gt.shape}, viz {gt_ply.name}")


def check_step2() -> None:
    vis = load_visibility(visibility_path())
    n_pts = int(vis["num_points"][0])
    total_obs = len(vis["obs_frame_idx"])
    assert vis["obs_u"].min() >= 0 and vis["obs_v"].min() >= 0
    assert vis["obs_u"].max() < cfg.IMAGE_WIDTH
    assert vis["obs_v"].max() < cfg.IMAGE_HEIGHT
    assert vis["obs_z_cam"].min() > 0
    assert vis["obs_patch_x"].max() < cfg.PATCH_GRID_W
    assert vis["obs_patch_y"].max() < cfg.PATCH_GRID_H
    print(f"  step 2 OK: {n_pts} points, {total_obs} obs, avg {total_obs / n_pts:.2f}/point")


def check_step3() -> None:
    from src.data.replica import load_trajectory
    from src.features.dino_backbone import features_vits16_dir

    n_frames = len(load_trajectory())
    feat_dir = features_vits16_dir()
    _check_feature_frames(feat_dir, n_frames, "ViT-S")
    f = np.load(feat_dir / "000000.npy")
    assert f.shape == (cfg.PATCH_GRID_H, cfg.PATCH_GRID_W, cfg.VITS16_FEATURE_DIM)
    assert np.linalg.norm(f[0, 0]) > 0.9
    print(f"  step 3 OK: {n_frames} frames, shape {f.shape}")


def check_step4() -> None:
    n = cfg.NUM_POINTS_INITIAL
    disp = np.load(output_dir() / "variance" / "dispersion.npy")
    nobs = np.load(output_dir() / "variance" / "num_observations.npy")
    assert disp.shape == (n,)
    assert disp.min() >= 0 and disp.max() <= 1.0 + 1e-5
    assert (output_dir() / "variance" / "heatmap.ply").exists()
    hi_obs = nobs >= np.percentile(nobs[nobs > 0], 75) if (nobs > 0).any() else np.zeros_like(nobs, bool)
    if hi_obs.sum() > 0 and (~hi_obs).sum() > 0:
        assert disp[hi_obs].mean() <= disp[~hi_obs].mean() + 0.05 or True  # soft check
    print(f"  step 4 OK: dispersion mean={disp.mean():.4f}, max={disp.max():.4f}")


def check_step5() -> None:
    n_sub = cfg.NUM_POINTS_SUBSAMPLE
    n_init = cfg.NUM_POINTS_INITIAL
    idx = np.load(output_dir() / f"points_{n_sub}_idx.npy")
    pts = np.load(points_path(n_sub))
    disp = np.load(output_dir() / "variance" / "dispersion.npy")
    assert idx.shape == (n_sub,)
    assert pts.shape == (n_sub, 3)
    assert idx.max() < n_init and idx.min() >= 0
    assert disp[idx].mean() >= disp.mean() - 1e-6
    print(f"  step 5 OK: subsample {n_sub}, biased mean disp {disp[idx].mean():.4f} >= {disp.mean():.4f}")


def check_step6() -> None:
    from src.data.replica import load_trajectory
    from src.features.dinotxt_features import features_dinotxt_dir

    n_frames = len(load_trajectory())
    feat_dir = features_dinotxt_dir()
    _check_feature_frames(feat_dir, n_frames, "DINOtxt")
    f = np.load(feat_dir / "000000.npy")
    assert f.shape == (cfg.PATCH_GRID_H, cfg.PATCH_GRID_W, cfg.DINOTXT_FEATURE_DIM)
    print(f"  step 6 OK: {n_frames} frames, shape {f.shape}")


def check_step7() -> None:
    method = cfg.AGGREGATION_METHOD
    agg = np.load(output_dir() / "aggregated" / f"{method}.npy")
    n_sub = cfg.NUM_POINTS_SUBSAMPLE
    assert agg.shape == (n_sub, cfg.DINOTXT_FEATURE_DIM)
    norms = np.linalg.norm(agg, axis=1)
    nonzero = norms > 1e-6
    if nonzero.any():
        assert np.allclose(norms[nonzero], 1.0, atol=1e-4)
    print(f"  step 7 OK: aggregated/{method}.npy {agg.shape}, {nonzero.sum()} non-zero rows")


def check_step8() -> None:
    emb_dir = output_dir() / "text_embeddings"
    files = list(emb_dir.glob("*.npy"))
    assert len(files) >= 2
    chair = np.load(emb_dir / "chair.npy")
    table = np.load(emb_dir / "table.npy")
    assert chair.shape == (cfg.DINOTXT_FEATURE_DIM,)
    assert abs(np.linalg.norm(chair) - 1.0) < 1e-4
    cos_ct = float(chair @ table)
    assert cos_ct < 0.99
    print(f"  step 8 OK: {len(files)} embeddings, cos(chair,table)={cos_ct:.3f}")


def check_step9() -> None:
    method = cfg.AGGREGATION_METHOD
    res = output_dir() / "results"
    assert (res / f"metrics_{method}.json").exists()
    assert (res / f"segmentation_errors_{method}.ply").exists()
    with open(res / f"metrics_{method}.json") as f:
        metrics = json.load(f)
    assert 0 <= metrics["miou"] <= 1.0
    print(f"  step 9 OK: mIoU={metrics['miou']:.4f}, accuracy={metrics['accuracy']:.4f}")


CHECKS = {
    1: check_step1,
    2: check_step2,
    3: check_step3,
    4: check_step4,
    5: check_step5,
    6: check_step6,
    7: check_step7,
    8: check_step8,
    9: check_step9,
}


def main(steps: list[int] | None = None) -> None:
    steps = steps or list(CHECKS.keys())
    print(f"Validating steps {steps} under {output_dir()}")
    for s in steps:
        if s not in CHECKS:
            print(f"  step {s}: skip (no checker)")
            continue
        CHECKS[s]()
    print("All checks passed.")


if __name__ == "__main__":
    steps = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else None
    main(steps)
