"""Interactive viewer for pipeline PLY point clouds (Open3D)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_bspec = importlib.util.spec_from_file_location(
    "project_bootstrap",
    Path(__file__).resolve().parents[1] / "bootstrap.py",
)
_bmod = importlib.util.module_from_spec(_bspec)
_bspec.loader.exec_module(_bmod)

import numpy as np

from config import settings as cfg
from src.aggregation.variance import variance_dir
from src.data.mesh import output_dir
from src.segmentation.classify import results_dir
from src.viz.segmentation_viz import COLOR_FN, COLOR_FP, COLOR_TP, COLOR_UNKNOWN


def collect_ply_paths(steps: list[int] | None = None) -> list[tuple[str, Path]]:
    """Return (label, path) pairs for PLY files produced by the given pipeline steps."""
    steps = steps or list(range(1, 10))
    out: list[tuple[str, Path]] = []
    root = output_dir()

    if 1 in steps:
        for name in ("gt_chair_table.ply", "gt_multiclass.ply"):
            path = root / "viz" / name
            if path.exists():
                out.append((f"GT labels ({name})", path))

    if 4 in steps:
        path = variance_dir() / "heatmap.ply"
        if path.exists():
            out.append(("Variance heatmap", path))

    if 9 in steps:
        for method in cfg.ABLATION_METHODS:
            path = results_dir() / f"segmentation_errors_{method}.ply"
            if path.exists():
                out.append((f"Segmentation errors ({method})", path))
        # Fallback if ablation not run
        if not any(m in label for label, _ in out for m in cfg.ABLATION_METHODS):
            path = results_dir() / f"segmentation_errors_{cfg.AGGREGATION_METHOD}.ply"
            if path.exists():
                out.append((f"Segmentation errors ({cfg.AGGREGATION_METHOD})", path))

    return out


def errors_ply_path(method: str | None = None) -> Path:
    """Resolve segmentation error PLY from pipeline results (step 9)."""
    method = method or cfg.AGGREGATION_METHOD
    root = results_dir()
    candidates = (
        root / "errors.ply",
        root / f"errors_{method}.ply",
        root / f"segmentation_errors_{method}.ply",
    )
    for path in candidates:
        if path.exists():
            return path
    return candidates[-1]


def _colors_to_uint8(colors: np.ndarray) -> np.ndarray:
    """Open3D stores colors as float [0, 1]; convert to uint8 for bucket matching."""
    if colors.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    if colors.dtype == np.uint8:
        return colors
    scaled = np.asarray(colors, dtype=np.float64)
    if scaled.max() <= 1.0 + 1e-6:
        scaled *= 255.0
    return np.clip(np.round(scaled), 0, 255).astype(np.uint8)


def summarize_error_ply(colors_u8: np.ndarray) -> dict[str, int]:
    """Count TP / FP / FN / unknown buckets from vertex colors."""
    buckets = {
        "true_positive": COLOR_TP,
        "false_positive": COLOR_FP,
        "false_negative": COLOR_FN,
        "unknown": COLOR_UNKNOWN,
    }
    counts: dict[str, int] = {name: 0 for name in buckets}
    other = 0
    for row in colors_u8:
        matched = False
        for name, ref in buckets.items():
            if tuple(row) == ref:
                counts[name] += 1
                matched = True
                break
        if not matched:
            other += 1
    counts["other"] = other
    counts["total"] = len(colors_u8)
    return counts


def print_error_legend() -> None:
    print("  Color legend:")
    print(f"    green  {COLOR_TP}  true positive  (correct class)")
    print(f"    red    {COLOR_FP}  false positive (wrong class assigned)")
    print(f"    yellow {COLOR_FN}  false negative (GT class missed → predicted unknown)")
    print(f"    gray   {COLOR_UNKNOWN}  unknown / no prediction")


def print_error_summary(counts: dict[str, int], metrics_path: Path | None = None) -> None:
    total = counts["total"]
    if total == 0:
        print("  (empty point cloud)")
        return

    tp = counts["true_positive"]
    fp = counts["false_positive"]
    fn = counts["false_negative"]
    unk = counts["unknown"]
    other = counts["other"]
    correct_pct = 100.0 * tp / total

    print(f"  Points: {total:,}")
    print(f"    TP (green):  {tp:6,}  ({100.0 * tp / total:5.1f}%)")
    print(f"    FP (red):    {fp:6,}  ({100.0 * fp / total:5.1f}%)")
    print(f"    FN (yellow): {fn:6,}  ({100.0 * fn / total:5.1f}%)")
    print(f"    unknown:     {unk:6,}  ({100.0 * unk / total:5.1f}%)")
    if other:
        print(f"    other:       {other:6,}  ({100.0 * other / total:5.1f}%)")
    print(f"  Approx accuracy (TP / total): {correct_pct:.2f}%")

    if metrics_path and metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        print(f"  Saved metrics: mIoU={metrics.get('miou', 0):.4f}, accuracy={metrics.get('accuracy', 0):.4f}")


def show_ply(path: Path, title: str | None = None) -> bool:
    """
    Open an interactive Open3D window for one PLY file.

    Returns True if the viewer opened, False on headless / missing display.
    """
    try:
        import open3d as o3d
    except ImportError:
        print("  open3d not installed — pip install open3d")
        return False

    path = Path(path)
    if not path.exists():
        print(f"  skip (missing): {path}")
        return False

    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) == 0:
        print(f"  skip (empty): {path}")
        return False

    window_name = title or path.name
    print(f"  Opening: {window_name} ({len(pcd.points)} points) — close window for next")
    try:
        o3d.visualization.draw_geometries(
            [pcd],
            window_name=window_name,
            width=1280,
            height=720,
        )
    except Exception as exc:
        print(f"  Could not open viewer ({exc}). Open manually in MeshLab: {path}")
        return False
    return True


def show_segmentation_errors_ply(
    path: Path | None = None,
    method: str | None = None,
) -> bool:
    """
    Open the step-9 segmentation error PLY with a printed color legend and counts.

    Default path: ``outputs/room2/results/segmentation_errors_{method}.ply``
    (also accepts ``errors.ply`` or ``errors_{method}.ply`` in the same folder).
    """
    method = method or cfg.AGGREGATION_METHOD
    path = Path(path) if path is not None else errors_ply_path(method)
    metrics_path = results_dir() / f"metrics_{method}.json"

    print(f"\nSegmentation errors — method={method}")
    print(f"  PLY: {path}")
    if not path.exists():
        print("  File not found. Run pipeline step 9 first (segmentation_viz).")
        return False

    try:
        import open3d as o3d
    except ImportError:
        print("  open3d not installed — pip install open3d")
        return False

    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) == 0:
        print(f"  skip (empty): {path}")
        return False

    colors_u8 = _colors_to_uint8(np.asarray(pcd.colors))
    counts = summarize_error_ply(colors_u8)
    print_error_legend()
    print_error_summary(counts, metrics_path)

    window_name = f"Segmentation errors ({method}) — close window to exit"
    print(f"\n  Opening viewer ({len(pcd.points):,} points)...")
    try:
        o3d.visualization.draw_geometries(
            [pcd],
            window_name=window_name,
            width=1280,
            height=720,
        )
    except Exception as exc:
        print(f"  Could not open viewer ({exc}). Open manually in MeshLab: {path}")
        return False
    return True


def show_ply_list(ply_items: list[tuple[str, Path]]) -> None:
    """Show each PLY in sequence; user closes each window to advance."""
    if not ply_items:
        print("No PLY files to visualize.")
        return

    print(f"\nPLY viewer — {len(ply_items)} file(s). Close each window to continue.")
    for label, path in ply_items:
        show_ply(path, title=label)


def show_dev_ply_outputs(steps: list[int] | None = None) -> None:
    """Show PLY outputs relevant to a dev pipeline run."""
    items = collect_ply_paths(steps)
    show_ply_list(items)


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        show_dev_ply_outputs()
        return

    if argv[0] in ("errors", "--errors", "-e"):
        method = argv[1] if len(argv) > 1 else None
        show_segmentation_errors_ply(method=method)
        return

    if len(argv) == 1 and Path(argv[0]).suffix == ".ply":
        path = Path(argv[0])
        if "error" in path.stem.lower():
            show_segmentation_errors_ply(path=path)
            return

    items = [(Path(p).name, Path(p)) for p in argv]
    show_ply_list(items)


if __name__ == "__main__":
    main()
