"""
Reconstruct 3D points from gs.ply and reproject to 2D.
This is for validating pose.json (translation/rotation/scale).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple, Optional

import h5py
import numpy as np
import trimesh
from PIL import Image
import matplotlib.pyplot as plt


def load_points_any(mesh_path: Path) -> np.ndarray:
    loaded = trimesh.load(mesh_path, process=False)
    if isinstance(loaded, trimesh.Scene):
        if len(loaded.geometry) == 0:
            raise ValueError(f"No geometry in scene: {mesh_path}")
        loaded = loaded.to_geometry()

    if isinstance(loaded, trimesh.Trimesh):
        return np.asarray(loaded.vertices)
    if isinstance(loaded, trimesh.PointCloud):
        return np.asarray(loaded.vertices)

    raise TypeError(f"Unsupported mesh type: {type(loaded)}")


def quaternion_to_matrix(quat: np.ndarray, order: str) -> np.ndarray:
    if order == "wxyz":
        w, x, y, z = quat
    elif order == "xyzw":
        x, y, z, w = quat
    else:
        raise ValueError(f"Unknown quaternion order: {order}")

    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def apply_pose(
    points: np.ndarray,
    pose: dict,
    quat_order: str,
    flip_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    translation = np.array(pose["translation"], dtype=np.float64)
    rotation_quat = np.array(pose["rotation"], dtype=np.float64)
    scale = np.array(pose["scale"], dtype=np.float64)

    if flip_matrix is None:
        flip_matrix = np.eye(3)

    points_scaled = points * scale
    R = quaternion_to_matrix(rotation_quat, quat_order)
    points_rotated = points_scaled @ (flip_matrix @ R).T
    points_transformed = points_rotated + translation
    return points_transformed


def load_intrinsics(h5_path: Path, key: str) -> Tuple[float, float, float, float]:
    with h5py.File(h5_path, "r") as f:
        if key not in f:
            available = list(f.keys())
            raise KeyError(f"{key} not found in {h5_path}. keys={available}")
        K = np.array(f[key], dtype=np.float64)
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    return fx, fy, cx, cy


def infer_k_key_from_image(image_path: Path, calib_path: Path) -> str:
    """
    Infer intrinsic key like 'NP1_rgb_K' from image filename.
    Example: NP1_0.jpg -> NP1_rgb_K
    """
    stem = image_path.stem
    prefix = stem.split("_")[0]
    candidate = f"{prefix}_rgb_K"
    with h5py.File(calib_path, "r") as f:
        if candidate in f:
            return candidate
        # Fallback: pick any *_rgb_K that matches prefix partially
        for key in f.keys():
            if key.endswith("_rgb_K") and prefix in key:
                return key
    raise KeyError(f"Could not infer K key from image {image_path.name}")


def project_points(
    points_camera: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    image_size: Tuple[int, int],
    camera_looks_negative_z: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = image_size

    x = points_camera[:, 0]
    y = points_camera[:, 1]
    z = points_camera[:, 2]
    if camera_looks_negative_z:
        z = -z

    valid = z > 1e-6
    x = x[valid]
    y = y[valid]
    z = z[valid]

    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    uv = np.stack([u, v], axis=1)

    in_bounds = (
        (uv[:, 0] >= 0) & (uv[:, 0] < width) &
        (uv[:, 1] >= 0) & (uv[:, 1] < height)
    )
    return uv[in_bounds], z[in_bounds]


def summarize_projection(label: str, points: np.ndarray, uv: np.ndarray, image_size: Tuple[int, int]) -> None:
    height, width = image_size
    if points.size == 0:
        print(f"[{label}] no points")
        return
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    print(f"[{label}] pts xyz min={mins} max={maxs}")
    print(f"[{label}] uv count={uv.shape[0]} (image {width}x{height})")


def plot_overlay(
    image_path: Path,
    uv: np.ndarray,
    depth: np.ndarray,
    save_path: Optional[Path] = None,
) -> None:
    img = Image.open(image_path).convert("RGB")
    plt.figure(figsize=(7, 7))
    plt.imshow(img)
    if uv.shape[0] > 0:
        plt.scatter(uv[:, 0], uv[:, 1], s=1, c=depth, cmap="turbo", alpha=0.6)
    plt.axis("off")
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_candidate_grid(
    image_path: Path,
    candidates: list,
    save_path: Path,
    image_size: Tuple[int, int],
    max_cols: int = 8,
) -> None:
    img = Image.open(image_path).convert("RGB")
    n = len(candidates)
    if n == 0:
        return
    cols = min(max_cols, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])

    for idx, cand in enumerate(candidates):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]
        ax.imshow(img)
        uv = cand["uv"]
        if uv.shape[0] > 0:
            ax.scatter(uv[:, 0], uv[:, 1], s=1, c=cand["depth"], cmap="turbo", alpha=0.6)
        ax.set_title(f"{cand['label']}\nIoU={cand['iou']:.3f}")
        ax.axis("off")

    # Hide any remaining axes
    for idx in range(n, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r, c].axis("off")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_mask(mask_path: Path, image_size: Tuple[int, int]) -> np.ndarray:
    mask_img = Image.open(mask_path).convert("L")
    if mask_img.size != (image_size[1], image_size[0]):
        mask_img = mask_img.resize((image_size[1], image_size[0]), Image.NEAREST)
    mask = np.array(mask_img) > 0
    return mask


def rasterize_points(
    uv: np.ndarray, image_size: Tuple[int, int], radius: int = 1
) -> np.ndarray:
    height, width = image_size
    mask = np.zeros((height, width), dtype=bool)
    if uv.shape[0] == 0:
        return mask

    u = np.round(uv[:, 0]).astype(int)
    v = np.round(uv[:, 1]).astype(int)
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[valid]
    v = v[valid]
    if u.size == 0:
        return mask

    if radius <= 0:
        mask[v, u] = True
        return mask

    for du in range(-radius, radius + 1):
        for dv in range(-radius, radius + 1):
            uu = u + du
            vv = v + dv
            valid2 = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
            mask[vv[valid2], uu[valid2]] = True
    return mask


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Reproject gs.ply using pose.json and intrinsics.")
    parser.add_argument("--data-root", help="Root directory containing NP1_0.jpg and calibration.h5")
    parser.add_argument("--gs-ply", help="Path to gs.ply (or mesh/pcd)")
    parser.add_argument("--pose-json", help="Path to pose.json")
    parser.add_argument("--image", help="Input RGB image path")
    parser.add_argument("--calib-h5", help="Calibration H5 path")
    parser.add_argument("--K-key", help="Intrinsic dataset key in H5 (auto if omitted)")
    parser.add_argument("--quat-order", default="wxyz", choices=["wxyz", "xyzw"])
    parser.add_argument("--max-points", type=int, default=200000)
    parser.add_argument("--mask", help="Binary mask path for IoU evaluation")
    parser.add_argument("--search-transform", action="store_true", help="Search flip/quaternion/z to maximize IoU")
    parser.add_argument("--point-radius", type=int, default=1, help="Point raster radius in pixels")
    parser.add_argument("--debug-dir", default=None, help="Directory to save debug overlays")
    z_group = parser.add_mutually_exclusive_group()
    z_group.add_argument("--camera-looks-negative-z", action="store_true")
    z_group.add_argument("--camera-looks-positive-z", action="store_true")
    parser.add_argument("--no-auto-flip-z", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.data_root:
        root = Path(args.data_root)
        image_path = Path(args.image) if args.image else root / "NP1_0.jpg"
        calib_path = Path(args.calib_h5) if args.calib_h5 else root / "calibration.h5"
        gs_path = Path(args.gs_ply) if args.gs_ply else root / "red_bowl" / "0_gs.ply"
        pose_path = Path(args.pose_json) if args.pose_json else root / "red_bowl" / "0_pose.json"
    else:
        if not all([args.gs_ply, args.pose_json, args.image, args.calib_h5]):
            raise ValueError("Provide --data-root or all of --gs-ply --pose-json --image --calib-h5.")
        gs_path = Path(args.gs_ply)
        pose_path = Path(args.pose_json)
        image_path = Path(args.image)
        calib_path = Path(args.calib_h5)

    points = load_points_any(gs_path)
    if points.shape[0] > args.max_points:
        idx = np.random.choice(points.shape[0], args.max_points, replace=False)
        points = points[idx]

    with open(pose_path, "r") as f:
        pose = json.load(f)

    img = Image.open(image_path)
    image_size = (img.height, img.width)

    k_key = args.K_key
    if not k_key:
        k_key = infer_k_key_from_image(image_path, calib_path)
    fx, fy, cx, cy = load_intrinsics(calib_path, k_key)
    print(f"Using intrinsics key: {k_key}")
    print(f"K = [[{fx:.3f}, 0, {cx:.3f}], [0, {fy:.3f}, {cy:.3f}], [0, 0, 1]]")

    looks_negative = False
    if args.camera_looks_positive_z:
        looks_negative = False
    elif args.camera_looks_negative_z:
        looks_negative = True
    auto_flip = not args.no_auto_flip_z

    debug_dir = Path(args.debug_dir) if args.debug_dir else None

    if args.search_transform:
        if not args.mask:
            raise ValueError("--search-transform requires --mask")
        mask_gt = load_mask(Path(args.mask), image_size)
        print(f"Mask gt: {mask_gt.sum()} pixels")

        flip_candidates = {
            "I": np.eye(3),
            "Fx": np.diag([-1.0, 1.0, 1.0]),
            "Fy": np.diag([1.0, -1.0, 1.0]),
            "Fz": np.diag([1.0, 1.0, -1.0]),
            "Fxy": np.diag([-1.0, -1.0, 1.0]),
            "Fxz": np.diag([-1.0, 1.0, -1.0]),
            "Fyz": np.diag([1.0, -1.0, -1.0]),
            "Fxyz": np.diag([-1.0, -1.0, -1.0]),
        }
        quat_orders = ["wxyz", "xyzw"]
        z_dirs = [False, True]

        best = {"iou": -1.0}
        candidates = []
        for quat_order in quat_orders:
            for flip_name, flip_mat in flip_candidates.items():
                points_camera = apply_pose(points, pose, quat_order, flip_mat)
                for z_neg in z_dirs:
                    uv, depth = project_points(points_camera, fx, fy, cx, cy, image_size, z_neg)
                    pred_mask = rasterize_points(uv, image_size, radius=args.point_radius)
                    iou = compute_iou(pred_mask, mask_gt)
                    label = f"{quat_order}|{flip_name}|zneg={int(z_neg)}"
                    candidates.append(
                        {"label": label, "uv": uv, "depth": depth, "iou": iou}
                    )
                    if iou > best["iou"]:
                        best = {
                            "iou": iou,
                            "quat_order": quat_order,
                            "flip": flip_name,
                            "camera_looks_negative_z": z_neg,
                            "uv": uv,
                            "points_camera": points_camera,
                            "depth": depth,
                        }

        print("Best transform search result:")
        print(f"  IoU: {best['iou']:.4f}")
        print(f"  quat_order: {best['quat_order']}")
        print(f"  flip: {best['flip']}")
        print(f"  camera_looks_negative_z: {best['camera_looks_negative_z']}")

        summarize_projection("best", best["points_camera"], best["uv"], image_size)
        if debug_dir is not None:
            overlay_path = debug_dir / "best_projection.png"
            plot_overlay(image_path, best["uv"], best["depth"], save_path=overlay_path)
            print(f"Saved best overlay to: {overlay_path}")
            grid_path = debug_dir / "all_candidates.png"
            plot_candidate_grid(image_path, candidates, grid_path, image_size)
            print(f"Saved candidate grid to: {grid_path}")
    else:
        points_camera = apply_pose(points, pose, args.quat_order)
        uv, depth = project_points(points_camera, fx, fy, cx, cy, image_size, looks_negative)
        if uv.shape[0] == 0 and auto_flip:
            uv, depth = project_points(points_camera, fx, fy, cx, cy, image_size, not looks_negative)
        plot_overlay(image_path, uv, depth, save_path=(debug_dir / "projection.png") if debug_dir else None)


if __name__ == "__main__":
    main()
